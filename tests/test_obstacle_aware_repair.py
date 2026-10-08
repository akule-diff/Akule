"""Exercise the pinned MMD CT child path with an exact static-world guard."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
MMD = ROOT / "external/mmd"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(MMD),
                str(MMD / "deps/torch_robotics"),
                str(MMD / "deps/motion_planning_baselines"),
                str(MMD / "deps/experiment_launcher")]

from mmd.common.conflicts import PointConflict
from mmd.common.constraints import MultiPointConstraint
from mmd.planners.multi_agent.cbs import CBS, SearchState, CBSExperienceReuseStrategy
from mmd.planners.single_agent.common import PlannerOutput
from diffuser.utils.mmd_xecbs_root_repair import repair_from_injected_root
from obstacle_aware_repair import (PlanarDiskWorld, filter_native_output,
                                   install_native_circles, repair_root_with_native_low_level)
from mmd.common.experiences import PathBatchExperience
from mmd.common.experiments import TrialSuccessStatus
from canonical_mmd_repair import SMDNativeRobotCollisionProxy
from native_rrt_static_fallback import smd_rrt_static_path
from torch_robotics.environments import EnvEmptyNoWait2DExtraObjects
from torch_robotics.robots import RobotPlanarDisk
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.tasks.tasks_ensemble import PlanningTaskEnsemble


def state(points):
    points = np.asarray(points, dtype=np.float32)
    return torch.as_tensor(np.concatenate((points, np.zeros_like(points)), axis=-1))


class FakeRobot:
    def get_position(self, path):
        return path[..., :2]

    def check_rr_collisions(self, positions):
        delta = positions[:, :, None] - positions[:, None, :]
        collision = torch.linalg.norm(delta, dim=-1) < .1
        count = positions.shape[1]
        collision[:, torch.arange(count), torch.arange(count)] = False
        points = (positions[:, :, None] + positions[:, None, :]) / 2
        return collision, points


class FakeLowLevel:
    def __init__(self, candidates, world, agent):
        self.candidates = torch.stack(candidates)
        self.world = world
        self.agent = agent
        self.num_samples = len(candidates)
        self.n_support_points = candidates[0].shape[0]
        self.calls = 0

    def __call__(self, start, goal, constraints_l=None, experience=None):
        self.calls += 1
        output = PlannerOutput()
        output.trajs_final = self.candidates.clone()
        output.trajs_final_free_idxs = torch.arange(len(self.candidates))
        output.idx_best_traj = 0
        return filter_native_output(output, self.world, self.agent)


def world_for(kind):
    obstacles = ({"kind": "circles", "items": [{"center": [0, .5], "radius": .12}]}
                 if kind == "circles" else
                 {"kind": "boxes", "centers": [[0, .5]], "sizes": [[.3, .3]]})
    return PlanarDiskWorld(obstacles=obstacles,
                           starts=np.array([[-.6, 0.], [0., 0.]]),
                           goals=np.array([[.6, 0.], [0., 0.]]),
                           workspace=np.array([[-1., -1.], [1., 1.]]),
                           horizon=4)


def paths():
    crossing = state([[-.6, 0], [0, .5], [.2, .5], [.6, 0]])
    safe = state([[-.6, 0], [-.2, -.3], [.2, -.3], [.6, 0]])
    root = state([[-.6, 0], [0, 0], [0, 0], [.6, 0]])
    stationary = state([[0, 0], [0, 0], [0, 0], [0, 0]])
    moved = state([[0, 0], [0, .2], [0, .2], [0, 0]])
    return crossing, safe, root, stationary, moved


def fake_cbs(world):
    crossing, safe, root, stationary, moved = paths()
    planner = object.__new__(CBS)
    planner.low_level_planner_l = [FakeLowLevel([crossing, safe], world, 0),
                                   FakeLowLevel([moved], world, 1)]
    planner.num_agents = 2
    planner.start_state_pos_l = [torch.as_tensor(point, dtype=torch.float32) for point in world.starts]
    planner.goal_state_pos_l = [torch.as_tensor(point, dtype=torch.float32) for point in world.goals]
    planner.start_time_l = [0, 0]
    planner.reference_robot = FakeRobot()
    planner.tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    planner.conflict_type_to_constraint_types = {PointConflict: {MultiPointConstraint}}
    planner.is_xcbs = True
    planner.experience_reuse_strategy = CBSExperienceReuseStrategy.XCBS
    planner.is_ecbs = False
    planner.low_level_choose_path_from_batch_strategy = "least_collisions"
    planner.open_l = []
    return planner, root, stationary


def test_invalid_root_is_repaired_by_native_low_level_for_box_and_circle():
    for kind in ("boxes", "circles"):
        world = world_for(kind)
        planner, _, stationary = fake_cbs(world)
        crossing = paths()[0]
        assert not world.path_valid(crossing.numpy(), 0)
        repaired, changed = repair_root_with_native_low_level(
            planner, [crossing, stationary], world, PathBatchExperience)
        assert changed == [0]
        assert world.path_valid(repaired[0].numpy(), 0)
        assert planner.low_level_planner_l[0].calls == 1


def test_static_fallback_is_used_only_after_native_low_level_exhaustion():
    world = world_for("boxes")
    planner, _, stationary = fake_cbs(world)
    crossing, safe, *_ = paths()
    planner.low_level_planner_l[0] = FakeLowLevel([crossing], world, 0)
    called = []

    def fallback(agent):
        called.append(agent)
        return safe

    repaired, changed = repair_root_with_native_low_level(
        planner, [crossing, stationary], world, PathBatchExperience,
        max_cold_attempts=1, static_fallback=fallback)
    assert changed == [0] and called == [0]
    assert planner.low_level_planner_l[0].calls == 2
    assert world.path_valid(repaired[0].numpy(), 0)


@pytest.mark.parametrize("kind", ["boxes", "circles"])
def test_real_cbs_expand_rejects_obstacle_child_and_accepts_safe_replan(kind):
    world = world_for(kind)
    planner, root_path, stationary = fake_cbs(world)
    root = SearchState([0, 0], [root_path[None], stationary[None]], {})
    root.conflict_l = planner.get_conflicts(root)
    assert root.conflict_l
    planner.expand(root)  # Actual pinned MMD CBS.expand.
    assert planner.open_l
    for child in planner.open_l:
        for agent, index in enumerate(child.ix_best_path_in_batch_l):
            assert world.path_valid(child.path_bl[agent][index].numpy(), agent)
            for candidate in child.path_bl[agent]:
                assert world.path_valid(candidate.numpy(), agent)
    assert planner.low_level_planner_l[0].calls == 1
    assert planner.low_level_planner_l[0].candidates.shape[0] == 2


@pytest.mark.parametrize("kind", ["boxes", "circles"])
def test_complete_search_and_final_independent_validator(kind):
    world = world_for(kind)
    planner, root_path, stationary = fake_cbs(world)
    root = SearchState([0, 0], [root_path[None], stationary[None]], {})
    root.conflict_l = planner.get_conflicts(root)
    output, _, status, conflicts, _ = repair_from_injected_root(
        planner, root, 3., TrialSuccessStatus,
        accept_state=lambda selected: world.joint_audit(torch.stack(selected, 1).numpy())["success"])
    assert status == TrialSuccessStatus.SUCCESS
    assert conflicts == 0
    assert world.joint_audit(torch.stack(output, 1).numpy())["success"]


@pytest.mark.parametrize("kind", ["boxes", "circles"])
def test_final_guard_refuses_an_obstacle_invalid_conflict_free_root(kind):
    world = world_for(kind)
    planner, _, stationary = fake_cbs(world)
    crossing = paths()[0]
    root = SearchState([0, 0], [crossing[None], stationary[None]], {})
    root.conflict_l = []
    _, _, status, _, _ = repair_from_injected_root(
        planner, root, 3., TrialSuccessStatus,
        accept_state=lambda selected: world.joint_audit(torch.stack(selected, 1).numpy())["success"])
    assert status == TrialSuccessStatus.FAIL_NO_SOLUTION


def test_only_obstacle_invalid_ct_candidate_is_rejected():
    world = world_for("boxes")
    crossing = paths()[0]
    low_level = FakeLowLevel([crossing], world, 0)
    result = low_level(torch.tensor([-.6, 0.]), torch.tensor([.6, 0.]))
    assert len(result.trajs_final_free_idxs) == 0
    assert result.idx_best_traj is None


@pytest.mark.parametrize("kind", ["boxes", "circles"])
def test_native_smoothing_endpoint_drift_is_safely_snapped(kind):
    world = world_for(kind)
    safe = paths()[1].clone()
    safe[0, 0] += .01
    safe[-1, 0] -= .015
    output = PlannerOutput()
    output.trajs_final = safe[None]
    output.trajs_final_free_idxs = torch.tensor([0])
    output.idx_best_traj = 0
    filter_native_output(output, world, 0)
    assert len(output.trajs_final_free_idxs) == 1
    assert output.static_endpoint_snap_m_max == pytest.approx(.015, abs=1e-6)
    assert world.path_valid(output.trajs_final[0].numpy(), 0)
    torch.testing.assert_close(output.trajs_final[0, 0, :2],
                               torch.tensor(world.starts[0], dtype=torch.float32))
    torch.testing.assert_close(output.trajs_final[0, -1, :2],
                               torch.tensor(world.goals[0], dtype=torch.float32))


def test_large_endpoint_drift_is_rejected():
    world = world_for("circles")
    path = paths()[1].clone()
    path[0, 0] += .06
    output = PlannerOutput()
    output.trajs_final = path[None]
    output.trajs_final_free_idxs = torch.tensor([0])
    output.idx_best_traj = 0
    filter_native_output(output, world, 0)
    assert len(output.trajs_final_free_idxs) == 0


def test_exact_scene_circles_enter_native_task_field():
    world = world_for("circles")
    local_env = SimpleNamespace(obj_fixed_list=[], obj_extra_list=[], obj_all_list=set())
    combined_env = SimpleNamespace(obj_fixed_list=[], obj_all_list=set())
    task = SimpleNamespace(tasks={0: SimpleNamespace(env=local_env)}, env=combined_env)
    install_native_circles(task, world,
                           tensor_args={"device": torch.device("cpu"), "dtype": torch.float32})
    field = local_env.obj_extra_list[0].fields[0]
    np.testing.assert_allclose(field.centers.numpy(), [[0., .5]])
    np.testing.assert_allclose(field.radii.numpy(), [.12])
    assert len(combined_env.obj_fixed_list) == 1


def test_pinned_empty_nowait_task_sees_installed_scene_circle():
    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    env = EnvEmptyNoWait2DExtraObjects(precompute_sdf_obj_fixed=False,
                                        tensor_args=tensor_args)
    robot = RobotPlanarDisk(tensor_args=tensor_args)
    local_task = PlanningTask(env=env, robot=robot, tensor_args=tensor_args)
    ensemble = PlanningTaskEnsemble({0: local_task}, {0: torch.zeros(2)},
                                    tensor_args=tensor_args)
    center = torch.tensor([0., .5])
    assert not bool(local_task.compute_collision(center))
    install_native_circles(ensemble, world_for("circles"), tensor_args=tensor_args)
    assert bool(local_task.compute_collision(center))


def test_smd_per_step_speed_endpoint_and_workspace_contract():
    world = PlanarDiskWorld(obstacles={"kind": "circles", "items": []},
                           starts=np.array([[0., 0.]]), goals=np.array([[.06, 0.]]),
                           workspace=np.array([[-1., -1.], [1., 1.]]),
                           horizon=4, max_step=.05)
    valid = state([[0, 0], [.02, 0], [.04, 0], [.06, 0]])
    assert world.path_valid(valid.numpy(), 0)
    jump = state([[0, 0], [0, 0], [0, 0], [.06, 0]])
    assert not world.path_valid(jump.numpy(), 0)
    bad_endpoint = valid.clone()
    bad_endpoint[-1, 0] = .07
    assert not world.path_valid(bad_endpoint.numpy(), 0)
    bad_workspace = valid.clone()
    bad_workspace[1, 1] = 1.01
    assert not world.path_valid(bad_workspace.numpy(), 0)


def test_disk_footprint_workspace_is_checked_after_repair():
    world = PlanarDiskWorld(obstacles={"kind": "circles", "items": []},
                           starts=np.array([[.97, 0.]]), goals=np.array([[.97, 0.]]),
                           workspace=np.array([[-1., -1.], [1., 1.]]), horizon=4)
    path = state([[.97, 0.]] * 4)
    assert not world.path_valid(path.numpy(), 0)
    audit = world.joint_audit(path[:, None].numpy())
    assert audit["workspace_margin_m"] > 0
    assert audit["footprint_workspace_margin_m"] < 0
    assert not audit["success"]


def test_smd_native_sampled_circle_contract_stays_separate_from_strict():
    world = PlanarDiskWorld(obstacles={"kind": "circles", "items": [
                                {"center": [0., 0.], "radius": .1}]},
                           starts=np.array([[0., .148]]), goals=np.array([[0., .148]]),
                           workspace=np.array([[-1., -1.], [1., 1.]]),
                           horizon=4, contract="smd_native")
    path = state([[0., .148]] * 4)
    assert world.path_valid(path.numpy(), 0)
    native = world.joint_audit(path[:, None].numpy())
    assert native["success"] and native["smd_native_success"]
    assert not native["common_strict_success"]
    assert not PlanarDiskWorld(**{**world.__dict__, "contract": "common_strict"}).path_valid(path.numpy(), 0)


def test_smd_native_repair_respects_projected_step_bound():
    world = PlanarDiskWorld(obstacles={"kind": "circles", "items": []},
                           starts=np.array([[0., 0.]]), goals=np.array([[.1, 0.]]),
                           workspace=np.array([[-1., -1.], [1., 1.]]),
                           horizon=4, max_step=.05, contract="smd_native")
    jump = state([[0., 0.], [0., 0.], [0., 0.], [.1, 0.]])
    assert not world.path_valid(jump.numpy(), 0)
    # The released SMD success evaluator itself has no speed test; keep its
    # native outcome separately visible instead of silently redefining it.
    assert world.joint_audit(jump[:, None].numpy())["smd_native_success"]
    assert not world.joint_audit(jump[:, None].numpy())["speed_valid"]


def test_smd_native_root_can_use_candidate_rejected_by_mmd_static_filter():
    world = PlanarDiskWorld(obstacles={"kind": "circles", "items": [
                                {"center": [0., 0.], "radius": .1}]},
                           starts=np.array([[0., .148]]), goals=np.array([[0., .148]]),
                           workspace=np.array([[-1., -1.], [1., 1.]]),
                           horizon=4, contract="smd_native")
    path = state([[0., .148]] * 4)
    output = PlannerOutput()
    output.trajs_final = path[None]
    output.trajs_final_free_idxs = torch.empty(0, dtype=torch.long)
    output.idx_best_traj = None
    filter_native_output(output, world, 0, diagnose_all=True)
    assert len(output.trajs_final_free_idxs) == 1
    assert world.path_valid(output.trajs_final[0].numpy(), 0)


def test_smd_native_cbs_pair_threshold_matches_released_evaluator():
    proxy = SMDNativeRobotCollisionProxy(SimpleNamespace(radius=.05))
    safe = torch.tensor([[[0., 0.], [.097, 0.]]])
    collision, _ = proxy.check_rr_collisions(safe)
    assert not collision.any()
    unsafe = torch.tensor([[[0., 0.], [.094, 0.]]])
    collision, _ = proxy.check_rr_collisions(unsafe)
    assert bool(collision[0, 0, 1])


def test_smd_native_child_checks_hard_constraint_after_smoothing():
    world = PlanarDiskWorld(obstacles={"kind": "circles", "items": []},
                           starts=np.array([[0., 0.]]), goals=np.array([[0., 0.]]),
                           workspace=np.array([[-1., -1.], [1., 1.]]),
                           horizon=4, contract="smd_native")
    violating = state([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
    valid = state([[0., 0.], [.2, 0.], [.2, 0.], [0., 0.]])
    constraint = SimpleNamespace(
        get_is_soft=lambda: False,
        get_q_l=lambda: [torch.tensor([0., 0.])],
        get_t_range_l=lambda: [(1, 2)],
        get_radius_l=lambda: [.1],
    )
    output = PlannerOutput()
    output.trajs_final = torch.stack([violating, valid])
    output.trajs_final_free_idxs = torch.empty(0, dtype=torch.long)
    output.idx_best_traj = None
    filter_native_output(output, world, 0, constraints=[constraint])
    assert output.trajs_final_free_idxs.tolist() == [1]
    assert torch.allclose(output.trajs_final[0], valid)


def test_smd_rrt_child_fallback_obeys_static_and_time_constraint():
    world = PlanarDiskWorld(obstacles={"kind": "circles", "items": [
                                {"center": [0., .7], "radius": .1}]},
                           starts=np.array([[-.5, 0.]]), goals=np.array([[.5, 0.]]),
                           workspace=np.array([[-1., -1.], [1., 1.]]),
                           horizon=64, max_step=.05, contract="smd_native")
    constraint = SimpleNamespace(
        get_is_soft=lambda: False,
        get_q_l=lambda: [torch.tensor([0., 0.])],
        get_t_range_l=lambda: [(32, 32)],
        get_radius_l=lambda: [.15],
    )
    path, attempts = smd_rrt_static_path(
        world, 0, constraints=[constraint], max_time_s=.5, seed_offsets=(37,))
    assert path is not None, attempts
    assert world.path_valid(path.numpy(), 0)
    assert np.linalg.norm(path[32, :2].numpy()) >= .15
    assert attempts[0]["start_hold_steps"] > 0
