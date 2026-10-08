"""Native low-level static-world guard for MMD CBS-family repair.

The low-level MPD keeps its pinned environment, diffusion model, guidance,
constraints, and experience semantics. This module only supplies exact scene
circles to the MMD EmptyNoWait task and rejects any candidate that fails the
portable physical world contract after the planner's own smoothing step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from common_strict_safety import evaluate, obstacle_segment_clearance, smd_native_success


class StaticRepairNoSolution(RuntimeError):
    """Expected outcome when all native static repair candidates are invalid."""


@dataclass(frozen=True)
class PlanarDiskWorld:
    obstacles: dict
    starts: np.ndarray
    goals: np.ndarray
    workspace: np.ndarray
    robot_radius: float = .05
    horizon: int = 64
    duration: float = 5.0
    max_step: float | None = None
    contract: str = "common_strict"

    @classmethod
    def from_scene(cls, scene, *, contract="common_strict"):
        if contract not in ("common_strict", "smd_native"):
            raise ValueError(f"Unsupported world contract: {contract}")
        if contract == "smd_native" and scene.get("obstacles", {}).get("kind") != "circles":
            raise ValueError("SMD-native contract requires exact circles")
        return cls(
            obstacles=scene["obstacles"],
            starts=np.asarray(scene["starts"], dtype=np.float64),
            goals=np.asarray(scene["goals"], dtype=np.float64),
            workspace=np.asarray(scene.get("workspace", ((-1., -1.), (1., 1.))), dtype=np.float64),
            robot_radius=float(scene.get("radii", [.05])[0]),
            max_step=(.05 if str(scene.get("family", "")).startswith("SMD-")
                      else scene.get("comparison_max_step_m")),
            contract=contract,
        )

    @property
    def dt(self):
        return self.duration / self.horizon

    def path_valid(self, path, agent):
        path = np.asarray(path, dtype=np.float64)
        if path.shape != (self.horizon, 4) or not np.isfinite(path).all():
            return False
        positions = path[:, :2]
        if (np.linalg.norm(positions[0] - self.starts[agent]) > 1e-5 or
                np.linalg.norm(positions[-1] - self.goals[agent]) > 1e-5):
            return False
        boundary_radius = self.robot_radius if self.contract == "common_strict" else 0.
        if (np.any(positions < self.workspace[0] + boundary_radius - 1e-9) or
                np.any(positions > self.workspace[1] - boundary_radius + 1e-9)):
            return False
        # The released SMD success evaluator comments out its velocity check.
        # Keep the 0.05 m support-step bound in COMMON_STRICT only; SMD_NATIVE
        # still preserves task endpoints and the workspace during planning.
        if (self.contract == "common_strict" and self.max_step is not None and
                np.linalg.norm(np.diff(positions, axis=0), axis=-1).max() > self.max_step + 1e-8):
            return False
        if self.contract == "smd_native":
            for item in self.obstacles["items"]:
                center = np.asarray(item["center"], dtype=np.float64)
                threshold_sq = (self.robot_radius + float(item["radius"]))**2 - 1e-3
                if np.any(np.sum((positions - center)**2, axis=-1) < threshold_sq):
                    return False
            return True
        return all(obstacle_segment_clearance(p, q, self.obstacles, self.robot_radius) >= -1e-9
                   for p, q in zip(positions[:-1], positions[1:]))

    def joint_audit(self, paths):
        paths = np.asarray(paths, dtype=np.float64)
        result = evaluate(paths, self.starts, self.goals, self.obstacles,
                          robot_radius=self.robot_radius, workspace=self.workspace)
        step = float(np.linalg.norm(np.diff(paths[..., :2], axis=0), axis=-1).max())
        result["max_step_m"] = step
        result["speed_valid"] = self.max_step is None or step <= self.max_step + 1e-8
        result["footprint_workspace_margin_m"] = result["workspace_margin_m"] - self.robot_radius
        if self.contract == "smd_native":
            result["common_strict_success"] = bool(result["success"] and result["speed_valid"]
                                                   and result["footprint_workspace_margin_m"] >= -1e-9)
            result["smd_native_success"] = smd_native_success(paths, self.obstacles,
                                                               robot_radius=self.robot_radius)
            result["success"] = result["smd_native_success"]
        else:
            result["success"] = (result["success"] and result["speed_valid"]
                                 and result["footprint_workspace_margin_m"] >= -1e-9)
        return result


def install_native_obstacles(task_ensemble, world, *, tensor_args):
    """Install exact circles or axis-aligned boxes in the pinned MMD task.

    The task's existing collision distance field calls ``env.get_df_obj_list``
    at evaluation time, so its native guidance and collision filter see the
    new scene's exact geometry. No circle-to-box approximation occurs.
    """
    from torch_robotics.environments.primitives import MultiBoxField, MultiSphereField, ObjectField

    obstacles = world.obstacles
    if obstacles["kind"] == "circles":
        items = obstacles["items"]
        centers = np.asarray([item["center"] for item in items], dtype=np.float64).reshape(-1, 2)
        radii = np.asarray([item["radius"] for item in items], dtype=np.float64)
        native_field = ObjectField([MultiSphereField(centers, radii, tensor_args=tensor_args)],
                                   "scene-circles")
    elif obstacles["kind"] == "boxes":
        if obstacles.get("full_sizes") is not True:
            raise ValueError("Box sizes must use the pinned full-width convention")
        native_field = ObjectField([MultiBoxField(
            np.asarray(obstacles["centers"], dtype=np.float64),
            np.asarray(obstacles["sizes"], dtype=np.float64), tensor_args=tensor_args)],
            "scene-boxes")
    else:
        raise ValueError(f"Unsupported static obstacle type: {obstacles['kind']}")
    task = task_ensemble.tasks[0]
    env = task.env
    env.obj_extra_list = [native_field]
    env.obj_all_list = set((*(env.obj_fixed_list or []), native_field))
    # EnvEnsemble snapshots its source objects at construction. Keep the
    # reference task's object list synchronized for endpoint checks.
    if task_ensemble.env is not env:
        aggregate = task_ensemble.env
        aggregate.obj_fixed_list = [*(env.obj_fixed_list or []), native_field]
        aggregate.obj_all_list = set(aggregate.obj_fixed_list)


def install_native_circles(task_ensemble, world, *, tensor_args):
    """Compatibility name for the pinned circle-world integration tests."""
    if world.obstacles["kind"] != "circles":
        raise ValueError("Expected circles")
    return install_native_obstacles(task_ensemble, world, tensor_args=tensor_args)


def _satisfies_hard_constraints(path, constraints):
    """Check the pinned CBS point constraints after MPD's smoothing step."""
    for constraint in constraints or ():
        if constraint.get_is_soft():
            continue
        for center, time_range, radius in zip(
                constraint.get_q_l(), constraint.get_t_range_l(), constraint.get_radius_l()):
            first = max(0, int(time_range[0]))
            last = min(len(path) - 1, int(time_range[1]))
            if first > last:
                continue
            point = np.asarray(center.detach().cpu() if isinstance(center, torch.Tensor) else center)
            distance = np.linalg.norm(path[first:last + 1, :2] - point[:2], axis=-1)
            if np.any(distance < float(radius) - 1e-9):
                return False
    return True


def filter_native_output(output, world, agent, diagnose_all=False, constraints=None):
    """Intersect MMD's native free set with exact post-smoothing validity.

    CBS only reads ``trajs_final_free_idxs``, ``trajs_final``, and
    ``idx_best_traj`` when creating a root or CT child. Keep all three coherent
    so an obstacle-invalid replacement can never enter a CT node.
    """
    batch = output.trajs_final
    if batch is None:
        return output
    native_free = [int(index) for index in output.trajs_final_free_idxs]
    # MMD's free set uses different obstacle tolerance and is computed before
    # smoothing. For SMD, inspect every final sample against the released
    # sampled rule and every hard CBS constraint after smoothing.
    source_indices = (list(range(len(batch))) if world.contract == "smd_native"
                      else native_free)
    # MMD smooths after applying its normalized hard conditions, which can
    # move physical start/goal positions by centimetres. Restore the exact
    # scene endpoints only for bounded smoothing drift, then check the chosen
    # contract again. This changes no interior point.
    batch = batch.clone()
    starts = torch.as_tensor(world.starts[agent], device=batch.device, dtype=batch.dtype)
    goals = torch.as_tensor(world.goals[agent], device=batch.device, dtype=batch.dtype)
    endpoint_snap = {}
    for index in source_indices:
        start_error = float(torch.linalg.norm(batch[index, 0, :2] - starts))
        goal_error = float(torch.linalg.norm(batch[index, -1, :2] - goals))
        if max(start_error, goal_error) > .05:
            continue
        batch[index, 0, :2] = starts
        batch[index, -1, :2] = goals
        batch[index, 0, 2:] = 0
        batch[index, -1, 2:] = 0
        endpoint_snap[index] = max(start_error, goal_error)
    output.trajs_final = batch
    values = batch.detach().cpu().numpy()
    accepted = [index for index in source_indices
                if world.path_valid(values[index], agent)
                and _satisfies_hard_constraints(values[index], constraints)]
    output.static_endpoint_snap_m_max = max(endpoint_snap.values(), default=0.)
    if diagnose_all:
        exact_free = 0
        for candidate in values:
            trial = candidate.copy()
            start_error = np.linalg.norm(trial[0, :2] - world.starts[agent])
            goal_error = np.linalg.norm(trial[-1, :2] - world.goals[agent])
            if max(start_error, goal_error) <= .05:
                trial[0, :2] = world.starts[agent]
                trial[-1, :2] = world.goals[agent]
            exact_free += int(world.path_valid(trial, agent))
        output.static_physical_safe_all_candidates = exact_free
    rejected = []
    for index in source_indices:
        if index in accepted:
            continue
        positions = values[index, :, :2].astype(np.float64)
        rejected.append(dict(
            candidate=index,
            start_error_m=float(np.linalg.norm(positions[0] - world.starts[agent])),
            goal_error_m=float(np.linalg.norm(positions[-1] - world.goals[agent])),
            workspace_margin_m=float(min((positions - world.workspace[0]).min(),
                                         (world.workspace[1] - positions).min())),
            max_step_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).max()),
            swept_static_clearance_m=float(min(
                obstacle_segment_clearance(a, b, world.obstacles, world.robot_radius)
                for a, b in zip(positions[:-1], positions[1:]))),
        ))
    output.static_rejection_diagnostics = rejected
    if accepted:
        # XCBS reuses the complete stored batch as experience. Replace every
        # invalid entry with a valid path, while retaining only the original
        # accepted indices for CT selection.
        safe_batch = batch.clone()
        invalid = [index for index in range(len(values)) if index not in accepted]
        if invalid:
            safe_batch[invalid] = batch[accepted[0]]
        output.trajs_final = safe_batch
        batch = safe_batch
    output.trajs_final_free_idxs = torch.as_tensor(accepted, device=batch.device, dtype=torch.long)
    output.trajs_final_free = batch[output.trajs_final_free_idxs]
    output.success_free_trajs = bool(accepted)
    output.fraction_free_trajs = len(accepted) / len(values)
    if accepted:
        previous = int(output.idx_best_traj) if output.idx_best_traj is not None else None
        output.idx_best_traj = previous if previous in accepted else accepted[0]
        output.traj_final_free_best = batch[output.idx_best_traj]
    else:
        output.idx_best_traj = None
        output.traj_final_free_best = None
    return output


def repair_root_with_native_low_level(planner, paths, world, make_experience,
                                      max_cold_attempts=1, static_fallback=None):
    """Use the pinned low-level planner for every obstacle-invalid root agent.

    First attempt the existing XCBS-style experience interface, then native
    cold inference. Both results pass ``filter_native_output`` through the
    caller's low-level wrapper. Failure is explicit; no invalid path is used.
    """
    repaired = list(paths)
    replaced = []
    for agent, path in enumerate(paths):
        if world.path_valid(path.detach().cpu().numpy(), agent):
            continue
        low_level = planner.low_level_planner_l[agent]
        seed = path[None].expand(low_level.num_samples, -1, -1).clone()
        selected = None
        for experience in [make_experience(seed), *([None] * max_cold_attempts)]:
            result = low_level(planner.start_state_pos_l[agent],
                               planner.goal_state_pos_l[agent],
                               constraints_l=[], experience=experience)
            if len(result.trajs_final_free_idxs):
                selected = result.trajs_final[int(result.idx_best_traj)]
                break
        if selected is None and static_fallback is not None:
            selected = static_fallback(agent)
        if selected is None or not world.path_valid(selected.detach().cpu().numpy(), agent):
            raise StaticRepairNoSolution(
                f"native low-level static repair failed for agent {agent} "
                f"after one warm and {max_cold_attempts} cold attempts")
        repaired[agent] = selected
        replaced.append(agent)
    return repaired, replaced
