"""Native MMD/XECBS requests with pinned physical and completion checks."""
from __future__ import annotations

import contextlib
import copy
import io
import os
import signal
import sys
import time
import traceback
from dataclasses import replace

import canonical_n28_runtime as c
import numpy as np
import torch

CAP = 45.0
NATIVE_CAP = 40.0
WATCHDOG = 60.0


class SMDNativeRobotCollisionProxy:
    """Use released SMD sampled pair tolerance in the pinned CBS search."""

    def __init__(self, robot):
        self.robot = robot

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def check_rr_collisions(self, positions):
        relative = positions.unsqueeze(-2) - positions.unsqueeze(-3)
        threshold_sq = (2 * self.robot.radius)**2 - 1e-3
        collided = (relative.square().sum(-1) < threshold_sq)
        collided &= ~torch.eye(collided.shape[-1], device=collided.device, dtype=torch.bool)
        midpoint = (positions.unsqueeze(-2) + positions.unsqueeze(-3)) / 2
        midpoint = midpoint.masked_fill(~collided[..., None], float("nan"))
        return collided, midpoint


class RequestTimeout(BaseException):
    pass


def classify_request_failure(exc):
    """Keep exhausted planner search distinct from programming failures."""
    from obstacle_aware_repair import StaticRepairNoSolution

    if isinstance(exc, RequestTimeout):
        return "EXTERNAL_TIMEOUT"
    if isinstance(exc, StaticRepairNoSolution):
        return "FAIL_STATIC_CHILD"
    return "EXCEPTION"


@contextlib.contextmanager
def watchdog():
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)[0]

    def stop(signum, frame):
        raise RequestTimeout(f"{WATCHDOG}-second external request watchdog")

    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(
        signal.ITIMER_REAL,
        min(WATCHDOG, previous_timer) if previous_timer else WATCHDOG,
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_timer - (time.monotonic() - started)),
            )


class Native:
    def __init__(self, engine, backend="XECBS"):
        if backend not in ("CBS", "ECBS", "XCBS", "XECBS"):
            raise ValueError(f"unsupported CBS-family backend: {backend}")
        self.engine = engine
        self.backend = backend
        self.population = engine.population
        begun = c.sync()
        root = c.MMD_ROOT
        sys.path[:0] = [str(root), str(root / "scripts/inference")]
        os.chdir(root / "scripts/inference")
        from inference_multi_agent import run_multi_agent_trial
        from mmd.common.experiments import (
            MultiAgentPlanningSingleTrialConfig,
            TrialSuccessStatus,
        )
        from mmd.config.mmd_params import MMDParams
        from mmd.planners.multi_agent.cbs import CBS, SearchState
        from mmd.planners.single_agent.mpd_ensemble import MPDEnsemble
        from torch_robotics.torch_utils.seed import fix_random_seed

        self.CBS, self.SearchState, self.Status, self.MPDEnsemble, self.fix_seed = (
            CBS,
            SearchState,
            TrialSuccessStatus,
            MPDEnsemble,
            fix_random_seed,
        )
        self.params = MMDParams
        self.repair = c.load(
            "n100_native_repair", c.ROOT / "integrations/smd_runtime/diffuser/utils/mmd_xecbs_root_repair.py"
        )
        MMDParams.results_dir = str(c.OUT / "native_scratch")
        (c.OUT / "native_scratch").mkdir(exist_ok=True)
        scene = engine.split["train"][0]
        config = MultiAgentPlanningSingleTrialConfig()
        config.time_str = str(c.OUT / "native_setup")
        # Factory-only initialization; no smaller-N planner rollout or quality gate.
        config.num_agents, config.runtime_limit = 1, 40
        config.multi_agent_planner_class, config.single_agent_planner_class = (
            backend,
            "MPDEnsemble",
        )
        config.instance_name = getattr(engine, "native_instance_name", "EnvEmpty2DRobotPlanarDiskBoundary")
        config.start_state_pos_l = [torch.tensor(scene["starts"][0], device="cuda")]
        config.goal_state_pos_l = [torch.tensor(scene["goals"][0], device="cuda")]
        config.global_model_ids = [[getattr(engine, "native_model_id", "EnvEmpty2D-RobotPlanarDisk")]]
        config.agent_skeleton_l = [[[0, 0]]]
        config.render_animation = False

        class FactoryReady(Exception):
            pass

        original = CBS.plan

        def capture(planner, *args, **kwargs):
            self.template = planner
            raise FactoryReady()

        CBS.plan = capture
        try:
            with (c.OUT / "logs/native_setup.log").open(
                "a"
            ) as log, contextlib.redirect_stdout(log):
                try:
                    run_multi_agent_trial(config)
                except FactoryReady:
                    pass
        finally:
            CBS.plan = original
            os.chdir(c.ROOT)
        prototype = self.template.low_level_planner_l[0]
        # Frozen prior/dataset/task objects are shared; mutable guidance and
        # per-agent planner state are independently deep-copied.
        shared = [
            *prototype.models.values(),
            *prototype.datasets,
            prototype.task,
            prototype.robot,
        ]
        self.bank = [
            copy.deepcopy(prototype, {id(x): x for x in shared})
            for _ in range(self.population)
        ]
        assert all(p.robot.dt == c.DT for p in self.bank)
        assert len({id(p.guides) for p in self.bank}) == self.population
        assert all(p.models[0] is prototype.models[0] for p in self.bank)
        self.setup_seconds = c.sync() - begun
        c.SETUP_SECONDS += self.setup_seconds
        c.write(
            c.OUT / "native_setup.json",
            dict(
                seconds=self.setup_seconds,
                factory=f"official run_multi_agent_trial captured before planning; one immutable prior shared across {self.population} isolated planner states",
                n_samples=MMDParams.n_samples,
                trajectory_duration=MMDParams.trajectory_duration,
                radius=MMDParams.robot_planar_disk_radius,
                source_sha256={
                    str(p): c.sha(p)
                    for p in [
                        root / "mmd/planners/multi_agent/cbs.py",
                        root / "mmd/planners/single_agent/mpd_ensemble.py",
                        c.ROOT / "integrations/smd_runtime/diffuser/utils/mmd_xecbs_root_repair.py",
                    ]
                },
            ),
        )

    def to_native_state(self, path):
        if not getattr(self.engine, "native_velocity_bridge", False):
            return path
        result = path.clone() if torch.is_tensor(path) else path.copy()
        result[..., 2:] /= c.DT
        return result

    def to_physical_state(self, path):
        if not getattr(self.engine, "native_velocity_bridge", False):
            return path
        result = path.clone() if torch.is_tensor(path) else path.copy()
        result[..., 2:] *= c.DT
        return result

    def configure(self, scene):
        if getattr(self, "world", None) is not None and self.world.obstacles["kind"] == "circles":
            from obstacle_aware_repair import install_native_circles

            # Bank members share the immutable model and task object. Update
            # the scene-dependent native obstacle field once per distinct task.
            tasks = {id(p.task): p.task for p in self.bank}
            tasks[id(self.template.reference_task)] = self.template.reference_task
            for task in tasks.values():
                install_native_circles(task, self.world, tensor_args=self.bank[0].tensor_args)
        starts, goals = [
            torch.tensor(scene[k], device="cuda", dtype=torch.float32)
            for k in ("starts", "goals")
        ]
        for i, p in enumerate(self.bank):
            s = p.task.inverse_transform_q(0, starts[i])
            g = p.task.inverse_transform_q(0, goals[i])
            hard = p.datasets[0].get_single_pt_hard_conditions(s, 0, True)
            hard.update(p.datasets[0].get_single_pt_hard_conditions(g, -1, True))
            p.hard_conds = {0: hard}
            p.start_state_pos = starts[i].clone()
            p.goal_state_pos = goals[i].clone()
        planner = self.CBS(
            self.bank,
            list(starts.unbind()),
            list(goals.unbind()),
            start_time_l=[0] * self.population,
            is_xcbs=self.backend in ("XCBS", "XECBS"),
            is_ecbs=self.backend in ("ECBS", "XECBS"),
            conflict_type_to_constraint_types=self.template.conflict_type_to_constraint_types,
            reference_robot=self.template.reference_robot,
            reference_task=self.template.reference_task,
        )
        if getattr(self, "world", None) is not None and self.world.contract == "smd_native":
            planner.reference_robot = SMDNativeRobotCollisionProxy(planner.reference_robot)
            # Pinned MMD prematurely abandons both CT branches when the first
            # low-level child fails. Correct this only for SMD hard repair.
            import types

            planner.expand = types.MethodType(self.repair.expand_all_branches, planner)
        return planner, starts, goals

    def request(self, scene, method, directory, profile=False):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (method + ".json")
        if target.exists():
            row = c.read(target)
            if not (directory / (method + ".contract.json")).exists():
                from canonical_result_export import export

                export(scene, row, self.engine, directory)
            return row
        if (
            getattr(c, "PROCESS_DEADLINE", float("inf")) - time.monotonic()
            < WATCHDOG + 5
        ):
            raise c.BudgetExpired(
                "Reserve insufficient for another fully recorded request"
            )
        if method == "random":
            reference = c.read(
                c.OUT
                / "evaluation/test"
                / f"scene_{scene['configuration_id']:03d}"
                / "sparse.json"
            )
            self.engine.control_degree_schedule = {
                step["timestep"]: step["degree"]
                for step in reference["accounting"]["steps"]
            }
        self.engine.noise(
            scene
        )  # resident request input/noise preparation outside timing
        self.fix_seed(scene["noise_seed"])
        self.params.seed = scene["noise_seed"]
        calls = []
        original = self.MPDEnsemble.__call__
        bank_agent_ids = {id(planner): agent for agent, planner in enumerate(self.bank)}
        world = None
        if scene.get("obstacles", {}).get("kind") in ("circles", "boxes"):
            from obstacle_aware_repair import PlanarDiskWorld

            contract = ("smd_native" if str(scene.get("family", "")).startswith("SMD-")
                        else "common_strict")
            world = PlanarDiskWorld.from_scene(scene, contract=contract)
        self.world = world

        def counted(p, *args, **kwargs):
            agent = bank_agent_ids.get(id(p))
            record = dict(agent_id=agent,
                          constraints=len(kwargs.get("constraints_l") or []),
                          warm_start=kwargs.get("experience") is not None)
            calls.append(record)
            mpd_begin = time.monotonic()
            result = original(p, *args, **kwargs)
            record["mpd_seconds"] = time.monotonic() - mpd_begin
            if world is not None and method != "official" and agent is not None:
                from obstacle_aware_repair import filter_native_output

                before = len(result.trajs_final_free_idxs)
                inspect_all = len(kwargs.get("constraints_l") or []) == 0
                filter_native_output(result, world, agent, diagnose_all=inspect_all,
                                     constraints=kwargs.get("constraints_l"))
                source_count = (len(result.trajs_final) if world.contract == "smd_native"
                                else before)
                record["static_rejected_candidates"] = source_count - len(result.trajs_final_free_idxs)
                record["native_pre_smoothing_free_candidates"] = before
                record["static_accepted_candidates"] = len(result.trajs_final_free_idxs)
                if record["constraints"]:
                    record["mpd_child_success"] = bool(len(result.trajs_final_free_idxs))
                record["static_rejection_diagnostics"] = result.static_rejection_diagnostics
                record["static_endpoint_snap_m_max"] = result.static_endpoint_snap_m_max
                if inspect_all:
                    record["static_physical_safe_all_candidates"] = result.static_physical_safe_all_candidates
                if (not inspect_all and not len(result.trajs_final_free_idxs) and
                        world.obstacles["kind"] in ("circles", "boxes")):
                    from native_rrt_static_fallback import rrt_static_path

                    fallback_begin = time.monotonic()
                    candidate, attempts = rrt_static_path(
                        world, agent, constraints=kwargs.get("constraints_l"),
                        max_time_s=(2. if world.obstacles["kind"] == "boxes" else 1.),
                        seed_offsets=(37, 0, 73, 149),
                    )
                    record["rrt_fallback_seconds"] = time.monotonic() - fallback_begin
                    record["rrt_fallback_attempts"] = attempts
                    record["rrt_fallback_accepted"] = candidate is not None
                    if candidate is not None:
                        candidate = candidate.to(device=result.trajs_final.device,
                                                 dtype=result.trajs_final.dtype)
                        # Keep the native batch shape for xCBS experience reuse.
                        result.trajs_final = candidate[None].expand_as(result.trajs_final).clone()
                        result.trajs_final_free_idxs = torch.zeros(
                            1, device=candidate.device, dtype=torch.long)
                        result.trajs_final_free = result.trajs_final[:1]
                        result.traj_final_free_best = result.trajs_final[0]
                        result.idx_best_traj = 0
                        result.success_free_trajs = True
                        result.fraction_free_trajs = 1. / len(result.trajs_final)
            return result

        self.MPDEnsemble.__call__ = counted
        captured = {}
        root = final = canonical = partial = None
        row = dict(
            contract_version=getattr(self.engine, "contract_version", "n100-empty-v1"),
            backend=self.backend,
            method=method,
            configuration_id=scene["configuration_id"],
            split=scene["split"],
            parent_geometry_id=scene["parent_geometry_id"],
            task_sha256=scene["input_sha256"],
            native_status="UNKNOWN",
            failure=None,
            timeout=False,
            ct_expansions=0,
            root_seconds=0.0,
            repair_seconds=0.0,
            post_check_seconds=0.0,
            setup_seconds=self.setup_seconds + self.engine.setup_seconds,
            external_budget_seconds=CAP,
        )
        torch.cuda.reset_peak_memory_stats()
        begin = c.sync()
        root_end = None
        try:
            with watchdog(), (directory / (method + ".log")).open(
                "w"
            ) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                planner, starts, goals = self.configure(scene)
                if method == "official":
                    original_conflicts = planner.get_conflicts

                    def observe(state):
                        conflicts = original_conflicts(state)
                        if (
                            "root" not in captured
                            and len(state.path_bl) == self.population
                        ):
                            captured["root"] = self.to_physical_state((
                                torch.stack(
                                    [
                                        state.path_bl[i][
                                            state.ix_best_path_in_batch_l[i]
                                        ].squeeze(0)
                                        for i in range(self.population)
                                    ],
                                    1,
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            ))
                            captured["root_end"] = c.sync()
                        return conflicts

                    planner.get_conflicts = observe
                    paths, expansions, status, conflicts = planner.plan(
                        runtime_limit=max(0.01, NATIVE_CAP - (c.sync() - begin))
                    )
                    root = captured.get("root")
                    root_end = captured.get("root_end")
                else:
                    physical, _, account = self.engine.rollout(
                        scene, method, profile=profile
                    )
                    row["accounting"] = account
                    row["proposal_quality"] = c.q.metrics(
                        physical[0].detach().cpu().numpy(),
                        scene["starts"], scene["goals"], self.engine.spec,
                    )
                    physical, paths, audit = self.repair.prepare_manifest_joint_paths(
                        physical, starts, goals
                    )
                    paths = [self.to_native_state(path) for path in paths]
                    if world is not None:
                        from mmd.common.experiences import PathBatchExperience
                        from obstacle_aware_repair import repair_root_with_native_low_level

                        strict_world = replace(world, contract="common_strict")
                        proposal_array = physical[0].detach().cpu().numpy()
                        row["proposal_common_strict"] = strict_world.joint_audit(proposal_array)
                        if world.contract == "smd_native":
                            row["proposal_smd_native"] = world.joint_audit(proposal_array)
                        static_fallback = None
                        if world.obstacles["kind"] in ("circles", "boxes"):
                            from native_rrt_static_fallback import rrt_static_path

                            row["native_rrt_static_fallback"] = []

                            def static_fallback(agent):
                                candidate, attempts = rrt_static_path(world, agent)
                                row["native_rrt_static_fallback"].append(
                                    dict(agent=agent, attempts=attempts, accepted=candidate is not None))
                                if candidate is None:
                                    return None
                                return candidate.to(device=starts.device, dtype=starts.dtype)

                        static_begin = c.sync()
                        try:
                            paths, replaced = repair_root_with_native_low_level(
                                planner, paths, world, PathBatchExperience,
                                static_fallback=static_fallback,
                            )
                        finally:
                            # Account for failed attempts as repair work too.
                            row["root_static_repair_seconds"] = c.sync() - static_begin
                        row["root_static_replanned_agents"] = replaced
                        physical = self.to_physical_state(torch.stack(paths, 1))[None]
                        root_array = physical[0].detach().cpu().numpy()
                        row["root_common_strict"] = strict_world.joint_audit(root_array)
                        if world.contract == "smd_native":
                            row["root_smd_native"] = world.joint_audit(root_array)
                    root = self.to_physical_state(torch.stack(paths, 1)).cpu().numpy()
                    state = self.repair.injected_root(
                        self.SearchState, planner, paths, starts, goals
                    )
                    row["root_native_conflicts"] = len(state.conflict_l)
                    root_end = c.sync()
                    (
                        paths,
                        expansions,
                        status,
                        conflicts,
                        _,
                    ) = self.repair.repair_from_injected_root(
                        planner,
                        state,
                        max(0.01, NATIVE_CAP - (root_end - begin)),
                        self.Status,
                        accept_state=(lambda paths: world.joint_audit(
                            torch.stack(paths, 1).detach().cpu().numpy()
                        )["success"]) if world is not None else None,
                    )
                row.update(
                    native_status=status.name,
                    ct_expansions=expansions,
                    final_native_conflicts=conflicts,
                    timeout="RUNTIME_LIMIT" in status.name,
                )
                if paths:
                    array = self.to_physical_state(torch.stack(paths, 1)).detach().cpu().numpy()
                    if len(paths) == self.population:
                        final = array
                    else:
                        partial = array
        except (RequestTimeout, Exception) as exc:
            row.update(
                native_status=classify_request_failure(exc),
                failure=traceback.format_exc(),
                timeout=isinstance(exc, RequestTimeout),
            )
            if root is None:
                root = captured.get("root")
                root_end = captured.get("root_end")
        finally:
            self.MPDEnsemble.__call__ = original
        native_end = c.sync()
        row.update(
            root_seconds=(root_end or native_end) - begin,
            repair_seconds=native_end - (root_end or native_end),
            low_level_calls=len(calls),
            low_level_call_records=calls,
            low_level_replanned_agents=sorted(
                {call["agent_id"] for call in calls if call["agent_id"] is not None}
            ),
            mpd_child_replans=sum(bool(call["constraints"]) for call in calls),
            mpd_child_successes=sum(bool(call.get("mpd_child_success")) for call in calls),
            mpd_child_seconds=sum(call.get("mpd_seconds", 0.) for call in calls
                                  if call["constraints"]),
            rrt_child_fallbacks=sum("rrt_fallback_seconds" in call for call in calls),
            rrt_child_fallback_successes=sum(bool(call.get("rrt_fallback_accepted")) for call in calls),
            rrt_child_fallback_seconds=sum(call.get("rrt_fallback_seconds", 0.) for call in calls),
        )
        for label, array in [("root_quality", root), ("raw_final_quality", final)]:
            row[label] = (
                c.q.metrics(array, scene["starts"], scene["goals"], self.engine.spec)
                if array is not None
                else None
            )
        if final is not None:
            canonical = final.copy()
            canonical[0, :, :2] = scene["starts"]
            canonical[-1, :, :2] = scene["goals"]
            canonical[0, :, 2:] = 0
            canonical[-1, :, 2:] = 0
            if world is not None and world.max_step is not None:
                from smd_temporal_metrics import reconstructed_grouped_state

                canonical = reconstructed_grouped_state(canonical).astype(final.dtype)
            row["final_quality"] = c.q.metrics(
                canonical, scene["starts"], scene["goals"], self.engine.spec
            )
            if world is not None:
                strict_world = replace(world, contract="common_strict")
                row["final_common_strict"] = strict_world.joint_audit(canonical)
                if world.contract == "smd_native":
                    row["final_smd_native"] = world.joint_audit(canonical)
        else:
            row["final_quality"] = None
        if world is not None and world.contract == "smd_native":
            strict = row.get("final_common_strict") or {}
            physical_valid = bool(row.get("final_smd_native", {}).get("success"))
            row["final_task_integrity"] = bool(
                strict.get("start_error_m", float("inf")) <= 1e-5
                and strict.get("goal_error_m", float("inf")) <= 1e-5
                and strict.get("workspace_margin_m", -float("inf")) >= -1e-9)
        else:
            physical_valid = bool(
                row["final_quality"] and row["final_quality"]["complete_valid"]
                and (world is None or row["final_common_strict"]["success"])
            )
        reasons = []
        if row["native_status"] != "SUCCESS":
            reasons.append("native:" + row["native_status"])
        if not physical_valid:
            if row["final_quality"]:
                for key in (
                    "collision_free_0100",
                    "native_clearance_0105",
                    "boundary_valid",
                    "endpoint_valid",
                ):
                    if not row["final_quality"].get(key, False):
                        reasons.append("common:" + key + "_failed")
            else:
                reasons.append("common:missing_full_team_output")
        if canonical is not None:
            arrival = c.completion.uniform_joint_completion(
                canonical,
                scene["goals"],
                c.DT,
                final_valid=physical_valid and row["native_status"] == "SUCCESS",
                planning_seconds=0,
                tolerance=0.05,
                failure_reasons=reasons,
            )
        else:
            arrival = dict(
                scheduled_execution_makespan_seconds=None,
                request_to_completion_seconds=None,
                goal_arrival_valid=False,
                failure_reasons=reasons + ["common:missing_full_team_output"],
            )
        end = c.sync()
        row.update(
            post_check_seconds=end - native_end,
            complete_total_seconds=end - begin,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        row["complete_success"] = bool(
            row["native_status"] == "SUCCESS"
            and physical_valid
            and row["complete_total_seconds"] <= CAP
            and arrival["goal_arrival_valid"]
        )
        if world is not None and world.contract == "smd_native":
            row["smd_native_success"] = bool(
                row["native_status"] == "SUCCESS"
                and row.get("final_smd_native", {}).get("success")
                and row["complete_total_seconds"] <= CAP)
            row["common_strict_success"] = bool(
                row["native_status"] == "SUCCESS"
                and row.get("final_common_strict", {}).get("success")
                and row["complete_total_seconds"] <= CAP)
        if row["complete_total_seconds"] > CAP:
            row["timeout"] = True
            reasons.append("common:external_budget_exceeded")
            arrival.update(
                scheduled_execution_makespan_seconds=None,
                request_to_completion_seconds=None,
                goal_arrival_valid=False,
                per_agent_goal_arrival_seconds=[None] * self.population,
                mean_agent_goal_arrival_seconds=None,
            )
        arrival["planning_wall_seconds"] = row["complete_total_seconds"]
        if row["complete_success"]:
            arrival["request_to_completion_seconds"] = (
                row["complete_total_seconds"]
                + arrival["scheduled_execution_makespan_seconds"]
            )
        row["completion"] = arrival
        if method == "random":
            row[
                "support_control"
            ] = "Replay exact learned per-agent degree at each reverse step, randomize identities with independent fixed seed; retain U scoring work but discard its support. Quality diagnostic only."
        row["failure_reasons"] = reasons
        # Exporting is outside resident request; checking and completion counted once.
        for label, array in [
            ("root", root),
            ("raw_final", final),
            ("final", canonical),
            ("partial", partial),
        ]:
            if array is not None:
                np.savez_compressed(
                    directory / (method + "_" + label + ".npz"),
                    physical=array,
                    positions=array[..., :2],
                    auxiliary=array[..., 2:],
                    timestamps=np.arange(len(array)) * c.DT,
                    dt=c.DT,
                    start_offsets=np.zeros(array.shape[1]),
                )
        c.write(target, row)
        from canonical_result_export import export

        export(scene, row, self.engine, directory)
        print(
            scene["split"],
            scene["configuration_id"],
            method,
            "sampled_success",
            row["complete_success"],
            "root_pairs",
            row["root_quality"]["native_pair_times"] if row["root_quality"] else None,
            "seconds",
            round(row["complete_total_seconds"], 3),
            flush=True,
        )
        return row
