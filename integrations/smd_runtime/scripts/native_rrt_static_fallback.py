"""Pinned RRTConnect fallback for exact circle or box static worlds.

The released MMD task supplies exact scene circles. RRT supplies geometry;
every shortcut edge is continuously checked. SMD-native support acceptance
uses the released sampled rule, while the stricter step and swept audits are
reported separately.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common_strict_safety import obstacle_segment_clearance
from obstacle_aware_repair import _satisfies_hard_constraints, install_native_obstacles

ROOT = Path(__file__).resolve().parents[3]
MMD = ROOT / "external/mmd"


def shortcut_route(points, obstacles, margin=.005, robot_radius=.05):
    result = [points[0]]
    left = 0
    while left < len(points) - 1:
        for right in range(len(points) - 1, left, -1):
            if obstacle_segment_clearance(points[left], points[right], obstacles, robot_radius) >= margin:
                result.append(points[right])
                left = right
                break
        else:
            return None
    return np.asarray(result)


def time_parameterize(polyline, horizon=64, max_step=.05):
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=-1)
    if len(lengths) == 0 or np.any(lengths <= 0):
        return None
    counts = np.ceil(lengths / max_step - 1e-10).astype(int)
    if counts.sum() > horizon - 1:
        return None
    extra = horizon - 1 - counts.sum()
    weights = lengths / lengths.sum()
    added = np.floor(extra * weights).astype(int)
    counts += added
    for index in np.argsort(-(extra * weights - added))[:extra - added.sum()]:
        counts[index] += 1
    positions = [polyline[0]]
    for start, goal, count in zip(polyline[:-1], polyline[1:], counts):
        positions.extend(start + (goal - start) * (step / count) for step in range(1, count + 1))
    positions = np.asarray(positions, dtype=np.float32)
    displacement = np.concatenate((np.diff(positions, axis=0), np.zeros((1, 2), np.float32)), axis=0)
    displacement[0] = 0
    return np.concatenate((positions, displacement), axis=-1)


def resample_continuous_route(polyline, horizon=64):
    """Use all safe polyline vertices while distributing 64 supports by arclength.

    Every consecutive support pair lies on one checked straight polyline edge.
    No corner is skipped, so resampling cannot cut through an obstacle.
    """
    polyline = np.asarray(polyline, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=-1)
    if not len(lengths) or np.any(lengths <= 0) or len(lengths) > horizon - 1:
        return None
    extra = horizon - 1 - len(lengths)
    shares = extra * lengths / lengths.sum()
    added = np.floor(shares).astype(int)
    for index in np.argsort(-(shares - added))[:extra - int(added.sum())]:
        added[index] += 1
    counts = added + 1
    positions = [polyline[0]]
    for start, goal, count in zip(polyline[:-1], polyline[1:], counts):
        positions.extend(start + (goal - start) * (index / count)
                         for index in range(1, count + 1))
    positions = np.asarray(positions, dtype=np.float32)
    displacement = np.concatenate((np.diff(positions, axis=0),
                                   np.zeros((1, 2), np.float32)), axis=0)
    displacement[0] = 0
    return np.concatenate((positions, displacement), axis=-1)


def constraint_aware_time_parameterize(polyline, constraints, horizon=64, max_step=.05):
    """Find a 64-support schedule on a checked polyline with waits anywhere.

    This changes only timing. The caller chooses the step bound for its
    contract and independently checks the resulting path against obstacles,
    endpoints, and CBS constraints.
    """
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=-1)
    if not len(lengths) or np.any(lengths <= 0):
        return None
    counts = np.ceil(lengths / (max_step - 1e-6)).astype(int)
    steps = int(counts.sum())
    if steps > horizon - 1:
        return None
    positions = [polyline[0]]
    for start, goal, count in zip(polyline[:-1], polyline[1:], counts):
        positions.extend(start + (goal - start) * (index / count)
                         for index in range(1, count + 1))
    positions = np.asarray(positions, dtype=np.float64)
    forbidden = np.zeros((horizon, steps + 1), dtype=bool)
    for constraint in constraints or ():
        if constraint.get_is_soft():
            continue
        for center, time_range, radius in zip(
                constraint.get_q_l(), constraint.get_t_range_l(), constraint.get_radius_l()):
            first = max(0, int(time_range[0]))
            last = min(horizon - 1, int(time_range[1]))
            if first > last:
                continue
            point = np.asarray(center.detach().cpu() if isinstance(center, torch.Tensor)
                               else center, dtype=np.float64)
            near = np.linalg.norm(positions - point[:2], axis=-1) < float(radius) - 1e-9
            forbidden[first:last + 1] |= near[None]
    if forbidden[0, 0] or forbidden[-1, -1]:
        return None
    cost = np.full((horizon, steps + 1), np.inf)
    parent = np.full((horizon, steps + 1), -1, dtype=np.int16)
    cost[0, 0] = 0.
    for t in range(1, horizon):
        for index in range(max(0, steps - (horizon - 1 - t)), min(steps, t) + 1):
            if forbidden[t, index]:
                continue
            # Prefer even progress but permit necessary waits at any point.
            penalty = (index - t * steps / (horizon - 1))**2
            for prior in (index - 1, index):
                if prior >= 0 and cost[t - 1, prior] + penalty < cost[t, index]:
                    cost[t, index] = cost[t - 1, prior] + penalty
                    parent[t, index] = prior
    if not np.isfinite(cost[-1, -1]):
        return None
    indices = np.empty(horizon, dtype=np.int16)
    indices[-1] = steps
    for t in range(horizon - 1, 0, -1):
        indices[t - 1] = parent[t, indices[t]]
    scheduled = positions[indices].astype(np.float32)
    displacement = np.concatenate((np.diff(scheduled, axis=0),
                                   np.zeros((1, 2), np.float32)), axis=0)
    displacement[0] = 0
    return np.concatenate((scheduled, displacement), axis=-1)


def rrt_static_path(world, agent, *, constraints=None, max_time_s=2.,
                    seed_offsets=(37, 0, 73, 149)):
    """Return a checked [64,4] path satisfying static and CBS constraints."""
    if world.obstacles.get("kind") not in ("circles", "boxes"):
        raise ValueError("RRT fallback expects exact circles or boxes")
    if abs(world.robot_radius - .05) > 1e-8:
        raise ValueError("Pinned RobotPlanarDisk fallback requires the released 0.05 m radius")
    sys.path[:0] = [str(MMD), str(MMD / "deps/torch_robotics"),
                    str(MMD / "deps/motion_planning_baselines")]
    from mp_baselines.planners.rrt_connect import RRTConnect
    from torch_robotics.environments import EnvEmptyNoWait2DExtraObjects
    from torch_robotics.robots import RobotPlanarDisk
    from torch_robotics.tasks.tasks import PlanningTask

    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    env = EnvEmptyNoWait2DExtraObjects(precompute_sdf_obj_fixed=False, tensor_args=tensor_args)
    robot = RobotPlanarDisk(tensor_args=tensor_args)
    task = PlanningTask(env=env, robot=robot, tensor_args=tensor_args)
    install_native_obstacles(SimpleNamespace(tasks={0: task}, env=env), world,
                             tensor_args=tensor_args)
    start = torch.as_tensor(world.starts[agent], **tensor_args)
    goal = torch.as_tensor(world.goals[agent], **tensor_args)
    strict = replace(world, contract="common_strict")
    records = []
    numpy_rng_state = np.random.get_state()
    try:
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            for offset in seed_offsets:
                seed = 900000 + int(world.starts[agent, 0] * 100000) + 100 * agent + offset
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32))
                begun = time.monotonic()
                rrt = RRTConnect(task=task, n_iters=20000, start_state_pos=start, goal_state_pos=goal,
                                 step_size=.02, n_radius=.10, max_time=max_time_s,
                                 n_pre_samples=3000, tensor_args=tensor_args)
                route = rrt.optimize()
                record = dict(seed=seed, seconds=time.monotonic() - begun, returned=route is not None)
                records.append(record)
                if route is None:
                    continue
                points = np.asarray([point.detach().cpu().numpy() for point in route], dtype=np.float64)
                record.update(raw_waypoints=len(points),
                              raw_path_length_m=float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum()),
                              raw_max_segment_m=float(np.linalg.norm(np.diff(points, axis=0), axis=-1).max()),
                              allowed_max_step_m=strict.max_step or .05,
                              support_count=strict.horizon)
                if (np.linalg.norm(points[0] - world.starts[agent]) > 1e-5 or
                        np.linalg.norm(points[-1] - world.goals[agent]) > 1e-5):
                    record["failure"] = "endpoints"
                    continue
                shorter = shortcut_route(points, world.obstacles,
                                         robot_radius=world.robot_radius)
                if shorter is None:
                    record["failure"] = "shortcut"
                    continue
                lengths = np.linalg.norm(np.diff(shorter, axis=0), axis=-1)
                record.update(shortcut_waypoints=len(shorter),
                              shortcut_path_length_m=float(lengths.sum()),
                              shortcut_max_segment_m=float(lengths.max()),
                              required_steps=int(np.ceil(lengths / (strict.max_step or .05) - 1e-10).sum()),
                              available_steps=strict.horizon - 1,
                              minimum_possible_mean_step_m=float(lengths.sum() / (strict.horizon - 1)))
                if world.contract == "smd_native":
                    base = resample_continuous_route(shorter, world.horizon)
                    if base is None:
                        record["failure"] = "route_has_more_edges_than_support_intervals"
                        continue
                    base[0, :2] = world.starts[agent]
                    base[-1, :2] = world.goals[agent]
                    if world.path_valid(base, agent) and _satisfies_hard_constraints(base, constraints):
                        record.update(valid=True, timing="native_arclength",
                                      path_m=float(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=-1).sum()),
                                      max_step_m=float(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=-1).max()),
                                      common_strict_valid=bool(strict.path_valid(base, agent)))
                        return torch.from_numpy(base), records
                    record["failure"] = "native_route_constraints_or_samples"
                    if not constraints:
                        continue
                    # A constrained CT child may need to wait at a safe
                    # waypoint. Schedule on the same checked geometry with
                    # no native speed budget, then apply the native sampled
                    # and CBS point checks. Every movement follows one of
                    # the already checked polyline edges.
                    scheduled_native = constraint_aware_time_parameterize(
                        shorter, constraints, world.horizon, max_step=1e9)
                    if scheduled_native is not None:
                        scheduled_native[0, :2] = world.starts[agent]
                        scheduled_native[-1, :2] = world.goals[agent]
                        if (world.path_valid(scheduled_native, agent) and
                                _satisfies_hard_constraints(scheduled_native, constraints)):
                            record.pop("failure", None)
                            record.update(valid=True, timing="native_constraint_aware",
                                          path_m=float(np.linalg.norm(
                                              np.diff(scheduled_native[:, :2], axis=0), axis=-1).sum()),
                                          max_step_m=float(np.linalg.norm(
                                              np.diff(scheduled_native[:, :2], axis=0), axis=-1).max()),
                                          common_strict_valid=bool(strict.path_valid(scheduled_native, agent)))
                            return torch.from_numpy(scheduled_native), records
                    # Preserve the existing strict timing as an additional
                    # candidate when the native schedule misses a constraint.
                base = time_parameterize(shorter, strict.horizon, strict.max_step or .05)
                if base is None:
                    record["failure"] = "speed_budget"
                    continue
                minimum_steps = int(np.ceil(np.linalg.norm(np.diff(shorter, axis=0), axis=-1) /
                                            (strict.max_step or .05) - 1e-10).sum())
                maximum_hold = strict.horizon - 1 - minimum_steps
                for hold in range(maximum_hold + 1):
                    if hold:
                        moving = time_parameterize(shorter, strict.horizon - hold,
                                                   strict.max_step or .05)
                        if moving is None:
                            continue
                        positions = np.concatenate((np.repeat(moving[:1, :2], hold, axis=0),
                                                    moving[:, :2]), axis=0)
                        displacement = np.concatenate((np.diff(positions, axis=0),
                                                       np.zeros((1, 2), np.float32)), axis=0)
                        displacement[0] = 0
                        path = np.concatenate((positions, displacement), axis=-1)
                    else:
                        path = base.copy()
                    # Preserve the scene's exact boundary conditions.
                    path[0, :2] = world.starts[agent]
                    path[-1, :2] = world.goals[agent]
                    if (strict.path_valid(path, agent) and
                            _satisfies_hard_constraints(path, constraints)):
                        record.pop("failure", None)
                        record.update(valid=True, start_hold_steps=hold,
                                      shortcut_waypoints=len(shorter),
                                      path_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=-1).sum()),
                                      max_step_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=-1).max()))
                        return torch.from_numpy(path), records
                scheduled = constraint_aware_time_parameterize(
                    shorter, constraints, strict.horizon, strict.max_step or .05)
                if scheduled is not None:
                    scheduled[0, :2] = world.starts[agent]
                    scheduled[-1, :2] = world.goals[agent]
                    if (strict.path_valid(scheduled, agent) and
                            _satisfies_hard_constraints(scheduled, constraints)):
                        record.pop("failure", None)
                        record.update(valid=True, timing="constraint_aware",
                                      path_m=float(np.linalg.norm(np.diff(scheduled[:, :2], axis=0), axis=-1).sum()),
                                      max_step_m=float(np.linalg.norm(np.diff(scheduled[:, :2], axis=0), axis=-1).max()))
                        return torch.from_numpy(scheduled), records
                record["failure"] = "no_constraint_safe_timing"
    finally:
        np.random.set_state(numpy_rng_state)
    return None, records


def smd_rrt_static_path(world, agent, **kwargs):
    """Preserve the previously validated SMD entry point and parameters."""
    return rrt_static_path(world, agent, **kwargs)
