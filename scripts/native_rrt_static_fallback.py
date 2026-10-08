"""Pinned RRTConnect fallback for SMD scene-static low-level failures.

The released MMD task supplies exact scene circles. RRT supplies geometry;
every shortcut, support point, step, endpoint, and swept segment is checked
against the independent physical world before injection into CBS.
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
from obstacle_aware_repair import _satisfies_hard_constraints, install_native_circles

ROOT = Path(__file__).resolve().parents[1]
MMD = ROOT / "external/mmd"


def shortcut_route(points, obstacles, margin=.005):
    result = [points[0]]
    left = 0
    while left < len(points) - 1:
        for right in range(len(points) - 1, left, -1):
            if obstacle_segment_clearance(points[left], points[right], obstacles, .05) >= margin:
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


def smd_rrt_static_path(world, agent, *, constraints=None, max_time_s=2.,
                        seed_offsets=(37, 0, 73, 149)):
    """Return a checked [64,4] path satisfying static and CBS constraints."""
    if world.obstacles.get("kind") != "circles":
        raise ValueError("SMD native RRT fallback expects exact circle obstacles")
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
    install_native_circles(SimpleNamespace(tasks={0: task}, env=env), world, tensor_args=tensor_args)
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
                if (np.linalg.norm(points[0] - world.starts[agent]) > 1e-5 or
                        np.linalg.norm(points[-1] - world.goals[agent]) > 1e-5):
                    record["failure"] = "endpoints"
                    continue
                shorter = shortcut_route(points, world.obstacles)
                if shorter is None:
                    record["failure"] = "shortcut"
                    continue
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
                        record.update(valid=True, start_hold_steps=hold,
                                      shortcut_waypoints=len(shorter),
                                      path_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=-1).sum()),
                                      max_step_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=-1).max()))
                        return torch.from_numpy(path), records
                record["failure"] = "no_constraint_safe_timing"
    finally:
        np.random.set_state(numpy_rng_state)
    return None, records
