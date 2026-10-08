"""Vectorized physical teacher/proposal qualification for fixed ScaledWeave."""

from __future__ import annotations

import numpy as np

from scaled_weave import RADIUS_M
from scaled_weave_frame import ScaledWeaveFrame, SUPPORTS


CANONICAL_MAX_SPEED_MPS = .05 / (5.0 / SUPPORTS)


def validate(path, scene, *, pair_chunk=4096, endpoint_tolerance_m=1e-5):
    """Check exact linearly swept disk separation, speed, workspace, endpoints."""
    n = scene["population"]
    frame = ScaledWeaveFrame(n)
    path = np.asarray(path, dtype=np.float64)
    if path.shape != (SUPPORTS, n, 4) or not np.isfinite(path).all():
        raise ValueError("Expected finite physical [64,N,4] path")
    positions = path[..., :2]
    starts = np.asarray(scene["starts"], dtype=np.float64)
    goals = np.asarray(scene["goals"], dtype=np.float64)
    if starts.shape != (n, 2) or goals.shape != (n, 2):
        raise ValueError("Scene endpoints do not match population")
    extent = float(scene["workspace_half_extent_m"])
    if abs(extent - frame.scale) > 1e-10:
        raise RuntimeError("Scene workspace differs from frozen frame")
    lower, upper = np.array([-extent, -extent]), np.array([extent, extent])
    workspace_margin = min(float((positions - lower).min()),
                           float((upper - positions).min())) - RADIUS_M
    start_error = float(np.linalg.norm(positions[0] - starts, axis=-1).max())
    goal_error = float(np.linalg.norm(positions[-1] - goals, axis=-1).max())
    steps = np.diff(positions, axis=0)
    speeds = np.linalg.norm(steps, axis=-1) / frame.dt_s
    max_speed = float(speeds.max())
    acceleration = np.diff(positions, n=2, axis=0) / frame.dt_s**2
    jerk = np.diff(positions, n=3, axis=0) / frame.dt_s**3
    sampled_clearance = swept_clearance = float("inf")
    i, j = np.triu_indices(n, 1)
    for begin in range(0, len(i), pair_chunk):
        a, b = i[begin:begin + pair_chunk], j[begin:begin + pair_chunk]
        relative = positions[:, a] - positions[:, b]
        sampled_clearance = min(sampled_clearance,
                                float(np.linalg.norm(relative, axis=-1).min()) - 2 * RADIUS_M)
        first = relative[:-1]
        delta = relative[1:] - first
        denominator = np.sum(delta * delta, axis=-1)
        fraction = np.divide(-np.sum(first * delta, axis=-1), denominator,
                             out=np.zeros_like(denominator), where=denominator > 0)
        fraction = np.clip(fraction, 0, 1)
        closest = first + fraction[..., None] * delta
        swept_clearance = min(swept_clearance,
                              float(np.linalg.norm(closest, axis=-1).min()) - 2 * RADIUS_M)
    result = dict(N=n, supports=SUPPORTS, duration_parameter_s=frame.duration_s,
                  dt_s=frame.dt_s, sampled_pair_clearance_m=sampled_clearance,
                  swept_pair_clearance_m=swept_clearance,
                  workspace_disk_margin_m=workspace_margin,
                  start_error_m=start_error, goal_error_m=goal_error,
                  max_speed_mps=max_speed,
                  speed_limit_mps=CANONICAL_MAX_SPEED_MPS,
                  path_m=float(np.linalg.norm(steps, axis=-1).sum(axis=0).mean()),
                  acceleration_rms_mps2=float(np.sqrt(np.mean(np.sum(acceleration**2, axis=-1)))),
                  jerk_rms_mps3=float(np.sqrt(np.mean(np.sum(jerk**2, axis=-1)))),
                  execution_makespan_s=frame.dt_s * (SUPPORTS - 1))
    result["success"] = bool(sampled_clearance >= -1e-9 and swept_clearance >= -1e-9 and
                             workspace_margin >= -1e-9 and
                             max(start_error, goal_error) <= endpoint_tolerance_m and
                             max_speed <= CANONICAL_MAX_SPEED_MPS + 1e-9)
    return result
