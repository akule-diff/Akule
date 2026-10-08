"""Exact continuous segment safety audit for planar disk benchmark paths.

The native SMD sampled evaluator is a separate contract.  This module only
implements the common strict physical audit used on both complete planners.
"""

from __future__ import annotations

import numpy as np


def _segment_point_distance(a, b, point):
    delta = b - a
    denominator = float(np.dot(delta, delta))
    fraction = float(np.clip(np.dot(point - a, delta) / denominator, 0, 1)) if denominator else 0.0
    return float(np.linalg.norm(a + fraction * delta - point))


def _point_box_distance(point, center, size):
    return float(np.linalg.norm(np.maximum(np.abs(point - center) - size / 2, 0)))


def _segment_box_distance(a, b, center, size):
    """Exact Euclidean distance between a 2D line segment and an AABB."""
    delta = b - a
    lower, upper = center - size / 2, center + size / 2
    cuts = [0.0, 1.0]
    for axis in range(2):
        if abs(delta[axis]) > 1e-15:
            cuts.extend(t for t in ((lower[axis] - a[axis]) / delta[axis],
                                    (upper[axis] - a[axis]) / delta[axis]) if 0 < t < 1)
    cuts = sorted(set(cuts))
    best = float("inf")
    for left, right in zip(cuts[:-1], cuts[1:]):
        midpoint = a + ((left + right) / 2) * delta
        active = (midpoint < lower) | (midpoint > upper)
        boundary = np.where(midpoint < lower, lower, upper)
        slope = delta * active
        intercept = (a - boundary) * active
        norm_sq = float(np.dot(slope, slope))
        stationary = -float(np.dot(intercept, slope)) / norm_sq if norm_sq else left
        for t in (left, right, float(np.clip(stationary, left, right))):
            best = min(best, _point_box_distance(a + t * delta, center, size))
    return best


def obstacle_segment_clearance(a, b, obstacles, robot_radius=0.05):
    """Exact continuous clearance of one disk-center segment from a scene."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    kind = (obstacles or {}).get("kind")
    if kind == "circles":
        return min((_segment_point_distance(a, b, np.asarray(item["center"], dtype=np.float64))
                    - robot_radius - float(item["radius"]) for item in obstacles["items"]),
                   default=float("inf"))
    if kind == "boxes":
        return min((_segment_box_distance(a, b, np.asarray(center, dtype=np.float64),
                                          np.asarray(size, dtype=np.float64)) - robot_radius
                    for center, size in zip(obstacles["centers"], obstacles["sizes"])),
                   default=float("inf"))
    if kind is None:
        return float("inf")
    raise ValueError(f"unsupported obstacle geometry: {kind}")


def smd_native_success(path, obstacles, *, robot_radius=.05, threshold=1e-3):
    """Released SMD ``is_collision.py`` sampled rule for candidate zero.

    This intentionally omits swept geometry, endpoints, workspace, and speed.
    Those belong to the separately labeled common-strict audit.
    """
    positions = np.asarray(path, dtype=np.float64)[..., :2]
    if positions.ndim != 3:
        raise ValueError("SMD path must be [T,N,2+]")
    count = positions.shape[1]
    for left in range(count):
        for right in range(left + 1, count):
            distance_sq = np.sum((positions[:, left] - positions[:, right]) ** 2, axis=-1)
            if np.any(distance_sq < (2 * robot_radius) ** 2 - threshold):
                return False
    if obstacles.get("kind") != "circles":
        raise ValueError("SMD native evaluator expects exact circle obstacles")
    for item in obstacles["items"]:
        center = np.asarray(item["center"], dtype=np.float64)
        distance_sq = np.sum((positions - center) ** 2, axis=-1)
        if np.any(distance_sq < (robot_radius + float(item["radius"])) ** 2 - threshold):
            return False
    return True


def evaluate(path, starts, goals, obstacles=None, *, robot_radius=0.05,
             workspace=((-1.0, -1.0), (1.0, 1.0)), endpoint_tolerance=1e-5):
    """Audit [T,N,2+] positions with exact swept disk separation.

    ``obstacles`` accepts scene records with ``kind=circles`` and ``items``,
    or ``kind=boxes`` with full ``sizes`` and ``centers``.
    """
    path = np.asarray(path, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    if path.ndim != 3 or path.shape[-1] < 2:
        raise ValueError("path must be [T,N,2+] with at least two support points")
    positions = path[..., :2]
    horizon, count, _ = positions.shape
    if horizon < 2 or starts.shape != (count, 2) or goals.shape != (count, 2):
        raise ValueError("path and endpoint shapes disagree")
    if not np.isfinite(positions).all():
        raise ValueError("path contains nonfinite positions")
    radius = float(robot_radius)
    sampled_pair = float("inf")
    swept_pair = float("inf")
    for left in range(count):
        for right in range(left + 1, count):
            relative = positions[:, left] - positions[:, right]
            sampled_pair = min(sampled_pair, float(np.linalg.norm(relative, axis=-1).min()) - 2 * radius)
            for a, b in zip(relative[:-1], relative[1:]):
                swept_pair = min(swept_pair, _segment_point_distance(a, b, np.zeros(2)) - 2 * radius)
    sampled_obstacle = float("inf")
    swept_obstacle = float("inf")
    obstacles = obstacles or {}
    kind = obstacles.get("kind")
    if kind == "circles":
        shapes = [(np.asarray(item["center"], dtype=np.float64), float(item["radius"]))
                  for item in obstacles["items"]]
        for center, obstacle_radius in shapes:
            distance = np.linalg.norm(positions - center, axis=-1)
            sampled_obstacle = min(sampled_obstacle, float(distance.min()) - radius - obstacle_radius)
            for a, b in zip(positions[:-1], positions[1:]):
                for p, q in zip(a, b):
                    swept_obstacle = min(swept_obstacle,
                                         _segment_point_distance(p, q, center) - radius - obstacle_radius)
    elif kind == "boxes":
        for center, size in zip(obstacles["centers"], obstacles["sizes"]):
            center = np.asarray(center, dtype=np.float64)
            size = np.asarray(size, dtype=np.float64)
            sampled_obstacle = min(sampled_obstacle,
                                   min(_point_box_distance(p, center, size) - radius
                                       for p in positions.reshape(-1, 2)))
            for a, b in zip(positions[:-1], positions[1:]):
                for p, q in zip(a, b):
                    swept_obstacle = min(swept_obstacle,
                                         _segment_box_distance(p, q, center, size) - radius)
    elif kind is not None:
        raise ValueError(f"unsupported obstacle geometry: {kind}")
    lower, upper = np.asarray(workspace, dtype=np.float64)
    margin = min(float((positions - lower).min()), float((upper - positions).min()))
    start_error = float(np.linalg.norm(positions[0] - starts, axis=-1).max())
    goal_error = float(np.linalg.norm(positions[-1] - goals, axis=-1).max())
    result = dict(sampled_pair_clearance_m=sampled_pair,
                  swept_pair_clearance_m=swept_pair,
                  sampled_obstacle_clearance_m=sampled_obstacle,
                  swept_obstacle_clearance_m=swept_obstacle,
                  workspace_margin_m=margin,
                  start_error_m=start_error, goal_error_m=goal_error)
    result["success"] = bool(sampled_pair >= -1e-9 and swept_pair >= -1e-9
                             and sampled_obstacle >= -1e-9 and swept_obstacle >= -1e-9
                             and margin >= -1e-9 and max(start_error, goal_error) <= endpoint_tolerance)
    return result
