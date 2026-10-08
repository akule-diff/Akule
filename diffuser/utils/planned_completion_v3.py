"""Sustained recorded position arrival, without retiming or stopping claims."""
from __future__ import annotations

import numpy as np


def planned_completion(
    paths,
    goals,
    timestamps,
    *,
    final_valid,
    planning_seconds,
    tolerance=0.05,
    failure_reasons=(),
):
    """Paths/time grids are per-agent lists, in metres and global seconds.

    Explicit grids include start offsets and any already recorded padding.
    The caller must check final trajectory validity independently. Arrival at
    the last recorded point is marked horizon-limited, not a stopping event.
    """
    if len(paths) != len(goals) or len(paths) != len(timestamps) or not len(paths):
        raise ValueError("one nonempty path and time grid required per goal")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid goal tolerance")
    if not np.isfinite(planning_seconds) or planning_seconds < 0:
        raise ValueError("invalid authoritative planning time")
    arrivals, limited, reasons = [], [], list(failure_reasons)
    for i, (path, goal, times) in enumerate(zip(paths, goals, timestamps)):
        p, g, t = np.asarray(path), np.asarray(goal), np.asarray(times)
        if p.ndim != 2 or p.shape[1] < 2 or t.ndim != 1 or len(t) != len(p):
            raise ValueError("path/time shape mismatch")
        if len(t) and (np.any(np.diff(t) <= 0) or t[0] < 0):
            raise ValueError("timestamps must increase on the global execution clock")
        if not len(t) or not all(np.isfinite(v).all() for v in (p, g, t)):
            arrivals.append(None)
            limited.append(None)
            reasons.append(f"agent_{i}:empty_or_nonfinite_trajectory")
            continue
        inside = np.linalg.norm(p[:, :2] - g[:2], axis=-1) <= tolerance
        sustained = np.logical_and.accumulate(inside[::-1])[::-1]
        indices = np.flatnonzero(sustained)
        if not len(indices):
            arrivals.append(None)
            limited.append(None)
            reasons.append(f"agent_{i}:no_sustained_goal_arrival")
        else:
            k = int(indices[0])
            arrivals.append(float(t[k]))
            limited.append(k == len(t) - 1)
    valid = bool(final_valid and all(a is not None for a in arrivals))
    if not final_valid:
        reasons.append("independent_final_validity_failed")
    # Invalid plans do not acquire successful agent/team completion times.
    diagnostic = arrivals
    if not valid:
        arrivals = [None] * len(paths)
    makespan = max(arrivals) if valid else None
    return dict(
        planning_wall_seconds=float(planning_seconds),
        scheduled_execution_makespan_seconds=makespan,
        request_to_completion_seconds=float(planning_seconds) + makespan
        if valid
        else None,
        per_agent_goal_arrival_seconds=arrivals,
        diagnostic_position_arrival_seconds=diagnostic,
        mean_agent_goal_arrival_seconds=float(np.mean(arrivals)) if valid else None,
        goal_arrival_valid=valid,
        goal_tolerance_m=float(tolerance),
        horizon_limited_arrival=limited,
        horizon_limited_agent_fraction=float(np.mean(limited)) if valid else None,
        team_horizon_limited=bool(
            any(limited[i] and arrivals[i] == makespan for i in range(len(paths)))
        )
        if valid
        else None,
        failure_reasons=reasons,
        settled_arrival_seconds=None,
        settled_arrival_convention="not established: no verified speed/dwell criterion",
        execution_time_convention="ideal tracking; global recorded timestamps; sustained position tolerance; no interpolation, retiming, or truncation",
    )


def uniform_joint_completion(physical, goals, dt, **kwargs):
    """Native globally padded joint output, zero global start, explicit dt."""
    x = np.asarray(physical)
    if x.ndim != 3 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("expected [time, agent, channel] and positive dt")
    times = np.arange(len(x), dtype=np.float64) * dt
    return planned_completion(
        [x[:, i] for i in range(x.shape[1])], goals, [times] * x.shape[1], **kwargs
    )
