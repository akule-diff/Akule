"""Physical timing and state reconstruction for released SMD composite paths."""

from __future__ import annotations

import numpy as np

HORIZON = 64
DURATION_S = 5.0
DT_S = DURATION_S / HORIZON  # Pinned SMD planner and projection convention.


def reconstructed_grouped_state(path):
    """[64,N,2+] -> [64,N,4] with interval-consistent velocity channels.

    Channels 2:4 use the canonical physical displacement convention. The
    released SMD hard condition sets endpoint velocity to zero. Interior
    support-point velocity is the forward position difference divided by dt.
    """
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != HORIZON or values.shape[-1] < 2:
        raise ValueError("SMD path must be [64,N,2+]")
    positions = values[..., :2]
    displacement = np.zeros_like(positions)
    displacement[1:-1] = np.diff(positions, axis=0)[1:]
    return np.concatenate((positions, displacement), axis=-1)


def temporal_metrics(path):
    """Report position-derived execution metrics, independent of stored velocity."""
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != HORIZON or values.shape[-1] < 2:
        raise ValueError("SMD path must be [64,N,2+]")
    positions = values[..., :2]
    if not np.isfinite(positions).all():
        raise ValueError("nonfinite SMD position")
    intervals = np.diff(positions, axis=0)
    speed = np.linalg.norm(intervals, axis=-1) / DT_S
    acceleration = np.linalg.norm(np.diff(positions, n=2, axis=0), axis=-1) / DT_S**2
    jerk = np.linalg.norm(np.diff(positions, n=3, axis=0), axis=-1) / DT_S**3
    result = dict(horizon=HORIZON, duration_parameter_s=DURATION_S, dt_s=DT_S,
                  last_support_time_s=(HORIZON - 1) * DT_S,
                  path_m=float(np.linalg.norm(intervals, axis=-1).sum(axis=0).mean()),
                  mean_speed_mps=float(speed.mean()), max_step_m=float(np.linalg.norm(intervals, axis=-1).max()),
                  mean_acceleration_mps2=float(acceleration.mean()),
                  acceleration_rms_mps2=float(np.sqrt(np.mean(acceleration**2))),
                  jerk_rms_mps3=float(np.sqrt(np.mean(jerk**2))))
    if values.shape[-1] >= 4:
        # MMD/ours grouped displacement is channels 2:4; native SMD grouped
        # state is converted to displacement by ``grouped_state`` first.
        stored = values[1:-1, :, 2:] / DT_S
        finite_difference = intervals[1:] / DT_S
        result["stored_velocity_consistency_rms_mps"] = float(
            np.sqrt(np.mean(np.sum((stored - finite_difference)**2, axis=-1))))
    return result
