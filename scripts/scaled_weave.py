"""Separate, physically scaled boundary-exchange benchmark (not canonical Weave)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RADIUS_M = 0.05
MIN_ENDPOINT_SPACING_M = 0.150
PHASE_LIMIT = 0.005
JITTER_FRACTION = 0.0025
CONSTRUCTION_RADIUS_AT_UNIT_SCALE_M = 0.8
SNAP_AT_UNIT_SCALE_M = 0.87
SAFETY_FACTOR = 1.001  # covers float32 rounding at the analytic lower bound


def scale_for_population(n: int) -> float:
    """Smallest analytic scale, with 0.1% numerical safety margin.

    Same-side neighbors have angular gap >= 2*pi/n*(1-2*jitter_fraction).
    On a snapped square side, |d tangential/d angle| >= r/sqrt(2).
    Adjacent-side points are separated by the snap-to-diagonal gap.
    """
    if n < 2:
        raise ValueError("Need at least two robots")
    minimum_scale = (MIN_ENDPOINT_SPACING_M * n * np.sqrt(2) /
                     (CONSTRUCTION_RADIUS_AT_UNIT_SCALE_M * 2 * np.pi *
                      (1 - 2 * JITTER_FRACTION)))
    return float(max(1.0, SAFETY_FACTOR * minimum_scale))


def positions(n: int, phase: float, jitter: np.ndarray):
    if jitter.shape != (n,):
        raise ValueError("One angular jitter per robot is required")
    scale = scale_for_population(n)
    angle = 2 * np.pi * np.arange(n) / n + phase + jitter
    starts = (CONSTRUCTION_RADIUS_AT_UNIT_SCALE_M * scale *
              np.stack((np.cos(angle), np.sin(angle)), axis=-1)).astype(np.float32)
    dominant = np.abs(starts[:, 0]) > np.abs(starts[:, 1])
    snap = np.float32(SNAP_AT_UNIT_SCALE_M * scale)
    starts[dominant, 0] = np.sign(starts[dominant, 0]) * snap
    starts[~dominant, 1] = np.sign(starts[~dominant, 1]) * snap
    goals = starts.copy()
    goals[dominant, 0] *= -1
    goals[~dominant, 1] *= -1
    return starts, goals, dominant


def min_endpoint_spacing(starts: np.ndarray, goals: np.ndarray) -> float:
    n = len(starts)
    i, j = np.triu_indices(n, 1)
    return float(min(np.linalg.norm(starts[i] - starts[j], axis=-1).min(),
                     np.linalg.norm(goals[i] - goals[j], axis=-1).min()))


def scene(n: int, ordinal: int):
    rng = np.random.default_rng(28_600_000 + 1000 * n + ordinal)
    phase = float(rng.uniform(-PHASE_LIMIT, PHASE_LIMIT))
    jitter = rng.uniform(-JITTER_FRACTION, JITTER_FRACTION, n) * 2 * np.pi / n
    starts, goals, dominant = positions(n, phase, jitter)
    scale = scale_for_population(n)
    separation = min_endpoint_spacing(starts, goals)
    if separation < MIN_ENDPOINT_SPACING_M:
        raise RuntimeError(f"Analytic ScaledWeave spacing bound failed: {separation}")
    if max(np.abs(starts).max(), np.abs(goals).max()) + RADIUS_M > scale:
        raise RuntimeError("Robot disk leaves ScaledWeave workspace")
    if np.any(np.linalg.norm(starts - goals, axis=-1) < MIN_ENDPOINT_SPACING_M):
        raise RuntimeError("Degenerate reflection goal")
    record = dict(family="ScaledWeave-v1", population=n, split="pilot",
                  starts=starts.tolist(), goals=goals.tolist(), radii=[RADIUS_M] * n,
                  workspace_half_extent_m=scale,
                  configuration_id=100_000_000 + 1_000_000 * n + ordinal,
                  noise_seed=200_000_000 + 1_000_000 * n + 4 * ordinal,
                  construction=dict(rule="scaled-circle-snap-reflect",
                                    phase_radians=phase, angle_jitter_radians=jitter.tolist(),
                                    construction_radius_m=CONSTRUCTION_RADIUS_AT_UNIT_SCALE_M * scale,
                                    snap_coordinate_m=SNAP_AT_UNIT_SCALE_M * scale,
                                    scale=scale, horizontal_reflections=int(dominant.sum()),
                                    vertical_reflections=int((~dominant).sum())),
                  endpoint_min_separation_m=separation)
    record["geometry_sha256"] = hashlib.sha256(
        np.round(np.concatenate((starts, goals), axis=1), 8).tobytes()).hexdigest()
    record["input_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=int, nargs="+", default=[40, 64, 100, 128, 256])
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("w")
    else:
        stream = None
    try:
        for n in args.populations:
            records = [scene(n, ordinal) for ordinal in range(args.count)]
            repeated = [scene(n, ordinal) for ordinal in range(args.count)]
            assert [r["geometry_sha256"] for r in records] == [
                r["geometry_sha256"] for r in repeated]
            assert len({r["geometry_sha256"] for r in records}) == args.count
            if stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(dict(population=n, count=args.count,
                                  scale=scale_for_population(n),
                                  minimum_spacing_m=min(r["endpoint_min_separation_m"]
                                                        for r in records),
                                  maximum_endpoint_m=max(max(abs(x) for pair in
                                    (r["starts"] + r["goals"]) for x in pair)
                                    for r in records),
                                  unique_geometry_hashes=args.count)))
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
