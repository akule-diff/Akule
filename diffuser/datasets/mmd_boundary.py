"""Provenance-safe geometry and records for the MMD Boundary-Family protocol.

The released Boundary benchmark is deterministic and is never constructed by
this module.  This module is for the explicitly labelled *training family*: a
seeded global phase and small per-agent angular perturbations are applied
before the released Boundary square snap and reflected-goal rule.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BoundaryFamilyScene:
    """One randomized Boundary-Family start/goal scene in agent order."""

    starts: np.ndarray
    goals: np.ndarray
    phase: float
    jitter: np.ndarray
    seed: int


def boundary_family_scene(
    num_agents: int,
    seed: int,
    jitter_fraction: float = 0.15,
    dist: float = 0.87,
) -> BoundaryFamilyScene:
    """Sample a deterministic Boundary-Family scene.

    This retains the released Boundary's equally spaced angular construction,
    square-boundary snap, and dominant-coordinate goal reflection.  It is not
    the deterministic official evaluation configuration.
    """

    if num_agents < 2:
        raise ValueError("Boundary-Family requires at least two agents.")
    if not 0 <= jitter_fraction < 0.5:
        raise ValueError("jitter_fraction must be in [0, 0.5).")
    rng = np.random.default_rng(seed)
    spacing = 2.0 * np.pi / num_agents
    phase = float(rng.uniform(0.0, spacing))
    jitter = rng.uniform(-jitter_fraction * spacing, jitter_fraction * spacing, num_agents)
    angles = spacing * np.arange(num_agents) + phase + jitter
    starts = np.stack((0.8 * np.cos(angles), 0.8 * np.sin(angles)), axis=-1)
    dominant_x = np.abs(starts[:, 0]) > np.abs(starts[:, 1])
    starts[dominant_x, 0] = np.sign(starts[dominant_x, 0]) * dist
    starts[~dominant_x, 1] = np.sign(starts[~dominant_x, 1]) * dist
    goals = starts.copy()
    goals[dominant_x, 0] *= -1.0
    goals[~dominant_x, 1] *= -1.0
    _assert_minimum_separation(starts, goals)
    return BoundaryFamilyScene(
        starts=starts.astype(np.float32),
        goals=goals.astype(np.float32),
        phase=phase,
        jitter=jitter.astype(np.float32),
        seed=seed,
    )


def _assert_minimum_separation(starts: np.ndarray, goals: np.ndarray, minimum: float = 0.15) -> None:
    for positions, name in ((starts, "starts"), (goals, "goals")):
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < minimum:
            raise ValueError(f"Boundary-Family {name} violate {minimum} minimum separation.")


def canonical_scene_hash(starts: np.ndarray, goals: np.ndarray) -> str:
    """Hash paired starts/goals invariant to agent permutations.

    Pairs are sorted lexicographically after rounding; a mere relabeling
    cannot appear as an independent train/validation/test scene.
    """

    starts = np.asarray(starts, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    if starts.shape != goals.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("starts and goals must both be [N,2].")
    pairs = np.concatenate((starts, goals), axis=1)
    pairs = np.round(pairs, decimals=8)
    pairs = pairs[np.lexsort(tuple(pairs[:, index] for index in range(pairs.shape[1] - 1, -1, -1)))]
    return hashlib.sha256(pairs.tobytes()).hexdigest()
