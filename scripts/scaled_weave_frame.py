"""Explicit similarity frame and motion clock for ScaledWeave pilot proposals.

Model coordinates occupy the frozen unary MPD's [-1,1]^2 frame. Physical
positions and all distance thresholds remain in metres. The pilot clock
scales with workspace side so the nominal route speed is population-invariant.
This adapter does not claim that frozen R/G/U are calibrated at large N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scaled_weave import RADIUS_M, scale_for_population


SUPPORTS = 64
REFERENCE_DURATION_S = 5.0


@dataclass(frozen=True)
class ScaledWeaveFrame:
    population: int

    @property
    def scale(self):
        return scale_for_population(self.population)

    @property
    def duration_s(self):
        return REFERENCE_DURATION_S * self.scale

    @property
    def dt_s(self):
        return self.duration_s / SUPPORTS

    @property
    def model_robot_radius(self):
        return RADIUS_M / self.scale

    def positions_to_model(self, physical):
        return np.asarray(physical, dtype=np.float32) / self.scale

    def positions_to_physical(self, model):
        return np.asarray(model, dtype=np.float32) * self.scale

    def displacements_to_model(self, physical):
        return np.asarray(physical, dtype=np.float32) / self.scale

    def displacements_to_physical(self, model):
        return np.asarray(model, dtype=np.float32) * self.scale

    def grouped_to_model(self, physical):
        state = np.asarray(physical, dtype=np.float32)
        if state.shape[-2:] != (self.population, 4):
            raise ValueError("Expected [...,N,4] grouped physical state")
        return state / self.scale

    def grouped_to_physical(self, model):
        state = np.asarray(model, dtype=np.float32)
        if state.shape[-2:] != (self.population, 4):
            raise ValueError("Expected [...,N,4] grouped model state")
        return state * self.scale

    def physical_speed_mps(self, model_step_displacement):
        # Both displacement and duration scale by s, so speed is unchanged.
        return np.linalg.norm(np.asarray(model_step_displacement), axis=-1) * SUPPORTS / REFERENCE_DURATION_S


def validate_frame(scene):
    frame = ScaledWeaveFrame(scene["population"])
    if abs(frame.scale - scene["workspace_half_extent_m"]) > 1e-10:
        raise RuntimeError("Scene/frame workspace scale differs")
    starts = np.asarray(scene["starts"], dtype=np.float32)
    goals = np.asarray(scene["goals"], dtype=np.float32)
    for endpoints in (starts, goals):
        model = frame.positions_to_model(endpoints)
        if np.max(np.abs(model)) > 1 - frame.model_robot_radius + 1e-6:
            raise RuntimeError("Model endpoint leaves scaled workspace")
        if not np.allclose(frame.positions_to_physical(model), endpoints, atol=2e-6):
            raise RuntimeError("Frame round trip failed")
    # The frozen coordination state stores displacement per support, not
    # velocity per second. Both displacement and clock scale by s(N), so a
    # decoded trajectory retains the model-frame physical speed.
    model_step = frame.displacements_to_model(goals - starts) / (SUPPORTS - 1)
    physical_step = frame.displacements_to_physical(model_step)
    speed_from_physical = np.linalg.norm(physical_step, axis=-1) / frame.dt_s
    speed_from_model = np.linalg.norm(model_step, axis=-1) * SUPPORTS / REFERENCE_DURATION_S
    if not np.allclose(speed_from_physical, speed_from_model, rtol=2e-6, atol=2e-6):
        raise RuntimeError("ScaledWeave motion-clock speed contract changed")
    return frame
