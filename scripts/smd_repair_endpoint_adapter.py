"""SMD endpoint precheck adapter for the pinned generic CBS repair backend.

The pinned MMD task uses a 1.1x robot link margin plus a 0.01 m cutoff when
checking world collisions. Those planning margins are not part of the released
SMD sampled obstacle predicate. Only CBS construction's start/goal check uses
this adapter; the planner's original task is restored immediately afterwards.
"""
from __future__ import annotations

import torch


class SMDNativeEndpointTask:
    def __init__(self, task, scene):
        if scene["obstacles"]["kind"] != "circles":
            raise ValueError("SMD native endpoint adapter requires exact circles")
        self.task = task
        self.scene = scene

    def __getattr__(self, name):
        return getattr(self.task, name)

    def compute_collision(self, q, **kwargs):
        """Return native sampled obstacle or center-workspace invalidity.

        The 1e-3 squared-distance threshold is from released SMD
        ``is_collision.py``. The robot radius is exactly 0.05 m, once.
        """
        positions = self.task.robot.get_position(q)[..., :2]
        bounds = torch.as_tensor(self.scene["workspace"], device=positions.device,
                                 dtype=positions.dtype)
        invalid = ~torch.isfinite(positions).all(dim=-1)
        invalid |= (positions < bounds[0]).any(dim=-1)
        invalid |= (positions > bounds[1]).any(dim=-1)
        for obstacle in self.scene["obstacles"]["items"]:
            center = torch.as_tensor(obstacle["center"], device=positions.device,
                                     dtype=positions.dtype)
            squared_distance = ((positions - center) ** 2).sum(dim=-1)
            native_threshold = (0.05 + float(obstacle["radius"])) ** 2 - 0.001
            invalid |= squared_distance < native_threshold
        return invalid.unsqueeze(-1)


def bind_smd_endpoint_precheck(native, scene):
    """Adapt CBS construction only, restoring its original task before use."""
    base_cbs = native.CBS

    class SMDPrecheckedCBS(base_cbs):
        def __init__(self, *args, reference_task=None, **kwargs):
            if reference_task is None:
                raise ValueError("SMD repair requires an explicit reference task")
            endpoint_task = SMDNativeEndpointTask(reference_task, scene)
            super().__init__(*args, reference_task=endpoint_task, **kwargs)
            self.reference_task = reference_task

    native.CBS = SMDPrecheckedCBS


class SMDNativeRootAdmissionWorld:
    """Root prepass gate using sampled native obstacles and task endpoints.

    This wrapper is passed only to the existing root static-repair function.
    Child filtering, RRT, and CBS continue to receive the original world.
    """
    def __init__(self, world):
        self.world = world

    def __getattr__(self, name):
        return getattr(self.world, name)

    def path_valid(self, path, agent):
        import numpy as np

        values = np.asarray(path, dtype=np.float64)
        if values.shape != (self.world.horizon, 4) or not np.isfinite(values).all():
            return False
        positions = values[:, :2]
        if (np.linalg.norm(positions[0] - self.world.starts[agent]) > 1e-5 or
                np.linalg.norm(positions[-1] - self.world.goals[agent]) > 1e-5):
            return False
        bounds = self.world.workspace
        for endpoint in (positions[0], positions[-1]):
            if np.any(endpoint < bounds[0]) or np.any(endpoint > bounds[1]):
                return False
        for obstacle in self.world.obstacles["items"]:
            center = np.asarray(obstacle["center"], dtype=np.float64)
            threshold = (self.world.robot_radius + float(obstacle["radius"])) ** 2 - 0.001
            if np.any(np.sum((positions - center) ** 2, axis=-1) < threshold):
                return False
        return True


def bind_smd_root_admission():
    """Keep the generic root algorithm; adapt only its SMD admission gate."""
    import obstacle_aware_repair

    original = obstacle_aware_repair.repair_root_with_native_low_level
    if getattr(original, "_smd_native_root_admission", False):
        return

    def adapted(planner, paths, world, make_experience, **kwargs):
        admission_world = (SMDNativeRootAdmissionWorld(world)
                           if world.contract == "smd_native" else world)
        return original(planner, paths, admission_world, make_experience, **kwargs)

    adapted._smd_native_root_admission = True
    obstacle_aware_repair.repair_root_with_native_low_level = adapted
