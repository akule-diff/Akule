"""Released-SMD start/goal admission for the final population evaluation.

Generic CBS applies an auxiliary 0.15 m endpoint separation. Released SMD
uses its sampled disk collision predicate, including its fixed 1e-3 squared
distance tolerance. The original CBS check is restored after construction.
"""
from __future__ import annotations

import torch

import smd_repair_endpoint_adapter as original_adapter


def native_start_goal_valid(reference_robot, reference_task, starts, goals,
                            is_enforce_min_dist=True):
    del is_enforce_min_dist
    scene = reference_task.scene
    radius = float(scene["radii"][0])
    if any(abs(float(value) - radius) > 1e-10 for value in scene["radii"]):
        raise ValueError("Mixed SMD robot radii are unsupported")
    threshold = (2.0 * radius) ** 2 - 0.001
    for endpoints in (starts, goals):
        state = torch.stack(endpoints)
        positions = reference_robot.get_position(state)[..., :2]
        if bool(torch.any(reference_task.compute_collision(state))):
            return False
        delta = positions[:, None, :] - positions[None, :, :]
        distance_squared = (delta * delta).sum(dim=-1)
        row, col = torch.triu_indices(len(endpoints), len(endpoints), offset=1,
                                      device=positions.device)
        if bool(torch.any(distance_squared[row, col] < threshold)):
            return False
    return True


def bind_smd_population_endpoint_precheck(native, scene):
    """Use exact native endpoint admission only during CBS construction."""
    import mmd.planners.multi_agent.cbs as cbs_module

    base_cbs = native.CBS

    class SMDPopulationCBS(base_cbs):
        def __init__(self, *args, reference_task=None, **kwargs):
            if reference_task is None:
                raise ValueError("SMD repair requires an explicit reference task")
            endpoint_task = original_adapter.SMDNativeEndpointTask(reference_task, scene)
            old_check = cbs_module.is_multi_agent_start_goal_states_valid
            cbs_module.is_multi_agent_start_goal_states_valid = native_start_goal_valid
            try:
                super().__init__(*args, reference_task=endpoint_task, **kwargs)
            finally:
                cbs_module.is_multi_agent_start_goal_states_valid = old_check
            self.reference_task = reference_task

    native.CBS = SMDPopulationCBS
