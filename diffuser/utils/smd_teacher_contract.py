"""Shared SMD physical constraints for offline signed teacher search only.

The normalized swept-circle, step, and workspace terms are the existing
Capacity V3 formulation with an absolute SMD floor/ceiling in place of V3's
relative-to-unary bounds. No learned-model objective imports this module.
"""
from __future__ import annotations

import numpy as np
import torch


class SMDTeacherContract:
    def __init__(self, scene, device):
        self.scene = scene
        items = scene["obstacles"]["items"]
        self.centers = torch.as_tensor([x["center"] for x in items], device=device,
                                       dtype=torch.float32).reshape(-1, 2)
        self.radii = torch.as_tensor([x["radius"] for x in items], device=device,
                                     dtype=torch.float32)

    def penalty(self, path):
        position = path[..., :2]
        a = position[:, :-1, :, None, :]
        b = position[:, 1:, :, None, :]
        delta = b-a
        if len(self.centers):
            center = self.centers[None, None, None]
            fraction = (((center-a)*delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
            clearance = torch.linalg.vector_norm(a+fraction[..., None]*delta-center, dim=-1)-self.radii-.05
            obstacle = (-clearance.amin(dim=(1, 3))).clamp_min(0)/.05
        else:
            obstacle = position.new_zeros((position.shape[0], position.shape[2]))
        step = torch.linalg.vector_norm(torch.diff(position, dim=1), dim=-1)
        speed = (step-.05).clamp_min(0)/.05
        workspace = (position.abs()-.95).clamp_min(0)/.05
        # Capacity V3's normalized hard-term form and fixed factor.
        return 10*sum(x.square().flatten(1).mean(1)+x.square().flatten(1).amax(1)
                      for x in (obstacle, speed, workspace))

    def audit(self, path):
        position = path.detach().cpu().numpy()[0, ..., :2].astype(np.float64)
        scene = self.scene
        if position.shape != (64, len(scene["starts"]), 2) or not np.isfinite(position).all():
            return dict(qualified=False, failure=["numerical"])
        start = float(np.linalg.norm(position[0]-scene["starts"], axis=-1).max())
        goal = float(np.linalg.norm(position[-1]-scene["goals"], axis=-1).max())
        workspace = float(.95-np.abs(position).max())
        delta = np.diff(position, axis=0)
        step = float(np.linalg.norm(delta, axis=-1).max())
        i, j = np.triu_indices(position.shape[1], 1)
        relative = position[:, i]-position[:, j]
        sampled_pair = float(np.linalg.norm(relative, axis=-1).min())
        pair_a = relative[:-1]
        pair_delta = np.diff(relative, axis=0)
        fraction = np.clip(-(pair_a*pair_delta).sum(-1)/np.maximum((pair_delta**2).sum(-1), 1e-30), 0, 1)
        swept_pair = float(np.linalg.norm(pair_a+fraction[..., None]*pair_delta, axis=-1).min())
        sampled_obstacle = float("inf")
        swept_obstacle = float("inf")
        for item in scene["obstacles"]["items"]:
            center = np.asarray(item["center"], dtype=np.float64)
            radius = .05+float(item["radius"])
            obstacle_a = position[:-1]-center
            obstacle_t = np.clip(-(obstacle_a*delta).sum(-1)/np.maximum((delta**2).sum(-1), 1e-30), 0, 1)
            sampled_obstacle = min(sampled_obstacle, float(np.linalg.norm(position-center, axis=-1).min()-radius))
            swept_obstacle = min(swept_obstacle, float(np.linalg.norm(obstacle_a+obstacle_t[..., None]*delta, axis=-1).min()-radius))
        failures = []
        if sampled_pair < .120-1e-8 or swept_pair < .120-1e-8:
            failures.append("pair")
        if swept_obstacle < -1e-8:
            failures.append("obstacle")
        if step > .05+1e-8:
            failures.append("max_step")
        if workspace < -1e-8:
            failures.append("workspace")
        if max(start, goal) > 1e-5:
            failures.append("endpoint")
        return dict(qualified=not failures, failure=failures,
                    sampled_pair_distance_m=sampled_pair, swept_pair_distance_m=swept_pair,
                    sampled_obstacle_clearance_m=sampled_obstacle,
                    swept_obstacle_clearance_m=swept_obstacle,
                    workspace_disk_margin_m=workspace, max_step_m=step,
                    start_error_m=start, goal_error_m=goal)
