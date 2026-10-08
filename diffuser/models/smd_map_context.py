"""Exact static SMD clearance context for the current MPD reverse state.

The scene's circle tensors are loaded once.  Every query is a vectorized,
analytic configuration-space clearance lookup, including the disk workspace.
"""
from __future__ import annotations

import torch


MAP_DIM = 4  # center clearance, unit gradient x/y, next-segment midpoint clearance
ROBOT_RADIUS = 0.05


class StaticMapContext:
    def __init__(self, obstacles, *, device, scale, workspace=((-1., -1.), (1., 1.))):
        if obstacles["kind"] != "circles":
            raise ValueError("SMD map context requires exact circle geometry")
        self.centers = torch.as_tensor([o["center"] for o in obstacles["items"]],
                                       device=device, dtype=torch.float32).reshape(-1, 2)
        self.radii = torch.as_tensor([o["radius"] + ROBOT_RADIUS for o in obstacles["items"]],
                                     device=device, dtype=torch.float32)
        self.low = torch.as_tensor(workspace[0], device=device, dtype=torch.float32) + ROBOT_RADIUS
        self.high = torch.as_tensor(workspace[1], device=device, dtype=torch.float32) - ROBOT_RADIUS
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError("TRAIN-derived clearance scale must be positive")

    def clearance_gradient(self, positions):
        boundary = torch.stack((positions[..., 0] - self.low[0],
                                self.high[0] - positions[..., 0],
                                positions[..., 1] - self.low[1],
                                self.high[1] - positions[..., 1]), dim=-1)
        distance, side = boundary.min(dim=-1)
        basis = positions.new_tensor(((1., 0.), (-1., 0.), (0., 1.), (0., -1.)))
        gradient = basis[side]
        if self.centers.numel():
            delta = positions[..., None, :] - self.centers
            norm = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-8)
            circle_distance, circle = (norm - self.radii).min(dim=-1)
            circle_gradient = (delta / norm[..., None]).gather(
                -2, circle[..., None, None].expand(*circle.shape, 1, 2)).squeeze(-2)
            use_circle = circle_distance < distance
            distance = torch.where(use_circle, circle_distance, distance)
            gradient = torch.where(use_circle[..., None], circle_gradient, gradient)
        gradient = gradient / torch.linalg.vector_norm(gradient, dim=-1, keepdim=True).clamp_min(1e-8)
        return distance, gradient

    def __call__(self, positions):
        """Return [batch,64,4] from physical current-state center positions."""
        clearance, gradient = self.clearance_gradient(positions)
        midpoint = (positions[:, :-1] + positions[:, 1:]) * 0.5
        mid_clearance, _ = self.clearance_gradient(midpoint)
        mid_clearance = torch.cat((mid_clearance, clearance[:, -1:]), dim=1)
        return torch.cat((clearance[..., None] / self.scale,
                          gradient, mid_clearance[..., None] / self.scale), dim=-1)
