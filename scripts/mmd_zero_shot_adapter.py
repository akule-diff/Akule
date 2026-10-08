"""Evaluation-only MMD geometry and official metrics for frozen SMD Akule.

No learned parameters, environment labels, or adherence conditioning are added.
Import this module before any other torch_robotics provider in a fresh process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MMD = ROOT / "external/mmd"
sys.path[:0] = [str(MMD), str(MMD / "deps/torch_robotics"),
                str(MMD / "deps/motion_planning_baselines"), str(ROOT)]

from torch_robotics.environments import EnvHighways2D, EnvConveyor2D, EnvDropRegion2D
from diffuser.models.smd_map_context import StaticMapContext

ENVIRONMENTS = {"highways": EnvHighways2D, "conveyor": EnvConveyor2D,
                "drop_region": EnvDropRegion2D}
POPULATIONS = (3, 6, 9, 12, 15, 20)


def official_environment(family, device="cpu"):
    return ENVIRONMENTS[family](tensor_args={"device": device, "dtype": torch.float32})


def official_geometry(env):
    items = []
    for obj in env.obj_fixed_list:
        for field in obj.fields:
            if type(field).__name__ == "MultiSphereField" and field.centers.numel() == 0:
                continue
            if type(field).__name__ not in ("MultiBoxField", "MultiRoundedBoxField"):
                raise ValueError(f"Unsupported official primitive: {type(field).__name__}")
            for index, (center, size) in enumerate(zip(field.centers, field.sizes)):
                items.append({"center": center.detach().cpu().tolist(),
                              "size": size.detach().cpu().tolist(),
                              "corner_radius": float(field.radius[index])
                              if type(field).__name__ == "MultiRoundedBoxField" else 0.})
    return {"kind": "boxes", "items": items}


class BoxMapContext(StaticMapContext):
    """Official rounded-box SDF minus disk radius, in unchanged context format."""

    def __init__(self, obstacles, *, device, scale=.2,
                 workspace=((-1., -1.), (1., 1.)), radius=.05):
        super().__init__({"kind": "circles", "items": []}, device=device,
                         scale=scale, workspace=workspace)
        if obstacles["kind"] != "boxes" or radius != .05:
            raise ValueError("This experiment requires exact boxes and radius .05")
        self.box_centers = torch.as_tensor([o["center"] for o in obstacles["items"]],
                                           device=device, dtype=torch.float32).reshape(-1, 2)
        self.half_sizes = torch.as_tensor([o["size"] for o in obstacles["items"]],
                                          device=device, dtype=torch.float32).reshape(-1, 2)/2
        self.radius = radius
        self.corner_radii = torch.as_tensor([o["corner_radius"] for o in obstacles["items"]],
                                            device=device, dtype=torch.float32)

    def clearance_gradient(self, positions):
        distance, gradient = super().clearance_gradient(positions)
        if not self.box_centers.numel():
            return distance, gradient
        delta = positions[..., None, :] - self.box_centers
        q = delta.abs() - self.half_sizes + self.corner_radii[:, None]
        outside = q.clamp_min(0)
        norm = torch.linalg.vector_norm(outside, dim=-1)
        inside, axis = q.max(dim=-1)
        sdf = norm + inside.clamp_max(0) - self.corner_radii - self.radius
        index = sdf.argmin(dim=-1)
        box_distance = sdf.gather(-1, index[..., None]).squeeze(-1)
        signs = torch.where(delta >= 0, torch.ones_like(delta), -torch.ones_like(delta))
        outside_gradient = outside/norm.clamp_min(1e-8)[..., None]*signs
        inside_gradient = torch.nn.functional.one_hot(axis, 2).to(positions.dtype)*signs
        gradients = torch.where((norm > 0)[..., None], outside_gradient, inside_gradient)
        box_gradient = gradients.gather(-2, index[..., None, None].expand(*index.shape, 1, 2)).squeeze(-2)
        use_box = box_distance < distance
        return (torch.where(use_box, box_distance, distance),
                torch.where(use_box[..., None], box_gradient, gradient))


class MMDSpatialMapField:
    def __init__(self, scene, *, device, clearance_scale=.2, size=64):
        self.exact = BoxMapContext(scene["obstacles"], device=device,
                                   scale=clearance_scale, workspace=scene["workspace"])
        axis = torch.linspace(-1., 1., size, device=device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        clearance, _ = self.exact.clearance_gradient(torch.stack((x, y), -1))
        self.raster = torch.stack(((clearance < 0).to(clearance.dtype),
                                   (clearance/clearance_scale).clamp(-4, 4)), 0)[None]
        self.obstacles = scene["obstacles"]

    def raw_context(self, positions):
        return self.exact(positions)

    def exact_clearance(self, positions):
        return self.exact.clearance_gradient(positions)[0]


def official_metrics(physical, scene, env=None):
    """Use returned supports verbatim; official D sees positions, A sees velocities.

    Final Akule channels encode per-step displacement. Convert those channels
    to native MMD m/s using the frozen 5/64 s interval, preserving positions.
    Official A then computes mean norm of adjacent velocity differences, with
    no further dt division. Never reconstruct velocities from positions.
    """
    from torch_robotics.trajectory.metrics import (compute_path_length_from_pos,
                                                   compute_average_acceleration_from_pos_vel)
    from torch_robotics.robots import RobotPlanarDisk
    from torch_robotics.tasks.tasks import PlanningTask
    path = torch.as_tensor(np.asarray(physical), dtype=torch.float32)
    if path.ndim != 3 or path.shape[1] != scene["population"] or path.shape[-1] != 4:
        raise ValueError("Expected returned trajectory [H,N,4]")
    if not torch.isfinite(path).all():
        raise ValueError("Nonfinite output")
    env = env or official_environment(scene["family"])
    robot = RobotPlanarDisk(radius=.05, tensor_args={"device": "cpu", "dtype": torch.float32})
    task = PlanningTask(env=env, robot=robot, tensor_args={"device": "cpu", "dtype": torch.float32})
    n = path.shape[1]
    i, j = torch.triu_indices(n, n, 1)
    pairs = int((torch.linalg.vector_norm(path[:, i, :2]-path[:, j, :2], dim=-1) < .1).sum())
    obstacle_invalid = task.compute_collision(path[..., :2])
    endpoints = (torch.max(torch.abs(path[0, :, :2]-torch.tensor(scene["starts"]))) < 1e-3 and
                 torch.max(torch.abs(path[-1, :, :2]-torch.tensor(scene["goals"]))) < 1e-3)
    # Match the single-local-map slice in official inference_multi_agent.py.
    from mmd.config.mmd_params import MMDParams
    data = [float(env.compute_traj_data_adherence(path[:MMDParams.horizon, a, :2])) for a in range(n)]
    lengths = [float(compute_path_length_from_pos(path[:, a, :2][None])) for a in range(n)]
    native_velocity = path[..., 2:] / (5./64.)
    acceleration = [float(compute_average_acceleration_from_pos_vel(
        path[:, a, :2][None], native_velocity[:, a][None])) for a in range(n)]
    return {"official_sampled_success": bool(endpoints and pairs == 0 and not obstacle_invalid.any()),
            "endpoints_valid": bool(endpoints), "pair_conflicts": pairs,
            "obstacle_invalid_samples": int(obstacle_invalid.sum()),
            "D": float(np.mean(data)), "D_per_agent": data,
            "path_length": float(np.mean(lengths)), "acceleration": float(np.mean(acceleration)),
            "metric_support_count": len(path), "interpolation": "none",
            "velocity_conversion": "Akule per-step displacement divided by frozen dt=5/64 to MMD m/s",
            "acceleration_evaluator": "official mean adjacent native-velocity difference; no additional dt division"}
