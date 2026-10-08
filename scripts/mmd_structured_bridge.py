"""Minimal pinned-MMD Highways/Conveyor model and obstacle bridge.

The canonical N=28 runtime remains untouched. These utilities keep the
environment-specific base model and collision geometry explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MMD = ROOT / "external/mmd"
MODELS = {
    "nowait": ("EnvEmptyNoWait2D-RobotPlanarDisk", "ed1180b1136d7bfbce1b1c08097d868fbf737df8f2b14bb521cebed1ee0b3444"),
    "highways": ("EnvHighways2D-RobotPlanarDisk", "ec2f16d9b173ed77644b3ceeb8a9e1a788eec697fb7c1720d645e6549bcb02fe"),
    "conveyor": ("EnvConveyor2D-RobotPlanarDisk", "43266a63347ab435f00ab0c2e6d05de513f9ecbe01e745cd146e8086884fa07a"),
}


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def official_boxes(environment):
    """Centers and *full* sizes from pinned EnvHighways2D/EnvConveyor2D."""
    if environment == "conveyor":
        centers = [[0, 0], [0, .35], [0, -.35]]
        sizes = [[.8, .1], [1., .1], [1., .1]]
    elif environment == "highways":
        centers = [[0, 0], [0, .875], [0, -.875], [.875, 0], [-.875, 0],
                   [.875, .875], [.875, -.875], [-.875, .875], [-.875, -.875]]
        sizes = [[.5, .5], [.5, .25], [.5, .25], [.25, .5], [.25, .5],
                 [.25, .25], [.25, .25], [.25, .25], [.25, .25]]
    else:
        raise ValueError(environment)
    return np.asarray(centers, dtype=np.float64), np.asarray(sizes, dtype=np.float64)


def point_box_clearance(point, center, full_size):
    outside = np.maximum(np.abs(np.asarray(point) - center) - full_size / 2, 0)
    return float(np.linalg.norm(outside))


def segment_box_clearance(start, end, center, full_size):
    """Exact minimum Euclidean center distance from a segment to an AABB."""
    start = np.asarray(start, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - start
    lower, upper = center - full_size / 2, center + full_size / 2
    cuts = [0., 1.]
    for axis in range(2):
        if abs(delta[axis]) > 1e-15:
            cuts.extend(t for t in ((lower[axis] - start[axis]) / delta[axis],
                                    (upper[axis] - start[axis]) / delta[axis])
                        if 0 < t < 1)
    cuts = sorted(set(cuts))
    best = float("inf")
    for left, right in zip(cuts[:-1], cuts[1:]):
        midpoint = (left + right) / 2
        position = start + midpoint * delta
        # Within each interval, the vector to the box is affine in t.
        mask = np.where(position < lower, -1, np.where(position > upper, 1, 0))
        boundary = np.where(mask < 0, lower, upper)
        slope = delta * (mask != 0)
        intercept = (start - boundary) * (mask != 0)
        denominator = float(np.dot(slope, slope))
        stationary = -float(np.dot(intercept, slope)) / denominator if denominator else left
        for t in (left, right, float(np.clip(stationary, left, right))):
            best = min(best, point_box_clearance(start + t * delta, center, full_size))
    return best


def obstacle_clearance(trajectory, environment, radius=.05):
    positions = np.asarray(trajectory, dtype=np.float64)[..., :2]
    if positions.ndim == 4:
        if positions.shape[0] != 1:
            raise ValueError("Expected one scene")
        positions = positions[0]
    if positions.ndim != 3:
        raise ValueError("Expected [T,N,2 or 4]")
    centers, sizes = official_boxes(environment)
    sampled = min(point_box_clearance(point, center, size) - radius
                  for frame in positions for point in frame
                  for center, size in zip(centers, sizes))
    swept = min(segment_box_clearance(p, q, center, size) - radius
                for first, second in zip(positions[:-1], positions[1:])
                for p, q in zip(first, second)
                for center, size in zip(centers, sizes))
    return dict(sampled_obstacle_clearance_m=float(sampled),
                swept_obstacle_clearance_m=float(swept),
                obstacle_collision=bool(swept < 0))


def limits_from_official_trajectories(environment):
    """Exact extrema exported from the frozen upstream trajectory corpus."""
    model_id, _ = MODELS[environment]
    return json.loads((ROOT / "configs/normalizers" / (model_id + ".json")).read_text())


def load_base(environment, asset_root, device="cpu"):
    model_id, expected = MODELS[environment]
    model_dir = Path(asset_root).resolve() / model_id
    args = yaml.safe_load((model_dir / "args.yaml").read_text())
    if args["dataset_subdir"] != model_id or args["n_diffusion_steps"] != 25 or not args["predict_epsilon"]:
        raise RuntimeError("Official model metadata disagrees with expected MMD contract")
    checkpoint = model_dir / "checkpoints/ema_model_current_state_dict.pth"
    if file_sha(checkpoint) != expected:
        raise RuntimeError("Official MMD checkpoint hash mismatch")
    os.chdir(MMD)
    sys.path[:0] = [str(MMD), str(MMD / "deps/torch_robotics"),
                    str(MMD / "deps/motion_planning_baselines"),
                    str(MMD / "deps/experiment_launcher")]
    from mmd.models import TemporalUnet, UNET_DIM_MULTS
    from mmd.trainer import get_model
    model = get_model(model_class=args["diffusion_model_class"],
                      model=TemporalUnet(state_dim=4, n_support_points=64,
                                         unet_input_dim=args["unet_input_dim"],
                                         dim_mults=UNET_DIM_MULTS[args["unet_dim_mults_option"]]),
                      tensor_args={"device": torch.device(device), "dtype": torch.float32},
                      variance_schedule=args["variance_schedule"], n_diffusion_steps=25,
                      predict_epsilon=True, state_dim=4, n_support_points=64)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval().requires_grad_(False)
    return model, args, expected


class NativeVelocityCodec:
    """Expose pinned unary channels as canonical per-step displacement."""

    def __init__(self, limits, device, *, native_velocity_is_mps):
        from diffuser.utils.quality_sparse_v1 import DT
        self.lo = torch.tensor(limits["minimum"], device=device, dtype=torch.float32)
        self.hi = torch.tensor(limits["maximum"], device=device, dtype=torch.float32)
        self.scale = (self.hi - self.lo) / 2
        self.mid = (self.hi + self.lo) / 2
        self.velocity_scale = DT if native_velocity_is_mps else 1.0

    def decode(self, normalized):
        native = normalized * self.scale + self.mid
        return torch.cat((native[..., :2], native[..., 2:] * self.velocity_scale), dim=-1)

    def encode(self, physical):
        native = torch.cat((physical[..., :2], physical[..., 2:] / self.velocity_scale), dim=-1)
        return (native - self.mid) / self.scale


def native_scene(environment, n, ordinal):
    """One deterministic scene from the pinned official MMD provider on CPU."""
    sys.path[:0] = [str(MMD), str(MMD / "deps/torch_robotics")]
    from mmd.config.mmd_params import MMDParams
    from mmd.config.mmd_experiment_configs import (
        EnvConveyor2DRobotPlanarDiskRandom, EnvHighways2DRobotPlanarDiskRandom,
    )
    MMDParams.tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    providers = {"highways": EnvHighways2DRobotPlanarDiskRandom,
                 "conveyor": EnvConveyor2DRobotPlanarDiskRandom}
    seed = 29_000_000 + 1000 * n + ordinal
    torch.manual_seed(seed)
    np.random.seed(seed)
    starts, goals, model_ids, _ = providers[environment]().get_planning_problem(n)
    model_id, digest = MODELS[environment]
    if model_ids != [[model_id]]:
        raise RuntimeError("Native provider/model identity mismatch")
    record = dict(family=f"MMD-{environment}-Random", population=n,
                  starts=[row.cpu().tolist() for row in starts],
                  goals=[row.cpu().tolist() for row in goals], radii=[.05] * n,
                  obstacles=dict(kind="boxes", full_sizes=True,
                                 centers=official_boxes(environment)[0].tolist(),
                                 sizes=official_boxes(environment)[1].tolist()),
                  model_id=model_id, base_checkpoint_sha256=digest,
                  construction=dict(provider="pinned native MMD", seed=seed),
                  configuration_id=2_000_000 + ordinal,
                  noise_seed=19_000_000 + 4 * ordinal)
    record["input_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


class StructuredEngine:
    """Frozen canonical coordination around matching official structured MPD."""

    def __init__(self, environment, asset_root):
        from canonical_n28_runtime import Engine
        from diffuser.models.mpd_v2 import MPDUnaryAdapter

        self.engine = Engine()
        model, _, digest = load_base(environment, asset_root, device="cuda")
        self.engine.unary = MPDUnaryAdapter(model)
        self.engine.normalizer = limits_from_official_trajectories(environment)
        # Pinned trajectory data encode EmptyNoWait channels as exact per-step
        # displacement; Highways and Conveyor encode m/s velocity.
        native_velocity_is_mps = environment in ("highways", "conveyor")
        self.engine.codec = NativeVelocityCodec(
            self.engine.normalizer, self.engine.device,
            native_velocity_is_mps=native_velocity_is_mps,
        )
        self.engine.native_velocity_bridge = native_velocity_is_mps
        self.environment = environment
        self.base_checkpoint_sha256 = digest

    def __getattr__(self, name):
        return getattr(self.engine, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=MODELS)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    limits = limits_from_official_trajectories(args.environment)
    result = dict(environment=args.environment, normalizer=limits,
                  obstacles=(0 if args.environment == "nowait" else len(official_boxes(args.environment)[0])))
    if args.check_model:
        _, metadata, digest = load_base(args.environment, args.asset_root)
        result.update(checkpoint_sha256=digest, unet_dim_mults_option=metadata["unet_dim_mults_option"])
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
