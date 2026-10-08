"""MPD-v2 view of preserved official Boundary-family scene files."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Kept local so this reader can run inside the isolated official-MMD venv.
# It is the frozen MPD state schema, not an independently chosen representation.
MPD_STATE_DIM = 4


def _load_payload(path: Path):
    # MMD scenes were serialised under NumPy 2; preserve their numbers exactly.
    sys.modules.setdefault("numpy._core", np.core)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trajectory = payload["trajectory"].float()  # official [N,H,4], not reconstructed
    if (
        trajectory.ndim != 3
        or trajectory.shape[-1] != MPD_STATE_DIM
        or not torch.isfinite(trajectory).all()
    ):
        raise ValueError("expected official Boundary trajectory [N,H,4]")
    return payload, trajectory


class MPDNormalizerAdapter:
    """Grouped view of the official MPD dataset normalizer; owns no statistics."""

    def __init__(self, official_mpd_dataset):
        self.official_mpd_dataset = official_mpd_dataset
        if getattr(official_mpd_dataset, "state_dim", None) != MPD_STATE_DIM:
            raise ValueError(
                "normalizer must come from the official four-state MPD dataset"
            )
        field = official_mpd_dataset.normalizer.normalizers[
            official_mpd_dataset.field_key_traj
        ]
        if not hasattr(field, "mins") or not hasattr(field, "maxs"):
            raise ValueError(
                "official MPD trajectory normalizer must expose min/max tensors"
            )
        self._mins_cpu = field.mins.detach().cpu().clone()
        self._maxs_cpu = field.maxs.detach().cpu().clone()
        self._affine_cache = {}

    def _affine(self, value: torch.Tensor):
        key = (value.device.type, value.device.index, value.dtype)
        if key not in self._affine_cache:
            self._affine_cache[key] = (
                self._mins_cpu.to(device=value.device, dtype=value.dtype),
                self._maxs_cpu.to(device=value.device, dtype=value.dtype),
            )
        return self._affine_cache[key]

    def metadata(self):
        """Stable affine identity for reusable teacher-cache compatibility."""
        digest = hashlib.sha256()
        digest.update(self._mins_cpu.numpy().tobytes())
        digest.update(self._maxs_cpu.numpy().tobytes())
        return {"kind": "official_limits_affine_v1", "sha256": digest.hexdigest()}

    def normalize(self, grouped: torch.Tensor) -> torch.Tensor:
        if grouped.ndim != 4 or grouped.shape[-1] != MPD_STATE_DIM:
            raise ValueError("expected physical grouped state [B,H,N,4]")
        b, h, n, _ = grouped.shape
        flat = grouped.permute(0, 2, 1, 3).reshape(b * n, h, MPD_STATE_DIM)
        mins, maxs = self._affine(flat)
        normalized = 2.0 * (flat - mins) / (maxs - mins) - 1.0
        return normalized.reshape(b, n, h, MPD_STATE_DIM).permute(0, 2, 1, 3)

    def unnormalize(self, grouped: torch.Tensor) -> torch.Tensor:
        if grouped.ndim != 4 or grouped.shape[-1] != MPD_STATE_DIM:
            raise ValueError("expected normalized grouped state [B,H,N,4]")
        b, h, n, _ = grouped.shape
        flat = grouped.permute(0, 2, 1, 3).reshape(b * n, h, MPD_STATE_DIM)
        mins, maxs = self._affine(flat)
        # This is exactly MMD LimitsNormalizer.unnormalize.  Its conditional
        # clip is equivalent to an unconditional elementwise clamp, while the
        # latter avoids a CUDA scalar-read synchronization and keeps the
        # official normalizer's CPU statistics immutable.
        physical = ((flat.clamp(-1.0, 1.0) + 1.0) / 2.0) * (maxs - mins) + mins
        return physical.reshape(b, n, h, MPD_STATE_DIM).permute(0, 2, 1, 3)


class MPDBoundaryFamilyV2Dataset(Dataset):
    """Returns one scene as ``trajectory [H,N,4]`` and endpoints ``[N,4]``.

    Batch collation gives the v2 grouped representation ``[B,H,N,4]``.  The
    velocity channels are read directly from the official MMD output; no finite
    differencing or new normalization is performed here.
    """

    def __init__(self, root: Path, split: str):
        root = Path(root)
        records = [
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text().splitlines()
            if line
        ]
        self.records = [r for r in records if r["split"] == split and r.get("success")]
        if not self.records:
            raise ValueError(f"no successful {split} scenes")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        payload, trajectory = _load_payload(Path(record["trajectory_path"]))
        starts = torch.as_tensor(payload["starts"], dtype=torch.float32)
        goals = torch.as_tensor(payload["goals"], dtype=torch.float32)
        if starts.shape != goals.shape or starts.ndim != 2 or starts.shape[-1] != 2:
            raise ValueError("Boundary endpoints must be [N,2]")
        if trajectory.shape[0] != starts.shape[0]:
            raise ValueError("trajectory/endpoints agent count mismatch")
        return {
            "trajectory": trajectory.transpose(0, 1),  # [H,N,4]
            "context": torch.cat((starts, goals), dim=-1),  # [N,4], endpoint metadata
            "scene_hash": record["scene_hash"],
            "seed": record["seed"],
        }
