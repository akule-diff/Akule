"""Construct the pinned official MPD model without loading its training corpus."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from diffuser.models.mpd_v2 import MPDUnaryAdapter

ROOT = Path(__file__).resolve().parents[1]


class FrozenNormalizer:
    def __init__(self, path):
        value = json.loads(Path(path).read_text())
        self._mins_cpu = torch.tensor(value["minimum"], dtype=torch.float32)
        self._maxs_cpu = torch.tensor(value["maximum"], dtype=torch.float32)


def load_official_mpd(mmd_root: Path, device: str):
    root = Path(mmd_root).resolve()
    if not (root / "mmd/models").exists():
        raise FileNotFoundError(f"Clone pinned MMD at {root}; see DELTAAI_SETUP.md")
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "deps/torch_robotics"),
                    str(root / "deps/motion_planning_baselines"), str(root / "deps/experiment_launcher")]
    from mmd.models import TemporalUnet, UNET_DIM_MULTS
    from mmd.trainer import get_model

    d = torch.device(device)
    model = get_model(
        model_class="GaussianDiffusionModel",
        model=TemporalUnet(state_dim=4, n_support_points=64, unet_input_dim=32,
                           dim_mults=UNET_DIM_MULTS[1]),
        tensor_args={"device": d, "dtype": torch.float32},
        variance_schedule="exponential", n_diffusion_steps=25,
        predict_epsilon=True, state_dim=4, n_support_points=64,
    )
    checkpoint = ROOT / "checkpoints/MPD.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=d, weights_only=False))
    model.eval()
    return MPDUnaryAdapter(model), FrozenNormalizer(ROOT / "artifacts/manifests/MPD_NORMALIZER.json"), None, root
