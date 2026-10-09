"""Pinned unary data distinguishes displacement from physical velocity."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mmd_structured_bridge import NativeVelocityCodec
from canonical_mmd_repair import Native


def test_codec_respects_pinned_empty_vs_structured_channel_units():
    limits = {"minimum": [-1., -1., -1., -1.],
              "maximum": [1., 1., 1., 1.]}
    normalized = torch.tensor([[[.2, .3, .4, .5]]])
    empty = NativeVelocityCodec(limits, "cpu", native_velocity_is_mps=False)
    structured = NativeVelocityCodec(limits, "cpu", native_velocity_is_mps=True)
    torch.testing.assert_close(empty.decode(normalized)[..., 2:], normalized[..., 2:])
    torch.testing.assert_close(structured.decode(normalized)[..., 2:],
                               normalized[..., 2:] * (5 / 64))
    torch.testing.assert_close(structured.encode(structured.decode(normalized)), normalized)


@pytest.mark.model_assets
def test_units_match_downloaded_official_trajectory_data():
    root = ROOT / "external/mmd/data_trajectories"
    for model, velocity_is_mps in (
        ("EnvEmptyNoWait2D-RobotPlanarDisk", False),
        ("EnvHighways2D-RobotPlanarDisk", True),
        ("EnvConveyor2D-RobotPlanarDisk", True),
    ):
        path = next((root / model).glob("*/trajs-free.pt"))
        values = torch.load(path, map_location="cpu", weights_only=False)
        delta = values[:, 1:, :2] - values[:, :-1, :2]
        stored = values[:, :-1, 2:]
        if velocity_is_mps:
            assert (delta - stored * (5 / 64)).square().mean().sqrt() < .01
            assert (delta - stored).square().mean().sqrt() > .01
        else:
            torch.testing.assert_close(delta, stored)


def test_highways_root_converts_to_and_from_native_mmd_state():
    repair = object.__new__(Native)
    repair.engine = SimpleNamespace(native_velocity_bridge=True)
    canonical = torch.tensor([[.2, .3, .02, -.01]])
    native = repair.to_native_state(canonical)
    torch.testing.assert_close(native[..., 2:], canonical[..., 2:] / (5 / 64))
    torch.testing.assert_close(repair.to_physical_state(native), canonical)
    torch.testing.assert_close(canonical, torch.tensor([[.2, .3, .02, -.01]]))
