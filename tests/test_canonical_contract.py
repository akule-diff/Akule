"""Checkpoint and metric contracts for the deployed planner."""

import json
import pytest

import torch

from diffuser.models.canonical_set_g import SetContextG
from diffuser.models.quality_dynamic_u_v2 import DynamicSupportU, full_support
from diffuser.models.quality_sparse_v1 import SignedQualityMixer
from diffuser.utils.quality_sparse_v1 import spline_matrix
from diffuser.utils.quality_sparse_v1 import metrics


@pytest.mark.model_assets
def test_locked_checkpoint_hashes():
    from scripts.canonical_n28_runtime import CHECKPOINTS, EXPECTED, sha

    assert {name: sha(CHECKPOINTS / name) for name in EXPECTED} == EXPECTED


def test_sampled_collision_boundary():
    positions = torch.zeros(1, 64, 2, 4)
    positions[:, :, 1, 0] = 0.0999
    starts = positions[0, 0, :, :2]
    goals = positions[0, -1, :, :2]
    assert metrics(positions, starts, goals)["collision_pair_times"] == 64
    positions[:, :, 1, 0] = 0.1001
    assert metrics(positions, positions[0, 0, :, :2], positions[0, -1, :, :2])["collision_pair_times"] == 0


def test_u_and_g_permutation_for_variable_populations():
    scales = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1]
                         / "artifacts/manifests/SIGNED_SCALES.json").read_text())["scales"]
    torch.manual_seed(28)
    u = DynamicSupportU(width=48, pair_residual=True, pair_residual_width=192).eval()
    g = SetContextG(scales).eval()
    for n in (3, 6):
        base = torch.randn(1, 64, n, 4)
        endpoints = torch.randn(1, n, 8)
        t = torch.tensor([12])
        order = torch.randperm(n)
        inverse = torch.argsort(order)
        with torch.no_grad():
            a = u.logits(base, endpoints, t)
            b = u.logits(base[:, :, order], endpoints[:, order], t)
            assert torch.allclose(a, b[:, inverse][:, :, inverse], atol=3e-5)

            def weights(state, ends):
                index = full_support(state).nonzero()
                _, i, j = index.unbind(-1)
                residual = (state[0, :, i, :2] - state[0, :, j, :2]).transpose(0, 1)
                gates = full_support(state).to(state)
                values = g(state, ends, t, index, residual, gates)
                matrix = state.new_zeros(n, n, 8)
                matrix[i, j] = values
                return matrix

            original = weights(base, endpoints)
            permuted = weights(base[:, :, order], endpoints[:, order])
            assert torch.allclose(original, permuted[inverse][:, inverse], atol=3e-5)


def test_empty_support_preserves_state_and_degree_contract():
    scales = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1]
                         / "artifacts/manifests/SIGNED_SCALES.json").read_text())["scales"]
    n = 4
    base = torch.randn(1, 64, n, 4)
    endpoints = torch.randn(1, n, 8)
    timestep = torch.tensor([7])
    mixer = SignedQualityMixer(spline_matrix(8), scales)
    mixer.g = SetContextG(scales)
    support = torch.zeros(1, n, n)
    output, coefficients = mixer.compose(base, endpoints, timestep,
                                         torch.empty(0, 3, dtype=torch.long),
                                         torch.empty(0, 64, 2), support)
    assert torch.equal(output, base)
    assert coefficients.shape == (0, 8)
    assert full_support(base).sum().item() == n * (n - 1)
