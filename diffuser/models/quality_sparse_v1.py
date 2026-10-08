"""Learned smooth pair residual and temporal/relational U/G, version 1.

The only trajectory-valued learned output belongs to the pair residual. U is
pre-residual; G produces eight scalar coefficients per admitted pair.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TemporalStack(nn.Module):
    def __init__(self, inputs, width=64):
        super().__init__()
        self.entry = nn.Conv1d(inputs, width, 1)
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(width, width, 5, padding=2 * d, dilation=d)
                for d in (1, 2, 4, 8)
            ]
        )

    def forward(self, x):
        x = F.silu(self.entry(x.transpose(1, 2)))
        for layer in self.layers:
            x = x + F.silu(layer(x))
        return x


def pair_features(ego, partner, ee, pe, timestep):
    e, h, _ = ego.shape
    reverse = timestep.to(ego)[:, None, None].expand(e, h, 1) / 24
    physical = torch.linspace(0, 1, h, device=ego.device, dtype=ego.dtype)[
        None, :, None
    ].expand(e, h, 1)
    rel = partner[..., :2] - ego[..., :2]
    distance = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
    return torch.cat(
        (
            ego,
            partner,
            ee[:, None].expand(e, h, 8),
            pe[:, None].expand(e, h, 8),
            rel,
            distance,
            reverse,
            physical,
        ),
        -1,
    )


class SmoothPairResidual(nn.Module):
    def __init__(self, basis, width=64):
        super().__init__()
        self.register_buffer("basis", basis)
        self.temporal = TemporalStack(29, width)
        self.head = nn.Conv1d(width, 2, 1)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, ego, partner, ee, pe, timestep):
        hidden = self.temporal(pair_features(ego, partner, ee, pe, timestep))
        controls = self.head(
            F.adaptive_avg_pool1d(hidden, self.basis.shape[1])
        ).transpose(1, 2)
        return torch.einsum("hk,ekd->ehd", self.basis, controls)

    def specific(self, ego, partner, ee, pe, timestep):
        real = self(ego, partner, ee, pe, timestep)
        null = self(ego, torch.zeros_like(partner), ee, torch.zeros_like(pe), timestep)
        return real - null


class QualitySupportU(nn.Module):
    def __init__(self, width=32, logit_bound=None, normalize_token=False):
        super().__init__()
        self.logit_bound = logit_bound
        self.normalize_token = normalize_token
        self.temporal = TemporalStack(29, width)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.constant_(self.head[-1].bias, 3.0)

    def forward(self, base, endpoints, timestep, stochastic=False, force_open=False):
        b, h, n, _ = base.shape
        idx = (
            (~torch.eye(n, device=base.device, dtype=torch.bool))[None]
            .expand(b, n, n)
            .nonzero()
        )
        bi, i, j = idx.unbind(-1)
        if force_open:
            a = base.new_ones(len(idx))
            probability = a
        else:
            features = pair_features(
                base[bi, :, i],
                base[bi, :, j],
                endpoints[bi, i],
                endpoints[bi, j],
                timestep[bi],
            )
            token = self.temporal(features).mean(-1)
            if self.normalize_token:
                token = F.layer_norm(token, (token.shape[-1],))
            alpha = self.head(token).squeeze(-1)
            if self.logit_bound is not None:
                alpha = self.logit_bound * torch.tanh(alpha / self.logit_bound)
            if stochastic:
                uniform = torch.rand_like(alpha).clamp(1e-6, 1 - 1e-6)
                s = torch.sigmoid(
                    (alpha + uniform.log() - torch.log1p(-uniform)) / (2 / 3)
                )
            else:
                s = torch.sigmoid(alpha)
            a = (1.2 * s - 0.1).clamp(0, 1)
            probability = torch.sigmoid(alpha - (2 / 3) * math.log(0.1 / 1.1))
        gates = base.new_zeros(b, n, n)
        probs = gates.clone()
        gates[bi, i, j], probs[bi, i, j] = a, probability
        return gates, probs


class QualityG(nn.Module):
    def __init__(self, width=64, blocks=8):
        super().__init__()
        self.blocks, self.width = blocks, width
        self.temporal = TemporalStack(32, width)
        self.relations = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, width)
                )
                for _ in range(2)
            ]
        )
        self.head = nn.Linear(width, 1)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, -1.0)

    def forward(self, base, endpoints, timestep, index, residual, gates):
        b, h, n, _ = base.shape
        if not len(index):
            return base.new_zeros((0, self.blocks))
        bi, i, j = index.unbind(-1)
        a = gates[bi, i, j]
        chunk = getattr(self, "pair_chunk_size", 0) or len(index)

        def encode(part, field):
            pb, pi, pj = part.unbind(-1)
            pa = gates[pb, pi, pj]
            features = pair_features(
                base[pb, :, pi],
                base[pb, :, pj],
                endpoints[pb, pi],
                endpoints[pb, pj],
                timestep[pb],
            )
            features = torch.cat(
                (features, field / 0.1, pa[:, None, None].expand(-1, h, 1)), -1
            )
            return F.adaptive_avg_pool1d(
                self.temporal(features), self.blocks
            ).transpose(1, 2)

        hidden_parts = []
        for part, field in zip(index.split(chunk), residual.split(chunk)):
            if self.training and torch.is_grad_enabled() and chunk < len(index):
                from torch.utils.checkpoint import checkpoint

                hidden_parts.append(
                    checkpoint(encode, part, field, use_reentrant=False)
                )
            else:
                hidden_parts.append(encode(part, field))
        hidden = torch.cat(hidden_parts)
        mass = a[:, None, None]
        for layer in self.relations:
            totals = hidden.new_zeros(b * n, self.blocks, self.width)
            counts = hidden.new_zeros(b * n, 1, 1)
            for ids in (bi * n + i, bi * n + j):
                totals = totals.index_add(0, ids, hidden * mass)
                counts = counts.index_add(0, ids, mass)
            context = totals / counts.clamp_min(1)
            hidden = hidden + layer(
                torch.cat((hidden, context[bi * n + i], context[bi * n + j]), -1)
            )
        return (
            self.activate(self.head(hidden).squeeze(-1), timestep[bi])
            * (a > 0)[:, None]
        )

    def activate(self, logits, timestep):
        return torch.sigmoid(logits)


class SignedQualityG(QualityG):
    """The same temporal/relational G with frozen timestep-scaled signed output."""

    def __init__(self, timestep_scales, width=64, blocks=8):
        super().__init__(width=width, blocks=blocks)
        scales = torch.as_tensor(timestep_scales, dtype=torch.float32)
        if (
            scales.shape != (25,)
            or not torch.isfinite(scales).all()
            or not (scales > 0).all()
        ):
            raise ValueError("Expected 25 finite positive signed coefficient scales")
        self.register_buffer("timestep_scales", scales.clone())
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def activate(self, logits, timestep):
        return self.timestep_scales[timestep.long(), None] * torch.tanh(logits)


class QualityMixer(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.u = QualitySupportU()
        self.g = QualityG()
        self.register_buffer("weight_basis", basis)

    def compose(self, base, endpoints, timestep, index, residual, gates):
        b, h, n, _ = base.shape
        weights = self.g(base, endpoints, timestep, index, residual, gates)
        bi, i, j = index.unbind(-1)
        a = gates[bi, i, j]
        time_weights = torch.einsum("ek,hk->eh", weights, self.weight_basis)
        delta = residual * (a[:, None] * time_weights)[..., None]
        summed = base.new_zeros(b * n, h, 2).index_add(0, bi * n + i, delta)
        position_delta = summed.reshape(b, n, h, 2).permute(0, 2, 1, 3)
        velocity_delta = torch.cat(
            (
                position_delta[:, 1:] - position_delta[:, :-1],
                torch.zeros_like(position_delta[:, :1]),
            ),
            1,
        )
        mask = torch.ones_like(velocity_delta[..., :1])
        mask[:, 0] = 0
        return base + torch.cat((position_delta, velocity_delta * mask), -1), weights


class SignedQualityMixer(QualityMixer):
    def __init__(self, basis, timestep_scales):
        super().__init__(basis)
        self.g = SignedQualityG(timestep_scales)
