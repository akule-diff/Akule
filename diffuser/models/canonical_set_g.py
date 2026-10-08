"""Set-context residual composer used by the frozen N=28 benchmark."""

import torch
from torch import nn
from torch.nn import functional as F

from diffuser.models import quality_sparse_v1 as models


class SetContextG(models.SignedQualityG):
    """Shared edge G with explicit invariant incident and scene summaries."""

    def __init__(self, scales):
        super().__init__(scales, width=112)
        w = self.width
        self.node_token = nn.Sequential(nn.Linear(14, w), nn.SiLU(), nn.Linear(w, w))
        # Three message summaries, three residual summaries, admitted and
        # available degree, and population. All are computed per time block.
        self.set_token = nn.Sequential(nn.Linear(3 * w + 6, w), nn.SiLU())
        self.scene_token = nn.Sequential(nn.Linear(2 * w + 1, w), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(6 * w + 2, w), nn.SiLU(), nn.Linear(w, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

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
            features = models.pair_features(
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

        pieces = []
        for part, field in zip(index.split(chunk), residual.split(chunk)):
            if self.training and torch.is_grad_enabled() and chunk < len(index):
                from torch.utils.checkpoint import checkpoint

                pieces.append(checkpoint(encode, part, field, use_reentrant=False))
            else:
                pieces.append(encode(part, field))
        hidden = torch.cat(pieces)
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

        ids = bi * n + i
        weighted = hidden * mass
        total = hidden.new_zeros(b * n, self.blocks, self.width).index_add(
            0, ids, weighted
        )
        degree = hidden.new_zeros(b * n, 1, 1).index_add(0, ids, mass)
        mean = total / degree.clamp_min(1)
        maximum = hidden.new_zeros(b * n, self.blocks, self.width)
        maximum.scatter_reduce_(
            0,
            ids[:, None, None].expand_as(hidden),
            hidden.masked_fill(mass == 0, float("-inf")),
            reduce="amax",
            include_self=False,
        )
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        sqrt_sum = total / degree.clamp_min(1).sqrt()

        magnitude = torch.linalg.vector_norm(residual, dim=-1)
        magnitude = F.adaptive_avg_pool1d(magnitude[:, None], self.blocks).transpose(
            1, 2
        )
        magnitude = magnitude * a[:, None, None]
        mag_sum = hidden.new_zeros(b * n, self.blocks, 1).index_add(0, ids, magnitude)
        mag_max = hidden.new_zeros(b * n, self.blocks, 1)
        mag_max.scatter_reduce_(
            0,
            ids[:, None, None].expand_as(magnitude),
            magnitude,
            reduce="amax",
            include_self=True,
        )
        available = (n - 1) * torch.ones_like(degree)
        population = n * torch.ones_like(degree)
        set_features = torch.cat(
            (
                mean,
                maximum,
                sqrt_sum,
                mag_sum / degree.clamp_min(1),
                mag_max,
                mag_sum / degree.clamp_min(1).sqrt(),
                (degree / max(n - 1, 1)).expand(-1, self.blocks, -1),
                (available / max(n, 1)).expand(-1, self.blocks, -1),
                (population / 100).expand(-1, self.blocks, -1),
            ),
            -1,
        )
        set_token = self.set_token(set_features)

        node_base = base.permute(0, 2, 3, 1).reshape(b * n, 4, h)
        node_blocks = F.adaptive_avg_pool1d(node_base, self.blocks)
        node_blocks = node_blocks.transpose(1, 2).reshape(b, n, self.blocks, 4)
        node_ep = endpoints[:, :, None, :].expand(b, n, self.blocks, 8)
        t_feature = timestep[:, None, None, None].to(base) / 24
        t_feature = t_feature.expand(b, n, self.blocks, 1)
        block_feature = torch.linspace(
            0, 1, self.blocks, device=base.device, dtype=base.dtype
        )
        block_feature = block_feature[None, None, :, None].expand(b, n, self.blocks, 1)
        node = self.node_token(
            torch.cat((node_blocks, node_ep, t_feature, block_feature), -1)
        )
        node = node.reshape(b * n, self.blocks, self.width)
        node_group = node.reshape(b, n, self.blocks, self.width)
        scene = self.scene_token(
            torch.cat(
                (
                    node_group.mean(1),
                    node_group.max(1).values,
                    base.new_full((b, self.blocks, 1), n / 100),
                ),
                -1,
            )
        )
        edge_context = torch.cat(
            (
                hidden,
                node[bi * n + i],
                node[bi * n + j],
                set_token[bi * n + i],
                set_token[bi * n + j],
                scene[bi],
                t_feature[bi, i],
                block_feature[bi, i],
            ),
            -1,
        )
        # QualityMixer.compose applies the gate to each residual contribution.
        # Applying it here too would square a straight-through gate and give a
        # rejected edge zero proposal-loss gradient at a=0. Binary inference
        # outputs are unchanged by removing this redundant multiplication.
        return self.activate(self.head(edge_context).squeeze(-1), timestep[bi])
