"""Deterministic, uncapped contextual pre-residual interaction selection."""
from __future__ import annotations

import torch
from torch import nn


def full_support(base):
    b, _, n, _ = base.shape
    return (~torch.eye(n, dtype=torch.bool, device=base.device))[None].expand(b, n, n)


class PairLocalSupportU(nn.Module):
    """Legacy shared pair scorer, retained only for checkpoint evaluation.

    Inputs are unclipped physical clean estimates: position [m] and stored
    displacement [m/sample], NOT raw noisy states or physical velocity.

    Although this module has per-agent geometry summaries, an edge token is
    scored independently of the *set of competing edge tokens*.  It is not
    the architecture used for fresh sparse-support training.
    """

    def __init__(
        self,
        width=48,
        blocks=8,
        dt=5.0 / 64,
        distance=0.105,
        include_agent_indices=False,
    ):
        super().__init__()
        self.config = dict(
            width=width,
            blocks=blocks,
            dt=dt,
            distance=distance,
            include_agent_indices=include_agent_indices,
        )
        self.dt, self.distance, self.blocks = dt, distance, blocks
        self.include_agent_indices = include_agent_indices
        # The pair token deliberately contains no array-slot identity.  ``i`` and
        # ``j`` below are only used to gather/scatter physical-agent tensors.
        # Including i/(N-1), j/(N-1) here made an otherwise shared scorer depend
        # on arbitrary input ordering.
        self.point = nn.Sequential(
            nn.Linear(28 if include_agent_indices else 26, 32), nn.SiLU()
        )
        self.temporal = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2), nn.SiLU())
        self.block = nn.Sequential(nn.Linear(64 + 24, width), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(blocks * width, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def scene_context(self, base):
        p = base[..., :2]
        relative = (p[:, :, None] - p[:, :, :, None]) / self.distance
        distance = torch.linalg.vector_norm(relative, dim=-1)
        n = p.shape[2]
        off = ~torch.eye(n, dtype=torch.bool, device=p.device)
        proximity = off / (1 + distance.square())
        nearest = distance.masked_fill(~off, float("inf")).amin(-1)
        direction = (relative * proximity[..., None]).sum(-2) / proximity.sum(
            -1
        ).clamp_min(1e-8)[..., None]
        return torch.cat(
            (
                nearest[..., None],
                proximity.sum(-1)[..., None],
                proximity.amax(-1)[..., None],
                direction,
            ),
            -1,
        )

    def logits(self, base, endpoints, timestep):
        b, h, n, _ = base.shape
        index = full_support(base).nonzero()
        context = self.scene_context(base)
        chunk = getattr(self, "pair_chunk_size", 0) or len(index)
        values = torch.cat(
            [
                self.pair_logits(base, endpoints, timestep, part, context)
                for part in index.split(chunk)
            ]
        )
        return base.new_full((b, n, n), -float("inf")).index_put(tuple(index.T), values)

    def pair_logits(self, base, endpoints, timestep, index, context):
        """Same pair encoder, with shared all-agent context computed once."""
        b, h, n, _ = base.shape
        bi, i, j = index.unbind(-1)
        scale = base.new_tensor([0.95, 0.95, self.dt * 0.5, self.dt * 0.5])
        ego, partner = base[bi, :, i], base[bi, :, j]
        relp = (partner[..., :2] - ego[..., :2]) / self.distance
        relv = (partner[..., 2:] - ego[..., 2:]) / (self.dt * 0.5)
        gap = torch.linalg.vector_norm(relp, dim=-1, keepdim=True) - 1
        approach = (relp * relv).sum(-1, keepdim=True)
        time = torch.linspace(-1, 1, h, device=base.device)[None, :, None].expand(
            len(index), h, 1
        )
        reverse = timestep[bi, None, None].expand(-1, h, 1) / 24
        # Compatibility-only path for frozen historical checkpoints.  New
        # learned U instances leave this disabled and are slot-equivariant.
        identity = (
            torch.stack((i, j), -1).to(base)[:, None].expand(-1, h, -1) / max(n - 1, 1)
            if self.include_agent_indices
            else None
        )
        point = torch.cat(
            (
                ego / scale,
                partner / scale,
                relp,
                relv,
                gap,
                approach,
                context[bi, :, i],
                context[bi, :, j],
                *((identity,) if identity is not None else ()),
                time,
                reverse,
            ),
            -1,
        )
        hidden = self.temporal(self.point(torch.asinh(point)).transpose(1, 2))
        hidden = hidden.reshape(len(index), 32, self.blocks, h // self.blocks)
        pooled = torch.cat((hidden.mean(-1), hidden.amax(-1)), 1).transpose(1, 2)
        es = scale.repeat(2)
        ep = torch.asinh(
            torch.cat(
                (
                    endpoints[bi, i] / es,
                    endpoints[bi, j] / es,
                    (endpoints[bi, j] - endpoints[bi, i]) / es,
                ),
                -1,
            )
        )
        token = self.block(
            torch.cat((pooled, ep[:, None].expand(-1, self.blocks, -1)), -1)
        )
        values = self.head(token.flatten(1)).squeeze(-1)
        return values

    def forward(self, base, endpoints, timestep, mode="dynamic"):
        off = full_support(base)
        if mode == "open":
            return off, base.new_full(off.shape, float("inf")).masked_fill(
                ~off, -float("inf")
            )
        logits = self.logits(base, endpoints, timestep)
        if mode == "dynamic":
            return (logits > 0) & off, logits
        if mode == "top6":
            mask = torch.zeros_like(off).scatter(
                -1, logits.topk(min(6, base.shape[2] - 1), -1).indices, True
            )
            return mask & off, logits
        raise ValueError(mode)


class DynamicSupportU(nn.Module):
    """Contextual, shared, permutation-equivariant directed support scorer.

    ``z_ij`` is not a pair-local decision.  A shared temporal pair encoder
    first constructs ``h_ij`` for every directed candidate, then the scorer
    pools the outgoing edge set of ``i``, incoming edge set of ``j``, and the
    complete directed-edge set.  A shared edge head consumes
    ``(h_ij, c_i_out, c_j_in, c_global)``.  All pools are masked means, so the
    architecture is variable-N and changes under a permutation only by the
    same permutation of the output matrix.

    Old pair-local checkpoints are accepted in compatibility-only mode via
    ``PairLocalSupportU``.  Fresh instances always use the contextual path.
    """

    def __init__(
        self,
        width=48,
        blocks=8,
        dt=5.0 / 64,
        distance=0.105,
        include_agent_indices=False,
        contextual=True,
        inducing=4,
        pair_residual=False,
        pair_residual_width=None,
        per_agent_density=False,
    ):
        super().__init__()
        if include_agent_indices:
            raise ValueError(
                "Fresh DynamicSupportU forbids arbitrary agent-index features; "
                "legacy checkpoints are loaded through compatibility mode."
            )
        self.config = dict(
            width=width,
            blocks=blocks,
            dt=dt,
            distance=distance,
            include_agent_indices=False,
            contextual=True,
            support_context=True,
            inducing=inducing,
            pair_residual=bool(pair_residual),
            pair_residual_width=(None if pair_residual_width is None else int(pair_residual_width)),
            per_agent_density=bool(per_agent_density),
        )
        self.dt, self.distance, self.blocks, self.width = dt, distance, blocks, width
        self.inducing = int(inducing)
        self.pair_residual = bool(pair_residual)
        self.pair_residual_width = int(pair_residual_width or width)
        self.per_agent_density = bool(per_agent_density)
        self._legacy = None
        # Same physically meaningful per-edge ingredients as the previous
        # scorer, but encoded for the complete directed edge set at once.
        # The final one-shot scorer receives zero here.  The same shared
        # architecture can additionally condition on a current continuous
        # support value during offline iterative-policy distillation; that
        # scalar belongs to its edge and therefore permutes with it.
        self.point = nn.Sequential(nn.Linear(27, 32), nn.SiLU())
        self.temporal = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2), nn.SiLU())
        self.block = nn.Sequential(nn.Linear(64 + 24, width), nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        # Fixed inducing queries summarize each incident/global edge set.
        # This is O(M*N^2), with M fixed independently of population, rather
        # than O(N^3) edge-to-edge attention.  It retains richer alternative
        # interaction structure than a bare mean/max pool.
        self.incident_queries = nn.Parameter(torch.randn(self.inducing, width) * 0.02)
        self.global_queries = nn.Parameter(torch.randn(self.inducing, width) * 0.02)
        self.incident_attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.global_attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.incident_context = nn.Sequential(
            nn.Linear(self.inducing * width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.global_context = nn.Sequential(
            nn.Linear(self.inducing * width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.context_update = nn.Sequential(
            nn.Linear(4 * width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.head = nn.Sequential(
            nn.Linear(4 * width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        # Optional minimal output-capacity correction for functional logit
        # amortization.  It is a shared edge score from h_ij, added to the
        # existing contextual score; no slots or fixed-N output are introduced.
        self.pair_head = (nn.Sequential(nn.Linear(width, self.pair_residual_width), nn.SiLU(),
                                        nn.Linear(self.pair_residual_width, 1))
                          if self.pair_residual else None)
        # Optional output decomposition for physical-logit replay.  The
        # centered relative score decides *which* outgoing interaction wins;
        # a shared state-conditioned per-agent scalar decides density.  It
        # remains O(N^2), shared, and permutation equivariant.
        self.density_head = (nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1))
                             if self.per_agent_density else None)
        if self.density_head is not None:
            nn.init.zeros_(self.density_head[-1].weight)
            nn.init.constant_(self.density_head[-1].bias, .01)

    def _legacy_module(self, state_dict):
        inputs = state_dict["point.0.weight"].shape[1]
        legacy = PairLocalSupportU(
            width=self.width,
            blocks=self.blocks,
            dt=self.dt,
            distance=self.distance,
            include_agent_indices=(inputs == 28),
        ).to(self.point[0].weight)
        legacy.load_state_dict(state_dict, strict=True)
        self._legacy = legacy
        return legacy

    def load_state_dict(self, state_dict, strict=True):
        # Previous U checkpoints have no contextual edge/context modules.
        # Keep their historical behavior available for diagnostics, without
        # weakening the fresh equivariant architecture to preserve them.
        if "edge.0.weight" not in state_dict:
            return self._legacy_module(state_dict)
        result = super().load_state_dict(state_dict, strict=False if self.per_agent_density else strict)
        if self.per_agent_density:
            unexpected = [key for key in result.unexpected_keys]
            permitted_missing = {"density_head.0.weight", "density_head.0.bias",
                                 "density_head.2.weight", "density_head.2.bias"}
            missing = set(result.missing_keys)
            if unexpected or not missing.issubset(permitted_missing):
                raise RuntimeError(f"unexpected density-decomposition checkpoint mismatch: {result}")
        return result

    def scene_context(self, base):
        """Physical agent summaries shared with every incident edge token."""
        p = base[..., :2]
        relative = (p[:, :, None] - p[:, :, :, None]) / self.distance
        distance = torch.linalg.vector_norm(relative, dim=-1)
        n = p.shape[2]
        off = ~torch.eye(n, dtype=torch.bool, device=p.device)
        proximity = off / (1 + distance.square())
        nearest = distance.masked_fill(~off, float("inf")).amin(-1)
        direction = (relative * proximity[..., None]).sum(-2) / proximity.sum(
            -1
        ).clamp_min(1e-8)[..., None]
        return torch.cat(
            (
                nearest[..., None], proximity.sum(-1)[..., None],
                proximity.amax(-1)[..., None], direction,
            ),
            -1,
        )

    def _edge_tokens(self, base, endpoints, timestep, support_state=None):
        """Return h_ij with shape [B, N, N, width] for all directed pairs."""
        b, h, n, _ = base.shape
        if support_state is None:
            support_state = base.new_zeros((b, n, n))
        if support_state.shape != (b, n, n):
            raise ValueError("support_state must have shape [batch, N, N]")
        context = self.scene_context(base)
        scale = base.new_tensor([0.95, 0.95, self.dt * 0.5, self.dt * 0.5])
        ego = base[:, :, :, None]
        partner = base[:, :, None, :]
        relp = (partner[..., :2] - ego[..., :2]) / self.distance
        relv = (partner[..., 2:] - ego[..., 2:]) / (self.dt * 0.5)
        gap = torch.linalg.vector_norm(relp, dim=-1, keepdim=True) - 1
        approach = (relp * relv).sum(-1, keepdim=True)
        time = torch.linspace(-1, 1, h, device=base.device, dtype=base.dtype)
        time = time[None, :, None, None, None].expand(b, h, n, n, 1)
        reverse = (timestep.to(base) / 24)[:, None, None, None, None]
        reverse = reverse.expand(b, h, n, n, 1)
        point = torch.cat(
            (
                ego.expand(-1, -1, -1, n, -1) / scale,
                partner.expand(-1, -1, n, -1, -1) / scale,
                relp, relv, gap, approach,
                context[:, :, :, None].expand(-1, -1, -1, n, -1),
                context[:, :, None, :].expand(-1, -1, n, -1, -1),
                torch.tanh(support_state)[:, None].expand(-1, h, -1, -1)[..., None],
                time, reverse,
            ),
            -1,
        )
        # B*N*N independent temporal encoders, then block-pool exactly as
        # before.  The subsequent set pooling is the new contextual step.
        point = torch.asinh(point).permute(0, 2, 3, 4, 1).reshape(b * n * n, 27, h)
        hidden = self.temporal(self.point(point.transpose(1, 2)).transpose(1, 2))
        if h % self.blocks:
            raise ValueError(f"horizon {h} must be divisible by blocks {self.blocks}")
        hidden = hidden.reshape(b * n * n, 32, self.blocks, h // self.blocks)
        pooled = torch.cat((hidden.mean(-1), hidden.amax(-1)), 1).transpose(1, 2)
        endpoints_i = endpoints[:, :, None].expand(-1, -1, n, -1)
        endpoints_j = endpoints[:, None].expand(-1, n, -1, -1)
        es = scale.repeat(2)
        endpoint_features = torch.asinh(torch.cat(
            (endpoints_i / es, endpoints_j / es, (endpoints_j - endpoints_i) / es), -1
        )).reshape(b * n * n, 1, 24).expand(-1, self.blocks, -1)
        token = self.block(torch.cat((pooled, endpoint_features), -1))
        token = torch.cat((token.mean(1), token.amax(1)), -1)
        return self.edge(token).reshape(b, n, n, self.width)

    def logits_from_tokens(self, token):
        """Score a complete set of contextual edge tokens.

        This separates the shared physical edge encoder from the invariant
        outgoing/incoming/global aggregation.  It permits a cheap proxy to
        predict the exact fixed-width representation consumed by the scorer
        without changing the final, permutation-equivariant edge-logit head.
        """
        b, n, _, _ = token.shape
        diagonal = torch.eye(n, dtype=torch.bool, device=token.device)
        padding = diagonal[None].expand(b, -1, -1).reshape(b * n, n)
        global_padding = diagonal.reshape(1, n * n).expand(b, -1)

        def summaries(value):
            queries = self.incident_queries[None].expand(b * n, -1, -1)
            outgoing, _ = self.incident_attention(
                queries, value.reshape(b * n, n, self.width), value.reshape(b * n, n, self.width),
                key_padding_mask=padding, need_weights=False,
            )
            incoming_value = value.transpose(1, 2)
            incoming, _ = self.incident_attention(
                queries, incoming_value.reshape(b * n, n, self.width),
                incoming_value.reshape(b * n, n, self.width), key_padding_mask=padding, need_weights=False,
            )
            outgoing = self.incident_context(outgoing.flatten(1)).reshape(b, n, self.width)
            incoming = self.incident_context(incoming.flatten(1)).reshape(b, n, self.width)
            global_value, _ = self.global_attention(
                self.global_queries[None].expand(b, -1, -1),
                value.reshape(b, n * n, self.width), value.reshape(b, n * n, self.width),
                key_padding_mask=global_padding, need_weights=False,
            )
            global_value = self.global_context(global_value.flatten(1))[:, None, None]
            return outgoing, incoming, global_value

        pair_token = token
        outgoing, incoming, global_context = summaries(token)
        # One residual message-passing round lets each edge representation
        # encode competing incident/global interactions before the final,
        # still fixed-latent O(N^2), summaries are computed.
        token = token + self.context_update(torch.cat((
            token,
            outgoing[:, :, None].expand(-1, -1, n, -1),
            incoming[:, None].expand(-1, n, -1, -1),
            global_context.expand(-1, n, n, -1),
        ), -1))
        outgoing, incoming, global_context = summaries(token)
        features = torch.cat(
            (
                token,
                outgoing[:, :, None].expand(-1, -1, n, -1),
                incoming[:, None].expand(-1, n, -1, -1),
                global_context.expand(-1, n, n, -1),
            ),
            -1,
        )
        values = self.head(features).squeeze(-1)
        if self.pair_head is not None:
            values = values + self.pair_head(pair_token).squeeze(-1)
        if self.density_head is not None:
            off = ~diagonal
            relative = values - (values.masked_fill(~off[None], 0).sum(-1, keepdim=True) /
                                 off.sum(-1, keepdim=True).clamp_min(1))
            density_features = torch.cat((outgoing, incoming, global_context[:, 0].expand(-1, n, -1)), -1)
            density = self.density_head(density_features).squeeze(-1)
            values = relative + density[:, :, None]
        return values.masked_fill(~(~diagonal)[None], -float("inf"))

    def logits(self, base, endpoints, timestep, support_state=None):
        if self._legacy is not None:
            return self._legacy.logits(base, endpoints, timestep)
        token = self._edge_tokens(base, endpoints, timestep, support_state)
        return self.logits_from_tokens(token)

    def forward(self, base, endpoints, timestep, mode="dynamic"):
        if self._legacy is not None:
            return self._legacy(base, endpoints, timestep, mode)
        off = full_support(base)
        if mode == "open":
            return off, base.new_full(off.shape, float("inf")).masked_fill(
                ~off, -float("inf")
            )
        logits = self.logits(base, endpoints, timestep)
        if mode == "dynamic":
            return (logits > 0) & off, logits
        if mode == "top6":
            mask = torch.zeros_like(off).scatter(
                -1, logits.topk(min(6, base.shape[2] - 1), -1).indices, True
            )
            return mask & off, logits
        raise ValueError(mode)
