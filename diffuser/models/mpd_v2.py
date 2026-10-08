"""Thin, population-independent adapter around the official MPD diffusion prior.

This module deliberately owns no diffusion coefficients or learned network.  It
uses the methods/buffers of a loaded ``mmd`` GaussianDiffusionModel so that an
additive future coordination score remains in the MPD noise parameterization.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import nn


MPD_STATE_DIM = 4
MPD_DIFFUSION_STEPS = 25


def grouped_to_mpd_per_agent(x: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``[B,H,N,4]`` state to MPD's ``[B*N,H,4]`` layout."""
    if x.ndim != 4 or x.shape[-1] != MPD_STATE_DIM:
        raise ValueError("expected normalized grouped MPD state [B,H,N,4]")
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], MPD_STATE_DIM)


def mpd_per_agent_to_grouped(
    x: torch.Tensor, batch_size: int, num_agents: int
) -> torch.Tensor:
    """Convert MPD's independent-agent ``[B*N,H,4]`` layout to ``[B,H,N,4]``."""
    if (
        x.ndim != 3
        or x.shape[-1] != MPD_STATE_DIM
        or x.shape[0] != batch_size * num_agents
    ):
        raise ValueError("expected MPD per-agent state [B*N,H,4] with matching B and N")
    return x.reshape(batch_size, num_agents, x.shape[1], MPD_STATE_DIM).permute(
        0, 2, 1, 3
    )


def _flatten_timestep(
    timestep: torch.Tensor | int, batch_size: int, num_agents: int, device
) -> torch.Tensor:
    t = torch.as_tensor(timestep, device=device, dtype=torch.long)
    if t.ndim == 0:
        return t.expand(batch_size * num_agents)
    if tuple(t.shape) == (batch_size,):
        return t[:, None].expand(batch_size, num_agents).reshape(-1)
    if tuple(t.shape) == (batch_size, num_agents):
        return t.reshape(-1)
    if tuple(t.shape) == (batch_size * num_agents,):
        return t
    raise ValueError("timestep must be scalar, [B], [B,N], or [B*N]")


class MPDUnaryAdapter(nn.Module):
    """Frozen MPD Unary with grouped population semantics and no cross-agent path.

    ``diffusion`` is the official ``GaussianDiffusionModel`` and must use
    epsilon prediction with four state channels.  Endpoint conditions are
    represented as normalized dictionaries ``{physical_time: [B,N,4]}`` and
    are applied at sampler boundaries exactly as MPD does; they are not hidden
    denoiser inputs in the official model.
    """

    def __init__(self, diffusion: nn.Module):
        super().__init__()
        if getattr(diffusion, "state_dim", None) != MPD_STATE_DIM:
            raise ValueError("MPD v2 accepts only the official four-channel state")
        if not bool(getattr(diffusion, "predict_epsilon", False)):
            raise ValueError("MPD v2 requires an epsilon-predicting MPD model")
        if getattr(diffusion, "n_diffusion_steps", None) != MPD_DIFFUSION_STEPS:
            raise ValueError("MPD v2 refuses non-official 25-step checkpoints")
        self.diffusion = diffusion
        self.diffusion.eval()
        for parameter in self.diffusion.parameters():
            parameter.requires_grad_(False)

    @property
    def model(self):
        """The official TemporalUnet; exposed for auditing, never replaced."""
        return self.diffusion.model

    def _flat_conditions(
        self,
        hard_conditions: Optional[Mapping[int, torch.Tensor]],
        batch_size: int,
        num_agents: int,
    ):
        if hard_conditions is None:
            return None
        flattened = {}
        for physical_time, value in hard_conditions.items():
            if tuple(value.shape) != (batch_size, num_agents, MPD_STATE_DIM):
                raise ValueError("each hard condition must be normalized [B,N,4]")
            flattened[int(physical_time)] = value.reshape(
                batch_size * num_agents, MPD_STATE_DIM
            )
        return flattened

    def apply_hard_conditions(
        self, grouped: torch.Tensor, hard_conditions=None
    ) -> torch.Tensor:
        """Apply MPD endpoint overwrites to a grouped state without altering other agents."""
        b, _, n, _ = grouped.shape
        flat = grouped_to_mpd_per_agent(grouped).clone()
        for physical_time, value in (
            self._flat_conditions(hard_conditions, b, n) or {}
        ).items():
            flat[:, physical_time, :] = value.clone()
        return mpd_per_agent_to_grouped(flat, b, n)

    def q_sample(
        self, x0: torch.Tensor, timestep, epsilon: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, _, n, _ = x0.shape
        flat = grouped_to_mpd_per_agent(x0)
        noise = None if epsilon is None else grouped_to_mpd_per_agent(epsilon)
        result = self.diffusion.q_sample(
            flat, _flatten_timestep(timestep, b, n, flat.device), noise=noise
        )
        return mpd_per_agent_to_grouped(result, b, n)

    def predict_epsilon(
        self, noisy: torch.Tensor, timestep, context=None, return_features: bool = False
    ):
        """Call the official denoiser once on the flattened independent-agent batch.

        Official Boundary MPD has ``context=None``.  A non-None context is
        flattened only when its leading dimensions are ``[B,N,...]``.
        """
        b, _, n, _ = noisy.shape
        flat = grouped_to_mpd_per_agent(noisy)
        t = _flatten_timestep(timestep, b, n, flat.device)
        flat_context = context
        if context is not None and context.shape[:2] == (b, n):
            flat_context = context.reshape(b * n, *context.shape[2:])
        epsilon = self.diffusion.model(flat, t, flat_context)
        result = mpd_per_agent_to_grouped(epsilon, b, n)
        # TemporalUnet has no public intermediate-feature API.  Do not install
        # hooks into the frozen official graph; future factors may add encoders.
        return (result, None) if return_features else result

    forward = predict_epsilon

    @torch.no_grad()
    def predict_clean(
        self, noisy: torch.Tensor, epsilon_hat: torch.Tensor, timestep
    ) -> torch.Tensor:
        """Official MPD epsilon-to-x0 conversion in grouped layout.

        This is intentionally just the conversion already used by
        :meth:`reverse_step`.  The privileged planning diagnostic uses it to
        score candidate G subsets without changing the DDPM sampler.
        """
        if noisy.shape != epsilon_hat.shape:
            raise ValueError("epsilon_hat must have the same grouped shape as noisy")
        b, _, n, _ = noisy.shape
        x = grouped_to_mpd_per_agent(noisy)
        eps = grouped_to_mpd_per_agent(epsilon_hat)
        t = _flatten_timestep(timestep, b, n, x.device)
        clean = self.diffusion.predict_start_from_noise(x, t=t, noise=eps)
        if self.diffusion.clip_denoised:
            clean = clean.clamp(-1.0, 1.0)
        return mpd_per_agent_to_grouped(clean, b, n)

    def reverse_mean(
        self,
        noisy: torch.Tensor,
        epsilon_hat: torch.Tensor,
        timestep,
        hard_conditions=None,
    ) -> torch.Tensor:
        """Exact deterministic DDPM mean used by :meth:`reverse_step`.

        This is deliberately a training/diagnostic hook, rather than a second
        sampler.  It follows official MPD's ``p_mean_variance`` path exactly:
        epsilon prediction -> clipped x0 -> ``q_posterior`` mean, followed by
        the same endpoint overwrite that the sampler applies after noise.  It
        is differentiable with respect to ``epsilon_hat`` so an offline loss
        can supervise an epsilon residual in reverse-mean space.  Normal
        deployment continues to call :meth:`reverse_step` only.
        """
        if noisy.shape != epsilon_hat.shape:
            raise ValueError(
                "epsilon_hat must have the same grouped [B,H,N,4] shape as noisy"
            )
        b, _, n, _ = noisy.shape
        x = grouped_to_mpd_per_agent(noisy)
        eps = grouped_to_mpd_per_agent(epsilon_hat)
        t = _flatten_timestep(timestep, b, n, x.device)
        x_start = self.diffusion.predict_start_from_noise(x, t=t, noise=eps)
        if self.diffusion.clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        mean, _, _ = self.diffusion.q_posterior(x_start=x_start, x_t=x, t=t)
        return self.apply_hard_conditions(
            mpd_per_agent_to_grouped(mean, b, n), hard_conditions
        )

    @torch.no_grad()
    def reverse_step_from_mean(
        self,
        noisy: torch.Tensor,
        reverse_mean: torch.Tensor,
        timestep,
        hard_conditions=None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample the official posterior around a caller-supplied DDPM mean.

        Reverse-mean residual mode changes only the deterministic mean before
        the existing posterior draw.  The variance, noise convention, and
        endpoint overwrite are identical to :meth:`reverse_step`; this method
        never evaluates an energy or invokes autograd.
        """
        if noisy.shape != reverse_mean.shape:
            raise ValueError("reverse_mean must match grouped noisy [B,H,N,4]")
        b, _, n, _ = noisy.shape
        x = grouped_to_mpd_per_agent(noisy)
        mean = grouped_to_mpd_per_agent(reverse_mean)
        t = _flatten_timestep(timestep, b, n, x.device)
        # In official q_posterior the returned variance depends only on t.
        _, _, log_variance = self.diffusion.q_posterior(
            x_start=torch.zeros_like(x), x_t=x, t=t
        )
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        noise[t == 0] = 0
        result = mean + torch.exp(0.5 * log_variance) * noise
        return self.apply_hard_conditions(
            mpd_per_agent_to_grouped(result, b, n), hard_conditions
        )

    @torch.no_grad()
    def reverse_step(
        self,
        noisy: torch.Tensor,
        epsilon_hat: torch.Tensor,
        timestep,
        hard_conditions=None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """MPD DDPM update with a caller-supplied epsilon prediction.

        The x0 conversion and posterior are the official MPD methods.  This is
        the one extension needed to inject future ``C + sum(M*S)`` corrections.
        """
        if noisy.shape != epsilon_hat.shape:
            raise ValueError(
                "epsilon_hat must have the same grouped [B,H,N,4] shape as noisy"
            )
        b, _, n, _ = noisy.shape
        x = grouped_to_mpd_per_agent(noisy)
        t = _flatten_timestep(timestep, b, n, x.device)
        x_start = grouped_to_mpd_per_agent(
            self.predict_clean(noisy, epsilon_hat, timestep)
        )
        mean, _, log_variance = self.diffusion.q_posterior(x_start=x_start, x_t=x, t=t)
        noise = torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        )
        noise[t == 0] = 0
        result = mean + torch.exp(0.5 * log_variance) * noise
        return self.apply_hard_conditions(
            mpd_per_agent_to_grouped(result, b, n), hard_conditions
        )
