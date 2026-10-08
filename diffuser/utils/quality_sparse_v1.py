"""Physical quality contract for the versioned sparse planner.

No model import, native repair, or online optimization is hidden here. Offline
solvers are explicitly named and called only by reference/target producers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.interpolate import BSpline

DT = 5.0 / 64


@dataclass(frozen=True)
class QualitySpec:
    dt: float = DT
    radius: float = 0.05
    native_distance: float = 0.105
    teacher_distance: float = 0.110
    center_bound: float = 0.95
    endpoint_tolerance: float = 0.05
    collision_weight: float = 50.0
    gp_weight: float = 0.2
    length_weight: float = 0.2
    trust_weight: float = 0.05
    workspace_weight: float = 100.0
    gp_scale: float = 1.0
    length_scale: float = 1.0
    trust_scale: float = 0.01

    def metadata(self):
        return asdict(self)


class PhysicalCodec:
    """Unclipped affine conversion; distinct from native model normalization."""

    def __init__(self, normalizer, device):
        self.lo = normalizer._mins_cpu.to(device)
        self.hi = normalizer._maxs_cpu.to(device)
        self.scale = (self.hi - self.lo) / 2
        self.mid = (self.hi + self.lo) / 2

    def decode(self, normalized):
        return normalized * self.scale + self.mid

    def encode(self, physical):
        return (physical - self.mid) / self.scale


def gp_cost(x, dt=DT, stored_displacement=True):
    """Native GP algebra; use w/dt for a dimensionally physical state."""
    p, v = x[..., :2], x[..., 2:]
    if stored_displacement:
        v = v / dt
    ep = p[:, 1:] - p[:, :-1] - dt * v[:, :-1]
    ev = v[:, 1:] - v[:, :-1]
    terms = (12 / dt**3) * ep.square() - (12 / dt**2) * ep * ev
    terms = terms + (4 / dt) * ev.square()
    return terms.sum((1, 3)).mean(-1)


def coherent_state(positions):
    """Model displacement channels, including its two hard zero endpoints."""
    w = torch.cat(
        (positions[:, 1:] - positions[:, :-1], torch.zeros_like(positions[:, :1])), 1
    )
    mask = torch.ones_like(w[..., :1])
    mask[:, 0] = 0
    return torch.cat((positions, w * mask), -1)


def endpoint_condition(x, endpoints):
    interior = x[:, 1:-1]
    return torch.cat((endpoints[:, :1], interior, endpoints[:, 1:]), 1)


def quality_terms(x, prior, spec):
    b, h, n, _ = x.shape
    p = x[..., :2]
    i, j = torch.triu_indices(n, n, 1, device=x.device)
    distances = torch.linalg.vector_norm(p[:, :, i] - p[:, :, j], dim=-1)
    collision = (
        (spec.teacher_distance - distances).clamp_min(0) / spec.native_distance
    ).square().sum((1, 2)) / n
    gp = gp_cost(x, spec.dt) / spec.gp_scale
    length = (
        torch.linalg.vector_norm(p[:, 1:] - p[:, :-1], dim=-1).sum(1).mean(1)
        / spec.length_scale
    )
    trust = (p - prior[..., :2]).square().mean((1, 2, 3)) / spec.trust_scale
    workspace = ((p.abs() - spec.center_bound).clamp_min(0) / spec.radius).square().sum(
        (1, 2, 3)
    ) / n
    loss = (
        spec.collision_weight * collision
        + spec.gp_weight * gp
        + spec.length_weight * length
        + spec.trust_weight * trust
        + spec.workspace_weight * workspace
    )
    return {
        "loss": loss,
        "collision": collision,
        "gp": gp,
        "length": length,
        "trust": trust,
        "workspace": workspace,
    }


def metrics(x, starts, goals, spec=None):
    spec = spec or QualitySpec()
    a = torch.as_tensor(x, dtype=torch.float64)
    if a.ndim == 3:
        a = a[None]
    if a.shape[0] != 1:
        raise ValueError("metrics expects one configuration")
    finite = bool(torch.isfinite(a).all())
    if not finite:
        return {"complete_valid": False, "physical_valid": False, "nonfinite": True}
    p = a[0, ..., :2]
    n = p.shape[1]
    i, j = torch.triu_indices(n, n, 1)
    d = torch.linalg.vector_norm(p[:, i] - p[:, j], dim=-1)
    physical, native = d < 0.100, d < spec.native_distance
    start = torch.linalg.vector_norm(p[0] - torch.as_tensor(starts), dim=-1).max()
    goal = torch.linalg.vector_norm(p[-1] - torch.as_tensor(goals), dim=-1).max()
    boundary = (p.abs() - spec.center_bound).clamp_min(0).max()
    epok = max(float(start), float(goal)) <= spec.endpoint_tolerance
    bok = float(boundary) <= 1e-6
    dp = torch.diff(p, dim=0)
    acc = torch.diff(p, n=2, dim=0) / spec.dt**2
    jerk = torch.diff(p, n=3, dim=0) / spec.dt**3
    consistency = (a[0, 1:-1, :, 2:] - dp[1:]) / spec.dt
    penetration = (0.100 - d).clamp_min(0)
    return {
        "complete_valid": bool(not native.any() and epok and bok),
        "physical_valid": bool(not physical.any() and epok and bok),
        "collision_free_0100": bool(not physical.any()),
        "native_clearance_0105": bool(not native.any()),
        "boundary_valid": bok,
        "endpoint_valid": epok,
        "nonfinite": False,
        "collision_pair_times": int(physical.sum()),
        "native_pair_times": int(native.sum()),
        "unique_collision_pairs": int(physical.any(0).sum()),
        "unique_native_pairs": int(native.any(0).sum()),
        "penetration_sum": float(penetration.sum()),
        "penetration_max": float(penetration.max()),
        "min_distance": float(d.min()),
        "boundary_excess_max": float(boundary),
        "start_error_max": float(start),
        "goal_error_max": float(goal),
        "mean_path_length": float(torch.linalg.vector_norm(dp, dim=-1).sum(0).mean()),
        "gp_physical": float(gp_cost(a, spec.dt)),
        "gp_native_stored": float(gp_cost(a, spec.dt, False)),
        "acceleration_rms": float(acc.square().sum(-1).mean().sqrt()),
        "jerk_rms": float(jerk.square().sum(-1).mean().sqrt()),
        "velocity_consistency_rms": float(consistency.square().sum(-1).mean().sqrt()),
        "velocity_consistency_convention": "interior w[t]/dt vs forward position difference/dt; hard endpoint w[0] excluded",
    }


def temporal_weights(coeff, horizon=64):
    """Continuous linear interpolation couples neighboring G blocks."""
    shape = coeff.shape[:-1]
    return torch.nn.functional.interpolate(
        coeff.reshape(-1, 1, coeff.shape[-1]),
        size=horizon,
        mode="linear",
        align_corners=True,
    ).reshape(*shape, horizon)


def spline_matrix(count, horizon=64, device=None, dtype=torch.float32):
    degree = 3
    knots = np.r_[np.zeros(4), np.linspace(0, 1, count - 2)[1:-1], np.ones(4)]
    values = BSpline(knots, np.eye(count), degree)(np.linspace(0, 1, horizon))
    return torch.as_tensor(values, dtype=dtype, device=device)


def smooth_projection(horizon=64, device=None, dtype=torch.float32):
    # Zero first/last two spline controls gives zero value and derivative at ends.
    basis = spline_matrix(16, horizon, device, torch.float64)[:, 2:-2]
    projection = basis @ torch.linalg.pinv(basis)
    return projection.to(dtype)


def smooth_v6_fields(fields):
    """One targeted change: physical clean-position spline projection."""
    proj = smooth_projection(fields.shape[-2], fields.device, fields.dtype)
    return torch.einsum("th,...hd->...td", proj, fields[..., :2])


def compose_smooth(base, fields, coeff, weight_basis=None):
    """Differentiate the COMPLETE weighted correction, including G transitions."""
    basis = (
        weight_basis
        if weight_basis is not None
        else spline_matrix(coeff.shape[-1], base.shape[1], base.device, base.dtype)
    )
    weights = torch.einsum("...k,hk->...h", coeff, basis)
    delta = (fields * weights[..., None]).sum(2).permute(0, 2, 1, 3)
    dstate = coherent_state(delta)
    return base + dstate


def compose_v6(base, fields, coeff):
    # fields [B,N,N,H,4], coeff [B,N,N,K].
    w = temporal_weights(coeff, base.shape[1])
    delta = (fields * w[..., None]).sum(2).permute(0, 2, 1, 3)
    return base + delta


def offline_v6_coefficients(
    base, fields, spec, iterations=180, initial=None, smooth=False
):
    """Bounded offline coefficient reference, no inference caller."""
    b, _, n, _ = base.shape
    basis = spline_matrix(8, base.shape[1], base.device, base.dtype) if smooth else None
    compose = (
        lambda weights: compose_smooth(base, fields, weights, basis)
        if smooth
        else compose_v6(base, fields, weights)
    )
    with torch.enable_grad():
        coeff = (
            torch.zeros((b, n, n, 8), device=base.device)
            if initial is None
            else initial.clone()
        )
        coeff.requires_grad_(True)
        opt = torch.optim.Adam([coeff], lr=0.025)
        best, best_loss = coeff.detach().clone(), base.new_full((b,), float("inf"))
        curve = []
        for step in range(iterations + 1):
            x = compose(coeff)
            terms = quality_terms(x, base, spec)
            loss = terms["loss"]
            with torch.no_grad():
                improved = loss < best_loss
                best[improved] = coeff[improved]
                best_loss = torch.minimum(best_loss, loss)
            if step % 30 == 0 or step == iterations:
                curve.append(
                    {
                        "step": step,
                        **{k: float(v.mean().detach()) for k, v in terms.items()},
                    }
                )
            if step == iterations:
                break
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                coeff.clamp_(0, 1)
        return compose(best).detach(), best, curve


def offline_free_reference(base, endpoints, spec, iterations=180, physical_contract=None):
    """Constructive offline full-position solver with explicit box/endpoints.

    Uses coherent displacement channels; no trajectory averaging or repair.
    All outputs (including infeasible local minima) must be independently checked.
    """
    with torch.enable_grad():
        p = (
            base[..., :2]
            .detach()
            .clone()
            .clamp(-spec.center_bound, spec.center_bound)
            .requires_grad_(True)
        )
        opt = torch.optim.Adam([p], lr=0.01)
        best = p.detach().clone()
        best_loss = p.new_full((len(p),), float("inf"))
        curve = []
        for step in range(iterations + 1):
            q = torch.cat(
                (endpoints[:, :1, :, :2], p[:, 1:-1], endpoints[:, 1:, :, :2]), 1
            )
            x = coherent_state(q)
            terms = quality_terms(x, base, spec)
            loss = terms["loss"]
            if physical_contract is not None:
                loss = loss + physical_contract.penalty(x)
            with torch.no_grad():
                improved = loss < best_loss
                best[improved] = q[improved]
                best_loss = torch.minimum(best_loss, loss)
            if step % 30 == 0 or step == iterations:
                curve.append(
                    {
                        "step": step,
                        **{k: float(v.mean().detach()) for k, v in terms.items()},
                    }
                )
            if step == iterations:
                break
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(-spec.center_bound, spec.center_bound)
        return coherent_state(best).detach(), curve
