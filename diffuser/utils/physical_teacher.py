"""Frozen signed physical teacher optimization used by the qualified corpus.

The numeric objective, starts, continuation, and acceptance gates are extracted
from ``train_n28_joint_ug_proposal.make_teacher`` and
``run_n28_robust_teachers``. Scene and guide acquisition are caller concerns.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import torch
from scipy.linalg import lstsq
from scipy.optimize import minimize

from diffuser.utils import quality_sparse_v1 as q

TARGET = 0.120
PENALTIES = (2_000.0, 200_000.0, 20_000_000.0, 2_000_000_000.0)
MOTION_KEYS = ("path_m", "gp", "acceleration_rms", "jerk_rms")


def swept_distances(output):
    positions = output[..., :2]
    i, j = torch.triu_indices(positions.shape[2], positions.shape[2], 1, device=positions.device)
    relative = positions[:, :, i] - positions[:, :, j]
    initial = relative[:, :-1]
    change = relative[:, 1:] - initial
    fraction = (-(initial * change).sum(-1) / change.square().sum(-1).clamp_min(1e-30)).clamp(0, 1)
    return torch.linalg.vector_norm(initial + fraction[..., None] * change, dim=-1)


def barrier(output, rho=TARGET):
    return ((rho - swept_distances(output)).clamp_min(0) / 0.01).square().sum() / output.shape[2]


def assess(output, scene, spec):
    """Historical float64 sampled metrics and exact linear swept checks."""
    physical = output.detach().cpu().numpy() if torch.is_tensor(output) else np.asarray(output)
    if physical.ndim == 4:
        physical = physical[0]
    metrics = q.metrics(physical, scene["starts"], scene["goals"], replace(spec, dt=q.DT))
    p = np.asarray(physical[..., :2], dtype=np.float64)
    i, j = np.triu_indices(p.shape[1], 1)
    a = p[:-1, i] - p[:-1, j]
    change = (p[1:, i] - p[1:, j]) - a
    fraction = np.clip(-(a * change).sum(-1) / np.maximum((change * change).sum(-1), 1e-30), 0, 1)
    distance = np.linalg.norm(a + fraction[..., None] * change, axis=-1)
    sweep = dict(min_distance=float(distance.min()),
                 physical_interval_pair_collisions=int((distance < 0.100).sum()),
                 native_interval_pair_collisions=int((distance < 0.105).sum()))
    return dict(sampled=metrics, swept=sweep,
                qualified=bool(metrics["complete_valid"] and sweep["native_interval_pair_collisions"] == 0))


def quality_brief(output, scene, spec):
    result = assess(output, scene, spec)
    sampled, swept = result["sampled"], result["swept"]
    return dict(qualified=result["qualified"],
                physical_0100_intervals=swept["physical_interval_pair_collisions"],
                common_0105_intervals=swept["native_interval_pair_collisions"],
                min_swept_distance_m=swept["min_distance"],
                path_m=sampled["mean_path_length"], gp=sampled["gp_physical"],
                acceleration_rms=sampled["acceleration_rms"], jerk_rms=sampled["jerk_rms"],
                endpoint_valid=sampled["endpoint_valid"], workspace_valid=sampled["boundary_valid"],
                boundary_excess_m=sampled["boundary_excess_max"])


def quality_terms(output, prior, spec):
    terms = q.quality_terms(output, prior, spec)
    sampled = terms["collision"]
    swept = ((spec.teacher_distance - swept_distances(output)).clamp_min(0) / spec.native_distance).square().sum((1, 2)) / output.shape[2]
    terms["loss"] = terms["loss"] + spec.collision_weight * (swept - sampled).clamp_min(0)
    terms["swept_collision"] = swept
    terms["swept_min_distance"] = swept_distances(output).amin((1, 2))
    return terms


def refine_guide(raw, endpoints, spec, iterations=480):
    """Historical Adam refinement of the secondary ORCA guide."""
    with torch.enable_grad():
        position = raw[..., :2].detach().clone().clamp(
            -spec.center_bound, spec.center_bound).requires_grad_(True)
        optimizer = torch.optim.Adam([position], lr=0.01)
        best = raw.detach().clone()
        best_loss = float("inf")
        feasible = None
        feasible_loss = float("inf")
        for step in range(iterations + 1):
            constrained = torch.cat(
                (endpoints[:, :1, :, :2], position[:, 1:-1], endpoints[:, 1:, :, :2]), 1)
            output = q.coherent_state(constrained)
            terms = quality_terms(output, raw, spec)
            loss = terms["loss"].mean()
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                best = output.detach().clone()
            if (bool((terms["swept_min_distance"] >= spec.native_distance).all())
                    and float(loss.detach()) < feasible_loss):
                feasible_loss = float(loss.detach())
                feasible = output.detach().clone()
            if step == iterations:
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                position.clamp_(-spec.center_bound, spec.center_bound)
        return feasible if feasible is not None else best


def design_matrices(fields, basis):
    n, _, horizon, dimensions = fields.shape
    partners = np.array([[j for j in range(n) if j != i] for i in range(n)])
    selected = fields[np.arange(n)[:, None], partners]
    matrix = np.einsum("ijhd,hk->ihdjk", selected.astype(np.float64), basis.astype(np.float64))
    return matrix.reshape(n, horizon * dimensions, -1), partners


def coefficient_tensor(values, partners, base):
    n = base.shape[2]
    dense = base.new_zeros((1, n, n, 8))
    dense[0, torch.arange(n, device=base.device)[:, None], torch.as_tensor(partners, device=base.device)] = values.reshape(n, n - 1, 8)
    return dense


def initial_signed_fit(base, fields, guide, scene, spec, basis, scale, updates=120,
                       workspace_feasible=True):
    """Historical t=0 TSVD start followed by bounded signed fit to the guide."""
    n = base.shape[2]
    matrices, partners = design_matrices(fields[0].cpu().numpy(), basis.cpu().numpy())
    desired = (guide - base)[0, ..., :2].cpu().numpy().transpose(1, 0, 2).reshape(n, 128)
    coefficients = np.stack([lstsq(a, b, cond=1e-6, lapack_driver="gelsd")[0]
                             for a, b in zip(matrices, desired)])
    initial = np.clip(coefficients.reshape(n, n - 1, 8) / scale, -1, 1)
    best, best_score, feasible = None, math.inf, False

    def evaluate(flat, gradient=False):
        nonlocal best, best_score, feasible
        weights = torch.tensor(flat.reshape(n, n - 1, 8), device=base.device,
                               dtype=torch.float32, requires_grad=gradient)
        dense = coefficient_tensor(scale * weights, partners, base)
        output = q.compose_smooth(base, fields, dense, basis)
        terms = quality_terms(output, base, spec)
        fit = (output[..., :2] - guide[..., :2]).square().mean() / 0.01
        loss = terms["loss"].mean() + fit
        ok = bool((terms["swept_min_distance"] >= spec.native_distance).all()
                  and (not workspace_feasible or
                       (output[..., :2].abs() <= spec.center_bound + 1e-6).all()))
        score = float(loss.detach())
        if best is None or (ok and not feasible) or (ok == feasible and score < best_score):
            best, best_score, feasible = weights.detach().clone(), score, ok
        if gradient:
            grad = torch.autograd.grad(loss, weights)[0]
            return score, grad.detach().cpu().numpy().astype(np.float64).ravel()
        return score

    zero = np.zeros_like(initial)
    zero_score = evaluate(zero)
    clipped_score = evaluate(initial)
    start = initial if clipped_score < zero_score else zero
    minimize(lambda x: evaluate(x, True), start.ravel(), jac=True, method="L-BFGS-B",
             bounds=[(-1.0, 1.0)] * initial.size,
             options=dict(maxiter=updates, maxfun=4 * updates, ftol=1e-10, gtol=1e-8, maxls=30))
    dense = coefficient_tensor(scale * best, partners, base)
    target = q.compose_smooth(base, fields, dense, basis).detach()
    return target, dense.detach()


def fit_start(base, fields, basis, external, scale):
    """Historical per-ego TSVD external-guide start."""
    n = base.shape[2]
    active = np.array([(i, j) for i in range(n) for j in range(n) if i != j])
    result = np.zeros((len(active), 8), dtype=np.float32)
    desired = (external[..., :2] - base[..., :2]).detach().cpu().numpy()
    field_np = fields.detach().cpu().numpy()
    basis_np = basis.detach().cpu().numpy()
    for ego in range(n):
        positions = np.flatnonzero(active[:, 0] == ego)
        part = field_np[0, ego, active[positions, 1]].transpose(1, 0, 2)
        matrix = np.einsum("hjd,hk->hdjk", part, basis_np).reshape(128, -1)
        rhs = desired[0, :, ego].reshape(-1)
        solution = lstsq(matrix * scale, rhs, cond=1e-6, lapack_driver="gelsd")[0]
        result[positions] = np.clip(solution.reshape(-1, 8), -1, 1)
    return torch.as_tensor(result, device=base.device)


def _candidate(state, weights, start, penalty, status, iterations, begun):
    with torch.no_grad():
        output = state["compose"](weights).detach()
        result = quality_brief(output, state["scene"], state["spec"])
        terms = quality_terms(output, state["base"], state["spec"])
        guide = (output[..., :2] - state["guide"][..., :2]).square().mean() / 0.01
        original = terms["loss"].mean() + guide
        correction = float((output[..., :2] - state["base"][..., :2]).square().mean().sqrt())
        double_output = state["compose"](weights.detach().double(), torch.float64)
        error = float((output[..., :2].double() - double_output[..., :2]).square().mean().sqrt())
        maximum = float(torch.linalg.vector_norm(output[..., :2].double() - double_output[..., :2], dim=-1).max())
        saturation = float((weights.abs() >= 0.999).float().mean())
        violation = float(barrier(output))
    baseline = state["baseline"]
    envelope = {key: result[key] <= 1.10 * max(baseline[key], 1e-8) for key in MOTION_KEYS}
    envelope["correction"] = correction <= max(1.5 * state["baseline_correction"], state["baseline_correction"] + 0.03)
    finite = all(math.isfinite(v) for v in (float(original), float(guide), correction, error, maximum, saturation, violation))
    numeric = finite and saturation <= 0.20 and error <= 0.001 and maximum <= 0.003
    qualified = bool(result["min_swept_distance_m"] >= TARGET and result["qualified"]
                     and result["endpoint_valid"] and result["workspace_valid"]
                     and all(envelope.values()) and numeric)
    record = dict(start=start, phase=penalty, status=status, iterations=iterations,
                  optimizer_seconds=time.monotonic() - begun, requested_rho_m=TARGET,
                  actual_min_swept_m=result["min_swept_distance_m"],
                  rho_met=bool(result["min_swept_distance_m"] >= TARGET),
                  valid_0100=result["physical_0100_intervals"] == 0,
                  valid_0105=result["common_0105_intervals"] == 0,
                  physical_0100_intervals=result["physical_0100_intervals"],
                  common_0105_intervals=result["common_0105_intervals"],
                  endpoint_valid=result["endpoint_valid"], workspace_valid=result["workspace_valid"],
                  path_m=result["path_m"], gp=result["gp"],
                  acceleration_rms=result["acceleration_rms"], jerk_rms=result["jerk_rms"],
                  correction_rms_m=correction, original_objective=float(original),
                  external_guide_fit=float(guide), coefficient_saturation=saturation,
                  float32_vs_float64_rms_m=error, float32_vs_float64_max_m=maximum,
                  barrier_value=violation, motion_checks=envelope,
                  numeric_stable=bool(numeric), qualified=qualified)
    return record, weights.detach().clone(), output


def optimize(base, fields, initial_coeff, initial_target, guide, scene, spec,
             scale, robust_start=None, tiny=False, *, start_limit=None,
             iteration_scale=1, acceptance=None, stop_on_accept=False,
             extra_barrier=None, require_canonical_qualified=True):
    """Historical four-level bounded L-BFGS-B with ordered restarts."""
    n = base.shape[2]
    basis = q.spline_matrix(8, device=base.device)
    index = torch.nonzero(~torch.eye(n, dtype=torch.bool, device=base.device), as_tuple=False)
    i, j = index.unbind(-1)
    original = (initial_coeff[0, i, j] / scale).clamp(-1, 1)

    def compose(weights, dtype=torch.float32):
        b, f = base.to(dtype), fields.to(dtype)
        dense = b.new_zeros((1, n, n, 8))
        dense[0, i, j] = scale * weights.to(dtype)
        return q.compose_smooth(b, f, dense, basis.to(dtype))

    baseline = quality_brief(initial_target, scene, spec)
    state = dict(base=base, guide=guide, scene=scene, spec=spec, compose=compose,
                 baseline=baseline,
                 baseline_correction=float((initial_target[..., :2] - base[..., :2]).square().mean().sqrt()))
    begun = time.monotonic()
    candidates = []

    def accepted(entry):
        return ((not require_canonical_qualified or entry[0]["qualified"])
                and (acceptance is None or acceptance(entry[2])))

    def record_candidate(weights, start, penalty, status, iterations):
        entry = _candidate(state, weights, start, penalty, status, iterations, begun)
        entry[0]["start_index"] = start_index
        entry[0]["external_acceptance"] = bool(accepted(entry))
        candidates.append(entry)
        return entry[0]["external_acceptance"]

    def starts():
        yield "saved", original
        yield "zero", torch.zeros_like(original)
        if not any(row[0]["external_acceptance"] for row in candidates) and robust_start is not None:
            transferred = torch.zeros_like(original)
            for edge, value in zip(robust_start["edges"], robust_start["coefficients_q"]):
                edge_i, edge_j = map(int, edge)
                if edge_i < n and edge_j < n and edge_i != edge_j:
                    position = edge_i * (n - 1) + edge_j - (edge_j > edge_i)
                    transferred[position] = torch.as_tensor(value, device=base.device, dtype=base.dtype)
            yield "frozen_training_robust_start", transferred
        if not any(row[0]["external_acceptance"] for row in candidates):
            external = fit_start(base, fields, basis, guide, scale)
            yield "external_fit", external
            rng = np.random.default_rng(280918 + int(scene["configuration_id"]) * 25)
            for k in range(max(3, (start_limit or 0) - 3)):
                perturb = torch.as_tensor(rng.normal(0, 0.02, original.shape), device=base.device, dtype=torch.float32)
                yield f"signed_small_{k + 1}", (original + perturb).clamp(-1, 1)

    for start_index, (start_name, initial) in enumerate(starts()):
        if start_limit is not None and start_index >= start_limit:
            break
        current = initial.detach().cpu().numpy().astype(np.float64).ravel()
        for penalty in PENALTIES[:1] if tiny else PENALTIES:
            def evaluate(flat):
                weights = torch.tensor(flat.reshape(-1, 8), device=base.device,
                                       dtype=torch.float32, requires_grad=True)
                output = compose(weights)
                terms = quality_terms(output, base, spec)
                motion = (spec.gp_weight * terms["gp"].mean()
                          + spec.length_weight * terms["length"].mean()
                          + spec.trust_weight * terms["trust"].mean()
                          + 10 * spec.workspace_weight * terms["workspace"].mean())
                fit = (output[..., :2] - guide[..., :2]).square().mean() / 0.01
                loss = penalty * (barrier(output) +
                                  (extra_barrier(output) if extra_barrier is not None else 0)) + motion + 0.01 * fit
                value = float(loss.detach())
                gradient = torch.autograd.grad(loss, weights)[0]
                if not math.isfinite(value) or not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite teacher objective or gradient")
                return value, gradient.detach().cpu().numpy().astype(np.float64).ravel()

            result = minimize(evaluate, current, jac=True, method="L-BFGS-B",
                              bounds=[(-1, 1)] * len(current),
                              options=dict(maxiter=2 if tiny else 80 * iteration_scale,
                                           maxfun=8 if tiny else 320 * iteration_scale,
                                           ftol=1e-10, gtol=1e-8, maxls=30))
            current = result.x
            weights = torch.as_tensor(current.reshape(-1, 8), device=base.device, dtype=torch.float32)
            passed = record_candidate(weights, start_name, penalty,
                                      str(result.message), int(result.nit))
            if not candidates[-1][0]["qualified"] and candidates[-1][0]["numeric_stable"]:
                projected = weights.detach().double().clone()
                for _ in range(24):
                    projected.requires_grad_(True)
                    distance = swept_distances(compose(projected, torch.float64)).amin()
                    if float(distance.detach()) >= TARGET + 1e-5:
                        break
                    gradient = torch.autograd.grad(distance, projected)[0]
                    squared = gradient.square().sum()
                    if float(squared) <= 1e-16:
                        break
                    with torch.no_grad():
                        projected = (projected + (TARGET + 1e-5 - distance) / squared * gradient).clamp(-1, 1).detach()
                projected = projected.detach().float()
                projected_passed = record_candidate(projected, start_name + "_feasible", penalty,
                                                    "active_swept_constraint_projection", int(result.nit))
                if projected_passed:
                    current = projected.cpu().numpy().astype(np.float64).ravel()
                    break
            if passed or candidates[-1][0]["external_acceptance"]:
                break
        if stop_on_accept and any(row[0]["external_acceptance"] for row in candidates):
            break
        if tiny:
            break
    valid = [entry for entry in candidates if entry[0]["external_acceptance"]]
    chosen = min(valid, key=lambda entry: (entry[0]["original_objective"],
                                            entry[0]["external_guide_fit"],
                                            entry[0]["start"], entry[0]["phase"])) if valid else None
    return chosen, candidates
