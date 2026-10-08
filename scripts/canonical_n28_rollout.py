"""Deployed 25-step sparse, dense, and unary MPD proposal paths."""

from __future__ import annotations

import torch

from diffuser.models.quality_dynamic_u_v2 import full_support
from diffuser.utils.quality_sparse_v1 import endpoint_condition


@torch.inference_mode()
def rollout(engine, scene, method="sparse", observer=None):
    if method not in ("sparse", "dense", "unary"):
        raise ValueError(method)
    hard, ends, endpoints = engine.conditions(scene)
    saved = engine.noise(scene)
    x = engine.unary.apply_hard_conditions(saved["initial"].clone(), hard)
    selected = []
    coefficient_magnitudes = []
    for step in reversed(range(25)):
        timestep = torch.tensor([step], device=x.device)
        base, c1, c2 = engine.base_with_grad(x, timestep, hard)
        if observer is not None:
            observer(step, "unary", x=x, base=base)
        if method == "unary":
            composed = base
        else:
            valid = full_support(base)
            gates = (engine.u.logits(base, endpoints, timestep) > 0) * valid if method == "sparse" else valid
            selected.append(gates)
            index = gates.nonzero()
            fields = engine.all_fields_with_grad(base, endpoints, timestep, index)
            composed, coefficients = engine.signed.compose(base, endpoints, timestep, index, fields, gates.to(base))
            if observer is not None:
                observer(step, "composition", base=base, fields=fields,
                         coefficients=coefficients, composed=composed,
                         admitted_edges=len(index))
            if method == "dense":
                coefficient_magnitudes.append(coefficients.detach().abs().mean())
        mean = engine.unary.apply_hard_conditions(c1 * engine.codec.encode(composed) + c2 * x, hard)
        noise = saved["posterior"][24 - step].clone()
        if step == 0:
            noise.zero_()
        x = engine.posterior_fixed(x, mean, timestep, hard, noise)
    return {
        "output": endpoint_condition(engine.codec.decode(x), ends),
        "gates": torch.stack(selected) if selected else None,
        "coefficient_magnitude": (torch.stack(coefficient_magnitudes).mean()
                                  if coefficient_magnitudes else None),
    }
