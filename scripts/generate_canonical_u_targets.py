"""Physical free-logit target generation for the selected contextual U.

Each condition uses the accepted scalar Adam optimizer. A hard forward gate
determines support; the dense residual basis is built only while taking the
straight-through gradient. Frozen G, R_ft, and MPD are never updated.
"""

from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path

import torch

import canonical_n28_runtime as runtime
from diffuser.models.quality_dynamic_u_v2 import full_support
from diffuser.utils.quality_sparse_v1 import endpoint_condition
from n28_teacher_corpus_dataset import TeacherCorpusDataset
from train_canonical_g import CLEARANCE_WEIGHT, MASTER_SPLIT_SHA, scene_item

DEGREE_WEIGHT = 0.00001885424111516031
CHECKPOINTS = (0, 5, 10, 25, 50)


def physical_loss(output, teacher):
    guidance = torch.nn.functional.huber_loss(output[..., :2], teacher[..., :2], delta=0.05)
    positions = output[..., :2]
    i, j = torch.triu_indices(positions.shape[2], positions.shape[2], 1, device=positions.device)
    distances = torch.linalg.vector_norm(positions[:, :, i] - positions[:, :, j], dim=-1)
    clearance = ((0.120 - distances).clamp_min(0) / 0.01).square().sum() / output.shape[2]
    return guidance + CLEARANCE_WEIGHT * clearance


def rollout_logits(engine, scene, logits, training=False, capture=False):
    hard, ends, endpoints = engine.conditions(scene)
    saved = engine.noise(scene)
    state = engine.unary.apply_hard_conditions(saved["initial"].clone(), hard)
    probabilities, bases = [], []
    for step in reversed(range(25)):
        timestep = torch.tensor([step], device=engine.device)
        base, c1, c2 = engine.base_with_grad(state, timestep, hard)
        if capture:
            bases.append(base.detach().cpu()[0])
        valid = full_support(base)
        step_logits = logits[step][None]
        probability = torch.sigmoid(step_logits.masked_fill(~valid, 0)) * valid
        hard_gate = (step_logits > 0).to(base) * valid
        gates = hard_gate + probability - probability.detach() if training else hard_gate
        field_support = valid if training else hard_gate.bool()
        index = field_support.nonzero()
        fields = engine.all_fields_with_grad(base, endpoints, timestep, index)
        composed, _ = engine.signed.compose(base, endpoints, timestep, index, fields, gates)
        mean = engine.unary.apply_hard_conditions(c1 * engine.codec.encode(composed) + c2 * state, hard)
        noise = saved["posterior"][24 - step].clone()
        if step == 0:
            noise.zero_()
        state = engine.posterior_fixed(state, mean, timestep, hard, noise)
        probabilities.append(probability.sum() / valid.sum())
    return endpoint_condition(engine.codec.decode(state), ends), torch.stack(probabilities).mean(), bases, endpoints


@torch.no_grad()
def metrics(engine, scene, teacher, logits):
    output, _, _, _ = rollout_logits(engine, scene, logits)
    quality = runtime.q.metrics(output.detach().cpu().numpy(), scene["starts"], scene["goals"], engine.spec)
    n = len(scene["starts"])
    off = ~torch.eye(n, dtype=torch.bool, device=engine.device)
    degree = float((logits[:, off] > 0).float().mean() * (n - 1))
    goals = torch.as_tensor(scene["goals"], device=engine.device)
    return {"sampled_0100_conflicts": quality["collision_pair_times"],
            "path_m": quality["mean_path_length"], "gp": quality["gp_physical"],
            "acceleration_rms": quality["acceleration_rms"], "jerk_rms": quality["jerk_rms"],
            "goals_completed_fraction": float((torch.linalg.vector_norm(output[0, -1, :, :2] - goals, dim=-1) <= 0.05).float().mean()),
            "degree": degree, "true_r_ft_savings": 1 - degree / (n - 1),
            "physical_objective": float(physical_loss(output, teacher))}


def sane(value, baseline):
    return value["goals_completed_fraction"] >= baseline["goals_completed_fraction"] and all(
        value[key] <= max(1e-6, baseline[key]) * 1.1
        for key in ("path_m", "gp", "acceleration_rms", "jerk_rms"))


def optimize_condition(engine, entry, variant, scene, teacher):
    n = len(scene["starts"])
    values = torch.full((25, n, n), 0.01, device=engine.device)
    values.diagonal(dim1=-2, dim2=-1).fill_(-20)
    logits = torch.nn.Parameter(values)
    optimizer = torch.optim.Adam([logits], lr=0.03)
    candidates = []
    baseline = None
    for update in range(51):
        if update in CHECKPOINTS:
            row = metrics(engine, scene, teacher, logits.detach())
            candidates.append({"update": update, "metrics": row,
                               "logits": logits.detach().cpu().clone()})
            if baseline is None:
                baseline = row
            if update >= 25 and len(candidates) >= 2:
                previous, current = candidates[-2]["metrics"], row
                stable = (current["sampled_0100_conflicts"] == previous["sampled_0100_conflicts"]
                          and abs(current["true_r_ft_savings"] - previous["true_r_ft_savings"]) < 0.005
                          and abs(current["physical_objective"] - previous["physical_objective"])
                          <= 0.01 * max(1e-6, previous["physical_objective"]))
                useful = (current["sampled_0100_conflicts"] <= baseline["sampled_0100_conflicts"] + 2
                          and current["true_r_ft_savings"] > 0 and sane(current, baseline))
                if stable and useful:
                    break
        if update == 50:
            break
        optimizer.zero_grad(set_to_none=True)
        output, probability, _, _ = rollout_logits(engine, scene, logits, training=True)
        loss = physical_loss(output, teacher) + DEGREE_WEIGHT * probability * (n - 1)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits.diagonal(dim1=-2, dim2=-1).fill_(-20)
    feasible = [row for row in candidates if sane(row["metrics"], baseline)
                and row["metrics"]["sampled_0100_conflicts"] <= baseline["sampled_0100_conflicts"] + 2
                and row["metrics"]["true_r_ft_savings"] > 0]
    selected = max(feasible, key=lambda row: (row["metrics"]["true_r_ft_savings"],
                                              -row["metrics"]["sampled_0100_conflicts"])) if feasible else None
    record = {"scene_id": entry["scene_id"], "geometry_hash": entry["geometry_hash"],
              "qualified_rank": entry["qualified_rank"], "noise_variant": variant,
              "noise_seed": scene["rollout_seed"], "baseline": baseline,
              "candidates": [{"update": row["update"], "metrics": row["metrics"]} for row in candidates],
              "valid": selected is not None}
    if selected is not None:
        chosen = selected["logits"].to(engine.device)
        with torch.no_grad():
            output, _, bases, endpoints = rollout_logits(engine, scene, chosen, capture=True)
        record.update({"selected_update": selected["update"],
                       "selected_metrics": metrics(engine, scene, teacher, chosen),
                       "logits": selected["logits"], "bases": torch.stack(bases),
                       "endpoints": endpoints.detach().cpu()[0]})
    return record


def generate(corpus_dir, output, attempted=256):
    raw = lzma.decompress((runtime.ROOT / "artifacts/manifests/master_split.json.xz").read_bytes())
    if __import__("hashlib").sha256(raw).hexdigest() != MASTER_SPLIT_SHA:
        raise RuntimeError("Master split mismatch")
    split = json.loads(raw)
    train = [row for row in split["scenes"] if row["split"] == "train"]
    train.sort(key=lambda row: __import__("hashlib").sha256(
        ("u-target:" + str(split["seed"]) + ":" + row["geometry_hash"]).encode()).digest())
    dataset = TeacherCorpusDataset(corpus_dir)
    engine = runtime.Engine()
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    manifest = {"schema_version": 1, "master_split_sha256": MASTER_SPLIT_SHA,
                "g0_checkpoint_sha256": runtime.EXPECTED["G.pt"],
                "r_ft_checkpoint_sha256": runtime.EXPECTED["R_ft.pt"],
                "optimizer": {"kind": "Adam", "lr": 0.03, "updates": 50,
                              "initial_logit": 0.01, "hard_st": True,
                              "degree_lambda": DEGREE_WEIGHT},
                "batch_size": 1, "checkpoint_mode": "none", "shards": [],
                "attempted_conditions": 0, "valid_conditions": 0}
    for first in range(0, attempted, 64):
        records = []
        for index in range(first, min(first + 64, attempted)):
            entry = train[index]
            variant = index % 8
            item = scene_item(dataset, entry, variant)
            records.append(optimize_condition(engine, entry, variant, item["scene"],
                                              item["teacher"].to(engine.device)))
        shard_path = Path("shards") / f"targets_{len(manifest['shards']):05d}.pt"
        torch.save({"records": records, "g0_sha256": runtime.EXPECTED["G.pt"]}, output / shard_path)
        manifest["shards"].append({"path": str(shard_path), "sha256": runtime.sha(output / shard_path),
                                   "conditions": len(records), "valid": sum(row["valid"] for row in records)})
        manifest["attempted_conditions"] += len(records)
        manifest["valid_conditions"] += sum(row["valid"] for row in records)
        runtime.write(output / "TARGET_MANIFEST.json", manifest)
        print(json.dumps({"attempted": manifest["attempted_conditions"],
                          "valid": manifest["valid_conditions"]}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempted", type=int, default=256)
    args = parser.parse_args()
    generate(args.corpus, args.output, args.attempted)
