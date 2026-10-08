"""Canonical fresh dense-G transition bootstrap and H=25 refinement.

Requires the external 25k teacher corpus. This is the accepted scalar Stage A
and dense Stage B path; selection never reads the final-test manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import canonical_n28_runtime as runtime
from canonical_n28_rollout import rollout
from diffuser.models import mpd_v2
from diffuser.models.canonical_set_g import SetContextG
from diffuser.models.quality_dynamic_u_v2 import full_support
from diffuser.utils.quality_sparse_v1 import endpoint_condition
from n28_teacher_corpus_dataset import TeacherCorpusDataset

SEED = 28092229
MODEL_SEED = 280918100
SAMPLER_SEED = 280920
CLEARANCE_WEIGHT = 3.9508342635550774e-6
MASTER_SPLIT_SHA = "0b3b2f01d3e4a724ac0d499618357885ffd06ad03f5f9f1152eb0bc72cf8f94c"


def setup(corpus_dir):
    raw = lzma.decompress((runtime.ROOT / "artifacts/manifests/master_split.json.xz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != MASTER_SPLIT_SHA:
        raise RuntimeError("Master split mismatch")
    split = json.loads(raw)
    dataset = TeacherCorpusDataset(corpus_dir)
    if dataset.qualified_count != 25000:
        raise RuntimeError("Expected complete 25k teacher corpus")
    groups = {name: sorted((row for row in split["scenes"] if row["split"] == name),
                           key=lambda row: hashlib.sha256((str(SEED) + ":" + row["geometry_hash"]).encode()).digest())
              for name in ("train", "selection", "shadow")}
    engine = runtime.Engine()
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    scales = runtime.read(runtime.ROOT / "artifacts/manifests/SIGNED_SCALES.json")["scales"]
    engine.signed.g = SetContextG(scales).to(engine.device)
    engine.signed.g.pair_chunk_size = 256
    torch.nn.init.normal_(engine.signed.g.head[-1].weight, std=0.01)
    torch.nn.init.zeros_(engine.signed.g.head[-1].bias)
    engine.signed.g.train().requires_grad_(True)
    calibration = runtime.read(runtime.ROOT / "artifacts/manifests/R_CONFIDENCE_CALIBRATION.json")
    weights = torch.tensor(calibration["weights"], device=engine.device, dtype=torch.float32)
    weights = weights / weights.mean()
    return engine, dataset, groups, weights


def scene_item(dataset, entry, variant):
    item = dataset[8 * entry["qualified_rank"] + variant]
    if item["geometry_hash"] != entry["geometry_hash"]:
        raise RuntimeError("Teacher corpus/split mismatch")
    return item


def optimizer_for(g, stage):
    trunk = list(g.temporal.parameters()) + list(g.relations.parameters())
    identities = {id(parameter) for parameter in trunk}
    head = [parameter for parameter in g.parameters() if id(parameter) not in identities]
    multiplier = 0.1 if stage == "b" else 1.0
    return torch.optim.AdamW([{"params": trunk, "lr": 3e-5 * multiplier},
                              {"params": head, "lr": 1e-4 * multiplier}],
                             weight_decay=1e-5)


def stratified_timesteps(batch_size, update):
    regions = np.array_split(np.arange(25), 3)
    return [int(regions[(update + item) % 3][(update * 17 + item * 7) % len(regions[(update + item) % 3])])
            for item in range(batch_size)]


@torch.no_grad()
def teacher_posterior(engine, clean, noisy, timestep, hard, noise):
    batch, _, agents, _ = noisy.shape
    flat_clean = mpd_v2.grouped_to_mpd_per_agent(clean)
    flat_noisy = mpd_v2.grouped_to_mpd_per_agent(noisy)
    flat_t = timestep.repeat_interleave(agents)
    mean, _, log_variance = engine.unary.diffusion.q_posterior(x_start=flat_clean, x_t=flat_noisy, t=flat_t)
    flat_noise = noise.clone()
    flat_noise[flat_t == 0] = 0
    next_state = mean + torch.exp(0.5 * log_variance) * flat_noise
    return engine.unary.apply_hard_conditions(mpd_v2.mpd_per_agent_to_grouped(next_state, batch, agents), hard)


class TeacherTransitionCache:
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.values = OrderedDict()

    @torch.no_grad()
    def get(self, engine, scene, teacher, key):
        value = self.values.pop(key, None)
        if value is None:
            hard, _, endpoints = engine.conditions(scene)
            saved = engine.noise(scene)
            clean = engine.unary.apply_hard_conditions(engine.codec.encode(teacher), hard)
            current = engine.unary.apply_hard_conditions(
                engine.unary.q_sample(clean, torch.tensor([24], device=engine.device), saved["initial"]), hard)
            states, next_states = {}, {}
            for step in reversed(range(25)):
                states[step] = current
                noise = saved["posterior"][24 - step].clone()
                if step == 0:
                    noise.zero_()
                current = teacher_posterior(engine, clean, current,
                                            torch.tensor([step], device=engine.device), hard, noise)
                next_states[step] = current
            value = {"states": states, "next_states": next_states,
                     "hard": hard, "endpoints": endpoints, "posterior": saved["posterior"]}
        self.values[key] = value
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def transition_batch(engine, conditions, cache):
    samples = []
    for entry, variant, step, scene, teacher in conditions:
        path = cache.get(engine, scene, teacher, (entry["qualified_rank"], variant))
        noise = path["posterior"][24 - step].clone()
        if step == 0:
            noise.zero_()
        samples.append((step, path, noise))
    return {"state": torch.cat([path["states"][step] for step, path, _ in samples]),
            "target": torch.cat([path["next_states"][step] for step, path, _ in samples]),
            "hard": {key: torch.cat([path["hard"][key] for _, path, _ in samples]) for key in samples[0][1]["hard"]},
            "endpoints": torch.cat([path["endpoints"] for _, path, _ in samples]),
            "noise": torch.cat([noise for _, _, noise in samples]),
            "timestep": torch.tensor([step for step, _, _ in samples], device=engine.device)}


def transition_loss(engine, batch, weights):
    state, timestep, hard = batch["state"], batch["timestep"], batch["hard"]
    base, c1, c2 = engine.base_with_grad(state, timestep, hard)
    support = full_support(base)
    index = support.nonzero()
    fields = engine.all_fields_with_grad(base, batch["endpoints"], timestep, index)
    composed, _ = engine.signed.compose(base, batch["endpoints"], timestep, index, fields, support.to(base))
    base_mean = engine.unary.apply_hard_conditions(c1 * engine.codec.encode(base) + c2 * state, hard)
    mean = engine.unary.apply_hard_conditions(c1 * engine.codec.encode(composed) + c2 * state, hard)
    base_next = engine.posterior_fixed(state, base_mean, timestep, hard, batch["noise"])
    predicted_next = engine.posterior_fixed(state, mean, timestep, hard, batch["noise"])
    predicted = engine.codec.decode(predicted_next)
    baseline = engine.codec.decode(base_next)
    target = engine.codec.decode(batch["target"])
    target_delta, predicted_delta = target - baseline, predicted - baseline
    position_scale = target_delta[..., :2].square().mean((1, 2, 3)).detach().clamp_min(1e-6)
    motion_scale = target_delta[..., 2:].square().mean((1, 2, 3)).detach().clamp_min(1e-6)
    position = (predicted_delta[..., :2] - target_delta[..., :2]).square().mean((1, 2, 3)) / position_scale
    motion = (predicted_delta[..., 2:] - target_delta[..., 2:]).square().mean((1, 2, 3)) / motion_scale
    return (weights[timestep] * (position + motion)).mean()


def dense_rollout_train(engine, scene):
    hard, ends, endpoints = engine.conditions(scene)
    saved = engine.noise(scene)
    state = engine.unary.apply_hard_conditions(saved["initial"].clone(), hard)
    for step in reversed(range(25)):
        timestep = torch.tensor([step], device=engine.device)
        noise = saved["posterior"][24 - step].clone()
        if step == 0:
            noise.zero_()

        def one_step(x, t, eps):
            base, c1, c2 = engine.base_with_grad(x, t, hard)
            support = full_support(base)
            index = support.nonzero()
            fields = engine.all_fields_with_grad(base, endpoints, t, index)
            composed, _ = engine.signed.compose(base, endpoints, t, index, fields, support.to(base))
            mean = engine.unary.apply_hard_conditions(c1 * engine.codec.encode(composed) + c2 * x, hard)
            return engine.posterior_fixed(x, mean, t, hard, eps)

        state = checkpoint(one_step, state, timestep, noise, use_reentrant=False)
    return endpoint_condition(engine.codec.decode(state), ends)


def physical_loss(output, teacher):
    guidance = F.huber_loss(output[..., :2], teacher[..., :2], delta=0.05)
    positions = output[..., :2]
    i, j = torch.triu_indices(positions.shape[2], positions.shape[2], 1, device=positions.device)
    distances = torch.linalg.vector_norm(positions[:, :, i] - positions[:, :, j], dim=-1)
    clearance = ((0.120 - distances).clamp_min(0) / 0.01).square().sum() / output.shape[2]
    return guidance + CLEARANCE_WEIGHT * clearance


@torch.no_grad()
def panel(engine, dataset, entries, count):
    engine.signed.g.eval()
    conflicts = []
    for index, entry in enumerate(entries[:count]):
        scene = scene_item(dataset, entry, index % 8)["scene"]
        output = rollout(engine, scene, "dense")["output"].detach().cpu().numpy()
        quality = runtime.q.metrics(output, scene["starts"], scene["goals"], engine.spec)
        conflicts.append(quality["collision_pair_times"])
    engine.signed.g.train()
    return float(np.mean(conflicts))


def save(path, engine, optimizer, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"g": {key: value.detach().cpu() for key, value in engine.signed.g.state_dict().items()},
                "optimizer": optimizer.state_dict(), "record": record}, path)


def train_stage(stage, corpus_dir, output, updates):
    engine, dataset, groups, weights = setup(corpus_dir)
    output.mkdir(parents=True, exist_ok=True)
    latest, best_path = output / f"{stage}_latest.pt", output / f"{stage}_best.pt"
    if stage == "b":
        parent = output / "a_best.pt"
        engine.signed.g.load_state_dict(torch.load(parent, map_location="cpu", weights_only=False)["g"])
    optimizer = optimizer_for(engine.signed.g, stage)
    first, best, best_sentinel, severe_regressions = 0, float("inf"), float("inf"), 0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        engine.signed.g.load_state_dict(saved["g"])
        optimizer.load_state_dict(saved["optimizer"])
        first = saved["record"]["update"]
        best_sentinel = saved["record"].get("best_sentinel", float("inf"))
        severe_regressions = saved["record"].get("severe_regressions", 0)
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=False)["record"]["selection_conflicts"]
    cache = TeacherTransitionCache()
    train = groups["train"]
    order = np.random.default_rng(SAMPLER_SEED).permutation(len(train) * 8)
    for update in range(first + 1, updates + 1):
        if stage == "a":
            flat = int(order[(update - 1) % len(order)])
            rank, variant = divmod(flat, 8)
            entry = train[rank]
            item = scene_item(dataset, entry, variant)
            condition = (entry, variant, stratified_timesteps(1, update)[0],
                         item["scene"], item["teacher"].to(engine.device))
            loss = transition_loss(engine, transition_batch(engine, [condition], cache), weights)
        else:
            cycle, ordinal = divmod(update - 1, 4096)
            entry = train[ordinal]
            variant = (cycle + ordinal) % 8
            item = scene_item(dataset, entry, variant)
            condition = (entry, variant, stratified_timesteps(1, update)[0],
                         item["scene"], item["teacher"].to(engine.device))
            output_state = dense_rollout_train(engine, item["scene"])
            loss = physical_loss(output_state, condition[4])
            loss = loss + 0.001 * transition_loss(engine, transition_batch(engine, [condition], cache), weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.signed.g.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        save_now = update % (25 if stage == "a" else 5) == 0 or update == updates
        sentinel_now = update % (25 if stage == "a" else 10) == 0 or update == updates
        if not save_now and not sentinel_now:
            continue
        record = {"stage": stage, "update": update, "loss": float(loss)}
        if save_now:
            save(output / f"{stage}_{update:06d}.pt", engine, optimizer, record)
        if sentinel_now:
            sentinel = panel(engine, dataset, groups["selection"], 16)
            record["sentinel_conflicts"] = sentinel
            promising = sentinel <= best_sentinel - (2.0 if stage == "a" else 1.0)
            best_sentinel = min(best_sentinel, sentinel)
            full = (promising or update % (500 if stage == "a" else 100) == 0
                    or update == updates)
            if full:
                selection = panel(engine, dataset, groups["selection"], 128)
                record["selection_conflicts"] = selection
                if selection < best:
                    best = selection
                    severe_regressions = 0
                    save(best_path, engine, optimizer, record)
                elif selection > (1.25 if stage == "a" else 1.15) * best:
                    severe_regressions += 1
                else:
                    severe_regressions = 0
                print(json.dumps(record), flush=True)
        record["best_sentinel"] = best_sentinel
        record["severe_regressions"] = severe_regressions
        save(latest, engine, optimizer, record)
        if stage == "a" and update >= 16384 and severe_regressions >= 3:
            break
        if stage == "b" and update >= 4096 and severe_regressions >= 3:
            break
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("a", "b"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int)
    args = parser.parse_args()
    train_stage(args.stage, args.corpus, args.output,
                args.updates or (40960 if args.stage == "a" else 8192))
