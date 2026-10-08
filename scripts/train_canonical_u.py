"""The selected three-sweep U fit from frozen physical support target shards.

The target shards are large training data and are supplied separately. Their
hashes and order are fixed by TARGET_MANIFEST.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

import canonical_n28_runtime as runtime
from diffuser.models.quality_dynamic_u_v2 import DynamicSupportU

SEED = 28092229


def load_examples(directory):
    manifest = json.loads((directory / "TARGET_MANIFEST.json").read_text())
    if manifest["g0_checkpoint_sha256"] != runtime.EXPECTED["G.pt"]:
        raise RuntimeError("U targets use a different G")
    examples = []
    conditions = 0
    for shard in manifest["shards"]:
        path = directory / shard["path"]
        if runtime.sha(path) != shard["sha256"]:
            raise RuntimeError(f"U target shard changed: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["g0_sha256"] != runtime.EXPECTED["G.pt"]:
            raise RuntimeError("U target shard G mismatch")
        for record in saved["records"]:
            if not record["valid"]:
                continue
            conditions += 1
            for position, step in enumerate(reversed(range(25))):
                examples.append((record["bases"][position], record["endpoints"], step,
                                 record["logits"][step]))
            if conditions == 128:
                if len(examples) != 3200:
                    raise RuntimeError("Incomplete target set")
                return examples
    raise RuntimeError(f"Need 128 valid target conditions; found {conditions}")


def balanced_huber(prediction, target):
    n = prediction.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool, device=prediction.device)
    per_edge = F.huber_loss(prediction.masked_fill(~off, 0), target.masked_fill(~off, 0),
                            delta=0.1, reduction="none")
    retained = (target > 0) & off
    removed = (target <= 0) & off
    pos = (per_edge * retained).sum(-1) / retained.sum(-1).clamp_min(1)
    neg = (per_edge * removed).sum(-1) / removed.sum(-1).clamp_min(1)
    pos = pos.sum(-1) / retained.any(-1).sum(-1).clamp_min(1)
    neg = neg.sum(-1) / removed.any(-1).sum(-1).clamp_min(1)
    return (2 * pos + neg).mean()


def train(directory, output, sweeps=3):
    examples = load_examples(directory)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    model = DynamicSupportU(width=48, pair_residual=True,
                            pair_residual_width=192).to(device)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(0.01)
        model.pair_head[-1].weight.zero_()
        model.pair_head[-1].bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    randomizer = random.Random(SEED)
    order = list(range(len(examples)))
    for sweep in range(1, sweeps + 1):
        randomizer.shuffle(order)
        losses = []
        for first in range(0, len(order), 8):
            chosen = [examples[index] for index in order[first:first + 8]]
            base = torch.stack([row[0] for row in chosen]).to(device)
            endpoints = torch.stack([row[1] for row in chosen]).to(device)
            timestep = torch.tensor([row[2] for row in chosen], device=device)
            target = torch.stack([row[3] for row in chosen]).to(device)
            loss = balanced_huber(model.logits(base, endpoints, timestep), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5,
                                            error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss))
        print(json.dumps({"sweep": sweep, "balanced_huber": sum(losses) / len(losses)}),
              flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"u": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "record": {"size": "current", "width": 48,
                           "pair_residual_width": 192, "sweep": sweeps,
                           "train_conditions": 128, "train_examples": 3200,
                           "g0_sha256": runtime.EXPECTED["G.pt"]}}, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=3)
    args = parser.parse_args()
    train(args.targets, args.output, args.sweeps)
