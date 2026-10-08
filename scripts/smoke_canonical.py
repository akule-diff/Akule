"""Small checkpoint, inference, and optional MMD repair smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import canonical_n28_runtime as runtime
from canonical_n28_rollout import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    scene = json.loads((runtime.ROOT / "artifacts/manifests/WARMUP_TRAIN_SCENE.json").read_text())
    engine = runtime.Engine()
    for method in ("sparse", "dense", "unary"):
        result = rollout(engine, scene, method)
        output = result["output"]
        if output.shape != (1, 64, 28, 4) or not torch.isfinite(output).all():
            raise RuntimeError(f"Invalid {method} proposal")
        print(json.dumps({"method": method, "shape": list(output.shape),
                          "conflicts": runtime.q.metrics(output.detach().cpu().numpy(), scene["starts"], scene["goals"])["collision_pair_times"]}), flush=True)
    if args.repair:
        import run_final_n28_benchmark as benchmark

        _, engine, native, scene = benchmark.setup()
        directory = runtime.OUT / "smoke_validation_selfcontained" / "sparse"
        _, proposal_path = benchmark.proposal(engine, scene, "sparse", directory)
        engine.repair_source_path = proposal_path
        record = native.request(scene, "sparse", directory)
        if "native_status" not in record or "repair_seconds" not in record:
            raise RuntimeError("Incomplete MMD repair result")
        print(json.dumps({"repair_status": record["native_status"],
                          "repair_seconds": record["repair_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
