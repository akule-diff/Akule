"""Resumable four-planner N=28 test using native MMD/XECBS semantics."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import canonical_n28_runtime as common
import canonical_mmd_repair as native_module
from canonical_n28_rollout import rollout as canonical_rollout


OUT = common.OUT
MANIFEST = common.ROOT / "artifacts/manifests/FINAL_N28_TEST_MANIFEST_V2.json"
G_PATH = common.CHECKPOINTS / "G.pt"
U_PATH = common.CHECKPOINTS / "U.pt"
G_HASH = "1146473798e2998e414fdf822e39fbea3c5c70182d205ba5987dc9364bdcc523"
U_HASH = "6952ef0965bbc29e2e242a6818959bf8bacd54c1e6d69bbdb3f8f3571965b601"
METHODS = ("sparse", "dense", "unary", "official")
NATIVE_CAP_SECONDS = 240.0
WATCHDOG_SECONDS = 255.0


def sync_seconds():
    torch.cuda.synchronize()
    return time.perf_counter()


def validate():
    if common.sha(G_PATH) != G_HASH or common.sha(U_PATH) != U_HASH:
        raise RuntimeError("Locked G/U checkpoint hash changed")
    value = json.loads(MANIFEST.read_text())
    protocol = json.loads((common.ROOT / "artifacts/manifests/FULL_BENCHMARK_PROTOCOL.json").read_text())
    if common.sha(MANIFEST) != protocol["fixed_test_manifest_sha256"]:
        raise RuntimeError("Final-test manifest hash changed")
    if value["condition_count"] != 1024 or len(value["conditions"]) != 1024:
        raise RuntimeError("Wrong final-test manifest length")
    if len({row["geometry_hash"] for row in value["conditions"]}) != 1024:
        raise RuntimeError("Repeated final-test geometry")
    return value


def setup(backend="XECBS"):
    value = validate()
    engine = common.Engine()
    engine.setup_seconds = 0.0
    engine.contract_version = "final-n28-four-method-v1"
    # Export metadata only; the actual frozen composer remains engine.signed.g.
    engine.mixer = type("GCheckpointReference", (), dict(
        checkpoint_path=str(G_PATH), checkpoint_sha256=G_HASH))()
    engine.u.checkpoint_path = str(U_PATH)
    engine.u.checkpoint_sha256 = U_HASH
    train_scene = json.loads((common.ROOT / "artifacts/manifests/WARMUP_TRAIN_SCENE.json").read_text())
    engine.split = {"train": [train_scene]}
    engine.repair_source_path = None

    def saved_rollout(scene, method="dense", capture=(), profile=False):
        del capture, profile
        if method not in ("sparse", "dense", "unary"):
            raise ValueError(method)
        if engine.repair_source_path is None:
            raise RuntimeError("No validated saved proposal for repair")
        saved = np.load(engine.repair_source_path)
        if saved["input_sha256"].item() != scene["input_sha256"]:
            raise RuntimeError("Repair proposal scene hash differs")
        physical = torch.as_tensor(saved["physical"], device=engine.device)
        return physical, [], dict(
            saved_proposal_path=str(engine.repair_source_path),
            proposal_reused_without_recomputation=True,
            initial_noise_sha256=str(saved["initial_noise_sha256"].item()),
            posterior_noise_sha256=str(saved["posterior_noise_sha256"].item()),
        )

    engine.rollout = saved_rollout
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    common.OUT = OUT
    native_module.CAP = NATIVE_CAP_SECONDS
    native_module.NATIVE_CAP = NATIVE_CAP_SECONDS
    native_module.WATCHDOG = WATCHDOG_SECONDS
    native = native_module.Native(engine, backend=backend)
    return value, engine, native, train_scene


@torch.inference_mode()
def unary_proposal(engine, scene):
    hard, ends, _ = engine.conditions(scene)
    saved = engine.noise(scene)
    x = engine.unary.apply_hard_conditions(saved["initial"].clone(), hard)
    for step in reversed(range(25)):
        timestep = torch.tensor([step], device=x.device)
        base, c1, c2 = engine.base_with_grad(x, timestep, hard)
        mean = engine.unary.apply_hard_conditions(c1 * engine.codec.encode(base) + c2 * x, hard)
        noise = saved["posterior"][24 - step].clone()
        if step == 0:
            noise.zero_()
        x = engine.posterior_fixed(x, mean, timestep, hard, noise)
    return common.q.endpoint_condition(engine.codec.decode(x), ends)


def proposal(engine, scene, method, directory):
    json_path = directory / "proposal.json"
    npz_path = directory / "proposal.npz"
    if json_path.exists() and npz_path.exists():
        saved = json.loads(json_path.read_text())
        if saved["input_sha256"] != scene["input_sha256"] or saved["npz_sha256"] != common.sha(npz_path):
            raise RuntimeError("Saved proposal changed or belongs to another scene")
        return saved, npz_path
    if json_path.exists() != npz_path.exists():
        raise RuntimeError("Incomplete proposal artifact pair")
    start = sync_seconds()
    with torch.inference_mode():
        if method == "sparse":
            result = canonical_rollout(engine, scene, method="sparse")
            physical = result["output"]
            degree = float(result["gates"].sum((-1, -2)).float().mean() / 28)
        elif method == "dense":
            physical, _ = engine.dense_rollout(scene)
            degree = 27.0
        elif method == "unary":
            physical = unary_proposal(engine, scene)
            degree = None
        else:
            raise ValueError(method)
    duration = sync_seconds() - start
    noise = engine.noise(scene)
    array = physical.detach().cpu().numpy()
    quality = common.q.metrics(array, scene["starts"], scene["goals"], engine.spec)
    if array.shape != (1, 64, 28, 4):
        raise RuntimeError("Wrong proposal shape")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "proposal.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, physical=array, input_sha256=scene["input_sha256"],
                            initial_noise_sha256=noise.get("initial_sha256", "corpus-seed:" + str(scene["rollout_seed"])),
                            posterior_noise_sha256=noise.get("posterior_sha256", "corpus-seed:" + str(scene["rollout_seed"])))
    os.replace(temporary, npz_path)
    info = dict(method=method, input_sha256=scene["input_sha256"],
                npz_sha256=common.sha(npz_path), proposal_seconds=duration,
                sampled_0100_conflicts=quality["collision_pair_times"],
                all_goals_completed_fraction=float(np.mean(
                    np.linalg.norm(array[0, -1, :, :2] - np.asarray(scene["goals"]), axis=-1) <= 0.05)),
                quality=quality, degree=degree,
                true_r_ft_savings=(1 - degree / 27) if degree is not None else None)
    common.write(json_path, info)
    return info, npz_path


def common_record(method, condition, scene, native_row, proposal_info, directory):
    root = native_row.get("root_quality")
    final = native_row.get("final_quality")
    native_seconds = float(native_row["complete_total_seconds"])
    proposal_seconds = (float(proposal_info["proposal_seconds"])
                        if proposal_info is not None else float(native_row["root_seconds"]))
    total_seconds = native_seconds + (float(proposal_info["proposal_seconds"])
                                      if proposal_info is not None else 0.0)
    root_path = directory / f"{method}_root.npz"
    if root_path.exists():
        root_array = np.load(root_path)["physical"]
        pre_goal_fraction = float(np.mean(np.linalg.norm(
            root_array[-1, :, :2] - np.asarray(scene["goals"]), axis=-1) <= 0.05))
    else:
        pre_goal_fraction = None
    final_path = directory / f"{method}_final.npz"
    if final_path.exists():
        final_array = np.load(final_path)["physical"]
        final_goal_fraction = float(np.mean(np.linalg.norm(
            final_array[-1, :, :2] - np.asarray(scene["goals"]), axis=-1) <= 0.05))
        all_goals = final_goal_fraction == 1.0
    else:
        final_array = None
        final_goal_fraction = None
        all_goals = False
    sampled_clean = bool(final and final["collision_pair_times"] == 0)
    success = bool(all_goals and sampled_clean and not native_row["timeout"])
    if final_array is not None:
        completion = common.completion.uniform_joint_completion(
            final_array, scene["goals"], common.DT, final_valid=success,
            planning_seconds=total_seconds, tolerance=0.05,
            failure_reasons=[] if success else ["sampled_or_native_final_failure"])
    else:
        completion = dict(scheduled_execution_makespan_seconds=None,
                          request_to_completion_seconds=None,
                          per_agent_goal_arrival_seconds=[None] * 28,
                          mean_agent_goal_arrival_seconds=None,
                          diagnostic_position_arrival_seconds=[None] * 28,
                          goal_arrival_valid=False)
    return dict(method=method, index=condition["index"], scene_id=condition["scene_id"],
                geometry_hash=condition["geometry_hash"], noise_variant=condition["noise_variant"],
                input_sha256=scene["input_sha256"], native_status=native_row["native_status"],
                timeout=bool(native_row["timeout"]), failure=native_row.get("failure"),
                proposal_seconds=proposal_seconds, repair_seconds=float(native_row["repair_seconds"]),
                total_planning_seconds=total_seconds,
                setup_or_injection_seconds=(float(native_row["root_seconds"])
                                            if proposal_info is not None else None),
                ct_expansions=native_row.get("ct_expansions"),
                low_level_calls=native_row.get("low_level_calls"),
                native_conflicts_entering_repair=native_row.get("root_native_conflicts"),
                native_conflicts_remaining=native_row.get("final_native_conflicts"),
                pre_repair_conflicts=root["collision_pair_times"] if root else None,
                post_repair_conflicts=final["collision_pair_times"] if final else None,
                pre_repair_quality=root, final_quality=final,
                pre_repair_goal_completion_fraction=pre_goal_fraction,
                final_goal_completion_fraction=final_goal_fraction,
                all_goals_completed=all_goals, final_success=success,
                degree=proposal_info["degree"] if proposal_info is not None else None,
                true_r_ft_savings=(proposal_info["true_r_ft_savings"]
                                   if proposal_info is not None else None),
                completion=completion,
                native_record=str(directory / f"{method}.json"))


def rebuild_smoke(value):
    for method in METHODS:
        for condition in value["conditions"][:16]:
            directory = OUT / "smoke" / method / f"{condition['index']:04d}"
            native_path = directory / f"{method}.json"
            if not native_path.exists():
                raise RuntimeError(f"Smoke method/scene incomplete: {native_path}")
            native_row = json.loads(native_path.read_text())
            proposal_info = (json.loads((directory / "proposal.json").read_text())
                             if method != "official" else None)
            record = common_record(method, condition, condition["scene"],
                                   native_row, proposal_info, directory)
            common.write(directory / "benchmark.json", record)
    print("Rebuilt 16 x 4 smoke summaries from saved native and proposal artifacts")


def run_method(value, engine, native, method, count, *, smoke):
    warm = engine.split["train"][0]
    # Warm the learned/unary proposal path without touching final-test scenes.
    if method in ("sparse", "dense", "unary"):
        warm_dir = OUT / "warmup" / method
        _, warm_path = proposal(engine, warm, method, warm_dir)
        engine.repair_source_path = warm_path
        native.request(warm, method, warm_dir)
    else:
        native.request(warm, "official", OUT / "warmup/official")
    target_root = OUT / ("smoke" if smoke else method)
    if smoke:
        target_root = OUT / "smoke" / method
    for condition in value["conditions"][:count]:
        scene = condition["scene"]
        directory = target_root / f"{condition['index']:04d}"
        target = directory / "benchmark.json"
        if not smoke and condition["index"] < 16:
            smoke_directory = OUT / "smoke" / method / f"{condition['index']:04d}"
            smoke_target = smoke_directory / "benchmark.json"
            if smoke_target.exists():
                prior = json.loads(smoke_target.read_text())
                if (prior["input_sha256"] != scene["input_sha256"]
                        or prior["scene_id"] != condition["scene_id"]):
                    raise RuntimeError("Smoke scene cannot be reused for full test")
                if not directory.exists():
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    directory.symlink_to(smoke_directory, target_is_directory=True)
                continue
        if target.exists():
            old = json.loads(target.read_text())
            if old["input_sha256"] != scene["input_sha256"] or old["scene_id"] != condition["scene_id"]:
                raise RuntimeError("Saved scene benchmark mismatched manifest")
            continue
        if method in ("sparse", "dense", "unary"):
            proposal_info, path = proposal(engine, scene, method, directory)
            engine.repair_source_path = path
        else:
            proposal_info = None
            engine.repair_source_path = None
        native_row = native.request(scene, method, directory, profile=False)
        record = common_record(method, condition, scene, native_row, proposal_info, directory)
        common.write(target, record)
        print(json.dumps(dict(method=method, smoke=smoke,
                              completed=condition["index"] + 1, target=count,
                              status=record["native_status"],
                              pre=record["pre_repair_conflicts"],
                              final=record["post_repair_conflicts"],
                              planning_seconds=round(record["total_planning_seconds"], 3))),
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS + ("all",), default="all")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild-smoke", action="store_true")
    args = parser.parse_args()
    if args.rebuild_smoke:
        rebuild_smoke(validate())
        return
    if args.smoke and args.count != 16:
        raise ValueError("Smoke must use exactly 16 scenes")
    if not args.smoke and args.count not in (512, 1024):
        raise ValueError("Full benchmark must use 512 or 1024 scenes")
    value, engine, native, _ = setup()
    methods = METHODS if args.method == "all" else (args.method,)
    for method in methods:
        run_method(value, engine, native, method, args.count, smoke=args.smoke)


if __name__ == "__main__":
    main()
