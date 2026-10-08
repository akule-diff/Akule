"""Portable generation of signed 0.120 m physical teachers.

The guide is secondary offline guidance. Production runs require a supplied
64-step physical guide per scene; the tiny smoke constructs one solely to
exercise shapes and optimizer plumbing and writes no training teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diffuser.models.quality_dynamic_u_v2 import full_support
from diffuser.utils import quality_sparse_v1 as q
from diffuser.utils.physical_teacher import (
    TARGET, initial_signed_fit, optimize, refine_guide,
)
from diffuser.utils.teacher_corpus import TeacherCorpusWriter, canonical_hash, sha256
from canonical_n28_runtime import Engine, EXPECTED

PARENT_SHA256 = "5ba260884837cc0ac4cbcbc649e9de6f74e354590a4080ac0e69e26d4ed39fb2"
OBJECTIVE_SHA256 = "2555a1e6144ef3a75417c58f799e105966c51e74b2a588c578ae7bc63d4b39ab"
SCALE_SHA256 = "e02868ec71319d5602f6ac6284ab5b6ace64e2ab277fe747e8f22396415637f7"
ROBUST_START_SHA256 = "975681c0d4c4c41b367d3089a7c9175e2e5cdb7b84600829bcbfd94360ad2967"


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def weave_positions(n, phase=0.0, jitter=None):
    angle = 2 * np.pi * np.arange(n) / n + phase
    if jitter is not None:
        angle += jitter
    starts = (0.8 * np.stack((np.cos(angle), np.sin(angle)), -1)).astype("float32")
    dominant = np.abs(starts[:, 0]) > np.abs(starts[:, 1])
    starts[dominant, 0] = np.sign(starts[dominant, 0]) * np.float32(0.87)
    starts[~dominant, 1] = np.sign(starts[~dominant, 1]) * np.float32(0.87)
    goals = starts.copy()
    goals[dominant, 0] *= -1
    goals[~dominant, 1] *= -1
    return starts, goals


def weave_scene(n, ordinal):
    rng = np.random.default_rng(2_860_000 + ordinal)
    for attempt in range(128):
        phase = float(rng.uniform(-0.005, 0.005))
        jitter = rng.uniform(-0.0025, 0.0025, n) * 2 * np.pi / n
        starts, goals = weave_positions(n, phase, jitter)
        i, j = np.triu_indices(n, 1)
        separation = min(np.linalg.norm(starts[i] - starts[j], axis=-1).min(),
                         np.linalg.norm(goals[i] - goals[j], axis=-1).min())
        if separation >= 0.150:
            break
    else:
        raise RuntimeError("Weave endpoint sampling exhausted")
    scene = dict(family=f"MMD-Weave-Boundary-N{n}-v2", population=n,
                 starts=starts.tolist(), goals=goals.tolist(), radii=[0.05] * n,
                 split="train", configuration_id=10_000 + ordinal,
                 noise_seed=12_000_000 + 4 * ordinal,
                 parent_geometry_id=canonical_hash(dict(starts=starts, goals=goals)),
                 construction=dict(phase_radians=phase, angle_jitter_radians=jitter.tolist(),
                                   attempts=attempt + 1,
                                   rule=f"Pinned N={n} Weave Boundary phase/jitter/snap/reflect generator"),
                 endpoint_min_separation_m=float(separation))
    scene["input_sha256"] = sha256(json.dumps(scene, sort_keys=True).encode())
    return scene


def validate_scene(scene, n):
    starts = np.asarray(scene["starts"], dtype=np.float32)
    goals = np.asarray(scene["goals"], dtype=np.float32)
    if starts.shape != (n, 2) or goals.shape != (n, 2):
        raise ValueError("Scene starts/goals must be [num_agents,2]")
    if not np.isfinite(starts).all() or not np.isfinite(goals).all():
        raise ValueError("Nonfinite scene endpoints")
    if len(scene.get("radii", [0.05] * n)) != n:
        raise ValueError("Scene radii length mismatch")
    if any(abs(float(radius) - 0.05) > 1e-9 for radius in scene.get("radii", [0.05] * n)):
        raise ValueError("This frozen physical contract requires radius 0.05 m")
    if np.abs(starts).max() > 0.95 or np.abs(goals).max() > 0.95:
        raise ValueError("Scene endpoints leave the frozen workspace")
    scene.setdefault("population", n)
    scene.setdefault("radii", [0.05] * n)
    scene.setdefault("split", "train")
    scene.setdefault("noise_seed", int(scene["configuration_id"]))
    scene.setdefault("input_sha256", sha256(json.dumps(scene, sort_keys=True).encode()))
    return scene


@torch.no_grad()
def capture_parent_state(engine, scene):
    """The historical dense G-parent proposal until and including its t=0 base."""
    hard, _, endpoints = engine.conditions(scene)
    # The corpus generator made 25 separate CUDA random draws. A single
    # A [25,N,64,4] draw changes Philox ordering and the historical t=0 state.
    generator = torch.Generator(device=engine.device).manual_seed(int(scene["noise_seed"]))
    n = len(scene["starts"])
    saved = dict(initial=torch.randn((1, 64, n, 4), device=engine.device, generator=generator),
                 posterior=torch.stack([torch.randn((n, 64, 4), device=engine.device,
                                                    generator=generator) for _ in range(25)]))
    x = engine.unary.apply_hard_conditions(saved["initial"].clone(), hard)
    for step in reversed(range(25)):
        timestep = torch.tensor([step], device=engine.device)
        base, c1, c2 = engine.base_with_grad(x, timestep, hard)
        support = full_support(base)
        index = support.nonzero()
        values = engine.all_fields_with_grad(base, endpoints, timestep, index)
        composed, _ = engine.signed.compose(base, endpoints, timestep, index, values, support.to(base))
        if step == 0:
            fields = base.new_zeros((1, base.shape[2], base.shape[2], 64, 2))
            fields[tuple(index.T)] = values
            return base.detach(), fields.detach(), endpoints.detach()
        mean = engine.unary.apply_hard_conditions(c1 * engine.codec.encode(composed) + c2 * x, hard)
        noise = saved["posterior"][24 - step].clone()
        x = engine.posterior_fixed(x, mean, timestep, hard, noise)
    raise AssertionError("No t=0 base")


def load_guide(path, n, device):
    with np.load(path, allow_pickle=False) as archive:
        guide = np.asarray(archive["physical"], dtype=np.float32)
    if guide.shape == (64, n, 4):
        guide = guide[None]
    if guide.shape != (1, 64, n, 4):
        raise ValueError("Guide must have shape [1,64,N,4] or [64,N,4]")
    return torch.as_tensor(guide, device=device)


def tiny_guide(scene, device):
    start = torch.as_tensor(scene["starts"], device=device, dtype=torch.float32)
    goal = torch.as_tensor(scene["goals"], device=device, dtype=torch.float32)
    alpha = torch.linspace(0, 1, 64, device=device)[:, None, None]
    position = (1 - alpha) * start + alpha * goal
    return q.coherent_state(position[None])


def generate_orca_guide(scene, binary, directory, engine, spec):
    """Historical RVO2 secondary guide plus 480-step physical refinement."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scene['configuration_id']}.npz"
    if path.exists():
        return path
    n = len(scene["starts"])
    dt = q.DT / 4
    data = f"{n} 1024 {dt} 0.055 0.75 0.5\n"
    data += "\n".join(" ".join(map(str, (*start, *goal)))
                       for start, goal in zip(scene["starts"], scene["goals"])) + "\n"
    result = subprocess.run([str(binary)], input=data, capture_output=True,
                            text=True, timeout=30, check=True)
    values = np.loadtxt(io.StringIO(result.stdout))
    timestamps = values[:, 0]
    raw = values[:, 1:].reshape(-1, n, 4)
    np.testing.assert_allclose(timestamps, np.arange(len(raw)) * dt, atol=1e-8)
    positions = raw[:253:4, :, :2]
    physical = q.coherent_state(torch.tensor(positions, dtype=torch.float64)[None])[0].numpy()
    guide = torch.as_tensor(physical, device=engine.device, dtype=torch.float32)[None]
    endpoints = engine.conditions(scene)[1]
    refined = refine_guide(guide, endpoints, spec, iterations=480)
    np.savez_compressed(path, physical=refined[0].detach().cpu().numpy())
    return path


def generate_one(engine, scene, guide, spec, scale, robust_start=None, tiny=False):
    base, fields, _ = capture_parent_state(engine, scene)
    if tiny:
        initial_target = base
        initial_coeff = base.new_zeros((1, base.shape[2], base.shape[2], 8))
    else:
        initial_target, initial_coeff = initial_signed_fit(
            base, fields, guide, scene, spec, q.spline_matrix(8, device=base.device), scale)
    chosen, attempts = optimize(base, fields, initial_coeff, initial_target, guide,
                                scene, spec, scale, robust_start, tiny=tiny)
    return chosen, attempts, dict(base=base, fields=fields,
                                  initial_target=initial_target, initial_coeff=initial_coeff)


def scenes_from_args(args):
    if args.environment == "weave":
        for ordinal in range(args.start_ordinal, args.start_ordinal + args.num_scenes):
            yield weave_scene(args.num_agents, ordinal)
    else:
        source = Path(args.scenes).resolve()
        for number, line in enumerate(source.read_text().splitlines()):
            if number >= args.num_scenes:
                break
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("weave", "scene-jsonl"), required=True)
    parser.add_argument("--scenes", type=Path, help="JSONL scene provider; records may include guide_npz")
    parser.add_argument("--guide-dir", type=Path, help="Directory of {configuration_id}.npz physical guides")
    parser.add_argument("--orca-binary", type=Path, help="Pinned RVO2 guide binary; used only when no guide NPZ is supplied")
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--num-scenes", type=int, required=True)
    parser.add_argument("--start-ordinal", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-parent", type=Path, default=ROOT / "checkpoints/TEACHER_G_PARENT.pt")
    parser.add_argument("--tiny-smoke", action="store_true", help="two optimizer iterations; no corpus writes")
    args = parser.parse_args()
    # Native MMD construction changes the process working directory.
    if args.scenes is not None:
        args.scenes = args.scenes.resolve()
    if args.guide_dir is not None:
        args.guide_dir = args.guide_dir.resolve()
    if args.orca_binary is not None:
        args.orca_binary = args.orca_binary.resolve()
    args.output = args.output.resolve()
    args.teacher_parent = args.teacher_parent.resolve()
    if args.environment == "scene-jsonl" and args.scenes is None:
        parser.error("--scenes is required for scene-jsonl")
    if not args.tiny_smoke and args.guide_dir is None and args.orca_binary is None and args.environment == "weave":
        parser.error("Full Weave generation requires --guide-dir or --orca-binary")
    objective_path = ROOT / "artifacts/manifests/TEACHER_OBJECTIVE.json"
    scale_path = ROOT / "artifacts/manifests/SIGNED_SCALES.json"
    if digest(objective_path) != OBJECTIVE_SHA256 or digest(scale_path) != SCALE_SHA256:
        raise RuntimeError("Frozen teacher objective/scales hash mismatch")
    manifest = json.loads((ROOT / "checkpoints/MANIFEST.json").read_text())
    components = {entry["component"]: entry for entry in manifest["components"]}
    for component in ("MPD", "R_ft", "G", "U", "teacher_G_parent"):
        entry = components[component]
        path = args.teacher_parent if component == "teacher_G_parent" else ROOT / entry["checkpoint"]
        if digest(path) != entry["sha256"]:
            raise RuntimeError("Checkpoint manifest hash mismatch: " + component)
    if components["teacher_G_parent"]["sha256"] != PARENT_SHA256:
        raise RuntimeError("Historical teacher-parent identity changed")
    for name, expected in EXPECTED.items():
        if digest(ROOT / "checkpoints" / name) != expected:
            raise RuntimeError("Canonical inference checkpoint hash mismatch: " + name)
    spec = q.QualitySpec(**json.loads(objective_path.read_text()))
    scale = float(json.loads(scale_path.read_text())["scales"][0])
    robust_file = ROOT / "artifacts/manifests/TEACHER_ROBUST_START.npz"
    if digest(robust_file) != ROBUST_START_SHA256:
        raise RuntimeError("Frozen teacher robust-start hash mismatch")
    with np.load(robust_file, allow_pickle=False) as archive:
        robust_start = {key: archive[key].copy() for key in archive.files}
    parent = torch.load(args.teacher_parent, map_location="cpu", weights_only=False)
    engine = Engine()
    engine.signed.load_state_dict(parent["model"])
    engine.signed.eval().requires_grad_(False)
    writer = None if args.tiny_smoke else TeacherCorpusWriter(args.output)
    try:
        for scene in scenes_from_args(args):
            scene = validate_scene(scene, args.num_agents)
            if writer is not None and writer.has_scene(scene["configuration_id"]):
                print("SKIP", scene["configuration_id"], flush=True)
                continue
            guide_path = None
            if args.guide_dir is not None:
                guide_path = args.guide_dir.resolve() / f"{scene['configuration_id']}.npz"
            elif scene.get("guide_npz"):
                guide_path = Path(scene["guide_npz"])
                if not guide_path.is_absolute():
                    guide_path = args.scenes.resolve().parent / guide_path
            if guide_path is None and args.orca_binary is not None and not args.tiny_smoke:
                guide_path = generate_orca_guide(
                    scene, args.orca_binary, args.output / "guides", engine, spec)
            guide = tiny_guide(scene, engine.device) if args.tiny_smoke and guide_path is None else load_guide(guide_path, args.num_agents, engine.device)
            chosen, attempts, state = generate_one(engine, scene, guide, spec, scale,
                                                   robust_start=robust_start, tiny=args.tiny_smoke)
            status = "qualified" if chosen is not None else "teacher_failed"
            if chosen is not None:
                record, coefficients, output = chosen
            else:
                record = max(attempts, key=lambda entry: entry[0]["actual_min_swept_m"])[0]
                coefficients, output = None, None
            if output is not None:
                positions = output[..., :2]
                i, j = torch.triu_indices(args.num_agents, args.num_agents, 1, device=positions.device)
                sampled = float(torch.linalg.vector_norm(positions[:, :, i] - positions[:, :, j], dim=-1).amin())
            else:
                sampled = None
            metadata = dict(status=status, quality=record,
                            actual_clearance_m=record["actual_min_swept_m"],
                            sampled_clearance_m=sampled,
                            checkpoint_sha256=dict(MPD=EXPECTED["MPD.pth"],
                                                   R_ft=EXPECTED["R_ft.pt"],
                                                   teacher_G_parent=PARENT_SHA256),
                            objective_sha256=OBJECTIVE_SHA256,
                            scales_sha256=SCALE_SHA256,
                            guide_sha256=digest(guide_path) if guide_path else None,
                            environment=args.environment,
                            optimization=dict(penalties=(2_000, 200_000, 20_000_000, 2_000_000_000),
                                              target_clearance_m=TARGET, tiny_smoke=args.tiny_smoke))
            if writer is not None:
                writer.append(scene, output.detach().cpu().numpy() if output is not None else None,
                              coefficients.detach().cpu().numpy() if coefficients is not None else None,
                              metadata, guide=guide_path)
            print("TEACHER", scene["configuration_id"], status,
                  "sampled", sampled, "swept", record["actual_min_swept_m"],
                  "start", record["start"], "phase", record["phase"], flush=True)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
