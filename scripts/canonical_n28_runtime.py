"""Frozen N=28 planner components used by the final benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffuser.models import mpd_v2
from diffuser.models import quality_dynamic_u_v2 as dynamic
from diffuser.models import quality_sparse_v1 as models
from diffuser.models.canonical_set_g import SetContextG
from diffuser.utils import planned_completion_v3 as completion
from diffuser.utils import quality_sparse_v1 as q

PROJECT = ROOT
MMD_ROOT = Path(os.environ.get("MMD_ROOT", ROOT / "external/mmd")).resolve()
OUT = ROOT / "results/n28_final_benchmark"
DT = 5 / 64
CHUNK = 512
SETUP_SECONDS = 0.0
CHECKPOINTS = ROOT / "checkpoints"
EXPECTED = {
    "MPD.pth": "e461a013971add07290bea58ec0e422019106a998338d01373cce23fae4f325c",
    "R_ft.pt": "c683fa1bf92f885308f55fac4493d204b0ee6c6b1bbc5ad379816f0a3ff929cc",
    "G.pt": "1146473798e2998e414fdf822e39fbea3c5c70182d205ba5987dc9364bdcc523",
    "U.pt": "6952ef0965bbc29e2e242a6818959bf8bacd54c1e6d69bbdb3f8f3571965b601",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def ready(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(ready(value), sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class BudgetExpired(BaseException):
    pass


class Engine:
    """Exact frozen MPD, R_ft, set-context G and contextual U assembly."""

    population = 28
    contract_version = "final-n28-four-method-v1"

    def __init__(self):
        from mpd_v2_runtime import load_official_mpd

        for name, expected in EXPECTED.items():
            if sha(CHECKPOINTS / name) != expected:
                raise RuntimeError("Checkpoint hash mismatch: " + name)
        if not torch.cuda.is_available():
            raise RuntimeError("Final planner requires CUDA")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        self.device = torch.device("cuda")
        self.unary, self.normalizer, _, _ = load_official_mpd(MMD_ROOT, "cuda")
        self.unary.eval().requires_grad_(False)
        self.codec = q.PhysicalCodec(self.normalizer, self.device)
        self.spec = q.QualitySpec()
        self.residual = models.SmoothPairResidual(q.spline_matrix(16, device=self.device)[:, 2:-2]).to(self.device)
        self.residual.load_state_dict(torch.load(CHECKPOINTS / "R_ft.pt", map_location="cpu", weights_only=False)["model"])
        self.residual.eval().requires_grad_(False)
        scales = read(ROOT / "artifacts/manifests/SIGNED_SCALES.json")["scales"]
        self.signed = models.SignedQualityMixer(q.spline_matrix(8, device=self.device), scales).to(self.device)
        self.signed.g = SetContextG(scales).to(self.device)
        self.signed.g.pair_chunk_size = 256
        self.signed.g.load_state_dict(torch.load(CHECKPOINTS / "G.pt", map_location="cpu", weights_only=False)["g"])
        self.signed.eval().requires_grad_(False)
        self.u = dynamic.DynamicSupportU(width=48, pair_residual=True, pair_residual_width=192).to(self.device)
        self.u.load_state_dict(torch.load(CHECKPOINTS / "U.pt", map_location="cpu", weights_only=False)["u"])
        self.u.pair_chunk_size = 256
        self.u.eval().requires_grad_(False)
        self.split = {"train": []}

    def conditions(self, scene):
        starts, goals = [torch.as_tensor(scene[key], device=self.device, dtype=torch.float32) for key in ("starts", "goals")]
        if len(starts) != len(goals):
            raise ValueError("Start/goal population mismatch")
        endpoints = torch.stack([torch.cat([p, torch.zeros_like(p)], -1) for p in (starts, goals)])[None]
        normalized = self.codec.encode(endpoints)
        return {0: normalized[:, 0], 63: normalized[:, 1]}, endpoints, torch.cat([endpoints[:, 0], endpoints[:, 1]], -1)

    def noise(self, scene):
        seed = int(scene["rollout_seed"])
        key = (seed, scene["input_sha256"])
        if getattr(self, "_noise_key", None) != key:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            n = len(scene["starts"])
            initial = torch.randn((1, 64, n, 4), device=self.device, generator=generator)
            posterior = torch.randn((25, n, 64, 4), device=self.device, generator=generator)
            self._noise_key = key
            self._noise_value = {"initial": initial, "posterior": posterior, "seed": seed}
        return self._noise_value

    def base_with_grad(self, x, timestep, hard):
        eps = self.unary(x, timestep)
        mean = self.unary.reverse_mean(x, eps, timestep, hard)
        c1 = self.unary.diffusion.posterior_mean_coef1[timestep][:, None, None, None]
        c2 = self.unary.diffusion.posterior_mean_coef2[timestep][:, None, None, None]
        clean = self.unary.apply_hard_conditions((mean - c2 * x) / c1, hard)
        return self.codec.decode(clean), c1, c2

    def all_fields_with_grad(self, base, endpoints, timestep, index):
        fields = []
        for part in index.split(CHUNK):
            if len(part):
                batch, i, j = part.unbind(-1)
                fields.append(self.residual.specific(base[batch, :, i], base[batch, :, j], endpoints[batch, i], endpoints[batch, j], timestep[batch]))
        return torch.cat(fields) if fields else base.new_empty((0, base.shape[1], 2))

    def posterior_fixed(self, x, mean, timestep, hard, noise):
        batch, _, agents, _ = x.shape
        flat = mpd_v2.grouped_to_mpd_per_agent(x)
        flat_t = timestep.repeat_interleave(agents)
        _, _, log_variance = self.unary.diffusion.q_posterior(x_start=torch.zeros_like(flat), x_t=flat, t=flat_t)
        flat_mean = mpd_v2.grouped_to_mpd_per_agent(mean)
        sampled = flat_mean + torch.exp(0.5 * log_variance) * noise
        return self.unary.apply_hard_conditions(mpd_v2.mpd_per_agent_to_grouped(sampled, batch, agents), hard)

    def dense_rollout(self, scene):
        from canonical_n28_rollout import rollout
        result = rollout(self, scene, method="dense")
        return result["output"], result["coefficient_magnitude"]
