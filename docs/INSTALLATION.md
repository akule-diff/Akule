# Installation

The reference platform is Linux x86-64, Python 3.9, PyTorch 2.7.0 with CUDA 12.8, NumPy 1.23.5, and SciPy 1.10.1. The complete dependency resolution is in `uv.lock`.

```bash
uv sync --locked --extra test
source .venv/bin/activate
python scripts/download_checkpoints.py --env weave
python scripts/evaluate.py --env weave --n 28 --mode sparse --repair
```

Alternatively, create and activate a Python 3.9 virtual environment, then:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[test]'
```

Run commands from the repository root. Keep the `external/`, `configs/`, and `checkpoints/` directories alongside the editable installation. The native planner source is vendored; a wheel containing only the Python package is not the complete artifact.

Checkpoint archives total approximately 91 MB compressed and 125 MB extracted. Python/CUDA dependencies require several GB. The smoke validation used an NVIDIA RTX 5090 with 32 GB VRAM; large populations can require substantially more memory and computation than a three-robot example. The code does not reserve the GPU, and timing-sensitive map evaluations reject concurrent GPU processes.

## CPU inference

```bash
python scripts/download_checkpoints.py --env basic
python scripts/evaluate.py --env basic --n 3 --device cpu --output outputs/basic-cpu
```

The SMD and map proposal paths support CPU execution. The frozen Weave, ScaledWeave, and downstream repair paths require CUDA. The command reports this before beginning a run.

## Rendering

Install FFmpeg through your operating system and ensure `ffmpeg` is on PATH. Matplotlib is included in the Python dependencies.

```bash
python scripts/visualize.py --trajectory outputs/basic-cpu/proposal.npz \
  --scene outputs/basic-cpu/scene.json --output outputs/basic-cpu/demo.mp4
```

## Troubleshooting

- **Checkpoint mismatch:** rerun the downloader. It checks the archive digest before extraction. Do not interchange Weave, SMD, and map G/U weights.
- **Missing native trajectory assets:** download the environment bundle, which includes the native DatasetNormalizer inputs and metadata.
- **CUDA unavailable:** check the NVIDIA driver and the installed PyTorch CUDA wheel. Use CPU proposal inference for a small SMD or map scene.
- **Output already exists:** choose a fresh `--output` directory. The wrapper protects existing result records.
- **No feasible plan:** inspect `metrics.json`, including native status and collision metrics. A proposal can contain conflicts before repair.
- **Native and strict checks disagree:** their predicates differ; native SMD uses a sampled squared-distance tolerance. Swept continuous collision checks are separate diagnostics.
- **First-run latency:** model construction, CUDA initialization, and compilation are distinct from the frozen planner timers. `wall_seconds_including_repair_setup` deliberately includes repair setup.
