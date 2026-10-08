# Benchmark reproduction

`python scripts/evaluate.py --help` lists the supported options. Every invocation selects a frozen scene and model family. `--mode unary`, `--mode sparse`, and `--mode dense` retain their distinct interaction behavior. `--repair` requests complete planning where supported.

## Canonical Weave

```bash
python scripts/download_checkpoints.py --env weave
python scripts/evaluate.py --env weave --n 28 --mode sparse --repair --output outputs/weave28
python scripts/evaluate.py --env weave --n 10 --mode sparse --repair --output outputs/weave10
```

The frozen unary is the Empty2D MPD prior; R, U, and G are the Weave checkpoints. The 25-step reverse-diffusion rollout applies interaction-aware composition at every step. The canonical MMD/XECBS repair retains its original implementation. The fixed full benchmark manifest and protocol are under `artifacts/manifests/`. The original `run_final_n28_benchmark.py` and reporter remain available for the full benchmark; inspect their `--help` before starting a long evaluation.

## Basic, Dense, Shelf, Room

```bash
python scripts/download_checkpoints.py --env smd
python scripts/evaluate.py --env room --n 3 --mode sparse --repair --output outputs/room3
```

All four environments share the final map-conditioned unary, composition-selected R, G, and U. The unary is bound to the current scene's obstacle geometry and start/goal context. The frozen reverse-diffusion rollout is followed by SMD-specific root admission, endpoint checks, targeted static repair, and XECBS. The saved 25-scene panels are in `benchmarks/smd/`.

The native SMD collision check uses sampled squared separation below `(sum of radii)^2 - 0.001`. Native collision outcomes must not be substituted for strict swept collision, speed, or workspace checks. The complete-planner cap is 900 seconds.

## Highways and Conveyor

```bash
python scripts/download_checkpoints.py --env highways
python scripts/evaluate.py --env highways --n 3 --mode sparse --repair --output outputs/highways3
```

Each environment uses its own official unary prior and normalization. The final R is shared with SMD; the two maps share their selected G and U. Independent official candidate generation uses 64 candidates per robot, batched in groups of up to 20 robots. Akule corrects the selected candidates **once at t=0**. The official native candidate banks, low-level planner, and search are retained. The terminal correction is not a full interaction-aware reverse rollout. Native velocity units are m/s; the interaction interface uses per-step displacement, with explicit conversion.

The released map pathway uses official targeted cleanup and the official search; it does not add the SMD RRT fallback. Native feasibility and the 60-second planning cap are retained. The full map task list is `benchmarks/mmd/tasks.json`.

## ScaledWeave

```bash
python scripts/download_checkpoints.py --env scaledweave
python scripts/evaluate.py --env scaledweave --n 40 --mode sparse --output outputs/scaled40
```

ScaledWeave uses a separate population-dependent coordinate frame and motion clock. The included inference path loads the EmptyNoWait unary, Weave R/U, and the selected ScaledWeave G, evaluates the 25-step proposal, and converts it back into physical coordinates for validation. Select N=40, 64, 100, 128, or 256. The CLI exposes proposal evaluation for this family.

## Another saved task

Extract one scene object from a panel's `scenes` array into a JSON file, then:

```bash
python scripts/evaluate.py --env shelf --scene my-scene.json --mode sparse --repair --output outputs/my-scene
```

Scene inputs include starts, goals, population, radii, obstacle geometry, the frozen noise seed, and input identity. Use the correct `--env` for the scene. Report every attempt when aggregating success; include timed-out attempts at the benchmark time cap when computing planning latency. Average trajectory quality over successful plans only. The manuscript's published SMD/DGD baseline times are not matched-hardware timings.
