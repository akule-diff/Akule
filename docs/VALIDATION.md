# Release validation

Validation checks the released implementations and assets; it does not rerun the full paper benchmark suite.

- Clean Python 3.9 environment installed with `uv sync --locked --extra test`.
- **37 tests passed** with `python -m pytest -q`. Coverage includes frozen checkpoint contracts, selector edges and empty selections, composer shapes, permutation equivariance, sparse/dense evaluation, collision checks, saved scenes, safe archive extraction, and checksum failures.
- **12 CUDA smoke commands completed**: Weave N=28 unary, dense, and sparse with repair; Basic N=3 unary, dense, and sparse with repair; Dense, Shelf, and Room N=3 sparse; Highways and Conveyor N=3 sparse with repair; ScaledWeave N=40 sparse proposal inference.
- Complete-planner smoke runs returned native `SUCCESS` for Weave, Basic, Highways, and Conveyor. A separate fresh-environment Weave N=28 run also returned `SUCCESS` with zero final native conflicts.
- CPU proposal inference passed for Basic and Highways. Basic also passed in the clean installed environment.
- All six checkpoint archives passed anonymous public download, SHA256/size checks, and extraction. A fresh public checkout passed all 37 tests and CPU proposal inference using these downloaded assets.
- The live GitHub Pages website passed Chromium playback and gallery checks for all eight environments at 1440, 768, and 390 pixel widths, with no horizontal overflow or JavaScript errors. Citation copying and reduced-motion behavior were checked.

Run the lightweight suite after downloading the model assets:

```bash
python scripts/download_checkpoints.py --all
python -m pytest -q
```

Representative GPU commands:

```bash
python scripts/evaluate.py --env weave --n 28 --mode sparse --repair --output outputs/verify-weave
python scripts/evaluate.py --env basic --n 3 --mode sparse --repair --output outputs/verify-basic
python scripts/evaluate.py --env highways --n 3 --mode sparse --repair --output outputs/verify-highways
python scripts/evaluate.py --env conveyor --n 3 --mode sparse --repair --output outputs/verify-conveyor
```

The GPU validation platform was Linux with an NVIDIA RTX 5090, PyTorch 2.7.0, and CUDA 12.8. These are functional smoke checks, not replacements for the manuscript's frozen timing measurements. An exit code of zero means evaluation completed; inspect the saved `success` field and native planner status to determine feasibility. Rendering was checked with the released visualization script and FFmpeg.
