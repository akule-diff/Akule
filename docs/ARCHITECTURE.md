# Model architecture and inference contracts

The unary predicts a single robot's denoising update. `diffuser/models/mpd_v2.py` adapts the official MPD tensor layout into grouped states `[batch, horizon, robot, channel]`. Position and displacement channels are normalized through the benchmark's frozen codec.

`SmoothPairResidual` in `quality_sparse_v1.py` predicts a reusable ordered-pair positional correction with an interior spline basis. `DynamicSupportU` in `quality_dynamic_u_v2.py` chooses directed non-self edges using contextual features before R runs. `SetContextG` in `canonical_set_g.py` assigns signed, temporally parameterized coefficients to those corrections, with symmetric team aggregation. Sparse inference evaluates R only for admitted edges; dense inference admits all non-self directed edges. No U/R/G module changes the frozen prior's weights at inference.

Canonical Weave and SMD use the original 25-step `canonical_n28_rollout.py`. The terminal map pathway uses `mmd_batched_independent_unary.py` and `mmd_corrected_akule_runtime.compose`, with the official search hooks from `mmd_official_search_equivalence.py`. ScaledWeave adds an explicit similarity frame. Downstream hard repair is separate from learned proposal generation.

The public checkpoint files contain inference state dictionaries. Optimization state and training records are omitted; all model tensors are equal to the selected frozen tensors. `checkpoints/manifest.json` specifies the released file checksums, sizes, families, and code version. Normalizers are in `configs/normalizers/` and `artifacts/manifests/MPD_NORMALIZER.json`.
