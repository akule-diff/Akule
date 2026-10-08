# Training overview

Inference requires only the released checkpoint bundles and saved scenes. Retraining is a separate offline workflow requiring teacher corpora; the inference bundles do not contain those corpora.

The scientific sequence is:

1. Freeze the unary prior and its normalization. Fit R to matched pairwise trajectory-optimization corrections.
2. Optimize joint physical trajectories using the frozen R correction basis to obtain composition teachers.
3. Train G on physical transition targets, then refine with complete denoising rollouts.
4. Optimize sparse-support targets with the frozen unary/R/G stack and train U on the selected supports.
5. Select checkpoints on disjoint held-out panels before benchmark evaluation.

## Included canonical entry points

```bash
python scripts/train_canonical_g.py --help
python scripts/generate_canonical_u_targets.py --help
python scripts/train_canonical_u.py --help
python scripts/generate_physical_teachers.py --help
```

These scripts preserve the canonical G and U training implementations and the teacher objective. They require compatible teacher-corpus manifests and shards, not just the inference checkpoints. Their help output specifies the actual arguments. The physical teacher source also requires its parent teacher checkpoint and guide assets. Optional ORCA guide generation uses RVO2 and the supplied C++ driver; RVO2 is not required for inference.

The reference G recipe uses physical transition fitting followed by full 25-step rollout refinement. U targets are learned from physical objectives rather than a prescribed ground-truth graph. Signed coefficient scales and R confidence calibration are retained under `artifacts/manifests/`.

This release provides the final R, map-conditioned unary, map G/U, and ScaledWeave G weights for inference. Environment-specific data generation and full retraining are not exposed as a one-command pipeline. Do not train on the released evaluation panels.
