# Frozen model assets

Download one environment or all released assets:

```bash
python scripts/download_checkpoints.py --env weave
python scripts/download_checkpoints.py --env basic
python scripts/download_checkpoints.py --env scaledweave
python scripts/download_checkpoints.py --all
```

`manifest.json` records each archive's version, download URL, byte count, SHA256, and required code version, as well as the individual inference model identities. Shared dependencies are resolved automatically. Archives are distributed separately from ordinary Git blobs using Git LFS. Git LFS is not required by the Python downloader.

| Bundle | Contents |
|---|---|
| Weave | Frozen MPD, R, U, G; Empty2D native assets |
| SMD | Shared Basic/Dense/Shelf/Room unary, R, U, G; EmptyNoWait native assets |
| Maps | Shared final Highways/Conveyor G and U; R supplied by the SMD bundle |
| Highways | Environment-specific official prior, configuration, native normalization data |
| Conveyor | Environment-specific official prior, configuration, native normalization data |
| ScaledWeave | Selected G; shared unary/R/U dependencies resolved through the manifest |

Archives are checked before installation, and archive paths are validated before any files are extracted. Downloads use temporary files and atomic replacement. Existing verified cached archives are reused. For offline installation, copy the archives to a cache directory and run:

```bash
python scripts/download_checkpoints.py --all --offline --cache /path/to/archives
```

The inference exports omit optimizer state and internal training records while preserving every selected model tensor. Upstream unary state dictionaries are retained. Normalization statistics and architecture settings must be used with their corresponding benchmark family.
