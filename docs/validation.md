# Validation

Verified on 2026-11-09 from the isolated source checkout:

- **198 regression tests passed**, covering fusion, sampling, exact-hour data
  pairing, normalization, observation masks, calibration caches, ERA5 AE save/reload,
  saved-run contracts and metrics. Tests for removed experiment/surface-split
  entrypoints were replaced with tests of the public workflow.
- Calibration tests write complete bundles for the six-source default and an
  arbitrary non-quarter range with MWTS/MWHS/CONVI, then load them through the
  actual inference routines. They verify variance coordinate conversion,
  squared-bias inclusion in MSE, and fallback at zero/single-count positions.
- PPGF_SD is checked with spatial likelihood variance and channel-only ordering.
  A separate test compares the retained reverse-priority implementation with
  the paper's increasing-MSE, stop-at-first-conflict rule.
- Time-axis tests cover default 2160 calibration hours and 364 test anchors,
  cross-year calibration, leap-day evaluation, custom cadence, subset seed
  indices, invalid intervals and rejection of mismatched/incomplete caches.
- A synthetic completed run is validated and decoded once per anchor. Its
  ERA69 and TEMP physical RMSEs and counts are checked. Empty TEMP channels
  remain NaN/null, and incomplete evaluations are rejected.
- **24 command checks passed**, including CLI help, all six paper method
  presets, the spatial additive ablation, custom dates/cadence, all training
  stages at C=128/512/1024, two-node AE launch construction, ATMS/MWTS/MWHS
  forward-model launchers and shell syntax. These checks use dry runs.
- `python -m risda check` validates 16 imports from this checkout and all six
  method configurations. Static local import closure, undefined-name/syntax
  checks, duplicate-key checks and private-path scans pass.

Environment: Python 3.11, PyTorch 2.5.1+cu124, xFormers 0.0.28.post3, NumPy
1.26.3, pandas 2.2.0, SciPy 1.14.1, xarray 2025.9.0, h5py 3.12.1, OmegaConf
2.3.0 and pytest 8.2.2. The suite reports existing pandas/W&B deprecations and
PyTorch's nested-tensor warning. Fresh installation of all dependency ranges
has not been tested.

These are software checks with synthetic fixtures. No full training,
real-data calibration or diffusion evaluation was launched for this revision.
Published scores, GPU sizing and cross-node communication have not been
revalidated in the exported checkout. Supply prepared observations, matching
checkpoints/configs, normalization arrays and compatible CUDA dependencies for
scientific runs.

The source archive is checked for file hashes, private paths and namespace
consistency. The supplied reference
manuscript remains local and is excluded from the code-only archive.
