# RISDA

Inference code for **RISDA: Reliability-Informed Score-based Data Assimilation of Multi-Source Atmospheric Observations**, using PPGF+VCLG.

## Setup

Use Python 3.11 with CUDA-compatible PyTorch 2.5.1.

```bash
python -m pip install -e .
cp configs/paths.env.example configs/paths.env
```

Edit the paths in `configs/paths.env`, then load them:

```bash
source configs/paths.env
```

Provide the pretrained prior, ERA5 decoder and observation encoders:

```text
models/era_latents/stats.pth
models/era_latents/ae/config.yaml
models/era_latents/ae/model.pth
models/era_latents/denoisers/prior/config.yaml
models/era_latents/denoisers/prior/model_best.pth
models/encoders/<source>/config.yaml
models/encoders/<source>/model_best.pth
```

Configure paths, sources and dates in [configs/assimilation.yaml](configs/assimilation.yaml).
The sources are ATMS, AMSU-A, MHS, MWTS, MWHS and CONVI.

## Variance and reliability inputs

Set `uncertainty.variance` and `uncertainty.reliability` to two PyTorch files.
Save your supplied tensors as:

```python
import torch

sources = ["atms", "amsua", "mhs", "mwts", "mwhs", "convi"]
torch.save({"sources": sources, "variance": variance.cpu()}, "variance.pth")
torch.save({"sources": sources, "reliability": reliability.cpu()}, "reliability.pth")
```

Both tensors have shape `(sources, H_lat, W_lat, C_lat)` or
`(sources, H_lat*W_lat, C_lat)`.


## Run

```bash
bash run_single.sh 8
```

Replace `8` with the GPU count. Preview the command with `--dry-run`.
Override settings with `key=value`. For example,

```bash
bash run_single.sh 8 timeline.start=2025-07-01T00 timeline.end=2025-09-30T18
```

