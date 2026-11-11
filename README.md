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

Prepared satellite files are `observations/<source>/**/*YYYYMMDDHH.nc` with variable
`x` shaped `(1, channels, H, W)`. Supply `mean_<source>.npy` and
`std_<source>.npy` in the normalization directory. CONVI uses SURF and ICOADS
files with channels `t2m,u10,v10,slp`, where pressure is in hPa, and the
69-channel `mean_era5.npy` / `std_era5.npy` normalization. Point the surface
directories to the prepared files in the configuration.

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
`(sources, H_lat*W_lat, C_lat)`, typically `(6,23,47,128)`.
The source order must match the configuration. Supply finite, nonnegative values.

Variance must use the diffusion-standardized latent coordinate. For variances
expressed using the raw latent standard deviation, convert externally by
`(raw_std / effective_std)**2`, where
`effective_std = sqrt(raw_std**2 + ae_noise**2)` when the prior uses AE noise.
`var_y_scale` multiplies only the supplied variance.
Reliability contains error scores such as MSE: smaller values have higher
priority. Priorities are fixed across time and sampling steps.

## Run

```bash
bash run_single.sh 8
```

Replace `8` with the GPU count. Preview the command with `--dry-run`.
Override settings with `key=value`:

```bash
bash run_single.sh 8 timeline.start=2025-07-01T00 timeline.end=2025-09-30T18
```

Each anchor uses six hourly frames `[T-3,T-2,T-1,T,T+1,T+2]` and retains frame four.
The observation data must cover the context hours outside the requested bounds.

Outputs are saved under `$RISDA_OUTPUT_ROOT/runs/PPGF+VCLG/`:
`trajectory.pth` contains dates and standardized latents; `era5/*.npy` contains
physical ERA5 fields shaped `(69,H,W)`; `launch/` stores frozen inputs and
`logs/inference.log` stores the run log. Set `save_decoded=false` to save latents only.
Rerun the same command to resume completed anchors.
