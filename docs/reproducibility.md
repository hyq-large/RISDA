# Reproducing PPGF+VCLG

## Scope

This package provides data normalization, observation preprocessing, ERA5 AE
training, native latent export and statistics, latent prior training, observation
Encoder training, residual calibration, PPGF+VCLG inference and ERA5 evaluation.
Run training stages with `bash run_train.sh <stage> <GPU_count>`; see
[training commands and configuration](training.md). Pretrained models can also
be supplied directly.

The example calibration interval is Q1 2025 at hourly resolution. The example
evaluation interval is Q2 2025 at six-hour resolution. Both are configurable.
The calibrator rejects intervals overlapping the evaluated analysis times.
The first six-frame analysis window includes three context hours before the
first analysis; the last includes two hours after the final analysis.

These example intervals do not describe every experiment or every checkpoint's
model-selection history. Use the configurations and artifacts associated with
the particular result being reproduced.

## Required artifacts

```text
models/era_latents/
  stats.pth
  era5.h5
  ae/config.yaml
  ae/model.pth
  denoisers/prior/config.yaml
  denoisers/prior/model_best.pth
models/encoders/<source>/
  config.yaml
  model_best.pth
models/variance.pth
models/reliability.pth
```

The HDF5 file is required for prior training and calibration. It contains native ERA5 latent
states `latents: (time,H_lat*W_lat,C_lat)` and UTC dates
`dates: (time,4)` with year, month, day and hour. These are deterministic ERA5
encoder outputs before latent standardization, with no injected latent noise.
`stats.pth` contains the corresponding native `mean` and `std`.

For the supplied architecture, the ERA5 field shape is `(69,720,1440)` and the
latent shape is `(23,47,128)`. All encoders, latent statistics and the prior
must use the same ERA5 latent coordinate. Use the supplied model configurations
when loading trained weights.

The source archive contains no weights, normalization arrays, observation files
or ERA5 fields. A usable review submission must additionally provide anonymous
access to those artifacts, or their exact acquisition/preparation instructions.
A small example should include one complete six-hour observation window,
same-time ERA5 truth, all required model files and the two uncertainty tensors.

## Calibration

Run `python -m risda.calibrate`. It uses the sources, data paths and models
from the assimilation configuration. All hourly candidates in the configured
calibration interval are considered, including hours outside 00/06/12/18 UTC.

For a supported latent position, let

```text
e = (Encoder(observation) - ERA5_native_latent) / native_latent_std
bias = mean(e)
MSE = mean(e²)
variance = max(MSE - bias², 0)
```

The denominator is the population count N. Spatial statistics pool time at
each source/location/channel. Channel statistics pool observed locations and
times within each source. Only the mask of original observations contributes.
A missing file or zero-support hour contributes no sample. At positions with
fewer than two samples, use the same source's channel variance and MSE.

Spatial forwards use float32. Satellite channel fallback statistics use
float16; CONVI channel fallback statistics use float32. Accumulation uses
float64. These settings are explicit in `configs/calibration.yaml`.

The variance supplied to inference must be in the diffusion-standardized
coordinate:

```text
effective_std² = native_std² + ae_noise²
variance_input = variance * native_std² / effective_std²
```

The noise term is zero when the prior configuration disables AE noise injection.
Inference subsequently applies `var_y_scale` and the variance floor.
The reliability tensor stores MSE in the native-std-standardized coordinate;
smaller MSE means higher reliability. Multiplying all sources at a given
location/channel by the same positive coordinate factor leaves their order
unchanged.

Both files contain `sources` in the configured source order and a tensor named
`variance` or `reliability`, shaped `(sources,H_lat,W_lat,C_lat)` or
`(sources,H_lat*W_lat,C_lat)`. The tensors remain valid external interfaces;
calibration is optional when the appropriate files are already supplied.

The calibration directory contains per-source moments, counts, fallback masks,
used/missing/empty hours, input identities and a summary. Calibration runs use
the configured output directory and recompute source statistics when launched.

## Hyperparameters

The runnable configurations in `configs/training` specify the architecture,
losses and optimization settings. All stages use base seed 0.

| Component | Architecture and optimization |
| --- | --- |
| ERA5 AE | 69 channels; resize 32; latent channels 128; widths 128/256/384/512/768/1024; three blocks per stage; latent noise 0.01 |
| AE optimization | SOAP; learning rate 3e−5; betas 0.9/0.99/0.99; 100000 updates; global batch 64; microbatch 2; linear decay; gradient clip 1 |
| Latent prior | ViDiT; hidden width 1024; 16 blocks; 8 heads; temporal patch 2; six hourly frames |
| Prior optimization | AdamW; learning rate 1e−4; betas 0.9/0.99; 125000 updates; global batch 256; microbatch 4; gradient clip 1 |
| Observation encoders | Widths 64/128/192/256/384/512; two blocks per stage; frozen ERA5 decoder |
| Encoder optimization | AdamW; learning rate 3e−5; betas 0.9/0.99; 100000 updates; global batch 8; microbatch 1; linear decay; gradient clip 1 |

Weight decay is zero in these configurations. The AE reconstruction loss uses
latitude and pressure-level weights. Satellite encoders minimize masked
reconstruction error against ERA5 at observation locations. The CONVI encoder
uses the original surface-observation values and per-channel masks as its
reconstruction targets. No additional latent loss is enabled.

The defaults use training dates in 2017–2024 and Q1 2025 validation. The prior
uses six adjacent hourly frames with the fourth frame at a six-hour analysis
time. Model configurations supplied with a particular checkpoint are the
authority for its architecture and training settings.

Inference defaults are 64 sampling steps, two predictor-corrector corrections
with amplitude 0.1, two moment-matching solver iterations, variance scale 1,
variance floor 1e−8 and float16 precision. The prior's noise schedule and bounds
are read from its model configuration.

## Seeds

All stages use base seed 0. Analysis i uses seed `0 + i`, where i is the
zero-based index in the configured timeline. GPU ranks partition these indices;
the per-analysis seed does not depend on rank. Changing the timeline's start
changes indices and therefore the seeds of overlapping dates. For paired
comparisons, retain the same complete timeline and base seed.

PPGF sorting is fixed and stable, with deterministic source-order tie breaking.
The implementation traverses decreasing MSE and replaces conflicting accumulated
gradients when a more reliable source is encountered. No stochastic source
ordering is used. CUDA kernels and mixed precision need not be bitwise identical
across hardware/software versions.

## Evaluation

Run `python -m risda.evaluate --run-dir <run-directory>` after inference with
`save_decoded=true`. Evaluation requires a completed trajectory and exactly
matching ERA5 hours; missing fields are errors. The evaluator pools weighted
squared errors across all configured analyses and grid cells, then takes one
square root per variable. It uses physical units and cosine latitude weights.
No average of per-day or per-frame RMSE values is used.

`metrics/era69_rmse.json` includes the sums, weights, time axis and hashes of
the compared arrays. `era69_rmse.log` and `.csv` provide the raw RMSE values.
Changing the evaluation timeline requires a separate inference output directory.

## Anonymous distribution

Build `dist/risda-anonymous.zip` with `python -m risda.package`. The packager
uses an explicit source-file allowlist, rejects symlinks and scans included text
for common machine paths and email addresses. The ZIP does not contain Git
history, local environment files, models, data or outputs, and includes a file
checksum manifest.

Distribute weights/data separately through an anonymous location. Inspect their
embedded configuration and metadata for author names, account paths and service
links before distribution. The source ZIP's automatic scan does not inspect
external model or data files.
