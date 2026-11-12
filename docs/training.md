# Training

Run commands from the repository root after installing the package and sourcing
`configs/paths.env`. Training starts from the hourly, gridded arrays described
in [data preparation](data.md). ERA5 and satellite normalization arrays must be
supplied before training.

## Configuration and outputs

| Stage | Configuration | Default output |
| --- | --- | --- |
| ERA5 autoencoder | `configs/training/era5_ae.yaml` | `$RISDA_MODEL_ROOT/era_autoencoder` |
| Native latent export and statistics | `configs/training/latents.yaml` | `$RISDA_LATENT_DIR` |
| Latent diffusion prior | `configs/training/latent_prior.yaml` | `$RISDA_DENOISER_DIR` |
| Satellite encoder | `configs/training/satellite_encoder.yaml` | `$RISDA_MODEL_ROOT/encoders/<source>` |
| CONVI encoder | `configs/training/convi_encoder.yaml` | `$RISDA_MODEL_ROOT/encoders/convi` |

Defaults train on 2017-01-01 00 UTC through 2024-12-31 23 UTC and select the
best checkpoint using Q1 2025 validation. All intervals are configurable.
Training and validation intervals must not overlap. When changing the dates,
update the four training configurations and the export/statistics intervals
together; keep evaluation separate from training, validation and calibration.

Each training configuration has its own `data` section. It uses the same
ERA5 coordinate conversion as `configs/data.yaml`. Editing
`configs/data.yaml` alone does not change the training configurations.
`data.era5.coordinate=physical` reads physical arrays; `normalized` first
undoes the stored normalization. The resulting field is standardized with
`data.target_mean` and `data.target_std`. All stages must share the frozen
AE's target normalization.

## Autoencoder

```bash
bash run_train.sh ae 8
```

The deterministic convolutional AE maps 69-channel fields to native latents.
Longitude is padded circularly; latitude is padded with zeros. The default
720 × 1440 field becomes a 23 × 47 × 128 latent. Encoder outputs use softclip2
with bound 5. Reconstruction adds Gaussian latent noise with standard deviation
0.01, including during validation. Native export does not add this noise.

The normalized reconstruction MSE uses latitude weight
`cos(latitude) + 1/N_lon`. Within each atmospheric variable, level weights
are pressure divided by the sum of the 13 pressures. Surface weights are
`[1, 0.1, 0.1, 0.1]` for `t2m,u10,v10,msl`. The weighted errors are averaged
over batch, channels and grid points. The optional latent L2 coefficient is
zero by default. Optimization uses SOAP.

## Native latents and statistics

```bash
bash run_train.sh dump 1
bash run_train.sh stats
```

Export loads the AE's `model_best.pth` and writes deterministic, float32 native
latents to `era5.h5`, plus `ae/config.yaml` and `ae/model.pth` for downstream
stages. The default export includes all hours from 2017 through Q1 2025.
Every requested ERA5 hour is required. Temporary output is
`era5.partial.h5`; it becomes `era5.h5` only after all rows are complete.

`stats` computes native channel means and population standard deviations,
pooling time and latent grid positions with float64 accumulation. The default
interval is 2017–2024, and the command rejects dates outside the AE training
interval. It writes `stats.pth` and `stats.json`. Validation/calibration
latents remain available in the HDF5 file but do not enter these statistics.

Changing the AE requires a new latent export, statistics, prior, observation
encoders and calibration. Use a separate `RISDA_LATENT_DIR` for a new AE.

## Latent prior

```bash
bash run_train.sh prior 8
```

The prior uses six consecutive hourly native latents with the fourth frame
at 00/06/12/18 UTC. All six frames must lie inside the selected training or
validation interval. Missing or incomplete windows are excluded.

Before standardization, Gaussian noise with the AE's noise level is added.
Standardization uses the training-only mean and
`sqrt(native_std² + ae_noise²)`. Diffusion time is uniform on [0,1), with
the configured noise schedule. The denoising MSE is weighted by
`1 + 1/(sigma + sigma_min_penalty)²`; the penalty is zero by default.
The saved checkpoint contains the bare ViDiT weights expected by inference.

## Observation encoders

```bash
for source in atms amsua mhs mwts mwhs; do
  bash run_train.sh satellite 8 data.sate_name="$source"
done
bash run_train.sh convi 8
```

Each satellite encoder is trained separately. ERA5 and observations are paired
at the exact UTC hour. The frozen ERA5 decoder converts predicted native
latents into normalized fields. The satellite loss compares these fields with
ERA5 over the original union observation mask, averaged over all 69 channels
and observed pixels. The latent-space loss is disabled.

CONVI uses Cressman-prepared inputs. Its targets are the original four normalized
surface-observation values, and the loss uses their original per-channel masks.
It is not supervised by ERA5 surface values. The ERA5 availability intersection
is retained for time selection. Cressman-expanded support does not add loss
locations. Set `data.icoa_dir=null` in both CONVI training and assimilation
only when excluding ICOADS.

## Batch sizes, validation and restart

The second launcher argument is the number of GPUs on one machine. Gradient
accumulation preserves `train.batch_size_per_step`, the global batch size.
It must be divisible by `GPU_count × train.microbatch`.

| Stage | Updates | Global batch | Per-GPU microbatch | Validation samples |
| --- | --- | --- | --- | --- |
| AE | 100000 | 64 | 2 | 64 |
| Prior | 125000 | 256 | 4 | 32 |
| Each observation encoder | 100000 | 8 | 1 | All paired validation hours |

Logging, saving and validation intervals are in optimizer steps. Defaults are
16/128/256 for the AE, 4/32/64 for the prior, and 50/2000/1000 for the
observation encoders, respectively. AE/prior validation uses a fixed subset
selected with seed 0; validation restores the training RNG afterward.
Distributed validation partitions samples without adding duplicate tail
samples. Training shuffles every epoch with base seed 0 and drops incomplete
distributed microbatches.

Every run writes `config.yaml`, `inputs.json`, `train.log`,
`model_best.pth`, `model_last.pth` and `training_state.pth`.
The best model minimizes validation loss. The training-state checkpoint stores
weights, optimizer, scheduler, mixed-precision scaler, data cursor and per-rank
random states.

```bash
bash run_train.sh ae 8 --resume
bash run_train.sh prior 8 --resume
bash run_train.sh satellite 8 --resume data.sate_name=atms
bash run_train.sh convi 8 --resume
bash run_train.sh dump 1 --resume
```

Resume uses the most recently saved state. Keep the same configuration, input
files and GPU count. Use the same overrides as the initial launch.
Existing outputs are protected against accidental overwrite.
The restart format applies to checkpoints written by these training entry points.

For an initial execution check, stop at an absolute optimizer-step count:

```bash
bash run_train.sh ae 8 --stop-after 2
bash run_train.sh ae 8 --resume
```

`--stop-after` preserves the full configured learning-rate schedule.
`--dry-run` prints the resolved configuration and launch command without
loading data or allocating a model:

```bash
bash run_train.sh satellite 8 --dry-run data.sate_name=mwts
```

A different YAML can be selected with `--config path/to/config.yaml`.
A small CPU check requires `--device cpu train.precision=float32` and a
suitably reduced architecture/data configuration.
