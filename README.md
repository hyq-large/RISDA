# RISDA

Code for **RISDA: Reliability-Informed Score-based Data Assimilation of Multi-Source Atmospheric Observations**, using PPGF+VCLG.

## Setup

Run the commands below from the repository root. Use Python 3.11 and
CUDA-compatible PyTorch 2.5.1.

```bash
python -m pip install -e .
cp configs/paths.env.example configs/paths.env
```

Edit the paths in `configs/paths.env`, then load them:

```bash
source configs/paths.env
```

## Data preparation

Inputs are hourly ERA5 fields on a 720 × 1440 grid and gridded observations
from ATMS, AMSU-A, MHS, MWTS, MWHS and CONVI. Pair observations with ERA5
at the exact UTC hour. Supply the ERA5 and satellite normalization arrays
described in [data preparation](docs/data.md).

Satellite brightness temperatures are standardized per channel. Missing values
become zero after normalization, with a separate observation mask and the
configured zenith-angle input. CONVI uses ERA5 mean/std for its four surface
variables, converts pressure from hPa to Pa, and gives SURF precedence over
collocated ICOADS values. Cressman interpolation prepares the CONVI encoder
input; loss and guidance masks retain the original observation support.

Training performs normalization when reading the data. To separately convert
existing gridded ERA5 arrays, edit [configs/data.yaml](configs/data.yaml), then run:

```bash
python -m risda.preprocess --start 2025-01-01T00 --end 2025-03-31T23 --output data/era5_normalized
```

This command uses supplied statistics. Raw-data retrieval and initial observation
gridding are described under the [processing boundary](docs/data.md#processing-boundary).

## Training

Edit the data paths and dates in [configs/training](configs/training), then
run these stages in order:

```bash
bash run_train.sh ae 8
bash run_train.sh dump 1
bash run_train.sh stats
bash run_train.sh prior 8

for source in atms amsua mhs mwts mwhs; do
  bash run_train.sh satellite 8 data.sate_name="$source"
done
bash run_train.sh convi 8
```

The stages train the ERA5 autoencoder, export native latents, compute latent
statistics, train the diffusion prior, and train the six observation encoders.
Replace `8` with the GPU count on one machine. Export uses one GPU; statistics
run on CPU. All stages use base seed **0**.

| Stage | Configuration | Default output |
| --- | --- | --- |
| ERA5 AE | `era5_ae.yaml` | `$RISDA_MODEL_ROOT/era_autoencoder` |
| Latent export and statistics | `latents.yaml` | `$RISDA_LATENT_DIR` |
| Diffusion prior | `latent_prior.yaml` | `$RISDA_DENOISER_DIR` |
| Satellite encoders | `satellite_encoder.yaml` | `$RISDA_MODEL_ROOT/encoders/<source>` |
| CONVI encoder | `convi_encoder.yaml` | `$RISDA_MODEL_ROOT/encoders/convi` |

The YAML files are under `configs/training/`. Each has its own data settings;
editing `configs/data.yaml` alone does not change them. Defaults use 2017–2024
for training and Q1 2025 for checkpoint selection. Latent export includes both
periods; latent normalization statistics use only the training period.
The output directories match the default calibration and inference paths.

To resume an interrupted stage, repeat its command with `--resume`:

```bash
bash run_train.sh ae 8 --resume
bash run_train.sh prior 8 --resume
bash run_train.sh satellite 8 --resume data.sate_name=atms
bash run_train.sh convi 8 --resume
bash run_train.sh dump 1 --resume
```

Keep the same GPU count, configuration and input files. Resume restores the
latest saved training state, including the optimizer and data cursor.
Each training directory contains `config.yaml`, `train.log`,
`model_best.pth`, `model_last.pth` and `training_state.pth`.

Use `--config path/to/config.yaml` to select another configuration,
`--dry-run` to print the command and resolved settings, or `--stop-after 2`
to save and stop after two optimizer steps. The latter preserves the full
learning-rate schedule for subsequent resume. See [training details](docs/training.md)
for losses, batch sizes and hyperparameters.

Pretrained models can be supplied using the
[artifact layout](docs/reproducibility.md#required-artifacts), allowing training
to be skipped.

## Calibration

Edit [configs/calibration.yaml](configs/calibration.yaml), then run:

```bash
python -m risda.calibrate
```

Defaults consider every hourly candidate in Q1 2025. The output contains
centered Encoder residual variance for VCLG and residual MSE for PPGF
reliability. Missing observations do not contribute. Precomputed tensors can
also be supplied through `uncertainty.variance` and `uncertainty.reliability`;
see the [tensor definitions](docs/reproducibility.md#calibration).

## Inference and evaluation

Configure paths, sources and dates in
[configs/assimilation.yaml](configs/assimilation.yaml). Defaults evaluate
Q2 2025 at 00/06/12/18 UTC.

```bash
bash run_single.sh 8
python -m risda.evaluate --run-dir "$RISDA_OUTPUT_ROOT/runs/PPGF+VCLG"
```

Replace `8` with the GPU count. All dates are configurable:

```bash
bash run_single.sh 8 timeline.start=2025-07-01T00 timeline.end=2025-09-30T18
```

ERA5 metrics are written to `metrics/era69_rmse.log`, `.csv` and `.json`
under the run directory. They cover all 69 variables in physical units and
pool cosine-latitude-weighted squared errors before taking the square root.

Sampling defaults are 64 steps, two moment-matching iterations, two corrector
steps with amplitude 0.1, variance scale 1.0, variance floor 1e−8 and float16
precision. Each analysis uses seed `0 + anchor_index`; PPGF ordering is fixed
by the reliability matrix. See [reproducibility details](docs/reproducibility.md).

## Anonymous source package

```bash
python -m risda.package --output dist/risda-anonymous.zip
```

The archive includes source, launchers, configurations and documentation.
Git history, local environment files, model weights, data and run outputs are
excluded. Data and weights require separate distribution.
