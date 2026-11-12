# RISDA

PPGF+VCLG for multi-source atmospheric data assimilation.

Run the commands below from the repository root.

## Setup

```bash
python -m pip install -e .
cp configs/paths.env.example configs/paths.env
```

Set the data, model and output paths in `configs/paths.env`, then load them:

```bash
source configs/paths.env
```

Inputs are hourly ERA5 fields, gridded observations from ATMS, AMSU-A, MHS,
MWTS, MWHS and CONVI, and normalization arrays. Set data paths in
`configs/data.yaml` and the stage configurations below.

## Train

Set paths and dates in `configs/training/*.yaml`, then run in order:

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

These stages train the ERA5 autoencoder, export latents, compute latent
statistics, train the diffusion prior and train the observation encoders.
Replace `8` with your GPU count; latent export uses one GPU.
Skip training if the models and latent statistics are already available.

## Calibrate

Set the dates in `configs/calibration.yaml` (default: Q1 2025), then run:

```bash
python -m risda.calibrate
```

## Run and evaluate

Set the sources and dates in `configs/assimilation.yaml` (default: Q2 2025).

```bash
bash run_single.sh 8
python -m risda.evaluate --run-dir "$RISDA_OUTPUT_ROOT/runs/PPGF+VCLG"
```

ERA5 results are saved to `metrics/era69_rmse.log` under the run directory.
