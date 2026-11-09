# RISDA

Code for **RISDA: Reliability-Informed Score-based Data Assimilation of Multi-Source Atmospheric Observations**.

## Setup

Use Python 3.11 with CUDA-compatible PyTorch 2.5.1 and xFormers 0.0.28.post3.
Run commands from the repository root.

```bash
python -m pip install -e .
cp configs/paths.env.example configs/paths.env
```

Set data, model and output paths in `configs/paths.env`, then load them:

```bash
source configs/paths.env
python -m risda check
```

Configure model checkpoints and observation sources in
[`configs/assimilation.yaml`](configs/assimilation.yaml). The default sources are
ATMS, AMSU-A, MHS, MWTS, MWHS and CONVI. See [data requirements](docs/data.md)
for the expected files.

Set calibration and evaluation dates in [`configs/timeline.yaml`](configs/timeline.yaml).
Defaults are Q1 2025 for hourly calibration and Q2 2025 for evaluation every six hours.

## Training

Train the ERA5 autoencoder, diffusion prior and observation encoders
with 128 latent channels and 8 GPUs:

```bash
python -m risda train 128 8 --output "$RISDA_OUTPUT_ROOT/training"
```

Add `--stop-after ae` to train only the ERA5 autoencoder. See
[training instructions](docs/training.md) for individual stages and multiple nodes.
Before calibration, point `RISDA_LATENT_DIR` and `RISDA_DENOISER_DIR` to the
generated latents and prior, and update the observation model paths in
`configs/assimilation.yaml`.

## Calibration

With trained models configured, run:

```bash
python -m risda calibrate run --config configs/calibration.yaml --gpus 0,1,2,3,4,5,6,7
```

## Run

Run RISDA and compute evaluation metrics using 8 GPUs:

```bash
bash run_single.sh 'PPGF+VCLG' 8
```

Replace `8` with the desired GPU count. Add `--dry-run` to preview the commands.
Other methods are `PPGF_SD+VCLG`, `AGF+VCLG`, `PCGrad+VCLG`, `AGF_m` and `Appa`.
For `Appa`, first train the satellite forward models and calibrate them:

```bash
for instrument in atms amsua mhs mwts mwhs; do
  python -m risda train-decoder "$instrument" --gpus 8
done
```

Set the resulting decoder model paths in `configs/assimilation.yaml`, then run:

```bash
python -m risda calibrate-decoder run --config configs/calibration.yaml --gpus 0,1,2,3,4,5,6,7
bash run_single.sh Appa 8
```

Results and logs are saved under `$RISDA_OUTPUT_ROOT/runs/<METHOD>/`.
To recompute metrics for a completed run:

```bash
python -m risda metrics --run-dir "$RISDA_OUTPUT_ROOT/runs/PPGF+VCLG/evaluation"
```
