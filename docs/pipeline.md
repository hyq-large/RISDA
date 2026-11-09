# Pipeline and implementation map

| Stage | Implementation | Inputs → outputs |
| --- | --- | --- |
| ERA5 AE | `experiments/autoencoder/train_1h.py`, `appa/nn/cae.py`, `appa/losses.py` | Normalized hourly ERA69 → trainable Encoder + Decoder |
| Latent export | `experiments/autoencoder/dump_1h.py` | Frozen AE + hourly fields → `era5.h5` and copied AE |
| Latent statistics | `scripts/data/latent_stats_v1.py` | Training-period native latents → per-channel `stats.pth` |
| Diffusion training | `experiments/diffusion/train_diff_1h.py`, `appa/diffusion.py`, `appa/nn/vit.py` | Standardized six-frame windows → denoiser |
| Satellite Encoders | `experiments/observations/train_obs2era.py`, `risda/models/encoders.py` | Exact-hour TB, mask, zenith → ERA latent |
| CONVI Encoder | `experiments/observations/train_convi_cressman_obs2era.py`, `convi_obs2era_data.py` | SURF/ICOADS observations with interpolated inputs → ERA latent |
| Forward controls | `experiments/observations/train_era5_latent_to_tb.py`, `risda/models/decoders.py` | ERA latent + zenith → standardized satellite TB |
| Encoder calibration | `risda/calibrate.py` | Encoder–truth residuals on observed latent positions → channel/spatial moments + MSE ordering |
| Decoder calibration | `risda/calibrate_decoder.py` | Forward predictions–observations → per-channel moments and equal-source constant |
| Likelihood | `appa/diffusion.py`, `experiments/observations/convi_physical.py`, `direct_e2o_assimilation.py` | Source operators + covariance → individual MMPS guidance |
| Fusion | `appa/math.py`, `appa/sampling.py` | Source gradients + calibrated order → fused guidance and sampling updates |
| Generation | `experiments/observations/evaluate_guided_six_frame.py` | Prior + observations + calibration → retained frame-four latents |
| Evaluation | `experiments/observations/evaluate_guided_six_frame_metrics.py` | Decoded frame-four fields → ERA69 and optional TEMP metrics |

The ERA5 state contains 69 channels on a 720×1440 grid. The reference AE uses
resize 32 and 128 latent channels, giving a 23×47×128 latent. Native latent
arrays are flattened as 1081×128. The public trainer also accepts 512/1024 channels;
new models require matching new latents, statistics, observation models and caches.

An assimilation anchor T uses six hourly frames `[T−3,T−2,T−1,T,T+1,T+2]`.
The fourth frame is retained. Every six-hourly anchor is sampled independently.
This is windowed assimilation with future observations through T+2, not a causal
forecast initialized at T. Source file timestamps are exact-hour; no nearest-time
or adjacent-hour merge is performed by the current satellite reader.

The exported `appa/` infrastructure contains the AE/DiT/MMPS and fusion
implementations. `experiments/observations/` provides observation data loaders,
calibration kernels and the six-frame generator. Use `python -m risda` for the
supported training, calibration and evaluation entrypoints.

