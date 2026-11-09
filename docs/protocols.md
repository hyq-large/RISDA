# Methods, calibration and evaluation

Public labels follow Sections 3–4 and Table 1 of *RISDA: Reliability-Informed
Score-based Data Assimilation of Multi-Source Atmospheric Observations*.

| CLI method | Observation route | Likelihood covariance | Gradient fusion |
| --- | --- | --- | --- |
| `Appa` | L2O; learned satellite operators, ERA5 decoder for CONVI | Shared calibrated scalar | Additive |
| `AGF_m` | O2L | Each source's channel variance averaged across its channels | Additive |
| `AGF+VCLG` | O2L | Channel-wise centered variance | Additive |
| `PCGrad+VCLG` | O2L | Channel-wise centered variance | PCGrad |
| `PPGF_SD+VCLG` | O2L | Spatial centered variance, per latent location and channel | PPGF; channel MSE priorities shared across locations |
| `PPGF+VCLG` | O2L | Spatial centered variance, per latent location and channel | PPGF; MSE priorities per latent location and channel |

`PPGF+VCLG` is the full RISDA method. Its launcher fixes both likelihood variance
and MSE priorities to spatial calibration. Here, spatial means a separate value
for each source, latent location and latent channel. Channel-wise calibration
instead shares each source's per-channel value across locations. Both use
diagonal likelihood covariance. `PPGF_SD+VCLG` also uses spatial likelihood
variance, but shares its per-channel MSE priorities across locations.

For the spatially calibrated additive
reference in the fusion ablation, run `AGF+VCLG vclg_variance=spatial` with a
separate `output_root`. The default PCGrad comparison uses fixed ordering.

## Residual statistics

For a valid observed latent position, compute
`e = (Encoder(obs) − mean)/raw_std − (ERA_latent − mean)/raw_std`.
Collect float64 moments on the original observation mask pooled onto the
latent grid (`mask_frac=0`: any observed pixel gives support):

```text
bias     = E[e]
MSE      = E[e²]
variance = E[e²] − E[e]²       # population denominator N
```

Centered variance parameterizes VCLG; MSE, including squared bias, determines
source reliability. Channel statistics pool time and space; spatial statistics
pool time at each location. Missing observations contribute no residuals.
Positions with fewer than two residuals use the same source's channel statistic.
Satellite channel-wise forwards use float16 and spatial forwards use float32;
CONVI uses float32 for both, retaining the numerical conventions of the supplied
calibration implementation. Moment accumulation uses float64.

Inference converts raw-coordinate variances by `(raw_std/effective_std)²`, with
`effective_std = sqrt(raw_std² + ae_noise²)`. `var_y_scale` is the global
likelihood-variance multiplier; it does not alter MSE priorities. `AGF_m` takes
the channel mean within each source in the inference coordinate.

L2O calibration uses normalized observation-space residuals. The shared Appa
scalar is the arithmetic mean across channel variances within each source,
followed by an equal-weight mean across sources. It is recomputed from the
configured calibration data.

## PPGF priority semantics

Lower MSE means higher reliability. The paper describes increasing-MSE order:
start with the most reliable source, accumulate compatible scalar gradients,
and stop at the first opposing contribution. The retained implementation uses
the equivalent reverse traversal: decreasing MSE, replacing the accumulator
when a more reliable gradient conflicts. A regression test checks equivalence
to the paper's stop rule. Ordering is fixed across frames, diffusion steps and
anchors. Spatial priorities vary across latent location and channel.

## Time axis and metrics

`configs/timeline.yaml` defines inclusive calibration and evaluation bounds.
Defaults are all Q1 2025 hourly calibration candidates and all Q2 2025
00/06/12/18 UTC evaluation anchors. Other ranges and evaluation cadences are
supported. Endpoints must lie on the configured cadence, and calibration must
end before the first evaluated anchor. Checkpoint training and validation splits
are described separately in [training.md](training.md).

Each independent anchor T uses six exact-hour frames `[T−3,T−2,T−1,T,T+1,T+2]`
and retains frame four. Context hours can extend beyond the configured test
bounds. An anchor subset retains its indices in the complete configured axis
so its seeds match a full run.

ERA69 metrics sum squared errors and cosine-latitude weights over every retained
anchor before one square root, then convert to physical units. They include
the four surface fields of ERA5. Optional TEMP metrics use all valid upper-air
observation records, with per-channel counts and physical-unit RMSE. Channels
with no TEMP observations report NaN in tensors/CSV and null in JSON.

The evaluator checks the saved configuration, generation contract hash,
trajectory completion, time axis, latent coordinates and each anchor payload.
Nonfinite predictions and incomplete runs are errors. Comparisons should reuse
the same inputs, checkpoints, normalization, time axis, sampler and seed.

Calibration records candidate, missing and zero-support hours, per-source
counts, model hashes, source data settings and worker logs. Frozen inputs prevent
changed models or calibrations from resuming into an existing run.
