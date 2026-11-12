# Data preparation

## Input layout

Paths are configured through `configs/paths.env`, `configs/data.yaml` and
`configs/assimilation.yaml`. The expected layout is:

```text
data/
  era5_hourly/YYYY/MM/YYYYMMDD_HH.npy
  era5_hourly/mean_std/era5_1h_mean.npy
  era5_hourly/mean_std/era5_1h_std.npy
  normalization/mean_era5.npy
  normalization/std_era5.npy
  normalization/mean_<source>.npy
  normalization/std_<source>.npy
  observations/<source>/<source>_YYYYMMDDHH.nc
```

CONVI uses the two directories configured as `data.surf_dir` and
`data.icoa_dir`. Set the latter to null only when the pretrained encoder was
trained without ICOADS.

## ERA5

Each array has shape `(69,720,1440)`. The channel sequence is
`z,t,u,v,r`, each at the following pressure levels in hPa:

```text
50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
```

The final channels are `t2m,u10,v10,msl`. Physical units are geopotential
m²/s² for z, K for temperature, m/s for wind, percent for relative humidity
and Pa for mean sea-level pressure. z is not geopotential height in metres.

Arrays and observation grids must share the same latitude/longitude orientation
and longitude origin as the pretrained models. The readers do not reorder,
regrid or infer orientation from array values. ERA5 evaluation follows the
model's stored grid and cosine latitude convention.

Set `era5.coordinate=physical` for unnormalized arrays. For normalized arrays,
supply the source mean/std in `configs/data.yaml`. Conversion is:

```text
physical = stored * source_std + source_mean
model_input = (physical - target_mean) / target_std
```

The target mean/std must match the ERA5 autoencoder and CONVI normalization.
The preparation command writes normalized arrays and a copy of their statistics
to a separate directory. Point subsequent data configurations at that directory
when using the converted files. Keep the original data when no conversion is needed.
All statistics are supplied artifacts; this command does not estimate them.

ERA5 timestamps are taken from filenames. Duplicate times, missing requested
times, wrong shapes and nonfinite values are errors. No nearest-time substitution
or temporal interpolation is performed.

## Satellite observations

NetCDF variable `x` has shape `(1,channels,720,1440)`. Files end in
`YYYYMMDDHH.nc`. Use UTC time T for the prepared record associated with hour T.

| Source | Brightness-temperature channels | Optional zenith input |
| --- | --- | --- |
| ATMS | 22 | First channel after brightness temperatures |
| AMSU-A | 15 | First channel after brightness temperatures |
| MHS | 5 | First channel after brightness temperatures |
| MWTS | 17 | Fourth channel after brightness temperatures |
| MWHS | 15 | Fourth channel after brightness temperatures |

The instrument mean/std arrays have one value per brightness-temperature
channel. Processing in `risda/data.py` performs channel-wise normalization,
uses finiteness as the valid-value indicator, replaces missing normalized values
with zero, and appends a union observation mask. The optional angle channel is
`cos(angle_in_degrees)`, selected by the pretrained encoder's `use_satzen`.
A missing hourly file contributes no guidance.

## Conventional observations

Each surface file exposes `x` with channel labels `t2m,u10,v10,slp`.
The reader converts slp from hPa to Pa before normalizing with the last four
ERA5 mean/std values. It takes each finite SURF value first and otherwise uses
the collocated ICOADS value. Missingness is preserved separately for each variable.

For the Cressman encoder, the raw input is four normalized values plus four
masks. The model preprocessor uses radius 10 grid cells, epsilon 1e−4 and
confidence_ignore 0 by default, producing four interpolated values, four
expanded masks and one confidence channel. The settings are loaded from the
encoder configuration. `appa/transforms.py` and `risda/encoders.py` contain
the interpolation implementation.

Interpolation does not enlarge the mask used to decide where observations
constrain the latent. `latent_mask` pads longitude circularly and latitude
with zeros, then pools by the autoencoder resize factor. A latent cell is
supported when at least one original physical pixel is observed. All available
surface observations participate; upper-air TEMP records are not used in this
release's guidance or ERA5 evaluation.

## Processing boundary

The included processing starts from hourly ERA5 arrays and gridded instrument
and surface observations. Raw product retrieval, instrument-level corrections,
screening and the initial mapping of records to the grid are upstream steps.
The current package does not implement these upstream transformations.
Providing their scripts or a precisely versioned prepared dataset is necessary
to reproduce the same inputs.

Normalization files and pretrained checkpoints must be distributed together
with their data/channel conventions. Changing these files changes the experiment;
it is not a path-only adaptation.
