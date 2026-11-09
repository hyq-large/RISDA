# Prepared data contract

The public pipeline begins with prepared arrays and gridded observations. Raw
satellite download, instrument calibration, upstream quality control and initial
gridding are external; this repository does not claim to reproduce those steps.

Default paths are below `RISDA_DATA_ROOT` (override through the documented
`RISDA_*` environment variables when reusing another layout):

```text
era5_hourly/YYYY/MM/YYYYMMDD_HH.npy
era5_hourly/mean_std/era5_1h_mean.npy
era5_hourly/mean_std/era5_1h_std.npy
normalization/mean_era5.npy
normalization/std_era5.npy
normalization/mean_<instrument>.npy
normalization/std_<instrument>.npy
observations/<instrument>/*.nc
observations/convi/surf_qc/*.nc
observations/convi/icoads_qc/*.nc
observations/convi/temp_time_qc/*.nc
observations/other_info/temp_bc_bias_0.npy
observations/other_info/temp_bc_bias_12.npy
```

ERA5 arrays have shape `(69,720,1440)` in the stored hourly normalized coordinate.
The channel order is `z,t,u,v,r` at 13 pressure levels, then `t2m,u10,v10,msl`.
The authoritative level sequence and mappings are in `appa/data/const.py` and
`appa/data/mappings.py`. Normalization conversion requires all four source/target
mean/std files; the target pair must match the frozen ERA Decoder.

Satellite NetCDF files expose variable `x` on the 720×1440 grid. ATMS has 22 TB
channels, AMSUA 15, MHS 5, MWTS 17 and MWHS 15. ATMS/AMSUA/MHS raw channels are
`[TB | satzen | solzen | satname]`. MWTS/MWHS add scan/land-orbit metadata and use
Earth incidence angle for zenith. The reader creates `[standardized TB | mask |
cos(zenith)]`. File T describes `[T,T+1h)` and is paired only with ERA time T.
Names are `<instrument>_YYYYMMDDHH.nc` (bare `YYYYMMDDHH.nc` is also supported
for MWTS/MWHS).

CONVI filenames end in `_YYYYMMDDHH.nc`. SURF and ICOADS contain the four surface
variables; pressure is converted from hPa to Pa. Values are standardized with
the **ERA5 mean/std**, not an independent CONVI normalization. SURF wins collocated
conflicts. Missingness is preserved per channel. Cressman interpolation changes
Encoder inputs without expanding the raw QC supervision/guidance support.
QC directory basenames are validated by the evaluator; keep `surf_qc`,
`icoads_qc` and `temp_time_qc` when adapting paths.

All valid QC surface observations are available to CONVI training,
calibration and guidance. TEMP is never used in guidance and is evaluated at
all valid upper-air records, normally at 00/12 UTC. TEMP evaluation can be
disabled with `convi.metrics_include_temp=false`; the evaluator then requires
neither the TEMP data tree nor its bias files.

Native latent dumps contain `latents: (time,1081,C)` and `dates: (time,4)` with
`[year,month,day,hour]`, plus `stats.pth` with `mean` and `std`. The containing
directory must also contain `ae/config.yaml` and `ae/model.pth`. The prior lives
under its `denoisers/<run_id>/` subdirectory. Missing ERA truth timestamps are
errors; missing satellite files are recorded explicitly by calibration.

