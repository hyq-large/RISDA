import csv
import h5py
import json
import torch

from einops import rearrange
from pathlib import Path
from scipy.spatial import KDTree
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm
from typing import Callable, Optional

from appa.data.const import (
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_SURFACE_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.diagnostics.const import EARTH_RADIUS, GEOCENTRIC_GRAV_CONST
from appa.grid import arc_to_chord, create_N320, latlon_to_xyz, xyz_to_latlon
from appa.save import safe_load, safe_save


def process_weather_5k_metadata(data_path: Path, save_path: Path) -> None:
    r"""Processes the metadata from the Weather 5k paper dataset and saves it to a file.

    Reference:
    | WEATHER-5K: A Large-scale Global Station Weather Dataset Towards Comprehensive Time-series Forecasting Benchmark
    | https://arxiv.org/abs/2406.14399v1

    Arguments:
        data_path: Path to meta_info.json.
        save_path: Path to the save location.
    """
    # Ensure data_path points to meta_info.json
    if data_path.name != "meta_info.json":
        data_path = data_path / "meta_info.json"

    with open(data_path, "r") as f:
        input_data = json.load(f)

    # Transform data into new format
    output_data = {}
    for station_file, metadata in input_data.items():
        station_id = station_file.replace(".csv", "")
        output_data[station_id] = {
            "lat": metadata["latitude"],
            "lon": metadata["longitude"],
            "ele": metadata["ELEVATION"],
        }

    # Save transformed data
    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)


def process_raw_noaa_csv(data_path: Path, save_path: Path) -> None:
    r"""Processes the data from raw NOAA csv files.

    Arguments:
        data_path: Path to dir with csv files.
        save_path: Path to the save location.
    """

    output_data = {}

    for csv_file in data_path.glob("*.csv"):
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["LATITUDE"] == "" or row["LONGITUDE"] == "":
                    continue
                station_id = row["STATION"]
                output_data[station_id] = {
                    "lat": float(row["LATITUDE"]),
                    "lon": float(row["LONGITUDE"]),
                    "ele": float(row["ELEVATION"]) if row["ELEVATION"] != "" else None,
                }

    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)


def create_observation_mask(processed_data: str, save_path: str) -> None:
    r"""Loads the information about weather stations, creates the mask and saves it.


    Arguments:
        processed_data: Path to the information json.
        save_path: Path to the save location.
    """
    with open(processed_data, "r") as f:
        station_info = json.load(f)

    station_lat_lon = torch.tensor([(info["lat"], info["lon"]) for info in station_info.values()])

    N320 = create_N320()

    mask = torch.zeros(721, 1440, dtype=torch.bool)
    stations_rounded = torch.round(station_lat_lon / 0.25) * 0.25

    N320 = N320.reshape(721, 1440, 2)
    for i in tqdm(range(721)):
        mask[i] = torch.any(
            torch.all(torch.abs(N320[i].unsqueeze(1) - stations_rounded) < 0.25, dim=-1), dim=1
        )

    mask = mask.bool()
    mask = mask.nonzero()  # Convert to indices
    mask = mask[..., 0] * 1440 + mask[..., 1]

    safe_save(mask, save_path)


def generate_orbit(
    orbital_altitude: float,
    inclination: float,
    initial_phase: float,
    observation_time: int,
    obs_freq: int,
    delay: int = 0,
):
    r"""Generates a satellite orbit trajectory.

    Simulates a circular orbit around Earth, accounting for Earth's rotation and orbital parameters.
    The orbit is defined by its height, inclination and initial phase, and returns latitude/longitude
    coordinates over time. The total orbit time is observation_time + delay.

    The function creates a circular orbit in the equatorial plane and applies
    two rotations: one around the x-axis (inclination) and one around the z-axis (initial phase).
    After, it accounts for Earth's rotation by subtracting the corresponding angular displacement
    from the longitude coordinates.

    Arguments:
        orbital_altitude: Orbit altitude above Earth's surface [km].
        inclination: Orbit inclination [deg].
        initial_phase: Initial orbit phase [deg].
        observation_time: Duration for which to generate observations [hours].
        obs_freq: Number of observations per hour (sampling frequency).
        delay: Initial time offset before starting observations [hours]. The satellite
               completes its normal orbit during this delay period, but no observation
               points are recorded before the delay has elapsed.

    Returns:
        The orbit trajectory in latitude/longitude coordinates, with shape :math:`(T, F, 2)`,
        where :math:`T` is the number of hours and :math:`F` is the observation frequency.
    """

    height = torch.tensor(orbital_altitude)
    inclination = torch.tensor(inclination)
    initial_phase = torch.tensor(initial_phase)
    observation_time = torch.as_tensor(observation_time)

    inclination = torch.deg2rad(90 - inclination)
    initial_phase = torch.deg2rad(initial_phase)

    R = (EARTH_RADIUS + height) * 1e3  # Earth radius in meters + height in meters

    T = 2 * torch.pi * torch.sqrt(R**3 / GEOCENTRIC_GRAV_CONST) / 3600  # in hours

    delay_distance = 360 * delay / T  # in deg as deg2rad inside latlon_to_xyz
    fly_distance = 360 * (observation_time + delay) / T  # in deg as deg2rad inside latlon_to_xyz

    earth_rotation_delay = 360 * delay / 24
    earth_rotation_end = 360 * (observation_time + delay) / 24

    num_obs = obs_freq * observation_time

    fly_distance = torch.linspace(delay_distance, fly_distance, num_obs + 1)[:-1]
    earth_rotation = torch.linspace(earth_rotation_delay, earth_rotation_end, num_obs + 1)[:-1]

    rot_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(inclination), -torch.sin(inclination)],
        [0, torch.sin(inclination), torch.cos(inclination)],
    ])
    rot_z = torch.tensor([
        [torch.cos(initial_phase), -torch.sin(initial_phase), 0],
        [torch.sin(initial_phase), torch.cos(initial_phase), 0],
        [0, 0, 1],
    ])

    # Rotating circle around the equator
    traj = torch.tensor([[-1, 0]]) * fly_distance[..., None]

    traj = latlon_to_xyz(traj)

    # Rotation along x- and z-axis to get the desired orbit
    orbit = torch.einsum("ij,kj->ki", rot_z, torch.einsum("ij,kj->ki", rot_x, traj))

    orbit = xyz_to_latlon(orbit)
    orbit[..., 1] = (orbit[..., 1] - earth_rotation) % 360
    orbit[..., 1] -= 180
    orbit = rearrange(orbit, "(h obs) c -> h obs c", c=2, h=observation_time)

    return orbit


def orbit_to_sparse(orbit: Tensor, fov: int = 5, grid: Optional[Tensor] = None) -> list[Tensor]:
    r"""Converts an orbit trajectory to a sparse visibility mask.

    Creates a sparse mask (i.e. indices) indicating which grid points are visible from the satellite's position,
    based on a field of view (FOV) angle. The default FOV corresponds to ~700-1400km swath width
    for typical Earth observation satellites.

    References:
        | Sentinel-3 Mission Overview
        | https://sentiwiki.copernicus.eu/web/s3-mission#S3Mission-OverviewofSentinel-3Mission

    Arguments:
        orbit: The orbit trajectory in latitude/longitude coordinates, with shape :math:`(T, F, 2)`.
        fov: The radius of the field of view [deg].
        grid: The grid vertices, with shape :math:`(N, 2)`. Defaults to N320 (721x1440).

    Returns:
        A list of tensors, each containing the indices of visible grid points for a given orbit segment.
    """
    if grid is None:
        grid = create_N320()
    tree = KDTree(latlon_to_xyz(grid))
    fov = fov * torch.pi / 180
    del grid

    orbit = latlon_to_xyz(orbit)

    indices = [tree.query_ball_point(o, arc_to_chord(fov)) for o in orbit]
    indices = [torch.unique(torch.cat([torch.tensor(ii) for ii in idx])) for idx in indices]
    return indices


def create_satellite_mask(
    orbital_altitude: float,
    inclination: float,
    initial_phase: float,
    observation_time: int,
    obs_freq: int = 60,
    fov: int = 5,
    delay: int = 0,
    trajectory_dt: int = 1,
    sparse: bool = True,
    grid: Optional[Tensor] = None,
    grid_shape: tuple = (721, 1440),
) -> list[Tensor]:
    r"""Creates a visibility mask for a satellite orbit, and returns it either sparsely or densely.

    Arguments:
        orbital_altitude: Orbit altitude above Earth's surface [km].
        inclination: Orbit inclination [deg].
        initial_phase: Initial orbit phase [deg].
        observation_time: Duration for which to generate observations [hours].
        obs_freq: Number of observations per hour (sampling frequency).
        fov: The radius of the field of view [deg].
        delay: Initial time offset before starting observations [hours]. The satellite
               completes its normal orbit during this delay period, but no observation
               points are recorded until after the delay has elapsed.
        trajectory_dt: Number of hours between two trajectory steps.
                       Temporary workaround: use the latest observation hour betweenn each time step.
        sparse: If True, returns a list of tensors with the indices of visible grid points for each orbit segment.

    Returns:
        If `sparse` is True, returns a list of :math:`T` tensors, each containing the indices of visible grid points
        for a given orbit segment. Otherwise, returns a dense mask with shape :math:`(T, 721, 1440)`.
    """
    orbit = generate_orbit(
        orbital_altitude, inclination, initial_phase, observation_time, obs_freq, delay
    )
    indices = orbit_to_sparse(orbit, fov, grid=grid)

    if trajectory_dt > 1:
        assert (
            observation_time % trajectory_dt == 0
        ), "Observation time must be divisible by trajectory_dt."
        indices = indices[trajectory_dt - 1 :: trajectory_dt]
        # TODO: Better satellite models & change observation time to be the number of steps.

    if sparse:
        return indices

    mask = torch.zeros((len(indices), grid_shape[0] * grid_shape[1]), dtype=torch.bool)
    for i, idx in enumerate(indices):
        mask[i, idx] = True

    mask = rearrange(mask, "T (LAT LON) -> T LAT LON", LAT=grid_shape[0], LON=grid_shape[1])
    return mask


def create_variable_mask(idx: Tensor, num_levels: int = len(SUB_PRESSURE_LEVELS), num_channels: int = None):
    r"""Creates a variable mask for atmospheric and surface variables.

    Constructs a boolean mask for the ERA5 atmospheric and surface variables.

    Arguments:
        idx: A tensor of indices indicating which variables to include in the mask of shape :math:`i`.
        num_levels: The number of pressure levels.
        num_channels: Total number of channels. If None, computed from ERA5 surface + atmospheric variables.

    Returns:
        A boolean mask with shape :math:`n`, where :math:`n` is the
        total number of variables, indicating the selected variables across the grid.
    """
    if num_channels is not None:
        n = num_channels
    else:
        n = len(ERA5_SURFACE_VARIABLES) + len(ERA5_ATMOSPHERIC_VARIABLES) * num_levels
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    return mask


def create_masks(
    masks_config,
    num_steps,
    trajectory_dt,
    delay,
    save=None,
    split_seed=0,
    grid=None,
    grid_shape=None,
    ratio=0,
):
    r"""Creates a set of masks from a configuration.

    Arguments:
        masks_config: A list of mask configurations.
                      Each mask should be:
                      mask:
                        name: ...
                        type: "satellite" or "stations"
                        covariance: ... (float)
                        config: dict for the parameters of the mask
        num_steps: The number of time steps for the trajectory.
        delay: The initial time offset before starting observations [hours].
        trajectory_dt: Number of hours between two trajectory steps.
        save: Path to save the masks. If None, does not save.
        split_seed: Seed for random splitting in the stations masks.

    Returns:
        A tuple of two tuples:
            - Satellite masks and their covariance tensors.
            - Assimilated stations masks, validation stations masks, and their covariance tensors.
    """
    satellite_mask = [torch.empty((0,), dtype=torch.int64) for _ in range(num_steps)]
    stations_mask_assim = torch.empty((0,), dtype=torch.int64)
    stations_mask_valid = torch.empty((0,), dtype=torch.int64)

    satellite_cov = [torch.empty((0,), dtype=torch.float32) for _ in range(num_steps)]
    stations_cov = torch.empty((0,), dtype=torch.float32)

    if save:
        mask_file = h5py.File(save / "masks.h5", "w")
        mask_file.create_group("satellite_masks")
        mask_file.create_group("stations_masks")

    for mask in masks_config:
        if mask.type == "satellite":
            if save:
                mask_file["satellite_masks"].create_group(mask.name)
                for k, v in mask.config.items():
                    mask_file["satellite_masks"][mask.name].attrs[k] = v

            grid_kwargs = {}
            if grid is not None:
                grid_kwargs["grid"] = grid
            if grid_shape is not None:
                grid_kwargs["grid_shape"] = grid_shape

            sat_masks = create_satellite_mask(
                observation_time=num_steps * trajectory_dt,
                delay=delay,
                trajectory_dt=trajectory_dt,
                **mask.config,
                **grid_kwargs,
            )
            if grid_shape is not None:
                import math
                max_index = math.prod(grid_shape)
                # `satellite_mask` outside is sized num_steps (one entry per
                # trajectory step). The random override must match — use
                # `num_steps`, NOT `num_steps * trajectory_dt`. With
                # trajectory_dt=1 the two are equal, but for trajectory_dt>1
                # the latter overruns satellite_mask and crashes downstream.
                sat_random_mask = []
                for obs_time in range(num_steps):
                    sat_random_mask.append(torch.randint(0,max_index-1,(int(max_index*ratio),)))
                sat_masks = sat_random_mask
            print('Ratio', ratio, len(sat_masks),'OBS num PER hour',len(sat_masks[0]))
            for i, sat_mask in enumerate(sat_masks):
                satellite_mask[i] = torch.cat((satellite_mask[i], sat_mask))

                cov_tensor = mask.covariance * torch.ones(len(sat_mask))
                satellite_cov[i] = torch.cat((satellite_cov[i], cov_tensor))

                if save:
                    mask_file["satellite_masks"][mask.name].create_dataset(
                        f"mask_{i}", data=satellite_mask[i]
                    )
        elif mask.type == "stations":
            stations_mask = safe_load(f"ground_masks15/mask_{mask.config.num_stations}.pt")

            if stations_mask.dtype == torch.bool:
                # Convert to indices
                last_dim = stations_mask.shape[-1]
                stations_mask = stations_mask.nonzero()
                stations_mask = stations_mask[:, 0] * last_dim + stations_mask[:, 1]

            generator = torch.Generator().manual_seed(split_seed)
            stations_mask = stations_mask[torch.randperm(len(stations_mask), generator=generator)]
            assim_idx = stations_mask[mask.config.num_valid_stations :]
            valid_idx = stations_mask[: mask.config.num_valid_stations]
            stations_mask_assim = torch.cat((stations_mask_assim, assim_idx))
            stations_mask_valid = torch.cat((stations_mask_valid, valid_idx))

            cov_tensor = mask.covariance * torch.ones(len(assim_idx))
            stations_cov = torch.cat((stations_cov, cov_tensor))

            if save:
                mask_file["stations_masks"].create_group(mask.name)
                mask_file["stations_masks"][mask.name].create_dataset(
                    "assimilation", data=stations_mask_assim
                )
                mask_file["stations_masks"][mask.name].create_dataset(
                    "validation", data=stations_mask_valid
                )
                for k, v in mask.config.items():
                    mask_file["stations_masks"][mask.name].attrs[k] = v
        else:
            raise ValueError(f"Unknown mask type: {mask.type}")

    if save:
        # Save some attributes about the masks
        mask_file["satellite_masks"].attrs["blanket_size"] = num_steps
        mask_file["satellite_masks"].attrs["num_stations"] = len(stations_mask_assim)
        mask_file["satellite_masks"].attrs["num_valid_stations"] = len(stations_mask_valid)
        mask_file["satellite_masks"].attrs["num_satellite_masks"] = len(satellite_mask)
        # Save total masks
        mask_file["satellite_masks"].create_dataset("total_mask", data=torch.cat(satellite_mask))
        mask_file["stations_masks"].create_dataset("total_mask_assim", data=stations_mask_assim)
        mask_file["stations_masks"].create_dataset("total_mask_valid", data=stations_mask_valid)
        mask_file.close()

    satellite_mask = [mask.cuda() for mask in satellite_mask]
    stations_mask_assim = stations_mask_assim.cuda()
    stations_mask_valid = stations_mask_valid.cuda()
    satellite_cov = [cov.cuda() for cov in satellite_cov]
    stations_cov = stations_cov.cuda()

    return (
        (satellite_mask, satellite_cov),
        ((stations_mask_assim, stations_mask_valid), stations_cov),
    )


def mask_state(
    x: Tensor,
    t: int,
    stations_mask: Tensor,
    stations_cov: Tensor,
    satellite_mask: list[Tensor],
    satellite_cov: Tensor,
    stations_variables: Optional[Tensor] = None,
    satellite_variables: Optional[Tensor] = None,
):
    r"""Masks a given physical state given its index t in the blanket
    Args:
        x: Physical state of shape :math:`(batch_size, Lat * Lon, num_channels)`
        t: Time index in the current total set of blankets of the rank.
        stations_mask: Mask tensor for the stations, of shape :math:`(1, N_stat)`.
        stations_cov: Covariance tensor for the stations, of shape :math:`(1, N_stat)`.
        stations_variables: Boolean tensor for observed variables for the stations, of shape :math:`(num_channels)`.
        satellite_mask: List of mask tensors for the satellites
        satellite_cov: Covariance for the satellites
        satellite_variables: Boolean tensor for observed variables for the stations, of shape :math:`(num_channels)`.
    Returns:
        y: flattened masked state pixels
        cov_y
    """

    obs_list = []
    cov_list = []

    if len(stations_mask) > 0 and stations_variables is not None:
        stations_obs = x[:, stations_mask][:, :, stations_variables]
        # Covariance assumed to be the same for all pressure levels for a given observator
        stations_obs_cov = stations_cov.repeat(stations_obs.shape[-1])
        obs_list.append(stations_obs.flatten())
        cov_list.append(stations_obs_cov)
    if len(satellite_mask[t]) > 0 and satellite_variables is not None:
        satellite_obs = x[:, satellite_mask[t]][:, :, satellite_variables]
        satellite_obs_cov = satellite_cov[t].repeat(satellite_obs.shape[-1])
        obs_list.append(satellite_obs.flatten())
        cov_list.append(satellite_obs_cov)
    return torch.cat(obs_list), torch.cat(cov_list)


def observator_partial(
    blanket: Tensor,
    blanket_id: int,
    context_fn: Optional[Callable],
    timestamp_fn: Callable,
    blanket_size: int,
    blanket_stride: int,
    num_latent_channels: int,
    autoencoder: Module,
    latent_mean: Tensor,
    latent_std: Tensor,
    slice_fn: Callable = None,
    mask_fn: Callable = None,
):
    r"""Observation function for masked pixel-space assimilation.
    Args:
        blanket: Tensor of shape :math:`(B, K * N * C)`, where B is the batch size,
            K is the blanket size, N is the number of vertices, and C is the number of channels.
        blanket_id: The ID of the blanket.
        context_fn: A function that takes a blanket_id and returns the context for the autoencoder.
        timestamp_fn: A function that takes a blanket_id and returns the timestamp for the autoencoder.
        blanket_size: The size of the blanket.
        blanket_stride: The stride of the blanket.
        num_latent_channels: The number of latent channels.
        autoencoder: The autoencoder model.
        latent_mean: The mean of the latent space.
        latent_std: The standard deviation of the latent space.
        slice_fn: A function blanket_id -> slice that determines the time slice to observe.
                        blanket_id is the index of the blanket in the rank batch.
        mask_fn: A function that takes a physical state and a time index and returns a masked
            version of the state.
    Returns:
        y: A tensor of shape :math:`(B, .)` containing the masked observations.
    """
    blanket = rearrange(
        blanket,
        "B (K N C) -> B K N C",
        K=blanket_size,
        C=num_latent_channels,
    )
    y = []

    if slice_fn is None:
        time_indices = slice(0, blanket_size)
    else:
        time_indices = slice_fn(blanket_id)

    context = context_fn(blanket_id) if context_fn is not None else None
    timestamp = timestamp_fn(blanket_id)

    for i, t in enumerate(range(blanket_size)[time_indices]):
        z_denorm = blanket[:, t] * latent_std.cuda() + latent_mean.cuda()
        ctx = context[i : i + 1] if context is not None else None
        x_p = autoencoder.decode(z_denorm, timestamp[i : i + 1], ctx)
        observed_values, _ = mask_fn(x_p, blanket_id * blanket_stride + t)
        y.append(observed_values)
        del x_p

    return torch.cat(y)


def observator_full(
    blanket: Tensor,
    blanket_id: int,
    blanket_size: int,
    num_latent_channels: int,
    slice_fn: Callable[[int], slice] = None,
):
    r"""Observation function for full latent-space assimilation.
    Args:
        blanket: Tensor of shape :math:`(B, K * N * C)`, where B is the batch size,
            K is the blanket size, N is the number of vertices, and C is the number of channels.
        blanket_id: The ID of the blanket.
        blanket_size: The size of the blanket.
        num_latent_channels: The number of latent channels.
        slice_fn: A function blanket_id -> slice that determines the slice of the blanket to observe.
    Returns:
        y: A tensor of shape :math:`(B, .)` containing the masked observations.
    """
    z = rearrange(blanket, "B (K N C) -> B K N C", K=blanket_size, C=num_latent_channels)

    if slice_fn is not None:
        return z[:, slice_fn(blanket_id)].flatten()
    else:
        return z.flatten()
