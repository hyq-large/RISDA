r"""Constants used in the diagnostics module for plotting."""

import cmocean.cm as cmo

# Mean Earth radius [km]
EARTH_RADIUS = 6371.0088
# Geocentric gravitational constant = G * M_earth [m s^-2]
GEOCENTRIC_GRAV_CONST = 3.986004415e14  # https://iau-a3.gitlab.io/NSFA/NSFA_cbe.html#GME2009

UNITS = {
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m/s",
    "10m_v_component_of_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "sea_surface_temperature": "K",
    "total_precipitation": "m",
    "temperature": "K",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
    "geopotential": "m^2/s^2",
    "specific_humidity": "kg/kg",
    "vertical_velocity": "Pa/s",
}

PRETTY_VAR_NAMES = {
    "2m_temperature": "2m temperature",
    "10m_u_component_of_wind": "10m U wind component",
    "10m_v_component_of_wind": "10m V wind component",
    "mean_sea_level_pressure": "Mean sea level pressure",
    "sea_surface_temperature": "Sea surface temperature",
    "temperature": "Temperature",
    "geopotential": "Geopotential",
    "u_component_of_wind": "U component of wind",
    "v_component_of_wind": "V component of wind",
    "specific_humidity": "Specific humidity",
    "total_precipitation": "Total precipitation",
}

# Colors used for line plots
CMAPS_LINE = {
    "2m_temperature": "#FF4500",
    "10m_u_component_of_wind": "#1E90FF",
    "10m_v_component_of_wind": "#1E90FF",
    "mean_sea_level_pressure": "#A9A9A9",
    "sea_surface_temperature": "#FF6347",
    "total_precipitation": "#00BFFF",
    "temperature": "#FF7F50",
    "u_component_of_wind": "#4682B4",
    "v_component_of_wind": "#4682B4",
    "geopotential": "#FFD700",
    "specific_humidity": "#7FFF00",
    "vertical_velocity": "#32CD32",
}

CMAPS_LINE_DEEPMIND = {
    "2m_temperature": {"color": "red", "marker": "*"},
    "10m_u_component_of_wind": {"color": "blue", "marker": "o"},
    "10m_v_component_of_wind": {"color": "green", "marker": "s"},
    "mean_sea_level_pressure": {"color": "cyan", "marker": "+"},
    "sea_surface_temperature": {"color": "orange", "marker": "x"},
    "total_precipitation": {"color": "purple", "marker": "d"},
    "temperature": {"color": "red", "marker": "*"},
    "geopotential": {"color": "magenta", "linestyle": ":"},
    "u_component_of_wind": {"color": "blue", "linestyle": "-.", "marker": "o"},
    "v_component_of_wind": {"color": "green", "linestyle": "-.", "marker": "s"},
    "specific_humidity": {"color": "y", "linestyle": "-."},
}

# Colormaps used for surface plots
CMAPS_SURF = {
    "2m_temperature": cmo.thermal,
    "sea_surface_temperature": cmo.thermal,
    "temperature": cmo.thermal,
    "10m_u_component_of_wind": "seismic_r",
    "10m_v_component_of_wind": "seismic_r",
    "u_component_of_wind": "seismic_r",
    "v_component_of_wind": "seismic_r",
    "mean_sea_level_pressure": cmo.solar,
    "total_precipitation": cmo.oxy,
    "geopotential": cmo.dense,
    "specific_humidity": cmo.rain,
    "vertical_velocity": cmo.matter,
}
