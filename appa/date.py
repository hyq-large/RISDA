r"""Date and time helpers."""

import numpy as np
import pandas as pd
import re
import torch

from datetime import datetime, timedelta
from torch import Tensor

from .data.const import BASE_DAYS_IN_MONTH


# TODO: Could be merged with `blanket_day_to_tensor` (requires adapting usages)
def create_trajectory_timestamps(start_day: str, start_hour: int, traj_size, dt=1):
    r"""Creates a tensor containing the timestamps of a trajectory.

    Arguments:
        start_day: The day of the year the trajectory starts at, formatted as "YYYY-MM-DD".
        start_hour: The hour of the day, between 0 and 23.
        traj_size: The size T of the trajectory.
        dt: The time difference between each timestamp in the trajectory, in hours.

    Returns:
        A tensor of shape (T, 4) containing the year, month, day, and hour for each
            element in the blanket.

    """
    year, month, day = map(int, start_day.split("-"))
    start_date = datetime(year, month, day, start_hour)
    dates = [start_date]
    for i in range(1, traj_size):
        dates.append(start_date + timedelta(hours=dt * i))
    dates = [[d.year, d.month, d.day, d.hour] for d in dates]

    return torch.tensor(dates)


def format_blanket_date(date):
    r"""Format a blanket date for logging.

    Args:
        date: The blanket date. A tensor of shape [(1,) blanket_size, 4]

    Returns:
        The formatted date.
    """
    import torch

    if len(date.shape) == 3:
        date = date.squeeze(0)
    start = date[0]
    end = date[-1]

    if date.shape[0] == 1:
        Y, M, D = start[:-1].int()
        return f"{D:02d}/{M:02d}/{Y} {start[-1]:02d}h"
    elif torch.all(start[:-1] == end[:-1]):
        Y, M, D = start[:-1].int()
        return f"{D:02d}/{M:02d}/{Y} {start[-1]:02d}h to {end[-1]:02d}h"
    else:
        Y1, M1, D1 = start[:-1].int()
        Y2, M2, D2 = end[:-1].int()
        return f"{D1:02d}/{M1:02d}/{Y1} {start[-1]:02d}h to {D2:02d}/{M2:02d}/{Y2} {end[-1]:02d}h"


def assert_date_format(date_string: str) -> None:
    r"""Asserts that the date string is in the correct format (YYYY-MM-DD or YYYY-MM-DDTHH).

    Arguments:
        date_string: Date string to check.

    Raises:
        ValueError: If the date string does not match the pattern
    """

    # YYYY-MM-DD or YYYY-MM-DDTHH
    pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(T([01]\d|2[0-3]))?$"

    if not re.match(pattern, date_string):
        raise ValueError("The format is incorrect. The date should be in YYYY-MM-DD format.")


def get_date_features(date: np.datetime64) -> Tensor:
    r"""Extracts year, month, day, and hour from a numpy.datetime64 object.

    Arguments:
        date: A numpy.datetime64 object representing a specific date and time.

    Returns:
        A tensor containing the year, month, day, and hour extracted from the input date.
    """
    timestamp = pd.to_datetime(date)
    return torch.as_tensor([timestamp.year, timestamp.month, timestamp.day, timestamp.hour])


def get_year_progress_encoding(date: torch.Tensor) -> torch.Tensor:
    r"""Encodes the date based on the day in the year.

    Arguments:
        date: tensor of shape :math:`(*, 4)`, where the last dimension is [year, month, day, hour].

    Returns:
        A tensor of shape (*, 2) containing the encoded dates.
    """
    year, month, day, hour = date.unbind(dim=-1)

    is_leap_year = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    is_after_feb = month > 2
    add_day = torch.logical_and(is_leap_year, is_after_feb)

    cumulative_days = torch.zeros(len(BASE_DAYS_IN_MONTH) + 1, device=date.device)
    cumulative_days[1:] = torch.as_tensor(BASE_DAYS_IN_MONTH, device=date.device).cumsum(-1)
    days_up_month = cumulative_days[month - 1]

    day = days_up_month + day + add_day - 1
    hour_of_year = day * 24 + hour

    total_days = is_leap_year + cumulative_days[-1]
    total_hours = total_days * 24

    angle = 2 * torch.pi * (hour_of_year / total_hours)

    sin_encoding = torch.sin(angle)
    cos_encoding = torch.cos(angle)

    encoded_dates = torch.stack([sin_encoding, cos_encoding], dim=-1)

    return encoded_dates


def get_local_time_encoding(date: Tensor, longitude: Tensor) -> torch.Tensor:
    r"""Encodes a local time based on GMT hour and longitude.

    Arguments:
        date: tensor of date at Greenwich meridian of shape :math:=`(*, 4)`, where each row is [year, month, day, hour].
        longitude: tensor of shape (N) representing the longitude in degrees (-180 to 180) shared across batch elements.

    Returns:
        A tensor of shape (*, N, 2) containing the encoded local times for each date and longitude.
    """
    longitude = torch.where(longitude < 0, longitude + 360, longitude)
    normalized_longitude = longitude / 360
    hour_GMT = date[..., -1].unsqueeze(-1)
    local_hour = (hour_GMT + 24 * normalized_longitude) % 24
    local_hour_normalized = local_hour / 24

    angle = 2 * torch.pi * local_hour_normalized

    sin_encoding = torch.sin(angle)
    cos_encoding = torch.cos(angle)

    encoded_local_times = torch.stack([sin_encoding, cos_encoding], dim=-1)

    return encoded_local_times


def blanket_day_to_tensor(day, blanket_size, hour=0, year=2022, dt=1):
    r"""Creates a tensor containing the timestamps of a blanket, given its start.

    Arguments:
        day: The day of the year the blanket starts at, between 0 and 364.
        blanket_size: The size K of the blanket.
        hour: The hour of the day, between 0 and 23.
        year: The year of the blanket.
        dt: The time difference between each timestamp in the blanket, in hours.

    Returns:
        A tensor of shape (K, 4) containing the year, month, day, and hour for each
            element in the blanket.

    """
    start_date = datetime(year, 1, 1, hour=hour)
    start_date = start_date + timedelta(day)
    dates = [start_date]
    for i in range(1, blanket_size):
        dates.append(start_date + timedelta(hours=dt * i))

    dates = [[d.year, d.month, d.day, d.hour] for d in dates]

    return torch.tensor(dates)


def add_hours(date: str, hour: int, offset: int) -> tuple[str, int]:
    r"""Adds an offset in hours to a date.

    Arguments:
        date: The date to modify, formatted as "YYYY-MM-DD".
        hour: The hour of the day, between 0 and 23.
        offset: The offset in hours to add.

    Returns:
        A tuple containing the modified date and hour, with the same formats.
    """
    dt = datetime.strptime(date, "%Y-%m-%d")
    dt += timedelta(hours=hour)
    dt += timedelta(hours=offset)

    date = dt.strftime("%Y-%m-%d")
    hour = int(dt.hour)

    return date, hour


def interval_to_tensor(
    start_date: str, end_date: str, start_hour: int = 0, end_hour: int = 23
) -> Tensor:
    r"""Creates a tensor containing the timestamps of an interval.

    Arguments:
        start_date: The start date of the interval, formatted as "YYYY-MM-DD".
        start_hour: The hour of the day at the start of the interval, between 0 and 23.
        end_date: The end date of the interval, formatted as "YYYY-MM-DD".
        end_hour: The hour of the day at the end of the interval, between 0 and 23.

    Returns:
        A tensor of shape (N, 4) containing the year, month, day, and hour for each
            element in the interval.

    """
    start = datetime.strptime(f"{start_date} {start_hour}", "%Y-%m-%d %H")
    end = datetime.strptime(f"{end_date} {end_hour}", "%Y-%m-%d %H")

    dates = []
    while start <= end:
        dates.append([start.year, start.month, start.day, start.hour])
        start += timedelta(hours=1)

    return torch.tensor(dates)


def split_interval(
    nb_splits: int, start_date: str, end_date: str, start_hour: int = 0, end_hour: int = 23
):
    r"""Splits an interval into `nb_splits` parts.

    Arguments:
        nb_splits: The number of splits to create.
        start_date: The start date of the interval, formatted as "YYYY-MM-DD".
        end_date: The end date of the interval, formatted as "YYYY-MM-DD".
        start_hour: The hour of the day at the start of the interval, between 0 and 23.
        end_hour: The hour of the day at the end of the interval, between 0 and 23.

    Returns:
        A list of tuples containing the start and end dates and hours for each split.
    """

    intervals = interval_to_tensor(start_date, end_date, start_hour, end_hour)
    intervals = torch.tensor_split(intervals, nb_splits)

    split_intervals = []
    for interval in intervals:
        start = interval[0]
        end = interval[-1]
        start_date = f"{start[0]:04d}-{start[1]:02d}-{start[2]:02d}"
        end_date = f"{end[0]:04d}-{end[1]:02d}-{end[2]:02d}"
        start_hour = start[3].item()
        end_hour = end[3].item()
        split_intervals.append((start_date, end_date, start_hour, end_hour))

    return split_intervals
