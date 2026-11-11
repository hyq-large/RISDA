import torch
from torch import Tensor

BASE_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def get_year_progress_encoding(date: torch.Tensor) -> torch.Tensor:
    (year, month, day, hour) = date.unbind(dim=-1)
    is_leap_year = (year % 4 == 0) & (year % 100 != 0) | (year % 400 == 0)
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
