"""Shared, configurable calibration and evaluation time axes."""
from datetime import datetime, timedelta


def parse_hour(value):
    value = str(value)
    parsed = datetime.strptime(value, "%Y-%m-%dT%H")
    if parsed.strftime("%Y-%m-%dT%H") != value:
        raise ValueError("Use UTC hours formatted as YYYY-MM-DDTHH")
    return parsed


def time_axis(start, end, step_hours=1):
    if isinstance(step_hours, bool) or int(step_hours) != step_hours or step_hours <= 0:
        raise ValueError("step_hours must be a positive integer")
    first, last = parse_hour(start), parse_hour(end)
    hours = int((last - first).total_seconds() // 3600)
    if hours < 0 or hours % int(step_hours):
        raise ValueError("Time range must be ordered and end on the requested cadence")
    return [first + timedelta(hours=h) for h in range(0, hours + 1, int(step_hours))]


def period_id(start, end):
    time_axis(start, end)
    return f"{start}--{end}"


def calibration_checks(period):
    try:
        start, end = str(period).split("--")
    except ValueError as exc:
        raise ValueError("Calibration period must encode start--end UTC hours") from exc
    return {"period": period_id(start, end), "window_start": start,
            "window_end": end, "expected_hours": len(time_axis(start, end)),
            "time_scope": "hourly", "smoke": False}


def validate_timeline(config):
    timeline = config.timeline
    calibration = time_axis(timeline.calibration.start, timeline.calibration.end)
    evaluation = time_axis(timeline.evaluation.start, timeline.evaluation.end,
                           timeline.evaluation.step_hours)
    if calibration[-1] >= evaluation[0]:
        raise ValueError("Calibration must end before the first evaluation anchor")
    # Sampling uses three preceding and two following context hours. They are
    # inference inputs, never additions to the calibration residual statistics.
    return calibration, evaluation


def select_anchors(config, canonical=False):
    _, complete = validate_timeline(config)
    def boundary(key, default):
        value = config.get(key)
        result = default if canonical or value is None else parse_hour(value)
        if result not in complete:
            raise ValueError(f"{key} must be an anchor in the configured evaluation timeline")
        return result
    first = boundary("anchor_start_time", complete[0])
    last = boundary("anchor_end_time", complete[-1])
    if last < first:
        raise ValueError("anchor_end_time precedes anchor_start_time")
    indices = [i for i, t in enumerate(complete) if first <= t <= last]
    return [complete[i] for i in indices], indices
