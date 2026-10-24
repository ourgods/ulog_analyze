#!/usr/bin/env python3
"""Reusable, cost-aware PX4 ULog data-access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import signal as scipy_signal
from pyulog import ULog


PathLike = str | Path


@dataclass(frozen=True)
class SignalSpec:
    """Declarative scalar field selection."""

    label: str
    topic: str
    field: str
    instance: int = 0
    scale: float = 1.0
    offset: float = 0.0
    interpolation: str = "linear"  # linear for continuous, previous for discrete


@dataclass(frozen=True)
class Series:
    spec: SignalSpec
    time_s: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class AlignedSignals:
    time_s: np.ndarray
    values: Mapping[str, np.ndarray]
    source: Mapping[str, Series]
    sampling: Mapping[str, Mapping[str, float | int | None]]
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChangeEvent:
    label: str
    time_s: float
    value: Any


@dataclass(frozen=True)
class TimelineResult:
    events: tuple[ChangeEvent, ...]
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class StateInterval:
    label: str
    start_s: float
    end_s: float
    value: Any


@dataclass(frozen=True)
class StatsResult:
    values: Mapping[str, Mapping[str, float | int | str | None]]
    missing: tuple[str, ...] = ()


def read_header(path: PathLike) -> ULog:
    """Read definitions, metadata, and initial parameters without flight data."""

    return ULog(str(Path(path)), parse_header_only=True)


def scan_log(path: PathLike, topics: Iterable[str] = ()) -> ULog:
    """Scan the file and retain only explicitly named topic arrays."""

    names = sorted(set(topics))
    return ULog(str(Path(path)), message_name_filter_list=names)


def load_topics(path: PathLike, topics: Iterable[str]) -> ULog:
    """Load explicitly selected topics; reject accidental all-topic loading."""

    if topics is None:
        raise ValueError("topics must be explicit; use bare pyulog.ULog only for deliberate all-topic loading")
    names = sorted(set(topics))
    if not names:
        raise ValueError("topics is empty; use scan_log() for a metadata-only full-file scan")
    return ULog(str(Path(path)), message_name_filter_list=names)


def load_for_specs(path: PathLike, specs: Sequence[SignalSpec]) -> ULog:
    """Load the explicit topic union declared by SignalSpec objects."""

    if not specs:
        raise ValueError("at least one SignalSpec is required")
    return load_topics(path, {spec.topic for spec in specs})


def _expanded_format_fields(message_format: Any) -> list[str]:
    fields: list[str] = []
    for _type_name, array_length, field_name in message_format.fields:
        if array_length:
            fields.extend(f"{field_name}[{index}]" for index in range(array_length))
        else:
            fields.append(field_name)
    return fields


def available_schemas(path: PathLike, include_padding: bool = False) -> dict[str, list[str]]:
    """Return header-declared message schemas without loading topic samples."""

    header = read_header(path)
    schemas = {
        name: _expanded_format_fields(message_format)
        for name, message_format in header.message_formats.items()
    }
    if not include_padding:
        schemas = {
            name: [field for field in fields if not field.startswith("_padding")]
            for name, fields in schemas.items()
        }
    return dict(sorted(schemas.items()))


def get_dataset(log: ULog, topic: str, instance: int = 0, required: bool = True):
    dataset = next(
        (item for item in log.data_list if item.name == topic and item.multi_id == instance),
        None,
    )
    if dataset is None and required:
        loaded = sorted({f"{item.name}[{item.multi_id}]" for item in log.data_list})
        raise KeyError(f"topic {topic}[{instance}] not loaded or not present; loaded={loaded}")
    return dataset


def get_topic(log: ULog, topic: str, instance: int = 0, required: bool = True):
    dataset = get_dataset(log, topic, instance, required)
    return None if dataset is None else dataset.data


def topic_instances(log: ULog, topic: str) -> list[int]:
    return sorted(item.multi_id for item in log.data_list if item.name == topic)


def relative_time_s(log: ULog, topic_data: Mapping[str, np.ndarray]) -> np.ndarray:
    if "timestamp" not in topic_data:
        raise KeyError("topic has no timestamp field")
    return np.asarray(topic_data["timestamp"], dtype=np.float64) / 1e6 - log.start_timestamp / 1e6


def read_series(
    log: ULog,
    spec: SignalSpec,
    start_s: float | None = None,
    end_s: float | None = None,
    required: bool = True,
) -> Series | None:
    """Read one scalar field and optionally crop it in log-relative seconds."""

    if spec.interpolation not in {"linear", "previous"}:
        raise ValueError(f"unsupported interpolation {spec.interpolation!r}")
    data = get_topic(log, spec.topic, spec.instance, required=required)
    if data is None:
        return None
    if spec.field not in data:
        if not required:
            return None
        raise KeyError(f"field {spec.field!r} missing from {spec.topic}[{spec.instance}]")
    time_s = relative_time_s(log, data)
    values = np.asarray(data[spec.field], dtype=np.float64) * spec.scale + spec.offset
    if values.ndim != 1:
        raise ValueError(f"field {spec.field!r} is not scalar; select an expanded field such as xyz[2]")
    mask = np.isfinite(time_s)
    if start_s is not None:
        mask &= time_s >= start_s
    if end_s is not None:
        mask &= time_s <= end_s
    return Series(spec=spec, time_s=time_s[mask], values=values[mask])


def sampling_info(time_s: Sequence[float], gap_factor: float = 5.0) -> dict[str, float | int | None]:
    time = np.asarray(time_s, dtype=np.float64)
    finite = time[np.isfinite(time)]
    if len(finite) < 2:
        return {"samples": int(len(finite)), "span_s": None, "median_hz": None, "max_gap_s": None, "large_gaps": 0}
    delta = np.diff(finite)
    positive = delta[delta > 0]
    if len(positive) == 0:
        return {"samples": int(len(finite)), "span_s": float(finite[-1] - finite[0]), "median_hz": None, "max_gap_s": None, "large_gaps": 0}
    median_dt = float(np.median(positive))
    return {
        "samples": int(len(finite)),
        "span_s": float(finite[-1] - finite[0]),
        "median_hz": 1.0 / median_dt,
        "max_gap_s": float(np.max(positive)),
        "large_gaps": int(np.sum(positive > gap_factor * median_dt)),
    }


def change_points(
    time_s: Sequence[float],
    values: Sequence[Any],
    include_first: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return times and values where a discrete series changes."""

    time = np.asarray(time_s)
    value = np.asarray(values)
    if len(time) != len(value):
        raise ValueError("time and values must have the same length")
    if len(value) == 0:
        return time, value
    equal = value[1:] == value[:-1]
    if np.issubdtype(value.dtype, np.number):
        equal |= np.isnan(value[1:]) & np.isnan(value[:-1])
    changed = np.r_[include_first, ~equal] if include_first else np.r_[False, ~equal]
    return time[changed], value[changed]


def state_intervals(
    time_s: Sequence[float],
    values: Sequence[Any],
    expected: Any,
    *,
    label: str = "state",
    series_end_s: float | None = None,
) -> tuple[StateInterval, ...]:
    """Return intervals where a discrete previous-value signal equals expected.

    A sparse discrete topic remains in its last published state until the next
    state change. Pass the enclosing log/window end as ``series_end_s`` when the
    final matching state should extend beyond the topic's last sample.
    """

    time = np.asarray(time_s, dtype=np.float64)
    value = np.asarray(values)
    if time.ndim != 1 or value.ndim != 1:
        raise ValueError("time_s and values must be one-dimensional")
    if len(time) != len(value):
        raise ValueError("time_s and values must have the same length")
    finite_time = np.isfinite(time)
    time, value = time[finite_time], value[finite_time]
    if len(time) == 0:
        return ()
    order = np.argsort(time, kind="stable")
    time, value = time[order], value[order]
    keep = np.r_[time[1:] != time[:-1], True]
    time, value = time[keep], value[keep]
    selected = value == expected
    starts = np.flatnonzero(selected & np.r_[True, ~selected[:-1]])
    intervals: list[StateInterval] = []
    for start in starts:
        later = np.flatnonzero((np.arange(len(value)) > start) & ~selected)
        end = float(time[later[0]]) if len(later) else float(time[-1] if series_end_s is None else series_end_s)
        if end < time[start]:
            raise ValueError(f"series_end_s {end} precedes matching state start {time[start]}")
        stored_value = expected.item() if isinstance(expected, np.generic) else expected
        intervals.append(StateInterval(label=label, start_s=float(time[start]), end_s=end, value=stored_value))
    return tuple(intervals)


def change_timeline(
    log: ULog,
    specs: Sequence[SignalSpec],
    start_s: float | None = None,
    end_s: float | None = None,
    on_missing: str = "error",
) -> TimelineResult:
    """Collect and sort changes from multiple discrete scalar fields."""

    if on_missing not in {"error", "skip"}:
        raise ValueError("on_missing must be 'error' or 'skip'")
    events: list[ChangeEvent] = []
    missing: list[str] = []
    for spec in specs:
        series = read_series(log, spec, start_s, end_s, required=False)
        if series is None or len(series.time_s) == 0:
            missing.append(spec.label)
            continue
        times, values = change_points(series.time_s, series.values)
        events.extend(
            ChangeEvent(
                label=spec.label,
                time_s=float(timestamp),
                value=value.item() if isinstance(value, np.generic) else value,
            )
            for timestamp, value in zip(times, values)
        )
    if missing and on_missing == "error":
        raise ValueError(f"signals missing or empty in the requested window: {missing}")
    return TimelineResult(events=tuple(sorted(events, key=lambda event: (event.time_s, event.label))), missing=tuple(missing))


def _clean_for_resample(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(time_s) & np.isfinite(values)
    time = np.asarray(time_s[valid], dtype=np.float64)
    value = np.asarray(values[valid], dtype=np.float64)
    if len(time) == 0:
        return time, value
    order = np.argsort(time, kind="stable")
    time, value = time[order], value[order]
    keep = np.r_[time[1:] != time[:-1], True]  # keep the last sample at duplicate timestamps
    return time[keep], value[keep]


def _linear_bridge_limit_s(time_s: np.ndarray, gap_factor: float) -> float:
    positive_dt = np.diff(time_s)
    positive_dt = positive_dt[positive_dt > 0]
    if len(positive_dt) == 0:
        return 0.0
    median_dt = float(np.median(positive_dt))
    # A two-interval series can have its median dominated by one dropout.
    # With ordinary log streams, keep the same median convention as sampling_info().
    if len(positive_dt) <= 2:
        median_dt = float(np.median(positive_dt[positive_dt <= median_dt]))
    return gap_factor * median_dt


def anti_alias_samples(time_s: np.ndarray, values: np.ndarray, target_hz: float) -> np.ndarray:
    """Apply a low-pass before decimation, separately across logging gaps.

    A nonuniform source is rejected rather than silently filtering it at a
    fictitious constant sample rate.
    """
    if target_hz <= 0 or not np.isfinite(target_hz):
        raise ValueError("target_hz must be positive and finite")
    if len(time_s) < 2:
        return values
    dt = np.diff(time_s)
    positive = dt[dt > 0]
    if not len(positive):
        raise ValueError("anti-aliasing requires increasing timestamps")
    period = float(np.median(positive))
    source_hz = 1.0 / period
    if source_hz <= target_hz * 1.1:
        return values
    regular_dt = positive[positive <= period * 5]
    if np.any(abs(regular_dt - period) > .10 * period):
        raise ValueError("cannot anti-alias irregularly sampled source safely; select a native source/window")
    sos = scipy_signal.butter(6, target_hz * .4, fs=source_hz,
                              btype="lowpass", output="sos")
    filtered = values.copy()
    cuts = np.r_[0, np.flatnonzero(dt > period * 5) + 1, len(values)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b-a < 25:
            raise ValueError("too few contiguous samples for anti-aliasing")
        filtered[a:b] = scipy_signal.sosfiltfilt(sos, values[a:b])
    return filtered


def resample_series(
    series: Series,
    grid_s: Sequence[float],
    *,
    max_linear_gap_factor: float = 5.0,
    anti_alias: bool = False,
) -> np.ndarray:
    """Align one series without extrapolation or interpolation across dropouts."""

    grid = np.asarray(grid_s, dtype=np.float64)
    time, values = _clean_for_resample(series.time_s, series.values)
    if anti_alias and series.spec.interpolation == "linear" and len(grid) >= 2:
        step = np.diff(grid)
        if np.any(step <= 0) or not np.all(np.isfinite(step)):
            raise ValueError("anti-aliasing requires increasing output timestamps")
        period = float(np.median(step))
        if np.max(np.abs(step-period)) > .001*period:
            raise ValueError("anti-aliasing requires a uniform output grid")
        values = anti_alias_samples(time, values, 1.0/period)
    output = np.full(len(grid), np.nan, dtype=np.float64)
    if not np.isfinite(max_linear_gap_factor) or max_linear_gap_factor <= 0:
        raise ValueError("max_linear_gap_factor must be finite and positive")
    if len(time) == 0:
        return output
    inside = (grid >= time[0]) & (grid <= time[-1])
    if series.spec.interpolation == "linear":
        if len(time) == 1:
            output[np.isclose(grid, time[0])] = values[0]
        else:
            # Never fabricate a continuous signal through a logging dropout.
            # Exact source samples remain usable even when their neighbours are far away.
            source_index = np.searchsorted(time, grid, side="left")
            right_index = np.clip(source_index, 0, len(time) - 1)
            left_index = np.clip(source_index - 1, 0, len(time) - 1)
            exact = (np.isclose(grid, time[right_index], rtol=0, atol=1e-9)
                     | np.isclose(grid, time[left_index], rtol=0, atol=1e-9))
            bridge_limit_s = _linear_bridge_limit_s(time, max_linear_gap_factor)
            between = inside & ~exact & (source_index > 0) & (source_index < len(time))
            bridgeable = np.zeros(len(grid), dtype=bool)
            bridgeable[between] = (
                time[source_index[between]] - time[source_index[between] - 1]
            ) <= bridge_limit_s
            allowed = inside & (exact | bridgeable)
            output[allowed] = np.interp(grid[allowed], time, values)
    else:
        indices = np.searchsorted(time, grid[inside], side="right") - 1
        output[inside] = values[indices]
    return output


def align_signals(
    log: ULog,
    specs: Sequence[SignalSpec],
    start_s: float | None = None,
    end_s: float | None = None,
    hz: float = 10.0,
    on_missing: str = "error",
    anti_alias: bool = False,
) -> AlignedSignals:
    """Crop and align scalar fields on a common grid."""

    if not specs:
        raise ValueError("at least one SignalSpec is required")
    if hz <= 0:
        raise ValueError("hz must be positive")
    if on_missing not in {"error", "skip"}:
        raise ValueError("on_missing must be 'error' or 'skip'")
    if len({spec.label for spec in specs}) != len(specs):
        raise ValueError("SignalSpec labels must be unique")
    source: dict[str, Series] = {}
    missing: list[str] = []
    for spec in specs:
        series = read_series(log, spec, start_s, end_s, required=False)
        if series is None or len(series.time_s) == 0:
            missing.append(spec.label)
        else:
            source[spec.label] = series
    if missing and on_missing == "error":
        raise ValueError(f"signals missing or empty in the requested window: {missing}")
    if not source:
        raise ValueError("no requested signals are available")
    nonempty = list(source.values())
    common_start = max(series.time_s[0] for series in nonempty) if start_s is None else start_s
    common_end = min(series.time_s[-1] for series in nonempty) if end_s is None else end_s
    if common_end <= common_start:
        raise ValueError(f"empty common time range: {common_start}..{common_end}s")
    step = 1.0 / hz
    grid = np.arange(common_start, common_end + step * 0.25, step)
    values = {label: resample_series(series, grid, anti_alias=anti_alias) for label, series in source.items()}
    sampling = {label: sampling_info(series.time_s) for label, series in source.items()}
    return AlignedSignals(time_s=grid, values=values, source=source, sampling=sampling, missing=tuple(missing))


def finite_values(values: Sequence[float]) -> np.ndarray:
    """Return finite scalar values as float64."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return array[np.isfinite(array)]


def unwrap_finite(values: Sequence[float], period: float = 2 * np.pi) -> np.ndarray:
    """Unwrap contiguous finite angle runs without propagating across NaN gaps."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be finite and positive")
    result = np.full_like(array, np.nan)
    indices = np.flatnonzero(np.isfinite(array))
    if len(indices) == 0:
        return result
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    for run in np.split(indices, split):
        result[run] = np.unwrap(array[run], period=period)
    return result


def common_finite(*values: Sequence[float]) -> tuple[np.ndarray, ...]:
    """Apply one shared finite mask to equally sized scalar series."""

    if not values:
        return ()
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("all values must be one-dimensional")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("all values must have the same length")
    mask = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    return tuple(array[mask] for array in arrays)


def stats(values: Sequence[float], detrend: str | None = None) -> dict[str, float | int | str | None]:
    """Return finite-value descriptive statistics, optionally after linear detrending."""

    raw = np.asarray(values, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    result: dict[str, float | int | str | None] = {
        "samples": int(len(raw)),
        "finite": int(len(finite)),
        "nan": int(len(raw) - len(finite)),
        "detrend": detrend,
    }
    if len(finite) == 0:
        result.update({key: None for key in ("mean", "std", "min", "max", "p2p", "p05", "p50", "p95", "p90_span")})
        return result
    work = finite.copy()
    if detrend not in {None, "constant", "linear"}:
        raise ValueError("detrend must be None, 'constant', or 'linear'")
    if detrend == "constant":
        work -= np.mean(work)
    elif detrend == "linear":
        if len(work) < 2:
            raise ValueError("linear detrending requires at least two finite samples")
        index = np.arange(len(work), dtype=np.float64)
        work -= np.polyval(np.polyfit(index, work, 1), index)
    p05, p50, p95 = np.percentile(work, [5, 50, 95])
    result.update(
        {
            "mean": float(np.mean(work)),
            "std": float(np.std(work)),
            "min": float(np.min(work)),
            "max": float(np.max(work)),
            "p2p": float(np.ptp(work)),
            "p05": float(p05),
            "p50": float(p50),
            "p95": float(p95),
            "p90_span": float(p95 - p05),
        }
    )
    return result


def summarize_series(
    log: ULog,
    specs: Sequence[SignalSpec],
    start_s: float | None = None,
    end_s: float | None = None,
    detrend: str | None = None,
    on_missing: str = "error",
) -> StatsResult:
    """Return descriptive statistics for multiple raw fields without resampling."""

    if on_missing not in {"error", "skip"}:
        raise ValueError("on_missing must be 'error' or 'skip'")
    values: dict[str, Mapping[str, float | int | str | None]] = {}
    missing: list[str] = []
    for spec in specs:
        series = read_series(log, spec, start_s, end_s, required=False)
        if series is None or len(series.time_s) == 0:
            missing.append(spec.label)
        else:
            values[spec.label] = stats(series.values, detrend=detrend)
    if missing and on_missing == "error":
        raise ValueError(f"signals missing or empty in the requested window: {missing}")
    return StatsResult(values=values, missing=tuple(missing))


def parameter_diff(
    paths: Sequence[PathLike],
    pattern: str | re.Pattern[str] | None = None,
    include_equal: bool = False,
    labels: Sequence[str] | None = None,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Compare header-stored initial parameters without loading flight data."""

    if abs_tol < 0 or rel_tol < 0:
        raise ValueError("parameter comparison tolerances must be non-negative")
    regex = re.compile(pattern) if isinstance(pattern, str) else pattern
    if labels is None:
        labels = [Path(path).name for path in paths]
    if len(labels) != len(paths):
        raise ValueError("labels must have the same length as paths")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    snapshots = {
        label: dict(read_header(path).initial_parameters)
        for label, path in zip(labels, paths)
    }
    names = sorted(set().union(*(snapshot.keys() for snapshot in snapshots.values())))
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        if regex is not None and not regex.search(name):
            continue
        present = {label: name in snapshot for label, snapshot in snapshots.items()}
        values = {
            label: snapshot[name] if present[label] else None
            for label, snapshot in snapshots.items()
        }
        equal = len(set(present.values())) == 1
        if equal:
            raw_values = list(values.values())
            numeric = all(
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                for value in raw_values
            )
            if numeric:
                reference = float(raw_values[0])
                equal = all(
                    bool(np.isclose(float(value), reference, atol=abs_tol, rtol=rel_tol, equal_nan=True))
                    for value in raw_values[1:]
                )
            else:
                equal = len({repr(value) for value in raw_values}) == 1
        if include_equal or not equal:
            result[name] = values
    return result


def quaternion_wxyz_to_euler(
    w: Sequence[float],
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    degrees: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert PX4 w,x,y,z quaternions to roll, pitch, yaw."""

    wv, xv, yv, zv = (np.asarray(value, dtype=np.float64) for value in (w, x, y, z))
    roll = np.arctan2(2 * (wv * xv + yv * zv), 1 - 2 * (xv * xv + yv * yv))
    pitch = np.arcsin(np.clip(2 * (wv * yv - zv * xv), -1, 1))
    yaw = np.arctan2(2 * (wv * zv + xv * yv), 1 - 2 * (yv * yv + zv * zv))
    if degrees:
        return tuple(np.degrees(value) for value in (roll, pitch, yaw))
    return roll, pitch, yaw


__all__ = [
    "AlignedSignals",
    "ChangeEvent",
    "Series",
    "SignalSpec",
    "StateInterval",
    "StatsResult",
    "TimelineResult",
    "align_signals",
    "available_schemas",
    "change_points",
    "change_timeline",
    "common_finite",
    "finite_values",
    "get_dataset",
    "get_topic",
    "load_topics",
    "load_for_specs",
    "parameter_diff",
    "quaternion_wxyz_to_euler",
    "read_header",
    "read_series",
    "relative_time_s",
    "resample_series",
    "sampling_info",
    "scan_log",
    "stats",
    "state_intervals",
    "summarize_series",
    "topic_instances",
    "unwrap_finite",
]
