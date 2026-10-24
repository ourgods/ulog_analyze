"""Reconstruct native IMU FIFO samples from logged PX4 sensor_*_fifo packets.

The sampling anchor ('first' or 'last') is explicit; never silently assign
packet timestamps to every sample.
"""
from __future__ import annotations

import re
from typing import Mapping

import numpy as np


def decode_fifo(
    data: Mapping[str, np.ndarray],
    log_start_us: int,
    *,
    anchor: str,
) -> dict[str, object]:
    """Expand packets using valid counts and per-packet dt/scale without resampling."""
    if anchor not in {"first", "last"}:
        raise ValueError("anchor must be first or last")
    for key in ("timestamp_sample", "dt", "scale", "samples"):
        if key not in data:
            raise KeyError(f"FIFO field {key!r} is missing")
    channels = {}
    for axis in "xyz":
        channels[axis] = sorted(
            int(match.group(1)) for field in data
            if (match := re.fullmatch(axis + r"\[(\d+)\]", field))
        )
    if not channels["x"] or any(channels[a] != channels["x"] for a in "yz"):
        raise ValueError("FIFO x/y/z array fields are missing or inconsistent")
    capacity = len(channels["x"])
    if channels["x"] != list(range(capacity)):
        raise ValueError("FIFO array indices are not contiguous")
    counts = np.asarray(data["samples"])
    timestamps = np.asarray(data["timestamp_sample"], dtype=np.float64)
    intervals = np.asarray(data["dt"], dtype=np.float64)
    scales = np.asarray(data["scale"], dtype=np.float64)
    if not (len(counts) == len(timestamps) == len(intervals) == len(scales)):
        raise ValueError("FIFO packet field lengths differ")
    pieces = {"time_s": [], "x": [], "y": [], "z": []}
    packets = 0
    for packet, raw_count in enumerate(counts):
        count = int(raw_count)
        if count == 0:
            continue
        if count < 0 or count > capacity:
            raise ValueError(f"invalid samples={count} at packet {packet}; capacity={capacity}")
        dt, scale, stamp = float(intervals[packet]), float(scales[packet]), float(timestamps[packet])
        if not (np.isfinite(dt) and dt > 0 and np.isfinite(scale) and np.isfinite(stamp)):
            raise ValueError(f"invalid FIFO dt/scale/timestamp at packet {packet}")
        offset = np.arange(count, dtype=np.float64)
        if anchor == "last":
            offset -= count - 1
        pieces["time_s"].append((stamp + offset * dt - log_start_us) / 1e6)
        for axis in "xyz":
            pieces[axis].append(
                np.asarray([data[f"{axis}[{i}]"][packet] for i in range(count)],
                           dtype=np.float64) * scale
            )
        packets += 1
    if not packets:
        raise ValueError("no FIFO samples in selected topic")
    output = {key: np.concatenate(value) for key, value in pieces.items()}
    order = np.argsort(output["time_s"], kind="stable")
    output = {key: value[order] for key, value in output.items()}
    output["packets"] = packets
    output["timestamp_anchor"] = anchor
    output["duplicate_sample_times"] = int(np.count_nonzero(np.diff(output["time_s"]) == 0))
    return output
