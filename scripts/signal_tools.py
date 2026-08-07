#!/usr/bin/env python3
"""Validated reusable signal-processing helpers for PX4 log analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class SpectrumResult:
    frequency_hz: np.ndarray
    psd: np.ndarray
    peak_frequency_hz: float
    peak_psd: float
    nperseg: int
    noverlap: int
    segments: int
    frequency_resolution_hz: float


@dataclass(frozen=True)
class TransferResult:
    frequency_hz: np.ndarray
    gain: np.ndarray
    phase_deg: np.ndarray
    coherence: np.ndarray
    nperseg: int
    noverlap: int
    segments: int
    frequency_resolution_hz: float


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite; crop to a common valid window first")
    return array


def segment_count(samples: int, nperseg: int, noverlap: int) -> int:
    if nperseg <= 0 or not 0 <= noverlap < nperseg:
        raise ValueError("require nperseg > 0 and 0 <= noverlap < nperseg")
    if samples < nperseg:
        return 0
    return 1 + (samples - nperseg) // (nperseg - noverlap)


def _segment_plan(samples: int, nperseg: int | None, noverlap: int | None, min_segments: int) -> tuple[int, int, int]:
    if min_segments <= 0:
        raise ValueError("min_segments must be positive")
    if nperseg is None:
        nperseg = min(256, max(16, samples // 4))
    if noverlap is None:
        noverlap = nperseg // 2
    segments = segment_count(samples, nperseg, noverlap)
    if segments < min_segments:
        raise ValueError(
            f"insufficient Welch segments: {segments} < {min_segments}; "
            f"samples={samples}, nperseg={nperseg}, noverlap={noverlap}"
        )
    return nperseg, noverlap, segments


def welch_peak(
    values: Sequence[float],
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    min_segments: int = 4,
    min_low_cycles: float = 4.0,
    detrend: str = "linear",
) -> SpectrumResult:
    """Estimate PSD and the dominant frequency inside a requested band."""

    x = _finite_vector(values, "values")
    if not 0 <= low_hz < high_hz <= fs_hz / 2:
        raise ValueError("frequency band must satisfy 0 <= low < high <= Nyquist")
    if low_hz > 0 and len(x) / fs_hz * low_hz < min_low_cycles:
        raise ValueError(
            f"window contains fewer than {min_low_cycles:g} cycles at the lower band edge "
            f"({len(x) / fs_hz * low_hz:.2f} cycles at {low_hz:g} Hz)"
        )
    nperseg, noverlap, segments = _segment_plan(len(x), nperseg, noverlap, min_segments)
    frequency, psd = signal.welch(
        x,
        fs=fs_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
    )
    band = (frequency >= low_hz) & (frequency <= high_hz)
    if not np.any(band):
        raise ValueError("frequency grid has no bins inside the requested band")
    local_index = int(np.argmax(psd[band]))
    indices = np.flatnonzero(band)
    peak_index = int(indices[local_index])
    return SpectrumResult(
        frequency_hz=frequency,
        psd=psd,
        peak_frequency_hz=float(frequency[peak_index]),
        peak_psd=float(psd[peak_index]),
        nperseg=nperseg,
        noverlap=noverlap,
        segments=segments,
        frequency_resolution_hz=float(frequency[1] - frequency[0]),
    )


def band_rms(
    values: Sequence[float],
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 3,
    min_cycles: float = 5.0,
) -> float:
    """Return zero-phase band-pass RMS after checking window duration."""

    x = _finite_vector(values, "values")
    if not 0 < low_hz < high_hz < fs_hz / 2:
        raise ValueError("band must satisfy 0 < low < high < Nyquist")
    duration_s = len(x) / fs_hz
    if duration_s * low_hz < min_cycles:
        raise ValueError(f"window contains fewer than {min_cycles:g} cycles at {low_hz:g} Hz")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz, output="sos")
    filtered = signal.sosfiltfilt(sos, x - np.mean(x))
    return float(np.sqrt(np.mean(filtered * filtered)))


def transfer_function(
    input_values: Sequence[float],
    output_values: Sequence[float],
    fs_hz: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    min_segments: int = 4,
    detrend: str = "linear",
) -> TransferResult:
    """Estimate input-to-output gain, phase, and coherence with segment validation."""

    x = _finite_vector(input_values, "input_values")
    y = _finite_vector(output_values, "output_values")
    if len(x) != len(y):
        raise ValueError("input and output must have the same length")
    nperseg, noverlap, segments = _segment_plan(len(x), nperseg, noverlap, min_segments)
    frequency, cross = signal.csd(
        x,
        y,
        fs=fs_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
    )
    _, input_psd = signal.welch(
        x,
        fs=fs_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
    )
    _, coherence = signal.coherence(
        x,
        y,
        fs=fs_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
    )
    transfer = np.divide(cross, input_psd, out=np.full_like(cross, np.nan), where=input_psd > 0)
    return TransferResult(
        frequency_hz=frequency,
        gain=np.abs(transfer),
        phase_deg=np.degrees(np.angle(transfer)),
        coherence=coherence,
        nperseg=nperseg,
        noverlap=noverlap,
        segments=segments,
        frequency_resolution_hz=float(frequency[1] - frequency[0]),
    )


def at_frequency(result: TransferResult, frequency_hz: float) -> dict[str, float]:
    index = int(np.argmin(np.abs(result.frequency_hz - frequency_hz)))
    return {
        "frequency_hz": float(result.frequency_hz[index]),
        "gain": float(result.gain[index]),
        "phase_deg": float(result.phase_deg[index]),
        "coherence": float(result.coherence[index]),
        "frequency_resolution_hz": result.frequency_resolution_hz,
    }


__all__ = [
    "SpectrumResult",
    "TransferResult",
    "at_frequency",
    "band_rms",
    "segment_count",
    "transfer_function",
    "welch_peak",
]
