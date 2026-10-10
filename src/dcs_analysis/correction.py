"""Interferogram timing/phase correction and coherent averaging."""

from __future__ import annotations

import numpy as np


def correct_interferograms(
    interferograms: np.ndarray,
    phase_rad: np.ndarray,
    timing_s: np.ndarray,
    fs_hz: float,
    *,
    phase_sign: float = -1.0,
    timing_sign: float = -1.0,
) -> np.ndarray:
    """Correct each IGM in the frequency domain.

    A positive ``timing_s`` means the measured waveform is delayed relative
    to the reference. The frequency-domain correction uses the same sign
    convention as the simulation and can be inverted with the two sign
    parameters when a vendor's beat-note convention is opposite.
    """
    x = np.asarray(interferograms)
    phase_rad = np.asarray(phase_rad, dtype=float)
    timing_s = np.asarray(timing_s, dtype=float)
    if x.ndim != 2 or phase_rad.shape != (x.shape[0],) or timing_s.shape != (x.shape[0],):
        raise ValueError("Input shapes do not match (n_igm, n_samples)")
    n_samples = x.shape[1]
    rf_hz = np.fft.fftfreq(n_samples, d=1.0 / fs_hz)
    spectrum = np.fft.fft(x, axis=1)
    timing_factor = np.exp(
        1j * timing_sign * 2.0 * np.pi * timing_s[:, None] * rf_hz[None, :]
    )
    phase_factor = np.exp(1j * phase_sign * phase_rad)[:, None]
    corrected = np.fft.ifft(spectrum * timing_factor * phase_factor, axis=1)
    return corrected


def coherent_average(
    interferograms: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Return a complex coherent average and per-IGM phase quality metric."""
    x = np.asarray(interferograms)
    if x.ndim != 2:
        raise ValueError("interferograms must be two-dimensional")
    if weights is None:
        weights = np.ones(x.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (x.shape[0],):
        raise ValueError("weights must have shape (n_igm,)")
    valid = np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("No valid interferograms remain")
    w = weights[valid]
    average = np.sum(x[valid] * w[:, None], axis=0) / np.sum(w)
    phasor = np.sum(np.exp(1j * np.angle(x[valid, x.shape[1] // 2])) * w) / np.sum(w)
    quality = np.array([float(np.abs(phasor))], dtype=float)
    return average, int(np.count_nonzero(valid)), quality
