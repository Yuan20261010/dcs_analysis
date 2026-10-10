"""FFT and optical-axis reconstruction for dual-comb spectra."""

from __future__ import annotations

import numpy as np

from .types import DCSConfig, ProcessedSpectrum


def rf_to_wavenumber(
    rf_hz: np.ndarray,
    config: DCSConfig,
) -> np.ndarray:
    """Map RF beat frequency to optical wavenumber in cm^-1."""
    rf_hz = np.asarray(rf_hz, dtype=float)
    if config.optical_anchor_wavenumber_cm is None:
        raise ValueError("optical_anchor_wavenumber_cm must be calibrated")
    return config.optical_anchor_wavenumber_cm + (
        (rf_hz - config.rf_anchor_hz)
        * config.optical_to_rf_scale
        / config.c_cm_s
    )


def _window(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    if name in {"rect", "none"}:
        return np.ones(n)
    raise ValueError(f"Unsupported FFT window: {name}")


def spectrum_from_interferogram(
    coherent_interferogram: np.ndarray,
    config: DCSConfig,
    *,
    n_averaged: int = 1,
    diagnostics: dict | None = None,
) -> ProcessedSpectrum:
    """Fourier transform a corrected complex IGM and build the optical axis."""
    trace = np.asarray(coherent_interferogram)
    if trace.ndim != 1:
        raise ValueError("coherent_interferogram must be one-dimensional")
    tapered = trace * _window(config.window, trace.size)
    full = np.fft.fft(tapered)
    rf = np.fft.fftfreq(trace.size, d=1.0 / config.fs_hz)
    mask = rf >= max(0.0, config.rf_min_hz)
    if config.rf_max_hz is not None:
        mask &= rf <= config.rf_max_hz
    if not np.any(mask):
        raise ValueError("RF limits exclude all FFT bins")
    rf = rf[mask]
    values = full[mask]
    order = np.argsort(rf)
    rf = rf[order]
    values = values[order]
    return ProcessedSpectrum(
        wavenumber_cm=rf_to_wavenumber(rf, config),
        rf_hz=rf,
        complex_spectrum=values,
        coherent_interferogram=trace,
        n_averaged=n_averaged,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def transmission_and_absorbance(
    sample: ProcessedSpectrum,
    background: ProcessedSpectrum,
    *,
    floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common wavenumber axis, intensity transmission and absorbance."""
    if not np.allclose(sample.wavenumber_cm, background.wavenumber_cm):
        background_values = np.interp(
            sample.wavenumber_cm,
            background.wavenumber_cm,
            np.abs(background.complex_spectrum),
            left=np.nan,
            right=np.nan,
        )
    else:
        background_values = np.abs(background.complex_spectrum)
    sample_values = np.abs(sample.complex_spectrum)
    transmission = sample_values / np.maximum(background_values, floor)
    absorbance = -np.log(np.maximum(transmission, floor))
    return sample.wavenumber_cm, transmission, absorbance
