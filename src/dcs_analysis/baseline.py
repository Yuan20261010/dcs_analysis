"""Baseline and etalon suppression helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def robust_polynomial_baseline(
    wavenumber_cm: np.ndarray,
    absorbance: np.ndarray,
    *,
    degree: int = 3,
    iterations: int = 4,
    sigma_clip: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a polynomial baseline while down-weighting absorption features."""
    x = np.asarray(wavenumber_cm, dtype=float)
    y = np.asarray(absorbance, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("wavenumber_cm and absorbance must be one-dimensional and aligned")
    xn = 2.0 * (x - x.min()) / max(np.ptp(x), np.finfo(float).eps) - 1.0
    keep = np.isfinite(xn) & np.isfinite(y)
    coeff = np.zeros(degree + 1)
    for _ in range(iterations):
        coeff = np.polynomial.polynomial.polyfit(xn[keep], y[keep], degree)
        baseline = np.polynomial.polynomial.polyval(xn, coeff)
        residual = y - baseline
        scale = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
        if not np.isfinite(scale) or scale == 0:
            break
        keep = np.isfinite(residual) & (np.abs(residual) < sigma_clip * scale)
    baseline = np.polynomial.polynomial.polyval(xn, coeff)
    return y - baseline, baseline


def mfid_baseline_correction(
    wavenumber_cm: np.ndarray,
    absorbance: np.ndarray,
    reference_absorbance: np.ndarray,
    *,
    early_fraction: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a reference-assisted modified-FID style correction.

    The measured-minus-reference residual is transformed to the conjugate
    delay domain. Contributions near zero delay are treated as baseline/etalon
    terms and removed. The exact mask should be tuned with a blank spectrum;
    this function is intentionally explicit rather than silently choosing a
    paper-specific cutoff.
    """
    x = np.asarray(wavenumber_cm, dtype=float)
    y = np.asarray(absorbance, dtype=float)
    ref = np.asarray(reference_absorbance, dtype=float)
    if x.shape != y.shape or y.shape != ref.shape or x.ndim != 1:
        raise ValueError("All m-FID arrays must be one-dimensional and aligned")
    if x.size < 8:
        raise ValueError("m-FID correction requires at least eight spectral points")
    residual = y - ref
    delay = np.fft.fftfreq(x.size, d=float(np.mean(np.diff(x))))
    scale = max(np.max(np.abs(delay)), np.finfo(float).eps)
    mask = np.abs(delay) <= abs(early_fraction) * scale
    baseline = np.fft.fft(np.fft.ifft(residual) * mask).real
    return y - baseline, baseline


def smooth_baseline(
    absorbance: np.ndarray,
    *,
    window_length: int = 101,
    polyorder: int = 2,
) -> np.ndarray:
    """Return a smoothed baseline for diagnostics, not a substitute for m-FID."""
    y = np.asarray(absorbance, dtype=float)
    if y.size < 5:
        return np.full_like(y, np.nanmedian(y))
    window_length = min(window_length, y.size if y.size % 2 else y.size - 1)
    window_length = max(window_length, polyorder + 3)
    if window_length % 2 == 0:
        window_length -= 1
    return savgol_filter(y, window_length, polyorder)
