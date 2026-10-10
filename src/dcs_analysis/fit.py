"""Voigt line-shape models and multi-species least-squares fitting."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import wofz

from .types import FitResult, SpectralLine


def voigt_profile(
    wavenumber_cm: np.ndarray,
    center_cm: float,
    sigma_cm: float,
    gamma_cm: float,
) -> np.ndarray:
    """Unit-area Voigt profile using the Faddeeva function."""
    x = np.asarray(wavenumber_cm, dtype=float)
    sigma_cm = max(float(sigma_cm), np.finfo(float).eps)
    gamma_cm = max(float(gamma_cm), 0.0)
    z = ((x - center_cm) + 1j * gamma_cm) / (sigma_cm * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_cm * np.sqrt(2.0 * np.pi))


def model_absorbance(
    wavenumber_cm: np.ndarray,
    lines: list[SpectralLine],
    mole_fractions: dict[str, float],
    *,
    pressure_atm: float,
    temperature_K: float,
    path_length_cm: float,
    baseline_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Generate absorbance from a compact line table.

    ``strength`` is intentionally a user-supplied effective line strength.
    For quantitative HITRAN work, convert HITRAN intensities and units before
    calling this function and record that conversion in the metadata.
    """
    x = np.asarray(wavenumber_cm, dtype=float)
    y = np.zeros_like(x)
    for line in lines:
        fraction = float(mole_fractions.get(line.species, 0.0))
        gamma = line.gamma_air_cm_atm * pressure_atm * (296.0 / temperature_K) ** line.temperature_exponent
        profile = voigt_profile(x, line.center_cm, line.doppler_sigma_cm, gamma)
        y += line.strength * fraction * pressure_atm * path_length_cm * profile
    if baseline_coefficients is not None:
        baseline_coefficients = np.asarray(baseline_coefficients, dtype=float)
        xn = 2.0 * (x - x.min()) / max(np.ptp(x), np.finfo(float).eps) - 1.0
        y += np.polynomial.polynomial.polyval(xn, baseline_coefficients)
    return y


def load_line_table(path: str | Path) -> list[SpectralLine]:
    """Load a CSV with species, center_cm, strength and optional broadening columns."""
    lines: list[SpectralLine] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            lines.append(
                SpectralLine(
                    species=row["species"].strip(),
                    center_cm=float(row["center_cm"]),
                    strength=float(row["strength"]),
                    gamma_air_cm_atm=float(row.get("gamma_air_cm_atm", 0.07)),
                    temperature_exponent=float(row.get("temperature_exponent", 0.5)),
                    doppler_sigma_cm=float(row.get("doppler_sigma_cm", 0.003)),
                )
            )
    if not lines:
        raise ValueError(f"No spectral lines found in {path}")
    return lines


def fit_mixture(
    wavenumber_cm: np.ndarray,
    absorbance: np.ndarray,
    lines: list[SpectralLine],
    *,
    pressure_atm: float,
    temperature_K: float,
    path_length_cm: float,
    baseline_degree: int = 2,
    initial_mole_fractions: dict[str, float] | None = None,
) -> FitResult:
    """Fit species fractions plus a low-order baseline."""
    x = np.asarray(wavenumber_cm, dtype=float)
    y = np.asarray(absorbance, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("wavenumber_cm and absorbance must be aligned one-dimensional arrays")
    species = tuple(OrderedDict((line.species, None) for line in lines).keys())
    initial_mole_fractions = initial_mole_fractions or {}
    n_species = len(species)
    x0 = np.array([initial_mole_fractions.get(name, 0.01) for name in species] + [0.0] * (baseline_degree + 1))
    lower = np.array([0.0] * n_species + [-np.inf] * (baseline_degree + 1))
    upper = np.array([1.0] * n_species + [np.inf] * (baseline_degree + 1))

    xn = 2.0 * (x - x.min()) / max(np.ptp(x), np.finfo(float).eps) - 1.0

    def residual(parameters: np.ndarray) -> np.ndarray:
        fractions = dict(zip(species, parameters[:n_species]))
        baseline = np.polynomial.polynomial.polyval(xn, parameters[n_species:])
        return model_absorbance(
            x,
            lines,
            fractions,
            pressure_atm=pressure_atm,
            temperature_K=temperature_K,
            path_length_cm=path_length_cm,
        ) + baseline - y

    result = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1")
    covariance = None
    if result.success and result.jac.shape[0] > result.jac.shape[1]:
        try:
            scale = np.sum(result.fun**2) / (result.jac.shape[0] - result.jac.shape[1])
            covariance = np.linalg.pinv(result.jac.T @ result.jac) * scale
        except np.linalg.LinAlgError:
            covariance = None
    return FitResult(
        species=species,
        mole_fractions=dict(zip(species, result.x[:n_species])),
        baseline_coefficients=result.x[n_species:],
        residual_rms=float(np.sqrt(np.mean(result.fun**2))),
        success=bool(result.success),
        message=str(result.message),
        covariance=covariance,
    )
