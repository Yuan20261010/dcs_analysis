"""Typed data contracts used by the DCS processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DCSConfig:
    """Instrument and processing parameters.

    The absolute optical anchor is a calibration parameter. It cannot be
    inferred from ``f_rep`` and ``Delta f_rep`` alone.
    """

    fs_hz: float
    frep_hz: float
    delta_frep_hz: float | None = None
    block_samples: int | None = None
    optical_anchor_wavenumber_cm: float | None = None
    rf_anchor_hz: float = 0.0
    rf_min_hz: float = 0.0
    rf_max_hz: float | None = None
    phase_sign: float = -1.0
    timing_sign: float = -1.0
    window: str = "hann"
    reference_wavelengths_m: tuple[float, float] | None = None
    beat_note_centers_hz: tuple[float, ...] = ()
    beat_note_bandwidth_hz: float = 5e6
    baseline_method: str = "mfid"
    baseline_que_frequency_cutoff: float = 0.02
    optical_wavelength_min_m: float | None = None
    optical_wavelength_max_m: float | None = None
    adc_bits: int | None = None
    adc_full_scale_peak_v: float | None = None
    adc_input_impedance_ohm: float | None = None
    adc_analog_bandwidth_hz: float | None = None
    adc_record_alignment_samples: int | None = None
    adc_min_record_samples: int | None = None
    adc_storage_bytes_per_sample: int | None = None
    detector_rf_low_cutoff_hz: float | None = None
    detector_rf_high_cutoff_hz: float | None = None
    detector_load_ohm: float | None = None
    detector_voltage_responsivity_v_w: float | None = None

    @property
    def optical_to_rf_scale(self) -> float:
        if self.delta_frep_hz is None:
            raise ValueError("delta_frep_hz must be measured before optical-axis mapping")
        return self.frep_hz / self.delta_frep_hz

    @property
    def interferogram_period_s(self) -> float:
        if self.delta_frep_hz is None:
            raise ValueError("delta_frep_hz must be measured before IGM timing is calculated")
        return 1.0 / self.delta_frep_hz

    @property
    def adc_nyquist_hz(self) -> float:
        return self.fs_hz / 2.0

    @property
    def dcs_nyquist_hz(self) -> float:
        """One-sided RF limit for unambiguous real-valued DCS detection."""
        return self.frep_hz / 2.0

    @property
    def adc_lsb_v(self) -> float | None:
        if self.adc_bits is None or self.adc_full_scale_peak_v is None:
            return None
        return 2.0 * self.adc_full_scale_peak_v / (2**self.adc_bits)

    @property
    def c_cm_s(self) -> float:
        return 2.99792458e10

    def validate(self) -> None:
        if self.fs_hz <= 0 or self.frep_hz <= 0:
            raise ValueError("Sampling and repetition frequencies must be positive")
        if self.delta_frep_hz is not None and self.delta_frep_hz <= 0:
            raise ValueError("delta_frep_hz must be positive when provided")
        if self.block_samples is not None and self.block_samples < 8:
            raise ValueError("block_samples is too small for an interferogram")
        if self.reference_wavelengths_m is not None:
            if len(self.reference_wavelengths_m) != 2:
                raise ValueError("Exactly two CW reference wavelengths are required")
            if self.reference_wavelengths_m[0] == self.reference_wavelengths_m[1]:
                raise ValueError("CW reference wavelengths must be distinct")
        if self.rf_max_hz is not None and self.rf_max_hz <= self.rf_min_hz:
            raise ValueError("rf_max_hz must be greater than rf_min_hz")
        if self.rf_max_hz is not None and self.rf_max_hz > self.adc_nyquist_hz:
            raise ValueError("rf_max_hz exceeds the ADC Nyquist frequency fs_hz/2")
        if self.rf_max_hz is not None and self.rf_max_hz > self.dcs_nyquist_hz:
            raise ValueError(
                "rf_max_hz exceeds the unambiguous one-sided DCS limit frep_hz/2"
            )
        if self.adc_bits is not None and self.adc_bits <= 0:
            raise ValueError("adc_bits must be positive")
        if self.adc_full_scale_peak_v is not None and self.adc_full_scale_peak_v <= 0:
            raise ValueError("adc_full_scale_peak_v must be positive")
        if (
            self.optical_wavelength_min_m is not None
            and self.optical_wavelength_max_m is not None
            and not 0 < self.optical_wavelength_min_m < self.optical_wavelength_max_m
        ):
            raise ValueError("Optical wavelength limits must satisfy 0 < min < max")

    def validate_processing_ready(self) -> None:
        """Validate calibrated quantities required by the numerical pipeline."""
        self.validate()
        missing: list[str] = []
        if self.delta_frep_hz is None:
            missing.append("delta_frep_hz")
        if self.block_samples is None:
            missing.append("block_samples")
        if self.optical_anchor_wavenumber_cm is None:
            missing.append("optical_anchor_wavenumber_cm")
        if self.reference_wavelengths_m is None:
            missing.append("reference_wavelengths_m")
        if missing:
            raise ValueError(
                "Configuration is a hardware template; calibrate: " + ", ".join(missing)
            )


@dataclass
class Acquisition:
    """One or more synchronized interferograms and reference channels.

    ``interferograms`` has shape ``(n_igm, n_samples)``. Reference channels
    may be supplied either as two already-demodulated phase traces or as four
    complex/raw beat-note channels in the order
    ``ref1_comb1, ref1_comb2, ref2_comb1, ref2_comb2``.
    """

    interferograms: np.ndarray
    reference_phase_1: np.ndarray | None = None
    reference_phase_2: np.ndarray | None = None
    beat_notes: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.interferograms = np.asarray(self.interferograms)
        if self.interferograms.ndim != 2:
            raise ValueError("interferograms must have shape (n_igm, n_samples)")
        n_igm = self.interferograms.shape[0]
        for name in ("reference_phase_1", "reference_phase_2"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                if value.shape != (n_igm,):
                    raise ValueError(f"{name} must have shape ({n_igm},)")
                setattr(self, name, value.astype(float, copy=False))
        if self.beat_notes is not None:
            self.beat_notes = np.asarray(self.beat_notes)
            if self.beat_notes.ndim != 3 or self.beat_notes.shape[0] != 4:
                raise ValueError("beat_notes must have shape (4, n_igm, n_samples)")
            if self.beat_notes.shape[1:] != self.interferograms.shape:
                raise ValueError("beat_notes and interferograms must be synchronized")
        if self.reference_phase_1 is None and self.beat_notes is None:
            raise ValueError("Provide reference phases or four beat-note channels")


@dataclass
class ProcessedSpectrum:
    wavenumber_cm: np.ndarray
    rf_hz: np.ndarray
    complex_spectrum: np.ndarray
    coherent_interferogram: np.ndarray
    n_averaged: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.complex_spectrum)


@dataclass(frozen=True)
class SpectralLine:
    species: str
    center_cm: float
    strength: float
    gamma_air_cm_atm: float = 0.07
    temperature_exponent: float = 0.5
    doppler_sigma_cm: float = 0.003


@dataclass
class FitResult:
    species: tuple[str, ...]
    mole_fractions: dict[str, float]
    baseline_coefficients: np.ndarray
    residual_rms: float
    success: bool
    message: str
    covariance: np.ndarray | None = None
