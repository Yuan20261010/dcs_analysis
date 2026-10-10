"""Derived hardware and dual-comb design limits."""

from __future__ import annotations

import math
from typing import Any

from .types import DCSConfig


def hardware_report(config: DCSConfig) -> dict[str, Any]:
    """Return derived limits and calibration warnings for an instrument config."""
    config.validate()
    limits = [config.adc_nyquist_hz, config.dcs_nyquist_hz]
    if config.adc_analog_bandwidth_hz is not None:
        limits.append(config.adc_analog_bandwidth_hz)
    if config.detector_rf_high_cutoff_hz is not None:
        limits.append(config.detector_rf_high_cutoff_hz)

    report: dict[str, Any] = {
        "sampling_rate_hz": config.fs_hz,
        "adc_nyquist_hz": config.adc_nyquist_hz,
        "repetition_rate_hz": config.frep_hz,
        "unambiguous_one_sided_rf_limit_hz": min(limits),
        "adc_lsb_v": config.adc_lsb_v,
        "delta_frep_hz": config.delta_frep_hz,
        "block_samples": config.block_samples,
        "warnings": [],
    }
    warnings: list[str] = report["warnings"]

    if config.delta_frep_hz is None:
        warnings.append("Measure delta_frep_hz; the IGM period and optical scale are unresolved.")
    else:
        report["interferogram_period_s"] = config.interferogram_period_s
        report["samples_per_interferogram_exact"] = config.fs_hz / config.delta_frep_hz
        if config.adc_record_alignment_samples is not None:
            alignment = config.adc_record_alignment_samples
            report["recommended_aligned_block_samples"] = (
                math.ceil(report["samples_per_interferogram_exact"] / alignment) * alignment
            )
        report["unaliased_optical_bandwidth_hz"] = (
            config.frep_hz**2 / (2.0 * config.delta_frep_hz)
        )
        report["unaliased_optical_bandwidth_cm-1"] = (
            report["unaliased_optical_bandwidth_hz"] / config.c_cm_s
        )

    if config.block_samples is None:
        warnings.append("Choose block_samples after delta_frep_hz is measured.")
    else:
        report["record_duration_s"] = config.block_samples / config.fs_hz
        if (
            config.adc_min_record_samples is not None
            and config.block_samples < config.adc_min_record_samples
        ):
            warnings.append("block_samples is below the digitizer minimum record length.")
        if (
            config.adc_record_alignment_samples is not None
            and config.block_samples % config.adc_record_alignment_samples
        ):
            warnings.append("block_samples does not satisfy the digitizer record alignment.")
        if config.adc_storage_bytes_per_sample is not None:
            report["record_bytes_per_channel"] = (
                config.block_samples * config.adc_storage_bytes_per_sample
            )

    lo = config.optical_wavelength_min_m
    hi = config.optical_wavelength_max_m
    if lo is not None and hi is not None:
        optical_bandwidth = config.c_cm_s * 1e-2 * (1.0 / lo - 1.0 / hi)
        max_delta = config.frep_hz**2 / (2.0 * optical_bandwidth)
        report["target_optical_bandwidth_hz"] = optical_bandwidth
        report["max_delta_frep_for_target_hz"] = max_delta
        report["min_igm_samples_for_target"] = math.ceil(config.fs_hz / max_delta)
        if config.adc_record_alignment_samples is not None:
            alignment = config.adc_record_alignment_samples
            report["min_aligned_igm_samples_for_target"] = (
                math.ceil(report["min_igm_samples_for_target"] / alignment) * alignment
            )
        if config.delta_frep_hz is not None and config.delta_frep_hz > max_delta:
            warnings.append(
                "delta_frep_hz is too large for alias-free coverage of the configured "
                "optical wavelength span; narrow the optical band or reduce delta_frep_hz."
            )

    if config.optical_anchor_wavenumber_cm is None:
        warnings.append("Calibrate optical_anchor_wavenumber_cm before assigning line positions.")
    if config.reference_wavelengths_m is None:
        warnings.append("Enter the two measured CW reference wavelengths used for phase correction.")
    if config.rf_max_hz is None:
        warnings.append("Set rf_max_hz at or below the reported RF limit.")
    return report
