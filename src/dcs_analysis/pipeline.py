"""High-level DCS processing pipeline."""

from __future__ import annotations

import numpy as np

from .correction import coherent_average, correct_interferograms
from .phase import estimate_reference_phases, solve_phase_and_timing
from .spectrum import spectrum_from_interferogram
from .types import Acquisition, DCSConfig, ProcessedSpectrum


def process_acquisition(
    acquisition: Acquisition,
    config: DCSConfig,
    *,
    weights: np.ndarray | None = None,
) -> ProcessedSpectrum:
    """Run reference extraction, phase/timing correction, averaging and FFT."""
    config.validate_processing_ready()
    acquisition.validate()
    assert config.block_samples is not None
    if acquisition.interferograms.shape[1] != config.block_samples:
        raise ValueError(
            f"Input has {acquisition.interferograms.shape[1]} samples per IGM, "
            f"but config.block_samples={config.block_samples}"
        )
    phase_1, phase_2 = estimate_reference_phases(acquisition, config)
    phase, timing = solve_phase_and_timing(phase_1, phase_2, config)
    corrected = correct_interferograms(
        acquisition.interferograms,
        phase,
        timing,
        config.fs_hz,
        phase_sign=config.phase_sign,
        timing_sign=config.timing_sign,
    )
    average, n_averaged, quality = coherent_average(corrected, weights=weights)
    diagnostics = {
        "phase_rms_rad": float(np.std(phase)),
        "timing_rms_s": float(np.std(timing)),
        "coherent_quality": float(quality[0]),
        "n_input_igm": int(acquisition.interferograms.shape[0]),
    }
    return spectrum_from_interferogram(
        average,
        config,
        n_averaged=n_averaged,
        diagnostics=diagnostics,
    )
