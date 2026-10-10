"""Small deterministic DCS dataset generator for pipeline development."""

from __future__ import annotations

import numpy as np

from .types import Acquisition, DCSConfig


def simulate_acquisition(
    config: DCSConfig,
    *,
    n_igm: int = 32,
    rf_tones_hz: tuple[float, ...] = (20e3, 35e3, 58e3),
    seed: int = 1234,
) -> Acquisition:
    """Generate complex baseband IGMs with known phase and timing drift."""
    config.validate_processing_ready()
    rng = np.random.default_rng(seed)
    n = config.block_samples
    assert n is not None
    assert config.reference_wavelengths_m is not None
    t = np.arange(n, dtype=float) / config.fs_hz
    index = np.arange(n_igm, dtype=float)
    phase0 = 0.35 * np.sin(2.0 * np.pi * index / max(n_igm, 1)) + 0.015 * index
    timing = 2.0e-9 * np.sin(2.0 * np.pi * index / max(n_igm, 1) + 0.4)
    amplitudes = np.array([1.0, 0.65, 0.38])[: len(rf_tones_hz)]
    interferograms = np.empty((n_igm, n), dtype=np.complex128)
    for k, (phi, delay) in enumerate(zip(phase0, timing)):
        trace = np.zeros(n, dtype=np.complex128)
        for amp, freq in zip(amplitudes, rf_tones_hz):
            trace += amp * np.exp(1j * (2.0 * np.pi * freq * (t + delay)))
        interferograms[k] = trace * np.exp(1j * phi)
        interferograms[k] += 0.01 * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )
    c_m_s = 299792458.0
    nu = c_m_s / np.asarray(config.reference_wavelengths_m)
    nu_center = float(np.mean(nu))
    phase1 = phase0 + 2.0 * np.pi * (nu[0] - nu_center) * timing
    phase2 = phase0 + 2.0 * np.pi * (nu[1] - nu_center) * timing
    acquisition = Acquisition(
        interferograms=interferograms,
        reference_phase_1=phase1,
        reference_phase_2=phase2,
        metadata={"simulated": True, "rf_tones_hz": list(rf_tones_hz)},
    )
    acquisition.validate()
    return acquisition
