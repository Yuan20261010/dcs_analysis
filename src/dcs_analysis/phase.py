"""CW-reference phase and timing extraction for free-running dual combs."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .types import Acquisition, DCSConfig


def _analytic_beat_note(
    signal: np.ndarray,
    fs_hz: float,
    center_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """Return an analytic, band-limited beat-note signal."""
    if np.iscomplexobj(signal):
        return np.asarray(signal, dtype=np.complex128)
    low = max(1.0, center_hz - bandwidth_hz / 2.0)
    high = min(fs_hz / 2.0 - 1.0, center_hz + bandwidth_hz / 2.0)
    if not 0.0 < low < high < fs_hz / 2.0:
        raise ValueError(f"Invalid beat-note band [{low}, {high}] Hz for fs={fs_hz} Hz")
    sos = butter(4, (low, high), btype="bandpass", fs=fs_hz, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(signal, dtype=float), axis=-1)
    return hilbert(filtered, axis=-1)


def _phase_at_block_centers(z: np.ndarray) -> np.ndarray:
    n_igm, n_samples = z.shape
    centers = n_samples // 2
    phase = np.unwrap(np.angle(z[:, centers]))
    return phase - phase[0]


def _phase_from_pair(z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    # The conjugate product cancels the CW laser phase and retains the
    # relative comb phase at the selected optical reference.
    pair = z_a * np.conj(z_b)
    return _phase_at_block_centers(pair)


def estimate_reference_phases(
    acquisition: Acquisition,
    config: DCSConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the two optical reference phase traces.

    If the acquisition already contains ``reference_phase_1/2``, they are
    returned after unwrapping. Otherwise four synchronized beat-note channels
    are demodulated and paired. The output phases are the two observables
    ``phi_i = phi_0 + 2*pi*nu_i*tau``.
    """
    acquisition.validate()
    if acquisition.reference_phase_1 is not None and acquisition.reference_phase_2 is not None:
        # These channels are explicitly documented as already-demodulated
        # phase traces. Do not run np.unwrap again: a large but valid timing
        # excursion can advance by more than pi between IGMs.
        return (
            acquisition.reference_phase_1 - acquisition.reference_phase_1[0],
            acquisition.reference_phase_2 - acquisition.reference_phase_2[0],
        )
    if acquisition.beat_notes is None:
        raise ValueError("Need two reference phase traces or four beat-note channels")
    centers = tuple(config.beat_note_centers_hz)
    if len(centers) != 4:
        raise ValueError(
            "Raw beat-note processing requires four calibrated beat_note_centers_hz"
        )
    z = np.stack(
        [
            _analytic_beat_note(channel, config.fs_hz, center, config.beat_note_bandwidth_hz)
            for channel, center in zip(acquisition.beat_notes, centers)
        ],
        axis=0,
    )
    return _phase_from_pair(z[0], z[1]), _phase_from_pair(z[2], z[3])


def solve_phase_and_timing(
    reference_phase_1: np.ndarray,
    reference_phase_2: np.ndarray,
    config: DCSConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve common carrier phase and relative timing from two CW references."""
    if config.reference_wavelengths_m is None:
        raise ValueError("reference_wavelengths_m must contain two calibrated wavelengths")
    c_m_s = 299792458.0
    nu = np.array([c_m_s / config.reference_wavelengths_m[0], c_m_s / config.reference_wavelengths_m[1]])
    # Centering the optical frequencies is essential: solving with absolute
    # 200-THz frequencies makes the 2x2 system unnecessarily ill-conditioned.
    nu_center = float(np.mean(nu))
    matrix = np.array(
        [
            [1.0, 2.0 * np.pi * (nu[0] - nu_center)],
            [1.0, 2.0 * np.pi * (nu[1] - nu_center)],
        ]
    )
    observations = np.vstack((reference_phase_1, reference_phase_2))
    solution = np.linalg.solve(matrix, observations)
    phase = solution[0] - solution[0, 0]
    timing = solution[1] - solution[1, 0]
    return phase, timing
