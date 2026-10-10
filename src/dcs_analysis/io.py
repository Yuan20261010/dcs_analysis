"""File adapters for synchronized DCS recordings.

The preferred interchange format is NPZ because it preserves complex arrays
without imposing a vendor-specific HDF5 schema. HDF5 support is intentionally
conservative: the adapter looks for documented dataset names and otherwise
fails with a clear message instead of guessing channel order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .types import Acquisition, ProcessedSpectrum


def _first_present(mapping: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def load_npz(path: str | Path) -> Acquisition:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        interferograms = _first_present(data, ("interferograms", "igm", "mir"))
        if interferograms is None:
            raise KeyError("NPZ must contain 'interferograms' (or legacy key 'igm'/'mir')")
        phase1 = _first_present(data, ("reference_phase_1", "phase_1"))
        phase2 = _first_present(data, ("reference_phase_2", "phase_2"))
        beat_notes = _first_present(data, ("beat_notes", "reference_beat_notes"))
        metadata: dict[str, Any] = {"source": str(path)}
        for key in ("fs_hz", "frep_hz", "delta_frep_hz"):
            if key in data:
                metadata[key] = float(np.asarray(data[key]).reshape(-1)[0])
    acquisition = Acquisition(interferograms, phase1, phase2, beat_notes, metadata)
    acquisition.validate()
    return acquisition


def _find_hdf5_datasets(group: Any, prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, item in group.items():
        key = f"{prefix}/{name}".strip("/")
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            found[key] = item
        elif hasattr(item, "items"):
            found.update(_find_hdf5_datasets(item, key))
    return found


def load_hdf5(path: str | Path) -> Acquisition:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 input requires 'pip install -e .[io]'") from exc

    path = Path(path)
    with h5py.File(path, "r") as handle:
        datasets = _find_hdf5_datasets(handle)
        def pick(names: tuple[str, ...]) -> Any | None:
            lowered = {key.lower(): value for key, value in datasets.items()}
            for name in names:
                if name in datasets:
                    return datasets[name][()]
                for key, value in lowered.items():
                    if key.endswith(name.lower()):
                        return value[()]
            return None

        interferograms = pick(("interferograms", "igm", "mir"))
        if interferograms is None:
            raise KeyError(
                "Could not find an interferogram dataset; expected a path ending in "
                "interferograms, igm, or mir"
            )
        phase1 = pick(("reference_phase_1", "phase_1"))
        phase2 = pick(("reference_phase_2", "phase_2"))
        beat_notes = pick(("beat_notes", "reference_beat_notes"))
    acquisition = Acquisition(
        np.asarray(interferograms),
        None if phase1 is None else np.asarray(phase1),
        None if phase2 is None else np.asarray(phase2),
        None if beat_notes is None else np.asarray(beat_notes),
        {"source": str(path), "datasets": sorted(datasets)},
    )
    acquisition.validate()
    return acquisition


def load_acquisition(path: str | Path) -> Acquisition:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz(path)
    if suffix in {".h5", ".hdf5"}:
        return load_hdf5(path)
    raise ValueError(f"Unsupported input format: {path.suffix}; use .npz, .h5, or .hdf5")


def save_acquisition(path: str | Path, acquisition: Acquisition) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    acquisition.validate()
    payload: dict[str, Any] = {"interferograms": acquisition.interferograms}
    if acquisition.reference_phase_1 is not None:
        payload["reference_phase_1"] = acquisition.reference_phase_1
    if acquisition.reference_phase_2 is not None:
        payload["reference_phase_2"] = acquisition.reference_phase_2
    if acquisition.beat_notes is not None:
        payload["beat_notes"] = acquisition.beat_notes
    np.savez_compressed(path, **payload)


def save_spectrum_csv(path: str | Path, spectrum: ProcessedSpectrum) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        (
            spectrum.wavenumber_cm,
            spectrum.rf_hz,
            spectrum.complex_spectrum.real,
            spectrum.complex_spectrum.imag,
            spectrum.amplitude,
        )
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="wavenumber_cm,rf_hz,spectrum_real,spectrum_imag,amplitude",
        comments="",
    )
