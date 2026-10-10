"""Command-line interface for the DCS processing reference implementation."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from .fit import fit_mixture, load_line_table
from .hardware import hardware_report
from .io import load_acquisition, save_acquisition, save_spectrum_csv
from .pipeline import process_acquisition
from .simulate import simulate_acquisition
from .types import DCSConfig


def load_config(path: str | Path) -> DCSConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    if "reference_wavelengths_m" in raw:
        raw["reference_wavelengths_m"] = tuple(raw["reference_wavelengths_m"])
    raw["beat_note_centers_hz"] = tuple(raw.get("beat_note_centers_hz", ()))
    config = DCSConfig(**raw)
    config.validate()
    return config


def _save_plot(path: Path, wavenumber: np.ndarray, values: np.ndarray, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.plot(wavenumber, values, linewidth=0.8)
    axis.set_xlabel("Wavenumber (cm$^{-1}$)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcs", description="LWIR dual-comb spectroscopy processing")
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="write a deterministic synthetic acquisition")
    simulate.add_argument("--config", default="configs/reference_lwir_dcs.toml")
    simulate.add_argument("--output", required=True)
    simulate.add_argument("--n-igm", type=int, default=16)

    process = sub.add_parser("process", help="process an NPZ/HDF5 acquisition")
    process.add_argument("input")
    process.add_argument("--config", required=True)
    process.add_argument("--output-dir", required=True)

    fit = sub.add_parser("fit", help="fit a line-table model to an absorbance CSV")
    fit.add_argument("spectrum_csv")
    fit.add_argument("--lines", required=True)
    fit.add_argument("--pressure-atm", type=float, required=True)
    fit.add_argument("--temperature-k", type=float, default=296.0)
    fit.add_argument("--path-length-cm", type=float, required=True)
    fit.add_argument("--output", required=True)

    inspect_config = sub.add_parser(
        "inspect-config", help="show hardware limits and missing calibration values"
    )
    inspect_config.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        config = load_config(args.config)
        acquisition = simulate_acquisition(config, n_igm=args.n_igm)
        save_acquisition(args.output, acquisition)
        print(f"wrote {args.output}")
        return 0

    if args.command == "process":
        config = load_config(args.config)
        acquisition = load_acquisition(args.input)
        spectrum = process_acquisition(acquisition, config)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_spectrum_csv(output_dir / "spectrum.csv", spectrum)
        np.save(output_dir / "coherent_interferogram.npy", spectrum.coherent_interferogram)
        _save_plot(output_dir / "spectrum_amplitude.png", spectrum.wavenumber_cm, spectrum.amplitude, "FFT amplitude")
        (output_dir / "diagnostics.json").write_text(
            json.dumps(spectrum.diagnostics, indent=2), encoding="utf-8"
        )
        print(json.dumps(spectrum.diagnostics, indent=2))
        return 0

    if args.command == "fit":
        table = np.loadtxt(args.spectrum_csv, delimiter=",", skiprows=1)
        lines = load_line_table(args.lines)
        result = fit_mixture(
            table[:, 0],
            table[:, 1],
            lines,
            pressure_atm=args.pressure_atm,
            temperature_K=args.temperature_k,
            path_length_cm=args.path_length_cm,
        )
        payload = {
            "success": result.success,
            "message": result.message,
            "mole_fractions": result.mole_fractions,
            "baseline_coefficients": result.baseline_coefficients.tolist(),
            "residual_rms": result.residual_rms,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "inspect-config":
        config = load_config(args.config)
        print(json.dumps(hardware_report(config), indent=2))
        return 0

    raise RuntimeError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
