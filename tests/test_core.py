import unittest

import numpy as np

from dcs_analysis.fit import fit_mixture, model_absorbance
from dcs_analysis.hardware import hardware_report
from dcs_analysis.pipeline import process_acquisition
from dcs_analysis.simulate import simulate_acquisition
from dcs_analysis.spectrum import rf_to_wavenumber
from dcs_analysis.types import DCSConfig, SpectralLine


class DCSCoreTests(unittest.TestCase):
    def setUp(self):
        self.config = DCSConfig(
            fs_hz=2e6,
            frep_hz=250e6,
            delta_frep_hz=1e4,
            block_samples=1024,
            optical_anchor_wavenumber_cm=1000.0,
            reference_wavelengths_m=(1.53e-6, 1.565e-6),
            rf_max_hz=500e3,
            window="rect",
        )

    def test_rf_mapping_uses_repetition_rate_ratio(self):
        mapped = rf_to_wavenumber(np.array([0.0, 1e4]), self.config)
        expected = 1e4 * self.config.frep_hz / self.config.delta_frep_hz / self.config.c_cm_s
        self.assertAlmostEqual(mapped[1] - mapped[0], expected, places=12)

    def test_synthetic_phase_and_timing_correction(self):
        acquisition = simulate_acquisition(
            self.config, n_igm=12, rf_tones_hz=(20e3, 35e3, 58e3), seed=5
        )
        spectrum = process_acquisition(acquisition, self.config)
        tone_wavenumbers = rf_to_wavenumber(np.array([20e3, 35e3, 58e3]), self.config)
        peak_positions = spectrum.wavenumber_cm[np.argsort(spectrum.amplitude)[-12:]]
        for expected in tone_wavenumbers:
            self.assertLess(np.min(np.abs(peak_positions - expected)), 0.01)
        self.assertGreater(spectrum.diagnostics["coherent_quality"], 0.8)

    def test_voigt_mixture_fit_recovers_fractions(self):
        x = np.linspace(999.0, 1006.0, 1800)
        lines = [
            SpectralLine("A", 1001.2, 0.035, gamma_air_cm_atm=0.01, doppler_sigma_cm=0.01),
            SpectralLine("B", 1004.4, 0.025, gamma_air_cm_atm=0.012, doppler_sigma_cm=0.01),
        ]
        expected = {"A": 0.08, "B": 0.12}
        y = model_absorbance(
            x,
            lines,
            expected,
            pressure_atm=0.5,
            temperature_K=296.0,
            path_length_cm=24.0,
            baseline_coefficients=np.array([0.002]),
        )
        rng = np.random.default_rng(7)
        y += rng.normal(0.0, 2e-5, x.size)
        result = fit_mixture(
            x,
            y,
            lines,
            pressure_atm=0.5,
            temperature_K=296.0,
            path_length_cm=24.0,
            baseline_degree=0,
            initial_mole_fractions={"A": 0.05, "B": 0.05},
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.mole_fractions["A"], expected["A"], delta=0.01)
        self.assertAlmostEqual(result.mole_fractions["B"], expected["B"], delta=0.01)

    def test_80mhz_hardware_limits(self):
        config = DCSConfig(
            fs_hz=500e6,
            frep_hz=80e6,
            rf_min_hz=300.0,
            rf_max_hz=39e6,
            optical_wavelength_min_m=7.5e-6,
            optical_wavelength_max_m=11.5e-6,
            adc_bits=12,
            adc_full_scale_peak_v=0.4,
            adc_record_alignment_samples=128,
            detector_rf_high_cutoff_hz=700e6,
        )
        report = hardware_report(config)
        self.assertEqual(report["unambiguous_one_sided_rf_limit_hz"], 40e6)
        self.assertEqual(report["adc_nyquist_hz"], 250e6)
        self.assertAlmostEqual(report["adc_lsb_v"], 0.8 / 4096)
        self.assertAlmostEqual(report["max_delta_frep_for_target_hz"], 230.16, delta=0.1)
        self.assertEqual(report["min_aligned_igm_samples_for_target"], 2172416)

    def test_rejects_aliased_rf_band(self):
        config = DCSConfig(
            fs_hz=1e9,
            frep_hz=80e6,
            rf_max_hz=41e6,
        )
        with self.assertRaisesRegex(ValueError, "frep_hz/2"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
