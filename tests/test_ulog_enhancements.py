from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import uloglib
import ulog_tool
from uloglib import Series, SignalSpec


class AntiAliasingTests(unittest.TestCase):
    def test_downsampling_filters_a_high_frequency_alias(self):
        spec = SignalSpec(label="x", topic="sensor_combined", field="x")
        time = np.arange(0, 5., .005)
        data = Series(spec, time, np.sin(2 * np.pi * 83 * time))
        grid = np.arange(0, 4.95, .05)
        plain = uloglib.resample_series(data, grid)
        safe = uloglib.resample_series(data, grid, anti_alias=True)
        self.assertGreater(np.nanstd(plain), .3)
        self.assertLess(np.nanstd(safe), .1)

    def test_irregular_timestamps_are_rejected(self):
        spec = SignalSpec(label="x", topic="sensor_combined", field="x")
        time = np.arange(0., 5., .005)
        time[50] += .002
        data = Series(spec, time, np.sin(2 * np.pi * 72 * time))
        with self.assertRaisesRegex(ValueError, "irregularly sampled"):
            uloglib.resample_series(data, np.arange(0., 4.95, .05), anti_alias=True)

    def test_discrete_signals_are_not_filtered(self):
        spec = SignalSpec(label="flag", topic="vehicle_status",
                          field="nav_state", interpolation="previous")
        time = np.arange(100) * .01
        data = Series(spec, time, np.arange(100) % 2)
        grid = np.arange(0., .95, .1)
        np.testing.assert_array_equal(
            uloglib.resample_series(data, grid, anti_alias=True),
            uloglib.resample_series(data, grid),
        )

    def test_spectral_commands_enable_anti_alias(self):
        parser = ulog_tool.build_parser()
        parsed = parser.parse_args(
            ["spectrum", "flight.ulg", "-s", "vehicle_angular_velocity.xyz[0]",
             "--low-hz", "0.5", "--high-hz", "20"]
        )
        self.assertTrue(parsed.anti_alias)
        parsed = parser.parse_args(
            ["transfer", "flight.ulg", "--cause", "x.y", "--effect", "z.y",
             "--at-hz", ".2"]
        )
        self.assertTrue(parsed.anti_alias)


class FullMotorCoverageTests(unittest.TestCase):
    def test_every_logged_motor_slot_is_requested(self):
        fields = {"actuator_motors": [f"control[{i}]" for i in range(12)]}
        values = {
            f"actuator_motors.control[{i}]": np.array([.3, .4])
            for i in range(12)
        }
        args = argparse.Namespace(
            log=Path("test.ulg"), max_file_mib=64,
            alias_file=Path("signal_aliases.yaml"),
            verbose=False, rate=10, output=None,
        )
        with (
            patch.object(ulog_tool, "available_schemas", return_value=fields),
            patch.object(ulog_tool, "check_file_size"),
            patch.object(ulog_tool, "load_aliases", return_value={}),
            patch.object(ulog_tool, "analyze",
                         return_value=(np.array([0., 1.]), values, (),
                                       {"sampling": {}, "file": "test.ulg", "window_s": [0, 1]})) as patched,
            patch.object(ulog_tool, "print_or_write") as printed,
        ):
            ulog_tool.command_motors_recipe(args)
            self.assertEqual(len(patched.call_args.args[1]), 12)
            self.assertIn("actuator_motors.control[5]", patched.call_args.args[1])
            self.assertEqual(len(printed.call_args.args[0]["channels"]), 12)

    def test_gps_ekf_recipe_contains_north_and_east(self):
        chosen = ulog_tool.RECIPES["gps-ekf"]
        for field in (
            "vehicle_local_position.vx", "vehicle_local_position.vy",
            "vehicle_gps_position.vel_n_m_s", "vehicle_gps_position.vel_e_m_s",
            "estimator_innovations.gps_hpos[0]",
            "estimator_innovations.gps_hpos[1]",
        ):
            self.assertIn(field, chosen)


if __name__ == "__main__":
    unittest.main()
