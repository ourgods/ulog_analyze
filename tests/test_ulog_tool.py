from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import ulog_tool
import uloglib
import signal_tools
from uloglib import SignalSpec


class ParameterDiffTests(unittest.TestCase):
    def test_numeric_tolerance_and_missing_parameters(self) -> None:
        headers = {
            "a.ulg": SimpleNamespace(initial_parameters={"P": 0.2, "ONLY_A": 1, "MODE": "x"}),
            "b.ulg": SimpleNamespace(initial_parameters={"P": 0.20005, "MODE": "x"}),
        }

        def fake_header(path: str | Path) -> SimpleNamespace:
            return headers[Path(path).name]

        with patch.object(uloglib, "read_header", side_effect=fake_header):
            exact = uloglib.parameter_diff(["a.ulg", "b.ulg"])
            tolerant = uloglib.parameter_diff(["a.ulg", "b.ulg"], abs_tol=0.0001)

        self.assertEqual(set(exact), {"ONLY_A", "P"})
        self.assertEqual(set(tolerant), {"ONLY_A"})
        self.assertEqual(tolerant["ONLY_A"], {"a.ulg": 1, "b.ulg": None})

    def test_negative_tolerance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            uloglib.parameter_diff(["a.ulg", "b.ulg"], abs_tol=-1)

    def test_include_equal_returns_matching_parameters(self) -> None:
        headers = {
            "a.ulg": SimpleNamespace(initial_parameters={"SAME": 1, "DIFF": 2}),
            "b.ulg": SimpleNamespace(initial_parameters={"SAME": 1, "DIFF": 3}),
        }

        def fake_header(path: str | Path) -> SimpleNamespace:
            return headers[Path(path).name]

        with patch.object(uloglib, "read_header", side_effect=fake_header):
            differences = uloglib.parameter_diff(["a.ulg", "b.ulg"])
            all_parameters = uloglib.parameter_diff(["a.ulg", "b.ulg"], include_equal=True)

        self.assertEqual(set(differences), {"DIFF"})
        self.assertEqual(set(all_parameters), {"DIFF", "SAME"})

    def test_duplicate_default_cli_labels_are_rejected(self) -> None:
        args = argparse.Namespace(
            logs=[Path("first/log.ulg"), Path("second/log.ulg")],
            labels=None,
        )
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            ulog_tool.command_params_diff(args)


class InstanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log = SimpleNamespace(data_list=[
            SimpleNamespace(name="vehicle_imu_status", multi_id=0),
            SimpleNamespace(name="vehicle_imu_status", multi_id=1),
        ])

    def test_implicit_multi_instance_is_rejected(self) -> None:
        spec = SignalSpec(
            "vehicle_imu_status.gyro_vibration_metric",
            "vehicle_imu_status",
            "gyro_vibration_metric",
        )
        with self.assertRaisesRegex(ValueError, r"instances \[0, 1\]"):
            ulog_tool.ensure_unambiguous_instances(self.log, [spec])

    def test_explicit_multi_instance_is_allowed(self) -> None:
        spec = SignalSpec(
            "vehicle_imu_status@1.gyro_vibration_metric",
            "vehicle_imu_status",
            "gyro_vibration_metric",
            instance=1,
        )
        ulog_tool.ensure_unambiguous_instances(self.log, [spec])

    def test_primary_instance_resolves_from_sensor_status(self) -> None:
        log = SimpleNamespace(data_list=[
            SimpleNamespace(
                name="sensors_status_imu",
                multi_id=0,
                data={
                    "accel_device_id_primary": np.array([1114122], dtype=np.uint32),
                    "gyro_device_id_primary": np.array([2228234], dtype=np.uint32),
                },
            ),
            SimpleNamespace(
                name="vehicle_imu_status",
                multi_id=0,
                data={
                    "accel_device_id": np.array([2162698], dtype=np.uint32),
                    "gyro_device_id": np.array([2162698], dtype=np.uint32),
                },
            ),
            SimpleNamespace(
                name="vehicle_imu_status",
                multi_id=1,
                data={
                    "accel_device_id": np.array([1114122], dtype=np.uint32),
                    "gyro_device_id": np.array([2228234], dtype=np.uint32),
                },
            ),
        ])
        spec = SignalSpec(
            "vehicle_imu_status@primary.gyro_vibration_metric",
            "vehicle_imu_status",
            "gyro_vibration_metric",
            instance=ulog_tool.PRIMARY_INSTANCE,
        )
        resolved, mapping = ulog_tool.resolve_primary_specs(log, [spec])
        self.assertEqual(resolved[0].instance, 1)
        self.assertEqual(mapping[spec.label], 1)

    def test_primary_reference_is_parsed(self) -> None:
        specs, _expressions = ulog_tool.source_specs(
            [],
            ["vibration=vehicle_imu_status@primary.gyro_vibration_metric"],
            {},
            "linear",
        )
        self.assertEqual(specs[0].instance, ulog_tool.PRIMARY_INSTANCE)

    def test_primary_change_in_window_is_rejected(self) -> None:
        log = SimpleNamespace(
            start_timestamp=0,
            data_list=[
                SimpleNamespace(
                    name="sensors_status_imu",
                    multi_id=0,
                    data={
                        "timestamp": np.array([0, 10_000_000, 20_000_000], dtype=np.uint64),
                        "accel_device_id_primary": np.array([111, 111, 222], dtype=np.uint32),
                        "gyro_device_id_primary": np.array([333, 333, 444], dtype=np.uint32),
                    },
                ),
                SimpleNamespace(
                    name="vehicle_imu_status",
                    multi_id=0,
                    data={"accel_device_id": np.array([111]), "gyro_device_id": np.array([333])},
                ),
                SimpleNamespace(
                    name="vehicle_imu_status",
                    multi_id=1,
                    data={"accel_device_id": np.array([222]), "gyro_device_id": np.array([444])},
                ),
            ],
        )
        spec = SignalSpec(
            "vehicle_imu_status@primary.gyro_vibration_metric",
            "vehicle_imu_status",
            "gyro_vibration_metric",
            instance=ulog_tool.PRIMARY_INSTANCE,
        )
        with self.assertRaisesRegex(ValueError, "primary IMU changed"):
            ulog_tool.resolve_primary_specs(log, [spec], start_s=5, end_s=25)

    def test_expression_reference_is_checked_for_ambiguity(self) -> None:
        specs, _expressions = ulog_tool.source_specs(
            [],
            ["vibration=vehicle_imu_status.gyro_vibration_metric"],
            {},
            "linear",
        )
        with self.assertRaisesRegex(ValueError, r"instances \[0, 1\]"):
            ulog_tool.ensure_unambiguous_instances(self.log, specs)

    def test_identifier_summary_uses_scalar_for_constant_id(self) -> None:
        summary = ulog_tool.identifier_summary({
            "accel_device_id": np.array([1114122, 1114122], dtype=np.uint32),
            "gyro_device_id": np.array([2228234, 2228234], dtype=np.uint32),
        })
        self.assertEqual(summary, {"accel_device_id": 1114122, "gyro_device_id": 2228234})


class SpectrumValidationTests(unittest.TestCase):
    def test_upsampling_is_rejected(self) -> None:
        metadata = {"sampling": {"sensor_gyro@1.x": {"median_hz": 1.0}}}
        args = argparse.Namespace(rate=10.0, high_hz=0.4)
        with self.assertRaisesRegex(ValueError, "would upsample"):
            ulog_tool.validate_spectrum_sampling(Path("test.ulg"), metadata, args)

    def test_source_nyquist_is_enforced(self) -> None:
        metadata = {"sampling": {"sensor_gyro@1.x": {"median_hz": 50.0}}}
        args = argparse.Namespace(rate=50.0, high_hz=30.0)
        with self.assertRaisesRegex(ValueError, "source Nyquist"):
            ulog_tool.validate_spectrum_sampling(Path("test.ulg"), metadata, args)

    def test_native_rate_spectrum_is_allowed(self) -> None:
        metadata = {"sampling": {"vehicle_angular_velocity.xyz[0]": {"median_hz": 50.0}}}
        args = argparse.Namespace(rate=50.0, high_hz=20.0)
        ulog_tool.validate_spectrum_sampling(Path("test.ulg"), metadata, args)


class TransferValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "sampling": {
                "cause": {"median_hz": 1.0},
                "effect": {"median_hz": 10.0},
            }
        }

    def test_upsampling_is_rejected_without_explicit_opt_in(self) -> None:
        args = argparse.Namespace(rate=5.0, at_hz=[0.2], allow_upsampling=False)
        with self.assertRaisesRegex(ValueError, "would upsample"):
            ulog_tool.transfer_sampling_safety(Path("test.ulg"), self.metadata, args)

    def test_raw_source_nyquist_is_enforced_after_opt_in(self) -> None:
        args = argparse.Namespace(rate=5.0, at_hz=[0.6], allow_upsampling=True)
        with self.assertRaisesRegex(ValueError, "raw-source Nyquist"):
            ulog_tool.transfer_sampling_safety(Path("test.ulg"), self.metadata, args)

    def test_opted_in_upsampling_is_reported(self) -> None:
        args = argparse.Namespace(rate=5.0, at_hz=[0.2], allow_upsampling=True)
        result = ulog_tool.transfer_sampling_safety(Path("test.ulg"), self.metadata, args)
        self.assertTrue(result["upsampling"])
        self.assertEqual(result["safe_frequency_range_hz"], [0.0, 0.5])

    def test_any_upsampling_is_rejected_without_opt_in(self) -> None:
        args = argparse.Namespace(rate=1.01, at_hz=[0.2], allow_upsampling=False)
        with self.assertRaisesRegex(ValueError, "would upsample"):
            ulog_tool.transfer_sampling_safety(Path("test.ulg"), self.metadata, args)

    def test_transfer_parser_requires_a_frequency(self) -> None:
        with self.assertRaises(SystemExit):
            ulog_tool.build_parser().parse_args([
                "transfer", "test.ulg", "--cause", "cause.x", "--effect", "effect.x",
            ])


class ResamplingSafetyTests(unittest.TestCase):
    def test_linear_resampling_does_not_bridge_a_large_gap(self) -> None:
        series = uloglib.Series(
            SignalSpec("x", "topic", "x"),
            np.array([0.0, 1.0, 10.0]),
            np.array([0.0, 1.0, 10.0]),
        )
        values = uloglib.resample_series(series, np.array([0.0, 0.5, 1.0, 5.0, 10.0]))
        np.testing.assert_allclose(values[:3], [0.0, 0.5, 1.0])
        self.assertTrue(np.isnan(values[3]))
        self.assertEqual(values[4], 10.0)

    def test_transfer_rejects_non_positive_segment_requirement(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_segments must be positive"):
            signal_tools.transfer_function(np.arange(32.0), np.arange(32.0), 10.0, min_segments=0)


class MissingSignalTests(unittest.TestCase):
    def _available_log(self) -> SimpleNamespace:
        return SimpleNamespace(
            start_timestamp=0,
            data_list=[
                SimpleNamespace(
                    name="available",
                    multi_id=0,
                    data={
                        "timestamp": np.array([0, 1_000_000, 2_000_000], dtype=np.uint64),
                        "x": np.array([1.0, 2.0, 3.0]),
                    },
                ),
            ],
        )

    def _args(self, *, start: float | None = None, end: float | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            max_file_mib=1.0,
            interpolation="linear",
            airborne=False,
            armed=False,
            mission=False,
            start=start,
            end=end,
            rate=1.0,
        )

    def test_analyze_reports_an_expression_with_a_missing_dependency(self) -> None:
        log = self._available_log()
        args = self._args()
        with patch.object(ulog_tool, "check_file_size"), patch.object(ulog_tool, "load_topics", return_value=log):
            _time, values, missing, _metadata = ulog_tool.analyze(
                Path("missing.ulg"),
                [],
                ["difference=available.x-absent.x"],
                {},
                args,
            )
        self.assertEqual(values, {})
        self.assertIn("absent.x", missing)
        self.assertIn("difference", missing)

    def test_analyze_preserves_an_open_ended_requested_window(self) -> None:
        log = self._available_log()
        with patch.object(ulog_tool, "check_file_size"), patch.object(ulog_tool, "load_topics", return_value=log):
            _time, _values, _missing, metadata = ulog_tool.analyze(
                Path("open.ulg"),
                ["available.x"],
                [],
                {},
                self._args(start=1.0),
            )
        self.assertEqual(metadata["requested_window_s"], [1.0, None])


class WindowAndLagSafetyTests(unittest.TestCase):
    def test_airborne_window_progress_is_sent_to_stderr(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(ulog_tool, "build_record", side_effect=lambda *_args: (print("progress"), {"flight_window_s": [1, 2]})[1]), redirect_stdout(stdout), redirect_stderr(stderr):
            window = ulog_tool.signal_window(Path("test.ulg"), "airborne")
        self.assertEqual(window, (1.0, 2.0))
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "progress\n")

    def test_compare_signals_uses_only_the_longest_continuous_run(self) -> None:
        args = argparse.Namespace(
            log=Path("test.ulg"),
            cause="cause.x",
            effect="effect.x",
            rate=1.0,
            max_lag=1.0,
            alias_file=Path("aliases.yaml"),
            verbose=False,
            all_lags=False,
            output=None,
        )
        time_s = np.arange(10.0)
        values = {
            "cause.x": np.array([0, 1, 2, np.nan, np.nan, 0, 1, 2, 3, 4], dtype=float),
            "effect.x": np.array([0, 1, 2, np.nan, np.nan, 0, 1, 2, 3, 4], dtype=float),
        }
        metadata = {"file": "test.ulg"}
        with patch.object(ulog_tool, "load_aliases", return_value={}), patch.object(ulog_tool, "analyze", return_value=(time_s, values, (), metadata)), patch.object(ulog_tool, "print_or_write") as emit:
            ulog_tool.command_compare_signals(args)
        payload = emit.call_args.args[0]
        self.assertEqual(payload["effective_window_s"], [5.0, 9.0])
        self.assertEqual(payload["samples_used"], 5)
        self.assertEqual(payload["paired_samples_available"], 8)
        self.assertEqual(payload["paired_samples_omitted"], 3)
        self.assertIsNotNone(payload["gap_warning"])


if __name__ == "__main__":
    unittest.main()
