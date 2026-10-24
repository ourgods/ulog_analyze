from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fifo_tools import decode_fifo


def example_packets():
    # Each packet has four physical slots; only samples=3 are valid.
    return {
        "timestamp_sample": np.array([1000000, 1004000], dtype=np.uint64),
        "dt": np.array([1000, 1000], dtype=np.float32),
        "scale": np.array([.01, .02], dtype=np.float32),
        "samples": np.array([3, 2], dtype=np.uint8),
        **{
            f"{axis}[{i}]": np.array([100 + i, 100 + i], dtype=np.int16)
            for axis in "xyz" for i in range(4)
        },
    }


class FifoDecoderTests(unittest.TestCase):
    def test_first_sample_anchor_and_per_packet_scale(self):
        data = decode_fifo(example_packets(), 0, anchor="first")
        self.assertEqual(data["packets"], 2)
        np.testing.assert_allclose(data["time_s"], [1., 1.001, 1.002, 1.004, 1.005])
        np.testing.assert_allclose(data["x"], [1., 1.01, 1.02, 2., 2.02], atol=1e-6)
        self.assertEqual(data["duplicate_sample_times"], 0)

    def test_last_sample_anchor(self):
        data = decode_fifo(example_packets(), 0, anchor="last")
        np.testing.assert_allclose(data["time_s"], [.998, .999, 1., 1.003, 1.004])

    def test_missing_axis_is_reported(self):
        data = example_packets()
        del data["z[3]"]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            decode_fifo(data, 0, anchor="first")

    def test_invalid_packet_count_is_rejected(self):
        data = example_packets()
        data["samples"][0] = 9
        with self.assertRaisesRegex(ValueError, "capacity"):
            decode_fifo(data, 0, anchor="first")

    def test_zero_interval_is_rejected(self):
        data = example_packets()
        data["dt"][0] = 0
        with self.assertRaisesRegex(ValueError, "invalid FIFO"):
            decode_fifo(data, 0, anchor="first")

    def test_anchor_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, "anchor"):
            decode_fifo(example_packets(), 0, anchor="middle")


if __name__ == "__main__":
    unittest.main()
