from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessTransformsTests(unittest.TestCase):
    def _make_position_major_segment_tokens(self, *, segments: int = 2, frames: int = 4) -> np.ndarray:
        segment_pieces: list[np.ndarray] = []
        for segment_index in range(segments):
            stream = np.full((128, frames), 1024, dtype=np.uint16)
            base_value = (segment_index + 1) * 3
            stream[:, 1] = base_value
            if frames > 2:
                stream[:, 2] = base_value
            if frames > 3:
                stream[:, 3] = base_value + 1
            stream[0, 1:] = np.array([base_value, base_value + 2, base_value + 3][: frames - 1], dtype=np.uint16)
            segment_pieces.append(stream.reshape(-1))
            segment_pieces.append(np.array([1025], dtype=np.uint16))
        return np.concatenate(segment_pieces)

    def test_frequency_remap_roundtrips_uint16_stream(self) -> None:
        from tac.lossless.transforms import apply_frequency_remap, invert_frequency_remap

        tokens = np.array([7, 7, 7, 2, 2, 9, 9, 1], dtype=np.uint16)

        remapped, mapping = apply_frequency_remap(tokens)
        restored = invert_frequency_remap(remapped, mapping)

        self.assertTrue(np.array_equal(restored, tokens))
        self.assertEqual(remapped.tolist()[:4], [0, 0, 0, 1])

    def test_frequency_remap_preserves_identity_for_empty_stream(self) -> None:
        from tac.lossless.transforms import apply_frequency_remap, invert_frequency_remap

        tokens = np.array([], dtype=np.uint16)

        remapped, mapping = apply_frequency_remap(tokens)
        restored = invert_frequency_remap(remapped, mapping)

        self.assertEqual(remapped.tolist(), [])
        self.assertEqual(mapping, {})
        self.assertEqual(restored.tolist(), [])

    def test_frequency_remap_sorts_by_frequency_then_symbol(self) -> None:
        from tac.lossless.transforms import apply_frequency_remap

        tokens = np.array([9, 9, 1, 1, 7], dtype=np.uint16)

        remapped, mapping = apply_frequency_remap(tokens)

        self.assertEqual(mapping, {1: 0, 9: 1, 7: 2})
        self.assertEqual(remapped.tolist(), [1, 1, 0, 0, 2])

    def test_recursive_bisect_frame_order_visits_anchors_then_midpoints(self) -> None:
        from tac.lossless.transforms import recursive_bisect_frame_order

        order = recursive_bisect_frame_order(8)

        self.assertEqual(order.tolist(), [0, 7, 3, 1, 5, 2, 4, 6])

    def test_recursive_bisect_frame_order_roundtrips_frame_major_tokens(self) -> None:
        from tac.lossless.transforms import (
            apply_frame_order,
            invert_frame_order,
            recursive_bisect_frame_order,
        )

        tokens = np.arange(8 * 2 * 2, dtype=np.int16).reshape(8, 2, 2)
        order = recursive_bisect_frame_order(tokens.shape[0])

        reordered = apply_frame_order(tokens, order)
        restored = invert_frame_order(reordered, order)

        self.assertTrue(np.array_equal(restored, tokens))
        self.assertTrue(np.array_equal(reordered[0], tokens[0]))
        self.assertTrue(np.array_equal(reordered[1], tokens[-1]))

    def test_temporal_residual_position_major_roundtrips(self) -> None:
        from tac.lossless.transforms import (
            invert_temporal_residual_position_major,
            temporal_residual_position_major,
        )

        streams = np.full((128, 4), 1024, dtype=np.uint16)
        streams[:, 1] = 10
        streams[:, 2] = 12
        streams[:, 3] = 11
        streams[1, 1:] = [1, 1, 2]
        tokens = np.concatenate([streams.reshape(-1), np.array([1025], dtype=np.uint16)])

        residual = temporal_residual_position_major(tokens)
        restored = invert_temporal_residual_position_major(residual)

        self.assertTrue(np.array_equal(restored, tokens))
        self.assertEqual(residual[:4].tolist(), [1024, 10, 4, 1])
        self.assertEqual(residual[4:8].tolist(), [1024, 1, 0, 2])
        self.assertEqual(int(residual[-1]), 1025)

    def test_temporal_residual_position_major_rejects_missing_terminal_eot(self) -> None:
        from tac.lossless.transforms import temporal_residual_position_major

        with self.assertRaisesRegex(ValueError, "segment EOT"):
            temporal_residual_position_major(np.array([1024, 1, 2], dtype=np.uint16))

    def test_temporal_residual_position_major_supports_multiple_segments(self) -> None:
        from tac.lossless.transforms import (
            invert_temporal_residual_position_major,
            temporal_residual_position_major,
        )

        stream_a = np.full((128, 3), 1024, dtype=np.uint16)
        stream_a[:, 1] = 10
        stream_a[:, 2] = 11
        stream_b = np.full((128, 3), 1024, dtype=np.uint16)
        stream_b[:, 1] = 4
        stream_b[:, 2] = 6
        tokens = np.concatenate(
            [
                stream_a.reshape(-1),
                np.array([1025], dtype=np.uint16),
                stream_b.reshape(-1),
                np.array([1025], dtype=np.uint16),
            ]
        )

        residual = temporal_residual_position_major(tokens)
        restored = invert_temporal_residual_position_major(residual)

        self.assertTrue(np.array_equal(restored, tokens))

    def test_temporal_residual_position_major_skips_eot_sentinel_collisions(self) -> None:
        from tac.lossless.transforms import (
            invert_temporal_residual_position_major,
            temporal_residual_position_major,
        )

        stream = np.full((128, 4), 1024, dtype=np.uint16)
        stream[:, 1] = 513
        stream[:, 2] = 0
        stream[:, 3] = 1
        tokens = np.concatenate([stream.reshape(-1), np.array([1025], dtype=np.uint16)])

        residual = temporal_residual_position_major(tokens)
        restored = invert_temporal_residual_position_major(residual)

        self.assertTrue(np.array_equal(restored, tokens))
        self.assertNotIn(1025, residual[:-1].tolist())

    def test_split_token_bitplanes_roundtrips_uint16_stream(self) -> None:
        from tac.lossless.transforms import invert_split_token_bitplanes, split_token_bitplanes

        tokens = np.array([0, 1, 31, 32, 255, 512, 1023], dtype=np.uint16)

        hi, lo = split_token_bitplanes(tokens, low_bits=5)
        restored = invert_split_token_bitplanes(hi, lo, low_bits=5)

        self.assertTrue(np.array_equal(restored, tokens))
        self.assertEqual(hi.tolist(), [0, 0, 0, 1, 7, 16, 31])
        self.assertEqual(lo.tolist(), [0, 1, 31, 0, 31, 0, 31])

    def test_sustain_attack_position_major_roundtrips(self) -> None:
        from tac.lossless.transforms import invert_sustain_attack_position_major, sustain_attack_position_major

        stream_a = np.full((128, 4), 1024, dtype=np.uint16)
        stream_a[:, 1] = 10
        stream_a[:, 2] = 10
        stream_a[:, 3] = 11
        stream_b = np.full((128, 4), 1024, dtype=np.uint16)
        stream_b[:, 1] = 4
        stream_b[:, 2] = 4
        stream_b[:, 3] = 6
        tokens = np.concatenate(
            [
                stream_a.reshape(-1),
                np.array([1025], dtype=np.uint16),
                stream_b.reshape(-1),
                np.array([1025], dtype=np.uint16),
            ]
        )

        first_values, hold_mask, changed_values = sustain_attack_position_major(tokens)
        restored = invert_sustain_attack_position_major(first_values, hold_mask, changed_values)

        self.assertTrue(np.array_equal(restored, tokens))
        self.assertEqual(first_values.shape[0], 128 * 2 * 2)
        self.assertEqual(hold_mask.dtype, np.uint8)

    def test_sample_position_major_segments_returns_requested_prefix(self) -> None:
        from tac.lossless.transforms import sample_position_major_segments

        tokens = self._make_position_major_segment_tokens(segments=3)

        sample = sample_position_major_segments(tokens, max_segments=2)

        self.assertEqual(int(np.count_nonzero(sample == 1025)), 2)
        self.assertTrue(np.array_equal(sample, tokens[: sample.size]))

    def test_sample_position_major_segments_rejects_non_positive_segment_count(self) -> None:
        from tac.lossless.transforms import sample_position_major_segments

        tokens = self._make_position_major_segment_tokens()

        with self.assertRaisesRegex(ValueError, "positive"):
            sample_position_major_segments(tokens, max_segments=0)

    def test_transform_sample_benchmarks_roundtrip_and_report_proxy_bytes(self) -> None:
        from tac.lossless.transforms import (
            benchmark_bitplane_split_sample,
            benchmark_frequency_remap_sample,
            benchmark_sustain_attack_sample,
            benchmark_temporal_residual_sample,
        )

        tokens = self._make_position_major_segment_tokens(segments=2)
        benchmark_fns = [
            benchmark_frequency_remap_sample,
            benchmark_temporal_residual_sample,
            benchmark_bitplane_split_sample,
            benchmark_sustain_attack_sample,
        ]

        for benchmark_fn in benchmark_fns:
            with self.subTest(benchmark=benchmark_fn.__name__):
                result = benchmark_fn(tokens, max_segments=2)
                self.assertTrue(result["roundtrip_ok"])
                self.assertEqual(result["sample_segments"], 2)
                self.assertEqual(result["sample_tokens"], int(tokens.size))
                self.assertEqual(result["sample_bytes"], int(tokens.nbytes))
                self.assertGreater(result["encoded_bytes"], 0)
                self.assertGreater(result["compression_ratio"], 0.0)

    def test_bitplane_benchmark_reports_expected_metadata(self) -> None:
        from tac.lossless.transforms import benchmark_bitplane_split_sample

        tokens = self._make_position_major_segment_tokens(segments=1)

        result = benchmark_bitplane_split_sample(tokens, max_segments=1, low_bits=5)

        self.assertEqual(result["transform"], "bitplane_split")
        self.assertEqual(result["low_bits"], 5)
        self.assertGreaterEqual(result["high_bit_width"], 1)

    def test_sustain_attack_benchmark_reports_component_bytes(self) -> None:
        from tac.lossless.transforms import benchmark_sustain_attack_sample

        tokens = self._make_position_major_segment_tokens(segments=1)

        result = benchmark_sustain_attack_sample(tokens, max_segments=1)

        self.assertEqual(result["transform"], "sustain_attack")
        self.assertIn("first_values_bytes", result)
        self.assertIn("packed_mask_bytes", result)
        self.assertIn("changed_values_bytes", result)


if __name__ == "__main__":
    unittest.main()
