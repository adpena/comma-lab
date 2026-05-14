# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessRangeCoderTests(unittest.TestCase):
    def test_normalize_probabilities_returns_positive_cumulative_frequencies(self) -> None:
        from tac.lossless.range_coder import normalize_probabilities

        freqs = normalize_probabilities([0.7, 0.2, 0.1], total=1024)

        self.assertEqual(sum(freqs), 1024)
        self.assertTrue(all(item > 0 for item in freqs))

    def test_normalize_probability_rows_matches_scalar_normalization(self) -> None:
        from tac.lossless.range_coder import normalize_probabilities, normalize_probability_rows

        rows = [
            [0.7, 0.2, 0.1],
            [0.05, 0.9, 0.05],
            [3.0, 1.0, 2.0],
        ]

        expected = [normalize_probabilities(row, total=1024) for row in rows]
        actual = normalize_probability_rows(rows, total=1024)

        self.assertEqual(actual.tolist(), expected)

    def test_cumulative_frequency_rows_matches_scalar_cumulative_tables(self) -> None:
        from tac.lossless.range_coder import cumulative_frequencies, cumulative_frequency_rows

        frequencies = [
            [700, 200, 124],
            [60, 900, 64],
        ]

        expected = [cumulative_frequencies(row)[0] for row in frequencies]
        actual = cumulative_frequency_rows(frequencies)

        self.assertEqual(actual.tolist(), expected)

    def test_range_coder_roundtrips_static_sequence(self) -> None:
        from tac.lossless.range_coder import decode_static_symbols, encode_static_symbols

        symbols = [0, 1, 0, 2, 1, 0, 0, 2]
        frequencies = [5, 2, 1]

        encoded = encode_static_symbols(symbols, frequencies=frequencies)
        restored = decode_static_symbols(encoded, count=len(symbols), frequencies=frequencies)

        self.assertEqual(restored, symbols)

    def test_range_coder_roundtrips_binary_sequence(self) -> None:
        from tac.lossless.range_coder import decode_static_symbols, encode_static_symbols

        symbols = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
        frequencies = [3, 7]

        encoded = encode_static_symbols(symbols, frequencies=frequencies)
        restored = decode_static_symbols(encoded, count=len(symbols), frequencies=frequencies)

        self.assertEqual(restored, symbols)

    def test_incremental_range_coder_roundtrips_dynamic_frequencies(self) -> None:
        from tac.lossless.range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies

        symbols = [0, 1, 0, 1, 1, 0]
        frequency_tables = [
            [4, 1],
            [1, 4],
            [4, 1],
            [1, 4],
            [1, 4],
            [4, 1],
        ]

        encoder = RangeEncoder()
        for symbol, frequencies in zip(symbols, frequency_tables):
            cumulative, total = cumulative_frequencies(frequencies)
            encoder.encode(symbol=symbol, cumulative=cumulative, total=total)
        encoded = encoder.finish()

        decoder = RangeDecoder(encoded)
        restored = []
        for frequencies in frequency_tables:
            cumulative, total = cumulative_frequencies(frequencies)
            target = decoder.target(total)
            symbol = max(index for index in range(len(frequencies)) if cumulative[index] <= target)
            decoder.update(low_count=cumulative[symbol], high_count=cumulative[symbol + 1], total=total)
            restored.append(symbol)

        self.assertEqual(restored, symbols)

    def test_range_encoder_rejects_malformed_cumulative_tables_with_value_error(self) -> None:
        from tac.lossless.range_coder import RangeEncoder

        encoder = RangeEncoder()

        with self.assertRaisesRegex(ValueError, "symbol is outside"):
            encoder.encode(symbol=2, cumulative=[0, 3, 5], total=5)
        with self.assertRaisesRegex(ValueError, "total"):
            encoder.encode(symbol=0, cumulative=[0, 1], total=0)
        with self.assertRaisesRegex(ValueError, "start at zero"):
            encoder.encode(symbol=0, cumulative=[1, 2], total=2)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            encoder.encode(symbol=0, cumulative=[0, 1, 1], total=1)

    def test_range_decoder_rejects_invalid_targets_and_intervals_with_value_error(self) -> None:
        from tac.lossless.range_coder import RangeDecoder

        decoder = RangeDecoder(b"\x00")

        with self.assertRaisesRegex(ValueError, "total"):
            decoder.target(0)
        with self.assertRaisesRegex(ValueError, "interval"):
            decoder.update(low_count=2, high_count=2, total=4)
        with self.assertRaisesRegex(ValueError, "interval"):
            decoder.update(low_count=0, high_count=5, total=4)

    def test_decode_static_symbols_rejects_empty_nonempty_range_stream(self) -> None:
        from tac.lossless.range_coder import decode_static_symbols

        with self.assertRaisesRegex(ValueError, "empty"):
            decode_static_symbols(b"", count=1, frequencies=[1, 1])


if __name__ == "__main__":
    unittest.main()
