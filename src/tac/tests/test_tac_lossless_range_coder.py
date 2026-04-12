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


if __name__ == "__main__":
    unittest.main()
