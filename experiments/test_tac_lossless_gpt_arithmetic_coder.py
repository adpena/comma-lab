from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessGptArithmeticCoderTests(unittest.TestCase):
    def test_encode_decode_tokens_with_logits_fn_roundtrips_teacher_forced_sample(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import (
            decode_tokens_with_logits_fn,
            encode_tokens_with_logits_fn,
        )

        tokens = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint16)

        def alternating_logits(context):
            logits = np.array([0.0, 0.0], dtype=np.float64)
            logits[1 - int(context[-1])] = 10.0
            return logits

        encoded = encode_tokens_with_logits_fn(
            tokens,
            logits_fn=alternating_logits,
            context_tokens=3,
            vocab_size=2,
        )
        restored = decode_tokens_with_logits_fn(
            encoded["encoded_bytes"],
            token_count=len(tokens),
            first_token=int(tokens[0]),
            logits_fn=alternating_logits,
            context_tokens=3,
            vocab_size=2,
        )

        self.assertEqual(restored, tokens.tolist())
        self.assertLess(encoded["bits_per_token"], 0.05)

    def test_encode_tokens_with_logits_fn_rejects_empty_tokens(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import encode_tokens_with_logits_fn

        with self.assertRaisesRegex(ValueError, "non-empty"):
            encode_tokens_with_logits_fn(
                np.array([], dtype=np.uint16),
                logits_fn=lambda _ctx: np.array([0.0, 0.0], dtype=np.float64),
                context_tokens=3,
                vocab_size=2,
            )


if __name__ == "__main__":
    unittest.main()
