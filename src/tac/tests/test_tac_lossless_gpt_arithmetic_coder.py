from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessGptArithmeticCoderTests(unittest.TestCase):
    def test_encode_decode_stream_with_logits_fn_roundtrips_teacher_forced_sample(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import (
            decode_token_stream_with_logits_fn,
            encode_token_stream_with_logits_fn,
        )

        tokens = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint16)

        def alternating_logits(context):
            logits = np.array([0.0, 0.0], dtype=np.float64)
            logits[1 - int(context[-1])] = 10.0
            return logits

        encoded = encode_token_stream_with_logits_fn(
            tokens,
            logits_fn=alternating_logits,
            context_tokens=3,
            vocab_size=2,
        )
        restored = decode_token_stream_with_logits_fn(
            encoded["encoded_bytes"],
            logits_fn=alternating_logits,
            context_tokens=3,
            vocab_size=2,
        )

        self.assertEqual(restored, tokens.tolist())
        self.assertLess(encoded["bits_per_token"], 0.05)
        self.assertTrue(encoded["encoded_bytes"].startswith(b"GTA1"))

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

    def test_encode_commavq_gpt_sample_uses_injected_model_loader(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import encode_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            np.array([1024, 0, 1, 0, 1025], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                def next_token_logits(self, context):
                    logits = np.array([0.0, 0.0], dtype=np.float64)
                    logits[1 - int(context[-1]) if int(context[-1]) < 2 else 0] = 10.0
                    return logits

            calls = []

            def fake_loader(*, device, dtype, cache_dir, model_url, gpt_module_path=None):
                calls.append(
                    {
                        "device": device,
                        "dtype": dtype,
                        "cache_dir": cache_dir,
                        "model_url": model_url,
                        "gpt_module_path": gpt_module_path,
                    }
                )
                return FakeModel()

            result = encode_commavq_gpt_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="gpt_arithmetic_small",
                max_tokens=4,
                context_tokens=3,
                vocab_size=2,
                device="cpu",
                dtype="float32",
                verify_decode=True,
                model_loader=fake_loader,
            )

        self.assertEqual(calls[0]["device"], "cpu")
        self.assertEqual(result["encoded_path"], str(output_path))
        self.assertEqual(result["token_count"], 4)
        self.assertTrue(result["exact_match"])

    def test_encode_commavq_gpt_sample_can_skip_decode_verification(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import encode_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            np.array([1024, 0, 1, 0, 1025], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                def next_token_logits(self, context):
                    logits = np.array([0.0, 0.0], dtype=np.float64)
                    logits[1 - int(context[-1]) if int(context[-1]) < 2 else 0] = 10.0
                    return logits

            result = encode_commavq_gpt_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="gpt_arithmetic_small",
                max_tokens=4,
                context_tokens=3,
                vocab_size=2,
                device="cpu",
                dtype="float32",
                verify_decode=False,
                model_loader=lambda **_: FakeModel(),
            )

        self.assertIsNone(result["exact_match"])

    def test_encode_commavq_gpt_sample_rejects_empty_segments(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import encode_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            np.array([1025], dtype=np.uint16).tofile(token_path)

            with self.assertRaisesRegex(ValueError, "no non-empty frame-major segments"):
                encode_commavq_gpt_sample(
                    token_path=token_path,
                    encoded_path=output_path,
                    profile="gpt_arithmetic_small",
                    max_tokens=4,
                    device="cpu",
                    model_loader=lambda **_: object(),
                )


if __name__ == "__main__":
    unittest.main()
