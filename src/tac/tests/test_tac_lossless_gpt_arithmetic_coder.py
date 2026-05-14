# SPDX-License-Identifier: MIT
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

    def test_encode_commavq_gpt_sample_writes_json_sidecar(self) -> None:
        import json

        from tac.lossless.gpt_arithmetic_coder import encode_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            sidecar_path = root / "sample.json"
            np.array([1024, 0, 1, 0, 1025], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                def token_logits(self, tokens, *, context_tokens):
                    logits = np.full((len(tokens) - 1, 2), -10.0, dtype=np.float64)
                    for row, target in enumerate([0, 1, 0]):
                        logits[row, target] = 10.0
                    return logits

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

            self.assertTrue(sidecar_path.exists())
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        self.assertEqual(sidecar["encoded_path"], str(output_path))
        self.assertEqual(sidecar["token_count"], 4)
        self.assertEqual(sidecar["encoded_bytes"], result["encoded_bytes"])

    def test_encode_commavq_gpt_sample_reports_runtime_metadata_from_loaded_model(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import encode_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            np.array([1024, 0, 1, 0, 1025], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                _tac_model_backend = "onnx"
                _tac_execution_provider = "CPUExecutionProvider"
                _tac_execution_providers = ["CPUExecutionProvider"]
                _tac_model_artifact_url = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/gpt2m.onnx"

                def token_logits(self, tokens, *, context_tokens):
                    logits = np.full((len(tokens) - 1, 2), -10.0, dtype=np.float64)
                    for row, target in enumerate([0, 1, 0]):
                        logits[row, target] = 10.0
                    return logits

                def next_token_logits(self, context):
                    raise AssertionError("verify_decode=False should avoid next_token_logits")

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

        self.assertEqual(result["model_backend"], "onnx")
        self.assertEqual(result["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(result["execution_providers"], ["CPUExecutionProvider"])
        self.assertEqual(
            result["model_url"],
            "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/gpt2m.onnx",
        )

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

    def test_encode_commavq_gpt_sample_prefers_batched_model_logits_when_available(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import encode_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            np.array([1024, 0, 1, 0, 1025], dtype=np.uint16).tofile(token_path)

            seen_calls = []

            class FakeModel:
                def token_logits(self, tokens, *, context_tokens):
                    seen_calls.append(
                        {
                            "tokens": [int(item) for item in tokens.tolist()],
                            "context_tokens": context_tokens,
                        }
                    )
                    logits = np.full((len(tokens) - 1, 2), -10.0, dtype=np.float64)
                    for row, target in enumerate([0, 1, 0]):
                        logits[row, target] = 10.0
                    return logits

                def next_token_logits(self, context):
                    raise AssertionError("encode_commavq_gpt_sample should prefer model.token_logits when available")

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

            self.assertTrue(output_path.exists())

        self.assertEqual(seen_calls, [{"tokens": [1024, 0, 1, 0], "context_tokens": 3}])
        self.assertEqual(result["encoded_path"], str(output_path))

    def test_encode_commavq_gpt_global_sample_prefers_batched_model_logits_when_available(self) -> None:
        import tac.lossless.gpt_arithmetic_coder as module

        encode_commavq_gpt_global_sample = getattr(module, "encode_commavq_gpt_global_sample", None)
        if encode_commavq_gpt_global_sample is None:
            self.fail("encode_commavq_gpt_global_sample is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            np.array([3, 4, 5, 6, 7], dtype=np.uint16).tofile(token_path)

            token_logits_calls = []
            next_token_logits_calls = []

            class FakeModel:
                def token_logits(self, tokens, *, context_tokens):
                    token_logits_calls.append(
                        {
                            "tokens": [int(item) for item in tokens.tolist()],
                            "context_tokens": context_tokens,
                        }
                    )
                    logits = np.full((len(tokens) - 1, 16), -10.0, dtype=np.float64)
                    for row, target in enumerate([4, 5, 6, 7]):
                        logits[row, target] = 10.0
                    return logits

                def next_token_logits(self, context):
                    next_token_logits_calls.append([int(item) for item in context.tolist()])
                    logits = np.full((16,), -10.0, dtype=np.float64)
                    logits[(int(context[-1]) + 1) % 16] = 10.0
                    return logits

            result = encode_commavq_gpt_global_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="gpt_arithmetic_small",
                max_tokens=5,
                context_tokens=3,
                vocab_size=16,
                device="cpu",
                dtype="float32",
                verify_decode=True,
                model_loader=lambda **_: FakeModel(),
            )

            self.assertTrue(output_path.exists())

        self.assertEqual(token_logits_calls, [{"tokens": [3, 4, 5, 6, 7], "context_tokens": 3}])
        self.assertEqual(next_token_logits_calls, [[3], [3, 4], [3, 4, 5], [4, 5, 6]])
        self.assertTrue(result["exact_match"])

    def test_encode_commavq_gpt_global_sample_writes_json_sidecar(self) -> None:
        import json

        import tac.lossless.gpt_arithmetic_coder as module

        encode_commavq_gpt_global_sample = getattr(module, "encode_commavq_gpt_global_sample", None)
        if encode_commavq_gpt_global_sample is None:
            self.fail("encode_commavq_gpt_global_sample is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            sidecar_path = root / "sample.json"
            np.array([3, 4, 5, 6, 7], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                def token_logits(self, tokens, *, context_tokens):
                    logits = np.full((len(tokens) - 1, 16), -10.0, dtype=np.float64)
                    for row, target in enumerate([4, 5, 6, 7]):
                        logits[row, target] = 10.0
                    return logits

                def next_token_logits(self, context):
                    logits = np.full((16,), -10.0, dtype=np.float64)
                    logits[(int(context[-1]) + 1) % 16] = 10.0
                    return logits

            result = encode_commavq_gpt_global_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="gpt_arithmetic_small",
                max_tokens=5,
                context_tokens=3,
                vocab_size=16,
                device="cpu",
                dtype="float32",
                verify_decode=False,
                model_loader=lambda **_: FakeModel(),
            )

            self.assertTrue(sidecar_path.exists())
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        self.assertEqual(sidecar["encoded_path"], str(output_path))
        self.assertEqual(sidecar["token_count"], 5)
        self.assertEqual(sidecar["encoded_bytes"], result["encoded_bytes"])

    def test_encode_commavq_gpt_global_sample_roundtrips_bounded_raw_stream_prefix(self) -> None:
        import tac.lossless.gpt_arithmetic_coder as module

        encode_commavq_gpt_global_sample = getattr(module, "encode_commavq_gpt_global_sample", None)
        if encode_commavq_gpt_global_sample is None:
            self.fail("encode_commavq_gpt_global_sample is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            np.array([3, 4, 5, 6, 7, 8, 9], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                def next_token_logits(self, context):
                    logits = np.full((16,), -10.0, dtype=np.float64)
                    logits[(int(context[-1]) + 1) % 16] = 10.0
                    return logits

            result = encode_commavq_gpt_global_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="gpt_arithmetic_small",
                max_tokens=5,
                context_tokens=3,
                vocab_size=16,
                device="cpu",
                dtype="float32",
                verify_decode=True,
                model_loader=lambda **_: FakeModel(),
            )

            self.assertTrue(output_path.exists())

        self.assertEqual(result["command"], "lossless_gpt_arithmetic_global_sample")
        self.assertEqual(result["token_count"], 5)
        self.assertEqual(result["raw_token_count"], 7)
        self.assertTrue(result["exact_match"])

    def test_encode_commavq_gpt_global_sample_rejects_prefix_shorter_than_two_tokens(self) -> None:
        import tac.lossless.gpt_arithmetic_coder as module

        encode_commavq_gpt_global_sample = getattr(module, "encode_commavq_gpt_global_sample", None)
        if encode_commavq_gpt_global_sample is None:
            self.fail("encode_commavq_gpt_global_sample is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.gta"
            np.array([9], dtype=np.uint16).tofile(token_path)

            with self.assertRaisesRegex(ValueError, "at least two tokens"):
                encode_commavq_gpt_global_sample(
                    token_path=token_path,
                    encoded_path=output_path,
                    profile="gpt_arithmetic_small",
                    max_tokens=1,
                    device="cpu",
                    model_loader=lambda **_: object(),
                )

    def test_probe_commavq_gpt_arithmetic_devices_reports_fastest_backend(self) -> None:
        from tac.lossless.gpt_arithmetic_coder import probe_commavq_gpt_arithmetic_devices

        calls = []

        def fake_encode(*, token_path, encoded_path, profile, max_tokens, context_tokens, device, dtype, verify_decode, cache_dir, model_url, gpt_module_path):
            calls.append((device, max_tokens))
            return {
                "command": "lossless_gpt_arithmetic_sample",
                "encoded_path": str(encoded_path),
                "device": device,
                "token_count": max_tokens,
                "encoded_bytes": 100 if device == "mps" else 120,
                "compression_ratio": 5.0 if device == "mps" else 4.0,
                "bits_per_token": 2.0 if device == "mps" else 2.2,
                "exact_match": None,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "probe.json"
            np.array([1024, 0, 1, 0, 1025], dtype=np.uint16).tofile(token_path)
            with mock.patch("tac.lossless.gpt_arithmetic_coder.time.perf_counter", side_effect=[0.0, 2.0, 2.0, 3.0]):
                result = probe_commavq_gpt_arithmetic_devices(
                    token_path=token_path,
                    output_path=output_path,
                    profile="gpt_arithmetic_small",
                    max_tokens=256,
                    devices=("cpu", "mps"),
                    encode_fn=fake_encode,
                )

        self.assertEqual(result["fastest_device"], "mps")
        self.assertEqual(result["best_ratio_device"], "mps")
        self.assertEqual(calls, [("cpu", 256), ("mps", 256)])

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
