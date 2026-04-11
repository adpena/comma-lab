from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessGptScoreTests(unittest.TestCase):
    def test_resolve_onnx_execution_providers_prefers_coreml_then_cpu(self) -> None:
        from tac.lossless.gpt_score import resolve_onnx_execution_providers

        providers = resolve_onnx_execution_providers(
            ["AzureExecutionProvider", "CPUExecutionProvider", "CoreMLExecutionProvider"]
        )

        self.assertEqual(providers, ["CoreMLExecutionProvider", "CPUExecutionProvider"])

    def test_load_official_commavq_gpt_onnx_model_prefers_coreml_then_cpu(self) -> None:
        from tac.lossless import gpt_score as module

        fake_session = mock.Mock()
        fake_session.get_inputs.return_value = [SimpleNamespace(name="tokens")]
        fake_session.run.return_value = [np.arange(12, dtype=np.float32).reshape(1, 3, 4)]
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: [
                "AzureExecutionProvider",
                "CPUExecutionProvider",
                "CoreMLExecutionProvider",
            ],
            InferenceSession=mock.Mock(return_value=fake_session),
        )

        with mock.patch.object(module, "_require_onnxruntime", return_value=fake_ort):
            with mock.patch.object(
                module,
                "ensure_official_gpt_onnx_path",
                return_value=Path("/tmp/gpt2m.onnx"),
            ):
                model = module.load_official_commavq_gpt_onnx_model()

        logits = model.next_token_logits(np.array([11, 12, 13], dtype=np.int64))

        fake_ort.InferenceSession.assert_called_once_with(
            "/tmp/gpt2m.onnx",
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertEqual(
            fake_session.run.call_args.args[1]["tokens"].tolist(),
            [[11, 12, 13]],
        )
        np.testing.assert_allclose(logits, np.array([8.0, 9.0, 10.0, 11.0], dtype=np.float32))
        self.assertEqual(model._tac_model_backend, "onnx")
        self.assertEqual(model._tac_execution_provider, "CoreMLExecutionProvider")
        self.assertEqual(model._tac_execution_providers, ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(model._tac_model_artifact_url, module.OFFICIAL_COMMAVQ_GPT_ONNX_URL)

    def test_load_official_commavq_gpt_model_falls_back_to_torch_when_onnx_unavailable(self) -> None:
        from tac.lossless import gpt_score as module

        expected_model = SimpleNamespace(_tac_model_backend="torch")

        with mock.patch.object(
            module,
            "load_official_commavq_gpt_onnx_model",
            side_effect=ImportError("onnxruntime missing"),
        ):
            with mock.patch.object(
                module,
                "load_official_commavq_gpt_torch_model",
                return_value=expected_model,
            ) as mocked_torch:
                model = module.load_official_commavq_gpt_model(device="cpu", dtype="float32")

        mocked_torch.assert_called_once_with(
            device="cpu",
            dtype="float32",
            cache_dir=None,
            model_url=module.OFFICIAL_COMMAVQ_GPT_URL,
            gpt_module_path=None,
        )
        self.assertIs(model, expected_model)
        self.assertIn("onnxruntime missing", model._tac_bridge_fallback_reason)

    def test_score_tokens_with_logits_fn_reports_bits_for_teacher_forced_sample(self) -> None:
        from tac.lossless.gpt_score import score_tokens_with_logits_fn

        tokens = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint16)

        def alternating_logits(context):
            import numpy as np

            last = int(context[-1])
            logits = np.array([0.0, 0.0], dtype=np.float64)
            logits[1 - last] = 10.0
            return logits

        result = score_tokens_with_logits_fn(
            tokens,
            logits_fn=alternating_logits,
            context_tokens=3,
            vocab_size=2,
        )

        self.assertEqual(result["command"], "lossless_gpt_score_sample")
        self.assertEqual(result["scored_tokens"], 5)
        self.assertEqual(result["context_tokens"], 3)
        self.assertLess(result["bits_per_token"], 0.01)
        self.assertLess(result["perplexity"], 1.01)

    def test_score_tokens_with_logits_fn_rejects_non_positive_context(self) -> None:
        from tac.lossless.gpt_score import score_tokens_with_logits_fn

        with self.assertRaisesRegex(ValueError, "context_tokens"):
            score_tokens_with_logits_fn(
                np.array([1, 2], dtype=np.uint16),
                logits_fn=lambda _context: np.array([0.0, 0.0, 0.0]),
                context_tokens=0,
                vocab_size=3,
            )

    def test_score_commavq_gpt_sample_uses_injected_model_loader(self) -> None:
        from tac.lossless.gpt_score import score_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            np.array([0, 1, 0, 1, 0, 1], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                def next_token_logits(self, context):
                    import numpy as np

                    last = int(context[-1])
                    logits = np.array([0.0, 0.0], dtype=np.float64)
                    logits[1 - last] = 10.0
                    return logits

            calls: list[dict[str, object]] = []

            def fake_loader(*, device, dtype, cache_dir, model_url):
                calls.append(
                    {
                        "device": device,
                        "dtype": dtype,
                        "cache_dir": cache_dir,
                        "model_url": model_url,
                    }
                )
                return FakeModel()

            result = score_commavq_gpt_sample(
                token_path,
                max_scored_tokens=5,
                context_tokens=3,
                vocab_size=2,
                device="cpu",
                dtype="float32",
                model_loader=fake_loader,
            )

        self.assertEqual(calls[0]["device"], "cpu")
        self.assertEqual(calls[0]["dtype"], "float32")
        self.assertEqual(result["token_path"], str(token_path))
        self.assertEqual(result["scored_tokens"], 5)
        self.assertLess(result["bits_per_token"], 0.01)

    def test_score_commavq_gpt_sample_reports_runtime_metadata_from_loaded_model(self) -> None:
        from tac.lossless.gpt_score import score_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            np.array([0, 1, 0, 1, 0, 1], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                _tac_model_backend = "onnx"
                _tac_execution_provider = "CPUExecutionProvider"
                _tac_execution_providers = ["CPUExecutionProvider"]
                _tac_model_artifact_url = "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/gpt2m.onnx"

                def score_tokens(self, tokens, *, context_tokens, max_scored_tokens=None):
                    return {
                        "scored_tokens": int(tokens.size - 1 if max_scored_tokens is None else max_scored_tokens),
                        "avg_nll_nats": 0.0,
                    }

            result = score_commavq_gpt_sample(
                token_path,
                max_scored_tokens=5,
                context_tokens=3,
                vocab_size=2,
                device="cpu",
                dtype="float32",
                model_loader=lambda **_: FakeModel(),
            )

        self.assertEqual(result["model_backend"], "onnx")
        self.assertEqual(result["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(result["execution_providers"], ["CPUExecutionProvider"])
        self.assertEqual(
            result["model_url"],
            "https://huggingface.co/commaai/commavq-gpt2m/resolve/main/gpt2m.onnx",
        )

    def test_score_commavq_gpt_sample_strips_segment_eot_and_scores_segments_independently(self) -> None:
        from tac.lossless.arithmetic import FRAME_BOS_TOKEN, SEGMENT_EOT_TOKEN
        from tac.lossless.gpt_score import score_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            np.array(
                [
                    FRAME_BOS_TOKEN,
                    0,
                    1,
                    SEGMENT_EOT_TOKEN,
                    FRAME_BOS_TOKEN,
                    0,
                    1,
                    SEGMENT_EOT_TOKEN,
                ],
                dtype=np.uint16,
            ).tofile(token_path)

            seen_contexts: list[list[int]] = []

            class FakeModel:
                def next_token_logits(self, context):
                    import numpy as np

                    seen_contexts.append([int(item) for item in context.tolist()])
                    logits = np.full(1025, -10.0, dtype=np.float64)
                    expected = {
                        (FRAME_BOS_TOKEN,): 0,
                        (FRAME_BOS_TOKEN, 0): 1,
                        (FRAME_BOS_TOKEN, 1): 0,
                    }
                    next_token = expected[tuple(int(item) for item in context.tolist())]
                    logits[next_token] = 10.0
                    return logits

            result = score_commavq_gpt_sample(
                token_path,
                context_tokens=32,
                max_scored_tokens=4,
                device="cpu",
                dtype="float32",
                model_loader=lambda **_: FakeModel(),
            )

        self.assertEqual(result["scored_tokens"], 4)
        self.assertEqual(seen_contexts, [[1024], [1024, 0], [1024], [1024, 0]])
        self.assertLess(result["bits_per_token"], 0.01)

    def test_score_commavq_gpt_sample_prefers_model_score_tokens_when_available(self) -> None:
        from tac.lossless.arithmetic import FRAME_BOS_TOKEN, SEGMENT_EOT_TOKEN
        from tac.lossless.gpt_score import score_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            np.array(
                [
                    FRAME_BOS_TOKEN,
                    0,
                    1,
                    SEGMENT_EOT_TOKEN,
                    FRAME_BOS_TOKEN,
                    0,
                    1,
                    SEGMENT_EOT_TOKEN,
                ],
                dtype=np.uint16,
            ).tofile(token_path)

            seen_calls: list[dict[str, object]] = []

            class FakeModel:
                def score_tokens(self, tokens, *, context_tokens, max_scored_tokens=None):
                    seen_calls.append(
                        {
                            "tokens": [int(item) for item in tokens.tolist()],
                            "context_tokens": context_tokens,
                            "max_scored_tokens": max_scored_tokens,
                        }
                    )
                    return {
                        "scored_tokens": 2,
                        "avg_nll_nats": 0.0,
                    }

                def next_token_logits(self, context):
                    raise AssertionError("score_commavq_gpt_sample should prefer model.score_tokens when available")

            result = score_commavq_gpt_sample(
                token_path,
                context_tokens=258,
                device="cpu",
                dtype="float32",
                model_loader=lambda **_: FakeModel(),
            )

        self.assertEqual(result["scored_tokens"], 4)
        self.assertEqual(
            seen_calls,
            [
                {"tokens": [FRAME_BOS_TOKEN, 0, 1], "context_tokens": 258, "max_scored_tokens": None},
                {"tokens": [FRAME_BOS_TOKEN, 0, 1], "context_tokens": 258, "max_scored_tokens": None},
            ],
        )
        self.assertLess(result["bits_per_token"], 0.01)

    def test_score_commavq_gpt_sample_defaults_to_official_block_context(self) -> None:
        from tac.lossless.arithmetic import FRAME_BOS_TOKEN, SEGMENT_EOT_TOKEN
        from tac.lossless.gpt_score import score_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            np.array(
                [
                    FRAME_BOS_TOKEN,
                    0,
                    1,
                    SEGMENT_EOT_TOKEN,
                ],
                dtype=np.uint16,
            ).tofile(token_path)

            seen_context_tokens: list[int] = []

            class FakeModel:
                def score_tokens(self, tokens, *, context_tokens, max_scored_tokens=None):
                    seen_context_tokens.append(context_tokens)
                    return {
                        "scored_tokens": 2,
                        "avg_nll_nats": 0.0,
                    }

            result = score_commavq_gpt_sample(
                token_path,
                device="cpu",
                dtype="float32",
                model_loader=lambda **_: FakeModel(),
            )

        self.assertEqual(result["context_tokens"], 2580)
        self.assertEqual(seen_context_tokens, [2580])

    def test_torch_score_tokens_matches_reference_across_chunk_boundary(self) -> None:
        torch = unittest.import_module("torch") if hasattr(unittest, "import_module") else None
        if torch is None:
            import importlib

            if importlib.util.find_spec("torch") is None:
                self.skipTest("torch not installed")
            import torch  # type: ignore[no-redef]

        from tac.lossless.gpt_score import _TorchNextTokenLogitsModel, score_tokens_with_logits_fn

        class FakeTorchModel:
            class config:
                block_size = 4

            def __call__(self, idx):
                batch, seq_len = idx.shape
                logits = torch.full((batch, seq_len, 3), -10.0, dtype=torch.float32, device=idx.device)
                first_tokens = idx[:, 0]
                for pos in range(seq_len):
                    logits[:, pos, :] = -10.0
                    logits[:, pos, first_tokens] = 10.0
                return logits

        tokens = np.array([0, 0, 0, 1, 0], dtype=np.uint16)
        model = _TorchNextTokenLogitsModel(FakeTorchModel(), device="cpu")

        fast = model.score_tokens(tokens, context_tokens=3)

        def logits_fn(context):
            logits = np.full(3, -10.0, dtype=np.float64)
            logits[int(context[0])] = 10.0
            return logits

        reference = score_tokens_with_logits_fn(tokens, logits_fn=logits_fn, context_tokens=3, vocab_size=3)
        self.assertAlmostEqual(fast["avg_nll_nats"], reference["avg_nll_nats"], places=6)

    def test_torch_token_logits_matches_reference_across_chunk_boundary(self) -> None:
        import importlib

        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")
        import torch

        from tac.lossless.gpt_score import _TorchNextTokenLogitsModel

        class FakeTorchModel:
            class config:
                block_size = 4

            def __call__(self, idx):
                batch, seq_len = idx.shape
                logits = torch.full((batch, seq_len, 3), -10.0, dtype=torch.float32, device=idx.device)
                first_tokens = idx[:, 0]
                for pos in range(seq_len):
                    logits[:, pos, :] = -10.0
                    logits[:, pos, first_tokens] = 10.0
                return logits

        tokens = np.array([0, 0, 0, 1, 0], dtype=np.uint16)
        model = _TorchNextTokenLogitsModel(FakeTorchModel(), device="cpu")
        rows = model.token_logits(tokens, context_tokens=3)

        self.assertEqual(rows.shape, (4, 3))
        expected = np.full((4, 3), -10.0, dtype=np.float32)
        expected[:, 0] = 10.0
        np.testing.assert_allclose(rows, expected)

    def test_score_commavq_gpt_sample_rejects_large_segment_with_invalid_frame_major_shape(self) -> None:
        from tac.lossless.arithmetic import FRAME_BOS_TOKEN, SEGMENT_EOT_TOKEN
        from tac.lossless.gpt_score import score_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            bad_segment = np.array([FRAME_BOS_TOKEN] + [0] * 129 + [SEGMENT_EOT_TOKEN], dtype=np.uint16)
            bad_segment.tofile(token_path)

            class FakeModel:
                def score_tokens(self, tokens, *, context_tokens, max_scored_tokens=None):
                    return {"scored_tokens": 1, "avg_nll_nats": 0.0}

            with self.assertRaisesRegex(ValueError, "frame-major"):
                score_commavq_gpt_sample(
                    token_path,
                    device="cpu",
                    dtype="float32",
                    model_loader=lambda **_: FakeModel(),
                )

    def test_score_commavq_gpt_sample_reports_consumed_prefix_for_partial_segment(self) -> None:
        from tac.lossless.gpt_score import score_commavq_gpt_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            np.array([1024, 0, 1, 0, 1, 1025], dtype=np.uint16).tofile(token_path)

            class FakeModel:
                def score_tokens(self, tokens, *, context_tokens, max_scored_tokens=None):
                    self.last_max = max_scored_tokens
                    return {"scored_tokens": 2, "avg_nll_nats": 0.0}

            result = score_commavq_gpt_sample(
                token_path,
                max_scored_tokens=2,
                context_tokens=2580,
                device="cpu",
                dtype="float32",
                model_loader=lambda **_: FakeModel(),
            )

        self.assertEqual(result["segment_token_count"], 3)
        self.assertEqual(result["scored_segment_count"], 1)

    def test_iter_score_chunks_retains_sliding_context_across_boundaries(self) -> None:
        from tac.lossless.gpt_score import _iter_score_chunks

        chunks = list(_iter_score_chunks(token_count=5, score_count=4, context_tokens=3, block_size=4))

        self.assertEqual(
            chunks,
            [
                (0, 4, 0, 3),
                (1, 5, 2, 1),
            ],
        )

    def test_probe_commavq_gpt_devices_reports_fastest_device(self) -> None:
        from tac.lossless.gpt_score import probe_commavq_gpt_devices

        calls: list[dict[str, object]] = []

        def fake_score_fn(token_path, *, output_path=None, profile, max_scored_tokens, context_tokens, device, dtype, cache_dir, model_url, gpt_module_path):
            calls.append(
                {
                    "token_path": str(token_path),
                    "profile": profile,
                    "max_scored_tokens": max_scored_tokens,
                    "context_tokens": context_tokens,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return {
                "command": "lossless_gpt_score_sample",
                "device": device,
                "dtype": dtype,
                "bits_per_token": 9.0 if device == "cpu" else 9.1,
                "scored_tokens": 64,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "tokens.bin"
            np.array([1024, 0, 1, 1025], dtype=np.uint16).tofile(token_path)
            with mock.patch("tac.lossless.gpt_score.time.perf_counter", side_effect=[0.0, 1.0, 1.0, 3.5]):
                result = probe_commavq_gpt_devices(
                    token_path,
                    profile="gpt_arithmetic_small",
                    max_scored_tokens=64,
                    devices=("cpu", "mps"),
                    score_fn=fake_score_fn,
                )

        self.assertEqual(result["command"], "lossless_gpt_score_probe")
        self.assertEqual(result["fastest_device"], "cpu")
        self.assertEqual(result["best_bits_device"], "cpu")
        self.assertEqual(result["context_tokens"], 2580)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(calls[0]["device"], "cpu")
        self.assertEqual(calls[1]["device"], "mps")

    def test_probe_commavq_gpt_devices_supports_explicit_context_override(self) -> None:
        from tac.lossless.gpt_score import probe_commavq_gpt_devices

        seen = []

        def fake_score_fn(token_path, *, output_path=None, profile, max_scored_tokens, context_tokens, device, dtype, cache_dir, model_url, gpt_module_path):
            seen.append((device, context_tokens))
            return {
                "command": "lossless_gpt_score_sample",
                "device": device,
                "dtype": dtype,
                "bits_per_token": 1.0,
                "scored_tokens": 32,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "tokens.bin"
            np.array([1024, 0, 1, 1025], dtype=np.uint16).tofile(token_path)
            with mock.patch("tac.lossless.gpt_score.time.perf_counter", side_effect=[0.0, 1.0]):
                result = probe_commavq_gpt_devices(
                    token_path,
                    profile="gpt_arithmetic_small",
                    max_scored_tokens=32,
                    context_tokens=1290,
                    devices=("cpu",),
                    score_fn=fake_score_fn,
                )

        self.assertEqual(result["context_tokens"], 1290)
        self.assertEqual(seen, [("cpu", 1290)])


if __name__ == "__main__":
    unittest.main()
