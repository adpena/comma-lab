# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessNextFrameCoderTests(unittest.TestCase):
    def test_build_position_transition_model_predicts_next_frame_logits(self) -> None:
        from tac.lossless.next_frame_coder import build_position_transition_model

        frames = np.array(
            [
                np.full((8, 16), 3, dtype=np.uint16),
                np.full((8, 16), 4, dtype=np.uint16),
                np.full((8, 16), 5, dtype=np.uint16),
                np.full((8, 16), 6, dtype=np.uint16),
            ]
        )

        model = build_position_transition_model(frames, vocab_size=16)
        logits = model.next_frame_logits(frames[:1], context_frames=1)

        self.assertEqual(logits.shape, (128, 16))
        self.assertEqual(int(np.argmax(logits[0])), 4)

    def test_build_position_pair_transition_model_predicts_next_frame_logits(self) -> None:
        from tac.lossless.next_frame_coder import build_position_pair_transition_model

        frames = np.array(
            [
                np.full((8, 16), 3, dtype=np.uint16),
                np.full((8, 16), 4, dtype=np.uint16),
                np.full((8, 16), 6, dtype=np.uint16),
                np.full((8, 16), 7, dtype=np.uint16),
            ]
        )

        model = build_position_pair_transition_model(frames, vocab_size=16)
        logits = model.next_frame_logits(frames[:2], context_frames=2)

        self.assertEqual(logits.shape, (128, 16))
        self.assertEqual(int(np.argmax(logits[0])), 6)

    def test_build_position_pair_transition_model_uses_single_frame_backoff(self) -> None:
        from tac.lossless.next_frame_coder import build_position_pair_transition_model

        frames = np.array(
            [
                np.full((8, 16), 2, dtype=np.uint16),
                np.full((8, 16), 5, dtype=np.uint16),
                np.full((8, 16), 2, dtype=np.uint16),
            ]
        )

        model = build_position_pair_transition_model(frames, vocab_size=8)
        logits = model.next_frame_logits(frames[:1], context_frames=2)

        self.assertEqual(logits.shape, (128, 8))
        self.assertEqual(int(np.argmax(logits[0])), 5)

    def test_build_position_pair_transition_model_uses_global_backoff_for_sparse_position_context(self) -> None:
        from tac.lossless.next_frame_coder import build_position_pair_transition_model

        prev2 = np.full((8, 16), 1, dtype=np.uint16)
        prev1 = np.full((8, 16), 2, dtype=np.uint16)
        current = np.full((8, 16), 9, dtype=np.uint16)
        current[0, 0] = 3
        frames = np.array([prev2, prev1, current])

        model = build_position_pair_transition_model(frames, vocab_size=16)
        logits = model.next_frame_logits(frames[:2], context_frames=2)

        self.assertEqual(logits.shape, (128, 16))
        self.assertEqual(int(np.argmax(logits[0])), 9)
        self.assertEqual(int(np.argmax(logits[1])), 9)

    def test_encode_decode_next_frame_stream_roundtrips_sample(self) -> None:
        from tac.lossless.next_frame_coder import (
            decode_next_frame_stream_with_logits_fn,
            encode_next_frame_stream_with_logits_fn,
        )

        frames = np.array(
            [
                np.full((8, 16), 3, dtype=np.uint16),
                np.full((8, 16), 4, dtype=np.uint16),
                np.full((8, 16), 5, dtype=np.uint16),
            ]
        )

        def next_frame_logits(prefix_frames):
            logits = np.full((128, 8), -10.0, dtype=np.float64)
            next_value = int(prefix_frames[-1].reshape(-1)[0]) + 1
            logits[:, next_value] = 10.0
            return logits

        encoded = encode_next_frame_stream_with_logits_fn(
            frames,
            logits_fn=next_frame_logits,
            vocab_size=8,
        )
        restored = decode_next_frame_stream_with_logits_fn(
            encoded["encoded_bytes"],
            logits_fn=next_frame_logits,
            vocab_size=8,
        )

        self.assertTrue(np.array_equal(restored, frames))
        self.assertTrue(encoded["encoded_bytes"].startswith(b"NFG1"))

    def test_encode_commavq_next_frame_sample_uses_injected_model_loader(self) -> None:
        from tac.lossless.next_frame_coder import encode_commavq_next_frame_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.nfg"
            frame0 = np.full((8, 16), 3, dtype=np.uint16)
            frame1 = np.full((8, 16), 4, dtype=np.uint16)
            frame2 = np.full((8, 16), 5, dtype=np.uint16)
            flat = np.concatenate(
                [
                    np.array([1024], dtype=np.uint16), frame0.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame1.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame2.reshape(-1),
                    np.array([1025], dtype=np.uint16),
                ]
            )
            flat.tofile(token_path)

            calls = []
            loader_kwargs = {}

            class FakeModel:
                def next_frame_logits(self, prefix_frames, *, context_frames):
                    calls.append((prefix_frames.shape[0], context_frames))
                    logits = np.full((128, 8), -10.0, dtype=np.float64)
                    logits[:, int(prefix_frames[-1].reshape(-1)[0]) + 1] = 10.0
                    return logits

            result = encode_commavq_next_frame_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="gpt_next_frame_small",
                max_frames=3,
                context_frames=1,
                vocab_size=8,
                verify_decode=True,
                device="cpu",
                dtype="float32",
                cache_dir=root / "cache",
                model_url="https://example.invalid/model.bin",
                gpt_module_path=root / "gpt.py",
                model_loader=lambda **kwargs: loader_kwargs.update(kwargs) or FakeModel(),
            )

        self.assertEqual(calls, [(1, 1), (1, 1), (1, 1), (1, 1)])
        self.assertEqual(result["frame_count"], 3)
        self.assertTrue(result["exact_match"])
        self.assertTrue(result["local_only"])
        self.assertFalse(result["measured"])
        self.assertEqual(result["dtype"], "float32")
        self.assertEqual(loader_kwargs["device"], "cpu")
        self.assertEqual(loader_kwargs["dtype"], "float32")
        self.assertEqual(loader_kwargs["model_url"], "https://example.invalid/model.bin")
        self.assertEqual(loader_kwargs["cache_dir"], root / "cache")
        self.assertEqual(loader_kwargs["gpt_module_path"], root / "gpt.py")

    def test_encode_commavq_next_frame_sample_rejects_nonpositive_context_frames(self) -> None:
        from tac.lossless.next_frame_coder import encode_commavq_next_frame_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.nfg"
            frame0 = np.full((8, 16), 3, dtype=np.uint16)
            frame1 = np.full((8, 16), 4, dtype=np.uint16)
            flat = np.concatenate(
                [
                    np.array([1024], dtype=np.uint16), frame0.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame1.reshape(-1),
                    np.array([1025], dtype=np.uint16),
                ]
            )
            flat.tofile(token_path)

            with self.assertRaisesRegex(ValueError, "context_frames must be positive"):
                encode_commavq_next_frame_sample(
                    token_path=token_path,
                    encoded_path=output_path,
                    context_frames=0,
                    model_loader=lambda **_: object(),
                )

    def test_encode_commavq_next_frame_sample_accepts_tiny_frame_predictor_profile(self) -> None:
        from tac.lossless.next_frame_coder import encode_commavq_next_frame_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.nfg"
            frame0 = np.full((8, 16), 3, dtype=np.uint16)
            frame1 = np.full((8, 16), 4, dtype=np.uint16)
            frame2 = np.full((8, 16), 5, dtype=np.uint16)
            flat = np.concatenate(
                [
                    np.array([1024], dtype=np.uint16), frame0.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame1.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame2.reshape(-1),
                    np.array([1025], dtype=np.uint16),
                ]
            )
            flat.tofile(token_path)

            class FakeTinyModel:
                def next_frame_logits(self, prefix_frames, *, context_frames):
                    logits = np.full((128, 16), -10.0, dtype=np.float64)
                    next_value = int(prefix_frames[-1].reshape(-1)[0]) + 1
                    logits[:, next_value] = 10.0
                    return logits

            result = encode_commavq_next_frame_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="tiny_frame_predictor_small",
                max_frames=3,
                context_frames=2,
                vocab_size=16,
                verify_decode=True,
                device="cpu",
                model_loader=lambda **_: FakeTinyModel(),
            )

        self.assertEqual(result["command"], "lossless_next_frame_sample")
        self.assertEqual(result["frame_count"], 3)
        self.assertTrue(result["exact_match"])

    def test_encode_commavq_next_frame_sample_builds_default_tiny_frame_predictor_runtime(self) -> None:
        from tac.lossless.next_frame_coder import encode_commavq_next_frame_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.nfg"
            frame0 = np.full((8, 16), 3, dtype=np.uint16)
            frame1 = np.full((8, 16), 4, dtype=np.uint16)
            frame2 = np.full((8, 16), 5, dtype=np.uint16)
            flat = np.concatenate(
                [
                    np.array([1024], dtype=np.uint16), frame0.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame1.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame2.reshape(-1),
                    np.array([1025], dtype=np.uint16),
                ]
            )
            flat.tofile(token_path)

            result = encode_commavq_next_frame_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="tiny_frame_predictor_small",
                max_frames=3,
                context_frames=2,
                verify_decode=False,
                device="cpu",
            )

            self.assertTrue(output_path.exists())

        self.assertEqual(result["command"], "lossless_next_frame_sample")
        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["model_backend"], "tiny_frame_predictor")
        self.assertEqual(result["model_profile"], "tiny_frame_predictor_small")

    def test_encode_commavq_next_frame_sample_loads_tiny_runtime_checkpoint(self) -> None:
        import torch

        from tac.lossless.next_frame_coder import encode_commavq_next_frame_sample
        from tac.lossless.tiny_frame_predictor import (
            load_tiny_frame_predictor_runtime,
            save_tiny_frame_predictor_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_path = root / "tokens.bin"
            output_path = root / "sample.nfg"
            checkpoint_path = root / "tiny_runtime.pt"
            frame0 = np.full((8, 16), 3, dtype=np.uint16)
            frame1 = np.full((8, 16), 4, dtype=np.uint16)
            frame2 = np.full((8, 16), 5, dtype=np.uint16)
            flat = np.concatenate(
                [
                    np.array([1024], dtype=np.uint16), frame0.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame1.reshape(-1),
                    np.array([1024], dtype=np.uint16), frame2.reshape(-1),
                    np.array([1025], dtype=np.uint16),
                ]
            )
            flat.tofile(token_path)

            runtime = load_tiny_frame_predictor_runtime(
                "tiny_frame_predictor_small",
                vocab_size=1024,
                device="cpu",
            )
            with torch.no_grad():
                runtime._model.output_projection.bias.zero_()
                runtime._model.output_projection.bias[11] = 6.0
            save_tiny_frame_predictor_checkpoint(runtime, checkpoint_path)

            result = encode_commavq_next_frame_sample(
                token_path=token_path,
                encoded_path=output_path,
                profile="tiny_frame_predictor_small",
                checkpoint_path=checkpoint_path,
                verify_decode=True,
                device="cpu",
            )

            self.assertTrue(output_path.exists())

        self.assertEqual(result["command"], "lossless_next_frame_sample")
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["model_backend"], "tiny_frame_predictor")
        self.assertEqual(result["model_profile"], "tiny_frame_predictor_small")
        self.assertEqual(result["model_path"], str(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
