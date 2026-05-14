# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessTinyFramePredictorTests(unittest.TestCase):
    def test_tiny_frame_predictor_emits_next_frame_logits(self) -> None:
        import torch

        from tac.lossless.tiny_frame_predictor import TinyFramePredictorConfig, build_tiny_frame_predictor

        config = TinyFramePredictorConfig(
            context_frames=4,
            positions=8,
            vocab_size=17,
            embed_dim=16,
            hidden_dim=32,
            mixer_layers=2,
        )
        model = build_tiny_frame_predictor(config)
        tokens = torch.randint(0, config.vocab_size, (3, config.context_frames, config.positions), dtype=torch.long)

        logits = model(tokens)

        self.assertEqual(tuple(logits.shape), (3, config.positions, config.vocab_size))
        self.assertEqual(logits.dtype, torch.float32)

    def test_tiny_frame_predictor_rejects_wrong_context_length(self) -> None:
        import torch

        from tac.lossless.tiny_frame_predictor import TinyFramePredictorConfig, build_tiny_frame_predictor

        config = TinyFramePredictorConfig(
            context_frames=4,
            positions=8,
            vocab_size=17,
            embed_dim=16,
            hidden_dim=32,
            mixer_layers=2,
        )
        model = build_tiny_frame_predictor(config)
        wrong = torch.randint(0, config.vocab_size, (1, config.context_frames - 1, config.positions), dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "context_frames"):
            model(wrong)

    def test_tiny_frame_predictor_default_profile_stays_small(self) -> None:
        from tac.lossless.profiles import load_tiny_frame_predictor_profile
        from tac.lossless.tiny_frame_predictor import summarize_tiny_frame_predictor

        config = load_tiny_frame_predictor_profile("tiny_frame_predictor_small")
        summary = summarize_tiny_frame_predictor(config)

        self.assertEqual(summary["command"], "lossless_tiny_frame_predictor_summary")
        self.assertLess(summary["parameter_count"], 1_000_000)
        self.assertEqual(summary["context_frames"], 8)
        self.assertEqual(summary["positions"], 128)

    def test_load_tiny_frame_predictor_runtime_is_deterministic(self) -> None:
        from tac.lossless.tiny_frame_predictor import load_tiny_frame_predictor_runtime

        prefix = np.arange(128, dtype=np.uint16).reshape(1, 8, 16) % 16

        model_a = load_tiny_frame_predictor_runtime(
            "tiny_frame_predictor_small",
            context_frames=2,
            vocab_size=16,
            device="cpu",
        )
        model_b = load_tiny_frame_predictor_runtime(
            "tiny_frame_predictor_small",
            context_frames=2,
            vocab_size=16,
            device="cpu",
        )

        logits_a = model_a.next_frame_logits(prefix, context_frames=2)
        logits_b = model_b.next_frame_logits(prefix, context_frames=2)

        self.assertEqual(model_a._tac_model_backend, "tiny_frame_predictor")
        self.assertEqual(model_a._tac_model_profile, "tiny_frame_predictor_small")
        self.assertEqual(tuple(logits_a.shape), (128, 16))
        self.assertTrue(np.array_equal(logits_a, logits_b))

    def test_tiny_frame_predictor_checkpoint_roundtrips_and_runtime_can_load_it(self) -> None:
        import torch

        from tac.lossless.tiny_frame_predictor import (
            TinyFramePredictorConfig,
            build_tiny_frame_predictor,
            load_tiny_frame_predictor_checkpoint,
            load_tiny_frame_predictor_runtime,
            save_tiny_frame_predictor_checkpoint,
        )

        config = TinyFramePredictorConfig(
            context_frames=2,
            positions=128,
            vocab_size=16,
            embed_dim=8,
            hidden_dim=16,
            mixer_layers=1,
        )
        model = build_tiny_frame_predictor(config)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            model.output_projection.bias[7] = 9.0

        prefix = np.arange(128, dtype=np.uint16).reshape(1, 8, 16) % config.vocab_size

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "tiny_predictor.pt"
            save_tiny_frame_predictor_checkpoint(model, checkpoint_path, config=config)
            checkpoint = load_tiny_frame_predictor_checkpoint(checkpoint_path)
            loaded_runtime = load_tiny_frame_predictor_runtime(
                config,
                checkpoint_path=checkpoint_path,
                device="cpu",
            )

            self.assertEqual(checkpoint.config, config)
            self.assertEqual(loaded_runtime._tac_model_artifact_path, str(checkpoint_path))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(checkpoint.state_dict[name], value), msg=name)

        baseline_runtime = load_tiny_frame_predictor_runtime(config, device="cpu")
        baseline_logits = baseline_runtime.next_frame_logits(prefix, context_frames=config.context_frames)
        loaded_logits = loaded_runtime.next_frame_logits(prefix, context_frames=config.context_frames)

        self.assertEqual(int(np.argmax(loaded_logits[0])), 7)
        self.assertFalse(np.array_equal(loaded_logits, baseline_logits))


if __name__ == "__main__":
    unittest.main()
