from __future__ import annotations

import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
