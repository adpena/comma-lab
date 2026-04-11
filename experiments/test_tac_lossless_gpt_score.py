from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TacLosslessGptScoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
