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


class TacLosslessNextFrameCoderExperimentsTests(unittest.TestCase):
    def test_encode_commavq_next_frame_sample_uses_profile_context_by_default(self) -> None:
        from tac.lossless import next_frame_coder as mod
        from tac.lossless import profiles as profile_mod

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

            class FakeModel:
                def next_frame_logits(self, prefix_frames, *, context_frames):
                    calls.append((prefix_frames.shape[0], context_frames))
                    logits = np.full((128, 8), -10.0, dtype=np.float64)
                    logits[:, int(prefix_frames[-1].reshape(-1)[0]) + 1] = 10.0
                    return logits

            patched_profiles = {
                **profile_mod.PROFILES,
                "gpt_next_frame_small": {
                    "method": "gpt_next_frame",
                    "model": "small",
                    "context_frames": 2,
                },
            }
            with mock.patch.dict(profile_mod.PROFILES, patched_profiles, clear=True):
                result = mod.encode_commavq_next_frame_sample(
                    token_path=token_path,
                    encoded_path=output_path,
                    profile="gpt_next_frame_small",
                    max_frames=3,
                    vocab_size=8,
                    verify_decode=True,
                    model_loader=lambda **_: FakeModel(),
                )

        self.assertEqual(calls, [(1, 2), (2, 2), (1, 2), (2, 2)])
        self.assertEqual(result["context_frames"], 2)
        self.assertTrue(result["exact_match"])
        self.assertTrue(result["local_only"])
        self.assertFalse(result["measured"])


if __name__ == "__main__":
    unittest.main()
