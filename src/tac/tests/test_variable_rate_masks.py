"""Tests for variable-rate per-frame mask encoding and difficulty map computation.

Variable-rate encoding allocates more bits to hard frames (lower CRF)
and fewer bits to easy frames (higher CRF). Same total bytes, better
score on the hard pairs that dominate PoseNet via the sqrt asymmetry.

Difficulty map identifies which pairs are hard by running the renderer
on all pairs and computing per-pair PoseNet distortion at compress time.
"""
import tempfile
import unittest
from pathlib import Path

import torch


class TestDifficultyMap(unittest.TestCase):
    """Difficulty map computation at compress time."""

    def test_compute_difficulty_returns_per_pair_scores(self):
        """compute_pair_difficulty returns (n_pairs,) float tensor."""
        from tac.variable_rate import compute_pair_difficulty

        # Simulate: 10 frames (5 pairs), fake renderer output
        n_pairs = 5
        # Mock difficulty — just needs shape and type
        difficulty = compute_pair_difficulty(
            n_pairs=n_pairs,
            pose_distortions=torch.tensor([0.01, 0.5, 0.02, 0.8, 0.03]),
        )
        self.assertEqual(difficulty.shape, (n_pairs,))
        self.assertEqual(difficulty.dtype, torch.float32)

    def test_difficulty_map_identifies_hard_pairs(self):
        """Hard pairs (high pose_d) get highest difficulty scores."""
        from tac.variable_rate import compute_pair_difficulty

        pose_d = torch.tensor([0.001, 0.5, 0.002, 0.9, 0.003])
        difficulty = compute_pair_difficulty(n_pairs=5, pose_distortions=pose_d)

        # Pair 3 (pose_d=0.9) should be hardest
        hardest = difficulty.argmax().item()
        self.assertEqual(hardest, 3)

    def test_difficulty_map_serialization(self):
        """Difficulty map round-trips through save/load (2.4KB for 600 pairs)."""
        from tac.variable_rate import save_difficulty_map, load_difficulty_map

        difficulty = torch.rand(600)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            nbytes = save_difficulty_map(difficulty, Path(f.name))
            self.assertLess(nbytes, 5000)  # under 5KB
            loaded = load_difficulty_map(Path(f.name))
            self.assertTrue(torch.allclose(difficulty, loaded))


class TestVariableRateEncoding(unittest.TestCase):
    """Per-frame CRF allocation based on difficulty."""

    def test_allocate_crf_per_frame(self):
        """Hard frames get lower CRF (better quality), easy get higher."""
        from tac.variable_rate import allocate_crf_per_frame

        difficulty = torch.tensor([0.01, 0.5, 0.02, 0.9, 0.03])
        crfs = allocate_crf_per_frame(
            difficulty=difficulty,
            crf_easy=60,
            crf_hard=20,
            hard_fraction=0.2,
        )
        self.assertEqual(len(crfs), 5)
        # Hardest pair (index 3) should get crf_hard
        self.assertEqual(crfs[3], 20)
        # Easy pairs should get crf_easy
        self.assertEqual(crfs[0], 60)

    def test_allocate_crf_respects_fractions(self):
        """Exactly hard_fraction of pairs get the hard CRF."""
        from tac.variable_rate import allocate_crf_per_frame

        difficulty = torch.rand(100)
        crfs = allocate_crf_per_frame(
            difficulty=difficulty,
            crf_easy=60,
            crf_hard=20,
            hard_fraction=0.3,
        )
        n_hard = sum(1 for c in crfs if c == 20)
        self.assertEqual(n_hard, 30)

    def test_encode_variable_rate_masks(self):
        """Variable-rate encoding produces a valid mask file."""
        from tac.variable_rate import encode_variable_rate_masks

        # Minimal masks: 20 frames (10 pairs) at 8x8
        masks = torch.randint(0, 5, (20, 8, 8)).long()
        difficulty = torch.rand(10)

        with tempfile.NamedTemporaryFile(suffix=".mkv") as f:
            size = encode_variable_rate_masks(
                masks=masks,
                difficulty=difficulty,
                output_path=Path(f.name),
                crf_easy=60,
                crf_hard=30,
                hard_fraction=0.2,
            )
            self.assertGreater(size, 0)

    def test_decode_variable_rate_masks_roundtrip(self):
        """Variable-rate encoded masks decode to correct class indices."""
        from tac.variable_rate import encode_variable_rate_masks, decode_masks

        masks = torch.randint(0, 5, (4, 16, 16)).long()
        difficulty = torch.rand(2)

        with tempfile.NamedTemporaryFile(suffix=".mkv") as f:
            encode_variable_rate_masks(
                masks=masks,
                difficulty=difficulty,
                output_path=Path(f.name),
                crf_easy=30,
                crf_hard=20,
                hard_fraction=0.5,
            )
            decoded = decode_masks(Path(f.name))
            self.assertEqual(decoded.shape, masks.shape)
            # Should be mostly correct (lossy compression at class boundaries)
            accuracy = (decoded == masks).float().mean().item()
            self.assertGreater(accuracy, 0.95)


class TestVariableRateVsUniform(unittest.TestCase):
    """Variable-rate should produce same or smaller file for same quality."""

    def test_variable_rate_same_total_bytes(self):
        """Variable-rate with same average CRF produces similar file size."""
        from tac.variable_rate import encode_variable_rate_masks
        from tac.mask_codec import encode_masks_monochrome

        masks = torch.randint(0, 5, (20, 16, 16)).long()
        difficulty = torch.rand(10)

        with tempfile.NamedTemporaryFile(suffix=".mkv") as f_uniform, \
             tempfile.NamedTemporaryFile(suffix=".mkv") as f_variable:
            uniform_size = encode_masks_monochrome(masks, Path(f_uniform.name), crf=45)
            variable_size = encode_variable_rate_masks(
                masks=masks, difficulty=difficulty,
                output_path=Path(f_variable.name),
                crf_easy=60, crf_hard=30, hard_fraction=0.2,
            )
            # Variable rate should be in the same ballpark (within 2x)
            self.assertLess(variable_size, uniform_size * 2)
            self.assertGreater(variable_size, 0)


if __name__ == "__main__":
    unittest.main()
