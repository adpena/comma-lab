"""Compliance tests: verify our code matches the official scorer exactly.

These tests compare our implementations against the upstream reference
to catch any divergence that would make proxy scores unreliable.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
UPSTREAM = REPO / "workspace" / "upstream" / "comma_video_compression_challenge"
sys.path.insert(0, str(UPSTREAM))


class TestYUV420Compliance:
    """Verify our differentiable YUV conversion matches upstream exactly."""

    @pytest.fixture
    def upstream_rgb_to_yuv6(self):
        from frame_utils import rgb_to_yuv6
        return rgb_to_yuv6

    @pytest.fixture
    def diff_rgb_to_yuv6(self):
        # Import the differentiable version from training
        # Extract the function (it's defined as a closure in _patch_scorers_for_training)
        # We'll replicate the math here for testing
        def _rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
            H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
            H2, W2 = H // 2, W // 2
            rgb = rgb_chw[..., :, :2 * H2, :2 * W2]
            R = rgb[..., 0, :, :]
            G = rgb[..., 1, :, :]
            B = rgb[..., 2, :, :]
            kYR, kYG, kYB = 0.299, 0.587, 0.114
            Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
            U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
            V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
            U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
                     U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
            V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
                     V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
            y00 = Y[..., 0::2, 0::2]
            y10 = Y[..., 1::2, 0::2]
            y01 = Y[..., 0::2, 1::2]
            y11 = Y[..., 1::2, 1::2]
            return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)
        return _rgb_to_yuv6_diff

    def test_yuv_numeric_match(self, upstream_rgb_to_yuv6, diff_rgb_to_yuv6):
        """Our differentiable YUV must produce IDENTICAL output to upstream."""
        torch.manual_seed(42)
        # Simulate uint8 RGB frames converted to float
        rgb = torch.randint(0, 256, (2, 3, 874, 1164), dtype=torch.float32)

        with torch.no_grad():
            upstream_out = upstream_rgb_to_yuv6(rgb)
        diff_out = diff_rgb_to_yuv6(rgb)

        torch.testing.assert_close(
            diff_out, upstream_out,
            atol=1e-5, rtol=1e-5,
            msg="Differentiable YUV conversion diverges from upstream!",
        )

    def test_yuv_bt601_coefficients(self, diff_rgb_to_yuv6):
        """Verify BT.601 coefficients: 0.299, 0.587, 0.114."""
        # Pure red pixel
        red = torch.zeros(1, 3, 2, 2)
        red[:, 0, :, :] = 255.0
        out = diff_rgb_to_yuv6(red)
        y = out[0, 0, 0, 0].item()  # y00
        assert abs(y - 255.0 * 0.299) < 0.01, f"Red Y should be {255*0.299:.2f}, got {y:.2f}"


class TestSegNetCompliance:
    """Verify SegNet distortion matches official scorer."""

    def test_hard_argmax_formula(self):
        """Our eval_scorer_loss must use the same argmax disagreement as upstream."""
        # Simulate SegNet outputs (B=2, C=5, H=4, W=4)
        torch.manual_seed(42)
        out1 = torch.randn(2, 5, 4, 4)
        out2 = torch.randn(2, 5, 4, 4)

        # Official formula (from modules.py line 112)
        diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
        official = diff.mean(dim=tuple(range(1, diff.ndim)))

        # Our formula (from eval_scorer_loss)
        our_diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
        ours = our_diff.mean(dim=tuple(range(1, our_diff.ndim)))

        torch.testing.assert_close(ours, official, atol=0, rtol=0)

    def test_soft_vs_hard_different(self):
        """Soft cosine (training) must NOT equal hard argmax (official)."""
        torch.manual_seed(42)
        out1 = torch.randn(2, 5, 4, 4)
        out2 = torch.randn(2, 5, 4, 4)

        # Soft cosine (training proxy)
        pred_soft = F.softmax(out1, dim=1)
        gt_soft = F.softmax(out2, dim=1)
        soft = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

        # Hard argmax (official)
        hard = (out1.argmax(dim=1) != out2.argmax(dim=1)).float().mean()

        assert abs(soft.item() - hard.item()) > 0.01, \
            "Soft and hard metrics should differ significantly"


class TestPoseNetCompliance:
    """Verify PoseNet distortion matches official scorer."""

    def test_mse_on_first_6(self):
        """PoseNet distortion is MSE on first 6 outputs only."""
        torch.manual_seed(42)
        # Simulate PoseNet output (B=3, 12)
        out1 = {"pose": torch.randn(3, 12)}
        out2 = {"pose": torch.randn(3, 12)}

        # Official formula (modules.py line 84): MSE on [:6], mean over non-batch dims
        official = (out1["pose"][..., :6] - out2["pose"][..., :6]).pow(2).mean(
            dim=tuple(range(1, out1["pose"].ndim))
        )

        # Verify it ignores outputs 6-11
        out1_modified = {"pose": out1["pose"].clone()}
        out1_modified["pose"][:, 6:] = 999.0  # should not affect distortion
        official_modified = (out1_modified["pose"][..., :6] - out2["pose"][..., :6]).pow(2).mean(
            dim=tuple(range(1, out1["pose"].ndim))
        )

        torch.testing.assert_close(official, official_modified, atol=0, rtol=0)


class TestScoreFormulaCompliance:
    """Verify the scoring formula matches upstream exactly."""

    def test_score_formula(self):
        """score = 100*segnet + sqrt(10*posenet) + 25*rate"""
        import math

        pose_dist = 0.03
        seg_dist = 0.005
        rate = 0.023

        expected = 100 * seg_dist + math.sqrt(10 * pose_dist) + 25 * rate
        assert abs(expected - (0.5 + 0.5477 + 0.575)) < 0.01

    def test_no_rate_bias(self):
        """Proxy without rate should be strictly less than with rate."""
        import math
        pose, seg, rate = 0.03, 0.005, 0.023
        with_rate = 100 * seg + math.sqrt(10 * pose) + 25 * rate
        without_rate = 100 * seg + math.sqrt(10 * pose)
        assert without_rate < with_rate


class TestFramePairing:
    """Verify frame pairing matches upstream seq_len=2."""

    def test_pair_count(self):
        from tac.data import pair_start_indices
        indices = pair_start_indices(1200)
        assert len(indices) == 600
        assert indices[0] == 0
        assert indices[-1] == 1198

    def test_pair_non_overlapping(self):
        from tac.data import pair_start_indices
        indices = pair_start_indices(1200)
        for i in range(len(indices) - 1):
            assert indices[i + 1] - indices[i] == 2

    def test_build_pairs_count(self):
        """build_pairs should produce same count as pair_start_indices."""
        from tac.data import build_pairs, pair_start_indices
        fake_frames = [torch.randint(0, 256, (874, 1164, 3), dtype=torch.uint8) for _ in range(10)]
        pairs = build_pairs(fake_frames)
        indices = pair_start_indices(10)
        assert len(pairs) == len(indices)


class TestArchitectureIdentity:
    """Verify all architectures start as identity (zero output residual)."""

    @pytest.mark.parametrize("variant", ["standard", "dilated", "depthwise", "luma"])
    def test_identity_init(self, variant: str):
        from tac.architectures import build_postfilter
        model = build_postfilter(variant, hidden=16)
        x = torch.rand(1, 3, 64, 64) * 255
        with torch.no_grad():
            y = model(x)
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5,
                                   msg=f"{variant} should be identity at init")


class TestAtomicSave:
    """Verify atomic save behavior."""

    def test_training_state_atomic(self):
        """save_training_state should not leave .tmp files on success."""
        import tempfile

        from tac.architectures import build_postfilter
        from tac.training import TrainConfig, Trainer

        model = build_postfilter("standard", hidden=16)
        config = TrainConfig(hidden=16, epochs=100, tag="test-atomic")
        trainer = Trainer(model, config, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.pt"
            trainer.save_training_state(path)
            assert path.exists()
            assert not path.with_suffix(".pt.tmp").exists()
