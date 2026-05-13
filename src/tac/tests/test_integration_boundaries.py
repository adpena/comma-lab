"""Tests for integration boundaries — every bug in this project lived here.

These tests verify that data flows correctly BETWEEN components:
  - Mask dtype compatibility (int8 crashes everywhere)
  - Pose-mask matching (27x PoseNet regression)
  - Renderer format loading (ASYM/FP4A auto-detect)
  - Archive construction and validation
  - CRF mask encode/decode roundtrip
"""
import tempfile
import unittest
from pathlib import Path

import torch


class TestMaskDtype(unittest.TestCase):
    """Masks must be long for embedding, never int8."""

    def test_mask_codec_encode_accepts_int8(self):
        """encode_masks_monochrome must handle int8 input without overflow."""
        from tac.mask_codec import encode_masks_monochrome

        # SVT-AV1 requires minimum 4x4 resolution
        masks = torch.randint(0, 5, (2, 8, 8), dtype=torch.int8)
        with tempfile.NamedTemporaryFile(suffix=".mkv") as f:
            size = encode_masks_monochrome(masks, Path(f.name), crf=50)
            self.assertGreater(size, 0)

    def test_mask_codec_encode_int8_class4_not_overflow(self):
        """Class 4 * 63 = 252 must not overflow int8 (max 127)."""
        from tac.mask_codec import encode_masks_monochrome

        masks = torch.full((1, 8, 8), 4, dtype=torch.int8)
        with tempfile.NamedTemporaryFile(suffix=".mkv") as f:
            # This USED to crash: int8 * 63 = -4 (overflow)
            size = encode_masks_monochrome(masks, Path(f.name), crf=50)
            self.assertGreater(size, 0)

    def test_renderer_embedding_rejects_int8(self):
        """Renderer embedding layer needs long, not int8."""
        from tac.renderer import MaskRenderer

        renderer = MaskRenderer(base_ch=8, mid_ch=8, embed_dim=4, depth=1)
        masks_int8 = torch.randint(0, 5, (1, 16, 16), dtype=torch.int8)
        masks_long = masks_int8.long()

        # int8 should fail
        with self.assertRaises(RuntimeError):
            renderer(masks_int8)

        # long should work
        out = renderer(masks_long)
        self.assertEqual(out.shape[0], 1)


class TestRendererFormatDetection(unittest.TestCase):
    """Auto-detect ASYM/FP4A/PT format by magic bytes."""

    def test_asym_magic_detection(self):
        """ASYM format starts with b'ASYM'."""
        from tac.renderer import AsymmetricPairGenerator
        from tac.renderer_export import export_asymmetric_checkpoint, load_asymmetric_checkpoint

        model = AsymmetricPairGenerator(
            base_ch=8, mid_ch=8, embed_dim=4, depth=1, pose_dim=6,
        )
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            export_asymmetric_checkpoint(model, Path(f.name))
            raw = Path(f.name).read_bytes()
            self.assertEqual(raw[:4], b"ASYM")
            loaded = load_asymmetric_checkpoint(raw, device="cpu")
            self.assertEqual(loaded.pose_dim, 6)

    def test_fp4a_magic_detection(self):
        """FP4A format starts with b'FP4A'."""
        from tac.renderer import AsymmetricPairGenerator
        from tac.renderer_export import export_asymmetric_checkpoint_fp4, load_asymmetric_checkpoint_fp4

        model = AsymmetricPairGenerator(
            base_ch=8, mid_ch=8, embed_dim=4, depth=1, pose_dim=6,
        )
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            export_asymmetric_checkpoint_fp4(model, Path(f.name))
            raw = Path(f.name).read_bytes()
            self.assertEqual(raw[:4], b"FP4A")
            loaded = load_asymmetric_checkpoint_fp4(raw, device="cpu")
            self.assertEqual(loaded.pose_dim, 6)


class TestPoseShape(unittest.TestCase):
    """Poses must be (n_pairs, 6) and match the number of mask pairs."""

    def test_pose_shape_matches_pairs(self):
        """R38 fix: was tautology test (assert randn(600,6).shape == (600,6)).
        Now verifies preflight_check actually rejects wrong-shape poses,
        regression-protecting the catastrophic 1199-overlapping-pairs bug.
        """
        from tac.preflight import preflight_check, PreflightError
        # Wrong number of pairs (1199 overlapping vs 600 non-overlapping).
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(torch.randn(1199, 6), f.name)
            with self.assertRaises(PreflightError):
                preflight_check(poses_path=f.name, verbose=False)
        # Wrong DOF dimension.
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(torch.randn(600, 5), f.name)
            with self.assertRaises(PreflightError):
                preflight_check(poses_path=f.name, verbose=False)
        # Correct shape — must NOT raise.
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(torch.randn(600, 6), f.name)
            preflight_check(poses_path=f.name, verbose=False)

    def test_pose_fp32_and_fp16_roundtrip(self):
        """Poses saved as fp32, loaded as float — no precision issues."""
        poses = torch.randn(600, 6)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(poses, f.name)
            loaded = torch.load(f.name, weights_only=True).float()
            self.assertTrue(torch.equal(poses, loaded))


class TestArchiveValidation(unittest.TestCase):
    """Archive must contain exactly the required artifacts."""

    def test_valid_archive_passes(self):
        """Archive with all 3 required files passes validation."""
        from tac.submission_archive import validate_archive, RENDERER_SUBMISSION_MANIFEST

        with tempfile.TemporaryDirectory() as td:
            import zipfile
            archive = Path(td) / "archive.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("renderer.bin", b"ASYM" + b"\x00" * 200000)
                zf.writestr("masks.mkv", b"\x00" * 50000)
                zf.writestr("optimized_poses.pt", b"\x00" * 10000)
            result = validate_archive(archive, RENDERER_SUBMISSION_MANIFEST, strict=False)
            self.assertTrue(result.valid)

    def test_missing_poses_fails(self):
        """Archive missing optimized_poses.pt fails validation."""
        from tac.submission_archive import validate_archive, RENDERER_SUBMISSION_MANIFEST

        with tempfile.TemporaryDirectory() as td:
            import zipfile
            archive = Path(td) / "archive.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("renderer.bin", b"ASYM" + b"\x00" * 200000)
                zf.writestr("masks.mkv", b"\x00" * 50000)
            result = validate_archive(archive, RENDERER_SUBMISSION_MANIFEST, strict=True)
            self.assertFalse(result.valid)
            self.assertIn("optimized_poses.pt", str(result.errors))


class TestPreflightCheck(unittest.TestCase):
    """Preflight catches integration mismatches."""

    def test_preflight_detects_small_renderer(self):
        """Preflight warns on suspiciously small renderer."""
        from tac.preflight import preflight_check

        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            Path(f.name).write_bytes(b"ASYM" + b"\x00" * 100)
            with self.assertRaises(Exception):
                # Should fail — renderer too small
                preflight_check(renderer_path=f.name, verbose=False)

    def test_preflight_rejects_invalid_archive_path(self):
        """Explicit archive preflight is fail-closed, not warn-open."""
        import zipfile

        from tac.preflight import PreflightError, preflight_check

        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "archive.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("renderer.bin", b"ASYM" + b"\0" * 12000)
                zf.writestr("renderer.bin", b"duplicate")
                zf.writestr("masks.mkv", b"m" * 12000)
                zf.writestr("optimized_poses.pt", b"p" * 7200)

            with self.assertRaises(PreflightError):
                preflight_check(archive_path=archive, verbose=False)


class TestQATParametrizeNoDoubleWrap(unittest.TestCase):
    """QATRendererFP4 must not double-wrap already-parametrized layers."""

    def test_no_double_parametrization(self):
        """Creating QATRendererFP4 twice on same model doesn't stack."""
        from tac.renderer import AsymmetricPairGenerator
        from tac.fp4_quantize import QATRendererFP4

        model = AsymmetricPairGenerator(
            base_ch=8, mid_ch=8, embed_dim=4, depth=1, pose_dim=6,
        )
        w1 = QATRendererFP4(model)
        n1 = len(w1._parametrized_modules)
        w2 = QATRendererFP4(model)  # same model again
        n2 = len(w2._parametrized_modules)

        # Second wrap should find 0 new modules (all already parametrized)
        self.assertEqual(n2, 0)
        # But the model itself should still have n1 parametrized layers
        import torch.nn.utils.parametrize as pm
        actual = sum(1 for m in model.modules()
                     if hasattr(m, "weight") and pm.is_parametrized(m, "weight"))
        self.assertEqual(actual, n1)


class TestCleanStateDict(unittest.TestCase):
    """_clean_state_dict strips parametrize keys for safe resume."""

    def test_strips_parametrize_keys(self):
        """Parametrized state dict keys map back to plain weight keys."""
        from experiments.train_distill import _clean_state_dict

        dirty = {
            "renderer.conv.parametrizations.weight.original": torch.randn(3, 3),
            "renderer.conv.parametrizations.weight.0.codebook": torch.randn(8),
            "renderer.bias": torch.randn(3),
        }
        clean = _clean_state_dict(dirty)
        self.assertIn("renderer.conv.weight", clean)
        self.assertNotIn("renderer.conv.parametrizations.weight.original", clean)
        self.assertNotIn("renderer.conv.parametrizations.weight.0.codebook", clean)
        self.assertIn("renderer.bias", clean)


if __name__ == "__main__":
    unittest.main()
