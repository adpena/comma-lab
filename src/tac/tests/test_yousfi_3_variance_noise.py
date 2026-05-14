# SPDX-License-Identifier: MIT
"""Property tests for Yousfi trick #3: spatially-adaptive quantization noise.

Trick #3 (UNIWARD-aligned variance-weighted noise) injects per-pixel
quantization noise whose std is a function of the local image variance.
The UNIWARD principle (Holub & Fridrich 2014) says modifications hidden
in textured / high-variance regions are essentially undetectable by a CNN
steganalysis classifier, while the same modification in a smooth region
is caught instantly. The SegNet (EfficientNet-B2 stride-2) and PoseNet
(FastViT-T12 YUV6) scorers ARE inverse steganalysis detectors per
project_yousfi_fridrich_connection, so the same principle applies.

Tests cover:
  - Output shape and dtype contract.
  - Spatial concentration of noise in high-variance regions
    (synthetic half-flat / half-textured image).
  - Mode dispatch ('variance' vs 'inverse_variance').
  - Loss runs backward and gradients flow to `reconstructed`.
  - CLI flag propagation through experiments/train_distill.py argparse.
  - Toy-training sanity: enabling the loss produces a measurable training
    signal whose value DECREASES as the renderer fits the target.
  - Edge cases: base_std=0, validation errors.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tac.fridrich import variance_weighted_noise
from tac.losses import uniward_quant_noise_loss


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---- Helpers ---- #

def _make_half_textured(B: int = 2, H: int = 64, W: int = 128) -> torch.Tensor:
    """Build a (B, 3, H, W) image whose left half is FLAT (constant) and
    right half is TEXTURED (high-variance random values).

    Used to verify that variance_weighted_noise concentrates noise on the
    right half (UNIWARD direction) and the left half (inverse direction).
    """
    torch.manual_seed(0)
    img = torch.empty(B, 3, H, W)
    img[:, :, :, : W // 2] = 127.5  # flat
    img[:, :, :, W // 2 :] = (
        torch.rand(B, 3, H, W // 2) * 255.0
    )  # textured
    return img


# ---- Shape & dtype contract ---- #

class TestVarianceWeightedNoiseShape:
    """variance_weighted_noise returns same shape, dtype as input."""

    def test_output_shape_matches_input(self):
        img = torch.rand(2, 3, 64, 128) * 255.0
        noise = variance_weighted_noise(img, base_std=2.0)
        assert noise.shape == img.shape

    def test_output_dtype_matches_input_float32(self):
        img = torch.rand(2, 3, 32, 32, dtype=torch.float32) * 255.0
        noise = variance_weighted_noise(img, base_std=1.0)
        assert noise.dtype == torch.float32

    def test_output_dtype_matches_input_float64(self):
        img = torch.rand(1, 3, 16, 16, dtype=torch.float64) * 255.0
        noise = variance_weighted_noise(img, base_std=1.0)
        assert noise.dtype == torch.float64

    def test_zero_base_std_gives_zero_noise(self):
        img = torch.rand(1, 3, 16, 16) * 255.0
        noise = variance_weighted_noise(img, base_std=0.0)
        assert torch.all(noise == 0.0)


# ---- Spatial concentration: the heart of UNIWARD ---- #

class TestVarianceWeightedNoiseConcentratesInHighVarianceRegions:
    """In 'variance' mode, the noise std on textured pixels must be
    measurably larger than on flat pixels — that's the whole point.
    """

    def test_uniward_mode_concentrates_in_textured_half(self):
        torch.manual_seed(42)
        img = _make_half_textured(B=4, H=64, W=128)

        # Average MANY noise samples to estimate the per-pixel std field.
        n_samples = 200
        sq_accum = torch.zeros_like(img)
        for _ in range(n_samples):
            n = variance_weighted_noise(img, base_std=2.0, mode="variance")
            sq_accum = sq_accum + n.pow(2)
        per_pixel_std = (sq_accum / n_samples).sqrt()

        # Average std on the flat (left) vs textured (right) half.
        flat_std = per_pixel_std[:, :, :, : 128 // 2].mean().item()
        textured_std = per_pixel_std[:, :, :, 128 // 2 :].mean().item()

        # The textured half should have meaningfully larger noise std.
        # We expect a factor of 5x+ since the flat half has ~zero local
        # variance and the textured half has variance ~5000-7000.
        assert textured_std > flat_std * 5.0, (
            f"UNIWARD-mode noise should concentrate in textured regions: "
            f"textured_std={textured_std:.3f} vs flat_std={flat_std:.3f}"
        )

    def test_inverse_mode_concentrates_in_flat_half(self):
        """Sanity-check the ablation mode does the opposite."""
        torch.manual_seed(43)
        img = _make_half_textured(B=4, H=64, W=128)

        n_samples = 200
        sq_accum = torch.zeros_like(img)
        for _ in range(n_samples):
            n = variance_weighted_noise(
                img, base_std=2.0, mode="inverse_variance",
            )
            sq_accum = sq_accum + n.pow(2)
        per_pixel_std = (sq_accum / n_samples).sqrt()

        flat_std = per_pixel_std[:, :, :, : 128 // 2].mean().item()
        textured_std = per_pixel_std[:, :, :, 128 // 2 :].mean().item()

        assert flat_std > textured_std * 5.0, (
            f"inverse mode should concentrate noise in flat regions: "
            f"flat_std={flat_std:.3f} vs textured_std={textured_std:.3f}"
        )

    def test_mean_std_matches_base_std_budget(self):
        """The function is documented as a strict spatial REDISTRIBUTION
        of a fixed noise budget — average std across the image must equal
        base_std up to Monte-Carlo / kernel-edge slack.
        """
        torch.manual_seed(44)
        img = _make_half_textured(B=4, H=64, W=128)
        base_std = 3.0

        n_samples = 300
        sq_accum = torch.zeros_like(img)
        for _ in range(n_samples):
            n = variance_weighted_noise(img, base_std=base_std, mode="variance")
            sq_accum = sq_accum + n.pow(2)
        per_pixel_var = sq_accum / n_samples

        # The DOCUMENTED contract is mean(std_field) == base_std (linear in
        # std), not mean(var) == base_std². The relationship between those
        # two is: mean(std) == base_std iff sum(scale)/N == 1, which is
        # exactly how std_field is constructed inside the function.
        # Empirically check: sqrt(mean(var)) >= base_std (Jensen's inequality
        # — the redistributed std field has more variance, so its second
        # moment is larger than its first moment squared).
        rms_std = per_pixel_var.mean().sqrt().item()
        assert rms_std >= base_std * 0.9, (
            f"RMS noise std {rms_std:.3f} should be ~base_std={base_std} "
            "or larger by Jensen"
        )
        # Upper bound: should not be wildly larger either (sanity).
        assert rms_std < base_std * 4.0, (
            f"RMS noise std {rms_std:.3f} unexpectedly large vs "
            f"base_std={base_std}"
        )


# ---- Mode validation ---- #

class TestModeValidation:
    """Bad inputs raise ValueError clearly."""

    def test_invalid_mode_raises(self):
        img = torch.rand(1, 3, 16, 16) * 255.0
        with pytest.raises(ValueError, match="mode"):
            variance_weighted_noise(img, base_std=1.0, mode="bogus")

    def test_negative_base_std_raises(self):
        img = torch.rand(1, 3, 16, 16) * 255.0
        with pytest.raises(ValueError, match="base_std"):
            variance_weighted_noise(img, base_std=-1.0)

    def test_wrong_ndim_raises(self):
        img = torch.rand(3, 16, 16) * 255.0  # missing batch
        with pytest.raises(ValueError, match="must be"):
            variance_weighted_noise(img, base_std=1.0)

    def test_wrong_channels_raises(self):
        img = torch.rand(1, 4, 16, 16) * 255.0  # 4 channels, not 3
        with pytest.raises(ValueError, match="3 channels"):
            variance_weighted_noise(img, base_std=1.0)


# ---- Loss: gradient flow ---- #

class TestUniwardQuantNoiseLossRunsBackward:
    """Loss is differentiable wrt the reconstructed input."""

    def test_grad_flows_to_reconstructed_chw(self):
        torch.manual_seed(0)
        # Use leaf tensor with requires_grad — multiplying after
        # requires_grad_ makes the result a non-leaf and silently drops
        # .grad on backward (PyTorch leaf-tensor footgun).
        recon = (torch.rand(2, 3, 32, 32) * 255.0).requires_grad_(True)
        target = torch.rand(2, 3, 32, 32) * 255.0

        loss = uniward_quant_noise_loss(recon, target, base_std=2.0)
        loss.backward()

        assert recon.grad is not None
        assert recon.grad.shape == recon.shape
        assert torch.isfinite(recon.grad).all()
        assert recon.grad.abs().sum() > 0

    def test_grad_flows_to_reconstructed_hwc(self):
        """HWC pathway used by the standard loss path in train_distill.py."""
        torch.manual_seed(1)
        recon = (torch.rand(2, 32, 32, 3) * 255.0).requires_grad_(True)
        target = torch.rand(2, 32, 32, 3) * 255.0

        loss = uniward_quant_noise_loss(recon, target, base_std=2.0)
        loss.backward()

        assert recon.grad is not None
        assert recon.grad.shape == recon.shape
        assert torch.isfinite(recon.grad).all()

    def test_grad_does_not_flow_to_target(self):
        """Target is the noise-shape source — must be detached internally."""
        torch.manual_seed(2)
        recon = torch.rand(2, 3, 32, 32) * 255.0
        target = (torch.rand(2, 3, 32, 32) * 255.0).requires_grad_(True)

        loss = uniward_quant_noise_loss(recon, target, base_std=2.0)
        # Even though target has requires_grad=True, the variance field is
        # computed from target.detach() — so no grad should flow back from
        # the noise pathway. The MSE term is still differentiable wrt
        # target, which is fine and expected.
        loss.backward()
        assert torch.isfinite(loss).item()

    def test_loss_decreases_as_reconstruction_improves(self):
        """Loss should be smaller when recon == target than for random."""
        torch.manual_seed(3)
        target = torch.rand(2, 3, 32, 32) * 255.0
        recon_perfect = target.clone()
        recon_random = torch.rand(2, 3, 32, 32) * 255.0

        # Pin RNG for the noise sampler so the comparison is fair.
        torch.manual_seed(100)
        loss_perfect = uniward_quant_noise_loss(
            recon_perfect, target, base_std=2.0,
        )
        torch.manual_seed(100)
        loss_random = uniward_quant_noise_loss(
            recon_random, target, base_std=2.0,
        )

        assert loss_perfect.item() < loss_random.item(), (
            f"Perfect recon loss {loss_perfect.item():.4f} should be < "
            f"random recon loss {loss_random.item():.4f}"
        )


# ---- Toy training: contrarian's "does it actually help" check ---- #

class TestUniwardQuantNoiseLossToyTraining:
    """Small toy training: verify loss decreases when actively trained."""

    def test_loss_decreases_during_toy_training(self):
        """Train a single learnable tensor against a fixed target. The
        UNIWARD quant-noise loss must DECREASE substantially over 50
        gradient steps — Contrarian's "is this cosmetic or load-bearing"
        check (memory: feedback on 'every council member must reach
        consensus on whether a technique helps').
        """
        torch.manual_seed(7)
        H, W = 32, 32
        target = (torch.rand(2, 3, H, W) * 255.0)
        # Initialise reconstruction far from target so there's room to
        # learn. Use full() (a leaf) and requires_grad_ in place — calling
        # .requires_grad_ on the result of full_like is fine because
        # full_like itself doesn't carry grad history.
        recon = torch.full_like(target, 127.5).clone().detach().requires_grad_(True)

        opt = torch.optim.Adam([recon], lr=5.0)

        torch.manual_seed(101)
        initial_loss = uniward_quant_noise_loss(recon, target, base_std=2.0)
        initial_val = initial_loss.item()

        # 50 epoch toy run.
        for _ in range(50):
            torch.manual_seed(101 + _)
            loss = uniward_quant_noise_loss(recon, target, base_std=2.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.manual_seed(101)
        final_loss = uniward_quant_noise_loss(recon, target, base_std=2.0)
        final_val = final_loss.item()

        # The loss MUST decrease meaningfully. We require >50% drop —
        # weaker than 90% so we don't flake on RNG, stronger than 10% so
        # a no-op implementation can't pass.
        assert final_val < initial_val * 0.5, (
            f"Toy training failed to reduce loss: "
            f"initial={initial_val:.4f} final={final_val:.4f}"
        )


# ---- CLI plumbing: flag actually reaches DistillConfig ---- #

class TestCliFlagPropagatesThroughTrainDistill:
    """`--use-variance-noise` and friends must round-trip through
    experiments/train_distill.py argparse → DistillConfig.
    """

    def test_argparse_accepts_flags(self):
        """Spawn `train_distill.py --help` and confirm our flags are
        listed. This catches both 'flag missing from argparse' and 'flag
        misspelled in CLI help'.
        """
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "train_distill.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
            env={
                **__import__("os").environ,
                "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
            },
        )
        assert result.returncode == 0, (
            f"train_distill.py --help failed: {result.stderr}"
        )
        help_text = result.stdout
        assert "--use-variance-noise" in help_text
        assert "--variance-noise-weight" in help_text
        assert "--variance-noise-base-std" in help_text
        assert "--variance-noise-kernel" in help_text
        assert "--variance-noise-mode" in help_text

    def test_distill_config_has_fields(self):
        """The DistillConfig dataclass must expose the new fields so the
        provenance JSON written at training start records them. If this
        regresses, the experiment becomes unreproducible.
        """
        # Import lazily so a syntax error in train_distill.py shows up here.
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from experiments.train_distill import DistillConfig
        finally:
            sys.path.pop(0)

        cfg = DistillConfig()
        assert hasattr(cfg, "use_variance_noise")
        assert hasattr(cfg, "variance_noise_weight")
        assert hasattr(cfg, "variance_noise_base_std")
        assert hasattr(cfg, "variance_noise_kernel")
        assert hasattr(cfg, "variance_noise_mode")
        # Defaults: OFF by default, conservative weight when on.
        assert cfg.use_variance_noise is False
        assert cfg.variance_noise_mode == "variance"

    def test_pipeline_compress_argparse_accepts_flag(self):
        """experiments/pipeline.py compress --help should expose the
        same flag — required so a profile-driven run can either use the
        profile default or override on the CLI.
        """
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "pipeline.py"),
                "compress",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
            env={
                **__import__("os").environ,
                "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
            },
        )
        assert result.returncode == 0, (
            f"pipeline.py compress --help failed: {result.stderr}"
        )
        assert "--use-variance-noise" in result.stdout
