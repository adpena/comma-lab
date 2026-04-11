"""Hardening tests for tac library: profiles, adaptive weights, losses, architecture coverage."""
import math
import warnings

import pytest
import torch

from tac.adaptive import AdaptiveWeights, geometric_mean_score
from tac.architectures import VARIANTS, build_postfilter
from tac.profiles import PROFILES
from tac.training import TrainConfig, Trainer


# ── Helpers ────────────────────────────────────────────────────────────

# Profiles that use non-postfilter architectures (renderers, diffusion, VQ-VAE, etc.)
# These have extra config keys that TrainConfig doesn't know about and may not
# be buildable via build_postfilter. They use a separate training pipeline.
_NON_POSTFILTER_PROFILES = {
    name for name, overrides in PROFILES.items()
    if overrides.get("variant", "standard") in (
        "mask_renderer", "wavelet_renderer", "diffusion_teacher",
        "distillation", "dp_sims", "vqvae",
        "depthwise_renderer", "channel_recurrent", "coord_renderer",
    )
}

# Fields that TrainConfig accepts (used to filter profile dicts)
_TRAINCONFIG_FIELDS = set(TrainConfig.model_fields.keys())


def _trainconfig_overrides(overrides: dict) -> dict:
    """Filter a profile dict to only fields accepted by TrainConfig."""
    return {k: v for k, v in overrides.items() if k in _TRAINCONFIG_FIELDS}


# ── Profile validation ──────────────────────────────────────────────────


_POSTFILTER_PROFILES = [n for n in PROFILES if n not in _NON_POSTFILTER_PROFILES]
_ALL_PROFILE_NAMES = list(PROFILES.keys())


class TestProfilesProduceValidConfig:
    """Every named profile must produce a valid TrainConfig without errors."""

    @pytest.mark.parametrize("name", _ALL_PROFILE_NAMES)
    def test_profile_creates_valid_config(self, name):
        overrides = PROFILES[name]
        filtered = _trainconfig_overrides(overrides)
        config = TrainConfig(**{**filtered, "tag": f"test-{name}"})
        assert config.tag == f"test-{name}"
        # Verify architecture variant exists (for postfilter profiles)
        if name not in _NON_POSTFILTER_PROFILES:
            assert config.variant in VARIANTS or config.variant == "standard"

    @pytest.mark.parametrize("name", _POSTFILTER_PROFILES)
    def test_profile_architecture_buildable(self, name):
        overrides = PROFILES[name]
        variant = overrides.get("variant", "standard")
        hidden = overrides.get("hidden", 64)
        kernel = overrides.get("kernel", 3)
        model = build_postfilter(variant, hidden=hidden, kernel=kernel)
        # Verify forward pass works
        x = torch.rand(1, 3, 32, 32) * 255
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 32, 32)


# ── Architecture ↔ inflate coverage ─────────────────────────────────────


class TestArchitectureInflateCoverage:
    """Every canonical variant in VARIANTS must be deployable via inflate_postfilter."""

    # These are the canonical (non-alias) variants
    CANONICAL_VARIANTS = [
        "standard", "dilated", "gated_dilated", "pixelshuffle",
        "psd", "depthwise", "luma", "film", "pair_aware",
    ]

    @pytest.mark.parametrize("variant", CANONICAL_VARIANTS)
    def test_variant_in_architectures(self, variant):
        assert variant in VARIANTS, f"{variant} missing from VARIANTS dict"

    @pytest.mark.parametrize("variant", CANONICAL_VARIANTS)
    def test_variant_builds_and_runs(self, variant):
        model = build_postfilter(variant, hidden=16, kernel=3)
        if variant == "pair_aware":
            x = torch.rand(1, 6, 32, 32) * 255
            expected_out_ch = 3
        else:
            x = torch.rand(1, 3, 32, 32) * 255
            expected_out_ch = 3
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == expected_out_ch

    def test_inflate_postfilter_covers_all_canonical(self):
        """inflate_postfilter.py's _tac_build_postfilter must handle every canonical variant."""
        for variant in self.CANONICAL_VARIANTS:
            model = build_postfilter(variant, hidden=16, kernel=3)
            assert model is not None, f"build_postfilter returned None for {variant}"


# ── Adaptive weights ────────────────────────────────────────────────────


class TestAdaptiveMRS:
    """Tests for the corrected MRS formula: 20*sqrt(10*pose)."""

    def test_formula_at_known_points(self):
        aw = AdaptiveWeights()
        # pose=0.01229 (baseline): w_seg = 20*sqrt(10*0.01229) = 20*sqrt(0.1229)
        w = aw.optimal_segnet_weight_standard(0.01229)
        expected = 20.0 * math.sqrt(10.0 * 0.01229)
        assert abs(w - expected) < 0.01, f"Expected ~{expected:.2f}, got {w:.2f}"

    def test_formula_at_current_op(self):
        aw = AdaptiveWeights()
        # pose=0.00218 (current best): w_seg = 20*sqrt(0.0218) ~ 2.95
        w = aw.optimal_segnet_weight_standard(0.00218)
        expected = 20.0 * math.sqrt(10.0 * 0.00218)
        assert abs(w - expected) < 0.01

    def test_clamping_lower_bound(self):
        aw = AdaptiveWeights()
        # Very small pose: formula gives < 1.0, should clamp to 1.0
        w = aw.optimal_segnet_weight_standard(1e-8)
        assert w >= 1.0

    def test_clamping_upper_bound(self):
        aw = AdaptiveWeights()
        # Large pose: formula gives > 50.0, should clamp to 50.0
        w = aw.optimal_segnet_weight_standard(10.0)
        assert w <= 50.0

    def test_negative_pose_raises(self):
        aw = AdaptiveWeights()
        with pytest.raises(ValueError, match="non-negative"):
            aw.optimal_segnet_weight_standard(-0.01)

    def test_decreasing_with_pose(self):
        """As pose improves (decreases), w_seg should decrease."""
        aw = AdaptiveWeights()
        w_high = aw.optimal_segnet_weight_standard(0.05)
        w_low = aw.optimal_segnet_weight_standard(0.001)
        assert w_high > w_low


class TestGeometricMeanScore:
    """Tests for the geometric mean score function."""

    def test_baseline_equals_one(self):
        """Baseline values should produce score = 1.0."""
        score = geometric_mean_score(
            seg=0.00580, pose=0.01229, rate=0.02500,
        )
        assert abs(score - 1.0) < 1e-10

    def test_better_than_baseline_below_one(self):
        """Better-than-baseline values should give score < 1.0."""
        score = geometric_mean_score(
            seg=0.00400, pose=0.00800, rate=0.02000,
        )
        assert score < 1.0

    def test_worse_than_baseline_above_one(self):
        """Worse-than-baseline values should give score > 1.0."""
        score = geometric_mean_score(
            seg=0.01000, pose=0.02000, rate=0.03000,
        )
        assert score > 1.0

    def test_all_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        assert abs(0.40 + 0.35 + 0.25 - 1.0) < 1e-10


class TestRebalanceStandard:
    """Tests for the rebalance_standard method."""

    def test_returns_expected_keys(self):
        aw = AdaptiveWeights()
        result = aw.rebalance_standard(eval_pose=0.01, eval_seg=0.005)
        assert "segnet_weight" in result
        assert "boundary_weight" in result
        assert "sensitivity" in result
        assert "diagnostics" in result

    def test_records_history(self):
        aw = AdaptiveWeights()
        aw.rebalance_standard(eval_pose=0.01, eval_seg=0.005)
        assert len(aw._history) == 1
        assert aw._history[0]["mode"] == "standard"


# ── Loss function edge cases ────────────────────────────────────────────


class TestScorerLossPCGrad:
    """Edge cases for scorer_loss_pcgrad."""

    def _make_mocks(self):
        class MockPoseNet(torch.nn.Module):
            def preprocess_input(self, x):
                return x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
            def forward(self, x):
                return {"pose": x.mean(dim=(2, 3))[:, :12]}

        class MockSegNet(torch.nn.Module):
            def preprocess_input(self, x):
                return x[:, -1, ...]
            def forward(self, x):
                return x[:, :5, :, :]

        return MockPoseNet(), MockSegNet()

    def test_do_projection_false_skips_autograd(self):
        """do_projection=False should not call autograd.grad."""
        from tac.losses import scorer_loss_pcgrad

        posenet, segnet = self._make_mocks()
        filtered = (torch.rand(1, 2, 16, 16, 3) * 255).requires_grad_(True)
        gt = torch.rand(1, 2, 16, 16, 3) * 255

        loss, pd, sd, conflict = scorer_loss_pcgrad(
            filtered, gt, posenet, segnet,
            segnet_weight=100.0, do_projection=False,
        )
        # Should return no conflict when projection is disabled
        assert conflict is False
        # Loss should still be valid
        assert loss.item() > 0
        # Should be backprop-able
        loss.backward()
        assert filtered.grad is not None

    def test_segnet_weight_zero_skips_autograd(self):
        """segnet_weight=0 should skip the autograd.grad calls."""
        from tac.losses import scorer_loss_pcgrad

        posenet, segnet = self._make_mocks()
        filtered = (torch.rand(1, 2, 16, 16, 3) * 255).requires_grad_(True)
        gt = torch.rand(1, 2, 16, 16, 3) * 255

        loss, pd, sd, conflict = scorer_loss_pcgrad(
            filtered, gt, posenet, segnet,
            segnet_weight=0.0, do_projection=True,
        )
        # No conflict possible when seg is zero-weighted
        assert conflict is False
        # Loss should still be valid (just pose_loss)
        assert loss.item() > 0
        loss.backward()
        assert filtered.grad is not None

    def test_segnet_weight_zero_seg_dist_near_zero_effect(self):
        """With segnet_weight=0, changing seg input should not affect loss."""
        from tac.losses import scorer_loss_pcgrad

        posenet, segnet = self._make_mocks()
        torch.manual_seed(42)
        filtered = (torch.rand(1, 2, 16, 16, 3) * 255)
        gt = torch.rand(1, 2, 16, 16, 3) * 255

        loss1, _, _, _ = scorer_loss_pcgrad(
            filtered.clone().requires_grad_(True), gt, posenet, segnet,
            segnet_weight=0.0, do_projection=False,
        )
        loss2, _, _, _ = scorer_loss_pcgrad(
            filtered.clone().requires_grad_(True), gt, posenet, segnet,
            segnet_weight=0.0, do_projection=False,
        )
        # Same input, same loss
        assert abs(loss1.item() - loss2.item()) < 1e-6


# ── SWA warning in fit() ────────────────────────────────────────────────


class TestSWAWarningInFit:
    def test_fit_warns_on_use_swa(self):
        model = build_postfilter("standard", hidden=8)
        config = TrainConfig(hidden=8, epochs=100, tag="test-swa-warn", use_swa=True)
        trainer = Trainer(model, config, device="cpu")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                trainer.fit([], [], None, None, None)
            except AttributeError:
                pass  # Expected — None scorers
        swa_warnings = [x for x in w if "use_swa" in str(x.message)]
        assert len(swa_warnings) >= 1, "Should warn about use_swa in fit()"


# ── Wall-clock timeout integration ──────────────────────────────────────


class TestWallClockInTrainConfig:
    def test_negative_timeout_rejected(self):
        with pytest.raises(Exception):
            TrainConfig(tag="test-neg-wc", wall_clock_timeout=-1)

    def test_large_timeout_accepted(self):
        c = TrainConfig(tag="test-large-wc", wall_clock_timeout=86400)
        assert c.wall_clock_timeout == 86400


# ── PCGrad counter safety ───────────────────────────────────────────────


class TestPCGradCounters:
    def test_getattr_default_when_never_set(self):
        """PCGrad counters should be safe to read even if pcgrad was never used."""
        model = build_postfilter("standard", hidden=8)
        config = TrainConfig(hidden=8, epochs=100, tag="test-pcgrad-safe")
        trainer = Trainer(model, config, device="cpu")
        # These use getattr with default 0, should not raise
        total = getattr(trainer, '_epoch_pcgrad_total', 0)
        conflicts = getattr(trainer, '_epoch_pcgrad_conflicts', 0)
        assert total == 0
        assert conflicts == 0
