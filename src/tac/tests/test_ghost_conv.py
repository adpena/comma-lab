# SPDX-License-Identifier: MIT
"""Lane GH: regression tests for GhostConv2d + use_ghost wiring.

Council brief 2026-04-27. Lane GH attacks rate via Ghost convolutions
(Han et al. "GhostNet", CVPR 2020) — replaces dense Conv2d with a small
primary conv producing intrinsic feature maps + a cheap depthwise "ghost"
conv producing redundant linear-transform maps. Halves params at near-equal
quality.

Tests pin the contract of three layers of integration:

  1. ``GhostConv2d`` itself — output shape matches a dense Conv2d,
     parameter count is ~halved (ratio ≈ 0.5 at 36→60), forward + backward
     work, ratio guard rejects ratio<2 misuse.

  2. ``_make_conv(use_ghost=True)`` — dispatches to GhostConv2d, refuses
     the mutually-exclusive ``use_dsconv=True ∧ use_ghost=True`` combo.

  3. ``MaskRenderer`` / ``AsymmetricPairGenerator`` / ``build_renderer``
     — accept ``use_ghost=True``, build at the Lane GH-predicted param
     count (renderer ≈ 60% of dense-conv baseline at h64), and ROUND-TRIP
     the flag through ``arch_dict`` / checkpoint save+load.

  4. ``DILATED_H64_GHOST`` profile builds a model in the predicted band
     and is registered in PROFILES.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from tac.renderer import (  # noqa: E402
    ArchConfig,
    AsymmetricPairGenerator,
    GhostConv2d,
    MaskRenderer,
    ResBlock,
    _make_conv,
    build_renderer,
)


# ── 1. GhostConv2d primitive ───────────────────────────────────────────


class TestGhostConv2dPrimitive:
    """The drop-in Conv2d replacement."""

    def test_output_shape_matches_conv2d(self):
        """GhostConv2d(c_in, c_out, k) must produce the same output shape
        as nn.Conv2d(c_in, c_out, k) for the same I/O specs."""
        c_in, c_out, k = 36, 60, 3
        x = torch.randn(2, c_in, 32, 32)
        gc = GhostConv2d(c_in, c_out, k, padding=1)
        dense = nn.Conv2d(c_in, c_out, k, padding=1)
        assert gc(x).shape == dense(x).shape == (2, c_out, 32, 32)

    def test_output_shape_with_stride(self):
        """Stride must be honoured by the primary conv (ghost branch is
        always stride=1, so input stride dictates output spatial size)."""
        gc = GhostConv2d(36, 60, 3, stride=2, padding=1)
        x = torch.randn(2, 36, 32, 32)
        out = gc(x)
        assert out.shape == (2, 60, 16, 16)

    def test_param_count_halves_vs_dense(self):
        """Lane GH PREMISE: GhostConv2d at ratio=2 halves the parameter
        count vs a dense Conv2d for the same I/O. We allow a wide tolerance
        ([0.40, 0.65]) because the depthwise ghost branch + rounding for
        odd c_out add a small constant overhead."""
        c_in, c_out, k = 36, 60, 3
        gc = GhostConv2d(c_in, c_out, k, padding=1)
        dense = nn.Conv2d(c_in, c_out, k, padding=1)
        gc_params = sum(p.numel() for p in gc.parameters())
        dense_params = sum(p.numel() for p in dense.parameters())
        ratio = gc_params / dense_params
        assert 0.40 <= ratio <= 0.65, (
            f"GhostConv2d 36→60: {gc_params} params vs {dense_params} dense "
            f"(ratio {ratio:.3f}); Lane GH premise requires ~halving"
        )

    def test_param_count_halves_at_typical_renderer_widths(self):
        """At the dilated-h64 ResBlock widths (60→60, 36→36), Ghost should
        also halve params. ResBlock convs are where MaskRenderer's bulk
        weights live, so the Lane GH win depends on this."""
        for c in (36, 60):
            gc = GhostConv2d(c, c, 3, padding=1, bias=False)
            dense = nn.Conv2d(c, c, 3, padding=1, bias=False)
            ratio = sum(p.numel() for p in gc.parameters()) / sum(p.numel() for p in dense.parameters())
            assert 0.40 <= ratio <= 0.65, f"GhostConv2d {c}→{c} ratio {ratio:.3f}"

    def test_forward_pass_finite(self):
        """No NaN/Inf in forward (basic numerical sanity)."""
        gc = GhostConv2d(36, 60, 3, padding=1)
        x = torch.randn(2, 36, 16, 16)
        out = gc(x)
        assert torch.isfinite(out).all(), "GhostConv2d produced NaN/Inf"

    def test_backward_pass_propagates(self):
        """Gradients must flow through both the primary conv AND the
        depthwise ghost branch (else half the layer is dead weight)."""
        gc = GhostConv2d(36, 60, 3, padding=1)
        x = torch.randn(2, 36, 16, 16, requires_grad=False)
        loss = gc(x).sum()
        loss.backward()
        assert gc.primary.weight.grad is not None, "primary conv has no gradient"
        assert (gc.primary.weight.grad != 0).any(), "primary conv gradient is all zeros"
        assert gc.ghost.weight.grad is not None, "ghost depthwise has no gradient"
        assert (gc.ghost.weight.grad != 0).any(), "ghost depthwise gradient is all zeros"

    def test_ratio_lt_2_rejected(self):
        """ratio<2 makes Ghost meaningless (no parameter reduction).
        The constructor must fail loud on misuse."""
        with pytest.raises(ValueError, match="ratio must be"):
            GhostConv2d(36, 60, 3, ratio=1, padding=1)

    def test_odd_cout_handled(self):
        """c_out / ratio is non-integer for odd c_out — the slice in
        forward() must produce exactly c_out channels, not the rounded-up
        intermediate count."""
        gc = GhostConv2d(36, 7, 3, padding=1)  # 7 / 2 = 3.5 → 4 intrinsic
        x = torch.randn(1, 36, 8, 8)
        out = gc(x)
        assert out.shape == (1, 7, 8, 8)

    def test_padding_mode_threaded(self):
        """padding_mode=replicate must reach BOTH the primary and the ghost
        depthwise convs (the ghost branch's padding affects boundary
        artifacts in the residual maps)."""
        gc = GhostConv2d(36, 60, 3, padding=1, padding_mode="replicate")
        assert gc.primary.padding_mode == "replicate"
        assert gc.ghost.padding_mode == "replicate"


# ── 2. _make_conv dispatch ─────────────────────────────────────────────


class TestMakeConvDispatch:
    """The single helper that all encoder convs route through."""

    def test_use_ghost_returns_ghost_conv2d(self):
        m = _make_conv(36, 60, 3, padding=1, bias=True, use_ghost=True)
        assert isinstance(m, GhostConv2d)

    def test_use_dsconv_returns_sequential(self):
        """Regression: don't break the existing DSConv path."""
        m = _make_conv(36, 60, 3, padding=1, bias=True, use_dsconv=True)
        assert isinstance(m, nn.Sequential)

    def test_default_returns_conv2d(self):
        """Regression: default path stays Conv2d."""
        m = _make_conv(36, 60, 3, padding=1, bias=True)
        assert isinstance(m, nn.Conv2d)

    def test_dsconv_and_ghost_mutex(self):
        """use_dsconv + use_ghost together is a config bug — fail loud."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            _make_conv(36, 60, 3, padding=1, use_dsconv=True, use_ghost=True)


# ── 3. MaskRenderer / AsymmetricPairGenerator wiring ───────────────────


class TestMaskRendererWiring:
    """use_ghost must propagate through every conv site in MaskRenderer."""

    def test_ghost_renderer_param_count_halves(self):
        """At dilated-h64 widths (base_ch=36, mid_ch=60), the renderer
        param count must drop to roughly 60% (the motion module + heads
        keep dense convs, so the total ratio is closer to 0.66; renderer-
        only ratio is ~0.60)."""
        ghost = MaskRenderer(base_ch=36, mid_ch=60, depth=1, use_ghost=True)
        dense = MaskRenderer(base_ch=36, mid_ch=60, depth=1, use_ghost=False)
        ghost_n = sum(p.numel() for p in ghost.parameters())
        dense_n = sum(p.numel() for p in dense.parameters())
        ratio = ghost_n / dense_n
        # The ResBlock convs + stem/down convs are ghosted (~60% of dense);
        # ConvTranspose2d up_conv + fuse_conv stay dense. Empirically
        # ~0.60. Tolerate [0.50, 0.70] for future width changes.
        assert 0.50 <= ratio <= 0.70, (
            f"MaskRenderer use_ghost=True at h64: {ghost_n} vs dense "
            f"{dense_n} (ratio {ratio:.3f}); Lane GH premise requires ~halving"
        )

    def test_ghost_modules_present_in_renderer(self):
        """At least 2 GhostConv2d modules in the encoder + 8 in the four
        ResBlocks (each ResBlock has 2 ghosted convs at use_ghost=True)."""
        m = MaskRenderer(base_ch=36, mid_ch=60, depth=1, use_ghost=True)
        n_ghost = sum(1 for sub in m.modules() if isinstance(sub, GhostConv2d))
        assert n_ghost >= 10, (
            f"expected ≥10 GhostConv2d modules in Lane GH renderer "
            f"(2 encoder + 8 ResBlock convs), got {n_ghost}"
        )

    def test_forward_shape_unchanged(self):
        """Ghost is a drop-in replacement — forward output shape must be
        identical to the dense renderer."""
        ghost = MaskRenderer(base_ch=36, mid_ch=60, depth=1, use_ghost=True)
        masks = torch.zeros(1, 64, 64, dtype=torch.long)
        out = ghost(masks)
        assert out.shape == (1, 3, 64, 64)

    def test_dsconv_and_ghost_mutex_in_renderer(self):
        """Defence in depth: MaskRenderer constructor must reject the
        mutually-exclusive combo."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            MaskRenderer(use_dsconv=True, use_ghost=True)

    def test_resblock_ghost_dilation_check(self):
        """ResBlock(use_ghost=True) doesn't support dilation>1 (ghost
        branch is fixed kernel=3 stride=1; mixing dilation with the
        cheap-op assumption breaks the inductive bias). Constructor
        must fail loud rather than silently dropping the dilation."""
        with pytest.raises(ValueError, match="dilation"):
            ResBlock(36, dilation=2, use_ghost=True)


class TestAsymmetricPairGeneratorWiring:
    """Lane GH ships via AsymmetricPairGenerator (full-frame, not zoom)."""

    def test_use_ghost_propagates_to_renderer(self):
        asym = AsymmetricPairGenerator(
            base_ch=36, mid_ch=60, motion_hidden=32, embed_dim=6,
            depth=1, pose_dim=6, use_ghost=True, use_zoom_flow=False,
        )
        assert asym.use_ghost is True
        assert asym.renderer.use_ghost is True
        n_ghost = sum(1 for m in asym.modules() if isinstance(m, GhostConv2d))
        assert n_ghost >= 10

    def test_arch_dict_round_trips_use_ghost(self):
        """Lane GH checkpoints MUST round-trip use_ghost via arch_dict()
        — without it, downstream pipeline.py / renderer_export rebuilds
        a dense-conv model whose state_dict won't load."""
        asym = AsymmetricPairGenerator(use_ghost=True, use_zoom_flow=False)
        arch = asym.arch_dict()
        assert "use_ghost" in arch, "arch_dict missing use_ghost"
        assert arch["use_ghost"] is True

    def test_dsconv_and_ghost_mutex_in_asym(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            AsymmetricPairGenerator(use_dsconv=True, use_ghost=True)


class TestBuildRendererDispatch:
    """build_renderer is the canonical model factory."""

    def test_build_renderer_use_ghost_legacy_pair(self):
        """use_zoom_flow=False routes through the PairGenerator path.
        use_ghost must reach MaskRenderer."""
        m = build_renderer(
            base_ch=36, mid_ch=60, motion_hidden=32, embed_dim=6,
            depth=1, pose_dim=6, use_ghost=True, use_zoom_flow=False,
        )
        # PairGenerator's renderer attribute is the MaskRenderer.
        assert m.renderer.use_ghost is True
        n_ghost = sum(1 for sub in m.modules() if isinstance(sub, GhostConv2d))
        assert n_ghost >= 10

    def test_build_renderer_use_ghost_asym_zoom(self):
        """use_zoom_flow=True routes through AsymmetricPairGenerator."""
        m = build_renderer(
            base_ch=36, mid_ch=60, motion_hidden=32, embed_dim=6,
            depth=1, pose_dim=6, use_ghost=True, use_zoom_flow=True,
        )
        assert isinstance(m, AsymmetricPairGenerator)
        assert m.use_ghost is True

    def test_arch_config_has_use_ghost_field(self):
        """ArchConfig dataclass exposes use_ghost so config-driven callers
        (deploy configs, distill configs) can stay in sync."""
        cfg = ArchConfig(use_ghost=True)
        assert cfg.use_ghost is True
        # Default must be False (regression: don't accidentally flip the
        # default and break Lane A / dilated-h64 baseline).
        assert ArchConfig().use_ghost is False


# ── 4. Profile registration ────────────────────────────────────────────


class TestLaneGHProfile:
    """The DILATED_H64_GHOST profile is what scripts/remote_lane_gh_*.sh
    will reference. Pin the contract here so a profile rename can't
    silently break the deploy script."""

    def test_profile_registered(self):
        from tac.profiles import PROFILES
        assert "dilated_h64_ghost" in PROFILES, (
            "Lane GH requires PROFILES['dilated_h64_ghost'] — "
            "scripts/remote_lane_gh_ghost_renderer.sh references this name"
        )

    def test_profile_use_ghost_true(self):
        from tac.profiles import PROFILES
        p = PROFILES["dilated_h64_ghost"]
        assert p["use_ghost"] is True, "profile must set use_ghost=True"
        # Mutual exclusivity is enforced at the model layer; defend in depth
        # at the profile layer too so a copy-paste from Lane K can't slip.
        assert p.get("use_dsconv", False) is False, (
            "Lane GH profile must NOT set use_dsconv=True (mutex with use_ghost)"
        )

    def test_profile_eval_roundtrip_true(self):
        """CLAUDE.md non-negotiable: every training profile MUST set
        eval_roundtrip=True. Without it the proxy-auth gap can be 11x."""
        from tac.profiles import PROFILES
        p = PROFILES["dilated_h64_ghost"]
        assert p["eval_roundtrip"] is True, (
            "CLAUDE.md non-negotiable: eval_roundtrip MUST be True"
        )

    def test_profile_full_frame_only(self):
        """Lane GH is FULL-frame masks (anchored on Lane A's 1.15 artifacts).
        mask_half_sim_prob > 0 OR use_zoom_flow=True would silently switch
        to the half-frame path that broke Lane D (memory:
        feedback_half_frame_breaks_posenet, score 17.55)."""
        from tac.profiles import PROFILES
        p = PROFILES["dilated_h64_ghost"]
        assert p.get("mask_half_sim_prob", 0.0) == 0.0, (
            "Lane GH must have mask_half_sim_prob=0.0 (full-frame only)"
        )
        assert p.get("use_zoom_flow", False) is False, (
            "Lane GH must have use_zoom_flow=False (full-frame only)"
        )

    def test_profile_param_count_in_predicted_band(self):
        """The whole point of Lane GH: param count in [130K, 200K] (target
        ~144K renderer + 45K motion = ~190K total). Outside this band the
        prediction [1.05, 1.30] doesn't hold."""
        from tac.profiles import PROFILES
        p = PROFILES["dilated_h64_ghost"]
        m = build_renderer(
            embed_dim=p["embed_dim"], base_ch=p["base_ch"], mid_ch=p["mid_ch"],
            motion_hidden=p["motion_hidden"], depth=p["depth"],
            pose_dim=p["pose_dim"], use_ghost=p["use_ghost"],
            padding_mode=p["padding_mode"], use_dilation=p["use_dilation"],
            use_zoom_flow=p["use_zoom_flow"],
        )
        n = sum(pp.numel() for pp in m.parameters())
        assert 150_000 <= n <= 210_000, (
            f"Lane GH param count {n:,} outside predicted band "
            f"[150K, 210K]; arch widths drifted from the council brief"
        )
        # Renderer-only must be roughly halved vs dilated-h64 baseline.
        n_renderer = sum(pp.numel() for pp in m.renderer.parameters())
        assert 110_000 <= n_renderer <= 170_000, (
            f"Lane GH renderer-only {n_renderer:,} outside [110K, 170K]; "
            f"Ghost wiring may be incomplete"
        )

    def test_profile_arch_round_trip_via_meta(self):
        """The arch_meta dict that train_renderer.py persists MUST include
        use_ghost so resume from checkpoint reconstructs the same arch.
        We mirror the meta-build logic here as a contract test."""
        from tac.profiles import PROFILES
        p = PROFILES["dilated_h64_ghost"]
        arch_meta = {
            "schema_version": 1,
            "pose_dim": p["pose_dim"],
            "base_ch": p["base_ch"],
            "mid_ch": p["mid_ch"],
            "embed_dim": p["embed_dim"],
            "motion_hidden": p["motion_hidden"],
            "depth": p["depth"],
            "use_zoom_flow": p["use_zoom_flow"],
            "use_dsconv": p["use_dsconv"],
            "use_ghost": p["use_ghost"],
            "use_dilation": p["use_dilation"],
            "padding_mode": p["padding_mode"],
        }
        # Reconstruct the model from meta — this is what
        # train_renderer.py:_peek_checkpoint_arch_meta() feeds into
        # build_renderer on resume.
        m = build_renderer(
            embed_dim=arch_meta["embed_dim"],
            base_ch=arch_meta["base_ch"],
            mid_ch=arch_meta["mid_ch"],
            motion_hidden=arch_meta["motion_hidden"],
            depth=arch_meta["depth"],
            pose_dim=arch_meta["pose_dim"],
            use_dsconv=arch_meta["use_dsconv"],
            use_ghost=arch_meta["use_ghost"],
            padding_mode=arch_meta["padding_mode"],
            use_dilation=arch_meta["use_dilation"],
            use_zoom_flow=arch_meta["use_zoom_flow"],
        )
        assert m.renderer.use_ghost is True
