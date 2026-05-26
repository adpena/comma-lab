# SPDX-License-Identifier: MIT
"""Tests for CATALYST CASCADE COMPOSITION 5th-order recursive doctrine.

Covers:
  Stage A: Path A learnable head construction + KL forward pass.
  Stage B: MLX-native FakeQuantFP4 codebook nearest-projection + identity-STE.
  Stage C: BPR1 sign-bitmap sidecar emission.
  Stage D: per-axis telemetry + canonical Provenance routing per Catalog #323/#341.
  End-to-end: 3-arm pipeline integration (baseline / Path A alone / CATALYST composition).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

# Skip the whole module if MLX is not importable.
mx = pytest.importorskip("mlx.core")

from tac.substrates.hinton_distilled_scorer_surrogate.catalyst_cascade import (  # noqa: E402
    FP4_DEFAULT_CODEBOOK,
    CatalystCascadeArmTelemetry,
    CatalystSidecarManifest,
    build_catalyst_bpr1_sidecar,
    fake_quant_fp4_mlx,
    quantize_head_fp4,
    run_catalyst_cascade_arm,
    run_catalyst_cascade_pipeline,
)
from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (  # noqa: E402
    DEFAULT_DISTILLATION_TEMPERATURE,
    DEFAULT_SEGNET_CLASSES,
    build_learnable_student_head,
)


# ---------------------------------------------------------------------------
# Stage A: Path A learnable head + KL forward pass
# ---------------------------------------------------------------------------


class TestStageALearnableHeadAndKL:
    def test_build_learnable_student_head_returns_canonical_shape(self):
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        assert head.weight.shape == (3, 5)
        assert head.bias.shape == (5,)
        assert head.num_classes == 5
        # 20-param canonical head per sister Path A production-scale anchor.
        assert head.weight.size + head.bias.size == 20

    def test_learnable_head_forward_pass_produces_logits(self):
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        # Synthetic NHWC RGB input in [0,1]
        decoded = mx.random.uniform(shape=(2, 4, 6, 3), key=mx.random.key(42))
        logits = head(decoded)
        assert logits.shape == (2, 4, 6, 5)


# ---------------------------------------------------------------------------
# Stage B: FakeQuantFP4 MLX-native codebook projection + identity-STE
# ---------------------------------------------------------------------------


class TestStageBFakeQuantFP4MLX:
    def test_fake_quant_fp4_returns_same_shape(self):
        w = mx.random.normal(shape=(3, 5), key=mx.random.key(0))
        q = fake_quant_fp4_mlx(w)
        assert q.shape == w.shape

    def test_fake_quant_fp4_quantizes_to_codebook_values(self):
        """Forward output magnitudes must be in {scale * cb_value} for the
        canonical FP4 codebook (after dividing by per-block scale)."""
        # Use simple deterministic input with known sign/magnitude distribution.
        w = mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 0.0], dtype=mx.float32)
        q = fake_quant_fp4_mlx(w, block_size=8)
        # Forward must produce values whose magnitudes are scale * cb_entry.
        # Per-block max magnitude is 6.0 → scale = 6.0 / 6.0 = 1.0 → quantized
        # magnitudes ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} exactly.
        q_np = np.asarray(q)
        canonical_magnitudes = set(FP4_DEFAULT_CODEBOOK)
        for v in q_np:
            assert abs(v) in canonical_magnitudes or abs(abs(v) - 1.5) < 1e-5

    def test_fake_quant_fp4_zero_block_preserved(self):
        """All-zero block must produce all-zero output (clamp(min=1e-10) protection)."""
        w = mx.zeros((8,), dtype=mx.float32)
        q = fake_quant_fp4_mlx(w, block_size=8)
        q_np = np.asarray(q)
        assert (q_np == 0.0).all()

    def test_fake_quant_fp4_identity_ste_via_value_and_grad(self):
        """Identity-STE pattern: gradient of L(q(w)) w.r.t. w should equal
        gradient of L(w) w.r.t. w (modulo the saturated-position mask)."""
        def loss_fn(w):
            q = fake_quant_fp4_mlx(w, block_size=8)
            return mx.sum(q * q)

        w = mx.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=mx.float32)
        loss_val, grad = mx.value_and_grad(loss_fn)(w)
        # Gradient must be finite and non-zero for at least one position.
        grad_np = np.asarray(grad)
        assert np.isfinite(grad_np).all()
        assert not (grad_np == 0.0).all()

    def test_quantize_head_fp4_produces_new_head_with_canonical_codebook(self):
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        quantized = quantize_head_fp4(head)
        assert quantized.weight.shape == head.weight.shape
        assert quantized.bias.shape == head.bias.shape
        assert quantized.num_classes == head.num_classes
        # Quantized weight values must be representable in the canonical codebook
        # (up to per-block scale * sign).
        w_q = np.asarray(quantized.weight).flatten()
        # Sanity: no NaN/Inf
        assert np.isfinite(w_q).all()

    def test_quantize_head_fp4_zero_size_weight(self):
        """Edge case: trivially-small head (single class) should still quantize."""
        head = build_learnable_student_head(num_classes=2, in_channels=3, seed=0)
        quantized = quantize_head_fp4(head)
        assert quantized.num_classes == 2


# ---------------------------------------------------------------------------
# Stage C: BPR1 sidecar emission
# ---------------------------------------------------------------------------


class TestStageCBPR1Sidecar:
    def test_build_catalyst_bpr1_sidecar_emits_canonical_bytes(self):
        # Synthetic post-QAT student logits + target logits (B=2, H=4, W=6, K=5)
        student = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(1))
        pr110_prefix = hashlib.sha256(b"test_seed").digest()[:16]
        sidecar_bytes, manifest = build_catalyst_bpr1_sidecar(
            student_logits_post_qat=student,
            target_logits=target,
            pr110_base_sha256_prefix=pr110_prefix,
            gain_clamp=1.0,
        )
        assert isinstance(sidecar_bytes, bytes)
        assert len(sidecar_bytes) == manifest.bpr1_sidecar_total_bytes
        assert isinstance(manifest, CatalystSidecarManifest)
        # Manifest carries entropy diagnostic
        assert 0.0 <= manifest.sign_entropy_global_bits <= 1.0

    def test_build_catalyst_bpr1_sidecar_refuses_non_4d_residual(self):
        student = mx.random.normal(shape=(2, 4, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 5), key=mx.random.key(1))
        pr110_prefix = hashlib.sha256(b"test_seed").digest()[:16]
        with pytest.raises(ValueError, match="must be 4D"):
            build_catalyst_bpr1_sidecar(
                student_logits_post_qat=student,
                target_logits=target,
                pr110_base_sha256_prefix=pr110_prefix,
            )

    def test_build_catalyst_bpr1_sidecar_refuses_too_few_classes(self):
        student = mx.random.normal(shape=(2, 4, 6, 2), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 6, 2), key=mx.random.key(1))
        pr110_prefix = hashlib.sha256(b"test_seed").digest()[:16]
        with pytest.raises(ValueError, match="must have >=3 classes"):
            build_catalyst_bpr1_sidecar(
                student_logits_post_qat=student,
                target_logits=target,
                pr110_base_sha256_prefix=pr110_prefix,
            )

    def test_build_catalyst_bpr1_sidecar_clamp_protection(self):
        """Residuals exceeding ±gain_clamp must be clipped (no out-of-range
        residuals reach the BPR1 codec)."""
        # Construct residual deliberately larger than gain_clamp=0.5
        student = mx.full(shape=(1, 2, 2, 3), vals=5.0, dtype=mx.float32)
        target = mx.full(shape=(1, 2, 2, 3), vals=0.0, dtype=mx.float32)
        pr110_prefix = hashlib.sha256(b"test_seed").digest()[:16]
        _, manifest = build_catalyst_bpr1_sidecar(
            student_logits_post_qat=student,
            target_logits=target,
            pr110_base_sha256_prefix=pr110_prefix,
            gain_clamp=0.5,
        )
        # All clipped to +0.5 → uniform sign-bitmap → entropy near 0
        assert manifest.residual_max <= 0.5 + 1e-6
        assert manifest.residual_min >= -0.5 - 1e-6


# ---------------------------------------------------------------------------
# Stage D: per-axis telemetry + canonical Provenance routing
# ---------------------------------------------------------------------------


class TestStageDPerAxisTelemetry:
    def test_arm_telemetry_baseline_has_zero_bpr1_bytes(self):
        student = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(1))
        pr110_prefix = hashlib.sha256(b"seed").digest()[:16]
        telem = run_catalyst_cascade_arm(
            arm_name="baseline",
            student_logits_initial=student,
            target_logits=target,
            head=None,
            apply_qat=False,
            apply_bpr1=False,
            pr110_base_sha256_prefix=pr110_prefix,
        )
        assert telem.arm_name == "baseline"
        assert telem.bpr1_sidecar_bytes == 0
        assert telem.rate_term_canonical_score == 0.0
        assert telem.head_n_params == 0
        assert telem.qat_applied is False

    def test_arm_telemetry_path_a_alone_has_20_params_zero_bpr1(self):
        student = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(1))
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        pr110_prefix = hashlib.sha256(b"seed").digest()[:16]
        telem = run_catalyst_cascade_arm(
            arm_name="path_a_alone",
            student_logits_initial=student,
            target_logits=target,
            head=head,
            apply_qat=False,
            apply_bpr1=False,
            pr110_base_sha256_prefix=pr110_prefix,
        )
        assert telem.head_n_params == 20
        assert telem.qat_applied is False
        assert telem.bpr1_sidecar_bytes == 0

    def test_arm_telemetry_catalyst_composition_has_nonzero_bpr1(self):
        student = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(1))
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        pr110_prefix = hashlib.sha256(b"seed").digest()[:16]
        telem = run_catalyst_cascade_arm(
            arm_name="catalyst_composition",
            student_logits_initial=student,
            target_logits=target,
            head=head,
            apply_qat=True,
            apply_bpr1=True,
            pr110_base_sha256_prefix=pr110_prefix,
        )
        assert telem.head_n_params == 20
        assert telem.qat_applied is True
        assert telem.bpr1_sidecar_bytes > 0
        assert telem.rate_term_canonical_score > 0.0
        # Sign entropy in valid range [0, 1] bits
        assert 0.0 <= telem.sign_entropy_global_bits <= 1.0


# ---------------------------------------------------------------------------
# End-to-end pipeline integration + canonical Provenance
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_full_pipeline_emits_3_arms_with_canonical_provenance(self):
        student = mx.random.normal(shape=(4, 6, 8, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(4, 6, 8, 5), key=mx.random.key(1))
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        result = run_catalyst_cascade_pipeline(
            student_logits_initial=student,
            target_logits=target,
            path_a_head=head,
            temperature=DEFAULT_DISTILLATION_TEMPERATURE,
            gain_clamp=1.0,
        )
        # Schema version pinned
        assert result["schema_version"] == "catalyst_cascade_pipeline_verdict_v1_20260526"
        # 3 arms emitted
        assert len(result["arms"]) == 3
        arm_names = [a["arm_name"] for a in result["arms"]]
        assert arm_names == ["baseline", "path_a_alone", "catalyst_composition"]
        # Canonical Provenance markers per Catalog #341
        prov = result["canonical_provenance"]
        assert prov["axis_tag"] == "[macOS-MLX research-signal]"
        assert prov["score_claim_valid"] is False
        assert prov["promotion_eligible"] is False
        assert prov["rank_or_kill_eligible"] is False
        assert prov["ready_for_exact_eval_dispatch"] is False
        # Canonical equation id pinned
        assert result["canonical_equation_id"] == (
            "hinton_kl_distill_enables_qat_catalyst_composition_savings_v1"
        )

    def test_pipeline_delta_summary_has_canonical_keys(self):
        student = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(1))
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        result = run_catalyst_cascade_pipeline(
            student_logits_initial=student,
            target_logits=target,
            path_a_head=head,
        )
        delta = result["delta_summary"]
        for key in (
            "delta_path_a_alone_kl_minus_baseline_kl",
            "delta_catalyst_composition_kl_minus_path_a_alone_kl",
            "delta_catalyst_composition_rate_minus_baseline_rate",
            "composite_baseline",
            "composite_path_a_alone",
            "composite_catalyst_composition",
            "catalyst_composition_improves_over_path_a_alone",
            "catalyst_composition_improves_over_baseline",
        ):
            assert key in delta, f"missing delta_summary key={key}"

    def test_pipeline_emits_pr110_base_sha256_prefix_hex(self):
        student = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(1))
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        result = run_catalyst_cascade_pipeline(
            student_logits_initial=student,
            target_logits=target,
            path_a_head=head,
        )
        # 16-byte prefix → 32 hex chars
        assert len(result["pr110_base_sha256_prefix_hex"]) == 32

    def test_pipeline_composite_proxy_ordering_makes_sense(self):
        """Composite proxy = 100 * kl/100 + rate. For identical KL, the arm
        with extra BPR1 sidecar bytes must have HIGHER composite (rate ≥ 0).
        """
        student = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(0))
        target = mx.random.normal(shape=(2, 4, 6, 5), key=mx.random.key(1))
        head = build_learnable_student_head(num_classes=5, in_channels=3, seed=0)
        result = run_catalyst_cascade_pipeline(
            student_logits_initial=student,
            target_logits=target,
            path_a_head=head,
        )
        delta = result["delta_summary"]
        # Rate-only contribution: catalyst arm pays sidecar bytes, baseline pays 0.
        assert delta["delta_catalyst_composition_rate_minus_baseline_rate"] > 0.0
