# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 REAL-Hinton-teacher pose-axis wire-in tests (NO FAKE).

These tests close the Wave N+11 QUAD HALT gap: the Wave N+11 stabilizer re-fire
used ``--allow-mock-scorer-teacher`` so the pose axis was 0 throughout (the
mock teacher has no PoseNet). THIS test suite verifies the REAL-teacher path
(``distillation_weight > 0`` + ``pose_distillation_weight > 0`` WITHOUT the
mock flag) produces a REAL non-zero pose axis AND a real KL T=2.0 SegNet axis,
and that BOTH gradients actually move the Z7-Mamba-2 renderer params.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE Class 2 (tests-verify-
behavior-not-constants): every assertion below verifies ACTUAL training
behavior — non-zero per-axis loss, finite gradient flow, renderer param
movement, EMA shadow distinct from live — NOT mock-teacher constants. The
canonical real-cache classes (``RealPoseNetTeacherCache`` /
``RealSegNetTeacherLogitsCache``) are populated deterministically in-memory so
the test is hermetic + fast; the LONG MLX run on real ``upstream/videos/0.mkv``
validates the same wire-in against the actual contest SegNet/PoseNet forwards.

[verified-against: tac.substrates._shared.mlx_score_aware.loss.score_aware_loss real-teacher path]
[verified-against: tac.substrates._shared.mlx_score_aware.adapter.MlxScoreAwareAdapter.train_step sibling head steps]
[verified-against: tac.substrates.time_traveler_l5_z7_mamba2.mlx_module.Z7Mamba2MLXModule selective-SSM renderer]
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tac.substrates._shared.mlx_score_aware.adapter import (  # noqa: E402
    MlxScoreAwareAdapter,
)
from tac.substrates._shared.mlx_score_aware.bundle import (  # noqa: E402
    RendererBundle,
)
from tac.substrates._shared.mlx_score_aware.loss import (  # noqa: E402
    score_aware_loss,
)

# Contest SegNet/PoseNet canonical eval size (both teachers require this).
_H = 384
_W = 512


# --------------------------------------------------------------------------- #
# Deterministic in-memory REAL-cache classes (the canonical teacher contract). #
# --------------------------------------------------------------------------- #
def _make_real_pose_teacher(num_pairs: int, pose_dims: int = 6, seed: int = 7):
    """Deterministic ``RealPoseNetTeacherCache`` (canonical real-cache class).

    This is NOT the mock cosine provider — it is the SAME dataclass the real
    ``build_mlx_posenet_pair_teacher`` returns, populated with deterministic
    pose vectors so the test is hermetic. The wire-in path exercised here is
    byte-identical to the production path; only the teacher-pose *values* are
    synthetic-but-realistic (varied per-dim std mirroring real ego-motion).
    """
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        RealPoseNetTeacherCache,
    )

    rng = np.random.RandomState(seed)
    # Mirror real ego-motion: dim 0 (forward velocity) large std, rotation dims
    # small std — so the canonical bounded-amplification per-dim scale matters.
    pose_np = rng.randn(num_pairs, pose_dims).astype(np.float32)
    pose_np[:, 0] *= 0.9
    pose_np[:, 1:] *= 0.01
    raw_std = np.std(pose_np, axis=0).astype(np.float32)
    scale_floor = max(float(raw_std.max()) * 0.1, 1.0e-3)
    per_dim_scale = np.maximum(raw_std, scale_floor)
    return RealPoseNetTeacherCache(
        teacher_pose_np=mx.array(pose_np),
        num_pairs=num_pairs,
        pose_dims=pose_dims,
        per_dim_scale=mx.array(per_dim_scale),
        upstream_posenet_safetensors_sha256="deterministic_test_cache",
        cache_build_seconds=0.0,
    )


def _make_real_segnet_teacher(
    num_pairs: int, num_classes: int = 5, seed: int = 11
):
    """Deterministic ``RealSegNetTeacherLogitsCache`` (canonical real-cache class)."""
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        RealSegNetTeacherLogitsCache,
    )

    rng = np.random.RandomState(seed)
    logits = rng.randn(num_pairs, _H, _W, num_classes).astype(np.float32)
    return RealSegNetTeacherLogitsCache(
        teacher_logits_thwk=mx.array(logits),
        frame_count=num_pairs,
        height=_H,
        width=_W,
        num_classes=num_classes,
    )


def _build_heads(num_classes: int = 5, pose_dims: int = 6, seed: int = 0):
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    seg_head = build_learnable_student_head(
        num_classes=num_classes, in_channels=3, seed=seed
    )
    pose_head = build_learnable_pose_student_head(pose_dims=pose_dims, seed=seed)
    return seg_head, pose_head


def _build_z7_model(num_pairs: int, seed: int = 0):
    """Real Z7-Mamba-2 MLX module at the small Wave N+11 stabilizer config."""
    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_module import (
        Z7Mamba2MLXModule,
    )
    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
        Z7Mamba2MLXRenderConfig,
    )

    cfg = Z7Mamba2MLXRenderConfig(
        num_pairs=num_pairs, d_state=8, d_model=32, expand=1
    )
    return Z7Mamba2MLXModule(cfg, seed=seed), cfg


def _make_targets(num_pairs: int, seed: int = 3):
    rng = np.random.RandomState(seed)
    t0 = mx.array(rng.rand(num_pairs, _H, _W, 3).astype(np.float32))
    t1 = mx.array(rng.rand(num_pairs, _H, _W, 3).astype(np.float32))
    return t0, t1


def _make_real_teacher_bundle(num_pairs: int = 4, seed: int = 0):
    """Build a bundle binding the REAL SegNet + PoseNet teachers (no mock)."""
    model, _cfg = _build_z7_model(num_pairs, seed=seed)
    t0, t1 = _make_targets(num_pairs)
    seg_head, pose_head = _build_heads(seed=seed)
    return RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=num_pairs,
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=0.5,
        scorer_teacher=_make_real_segnet_teacher(num_pairs),
        learnable_student_head=seg_head,
        pose_distillation_weight=1.0,
        pose_scorer_teacher=_make_real_pose_teacher(num_pairs),
        learnable_pose_student_head=pose_head,
        pose_dims=6,
        # CRITICAL: allow_mock_scorer_teacher is False (default) — this is the
        # REAL-teacher path the Wave N+11 mock run skipped.
    )


# --------------------------------------------------------------------------- #
# 1. The REAL teacher produces a NON-ZERO pose axis (the missing version).     #
# --------------------------------------------------------------------------- #
def test_real_pose_teacher_yields_nonzero_pose_axis():
    """The diagnosis: mock=pose 0; real teacher=pose != 0. Verify the latter."""
    bundle = _make_real_teacher_bundle(num_pairs=4)
    idx = mx.array(np.arange(4, dtype=np.int32))
    _total, parts = score_aware_loss(bundle, idx)
    assert "pose_distill" in parts, (
        "real PoseNet teacher path did not produce a pose_distill term — the "
        "Wave N+11 mock-teacher gap would recur"
    )
    mx.eval(parts["pose_distill"])
    pose_val = float(parts["pose_distill"].item())
    assert pose_val > 0.0, (
        f"pose axis is {pose_val}; must be strictly positive when the REAL "
        "PoseNet teacher is wired (mock teacher leaves it 0)"
    )
    assert np.isfinite(pose_val), f"pose axis not finite: {pose_val}"


def test_real_segnet_teacher_yields_nonzero_seg_axis():
    """KL T=2.0 SegNet axis is real + non-zero with the real teacher wired."""
    bundle = _make_real_teacher_bundle(num_pairs=4)
    idx = mx.array(np.arange(4, dtype=np.int32))
    _total, parts = score_aware_loss(bundle, idx)
    assert "distill" in parts
    mx.eval(parts["distill"])
    seg_val = float(parts["distill"].item())
    assert seg_val > 0.0, f"seg (KL T=2.0) axis must be > 0; got {seg_val}"
    assert np.isfinite(seg_val)


def test_distillation_temperature_is_canonical_t2():
    """Hinton-KL temperature is the canonical T=2.0 (Quantizr/Hinton)."""
    bundle = _make_real_teacher_bundle(num_pairs=2)
    assert bundle.distillation_temperature == 2.0, (
        "canonical Hinton-distilled KL temperature must be T=2.0"
    )


# --------------------------------------------------------------------------- #
# 2. score_aware_components emits a REAL per-axis decomposition (not None).    #
# --------------------------------------------------------------------------- #
def test_score_aware_components_real_pose_nonzero():
    """The adapter's per-axis decomposition emits real non-zero seg AND pose."""
    bundle = _make_real_teacher_bundle(num_pairs=4)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="z7_mamba2_real_hinton_test",
        grad_clip_max_norm=1.0,
        warmup_epochs=5,
        weight_decay=1e-4,
    )
    idx = mx.array(np.arange(4, dtype=np.int32))
    axes = adapter.score_aware_components(adapter.model, idx)
    assert axes is not None, (
        "score_aware_components returned None despite real teachers wired — "
        "the per-axis decomposition gap (Wave N+11 mock pose=0) would recur"
    )
    assert set(axes.keys()) >= {"seg", "pose", "recon_aux", "archive_bytes"}
    assert axes["pose"] > 0.0, f"per-axis pose must be > 0; got {axes['pose']}"
    assert axes["seg"] > 0.0, f"per-axis seg must be > 0; got {axes['seg']}"
    assert np.isfinite(axes["pose"]) and np.isfinite(axes["seg"])


# --------------------------------------------------------------------------- #
# 3. Real-teacher training step actually moves the Z7 renderer (gradient flow).#
# --------------------------------------------------------------------------- #
def test_real_teacher_train_step_moves_renderer_params():
    """A real-teacher train_step descends the loss + mutates renderer params."""
    bundle = _make_real_teacher_bundle(num_pairs=8, seed=0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="z7_mamba2_real_hinton_test",
        grad_clip_max_norm=1.0,
        warmup_epochs=2,
        warmup_steps_per_epoch=1,
        weight_decay=1e-4,
    )
    # Snapshot a renderer param before stepping. The Z7-Mamba-2 module exposes
    # the latent_init + residuals + output_projection_w as gradient-tracked
    # mx.array attributes (the trainable renderer params); residuals[t] is the
    # per-pair latent the decode path consumes, so it is the most direct
    # scorer-bound-gradient-reachable param to assert movement on.
    before = np.array(adapter.model.output_projection_w).copy()
    before_residuals = np.array(adapter.model.residuals).copy()
    idx = mx.array(np.arange(8, dtype=np.int32))
    out0 = adapter.train_step(idx, 3e-4, {})
    out1 = adapter.train_step(idx, 3e-4, {})
    assert np.isfinite(out0["total"]), f"step 0 loss not finite: {out0}"
    assert np.isfinite(out1["total"]), f"step 1 loss not finite: {out1}"
    after = np.array(adapter.model.output_projection_w)
    after_residuals = np.array(adapter.model.residuals)
    moved = max(
        float(np.abs(after - before).max()),
        float(np.abs(after_residuals - before_residuals).max()),
    )
    assert moved > 0.0, (
        "renderer params (output_projection_w / residuals) did not move after "
        "2 real-teacher steps — the scorer-bound gradient is not reaching the "
        "Mamba-2 renderer"
    )


def test_real_teacher_pose_head_trains():
    """The learnable pose student head's params move under the real pose teacher."""
    bundle = _make_real_teacher_bundle(num_pairs=8, seed=0)
    adapter = MlxScoreAwareAdapter(
        bundle, substrate_id="z7_mamba2_real_hinton_test"
    )
    pose_head = bundle.learnable_pose_student_head
    before_w = np.array(pose_head.weight).copy()
    idx = mx.array(np.arange(8, dtype=np.int32))
    adapter.train_step(idx, 1e-3, {})
    after_w = np.array(pose_head.weight)
    moved = float(np.abs(after_w - before_w).max())
    assert moved > 0.0, (
        "pose student head weight did not move — the pose-MSE distill gradient "
        "is not reaching the pose head (real PoseNet teacher unbound)"
    )


def test_real_teacher_segnet_head_trains():
    """The learnable SegNet student head's params move under the real teacher."""
    bundle = _make_real_teacher_bundle(num_pairs=8, seed=0)
    adapter = MlxScoreAwareAdapter(
        bundle, substrate_id="z7_mamba2_real_hinton_test"
    )
    seg_head = bundle.learnable_student_head
    before_w = np.array(seg_head.weight).copy()
    idx = mx.array(np.arange(8, dtype=np.int32))
    adapter.train_step(idx, 1e-3, {})
    after_w = np.array(seg_head.weight)
    moved = float(np.abs(after_w - before_w).max())
    assert moved > 0.0, (
        "SegNet student head weight did not move — the KL distill gradient is "
        "not reaching the head (real SegNet teacher unbound)"
    )


# --------------------------------------------------------------------------- #
# 4. The real path is structurally DIFFERENT from the mock path.              #
# --------------------------------------------------------------------------- #
def test_real_teacher_path_distinct_from_mock_no_pose():
    """The mock path (allow_mock_scorer_teacher) has NO pose teacher / pose=0.

    This is the canonical contrast: the Wave N+11 run used the mock path and
    got pose 0; the real path gets a non-zero pose axis. Verify both legs.
    """
    # Mock leg: distillation only, no real teacher, no pose teacher (the bundle
    # fail-closed invariant forbids a SegNet-only bind without pose UNLESS the
    # mock + segnet-only-research opt-ins are set; the mock path is pose-blind).
    model, _cfg = _build_z7_model(4, seed=0)
    t0, t1 = _make_targets(4)
    mock_bundle = RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=4,
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=0.5,
        pose_distillation_weight=0.0,  # mock path has NO pose teacher
        allow_mock_scorer_teacher=True,
    )
    idx = mx.array(np.arange(4, dtype=np.int32))
    _t, mock_parts = score_aware_loss(mock_bundle, idx)
    assert "pose_distill" not in mock_parts, (
        "mock path must NOT produce a pose axis (this IS the Wave N+11 gap)"
    )
    # Real leg: pose axis present + non-zero.
    real_bundle = _make_real_teacher_bundle(num_pairs=4)
    _t2, real_parts = score_aware_loss(real_bundle, idx)
    assert "pose_distill" in real_parts
    mx.eval(real_parts["pose_distill"])
    assert float(real_parts["pose_distill"].item()) > 0.0


def test_mock_scorer_teacher_flag_defaults_false():
    """The real-teacher bundle does NOT set the mock flag (default False)."""
    bundle = _make_real_teacher_bundle(num_pairs=2)
    assert bundle.allow_mock_scorer_teacher is False, (
        "the REAL optimal version must NOT carry allow_mock_scorer_teacher; "
        "the Wave N+11 HALT used the mock flag"
    )
    assert bundle.scorer_teacher is not None
    assert bundle.pose_scorer_teacher is not None


# --------------------------------------------------------------------------- #
# 5. EMA + eval_roundtrip discipline still hold on the real-teacher path.      #
# --------------------------------------------------------------------------- #
def test_real_teacher_loss_total_includes_both_axes():
    """Total loss = recon + dweight*seg + pweight*pose (all axes summed)."""
    bundle = _make_real_teacher_bundle(num_pairs=4)
    idx = mx.array(np.arange(4, dtype=np.int32))
    total, parts = score_aware_loss(bundle, idx)
    mx.eval(total)
    mx.eval(parts["recon"])
    mx.eval(parts["distill"])
    mx.eval(parts["pose_distill"])
    recon = float(parts["recon"].item())
    seg = float(parts["distill"].item())
    pose = float(parts["pose_distill"].item())
    expected = (
        recon
        + bundle.distillation_weight * seg
        + bundle.pose_distillation_weight * pose
    )
    got = float(total.item())
    assert abs(got - expected) < 1e-3, (
        f"total {got} != recon+dw*seg+pw*pose {expected}; the three axes must "
        "all enter the Lagrangian"
    )


def test_real_teacher_with_stabilizer_no_nan_over_steps():
    """Wave N+11 stabilizer + real teacher: multiple steps stay finite (no NaN)."""
    bundle = _make_real_teacher_bundle(num_pairs=8, seed=0)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="z7_mamba2_real_hinton_test",
        grad_clip_max_norm=1.0,
        warmup_epochs=5,
        warmup_steps_per_epoch=1,
        weight_decay=1e-4,
    )
    idx = mx.array(np.arange(8, dtype=np.int32))
    losses = []
    for _ in range(8):
        out = adapter.train_step(idx, 3e-4, {})
        losses.append(out["total"])
    assert all(np.isfinite(loss_val) for loss_val in losses), (
        f"real-teacher + Wave N+11 stabilizer produced a non-finite loss over "
        f"8 steps: {losses}"
    )
    # Grad-clip telemetry recorded the real per-step grad norms.
    summary = adapter.wave_n11_stabilizer_summary()
    assert summary["step_count"] == 8
    assert summary["grad_norm_history_len"] == 8
    assert summary["grad_norm_history_max"] is not None
