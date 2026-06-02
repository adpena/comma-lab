# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the Apple-GPU SCORER parity crux harness.

These tests run the REAL frozen SegNet+PoseNet on REAL ``upstream/videos/0.mkv``
frames (Catalog #213) on the MPS backend vs the bit-exact PyTorch-CPU reference.
They are BEHAVIORAL (Catalog #105/#139/#272 + Slot EEE Class 2): each headline
test FAILS if the cliff fix is reverted or the harness degenerates into a
constant, NOT if a metadata constant changes.

Tests requiring MPS + the frozen scorer weights + the real video skip
gracefully when those are unavailable (e.g. Linux CI), but assert the real
parity behavior whenever they CAN run (the M-series dev loop).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration import apple_gpu_scorer_parity_crux as crux

# ---------------------------------------------------------------------------
# Availability gates (skip-not-fake when the local substrate is missing)
# ---------------------------------------------------------------------------


def _mps_and_weights_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    if not torch.backends.mps.is_available():
        return False
    up = Path("upstream")
    return (
        (up / "models" / "segnet.safetensors").exists()
        and (up / "models" / "posenet.safetensors").exists()
        and (up / "videos" / "0.mkv").exists()
    )


requires_apple_gpu = pytest.mark.skipif(
    not _mps_and_weights_available(),
    reason="requires MPS backend + frozen scorer weights + real 0.mkv",
)


# ---------------------------------------------------------------------------
# Pure-contract tests (no torch / no MPS needed) — frozen invariants
# ---------------------------------------------------------------------------


def test_constants_and_tags_are_canonical():
    assert crux.SEGNET_CLIFF_LAYER == "decoder.blocks.0.conv1.0"
    assert crux.CLIFF_THRESHOLD == 1e-3
    assert crux.GRAD_COSINE_FAITHFUL_FLOOR == 0.999
    assert "Apple-GPU vs PyTorch-CPU parity" in crux.PARITY_TAG
    assert "macOS-CPU advisory" in crux.ADVISORY_TAG


def test_verdict_dataclasses_are_non_promotable():
    # The verdict dicts MUST carry the non-promotable markers (Catalog #192/#341).
    v = crux.ScorerParityVerdict(
        backend="mps",
        segnet_logits_l_inf=4e-5,
        segnet_argmax_flip_rate=0.0,
        posenet_scored_slice_l_inf=4e-6,
        posenet_scored_slice_rel=1e-7,
        segnet_worst_op="decoder.blocks.0.conv1.0",
        segnet_worst_l_inf=4e-4,
        posenet_worst_op="vision.x",
        posenet_worst_l_inf=4e-4,
        num_ops_above_cliff=0,
        grad_cosine_cpu_vs_apple=1.0,
        grad_norm_ratio=1.0,
        grad_max_abs=6e-7,
        cliff_fix_applied="cpu_wrap",
        forward_faithful=True,
        backward_faithful=True,
        residual_drift_factor_vs_stale_23x=1.1,
    )
    d = v.as_dict()
    assert d["score_claim"] is False
    assert d["promotion_eligible"] is False
    assert d["parity_tag"] == crux.PARITY_TAG

    p = crux.FaithfulnessProofVerdict(
        steps=25,
        backend_trained_on="mps",
        true_cpu_pose_before=0.0138,
        true_cpu_pose_after=0.0045,
        true_cpu_seg_before=0.00043,
        true_cpu_seg_after=0.00011,
        apple_pose_before=0.0138,
        apple_pose_after=0.0045,
        apple_seg_before=0.00043,
        apple_seg_after=0.00011,
        true_pose_reduced=True,
        true_seg_reduced=True,
        unlock=True,
    )
    pdct = p.as_dict()
    assert pdct["score_claim"] is False
    assert pdct["promotion_eligible"] is False
    assert pdct["advisory_tag"] == crux.ADVISORY_TAG


def test_build_faithful_scorers_rejects_bad_backend():
    with pytest.raises(RuntimeError):
        crux.build_faithful_apple_gpu_scorers(backend="cuda")


# ---------------------------------------------------------------------------
# Behavioral tests on the REAL scorer + MPS (the crux)
# ---------------------------------------------------------------------------


@requires_apple_gpu
def test_segnet_forward_argmax_is_faithful_on_mps():
    """SegNet argmax-flip rate (the metric that drives d_seg) is zero on MPS.

    Behavioral: if MPS SegNet ever drifted enough to flip an argmax class,
    this fails. It does not assert any constant.
    """
    v = crux.measure_forward_backward_parity(num_pairs=2, backend="mps", apply_cliff_fix=True)
    assert v.segnet_argmax_flip_rate == 0.0
    assert v.forward_faithful is True


@requires_apple_gpu
def test_posenet_forward_scored_slice_is_faithful_on_mps():
    v = crux.measure_forward_backward_parity(num_pairs=2, backend="mps", apply_cliff_fix=True)
    # The scored pose slice must track the CPU reference within the uint8 /
    # contest noise floor (< 1e-4 relative).
    assert v.posenet_scored_slice_rel < 1e-4


@requires_apple_gpu
def test_score_aware_gradient_direction_is_faithful_on_mps():
    """The TRAINING signal: gradient cosine(CPU, MPS) of the score-aware loss
    must be effectively 1.0. This is the backward-faithfulness crux."""
    v = crux.measure_forward_backward_parity(num_pairs=2, backend="mps", apply_cliff_fix=True)
    assert v.grad_cosine_cpu_vs_apple >= crux.GRAD_COSINE_FAITHFUL_FLOOR
    assert v.backward_faithful is True
    # Gradient norm must also be preserved (no systematic scaling drift).
    assert abs(v.grad_norm_ratio - 1.0) < 1e-3


@requires_apple_gpu
def test_cliff_fix_actually_collapses_the_segnet_conv_cliff():
    """HEADLINE NO-FAKE GUARD: the cpu_wrap fix on the ONE cliff op must
    reduce the worst per-op SegNet drift below the cliff threshold AND below
    the un-fixed drift. If a future edit reverts the fix wire-in, this FAILS.
    """
    # WITHOUT the fix: the cliff is present and above threshold.
    _, s_rows_unfixed = crux.measure_op_drift_table(
        num_pairs=1, backend="mps", apply_cliff_fix=False
    )
    worst_unfixed = s_rows_unfixed[0].l_inf
    cliff_row = next(
        (r for r in s_rows_unfixed if r.layer_name.startswith("decoder.blocks.0.conv1")), None
    )
    assert cliff_row is not None, "expected the SegNet conv cliff op in the drift table"
    assert worst_unfixed > crux.CLIFF_THRESHOLD, (
        f"un-fixed SegNet worst drift {worst_unfixed:.4e} should exceed the cliff "
        f"threshold {crux.CLIFF_THRESHOLD:.0e} (the bug class this fix targets)"
    )

    # WITH the fix: the worst op drops below threshold (cliff collapsed).
    _, s_rows_fixed = crux.measure_op_drift_table(
        num_pairs=1, backend="mps", apply_cliff_fix=True, cliff_strategy="cpu_wrap"
    )
    worst_fixed = s_rows_fixed[0].l_inf
    assert worst_fixed < crux.CLIFF_THRESHOLD, (
        f"cpu_wrap fix should collapse SegNet worst drift below {crux.CLIFF_THRESHOLD:.0e}, "
        f"got {worst_fixed:.4e}"
    )
    assert worst_fixed < worst_unfixed, "the fix must strictly reduce the worst drift"
    assert all(not r.above_cliff for r in s_rows_fixed), "no op should remain above cliff after fix"


@requires_apple_gpu
def test_fp32_force_is_a_noop_for_this_cliff_honest_finding():
    """HONEST NEGATIVE: fp32_force does NOT fix this cliff (the weight is
    already fp32; the MPS conv kernel still drifts). cpu_wrap is required.

    This documents the real per-op behavior; a future agent must not assume
    fp32_force is sufficient.
    """
    _, s_rows_fp32 = crux.measure_op_drift_table(
        num_pairs=1, backend="mps", apply_cliff_fix=True, cliff_strategy="fp32_force"
    )
    worst_fp32 = s_rows_fp32[0].l_inf
    # fp32_force leaves the cliff essentially unchanged (still near/above threshold).
    assert worst_fp32 > 5e-4, (
        f"fp32_force should NOT collapse this cliff (got {worst_fp32:.4e}); "
        "cpu_wrap is the required strategy"
    )


@requires_apple_gpu
def test_residual_drift_factor_is_far_below_stale_23x():
    """The canonical 23x PoseNet anchor is stale. The current residual factor
    must be order ~1x (single digits), proving the drift was engineered away
    / no longer exists on torch 2.11.0 MPS."""
    v = crux.measure_forward_backward_parity(num_pairs=2, backend="mps", apply_cliff_fix=True)
    assert v.residual_drift_factor_vs_stale_23x < 10.0, (
        f"residual drift factor {v.residual_drift_factor_vs_stale_23x:.3f} should be "
        "far below the stale 23x"
    )
    assert v.num_ops_above_cliff == 0


@requires_apple_gpu
def test_apple_gpu_score_aware_fit_reduces_TRUE_cpu_distortion_unlock():
    """THE UNLOCK PROOF (Slot EEE Class 2 behavioral): a score-aware fit run
    ON THE APPLE GPU must reduce the TRUE (CPU-mirror) d_pose AND d_seg, not
    just the Apple-GPU-measured proxy. If the scorer drift ever corrupted the
    training signal, the TRUE-after would exceed the TRUE-before and this
    FAILS.
    """
    p = crux.run_score_aware_faithfulness_smoke_fit(
        steps=25, lr=0.05, num_pairs=2, backend="mps", apply_cliff_fix=True
    )
    assert p.unlock is True
    assert p.true_cpu_pose_after < p.true_cpu_pose_before, "TRUE CPU d_pose must drop"
    assert p.true_cpu_seg_after < p.true_cpu_seg_before, "TRUE CPU d_seg must drop"
    # And the Apple-GPU proxy must track the TRUE value on the optimized
    # candidate (no drift accumulation pulling them apart during training).
    pose_gap = abs(p.apple_pose_after - p.true_cpu_pose_after) / (abs(p.true_cpu_pose_after) + 1e-12)
    assert pose_gap < 1e-2, (
        f"Apple-GPU-measured d_pose must track TRUE CPU on the optimized candidate "
        f"(rel gap {pose_gap:.4e})"
    )


@requires_apple_gpu
def test_real_scorer_outputs_are_nontrivial_not_constant():
    """Slot EEE Class 1 guard: the scorers must produce real, varying outputs
    on real frames (not a degenerate constant the parity check would trivially
    pass)."""
    import torch

    pn, sn = crux.build_faithful_apple_gpu_scorers(backend="mps", apply_cliff_fix=True)
    x = crux.load_real_btchw_frames(num_pairs=2, device="mps")
    with torch.no_grad():
        slog = sn(sn.preprocess_input(x)).float().cpu().numpy()
        pose = pn(pn.preprocess_input(x))["pose"].float().cpu().numpy()
    # SegNet logits must vary across classes/pixels (std well above zero).
    assert float(np.std(slog)) > 1e-2
    # PoseNet pose head must vary across pairs/dims.
    assert float(np.std(pose)) > 1e-3
    # And the argmax must use more than one class (real segmentation).
    assert len(np.unique(slog.argmax(1))) > 1
