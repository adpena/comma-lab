"""Tests for the --micro-batch-pairs bit-identity DECOMPOSITION probe.

The probe answers the crux-engineering question "can B>1 be made bit-identical to serial by
fixed-order reduction?" NO on the real scorer (its batched forward is batch-dependent); the
reduction-order source is real but SECONDARY. These tests pin (a) the reduction-order
measurement on a batch-INVARIANT mock scorer (isolates the controllable source), and (b) the
classification logic (the honest verdict) across regimes.

Run on MLX CPU (deterministic).
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from tac.boundary_math.micro_batch_bit_identity_probe import (  # noqa: E402
    MEASURED_SCORER_FWD_CPU_ARGMAX_FLIPS,
    MEASURED_SCORER_FWD_GPU_SEG_MAXABS,
    MEASURED_SCORER_FWD_SPEEDUP_GPU,
    BitIdentityVerdict,
    ReductionOrderDrift,
    classify_micro_batch_bit_identity,
    measure_reduction_order_drift,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reduction-order measurement (source B, batch-invariant mock scorer)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seg_form", ["ce", "tau_softplus", "l7_softplus", "margin_hinge"])
@pytest.mark.parametrize("K", [2, 4])
def test_reduction_order_drift_is_finite_and_present(seg_form, K):
    d = measure_reduction_order_drift(K=K, seg_form=seg_form)
    assert isinstance(d, ReductionOrderDrift)
    assert d.K == K and d.seg_form == seg_form
    assert np.isfinite(d.grad_maxabs) and d.grad_maxabs >= 0.0
    assert np.isfinite(d.grad_rel_l2) and d.grad_rel_l2 >= 0.0
    assert np.isfinite(d.loss_abs) and d.loss_abs >= 0.0


def test_reduction_order_is_nondeterministic_but_bounded():
    # STRENGTHENED finding: MLX's batched-backward reduction order is itself NOT stable
    # run-to-run (even on CPU, even with byte-identical inputs) -> the drift lands on
    # different ULP boundaries across calls (e.g. 1/1024 vs 2/1024 vs 4/1024 on the
    # out_tex leaf). This is the sister of the #348 MLX-GPU cross-process non-determinism:
    # you cannot "fixed-order match" a reduction whose order is itself non-deterministic.
    # We therefore assert BOUNDED + present, not exactly reproducible.
    vals = [measure_reduction_order_drift(K=4, seg_form="ce").grad_maxabs for _ in range(4)]
    assert all(np.isfinite(v) and 0.0 <= v < 1e-1 for v in vals), vals
    assert max(vals) > 0.0  # the reorder is real (never exactly bit-identical)


def test_reduction_order_maxabs_exceeds_rel_l2_hidden_by_global_metric():
    # The key finding: the trajectory-relevant max|Δ| (~1e-3) is HIDDEN by the global-L2
    # metric (~1e-7) the equivalence tests use. maxabs must be many orders above rel_l2.
    d = measure_reduction_order_drift(K=4, seg_form="ce")
    assert d.grad_maxabs > 1e-4, d.grad_maxabs
    assert d.grad_rel_l2 < 1e-5, d.grad_rel_l2
    assert d.grad_maxabs > d.grad_rel_l2 * 100.0


def test_reduction_order_nonzero_means_not_bit_identical_even_on_invariant_scorer():
    # Even with a perfectly batch-invariant (mock) scorer, the batched twin's one-shot
    # value_and_grad reduces the K per-pair contributions in a DIFFERENT order than the
    # serial left-fold -> NOT bit-identical. This is the source the fixed-order fix targets.
    d = measure_reduction_order_drift(K=4, seg_form="ce")
    assert d.grad_maxabs > 0.0


def test_reduction_order_present_at_multiple_K():
    # Drift is present + bounded at both K (max over runs, since the order is
    # non-deterministic — see test_reduction_order_is_nondeterministic_but_bounded).
    m2 = max(measure_reduction_order_drift(K=2, seg_form="ce").grad_maxabs for _ in range(4))
    m4 = max(measure_reduction_order_drift(K=4, seg_form="ce").grad_maxabs for _ in range(4))
    assert m2 > 0.0 and m4 > 0.0
    assert m2 < 1e-1 and m4 < 1e-1


# ─────────────────────────────────────────────────────────────────────────────
# Classification (the honest verdict)
# ─────────────────────────────────────────────────────────────────────────────
def test_classify_real_gpu_scorer_is_not_bit_identical_at_speedup():
    v = classify_micro_batch_bit_identity(
        device="gpu", scorer_fwd_seg_maxabs=MEASURED_SCORER_FWD_GPU_SEG_MAXABS,
        scorer_fwd_argmax_flips=11, scorer_fwd_pose_maxabs=7.7e-3,
        reduction_order_grad_maxabs=3.9e-3, scorer_fwd_speedup=MEASURED_SCORER_FWD_SPEEDUP_GPU)
    assert isinstance(v, BitIdentityVerdict)
    assert v.scorer_forward_is_batch_invariant is False
    assert v.bit_identical_at_speedup_possible is False
    assert v.surviving_speedup_at_bit_identity == 1.0
    assert v.dominant_source == "scorer_forward"
    assert "bounded" in v.admission_path or "batch-invariant" in v.admission_path


def test_classify_real_cpu_scorer_also_not_bit_identical():
    # CPU seg 7e-5 exceeds the fp32-eps invariance tol -> still not bit-identical.
    v = classify_micro_batch_bit_identity(
        device="cpu", scorer_fwd_seg_maxabs=7.1e-5, scorer_fwd_argmax_flips=0,
        scorer_fwd_pose_maxabs=2.0e-6, reduction_order_grad_maxabs=3.9e-3,
        scorer_fwd_speedup=1.75)
    assert v.scorer_forward_is_batch_invariant is False
    assert v.bit_identical_at_speedup_possible is False
    assert v.surviving_speedup_at_bit_identity == 1.0
    # argmax IS invariant on CPU (the load-bearing sub-property for the bounded A/B path)
    assert v.argmax_is_batch_invariant is True


def test_classify_hypothetical_invariant_scorer_admits_at_speedup():
    # IF a batch-invariant scorer kernel existed, the reduction-order fix makes B>1
    # bit-identical AND the batched speedup survives.
    v = classify_micro_batch_bit_identity(
        device="ideal", scorer_fwd_seg_maxabs=0.0, scorer_fwd_argmax_flips=0,
        scorer_fwd_pose_maxabs=0.0, reduction_order_grad_maxabs=3.9e-3, scorer_fwd_speedup=1.56)
    assert v.scorer_forward_is_batch_invariant is True
    assert v.bit_identical_at_speedup_possible is True
    assert v.surviving_speedup_at_bit_identity == 1.56
    assert v.dominant_source == "reduction_order"
    assert "without A/B" in v.admission_path


def test_classify_invariant_scorer_zero_reduction_source_is_none():
    v = classify_micro_batch_bit_identity(
        device="ideal", scorer_fwd_seg_maxabs=0.0, scorer_fwd_argmax_flips=0,
        scorer_fwd_pose_maxabs=0.0, reduction_order_grad_maxabs=0.0, scorer_fwd_speedup=1.5)
    assert v.dominant_source == "none"


def test_classify_argmax_flips_recorded_and_gpu_flips_nonzero():
    v = classify_micro_batch_bit_identity(
        device="gpu", scorer_fwd_seg_maxabs=2.3e-2, scorer_fwd_argmax_flips=11,
        scorer_fwd_pose_maxabs=7.7e-3, reduction_order_grad_maxabs=3.9e-3, scorer_fwd_speedup=1.56)
    assert v.scorer_fwd_argmax_flips == 11
    assert v.argmax_is_batch_invariant is False


def test_verdict_as_dict_roundtrips_all_fields_and_carries_pointer():
    v = classify_micro_batch_bit_identity(
        device="gpu", scorer_fwd_seg_maxabs=2.3e-2, scorer_fwd_argmax_flips=11,
        scorer_fwd_pose_maxabs=7.7e-3, reduction_order_grad_maxabs=3.9e-3, scorer_fwd_speedup=1.56)
    d = v.as_dict()
    for key in ("device", "scorer_fwd_seg_maxabs", "scorer_fwd_argmax_flips",
                "bit_identical_at_speedup_possible", "surviving_speedup_at_bit_identity",
                "dominant_source", "admission_path"):
        assert key in d
    assert "0.19110" in d["pointer"]  # MEANS discipline: pointer carried on every verdict


def test_bit_invariant_tol_boundary_is_fp32_eps_scale():
    # Exactly at the tol -> invariant; just above -> not. Pins the threshold semantics.
    tol = 1e-6
    v_at = classify_micro_batch_bit_identity(
        device="x", scorer_fwd_seg_maxabs=tol, scorer_fwd_argmax_flips=0,
        scorer_fwd_pose_maxabs=0.0, reduction_order_grad_maxabs=1e-3, scorer_fwd_speedup=2.0,
        scorer_fwd_bit_invariant_tol=tol)
    v_above = classify_micro_batch_bit_identity(
        device="x", scorer_fwd_seg_maxabs=tol * 10, scorer_fwd_argmax_flips=0,
        scorer_fwd_pose_maxabs=0.0, reduction_order_grad_maxabs=1e-3, scorer_fwd_speedup=2.0,
        scorer_fwd_bit_invariant_tol=tol)
    assert v_at.scorer_forward_is_batch_invariant is True
    assert v_above.scorer_forward_is_batch_invariant is False


def test_measured_anchors_are_consistent_with_the_finding():
    # The recorded anchors must encode the finding: GPU seg drift >> CPU; CPU argmax flips 0.
    assert MEASURED_SCORER_FWD_GPU_SEG_MAXABS > 1e-3
    assert MEASURED_SCORER_FWD_CPU_ARGMAX_FLIPS == 0
    assert MEASURED_SCORER_FWD_SPEEDUP_GPU > 1.0
