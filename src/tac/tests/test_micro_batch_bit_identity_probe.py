"""Tests for the --micro-batch-pairs bit-identity DECOMPOSITION probe.

The probe answers the crux-engineering question "can B>1 be made bit-identical to serial by
fixed-order reduction?" NO on the real scorer (its batched forward is batch-dependent); the
reduction-order source is real but SECONDARY. These tests pin (a) the reduction-order
measurement on a batch-INVARIANT mock scorer (isolates the controllable source), and (b) the
classification logic across regimes. Historical bit-identity drift remains diagnostic. Functional
parity is classified under code-owned policy, while persisted timing is telemetry only and can
never grant training or score authority.

Run on MLX CPU (deterministic).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)

from tac.boundary_math.micro_batch_bit_identity_probe import (  # noqa: E402
    MEASURED_SCORER_FWD_CPU_ARGMAX_FLIPS,
    MEASURED_SCORER_FWD_GPU_SEG_MAXABS,
    MEASURED_SCORER_FWD_SPEEDUP_GPU,
    THETA_RELEVANT_VJP_INPUTS,
    BitIdentityVerdict,
    CompiledConfigIdentity,
    FunctionalParityReceipt,
    ReductionOrderDrift,
    _canonical_v9_synthetic_probe_context,
    build_schema_validated_functional_parity_telemetry,
    build_schema_validated_timing_telemetry,
    canonical_compiled_config_identity,
    canonical_scorer_fingerprint,
    classify_micro_batch_bit_identity,
    classify_training_admission,
    load_functional_parity_telemetry,
    load_timing_telemetry,
    make_functional_parity_receipt,
    measure_reduction_order_drift,
    measure_v9_synthetic_map_parity,
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
    assert v.training_throughput_admitted is False
    assert v.no_score_authority is True
    assert "operator override" in v.admission_path
    assert "TRAINING REFUSE" in v.admission_path


def test_classify_real_cpu_scorer_also_not_bit_identical():
    # CPU seg 7e-5 exceeds the fp32-eps invariance tol -> still not bit-identical.
    v = classify_micro_batch_bit_identity(
        device="cpu", scorer_fwd_seg_maxabs=7.1e-5, scorer_fwd_argmax_flips=0,
        scorer_fwd_pose_maxabs=2.0e-6, reduction_order_grad_maxabs=3.9e-3,
        scorer_fwd_speedup=1.75)
    assert v.scorer_forward_is_batch_invariant is False
    assert v.bit_identical_at_speedup_possible is False
    assert v.surviving_speedup_at_bit_identity == 1.0
    # Historical diagnostic remains intact: CPU batching did not flip scorer argmax pixels.
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
    assert v.training_throughput_admitted is False


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
                "dominant_source", "admission_path", "reported_functional_parity_supplied",
                "authoritative_functional_parity_established",
                "training_throughput_admitted", "training_only", "no_score_authority"):
        assert key in d
    assert "reports/latest.md" in d["pointer"]
    assert d["training_only"] is True
    assert d["no_score_authority"] is True


@pytest.mark.parametrize("lever", ["chroma", "phase", "temporal", "area"])
def test_functional_parity_receipt_records_all_measured_fields(lever):
    receipt = make_functional_parity_receipt(
        lever=lever, K=4, batched_loss=12.000001, serial_mean_loss=12.0,
        grad_rel_l2=2e-6, grad_maxabs=3e-4, backend_receipt="metal")
    assert isinstance(receipt, FunctionalParityReceipt)
    payload = receipt.as_dict()
    assert payload["lever"] == lever and payload["K"] == 4
    assert payload["loss_abs"] > 0.0 and payload["loss_rel"] > 0.0
    assert payload["grad_rel_l2"] == 2e-6 and payload["grad_maxabs"] == 3e-4
    assert payload["loss_rel_tolerance"] == 1e-4
    assert payload["grad_rel_tolerance"] == 1e-4
    assert payload["grad_maxabs_tolerance"] == 1e-2
    assert payload["backend_receipt"] == "metal"
    assert payload["passed"] is True


def test_faithful_functional_measurement_requires_micro_batch_K():
    with pytest.raises(ValueError, match="K >= 2"):
        measure_v9_synthetic_map_parity(K=1)


def test_synthetic_probe_context_uses_canonical_v9_semantics():
    context = _canonical_v9_synthetic_probe_context(K=2, height=4, width=5, seed=11)
    frame = context["frame_rgb"]
    target = context["target_chroma"]
    assert frame.shape == (2, 4, 5, 3)
    assert float(frame.min()) >= 0.0 and float(frame.max()) <= 255.0
    # A derived chroma target is luma-free under the same BT.601 projection.
    projected = 0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2]
    assert np.max(np.abs(projected)) < 5e-5
    assert np.array_equal(context["temporal_class_mask"], np.ones((3,), np.float32))
    for key in ("g1_probability", "g0_warped"):
        probability = context[key]
        assert float(probability.min()) >= 0.0 and float(probability.max()) <= 1.0
        assert np.all(np.sum(probability, axis=-1) <= 1.0 + 1e-6)
    assert context["area_classes"] == (1, 3)
    assert context["lever_weights"] == {"chroma": 0.1, "phase": 0.4, "temporal": 0.1}
    assert THETA_RELEVANT_VJP_INPUTS["temporal"] == ("g1", "g0_warped")


@pytest.fixture(scope="module")
def admission_fixture_dir():
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "experiments" / "results" / f".micro_batch_telemetry_test_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _write_artifact(root: Path, name: str, payload: str = "measured\n") -> Path:
    path = root / name
    path.write_text(payload)
    return path


def _functional_payload(lever: str, *, K: int = 2, **overrides):
    payload = {
        "schema": "micro_batch_functional_telemetry.v1",
        "lever": lever, "K": K, "batched_loss": 1.0, "serial_mean_loss": 1.0,
        "grad_rel_l2": 0.0, "grad_maxabs": 0.0,
        "loss_abs_tolerance": 1e-4, "loss_rel_tolerance": 1e-4,
        "grad_rel_tolerance": 1e-4, "grad_maxabs_tolerance": 1e-2,
        "reported_backend": {
            "chroma": "metal", "phase": "metal", "temporal": "metal",
            "area": "mlx_vectorized", "full_v9": "metal+mlx",
        }[lever],
        "reported_device": "gpu", "reported_scorer_surface": "real_frozen_v9",
        "reported_spatial_scale": True,
        "config_identity": canonical_compiled_config_identity(K).as_dict(),
        "scorer_fingerprint": canonical_scorer_fingerprint().as_dict(),
    }
    payload.update(overrides)
    return payload


def _passing_receipts(root: Path, *, K: int = 2):
    return [build_schema_validated_functional_parity_telemetry(
        _write_artifact(root, f"{lever}_k{K}.json", json.dumps(_functional_payload(lever, K=K))))
        for lever in ("chroma", "phase", "temporal", "area", "full_v9")]


def _passing_timing(root: Path, *, K: int = 2):
    payload = {"schema": "micro_batch_e2e_timing.v1", "serial_seconds": 1.2,
               "micro_batch_seconds": 1.0,
               "device": "gpu", "benchmark_surface": "full_v9_training_step",
               "faithful_scale": True, "serial_measured_steps": 8,
               "micro_batch_measured_steps": 8, "warmup_steps": 2,
               "clock": "time.perf_counter",
               "config_identity": canonical_compiled_config_identity(K).as_dict()}
    return build_schema_validated_timing_telemetry(
        _write_artifact(root, f"timing_k{K}.json", json.dumps(payload)))


def test_training_admission_classifies_complete_parity_but_disk_timing_never_go(
    admission_fixture_dir,
):
    receipts = _passing_receipts(admission_fixture_dir)
    timing = _passing_timing(admission_fixture_dir)
    refused = classify_training_admission(receipts, timing_receipt=timing)
    assert refused.functional_parity_passed is False
    assert refused.reported_functional_metrics_within_tolerance is True
    assert refused.timing_telemetry_valid is True
    assert refused.reported_end_to_end_speedup == pytest.approx(1.2)
    assert refused.training_throughput_admitted is False
    assert refused.runtime_admission_evidence_present is False
    assert "cannot attest execution" in refused.admission_blocker
    assert refused.training_only is True
    assert refused.no_score_authority is True

    missing = classify_training_admission(receipts[:-1], timing_receipt=timing)
    assert missing.missing_levers == ("full_v9",)
    assert missing.training_throughput_admitted is False

    no_speedup = classify_training_admission(receipts, reported_end_to_end_speedup=999.0)
    assert no_speedup.functional_parity_passed is False
    assert no_speedup.reported_functional_metrics_within_tolerance is True
    assert no_speedup.training_throughput_admitted is False

    serialized = receipts[0].as_dict()
    assert serialized["telemetry_only"] is True
    assert serialized["execution_attested"] is False
    assert serialized["can_establish_functional_parity"] is False
    assert "passed" not in serialized["reported_metrics"]
    assert serialized["reported_metrics"]["reported_metrics_within_tolerance"] is True


def test_training_admission_recomputes_reported_metrics(admission_fixture_dir):
    receipts = _passing_receipts(admission_fixture_dir)
    evidence = _functional_payload(
        "phase", batched_loss=1000.0, serial_mean_loss=1.0,
        grad_rel_l2=999.0, grad_maxabs=999.0)
    receipts[1] = build_schema_validated_functional_parity_telemetry(
        _write_artifact(admission_fixture_dir, "forged_phase_measurement.json",
                        json.dumps(evidence)))
    verdict = classify_training_admission(
        receipts, timing_receipt=_passing_timing(admission_fixture_dir))
    assert verdict.failed_levers == ("phase",)
    assert verdict.functional_parity_passed is False
    assert verdict.training_throughput_admitted is False


def test_training_admission_rejects_unchecked_receipts_even_if_json_fields_look_valid(
    admission_fixture_dir,
):
    receipts = _passing_receipts(admission_fixture_dir)
    # This is exactly what unchecked ``FunctionalParityReceipt(**json.loads(...))`` can do.
    receipts[0] = FunctionalParityReceipt(**receipts[0].receipt.as_dict())
    verdict = classify_training_admission(
        receipts, timing_receipt=_passing_timing(admission_fixture_dir))
    assert verdict.missing_levers == ("chroma",)
    assert verdict.functional_parity_passed is False
    assert verdict.training_throughput_admitted is False


def test_functional_loader_rejects_fake_hash_and_nonexistent_tmp(admission_fixture_dir):
    receipt = _passing_receipts(admission_fixture_dir)[0].as_dict()
    receipt["measurement_artifact"]["sha256"] = "0" * 64
    path = _write_artifact(admission_fixture_dir, "fake_hash.json", json.dumps([receipt]))
    with pytest.raises(ValueError, match="bytes/SHA-256"):
        load_functional_parity_telemetry(path)

    receipt["measurement_artifact"].update(
        {"path": "/tmp/nonexistent_micro_batch_receipt.json", "bytes": 1, "sha256": "a" * 64})
    path.write_text(json.dumps([receipt]))
    with pytest.raises(ValueError, match="bytes/SHA-256"):
        load_functional_parity_telemetry(path)


def test_config_and_scorer_are_bound_to_canonical_compiler_and_content(admission_fixture_dir):
    identity = canonical_compiled_config_identity(2)
    fake_identity = CompiledConfigIdentity(
        identity.config_id, identity.micro_batch_pairs, "0" * 64,
        identity.compiled_argv_sha256, identity.flag_manifest_sha256)
    payload = _functional_payload("chroma")
    payload["config_identity"] = fake_identity.as_dict()
    path = _write_artifact(admission_fixture_dir, "config_check.json", json.dumps(payload))
    with pytest.raises(ValueError, match="canonical compiled V9"):
        build_schema_validated_functional_parity_telemetry(path)

    scorer = canonical_scorer_fingerprint()
    fake_scorer = type(scorer)("0" * 64, scorer.posenet_sha256, scorer.aggregate_sha256)
    payload = _functional_payload("chroma")
    payload["scorer_fingerprint"] = fake_scorer.as_dict()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="canonical scorer bytes"):
        build_schema_validated_functional_parity_telemetry(path)


def test_timing_loader_verifies_bytes_ratio_and_rejects_inf(admission_fixture_dir):
    timing = _passing_timing(admission_fixture_dir)
    timing_path = _write_artifact(admission_fixture_dir, "timing_receipt.json",
                                  json.dumps(timing.as_dict()))
    loaded = load_timing_telemetry(timing_path)
    assert loaded.receipt.reported_end_to_end_speedup == pytest.approx(1.2)
    assert loaded.as_dict()["telemetry_only"] is True
    assert loaded.as_dict()["can_authorize_training"] is False

    inf_evidence = {"schema": "micro_batch_e2e_timing.v1", "serial_seconds": float("inf"),
                    "micro_batch_seconds": 1.0,
                    "device": "gpu", "benchmark_surface": "full_v9_training_step",
                    "faithful_scale": True, "serial_measured_steps": 8,
                    "micro_batch_measured_steps": 8, "warmup_steps": 2,
                    "clock": "time.perf_counter",
                    "config_identity": canonical_compiled_config_identity(2).as_dict()}
    inf_path = _write_artifact(admission_fixture_dir, "inf_timing.json", json.dumps(inf_evidence))
    with pytest.raises(ValueError, match="must be finite"):
        build_schema_validated_timing_telemetry(inf_path)

    forged = timing.as_dict()
    forged["measurement_artifact"]["sha256"] = "f" * 64
    timing_path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="bytes/SHA-256"):
        load_timing_telemetry(timing_path)


def test_post_build_artifact_mutation_revokes_admission(admission_fixture_dir):
    receipts = _passing_receipts(admission_fixture_dir)
    Path(receipts[0].measurement_artifact.path).write_text("mutated after validation\n")
    verdict = classify_training_admission(
        receipts, timing_receipt=_passing_timing(admission_fixture_dir))
    assert verdict.invalid_context_levers == ("chroma",)
    assert verdict.training_throughput_admitted is False


def test_private_wrapper_mutation_cannot_bypass_artifact_reparse(admission_fixture_dir):
    receipts = _passing_receipts(admission_fixture_dir)
    bad_payload = _functional_payload(
        "phase", batched_loss=1000.0, serial_mean_loss=1.0,
        grad_rel_l2=999.0, grad_maxabs=999.0)
    bad = build_schema_validated_functional_parity_telemetry(
        _write_artifact(admission_fixture_dir, "bad_wrapped_phase.json", json.dumps(bad_payload)))
    bad.receipt = replace(
        bad.receipt, loss_abs=0.0, loss_rel=0.0, grad_rel_l2=0.0,
        grad_maxabs=0.0, passed=True)
    receipts[1] = bad
    timing = _passing_timing(admission_fixture_dir)
    timing.receipt = replace(timing.receipt, reported_end_to_end_speedup=float("inf"))
    verdict = classify_training_admission(receipts, timing_receipt=timing)
    assert set(verdict.invalid_context_levers) == {
        "chroma", "phase", "temporal", "area", "full_v9",
    }
    assert verdict.training_throughput_admitted is False


def test_timing_builder_rejects_nonfaithful_cpu_or_unequal_steps(admission_fixture_dir):
    payload = {"schema": "micro_batch_e2e_timing.v1", "serial_seconds": 2.0,
               "micro_batch_seconds": 1.0, "device": "cpu",
               "benchmark_surface": "tiny_mock", "faithful_scale": False,
               "serial_measured_steps": 8, "micro_batch_measured_steps": 1,
               "warmup_steps": 0, "clock": "time.perf_counter",
               "config_identity": canonical_compiled_config_identity(2).as_dict()}
    path = _write_artifact(admission_fixture_dir, "nonfaithful_timing.json", json.dumps(payload))
    with pytest.raises(ValueError, match="faithful GPU full-V9"):
        build_schema_validated_timing_telemetry(path)


def test_artifact_cannot_select_huge_functional_tolerances(admission_fixture_dir):
    payload = _functional_payload("phase", loss_abs_tolerance=1e99,
                                  grad_maxabs_tolerance=1e99)
    path = _write_artifact(admission_fixture_dir, "huge_tolerances.json", json.dumps(payload))
    with pytest.raises(ValueError, match="module-owned canonical threshold"):
        build_schema_validated_functional_parity_telemetry(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reported_device", "cpu", "device must be canonical"),
        ("reported_scorer_surface", "mock", "scorer surface must be canonical"),
        ("reported_spatial_scale", False, "reported_spatial_scale must be true"),
        ("reported_backend", "mlx_reference", "backend must be canonical"),
    ],
)
def test_artifact_cannot_select_functional_context_policy(
    admission_fixture_dir, field, value, message,
):
    payload = _functional_payload("chroma", **{field: value})
    path = _write_artifact(admission_fixture_dir, f"wrong_{field}.json", json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        build_schema_validated_functional_parity_telemetry(path)


def test_required_levers_cannot_be_weakened_by_caller(admission_fixture_dir):
    receipts = _passing_receipts(admission_fixture_dir)
    with pytest.raises(TypeError):
        classify_training_admission(receipts, required_levers=())


def test_functional_receipts_must_share_exact_config_identity_and_K(admission_fixture_dir):
    receipts_k2 = _passing_receipts(admission_fixture_dir, K=2)
    receipt_k4 = _passing_receipts(admission_fixture_dir, K=4)[1]
    receipts_k2[1] = receipt_k4
    verdict = classify_training_admission(receipts_k2)
    assert verdict.functional_parity_passed is False
    assert set(verdict.invalid_context_levers) == {
        "chroma", "phase", "temporal", "area", "full_v9",
    }
    assert verdict.training_throughput_admitted is False


def test_timing_telemetry_must_share_functional_config_identity_and_K(admission_fixture_dir):
    receipts = _passing_receipts(admission_fixture_dir, K=2)
    timing_k4 = _passing_timing(admission_fixture_dir, K=4)
    verdict = classify_training_admission(receipts, timing_receipt=timing_k4)
    assert verdict.timing_telemetry_valid is False
    assert verdict.functional_parity_passed is False
    assert verdict.training_throughput_admitted is False


@pytest.mark.parametrize(
    "transient_path",
    [
        "/tmp/micro_batch_receipt.json",
        "/private/tmp/micro_batch_receipt.json",
        "/var/tmp/micro_batch_receipt.json",
        "/dev/shm/micro_batch_receipt.json",
        str(Path(tempfile.gettempdir()) / "micro_batch_receipt.json"),
        str(Path(__file__).resolve().parents[3] / ".cache" / "micro_batch_receipt.json"),
        str(Path(__file__).resolve().parents[3] / "tmp" / "micro_batch_receipt.json"),
    ],
)
def test_admission_artifacts_reject_transient_and_repo_cache_paths(transient_path):
    with pytest.raises(ValueError, match="allowlisted durable result root"):
        build_schema_validated_functional_parity_telemetry(transient_path)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": "micro_batch_functional_telemetry.v1"},
        {"schema": "micro_batch_functional_telemetry.v1", "lever": "chroma",
         "config_identity": []},
        ["not", "a", "mapping"],
    ],
)
def test_malformed_functional_mappings_raise_value_error(admission_fixture_dir, payload):
    path = _write_artifact(admission_fixture_dir, "malformed_functional.json",
                           json.dumps(payload))
    with pytest.raises(ValueError):
        build_schema_validated_functional_parity_telemetry(path)


@pytest.mark.parametrize(("field", "value"), [
    ("batched_loss", True),
    ("serial_mean_loss", "1.0"),
    ("grad_rel_l2", False),
    ("grad_maxabs", "0.0"),
])
def test_functional_telemetry_rejects_bool_and_numeric_strings(
    admission_fixture_dir, field, value,
):
    payload = _functional_payload("chroma", **{field: value})
    path = _write_artifact(admission_fixture_dir, f"bad_numeric_{field}.json", json.dumps(payload))
    with pytest.raises(ValueError, match="JSON number"):
        build_schema_validated_functional_parity_telemetry(path)


@pytest.mark.parametrize(("field", "value"), [
    ("serial_seconds", True),
    ("micro_batch_seconds", "0.5"),
])
def test_timing_telemetry_rejects_bool_and_numeric_strings(
    admission_fixture_dir, field, value,
):
    payload = {
        "schema": "micro_batch_e2e_timing.v1", "serial_seconds": 1.0,
        "micro_batch_seconds": 0.5, "device": "gpu",
        "benchmark_surface": "full_v9_training_step", "faithful_scale": True,
        "serial_measured_steps": 2, "micro_batch_measured_steps": 2,
        "warmup_steps": 1, "clock": "time.perf_counter",
        "config_identity": canonical_compiled_config_identity(2).as_dict(),
    }
    payload[field] = value
    path = _write_artifact(admission_fixture_dir, f"bad_timing_{field}.json", json.dumps(payload))
    with pytest.raises(ValueError, match="JSON number"):
        build_schema_validated_timing_telemetry(path)


def test_malformed_receipt_bundle_row_raises_value_error(admission_fixture_dir):
    path = _write_artifact(admission_fixture_dir, "malformed_bundle.json", json.dumps([42]))
    with pytest.raises(ValueError, match="JSON object"):
        load_functional_parity_telemetry(path)

    path.write_text(json.dumps([{
        "measurement_artifact": {"path": None, "bytes": "many", "sha256": []},
    }]))
    with pytest.raises(ValueError):
        load_functional_parity_telemetry(path)


def test_malformed_timing_mapping_raises_value_error(admission_fixture_dir):
    payload = {
        "schema": "micro_batch_e2e_timing.v1",
        "serial_seconds": 1.0,
        "micro_batch_seconds": 0.5,
        "device": "gpu",
        "benchmark_surface": "full_v9_training_step",
        "faithful_scale": True,
        "serial_measured_steps": 2,
        "micro_batch_measured_steps": 2,
        "warmup_steps": 1,
        "clock": "time.perf_counter",
        "config_identity": [],
    }
    path = _write_artifact(admission_fixture_dir, "malformed_timing.json", json.dumps(payload))
    with pytest.raises(ValueError, match="JSON object"):
        build_schema_validated_timing_telemetry(path)


def test_handwritten_timing_json_is_telemetry_only_and_never_go(admission_fixture_dir):
    receipts = _passing_receipts(admission_fixture_dir)
    timing = _passing_timing(admission_fixture_dir)
    verdict = classify_training_admission(receipts, timing_receipt=timing)
    assert verdict.timing_telemetry_valid is True
    assert verdict.reported_end_to_end_speedup > 1.0
    assert verdict.training_throughput_admitted is False


def test_bit_identity_verdict_refuses_training_when_functional_parity_fails():
    verdict = classify_micro_batch_bit_identity(
        device="gpu", scorer_fwd_seg_maxabs=2.3e-2, scorer_fwd_argmax_flips=11,
        scorer_fwd_pose_maxabs=7.7e-3, reduction_order_grad_maxabs=3.9e-3,
        scorer_fwd_speedup=1.56, reported_functional_parity_supplied=False)
    assert verdict.reported_functional_parity_supplied is False
    assert verdict.authoritative_functional_parity_established is False
    assert verdict.training_throughput_admitted is False
    assert "TRAINING REFUSE" in verdict.admission_path


def test_bit_identity_microbench_never_substitutes_for_end_to_end_admission():
    verdict = classify_micro_batch_bit_identity(
        device="gpu", scorer_fwd_seg_maxabs=2.3e-2, scorer_fwd_argmax_flips=11,
        scorer_fwd_pose_maxabs=7.7e-3, reduction_order_grad_maxabs=3.9e-3,
        scorer_fwd_speedup=1.56, reported_functional_parity_supplied=True)
    assert verdict.reported_functional_parity_supplied is True
    assert verdict.authoritative_functional_parity_established is False
    assert "functional_parity_passed" not in verdict.as_dict()
    assert verdict.training_throughput_admitted is False
    assert "DIAGNOSTIC ONLY" in verdict.admission_path
    assert "persisted" in verdict.admission_path
    assert "REFUSE" in verdict.admission_path


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
