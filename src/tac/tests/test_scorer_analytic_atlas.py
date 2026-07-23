"""Focused no-fake tests for the Phase-0 scorer analytic atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.scorer_analytic_atlas import (
    AnalyticAtlasError,
    AtlasCheckpoint,
    ConsumptionStatus,
    FactorStatus,
    NonAdditivePool,
    SourceHashStamp,
    TensorArtifactRef,
    build_ddm_lambda_bundle,
    build_factor,
    build_gaze_factor,
    build_manifest,
    build_r_null_band_certificate,
    build_sdwl1_e2_coordinate_bridge,
    compose_jacobian_factors,
    derive_batchnorm_expected_stats,
    derive_bn_silu_contrast,
    derive_kernel_dft_bank,
    derive_se_gate_closed_form,
    evaluate_batchnorm_factor,
    evaluate_se_gate_factor,
    fp32_aggregation_order_envelope,
    fp32_sequential_sum,
    project_gaze_onto_axis,
    pull_back_gaze,
    require_total_gaze_coverage,
    write_stage_checkpoint,
)


def _stamp(source_id: str, token: str = "a") -> SourceHashStamp:
    return SourceHashStamp(
        source_id=source_id,
        path=f"fixture/{source_id}",
        sha256=token * 64,
        bytes=1,
        validity_horizon="exact input hash equality",
    )


def _tensor(name: str) -> TensorArtifactRef:
    return TensorArtifactRef(
        path=f"fixture/{name}.npy",
        sha256="b" * 64,
        bytes=4,
        shape=(1,),
        dtype="float32",
    )


def test_source_hash_and_factor_freshness_fail_closed() -> None:
    with pytest.raises(AnalyticAtlasError, match="invalid SHA"):
        _stamp("bad", "z")
    factor = build_factor(
        factor_id="one",
        factor_kind="closed_form.fixture",
        status=FactorStatus.DERIVED,
        payload={"value": 1},
        source_hashes=(_stamp("model"),),
    )
    factor.verify_fresh({"model": "a" * 64})
    with pytest.raises(AnalyticAtlasError, match="stale"):
        factor.verify_fresh({"model": "c" * 64})


def test_fp32_aggregation_envelope_reproduces_sequential_rounding() -> None:
    values = [16_777_216.0, 1.0, 1.0]
    assert fp32_sequential_sum(values) == 16_777_216.0
    assert fp32_sequential_sum([values[1], values[2], values[0]]) == 16_777_218.0
    envelope = fp32_aggregation_order_envelope(
        pose_batch_sums=values,
        seg_batch_sums=[0.1, 0.2, 0.3],
        total_pairs=3,
    )
    assert envelope["first_rung"] is True
    assert envelope["pose"]["raw_accumulator_span"] > 0.0
    assert envelope["score_span_upper_bound_if_term_extrema_cooccur"] > 0.0


def test_bn_expected_stats_is_exact_eval_affine_and_composes_silu() -> None:
    mean = np.array([1.0, -2.0])
    variance = np.array([4.0, 9.0])
    gamma = np.array([2.0, -3.0])
    beta = np.array([0.5, 1.5])
    epsilon = 0.25
    factor = derive_batchnorm_expected_stats(
        layer_id="stem.bn",
        running_mean=mean,
        running_variance=variance,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        source_hashes=(_stamp("segnet_checkpoint"),),
    )
    value = np.array([[3.0, 4.0], [-1.0, -2.0]])
    expected = gamma * (value - mean) / np.sqrt(variance + epsilon) + beta
    np.testing.assert_allclose(evaluate_batchnorm_factor(factor, value), expected)
    contrast = derive_bn_silu_contrast(layer_id="stem.bn", bn_factor=factor)
    assert contrast.payload["bn_factor_id"] == factor.factor_id
    assert contrast.first_rung is True


def test_se_gate_closed_form_matches_direct_composition() -> None:
    w1 = np.array([[1.0, -0.5], [0.25, 0.75]])
    b1 = np.array([0.1, -0.2])
    w2 = np.array([[0.5, 0.5], [-1.0, 0.25]])
    b2 = np.array([0.0, 0.3])
    factor = derive_se_gate_closed_form(
        layer_id="block.se",
        reduce_weight=w1,
        reduce_bias=b1,
        expand_weight=w2,
        expand_bias=b2,
        source_hashes=(_stamp("segnet_checkpoint"),),
    )
    x = np.array([[0.2, -0.4]])
    hidden = x @ w1.T + b1
    hidden = hidden / (1.0 + np.exp(-hidden))
    expected = 1.0 / (1.0 + np.exp(-(hidden @ w2.T + b2)))
    np.testing.assert_allclose(evaluate_se_gate_factor(factor, x), expected)


def test_kernel_dft_bank_is_derived_not_a_residual_basis_claim() -> None:
    kernels = np.zeros((1, 1, 2, 2), dtype=np.float64)
    kernels[0, 0, 0, 0] = 1.0
    factor = derive_kernel_dft_bank(
        layer_id="stem.conv",
        kernels=kernels,
        source_hashes=(_stamp("segnet_checkpoint"),),
    )
    np.testing.assert_allclose(
        np.asarray(factor.payload["magnitude"]),
        np.ones((1, 1, 2, 2)),
    )
    assert "curvelet/shearlet" in factor.payload["carrier_basis_authority"]


def test_amplitude_axis_requires_uint8_surviving_projection() -> None:
    common = {
        "factor_id": "amplitude:test",
        "gaze": np.array([1.0, 2.0]),
        "basis": np.eye(2),
        "axis_name": "amplitude",
        "source_hashes": (_stamp("gaze"),),
    }
    with pytest.raises(AnalyticAtlasError, match="uint8-surviving"):
        project_gaze_onto_axis(**common)
    factor = project_gaze_onto_axis(
        **common,
        uint8_surviving_projection={
            "projection_kind": "exact_integer_R_parseback",
            "survives_uint8_r": False,
            "artifact_sha256": "c" * 64,
        },
    )
    assert factor.uint8_surviving_projection["survives_uint8_r"] is False


def test_jacobian_composition_and_input_gaze_restriction() -> None:
    first = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])
    second = np.array([[1.0, 0.0, -1.0], [0.5, 2.0, 0.0]])
    composed = compose_jacobian_factors((first, second))
    np.testing.assert_allclose(composed, second @ first)
    terminal = np.array([2.0, -1.0])
    np.testing.assert_allclose(
        pull_back_gaze(terminal, (first, second)),
        first.T @ second.T @ terminal,
    )


def test_exact_gaze_contract_requires_both_n600_networks() -> None:
    sources = (_stamp("model"), _stamp("trajectory", "d"))
    seg = build_gaze_factor(
        network="segnet",
        layer_id="head_input",
        pair_start=0,
        pair_stop=600,
        tensor=_tensor("seg"),
        source_hashes=sources,
        vjp_count_per_pair=0,
        head_pullback_rank=4,
    )
    pose = build_gaze_factor(
        network="posenet",
        layer_id="pose6",
        pair_start=0,
        pair_stop=599,
        tensor=_tensor("pose_partial"),
        source_hashes=sources,
        vjp_count_per_pair=6,
        head_pullback_rank=None,
    )
    with pytest.raises(AnalyticAtlasError, match="not n600-complete"):
        require_total_gaze_coverage(
            (seg, pose),
            expected_layers={"segnet": ("head_input",), "posenet": ("pose6",)},
        )
    pose_full = build_gaze_factor(
        network="posenet",
        layer_id="pose6",
        pair_start=0,
        pair_stop=600,
        tensor=_tensor("pose"),
        source_hashes=sources,
        vjp_count_per_pair=6,
        head_pullback_rank=None,
    )
    require_total_gaze_coverage(
        (seg, pose_full),
        expected_layers={"segnet": ("head_input",), "posenet": ("pose6",)},
    )


def test_manifest_counts_unconsumed_factors_and_emits_nonadditive_pool() -> None:
    factor = build_factor(
        factor_id="frequency:waiting",
        factor_kind="axis_projection.frequency",
        status=FactorStatus.DERIVED,
        payload={"value": 1},
        source_hashes=(_stamp("model"),),
        nonadditive_pool_id="pool:stem",
    )
    pool = NonAdditivePool(
        pool_id="pool:stem",
        member_factor_ids=(factor.factor_id,),
        kkt_constraint="sum(member_spend)<=pool_budget",
        source_hashes=(_stamp("model"),),
    )
    manifest = build_manifest(
        factors=(factor,),
        pools=(pool,),
        materialization_status="PHASE0_TYPED_FOUNDATION_ONLY",
    )
    assert manifest["unconsumed_counted_inert_factor_ids"] == [factor.factor_id]
    assert manifest["nonadditive_pools"][0]["kkt_constraint"].startswith("sum(")


def test_stage_checkpoint_is_preserved_and_hash_stale_resume_refuses(
    tmp_path: Path,
) -> None:
    stamp = _stamp("model")
    checkpoint = AtlasCheckpoint(
        stage_id="closed_forms",
        completed_factor_ids=("bn:stem",),
        source_hashes=(stamp,),
        manifest_sha256="e" * 64,
    )
    checkpoint.verify_fresh({"model": stamp.sha256})
    with pytest.raises(AnalyticAtlasError, match="stale"):
        checkpoint.verify_fresh({"model": "f" * 64})
    path = tmp_path / "stage_closed_forms.json"
    write_stage_checkpoint(path, checkpoint)
    assert json.loads(path.read_text())["stage_id"] == "closed_forms"
    with pytest.raises(AnalyticAtlasError, match="already exists"):
        write_stage_checkpoint(path, checkpoint)


def test_r_null_certificate_reuses_580_without_inventing_dead_dft_bands() -> None:
    certificate = build_r_null_band_certificate(
        resize_authority=_stamp("resize_580"),
        requested_band_ids=("low", "high"),
    )
    assert certificate.payload["spatial_kernel"]["exact_rational_support_authority"]
    assert certificate.payload["exact_dead_band_ids"] == []
    assert (
        certificate.payload["frequency_band_admission"]
        == "REFUSE_ZERO_BYTE_TRUNCATION"
    )


def test_sdwl1_e2_bridge_accounts_every_known_asymmetry() -> None:
    bridge = build_sdwl1_e2_coordinate_bridge(
        source_hashes=(_stamp("sdwl1"), _stamp("e2_manifest", "b"))
    )
    assert bridge["status"] == "LOSS_ACCOUNTED_NOT_INVERTIBLE"
    assert bridge["sdwl1"]["declared_scalar_fact_count"] == 45_600
    assert bridge["e2"]["semantic_role_coordinates"] == 117_964_800
    assert bridge["e2"]["chart_coordinates"] == 702_000
    relations = {row["relation_id"]: row for row in bridge["relations"]}
    assert (
        relations["pair_screw_to_pose_runtime"]["loss_account"][
            "e2_packet_pose_bytes"
        ]
        == 0
    )
    assert bridge["pricing"]["u1_ladder"] == "BLOCKED_ON_LOSSY_RELATIONS"
    assert "families remain open" in bridge["verdict_scope"]


def test_lambda_producer_requires_complete_g3_before_v19_join() -> None:
    with pytest.raises(AnalyticAtlasError, match="exact n600"):
        build_ddm_lambda_bundle(
            atlas={},
            v19={"pair_recursion_ledger": {"rows": []}},
            source_hashes={
                "g3": "a" * 64,
                "v19": "b" * 64,
                "g3_full_atlas": "c" * 64,
            },
        )


def test_live_organ_consumes_atlas_lambda_bundle_only() -> None:
    from tac.ddm_costate_organ import build_live_ddm_costate

    report = build_live_ddm_costate()
    assert report["available"] is True
    assert report["lambda"]["producer"].endswith("build_ddm_lambda_bundle")
    assert report["lambda"]["producer_schema"] == "ddm_scorer_analytic_lambda_bundle.v1"
    assert report["lambda"]["unconsumed_missing_pairs_counted_inert"] is True
    if report["source_custody"]["g3_full_atlas"]["status"] == "VERIFIED":
        assert report["lambda"]["missing_exact_pair_lambda_count"] == 592
        assert all(row["first_rung"] is True for row in report["lambda"]["pair_rows"])


def test_factor_payload_hash_is_canonical() -> None:
    factor = build_factor(
        factor_id="canonical",
        factor_kind="closed_form.fixture",
        status=FactorStatus.DERIVED,
        payload={"b": 2, "a": 1},
        source_hashes=(_stamp("model"),),
        consumer="fixture.consumer",
        consumption_status=ConsumptionStatus.CONSUMED,
    )
    expected = hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    assert factor.content_sha256 == expected
