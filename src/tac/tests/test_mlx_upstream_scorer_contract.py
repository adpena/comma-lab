# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.local_acceleration.mlx_preprocess import CAMERA_HW, SEGNET_INPUT_HW, SEQ_LEN, YUV6_INPUT_HW
from tac.local_acceleration.mlx_upstream_scorer_contract import (
    MLX_UPSTREAM_SCORER_CONTRACT_SCHEMA,
    build_mlx_upstream_scorer_contract_fidelity,
    validate_mlx_cache_manifest_against_upstream_contract,
)


def _full_shape_manifest() -> dict:
    pair_count = 600
    return {
        "pair_count": pair_count,
        "segnet_last_rgb_shape": [pair_count, 3, *SEGNET_INPUT_HW],
        "posenet_yuv6_pair_shape": [pair_count, 12, *YUV6_INPUT_HW],
        "pair_indices_shape": [pair_count, SEQ_LEN],
        "frame_shape_hwc": [*CAMERA_HW, 3],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_mlx_upstream_scorer_contract_passes_static_source_and_shape_gate() -> None:
    contract = build_mlx_upstream_scorer_contract_fidelity()

    assert contract["schema"] == MLX_UPSTREAM_SCORER_CONTRACT_SCHEMA
    assert contract["passed"] is True
    assert contract["blockers"] == []
    assert contract["score_claim"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["upstream_contract"]["contract_valid"] is True
    assert contract["mlx_contract"]["adapter_consumes_fixed_inputs_not_raw_frames"] is True
    assert contract["mlx_contract"]["segnet_last_rgb_shape_nchw"] == ["N", 3, *SEGNET_INPUT_HW]
    assert contract["mlx_contract"]["posenet_yuv6_pair_shape_nchw"] == ["N", 12, *YUV6_INPUT_HW]


def test_mlx_cache_manifest_validation_accepts_full_contest_shapes_fail_closed() -> None:
    validation = validate_mlx_cache_manifest_against_upstream_contract(_full_shape_manifest())

    assert validation["passed"] is True
    assert validation["blockers"] == []


def test_mlx_cache_manifest_validation_rejects_authority_flags_and_bad_shapes() -> None:
    bad = _full_shape_manifest()
    bad["pair_count"] = 1
    bad["segnet_last_rgb_shape"] = [1, 3, 64, 64]
    bad["promotion_eligible"] = True

    validation = validate_mlx_cache_manifest_against_upstream_contract(bad)

    assert validation["passed"] is False
    assert "segnet_last_rgb_shape_mismatch" in validation["blockers"]
    assert "cache_manifest_attempts_promotion_eligible" in validation["blockers"]


def test_mlx_cache_manifest_validation_reports_missing_shape_without_crashing() -> None:
    bad = _full_shape_manifest()
    del bad["posenet_yuv6_pair_shape"]

    validation = validate_mlx_cache_manifest_against_upstream_contract(bad)

    assert validation["passed"] is False
    assert "posenet_yuv6_pair_shape_mismatch" in validation["blockers"]
