# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.canonical_equations.scorer_input_cache_hash_identity import (
    EQUATION_ID,
    build_scorer_input_cache_hash_identity_v1,
    scorer_input_cache_hash_identity,
)

HASH_DOMAIN = "_array_sha256(dtype_string + json_shape + contiguous_bytes)"
HASHES = {
    "segnet_last_rgb": "c" * 64,
    "posenet_yuv6_pair": "p" * 64,
    "pair_indices": "i" * 64,
}
SHAPES = {
    "segnet_last_rgb": [600, 3, 384, 512],
    "posenet_yuv6_pair": [600, 12, 192, 256],
    "pair_indices": [600, 2],
}


def test_scorer_input_cache_hash_identity_passes_matching_surface() -> None:
    result = scorer_input_cache_hash_identity(
        cache_archive_sha256="a" * 64,
        auth_archive_sha256="a" * 64,
        cache_inflated_outputs_aggregate_sha256="b" * 64,
        auth_inflated_outputs_aggregate_sha256="b" * 64,
        cache_raw_sha256="r" * 64,
        auth_raw_sha256="r" * 64,
        cache_pair_count=600,
        auth_n_samples=600,
        cache_hash_domain=HASH_DOMAIN,
        auth_hash_domain=HASH_DOMAIN,
        cache_array_sha256=HASHES,
        auth_scorer_input_array_sha256=HASHES,
        cache_shapes=SHAPES,
        auth_shapes=SHAPES,
    )

    assert result["equation_id"] == EQUATION_ID
    assert result["identity_residual"] == 0
    assert result["eligible_for_local_mlx_transfer_calibration"] is True
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False


def test_scorer_input_cache_hash_identity_fails_mismatched_surface() -> None:
    result = scorer_input_cache_hash_identity(
        cache_archive_sha256="a" * 64,
        auth_archive_sha256="a" * 64,
        cache_inflated_outputs_aggregate_sha256="b" * 64,
        auth_inflated_outputs_aggregate_sha256="d" * 64,
        cache_raw_sha256="r" * 64,
        auth_raw_sha256="r" * 64,
        cache_pair_count=600,
        auth_n_samples=600,
        cache_hash_domain=HASH_DOMAIN,
        auth_hash_domain=HASH_DOMAIN,
        cache_array_sha256=HASHES,
        auth_scorer_input_array_sha256={**HASHES, "segnet_last_rgb": "e" * 64},
        cache_shapes=SHAPES,
        auth_shapes=SHAPES,
    )

    assert result["identity_residual"] == 2
    assert result["eligible_for_local_mlx_transfer_calibration"] is False
    assert "inflated_outputs_aggregate_sha256_mismatch_or_missing" in result["blockers"]
    assert "scorer_input_array_sha256_mismatch:segnet_last_rgb" in result["blockers"]


def test_scorer_input_cache_hash_identity_equation_builds_with_anchor() -> None:
    eq = build_scorer_input_cache_hash_identity_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.python_callable_module_path.endswith(":scorer_input_cache_hash_identity")
    assert len(eq.empirical_anchors) == 1
    assert eq.predicted_vs_empirical_residual[
        "fec6_pr101_full_cache_vs_streaming_hash_identity"
    ] == 0.0
