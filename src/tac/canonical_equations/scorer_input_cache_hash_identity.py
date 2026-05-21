# SPDX-License-Identifier: MIT
"""Canonical equation: scorer-input cache hash identity.

This equation formalizes the MLX/local-acceleration transfer bridge:

    cache is auth-axis-transfer eligible
        iff archive SHA, inflated-output aggregate SHA, raw SHA, pair/sample
        count, hash domain, scorer-input array hashes, and tensor shapes match
        the target auth surface.

The equation is intentionally an identity invariant, not a score predictor.
It creates no score authority; it only prevents local MLX cache/surrogate
signals from being treated as contest-axis-faithful when their byte surface
differs from the target auth-eval run.
"""

from __future__ import annotations

from typing import Any, Mapping

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "scorer_input_cache_hash_identity_v1"


def scorer_input_cache_hash_identity(
    *,
    cache_archive_sha256: str | None,
    auth_archive_sha256: str | None,
    cache_inflated_outputs_aggregate_sha256: str | None,
    auth_inflated_outputs_aggregate_sha256: str | None,
    cache_pair_count: int | None,
    auth_n_samples: int | None,
    cache_raw_sha256: str | None = None,
    auth_raw_sha256: str | None = None,
    cache_hash_domain: str | None = None,
    auth_hash_domain: str | None = None,
    cache_array_sha256: Mapping[str, str] | None = None,
    auth_scorer_input_array_sha256: Mapping[str, str] | None = None,
    cache_shapes: Mapping[str, object] | None = None,
    auth_shapes: Mapping[str, object] | None = None,
    required_scorer_input_hashes: tuple[str, ...] = (
        "segnet_last_rgb",
        "posenet_yuv6_pair",
        "pair_indices",
    ),
) -> dict[str, Any]:
    """Evaluate the scorer-input cache identity invariant.

    ``identity_residual`` is the count of mismatched required identity fields
    plus mismatched scorer-input hashes. Zero means the cache can be used for
    transfer calibration against the matching auth axis.
    """

    blockers: list[str] = []
    if _clean(cache_archive_sha256) != _clean(auth_archive_sha256):
        blockers.append("archive_sha256_mismatch_or_missing")
    if _clean(cache_inflated_outputs_aggregate_sha256) != _clean(
        auth_inflated_outputs_aggregate_sha256
    ):
        blockers.append("inflated_outputs_aggregate_sha256_mismatch_or_missing")
    if _clean(cache_raw_sha256) != _clean(auth_raw_sha256):
        blockers.append("raw_sha256_mismatch_or_missing")
    if _clean(cache_hash_domain) != _clean(auth_hash_domain):
        blockers.append("hash_domain_mismatch_or_missing")
    if cache_pair_count is None or auth_n_samples is None:
        blockers.append("pair_count_or_auth_n_samples_missing")
    elif int(cache_pair_count) != int(auth_n_samples):
        blockers.append(
            f"pair_count_mismatch:cache={int(cache_pair_count)}:auth={int(auth_n_samples)}"
        )

    cache_hashes = _hashes(cache_array_sha256)
    auth_hashes = _hashes(auth_scorer_input_array_sha256)
    cache_shape_map = _shapes(cache_shapes)
    auth_shape_map = _shapes(auth_shapes)
    compared_hashes: dict[str, dict[str, str | None | bool]] = {}
    compared_shapes: dict[str, dict[str, list[int] | None | bool]] = {}
    required = tuple(required_scorer_input_hashes)
    for name in required:
        auth_hash = auth_hashes.get(name)
        cache_hash = cache_hashes.get(name)
        matches = bool(cache_hash and auth_hash and cache_hash == auth_hash)
        compared_hashes[name] = {
            "cache": cache_hash,
            "auth": auth_hash,
            "matches": matches,
        }
        if not auth_hash:
            blockers.append(f"auth_scorer_input_array_sha256_missing:{name}")
        elif not cache_hash or cache_hash != auth_hash:
            blockers.append(f"scorer_input_array_sha256_mismatch:{name}")
        auth_shape = auth_shape_map.get(name)
        cache_shape = cache_shape_map.get(name)
        shape_matches = bool(cache_shape and auth_shape and cache_shape == auth_shape)
        compared_shapes[name] = {
            "cache": cache_shape,
            "auth": auth_shape,
            "matches": shape_matches,
        }
        if auth_shape is None:
            blockers.append(f"auth_scorer_input_shape_missing:{name}")
        elif cache_shape is None or cache_shape != auth_shape:
            blockers.append(f"scorer_input_shape_mismatch:{name}")

    for name, auth_hash in sorted(auth_hashes.items()):
        if name in required:
            continue
        cache_hash = cache_hashes.get(name)
        matches = bool(cache_hash and cache_hash == auth_hash)
        compared_hashes[name] = {
            "cache": cache_hash,
            "auth": auth_hash,
            "matches": matches,
        }
        if not matches:
            blockers.append(f"scorer_input_array_sha256_mismatch:{name}")
        auth_shape = auth_shape_map.get(name)
        cache_shape = cache_shape_map.get(name)
        if auth_shape is not None or cache_shape is not None:
            shape_matches = bool(cache_shape and auth_shape and cache_shape == auth_shape)
            compared_shapes[name] = {
                "cache": cache_shape,
                "auth": auth_shape,
                "matches": shape_matches,
            }
            if not shape_matches:
                blockers.append(f"scorer_input_shape_mismatch:{name}")

    passed = not blockers
    return {
        "equation_id": EQUATION_ID,
        "verdict": "PASS_SCORER_INPUT_CACHE_IDENTITY" if passed else "FAIL_SCORER_INPUT_CACHE_IDENTITY",
        "identity_residual": len(blockers),
        "blockers": blockers,
        "hash_domain": {
            "cache": _clean(cache_hash_domain),
            "auth": _clean(auth_hash_domain),
            "matches": _clean(cache_hash_domain) == _clean(auth_hash_domain),
        },
        "compared_scorer_input_hashes": compared_hashes,
        "compared_scorer_input_shapes": compared_shapes,
        "eligible_for_local_mlx_transfer_calibration": passed,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }


def build_scorer_input_cache_hash_identity_v1() -> CanonicalEquation:
    """Build the scorer-input identity equation with the first local anchor."""

    anchor = EmpiricalAnchor(
        anchor_id="fec6_pr101_full_cache_vs_streaming_hash_identity_20260521",
        measurement_utc="2026-05-21T21:12:00Z",
        inputs={
            "archive_sha256": "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
            "inflated_outputs_aggregate_sha256": "dbc67c898ecb158912f86c920f09bf2c68307b77c1cec3c1baa27a845d3850f1",
            "raw_sha256": "d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c",
            "pair_count": 600,
            "streaming_batch_pairs": 16,
        },
        predicted_output={
            "full_cache_raw_hash_equals_streaming_hash_manifest_raw_hash": True,
            "full_cache_hash_domain_equals_streaming_hash_manifest_hash_domain": True,
            "full_cache_array_hashes_equal_streaming_hash_only_array_hashes": True,
            "full_cache_shapes_equal_streaming_hash_manifest_shapes": True,
        },
        empirical_output={
            "segnet_last_rgb_sha256": "ea4cf2c4879fcdf4cd177cc4e3c762433aa076b631ce252947372cda4da37536",
            "posenet_yuv6_pair_sha256": "aae96b7cb270059174d987740a95e9fd0d9f4474142fd77ed1c1fce6a4124ed0",
            "pair_indices_sha256": "b5d8a47e63045d3032bdc9da91c26e221e453a89f13c94049c6f5e850e49ba81",
            "full_vs_streaming_raw_hash_match": True,
            "full_vs_streaming_hash_domain_match": True,
            "full_vs_streaming_hash_match": True,
            "full_vs_streaming_shape_match": True,
            "elapsed_seconds": 3.47,
        },
        residual=0.0,
        source_artifact=".omx/research/codex_findings_mlx_streaming_hash_cache_modal_bridge_20260521T211200Z_codex.md",
        measurement_method="local_full_cache_manifest_vs_streaming_hash_only_manifest_hash_domain_parity",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/codex_findings_mlx_streaming_hash_cache_modal_bridge_20260521T211200Z_codex.md",
            reactivation_criteria="rerun on Modal Linux contest-CPU scorer_input_cache_hashes.json and compare against local MLX cache",
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="darwin_arm64_m5_max",
            captured_at_utc="2026-05-21T21:12:00Z",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Scorer-input cache hash identity",
        one_line_summary=(
            "MLX cache transfer calibration is valid only when archive, inflated raw, raw file, pair count, hash domain, scorer-input hashes, and tensor shapes match."
        ),
        latex_form=(
            r"I_{\text{cache}} = \mathbf{1}[H_A^c=H_A^a \land H_R^c=H_R^a "
            r"\land H_{\text{raw}}^c=H_{\text{raw}}^a \land N_c=N_a "
            r"\land D_X^c=D_X^a \land H_X^c=H_X^a \land \mathrm{shape}(X^c)=\mathrm{shape}(X^a)]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.scorer_input_cache_hash_identity:"
            "scorer_input_cache_hash_identity"
        ),
        domain_of_validity={
            "contexts": [
                "mlx_scorer_input_cache_transfer_calibration",
                "contest_linux_scorer_input_hash_manifest",
                "local_full_cache_vs_hash_only_manifest_parity",
            ],
            "requires_matching_archive_sha256": True,
            "requires_matching_inflated_outputs_aggregate_sha256": True,
            "requires_matching_raw_sha256": True,
            "requires_matching_hash_domain": True,
            "requires_matching_pair_count_or_auth_n_samples": True,
            "requires_matching_scorer_input_array_sha256": (
                "segnet_last_rgb, posenet_yuv6_pair, and pair_indices"
            ),
            "requires_matching_scorer_input_shapes": (
                "segnet_last_rgb, posenet_yuv6_pair, and pair_indices"
            ),
            "prediction_scope": "identity_invariant_no_score_claim",
        },
        units_in={
            "cache_archive_sha256": "sha256_hex",
            "auth_archive_sha256": "sha256_hex",
            "cache_inflated_outputs_aggregate_sha256": "sha256_hex",
            "auth_inflated_outputs_aggregate_sha256": "sha256_hex",
            "cache_raw_sha256": "sha256_hex",
            "auth_raw_sha256": "sha256_hex",
            "cache_hash_domain": "hash_domain_string",
            "auth_hash_domain": "hash_domain_string",
            "cache_pair_count": "int_pairs",
            "auth_n_samples": "int_pairs",
            "cache_array_sha256": "mapping_tensor_name_to_sha256_hex",
            "auth_scorer_input_array_sha256": "mapping_tensor_name_to_sha256_hex",
            "cache_shapes": "mapping_tensor_name_to_int_shape",
            "auth_shapes": "mapping_tensor_name_to_int_shape",
        },
        units_out={
            "identity_residual": "int_mismatch_count",
            "eligible_for_local_mlx_transfer_calibration": "bool",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "fec6_pr101_full_cache_vs_streaming_hash_identity": 0.0,
        },
        last_calibration_utc="2026-05-21T21:12:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.local_acceleration.mlx_cache_audit",
            "tools/audit_mlx_scorer_input_cache.py",
            "experiments.modal_auth_eval_cpu",
        ),
        canonical_producers=(
            "tac.local_acceleration.mlx_preprocess.write_scorer_input_cache_hash_manifest_from_raw_file",
            "tools/build_mlx_scorer_input_cache.py",
            "experiments.contest_auth_eval",
        ),
        provenance=build_provenance_for_predicted(
            model_id="scorer_input_cache_hash_identity.v1",
            inputs_sha256="3150586045f1af79376bdbae8942ef3a5692d1f53bcfaca18a12f71edff39730",
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
            captured_at_utc="2026-05-21T21:12:00Z",
        ),
    )


def _clean(value: str | None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _hashes(value: Mapping[str, str] | None) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        if isinstance(key, str) and isinstance(raw, str) and raw.strip():
            out[key] = raw.strip().lower()
    return out


def _shapes(value: Mapping[str, object] | None) -> dict[str, list[int]]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, list[int]] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not isinstance(raw, (list, tuple)):
            continue
        shape: list[int] = []
        valid = True
        for dim in raw:
            if isinstance(dim, bool) or not isinstance(dim, int):
                valid = False
                break
            shape.append(int(dim))
        if valid:
            out[key] = shape
    return out


__all__ = [
    "EQUATION_ID",
    "build_scorer_input_cache_hash_identity_v1",
    "scorer_input_cache_hash_identity",
]
