# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.local_acceleration.mlx_cache_audit import (
    FAIL_VERDICT,
    PASS_VERDICT,
    audit_mlx_scorer_input_cache_against_auth_eval,
)

REPO = Path(__file__).resolve().parents[3]
HASH_DOMAIN = "_array_sha256(dtype_string + json_shape + contiguous_bytes)"
ARRAY_HASHES = {
    "segnet_last_rgb": "s" * 64,
    "posenet_yuv6_pair": "p" * 64,
    "pair_indices": "i" * 64,
}


def _cache(*, aggregate: str = "b" * 64, pair_count: int = 600) -> dict[str, object]:
    return {
        "schema_version": "mlx_scorer_input_cache.v1",
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": aggregate,
        "raw_sha256": "r" * 64,
        "hash_domain": HASH_DOMAIN,
        "pair_count": pair_count,
        "segnet_last_rgb_shape": [pair_count, 3, 384, 512],
        "posenet_yuv6_pair_shape": [pair_count, 12, 192, 256],
        "pair_indices_shape": [pair_count, 2],
        "array_sha256": dict(ARRAY_HASHES),
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
    }


def _auth(
    *,
    aggregate: str = "b" * 64,
    n_samples: int = 600,
    evidence_grade: str = "contest-CPU",
    score_axis: str = "contest_cpu",
) -> dict[str, object]:
    archive_size = 178_517
    seg = 0.00056029
    pose = 0.00002943
    return {
        "canonical_score": contest_formula_score(
            seg_dist=seg,
            pose_dist=pose,
            archive_bytes=archive_size,
        ),
        "canonical_score_source": "score_recomputed_from_components",
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_size,
        "score_rate_contribution": 25.0 * archive_size / ORIGINAL_VIDEO_BYTES,
        "rate_unscaled": archive_size / ORIGINAL_VIDEO_BYTES,
        "n_samples": n_samples,
        "evidence_grade": evidence_grade,
        "lane_tag": "[contest-CPU]",
        "score_axis": score_axis,
        "evidence_semantics": "public_leaderboard_cpu_reproduction",
        "exact_cuda_eval_complete": False,
        "score_claim": True,
        "score_claim_valid": True,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "cpu_leaderboard_reproduction_eligible": True,
        "provenance": {
            "device": "cpu",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "archive_sha256": "a" * 64,
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": aggregate,
                    "files": [{"sha256": "r" * 64}],
                }
            },
        },
    }


def _auth_with_scorer_input_hash(
    *,
    array_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    auth = _auth()
    provenance = auth["provenance"]
    assert isinstance(provenance, dict)
    provenance["scorer_input_cache_hash_manifest"] = {
        "payload": {
            "schema_version": "mlx_scorer_input_cache_hashes.v1",
            "hash_domain": HASH_DOMAIN,
            "raw_sha256": "r" * 64,
            "pair_count": 600,
            "segnet_last_rgb_shape": [600, 3, 384, 512],
            "posenet_yuv6_pair_shape": [600, 12, 192, 256],
            "pair_indices_shape": [600, 2],
            "array_sha256": dict(array_hashes or ARRAY_HASHES),
        }
    }
    return auth


def _auth_with_scorer_input_tensor_manifest(
    *,
    array_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    auth = _auth()
    provenance = auth["provenance"]
    assert isinstance(provenance, dict)
    provenance["scorer_input_cache_tensor_manifest"] = {
        "payload": {
            "schema_version": "mlx_scorer_input_cache.v1",
            "archive_sha256": "a" * 64,
            "inflated_outputs_aggregate_sha256": "b" * 64,
            "raw_sha256": "r" * 64,
            "hash_domain": HASH_DOMAIN,
            "pair_count": 600,
            "segnet_last_rgb_shape": [600, 3, 384, 512],
            "posenet_yuv6_pair_shape": [600, 12, 192, 256],
            "pair_indices_shape": [600, 2],
            "array_sha256": dict(array_hashes or ARRAY_HASHES),
            "score_claim": False,
            "promotion_eligible": False,
            "promotable": False,
        }
    }
    return auth


def _reference_manifest(*, array_hashes: dict[str, str] | None = None) -> dict[str, object]:
    return {
        "schema_version": "mlx_scorer_input_cache_hashes.v1",
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "raw_sha256": "r" * 64,
        "hash_domain": HASH_DOMAIN,
        "pair_count": 600,
        "segnet_last_rgb_shape": [600, 3, 384, 512],
        "posenet_yuv6_pair_shape": [600, 12, 192, 256],
        "pair_indices_shape": [600, 2],
        "array_sha256": dict(array_hashes or ARRAY_HASHES),
    }


def test_cache_audit_passes_matching_identity() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        _cache(),
        _auth_with_scorer_input_hash(),
    )
    assert audit["passed"] is True
    assert audit["verdict"] == PASS_VERDICT
    assert audit["score_claim"] is False
    assert audit["score_claim_valid"] is False
    assert audit["promotion_eligible"] is False
    assert audit["cache"]["hash_domain"] == HASH_DOMAIN
    assert audit["cache"]["pair_indices_shape"] == [600, 2]
    assert "local_mlx_training_transfer_calibration" in audit["allowed_use"]
    assert audit["auth_eval_contract"]["evidence_grade"] == "contest-CPU"
    assert audit["auth_eval_contract"]["score_axis"] == "contest_cpu"


def test_cache_audit_fails_inflated_aggregate_mismatch() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        _cache(aggregate="c" * 64),
        _auth_with_scorer_input_hash(),
    )
    assert audit["passed"] is False
    assert audit["verdict"] == FAIL_VERDICT
    assert "inflated_outputs_aggregate_sha256_mismatch_or_missing" in audit["blockers"]
    assert "do_not_use_for_auth_axis_transfer_calibration" in audit["allowed_use"]


def test_cache_audit_fails_pair_count_mismatch() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        _cache(pair_count=16),
        _auth_with_scorer_input_hash(),
    )
    assert audit["passed"] is False
    assert "cache_pair_count_mismatch:cache=16:expected=600" in audit["blockers"]


def test_cache_audit_requires_full_contest_auth_axis_even_with_debug_expected_pair_count() -> None:
    cache = _cache(pair_count=16)
    auth = _auth_with_scorer_input_hash()
    auth["n_samples"] = 16
    provenance = auth["provenance"]
    assert isinstance(provenance, dict)
    scorer_manifest = provenance["scorer_input_cache_hash_manifest"]
    assert isinstance(scorer_manifest, dict)
    payload = scorer_manifest["payload"]
    assert isinstance(payload, dict)
    payload["pair_count"] = 16
    payload["segnet_last_rgb_shape"] = [16, 3, 384, 512]
    payload["posenet_yuv6_pair_shape"] = [16, 12, 192, 256]
    payload["pair_indices_shape"] = [16, 2]

    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        cache,
        auth,
        expected_pair_count=16,
    )

    assert audit["passed"] is False
    assert "n_samples_mismatch:manifest=16:expected=600" in audit["blockers"]


def test_cache_audit_cli_writes_output(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    auth_path = tmp_path / "auth.json"
    out_path = tmp_path / "audit.json"
    cache_path.write_text(json.dumps(_cache()), encoding="utf-8")
    auth_path.write_text(json.dumps(_auth()), encoding="utf-8")
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference_manifest()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_mlx_scorer_input_cache.py"),
            "--cache-manifest",
            str(cache_path),
            "--auth-eval",
            str(auth_path),
            "--output",
            str(out_path),
            "--reference-cache-manifest",
            str(reference_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"passed": true' in completed.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_cache_audit_cli_stamps_cache_manifest_on_pass(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    auth_path = tmp_path / "auth.json"
    out_path = tmp_path / "audit.json"
    reference_path = tmp_path / "reference.json"
    cache_path.write_text(json.dumps(_cache()), encoding="utf-8")
    auth_path.write_text(json.dumps(_auth()), encoding="utf-8")
    reference_path.write_text(json.dumps(_reference_manifest()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_mlx_scorer_input_cache.py"),
            "--cache-manifest",
            str(cache_path),
            "--auth-eval",
            str(auth_path),
            "--output",
            str(out_path),
            "--reference-cache-manifest",
            str(reference_path),
            "--stamp-cache-manifest-on-pass",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(cache_path.read_text(encoding="utf-8"))
    assert manifest["eligible_for_local_mlx_transfer_calibration"] is True
    stamp = manifest["auth_eval_identity_audit"]
    assert stamp["verdict"] == PASS_VERDICT
    assert stamp["passed"] is True
    assert stamp["identity_residual"] == 0
    assert len(stamp["sha256"]) == 64


def test_cache_audit_alias_cli_writes_output(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    auth_path = tmp_path / "auth.json"
    out_path = tmp_path / "audit.json"
    cache_path.write_text(json.dumps(_cache()), encoding="utf-8")
    auth_path.write_text(json.dumps(_auth()), encoding="utf-8")
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(_reference_manifest()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "mlx_cache_audit.py"),
            "--cache-manifest",
            str(cache_path),
            "--auth-eval",
            str(auth_path),
            "--output",
            str(out_path),
            "--reference-cache-manifest",
            str(reference_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"passed": true' in completed.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_cache_audit_uses_canonical_equation_for_scorer_input_hashes() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        _cache(),
        _auth_with_scorer_input_hash(),
    )

    assert audit["passed"] is True
    assert audit["canonical_equation"]["equation_id"] == "scorer_input_cache_hash_identity_v1"
    assert audit["canonical_equation"]["identity_residual"] == 0
    assert audit["eligible_for_local_mlx_transfer_calibration"] is True
    assert audit["identity_residual"] == 0


def test_cache_audit_accepts_auth_eval_tensor_manifest() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        _cache(),
        _auth_with_scorer_input_tensor_manifest(),
    )

    assert audit["passed"] is True
    assert audit["auth_eval"]["scorer_input_hash_reference_source"] == "auth_eval_provenance"
    assert audit["canonical_equation"]["identity_residual"] == 0
    assert audit["eligible_for_local_mlx_transfer_calibration"] is True
    assert audit["identity_residual"] == 0


def test_cache_audit_fails_scorer_input_hash_mismatch_when_auth_provides_hashes() -> None:
    cache = _cache()
    cache["array_sha256"] = {**ARRAY_HASHES, "segnet_last_rgb": "x" * 64}
    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        cache,
        _auth_with_scorer_input_hash(),
    )

    assert audit["passed"] is False
    assert "scorer_input_array_sha256_mismatch:segnet_last_rgb" in audit["blockers"]
    assert audit["canonical_equation"]["identity_residual"] == 1


def test_cache_audit_fails_without_auth_or_reference_scorer_input_hashes() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(_cache(), _auth())

    assert audit["passed"] is False
    assert "auth_scorer_input_array_sha256_missing:segnet_last_rgb" in audit["blockers"]
    assert "auth_scorer_input_array_sha256_missing:posenet_yuv6_pair" in audit["blockers"]
    assert "auth_scorer_input_array_sha256_missing:pair_indices" in audit["blockers"]


def test_cache_audit_accepts_independent_reference_hash_manifest() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        _cache(),
        _auth(),
        reference_cache_manifest=_reference_manifest(),
    )

    assert audit["passed"] is True
    assert audit["auth_eval"]["scorer_input_hash_reference_source"] == "reference_cache_manifest"


def test_cache_audit_fails_undercustodied_reference_manifest() -> None:
    reference = {
        "schema_version": "mlx_scorer_input_cache_hashes.v1",
        "hash_domain": HASH_DOMAIN,
        "array_sha256": dict(ARRAY_HASHES),
    }

    audit = audit_mlx_scorer_input_cache_against_auth_eval(
        _cache(),
        _auth(),
        reference_cache_manifest=reference,
    )

    assert audit["passed"] is False
    assert "reference_archive_sha256_missing" in audit["blockers"]
    assert "reference_inflated_outputs_aggregate_sha256_missing" in audit["blockers"]
    assert "reference_raw_sha256_missing" in audit["blockers"]
    assert "reference_pair_count_missing" in audit["blockers"]
    assert "reference_segnet_last_rgb_shape_missing" in audit["blockers"]
    assert "do_not_use_for_auth_axis_transfer_calibration" in audit["allowed_use"]


def test_cache_audit_rejects_non_contest_auth_eval_axis() -> None:
    bad_auth = _auth_with_scorer_input_hash()
    bad_auth["evidence_grade"] = "B"
    bad_auth["score_axis"] = "diagnostic_cuda"
    bad_auth["score_claim"] = False
    bad_auth["score_claim_valid"] = False
    blocked = audit_mlx_scorer_input_cache_against_auth_eval(_cache(), bad_auth)

    assert blocked["passed"] is False
    assert "auth_eval_evidence_grade_not_contest_cpu_or_cuda" in blocked["blockers"]


def test_cache_audit_requires_auth_axis_matching_grade() -> None:
    bad_auth = _auth_with_scorer_input_hash()
    bad_auth["evidence_grade"] = "contest-CUDA"
    bad_auth["score_axis"] = "contest_cpu"
    audit = audit_mlx_scorer_input_cache_against_auth_eval(_cache(), bad_auth)

    assert audit["passed"] is False
    assert "score_axis_not_contest_cuda" in audit["blockers"]


def test_cache_audit_rejects_forged_contest_cpu_auth_eval_custody() -> None:
    bad_auth = _auth_with_scorer_input_hash()
    provenance = bad_auth["provenance"]
    assert isinstance(provenance, dict)
    provenance["platform_system"] = "Darwin"
    provenance["platform_machine"] = "arm64"

    audit = audit_mlx_scorer_input_cache_against_auth_eval(_cache(), bad_auth)

    assert audit["passed"] is False
    assert "contest_cpu_platform_system_not_linux" in audit["blockers"]
    assert "contest_cpu_platform_machine_not_x86_64" in audit["blockers"]
