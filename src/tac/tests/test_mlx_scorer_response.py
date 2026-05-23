# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES
from tac.local_acceleration.mlx_preprocess import ScorerInputBatch, write_scorer_input_cache
from tac.local_acceleration.mlx_scorer_response import (
    AUDIT_STAMP_DEREFERENCE_BLOCKER,
    BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER,
    CACHE_INTEGRITY_BLOCKER,
    CANDIDATE_CACHE_TRANSFER_BLOCKER,
    GPU_RESEARCH_SIGNAL_BLOCKER,
    LOCAL_ADVISORY_CACHE_IDENTITY_BLOCKER,
    build_mlx_scorer_response_payload,
    load_scorer_input_cache,
)

REPO = Path(__file__).resolve().parents[3]


def test_mlx_scorer_response_cache_cli_is_non_authoritative(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(tmp_path / "reference", seg=seg, pose=pose, pair_indices=pair_indices)
    candidate_dir = _write_test_cache(tmp_path / "candidate", seg=seg, pose=pose, pair_indices=pair_indices)
    output = tmp_path / "mlx_response.json"
    archive_size_bytes = 1000

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            str(archive_size_bytes),
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected_rate_score = 25.0 * archive_size_bytes / ORIGINAL_VIDEO_BYTES
    assert stdout["score_claim"] is False
    assert stdout["promotable"] is False
    assert payload["score_claim"] is False
    assert payload["promotable"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["n_samples"] == 1
    assert payload["avg_posenet_dist"] == 0.0
    assert payload["avg_segnet_dist"] == 0.0
    assert abs(payload["canonical_score"] - expected_rate_score) < 1.0e-12
    assert payload["cache_identity"]["pair_indices_equal"] is True
    assert payload["cache_integrity"]["reference"]["passed"] is True
    assert payload["cache_integrity"]["candidate"]["passed"] is True
    assert payload["cache_identity"]["candidate"]["cache_integrity"]["passed"] is True


def test_load_scorer_input_cache_rejects_hash_only_manifest(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hash_only"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(
        json.dumps({"hash_only": True}) + "\n",
        encoding="utf-8",
    )

    try:
        load_scorer_input_cache(cache_dir)
    except ValueError as exc:
        assert "hash-only" in str(exc)
    else:
        raise AssertionError("hash-only cache was accepted")


def test_mlx_scorer_response_cli_rejects_unaudited_candidate_cache(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(
        tmp_path / "reference",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    candidate_dir = _write_test_cache(
        tmp_path / "candidate",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
        audited=False,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert CANDIDATE_CACHE_TRANSFER_BLOCKER in completed.stderr


def test_mlx_scorer_response_cli_rejects_missing_auth_audit_stamp_file(
    tmp_path: Path,
) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(
        tmp_path / "reference",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    candidate_dir = _write_test_cache(
        tmp_path / "candidate",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    audit_path = candidate_dir / "auth_eval_identity_audit.json"
    audit_path.unlink()

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert CANDIDATE_CACHE_TRANSFER_BLOCKER in completed.stderr
    assert AUDIT_STAMP_DEREFERENCE_BLOCKER in completed.stderr
    assert "auth_eval_identity_audit_path_not_found" in completed.stderr


def test_mlx_scorer_response_cli_rejects_stale_auth_audit_cache_identity(
    tmp_path: Path,
) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(
        tmp_path / "reference",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    candidate_dir = _write_test_cache(
        tmp_path / "candidate",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    audit_path = candidate_dir / "auth_eval_identity_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["cache"]["archive_sha256"] = "z" * 64
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_stamp_sha(candidate_dir / "manifest.json", "auth_eval_identity_audit", audit_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert AUDIT_STAMP_DEREFERENCE_BLOCKER in completed.stderr
    assert "auth_eval_identity_audit_audit_cache_archive_sha256_mismatch" in completed.stderr


def test_mlx_scorer_response_local_advisory_identity_has_limited_allowed_uses(
    tmp_path: Path,
) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(
        tmp_path / "reference",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    candidate_dir = _write_test_cache(
        tmp_path / "candidate",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
        audited=False,
    )
    _stamp_local_cpu_advisory_identity(candidate_dir)
    output = tmp_path / "mlx_response.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
            "--allow-local-cpu-advisory-cache-identity",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    contract = payload["device_contract"]
    candidate_identity = payload["cache_identity"]["candidate"]
    assert stdout["score_claim"] is False
    assert contract["candidate_cache_identity_mode"] == "local_cpu_advisory_identity"
    assert candidate_identity["candidate_cache_identity_mode"] == "local_cpu_advisory_identity"
    assert candidate_identity["eligible_for_local_mlx_local_advisory_debug"] is True
    assert (
        candidate_identity["local_cpu_advisory_cache_identity_audit"]["verdict"]
        == "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY"
    )
    assert contract["allowed_uses"] == [
        "local_mlx_debug_against_matching_local_cpu_advisory_raw",
        "local_speed_quality_delta_measurement",
    ]
    assert "prepaid_dispatch_spend_filter_after_score_calibration" not in contract["allowed_uses"]


def test_mlx_scorer_response_from_local_advisory_cli_uses_advisory_archive_size(
    tmp_path: Path,
) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(
        tmp_path / "reference",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    candidate_dir = _write_test_cache(
        tmp_path / "candidate",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
        audited=False,
    )
    _stamp_local_cpu_advisory_identity(candidate_dir)
    advisory = tmp_path / "local_cpu_advisory.json"
    advisory.write_text(
        json.dumps(
            {
                "score_axis": "cpu_advisory",
                "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
                "archive_size_bytes": 1234,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "promotable": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "mlx_response.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_from_local_advisory.py"),
            "--local-cpu-advisory",
            str(advisory),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
            "--device",
            "cpu",
            "--allow-local-cpu-advisory-cache-identity",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected_rate_score = 25.0 * 1234 / ORIGINAL_VIDEO_BYTES
    assert stdout["archive_size_bytes"] == 1234
    assert payload["source_local_cpu_advisory"]["path"] == str(advisory)
    assert payload["source_local_cpu_advisory"]["score_claim"] is False
    assert payload["score_claim"] is False
    assert abs(payload["canonical_score"] - expected_rate_score) < 1.0e-12


def test_mlx_scorer_response_local_advisory_stamp_must_dereference(
    tmp_path: Path,
) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(
        tmp_path / "reference",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    candidate_dir = _write_test_cache(
        tmp_path / "candidate",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
        audited=False,
    )
    _stamp_local_cpu_advisory_identity(candidate_dir)
    (candidate_dir / "local_cpu_advisory_cache_identity_audit.json").write_text(
        "tampered\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
            "--allow-local-cpu-advisory-cache-identity",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert LOCAL_ADVISORY_CACHE_IDENTITY_BLOCKER in completed.stderr
    assert AUDIT_STAMP_DEREFERENCE_BLOCKER in completed.stderr
    assert "local_cpu_advisory_cache_identity_audit_sha256_mismatch" in completed.stderr


def test_mlx_scorer_response_cli_rejects_mutated_cache_after_manifest_stamp(
    tmp_path: Path,
) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(
        tmp_path / "reference",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    candidate_dir = _write_test_cache(
        tmp_path / "candidate",
        seg=seg,
        pose=pose,
        pair_indices=pair_indices,
    )
    np.save(candidate_dir / "segnet_last_rgb.npy", np.ones_like(seg))

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert CACHE_INTEGRITY_BLOCKER in completed.stderr
    assert "array_sha256_segnet_last_rgb_mismatch" in completed.stderr
    assert "artifact_segnet_last_rgb_sha256_mismatch" in completed.stderr


def test_mlx_scorer_response_cli_can_score_deterministic_pair_window(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    seg = np.zeros((2, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((2, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(tmp_path / "reference", seg=seg, pose=pose, pair_indices=pair_indices)
    candidate_dir = _write_test_cache(tmp_path / "candidate", seg=seg, pose=pose, pair_indices=pair_indices)
    output = tmp_path / "mlx_response_window.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1000",
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
            "--start-pair",
            "1",
            "--max-pairs",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["n_samples"] == 1
    assert payload["n_samples"] == 1
    assert payload["total_cache_pairs"] == 2
    assert payload["start_pair"] == 1
    assert payload["max_pairs"] == 1
    assert payload["pair_window"] == [1, 2]
    assert payload["avg_posenet_dist"] == 0.0
    assert payload["avg_segnet_dist"] == 0.0


def test_mlx_scorer_response_rejects_invalid_pair_window_before_loading() -> None:
    try:
        build_mlx_scorer_response_payload(
            reference_cache_dir="/does/not/exist/reference",
            candidate_cache_dir="/does/not/exist/candidate",
            archive_size_bytes=1,
            max_pairs=0,
        )
    except ValueError as exc:
        assert "max_pairs must be >= 1" in str(exc)
    else:
        raise AssertionError("invalid max_pairs was accepted")


def test_mlx_scorer_response_rejects_gpu_without_explicit_research_allowance() -> None:
    try:
        build_mlx_scorer_response_payload(
            reference_cache_dir="/does/not/exist/reference",
            candidate_cache_dir="/does/not/exist/candidate",
            archive_size_bytes=1,
            device_type="gpu",
        )
    except ValueError as exc:
        assert GPU_RESEARCH_SIGNAL_BLOCKER in str(exc)
        assert "allow_gpu_research_signal=True" in str(exc)
    else:
        raise AssertionError("MLX GPU scorer-response path was accepted without explicit allowance")


def test_mlx_scorer_response_rejects_non_singleton_cpu_batch_without_research_allowance() -> None:
    try:
        build_mlx_scorer_response_payload(
            reference_cache_dir="/does/not/exist/reference",
            candidate_cache_dir="/does/not/exist/candidate",
            archive_size_bytes=1,
            device_type="cpu",
            batch_pairs=2,
        )
    except ValueError as exc:
        assert BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER in str(exc)
        assert "batch_pairs=1" in str(exc)
    else:
        raise AssertionError("MLX CPU non-singleton batch was accepted without research allowance")


def test_mlx_scorer_response_cli_rejects_gpu_without_explicit_research_allowance(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    seg = np.zeros((1, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((1, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(tmp_path / "reference", seg=seg, pose=pose, pair_indices=pair_indices)
    candidate_dir = _write_test_cache(tmp_path / "candidate", seg=seg, pose=pose, pair_indices=pair_indices)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
            "--device",
            "gpu",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert GPU_RESEARCH_SIGNAL_BLOCKER in completed.stderr


def test_mlx_scorer_response_rejects_non_singleton_gpu_batch_after_allowance() -> None:
    try:
        build_mlx_scorer_response_payload(
            reference_cache_dir="/does/not/exist/reference",
            candidate_cache_dir="/does/not/exist/candidate",
            archive_size_bytes=1,
            device_type="gpu",
            batch_pairs=2,
            allow_gpu_research_signal=True,
        )
    except ValueError as exc:
        assert BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER in str(exc)
        assert "batch_pairs=1" in str(exc)
    else:
        raise AssertionError("MLX GPU non-singleton batch was accepted without invariance override")


def test_mlx_scorer_response_cli_rejects_non_singleton_gpu_batch_after_allowance(tmp_path: Path) -> None:
    pair_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    seg = np.zeros((2, 3, 64, 80), dtype=np.float32)
    pose = np.zeros((2, 12, 64, 80), dtype=np.float32)
    reference_dir = _write_test_cache(tmp_path / "reference", seg=seg, pose=pose, pair_indices=pair_indices)
    candidate_dir = _write_test_cache(tmp_path / "candidate", seg=seg, pose=pose, pair_indices=pair_indices)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "run_mlx_scorer_response_cache.py"),
            "--reference-cache-dir",
            str(reference_dir),
            "--candidate-cache-dir",
            str(candidate_dir),
            "--archive-size-bytes",
            "1",
            "--output",
            str(tmp_path / "mlx_response.json"),
            "--repo-root",
            str(REPO),
            "--device",
            "gpu",
            "--allow-gpu-research-signal",
            "--batch-pairs",
            "2",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert BATCH_SHAPE_RESEARCH_SIGNAL_BLOCKER in completed.stderr


def _write_test_cache(
    path: Path,
    *,
    seg: np.ndarray,
    pose: np.ndarray,
    pair_indices: np.ndarray,
    audited: bool = True,
) -> Path:
    metadata = {
        "schema_version": "mlx_scorer_input_cache.v1",
        "pair_count": int(pair_indices.shape[0]),
        "segnet_last_rgb_shape": list(seg.shape),
        "posenet_yuv6_pair_shape": list(pose.shape),
        "pair_indices_shape": list(pair_indices.shape),
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if audited:
        metadata.update(
            {
                "eligible_for_local_mlx_transfer_calibration": True,
                "auth_eval_identity_audit": {
                    "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
                    "verdict": "PASS_CACHE_AUTH_EVAL_IDENTITY",
                    "passed": True,
                    "identity_residual": 0,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
            }
        )
    batch = ScorerInputBatch(
        segnet_last_rgb=seg,
        posenet_yuv6_pair=pose,
        pair_indices=pair_indices,
        metadata=metadata,
    )
    write_scorer_input_cache(
        batch,
        path,
        archive_sha256="a" * 64,
        inflated_outputs_aggregate_sha256="b" * 64,
        raw_sha256="c" * 64,
    )
    if audited:
        _stamp_auth_eval_identity(path)
    return path


def _stamp_auth_eval_identity(path: Path) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_path = path / "auth_eval_identity_audit.json"
    audit = {
        "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
        "verdict": "PASS_CACHE_AUTH_EVAL_IDENTITY",
        "passed": True,
        "identity_residual": 0,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cache": {
            "archive_sha256": manifest["archive_sha256"],
            "inflated_outputs_aggregate_sha256": manifest[
                "inflated_outputs_aggregate_sha256"
            ],
            "raw_sha256": manifest["raw_sha256"],
            "pair_count": manifest["pair_count"],
            "hash_domain": manifest["hash_domain"],
            "segnet_last_rgb_shape": manifest["segnet_last_rgb_shape"],
            "posenet_yuv6_pair_shape": manifest["posenet_yuv6_pair_shape"],
            "pair_indices_shape": manifest["pair_indices_shape"],
            "array_sha256": manifest["array_sha256"],
        },
    }
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    manifest["eligible_for_local_mlx_transfer_calibration"] = True
    manifest["auth_eval_identity_audit"] = {
        "schema_version": audit["schema_version"],
        "path": str(audit_path),
        "sha256": _file_sha256(audit_path),
        "verdict": audit["verdict"],
        "passed": True,
        "identity_residual": 0,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def _stamp_local_cpu_advisory_identity(path: Path) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_path = path / "local_cpu_advisory_cache_identity_audit.json"
    audit = {
        "schema_version": "mlx_scorer_input_cache_local_cpu_advisory_audit.v1",
        "verdict": "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
        "passed": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cache": {
            "archive_sha256": manifest["archive_sha256"],
            "inflated_outputs_aggregate_sha256": manifest[
                "inflated_outputs_aggregate_sha256"
            ],
            "raw_sha256": manifest["raw_sha256"],
            "pair_count": manifest["pair_count"],
            "hash_domain": manifest["hash_domain"],
            "array_sha256": manifest["array_sha256"],
        },
    }
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    manifest["eligible_for_local_mlx_local_advisory_debug"] = True
    manifest["eligible_for_local_mlx_transfer_calibration"] = False
    manifest["local_cpu_advisory_cache_identity_audit"] = {
        "schema_version": audit["schema_version"],
        "path": str(audit_path),
        "sha256": _file_sha256(audit_path),
        "verdict": audit["verdict"],
        "passed": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_stamp_sha(manifest_path: Path, stamp_key: str, audit_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[stamp_key]["sha256"] = _file_sha256(audit_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
