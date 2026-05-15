# SPDX-License-Identifier: MIT
"""Regression tests for the DP1 engineering status gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_tool() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "dp1_engineering_status_gate.py"
    spec = importlib.util.spec_from_file_location("dp1_engineering_status_gate", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _base_artifacts(tmp_path: Path) -> dict[str, Path]:
    return {
        "readiness_path": _write_json(
            tmp_path / "readiness.json",
            {
                "schema": "tac_2032_driving_prior_readiness.v1",
                "training_started": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
        "smoke_manifest_path": _write_json(
            tmp_path / "smoke_manifest.json",
            {
                "schema": "dp1_readiness_manifest_v1",
                "archive_path": "archive.zip",
                "archive_bytes": 12032,
                "evidence_grade": "[proxy]",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": ["contest_cuda_eval_not_run"],
            },
        ),
        "tiny_full_manifest_path": _write_json(
            tmp_path / "tiny_manifest.json",
            {
                "schema": "dp1_readiness_manifest_v1",
                "archive_path": "archive.zip",
                "archive_bytes": 25914,
                "evidence_grade": "[proxy]",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        ),
        "tiny_full_provenance_path": _write_json(
            tmp_path / "provenance.json",
            {
                "schema": "dpp_phase_2_provenance_v1",
                "trainer": "experiments/train_substrate_pretrained_driving_prior.py",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "codebook_provenance": {
                    "dataset_provenance": "synthetic_test",
                    "license_tags": ["synthetic-test-only"],
                },
            },
        ),
        "tier_c_path": _write_json(
            tmp_path / "tier_c.json",
            {
                "schema": "tac_tier_c_real_scorer_archive_result_v1",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "archive_result": {"notes": ["[real-scorer CPU Tier-C delta curves; no score claim]"]},
            },
        ),
    }


def test_dp1_gate_reports_current_proxy_status_and_next_source_gate(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    status = tool.build_status(**_base_artifacts(tmp_path))

    assert status["engineering_status"] == ("implemented_proxy_ready_untrained_real_prior_missing")
    assert status["classification"] == "untrained_unpromoted_promising_substrate"
    assert status["false_claims"] == []
    assert status["capabilities_confirmed"]["smoke_archive_materialized"] is True
    assert status["capabilities_confirmed"]["tiny_full_cpu_advisory_ran"] is True
    assert status["capabilities_confirmed"]["real_dataset_source_ready"] is False
    assert status["next_gate"]["id"] == ("dp1_comma2k19_onechunk_cpu_advisory_source_custody")
    assert status["next_gate"]["status"] == "blocked_until_source_supplied"
    assert "dp1_real_dataset_source_manifest_missing" in status["blockers"]


def test_dp1_gate_fails_closed_on_score_claim_without_score(tmp_path: Path) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    _write_json(
        paths["smoke_manifest_path"],
        {
            "schema": "dp1_readiness_manifest_v1",
            "archive_path": "archive.zip",
            "archive_bytes": 12032,
            "evidence_grade": "[proxy]",
            "score_claim": True,
            "score_claim_valid": True,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    status = tool.build_status(**paths)

    assert status["engineering_status"] == "blocked_false_dp1_score_or_promotion_claim"
    assert len(status["false_claims"]) == 2
    assert all("without contest score field" in item for item in status["false_claims"])


def test_dp1_gate_rejects_proxy_canonical_score_as_exact_score_claim(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    _write_json(
        paths["smoke_manifest_path"],
        {
            "schema": "dp1_readiness_manifest_v1",
            "archive_path": "archive.zip",
            "archive_bytes": 12032,
            "canonical_score": 0.91,
            "evidence_grade": "[proxy]",
            "score_axis": "proxy",
            "score_claim": True,
            "score_claim_valid": True,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    status = tool.build_status(**paths)

    assert status["engineering_status"] == "blocked_false_dp1_score_or_promotion_claim"
    assert status["capabilities_confirmed"]["exact_score_artifact_present"] is False
    assert any("score_claim=true without contest score field" in item for item in status["false_claims"])
    assert any("score_claim_valid=true without contest score field" in item for item in status["false_claims"])


def test_dp1_gate_requires_authoritative_axis_metadata_for_generic_scores(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    _write_json(
        paths["smoke_manifest_path"],
        {
            "schema": "dp1_readiness_manifest_v1",
            "archive_path": "archive.zip",
            "archive_bytes": 12032,
            "canonical_score": 0.91,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    status = tool.build_status(**paths)

    assert status["capabilities_confirmed"]["exact_score_artifact_present"] is False
    assert status["next_gate"]["id"] != "dp1_result_review_packet"


def test_dp1_gate_accepts_authoritative_contest_axis_score_metadata(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    _write_json(
        paths["smoke_manifest_path"],
        {
            "schema": "dp1_readiness_manifest_v1",
            "archive_path": "archive.zip",
            "archive_bytes": 12032,
            "canonical_score": 0.91,
            "evidence_grade": "[contest-CUDA]",
            "score_axis": "contest_cuda",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    status = tool.build_status(**paths)

    assert status["capabilities_confirmed"]["exact_score_artifact_present"] is True


def test_dp1_gate_rejects_macos_cpu_advisory_score_as_exact_evidence(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    _write_json(
        paths["smoke_manifest_path"],
        {
            "schema": "dp1_readiness_manifest_v1",
            "archive_path": "archive.zip",
            "archive_bytes": 12032,
            "canonical_score": 0.91,
            "evidence_grade": "[macOS-CPU advisory]",
            "score_axis": "cpu_advisory",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    status = tool.build_status(**paths)

    assert status["capabilities_confirmed"]["exact_score_artifact_present"] is False


def test_dp1_gate_recognizes_complete_real_comma2k19_source_manifest(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    _write_json(
        paths["tiny_full_provenance_path"],
        {
            "schema": "dpp_phase_2_provenance_v1",
            "trainer": "experiments/train_substrate_pretrained_driving_prior.py",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dataset_source_manifest": {
                "schema": "dp1_dataset_source_manifest.v1",
                "dataset_name": "comma2k19",
                "source_mode": "local_chunks",
                "chunk_ids": ["Chunk_A/video.hevc"],
                "chunk_sha256_manifest": {"Chunk_A/video.hevc": "a" * 64},
                "chunk_sha256_coverage": {
                    "scope": "selected_chunks_only",
                    "chunk_count": 1,
                    "covered_count": 1,
                    "complete_for_selected_chunks": True,
                    "full_dataset_complete": False,
                },
                "reproducibility_blockers": [],
            },
        },
    )

    status = tool.build_status(**paths)

    assert status["capabilities_confirmed"]["real_dataset_source_ready"] is True
    assert status["engineering_status"] == "real_source_probe_ready_no_exact_score"
    assert status["next_gate"]["status"] == "executable_real_source_probe"
    assert status["next_gate"]["blocked_by"] == []


def test_dp1_gate_preflights_real_local_chunk_custody_without_private_paths(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    chunks_root = tmp_path / "private" / "comma2k19" / "test_videos"
    video = chunks_root / "dongle" / "route" / "40" / "video.hevc"
    video.parent.mkdir(parents=True)
    payload = b"real-ish dp1 chunk bytes"
    video.write_bytes(payload)

    status = tool.build_status(
        **paths,
        source_chunks_dir=chunks_root,
        source_max_chunks=1,
    )

    custody = status["source_custody_preflight"]
    expected_sha = hashlib.sha256(payload).hexdigest()
    assert custody["status"] == "passed"
    assert custody["chunk_count"] == 1
    assert custody["total_size_bytes"] == len(payload)
    assert custody["chunk_manifest"] == [
        {
            "relative_path": "dongle/route/40/video.hevc",
            "size_bytes": len(payload),
            "sha256": expected_sha,
        }
    ]
    assert custody["path_kind"] == "relative_to_DPP_COMMA2K19_CHUNKS_DIR"
    coverage = status["dataset_source_manifest"]["chunk_sha256_coverage"]
    assert coverage["scope"] == "selected_chunks_only"
    assert coverage["complete_for_selected_chunks"] is True
    assert coverage["full_dataset_complete"] is False
    assert str(chunks_root) not in json.dumps(custody, sort_keys=True)
    assert str(chunks_root) not in json.dumps(status["dataset_source_manifest"], sort_keys=True)
    assert status["capabilities_confirmed"]["real_dataset_source_ready"] is True
    assert status["engineering_status"] == "real_source_probe_ready_no_exact_score"
    assert status["next_gate"]["status"] == "executable_real_source_probe"
    assert "$DPP_COMMA2K19_CHUNKS_DIR" in status["next_gate"]["commands"][0]
    assert str(chunks_root) not in json.dumps(status["next_gate"], sort_keys=True)


def test_dp1_gate_fails_closed_when_requested_real_chunks_dir_is_empty(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    chunks_root = tmp_path / "empty_chunks"
    chunks_root.mkdir()
    _write_json(
        paths["tiny_full_provenance_path"],
        {
            "schema": "dpp_phase_2_provenance_v1",
            "trainer": "experiments/train_substrate_pretrained_driving_prior.py",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dataset_source_manifest": {
                "schema": "dp1_dataset_source_manifest.v1",
                "dataset_name": "comma2k19",
                "source_mode": "local_chunks",
                "chunk_ids": ["stale/video.hevc"],
                "chunk_sha256_manifest": {"stale/video.hevc": "b" * 64},
                "chunk_sha256_coverage": {
                    "scope": "selected_chunks_only",
                    "chunk_count": 1,
                    "covered_count": 1,
                    "complete_for_selected_chunks": True,
                    "full_dataset_complete": False,
                },
                "reproducibility_blockers": [],
            },
        },
    )

    status = tool.build_status(
        **paths,
        source_chunks_dir=chunks_root,
        source_max_chunks=1,
    )

    assert status["source_custody_preflight"]["status"] == "blocked"
    assert "dp1_real_chunks_missing_video_hevc" in status["blockers"]
    assert status["capabilities_confirmed"]["real_dataset_source_ready"] is False
    assert status["next_gate"]["status"] == "blocked_until_source_supplied"


def test_dp1_gate_redacts_private_paths_from_existing_source_manifest(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    paths = _base_artifacts(tmp_path)
    private_root = tmp_path / "private" / "chunks"
    _write_json(
        paths["tiny_full_provenance_path"],
        {
            "schema": "dpp_phase_2_provenance_v1",
            "trainer": "experiments/train_substrate_pretrained_driving_prior.py",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dataset_source_manifest": {
                "schema": "dp1_dataset_source_manifest.v1",
                "dataset_name": "comma2k19",
                "source_mode": "local_chunks",
                "chunks_dir": str(private_root),
                "cache_dir": str(tmp_path / "private" / "cache"),
                "stream_log_dir": str(tmp_path / "private" / "stream_logs"),
                "codebook_path": str(tmp_path / "private" / "codebook.bin"),
                "chunk_ids": ["x/video.hevc"],
                "chunk_sha256_manifest": {"x/video.hevc": "c" * 64},
                "chunk_sha256_coverage": {
                    "scope": "selected_chunks_only",
                    "chunk_count": 1,
                    "covered_count": 1,
                    "complete_for_selected_chunks": True,
                    "full_dataset_complete": False,
                },
                "reproducibility_blockers": [],
            },
        },
    )

    status = tool.build_status(**paths)
    manifest_json = json.dumps(status["dataset_source_manifest"], sort_keys=True)

    assert status["capabilities_confirmed"]["real_dataset_source_ready"] is True
    assert str(tmp_path) not in manifest_json
    assert status["dataset_source_manifest"]["chunks_dir"] == "<redacted:chunks_dir:set>"
    assert status["dataset_source_manifest"]["cache_dir"] == "<redacted:cache_dir:set>"
    assert status["dataset_source_manifest"]["stream_log_dir"] == ("<redacted:stream_log_dir:set>")
    assert status["dataset_source_manifest"]["codebook_path"] == ("<redacted:codebook_path:set>")
