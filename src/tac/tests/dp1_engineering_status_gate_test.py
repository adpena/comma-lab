# SPDX-License-Identifier: MIT
"""Regression tests for the DP1 engineering status gate."""

from __future__ import annotations

import importlib.util
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
                "archive_result": {
                    "notes": ["[real-scorer CPU Tier-C delta curves; no score claim]"]
                },
            },
        ),
    }


def test_dp1_gate_reports_current_proxy_status_and_next_source_gate(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    status = tool.build_status(**_base_artifacts(tmp_path))

    assert status["engineering_status"] == (
        "implemented_proxy_ready_untrained_real_prior_missing"
    )
    assert status["classification"] == "untrained_unpromoted_promising_substrate"
    assert status["false_claims"] == []
    assert status["capabilities_confirmed"]["smoke_archive_materialized"] is True
    assert status["capabilities_confirmed"]["tiny_full_cpu_advisory_ran"] is True
    assert status["capabilities_confirmed"]["real_dataset_source_ready"] is False
    assert status["next_gate"]["id"] == (
        "dp1_comma2k19_onechunk_cpu_advisory_source_custody"
    )
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
                    "chunk_count": 1,
                    "covered_count": 1,
                    "complete": True,
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
