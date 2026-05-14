# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from tac.repo_io import read_json

REPO_ROOT = Path(__file__).resolve().parents[3]
ADJUDICATOR = REPO_ROOT / "scripts" / "adjudicate_contest_auth_eval.py"


def _load_adjudicator() -> Any:
    for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tools"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("adjudicate_contest_auth_eval_test", ADJUDICATOR)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(
    *,
    contest_json: Path,
    archive: Path,
    provenance: Path,
    result_copy: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        contest_json=contest_json,
        archive=archive,
        provenance=provenance,
        result_copy=result_copy,
        baseline_score=0.2,
        baseline_archive_bytes=None,
        predicted_band=[0.1, 0.2],
        regression_threshold=1.0,
        hard_kill_above=None,
        delta_key="score_delta_vs_baseline",
        max_posenet_dist=None,
        max_segnet_dist=None,
        baseline_posenet_dist=None,
        baseline_segnet_dist=None,
        max_posenet_relative=None,
        max_segnet_relative=None,
        component_reference_label="baseline",
        required_device="cuda",
        required_samples=600,
        max_sane_score=10.0,
        allow_sane_score_forensic_success=False,
        allow_component_gate_forensic_success=False,
        allow_distillation_gate_forensic_success=False,
    )


def _write_minimal_inputs(tmp_path: Path, payload_extra: dict[str, object]) -> tuple[Path, Path, Path]:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"stored archive bytes")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    payload = {
        "score_recomputed_from_components": 0.18,
        "final_score": 0.18,
        "n_samples": 600,
        "archive_size_bytes": archive.stat().st_size,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.001,
        "provenance": {
            "device": "cuda",
            "archive_sha256": archive_sha,
            "gpu_model": "Tesla T4",
            "gpu_t4_match": True,
        },
        **payload_extra,
    }
    contest_json = tmp_path / "contest_auth_eval.json"
    contest_json.write_text(
        _load_adjudicator().json_text(payload),
        encoding="utf-8",
    )
    provenance = tmp_path / "lane_provenance.json"
    return contest_json, archive, provenance


def test_adjudicator_raw_promotion_blockers_fail_closed(tmp_path: Path) -> None:
    module = _load_adjudicator()
    contest_json, archive, provenance = _write_minimal_inputs(
        tmp_path,
        {
            "promotion_blockers": [
                "cpu_leaderboard_reproduction_not_adjudicated",
                "pre_submission_compliance_check_not_recorded",
            ],
            "rank_or_kill_blockers": ["pre_submission_compliance_check_not_recorded"],
            "allowed_uses": ["promotion_review", "rank_frontier_candidate"],
        },
    )
    result_copy = tmp_path / "adjudicated.json"

    result = module.adjudicate(
        _args(
            contest_json=contest_json,
            archive=archive,
            provenance=provenance,
            result_copy=result_copy,
        )
    )

    assert result["raw_promotion_policy_gate_triggered"] is True
    assert result["promotion_eligible"] is False
    assert result["score_claim_valid"] is False
    assert result["allowed_use"] == ["forensic", "no_rank_frontier", "no_promotion"]
    assert "RAW_PROMOTION_POLICY" in result["lane_status"]

    payload = read_json(result_copy)
    assert payload["score_claim"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["promotion_eligible"] is False
    assert payload["allowed_uses"] == ["forensic", "no_rank_frontier", "no_promotion"]
    assert payload["raw_promotion_policy_gate_triggered"] is True


def test_adjudicator_promotes_only_when_no_raw_blockers(tmp_path: Path) -> None:
    module = _load_adjudicator()
    contest_json, archive, provenance = _write_minimal_inputs(tmp_path, {})
    result_copy = tmp_path / "adjudicated.json"

    result = module.adjudicate(
        _args(
            contest_json=contest_json,
            archive=archive,
            provenance=provenance,
            result_copy=result_copy,
        )
    )

    assert result["raw_promotion_policy_gate_triggered"] is False
    assert result["promotion_eligible"] is True
    payload = read_json(result_copy)
    assert payload["score_claim"] is True
    assert payload["rank_or_kill_eligible"] is True
    assert payload["allowed_uses"] == ["promotion_review", "rank_frontier_candidate"]
