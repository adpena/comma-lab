# SPDX-License-Identifier: MIT
"""Catalog #221 regression tests for auth-eval/PacketIR result authority leaks."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_auth_eval_result_artifacts_fail_closed_for_score_claims,
    preflight_all,
)


def _results_dir(root: Path, name: str) -> Path:
    path = root / "experiments" / "results" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _roundtrip_payload(*, row_score_claim: bool = False) -> dict:
    return {
        "schema": "auth_eval_roundtrip_results_v1",
        "score_claim": False,
        "rows": [
            {
                "target_id": "modal_contest_cuda_t4_auto",
                "contest_axis_anchor": True,
                "score_claim_possible_after_result_review": True,
                "score_claim": row_score_claim,
                "rank_or_kill_eligible": False,
                "result_review_blockers": [
                    "roundtrip_matrix_is_command_planner_not_claim_surface",
                    "requires_separate_auth_eval_result_review_before_score_claim",
                ],
            }
        ],
    }


def _closure_payload(*, with_cpu_blockers: bool) -> dict:
    blockers = (
        [
            "not_contest_cuda_axis",
            "cpu_axis_not_rank_or_kill_authority",
            "requires_cuda_cpu_policy_review",
        ]
        if with_cpu_blockers
        else []
    )
    return {
        "schema": "packetir_exact_eval_closure_v1",
        "axes": {
            "contest_cpu": {
                "score_axis": "contest_cpu",
                "score_claim": False,
                "promotion_blockers": blockers,
                "rank_or_kill_blockers": blockers,
            }
        },
    }


def test_check221_roundtrip_row_score_claim_is_violation(tmp_path: Path) -> None:
    out = _results_dir(tmp_path, "bad_roundtrip") / "auth_eval_roundtrip_results.json"
    _write_json(out, _roundtrip_payload(row_score_claim=True))

    violations = check_auth_eval_result_artifacts_fail_closed_for_score_claims(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "row score_claim must be false" in violations[0]


def test_check221_roundtrip_missing_review_blocker_is_violation(tmp_path: Path) -> None:
    payload = _roundtrip_payload()
    payload["rows"][0]["result_review_blockers"] = []
    out = _results_dir(tmp_path, "bad_roundtrip") / "auth_eval_roundtrip_results.json"
    _write_json(out, payload)

    violations = check_auth_eval_result_artifacts_fail_closed_for_score_claims(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "missing fail-closed result_review_blockers" in violations[0]


def test_check221_packetir_cpu_axis_requires_diagnostic_blockers(tmp_path: Path) -> None:
    out = _results_dir(tmp_path, "bad_closure") / "closure.json"
    _write_json(out, _closure_payload(with_cpu_blockers=False))

    violations = check_auth_eval_result_artifacts_fail_closed_for_score_claims(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 2
    assert all("axes.contest_cpu" in item for item in violations)
    assert "not_contest_cuda_axis" in "\n".join(violations)


def test_check221_clean_artifacts_pass_and_strict_is_silent(tmp_path: Path) -> None:
    _write_json(
        _results_dir(tmp_path, "roundtrip") / "auth_eval_roundtrip_results.json",
        _roundtrip_payload(),
    )
    _write_json(
        _results_dir(tmp_path, "closure") / "closure.json",
        _closure_payload(with_cpu_blockers=True),
    )

    assert (
        check_auth_eval_result_artifacts_fail_closed_for_score_claims(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
        == []
    )


def test_check221_strict_raises(tmp_path: Path) -> None:
    out = _results_dir(tmp_path, "bad_roundtrip") / "auth_eval_roundtrip_results.json"
    _write_json(out, _roundtrip_payload(row_score_claim=True))

    with pytest.raises(PreflightError, match="Catalog|score_claim"):
        check_auth_eval_result_artifacts_fail_closed_for_score_claims(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_check221_live_repo_clean() -> None:
    assert check_auth_eval_result_artifacts_fail_closed_for_score_claims(
        strict=False,
        verbose=False,
    ) == []


def test_check221_wired_into_preflight_all_strict() -> None:
    source = inspect.getsource(preflight_all)
    assert "check_auth_eval_result_artifacts_fail_closed_for_score_claims" in source
    callsite = source.split(
        "check_auth_eval_result_artifacts_fail_closed_for_score_claims",
        1,
    )[1][:160]
    assert "strict=True" in callsite
