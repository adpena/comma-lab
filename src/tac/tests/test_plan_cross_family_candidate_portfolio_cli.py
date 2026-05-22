# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _pairset_acquisition() -> dict[str, object]:
    def row(candidate_id: str, rank: int, count: int) -> dict[str, object]:
        return {
            "schema": "decoder_q_pairset_acquisition_candidate.v1",
            **_false_authority(),
            "acquisition_id": candidate_id,
            "acquisition_rank": rank,
            "selector_kind": "diversity_spaced",
            "selected_pair_count": count,
            "selected_pair_indices": list(range(count)),
            "payload_bytes": 12 + count,
            "rate_delta": 0.00001,
            "predicted_score_mean": 0.195,
            "predicted_score_source": "fixture_non_authoritative",
        }

    return {
        "schema": "decoder_q_pairset_acquisition.v1",
        **_false_authority(),
        "candidates": [
            row("pairset_diversity_k002", 1, 2),
            row("pairset_diversity_k004", 2, 4),
            row("pairset_diversity_k008", 3, 8),
        ],
    }


def _auth_eval_json(
    tmp_path: Path,
    *,
    candidate_id: str,
    score: float,
    archive_char: str,
    raw_char: str,
) -> tuple[Path, str, str]:
    archive_sha = archive_char * 64
    raw_sha = raw_char * 64
    path = tmp_path / f"{candidate_id}_contest_auth_eval.json"
    path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cpu",
                "evidence_grade": "contest-CPU",
                "score_claim_valid": True,
                "canonical_score": score,
                "provenance": {
                    "archive_sha256": archive_sha,
                    "inflated_output_manifest": {
                        "payload": {"aggregate_sha256": raw_sha}
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path, archive_sha, raw_sha


def _observation_row(
    tmp_path: Path,
    *,
    candidate_id: str,
    score: float,
    archive_char: str,
    raw_char: str,
) -> dict[str, object]:
    source_path, archive_sha, raw_sha = _auth_eval_json(
        tmp_path,
        candidate_id=candidate_id,
        score=score,
        archive_char=archive_char,
        raw_char=raw_char,
    )
    return {
        "schema": "mlx_dynamic_sweep_observation.v1",
        "candidate_id": candidate_id,
        "sweep_config_id": "contest_cpu_exact_candidate",
        "optimization_pass_id": "exact_cpu_calibration",
        "family": "decoder_q_pairset_diversity",
        "observed_axis": "contest_cpu",
        "evidence_tag": "[contest-CPU]",
        "evidence_grade": "contest-CPU",
        "observed_score_or_delta": score,
        "archive_sha256": archive_sha,
        "runtime_sha256": "c" * 64,
        "raw_output_or_cache_sha256": raw_sha,
        "component_deltas": {
            "segnet_delta": 0.0,
            "posenet_delta": 0.0,
            "rate_delta": 0.0,
        },
        "source_artifact_path": source_path.as_posix(),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cli_writes_action_summary_and_response_model_markdown(
    tmp_path: Path,
) -> None:
    pairset_path = tmp_path / "pairset.json"
    observation_path = tmp_path / "observations.jsonl"
    json_out = tmp_path / "portfolio.json"
    md_out = tmp_path / "portfolio.md"
    summary_out = tmp_path / "action_summary.json"
    _write_json(pairset_path, _pairset_acquisition())
    _write_jsonl(
        observation_path,
        [
            _observation_row(
                tmp_path,
                candidate_id="pairset_diversity_k002",
                score=0.193,
                archive_char="a",
                raw_char="b",
            ),
            _observation_row(
                tmp_path,
                candidate_id="pairset_diversity_k004",
                score=0.1928,
                archive_char="d",
                raw_char="e",
            ),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_cross_family_candidate_portfolio.py"),
            "--incumbent-score",
            "0.195",
            "--incumbent-score-by-axis",
            "contest_cpu=0.1927",
            "--pairset-acquisition",
            str(pairset_path),
            "--observation-jsonl",
            str(observation_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
            "--summary-json-out",
            str(summary_out),
            "--top-actions",
            "2",
            "--require-active-pairset-observation-model",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(result.stdout)
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    model = summary["pairset_observation_response_model"]
    assert stdout["summary_json_out"] == str(summary_out)
    assert summary["schema"] == "cross_family_candidate_portfolio_action_summary.v1"
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert model["active"] is True
    assert model["score_claim"] is False
    assert len(summary["top_operator_actions"]) == 2
    assert all(
        row["ready_for_exact_eval_dispatch"] is False
        for row in summary["top_operator_actions"]
    )
    markdown = md_out.read_text(encoding="utf-8")
    assert "## CLI Action Summary" in markdown
    assert "### Observation Response Model" in markdown
    assert "### Top Next Actions" in markdown


def test_cli_require_active_pairset_model_fails_closed_without_observations(
    tmp_path: Path,
) -> None:
    pairset_path = tmp_path / "pairset.json"
    json_out = tmp_path / "portfolio.json"
    _write_json(pairset_path, _pairset_acquisition())

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_cross_family_candidate_portfolio.py"),
            "--incumbent-score",
            "0.195",
            "--pairset-acquisition",
            str(pairset_path),
            "--json-out",
            str(json_out),
            "--require-active-pairset-observation-model",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "pairset observation response model inactive" in result.stderr
    assert "at least two exact-axis pairset observations" in result.stderr
    assert not json_out.exists()
