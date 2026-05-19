# SPDX-License-Identifier: MIT
"""Coverage for the 600-pair independence diagnostic tool."""

from __future__ import annotations

import random
from pathlib import Path

from tac.repo_io import read_json, write_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = load_repo_tool(
    REPO_ROOT,
    "tools/test_600_pair_independence.py",
    "test_600_pair_independence_tool_under_test",
)


def test_independence_report_detects_serial_dependence(tmp_path: Path) -> None:
    value = 0.0
    correlated: list[float] = []
    for idx in range(600):
        value = 0.96 * value + 0.04 * ((idx * 17) % 19)
        correlated.append(value)
    source = tmp_path / "correlated.json"
    write_json(source, {"rows": [{"component_score_no_rate": item} for item in correlated]})

    report = TOOL.build_report(
        input_jsons=[source],
        explicit_paths=["rows[].component_score_no_rate"],
        min_length=600,
        max_lag=24,
        max_depth=4,
    )

    assert report["aggregate_verdict"] == "independence_assumption_blocked"
    assert report["score_claim"] is False
    assert report["score_claim_valid"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["dispatchable"] is False
    assert report["scorer_invoked"] is False


def test_independence_report_accepts_seeded_low_correlation_series(tmp_path: Path) -> None:
    rng = random.Random(20260519)
    source = tmp_path / "independent.json"
    write_json(source, {"per_pair_score_marginals": [rng.random() for _ in range(600)]})

    report = TOOL.build_report(
        input_jsons=[source],
        explicit_paths=["per_pair_score_marginals"],
        min_length=600,
        max_lag=12,
        max_depth=4,
    )

    assert report["aggregate_verdict"] == "independence_ok_for_factorized_assumption"
    assert report["series_reports"][0]["n"] == 600
    assert report["consumer_contract"]["must_not_promote_score_or_candidate"] is True


def test_auto_discovery_extracts_numeric_600_pair_columns(tmp_path: Path) -> None:
    rows = [
        {
            "pair_idx": idx,
            "pose_dist": float((idx * 13) % 97) / 97.0,
            "label": f"pair-{idx}",
        }
        for idx in range(600)
    ]
    source = tmp_path / "xray.json"
    write_json(
        source,
        {
            "schema": "fixture",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "rows": rows,
        },
    )

    candidates = TOOL.load_series_candidates(
        [source],
        explicit_paths=[],
        min_length=600,
        max_depth=4,
    )

    discovered_paths = {candidate.series_path for candidate in candidates}
    assert "rows[*].pose_dist" in discovered_paths
    assert "rows[*].pair_idx" not in discovered_paths


def test_cli_writes_report_and_summary(tmp_path: Path, capsys) -> None:
    rng = random.Random(11)
    source = tmp_path / "packet.json"
    output = tmp_path / "report.json"
    write_json(source, {"per_pair_score_marginals": [rng.random() for _ in range(600)]})

    rc = TOOL.main(
        [
            "--input-json",
            str(source),
            "--series-path",
            "per_pair_score_marginals",
            "--max-lag",
            "12",
            "--output-json",
            str(output),
            "--summary",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "[600-pair-independence]" in captured.out
    report = read_json(output)
    assert report["schema"] == "pair_independence_diagnostic_v1"
    assert report["dispatch_attempted"] is False
