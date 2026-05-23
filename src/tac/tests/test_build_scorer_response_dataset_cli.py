# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _source_authority() -> dict:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _write_distilled_vs_direct_smoke(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "distilled_vs_direct_scorer_paired_smoke.v1",
                "producer": "pact_nerv_distilled_scorer_stage1",
                "smoke_kind": "distilled_vs_direct_scorer_paired_smoke",
                "candidate": {
                    "candidate_id": "pds_stage1_smoke",
                    "advisory_eval": {
                        "canonical_score": 0.99,
                        "archive_size_bytes": 99,
                        "avg_posenet_dist": 0.003,
                        "avg_segnet_dist": 0.009,
                        "axis": "[macOS-CPU advisory test]",
                        "archive": {"sha256": "a" * 64, "bytes": 99},
                        "raw": {"sha256": "b" * 64},
                    },
                },
                "authority": _source_authority(),
            }
        ),
        encoding="utf-8",
    )


def test_build_scorer_response_dataset_cli_distilled_rows_are_opt_in(
    tmp_path: Path,
) -> None:
    smoke = tmp_path / "distilled_vs_direct.json"
    json_out = tmp_path / "dataset.json"
    routing_out = tmp_path / "routing.json"
    _write_distilled_vs_direct_smoke(smoke)

    base_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "build_scorer_response_dataset.py"),
        "--input",
        str(smoke),
        "--json-out",
        str(json_out),
        "--baseline-score",
        "1.0",
        "--baseline-archive-bytes",
        "100",
        "--consumer-routing-json-out",
        str(routing_out),
    ]
    strict = subprocess.run(
        base_cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert strict.returncode == 3
    assert "skipped requested input" in strict.stderr

    subprocess.run(
        [*base_cmd, "--allow-skipped"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    skipped = json.loads(json_out.read_text(encoding="utf-8"))
    assert skipped["rows"] == []
    assert "requires include_distilled_vs_direct_rows" in skipped["skipped"][0]["reason"]
    skipped_routing = json.loads(routing_out.read_text(encoding="utf-8"))
    assert skipped_routing["row_count"] == 0
    assert skipped_routing["score_claim"] is False

    subprocess.run(
        [*base_cmd, "--include-distilled-vs-direct-rows"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    included = json.loads(json_out.read_text(encoding="utf-8"))
    assert included["rows"][0]["family"] == "distilled_vs_direct_scorer_paired_smoke"
    assert included["summary"]["family_counts"] == {
        "distilled_vs_direct_scorer_paired_smoke": 1
    }
    included_routing = json.loads(routing_out.read_text(encoding="utf-8"))
    assert included_routing["schema"] == "scorer_response_dataset_consumer_routing.v1"
    assert included_routing["row_count"] == 1
    assert included_routing["score_claim_valid"] is False
    assert any(
        verdict["consumer_name"] == "distilled_scorer_surrogate_canonical_equation_consumer"
        for verdict in included_routing["verdicts"]
    )


def test_build_scorer_response_dataset_cli_fails_on_malformed_input_by_default(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.json"
    bad = tmp_path / "bad.json"
    json_out = tmp_path / "dataset.json"
    _write_distilled_vs_direct_smoke(valid)
    bad.write_text("{not json", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_scorer_response_dataset.py"),
            "--input",
            str(valid),
            "--input",
            str(bad),
            "--include-distilled-vs-direct-rows",
            "--json-out",
            str(json_out),
            "--baseline-score",
            "1.0",
            "--baseline-archive-bytes",
            "100",
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 3
    assert "skipped requested input" in result.stderr
    dataset = json.loads(json_out.read_text(encoding="utf-8"))
    assert dataset["summary"]["row_count"] == 1
    assert dataset["skipped"]
