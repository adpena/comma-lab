# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tac.optimization.architecture_sweep_recommendations import (
    ArchitectureSweepRecommendationError,
    ManifestSource,
    build_architecture_sweep_recommendations,
)

REPO = Path(__file__).resolve().parents[3]


def _evidence_grades(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        out: list[str] = []
        for key, value in payload.items():
            if key == "evidence_grade":
                out.append(str(value))
            out.extend(_evidence_grades(value))
        return out
    if isinstance(payload, list):
        out = []
        for item in payload:
            out.extend(_evidence_grades(item))
        return out
    return []


def _assert_fail_closed_dispatch_fields(payload: dict[str, Any]) -> None:
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["dispatchable"] is False
    assert "not_exact_cuda_auth_eval" in payload["dispatch_blockers"]
    assert payload.get("reason") or payload.get("dispatch_block_reason")


def _claim_row(
    *,
    timestamp: str,
    lane_id: str,
    instance_job_id: str,
    expires_at: str,
    status: str,
    notes: str,
) -> str:
    return (
        f"| {timestamp} | claude_lab | {lane_id} | lightning | {instance_job_id} | "
        f"{expires_at} | {status} | {notes} |\n"
    )


def _mps_manifest() -> dict[str, object]:
    return {
        "schema": "mps_research_signal_manifest.v1",
        "source": "mps_fixture.json",
        "evidence_grade": "MPS-research-signal",
        "evidence_semantics": "mps_proxy_curve_shape_only",
        "rows": [
            {
                "family": "arch_shrink",
                "curve_id": "width_x0_4",
                "variant_id": "epoch_000",
                "device": "mps",
                "archive_bytes": 100_000,
                "proxy_loss": 0.55,
                "dispatch_blockers": ["mps_proxy_signal_not_score_evidence"],
            },
            {
                "family": "arch_shrink",
                "curve_id": "width_x0_4",
                "variant_id": "epoch_010",
                "device": "mps",
                "archive_bytes": 120_000,
                "proxy_loss": 0.20,
                "dispatch_blockers": ["mps_proxy_signal_not_score_evidence"],
            },
        ],
    }


def _arch_plan() -> dict[str, object]:
    return {
        "schema": "pr101_arch_shrink_retraining_plan.v1",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "scenarios": [
            {
                "name": "stage_d_zeta_width_precision_int4",
                "driver_family": "self_compress_width_precision_hnerv",
                "evidence_grade": "prediction",
                "targets": {"element_retention": 0.45, "quant_bits": 4.0},
                "byte_estimate": {
                    "expected_archive_bytes": 67_235,
                    "delta_archive_bytes_vs_reference": -110_909,
                },
                "dispatch_blockers": ["no_exact_cuda_auth_eval"],
            }
        ],
    }


def _sparsity_manifest() -> dict[str, object]:
    return {
        "schema": "pr101_sparsity_block_sweep.v1",
        "evidence_grade": "[CPU-prep empirical byte-anchor only]",
        "dispatch_blockers": ["post_hoc_sparsity_not_retrained"],
        "rows": [
            {"alpha": 0.3, "archive_bytes": 140_000, "fraction_zeroed": 0.3},
            {"alpha": 0.7, "archive_bytes": 72_000, "fraction_zeroed": 0.7},
        ],
    }


def test_normalizes_cpu_mps_sources_and_blocks_dispatch() -> None:
    manifest = build_architecture_sweep_recommendations(
        [
            ManifestSource("mps.json", _mps_manifest()),
            ManifestSource("arch_plan.json", _arch_plan()),
            ManifestSource("sparsity.json", _sparsity_manifest()),
        ],
        run_id="fixture",
        lightning_active_jobs=[],
        dispatch_claims_markdown="",
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["evidence_grade"] == "empirical"
    assert manifest["row_count"] == 5
    assert {row["device_family"] for row in manifest["rows"]} == {"cpu", "mps"}
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in manifest["rows"])
    assert all(row["score_claim"] is False for row in manifest["rows"])
    assert all(row["promotion_eligible"] is False for row in manifest["rows"])
    assert all(row["rank_or_kill_eligible"] is False for row in manifest["rows"])
    assert all(row["dispatchable"] is False for row in manifest["rows"])
    assert {row["evidence_grade"] for row in manifest["rows"]} == {"empirical", "prediction"}
    assert set(_evidence_grades(manifest)) <= {"empirical", "prediction"}
    assert "MPS-research-signal" not in json.dumps(manifest)
    assert "[CPU-prep empirical byte-anchor only]" not in json.dumps(manifest)
    assert all("not_exact_cuda_auth_eval" in row["dispatch_blockers"] for row in manifest["rows"])

    curves = {(curve["family"], curve["curve_id"]): curve for curve in manifest["curves"]}
    assert curves[("arch_shrink", "width_x0_4")]["best_research_signal_row"]["variant_id"] == "epoch_010"
    assert curves[("pr101_sparsity", "post_hoc_sparsity_alpha")]["best_research_signal_row"]["variant_id"] == "alpha_0_7"
    for curve in manifest["curves"]:
        assert curve["score_claim"] is False
        assert curve["promotion_eligible"] is False
        assert curve["ready_for_exact_eval_dispatch"] is False
        best = curve["best_research_signal_row"]
        assert best["candidate_generation_only"] is True
        _assert_fail_closed_dispatch_fields(best)

    for recommendation in manifest["dispatch_recommendations"]:
        _assert_fail_closed_dispatch_fields(recommendation)
        for candidate in recommendation.get("candidate_generation_curve_ids", []):
            assert candidate["candidate_generation_only"] is True
            _assert_fail_closed_dispatch_fields(candidate)

    recommendations = [row["recommendation"] for row in manifest["dispatch_recommendations"]]
    assert "preserve_exact_cuda_promotion_gates" in recommendations


def test_cli_writes_manifest_markdown_and_active_lightning_warning(tmp_path: Path) -> None:
    mps = tmp_path / "mps.json"
    mps.write_text(json.dumps(_mps_manifest()), encoding="utf-8")
    active_jobs = tmp_path / "lightning_active_jobs.json"
    active_jobs.write_text(
        json.dumps(
            [
                {
                    "schema_version": "lightning_active_jobs.v1",
                    "lane_id": "arch_shrink_x0.4_lightning",
                    "job_name": "arch-shrink-x0-4-lightning-20260508T024304Z",
                    "submitted_at_utc": "2026-05-08T02:43:10Z",
                    "machine": "g4dn.2xlarge",
                    "profile": "q_faithful_dilated_88k",
                    "target_elements": 88_000,
                }
            ]
        ),
        encoding="utf-8",
    )
    claims = tmp_path / "claims.md"
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | expires_at | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 2026-05-08T02:43:09Z | claude_lab | arch_shrink_x0.4_lightning | lightning | arch-shrink-x0-4-lightning-20260508T024304Z | 2026-05-08T20:43:09Z | active_dispatching | existing dispatch |\n",
        encoding="utf-8",
    )
    output = tmp_path / "recommendations.json"
    markdown = tmp_path / "recommendations.md"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_architecture_sweep_recommendations.py"),
            "--input",
            str(mps),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
            "--run-id",
            "fixture_cli",
            "--lightning-active-jobs",
            str(active_jobs),
            "--dispatch-claims",
            str(claims),
        ],
        check=True,
        text=True,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["lightning_arch_shrink_state"]["do_not_duplicate_dispatch"] is True
    assert manifest["lightning_arch_shrink_state"]["active_jobs"][0]["job_name"] == (
        "arch-shrink-x0-4-lightning-20260508T024304Z"
    )
    assert manifest["lightning_arch_shrink_state"]["active_claims"][0]["status"] == "active_dispatching"
    active_warning = next(
        row
        for row in manifest["dispatch_recommendations"]
        if row["recommendation"] == "do_not_duplicate_active_arch_shrink_lightning_dispatch"
    )
    assert active_warning["score_claim"] is False
    assert active_warning["promotion_eligible"] is False
    assert active_warning["rank_or_kill_eligible"] is False
    assert active_warning["ready_for_exact_eval_dispatch"] is False
    assert active_warning["dispatchable"] is False
    assert "not_exact_cuda_auth_eval" in active_warning["dispatch_blockers"]
    assert "CPU/MPS research-signal planning only" in markdown.read_text(encoding="utf-8")


def test_dispatch_claim_parser_closes_older_active_rows() -> None:
    manifest = build_architecture_sweep_recommendations(
        [ManifestSource("mps.json", _mps_manifest())],
        run_id="fixture_claims",
        lightning_active_jobs=[],
        dispatch_claims_markdown=(
            "| timestamp_utc | agent | lane_id | platform | instance/job_id | expires_at | status | notes |\n"
            "|---|---|---|---|---|---|---|---|\n"
            "| 2026-05-08T02:47:14Z | claude_lab | arch_shrink_x0.4_lightning | lightning | arch-shrink-x0-4-lightning-20260508T020205Z |  | failed_no_auth_eval_json | terminal |\n"
            "| 2026-05-08T02:43:09Z | claude_lab | arch_shrink_x0.4_lightning | lightning | arch-shrink-x0-4-lightning-20260508T024304Z | 2026-05-08T20:43:09Z | active_dispatching | active |\n"
            "| 2026-05-08T02:02:09Z | claude_lab | arch_shrink_x0.4_lightning | lightning | arch-shrink-x0-4-lightning-20260508T020205Z | 2026-05-08T20:02:09Z | active_dispatching | older active |\n"
        ),
    )

    claims = manifest["lightning_arch_shrink_state"]["active_claims"]
    assert [claim["instance_job_id"] for claim in claims] == [
        "arch-shrink-x0-4-lightning-20260508T024304Z"
    ]


def test_dispatch_state_detects_hyphenated_arch_shrink_and_dedupes() -> None:
    manifest = build_architecture_sweep_recommendations(
        [ManifestSource("mps.json", _mps_manifest())],
        run_id="fixture_hyphenated_claim",
        lightning_active_jobs=[
            {
                "lane_id": "arch-shrink-x0.4-lightning",
                "job_name": "manual-width-retrain-20260508T024304Z",
                "submitted_at_utc": "2026-05-08T02:43:10Z",
            },
            {
                "lane_id": "arch_shrink_x0.4_lightning",
                "job_name": "manual-width-retrain-20260508T024304Z",
                "submitted_at_utc": "2026-05-08T02:43:10Z",
            },
        ],
        dispatch_claims_markdown=(
            "| timestamp_utc | agent | lane_id | platform | instance/job_id | expires_at | status | notes |\n"
            "|---|---|---|---|---|---|---|---|\n"
            + _claim_row(
                timestamp="2026-05-08T02:48:09Z",
                lane_id="arch-shrink-x0.4-lightning",
                instance_job_id="arch-shrink-x0-4-lightning-20260508T020205Z",
                expires_at="",
                status="failed_no_auth_eval_json",
                notes="terminal with alternate spelling",
            )
            + _claim_row(
                timestamp="2026-05-08T02:43:09Z",
                lane_id="arch-shrink-x0.4-lightning",
                instance_job_id="arch-shrink-x0-4-lightning-20260508T024304Z",
                expires_at="2026-05-08T20:43:09Z",
                status="active_dispatching",
                notes="active",
            )
            + _claim_row(
                timestamp="2026-05-08T02:42:09Z",
                lane_id="arch_shrink_x0.4_lightning",
                instance_job_id="arch-shrink-x0-4-lightning-20260508T024304Z",
                expires_at="2026-05-08T20:42:09Z",
                status="active_dispatching",
                notes="duplicate active alternate spelling",
            )
            + _claim_row(
                timestamp="2026-05-08T02:02:09Z",
                lane_id="arch_shrink_x0.4_lightning",
                instance_job_id="arch-shrink-x0-4-lightning-20260508T020205Z",
                expires_at="2026-05-08T20:02:09Z",
                status="active_dispatching",
                notes="older active closed by hyphenated terminal",
            )
        ),
    )

    claims = manifest["lightning_arch_shrink_state"]["active_claims"]
    jobs = manifest["lightning_arch_shrink_state"]["active_jobs"]
    assert manifest["lightning_arch_shrink_state"]["do_not_duplicate_dispatch"] is True
    assert [claim["lane_id"] for claim in claims] == ["arch-shrink-x0.4-lightning"]
    assert [job["lane_id"] for job in jobs] == ["arch-shrink-x0.4-lightning"]
    assert (
        sum(
            row["recommendation"] == "do_not_duplicate_active_arch_shrink_lightning_dispatch"
            for row in manifest["dispatch_recommendations"]
        )
        == 1
    )


def test_unknown_source_schema_fails_closed() -> None:
    try:
        build_architecture_sweep_recommendations(
            [ManifestSource("unknown.json", {"schema": "unknown.v1", "rows": []})],
            run_id="fixture_unknown",
            lightning_active_jobs=[],
            dispatch_claims_markdown="",
        )
    except ArchitectureSweepRecommendationError as exc:
        assert "unsupported architecture sweep manifest schema" in str(exc)
    else:  # pragma: no cover - explicit failure path for readability
        raise AssertionError("unsupported manifest schema did not fail closed")
