# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "tools" / "sub_0192_viability_guard.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("sub_0192_viability_guard", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_pr101_fec6_profile_is_not_frontier_eligible_by_byte_mass() -> None:
    module = _load_module()
    fixture = REPO_ROOT / ".omx/research/pr101_fec6_byte_escape_profile_20260515_codex.json"

    report = module.build_report([fixture], axis="contest_cpu")

    assert report["score_claim"] is False
    assert report["dispatch_attempted"] is False
    assert report["not_frontier_eligible_count"] == 1
    row = report["reviews"][0]
    assert row["axis"] == "contest_cpu"
    assert row["frontier_eligible"] is False
    assert row["bytes_to_threshold_if_components_unchanged"] == 78
    assert row["remaining_byte_mass_bytes"] == 16
    assert "insufficient_remaining_byte_mass" in row["blockers"]


def test_synthetic_result_review_rejects_insufficient_measured_delta(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "measured_delta_result_review.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "tac_result_review_packet_v1",
                "lane_id": "lane_small_delta",
                "score_axis": "contest_cuda",
                "score_claim": True,
                "score_recomputation": {
                    "recomputed_score": 0.195,
                    "archive_bytes": 200_000,
                },
                "deltas": {
                    "delta_score_vs_baseline": -0.0005,
                },
            }
        ),
        encoding="utf-8",
    )

    row = module.build_report([artifact])["reviews"][0]

    assert row["score_claim"] is False
    assert row["input_score_claim_field"] is True
    assert row["frontier_eligible"] is False
    assert row["bytes_to_threshold_if_components_unchanged"] == 4506
    assert row["best_measured_delta"] == -0.0005
    assert "insufficient_measured_score_delta" in row["blockers"]


def test_jsonl_profile_rows_and_markdown_output(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "profiles.jsonl"
    artifact.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "technique": "byte_mass_plausible",
                        "anchor": {
                            "archive_bytes": 180_000,
                            "cpu_score": 0.1922,
                        },
                        "conclusion": {
                            "same_frame_realistic_saving_upper_bound_bytes": 500,
                        },
                    }
                ),
                json.dumps(
                    {
                        "technique": "byte_mass_short",
                        "exact_results": {"contest_cpu_score": 0.1922},
                        "conclusion": {
                            "same_frame_realistic_saving_upper_bound_bytes": 10,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "guard.md"

    rc = module.main(
        [
            "--input",
            str(artifact),
            "--axis",
            "contest_cpu",
            "--format",
            "markdown",
            "--output",
            str(output),
        ]
    )

    text = output.read_text(encoding="utf-8")
    assert rc == 0
    assert "Sub-0.192 Viability Guard" in text
    assert "`frontier-eligible`" in text
    assert "`not-frontier-eligible`" in text
    report = module.build_report([artifact], axis="contest_cpu")
    assert report["frontier_eligible_count"] == 1
    assert report["not_frontier_eligible_count"] == 1
    assert report["reviews"][0]["archive_bytes_source"] == "anchor.archive_bytes"


def test_forced_cuda_axis_does_not_fall_back_to_cpu_only_score(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "cpu_only.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "paired_guard_regression",
                "score_axis": "contest_cpu",
                "exact_results": {"contest_cpu_score": 0.1919},
                "remaining_byte_mass_bytes": 10_000,
            }
        ),
        encoding="utf-8",
    )

    row = module.build_report([artifact], axis="contest_cuda")["reviews"][0]

    assert row["axis"] == "contest_cuda"
    assert row["current_score"] is None
    assert row["frontier_eligible"] is False
    assert "missing_current_score" in row["blockers"]


def test_forced_axes_choose_matching_paired_score(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "paired.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "paired_guard_regression",
                "exact_results": {
                    "contest_cpu_score": 0.1919,
                    "contest_cuda_score": 0.2262,
                },
                "remaining_byte_mass_bytes": 1,
            }
        ),
        encoding="utf-8",
    )

    cpu = module.build_report([artifact], axis="contest_cpu")["reviews"][0]
    cuda = module.build_report([artifact], axis="contest_cuda")["reviews"][0]

    assert cpu["current_score"] == 0.1919
    assert cpu["current_score_source"] == "exact_results.contest_cpu_score"
    assert cuda["current_score"] == 0.2262
    assert cuda["current_score_source"] == "exact_results.contest_cuda_score"


def test_below_threshold_proxy_score_requires_validated_exact_claim(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "proxy_below_threshold.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "proxy_score_row",
                "score_axis": "contest_cpu",
                "canonical_score": 0.1919,
                "score_claim": False,
                "score_claim_valid": False,
                "remaining_byte_mass_bytes": 10_000,
            }
        ),
        encoding="utf-8",
    )

    row = module.build_report([artifact], axis="contest_cpu")["reviews"][0]

    assert row["axis"] == "contest_cpu"
    assert row["current_score"] == 0.1919
    assert row["frontier_eligible"] is False
    assert row["classification"] == "not_frontier_eligible"
    assert "below_threshold_score_without_validated_exact_claim" in row["blockers"]


def test_provider_cuda_advisory_label_is_not_treated_as_contest_cuda(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "provider_cuda_advisory.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "provider_advisory_row",
                "score_axis": "provider_cuda_kaggle_advisory",
                "lane_tag": "[provider-CUDA:kaggle advisory]",
                "canonical_score": 0.1919,
                "remaining_byte_mass_bytes": 10_000,
            }
        ),
        encoding="utf-8",
    )

    auto = module.build_report([artifact], axis="auto")["reviews"][0]
    forced = module.build_report([artifact], axis="contest_cuda")["reviews"][0]

    assert auto["axis"] == "unknown"
    assert auto["current_score"] is None
    assert "missing_current_score" in auto["blockers"]
    assert forced["axis"] == "contest_cuda"
    assert forced["current_score"] is None
    assert "missing_current_score" in forced["blockers"]


def test_below_threshold_exact_cpu_claim_can_clear_threshold_shortcut(tmp_path: Path) -> None:
    module = _load_module()
    artifact = tmp_path / "exact_cpu_below_threshold.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "tac_result_review_packet_v1",
                "score_axis": "contest_cpu",
                "score_claim_valid": True,
                "exact_cpu_evidence": True,
                "canonical_score": 0.1919,
                "custody": {
                    "archive_sha256": "a" * 64,
                    "archive_bytes": 180_000,
                },
            }
        ),
        encoding="utf-8",
    )

    row = module.build_report([artifact], axis="contest_cpu")["reviews"][0]

    assert row["frontier_eligible"] is True
    assert row["classification"] == "already_below_or_equal_threshold"
    assert "validated_exact_score_claim_for_axis" in row["reasons"]
