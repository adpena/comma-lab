# SPDX-License-Identifier: MIT
"""Tests for the NeRV model-size budget builder CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.build_nerv_modelsize_budget as tool
from tac.repo_io import ArtifactWriteError, sha256_file


def test_build_nerv_modelsize_budget_tool_writes_both_family_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"
    md = tmp_path / "budget.md"

    rc = tool.main(
        [
            "--output-hinerv-json",
            str(hinerv),
            "--output-snerv-json",
            str(snerv),
            "--output-md",
            str(md),
            "--hard-byte-ceiling",
            "36000",
            "--num-pairs",
            "17",
            "--per-ceiling-limit",
            "2",
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == "nerv_modelsize_budget_build.v1"
    assert summary["inputs"] == {
        "hard_byte_ceilings": [36000],
        "num_pairs": 17,
        "per_ceiling_limit": 2,
    }
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert hinerv.is_file()
    assert snerv.is_file()
    assert md.is_file()

    hinerv_payload = json.loads(hinerv.read_text(encoding="utf-8"))
    snerv_payload = json.loads(snerv.read_text(encoding="utf-8"))
    assert hinerv_payload["schema"] == "nerv_modelsize_budget.v1"
    assert snerv_payload["schema"] == "snerv_modelsize_budget.v1"
    assert hinerv_payload["family"] == "hi_nerv"
    assert snerv_payload["family"] == "snerv"
    assert snerv_payload["selected_candidates"][0]["candidate_id"].startswith(
        "snerv_np17_"
    )
    assert "_mfu" in snerv_payload["selected_candidates"][0]["candidate_id"]


def test_build_nerv_modelsize_budget_tool_requires_expected_hashes_for_overwrite(
    tmp_path: Path,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"
    md = tmp_path / "budget.md"
    argv = [
        "--output-hinerv-json",
        str(hinerv),
        "--output-snerv-json",
        str(snerv),
        "--output-md",
        str(md),
        "--hard-byte-ceiling",
        "36000",
        "--num-pairs",
        "17",
        "--per-ceiling-limit",
        "2",
    ]

    assert tool.main(argv) == 0

    with pytest.raises(ArtifactWriteError, match="expected_existing_sha256"):
        tool.main([*argv, "--allow-overwrite"])


def test_build_nerv_modelsize_budget_tool_overwrites_with_explicit_hash_custody(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"
    md = tmp_path / "budget.md"
    argv = [
        "--output-hinerv-json",
        str(hinerv),
        "--output-snerv-json",
        str(snerv),
        "--output-md",
        str(md),
        "--hard-byte-ceiling",
        "36000",
        "--num-pairs",
        "17",
        "--per-ceiling-limit",
        "2",
    ]

    assert tool.main(argv) == 0
    capsys.readouterr()
    assert tool.main(
        [
            *argv,
            "--allow-overwrite",
            "--expected-output-hinerv-json-sha256",
            sha256_file(hinerv),
            "--expected-output-snerv-json-sha256",
            sha256_file(snerv),
            "--expected-output-md-sha256",
            sha256_file(md),
        ]
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == "nerv_modelsize_budget_build.v1"
    assert summary["promotion_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    md_text = md.read_text(encoding="utf-8")
    assert "payload=`None`" not in md_text
    assert "nominal_under_ceiling=`" in md_text


def test_build_nerv_modelsize_budget_tool_can_rerun_with_guarded_overwrite(
    tmp_path: Path,
    capsys,
) -> None:
    hinerv = tmp_path / "hinerv.json"
    snerv = tmp_path / "snerv.json"
    md = tmp_path / "budget.md"
    args = [
        "--output-hinerv-json",
        str(hinerv),
        "--output-snerv-json",
        str(snerv),
        "--output-md",
        str(md),
        "--hard-byte-ceiling",
        "36000",
        "--num-pairs",
        "17",
        "--per-ceiling-limit",
        "1",
    ]

    assert tool.main(args) == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert tool.main(
        [
            *args,
            "--allow-overwrite",
            "--expected-output-hinerv-json-sha256",
            sha256_file(hinerv),
            "--expected-output-snerv-json-sha256",
            sha256_file(snerv),
            "--expected-output-md-sha256",
            sha256_file(md),
        ]
    ) == 0
    second_summary = json.loads(capsys.readouterr().out)

    assert second_summary["hinerv_output_sha256"] == first_summary["hinerv_output_sha256"]
    assert second_summary["snerv_output_sha256"] == first_summary["snerv_output_sha256"]
    assert second_summary["output_md_sha256"] == first_summary["output_md_sha256"]
