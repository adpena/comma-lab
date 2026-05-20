# SPDX-License-Identifier: MIT
"""Tests for null-space seed replacement planning."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from tac.procedural_codebook_generator import (
    build_null_seed_replacement_plan,
    contiguous_runs,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _summary(n_total: int, n_null: int) -> dict[str, object]:
    return {
        "schema": "null_byte_master_gradient_probe_v1",
        "n_total_bytes": n_total,
        "n_null_bytes": n_null,
        "null_fraction": n_null / n_total,
        "epsilon": 1e-9,
        "section_breakdown": {
            "OUTER_MAGIC": {
                "range": [0, 4],
                "length_bytes": 4,
                "n_null": 4,
                "null_fraction_within_section": 1.0,
            },
            "source_len_hdr": {
                "range": [4, 8],
                "length_bytes": 4,
                "n_null": 4,
                "null_fraction_within_section": 1.0,
            },
            "source_payload": {
                "range": [8, 90],
                "length_bytes": 82,
                "n_null": 24,
                "null_fraction_within_section": 24 / 82,
            },
            "selector_len_hdr": {
                "range": [90, 92],
                "length_bytes": 2,
                "n_null": 2,
                "null_fraction_within_section": 1.0,
            },
            "selector_payload": {
                "range": [92, 130],
                "length_bytes": 38,
                "n_null": 38,
                "null_fraction_within_section": 1.0,
            },
        },
    }


def test_contiguous_runs_and_null_seed_plan_rank_cross_section_selector_run() -> None:
    null_indices = np.array(
        list(range(0, 20)) + list(range(40, 52)) + list(range(90, 130)),
        dtype=np.int64,
    )

    assert contiguous_runs(null_indices) == [(0, 20), (40, 52), (90, 130)]

    plan = build_null_seed_replacement_plan(
        null_summary=_summary(n_total=160, n_null=int(null_indices.size)),
        null_indices=null_indices,
        inner_bytes=bytes(range(160)),
        seed_bytes=8,
        min_run_length=10,
    )

    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["summary"]["best_net_saved_inner_bytes"] == 32
    assert plan["summary"]["positive_candidate_count"] == 4
    top = plan["candidates"][0]
    assert top["range"] == [90, 130]
    assert top["section"] == "selector_len_hdr+selector_payload"
    assert top["net_saved_inner_bytes"] == 32
    assert "exact_cuda_eval_on_shrunk_archive" in top["required_next_proofs"]


def test_plan_null_codebook_replacements_cli(tmp_path: Path) -> None:
    summary = {
        "schema": "null_byte_master_gradient_probe_v1",
        "n_total_bytes": 100,
        "n_null_bytes": 40,
        "null_fraction": 0.4,
        "epsilon": 1e-9,
        "section_breakdown": {
            "source_payload": {
                "range": [0, 100],
                "length_bytes": 100,
                "n_null": 40,
                "null_fraction_within_section": 0.4,
            }
        },
    }
    summary_path = tmp_path / "null_byte_summary.json"
    indices_path = tmp_path / "null_byte_indices.npy"
    archive_path = tmp_path / "archive.zip"
    output_json = tmp_path / "plan.json"
    output_md = tmp_path / "plan.md"

    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    np.save(indices_path, np.arange(10, 50, dtype=np.int64))
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, bytes(range(100)), compress_type=zipfile.ZIP_STORED)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_null_codebook_replacements.py"),
            "--null-summary",
            str(summary_path),
            "--null-indices",
            str(indices_path),
            "--archive-zip",
            str(archive_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    plan = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout["best_net_saved_inner_bytes"] == 32
    assert plan["schema"] == "null_space_seed_replacement_plan_v1"
    assert plan["score_claim"] is False
    assert plan["summary"]["best_net_saved_inner_bytes"] == 32
    assert plan["candidates"][0]["range"] == [10, 50]
    assert "Score claim: `false`" in output_md.read_text(encoding="utf-8")
