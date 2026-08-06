# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tools.build_ddm_vo2_instrument_registry import build_registry, write_registry
from tools.check_instrument_registry_form_grade_refs import scan_receipts


def test_build_registry_contains_round0_and_source_candidate_rows() -> None:
    build = build_registry()

    assert build.summary["round_reached"] == "R1"
    assert build.summary["row_count"] == len(build.rows)
    assert build.summary["row_groups"]["ca1-round0"] == 89
    assert build.summary["row_groups"]["sw1-round0"] == 16
    assert any(row["instrument_id"] == "iteration_cap_stop_defaults" for row in build.rows)
    assert any(
        row["candidate_status"] == "OVERINCLUSIVE_SOURCE_CANDIDATE_NEEDS_R2_CONSUMER_CONFIRMATION"
        for row in build.rows
    )


def test_write_registry_outputs_parseable_jsonl(tmp_path: Path) -> None:
    manifest = write_registry(tmp_path)

    registry = tmp_path / "INSTRUMENT_REGISTRY.jsonl"
    summary = tmp_path / "ROUND_SUMMARY.json"
    assert any(path.endswith("INSTRUMENT_REGISTRY.jsonl") for path in manifest)
    assert not any(path.endswith("MANIFEST.sha256.json") for path in manifest)
    assert registry.exists()
    assert summary.exists()
    rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all({"instrument_id", "elements", "calibration_lineage"} <= set(row) for row in rows)


def test_form_grade_ref_positive_control(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps(
            {
                "instrument_id": "demo_instrument",
                "elements": [{"name": "stopping_rule", "form_grade": "NAIVE-NAMED"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bad = tmp_path / "bad_receipt.md"
    bad.write_text(
        "# Receipt\n\nverdict: HOLD\nscore_claim: false\nUses demo_instrument.\n",
        encoding="utf-8",
    )
    good = tmp_path / "good_receipt.md"
    good.write_text(
        "# Receipt\n\nverdict: HOLD\nscore_claim: false\n"
        "Uses demo_instrument.\nform_grade_ref:demo_instrument\n",
        encoding="utf-8",
    )

    bad_report = scan_receipts(registry_path=registry, receipt_paths=[bad])
    good_report = scan_receipts(registry_path=registry, receipt_paths=[good])

    assert bad_report["missing_form_grade_ref_count"] == 1
    assert good_report["missing_form_grade_ref_count"] == 0
