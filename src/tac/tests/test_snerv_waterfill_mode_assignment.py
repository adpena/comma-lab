# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.analysis.snerv_waterfill_mode_assignment import (
    SNERV_WATERFILL_MODE_ASSIGNMENT_SCHEMA,
    SnervWaterfillModeAssignmentError,
    build_snerv_waterfill_mode_assignment,
)
from tools.build_snerv_waterfill_mode_assignment import main as tool_main


def test_waterfill_mode_assignment_compiles_archive_order_modes() -> None:
    report = build_snerv_waterfill_mode_assignment(
        _waterfill_report(
            [
                _action("decoder.level0.HH.kernel", "zero_rle", 0),
                _action("decoder.level0.LH.kernel", "int2", 2),
                _action("decoder.level0.HL.kernel", "int4", 4),
            ],
            row_blockers=[],
            plan_blockers=[],
        )
    )

    row = report["rows"][0]
    assert report["schema"] == SNERV_WATERFILL_MODE_ASSIGNMENT_SCHEMA
    assert report["score_claim"] is False
    assert row["modes"] == ["int2", "int4", "zero"]
    assert row["mode_plan_cli_arg"] == "int2,int4,zero"
    assert row["ready_for_local_advisory_probe"] is True
    assert row["ready_for_receiver_mode_export"] is True
    assert row["mode_histogram"] == {"int2": 1, "int4": 1, "zero": 1}


def test_fp32_protect_maps_to_fp16_but_blocks_export() -> None:
    report = build_snerv_waterfill_mode_assignment(
        _waterfill_report(
            [
                _action("decoder.level0.LH.kernel", "fp32_protect", 32),
                _action("decoder.level0.HL.kernel", "int8", 8),
                _action("decoder.level0.HH.kernel", "fp16", 16),
            ],
            row_blockers=["decoder_weight_saliency_missing_for_some_groups"],
            plan_blockers=[],
        )
    )

    row = report["rows"][0]
    assert row["modes"] == ["fp16", "int8", "fp16"]
    assert row["ready_for_local_advisory_probe"] is True
    assert row["ready_for_receiver_mode_export"] is False
    assert "mixed_decoder_modes_do_not_support_fp32" in row["blockers"]
    assert "fp32_protect_downgraded_to_fp16_requires_receiver_replay" in row["blockers"]
    assert "decoder_weight_saliency_missing_for_some_groups" in row["blockers"]


def test_missing_decoder_group_blocks_probe() -> None:
    report = build_snerv_waterfill_mode_assignment(
        _waterfill_report(
            [
                _action("decoder.level0.LH.kernel", "int2", 2),
                _action("decoder.level0.HL.kernel", "int4", 4),
            ],
            row_blockers=[],
            plan_blockers=[],
        )
    )

    row = report["rows"][0]
    assert row["ready_for_local_advisory_probe"] is False
    assert "decoder_mode_group_missing:level0.HH" in row["blockers"]


def test_wrong_source_schema_rejected() -> None:
    with pytest.raises(SnervWaterfillModeAssignmentError, match="expected"):
        build_snerv_waterfill_mode_assignment({"schema": "wrong"})


def test_build_snerv_waterfill_mode_assignment_cli(tmp_path: Path) -> None:
    source_path = tmp_path / "waterfill.json"
    output_json = tmp_path / "modes.json"
    output_md = tmp_path / "modes.md"
    source_path.write_text(
        json.dumps(
            _waterfill_report(
                [
                    _action("decoder.level0.LH.kernel", "int2", 2),
                    _action("decoder.level0.HL.kernel", "int4", 4),
                    _action("decoder.level0.HH.kernel", "zero_rle", 0),
                ],
                row_blockers=[],
                plan_blockers=[],
            )
        ),
        encoding="utf-8",
    )

    rc = tool_main(
        [
            "--waterfill-json",
            str(source_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--candidate-id",
            "candidate_a",
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["rows"][0]["mode_plan_cli_arg"] == "int2,int4,zero"
    assert "SNeRV waterfill mode assignment" in output_md.read_text(encoding="utf-8")


def _waterfill_report(
    actions: list[dict[str, object]],
    *,
    row_blockers: list[str],
    plan_blockers: list[str],
) -> dict[str, object]:
    return {
        "schema": "snerv_trained_ladder_waterfill.v1",
        "candidate_id": "candidate",
        "blockers": [],
        "rows": [
            {
                "row_id": "row_a",
                "archive_sha256_actual": "a" * 64,
                "decoder_payload_schema": "snerv_decoder_payload.v3",
                "decoder_precision_mode": "mixed_magnitude_symmetric",
                "blockers": row_blockers,
                "waterfill_plan": {
                    "blockers": plan_blockers,
                    "rows": actions,
                },
            }
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _action(group: str, action: str, bits: int) -> dict[str, object]:
    return {
        "group_name": group,
        "selected_action": action,
        "selected_bits": bits,
    }
