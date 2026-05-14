# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_c091_mask_packer_bigmove.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("plan_c091_mask_packer_bigmove_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_slice_parser_uses_public_pr75_lengths() -> None:
    planner = _load_script()

    slices = planner.parse_payload_slices(b"x" * 276_381)

    assert slices.payload_format.endswith("pr75_minp_fixed_actions255_model55756")
    assert len(slices.mask_br) == 219_472
    assert len(slices.renderer_br) == 55_756
    assert len(slices.actions_br) == 255
    assert len(slices.pose_br) == 898


def test_score_math_uses_explicit_target_score() -> None:
    planner = _load_script()

    row = planner.score_math(
        frontier_score=0.31516575028285976,
        frontier_bytes=276_481,
        candidate_bytes=276_329,
        target_score=0.314,
    )

    assert row["target_score"] == 0.314
    assert row["delta_bytes_vs_frontier"] == -152
    assert row["score_if_components_unchanged"] > 0.315
    assert row["target_equivalent_bytes_needed_after_candidate"] == 1599
    assert row["sub314_equivalent_bytes_needed_after_candidate"] == 1599


def test_score_math_defaults_to_active_sub031_target() -> None:
    planner = _load_script()

    row = planner.score_math(
        frontier_score=0.31514430182167497,
        frontier_bytes=276_485,
        candidate_bytes=276_485,
    )

    assert row["target_score"] == 0.31
    assert row["target_equivalent_bytes_needed_after_candidate"] == 7726


def test_mask_lossless_probe_reports_exact_savings_against_larger_source() -> None:
    planner = _load_script()
    raw = (b"mask-obu" * 512) + bytes(range(32))
    source = raw + b"uncompressed-placeholder"

    probe = planner.mask_lossless_probes(raw, source)

    assert probe["best"]["roundtrip_equal"] is True
    assert probe["best"]["bytes"] < len(source)
    assert probe["best_savings_bytes"] == len(source) - probe["best"]["bytes"]


def test_collect_candidate_rows_marks_active_and_rate_only_rows(tmp_path: Path) -> None:
    planner = _load_script()
    matrix = tmp_path / "candidate_matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "pr77_actions_pr75mask_renderer_c089pose_fixedslice",
                        "archive_bytes": 276_329,
                        "archive_sha256": "a" * 64,
                        "dispatchable_after_gate": True,
                        "noop": False,
                        "score_claim": False,
                    },
                    {
                        "candidate_id": "rate_only_sub314_probe",
                        "archive_bytes": 274_000,
                        "archive_sha256": "b" * 64,
                        "dispatchable_after_gate": True,
                        "noop": False,
                        "score_claim": False,
                    },
                ]
            }
        )
    )

    rows = planner.collect_candidate_rows(
        [matrix],
        frontier={"score": 0.31516575028285976, "bytes": 276_481},
        active_candidate_id="pr77_actions_pr75mask_renderer_c089pose_fixedslice",
        target_score=0.314,
    )
    by_id = {row["candidate_id"]: row for row in rows}

    assert by_id["pr77_actions_pr75mask_renderer_c089pose_fixedslice"]["active_elsewhere"] is True
    assert (
        by_id["pr77_actions_pr75mask_renderer_c089pose_fixedslice"]["exact_eval_status"]
        == "already_active_do_not_touch_or_duplicate"
    )
    assert by_id["rate_only_sub314_probe"]["exact_eval_justified_from_this_plan"] is True
    assert by_id["rate_only_sub314_probe"]["break_even_vs_c091"]["target_equivalent_bytes_needed_after_candidate"] == 0


def test_parse_cq_values_rejects_out_of_range() -> None:
    planner = _load_script()

    assert planner.parse_cq_values("63,50,0") == [63, 50, 0]
    try:
        planner.parse_cq_values("64")
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected out-of-range CQ value to fail")
