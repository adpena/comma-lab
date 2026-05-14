# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "analyze_public_qpose_payload_family.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_public_qpose_payload_family_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_p3_layout_uses_self_describing_lengths() -> None:
    script = _load_script()
    payload = (
        b"P3"
        + (3).to_bytes(4, "little")
        + (4).to_bytes(2, "little")
        + (5).to_bytes(2, "little")
        + b"mmm"
        + b"rrrr"
        + b"aaaaa"
        + b"pp"
    )

    layout = script.split_payload_layout(payload, SimpleNamespace())

    assert layout["payload_format"] == "public_pr75_qzs3_qp1_segactions_p3"
    assert layout["boundary_authority"] == "self_describing_p3_header"
    assert layout["header"]["bytes"] == 10
    assert [(s["name"], s["offset"], s["charged_bytes"]) for s in layout["segments"]] == [
        ("masks.mkv", 10, 3),
        ("renderer.bin", 13, 4),
        ("seg_tile_actions.bin", 17, 5),
        ("optimized_poses.qp1", 22, 2),
    ]


def test_split_qp19_layout_uses_self_describing_lengths() -> None:
    script = _load_script()
    payload = (
        b"QP19"
        + bytes([1, 0])
        + (3).to_bytes(4, "little")
        + (4).to_bytes(4, "little")
        + (5).to_bytes(4, "little")
        + b"mmm"
        + b"rrrr"
        + b"ppppp"
    )

    layout = script.split_payload_layout(payload, SimpleNamespace())

    assert layout["payload_format"] == "public_pr77_qp19_qzs3_qpv1_v1"
    assert layout["boundary_authority"] == "self_describing_qp19_header"
    assert layout["header"]["bytes"] == 18
    assert [(s["name"], s["offset"], s["charged_bytes"]) for s in layout["segments"]] == [
        ("masks.mkv", 18, 3),
        ("renderer.bin", 21, 4),
        ("optimized_poses.qpv1", 25, 5),
    ]


def test_pairwise_action_diffs_are_multiset_aware() -> None:
    script = _load_script()
    archives = [
        {
            "label": "left",
            "_action_records_ref": [(1, 2, 3), (1, 2, 3), (5, 6, 7)],
            "actions": {"runtime_record_sha256": "aaa"},
        },
        {
            "label": "right",
            "_action_records_ref": [(1, 2, 3), (8, 9, 10)],
            "actions": {"runtime_record_sha256": "bbb"},
        },
    ]

    diff = script.pairwise_action_diffs(archives)["left_vs_right"]

    assert diff["common_record_count"] == 1
    assert diff["left_only_record_count"] == 2
    assert diff["right_only_record_count"] == 1
    assert diff["sequence_equal"] is False


def test_c102_byte_savings_needed_for_031_matches_formula() -> None:
    script = _load_script()
    eval_summary = {
        "score_gap_to_target": 0.31514430182167497 - 0.31,
        "score_pose_contribution": 0.070240301821675,
        "avg_posenet_dist": 0.00049337,
    }

    # A byte-only path must close the score gap through 25*bytes/37,545,489.
    assert int(__import__("math").ceil(eval_summary["score_gap_to_target"] / script.RATE_LAMBDA)) == 7726
    stripped = script._break_even(eval_summary, -10)
    assert stripped["required_total_component_score_gain_to_reach_0_31"] > 0.00513


def test_parse_archive_specs_supports_label_path_pairs() -> None:
    script = _load_script()

    specs = script.parse_archive_specs(["a=/tmp/a.zip", "b=relative.zip"])

    assert [spec.label for spec in specs] == ["a", "b"]
    assert [str(spec.path) for spec in specs] == ["/tmp/a.zip", "relative.zip"]
