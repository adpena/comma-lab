# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.repo_io import read_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/build_d1_pair_mask_from_xray.py",
        "build_d1_pair_mask_from_xray_test",
    )


def _write_xray(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.write_text(
        json.dumps({"rows": rows, "score_claim": False}) + "\n",
        encoding="utf-8",
    )


def test_build_d1_pair_mask_from_xray_selects_only_improving_pairs(tmp_path: Path) -> None:
    module = _load_tool()
    baseline = tmp_path / "baseline.json"
    positive = tmp_path / "positive.json"
    negative = tmp_path / "negative.json"
    _write_xray(
        baseline,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.010},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.010},
        ],
    )
    _write_xray(
        positive,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.008},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.012},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.009},
        ],
    )
    _write_xray(
        negative,
        [
            {"pair_idx": 0, "pose_dist": 0.01, "seg_dist": 0.011},
            {"pair_idx": 1, "pose_dist": 0.01, "seg_dist": 0.007},
            {"pair_idx": 2, "pose_dist": 0.01, "seg_dist": 0.0105},
        ],
    )
    output = tmp_path / "mask.json"

    rc = module.main(
        [
            "--baseline-xray",
            str(baseline),
            "--positive-xray",
            str(positive),
            "--negative-xray",
            str(negative),
            "--improvement-guard",
            "0.02",
            "--output-n-pairs",
            "5",
            "--output-json",
            str(output),
        ]
    )

    assert rc == 0
    payload = read_json(output)
    assert payload["pair_signs"] == [1, -1, 1, 0, 0]
    assert payload["measured_pairs"] == 3
    assert payload["active_pairs"] == 3
    assert payload["positive_pairs"] == 2
    assert payload["negative_pairs"] == 1
    assert payload["objective"] == "contest_score_linearized_at_baseline_mean_pose_v1"
    assert payload["predicted_component_no_rate_delta"] < 0.0
    assert payload["score_claim"] is False
