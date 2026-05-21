# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.local_acceleration.mlx_scorer_state_map import (
    STATUS_MAPPED,
    STATUS_REQUIRES_ADAPTER,
    STATUS_UNUSED_EVAL_BUFFER,
)

REPO = Path(__file__).resolve().parents[3]


def test_plan_mlx_scorer_state_map_cli_maps_all_upstream_keys(tmp_path: Path) -> None:
    output = tmp_path / "state_map.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_scorer_state_map.py"),
            "--repo-root",
            str(REPO),
            "--load-weights",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(completed.stdout)
    assert stdout["weights_loaded"] is True
    summary = stdout["summary"]
    assert summary["state_key_count"] == 1160
    assert summary["hard_unmapped_key_count"] == 0
    assert summary["requires_module_adapter_key_count"] > 0
    assert summary["full_mlx_port_claim_allowed"] is False

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = {f'{row["model_name"]}:{row["state_key"]}': row for row in payload["rows"]}
    assert rows["posenet:vision.stem.0.conv_kxk.0.conv.weight"]["transform_policy"] == (
        "conv2d_weight_oihw_to_ohwi"
    )
    assert rows["posenet:vision.stem.0.conv_kxk.0.conv.weight"]["target_shape"] == [
        64,
        3,
        3,
        12,
    ]
    assert rows["posenet:summarizer.0.weight"]["transform_policy"] == (
        "linear_weight_identity_out_in"
    )
    assert rows["posenet:_mean"]["mapping_status"] == STATUS_MAPPED
    bn_counter_rows = [
        row for row in payload["rows"] if row["state_key"].endswith("num_batches_tracked")
    ]
    assert bn_counter_rows
    assert {row["mapping_status"] for row in bn_counter_rows} == {STATUS_UNUSED_EVAL_BUFFER}
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False


def test_state_map_status_counts_are_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "state_map.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_scorer_state_map.py"),
            "--repo-root",
            str(REPO),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    counts = payload["summary"]["status_counts"]
    assert counts[STATUS_MAPPED] > 0
    assert counts[STATUS_REQUIRES_ADAPTER] > 0
    assert "state_map_is_layout_plan_not_full_scorer_port" in payload["summary"]["claim_blockers"]
