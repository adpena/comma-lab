# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from tac.local_acceleration.mlx_scorer_port_inventory import (
    STATUS_ADAPTER,
    STATUS_DIRECT,
    STATUS_UNKNOWN,
    classify_module_class,
    inspect_model,
)

REPO = Path(__file__).resolve().parents[3]


class UnknownLayer(nn.Module):
    def forward(self, x):  # pragma: no cover - inventory only
        return x


def test_classification_rules_keep_conv_adapter_and_linear_direct() -> None:
    assert classify_module_class("torch.nn.modules.linear.Linear").status == STATUS_DIRECT
    assert classify_module_class("torch.nn.modules.conv.Conv2d").status == STATUS_ADAPTER
    assert classify_module_class("vendor.CustomThing").status == STATUS_UNKNOWN


def test_inspect_model_counts_direct_parameters_and_unknown_blockers() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        UnknownLayer(),
        nn.Flatten(),
        nn.Linear(4, 2),
    )

    inv = inspect_model("synthetic", model)

    assert inv.parameter_count == sum(p.numel() for p in model.parameters())
    assert inv.parameter_bytes == sum(p.numel() * p.element_size() for p in model.parameters())
    assert inv.state_dict_key_count == len(model.state_dict())
    assert inv.state_dict_parameter_key_count == len(list(model.named_parameters()))
    assert inv.state_dict_buffer_key_count == len(list(model.named_buffers()))
    assert inv.status_counts[STATUS_ADAPTER] >= 2
    assert inv.status_counts[STATUS_DIRECT] >= 2
    assert inv.status_counts[STATUS_UNKNOWN] >= 1
    rows_by_class = {row.class_path: row for row in inv.rows}
    assert rows_by_class["torch.nn.modules.conv.Conv2d"].direct_parameter_count == 108
    assert rows_by_class["torch.nn.modules.linear.Linear"].direct_parameter_count == 10


def test_plan_mlx_scorer_port_cli_writes_fail_closed_inventory(tmp_path: Path) -> None:
    output = tmp_path / "inventory.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_scorer_port.py"),
            "--repo-root",
            str(REPO),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    summary = json.loads(completed.stdout)["summary"]
    assert summary["total_parameter_count"] > 20_000_000
    assert summary["total_state_dict_key_count"] > 1_000
    assert summary["full_mlx_port_claim_allowed"] is False
    assert summary["total_blocking_modules"] > 0
    assert any(
        blocker.startswith("state_dict_key_mapping_required:")
        for blocker in summary["claim_blockers"]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert {model["name"] for model in payload["models"]} == {"posenet", "segnet"}
