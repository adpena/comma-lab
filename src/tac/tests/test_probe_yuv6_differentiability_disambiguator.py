# SPDX-License-Identifier: MIT
"""Tests for the YUV6 differentiability disambiguator CLI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "probe_yuv6_differentiability_disambiguator.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("probe_yuv6_tool", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arbitrate_prefers_monkey_patch_when_both_modes_pass() -> None:
    tool = _load_tool()
    result = tool._arbitrate(
        [
            {"mode": "monkey_patch_global", "available": True, "grad_l2": 2.0},
            {
                "mode": "tac_differentiable_routing",
                "available": True,
                "grad_l2": 2.0,
            },
        ]
    )

    assert result["recommendation"] == "monkey_patch_global"
    assert result["monkey_patch_global_passed"] is True
    assert result["tac_differentiable_routing_passed"] is True


def test_arbitrate_falls_back_to_tac_when_monkey_patch_unavailable() -> None:
    tool = _load_tool()
    result = tool._arbitrate(
        [
            {"mode": "monkey_patch_global", "available": False, "grad_l2": 0.0},
            {
                "mode": "tac_differentiable_routing",
                "available": True,
                "grad_l2": 1.0,
            },
        ]
    )

    assert result["recommendation"] == "tac_differentiable_routing"


def test_probe_cli_writes_non_promotable_json(tmp_path: Path) -> None:
    tool = _load_tool()
    out = tmp_path / "probe.json"
    rc = tool.main(["--output", str(out), "--seed", "7"])

    assert rc in {0, 3}
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["seed"] == 7
    assert "probes" in report
    assert "arbitration" in report
