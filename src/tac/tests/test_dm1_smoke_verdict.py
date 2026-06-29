# SPDX-License-Identifier: MIT
"""Tests for the DM1 decisive-smoke executable verdict harvester (review M1 + M2).

Locks the firewall taxonomy GO / SYMPTOM / MEANS_FALSIFIED / VOID + the
self-calibrating PR bar + the log-parsing (shadow-PR preference, daemon-prefix
tolerance). Pure-function adjudication => no GPU/MLX needed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "dm1_smoke_verdict.py"
    spec = importlib.util.spec_from_file_location("dm1_smoke_verdict", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOL = _load_tool()


def _arm(pr_start, pr_end, dseg_start, dseg_end):
    return {
        "pr_start": pr_start, "pr_start_epoch": 300, "pr_end": pr_end, "pr_end_epoch": 399,
        "dseg_start": dseg_start, "dseg_start_epoch": 299, "dseg_end": dseg_end, "dseg_end_epoch": 399,
        "exists": True,
    }


# --- adjudicate(): the 4-way firewall taxonomy + self-calibrating bar ---
def test_go_when_pr_held_and_dseg_improves():
    arms = {
        "A0_baseline": _arm(3.34, 1.30, 0.0050, 0.0070),  # PR collapses, d_seg degrades = disease present
        "A3_minimal":  _arm(3.34, 3.10, 0.0050, 0.0050),  # PR held high, d_seg << 0.9*A0_end
    }
    r = _TOOL.adjudicate(arms)
    assert r["verdict"] == "GO", r
    # bar = a0_end + 0.5*(start-a0_end) = 1.30 + 0.5*(3.34-1.30) = 2.32
    assert r["pr_bar"] == pytest.approx(2.32)
    assert r["a3_holds_pr"] and r["a3_improves_dseg"]


def test_symptom_when_pr_held_but_dseg_flat():
    arms = {
        "A0_baseline": _arm(3.34, 1.30, 0.0050, 0.0070),
        "A3_minimal":  _arm(3.34, 3.10, 0.0050, 0.0069),  # PR held but d_seg ~ A0_end (not >=10% lower)
    }
    r = _TOOL.adjudicate(arms)
    assert r["verdict"] == "SYMPTOM", r
    assert r["a3_holds_pr"] and not r["a3_improves_dseg"]


def test_means_falsified_when_pr_not_held():
    arms = {
        "A0_baseline": _arm(3.34, 1.30, 0.0050, 0.0070),
        "A3_minimal":  _arm(3.34, 1.40, 0.0050, 0.0040),  # PR also collapses (< bar 2.32) even if d_seg ok
    }
    r = _TOOL.adjudicate(arms)
    assert r["verdict"] == "MEANS_FALSIFIED", r
    assert not r["a3_holds_pr"]


def test_void_when_disease_not_reproduced_pr():
    # A0 PR barely drops (< 15%) => disease NOT reproduced => VOID regardless of A3.
    arms = {
        "A0_baseline": _arm(3.34, 3.20, 0.0050, 0.0070),
        "A3_minimal":  _arm(3.34, 3.30, 0.0050, 0.0040),
    }
    r = _TOOL.adjudicate(arms)
    assert r["verdict"] == "VOID", r


def test_void_when_a0_dseg_does_not_degrade():
    # PR collapses but A0 d_seg does NOT degrade => disease not reproduced on the end axis => VOID.
    arms = {
        "A0_baseline": _arm(3.34, 1.30, 0.0050, 0.0050),
        "A3_minimal":  _arm(3.34, 3.10, 0.0050, 0.0040),
    }
    r = _TOOL.adjudicate(arms)
    assert r["verdict"] == "VOID", r


def test_incomplete_when_arm_missing_endpoints():
    arms = {"A0_baseline": _arm(3.34, 1.30, 0.0050, 0.0070), "A3_minimal": {"exists": False}}
    r = _TOOL.adjudicate(arms)
    assert r["verdict"] == "INCOMPLETE", r


def test_bar_is_self_calibrating_not_constant():
    # different A0 trajectories => DIFFERENT bars (M2: derived from A0's measured endpoints, NOT a
    # hardcoded ~3.0 / ~1.2 l7-era constant). bar = a0_end + hold_frac*(start - a0_end).
    a = _TOOL.adjudicate({"A0_baseline": _arm(4.0, 1.0, 0.005, 0.007),
                          "A3_minimal": _arm(4.0, 3.9, 0.005, 0.004)})
    b = _TOOL.adjudicate({"A0_baseline": _arm(3.0, 1.0, 0.005, 0.007),
                          "A3_minimal": _arm(3.0, 2.9, 0.005, 0.004)})
    assert a["pr_bar"] == 2.5   # 1.0 + 0.5*(4.0-1.0)
    assert b["pr_bar"] == 2.0   # 1.0 + 0.5*(3.0-1.0) -> a DIFFERENT bar from the same formula
    # hold_frac moves the bar (recover MORE of the collapse => higher bar):
    c = _TOOL.adjudicate({"A0_baseline": _arm(4.0, 1.0, 0.005, 0.007),
                          "A3_minimal": _arm(4.0, 3.9, 0.005, 0.004)}, hold_frac=0.75)
    assert c["pr_bar"] == 3.25  # 1.0 + 0.75*(4.0-1.0)


# --- log parsing: shadow-PR preference + daemon-prefix tolerance ---
def test_parse_prefers_shadow_pr_and_tolerates_prefixes(tmp_path: Path):
    log = tmp_path / "daemon.log"
    rows = [
        "[durable-daemon] launching ...",            # non-JSON prefix line (must be skipped)
        json.dumps({"stage": "verdict", "epoch": 299, "d_seg": 0.0050}),
        json.dumps({"stage": "dm1_telemetry", "epoch": 300, "pr_film_M": 3.0, "pr_film_M_shadow": 2.5}),
        "[daemon] " + json.dumps({"stage": "dm1_telemetry", "epoch": 399, "pr_film_M": 1.5, "pr_film_M_shadow": 1.2}),
        json.dumps({"stage": "verdict", "epoch": 399, "d_seg": 0.0070}),
        "garbage line { not json",
    ]
    log.write_text("\n".join(rows))
    arm = _TOOL.parse_arm(log)
    # shadow PR preferred over live; daemon-prefixed JSON still parsed
    assert arm["pr_start"] == 2.5 and arm["pr_end"] == 1.2
    assert arm["dseg_start"] == 0.0050 and arm["dseg_end"] == 0.0070
    assert arm["n_telemetry"] == 2 and arm["n_verdict"] == 2


def test_parse_falls_back_to_live_pr_when_shadow_absent(tmp_path: Path):
    log = tmp_path / "daemon.log"
    log.write_text(json.dumps({"stage": "dm1_telemetry", "epoch": 300, "pr_film_M": 3.1}))
    arm = _TOOL.parse_arm(log)
    assert arm["pr_start"] == 3.1
