from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reconcile_vast_dispatch_state.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_reconcile_vast_dispatch_state", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_active_dispatches_only_active_section() -> None:
    mod = _load_module()
    text = """
# Active Dispatches

## Active

| timestamp_utc | lane_label | instance_id | predicted_band | est_cost | ETA | kill_criteria | log_dir |
|---|---|---|---|---|---|---|---|
| 2026-04-30T12:51Z | lane_12_nerv_2026-04-30_b | (in flight, attempts in progress; PID 4889) | [1.00, 1.10] | $0.85 | ~3.4h | auth eval | /tmp/a |
| 2026-04-30T12:53Z | lane_19_logit_margin_2026-04-30_b | 35899850 (_a4, running) | [0.85, 1.00] | $1.50 | ~5h | auth eval | /tmp/b |

## Completed

| timestamp_utc | lane_label | instance_id | result | notes |
|---|---|---|---|---|
| 2026-04-30T12:31Z | old | 12345 | done | no |
"""
    rows = mod.parse_active_dispatches(text)

    assert len(rows) == 2
    assert rows[0].lane_label == "lane_12_nerv_2026-04-30_b"
    assert rows[0].instance_id is None
    assert rows[1].instance_id == "35899850"


def test_reconcile_reports_all_three_drift_classes() -> None:
    mod = _load_module()
    live = [
        {
            "id": 35899850,
            "label": "lane_19_logit_margin_2026-04-30_b_a4",
            "actual_status": "running",
            "ssh_host": "ssh2.vast.ai",
            "ssh_port": 19850,
        },
        {
            "id": 35906669,
            "label": "lane_sa_segmap_clone_2026-04-30_codex_a2",
            "actual_status": "running",
            "ssh_host": "ssh2.vast.ai",
            "ssh_port": 26668,
        },
    ]
    tracker = [
        {"instance_id": "35899850", "label": "lane_19_logit_margin_2026-04-30_b_a4"},
        {"instance_id": "11111111", "label": "stale_gone"},
    ]
    active_rows = [
        mod.ActiveDispatchRow(
            lane_label="lane_19_logit_margin_2026-04-30_b",
            instance_id="35899435",
            raw_instance_cell="35899435 (_a1, running)",
            timestamp_utc="2026-04-30T12:53Z",
        ),
        mod.ActiveDispatchRow(
            lane_label="lane_12_nerv_2026-04-30_b",
            instance_id=None,
            raw_instance_cell="(in flight)",
            timestamp_utc="2026-04-30T12:51Z",
        ),
    ]

    report = mod.reconcile(live, tracker, active_rows)

    assert report["tracker_missing_live"] == [
        {"instance_id": "11111111", "label": "stale_gone", "registered_at_utc": None}
    ]
    assert report["live_missing_tracker"] == [{
        "instance_id": "35906669",
        "label": "lane_sa_segmap_clone_2026-04-30_codex_a2",
        "actual_status": "running",
        "ssh_host": "ssh2.vast.ai",
        "ssh_port": 26668,
    }]
    assert report["active_missing_live"][0]["instance_id"] == "35899435"
    assert report["active_label_without_live_prefix"][0]["lane_label"] == "lane_12_nerv_2026-04-30_b"
    assert report["live_missing_active"][0]["instance_id"] == "35906669"


def test_attempt_suffix_normalization() -> None:
    mod = _load_module()
    assert (
        mod.normalize_attempt_label("lane_sa_segmap_clone_2026-04-30_codex_a2")
        == "lane_sa_segmap_clone_2026-04-30_codex"
    )
    assert mod.normalize_attempt_label("lane_19_logit_margin") == "lane_19_logit_margin"
