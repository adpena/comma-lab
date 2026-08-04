# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

from tac.crossrun_trajectory_mining import analyze_frames, harvest_roots


def test_harvests_loss_terms_gates_and_config(tmp_path):
    run = tmp_path / "burn" / "window_01"
    run.mkdir(parents=True)
    telemetry = run / "telemetry.jsonl"
    rows = [
        {"event": "start", "cfg": {"gate_every": 5, "num_pairs": 600}, "epoch": 10},
        {"event": "resume", "epoch": 10, "stage": "seg"},
        {"event": "epoch", "epoch": 11, "ep_loss": 1.0},
        {
            "event": "a1_gate",
            "epoch": 15,
            "realized_gate_dseg_mean": 0.0100,
            "a1_classification": "FIRST_GATE",
            "topology_per_class": {"gt_components_erased": [1, 4, 0, 0, 0]},
        },
        {"stage": "loss_terms", "epoch": 15, "terms": {"seg": 0.7, "rate": 0.1}},
        {"event": "epoch", "epoch": 16, "ep_loss": 0.6},
        {
            "event": "a1_gate",
            "epoch": 20,
            "realized_gate_dseg_mean": 0.01001,
            "a1_classification": "A1_REALIZATION_GAP_ALARM",
            "topology_per_class": {"gt_components_erased": [1, 5, 0, 0, 0]},
        },
        {"event": "confound_alarm", "kind": "term_domination", "term": "seg", "epoch": 20},
        {"event": "epoch", "epoch": 21, "ep_loss": 0.5},
    ]
    telemetry.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    result = harvest_roots([tmp_path])
    metrics = {f.metric for f in result.frames}
    assert "cfg.gate_every" in metrics
    assert "loss_term.seg" in metrics
    assert "realized_gate_dseg_mean" in metrics
    assert "topology_per_class.gt_components_erased.class_1" in metrics

    analysis = analyze_frames(result.frames)
    assert analysis["event_timing"]["all_cadence_matched_runs"] == 1
    assert analysis["plateau_census"]["plateau_segments_threshold_abs_rel_lte_0p005"] == 1
    assert analysis["warm_resume"]["resume_events"] == 1


def test_markdown_json_fences_are_harvested(tmp_path):
    memo = tmp_path / "costate_organ_trajectory_ledger.md"
    memo.write_text(
        "# memo\n\n```json\n"
        + json.dumps({"run_ref": "r1", "n_intervals": 8, "axis_tag": "[macOS advisory]"})
        + "\n```\n",
        encoding="utf-8",
    )

    result = harvest_roots([tmp_path])
    assert result.to_summary()["records_parsed"] == 1
    assert any(f.metric == "n_intervals" and f.value == 8 for f in result.frames)

