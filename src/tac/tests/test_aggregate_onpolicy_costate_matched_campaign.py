from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools/aggregate_onpolicy_costate_matched_campaign.py"
SPEC = importlib.util.spec_from_file_location("_aggregate_task455", TOOL)
assert SPEC is not None and SPEC.loader is not None
aggregate_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aggregate_tool
SPEC.loader.exec_module(aggregate_tool)


def _receipt(regime: str, *, verdict: str = "NO-GO") -> dict:
    exact_trace = [
        {"step": 0, "ce": 0.2, "d_seg": 0.1, "d_pose": 1.0},
        {"step": 1, "ce": 0.1, "d_seg": 0.09, "d_pose": 0.9},
    ]
    target_trace = [dict(row) for row in exact_trace]
    if verdict == "NO-GO":
        target_trace[-1]["ce"] += 0.01
    timings = {
        name: {"count": 1, "total_seconds": value, "mean_seconds": value}
        for name, value in {
            "exact_forward_only": 2.0,
            "exact_costate_forward_backward": 4.0,
            "anchor_fit": 1.0,
            "surrogate_inference": 0.25,
            "renderer_vjp_exact_control": 0.5,
            "renderer_vjp_surrogate_target": 0.5,
            "surrogate_anchor_exact_costate": 4.0,
            "surrogate_nonanchor_operational_step": 1.0,
            "exact_window_operational_step": 10.0,
            "surrogate_window_operational_step": 5.0,
        }.items()
    }
    return {
        "schema": "onpolicy_costate_matched_window_probe.v1",
        "status": "MEASURED",
        "score_claim": False,
        "research_only": True,
        "config": {"regime": regime, "seed": 455},
        "false_authority_flags": {
            "score_claim": False,
            "mps_authority": False,
            "surrogate_eval_authority": False,
            "contest_cpu_or_cuda_eval": False,
        },
        "teacher_accounting": {
            "segnet_forward_reconciliation": "PASS",
            "posenet_forward_reconciliation": "PASS",
        },
        "run_contract": {"payload": {"source_custody": {"tool": {"sha256": "a" * 64}}}},
        "timing": {"measured": timings},
        "exact_trace": exact_trace,
        "surrogate_trace": target_trace,
        "deterministic_repeat_noise_floor": {"ce": 0.0, "d_seg": 0.0, "d_pose": 0.0},
        "fidelity_verdict": {"verdict": verdict},
        "mission_verdict": verdict,
        "mission_verdict_reason": "test",
        "collection_fit_rows": [{"admitted": verdict != "NO-GO"}],
        "window_economics": {
            "comparison_basis": aggregate_tool.RECEIPT_WINDOW_BASIS,
            "exact_window_operational_seconds": 10.0,
            "surrogate_window_operational_seconds": 5.0,
            "observed_window_steps": 1,
            "observed_exact_teacher_skip_fraction": 0.8,
            "target_cadence_fidelity_validated": verdict == "GO",
            "speedup": 2.0,
        },
    }


def _write_receipts(tmp_path: Path, *, verdicts: tuple[str, str, str]) -> dict[str, Path]:
    paths = {}
    for regime, verdict in zip(aggregate_tool.REQUIRED_REGIMES, verdicts, strict=True):
        path = tmp_path / f"{regime}.json"
        path.write_text(json.dumps(_receipt(regime, verdict=verdict)), encoding="utf-8")
        paths[regime] = path
    return paths


def test_campaign_aggregates_disjoint_timings_and_fail_closed_verdict(tmp_path: Path) -> None:
    payload = aggregate_tool.aggregate(
        _write_receipts(tmp_path, verdicts=("GO", "NO-GO", "GO"))
    )
    assert payload["mission_verdict"] == "NO-GO"
    assert payload["aggregate_isolated_timings"]["exact_forward_only"]["count"] == 3
    assert payload["forward_replacement_economics"][
        "measured_same-run_exact_forward_over_surrogate_inference_speedup"
    ] == 8.0
    assert payload["aggregate_whole_matched_window"]["comparison_basis"] == (
        "sum of symmetric complete per-step operational timers under each regime's "
        "exact-derived schedule; each step includes render, provider, renderer VJP, "
        "and candidate update; line-search and exact validation calls excluded"
    )
    assert payload["authority"]["score_claim"] is False
    assert "seed455" in payload["verdict_scope"]


def test_campaign_no_go_precedes_needs_more(tmp_path: Path) -> None:
    payload = aggregate_tool.aggregate(
        _write_receipts(tmp_path, verdicts=("NO-GO", "NEEDS-MORE", "GO"))
    )
    assert payload["mission_verdict"] == "NO-GO"


def test_campaign_needs_more_precedes_go_without_failure(tmp_path: Path) -> None:
    payload = aggregate_tool.aggregate(
        _write_receipts(tmp_path, verdicts=("GO", "NEEDS-MORE", "GO"))
    )
    assert payload["mission_verdict"] == "NEEDS-MORE"


def test_campaign_rejects_source_mismatch(tmp_path: Path) -> None:
    paths = _write_receipts(tmp_path, verdicts=("GO", "GO", "GO"))
    boundary = json.loads(paths["boundary"].read_text(encoding="utf-8"))
    boundary["run_contract"]["payload"]["source_custody"]["tool"]["sha256"] = "b" * 64
    paths["boundary"].write_text(json.dumps(boundary), encoding="utf-8")
    with pytest.raises(aggregate_tool.CampaignError, match="source bundle differs"):
        aggregate_tool.aggregate(paths)


def test_campaign_rejects_treatment_config_mismatch(tmp_path: Path) -> None:
    paths = _write_receipts(tmp_path, verdicts=("GO", "GO", "GO"))
    boundary = json.loads(paths["boundary"].read_text(encoding="utf-8"))
    boundary["config"]["seed"] = 999
    paths["boundary"].write_text(json.dumps(boundary), encoding="utf-8")
    with pytest.raises(aggregate_tool.CampaignError, match="treatment config differs"):
        aggregate_tool.aggregate(paths)


def test_campaign_rejects_asymmetric_or_component_sum_window_claim(tmp_path: Path) -> None:
    paths = _write_receipts(tmp_path, verdicts=("GO", "GO", "GO"))
    early = json.loads(paths["early"].read_text(encoding="utf-8"))
    early["window_economics"]["comparison_basis"] = (
        "isolated component sums; candidate update omitted"
    )
    paths["early"].write_text(json.dumps(early), encoding="utf-8")
    with pytest.raises(aggregate_tool.CampaignError, match="timing basis"):
        aggregate_tool.aggregate(paths)


def test_campaign_rejects_window_total_not_backed_by_step_timers(tmp_path: Path) -> None:
    paths = _write_receipts(tmp_path, verdicts=("GO", "GO", "GO"))
    early = json.loads(paths["early"].read_text(encoding="utf-8"))
    early["timing"]["measured"]["surrogate_window_operational_step"]["total_seconds"] = 4.0
    paths["early"].write_text(json.dumps(early), encoding="utf-8")
    with pytest.raises(aggregate_tool.CampaignError, match="does not reconcile"):
        aggregate_tool.aggregate(paths)


def test_campaign_rejects_declared_verdict_not_backed_by_raw_prefix(tmp_path: Path) -> None:
    paths = _write_receipts(tmp_path, verdicts=("GO", "GO", "GO"))
    early = json.loads(paths["early"].read_text(encoding="utf-8"))
    early["mission_verdict"] = "NO-GO"
    paths["early"].write_text(json.dumps(early), encoding="utf-8")
    with pytest.raises(aggregate_tool.CampaignError, match="raw prefix evidence"):
        aggregate_tool.aggregate(paths)


def test_raw_prefix_drift_alone_rejects_an_ema_admitted_formulation(tmp_path: Path) -> None:
    paths = _write_receipts(tmp_path, verdicts=("NO-GO", "GO", "GO"))
    early = json.loads(paths["early"].read_text(encoding="utf-8"))
    early["collection_fit_rows"][-1]["admitted"] = True
    paths["early"].write_text(json.dumps(early), encoding="utf-8")
    payload = aggregate_tool.aggregate(paths)
    early_row = next(row for row in payload["regimes"] if row["regime"] == "early")
    assert early_row["ema_final_admitted"] is True
    assert early_row["raw_exact_metric_trace_comparison"]["ce"][
        "within_floor_at_every_step"
    ] is False
    assert early_row["mission_verdict"] == "NO-GO"
