from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "plan_factorized_hnerv_relerr_schedule.py"
    spec = importlib.util.spec_from_file_location(
        "plan_factorized_hnerv_relerr_schedule", tool_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_low_rank_plan_is_fail_closed_but_writes_plan_config(tmp_path: Path) -> None:
    mod = _load_tool()
    json_out = tmp_path / "schedule.json"
    md_out = tmp_path / "schedule.md"
    plan_config = tmp_path / "plan_config.json"

    rc = mod.main([
        "--synthetic-low-rank",
        "--candidate-indices",
        "0,2",
        "--relerr-caps",
        "0.04,0.08",
        "--recommended-max-relerr",
        "0.08",
        "--json-out",
        str(json_out),
        "--markdown-out",
        str(md_out),
        "--plan-config-out",
        str(plan_config),
    ])

    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    config = json.loads(plan_config.read_text(encoding="utf-8"))
    assert payload["schema"] == "factorized_hnerv_relerr_schedule_plan.v1"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["recommended_schedule"]["ready_for_packet_build"] is True
    assert payload["recommended_schedule"]["selected_factorized_indices"]
    assert config == payload["recommended_plan_config"]
    assert "dual_axis_exact_eval_required_before_score_claim" in payload["global_blockers"]
    assert "Factorized HNeRV Rel-Err Schedule Plan" in md_out.read_text(encoding="utf-8")


def test_no_positive_schedule_refuses_plan_config(tmp_path: Path) -> None:
    mod = _load_tool()
    state = mod.synthetic_low_rank_state_dict()

    plan = mod.build_plan(
        state,
        substrate_label="synthetic_low_rank",
        candidate_indices=(0,),
        relerr_caps=(0.001,),
        recommended_max_relerr=0.001,
        brotli_quality=11,
    )

    assert plan["score_claim"] is False
    assert plan["recommended_schedule"] is None
    assert plan["recommended_plan_config"] is None
    assert plan["schedules"][0]["ready_for_packet_build"] is False


def test_rejects_bias_indices_before_estimating() -> None:
    mod = _load_tool()
    state = mod.synthetic_low_rank_state_dict()

    try:
        mod.build_plan(
            state,
            substrate_label="synthetic_low_rank",
            candidate_indices=(1,),
            relerr_caps=(0.05,),
        )
    except Exception as exc:
        assert "non-matrixizable candidate indices" in str(exc)
    else:
        raise AssertionError("expected non-matrixizable index rejection")
