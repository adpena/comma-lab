# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO / "experiments" / "plan_yousfi_fridrich_field_equations.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("yf_field_planner_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _atom(atom_id: str, *, frame: int, y: int, x0: int, x1: int, cls: int, benefit: float) -> dict:
    charged = 6
    return {
        "atom_id": atom_id,
        "atom_family": "row_run",
        "identity": {
            "frame_index": frame,
            "pair_index": frame // 2,
            "class_id": cls,
            "y": y,
            "x0": x0,
            "x1_exclusive": x1,
        },
        "pair_indices": [frame // 2],
        "frame_indices": [frame],
        "class_ids": [cls],
        "residual_pixels": x1 - x0,
        "cost_model": {"estimated_charged_bytes": charged},
        "weights": {"weighted_residual_pixel_proxy": x1 - x0},
        "lagrangian": {
            "estimated_marginal_score_saved_proxy": benefit,
            "estimated_rate_score_cost": 0.000003995154,
            "estimated_lagrangian_net_proxy": benefit - 0.000003995154,
            "estimated_score_saved_per_charged_byte": benefit / charged,
        },
        "interaction_assumptions": ["fixture_first_order_row_run"],
        "score_claim": False,
        "evidence_grade": "planning_only",
    }


def _write_ledger(path: Path) -> None:
    payload = {
        "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
        "score_claim": False,
        "evidence_grade": "planning_only",
        "atom_count": 3,
        "tensor": {"shape": {"frames": 2, "height": 384, "width": 512}},
        "top_atoms": [
            _atom("a", frame=0, y=10, x0=20, x1=40, cls=1, benefit=0.20),
            _atom("b", frame=1, y=11, x0=50, x1=90, cls=2, benefit=0.10),
            _atom("c", frame=0, y=12, x0=100, x1=110, cls=1, benefit=0.05),
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_build_plan_emits_practical_and_ideal_equations(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger)

    payload = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "plan.json",
        mode="both",
        candidate_sizes=(1, 2),
        max_source_atoms=8,
        curvature_strength=0.0,
        pair_antagonism=0.0,
        frame_antagonism=0.0,
        class_synergy=0.0,
        policy_prefix="unit",
    )

    assert json.loads((tmp_path / "plan.json").read_text()) == payload
    assert payload["schema"] == "yousfi_fridrich_atom_field_allocator_v1"
    assert payload["score_claim"] is False
    assert payload["contest_practical_equations"]["variables"]["x_a"]
    assert payload["ideal_infinite_compute_equations"]["exact_objective"].startswith("min_A")
    assert [p["selected_atom_count"] for p in payload["candidate_policies"]] == [1, 2]
    assert payload["candidate_policies"][0]["policy_id"] == "unit_sparse_pair_frame_class_top0001"
    assert payload["candidate_policies"][0]["ready_for_exact_eval_dispatch"] is False
    assert payload["candidate_policies"][0]["dispatchable"] is False
    assert "field_policy_is_proxy_row" in payload["candidate_policies"][0]["dispatch_blockers"]
    assert "fixture_first_order_row_run" in payload["candidate_policies"][0]["interaction_assumptions"]
    assert payload["byte_closed_manifest_gate"]["candidate_policies_dispatchable"] is False
    assert payload["candidate_policies"][0]["selected_row_run_atoms"][0] == {
        "frame_index": 0,
        "y": 10,
        "x0": 20,
        "x1_exclusive": 40,
        "class_id": 1,
    }


def test_positive_proxy_only_filters_negative_atoms(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "ledger.json"
    payload = {
        "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
        "score_claim": False,
        "evidence_grade": "planning_only",
        "atom_count": 2,
        "top_atoms": [
            _atom("positive", frame=0, y=1, x0=1, x1=3, cls=1, benefit=0.01),
            _atom("negative", frame=0, y=2, x0=1, x1=3, cls=1, benefit=0.0),
        ],
    }
    payload["top_atoms"][1]["lagrangian"]["estimated_lagrangian_net_proxy"] = -1.0
    ledger.write_text(json.dumps(payload, sort_keys=True) + "\n")

    plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "plan.json",
        mode="contest",
        candidate_sizes=(1, 2),
        positive_proxy_only=True,
    )

    assert plan["atom_summary"]["deduped_row_run_atom_count"] == 1
    assert len(plan["candidate_policies"]) == 1


def test_cmg3a_policy_filters_unrepresentable_background_atoms(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "ledger.json"
    payload = {
        "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
        "score_claim": False,
        "evidence_grade": "planning_only",
        "atom_count": 2,
        "top_atoms": [
            _atom("background", frame=0, y=1, x0=1, x1=30, cls=0, benefit=1.0),
            _atom("foreground", frame=0, y=2, x0=1, x1=10, cls=2, benefit=0.01),
        ],
    }
    ledger.write_text(json.dumps(payload, sort_keys=True) + "\n")

    plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "plan.json",
        mode="contest",
        candidate_sizes=(1,),
        pair_antagonism=0.0,
        frame_antagonism=0.0,
        class_synergy=0.0,
    )

    assert plan["atom_summary"]["deduped_row_run_atom_count"] == 1
    assert plan["candidate_policies"][0]["selected_row_run_atoms"][0]["class_id"] == 2


def test_negative_field_energy_is_filtered_unless_explicitly_allowed(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger)

    filtered = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "filtered.json",
        mode="contest",
        candidate_sizes=(1,),
        pair_antagonism=1.0,
        frame_antagonism=1.0,
        class_synergy=0.0,
    )

    assert filtered["candidate_policies"] == []
    assert filtered["filtered_candidate_policies"]["negative_field_energy"][0]["selected_atom_count"] == 1

    allowed = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "allowed.json",
        mode="contest",
        candidate_sizes=(1,),
        pair_antagonism=1.0,
        frame_antagonism=1.0,
        class_synergy=0.0,
        allow_negative_field_energy=True,
    )

    assert len(allowed["candidate_policies"]) == 1
    assert allowed["candidate_policies"][0]["estimated_proxy"]["field_energy"] < 0.0


def test_policy_records_expected_base_runs_from_cmg3_ledger(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "ledger.json"
    payload = {
        "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
        "score_claim": False,
        "evidence_grade": "planning_only",
        "atom_count": 1,
        "inputs": {
            "candidate": {
                "mode": "reconstructed_from_cmg3_nonzero_row_runs_manifest",
                "max_runs_per_row": 2,
            }
        },
        "top_atoms": [_atom("a", frame=0, y=10, x0=20, x1=40, cls=1, benefit=0.20)],
    }
    ledger.write_text(json.dumps(payload, sort_keys=True) + "\n")

    plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "plan.json",
        mode="contest",
        candidate_sizes=(1,),
        pair_antagonism=0.0,
        frame_antagonism=0.0,
        class_synergy=0.0,
    )

    assert plan["inputs"][0]["expected_builder_base_runs_per_row"] == 2
    assert plan["candidate_policies"][0]["required_base_runs_per_row"] == 2

    payload["inputs"]["candidate"] = {
        "mode": "reconstructed_from_cmg3a_adaptive_manifest",
        "base_runs_per_row": 1,
    }
    ledger.write_text(json.dumps(payload, sort_keys=True) + "\n")
    cmg3a_plan = planner.build_plan(
        ledger_jsons=[ledger],
        output_json=tmp_path / "cmg3a_plan.json",
        mode="contest",
        candidate_sizes=(1,),
        pair_antagonism=0.0,
        frame_antagonism=0.0,
        class_synergy=0.0,
    )
    assert cmg3a_plan["candidate_policies"][0]["required_base_runs_per_row"] == 1


def test_rejects_score_claim_ledgers(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "bad.json"
    ledger.write_text(json.dumps({"score_claim": True, "top_atoms": []}) + "\n")

    try:
        planner.build_plan(ledger_jsons=[ledger], output_json=tmp_path / "plan.json")
    except planner.FieldPlanError as exc:
        assert "score_claim=true" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FieldPlanError")


def test_rejects_dispatchable_proxy_atoms(tmp_path: Path) -> None:
    planner = _load_planner()
    ledger = tmp_path / "bad-dispatchable.json"
    payload = {
        "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
        "score_claim": False,
        "evidence_grade": "planning_only",
        "atom_count": 1,
        "top_atoms": [_atom("dispatchable_proxy", frame=0, y=1, x0=1, x1=3, cls=1, benefit=0.01)],
    }
    payload["top_atoms"][0]["ready_for_exact_eval_dispatch"] = True
    ledger.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(planner.FieldPlanError, match="dispatchable proxy row"):
        planner.build_plan(
            ledger_jsons=[ledger],
            output_json=tmp_path / "plan.json",
            mode="contest",
            candidate_sizes=(1,),
        )
