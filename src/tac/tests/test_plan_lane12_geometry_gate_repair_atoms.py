# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_lane12_geometry_gate_repair_atoms.py"
SPEC = importlib.util.spec_from_file_location("plan_lane12_geometry_gate_repair_atoms", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _geometry_payload() -> dict[str, Any]:
    return {
        "diagnostic": "alpha_geo_0_nerv_geometry",
        "schema_version": 1,
        "score_evidence_grade": "empirical",
        "scorer_proxy": False,
        "shape": {"frames": 1200, "height": 384, "width": 512, "num_classes": 5},
        "global": {"global_disagreement": 0.0123, "disagreement_pixels": 1000, "total_pixels": 235929600},
        "boundary_bands": {
            "1": {"disagreement_rate": 0.2, "disagreement_pixels": 100, "band_pixels": 500},
            "2": {"disagreement_rate": 0.1, "disagreement_pixels": 90, "band_pixels": 900},
        },
        "temporal": {
            "pair_transition": {"disagreement_rate": 0.009, "disagreement_pixels": 120},
            "stable_region": {"false_flip_rate": 0.003, "false_flip_pixels": 60},
            "worst_frame_pairs": [
                {
                    "pair_index": 1,
                    "frames": [1, 2],
                    "transition_disagreement_pixels": 9,
                    "transition_disagreement_rate": 0.03,
                    "stable_false_flip_pixels": 3,
                    "stable_false_flip_rate": 0.001,
                    "pair_frame_disagreement_rate": 0.02,
                }
            ],
        },
        "components": {
            "centroid": {"max_matched_jump_px": 5.0, "missing_component_rate": 0.4}
        },
        "pass_fail": {"overall_pass": False, "checks": {}, "thresholds": {}},
        "residual_region_ranking": {
            "diagnostic": "alpha_geo_residual_region_ranking",
            "regions": [
                {
                    "rank": 1,
                    "residual_region_id": "f0001_c0001",
                    "frame": 1,
                    "box_xyxy": [1, 1, 5, 4],
                    "area_px": 12,
                    "critical_class_pixels": 7,
                    "boundary_band_pixels": 5,
                    "temporal_transition_disagreement_pixels": 4,
                    "dominant_baseline_class": 1,
                    "dominant_candidate_class": 0,
                    "priority_bucket": 0,
                    "priority_label": "lower_field_lane_marking",
                    "suggested_repair": "lane_lower_field_residual",
                    "confusion_pairs": [
                        {"baseline_class": 1, "candidate_class": 0, "pixels": 7}
                    ],
                }
            ],
        },
    }


def _primitive_contract_payload() -> dict[str, Any]:
    return {
        "diagnostic": "alpha_geo_primitive_contract_v1",
        "schema_version": 1,
        "score_claim_eligible": False,
        "promotion_eligible": False,
        "exact_eval_claim": False,
        "ranked_critical_boxes": [
            {
                "rank": 1,
                "frame": 1,
                "box_xyxy": [1, 1, 5, 4],
                "area_px": 12,
                "class_id": 1,
                "class_name": "lane_marking",
                "failure_type": "missing_component",
                "pose_sensitive": True,
                "mask_iou": 0.0,
                "box_iou": 0.0,
            }
        ],
        "worst_transition_pairs": [
            {
                "rank": 1,
                "pair_index": 1,
                "frames": [1, 2],
                "transition_disagreement_pixels": 9,
                "transition_disagreement_rate": 0.03,
                "stable_false_flip_pixels": 3,
                "stable_false_flip_rate": 0.001,
            }
        ],
    }


def _write_mask_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    baseline = torch.zeros((1200, 384, 512), dtype=torch.uint8)
    candidate = torch.zeros((1200, 384, 512), dtype=torch.uint8)
    baseline[1, 1:4, 1:5] = 1
    candidate[1, 1:4, 1:5] = 0
    torch.save(baseline, cache / "baseline.pt")
    torch.save(candidate, cache / "candidate.pt")
    _write_json(
        cache / "baseline.json",
        {
            "tensor_file": "baseline.pt",
            "decoded_mask_sha256": "b" * 64,
            "decoded_mask_shape": [1200, 384, 512],
            "fingerprint": {"archive_member_resolved": "masks.mkv", "source_sha256": "a" * 64},
        },
    )
    _write_json(
        cache / "candidate.json",
        {
            "tensor_file": "candidate.pt",
            "decoded_mask_sha256": "c" * 64,
            "decoded_mask_shape": [1200, 384, 512],
            "fingerprint": {"archive_member_resolved": "masks.nrv", "source_sha256": "d" * 64},
        },
    )
    return cache


def test_geometry_repair_atom_plan_uses_tensor_cache_and_stays_non_promotable(tmp_path: Path) -> None:
    geometry = _write_json(tmp_path / "geometry.json", _geometry_payload())
    contract = _write_json(tmp_path / "contract.json", _primitive_contract_payload())
    cache = _write_mask_cache(tmp_path)
    output = tmp_path / "plan.json"

    payload = planner.build_geometry_repair_atom_plan(
        geometry_json=geometry,
        primitive_contract_json=contract,
        mask_cache_dir=cache,
        output_json=output,
        max_region_atoms=1,
        max_critical_box_atoms=1,
        max_transition_atoms=1,
        policy_budgets=(64, 512),
    )

    assert output.read_text() == json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert payload["schema"] == "lane12_geometry_gate_repair_atom_plan_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["exact_eval_claim"] is False
    assert payload["remote_jobs_dispatched"] is False
    assert payload["byte_closed_exact_eval_candidate_created"] is False
    assert payload["inputs"]["tensor_status"]["loaded"] is True
    assert payload["atom_count"] == 3
    assert payload["atoms"][0]["cost_model"]["charge_status"] == "planning_estimate_not_archive_measured"
    assert any(atom["tensor_measurement"] and atom["tensor_measurement"]["changed_pixels"] == 12 for atom in payload["atoms"])
    assert payload["candidate_policies"][0]["dispatch_allowed"] is False
    assert payload["candidate_policies"][1]["selected_atom_count"] >= 1


def test_geometry_repair_atom_plan_falls_back_without_tensors(tmp_path: Path) -> None:
    geometry = _write_json(tmp_path / "geometry.json", _geometry_payload())
    output = tmp_path / "plan.json"

    payload = planner.build_geometry_repair_atom_plan(
        geometry_json=geometry,
        primitive_contract_json=None,
        mask_cache_dir=tmp_path / "missing_cache",
        output_json=output,
        max_region_atoms=1,
        max_critical_box_atoms=0,
        max_transition_atoms=1,
        load_tensors=False,
        policy_budgets=(512,),
    )

    assert payload["inputs"]["tensor_status"]["loaded"] is False
    assert payload["inputs"]["tensor_status"]["reason"] == "not_requested"
    assert payload["atom_count"] == 2
    region = next(atom for atom in payload["atoms"] if atom["atom_kind"] == "residual_region_patch")
    assert region["tensor_measurement"] is None
    assert region["cost_model"]["model"] == "estimated_from_area"
    assert payload["candidate_policies"][0]["builder_status"] == "not_byte_closed_no_archive_emitted"


def test_geometry_repair_atom_plan_rejects_wrong_diagnostic(tmp_path: Path) -> None:
    geometry_payload = _geometry_payload()
    geometry_payload["diagnostic"] = "wrong"
    geometry = _write_json(tmp_path / "geometry.json", geometry_payload)

    try:
        planner.build_geometry_repair_atom_plan(
            geometry_json=geometry,
            primitive_contract_json=None,
            mask_cache_dir=None,
            output_json=tmp_path / "plan.json",
        )
    except planner.Lane12GeometryRepairPlannerError as exc:
        assert "alpha_geo_0_nerv_geometry" in str(exc)
    else:
        raise AssertionError("expected planner error")
