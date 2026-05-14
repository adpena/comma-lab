# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILER_PATH = REPO_ROOT / "experiments" / "profile_atom_ledger_subspaces.py"


def _load_profiler():
    spec = importlib.util.spec_from_file_location("atom_subspace_profiler_test", PROFILER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _atom(atom_id: str, *, pair: int, frame: int, cls: int, benefit: float) -> dict:
    return {
        "atom_family": "row_run",
        "atom_id": atom_id,
        "bbox_xyxy": [10 + frame, 20, 30 + frame, 21],
        "candidate_class_histogram_pixels": {"3": 20},
        "class_ids": [cls],
        "cost_model": {"estimated_charged_bytes": 6},
        "evidence_grade": "planning_only",
        "frame_indices": [frame],
        "identity": {
            "class_id": cls,
            "frame_index": frame,
            "pair_index": pair,
            "x0": 10,
            "x1_exclusive": 30,
            "y": 20,
        },
        "lagrangian": {
            "estimated_lagrangian_net_proxy": benefit - 0.000003995154,
            "estimated_marginal_score_saved_proxy": benefit,
        },
        "pair_indices": [pair],
        "residual_pixels": 20,
        "score_claim": False,
        "source_class_histogram_pixels": {str(cls): 20},
    }


def _write_ledger(path: Path, *, score_claim: bool | str = False) -> Path:
    payload = {
        "atom_count": 3,
        "evidence_grade": "planning_only",
        "schema": "cmg3_pixel_lagrangian_atom_ledger_v1",
        "score_claim": score_claim,
        "top_atoms": [
            _atom("a", pair=7, frame=14, cls=0, benefit=0.030),
            _atom("b", pair=7, frame=15, cls=0, benefit=0.020),
            _atom("c", pair=9, frame=18, cls=2, benefit=0.010),
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def test_profile_emits_active_subspaces_without_score_claim(tmp_path: Path) -> None:
    profiler = _load_profiler()
    ledger = _write_ledger(tmp_path / "ledger.json")

    payload = profiler.build_profile(
        ledger_paths=[ledger],
        output_json=tmp_path / "profile.json",
        top_k=3,
        table_limit=4,
    )

    assert json.loads((tmp_path / "profile.json").read_text()) == payload
    assert payload["schema"] == "atom_ledger_active_subspace_profile_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["evidence_grade"] == "planning_only"
    pairs = payload["profiles"][0]["subspaces"]["pairs"]
    assert pairs[0]["key"] == "7"
    assert pairs[0]["hit_count"] == 2
    assert pairs[0]["estimated_marginal_score_saved_proxy_sum"] == 0.05
    source_to_candidate = payload["profiles"][0]["subspaces"]["source_to_candidate"]
    assert source_to_candidate[0]["key"] == "0->3"
    aggregate_pairs = payload["aggregate_subspaces"]["pairs"]
    assert aggregate_pairs[0]["key"] == "7"
    assert aggregate_pairs[0]["profile_hit_count"] == 1
    assert payload["fridrich_yousfi_signal_surface"]["low_dimensional_consensus"]["top_pairs"][0] == 7
    assert payload["fridrich_yousfi_signal_surface"]["score_claim"] is False


def test_profile_records_pairwise_overlap(tmp_path: Path) -> None:
    profiler = _load_profiler()
    left = _write_ledger(tmp_path / "left.json")
    right_payload = json.loads(left.read_text())
    right_payload["top_atoms"][0]["atom_id"] = "different"
    right = tmp_path / "right.json"
    right.write_text(json.dumps(right_payload, sort_keys=True) + "\n")

    payload = profiler.build_profile(
        ledger_paths=[left, right],
        output_json=tmp_path / "profile.json",
        top_k=2,
        overlap_top_k=2,
    )

    assert payload["pairwise_overlaps"][0]["atom_id_overlap"] == 1
    assert payload["pairwise_overlaps"][0]["pair_overlap"] == 1
    assert payload["fridrich_yousfi_signal_surface"]["overlap_summary"]["comparison_count"] == 1


def test_score_claim_inputs_fail_closed(tmp_path: Path) -> None:
    profiler = _load_profiler()
    ledger = _write_ledger(tmp_path / "bad.json", score_claim="true")

    with pytest.raises(ValueError, match="score-claim input ledgers"):
        profiler.build_profile(
            ledger_paths=[ledger],
            output_json=tmp_path / "profile.json",
        )
