# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.decoder_q_pairset_acquisition import (
    FALSE_ACQUISITION_AUTHORITY,
    build_decoder_q_pairset_acquisition_plan,
)
from tac.optimization.pair_frame_scorer_geometry_lattice import (
    REQUEST_SCHEMA,
    SCHEMA,
    PairFrameScorerGeometryLatticeError,
    build_pair_frame_scorer_geometry_lattice,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _selector_pareto(pair_count: int = 8) -> dict[str, object]:
    pairs = list(range(pair_count))
    return {
        "schema": "decoder_q_selective_selector_pareto.v1",
        **_false_authority(),
        "summary": {"recommended_selector_id": "best_top"},
        "candidates": [
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "best_top",
                "selector_kind": "top_rank_prefix",
                "selector_rank": 1,
                "rank_order_pair_indices": pairs,
                "selected_pair_indices": pairs,
                "selected_pair_count": len(pairs),
                "payload_bytes": 64,
                "predicted_score_mean": 0.19203,
            }
        ],
    }


def _pairset_acquisition(pair_count: int = 8) -> dict[str, object]:
    return build_decoder_q_pairset_acquisition_plan(
        _selector_pareto(pair_count),
        prefix_ks=[pair_count],
        diversity_ks=[pair_count],
        max_drop_two=0,
        max_swap_in=0,
        include_drop_one=False,
    )


def _frame_pair_curriculum(pair_count: int = 8) -> dict[str, object]:
    return {
        "schema": "frame_pair_curriculum_fixture.v1",
        **FALSE_ACQUISITION_AUTHORITY,
        "pair_rows": [
            {
                "pair_index": pair,
                "total_l1": float(pair_count - pair),
                "axis_mix": {
                    "seg_share": 0.25,
                    "pose_share": 0.70,
                    "rate_share": 0.05,
                },
            }
            for pair in range(pair_count)
        ],
    }


def _component_xray(pair_count: int = 8) -> dict[str, object]:
    return {
        "schema": "pair_component_error_xray_v1",
        **FALSE_ACQUISITION_AUTHORITY,
        "rows": [
            {
                "pair_idx": pair,
                "pose_score_contribution": float(pair_count - pair) * 0.25,
                "seg_score_contribution": float(pair_count - pair) * 0.75,
            }
            for pair in range(pair_count)
        ],
    }


def test_pair_frame_lattice_emits_false_authority_drop_requests() -> None:
    payload = build_pair_frame_scorer_geometry_lattice(
        _pairset_acquisition(),
        frame_pair_curriculum=_frame_pair_curriculum(),
        pair_component_xrays=(_component_xray(),),
        drop_counts=(3, 4),
        max_requests=8,
    )

    assert payload["schema"] == SCHEMA
    assert payload["coverage"]["geometry_coverage"] == pytest.approx(1.0)
    assert payload["summary"]["queue_executable_request_count"] == 2
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False

    requests = payload["queue_executable_pairset_drop_requests"]
    assert [request["schema"] for request in requests] == [REQUEST_SCHEMA, REQUEST_SCHEMA]
    assert [len(request["dropped_pair_indices"]) for request in requests] == [3, 4]
    for request in requests:
        assert request["queue_executable"] is True
        assert request["score_claim"] is False
        assert request["ready_for_exact_eval_dispatch"] is False
        assert set(request["dropped_pair_indices"]).isdisjoint(
            request["selected_pair_indices"]
        )


def test_pair_frame_lattice_binds_into_pairset_acquisition() -> None:
    source_plan = _pairset_acquisition()
    lattice = build_pair_frame_scorer_geometry_lattice(
        source_plan,
        frame_pair_curriculum=_frame_pair_curriculum(),
        pair_component_xrays=(_component_xray(),),
        drop_counts=(3, 4),
    )

    plan = build_decoder_q_pairset_acquisition_plan(
        _selector_pareto(),
        prefix_ks=[8],
        diversity_ks=[8],
        max_drop_two=0,
        max_swap_in=0,
        include_drop_one=False,
        pair_frame_geometry_lattice=lattice,
    )

    assert plan["summary"]["pair_frame_geometry_candidate_count"] == 2
    binding = plan["selection_policy"]["pair_frame_geometry_lattice"]
    assert binding["active"] is True
    assert binding["queue_executable_request_count"] == 2
    assert "global_low_impact_full_pair_drop_probe" in plan["selection_policy"][
        "eureka_expansion"
    ]["executable_families_this_pass"]

    rows = [
        row
        for row in plan["candidates"]
        if row["selector_kind"] == "pair_frame_geometry_low_impact_drop_many"
    ]
    assert len(rows) == 2
    assert all(
        row["acquisition_operation"]["op"] == "pair_frame_geometry_low_impact_drop_many"
        for row in rows
    )
    assert all(row["score_claim"] is False for row in rows)


def test_pair_frame_lattice_rejects_authority() -> None:
    authoritative = _pairset_acquisition()
    authoritative["score_claim"] = True
    with pytest.raises(PairFrameScorerGeometryLatticeError, match="score_claim"):
        build_pair_frame_scorer_geometry_lattice(authoritative)


def test_pair_frame_lattice_cli_and_pairset_cli_round_trip(tmp_path: Path) -> None:
    pairset_path = tmp_path / "pairset_acquisition.json"
    curriculum_path = tmp_path / "frame_pair_curriculum.json"
    xray_path = tmp_path / "pair_component_xray.json"
    lattice_path = tmp_path / "pair_frame_lattice.json"
    lattice_md = tmp_path / "pair_frame_lattice.md"
    rebound_path = tmp_path / "rebound_pairset.json"
    pairset_path.write_text(json.dumps(_pairset_acquisition()), encoding="utf-8")
    curriculum_path.write_text(json.dumps(_frame_pair_curriculum()), encoding="utf-8")
    xray_path.write_text(json.dumps(_component_xray()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_pair_frame_scorer_geometry_lattice.py"),
            "--pairset-acquisition",
            str(pairset_path),
            "--frame-pair-curriculum",
            str(curriculum_path),
            "--pair-component-xray",
            str(xray_path),
            "--drop-counts",
            "3,4",
            "--json-out",
            str(lattice_path),
            "--md-out",
            str(lattice_md),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["queue_executable_request_count"] == 2
    assert stdout["score_claim"] is False
    assert "Pair-Frame Scorer Geometry Lattice" in lattice_md.read_text(
        encoding="utf-8"
    )

    rebound = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_decoder_q_pairset_acquisition.py"),
            "--selector-pareto",
            str(tmp_path / "selector_pareto.json"),
            "--prefix-ks",
            "8",
            "--diversity-ks",
            "8",
            "--max-drop-two",
            "0",
            "--max-swap-in",
            "0",
            "--pair-frame-geometry-lattice-json",
            str(lattice_path),
            "--json-out",
            str(rebound_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert rebound.returncode != 0

    (tmp_path / "selector_pareto.json").write_text(
        json.dumps(_selector_pareto()), encoding="utf-8"
    )
    rebound = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_decoder_q_pairset_acquisition.py"),
            "--selector-pareto",
            str(tmp_path / "selector_pareto.json"),
            "--prefix-ks",
            "8",
            "--diversity-ks",
            "8",
            "--max-drop-two",
            "0",
            "--max-swap-in",
            "0",
            "--pair-frame-geometry-lattice-json",
            str(lattice_path),
            "--json-out",
            str(rebound_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert rebound.returncode == 0, rebound.stderr
    rebound_stdout = json.loads(rebound.stdout)
    assert rebound_stdout["pair_frame_geometry_candidate_count"] == 2
    assert rebound_stdout["score_claim"] is False

