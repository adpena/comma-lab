from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.analysis.lapose_motion_atoms import LaposeMotionAtomError, build_motion_atom_manifest
from tac.analysis.lapose_motion_evidence import records_from_component_response

REPO = Path(__file__).resolve().parents[3]


def test_records_from_component_response_allocates_best_cuda_delta() -> None:
    payload = records_from_component_response(
        _component_response(),
        latent_actions=_latent_actions(),
        pair_opportunities=_pair_opportunities(),
        evidence_source_path="component_response.json",
        evidence_source_sha256="a" * 64,
    )

    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["source_archive_sha256"] == "b" * 64
    assert (
        payload["paper_reference"]["implementation_alignment"]
        == "inspired_planning_only_not_paper_faithful_model"
    )
    assert "lapose_lite_is_not_paper_faithful_lapose_model" in payload["dispatch_blockers"]
    assert payload["allocation"]["response_atom"]["epsilon"] == -1.0
    assert payload["allocation"]["allocation_inference"] is True
    assert payload["records"][0]["evidence_grade"] == "diagnostic_cuda_global_response_allocated"
    assert payload["records"][0]["allocation_inference"] is True
    assert payload["records"][0]["pair_support"] == [10]
    assert payload["records"][0]["hard_pair_support"] == [10]
    assert payload["records"][0]["source_archive_sha256"] == "b" * 64
    assert payload["records"][0]["byte_delta"] == -1

    manifest = build_motion_atom_manifest(
        payload["records"],
        base_pose_dist=0.01,
        source="fixture",
    )
    assert manifest["atom_ledger"]["rows"][0]["evidence_source_sha256"] == "a" * 64
    assert manifest["atom_ledger"]["rows"][0]["rankable"] is False
    assert "allocated_global_response_not_rankable" in manifest["atom_ledger"]["rows"][0][
        "dispatch_blockers"
    ]


def test_records_from_component_response_fails_closed_on_non_cuda_or_missing_latents() -> None:
    bad = dict(_component_response())
    bad["device"] = "cpu"
    with pytest.raises(LaposeMotionAtomError, match="CUDA"):
        records_from_component_response(
            bad,
            latent_actions=_latent_actions(),
            pair_opportunities=_pair_opportunities(),
            evidence_source_path="component_response.json",
        )

    with pytest.raises(LaposeMotionAtomError, match="missing latent_action"):
        records_from_component_response(
            _component_response(),
            latent_actions=[],
            pair_opportunities=_pair_opportunities(),
            evidence_source_path="component_response.json",
        )


def test_records_from_component_response_rejects_no_improving_point() -> None:
    component = _component_response()
    component["points"][1]["values"]["combined"] = 1.1

    with pytest.raises(LaposeMotionAtomError, match="no improving point"):
        records_from_component_response(
            component,
            latent_actions=_latent_actions(),
            pair_opportunities=_pair_opportunities(),
            evidence_source_path="component_response.json",
        )


def test_build_lapose_motion_records_from_component_response_cli(tmp_path: Path) -> None:
    component = tmp_path / "component.json"
    latent = tmp_path / "latent.json"
    opportunities = tmp_path / "opportunities.json"
    out = tmp_path / "records.json"
    component.write_text(json.dumps(_component_response()), encoding="utf-8")
    latent.write_text(json.dumps({"latent_actions": _latent_actions()}), encoding="utf-8")
    opportunities.write_text(
        json.dumps({"pair_opportunities": _pair_opportunities()}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_lapose_motion_records_from_component_response.py"),
            "--component-response-json",
            str(component),
            "--latent-actions-json",
            str(latent),
            "--pair-opportunities-json",
            str(opportunities),
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["records"][1]["pair_index"] == 75
    assert payload["dispatch_blockers"]


def _component_response() -> dict:
    return {
        "schema_version": 1,
        "score_claim": False,
        "device": "cuda",
        "promotion_eligible": False,
        "baseline_archive": {
            "bytes": 1000,
            "sha256": "b" * 64,
        },
        "points": [
            {
                "epsilon": 0.0,
                "archive": {"bytes": 1000, "sha256": "b" * 64},
                "values": {"combined": 1.0, "segnet": 0.004, "posenet": 0.01},
            },
            {
                "epsilon": -1.0,
                "archive": {"bytes": 998, "sha256": "c" * 64},
                "values": {"combined": 0.9, "segnet": 0.003, "posenet": 0.009},
            },
        ],
    }


def _latent_actions() -> list[dict]:
    return [
        {"pair_index": 10, "latent_action": [0.0, 0.1, 0.2]},
        {"pair_index": 75, "latent_action": [1.0, 0.0, 0.1]},
    ]


def _pair_opportunities() -> list[dict]:
    return [
        {
            "pair_index": 10,
            "opportunity_mass": 1.0,
            "hard_pair_score": 1.0,
            "confidence": 0.5,
            "class_support": [2],
        },
        {
            "pair_index": 75,
            "opportunity_mass": 3.0,
            "hard_pair_score": 3.0,
            "hard_pair_support": [75],
            "confidence": 0.75,
            "class_support": [2, 3],
            "geometry_priors": ["lane_boundary"],
            "openpilot_priors": ["ego_motion"],
        },
    ]
