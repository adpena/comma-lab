from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.analysis.lapose_foveation_atoms import build_foveation_transport_atom_manifest
from tac.analysis.lapose_motion_atoms import LaposeMotionAtomError
from tac.optimization.field_equation_planner import build_field_equation_plan

REPO = Path(__file__).resolve().parents[3]


def test_foveation_transport_atoms_are_byte_bearing_and_dispatch_blocked() -> None:
    manifest = build_foveation_transport_atom_manifest(
        _records(),
        base_pose_dist=0.02,
        source="fixture",
        frame_width=320,
        frame_height=200,
        foveal_center=(160.0, 90.0),
        center_gain=(12.0, 8.0),
    )

    assert manifest["schema"] == "lapose_foveation_transport_atom_manifest_v1"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["charged_byte_model"]["estimated_bytes_per_pair_atom"] == 13
    assert manifest["charged_byte_model"]["exact_archive_bytes_required"] is True
    assert "foveated_diffusion_2026" in {
        source["basis_id"] for source in manifest["research_basis"]["sources"]
    }
    atom = manifest["atoms"][0]
    assert atom["byte_delta"] == 13
    assert atom["byte_delta_is_estimate"] is True
    assert atom["ready_for_exact_eval_dispatch"] is False
    assert atom["foveation_parameters"]["schema"] == "lapose_guided_hyperbolic_foveation_tuple_v1"
    assert 0.0 <= atom["foveation_parameters"]["origin_x"] <= 319.0
    assert 0.0 <= atom["foveation_parameters"]["origin_y"] <= 199.0
    assert "requires_runtime_consumption_proof" in atom["dispatch_blockers"]
    row = manifest["atom_ledger"]["rows"][0]
    assert row["byte_delta"] == 13
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["proxy_row"] is True
    assert "proxy_row_not_dispatchable" in row["dispatch_blockers"]
    assert "requires_byte_closed_archive" in row["dispatch_blockers"]


def test_foveation_transport_atoms_feed_field_planning_without_dispatch() -> None:
    manifest = build_foveation_transport_atom_manifest(
        _records(),
        base_pose_dist=0.02,
        source="fixture",
    )
    plan = build_field_equation_plan(
        manifest["atom_ledger"],
        source="fixture",
        research_basis_ids=["telescope_2026"],
    )

    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    row = plan["rows"][0]
    assert row["family_group"] == "pose_foveation"
    assert "lapose_2026" in row["research_basis_ids"]
    assert "telescope_2026" in row["research_basis_ids"]
    assert "foveated_diffusion_2026" in row["research_basis_ids"]
    assert "geometric_visual_servo_ot_2026" in row["research_basis_ids"]
    assert row["frechet_derivatives"]["d_bytes_d_epsilon"] == 13
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "missing_byte_closed_archive_manifest" in row["dispatch_blockers"]
    assert "proxy_evidence_not_kkt_ready" in row["dispatch_blockers"]


def test_foveation_transport_manifest_is_order_invariant_and_truncates() -> None:
    forward = build_foveation_transport_atom_manifest(
        _records(),
        base_pose_dist=0.02,
        source="fixture",
        max_atoms=1,
    )
    reverse = build_foveation_transport_atom_manifest(
        list(reversed(_records())),
        base_pose_dist=0.02,
        source="fixture",
        max_atoms=1,
    )

    assert reverse["record_sha256"] == forward["record_sha256"]
    assert reverse["atoms"] == forward["atoms"]
    assert reverse["atom_ledger"]["rows"] == forward["atom_ledger"]["rows"]
    assert forward["source_atom_count"] == 3
    assert forward["atom_count"] == 1
    assert forward["atom_ledger"]["truncation"]["dropped_atom_count"] == 2


def test_foveation_transport_manifest_fails_closed_on_bad_inputs() -> None:
    with pytest.raises(LaposeMotionAtomError, match="latent_action"):
        build_foveation_transport_atom_manifest(
            [{"pair_index": 1, "latent_action": []}],
            base_pose_dist=0.02,
            source="bad",
        )
    with pytest.raises(LaposeMotionAtomError, match="inside frame"):
        build_foveation_transport_atom_manifest(
            _records(),
            base_pose_dist=0.02,
            source="bad_center",
            foveal_center=(999.0, 10.0),
        )
    duplicated = [*_records(), dict(_records()[0])]
    with pytest.raises(LaposeMotionAtomError, match="duplicate pair_index values"):
        build_foveation_transport_atom_manifest(
            duplicated,
            base_pose_dist=0.02,
            source="dup",
        )


def test_build_lapose_foveation_atom_manifest_cli(tmp_path: Path) -> None:
    records = tmp_path / "records.json"
    out = tmp_path / "manifest.json"
    records.write_text(json.dumps({"records": _records()}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_lapose_foveation_atom_manifest.py"),
            "--records-json",
            str(records),
            "--base-pose-dist",
            "0.02",
            "--source",
            "fixture",
            "--max-atoms",
            "2",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["atom_count"] == 2
    assert payload["atom_ledger"]["score_claim"] is False
    assert payload["atom_ledger"]["ready_for_exact_eval_dispatch"] is False
    assert (
        payload["tool_run_manifest"]["tool"]
        == "tools/build_lapose_foveation_atom_manifest.py"
    )


def _records() -> list[dict]:
    return [
        {
            "pair_index": 10,
            "latent_action": [-0.3, 0.0, 1.0, 0.1, -0.2, 0.3, 0.02, -0.03],
            "expected_seg_dist_delta": -0.00001,
            "expected_pose_dist_delta": -0.00002,
            "confidence": 0.6,
            "class_support": [1],
            "pair_support": [10],
            "geometry_priors": ["lane_boundary"],
        },
        {
            "pair_index": 75,
            "hard_pair_rank": 0,
            "latent_action": [0.0, 1.0, 0.0, 1.2, 0.8, 1.5, 0.4, 0.2],
            "expected_seg_dist_delta": -0.0002,
            "expected_pose_dist_delta": -0.00005,
            "confidence": 0.8,
            "hard_pair_score": 4.2,
            "pair_support": [75],
            "hard_pair_support": [75],
            "class_support": [2, 3],
            "geometry_priors": ["lane_boundary"],
            "openpilot_priors": ["ego_motion"],
        },
        {
            "pair_index": 127,
            "latent_action": [0.4, 0.5, -0.5, -0.3, 0.9, 0.5, -0.2, 0.4],
            "expected_seg_dist_delta": -0.00005,
            "expected_pose_dist_delta": -0.00001,
            "confidence": 0.7,
            "class_support": [3],
            "openpilot_priors": ["yaw_rate"],
            "evidence_grade": "planning_lapose_foveation_transport",
        },
    ]
