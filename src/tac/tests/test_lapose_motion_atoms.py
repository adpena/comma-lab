# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.analysis.lapose_motion_atoms import (
    LaposeMotionAtomError,
    build_motion_atom_manifest,
    records_from_json_payload,
)

REPO = Path(__file__).resolve().parents[3]


def test_build_motion_atom_manifest_ranks_hard_pair_atoms() -> None:
    manifest = build_motion_atom_manifest(
        _records(),
        base_pose_dist=0.01,
        source="fixture",
        target_average_degree=2.0,
    )

    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["graph"]["node_count"] == 4
    assert manifest["graph"]["edge_count"] >= 3
    assert manifest["atom_ledger"]["rows"][0]["atom_id"] == "lapose_motion_pair:75"
    assert manifest["atom_ledger"]["rows"][0]["pair_support"] == [75]
    assert manifest["atom_ledger"]["rows"][0]["hard_pair_support"] == [75]
    assert manifest["atom_ledger"]["rows"][0]["class_support"] == [2, 3]
    assert "ego_motion" in manifest["atom_ledger"]["rows"][0]["openpilot_priors"]


def test_motion_atom_manifest_fails_closed_on_bad_records() -> None:
    with pytest.raises(LaposeMotionAtomError, match="latent_action"):
        build_motion_atom_manifest(
            [{"pair_index": 1, "latent_action": []}],
            base_pose_dist=0.01,
            source="bad",
        )
    signed_manifest = build_motion_atom_manifest(
        [{"pair_index": 1, "latent_action": [0.0], "byte_delta": -1}],
        base_pose_dist=0.01,
        source="signed",
    )
    assert signed_manifest["atoms"][0]["byte_delta"] == -1
    with pytest.raises(LaposeMotionAtomError, match="non-negative"):
        build_motion_atom_manifest(
            [{"pair_index": 1, "latent_action": [0.0], "estimated_charged_bytes": -1}],
            base_pose_dist=0.01,
            source="bad",
        )


def test_motion_atom_manifest_truncates_after_ranking() -> None:
    records = _records()
    records.append(
        {
            "pair_index": 599,
            "latent_action": [10.0, 0.0, 0.0],
            "byte_delta": 1,
            "expected_seg_dist_delta": -0.01,
            "expected_pose_dist_delta": -0.00001,
            "confidence": 1.0,
            "pair_support": [599],
        }
    )

    manifest = build_motion_atom_manifest(
        records,
        base_pose_dist=0.01,
        source="fixture",
        max_atoms=1,
    )

    assert manifest["source_atom_count"] == 5
    assert manifest["atoms"][0]["atom_id"] == "lapose_motion_pair:599"
    assert manifest["atom_ledger"]["rows"][0]["atom_id"] == "lapose_motion_pair:599"
    assert manifest["atom_ledger"]["truncation"]["dropped_atom_count"] == 4


def test_motion_atom_manifest_is_order_invariant() -> None:
    forward = build_motion_atom_manifest(
        _records(),
        base_pose_dist=0.01,
        source="fixture",
        target_average_degree=2.0,
    )
    reversed_order = build_motion_atom_manifest(
        list(reversed(_records())),
        base_pose_dist=0.01,
        source="fixture",
        target_average_degree=2.0,
    )

    assert reversed_order["record_sha256"] == forward["record_sha256"]
    assert reversed_order["graph"]["edges"] == forward["graph"]["edges"]
    assert reversed_order["atoms"] == forward["atoms"]
    assert reversed_order["atom_ledger"]["rows"] == forward["atom_ledger"]["rows"]


def test_motion_atom_manifest_rejects_duplicate_pair_indices() -> None:
    records = _records()
    records.append({**records[0], "latent_action": [9.0, 9.0, 9.0]})

    with pytest.raises(LaposeMotionAtomError, match="duplicate pair_index values: 10"):
        build_motion_atom_manifest(
            records,
            base_pose_dist=0.01,
            source="duplicate",
        )


def test_records_from_json_payload_accepts_list_or_records_object() -> None:
    records = _records()
    assert records_from_json_payload(records) == records
    assert records_from_json_payload({"records": records}) == records
    with pytest.raises(LaposeMotionAtomError):
        records_from_json_payload({"wrong": records})


def test_build_lapose_motion_atom_manifest_cli(tmp_path: Path) -> None:
    records = tmp_path / "records.json"
    out = tmp_path / "manifest.json"
    records.write_text(json.dumps({"records": _records()}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_lapose_motion_atom_manifest.py"),
            "--records-json",
            str(records),
            "--base-pose-dist",
            "0.01",
            "--source",
            "fixture",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["record_count"] == 4
    assert payload["atom_ledger"]["score_claim"] is False
    assert payload["atom_ledger"]["ready_for_exact_eval_dispatch"] is False


def _records() -> list[dict]:
    return [
        {
            "pair_index": 10,
            "latent_action": [0.0, 0.0, 0.0],
            "byte_delta": 40,
            "expected_seg_dist_delta": -0.00001,
            "expected_pose_dist_delta": -0.000001,
            "confidence": 0.5,
            "class_support": [1],
        },
        {
            "pair_index": 75,
            "latent_action": [1.0, 0.0, 0.0],
            "byte_delta": 50,
            "expected_seg_dist_delta": -0.0002,
            "expected_pose_dist_delta": -0.00001,
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
            "latent_action": [1.0, 1.0, 0.0],
            "byte_delta": 60,
            "expected_seg_dist_delta": -0.00005,
            "expected_pose_dist_delta": -0.00002,
            "confidence": 0.7,
            "class_support": [3],
            "openpilot_priors": ["yaw_rate"],
        },
        {
            "pair_index": 300,
            "latent_action": [0.0, 1.0, 0.0],
            "byte_delta": 30,
            "expected_seg_dist_delta": 0.0,
            "expected_pose_dist_delta": -0.000001,
            "confidence": 0.4,
            "class_support": [0],
        },
    ]
