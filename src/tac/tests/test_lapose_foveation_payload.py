from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.analysis.lapose_foveation_atoms import build_foveation_transport_atom_manifest
from tac.analysis.lapose_foveation_payload import (
    LaposeFoveationPayloadError,
    build_lapose_foveation_tuple_payload_artifact,
    decode_lapose_foveation_tuple_payload,
    pack_lapose_foveation_tuple_payload,
)
from tac.repo_io import sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_lapose_foveation_tuple_payload_records_bytes_sha_and_blockers(tmp_path: Path) -> None:
    manifest = _manifest()
    payload_path = tmp_path / "lapose_foveation_tuples.lfv1"

    readiness = build_lapose_foveation_tuple_payload_artifact(
        manifest,
        payload_path=payload_path,
        repo_root=tmp_path,
        max_atoms=2,
    )

    assert payload_path.is_file()
    assert readiness["schema"] == "lapose_foveation_tuple_payload_readiness_v1"
    assert readiness["score_claim"] is False
    assert readiness["dispatch_attempted"] is False
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["promotion_eligible"] is False
    assert readiness["wire_format"] == "LFV1"
    assert readiness["bytes"] == payload_path.stat().st_size
    assert readiness["sha256"] == sha256_file(payload_path)
    assert readiness["payload"]["row_count"] == 2
    assert readiness["payload"]["tuple_row_bytes"] == 13
    assert readiness["payload"]["tuple_body_bytes"] == 26
    assert readiness["payload"]["total_bytes"] == 38
    assert readiness["byte_closure"] == {
        "local_payload_bytes_measured": True,
        "local_payload_sha256_measured": True,
        "archive_member_proven": False,
        "archive_consumed_by_runtime": False,
        "noop_controls_ran": False,
        "exact_cuda_auth_eval_ran": False,
    }
    assert {
        "no_runtime_consumer",
        "no_noop_controls",
        "no_exact_cuda_eval",
        "not_archive_consumed_payload",
    }.issubset(set(readiness["dispatch_blockers"]))
    assert readiness["runtime_contract"]["scorer_loads_at_pack_time"] is False
    decoded = decode_lapose_foveation_tuple_payload(payload_path.read_bytes())
    assert decoded == readiness["decoded_tuple_preview"]
    assert decoded["magic"] == "LFV1"
    assert decoded["row_count"] == 2
    assert [row["pair_index"] for row in decoded["rows"]] == sorted(
        atom["pair_index"] for atom in readiness["selected_atoms"]
    )


def test_lapose_foveation_tuple_payload_is_deterministic_for_reordered_source_rows(
    tmp_path: Path,
) -> None:
    forward = _manifest()
    reverse = build_foveation_transport_atom_manifest(
        list(reversed(_records())),
        base_pose_dist=0.02,
        source="fixture",
        frame_width=320,
        frame_height=200,
        foveal_center=(160.0, 90.0),
        center_gain=(12.0, 8.0),
    )

    forward_payload, forward_pack = pack_lapose_foveation_tuple_payload(forward, max_atoms=3)
    reverse_payload, reverse_pack = pack_lapose_foveation_tuple_payload(reverse, max_atoms=3)

    assert forward_payload == reverse_payload
    assert forward_pack["selected_atoms"] == reverse_pack["selected_atoms"]


def test_lapose_foveation_tuple_payload_fails_closed_on_bad_rows() -> None:
    manifest = _manifest()
    bad = dict(manifest)
    bad["atoms"] = [dict(manifest["atoms"][0], score_claim=True)]
    with pytest.raises(LaposeFoveationPayloadError, match="score_claim"):
        pack_lapose_foveation_tuple_payload(bad)

    bad = dict(manifest)
    bad["atoms"] = [dict(manifest["atoms"][0]), dict(manifest["atoms"][0], atom_id="dup")]
    with pytest.raises(LaposeFoveationPayloadError, match="duplicate foveation pair_index"):
        pack_lapose_foveation_tuple_payload(bad)

    with pytest.raises(LaposeFoveationPayloadError, match="selected_atom_ids not found"):
        pack_lapose_foveation_tuple_payload(manifest, selected_atom_ids=["missing"])


def test_build_lapose_foveation_tuple_payload_cli(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload_path = tmp_path / "payload.lfv1"
    out = tmp_path / "readiness.json"
    manifest_path.write_text(json.dumps(_manifest(), sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_lapose_foveation_tuple_payload.py"),
            "--manifest-json",
            str(manifest_path),
            "--payload-out",
            str(payload_path),
            "--max-atoms",
            "2",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    readiness = json.loads(out.read_text(encoding="utf-8"))
    assert readiness["path"] == payload_path.as_posix()
    assert readiness["bytes"] == payload_path.stat().st_size
    assert readiness["sha256"] == sha256_file(payload_path)
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["source_manifest"]["sha256"] == sha256_file(manifest_path)
    assert readiness["tool_run_manifest"]["tool"] == "tools/build_lapose_foveation_tuple_payload.py"
    assert readiness["tool_run_manifest"]["input_files"] == [
        {
            "path": manifest_path.as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    ]


def _manifest() -> dict:
    return build_foveation_transport_atom_manifest(
        _records(),
        base_pose_dist=0.02,
        source="fixture",
        frame_width=320,
        frame_height=200,
        foveal_center=(160.0, 90.0),
        center_gain=(12.0, 8.0),
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
