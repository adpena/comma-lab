# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "hoist_pose_bytes_from_master_gradient.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("hoist_pose_bytes_from_master_gradient", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_anchor(
    tmp_path: Path,
    *,
    with_score_axis_dominance: bool = False,
    with_scored_custody: bool = True,
) -> tuple[str, Path]:
    archive = "c" * 64
    arr = np.array(
        [
            [10.0, 0.01, 0.0],
            [0.01, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.01, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    npy_path = tmp_path / "aggregate.npy"
    np.save(npy_path, arr)
    anchor = {
        "archive_sha256": archive,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "gradient_byte_domain": "zip_inner_member_payload" if with_scored_custody else "scored_archive_bytes",
        "measurement_axis": "[diagnostic-CPU]",
        "measurement_hardware": "linux_x86_64_cpu_diagnostic",
        "measurement_method": "aggregate_projection",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 4,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    if with_scored_custody:
        anchor.update(
            {
                "scored_archive_sha256": archive,
                "scored_archive_bytes": 12345,
            }
        )
    if with_score_axis_dominance:
        anchor["score_axis_dominance"] = {
            "formula": "fixture",
            "selected_count": 1,
        }
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    ledger.write_text(json.dumps(anchor) + "\n", encoding="utf-8")
    return archive, ledger


def test_build_pose_byte_hoist_manifest_is_planning_only(tmp_path: Path):
    tool = _load_tool()
    archive, ledger = _write_anchor(tmp_path)
    manifest = tool.build_pose_byte_hoist_manifest(
        archive_sha256=archive,
        top_k=2,
        axis_dominance_threshold=0.7,
        anchor_path=ledger,
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == tool.SCHEMA_VERSION
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_operator_probe"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["selection"]["selected_count"] == 2
    assert manifest["selection"]["raw_archive_byte_coordinates_allowed"] is False
    assert manifest["source_anchor"]["score_axis_dominance_available"] is False
    assert manifest["source_anchor"]["score_axis_dominance_source"] == "derived_from_gradient_tensor_at_runtime"
    assert len(manifest["source_anchor"]["anchor_row_canonical_json_sha256"]) == 64
    assert len(manifest["source_anchor"]["anchor_ledger_sha256"]) == 64
    assert len(manifest["source_anchor"]["gradient_array_sha256"]) == 64
    expected_sidecar = tmp_path / "out" / "master_gradient_consumers" / f"pose_axis_dominant_bytes_{archive[:12]}_op7_manifest_v1.json"
    assert manifest["selector_sidecar_path"] == expected_sidecar.as_posix()
    assert len(manifest["selector_sidecar_sha256"]) == 64
    assert "anchor_score_axis_dominance_not_persisted" in manifest["blockers"]
    assert "grammar_aware_pose_axis_mutation_builder_missing" in manifest["blockers"]
    first = manifest["candidate_modification_specs"][0]
    assert first["axis_label"] == "pose"
    assert first["score_claim"] is False
    assert first["raw_archive_byte_coordinates_allowed"] is False
    assert Path(tmp_path / "out" / "master_gradient_consumers").exists()


def test_missing_scored_archive_custody_never_falls_back_to_archive_sha(tmp_path: Path):
    tool = _load_tool()
    archive, ledger = _write_anchor(tmp_path, with_scored_custody=False)
    manifest = tool.build_pose_byte_hoist_manifest(
        archive_sha256=archive,
        top_k=1,
        axis_dominance_threshold=0.7,
        anchor_path=ledger,
        output_dir=tmp_path / "out",
    )

    first = manifest["candidate_modification_specs"][0]
    assert manifest["source_anchor"]["scored_archive_custody_available"] is False
    assert manifest["authority_boundary"]["source_archive_custody_available"] is False
    assert first["source_archive_sha256"] is None
    assert first["source_archive_bytes"] is None
    assert first["section_name"] == "diagnostic_uncustodied_gradient_subject_bytes"
    assert "scored_archive_custody_missing" in first["blockers"]
    assert "scored_archive_custody_missing" in manifest["blockers"]


def test_build_pose_byte_hoist_manifest_records_persisted_dominance(tmp_path: Path):
    tool = _load_tool()
    archive, ledger = _write_anchor(tmp_path, with_score_axis_dominance=True)
    manifest = tool.build_pose_byte_hoist_manifest(
        archive_sha256=archive,
        top_k=1,
        anchor_path=ledger,
        output_dir=tmp_path / "out",
    )

    assert manifest["source_anchor"]["score_axis_dominance_available"] is True
    assert manifest["source_anchor"]["score_axis_dominance_source"] == "anchor_field"
    assert "anchor_score_axis_dominance_not_persisted" not in manifest["blockers"]


def test_build_pose_byte_hoist_manifest_resolves_layout_candidates(tmp_path: Path):
    tool = _load_tool()
    archive, ledger = _write_anchor(tmp_path)
    layout = {
        "schema": "tac_frontier_archive_layout_v1",
        "archive_path": str(tmp_path / "archive.zip"),
        "archive_sha256": archive,
        "archive_bytes": 12345,
        "logical_layout": {
            "grammar": "fixture_pose_axis_layout",
            "sections": [
                {
                    "name": "header_u32le",
                    "role": "internal_length_header",
                    "offset": 0,
                    "len": 1,
                    "sha256": "1" * 64,
                },
                {
                    "name": "decoder_blob",
                    "role": "renderer_decoder_weights",
                    "offset": 1,
                    "len": 2,
                    "sha256": "2" * 64,
                },
                {
                    "name": "latent_blob",
                    "role": "latent_motion_or_frame_conditioning",
                    "offset": 3,
                    "len": 1,
                    "sha256": "3" * 64,
                },
            ],
        },
    }
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    manifest = tool.build_pose_byte_hoist_manifest(
        archive_sha256=archive,
        top_k=1,
        axis_dominance_threshold=0.7,
        anchor_path=ledger,
        layout_manifest_path=layout_path,
        output_dir=tmp_path / "out",
    )

    assert manifest["grammar_aware_layout_manifest_path"] == layout_path.as_posix()
    assert len(manifest["grammar_aware_layout_manifest_sha256"]) == 64
    resolution = manifest["grammar_aware_operator_candidate_resolution"]
    assert resolution["schema"] == "tac_pose_axis_master_gradient_operator_candidates_v1"
    assert resolution["resolved_count"] == 1
    assert resolution["resolved_pose_axis_candidates"][0]["section_name"] == "decoder_blob"
    assert resolution["candidate_modification_specs"][0]["score_claim"] is False
    assert resolution["candidate_modification_specs"][0]["raw_archive_byte_coordinates_allowed"] is False
    assert manifest["smoke"]["status"] == "blocked_missing_packet_proofs"
    assert "packet_proofs_missing" in manifest["blockers"]
    assert "grammar_aware_pose_axis_mutation_builder_missing" not in manifest["blockers"]


def test_cli_writes_manifest_json(tmp_path: Path):
    tool = _load_tool()
    archive, ledger = _write_anchor(tmp_path)
    out_dir = tmp_path / "out"
    manifest_path = out_dir / "manifest.json"

    rc = tool.main([
        "--archive-sha256",
        archive,
        "--anchor-path",
        str(ledger),
        "--output-dir",
        str(out_dir),
        "--manifest-path",
        str(manifest_path),
        "--top-k",
        "2",
    ])

    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["manifest_path"] == manifest_path.as_posix()
    assert payload["selector_sidecar_path"] == (
        out_dir / "master_gradient_consumers" / f"pose_axis_dominant_bytes_{archive[:12]}_op7_manifest_v1.json"
    ).as_posix()
    assert len(payload["selector_sidecar_sha256"]) == 64
    assert payload["selection"]["selected_count"] == 2
