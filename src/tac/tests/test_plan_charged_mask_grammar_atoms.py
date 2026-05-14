# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_charged_mask_grammar_atoms.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_charged_mask_grammar_atoms_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stored_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def _rpk1_payload(mask_stream: bytes) -> bytes:
    logical_members = {
        "renderer.bin": b"QZS3synthetic-renderer",
        "masks.mkv": mask_stream,
        "optimized_poses.bin": b"synthetic-poses",
    }
    header = {
        "schema": "renderer_payload_v1",
        "members": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": _sha(data),
                "codec": "raw",
            }
            for name, data in logical_members.items()
        ],
    }
    header_json = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = b"".join(logical_members.values())
    return b"RPK1" + struct.pack("<I", len(header_json)) + header_json + body


def _source_archive(tmp_path: Path, *, mask_stream: bytes = b"\x12\x00mask-stream") -> Path:
    archive = tmp_path / "source.zip"
    _stored_zip(archive, {"p": _rpk1_payload(mask_stream)})
    return archive


def test_extracts_mask_stream_and_writes_non_score_manifest(tmp_path: Path) -> None:
    planner = _load_planner()
    mask_stream = b"\x12\x00synthetic-mask-stream" * 9
    archive = _source_archive(tmp_path, mask_stream=mask_stream)

    manifest = planner.build_plan(
        source_archive=archive,
        output_dir=tmp_path / "plan",
        max_frame_groups=12,
        max_stream_chunks=3,
        max_component_atoms=8,
        max_rle_atoms=8,
    )

    assert manifest["schema"] == planner.SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["gpu_required"] is False
    assert manifest["cuda_jobs_launched"] is False
    assert manifest["input_archive"]["sha256"] == planner._sha256_file(archive)
    assert manifest["single_blob_member"]["name"] == "p"
    assert manifest["renderer_payload_extraction"]["helper"] == "submissions/robust_current/unpack_renderer_payload.py"
    assert manifest["renderer_payload_extraction"]["payload_schema"] == "renderer_payload_v1"
    assert manifest["extracted_mask_stream"]["archive_member_name"] == "masks.mkv"
    assert manifest["extracted_mask_stream"]["bytes"] == len(mask_stream)
    assert manifest["extracted_mask_stream"]["sha256"] == _sha(mask_stream)
    assert (tmp_path / "plan" / "extracted_mask_stream.bin").read_bytes() == mask_stream
    assert (tmp_path / "plan" / "atom_plan_manifest.json").exists()
    assert manifest["atom_tables"]["frame_groups"]
    assert manifest["atom_tables"]["ego_foveal_regions"]
    assert manifest["atom_tables"]["mask_stream_chunks"]
    assert manifest["dense_mask_analysis"] is None
    assert manifest["mask_decode_contract"]["status"] == "not_requested"
    assert any(policy["policy_name"] == "cmg_foveal_horizon_boundary_first" for policy in manifest["candidate_policies"])


def test_dense_mask_array_enables_deterministic_components_and_rle(tmp_path: Path) -> None:
    planner = _load_planner()
    archive = _source_archive(tmp_path)
    masks = np.zeros((4, 6, 8), dtype=np.uint8)
    masks[0, 1:5, 2:6] = 1
    masks[1, :, :] = 1
    masks[2, 1:3, 1:4] = 2
    masks[2, 4, :] = 3
    masks[3, 0:2, 4:8] = 4
    mask_array = tmp_path / "decoded_masks.npy"
    np.save(mask_array, masks)

    kwargs = {
        "source_archive": archive,
        "mask_array": mask_array,
        "max_frame_groups": 16,
        "max_stream_chunks": 2,
        "max_dense_frames": 8,
        "max_component_atoms": 12,
        "max_rle_atoms": 12,
        "max_class_atoms": 12,
        "max_boundary_atoms": 12,
        "min_component_area": 2,
        "min_span_len": 3,
        "min_boundary_edges": 2,
    }
    first = planner.build_plan(output_dir=tmp_path / "plan_a", **kwargs)
    second = planner.build_plan(output_dir=tmp_path / "plan_b", **kwargs)

    assert first == second
    dense = first["dense_mask_analysis"]
    assert dense["mask_array"]["shape"] == {"frames": 4, "height": 6, "width": 8, "class_count": 5}
    assert dense["analysis_frames"] == [0, 1, 2, 3]
    assert dense["class_histogram_pixels"]["1"] > 0
    assert first["atom_tables"]["connected_components"]
    assert first["atom_tables"]["rle_spans"]
    assert first["atom_tables"]["class_regions"]
    assert first["atom_tables"]["class_boundaries"]
    assert first["atom_tables"]["connected_components"][0]["atom_family"] == "connected_component"
    assert first["atom_tables"]["rle_spans"][0]["atom_family"] == "scanline_span"
    assert first["atom_tables"]["class_regions"][0]["atom_family"] == "class_region"
    assert first["atom_tables"]["class_boundaries"][0]["atom_family"] == "class_boundary"
    names = {policy["policy_name"] for policy in first["candidate_policies"]}
    assert "cmg_large_component_templates" in names
    assert "cmg_scanline_span_residual_tiles" in names
    assert "cmg_class_region_support_maps" in names
    assert "cmg_boundary_pair_refinement" in names


def test_component_trace_prior_emits_charged_allocation_specs(tmp_path: Path) -> None:
    planner = _load_planner()
    archive = _source_archive(tmp_path, mask_stream=b"mask-stream-for-allocation" * 80)
    masks = np.zeros((6, 12, 16), dtype=np.uint8)
    masks[2, 3:8, 4:12] = 1
    masks[3, 3:8, 4:12] = 1
    masks[4, 6:, 2:14] = 2
    masks[5, 1:10, 7:9] = 3
    mask_array = tmp_path / "decoded_masks.npy"
    np.save(mask_array, masks)
    trace = {
        "schema_version": "contest_component_trace_v1",
        "score_claim": False,
        "evidence_grade": "diagnostic_cuda_component_trace",
        "n_samples": 3,
        "archive_size_bytes": 276223,
        "samples": [
            {
                "pair_index": 0,
                "frame_indices": [0, 1],
                "posenet_dist": 0.0001,
                "segnet_dist": 0.0002,
                "score_combined_contribution_first_order": 0.0001,
            },
            {
                "pair_index": 1,
                "frame_indices": [2, 3],
                "posenet_dist": 0.0025,
                "segnet_dist": 0.0012,
                "score_combined_contribution_first_order": 0.0009,
            },
            {
                "pair_index": 2,
                "frame_indices": [4, 5],
                "posenet_dist": 0.001,
                "segnet_dist": 0.001,
                "score_combined_contribution_first_order": 0.0004,
            },
        ],
    }
    trace_path = tmp_path / "component_trace.json"
    trace_path.write_text(json.dumps(trace, sort_keys=True) + "\n")

    kwargs = {
        "source_archive": archive,
        "mask_array": mask_array,
        "component_trace_json": trace_path,
        "trace_top_pairs": 2,
        "allocation_byte_budgets": (256, 512),
        "max_allocation_rows": 24,
        "max_dense_frames": 6,
        "max_component_atoms": 16,
        "max_rle_atoms": 16,
        "max_class_atoms": 16,
        "max_boundary_atoms": 16,
        "min_component_area": 3,
        "min_span_len": 2,
        "min_boundary_edges": 2,
        "anchor_archive_bytes": 276223,
    }
    first = planner.build_plan(output_dir=tmp_path / "plan_a", **kwargs)
    second = planner.build_plan(output_dir=tmp_path / "plan_b", **kwargs)

    assert first == second
    prior = first["component_trace_prior"]
    assert prior["path"] == str(trace_path.resolve())
    assert prior["sha256"] == planner._sha256_file(trace_path)
    assert prior["score_claim"] is False
    assert [row["pair_index"] for row in prior["top_pairs"]][:1] == [1]

    allocation = first["trace_weighted_allocation"]
    assert allocation["schema"] == planner.ALLOCATION_SCHEMA
    assert allocation["score_claim"] is False
    assert allocation["promotion_eligible"] is False
    assert allocation["all_score_affecting_payloads_charged"] is True
    assert allocation["external_sidecars_allowed"] is False
    rows = allocation["allocation_table"]
    assert rows
    assert rows[0]["payload_charged"] is True
    assert rows[0]["evidence_grade"] == "empirical_allocation_only"
    assert 1 in rows[0]["trace_top_pair_hits"]

    specs = allocation["candidate_specs"]
    assert [spec["budget_bytes"] for spec in specs] == [512, 256]
    assert all(spec["score_claim"] is False for spec in specs)
    assert all(spec["build_contract"]["all_score_affecting_payloads_charged"] is True for spec in specs)
    assert all(spec["build_contract"]["external_sidecars_allowed"] is False for spec in specs)
    assert specs[0]["charged_side_info_accounting"]["total_side_info_bytes"] > 0
    assert specs[0]["archive_bytes_if_replaces_mask_estimate"] < 276223
    assert "naive CRF/RPK1" in specs[0]["why_higher_ev_than_naive_crf_or_rpk1"]


def test_decode_mask_array_uses_runtime_helper_and_feeds_dense_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _load_planner()
    mask_stream = b"\x12\x00synthetic-mask-stream" * 3
    archive = _source_archive(tmp_path, mask_stream=mask_stream)
    decoded = np.zeros((3, 5, 7), dtype=np.uint8)
    decoded[0, 1:4, 2:6] = 1
    decoded[1, :, 3:] = 2
    decoded[2, 2:, :] = 3
    calls = []

    class Runtime:
        @staticmethod
        def _load_masks_from_archive(path: Path, *, expected_frames: int):
            calls.append((path.name, path.suffix, expected_frames, path.read_bytes()))
            return decoded

    monkeypatch.setattr(planner, "_load_inflate_renderer_module", lambda: Runtime)

    manifest = planner.build_plan(
        source_archive=archive,
        output_dir=tmp_path / "plan",
        decode_mask_array=True,
        decode_expected_frames=1200,
        max_dense_frames=8,
        max_component_atoms=8,
        max_rle_atoms=8,
        max_class_atoms=8,
        max_boundary_atoms=8,
        min_component_area=2,
        min_span_len=2,
        min_boundary_edges=1,
    )

    assert calls == [(calls[0][0], ".mkv", 1200, mask_stream)]
    assert calls[0][0].startswith("cmg_plan_extracted_mask_stream_")
    decode_contract = manifest["mask_decode_contract"]
    assert decode_contract["status"] == "decoded"
    assert decode_contract["helper"] == "submissions/robust_current/inflate_renderer.py"
    assert decode_contract["output_array"]["shape"] == [3, 5, 7]
    decoded_path = tmp_path / "plan" / "decoded_mask_array.npy"
    assert decoded_path.exists()
    np.testing.assert_array_equal(np.load(decoded_path, allow_pickle=False), decoded)
    assert manifest["dense_mask_analysis"]["mask_array"]["path"] == str(decoded_path.resolve())
    assert manifest["atom_tables"]["connected_components"]
    assert manifest["atom_tables"]["class_boundaries"]


def test_decode_mask_array_rejects_out_of_range_runtime_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_planner()
    archive = _source_archive(tmp_path, mask_stream=b"mask-stream")

    class Runtime:
        @staticmethod
        def _load_masks_from_archive(path: Path, *, expected_frames: int):
            return np.array([[[0, 5]]], dtype=np.uint8)

    monkeypatch.setattr(planner, "_load_inflate_renderer_module", lambda: Runtime)

    with pytest.raises(planner.PlannerError, match="runtime mask decode failed; blocker_class=runtime_decode_contract"):
        planner.build_plan(
            source_archive=archive,
            output_dir=tmp_path / "plan",
            decode_mask_array=True,
        )


def test_rejects_multi_member_one_blob_inputs(tmp_path: Path) -> None:
    planner = _load_planner()
    archive = tmp_path / "bad.zip"
    _stored_zip(archive, {"p": _rpk1_payload(b"mask"), "debug.txt": b"sidecar"})

    with pytest.raises(planner.PlannerError, match="expected one non-directory archive member"):
        planner.build_plan(source_archive=archive, output_dir=tmp_path / "plan")
