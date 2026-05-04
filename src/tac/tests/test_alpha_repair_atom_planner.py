from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "alpha_repair_atom_planner.py"
SPEC = importlib.util.spec_from_file_location("alpha_repair_atom_planner", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(planner)

ARCHIVE_BUILDER_PATH = REPO_ROOT / "experiments" / "build_alpha_mask_replacement_archive.py"
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_alpha_mask_replacement_archive_for_atom_planner_test",
    ARCHIVE_BUILDER_PATH,
)
assert BUILDER_SPEC is not None
builder = importlib.util.module_from_spec(BUILDER_SPEC)
assert BUILDER_SPEC.loader is not None
BUILDER_SPEC.loader.exec_module(builder)


def _write_candidate_manifest(path: Path, grayscale: Path, repair: Path) -> None:
    payload = {
        "schema": "alpha_mask_candidate_builder_v1",
        "candidate": {
            "candidate_archive_readiness": {
                "full_sequence_candidate": True,
                "ready_for_exact_eval_finalist_archive_assembly": True,
            },
            "alpha4": {
                "crf": 63,
                "agreement_before_repair": {"argmax_agreement": 0.99},
            },
            "artifacts": [
                {
                    "role": "alpha4_grayscale_lut_video",
                    "candidate_archive_member": "grayscale.mkv",
                    "path": str(grayscale),
                    "size_bytes": grayscale.stat().st_size,
                    "sha256": builder._sha256_file(grayscale),
                },
                {
                    "role": "alpha4_residual_repair_payload",
                    "candidate_archive_member": "alpha4_residual_repair.amr1",
                    "path": str(repair),
                    "size_bytes": repair.stat().st_size,
                    "sha256": builder._sha256_file(repair),
                },
            ],
        },
    }
    path.write_text(json.dumps(payload))


def _write_repair_payload(path: Path) -> None:
    alpha_builder = builder._load_alpha_builder_module()
    runs = [
        alpha_builder.RepairRun(frame_index=0, y=0, x0=0, length=2, class_id=2),
        alpha_builder.RepairRun(frame_index=1, y=0, x0=2, length=1, class_id=1),
        alpha_builder.RepairRun(frame_index=2, y=0, x0=0, length=3, class_id=3),
    ]
    payload = alpha_builder._encode_repair_payload(
        runs,
        shape=(3, 1, 4),
        source_mask_sha256="a" * 64,
        candidate_mask_sha256="b" * 64,
        selection_meta={
            "total_residual_pixels": 6,
            "selected_repair_pixels": 6,
            "partial_repair": False,
        },
    )
    path.write_bytes(payload)


def _write_pair_meta(path: Path) -> None:
    pose = [0.0] * 600
    seg = [0.0] * 600
    pose[0] = 0.4
    seg[0] = 0.01
    pose[1] = 0.1
    seg[1] = 0.005
    payload = {
        "schema_version": 1,
        "lane": "W",
        "mode": "topk",
        "n_pairs": 600,
        "hardest_pair_indices": [0],
        "per_pair_pose_dist": pose,
        "per_pair_seg_dist": seg,
    }
    path.write_text(json.dumps(payload))


def _write_component_trace(path: Path, *, cross_check: bool = True) -> None:
    samples = []
    pose_sum = 0.0
    seg_sum = 0.0
    for pair_index in range(600):
        pose = 0.000001
        seg = 0.000001
        signal = 0.0000001
        if pair_index == 0:
            pose = 0.1
            seg = 0.005
            signal = 0.01
        elif pair_index == 1:
            pose = 0.4
            seg = 0.02
            signal = 0.2
        pose_sum += pose
        seg_sum += seg
        samples.append(
            {
                "pair_index": pair_index,
                "video_name": "video.hevc",
                "video_pair_index": pair_index,
                "frame_start": pair_index * 2,
                "frame_indices": [pair_index * 2, pair_index * 2 + 1],
                "posenet_dist": pose,
                "segnet_dist": seg,
                "score_seg_contribution_exact": 100.0 * seg / 600,
                "score_pose_contribution_first_order": signal / 2,
                "score_combined_contribution_first_order": signal,
            }
        )
    payload = {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "diagnostic_component_trace",
        "n_samples": 600,
        "expected_contest_samples": 600,
        "avg_posenet_dist": pose_sum / 600,
        "avg_segnet_dist": seg_sum / 600,
        "archive_size_bytes": 594_047,
        "score_recomputed_from_components": 0.9867,
        "contest_auth_eval_cross_check": {
            "all_match": cross_check,
            "contest_auth_eval_json_sha256": "c" * 64,
        },
        "top_combined_samples": [samples[1], samples[0]],
        "samples": samples,
    }
    path.write_text(json.dumps(payload))


def _write_contest_eval(
    path: Path,
    *,
    score: float,
    bytes_: int,
    pose: float,
    seg: float,
    sha: str,
) -> None:
    payload = {
        "score_recomputed_from_components": score,
        "archive_size_bytes": bytes_,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": sha,
            "device": "cuda",
        },
    }
    path.write_text(json.dumps(payload))


def test_pair_atom_planner_emits_non_promotable_pair_policies(tmp_path: Path) -> None:
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 128)
    repair = tmp_path / "alpha4_residual_repair.amr1"
    _write_repair_payload(repair)
    manifest = tmp_path / "candidate.json"
    _write_candidate_manifest(manifest, grayscale, repair)
    pair_meta = tmp_path / "pair_weights.pt.meta.json"
    _write_pair_meta(pair_meta)

    output = tmp_path / "atom_plan.json"
    payload = planner.build_atom_plan(
        candidate_manifest_path=manifest,
        output_json=output,
        pair_weights_meta=pair_meta,
        atom_kind="pair",
        compressor="zlib",
        max_atoms=8,
        top_policy_counts=(1, 2),
    )

    assert output.exists()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["atom_count"] == 2
    assert payload["top_atoms"][0]["identity"]["pair_index"] == 0
    assert payload["top_atoms"][0]["prior"]["hard_pair_count"] == 1
    assert payload["recommended_archive_policies"][0]["policy_kind"] == "pair_indices"
    assert payload["recommended_archive_policies"][0]["policy_name"].startswith("pair_indices_")
    assert payload["water_filling_allowed"] is True
    assert payload["geometry_basin_check"] is None


def test_frame_class_atom_planner_records_class_identity(tmp_path: Path) -> None:
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 128)
    repair = tmp_path / "alpha4_residual_repair.amr1"
    _write_repair_payload(repair)
    manifest = tmp_path / "candidate.json"
    _write_candidate_manifest(manifest, grayscale, repair)

    output = tmp_path / "atom_plan.json"
    payload = planner.build_atom_plan(
        candidate_manifest_path=manifest,
        output_json=output,
        pair_weights_meta=None,
        atom_kind="frame_class",
        compressor="raw",
        max_atoms=8,
        top_policy_counts=(1,),
    )

    class_ids = {atom["identity"]["class_id"] for atom in payload["top_atoms"]}
    assert class_ids == {1, 2, 3}
    assert payload["recommended_archive_policies"] == []


def test_pair_atom_planner_blocks_water_filling_on_collapsed_geometry(tmp_path: Path) -> None:
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 128)
    repair = tmp_path / "alpha4_residual_repair.amr1"
    _write_repair_payload(repair)
    manifest = tmp_path / "candidate.json"
    _write_candidate_manifest(manifest, grayscale, repair)
    pair_meta = tmp_path / "pair_weights.pt.meta.json"
    _write_pair_meta(pair_meta)
    baseline = tmp_path / "baseline_contest_auth_eval.json"
    candidate = tmp_path / "candidate_contest_auth_eval.json"
    _write_contest_eval(
        baseline,
        score=1.0,
        bytes_=610_000,
        pose=0.0035,
        seg=0.004,
        sha="a" * 64,
    )
    _write_contest_eval(
        candidate,
        score=4.0,
        bytes_=480_000,
        pose=1.0,
        seg=0.007,
        sha="b" * 64,
    )

    payload = planner.build_atom_plan(
        candidate_manifest_path=manifest,
        output_json=tmp_path / "blocked_atom_plan.json",
        pair_weights_meta=pair_meta,
        atom_kind="pair",
        compressor="zlib",
        max_atoms=8,
        top_policy_counts=(1, 2),
        baseline_contest_json=baseline,
        candidate_contest_json=candidate,
        max_posenet_relative=1.25,
        max_segnet_relative=1.25,
    )

    assert payload["water_filling_allowed"] is False
    assert payload["recommended_archive_policies"] == []
    assert payload["geometry_basin_check"]["passed"] is False
    assert {item["component"] for item in payload["geometry_basin_check"]["violations"]} == {
        "posenet",
        "segnet",
    }
    assert payload["water_filling_blockers"]


def test_pair_atom_planner_allows_water_filling_inside_geometry_basin(tmp_path: Path) -> None:
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 128)
    repair = tmp_path / "alpha4_residual_repair.amr1"
    _write_repair_payload(repair)
    manifest = tmp_path / "candidate.json"
    _write_candidate_manifest(manifest, grayscale, repair)
    pair_meta = tmp_path / "pair_weights.pt.meta.json"
    _write_pair_meta(pair_meta)
    baseline = tmp_path / "baseline_contest_auth_eval.json"
    candidate = tmp_path / "candidate_contest_auth_eval.json"
    _write_contest_eval(
        baseline,
        score=1.0,
        bytes_=610_000,
        pose=0.004,
        seg=0.004,
        sha="a" * 64,
    )
    _write_contest_eval(
        candidate,
        score=0.99,
        bytes_=600_000,
        pose=0.0044,
        seg=0.0042,
        sha="b" * 64,
    )

    payload = planner.build_atom_plan(
        candidate_manifest_path=manifest,
        output_json=tmp_path / "allowed_atom_plan.json",
        pair_weights_meta=pair_meta,
        atom_kind="pair",
        compressor="zlib",
        max_atoms=8,
        top_policy_counts=(1,),
        baseline_contest_json=baseline,
        candidate_contest_json=candidate,
        max_posenet_relative=1.25,
        max_segnet_relative=1.25,
    )

    assert payload["water_filling_allowed"] is True
    assert payload["geometry_basin_check"]["passed"] is True
    assert payload["recommended_archive_policies"]


def test_pair_atom_planner_accepts_cuda_cross_checked_component_trace(tmp_path: Path) -> None:
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 128)
    repair = tmp_path / "alpha4_residual_repair.amr1"
    _write_repair_payload(repair)
    manifest = tmp_path / "candidate.json"
    _write_candidate_manifest(manifest, grayscale, repair)
    component_trace = tmp_path / "component_trace.json"
    _write_component_trace(component_trace)

    payload = planner.build_atom_plan(
        candidate_manifest_path=manifest,
        output_json=tmp_path / "component_trace_atom_plan.json",
        pair_weights_meta=component_trace,
        atom_kind="pair",
        compressor="zlib",
        max_atoms=8,
        top_policy_counts=(1,),
        pair_signal_top_k=2,
    )

    assert payload["pair_weights_meta"]["source_schema"] == "diagnostic_component_trace"
    assert payload["pair_weights_meta"]["signal_source"] == (
        "score_combined_contribution_first_order"
    )
    assert payload["pair_weights_meta"]["hardest_pair_indices"] == [1, 0]
    assert payload["top_atoms"][0]["identity"]["pair_index"] == 1
    assert payload["top_atoms"][0]["score_signal_prior_per_compressed_byte"] is not None
    assert payload["recommended_archive_policies"][0]["pair_indices"] == [1]


def test_pair_atom_planner_rejects_uncross_checked_component_trace(tmp_path: Path) -> None:
    component_trace = tmp_path / "component_trace.json"
    _write_component_trace(component_trace, cross_check=False)

    try:
        planner._load_pair_meta(component_trace)
    except planner.AlphaRepairAtomPlannerError as exc:
        assert "cross-check" in str(exc)
    else:
        raise AssertionError("expected uncross-checked component trace to be rejected")
