"""Tests for the Alpha lossy repair budget planner."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "alpha_lossy_repair_budget_planner.py"
SPEC = importlib.util.spec_from_file_location("alpha_lossy_repair_budget_planner", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def _write_archive(path: Path, mask_bytes: bytes = b"mask-member" * 100) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("masks.mkv", mask_bytes)
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("optimized_poses.bin", b"poses")
    return path


def _write_matrix_manifest(tmp_path: Path, *, score_claim: bool = False, bad_artifact_sha: bool = False) -> Path:
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    archive = _write_archive(output_dir / "archive.zip")
    mask_bytes = b"mask-member" * 100
    exact_small = output_dir / "coco.amcrle"
    exact_large = output_dir / "transitions.amcte"
    exact_small.write_bytes(b"exact-rle" * 170)
    exact_large.write_bytes(b"transition" * 190)

    manifest = {
        "schema": "alpha_mask_codec_candidate_matrix_v1",
        "score_claim": score_claim,
        "promotion_eligible": False,
        "evidence_grade": "empirical",
        "local_planning_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": planner.CUDA_AUTH_EVAL_PATH,
        "source": {
            "archive_path": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": planner._sha256_file(archive),
            "decoded_masks": {
                "class_id_u8_sha256": "a" * 64,
                "shape": [4, 8, 8],
                "num_pixels": 256,
            },
            "mask_member": {
                "name": "masks.mkv",
                "size_bytes": len(mask_bytes),
                "compressed_size_bytes": len(mask_bytes) - 10,
                "sha256": planner._sha256_bytes(mask_bytes),
            },
        },
        "candidates": [
            {
                "name": "coco_rle_per_frame_foreground_runs",
                "family": "coco_rle",
                "payload_format": "alpha_mask_coco_rle_runs_v1",
                "charged_representation": True,
                "diagnostic_reference": False,
                "exact_reconstruction": True,
                "score_claim": False,
                "promotion_eligible": False,
                "evidence_grade": "empirical",
                "runtime_decoder_integration_required": True,
                "artifact": {
                    "role": "coco_rle",
                    "path": str(exact_small),
                    "candidate_archive_member": "coco.amcrle",
                    "size_bytes": exact_small.stat().st_size,
                    "sha256": "0" * 64 if bad_artifact_sha else planner._sha256_file(exact_small),
                },
                "agreement": {
                    "argmax_agreement": 1.0,
                    "different_pixels": 0,
                    "exact_reconstruction": True,
                },
            },
            {
                "name": "class_transition_endpoint_packets",
                "family": "transition_endpoints",
                "payload_format": "alpha_mask_class_transition_endpoints_v1",
                "charged_representation": True,
                "diagnostic_reference": False,
                "exact_reconstruction": True,
                "score_claim": False,
                "promotion_eligible": False,
                "evidence_grade": "empirical",
                "runtime_decoder_integration_required": True,
                "artifact": {
                    "role": "transition_endpoints",
                    "path": str(exact_large),
                    "candidate_archive_member": "transitions.amcte",
                    "size_bytes": exact_large.stat().st_size,
                    "sha256": planner._sha256_file(exact_large),
                },
                "agreement": {
                    "argmax_agreement": 1.0,
                    "different_pixels": 0,
                    "exact_reconstruction": True,
                },
            },
        ],
    }
    path = output_dir / "alpha_mask_codec_candidate_matrix.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _point(
    tmp_path: Path,
    *,
    index: int,
    kind: str,
    changed_pixels: int,
    mask_member_size: int,
    selection_weight: float,
    source_class: int,
    target_class: int,
) -> dict[str, Any]:
    archive_rel = f"archives/point_{index:03d}.zip"
    archive_path = tmp_path / "primitive" / archive_rel
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(f"archive-{index}".encode("ascii") * 50)
    primitive_id = f"{kind}_{index:03d}"
    return {
        "index": index,
        "archive": archive_rel,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": planner._sha256_file(archive_path),
        "epsilon": float(index),
        "official_component_response": False,
        "predicted_delta": None,
        "score_claim": False,
        "promotion_eligible": False,
        "selection_weight": selection_weight,
        "primitive_id": primitive_id,
        "primitive": {
            "primitive_id": primitive_id,
            "kind": kind,
            "operation": "unit_test_mutation",
            "frame_index": index,
            "source_class": source_class,
            "target_class": target_class,
            "selection_weight": selection_weight,
            "params": {},
        },
        "mask_delta": {
            "changed_pixels": changed_pixels,
            "changed_fraction": changed_pixels / 256.0,
            "selected_pixels_before_cap": changed_pixels * 2,
            "selected_pixels_after_cap": changed_pixels,
        },
        "mask_member": {
            "name": "masks.mkv",
            "size_bytes": mask_member_size,
            "sha256": f"{index:064x}"[-64:],
        },
    }


def _write_primitive_plan(tmp_path: Path, *, bad_variants_sha: bool = False) -> Path:
    output_dir = tmp_path / "primitive"
    output_dir.mkdir()
    baseline = _write_archive(output_dir / "baseline.zip")
    points = [
        {
            "index": 0,
            "role": "baseline",
            "epsilon": 0.0,
            "archive_bytes": baseline.stat().st_size,
            "archive_sha256": planner._sha256_file(baseline),
            "official_component_response": False,
            "predicted_delta": {"combined": 0.0, "posenet": 0.0, "segnet": 0.0},
            "score_claim": False,
            "promotion_eligible": False,
        },
        _point(
            tmp_path,
            index=1,
            kind="connected_component",
            changed_pixels=64,
            mask_member_size=280,
            selection_weight=0.8,
            source_class=2,
            target_class=0,
        ),
        _point(
            tmp_path,
            index=2,
            kind="boundary_band",
            changed_pixels=32,
            mask_member_size=300,
            selection_weight=0.9,
            source_class=1,
            target_class=0,
        ),
        _point(
            tmp_path,
            index=3,
            kind="transition_endpoint",
            changed_pixels=16,
            mask_member_size=310,
            selection_weight=0.4,
            source_class=0,
            target_class=3,
        ),
    ]
    variants = {
        "schema_version": 1,
        "format": "alpha_mask_primitive_archive_variants_v1",
        "producer": "experiments/build_alpha_mask_primitive_response_plan.py",
        "score_claim": False,
        "promotion_eligible": False,
        "official_component_response": False,
        "evidence_grade": "empirical",
        "baseline_archive": {
            "path_hint": str(baseline),
            "bytes": baseline.stat().st_size,
            "sha256": planner._sha256_file(baseline),
        },
        "points": points[1:],
        "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        "auth_eval_required": "cuda",
    }
    variants_path = output_dir / "alpha_mask_primitive_archive_variants_manifest.json"
    variants_path.write_text(json.dumps(variants, indent=2, sort_keys=True) + "\n")
    variants_sha = planner._sha256_file(variants_path)

    plan = {
        "schema_version": 1,
        "format": "official_component_response_plan_v1",
        "alpha_plan_format": "alpha_mask_primitive_component_response_plan_v1",
        "producer": "experiments/build_alpha_mask_primitive_response_plan.py",
        "score_claim": False,
        "promotion_eligible": False,
        "official_component_response": False,
        "evidence_grade": "empirical",
        "local_diagnostic_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": planner.CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": "No score claim.",
        "baseline_archive": {
            "path_hint": str(baseline),
            "bytes": baseline.stat().st_size,
            "sha256": planner._sha256_file(baseline),
        },
        "source": {
            "baseline_archive": {
                "path_hint": str(baseline),
                "bytes": baseline.stat().st_size,
                "sha256": planner._sha256_file(baseline),
            },
            "decoded_masks": {
                "class_id_u8_sha256": "a" * 64,
                "shape": [4, 8, 8],
                "num_pixels": 256,
            },
            "mask_member": {
                "name": "masks.mkv",
                "raw_bytes": 1000,
                "sha256": "b" * 64,
            },
            "validated_zip_safety": True,
        },
        "perturbation": {
            "format": "alpha_mask_primitive_component_response_plan_v1",
            "basis_kind": "alpha_mask_geometry_primitive",
            "archive_variants_manifest": "alpha_mask_primitive_archive_variants_manifest.json",
            "archive_variants_manifest_sha256": "0" * 64 if bad_variants_sha else variants_sha,
            "auth_eval_required": "cuda",
            "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
            "primitive_count": 3,
            "primitive_ids": [point["primitive_id"] for point in points[1:]],
        },
        "points": points,
    }
    plan_path = output_dir / "alpha_mask_primitive_response_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan_path


def test_lossy_repair_budget_planner_writes_non_promotable_specs(tmp_path: Path) -> None:
    matrix_path = _write_matrix_manifest(tmp_path)
    primitive_path = _write_primitive_plan(tmp_path)
    output_dir = tmp_path / "budget"

    report = planner.plan_lossy_repair_budgets(
        primitive_plan=primitive_path,
        matrix_manifest=matrix_path,
        output_dir=output_dir,
        config=planner.BudgetConfig(max_specs=6, max_policy_points=2),
        command=["alpha_lossy_repair_budget_planner.py", "--unit-test"],
    )

    report_path = output_dir / planner.REPORT_NAME
    assert report_path.exists()
    assert json.loads(report_path.read_text()) == report
    assert report["schema"] == "alpha_lossy_repair_budget_planner_v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["scorer_network_loaded"] is False
    assert report["remote_jobs_launched"] is False
    assert report["archives_built"] is False
    assert "component-response" in report["planner_summary"]["next_step"]

    matrix_context = report["matrix_context"]
    assert matrix_context["source_mask_member_size_bytes"] == len(b"mask-member" * 100)
    assert matrix_context["best_exact_candidate"]["family"] == "coco_rle"
    assert matrix_context["best_exact_candidate"]["size_bytes"] > matrix_context["source_mask_member_size_bytes"]

    records = report["budget_records"]
    assert records == sorted(
        records,
        key=lambda item: (
            item["estimated_mask_stream_bytes"],
            -item["repair_estimate"]["selected_changed_pixels"],
            item["spec_id"],
        ),
    )
    assert all(record["score_claim"] is False for record in records)
    assert all(record["promotion_eligible"] is False for record in records)
    assert all(record["candidate_archive_built"] is False for record in records)
    assert any(record["under_current_mask_member"] is True for record in records)

    specs = report["candidate_next_archive_specs"]
    assert specs
    assert len(specs) <= 6
    for spec_record in specs:
        spec_path = REPO_ROOT / spec_record["path"] if not Path(spec_record["path"]).is_absolute() else Path(spec_record["path"])
        spec = json.loads(spec_path.read_text())
        assert spec["schema"] == "alpha_lossy_sparse_repair_archive_build_spec_v1"
        assert spec["candidate_archive_built"] is False
        assert spec["archive_path"] is None
        assert spec["score_claim"] is False
        assert spec["promotion_eligible"] is False
        assert spec["requires_exact_cuda_auth_eval"] is True
        assert spec["component_response_handoff"]["status"] == "pending"


def test_lossy_repair_budget_planner_rejects_score_claim_inputs(tmp_path: Path) -> None:
    matrix_path = _write_matrix_manifest(tmp_path, score_claim=True)
    primitive_path = _write_primitive_plan(tmp_path)

    with pytest.raises(planner.AlphaLossyRepairPlannerError, match="matrix.score_claim must be false"):
        planner.plan_lossy_repair_budgets(
            primitive_plan=primitive_path,
            matrix_manifest=matrix_path,
            output_dir=tmp_path / "budget",
            config=planner.BudgetConfig(),
            command=["alpha_lossy_repair_budget_planner.py", "--unit-test"],
        )


def test_lossy_repair_budget_planner_rejects_variants_sha_mismatch(tmp_path: Path) -> None:
    matrix_path = _write_matrix_manifest(tmp_path)
    primitive_path = _write_primitive_plan(tmp_path, bad_variants_sha=True)

    with pytest.raises(planner.AlphaLossyRepairPlannerError, match="variants manifest.*sha256 mismatch"):
        planner.plan_lossy_repair_budgets(
            primitive_plan=primitive_path,
            matrix_manifest=matrix_path,
            output_dir=tmp_path / "budget",
            config=planner.BudgetConfig(),
            command=["alpha_lossy_repair_budget_planner.py", "--unit-test"],
        )


def test_lossy_repair_budget_planner_rejects_matrix_artifact_sha_mismatch(tmp_path: Path) -> None:
    matrix_path = _write_matrix_manifest(tmp_path, bad_artifact_sha=True)
    primitive_path = _write_primitive_plan(tmp_path)

    with pytest.raises(planner.AlphaLossyRepairPlannerError, match="matrix candidate artifact.*sha256 mismatch"):
        planner.plan_lossy_repair_budgets(
            primitive_plan=primitive_path,
            matrix_manifest=matrix_path,
            output_dir=tmp_path / "budget",
            config=planner.BudgetConfig(),
            command=["alpha_lossy_repair_budget_planner.py", "--unit-test"],
        )


def test_lossy_repair_budget_planner_cli_defaults_are_bounded() -> None:
    parser = planner._build_arg_parser()
    args = parser.parse_args([])

    assert args.max_primitive_points == planner.BudgetConfig.max_primitive_points
    assert args.max_specs == planner.BudgetConfig.max_specs
    assert args.max_policy_points == planner.BudgetConfig.max_policy_points
    assert args.lossy_base_bytes == ()
