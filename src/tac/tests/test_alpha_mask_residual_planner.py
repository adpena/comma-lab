"""Tests for the Alpha mask residual planner."""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "alpha_mask_residual_planner.py"
SPEC = importlib.util.spec_from_file_location("alpha_mask_residual_planner", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


def _tiny_masks() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0, 0, 1, 1, 1, 4],
                [0, 2, 2, 1, 4, 4],
                [3, 3, 2, 4, 4, 4],
            ],
            [
                [0, 1, 1, 1, 1, 4],
                [0, 2, 2, 2, 4, 4],
                [3, 3, 2, 4, 4, 0],
            ],
        ],
        dtype=torch.int64,
    )


def _write_archive(tmp_path: Path, mask_bytes: bytes, *, unsafe_sidecar: bool = False) -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", mask_bytes)
        if unsafe_sidecar:
            zf.writestr("._masks.mkv", b"hidden")
    return archive


def _write_candidate_manifest(
    tmp_path: Path,
    *,
    unsafe_archive: bool = False,
    bad_repair_sha: bool = False,
) -> Path:
    source = _tiny_masks()
    candidate = source.clone()
    candidate[0, 1, 1:3] = 0
    candidate[1, 2, 5] = 4
    source_sha = planner.builder._tensor_u8_sha256(source)
    candidate_sha = planner.builder._tensor_u8_sha256(candidate)
    runs, selection = planner.builder._build_repair_runs(
        source,
        candidate,
        config=planner.builder.BuilderConfig(max_frames=None),
    )
    repair_payload = planner.builder._encode_repair_payload(
        runs,
        shape=tuple(int(value) for value in source.shape),
        source_mask_sha256=source_sha,
        candidate_mask_sha256=candidate_sha,
        selection_meta=selection,
    )

    mask_bytes = b"mask" * 600
    archive = _write_archive(tmp_path, mask_bytes, unsafe_sidecar=unsafe_archive)
    grayscale_path = tmp_path / "grayscale.mkv"
    repair_path = tmp_path / "alpha4_residual_repair.amr1"
    grayscale_path.write_bytes(b"gray-payload" * 3)
    repair_path.write_bytes(repair_payload)
    before = planner.builder._agreement_metrics(source, candidate)

    manifest = {
        "schema": planner.builder.SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical",
        "local_builder_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": planner.CUDA_AUTH_EVAL_PATH,
        "builder_config": {
            "class_priority": [2, 1, 3, 4, 0],
        },
        "source": {
            "archive_path": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": planner._sha256_file(archive),
            "decoded_masks": {
                "class_id_u8_sha256": source_sha,
            },
            "mask_member": {
                "name": "masks.mkv",
                "size_bytes": len(mask_bytes),
                "compressed_size_bytes": len(mask_bytes),
                "crc32": "00000000",
                "sha256": planner._sha256_bytes(mask_bytes),
            },
        },
        "candidate": {
            "score_claim": False,
            "promotion_eligible": False,
            "artifacts": [
                {
                    "role": "alpha4_grayscale_lut_video",
                    "path": str(grayscale_path),
                    "candidate_archive_member": "grayscale.mkv",
                    "size_bytes": grayscale_path.stat().st_size,
                    "sha256": planner._sha256_file(grayscale_path),
                },
                {
                    "role": "alpha4_residual_repair_payload",
                    "path": str(repair_path),
                    "candidate_archive_member": "alpha4_residual_repair.amr1",
                    "size_bytes": repair_path.stat().st_size,
                    "sha256": "0" * 64 if bad_repair_sha else planner._sha256_file(repair_path),
                },
            ],
            "alpha4": {
                "decoded_candidate_masks": {
                    "class_id_u8_sha256": candidate_sha,
                },
                "agreement_before_repair": before,
            },
        },
    }
    manifest_path = tmp_path / "alpha_mask_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def test_residual_planner_writes_non_promotable_byte_report(tmp_path: Path) -> None:
    manifest_path = _write_candidate_manifest(tmp_path)
    output_dir = tmp_path / "planner"

    report = planner.plan_residual_alternatives(
        candidate_manifest=manifest_path,
        output_dir=output_dir,
        config=planner.PlannerConfig(frame_group_size=1, max_policies=10, min_agreement=0.9),
        command=["alpha_mask_residual_planner.py", "--unit-test"],
    )

    report_path = output_dir / planner.REPORT_NAME
    assert report_path.exists()
    assert json.loads(report_path.read_text()) == report
    assert report["schema"] == "alpha_mask_residual_planner_v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["scorer_network_loaded"] is False
    assert report["custody"]["source_archive"]["validated_zip_safety"] is True
    assert report["byte_target"]["grayscale_alone_under_target"] is True
    assert report["planner_summary"]["best_under_target_meeting_min_agreement"]
    frontier = report["planner_summary"]["byte_agreement_pareto_frontier"]
    assert frontier
    assert [record["candidate_total_payload_bytes"] for record in frontier] == sorted(
        record["candidate_total_payload_bytes"] for record in frontier
    )
    assert [record["agreement_estimate_after_selected_repair"] for record in frontier] == sorted(
        record["agreement_estimate_after_selected_repair"] for record in frontier
    )

    records = report["candidate_records"]
    assert any(
        record["policy_name"] == "omit_repair_payload"
        and record["compressor"] == "omit_repair_payload"
        and record["candidate_total_payload_bytes"] == report["byte_target"]["grayscale_size_bytes"]
        for record in records
    )
    assert any(
        record["policy_kind"] == "class_priority_prefix"
        and record["compressor"] == "zlib9"
        and record["available"] is True
        for record in records
    )
    assert all(record.get("score_claim") is False for record in records)
    assert all(record.get("promotion_eligible") is False for record in records)


def test_residual_planner_fails_closed_on_unsafe_source_archive(tmp_path: Path) -> None:
    manifest_path = _write_candidate_manifest(tmp_path, unsafe_archive=True)

    with pytest.raises(ValueError, match="hidden/system archive member"):
        planner.plan_residual_alternatives(
            candidate_manifest=manifest_path,
            output_dir=tmp_path / "planner",
            config=planner.PlannerConfig(),
            command=["alpha_mask_residual_planner.py", "--unit-test"],
        )


def test_residual_planner_fails_closed_on_artifact_sha_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_candidate_manifest(tmp_path, bad_repair_sha=True)

    with pytest.raises(ValueError, match="repair artifact sha256 mismatch"):
        planner.plan_residual_alternatives(
            candidate_manifest=manifest_path,
            output_dir=tmp_path / "planner",
            config=planner.PlannerConfig(),
            command=["alpha_mask_residual_planner.py", "--unit-test"],
        )


def test_residual_planner_output_dir_requires_force(tmp_path: Path) -> None:
    manifest_path = _write_candidate_manifest(tmp_path)
    output_dir = tmp_path / "planner"
    config = planner.PlannerConfig(frame_group_size=1, max_policies=6, min_agreement=0.9)

    planner.plan_residual_alternatives(
        candidate_manifest=manifest_path,
        output_dir=output_dir,
        config=config,
        command=["alpha_mask_residual_planner.py", "--unit-test"],
    )

    with pytest.raises(FileExistsError, match="use --force"):
        planner.plan_residual_alternatives(
            candidate_manifest=manifest_path,
            output_dir=output_dir,
            config=config,
            command=["alpha_mask_residual_planner.py", "--unit-test"],
        )


def test_residual_planner_cli_defaults_are_bounded() -> None:
    parser = planner._build_arg_parser()
    args = parser.parse_args([])

    assert args.frame_group_size == planner.PlannerConfig.frame_group_size
    assert args.max_policies == planner.PlannerConfig.max_policies
    assert args.max_repair_runs == planner.PlannerConfig.max_repair_runs
    assert args.max_payload_bytes == planner.PlannerConfig.max_payload_bytes
    assert args.byte_target is None
    assert args.force is False
