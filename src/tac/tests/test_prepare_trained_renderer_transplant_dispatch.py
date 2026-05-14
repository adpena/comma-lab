# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
PREPARE_PATH = REPO / "experiments" / "prepare_trained_renderer_transplant_dispatch.py"


def _load_prepare(name: str = "_prepare_trained_renderer_dispatch_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, PREPARE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_eval_json(path: Path, *, archive_bytes: int, score: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "archive_size_bytes": archive_bytes,
                "score_recomputed_from_components": score,
                "avg_segnet_dist": 0.00061038,
                "avg_posenet_dist": 0.00049601,
                "provenance": {"device": "cuda"},
                "num_samples": 600,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_pose_safety_report(
    path: Path,
    *,
    source_sha: str,
    candidate_sha: str,
    safe: bool = True,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "renderer_transplant_pose_safety_preflight_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "remote_gpu_dispatch_performed": False,
                "safe_for_exact_eval_dispatch": safe,
                "failure_class": None if safe else "renderer_transplant_pose_safety_failed",
                "fail_closed_reasons": [] if safe else ["render_output_parity_unsafe"],
                "source_archive": {"sha256": source_sha},
                "candidate_archive": {"sha256": candidate_sha},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_prepare_refuses_missing_renderer_export(tmp_path: Path) -> None:
    prepare = _load_prepare("_prepare_missing_export_test")
    source = tmp_path / "source.zip"
    source.write_bytes(b"source-archive")
    eval_json = _write_eval_json(
        tmp_path / "contest_auth_eval.adjudicated.json",
        archive_bytes=source.stat().st_size,
        score=0.3154707273953505,
    )

    with pytest.raises(FileNotFoundError, match="renderer export does not exist"):
        prepare.build_plan(
            renderer_export=tmp_path / "missing_renderer.bin",
            source_archive=source,
            eval_json=eval_json,
            output_dir=tmp_path / "out",
            expected_source_bytes=source.stat().st_size,
        )


def test_prepare_verifies_source_bytes_and_sha(tmp_path: Path) -> None:
    prepare = _load_prepare("_prepare_source_custody_test")
    source = tmp_path / "source.zip"
    source.write_bytes(b"source-archive")
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"QZS3renderer")
    eval_json = _write_eval_json(
        tmp_path / "contest_auth_eval.adjudicated.json",
        archive_bytes=source.stat().st_size,
        score=0.3154707273953505,
    )

    with pytest.raises(ValueError, match="bytes do not match expected custody"):
        prepare.build_plan(
            renderer_export=renderer,
            source_archive=source,
            eval_json=eval_json,
            output_dir=tmp_path / "out",
            expected_source_bytes=source.stat().st_size + 1,
        )

    with pytest.raises(ValueError, match="SHA-256 does not match expected custody"):
        prepare.build_plan(
            renderer_export=renderer,
            source_archive=source,
            eval_json=eval_json,
            output_dir=tmp_path / "out",
            expected_source_bytes=source.stat().st_size,
            expected_source_sha256="0" * 64,
        )


def test_prepare_emits_deterministic_commands_and_blocks_without_pose_safety(
    tmp_path: Path,
) -> None:
    prepare = _load_prepare("_prepare_blocked_test")
    source = tmp_path / "source.zip"
    source.write_bytes(b"x" * 276_342)
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"QZS3trained-renderer")
    eval_json = _write_eval_json(
        tmp_path / "contest_auth_eval.adjudicated.json",
        archive_bytes=source.stat().st_size,
        score=0.3154707273953505,
    )

    plan = prepare.build_plan(
        renderer_export=renderer,
        source_archive=source,
        eval_json=eval_json,
        output_dir=tmp_path / "out",
        expected_source_bytes=source.stat().st_size,
        modal_call_ids=("fc-test-h100",),
    )

    assert plan["schema"] == "trained_renderer_transplant_dispatch_prepare_v1"
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["remote_gpu_dispatch_performed"] is False
    assert plan["source_archive"]["verified"] is True
    assert plan["exact_eval_dispatch_ready"] is False
    assert plan["pose_safety_gate"]["status"] == "waiting_for_candidate_archive"
    assert (
        plan["score_break_even"]["bytes_to_save_at_unchanged_distortion"]
        == 2209
    )
    assert (
        plan["score_break_even"]["max_archive_bytes_for_byte_only_crossing"]
        == 274_133
    )
    assert "experiments/build_renderer_shrink_candidate.py" in plan["commands"][
        "build_candidate_archives"
    ]["text"]
    assert "--qzs3-block-sizes" in plan["commands"]["build_candidate_archives"]["argv"]
    assert "experiments/preflight_renderer_transplant_pose_safety.py" in plan[
        "commands"
    ]["run_pose_safety_preflight"]["text"]
    assert "tools/claim_lane_dispatch.py claim" in plan["commands"][
        "dispatch_claim_command"
    ]["text"]
    expected_score = plan["source_archive"]["eval_score"]["score_recomputed_from_components"]
    assert f"--baseline-score {expected_score:.17g}" in plan["commands"][
        "lightning_exact_eval_dry_run_command"
    ]["text"]
    assert "--baseline-archive-bytes 276342" in plan["commands"][
        "lightning_exact_eval_dry_run_command"
    ]["text"]
    assert plan["commands"]["recover_modal_exports"][0]["argv"][-1] == "fc-test-h100"


def test_prepare_marks_ready_only_with_matching_pose_safety_json(
    tmp_path: Path,
) -> None:
    prepare = _load_prepare("_prepare_ready_test")
    source = tmp_path / "source.zip"
    source.write_bytes(b"x" * 276_342)
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"QZS3trained-renderer")
    candidate_archive = tmp_path / "out" / "preflight" / "trained_qbf1_b0064" / "archive.zip"
    candidate_archive.parent.mkdir(parents=True)
    candidate_archive.write_bytes(b"candidate-archive")
    eval_json = _write_eval_json(
        tmp_path / "contest_auth_eval.adjudicated.json",
        archive_bytes=source.stat().st_size,
        score=0.3154707273953505,
    )
    source_sha = prepare._sha256_file(source)
    candidate_sha = prepare._sha256_file(candidate_archive)
    preflight_summary = tmp_path / "out" / "preflight" / "summary.json"
    preflight_summary.write_text(
        json.dumps(
            {
                "schema": "renderer_shrink_candidate_summary_v1",
                "best_by_archive_bytes": {
                    "candidate_id": "trained_qbf1_b0064",
                    "archive": str(candidate_archive),
                    "archive_bytes": candidate_archive.stat().st_size,
                    "archive_sha256": candidate_sha,
                    "manifest": str(candidate_archive.with_name("build_manifest.json")),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pose_safety = _write_pose_safety_report(
        tmp_path / "pose_safety.json",
        source_sha=source_sha,
        candidate_sha=candidate_sha,
    )

    without_pose = prepare.build_plan(
        renderer_export=renderer,
        source_archive=source,
        eval_json=eval_json,
        output_dir=tmp_path / "out",
        expected_source_bytes=source.stat().st_size,
    )
    assert without_pose["exact_eval_dispatch_ready"] is False
    assert without_pose["pose_safety_gate"]["status"] == "missing_pose_safety_json"

    ready = prepare.build_plan(
        renderer_export=renderer,
        source_archive=source,
        eval_json=eval_json,
        output_dir=tmp_path / "out",
        expected_source_bytes=source.stat().st_size,
        pose_safety_json=(pose_safety,),
    )

    assert ready["selected_candidate"]["archive_matches_preflight_summary"] is True
    assert ready["selected_candidate"]["byte_only_crosses_target"] is True
    assert ready["pose_safety_gate"]["status"] == "pass"
    assert ready["exact_eval_dispatch_ready"] is True
    assert ready["exact_eval_dispatch_blockers"] == []
    assert "trained_qbf1_b0064" in ready["commands"]["dispatch_claim_command"]["text"]
    assert "--dry-run" in ready["commands"]["lightning_exact_eval_dry_run_command"]["argv"]


def test_prepare_can_emit_blocked_plan_when_modal_export_is_not_terminal(
    tmp_path: Path,
) -> None:
    prepare = _load_prepare("_prepare_missing_terminal_export_test")
    source = tmp_path / "source.zip"
    source.write_bytes(b"x" * 276_481)
    eval_json = _write_eval_json(
        tmp_path / "contest_auth_eval.adjudicated.json",
        archive_bytes=source.stat().st_size,
        score=0.31516575028285976,
    )

    plan = prepare.build_plan(
        renderer_export=None,
        source_archive=source,
        eval_json=eval_json,
        output_dir=tmp_path / "out",
        expected_source_bytes=source.stat().st_size,
        allow_missing_renderer_export=True,
        modal_call_ids=("fc-h100-running",),
    )

    assert plan["terminal_exports_exist"] is False
    assert plan["renderer_export"]["exists"] is False
    assert plan["exact_eval_dispatch_ready"] is False
    assert "no terminal Modal renderer export is available locally" in plan[
        "missing_prerequisites"
    ]
    assert "<recovered_renderer_qzs3.bin>" in plan["commands"][
        "build_candidate_archives"
    ]["text"]
    assert plan["commands"]["recover_modal_exports"][0]["argv"][-1] == "fc-h100-running"
    assert (
        plan["score_break_even"]["bytes_to_save_at_unchanged_distortion"]
        == 1751
    )


def test_prepare_main_writes_manifest_and_markdown(tmp_path: Path) -> None:
    prepare = _load_prepare("_prepare_main_write_test")
    source = tmp_path / "source.zip"
    source.write_bytes(b"source-archive")
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"QZS3renderer")
    eval_json = _write_eval_json(
        tmp_path / "contest_auth_eval.adjudicated.json",
        archive_bytes=source.stat().st_size,
        score=0.3154707273953505,
    )
    output_dir = tmp_path / "out"

    rc = prepare.main(
        [
            "--renderer-export",
            str(renderer),
            "--source-archive",
            str(source),
            "--eval-json",
            str(eval_json),
            "--output-dir",
            str(output_dir),
            "--expected-source-bytes",
            str(source.stat().st_size),
        ]
    )

    assert rc == 0
    manifest = json.loads((output_dir / "handoff_manifest.json").read_text())
    assert manifest["exact_eval_dispatch_ready"] is False
    assert (output_dir / "handoff_manifest.md").exists()
