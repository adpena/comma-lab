# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from tac.local_acceleration.pr95_hnerv_mlx import FALSE_AUTHORITY
from tac.optimization.representation_training_probe_integration import (
    adapt_representation_training_manifest_to_candidate,
    validate_representation_training_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_run_pr95_mlx_timing_smoke_cli_writes_queueable_manifests(tmp_path: Path) -> None:
    output_dir = tmp_path / "pr95_mlx_smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_timing_smoke.py"),
            "--stage",
            "1",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--synthetic-pairs",
            "1",
            "--seed",
            "11",
            "--base-channels",
            "4",
            "--output-dir",
            str(output_dir),
            "--write-byte-closed-smoke",
            "--write-pr95-public-archive-export",
            "--prove-pr95-runtime-consumption",
            "--runtime-proof-max-output-bytes",
            "7000000",
            "--runtime-proof-timeout-seconds",
            "180",
            "--write-source-faithful-preprocess-smoke",
            "--source-preprocess-shape",
            "1,2,8,10,3",
            "--source-preprocess-camera-hw",
            "11,13",
            "--source-preprocess-gradient-shape",
            "1,2,8,10,3",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime_profile = json.loads(
        (output_dir / "runtime_profile.json").read_text(encoding="utf-8")
    )
    representation = json.loads(
        (output_dir / "representation_training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    export_summary = json.loads(
        (output_dir / "pr95_public_archive_export.json").read_text(encoding="utf-8")
    )
    runtime_proof = json.loads(
        (output_dir / "runtime_consumption_proof.json").read_text(encoding="utf-8")
    )
    preprocess_smoke = json.loads(
        (output_dir / "source_faithful_preprocess_smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["ok"] is True
    assert summary["byte_closed_smoke_archive"]["member"] == "0.bin"
    assert summary["pr95_public_archive_export"]["schema"] == (
        "pr95_hnerv_archive_export.v1"
    )
    assert export_summary["runtime_consumption_proof_present"] is True
    assert summary["runtime_consumption_proof"]["runtime_consumption_proven"] is True
    assert summary["source_faithful_preprocess_smoke"]["source_faithful_preprocess_ready"] is True
    assert preprocess_smoke["gradient_probe"]["gradient_reachable"] is True
    assert manifest["stage_module"] == "stage1_v328_ce"
    assert manifest["source_preprocess_shape"] == "1,2,8,10,3"
    assert manifest["source_preprocess_camera_hw"] == "11,13"
    assert manifest["source_preprocess_gradient_shape"] == "1,2,8,10,3"
    assert manifest["source_faithful_preprocess_smoke_path"].endswith(
        "source_faithful_preprocess_smoke.json"
    )
    assert "runtime_consumption_proof_missing" not in manifest[
        "exact_readiness_refusal"
    ]["blockers"]
    assert "pr95_eval_roundtrip_scorer_preprocess_loss_not_ported_to_mlx" not in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert "pr95_eval_roundtrip_yuv6_preprocess_ported_but_scorer_loss_not_wired_to_mlx" in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert manifest["pr95_public_archive_export"]["sha256"] == export_summary["sha256"]
    assert manifest["runtime_consumption_proof"]["raw_output_bytes"] == (
        runtime_proof["expected_raw_bytes"]
    )
    assert manifest["optimizer_recipe"]["optimizer_descriptor_id"] == (
        "pr95_stage1_adamw_baseline_mlx"
    )
    assert len(manifest["optimizer_recipe"]["parameter_group_fingerprint_sha256"]) == 64
    assert runtime_profile["training_backend"] == "mlx"
    assert runtime_profile["optimizer_descriptor_id"] == (
        "pr95_stage1_adamw_baseline_mlx"
    )
    assert representation["training_recipe"]["quality_comparable"] is False
    assert representation["candidate_params"]["optimizer_descriptor_id"] == (
        "pr95_stage1_adamw_baseline_mlx"
    )
    assert representation["runtime_profile"]["profile_id"] == runtime_profile["profile_id"]
    assert representation["byte_closed_smoke_archive"]["sha256"] == summary[
        "byte_closed_smoke_archive"
    ]["sha256"]
    assert representation["archive_zip"]["sha256"] == export_summary["sha256"]
    assert representation["pr95_public_archive_export"]["sha256"] == export_summary["sha256"]
    assert representation["runtime_consumption_proof"]["runtime_consumption_proven"] is True
    assert representation["source_faithful_preprocess_smoke"][
        "source_faithful_preprocess_ready"
    ] is True
    assert representation["candidate_params"]["source_faithful_preprocess_smoke_present"] is True
    assert representation["candidate_params"]["source_preprocess_shape"] == "1,2,8,10,3"
    validate_representation_training_manifest(representation)
    row = adapt_representation_training_manifest_to_candidate(
        representation,
        source_path=output_dir / "representation_training_manifest.json",
        repo_root=REPO_ROOT,
    )
    timing = row["consumer_payload"]["representation_training_probe"]["timing_smoke"]
    assert timing["runtime_profile_summary"]["best_local_backend"] == "mlx"
    assert "runtime_consumption_proof_missing" not in row["dispatch_blockers"]
    assert "pr95_eval_roundtrip_scorer_preprocess_loss_not_ported_to_mlx" not in row[
        "dispatch_blockers"
    ]
    assert "pr95_eval_roundtrip_yuv6_preprocess_ported_but_scorer_loss_not_wired_to_mlx" in row[
        "dispatch_blockers"
    ]
    assert "full_frame_inflate_parity_against_source_runtime_not_run" in row[
        "dispatch_blockers"
    ]
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in row[
        "dispatch_blockers"
    ]
    assert "pr95_training_loop_not_yet_source_faithful" in row["dispatch_blockers"]
    preprocess_signal = row["consumer_payload"]["representation_training_probe"][
        "source_faithful_preprocess"
    ]
    assert preprocess_signal["present"] is True
    assert preprocess_signal["gradient_reachable"] is True
    assert preprocess_signal["exact_readiness_ready"] is False
    assert "pr95_training_loop_not_yet_source_faithful" in (
        preprocess_signal["exact_readiness_blockers"]
    )
    for payload in (summary, manifest, representation, preprocess_smoke):
        for key in FALSE_AUTHORITY:
            assert payload[key] is False
