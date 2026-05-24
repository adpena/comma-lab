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

    assert summary["ok"] is True
    assert summary["byte_closed_smoke_archive"]["member"] == "0.bin"
    assert manifest["stage_module"] == "stage1_v328_ce"
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
    validate_representation_training_manifest(representation)
    row = adapt_representation_training_manifest_to_candidate(
        representation,
        source_path=output_dir / "representation_training_manifest.json",
        repo_root=REPO_ROOT,
    )
    timing = row["consumer_payload"]["representation_training_probe"]["timing_smoke"]
    assert timing["runtime_profile_summary"]["best_local_backend"] == "mlx"
    assert "runtime_consumption_proof_missing" in row["dispatch_blockers"]
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in row[
        "dispatch_blockers"
    ]
    for payload in (summary, manifest, representation):
        for key in FALSE_AUTHORITY:
            assert payload[key] is False
