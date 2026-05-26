# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "import_frontier_component_response_cache.py"


def _false_authority() -> dict[str, bool]:
    return {
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_claim": False,
        "score_claim_valid": False,
        "score_claim_eligible": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "gpu_launched": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_import_frontier_component_response_cache_cli_copies_false_authority_json(
    tmp_path: Path,
) -> None:
    source_advisory = tmp_path / "source" / "local_cpu_advisory.json"
    source_hashes = tmp_path / "source" / "scorer_input_cache_hashes.json"
    advisory_out = tmp_path / "out" / "local_cpu_advisory.json"
    hashes_out = tmp_path / "out" / "scorer_input_cache_hashes.json"
    _write_json(
        source_advisory,
        {
            "schema": "unit_local_cpu_advisory.v1",
            "score_axis": "cpu_advisory",
            "component": {"pose": 1.0},
            **_false_authority(),
        },
    )
    _write_json(
        source_hashes,
        {
            "schema_version": "mlx_scorer_input_cache_hashes.v1",
            "pairs": [],
            **_false_authority(),
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source-local-cpu-advisory",
            str(source_advisory),
            "--source-scorer-input-cache-hashes",
            str(source_hashes),
            "--local-cpu-advisory-out",
            str(advisory_out),
            "--scorer-input-cache-hashes-out",
            str(hashes_out),
            "--role",
            "candidate",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(proc.stdout)
    assert report["schema"] == "frontier_component_response_cache_import.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert json.loads(advisory_out.read_text(encoding="utf-8"))["score_axis"] == (
        "cpu_advisory"
    )
    assert json.loads(hashes_out.read_text(encoding="utf-8"))["schema_version"] == (
        "mlx_scorer_input_cache_hashes.v1"
    )


def test_import_frontier_component_response_cache_cli_refuses_truthy_authority(
    tmp_path: Path,
) -> None:
    source_advisory = tmp_path / "source" / "local_cpu_advisory.json"
    source_hashes = tmp_path / "source" / "scorer_input_cache_hashes.json"
    _write_json(
        source_advisory,
        {
            "schema": "unit_local_cpu_advisory.v1",
            "score_axis": "cpu_advisory",
            "score_claim": True,
        },
    )
    _write_json(
        source_hashes,
        {
            "schema_version": "mlx_scorer_input_cache_hashes.v1",
            **_false_authority(),
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source-local-cpu-advisory",
            str(source_advisory),
            "--source-scorer-input-cache-hashes",
            str(source_hashes),
            "--local-cpu-advisory-out",
            str(tmp_path / "out" / "local_cpu_advisory.json"),
            "--scorer-input-cache-hashes-out",
            str(tmp_path / "out" / "scorer_input_cache_hashes.json"),
            "--role",
            "candidate",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "forbidden truthy authority fields" in proc.stderr
