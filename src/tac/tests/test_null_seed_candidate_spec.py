# SPDX-License-Identifier: MIT
"""Tests for null-seed candidate spec lowering."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.procedural_codebook_generator import (
    NullSeedCandidateSpecError,
    build_null_seed_candidate_spec,
    derive_codebook_from_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _archive(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def _plan(archive_path: Path, payload: bytes, start: int, end: int) -> dict[str, object]:
    span = payload[start:end]
    return {
        "schema": "null_space_seed_replacement_plan_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "input_paths": {
            "archive_zip": str(archive_path),
            "inner_member_name": "x",
            "null_indices": "not-needed.npy",
            "null_summary": "not-needed.json",
        },
        "inputs": {
            "inner_bytes_sha256": hashlib.sha256(payload).hexdigest(),
            "seed_bytes": 8,
        },
        "candidates": [
            {
                "candidate_id": "null-seed-contiguous-source-8-24",
                "source": "contiguous_null_run",
                "section": "source_payload",
                "range": [start, end],
                "original_bytes": end - start,
                "seed_bytes": 8,
                "runtime_header_bytes": 0,
                "net_saved_inner_bytes": end - start - 8,
                "predicted_rate_delta_upper_bound": -0.0001,
                "original_sha256": hashlib.sha256(span).hexdigest(),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    }


def test_null_seed_candidate_spec_blocks_generative_substitute(tmp_path: Path) -> None:
    payload = bytes(range(64))
    archive_path = tmp_path / "archive.zip"
    _archive(archive_path, payload)

    spec = build_null_seed_candidate_spec(
        _plan(archive_path, payload, 8, 32),
        seed_material=b"abcdefgh",
        generator_kind="pcg64",
    )

    assert spec["score_claim"] is False
    assert spec["promotion_eligible"] is False
    assert spec["raw_archive_byte_coordinates_allowed"] is False
    assert spec["direct_replacement_ready"] is False
    assert spec["runtime_adapter_required"] is True
    assert "seed_reconstruction_mismatch" in spec["blockers"]
    assert "source_payload_seed_substitution_parse_risk" in spec["blockers"]
    assert spec["candidate_modification_spec"]["packet_proofs_available"] is False


def test_null_seed_candidate_spec_accepts_direct_reconstruction(tmp_path: Path) -> None:
    seed = b"abcdefgh"
    derived = derive_codebook_from_seed(
        seed,
        output_shape=(24,),
        dtype=np.uint8,
        generator_kind="pcg64",
    ).tobytes()
    payload = b"prefix00" + derived + b"suffix"
    archive_path = tmp_path / "archive.zip"
    _archive(archive_path, payload)

    spec = build_null_seed_candidate_spec(
        _plan(archive_path, payload, 8, 32),
        seed_material=seed,
        generator_kind="pcg64",
    )

    assert spec["direct_replacement_ready"] is True
    assert spec["runtime_adapter_required"] is False
    assert "seed_reconstruction_mismatch" not in spec["blockers"]
    assert spec["seed_replacement"]["seed_reconstructs_original_payload"] is True


def test_null_seed_candidate_spec_fails_on_stale_original_sha(tmp_path: Path) -> None:
    payload = bytes(range(64))
    archive_path = tmp_path / "archive.zip"
    _archive(archive_path, payload)
    plan = _plan(archive_path, payload, 8, 32)
    plan["candidates"][0]["original_sha256"] = "0" * 64  # type: ignore[index]

    with pytest.raises(NullSeedCandidateSpecError, match="original_sha256 mismatch"):
        build_null_seed_candidate_spec(plan, seed_material=b"abcdefgh")


def test_build_null_seed_candidate_spec_cli(tmp_path: Path) -> None:
    payload = bytes(range(64))
    archive_path = tmp_path / "archive.zip"
    plan_path = tmp_path / "plan.json"
    output_json = tmp_path / "spec.json"
    output_md = tmp_path / "spec.md"
    _archive(archive_path, payload)
    plan_path.write_text(json.dumps(_plan(archive_path, payload, 8, 32)), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_null_seed_candidate_spec.py"),
            "--plan-json",
            str(plan_path),
            "--seed-hex",
            b"abcdefgh".hex(),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    spec = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout["schema"] == "null_seed_candidate_spec_v1"
    assert stdout["runtime_adapter_required"] is True
    assert spec["verdict"] == "blocked_until_runtime_adapter_and_exact_eval"
    assert "Runtime adapter required: `true`" in output_md.read_text(encoding="utf-8")
