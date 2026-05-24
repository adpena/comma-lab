# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np

from tac.optimization.family_agnostic_materializers import (
    ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
    PACKET_MEMBER_RECOMPRESS_SCHEMA,
    TENSOR_FACTORIZE_SCHEMA,
    materialize_archive_section_entropy_recode_candidate,
    materialize_packet_member_recompress_candidate,
    materialize_tensor_factorize_candidate,
)
from tac.repo_io import sha256_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_zip(path: Path, members: dict[str, bytes], *, compression: int = zipfile.ZIP_STORED) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)


def test_packet_member_recompress_materializer_preserves_member_payload(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    payload = b"A" * 8192
    _write_zip(archive, {"payload.bin": payload})

    result = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=output,
        member_name="payload.bin",
        repo_root=tmp_path,
    )

    assert result["schema"] == PACKET_MEMBER_RECOMPRESS_SCHEMA
    assert result["byte_closed_candidate_emitted"] is True
    assert result["source_member"]["sha256"] == sha256_bytes(payload)
    assert result["candidate_member"]["sha256"] == sha256_bytes(payload)
    assert result["candidate_archive"]["bytes"] < result["source_archive"]["bytes"]
    assert result["selected_compression"]["saved_bytes"] > 0
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumption_proof_missing" in result["readiness_blockers"]


def test_archive_section_entropy_recode_materializer_uses_section_manifest(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    section_a = brotli.compress(b"A" * 4096, quality=0)
    section_b = brotli.compress(b"B" * 4096, quality=0)
    member_payload = section_a + section_b
    _write_zip(archive, {"0.raw": member_payload})
    manifest = {
        "schema": "fixture_section_manifest.v1",
        "member": {"name": "0.raw"},
        "sections": [
            {
                "name": "section_a",
                "index": 0,
                "offset": 0,
                "length": len(section_a),
                "sha256": sha256_bytes(section_a),
                "optimization_role": "decoder_weight_stream",
            },
            {
                "name": "section_b",
                "index": 1,
                "offset": len(section_a),
                "length": len(section_b),
                "sha256": sha256_bytes(section_b),
                "optimization_role": "latent_stream",
            },
        ],
    }

    result = materialize_archive_section_entropy_recode_candidate(
        archive_path=archive,
        section_manifest=manifest,
        output_archive=output,
        section_names=("section_a",),
        brotli_qualities=(11,),
        repo_root=tmp_path,
    )

    assert result["schema"] == ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA
    assert result["byte_closed_candidate_emitted"] is True
    assert result["candidate_archive"]["bytes"] < result["source_archive"]["bytes"]
    assert result["section_recode"]["changed_section_count"] == 1
    assert result["sections"][0]["raw_payload_sha256"] == sha256_bytes(b"A" * 4096)
    assert result["sections"][1]["selected"] is False
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumption_proof_missing" in result["readiness_blockers"]


def test_tensor_factorize_materializer_emits_cooperative_receiver_packet(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    tensor_path = tmp_path / "tensor.npy"
    vector_a = np.arange(256, dtype=np.float32)[:, None]
    vector_b = np.linspace(0.25, 2.0, 256, dtype=np.float32)[None, :]
    np.save(tensor_path, vector_a @ vector_b)
    _write_zip(archive, {"weights.npy": tensor_path.read_bytes()})

    result = materialize_tensor_factorize_candidate(
        archive_path=archive,
        tensor_manifest={"member_name": "weights.npy"},
        factorization_contract={"rank": 1},
        output_archive=output,
        repo_root=tmp_path,
    )

    assert result["schema"] == TENSOR_FACTORIZE_SCHEMA
    assert result["byte_closed_candidate_emitted"] is True
    assert result["factorization"]["rank"] == 1
    assert result["candidate_archive"]["bytes"] < result["source_archive"]["bytes"]
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "tensor_factorized_payload_requires_cooperative_receiver" in (
        result["readiness_blockers"]
    )


def test_family_agnostic_materializer_cli_writes_false_authority_manifest(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate.json"
    _write_zip(archive, {"payload.bin": b"A" * 4096})

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_family_agnostic_materializer.py"),
            "--target-kind",
            "packet_member_recompress_v1",
            "--archive-path",
            str(archive),
            "--output-archive",
            str(output),
            "--output-manifest",
            str(manifest),
            "--member-name",
            "payload.bin",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout)
    assert payload["schema"] == PACKET_MEMBER_RECOMPRESS_SCHEMA
    assert stdout_payload["schema"] == PACKET_MEMBER_RECOMPRESS_SCHEMA
    assert payload["tool_run_manifest"]["tool"] == "tools/run_family_agnostic_materializer.py"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
