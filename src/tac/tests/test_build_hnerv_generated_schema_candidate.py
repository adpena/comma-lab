# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_hnerv_generated_schema_candidate.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_hnerv_generated_schema_candidate",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_hnerv_generated_schema_candidate"] = module
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    hngs_decoder = tmp_path / "decoder.hngs"
    latent_blob = tmp_path / "latents.bin"
    sidecar_blob = tmp_path / "sidecar.bin"
    hngs_decoder.write_bytes(b"HNGS" + b"\x00generated-schema-decoder")
    latent_blob.write_bytes(b"latent-payload")
    sidecar_blob.write_bytes(b"sidecar-payload")
    return hngs_decoder, latent_blob, sidecar_blob


def test_cli_builds_deterministic_single_member_stored_archive(tmp_path: Path) -> None:
    tool = _load_tool()
    hngs_decoder, latent_blob, sidecar_blob = _write_inputs(tmp_path)
    out_a = tmp_path / "candidate-a.zip"
    out_b = tmp_path / "candidate-b.zip"
    manifest_a = tmp_path / "manifest-a.json"
    manifest_b = tmp_path / "manifest-b.json"
    argv_base = [
        "--hngs-decoder",
        str(hngs_decoder),
        "--latent-blob",
        str(latent_blob),
        "--sidecar-blob",
        str(sidecar_blob),
        "--candidate-id",
        "unit_hngp_schema",
    ]

    assert tool.main(
        [
            *argv_base,
            "--output-archive",
            str(out_a),
            "--manifest-output",
            str(manifest_a),
        ]
    ) == 0
    assert tool.main(
        [
            *argv_base,
            "--output-archive",
            str(out_b),
            "--manifest-output",
            str(manifest_b),
        ]
    ) == 0

    archive_bytes = out_a.read_bytes()
    assert out_b.read_bytes() == archive_bytes
    with zipfile.ZipFile(out_a) as zf:
        infos = zf.infolist()
        assert len(infos) == 1
        info = infos[0]
        packet = zf.read(info.filename)
    assert info.filename == "unit_hngp_schema.hngp"
    assert info.compress_type == zipfile.ZIP_STORED
    assert info.date_time == (1980, 1, 1, 0, 0, 0)
    assert info.create_system == 3
    assert info.external_attr == 0o100644 << 16

    manifest = json.loads(manifest_a.read_text(encoding="utf-8"))
    assert manifest["archive"]["bytes"] == len(archive_bytes)
    assert manifest["archive"]["sha256"] == hashlib.sha256(archive_bytes).hexdigest()
    assert manifest["packet"]["sha256"] == hashlib.sha256(packet).hexdigest()
    assert manifest["archive"]["member_sha256"] == manifest["packet"]["sha256"]
    assert [section["name"] for section in manifest["sections"]] == [
        "header",
        "hngs_decoder",
        "latent_blob",
        "sidecar_blob",
    ]


def test_rejects_unsafe_member_name_from_candidate_id(tmp_path: Path) -> None:
    tool = _load_tool()
    hngs_decoder, latent_blob, sidecar_blob = _write_inputs(tmp_path)
    output = tmp_path / "candidate.zip"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(
        tool.HNeRVGeneratedSchemaCandidateError,
        match="unsafe candidate_id",
    ):
        tool.build_hnerv_generated_schema_candidate_archive(
            hngs_decoder=hngs_decoder,
            latent_blob=latent_blob,
            sidecar_blob=sidecar_blob,
            output_archive=output,
            manifest_output=manifest,
            candidate_id="../escape",
        )

    assert not output.exists()
    assert not manifest.exists()


def test_manifest_fields_are_non_promotable_and_non_dispatchable(tmp_path: Path) -> None:
    tool = _load_tool()
    hngs_decoder, latent_blob, sidecar_blob = _write_inputs(tmp_path)
    manifest = tool.build_hnerv_generated_schema_candidate_archive(
        hngs_decoder=hngs_decoder,
        latent_blob=latent_blob,
        sidecar_blob=sidecar_blob,
        output_archive=tmp_path / "candidate.zip",
        manifest_output=tmp_path / "manifest.json",
        candidate_id="unit_non_promotable",
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["gpu_required"] is False
    assert manifest["runtime_inflate_included"] is False
    assert manifest["omx_state_touched"] is False
    assert "non_score_archive_wrapper_no_runtime_inflate" in manifest["dispatch_blockers"]
    assert "contest_cuda_auth_eval_missing" in manifest["promotion_blockers"]
    assert ".omx/state" not in json.dumps(manifest, sort_keys=True)


def test_rejects_malformed_hngs_before_writing_outputs(tmp_path: Path) -> None:
    tool = _load_tool()
    hngs_decoder, latent_blob, sidecar_blob = _write_inputs(tmp_path)
    hngs_decoder.write_bytes(b"BAD" + b"\x00generated-schema-decoder")
    output = tmp_path / "candidate.zip"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(tool.HNeRVGeneratedSchemaPacketError, match="HNGS magic"):
        tool.build_hnerv_generated_schema_candidate_archive(
            hngs_decoder=hngs_decoder,
            latent_blob=latent_blob,
            sidecar_blob=sidecar_blob,
            output_archive=output,
            manifest_output=manifest,
            candidate_id="unit_bad_hngs",
        )

    assert not output.exists()
    assert not manifest.exists()
