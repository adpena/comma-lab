from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli
import pytest

from tac.hnerv_lowlevel_packer import (
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_wavelet_apply_transform import (
    RUNTIME_DECODE_VALIDATION_FILENAME,
    RUNTIME_DECODE_VALIDATION_SCHEMA,
    HnervWaveletApplyTransformError,
    apply_wr01_atoms_to_raw,
    build_wavelet_apply_transform_candidate,
)
from tac.hnerv_wavelet_sidechannel import (
    build_wavelet_sidechannel_archive_bytes,
    encode_wavelet_atom_sidechannel,
)

REPO = Path(__file__).resolve().parents[3]


def test_apply_wr01_atoms_to_raw_skips_malformed_mapping_atoms() -> None:
    """Round 7 R7-2 fix (2026-05-06, 88%): a Mapping atom missing required
    keys (raw_offset / raw_end / coefficient_quantized) used to crash the
    whole apply pass with `TypeError: int() argument must be a string ...,
    not 'NoneType'`. R7-2 wraps the int() coercion in try/except so the
    malformed atom is counted as skipped and the apply continues.
    """
    raw = bytes([10, 20, 30, 40])
    section = {
        "atoms": [
            # Valid atom — should apply.
            {
                "raw_offset": 0,
                "raw_end": 2,
                "level": 0,
                "coefficient_index": 0,
                "coefficient_quantized": -5,
            },
            # Malformed Mapping atom — missing raw_end. Pre-R7-2 this crashed.
            {
                "raw_offset": 2,
                "level": 0,
                "coefficient_index": 1,
                "coefficient_quantized": -3,
            },
            # Malformed Mapping atom — non-int raw_offset. ValueError path.
            {
                "raw_offset": "not-an-int",
                "raw_end": 4,
                "level": 0,
                "coefficient_index": 2,
                "coefficient_quantized": -1,
            },
        ]
    }

    transformed, stats = apply_wr01_atoms_to_raw(
        raw,
        section,
        strength_numerator=1,
        strength_denominator=1,
    )

    # Valid atom applied; two malformed atoms skipped without crashing.
    assert stats["applied_atom_count"] == 1
    assert stats["skipped_atom_count"] == 2
    assert transformed == bytes([15, 15, 30, 40])


def test_apply_wr01_atoms_to_raw_attenuates_detail() -> None:
    raw = bytes([10, 20, 30, 40])
    section = {
        "atoms": [
            {
                "raw_offset": 0,
                "raw_end": 2,
                "level": 0,
                "coefficient_index": 0,
                "coefficient_quantized": -5,
            }
        ]
    }

    transformed, stats = apply_wr01_atoms_to_raw(
        raw,
        section,
        strength_numerator=1,
        strength_denominator=1,
    )

    assert transformed == bytes([15, 15, 30, 40])
    assert stats["applied_atom_count"] == 1
    assert stats["changed_raw_positions"] == 2


def test_build_wavelet_apply_transform_candidate_fails_fast_on_invalid_gate() -> None:
    with pytest.raises(HnervWaveletApplyTransformError, match="strength_numerator"):
        build_wavelet_apply_transform_candidate(
            wavelet_archive="/not-read/wavelet.zip",
            output_dir="/not-written/out",
            source_label="fixture",
            strength_numerator=0,
        )

    with pytest.raises(HnervWaveletApplyTransformError, match="must not be under"):
        build_wavelet_apply_transform_candidate(
            wavelet_archive="/tmp/wavelet.zip",
            output_dir="/not-written/out",
            source_label="fixture",
        )


def test_build_hnerv_wavelet_apply_transform_candidate_cli(tmp_path: Path) -> None:
    raw_latents = bytes((idx * 7) % 256 for idx in range(96))
    source_payload = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=brotli.compress(b"decoder", quality=11),
        latents_and_sidecar_brotli=brotli.compress(raw_latents, quality=11),
    ).to_bytes()
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(source_archive, member_name="0.bin", payload=source_payload)
    source_zip = read_strict_single_member_zip(source_archive)
    sidechannel = encode_wavelet_atom_sidechannel(
        {
            "sections": [
                {
                    "section_name": "latents_and_sidecar_brotli",
                    "source_section_sha256": sha256_bytes(parse_ff_packed_brotli_hnerv(source_payload).latents_and_sidecar_brotli),
                    "raw_bytes": len(raw_latents),
                    "atoms": [
                        {
                            "raw_offset": 0,
                            "raw_end": 2,
                            "level": 0,
                            "coefficient_index": 0,
                            "coefficient_quantized": -4,
                        }
                    ],
                }
            ]
        }
    )
    wavelet_payload = build_wavelet_sidechannel_archive_bytes(
        source_payload=source_payload,
        sidechannel_blob=sidechannel,
    )
    wavelet_archive = tmp_path / "wavelet.zip"
    write_stored_single_member_zip(wavelet_archive, member_name="0.bin", payload=wavelet_payload)
    out_dir = tmp_path / "out"
    manifest_path = tmp_path / "manifest.json"

    # Round 5 R5-3 fix (2026-05-06): the candidate is correctly NOT ready
    # for archive preflight because `requires_archive_manifest_preflight`
    # remains in dispatch_blockers (no caller has wired the upstream
    # archive-manifest preflight call yet). Pre-R5-3 the manifest reported
    # `ready_for_archive_preflight=True` while ALSO carrying the unwired
    # blocker — split-brain. The test now asserts the corrected, honest
    # state and DOES NOT pass --fail-if-not-archive-preflight-ready (which
    # would now correctly exit 2 since the manifest is unready).
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_wavelet_apply_transform_candidate.py"),
            "--wavelet-archive",
            str(wavelet_archive),
            "--output-dir",
            str(out_dir),
            "--source-label",
            "fixture",
            "--source-archive",
            str(source_archive),
            "--strength-numerator",
            "1",
            "--strength-denominator",
            "1",
            "--json-out",
            str(manifest_path),
        ],
        check=True,
        text=True,
    )

    manifest = json.loads(manifest_path.read_text())
    builder_manifest_path = out_dir / "hnerv_wavelet_apply_transform_candidate.json"
    runtime_decode_validation_path = out_dir / RUNTIME_DECODE_VALIDATION_FILENAME
    runtime_decode_validation = json.loads(runtime_decode_validation_path.read_text())
    assert json.loads(builder_manifest_path.read_text()) == manifest
    assert runtime_decode_validation == manifest["runtime_decode_validation"]
    assert manifest["score_claim"] is False
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["source_archive_sha256"] == source_zip.archive_sha256
    assert manifest["source_archive_bytes"] == source_zip.archive_bytes
    assert manifest["source_payload_sha256"] == sha256_bytes(source_payload)
    assert manifest["source_archive_custody_mode"] == "verified_source_archive_payload_match"
    assert manifest["changed_section_name"] == "latents_and_sidecar_brotli"
    assert manifest["changed_section_sha256"] == manifest["candidate_section_sha256"]
    assert any(
        "archive_manifest_preflight" in str(blocker).lower()
        for blocker in manifest.get("dispatch_blockers", [])
    ), (
        "expected dispatch_blockers to record the unwired archive-manifest "
        f"preflight; got {manifest.get('dispatch_blockers')}"
    )
    assert manifest["transform_stats"]["applied_atom_count"] == 1
    assert manifest["runtime_apply"]["ready_for_runtime_apply_review"] is True
    assert manifest["runtime_apply"]["applied_atom_count"] == 1
    assert manifest["runtime_apply"]["applied_atom_ids"] == manifest["runtime_apply_atom_ids"]
    assert len(manifest["runtime_apply_atom_ids"]) == 1
    assert manifest["runtime_apply_atom_ids"][0].startswith("wr01-latents-and-sidecar-brotli-")
    assert manifest["runtime_decode_validation_schema"] == RUNTIME_DECODE_VALIDATION_SCHEMA
    assert manifest["runtime_decode_validation_manifest_path"] == str(runtime_decode_validation_path)
    assert manifest["runtime_decode_validation_manifest_sha256"] == runtime_decode_validation[
        "manifest_sha256_excluding_self"
    ]
    assert runtime_decode_validation["schema"] == RUNTIME_DECODE_VALIDATION_SCHEMA
    assert runtime_decode_validation["validation_mode"] == "local_wr01_runtime_decode_validation_not_score"
    assert runtime_decode_validation["ready_for_runtime_decode_review"] is True
    assert runtime_decode_validation["ready_for_archive_preflight"] is False
    assert runtime_decode_validation["ready_for_exact_eval_dispatch"] is False
    assert runtime_decode_validation["exact_cuda_auth_eval"] is False
    assert runtime_decode_validation["score_claim"] is False
    assert runtime_decode_validation["blockers"] == []
    assert runtime_decode_validation["changed_section_names"] == ["latents_and_sidecar_brotli"]
    assert runtime_decode_validation["changed_section_only"] is True
    changed_validation_section = next(
        section
        for section in runtime_decode_validation["sections"]
        if section["section_name"] == "latents_and_sidecar_brotli"
    )
    assert changed_validation_section["section_changed"] is True
    assert changed_validation_section["candidate_raw_sha256"] == manifest["candidate_raw_sha256"]
    candidate = read_strict_single_member_zip(manifest["candidate_archive_path"])
    assert candidate.payload[0] == 0xFF
    candidate_latents = brotli.decompress(
        parse_ff_packed_brotli_hnerv(candidate.payload).latents_and_sidecar_brotli
    )
    assert candidate_latents != raw_latents
