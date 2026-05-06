from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_lowlevel_packer import (
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_wavelet_apply_transform import apply_wr01_atoms_to_raw
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


def test_build_hnerv_wavelet_apply_transform_candidate_cli(tmp_path: Path) -> None:
    raw_latents = bytes((idx * 7) % 256 for idx in range(96))
    source_payload = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=brotli.compress(b"decoder", quality=11),
        latents_and_sidecar_brotli=brotli.compress(raw_latents, quality=11),
    ).to_bytes()
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
    assert manifest["score_claim"] is False
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert any(
        "archive_manifest_preflight" in str(blocker).lower()
        for blocker in manifest.get("dispatch_blockers", [])
    ), (
        "expected dispatch_blockers to record the unwired archive-manifest "
        f"preflight; got {manifest.get('dispatch_blockers')}"
    )
    assert manifest["transform_stats"]["applied_atom_count"] == 1
    candidate = read_strict_single_member_zip(manifest["candidate_archive_path"])
    assert candidate.payload[0] == 0xFF
    candidate_latents = brotli.decompress(
        parse_ff_packed_brotli_hnerv(candidate.payload).latents_and_sidecar_brotli
    )
    assert candidate_latents != raw_latents
