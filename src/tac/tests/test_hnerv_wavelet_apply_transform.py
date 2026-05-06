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
            "--fail-if-not-archive-preflight-ready",
        ],
        check=True,
        text=True,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["score_claim"] is False
    assert manifest["ready_for_archive_preflight"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["transform_stats"]["applied_atom_count"] == 1
    candidate = read_strict_single_member_zip(manifest["candidate_archive_path"])
    assert candidate.payload[0] == 0xFF
    candidate_latents = brotli.decompress(
        parse_ff_packed_brotli_hnerv(candidate.payload).latents_and_sidecar_brotli
    )
    assert candidate_latents != raw_latents
