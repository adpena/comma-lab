# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from tac.analysis.snerv_step_map_coder import (
    ADAPTIVE_BINARY_MAGIC,
    ADAPTIVE_MAGIC,
    decode_step_maps,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    pack_snerv_archive_snar2,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.receiver_proof import (
    build_snerv_receiver_archive_proof,
)
from tools import materialize_snerv_step_map_compaction as cli


def test_materialize_snerv_step_map_compaction_cli_rewrites_verbose_constant_group(
    tmp_path: Path,
) -> None:
    _proof, archive = build_snerv_receiver_archive_proof(
        bins=4,
        levels=1,
        wavelet="haar",
        hw=(12, 16),
        full_frame_packet=True,
    )
    decoded = unpack_snerv_archive(archive.packet)
    source_step_maps = [
        np.full((3, 5), 2.0 ** (index * 0.125), dtype=np.float32)
        for index in range(12)
    ]
    verbose_step_packet = _verbose_constant_step_packet(source_step_maps)
    source = pack_snerv_archive_snar2(
        metadata_payload=decoded.sections["metadata_payload"],
        lf_payload=decoded.sections["lf_payload"],
        decoder_payload=decoded.sections["decoder_payload"],
        step_map_packet=verbose_step_packet,
        metadata=decoded.metadata,
    )
    source_path = tmp_path / "source.snar2"
    source_path.write_bytes(source.packet)
    output_packet = tmp_path / "candidate.snar2"
    output_json = tmp_path / "report.json"

    assert cli.main(
        [
            "--packet",
            str(source_path),
            "--candidate-id",
            "snerv_stepmap_unit_candidate",
            "--wire-format",
            "snar2",
            "--output-packet",
            str(output_packet),
            "--output-json",
            str(output_json),
            "--hard-byte-ceiling",
            str(len(source.packet) - 1),
        ]
    ) == 0

    report = json.loads(output_json.read_text(encoding="utf-8"))
    candidate_packet = output_packet.read_bytes()
    candidate = unpack_snerv_archive(candidate_packet)
    assert report["schema"] == cli.SCHEMA
    assert report["wire_format"] == "snar2"
    assert report["candidate_binding"]["candidate_id"] == (
        "snerv_stepmap_unit_candidate"
    )
    assert report["decoded_step_maps_exact_equal"] is True
    assert report["packet_byte_delta"] < 0
    assert report["section_bytes"]["step_map_packet_delta"] < 0
    ceiling_row = report["hard_byte_ceiling_rows"][0]
    assert ceiling_row["candidate_packet_under_ceiling"] is True
    assert ceiling_row["packet_under_ceiling_is_byte_authority"] is False
    assert ceiling_row["candidate_archive_zip_under_ceiling"] is None
    assert ceiling_row["byte_authority"] == "packet_preview_only_no_receiver_proof"
    assert ceiling_row["hard_byte_ceiling_satisfied_for_long_training"] is False
    assert (
        "snerv_step_map_compaction_receiver_proven_archive_zip_missing"
        in ceiling_row["blockers"]
    )
    assert report["runtime_consumption_proof_passed"] is False
    assert "snerv_step_map_compaction_receiver_proof_missing" in report["blockers"]
    assert (
        "snerv_step_map_compaction_packet_preview_not_byte_authority"
        in report["blockers"]
    )
    assert candidate_packet.startswith(b"SNAR2")
    assert candidate.sections["step_map_packet"] != verbose_step_packet
    assert candidate.sections["step_map_packet"].startswith(ADAPTIVE_BINARY_MAGIC)
    assert len(candidate.sections["step_map_packet"]) < len(verbose_step_packet)
    for got, ref in zip(candidate.decode_step_maps(), source_step_maps, strict=True):
        np.testing.assert_array_equal(got, ref)


def _verbose_constant_step_packet(step_maps: list[np.ndarray]) -> bytes:
    header = {
        "schema": "snerv_step_map_coder.adaptive.v1",
        "map_count": len(step_maps),
        "groups": [
            {
                "kind": "constant_log2_fill",
                "precision_label": "constant",
                "bins": 0,
                "bits_per_code": 0,
                "code_storage": "run_length_constant_log2_f32",
                "map_indices": list(range(len(step_maps))),
                "payload_offset": 0,
                "payload_bytes": 0,
                "packed_code_bytes": 0,
                "log2_values": [
                    float(np.mean(np.log2(array.astype(np.float64))))
                    for array in step_maps
                ],
                "shapes": [list(array.shape) for array in step_maps],
                "code_count": 0,
            }
        ],
    }
    header_bytes = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    packet = ADAPTIVE_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes
    decoded = decode_step_maps(packet)
    for got, ref in zip(decoded, step_maps, strict=True):
        np.testing.assert_array_equal(got, ref)
    return packet
