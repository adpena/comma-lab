# SPDX-License-Identifier: MIT
from __future__ import annotations

import struct


def _member() -> bytes:
    source = b"d" * 10 + b"l" * 4 + b"s" * 3
    selector = b"FEC6" + struct.pack("<H", 600) + b"\x00"
    return b"FP11" + struct.pack("<I", len(source)) + source + struct.pack("<H", len(selector)) + selector


def test_parse_fec6_sections_maps_inner_pr101_boundaries() -> None:
    from tac.optimization.fec6_byte_targets import parse_fec6_sections

    sections = {section.name: section for section in parse_fec6_sections(
        _member(),
        decoder_blob_len=10,
        latent_blob_len=4,
    )}

    assert sections["decoder"].byte_range.as_dict() == {"start": 8, "end": 18, "length": 10}
    assert sections["latent"].byte_range.as_dict() == {"start": 18, "end": 22, "length": 4}
    assert sections["sidecar"].byte_range.as_dict() == {"start": 22, "end": 25, "length": 3}
    assert sections["selector_payload"].byte_range.start == 27


def test_summarize_tensor_shortlist_groups_top_records() -> None:
    from tac.optimization.fec6_byte_targets import (
        ByteRange,
        DecoderTensorRange,
        summarize_tensor_shortlist,
    )

    tensor = DecoderTensorRange(
        name="layer.weight",
        storage_index=1,
        shape=(2,),
        numel=2,
        byte_map="zig",
        fp16_scale=0.5,
        decoded_mantissa_range=ByteRange(0, 2),
        decoded_scale_range=ByteRange(2, 4),
        compressed_range=ByteRange(10, 20),
    )
    records = [
        {
            "byte_index": 11,
            "score_impact_abs_sum": 3.0,
            "axis_score_impact": {"seg": 2.0, "pose": 1.0, "rate": 0.0},
        }
    ]

    rows = summarize_tensor_shortlist(records, [tensor])

    assert rows[0]["tensor_name"] == "layer.weight"
    assert rows[0]["top_byte_indices"] == [11]
    assert rows[0]["axis_score_impact_abs_sum"]["seg"] == 2.0
