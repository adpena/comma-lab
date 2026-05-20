from __future__ import annotations

import numpy as np

from tac.optimization.fec6_decoder_mutations import (
    DecoderQMutation,
    apply_q_mutation,
    apply_q_mutations,
    build_mutation_grid,
    prepare_decoder_blob,
    probe_q_mutation,
    recompress_prepared_decoder,
    split_brotli_streams,
)
from tac.pr101_split_brotli_codec import (
    DECODER_BYTE_MAPS,
    DECODER_STORAGE_ORDER,
    DECODER_STREAM_ENDS,
    FIXED_STATE_SCHEMA,
    _encode_mapped_u8,
    pack_brotli_stream,
)


def _synthetic_decoder_blob(*, quality: int = 1) -> bytes:
    parts = []
    for storage_index in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[int(storage_index)]
        _ = name
        numel = int(np.prod(shape))
        q = np.zeros(numel, dtype=np.int8)
        mapped = _encode_mapped_u8(q, DECODER_BYTE_MAPS.get(int(storage_index), "zig"))
        parts.append(mapped.tobytes() + np.array([1.0], dtype=np.float16).tobytes())

    streams = []
    start = 0
    for end in DECODER_STREAM_ENDS:
        streams.append(pack_brotli_stream(b"".join(parts[start:end]), quality=quality))
        start = int(end)
    return b"".join(streams)


def test_split_brotli_streams_preserves_stream_spans() -> None:
    first = pack_brotli_stream(b"aaa", quality=1)
    second = pack_brotli_stream(b"bbbccc", quality=1)

    raw, ranges = split_brotli_streams(first + second, 2)

    assert raw == (b"aaa", b"bbbccc")
    assert ranges[0].as_dict() == {"start": 0, "end": len(first), "length": len(first)}
    assert ranges[1].as_dict() == {
        "start": len(first),
        "end": len(first) + len(second),
        "length": len(second),
    }


def test_prepare_and_identity_recompress_round_trip() -> None:
    blob = _synthetic_decoder_blob(quality=1)
    prepared = prepare_decoder_blob(blob)

    reencoded = recompress_prepared_decoder(prepared, prepared.raw, brotli_quality=1)

    assert reencoded == blob
    assert prepared.tensor_by_name()["rgb_1.weight"].storage_index == 26
    assert prepared.tensor_by_name()["rgb_1.bias"].byte_map == "off"


def test_q_mutation_is_legal_and_reports_fixed_length_status() -> None:
    blob = _synthetic_decoder_blob(quality=1)
    prepared = prepare_decoder_blob(blob)

    raw, tensor, q_before, q_after = apply_q_mutation(
        prepared,
        DecoderQMutation(tensor_name="rgb_1.bias", q_offset=1, delta=1),
    )
    result = probe_q_mutation(
        prepared,
        DecoderQMutation(tensor_name="rgb_1.bias", q_offset=1, delta=1),
        brotli_quality=1,
    )

    assert raw != prepared.raw
    assert tensor.name == "rgb_1.bias"
    assert (q_before, q_after) == (0, 1)
    assert result.mutation_id
    assert result.source_decoder_len == len(blob)


def test_build_mutation_grid_is_deterministic_and_bounded() -> None:
    blob = _synthetic_decoder_blob(quality=1)
    prepared = prepare_decoder_blob(blob)
    grid = build_mutation_grid(
        prepared.tensor_by_name(),
        ["rgb_1.weight"],
        deltas=(-1, 1),
        max_offsets_per_tensor=3,
    )

    assert [row.as_dict() for row in grid] == [
        {"tensor_name": "rgb_1.weight", "q_offset": 0, "delta": -1},
        {"tensor_name": "rgb_1.weight", "q_offset": 0, "delta": 1},
        {"tensor_name": "rgb_1.weight", "q_offset": 242, "delta": -1},
        {"tensor_name": "rgb_1.weight", "q_offset": 242, "delta": 1},
        {"tensor_name": "rgb_1.weight", "q_offset": 485, "delta": -1},
        {"tensor_name": "rgb_1.weight", "q_offset": 485, "delta": 1},
    ]


def test_apply_q_mutations_is_cumulative_and_rejects_duplicate_targets() -> None:
    blob = _synthetic_decoder_blob(quality=1)
    prepared = prepare_decoder_blob(blob)

    raw, records = apply_q_mutations(
        prepared,
        [
            DecoderQMutation(tensor_name="rgb_1.weight", q_offset=0, delta=1),
            DecoderQMutation(tensor_name="rgb_1.bias", q_offset=1, delta=-1),
        ],
    )

    assert raw != prepared.raw
    assert [record["q_after"] for record in records] == [1, -1]

    import pytest

    with pytest.raises(Exception, match="duplicate q mutation target"):
        apply_q_mutations(
            prepared,
            [
                DecoderQMutation(tensor_name="rgb_1.weight", q_offset=0, delta=1),
                DecoderQMutation(tensor_name="rgb_1.weight", q_offset=0, delta=-1),
            ],
        )
