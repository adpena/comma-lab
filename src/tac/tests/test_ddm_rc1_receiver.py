"""Behavior tests for the PR130 three-stream + ANS strict-superset receiver."""

from __future__ import annotations

import hashlib
import importlib
import lzma
import struct
import sys
import time
import zipfile
from pathlib import Path

import brotli
import constriction
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / "src" / "tac" / "pr130_runtime" / "fx1_runtime_tree"
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))
receiver = importlib.import_module("receiver")

LEGACY_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/"
    "reproduction/archive.zip"
)
SPLIT_BROTLI_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/"
    "splitpack/archive_brotli_q11.zip"
)
SPLIT_LZMA2_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/"
    "splitpack/archive_lzma2_free.zip"
)
TAGGED_BROTLI_ARCHIVE = SPLIT_BROTLI_ARCHIVE.with_name(
    "archive_brotli_q11_tagged_range.zip"
)
TAGGED_LZMA2_ARCHIVE = SPLIT_LZMA2_ARCHIVE.with_name(
    "archive_lzma2_free_tagged_range.zip"
)
REAL_GT_CACHE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/"
    "caches/gt_cache_600_official_ada.pt"
)
EXPECTED_MODELS_RAW_SHA256 = "62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517"


def _raw_sections() -> tuple[bytes, bytes, bytes, bytes]:
    semantic = bytes(range(97)) * 3
    carrier = bytes(reversed(range(113))) * 2
    hpac = b"integer-hpac-state" * 19
    raw = struct.pack("<II", len(semantic), len(carrier)) + semantic + carrier + hpac
    return semantic, carrier, hpac, raw


def _split_pack(streams: tuple[bytes, bytes, bytes]) -> bytes:
    return struct.pack("<III", *(len(stream) for stream in streams)) + b"".join(streams)


def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    symbols_a = np.array([1, 4, 0], dtype=np.int32)
    symbols_b = np.array([3, 2, 1], dtype=np.int32)
    tables_a = np.array(
        [
            [0.08, 0.44, 0.12, 0.20, 0.16],
            [0.11, 0.09, 0.13, 0.17, 0.50],
            [0.52, 0.08, 0.14, 0.15, 0.11],
        ],
        dtype=np.float32,
    )
    tables_b = np.array(
        [
            [0.07, 0.10, 0.16, 0.51, 0.16],
            [0.13, 0.15, 0.48, 0.09, 0.15],
            [0.09, 0.49, 0.11, 0.18, 0.13],
        ],
        dtype=np.float32,
    )
    return symbols_a, tables_a, symbols_b, tables_b


def test_legacy_and_both_split_model_forms_reconstruct_identical_loader_bytes() -> None:
    semantic, carrier, hpac, expected = _raw_sections()
    legacy = lzma.compress(expected, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    split_brotli = _split_pack(tuple(
        brotli.compress(raw, quality=11) for raw in (semantic, carrier, hpac)
    ))
    split_lzma2 = _split_pack(tuple(
        lzma.compress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=[{
                "id": lzma.FILTER_LZMA2,
                "preset": 9 | lzma.PRESET_EXTREME,
            }],
        )
        for raw in (semantic, carrier, hpac)
    ))

    decoded = [
        receiver.decode_models(legacy, model_codec="legacy_lzma"),
        receiver.decode_models(split_brotli, model_codec="split_brotli"),
        receiver.decode_models(split_lzma2, model_codec="split_lzma2"),
    ]
    assert [item.codec for item in decoded] == [
        "legacy_lzma",
        "split_brotli",
        "split_lzma2",
    ]
    assert all(item.raw == expected for item in decoded)


def test_model_codec_selector_fails_closed_instead_of_guessing() -> None:
    semantic, carrier, hpac, _ = _raw_sections()
    split_brotli = _split_pack(tuple(
        brotli.compress(raw, quality=11) for raw in (semantic, carrier, hpac)
    ))
    with pytest.raises(receiver.ReceiverFormatError, match="legacy model flag"):
        receiver.decode_models(split_brotli, model_codec="legacy_lzma")
    with pytest.raises(receiver.ReceiverFormatError, match="raw-LZMA2"):
        receiver.decode_models(split_brotli, model_codec="split_lzma2")


def test_brotli_format_has_a_named_dependency_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    semantic, carrier, hpac, _ = _raw_sections()
    split_brotli = _split_pack(tuple(
        brotli.compress(raw, quality=11) for raw in (semantic, carrier, hpac)
    ))
    monkeypatch.setattr(receiver, "brotli", None)
    with pytest.raises(receiver.BrotliDependencyError, match=r"Brotli==1\.2\.0"):
        receiver.decode_models(split_brotli, model_codec="split_brotli")


def test_zero_byte_codec_tags_are_backward_compatible_and_explicit() -> None:
    models = b"models"
    tokens = b"\x01\x02\x03\x04"
    legacy_payload = receiver.pack_payload(
        models, tokens, token_codec="range", model_codec="legacy_lzma"
    )
    brotli_payload = receiver.pack_payload(
        models, tokens, token_codec="range", model_codec="split_brotli"
    )
    ans_lzma2_payload = receiver.pack_payload(
        models, tokens, token_codec="ans", model_codec="split_lzma2"
    )

    assert len(legacy_payload) == len(brotli_payload) == len(ans_lzma2_payload)
    assert struct.unpack_from("<I", legacy_payload)[0] == len(models)
    assert struct.unpack_from("<I", brotli_payload)[0] == (
        receiver.SPLIT_BROTLI_FLAG | len(models)
    )
    assert struct.unpack_from("<I", ans_lzma2_payload)[0] == (
        receiver.ANS_TOKEN_FLAG | receiver.SPLIT_LZMA2_FLAG | len(models)
    )
    assert receiver.split_payload(legacy_payload).model_codec == "legacy_lzma"
    assert receiver.split_payload(brotli_payload).model_codec == "split_brotli"
    assert receiver.split_payload(ans_lzma2_payload).model_codec == "split_lzma2"
    assert receiver.split_payload(legacy_payload).token_codec == "range"
    assert receiver.split_payload(ans_lzma2_payload).token_codec == "ans"
    assert receiver.split_payload(brotli_payload).models == models
    assert receiver.split_payload(ans_lzma2_payload).tokens == tokens

    reserved_word = receiver.MODEL_CODEC_MASK | len(models)
    with pytest.raises(receiver.ReceiverFormatError, match="reserved model-codec"):
        receiver.split_payload(struct.pack("<I", reserved_word) + models + tokens)


def test_truncated_corrupt_and_trailing_fields_fail_closed() -> None:
    with pytest.raises(receiver.ReceiverFormatError, match="truncated"):
        receiver.split_payload(b"\x00\x00\x00")
    with pytest.raises(receiver.ReceiverFormatError, match="empty model"):
        receiver.split_payload(struct.pack("<I", 0) + b"tokens")
    with pytest.raises(receiver.ReceiverFormatError, match="no complete token"):
        receiver.split_payload(struct.pack("<I", 1) + b"m")

    semantic, carrier, hpac, _ = _raw_sections()
    framed = _split_pack(tuple(
        brotli.compress(raw, quality=11) for raw in (semantic, carrier, hpac)
    ))
    with pytest.raises(receiver.ReceiverFormatError, match="exact three-stream"):
        receiver.decode_models(
            framed + b"trailing",
            model_codec="split_brotli",
        )

    legacy = lzma.compress(
        struct.pack("<II", len(semantic), len(carrier)) + semantic + carrier + hpac,
        format=lzma.FORMAT_XZ,
    )
    with pytest.raises(receiver.ReceiverFormatError, match="trailing bytes"):
        receiver.decode_models(legacy + legacy, model_codec="legacy_lzma")

    raw_lzma2_streams = tuple(
        lzma.compress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=[{
                "id": lzma.FILTER_LZMA2,
                "preset": 9 | lzma.PRESET_EXTREME,
            }],
        )
        for raw in (semantic, carrier, hpac)
    )
    with pytest.raises(receiver.ReceiverFormatError, match="trailing bytes"):
        receiver.decode_models(
            _split_pack((
                raw_lzma2_streams[0] + b"junk",
                raw_lzma2_streams[1],
                raw_lzma2_streams[2],
            )),
            model_codec="split_lzma2",
        )
    with pytest.raises(receiver.ReceiverFormatError, match="truncated"):
        receiver.decode_models(
            _split_pack((
                raw_lzma2_streams[0][:-1],
                raw_lzma2_streams[1],
                raw_lzma2_streams[2],
            )),
            model_codec="split_lzma2",
        )

    symbols_a, tables_a, symbols_b, tables_b = _tables()
    ans_blob = receiver.encode_ans_blocks_reverse([
        (symbols_a, tables_a),
        (symbols_b, tables_b),
    ])
    with pytest.raises(receiver.ReceiverFormatError, match="multiple of four"):
        receiver.new_token_decoder(ans_blob[:-1], "ans")
    decoder = receiver.new_token_decoder(ans_blob + struct.pack("<I", 1), "ans")
    family = constriction.stream.model.Categorical(perfect=False)
    decoder.decode(family, tables_a)
    decoder.decode(family, tables_b)
    with pytest.raises(receiver.ReceiverFormatError, match="retained state"):
        receiver.finish_token_decode(decoder, "ans")


def test_range_and_ans_decoders_recover_the_same_forward_blocks() -> None:
    symbols_a, tables_a, symbols_b, tables_b = _tables()
    family = constriction.stream.model.Categorical(perfect=False)

    range_encoder = constriction.stream.queue.RangeEncoder()
    range_encoder.encode(symbols_a, family, tables_a)
    range_encoder.encode(symbols_b, family, tables_b)
    range_blob = range_encoder.get_compressed().astype("<u4", copy=False).tobytes()

    ans_blob = receiver.encode_ans_blocks_reverse([
        (symbols_a, tables_a),
        (symbols_b, tables_b),
    ])
    expected = np.concatenate([symbols_a, symbols_b])
    for codec, blob in (("range", range_blob), ("ans", ans_blob)):
        decoder = receiver.new_token_decoder(blob, codec)
        actual = np.concatenate([
            decoder.decode(family, tables_a),
            decoder.decode(family, tables_b),
        ])
        receiver.finish_token_decode(decoder, codec)
        np.testing.assert_array_equal(actual, expected)


def test_ans_chunked_reverse_pass_is_byte_identical_to_uninterrupted(
    tmp_path: Path,
) -> None:
    symbols_a, tables_a, symbols_b, tables_b = _tables()
    chunks = []
    for index, (symbols, tables) in enumerate(
        ((symbols_a, tables_a), (symbols_b, tables_b))
    ):
        symbols_path = tmp_path / f"symbols_{index}.npy"
        tables_path = tmp_path / f"tables_{index}.npy"
        np.save(symbols_path, symbols, allow_pickle=False)
        np.save(tables_path, tables, allow_pickle=False)
        chunks.append(receiver.AnsChunk(symbols_path, tables_path))

    expected = receiver.encode_ans_blocks_reverse([
        (symbols_a, tables_a),
        (symbols_b, tables_b),
    ])
    assert receiver.encode_ans_chunks_reverse(chunks) == expected


def test_forward_ans_chunk_calls_pop_in_the_wrong_order() -> None:
    symbols_a, tables_a, symbols_b, tables_b = _tables()
    family = constriction.stream.model.Categorical(perfect=False)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols_a, family, tables_a)
    coder.encode_reverse(symbols_b, family, tables_b)
    decoder = constriction.stream.stack.AnsCoder(coder.get_compressed())
    first = decoder.decode(family, tables_b)
    second = decoder.decode(family, tables_a)
    np.testing.assert_array_equal(first, symbols_b)
    np.testing.assert_array_equal(second, symbols_a)
    assert decoder.is_empty()


def test_pinned_ans_fixed_vector_and_int16_code_rehydration_are_byte_identical(
    tmp_path: Path,
) -> None:
    symbols = np.array([1, 4, 0, 3, 2, 1], dtype=np.int32)
    tables = np.array(
        [
            [0.05, 0.60, 0.10, 0.15, 0.10],
            [0.10, 0.10, 0.10, 0.10, 0.60],
            [0.60, 0.10, 0.10, 0.10, 0.10],
            [0.05, 0.10, 0.10, 0.65, 0.10],
            [0.10, 0.10, 0.60, 0.10, 0.10],
            [0.10, 0.55, 0.10, 0.15, 0.10],
        ],
        dtype=np.float32,
    )
    direct = receiver.encode_ans_blocks_reverse([(symbols, tables)])
    assert direct.hex() == "6c666601"
    assert hashlib.sha256(direct).hexdigest() == (
        "607631237db3862296ce51b2efe95792a98f4ca4a673966769846394681e7adb"
    )

    # Separately verify the PR130 int16 spill path against an independent
    # transcription of probability_table's float64 softmax construction.
    codes = np.array(
        [
            [1, 17, -3, 5, 2],
            [-7, 0, 9, 3, -2],
            [12, 2, -4, 6, 1],
            [0, 3, -6, 15, 2],
            [-2, 5, 11, -4, 1],
            [3, 14, -1, 6, 0],
        ],
        dtype=np.int16,
    )
    rehydrated = receiver.probability_tables_from_codes(codes, logit_precision=8)
    reference = codes.astype(np.float64) / 8
    reference -= reference.max(axis=1, keepdims=True)
    reference = np.exp(reference)
    reference /= reference.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(rehydrated, reference.astype(np.float32))
    direct_rehydrated = receiver.encode_ans_blocks_reverse([(symbols, rehydrated)])
    symbols_path = tmp_path / "symbols.npy"
    codes_path = tmp_path / "codes.npy"
    np.save(symbols_path, symbols.astype(np.uint8), allow_pickle=False)
    np.save(codes_path, codes, allow_pickle=False)
    chunked = receiver.encode_ans_code_chunks_reverse([
        receiver.AnsCodeChunk(symbols_path, codes_path, 8)
    ])
    assert chunked == direct_rehydrated


def _read_p(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read("p")


@pytest.mark.skipif(
    not all(path.is_file() for path in (
        LEGACY_ARCHIVE,
        SPLIT_BROTLI_ARCHIVE,
        SPLIT_LZMA2_ARCHIVE,
    )),
    reason="real PR130 archive custody is not mounted",
)
def test_real_archives_reconstruct_the_same_models_and_feed_real_loaders() -> None:
    variants = []
    sources = (
        (LEGACY_ARCHIVE, "legacy_lzma"),
        (SPLIT_BROTLI_ARCHIVE, "split_brotli"),
        (SPLIT_LZMA2_ARCHIVE, "split_lzma2"),
    )
    for path, model_codec in sources:
        source_payload = _read_p(path)
        source_parts = receiver.split_payload(source_payload)
        payload = (
            source_payload
            if model_codec == "legacy_lzma"
            else receiver.pack_payload(
                source_parts.models,
                source_parts.tokens,
                token_codec="range",
                model_codec=model_codec,
            )
        )
        assert len(payload) == len(source_payload)
        parts = receiver.split_payload(payload)
        decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
        variants.append(decoded)
        assert parts.token_codec == "range"
        assert parts.model_codec == model_codec
        assert hashlib.sha256(decoded.raw).hexdigest() == EXPECTED_MODELS_RAW_SHA256
    assert [item.codec for item in variants] == [
        "legacy_lzma",
        "split_brotli",
        "split_lzma2",
    ]
    assert variants[0].raw == variants[1].raw == variants[2].raw

    inflate = importlib.import_module("inflate")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", variants[1].raw)
    semantic_pose_end = 8 + semantic_bytes + carrier_bytes
    semantic, basis, coefficients = inflate.unpack_semantic_pose(
        variants[1].raw[:semantic_pose_end]
    )
    hpac = inflate.load_hpac(variants[1].raw[semantic_pose_end:], torch.device("cpu"))
    assert semantic is not None
    assert basis.shape == (inflate.CARRIER_DIM, 3, inflate.CARRIER_H, inflate.CARRIER_W)
    assert coefficients.shape == (inflate.N, inflate.CARRIER_DIM)
    assert hpac.frame_embed.weight.shape[0] == inflate.N


@pytest.mark.skipif(
    not (TAGGED_BROTLI_ARCHIVE.is_file() and TAGGED_LZMA2_ARCHIVE.is_file()),
    reason="durable tagged PR130 archive custody is not mounted",
)
def test_durable_tagged_range_archives_are_size_neutral_and_receiver_closed() -> None:
    variants = (
        (
            SPLIT_BROTLI_ARCHIVE,
            TAGGED_BROTLI_ARCHIVE,
            "split_brotli",
            "4c9751582937e48e22be8336dbf36cbe229207e65875fe2196694032b40aa891",
        ),
        (
            SPLIT_LZMA2_ARCHIVE,
            TAGGED_LZMA2_ARCHIVE,
            "split_lzma2",
            "622cc7d8eb512d728b9e579a5d9cca73eccab3c5bf1a1495158c04ce509432c1",
        ),
    )
    for source, tagged, model_codec, archive_sha256 in variants:
        source_parts = receiver.split_payload(_read_p(source))
        tagged_parts = receiver.split_payload(_read_p(tagged))
        assert tagged.stat().st_size == source.stat().st_size
        assert hashlib.sha256(tagged.read_bytes()).hexdigest() == archive_sha256
        assert tagged_parts.model_codec == model_codec
        assert tagged_parts.token_codec == "range"
        assert tagged_parts.tokens == source_parts.tokens
        decoded = receiver.decode_models(
            tagged_parts.models,
            model_codec=tagged_parts.model_codec,
        )
        assert hashlib.sha256(decoded.raw).hexdigest() == EXPECTED_MODELS_RAW_SHA256


@pytest.mark.skipif(
    not (SPLIT_BROTLI_ARCHIVE.is_file() and REAL_GT_CACHE.is_file()),
    reason="real PR130 archive/cache custody is not mounted",
)
def test_real_pr130_n2_ans_roundtrip_preserves_temporal_and_group_causality() -> None:
    """TOY-BRACKET scope only: real mechanism and inputs, two of 600 frames."""

    inflate = importlib.import_module("inflate")
    device = torch.device("cpu")
    source_parts = receiver.split_payload(_read_p(SPLIT_BROTLI_ARCHIVE))
    split_payload = receiver.pack_payload(
        source_parts.models,
        source_parts.tokens,
        token_codec="range",
        model_codec="split_brotli",
    )
    split_parts = receiver.split_payload(split_payload)
    decoded_models = receiver.decode_models(
        split_parts.models, model_codec=split_parts.model_codec
    )
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", decoded_models.raw)
    hpac_offset = 8 + semantic_bytes + carrier_bytes
    hpac = inflate.load_hpac(decoded_models.raw[hpac_offset:], device)
    raw_tokens = torch.load(
        REAL_GT_CACHE, map_location="cpu", weights_only=False
    )["seg"][:2].long()

    range_decode_started = time.perf_counter()
    range_output = inflate.decode_tokens(
        hpac,
        split_parts.tokens,
        device,
        token_codec=split_parts.token_codec,
        frame_count=2,
    )
    range_decode_seconds = time.perf_counter() - range_decode_started
    assert torch.equal(range_output, raw_tokens.to(torch.uint8))

    masks = inflate.group_masks(device)
    sparse = inflate.SparseIntegerHPAC(hpac, inflate.EVAL_H, inflate.EVAL_W)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    previous = torch.zeros(
        (1, inflate.EVAL_H, inflate.EVAL_W), dtype=torch.long, device=device
    )
    materialize_started = time.perf_counter()
    with torch.no_grad():
        for frame in range(2):
            idx = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = hpac.prepare_frame_context(idx, previous)
            target = raw_tokens[frame]
            for group, mask in enumerate(masks):
                selected = sparse.selected_logits(current, context, group)
                table = inflate.probability_table(selected)
                symbols = target[mask].numpy().astype(np.int32)
                blocks.append((symbols, table))
                current[0, mask] = target[mask]
            previous = raw_tokens[frame].view(1, inflate.EVAL_H, inflate.EVAL_W)
    materialize_seconds = time.perf_counter() - materialize_started

    ans_blob = receiver.encode_ans_blocks_reverse(blocks)
    tagged_payload = receiver.pack_payload(
        split_parts.models,
        ans_blob,
        token_codec="ans",
        model_codec=split_parts.model_codec,
    )
    tagged_parts = receiver.split_payload(tagged_payload)
    assert tagged_parts.token_codec == "ans"
    assert hashlib.sha256(receiver.decode_models(
        tagged_parts.models, model_codec=tagged_parts.model_codec
    ).raw).hexdigest() == EXPECTED_MODELS_RAW_SHA256

    decode_started = time.perf_counter()
    output = inflate.decode_tokens(
        hpac,
        tagged_parts.tokens,
        device,
        token_codec=tagged_parts.token_codec,
        frame_count=2,
    )
    decode_seconds = time.perf_counter() - decode_started
    assert torch.equal(output, raw_tokens.to(torch.uint8))
    print({
        "scope": "TOY-BRACKET real PR130 n2/600",
        "ans_bytes": len(ans_blob),
        "materialize_seconds": materialize_seconds,
        "range_decode_seconds": range_decode_seconds,
        "decode_seconds": decode_seconds,
        "token_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
    })
