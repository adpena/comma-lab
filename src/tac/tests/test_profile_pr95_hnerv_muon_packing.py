from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import brotli

from experiments.profile_pr95_hnerv_muon_packing import (
    LatentPayload,
    decoder_raw_variants,
    parse_decoder_records_structured,
    parse_latents_raw,
    parse_top_blob,
    reorder_latents_raw,
    reorder_stem_weight_record,
    run,
)


def _decoder_record(name: str, q: bytes, shape: tuple[int, ...] | None = None) -> bytes:
    shape = shape or (len(q),)
    total = 1
    for dim in shape:
        total *= dim
    assert total == len(q)
    out = io.BytesIO()
    name_b = name.encode("utf-8")
    out.write(struct.pack("<I", len(name_b)))
    out.write(name_b)
    out.write(struct.pack("<I", len(shape)))
    for dim in shape:
        out.write(struct.pack("<I", dim))
    out.write(struct.pack("<f", 0.25))
    out.write(struct.pack("<I", len(q)))
    out.write(q)
    return out.getvalue()


def _decoder_raw(records: list[bytes]) -> bytes:
    out = io.BytesIO()
    out.write(struct.pack("<I", len(records)))
    for record in records:
        out.write(record)
    return out.getvalue()


def _top_blob(meta_raw: bytes, decoder_raw: bytes, latents_raw: bytes) -> bytes:
    meta = brotli.compress(meta_raw, quality=5)
    dec = brotli.compress(decoder_raw, quality=5)
    lat = brotli.compress(latents_raw, quality=5)
    out = io.BytesIO()
    for payload in (meta, dec, lat):
        out.write(struct.pack("<I", len(payload)))
        out.write(payload)
    return out.getvalue()


def test_decoder_raw_variants_preserve_record_set() -> None:
    records = [_decoder_record("b.weight", b"\x01\x02"), _decoder_record("a.weight", b"\x03")]
    raw = _decoder_raw(records)

    variants = decoder_raw_variants(raw)

    assert set(variants) == {"original", "name_asc", "name_desc", "size_desc", "size_asc"}
    for candidate in variants.values():
        assert sorted(decoder_raw_variants(candidate)) == sorted(variants)


def test_parse_top_blob_roundtrips_raw_streams() -> None:
    meta_raw = b'{"latent_dim": 1, "n_pairs": 1}'
    decoder_raw = _decoder_raw([_decoder_record("x", b"\x01")])
    latents_raw = b"latent-bytes"
    blob = _top_blob(meta_raw, decoder_raw, latents_raw)

    parsed = parse_top_blob(blob)

    assert parsed["meta_raw"] == meta_raw
    assert parsed["decoder_raw"] == decoder_raw
    assert parsed["latents_raw"] == latents_raw


def test_latent_dimension_permutation_compensates_stem_weight() -> None:
    stem = _decoder_record("stem.weight", bytes([1, 2, 3, 4, 5, 6]), shape=(2, 3))
    decoder_raw = _decoder_raw([stem])
    latents_raw = LatentPayload(
        n_pairs=3,
        latent_dim=3,
        mins_f16=b"aabbcc",
        scales_f16=b"ddeeff",
        quantized=((1, 2, 3), (2, 4, 6), (3, 6, 9)),
    ).to_bytes()
    permutation = [2, 0, 1]

    reordered_latents = parse_latents_raw(reorder_latents_raw(latents_raw, permutation))
    reordered_stem = reorder_stem_weight_record(parse_decoder_records_structured(decoder_raw)[0], permutation)

    assert reordered_latents.quantized == ((3, 1, 2), (6, 2, 4), (9, 3, 6))
    assert reordered_stem.q_zz == bytes([3, 1, 2, 6, 4, 5])

    original_rows = parse_latents_raw(latents_raw).quantized
    original_w = [[1, 2, 3], [4, 5, 6]]
    reordered_w = [[3, 1, 2], [6, 4, 5]]
    for row, reordered_row in zip(original_rows, reordered_latents.quantized):
        assert [
            sum(value * weight for value, weight in zip(row, weights))
            for weights in original_w
        ] == [
            sum(value * weight for value, weight in zip(reordered_row, weights))
            for weights in reordered_w
        ]


def test_profiler_emits_no_score_claim_candidate(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    blob = _top_blob(
        b'{"latent_dim": 1, "n_pairs": 1}',
        _decoder_raw([
            _decoder_record("stem.weight", b"\x01\x02", shape=(2, 1)),
            _decoder_record("b", b"\x01\x02"),
            _decoder_record("a", b"\x03"),
        ]),
        LatentPayload(
            n_pairs=3,
            latent_dim=1,
            mins_f16=b"aa",
            scales_f16=b"bb",
            quantized=((1,), (2,), (3,)),
        ).to_bytes(),
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", blob)

    class Args:
        min_quality = 4
        max_quality = 5
        min_lgwin = 10
        max_lgwin = 12

        def __init__(self) -> None:
            self.archive = str(archive)
            self.output_dir = str(tmp_path / "out")

    assert run(Args()) == 0
    manifest = json.loads((tmp_path / "out" / "profile_pr95_hnerv_muon_packing.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["safety"]["no_tensor_requantization"] is True
    assert manifest["safety"]["uses_existing_pr95_runtime_contract"] is True
    assert manifest["no_op_detection"]["source_archive_reused"] is False
    assert "decoder_records" in manifest["accounting"]
    assert Path(manifest["candidate_archive"]).is_file()
