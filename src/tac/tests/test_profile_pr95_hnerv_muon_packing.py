from __future__ import annotations

import importlib.util
import io
import json
import struct
import zipfile
from pathlib import Path

import brotli


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "experiments" / "profile_pr95_hnerv_muon_packing.py"
spec = importlib.util.spec_from_file_location("profile_pr95_hnerv_muon_packing", TOOL_PATH)
assert spec is not None
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
# Register before exec so @dataclasses.dataclass(frozen=True) sees its module
# in sys.modules. Python 3.12's dataclass KW_ONLY scan does
# `sys.modules.get(cls.__module__).__dict__`, which raises AttributeError on
# None when the module isn't registered yet.
import sys as _sys
_sys.modules["profile_pr95_hnerv_muon_packing"] = tool
spec.loader.exec_module(tool)


def _decoder_record(name: str, q: bytes, shape: tuple[int, ...] | None = None) -> tool.DecoderRecord:
    if shape is None:
        shape = (len(q),)
    return tool.DecoderRecord(name=name, shape=shape, scale=0.125, quantized_zigzag=q)


def _decoder_raw(records: list[tool.DecoderRecord]) -> bytes:
    return tool.rebuild_structured_decoder_raw(records)


def _top_blob(meta_raw: bytes, decoder_raw: bytes, latents_raw: bytes) -> bytes:
    out = io.BytesIO()
    for payload in (
        brotli.compress(meta_raw, quality=5),
        brotli.compress(decoder_raw, quality=5),
        brotli.compress(latents_raw, quality=5),
    ):
        out.write(struct.pack("<I", len(payload)))
        out.write(payload)
    return out.getvalue()


def _write_archive(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w", strict_timestamps=True) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def test_parse_top_blob_roundtrips_raw_streams() -> None:
    meta_raw = b'{"latent_dim":1,"n_pairs":1}'
    decoder_raw = _decoder_raw([_decoder_record("x", b"\x01")])
    latents_raw = tool.LatentPayload(
        n_pairs=1,
        latent_dim=1,
        mins_f16=b"aa",
        scales_f16=b"bb",
        quantized=((7,),),
    ).to_bytes()

    blob = _top_blob(meta_raw, decoder_raw, latents_raw)
    parsed = tool.parse_top_blob(blob)

    assert parsed["meta_raw"] == meta_raw
    assert parsed["decoder_raw"] == decoder_raw
    assert parsed["latents_raw"] == latents_raw
    assert tool.encode_top_blob(
        parsed["meta_brotli"], parsed["decoder_brotli"], parsed["latents_brotli"]
    ) == blob


def test_decoder_raw_variants_preserve_record_set() -> None:
    records = [_decoder_record("b.weight", b"\x01\x02"), _decoder_record("a.weight", b"\x03")]
    raw = _decoder_raw(records)
    variants = tool.decoder_raw_variants(raw)

    assert set(variants) == {"original", "name_asc", "name_desc", "size_desc", "size_asc"}
    for variant in variants.values():
        parsed = tool.parse_decoder_records_structured(variant)
        assert sorted(record.name for record in parsed) == ["a.weight", "b.weight"]


def test_latent_dimension_permutation_compensates_stem_weight() -> None:
    stem = _decoder_record("stem.weight", bytes([1, 2, 3, 4, 5, 6]), shape=(2, 3))
    latents_raw = tool.LatentPayload(
        n_pairs=3,
        latent_dim=3,
        mins_f16=b"aabbcc",
        scales_f16=b"ddeeff",
        quantized=((1, 2, 3), (2, 4, 6), (3, 6, 9)),
    ).to_bytes()
    permutation = [2, 0, 1]

    reordered_latents = tool.parse_latents_raw(tool.reorder_latents_raw(latents_raw, permutation))
    reordered_stem = tool.reorder_stem_weight_record(stem, permutation)

    assert reordered_latents.quantized == ((3, 1, 2), (6, 2, 4), (9, 3, 6))
    assert reordered_latents.mins_f16 == b"ccaabb"
    assert reordered_latents.scales_f16 == b"ffddee"
    assert reordered_stem.quantized_zigzag == bytes([3, 1, 2, 6, 4, 5])


def test_profiler_emits_no_score_claim(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    profile_json = tmp_path / "profile.json"
    profile_md = tmp_path / "profile.md"
    meta_raw = json.dumps({"latent_dim": 2, "n_pairs": 2}, sort_keys=True).encode()
    decoder_raw = _decoder_raw([_decoder_record("stem.weight", b"\x01\x02\x03\x04", shape=(2, 2))])
    latents_raw = tool.LatentPayload(
        n_pairs=2,
        latent_dim=2,
        mins_f16=b"aabb",
        scales_f16=b"ccdd",
        quantized=((1, 2), (3, 1)),
    ).to_bytes()
    _write_archive(archive, _top_blob(meta_raw, decoder_raw, latents_raw))

    record = tool.run(archive, output_json=profile_json, output_md=profile_md)

    assert record["score_claim"] is False
    assert record["evidence_grade"] == "forensic_byte_profile"
    assert record["decoder"]["record_count"] == 1
    assert record["latents"]["n_pairs"] == 2
    assert json.loads(profile_json.read_text())["member_name"] == "0.bin"
    assert "PR95 HNeRV Muon Packing Profile" in profile_md.read_text()
