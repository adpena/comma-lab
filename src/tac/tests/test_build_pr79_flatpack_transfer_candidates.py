from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import pytest


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr79_flatpack_transfer_candidates.py"
MINP_BUILDER_PATH = REPO / "experiments" / "build_pr75_minp_lossless_micro_candidates.py"
PR79_ARCHIVE = REPO / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)


def _action_records(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair, tile, action in records:
        out += int(pair).to_bytes(2, "little") + bytes([int(tile), int(action)])
    return bytes(out)


def _uleb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _p6_actions(raw_actions: bytes) -> bytes:
    out = bytearray()
    previous = 0
    for offset in range(0, len(raw_actions), 4):
        pair = int.from_bytes(raw_actions[offset : offset + 2], "little")
        delta = pair if offset == 0 else pair - previous
        out.extend(_uleb128(delta))
        out.extend(raw_actions[offset + 2 : offset + 4])
        previous = pair
    return bytes(out)


def _p6_payload(records: list[tuple[int, int, int]], *, tag: bytes) -> bytes:
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + tag + b"m" * 64, quality=0, lgwin=10)
    renderer_br = brotli.compress(b"QZS3" + tag + b"r" * 64, quality=0, lgwin=10)
    raw_actions = _action_records(records)
    actions_br = brotli.compress(_p6_actions(raw_actions), quality=0, lgwin=10)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little") + tag, quality=0, lgwin=10)
    return (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), len(records))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def test_flatpack_transfer_emits_runtime_closed_rpk1_matrix(tmp_path: Path, monkeypatch) -> None:
    builder = _load_module(BUILDER_PATH, "pr79_flatpack_builder_test")
    c102_archive = tmp_path / "c102.zip"
    pr79_archive = tmp_path / "pr79.zip"
    _stored_zip(c102_archive, _p6_payload([(1, 2, 3), (4, 5, 6)], tag=b"c102"))
    _stored_zip(pr79_archive, _p6_payload([(1, 2, 3), (8, 5, 9), (11, 6, 7)], tag=b"pr79"))
    monkeypatch.setattr(builder, "C102_SHA256", builder._sha256_file(c102_archive))
    monkeypatch.setattr(builder, "C102_BYTES", c102_archive.stat().st_size)
    monkeypatch.setattr(builder, "PR79_SHA256", builder._sha256_file(pr79_archive))

    out = tmp_path / "out"
    summary = builder.build_candidates(
        c102_archive=c102_archive,
        pr79_archive=pr79_archive,
        existing_matrix=tmp_path / "missing.json",
        output_dir=out,
        max_existing_sources=0,
        params=builder.fast_brotli_param_grid(),
    )

    assert (out / "candidate_matrix.json").exists()
    assert summary["score_claim"] is False
    assert summary["compressor_attempts"]["brotli"]["runtime_closed"] is True
    assert summary["compressor_attempts"]["zstd"]["attempted"] is False
    assert summary["compressor_attempts"]["lzma2"]["attempted"] is False
    assert summary["archive_anatomy_comparison"]["stream_sha_equalities"]["masks.mkv"] is False
    assert len(summary["candidates"]) == 2
    for row in summary["candidates"]:
        assert row["container"] == "brotli_rpk1_single_stream_permuted"
        assert row["decode_parity_status"] == "passed"
        assert row["payload_changed_vs_source"] is True
        assert row["score_claim"] is False
        manifest = json.loads((out / row["candidate_id"] / "manifest.json").read_text())
        assert manifest["runtime_parse_validation"]["status"] == "passed"
        assert manifest["promotion_eligible"] is False


@pytest.mark.skipif(not PR79_ARCHIVE.exists(), reason="local PR79 reverse-engineering archive absent")
def test_existing_minp_builder_recognizes_pr79_fixed_raw_slice_archive() -> None:
    minp = _load_module(MINP_BUILDER_PATH, "pr75_minp_builder_pr79_fixed_test")
    unpacker = minp._load_unpacker()  # noqa: SLF001

    source = minp._load_source("pr79", PR79_ARCHIVE, unpacker)  # noqa: SLF001

    assert source.archive_sha256 == minp.PUBLIC_PR79_MINP_V2_SHA256
    assert source.payload_format == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    assert len(source.encoded.actions_br) == 1162
    assert len(source.decoded["seg_tile_actions.bin"]) == 2688
