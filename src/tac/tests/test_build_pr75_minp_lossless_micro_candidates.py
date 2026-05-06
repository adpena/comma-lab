from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr75_minp_lossless_micro_candidates.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pr75_lossless_micro_builder_test", BUILDER_PATH)
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
    for pair_index, tile_id, action_id in records:
        out += pair_index.to_bytes(2, "little") + bytes([tile_id, action_id])
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


def _p6_delta_actions(raw_actions: bytes) -> bytes:
    out = bytearray()
    previous = 0
    for index in range(0, len(raw_actions), 4):
        pair = int.from_bytes(raw_actions[index : index + 2], "little")
        delta = pair if index == 0 else pair - previous
        out.extend(_uleb128(delta))
        out.extend(raw_actions[index + 2 : index + 4])
        previous = pair
    return bytes(out)


def _p6_payload(records: list[tuple[int, int, int]], *, renderer_tag: bytes, pose_tag: bytes) -> bytes:
    raw_actions = _action_records(records)
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + b"mask" * 16, quality=0, lgwin=10)
    renderer_br = brotli.compress(b"QZS3" + renderer_tag + b"renderer" * 8, quality=0, lgwin=10)
    actions_br = brotli.compress(_p6_delta_actions(raw_actions), quality=0, lgwin=10)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little") + pose_tag, quality=0, lgwin=10)
    return (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), len(records))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def _p3_payload(records: list[tuple[int, int, int]], *, renderer_tag: bytes, pose_tag: bytes) -> bytes:
    raw_actions = _action_records(records)
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + b"mask" * 16, quality=0, lgwin=10)
    renderer_br = brotli.compress(b"QZS3" + renderer_tag + b"renderer" * 8, quality=0, lgwin=10)
    actions_br = brotli.compress(raw_actions, quality=0, lgwin=10)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little") + pose_tag, quality=0, lgwin=10)
    return (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(actions_br))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def test_micro_candidates_are_deterministic_and_gate_strict_lossless_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    builder = _load_builder()
    c089_archive = tmp_path / "c089.zip"
    public_archive = tmp_path / "public.zip"
    _stored_zip(
        c089_archive,
        _p6_payload(
            [(3, 7, 92), (9, 8, 7), (12, 9, 106)],
            renderer_tag=b"c089",
            pose_tag=b"pose-c089",
        ),
    )
    _stored_zip(
        public_archive,
        _p3_payload(
            [(9, 3, 7), (5, 2, 92), (9, 3, 8)],
            renderer_tag=b"pub!",
            pose_tag=b"pose-public",
        ),
    )
    monkeypatch.setattr(builder, "BASELINE_SHA256", builder._sha256_file(c089_archive))
    monkeypatch.setattr(builder, "PUBLIC_MINP_SHA256", builder._sha256_file(public_archive))

    out = tmp_path / "out"
    summary = builder.build_candidates(
        c089_archive=c089_archive,
        public_archive=public_archive,
        output_dir=out,
        params=[(0, 0, 10, 0)],
    )
    first_matrix = (out / "candidate_matrix.json").read_bytes()
    summary_again = builder.build_candidates(
        c089_archive=c089_archive,
        public_archive=public_archive,
        output_dir=out,
        force=True,
        params=[(0, 0, 10, 0)],
    )

    assert (out / "candidate_matrix.json").read_bytes() == first_matrix
    assert summary_again["candidates"] == summary["candidates"]
    rows = {row["candidate_id"]: row for row in summary["candidates"]}
    strict = rows["c089_p6_lossless_stream_resweep"]
    assert strict["score_claim"] is False
    assert strict["source_preserving_vs_c089"] is True
    assert strict["dispatchable_after_gate"] is (not strict["noop"])
    assert strict["parse_status"] == "passed"
    manifest = json.loads(Path(strict["manifest_path"]).read_text())
    assert manifest["runtime_parse_validation"]["decoded_parity_status"] == "passed"
    assert (
        manifest["decoded_change_summary_vs_c089"]["status"]
        == "decoded_stream_byte_identical_vs_c089"
    )
    assert manifest["promotion_eligible"] is False

    noop = rows["c089_zip_rewrite_noop"]
    assert noop["dispatchable_after_gate"] is False
    assert noop["noop"] is True


def test_probe_rows_are_non_dispatchable_and_explain_why(tmp_path: Path, monkeypatch) -> None:
    builder = _load_builder()
    c089_archive = tmp_path / "c089.zip"
    public_archive = tmp_path / "public.zip"
    _stored_zip(
        c089_archive,
        _p6_payload(
            [(3, 7, 92), (9, 8, 7), (12, 9, 106)],
            renderer_tag=b"c089",
            pose_tag=b"pose-c089",
        ),
    )
    _stored_zip(
        public_archive,
        _p3_payload(
            [(9, 3, 7), (5, 2, 92), (9, 3, 8)],
            renderer_tag=b"pub!",
            pose_tag=b"pose-public",
        ),
    )
    monkeypatch.setattr(builder, "BASELINE_SHA256", builder._sha256_file(c089_archive))
    monkeypatch.setattr(builder, "PUBLIC_MINP_SHA256", builder._sha256_file(public_archive))

    summary = builder.build_candidates(
        c089_archive=c089_archive,
        public_archive=public_archive,
        output_dir=tmp_path / "out",
        params=[(0, 0, 10, 0)],
    )
    rows = {row["candidate_id"]: row for row in summary["candidates"]}

    raw_probe = rows["c089_raw_no_header_fixedslice_probe"]
    assert raw_probe["dispatchable_after_gate"] is False
    assert raw_probe["parse_status"] == "failed_parser_rejected"
    raw_manifest = json.loads(Path(raw_probe["manifest_path"]).read_text())
    assert "fixed-slice overhead opportunity" in " ".join(raw_manifest["notes"])

    sorted_probe = rows["public_minp_p6_sorted_actions_probe"]
    assert sorted_probe["dispatchable_after_gate"] is False
    sorted_manifest = json.loads(Path(sorted_probe["manifest_path"]).read_text())
    sort_summary = sorted_manifest["stream_packing"]["actions_sorted_delta_varint"]["sort_summary"]
    assert sort_summary["duplicate_pair_tile_count"] == 1
    assert sorted_manifest["runtime_parse_validation"]["decoded_parity_status"] == "passed"
    assert sorted_manifest["semantic_contract"].startswith("non_dispatchable")

    p5_probe = rows["c089_p5_action_dict_probe"]
    assert p5_probe["dispatchable_after_gate"] is False
    p5_manifest = json.loads(Path(p5_probe["manifest_path"]).read_text())
    assert "seg_tile_action_dict.bin" in p5_manifest["decoded_change_summary_vs_c089"][
        "changed_decoded_streams_vs_c089"
    ]
    assert p5_manifest["stream_packing"]["semantic_summary"]["unique_action_count"] == 3
