# SPDX-License-Identifier: MIT
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
BUILDER_PATH = REPO / "experiments" / "build_pr79_action_subset_candidates.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("build_pr79_action_subset_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _uvarint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _raw4(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair, tile, action in records:
        out += int(pair).to_bytes(2, "little") + bytes([tile, action])
    return bytes(out)


def _sg2(records_by_tile: list[tuple[int, list[tuple[int, int]]]]) -> bytes:
    raw = bytearray(b"SG2")
    for tile, records in records_by_tile:
        raw += _uvarint(tile)
        raw += _uvarint(len(records))
        previous = 0
        for index, (pair, action) in enumerate(records):
            raw += _uvarint(pair if index == 0 else pair - previous)
            raw.append(action)
            previous = pair
    return bytes(raw)


def _write_p3_archive(path: Path, *, action_wire: bytes) -> Path:
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + b"m" * 32, quality=0)
    renderer_br = brotli.compress(b"QZS3" + b"r" * 32, quality=0)
    actions_br = brotli.compress(action_wire, quality=0)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little") + b"p" * 8, quality=0)
    payload = (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(actions_br))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)
    return path


def test_pr79_sg2_decode_expands_grouped_records() -> None:
    builder = _load_builder()
    raw = _sg2(
        [
            (83, [(12, 92), (16, 7), (21, 23)]),
            (140, [(2, 5)]),
        ]
    )

    records = builder.decode_sg2_action_wire(raw)
    prefixless = builder.decode_sg2_action_wire(raw[3:])

    assert [(record.pair_index, record.tile_id, record.action_id) for record in records] == [
        (12, 83, 92),
        (16, 83, 7),
        (21, 83, 23),
        (2, 140, 5),
    ]
    assert [(record.pair_index, record.tile_id, record.action_id) for record in prefixless] == [
        (12, 83, 92),
        (16, 83, 7),
        (21, 83, 23),
        (2, 140, 5),
    ]


def test_build_candidates_is_deterministic_and_non_noop(tmp_path: Path) -> None:
    builder = _load_builder()
    base_archive = _write_p3_archive(
        tmp_path / "c102.zip",
        action_wire=_raw4([(1, 83, 7), (5, 84, 9)]),
    )
    pr79_archive = _write_p3_archive(
        tmp_path / "pr79.zip",
        action_wire=_sg2([(83, [(2, 11), (9, 13)])]),
    )
    output_dir = tmp_path / "out"

    summary = builder.build_candidates(
        c102_archive=base_archive,
        c102_eval=tmp_path / "missing_eval.json",
        c102_trace=tmp_path / "missing_trace.json",
        pr79_archive=pr79_archive,
        pr79_profile=tmp_path / "missing_profile.json",
        prior_policy=None,
        output_dir=output_dir,
        policies=["replace_pr79_first1_p6"],
    )
    first_archive = Path(summary["candidates"][0]["archive_path"])
    first_bytes = first_archive.read_bytes()
    again = builder.build_candidates(
        c102_archive=base_archive,
        c102_eval=tmp_path / "missing_eval.json",
        c102_trace=tmp_path / "missing_trace.json",
        pr79_archive=pr79_archive,
        pr79_profile=tmp_path / "missing_profile.json",
        prior_policy=None,
        output_dir=output_dir,
        policies=["replace_pr79_first1_p6"],
        force=True,
    )
    manifest = json.loads(
        (output_dir / "replace_pr79_first1_on_c102_p6" / "manifest.json").read_text()
    )

    assert first_archive.read_bytes() == first_bytes
    assert summary["candidates"][0]["archive_sha256"] == again["candidates"][0]["archive_sha256"]
    assert manifest["runtime_parse_validation"]["non_action_streams_preserved"] is True
    assert (
        manifest["runtime_parse_validation"]["action_semantic_guard"]["no_op_status"]
        == "changes_c102_action_stream"
    )
    assert (
        manifest["runtime_parse_validation"]["action_semantic_guard"]["decoded_action_sha256"]
        != manifest["runtime_parse_validation"]["action_semantic_guard"]["source_decoded_action_sha256"]
    )


def test_validation_rejects_unchanged_decoded_action_stream(tmp_path: Path) -> None:
    builder = _load_builder()
    unpacker = builder._load_unpacker()
    action_raw = _raw4([(2, 83, 11)])
    base = builder.load_archive(
        "c102",
        _write_p3_archive(tmp_path / "c102.zip", action_wire=action_raw),
        unpacker=unpacker,
    )
    base_records = builder._parse_runtime_action_records(
        base.decoded["seg_tile_actions.bin"],
        source_label="c102",
    )
    payload, runtime_raw, _actions_br, ordered = builder._build_p6_payload(base, base_records)

    with pytest.raises(builder.CandidateBuildError, match="unchanged decoded action semantics"):
        builder._validate_candidate_payload(
            base=base,
            base_records=base_records,
            payload=payload,
            selected=ordered,
            expected_runtime_raw=runtime_raw,
            unpacker=unpacker,
        )


def test_break_even_math_uses_target_031() -> None:
    builder = _load_builder()

    screen = builder._break_even_screen(
        archive_bytes=276_485,
        base_archive_bytes=276_485,
        base_score=0.31514430182167497,
    )

    assert screen["target_score"] == 0.31
    assert screen["target_component_score_improvement_needed"] == pytest.approx(
        0.005144301821674976
    )
