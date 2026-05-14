# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import pytest


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_c101_native_action_atom_candidate.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("build_c101_native_action_atom_candidate_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_p3_archive(path: Path, action_raw: bytes) -> Path:
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + b"m" * 96, quality=0)
    renderer_br = brotli.compress(b"QZS3" + b"r" * 96, quality=0)
    actions_br = brotli.compress(action_raw, quality=0)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little") + b"p" * 8, quality=0)
    payload = (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(actions_br))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)
    return path


def _records() -> bytes:
    return (
        (33).to_bytes(2, "little")
        + bytes([109, 5])
        + (36).to_bytes(2, "little")
        + bytes([108, 17])
    )


def test_build_p6_candidate_preserves_non_action_streams_and_changes_actions(tmp_path: Path) -> None:
    builder = _load_builder()
    unpacker = builder._load_unpacker()
    source = builder.load_source_archive(_write_p3_archive(tmp_path / "source.zip", _records()), unpacker=unpacker)
    source_records = builder._parse_action_records(source.decoded["seg_tile_actions.bin"])
    ranked = [
        builder.replace(
            rec,
            support={
                "mean_combined_delta": 0.1,
                "mean_seg_delta": 0.1,
                "mean_pose_delta": -0.2,
                "pose_toxic_votes": 1,
                "positive_votes": 1,
            },
        )
        for rec in source_records
    ]
    selected = builder._synthesize_policy(ranked, "native_pose_guard_ampfit")

    payload, runtime_raw, _actions_br = builder._build_p6_payload(source, selected)
    validation = builder._validate_candidate_payload(
        source=source,
        payload=payload,
        records=selected,
        unpacker=unpacker,
    )

    assert validation["non_action_streams_preserved"] is True
    assert runtime_raw != source.decoded["seg_tile_actions.bin"]
    assert validation["action_semantic_guard"]["changed_action_id_record_count"] == 2


def test_validation_rejects_unchanged_decoded_action_semantics(tmp_path: Path) -> None:
    builder = _load_builder()
    unpacker = builder._load_unpacker()
    source = builder.load_source_archive(_write_p3_archive(tmp_path / "source.zip", _records()), unpacker=unpacker)
    source_records = builder._parse_action_records(source.decoded["seg_tile_actions.bin"])
    payload, _runtime_raw, _actions_br = builder._build_p6_payload(source, source_records)

    with pytest.raises(builder.CandidateBuildError, match="unchanged decoded action semantics"):
        builder._validate_candidate_payload(
            source=source,
            payload=payload,
            records=source_records,
            unpacker=unpacker,
        )


def test_build_one_candidate_rejects_duplicate_exact_archive_sha(tmp_path: Path) -> None:
    builder = _load_builder()
    unpacker = builder._load_unpacker()
    source = builder.load_source_archive(_write_p3_archive(tmp_path / "source.zip", _records()), unpacker=unpacker)
    source_records = builder._parse_action_records(source.decoded["seg_tile_actions.bin"])
    ranked = [
        builder.replace(
            rec,
            support={
                "mean_combined_delta": 0.1,
                "mean_seg_delta": 0.1,
                "mean_pose_delta": -0.2,
                "pose_toxic_votes": 1,
                "positive_votes": 1,
            },
        )
        for rec in source_records
    ]
    payload, _runtime_raw, _actions_br = builder._build_p6_payload(
        source,
        builder._synthesize_policy(ranked, "native_pose_guard_ampfit"),
    )
    archive = tmp_path / "out" / "archive.zip"
    builder._write_archive(archive, payload)
    forbidden_sha = builder._sha256_file(archive)
    anchor = builder.Trace(
        label="anchor",
        path=tmp_path / "trace.json",
        score=0.315,
        archive_bytes=source.archive_bytes,
        samples_by_pair={33: {"seg": 0.1, "pose": 0.1, "combined": 0.2}},
    )

    with pytest.raises(builder.CandidateBuildError, match="duplicate exact SHA"):
        builder.build_one_candidate(
            source=source,
            anchor_trace=anchor,
            ranked_records=ranked,
            observations=[],
            output_dir=tmp_path / "candidates",
            policy="native_pose_guard_ampfit",
            unpacker=unpacker,
            forbidden_archive_shas={forbidden_sha},
        )
