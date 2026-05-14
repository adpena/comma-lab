# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import brotli
import pytest

from tac.submission_archive import (
    validate_archive_seg_tile_actions_payloads,
    validate_seg_tile_actions_payload,
)


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr79_action_dictionary_repack_candidates_v2.py"
PR79_ARCHIVE = (
    REPO / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_action_dict_v2_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw4(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair, tile, action in records:
        out += int(pair).to_bytes(2, "little") + bytes([tile, action])
    return bytes(out)


def _p3_payload(action_wire: bytes) -> bytes:
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + b"m" * 32, quality=0)
    renderer_br = brotli.compress(b"QZS3" + b"r" * 32, quality=0)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little") + b"p" * 8, quality=0)
    return (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(action_wire))
        + mask_br
        + renderer_br
        + action_wire
        + pose_br
    )


def test_s2_adaptive_action_wire_decodes_to_identical_runtime_records() -> None:
    script = _load_script()
    raw_actions = _raw4(
        [
            (12, 83, 92),
            (16, 83, 7),
            (2, 140, 5),
            (21, 83, 23),
            (6, 140, 1),
            (9, 140, 92),
        ]
    )

    encoded = script.encode_s2_adaptive_actions(raw_actions)
    validation = validate_seg_tile_actions_payload(encoded["wire"])
    unpacker = script.BASE._load_unpacker()  # noqa: SLF001
    header, decoded = unpacker._parse_payload(_p3_payload(encoded["wire"]))  # noqa: SLF001

    assert encoded["wire"].startswith(b"S2")
    assert validation["encoding"] == "S2"
    assert validation["record_count"] == 6
    assert header["members"][2]["codec"] == "seg_tile_actions_split_s2_adaptive_arith_v1"
    assert decoded["seg_tile_actions.bin"] == _raw4(
        [
            (12, 83, 92),
            (16, 83, 7),
            (21, 83, 23),
            (2, 140, 5),
            (6, 140, 1),
            (9, 140, 92),
        ]
    )


def test_action_record_accounting_records_duplicates_and_parity_requirement() -> None:
    script = _load_script()
    source = _raw4(
        [
            (12, 83, 92),
            (12, 83, 7),
            (2, 140, 5),
        ]
    )
    repacked = _raw4(
        [
            (2, 140, 5),
            (12, 83, 92),
            (12, 83, 7),
        ]
    )

    accounting = script.action_record_accounting(source, repacked)

    duplicate_accounting = accounting["duplicate_pair_tile_accounting"]
    assert duplicate_accounting["duplicate_pair_tile_record_count"] == 1
    assert duplicate_accounting["duplicate_pair_record_count"] == 1
    assert duplicate_accounting["duplicate_tile_record_count"] == 1
    assert accounting["record_order"]["encoder_reorders_records"] is True
    assert accounting["raw_output_parity_requirement"]["required"] is True
    assert (
        accounting["raw_output_parity_requirement"]["satisfied_by_runtime_parse_validation"]
        is False
    )


@pytest.mark.skipif(not PR79_ARCHIVE.exists(), reason="PR79 reverse-engineering archive missing")
def test_pr79_s2_builder_emits_runtime_closed_candidate_better_than_s1(tmp_path: Path) -> None:
    script = _load_script()

    matrix = script.build_candidates(
        pr79_archive=PR79_ARCHIVE,
        output_dir=tmp_path,
        force=True,
    )
    best = next(
        item for item in matrix["byte_matrix"] if item["candidate_id"] == "pr79_s2_fixed_adaptive_actions"
    )
    s1 = next(item for item in matrix["byte_matrix"] if item["candidate_id"] == "halley_s1_fixed_reference")
    manifest = json.loads(Path(best["manifest_path"]).read_text())

    assert best["archive_bytes"] < s1["archive_bytes"]
    assert best["delta_bytes_vs_pr79"] < s1["delta_bytes_vs_pr79"]
    assert best["decoded_action_sha256"] == matrix["source_action_stream"]["decoded_sha256"]
    assert validate_archive_seg_tile_actions_payloads(best["archive_path"]) == []
    assert manifest["runtime_parse_validation"]["action_record_parity"] is True
    assert manifest["runtime_parse_validation"]["non_action_streams_preserved"] is True
    assert manifest["action_record_accounting"]["duplicate_pair_tile_accounting"][
        "duplicate_pair_tile_record_count"
    ] >= 0
    assert manifest["break_even_math"]["versus_pr79"]["archive_byte_delta"] == best[
        "delta_bytes_vs_pr79"
    ]
    assert "versus_pr79_s2" in manifest["break_even_math"]
    assert (
        manifest["no_op_detection"]["status"]
        == "decoded_action_semantics_preserved_action_bytes_changed"
    )
    assert manifest["remote_dispatch_performed"] is False
