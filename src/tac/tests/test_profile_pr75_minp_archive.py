from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_pr75_minp_archive.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_pr75_minp_archive_test", SCRIPT)
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


def test_current_minp_fixed_slice_plan_matches_public_runtime_table() -> None:
    script = _load_script()

    plan = script.fixed_slice_plan_for_payload(b"x" * 276_381)

    assert plan.label == "pr75_minp_fixed_actions255_model55756"
    assert plan.mask_br_bytes == 219_472
    assert plan.renderer_br_bytes == 55_756
    assert plan.actions_br_bytes == 255
    assert plan.pose_br_bytes == 898


def test_pr79_minp_v2_fixed_slice_plan_matches_public_runtime_table() -> None:
    script = _load_script()

    plan = script.fixed_slice_plan_for_payload(b"x" * 277_288)

    assert plan.label == "pr79_minp_v2_fixed_actions1162_model55756"
    assert plan.mask_br_bytes == 219_472
    assert plan.renderer_br_bytes == 55_756
    assert plan.actions_br_bytes == 1_162
    assert plan.pose_br_bytes == 898


def test_sg2_grouped_actions_decode_to_runtime_raw4_records() -> None:
    script = _load_script()
    # Tile 17 has two frame-delta actions: frame 33, then 36.
    # Tile 20 has one independent action at frame 598.
    raw = (
        b"SG2"
        + _uvarint(17)
        + _uvarint(2)
        + _uvarint(33)
        + bytes([92])
        + _uvarint(3)
        + bytes([93])
        + _uvarint(20)
        + _uvarint(1)
        + _uvarint(598)
        + bytes([107])
    )

    wire_kind, records = script.decode_seg_tile_actions_raw(raw)

    assert wire_kind == "SG2_grouped_tile_frame_delta_varint"
    assert records == [(33, 17, 92), (36, 17, 93), (598, 20, 107)]
    assert script.encode_runtime_action_records(records) == (
        (33).to_bytes(2, "little")
        + bytes([17, 92])
        + (36).to_bytes(2, "little")
        + bytes([17, 93])
        + (598).to_bytes(2, "little")
        + bytes([20, 107])
    )


def test_action_summary_records_counts_hash_and_pairs() -> None:
    script = _load_script()
    records = [(33, 17, 92), (36, 17, 93), (36, 18, 93)]
    runtime = script.encode_runtime_action_records(records)

    summary = script.summarize_action_records(
        raw_wire=b"raw",
        charged=b"compressed",
        records=records,
        wire_kind="unit",
    )

    assert summary["record_count"] == 3
    assert summary["unique_pair_count"] == 2
    assert summary["pair_min"] == 33
    assert summary["pair_max"] == 36
    assert summary["unique_tile_count"] == 2
    assert summary["unique_action_count"] == 2
    assert summary["runtime_record_bytes"] == len(runtime)
    assert summary["runtime_record_sha256"] == script._sha256_bytes(runtime)
