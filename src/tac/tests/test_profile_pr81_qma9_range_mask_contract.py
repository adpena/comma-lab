# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_pr81_qma9_range_mask_contract.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_pr81_qma9_range_mask_contract_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("p")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_parse_qma9_header_records_dimensions_and_bitstream_hash() -> None:
    script = _load_script()
    mask = struct.pack("<4sIIII", b"QMA9", 2, 4, 3, 5) + b"abcde"

    profile = script.parse_qma9_mask(mask)

    assert profile.magic == "QMA9"
    assert profile.frame_count == 2
    assert profile.width == 4
    assert profile.height == 3
    assert profile.bitstream_bytes == 5
    assert profile.packed_bytes == 25
    assert profile.decoded_mask_bytes == 24
    assert profile.context_symbol_count == 6**9
    assert profile.bitstream_sha256 == script._sha256_bytes(b"abcde")


def test_pr81_split_uses_constant_table_and_qma9_self_header() -> None:
    script = _load_script()
    constants = {
        "PACKED_PAYLOAD_BYTES": 43,
        "RANGE_MASK_BYTES": 25,
        "SPLIT_MODEL_REORDERED_BYTES": 8,
        "POSE_STREAM_BYTES": 6,
        "ROUTER_ACTION_BYTES": 4,
    }
    mask = struct.pack("<4sIIII", b"QMA9", 2, 4, 3, 5) + b"abcde"
    payload = mask + b"model123" + b"pose12" + b"rout"

    split, qma9 = script.build_pr81_split(payload, constants)

    assert split.payload_format == "public_pr81_qma9_range_mask_qzs3_split_model_qp1_router_v1"
    assert split.boundary_authority == "public_pr81_inflate_constant_table_plus_qma9_self_header"
    assert [(s.name, s.offset, s.bytes) for s in split.segments] == [
        ("range_mask.qma9", 0, 25),
        ("split_model_reordered.br_bundle", 25, 8),
        ("optimized_poses.qp1.br", 33, 6),
        ("router_actions.3bit", 39, 4),
    ]
    assert qma9.decoded_mask_bytes == 24


def test_no_router_qma9_split_derives_pose_stream_default() -> None:
    script = _load_script()
    constants = {
        "RANGE_MASK_BYTES": 25,
        "SPLIT_MODEL_PACKED_REORDERED_BR_BYTES": 3,
        "SPLIT_MODEL_SCALES_REORDERED_BR_BYTES": 2,
        "SPLIT_MODEL_TAIL_REORDERED_BR_BYTES": 3,
        "SPLIT_MODEL_REORDERED_BYTES": 8,
        "ROUTER_ACTION_BYTES": 4,
        "PACKED_PAYLOAD_BYTES": 37,
    }
    mask = struct.pack("<4sIIII", b"QMA9", 2, 4, 3, 5) + b"abcde"
    payload = mask + b"model123" + (b"p" * 899)

    split, qma9 = script.build_pr81_split(payload, constants)

    assert split.payload_format == "public_qma9_range_mask_qzs3_split_model_qp1_no_router_v1"
    assert split.expected_payload_bytes == len(payload)
    assert [(s.name, s.offset, s.bytes) for s in split.segments] == [
        ("range_mask.qma9", 0, 25),
        ("split_model_reordered.br_bundle", 25, 8),
        ("optimized_poses.qp1.br", 33, 899),
    ]
    assert qma9.decoded_mask_bytes == 24


def test_break_even_vs_reference_is_static_rate_math_only() -> None:
    script = _load_script()
    ref = script.ReferenceProfile(label="PR79_S2", available=True, archive_bytes=277_321)

    be = script.break_even_vs_reference(215_960, ref)

    assert be is not None
    assert be.archive_byte_delta_pr81_minus_reference == -61_361
    assert be.bytes_saved_before_equal_rate == 61_361
    expected = -61_361 * 25.0 / 37_545_489
    assert math.isclose(be.rate_score_delta_if_components_unchanged, expected)
    assert math.isclose(be.component_score_worsening_budget_before_equal_total, -expected)
    assert "not a score claim" in be.note


def test_build_profile_is_planning_only_and_never_requests_decode_by_default(tmp_path: Path) -> None:
    script = _load_script()
    constants_py = tmp_path / "inflate_constants.py"
    constants_py.write_text(
        "\n".join(
            [
                "RANGE_MASK_BYTES = 25",
                "SPLIT_MODEL_PACKED_REORDERED_BR_BYTES = 3",
                "SPLIT_MODEL_SCALES_REORDERED_BR_BYTES = 2",
                "SPLIT_MODEL_TAIL_REORDERED_BR_BYTES = 3",
                "SPLIT_MODEL_REORDERED_BYTES = (",
                "    SPLIT_MODEL_PACKED_REORDERED_BR_BYTES",
                "    + SPLIT_MODEL_SCALES_REORDERED_BR_BYTES",
                "    + SPLIT_MODEL_TAIL_REORDERED_BR_BYTES",
                ")",
                "POSE_STREAM_BYTES = 6",
                "ROUTER_ACTION_BYTES = 4",
                "ROUTER_ACTION_COUNT = 600",
                "ROUTER_ACTION_BITS = 3",
                "PACKED_PAYLOAD_BYTES = RANGE_MASK_BYTES + SPLIT_MODEL_REORDERED_BYTES + POSE_STREAM_BYTES + ROUTER_ACTION_BYTES",
            ]
        )
    )
    mask = struct.pack("<4sIIII", b"QMA9", 2, 4, 3, 5) + b"abcde"
    archive = tmp_path / "archive.zip"
    _write_zip(archive, mask + b"model123" + b"pose12" + b"rout")
    pr79 = tmp_path / "pr79.json"
    pr79.write_text(
        json.dumps(
            {
                "public_archive": {
                    "archive": {"bytes": 100, "sha256": "pr79"},
                    "payload": {"bytes": 90, "sha256": "payload"},
                    "decoded_streams": {
                        "masks.mkv": {"charged_bytes": 40},
                        "renderer.bin": {"charged_bytes": 8},
                        "optimized_poses.qp1": {"charged_bytes": 6},
                        "seg_tile_actions.bin": {"charged_bytes": 10},
                    },
                }
            }
        )
    )
    s2 = tmp_path / "missing_s2.json"

    profile = script.build_profile(
        archive_path=archive,
        split_constants_path=constants_py,
        pr79_profile_path=pr79,
        pr79_s2_profile_path=s2,
        range_mask_codec_cpp=tmp_path / "range_mask_codec.cpp",
        try_cpp_decode_hash=False,
        cpp_timeout_seconds=1,
    )

    assert profile["evidence_grade"] == "external/planning_only"
    assert profile["score_claim"] is False
    assert profile["dispatch_performed"] is False
    assert profile["gpu_required"] is False
    assert profile["optional_cpp_decode_hash_validation"] == {
        "attempted": False,
        "status": "skipped_not_requested",
    }
    assert profile["qma9"]["decoded_mask_bytes"] == 24
    assert profile["component_byte_deltas"][0]["delta_bytes_pr81_minus_reference"] == -15
    assert profile["transfer_recommendations"][0]["evidence_grade"] == "external/planning_only"
