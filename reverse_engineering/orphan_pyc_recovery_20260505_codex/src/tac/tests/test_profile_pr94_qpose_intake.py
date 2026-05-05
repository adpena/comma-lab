# pyc-recovery pass2: rehydrated from git blob 7fa79e90334b20596dc389a5c438b6c9579b2368 via `git fsck --lost-found`
# original path: src/tac/tests/test_profile_pr94_qpose_intake.py
# OUR source dropped during commit 66c59aae filter-repo cleanup; .pyc was sole orphan left.
# Blob verified intact + parses cleanly with python ast.
# Recovered: 2026-05-05 by Sherlock pass2
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import brotli


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments/profile_pr94_qpose_intake.py"


def load_module():
    spec = importlib.util.spec_from_file_location("profile_pr94_qpose_intake", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def uvarint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def test_infer_pr94_fixed_range_layout_for_observed_payload_size():
    module = load_module()

    layout = module.infer_pr94_layout(b"\0" * 276_987)

    assert layout.payload_format == "pr94_fixed_range_qpose_tile_actions"
    assert layout.mask_len == 219_472
    assert layout.model_len == 55_756
    assert layout.actions_len == 861
    assert layout.pose_len == 898


def test_profile_payload_decodes_qp1_velocity_and_sg2_actions():
    module = load_module()
    mask = brotli.compress(b"mask-obu")
    model = brotli.compress(b"QZS3renderer")
    actions_raw = b"SG2" + uvarint(5) + uvarint(2) + uvarint(10) + bytes([1]) + uvarint(2) + bytes([3])
    actions = brotli.compress(actions_raw)
    pose_raw = b"QP1" + (100).to_bytes(2, "little") + uvarint(zigzag(1)) + uvarint(zigzag(-2))
    pose = brotli.compress(pose_raw)
    payload = mask + model + actions + pose
    layout = module.SegmentLayout(
        payload_format="test",
        boundary_authority="unit_test",
        header_bytes=0,
        mask_len=len(mask),
        model_len=len(model),
        actions_len=len(actions),
        pose_len=len(pose),
    )

    profile = module.profile_payload(payload, layout=layout)

    assert profile["classification"]["renderer_magic"] == "QZS3"
    assert profile["classification"]["qpose"]["format"] == "QP1_velocity_delta_varint"
    assert profile["classification"]["qpose"]["pose_rows"] == 3
    assert profile["classification"]["qpose"]["non_velocity_columns_fixed_zero"] is True
    assert profile["classification"]["tile_actions"]["format"] == "sg2_tile_group_varint"
    assert profile["classification"]["tile_actions"]["record_count"] == 2
    assert profile["classification"]["tile_actions"]["first_records"] == [
        {"frame": 10, "tile": 5, "action": 1},
        {"frame": 12, "tile": 5, "action": 3},
    ]
