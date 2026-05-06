from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import brotli

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "profile_pr75_minp_archive.py"


def load_module():
    spec = importlib.util.spec_from_file_location("profile_pr75_minp_archive", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
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


def test_split_fixed_public_payload_uses_measured_pr75_boundaries() -> None:
    module = load_module()
    plan = module.FIXED_SLICE_PLANS[276_362]
    payload = (
        b"m" * plan.mask_br_bytes
        + b"r" * plan.renderer_br_bytes
        + b"a" * plan.actions_br_bytes
        + b"p" * plan.pose_br_bytes
    )

    parsed_plan, parts = module.split_fixed_public_payload(payload)

    assert parsed_plan.label == "pr75_minp_fixed_actions236_model55756"
    assert len(parts["masks.mkv.br"]) == plan.mask_br_bytes
    assert len(parts["renderer.bin.br"]) == plan.renderer_br_bytes
    assert len(parts["seg_tile_actions.br"]) == plan.actions_br_bytes
    assert len(parts["optimized_poses.qp1.br"]) == plan.pose_br_bytes


def test_decode_seg_tile_actions_supports_s1_split_and_raw4() -> None:
    module = load_module()
    s1 = (
        b"S1"
        + _uvarint(1)
        + _uvarint(5)
        + _uvarint(2)
        + _uvarint(10)
        + _uvarint(2)
        + bytes([1, 3])
    )

    kind, records = module.decode_seg_tile_actions_raw(s1)

    assert kind == "S1_split_tile_delta_count_pair_delta_actions"
    assert records == [(10, 5, 1), (12, 5, 3)]

    kind, records = module.decode_seg_tile_actions_raw(b"\x01\x00\x02\x03")
    assert kind == "raw4_u16pair_u8tile_u8action"
    assert records == [(1, 2, 3)]


def test_profile_streams_from_fixed_payload_without_score_claim() -> None:
    module = load_module()
    plan = module.FIXED_SLICE_PLANS[276_362]
    mask = brotli.compress(b"mask-obu", quality=5)
    renderer = brotli.compress(b"QZS3renderer", quality=5)
    actions = brotli.compress(b"\x01\x00\x02\x03", quality=5)
    pose = brotli.compress(b"QP1pose", quality=5)
    payload = (
        mask.ljust(plan.mask_br_bytes, b"\0")
        + renderer.ljust(plan.renderer_br_bytes, b"\0")
        + actions.ljust(plan.actions_br_bytes, b"\0")
        + pose.ljust(plan.pose_br_bytes, b"\0")
    )

    _, parts = module.split_fixed_public_payload(payload)

    assert set(parts) == {
        "masks.mkv.br",
        "renderer.bin.br",
        "seg_tile_actions.br",
        "optimized_poses.qp1.br",
    }
