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


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr77_action_pose_mixed_container_candidates.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pr77_action_pose_builder_test", BUILDER_PATH)
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


def _records(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair, tile, action in records:
        out += pair.to_bytes(2, "little") + bytes([tile, action])
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


def _delta_varint(raw_actions: bytes) -> bytes:
    out = bytearray()
    previous = 0
    for offset in range(0, len(raw_actions), 4):
        pair = int.from_bytes(raw_actions[offset : offset + 2], "little")
        delta = pair if offset == 0 else pair - previous
        out.extend(_uleb128(delta))
        out.extend(raw_actions[offset + 2 : offset + 4])
        previous = pair
    return bytes(out)


def _p6_payload(
    *,
    mask_br: bytes,
    renderer_br: bytes,
    actions_br: bytes,
    record_count: int,
    pose_br: bytes,
) -> bytes:
    return (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), record_count)
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


class FakePr75Unpacker:
    def __init__(self, fixed_lengths: list[tuple[int, int, int]]) -> None:
        self.fixed_lengths = fixed_lengths

    def _decode_delta_actions(self, data: bytes, record_count: int) -> bytes:
        packed = brotli.decompress(data)
        out = bytearray(record_count * 4)
        offset = 0
        out_offset = 0
        pair = 0
        for _ in range(record_count):
            shift = 0
            delta = 0
            while True:
                byte = packed[offset]
                offset += 1
                delta |= (byte & 0x7F) << shift
                if byte < 0x80:
                    break
                shift += 7
            pair += delta
            out[out_offset : out_offset + 2] = pair.to_bytes(2, "little")
            out[out_offset + 2] = packed[offset]
            out[out_offset + 3] = packed[offset + 1]
            offset += 2
            out_offset += 4
        assert offset == len(packed)
        return bytes(out)

    def _parse_payload(self, payload: bytes):
        if payload.startswith(b"P3"):
            cursor = 2 + struct.calcsize("<IHH")
            mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
            payload_format = "public_pr75_qzs3_qp1_segactions_p3"
            action_decode = lambda data: brotli.decompress(data)
        elif payload.startswith(b"P6"):
            cursor = 2 + struct.calcsize("<IHHH")
            mask_len, renderer_len, actions_len, record_count = struct.unpack_from(
                "<IHHH",
                payload,
                2,
            )
            payload_format = "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
            action_decode = lambda data: self._decode_delta_actions(data, record_count)
        else:
            cursor = 0
            payload_format = "public_pr75_qzs3_qp1_segactions_fixed_slices"
            action_decode = lambda data: brotli.decompress(data)
            parsed = None
            for mask_len, renderer_len, actions_len in self.fixed_lengths:
                if len(payload) <= mask_len + renderer_len + actions_len:
                    continue
                try:
                    parsed = self._decode_slices(
                        payload,
                        cursor,
                        mask_len,
                        renderer_len,
                        actions_len,
                        action_decode,
                    )
                except brotli.error:
                    continue
                break
            if parsed is None:
                raise ValueError("no fixed-slice parse")
            raw, decoded = parsed
            return self._header(payload_format, raw, decoded), decoded

        raw, decoded = self._decode_slices(
            payload,
            cursor,
            mask_len,
            renderer_len,
            actions_len,
            action_decode,
        )
        return self._header(payload_format, raw, decoded), decoded

    def _decode_slices(
        self,
        payload: bytes,
        cursor: int,
        mask_len: int,
        renderer_len: int,
        actions_len: int,
        action_decode,
    ):
        mask_end = cursor + mask_len
        renderer_end = mask_end + renderer_len
        actions_end = renderer_end + actions_len
        raw = {
            "masks.mkv": payload[cursor:mask_end],
            "renderer.bin": payload[mask_end:renderer_end],
            "seg_tile_actions.bin": payload[renderer_end:actions_end],
            "optimized_poses.qp1": payload[actions_end:],
        }
        decoded = {
            "masks.mkv": brotli.decompress(raw["masks.mkv"]),
            "optimized_poses.qp1": brotli.decompress(raw["optimized_poses.qp1"]),
            "renderer.bin": brotli.decompress(raw["renderer.bin"]),
            "seg_tile_actions.bin": action_decode(raw["seg_tile_actions.bin"]),
        }
        return raw, decoded

    def _header(self, payload_format: str, raw: dict[str, bytes], decoded: dict[str, bytes]):
        import hashlib

        members = []
        for name in ("renderer.bin", "masks.mkv", "seg_tile_actions.bin", "optimized_poses.qp1"):
            members.append(
                {
                    "bytes": len(raw[name]),
                    "codec": "test_codec",
                    "decoded_bytes": len(decoded[name]),
                    "decoded_sha256": hashlib.sha256(decoded[name]).hexdigest(),
                    "name": name,
                    "sha256": hashlib.sha256(raw[name]).hexdigest(),
                }
            )
        return {"members": members, "payload_format": payload_format}


def test_mixed_builder_emits_fixedslice_p3_and_fail_closed_p6_rows(tmp_path: Path) -> None:
    builder = _load_builder()
    mask_dec = b"\x12\x00\x0a\x0a" + b"mask" * 8
    renderer_public_dec = b"QZS3-public-renderer"
    renderer_c089_dec = b"QZS3-c089-renderer"
    pose_public_dec = b"QP1" + b"pose-public"
    pose_c089_dec = b"QP1" + b"pose-c089"
    c091_actions_dec = _records([(4, 1, 3), (8, 2, 4)])
    pr77_actions_dec = _records([(9, 4, 6), (2, 3, 5), (12, 7, 8)])
    c089_actions_dec = _records([(2, 1, 1), (5, 2, 2)])

    mask_public_br = brotli.compress(mask_dec, quality=0, lgwin=10)
    renderer_public_br = brotli.compress(renderer_public_dec, quality=0, lgwin=10)
    pose_public_br = brotli.compress(pose_public_dec, quality=0, lgwin=10)
    pose_c089_br = brotli.compress(pose_c089_dec, quality=0, lgwin=10)
    renderer_c089_br = brotli.compress(renderer_c089_dec, quality=0, lgwin=10)
    c091_actions_br = brotli.compress(c091_actions_dec, quality=0, lgwin=10)
    pr77_actions_br = brotli.compress(pr77_actions_dec, quality=0, lgwin=10)
    c089_actions_br = brotli.compress(_delta_varint(c089_actions_dec), quality=0, lgwin=10)

    c091_payload = mask_public_br + renderer_public_br + c091_actions_br + pose_public_br
    pr77_payload = mask_public_br + renderer_public_br + pr77_actions_br + pose_public_br
    c089_payload = _p6_payload(
        mask_br=mask_public_br,
        renderer_br=renderer_c089_br,
        actions_br=c089_actions_br,
        record_count=2,
        pose_br=pose_c089_br,
    )
    c091_archive = tmp_path / "c091.zip"
    c089_archive = tmp_path / "c089.zip"
    pr77_archive = tmp_path / "pr77.zip"
    _stored_zip(c091_archive, c091_payload)
    _stored_zip(c089_archive, c089_payload)
    _stored_zip(pr77_archive, pr77_payload)
    unpacker = FakePr75Unpacker(
        fixed_lengths=[
            (len(mask_public_br), len(renderer_public_br), len(pr77_actions_br)),
            (len(mask_public_br), len(renderer_public_br), len(c091_actions_br)),
        ]
    )

    output_dir = tmp_path / "out"
    summary = builder.build_candidates(
        c091_archive=c091_archive,
        c089_archive=c089_archive,
        pr77_archive=pr77_archive,
        output_dir=output_dir,
        unpacker=unpacker,
        verify_anchor_hashes=False,
    )
    first_matrix = (output_dir / "candidate_matrix.json").read_bytes()
    summary_again = builder.build_candidates(
        c091_archive=c091_archive,
        c089_archive=c089_archive,
        pr77_archive=pr77_archive,
        output_dir=output_dir,
        force=True,
        unpacker=unpacker,
        verify_anchor_hashes=False,
    )

    assert (output_dir / "candidate_matrix.json").read_bytes() == first_matrix
    assert summary_again["candidates"] == summary["candidates"]
    assert summary["score_claim"] is False
    rows = {row["candidate_id"]: row for row in summary["candidates"]}

    fixed = rows["pr77_actions_pr75mask_renderer_c089pose_fixedslice"]
    assert fixed["dispatchable_after_gate"] is True
    assert fixed["payload_format"] == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    fixed_manifest = json.loads(Path(fixed["manifest_path"]).read_text())
    assert fixed_manifest["fixed_slice_boundary_validation"]["status"] == "passed"
    assert fixed_manifest["runtime_parse_validation"]["decoded_parity_status"] == "passed"
    assert fixed_manifest["score_claim"] is False
    assert fixed_manifest["promotion_eligible"] is False

    p3 = rows["pr77_actions_c089mask_pr75renderer_c089pose_p3"]
    assert p3["dispatchable_after_gate"] is True
    assert p3["payload_format"] == "public_pr75_qzs3_qp1_segactions_p3"

    sorted_probe = rows["pr77_actions_sorted_c089mask_pr75renderer_c089pose_p6_probe"]
    assert sorted_probe["dispatchable_after_gate"] is False
    sorted_manifest = json.loads(Path(sorted_probe["manifest_path"]).read_text())
    assert "raw-output parity" in sorted_manifest["dispatch_safety"]["next_dispatch_safety_gate"]
    assert (
        sorted_manifest["stream_packing"]["action_order_probe"]["proof_status"]
        == "structural_disjoint_pair_tile_targets_only"
    )

    c091_noop = rows["c091_pr75_replay_noop_control"]
    assert c091_noop["noop"] is True
    pr77_noop = rows["pr77_replay_noop_control"]
    assert pr77_noop["noop"] is True


def test_sorted_p6_action_probe_fails_closed_on_duplicate_pair_tiles() -> None:
    builder = _load_builder()
    raw_actions = _records([(2, 3, 5), (2, 3, 6), (4, 1, 7)])
    _sorted_raw, _encoded, summary = builder._sorted_p6_action_choice(raw_actions)

    assert summary["original_action_stats"]["duplicate_pair_tile_count"] == 1
    assert summary["proof_status"] == "failed_duplicate_pair_tile_targets"
