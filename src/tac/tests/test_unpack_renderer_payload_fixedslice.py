# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import random
import struct
import sys
import types
import zipfile
from pathlib import Path

import brotli
import pytest

from tac.qp1_pose_codec import QPV1DimensionStream, QPV1Payload


REPO = Path(__file__).resolve().parents[3]
UNPACK_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"
INFLATE_RENDERER_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"
PR81_ARCHIVE = REPO / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/archive.zip"
PR81_PROFILE = (
    REPO
    / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/pr81_qma9_semantic_range_mask_profile.json"
)


def _load_unpacker():
    spec = importlib.util.spec_from_file_location("unpack_renderer_payload_fixedslice_test", UNPACK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_inflate_renderer():
    spec = importlib.util.spec_from_file_location("inflate_renderer_fixedslice_test", INFLATE_RENDERER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _deterministic_bytes(size: int, *, seed: int, prefix: bytes, vlq_safe: bool = False) -> bytes:
    rng = random.Random(seed)
    limit = 128 if vlq_safe else 256
    return prefix + bytes(rng.randrange(0, limit) for _ in range(size - len(prefix)))


def _brotli_stream_with_exact_len(
    target_len: int,
    *,
    seed: int,
    prefix: bytes,
    vlq_safe: bool = False,
) -> tuple[bytes, bytes]:
    for raw_len in range(max(len(prefix), target_len - 32), target_len + 1):
        raw = _deterministic_bytes(raw_len, seed=seed, prefix=prefix, vlq_safe=vlq_safe)
        compressed = brotli.compress(raw, quality=0, mode=brotli.MODE_GENERIC, lgwin=22)
        if len(compressed) == target_len:
            return raw, compressed
    raise AssertionError(f"could not synthesize Brotli stream of length {target_len}")


def _brotli_action_stream_with_exact_len(target_len: int, *, seed: int) -> tuple[bytes, bytes]:
    for raw_len in range(4, target_len + 1, 4):
        rng = random.Random(seed + raw_len)
        raw = bytearray()
        while len(raw) < raw_len:
            pair = rng.randrange(0, 600)
            tile = rng.randrange(0, 128)
            action = rng.randrange(0, 128)
            raw.extend(pair.to_bytes(2, "little"))
            raw.append(tile)
            raw.append(action)
        compressed = brotli.compress(bytes(raw), quality=0, mode=brotli.MODE_GENERIC, lgwin=22)
        if len(compressed) == target_len:
            return bytes(raw), compressed
    raise AssertionError(f"could not synthesize Brotli action stream of length {target_len}")


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


@pytest.mark.skipif(
    not (PR81_ARCHIVE.exists() and PR81_PROFILE.exists()),
    reason="PR81 public intake artifacts missing",
)
def test_actual_pr81_reordered_qzs3_model_payload_restores_to_qzs3() -> None:
    import json

    inflate_renderer = _load_inflate_renderer()
    profile = json.loads(PR81_PROFILE.read_text(encoding="utf-8"))
    split = profile["split_constants"]
    range_mask_bytes = int(split["RANGE_MASK_BYTES"])
    model_bytes = int(split["SPLIT_MODEL_REORDERED_BYTES"])
    with zipfile.ZipFile(PR81_ARCHIVE, "r") as zf:
        payload = zf.read("p")
    model_payload = payload[range_mask_bytes : range_mask_bytes + model_bytes]

    restored = inflate_renderer._restore_pr81_reordered_qzs3_model_payload(model_payload)

    assert restored.startswith(b"QZS3")
    assert int.from_bytes(restored[4:6], "little") == 32
    assert len(restored) > model_bytes


def _sg2_action_stream_with_exact_len(target_len: int) -> tuple[bytes, bytes, bytes]:
    """Synthesize an observed PR75-minp SG2 action stream at exact byte size."""
    for group_count in range(1, 200):
        raw = bytearray(b"SG2")
        decoded = bytearray()
        for group in range(group_count):
            tile = (83 + group) % 141
            count = 1 + (group % 3 == 0)
            raw += _uvarint(tile) + _uvarint(count)
            frame = 30 + group * 5
            for idx in range(count):
                delta = frame if idx == 0 else 1
                action = (2 + group + idx) % 108
                raw += _uvarint(delta) + bytes([action])
                frame = delta if idx == 0 else frame + delta
                decoded += frame.to_bytes(2, "little") + bytes([tile, action])
        for quality in range(12):
            compressed = brotli.compress(
                bytes(raw),
                quality=quality,
                mode=brotli.MODE_GENERIC,
                lgwin=10,
            )
            if len(compressed) == target_len:
                return bytes(decoded), bytes(raw), compressed
    raise AssertionError(f"could not synthesize SG2 Brotli stream of length {target_len}")


def _nrv_payload(*, version: int = 2, weight_bytes: bytes = b"nerv-weights") -> bytes:
    header = bytearray(b"NRV1")
    header.extend(struct.pack("<HHHHHH", version, 4, 16, 5, 2, 0))
    header.extend(struct.pack("<Q", len(weight_bytes)))
    if version == 2:
        header.extend(struct.pack("<Q", 0))
    return bytes(header) + weight_bytes


def test_public_pr67_fixedslice_parser_uses_content_validation_not_total_length_bucket() -> None:
    unpacker = _load_unpacker()
    mask_raw, mask_br = _brotli_stream_with_exact_len(
        219_472,
        seed=1,
        prefix=b"\x12\x00\x0a\x0a",
    )
    renderer_raw, renderer_br = _brotli_stream_with_exact_len(
        56_093,
        seed=2,
        prefix=b"QZS3",
    )
    pose_raw, pose_br = _brotli_stream_with_exact_len(
        677,
        seed=3,
        prefix=b"QP1" + (5120).to_bytes(2, "little"),
        vlq_safe=True,
    )

    # This payload length falls into the broad repo-generated C067 bucket where
    # 55,965-byte renderers are tried first. The parser must reject that wrong
    # boundary and recover the true 56,093-byte renderer by Brotli/QZS3/QP1
    # content validation before inflate spends a GPU.
    header, decoded = unpacker._parse_payload(mask_br + renderer_br + pose_br)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr67_qzs3_qp1_fixed_slices"
    assert members["masks.mkv"]["bytes"] == len(mask_br)
    assert members["renderer.bin"]["bytes"] == len(renderer_br)
    assert members["optimized_poses.bin"]["bytes"] == len(pose_br)
    assert decoded["masks.mkv"] == mask_raw
    assert decoded["renderer.bin"] == renderer_raw
    assert decoded["optimized_poses.bin"]


def test_public_pr67_nerv_fixedslice_parser_requires_nrv_qzs3_qp1_contract() -> None:
    unpacker = _load_unpacker()
    masks_nrv = _nrv_payload(weight_bytes=b"\x11" * 32)
    renderer_raw, renderer_br = _brotli_stream_with_exact_len(
        56_093,
        seed=4,
        prefix=b"QZS3",
    )
    pose_raw, pose_br = _brotli_stream_with_exact_len(
        677,
        seed=5,
        prefix=b"QP1" + (5120).to_bytes(2, "little"),
        vlq_safe=True,
    )

    header, decoded = unpacker._parse_payload(masks_nrv + renderer_br + pose_br)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr67_nerv_qzs3_qp1_fixed_slices"
    assert members["masks.nrv"]["bytes"] == len(masks_nrv)
    assert members["renderer.bin"]["bytes"] == len(renderer_br)
    assert members["optimized_poses.bin"]["bytes"] == len(pose_br)
    assert decoded["masks.nrv"] == masks_nrv
    assert decoded["renderer.bin"] == renderer_raw
    assert decoded["optimized_poses.bin"]
    assert pose_raw.startswith(b"QP1")


def test_public_pr67_nerv_single_blob_without_boundary_contract_fails_clearly() -> None:
    unpacker = _load_unpacker()
    payload = _nrv_payload(weight_bytes=b"\x22" * 32) + b"not-a-supported-renderer-pose-tail"

    with pytest.raises(ValueError, match="PR67/C067 NeRV single-blob payload.*no valid"):
        unpacker._parse_payload(payload)


def test_public_pr67_nerv_single_blob_with_malformed_nrv_header_fails_clearly() -> None:
    unpacker = _load_unpacker()

    with pytest.raises(ValueError, match="malformed masks\\.nrv"):
        unpacker._parse_payload(b"NRV1")


def test_public_pr81_qma9_payload_parser_emits_typed_runtime_members() -> None:
    unpacker = _load_unpacker()
    archive = REPO / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/archive.zip"
    if not archive.exists():
        pytest.skip("public PR81 intake archive fixture is not present")
    with zipfile.ZipFile(archive, "r") as zf:
        payload = zf.read("p")
    qma9 = payload[:159_011]
    model = payload[159_011:159_011 + 55_725]
    pose_br = payload[159_011 + 55_725:159_011 + 55_725 + 899]
    router = payload[-225:]

    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr81_qma9_reordered_qzs3_qp1_router_fixed_slices"
    assert decoded["masks.qma9"] == qma9
    assert decoded["renderer.bin"].startswith(b"Q81R")
    assert decoded["renderer.bin"][4:] == model
    assert decoded["optimized_poses.qp1"] == brotli.decompress(pose_br)
    assert decoded["router_actions.3bit"] == router
    assert members["masks.qma9"]["codec"] == "qma9_adaptive9_binary_range_mask"
    assert members["router_actions.3bit"]["decoded_bytes"] == 600


def test_public_pr84_qma9_no_router_payload_parser_emits_typed_runtime_members() -> None:
    unpacker = _load_unpacker()
    archive = REPO / "experiments/results/top_submission_reverse_engineering_20260503_pr84/archive.zip"
    if not archive.exists():
        pytest.skip("public PR84 intake archive fixture is not present")
    with zipfile.ZipFile(archive, "r") as zf:
        payload = zf.read("p")
    qma9 = payload[:159_011]
    model = payload[159_011:159_011 + 55_725]
    pose_br = payload[159_011 + 55_725:159_011 + 55_725 + 899]

    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_qma9_reordered_qzs3_qp1_no_router_fixed_slices"
    assert decoded["masks.qma9"] == qma9
    assert decoded["renderer.bin"].startswith(b"Q81R")
    assert decoded["renderer.bin"][4:] == model
    assert decoded["optimized_poses.qp1"] == brotli.decompress(pose_br)
    assert "router_actions.3bit" not in decoded
    assert "router_actions.3bit" not in members
    assert members["masks.qma9"]["codec"] == "qma9_adaptive9_binary_range_mask"


def test_public_pr75_p3_segactions_parser_decodes_charged_tile_actions() -> None:
    unpacker = _load_unpacker()
    mask_raw = _deterministic_bytes(128, seed=10, prefix=b"\x12\x00\x0a\x0a")
    renderer_raw = _deterministic_bytes(96, seed=11, prefix=b"QZS3")
    actions_raw = (
        (33).to_bytes(2, "little") + bytes([109, 92])
        + (36).to_bytes(2, "little") + bytes([109, 93])
    )
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"\x01\x02\x03"
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    actions_br = brotli.compress(actions_raw, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(actions_br))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )

    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_p3"
    assert members["seg_tile_actions.bin"]["bytes"] == len(actions_br)
    assert members["seg_tile_actions.bin"]["decoded_bytes"] == len(actions_raw)
    assert decoded["masks.mkv"] == mask_raw
    assert decoded["renderer.bin"] == renderer_raw
    assert decoded["seg_tile_actions.bin"] == actions_raw
    assert members["optimized_poses.qp1"]["decoded_bytes"] == len(pose_raw)
    assert decoded["optimized_poses.qp1"] == pose_raw


def test_public_pr75_current_release_fixedslice_parser_tries_observed_model_lengths() -> None:
    unpacker = _load_unpacker()
    mask_raw, mask_br = _brotli_stream_with_exact_len(
        219_472,
        seed=110,
        prefix=b"\x12\x00\x0a\x0a",
    )
    renderer_raw, renderer_br = _brotli_stream_with_exact_len(
        55_914,
        seed=111,
        prefix=b"QZS3",
    )
    actions_raw, actions_br = _brotli_action_stream_with_exact_len(236, seed=112)
    pose_raw, pose_br = _brotli_stream_with_exact_len(
        677,
        seed=113,
        prefix=b"QP1" + (5120).to_bytes(2, "little"),
        vlq_safe=True,
    )

    header, decoded = unpacker._parse_payload(mask_br + renderer_br + actions_br + pose_br)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    assert members["renderer.bin"]["bytes"] == 55_914
    assert members["seg_tile_actions.bin"]["bytes"] == 236
    assert decoded["masks.mkv"] == mask_raw
    assert decoded["renderer.bin"] == renderer_raw
    assert decoded["seg_tile_actions.bin"] == actions_raw
    assert decoded["optimized_poses.qp1"] == pose_raw


def test_public_pr75_minp_fixedslice_parser_decodes_sg2_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpacker = _load_unpacker()
    mask_br = b"M" * 219_472
    renderer_br = b"R" * 55_756
    actions_br = b"A" * 255
    pose_br = b"P" * 898
    mask_raw = b"\x12\x00\x0a\x0a" + b"mask"
    renderer_raw = b"QZS3" + b"renderer"
    actions_decoded = (33).to_bytes(2, "little") + bytes([109, 2])
    actions_sg2 = b"SG2" + _uvarint(109) + _uvarint(1) + _uvarint(33) + bytes([2])
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"pose"

    def fake_decompress(data: bytes) -> bytes:
        if data == mask_br:
            return mask_raw
        if data == renderer_br:
            return renderer_raw
        if data == actions_br:
            return actions_sg2
        if data == pose_br:
            return pose_raw
        raise ValueError("unexpected Brotli payload")

    monkeypatch.setitem(sys.modules, "brotli", types.SimpleNamespace(decompress=fake_decompress))

    payload = mask_br + renderer_br + actions_br + pose_br
    assert len(payload) == 276_381
    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    assert members["renderer.bin"]["bytes"] == 55_756
    assert members["seg_tile_actions.bin"]["bytes"] == 255
    assert members["seg_tile_actions.bin"]["decoded_bytes"] == len(actions_decoded)
    assert actions_sg2.startswith(b"SG2")
    assert decoded["masks.mkv"] == mask_raw
    assert decoded["renderer.bin"] == renderer_raw
    assert decoded["seg_tile_actions.bin"] == actions_decoded
    assert decoded["optimized_poses.qp1"] == pose_raw


def test_public_pr77_fixedslice_parser_decodes_tile_delta_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpacker = _load_unpacker()
    mask_br = b"M" * 219_472
    renderer_br = b"R" * 55_756
    actions_br = b"A" * 325
    pose_br = b"P" * 898
    mask_raw = b"\x12\x00\x0a\x0a" + b"mask"
    renderer_raw = b"QZS3" + b"renderer"
    actions_decoded = (
        (33).to_bytes(2, "little") + bytes([109, 2])
        + (36).to_bytes(2, "little") + bytes([109, 3])
    )
    actions_grouped = (
        _uvarint(109) + _uvarint(2)
        + _uvarint(33) + bytes([2])
        + _uvarint(3) + bytes([3])
    )
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"pose"

    def fake_decompress(data: bytes) -> bytes:
        if data == mask_br:
            return mask_raw
        if data == renderer_br:
            return renderer_raw
        if data == actions_br:
            return actions_grouped
        if data == pose_br:
            return pose_raw
        raise ValueError("unexpected Brotli payload")

    monkeypatch.setitem(sys.modules, "brotli", types.SimpleNamespace(decompress=fake_decompress))

    payload = mask_br + renderer_br + actions_br + pose_br
    assert len(payload) == 276_451
    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    assert members["renderer.bin"]["bytes"] == 55_756
    assert members["seg_tile_actions.bin"]["bytes"] == 325
    assert members["seg_tile_actions.bin"]["decoded_bytes"] == len(actions_decoded)
    assert decoded["masks.mkv"] == mask_raw
    assert decoded["renderer.bin"] == renderer_raw
    assert decoded["seg_tile_actions.bin"] == actions_decoded
    assert decoded["optimized_poses.qp1"] == pose_raw


def test_public_pr77_qp19_qpv1_parser_decodes_multidim_pose() -> None:
    unpacker = _load_unpacker()
    mask_raw = b"\x12\x00\x0a\x0a" + b"mask"
    renderer_raw = b"QZS3" + b"renderer"
    qpv1 = QPV1Payload(
        count=2,
        pose_dim=6,
        streams=(
            QPV1DimensionStream(0, 20.0, 512.0, (5120, 5136)),
            QPV1DimensionStream(2, -0.5, 2048.0, (0, 4)),
        ),
    )
    pose_raw = qpv1.to_bytes()
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"QP19"
        + bytes([1, 0])
        + struct.pack("<III", len(mask_br), len(renderer_br), len(pose_br))
        + mask_br
        + renderer_br
        + pose_br
    )

    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr77_qp19_qzs3_qpv1_v1"
    assert members["optimized_poses.bin"]["codec"] == "public_qpv1_brotli"
    assert decoded["masks.mkv"] == mask_raw
    assert decoded["renderer.bin"] == renderer_raw
    values = struct.unpack("<" + "e" * 12, decoded["optimized_poses.bin"])
    assert values[0] == pytest.approx(30.0)
    assert values[6] == pytest.approx(30.0 + 16.0 / 512.0)
    assert values[2] == pytest.approx(-0.5)
    assert values[8] == pytest.approx(-0.5 + 4.0 / 2048.0)


def test_public_pr77_qp19_rejects_bad_lengths_and_unknown_pose_magic() -> None:
    unpacker = _load_unpacker()
    mask_raw = b"\x12\x00\x0a\x0a" + b"mask"
    renderer_raw = b"QZS3" + b"renderer"
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    pose_br = brotli.compress(b"BAD!" + b"pose", quality=0)
    payload = (
        b"QP19"
        + bytes([1, 0])
        + struct.pack("<III", len(mask_br), len(renderer_br), len(pose_br) + 1)
        + mask_br
        + renderer_br
        + pose_br
    )
    with pytest.raises(ValueError, match="QP19 payload length mismatch"):
        unpacker._parse_payload(payload)

    payload = (
        b"QP19"
        + bytes([1, 0])
        + struct.pack("<III", len(mask_br), len(renderer_br), len(pose_br))
        + mask_br
        + renderer_br
        + pose_br
    )
    with pytest.raises(ValueError, match="failed QPV1 magic"):
        unpacker._parse_payload(payload)


def test_public_pr79_fixedslice_parser_decodes_large_sg2_action_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpacker = _load_unpacker()
    mask_br = b"M" * 219_472
    renderer_br = b"R" * 55_756
    actions_br = b"A" * 1_162
    pose_br = b"P" * 898
    mask_raw = b"\x12\x00\x0a\x0a" + b"mask"
    renderer_raw = b"QZS3" + b"renderer"
    actions_decoded = (
        (12).to_bytes(2, "little") + bytes([83, 92])
        + (16).to_bytes(2, "little") + bytes([83, 7])
    )
    actions_sg2 = (
        b"SG2"
        + _uvarint(83) + _uvarint(2)
        + _uvarint(12) + bytes([92])
        + _uvarint(4) + bytes([7])
    )
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"pose"

    def fake_decompress(data: bytes) -> bytes:
        if data == mask_br:
            return mask_raw
        if data == renderer_br:
            return renderer_raw
        if data == actions_br:
            return actions_sg2
        if data == pose_br:
            return pose_raw
        raise ValueError("unexpected Brotli payload")

    monkeypatch.setitem(sys.modules, "brotli", types.SimpleNamespace(decompress=fake_decompress))

    payload = mask_br + renderer_br + actions_br + pose_br
    assert len(payload) == 277_288
    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    assert members["renderer.bin"]["bytes"] == 55_756
    assert members["seg_tile_actions.bin"]["bytes"] == 1_162
    assert members["seg_tile_actions.bin"]["decoded_bytes"] == len(actions_decoded)
    assert decoded["masks.mkv"] == mask_raw
    assert decoded["renderer.bin"] == renderer_raw
    assert decoded["seg_tile_actions.bin"] == actions_decoded
    assert decoded["optimized_poses.qp1"] == pose_raw


def test_public_pr75_fixedslice_parser_accepts_pr77_actions_with_other_observed_model_len(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpacker = _load_unpacker()
    mask_br = b"M" * 219_472
    renderer_br = b"R" * 56_034
    actions_br = b"A" * 325
    pose_br = b"P" * 899
    mask_raw = b"\x12\x00\x0a\x0a" + b"mask"
    renderer_raw = b"QZS3" + b"renderer"
    actions_decoded = (
        (33).to_bytes(2, "little") + bytes([109, 2])
        + (36).to_bytes(2, "little") + bytes([109, 3])
    )
    actions_grouped = (
        _uvarint(109) + _uvarint(2)
        + _uvarint(33) + bytes([2])
        + _uvarint(3) + bytes([3])
    )
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"pose"

    def fake_decompress(data: bytes) -> bytes:
        if data == mask_br:
            return mask_raw
        if data == renderer_br:
            return renderer_raw
        if data == actions_br:
            return actions_grouped
        if data == pose_br:
            return pose_raw
        raise ValueError("unexpected Brotli payload")

    monkeypatch.setitem(sys.modules, "brotli", types.SimpleNamespace(decompress=fake_decompress))

    payload = mask_br + renderer_br + actions_br + pose_br
    assert len(payload) == 276_730
    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_fixed_slices"
    assert members["renderer.bin"]["bytes"] == 56_034
    assert members["seg_tile_actions.bin"]["bytes"] == 325
    assert members["seg_tile_actions.bin"]["decoded_bytes"] == len(actions_decoded)
    assert decoded["masks.mkv"] == mask_raw
    assert decoded["renderer.bin"] == renderer_raw
    assert decoded["seg_tile_actions.bin"] == actions_decoded
    assert decoded["optimized_poses.qp1"] == pose_raw


def _seg_tile_action_dict_raw(values: list[tuple[float, float, float]]) -> bytes:
    body = bytearray()
    for rgb in values:
        body.extend(struct.pack("<fff", *rgb))
    return b"TAD1" + struct.pack("<HH", 1, len(values)) + bytes(body)


def _pack_p5_record(pair: int, tile: int, action: int) -> bytes:
    word = pair | (tile << 10) | (action << 18)
    return bytes([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF])


def test_public_pr75_p4_custom_dict_parser_decodes_charged_dictionary() -> None:
    unpacker = _load_unpacker()
    mask_raw = _deterministic_bytes(128, seed=20, prefix=b"\x12\x00\x0a\x0a")
    renderer_raw = _deterministic_bytes(96, seed=21, prefix=b"QZS3")
    action_dict_raw = _seg_tile_action_dict_raw([(2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)])
    actions_raw = (
        (33).to_bytes(2, "little") + bytes([109, 0])
        + (36).to_bytes(2, "little") + bytes([109, 1])
    )
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"\x01\x02\x03"
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    dict_br = brotli.compress(action_dict_raw, quality=0)
    actions_br = brotli.compress(actions_raw, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"P4"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(dict_br), len(actions_br))
        + mask_br
        + renderer_br
        + dict_br
        + actions_br
        + pose_br
    )

    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_p4_custom_dict"
    assert members["seg_tile_action_dict.bin"]["decoded_bytes"] == len(action_dict_raw)
    assert decoded["seg_tile_action_dict.bin"] == action_dict_raw
    assert decoded["seg_tile_actions.bin"] == actions_raw


def test_public_pr75_p5_packed_custom_dict_parser_decodes_runtime_records() -> None:
    unpacker = _load_unpacker()
    mask_raw = _deterministic_bytes(128, seed=30, prefix=b"\x12\x00\x0a\x0a")
    renderer_raw = _deterministic_bytes(96, seed=31, prefix=b"QZS3")
    action_dict_raw = _seg_tile_action_dict_raw([(2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)])
    packed_actions = _pack_p5_record(33, 109, 0) + _pack_p5_record(36, 109, 1)
    expected_actions = (
        (33).to_bytes(2, "little") + bytes([109, 0])
        + (36).to_bytes(2, "little") + bytes([109, 1])
    )
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"\x01\x02\x03"
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    dict_br = brotli.compress(action_dict_raw, quality=0)
    actions_br = brotli.compress(packed_actions, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"P5"
        + struct.pack(
            "<IHHHH",
            len(mask_br),
            len(renderer_br),
            len(dict_br),
            len(actions_br),
            2,
        )
        + mask_br
        + renderer_br
        + dict_br
        + actions_br
        + pose_br
    )

    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_p5_packed_custom_dict"
    assert members["seg_tile_actions.bin"]["decoded_bytes"] == len(expected_actions)
    assert decoded["seg_tile_action_dict.bin"] == action_dict_raw
    assert decoded["seg_tile_actions.bin"] == expected_actions


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


def test_public_pr75_p6_delta_varint_parser_decodes_runtime_records() -> None:
    unpacker = _load_unpacker()
    mask_raw = _deterministic_bytes(128, seed=40, prefix=b"\x12\x00\x0a\x0a")
    renderer_raw = _deterministic_bytes(96, seed=41, prefix=b"QZS3")
    expected_actions = (
        (33).to_bytes(2, "little") + bytes([109, 92])
        + (36).to_bytes(2, "little") + bytes([109, 93])
    )
    delta_actions = (
        _uleb128(33) + bytes([109, 92])
        + _uleb128(3) + bytes([109, 93])
    )
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"\x01\x02\x03"
    mask_br = brotli.compress(mask_raw, quality=0)
    renderer_br = brotli.compress(renderer_raw, quality=0)
    actions_br = brotli.compress(delta_actions, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    payload = (
        b"P6"
        + struct.pack(
            "<IHHH",
            len(mask_br),
            len(renderer_br),
            len(actions_br),
            2,
        )
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )

    header, decoded = unpacker._parse_payload(payload)

    members = {item["name"]: item for item in header["members"]}
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
    assert members["seg_tile_actions.bin"]["decoded_bytes"] == len(expected_actions)
    assert decoded["seg_tile_actions.bin"] == expected_actions
    assert members["optimized_poses.qp1"]["decoded_bytes"] == len(pose_raw)
    assert decoded["optimized_poses.qp1"] == pose_raw
