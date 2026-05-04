from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import pytest

from tac.qp1_pose_codec import encode_qp1


REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "experiments" / "plan_pr65_henosis_stream_transfer.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("plan_pr65_henosis_stream_transfer_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def _br(data: bytes) -> bytes:
    return brotli.compress(data, quality=0, lgwin=10)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _uleb(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _p1d1_pose_br() -> bytes:
    streams: list[bytes] = []
    dims = [0, 1, 2]
    seed = 12345
    for dim in dims:
        stream = bytearray()
        for _idx in range(600):
            seed = (1103515245 * seed + 12345 + dim) & 0x7FFFFFFF
            delta = seed % 3 if dim == 0 else (seed % 31) - 15
            stream.extend(_uleb(_zigzag(delta)))
        streams.append(bytes(stream))
    header = b"P1D1" + bytes([len(dims)])
    for dim, stream in zip(dims, streams):
        header += bytes([dim]) + len(stream).to_bytes(2, "little")
    raw = header + b"".join(streams)
    return _br(raw)


def _dense_stream(magic: bytes, default: int, overrides: dict[int, int] | None = None) -> bytes:
    arr = np.full(600, default, dtype=np.uint8)
    for idx, value in (overrides or {}).items():
        arr[idx] = value
    return _br(magic + arr.tobytes())


def _pr65_fixture_archive(path: Path) -> Path:
    core_mask = _br(bytes((idx * 17 + 3) % 256 for idx in range(6000)))
    core_model = _br(b"QH0" + bytes((idx * 29 + 11) % 256 for idx in range(6000)))
    pose = _p1d1_pose_br()
    streams = {
        "mask": core_mask,
        "model": core_model,
        "pose": pose,
        "post": _br(bytes([0, 1, 0, 2]) * 600),
        "shift": _dense_stream(b"SH4", 40),
        "frac": _dense_stream(b"FH1", 4),
        "frac2": _dense_stream(b"FH2", 4),
        "frac3": _dense_stream(b"FH3", 4),
        "bias": _dense_stream(b"BH1", 13, {0: 14, 1: 12, 2: 15}),
        "region": _dense_stream(b"RH1", 0, {0: 1, 1: 2}),
        "randmulti": _br(b"randmulti" + bytes(range(64))),
    }
    ordered = ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3", "bias", "region")
    header = b"".join(len(streams[name]).to_bytes(3, "little") for name in ordered)
    payload = header + b"".join(streams[name] for name in ordered) + streams["randmulti"]
    _stored_zip(path, [("x", payload)])
    return path


def _actions(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for pair, tile, action in records:
        out += pair.to_bytes(2, "little") + bytes([tile, action])
    return bytes(out)


def _p6_fixture_archive(module: Any, path: Path, *, pose_shift: float = 0.0) -> Path:
    mask_br = _br(b"\x12\x00\x0a\x0a" + bytes((idx * 7) % 256 for idx in range(256)))
    renderer_br = _br(b"QZS3" + bytes((idx * 13) % 256 for idx in range(512)))
    raw_actions = _actions([(0, 1, 2), (1, 2, 3), (4, 3, 4)])
    actions_br = _br(module.encode_delta_varint_actions(raw_actions))
    poses = np.zeros((600, 6), dtype=np.float32)
    poses[:, 0] = 20.0 + pose_shift
    pose_br = _br(encode_qp1(poses))
    payload = (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), len(raw_actions) // 4)
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )
    _stored_zip(path, [("p", payload)])
    return path


def _trace(path: Path, rows: list[tuple[int, float, float]]) -> Path:
    samples = []
    for pair, seg, pose in rows:
        samples.append(
            {
                "pair_index": pair,
                "score_seg_contribution_exact": seg,
                "score_pose_contribution_first_order": pose,
                "score_combined_contribution_first_order": seg + pose,
            }
        )
    path.write_text(json.dumps({"samples": samples}) + "\n")
    return path


def test_strict_zip_inventory_rejects_duplicate_member(tmp_path: Path) -> None:
    module = _load_tool()
    archive = tmp_path / "dup.zip"
    _stored_zip(archive, [("p", b"one"), ("p", b"two")])

    with pytest.raises(module.HenosisTransferError, match="duplicate ZIP member"):
        module.strict_zip_inventory(archive)


def test_parse_pr65_henosis_fixture_and_decode_pose(tmp_path: Path) -> None:
    module = _load_tool()
    archive = _pr65_fixture_archive(tmp_path / "pr65.zip")

    anatomy = module.parse_pr65_henosis_archive(archive, expected_sha256=None)
    pose = module.decode_pr65_p1d1_pose(anatomy["_segments_bytes"]["pose"])

    assert anatomy["zip_inventory"]["strict_zip"] is True
    assert anatomy["payload_header"]["header_bytes"] == 30
    assert anatomy["payload_header"]["qpost_encoded_bytes"] > 0
    assert anatomy["segments"]["bias"]["decoded_bytes"] == 603
    assert pose.shape == (600, 6)
    assert np.isfinite(pose).all()


def test_build_candidate_matrix_writes_non_dispatchable_manifests(tmp_path: Path) -> None:
    module = _load_tool()
    pr65 = _pr65_fixture_archive(tmp_path / "pr65.zip")
    c091 = _p6_fixture_archive(module, tmp_path / "c091.zip", pose_shift=0.0)
    c089 = _p6_fixture_archive(module, tmp_path / "c089.zip", pose_shift=0.25)
    c091_trace = _trace(tmp_path / "c091_trace.json", [(0, 0.010, 0.003), (1, 0.005, 0.002), (2, 0.001, 0.001)])
    pr65_trace = _trace(tmp_path / "pr65_trace.json", [(0, 0.002, 0.002), (1, 0.001, 0.003), (2, 0.002, 0.001)])
    c089_trace = _trace(tmp_path / "c089_trace.json", [(0, 0.009, 0.003)])

    summary = module.build_candidate_matrix(
        pr65_archive=pr65,
        c091_archive=c091,
        c089_archive=c089,
        pr65_trace=pr65_trace,
        c091_trace=c091_trace,
        c089_trace=c089_trace,
        output_dir=tmp_path / "out",
        qpost_specs=(
            module.QPostSpec(
                "fixture_c091_pr65_bias_top002",
                "c091",
                ("bias",),
                "seg",
                2,
                "fixture",
            ),
        ),
        expected_pr65_sha256=None,
        expected_c091_sha256=None,
        expected_c089_sha256=None,
    )

    assert summary["score_claim"] is False
    assert summary["remote_dispatch"] == {"dispatched": False, "lightning_state_touched": False}
    assert {row["candidate_id"] for row in summary["candidates"]} >= {
        "c091_renderer_pose_c089_actions_p6_control",
        "c091_pr65_pose_qp1_c089_actions_p6",
        "fixture_c091_pr65_bias_top002",
    }
    manifest_path = tmp_path / "out" / "fixture_c091_pr65_bias_top002" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["selected_pair_indices"] == [0, 1]
    assert manifest["dispatch_recommendation"]["class"] == "non_dispatchable_planning_artifact"
    with zipfile.ZipFile(manifest["archive"]) as zf:
        assert [info.filename for info in zf.infolist()] == ["p", "qpost.bin"]
        assert zf.read("qpost.bin").startswith(b"QPS1")
