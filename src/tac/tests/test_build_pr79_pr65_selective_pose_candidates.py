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
import numpy as np

from tac.qp1_pose_codec import encode_qp1


REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "experiments" / "build_pr79_pr65_selective_pose_candidates.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("build_pr79_pr65_selective_pose_candidates_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, members: list[tuple[str, bytes]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)
    return path


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


def _pseudo_bytes(n: int, seed: int) -> bytes:
    out = bytearray()
    state = seed
    for _idx in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out.append((state >> 8) & 0xFF)
    return bytes(out)


def _p1d1_pose_br() -> bytes:
    streams: list[bytes] = []
    dims = [0, 1, 2]
    seed = 12345
    for dim in dims:
        stream = bytearray()
        for _idx in range(600):
            seed = (1103515245 * seed + 12345 + dim) & 0x7FFFFFFF
            step = seed % 5 if dim == 0 else (seed % 41) - 20
            stream.extend(_uleb(_zigzag(step)))
        streams.append(bytes(stream))
    raw = b"P1D1" + bytes([len(dims)])
    for dim, stream in zip(dims, streams):
        raw += bytes([dim]) + len(stream).to_bytes(2, "little")
    raw += b"".join(streams)
    return _br(raw)


def _dense_stream(magic: bytes, default: int) -> bytes:
    arr = np.full(600, default, dtype=np.uint8)
    return _br(magic + arr.tobytes())


def _pr65_fixture(path: Path) -> Path:
    streams = {
        "mask": _br(b"\x12\x00\x0a\x0a" + _pseudo_bytes(3000, 7)),
        "model": _br(b"QZS3" + _pseudo_bytes(3000, 11)),
        "pose": _p1d1_pose_br(),
        "post": _br(bytes([0, 1, 0, 2]) * 600),
        "shift": _dense_stream(b"SH4", 40),
        "frac": _dense_stream(b"FH1", 4),
        "frac2": _dense_stream(b"FH2", 4),
        "frac3": _dense_stream(b"FH3", 4),
        "bias": _dense_stream(b"BH1", 13),
        "region": _dense_stream(b"RH1", 0),
        "randmulti": _br(b"randmulti" + bytes(range(32))),
    }
    ordered = ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3", "bias", "region")
    payload = b"".join(len(streams[name]).to_bytes(3, "little") for name in ordered)
    payload += b"".join(streams[name] for name in ordered) + streams["randmulti"]
    return _stored_zip(path, [("x", payload)])


def _source_p3_archive(path: Path) -> Path:
    masks = _br(b"\x12\x00\x0a\x0a" + bytes((idx * 7) % 256 for idx in range(2048)))
    renderer = _br(b"QZS3" + bytes((idx * 11) % 256 for idx in range(2048)))
    actions_raw = b"".join(
        pair.to_bytes(2, "little") + bytes([pair % 8, (pair * 3) % 32])
        for pair in range(12)
    )
    actions = _br(actions_raw)
    poses = np.zeros((600, 6), dtype=np.float32)
    poses[:, 0] = 20.5
    pose_br = _br(encode_qp1(poses))
    payload = (
        b"P3"
        + struct.pack("<IHH", len(masks), len(renderer), len(actions))
        + masks
        + renderer
        + actions
        + pose_br
    )
    return _stored_zip(path, [("p", payload)])


def _pose_plan(path: Path, *, expected_benefit_proxy: float = 0.002) -> Path:
    coefs = [
        {
            "pair_index": idx,
            "delta_q": 1 if idx % 2 == 0 else -1,
            "delta_velocity": (1 if idx % 2 == 0 else -1) / 512.0,
            "raw_basis_delta_q": float(1 if idx % 2 == 0 else -1),
        }
        for idx in range(40)
    ]
    payload = {
        "ranked_candidates": [
            {
                "basis_id": "public_difference_pr65",
                "candidate_id": "fixture_pr65_pose_atoms",
                "expected_benefit_proxy": expected_benefit_proxy,
                "selected_coefs": coefs,
            }
        ]
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _exact_json(path: Path, *, score: float = 0.3145) -> Path:
    payload = {
        "archive_size_bytes": 1000,
        "avg_posenet_dist": 0.00049,
        "avg_segnet_dist": 0.00059,
        "canonical_score": score,
        "n_samples": 600,
        "provenance": {"archive_sha256": "abc"},
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def test_build_candidates_preserves_non_pose_streams_and_writes_manifests(tmp_path: Path) -> None:
    module = _load_tool()
    source = _source_p3_archive(tmp_path / "source.zip")
    pr65 = _pr65_fixture(tmp_path / "pr65.zip")
    plan = _pose_plan(tmp_path / "pose_plan.json")
    exact = _exact_json(tmp_path / "exact.json")

    matrix = module.build_candidates(
        pr79_s2_archive=source,
        pr65_archive=pr65,
        pose_plan_json=plan,
        pr79_s2_exact_json=exact,
        negative_exact_json=None,
        output_dir=tmp_path / "out",
        expected_pr79_s2_sha256=None,
        expected_pr65_sha256=None,
    )

    assert matrix["score_claim"] is False
    assert matrix["no_remote_dispatch_performed"] is True
    candidate_ids = {row["candidate_id"] for row in matrix["candidate_matrix"]}
    assert "pr79_s2_pr65_pose_wholesale_qp1" in candidate_ids
    assert "pr79_s2_pr65_pose_atoms_top040" in candidate_ids

    atom_manifest = json.loads(
        (tmp_path / "out" / "pr79_s2_pr65_pose_atoms_top040" / "manifest.json").read_text()
    )
    assert atom_manifest["runtime_parse_validation"]["status"] == "passed"
    assert atom_manifest["runtime_parse_validation"]["changed_decoded_streams_vs_source"] == [
        "optimized_poses.qp1"
    ]
    assert atom_manifest["stream_closure"]["non_pose_streams_preserved"] is True
    assert atom_manifest["pose_change"]["changed_pose_word_count"] == 40
    assert atom_manifest["proxy_screen"]["high_ev_enough_for_exact_eval"] is True


def test_nearby_exact_negative_blocks_dispatch_recommendation(tmp_path: Path) -> None:
    module = _load_tool()
    source = _source_p3_archive(tmp_path / "source.zip")
    pr65 = _pr65_fixture(tmp_path / "pr65.zip")
    plan = _pose_plan(tmp_path / "pose_plan.json", expected_benefit_proxy=0.002)
    exact = _exact_json(tmp_path / "exact.json")
    negative = _exact_json(tmp_path / "negative.json", score=0.3188)

    matrix = module.build_candidates(
        pr79_s2_archive=source,
        pr65_archive=pr65,
        pose_plan_json=plan,
        pr79_s2_exact_json=exact,
        negative_exact_json=negative,
        output_dir=tmp_path / "out",
        expected_pr79_s2_sha256=None,
        expected_pr65_sha256=None,
    )

    assert matrix["dispatch_decision"]["exact_eval_recommended"] is False
    for row in matrix["candidate_matrix"]:
        assert row["dispatch_recommendation"]["remote_dispatch_performed"] is False
        assert row["high_ev_enough_for_exact_eval"] is False
