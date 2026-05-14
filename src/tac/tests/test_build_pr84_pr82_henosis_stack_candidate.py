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
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments/build_pr84_pr82_henosis_stack_candidate.py"
PR84_ARCHIVE = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr84/archive.zip"
PR84_SOURCE_INFLATE = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr84/sources/inflate.py"
PR82_ARCHIVE = REPO_ROOT / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/archive.zip"
PR82_REPLAY = (
    REPO_ROOT
    / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/replay_submission/inflate.py"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("build_pr84_pr82_henosis_stack_candidate", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _vlq(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _qp1_pose_stream(pair_count: int = 600) -> bytes:
    raw = bytearray(b"QP1")
    raw.extend((7200).to_bytes(2, "little"))
    for _ in range(pair_count - 1):
        raw.extend(_vlq(0))
    return brotli.compress(bytes(raw))


def _qma9_payload(size: int) -> bytes:
    header = struct.pack("<4sIIII", b"QMA9", 600, 512, 384, size - 20)
    return header + bytes(size - len(header))


def _source_inflate(
    path: Path,
    *,
    range_mask_bytes: int,
    model_bytes: int,
    pose_bytes: int,
    router_bytes: int = 225,
) -> None:
    packed = max(1, model_bytes // 3)
    scales = max(1, model_bytes // 4)
    tail = model_bytes - packed - scales
    path.write_text(
        "\n".join(
            [
                f"RANGE_MASK_BYTES = {range_mask_bytes}",
                f"POSE_STREAM_BYTES = {pose_bytes}",
                f"SPLIT_MODEL_PACKED_REORDERED_BR_BYTES = {packed}",
                f"SPLIT_MODEL_SCALES_REORDERED_BR_BYTES = {scales}",
                f"SPLIT_MODEL_TAIL_REORDERED_BR_BYTES = {tail}",
                "SPLIT_MODEL_REORDERED_BYTES = (",
                "    SPLIT_MODEL_PACKED_REORDERED_BR_BYTES",
                "    + SPLIT_MODEL_SCALES_REORDERED_BR_BYTES",
                "    + SPLIT_MODEL_TAIL_REORDERED_BR_BYTES",
                ")",
                f"ROUTER_ACTION_BYTES = {router_bytes}",
            ]
        ),
        encoding="utf-8",
    )


def _randmulti_sparse_row(indices: list[int], values: list[int]) -> bytes:
    out = bytearray([len(indices)])
    previous = -1
    for index in indices:
        out.extend(_vlq(index - previous - 1))
        previous = index
    out.extend(values)
    return bytes(out)


def _synthetic_pr82_archive(path: Path, replay: Path) -> None:
    post = bytes([0] * 600 + [1] + [0] * 599 + [0] * 1200)
    shift = b"SH4" + bytes([40] * 600)
    frac = b"FH1" + bytes([4] * 600)
    frac2 = b"FH2" + bytes([4] * 600)
    frac3 = b"FH3" + bytes([4] * 600)
    bias = b"BH1" + bytes([13] * 600)
    region = b"RH1" + bytes([0] * 600)
    specs = [(24, 32, 1, 12), (12, 16, 1, 1)]
    randmulti_rows = []
    for row_index in range(12):
        if row_index == 0:
            randmulti_rows.append(_randmulti_sparse_row([5], [7]))
        else:
            randmulti_rows.append(_randmulti_sparse_row([], []))
    randmulti_rows.append(_randmulti_sparse_row([8], [2]))
    randmulti_raw = b"".join(randmulti_rows)
    encoded = {
        "mask": brotli.compress(b"mask"),
        "model": brotli.compress(b"QH0" + bytes(16)),
        "pose": brotli.compress(b"P1D1" + bytes([1, 0, 0, 0])),
        "post": brotli.compress(post),
        "shift": brotli.compress(shift),
        "frac": brotli.compress(frac),
        "frac2": brotli.compress(frac2),
        "frac3": brotli.compress(frac3),
        "bias": brotli.compress(bias),
        "region": brotli.compress(region),
        "randmulti": brotli.compress(randmulti_raw),
    }
    replay.write_text(
        "\n".join(
            [
                "def load_compact_archive_bundle():",
                f"    l_bias = {len(encoded['bias'])}",
                f"    l_region = {len(encoded['region'])}",
                "def main():",
                f"    specs_n = {specs!r}",
            ]
        ),
        encoding="utf-8",
    )
    header_names = ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3")
    payload = b"".join(len(encoded[name]).to_bytes(3, "little") for name in header_names)
    payload += b"".join(encoded[name] for name in header_names)
    payload += encoded["bias"] + encoded["region"] + encoded["randmulti"]
    _write_zip(path, "x", payload)


def test_synthetic_pr84_no_router_stack_expands_runtime_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    monkeypatch.setattr(
        script,
        "_pr84_reordered_model_restore_preflight",
        lambda model_payload: {
            "input_model_payload_bytes": len(model_payload),
            "input_model_payload_sha256": script.sha256_bytes(model_payload),
            "restored_block_size": 32,
            "restored_model_bytes": len(model_payload) + 6,
            "restored_model_sha256": script.sha256_bytes(b"QZS3xx" + model_payload),
            "runtime_inflate_renderer": "synthetic-test-double",
            "runtime_inflate_renderer_sha256": "0" * 64,
            "status": "synthetic_test_double",
        },
    )
    monkeypatch.setattr(
        script,
        "_public_payload_unpack_preflight",
        lambda payload: {
            "payload_bytes": len(payload),
            "payload_format": "synthetic_public_qma9_payload_test_double",
            "schema": "renderer_payload_pr64_len_table_v1",
            "members": {
                "masks.qma9": {"bytes": range_mask_bytes, "sha256": script.sha256_bytes(payload[:range_mask_bytes])},
                "renderer.bin": {"bytes": model_bytes + 4, "sha256": "1" * 64},
                "optimized_poses.qp1": {"bytes": 1202, "sha256": "2" * 64},
            },
            "runtime_unpacker": "synthetic-test-double",
            "runtime_unpacker_sha256": "0" * 64,
            "status": "synthetic_test_double",
        },
    )
    range_mask_bytes = 64
    model_bytes = 12
    pose_stream = _qp1_pose_stream()
    source = tmp_path / "inflate.py"
    _source_inflate(
        source,
        range_mask_bytes=range_mask_bytes,
        model_bytes=model_bytes,
        pose_bytes=len(pose_stream),
    )
    pr84_payload = _qma9_payload(range_mask_bytes) + bytes(range(model_bytes)) + pose_stream
    pr84_archive = tmp_path / "pr84.zip"
    _write_zip(pr84_archive, "p", pr84_payload)
    pr82_archive = tmp_path / "pr82.zip"
    replay = tmp_path / "pr82_inflate.py"
    _synthetic_pr82_archive(pr82_archive, replay)

    summary = script.build_candidates(
        pr84_archive=pr84_archive,
        pr84_source_inflate=source,
        pr82_archive=pr82_archive,
        replay_inflate=replay,
        output_dir=tmp_path / "out",
        expected_pr84_sha256=None,
        expected_pr84_bytes=None,
        expected_pr82_sha256=None,
    )

    assert summary["score_claim"] is False
    assert summary["no_remote_dispatch"] is True
    assert summary["candidate_count"] == 3
    assert summary["pr84_profile"]["payload_contract"]["expected_pr84_no_router"] is True
    assert summary["pr84_profile"]["payload_contract"]["router_action_bytes"] == 0
    manifest = json.loads((tmp_path / "out/pr84_qma9_pr82_qps1_controls_qrm1_all072/manifest.json").read_text())
    assert manifest["source_pr84_payload_preflight"]["payload_contract"]["contract"] == (
        "pr84_qma9_reordered_qzs3_qp1_no_router"
    )
    assert manifest["dispatch_gate"]["remote_dispatch_performed"] is False
    with zipfile.ZipFile(tmp_path / "out/pr84_qma9_pr82_qps1_controls_qrm1_all072/archive.zip") as zf:
        names = sorted(info.filename for info in zf.infolist() if not info.is_dir())
    assert names == ["masks.qma9", "optimized_poses.qp1", "qpost.bin", "renderer.bin"]


def test_synthetic_pr84_public_payload_plus_qpost_layout_preserves_p_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    monkeypatch.setattr(
        script,
        "_pr84_reordered_model_restore_preflight",
        lambda model_payload: {
            "input_model_payload_bytes": len(model_payload),
            "input_model_payload_sha256": script.sha256_bytes(model_payload),
            "restored_block_size": 32,
            "restored_model_bytes": len(model_payload) + 6,
            "restored_model_sha256": script.sha256_bytes(b"QZS3xx" + model_payload),
            "runtime_inflate_renderer": "synthetic-test-double",
            "runtime_inflate_renderer_sha256": "0" * 64,
            "status": "synthetic_test_double",
        },
    )
    monkeypatch.setattr(
        script,
        "_public_payload_unpack_preflight",
        lambda payload: {
            "payload_bytes": len(payload),
            "payload_format": "synthetic_public_qma9_payload_test_double",
            "schema": "renderer_payload_pr64_len_table_v1",
            "members": {
                "masks.qma9": {"bytes": range_mask_bytes, "sha256": script.sha256_bytes(payload[:range_mask_bytes])},
                "renderer.bin": {"bytes": model_bytes + 4, "sha256": "1" * 64},
                "optimized_poses.qp1": {"bytes": 1202, "sha256": "2" * 64},
            },
            "runtime_unpacker": "synthetic-test-double",
            "runtime_unpacker_sha256": "0" * 64,
            "status": "synthetic_test_double",
        },
    )
    range_mask_bytes = 64
    model_bytes = 12
    pose_stream = _qp1_pose_stream()
    source = tmp_path / "inflate.py"
    _source_inflate(
        source,
        range_mask_bytes=range_mask_bytes,
        model_bytes=model_bytes,
        pose_bytes=len(pose_stream),
    )
    pr84_payload = _qma9_payload(range_mask_bytes) + bytes(range(model_bytes)) + pose_stream
    pr84_archive = tmp_path / "pr84.zip"
    _write_zip(pr84_archive, "p", pr84_payload)
    pr82_archive = tmp_path / "pr82.zip"
    replay = tmp_path / "pr82_inflate.py"
    _synthetic_pr82_archive(pr82_archive, replay)

    summary = script.build_candidates(
        pr84_archive=pr84_archive,
        pr84_source_inflate=source,
        pr82_archive=pr82_archive,
        replay_inflate=replay,
        output_dir=tmp_path / "out",
        expected_pr84_sha256=None,
        expected_pr84_bytes=None,
        expected_pr82_sha256=None,
        archive_layout=script.ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST,
    )

    assert summary["archive_layout"] == script.ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST
    assert summary["candidate_count"] == 3
    row = summary["candidates"][0]
    assert row["archive_layout"] == script.ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST
    assert row["candidate_id"].endswith("_packedp")
    manifest = json.loads((tmp_path / "out/pr84_qma9_pr82_qps1_controls_all600_packedp/manifest.json").read_text())
    assert manifest["payload_packing_parity"]["exact_pr84_public_payload_preserved"] is True
    assert manifest["payload_packing_parity"]["output_p_member_sha256"] == script.sha256_bytes(pr84_payload)
    assert manifest["payload_packing_parity"]["source_runtime_unpack_preflight"]["status"] == "synthetic_test_double"
    assert manifest["output_archive"]["members"] == ["p", "qpost.bin"]
    with zipfile.ZipFile(tmp_path / "out/pr84_qma9_pr82_qps1_controls_all600_packedp/archive.zip") as zf:
        assert sorted(info.filename for info in zf.infolist() if not info.is_dir()) == ["p", "qpost.bin"]
        assert zf.read("p") == pr84_payload


def test_pr84_payload_length_must_match_no_router_or_router_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    monkeypatch.setattr(
        script,
        "_pr84_reordered_model_restore_preflight",
        lambda model_payload: {"status": "not_reached"},
    )
    source = tmp_path / "inflate.py"
    pose_stream = _qp1_pose_stream()
    _source_inflate(source, range_mask_bytes=64, model_bytes=12, pose_bytes=len(pose_stream))
    bad_payload = _qma9_payload(64) + bytes(12) + pose_stream + b"stale-tail"
    bad_archive = tmp_path / "bad_pr84.zip"
    _write_zip(bad_archive, "p", bad_payload)

    with pytest.raises(script.Pr84Pr82StackError, match="fixed-slice contract"):
        script._load_pr84_source(
            bad_archive,
            source,
            expected_sha256=None,
            expected_bytes=None,
        )


@pytest.mark.skipif(
    not (PR84_ARCHIVE.exists() and PR84_SOURCE_INFLATE.exists() and PR82_ARCHIVE.exists() and PR82_REPLAY.exists()),
    reason="PR84/PR82 public intake artifacts missing",
)
def test_actual_pr84_pr82_stack_builds_local_candidates(tmp_path: Path) -> None:
    script = _load_script()

    summary = script.build_candidates(
        pr84_archive=PR84_ARCHIVE,
        pr84_source_inflate=PR84_SOURCE_INFLATE,
        pr82_archive=PR82_ARCHIVE,
        replay_inflate=PR82_REPLAY,
        output_dir=tmp_path,
        expected_pr84_sha256=None,
        expected_pr84_bytes=None,
        expected_pr82_sha256=None,
    )

    assert summary["pr84_profile"]["archive_bytes"] == 215_735
    assert summary["pr84_profile"]["payload_bytes"] == 159_011 + 55_725 + 899
    assert summary["pr84_profile"]["payload_contract"]["expected_pr84_no_router"] is True
    assert summary["pr84_profile"]["segments"][0]["sha256"] == script.EXPECTED_PR84_RANGE_MASK_SHA256
    assert summary["highest_ev_local_candidate"]["dispatch_gate"]["remote_dispatch_performed"] is False
    manifest = json.loads((tmp_path / "pr84_qma9_pr82_qps1_controls_qrm1_all072/manifest.json").read_text())
    assert manifest["source_pr84_payload_preflight"]["payload_contract"]["router_action_bytes"] == 0
    assert manifest["raw_output_delta_proof"]["changed_values"] > 0
