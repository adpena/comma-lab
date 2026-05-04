from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr75_qpost_microstack_candidate.py"
UNPACKER_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def _p6_payload() -> bytes:
    mask_raw = b"\x12\x00\x0a\x0a" + b"m" * 96
    renderer_raw = b"QZS3" + b"r" * 96
    actions_delta_raw = b"\x02\x09\x04\x03\x0a\x05"
    pose_raw = b"QP1" + (5120).to_bytes(2, "little") + b"p" * 8
    mask_br = brotli.compress(mask_raw, quality=0)
    model_br = brotli.compress(renderer_raw, quality=0)
    actions_br = brotli.compress(actions_delta_raw, quality=0)
    pose_br = brotli.compress(pose_raw, quality=0)
    return (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(model_br), len(actions_br), 2)
        + mask_br
        + model_br
        + actions_br
        + pose_br
    )


def _fixture_pr65_archive(path: Path, *, active_pair: int = 7) -> Path:
    post = np.zeros((4, 600), dtype=np.uint8)
    bias = np.full(600, 13, dtype=np.uint8)
    bias[active_pair] = 14
    region = np.zeros(600, dtype=np.uint8)
    streams = {
        "post": brotli.compress(post.tobytes(), quality=11),
        "shift": brotli.compress(b"SH4" + np.full(600, 40, dtype=np.uint8).tobytes(), quality=11),
        "frac": brotli.compress(b"FH1" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "frac2": brotli.compress(b"FH2" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "frac3": brotli.compress(b"FH3" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "bias": brotli.compress(b"BH1" + bias.tobytes(), quality=11),
        "region": brotli.compress(b"RH1" + region.tobytes(), quality=11),
        "randmulti": b"rand",
    }
    core_lengths = [1001, 1002, 101]
    qpost_lengths = [
        len(streams[name])
        for name in ("post", "shift", "frac", "frac2", "frac3", "bias", "region")
    ]
    header = b"".join(int(n).to_bytes(3, "little") for n in [*core_lengths, *qpost_lengths])
    core = b"a" * core_lengths[0] + b"b" * core_lengths[1] + b"c" * core_lengths[2]
    body = (
        header
        + core
        + b"".join(
            streams[name]
            for name in ("post", "shift", "frac", "frac2", "frac3", "bias", "region")
        )
        + streams["randmulti"]
    )
    _stored_zip(path, {"x": body})
    return path


def _trace(path: Path, *, pair: int, combined: float) -> Path:
    samples = []
    for idx in range(600):
        value = combined if idx == pair else 0.0
        samples.append(
            {
                "pair_index": idx,
                "score_combined_contribution_first_order": value,
                "score_pose_contribution_first_order": value / 2.0,
                "score_seg_contribution_exact": value / 2.0,
                "posenet_dist": 0.0,
                "segnet_dist": 0.0,
            }
        )
    path.write_text(json.dumps({"samples": samples}) + "\n")
    return path


def test_microstack_builds_p6_resweep_plus_bias_qpost(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "build_pr75_qpost_microstack_candidate_test")
    unpacker = _load(UNPACKER_PATH, "build_pr75_qpost_microstack_unpacker_test")
    source = tmp_path / "source.zip"
    source_payload = _p6_payload()
    _stored_zip(source, {"p": source_payload})
    _source_header, source_decoded = unpacker._parse_payload(source_payload)
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    c089_trace = _trace(tmp_path / "c089_trace.json", pair=7, combined=0.003)
    pr65_trace = _trace(tmp_path / "pr65_trace.json", pair=7, combined=0.001)

    summary = builder.build_microstack(
        source_archive=source,
        pr65_archive=pr65,
        c089_trace=c089_trace,
        pr65_trace=pr65_trace,
        output_dir=tmp_path / "out",
        top_pairs=1,
        expected_source_sha256=None,
        expected_pr65_sha256=None,
        brotli_params=[(0, 0, 10, 0)],
    )

    best = summary["best_candidate"]
    archive = Path(best["archive"])
    assert archive.is_file()
    manifest = json.loads((archive.parent / "manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["qpost"]["selected_pairs"] == [7]
    assert manifest["qpost"]["no_op_proof"]["is_noop"] is False
    assert manifest["resweep_base"]["decoded_stream_parity"] is True
    assert manifest["safety_preflight"]["qpost_non_noop"] is True
    assert manifest["dispatch"]["claim_command_draft"].startswith("ETA_UTC=")
    assert "tools/claim_lane_dispatch.py claim" in manifest["dispatch"]["claim_command_draft"]

    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["p", "qpost.bin"]
        header, decoded = unpacker._parse_payload(zf.read("p"))
        assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
        assert zf.getinfo("qpost.bin").file_size > 0
    for name, data in source_decoded.items():
        assert decoded[name] == data


def test_microstack_rejects_non_p6_source_payload(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "build_pr75_qpost_microstack_candidate_nonp6_test")
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"P3not-a-valid-p6"})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    trace = _trace(tmp_path / "trace.json", pair=7, combined=0.001)

    with pytest.raises(builder.MicrostackBuildError, match="PR75 P6"):
        builder.build_microstack(
            source_archive=source,
            pr65_archive=pr65,
            c089_trace=trace,
            pr65_trace=trace,
            output_dir=tmp_path / "out",
            top_pairs=1,
            expected_source_sha256=None,
            expected_pr65_sha256=None,
            brotli_params=[(0, 0, 10, 0)],
        )


def test_microstack_fails_closed_on_source_sha_mismatch(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "build_pr75_qpost_microstack_candidate_sha_test")
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": _p6_payload()})
    pr65 = _fixture_pr65_archive(tmp_path / "pr65.zip", active_pair=7)
    trace = _trace(tmp_path / "trace.json", pair=7, combined=0.001)

    with pytest.raises(builder.MicrostackBuildError, match="source archive SHA mismatch"):
        builder.build_microstack(
            source_archive=source,
            pr65_archive=pr65,
            c089_trace=trace,
            pr65_trace=trace,
            output_dir=tmp_path / "out",
            top_pairs=1,
            expected_source_sha256="not-the-source-sha",
            expected_pr65_sha256=None,
            brotli_params=[(0, 0, 10, 0)],
        )
