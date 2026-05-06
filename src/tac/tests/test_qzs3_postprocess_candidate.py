from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_qzs3_postprocess_candidate.py"
APPLY_PATH = REPO / "submissions" / "robust_current" / "apply_qzs3_postprocess.py"


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


def test_qpost_container_can_be_applied_as_identity_on_tiny_raw(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "build_qzs3_postprocess_candidate_test")
    apply = _load(APPLY_PATH, "apply_qzs3_postprocess_test")
    streams = {name: b"" for name in builder.QPOST_STREAM_NAMES}
    streams["post"] = brotli.compress(bytes(600 * 3), quality=11)
    qpost = builder.encode_qpost(streams)
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(qpost)
    state = apply.read_qpost(qpost_path, torch.device("cpu"))

    raw = tmp_path / "0.raw"
    original = np.arange(2 * 2 * 2 * 3, dtype=np.uint8)
    raw.write_bytes(original.tobytes())
    record = apply.apply_qpost_to_raw(
        raw,
        state,
        height=2,
        width=2,
        batch_pairs=1,
        device=torch.device("cpu"),
    )

    assert record["pair_count"] == 1
    assert raw.read_bytes() == original.tobytes()


def test_builder_copies_base_p_and_counts_qpost_sidecar(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "build_qzs3_postprocess_candidate_build_test")
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"base-payload"})

    # Minimal PR65-like x layout: 10 24-bit lengths, core blobs, then qpost
    # stream blobs.  Contents need not be valid Brotli for this build test; the
    # runtime parser owns stream semantic validation.
    lengths = [1001, 1002, 101, 3, 4, 5, 6, 7, 8, 9]
    header = b"".join(int(n).to_bytes(3, "little") for n in lengths)
    body = b"".join(bytes([i % 251]) * n for i, n in enumerate(lengths, start=1))
    randmulti = b"tail"
    pr65 = tmp_path / "pr65.zip"
    _stored_zip(pr65, {"x": header + body + randmulti})

    out = tmp_path / "out" / "archive.zip"
    meta = builder.build_candidate(source, pr65, out)
    assert meta["score_claim"] is False
    assert meta["include_streams"] == list(builder.QPOST_STREAM_NAMES)
    assert meta["members"]["p"]["bytes"] == len(b"base-payload")
    assert meta["members"]["qpost.bin"]["bytes"] == 4 + 8 * 4 + sum(lengths[3:]) + len(randmulti)
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["p", "qpost.bin"]


def test_builder_can_zero_omitted_qpost_streams(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "build_qzs3_postprocess_candidate_subset_test")
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"base-payload"})

    lengths = [1001, 1002, 101, 3, 4, 5, 6, 7, 8, 9]
    header = b"".join(int(n).to_bytes(3, "little") for n in lengths)
    body = b"".join(bytes([i % 251]) * n for i, n in enumerate(lengths, start=1))
    randmulti = b"tail"
    pr65 = tmp_path / "pr65.zip"
    _stored_zip(pr65, {"x": header + body + randmulti})

    out = tmp_path / "out" / "archive.zip"
    meta = builder.build_candidate(source, pr65, out, include_streams=("post", "region"))
    assert meta["include_streams"] == ["post", "region"]
    assert meta["omitted_streams"] == ["shift", "frac", "frac2", "frac3", "bias", "randmulti"]
    assert meta["qpost_streams"]["post"]["active"] is True
    assert meta["qpost_streams"]["post"]["bytes"] == 3
    assert meta["qpost_streams"]["region"]["bytes"] == 9
    assert meta["qpost_streams"]["bias"]["active"] is False
    assert meta["qpost_streams"]["bias"]["bytes"] == 0
    assert meta["qpost_streams"]["bias"]["original_bytes"] == 8
    assert meta["members"]["qpost.bin"]["bytes"] == 4 + 8 * 4 + 3 + 9


def test_builder_can_pair_filter_qpost_streams_to_identity_defaults(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "build_qzs3_postprocess_candidate_pair_filter_test")
    apply = _load(APPLY_PATH, "apply_qzs3_postprocess_pair_filter_test")
    source = tmp_path / "source.zip"
    _stored_zip(source, {"p": b"base-payload"})

    post = np.zeros((3, 600), dtype=np.uint8)
    post[:, 2] = [1, 2, 3]
    post[:, 4] = [4, 5, 6]
    bias = np.full(600, 13, dtype=np.uint8)
    bias[2] = 14
    bias[4] = 15
    region = np.zeros(600, dtype=np.uint8)
    region[2] = 9
    region[4] = 10
    streams = {
        "post": brotli.compress(post.tobytes(), quality=11),
        "shift": brotli.compress(b"SH4" + np.full(600, 40, dtype=np.uint8).tobytes(), quality=11),
        "frac": brotli.compress(b"FH1" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "frac2": brotli.compress(b"FH2" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "frac3": brotli.compress(b"FH3" + np.full(600, 4, dtype=np.uint8).tobytes(), quality=11),
        "bias": brotli.compress(b"BH1" + bias.tobytes(), quality=11),
        "region": brotli.compress(b"RH1" + region.tobytes(), quality=11),
        "randmulti": b"randtail",
    }
    core_lengths = [1001, 1002, 101]
    qpost_lengths = [len(streams[name]) for name in builder.QPOST_STREAM_NAMES[:-1]]
    header = b"".join(int(n).to_bytes(3, "little") for n in [*core_lengths, *qpost_lengths])
    core = b"a" * core_lengths[0] + b"b" * core_lengths[1] + b"c" * core_lengths[2]
    pr65 = tmp_path / "pr65.zip"
    _stored_zip(
        pr65,
        {
            "x": header
            + core
            + b"".join(streams[name] for name in builder.QPOST_STREAM_NAMES[:-1])
            + streams["randmulti"]
        },
    )

    out = tmp_path / "out" / "archive.zip"
    meta = builder.build_candidate(
        source,
        pr65,
        out,
        include_streams=("post", "bias", "region"),
        pair_indices=(2,),
    )
    assert meta["pair_filter"]["pair_indices"] == [2]
    assert meta["qpost_streams"]["post"]["bytes"] < meta["qpost_streams"]["post"]["original_bytes"]
    assert meta["qpost_streams"]["randmulti"]["bytes"] == 0

    with zipfile.ZipFile(out) as zf:
        qpost = zf.read("qpost.bin")
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(qpost)
    state = apply.read_qpost(qpost_path, torch.device("cpu"))

    assert state.postprocess is not None
    assert [int(stage[2][2].item()) for stage in state.postprocess] == [1, 2, 3]
    assert [int(stage[2][4].item()) for stage in state.postprocess] == [0, 0, 0]
    assert state.f1_bias is not None
    assert int(state.f1_bias[2].item()) == 14
    assert int(state.f1_bias[4].item()) == 13
    assert state.f1_region is not None
    assert int(state.f1_region[2].item()) == 9
    assert int(state.f1_region[4].item()) == 0
