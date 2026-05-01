"""End-to-end test: build_qpose_archive composes a structurally valid PR #67
single-blob archive, and PR #67's inflate decodes its slices without error.

The test does NOT exercise PR #67's brittle ``model_br_len`` heuristic
(which depends on total payload length); instead it slices the blob using
the metadata our orchestrator emits, then feeds each slice into PR #67's
own decoders. That's the right boundary: our orchestrator is responsible
for SHA-pinning slice offsets in metadata.json so dispatch glue can avoid
the empirical heuristic.
"""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import brotli
import pytest
import torch

import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.build_qpose_archive import (
    PR67_REFERENCE_MASK_BYTES,
    build_qpose_archive,
)
from tac.qp1_pose_codec import decode_qp1
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

PR67_INFLATE = (
    _REPO_ROOT / "reports/raw/leaderboard_intel_20260501/pr67_inflate.py"
)


def _import_pr67():
    if not PR67_INFLATE.exists():
        pytest.skip(f"pr67 inflate reference missing: {PR67_INFLATE}")
    spec = importlib.util.spec_from_file_location("_pr67_inflate_eb", PR67_INFLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_build_emits_valid_archive(tmp_path) -> None:
    out = tmp_path / "smoke"
    archive = out / "archive.zip"
    meta = build_qpose_archive(
        renderer_state_dict=None,
        pose_array=None,
        mask_obu_br_bytes=b"\x00" * PR67_REFERENCE_MASK_BYTES,
        output_archive=archive,
    )
    assert archive.exists()
    assert meta["mask_br_bytes"] == PR67_REFERENCE_MASK_BYTES
    assert meta["model_uncompressed_bytes"] == 59288
    assert meta["pose_codec"] == "qp1"
    # Archive is a single STORED member named 'p'
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert names == ["p"]
        info = zf.getinfo("p")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.file_size == meta["blob_bytes"]
    # Archive overhead < 200 bytes (deterministic stored zip)
    assert meta["archive_bytes"] - meta["blob_bytes"] < 200


def test_pr67_inflate_decodes_smoke_archive_end_to_end(tmp_path) -> None:
    pr67 = _import_pr67()

    out = tmp_path / "smoke"
    archive = out / "archive.zip"
    meta = build_qpose_archive(
        renderer_state_dict=None,
        pose_array=None,
        mask_obu_br_bytes=b"\x00" * PR67_REFERENCE_MASK_BYTES,
        output_archive=archive,
    )

    with zipfile.ZipFile(archive) as zf:
        blob = zf.read("p")

    mask_len = meta["mask_br_bytes"]
    model_len = meta["model_br_bytes"]
    model_br = blob[mask_len : mask_len + model_len]
    pose_q_br = blob[mask_len + model_len :]

    # Decode QZS3 weights through PR #67's unpacker
    model_payload = brotli.decompress(model_br)
    assert model_payload.startswith(b"QZS3")
    state = pr67.get_grouped_qv_state_dict(model_payload, torch.device("cpu"))
    assert len(state) == 111  # JointFrameGenerator full state-dict count

    # Load into PR #67's own JointFrameGenerator definition (NOT ours) and
    # run forward — the structural compat test that matters at deploy time.
    gen_pr67 = pr67.JointFrameGenerator()
    gen_pr67.load_state_dict(state, strict=True)
    gen_pr67.eval()
    mask = torch.zeros(1, 384, 512, dtype=torch.long)
    pose = torch.zeros(1, 6)
    with torch.no_grad():
        f1, f2 = gen_pr67(mask, pose)
    assert f1.shape == (1, 3, 384, 512)
    assert f2.shape == (1, 3, 384, 512)
    assert torch.isfinite(f1).all() and torch.isfinite(f2).all()

    # Decode QP1 pose through OUR decoder and assert shape
    pose_payload = brotli.decompress(pose_q_br)
    assert pose_payload.startswith(b"QP1")
    decoded_pose = decode_qp1(pose_payload)
    assert decoded_pose.shape == (600, 6)
    # Smoke pose was constant 30.0 m/s
    assert abs(float(decoded_pose[:, 0].mean()) - 30.0) < 0.01


def test_with_real_renderer_state_dict_round_trips(tmp_path) -> None:
    """The orchestrator must accept a state_dict directly (not just a file)."""

    pr67 = _import_pr67()

    torch.manual_seed(31)
    template = build_quantizr_faithful_renderer().eval()
    state_dict = template.state_dict()

    archive = tmp_path / "candidate.zip"
    meta = build_qpose_archive(
        renderer_state_dict=state_dict,
        pose_array=None,
        mask_obu_br_bytes=b"\x00" * PR67_REFERENCE_MASK_BYTES,
        output_archive=archive,
    )
    assert meta["model_uncompressed_bytes"] == 59288

    with zipfile.ZipFile(archive) as zf:
        blob = zf.read("p")
    model_br = blob[meta["mask_br_bytes"] : meta["mask_br_bytes"] + meta["model_br_bytes"]]
    state = pr67.get_grouped_qv_state_dict(brotli.decompress(model_br), torch.device("cpu"))
    fresh = pr67.JointFrameGenerator()
    fresh.load_state_dict(state, strict=True)


def test_metadata_sha256_round_trips_to_blob(tmp_path) -> None:
    archive = tmp_path / "out" / "archive.zip"
    meta = build_qpose_archive(
        renderer_state_dict=None,
        pose_array=None,
        mask_obu_br_bytes=b"\x00" * 1024,  # smaller mask just for sha cross-check
        output_archive=archive,
    )
    import hashlib
    with zipfile.ZipFile(archive) as zf:
        blob = zf.read("p")
    assert hashlib.sha256(blob).hexdigest() == meta["blob_sha256"]
