# SPDX-License-Identifier: MIT
"""Tests for tac.v2_compose.archive_grammar — 4-section byte-close + the NO-FAKE inflate parity.

The inflate-parity test is the load-bearing NO-FAKE check: it proves the SELF-CONTAINED inflate.py's
INLINED numpy warp/render is BIT-IDENTICAL to the proven reach-tool path (via bulk_generator), so the
"compile the generator into inflate.py" (rule-118 FREE) does not silently diverge.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.v2_compose.archive_grammar import (
    MAGIC_V2,
    assemble_v2_packet,
    build_store_blob,
    byte_accounting,
    pack_v2_archive,
    parse_store_blob,
    unpack_v2_archive,
)

_REPO = Path(__file__).resolve().parents[3]


def test_pack_unpack_roundtrip():
    store = b"STORE-bytes-xyz"
    residual = b""  # empty residual (deterministic floor)
    pose = b"PNTG-fake-pose"
    manifest = b'{"format_version":"v2.0","n_pairs":7}'
    blob = pack_v2_archive(store, residual, pose, manifest)
    assert blob[: len(MAGIC_V2)] == MAGIC_V2
    s2, r2, p2, m2 = unpack_v2_archive(blob)
    assert s2 == store
    assert r2 == residual
    assert p2 == pose
    assert m2["format_version"] == "v2.0"
    assert m2["n_pairs"] == 7


def test_unpack_bad_magic_raises():
    with pytest.raises(ValueError):
        unpack_v2_archive(b"NOPE" + b"\x00" * 40)


def _synthetic_keyframes(n_kf=2, H=384, W=512, seed=0):
    rng = np.random.default_rng(seed)
    # structured-ish label maps (regions) so contour coding is non-degenerate
    kf = np.zeros((n_kf, H, W), np.int64)
    for i in range(n_kf):
        kf[i, : H // 2, :] = 2          # top = Undriv
        kf[i, H // 2 :, :] = 0          # bottom = Road
        kf[i, -40:, :] = 4              # hood = MyCar
        kf[i, :, (W // 2 - 2 + i) : (W // 2 + 2 + i)] = 1  # a lane stripe (shifts per kf)
    return kf


def test_store_blob_roundtrip_keyframes_bit_exact():
    kf = _synthetic_keyframes()
    indices = [0, 47]
    palette = np.array([[128, 128, 128], [170, 170, 170], [100, 80, 60],
                        [120, 140, 160], [180, 200, 230]], np.float32)
    calib = (-0.003224, 0.0, -0.01)
    warp_codes = [0, 3, 2, 3, 1]  # Road=ground, Lane=learn, Undriv=rot, Movable=learn, MyCar=identity
    blob = build_store_blob(indices, kf, palette, calib, warp_codes, reach_kstar=47, n_pairs=600)
    parsed = parse_store_blob(blob)
    assert parsed.keyframe_indices == indices
    assert np.array_equal(parsed.keyframe_lstars, kf)  # contour codec is bit-exact
    assert parsed.calib == pytest.approx(calib)
    assert parsed.warp_type_codes == warp_codes
    assert parsed.reach_kstar == 47
    assert parsed.n_pairs == 600
    assert parsed.shape == (384, 512)
    # palette survives fp16 (close, not exact)
    assert np.allclose(parsed.palette, palette, atol=0.5)


def test_byte_accounting_real_archive(tmp_path):
    kf = _synthetic_keyframes(n_kf=1)
    store = build_store_blob([0], kf, np.zeros((5, 3), np.float32), (0.0, 0.0, 0.0),
                             [0, 3, 2, 3, 1], reach_kstar=47, n_pairs=600)
    pose = b"PNTGxx"
    manifest = b'{"v":2}'
    blob = pack_v2_archive(store, b"", pose, manifest)
    zip_path, zbytes = assemble_v2_packet(blob, tmp_path / "packet")
    assert zip_path.exists()
    assert (tmp_path / "packet" / "inflate.py").exists()
    assert (tmp_path / "packet" / "inflate.sh").exists()
    acct = byte_accounting(zip_path, store, b"", pose, manifest)
    assert acct["archive_zip_bytes"] == zbytes
    assert acct["rate"] == zbytes / 37_545_489.0
    assert acct["section_bytes"]["store_blob"] == len(store)
    assert acct["residual_inr_present"] is False
    assert len(acct["archive_zip_sha256"]) == 64


# --- the NO-FAKE inflate parity (needs torch for the bicubic R) ---
def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_torch(), reason="torch needed for bicubic R")
def test_inflate_bulk_bit_identical_to_proven_path(tmp_path):
    """The inflate.py inlined warp/render produces frames BIT-IDENTICAL to the proven tools path."""
    # path bootstrap for the proven tools primitives (the reference)
    for p in (_REPO, _REPO / "src", _REPO / "upstream"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from tools.measure_pose_warp_dseg import intrinsics_at, _target_grid
    from tools.measure_screw_reach_through_R import composite_warped_labels
    from tac.v2_compose.bulk_generator import bicubic_up_to_camera, render_partition
    from tac.v2_compose.pose_sidecar import build_pose_sidecar_from_cache_poses

    H, W = 384, 512
    kf = _synthetic_keyframes(n_kf=1, H=H, W=W)  # keyframe at SEG res (inflate assumes this)
    indices = [0]
    n_pairs = 2
    palette = np.array([[128, 128, 128], [170, 170, 170], [100, 80, 60],
                        [120, 140, 160], [180, 200, 230]], np.float32)
    calib = (-0.003224, 0.0, -0.01)
    # non-trivial poses to exercise the warp (pair 1 warps from keyframe 0 by k=1)
    poses = np.array([[0.5, 0.1, 1.0, 0.001, 0.002, 0.003],
                      [0.6, 0.0, 1.1, 0.002, 0.001, 0.004]], np.float64)

    store = build_store_blob(indices, kf, palette, calib, [0, 3, 2, 3, 1],
                             reach_kstar=47, n_pairs=n_pairs)
    pose_path = tmp_path / "posenet_targets.bin"
    build_pose_sidecar_from_cache_poses(poses, pose_path)
    pose_blob = pose_path.read_bytes()
    manifest = b'{"format_version":"v2.0","n_pairs":2}'
    blob = pack_v2_archive(store, b"", pose_blob, manifest)

    packet_dir = tmp_path / "packet"
    zip_path, _ = assemble_v2_packet(blob, packet_dir)
    # extract + run the generated inflate.py exactly as the contest runtime would
    import zipfile
    (packet_dir / "archive").mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(packet_dir / "archive")
    dst = packet_dir / "0.raw"
    proc = subprocess.run(
        [sys.executable, str(packet_dir / "inflate.py"), str(packet_dir / "archive" / "0.bin"), str(dst)],
        capture_output=True, text=True, cwd=str(packet_dir),
    )
    assert proc.returncode == 0, f"inflate failed: {proc.stderr}"

    CAM_H, CAM_W = 874, 1164
    fb = CAM_H * CAM_W * 3
    assert dst.stat().st_size == 2 * n_pairs * fb  # frame0+frame1 per pair, full camera res

    # --- reference via the proven tools path (with the SAME fp16-quantized poses + palette) ---
    poses_q = poses.astype(np.float16).astype(np.float64)         # PNTG stores fp16
    palette_q = palette.astype(np.float16).astype(np.float64)     # store_blob stores fp16
    K = intrinsics_at(W, H)
    Kinv = np.linalg.inv(K)
    grid = _target_grid(H, W)

    with open(dst, "rb") as f:
        for p in range(n_pairs):
            f0 = np.frombuffer(f.read(fb), dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
            f1 = np.frombuffer(f.read(fb), dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
            assert np.array_equal(f0, f1)  # frame0 == frame1 == bulk render
            anchor, k = 0, p  # single keyframe at 0
            if k == 0:
                warped = kf[0]
            else:
                warped = composite_warped_labels(kf[0], poses_q, anchor, k, K, Kinv, calib, grid)
            ref = bicubic_up_to_camera(render_partition(warped, palette_q))
            assert np.array_equal(f1, ref), f"inflate frame {p} diverges from proven path"
