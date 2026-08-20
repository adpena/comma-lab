# SPDX-License-Identifier: MIT
"""Tests for tac.v2_compose.archive_grammar — 4-section byte-close + the NO-FAKE inflate parity.

The inflate-parity test is the load-bearing NO-FAKE check: it proves the SELF-CONTAINED inflate.py's
INLINED numpy warp/render is BIT-IDENTICAL to the proven reach-tool path (via bulk_generator), so the
"compile the generator into inflate.py" (rule-118 FREE) does not silently diverge.
"""

from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from tac.v2_compose.archive_grammar import (
    MAGIC_V2,
    assemble_v2_packet,
    build_store_blob,
    byte_accounting,
    generate_v2_inflate_py,
    pack_v2_archive,
    parse_pose_sidecar_blob,
    parse_store_blob,
    unpack_v2_archive,
    unpack_v2_sections,
    validate_v2_receiver_payload,
)

_REPO = Path(__file__).resolve().parents[3]


def _test_pose_blob(*, n_pairs: int = 1, trailing_compressed: bytes = b"") -> bytes:
    poses = np.zeros((n_pairs, 6), dtype=np.float16)
    compressed = zlib.compress(poses.tobytes(), level=9) + trailing_compressed
    return (
        b"PNTG"
        + struct.pack("<HII", 1, n_pairs, 2 * n_pairs)
        + struct.pack("<I", len(compressed))
        + compressed
    )


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


def test_unpack_refuses_every_truncated_section_boundary_and_trailing_bytes():
    blob = pack_v2_archive(b"store", b"residual", b"pose", b'{"v":2}')
    for cut in range(len(MAGIC_V2), len(blob)):
        with pytest.raises(ValueError, match="truncated"):
            unpack_v2_sections(blob[:cut])
    with pytest.raises(ValueError, match="trailing bytes"):
        unpack_v2_sections(blob + b"smuggled")


def test_unpack_preserves_manifest_bytes_and_reemits_canonically():
    manifest = b'{"format_version":"v2.0","n_pairs":7}'
    blob = pack_v2_archive(b"store", b"residual", b"pose", manifest)
    parsed = unpack_v2_sections(blob)
    assert parsed.manifest_bytes == manifest
    assert pack_v2_archive(
        parsed.store_blob,
        parsed.residual_inr_blob,
        parsed.pose_sidecar_blob,
        parsed.manifest_bytes,
    ) == blob


def test_unpack_refuses_duplicate_or_non_object_manifest():
    with pytest.raises(ValueError, match="duplicate v2 manifest key"):
        unpack_v2_sections(pack_v2_archive(b"s", b"r", b"p", b'{"v":1,"v":2}'))
    with pytest.raises(ValueError, match="JSON object"):
        unpack_v2_sections(pack_v2_archive(b"s", b"r", b"p", b"[]"))


def test_generated_receiver_uses_the_same_strict_packet_boundaries(tmp_path):
    inflate_path = tmp_path / "inflate.py"
    inflate_path.write_text(generate_v2_inflate_py())
    spec = importlib.util.spec_from_file_location("generated_v2_inflate", inflate_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    store = build_store_blob(
        [0],
        np.zeros((1, 8, 8), dtype=np.int64),
        np.zeros((5, 3), dtype=np.float32),
        (0.0, 0.0, 0.0),
        [0, 0, 2, 0, 1],
        reach_kstar=1,
        n_pairs=1,
    )
    pose = _test_pose_blob()
    valid = pack_v2_archive(store, b"", pose, b'{"v":2}')
    valid_path = tmp_path / "valid.bin"
    valid_path.write_bytes(valid)
    generated_sections = module._read_sections(valid_path)
    assert generated_sections[3] == {"v": 2}
    assert module._parse_store(generated_sections[0])["n_pairs"] == 1
    assert module._decode_pntg_poses(generated_sections[2]).shape == (1, 6)
    assert validate_v2_receiver_payload(valid).pose_sidecar.n_pairs == 1

    bad_pose = _test_pose_blob(n_pairs=2)
    generated_store = module._parse_store(generated_sections[0])
    with pytest.raises(ValueError, match="pair domains differ"):
        module._validate_receiver_contract(
            generated_store,
            module._decode_pntg_poses(bad_pose),
        )
    residual_contract = {
        "n_hidden": 1,
        "hidden_dim": 2,
        "mod_dim": 2,
        "n_classes": 5,
        "activation": "hosc",
        "hosc_beta": 4.0,
        "hosc_omega": 1.0,
        "softmax_temp": 0.1,
        "wire_w0": 20.0,
        "wire_s0": 10.0,
        "chroma": True,
        "render_h": 8,
        "render_w": 8,
        "bank_n_scales": 1,
        "bank_n_orient0": 1,
        "bank_f0": 2.0,
        "bank_base": 2.0,
        "bank_n_iso": 0,
        "learn_classes": [1, 3],
        "dilate": 1,
        "mask_mode": "boundary_annulus",
    }
    with pytest.raises(ValueError, match="code shape outside"):
        module._validate_receiver_contract(
            generated_store,
            None,
            residual_contract,
            {"code": np.zeros((1, 2), dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="mask mode"):
        module._validate_receiver_contract(
            generated_store,
            None,
            {**residual_contract, "mask_mode": "caller-invented"},
            {"code": np.zeros((2, 2), dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="max_bank_freq"):
        module._validate_receiver_contract(
            generated_store,
            None,
            {**residual_contract, "max_bank_freq": -1.0},
            {"code": np.zeros((2, 2), dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="max_bank_freq"):
        module._curvelet_B({**residual_contract, "max_bank_freq": float("nan")})
    invalid_packet_dir = tmp_path / "invalid-packet"
    with pytest.raises(ValueError, match="pair counts differ"):
        assemble_v2_packet(
            pack_v2_archive(store, b"", bad_pose, b'{}'),
            invalid_packet_dir,
        )
    assert not invalid_packet_dir.exists()

    trailing_path = tmp_path / "trailing.bin"
    trailing_path.write_bytes(valid + b"x")
    with pytest.raises(ValueError, match="trailing bytes"):
        module._read_sections(trailing_path)

    duplicate_path = tmp_path / "duplicate.bin"
    duplicate_path.write_bytes(pack_v2_archive(b"store", b"", b"", b'{"v":1,"v":2}'))
    with pytest.raises(ValueError, match="duplicate v2 manifest key"):
        module._read_sections(duplicate_path)

    compressed_trailing = _test_pose_blob(trailing_compressed=b"hidden")
    with pytest.raises(ValueError, match="trailing stream"):
        parse_pose_sidecar_blob(compressed_trailing)
    with pytest.raises(ValueError, match="trailing PNTG compressed stream"):
        module._decode_pntg_poses(compressed_trailing)


def _synthetic_keyframes(n_kf=2, H=384, W=512, seed=0):
    del seed  # retained for fixture-call compatibility
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
    # PHYSICAL per-class warp-regime codes (A3.2): Road=ground(0), Lane=ground(0), Undriv=rot(2),
    # Movable=ground(0), MyCar=identity(1). The inflate CONSUMES these (no hardcoded routing).
    warp_codes = [0, 0, 2, 0, 1]
    blob = build_store_blob(indices, kf, palette, calib, warp_codes, reach_kstar=47, n_pairs=600)
    parsed = parse_store_blob(blob)
    assert parsed.keyframe_indices == indices
    assert np.array_equal(parsed.keyframe_lstars, kf)  # dense-raster LZMA is bit-exact
    assert parsed.calib == pytest.approx(calib)
    assert parsed.warp_type_codes == warp_codes
    assert parsed.reach_kstar == 47
    assert parsed.n_pairs == 600
    assert parsed.shape == (384, 512)
    # palette survives fp16 (close, not exact)
    assert np.allclose(parsed.palette, palette, atol=0.5)


def test_code_aware_warp_router_consumes_noncanonical_regime_map():
    from tac.v2_compose.bulk_generator import composite_warped_labels_by_codes
    from tools.measure_pose_warp_dseg import _target_grid, intrinsics_at

    height, width = 24, 32
    source = np.arange(height * width, dtype=np.int64).reshape(height, width) % 5
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.1, 1.1, 0.002, 0.001, 0.004],
        ],
        dtype=np.float64,
    )
    intrinsics = intrinsics_at(width, height)
    inverse = np.linalg.inv(intrinsics)
    grid = _target_grid(height, width)
    args = (source, poses, 0, 1, intrinsics, inverse, (0.16, 0.05, 0.02), grid)

    canonical = composite_warped_labels_by_codes(*args, [0, 0, 2, 0, 1])
    all_identity = composite_warped_labels_by_codes(*args, [1, 1, 1, 1, 1])

    assert np.array_equal(all_identity, source)
    assert not np.array_equal(canonical, all_identity)


def test_store_parser_refuses_truncation_trailing_and_duplicate_indices():
    kf = _synthetic_keyframes(n_kf=2, H=32, W=48)
    blob = build_store_blob(
        [0, 1],
        kf,
        np.zeros((5, 3), np.float32),
        (0.0, 0.0, 0.0),
        [0, 0, 2, 0, 1],
        reach_kstar=1,
        n_pairs=2,
    )
    with pytest.raises(ValueError, match="truncated"):
        parse_store_blob(blob[:-1])
    with pytest.raises(ValueError, match="trailing bytes"):
        parse_store_blob(blob + b"x")

    # The first keyframe index starts after magic + fixed header + palette +
    # calibration + warp codes. Rewrite the second index to duplicate it while
    # preserving both contour payloads.
    first_index_off = 6 + 24 + (5 * 3 * 2) + 24 + 5
    (first_len,) = struct.unpack_from("<I", blob, first_index_off + 4)
    second_index_off = first_index_off + 8 + first_len
    duplicate = bytearray(blob)
    duplicate[second_index_off : second_index_off + 4] = struct.pack("<I", 0)
    with pytest.raises(ValueError, match="duplicate keyframe index"):
        parse_store_blob(bytes(duplicate))

    corrupt_contour = bytearray(blob)
    (contour_len,) = struct.unpack_from("<I", corrupt_contour, first_index_off + 4)
    contour_start = first_index_off + 8
    corrupt_contour[contour_start : contour_start + contour_len] = b"\x00" * contour_len
    with pytest.raises(ValueError, match="contour failed to decode"):
        parse_store_blob(bytes(corrupt_contour))

    empty_store = build_store_blob(
        [],
        np.zeros((0, 8, 8), dtype=np.int64),
        np.zeros((5, 3), dtype=np.float32),
        (0.0, 0.0, 0.0),
        [0, 0, 2, 0, 1],
        reach_kstar=1,
        n_pairs=1,
    )
    with pytest.raises(ValueError, match="invalid keyframe/pair counts"):
        parse_store_blob(empty_store)


def test_byte_accounting_real_archive(tmp_path):
    kf = _synthetic_keyframes(n_kf=1)
    store = build_store_blob([0], kf, np.zeros((5, 3), np.float32), (0.0, 0.0, 0.0),
                             [0, 0, 2, 0, 1], reach_kstar=47, n_pairs=600)
    pose = _test_pose_blob(n_pairs=600)
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
    from tac.v2_compose.bulk_generator import bicubic_up_to_camera, render_partition
    from tac.v2_compose.pose_sidecar import build_pose_sidecar_from_cache_poses
    from tools.measure_pose_warp_dseg import _target_grid, intrinsics_at
    from tools.measure_screw_reach_through_R import composite_warped_labels

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

    store = build_store_blob(indices, kf, palette, calib, [0, 0, 2, 0, 1],
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
            warped = (
                kf[0]
                if k == 0
                else composite_warped_labels(kf[0], poses_q, anchor, k, K, Kinv, calib, grid)
            )
            ref = bicubic_up_to_camera(render_partition(warped, palette_q))
            assert np.array_equal(f1, ref), f"inflate frame {p} diverges from proven path"
