# SPDX-License-Identifier: MIT
"""v4c inflate runner — base-adjudicated static two-plane warp + photometric.

Grammar v4c (rule-118 FREE generic code; NO scorers, NO GT masks, NO shipped
mask).  frame_0 is reconstructed from decoded frame_1 by a per-pair choice of
  selector 0  -> SINGLE-plane full homography H = K(R - t n^T/d)K^{-1}, s_r=1.0
  selector 1  -> TWO-plane STATIC compose:
                   rows <  v (above horizon) -> far = H_inf (s_t=0, s_r=1.0)
                   rows >= v (below horizon) -> ground = full H
then per-pair PHOTOMETRIC auto-exposure  f0 := a*warp(f1) + b  (rung B, camerad
AE re-tunes frame-to-frame), and an optional global rolling-shutter row-shear
(rung A: rotation scaled linearly across rows, sign from the pose yaw dim; ~0
bytes, one manifest constant `rs_beta_global`).  The horizon row v is DERIVED at
decode from the native intrinsics (v = round(cy) = 437) — a static-global split,
pure code, 0 counted bytes.

Payload consumed (state/pose_warp.stp, MAGIC PFS1WPC1):
  <8s magic><I n_pairs>
    <I l1><l1 tp_member : kl1 byte-plane f16 (n,6) p_best>
    <I l2><l2 st_coded  : r7 token codes -> s_t grid idx>
    <I l3><l3 sel_coded : brotli packbits(n selector bits)>
    <I l4><l4 ab_member : kl1 byte-plane f16 (n,2) exposure (a,b)>
Every byte is consumed (parse asserts offset == len) — the counted-vs-inert
receiver-consumption bijection (#417).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path, PurePosixPath

import brotli
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ddm_r7_token_coder import decode_token_codes
from ddm_tr1_runtime import (
    _encode_tokens,
    build_packet,
    parse_packet,
    render_frame1_camera_uint8,
)
from pfs1_warp_receiver import (
    ST_GRID,
    _target_grid,
    _to_uint8,
    intrinsics_native,
    pose_to_homography,
    warp_rgb,
)

MAGIC = b"PFS1WPC1"
KL1_MAGIC = b"KL1PWF01"
CAMERA_H, CAMERA_W = 874, 1164
FRAME0_POLICY = "warp_two_plane_static_photo_v4c"


def geometric_horizon_row(K: np.ndarray) -> int:
    """Ground-plane vanishing row for pitch=0: n=[0,-1,0], l=K^{-T}n, v=-l2/l1.
    Returns round(cy)=437 for the EON intrinsics.  Pure code, 0 bytes."""
    n = np.array([0.0, -1.0, 0.0], np.float64)
    ln = np.linalg.inv(K).T @ n
    return round(-ln[2] / ln[1])


def decode_kl1_field(member: bytes) -> np.ndarray:
    """Rule-118-free byte-plane f16 field decoder (inlined kl1 KL1PWF01).
    Reconstructs the EXACT (n,d) float16 field the encoder produced."""
    if member[:8] != KL1_MAGIC:
        raise SystemExit("kl1 field magic differs")
    n, d, plen = struct.unpack("<HHI", member[8:16])
    raw = brotli.decompress(member[16:16 + plen])
    half = n * d
    hi = np.frombuffer(raw[:half], dtype=np.uint8).astype(np.uint16)
    lo = np.frombuffer(raw[half:2 * half], dtype=np.uint8).astype(np.uint16)
    cm = ((hi << 8) | lo).reshape(d, n)  # (d,n) uint16
    return np.ascontiguousarray(cm.T).view(np.float16)  # (n,d)


def parse_pose_warp_v4c(payload: bytes):
    if payload[:8] != MAGIC:
        raise SystemExit("pose_warp magic differs (expected v4c PFS1WPC1)")
    off = 8
    (n_pairs,) = struct.unpack_from("<I", payload, off)
    off += 4
    secs = []
    for _ in range(4):
        (ln,) = struct.unpack_from("<I", payload, off)
        off += 4
        secs.append(payload[off:off + ln])
        off += ln
    if off != len(payload):
        raise SystemExit(f"pose_warp has {len(payload) - off} unconsumed bytes")
    tp_member, st_coded, sel_coded, ab_member = secs
    tp = decode_kl1_field(tp_member).astype(np.float64).reshape(n_pairs, 6)
    st_idx = np.asarray(decode_token_codes(st_coded),
                        dtype=np.int64).reshape(-1)[:n_pairs]
    sel = np.unpackbits(np.frombuffer(brotli.decompress(sel_coded),
                                      dtype=np.uint8))[:n_pairs].astype(np.int64)
    ab = decode_kl1_field(ab_member).astype(np.float64).reshape(n_pairs, 2)
    return n_pairs, tp, st_idx, sel, ab


class Decoder:
    """Factored per-pair decoder (reused by the verify tool)."""

    def __init__(self, archive_dir: Path) -> None:
        manifest = json.loads((archive_dir / "manifest.json").read_text())
        if manifest["frame0_policy"] != FRAME0_POLICY:
            raise SystemExit(f"v4c receiver requires frame0_policy={FRAME0_POLICY}")
        self.rs_beta = float(manifest.get("rs_beta_global", 0.0))
        codes = decode_token_codes((archive_dir / "state/tokens.dr7t").read_bytes())
        token_payload = _encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8))
        packet_bytes = build_packet(manifest["tr1_metadata"], {
            "tokens": token_payload,
            "lotto_renderer": (archive_dir / "state/renderer.sec").read_bytes(),
            "selector": (archive_dir / "state/selector.sec").read_bytes(),
            "pose_stub": (archive_dir / "state/pose_stub.sec").read_bytes(),
        })
        self.packet = parse_packet(packet_bytes)
        (self.n_pairs, self.p_best, self.st_idx, self.sel, self.ab) = \
            parse_pose_warp_v4c((archive_dir / "state/pose_warp.stp").read_bytes())
        if int(self.packet.selector["num_pairs"]) != self.n_pairs:
            raise SystemExit("pose_warp n_pairs differs from selector")
        self.st_vals = np.asarray(ST_GRID, np.float64)
        self.K = intrinsics_native()
        self.Kinv = np.linalg.inv(self.K)
        self.grid = _target_grid(CAMERA_H, CAMERA_W)
        self.v_row = geometric_horizon_row(self.K)
        self._far = (np.arange(CAMERA_H)[:, None] < self.v_row) & np.ones(
            (1, CAMERA_W), bool)
        self._alpha = (np.arange(CAMERA_H) / (CAMERA_H - 1.0))[:, None, None]

    def f1(self, i: int) -> np.ndarray:
        return render_frame1_camera_uint8(self.packet, i)

    def _warp_pair(self, f1_f: np.ndarray, pose: np.ndarray, s_t: float,
                   sel: int, rot: float) -> np.ndarray:
        """Static compose f0 (float, pre-photometric) at rotation scale rot."""
        hg = pose_to_homography(pose, self.K, self.Kinv, s_t, rot, 0.0)
        warp_g = warp_rgb(f1_f, hg, self.grid)
        if sel == 0:
            return warp_g
        hf = pose_to_homography(pose, self.K, self.Kinv, 0.0, rot, 0.0)
        warp_f = warp_rgb(f1_f, hf, self.grid)
        return np.where(self._far[..., None], warp_f, warp_g)

    def f0(self, i: int, f1_u8: np.ndarray | None = None) -> np.ndarray:
        if f1_u8 is None:
            f1_u8 = self.f1(i)
        f1_f = f1_u8.astype(np.float64)
        s_t = float(self.st_vals[self.st_idx[i]])
        pose = self.p_best[i]
        sel = int(self.sel[i])
        a, b = float(self.ab[i][0]), float(self.ab[i][1])
        if self.rs_beta != 0.0:
            # rung A: rolling-shutter row-shear, sign from the pose yaw dim (5).
            beta = self.rs_beta * (1.0 if pose[5] >= 0.0 else -1.0)
            f0f = ((1.0 - self._alpha) * self._warp_pair(f1_f, pose, s_t, sel,
                                                         1.0 - beta / 2.0)
                   + self._alpha * self._warp_pair(f1_f, pose, s_t, sel,
                                                   1.0 + beta / 2.0))
        else:
            f0f = self._warp_pair(f1_f, pose, s_t, sel, 1.0)
        if a != 1.0 or b != 0.0:
            f0f = a * f0f + b
        return _to_uint8(f0f)


def main() -> None:
    archive_dir, output_dir, names_file = map(Path, sys.argv[1:4])
    names = [row.strip() for row in
             names_file.read_text().splitlines() if row.strip()]
    if names != ["0.mkv"]:
        raise SystemExit("this receiver serves exactly the custodied 0.mkv")
    dec = Decoder(archive_dir)
    name = PurePosixPath(names[0])
    if name.is_absolute() or ".." in name.parts:
        raise SystemExit("unsafe video name")
    target = output_dir / name.with_suffix(".raw")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as sink:
        for i in range(dec.n_pairs):
            f1 = dec.f1(i)
            f0 = dec.f0(i, f1)
            sink.write(f0.tobytes())
            sink.write(f1.tobytes())


if __name__ == "__main__":
    main()
