# SPDX-License-Identifier: MIT
"""v4b inflate runner — two-plane STATIC-horizon warp receiver (task #776).

Grammar v4b (rule-118 FREE generic code; NO scorers, NO GT masks, NO shipped
mask): frame_0 is reconstructed from the decoded frame_1 by a per-pair choice of
  selector 0  -> SINGLE-plane full homography H = K(R - t n^T/d)K^{-1}, s_r=1.0
  selector 1  -> TWO-plane STATIC compose:
                   rows <  v  (above the horizon) -> far = H_inf (s_t=0, s_r=1.0)
                   rows >= v  (below the horizon) -> ground = full H (s_t, s_r=1.0)
The horizon row v is DERIVED at decode from the native intrinsics
(l = K^{-T}[0,-1,0], v = round(-l2/l1) = cy = 437) — a static-global split, pure
code, 0 counted bytes.  Byte-identical to the v4a single-plane receiver on every
selector-0 pair (same primitives, same pose).

Payload consumed (state/pose_warp.stp, MAGIC PFS1WPB2):
  <8s magic><I n_pairs><I l1><l1 tp_coded: brotli f16 (n,6) p_best>
            <I l2><l2 st_coded: r7 token codes -> s_t grid idx>
            <I l3><l3 sel_coded: brotli packbits(n selector bits)>
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

MAGIC = b"PFS1WPB2"
CAMERA_H, CAMERA_W = 874, 1164
FRAME0_POLICY = "warp_two_plane_static_v4b"


def geometric_horizon_row(K: np.ndarray) -> int:
    """Ground-plane vanishing row for pitch=0: n=[0,-1,0], l=K^{-T}n, v=-l2/l1.
    Returns round(cy)=437 for the EON intrinsics.  Pure code, 0 bytes."""
    n = np.array([0.0, -1.0, 0.0], np.float64)
    ln = np.linalg.inv(K).T @ n
    return round(-ln[2] / ln[1])


def parse_pose_warp_v4b(payload: bytes):
    if payload[:8] != MAGIC:
        raise SystemExit("pose_warp magic differs (expected v4b PFS1WPB2)")
    off = 8
    (n_pairs,) = struct.unpack_from("<I", payload, off)
    off += 4
    (l1,) = struct.unpack_from("<I", payload, off)
    off += 4
    tp_coded = payload[off:off + l1]
    off += l1
    (l2,) = struct.unpack_from("<I", payload, off)
    off += 4
    st_coded = payload[off:off + l2]
    off += l2
    (l3,) = struct.unpack_from("<I", payload, off)
    off += 4
    sel_coded = payload[off:off + l3]
    off += l3
    if off != len(payload):
        raise SystemExit(f"pose_warp has {len(payload) - off} unconsumed bytes")
    tp = np.frombuffer(brotli.decompress(tp_coded),
                       dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)
    st_idx = np.asarray(decode_token_codes(st_coded),
                        dtype=np.int64).reshape(-1)[:n_pairs]
    sel = np.unpackbits(np.frombuffer(brotli.decompress(sel_coded),
                                      dtype=np.uint8))[:n_pairs].astype(np.int64)
    return n_pairs, tp, st_idx, sel


class Decoder:
    """Factored per-pair decoder (reused by the verify tool)."""

    def __init__(self, archive_dir: Path) -> None:
        manifest = json.loads((archive_dir / "manifest.json").read_text())
        if manifest["frame0_policy"] != FRAME0_POLICY:
            raise SystemExit(f"v4b receiver requires frame0_policy={FRAME0_POLICY}")
        codes = decode_token_codes((archive_dir / "state/tokens.dr7t").read_bytes())
        token_payload = _encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8))
        packet_bytes = build_packet(manifest["tr1_metadata"], {
            "tokens": token_payload,
            "lotto_renderer": (archive_dir / "state/renderer.sec").read_bytes(),
            "selector": (archive_dir / "state/selector.sec").read_bytes(),
            "pose_stub": (archive_dir / "state/pose_stub.sec").read_bytes(),
        })
        self.packet = parse_packet(packet_bytes)
        (self.n_pairs, self.p_best, self.st_idx, self.sel) = parse_pose_warp_v4b(
            (archive_dir / "state/pose_warp.stp").read_bytes())
        if int(self.packet.selector["num_pairs"]) != self.n_pairs:
            raise SystemExit("pose_warp n_pairs differs from selector")
        self.st_vals = np.asarray(ST_GRID, np.float64)
        self.K = intrinsics_native()
        self.Kinv = np.linalg.inv(self.K)
        self.grid = _target_grid(CAMERA_H, CAMERA_W)
        self.v_row = geometric_horizon_row(self.K)
        self._far = (np.arange(CAMERA_H)[:, None] < self.v_row) & np.ones(
            (1, CAMERA_W), bool)

    def f1(self, i: int) -> np.ndarray:
        return render_frame1_camera_uint8(self.packet, i)

    def f0(self, i: int, f1_u8: np.ndarray | None = None) -> np.ndarray:
        if f1_u8 is None:
            f1_u8 = self.f1(i)
        f1_f = f1_u8.astype(np.float64)
        s_t = float(self.st_vals[self.st_idx[i]])
        pose = self.p_best[i]
        hg = pose_to_homography(pose, self.K, self.Kinv, s_t, 1.0, 0.0)
        if self.sel[i] == 0:
            # SINGLE-plane path — byte-identical to the v4a receiver
            return _to_uint8(warp_rgb(f1_f, hg, self.grid))
        # TWO-plane STATIC compose: far (H_inf) above horizon, ground (H) below
        hf = pose_to_homography(pose, self.K, self.Kinv, 0.0, 1.0, 0.0)
        warp_g = warp_rgb(f1_f, hg, self.grid)
        warp_f = warp_rgb(f1_f, hf, self.grid)
        f0f = np.where(self._far[..., None], warp_f, warp_g)
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
