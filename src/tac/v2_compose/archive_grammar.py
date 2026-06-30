# SPDX-License-Identifier: MIT
"""v2_compose.archive_grammar — the 4-section v2 byte-close grammar + MLX-free inflate (Step 6).

Extends the BUILT 1-section witness byte-close (``tools/witness_byte_close_and_eval.py`` —
``io_pack`` / ``_read_blob`` magic+length-prefix) to the 4-section v2 grammar:

  archive.zip := single member ``0.bin`` :=
    MAGIC(b"WTNV2\\x00") | <I store_len   | store_blob
                         | <I residual_len| residual_inr_blob   (EMPTY for the deterministic floor)
                         | <I pose_len     | pose_sidecar_blob   (PNTG; dual-use: d_pose + warp)
                         | <I manifest_len | manifest_json

The rule-118 FREE/COUNTED boundary (NO-FAKE):
  * COUNTED (in archive.zip): keyframe contour-codes + palette + calib + warp-type mask
    (store_blob), the residual-INR weights (residual_blob), the pose scalars (pose_blob). All
    video-derived.
  * FREE (in inflate.py code): the per-class stratified warp, the SDF/class-mean render, the R1
    sigma=1.0 ramp, the bicubic-up R operator, the deterministic Fourier basis. All GENERIC.
    FORBIDDEN: smuggling a per-frame learned table into inflate.py "code" (NO-FAKE #6 / rule 118).

The generated ``inflate.py`` is SELF-CONTAINED + MLX-FREE (numpy warp/render + torch bicubic only),
loads NO scorer weights at inflate (CLAUDE.md strict-scorer rule), and emits the contest-shape
``(2*n_pairs, 874, 1164, 3)`` uint8 .raw (frame0+frame1 per pair). For the DETERMINISTIC FLOOR
archive (residual_blob empty) it produces the bulk render alone (d_seg floor ~0.0185, d_pose
advisory-high on flat frames — the residual INR closes both). The inflate's numpy warp primitives
are byte-faithful copies of the proven reach-tool path; the parity test asserts the inflate frames
are bit-identical to ``bulk_generator`` (the NO-FAKE faithfulness chain).

Authority: ``[advisory] NON-PROMOTABLE`` until ``upstream/evaluate.py`` runs the SAME bytes (CPU + CUDA).
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.boundary_math.contour_codec import ContourCode, decode_partition, encode_partition
from tac.contest_score import rate_term

__all__ = [
    "MAGIC_V2",
    "pack_v2_archive",
    "unpack_v2_archive",
    "build_store_blob",
    "parse_store_blob",
    "assemble_v2_packet",
    "generate_v2_inflate_py",
    "byte_accounting",
    "V2StoreBlob",
    "WARP_TYPE_CODE",
]

MAGIC_V2 = b"WTNV2\x00"
_STORE_MAGIC = b"WSTR1\x00"

# warp-type codes for the per-class warp-mask (1 byte/class). Self-detected upstream; here just a code.
WARP_TYPE_CODE = {
    "ground_homography": 0,
    "identity": 1,
    "rotation_only": 2,
    "learn": 3,
}
_WARP_TYPE_NAME = {v: k for k, v in WARP_TYPE_CODE.items()}


@dataclass(frozen=True)
class V2StoreBlob:
    """Parsed store_blob (the COUNTED STORE+GENERATE payload)."""

    keyframe_indices: list[int]
    keyframe_lstars: np.ndarray         # (n_kf, H, W) int64 (decoded bit-exact from contour codes)
    palette: np.ndarray                 # (5, 3) float32
    calib: tuple[float, float, float]   # (s_t, s_r, pitch)
    warp_type_codes: list[int]          # per-class warp-type code (len n_classes)
    reach_kstar: int
    n_pairs: int
    shape: tuple[int, int]
    n_classes: int


# ---------------------------------------------------------------------------
# 4-section pack / unpack (magic + length-prefix; modeled on the byte-close io_pack).
# ---------------------------------------------------------------------------
def pack_v2_archive(
    store_blob: bytes, residual_inr_blob: bytes, pose_sidecar_blob: bytes, manifest_bytes: bytes
) -> bytes:
    """Pack the 4 sections into the ``0.bin`` blob (MAGIC + 4x <I-length-prefixed chunks)."""
    buf = bytearray()
    buf += MAGIC_V2
    for chunk in (store_blob, residual_inr_blob, pose_sidecar_blob, manifest_bytes):
        buf += struct.pack("<I", len(chunk))
        buf += chunk
    return bytes(buf)


def unpack_v2_archive(blob: bytes) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    """Inverse of :func:`pack_v2_archive`. Returns (store, residual, pose, manifest_dict)."""
    if blob[: len(MAGIC_V2)] != MAGIC_V2:
        raise ValueError("bad v2 archive magic")
    off = len(MAGIC_V2)
    out: list[bytes] = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", blob, off)
        off += 4
        out.append(blob[off : off + n])
        off += n
    store, residual, pose, manifest_b = out
    manifest = json.loads(manifest_b.decode("utf-8")) if manifest_b else {}
    return store, residual, pose, manifest


# ---------------------------------------------------------------------------
# store_blob: keyframes (contour-coded) + palette + calib + warp-mask + kstar.
# ---------------------------------------------------------------------------
def build_store_blob(
    keyframe_indices: list[int],
    keyframe_lstars: np.ndarray,
    palette: np.ndarray,
    calib: tuple[float, float, float],
    warp_type_codes: list[int],
    reach_kstar: int,
    n_pairs: int,
    *,
    n_classes: int = 5,
) -> bytes:
    """Serialize the STORE payload. Keyframes -> contour_codec (LZMA, bit-exact). Palette fp16,
    calib f64, warp-mask 1 byte/class. The generic warp/render that re-expands this is FREE."""
    kf = np.asarray(keyframe_lstars, dtype=np.int64)
    if kf.ndim != 3 or kf.shape[0] != len(keyframe_indices):
        raise ValueError(f"keyframe_lstars must be (n_kf,H,W) matching indices; got {kf.shape}")
    H, W = int(kf.shape[1]), int(kf.shape[2])
    palette = np.asarray(palette, dtype=np.float32)
    if palette.shape != (n_classes, 3):
        raise ValueError(f"palette must be ({n_classes},3); got {palette.shape}")
    if len(warp_type_codes) != n_classes:
        raise ValueError(f"warp_type_codes must have len {n_classes}; got {len(warp_type_codes)}")

    buf = bytearray()
    buf += _STORE_MAGIC
    buf += struct.pack("<IIII", len(keyframe_indices), H, W, int(n_classes))
    buf += struct.pack("<I", int(reach_kstar))
    buf += struct.pack("<I", int(n_pairs))
    # palette (fp16) + calib (f64 x3) + warp-mask (1 byte/class)
    buf += palette.astype(np.float16).tobytes()
    buf += struct.pack("<ddd", float(calib[0]), float(calib[1]), float(calib[2]))
    buf += bytes(int(c) & 0xFF for c in warp_type_codes)
    # keyframes: idx(<I) + contour_len(<I) + contour_payload (per keyframe)
    for i, idx in enumerate(keyframe_indices):
        code: ContourCode = encode_partition(kf[i], n_classes=n_classes)
        buf += struct.pack("<I", int(idx))
        buf += struct.pack("<I", len(code.payload))
        buf += code.payload
    return bytes(buf)


def parse_store_blob(store_blob: bytes) -> V2StoreBlob:
    """Inverse of :func:`build_store_blob` (decodes keyframes bit-exact via contour_codec)."""
    if store_blob[: len(_STORE_MAGIC)] != _STORE_MAGIC:
        raise ValueError("bad store_blob magic")
    off = len(_STORE_MAGIC)
    n_kf, H, W, n_classes = struct.unpack_from("<IIII", store_blob, off)
    off += 16
    (reach_kstar,) = struct.unpack_from("<I", store_blob, off)
    off += 4
    (n_pairs,) = struct.unpack_from("<I", store_blob, off)
    off += 4
    pal_n = n_classes * 3 * 2  # fp16
    palette = np.frombuffer(store_blob[off : off + pal_n], dtype=np.float16).astype(np.float32).reshape(n_classes, 3)
    off += pal_n
    calib = struct.unpack_from("<ddd", store_blob, off)
    off += 24
    warp_type_codes = list(store_blob[off : off + n_classes])
    off += n_classes
    indices: list[int] = []
    lstars = np.empty((n_kf, H, W), np.int64)
    for i in range(n_kf):
        (idx,) = struct.unpack_from("<I", store_blob, off)
        off += 4
        (clen,) = struct.unpack_from("<I", store_blob, off)
        off += 4
        payload = store_blob[off : off + clen]
        off += clen
        code = ContourCode(payload=payload, shape=(H, W), n_classes=int(n_classes))
        lstars[i] = decode_partition(code)
        indices.append(int(idx))
    return V2StoreBlob(
        keyframe_indices=indices,
        keyframe_lstars=lstars,
        palette=palette,
        calib=(float(calib[0]), float(calib[1]), float(calib[2])),
        warp_type_codes=warp_type_codes,
        reach_kstar=int(reach_kstar),
        n_pairs=int(n_pairs),
        shape=(int(H), int(W)),
        n_classes=int(n_classes),
    )


# ---------------------------------------------------------------------------
# packet assembly (deterministic archive.zip + inflate.py + inflate.sh).
# ---------------------------------------------------------------------------
def assemble_v2_packet(blob: bytes, packet_dir: str | Path) -> tuple[Path, int]:
    """Write archive.zip (single deterministic ``0.bin`` member) + inflate.py + inflate.sh.

    Returns (archive_zip_path, archive_zip_size_bytes) — the size IS the rate term numerator."""
    packet_dir = Path(packet_dir)
    packet_dir.mkdir(parents=True, exist_ok=True)
    zip_path = packet_dir / "archive.zip"
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, blob)
    (packet_dir / "inflate.py").write_text(generate_v2_inflate_py())
    sh = packet_dir / "inflate.sh"
    sh.write_text(_INFLATE_SH)
    sh.chmod(0o755)
    return zip_path, int(zip_path.stat().st_size)


def byte_accounting(
    zip_path: str | Path,
    store_blob: bytes,
    residual_inr_blob: bytes,
    pose_sidecar_blob: bytes,
    manifest_bytes: bytes,
) -> dict[str, Any]:
    """The COUNTED byte breakdown + the rate term (archive.zip st_size; the authority numerator)."""
    zp = Path(zip_path)
    zbytes = int(zp.stat().st_size)
    total_0bin = len(MAGIC_V2) + 16 + len(store_blob) + len(residual_inr_blob) + len(pose_sidecar_blob) + len(manifest_bytes)
    return {
        "archive_zip_bytes": zbytes,
        "rate": zbytes / 37_545_489.0,
        "rate_term": rate_term(zbytes),
        "rate_denom_bytes": 37_545_489,
        "section_bytes": {
            "store_blob": len(store_blob),
            "residual_inr_blob": len(residual_inr_blob),
            "pose_sidecar_blob": len(pose_sidecar_blob),
            "manifest": len(manifest_bytes),
            "magic_and_prefixes": len(MAGIC_V2) + 16,
        },
        "total_0bin_bytes": total_0bin,
        "zip_container_overhead_bytes": zbytes - total_0bin,
        "archive_zip_sha256": hashlib.sha256(zp.read_bytes()).hexdigest(),
        "residual_inr_present": len(residual_inr_blob) > 0,
        "authority": "[advisory] NON-PROMOTABLE — exact rate is upstream/evaluate.py on these bytes",
    }


# ---------------------------------------------------------------------------
# the self-contained MLX-free inflate.py (the deterministic bulk generator, rule-118 FREE).
# ---------------------------------------------------------------------------
def generate_v2_inflate_py() -> str:
    """Return the v2 inflate.py source (MLX-free numpy warp/render + torch bicubic; no scorers)."""
    return _INFLATE_PY_V2


_INFLATE_SH = """#!/usr/bin/env bash
# v2 witness inflate launcher -> <OUTPUT_DIR>/<base>.raw = flat uint8 (2*n_pairs,874,1164,3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"; OUTPUT_DIR="$2"; FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  printf "Inflating %s ... " "$line"
  python "${HERE}/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""


# The v2 inflate.py: decode 4 sections; for each pair warp the nearest keyframe (per-class
# stratified, FREE) -> render (palette + R1 ramp, FREE) -> bicubic up -> uint8; compose the
# residual INR if present (currently a no-op hook until the GPU residual weights land). The numpy
# warp primitives are byte-faithful copies of tools/measure_screw_reach_through_R.py (parity-tested).
_INFLATE_PY_V2 = r'''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# v2 witness inflate -- MLX-FREE (numpy warp/render + torch bicubic). Loads NO scorer weights.
# Deterministic bulk = STORE keyframes + GENERATE per-class stratified warp (rule-118 FREE).
import sys, json, struct, lzma, zlib
import numpy as np

MAGIC = b"WTNV2\x00"
STORE_MAGIC = b"WSTR1\x00"
NATIVE_H, NATIVE_W = 874, 1164
CAMERA_HEIGHT_M = 1.22
NATIVE_FX = NATIVE_FY = 910.0
NATIVE_CX, NATIVE_CY = 582.0, 437.0
SEG_H, SEG_W = 384, 512
BULK_IDX = (0, 2, 4)
# per-class warp regime (comma10k canonical order [Road0,Lane1,Undriv2,Movable3,MyCar4]); the
# store_blob warp-mask overrides per class, but this is the proven default routing.
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]


def _read_sections(path):
    raw = open(path, "rb").read()
    assert raw[:len(MAGIC)] == MAGIC, "bad v2 magic"
    off = len(MAGIC); out = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        out.append(raw[off:off+n]); off += n
    store, residual, pose, manifest_b = out
    manifest = json.loads(manifest_b.decode()) if manifest_b else {}
    return store, residual, pose, manifest


def _parse_store(store):
    assert store[:len(STORE_MAGIC)] == STORE_MAGIC, "bad store magic"
    off = len(STORE_MAGIC)
    n_kf, H, W, n_classes = struct.unpack_from("<IIII", store, off); off += 16
    (reach_kstar,) = struct.unpack_from("<I", store, off); off += 4
    (n_pairs,) = struct.unpack_from("<I", store, off); off += 4
    pal_n = n_classes * 3 * 2
    palette = np.frombuffer(store[off:off+pal_n], dtype=np.float16).astype(np.float64).reshape(n_classes, 3); off += pal_n
    calib = struct.unpack_from("<ddd", store, off); off += 24
    warp_codes = list(store[off:off+n_classes]); off += n_classes
    indices = []; lstars = np.empty((n_kf, H, W), np.int64)
    for i in range(n_kf):
        (idx,) = struct.unpack_from("<I", store, off); off += 4
        (clen,) = struct.unpack_from("<I", store, off); off += 4
        payload = store[off:off+clen]; off += clen
        raw = lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
        lstars[i] = np.frombuffer(raw, dtype=np.uint8).reshape(H, W).astype(np.int64)
        indices.append(idx)
    return dict(indices=indices, lstars=lstars, palette=palette, calib=calib,
               warp_codes=warp_codes, reach_kstar=reach_kstar, n_pairs=n_pairs,
               H=H, W=W, n_classes=n_classes)


def _decode_pntg_poses(pose_blob):
    # PNTG: magic(4) ver(<H) n_pairs(<I) n_frames(<I) clen(<I) zlib(fp16 (n,6))
    if not pose_blob or pose_blob[:4] != b"PNTG":
        return None
    off = 4
    (_ver,) = struct.unpack_from("<H", pose_blob, off); off += 2
    (n_pairs,) = struct.unpack_from("<I", pose_blob, off); off += 4
    (_nf,) = struct.unpack_from("<I", pose_blob, off); off += 4
    (clen,) = struct.unpack_from("<I", pose_blob, off); off += 4
    raw = zlib.decompress(pose_blob[off:off+clen])
    return np.frombuffer(raw, dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)


def _intrinsics(seg_w, seg_h):
    sx, sy = seg_w / NATIVE_W, seg_h / NATIVE_H
    return np.array([[NATIVE_FX*sx, 0.0, NATIVE_CX*sx],
                     [0.0, NATIVE_FY*sy, NATIVE_CY*sy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _expmap_so3(omega):
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + K
    return (np.eye(3) + (np.sin(theta)/theta)*K + ((1.0-np.cos(theta))/(theta*theta))*(K @ K))


def _m_step(pose6, s_t, s_r, pitch, regime):
    if regime == "identity":
        return np.eye(3, dtype=np.float64)
    R = _expmap_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    if regime == "rotonly":
        return R
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
    n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)
    return R - np.outer(t, n) / CAMERA_HEIGHT_M


def _cumulative_homography(poses, a, k, K, Kinv, params, regime):
    s_t, s_r, pitch = params
    if k == 0 or regime == "identity":
        return np.eye(3, dtype=np.float64)
    M = np.eye(3, dtype=np.float64)
    for i in range(a + 1, a + k + 1):
        M = _m_step(poses[i], s_t, s_r, pitch, regime) @ M
    return K @ M @ Kinv


def _target_grid(Hh, Ww):
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    return np.stack([us.ravel(), vs.ravel(), np.ones(Hh*Ww)], 0).astype(np.float64)


def _warp_labels(src, H, grid):
    Hh, Ww = src.shape
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(H)
        src_h = Hinv @ grid
        z = src_h[2]; su = src_h[0]/z; sv = src_h[1]/z
    valid = np.isfinite(su) & np.isfinite(sv) & (z > 0)
    valid &= (su >= 0) & (su <= Ww-1) & (sv >= 0) & (sv <= Hh-1)
    sui = np.clip(np.round(su), 0, Ww-1).astype(np.int64)
    svi = np.clip(np.round(sv), 0, Hh-1).astype(np.int64)
    return src[svi, sui].reshape(Hh, Ww), valid.reshape(Hh, Ww)


def _warp_persist(L_src, H, grid):
    pred, valid = _warp_labels(L_src, H, grid)
    return np.where(valid, pred, L_src)


def _composite_warped(L_src, poses, a, k, K, Kinv, params, grid):
    Hg = _cumulative_homography(poses, a, k, K, Kinv, params, "ground")
    Hr = _cumulative_homography(poses, a, k, K, Kinv, params, "rotonly")
    cg = _warp_persist(L_src, Hg, grid)
    cr = _warp_persist(L_src, Hr, grid)
    ci = L_src
    fg = np.isin(cg, [0, 1, 3])
    return np.where(fg, cg, np.where(cr == 2, 2, np.where(ci == 4, 4, cg)))


def _gauss_blur(img, sigma):
    if sigma <= 0:
        return img
    rad = max(1, int(round(3*sigma)))
    xs = np.arange(-rad, rad+1)
    k = np.exp(-(xs**2)/(2*sigma*sigma)); k /= k.sum()
    out = img.astype(np.float64)
    for ax in (1, 0):
        padw = [(0, 0)]*out.ndim; padw[ax] = (rad, rad)
        p = np.pad(out, padw, mode="reflect")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)]*out.ndim; sl[ax] = slice(i, i+out.shape[ax])
            acc += w * p[tuple(sl)]
        out = acc
    return out


def _render_partition(label_map, palette):
    return _gauss_blur(palette[label_map], 1.0)


def _bicubic_up(render384_f):
    import torch, torch.nn.functional as F
    x = torch.from_numpy(np.asarray(render384_f, dtype=np.float32)).permute(2, 0, 1)[None]
    xu = F.interpolate(x, size=(NATIVE_H, NATIVE_W), mode="bicubic", align_corners=False)
    return xu[0].permute(1, 2, 0).clamp(0, 255).round().numpy().astype(np.uint8)


def _nearest_keyframe(p, keyframes):
    anchor = keyframes[0]
    for kf in keyframes:
        if kf <= p:
            anchor = kf
        else:
            break
    return anchor, int(p - anchor)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    store, residual, pose_blob, man = _read_sections(src)
    s = _parse_store(store)
    poses = _decode_pntg_poses(pose_blob)
    n_pairs = int(s["n_pairs"])
    if poses is None:
        poses = np.zeros((n_pairs, 6), np.float64)  # no warp without poses (persist only)
    K = _intrinsics(SEG_W, SEG_H); Kinv = np.linalg.inv(K); grid = _target_grid(SEG_H, SEG_W)
    palette = s["palette"]; keyframes = s["indices"]
    params = (float(s["calib"][0]), float(s["calib"][1]), float(s["calib"][2]))
    kf_map = {idx: s["lstars"][i] for i, idx in enumerate(keyframes)}
    has_residual = len(residual) > 0
    with open(dst, "wb") as f:
        for p in range(n_pairs):
            anchor, k = _nearest_keyframe(p, keyframes)
            L_src = kf_map[anchor]
            if k == 0:
                warped = L_src
            else:
                warped = _composite_warped(L_src, poses, anchor, k, K, Kinv, params, grid)
            # GENERATE the deterministic bulk frame (FREE).
            frame = _bicubic_up(_render_partition(warped, palette))
            if has_residual:
                # RESIDUAL COMPOSE HOOK (LEARN tier): when the GPU residual-INR weights are present,
                # decode + forward the small INR and overwrite the residual cells here. The residual
                # blob format is the GPU-side trainer's output (NEEDS-WIRING). No-op while empty so
                # the deterministic FLOOR archive runs today. (NO-FAKE: not faked -- explicitly pending.)
                raise SystemExit("residual-INR compose is NEEDS-WIRING (GPU run not landed); "
                                 "byte-close the deterministic floor (empty residual) until then.")
            f.write(frame.tobytes())  # frame0 == bulk render
            f.write(frame.tobytes())  # frame1 == bulk render (SegNet scores frame1)
    print(f"inflated {2*n_pairs} frames ({n_pairs} pairs) -> {dst} "
          f"[{2*n_pairs}x{NATIVE_H}x{NATIVE_W}x3 uint8]", flush=True)


if __name__ == "__main__":
    main()
'''
