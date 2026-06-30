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
    "build_residual_blob",
    "parse_residual_blob",
    "residual_inflate_reference",
    "assemble_v2_packet",
    "generate_v2_inflate_py",
    "byte_accounting",
    "V2StoreBlob",
    "V2ResidualBlob",
    "WARP_TYPE_CODE",
    "screw_regime_warp_codes",
]

MAGIC_V2 = b"WTNV2\x00"
_STORE_MAGIC = b"WSTR1\x00"
_RESIDUAL_MAGIC = b"WRES1\x00"

# warp-type codes for the per-class warp-mask (1 byte/class). Self-detected upstream; here just a code.
WARP_TYPE_CODE = {
    "ground_homography": 0,
    "identity": 1,
    "rotation_only": 2,
    "learn": 3,
}
_WARP_TYPE_NAME = {v: k for k, v in WARP_TYPE_CODE.items()}

# bridge the SCREW_REGIME names (tools.measure_screw_reach_through_R: ground/rotonly/identity) to the
# store_blob WARP_TYPE_CODE names. The store_blob warp-mask carries the PHYSICAL per-class regime so
# the inflate's _composite_warped can ROUTE each class (A3.2: consumed, not dead) -- NOT the
# store/learn DECISION (LEARN classes still ride the ground bulk; the residual INR overrides them).
_SCREW_REGIME_TO_WARP_NAME = {
    "ground": "ground_homography",
    "rotonly": "rotation_only",
    "identity": "identity",
}


def screw_regime_warp_codes(screw_regime: dict[int, str], n_classes: int = 5) -> list[int]:
    """Per-class PHYSICAL warp-regime codes for the store_blob warp-mask, derived from the
    ``SCREW_REGIME`` class->regime map (the single source of truth in
    ``tools.measure_screw_reach_through_R``). The inflate's ``_composite_warped`` CONSUMES these
    (A3.2: no hardcoded ``[0,1,3]/2/4`` routing; A3.1: one derivation shared by phase_a + phase_b).

    For the canonical comma rig ``{0:ground,1:ground,2:rotonly,3:ground,4:identity}`` this returns
    ``[0,0,2,0,1]``, which makes ``_composite_warped`` bit-identical to the proven
    ``composite_warped_labels`` router. SELF-DETECTED: the regime is a physical per-NAMED-class fact
    (Road rides the ground plane, sky rides rotation-only, the hood is static), not a SegNet index
    decision -- the canonical comma10k order is SCORER_FIXED.
    """
    return [WARP_TYPE_CODE[_SCREW_REGIME_TO_WARP_NAME[screw_regime[c]]] for c in range(n_classes)]


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
# residual_blob: the LEARN-tier INR (int8+brotli weights + per-frame code) + its cfg. The COUNTED
# rate term. The curvelet bank B is NOT stored (rule-118: regenerated free from the 5 bank scalars).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class V2ResidualBlob:
    """Parsed residual_blob (the COUNTED LEARN-tier INR + its forward cfg)."""

    params: dict[str, np.ndarray]       # dequantized weights (incl. "code") -- the INR
    manifest: dict[str, Any]            # INR + bank + learn-class/dilate cfg (the forward contract)


def _int8_sym(a: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quant (mirror of lever_b_levelset_generator._int8_symmetric)."""
    s = float(np.abs(np.asarray(a, np.float64)).max()) + 1e-8
    q = np.clip(np.round(np.asarray(a, np.float64) / s * 127.0), -127, 127).astype(np.int8)
    return q, (s / 127.0)


def build_residual_blob(params: dict[str, np.ndarray], inr_cfg: dict[str, Any]) -> bytes:
    """Serialize the residual INR: ``WRES1`` magic + <I manifest_len + manifest_json + <I base_len +
    brotli(int8 base) + <I code_len + brotli(int8 code). The bank ``B`` is EXCLUDED (rule-118 free;
    regenerated at decode from the 5 bank scalars). ``inr_cfg`` MUST carry the forward contract:
    n_hidden/hidden_dim/mod_dim/n_classes/activation/hosc_beta/hosc_omega/softmax_temp/wire_w0/
    wire_s0/chroma/render_h/render_w + bank_* + learn_classes + dilate."""
    import brotli

    base_order = [k for k in params if k != "code" and not (k == "B" or k.endswith("_B"))]
    base_chunks: list[bytes] = []
    shapes: dict[str, list[int]] = {}
    scales: dict[str, float] = {}
    for name in base_order:
        q, sc = _int8_sym(np.asarray(params[name], np.float32))
        base_chunks.append(q.tobytes())
        shapes[name] = list(np.asarray(params[name]).shape)
        scales[name] = float(sc)
    base_brotli = brotli.compress(b"".join(base_chunks), quality=11)
    if "code" not in params:
        raise ValueError("residual params must include the per-frame 'code' latent")
    qc, code_scale = _int8_sym(np.asarray(params["code"], np.float32))
    code_brotli = brotli.compress(qc.tobytes(), quality=11)

    manifest = dict(inr_cfg)
    manifest.update({
        "base_param_order": base_order,
        "base_shapes": shapes,
        "base_scales": scales,
        "code_shape": list(np.asarray(params["code"]).shape),
        "code_scale": float(code_scale),
        "learn_classes": [int(c) for c in inr_cfg.get("learn_classes", (1, 3))],
        "dilate": int(inr_cfg.get("dilate", 0)),
        # the composition mask mode (B2): the inflate re-derives the SAME mask the trainer used,
        # so train == inflate. boundary_annulus (default) covers ALL codim-1 flips, GT-free.
        "mask_mode": str(inr_cfg.get("mask_mode", "boundary_annulus")),
    })
    mj = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    buf = bytearray()
    buf += _RESIDUAL_MAGIC
    for chunk in (mj, base_brotli, code_brotli):
        buf += struct.pack("<I", len(chunk))
        buf += chunk
    return bytes(buf)


def parse_residual_blob(residual_blob: bytes) -> V2ResidualBlob:
    """Inverse of :func:`build_residual_blob` (dequantizes the INR weights + code)."""
    import brotli

    if residual_blob[: len(_RESIDUAL_MAGIC)] != _RESIDUAL_MAGIC:
        raise ValueError("bad residual_blob magic")
    off = len(_RESIDUAL_MAGIC)
    out: list[bytes] = []
    for _ in range(3):
        (n,) = struct.unpack_from("<I", residual_blob, off)
        off += 4
        out.append(residual_blob[off : off + n])
        off += n
    mj, base_brotli, code_brotli = out
    manifest = json.loads(mj.decode("utf-8"))
    base_flat = np.frombuffer(brotli.decompress(base_brotli), dtype=np.int8)
    params: dict[str, np.ndarray] = {}
    o = 0
    for name in manifest["base_param_order"]:
        shp = tuple(manifest["base_shapes"][name])
        n = int(np.prod(shp))
        params[name] = (base_flat[o : o + n].astype(np.float32) * float(manifest["base_scales"][name])).reshape(shp)
        o += n
    code_flat = np.frombuffer(brotli.decompress(code_brotli), dtype=np.int8)
    params["code"] = (code_flat.astype(np.float32) * float(manifest["code_scale"])).reshape(
        tuple(manifest["code_shape"]))
    return V2ResidualBlob(params=params, manifest=manifest)


def residual_inflate_reference(
    store: V2StoreBlob, residual: V2ResidualBlob, poses: np.ndarray, n_pairs: int
) -> np.ndarray:
    """NUMPY ORACLE (the NO-FAKE faithfulness anchor): the EXACT camera frames the residual inflate
    MUST produce, computed from the BUILT tac primitives (bulk_generator render/warp +
    levelset_rgb_forward_numpy + residual_compose). The parity test asserts the self-contained
    inflate.py output is bit-identical to this. Returns (2*n_pairs, 874, 1164, 3) uint8.

    Composition (the ONE rule): per pair p, warp the nearest keyframe -> bulk label -> bulk RGB
    (render res, pre-R); mask = derive_composition_mask(bulk_label, learn_classes, dilate); INR RGB
    via levelset_rgb_forward_numpy; composed = where(mask, INR, bulk); bicubic-up -> uint8."""
    from tac.boundary_math.lever_b_levelset_generator import (
        CurveletBankConfig,
        curvelet_directional_B,
        curvelet_feats,
        levelset_rgb_forward_numpy,
    )
    from tac.v2_compose import bulk_generator as _bg
    from tac.v2_compose.residual_compose import compose_residual_rgb, derive_composition_mask

    m = residual.manifest
    H, W = store.shape
    learn = tuple(int(c) for c in m["learn_classes"])
    dilate = int(m["dilate"])
    mask_mode = str(m.get("mask_mode", "boundary_annulus"))
    palette = np.asarray(store.palette, np.float64)
    params = residual.params
    code = np.asarray(params["code"], np.float64)

    bank = CurveletBankConfig(
        n_scales=int(m["bank_n_scales"]), n_orient0=int(m["bank_n_orient0"]),
        f0=float(m["bank_f0"]), base=float(m["bank_base"]), n_iso=int(m["bank_n_iso"]))
    B = curvelet_directional_B(bank, max_freq=m.get("max_bank_freq"))
    coords = _build_render_coords_np(H, W)
    feats = curvelet_feats(coords, B)

    keyframes = store.keyframe_indices
    kf_map = {idx: store.keyframe_lstars[i] for i, idx in enumerate(keyframes)}
    K = _bg.intrinsics_at(W, H)
    Kinv = np.linalg.inv(K)
    grid = _bg._target_grid(H, W)
    params_calib = store.calib

    def _fwd(code_idx: int) -> np.ndarray:
        rgb, _phi = levelset_rgb_forward_numpy(
            params, feats, code[code_idx], n_hidden=int(m["n_hidden"]), hidden_dim=int(m["hidden_dim"]),
            n_classes=int(m["n_classes"]), activation=str(m["activation"]),
            softmax_temp=float(m["softmax_temp"]), wire_w0=float(m["wire_w0"]), wire_s0=float(m["wire_s0"]),
            hosc_beta=float(m["hosc_beta"]), hosc_omega=float(m["hosc_omega"]), chroma=bool(m["chroma"]))
        return rgb.reshape(H, W, 3)

    frames = np.empty((2 * n_pairs, _bg.NATIVE_H, _bg.NATIVE_W, 3), np.uint8)
    for p in range(n_pairs):
        anchor, k = _bg.nearest_keyframe(p, keyframes)
        L_src = kf_map[anchor]
        warped = L_src if k == 0 else _bg.composite_warped_labels(
            L_src, poses, anchor, k, K, Kinv, params_calib, grid)
        bulk_rgb = _bg.render_partition(warped, palette)  # (H,W,3) pre-R
        mask = derive_composition_mask(warped, learn, dilate, mode=mask_mode)
        for fk in range(2):
            comp = compose_residual_rgb(bulk_rgb, _fwd(2 * p + fk), mask)
            frames[2 * p + fk] = _bg.bicubic_up_to_camera(comp).astype(np.uint8)
    return frames


def _build_render_coords_np(h: int, w: int) -> np.ndarray:
    """The render coord grid (mirror of train_witness_realized_through_R_mlx._build_render_coords)."""
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


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
RES_MAGIC = b"WRES1\x00"
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


def _composite_warped(L_src, poses, a, k, K, Kinv, params, grid, warp_codes):
    # warp_codes[c] = the per-class PHYSICAL warp-regime code (0=ground, 1=identity, 2=rotonly),
    # CONSUMED from the store_blob warp-mask (A3.2: no hardcoded [0,1,3]/2/4 routing). The decode
    # routes each target pixel by the warped-source label's class regime: ground foreground first,
    # then rotonly, then identity, else ground fallback. For the canonical comma rig
    # warp_codes=[0,0,2,0,1] this is bit-identical to the proven composite_warped_labels router.
    ground_cls = [c for c, wc in enumerate(warp_codes) if int(wc) == 0]
    iden_cls = [c for c, wc in enumerate(warp_codes) if int(wc) == 1]
    rot_cls = [c for c, wc in enumerate(warp_codes) if int(wc) == 2]
    Hg = _cumulative_homography(poses, a, k, K, Kinv, params, "ground")
    Hr = _cumulative_homography(poses, a, k, K, Kinv, params, "rotonly")
    cg = _warp_persist(L_src, Hg, grid)
    cr = _warp_persist(L_src, Hr, grid)
    ci = L_src
    fg = np.isin(cg, ground_cls)
    return np.where(fg, cg, np.where(np.isin(cr, rot_cls), cr, np.where(np.isin(ci, iden_cls), ci, cg)))


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


# ---- RESIDUAL INR (LEARN tier): self-contained MLX-free forward, bit-mirror of
# tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy + the curvelet bank +
# tac.v2_compose.residual_compose composition. Active ONLY when the residual section is non-empty.
import brotli


def _parse_residual(blob):
    assert blob[:len(RES_MAGIC)] == RES_MAGIC, "bad residual magic"
    off = len(RES_MAGIC); out = []
    for _ in range(3):
        (n,) = struct.unpack_from("<I", blob, off); off += 4
        out.append(blob[off:off+n]); off += n
    man = json.loads(out[0].decode())
    base_flat = np.frombuffer(brotli.decompress(out[1]), dtype=np.int8)
    params = {}; o = 0
    for name in man["base_param_order"]:
        shp = tuple(man["base_shapes"][name]); n = int(np.prod(shp))
        params[name] = (base_flat[o:o+n].astype(np.float32) * float(man["base_scales"][name])).reshape(shp)
        o += n
    code_flat = np.frombuffer(brotli.decompress(out[2]), dtype=np.int8)
    params["code"] = (code_flat.astype(np.float32) * float(man["code_scale"])).reshape(tuple(man["code_shape"]))
    return man, params


def _build_render_coords(h, w):
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def _curvelet_B(man):
    cols = []
    for j in range(int(man["bank_n_scales"])):
        f_j = float(man["bank_f0"]) * (float(man["bank_base"]) ** j)
        l_j = int(man["bank_n_orient0"]) * (2 ** (j // 2))
        for l in range(l_j):
            theta = np.pi * l / l_j
            cols.append(np.array([f_j*np.cos(theta), f_j*np.sin(theta)], dtype=np.float32))
    for i in range(int(man["bank_n_iso"])):
        theta = np.pi * i / max(int(man["bank_n_iso"]), 1)
        f_low = float(man["bank_f0"]) * 0.5
        cols.append(np.array([f_low*np.cos(theta), f_low*np.sin(theta)], dtype=np.float32))
    stacked = np.stack(cols, axis=1).astype(np.float32)
    mf = man.get("max_bank_freq")
    if mf is not None:
        norms = np.sqrt((stacked.astype(np.float64) ** 2).sum(axis=0))
        keep = norms <= float(mf) + 1e-6
        if not keep.any():
            keep = norms <= float(norms.min()) + 1e-6
        stacked = stacked[:, keep]
    return stacked


def _curvelet_feats(coords, B):
    with np.errstate(all="ignore"):
        proj = (2.0*np.pi) * (np.asarray(coords, np.float64) @ np.asarray(B, np.float64))
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


def _act_ls(u, kind, beta, omega, w0, s0):
    u = np.asarray(u, np.float64)
    if kind == "wire":
        return (np.cos(w0*u) * np.exp(-((s0*u)**2))).astype(np.float32)
    if kind == "hosc":
        return np.tanh(beta * np.sin(omega*u)).astype(np.float32)
    return np.maximum(u, 0.0).astype(np.float32)


def _levelset_forward(p, feats, code_row, man):
    nh, hd = int(man["n_hidden"]), int(man["hidden_dim"])
    kind = str(man["activation"]); beta = float(man["hosc_beta"]); omega = float(man["hosc_omega"])
    w0 = float(man["wire_w0"]); s0 = float(man["wire_s0"]); st = float(man["softmax_temp"])
    pp = {k: np.asarray(v, np.float64) for k, v in p.items()}
    feats = np.asarray(feats, np.float64); code_row = np.asarray(code_row, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = _act_ls(feats @ pp["in_proj.weight"].T + pp["in_proj.bias"], kind, beta, omega, w0, s0)
        film = (code_row @ pp["film.weight"].T + pp["film.bias"]).reshape(nh, 2, hd)
        has_pl = any(k.startswith("film_pl.") for k in pp)
        has_cc = any(k.startswith("concat_pl.") for k in pp)
        for li in range(nh):
            scale = 1.0 + film[li, 0]; shift = film[li, 1]
            if has_pl:
                pl = (code_row @ pp[f"film_pl.{li}.weight"].T + pp[f"film_pl.{li}.bias"]).reshape(2, hd)
                scale = scale + pl[0]; shift = shift + pl[1]
            pre = (h @ pp[f"hidden.{li}.weight"].T + pp[f"hidden.{li}.bias"]) * scale + shift
            if has_cc:
                pre = pre + (code_row @ pp[f"concat_pl.{li}.weight"].T + pp[f"concat_pl.{li}.bias"])
            h = _act_ls(pre, kind, beta, omega, w0, s0)
        phi = h @ pp["out_sdf.weight"].T + pp["out_sdf.bias"]
        tex = h @ pp["out_tex.weight"].T + pp["out_tex.bias"]
        z = phi / st; z = z - z.max(axis=-1, keepdims=True)
        soft = np.exp(z); soft = soft / soft.sum(axis=-1, keepdims=True)
        base = soft @ pp["palette"]
        rgb = (1.0 / (1.0 + np.exp(-(base + tex)))) * 255.0
        if not bool(man["chroma"]):
            luma = 0.299*rgb[:, 0:1] + 0.587*rgb[:, 1:2] + 0.114*rgb[:, 2:3]
            rgb = np.concatenate([luma, luma, luma], axis=-1)
    return rgb.astype(np.float32)


def _dilate_bool(mask, rounds):
    m = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    for _ in range(int(rounds)):
        out = m.copy()
        out[:-1, :] |= m[1:, :]; out[1:, :] |= m[:-1, :]
        out[:, :-1] |= m[:, 1:]; out[:, 1:] |= m[:, :-1]
        m = out
    return m


def _inter_class_boundary(label):
    # codim-1 inter-class boundary (both sides of every edge); self-detected, GT-free.
    b = np.zeros(label.shape, dtype=bool)
    ne_v = label[:-1, :] != label[1:, :]
    b[:-1, :] |= ne_v; b[1:, :] |= ne_v
    ne_h = label[:, :-1] != label[:, 1:]
    b[:, :-1] |= ne_h; b[:, 1:] |= ne_h
    return b


def _derive_comp_mask(warped_label, learn_classes, dilate, mode="boundary_annulus"):
    # bit-mirror of tac.v2_compose.residual_compose.derive_composition_mask (train == inflate).
    if mode == "learn_classes":
        mask = np.isin(warped_label, np.asarray(learn_classes, dtype=warped_label.dtype))
    elif mode == "union":
        mask = _inter_class_boundary(warped_label) | np.isin(
            warped_label, np.asarray(learn_classes, dtype=warped_label.dtype))
    else:  # boundary_annulus (default)
        mask = _inter_class_boundary(warped_label)
    if int(dilate) > 0:
        mask = _dilate_bool(mask, int(dilate))
    return mask.astype(bool)


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
    H, W = int(s["H"]), int(s["W"])  # store (== render == scorer) res; 384x512 in production.
    K = _intrinsics(W, H); Kinv = np.linalg.inv(K); grid = _target_grid(H, W)
    palette = s["palette"]; keyframes = s["indices"]
    params = (float(s["calib"][0]), float(s["calib"][1]), float(s["calib"][2]))
    warp_codes = s["warp_codes"]  # per-class physical warp-regime codes (A3.2: consumed, not dead)
    kf_map = {idx: s["lstars"][i] for i, idx in enumerate(keyframes)}
    has_residual = len(residual) > 0
    if has_residual:
        # RESIDUAL COMPOSE (LEARN tier): decode the small INR + regen its FREE curvelet feats
        # (rule-118; B is regenerated, not stored). composed = where(bulk_label_mask, INR, bulk).
        r_man, r_params = _parse_residual(residual)
        r_feats = _curvelet_feats(_build_render_coords(H, W), _curvelet_B(r_man))
        r_code = np.asarray(r_params["code"], np.float64)
        learn_classes = r_man["learn_classes"]; dilate = int(r_man["dilate"])
        mask_mode = r_man.get("mask_mode", "boundary_annulus")
    with open(dst, "wb") as f:
        for p in range(n_pairs):
            anchor, k = _nearest_keyframe(p, keyframes)
            L_src = kf_map[anchor]
            if k == 0:
                warped = L_src
            else:
                warped = _composite_warped(L_src, poses, anchor, k, K, Kinv, params, grid, warp_codes)
            # GENERATE the deterministic bulk RGB at render res (FREE, pre-R).
            bulk_rgb = _render_partition(warped, palette)
            if has_residual:
                # the bulk-LABEL-derived override region (FREE; regenerated, ships 0 bytes).
                mask = _derive_comp_mask(warped, learn_classes, dilate, mask_mode)[..., None]
                inr0 = _levelset_forward(r_params, r_feats, r_code[2*p+0], r_man).reshape(H, W, 3)
                inr1 = _levelset_forward(r_params, r_feats, r_code[2*p+1], r_man).reshape(H, W, 3)
                frame0 = _bicubic_up(np.where(mask, inr0, bulk_rgb))
                frame1 = _bicubic_up(np.where(mask, inr1, bulk_rgb))
                f.write(frame0.tobytes())  # frame0 == composed (pose)
                f.write(frame1.tobytes())  # frame1 == composed (SegNet scores frame1)
            else:
                frame = _bicubic_up(bulk_rgb)  # deterministic FLOOR (empty residual)
                f.write(frame.tobytes())  # frame0 == bulk render
                f.write(frame.tobytes())  # frame1 == bulk render
    print(f"inflated {2*n_pairs} frames ({n_pairs} pairs) -> {dst} "
          f"[{2*n_pairs}x{NATIVE_H}x{NATIVE_W}x3 uint8]", flush=True)


if __name__ == "__main__":
    main()
'''
