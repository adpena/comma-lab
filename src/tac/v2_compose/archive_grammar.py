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
import io
import json
import lzma
import struct
import zipfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from tac.boundary_math.dense_raster_lzma_baseline import ContourCode, decode_partition, encode_partition
from tac.contest_score import rate_term

__all__ = [
    "MAGIC_V2",
    "WARP_TYPE_CODE",
    "V2ArchiveSections",
    "V2PoseSidecarBlob",
    "V2ReceiverPayload",
    "V2ResidualBlob",
    "V2StoreBlob",
    "assemble_v2_packet",
    "build_residual_blob",
    "build_store_blob",
    "build_v2_archive_zip_bytes",
    "byte_accounting",
    "generate_v2_inflate_py",
    "pack_v2_archive",
    "parse_pose_sidecar_blob",
    "parse_residual_blob",
    "parse_store_blob",
    "residual_inflate_reference",
    "screw_regime_warp_codes",
    "unpack_v2_archive",
    "unpack_v2_sections",
    "validate_v2_receiver_payload",
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
    keyframe_lstars: np.ndarray         # (n_kf, H, W) int64 (decoded from dense-raster LZMA)
    palette: np.ndarray                 # (5, 3) float32
    calib: tuple[float, float, float]   # (s_t, s_r, pitch)
    warp_type_codes: list[int]          # per-class warp-type code (len n_classes)
    reach_kstar: int
    n_pairs: int
    shape: tuple[int, int]
    n_classes: int


@dataclass(frozen=True)
class V2ArchiveSections:
    """Strictly parsed raw sections of one ``WTNV2`` payload.

    Keeping the canonical manifest bytes alongside the decoded object is
    essential for packet linking: decoding JSON and dumping it again is not a
    byte-preserving identity operation.
    """

    store_blob: bytes
    residual_inr_blob: bytes
    pose_sidecar_blob: bytes
    manifest_bytes: bytes
    manifest: dict[str, Any]


def _require_bytes(value: Any, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    return bytes(value)


def _read_u32_chunk(blob: bytes, off: int, *, label: str) -> tuple[bytes, int]:
    if off < 0 or off + 4 > len(blob):
        raise ValueError(f"truncated {label} length prefix")
    (size,) = struct.unpack_from("<I", blob, off)
    off += 4
    end = off + int(size)
    if end > len(blob):
        raise ValueError(
            f"truncated {label} payload: declared {size} bytes, "
            f"only {len(blob) - off} remain"
        )
    return blob[off:end], end


def _strict_json_object(blob: bytes, *, label: str) -> dict[str, Any]:
    if not blob:
        return {}

    def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate {label} key: {key}")
            out[key] = value
        return out

    try:
        decoded = json.loads(
            blob.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite {label} JSON constant: {token}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return decoded


# ---------------------------------------------------------------------------
# 4-section pack / unpack (magic + length-prefix; modeled on the byte-close io_pack).
# ---------------------------------------------------------------------------
def pack_v2_archive(
    store_blob: bytes, residual_inr_blob: bytes, pose_sidecar_blob: bytes, manifest_bytes: bytes
) -> bytes:
    """Pack the 4 sections into the ``0.bin`` blob (MAGIC + 4x <I-length-prefixed chunks)."""
    chunks = tuple(
        _require_bytes(value, name)
        for value, name in (
            (store_blob, "store_blob"),
            (residual_inr_blob, "residual_inr_blob"),
            (pose_sidecar_blob, "pose_sidecar_blob"),
            (manifest_bytes, "manifest_bytes"),
        )
    )
    buf = bytearray()
    buf += MAGIC_V2
    for chunk in chunks:
        if len(chunk) > 0xFFFFFFFF:
            raise ValueError("v2 section exceeds the uint32 grammar limit")
        buf += struct.pack("<I", len(chunk))
        buf += chunk
    return bytes(buf)


def unpack_v2_sections(blob: bytes) -> V2ArchiveSections:
    """Strict inverse of :func:`pack_v2_archive`, preserving raw section bytes.

    Length-prefix truncation and trailing bytes are rejected.  These are packet
    identity failures, not recoverable metadata quirks.
    """

    blob = _require_bytes(blob, "blob")
    if blob[: len(MAGIC_V2)] != MAGIC_V2:
        raise ValueError("bad v2 archive magic")
    off = len(MAGIC_V2)
    out: list[bytes] = []
    for label in ("store", "residual", "pose", "manifest"):
        chunk, off = _read_u32_chunk(blob, off, label=label)
        out.append(chunk)
    if off != len(blob):
        raise ValueError(f"trailing bytes after v2 packet: {len(blob) - off}")
    store, residual, pose, manifest_b = out
    return V2ArchiveSections(
        store_blob=store,
        residual_inr_blob=residual,
        pose_sidecar_blob=pose,
        manifest_bytes=manifest_b,
        manifest=_strict_json_object(manifest_b, label="v2 manifest"),
    )


def unpack_v2_archive(blob: bytes) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    """Inverse of :func:`pack_v2_archive`. Returns (store, residual, pose, manifest_dict)."""

    parsed = unpack_v2_sections(blob)
    return (
        parsed.store_blob,
        parsed.residual_inr_blob,
        parsed.pose_sidecar_blob,
        parsed.manifest,
    )


# ---------------------------------------------------------------------------
# store_blob: keyframes (dense-raster LZMA) + palette + calib + warp-mask + kstar.
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
    """Serialize the STORE payload. Keyframes -> dense-raster LZMA (bit-exact). Palette fp16,
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
    # keyframes: idx(<I) + lzma_len(<I) + dense-label LZMA payload (per keyframe)
    for i, idx in enumerate(keyframe_indices):
        code: ContourCode = encode_partition(kf[i], n_classes=n_classes)
        buf += struct.pack("<I", int(idx))
        buf += struct.pack("<I", len(code.payload))
        buf += code.payload
    return bytes(buf)


def parse_store_blob(store_blob: bytes) -> V2StoreBlob:
    """Inverse of :func:`build_store_blob` (decodes keyframes bit-exact via dense-raster LZMA)."""
    store_blob = _require_bytes(store_blob, "store_blob")
    if store_blob[: len(_STORE_MAGIC)] != _STORE_MAGIC:
        raise ValueError("bad store_blob magic")
    off = len(_STORE_MAGIC)
    if off + 24 > len(store_blob):
        raise ValueError("truncated store_blob fixed header")
    n_kf, H, W, n_classes = struct.unpack_from("<IIII", store_blob, off)
    off += 16
    (reach_kstar,) = struct.unpack_from("<I", store_blob, off)
    off += 4
    (n_pairs,) = struct.unpack_from("<I", store_blob, off)
    off += 4
    if H == 0 or W == 0 or H > 4096 or W > 4096:
        raise ValueError(f"invalid store_blob spatial shape: {(H, W)}")
    if n_classes == 0 or n_classes > 255:
        raise ValueError(f"invalid store_blob class count: {n_classes}")
    if n_pairs == 0 or n_kf == 0 or n_kf > n_pairs:
        raise ValueError(f"invalid keyframe/pair counts: n_kf={n_kf}, n_pairs={n_pairs}")
    if int(n_kf) * int(H) * int(W) > 200_000_000:
        raise ValueError("store_blob decoded keyframes exceed the parser allocation limit")
    pal_n = n_classes * 3 * 2  # fp16
    fixed_tail = int(pal_n) + 24 + int(n_classes)
    if off + fixed_tail > len(store_blob):
        raise ValueError("truncated store_blob palette/calibration/warp header")
    palette = np.frombuffer(store_blob[off : off + pal_n], dtype=np.float16).astype(np.float32).reshape(n_classes, 3)
    off += pal_n
    calib = struct.unpack_from("<ddd", store_blob, off)
    off += 24
    if not np.isfinite(palette).all() or not np.isfinite(np.asarray(calib)).all():
        raise ValueError("store_blob palette/calibration contains non-finite values")
    warp_type_codes = list(store_blob[off : off + n_classes])
    off += n_classes
    unknown_warp_codes = sorted(set(warp_type_codes) - set(_WARP_TYPE_NAME))
    if unknown_warp_codes:
        raise ValueError(f"unknown store_blob warp type codes: {unknown_warp_codes}")
    indices: list[int] = []
    lstars = np.empty((n_kf, H, W), np.int64)
    for i in range(n_kf):
        if off + 4 > len(store_blob):
            raise ValueError(f"truncated keyframe[{i}] index")
        (idx,) = struct.unpack_from("<I", store_blob, off)
        off += 4
        payload, off = _read_u32_chunk(store_blob, off, label=f"keyframe[{i}] contour")
        if idx >= n_pairs:
            raise ValueError(f"keyframe[{i}] index {idx} is outside n_pairs={n_pairs}")
        if int(idx) in indices:
            raise ValueError(f"duplicate keyframe index: {idx}")
        code = ContourCode(payload=payload, shape=(H, W), n_classes=int(n_classes))
        try:
            lstars[i] = decode_partition(code)
        except (lzma.LZMAError, ValueError) as exc:
            raise ValueError(f"keyframe[{i}] contour failed to decode") from exc
        if np.any(lstars[i] < 0) or np.any(lstars[i] >= n_classes):
            raise ValueError(f"keyframe[{i}] decoded labels outside class range")
        indices.append(int(idx))
    if off != len(store_blob):
        raise ValueError(f"trailing bytes after store_blob: {len(store_blob) - off}")
    if indices[0] != 0 or any(right <= left for left, right in pairwise(indices)):
        raise ValueError("store_blob keyframe indices must start at 0 and strictly increase")
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


@dataclass(frozen=True)
class V2PoseSidecarBlob:
    """Strict parse of the receiver-consumed ``PNTG`` pose section."""

    version: int
    n_pairs: int
    n_frames: int
    poses: np.ndarray


@dataclass(frozen=True)
class V2ReceiverPayload:
    """All receiver-consumed sections after strict inner parse-back."""

    sections: V2ArchiveSections
    store: V2StoreBlob
    residual: V2ResidualBlob | None
    pose_sidecar: V2PoseSidecarBlob | None


def _int8_sym(a: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quant (mirror of lever_b_levelset_generator._int8_symmetric)."""
    s = float(np.abs(np.asarray(a, np.float64)).max()) + 1e-8
    q = np.clip(np.round(np.asarray(a, np.float64) / s * 127.0), -127, 127).astype(np.int8)
    return q, (s / 127.0)


def _expected_residual_parameter_names(
    parameter_names: set[str],
    *,
    n_hidden: int,
) -> set[str]:
    expected = {
        "code",
        "in_proj.weight",
        "in_proj.bias",
        "film.weight",
        "film.bias",
        "out_sdf.weight",
        "out_sdf.bias",
        "out_tex.weight",
        "out_tex.bias",
        "palette",
    }
    for layer in range(n_hidden):
        expected.update(
            {
                f"hidden.{layer}.weight",
                f"hidden.{layer}.bias",
            }
        )
    if any(name.startswith("film_pl.") for name in parameter_names):
        for layer in range(n_hidden):
            expected.update(
                {
                    f"film_pl.{layer}.weight",
                    f"film_pl.{layer}.bias",
                }
            )
    if any(name.startswith("concat_pl.") for name in parameter_names):
        for layer in range(n_hidden):
            expected.update(
                {
                    f"concat_pl.{layer}.weight",
                    f"concat_pl.{layer}.bias",
                }
            )
    return expected


def build_residual_blob(params: dict[str, np.ndarray], inr_cfg: dict[str, Any]) -> bytes:
    """Serialize the residual INR: ``WRES1`` magic + <I manifest_len + manifest_json + <I base_len +
    brotli(int8 base) + <I code_len + brotli(int8 code). The bank ``B`` is EXCLUDED (rule-118 free;
    regenerated at decode from the 5 bank scalars). ``inr_cfg`` MUST carry the forward contract:
    n_hidden/hidden_dim/mod_dim/n_classes/activation/hosc_beta/hosc_omega/softmax_temp/wire_w0/
    wire_s0/chroma/render_h/render_w + bank_* + learn_classes + dilate."""
    import brotli

    n_hidden = inr_cfg.get("n_hidden")
    if isinstance(n_hidden, bool) or not isinstance(n_hidden, int) or n_hidden <= 0:
        raise ValueError("residual n_hidden must be a positive exact integer")
    serialized_names = {
        name for name in params if not (name == "B" or name.endswith("_B"))
    }
    expected_names = _expected_residual_parameter_names(
        serialized_names,
        n_hidden=n_hidden,
    )
    if serialized_names != expected_names:
        raise ValueError(
            "residual serialized parameter set differs from receiver-consumed set: "
            f"missing={sorted(expected_names - serialized_names)}, "
            f"extra={sorted(serialized_names - expected_names)}"
        )
    base_order = [name for name in params if name in expected_names and name != "code"]
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

    residual_blob = _require_bytes(residual_blob, "residual_blob")
    if residual_blob[: len(_RESIDUAL_MAGIC)] != _RESIDUAL_MAGIC:
        raise ValueError("bad residual_blob magic")
    off = len(_RESIDUAL_MAGIC)
    out: list[bytes] = []
    for label in ("residual manifest", "residual base", "residual code"):
        chunk, off = _read_u32_chunk(residual_blob, off, label=label)
        out.append(chunk)
    if off != len(residual_blob):
        raise ValueError(f"trailing bytes after residual_blob: {len(residual_blob) - off}")
    mj, base_brotli, code_brotli = out
    manifest = _strict_json_object(mj, label="residual manifest")
    required = {
        "base_param_order",
        "base_shapes",
        "base_scales",
        "code_shape",
        "code_scale",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"residual manifest missing required keys: {missing}")
    order = manifest["base_param_order"]
    shapes = manifest["base_shapes"]
    scales = manifest["base_scales"]
    if not isinstance(order, list) or len(order) != len(set(order)):
        raise ValueError("residual base_param_order must be a unique list")
    if not isinstance(shapes, dict) or not isinstance(scales, dict):
        raise ValueError("residual base_shapes/base_scales must be JSON objects")

    def _shape_size(value: Any, *, label: str) -> tuple[tuple[int, ...], int]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{label} must be a non-empty dimension list")
        dims: list[int] = []
        size = 1
        for dim in value:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"{label} contains an invalid dimension")
            size *= int(dim)
            if size > 1_000_000_000:
                raise ValueError(f"{label} exceeds the parser allocation limit")
            dims.append(int(dim))
        return tuple(dims), size

    def _positive_finite_scale(value: Any, *, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite positive number")
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{label} must be a finite positive number")
        return parsed

    base_shapes: dict[str, tuple[int, ...]] = {}
    expected_base_size = 0
    for raw_name in order:
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("residual base_param_order contains an invalid name")
        if raw_name not in shapes or raw_name not in scales:
            raise ValueError(f"residual manifest lacks shape/scale for {raw_name}")
        shape, size = _shape_size(shapes[raw_name], label=f"base_shapes[{raw_name}]")
        _positive_finite_scale(scales[raw_name], label=f"base_scales[{raw_name}]")
        base_shapes[raw_name] = shape
        expected_base_size += size
        if expected_base_size > 1_000_000_000:
            raise ValueError("residual base payload exceeds the parser allocation limit")
    code_shape, expected_code_size = _shape_size(manifest["code_shape"], label="code_shape")
    _positive_finite_scale(manifest["code_scale"], label="code_scale")
    try:
        base_raw = brotli.decompress(base_brotli)
        code_raw = brotli.decompress(code_brotli)
    except brotli.error as exc:
        raise ValueError("residual brotli payload failed to decode") from exc
    if len(base_raw) != expected_base_size:
        raise ValueError(
            f"residual base size differs: expected {expected_base_size}, got {len(base_raw)}"
        )
    if len(code_raw) != expected_code_size:
        raise ValueError(
            f"residual code size differs: expected {expected_code_size}, got {len(code_raw)}"
        )
    base_flat = np.frombuffer(base_raw, dtype=np.int8)
    params: dict[str, np.ndarray] = {}
    o = 0
    for name in order:
        shp = base_shapes[name]
        n = int(np.prod(shp, dtype=np.int64))
        params[name] = (base_flat[o : o + n].astype(np.float32) * float(manifest["base_scales"][name])).reshape(shp)
        o += n
    code_flat = np.frombuffer(code_raw, dtype=np.int8)
    params["code"] = (code_flat.astype(np.float32) * float(manifest["code_scale"])).reshape(
        code_shape)
    return V2ResidualBlob(params=params, manifest=manifest)


def parse_pose_sidecar_blob(pose_blob: bytes) -> V2PoseSidecarBlob:
    """Strictly parse the exact ``PNTG`` dialect consumed by v2 inflate."""

    import zlib

    pose_blob = _require_bytes(pose_blob, "pose_blob")
    if pose_blob[:4] != b"PNTG":
        raise ValueError("bad PNTG magic")
    off = 4
    if off + 10 > len(pose_blob):
        raise ValueError("truncated PNTG fixed header")
    (version,) = struct.unpack_from("<H", pose_blob, off)
    off += 2
    (n_pairs,) = struct.unpack_from("<I", pose_blob, off)
    off += 4
    (n_frames,) = struct.unpack_from("<I", pose_blob, off)
    off += 4
    if version != 1:
        raise ValueError(f"unsupported PNTG version: {version}")
    if n_pairs == 0 or n_pairs > 1_000_000:
        raise ValueError(f"invalid PNTG pair count: {n_pairs}")
    if n_frames != 2 * n_pairs:
        raise ValueError(
            f"PNTG frame count differs: expected {2 * n_pairs}, got {n_frames}"
        )
    payload, off = _read_u32_chunk(pose_blob, off, label="PNTG compressed poses")
    if off != len(pose_blob):
        raise ValueError(f"trailing bytes after PNTG poses: {len(pose_blob) - off}")
    expected = int(n_pairs) * 6 * np.dtype(np.float16).itemsize
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload, expected + 1)
        if decoder.unconsumed_tail or len(raw) > expected:
            raise ValueError("decoded PNTG pose payload exceeds declared shape")
        raw += decoder.flush()
    except zlib.error as exc:
        raise ValueError("PNTG compressed poses failed to decode") from exc
    if not decoder.eof or decoder.unused_data:
        raise ValueError("PNTG compressed poses contain an incomplete or trailing stream")
    if len(raw) != expected:
        raise ValueError(
            f"decoded PNTG pose size differs: expected {expected}, got {len(raw)}"
        )
    poses = np.frombuffer(raw, dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)
    if not np.isfinite(poses).all():
        raise ValueError("PNTG poses contain non-finite values")
    return V2PoseSidecarBlob(
        version=int(version),
        n_pairs=int(n_pairs),
        n_frames=int(n_frames),
        poses=poses,
    )


def _validate_residual_receiver_contract(
    residual: V2ResidualBlob,
    store: V2StoreBlob,
) -> None:
    manifest = residual.manifest
    required = {
        "activation",
        "bank_base",
        "bank_f0",
        "bank_n_iso",
        "bank_n_orient0",
        "bank_n_scales",
        "chroma",
        "dilate",
        "hidden_dim",
        "hosc_beta",
        "hosc_omega",
        "learn_classes",
        "mod_dim",
        "n_classes",
        "n_hidden",
        "render_h",
        "render_w",
        "softmax_temp",
        "wire_s0",
        "wire_w0",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"residual forward contract missing keys: {missing}")

    def _integer(name: str, *, allow_zero: bool = False) -> int:
        value = manifest[name]
        lower = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"residual {name} must be an integer >= {lower}")
        return int(value)

    def _positive_number(name: str) -> float:
        value = manifest[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"residual {name} must be a finite positive number")
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"residual {name} must be a finite positive number")
        return parsed

    n_hidden = _integer("n_hidden")
    hidden_dim = _integer("hidden_dim")
    mod_dim = _integer("mod_dim")
    n_classes = _integer("n_classes")
    render_h = _integer("render_h")
    render_w = _integer("render_w")
    bank_n_scales = _integer("bank_n_scales")
    bank_n_orient0 = _integer("bank_n_orient0")
    bank_n_iso = _integer("bank_n_iso", allow_zero=True)
    for name in (
        "bank_base",
        "bank_f0",
        "hosc_beta",
        "hosc_omega",
        "softmax_temp",
        "wire_s0",
        "wire_w0",
    ):
        _positive_number(name)
    if n_classes != store.n_classes:
        raise ValueError("residual/store class counts differ")
    if (render_h, render_w) != store.shape:
        raise ValueError("residual render shape differs from store shape")
    if manifest["activation"] not in {"hosc", "relu", "wire"}:
        raise ValueError("residual activation is not receiver-supported")
    if not isinstance(manifest["chroma"], bool):
        raise ValueError("residual chroma must be boolean")
    dilate = _integer("dilate", allow_zero=True)
    del dilate
    mask_mode = manifest.get("mask_mode", "boundary_annulus")
    if mask_mode not in {"boundary_annulus", "learn_classes", "union"}:
        raise ValueError("residual mask_mode is not receiver-supported")
    learn_classes = manifest["learn_classes"]
    if (
        not isinstance(learn_classes, list)
        or not learn_classes
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= n_classes
            for value in learn_classes
        )
        or len(learn_classes) != len(set(learn_classes))
    ):
        raise ValueError("residual learn_classes are outside the store class domain")

    code = residual.params["code"]
    if code.ndim != 2 or code.shape != (2 * store.n_pairs, mod_dim):
        raise ValueError(
            "residual code shape differs from receiver frame/modulation domain: "
            f"{code.shape} != {(2 * store.n_pairs, mod_dim)}"
        )

    from tac.boundary_math.lever_b_levelset_generator import (
        CurveletBankConfig,
        curvelet_directional_B,
    )

    bank = CurveletBankConfig(
        n_scales=bank_n_scales,
        n_orient0=bank_n_orient0,
        f0=float(manifest["bank_f0"]),
        base=float(manifest["bank_base"]),
        n_iso=bank_n_iso,
    )
    max_bank_freq = manifest.get("max_bank_freq")
    if max_bank_freq is not None:
        if (
            isinstance(max_bank_freq, bool)
            or not isinstance(max_bank_freq, (int, float))
            or not np.isfinite(float(max_bank_freq))
            or float(max_bank_freq) <= 0.0
        ):
            raise ValueError("residual max_bank_freq must be null or finite positive")
        max_bank_freq = float(max_bank_freq)
    in_features = 2 * int(
        curvelet_directional_B(bank, max_freq=max_bank_freq).shape[1]
    )
    expected_shapes: dict[str, tuple[int, ...]] = {
        "in_proj.weight": (hidden_dim, in_features),
        "in_proj.bias": (hidden_dim,),
        "film.weight": (2 * hidden_dim * n_hidden, mod_dim),
        "film.bias": (2 * hidden_dim * n_hidden,),
        "out_sdf.weight": (n_classes, hidden_dim),
        "out_sdf.bias": (n_classes,),
        "out_tex.weight": (3, hidden_dim),
        "out_tex.bias": (3,),
        "palette": (n_classes, 3),
    }
    for layer in range(n_hidden):
        expected_shapes[f"hidden.{layer}.weight"] = (hidden_dim, hidden_dim)
        expected_shapes[f"hidden.{layer}.bias"] = (hidden_dim,)
    has_film_pl = any(name.startswith("film_pl.") for name in residual.params)
    has_concat_pl = any(name.startswith("concat_pl.") for name in residual.params)
    for layer in range(n_hidden):
        if has_film_pl:
            expected_shapes[f"film_pl.{layer}.weight"] = (2 * hidden_dim, mod_dim)
            expected_shapes[f"film_pl.{layer}.bias"] = (2 * hidden_dim,)
        if has_concat_pl:
            expected_shapes[f"concat_pl.{layer}.weight"] = (hidden_dim, mod_dim)
            expected_shapes[f"concat_pl.{layer}.bias"] = (hidden_dim,)
    expected_parameter_names = set(expected_shapes) | {"code"}
    if set(residual.params) != expected_parameter_names:
        raise ValueError(
            "residual parameter set differs from receiver-consumed set: "
            f"missing={sorted(expected_parameter_names - set(residual.params))}, "
            f"extra={sorted(set(residual.params) - expected_parameter_names)}"
        )
    for name, expected_shape in expected_shapes.items():
        value = residual.params.get(name)
        if value is None or value.shape != expected_shape:
            actual = None if value is None else value.shape
            raise ValueError(
                f"residual parameter {name} shape differs: {actual} != {expected_shape}"
            )


def validate_v2_receiver_payload(blob: bytes) -> V2ReceiverPayload:
    """Strictly parse every section the generated receiver will consume.

    Empty residual and pose sections are the only optional sections in the
    current grammar.  The result proves syntax and shape survival, not score or
    semantic equivalence.
    """

    sections = unpack_v2_sections(blob)
    store = parse_store_blob(sections.store_blob)
    residual = (
        None
        if not sections.residual_inr_blob
        else parse_residual_blob(sections.residual_inr_blob)
    )
    if residual is not None:
        _validate_residual_receiver_contract(residual, store)
    pose_sidecar = (
        None
        if not sections.pose_sidecar_blob
        else parse_pose_sidecar_blob(sections.pose_sidecar_blob)
    )
    if pose_sidecar is not None and pose_sidecar.n_pairs != store.n_pairs:
        raise ValueError(
            "store/PNTG pair counts differ: "
            f"{store.n_pairs} != {pose_sidecar.n_pairs}"
        )
    return V2ReceiverPayload(
        sections=sections,
        store=store,
        residual=residual,
        pose_sidecar=pose_sidecar,
    )


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
        warped = L_src if k == 0 else _bg.composite_warped_labels_by_codes(
            L_src,
            poses,
            anchor,
            k,
            K,
            Kinv,
            params_calib,
            grid,
            store.warp_type_codes,
        )
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
def build_v2_archive_zip_bytes(blob: bytes) -> bytes:
    """Return the deterministic single-member ``archive.zip`` for ``blob``."""

    blob = _require_bytes(blob, "blob")
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, blob)
    return buffer.getvalue()


def assemble_v2_packet(blob: bytes, packet_dir: str | Path) -> tuple[Path, int]:
    """Write archive.zip (single deterministic ``0.bin`` member) + inflate.py + inflate.sh.

    Returns (archive_zip_path, archive_zip_size_bytes) — the size IS the rate term numerator."""
    validate_v2_receiver_payload(blob)
    packet_dir = Path(packet_dir)
    packet_dir.mkdir(parents=True, exist_ok=True)
    zip_path = packet_dir / "archive.zip"
    zip_path.write_bytes(build_v2_archive_zip_bytes(blob))
    (packet_dir / "inflate.py").write_text(generate_v2_inflate_py())
    sh = packet_dir / "inflate.sh"
    sh.write_text(generate_v2_inflate_sh())
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


def generate_v2_inflate_sh() -> str:
    """Return the deterministic contest packet launcher paired with inflate.py."""

    return _INFLATE_SH


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


def _read_chunk(raw, off, label):
    if off < 0 or off + 4 > len(raw):
        raise ValueError("truncated %s length prefix" % label)
    (n,) = struct.unpack_from("<I", raw, off); off += 4
    end = off + int(n)
    if end > len(raw):
        raise ValueError("truncated %s payload" % label)
    return raw[off:end], end


def _json_object(raw, label):
    def reject_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate %s key: %s" % (label, key))
            out[key] = value
        return out
    def reject_constant(token):
        raise ValueError("non-finite %s JSON constant: %s" % (label, token))
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
                       parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("%s must be a JSON object" % label)
    return value


def _read_sections(path):
    raw = open(path, "rb").read()
    if raw[:len(MAGIC)] != MAGIC:
        raise ValueError("bad v2 magic")
    off = len(MAGIC); out = []
    for label in ("store", "residual", "pose", "manifest"):
        chunk, off = _read_chunk(raw, off, label)
        out.append(chunk)
    if off != len(raw):
        raise ValueError("trailing bytes after v2 packet")
    store, residual, pose, manifest_b = out
    manifest = _json_object(manifest_b, "v2 manifest") if manifest_b else {}
    return store, residual, pose, manifest


def _parse_store(store):
    if store[:len(STORE_MAGIC)] != STORE_MAGIC:
        raise ValueError("bad store magic")
    off = len(STORE_MAGIC)
    if off + 24 > len(store):
        raise ValueError("truncated store fixed header")
    n_kf, H, W, n_classes = struct.unpack_from("<IIII", store, off); off += 16
    (reach_kstar,) = struct.unpack_from("<I", store, off); off += 4
    (n_pairs,) = struct.unpack_from("<I", store, off); off += 4
    if H == 0 or W == 0 or H > 4096 or W > 4096:
        raise ValueError("invalid store spatial shape")
    if n_classes == 0 or n_classes > 255 or n_pairs == 0 or n_kf == 0 or n_kf > n_pairs:
        raise ValueError("invalid store counts")
    if int(n_kf) * int(H) * int(W) > 200000000:
        raise ValueError("store decoded keyframes exceed allocation limit")
    pal_n = n_classes * 3 * 2
    if off + pal_n + 24 + n_classes > len(store):
        raise ValueError("truncated store palette/calibration/warp header")
    palette = np.frombuffer(store[off:off+pal_n], dtype=np.float16).astype(np.float64).reshape(n_classes, 3); off += pal_n
    calib = struct.unpack_from("<ddd", store, off); off += 24
    if not np.isfinite(palette).all() or not np.isfinite(np.asarray(calib)).all():
        raise ValueError("store palette/calibration contains non-finite values")
    warp_codes = list(store[off:off+n_classes]); off += n_classes
    if any(code not in (0, 1, 2, 3) for code in warp_codes):
        raise ValueError("unknown store warp code")
    indices = []; lstars = np.empty((n_kf, H, W), np.int64)
    for i in range(n_kf):
        if off + 4 > len(store):
            raise ValueError("truncated keyframe index")
        (idx,) = struct.unpack_from("<I", store, off); off += 4
        payload, off = _read_chunk(store, off, "keyframe contour")
        if idx >= n_pairs or idx in indices:
            raise ValueError("invalid or duplicate keyframe index")
        try:
            raw = lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
        except lzma.LZMAError as exc:
            raise ValueError("keyframe contour failed to decode") from exc
        if len(raw) != H * W:
            raise ValueError("decoded keyframe size differs")
        lstars[i] = np.frombuffer(raw, dtype=np.uint8).reshape(H, W).astype(np.int64)
        if np.any(lstars[i] < 0) or np.any(lstars[i] >= n_classes):
            raise ValueError("decoded keyframe labels outside class range")
        indices.append(idx)
    if off != len(store):
        raise ValueError("trailing bytes after store")
    if indices[0] != 0 or any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("store keyframe indices must start at 0 and strictly increase")
    return dict(indices=indices, lstars=lstars, palette=palette, calib=calib,
               warp_codes=warp_codes, reach_kstar=reach_kstar, n_pairs=n_pairs,
               H=H, W=W, n_classes=n_classes)


def _decode_pntg_poses(pose_blob):
    # PNTG: magic(4) ver(<H) n_pairs(<I) n_frames(<I) clen(<I) zlib(fp16 (n,6))
    if not pose_blob:
        return None
    if pose_blob[:4] != b"PNTG":
        raise ValueError("bad PNTG magic")
    off = 4
    if off + 10 > len(pose_blob):
        raise ValueError("truncated PNTG fixed header")
    (_ver,) = struct.unpack_from("<H", pose_blob, off); off += 2
    (n_pairs,) = struct.unpack_from("<I", pose_blob, off); off += 4
    (n_frames,) = struct.unpack_from("<I", pose_blob, off); off += 4
    if _ver != 1:
        raise ValueError("unsupported PNTG version")
    if n_pairs == 0 or n_pairs > 1000000 or n_frames != 2 * n_pairs:
        raise ValueError("invalid PNTG pair/frame counts")
    payload, off = _read_chunk(pose_blob, off, "PNTG compressed poses")
    if off != len(pose_blob):
        raise ValueError("trailing bytes after PNTG poses")
    expected = int(n_pairs) * 6 * 2
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, expected + 1)
    if decoder.unconsumed_tail or len(raw) > expected:
        raise ValueError("decoded PNTG pose payload exceeds declared shape")
    raw += decoder.flush()
    if not decoder.eof or decoder.unused_data:
        raise ValueError("incomplete or trailing PNTG compressed stream")
    if len(raw) != expected:
        raise ValueError("decoded PNTG pose size differs")
    poses = np.frombuffer(raw, dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)
    if not np.isfinite(poses).all():
        raise ValueError("PNTG poses contain non-finite values")
    return poses


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
    if blob[:len(RES_MAGIC)] != RES_MAGIC:
        raise ValueError("bad residual magic")
    off = len(RES_MAGIC); out = []
    for label in ("residual manifest", "residual base", "residual code"):
        chunk, off = _read_chunk(blob, off, label)
        out.append(chunk)
    if off != len(blob):
        raise ValueError("trailing bytes after residual")
    man = _json_object(out[0], "residual manifest")
    order = man.get("base_param_order")
    shapes = man.get("base_shapes")
    scales = man.get("base_scales")
    if not isinstance(order, list) or len(order) != len(set(order)):
        raise ValueError("invalid residual parameter order")
    if not isinstance(shapes, dict) or not isinstance(scales, dict):
        raise ValueError("invalid residual shape/scale tables")
    def shape_size(value, label):
        if not isinstance(value, list) or not value:
            raise ValueError("invalid %s" % label)
        size = 1
        for dim in value:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ValueError("invalid %s dimension" % label)
            size *= int(dim)
            if size > 1000000000:
                raise ValueError("%s exceeds allocation limit" % label)
        return tuple(value), size
    def finite_scale(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid %s" % label)
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError("invalid %s" % label)
        return parsed
    expected = 0
    parsed_shapes = {}
    for name in order:
        if not isinstance(name, str) or not name or name not in shapes or name not in scales:
            raise ValueError("incomplete residual parameter metadata")
        parsed_shape, size = shape_size(shapes[name], "residual parameter shape")
        finite_scale(scales[name], "residual parameter scale")
        parsed_shapes[name] = parsed_shape
        expected += size
        if expected > 1000000000:
            raise ValueError("residual base exceeds allocation limit")
    base_raw = brotli.decompress(out[1])
    if len(base_raw) != expected:
        raise ValueError("decoded residual base size differs")
    base_flat = np.frombuffer(base_raw, dtype=np.int8)
    params = {}; o = 0
    for name in order:
        shp = parsed_shapes[name]; n = int(np.prod(shp))
        params[name] = (base_flat[o:o+n].astype(np.float32) * float(man["base_scales"][name])).reshape(shp)
        o += n
    code_shape, code_expected = shape_size(man.get("code_shape"), "residual code shape")
    finite_scale(man.get("code_scale"), "residual code scale")
    code_raw = brotli.decompress(out[2])
    if len(code_raw) != code_expected:
        raise ValueError("decoded residual code size differs")
    code_flat = np.frombuffer(code_raw, dtype=np.int8)
    params["code"] = (code_flat.astype(np.float32) * float(man["code_scale"])).reshape(code_shape)
    return man, params


def _validate_receiver_contract(store, poses, residual_man=None, residual_params=None):
    n_pairs = int(store["n_pairs"])
    if poses is not None and poses.shape != (n_pairs, 6):
        raise ValueError("store/PNTG pair domains differ")
    if residual_man is None and residual_params is None:
        return
    if residual_man is None or residual_params is None:
        raise ValueError("incomplete residual receiver contract")
    required = {
        "activation", "bank_base", "bank_f0", "bank_n_iso", "bank_n_orient0",
        "bank_n_scales", "chroma", "dilate", "hidden_dim", "hosc_beta",
        "hosc_omega", "learn_classes", "mod_dim", "n_classes", "n_hidden",
        "render_h", "render_w", "softmax_temp", "wire_s0", "wire_w0",
    }
    missing = sorted(required - set(residual_man))
    if missing:
        raise ValueError("residual forward contract missing keys: %s" % missing)
    def positive_int(name, allow_zero=False):
        value = residual_man[name]; lower = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError("invalid residual %s" % name)
        return int(value)
    def positive_number(name):
        value = residual_man[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid residual %s" % name)
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError("invalid residual %s" % name)
        return parsed
    nh = positive_int("n_hidden"); hd = positive_int("hidden_dim")
    md = positive_int("mod_dim"); nc = positive_int("n_classes")
    rh = positive_int("render_h"); rw = positive_int("render_w")
    positive_int("bank_n_scales"); positive_int("bank_n_orient0")
    positive_int("bank_n_iso", allow_zero=True); positive_int("dilate", allow_zero=True)
    for name in ("bank_base", "bank_f0", "hosc_beta", "hosc_omega",
                 "softmax_temp", "wire_s0", "wire_w0"):
        positive_number(name)
    max_bank_freq = residual_man.get("max_bank_freq")
    if max_bank_freq is not None:
        if (isinstance(max_bank_freq, bool)
                or not isinstance(max_bank_freq, (int, float))
                or not np.isfinite(float(max_bank_freq))
                or float(max_bank_freq) <= 0.0):
            raise ValueError("invalid residual max_bank_freq")
    if nc != int(store["n_classes"]) or (rh, rw) != (int(store["H"]), int(store["W"])):
        raise ValueError("residual/store class or render domains differ")
    if residual_man["activation"] not in ("hosc", "relu", "wire"):
        raise ValueError("unsupported residual activation")
    if not isinstance(residual_man["chroma"], bool):
        raise ValueError("residual chroma must be boolean")
    if residual_man.get("mask_mode", "boundary_annulus") not in (
        "boundary_annulus", "learn_classes", "union"):
        raise ValueError("unsupported residual mask mode")
    learn = residual_man["learn_classes"]
    if (not isinstance(learn, list) or not learn or len(learn) != len(set(learn))
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 or v >= nc for v in learn)):
        raise ValueError("residual learn classes outside store domain")
    code = np.asarray(residual_params.get("code"))
    if code.shape != (2 * n_pairs, md):
        raise ValueError("residual code shape outside receiver frame/modulation domain")
    in_features = 2 * int(_curvelet_B(residual_man).shape[1])
    expected = {
        "in_proj.weight": (hd, in_features), "in_proj.bias": (hd,),
        "film.weight": (2 * hd * nh, md), "film.bias": (2 * hd * nh,),
        "out_sdf.weight": (nc, hd), "out_sdf.bias": (nc,),
        "out_tex.weight": (3, hd), "out_tex.bias": (3,), "palette": (nc, 3),
    }
    for li in range(nh):
        expected["hidden.%d.weight" % li] = (hd, hd)
        expected["hidden.%d.bias" % li] = (hd,)
    has_pl = any(name.startswith("film_pl.") for name in residual_params)
    has_cc = any(name.startswith("concat_pl.") for name in residual_params)
    for li in range(nh):
        if has_pl:
            expected["film_pl.%d.weight" % li] = (2 * hd, md)
            expected["film_pl.%d.bias" % li] = (2 * hd,)
        if has_cc:
            expected["concat_pl.%d.weight" % li] = (hd, md)
            expected["concat_pl.%d.bias" % li] = (hd,)
    expected_names = set(expected) | {"code"}
    actual_names = set(residual_params)
    if actual_names != expected_names:
        raise ValueError(
            "residual parameter set differs from receiver-consumed set: missing=%s extra=%s"
            % (sorted(expected_names - actual_names), sorted(actual_names - expected_names))
        )
    for name, shape in expected.items():
        if name not in residual_params or np.asarray(residual_params[name]).shape != shape:
            raise ValueError("residual parameter shape differs: %s" % name)


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
        if (isinstance(mf, bool) or not isinstance(mf, (int, float))
                or not np.isfinite(float(mf)) or float(mf) <= 0.0):
            raise ValueError("invalid residual max_bank_freq")
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
    _validate_receiver_contract(s, poses)
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
        _validate_receiver_contract(s, poses, r_man, r_params)
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
