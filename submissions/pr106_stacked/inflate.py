#!/usr/bin/env python
# ruff: noqa: E402,I001
"""Inflate pr106_stacked: PR106 HNeRV decoder + composable subset of all
4 score-aware sidechannels (latent + yshift + lrl1 + wavelet-WR01) layered
into a single archive.

This is the meta-composition lane. Sidechannels can appear in any subset
(e.g., latent only, latent+yshift, all 3). Inflate applies them in
canonical order:

    1. latent  (section 0x01, applies to latents[p, d] BEFORE decoder)
    2. yshift  (section 0x02, applies to frames[k]    AFTER decoder)
    3. lrl1    (section 0x03, applies to frames[k]    AFTER yshift)
    4. wavelet (section 0x04, explicit no-op runtime-consumption proof)

End-of-sections sentinel = 0x00.

Wire format (single 0.bin in archive.zip):

    Offset  Bytes  Field                                    Notes
    ────────────────────────────────────────────────────────────────────────
    0       1      magic 0xFD                               stacked dispatch byte
    1       3      pr106_len (uint24 LE)                    bytes of inner PR106 archive
    4       N      pr106 packed archive (raw bytes)         starts with 0xFF magic
    4+N     1      section_id_byte (0x01..0x04 OR 0x00)     0x00 = end-of-sections
    5+N     2      section_len (uint16 LE)                  brotli'd section payload size
    7+N     M      brotli(section payload)                  decompresses to header + raw
    ...                                                     (more sections OR 0x00)

Section IDs:
    0x00 = end-of-sections
    0x01 = latent payload   (PR100-style hnerv_lc_v2 wire: u16 n_pairs +
                             per-pair (u8 dim_idx, i8 delta_q))
    0x02 = yshift payload   (SC01 mode-7: "SC01" + u8 mode_id=7 + u8 channels=3 +
                             u32 n_frames + f32 step + i8[n_frames, 3])
    0x03 = lrl1 payload     (LR01 mode-8: "LR01" + u8 mode_id=8 + u8 K +
                             u16 low_h + u16 low_w + u32 n_frames + f32 coeff_step +
                             f32 basis_step + i8[K*low_h*low_w] + i8[n_frames*K])
    0x04 = wavelet payload  (WR01 v1: charged atom coordinates; explicit no-op
                             until a transform/apply mode is reviewed)

Each section appears at most ONCE; sections may appear in any order on the
wire but inflate APPLIES them in canonical order regardless of wire order.

CUDA-required at inflate time (CLAUDE.md MPS-auth-eval-is-NOISE non-negotiable);
CPU acceptable only with --device cpu --smoke for [advisory only] roundtrip.

Reuses PR106 codec from sister submission via sys.path manipulation:
    - submissions/pr106_latent_sidecar/src/{codec,model}.py (vendored PR106)

Sister of all 3 single-sidechannel lanes; meta-composition variant per
docs/INDEX_score_aware_sidechannel_thread_20260504.md.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# Reuse the vendored PR106 codec/model from the latent_sidecar sister lane.
# Per Catalog #295 audit Priority 4 option (b): pr106_stacked is the meta-
# composition lane that stacks ALL 4 score-aware sidechannels (latent +
# yshift + lrl1 + wavelet-WR01) onto the canonical PR106 codec from sibling
# submissions/pr106_latent_sidecar/src/{codec,model}.py. The contest dispatch
# packet manifest declares both submissions as a co-shipped pair (sibling-
# submission shipping requirement). NOT acceptable for OSS-release path
# where each submission must be self-contained; LAB dispatch flow mounts
# both alongside.
PR106_SRC = HERE.parent / "pr106_latent_sidecar" / "src"
sys.path.insert(0, str(PR106_SRC))  # SUBMISSION_PYTHONPATH_SHIM_OK:pr106-stacked-meta-composition-lane-stacks-on-canonical-pr106-codec-from-sibling-pr106-latent-sidecar-submission-co-shipped-in-lab-dispatch-packet-manifest-not-oss-release-self-contained-per-catalog-295-audit-priority-4-option-b

from model import HNeRVDecoder  # type: ignore[import-not-found]
from codec import parse_packed_archive  # type: ignore[import-not-found]


CAMERA_H, CAMERA_W = 874, 1164

STACKED_MAGIC_BYTE = 0xFD  # outer dispatch byte for pr106_stacked

# Section IDs (canonical-order application: latent → yshift → lrl1)
SECTION_END = 0x00
SECTION_LATENT = 0x01
SECTION_YSHIFT = 0x02
SECTION_LRL1 = 0x03
SECTION_WAVELET = 0x04

# ── Latent sidecar constants (mirror submissions/pr106_latent_sidecar) ─────
LATENT_DELTA_SCALE = 0.01
LATENT_NO_OP_DIM = 255

# ── YSHIFT (SC01) constants (mirror submissions/pr106_yshift_sidechannel) ──
SC01_MAGIC = b"SC01"
SC01_HEADER = struct.Struct("<4sBBIf")  # magic + mode_id + n_channels + n_frames + step
SIDECHANNEL_MODE_Y_SHIFT = 7

# ── LRL1 (LR01) constants (mirror submissions/pr106_lrl1_sidechannel) ──────
LR01_MAGIC = b"LR01"
LR01_HEADER = struct.Struct("<4sBBHHIff")  # magic + mode_id + K + low_h + low_w +
#                                            n_frames + coeff_step + basis_step
SIDECHANNEL_MODE_LRL1 = 8

# ── Wavelet residual sidechannel (WR01) constants ─────────────────────────
WR01_MAGIC = b"WR01"
WR01_SCHEMA_VERSION = 1


# ===================================================================
# Outer parser: PR106 + section-list
# ===================================================================


def parse_stacked_archive(
    archive_bytes: bytes,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict, dict[int, dict]]:
    """Parse pr106_stacked 0.bin layout.

    Returns (state_dict, latents, meta, sections_by_id)
    where sections_by_id is {section_id: parsed_payload_dict, ...}.
    Empty dict means "no sidechannels present" (pure PR106 passthrough).
    """
    if not archive_bytes:
        raise ValueError("empty archive")
    magic = archive_bytes[0]
    if magic != STACKED_MAGIC_BYTE:
        raise ValueError(
            f"pr106_stacked magic mismatch: got 0x{magic:02X}, "
            f"expected 0x{STACKED_MAGIC_BYTE:02X}"
        )
    pr106_len = int.from_bytes(archive_bytes[1:4], "little")
    pr106_end = 4 + pr106_len
    if pr106_end > len(archive_bytes):
        raise ValueError(
            f"pr106_len {pr106_len} exceeds archive size {len(archive_bytes)}"
        )
    pr106_bytes = archive_bytes[4:pr106_end]
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)

    sections: dict[int, dict] = {}
    pos = pr106_end
    sentinel_seen = False
    while pos < len(archive_bytes):
        section_id = archive_bytes[pos]
        pos += 1
        if section_id == SECTION_END:
            # Sentinel: must be the LAST byte (no trailing garbage).
            if pos != len(archive_bytes):
                raise ValueError(
                    f"trailing bytes after end-of-sections sentinel: "
                    f"pos={pos}, total={len(archive_bytes)}"
                )
            sentinel_seen = True
            break
        if pos + 2 > len(archive_bytes):
            raise ValueError(
                f"truncated archive before section {section_id} length field"
            )
        sc_len = struct.unpack_from("<H", archive_bytes, pos)[0]
        pos += 2
        sc_payload_end = pos + sc_len
        if sc_payload_end > len(archive_bytes):
            raise ValueError(
                f"section {section_id} declared length {sc_len} exceeds remaining "
                f"archive size {len(archive_bytes) - pos}"
            )
        blob = archive_bytes[pos:sc_payload_end]
        pos = sc_payload_end
        if section_id in sections:
            raise ValueError(
                f"duplicate section id 0x{section_id:02X} on wire"
            )
        if section_id == SECTION_LATENT:
            sections[section_id] = decode_latent_blob(blob)
        elif section_id == SECTION_YSHIFT:
            sections[section_id] = decode_yshift_blob(blob)
        elif section_id == SECTION_LRL1:
            sections[section_id] = decode_lrl1_blob(blob)
        elif section_id == SECTION_WAVELET:
            sections[section_id] = decode_wavelet_blob(blob)
        else:
            raise ValueError(
                f"unknown section id 0x{section_id:02X} (expected one of "
                f"{{0x01, 0x02, 0x03, 0x04}} or 0x00 sentinel)"
            )
    if not sentinel_seen:
        # Loop exited via pos >= len(archive_bytes) WITHOUT hitting SECTION_END.
        raise ValueError(
            "missing end-of-sections sentinel (0x00) at archive tail"
        )
    return decoder_sd, latents, meta, sections


# ===================================================================
# Section payload decoders (one per section id)
# ===================================================================


def decode_latent_blob(blob: bytes) -> dict:
    """Brotli-decompress + parse PR100-style hnerv_lc_v2 latent payload.

    Returns dict {dim: uint8 array (n_pairs,), delta_q: int8 array (n_pairs,),
                  n_pairs: int}.
    """
    raw = brotli.decompress(blob)
    if len(raw) < 2:
        raise ValueError(f"latent blob too small: {len(raw)} bytes (need >= 2)")
    n_pairs = struct.unpack_from("<H", raw, 0)[0]
    expected = 2 + 2 * n_pairs
    if len(raw) != expected:
        raise ValueError(
            f"bad latent blob length: expected {expected} (header=2 + "
            f"{n_pairs}*2 pair bytes), got {len(raw)}"
        )
    arr = np.frombuffer(raw[2 : 2 + 2 * n_pairs], dtype=np.uint8).reshape(n_pairs, 2)
    dim = arr[:, 0].copy()  # uint8 with 255 sentinel
    delta_q = arr[:, 1].view(np.int8).copy()
    return {
        "n_pairs": int(n_pairs),
        "dim": dim,
        "delta_q": delta_q,
    }


def decode_yshift_blob(blob: bytes) -> dict:
    """Brotli-decompress + parse SC01-YSHIFT payload (mode 7)."""
    raw = brotli.decompress(blob)
    if len(raw) < SC01_HEADER.size:
        raise ValueError(f"yshift blob too small: {len(raw)} < {SC01_HEADER.size}")
    magic, mode_id, channels, frame_count, step = SC01_HEADER.unpack_from(raw)
    if magic != SC01_MAGIC:
        raise ValueError(f"bad SC01 magic: {magic!r}")
    if mode_id != SIDECHANNEL_MODE_Y_SHIFT:
        raise ValueError(
            f"unsupported yshift mode_id={mode_id}, expected {SIDECHANNEL_MODE_Y_SHIFT}"
        )
    if channels != 3:
        raise ValueError(f"YSHIFT expects 3 channels, got {channels}")
    expected = SC01_HEADER.size + frame_count * channels
    if len(raw) != expected:
        raise ValueError(f"bad SC01 length: expected {expected}, got {len(raw)}")
    raw_int = np.frombuffer(raw[SC01_HEADER.size:], dtype=np.int8).copy()
    raw_arr = raw_int.reshape(int(frame_count), channels)
    return {
        "mode_id": int(mode_id),
        "channels": int(channels),
        "step": float(step),
        "raw": raw_arr,                # (n_frames, 3) int8 — [y_off, dy, dx]
        "n_frames": int(frame_count),
    }


def decode_lrl1_blob(blob: bytes) -> dict:
    """Brotli-decompress + parse LR01-LRL1 payload (mode 8)."""
    raw = brotli.decompress(blob)
    if len(raw) < LR01_HEADER.size:
        raise ValueError(f"lrl1 blob too small: {len(raw)} < {LR01_HEADER.size}")
    magic, mode_id, K, low_h, low_w, n_frames, coeff_step, basis_step = (
        LR01_HEADER.unpack_from(raw)
    )
    if magic != LR01_MAGIC:
        raise ValueError(f"bad LR01 magic: {magic!r}")
    if mode_id != SIDECHANNEL_MODE_LRL1:
        raise ValueError(
            f"unsupported lrl1 mode_id={mode_id}, expected {SIDECHANNEL_MODE_LRL1}"
        )
    if K < 1:
        raise ValueError(f"LRL1 expects K >= 1, got {K}")
    if low_h < 1 or low_w < 1:
        raise ValueError(f"LRL1 expects low_h,low_w >= 1, got {low_h}x{low_w}")
    basis_size = int(K) * int(low_h) * int(low_w)
    coeff_size = int(n_frames) * int(K)
    expected = LR01_HEADER.size + basis_size + coeff_size
    if len(raw) != expected:
        raise ValueError(
            f"bad LR01 length: expected {expected} (header={LR01_HEADER.size} + "
            f"basis={basis_size} + coeffs={coeff_size}), got {len(raw)}"
        )
    basis_bytes = raw[LR01_HEADER.size:LR01_HEADER.size + basis_size]
    coeff_bytes = raw[LR01_HEADER.size + basis_size:]
    basis_int = np.frombuffer(basis_bytes, dtype=np.int8).copy().reshape(
        int(K), int(low_h), int(low_w),
    )
    coeff_int = np.frombuffer(coeff_bytes, dtype=np.int8).copy().reshape(
        int(n_frames), int(K),
    )
    return {
        "mode_id": int(mode_id),
        "K": int(K),
        "low_h": int(low_h),
        "low_w": int(low_w),
        "n_frames": int(n_frames),
        "coeff_step": float(coeff_step),
        "basis_step": float(basis_step),
        "basis": basis_int,    # (K, low_h, low_w) int8
        "coeffs": coeff_int,   # (n_frames, K) int8
    }


def decode_wavelet_blob(blob: bytes) -> dict:
    """Brotli-decompress + parse WR01 wavelet atom sidechannel.

    This parser intentionally consumes atom coordinates without applying them
    to pixels. That explicit no-op mode keeps the archive byte-closed and
    stack-testable while preventing a false score claim before a reviewed
    transform exists.
    """
    raw = brotli.decompress(blob)
    reader = _Reader(raw)
    magic = reader.read_exact(4)
    if magic != WR01_MAGIC:
        raise ValueError(f"bad WR01 magic: {magic!r}")
    version = reader.read_u16()
    if version != WR01_SCHEMA_VERSION:
        raise ValueError(f"unsupported WR01 schema version: {version}")
    section_count = reader.read_u16()
    sections: list[dict] = []
    total_atoms = 0
    for _section_idx in range(section_count):
        name_len = reader.read_u8()
        section_name = reader.read_exact(name_len).decode("ascii")
        source_section_sha256 = reader.read_exact(32).hex()
        raw_bytes = reader.read_u32()
        atom_count = reader.read_u16()
        atoms: list[dict] = []
        seen: set[tuple[int, int, int, int]] = set()
        for _atom_idx in range(atom_count):
            raw_offset = reader.read_u32()
            raw_end = reader.read_u32()
            level = reader.read_u8()
            coefficient_index = reader.read_u32()
            coefficient_quantized = reader.read_i32()
            key = (raw_offset, raw_end, level, coefficient_index)
            if key in seen:
                raise ValueError(f"duplicate WR01 atom in {section_name}: {key}")
            seen.add(key)
            if raw_end < raw_offset or raw_end > raw_bytes:
                raise ValueError(
                    f"bad WR01 atom support for {section_name}: "
                    f"{raw_offset}:{raw_end}/{raw_bytes}"
                )
            atoms.append(
                {
                    "raw_offset": int(raw_offset),
                    "raw_end": int(raw_end),
                    "level": int(level),
                    "coefficient_index": int(coefficient_index),
                    "coefficient_quantized": int(coefficient_quantized),
                }
            )
        total_atoms += atom_count
        sections.append(
            {
                "section_name": section_name,
                "source_section_sha256": source_section_sha256,
                "raw_bytes": int(raw_bytes),
                "atom_count": int(atom_count),
                "atoms": atoms,
            }
        )
    reader.assert_eof()
    return {
        "magic": WR01_MAGIC.decode("ascii"),
        "schema_version": int(version),
        "section_count": int(section_count),
        "total_atom_count": int(total_atoms),
        "sections": sections,
        "runtime_mode": "explicit_noop_consume_only",
        "runtime_consumption_proof": wavelet_runtime_consumption_proof(sections),
    }


def wavelet_runtime_consumption_proof(sections: list[dict]) -> dict:
    """Return a deterministic digest proving WR01 atom coordinates were read."""
    h = hashlib.sha256()
    total_atoms = 0
    for section in sections:
        h.update(str(section["section_name"]).encode())
        h.update(str(section["source_section_sha256"]).encode())
        h.update(str(section["raw_bytes"]).encode())
        for atom in section["atoms"]:
            total_atoms += 1
            h.update(
                (
                    f"{atom['raw_offset']}:{atom['raw_end']}:"
                    f"{atom['level']}:{atom['coefficient_index']}:"
                    f"{atom['coefficient_quantized']};"
                ).encode()
            )
    return {
        "runtime_consumed": total_atoms > 0,
        "decoded_atom_count": int(total_atoms),
        "atom_coordinate_sha256": h.hexdigest(),
        "score_claim": False,
    }


# ===================================================================
# Sidechannel application functions
# ===================================================================


def apply_latent_corrections(
    latents: torch.Tensor,
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    scale: float = LATENT_DELTA_SCALE,
) -> torch.Tensor:
    """In-place add per-pair correction to (n, latent_dim) latents tensor.

    Mirrors submissions/pr106_latent_sidecar.inflate.apply_sidecar_corrections.
    """
    n = latents.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == LATENT_NO_OP_DIM:
            continue
        latents[p, d] = latents[p, d] + float(delta_q_arr[p]) * scale
    return latents


def shift_rgb_uint8(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer pixel translation; mirrors yshift sister inflate.shift_rgb_uint8."""
    if dy == 0 and dx == 0:
        return frame
    h, w, _ = frame.shape
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    out = frame.copy()
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return out


def apply_yshift(frame_u8: np.ndarray, sc_row: np.ndarray, step: float) -> np.ndarray:
    """Apply per-frame [y_off, dy, dx] correction. Mirrors yshift sister inflate."""
    y_off = float(sc_row[0]) * step
    dy = int(sc_row[1])
    dx = int(sc_row[2])
    if y_off != 0.0:
        out = frame_u8.astype(np.float32) + y_off
        out = np.clip(out, 0, 255).round().astype(np.uint8)
    else:
        out = frame_u8
    return shift_rgb_uint8(out, dy, dx)


def upsample_basis(
    basis_int8: np.ndarray, basis_step: float, target_h: int, target_w: int,
    *, device: torch.device | None = None,
) -> torch.Tensor:
    """Bilinear-upsample (K, low_h, low_w) int8 basis to (K, target_h, target_w) float."""
    K, low_h, low_w = basis_int8.shape
    basis_f = torch.from_numpy(basis_int8.astype(np.float32) * float(basis_step))
    if device is not None:
        basis_f = basis_f.to(device)
    basis_4d = basis_f.unsqueeze(0)
    up = F.interpolate(
        basis_4d, size=(target_h, target_w), mode="bilinear", align_corners=False,
    )
    return up.squeeze(0)  # (K, H, W)


def apply_lrl1_to_frame(
    frame_u8: np.ndarray,
    upsampled_basis: torch.Tensor,
    coeffs_int8: np.ndarray,
    coeff_step: float,
) -> np.ndarray:
    """Apply per-frame LRL1 correction to a (H, W, 3) uint8 frame.

    Mirrors submissions/pr106_lrl1_sidechannel.inflate.apply_lrl1_to_frame.
    """
    if frame_u8.ndim != 3 or frame_u8.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8 frame, got shape {frame_u8.shape}")
    if coeffs_int8.shape != (upsampled_basis.shape[0],):
        raise ValueError(
            f"coeffs shape {coeffs_int8.shape} doesn't match basis K="
            f"{upsampled_basis.shape[0]}"
        )
    coeffs_f = torch.from_numpy(coeffs_int8.astype(np.float32) * float(coeff_step))
    coeffs_f = coeffs_f.to(upsampled_basis.device)
    correction = torch.einsum("k,khw->hw", coeffs_f, upsampled_basis)  # (H, W)
    correction_np = correction.detach().cpu().numpy()
    out = frame_u8.astype(np.float32) + correction_np[..., None]
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


# ===================================================================
# Inflate driver
# ===================================================================


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta, sections = parse_stacked_archive(archive_bytes)

    if not torch.cuda.is_available():
        sys.exit(
            "pr106_stacked inflate requires GPU "
            "(per CLAUDE.md MPS-auth-eval-is-NOISE)."
        )
    device = torch.device("cuda")
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]
    expected_frames = n_pairs * 2

    # ── Apply LATENT correction BEFORE decoder forward (canonical-order step 1) ──
    latent_section = sections.get(SECTION_LATENT)
    if latent_section is not None:
        if latent_section["n_pairs"] != n_pairs:
            raise ValueError(
                f"latent section n_pairs={latent_section['n_pairs']} mismatches "
                f"PR106 n_pairs={n_pairs}"
            )
        n_corr = int((latent_section["dim"] != LATENT_NO_OP_DIM).sum())
        print(
            f"[inflate] latent applied: {n_corr}/{latent_section['n_pairs']} "
            f"pairs corrected",
            file=sys.stderr,
        )
        apply_latent_corrections(
            latents, latent_section["dim"], latent_section["delta_q"],
        )

    # ── Validate post-decoder section frame counts ────────────────────────
    yshift_section = sections.get(SECTION_YSHIFT)
    if yshift_section is not None and yshift_section["n_frames"] != expected_frames:
        raise ValueError(
            f"yshift section n_frames={yshift_section['n_frames']} mismatches "
            f"decoder expected_frames={expected_frames}"
        )
    lrl1_section = sections.get(SECTION_LRL1)
    if lrl1_section is not None and lrl1_section["n_frames"] != expected_frames:
        raise ValueError(
            f"lrl1 section n_frames={lrl1_section['n_frames']} mismatches "
            f"decoder expected_frames={expected_frames}"
        )
    wavelet_section = sections.get(SECTION_WAVELET)
    if wavelet_section is not None:
        proof = wavelet_section["runtime_consumption_proof"]
        print(
            f"[inflate] wavelet WR01 consumed in explicit no-op mode: "
            f"atoms={proof['decoded_atom_count']}, "
            f"digest={proof['atom_coordinate_sha256']}",
            file=sys.stderr,
        )

    # ── One-time LRL1 basis upsample (cached across all frames) ───────────
    upsampled_basis: torch.Tensor | None = None
    if lrl1_section is not None:
        upsampled_basis = upsample_basis(
            lrl1_section["basis"], lrl1_section["basis_step"],
            CAMERA_H, CAMERA_W, device=device,
        )

    # Banner for forensics
    sec_status = ", ".join(
        name for name, sec in (
            ("latent", latent_section),
            ("yshift", yshift_section),
            ("lrl1", lrl1_section),
            ("wavelet_wr01_noop", wavelet_section),
        ) if sec is not None
    ) or "NONE"
    print(
        f"[inflate] pr106_stacked: decoder loaded; sections={{{sec_status}}}; "
        f"n_pairs={n_pairs}",
        file=sys.stderr,
    )

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False,
            )
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )  # (B*2, H, W, 3) uint8

            # ── Apply YSHIFT (canonical-order step 2, AFTER decoder) ─────
            if yshift_section is not None:
                sc_raw = yshift_section["raw"]
                sc_step = yshift_section["step"]
                for k in range(B * 2):
                    frame_idx = i * 2 + k
                    frames[k] = apply_yshift(frames[k], sc_raw[frame_idx], sc_step)

            # ── Apply LRL1 (canonical-order step 3, AFTER yshift) ────────
            if upsampled_basis is not None and lrl1_section is not None:
                sc_coeffs = lrl1_section["coeffs"]
                sc_coeff_step = lrl1_section["coeff_step"]
                for k in range(B * 2):
                    frame_idx = i * 2 + k
                    frames[k] = apply_lrl1_to_frame(
                        frames[k], upsampled_basis, sc_coeffs[frame_idx],
                        sc_coeff_step,
                    )

            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_exact(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._data):
            raise ValueError("WR01 sidechannel truncated")
        out = self._data[self._pos:end]
        self._pos = end
        return out

    def read_u8(self) -> int:
        return self.read_exact(1)[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self.read_exact(2))[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def read_i32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def assert_eof(self) -> None:
        if self._pos != len(self._data):
            raise ValueError(
                f"WR01 sidechannel trailing bytes: pos={self._pos}, total={len(self._data)}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.pr106_stacked.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
