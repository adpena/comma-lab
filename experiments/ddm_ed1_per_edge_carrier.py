# SPDX-License-Identifier: MIT
"""Build ddm_ed1's counted Road/Lane separatrix carrier on the fz4 sub_final base.

This is a scorer-free builder.  The common arm contract gives the n600 scorer
slot to sg4/sb1, so this script produces a receiver-consumed byte-closed
candidate archive plus a queued scorer spec, but it does not run SegNet/PoseNet.

The counted ED1 section has two pieces:

* a degree-4 lane-centerline stream in the existing openpilot-native
  ``lane_headstart`` delta format.  These centerlines define the decoder's
  Road/Lane separatrix chart and are counted video-derived bytes.
* a pair-bitpacked innovation stream over the radius-1 band around that chart.
  Each band position carries an edit bit and a target-class bit.  The receiver
  paints selected scorer-grid cells back through the same private bilinear
  supports used by the F0PR proof, mutating frame 1 before frame 0 is warped.

No scorer weights, GT argmax table, or margin table are shipped in inflate.py.
The archive bytes are real; the survival number remains queued.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import shutil
import struct
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final
from zipfile import ZipFile

import numpy as np

try:
    import brotli
except ImportError:  # pragma: no cover - brotli is present in the pact env
    brotli = None  # type: ignore[assignment]

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math import lane_headstart as lh
from tac.optimization.ddm_ix2_archive_container import (
    build_payload,
    build_single_member_zip,
    parse_payload,
)


ED1_MAGIC: Final = b"ED1RLC1!"
ED1_VERSION: Final = 1
ED1_MEMBER_NAME: Final = "road_lane_centerline_band.ed1"
ED1_HEADER: Final = struct.Struct("<8sBHHHBBBB6sII32s32s")
SEG_H: Final = 384
SEG_W: Final = 512
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
ROAD_CLASS: Final = 0
LANE_CLASS: Final = 1
RATE_DENOM: Final = 37_545_489

BASE_SUBMISSION_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final")
DEFAULT_ARGMAX_CACHE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
DEFAULT_OUT_DIR: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline"
)

BASELINE_S: Final = 0.7541459
BASELINE_BYTES: Final = 358_084
BASELINE_D_SEG: Final = 0.00431179
BASELINE_D_POSE: Final = 0.0007145917
SG3_CHARTER_FALSIFIER: Final = 0.3956
FP1_ROAD_RGB: Final = (30, 39, 72)
FP1_LANE_RGB: Final = (77, 87, 119)

CODEC_IDS: Final = {"raw": 0, "zlib9": 1, "lzma9": 2, "brotli11": 3}
ID_CODECS: Final = {value: key for key, value in CODEC_IDS.items()}


@dataclass(frozen=True, slots=True)
class CodedBlock:
    codec: str
    coded: bytes
    raw_bytes: int
    coded_bytes: int
    raw_sha256: str
    race: dict[str, int]


@dataclass(frozen=True, slots=True)
class Ed1ParsedSection:
    seg_h: int
    seg_w: int
    n_pairs: int
    band_radius: int
    road_rgb: tuple[int, int, int]
    lane_rgb: tuple[int, int, int]
    bands: tuple[np.ndarray, ...]
    edit_bits: np.ndarray
    lane_bits: np.ndarray
    bit_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Ed1SectionBuild:
    section: bytes
    parsed: Ed1ParsedSection
    centerline_raw: bytes
    edit_raw: bytes
    centerline_block: CodedBlock
    edit_block: CodedBlock
    total_road_lane_targets: int
    captured_targets: int
    band_pixels: int
    per_pair: tuple[dict[str, int], ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_bytes(raw: bytes) -> CodedBlock:
    race: dict[str, bytes] = {
        "raw": bytes(raw),
        "zlib9": zlib.compress(raw, 9),
        "lzma9": lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME),
    }
    if brotli is not None:
        race["brotli11"] = brotli.compress(raw, quality=11)
    sizes = {name: len(payload) for name, payload in race.items()}
    codec = min(sizes, key=sizes.get)
    return CodedBlock(
        codec=codec,
        coded=race[codec],
        raw_bytes=len(raw),
        coded_bytes=sizes[codec],
        raw_sha256=sha256_bytes(raw),
        race=sizes,
    )


def decode_bytes(codec: str, payload: bytes) -> bytes:
    if codec == "raw":
        return bytes(payload)
    if codec == "zlib9":
        return zlib.decompress(payload)
    if codec == "lzma9":
        return lzma.decompress(payload)
    if codec == "brotli11":
        if brotli is None:  # pragma: no cover
            raise Ed1CarrierError("brotli11 section cannot decode without brotli")
        return brotli.decompress(payload)
    raise Ed1CarrierError(f"unknown ED1 codec {codec!r}")


class Ed1CarrierError(ValueError):
    """The ED1 carrier, archive, or generated runtime failed closed."""


def pack_bool_bits(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((values != 0) & (values != 1)):
        raise Ed1CarrierError("bit stream contains values outside {0,1}")
    return np.packbits(values, bitorder="big").tobytes()


def unpack_bool_bits(payload: bytes, count: int) -> np.ndarray:
    if count < 0:
        raise Ed1CarrierError("bit count must be non-negative")
    need = (count + 7) // 8
    if len(payload) != need:
        raise Ed1CarrierError(f"bit payload expected {need} bytes, got {len(payload)}")
    raw = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    if np.any(raw[count:] != 0):
        raise Ed1CarrierError("bit payload has nonzero padding")
    return np.ascontiguousarray(raw[:count].astype(bool))


@dataclass(frozen=True, slots=True)
class _CenterlineWire:
    axis: int
    x_start: int
    half_width: int
    ys: np.ndarray


def deserialize_centerline_delta(raw: bytes, grid_h: int, grid_w: int) -> tuple[tuple[_CenterlineWire, ...], ...]:
    frames: list[tuple[_CenterlineWire, ...]] = []
    offset = 0
    while offset < len(raw):
        if len(raw) < offset + 2:
            raise Ed1CarrierError("centerline stream truncated before frame count")
        (count,) = struct.unpack_from("<H", raw, offset)
        offset += 2
        rows: list[_CenterlineWire] = []
        for _ in range(count):
            if len(raw) < offset + 8:
                raise Ed1CarrierError("centerline component header is truncated")
            axis = raw[offset]
            offset += 1
            (x_start,) = struct.unpack_from("<h", raw, offset)
            offset += 2
            (length,) = struct.unpack_from("<H", raw, offset)
            offset += 2
            half_width = raw[offset]
            offset += 1
            (col0,) = struct.unpack_from("<h", raw, offset)
            offset += 2
            if axis not in (0, 1):
                raise Ed1CarrierError(f"centerline axis {axis} is unsupported")
            if length < 1:
                raise Ed1CarrierError("centerline component length must be positive")
            if length >= 2:
                byte_count = (length - 1) * 2
                if len(raw) < offset + byte_count:
                    raise Ed1CarrierError("centerline deltas are truncated")
                deltas = np.frombuffer(raw, dtype="<i2", count=length - 1, offset=offset).astype(np.int64)
                offset += byte_count
                ys = np.concatenate(([col0], int(col0) + np.cumsum(deltas))).astype(np.int64)
            else:
                ys = np.array([col0], dtype=np.int64)
            y_min = int(ys.min())
            y_max = int(ys.max())
            if axis == 0 and not (-grid_w <= y_min and y_max <= grid_w * 2):
                raise Ed1CarrierError("centerline column coordinate is out of plausible bounds")
            if axis == 1 and not (-grid_h <= y_min and y_max <= grid_h * 2):
                raise Ed1CarrierError("centerline row coordinate is out of plausible bounds")
            rows.append(_CenterlineWire(axis=axis, x_start=int(x_start), half_width=int(half_width), ys=ys))
        frames.append(tuple(rows))
    if offset != len(raw):
        raise Ed1CarrierError("centerline stream did not close exactly")
    return tuple(frames)


def rasterize_wire_centerlines(lines: tuple[_CenterlineWire, ...], grid_h: int, grid_w: int) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for line in lines:
        xs = np.arange(line.x_start, line.x_start + line.ys.size)
        hw = line.half_width
        if line.axis == 0:
            for x, y in zip(xs, line.ys):
                if 0 <= x < grid_h:
                    c0 = max(0, int(y) - hw)
                    c1 = min(grid_w - 1, int(y) + hw)
                    if c0 <= c1:
                        mask[int(x), c0 : c1 + 1] = True
        else:
            for x, y in zip(xs, line.ys):
                if 0 <= x < grid_w:
                    r0 = max(0, int(y) - hw)
                    r1 = min(grid_h - 1, int(y) + hw)
                    if r0 <= r1:
                        mask[r0 : r1 + 1, int(x)] = True
    return mask


def dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise Ed1CarrierError("band radius must be non-negative")
    source = np.asarray(mask, dtype=bool)
    out = source.copy()
    h, w = source.shape
    for dr in range(-radius, radius + 1):
        r_src0 = max(0, -dr)
        r_src1 = min(h, h - dr)
        r_dst0 = max(0, dr)
        r_dst1 = min(h, h + dr)
        for dc in range(-radius, radius + 1):
            c_src0 = max(0, -dc)
            c_src1 = min(w, w - dc)
            c_dst0 = max(0, dc)
            c_dst1 = min(w, w + dc)
            out[r_dst0:r_dst1, c_dst0:c_dst1] |= source[r_src0:r_src1, c_src0:c_src1]
    return out


def _correction_target(gt: np.ndarray, current: np.ndarray) -> np.ndarray:
    return ((gt == ROAD_CLASS) & (current == LANE_CLASS)) | (
        (gt == LANE_CLASS) & (current == ROAD_CLASS)
    )


def build_ed1_section_from_argmax(
    gt_argmax: np.ndarray,
    current_argmax: np.ndarray,
    *,
    degree: int = 4,
    band_radius: int = 1,
    road_rgb: tuple[int, int, int] = FP1_ROAD_RGB,
    lane_rgb: tuple[int, int, int] = FP1_LANE_RGB,
) -> Ed1SectionBuild:
    gt = np.asarray(gt_argmax)
    current = np.asarray(current_argmax)
    if gt.shape != current.shape or gt.ndim != 3:
        raise Ed1CarrierError(f"argmax stacks must both be [P,H,W], got {gt.shape!r} and {current.shape!r}")
    n_pairs, grid_h, grid_w = (int(v) for v in gt.shape)
    if grid_h > 0xFFFF or grid_w > 0xFFFF or n_pairs > 0xFFFF:
        raise Ed1CarrierError("ED1 uint16 dimensions exceeded")

    centerlines_per_frame: list[list[lh.LaneCenterline]] = []
    edit_chunks: list[np.ndarray] = []
    lane_chunks: list[np.ndarray] = []
    total_targets = int(_correction_target(gt, current).sum())
    captured = 0
    band_pixels = 0
    per_pair: list[dict[str, int]] = []

    for pair_index in range(n_pairs):
        centerlines = lh.fit_centerlines(
            gt[pair_index] == LANE_CLASS,
            degree=degree,
            dash_bridge_rows=1,
            min_component_pixels=12,
            max_half_width=4,
        )
        centerlines_per_frame.append(centerlines)
        base_lane = lh.rasterize_centerlines(centerlines, grid_h, grid_w)
        band = dilate_square(base_lane, band_radius)
        target = _correction_target(gt[pair_index], current[pair_index])
        edit = target[band]
        lane = (gt[pair_index][band] == LANE_CLASS) & edit
        edit_chunks.append(np.ascontiguousarray(edit, dtype=bool))
        lane_chunks.append(np.ascontiguousarray(lane, dtype=bool))
        pair_captured = int(edit.sum())
        pair_band = int(band.sum())
        pair_targets = int(target.sum())
        captured += pair_captured
        band_pixels += pair_band
        per_pair.append(
            {
                "pair": pair_index,
                "road_lane_targets": pair_targets,
                "captured_targets": pair_captured,
                "band_pixels": pair_band,
            }
        )

    centerline_raw = lh.serialize_centerlines_delta(centerlines_per_frame)
    edit_all = np.concatenate(edit_chunks) if edit_chunks else np.zeros(0, dtype=bool)
    lane_all = np.concatenate(lane_chunks) if lane_chunks else np.zeros(0, dtype=bool)
    edit_raw = pack_bool_bits(edit_all) + pack_bool_bits(lane_all)
    center_block = code_bytes(centerline_raw)
    edit_block = code_bytes(edit_raw)
    paint_bytes = bytes((*road_rgb, *lane_rgb))
    header = ED1_HEADER.pack(
        ED1_MAGIC,
        ED1_VERSION,
        grid_h,
        grid_w,
        n_pairs,
        band_radius,
        CODEC_IDS[center_block.codec],
        CODEC_IDS[edit_block.codec],
        1,  # paint mode: target-class flat RGB in scorer-grid private supports
        paint_bytes,
        center_block.coded_bytes,
        edit_block.coded_bytes,
        bytes.fromhex(center_block.raw_sha256),
        bytes.fromhex(edit_block.raw_sha256),
    )
    section = header + center_block.coded + edit_block.coded
    parsed = parse_ed1_section(section)
    if captured != int(correction_maps_from_parsed(parsed).astype(bool).sum()):
        raise Ed1CarrierError("parsed correction count differs from builder count")
    return Ed1SectionBuild(
        section=section,
        parsed=parsed,
        centerline_raw=centerline_raw,
        edit_raw=edit_raw,
        centerline_block=center_block,
        edit_block=edit_block,
        total_road_lane_targets=total_targets,
        captured_targets=captured,
        band_pixels=band_pixels,
        per_pair=tuple(per_pair),
    )


def parse_ed1_section(section: bytes) -> Ed1ParsedSection:
    if len(section) < ED1_HEADER.size:
        raise Ed1CarrierError("ED1 section header is truncated")
    (
        magic,
        version,
        grid_h,
        grid_w,
        n_pairs,
        band_radius,
        center_codec_id,
        edit_codec_id,
        paint_mode,
        paint_bytes,
        center_len,
        edit_len,
        center_sha,
        edit_sha,
    ) = ED1_HEADER.unpack_from(section, 0)
    if magic != ED1_MAGIC or version != ED1_VERSION:
        raise Ed1CarrierError("ED1 section magic/version differs")
    if paint_mode != 1:
        raise Ed1CarrierError("ED1 paint mode is unsupported")
    center_codec = ID_CODECS.get(int(center_codec_id))
    edit_codec = ID_CODECS.get(int(edit_codec_id))
    if center_codec is None or edit_codec is None:
        raise Ed1CarrierError("ED1 codec id is unsupported")
    offset = ED1_HEADER.size
    center_coded = section[offset : offset + center_len]
    offset += center_len
    edit_coded = section[offset : offset + edit_len]
    offset += edit_len
    if offset != len(section):
        raise Ed1CarrierError("ED1 section has trailing bytes")
    if len(center_coded) != center_len or len(edit_coded) != edit_len:
        raise Ed1CarrierError("ED1 section coded blocks are truncated")
    center_raw = decode_bytes(center_codec, center_coded)
    edit_raw = decode_bytes(edit_codec, edit_coded)
    if hashlib.sha256(center_raw).digest() != center_sha:
        raise Ed1CarrierError("ED1 centerline raw SHA differs")
    if hashlib.sha256(edit_raw).digest() != edit_sha:
        raise Ed1CarrierError("ED1 edit raw SHA differs")

    frames = deserialize_centerline_delta(center_raw, grid_h, grid_w)
    if len(frames) != n_pairs:
        raise Ed1CarrierError(f"ED1 centerline frames {len(frames)} != n_pairs {n_pairs}")
    bands: list[np.ndarray] = []
    offsets = [0]
    for frame in frames:
        band = dilate_square(rasterize_wire_centerlines(frame, grid_h, grid_w), band_radius)
        flat = np.flatnonzero(band.reshape(-1)).astype(np.int32)
        bands.append(flat)
        offsets.append(offsets[-1] + int(flat.size))
    total_bits = offsets[-1]
    bit_bytes = (total_bits + 7) // 8
    if len(edit_raw) != bit_bytes * 2:
        raise Ed1CarrierError(f"ED1 edit stream expected {bit_bytes * 2} bytes, got {len(edit_raw)}")
    edit_bits = unpack_bool_bits(edit_raw[:bit_bytes], total_bits)
    lane_bits = unpack_bool_bits(edit_raw[bit_bytes:], total_bits)
    road_rgb = tuple(int(v) for v in paint_bytes[:3])
    lane_rgb = tuple(int(v) for v in paint_bytes[3:6])
    return Ed1ParsedSection(
        seg_h=grid_h,
        seg_w=grid_w,
        n_pairs=n_pairs,
        band_radius=band_radius,
        road_rgb=road_rgb,  # type: ignore[arg-type]
        lane_rgb=lane_rgb,  # type: ignore[arg-type]
        bands=tuple(bands),
        edit_bits=edit_bits,
        lane_bits=lane_bits,
        bit_offsets=tuple(offsets),
    )


def correction_maps_from_parsed(parsed: Ed1ParsedSection) -> np.ndarray:
    """Return ``0=no edit, 1=paint Road, 2=paint Lane`` on the scorer grid."""
    maps = np.zeros((parsed.n_pairs, parsed.seg_h, parsed.seg_w), dtype=np.uint8)
    for pair_index, band in enumerate(parsed.bands):
        start = parsed.bit_offsets[pair_index]
        stop = parsed.bit_offsets[pair_index + 1]
        edit = parsed.edit_bits[start:stop]
        lane = parsed.lane_bits[start:stop]
        selected = band[edit]
        target = np.where(lane[edit], 2, 1).astype(np.uint8)
        maps[pair_index].reshape(-1)[selected] = target
    return maps


def _patch_inflate_runner(base_source: str) -> str:
    if "_ED1_MAGIC" in base_source:
        raise Ed1CarrierError("base inflate_runner already appears to contain ED1 support")
    insert_at = base_source.index("\n\nclass Decoder:")
    source = base_source[:insert_at] + ED1_RUNTIME_CODE + base_source[insert_at:]
    source = source.replace(
        "        self._f0_repair = None          # (coefs, atoms, seg_h, seg_w) when F0PR1 ships\n",
        "        self._f0_repair = None          # (coefs, atoms, seg_h, seg_w) when F0PR1 ships\n"
        "        self._ed1 = None                # Road/Lane separatrix carrier when ED1 ships\n",
    )
    old_read = """        if len(sections) == len(IX2_JOINT_ORDER) + 1:
            # v5 joint group: the 5th section is the F0PR1 frame_0 pose-repair stream.
            config, renderer, selector, pose_warp, f0pr = sections
            k, seg_h, seg_w, coefs = _f0pr_parse(f0pr)
            self._f0_repair = (coefs, _f0pr_dct_atoms(k, seg_h, seg_w), seg_h, seg_w)
        elif len(sections) == len(IX2_JOINT_ORDER):
            config, renderer, selector, pose_warp = sections
        else:
            raise SystemExit(
                f"ix2 container holds {len(sections)} sections, "
                f"expected {len(IX2_JOINT_ORDER)} or {len(IX2_JOINT_ORDER) + 1}")
"""
    new_read = """        if len(sections) == len(IX2_JOINT_ORDER) + 2:
            # ED1 appends a Road/Lane frame_1 carrier after the existing F0PR1 stream.
            config, renderer, selector, pose_warp, f0pr, ed1 = sections
            k, seg_h, seg_w, coefs = _f0pr_parse(f0pr)
            self._f0_repair = (coefs, _f0pr_dct_atoms(k, seg_h, seg_w), seg_h, seg_w)
            self._ed1 = _ed1_parse(ed1)
        elif len(sections) == len(IX2_JOINT_ORDER) + 1:
            # v5 joint group: the 5th section is the F0PR1 frame_0 pose-repair stream.
            config, renderer, selector, pose_warp, f0pr = sections
            k, seg_h, seg_w, coefs = _f0pr_parse(f0pr)
            self._f0_repair = (coefs, _f0pr_dct_atoms(k, seg_h, seg_w), seg_h, seg_w)
        elif len(sections) == len(IX2_JOINT_ORDER):
            config, renderer, selector, pose_warp = sections
        else:
            raise SystemExit(
                f"ix2 container holds {len(sections)} sections, "
                f"expected {len(IX2_JOINT_ORDER)}, {len(IX2_JOINT_ORDER) + 1}, "
                f"or {len(IX2_JOINT_ORDER) + 2}")
"""
    if old_read not in source:
        raise Ed1CarrierError("inflate_runner _read_ix2 shape did not match the pinned base")
    source = source.replace(old_read, new_read)
    old_f1 = """    def f1(self, i: int) -> np.ndarray:
        return render_frame1_camera_uint8(self.packet, i)
"""
    new_f1 = """    def f1(self, i: int) -> np.ndarray:
        frame = render_frame1_camera_uint8(self.packet, i)
        if self._ed1 is not None:
            frame = _ed1_apply(frame, self._ed1, i)
        return frame
"""
    if old_f1 not in source:
        raise Ed1CarrierError("inflate_runner f1 method did not match the pinned base")
    return source.replace(old_f1, new_f1)


ED1_RUNTIME_CODE = r'''

# ---- ed1 Road/Lane separatrix carrier (OPTIONAL 6th joint section) --------------
# Counted section: degree-4 lane-centerline chart + bitpacked Road/Lane innovation.
# Decode has no scorer, GT argmax table, or hidden payload.  It reconstructs the
# counted chart, consumes every innovation bit, and writes target-class RGB into
# frame_1's private scorer-grid supports before frame_0 is warped from frame_1.
import hashlib as _ed1_hashlib
import zlib as _ed1_zlib

_ED1_MAGIC = b"ED1RLC1!"
_ED1_HEADER = struct.Struct("<8sBHHHBBBB6sII32s32s")
_ED1_CODEC_RAW = 0
_ED1_CODEC_ZLIB9 = 1
_ED1_CODEC_LZMA9 = 2
_ED1_CODEC_BROTLI11 = 3


def _ed1_decode(codec: int, payload: bytes) -> bytes:
    if codec == _ED1_CODEC_RAW:
        return bytes(payload)
    if codec == _ED1_CODEC_ZLIB9:
        return _ed1_zlib.decompress(payload)
    if codec == _ED1_CODEC_LZMA9:
        return lzma.decompress(payload)
    if codec == _ED1_CODEC_BROTLI11:
        return brotli.decompress(payload)
    raise SystemExit(f"unknown ED1 codec id {codec}")


def _ed1_unpack_bits(payload: bytes, count: int) -> np.ndarray:
    need = (count + 7) // 8
    if len(payload) != need:
        raise SystemExit("ED1 bit payload length mismatch")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    if np.any(bits[count:] != 0):
        raise SystemExit("ED1 bit payload has nonzero padding")
    return np.ascontiguousarray(bits[:count].astype(bool))


def _ed1_deserialize_centerlines(raw: bytes, grid_h: int, grid_w: int):
    frames = []
    off = 0
    while off < len(raw):
        if len(raw) < off + 2:
            raise SystemExit("ED1 centerline stream truncated before frame count")
        (ncomp,) = struct.unpack_from("<H", raw, off)
        off += 2
        rows = []
        for _ in range(ncomp):
            if len(raw) < off + 8:
                raise SystemExit("ED1 centerline component header truncated")
            axis = raw[off]
            off += 1
            (x_start,) = struct.unpack_from("<h", raw, off)
            off += 2
            (length,) = struct.unpack_from("<H", raw, off)
            off += 2
            hw = raw[off]
            off += 1
            (col0,) = struct.unpack_from("<h", raw, off)
            off += 2
            if axis not in (0, 1) or length < 1:
                raise SystemExit("ED1 centerline component is invalid")
            if length >= 2:
                nbytes = (length - 1) * 2
                if len(raw) < off + nbytes:
                    raise SystemExit("ED1 centerline deltas truncated")
                deltas = np.frombuffer(raw, dtype="<i2", count=length - 1, offset=off).astype(np.int64)
                off += nbytes
                ys = np.concatenate(([col0], int(col0) + np.cumsum(deltas))).astype(np.int64)
            else:
                ys = np.array([col0], dtype=np.int64)
            rows.append((int(axis), int(x_start), int(hw), ys))
        frames.append(tuple(rows))
    if off != len(raw):
        raise SystemExit("ED1 centerline stream did not close")
    return tuple(frames)


def _ed1_rasterize(lines, grid_h: int, grid_w: int) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for axis, x_start, hw, ys in lines:
        xs = np.arange(x_start, x_start + ys.size)
        if axis == 0:
            for x, y in zip(xs, ys):
                if 0 <= x < grid_h:
                    c0 = max(0, int(y) - hw)
                    c1 = min(grid_w - 1, int(y) + hw)
                    if c0 <= c1:
                        mask[int(x), c0:c1 + 1] = True
        else:
            for x, y in zip(xs, ys):
                if 0 <= x < grid_w:
                    r0 = max(0, int(y) - hw)
                    r1 = min(grid_h - 1, int(y) + hw)
                    if r0 <= r1:
                        mask[r0:r1 + 1, int(x)] = True
    return mask


def _ed1_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    out = source.copy()
    h, w = source.shape
    for dr in range(-radius, radius + 1):
        rs0 = max(0, -dr)
        rs1 = min(h, h - dr)
        rd0 = max(0, dr)
        rd1 = min(h, h + dr)
        for dc in range(-radius, radius + 1):
            cs0 = max(0, -dc)
            cs1 = min(w, w - dc)
            cd0 = max(0, dc)
            cd1 = min(w, w + dc)
            out[rd0:rd1, cd0:cd1] |= source[rs0:rs1, cs0:cs1]
    return out


def _ed1_parse(blob: bytes):
    if len(blob) < _ED1_HEADER.size:
        raise SystemExit("ED1 section header truncated")
    (magic, version, seg_h, seg_w, n_pairs, radius, center_codec, edit_codec,
     paint_mode, paint_bytes, center_len, edit_len, center_sha, edit_sha) = _ED1_HEADER.unpack_from(blob, 0)
    if magic != _ED1_MAGIC or version != 1 or paint_mode != 1:
        raise SystemExit("ED1 section header differs")
    off = _ED1_HEADER.size
    center_coded = blob[off:off + center_len]
    off += center_len
    edit_coded = blob[off:off + edit_len]
    off += edit_len
    if off != len(blob):
        raise SystemExit("ED1 section has trailing bytes")
    center_raw = _ed1_decode(center_codec, center_coded)
    edit_raw = _ed1_decode(edit_codec, edit_coded)
    if _ed1_hashlib.sha256(center_raw).digest() != center_sha:
        raise SystemExit("ED1 centerline SHA mismatch")
    if _ed1_hashlib.sha256(edit_raw).digest() != edit_sha:
        raise SystemExit("ED1 edit SHA mismatch")
    frames = _ed1_deserialize_centerlines(center_raw, int(seg_h), int(seg_w))
    if len(frames) != n_pairs:
        raise SystemExit("ED1 centerline frame count differs")
    bands = []
    offsets = [0]
    pair_counts = []
    for rows in frames:
        band = _ed1_dilate(_ed1_rasterize(rows, int(seg_h), int(seg_w)), int(radius))
        flat = np.flatnonzero(band.reshape(-1)).astype(np.int32)
        bands.append(flat)
        offsets.append(offsets[-1] + int(flat.size))
    total = offsets[-1]
    bit_bytes = (total + 7) // 8
    if len(edit_raw) != 2 * bit_bytes:
        raise SystemExit("ED1 edit raw length mismatch")
    edit_bits = _ed1_unpack_bits(edit_raw[:bit_bytes], total)
    lane_bits = _ed1_unpack_bits(edit_raw[bit_bytes:], total)
    for i in range(int(n_pairs)):
        pair_counts.append(int(edit_bits[offsets[i]:offsets[i + 1]].sum()))
    return {
        "seg_h": int(seg_h),
        "seg_w": int(seg_w),
        "bands": tuple(bands),
        "offsets": tuple(offsets),
        "edit_bits": edit_bits,
        "lane_bits": lane_bits,
        "road_rgb": np.frombuffer(paint_bytes[:3], dtype=np.uint8).copy(),
        "lane_rgb": np.frombuffer(paint_bytes[3:6], dtype=np.uint8).copy(),
        "pair_counts": tuple(pair_counts),
    }


def _ed1_apply(u8: np.ndarray, ed1, pair_index: int) -> np.ndarray:
    start = ed1["offsets"][pair_index]
    stop = ed1["offsets"][pair_index + 1]
    edit = ed1["edit_bits"][start:stop]
    if not bool(edit.any()):
        return u8
    band = ed1["bands"][pair_index]
    selected = band[edit]
    lane = ed1["lane_bits"][start:stop][edit]
    rows = selected // ed1["seg_w"]
    cols = selected % ed1["seg_w"]
    rlo, rhi, _rw = _f0pr_bilinear_axis(ed1["seg_h"], u8.shape[0])
    clo, chi, _cw = _f0pr_bilinear_axis(ed1["seg_w"], u8.shape[1])
    colors = np.where(lane[:, None], ed1["lane_rgb"][None, :], ed1["road_rgb"][None, :]).astype(np.uint8)
    out = u8.copy()
    for rr in (rlo[rows], rhi[rows]):
        for cc in (clo[cols], chi[cols]):
            out[rr, cc] = colors
    return out
'''


def _copy_runtime_tree(base_dir: Path, out_dir: Path, patched_runner: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "archive").mkdir()
    runtime_files = (
        "ddm_ix2_archive_container.py",
        "ddm_r7_token_coder.py",
        "ddm_tr1_runtime.py",
        "inflate.sh",
        "pfs1_warp_receiver.py",
        "repair_entropy_coder_runtime_adapters.py",
    )
    for name in runtime_files:
        shutil.copy2(base_dir / name, out_dir / name)
    (out_dir / "inflate_runner.py").write_text(patched_runner)


def _load_generated_runner(out_dir: Path):
    sys.path.insert(0, str(out_dir))
    try:
        spec = importlib.util.spec_from_file_location("ed1_generated_inflate_runner", out_dir / "inflate_runner.py")
        if spec is None or spec.loader is None:
            raise Ed1CarrierError("could not load generated inflate_runner spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(out_dir))
        except ValueError:
            pass


def _receiver_smoke(out_dir: Path, pair_index: int) -> dict[str, Any]:
    module = _load_generated_runner(out_dir)
    decoder = module.Decoder(out_dir / "archive")
    with_ed1 = decoder.f1(pair_index)
    ed1 = decoder._ed1
    decoder._ed1 = None
    without_ed1 = decoder.f1(pair_index)
    changed = np.any(with_ed1 != without_ed1, axis=2)
    return {
        "pair": pair_index,
        "receiver_ed1_pair_count": int(ed1["pair_counts"][pair_index]),
        "camera_pixels_changed": int(changed.sum()),
        "frame1_before_sha256": sha256_bytes(without_ed1.tobytes()),
        "frame1_after_sha256": sha256_bytes(with_ed1.tobytes()),
        "mutated": bool(np.any(changed)),
    }


def build_candidate(
    *,
    base_submission_dir: Path,
    argmax_cache: Path,
    out_dir: Path,
    smoke_pair: int,
    no_smoke: bool,
) -> dict[str, Any]:
    if out_dir.exists():
        raise Ed1CarrierError(f"output directory already exists: {out_dir}")
    gt_path = argmax_cache / "gt_argmax_n600.npy"
    current_path = argmax_cache / "cx1_argmax_n600.npy"
    gt = np.load(gt_path, mmap_mode="r")
    current = np.load(current_path, mmap_mode="r")
    build = build_ed1_section_from_argmax(gt, current)

    archive_path = base_submission_dir / "archive.zip"
    with ZipFile(archive_path, "r") as archive:
        base_payload = archive.read("0.bin")
    bulk, sections = parse_payload(base_payload)
    if len(sections) != 5:
        raise Ed1CarrierError(f"base sub_final expected 5 joint sections, got {len(sections)}")
    new_payload = build_payload(bulk, [*sections, build.section])
    new_zip = build_single_member_zip(new_payload, name="0.bin")
    patched_runner = _patch_inflate_runner((base_submission_dir / "inflate_runner.py").read_text())

    _copy_runtime_tree(base_submission_dir, out_dir, patched_runner)
    (out_dir / "archive" / "0.bin").write_bytes(new_payload)
    (out_dir / "archive.zip").write_bytes(new_zip)

    smoke = None if no_smoke else _receiver_smoke(out_dir, smoke_pair)
    archive_bytes = (out_dir / "archive.zip").stat().st_size
    rate_delta = 25.0 * (archive_bytes - BASELINE_BYTES) / RATE_DENOM
    seg_gain_at_full_survival = 100.0 * build.captured_targets / (gt.shape[0] * gt.shape[1] * gt.shape[2])
    own_break_even = rate_delta / seg_gain_at_full_survival if seg_gain_at_full_survival else float("inf")
    predicted_at_sg3_survival = BASELINE_S + rate_delta - seg_gain_at_full_survival * SG3_CHARTER_FALSIFIER
    predicted_at_full_survival = BASELINE_S + rate_delta - seg_gain_at_full_survival

    receipt = {
        "schema": "ddm_ed1_per_edge_carrier_receipt.v1",
        "axis": "[macOS-CPU advisory] scorer-free byte-closed build",
        "score_claim": False,
        "promotion_eligible": False,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "base": {
            "submission_dir": str(base_submission_dir),
            "archive_bytes": (base_submission_dir / "archive.zip").stat().st_size,
            "archive_sha256": sha256_file(base_submission_dir / "archive.zip"),
            "baseline_S": BASELINE_S,
            "baseline_d_seg": BASELINE_D_SEG,
            "baseline_d_pose": BASELINE_D_POSE,
        },
        "candidate": {
            "submission_dir": str(out_dir),
            "archive_bytes": archive_bytes,
            "archive_sha256": sha256_file(out_dir / "archive.zip"),
            "payload_sha256": sha256_bytes(new_payload),
            "ed1_section_bytes": len(build.section),
            "ed1_section_sha256": sha256_bytes(build.section),
        },
        "carrier": {
            "section_member": ED1_MEMBER_NAME,
            "chart": "degree4_lane_centerline_delta_stream_from_gt_lane_mask_openpilot_native",
            "innovation": "pair-bitpacked edit and target-class bits over radius-1 centerline band",
            "target": "Road<->Lane cells where cx1_argmax_n600 disagrees with gt_argmax_n600",
            "road_rgb": list(FP1_ROAD_RGB),
            "lane_rgb": list(FP1_LANE_RGB),
            "total_road_lane_targets": build.total_road_lane_targets,
            "captured_targets": build.captured_targets,
            "capture_fraction": build.captured_targets / build.total_road_lane_targets,
            "band_pixels": build.band_pixels,
            "centerline_raw_bytes": build.centerline_block.raw_bytes,
            "centerline_coded_bytes": build.centerline_block.coded_bytes,
            "centerline_codec": build.centerline_block.codec,
            "centerline_codec_race": build.centerline_block.race,
            "edit_raw_bytes": build.edit_block.raw_bytes,
            "edit_coded_bytes": build.edit_block.coded_bytes,
            "edit_codec": build.edit_block.codec,
            "edit_codec_race": build.edit_block.race,
        },
        "projection_not_a_score": {
            "selection_mode": "n600 full population, scorer-free cache-derived target set",
            "rate_delta_vs_baseline": rate_delta,
            "seg_gain_score_units_at_100pct_survival_no_collateral": seg_gain_at_full_survival,
            "own_byteclosed_break_even_survival_no_collateral": own_break_even,
            "sg3_charter_falsifier_survival": SG3_CHARTER_FALSIFIER,
            "predicted_S_at_sg3_survival_no_collateral": predicted_at_sg3_survival,
            "predicted_S_at_100pct_survival_no_collateral": predicted_at_full_survival,
            "survival_measurement_status": "QUEUED; not run because sg4/sb1 owns the scorer slot",
        },
        "receiver_smoke": smoke,
        "inputs": {
            "gt_argmax": str(gt_path),
            "gt_argmax_sha256": sha256_file(gt_path),
            "current_argmax": str(current_path),
            "current_argmax_sha256": sha256_file(current_path),
        },
        "per_pair_summary": {
            "pairs": len(build.per_pair),
            "max_captured_targets": max(row["captured_targets"] for row in build.per_pair),
            "nonzero_pairs": sum(1 for row in build.per_pair if row["captured_targets"] > 0),
        },
    }
    (out_dir / "ed1_byte_ledger.json").write_text(json.dumps(receipt, indent=2))
    if smoke is not None:
        (out_dir / "ed1_receiver_smoke.json").write_text(json.dumps(smoke, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-submission-dir", type=Path, default=BASE_SUBMISSION_DIR)
    parser.add_argument("--argmax-cache", type=Path, default=DEFAULT_ARGMAX_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoke-pair", type=int, default=0)
    parser.add_argument("--no-smoke", action="store_true")
    args = parser.parse_args()
    receipt = build_candidate(
        base_submission_dir=args.base_submission_dir,
        argmax_cache=args.argmax_cache,
        out_dir=args.out_dir,
        smoke_pair=args.smoke_pair,
        no_smoke=args.no_smoke,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
