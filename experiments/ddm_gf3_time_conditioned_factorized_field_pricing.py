#!/usr/bin/env python3
"""Price time-conditioned shared categorical fields before any trainer build.

The proof and the fit are deliberately different objects:

* ``relaxed_translation_plurality_lower_bound`` is a certified lower bound.
  It lets every frame choose a different allowed translation at every lattice
  cell, then takes the best shared class at that cell.  That relaxation is a
  superset of one global translation per frame, so its error cannot exceed the
  optimum of the declared GOP field plus global-offset family.
* ``observed_member`` is an achievable upper bound.  It starts from a GOP
  plurality, exhaustively fits one global integer offset per frame, and updates
  the GOP plurality once for those fixed offsets.  It is one block-coordinate
  sweep and is never called an optimum or a bound.

Every certificate, packet, correction stream, decode, deterministic repeat,
checkpoint, and manifest is retained.  This runner is CPU-only and scorer-free.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import platform
import resource
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_up2_shipping_pose_solve as up2

SCHEMA: Final = "ddm_gf3_time_conditioned_factorized_field_pricing.v1"
MANIFEST_SCHEMA: Final = "ddm_gf3_retained_manifest.v1"
AXIS: Final = "[macOS-CPU scorer-free exact field measurement, n600]"
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
NUM_CLASSES: Final = 5
FIELD_SHAPE: Final = (N_PAIRS, HEIGHT, WIDTH)
FIELD_BYTES: Final = int(np.prod(FIELD_SHAPE))
FIELD_SHA256: Final = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
DEFAULT_FIELD: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/"
    "converged_v3/retained/source_afr1_jbp1_field.u8"
)
OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_gf3_time_conditioned_pricing")
GOP_LENGTHS: Final = (2, 3, 5, 10, 20, 50)
SEARCH_RADIUS: Final = 12
FILL_CLASS: Final = 2
PACKET_CAP_BYTES: Final = 71_404.5
MISMATCH_TARGET: Final = 46_804
REPLACEMENT_CAP_BYTES: Final = 85_020
MINIMUM_FREE_BYTES: Final = 1 << 30
PEAK_RSS_LIMIT_BYTES: Final = 20_000_000_000

PACKET_MAGIC: Final = b"GF3P"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct("<4sBHHHHI")
CODER_SUFFIXES: Final = {
    "brotli_q11": "br",
    "zlib_9": "zlib",
    "lzma2_extreme": "xz",
}

LTG1_EXACT_LANE_PACKET_BYTES: Final = 233_262
BLP1_WEIGHTS_BEFORE_RESIDUAL_BYTES: Final = 60_191
GF1_LANE_STREAM_BYTES: Final = 36_044
LANE_CARRIAGE_DOOR_BYTES: Final = 21_699


class GF3Error(RuntimeError):
    """A theorem input, packet, custody, or resource invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fact_matches(fact: object) -> bool:
    if not isinstance(fact, Mapping) or "path" not in fact:
        return False
    path = Path(str(fact["path"]))
    return bool(
        path.is_file() and path.stat().st_size == int(fact.get("bytes", -1)) and sha256_file(path) == fact.get("sha256")
    )


def atomic_bytes_once(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        fact = file_fact(path)
        if fact["bytes"] != len(payload) or fact["sha256"] != sha256_bytes(payload):
            raise GF3Error(f"refusing to overwrite changed retained payload: {path}")
        return fact
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return file_fact(path)


def atomic_array_once(path: Path, array: np.ndarray) -> dict[str, object]:
    return atomic_bytes_once(path, np.ascontiguousarray(array).tobytes())


def atomic_json_once(path: Path, value: object) -> dict[str, object]:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return atomic_bytes_once(path, payload)


def canonical_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Return deterministic compressed NPZ bytes with fixed member metadata."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def atomic_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return atomic_bytes_once(path, canonical_npz_bytes(arrays))


def peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(observed if sys.platform == "darwin" else observed * 1024)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_source(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise GF3Error(f"exact source field does not exist: {path}")
    fact = file_fact(path)
    if fact["bytes"] != FIELD_BYTES or fact["sha256"] != FIELD_SHA256:
        raise GF3Error(f"exact field identity mismatch: {fact}")
    return fact


def storage_preflight(output: Path, *, label: str) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(output)
    free = int(stat.f_bavail * stat.f_frsize)
    row = {
        "path": str(output),
        "observed_free_bytes": free,
        "required_free_bytes": MINIMUM_FREE_BYTES,
        "status": "PASS" if free >= MINIMUM_FREE_BYTES else "REFUSED",
        "policy": "certify-or-block; no generated payload is deleted",
    }
    receipt_path = output / (f"stage_checkpoints/00_storage_preflight_{label}_{time.time_ns()}_{os.getpid()}.json")
    row["receipt"] = atomic_json_once(receipt_path, row)
    if free < MINIMUM_FREE_BYTES:
        raise GF3Error(f"storage preflight refused: {row}")
    return row


def window_class_presence(frame: np.ndarray, radius: int) -> np.ndarray:
    """Exact reachable-class presence on the common translation interior."""

    values = np.asarray(frame)
    if values.ndim != 2 or radius < 0:
        raise GF3Error("window presence expects a 2D field and non-negative radius")
    height, width = values.shape
    if 2 * radius >= height or 2 * radius >= width:
        raise GF3Error("translation radius leaves no common interior")
    if np.any(values >= NUM_CLASSES):
        raise GF3Error("categorical field carries a class outside [0,4]")
    span = 2 * radius + 1
    result = np.empty((NUM_CLASSES, height - 2 * radius, width - 2 * radius), dtype=np.bool_)
    for label in range(NUM_CLASSES):
        integral = np.zeros((height + 1, width + 1), dtype=np.int32)
        integral[1:, 1:] = np.cumsum(
            np.cumsum(values == label, axis=0, dtype=np.int32),
            axis=1,
            dtype=np.int32,
        )
        window_count = (
            integral[span:, span:] - integral[:-span, span:] - integral[span:, :-span] + integral[:-span, :-span]
        )
        result[label] = window_count > 0
    return result


def build_presence_cache(target: np.ndarray, radius: int) -> np.ndarray:
    if target.ndim != 3:
        raise GF3Error("presence cache expects (frames,height,width)")
    frames, height, width = target.shape
    cache = np.empty(
        (
            frames,
            NUM_CLASSES,
            height - 2 * radius,
            width - 2 * radius,
        ),
        dtype=np.bool_,
    )
    for frame in range(frames):
        cache[frame] = window_class_presence(target[frame], radius)
    return cache


def pack_presence(cache: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(cache).reshape(cache.shape[0], cache.shape[1], -1)
    return np.packbits(flat, axis=2, bitorder="little")


def unpack_presence(packed: np.ndarray, *, interior_height: int, interior_width: int) -> np.ndarray:
    count = interior_height * interior_width
    unpacked = np.unpackbits(packed, axis=2, count=count, bitorder="little")
    return unpacked.reshape(packed.shape[0], packed.shape[1], interior_height, interior_width).astype(
        np.bool_, copy=False
    )


def relaxed_translation_plurality_lower_bound(
    presence: np.ndarray, gop_length: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return full cellwise certificates and the exact relaxed lower bound."""

    if presence.ndim != 4 or presence.shape[1] != NUM_CLASSES:
        raise GF3Error("presence must have shape (frames,5,height,width)")
    frame_count = presence.shape[0]
    if gop_length <= 0 or frame_count % gop_length:
        raise GF3Error("GOP length must divide the frame count")
    gops = frame_count // gop_length
    maps = np.empty((gops, *presence.shape[2:]), dtype=np.uint8)
    totals = np.empty(gops, dtype=np.int64)
    for group in range(gops):
        start = group * gop_length
        stop = start + gop_length
        votes = presence[start:stop].sum(axis=0, dtype=np.uint16)
        maps[group] = gop_length - votes.max(axis=0)
        totals[group] = maps[group].sum(dtype=np.int64)
    return maps, totals, int(totals.sum(dtype=np.int64))


def plurality_field(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim != 3:
        raise GF3Error("plurality expects (frames,height,width)")
    counts = np.empty((NUM_CLASSES, *values.shape[1:]), dtype=np.uint16)
    for label in range(NUM_CLASSES):
        counts[label] = np.count_nonzero(values == label, axis=0)
    return np.argmax(counts, axis=0).astype(np.uint8)


def translation_slices(height: int, width: int, dy: int, dx: int) -> tuple[slice, slice, slice, slice]:
    source_y = slice(max(0, -dy), min(height, height - dy))
    source_x = slice(max(0, -dx), min(width, width - dx))
    target_y = slice(max(0, dy), min(height, height + dy))
    target_x = slice(max(0, dx), min(width, width + dx))
    return source_y, source_x, target_y, target_x


def render_translation(template: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = template.shape
    rendered = np.full((height, width), FILL_CLASS, dtype=np.uint8)
    sy, sx, ty, tx = translation_slices(height, width, dy, dx)
    rendered[ty, tx] = template[sy, sx]
    return rendered


def ordered_translations(radius: int) -> list[tuple[int, int]]:
    values = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    return sorted(
        values,
        key=lambda row: (
            abs(row[0]) + abs(row[1]),
            abs(row[0]),
            abs(row[1]),
            row,
        ),
    )


def fit_offsets_to_field(frames: np.ndarray, template: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustively fit a global integer offset per frame for a fixed field."""

    best = np.full(frames.shape[0], -1, dtype=np.int32)
    offsets = np.zeros((frames.shape[0], 2), dtype=np.int8)
    for dy, dx in ordered_translations(radius):
        rendered = render_translation(template, dy, dx)
        scores = np.count_nonzero(frames == rendered, axis=(1, 2))
        improve = scores > best
        if np.any(improve):
            best[improve] = scores[improve]
            offsets[improve, 0] = dy
            offsets[improve, 1] = dx
    return offsets, best


def plurality_after_offsets(frames: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    height, width = frames.shape[1:]
    counts = np.zeros((NUM_CLASSES, height, width), dtype=np.uint16)
    for frame, (dy_raw, dx_raw) in zip(frames, offsets.tolist(), strict=True):
        dy, dx = int(dy_raw), int(dx_raw)
        sy, sx, ty, tx = translation_slices(height, width, dy, dx)
        aligned = frame[ty, tx]
        for label in range(NUM_CLASSES):
            counts[label, sy, sx] += aligned == label
    return np.argmax(counts, axis=0).astype(np.uint8)


def render_gop_fields(fields: np.ndarray, offsets: np.ndarray, gop_length: int) -> np.ndarray:
    rendered = np.empty((offsets.shape[0], *fields.shape[1:]), dtype=np.uint8)
    for frame, (dy_raw, dx_raw) in enumerate(offsets.tolist()):
        rendered[frame] = render_translation(fields[frame // gop_length], int(dy_raw), int(dx_raw))
    return rendered


def observed_member(
    target: np.ndarray, gop_length: int, radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build one legal member with one exact block-coordinate sweep."""

    if target.shape[0] % gop_length:
        raise GF3Error("GOP length must divide target frames")
    gops = target.shape[0] // gop_length
    zero_fields = np.empty((gops, *target.shape[1:]), dtype=np.uint8)
    for group in range(gops):
        start = group * gop_length
        zero_fields[group] = plurality_field(target[start : start + gop_length])
    zero_offsets = np.zeros((target.shape[0], 2), dtype=np.int8)
    zero_decode = render_gop_fields(zero_fields, zero_offsets, gop_length)
    zero_mismatches = int(np.count_nonzero(zero_decode != target))
    del zero_decode

    offsets = np.empty((target.shape[0], 2), dtype=np.int8)
    fields = np.empty_like(zero_fields)
    for group in range(gops):
        start = group * gop_length
        stop = start + gop_length
        group_offsets, _ = fit_offsets_to_field(target[start:stop], zero_fields[group], radius)
        offsets[start:stop] = group_offsets
        fields[group] = plurality_after_offsets(target[start:stop], group_offsets)
    decoded = render_gop_fields(fields, offsets, gop_length)
    mismatches = int(np.count_nonzero(decoded != target))
    if mismatches > zero_mismatches:
        raise GF3Error("one exact block-coordinate sweep increased mismatch count")
    return fields, offsets, decoded, zero_mismatches, mismatches


def pack_packet(fields: np.ndarray, offsets: np.ndarray, gop_length: int) -> bytes:
    if fields.shape != (N_PAIRS // gop_length, HEIGHT, WIDTH):
        raise GF3Error("shared-field shape does not match GOP length")
    if offsets.shape != (N_PAIRS, 2) or np.any(np.abs(offsets.astype(np.int16)) > SEARCH_RADIUS):
        raise GF3Error("offset shape or range is invalid")
    header = PACKET_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        gop_length,
        HEIGHT,
        WIDTH,
        fields.shape[0],
        N_PAIRS,
    )
    return (
        header
        + np.ascontiguousarray(fields, dtype=np.uint8).tobytes()
        + np.ascontiguousarray(offsets, dtype=np.int8).tobytes()
    )


def unpack_packet(payload: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    if len(payload) < PACKET_HEADER.size:
        raise GF3Error("GF3 packet is truncated")
    magic, version, gop_length, height, width, gops, pairs = PACKET_HEADER.unpack_from(payload)
    if (magic, version, height, width, pairs) != (
        PACKET_MAGIC,
        PACKET_VERSION,
        HEIGHT,
        WIDTH,
        N_PAIRS,
    ):
        raise GF3Error("GF3 packet header differs from the declared family")
    if gop_length not in GOP_LENGTHS or gops != N_PAIRS // gop_length:
        raise GF3Error("GF3 packet GOP geometry is invalid")
    field_bytes = gops * HEIGHT * WIDTH
    offset_bytes = N_PAIRS * 2
    if len(payload) != PACKET_HEADER.size + field_bytes + offset_bytes:
        raise GF3Error("GF3 packet length differs from its header")
    body = memoryview(payload)[PACKET_HEADER.size :]
    fields = np.frombuffer(body[:field_bytes], dtype=np.uint8).reshape(gops, HEIGHT, WIDTH).copy()
    offsets = np.frombuffer(body[field_bytes:], dtype=np.int8).reshape(N_PAIRS, 2).copy()
    if np.any(fields >= NUM_CLASSES) or np.any(np.abs(offsets.astype(np.int16)) > SEARCH_RADIUS):
        raise GF3Error("GF3 packet carries an invalid field or offset")
    return gop_length, fields, offsets


def coder_race(name: str, raw: bytes, output: Path) -> dict[str, object]:
    encoders = {
        "brotli_q11": lambda: brotli.compress(raw, quality=11),
        "zlib_9": lambda: zlib.compress(raw, level=9),
        "lzma2_extreme": lambda: lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME),
    }
    decoders = {
        "brotli_q11": brotli.decompress,
        "zlib_9": zlib.decompress,
        "lzma2_extreme": lzma.decompress,
    }
    rows: dict[str, dict[str, object]] = {}
    for coder, encode in encoders.items():
        payload = encode()
        repeat = encode()
        if payload != repeat:
            raise GF3Error(f"{name}/{coder} deterministic repeat differs")
        if decoders[coder](payload) != raw:
            raise GF3Error(f"{name}/{coder} parse-back differs")
        suffix = CODER_SUFFIXES[coder]
        primary = atomic_bytes_once(output / f"coded/{name}.{suffix}", payload)
        repeated = atomic_bytes_once(output / f"coded/{name}.repeat.{suffix}", repeat)
        primary_parseback = decoders[coder](Path(str(primary["path"])).read_bytes())
        repeat_parseback = decoders[coder](Path(str(repeated["path"])).read_bytes())
        if primary_parseback != raw or repeat_parseback != raw:
            raise GF3Error(f"{name}/{coder} persisted parse-back differs")
        rows[coder] = {
            "primary": primary,
            "repeat": repeated,
            "byte_identical": True,
            "parseback_exact": True,
        }
    selected = min(rows, key=lambda coder: int(rows[coder]["primary"]["bytes"]))
    return {
        "raw_bytes": len(raw),
        "coders": rows,
        "selected_coder": selected,
        "selected_bytes": int(rows[selected]["primary"]["bytes"]),
        "rc64_not_run": (
            "the shipping HPAC RC64 needs receiver-model probabilities and is a different "
            "object; no 32-bit or static coder is relabelled RC64"
        ),
    }


def selected_coder_parseback(race: Mapping[str, object]) -> bytes:
    coder = str(race["selected_coder"])
    row = race["coders"][coder]
    payload = Path(str(row["primary"]["path"])).read_bytes()
    if coder == "brotli_q11":
        return brotli.decompress(payload)
    if coder == "zlib_9":
        return zlib.decompress(payload)
    if coder == "lzma2_extreme":
        return lzma.decompress(payload)
    raise GF3Error(f"unsupported selected coder: {coder}")


def apply_domain_residual(predicted: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if predicted.ndim != 3:
        raise GF3Error("domain residual prediction must have (frames,height,width) geometry")
    frames, height, width = predicted.shape
    if residual.shape != (height, width, frames):
        raise GF3Error("domain residual or prediction has invalid geometry")
    predicted_domain = np.ascontiguousarray(predicted.transpose(1, 2, 0)).reshape(-1)
    residual_flat = np.ascontiguousarray(residual).reshape(-1)
    if np.any(residual_flat > NUM_CLASSES):
        raise GF3Error("domain residual carries a symbol outside [0,5]")
    decoded_domain = np.where(residual_flat == 0, predicted_domain, residual_flat - 1).astype(np.uint8)
    return decoded_domain.reshape(height, width, frames).transpose(2, 0, 1)


def domain_residual_to_target(
    target: np.ndarray, predicted: np.ndarray, remaining_target: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Correct a deterministic pixel-time prefix and leave the requested debt."""

    mismatch = predicted != target
    observed = int(np.count_nonzero(mismatch))
    corrections = max(0, observed - remaining_target)
    mismatch_domain = np.ascontiguousarray(mismatch.transpose(1, 2, 0)).reshape(-1)
    target_domain = np.ascontiguousarray(target.transpose(1, 2, 0)).reshape(-1)
    residual_flat = np.zeros(mismatch_domain.size, dtype=np.uint8)
    mismatch_positions = np.flatnonzero(mismatch_domain)
    chosen = mismatch_positions[:corrections]
    residual_flat[chosen] = target_domain[chosen] + 1
    residual = residual_flat.reshape(HEIGHT, WIDTH, N_PAIRS)

    decoded = apply_domain_residual(predicted, residual)
    remaining = int(np.count_nonzero(decoded != target))
    expected = min(observed, remaining_target)
    if remaining != expected:
        raise GF3Error(f"residual left {remaining} mismatches instead of {expected}")
    return residual, decoded, corrections, remaining


def inventory(root: Path, manifest: Path) -> list[dict[str, object]]:
    return [file_fact(path) for path in sorted(root.rglob("*")) if path.is_file() and path != manifest]


def presence_stage(
    output: Path, target: np.ndarray, source: dict[str, object], runner: dict[str, object]
) -> np.ndarray:
    checkpoint_path = output / "stage_checkpoints/01_presence_complete.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema") != SCHEMA
            or checkpoint.get("source_field") != source
            or checkpoint.get("runner") != runner
            or not fact_matches(checkpoint.get("primary"))
            or not fact_matches(checkpoint.get("repeat"))
        ):
            raise GF3Error("presence checkpoint is stale or belongs to different inputs")
        if checkpoint["primary"]["sha256"] != checkpoint["repeat"]["sha256"]:
            raise GF3Error("presence checkpoint repeat differs")
        with np.load(checkpoint["primary"]["path"], allow_pickle=False) as archive:
            packed = np.asarray(archive["presence_packbits"], dtype=np.uint8)
        return unpack_presence(
            packed,
            interior_height=HEIGHT - 2 * SEARCH_RADIUS,
            interior_width=WIDTH - 2 * SEARCH_RADIUS,
        )

    presence = build_presence_cache(target, SEARCH_RADIUS)
    packed = pack_presence(presence)
    primary = atomic_npz_once(output / "retained/presence_cache.npz", {"presence_packbits": packed})
    repeated_presence = build_presence_cache(target, SEARCH_RADIUS)
    repeated_packed = pack_presence(repeated_presence)
    repeat = atomic_npz_once(
        output / "retained/presence_cache.repeat.npz",
        {"presence_packbits": repeated_packed},
    )
    if primary["sha256"] != repeat["sha256"] or not np.array_equal(packed, repeated_packed):
        raise GF3Error("independent presence-cache repeat differs")
    atomic_json_once(
        checkpoint_path,
        {
            "schema": SCHEMA,
            "source_field": source,
            "runner": runner,
            "primary": primary,
            "repeat": repeat,
            "byte_identical_repeat": True,
            "shape_before_packbits": list(presence.shape),
            "common_interior": [
                SEARCH_RADIUS,
                HEIGHT - SEARCH_RADIUS,
                SEARCH_RADIUS,
                WIDTH - SEARCH_RADIUS,
            ],
        },
    )
    return presence


def run_gop(
    *,
    output: Path,
    target: np.ndarray,
    presence: np.ndarray,
    source: dict[str, object],
    runner: dict[str, object],
    gop_length: int,
) -> dict[str, object]:
    stage = output / f"L_{gop_length:03d}"
    result_path = stage / "RESULT.json"
    manifest_path = stage / "MANIFEST.json"
    checkpoint_path = output / f"stage_checkpoints/02_L_{gop_length:03d}_complete.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("schema") != SCHEMA
            or result.get("gop_length") != gop_length
            or result.get("source_field") != source
            or result.get("provenance", {}).get("runner") != runner
        ):
            raise GF3Error(f"completed L={gop_length} stage is stale")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            entries = inventory(stage, manifest_path)
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "gop_length": gop_length,
                "entry_count": len(entries),
                "total_bytes": sum(int(entry["bytes"]) for entry in entries),
                "entries": entries,
            }
            atomic_json_once(manifest_path, manifest)
        if manifest.get("schema") != MANIFEST_SCHEMA or any(
            not fact_matches(fact) for fact in manifest.get("entries", [])
        ):
            raise GF3Error(f"completed L={gop_length} manifest is stale")
        atomic_json_once(
            checkpoint_path,
            {
                "schema": SCHEMA,
                "gop_length": gop_length,
                "result": file_fact(result_path),
                "manifest": file_fact(manifest_path),
                "runner": runner,
                "source_field": source,
            },
        )
        return result
    if manifest_path.exists():
        raise GF3Error(f"L={gop_length} manifest exists without its result")

    started = time.time()
    stage.mkdir(parents=True, exist_ok=True)
    bound_maps, per_gop_bounds, lower_bound = relaxed_translation_plurality_lower_bound(presence, gop_length)
    certificate_arrays = {
        "cellwise_lower_bounds": bound_maps,
        "per_gop_lower_bounds": per_gop_bounds,
    }
    certificate = atomic_npz_once(stage / "retained/lower_bound_certificate.npz", certificate_arrays)
    repeat_maps, repeat_totals, repeat_bound = relaxed_translation_plurality_lower_bound(presence, gop_length)
    certificate_repeat = atomic_npz_once(
        stage / "retained/lower_bound_certificate.repeat.npz",
        {
            "cellwise_lower_bounds": repeat_maps,
            "per_gop_lower_bounds": repeat_totals,
        },
    )
    if (
        lower_bound != repeat_bound
        or not np.array_equal(bound_maps, repeat_maps)
        or not np.array_equal(per_gop_bounds, repeat_totals)
        or certificate["sha256"] != certificate_repeat["sha256"]
    ):
        raise GF3Error(f"L={gop_length} lower-bound repeat differs")

    fields, offsets, decoded, zero_mismatches, observed_mismatches = observed_member(target, gop_length, SEARCH_RADIUS)
    observed_arrays = {"decoded": decoded, "fields": fields, "offsets": offsets}
    observed_fact = atomic_npz_once(stage / "retained/observed_member.npz", observed_arrays)
    repeat_fields, repeat_offsets, repeat_decoded, repeat_zero, repeat_mismatches = observed_member(
        target, gop_length, SEARCH_RADIUS
    )
    observed_repeat = atomic_npz_once(
        stage / "retained/observed_member.repeat.npz",
        {"decoded": repeat_decoded, "fields": repeat_fields, "offsets": repeat_offsets},
    )
    if (
        zero_mismatches != repeat_zero
        or observed_mismatches != repeat_mismatches
        or not np.array_equal(fields, repeat_fields)
        or not np.array_equal(offsets, repeat_offsets)
        or not np.array_equal(decoded, repeat_decoded)
        or observed_fact["sha256"] != observed_repeat["sha256"]
    ):
        raise GF3Error(f"L={gop_length} independent observed-member repeat differs")

    raw_packet = pack_packet(fields, offsets, gop_length)
    packet_raw_fact = atomic_bytes_once(stage / "retained/gop_fields_offsets.packet.raw", raw_packet)
    repeat_packet = pack_packet(repeat_fields, repeat_offsets, gop_length)
    packet_raw_repeat = atomic_bytes_once(stage / "retained/gop_fields_offsets.packet.repeat.raw", repeat_packet)
    if raw_packet != repeat_packet or packet_raw_fact["sha256"] != packet_raw_repeat["sha256"]:
        raise GF3Error(f"L={gop_length} independent raw-packet repeat differs")
    parsed_length, parsed_fields, parsed_offsets = unpack_packet(raw_packet)
    if (
        parsed_length != gop_length
        or not np.array_equal(parsed_fields, fields)
        or not np.array_equal(parsed_offsets, offsets)
    ):
        raise GF3Error(f"L={gop_length} raw packet parse-back differs")
    packet_decode = render_gop_fields(parsed_fields, parsed_offsets, gop_length)
    if not np.array_equal(packet_decode, decoded):
        raise GF3Error(f"L={gop_length} packet decode differs from observed member")
    packet_decode_fact = atomic_npz_once(stage / "retained/packet_decode.npz", {"decoded": packet_decode})
    _, repeat_parsed_fields, repeat_parsed_offsets = unpack_packet(repeat_packet)
    repeat_packet_decode = render_gop_fields(repeat_parsed_fields, repeat_parsed_offsets, gop_length)
    packet_decode_repeat = atomic_npz_once(
        stage / "retained/packet_decode.repeat.npz", {"decoded": repeat_packet_decode}
    )
    if (
        not np.array_equal(packet_decode, repeat_packet_decode)
        or packet_decode_fact["sha256"] != packet_decode_repeat["sha256"]
    ):
        raise GF3Error(f"L={gop_length} independent packet-decode repeat differs")
    packet_race = coder_race("gop_fields_offsets.packet", raw_packet, stage)
    persisted_packet = selected_coder_parseback(packet_race)
    persisted_length, persisted_fields, persisted_offsets = unpack_packet(persisted_packet)
    persisted_packet_decode = render_gop_fields(persisted_fields, persisted_offsets, gop_length)
    if (
        persisted_length != gop_length
        or not np.array_equal(persisted_fields, fields)
        or not np.array_equal(persisted_offsets, offsets)
        or not np.array_equal(persisted_packet_decode, decoded)
    ):
        raise GF3Error(f"L={gop_length} persisted coded packet receiver differs")

    residual, residual_decode, corrections, remaining = domain_residual_to_target(target, decoded, MISMATCH_TARGET)
    residual_raw_fact = atomic_array_once(stage / "retained/residual.pixel_time.u8", residual)
    repeat_residual, repeat_residual_decode, repeat_corrections, repeat_remaining = domain_residual_to_target(
        target, repeat_decoded, MISMATCH_TARGET
    )
    residual_raw_repeat = atomic_array_once(stage / "retained/residual.pixel_time.repeat.u8", repeat_residual)
    residual_decode_fact = atomic_npz_once(stage / "retained/residual_decode.npz", {"decoded": residual_decode})
    residual_decode_repeat = atomic_npz_once(
        stage / "retained/residual_decode.repeat.npz", {"decoded": repeat_residual_decode}
    )
    if (
        corrections != repeat_corrections
        or remaining != repeat_remaining
        or not np.array_equal(residual, repeat_residual)
        or not np.array_equal(residual_decode, repeat_residual_decode)
        or residual_raw_fact["sha256"] != residual_raw_repeat["sha256"]
        or residual_decode_fact["sha256"] != residual_decode_repeat["sha256"]
    ):
        raise GF3Error(f"L={gop_length} independent residual repeat differs")
    if corrections:
        residual_race = coder_race("residual.pixel_time", np.ascontiguousarray(residual).tobytes(), stage)
        persisted_residual_raw = selected_coder_parseback(residual_race)
        persisted_residual = np.frombuffer(persisted_residual_raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, N_PAIRS)
        persisted_residual_decode = apply_domain_residual(decoded, persisted_residual)
        if not np.array_equal(persisted_residual_decode, residual_decode):
            raise GF3Error(f"L={gop_length} persisted coded residual receiver differs")
    else:
        residual_race = {
            "raw_bytes": int(residual.nbytes),
            "coders": {},
            "selected_coder": "no_section",
            "selected_bytes": 0,
            "parseback_exact": True,
            "reason": "the observed member already meets the mismatch target",
            "rc64_not_run": "no residual section exists",
        }

    packet_bytes = int(packet_race["selected_bytes"])
    residual_bytes = int(residual_race["selected_bytes"])
    packet_plus_residual = packet_bytes + residual_bytes
    bound_direct = lower_bound <= MISMATCH_TARGET and packet_bytes <= PACKET_CAP_BYTES
    observed_direct = observed_mismatches <= MISMATCH_TARGET and packet_bytes <= PACKET_CAP_BYTES
    observed_hybrid = packet_plus_residual <= REPLACEMENT_CAP_BYTES
    if observed_direct or observed_hybrid:
        disposition = "ACHIEVABLE-ROW-ADMITS-BUILD"
    elif bound_direct:
        disposition = "BOUND-ADMITS-BUILD"
    else:
        disposition = "ROW-REFUSED"

    rss = peak_rss_bytes()
    if rss > PEAK_RSS_LIMIT_BYTES:
        raise GF3Error(f"peak RSS {rss} exceeds {PEAK_RSS_LIMIT_BYTES}")
    result: dict[str, object] = {
        "schema": SCHEMA,
        "axis": AXIS,
        "source_field": source,
        "gop_length": gop_length,
        "gop_count": N_PAIRS // gop_length,
        "certified_lower_bound": {
            "mismatches": lower_bound,
            "denominator": int(N_PAIRS * (HEIGHT - 2 * SEARCH_RADIUS) * (WIDTH - 2 * SEARCH_RADIUS)),
            "per_gop_min": int(per_gop_bounds.min()),
            "per_gop_median": float(np.median(per_gop_bounds)),
            "per_gop_max": int(per_gop_bounds.max()),
            "certificate": certificate,
            "certificate_repeat": certificate_repeat,
            "byte_identical_repeat": True,
            "theorem": (
                "Each cell independently grants every frame any translation in [-12,+12]^2, "
                "then selects the plurality class. This cellwise translation relaxation is a "
                "superset of one global offset per frame. All border errors are discarded."
            ),
            "global_optimum_claim": False,
        },
        "observed_fit_upper_bound": {
            "method": (
                "zero-offset exact GOP plurality, exhaustive best global offset per frame for "
                "that fixed field, then one exact plurality update at fixed offsets"
            ),
            "coordinate_sweeps": 1,
            "global_optimum_claim": False,
            "zero_offset_mismatches": zero_mismatches,
            "post_sweep_mismatches": observed_mismatches,
            "gap_above_certified_lower_bound": observed_mismatches - lower_bound,
            "observed_member": observed_fact,
            "observed_member_repeat": observed_repeat,
            "packet_decode": packet_decode_fact,
            "packet_decode_repeat": packet_decode_repeat,
            "decode_repeat_byte_identical": (packet_decode_fact["sha256"] == packet_decode_repeat["sha256"]),
        },
        "packet": {
            "raw": packet_raw_fact,
            "raw_repeat": packet_raw_repeat,
            "coder_race": packet_race,
            "selected_coder": packet_race["selected_coder"],
            "selected_bytes": packet_bytes,
            "fields_and_offsets_parseback_exact": True,
            "decoded_observed_field_exact": True,
        },
        "domain_matched_residual_to_target": {
            "ordering": "pixel_time: all 600 pair symbols for one (y,x) are contiguous",
            "raw": residual_raw_fact,
            "raw_repeat": residual_raw_repeat,
            "decoded": residual_decode_fact,
            "decoded_repeat": residual_decode_repeat,
            "corrections_applied": corrections,
            "remaining_mismatches": remaining,
            "coder_race": residual_race,
            "selected_coder": residual_race["selected_coder"],
            "selected_bytes": residual_bytes,
            "receiver_verified": True,
        },
        "pricing": {
            "packet_bytes": packet_bytes,
            "residual_bytes": residual_bytes,
            "packet_plus_residual_bytes": packet_plus_residual,
            "packet_cap_bytes": PACKET_CAP_BYTES,
            "mismatch_target": MISMATCH_TARGET,
            "replacement_cap_bytes": REPLACEMENT_CAP_BYTES,
            "bound_direct_gate_passes": bound_direct,
            "observed_direct_gate_passes": observed_direct,
            "observed_hybrid_gate_passes": observed_hybrid,
            "union_gate_passes": bound_direct or observed_hybrid,
            "disposition": disposition,
        },
        "boundary_obligations": {
            "ltg1_exact_lane_topology_plus_shape_bytes": LTG1_EXACT_LANE_PACKET_BYTES,
            "blp1_receiver_consumed_weights_before_residual_bytes": (BLP1_WEIGHTS_BEFORE_RESIDUAL_BYTES),
            "gf1_lossy_lane_stream_bytes": GF1_LANE_STREAM_BYTES,
            "lane_carriage_door_bytes": LANE_CARRIAGE_DOOR_BYTES,
            "accounting": (
                "comparison facts only; they are not added to the whole-field GOP packet because "
                "that would double-count Lane already represented in each shared field"
            ),
        },
        "resource": {
            "elapsed_seconds": time.time() - started,
            "peak_rss_bytes": rss,
            "peak_rss_limit_bytes": PEAK_RSS_LIMIT_BYTES,
        },
        "provenance": {
            "runner": runner,
            "git_head_before_commit": git_head(),
            "argv": sys.argv,
            "seed": None,
            "determinism": "closed-form integer operations; no RNG",
            "execution_env": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "brotli": getattr(brotli, "__version__", "unknown"),
        },
        "constraints": {
            "scorer_invoked": False,
            "training_invoked": False,
            "modal_invoked": False,
            "metal_or_mps_invoked": False,
            "upstream_modified": False,
            "submission_modified": False,
            "score_claim": False,
            "pointer_moved": False,
            "all_materialized_payloads_retained": True,
        },
    }
    atomic_json_once(result_path, result)
    entries = inventory(stage, manifest_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "gop_length": gop_length,
        "entry_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "entries": entries,
    }
    atomic_json_once(manifest_path, manifest)
    atomic_json_once(
        checkpoint_path,
        {
            "schema": SCHEMA,
            "gop_length": gop_length,
            "result": file_fact(result_path),
            "manifest": file_fact(manifest_path),
            "runner": runner,
            "source_field": source,
        },
    )
    return result


def finalize(
    output: Path,
    source: dict[str, object],
    runner: dict[str, object],
    lineage: dict[str, str],
    storage: dict[str, object],
) -> dict[str, object]:
    result_path = output / "RESULT.json"
    manifest_path = output / "MANIFEST.json"
    checkpoint_path = output / "stage_checkpoints/03_finalize_result_complete.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("schema") != SCHEMA
            or result.get("source_field") != source
            or result.get("provenance", {}).get("runner") != runner
        ):
            raise GF3Error("completed final result is stale")
        atomic_json_once(
            checkpoint_path,
            {
                "schema": SCHEMA,
                "result": file_fact(result_path),
                "runner": runner,
                "source_field": source,
            },
        )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            entries = inventory(output, manifest_path)
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "entry_count": len(entries),
                "total_bytes": sum(int(entry["bytes"]) for entry in entries),
                "entries": entries,
            }
            atomic_json_once(manifest_path, manifest)
        if manifest.get("schema") != MANIFEST_SCHEMA or any(
            not fact_matches(fact) for fact in manifest.get("entries", [])
        ):
            raise GF3Error("completed final manifest is stale")
        return result
    if manifest_path.exists():
        raise GF3Error("final manifest exists without its result")

    rows = []
    for gop_length in GOP_LENGTHS:
        path = output / f"L_{gop_length:03d}/RESULT.json"
        if not path.is_file():
            raise GF3Error(f"cannot finalize before L={gop_length} completes")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("source_field") != source or row.get("provenance", {}).get("runner") != runner:
            raise GF3Error(f"L={gop_length} result belongs to different inputs")
        rows.append(row)
    admitting = [
        row
        for row in rows
        if row["pricing"]["bound_direct_gate_passes"] or row["pricing"]["observed_hybrid_gate_passes"]
    ]
    achievable = [row for row in rows if row["pricing"]["observed_hybrid_gate_passes"]]
    if achievable:
        disposition = "ACHIEVABLE-ROW-ADMITS-BUILD"
    elif admitting:
        disposition = "BOUND-ADMITS-BUILD"
    else:
        disposition = "PRICING-CLOSED"
    summary_rows = [
        {
            "gop_length": row["gop_length"],
            "certified_lower_bound_mismatches": row["certified_lower_bound"]["mismatches"],
            "observed_fit_mismatches": row["observed_fit_upper_bound"]["post_sweep_mismatches"],
            "observed_minus_bound": row["observed_fit_upper_bound"]["gap_above_certified_lower_bound"],
            "packet_bytes": row["pricing"]["packet_bytes"],
            "residual_bytes": row["pricing"]["residual_bytes"],
            "packet_plus_residual_bytes": row["pricing"]["packet_plus_residual_bytes"],
            "bound_direct_gate_passes": row["pricing"]["bound_direct_gate_passes"],
            "observed_direct_gate_passes": row["pricing"]["observed_direct_gate_passes"],
            "observed_hybrid_gate_passes": row["pricing"]["observed_hybrid_gate_passes"],
            "disposition": row["pricing"]["disposition"],
        }
        for row in rows
    ]
    result = {
        "schema": SCHEMA,
        "axis": AXIS,
        "source_field": source,
        "lineage_gate": lineage,
        "storage_preflight": storage,
        "gop_lengths": list(GOP_LENGTHS),
        "rows": summary_rows,
        "decision": {
            "disposition": disposition,
            "admitting_gop_lengths": [row["gop_length"] for row in admitting],
            "verdict_scope": (
                "the declared discrete GOP grid of full categorical shared fields plus one "
                "global integer translation per frame; no low-rank factors, non-rigid warps, "
                "boundary atoms, scorer result, or trained generator is adjudicated"
            ),
            "build_trigger": bool(admitting),
            "scorer_fire_order": None,
            "pointer_moved": False,
        },
        "boundary_obligations": rows[0]["boundary_obligations"],
        "provenance": {
            "runner": runner,
            "git_head_before_commit": git_head(),
            "argv": sys.argv,
        },
        "constraints": rows[0]["constraints"],
    }
    atomic_json_once(result_path, result)
    atomic_json_once(
        checkpoint_path,
        {
            "schema": SCHEMA,
            "result": file_fact(result_path),
            "runner": runner,
            "source_field": source,
        },
    )
    entries = inventory(output, manifest_path)
    atomic_json_once(
        manifest_path,
        {
            "schema": MANIFEST_SCHEMA,
            "entry_count": len(entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "entries": entries,
        },
    )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--stage", choices=("presence", "gop", "finalize"), required=True)
    parser.add_argument("--gop-length", type=int, choices=GOP_LENGTHS)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.resolve() != OUTPUT.resolve():
        raise GF3Error(f"output must be exactly {OUTPUT}")
    if args.resume_from.resolve() != args.output.resolve():
        raise GF3Error("--resume-from must equal --output")
    if (args.stage == "gop") != (args.gop_length is not None):
        raise GF3Error("--gop-length is required exactly for --stage gop")

    source = validate_source(args.field)
    runner = file_fact(Path(__file__).resolve())
    stage_label = f"gop_L{args.gop_length:03d}" if args.stage == "gop" else args.stage
    storage = storage_preflight(args.output, label=stage_label)
    lineage = up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
    target = np.memmap(args.field, mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
    presence = presence_stage(args.output, target, source, runner)
    if args.stage == "presence":
        payload: object = {
            "schema": SCHEMA,
            "stage": "presence",
            "presence_shape": list(presence.shape),
            "lineage_gate": lineage,
        }
    elif args.stage == "gop":
        payload = run_gop(
            output=args.output,
            target=target,
            presence=presence,
            source=source,
            runner=runner,
            gop_length=args.gop_length,
        )
    else:
        payload = finalize(args.output, source, runner, lineage, storage)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
