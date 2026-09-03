#!/usr/bin/env python3
"""Measure the GF2 static/dynamic generator ceiling on the exact AFR1 field.

This runner is deliberately ceiling-first.  It gives the proposed static term
more freedom than a shipped GF1-family generator: a full 512x384 categorical
field, plus independently fitted per-pair integer translations.  The final
field is the per-site modal class after alignment.  If even this favorable
object leaves more errors than the entire 71,404.5-byte packet could repair at
the charter's 0.2909 B/site comparison rate, the charter requires
CEILING-REFUSED and forbids the three packet-size fits.

Every materialized field, residual traversal, and coder output is retained.
The process is CPU-only, scorer-free, resumable at stage boundaries, and does
not touch the parallel gc1 arm or either protected submission tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import platform
import resource
import shutil
import struct
import subprocess
import sys
import time
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import brotli
import numpy as np

SCHEMA: Final = "ddm_gf2_static_dynamic_generator_form.v2"
AXIS: Final = "[macOS-CPU scorer-free exact field measurement, n600]"
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
CLASSES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
NUM_CLASSES: Final = len(CLASSES)
FIELD_SHAPE: Final = (N_PAIRS, HEIGHT, WIDTH)
FIELD_BYTES: Final = int(np.prod(FIELD_SHAPE))
FIELD_SHA256: Final = (
    "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
)
DEFAULT_FIELD: Final = Path(
    "/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/retained/fields/"
    "sfp1_null_empty.u8"
)
OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/converged_v2"
)

PACKET_CAP_BYTES: Final = 71_404.5
PACKET_CAP_HALF_BYTE_NUMERATOR: Final = 142_809
MISMATCH_TARGET: Final = 46_804
GENERIC_BYTES_PER_SITE: Final = 0.2909
GENERIC_BYTES_PER_SITE_NUMERATOR: Final = 2_909
GENERIC_BYTES_PER_SITE_DENOMINATOR: Final = 10_000
REPLACEMENT_CAP_BYTES: Final = 85_020
SEARCH_RADIUS: Final = 12
MAX_ALIGNMENT_ITERATIONS: Final = 20
FILL_CLASS: Final = 2
MINIMUM_FREE_BYTES: Final = 1 << 30

STATIC_HEADER: Final = struct.Struct("<4sBHHI")
STATIC_MAGIC: Final = b"GF2S"
OFFSETS_HEADER: Final = struct.Struct("<4sBHH")
OFFSETS_MAGIC: Final = b"GF2D"


class GF2Error(RuntimeError):
    """A source, ceiling, coder, or custody invariant failed."""


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
    if not isinstance(fact, dict) or "path" not in fact:
        return False
    path = Path(str(fact["path"]))
    return bool(
        path.is_file()
        and path.stat().st_size == int(fact.get("bytes", -1))
        and sha256_file(path) == fact.get("sha256")
    )


def atomic_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return file_fact(path)


def atomic_array(path: Path, array: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    np.ascontiguousarray(array).tofile(temporary)
    os.replace(temporary, path)
    return file_fact(path)


def atomic_json(path: Path, value: object) -> dict[str, object]:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return atomic_bytes(path, payload)


def peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(observed if sys.platform == "darwin" else observed * 1024)


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def storage_preflight(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(output)
    free = int(stat.f_bavail * stat.f_frsize)
    result = {
        "path": str(output),
        "observed_free_bytes": free,
        "required_free_bytes": MINIMUM_FREE_BYTES,
        "status": "PASS" if free >= MINIMUM_FREE_BYTES else "REFUSED",
        "retention_policy": "certify-or-block; no generated payload is deleted",
    }
    atomic_json(output / "stage_checkpoints/00_storage_preflight.json", result)
    if free < MINIMUM_FREE_BYTES:
        raise GF2Error(f"Vertigo storage preflight refused: {result}")
    return result


def validate_source(path: Path) -> dict[str, object]:
    fact = file_fact(path)
    if fact["bytes"] != FIELD_BYTES or fact["sha256"] != FIELD_SHA256:
        raise GF2Error(f"AFR1/JBP1 field identity mismatch: {fact}")
    return fact


def retain_source_mirror(source: Path, output: Path) -> dict[str, object]:
    mirror = output / "retained/source_afr1_jbp1_field.u8"
    if mirror.is_file():
        fact = file_fact(mirror)
    else:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        temporary = mirror.with_name(f".{mirror.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, mirror)
        fact = file_fact(mirror)
    if fact["bytes"] != FIELD_BYTES or fact["sha256"] != FIELD_SHA256:
        raise GF2Error(f"retained source mirror identity mismatch: {fact}")
    return fact


def modal_field(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.empty((NUM_CLASSES, HEIGHT, WIDTH), dtype=np.uint16)
    for label in range(NUM_CLASSES):
        counts[label] = np.count_nonzero(target == label, axis=0)
    return np.argmax(counts, axis=0).astype(np.uint8), counts


def translation_slices(
    dy: int, dx: int
) -> tuple[slice, slice, slice, slice]:
    """Return template-y/x and target-y/x overlap slices for a translation."""

    template_y = slice(max(0, -dy), min(HEIGHT, HEIGHT - dy))
    template_x = slice(max(0, -dx), min(WIDTH, WIDTH - dx))
    target_y = slice(max(0, dy), min(HEIGHT, HEIGHT + dy))
    target_x = slice(max(0, dx), min(WIDTH, WIDTH + dx))
    return template_y, template_x, target_y, target_x


def render_translation(template: np.ndarray, dy: int, dx: int) -> np.ndarray:
    rendered = np.full((HEIGHT, WIDTH), FILL_CLASS, dtype=np.uint8)
    sy, sx, ty, tx = translation_slices(dy, dx)
    rendered[ty, tx] = template[sy, sx]
    return rendered


def ordered_translations(radius: int) -> list[tuple[int, int]]:
    values = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    return sorted(values, key=lambda row: (abs(row[0]) + abs(row[1]), abs(row[0]), abs(row[1]), row))


def fit_offsets(
    target: np.ndarray,
    template: np.ndarray,
    *,
    radius: int,
    pair_chunk: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustively fit the best translation per pair in the declared window."""

    best_scores = np.full(N_PAIRS, -1, dtype=np.int32)
    best_offsets = np.zeros((N_PAIRS, 2), dtype=np.int16)
    for dy, dx in ordered_translations(radius):
        rendered = render_translation(template, dy, dx)
        for start in range(0, N_PAIRS, pair_chunk):
            stop = min(start + pair_chunk, N_PAIRS)
            scores = np.count_nonzero(target[start:stop] == rendered, axis=(1, 2))
            improve = scores > best_scores[start:stop]
            if np.any(improve):
                chosen = np.flatnonzero(improve) + start
                best_scores[chosen] = scores[improve]
                best_offsets[chosen, 0] = dy
                best_offsets[chosen, 1] = dx
    return best_offsets, best_scores


def modal_after_alignment(target: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    counts = np.zeros((NUM_CLASSES, HEIGHT, WIDTH), dtype=np.uint16)
    for pair, (dy_raw, dx_raw) in enumerate(offsets.tolist()):
        dy, dx = int(dy_raw), int(dx_raw)
        sy, sx, ty, tx = translation_slices(dy, dx)
        aligned = target[pair, ty, tx]
        for label in range(NUM_CLASSES):
            counts[label, sy, sx] += aligned == label
    return np.argmax(counts, axis=0).astype(np.uint8)


def render_all(template: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    rendered = np.empty(FIELD_SHAPE, dtype=np.uint8)
    for pair, (dy_raw, dx_raw) in enumerate(offsets.tolist()):
        rendered[pair] = render_translation(template, int(dy_raw), int(dx_raw))
    return rendered


def packed_static(template: np.ndarray) -> bytes:
    body = np.ascontiguousarray(template, dtype=np.uint8).tobytes()
    return STATIC_HEADER.pack(STATIC_MAGIC, 1, HEIGHT, WIDTH, len(body)) + body


def unpack_static(payload: bytes) -> np.ndarray:
    if len(payload) < STATIC_HEADER.size:
        raise GF2Error("static packet is truncated")
    magic, version, height, width, body_bytes = STATIC_HEADER.unpack_from(payload)
    body = payload[STATIC_HEADER.size :]
    if (magic, version, height, width, body_bytes) != (
        STATIC_MAGIC,
        1,
        HEIGHT,
        WIDTH,
        len(body),
    ):
        raise GF2Error("static packet header or length differs")
    decoded = np.frombuffer(body, dtype=np.uint8).reshape(HEIGHT, WIDTH).copy()
    if np.any(decoded >= NUM_CLASSES):
        raise GF2Error("static packet carries a class outside [0,4]")
    return decoded


def packed_offsets(offsets: np.ndarray) -> bytes:
    body = np.ascontiguousarray(offsets, dtype="<i2").tobytes()
    return OFFSETS_HEADER.pack(OFFSETS_MAGIC, 1, N_PAIRS, 2) + body


def unpack_offsets(payload: bytes) -> np.ndarray:
    if len(payload) < OFFSETS_HEADER.size:
        raise GF2Error("dynamic-offset packet is truncated")
    magic, version, pairs, dimensions = OFFSETS_HEADER.unpack_from(payload)
    body = payload[OFFSETS_HEADER.size :]
    if (magic, version, pairs, dimensions) != (OFFSETS_MAGIC, 1, N_PAIRS, 2):
        raise GF2Error("dynamic-offset packet header differs")
    expected = N_PAIRS * dimensions * np.dtype("<i2").itemsize
    if len(body) != expected:
        raise GF2Error("dynamic-offset packet length differs")
    decoded = np.frombuffer(body, dtype="<i2").reshape(N_PAIRS, dimensions).copy()
    if np.any(np.abs(decoded.astype(np.int32)) > SEARCH_RADIUS):
        raise GF2Error("dynamic-offset packet exceeds the declared search family")
    return decoded


def coder_race(name: str, raw_path: Path, output: Path) -> dict[str, object]:
    raw = raw_path.read_bytes()
    encoders = {
        "brotli_q11": lambda: brotli.compress(raw, quality=11),
        "zlib_9": lambda: zlib.compress(raw, level=9),
        "lzma2_extreme": lambda: lzma.compress(
            raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME
        ),
    }
    decoders = {
        "brotli_q11": brotli.decompress,
        "zlib_9": zlib.decompress,
        "lzma2_extreme": lzma.decompress,
    }
    rows: dict[str, object] = {}
    for coder, encode in encoders.items():
        payload = encode()
        repeat = encode()
        if payload != repeat:
            raise GF2Error(f"{name}/{coder} is not deterministic")
        if decoders[coder](payload) != raw:
            raise GF2Error(f"{name}/{coder} parse-back differs from raw payload")
        suffix = {"brotli_q11": "br", "zlib_9": "zlib", "lzma2_extreme": "xz"}[coder]
        primary = atomic_bytes(output / f"coded/{name}.{suffix}", payload)
        repeated = atomic_bytes(output / f"coded/{name}.repeat.{suffix}", repeat)
        rows[coder] = {
            "primary": primary,
            "repeat": repeated,
            "byte_identical": True,
            "parseback_exact": True,
        }
    selected = min(rows, key=lambda coder: int(rows[coder]["primary"]["bytes"]))
    return {
        "raw": file_fact(raw_path),
        "coders": rows,
        "selected_coder": selected,
        "selected_bytes": int(rows[selected]["primary"]["bytes"]),
    }


def mismatch_statistics(
    target: np.ndarray, predicted: np.ndarray
) -> dict[str, object]:
    mismatch = target != predicted
    by_target = {
        CLASSES[label]: int(np.count_nonzero(mismatch & (target == label)))
        for label in range(NUM_CLASSES)
    }
    by_static = {
        CLASSES[label]: int(np.count_nonzero(mismatch & (predicted == label)))
        for label in range(NUM_CLASSES)
    }
    joint = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for static_label in range(NUM_CLASSES):
        selected = predicted == static_label
        for target_label in range(NUM_CLASSES):
            joint[static_label, target_label] = np.count_nonzero(
                selected & (target == target_label)
            )
    class_conditional_bits = 0.0
    for row in joint:
        total = int(row.sum())
        nonzero = row[row > 0].astype(np.float64)
        class_conditional_bits += float(np.sum(nonzero * np.log2(total / nonzero)))
    residual = np.where(mismatch, target + 1, 0).astype(np.uint8)
    per_site_counts = np.empty((NUM_CLASSES + 1, HEIGHT, WIDTH), dtype=np.uint16)
    for symbol in range(NUM_CLASSES + 1):
        per_site_counts[symbol] = np.count_nonzero(residual == symbol, axis=0)
    nonzero = per_site_counts[per_site_counts > 0].astype(np.float64)
    site_conditional_bits = float(np.sum(nonzero * np.log2(N_PAIRS / nonzero)))
    del residual, per_site_counts
    total_mismatch = int(np.count_nonzero(mismatch))
    return {
        "total": total_mismatch,
        "fraction": total_mismatch / FIELD_BYTES,
        "by_target_class": by_target,
        "by_static_prediction_class": by_static,
        "joint_static_target_counts": joint.tolist(),
        "exact_empirical_class_conditional_entropy_bits": class_conditional_bits,
        "exact_empirical_class_conditional_entropy_bytes": class_conditional_bits / 8.0,
        "class_conditional_entropy_definition": (
            "sum_s n_s H_2(target_class | rendered_static_class=s); empirical n600"
        ),
        "exact_empirical_per_site_residual_entropy_bits": site_conditional_bits,
        "exact_empirical_per_site_residual_entropy_bytes": site_conditional_bits / 8.0,
        "per_site_residual_entropy_definition": (
            "sum_(y,x) 600*H_2(residual_symbol_t | decoded lattice site y,x); "
            "zero is match and 1..5 are exact target classes"
        ),
    }


def residual_streams(
    target: np.ndarray, predicted: np.ndarray, output: Path
) -> dict[str, object]:
    # Zero means the static prediction is correct; 1..5 carry the exact target.
    residual = np.where(target == predicted, 0, target + 1).astype(np.uint8)
    frame_path = output / "retained/residual.frame_raster.u8"
    pixel_path = output / "retained/residual.pixel_time.u8"
    frame_fact = atomic_array(frame_path, residual)
    pixel_fact = atomic_array(pixel_path, residual.transpose(1, 2, 0))
    reconstructed = np.where(residual == 0, predicted, residual - 1).astype(np.uint8)
    if not np.array_equal(reconstructed, target):
        raise GF2Error("residual receiver reconstruction differs from the exact field")
    reconstruction_fact = atomic_array(
        output / "retained/residual_reconstructed_field.u8", reconstructed
    )
    if reconstruction_fact["sha256"] != FIELD_SHA256:
        raise GF2Error("residual reconstruction has the wrong exact-field SHA-256")
    del reconstructed
    del residual
    return {
        "receiver_reconstruction": {
            "field": reconstruction_fact,
            "matches_source_sha256": True,
        },
        "generic_frame_raster": {
            "payload": frame_fact,
            "coder_race": coder_race("residual.frame_raster", frame_path, output),
        },
        "domain_matched_pixel_time": {
            "payload": pixel_fact,
            "coder_race": coder_race("residual.pixel_time", pixel_path, output),
            "domain_match": (
                "all 600 residual symbols for one lattice site are contiguous before the next site"
            ),
        },
    }


def completed_result(output: Path, source: dict[str, object]) -> dict[str, object] | None:
    path = output / "RESULT.json"
    manifest_path = output / "MANIFEST.json"
    if not path.is_file() or not manifest_path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA or result.get("source_field") != source:
        raise GF2Error("retained RESULT.json does not bind the current exact source")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "ddm_gf2_retained_manifest.v1" or any(
        not fact_matches(fact) for fact in manifest.get("entries", [])
    ):
        raise GF2Error("retained MANIFEST.json is incomplete or stale")
    return result


def inventory(output: Path, manifest_path: Path) -> list[dict[str, object]]:
    return [
        file_fact(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path
    ]


def run(output: Path, field_path: Path) -> dict[str, object]:
    if output.resolve() != OUTPUT.resolve():
        raise GF2Error(f"output must be exactly {OUTPUT}")
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    source = validate_source(field_path)
    runner = file_fact(Path(__file__).resolve())
    prior = completed_result(output, source)
    if prior is not None:
        if prior.get("provenance", {}).get("runner") != runner:
            raise GF2Error("completed result was produced by a different runner")
        return prior
    storage = storage_preflight(output)
    source_mirror = retain_source_mirror(field_path, output)
    source_checkpoint_path = output / "stage_checkpoints/00_source_validated.json"
    if source_checkpoint_path.is_file():
        source_checkpoint = json.loads(source_checkpoint_path.read_text(encoding="utf-8"))
        if (
            source_checkpoint.get("schema") != SCHEMA
            or source_checkpoint.get("source_field") != source
            or source_checkpoint.get("source_mirror") != source_mirror
            or source_checkpoint.get("runner") != runner
        ):
            raise GF2Error("source checkpoint belongs to a different source or runner")
    else:
        atomic_json(
            source_checkpoint_path,
            {
                "schema": SCHEMA,
                "source_field": source,
                "source_mirror": source_mirror,
                "runner": runner,
                "storage": storage,
            },
        )

    target = np.memmap(source_mirror["path"], mode="r", dtype=np.uint8, shape=FIELD_SHAPE)
    initial_checkpoint_path = output / "stage_checkpoints/01_unaligned_modal_complete.json"
    if initial_checkpoint_path.is_file():
        initial_checkpoint = json.loads(initial_checkpoint_path.read_text(encoding="utf-8"))
        if initial_checkpoint.get("schema") != SCHEMA or not fact_matches(
            initial_checkpoint.get("modal_field")
        ):
            raise GF2Error("unaligned-modal checkpoint is incomplete or stale")
        initial_fact = initial_checkpoint["modal_field"]
        initial = np.fromfile(initial_fact["path"], dtype=np.uint8).reshape(HEIGHT, WIDTH)
        initial_mismatches = int(initial_checkpoint["mismatches"])
    else:
        initial, initial_counts = modal_field(target)
        initial_path = output / "stage_checkpoints/01_unaligned_modal.u8"
        initial_fact = atomic_array(initial_path, initial)
        initial_predicted = np.broadcast_to(initial, FIELD_SHAPE)
        initial_mismatches = int(np.count_nonzero(target != initial_predicted))
        atomic_json(
            initial_checkpoint_path,
            {
                "schema": SCHEMA,
                "modal_field": initial_fact,
                "mismatches": initial_mismatches,
                "per_site_vote_totals_min": int(initial_counts.sum(axis=0).min()),
                "per_site_vote_totals_max": int(initial_counts.sum(axis=0).max()),
            },
        )
        del initial_counts, initial_predicted

    template = initial
    alignment_rows: list[dict[str, object]] = []
    offsets = np.zeros((N_PAIRS, 2), dtype=np.int16)
    converged = False
    for iteration in range(1, MAX_ALIGNMENT_ITERATIONS + 1):
        checkpoint_path = (
            output / f"stage_checkpoints/02_align_iter_{iteration:02d}_complete.json"
        )
        if checkpoint_path.is_file():
            row = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            required_facts = (row.get("modal_field"), row.get("offsets"), row.get("decoded_field"))
            if any(not fact_matches(fact) for fact in required_facts):
                raise GF2Error(f"alignment checkpoint {iteration} is incomplete or stale")
            alignment_rows.append(row)
            template = np.fromfile(row["modal_field"]["path"], dtype=np.uint8).reshape(
                HEIGHT, WIDTH
            )
            offsets = np.fromfile(row["offsets"]["path"], dtype="<i2").reshape(
                N_PAIRS, 2
            )
            if bool(row["template_stable"]):
                converged = True
                break
            continue
        offsets, fit_scores = fit_offsets(
            target, template, radius=SEARCH_RADIUS
        )
        updated = modal_after_alignment(target, offsets)
        predicted = render_all(updated, offsets)
        mismatches = int(np.count_nonzero(target != predicted))
        template_stable = bool(np.array_equal(template, updated))
        boundary_hits = int(
            np.count_nonzero(np.max(np.abs(offsets.astype(np.int32)), axis=1) == SEARCH_RADIUS)
        )
        template_fact = atomic_array(
            output / f"stage_checkpoints/02_align_iter_{iteration:02d}_modal.u8",
            updated,
        )
        offsets_fact = atomic_array(
            output / f"stage_checkpoints/02_align_iter_{iteration:02d}_offsets.i16le",
            offsets.astype("<i2"),
        )
        decode_fact = atomic_array(
            output / f"stage_checkpoints/02_align_iter_{iteration:02d}_decode.u8",
            predicted,
        )
        row = {
            "iteration": iteration,
            "search_family": f"integer translations dy,dx in [-{SEARCH_RADIUS},+{SEARCH_RADIUS}]",
            "tie_break": "minimum L1 norm, then abs(dy), abs(dx), then lexical",
            "template_stable": template_stable,
            "search_boundary_hits": boundary_hits,
            "fit_score_min": int(fit_scores.min()),
            "fit_score_max": int(fit_scores.max()),
            "mismatches_after_modal_update": mismatches,
            "modal_field": template_fact,
            "offsets": offsets_fact,
            "decoded_field": decode_fact,
        }
        alignment_rows.append(row)
        atomic_json(checkpoint_path, row)
        template = updated
        del predicted
        if template_stable:
            converged = True
            break

    # Refit once against the final modal field so each reported shift is the
    # exact best member of the declared rigid family for the retained S.
    offsets, fit_scores = fit_offsets(target, template, radius=SEARCH_RADIUS)
    final_boundary_hits = int(
        np.count_nonzero(np.max(np.abs(offsets.astype(np.int32)), axis=1) == SEARCH_RADIUS)
    )
    static_raw = atomic_bytes(output / "retained/static.packet.raw", packed_static(template))
    offsets_raw = atomic_bytes(output / "retained/dynamic_offsets.packet.raw", packed_offsets(offsets))
    decoded_template = unpack_static(Path(static_raw["path"]).read_bytes())
    decoded_offsets = unpack_offsets(Path(offsets_raw["path"]).read_bytes())
    if not np.array_equal(decoded_template, template) or not np.array_equal(
        decoded_offsets, offsets
    ):
        raise GF2Error("integer packet parse-back differs from fitted S or D")
    predicted = render_all(decoded_template, decoded_offsets)
    static_decode = atomic_array(output / "retained/static_decode.u8", predicted)
    static_decode_repeat_array = render_all(decoded_template, decoded_offsets)
    static_decode_repeat = atomic_array(
        output / "retained/static_decode.repeat.u8", static_decode_repeat_array
    )
    if static_decode["sha256"] != static_decode_repeat["sha256"]:
        raise GF2Error("static decode repeat differs")
    del static_decode_repeat_array

    stats = mismatch_statistics(target, predicted)
    streams = residual_streams(target, predicted, output)
    static_coder = coder_race("static.packet", output / "retained/static.packet.raw", output)
    offsets_coder = coder_race(
        "dynamic_offsets.packet", output / "retained/dynamic_offsets.packet.raw", output
    )

    max_generic_repairs = (
        PACKET_CAP_HALF_BYTE_NUMERATOR * GENERIC_BYTES_PER_SITE_DENOMINATOR
    ) // (2 * GENERIC_BYTES_PER_SITE_NUMERATOR)
    optimistic_ceiling = MISMATCH_TARGET + max_generic_repairs
    domain_bytes = int(
        streams["domain_matched_pixel_time"]["coder_race"]["selected_bytes"]
    )
    generic_coded_bytes = int(
        streams["generic_frame_raster"]["coder_race"]["selected_bytes"]
    )
    static_bytes = int(static_coder["selected_bytes"])
    offset_bytes = int(offsets_coder["selected_bytes"])
    optimistic_section_sum = static_bytes + offset_bytes + domain_bytes
    domain_match_reopens = optimistic_section_sum <= REPLACEMENT_CAP_BYTES
    alignment_authoritative = converged and final_boundary_hits == 0
    refused = (
        int(stats["total"]) > optimistic_ceiling
        and not domain_match_reopens
        and alignment_authoritative
    )
    if not alignment_authoritative:
        disposition = "CEILING-INCONCLUSIVE-ALIGNMENT"
    elif refused:
        disposition = "CEILING-REFUSED"
    else:
        disposition = "CEILING-ADMITTED-BUILD-REQUIRED"
    # A passed ceiling is a loud incomplete state, never a fake FORM-CLOSED.
    completion = (
        "INCOMPLETE_CHARTER_BUILD_REQUIRED"
        if not refused
        else "COMPLETE_BY_CHARTER_CEILING_GATE"
    )

    result: dict[str, object] = {
        "schema": SCHEMA,
        "axis": AXIS,
        "source_field": source,
        "retained_source_mirror": source_mirror,
        "scope_reduction": (
            "full n600 field; static ceiling uses a full categorical lattice rather than a restricted "
            "GF1 SDF, and independently fits one integer translation per pair in the declared window"
        ),
        "mechanism_relaxation": {
            "used": True,
            "reason": (
                "the ceiling grants S a full categorical lattice, which is strictly more flexible "
                "than the charter's GF1-family generated static term; a refusal is therefore favorable"
            ),
        },
        "alignment": {
            "family": f"per-pair integer translation dy,dx in [-{SEARCH_RADIUS},+{SEARCH_RADIUS}]",
            "global_optimum_claim": False,
            "exact_best_per_pair_for_final_static_within_declared_family": True,
            "coordinate_descent_converged": converged,
            "authoritative_for_ceiling_refusal": alignment_authoritative,
            "iterations": alignment_rows,
            "final_search_boundary_hits": final_boundary_hits,
            "final_fit_score_min": int(fit_scores.min()),
            "final_fit_score_max": int(fit_scores.max()),
        },
        "static_ceiling": {
            "shared_static_field_count": 1,
            "unaligned_mismatches": initial_mismatches,
            "aligned": stats,
            "static_field_raw": static_raw,
            "static_field_coder_race": static_coder,
            "dynamic_offsets_raw": offsets_raw,
            "dynamic_offsets_coder_race": offsets_coder,
            "decoded_field": static_decode,
            "decoded_field_repeat": static_decode_repeat,
            "decode_repeat_byte_identical": True,
            "integer_packet_parseback_exact": True,
        },
        "residual": streams,
        "closed_form_gate": {
            "packet_cap_bytes": PACKET_CAP_BYTES,
            "mismatch_target": MISMATCH_TARGET,
            "generic_bytes_per_site": GENERIC_BYTES_PER_SITE,
            "max_repairs_if_entire_packet_is_dynamic": max_generic_repairs,
            "largest_static_mismatch_count_plausibly_repairable": optimistic_ceiling,
            "observed_static_mismatches": int(stats["total"]),
            "excess_mismatches_over_optimistic_ceiling": int(stats["total"]) - optimistic_ceiling,
            "domain_matched_optimistic_section_sum_bytes": optimistic_section_sum,
            "domain_matched_reopens_ceiling": domain_match_reopens,
            "favorable_omissions": (
                "charges zero bytes for the static field and rigid offsets before assigning the entire "
                "packet cap to repairs; the domain-matched section sum also omits coder tags and container "
                "framing"
            ),
            "passes": None if not alignment_authoritative else not refused,
        },
        "static_dynamic_mismatch_split": {
            "mismatches_left_by_static_plus_rigid_geometry": int(stats["total"]),
            "mismatches_repaired_by_fitted_sparse_dynamic_events": 0,
            "dynamic_repairs_needed_to_reach_target": max(
                0, int(stats["total"]) - MISMATCH_TARGET
            ),
            "reason_dynamic_events_not_fit": (
                "forbidden by the charter's ceiling gate"
                if refused
                else "ceiling not refused; build remains required"
            ),
        },
        "byte_split": {
            "static_selected_coded_bytes": static_bytes,
            "dynamic_geometry_selected_coded_bytes": offset_bytes,
            "domain_matched_residual_selected_coded_bytes": domain_bytes,
            "generic_frame_raster_residual_selected_coded_bytes": generic_coded_bytes,
            "static_plus_dynamic_geometry_bytes": static_bytes + offset_bytes,
            "optimistic_packet_plus_domain_matched_residual_bytes": optimistic_section_sum,
            "replacement_cap_bytes": REPLACEMENT_CAP_BYTES,
        },
        "packet_curve": [],
        "packet_curve_omission_reason": (
            "charter says refuse at the static ceiling when it exceeds the optimistic dynamic-repair "
            "capacity; no fitted packet point is authorized"
            if refused
            else (
                "alignment ceiling is inconclusive; do not fit packet points until it is closed"
                if not alignment_authoritative
                else "ceiling passed; three fitted packet points remain required"
            )
        ),
        "decision": {
            "disposition": disposition,
            "completion": completion,
            "verdict_scope": (
                "FORMULATION: one full-lattice shared static field plus per-pair integer translations, "
                "under the charter's generic 0.2909 B/site optimistic repair arithmetic"
            ),
            "candidate": False,
            "scorer_fire_order": None,
            "scorer_invocations": 0,
            "metal_invocations": 0,
            "modal_invocations": 0,
            "frontier_moved": False,
        },
        "resource": {
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_limit_bytes": 20_000_000_000,
            "peak_rss_pass": peak_rss_bytes() <= 20_000_000_000,
            "elapsed_seconds": time.time() - started,
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "provenance": {
            "runner": runner,
            "git_head": git_head(),
        },
        "command": [
            ".venv/bin/python",
            "experiments/ddm_gf2_static_dynamic_generator_form.py",
            "--field",
            str(field_path),
            "--output",
            str(output),
            "--resume-from",
            str(output),
        ],
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    if not result["resource"]["peak_rss_pass"]:
        raise GF2Error(f"peak RSS exceeded 20 GB: {result['resource']}")
    atomic_json(output / "RESULT.json", result)
    manifest_path = output / "MANIFEST.json"
    rows = inventory(output, manifest_path)
    manifest = {
        "schema": "ddm_gf2_retained_manifest.v1",
        "entries": rows,
        "entry_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "self_excluded": str(manifest_path),
        "cleanup": "none; every payload retained on Vertigo",
    }
    atomic_json(manifest_path, manifest)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    root.add_argument("--output", type=Path, default=OUTPUT)
    root.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="must name the exact output root; completed stage payloads are reused",
    )
    return root


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.resume_from.resolve() != args.output.resolve():
        raise GF2Error("--resume-from must equal --output")
    result = run(args.output, args.field)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "axis": result["axis"],
                "disposition": result["decision"]["disposition"],
                "mismatches": result["static_ceiling"]["aligned"]["total"],
                "peak_rss_bytes": result["resource"]["peak_rss_bytes"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
