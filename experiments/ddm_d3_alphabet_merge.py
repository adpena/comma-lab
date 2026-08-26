#!/usr/bin/env python3
"""D3: true four-symbol Lane->Road quotient over the shipped GB1 entropy path.

This experiment answers one narrow question with retained bytes: how much does the
GB1 token stream cost when Lane is removed from the *coded alphabet*, rather than
merely relabelled while the coder still spends a fifth frequency slot?

The shipped five-output HPAC and F26 corrector remain the causal probability source.
At the entropy interface, Road and Lane probability are pooled and the four live
canonical symbols ``(Road, Undrivable, Movable, MyCar) == (0, 2, 3, 4)`` are mapped
to dense RC64 symbols ``(0, 1, 2, 3)``.  The receiver maps them back before every
HPAC/corrector feedback edge.  Thus this is a real four-symbol stream, but it is NOT
claimed to be a trained four-output HPAC model.  Keeping that distinction explicit
prevents a rate win from becoming a fake model-refit claim.

Stages are crash-resumable and retain every payload they materialize:

``prepare``
    Pin the exact cc10a7b0 field, retain the Lane->Road field, dense quotient field,
    and exact Lane mask.  Brotli-q11 the exact mask as the crop/fallback carriage
    point; the uncompressed and compressed payloads are both retained.
``carriers``
    Retain a real-coded race of lossless and block-raster Lane masks, parse each
    Brotli payload back, and select only among candidates whose counted bytes fit
    inside the measured four-symbol rate credit.
``encode``
    Re-encode through the shipped GB1 model, group plan, residual table, and F26
    adaptive state using a genuinely four-symbol RC64 build.  Retain stream, bit
    ledger, candidate container, compiler sources, and periodic full-state saves.
``decode``
    Independently run the symmetric receiver and require exact equality with the
    retained merged field.  Retain its decoded field and per-stage checkpoints.
``render``
    Parse the selected counted carrier, Road-gate its Lane paint, and run the
    retained GB1 renderer over the resulting token field.
``score``
    Measure that exact retained raw at n600 with the pinned CPU scorer and DALI
    ground-truth tables.  This is advisory route triage, never score authority.

AXIS: [macOS-CPU advisory / scorer-free exact rate and receiver measurement].
No output of this module is a contest score.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jg2_tail_reencode as jg2

STORE = Path("/Volumes/APDataStore/pact/ddm_d3_alphabet_merge")
RUNTIME = Path("/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/runtime_fire_v1")
BASE_ARCHIVE = RUNTIME / "archive.zip"
BASE_ARCHIVE_BYTES = 180_215
BASE_ARCHIVE_SHA256 = "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4"
SOURCE_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/"
    "retained/fields/decoded_tokens_instrumented.u8"
)
SOURCE_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"

N, H, W = 600, 384, 512
PLANE = H * W
FIELD_BYTES = N * PLANE
CANONICAL_ALPHABET = 5
QUOTIENT_ALPHABET = 4
LANE = 1
ROAD = 0
LIVE_CANONICAL = np.array([0, 2, 3, 4], dtype=np.uint8)
CANONICAL_TO_DENSE = np.array([0, 255, 1, 2, 3], dtype=np.uint8)
MINIMUM_FREE_BYTES = 2 << 30
CHECKPOINT_SCHEMA = "ddm_d3_alphabet_merge.v1"
CHECKPOINT_KEYS = frozenset(
    {"schema", "frame", "code_bits", "per_frame", "previous"}
)
DECODE_CHECKPOINT_KEYS = frozenset({"schema", "frame", "previous", "decoder"})
AXIS = "[macOS-CPU advisory / scorer-free exact rate and receiver measurement]"


class D3Error(RuntimeError):
    """A custody, causality, or receiver proof gate refused."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def progress(**record: Any) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def verify_inputs(store: Path) -> dict[str, Any]:
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise D3Error(
            f"storage preflight failed at {store}: {free} B free < {MINIMUM_FREE_BYTES} B"
        )
    if BASE_ARCHIVE.stat().st_size != BASE_ARCHIVE_BYTES:
        raise D3Error(f"GB1 archive size drifted: {BASE_ARCHIVE}")
    if sha256_file(BASE_ARCHIVE) != BASE_ARCHIVE_SHA256:
        raise D3Error(f"GB1 archive sha drifted: {BASE_ARCHIVE}")
    if SOURCE_FIELD.stat().st_size != FIELD_BYTES:
        raise D3Error(f"source field size drifted: {SOURCE_FIELD}")
    if sha256_file(SOURCE_FIELD) != SOURCE_FIELD_SHA256:
        raise D3Error(f"source field sha drifted: {SOURCE_FIELD}")
    return {
        "storage": {
            "path": str(store),
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "observed_free_bytes": free,
            "status": "PASS",
        },
        "base_archive": file_fact(BASE_ARCHIVE),
        "source_field": file_fact(SOURCE_FIELD),
    }


def retained_paths(store: Path) -> dict[str, Path]:
    root = store / "retained"
    return {
        "merged": root / "fields/tokens_lane_to_road_canonical.u8",
        "dense": root / "fields/tokens_lane_to_road_dense4.u8",
        "class1_mask": root / "carriers/lane_mask_exact.packbits",
        "class1_mask_q11": root / "carriers/lane_mask_exact.packbits.br",
        "manifest": root / "carriers/lane_mask_exact.json",
    }


def stage_prepare(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    custody = verify_inputs(store)
    paths = retained_paths(store)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    merged_tmp = paths["merged"].with_suffix(".u8.partial")
    dense_tmp = paths["dense"].with_suffix(".u8.partial")
    mask_tmp = paths["class1_mask"].with_suffix(".packbits.partial")
    source = np.memmap(SOURCE_FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    lane_pixels = 0
    if not (paths["merged"].is_file() and paths["dense"].is_file() and paths["class1_mask"].is_file()):
        with merged_tmp.open("wb") as merged_out, dense_tmp.open("wb") as dense_out, mask_tmp.open("wb") as mask_out:
            for frame in range(N):
                plane = np.asarray(source[frame], dtype=np.uint8)
                if plane.max() >= CANONICAL_ALPHABET:
                    raise D3Error(f"frame {frame} contains a symbol outside 0..4")
                lane = plane == LANE
                lane_pixels += int(lane.sum())
                merged = plane.copy()
                merged[lane] = ROAD
                dense = CANONICAL_TO_DENSE[merged]
                if np.any(dense == 255):
                    raise D3Error(f"frame {frame} retains Lane after the quotient")
                merged_out.write(merged.tobytes(order="C"))
                dense_out.write(dense.tobytes(order="C"))
                mask_out.write(np.packbits(lane.reshape(-1), bitorder="little").tobytes())
        os.replace(merged_tmp, paths["merged"])
        os.replace(dense_tmp, paths["dense"])
        os.replace(mask_tmp, paths["class1_mask"])
    else:
        lane_pixels = int((source == LANE).sum())

    for name in ("merged", "dense"):
        if paths[name].stat().st_size != FIELD_BYTES:
            raise D3Error(f"retained {name} field has the wrong size")
    expected_mask_bytes = (FIELD_BYTES + 7) // 8
    if paths["class1_mask"].stat().st_size != expected_mask_bytes:
        raise D3Error("retained exact Lane mask has the wrong size")

    command = [
        "/opt/homebrew/bin/brotli", "-f", "-q", "11",
        "-o", str(paths["class1_mask_q11"]), str(paths["class1_mask"]),
    ]
    subprocess.run(command, check=True)
    manifest = {
        "schema": "ddm_d3_lane_mask_exact.v1",
        "shape": [N, H, W],
        "bit_order": "little",
        "frame_order": "pair-major raster",
        "meaning": "1 restores canonical Lane class 1 over quotient Road class 0",
        "class1_pixels": lane_pixels,
        "payload_raw": file_fact(paths["class1_mask"]),
        "payload_brotli_q11": file_fact(paths["class1_mask_q11"]),
        "coder_argv": command,
        "lossless": True,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(paths["manifest"], manifest)
    result = {
        "schema": "ddm_d3_prepare.v1",
        "complete": True,
        "custody": custody,
        "merged_field": file_fact(paths["merged"]),
        "dense4_field": file_fact(paths["dense"]),
        "class1_carrier_exact": manifest,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(store / "PREPARE_RESULT.json", result)
    return result


def exact_lane_mask(store: Path) -> np.ndarray:
    path = retained_paths(store)["class1_mask"]
    packed = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    mask = np.unpackbits(packed, bitorder="little")[:FIELD_BYTES]
    return mask.reshape(N, H, W).astype(bool, copy=False)


def block_candidate(mask: np.ndarray, scale: int, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    padded_h = ((H + scale - 1) // scale) * scale
    padded_w = ((W + scale - 1) // scale) * scale
    padded = np.zeros((N, padded_h, padded_w), dtype=bool)
    padded[:, :H, :W] = mask
    counts = padded.reshape(
        N, padded_h // scale, scale, padded_w // scale, scale
    ).sum(axis=(2, 4))
    coarse = counts >= threshold
    decoded = np.repeat(np.repeat(coarse, scale, axis=1), scale, axis=2)[:, :H, :W]
    return coarse, decoded


def mask_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    true_positive = int(np.logical_and(reference, candidate).sum())
    false_positive = int(np.logical_and(~reference, candidate).sum())
    false_negative = int(np.logical_and(reference, ~candidate).sum())
    union = true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "hamming": false_positive + false_negative,
        "iou": true_positive / union if union else 1.0,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 1.0,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 1.0,
    }


def stage_carriers(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    verify_inputs(store)
    mask = exact_lane_mask(store)
    base = np.memmap(retained_paths(store)["merged"], dtype=np.uint8, mode="r", shape=(N, H, W))
    road_support = np.asarray(base) == ROAD
    root = store / "retained/carrier_race"
    root.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, int, int]] = [
        ("lossless_xor", 0, 0),
        ("lossless_pixel_major", -1, 0),
    ]
    specs.extend((f"block_s2_t{threshold}", 2, threshold) for threshold in range(1, 5))
    specs.extend((f"block_s3_t{threshold}", 3, threshold) for threshold in (1, 2, 3, 5, 7, 9))
    specs.extend((f"block_s4_t{threshold}", 4, threshold) for threshold in (1, 2, 4, 6, 8, 10, 12, 16))
    rows: list[dict[str, Any]] = []
    for candidate_id, scale, threshold in specs:
        receipt_path = root / f"{candidate_id}.json"
        if args.resume and receipt_path.is_file():
            rows.append(json.loads(receipt_path.read_text()))
            continue
        if scale == 0:
            transformed = np.empty_like(mask)
            transformed[0] = mask[0]
            transformed[1:] = np.logical_xor(mask[1:], mask[:-1])
            wire_array = transformed
            decoded = np.logical_xor.accumulate(transformed, axis=0)
            transform = "first frame then temporal XOR"
            header = bytes((1, 0))
        elif scale == -1:
            wire_array = mask.reshape(N, PLANE).T
            decoded = wire_array.T.reshape(N, H, W)
            transform = "pixel-major temporal transpose"
            header = bytes((2, 0))
        else:
            wire_array, decoded = block_candidate(mask, scale, threshold)
            transform = f"{scale}x{scale} occupancy threshold {threshold}, nearest repeat decode"
            header = bytes((scale, threshold))
        wire = np.packbits(wire_array.reshape(-1), bitorder="little").tobytes()
        wire_path = root / f"{candidate_id}.wire.packbits"
        compressed_path = root / f"{candidate_id}.wire.packbits.br"
        counted_path = root / f"{candidate_id}.carrier"
        decoded_path = root / f"{candidate_id}.decoded.packbits"
        atomic_bytes(wire_path, wire)
        command = [
            "/opt/homebrew/bin/brotli", "-f", "-q", "11",
            "-o", str(compressed_path), str(wire_path),
        ]
        subprocess.run(command, check=True)
        decompressed = subprocess.run(
            ["/opt/homebrew/bin/brotli", "-d", "-c", str(compressed_path)],
            check=True,
            capture_output=True,
        ).stdout
        if decompressed != wire:
            raise D3Error(f"{candidate_id}: Brotli parse-back changed the transform wire")
        atomic_bytes(counted_path, header + compressed_path.read_bytes())
        decoded_wire = np.packbits(decoded.reshape(-1), bitorder="little").tobytes()
        atomic_bytes(decoded_path, decoded_wire)
        row = {
            "candidate_id": candidate_id,
            "transform": transform,
            "config_header_bytes": len(header),
            "wire": file_fact(wire_path),
            "brotli_q11": file_fact(compressed_path),
            "counted_carrier": file_fact(counted_path),
            "decoded_mask": file_fact(decoded_path),
            "metrics_vs_exact_lane": mask_metrics(mask, decoded),
            "metrics_after_receiver_road_gate": mask_metrics(mask, np.logical_and(decoded, road_support)),
            "parse_back_wire_identical": True,
            "composed_archive_bytes_projection": 116_287 + counted_path.stat().st_size,
            "composed_archive_delta_vs_gb1_projection": (
                116_287 + counted_path.stat().st_size - BASE_ARCHIVE_BYTES
            ),
            "axis": AXIS,
            "score_claim": False,
        }
        atomic_json(receipt_path, row)
        rows.append(row)
        atomic_json(
            store / "CARRIER_RACE_CHECKPOINT.json",
            {"schema": "ddm_d3_carrier_race_checkpoint.v1", "completed": rows},
        )
        progress(
            stage="carriers", event="candidate",
            candidate_id=candidate_id,
            bytes=counted_path.stat().st_size,
            iou=row["metrics_vs_exact_lane"]["iou"],
        )
    within_rate = [
        row for row in rows if row["composed_archive_delta_vs_gb1_projection"] < 0
    ]
    selected = max(
        within_rate,
        key=lambda row: (
            row["metrics_after_receiver_road_gate"]["iou"],
            -row["counted_carrier"]["bytes"],
        ),
        default=None,
    )
    result = {
        "schema": "ddm_d3_carrier_race.v1",
        "complete": True,
        "candidate_count": len(rows),
        "rows": rows,
        "selection_rule": "highest exact-mask IoU after receiver-derived Road-support gate among carriers keeping projected archive below GB1",
        "selected_for_real_renderer": selected,
        "projection_warning": "archive arithmetic only; selected carrier still needs an actual packed container and scorer",
        "analytic_cb1_transplant": {
            "disposition": "REFUSED_HIDDEN_PARENT_DEPENDENCY",
            "reason": "the 2052-byte program and knots mutate an inherited video-derived Lane chart absent from GB1; counting only the marginal would hide required instance bytes",
        },
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(store / "CARRIER_RACE_RESULT.json", result)
    return result


def load_packed_mask(path: Path) -> np.ndarray:
    packed = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little")[:FIELD_BYTES].reshape(N, H, W).astype(bool, copy=False)


def decode_counted_carrier(selected: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse the exact counted carrier bytes into the mask used by the renderer."""

    candidate_id = str(selected["candidate_id"])
    counted_path = Path(selected["counted_carrier"]["path"])
    compressed_path = Path(selected["brotli_q11"]["path"])
    if file_fact(counted_path) != selected["counted_carrier"]:
        raise D3Error(f"{candidate_id}: counted carrier drifted before render")
    if file_fact(compressed_path) != selected["brotli_q11"]:
        raise D3Error(f"{candidate_id}: compressed carrier drifted before render")
    counted = counted_path.read_bytes()
    if len(counted) < 2 or counted[2:] != compressed_path.read_bytes():
        raise D3Error(f"{candidate_id}: counted carrier is not header || retained Brotli payload")
    scale, threshold = counted[:2]
    wire = subprocess.run(
        ["/opt/homebrew/bin/brotli", "-d", "-c", str(compressed_path)],
        check=True,
        capture_output=True,
    ).stdout
    if candidate_id == "lossless_xor":
        if (scale, threshold) != (1, 0):
            raise D3Error(f"{candidate_id}: header drifted")
        bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8), bitorder="little")[:FIELD_BYTES]
        transformed = bits.reshape(N, H, W).astype(bool, copy=False)
        decoded = np.logical_xor.accumulate(transformed, axis=0)
    elif candidate_id == "lossless_pixel_major":
        if (scale, threshold) != (2, 0):
            raise D3Error(f"{candidate_id}: header drifted")
        bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8), bitorder="little")[:FIELD_BYTES]
        decoded = bits.reshape(PLANE, N).T.reshape(N, H, W).astype(bool, copy=False)
    else:
        expected_prefix = f"block_s{scale}_t{threshold}"
        if candidate_id != expected_prefix or scale < 2:
            raise D3Error(f"{candidate_id}: block-carrier header drifted")
        coarse_h = (H + scale - 1) // scale
        coarse_w = (W + scale - 1) // scale
        coarse_bits = N * coarse_h * coarse_w
        bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8), bitorder="little")
        if bits.size < coarse_bits:
            raise D3Error(f"{candidate_id}: block-carrier wire is truncated")
        coarse = bits[:coarse_bits].reshape(N, coarse_h, coarse_w).astype(bool, copy=False)
        decoded = np.repeat(np.repeat(coarse, scale, axis=1), scale, axis=2)[:, :H, :W]
    decoded_wire = np.packbits(decoded.reshape(-1), bitorder="little").tobytes()
    expected_decoded = Path(selected["decoded_mask"]["path"])
    if file_fact(expected_decoded) != selected["decoded_mask"]:
        raise D3Error(f"{candidate_id}: retained decoded-mask receipt drifted")
    if decoded_wire != expected_decoded.read_bytes():
        raise D3Error(f"{candidate_id}: counted-carrier receiver disagrees with retained mask")
    return decoded, {
        "candidate_id": candidate_id,
        "counted_carrier": file_fact(counted_path),
        "parsed_wire_bytes": len(wire),
        "decoded_mask": file_fact(expected_decoded),
        "parse_back_identical": True,
    }


def stage_render(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    store = Path(args.store)
    env = load_environment(store)
    race = json.loads((store / "CARRIER_RACE_RESULT.json").read_text())
    selected = race.get("selected_for_real_renderer")
    if not isinstance(selected, dict):
        raise D3Error("carrier race selected no in-rate candidate")
    candidate_id = str(selected["candidate_id"])
    decoded_mask, carrier_receiver = decode_counted_carrier(selected)
    root = store / "retained/render" / candidate_id
    submission = root / "submission"
    raw = submission / "inflated/0.raw"
    result_path = root / "RENDER_RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        archive_path = Path(result["archive"]["path"])
        token_path = Path(result["tokens"]["path"])
        if (
            file_fact(raw) != result["raw"]
            or file_fact(archive_path) != result["archive"]
            or file_fact(token_path) != result["tokens"]
        ):
            raise D3Error("retained render result no longer matches its payloads")
        token_field = np.memmap(token_path, dtype=np.uint8, mode="r", shape=(N, H, W))
        merged = np.memmap(retained_paths(store)["merged"], dtype=np.uint8, mode="r", shape=(N, H, W))
        for frame in range(N):
            expected = np.asarray(merged[frame], dtype=np.uint8).copy()
            applied = np.logical_and(decoded_mask[frame], expected == ROAD)
            expected[applied] = LANE
            if not np.array_equal(expected, np.asarray(token_field[frame])):
                raise D3Error(f"counted-carrier parse-back disagrees with rendered token frame {frame}")
        result["carrier_receiver"] = carrier_receiver
        atomic_json(result_path, result)
        return result
    free = shutil.disk_usage(store).free
    required = 5 << 30
    if free < required:
        raise D3Error(f"render storage preflight failed: {free} B free < {required} B")

    merged = np.memmap(retained_paths(store)["merged"], dtype=np.uint8, mode="r", shape=(N, H, W))
    painted = np.array(merged, dtype=np.uint8)
    applied = np.logical_and(decoded_mask, painted == ROAD)
    painted[applied] = LANE
    token_path = root / "payloads/tokens_road_gated.u8"
    atomic_bytes(token_path, painted.tobytes(order="C"))
    token_record = file_fact(token_path)

    carrier = Path(selected["counted_carrier"]["path"]).read_bytes()
    stream = Path(json.loads((store / "ENCODE_RESULT.json").read_text())["stream"]["path"]).read_bytes()
    sections = dict(env["sections"])
    d3_tail = (
        sections["tail"][:jg2.RESIDUAL_COMPACT_BYTES]
        + b"D3Q1"
        + struct.pack("<I", len(carrier))
        + carrier
        + stream
    )
    sections["tail"] = d3_tail
    archive_path = submission / "archive.zip"
    jg2.pack_archive(jg2.join_member(sections), archive_path)

    from runtime import ddm_wc1_advisory_runtime as wc1  # type: ignore[import-not-found]
    from runtime.carrier_repack import (  # type: ignore[import-not-found]
        materialize_cpr1,
        split_frame0_selector_carrier,
    )
    from runtime.compensation_overlay import apply_compensation_overlay  # type: ignore[import-not-found]
    from runtime.entropy.renderer_weight_codec import (  # type: ignore[import-not-found]
        decode_wans1,
    )

    renderer = env["renderer"]
    parts = env["parts"]
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob)
    canonical_carrier = materialize_cpr1(carrier_blob, renderer)
    semantic_pose = struct.pack("<II", 40_252, len(canonical_carrier)) + bytes(40_252) + canonical_carrier
    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
    compensation = None
    if parts.compensation_blob is not None:
        basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
        _, _, coefficient_scales, encoded = renderer.decode_compact_carrier(
            canonical_carrier,
            basis_count=basis_count,
            frames=renderer.N,
            dimensions=renderer.CARRIER_DIM,
        )
        delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
        base_codes = np.cumsum(delta, axis=0) & 0xFFF
        base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes).astype(np.int32)
        candidate_codes = apply_compensation_overlay(base_codes, parts.compensation_blob)
        coefficients = torch.from_numpy(candidate_codes).float() * torch.from_numpy(coefficient_scales)[None]
        compensation = {
            "payload_bytes": len(parts.compensation_blob),
            "payload_sha256": hashlib.sha256(parts.compensation_blob).hexdigest(),
        }
    semantic = renderer.SemanticTokenRenderer(96)
    tagged = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, semantic.state_dict())
    if tagged is None:
        tagged = {
            record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
            for record in decode_wans1(parts.semantic_blob)
        }
    semantic.load_state_dict(tagged, strict=True)

    render_stage = root / "render_stage.raw"
    parallel = wc1.render_video_parallel(
        semantic=semantic,
        basis=basis,
        coefficients=coefficients,
        token_path=token_path,
        token_sha256=token_record["sha256"],
        renderer_dir=RUNTIME / "cpr1",
        output_path=render_stage,
        progress_dir=root / "render_checkpoints",
        pair_count=N,
        camera_height=int(renderer.CAMERA_H),
        camera_width=int(renderer.CAMERA_W),
        requested_workers="2",
        per_process_threads=4,
        measured_worker_rss_bytes=None,
    )
    selector = None
    if selector_blob is not None:
        from runtime.f26_inflate import _apply_frame0_selector  # type: ignore[import-not-found]

        selector = _apply_frame0_selector(render_stage, renderer, selector_blob, pair_count=N)
    raw.parent.mkdir(parents=True, exist_ok=True)
    os.replace(render_stage, raw)
    result = {
        "schema": "ddm_d3_render.v1",
        "complete": True,
        "candidate_id": candidate_id,
        "storage_preflight": {"required_free_bytes": required, "observed_free_bytes": free},
        "tokens": token_record,
        "carrier_receiver": carrier_receiver,
        "paint": {
            "decoded_mask_pixels": int(decoded_mask.sum()),
            "road_gated_pixels": int(applied.sum()),
            "metrics": mask_metrics(exact_lane_mask(store), applied),
        },
        "archive": file_fact(archive_path),
        "archive_delta_bytes": archive_path.stat().st_size - BASE_ARCHIVE_BYTES,
        "container_schema": "residual96 || D3Q1 || uint32 carrier_bytes || carrier || rc64_alphabet4_stream",
        "raw": file_fact(raw),
        "parallel_render": parallel,
        "selector": selector,
        "compensation": compensation,
        "runtime_status": "research renderer closed; public inflate runtime not yet emitted",
        "axis": "[macOS-CPU advisory / unscored real renderer output]",
        "score_claim": False,
    }
    atomic_json(result_path, result)
    return result


def stage_score(args: argparse.Namespace) -> dict[str, Any]:
    import math

    import torch

    from experiments import ddm_ap1_residue_purchase_scorer as ap1

    store = Path(args.store)
    race = json.loads((store / "CARRIER_RACE_RESULT.json").read_text())
    selected = race.get("selected_for_real_renderer")
    if not isinstance(selected, dict):
        raise D3Error("carrier race selected no candidate for scoring")
    candidate_id = str(selected["candidate_id"])
    render = json.loads(
        (store / f"retained/render/{candidate_id}/RENDER_RESULT.json").read_text()
    )
    raw = Path(render["raw"]["path"])
    archive = Path(render["archive"]["path"])
    if file_fact(raw) != render["raw"] or file_fact(archive) != render["archive"]:
        raise D3Error("rendered candidate drifted before scoring")
    if sha256_file(ap1.GT_SEG) != ap1.GT_SEG_SHA256:
        raise D3Error("pinned DALI SegNet GT table drifted")
    if sha256_file(ap1.GT_POSE) != ap1.GT_POSE_SHA256:
        raise D3Error("pinned DALI PoseNet GT table drifted")
    out = store / "retained/scorer" / candidate_id
    result_path = out / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if file_fact(raw) != result["candidate_raw"]:
            raise D3Error("scorer result points at a changed raw")
        return result

    if str(ap1.ADVISORY_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(ap1.ADVISORY_UPSTREAM))
    from frame_utils import TensorVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.manual_seed(12_341)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    distortion_net = DistortionNet().eval().to("cpu")
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    names = [
        line.strip()
        for line in ap1.VIDEO_NAMES.read_text().splitlines()
        if line.strip()
    ]
    if len(names) != 1:
        raise D3Error("advisory scorer expected one public video")
    dataset = TensorVideoDataset(
        names,
        data_dir=raw.parent,
        batch_size=16,
        device=torch.device("cpu"),
        num_threads=2,
        seed=1234,
    )
    dataset.prepare_data()
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
    gt_seg = np.load(ap1.GT_SEG, allow_pickle=False, mmap_mode="r")
    gt_pose = np.load(ap1.GT_POSE, allow_pickle=False, mmap_mode="r")
    receipts = []
    start = 0
    started = time.perf_counter()
    for chunk_index, (_, _, batch_candidate) in enumerate(loader):
        stop = start + int(batch_candidate.shape[0])
        receipts.append(
            ap1._score_chunk(
                chunk_index=chunk_index,
                start=start,
                stop=stop,
                batch_candidate=batch_candidate,
                distortion_net=distortion_net,
                gt_seg=gt_seg,
                gt_pose=gt_pose,
                out_dir=out,
            )
        )
        progress(
            stage="score", event="chunk", stop=stop,
            elapsed_seconds=time.perf_counter() - started,
        )
        start = stop
    if start != N:
        raise D3Error(f"scorer covered {start} pairs instead of n600")
    summary = ap1._aggregate_chunks(receipts)
    full_seg = np.concatenate(
        [ap1.verify_array_fact(receipt["candidate_argmax"]) for receipt in receipts],
        axis=0,
    )
    full_pose = np.concatenate(
        [ap1.verify_array_fact(receipt["candidate_pose6"]) for receipt in receipts],
        axis=0,
    )
    full_seg_record = ap1.atomic_npy(out / "candidate_argmax_n600.uint8.npy", full_seg)
    full_pose_record = ap1.atomic_npy(out / "candidate_pose6_n600.float32.npy", full_pose)
    score = (
        100.0 * summary["d_seg"]
        + math.sqrt(10.0 * summary["d_pose"])
        + 25.0 * archive.stat().st_size / 37_545_489
    )
    gb1 = {
        "score": 0.14811799921260607,
        "archive_bytes": BASE_ARCHIVE_BYTES,
        "d_seg_report_8dp": 0.00020139,
        "d_pose_report_8dp": 0.00000637,
        "axis": "[contest-CUDA T4 n600]",
    }
    result = {
        "schema": "ddm_d3_dali_score.v1",
        "complete": True,
        "candidate_id": candidate_id,
        "axis": "[macOS-CPU advisory; DALI-GT pinned n600]",
        "promotable": False,
        "score_claim": False,
        "candidate_raw": file_fact(raw),
        "candidate_archive": file_fact(archive),
        "gt_seg": file_fact(ap1.GT_SEG),
        "gt_pose": file_fact(ap1.GT_POSE),
        "segnet_weights": file_fact(Path(segnet_sd_path)),
        "posenet_weights": file_fact(Path(posenet_sd_path)),
        "chunk_receipts": [
            file_fact(out / "chunks" / f"{int(receipt['pair_start']):04d}_{int(receipt['pair_stop_exclusive']) - 1:04d}.json")
            for receipt in receipts
        ],
        "candidate_argmax_n600": full_seg_record,
        "candidate_pose6_n600": full_pose_record,
        "summary": summary,
        "advisory_composed_S": score,
        "gb1_reference": gb1,
        "mixed_axis_delta_warning": "candidate is macOS-CPU model forward while GB1 reference is contest-CUDA; delta is route triage, not a promotable comparison",
        "delta_vs_gb1_reference": {
            "archive_bytes": archive.stat().st_size - gb1["archive_bytes"],
            "d_seg": summary["d_seg"] - gb1["d_seg_report_8dp"],
            "d_pose": summary["d_pose"] - gb1["d_pose_report_8dp"],
            "S": score - gb1["score"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    return result


def load_route_b_four():
    route_b = jg2.load_route_b()
    route_b.ALPHABET = QUOTIENT_ALPHABET
    return route_b


def compile_rc64_four(store: Path, route_b) -> tuple[Path, dict[str, Any]]:
    build = store / "retained/build/rc64_alphabet4"
    build.mkdir(parents=True, exist_ok=True)
    base = jg2.resolve_rc64_base(route_b, build)
    source = base.read_text()
    needle = "#define RC64_ALPHABET 5u"
    if source.count(needle) != 1:
        raise D3Error("pinned RC64 base no longer has exactly one alphabet macro")
    source = source.replace(needle, "#define RC64_ALPHABET 4u")
    generated = build / "rc64_backend_alphabet4.c"
    library = build / "librc64_alphabet4.dylib"
    atomic_bytes(
        generated,
        (source + "\n" + route_b.RC64_CHECKPOINT_EXTENSION).encode("utf-8"),
    )
    command = [
        "/usr/bin/cc", "-O3", "-std=c11", "-shared", "-fPIC",
        "-ffp-contract=off", "-fno-fast-math", str(generated), "-o", str(library),
    ]
    subprocess.run(command, check=True)
    return library, {
        "alphabet": QUOTIENT_ALPHABET,
        "argv": command,
        "base_source": file_fact(base),
        "generated_source": file_fact(generated),
        "library": file_fact(library),
    }


def load_environment(store: Path) -> dict[str, Any]:
    verify_inputs(store)
    prepared = store / "PREPARE_RESULT.json"
    if not prepared.is_file():
        raise D3Error("prepare stage has not completed")
    os.environ.setdefault("CP135_BROTLI_CLI", "/opt/homebrew/bin/brotli")
    route_b = load_route_b_four()
    library, build = compile_rc64_four(store, route_b)
    residual, renderer, renderer_dir = jg2.load_runtime(RUNTIME)
    parts = residual.read_residual_archive(BASE_ARCHIVE)
    sections = jg2.split_member(jg2.read_archive_member(BASE_ARCHIVE))
    if sections["tail"][jg2.RESIDUAL_COMPACT_BYTES:] != parts.token_stream:
        raise D3Error("GB1 member tail disagrees with its runtime parser")
    return {
        "route_b": route_b,
        "library": library,
        "build": build,
        "residual": residual,
        "renderer": renderer,
        "renderer_dir": renderer_dir,
        "parts": parts,
        "sections": sections,
    }


def merged_field(store: Path) -> np.memmap:
    path = retained_paths(store)["merged"]
    if path.stat().st_size != FIELD_BYTES:
        raise D3Error("merged field is absent or truncated")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(N, H, W))


def pool_road_lane(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != CANONICAL_ALPHABET:
        raise D3Error(f"expected a five-class coding row, found {values.shape}")
    pooled = np.column_stack(
        (values[:, ROAD] + values[:, LANE], values[:, 2], values[:, 3], values[:, 4])
    ).astype(np.float32, copy=False)
    if np.any(pooled <= 0.0) or np.any(~np.isfinite(pooled)):
        raise D3Error("pooled coding row is not finite and strictly positive")
    if np.any(np.abs(pooled.astype(np.float64).sum(axis=1) - 1.0) > 2e-5):
        raise D3Error("pooled coding row does not sum to one")
    return np.ascontiguousarray(pooled)


def group_machine(env: dict[str, Any]):
    import torch
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    from runtime.hpac_inference import optimize_sparse_evaluator  # type: ignore[import-not-found]

    residual = env["residual"]
    renderer = env["renderer"]
    parts = env["parts"]
    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(env["renderer_dir"])(model, H, W)
    optimize_sparse_evaluator(sparse)
    corrector = FreeCorrector(PLANE)
    groups = []
    for mask in masks:
        positions = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        groups.append((torch.from_numpy(positions).to(device), positions))
    return torch, device, model, sparse, corrector, groups


def probability_state(env: dict[str, Any], sparse: Any, corrector: Any, parts: Any,
                      current: Any, context: Any, boundary: np.ndarray, group: int,
                      flat_positions: np.ndarray) -> tuple[Any, np.ndarray]:
    selected = sparse.selected_logits(current, context, group)
    base_logits = selected.cpu().numpy()
    predicted = base_logits.argmax(axis=1).astype(np.int64)
    feature = boundary[flat_positions].astype(np.int64) * CANONICAL_ALPHABET + predicted
    corrected = base_logits + parts.table.values[feature]
    probability = env["residual"]._probability_table(
        corrected, env["renderer"].HPAC_LOGIT_PRECISION
    )
    state = corrector.group_state(probability, predicted, flat_positions)
    return state, pool_road_lane(corrector.coding_row(state))


def save_encode_checkpoint(path: Path, encoder_path: Path, route_b: Any, encoder: Any,
                           corrector: Any, cold: Any, frame: int, code_bits: float,
                           per_frame: np.ndarray, previous: np.ndarray) -> None:
    state = jg2.corrector_state(corrector)
    lost = jg2.uncaptured_divergent_state(corrector, cold, set(state))
    if lost:
        raise D3Error(f"checkpoint would lose live corrector state: {lost[:8]}")
    atomic_bytes(encoder_path, encoder.snapshot())
    temporary = path.with_suffix(".partial.npz")
    np.savez(
        temporary,
        schema=np.array([CHECKPOINT_SCHEMA]),
        frame=np.array([frame], dtype=np.int64),
        code_bits=np.array([code_bits], dtype=np.float64),
        per_frame=per_frame,
        previous=previous,
        **state,
    )
    os.replace(temporary, path)


def stage_encode(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    env = load_environment(store)
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]

    target = merged_field(store)
    torch_mod, device, model, sparse, corrector, groups = group_machine(env)
    cold = FreeCorrector(PLANE)
    route_b = env["route_b"]
    work = store / "retained/encode"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "alphabet4.checkpoint.npz"
    encoder_checkpoint = work / "alphabet4.encoder.bin"
    start = 0
    code_bits = 0.0
    per_frame = np.zeros(N, dtype=np.float64)
    previous_seed: np.ndarray | None = None
    if args.resume and checkpoint.is_file() and encoder_checkpoint.is_file():
        blob = np.load(checkpoint, allow_pickle=False)
        schema = str(np.asarray(blob["schema"]).reshape(-1)[0])
        if schema != CHECKPOINT_SCHEMA:
            raise D3Error(f"refusing checkpoint schema {schema!r}")
        start = int(blob["frame"][0])
        code_bits = float(blob["code_bits"][0])
        per_frame = np.asarray(blob["per_frame"], dtype=np.float64).copy()
        previous_seed = np.asarray(blob["previous"], dtype=np.uint8).copy()
        jg2.load_corrector_state(
            corrector, {key: blob[key] for key in blob.files if key not in CHECKPOINT_KEYS}
        )
        encoder = route_b.NativeRc64Encoder(env["library"], encoder_checkpoint.read_bytes())
        progress(stage="encode", event="resumed", frame=start)
    else:
        encoder = route_b.NativeRc64Encoder(env["library"])

    started = time.perf_counter()
    with torch_mod.inference_mode():
        if previous_seed is None:
            previous = torch_mod.zeros((1, H, W), dtype=torch_mod.long, device=device)
        else:
            previous = torch_mod.from_numpy(previous_seed.astype(np.int64)).reshape(1, H, W).to(device)
        for frame in range(start, args.frames):
            index = torch_mod.tensor([frame], dtype=torch_mod.long, device=device)
            current = torch_mod.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
                boundary = env["residual"]._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)
            plane_target = np.asarray(target[frame], dtype=np.uint8).reshape(-1)
            frame_bits = 0.0
            for group, (device_positions, flat_positions) in enumerate(groups):
                state, coding4 = probability_state(
                    env, sparse, corrector, env["parts"], current, context,
                    boundary, group, flat_positions,
                )
                canonical = plane_target[flat_positions].astype(np.int64)
                dense = CANONICAL_TO_DENSE[canonical]
                if np.any(dense == 255):
                    raise D3Error(f"frame {frame} group {group} contains Lane")
                dense_i64 = dense.astype(np.int64)
                frame_bits += jg2._row_bits(coding4, dense_i64)
                encoder.encode(dense.astype(np.int32), coding4)
                corrector.observe(state, canonical)
                current.reshape(-1)[device_positions] = torch_mod.from_numpy(canonical).to(device)
            code_bits += frame_bits
            per_frame[frame] = frame_bits
            frame_tokens = current[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
            if not np.array_equal(frame_tokens.reshape(-1), plane_target):
                raise D3Error(f"frame {frame}: causal encoder field diverged")
            corrector.end_frame(frame_tokens.reshape(-1))
            previous = current
            if args.checkpoint_every and (frame + 1) % args.checkpoint_every == 0 and frame + 1 < args.frames:
                save_encode_checkpoint(
                    checkpoint, encoder_checkpoint, route_b, encoder, corrector, cold,
                    frame + 1, code_bits, per_frame, frame_tokens,
                )
                progress(
                    stage="encode", event="checkpoint", frame=frame + 1,
                    ideal_bytes=code_bits / 8.0,
                    elapsed_seconds=time.perf_counter() - started,
                )

    encoder.finish()
    size = int(encoder.library.rc64_encoder_size(encoder.context))
    pointer = encoder.library.rc64_encoder_data(encoder.context)
    if not size or not pointer:
        raise D3Error("four-symbol RC64 encoder emitted no stream")
    body = ctypes.string_at(pointer, size)
    stream = work / f"token_stream_alphabet4_n{args.frames}.bin"
    atomic_bytes(stream, body)
    bit_ledger = work / f"bits_per_frame_alphabet4_n{args.frames}.npy"
    np.save(bit_ledger, per_frame)

    sections = dict(env["sections"])
    sections["tail"] = sections["tail"][:jg2.RESIDUAL_COMPACT_BYTES] + body
    candidate = store / "retained/candidates/candidate_d3_rate_only.zip"
    jg2.pack_archive(jg2.join_member(sections), candidate)
    base_stream = env["parts"].token_stream
    result = {
        "schema": "ddm_d3_encode.v1",
        "complete": args.frames == N,
        "frames": args.frames,
        "mechanism": {
            "coded_alphabet": ["Road_or_Lane", "Undrivable", "Movable", "MyCar"],
            "coded_alphabet_size": QUOTIENT_ALPHABET,
            "probability_source": "unchanged shipped GB1 five-output HPAC plus F26 corrector",
            "pool": "q0=pRoad+pLane; q1=pUndrivable; q2=pMovable; q3=pMyCar",
            "feedback": "decoded dense symbols map to canonical 0,2,3,4 before HPAC/F26 state updates",
            "model_refit_claim": False,
        },
        "merged_field": file_fact(retained_paths(store)["merged"]),
        "stream": file_fact(stream),
        "bits_per_frame": file_fact(bit_ledger),
        "stream_bytes_base": len(base_stream),
        "stream_bytes_candidate": len(body),
        "stream_delta_bytes": len(body) - len(base_stream),
        "model_bytes_base_and_candidate": len(env["sections"]["hpac"]),
        "combined_stream_model_base": len(base_stream) + len(env["sections"]["hpac"]),
        "combined_stream_model_candidate": len(body) + len(env["sections"]["hpac"]),
        "candidate_archive": file_fact(candidate),
        "archive_delta_bytes": candidate.stat().st_size - BASE_ARCHIVE_BYTES,
        "delta_S_rate": (candidate.stat().st_size - BASE_ARCHIVE_BYTES) * jg2.S_PER_ARCHIVE_BYTE,
        "ideal_code_bits": code_bits,
        "rc64_build": env["build"],
        "receiver_proof": "PENDING decode stage",
        "score_claim": False,
        "axis": AXIS,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "ENCODE_RESULT.json", result)
    return result


def stage_decode(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    env = load_environment(store)
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]

    encode_result = json.loads((store / "ENCODE_RESULT.json").read_text())
    if not encode_result.get("complete"):
        raise D3Error("decode requires a completed n600 encode")
    stream = Path(encode_result["stream"]["path"])
    if file_fact(stream) != encode_result["stream"]:
        raise D3Error("retained four-symbol stream drifted since encode")
    target = merged_field(store)
    torch_mod, device, model, sparse, corrector, groups = group_machine(env)
    cold = FreeCorrector(PLANE)
    output = store / "retained/decode/tokens_lane_to_road_receiver.u8"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".u8.partial")
    start = 0
    previous_seed: np.ndarray | None = None
    checkpoints = sorted(output.parent.glob("checkpoint_frame_*.npz"))
    if args.resume and checkpoints:
        checkpoint_path = checkpoints[-1]
        blob = np.load(checkpoint_path, allow_pickle=False)
        schema = str(np.asarray(blob["schema"]).reshape(-1)[0])
        if schema != CHECKPOINT_SCHEMA:
            raise D3Error(f"refusing decode checkpoint schema {schema!r}")
        start = int(blob["frame"][0])
        previous_seed = np.asarray(blob["previous"], dtype=np.uint8).copy()
        jg2.load_corrector_state(
            corrector,
            {key: blob[key] for key in blob.files if key not in DECODE_CHECKPOINT_KEYS},
        )
        decoder = env["route_b"].NativeRc64Decoder(
            env["library"], np.asarray(blob["decoder"], dtype=np.uint8).tobytes()
        )
        if not temporary.is_file() or temporary.stat().st_size != start * PLANE:
            raise D3Error(
                f"decode checkpoint frame {start} lacks its exact {start * PLANE} B partial field"
            )
        progress(stage="decode", event="resumed", frame=start, checkpoint=file_fact(checkpoint_path))
    else:
        decoder = env["route_b"].NativeRc64Decoder(
            env["library"], env["route_b"].TOKEN_MAGIC + stream.read_bytes()
        )
    started = time.perf_counter()
    mode = "ab" if start else "wb"
    with temporary.open(mode) as handle, torch_mod.inference_mode():
        if previous_seed is None:
            previous = torch_mod.zeros((1, H, W), dtype=torch_mod.long, device=device)
        else:
            previous = torch_mod.from_numpy(previous_seed.astype(np.int64)).reshape(1, H, W).to(device)
        for frame in range(start, N):
            index = torch_mod.tensor([frame], dtype=torch_mod.long, device=device)
            current = torch_mod.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
                boundary = env["residual"]._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)
            for group, (device_positions, flat_positions) in enumerate(groups):
                state, coding4 = probability_state(
                    env, sparse, corrector, env["parts"], current, context,
                    boundary, group, flat_positions,
                )
                dense = decoder.decode(None, coding4)
                if np.any((dense < 0) | (dense >= QUOTIENT_ALPHABET)):
                    raise D3Error(f"frame {frame} group {group}: decoder left alphabet")
                canonical = LIVE_CANONICAL[dense]
                corrector.observe(state, canonical.astype(np.int64))
                current.reshape(-1)[device_positions] = torch_mod.from_numpy(canonical.astype(np.int64)).to(device)
            frame_tokens = current[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
            expected = np.asarray(target[frame], dtype=np.uint8)
            if not np.array_equal(frame_tokens, expected):
                mismatch = int(np.flatnonzero(frame_tokens.reshape(-1) != expected.reshape(-1))[0])
                raise D3Error(f"receiver mismatch at frame {frame}, raster position {mismatch}")
            handle.write(frame_tokens.tobytes(order="C"))
            corrector.end_frame(frame_tokens.reshape(-1))
            previous = current
            if args.checkpoint_every and (frame + 1) % args.checkpoint_every == 0 and frame + 1 < N:
                handle.flush()
                os.fsync(handle.fileno())
                state = jg2.corrector_state(corrector)
                lost = jg2.uncaptured_divergent_state(corrector, cold, set(state))
                if lost:
                    raise D3Error(f"decode checkpoint would lose live corrector state: {lost[:8]}")
                decoder_payload = decoder.get_compressed().tobytes()
                decoder_checkpoint = output.parent / f"checkpoint_frame_{frame + 1:04d}.u32"
                atomic_bytes(decoder_checkpoint, decoder_payload)
                checkpoint_path = output.parent / f"checkpoint_frame_{frame + 1:04d}.npz"
                checkpoint_tmp = checkpoint_path.with_suffix(".partial.npz")
                np.savez(
                    checkpoint_tmp,
                    schema=np.array([CHECKPOINT_SCHEMA]),
                    frame=np.array([frame + 1], dtype=np.int64),
                    previous=frame_tokens,
                    decoder=np.frombuffer(decoder_payload, dtype=np.uint8),
                    **state,
                )
                os.replace(checkpoint_tmp, checkpoint_path)
                progress(
                    stage="decode", event="checkpoint", frame=frame + 1,
                    decoder_payload=file_fact(decoder_checkpoint),
                    full_state=file_fact(checkpoint_path),
                    elapsed_seconds=time.perf_counter() - started,
                )
    os.replace(temporary, output)
    if output.stat().st_size != FIELD_BYTES or sha256_file(output) != sha256_file(retained_paths(store)["merged"]):
        raise D3Error("receiver output is not byte-identical to the retained merged field")
    if not decoder.is_empty():
        raise D3Error("receiver did not consume exactly n600 symbols")
    result = {
        "schema": "ddm_d3_decode.v1",
        "complete": True,
        "decoded_field": file_fact(output),
        "target_field": file_fact(retained_paths(store)["merged"]),
        "byte_identical": True,
        "decoded_symbols": FIELD_BYTES,
        "coded_alphabet_size": QUOTIENT_ALPHABET,
        "receiver_closed": True,
        "score_claim": False,
        "axis": AXIS,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "DECODE_RESULT.json", result)
    encode_result["receiver_proof"] = file_fact(store / "DECODE_RESULT.json")
    encode_result["receiver_closed"] = True
    atomic_json(store / "ENCODE_RESULT.json", encode_result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", required=True,
        choices=("prepare", "carriers", "encode", "decode", "render", "score")
    )
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument("--frames", type=int, default=N)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 1 or args.frames > N:
        raise SystemExit(f"--frames must be in 1..{N}")
    if args.stage == "decode" and args.frames != N:
        raise SystemExit("decode stage is n600-only")
    result = {
        "prepare": stage_prepare,
        "carriers": stage_carriers,
        "encode": stage_encode,
        "decode": stage_decode,
        "render": stage_render,
        "score": stage_score,
    }[args.stage](args)
    progress(stage=args.stage, event="done", complete=result.get("complete"), axis=AXIS)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
