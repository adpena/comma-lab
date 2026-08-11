#!/usr/bin/env python3
"""Scorer-free n600 carriage probes for the HY1 capstone hybrid.

This program reconstructs the exact C1 batch-16 solved argmax field, encodes
that field with the frozen F26 HPAC probability model and RC64 wire, proves an
independent causal decode, and records where C1 error mass falls in the HPAC
wire's ideal codelength allocation.  It never invokes SegNet, PoseNet, the
contest evaluator, a renderer, or a paid service.

Every materialized stream is retained below ``--output-root``.  RC64 encode
and decode state is checkpointed every 25 frames and can be resumed by
rerunning the same stage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811"
)
C1_EVENTS = REPO / (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "c1_live_target_debt_n600_batch16.json"
)
C1_REPLAY = Path(
    "/Volumes/VertigoDataTier/pact/c1_batch16_exact_replay_20260726/"
    "11_batch_replay_receipt.json"
)
BATCH16_TARGETS = Path(
    "/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/"
    "11_target_labels/target_labels_n600_or_bounded.u8"
)
PR135_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/replay_submission"
)
PR135_ARCHIVE = PR135_ROOT / "archive.zip"
PR135_DECODED_TOKENS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/coders/control/"
    "decoded_spatial_tokens.rc64.bin"
)
RC64_SOURCE = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/"
    "src/cpr1_sub4/entropy/rc64_backend.c"
)
ROUTE_B_MODULE = REPO / (
    "experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py"
)

N = 600
H = 384
W = 512
PIXELS = N * H * W
CLASSES = 5
PATCH = 64
PATCH_ROWS = H // PATCH
PATCH_COLS = W // PATCH
PATCHES = PATCH_ROWS * PATCH_COLS
SHIPPED_F26_RC64_BYTES = 114_706
CP135_ARCHIVE_BYTES = 186_252
CP135_RC64_BYTES = 115_231
CP135_SCORE = 0.16195513827824176
F26_DSEG = 0.00029639352578669786
C1_DSEG = 0.00015196058485243054
RATE_DENOMINATOR = 37_545_489
TARGET_SHA256 = "6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85"
PR135_ARCHIVE_SHA256 = "12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004"
PR135_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_file(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{path} has {path.stat().st_size} bytes, expected {expected_bytes}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{path} SHA-256 changed")


def storage_preflight(root: Path, required_bytes: int = 1_000_000_000) -> dict[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    if usage.free < required_bytes:
        raise RuntimeError(
            f"storage preflight failed: {root} has {usage.free} free bytes; "
            f"need {required_bytes}"
        )
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_bytes": required_bytes,
    }


def load_route_b():
    spec = importlib.util.spec_from_file_location("ddm_hy1_route_b", ROUTE_B_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {ROUTE_B_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_pr135_runtime():
    root_text = str(PR135_ROOT)
    code_text = str(PR135_ROOT / "cpr1")
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    if code_text not in sys.path:
        sys.path.insert(0, code_text)
    import inflate as renderer
    from runtime.hpac_inference import optimize_sparse_evaluator
    from runtime.ihs2 import materialize_ihs1
    from runtime.residual_archive import (
        _boundary_buckets,
        _probability_table,
        _sparse_class,
        read_residual_archive,
    )

    return {
        "materialize_ihs1": materialize_ihs1,
        "optimize_sparse_evaluator": optimize_sparse_evaluator,
        "boundary_buckets": _boundary_buckets,
        "probability_table": _probability_table,
        "sparse_class": _sparse_class,
        "read_residual_archive": read_residual_archive,
        "renderer": renderer,
    }


def input_receipt(root: Path) -> dict[str, Any]:
    require_file(BATCH16_TARGETS, expected_bytes=PIXELS, expected_sha256=TARGET_SHA256)
    require_file(PR135_ARCHIVE, expected_sha256=PR135_ARCHIVE_SHA256)
    require_file(
        PR135_DECODED_TOKENS,
        expected_bytes=PIXELS,
        expected_sha256=PR135_TOKEN_SHA256,
    )
    require_file(C1_EVENTS)
    require_file(C1_REPLAY)
    require_file(RC64_SOURCE)
    require_file(ROUTE_B_MODULE)
    result = {
        "schema": "ddm_hy1_input_preflight.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU scorer-free custody preflight; n600]",
        "score_claim": False,
        "storage": storage_preflight(root),
        "inputs": {
            "batch16_targets": file_fact(BATCH16_TARGETS),
            "pr135_decoded_tokens": file_fact(PR135_DECODED_TOKENS),
            "c1_events": file_fact(C1_EVENTS),
            "c1_exact_replay": file_fact(C1_REPLAY),
            "pr135_archive": file_fact(PR135_ARCHIVE),
            "rc64_source": file_fact(RC64_SOURCE),
            "checkpointable_rc64_wrapper": file_fact(ROUTE_B_MODULE),
        },
    }
    atomic_json(root / "00_INPUT_PREFLIGHT.json", result)
    return result


def stage_materialize(root: Path) -> dict[str, Any]:
    receipt_path = root / "10_MATERIALIZE_RESULT.json"
    retained = root / "retained"
    shipped_path = retained / "pr135_shipped_tokens_n600.u8"
    solved_path = retained / "c1_solved_tokens_n600.u8"
    if receipt_path.is_file() and shipped_path.is_file() and solved_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if (
            sha256_file(shipped_path) == prior["shipped_tokens"]["sha256"]
            and sha256_file(solved_path) == prior["c1_solved_tokens"]["sha256"]
        ):
            return prior

    retained.mkdir(parents=True, exist_ok=True)
    shipped_source = np.memmap(
        PR135_DECODED_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W)
    )
    target = np.memmap(BATCH16_TARGETS, mode="r", dtype=np.uint8, shape=(N, H, W))
    events_doc = json.loads(C1_EVENTS.read_text())
    if events_doc["aggregate"]["pair_count"] != N:
        raise RuntimeError("C1 event bank is not n600")
    if events_doc["aggregate"]["seg_event_count"] != 17_926:
        raise RuntimeError("C1 event count changed from the authoritative batch-16 bank")

    shipped = np.memmap(shipped_path, mode="w+", dtype=np.uint8, shape=(N, H, W))
    solved = np.memmap(solved_path, mode="w+", dtype=np.uint8, shape=(N, H, W))
    event_count = 0
    event_target_mismatches = 0
    event_candidate_equal_target = 0
    pair_event_counts = np.zeros(N, dtype=np.int64)
    class_event_counts = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    for frame, pair in enumerate(events_doc["pairs"]):
        if pair["pair_id"] != frame:
            raise RuntimeError("C1 pair rows are not in exact pair order")
        shipped[frame] = shipped_source[frame]
        solved[frame] = target[frame]
        for row, column, target_class, candidate_class in pair["seg_events"]:
            event_count += 1
            pair_event_counts[frame] += 1
            class_event_counts[target_class, candidate_class] += 1
            if int(target[frame, row, column]) != target_class:
                event_target_mismatches += 1
            if target_class == candidate_class:
                event_candidate_equal_target += 1
            solved[frame, row, column] = candidate_class
    shipped.flush()
    solved.flush()

    if event_count != 17_926 or event_target_mismatches or event_candidate_equal_target:
        raise RuntimeError(
            "C1 event application failed: "
            f"count={event_count}, target_mismatches={event_target_mismatches}, "
            f"equal_classes={event_candidate_equal_target}"
        )
    shipped_hash = sha256_file(shipped_path)
    if shipped_hash != PR135_TOKEN_SHA256:
        raise RuntimeError(
            "the cache lstars field does not equal the exact PR135 decoded token field"
        )

    target_vs_shipped = 0
    solved_vs_shipped = 0
    solved_vs_target = 0
    pair_shipped_disagreements = np.zeros(N, dtype=np.int64)
    for frame in range(N):
        shipped_frame = np.asarray(shipped[frame])
        solved_frame = np.asarray(solved[frame])
        target_frame = np.asarray(target[frame])
        target_vs_shipped += int(np.count_nonzero(target_frame != shipped_frame))
        pair_shipped_disagreements[frame] = np.count_nonzero(
            solved_frame != shipped_frame
        )
        solved_vs_shipped += int(pair_shipped_disagreements[frame])
        solved_vs_target += int(np.count_nonzero(solved_frame != target_frame))

    arrays_path = retained / "c1_disagreement_counts_n600.npz"
    atomic_npz(
        arrays_path,
        pair_c1_events=pair_event_counts,
        pair_shipped_disagreements=pair_shipped_disagreements,
        class_event_counts=class_event_counts,
    )
    result = {
        "schema": "ddm_hy1_c1_token_materialization.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU scorer-free token-grid reconstruction; n600]",
        "score_claim": False,
        "source_event_count": event_count,
        "source_event_denominator_pixels": PIXELS,
        "target_vs_pr135_shipped_disagreement_pixels": target_vs_shipped,
        "c1_solved_vs_target_disagreement_pixels": solved_vs_target,
        "c1_solved_vs_pr135_shipped_disagreement_pixels": solved_vs_shipped,
        "event_target_mismatches": event_target_mismatches,
        "event_candidate_equal_target": event_candidate_equal_target,
        "shipped_tokens": file_fact(shipped_path),
        "c1_solved_tokens": file_fact(solved_path),
        "disagreement_arrays": file_fact(arrays_path),
        "representability": {
            "grammar": "dense [600,384,512] five-class PR135 semantic-token lattice",
            "disagreement_sites": solved_vs_shipped,
            "sites_with_one_legal_replacement_token": solved_vs_shipped,
            "token_grid_representability_fraction": 1.0,
            "renderer_realization_survival": "UNMEASURED_SCORER_LANE_OWNED_BY_PS135B",
        },
        "boundary": (
            "The batch-16 target bank is encoder-side research guidance only. "
            "No candidate archive is built or claimed."
        ),
    }
    atomic_json(receipt_path, result)
    return result


def stage_brotli(root: Path) -> dict[str, Any]:
    receipt_path = root / "20_BROTLI_RESULT.json"
    source = root / "retained/c1_solved_tokens_n600.u8"
    output = root / "retained/c1_solved_tokens_n600.brotli_q11"
    repeat = root / "retained/c1_solved_tokens_n600.repeat.brotli_q11"
    if receipt_path.is_file() and output.is_file() and repeat.is_file():
        prior = json.loads(receipt_path.read_text())
        if (
            sha256_file(output) == prior["payload"]["sha256"]
            and sha256_file(repeat) == prior["repeat_payload"]["sha256"]
        ):
            return prior
    import brotli

    raw = source.read_bytes()
    started = time.time()
    payload = brotli.compress(raw, quality=11)
    atomic_bytes(output, payload)
    repeat_payload = brotli.compress(raw, quality=11)
    atomic_bytes(repeat, repeat_payload)
    decoded = brotli.decompress(payload)
    if decoded != raw or repeat_payload != payload:
        raise RuntimeError("Brotli q11 failed exact decode or deterministic repeat")
    payload_fact = file_fact(output)
    result = {
        "schema": "ddm_hy1_brotli_control.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU scorer-free real Brotli q11; n600]",
        "score_claim": False,
        "source": file_fact(source),
        "payload": payload_fact,
        "repeat_payload": file_fact(repeat),
        "repeat_identical": True,
        "decode_exact": True,
        "elapsed_seconds": time.time() - started,
        "delta_vs_shipped_f26_rc64_bytes": (
            payload_fact["bytes"] - SHIPPED_F26_RC64_BYTES
        ),
    }
    atomic_json(receipt_path, result)
    return result


def stage_prepare_rc64(root: Path) -> dict[str, Any]:
    receipt_path = root / "30_RC64_PREPARE_RESULT.json"
    build = root / "build"
    source_path = build / "rc64_backend_checkpointable.c"
    library_a = build / "a/libhy1_rc64.dylib"
    library_b = build / "b/libhy1_rc64.dylib"
    if receipt_path.is_file() and library_a.is_file() and library_b.is_file():
        prior = json.loads(receipt_path.read_text())
        if (
            sha256_file(library_a) == prior["library_a"]["sha256"]
            and sha256_file(library_b) == prior["library_b"]["sha256"]
        ):
            return prior

    route_b = load_route_b()
    source = RC64_SOURCE.read_bytes() + (
        "\n" + route_b.RC64_CHECKPOINT_EXTENSION
    ).encode("utf-8")
    atomic_bytes(source_path, source)
    commands = []
    for destination in (library_a, library_b):
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "/usr/bin/cc",
            "-O3",
            "-std=c11",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-Wl,-install_name,@rpath/libhy1_rc64.dylib",
            str(source_path),
            "-o",
            str(destination),
        ]
        subprocess.run(command, check=True)
        commands.append(command)
    if library_a.read_bytes() != library_b.read_bytes():
        raise RuntimeError("repeat RC64 builds are not byte-identical")
    result = {
        "schema": "ddm_hy1_rc64_prepare.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU scorer-free native build]",
        "score_claim": False,
        "source": file_fact(source_path),
        "library_a": file_fact(library_a),
        "library_b": file_fact(library_b),
        "compile_repeat_identical": True,
        "argv": commands,
    }
    atomic_json(receipt_path, result)
    return result


def hpac_objects():
    import torch

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this process-wide setting only before parallel work.
        pass
    torch.manual_seed(20260811)
    runtime = load_pr135_runtime()
    parts = runtime["read_residual_archive"](PR135_ARCHIVE)
    renderer = runtime["renderer"]
    base_hpac = runtime["materialize_ihs1"](parts.hpac_blob, renderer)
    device = torch.device("cpu")
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = runtime["sparse_class"](PR135_ROOT / "cpr1")(
        model, renderer.EVAL_H, renderer.EVAL_W
    )
    runtime["optimize_sparse_evaluator"](sparse)
    plans = []
    for mask in masks:
        positions = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        plans.append((mask, positions))
    return runtime, parts, renderer, model, sparse, plans, device


def empty_allocation(group_count: int) -> dict[str, np.ndarray]:
    return {
        "pair_bits": np.zeros(N, dtype=np.float64),
        "pair_disagreement_bits": np.zeros(N, dtype=np.float64),
        "pair_event_bits": np.zeros(N, dtype=np.float64),
        "group_bits": np.zeros(group_count, dtype=np.float64),
        "group_disagreement_bits": np.zeros(group_count, dtype=np.float64),
        "group_event_bits": np.zeros(group_count, dtype=np.float64),
        "group_event_counts": np.zeros(group_count, dtype=np.int64),
        "patch_bits": np.zeros((N, PATCHES), dtype=np.float64),
        "patch_disagreement_bits": np.zeros((N, PATCHES), dtype=np.float64),
        "patch_event_bits": np.zeros((N, PATCHES), dtype=np.float64),
        "patch_event_counts": np.zeros((N, PATCHES), dtype=np.int64),
        "class_bits": np.zeros(CLASSES, dtype=np.float64),
        "frame_corrected_hashes": np.zeros((N, 32), dtype=np.uint8),
        "frame_probability_hashes": np.zeros((N, 32), dtype=np.uint8),
    }


def load_allocation(path: Path, group_count: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        return empty_allocation(group_count)
    with np.load(path) as data:
        result = {name: data[name].copy() for name in data.files}
    expected = set(empty_allocation(group_count))
    if set(result) != expected:
        raise RuntimeError("allocation checkpoint schema changed")
    return result


def encode_checkpoint(
    root: Path,
    frame: int,
    encoder,
    allocation: dict[str, np.ndarray],
) -> None:
    checkpoint_dir = root / "checkpoints/encode"
    state_path = checkpoint_dir / f"through_frame_{frame - 1:03d}.encoder"
    allocation_path = checkpoint_dir / f"through_frame_{frame - 1:03d}.allocation.npz"
    atomic_bytes(state_path, encoder.snapshot())
    atomic_npz(allocation_path, **allocation)
    receipt = {
        "schema": "ddm_hy1_rc64_encode_checkpoint.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "next_frame": frame,
        "encoder_checkpoint": file_fact(state_path),
        "allocation_checkpoint": file_fact(allocation_path),
    }
    atomic_json(checkpoint_dir / f"through_frame_{frame - 1:03d}.json", receipt)
    atomic_json(checkpoint_dir / "LATEST.json", receipt)


def stage_encode(root: Path) -> dict[str, Any]:
    receipt_path = root / "40_RC64_ENCODE_RESULT.json"
    token_path = root / "retained/c1_solved_tokens_n600.f26_hpac.rc64"
    allocation_path = root / "retained/hpac_wire_allocation_n600.npz"
    if receipt_path.is_file() and token_path.is_file() and allocation_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if (
            sha256_file(token_path) == prior["token_payload"]["sha256"]
            and sha256_file(allocation_path) == prior["wire_allocation"]["sha256"]
        ):
            return prior

    import torch

    route_b = load_route_b()
    runtime, parts, renderer, model, sparse, plans, device = hpac_objects()
    group_count = len(plans)
    solved = np.memmap(
        root / "retained/c1_solved_tokens_n600.u8",
        mode="r",
        dtype=np.uint8,
        shape=(N, H, W),
    )
    shipped = np.memmap(
        root / "retained/pr135_shipped_tokens_n600.u8",
        mode="r",
        dtype=np.uint8,
        shape=(N, H, W),
    )
    target = np.memmap(BATCH16_TARGETS, mode="r", dtype=np.uint8, shape=(N, H, W))
    latest_path = root / "checkpoints/encode/LATEST.json"
    start_frame = 0
    checkpoint = None
    allocation = empty_allocation(group_count)
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text())
        state_path = Path(latest["encoder_checkpoint"]["path"])
        saved_allocation = Path(latest["allocation_checkpoint"]["path"])
        if sha256_file(state_path) != latest["encoder_checkpoint"]["sha256"]:
            raise RuntimeError("encoder checkpoint SHA-256 changed")
        if sha256_file(saved_allocation) != latest["allocation_checkpoint"]["sha256"]:
            raise RuntimeError("allocation checkpoint SHA-256 changed")
        checkpoint = state_path.read_bytes()
        allocation = load_allocation(saved_allocation, group_count)
        start_frame = int(latest["next_frame"])

    encoder = route_b.NativeRc64Encoder(
        root / "build/a/libhy1_rc64.dylib", checkpoint=checkpoint
    )
    started = time.time()
    try:
        with torch.inference_mode():
            for frame in range(start_frame, N):
                previous_np = (
                    np.zeros((H, W), dtype=np.uint8)
                    if frame == 0
                    else np.asarray(solved[frame - 1])
                )
                previous = torch.from_numpy(previous_np.copy()).long().view(1, H, W)
                index = torch.tensor([frame], dtype=torch.long, device=device)
                current = torch.zeros_like(previous)
                context = model.prepare_frame_context(index, previous)
                boundary = (
                    np.full(H * W, 4, dtype=np.uint8)
                    if frame == 0
                    else runtime["boundary_buckets"](previous_np).reshape(-1)
                )
                solved_flat = np.asarray(solved[frame]).reshape(-1)
                shipped_flat = np.asarray(shipped[frame]).reshape(-1)
                target_flat = np.asarray(target[frame]).reshape(-1)
                corrected_digest = hashlib.sha256()
                probability_digest = hashlib.sha256()
                for group, (_mask, flat_positions) in enumerate(plans):
                    selected = sparse.selected_logits(current, context, group)
                    base_logits = selected.cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = (
                        boundary[flat_positions].astype(np.int64) * CLASSES
                        + predicted
                    )
                    corrected = np.ascontiguousarray(
                        base_logits + parts.table.values[feature], dtype=np.float32
                    )
                    probability = runtime["probability_table"](
                        corrected, renderer.HPAC_LOGIT_PRECISION
                    )
                    symbols = np.ascontiguousarray(
                        solved_flat[flat_positions], dtype=np.int32
                    )
                    encoder.encode(symbols, probability)
                    nll = -np.log2(
                        probability[np.arange(len(symbols)), symbols].astype(np.float64)
                    )
                    disagreement = (
                        solved_flat[flat_positions] != shipped_flat[flat_positions]
                    )
                    event = solved_flat[flat_positions] != target_flat[flat_positions]
                    rows = flat_positions // W
                    columns = flat_positions % W
                    patch_indices = (rows // PATCH) * PATCH_COLS + columns // PATCH
                    allocation["pair_bits"][frame] += nll.sum()
                    allocation["pair_disagreement_bits"][frame] += nll[disagreement].sum()
                    allocation["pair_event_bits"][frame] += nll[event].sum()
                    allocation["group_bits"][group] += nll.sum()
                    allocation["group_disagreement_bits"][group] += nll[disagreement].sum()
                    allocation["group_event_bits"][group] += nll[event].sum()
                    allocation["group_event_counts"][group] += int(event.sum())
                    allocation["patch_bits"][frame] += np.bincount(
                        patch_indices, weights=nll, minlength=PATCHES
                    )
                    allocation["patch_disagreement_bits"][frame] += np.bincount(
                        patch_indices[disagreement],
                        weights=nll[disagreement],
                        minlength=PATCHES,
                    )
                    allocation["patch_event_bits"][frame] += np.bincount(
                        patch_indices[event], weights=nll[event], minlength=PATCHES
                    )
                    allocation["patch_event_counts"][frame] += np.bincount(
                        patch_indices[event], minlength=PATCHES
                    )
                    allocation["class_bits"] += np.bincount(
                        symbols, weights=nll, minlength=CLASSES
                    )
                    corrected_digest.update(corrected.astype("<f4", copy=False).tobytes())
                    probability_digest.update(
                        np.ascontiguousarray(probability, dtype="<f4").tobytes()
                    )
                    current.reshape(-1)[torch.from_numpy(flat_positions)] = (
                        torch.from_numpy(symbols).long()
                    )
                allocation["frame_corrected_hashes"][frame] = np.frombuffer(
                    corrected_digest.digest(), dtype=np.uint8
                )
                allocation["frame_probability_hashes"][frame] = np.frombuffer(
                    probability_digest.digest(), dtype=np.uint8
                )
                if (frame + 1) % 25 == 0:
                    encode_checkpoint(root, frame + 1, encoder, allocation)
                    print(
                        json.dumps(
                            {
                                "encoded_frames": frame + 1,
                                "elapsed_seconds": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
        framed = encoder.finish()
        raw_size = int(encoder.library.rc64_encoder_size(encoder.context))
        if not framed.startswith(route_b.TOKEN_MAGIC) or raw_size <= 0:
            raise RuntimeError("checkpointable RC64 wrapper returned invalid framing")
        token_payload = framed[len(route_b.TOKEN_MAGIC) : len(route_b.TOKEN_MAGIC) + raw_size]
    finally:
        encoder.close()

    atomic_bytes(token_path, token_payload)
    atomic_npz(allocation_path, **allocation)
    corrected_hash = hashlib.sha256(
        allocation["frame_corrected_hashes"].tobytes()
    ).hexdigest()
    probability_hash = hashlib.sha256(
        allocation["frame_probability_hashes"].tobytes()
    ).hexdigest()
    result = {
        "schema": "ddm_hy1_f26_hpac_rc64_encode.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU scorer-free frozen-F26-HPAC native-RC64; n600]",
        "score_claim": False,
        "frames": N,
        "symbols": PIXELS,
        "resumed_from_frame": start_frame,
        "elapsed_seconds_this_invocation": time.time() - started,
        "token_payload": file_fact(token_path),
        "wire_allocation": file_fact(allocation_path),
        "ideal_bits": float(allocation["pair_bits"].sum()),
        "ideal_bytes": float(allocation["pair_bits"].sum() / 8),
        "coder_overhead_bytes": float(
            token_path.stat().st_size - allocation["pair_bits"].sum() / 8
        ),
        "delta_vs_shipped_f26_rc64_bytes": (
            token_path.stat().st_size - SHIPPED_F26_RC64_BYTES
        ),
        "frame_digest_aggregate_schema": "sha256(concat(per-frame-sha256-bytes))",
        "corrected_logit_frame_digest_aggregate": corrected_hash,
        "probability_frame_digest_aggregate": probability_hash,
        "hpac_blob": {
            "bytes": len(parts.hpac_blob),
            "sha256": hashlib.sha256(parts.hpac_blob).hexdigest(),
        },
        "residual_table": {
            "name": parts.table.name,
            "scale": parts.table.scale,
            "codes_sha256": hashlib.sha256(parts.table.codes.tobytes()).hexdigest(),
        },
        "borrowed_substrate": (
            "PR135/F26 HPAC model, fixed residual table, group grammar, and RC64 recurrence"
        ),
    }
    atomic_json(receipt_path, result)
    return result


def decoder_checkpoint(root: Path, frame: int, decoder, decoded_path: Path) -> None:
    checkpoint_dir = root / "checkpoints/decode"
    state_path = checkpoint_dir / f"through_frame_{frame - 1:03d}.decoder"
    atomic_bytes(state_path, decoder.get_compressed().tobytes())
    receipt = {
        "schema": "ddm_hy1_rc64_decode_checkpoint.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "next_frame": frame,
        "decoder_checkpoint": file_fact(state_path),
        "decoded_output": file_fact(decoded_path),
    }
    atomic_json(checkpoint_dir / f"through_frame_{frame - 1:03d}.json", receipt)
    atomic_json(checkpoint_dir / "LATEST.json", receipt)


def stage_decode(root: Path) -> dict[str, Any]:
    receipt_path = root / "50_RC64_DECODE_RESULT.json"
    token_path = root / "retained/c1_solved_tokens_n600.f26_hpac.rc64"
    solved_path = root / "retained/c1_solved_tokens_n600.u8"
    decoded_path = root / "retained/c1_solved_tokens_n600.rc64_decoded.u8"
    if receipt_path.is_file() and decoded_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if sha256_file(decoded_path) == prior["decoded_tokens"]["sha256"]:
            return prior

    import torch

    route_b = load_route_b()
    runtime, parts, renderer, model, sparse, plans, device = hpac_objects()
    raw = token_path.read_bytes()
    framed = route_b.TOKEN_MAGIC + raw
    framed += b"\0" * ((-len(framed)) % 4)
    latest_path = root / "checkpoints/decode/LATEST.json"
    start_frame = 0
    decoder_payload = framed
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text())
        state_path = Path(latest["decoder_checkpoint"]["path"])
        if sha256_file(state_path) != latest["decoder_checkpoint"]["sha256"]:
            raise RuntimeError("decoder checkpoint SHA-256 changed")
        if decoded_path.is_file() and decoded_path.stat().st_size != PIXELS:
            raise RuntimeError("retained decoded token field has the wrong size")
        decoder_payload = state_path.read_bytes()
        start_frame = int(latest["next_frame"])
    if not decoded_path.is_file():
        decoded_path.parent.mkdir(parents=True, exist_ok=True)
        with decoded_path.open("wb") as handle:
            handle.truncate(PIXELS)
            handle.flush()
            os.fsync(handle.fileno())
    output = np.memmap(decoded_path, mode="r+", dtype=np.uint8, shape=(N, H, W))
    decoder = route_b.NativeRc64Decoder(
        root / "build/a/libhy1_rc64.dylib", decoder_payload
    )
    started = time.time()
    try:
        with torch.inference_mode():
            for frame in range(start_frame, N):
                previous_np = (
                    np.zeros((H, W), dtype=np.uint8)
                    if frame == 0
                    else np.asarray(output[frame - 1])
                )
                previous = torch.from_numpy(previous_np.copy()).long().view(1, H, W)
                index = torch.tensor([frame], dtype=torch.long, device=device)
                current = torch.zeros_like(previous)
                context = model.prepare_frame_context(index, previous)
                boundary = (
                    np.full(H * W, 4, dtype=np.uint8)
                    if frame == 0
                    else runtime["boundary_buckets"](previous_np).reshape(-1)
                )
                for group, (_, flat_positions) in enumerate(plans):
                    selected = sparse.selected_logits(current, context, group)
                    base_logits = selected.cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = (
                        boundary[flat_positions].astype(np.int64) * CLASSES
                        + predicted
                    )
                    corrected = np.ascontiguousarray(
                        base_logits + parts.table.values[feature], dtype=np.float32
                    )
                    probability = runtime["probability_table"](
                        corrected, renderer.HPAC_LOGIT_PRECISION
                    )
                    symbols = decoder.decode(None, probability).astype(np.int64)
                    current.reshape(-1)[torch.from_numpy(flat_positions)] = (
                        torch.from_numpy(symbols)
                    )
                output[frame] = current[0].to(torch.uint8).cpu().numpy()
                if (frame + 1) % 25 == 0:
                    output.flush()
                    decoder_checkpoint(root, frame + 1, decoder, decoded_path)
                    print(
                        json.dumps(
                            {
                                "decoded_frames": frame + 1,
                                "elapsed_seconds": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
        output.flush()
        if not decoder.is_empty():
            raise RuntimeError("RC64 decoder did not consume exactly n600 symbols")
    finally:
        decoder.close()

    solved_hash = sha256_file(solved_path)
    decoded_hash = sha256_file(decoded_path)
    if decoded_hash != solved_hash or decoded_path.read_bytes() != solved_path.read_bytes():
        raise RuntimeError("independent RC64 decode differs from C1 solved tokens")
    result = {
        "schema": "ddm_hy1_f26_hpac_rc64_decode.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU scorer-free independent causal RC64 decode; n600]",
        "score_claim": False,
        "resumed_from_frame": start_frame,
        "elapsed_seconds_this_invocation": time.time() - started,
        "token_payload": file_fact(token_path),
        "decoded_tokens": file_fact(decoded_path),
        "source_tokens": file_fact(solved_path),
        "decoded_symbol_count": PIXELS,
        "exact_decode_equality": True,
        "renderer_realization_survival": "UNMEASURED_SCORER_LANE_OWNED_BY_PS135B",
    }
    atomic_json(receipt_path, result)
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    pearson = float(np.corrcoef(left, right)[0, 1])
    spearman = float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def concentration_rows(events: np.ndarray, bits: np.ndarray) -> list[dict[str, Any]]:
    events = np.asarray(events).reshape(-1)
    bits = np.asarray(bits).reshape(-1)
    event_order = np.argsort(-events, kind="mergesort")
    bit_order = np.argsort(-bits, kind="mergesort")
    rows = []
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(np.ceil(len(events) * fraction)))
        event_top = event_order[:count]
        bit_top = bit_order[:count]
        overlap = len(np.intersect1d(event_top, bit_top, assume_unique=False))
        rows.append(
            {
                "top_fraction": fraction,
                "cell_count": count,
                "event_mass_share_in_event_ranked_cells": float(
                    events[event_top].sum() / events.sum()
                ),
                "wire_bit_share_in_event_ranked_cells": float(
                    bits[event_top].sum() / bits.sum()
                ),
                "top_event_vs_top_wire_overlap_fraction": overlap / count,
            }
        )
    return rows


def stage_finalize(root: Path) -> dict[str, Any]:
    material = json.loads((root / "10_MATERIALIZE_RESULT.json").read_text())
    brotli_result = json.loads((root / "20_BROTLI_RESULT.json").read_text())
    encoded = json.loads((root / "40_RC64_ENCODE_RESULT.json").read_text())
    decoded = json.loads((root / "50_RC64_DECODE_RESULT.json").read_text())
    with np.load(root / "retained/hpac_wire_allocation_n600.npz") as data:
        allocation = {name: data[name].copy() for name in data.files}
    with np.load(root / "retained/c1_disagreement_counts_n600.npz") as data:
        disagreement = {name: data[name].copy() for name in data.files}

    total_bits = float(allocation["pair_bits"].sum())
    event_bits = float(allocation["pair_event_bits"].sum())
    disagreement_bits = float(allocation["pair_disagreement_bits"].sum())
    event_count = int(disagreement["pair_c1_events"].sum())
    disagreement_count = int(disagreement["pair_shipped_disagreements"].sum())
    stream_bytes = encoded["token_payload"]["bytes"]
    delta_bytes = stream_bytes - SHIPPED_F26_RC64_BYTES
    rate_delta = 25 * delta_bytes / RATE_DENOMINATOR
    seg_gain_full_transport = 100 * (C1_DSEG - F26_DSEG)
    full_transport_proxy = CP135_SCORE + rate_delta + seg_gain_full_transport
    required_transport_fraction = (
        (CP135_SCORE + rate_delta - 0.15) / (-seg_gain_full_transport)
    )
    pair_events = disagreement["pair_c1_events"]
    pair_bits = allocation["pair_bits"]
    patch_events = allocation["patch_event_counts"]
    patch_bits = allocation["patch_bits"]
    group_events = allocation["group_event_counts"]
    group_bits = allocation["group_bits"]
    representability_fraction = material["representability"][
        "token_grid_representability_fraction"
    ]
    result = {
        "schema": "ddm_hy1_capstone_hybrid_probe.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU scorer-free real-coder n600]",
        "score_claim": False,
        "pointer_moved": False,
        "effective_frontier": {
            "score": CP135_SCORE,
            "archive_bytes": CP135_ARCHIVE_BYTES,
            "axis": "[contest-CUDA T4,n600] prior exact receipt",
        },
        "probe_a_c1_solved_partition_coding": {
            "measured": {
                "f26_hpac_rc64_bytes": stream_bytes,
                "f26_hpac_rc64_sha256": encoded["token_payload"]["sha256"],
                "delta_vs_shipped_f26_rc64_bytes": delta_bytes,
                "relative_delta_vs_shipped_f26_rc64": (
                    delta_bytes / SHIPPED_F26_RC64_BYTES
                ),
                "ideal_bits": total_bits,
                "ideal_bytes": total_bits / 8,
                "coder_overhead_bytes": encoded["coder_overhead_bytes"],
                "brotli_q11_bytes": brotli_result["payload"]["bytes"],
                "brotli_q11_delta_vs_f26_rc64_bytes": brotli_result[
                    "delta_vs_shipped_f26_rc64_bytes"
                ],
                "independent_rc64_decode_exact": decoded["exact_decode_equality"],
            },
            "prediction_within_plus_or_minus_10_percent": (
                abs(delta_bytes) <= 0.10 * SHIPPED_F26_RC64_BYTES
            ),
            "preregistered_rate_falsifier_delta_gt_8000_bytes": delta_bytes > 8_000,
            "verdict_scope": (
                "INSTANCE: C1 batch-16 solved argmax tokens under frozen F26 HPAC, "
                "fixed residual table, and native RC64"
            ),
        },
        "probe_b_token_grid_representability": {
            "measured": material["representability"],
            "source_c1_event_count": event_count,
            "pr135_shipped_vs_c1_solved_disagreement_count": disagreement_count,
            "batch16_target_vs_pr135_shipped_disagreement_count": material[
                "target_vs_pr135_shipped_disagreement_pixels"
            ],
            "preregistered_representability_gate_ge_50_percent": (
                representability_fraction >= 0.50
            ),
            "preregistered_falsifier_lt_20_percent": (
                representability_fraction < 0.20
            ),
            "boundary": (
                "Dense token replacement is exact at the parser grammar. Survival through "
                "the learned renderer, R, and SegNet is not measured here."
            ),
        },
        "probe_c_flip_mass_by_wire_layout": {
            "denominators": {
                "pairs": N,
                "pixels": PIXELS,
                "c1_event_pixels": event_count,
                "pr135_disagreement_pixels": disagreement_count,
                "hpac_groups": len(group_bits),
                "pair_patch_cells": N * PATCHES,
            },
            "c1_event_density": event_count / PIXELS,
            "c1_event_ideal_bit_share": event_bits / total_bits,
            "c1_event_wire_enrichment_over_pixel_density": (
                (event_bits / total_bits) / (event_count / PIXELS)
            ),
            "pr135_disagreement_ideal_bit_share": disagreement_bits / total_bits,
            "pr135_disagreement_wire_enrichment_over_pixel_density": (
                (disagreement_bits / total_bits) / (disagreement_count / PIXELS)
            ),
            "pair_event_mass_vs_pair_wire_bits": correlation(pair_events, pair_bits),
            "patch_event_mass_vs_patch_wire_bits": correlation(patch_events, patch_bits),
            "group_event_mass_vs_group_wire_bits": correlation(group_events, group_bits),
            "pair_patch_concentration": concentration_rows(patch_events, patch_bits),
            "group_concentration": concentration_rows(group_events, group_bits),
            "allocation_boundary": (
                "Wire allocation is exact ideal codelength under the candidate's frozen HPAC "
                "probabilities. RC64 is one adaptive stream, so literal emitted bytes are not "
                "separable by pair, patch, or group."
            ),
        },
        "derived_pricing": {
            "label": "DERIVED_NOT_A_CANDIDATE_SCORE",
            "rate_delta_if_the_measured_f26_stream_delta_transferred_to_cp135": rate_delta,
            "full_c1_seg_transport_delta_from_f26_dseg": seg_gain_full_transport,
            "full_transport_score_proxy_from_cp135": full_transport_proxy,
            "required_realized_fraction_of_c1_seg_gain_for_sub_0_15": (
                required_transport_fraction
            ),
            "non_additivity_boundary": (
                "CP135 ships a different HP3 probability object and a 115,231-byte token "
                "stream. The measured F26 stream delta cannot be inserted into CP135 without "
                "a whole-container rebuild and joint scorer replay. C1 d_seg belongs to its "
                "409.5 MB RGB solve, not to this token carriage."
            ),
        },
        "retained_artifacts": {
            "materialization": material,
            "brotli": brotli_result,
            "rc64_encode": encoded,
            "rc64_decode": decoded,
        },
    }
    output = root / "HY1_PROBE_RESULT.json"
    atomic_json(output, result)
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "TREE_MANIFEST.json":
            manifest.append(file_fact(path))
    atomic_json(
        root / "TREE_MANIFEST.json",
        {
            "schema": "ddm_hy1_retained_tree_manifest.v1",
            "complete": True,
            "written_at_utc": utc_now(),
            "file_count": len(manifest),
            "files": manifest,
        },
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    result.add_argument(
        "--stage",
        choices=("preflight", "materialize", "brotli", "prepare", "encode", "decode", "finalize", "all"),
        default="all",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    root = args.output_root.resolve()
    stages = {
        "preflight": lambda: input_receipt(root),
        "materialize": lambda: stage_materialize(root),
        "brotli": lambda: stage_brotli(root),
        "prepare": lambda: stage_prepare_rc64(root),
        "encode": lambda: stage_encode(root),
        "decode": lambda: stage_decode(root),
        "finalize": lambda: stage_finalize(root),
    }
    order = ("preflight", "materialize", "brotli", "prepare", "encode", "decode", "finalize")
    selected = order if args.stage == "all" else (args.stage,)
    for name in selected:
        result = stages[name]()
        print(json.dumps({"stage": name, "complete": result.get("complete", False)}), flush=True)


if __name__ == "__main__":
    main()
