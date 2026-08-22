#!/usr/bin/env python3
"""Retain the shipped DX2 HPAC/RC64 cost at every decoded token position.

This is scorer-free receiver instrumentation.  It drives the unmodified shipped
Python HPAC fallback and the unmodified shipped RC64 C decoder, verifies every
decoded symbol against TO2's retained n600 field, and persists two aligned cost
fields:

* the primary RC64 integer-frequency cost, ``-log2(freq / 2**31)``;
* the float32 probability-input cost, ``-log2(p)`` before RC64 quantization.

The run is crash-resumable at 20-frame stage boundaries.  Every stage preserves
its cost fields, decoded tokens, complete adaptive-corrector state, and RC64
decoder state under a distinct filename.  Finalization keeps those checkpoints
and assembles full raster-order fields plus denominator-complete joins.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
DEFAULT_STORE = VERTIGO_ROOT / "ddm_bl1_per_position_bit_allocation" / "measurement_v1"
RUNTIME_ROOT = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
TO2_ROOT = VERTIGO_ROOT / "ddm_to2_token_ordering_race" / "measurement_v1" / "retained" / "input"
DX2_ARCHIVE = TO2_ROOT / "archive.zip"
TO2_TOKENS = TO2_ROOT / "dx2_tokens_decoded.u8"
TO2_STREAM = TO2_ROOT / "dx2_token_stream_rc64.bin"
TO2_CHECKPOINT = TO2_ROOT / "tokens_cpu_stage_complete.json"
RC64_SOURCE = RUNTIME_ROOT / "runtime" / "entropy" / "rc64_backend.c"
DC1_LEDGER = Path("/Volumes/APDataStore/pact/ddm_fx5/work/bits_per_frame_e1_19member.npy")
MS9_ROOT = VERTIGO_ROOT / "ddm_ms9_dx2_seg_manufactured_fraction"
MS9_RECEIPT = MS9_ROOT / "MS9_FIELD_REPLAY.json"
MS9_MASK_MANIFEST = MS9_ROOT / "MASK_MANIFEST.json"
MS9_FINAL_ERROR = MS9_ROOT / "retained" / "masks" / "final_error.n600.packbits"
GT_FIELD = VERTIGO_ROOT / "ddm_qs3_20260813" / "retained" / "inputs" / "gt_argmax_n600.npy"
ENCODER_SOURCE = VERTIGO_ROOT / (
    "pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/rc64_backend.c"
)

EXPECTED = {
    "archive_bytes": 180_368,
    "archive_sha256": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    "tokens_bytes": 117_964_800,
    "tokens_sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",  # gitleaks:allow -- public content digest
    "stream_bytes": 113_777,
    "stream_sha256": "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5",
    "checkpoint_sha256": "c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9",
    "dc1_ledger_sha256": "0585b0d98ba2958be3e20021641dd0a74bc61714d1434cab16efb005320418df",
    "dc1_ledger_bits": 910_209.4321425341,
    "gt_sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    "ms9_receipt_sha256": "c14494194bda3d0dba30c1a5e5813d4a01646571239352398edfa29f8f79ddd5",
    "ms9_manifest_sha256": "2df0abbae76a1234f8af0e5a08bd857254cf9a6299f63a61b9ae06021b329cdd",
}

N = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
GROUPS = 190
CLASSES = 5
TOTAL_FREQUENCY = 1 << 31
STAGE_FRAMES = 20
STREAM_BITS = EXPECTED["stream_bytes"] * 8
TERMINAL_OVERHEAD_BOUND_BITS = 9.0
ESTIMATED_ARTIFACT_BYTES = 5 << 30
RESERVE_BYTES = 8 << 30
MIN_FREE_BYTES = ESTIMATED_ARTIFACT_BYTES + RESERVE_BYTES
FIELD_DTYPE = np.dtype("<f8")
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CHECKPOINT_SCHEMA = "ddm_bl1_stage_checkpoint.v1"


class Bl1Error(RuntimeError):
    """A fail-closed custody, receiver, resume, or reconciliation error."""


class Rc64DecoderLayout(ctypes.Structure):
    """Exact public-source layout of the shipped ``rc64_decoder`` struct."""

    _fields_ = [
        ("low", ctypes.c_uint64),
        ("high", ctypes.c_uint64),
        ("code", ctypes.c_uint64),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
        ("bit_position", ctypes.c_size_t),
        ("error", ctypes.c_int),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_file(path: Path, expected_sha: str, expected_bytes: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise Bl1Error(f"required custody file is absent: {path}")
    fact = file_fact(path)
    if fact["sha256"] != expected_sha:
        raise Bl1Error(f"SHA-256 drift for {path}: {fact['sha256']} != {expected_sha}")
    if expected_bytes is not None and fact["bytes"] != expected_bytes:
        raise Bl1Error(f"byte-size drift for {path}: {fact['bytes']} != {expected_bytes}")
    return fact


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_raw(path: Path, arrays: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            for array in arrays:
                handle.write(np.ascontiguousarray(array).tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def source_binding() -> dict[str, object]:
    sources = {
        "archive": verify_file(DX2_ARCHIVE, EXPECTED["archive_sha256"], EXPECTED["archive_bytes"]),
        "to2_tokens": verify_file(TO2_TOKENS, EXPECTED["tokens_sha256"], EXPECTED["tokens_bytes"]),
        "to2_stream": verify_file(TO2_STREAM, EXPECTED["stream_sha256"], EXPECTED["stream_bytes"]),
        "to2_checkpoint": verify_file(TO2_CHECKPOINT, EXPECTED["checkpoint_sha256"]),
        "dc1_ledger": verify_file(DC1_LEDGER, EXPECTED["dc1_ledger_sha256"]),
        "gt_field": verify_file(GT_FIELD, EXPECTED["gt_sha256"]),
        "ms9_receipt": verify_file(MS9_RECEIPT, EXPECTED["ms9_receipt_sha256"]),
        "ms9_manifest": verify_file(MS9_MASK_MANIFEST, EXPECTED["ms9_manifest_sha256"]),
        "rc64_decoder_source": file_fact(RC64_SOURCE),
        "rc64_encoder_source": file_fact(ENCODER_SOURCE),
        "implementation": file_fact(Path(__file__)),
    }
    mask_manifest = json.loads(MS9_MASK_MANIFEST.read_text())
    mask_rows = [row for row in mask_manifest["masks"] if row["name"] == "final_error"]
    if len(mask_rows) != 1:
        raise Bl1Error("MS9 manifest does not identify exactly one final_error field")
    sources["ms9_final_error"] = verify_file(
        MS9_FINAL_ERROR, str(mask_rows[0]["sha256"]), int(mask_rows[0]["bytes"])
    )
    ledger = np.load(DC1_LEDGER, allow_pickle=False)
    if ledger.shape != (N,) or not math.isclose(
        float(ledger.sum()), EXPECTED["dc1_ledger_bits"], rel_tol=0.0, abs_tol=1e-9
    ):
        raise Bl1Error("the retained same-stream DC1 ideal ledger drifted")
    return {
        "schema": "ddm_bl1_source_binding.v1",
        "shape": [N, HEIGHT, WIDTH],
        "positions": POSITIONS,
        "axis": "[macOS-CPU advisory / scorer-free shipped-receiver instrumentation]",
        "sources": sources,
    }


def build_decoder(store: Path) -> Path:
    build = store / "retained" / "build"
    build.mkdir(parents=True, exist_ok=True)
    library = build / "rc64_backend.so"
    command = [
        os.environ.get("CC", "cc"),
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        str(RC64_SOURCE),
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True)
    atomic_json(
        build / "BUILD.json",
        {
            "schema": "ddm_bl1_rc64_build.v1",
            "command": command,
            "source": file_fact(RC64_SOURCE),
            "library": file_fact(library),
            "note": "unmodified shipped decoder source; additions exist only in this readout",
        },
    )
    return library


def decoder_state(decoder: object) -> np.ndarray:
    pointer = ctypes.cast(decoder.context, ctypes.POINTER(Rc64DecoderLayout))
    state = pointer.contents
    return np.asarray(
        [state.low, state.high, state.code, state.size, state.bit_position, state.error],
        dtype=np.uint64,
    )


def restore_decoder_state(decoder: object, saved: np.ndarray) -> None:
    values = np.asarray(saved, dtype=np.uint64).reshape(6)
    pointer = ctypes.cast(decoder.context, ctypes.POINTER(Rc64DecoderLayout))
    state = pointer.contents
    if state.size != int(values[3]):
        raise Bl1Error("RC64 resume state is bound to a different payload size")
    if int(values[4]) < 63:
        raise Bl1Error("RC64 resume state precedes decoder initialization")
    state.low = int(values[0])
    state.high = int(values[1])
    state.code = int(values[2])
    state.bit_position = int(values[4])
    state.error = int(values[5])
    if not np.array_equal(decoder_state(decoder), values):
        raise Bl1Error("RC64 decoder state restore did not land byte-for-byte")


def rc64_costs(coding: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact integer-frequency and float32-input costs for shipped RC64."""
    values = np.ascontiguousarray(coding, dtype=np.float32)
    symbols64 = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape != (symbols64.size, CLASSES):
        raise Bl1Error("coding rows do not have the shipped [N,5] shape")
    frequencies = (values.astype(np.float64) * TOTAL_FREQUENCY).astype(np.uint64)
    np.maximum(frequencies, 1, out=frequencies)
    winners = values.argmax(axis=1)
    sums = frequencies.sum(axis=1, dtype=np.uint64)
    balance = TOTAL_FREQUENCY - sums.astype(np.int64)
    row_index = np.arange(symbols64.size)
    adjusted = frequencies[row_index, winners].astype(np.int64) + balance
    if np.any(adjusted <= 0) or np.any(adjusted >= TOTAL_FREQUENCY):
        raise Bl1Error("RC64 winner-frequency balance is invalid")
    frequencies[row_index, winners] = adjusted.astype(np.uint64)
    selected_frequency = frequencies[row_index, symbols64]
    selected_probability = values[row_index, symbols64].astype(np.float64)
    frequency_cost = 31.0 - np.log2(selected_frequency.astype(np.float64))
    probability_cost = -np.log2(selected_probability)
    return frequency_cost.astype(FIELD_DTYPE), probability_cost.astype(FIELD_DTYPE)


def stage_paths(store: Path, start: int, end: int) -> dict[str, Path]:
    root = store / "retained" / "stages" / f"frames_{start:04d}_{end - 1:04d}"
    return {
        "root": root,
        "frequency": root / "rc64_frequency_cost_bits.f64.npy",
        "probability": root / "probability_input_cost_bits.f64.npy",
        "decoded": root / "decoded_tokens.u8.npy",
        "model64": root / "model_float64_frame_bits.f64.npy",
        "state": root / "receiver_state.npz",
        "receipt": root / "RECEIPT.json",
    }


def validate_stage(paths: dict[str, Path], binding: dict[str, object], start: int, end: int) -> dict[str, Any]:
    if not paths["receipt"].is_file():
        raise Bl1Error(f"stage receipt is absent: {paths['receipt']}")
    receipt = json.loads(paths["receipt"].read_text())
    if receipt.get("source_binding") != binding:
        raise Bl1Error(f"stage {start}:{end} is bound to different sources")
    if (receipt.get("frame_start"), receipt.get("frame_end")) != (start, end):
        raise Bl1Error(f"stage {start}:{end} frame bounds drifted")
    for key in ("frequency", "probability", "decoded", "model64", "state"):
        expected = receipt["artifacts"][key]
        verify_file(paths[key], str(expected["sha256"]), int(expected["bytes"]))
    expected_shape = (end - start, HEIGHT, WIDTH)
    for key in ("frequency", "probability"):
        payload = np.load(paths[key], mmap_mode="r", allow_pickle=False)
        if payload.shape != expected_shape or payload.dtype != FIELD_DTYPE:
            raise Bl1Error(f"stage field shape/dtype drift: {paths[key]}")
    decoded = np.load(paths["decoded"], mmap_mode="r", allow_pickle=False)
    if decoded.shape != expected_shape or decoded.dtype != np.uint8:
        raise Bl1Error(f"stage decoded field shape/dtype drift: {paths['decoded']}")
    return receipt


def completed_stages(store: Path, binding: dict[str, object]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        paths = stage_paths(store, start, end)
        if not paths["receipt"].exists():
            break
        receipts.append(validate_stage(paths, binding, start, end))
    later = list((store / "retained" / "stages").glob("frames_*/RECEIPT.json"))
    if len(later) != len(receipts):
        raise Bl1Error("stage receipts are not a contiguous prefix; refuse ambiguous resume")
    return receipts


def save_stage(
    paths: dict[str, Path],
    binding: dict[str, object],
    start: int,
    end: int,
    frequency_cost: np.ndarray,
    probability_cost: np.ndarray,
    decoded: np.ndarray,
    model64_frame_bits: np.ndarray,
    receiver_state: dict[str, np.ndarray],
    decoder_snapshot: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_npy(paths["frequency"], np.asarray(frequency_cost, dtype=FIELD_DTYPE))
    atomic_npy(paths["probability"], np.asarray(probability_cost, dtype=FIELD_DTYPE))
    atomic_npy(paths["decoded"], np.asarray(decoded, dtype=np.uint8))
    atomic_npy(paths["model64"], np.asarray(model64_frame_bits, dtype=FIELD_DTYPE))
    state_arrays = {
        "schema": np.frombuffer(CHECKPOINT_SCHEMA.encode(), dtype=np.uint8),
        "frame_end": np.asarray([end], dtype=np.int64),
        "decoder": np.asarray(decoder_snapshot, dtype=np.uint64),
        "previous": np.asarray(decoded[-1], dtype=np.uint8),
        **{f"corrector__{key}": value for key, value in receiver_state.items()},
    }
    atomic_npz(paths["state"], state_arrays)
    artifacts = {key: file_fact(paths[key]) for key in ("frequency", "probability", "decoded", "model64", "state")}
    receipt: dict[str, Any] = {
        "schema": "ddm_bl1_stage_receipt.v1",
        "source_binding": binding,
        "frame_start": start,
        "frame_end": end,
        "positions": (end - start) * PLANE,
        "frequency_cost_bits": float(np.asarray(frequency_cost, dtype=np.float64).sum()),
        "probability_input_cost_bits": float(np.asarray(probability_cost, dtype=np.float64).sum()),
        "model_float64_cost_bits": float(np.asarray(model64_frame_bits, dtype=np.float64).sum()),
        "decoded_sha256_logical_raw": hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest(),
        "decoder_bit_position": int(decoder_snapshot[4]),
        "corrector_state_arrays": len(receiver_state),
        "elapsed_seconds": elapsed,
        "artifacts": artifacts,
    }
    atomic_json(paths["receipt"], receipt)
    return receipt


def load_receiver(binding: dict[str, object], library: Path) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(RUNTIME_ROOT / "cpr1"))
    sys.path.insert(0, str(RUNTIME_ROOT))
    renderer = importlib.import_module("cpr1.inflate")
    residual = importlib.import_module("runtime.residual_archive")
    rc64 = importlib.import_module("runtime.entropy.rc64")
    free_corrector = importlib.import_module("runtime.free_corrector")
    hpac_inference = importlib.import_module("runtime.hpac_inference")
    jg2 = importlib.import_module("experiments.ddm_jg2_tail_reencode")
    parts = residual.read_residual_archive(DX2_ARCHIVE)
    if hashlib.sha256(parts.token_stream).hexdigest() != EXPECTED["stream_sha256"]:
        raise Bl1Error("token stream extracted by the shipped parser drifted")
    if len(parts.token_stream) != EXPECTED["stream_bytes"]:
        raise Bl1Error("token stream extracted by the shipped parser has wrong length")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(RUNTIME_ROOT / "cpr1")(model, HEIGHT, WIDTH)
    hpac_inference.optimize_sparse_evaluator(sparse)
    plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)).astype(np.int64)
        plans.append((torch.from_numpy(flat).to(device), flat))
    if len(plans) != GROUPS or sum(len(plan[1]) for plan in plans) != PLANE:
        raise Bl1Error("shipped group masks do not partition the scorer grid")
    seen = np.concatenate([plan[1] for plan in plans])
    if np.unique(seen).size != PLANE:
        raise Bl1Error("shipped group masks overlap or omit raster positions")
    corrector = free_corrector.FreeCorrector(PLANE)
    cold_corrector = free_corrector.FreeCorrector(PLANE)
    decoder = rc64.NativeDecoder(library, parts.token_stream)
    return {
        "torch": torch,
        "renderer": renderer,
        "residual": residual,
        "parts": parts,
        "model": model,
        "sparse": sparse,
        "plans": plans,
        "corrector": corrector,
        "cold_corrector": cold_corrector,
        "decoder": decoder,
        "jg2": jg2,
        "device": device,
        "binding": binding,
    }


def resume_receiver(runtime: dict[str, Any], receipts: list[dict[str, Any]], store: Path) -> tuple[int, Any]:
    torch = runtime["torch"]
    if not receipts:
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=runtime["device"])
        return 0, previous
    end = int(receipts[-1]["frame_end"])
    paths = stage_paths(store, end - STAGE_FRAMES, end)
    with np.load(paths["state"], allow_pickle=False) as payload:
        schema = bytes(payload["schema"]).decode()
        if schema != CHECKPOINT_SCHEMA or int(payload["frame_end"][0]) != end:
            raise Bl1Error("receiver resume checkpoint schema/frame drifted")
        corrector_payload = {
            key.removeprefix("corrector__"): payload[key].copy()
            for key in payload.files
            if key.startswith("corrector__")
        }
        runtime["jg2"].load_corrector_state(runtime["corrector"], corrector_payload)
        restore_decoder_state(runtime["decoder"], payload["decoder"])
        previous_np = np.asarray(payload["previous"], dtype=np.uint8).copy()
    previous = torch.from_numpy(previous_np.astype(np.int64)).reshape(1, HEIGHT, WIDTH).to(runtime["device"])
    return end, previous


def run_decode(store: Path, binding: dict[str, object], library: Path) -> list[dict[str, Any]]:
    import torch

    receipts = completed_stages(store, binding)
    runtime = load_receiver(binding, library)
    start_frame, previous = resume_receiver(runtime, receipts, store)
    truth = np.memmap(TO2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    residual = runtime["residual"]
    renderer = runtime["renderer"]
    parts = runtime["parts"]
    model = runtime["model"]
    sparse = runtime["sparse"]
    plans = runtime["plans"]
    corrector = runtime["corrector"]
    decoder = runtime["decoder"]
    device = runtime["device"]
    jg2 = runtime["jg2"]
    ledger = np.load(DC1_LEDGER, allow_pickle=False)

    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            stage_started = time.perf_counter()
            shape = (stage_end - stage_start, HEIGHT, WIDTH)
            frequency_cost = np.empty(shape, dtype=FIELD_DTYPE)
            probability_cost = np.empty(shape, dtype=FIELD_DTYPE)
            decoded = np.empty(shape, dtype=np.uint8)
            model64_frame_bits = np.zeros(stage_end - stage_start, dtype=FIELD_DTYPE)
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
                index = torch.tensor([frame], dtype=torch.long, device=device)
                current = torch.zeros_like(previous)
                context = model.prepare_frame_context(index, previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                corrector.begin_frame(boundary)
                for group, (device_positions, flat_positions) in enumerate(plans):
                    base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * CLASSES + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                    state = corrector.group_state(probability, predicted, flat_positions)
                    coding = np.asarray(corrector.coding_row(state), dtype=np.float64)
                    symbols = decoder.decode(coding).astype(np.int64)
                    expected = np.asarray(truth[frame]).reshape(-1)[flat_positions].astype(np.int64)
                    if not np.array_equal(symbols, expected):
                        raise Bl1Error(f"shipped decoder diverged from TO2 at frame={frame} group={group}")
                    freq_bits, prob_bits = rc64_costs(coding, symbols)
                    frequency_cost[offset].reshape(-1)[flat_positions] = freq_bits
                    probability_cost[offset].reshape(-1)[flat_positions] = prob_bits
                    model64_frame_bits[offset] += float(
                        (-np.log2(coding[np.arange(symbols.size), symbols])).sum()
                    )
                    corrector.observe(state, symbols)
                    current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
                decoded[offset] = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                if not np.array_equal(decoded[offset], truth[frame]):
                    raise Bl1Error(f"shipped decoded frame {frame} differs from TO2")
                corrector.end_frame(decoded[offset].reshape(-1))
                if not math.isclose(
                    float(model64_frame_bits[offset]),
                    float(ledger[frame]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise Bl1Error(f"frame {frame} model cost disagrees with the retained ideal ledger")
                previous = current
            state = jg2.corrector_state(corrector)
            lost = jg2.uncaptured_divergent_state(corrector, runtime["cold_corrector"], set(state))
            if lost:
                raise Bl1Error(f"stage checkpoint would lose adaptive corrector state: {lost[:8]}")
            snapshot = decoder_state(decoder)
            paths = stage_paths(store, stage_start, stage_end)
            receipt = save_stage(
                paths,
                binding,
                stage_start,
                stage_end,
                frequency_cost,
                probability_cost,
                decoded,
                model64_frame_bits,
                state,
                snapshot,
                time.perf_counter() - stage_started,
            )
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "stage": [stage_start, stage_end],
                        "frequency_bits": receipt["frequency_cost_bits"],
                        "decoder_bit_position": receipt["decoder_bit_position"],
                        "receipt": str(paths["receipt"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return receipts


def assemble_fields(store: Path, binding: dict[str, object]) -> dict[str, dict[str, object]]:
    receipts = completed_stages(store, binding)
    if len(receipts) != N // STAGE_FRAMES:
        raise Bl1Error("cannot finalize before all 30 stage checkpoints are present")
    output = store / "retained" / "fields"
    output.mkdir(parents=True, exist_ok=True)
    frequency_path = output / "position_rc64_frequency_cost_bits.f64le.bin"
    probability_path = output / "position_probability_input_cost_bits.f64le.bin"
    decoded_path = output / "decoded_tokens_instrumented.u8"
    model64_path = output / "model_float64_frame_bits.f64.npy"
    frequency_arrays: list[np.ndarray] = []
    probability_arrays: list[np.ndarray] = []
    decoded_arrays: list[np.ndarray] = []
    model64_arrays: list[np.ndarray] = []
    for receipt in receipts:
        start, end = int(receipt["frame_start"]), int(receipt["frame_end"])
        paths = stage_paths(store, start, end)
        frequency_arrays.append(np.load(paths["frequency"], mmap_mode="r", allow_pickle=False))
        probability_arrays.append(np.load(paths["probability"], mmap_mode="r", allow_pickle=False))
        decoded_arrays.append(np.load(paths["decoded"], mmap_mode="r", allow_pickle=False))
        model64_arrays.append(np.load(paths["model64"], allow_pickle=False))
    atomic_raw(frequency_path, frequency_arrays)
    atomic_raw(probability_path, probability_arrays)
    atomic_raw(decoded_path, decoded_arrays)
    atomic_npy(model64_path, np.concatenate(model64_arrays))
    fields = {
        "frequency_cost": file_fact(frequency_path),
        "probability_cost": file_fact(probability_path),
        "decoded": file_fact(decoded_path),
        "model64_frame_bits": file_fact(model64_path),
    }
    if fields["frequency_cost"]["bytes"] != POSITIONS * FIELD_DTYPE.itemsize:
        raise Bl1Error("assembled primary cost field has wrong byte size")
    if fields["probability_cost"]["bytes"] != POSITIONS * FIELD_DTYPE.itemsize:
        raise Bl1Error("assembled probability field has wrong byte size")
    if fields["decoded"]["sha256"] != EXPECTED["tokens_sha256"]:
        raise Bl1Error("instrumented full decode does not reproduce TO2 byte-for-byte")
    return fields


def packed_frame(path: Path, frame: int) -> np.ndarray:
    bytes_per_frame = PLANE // 8
    packed = np.memmap(path, dtype=np.uint8, mode="r", offset=frame * bytes_per_frame, shape=(bytes_per_frame,))
    return np.unpackbits(packed, bitorder="little", count=PLANE).astype(bool, copy=False)


def weighted_gini(sorted_cost: np.ndarray, total: float) -> float:
    weighted = 0.0
    chunk = 1 << 22
    for start in range(0, sorted_cost.size, chunk):
        stop = min(start + chunk, sorted_cost.size)
        indices = np.arange(start + 1, stop + 1, dtype=np.float64)
        weighted += float(np.dot(indices, sorted_cost[start:stop]))
    n = sorted_cost.size
    return 2.0 * weighted / (n * total) - (n + 1.0) / n


def make_top_mask(cost: np.ndarray, store: Path, fraction: float, threshold: float, strict_count: int) -> dict[str, object]:
    count = math.ceil(POSITIONS * fraction)
    ties_needed = count - strict_count
    fraction_label = f"{fraction * 100:g}".replace(".", "p")
    path = store / "retained" / "fields" / f"top_{fraction_label}pct_positions.n600.packbits"
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    selected = 0
    try:
        with tmp.open("wb") as handle:
            for frame in range(N):
                values = np.asarray(cost[frame]).reshape(-1)
                mask = values > threshold
                if ties_needed:
                    equal = np.flatnonzero(values == threshold)
                    take = min(ties_needed, equal.size)
                    mask[equal[:take]] = True
                    ties_needed -= take
                selected += int(mask.sum())
                handle.write(np.packbits(mask, bitorder="little").tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    if selected != count or ties_needed:
        raise Bl1Error("deterministic top-position tie resolution failed")
    return {**file_fact(path), "positions": count, "fraction": fraction, "threshold_bits": threshold}


def analyze_fields(store: Path, fields: dict[str, dict[str, object]], binding: dict[str, object]) -> dict[str, Any]:
    frequency = np.memmap(
        fields["frequency_cost"]["path"], dtype=FIELD_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH)
    )
    probability = np.memmap(
        fields["probability_cost"]["path"], dtype=FIELD_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH)
    )
    tokens = np.memmap(fields["decoded"]["path"], dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, HEIGHT, WIDTH) or gt.dtype != np.uint8:
        raise Bl1Error("GT field shape/dtype is not joinable")
    total_frequency_bits = float(frequency.sum(dtype=np.float64))
    total_probability_bits = float(probability.sum(dtype=np.float64))
    model64 = np.load(fields["model64_frame_bits"]["path"], allow_pickle=False)
    total_model64_bits = float(model64.sum())
    overhead = STREAM_BITS - total_frequency_bits
    if not (0.0 <= overhead < TERMINAL_OVERHEAD_BOUND_BITS):
        raise Bl1Error(
            f"VOID: primary cost sum {total_frequency_bits} does not reconcile to {STREAM_BITS} bits"
        )
    if not math.isclose(total_model64_bits, EXPECTED["dc1_ledger_bits"], rel_tol=0.0, abs_tol=1e-7):
        raise Bl1Error("float64 model replay does not reproduce the same-stream retained ideal ledger")

    sorted_cost = np.sort(np.asarray(frequency).reshape(-1))
    fractions = (0.001, 0.01, 0.05, 0.10, 0.50)
    concentration = []
    thresholds: dict[float, tuple[float, int]] = {}
    for fraction in fractions:
        count = math.ceil(POSITIONS * fraction)
        threshold = float(sorted_cost[-count])
        strict_count = POSITIONS - int(np.searchsorted(sorted_cost, threshold, side="right"))
        bits = float(sorted_cost[-count:].sum(dtype=np.float64))
        concentration.append(
            {
                "top_position_fraction": fraction,
                "positions": count,
                "position_denominator": POSITIONS,
                "bits": bits,
                "bit_denominator": total_frequency_bits,
                "bit_fraction": bits / total_frequency_bits,
                "bytes_equivalent": bits / 8.0,
                "threshold_bits": threshold,
            }
        )
        thresholds[fraction] = (threshold, strict_count)
    top1 = make_top_mask(frequency, store, 0.01, *thresholds[0.01])
    gini = weighted_gini(sorted_cost, total_frequency_bits)

    histogram_edges = np.asarray(
        [
            0.0,
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
        ],
        dtype=np.float64,
    )
    counts, _ = np.histogram(sorted_cost, bins=histogram_edges)
    bin_ids = np.searchsorted(histogram_edges, sorted_cost, side="right") - 1
    bit_sums = np.bincount(bin_ids, weights=sorted_cost, minlength=len(histogram_edges) - 1)
    histogram = [
        {
            "low_bits_inclusive": float(histogram_edges[index]),
            "high_bits_exclusive": float(histogram_edges[index + 1]),
            "positions": int(counts[index]),
            "position_denominator": POSITIONS,
            "position_fraction": int(counts[index]) / POSITIONS,
            "bits": float(bit_sums[index]),
            "bit_denominator": total_frequency_bits,
            "bit_fraction": float(bit_sums[index]) / total_frequency_bits,
        }
        for index in range(len(counts))
    ]
    del bin_ids, sorted_cost

    class_count = np.zeros(CLASSES, dtype=np.int64)
    class_bits = np.zeros(CLASSES, dtype=np.float64)
    class_top_count = np.zeros(CLASSES, dtype=np.int64)
    class_top_bits = np.zeros(CLASSES, dtype=np.float64)
    token_count = np.zeros(CLASSES, dtype=np.int64)
    token_bits = np.zeros(CLASSES, dtype=np.float64)
    frame_bits = np.zeros(N, dtype=np.float64)
    frame_top_bits = np.zeros(N, dtype=np.float64)
    frame_top_count = np.zeros(N, dtype=np.int64)
    group_bits = np.zeros(GROUPS, dtype=np.float64)
    group_count = np.zeros(GROUPS, dtype=np.int64)
    group_top_bits = np.zeros(GROUPS, dtype=np.float64)
    group_top_count = np.zeros(GROUPS, dtype=np.int64)
    spatial_bits = np.zeros(PLANE, dtype=np.float64)
    error_bits = 0.0
    error_count = 0
    error_top_count = 0
    error_top_bits = 0.0
    yy, xx = np.indices((HEIGHT, WIDTH))
    group_index = ((xx % 64) + 2 * (yy % 64)).reshape(-1)
    top_path = Path(str(top1["path"]))
    for frame in range(N):
        costs = np.asarray(frequency[frame]).reshape(-1)
        gt_frame = np.asarray(gt[frame]).reshape(-1)
        token_frame = np.asarray(tokens[frame]).reshape(-1)
        top_mask = packed_frame(top_path, frame)
        error_mask = packed_frame(MS9_FINAL_ERROR, frame)
        frame_bits[frame] = float(costs.sum())
        frame_top_bits[frame] = float(costs[top_mask].sum())
        frame_top_count[frame] = int(top_mask.sum())
        spatial_bits += costs
        class_count += np.bincount(gt_frame, minlength=CLASSES)
        class_bits += np.bincount(gt_frame, weights=costs, minlength=CLASSES)
        class_top_count += np.bincount(gt_frame[top_mask], minlength=CLASSES)
        class_top_bits += np.bincount(
            gt_frame[top_mask], weights=costs[top_mask], minlength=CLASSES
        )
        token_count += np.bincount(token_frame, minlength=CLASSES)
        token_bits += np.bincount(token_frame, weights=costs, minlength=CLASSES)
        group_count += np.bincount(group_index, minlength=GROUPS)
        group_bits += np.bincount(group_index, weights=costs, minlength=GROUPS)
        group_top_count += np.bincount(group_index[top_mask], minlength=GROUPS)
        group_top_bits += np.bincount(group_index[top_mask], weights=costs[top_mask], minlength=GROUPS)
        error_count += int(error_mask.sum())
        error_bits += float(costs[error_mask].sum())
        both = error_mask & top_mask
        error_top_count += int(both.sum())
        error_top_bits += float(costs[both].sum())
    if error_count != 23_757:
        raise Bl1Error("MS9 final-error mask count drifted from its exact receipt")

    aggregates = store / "retained" / "aggregates"
    atomic_npy(aggregates / "frame_bits.f64.npy", frame_bits)
    atomic_npy(aggregates / "group_bits.f64.npy", group_bits)
    atomic_npy(aggregates / "spatial_site_bits.f64.npy", spatial_bits.reshape(HEIGHT, WIDTH))
    aggregate_facts = {
        "frame_bits": file_fact(aggregates / "frame_bits.f64.npy"),
        "group_bits": file_fact(aggregates / "group_bits.f64.npy"),
        "spatial_site_bits": file_fact(aggregates / "spatial_site_bits.f64.npy"),
    }

    class_rows = [
        {
            "class_id": index,
            "class_name": CLASS_NAMES[index],
            "positions": int(class_count[index]),
            "position_denominator": POSITIONS,
            "area_fraction": int(class_count[index]) / POSITIONS,
            "bits": float(class_bits[index]),
            "bit_denominator": total_frequency_bits,
            "bit_fraction": float(class_bits[index]) / total_frequency_bits,
            "bits_per_position": float(class_bits[index]) / int(class_count[index]),
            "bpp_enrichment_over_all_positions": (
                float(class_bits[index]) / int(class_count[index])
            ) / (total_frequency_bits / POSITIONS),
            "top_1pct_positions": int(class_top_count[index]),
            "top_1pct_position_denominator": int(top1["positions"]),
            "top_1pct_bits": float(class_top_bits[index]),
            "top_1pct_bit_denominator": float(
                next(
                    row["bits"]
                    for row in concentration
                    if row["top_position_fraction"] == 0.01
                )
            ),
        }
        for index in range(CLASSES)
    ]
    token_rows = [
        {
            "token_class": index,
            "positions": int(token_count[index]),
            "position_denominator": POSITIONS,
            "bits": float(token_bits[index]),
            "bit_denominator": total_frequency_bits,
            "bits_per_position": float(token_bits[index]) / int(token_count[index]),
        }
        for index in range(CLASSES)
    ]
    frame_rows = [
        {
            "frame": frame,
            "positions": PLANE,
            "position_denominator": PLANE,
            "bits": float(frame_bits[frame]),
            "bit_denominator": total_frequency_bits,
            "bits_per_position": float(frame_bits[frame]) / PLANE,
            "top_1pct_positions": int(frame_top_count[frame]),
            "top_1pct_bits": float(frame_top_bits[frame]),
        }
        for frame in range(N)
    ]
    time_blocks = []
    for start in range(0, N, 100):
        end = start + 100
        bits = float(frame_bits[start:end].sum())
        time_blocks.append(
            {
                "frames": [start, end],
                "positions": (end - start) * PLANE,
                "position_denominator": POSITIONS,
                "bits": bits,
                "bit_denominator": total_frequency_bits,
                "bits_per_position": bits / ((end - start) * PLANE),
                "prefix_bits_per_position": float(frame_bits[:end].sum()) / (end * PLANE),
            }
        )
    group_rows = [
        {
            "group": group,
            "positions": int(group_count[group]),
            "position_denominator": POSITIONS,
            "bits": float(group_bits[group]),
            "bit_denominator": total_frequency_bits,
            "bits_per_position": float(group_bits[group]) / int(group_count[group]),
            "top_1pct_positions": int(group_top_count[group]),
            "top_1pct_bits": float(group_top_bits[group]),
        }
        for group in range(GROUPS)
    ]
    top_spatial = np.argsort(spatial_bits)[-100:][::-1]
    spatial_rows = [
        {
            "raster_index": int(index),
            "y": int(index // WIDTH),
            "x": int(index % WIDTH),
            "group": int(group_index[index]),
            "positions": N,
            "position_denominator": POSITIONS,
            "bits": float(spatial_bits[index]),
            "bit_denominator": total_frequency_bits,
            "bits_per_position": float(spatial_bits[index]) / N,
        }
        for index in top_spatial
    ]
    top_frames = sorted(frame_rows, key=lambda row: (-row["bits"], row["frame"]))[:20]
    top_groups = sorted(group_rows, key=lambda row: (-row["bits_per_position"], row["group"]))[:20]
    top1_row = next(row for row in concentration if row["top_position_fraction"] == 0.01)
    top1_fraction = float(top1_row["bit_fraction"])
    if top1_fraction > 0.50:
        adjudication = "STRONGLY_CONCENTRATED"
    elif top1_fraction < 0.25:
        adjudication = "DIFFUSE_REGISTERED_FALSIFIER"
    else:
        adjudication = "INTERMEDIATE_NOT_REGISTERED_EXTREME"
    return {
        "schema": "ddm_bl1_per_position_bit_allocation.v1",
        "status": "MEASURED_RECONCILED",
        "axis": binding["axis"],
        "score_claim": False,
        "source_binding": binding,
        "shape": [N, HEIGHT, WIDTH],
        "positions": POSITIONS,
        "stream": {
            "bytes": EXPECTED["stream_bytes"],
            "bits": STREAM_BITS,
            "bits_per_position": STREAM_BITS / POSITIONS,
            "sha256": EXPECTED["stream_sha256"],
        },
        "cost_accounting": {
            "primary_definition": "-log2(integer_frequency/2**31) using the exact shipped C quantizer",
            "primary_frequency_cost_bits": total_frequency_bits,
            "probability_input_cost_bits": total_probability_bits,
            "model_float64_cost_bits": total_model64_bits,
            "same_stream_dc1_ledger_bits": EXPECTED["dc1_ledger_bits"],
            "stream_minus_primary_bits": overhead,
            "stream_minus_primary_bytes": overhead / 8.0,
            "reconciliation_bound_bits": TERMINAL_OVERHEAD_BOUND_BITS,
            "bound_basis": (
                "classic finite-precision arithmetic interval overhead <2 bits plus "
                "the shipped encoder's final partial-byte padding <7 bits"
            ),
            "passed": True,
            "decoder_final_bit_position": int(completed_stages(store, binding)[-1]["decoder_bit_position"]),
        },
        "fields": fields,
        "aggregate_fields": aggregate_facts,
        "top_1pct_mask": top1,
        "distribution": {
            "minimum_bits": float(frequency.min()),
            "maximum_bits": float(frequency.max()),
            "mean_bits_per_position": total_frequency_bits / POSITIONS,
            "gini": gini,
            "histogram": histogram,
            "concentration": concentration,
        },
        "gt_class_join": class_rows,
        "decoded_token_class_join": token_rows,
        "time_join": {
            "frames": frame_rows,
            "hundred_frame_blocks": time_blocks,
            "top_frames_by_bits": top_frames,
        },
        "spatial_join": {
            "group_formula": "g=(x mod 64)+2*(y mod 64)",
            "groups": group_rows,
            "top_groups_by_bpp": top_groups,
            "top_100_raster_sites_by_bits": spatial_rows,
        },
        "ms9_final_error_join": {
            "evidence_axis": "[contest-CUDA T4 component-only exact field replay]",
            "error_positions": error_count,
            "position_denominator": POSITIONS,
            "error_position_fraction": error_count / POSITIONS,
            "bits_on_error_positions": error_bits,
            "bit_denominator": total_frequency_bits,
            "bit_fraction": error_bits / total_frequency_bits,
            "bits_per_error_position": error_bits / error_count,
            "bpp_enrichment_over_all_positions": (error_bits / error_count) / (total_frequency_bits / POSITIONS),
            "top_1pct_error_positions": error_top_count,
            "top_1pct_position_denominator": int(top1["positions"]),
            "top_1pct_error_bits": error_top_bits,
            "causal_warning": "spatial coincidence only; no scorer was run and no intervention was measured",
        },
        "adjudication": {
            "classification": adjudication,
            "prior_prediction_top_1pct_over_50pct": top1_fraction > 0.50,
            "registered_diffuse_falsifier_top_1pct_under_25pct": top1_fraction < 0.25,
            "target_positions": int(top1["positions"]),
            "target_bits": float(top1_row["bits"]),
            "target_bytes_equivalent": float(top1_row["bytes_equivalent"]),
        },
    }


def write_manifest(store: Path, result_path: Path, binding: dict[str, object]) -> None:
    artifacts = []
    for path in sorted((store / "retained").rglob("*")):
        if path.is_file() and not path.name.startswith("._"):
            artifacts.append(file_fact(path))
    atomic_json(
        store / "MANIFEST.json",
        {
            "schema": "ddm_bl1_manifest.v1",
            "source_binding": binding,
            "result": file_fact(result_path),
            "artifacts": artifacts,
            "retention": (
                "All 30 stage payloads and complete receiver states are preserved. "
                "Final fields and every derived field are on Vertigo; no cleanup is authorized "
                "without a replacement custody manifest."
            ),
        },
    )


def self_test() -> None:
    coding = np.asarray([[0.5, 0.25, 0.125, 0.0625, 0.0625]], dtype=np.float64)
    frequency, probability = rc64_costs(coding, np.asarray([2], dtype=np.int64))
    if frequency[0] != 3.0 or probability[0] != 3.0:
        raise AssertionError("power-of-two cost control failed")
    with tempfile.TemporaryDirectory(prefix="ddm_bl1_selftest_") as tmp:
        root = Path(tmp)
        library = root / "rc64_backend.so"
        subprocess.run(
            ["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(RC64_SOURCE), "-o", str(library)],
            check=True,
        )
        sys.path.insert(0, str(RUNTIME_ROOT))
        rc64 = importlib.import_module("runtime.entropy.rc64")
        payload = bytes([0xA5] * 16)
        decoder = rc64.NativeDecoder(library, payload)
        saved = decoder_state(decoder)
        if int(saved[3]) != len(payload) or int(saved[4]) != 63:
            raise AssertionError("decoder struct layout control failed")
        clone = rc64.NativeDecoder(library, payload)
        restore_decoder_state(clone, saved)
        if not np.array_equal(saved, decoder_state(clone)):
            raise AssertionError("decoder snapshot control failed")
        rows = np.repeat(coding, 8, axis=0)
        if not np.array_equal(decoder.decode(rows), clone.decode(rows)):
            raise AssertionError("resumed decoder symbol control failed")
        if not np.array_equal(decoder_state(decoder), decoder_state(clone)):
            raise AssertionError("resumed decoder trajectory control failed")
    print("ddm_bl1 self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    store = args.store.resolve()
    if not store.is_relative_to(VERTIGO_ROOT.resolve()):
        raise Bl1Error(f"output must remain on the Vertigo SSD tier: {VERTIGO_ROOT}")
    if store.exists() and not args.resume:
        raise Bl1Error("output exists; use --resume after verifying its retained receipts")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MIN_FREE_BYTES:
        raise Bl1Error(f"storage preflight failed: {free} free bytes < {MIN_FREE_BYTES}")
    binding = source_binding()
    source_dir = store / "retained" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = source_dir / Path(__file__).name
    shutil.copyfile(__file__, source_snapshot)
    atomic_json(
        store / "PREFLIGHT.json",
        {
            "schema": "ddm_bl1_preflight.v1",
            "source_binding": binding,
            "store": str(store),
            "free_bytes_before": free,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "estimated_artifact_bytes": ESTIMATED_ARTIFACT_BYTES,
            "reserve_bytes": RESERVE_BYTES,
            "argv": sys.argv,
            "source_snapshot": file_fact(source_snapshot),
            "stage_frames": STAGE_FRAMES,
            "determinism": "no RNG; CPU threads=4; interop threads=1; exact source and payload pins",
        },
    )
    library = build_decoder(store)
    run_decode(store, binding, library)
    fields = assemble_fields(store, binding)
    result = analyze_fields(store, fields, binding)
    result_path = store / "RESULT.json"
    atomic_json(result_path, result)
    write_manifest(store, result_path, binding)
    print(json.dumps({"result": file_fact(result_path), "manifest": file_fact(store / "MANIFEST.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
