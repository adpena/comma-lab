#!/usr/bin/env python3
"""Retain and decompose a GB1-lineage token stream's wrong-indicator cost.

This scorer-free instrument replays a copied, fingerprint-bound Python receiver.  For every
decoded token it retains the exact RC64 integer-frequency decomposition

    selected-symbol bits = is-coder-argmax-wrong bits + which-class-given-wrong bits.

Twenty-pair stages retain the full per-position ledger and complete receiver
state.  GB1 inputs are the defaults; explicit custody flags let Stage B bind a
selected moved object without editing code.  The n600 analysis adds G4
image/xi-proxy stationarity tags and prices a
decoder-known class x margin x fixed-tile oracle bound.  It never edits the
sealed GB1 tree, emits a candidate, invokes a scorer, or changes counted bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from experiments import ddm_bl1_per_position_bit_allocation as bl1  # noqa: E402
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_g4_spatial_stationarity import (  # noqa: E402
    build_xi_tracks,
    stratum_masks,
    transition_codes,
)

SOURCE_RUNTIME = Path(
    "/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/runtime_fire_v1"
)
DEFAULT_STORE = Path(
    "/Volumes/APDataStore/pact/ddm_wh1_wrong_half_decomposition/measurement_v1"
)
TRUTH = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/"
    "retained/fields/decoded_tokens_instrumented.u8"
)
GT_FIELD = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
V12_RECEIPT = REPO / (
    ".omx/research/ddm_v12_obligation_n600_20260722T161517Z/"
    "ddm_v12_obligation_search_n600_receipt.json"
)

N, HEIGHT, WIDTH = 600, 384, 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
CLASSES = 5
TOTAL_FREQUENCY = 1 << 31
STAGE_FRAMES = 20
MARGIN_EDGES = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, np.inf])
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
G4_NAMES = ("STATIC_IN_IMAGE", "STATIC_IN_XI_PROXY", "TRANSIENT", "CORRECT")
GEOMETRIC_NAMES = ("lane_corridor", "movable_band", "hood_rim", "boundaries")
CHECKPOINT_SCHEMA = "ddm_wh1_receiver_checkpoint.v1"
AXIS = "[macOS-CPU advisory / scorer-free shipped-GB1-coder instrumentation]"
EXPECTED = {
    "archive_bytes": 180_215,
    "archive_sha256": "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4",
    "stream_bytes": 113_624,
    "stream_sha256": "a2e3bfc056fba17c387751866105c2c0568437a8ca211e9a6a3714cad0ed782a",
    "tokens_bytes": POSITIONS,
    "tokens_sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    "gt_bytes": 117_964_928,
    "gt_sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    "v12_receipt_sha256": "eab2ef2478fb07f6a3242781887442c3fc49e9c34e10bd73a93f25d9a0262f0a",
}


class Wh1Error(RuntimeError):
    """Fail-closed custody, replay, checkpoint, or accounting error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_file(path: Path, sha256: str, size: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise Wh1Error(f"required file absent: {path}")
    fact = file_fact(path)
    if fact["sha256"] != sha256 or (size is not None and fact["bytes"] != size):
        raise Wh1Error(f"custody drift: {fact}, expected sha={sha256} bytes={size}")
    return fact


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_runtime_copy(store: Path) -> Path:
    """Make or verify the only runtime tree the instrument is allowed to import."""
    destination = store / "retained/runtime_copy"
    manifest_path = destination / "SOURCE_COPY_MANIFEST.json"
    source_files = sorted(path for path in SOURCE_RUNTIME.rglob("*") if path.is_file())
    source_rows = [
        {"relative_path": str(path.relative_to(SOURCE_RUNTIME)), **file_fact(path)}
        for path in source_files
    ]
    if destination.exists():
        if not manifest_path.is_file():
            raise Wh1Error("runtime copy exists without its source-copy manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_files") != source_rows:
            raise Wh1Error("sealed GB1 source tree changed after runtime-copy creation")
        for row in source_rows:
            copied = destination / row["relative_path"]
            if file_fact(copied)["sha256"] != row["sha256"]:
                raise Wh1Error(f"runtime-copy drift: {copied}")
        return destination
    store.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(store.parent).free < (8 << 30):
        raise Wh1Error("APDataStore storage preflight failed: require 8 GiB free")
    store.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        shutil.copytree(SOURCE_RUNTIME, temporary)
        atomic_json(
            temporary / "SOURCE_COPY_MANIFEST.json",
            {
                "schema": "ddm_wh1_runtime_source_copy.v1",
                "source_root": str(SOURCE_RUNTIME),
                "copy_root": str(destination),
                "source_files": source_rows,
                "policy": "sealed source read-only; all imports and compilation use this copy",
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def source_binding(store: Path, runtime_copy: Path) -> dict[str, Any]:
    archive = verify_file(
        runtime_copy / "archive.zip", EXPECTED["archive_sha256"], EXPECTED["archive_bytes"]
    )
    binding = {
        "schema": "ddm_wh1_source_binding.v1",
        "axis": AXIS,
        "shape": [N, HEIGHT, WIDTH],
        "positions": POSITIONS,
        "sources": {
            "copied_gb1_archive": archive,
            "runtime_copy_manifest": file_fact(runtime_copy / "SOURCE_COPY_MANIFEST.json"),
            "truth_tokens": verify_file(
                TRUTH, EXPECTED["tokens_sha256"], EXPECTED["tokens_bytes"]
            ),
            "dali_gt": verify_file(GT_FIELD, EXPECTED["gt_sha256"], EXPECTED["gt_bytes"]),
            "v12_receipt": verify_file(
                V12_RECEIPT, EXPECTED["v12_receipt_sha256"]
            ),
            "implementation": file_fact(Path(__file__)),
        },
        "store": str(store),
    }
    # Analysis-only fixes must not force a 600-pair re-decode.  If a contiguous
    # replay already exists, retain its exact source binding after proving that
    # every non-implementation source is unchanged and that the old instrument
    # bytes remain available in Git custody.
    first_receipt = stage_paths(store, 0, STAGE_FRAMES)["receipt"]
    if first_receipt.is_file():
        prior = json.loads(first_receipt.read_text(encoding="utf-8"))["source_binding"]
        candidate = json.loads(json.dumps(binding))
        candidate["sources"]["implementation"] = prior["sources"]["implementation"]
        if candidate != prior:
            raise Wh1Error("existing replay sources changed beyond the analysis implementation")
        expected = prior["sources"]["implementation"]
        commits = subprocess.run(
            ["git", "log", "--all", "--format=%H", "--", "tools/token_wrong_half_ledger.py"],
            cwd=REPO, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        historical_match = False
        for commit in commits:
            content = subprocess.run(
                ["git", "show", f"{commit}:tools/token_wrong_half_ledger.py"],
                cwd=REPO, check=True, capture_output=True,
            ).stdout
            if len(content) == int(expected["bytes"]) and hashlib.sha256(content).hexdigest() == expected["sha256"]:
                historical_match = True
                break
        if not historical_match:
            raise Wh1Error("stage implementation bytes are not recoverable from Git custody")
        return prior
    return binding


def build_decoder(store: Path, runtime_copy: Path) -> Path:
    source = runtime_copy / "runtime/entropy/rc64_backend.c"
    library = store / "retained/build/rc64_backend.so"
    library.parent.mkdir(parents=True, exist_ok=True)
    command = [
        os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
        str(source), "-o", str(library),
    ]
    subprocess.run(command, check=True)
    atomic_json(
        library.parent / "BUILD.json",
        {
            "schema": "ddm_wh1_rc64_build.v1",
            "command": command,
            "source": file_fact(source),
            "library": file_fact(library),
            "runtime_root": str(runtime_copy),
        },
    )
    return library


def integer_decomposition(
    coding: np.ndarray, symbols: np.ndarray
) -> dict[str, np.ndarray]:
    """Apply shipped RC64 quantization and split the selected-symbol cost."""
    values = np.ascontiguousarray(coding, dtype=np.float32)
    actual = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape != (actual.size, CLASSES):
        raise Wh1Error("coding rows must have shape [positions,5]")
    frequency = (values.astype(np.float64) * TOTAL_FREQUENCY).astype(np.uint64)
    np.maximum(frequency, 1, out=frequency)
    winner = values.argmax(axis=1).astype(np.int64)
    row = np.arange(actual.size)
    balance = TOTAL_FREQUENCY - frequency.sum(axis=1, dtype=np.uint64).astype(np.int64)
    adjusted = frequency[row, winner].astype(np.int64) + balance
    if np.any(adjusted <= 0) or np.any(adjusted >= TOTAL_FREQUENCY):
        raise Wh1Error("invalid balanced RC64 winner frequency")
    frequency[row, winner] = adjusted.astype(np.uint64)
    selected = frequency[row, actual].astype(np.float64)
    winner_frequency = frequency[row, winner].astype(np.float64)
    miss_frequency = TOTAL_FREQUENCY - winner_frequency
    wrong = actual != winner
    indicator_frequency = np.where(wrong, miss_frequency, winner_frequency)
    indicator = 31.0 - np.log2(indicator_frequency)
    which = np.where(wrong, np.log2(miss_frequency) - np.log2(selected), 0.0)
    total = 31.0 - np.log2(selected)
    if not np.allclose(indicator + which, total, rtol=0.0, atol=2e-12):
        raise Wh1Error("wrong-indicator plus which-class does not close to selected cost")
    runner_frequency = np.partition(frequency, -2, axis=1)[:, -2].astype(np.float64)
    margin = np.log2(winner_frequency) - np.log2(runner_frequency)
    bucket = np.searchsorted(MARGIN_EDGES[1:], margin, side="right").astype(np.uint8)
    return {
        "indicator": indicator.astype("<f8"),
        "which": which.astype("<f8"),
        "predicted": winner.astype(np.uint8),
        "margin": margin.astype("<f4"),
        "margin_bucket": bucket,
    }


def binary_entropy_bits(wrong: np.ndarray | int, count: np.ndarray | int) -> np.ndarray:
    k = np.asarray(wrong, dtype=np.float64)
    n = np.asarray(count, dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(k.shape, n.shape), dtype=np.float64)
    valid = (n > 0) & (k > 0) & (k < n)
    p = np.divide(k, n, out=np.zeros_like(out), where=n > 0)
    out[valid] = -n[valid] * (
        p[valid] * np.log2(p[valid]) + (1.0 - p[valid]) * np.log2(1.0 - p[valid])
    )
    return out


def stage_paths(store: Path, start: int, end: int) -> dict[str, Path]:
    root = store / "retained/stages" / f"frames_{start:04d}_{end - 1:04d}"
    return {
        "root": root,
        "indicator": root / "wrong_indicator_bits.f64.npy",
        "which": root / "which_class_bits.f64.npy",
        "predicted": root / "coder_argmax.u8.npy",
        "decoded": root / "decoded_class.u8.npy",
        "margin": root / "top1_runnerup_margin_bits.f32.npy",
        "bucket": root / "margin_bucket.u8.npy",
        "geometry": root / "g4_geometric_mask.u8.npy",
        "state": root / "receiver_state.npz",
        "receipt": root / "RECEIPT.json",
    }


def validate_stage(paths: dict[str, Path], binding: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if receipt.get("source_binding") != binding or [start, end] != receipt.get("frames"):
        raise Wh1Error(f"stage source/bounds drift: {start}:{end}")
    shape = (end - start, HEIGHT, WIDTH)
    dtypes = {
        "indicator": np.dtype("<f8"), "which": np.dtype("<f8"),
        "predicted": np.dtype("u1"), "decoded": np.dtype("u1"),
        "margin": np.dtype("<f4"), "bucket": np.dtype("u1"),
        "geometry": np.dtype("u1"),
    }
    for key, dtype in dtypes.items():
        fact = receipt["artifacts"][key]
        verify_file(paths[key], fact["sha256"], fact["bytes"])
        value = np.load(paths[key], mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != dtype:
            raise Wh1Error(f"stage shape/dtype drift: {paths[key]}")
    state = receipt["artifacts"]["state"]
    verify_file(paths["state"], state["sha256"], state["bytes"])
    return receipt


def completed_stages(store: Path, binding: dict[str, Any]) -> list[dict[str, Any]]:
    receipts = []
    for start in range(0, N, STAGE_FRAMES):
        end = min(N, start + STAGE_FRAMES)
        paths = stage_paths(store, start, end)
        if not paths["receipt"].is_file():
            break
        receipts.append(validate_stage(paths, binding, start, end))
    all_receipts = list((store / "retained/stages").glob("frames_*/RECEIPT.json"))
    if len(receipts) != len(all_receipts):
        raise Wh1Error("stage receipts are not a contiguous prefix")
    return receipts


def save_stage(
    paths: dict[str, Path], binding: dict[str, Any], start: int, end: int,
    fields: dict[str, np.ndarray], receiver_state: dict[str, np.ndarray],
    decoder_state: np.ndarray, elapsed: float,
) -> dict[str, Any]:
    paths["root"].mkdir(parents=True, exist_ok=True)
    for key in ("indicator", "which", "predicted", "decoded", "margin", "bucket", "geometry"):
        atomic_npy(paths[key], fields[key])
    atomic_npz(
        paths["state"],
        {
            "schema": np.frombuffer(CHECKPOINT_SCHEMA.encode(), dtype=np.uint8),
            "frame_end": np.asarray([end], dtype=np.int64),
            "decoder": np.asarray(decoder_state, dtype=np.uint64),
            "previous": np.asarray(fields["decoded"][-1], dtype=np.uint8),
            **{f"corrector__{key}": value for key, value in receiver_state.items()},
        },
    )
    artifacts = {
        key: file_fact(paths[key])
        for key in ("indicator", "which", "predicted", "decoded", "margin", "bucket", "geometry", "state")
    }
    receipt = {
        "schema": "ddm_wh1_stage_receipt.v1",
        "source_binding": binding,
        "frames": [start, end],
        "positions_numerator": (end - start) * PLANE,
        "positions_denominator_n600": POSITIONS,
        "indicator_bits": float(fields["indicator"].sum(dtype=np.float64)),
        "which_bits": float(fields["which"].sum(dtype=np.float64)),
        "wrong_positions": int(np.count_nonzero(fields["predicted"] != fields["decoded"])),
        "decoder_bit_position": int(decoder_state[4]),
        "elapsed_seconds": elapsed,
        "artifacts": artifacts,
    }
    atomic_json(paths["receipt"], receipt)
    return receipt


def _resume_receiver(runtime: dict[str, Any], receipts: list[dict[str, Any]], store: Path) -> tuple[int, Any]:
    torch = runtime["torch"]
    if not receipts:
        return 0, torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=runtime["device"])
    end = int(receipts[-1]["frames"][1])
    with np.load(stage_paths(store, end - STAGE_FRAMES, end)["state"], allow_pickle=False) as payload:
        if bytes(payload["schema"]).decode() != CHECKPOINT_SCHEMA or int(payload["frame_end"][0]) != end:
            raise Wh1Error("receiver resume checkpoint schema/frame drift")
        corrector_state = {
            key.removeprefix("corrector__"): payload[key].copy()
            for key in payload.files if key.startswith("corrector__")
        }
        runtime["jg2"].load_corrector_state(runtime["corrector"], corrector_state)
        bl1.restore_decoder_state(runtime["decoder"], payload["decoder"])
        previous = np.asarray(payload["previous"], dtype=np.uint8).copy()
    tensor = torch.from_numpy(previous.astype(np.int64)).reshape(1, HEIGHT, WIDTH)
    return end, tensor.to(runtime["device"])


def load_receiver(binding: dict[str, Any], runtime_copy: Path, library: Path) -> dict[str, Any]:
    bl1.RUNTIME_ROOT = runtime_copy
    bl1.DX2_ARCHIVE = runtime_copy / "archive.zip"
    bl1.RC64_SOURCE = runtime_copy / "runtime/entropy/rc64_backend.c"
    bl1.EXPECTED["stream_sha256"] = EXPECTED["stream_sha256"]
    bl1.EXPECTED["stream_bytes"] = EXPECTED["stream_bytes"]
    return bl1.load_receiver(binding, library)


def replay(
    store: Path, binding: dict[str, Any], runtime_copy: Path, library: Path,
    max_new_stages: int | None,
) -> list[dict[str, Any]]:
    import torch

    receipts = completed_stages(store, binding)
    runtime = load_receiver(binding, runtime_copy, library)
    start_frame, previous = _resume_receiver(runtime, receipts, store)
    truth = np.memmap(TRUTH, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    residual, renderer, parts = runtime["residual"], runtime["renderer"], runtime["parts"]
    model, sparse, plans = runtime["model"], runtime["sparse"], runtime["plans"]
    corrector, decoder, device, jg2 = (
        runtime["corrector"], runtime["decoder"], runtime["device"], runtime["jg2"]
    )
    new_stages = 0
    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            if max_new_stages is not None and new_stages >= max_new_stages:
                break
            stage_end = min(N, stage_start + STAGE_FRAMES)
            started = time.perf_counter()
            shape = (stage_end - stage_start, HEIGHT, WIDTH)
            fields = {
                "indicator": np.empty(shape, dtype="<f8"),
                "which": np.empty(shape, dtype="<f8"),
                "predicted": np.empty(shape, dtype=np.uint8),
                "decoded": np.empty(shape, dtype=np.uint8),
                "margin": np.empty(shape, dtype="<f4"),
                "bucket": np.empty(shape, dtype=np.uint8),
                "geometry": np.empty(shape, dtype=np.uint8),
            }
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
                index = torch.tensor([frame], dtype=torch.long, device=device)
                current = torch.zeros_like(previous)
                frame_context = model.prepare_frame_context(index, previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                corrector.begin_frame(boundary)
                for group, (device_positions, flat_positions) in enumerate(plans):
                    base_logits = sparse.selected_logits(current, frame_context, group).cpu().numpy()
                    base_argmax = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * CLASSES + base_argmax
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                    state = corrector.group_state(probability, base_argmax, flat_positions)
                    coding = np.asarray(corrector.coding_row(state), dtype=np.float64)
                    symbols = decoder.decode(coding).astype(np.int64)
                    expected = np.asarray(truth[frame]).reshape(-1)[flat_positions].astype(np.int64)
                    if not np.array_equal(symbols, expected):
                        raise Wh1Error(f"GB1 decoder diverged frame={frame} group={group}")
                    split = integer_decomposition(coding, symbols)
                    for key in ("indicator", "which", "predicted", "margin", "margin_bucket"):
                        destination = "bucket" if key == "margin_bucket" else key
                        fields[destination][offset].reshape(-1)[flat_positions] = split[key]
                    corrector.observe(state, symbols)
                    current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
                decoded = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                if not np.array_equal(decoded, truth[frame]):
                    raise Wh1Error(f"GB1 decoded frame {frame} differs from retained truth")
                fields["decoded"][offset] = decoded
                masks = stratum_masks(decoded, fields["predicted"][offset])
                geometric = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                for bit, name in enumerate(GEOMETRIC_NAMES):
                    geometric |= np.asarray(masks[name], dtype=np.uint8) << bit
                fields["geometry"][offset] = geometric
                corrector.end_frame(decoded.reshape(-1))
                previous = current
            corrector_state = jg2.corrector_state(corrector)
            lost = jg2.uncaptured_divergent_state(
                corrector, runtime["cold_corrector"], set(corrector_state)
            )
            if lost:
                raise Wh1Error(f"checkpoint omits divergent corrector state: {lost[:8]}")
            decoder_state = bl1.decoder_state(decoder)
            paths = stage_paths(store, stage_start, stage_end)
            receipt = save_stage(
                paths, binding, stage_start, stage_end, fields, corrector_state,
                decoder_state, time.perf_counter() - started,
            )
            receipts.append(receipt)
            new_stages += 1
            print(json.dumps({"stage": [stage_start, stage_end], "receipt": str(paths["receipt"]),
                              "indicator_bits": receipt["indicator_bits"],
                              "which_bits": receipt["which_bits"]}), flush=True)
    return receipts


def _axis_rows(
    values: np.ndarray, labels: tuple[str, ...] | None, positions: np.ndarray,
    wrong_positions: np.ndarray, wrong_bits: np.ndarray, which_bits: np.ndarray,
    total_wrong_positions: int, total_wrong_bits: float, total_which_bits: float,
) -> list[dict[str, Any]]:
    rows = []
    for value in values:
        index = int(value)
        area_fraction = float(positions[index]) / POSITIONS
        bit_share = float(wrong_bits[index]) / total_wrong_bits if total_wrong_bits else 0.0
        rows.append({
            "value": index,
            "name": labels[index] if labels is not None else str(index),
            "positions_numerator": int(positions[index]),
            "positions_denominator_n600": POSITIONS,
            "area_fraction": area_fraction,
            "wrong_positions_numerator": int(wrong_positions[index]),
            "wrong_positions_denominator": total_wrong_positions,
            "wrong_indicator_bits_numerator": float(wrong_bits[index]),
            "wrong_indicator_bits_denominator": total_wrong_bits,
            "wrong_indicator_bit_share": bit_share,
            "which_bits_numerator": float(which_bits[index]),
            "which_bits_denominator": total_which_bits,
            "wrong_bit_enrichment_over_area": bit_share / area_fraction if area_fraction else None,
        })
    return rows


def analyze(store: Path, binding: dict[str, Any]) -> Path:
    receipts = completed_stages(store, binding)
    if len(receipts) != N // STAGE_FRAMES:
        raise Wh1Error("n600 analysis requires all 30 replay stages")
    predicted = np.concatenate([
        np.load(stage_paths(store, int(row["frames"][0]), int(row["frames"][1]))["predicted"],
                mmap_mode="r", allow_pickle=False) for row in receipts
    ])
    decoded = np.concatenate([
        np.load(stage_paths(store, int(row["frames"][0]), int(row["frames"][1]))["decoded"],
                mmap_mode="r", allow_pickle=False) for row in receipts
    ])
    if predicted.shape != (N, HEIGHT, WIDTH) or decoded.shape != predicted.shape:
        raise Wh1Error("assembled categorical ledger shape drift")
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    if gt.shape != decoded.shape or gt.dtype != np.uint8:
        raise Wh1Error("SHA-bound DALI GT class field shape/dtype drift")
    token_gt_mismatch_positions = int(np.count_nonzero(decoded != gt))
    codes = transition_codes(predicted, decoded)
    transition_counts = np.empty((CLASSES * CLASSES, HEIGHT, WIDTH), dtype=np.uint16)
    for code in range(CLASSES * CLASSES):
        transition_counts[code] = np.count_nonzero(codes == code, axis=0).astype(np.uint16)
    v12 = json.loads(V12_RECEIPT.read_text(encoding="utf-8"))
    target_cache = Path(v12["target_custody"]["cache_path"])
    if target_cache.stat().st_size != int(v12["target_custody"]["cache_bytes"]):
        raise Wh1Error("V12 target cache size drifted")
    if sha256_file(target_cache) != str(v12["target_custody"]["cache_sha256"]):
        raise Wh1Error("V12 target cache SHA-256 drifted")
    poses = np.asarray(open_stored_npy_memmap(target_cache, "gt_poses"), dtype=np.float64)
    xi_tracks, xi_membership, xi_summary = build_xi_tracks(codes, transition_counts, poses)
    track_lengths = np.asarray([track.length for track in xi_tracks], dtype=np.uint32)
    track_offsets = np.concatenate(
        [np.asarray([0], dtype=np.uint64), np.cumsum(track_lengths, dtype=np.uint64)]
    )
    track_events = np.fromiter(
        (event for track in xi_tracks for event in track.event_ids), dtype=np.uint64,
        count=int(track_offsets[-1]),
    )
    g4_payload_path = store / "retained/g4_stationarity/G4_PAYLOAD.npz"
    atomic_npz(
        g4_payload_path,
        {
            "transition_counts": transition_counts,
            "xi_membership_packbits_little": np.packbits(
                xi_membership.reshape(-1), bitorder="little"
            ),
            "track_start_pair": np.asarray([track.start_pair for track in xi_tracks], dtype=np.uint16),
            "track_start_row": np.asarray([track.start_row for track in xi_tracks], dtype=np.uint16),
            "track_start_col": np.asarray([track.start_col for track in xi_tracks], dtype=np.uint16),
            "track_transition_code": np.asarray(
                [track.transition_code for track in xi_tracks], dtype=np.uint8
            ),
            "track_event_offsets": track_offsets,
            "track_event_ids": track_events,
        },
    )
    g4_payload_fact = file_fact(g4_payload_path)
    stationarity_facts = []
    rows_index = np.arange(HEIGHT)[:, None]
    cols_index = np.arange(WIDTH)[None, :]
    stationarity_root = store / "retained/g4_stationarity"
    for receipt in receipts:
        start, end = map(int, receipt["frames"])
        category = np.full((end - start, HEIGHT, WIDTH), 3, dtype=np.uint8)
        for frame in range(start, end):
            local = frame - start
            wrong = predicted[frame] != decoded[frame]
            recurrence = transition_counts[codes[frame], rows_index, cols_index]
            image = wrong & (recurrence >= 2)
            xi = wrong & ~image & xi_membership[frame]
            transient = wrong & ~image & ~xi
            category[local][image] = 0
            category[local][xi] = 1
            category[local][transient] = 2
        path = stationarity_root / f"frames_{start:04d}_{end - 1:04d}.u8.npy"
        atomic_npy(path, category)
        stationarity_facts.append(file_fact(path))

    class_positions = np.zeros(CLASSES, dtype=np.int64)
    class_wrong_positions = np.zeros(CLASSES, dtype=np.int64)
    class_wrong_bits = np.zeros(CLASSES, dtype=np.float64)
    class_which_bits = np.zeros(CLASSES, dtype=np.float64)
    token_positions = np.zeros(CLASSES, dtype=np.int64)
    token_wrong_positions = np.zeros(CLASSES, dtype=np.int64)
    token_wrong_bits = np.zeros(CLASSES, dtype=np.float64)
    token_which_bits = np.zeros(CLASSES, dtype=np.float64)
    pred_positions = np.zeros(CLASSES, dtype=np.int64)
    pred_wrong_positions = np.zeros(CLASSES, dtype=np.int64)
    pred_wrong_bits = np.zeros(CLASSES, dtype=np.float64)
    pred_which_bits = np.zeros(CLASSES, dtype=np.float64)
    buckets = len(MARGIN_EDGES) - 1
    bucket_positions = np.zeros(buckets, dtype=np.int64)
    bucket_wrong_positions = np.zeros(buckets, dtype=np.int64)
    bucket_wrong_bits = np.zeros(buckets, dtype=np.float64)
    bucket_which_bits = np.zeros(buckets, dtype=np.float64)
    g4_positions = np.zeros(4, dtype=np.int64)
    g4_wrong_positions = np.zeros(4, dtype=np.int64)
    g4_wrong_bits = np.zeros(4, dtype=np.float64)
    g4_which_bits = np.zeros(4, dtype=np.float64)
    geo_positions = np.zeros(4, dtype=np.int64)
    geo_wrong_positions = np.zeros(4, dtype=np.int64)
    geo_wrong_bits = np.zeros(4, dtype=np.float64)
    geo_which_bits = np.zeros(4, dtype=np.float64)
    pair_rows = []
    tile_shape = (CLASSES, buckets, HEIGHT // 16, WIDTH // 16)
    oracle_count = np.zeros(tile_shape, dtype=np.int64)
    oracle_wrong = np.zeros(tile_shape, dtype=np.int64)
    oracle_actual = np.zeros(tile_shape, dtype=np.float64)
    top_position_heap: list[tuple[float, int, int, int, int, int, int, int, int, float]] = []
    total_indicator = total_which = total_wrong_bits = 0.0
    total_wrong_positions = 0
    for receipt, stationarity_fact in zip(receipts, stationarity_facts, strict=True):
        start, end = map(int, receipt["frames"])
        paths = stage_paths(store, start, end)
        indicator = np.load(paths["indicator"], mmap_mode="r", allow_pickle=False)
        which = np.load(paths["which"], mmap_mode="r", allow_pickle=False)
        bucket = np.load(paths["bucket"], mmap_mode="r", allow_pickle=False)
        geometry = np.load(paths["geometry"], mmap_mode="r", allow_pickle=False)
        stationarity = np.load(stationarity_fact["path"], mmap_mode="r", allow_pickle=False)
        for local, frame in enumerate(range(start, end)):
            actual = decoded[frame]
            gt_class = np.asarray(gt[frame], dtype=np.uint8)
            pred = predicted[frame]
            wrong = pred != actual
            wrong_indicator = np.where(wrong, indicator[local], 0.0)
            total_indicator += float(indicator[local].sum(dtype=np.float64))
            total_which += float(which[local].sum(dtype=np.float64))
            total_wrong_bits += float(wrong_indicator.sum(dtype=np.float64))
            total_wrong_positions += int(wrong.sum())
            for values, arrays in (
                (gt_class, (class_positions, class_wrong_positions, class_wrong_bits, class_which_bits)),
                (actual, (token_positions, token_wrong_positions, token_wrong_bits, token_which_bits)),
                (pred, (pred_positions, pred_wrong_positions, pred_wrong_bits, pred_which_bits)),
                (bucket[local], (bucket_positions, bucket_wrong_positions, bucket_wrong_bits, bucket_which_bits)),
                (stationarity[local], (g4_positions, g4_wrong_positions, g4_wrong_bits, g4_which_bits)),
            ):
                arrays[0][:] += np.bincount(values.ravel(), minlength=arrays[0].size)
                arrays[1][:] += np.bincount(values[wrong].ravel(), minlength=arrays[1].size)
                arrays[2][:] += np.bincount(values.ravel(), weights=wrong_indicator.ravel(), minlength=arrays[2].size)
                arrays[3][:] += np.bincount(values.ravel(), weights=which[local].ravel(), minlength=arrays[3].size)
            for bit in range(4):
                mask = (geometry[local] & (1 << bit)) != 0
                geo_positions[bit] += int(mask.sum())
                geo_wrong_positions[bit] += int(np.count_nonzero(mask & wrong))
                geo_wrong_bits[bit] += float(wrong_indicator[mask].sum())
                geo_which_bits[bit] += float(which[local][mask].sum())
            tile_y = np.arange(HEIGHT)[:, None] // 16
            tile_x = np.arange(WIDTH)[None, :] // 16
            np.add.at(oracle_count, (pred, bucket[local], tile_y, tile_x), 1)
            np.add.at(oracle_wrong, (pred, bucket[local], tile_y, tile_x), wrong.astype(np.int64))
            np.add.at(oracle_actual, (pred, bucket[local], tile_y, tile_x), indicator[local])
            pair_rows.append({
                "pair": frame,
                "positions_denominator": PLANE,
                "wrong_positions_numerator": int(wrong.sum()),
                "wrong_indicator_bits": float(wrong_indicator.sum()),
                "which_bits": float(which[local].sum()),
            })
            candidates = np.flatnonzero(wrong)
            if candidates.size:
                costs = wrong_indicator.ravel()[candidates]
                # A global top-1000 position cannot rank below 1000 within its
                # own pair, so reducing each pair to its local top 1000 is exact.
                take_count = min(1000, costs.size)
                take = candidates[np.argpartition(costs, -take_count)[-take_count:]]
                for flat in take:
                    y, x = divmod(int(flat), WIDTH)
                    entry = (
                        float(indicator[local, y, x]), frame, y, x,
                        int(actual[y, x]), int(gt_class[y, x]), int(pred[y, x]), int(bucket[local, y, x]),
                        int(stationarity[local, y, x]), float(which[local, y, x]),
                    )
                    if len(top_position_heap) < 1000:
                        heapq.heappush(top_position_heap, entry)
                    elif entry > top_position_heap[0]:
                        heapq.heapreplace(top_position_heap, entry)
    stream_bits = EXPECTED["stream_bytes"] * 8
    total_selected = total_indicator + total_which
    if abs(total_selected - stream_bits) >= 9.0:
        raise Wh1Error("selected ideal bits fail the bounded RC64 physical-stream reconciliation")
    oracle_bound = binary_entropy_bits(oracle_wrong, oracle_count)
    oracle_gap = oracle_actual - oracle_bound
    oracle_payload_path = store / "retained/conditional_oracle_cells.npz"
    atomic_npz(
        oracle_payload_path,
        {
            "positions": oracle_count,
            "wrong_positions": oracle_wrong,
            "actual_indicator_bits": oracle_actual,
            "empirical_binary_entropy_bound_bits": oracle_bound,
            "oracle_gap_bits": oracle_gap,
        },
    )
    oracle_payload_fact = file_fact(oracle_payload_path)
    cell_rows = []
    for flat in np.argsort(oracle_actual.ravel())[::-1][:200]:
        pred, margin, tile_y, tile_x = np.unravel_index(int(flat), tile_shape)
        if oracle_count[pred, margin, tile_y, tile_x] == 0:
            continue
        cell_rows.append({
            "predicted_class": int(pred), "predicted_class_name": CLASS_NAMES[pred],
            "margin_bucket": int(margin), "tile_row": int(tile_y), "tile_column": int(tile_x),
            "positions": int(oracle_count[pred, margin, tile_y, tile_x]),
            "wrong_positions": int(oracle_wrong[pred, margin, tile_y, tile_x]),
            "actual_indicator_bits": float(oracle_actual[pred, margin, tile_y, tile_x]),
            "empirical_binary_entropy_bound_bits": float(oracle_bound[pred, margin, tile_y, tile_x]),
            "oracle_gap_bits": float(oracle_gap[pred, margin, tile_y, tile_x]),
        })
    top_positions = [
        {
            "pair": frame, "row": y, "column": x,
            "decoded_token_class": actual, "dali_gt_class": gt_class,
            "predicted_class": predicted_class,
            "margin_bucket": margin_bucket, "g4_stationarity": G4_NAMES[stationarity],
            "wrong_indicator_bits": indicator_bits, "which_bits": which_class_bits,
        }
        for (
            indicator_bits, frame, y, x, actual, gt_class, predicted_class,
            margin_bucket, stationarity, which_class_bits,
        ) in sorted(top_position_heap, reverse=True)
    ]
    total_which_bits = total_which
    tables = {
        "dali_gt_class": _axis_rows(np.arange(CLASSES), CLASS_NAMES, class_positions,
                                   class_wrong_positions, class_wrong_bits, class_which_bits,
                                   total_wrong_positions, total_wrong_bits, total_which_bits),
        "decoded_token_class": _axis_rows(np.arange(CLASSES), CLASS_NAMES, token_positions,
                                          token_wrong_positions, token_wrong_bits, token_which_bits,
                                          total_wrong_positions, total_wrong_bits, total_which_bits),
        "predicted_class": _axis_rows(np.arange(CLASSES), CLASS_NAMES, pred_positions,
                                      pred_wrong_positions, pred_wrong_bits, pred_which_bits,
                                      total_wrong_positions, total_wrong_bits, total_which_bits),
        "margin_bucket": _axis_rows(np.arange(buckets), None, bucket_positions,
                                    bucket_wrong_positions, bucket_wrong_bits, bucket_which_bits,
                                    total_wrong_positions, total_wrong_bits, total_which_bits),
        "g4_stationarity": _axis_rows(np.arange(4), G4_NAMES, g4_positions,
                                      g4_wrong_positions, g4_wrong_bits, g4_which_bits,
                                      total_wrong_positions, total_wrong_bits, total_which_bits),
        "g4_geometric_strata_overlapping": _axis_rows(np.arange(4), GEOMETRIC_NAMES, geo_positions,
                                                       geo_wrong_positions, geo_wrong_bits, geo_which_bits,
                                                       total_wrong_positions, total_wrong_bits, total_which_bits),
        "pair_rank": sorted(pair_rows, key=lambda row: row["wrong_indicator_bits"], reverse=True),
    }
    top_path = store / "TOP_CONCENTRATION_POSITIONS.json"
    cells_path = store / "CONDITIONAL_ORACLE_CELLS.json"
    atomic_json(top_path, {"schema": "ddm_wh1_top_positions.v1", "rows": top_positions})
    atomic_json(cells_path, {"schema": "ddm_wh1_conditional_cells.v1", "rows": cell_rows})
    result = {
        "schema": "ddm_wh1_wrong_half_decomposition_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "source_binding": binding,
        "analysis_implementation": file_fact(Path(__file__)),
        "accounting": {
            "positions_numerator": POSITIONS,
            "positions_denominator_n600": POSITIONS,
            "physical_stream_bits_numerator": stream_bits,
            "physical_stream_bits_denominator": stream_bits,
            "selected_ideal_bits": total_selected,
            "rc64_terminal_residual_bits": stream_bits - total_selected,
            "indicator_bits": total_indicator,
            "indicator_bytes": total_indicator / 8.0,
            "wrong_indicator_bits": total_wrong_bits,
            "wrong_indicator_bytes": total_wrong_bits / 8.0,
            "which_class_bits": total_which_bits,
            "which_class_bytes": total_which_bits / 8.0,
            "wrong_positions_numerator": total_wrong_positions,
            "wrong_positions_denominator": POSITIONS,
            "decoded_token_vs_dali_gt_mismatch_positions_numerator": token_gt_mismatch_positions,
            "decoded_token_vs_dali_gt_mismatch_positions_denominator": POSITIONS,
        },
        "margin_bucket_edges_bits": [float(value) if np.isfinite(value) else None for value in MARGIN_EDGES],
        "tables": tables,
        "conditional_entropy_pricing": {
            "conditioning_axes": ["coder_argmax_class", "coder_top1_runnerup_margin_bucket", "fixed_16x16_spatial_tile"],
            "axes_decoder_known": True,
            "parameter_bytes_charged": False,
            "actual_indicator_bits": float(oracle_actual.sum()),
            "oracle_empirical_binary_entropy_bound_bits": float(oracle_bound.sum()),
            "gross_oracle_gap_bits": float(oracle_gap.sum()),
            "gross_oracle_gap_bytes": float(oracle_gap.sum() / 8.0),
            "positive_cell_gap_bits": float(np.maximum(oracle_gap, 0.0).sum()),
            "measured_gb1_model_axis_remaining_ceiling_bytes": 2009.0,
            "ceiling_comparison": "diagnostic upper bound only; table/model description bytes are omitted",
        },
        "g4": {
            "stationarity_definitions": G4_NAMES,
            "xi_summary": xi_summary,
            "xi_track_count": len(xi_tracks),
            "proxy_boundary": "target-cache metric-Pose6 G1 translation-only proxy; not physical BEV and not decoder-free",
            "stationarity_fields": stationarity_facts,
            "retained_payload": g4_payload_fact,
        },
        "receipts": {
            "stage_receipts": [file_fact(stage_paths(store, int(row["frames"][0]), int(row["frames"][1]))["receipt"]) for row in receipts],
            "top_positions": file_fact(top_path),
            "conditional_cells": file_fact(cells_path),
            "conditional_oracle_payload": oracle_payload_fact,
        },
        "scope": "full n600 GB1 token stream; copied Python shipped receiver; no scorer and no candidate",
    }
    result_path = store / "RESULT.json"
    atomic_json(result_path, result)
    atomic_json(
        store / "CLEANUP_MANIFEST.json",
        {
            "schema": "certified_rebuildable_artifact_manifest.v1",
            "policy": "certify-or-block; delete_authorized=false",
            "result": file_fact(result_path),
            "runtime_copy_manifest": file_fact(store / "retained/runtime_copy/SOURCE_COPY_MANIFEST.json"),
            "stage_receipts": result["receipts"]["stage_receipts"],
            "stationarity_fields": stationarity_facts,
            "g4_payload": g4_payload_fact,
            "conditional_oracle_payload": oracle_payload_fact,
            "rebuild_command": f".venv/bin/python tools/token_wrong_half_ledger.py --store {store} all",
            "source_archive": binding["sources"]["copied_gb1_archive"],
            "false_authority": {"axis": AXIS, "score_claim": False, "pointer_moved": False},
            "delete_authorized": False,
        },
    )
    return result_path


def verify_result(store: Path, binding: dict[str, Any]) -> None:
    result_path = store / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("source_binding") != binding or result.get("schema") != "ddm_wh1_wrong_half_decomposition_result.v1":
        raise Wh1Error("result source/schema drift")
    completed_stages(store, binding)
    for fact in result["g4"]["stationarity_fields"]:
        verify_file(Path(fact["path"]), fact["sha256"], fact["bytes"])
    fact = result["g4"]["retained_payload"]
    verify_file(Path(fact["path"]), fact["sha256"], fact["bytes"])
    for key in ("top_positions", "conditional_cells"):
        fact = result["receipts"][key]
        verify_file(Path(fact["path"]), fact["sha256"], fact["bytes"])
    fact = result["receipts"]["conditional_oracle_payload"]
    verify_file(Path(fact["path"]), fact["sha256"], fact["bytes"])
    cleanup = store / "CLEANUP_MANIFEST.json"
    print(json.dumps({"status": "PASS", "result": file_fact(result_path), "cleanup": file_fact(cleanup)}, sort_keys=True))


def configure_inputs(args: argparse.Namespace) -> None:
    """Bind one runtime/object without requiring code edits for Stage B."""
    global SOURCE_RUNTIME, TRUTH, GT_FIELD, V12_RECEIPT
    SOURCE_RUNTIME = args.source_runtime.resolve()
    TRUTH = args.truth.resolve()
    GT_FIELD = args.gt_field.resolve()
    V12_RECEIPT = args.v12_receipt.resolve()
    EXPECTED.update(
        {
            "archive_sha256": args.archive_sha256,
            "archive_bytes": args.archive_bytes,
            "stream_sha256": args.stream_sha256,
            "stream_bytes": args.stream_bytes,
            "tokens_sha256": args.truth_sha256,
            "tokens_bytes": args.truth_bytes,
            "gt_sha256": args.gt_sha256,
            "gt_bytes": args.gt_bytes,
            "v12_receipt_sha256": args.v12_receipt_sha256,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--max-new-stages", type=int)
    parser.add_argument("--source-runtime", type=Path, default=SOURCE_RUNTIME)
    parser.add_argument("--archive-sha256", default=EXPECTED["archive_sha256"])
    parser.add_argument("--archive-bytes", type=int, default=EXPECTED["archive_bytes"])
    parser.add_argument("--stream-sha256", default=EXPECTED["stream_sha256"])
    parser.add_argument("--stream-bytes", type=int, default=EXPECTED["stream_bytes"])
    parser.add_argument("--truth", type=Path, default=TRUTH)
    parser.add_argument("--truth-sha256", default=EXPECTED["tokens_sha256"])
    parser.add_argument("--truth-bytes", type=int, default=EXPECTED["tokens_bytes"])
    parser.add_argument("--gt-field", type=Path, default=GT_FIELD)
    parser.add_argument("--gt-sha256", default=EXPECTED["gt_sha256"])
    parser.add_argument("--gt-bytes", type=int, default=EXPECTED["gt_bytes"])
    parser.add_argument("--v12-receipt", type=Path, default=V12_RECEIPT)
    parser.add_argument("--v12-receipt-sha256", default=EXPECTED["v12_receipt_sha256"])
    parser.add_argument("command", choices=("prepare", "replay", "analyze", "verify", "all"))
    args = parser.parse_args()
    configure_inputs(args)
    store = args.store.resolve()
    runtime_copy = prepare_runtime_copy(store)
    binding = source_binding(store, runtime_copy)
    if args.command == "prepare":
        print(json.dumps({"runtime_copy": str(runtime_copy), "source_binding": binding}, sort_keys=True))
        return 0
    library = build_decoder(store, runtime_copy)
    if args.command in ("replay", "all"):
        replay(store, binding, runtime_copy, library, args.max_new_stages)
    if args.command in ("analyze", "all"):
        analyze(store, binding)
    if args.command in ("verify", "all"):
        verify_result(store, binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
