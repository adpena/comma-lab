#!/usr/bin/env python3
"""RX2 terminal IHS1 pack, table fit, native-RC64, and identity race.

This runner consumes the terminal epoch-60 EMA checkpoint produced by
``train_ddm_cl1_hpac_capacity.py --profile rx2_mc36``.  It retains the raw
IHS1 payload, every lossless model representation, base float logits, fitted
table payloads, every selected candidate's quantized probabilities and RC64
stream, all complete archives and repeats, and the full raw receiver output.

The fitted-table development screen is stratified n=120 and selects only which
tables receive the expensive full-n600 native-RC64 race.  Every verdict and
fire gate comes from the real n600 coded streams plus exact token/raw identity.
No scorer is invoked here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import lzma
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

rx1 = importlib.import_module("experiments.ddm_rx1_rate_representation_attack")
cp = importlib.import_module("experiments.ddm_cp135_rate_compose")

WORK_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac")
BULK_ROOT = Path("/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac")
BASE_ARCHIVE = rx1.DEFAULT_ARCHIVE
BASE_RUNTIME = rx1.DEFAULT_RUNTIME
SOURCE_MANIFEST = rx1.DEFAULT_SOURCE_MANIFEST
EXPECTED_SPATIAL = rx1.DEFAULT_EXPECTED_SPATIAL
EXPERIMENT_BOOK = rx1.DEFAULT_EXPERIMENT_BOOK
INTAKE_CODE = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code")
TERMINAL_CHECKPOINT = BULK_ROOT / "training/checkpoints/mc36_hpac_best_ema.checkpoints/qat_stage_end_epoch_0060.pt"
F26P_LIFTED_RUNTIME = rx1.F26P_LIFTED_RUNTIME

AXIS = "[macOS-CPU advisory, scorer-free lossless composition]"
SCORE_CLAIM = False
TOKEN_COUNT = rx1.EXPECTED_EVENTS
EVENTS_PER_FRAME = rx1.EVENTS_PER_FRAME
LOGIT_PRECISION = 8
TABLE_STATES = 25
CLASSES = 5
TABLE_BITS = 6
TABLE_FULL_BYTES = 100
TABLE_COMPACT_BYTES = 96
DEVELOPMENT_SEED = 20260814
DEVELOPMENT_FRAMES = 120
SELECTED_FITTED_TABLES = 4
SHRINK_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
CLIP_SCALE_GRID = (0.5, 0.75, 1.0, 1.25)
SMOOTHING = 1.0


class RX2RaceError(RuntimeError):
    """Fail-closed error for a broken RX2 byte or identity gate."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_record(path: Path, *, role: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise RX2RaceError(f"retained artifact is absent: {path}")
    row: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if role is not None:
        row["role"] = role
    return row


def atomic_bytes(path: Path, value: bytes, *, executable: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if executable:
            temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return atomic_bytes(path, buffer.getvalue())


def atomic_npz(path: Path, **values: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    return atomic_bytes(path, buffer.getvalue())


def _require(path: Path, *, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file():
        raise RX2RaceError(f"required input is absent: {path}")
    if size is not None and path.stat().st_size != size:
        raise RX2RaceError(f"required input size changed: {path}")
    if digest is not None and sha256_file(path) != digest:
        raise RX2RaceError(f"required input SHA-256 changed: {path}")


def _require_bulk_free(required_gib: int) -> dict[str, int]:
    BULK_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(BULK_ROOT).free
    required = required_gib * (1 << 30)
    if free < required:
        raise RX2RaceError(f"RX2 race needs {required} free bytes; observed {free}")
    return {"bulk_free_bytes": free, "required_free_bytes": required}


def _source() -> Any:
    return cp.SourceSymbols(SOURCE_MANIFEST)


def _prepared() -> dict[str, Any]:
    path = BULK_ROOT / "PREPARE_RESULT.json"
    if not path.is_file():
        raise RX2RaceError("RX2 terminal prepare stage is incomplete")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("complete") is not True:
        raise RX2RaceError("RX2 terminal prepare receipt is incomplete")
    return value


def _fit_result() -> dict[str, Any]:
    path = BULK_ROOT / "FIT_RESULT.json"
    if not path.is_file():
        raise RX2RaceError("RX2 table-fit stage is incomplete")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("complete") is not True:
        raise RX2RaceError("RX2 table-fit receipt is incomplete")
    return value


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(BASE_ARCHIVE, size=rx1.EXPECTED_ARCHIVE_BYTES, digest=rx1.EXPECTED_ARCHIVE_SHA256)
    _require(EXPECTED_SPATIAL, size=TOKEN_COUNT, digest=rx1.EXPECTED_SPATIAL_SHA256)
    _require(args.checkpoint)
    if not BASE_RUNTIME.is_dir() or not EXPERIMENT_BOOK.is_dir() or not INTAKE_CODE.is_dir():
        raise RX2RaceError("pinned runtime, ExperimentBook, or intake source is absent")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != "ddm_cl1_hpac_capacity_checkpoint.v2"
        or checkpoint.get("epoch") != 60
        or checkpoint.get("phase") != "discrete_qat"
        or checkpoint.get("deployment_weights") != "ema_shadow"
        or checkpoint.get("run_identity", {}).get("training_config", {}).get("profile") != "rx2_mc36"
    ):
        raise RX2RaceError("selected checkpoint is not the terminal RX2 EMA QAT checkpoint")
    if checkpoint.get("causal_state_sha256") is None:
        raise RX2RaceError("selected checkpoint lacks its causal-state hash")
    trainer = importlib.import_module("tools.train_ddm_cl1_hpac_capacity")
    if trainer._causal_state_sha256(checkpoint) != checkpoint["causal_state_sha256"]:
        raise RX2RaceError("selected checkpoint causal-state hash does not verify")
    source_digest = _source().digest()
    if source_digest != rx1.EXPECTED_EVENT_SHA256:
        raise RX2RaceError("MC36 event-order symbol source changed")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    storage = _require_bulk_free(args.required_free_gib)
    return {
        "schema": "ddm_rx2_identity_race_preflight.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "checkpoint": file_record(args.checkpoint),
        "checkpoint_causal_state_sha256": checkpoint["causal_state_sha256"],
        "runner": file_record(Path(__file__)),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "base_archive": file_record(BASE_ARCHIVE),
        "expected_spatial": file_record(EXPECTED_SPATIAL),
        "source_manifest": file_record(SOURCE_MANIFEST),
        "source_event_order_sha256": source_digest,
        "bulk_root": str(BULK_ROOT),
        **storage,
        "all_long_stages_checkpointed": True,
    }


def _import_packer() -> Any:
    code = str(INTAKE_CODE)
    if code not in sys.path:
        sys.path.insert(0, code)
    return importlib.import_module("pack_hpac_self_compress")


def _pack_args() -> SimpleNamespace:
    return SimpleNamespace(
        channels=64,
        patch=64,
        delta=2,
        frame_dim=8,
        weight_bound=127,
        activation_bound=127,
        weight_exponent_min=-6,
    )


def _pack_terminal_ihs1(checkpoint_path: Path, output: Path) -> dict[str, Any]:
    packer = _import_packer()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    topology = _pack_args()
    source = packer.model_from_args(topology, True)
    source.load_state_dict(checkpoint["state_dict"])
    packer.set_deployed_bit_depths(source, True)
    source.eval()
    raw = packer.serialize_self_compressed(source)
    raw_path = output / "hpac.ihs1.raw"
    atomic_bytes(raw_path, raw)
    restored = packer.model_from_args(topology, False).eval()
    packer.deserialize_self_compressed(restored, raw)
    source_state = source.state_dict()
    restored_state = restored.state_dict()
    if source_state.keys() != restored_state.keys():
        raise RX2RaceError("IHS1 pack changed the deployed state-dict schema")
    changed_tensors = [name for name in source_state if not torch.equal(source_state[name], restored_state[name])]
    if changed_tensors:
        raise RX2RaceError(f"IHS1 pack changed deployed tensors: {changed_tensors[:8]}")
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    current = torch.randint(0, 5, (2, 384, 512), generator=generator)
    previous = torch.randint(0, 5, (2, 384, 512), generator=generator)
    frame_index = torch.tensor([0, 599])
    with torch.inference_mode():
        expected = source(current, frame_index, previous)
        actual = restored(current, frame_index, previous)
    max_diff = float((expected - actual).abs().max())
    if max_diff != 0.0:
        raise RX2RaceError(f"IHS1 exact deploy-bound pack changed logits by {max_diff}")
    xz = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=packer.LZMA_FILTERS)
    xz_path = output / "hpac.ihs1.xz"
    atomic_bytes(xz_path, xz)
    return {
        "raw": file_record(raw_path),
        "xz": file_record(xz_path),
        "verified_exact": True,
        "state_dict_exact": True,
        "state_tensor_count": len(source_state),
        "max_logit_diff": max_diff,
        "weight_bound": 127,
        "activation_bound": 127,
        "deploy_bound_intersection_enforced": True,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    admission = preflight(args)
    model_root = BULK_ROOT / "retained/models/rx2_terminal"
    packed = _pack_terminal_ihs1(args.checkpoint, model_root)
    ihs1_path = Path(packed["raw"]["path"])
    ihs1 = ihs1_path.read_bytes()
    if not ihs1.startswith(b"IHS1"):
        raise RX2RaceError("terminal packed model is not canonical IHS1")
    representations = [
        {
            "name": "xz_custodied",
            "codec": "xz",
            "codec_id": rx1.RX1_CODEC_XZ,
            "payload": packed["xz"],
            "parseback_sha256": sha256_bytes(lzma.decompress(Path(packed["xz"]["path"]).read_bytes())),
        }
    ]
    for quality in range(12):
        payload = rx1._brotli(ihs1, quality, args.brotli)
        path = model_root / f"hpac.ihs1.br.q{quality}"
        atomic_bytes(path, payload)
        restored = rx1._brotli_restore(payload, args.brotli)
        if restored != ihs1:
            raise RX2RaceError(f"Brotli q{quality} IHS1 parse-back differs")
        representations.append(
            {
                "name": f"brotli_q{quality}",
                "codec": "brotli",
                "codec_id": rx1.RX1_CODEC_BROTLI,
                "payload": file_record(path),
                "parseback_sha256": sha256_bytes(restored),
            }
        )

    runtime_module = cp.load_runtime(BASE_RUNTIME)
    parts = runtime_module.read_residual_archive(BASE_ARCHIVE)
    fields, compact_section = rx1._hp4_fields(BASE_ARCHIVE)
    if len(fields) != 5:
        raise RX2RaceError("MC36 HP4 physical field count changed")
    semantic_stream, carrier_stream = fields[3], fields[4]
    frozen = BULK_ROOT / "retained/models/mc36_frozen"
    frozen_records = {
        "semantic_stream": atomic_bytes(frozen / "semantic.br", semantic_stream),
        "carrier_stream": atomic_bytes(frozen / "carrier.br", carrier_stream),
        "base_residual": atomic_bytes(frozen / "residual.rcf1", parts.residual_payload),
        "base_tokens": atomic_bytes(frozen / "tokens.rc64", parts.token_stream),
    }
    if (
        len(parts.residual_payload) != TABLE_FULL_BYTES
        or len(compact_section) != TABLE_COMPACT_BYTES
        or parts.residual_payload != b"RCF1" + compact_section
    ):
        raise RX2RaceError("MC36 residual-table accounting changed")

    adapted = WORK_ROOT / "adapted_runtime"
    if not adapted.exists():
        shutil.copytree(BASE_RUNTIME, adapted)
    receiver = adapted / "runtime/residual_archive.py"
    atomic_bytes(receiver, rx1._patch_runtime(receiver.read_text(encoding="utf-8")).encode())
    winner_representation = min(representations, key=lambda row: row["payload"]["bytes"])
    smoke_model = rx1.pack_rx1_model(
        Path(winner_representation["payload"]["path"]).read_bytes(),
        semantic_stream,
        carrier_stream,
        codec_id=int(winner_representation["codec_id"]),
        table_mode=rx1.RX1_TABLE_ON,
    )
    smoke_member = smoke_model + parts.residual_payload[4:] + parts.token_stream
    smoke_root = BULK_ROOT / "retained/receiver_smoke"
    smoke_archive = rx1.deterministic_zip(smoke_member)
    smoke_records = {
        "model": atomic_bytes(smoke_root / "models.rx1m", smoke_model),
        "member": atomic_bytes(smoke_root / "p", smoke_member),
        "archive": atomic_bytes(smoke_root / "archive.zip", smoke_archive),
    }
    parsed = rx1._receiver_parseback_subprocess(adapted, smoke_root / "archive.zip", brotli_binary=args.brotli)
    expected = {
        "hpac_sha256": sha256_bytes(ihs1),
        "semantic_sha256": sha256_bytes(parts.semantic_blob),
        "carrier_sha256": sha256_bytes(parts.carrier_blob),
        "residual_sha256": sha256_bytes(parts.residual_payload),
        "token_sha256": sha256_bytes(parts.token_stream),
    }
    if any(parsed[name] != digest for name, digest in expected.items()):
        raise RX2RaceError("terminal IHS1 shipped-receiver smoke parse-back differs")
    result = {
        "schema": "ddm_rx2_prepare.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "preflight": admission,
        "selected_checkpoint": file_record(args.checkpoint),
        "pack": packed,
        "ihs1": file_record(ihs1_path),
        "representations": representations,
        "representation_winner": winner_representation,
        "semantic_stream": frozen_records["semantic_stream"],
        "carrier_stream": frozen_records["carrier_stream"],
        "base_residual": frozen_records["base_residual"],
        "base_tokens": frozen_records["base_tokens"],
        "adapted_runtime_receiver": file_record(receiver),
        "receiver_smoke": {"artifacts": smoke_records, "parsed": parsed, "expected": expected, "exact": True},
        "borrowed_substrate": (
            "PR130 supplied IntegerHPAC/IHS1 and HB2 supplied the deploy-bound repair; "
            "RX2 contribution is the target-MC36 fit, checkpoint-specific table, and whole-container race"
        ),
        "all_materialized_payloads_retained": True,
    }
    atomic_json(BULK_ROOT / "PREPARE_RESULT.json", result)
    atomic_json(WORK_ROOT / "PREPARE_RESULT.json", result)
    return result


def _frame_record(logits_path: Path, feature_path: Path, frame: int) -> dict[str, Any]:
    logits = np.load(logits_path, mmap_mode="r", allow_pickle=False)
    feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if logits.dtype != np.float32 or logits.shape != (EVENTS_PER_FRAME, CLASSES):
        raise RX2RaceError(f"base logits have invalid geometry: {logits_path}")
    if feature.dtype != np.uint8 or feature.shape != (EVENTS_PER_FRAME,) or np.any(feature >= TABLE_STATES):
        raise RX2RaceError(f"table features have invalid geometry: {feature_path}")
    return {
        "frame": frame,
        "logits": file_record(logits_path),
        "feature": file_record(feature_path),
        "complete": True,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted, dtype=np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def export_base(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepared()
    _require_bulk_free(args.required_free_gib)

    runtime_module = cp.load_runtime(BASE_RUNTIME)
    archive_module = importlib.import_module("runtime.residual_archive")
    renderer = runtime_module._load_renderer(BASE_RUNTIME / "cpr1")
    ihs1_path = Path(prepared["ihs1"]["path"])
    model = renderer.load_hpac(ihs1_path.read_bytes(), torch.device("cpu"))
    masks = renderer.group_masks(torch.device("cpu"))
    sparse = archive_module._sparse_class(BASE_RUNTIME / "cpr1")(model, renderer.EVAL_H, renderer.EVAL_W)
    importlib.import_module("runtime.hpac_inference").optimize_sparse_evaluator(sparse)
    group_positions = [np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)) for mask in masks]
    source = _source()
    root = BULK_ROOT / "retained/probabilities/base_float"
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with torch.inference_mode():
        for frame in range(args.start_frame, args.end_frame):
            logits_path = root / f"logits_{frame:04d}.npy"
            feature_path = root / f"feature_{frame:04d}.npy"
            receipt_path = root / f"frame_{frame:04d}.json"
            if logits_path.is_file() and feature_path.is_file() and receipt_path.is_file():
                if json.loads(receipt_path.read_text()) != _frame_record(logits_path, feature_path, frame):
                    raise RX2RaceError(f"base-probability checkpoint changed at frame {frame}")
                continue
            events = source.frame(frame)
            previous_events = np.zeros(EVENTS_PER_FRAME, dtype=np.uint8) if frame == 0 else source.frame(frame - 1)
            previous_np = (
                np.zeros((384, 512), dtype=np.uint8)
                if frame == 0
                else rx1.spatial_frame(previous_events, group_positions)
            )
            previous = torch.from_numpy(previous_np.astype(np.int64, copy=False))[None]
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame]), previous)
            boundary = (
                np.full(EVENTS_PER_FRAME, 4, dtype=np.uint8)
                if frame == 0
                else archive_module._boundary_buckets(previous_np).reshape(-1)
            )
            frame_logits = np.empty((EVENTS_PER_FRAME, CLASSES), dtype=np.float32)
            frame_feature = np.empty(EVENTS_PER_FRAME, dtype=np.uint8)
            offset = 0
            for group, positions in enumerate(group_positions):
                selected = sparse.selected_logits(current, context, group).cpu().numpy().astype(np.float32)
                predicted = selected.argmax(axis=1).astype(np.uint8)
                end = offset + len(positions)
                symbols = events[offset:end]
                frame_logits[offset:end] = selected
                frame_feature[offset:end] = boundary[positions] * 5 + predicted
                current.reshape(-1)[torch.from_numpy(positions)] = torch.from_numpy(
                    symbols.astype(np.int64, copy=False)
                )
                offset = end
            if offset != EVENTS_PER_FRAME:
                raise RX2RaceError("base probability export did not consume one frame")
            if not np.array_equal(current[0].numpy(), rx1.spatial_frame(events, group_positions)):
                raise RX2RaceError("base probability teacher forcing changed the MC36 labels")
            atomic_npy(logits_path, frame_logits)
            atomic_npy(feature_path, frame_feature)
            atomic_json(receipt_path, _frame_record(logits_path, feature_path, frame))
            print(json.dumps({"base_export_frame": frame + 1, "elapsed_s": time.time() - started}), flush=True)

    frames = []
    for frame in range(600):
        logits_path = root / f"logits_{frame:04d}.npy"
        feature_path = root / f"feature_{frame:04d}.npy"
        receipt_path = root / f"frame_{frame:04d}.json"
        if logits_path.is_file() and feature_path.is_file() and receipt_path.is_file():
            record = _frame_record(logits_path, feature_path, frame)
            if json.loads(receipt_path.read_text()) != record:
                raise RX2RaceError(f"base probability receipt changed at frame {frame}")
            frames.append(record)
    complete = len(frames) == 600
    stats_record = None
    if complete:
        counts = np.zeros((TABLE_STATES, CLASSES), dtype=np.float64)
        predicted = np.zeros((TABLE_STATES, CLASSES), dtype=np.float64)
        for frame in range(600):
            logits = np.load(root / f"logits_{frame:04d}.npy", mmap_mode="r", allow_pickle=False)
            feature = np.load(root / f"feature_{frame:04d}.npy", mmap_mode="r", allow_pickle=False)
            symbols = source.frame(frame)
            counts += np.bincount(
                feature.astype(np.int64) * CLASSES + symbols.astype(np.int64),
                minlength=TABLE_STATES * CLASSES,
            ).reshape(TABLE_STATES, CLASSES)
            probabilities = _softmax(logits)
            indices = feature.astype(np.int64)[:, None] * CLASSES + np.arange(CLASSES)
            predicted += np.bincount(
                indices.reshape(-1),
                weights=probabilities.reshape(-1),
                minlength=TABLE_STATES * CLASSES,
            ).reshape(TABLE_STATES, CLASSES)
        if int(counts.sum()) != TOKEN_COUNT or not math.isclose(float(predicted.sum()), TOKEN_COUNT, abs_tol=64):
            raise RX2RaceError("checkpoint-specific table statistics have the wrong denominator")
        stats_record = atomic_npz(root / "full_field_stats.npz", counts=counts, predicted=predicted)
    identity = {
        "schema": "ddm_rx2_base_probability_export.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "ihs1": file_record(ihs1_path),
        "source_event_order_sha256": source.digest() if complete else None,
        "completed_frames": len(frames),
        "complete_n600": complete,
        "frames": frames,
        "stats": stats_record,
        "wall_s": time.time() - started,
        "torch_threads": args.torch_threads,
        "all_probability_payloads_retained": True,
    }
    atomic_json(root / "EXPORT_RESULT.json", identity)
    return identity


def pack_signed(values: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if np.any(flat < low) or np.any(flat > high):
        raise RX2RaceError("signed table code exceeds its packed domain")
    mask = (1 << bits) - 1
    accumulator = 0
    available = 0
    output = bytearray()
    for value in (flat & mask).tolist():
        accumulator |= int(value) << available
        available += bits
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def serialize_table(codes: np.ndarray, scale: float) -> bytes:
    codes = np.asarray(codes, dtype=np.int8)
    if codes.shape != (TABLE_STATES, CLASSES):
        raise RX2RaceError("fitted table has the wrong geometry")
    deployed_scale = float(np.asarray([scale], dtype="<f2")[0])
    if not np.isfinite(deployed_scale) or deployed_scale <= 0:
        raise RX2RaceError("fitted table scale is not positive fp16")
    payload = b"RCF1" + np.asarray([deployed_scale], dtype="<f2").tobytes() + pack_signed(codes, TABLE_BITS)
    if len(payload) != TABLE_FULL_BYTES:  # MEASURE_ONLY_OK:returned payload is retained by every caller
        raise RX2RaceError("fitted table serialized to the wrong byte count")
    return payload


def _table_values(payload: bytes) -> np.ndarray:
    cp.load_runtime(BASE_RUNTIME)
    table = importlib.import_module("runtime.residual_archive")._decode_fixed_table(payload)
    return np.asarray(table.values, dtype=np.float32)


def _stratified_frames() -> np.ndarray:
    generator = np.random.default_rng(DEVELOPMENT_SEED)
    edges = np.linspace(0, 600, DEVELOPMENT_FRAMES + 1, dtype=np.int64)
    return np.asarray(
        [generator.integers(edges[index], edges[index + 1]) for index in range(DEVELOPMENT_FRAMES)],
        dtype=np.int64,
    )


def _candidate_table(ratio: np.ndarray, shrink: float, clip_scale: float) -> tuple[np.ndarray, float]:
    values = ratio * shrink
    values -= (values.max(axis=1, keepdims=True) + values.min(axis=1, keepdims=True)) / 2.0
    maximum = float(np.max(np.abs(values)))
    scale = float(np.asarray([max(maximum * clip_scale / 31.0, np.finfo(np.float16).tiny)], dtype="<f2")[0])
    codes = np.clip(np.rint(values / scale), -32, 31).astype(np.int8)
    return codes, scale


def fit_tables(_args: argparse.Namespace) -> dict[str, Any]:
    _require_bulk_free(_args.required_free_gib)
    base = BULK_ROOT / "retained/probabilities/base_float"
    export = json.loads((base / "EXPORT_RESULT.json").read_text(encoding="utf-8"))
    if not export.get("complete_n600"):
        raise RX2RaceError("base n600 probability export is incomplete")
    with np.load(base / "full_field_stats.npz", allow_pickle=False) as loaded:
        counts = loaded["counts"]
        predicted = loaded["predicted"]
    ratio = np.log((counts + SMOOTHING) / (predicted + SMOOTHING))
    table_root = BULK_ROOT / "retained/tables"
    candidates: list[dict[str, Any]] = []
    neutral_payload = serialize_table(np.zeros((TABLE_STATES, CLASSES), dtype=np.int8), 1.0)
    neutral_path = table_root / "neutral.rcf1"
    atomic_bytes(neutral_path, neutral_payload)
    candidates.append(
        {
            "variant": "neutral",
            "kind": "neutral_control",
            "shrink": 0.0,
            "clip_scale": 1.0,
            "table": file_record(neutral_path),
        }
    )
    for shrink in SHRINK_GRID:
        for clip_scale in CLIP_SCALE_GRID:
            variant = f"s{str(shrink).replace('.', 'p')}_c{str(clip_scale).replace('.', 'p')}"
            codes, scale = _candidate_table(ratio, shrink, clip_scale)
            payload = serialize_table(codes, scale)
            path = table_root / f"{variant}.rcf1"
            atomic_bytes(path, payload)
            candidates.append(
                {
                    "variant": variant,
                    "kind": "checkpoint_specific_boundary_predicted_int6",
                    "shrink": shrink,
                    "clip_scale": clip_scale,
                    "scale_fp16": float(np.frombuffer(payload[4:6], dtype="<f2")[0]),
                    "code_min": int(codes.min()),
                    "code_max": int(codes.max()),
                    "nonzero_codes": int(np.count_nonzero(codes)),
                    "table": file_record(path),
                }
            )
    frames = _stratified_frames()
    source = _source()
    started = time.time()
    for row in candidates:
        values = _table_values(Path(row["table"]["path"]).read_bytes())
        bits = 0.0
        for frame in frames.tolist():
            logits = np.load(base / f"logits_{frame:04d}.npy", mmap_mode="r", allow_pickle=False)
            feature = np.load(base / f"feature_{frame:04d}.npy", mmap_mode="r", allow_pickle=False)
            codes = np.clip(np.rint((logits + values[feature]) * LOGIT_PRECISION), -32768, 32767).astype(np.int16)
            probabilities = cp.probability_from_codes(codes, LOGIT_PRECISION)
            selected = probabilities[np.arange(EVENTS_PER_FRAME), source.frame(frame).astype(np.int64)]
            bits -= float(np.log2(selected.astype(np.float64)).sum())
        row["development"] = {
            "selection_mode": "120 equal-width five-frame strata; one seeded random frame per stratum",
            "seed": DEVELOPMENT_SEED,
            "frames": [int(frame) for frame in frames],
            "prefix": False,
            "tokens": len(frames) * EVENTS_PER_FRAME,
            "ideal_bits": bits,
            "ideal_bytes": bits / 8.0,
            "linear_n600_ideal_bytes_projection": bits / 8.0 * (600 / len(frames)),
            "authority": "SELECTION_ONLY_NOT_VERDICT",
        }
    fitted = sorted(
        (row for row in candidates if row["kind"] != "neutral_control"),
        key=lambda row: (row["development"]["ideal_bytes"], row["variant"]),
    )
    selected = ["neutral", *[row["variant"] for row in fitted[:SELECTED_FITTED_TABLES]]]
    result = {
        "schema": "ddm_rx2_checkpoint_specific_table_fit.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "fit_population_tokens": TOKEN_COUNT,
        "fit_method": (
            "full-field observed-vs-model class-count log ratio in the exact 25-state "
            "boundary-bucket x base-predicted-class receiver feature; row-centered and int6/fp16 quantized"
        ),
        "fit_stats": file_record(base / "full_field_stats.npz"),
        "smoothing": SMOOTHING,
        "grid": {"shrink": list(SHRINK_GRID), "clip_scale": list(CLIP_SCALE_GRID)},
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_for_full_n600_real_rc64": selected,
        "full_n600_real_rc64_required_for_verdict": True,
        "development_elapsed_s": time.time() - started,
        "all_table_payloads_retained": True,
    }
    atomic_json(BULK_ROOT / "FIT_RESULT.json", result)
    atomic_json(WORK_ROOT / "FIT_RESULT.json", result)
    return result


def _candidate_row(variant: str) -> dict[str, Any]:
    fit = _fit_result()
    row = next((item for item in fit["candidates"] if item["variant"] == variant), None)
    if row is None:
        raise RX2RaceError(f"unknown fitted table variant: {variant}")
    if variant not in fit["selected_for_full_n600_real_rc64"]:
        raise RX2RaceError(f"variant was not selected for the full n600 real race: {variant}")
    return row


def materialize_probabilities(args: argparse.Namespace) -> dict[str, Any]:
    _require_bulk_free(args.required_free_gib)
    row = _candidate_row(args.variant)
    values = _table_values(Path(row["table"]["path"]).read_bytes())
    base = BULK_ROOT / "retained/probabilities/base_float"
    root = BULK_ROOT / "retained/probabilities" / args.variant
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for frame in range(args.start_frame, args.end_frame):
        path = root / f"codes_{frame:04d}.npy"
        receipt = root / f"codes_{frame:04d}.json"
        if path.is_file() and receipt.is_file():
            record = rx1._frame_record(path, frame, args.variant)
            if json.loads(receipt.read_text()) != record:
                raise RX2RaceError(f"candidate probability checkpoint changed at frame {frame}")
            continue
        logits = np.load(base / f"logits_{frame:04d}.npy", mmap_mode="r", allow_pickle=False)
        feature = np.load(base / f"feature_{frame:04d}.npy", mmap_mode="r", allow_pickle=False)
        codes = np.clip(np.rint((logits + values[feature]) * LOGIT_PRECISION), -32768, 32767).astype(np.int16)
        atomic_npy(path, codes)
        atomic_json(receipt, rx1._frame_record(path, frame, args.variant))
        print(json.dumps({"variant": args.variant, "probability_frame": frame + 1}), flush=True)
    frames = []
    for frame in range(600):
        path = root / f"codes_{frame:04d}.npy"
        receipt = root / f"codes_{frame:04d}.json"
        if path.is_file() and receipt.is_file():
            record = rx1._frame_record(path, frame, args.variant)
            if json.loads(receipt.read_text()) != record:
                raise RX2RaceError(f"candidate probability receipt changed at frame {frame}")
            frames.append(record)
    identity = {
        "schema": "ddm_rx2_probability_identity.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": args.variant,
        "table": row["table"],
        "completed_frames": len(frames),
        "complete_n600": len(frames) == 600,
        "frames": frames,
    }
    identity_path = root / "PROBABILITY_IDENTITY.json"
    atomic_json(identity_path, identity)
    result = {
        **identity,
        "probability_identity": file_record(identity_path),
        "wall_s": time.time() - started,
        "all_probability_payloads_retained": True,
    }
    atomic_json(root / "EXPORT_RESULT.json", result)
    return result


def encode_rc64(args: argparse.Namespace) -> dict[str, Any]:
    _require_bulk_free(args.required_free_gib)
    _candidate_row(args.variant)
    rx_args = SimpleNamespace(
        output=BULK_ROOT,
        variant=args.variant,
        source_manifest=SOURCE_MANIFEST,
        runtime=BASE_RUNTIME,
        experiment_book=EXPERIMENT_BOOK,
    )
    return rx1.encode_rc64(rx_args)


def build(args: argparse.Namespace) -> dict[str, Any]:
    _require_bulk_free(args.required_free_gib)
    row = _candidate_row(args.variant)
    prepared = _prepared()
    rc64_path = BULK_ROOT / "retained/coders" / args.variant / "RC64_RESULT.json"
    rc64 = json.loads(rc64_path.read_text(encoding="utf-8"))
    if not rc64.get("event_order_identity") or not rc64.get("spatial_token_identity"):
        raise RX2RaceError("native RC64 token-identity gate is incomplete")
    token_path = Path(rc64["token_payload"]["path"])
    if file_record(token_path) != rc64["token_payload"]:
        raise RX2RaceError("native RC64 payload failed custody")
    token = token_path.read_bytes()
    table_full = Path(row["table"]["path"]).read_bytes()
    if len(table_full) != TABLE_FULL_BYTES or not table_full.startswith(b"RCF1"):
        raise RX2RaceError("candidate residual table is malformed")
    residual = table_full[4:]
    semantic_stream = Path(prepared["semantic_stream"]["path"]).read_bytes()
    carrier_stream = Path(prepared["carrier_stream"]["path"]).read_bytes()
    ihs1 = Path(prepared["ihs1"]["path"]).read_bytes()
    rows = []
    for representation in prepared["representations"]:
        hpac_path = Path(representation["payload"]["path"])
        if file_record(hpac_path) != representation["payload"]:
            raise RX2RaceError("lossless IHS1 representation failed custody")
        model = rx1.pack_rx1_model(
            hpac_path.read_bytes(),
            semantic_stream,
            carrier_stream,
            codec_id=int(representation["codec_id"]),
            table_mode=rx1.RX1_TABLE_ON,
        )
        unpacked = rx1.unpack_rx1_model(model, brotli_binary=args.brotli)
        if unpacked["hpac"] != ihs1:
            raise RX2RaceError("RX2 model representation parse-back changed IHS1")
        member = model + residual + token
        archive = rx1.deterministic_zip(member)
        repeat = rx1.deterministic_zip(member)
        root = BULK_ROOT / "retained/candidates" / args.variant / representation["name"]
        records = {
            "model": atomic_bytes(root / "models.rx1m", model),
            "residual": atomic_bytes(root / "residual.compact.bin", residual),
            "token": atomic_bytes(root / "tokens.rc64", token),
            "member": atomic_bytes(root / "p", member),
            "archive": atomic_bytes(root / "archive.zip", archive),
            "repeat_archive": atomic_bytes(root / "archive.repeat.zip", repeat),
        }
        if archive != repeat or rx1.read_stored_member(root / "archive.zip") != member:
            raise RX2RaceError("RX2 deterministic archive repeat or member parse-back differs")
        parsed = rx1._receiver_parseback_subprocess(
            WORK_ROOT / "adapted_runtime", root / "archive.zip", brotli_binary=args.brotli
        )
        expected = {
            "hpac_sha256": sha256_bytes(ihs1),
            "semantic_sha256": prepared["receiver_smoke"]["expected"]["semantic_sha256"],
            "carrier_sha256": prepared["receiver_smoke"]["expected"]["carrier_sha256"],
            "residual_sha256": sha256_bytes(table_full),
            "token_sha256": sha256_bytes(token),
        }
        if any(parsed[name] != digest for name, digest in expected.items()):
            raise RX2RaceError("shipped receiver parse-back changed a candidate component")
        archive_bytes = records["archive"]["bytes"]
        candidate = {
            "variant": args.variant,
            "representation": representation["name"],
            "archive": records["archive"],
            "repeat_archive": records["repeat_archive"],
            "repeat_byte_identical": archive == repeat,
            "member": records["member"],
            "model": records["model"],
            "residual": records["residual"],
            "token": records["token"],
            "table": row["table"],
            "receiver_parseback": parsed,
            "receiver_component_identity": True,
            "decoded_event_order_identity": True,
            "decoded_spatial_token_identity": True,
            "delta_distortion": 0.0,
            "archive_delta_vs_mc36": archive_bytes - rx1.EXPECTED_ARCHIVE_BYTES,
            "projected_score_if_mc36_distortion_held": (
                rx1.MC36_SCORE + 25.0 * (archive_bytes - rx1.EXPECTED_ARCHIVE_BYTES) / rx1.RATE_DENOMINATOR
            ),
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
        }
        atomic_json(root / "RESULT.json", candidate)
        rows.append(candidate)
    winner = min(rows, key=lambda item: item["archive"]["bytes"])
    result = {
        "schema": "ddm_rx2_build.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": args.variant,
        "base": file_record(BASE_ARCHIVE),
        "candidates": rows,
        "winner": winner,
        "candidate_count": len(rows),
        "all_payloads_retained": True,
        "whole_container_recount": True,
        "receiver_parseback": True,
        "zero_token_distortion": True,
    }
    atomic_json(BULK_ROOT / f"BUILD_RESULT_{args.variant}.json", result)
    atomic_json(WORK_ROOT / f"BUILD_RESULT_{args.variant}.json", result)
    return result


def _all_builds() -> list[dict[str, Any]]:
    fit = _fit_result()
    builds = []
    for variant in fit["selected_for_full_n600_real_rc64"]:
        path = BULK_ROOT / f"BUILD_RESULT_{variant}.json"
        if not path.is_file():
            raise RX2RaceError(f"full-n600 build is incomplete for {variant}")
        builds.append(json.loads(path.read_text(encoding="utf-8")))
    return builds


def cpu_decode(_args: argparse.Namespace) -> dict[str, Any]:
    builds = _all_builds()
    winner = min((build_row["winner"] for build_row in builds), key=lambda item: item["archive"]["bytes"])
    candidate = Path(winner["archive"]["path"])
    if file_record(candidate) != winner["archive"]:
        raise RX2RaceError("RX2 CPU-decode winner failed custody")
    if not F26P_LIFTED_RUNTIME.is_dir():
        raise RX2RaceError("custodied F26P lifted CPU runtime is absent")
    root = BULK_ROOT / "retained/cpu_decode/best_rx2"
    runtime_root = root / "lifted_submission_cpu"
    input_root = root / "input"
    output_root = root / "output"
    receipt_root = root / "receipts"
    log_root = root / "logs"
    file_list = root / "file_list.txt"
    root.mkdir(parents=True, exist_ok=True)
    required = rx1.EXPECTED_CPU_RAW_BYTES + 300_000_000
    if shutil.disk_usage(root).free < required:
        raise RX2RaceError("insufficient APDataStore space for the full raw decode")
    if not runtime_root.exists():
        shutil.copytree(F26P_LIFTED_RUNTIME, runtime_root)
    atomic_bytes(runtime_root / "archive.zip", candidate.read_bytes())
    receiver = runtime_root / "runtime/residual_archive.py"
    atomic_bytes(receiver, rx1._patch_runtime(receiver.read_text(encoding="utf-8")).encode())
    entrypoint = runtime_root / "inflate.py"
    source = entrypoint.read_text(encoding="utf-8")
    source, sha_count = re.subn(
        r'^ARCHIVE_SHA256 = "[0-9a-f]{64}"$',
        f'ARCHIVE_SHA256 = "{winner["archive"]["sha256"]}"',
        source,
        count=1,
        flags=re.MULTILINE,
    )
    source, byte_count = re.subn(
        r"^ARCHIVE_BYTES = [0-9_]+$",
        f"ARCHIVE_BYTES = {winner['archive']['bytes']:_}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if (sha_count, byte_count) != (1, 1):
        raise RX2RaceError("lifted CPU entrypoint archive pins could not be updated")
    atomic_bytes(entrypoint, source.encode(), executable=True)
    atomic_bytes(input_root / "p", rx1.read_stored_member(candidate))
    atomic_bytes(file_list, b"0.mkv\n")
    raw_path = output_root / "0.raw"
    receipt_path = receipt_root / "CPU_DECODE_RESULT.json"
    if receipt_path.is_file() and raw_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("complete") and file_record(raw_path) == receipt["raw_output"]:
            return receipt
        raise RX2RaceError("existing full raw decode differs from its receipt")
    if raw_path.exists():
        raise RX2RaceError("full raw output exists without a completion receipt")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "decode.log"
    command = [str(runtime_root / "inflate.sh"), str(input_root), str(output_root), str(file_list)]
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "VECLIB_MAXIMUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "PATH": os.pathsep.join([str(Path(sys.executable).parent), environment.get("PATH", "")]),
        }
    )
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "candidate": winner["archive"]}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        atomic_json(
            receipt_root / "CPU_DECODE_FAILURE.json",
            {
                "schema": "ddm_rx2_cpu_decode_failure.v1",
                "complete": False,
                "return_code": return_code,
                "candidate": winner["archive"],
                "log": file_record(log_path),
            },
        )
        raise RX2RaceError(f"RX2 lifted CPU decoder exited {return_code}; payloads retained")
    _require(raw_path, size=rx1.EXPECTED_CPU_RAW_BYTES, digest=rx1.EXPECTED_CPU_RAW_SHA256)
    report = None
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("schema") == "ddm_f26p_inflate_report.v1":
            report = value
            break
    if report is None or report.get("raw_sha256") != rx1.EXPECTED_CPU_RAW_SHA256:
        raise RX2RaceError("full CPU decode log lacks the exact raw-output receipt")
    result = {
        "schema": "ddm_rx2_cpu_decode.v1",
        "complete": True,
        "axis": "[macOS-CPU advisory, four-thread lifted F26]",
        "score_claim": False,
        "candidate": winner["archive"],
        "variant": winner["variant"],
        "adapted_runtime_receiver": file_record(receiver),
        "adapted_runtime_entrypoint": file_record(entrypoint),
        "input_member": file_record(input_root / "p"),
        "raw_output": file_record(raw_path),
        "expected_mc36_cpu_raw_sha256": rx1.EXPECTED_CPU_RAW_SHA256,
        "raw_identity_vs_mc36_cpu": True,
        "decoded_token_sha256": report["token_decoder"]["decoded_token_sha256"],
        "decoded_token_identity": report["token_decoder"]["decoded_token_sha256"] == rx1.EXPECTED_SPATIAL_SHA256,
        "checkpoint_resume": report["checkpoint_resume"],
        "checkpoint_dir": report["checkpoint_dir"],
        "wall_seconds": time.time() - started,
        "inflate_report": report,
        "log": file_record(log_path),
    }
    if not result["decoded_token_identity"]:
        raise RX2RaceError("full CPU decoder token identity differs")
    atomic_json(receipt_path, result)
    return result


def finalize(_args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepared()
    fit = _fit_result()
    builds = _all_builds()
    candidates = [candidate for build_row in builds for candidate in build_row["candidates"]]
    winner = min(candidates, key=lambda item: item["archive"]["bytes"])
    cpu_path = BULK_ROOT / "retained/cpu_decode/best_rx2/receipts/CPU_DECODE_RESULT.json"
    if not cpu_path.is_file():
        raise RX2RaceError("finalize requires the full lifted CPU raw identity receipt")
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    if not cpu.get("raw_identity_vs_mc36_cpu") or not cpu.get("decoded_token_identity"):
        raise RX2RaceError("finalize full raw/token identity gate is incomplete")
    admitted = winner["archive"]["bytes"] < rx1.EXPECTED_ARCHIVE_BYTES
    retained_inventory = rx1.retention_inventory(BULK_ROOT)
    result = {
        "schema": "ddm_rx2_final.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "base_mc36": {
            "archive": file_record(BASE_ARCHIVE),
            "exact_score": rx1.MC36_SCORE,
            "authority": "[contest-CUDA] n600 T4",
        },
        "selected_checkpoint": prepared["selected_checkpoint"],
        "pack": prepared["pack"],
        "table_fit": fit,
        "builds": builds,
        "winner": winner,
        "winner_admitted_vs_mc36": admitted,
        "measured_archive_delta_bytes": winner["archive_delta_vs_mc36"],
        "measured_delta_distortion": 0.0,
        "projected_score_not_authority": winner["projected_score_if_mc36_distortion_held"],
        "all_payloads_retained": True,
        "receiver_parseback": True,
        "decoded_token_identity": True,
        "local_rgb_raw_decode": cpu,
        "retention_inventory": retained_inventory,
        "main_t4_fire_order": {
            "sealed": True,
            "owner": "MAIN",
            "consumer_store": ".omx/state/main_hot_state.md plus canonical frontier pointer",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER" if admitted else "FOLDED",
            "trigger": (
                "the exact retained archive is strictly smaller than MC36 and has shipped-receiver, "
                "n600 token, and full raw identity"
            ),
            "reason": (
                "RX2 winner is strictly smaller than MC36"
                if admitted
                else "RX2 grid winner is not smaller than MC36, so exact evaluation cannot improve the frontier"
            ),
            "command_template": (
                "MAIN-owned governed T4 upstream/evaluate.py on the exact retained winner archive" if admitted else None
            ),
        },
    }
    atomic_json(BULK_ROOT / "FINAL_RESULT.json", result)
    atomic_json(WORK_ROOT / "FINAL_RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "stage",
        choices=(
            "preflight",
            "prepare",
            "export-base",
            "fit",
            "materialize",
            "encode",
            "build",
            "cpu-decode",
            "finalize",
        ),
    )
    value.add_argument("--checkpoint", type=Path, default=TERMINAL_CHECKPOINT)
    value.add_argument("--variant", default="neutral")
    value.add_argument("--start-frame", type=int, default=0)
    value.add_argument("--end-frame", type=int, default=600)
    value.add_argument("--torch-threads", type=int, default=4)
    value.add_argument("--required-free-gib", type=int, default=32)
    value.add_argument("--brotli", default=shutil.which("brotli") or "brotli")
    return value


def main() -> None:
    from tac.admission_guard import assert_governed_admission

    assert_governed_admission("ddm_rx2_mc36_identity_race")
    args = parser().parse_args()
    if not 0 <= args.start_frame < args.end_frame <= 600:
        raise SystemExit("invalid frame interval")
    if args.stage == "preflight":
        result = preflight(args)
    elif args.stage == "prepare":
        result = prepare(args)
    elif args.stage == "export-base":
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        result = export_base(args)
    elif args.stage == "fit":
        result = fit_tables(args)
    elif args.stage == "materialize":
        result = materialize_probabilities(args)
    elif args.stage == "encode":
        result = encode_rc64(args)
    elif args.stage == "build":
        result = build(args)
    elif args.stage == "cpu-decode":
        result = cpu_decode(args)
    else:
        result = finalize(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
