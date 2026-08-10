#!/usr/bin/env python3
"""Retained, scorer-free lossless composition race on the custodied PR135 base.

The expensive stage exports the exact F26 probability lattice one frame at a
time.  Each frame is an independently reloadable checkpoint because F26's
causal state is exactly the preceding decoded token frame plus the already
decoded groups in the current frame.  The token targets come from DT1's
retained, byte-identified PR130 export; the final RC64 verification proves that
F26 decodes the same symbols before any candidate is admitted.

Every materialized payload is written below ``--output`` before its size is
reported.  The script never runs a scorer or renders frames.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import lzma
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810")
DEFAULT_RUNTIME = DEFAULT_OUTPUT / "adapted_runtime"
DEFAULT_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/archive.zip")
DEFAULT_DT1_MANIFEST = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/chunk_manifest.json")
DEFAULT_HP3_MANIFEST = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hp3_20260810/retained/codes/requant_frame_embed_step2/chunk_manifest.json"
)
DEFAULT_EXPERIMENT_BOOK = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book")
EXPECTED_ARCHIVE_BYTES = 186_724
EXPECTED_ARCHIVE_SHA256 = "12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004"
EXPECTED_RC64_BYTES = 114_706
EXPECTED_RC64_SHA256 = "e77c075f4f7ee93a2d0c40343df263429b58e4c4d4f14c794529b53b466b9c73"
EXPECTED_EVENT_ORDER_SHA256 = "8eb51ab7a2884c9d7b6e73ee60f78ded38c691d6b82e639b75dddec6e0ac1366"
EXPECTED_SPATIAL_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
EXPECTED_HP3_IHS1_SHA256 = "d9391a5ef4c391d04972c82fa0942ea9faa20ca15418f78ffead90591842513c"
EXPECTED_EVENTS = 600 * 384 * 512
EVENTS_PER_FRAME = 384 * 512
AXIS = "[macOS-CPU advisory, scorer-free lossless composition]"
SCORE_CLAIM = False
RATE_DENOMINATOR = 37_545_489
BASE_SCORE = 0.16226942370411543
SPLIT_HEADER = struct.Struct("<HHH")
LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 16,
        "lc": 0,
        "lp": 1,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]
VARIANTS = ("control", "hp3_step2")
Variant = Literal["control", "hp3_step2"]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_record(root: Path) -> dict[str, Any]:
    """Hash every durable receiver file without including transient caches."""

    rows = []
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        record = file_record(path)
        relative = path.relative_to(root).as_posix()
        rows.append({"relative_path": relative, **record})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\n")
    return {
        "root": str(root.resolve()),
        "files": rows,
        "file_count": len(rows),
        "tree_sha256": digest.hexdigest(),
    }


def atomic_bytes(path: Path, value: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    if executable:
        temporary.chmod(0o755)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def read_stored_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        info = archive.infolist()
        if len(info) != 1 or info[0].filename != "p":
            raise RuntimeError("archive must contain exactly one member p")
        if info[0].compress_type != zipfile.ZIP_STORED:
            raise RuntimeError("member p must be stored")
        value = archive.read(info[0])
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        return value


def require_base(path: Path) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != EXPECTED_ARCHIVE_BYTES
        or sha256_file(path) != EXPECTED_ARCHIVE_SHA256
    ):
        raise RuntimeError("PR135 archive differs from the charter pin")


@dataclass(frozen=True)
class SourceChunk:
    start: int
    end: int
    symbols: Path
    sha256: str


class SourceSymbols:
    def __init__(self, manifest_path: Path) -> None:
        payload = json.loads(manifest_path.read_text())
        chunks = []
        for row in payload["chunks"]:
            chunks.append(
                SourceChunk(
                    int(row["start_frame"]),
                    int(row["end_frame"]),
                    Path(row["symbols_path"]),
                    str(row["symbols_sha256"]),
                )
            )
        if not chunks or chunks[0].start != 0 or chunks[-1].end != 600:
            raise RuntimeError("DT1 symbol manifest does not cover n600")
        if any(left.end != right.start for left, right in itertools.pairwise(chunks)):
            raise RuntimeError("DT1 symbol manifest is not contiguous")
        for chunk in chunks:
            if not chunk.symbols.is_file() or sha256_file(chunk.symbols) != chunk.sha256:
                raise RuntimeError(f"DT1 symbol chunk failed custody: {chunk.symbols}")
        self.chunks = tuple(chunks)
        self._loaded_path: Path | None = None
        self._loaded: np.ndarray | None = None

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < 600:
            raise IndexError(index)
        chunk = next(row for row in self.chunks if row.start <= index < row.end)
        if self._loaded_path != chunk.symbols:
            self._loaded = np.load(chunk.symbols, mmap_mode="r", allow_pickle=False)
            self._loaded_path = chunk.symbols
        assert self._loaded is not None
        if self._loaded.dtype not in (np.uint8, np.int32):
            raise RuntimeError("DT1 symbols have an unsupported dtype")
        local = index - chunk.start
        start = local * EVENTS_PER_FRAME
        result = np.asarray(self._loaded[start : start + EVENTS_PER_FRAME], dtype=np.uint8)
        if result.shape != (EVENTS_PER_FRAME,) or np.any(result >= 5):
            raise RuntimeError(f"DT1 frame {index} has invalid symbols")
        return result

    def digest(self) -> str:
        digest = hashlib.sha256()
        for frame in range(600):
            digest.update(self.frame(frame).tobytes())
        return digest.hexdigest()


class PriorCodes:
    def __init__(self, manifest_path: Path, expected_hpac_sha256: str) -> None:
        self.manifest_path = manifest_path
        payload = json.loads(manifest_path.read_text())
        if payload.get("complete") is not True:
            raise RuntimeError("prior code manifest is incomplete")
        if payload.get("hpac_sha256") != expected_hpac_sha256:
            raise RuntimeError("prior code manifest has a different HPAC payload")
        self.rows = tuple(payload["chunks"])
        expected_start = 0
        for row in self.rows:
            if int(row["start_frame"]) != expected_start:
                raise RuntimeError("prior code chunks are not contiguous")
            expected_start = int(row["end_frame"])
            for name in ("symbols", "codes"):
                record = row[name]
                path = Path(record["path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != int(record["bytes"])
                    or sha256_file(path) != record["sha256"]
                ):
                    raise RuntimeError(f"prior {name} payload failed custody: {path}")
        if expected_start != 600:
            raise RuntimeError("prior code manifest does not cover n600")
        self._loaded_path: Path | None = None
        self._loaded_codes: np.ndarray | None = None
        self._loaded_symbols: np.ndarray | None = None

    def frame(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        row = next(item for item in self.rows if int(item["start_frame"]) <= frame < int(item["end_frame"]))
        codes_path = Path(row["codes"]["path"])
        if self._loaded_path != codes_path:
            self._loaded_codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
            self._loaded_symbols = np.load(Path(row["symbols"]["path"]), mmap_mode="r", allow_pickle=False)
            self._loaded_path = codes_path
        assert self._loaded_codes is not None and self._loaded_symbols is not None
        local = frame - int(row["start_frame"])
        start = local * EVENTS_PER_FRAME
        end = start + EVENTS_PER_FRAME
        codes = np.asarray(self._loaded_codes[start:end])
        symbols = np.asarray(self._loaded_symbols[start:end], dtype=np.uint8)
        if codes.dtype != np.int16 or codes.shape != (EVENTS_PER_FRAME, 5):
            raise RuntimeError("prior probability codes have invalid geometry")
        return symbols, codes


def load_runtime(runtime_root: Path):
    import importlib

    root = runtime_root.resolve()
    sys.path.insert(0, str(root))
    try:
        existing = sys.modules.get("runtime.f26_inflate")
        if existing is not None:
            existing_path = Path(existing.__file__).resolve()
            expected_path = root / "runtime" / "f26_inflate.py"
            if existing_path != expected_path:
                raise RuntimeError("a different F26 runtime is already loaded")
            return existing
        return importlib.import_module("runtime.f26_inflate")
    finally:
        sys.path.pop(0)


def step2_ihs2(blob: bytes) -> tuple[bytes, dict[str, Any]]:
    """Apply HP3's nearest-even/ties-to-zero rule to F26's E0L0 frame field."""

    if len(blob) != 16_599 or blob[:6] != b"IHS2\x03\x31":
        raise RuntimeError("HP3 composition requires the exact F26 IHS2-v3 layout")
    frame_offset = 13_379
    frame_bytes = 2_400
    packed = np.frombuffer(blob[frame_offset : frame_offset + frame_bytes], dtype=np.uint8)
    codes = np.empty(frame_bytes * 2, dtype=np.int16)
    codes[0::2] = packed & 0x0F
    codes[1::2] = packed >> 4
    values = np.where(codes >= 8, codes - 16, codes).astype(np.int8)
    changed = values != (np.trunc(values.astype(np.float32) / 2.0).astype(np.int8) * 2)
    step2 = np.trunc(values.astype(np.float32) / 2.0).astype(np.int8) * 2
    unsigned = np.where(step2 < 0, step2.astype(np.int16) + 16, step2).astype(np.uint8)
    repacked = (unsigned[0::2] | (unsigned[1::2] << 4)).tobytes()
    result = blob[:frame_offset] + repacked + blob[frame_offset + frame_bytes :]
    return result, {
        "source_bytes": len(blob),
        "source_sha256": sha256_bytes(blob),
        "result_bytes": len(result),
        "result_sha256": sha256_bytes(result),
        "changed_values": int(changed.sum()),
        "frame_values": int(values.size),
        "source_min": int(values.min()),
        "source_max": int(values.max()),
        "result_min": int(step2.min()),
        "result_max": int(step2.max()),
    }


def variant_hpac(parts, variant: Variant) -> tuple[bytes, dict[str, Any]]:
    if variant == "control":
        return parts.hpac_blob, {
            "variant": variant,
            "changed_values": 0,
            "source_sha256": sha256_bytes(parts.hpac_blob),
            "result_sha256": sha256_bytes(parts.hpac_blob),
        }
    value, report = step2_ihs2(parts.hpac_blob)
    report["variant"] = variant
    return value, report


def spatial_frame(events: np.ndarray, group_positions: list[np.ndarray]) -> np.ndarray:
    flat = np.empty(EVENTS_PER_FRAME, dtype=np.uint8)
    offset = 0
    for positions in group_positions:
        end = offset + len(positions)
        flat[positions] = events[offset:end]
        offset = end
    if offset != EVENTS_PER_FRAME:
        raise RuntimeError("group positions do not consume one token frame")
    return flat.reshape(384, 512)


def probability_from_codes(codes: np.ndarray, precision: int) -> np.ndarray:
    values = np.asarray(codes, dtype=np.int16).astype(np.float64) / precision
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def _frame_record(path: Path, frame: int, variant: Variant) -> dict[str, Any]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.int16 or value.shape != (EVENTS_PER_FRAME, 5):
        raise RuntimeError(f"invalid retained code checkpoint: {path}")
    return {
        "frame": frame,
        "variant": variant,
        "codes": file_record(path),
        "events": EVENTS_PER_FRAME,
        "complete": True,
    }


def export_probabilities(args: argparse.Namespace) -> dict[str, Any]:
    import importlib

    import torch

    variant: Variant = args.variant
    require_base(args.archive)
    runtime_module = load_runtime(args.runtime)
    archive_module = importlib.import_module("runtime.residual_archive")
    parts = runtime_module.read_residual_archive(args.archive)
    renderer = runtime_module._load_renderer(args.runtime / "cpr1")
    hpac, hpac_report = variant_hpac(parts, variant)
    canonical_hpac = importlib.import_module("runtime.ihs2").materialize_ihs1(hpac, renderer)
    model = renderer.load_hpac(canonical_hpac, torch.device("cpu"))
    masks = renderer.group_masks(torch.device("cpu"))
    sparse = archive_module._sparse_class(args.runtime / "cpr1")(model, renderer.EVAL_H, renderer.EVAL_W)
    archive_module.optimize_sparse_evaluator(sparse)
    group_positions = [np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)) for mask in masks]
    source = SourceSymbols(args.dt1_manifest)
    output = args.output / "retained" / "probabilities" / variant
    output.mkdir(parents=True, exist_ok=True)
    stage_records = []
    started = time.time()
    table = parts.table
    if table is None:
        raise RuntimeError("F26 residual table is missing")
    with torch.inference_mode():
        for frame in range(args.start_frame, args.end_frame):
            path = output / f"codes_{frame:04d}.npy"
            receipt = output / f"codes_{frame:04d}.json"
            if path.is_file() and receipt.is_file():
                record = json.loads(receipt.read_text())
                if record != _frame_record(path, frame, variant):
                    raise RuntimeError(f"retained frame receipt changed: {receipt}")
                stage_records.append(record)
                print(json.dumps({"variant": variant, "frame": frame + 1, "status": "reused"}), flush=True)
                continue
            events = source.frame(frame)
            previous_events = np.zeros(EVENTS_PER_FRAME, dtype=np.uint8) if frame == 0 else source.frame(frame - 1)
            previous_np = (
                np.zeros((384, 512), dtype=np.uint8) if frame == 0 else spatial_frame(previous_events, group_positions)
            )
            previous = torch.from_numpy(previous_np.astype(np.int64, copy=False))[None]
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame]), previous)
            boundary = (
                np.full(EVENTS_PER_FRAME, 4, dtype=np.uint8)
                if frame == 0
                else archive_module._boundary_buckets(previous_np).reshape(-1)
            )
            frame_codes = np.empty((EVENTS_PER_FRAME, 5), dtype=np.int16)
            symbol_offset = 0
            for group, (_mask, positions) in enumerate(zip(masks, group_positions, strict=True)):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = boundary[positions].astype(np.int64) * 5 + predicted
                corrected = base_logits + table.values[feature]
                codes = np.clip(
                    np.rint(np.asarray(corrected, dtype=np.float32) * renderer.HPAC_LOGIT_PRECISION),
                    -32768,
                    32767,
                ).astype(np.int16)
                end = symbol_offset + len(positions)
                symbols = events[symbol_offset:end]
                current.reshape(-1)[torch.from_numpy(positions)] = torch.from_numpy(
                    symbols.astype(np.int64, copy=False)
                )
                frame_codes[symbol_offset:end] = codes
                symbol_offset = end
            if symbol_offset != EVENTS_PER_FRAME:
                raise RuntimeError("probability export did not consume the frame")
            if not np.array_equal(current[0].numpy(), spatial_frame(events, group_positions)):
                raise RuntimeError("teacher-forced frame reconstruction differs from DT1 symbols")
            atomic_npy(path, frame_codes)
            record = _frame_record(path, frame, variant)
            atomic_json(receipt, record)
            stage_records.append(record)
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "frame": frame + 1,
                        "elapsed_s": round(time.time() - started, 3),
                        "codes_sha256": record["codes"]["sha256"],
                    }
                ),
                flush=True,
            )
    completed = []
    for frame in range(600):
        path = output / f"codes_{frame:04d}.npy"
        receipt = output / f"codes_{frame:04d}.json"
        if path.is_file() and receipt.is_file():
            completed.append(_frame_record(path, frame, variant))
    result = {
        "schema": "ddm_cp135_probability_export.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": variant,
        "archive": file_record(args.archive),
        "hpac": hpac_report,
        "canonical_hpac_sha256": sha256_bytes(canonical_hpac),
        "completed_frames": len(completed),
        "requested_range": [args.start_frame, args.end_frame],
        "complete_n600": len(completed) == 600,
        "source_symbol_sha256": source.digest() if len(completed) == 600 else None,
        "frames": completed,
        "wall_s": time.time() - started,
    }
    atomic_json(output / "EXPORT_RESULT.json", result)
    return result


def derive_hp3_probabilities(args: argparse.Namespace) -> dict[str, Any]:
    """Compose PR135's exact residual table with HP3's retained n600 logits."""

    import importlib

    require_base(args.archive)
    runtime_module = load_runtime(args.runtime)
    archive_module = importlib.import_module("runtime.residual_archive")
    parts = runtime_module.read_residual_archive(args.archive)
    prior = PriorCodes(args.hp3_manifest, EXPECTED_HP3_IHS1_SHA256)
    source = SourceSymbols(args.dt1_manifest)
    group_positions = _group_positions(args.runtime)
    event_positions = np.concatenate(group_positions)
    output = args.output / "retained" / "probabilities" / "hp3_step2"
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    started = time.time()
    for frame in range(args.start_frame, args.end_frame):
        path = output / f"codes_{frame:04d}.npy"
        receipt = output / f"codes_{frame:04d}.json"
        if path.is_file() and receipt.is_file():
            record = _frame_record(path, frame, "hp3_step2")
            if json.loads(receipt.read_text()) != record:
                raise RuntimeError(f"retained HP3 frame receipt changed: {receipt}")
            completed.append(record)
            continue
        prior_symbols, prior_codes = prior.frame(frame)
        expected = source.frame(frame)
        if not np.array_equal(prior_symbols, expected):
            raise RuntimeError(f"HP3 source symbols differ at frame {frame}")
        previous = (
            np.zeros((384, 512), dtype=np.uint8)
            if frame == 0
            else spatial_frame(source.frame(frame - 1), group_positions)
        )
        boundary = (
            np.full(EVENTS_PER_FRAME, 4, dtype=np.uint8)
            if frame == 0
            else archive_module._boundary_buckets(previous).reshape(-1)
        )
        predicted = prior_codes.argmax(axis=1).astype(np.int64)
        feature = boundary[event_positions].astype(np.int64) * 5 + predicted
        base_logits = prior_codes.astype(np.float32) / 8
        corrected = base_logits + parts.table.values[feature]
        codes = np.clip(np.rint(corrected.astype(np.float32) * 8), -32768, 32767).astype(np.int16)
        atomic_npy(path, codes)
        record = _frame_record(path, frame, "hp3_step2")
        atomic_json(receipt, record)
        completed.append(record)
        if (frame + 1) % 24 == 0:
            print(
                json.dumps(
                    {
                        "variant": "hp3_step2",
                        "derived_frames": frame + 1,
                        "elapsed_s": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
    all_frames = []
    for frame in range(600):
        path = output / f"codes_{frame:04d}.npy"
        receipt = output / f"codes_{frame:04d}.json"
        if path.is_file() and receipt.is_file():
            all_frames.append(_frame_record(path, frame, "hp3_step2"))
    result = {
        "schema": "ddm_cp135_hp3_probability_derivation.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": "hp3_step2",
        "archive": file_record(args.archive),
        "source_manifest": file_record(args.hp3_manifest),
        "source_hpac_sha256": EXPECTED_HP3_IHS1_SHA256,
        "source_symbol_sha256": source.digest() if len(all_frames) == 600 else None,
        "derivation": (
            "HP3 retained int16 codes are exact selected logits times 8; add the "
            "PR135 fixed-table correction in float32 and apply the shipping round/clip"
        ),
        "runtime_selected_logits": file_record(args.runtime / "cpr1" / "hpac_integer_sparse.py"),
        "completed_frames": len(all_frames),
        "complete_n600": len(all_frames) == 600,
        "requested_range": [args.start_frame, args.end_frame],
        "frames": all_frames,
        "wall_s": time.time() - started,
    }
    atomic_json(output / "EXPORT_RESULT.json", result)
    return result


def _load_codes(root: Path, variant: Variant, frame: int) -> np.ndarray:
    path = root / "retained" / "probabilities" / variant / f"codes_{frame:04d}.npy"
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.int16 or value.shape != (EVENTS_PER_FRAME, 5):
        raise RuntimeError(f"invalid probability checkpoint: {path}")
    return value


def encode_ans(args: argparse.Namespace) -> dict[str, Any]:
    import constriction

    variant: Variant = args.variant
    export = json.loads((args.output / "retained" / "probabilities" / variant / "EXPORT_RESULT.json").read_text())
    if not export.get("complete_n600"):
        raise RuntimeError(f"{variant} probability export is incomplete")
    source = SourceSymbols(args.dt1_manifest)
    retained = args.output / "retained" / "coders" / variant
    checkpoint_root = retained / "checkpoints"
    progress_path = checkpoint_root / "LATEST.json"
    completed_result_path = retained / "ANS_RESULT.json"
    completed_token_path = retained / "tokens.ans"
    if not progress_path.is_file() and completed_result_path.is_file() and completed_token_path.is_file():
        completed_result = json.loads(completed_result_path.read_text())
        if not completed_result.get("symbol_identity") or file_record(completed_token_path) != completed_result.get(
            "token_payload"
        ):
            raise RuntimeError("completed ANS payload failed custody before checkpoint adoption")
        final_checkpoint_path = checkpoint_root / "through_frame_0000.ans.state"
        atomic_bytes(final_checkpoint_path, completed_token_path.read_bytes())
        final_checkpoint = {
            "schema": "ddm_cp135_ans_checkpoint.v1",
            "variant": variant,
            "through_frame": 0,
            "next_frame": -1,
            "state": file_record(final_checkpoint_path),
            "probability_export": file_record(
                args.output / "retained" / "probabilities" / variant / "EXPORT_RESULT.json"
            ),
            "adopted_from_completed_verified_payload": True,
        }
        atomic_json(final_checkpoint_path.with_suffix(".json"), final_checkpoint)
        atomic_json(progress_path, final_checkpoint)
    start_frame = 599
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        state_path = Path(progress["state"]["path"])
        if file_record(state_path) != progress["state"]:
            raise RuntimeError("ANS checkpoint state failed custody")
        state = np.frombuffer(state_path.read_bytes(), dtype="<u4").astype(np.uint32, copy=False)
        coder = constriction.stream.stack.AnsCoder(state)
        start_frame = int(progress["next_frame"])
        if not -1 <= start_frame < 600:
            raise RuntimeError("ANS checkpoint next-frame marker differs")
    else:
        coder = constriction.stream.stack.AnsCoder()
    if start_frame == -1 and completed_result_path.is_file():
        completed_result = json.loads(completed_result_path.read_text())
        if not completed_result.get("symbol_identity") or file_record(completed_token_path) != completed_result.get(
            "token_payload"
        ):
            raise RuntimeError("completed ANS result failed custody on resume")
        completed_result.update(
            {
                "resumable_from_disk": True,
                "checkpoint_count": len(list(checkpoint_root.glob("through_frame_*.ans.state"))),
                "final_checkpoint": file_record(checkpoint_root / "through_frame_0000.ans.state"),
                "resumed_completed_result": True,
            }
        )
        atomic_json(completed_result_path, completed_result)
        return completed_result
    family = constriction.stream.model.Categorical(perfect=False)
    started = time.time()
    for frame in range(start_frame, -1, -1):
        symbols = source.frame(frame).astype(np.int32)
        probabilities = probability_from_codes(_load_codes(args.output, variant, frame), 8)
        coder.encode_reverse(symbols, family, probabilities)
        if frame % 24 == 0 or frame == 0:
            checkpoint = coder.get_compressed().astype("<u4", copy=False).tobytes(order="C")
            checkpoint_path = checkpoint_root / f"through_frame_{frame:04d}.ans.state"
            atomic_bytes(checkpoint_path, checkpoint)
            checkpoint_receipt = {
                "schema": "ddm_cp135_ans_checkpoint.v1",
                "variant": variant,
                "through_frame": frame,
                "next_frame": frame - 1,
                "state": file_record(checkpoint_path),
                "probability_export": file_record(
                    args.output / "retained" / "probabilities" / variant / "EXPORT_RESULT.json"
                ),
            }
            atomic_json(checkpoint_path.with_suffix(".json"), checkpoint_receipt)
            atomic_json(progress_path, checkpoint_receipt)
            print(
                json.dumps(
                    {"variant": variant, "encoded_through_frame": frame, "elapsed_s": round(time.time() - started, 3)}
                ),
                flush=True,
            )
    payload = coder.get_compressed().astype("<u4", copy=False).tobytes(order="C")
    token_path = retained / "tokens.ans"
    atomic_bytes(token_path, payload)

    decoder = constriction.stream.stack.AnsCoder(np.frombuffer(payload, dtype="<u4").astype(np.uint32, copy=False))
    decoded_path = retained / "decoded_symbols.ans.bin"
    spatial_path = retained / "decoded_spatial_tokens.ans.bin"
    temporary = decoded_path.with_name(f".{decoded_path.name}.{os.getpid()}.tmp")
    spatial_temporary = spatial_path.with_name(f".{spatial_path.name}.{os.getpid()}.tmp")
    event_digest = hashlib.sha256()
    spatial_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    group_positions = _group_positions(args.runtime)
    with temporary.open("wb") as stream, spatial_temporary.open("wb") as spatial_stream:
        for frame in range(600):
            probabilities = probability_from_codes(_load_codes(args.output, variant, frame), 8)
            decoded = decoder.decode(family, probabilities).astype(np.uint8)
            expected = source.frame(frame)
            if not np.array_equal(decoded, expected):
                raise RuntimeError(f"ANS decoded symbols differ at frame {frame}")
            raw = decoded.tobytes()
            spatial_raw = spatial_frame(decoded, group_positions).tobytes()
            stream.write(raw)
            spatial_stream.write(spatial_raw)
            event_digest.update(raw)
            spatial_digest.update(spatial_raw)
            source_digest.update(expected.tobytes())
        stream.flush()
        os.fsync(stream.fileno())
        spatial_stream.flush()
        os.fsync(spatial_stream.fileno())
    if not decoder.is_empty():
        raise RuntimeError("ANS terminal state is not empty")
    os.replace(temporary, decoded_path)
    os.replace(spatial_temporary, spatial_path)
    if (
        event_digest.hexdigest() != source_digest.hexdigest()
        or event_digest.hexdigest() != EXPECTED_EVENT_ORDER_SHA256
        or spatial_digest.hexdigest() != EXPECTED_SPATIAL_TOKEN_SHA256
    ):
        raise RuntimeError("ANS decoded symbols differ from the canonical token digests")
    result = {
        "schema": "ddm_cp135_ans_result.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": variant,
        "codec": "constriction-0.5.0 stack ANS categorical perfect=False",
        "token_payload": file_record(token_path),
        "decoded_symbols": file_record(decoded_path),
        "decoded_spatial_tokens": file_record(spatial_path),
        "decoded_event_order_sha256": event_digest.hexdigest(),
        "decoded_spatial_token_sha256": spatial_digest.hexdigest(),
        "symbol_identity": True,
        "terminal_state_empty": True,
        "events": EXPECTED_EVENTS,
        "resumable_from_disk": True,
        "checkpoint_count": len(list(checkpoint_root.glob("through_frame_*.ans.state"))),
        "final_checkpoint": file_record(checkpoint_root / "through_frame_0000.ans.state"),
        "wall_s": time.time() - started,
    }
    atomic_json(retained / "ANS_RESULT.json", result)
    return result


def compile_rc64(runtime: Path, output: Path) -> Path:
    source = runtime / "runtime" / "entropy" / "rc64_backend.c"
    library = output / "work" / "librc64_cp135.so"
    library.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["cc", "-std=c11", "-O3", "-fPIC", "-shared", str(source), "-o", str(library)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"RC64 compilation failed: {completed.stderr}")
    return library


def _group_positions(runtime: Path) -> list[np.ndarray]:
    import torch

    runtime_module = load_runtime(runtime)
    renderer = runtime_module._load_renderer(runtime / "cpr1")
    return [
        np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)) for mask in renderer.group_masks(torch.device("cpu"))
    ]


def verify_rc64(args: argparse.Namespace) -> dict[str, Any]:
    import importlib

    require_base(args.archive)
    runtime_module = load_runtime(args.runtime)
    parts = runtime_module.read_residual_archive(args.archive)
    if len(parts.token_stream) != EXPECTED_RC64_BYTES or sha256_bytes(parts.token_stream) != EXPECTED_RC64_SHA256:
        raise RuntimeError("F26 RC64 stream differs from its custody pin")
    library = compile_rc64(args.runtime, args.output)
    decoder = importlib.import_module("runtime.residual_archive").NativeDecoder(library, parts.token_stream)
    source = SourceSymbols(args.dt1_manifest)
    retained = args.output / "retained" / "coders" / "control"
    retained.mkdir(parents=True, exist_ok=True)
    decoded_path = retained / "decoded_symbols.rc64.bin"
    spatial_path = retained / "decoded_spatial_tokens.rc64.bin"
    temporary = decoded_path.with_name(f".{decoded_path.name}.{os.getpid()}.tmp")
    spatial_temporary = spatial_path.with_name(f".{spatial_path.name}.{os.getpid()}.tmp")
    event_digest = hashlib.sha256()
    spatial_digest = hashlib.sha256()
    group_positions = _group_positions(args.runtime)
    started = time.time()
    with temporary.open("wb") as stream, spatial_temporary.open("wb") as spatial_stream:
        for frame in range(600):
            probabilities = probability_from_codes(_load_codes(args.output, "control", frame), 8)
            decoded = decoder.decode(probabilities).astype(np.uint8)
            expected = source.frame(frame)
            if not np.array_equal(decoded, expected):
                raise RuntimeError(f"RC64 decoded symbols differ at frame {frame}")
            raw = decoded.tobytes()
            spatial_raw = spatial_frame(decoded, group_positions).tobytes()
            stream.write(raw)
            spatial_stream.write(spatial_raw)
            event_digest.update(raw)
            spatial_digest.update(spatial_raw)
            if (frame + 1) % 24 == 0:
                print(json.dumps({"rc64_decoded_frames": frame + 1, "bit_position": decoder.bit_position}), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
        spatial_stream.flush()
        os.fsync(spatial_stream.fileno())
    os.replace(temporary, decoded_path)
    os.replace(spatial_temporary, spatial_path)
    if (
        event_digest.hexdigest() != EXPECTED_EVENT_ORDER_SHA256
        or spatial_digest.hexdigest() != EXPECTED_SPATIAL_TOKEN_SHA256
    ):
        raise RuntimeError("RC64 decoded symbols differ from the canonical token digests")
    result = {
        "schema": "ddm_cp135_rc64_parseback.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "source_stream": {
            "bytes": len(parts.token_stream),
            "sha256": sha256_bytes(parts.token_stream),
        },
        "decoded_symbols": file_record(decoded_path),
        "decoded_spatial_tokens": file_record(spatial_path),
        "decoded_event_order_sha256": event_digest.hexdigest(),
        "decoded_spatial_token_sha256": spatial_digest.hexdigest(),
        "symbol_identity": True,
        "events": EXPECTED_EVENTS,
        "decoder_bit_position": decoder.bit_position,
        "library": file_record(library),
        "wall_s": time.time() - started,
    }
    atomic_json(retained / "RC64_RESULT.json", result)
    return result


def _brotli_compress(value: bytes, quality: int, binary: str) -> bytes:
    completed = subprocess.run(
        [binary, "-q", str(quality), "-c"],
        input=value,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(f"Brotli q{quality} failed: {completed.stderr.decode(errors='replace')}")
    return completed.stdout


def pack_split_models(
    hpac_body: bytes,
    semantic_body: bytes,
    carrier_selector_body: bytes,
    *,
    qualities: tuple[int, int, int],
    brotli_binary: str,
) -> tuple[bytes, dict[str, Any]]:
    sources = (hpac_body, semantic_body, carrier_selector_body)
    streams = tuple(
        _brotli_compress(value, quality, brotli_binary) for value, quality in zip(sources, qualities, strict=True)
    )
    if any(len(value) >= 1 << 16 for value in streams):
        raise RuntimeError("split-model stream exceeds its u16 length field")
    header = SPLIT_HEADER.pack(*(len(value) for value in streams))
    payload = header + b"".join(streams)
    return payload, {
        "qualities": list(qualities),
        "header_bytes": len(header),
        "source_bytes": list(map(len, sources)),
        "stream_bytes": list(map(len, streams)),
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
    }


def unpack_split_models(payload: bytes, *, brotli_binary: str) -> tuple[bytes, bytes, bytes]:
    if len(payload) < SPLIT_HEADER.size:
        raise RuntimeError("truncated split-model payload")
    a, b, c = SPLIT_HEADER.unpack_from(payload)
    if SPLIT_HEADER.size + a + b + c != len(payload) or min(a, b, c) <= 0:
        raise RuntimeError("invalid split-model lengths")
    offset = SPLIT_HEADER.size
    streams = (
        payload[offset : offset + a],
        payload[offset + a : offset + a + b],
        payload[offset + a + b :],
    )
    restored = []
    for stream in streams:
        completed = subprocess.run(
            [brotli_binary, "-d", "-c"],
            input=stream,
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError("Brotli split-model parse-back failed")
        restored.append(completed.stdout)
    return restored[0], restored[1], restored[2]


def _lzma_models(models: bytes) -> bytes:
    return lzma.compress(models, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)


def _base_physical_models(archive: Path) -> bytes:
    outer = read_stored_member(archive)
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    models = decoder.decompress(outer)
    if not decoder.eof or not decoder.unused_data or len(models) != 74_860 or not models.startswith(b"F24S"):
        raise RuntimeError("PR135 physical model section failed parse-back")
    return models


def _physical_model_parts(parts, hpac: bytes, base_models: bytes) -> tuple[bytes, bytes, bytes, bytes]:
    if hpac[:6] != b"IHS2\x03\x31" or len(hpac) != 16_599:
        raise RuntimeError("unexpected HPAC representation")
    if len(base_models) != 74_860 or not base_models.startswith(b"F24S"):
        raise RuntimeError("unexpected base physical model section")
    hpac_body = hpac[6:]
    semantic_start = 4 + 16_593
    carrier_start = semantic_start + 36_040
    semantic_body = base_models[semantic_start:carrier_start]
    carrier_selector_body = base_models[carrier_start:]
    raw = base_models[:4] + hpac_body + semantic_body + carrier_selector_body
    if len(raw) != 74_860:
        raise RuntimeError(f"F24S physical model reconstruction has {len(raw)} bytes")
    return raw, hpac_body, semantic_body, carrier_selector_body


def _pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if bits <= 0 or np.any(values < 0) or np.any(values >= 1 << bits):
        raise RuntimeError("value is outside its packed unsigned field")
    output = bytearray((len(values) * bits + 7) // 8)
    offset = 0
    for value in values:
        byte, shift = divmod(offset, 8)
        word = int(value) << shift
        output[byte] |= word & 0xFF
        if shift + bits > 8:
            output[byte + 1] |= (word >> 8) & 0xFF
        offset += bits
    return bytes(output)


def _unpack_unsigned(raw: bytes, count: int, bits: int) -> np.ndarray:
    if len(raw) != (count * bits + 7) // 8:
        raise RuntimeError("packed unsigned field has the wrong length")
    if count * bits % 8 and raw[-1] >> (count * bits % 8):
        raise RuntimeError("packed unsigned field has nonzero padding")
    output = np.empty(count, dtype=np.int16)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = raw[byte]
        if byte + 1 < len(raw):
            word |= raw[byte + 1] << 8
        output[index] = (word >> shift) & ((1 << bits) - 1)
    return output


def pack_cap1_metadata(carrier_selector: bytes) -> tuple[bytes, dict[str, Any]]:
    """Pack fd135's exact CAP1 metadata hypothesis, retaining both bases."""

    if len(carrier_selector) != 22_223:
        raise RuntimeError("CAP1 metadata pack requires the exact F26 physical section")
    bit_counts = carrier_selector[:6]
    scales = carrier_selector[6:102]
    factors = np.frombuffer(carrier_selector[102:126], dtype="<i2").astype(np.int16)
    biases = np.frombuffer(carrier_selector[126:138], dtype=np.int8).astype(np.int16)
    lengths = np.frombuffer(carrier_selector[138:170], dtype=np.uint8).astype(np.int16)
    ks = np.frombuffer(carrier_selector[170:182], dtype=np.uint8).astype(np.int16)
    tail = carrier_selector[182:]
    factor_base = int(factors.min())
    k_base = int(ks.min())
    if not 0 <= factor_base <= 255 or not 0 <= k_base <= 255:
        raise RuntimeError("CAP1 packed base is outside u8")
    packed = (
        bit_counts
        + scales
        + bytes((factor_base,))
        + _pack_unsigned(factors - factor_base, 7)
        + _pack_unsigned(biases & 0x3F, 6)
        + _pack_unsigned(lengths, 4)
        + bytes((k_base,))
        + _pack_unsigned(ks - k_base, 1)
        + tail
    )
    restored = unpack_cap1_metadata(packed)
    if restored != carrier_selector:
        raise RuntimeError("CAP1 metadata pack did not restore its exact source")
    return packed, {
        "schema": "ddm_cp135_cap1_metadata_pack.v1",
        "source_bytes": len(carrier_selector),
        "source_sha256": sha256_bytes(carrier_selector),
        "payload_bytes": len(packed),
        "payload_sha256": sha256_bytes(packed),
        "raw_delta": len(packed) - len(carrier_selector),
        "factor_base": factor_base,
        "factor_delta_bits": 7,
        "bias_bits": 6,
        "length_bits": 4,
        "rice_k_base": k_base,
        "rice_k_delta_bits": 1,
    }


def unpack_cap1_metadata(packed: bytes) -> bytes:
    if len(packed) != 22_183:
        raise RuntimeError("packed CAP1 physical section has the wrong length")
    bit_counts = packed[:6]
    scales = packed[6:102]
    factor_base = packed[102]
    factors = factor_base + _unpack_unsigned(packed[103:114], 12, 7)
    bias_codes = _unpack_unsigned(packed[114:123], 12, 6)
    biases = np.where(bias_codes >= 32, bias_codes - 64, bias_codes).astype(np.int8)
    lengths = _unpack_unsigned(packed[123:139], 32, 4).astype(np.uint8)
    k_base = packed[139]
    ks = (k_base + _unpack_unsigned(packed[140:142], 12, 1)).astype(np.uint8)
    tail = packed[142:]
    result = (
        bit_counts
        + scales
        + factors.astype("<i2").tobytes()
        + biases.tobytes()
        + lengths.tobytes()
        + ks.tobytes()
        + tail
    )
    return result


def _optimal_split_models(
    sections: tuple[bytes, bytes, bytes],
    *,
    variant: Variant,
    representation: str,
    output: Path,
    brotli_binary: str,
) -> tuple[bytes, dict[str, Any]]:
    names = ("hpac", "semantic", "carrier_selector")
    selected_streams: list[bytes] = []
    selected_qualities: list[int] = []
    section_rows: list[dict[str, Any]] = []
    race_root = output / "retained" / "models" / variant / representation / "brotli_section_race"
    for name, source in zip(names, sections, strict=True):
        candidates = []
        for quality in range(12):
            stream = _brotli_compress(source, quality, brotli_binary)
            path = race_root / name / f"q{quality:02d}.br"
            atomic_bytes(path, stream)
            candidates.append(
                {
                    "quality": quality,
                    "payload": file_record(path),
                    "source_bytes": len(source),
                    "source_sha256": sha256_bytes(source),
                }
            )
        selected = min(
            candidates,
            key=lambda row: (row["payload"]["bytes"], row["quality"]),
        )
        selected_streams.append(Path(selected["payload"]["path"]).read_bytes())
        selected_qualities.append(int(selected["quality"]))
        section_rows.append(
            {
                "section": name,
                "denominator": len(candidates),
                "selection": "minimum retained bytes; lower quality breaks ties",
                "candidates": candidates,
                "selected": selected,
            }
        )
    if any(len(value) >= 1 << 16 for value in selected_streams):
        raise RuntimeError("selected split-model stream exceeds u16")
    payload = SPLIT_HEADER.pack(*(len(value) for value in selected_streams)) + b"".join(selected_streams)
    restored = unpack_split_models(payload, brotli_binary=brotli_binary)
    if restored != sections:
        raise RuntimeError("optimized split-model payload did not parse back")
    return payload, {
        "header_bytes": SPLIT_HEADER.size,
        "section_denominator": 12,
        "total_coder_runs": 36,
        "qualities": selected_qualities,
        "sections": section_rows,
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
    }


def _persist_candidate(
    *,
    args: argparse.Namespace,
    variant: Variant,
    model_name: str,
    model_payload: bytes,
    residual_payload: bytes,
    token_payload: bytes,
    token_name: str,
    model_report: dict[str, Any],
    hpac_report: dict[str, Any],
    symbol_identity_receipt: str,
    symbol_identity: bool,
) -> dict[str, Any]:
    member = model_payload + residual_payload + token_payload
    archive = deterministic_zip(member)
    repeat = deterministic_zip(member)
    candidate_root = args.output / "retained" / "candidates" / variant / f"{model_name}__{token_name}"
    paths = {
        "archive": candidate_root / "archive.zip",
        "repeat": candidate_root / "archive.repeat.zip",
        "model": candidate_root / "models.bin",
        "residual": candidate_root / "residual.compact.bin",
        "token": candidate_root / f"tokens.{token_name}",
        "member": candidate_root / "p",
    }
    for key, value in (
        ("model", model_payload),
        ("residual", residual_payload),
        ("token", token_payload),
        ("member", member),
        ("archive", archive),
        ("repeat", repeat),
    ):
        atomic_bytes(paths[key], value)
    if archive != repeat or read_stored_member(paths["archive"]) != member:
        raise RuntimeError("candidate repeat/ZIP parse-back failed")
    row = {
        "variant": variant,
        "model_codec": model_name,
        "token_codec": token_name,
        "archive": file_record(paths["archive"]),
        "repeat_archive": file_record(paths["repeat"]),
        "repeat_byte_identical": archive == repeat,
        "member": file_record(paths["member"]),
        "model_payload": file_record(paths["model"]),
        "residual_payload": file_record(paths["residual"]),
        "token_payload": file_record(paths["token"]),
        "model_report": model_report,
        "hpac_report": hpac_report,
        "symbol_identity_receipt": symbol_identity_receipt,
        "symbol_identity": symbol_identity,
        "archive_delta_vs_pr135": len(archive) - EXPECTED_ARCHIVE_BYTES,
        "projected_score_if_pr135_distortion_held": (
            BASE_SCORE + 25.0 * (len(archive) - EXPECTED_ARCHIVE_BYTES) / RATE_DENOMINATOR
        ),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
    }
    atomic_json(candidate_root / "RESULT.json", row)
    return row


def build_model_only_candidates(args: argparse.Namespace) -> dict[str, Any]:
    """Race lossless model representations while retaining PR135's RC64 stream."""

    require_base(args.archive)
    rc64_result_path = args.output / "retained" / "coders" / "control" / "RC64_RESULT.json"
    rc64_result = json.loads(rc64_result_path.read_text()) if rc64_result_path.is_file() else {}
    rc64_identity = bool(rc64_result.get("symbol_identity"))
    rc64_receipt = str(rc64_result_path) if rc64_identity else "pending full CPU RC64 parse-back"
    runtime_module = load_runtime(args.runtime)
    parts = runtime_module.read_residual_archive(args.archive)
    base_models = _base_physical_models(args.archive)
    raw, hpac, semantic, carrier = _physical_model_parts(parts, parts.hpac_blob, base_models)
    if raw != base_models:
        raise RuntimeError("model-only control did not reproduce physical F24S bytes")
    split, report = _optimal_split_models(
        (hpac, semantic, carrier),
        variant="control",
        representation="canonical_cap1",
        output=args.output,
        brotli_binary=args.brotli,
    )
    rows = []
    rows.append(
        _persist_candidate(
            args=args,
            variant="control",
            model_name="split_brotli_per_section_opt",
            model_payload=split,
            residual_payload=parts.residual_payload[4:],
            token_payload=parts.token_stream,
            token_name="rc64",
            model_report=report,
            hpac_report=variant_hpac(parts, "control")[1],
            symbol_identity_receipt=rc64_receipt,
            symbol_identity=rc64_identity,
        )
    )
    packed_carrier, cap1_report = pack_cap1_metadata(carrier)
    packed_split, packed_report = _optimal_split_models(
        (hpac, semantic, packed_carrier),
        variant="control",
        representation="packed_cap1_metadata",
        output=args.output,
        brotli_binary=args.brotli,
    )
    if unpack_cap1_metadata(unpack_split_models(packed_split, brotli_binary=args.brotli)[2]) != carrier:
        raise RuntimeError("packed CAP1 split candidate failed exact restoration")
    packed_report["cap1_metadata"] = cap1_report
    rows.append(
        _persist_candidate(
            args=args,
            variant="control",
            model_name="split_brotli_per_section_opt_cap1_metadata",
            model_payload=packed_split,
            residual_payload=parts.residual_payload[4:],
            token_payload=parts.token_stream,
            token_name="rc64",
            model_report=packed_report,
            hpac_report=variant_hpac(parts, "control")[1],
            symbol_identity_receipt=rc64_receipt,
            symbol_identity=rc64_identity,
        )
    )
    winner = min(rows, key=lambda candidate: candidate["archive"]["bytes"])
    result = {
        "schema": "ddm_cp135_model_only_result.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "base": file_record(args.archive),
        "candidates": rows,
        "winner": winner,
        "candidate_count": len(rows),
        "raw_model_identity": True,
        "whole_container_recount": True,
        "all_payloads_retained": True,
    }
    atomic_json(args.output / "MODEL_ONLY_RESULT.json", result)
    return result


def build_candidates(args: argparse.Namespace) -> dict[str, Any]:
    require_base(args.archive)
    runtime_module = load_runtime(args.runtime)
    base_parts = runtime_module.read_residual_archive(args.archive)
    base_models = _base_physical_models(args.archive)
    candidate_rows = []
    model_rows = []
    for variant in VARIANTS:
        ans_result_path = args.output / "retained" / "coders" / variant / "ANS_RESULT.json"
        if not ans_result_path.is_file():
            raise RuntimeError(f"missing ANS result for {variant}")
        ans_result = json.loads(ans_result_path.read_text())
        ans_path = Path(ans_result["token_payload"]["path"])
        ans = ans_path.read_bytes()
        token_choices = [("ans", ans, ans_result_path, ans_result)]
        if variant == "control":
            rc64_result_path = args.output / "retained" / "coders" / "control" / "RC64_RESULT.json"
            rc64_result = json.loads(rc64_result_path.read_text())
            token_choices.append(("rc64", base_parts.token_stream, rc64_result_path, rc64_result))
        else:
            rc64_result_path = args.output / "retained" / "coders" / variant / "FRESH_RC64_RESULT.json"
            if not rc64_result_path.is_file():
                raise RuntimeError(f"missing fresh RC64 result for {variant}")
            rc64_result = json.loads(rc64_result_path.read_text())
            rc64_path = Path(rc64_result["token_payload"]["path"])
            token_choices.append(("rc64", rc64_path.read_bytes(), rc64_result_path, rc64_result))
        hpac, hpac_report = variant_hpac(base_parts, variant)
        raw_models, hpac_body, semantic_body, carrier_selector_body = _physical_model_parts(
            base_parts, hpac, base_models
        )
        joint = _lzma_models(raw_models)
        model_root = args.output / "retained" / "models" / variant
        atomic_bytes(model_root / "models.f24s.raw", raw_models)
        atomic_bytes(model_root / "models.joint.raw_lzma2", joint)
        split, split_report = _optimal_split_models(
            (hpac_body, semantic_body, carrier_selector_body),
            variant=variant,
            representation="canonical_cap1",
            output=args.output,
            brotli_binary=args.brotli,
        )
        split_path = model_root / "models.split_brotli_per_section_opt"
        atomic_bytes(split_path, split)
        split_report["path"] = str(split_path)
        packed_carrier, cap1_report = pack_cap1_metadata(carrier_selector_body)
        packed_split, packed_report = _optimal_split_models(
            (hpac_body, semantic_body, packed_carrier),
            variant=variant,
            representation="packed_cap1_metadata",
            output=args.output,
            brotli_binary=args.brotli,
        )
        packed_path = model_root / "models.split_brotli_per_section_opt_cap1_metadata"
        atomic_bytes(packed_path, packed_split)
        packed_report["path"] = str(packed_path)
        packed_report["cap1_metadata"] = cap1_report
        choices = [
            (
                "joint_raw_lzma2",
                joint,
                {
                    "codec": "python-lzma raw LZMA2",
                    "fresh_recode": True,
                    "payload_bytes": len(joint),
                },
            ),
            ("split_brotli_per_section_opt", split, split_report),
            (
                "split_brotli_per_section_opt_cap1_metadata",
                packed_split,
                packed_report,
            ),
        ]
        for model_name, model_payload, model_report in choices:
            if model_name.startswith("split_brotli"):
                restored = unpack_split_models(model_payload, brotli_binary=args.brotli)
                restored_carrier = (
                    unpack_cap1_metadata(restored[2]) if model_name.endswith("cap1_metadata") else restored[2]
                )
                if (restored[0], restored[1], restored_carrier) != (
                    hpac_body,
                    semantic_body,
                    carrier_selector_body,
                ):
                    raise RuntimeError("candidate split-model parse-back differs")
            else:
                decoder = lzma.LZMADecompressor(
                    format=lzma.FORMAT_RAW,
                    filters=LZMA_FILTERS,
                )
                restored_raw = decoder.decompress(model_payload)
                if not decoder.eof or decoder.unused_data or restored_raw != raw_models:
                    raise RuntimeError("candidate joint-model parse-back differs")
            for token_name, token_payload, token_result_path, token_result in token_choices:
                row = _persist_candidate(
                    args=args,
                    variant=variant,
                    model_name=model_name,
                    model_payload=model_payload,
                    residual_payload=base_parts.residual_payload[4:],
                    token_payload=token_payload,
                    token_name=token_name,
                    model_report=model_report,
                    hpac_report=hpac_report,
                    symbol_identity_receipt=str(token_result_path),
                    symbol_identity=bool(token_result["symbol_identity"]),
                )
                candidate_rows.append(row)
        model_rows.append(
            {
                "variant": variant,
                "raw": file_record(model_root / "models.f24s.raw"),
                "joint": file_record(model_root / "models.joint.raw_lzma2"),
                "hpac": hpac_report,
            }
        )
    winner = min(candidate_rows, key=lambda row: row["archive"]["bytes"])
    result = {
        "schema": "ddm_cp135_build_result.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "base": file_record(args.archive),
        "models": model_rows,
        "candidates": candidate_rows,
        "winner": winner,
        "candidate_count": len(candidate_rows),
        "all_payloads_retained": True,
        "whole_container_recount": True,
    }
    atomic_json(args.output / "BUILD_RESULT.json", result)
    return result


def parseback_candidate(args: argparse.Namespace) -> dict[str, Any]:
    """Prove the adapted receiver sees the promoted candidate's intended values."""

    require_base(args.archive)
    build_path = args.output / "BUILD_RESULT.json"
    if not build_path.is_file():
        raise RuntimeError("missing final candidate build receipt")
    build = json.loads(build_path.read_text())
    winner = build["winner"]
    candidate_path = args.runtime / "archive.zip"
    candidate_record = file_record(candidate_path)
    if (
        candidate_record["bytes"] != winner["archive"]["bytes"]
        or candidate_record["sha256"] != winner["archive"]["sha256"]
    ):
        raise RuntimeError("adapted runtime archive differs from the retained winner")
    if winner["variant"] != "hp3_step2" or winner["token_codec"] != "rc64":
        raise RuntimeError("unexpected promoted candidate formulation")

    import importlib

    sys.path.insert(0, str(args.runtime.resolve()))
    try:
        archive_module = importlib.import_module("runtime.residual_archive")
    finally:
        sys.path.pop(0)
    base = archive_module.read_residual_archive(args.archive)
    candidate = archive_module.read_residual_archive(candidate_path)
    target_hpac, hpac_report = step2_ihs2(base.hpac_blob)
    token_result_path = args.output / "retained" / "coders" / "hp3_step2" / "FRESH_RC64_RESULT.json"
    token_result = json.loads(token_result_path.read_text())
    retained_token_path = Path(token_result["token_payload"]["path"])

    identities = {
        "semantic_blob": candidate.semantic_blob == base.semantic_blob,
        "carrier_blob": candidate.carrier_blob == base.carrier_blob,
        "hpac_step2_blob": candidate.hpac_blob == target_hpac,
        "residual_payload": candidate.residual_payload == base.residual_payload,
        "token_stream": candidate.token_stream == retained_token_path.read_bytes(),
    }
    if not all(identities.values()):
        raise RuntimeError(f"adapted receiver value mismatch: {identities}")
    if token_result.get("symbol_identity") is not True:
        raise RuntimeError("promoted token stream lacks a full symbol-identity receipt")

    result = {
        "schema": "ddm_cp135_parseback.v2",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "candidate_archive": candidate_record,
        "retained_winner_archive": winner["archive"],
        "build_receipt": file_record(build_path),
        "base_archive": file_record(args.archive),
        "semantic_blob": {
            "bytes": len(candidate.semantic_blob),
            "sha256": sha256_bytes(candidate.semantic_blob),
            "identity_to_pr135": identities["semantic_blob"],
        },
        "carrier_blob": {
            "bytes": len(candidate.carrier_blob),
            "sha256": sha256_bytes(candidate.carrier_blob),
            "identity_to_pr135": identities["carrier_blob"],
        },
        "hpac_blob": {
            "bytes": len(candidate.hpac_blob),
            "sha256": sha256_bytes(candidate.hpac_blob),
            "identity_to_retained_hp3_step2": identities["hpac_step2_blob"],
            "derivation": hpac_report,
        },
        "residual_payload": {
            "bytes": len(candidate.residual_payload),
            "sha256": sha256_bytes(candidate.residual_payload),
            "identity_to_pr135": identities["residual_payload"],
        },
        "token_stream": {
            "bytes": len(candidate.token_stream),
            "sha256": sha256_bytes(candidate.token_stream),
            "identity_to_retained_hp3_rc64": identities["token_stream"],
            "full_symbol_identity": True,
            "events": int(token_result["events"]),
            "symbol_identity_receipt": file_record(token_result_path),
        },
        "receiver_values_equal_by_construction": True,
        "literal_frame_render_performed": False,
        "cuda_exact_eval_queued_to_main": True,
        "adapted_runtime": tree_record(args.runtime),
    }
    atomic_json(args.output / "PARSEBACK_RESULT.json", result)
    return result


def finalize_result(args: argparse.Namespace) -> dict[str, Any]:
    """Seal the whole-container lever ledger from retained machine receipts."""

    build_path = args.output / "BUILD_RESULT.json"
    parseback_path = args.output / "PARSEBACK_RESULT.json"
    corpus_path = args.output / "CORPUS_CODEC_RACE_RESULT.json"
    lotto_path = args.output / "LOTTO_RENDERER_RACE_RESULT.json"
    for path in (build_path, parseback_path, corpus_path, lotto_path):
        if not path.is_file():
            raise RuntimeError(f"missing final receipt: {path}")
    build = json.loads(build_path.read_text())
    parseback = json.loads(parseback_path.read_text())
    corpus = json.loads(corpus_path.read_text())
    lotto = json.loads(lotto_path.read_text())

    def candidate(variant: str, model: str, token: str) -> dict[str, Any]:
        matches = [
            row
            for row in build["candidates"]
            if row["variant"] == variant and row["model_codec"] == model and row["token_codec"] == token
        ]
        if len(matches) != 1:
            raise RuntimeError(f"candidate lookup differs: {variant}/{model}/{token}")
        return matches[0]

    control_split = candidate("control", "split_brotli_per_section_opt", "rc64")
    control_cap = candidate("control", "split_brotli_per_section_opt_cap1_metadata", "rc64")
    control_ans = candidate("control", "split_brotli_per_section_opt_cap1_metadata", "ans")
    hp3_rc64 = candidate("hp3_step2", "split_brotli_per_section_opt_cap1_metadata", "rc64")
    hp3_ans = candidate("hp3_step2", "split_brotli_per_section_opt_cap1_metadata", "ans")
    winner = build["winner"]
    if winner != hp3_rc64 or parseback.get("receiver_values_equal_by_construction") is not True:
        raise RuntimeError("final winner or parse-back seal differs")

    ledger = [
        {
            "lever": "VP1 split-model Brotli per physical section",
            "from_archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "to_archive_bytes": control_split["archive"]["bytes"],
            "delta_bytes": control_split["archive"]["bytes"] - EXPECTED_ARCHIVE_BYTES,
            "absorbed": False,
            "receipt": control_split["archive"],
        },
        {
            "lever": "CAP1 metadata fixed-field pack",
            "from_archive_bytes": control_split["archive"]["bytes"],
            "to_archive_bytes": control_cap["archive"]["bytes"],
            "delta_bytes": control_cap["archive"]["bytes"] - control_split["archive"]["bytes"],
            "raw_section_delta_bytes": -40,
            "absorbed": False,
            "receipt": control_cap["archive"],
        },
        {
            "lever": "HP3 IHS2 step2 plus exact F26 probability recode",
            "from_archive_bytes": control_cap["archive"]["bytes"],
            "to_archive_bytes": hp3_rc64["archive"]["bytes"],
            "delta_bytes": hp3_rc64["archive"]["bytes"] - control_cap["archive"]["bytes"],
            "model_delta_bytes": hp3_rc64["model_payload"]["bytes"] - control_cap["model_payload"]["bytes"],
            "token_delta_bytes": hp3_rc64["token_payload"]["bytes"] - control_cap["token_payload"]["bytes"],
            "absorbed": False,
            "receipt": hp3_rc64["archive"],
        },
        {
            "lever": "LC2 same-state ANS token recode",
            "control_delta_vs_rc64_bytes": control_ans["archive"]["bytes"] - control_cap["archive"]["bytes"],
            "hp3_delta_vs_rc64_bytes": hp3_ans["archive"]["bytes"] - hp3_rc64["archive"]["bytes"],
            "absorbed": True,
            "verdict_scope": "INSTANCE: exact PR135 control and HP3-step2 probability states",
        },
        {
            "lever": "SMEVR recalled R7 nibble coder",
            "section_denominator": corpus["section_denominator"],
            "coder_runs_per_section": corpus["coders_per_section"],
            "section_wins": sum(row["winner"]["codec"] == "smevr_r7_nibble" for row in corpus["sections"]),
            "absorbed": True,
            "verdict_scope": corpus["verdict_scope"],
        },
        {
            "lever": "LOTTO shared-dictionary and supermask renderer recodes",
            "row_denominator": lotto["row_denominator"],
            "best_delta_vs_selected_wans_brotli_bytes": min(
                row["delta_vs_selected_wans_brotli"] for row in lotto["rows"]
            ),
            "absorbed": True,
            "verdict_scope": lotto["verdict_scope"],
        },
    ]
    fire_order_path = args.output / "EXACT_EVAL_FIRE_ORDER.json"
    fire_order = {
        "schema": "ddm_cp135_exact_eval_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN exact contest-row owner",
        "consumer_store": str(args.output / "exact_eval"),
        "suggested_lane_id": "lane_ddm_cp135_exact_20260810_contest_cuda",
        "single_flight": True,
        "axis": "[contest-CUDA T4, locked upstream venv, n600]",
        "source_archive": parseback["candidate_archive"],
        "adapted_runtime": {
            "root": parseback["adapted_runtime"]["root"],
            "tree_sha256": parseback["adapted_runtime"]["tree_sha256"],
            "file_count": parseback["adapted_runtime"]["file_count"],
        },
        "fire_trigger": (
            "MAIN owns the sole exact-eval lane; archive size and SHA match; the locked T4 "
            "environment passes the Brotli==1.2.0 install/import preflight; then run exactly "
            "one n600 upstream/evaluate.py contest-CUDA row"
        ),
        "promotion_gate": (
            "harvested evaluator input archive matches the pinned bytes and the exact recomputed "
            "score is below the current qualifying pointer"
        ),
        "producer_dispatched_modal": False,
    }
    atomic_json(fire_order_path, fire_order)
    result = {
        "schema": "ddm_cp135_final_result.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "base": build["base"],
        "winner": winner,
        "archive_delta_bytes": winner["archive"]["bytes"] - EXPECTED_ARCHIVE_BYTES,
        "derived_score_if_pr135_distortion_holds": winner["projected_score_if_pr135_distortion_held"],
        "exact_cuda_score_measured": False,
        "literal_frame_render_performed": False,
        "receiver_values_equal_by_construction": True,
        "per_lever_ledger": ledger,
        "receipts": {
            "build": file_record(build_path),
            "parseback": file_record(parseback_path),
            "corpus_coder_race": file_record(corpus_path),
            "lotto_renderer_race": file_record(lotto_path),
            "exact_eval_fire_order": file_record(fire_order_path),
        },
        "borrowed_substrate_accounting": {
            "borrowed": "PR135 archive, learned state, and CUDA renderer runtime from codexblack",
            "ours_original": (
                "VP1 split representation, CAP1 fixed-field metadata pack, HP3 step2 "
                "composition with exact recodes, and the composition/receiver-equality harness"
            ),
        },
        "next_disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "next_owner": "MAIN exact contest-row owner",
    }
    atomic_json(args.output / "FINAL_RESULT.json", result)
    return result


def _raw_lzma_roundtrip(value: bytes) -> bytes:
    payload = _lzma_models(value)
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    restored = decoder.decompress(payload)
    if not decoder.eof or decoder.unused_data or restored != value:
        raise RuntimeError("raw LZMA2 section race failed exact parse-back")
    return payload


def _smevr_roundtrip(value: bytes) -> bytes:
    """Apply the recalled R7 nibble SMEVR record framing to arbitrary bytes."""

    import ddm_bd1_class_field_receiver as bd1

    # R7's four-dimensional frame uses uint16 dimensions.  The recalled BD1
    # wrapper only chunks on total values, so split long one-record sections
    # into fixed records before entering it.
    records = [value[start : start + 32_760] for start in range(0, len(value), 32_760)]
    if not records:
        records = [b""]
    payload = bd1.smevr_records(records)
    if b"".join(bd1.unsmevr_records(payload)) != value:
        raise RuntimeError("SMEVR section race failed exact parse-back")
    return payload


def race_corpus_codecs(args: argparse.Namespace) -> dict[str, Any]:
    """Race recalled SMEVR on every physical/decomposed PR135 byte section."""

    require_base(args.archive)
    runtime_module = load_runtime(args.runtime)
    parts = runtime_module.read_residual_archive(args.archive)
    models = _base_physical_models(args.archive)
    _, hpac, semantic, carrier = _physical_model_parts(parts, parts.hpac_blob, models)
    hp3_blob, _ = step2_ihs2(parts.hpac_blob)
    hp3_models, hp3_hpac, _, _ = _physical_model_parts(parts, hp3_blob, models)
    packed_carrier, _ = pack_cap1_metadata(carrier)
    sources = {
        "model_raw": models,
        "model_physical_raw_lzma2": parts.compressed_models,
        "hpac_ihs2_body": hpac,
        "hp3_model_raw": hp3_models,
        "hp3_ihs2_body": hp3_hpac,
        "semantic_wans_f12_body": semantic,
        "carrier_selector_cap1_body": carrier,
        "carrier_cap1_only": carrier[:-9],
        "carrier_cap1_metadata": carrier[:182],
        "carrier_cap1_coefficient_streams": carrier[182:-9],
        "frame0_selector_body": carrier[-9:],
        "carrier_selector_cap1_packed_metadata": packed_carrier,
        "residual_compact_body": parts.residual_payload[4:],
        "token_rc64_stream": parts.token_stream,
    }
    root = args.output / "retained" / "corpus_codec_race"
    rows = []
    for name, source in sources.items():
        section_root = root / name
        atomic_bytes(section_root / "source.bin", source)
        candidates = []
        identity_path = section_root / "identity.bin"
        atomic_bytes(identity_path, source)
        candidates.append({"codec": "identity", "payload": file_record(identity_path)})
        lzma_payload = _raw_lzma_roundtrip(source)
        lzma_path = section_root / "raw_lzma2.bin"
        atomic_bytes(lzma_path, lzma_payload)
        candidates.append({"codec": "raw_lzma2", "payload": file_record(lzma_path)})
        for quality in range(12):
            payload = _brotli_compress(source, quality, args.brotli)
            completed = subprocess.run([args.brotli, "-d", "-c"], input=payload, check=False, capture_output=True)
            if completed.returncode or completed.stdout != source:
                raise RuntimeError(f"Brotli q{quality} section race failed exact parse-back")
            path = section_root / f"brotli_q{quality:02d}.br"
            atomic_bytes(path, payload)
            candidates.append({"codec": f"brotli_q{quality}", "payload": file_record(path)})
        smevr_payload = _smevr_roundtrip(source)
        smevr_path = section_root / "smevr_r7_nibble.cgsv1"
        atomic_bytes(smevr_path, smevr_payload)
        candidates.append({"codec": "smevr_r7_nibble", "payload": file_record(smevr_path)})
        winner = min(candidates, key=lambda row: (row["payload"]["bytes"], row["codec"]))
        rows.append(
            {
                "section": name,
                "source": file_record(section_root / "source.bin"),
                "candidates": candidates,
                "denominator": len(candidates),
                "winner": winner,
                "smevr_delta_vs_identity": (
                    next(row for row in candidates if row["codec"] == "smevr_r7_nibble")["payload"]["bytes"]
                    - len(source)
                ),
                "selection": "minimum retained bytes; lexical codec name breaks ties",
            }
        )
    result = {
        "schema": "ddm_cp135_corpus_codec_race.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "archive": file_record(args.archive),
        "sections": rows,
        "section_denominator": len(rows),
        "coders_per_section": 15,
        "all_payloads_retained": True,
        "verdict_scope": "INSTANCE: exact custodied PR135 physical/decomposed sections",
    }
    atomic_json(args.output / "CORPUS_CODEC_RACE_RESULT.json", result)
    return result


def _pack_nibbles_low_first(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if np.any(values >= 16):
        raise RuntimeError("LOTTO selector escaped its four-bit dictionary")
    packed = np.zeros((values.size + 1) // 2, dtype=np.uint8)
    packed[:] = values[0::2]
    packed[: values.size // 2] |= values[1::2] << 4
    return packed.tobytes()


def _unpack_nibbles_low_first(raw: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(raw, dtype=np.uint8)
    if len(raw) != (count + 1) // 2 or (count % 2 and packed[-1] >> 4):
        raise RuntimeError("LOTTO selector framing differs")
    values = np.empty(count, dtype=np.uint8)
    values[0::2] = packed & 15
    values[1::2] = packed[: count // 2] >> 4
    return values


def race_lotto_renderer(args: argparse.Namespace) -> dict[str, Any]:
    """Race exact shared-dictionary/supermask forms of PR135's W4 renderer."""

    require_base(args.archive)
    import importlib

    runtime_module = load_runtime(args.runtime)
    archive_module = importlib.import_module("runtime.residual_archive")
    parts = runtime_module.read_residual_archive(args.archive)
    book_src = args.experiment_book / "src"
    sys.path.insert(0, str(book_src))
    try:
        from cpr1_sub4.baseline import TensorStorage
        from cpr1_sub4.entropy.renderer_weight_codec import (
            decode_wans1,
            encode_f12_wans_body,
            encode_wans1,
        )
    finally:
        sys.path.pop(0)
    records = decode_wans1(parts.semantic_blob)
    metadata = b"".join((record.raw_fp16 if record.schema.is_fp16 else record.raw_scales) or b"" for record in records)
    codes = np.concatenate([record.codes.reshape(-1) for record in records if not record.schema.is_fp16]).astype(
        np.int8
    )
    direct_selectors = (codes.astype(np.int16) + 7).astype(np.uint8)
    nonzero = codes != 0
    nonzero_codes = codes[nonzero].astype(np.int16)
    sparse_selectors = np.where(nonzero_codes < 0, nonzero_codes + 7, nonzero_codes + 6).astype(np.uint8)
    representations = {
        "shared_dictionary": b"LTS1" + metadata + _pack_nibbles_low_first(direct_selectors),
        "supermask_shared_dictionary": (
            b"LTS2"
            + metadata
            + np.packbits(nonzero, bitorder="little").tobytes()
            + _pack_nibbles_low_first(sparse_selectors)
        ),
    }

    def restore(raw: bytes) -> tuple[Any, ...]:
        if raw[:4] not in {b"LTS1", b"LTS2"}:
            raise RuntimeError("LOTTO representation magic differs")
        offset = 4
        restored = []
        code_offset = 0
        if raw[:4] == b"LTS1":
            selector_offset = 4 + len(metadata)
            decoded_codes = _unpack_nibbles_low_first(raw[selector_offset:], codes.size).astype(np.int16) - 7
        else:
            selector_offset = 4 + len(metadata)
            mask_bytes = (codes.size + 7) // 8
            packed_mask = raw[selector_offset : selector_offset + mask_bytes]
            bits = np.unpackbits(np.frombuffer(packed_mask, dtype=np.uint8), bitorder="little")
            if np.any(bits[codes.size :]):
                raise RuntimeError("LOTTO supermask has nonzero padding")
            mask = bits[: codes.size].astype(bool)
            selected = _unpack_nibbles_low_first(raw[selector_offset + mask_bytes :], int(mask.sum())).astype(np.int16)
            selected = np.where(selected < 7, selected - 7, selected - 6)
            decoded_codes = np.zeros(codes.size, dtype=np.int16)
            decoded_codes[mask] = selected
        for source in records:
            schema = source.schema
            if schema.is_fp16:
                size = schema.count * 2
                raw_fp16 = raw[offset : offset + size]
                offset += size
                values = np.frombuffer(raw_fp16, dtype="<f2").astype(np.float32).reshape(schema.shape)
                restored.append(TensorStorage(schema, "fp16", values, None, None, raw_fp16=raw_fp16))
                continue
            scale_size = schema.scale_count * 2
            raw_scales = raw[offset : offset + scale_size]
            offset += scale_size
            scales = np.frombuffer(raw_scales, dtype="<f2").astype(np.float32)
            tensor_codes = decoded_codes[code_offset : code_offset + schema.count].astype(np.int8)
            code_offset += schema.count
            scale_shape = [1] * len(schema.shape)
            scale_shape[-1 if schema.name.endswith("embed.weight") else 0] = schema.scale_count
            tensor_codes = tensor_codes.reshape(schema.shape)
            restored.append(
                TensorStorage(
                    schema,
                    "w4",
                    tensor_codes.astype(np.float32) * scales.reshape(scale_shape),
                    scales,
                    tensor_codes,
                    raw_scales=raw_scales,
                )
            )
        if offset != 4 + len(metadata) or code_offset != codes.size:
            raise RuntimeError("LOTTO shared metadata/schema accounting differs")
        return tuple(restored)

    root = args.output / "retained" / "lotto_renderer_race"
    rows = []
    semantic_body = _physical_model_parts(parts, parts.hpac_blob, _base_physical_models(args.archive))[2]
    for name, raw in representations.items():
        representation_root = root / name
        raw_path = representation_root / "renderer.lotto.raw"
        atomic_bytes(raw_path, raw)
        restored_records = restore(raw)
        canonical, _ = encode_wans1(restored_records, strategy="global")
        if canonical != parts.semantic_blob:
            raise RuntimeError("LOTTO representation did not restore canonical WANS1 state")
        restored_f12 = encode_f12_wans_body(canonical, archive_module.WANS_STREAM_ORDER)
        if restored_f12 != semantic_body:
            raise RuntimeError("LOTTO representation did not restore exact F12 physical bytes")
        candidates = []
        for quality in range(12):
            payload = _brotli_compress(raw, quality, args.brotli)
            completed = subprocess.run([args.brotli, "-d", "-c"], input=payload, check=False, capture_output=True)
            if completed.returncode or completed.stdout != raw:
                raise RuntimeError("LOTTO Brotli race failed exact parse-back")
            path = representation_root / f"brotli_q{quality:02d}.br"
            atomic_bytes(path, payload)
            candidates.append({"codec": f"brotli_q{quality}", "payload": file_record(path)})
        smevr = _smevr_roundtrip(raw)
        smevr_path = representation_root / "smevr_r7_nibble.cgsv1"
        atomic_bytes(smevr_path, smevr)
        candidates.append({"codec": "smevr_r7_nibble", "payload": file_record(smevr_path)})
        winner = min(candidates, key=lambda row: (row["payload"]["bytes"], row["codec"]))
        rows.append(
            {
                "representation": name,
                "raw": file_record(raw_path),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "winner": winner,
                "delta_vs_selected_wans_brotli": winner["payload"]["bytes"] - 34_763,
                "restores_canonical_wans1": True,
                "restores_physical_f12": True,
            }
        )
    result = {
        "schema": "ddm_cp135_lotto_renderer_race.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "archive": file_record(args.archive),
        "w4_code_values": int(codes.size),
        "w4_nonzero_values": int(nonzero.sum()),
        "w4_nonzero_fraction": float(nonzero.mean()),
        "implicit_shared_dictionary": list(range(-7, 8)),
        "rows": rows,
        "row_denominator": len(rows),
        "all_payloads_retained": True,
        "verdict_scope": "INSTANCE: exact PR135 W4 renderer state, lossless same-state forms",
    }
    atomic_json(args.output / "LOTTO_RENDERER_RACE_RESULT.json", result)
    return result


RC64_STATE_HEADER = struct.Struct("<4sQQQBBQ")


def _compile_checkpointable_rc64(args: argparse.Namespace) -> Path:
    """Compile the ExperimentBook RC64 encoder with an exact state ABI."""

    source_path = args.experiment_book / "src" / "cpr1_sub4" / "entropy" / "rc64_backend.c"
    source = source_path.read_text()
    marker = "int rc64_encoder_encode(\n"
    if source.count(marker) != 1:
        raise RuntimeError("RC64 checkpoint injection marker differs")
    injection = r"""
int rc64_encoder_snapshot(
    const void *opaque,
    uint64_t *low,
    uint64_t *high,
    uint64_t *pending,
    uint8_t *partial,
    uint8_t *partial_bits,
    const uint8_t **data,
    size_t *size
) {
    const rc64_encoder *encoder = (const rc64_encoder *)opaque;
    if (!encoder || encoder->error || encoder->finished || !low || !high ||
        !pending || !partial || !partial_bits || !data || !size) return -1;
    *low = encoder->low;
    *high = encoder->high;
    *pending = encoder->pending;
    *partial = encoder->partial;
    *partial_bits = encoder->partial_bits;
    *data = encoder->data;
    *size = encoder->size;
    return 0;
}

void *rc64_encoder_resume(
    uint64_t low,
    uint64_t high,
    uint64_t pending,
    uint8_t partial,
    uint8_t partial_bits,
    const uint8_t *data,
    size_t size
) {
    rc64_encoder *encoder = (rc64_encoder *)rc64_encoder_create();
    if (!encoder || partial_bits > 7u || low > high || high > RC64_TOP) {
        rc64_encoder_destroy(encoder);
        return NULL;
    }
    if (size && (!data || !rc64_reserve(encoder, size))) {
        rc64_encoder_destroy(encoder);
        return NULL;
    }
    if (size) memcpy(encoder->data, data, size);
    encoder->size = size;
    encoder->low = low;
    encoder->high = high;
    encoder->pending = pending;
    encoder->partial = partial;
    encoder->partial_bits = partial_bits;
    return encoder;
}

"""
    generated = source.replace(marker, injection + marker)
    generated_path = args.output / "work" / "rc64_checkpoint_backend.c"
    library_path = args.output / "work" / "librc64_checkpoint.so"
    atomic_bytes(generated_path, generated.encode())
    completed = subprocess.run(
        ["cc", "-std=c11", "-O3", "-fPIC", "-shared", str(generated_path), "-o", str(library_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"checkpointable RC64 compilation failed: {completed.stderr}")
    return library_path


def _rc64_snapshot(encoder: Any) -> bytes:
    import ctypes

    library = encoder.library
    u8_pointer = ctypes.POINTER(ctypes.c_uint8)
    library.rc64_encoder_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(u8_pointer),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.rc64_encoder_snapshot.restype = ctypes.c_int
    low, high, pending = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
    partial, partial_bits = ctypes.c_uint8(), ctypes.c_uint8()
    data, size = u8_pointer(), ctypes.c_size_t()
    status = library.rc64_encoder_snapshot(
        encoder.context,
        ctypes.byref(low),
        ctypes.byref(high),
        ctypes.byref(pending),
        ctypes.byref(partial),
        ctypes.byref(partial_bits),
        ctypes.byref(data),
        ctypes.byref(size),
    )
    if status:
        raise RuntimeError(f"RC64 checkpoint snapshot failed with status {status}")
    body = ctypes.string_at(data, size.value) if size.value else b""
    return (
        RC64_STATE_HEADER.pack(
            b"R6S1",
            low.value,
            high.value,
            pending.value,
            partial.value,
            partial_bits.value,
            size.value,
        )
        + body
    )


def _rc64_resume(native_encoder: Any, library_path: Path, state: bytes) -> Any:
    import ctypes

    if len(state) < RC64_STATE_HEADER.size:
        raise RuntimeError("RC64 checkpoint is truncated")
    magic, low, high, pending, partial, partial_bits, size = RC64_STATE_HEADER.unpack_from(state)
    body = state[RC64_STATE_HEADER.size :]
    if magic != b"R6S1" or len(body) != size:
        raise RuntimeError("RC64 checkpoint framing differs")
    encoder = object.__new__(native_encoder)
    encoder.library = native_encoder.__init__.__globals__["_library"](library_path)
    u8_pointer = ctypes.POINTER(ctypes.c_uint8)
    encoder.library.rc64_encoder_resume.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint8,
        ctypes.c_uint8,
        u8_pointer,
        ctypes.c_size_t,
    ]
    encoder.library.rc64_encoder_resume.restype = ctypes.c_void_p
    storage = (ctypes.c_uint8 * len(body)).from_buffer_copy(body) if body else None
    pointer = ctypes.cast(storage, u8_pointer) if storage is not None else u8_pointer()
    encoder.context = encoder.library.rc64_encoder_resume(low, high, pending, partial, partial_bits, pointer, len(body))
    if not encoder.context:
        raise RuntimeError("RC64 checkpoint resume failed")
    encoder.finished = False
    return encoder


def encode_rc64(args: argparse.Namespace) -> dict[str, Any]:
    """Fresh, resumable RC64 recode of one retained probability variant."""

    variant: Variant = args.variant
    export_path = args.output / "retained" / "probabilities" / variant / "EXPORT_RESULT.json"
    export = json.loads(export_path.read_text())
    if not export.get("complete_n600"):
        raise RuntimeError(f"{variant} probability export is incomplete")
    source = SourceSymbols(args.dt1_manifest)
    library_path = _compile_checkpointable_rc64(args)
    book_src = args.experiment_book / "src"
    sys.path.insert(0, str(book_src))
    try:
        from cpr1_sub4.entropy.rc64 import NativeDecoder, NativeEncoder
    finally:
        sys.path.pop(0)
    retained = args.output / "retained" / "coders" / variant
    checkpoint_root = retained / "rc64_checkpoints"
    progress_path = checkpoint_root / "LATEST.json"
    start_frame = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        state_path = Path(progress["state"]["path"])
        if file_record(state_path) != progress["state"]:
            raise RuntimeError("RC64 checkpoint failed custody")
        encoder = _rc64_resume(NativeEncoder, library_path, state_path.read_bytes())
        start_frame = int(progress["next_frame"])
    else:
        encoder = NativeEncoder(library_path)
    started = time.time()
    for frame in range(start_frame, 600):
        probabilities = probability_from_codes(_load_codes(args.output, variant, frame), 8)
        encoder.encode(source.frame(frame).astype(np.int32), probabilities)
        if (frame + 1) % 24 == 0 or frame == 599:
            state = _rc64_snapshot(encoder)
            state_path = checkpoint_root / f"through_frame_{frame:04d}.rc64.state"
            atomic_bytes(state_path, state)
            receipt = {
                "schema": "ddm_cp135_rc64_checkpoint.v1",
                "variant": variant,
                "through_frame": frame,
                "next_frame": frame + 1,
                "state": file_record(state_path),
                "probability_export": file_record(export_path),
            }
            atomic_json(state_path.with_suffix(".json"), receipt)
            atomic_json(progress_path, receipt)
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "rc64_encoded_frames": frame + 1,
                        "elapsed_s": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
    payload = encoder.finish()
    token_path = retained / "tokens.rc64"
    atomic_bytes(token_path, payload)
    if variant == "control" and (len(payload) != EXPECTED_RC64_BYTES or sha256_bytes(payload) != EXPECTED_RC64_SHA256):
        raise RuntimeError("fresh control RC64 stream differs from custodied PR135")

    decoder = NativeDecoder(library_path, payload)
    decoded_path = retained / "decoded_symbols.fresh_rc64.bin"
    spatial_path = retained / "decoded_spatial_tokens.fresh_rc64.bin"
    temporary = decoded_path.with_name(f".{decoded_path.name}.{os.getpid()}.tmp")
    spatial_temporary = spatial_path.with_name(f".{spatial_path.name}.{os.getpid()}.tmp")
    event_digest = hashlib.sha256()
    spatial_digest = hashlib.sha256()
    group_positions = _group_positions(args.runtime)
    with temporary.open("wb") as stream, spatial_temporary.open("wb") as spatial_stream:
        for frame in range(600):
            probabilities = probability_from_codes(_load_codes(args.output, variant, frame), 8)
            decoded = decoder.decode(probabilities).astype(np.uint8)
            expected = source.frame(frame)
            if not np.array_equal(decoded, expected):
                raise RuntimeError(f"fresh RC64 symbols differ at frame {frame}")
            raw = decoded.tobytes()
            spatial_raw = spatial_frame(decoded, group_positions).tobytes()
            stream.write(raw)
            spatial_stream.write(spatial_raw)
            event_digest.update(raw)
            spatial_digest.update(spatial_raw)
        stream.flush()
        os.fsync(stream.fileno())
        spatial_stream.flush()
        os.fsync(spatial_stream.fileno())
    os.replace(temporary, decoded_path)
    os.replace(spatial_temporary, spatial_path)
    if (
        event_digest.hexdigest() != EXPECTED_EVENT_ORDER_SHA256
        or spatial_digest.hexdigest() != EXPECTED_SPATIAL_TOKEN_SHA256
    ):
        raise RuntimeError("fresh RC64 decoded-token digest differs")
    result = {
        "schema": "ddm_cp135_fresh_rc64_result.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": variant,
        "token_payload": file_record(token_path),
        "decoded_symbols": file_record(decoded_path),
        "decoded_spatial_tokens": file_record(spatial_path),
        "decoded_event_order_sha256": event_digest.hexdigest(),
        "decoded_spatial_token_sha256": spatial_digest.hexdigest(),
        "symbol_identity": True,
        "events": EXPECTED_EVENTS,
        "decoder_bit_position": decoder.bit_position,
        "resumable_from_disk": True,
        "checkpoint_count": len(list(checkpoint_root.glob("through_frame_*.rc64.state"))),
        "source_backend": file_record(args.experiment_book / "src" / "cpr1_sub4" / "entropy" / "rc64_backend.c"),
        "checkpoint_backend": file_record(args.output / "work" / "rc64_checkpoint_backend.c"),
        "library": file_record(library_path),
        "wall_s": time.time() - started,
    }
    atomic_json(retained / "FRESH_RC64_RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "stage",
        choices=(
            "export",
            "derive-hp3",
            "encode-ans",
            "encode-rc64",
            "verify-rc64",
            "race-corpus-codecs",
            "race-lotto-renderer",
            "build-models",
            "build",
            "parseback",
            "finalize",
        ),
    )
    value.add_argument("--variant", choices=VARIANTS, default="control")
    value.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    value.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--dt1-manifest", type=Path, default=DEFAULT_DT1_MANIFEST)
    value.add_argument("--hp3-manifest", type=Path, default=DEFAULT_HP3_MANIFEST)
    value.add_argument("--experiment-book", type=Path, default=DEFAULT_EXPERIMENT_BOOK)
    value.add_argument("--start-frame", type=int, default=0)
    value.add_argument("--end-frame", type=int, default=600)
    value.add_argument("--torch-threads", type=int, default=4)
    value.add_argument("--brotli", default=shutil.which("brotli") or "brotli")
    return value


def main() -> None:
    args = parser().parse_args()
    if not 0 <= args.start_frame < args.end_frame <= 600:
        raise SystemExit("invalid frame interval")
    if args.stage == "export":
        import torch

        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        result = export_probabilities(args)
    elif args.stage == "derive-hp3":
        result = derive_hp3_probabilities(args)
    elif args.stage == "encode-ans":
        result = encode_ans(args)
    elif args.stage == "encode-rc64":
        result = encode_rc64(args)
    elif args.stage == "verify-rc64":
        result = verify_rc64(args)
    elif args.stage == "race-corpus-codecs":
        result = race_corpus_codecs(args)
    elif args.stage == "race-lotto-renderer":
        result = race_lotto_renderer(args)
    elif args.stage == "build-models":
        result = build_model_only_candidates(args)
    elif args.stage == "parseback":
        result = parseback_candidate(args)
    elif args.stage == "finalize":
        result = finalize_result(args)
    else:
        result = build_candidates(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
