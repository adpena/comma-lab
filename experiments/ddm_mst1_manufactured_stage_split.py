#!/usr/bin/env python3
"""Retain and analyse the exact DX2 manufactured-Seg stage split.

This module is copied into an *instrumented copy* of the shipped DX2 renderer
and called from ``cpr1/inflate.py::render_video``.  The shipped runtime remains
read-only.  During the canonical local advisory fire, :class:`StageCapture`
retains the renderer's native 384x512 float field, the bilinear camera lift
before and after uint8, both evaluator-resized fields, frozen-SegNet logits,
and argmax fields.  Payloads are committed in immutable chunks of at most 16
pairs, with a checkpoint after every chunk.

The ``migrate-capture`` command verifies and copies a pre-existing complete
capture into the charter-mandated local receipt tier without writing either
full SSD.  The copy is atomic per payload and checkpointed per chunk.  The
``analyse`` command then joins those retained advisory fields to MS9's
exact contest-CUDA DALI-GT, transmitted-label, and terminal-argmax fields.  It
charges final manufactured errors and beneficial repairs to the earliest
observable stage and retains every resulting per-pixel mask.

Axis: stage fields are ``[macOS-CPU advisory]``.  Final support and all GT
charges are ``[contest-CUDA T4 component-only exact field replay]`` from MS9.
This tool never claims a score or edits ``upstream/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

N_PAIRS = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164
N_CLASSES = 5
SHAPE = (N_PAIRS, SEG_H, SEG_W)
N_PIXELS = int(np.prod(SHAPE))
CHUNK_PAIRS = 16
MIN_FREE_BYTES = 28 << 30
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
LOCAL_RECEIPT_ROOT = Path(
    "/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/"
    "ddm_mst1_manufactured_stage_split"
)
DEFAULT_STORE = LOCAL_RECEIPT_ROOT / "capture_r2_local"
DEFAULT_LEGACY_CAPTURE = (
    VERTIGO_ROOT / "ddm_mst1_manufactured_stage_split/capture_r1"
)
DEFAULT_ARCHIVE = (
    VERTIGO_ROOT
    / "ddm_mst1_manufactured_stage_split/instrumented_runtime_r1/archive.zip"
)
DEFAULT_UPSTREAM = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")
DEFAULT_GT = VERTIGO_ROOT / "ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
DEFAULT_LABELS = Path(
    "/Volumes/APDataStore/pact/ddm_rc2/composed_decode_r2/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
DEFAULT_ARGMAX = Path(
    "/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/"
    "payloads_r8/fx5_e1_argmax_n600.npy"
)

EXPECTED = {
    "archive_sha256": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    "token_sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",  # gitleaks:allow -- public content digest
    "gt_sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    "argmax_sha256": "e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34",
    "segnet_sha256": "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
    "modules_sha256": "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa",
    "evaluate_sha256": "7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b",
    "final_errors": 23_757,
    "manufactured": 21_493,
    "survived": 2_264,
    "representation_errors": 9_182,
    "representation_corrected": 6_918,
}

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
STAGE_NAMES = (
    "native_render_head",
    "preuint8_roundtrip_head",
    "uint8_roundtrip_head",
    "cpu_to_cuda_terminal_unseparated_head",
)
CAPTURE_ARRAYS: tuple[tuple[str, str], ...] = (
    ("native_rgb.float32.npy", "float32"),
    ("camera_preuint8_rgb.float32.npy", "float32"),
    ("camera_uint8_rgb.uint8.npy", "uint8"),
    ("evaluator_resized_preuint8_rgb.float32.npy", "float32"),
    ("evaluator_resized_uint8_rgb.float32.npy", "float32"),
    ("logits_native.float32.npy", "float32"),
    ("logits_preuint8.float32.npy", "float32"),
    ("logits_uint8.float32.npy", "float32"),
    ("argmax_native.uint8.npy", "uint8"),
    ("argmax_preuint8.uint8.npy", "uint8"),
    ("argmax_uint8.uint8.npy", "uint8"),
)


class Mst1Error(RuntimeError):
    """Fail-closed error for the MST1 instrument."""


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise Mst1Error(f"required file is absent: {path}")
    fact = file_fact(path)
    if fact["sha256"] != expected_sha256:
        raise Mst1Error(f"SHA-256 drift for {path}: {fact['sha256']} != {expected_sha256}")
    if expected_bytes is not None and fact["bytes"] != expected_bytes:
        raise Mst1Error(f"byte-count drift for {path}: {fact['bytes']} != {expected_bytes}")
    return fact


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    source_fact = verify_file(source, expected_sha256, expected_bytes)
    if destination.exists():
        return {
            **verify_file(destination, expected_sha256, expected_bytes),
            "copied_from": source_fact,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    copied_bytes = temporary.stat().st_size if temporary.exists() else 0
    if copied_bytes > expected_bytes:
        raise Mst1Error(
            f"interrupted retained copy exceeds expected size: {temporary} "
            f"{copied_bytes} > {expected_bytes}"
        )
    with source.open("rb") as source_stream, temporary.open("ab") as destination_stream:
        source_stream.seek(copied_bytes)
        shutil.copyfileobj(source_stream, destination_stream, length=8 << 20)
        destination_stream.flush()
        os.fsync(destination_stream.fileno())
    copied_fact = verify_file(temporary, expected_sha256, expected_bytes)
    os.replace(temporary, destination)
    return {
        **copied_fact,
        "path": str(destination),
        "copied_from": source_fact,
    }


def _tree_facts(root: Path) -> list[dict[str, Any]]:
    return [file_fact(path) for path in sorted(root.rglob("*")) if path.is_file()]


def _preserve_incomplete_directory(path: Path, interrupted_root: Path, reason: str) -> None:
    if not path.exists():
        return
    interrupted_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = interrupted_root / f"{path.name}_{stamp}_{os.getpid()}"
    os.replace(path, target)
    atomic_json(
        target / "INTERRUPTED_MANIFEST.json",
        {
            "schema": "ddm_mst1_interrupted_chunk.v1",
            "reason": reason,
            "preserved_at_utc": stamp,
            "files": _tree_facts(target),
        },
    )


def _ensure_local_store(store: Path) -> Path:
    store = store.resolve()
    if not store.is_relative_to(LOCAL_RECEIPT_ROOT.resolve()):
        raise Mst1Error(f"MST1 payloads must remain under the local receipt root: {store}")
    LOCAL_RECEIPT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(LOCAL_RECEIPT_ROOT.parent).free
    if not store.exists() and free < MIN_FREE_BYTES:
        raise Mst1Error(
            f"local receipt tier has {free} free bytes; MST1 requires "
            f"{MIN_FREE_BYTES} before first materialization"
        )
    return store


def _token_sha256(tokens: Any) -> str:
    array = tokens.detach().cpu().contiguous().numpy().astype(np.uint8, copy=False)
    return hashlib.sha256(memoryview(array)).hexdigest()


class StageCapture:
    """Chunked, resumable retained-field capture called by the copied renderer."""

    def __init__(self, *, tokens: Any, renderer_source: Path) -> None:
        store_text = os.environ.get("INFLATE_DDM_MST1_STORE", "").strip()
        if not store_text:
            raise Mst1Error("INFLATE_DDM_MST1_STORE is required in the instrumented runtime")
        self.store = _ensure_local_store(Path(store_text))
        archive_sha = os.environ.get("INFLATE_DDM_MST1_ARCHIVE_SHA256", "").strip()
        if archive_sha != EXPECTED["archive_sha256"]:
            raise Mst1Error("instrumented runtime archive binding is absent or drifted")
        upstream_text = os.environ.get("INFLATE_DDM_MST1_UPSTREAM_DIR", "").strip()
        self.upstream = Path(upstream_text).resolve() if upstream_text else DEFAULT_UPSTREAM.resolve()
        model = self.upstream / "models/segnet.safetensors"
        modules = self.upstream / "modules.py"
        evaluate = self.upstream / "evaluate.py"
        source_binding = {
            "archive_sha256": archive_sha,
            "tokens_sha256": _token_sha256(tokens),
            "segnet": verify_file(model, EXPECTED["segnet_sha256"]),
            "modules": verify_file(modules, EXPECTED["modules_sha256"]),
            "evaluate": verify_file(evaluate, EXPECTED["evaluate_sha256"]),
            "renderer_source": file_fact(renderer_source.resolve()),
            "capture_source": file_fact(Path(__file__).resolve()),
            "pairs": N_PAIRS,
            "chunk_pairs": CHUNK_PAIRS,
            "device": "cpu",
            "torch_threads": 4,
            "scorer_batch": CHUNK_PAIRS,
            "native_shape": [3, SEG_H, SEG_W],
            "camera_shape": [3, CAM_H, CAM_W],
        }
        if source_binding["tokens_sha256"] != EXPECTED["token_sha256"]:
            raise Mst1Error("decoded semantic-token field differs from MS9/TO2")
        self.binding = source_binding
        self.checkpoint_path = self.store / "CAPTURE_CHECKPOINT.json"
        self.chunks_root = self.store / "retained/chunks"
        self.interrupted_root = self.store / "retained/interrupted"
        self.checkpoint: dict[str, Any] = {
            "schema": "ddm_mst1_stage_capture_checkpoint.v1",
            "source_binding": self.binding,
            "completed_chunks": {},
        }
        if self.checkpoint_path.exists():
            loaded = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if loaded.get("source_binding") != self.binding:
                raise Mst1Error("existing capture checkpoint is bound to different sources")
            self.checkpoint = loaded
            self._verify_completed_chunks()
        else:
            if self.store.exists() and any(self.store.iterdir()):
                raise Mst1Error("fresh MST1 store is non-empty without a capture checkpoint")
            self.store.mkdir(parents=True, exist_ok=True)
            atomic_json(
                self.store / "STORAGE_PREFLIGHT.json",
                {
                    "schema": "ddm_mst1_storage_preflight.v1",
                    "tier": "local_disk_explicit_opt_in",
                    "store": str(self.store),
                    "free_bytes_before": shutil.disk_usage(LOCAL_RECEIPT_ROOT.parent).free,
                    "minimum_free_bytes": MIN_FREE_BYTES,
                    "estimated_payload_bytes": 25_000_000_000,
                    "routing_reason": "both SSD tiers were measured at 100% on 2026-08-22",
                    "policy": "certify-or-block; no captured payload is deleted; no /Volumes write",
                    "source_binding": self.binding,
                },
            )
            atomic_json(self.checkpoint_path, self.checkpoint)
        self._buffer: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        self._last_pair = -1
        self._load_scorer()

    def _load_scorer(self) -> None:
        import torch
        from safetensors.torch import load_file

        upstream_text = str(self.upstream)
        if upstream_text not in sys.path:
            sys.path.insert(0, upstream_text)
        import modules as upstream_modules  # type: ignore[import-not-found]

        loaded_modules_path = Path(upstream_modules.__file__).resolve()
        expected_modules_path = (self.upstream / "modules.py").resolve()
        if loaded_modules_path != expected_modules_path:
            raise Mst1Error(
                f"upstream modules import resolved to {loaded_modules_path}, expected {expected_modules_path}"
            )

        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        self.torch = torch
        self.segnet = upstream_modules.SegNet().eval()
        self.segnet.load_state_dict(
            load_file(str(self.upstream / "models/segnet.safetensors"), device="cpu")
        )

    def _verify_completed_chunks(self) -> None:
        for key, row in self.checkpoint["completed_chunks"].items():
            manifest_path = Path(row["manifest"]["path"])
            verify_file(manifest_path, row["manifest"]["sha256"], row["manifest"]["bytes"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_binding") != self.binding:
                raise Mst1Error(f"completed chunk {key} has different source binding")
            for fact in manifest["payloads"]:
                verify_file(Path(fact["path"]), fact["sha256"], fact["bytes"])

    def _chunk_key(self, pair: int) -> str:
        start = (pair // CHUNK_PAIRS) * CHUNK_PAIRS
        stop = min(start + CHUNK_PAIRS, N_PAIRS)
        return f"{start:04d}_{stop - 1:04d}"

    def record(self, pair_indices: Iterable[int], native: Any, camera_float: Any, camera_u8: Any) -> None:
        indices = [int(value) for value in pair_indices]
        if native.shape[0] != len(indices) or camera_float.shape[0] != len(indices) or camera_u8.shape[0] != len(indices):
            raise Mst1Error("capture batch indices and tensors disagree")
        for offset, pair in enumerate(indices):
            if pair != self._last_pair + 1:
                raise Mst1Error(f"renderer pair order drifted: {pair} after {self._last_pair}")
            self._last_pair = pair
            key = self._chunk_key(pair)
            if key in self.checkpoint["completed_chunks"]:
                continue
            self._buffer.append(
                (
                    pair,
                    native[offset : offset + 1].detach().cpu().numpy().astype(np.float32, copy=True),
                    camera_float[offset : offset + 1].detach().cpu().numpy().astype(np.float32, copy=True),
                    camera_u8[offset : offset + 1].detach().cpu().numpy().astype(np.uint8, copy=True),
                )
            )
            expected_stop = min(((pair // CHUNK_PAIRS) + 1) * CHUNK_PAIRS, N_PAIRS)
            if pair + 1 == expected_stop:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        indices = [row[0] for row in self._buffer]
        start, stop = indices[0], indices[-1] + 1
        if indices != list(range(start, stop)) or stop - start > CHUNK_PAIRS:
            raise Mst1Error("capture buffer is not one contiguous bounded chunk")
        key = f"{start:04d}_{stop - 1:04d}"
        chunk_dir = self.chunks_root / key
        if chunk_dir.exists():
            _preserve_incomplete_directory(chunk_dir, self.interrupted_root, "chunk existed without checkpoint admission")
        chunk_dir.mkdir(parents=True, exist_ok=False)
        native = np.concatenate([row[1] for row in self._buffer], axis=0)
        camera_float = np.concatenate([row[2] for row in self._buffer], axis=0)
        camera_u8 = np.concatenate([row[3] for row in self._buffer], axis=0)
        torch = self.torch
        with torch.inference_mode():
            native_t = torch.from_numpy(native)
            camera_float_t = torch.from_numpy(camera_float)
            camera_u8_t = torch.from_numpy(camera_u8).float()
            native_input = self.segnet.preprocess_input(native_t[:, None])
            resized_pre = self.segnet.preprocess_input(camera_float_t[:, None])
            resized_u8 = self.segnet.preprocess_input(camera_u8_t[:, None])
            logits_native = self.segnet(native_input)
            logits_pre = self.segnet(resized_pre)
            logits_u8 = self.segnet(resized_u8)
            arrays: dict[str, np.ndarray] = {
                "native_rgb.float32.npy": native,
                "camera_preuint8_rgb.float32.npy": camera_float,
                "camera_uint8_rgb.uint8.npy": camera_u8,
                "evaluator_resized_preuint8_rgb.float32.npy": resized_pre.numpy().astype(np.float32, copy=False),
                "evaluator_resized_uint8_rgb.float32.npy": resized_u8.numpy().astype(np.float32, copy=False),
                "logits_native.float32.npy": logits_native.numpy().astype(np.float32, copy=False),
                "logits_preuint8.float32.npy": logits_pre.numpy().astype(np.float32, copy=False),
                "logits_uint8.float32.npy": logits_u8.numpy().astype(np.float32, copy=False),
                "argmax_native.uint8.npy": logits_native.argmax(dim=1).numpy().astype(np.uint8, copy=False),
                "argmax_preuint8.uint8.npy": logits_pre.argmax(dim=1).numpy().astype(np.uint8, copy=False),
                "argmax_uint8.uint8.npy": logits_u8.argmax(dim=1).numpy().astype(np.uint8, copy=False),
            }
        payloads: list[dict[str, Any]] = []
        for name, dtype in CAPTURE_ARRAYS:
            array = arrays[name]
            if array.dtype != np.dtype(dtype):
                raise Mst1Error(f"capture dtype drift for {name}: {array.dtype} != {dtype}")
            fact = atomic_npy(chunk_dir / name, array)
            payloads.append({**fact, "shape": list(array.shape), "dtype": str(array.dtype)})
        manifest = {
            "schema": "ddm_mst1_stage_capture_chunk.v1",
            "source_binding": self.binding,
            "pair_start": start,
            "pair_stop_exclusive": stop,
            "payloads": payloads,
            "axis": "[macOS-CPU advisory] frozen CPU-torch SegNet, batch16, 4 threads",
            "wrong_definition": "argmax of the exact frozen SegNet head applied to this retained RGB intermediate",
        }
        manifest_path = chunk_dir / "MANIFEST.json"
        atomic_json(manifest_path, manifest)
        self.checkpoint["completed_chunks"][key] = {
            "pair_start": start,
            "pair_stop_exclusive": stop,
            "manifest": file_fact(manifest_path),
        }
        atomic_json(self.checkpoint_path, self.checkpoint)
        self._buffer.clear()

    def finish(self) -> None:
        self._flush_buffer()
        if self._last_pair != N_PAIRS - 1:
            raise Mst1Error(f"capture ended at pair {self._last_pair}, expected {N_PAIRS - 1}")
        expected = {
            f"{start:04d}_{min(start + CHUNK_PAIRS, N_PAIRS) - 1:04d}"
            for start in range(0, N_PAIRS, CHUNK_PAIRS)
        }
        if set(self.checkpoint["completed_chunks"]) != expected:
            raise Mst1Error("capture checkpoint does not cover all 600 pairs")
        manifests = [row["manifest"] for _, row in sorted(self.checkpoint["completed_chunks"].items())]
        total_payload_bytes = 0
        payload_count = 0
        for fact in manifests:
            manifest = json.loads(Path(fact["path"]).read_text(encoding="utf-8"))
            total_payload_bytes += sum(int(row["bytes"]) for row in manifest["payloads"])
            payload_count += len(manifest["payloads"])
        atomic_json(
            self.store / "CAPTURE_COMPLETE.json",
            {
                "schema": "ddm_mst1_stage_capture_complete.v1",
                "complete": True,
                "source_binding": self.binding,
                "manifests": manifests,
                "payload_count": payload_count,
                "payload_bytes": total_payload_bytes,
                "n_pairs": N_PAIRS,
                "chunk_pairs": CHUNK_PAIRS,
                "axis": "[macOS-CPU advisory] frozen CPU-torch SegNet, batch16, 4 threads",
                "host": platform.node(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch": self.torch.__version__,
            },
        )


def _load_capture_manifests(store: Path) -> list[dict[str, Any]]:
    complete_path = store / "CAPTURE_COMPLETE.json"
    if not complete_path.is_file():
        raise Mst1Error(f"capture is not complete: {complete_path}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("n_pairs") != N_PAIRS:
        raise Mst1Error("capture completion receipt is not a full n600 receipt")
    binding = complete.get("source_binding", {})
    if binding.get("archive_sha256") != EXPECTED["archive_sha256"]:
        raise Mst1Error("capture completion receipt is not bound to the exact DX2 archive")
    if binding.get("tokens_sha256") != EXPECTED["token_sha256"]:
        raise Mst1Error("capture completion receipt is not bound to the exact TO2 token field")
    manifests: list[dict[str, Any]] = []
    next_pair = 0
    for fact in complete["manifests"]:
        path = Path(fact["path"])
        verify_file(path, fact["sha256"], fact["bytes"])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["pair_start"] != next_pair or manifest["pair_stop_exclusive"] > next_pair + CHUNK_PAIRS:
            raise Mst1Error("capture manifests are not contiguous or exceed the 120-pair law")
        pair_count = manifest["pair_stop_exclusive"] - manifest["pair_start"]
        expected_payload_names = {name for name, _ in CAPTURE_ARRAYS}
        actual_payload_names = {Path(payload["path"]).name for payload in manifest["payloads"]}
        if actual_payload_names != expected_payload_names:
            raise Mst1Error("capture chunk payload set drifted")
        for payload in manifest["payloads"]:
            payload_path = Path(payload["path"])
            verify_file(payload_path, payload["sha256"], payload["bytes"])
            array = np.load(payload_path, mmap_mode="r")
            name = payload_path.name
            channels = 3 if "rgb." in name else N_CLASSES if name.startswith("logits_") else None
            height, width = (CAM_H, CAM_W) if name.startswith("camera_") else (SEG_H, SEG_W)
            expected_shape = (
                (pair_count, channels, height, width)
                if channels is not None
                else (pair_count, height, width)
            )
            expected_dtype = dict(CAPTURE_ARRAYS)[name]
            if array.shape != expected_shape or array.dtype != np.dtype(expected_dtype):
                raise Mst1Error(
                    f"capture payload shape/dtype drift for {payload_path}: "
                    f"{array.shape}/{array.dtype} != {expected_shape}/{expected_dtype}"
                )
        next_pair = manifest["pair_stop_exclusive"]
        manifests.append(manifest)
    if next_pair != N_PAIRS:
        raise Mst1Error("capture manifests do not cover n600")
    return manifests


def _source_file_facts(complete: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    binding = complete.get("source_binding", {})
    rows: list[tuple[str, dict[str, Any]]] = []
    for key in ("segnet", "modules", "evaluate", "renderer_source", "capture_source"):
        fact = binding.get(key)
        if not isinstance(fact, dict):
            raise Mst1Error(f"capture source binding is missing {key}")
        rows.append((key, fact))
    return rows


def migrate_capture(args: argparse.Namespace) -> int:
    """Copy an already-complete exact capture into local, self-contained custody."""
    source = args.source.resolve()
    store = _ensure_local_store(args.store)
    if not source.is_relative_to(VERTIGO_ROOT.resolve()):
        raise Mst1Error(f"legacy capture source must be read from Vertigo: {source}")
    complete_path = source / "CAPTURE_COMPLETE.json"
    complete_fact = file_fact(complete_path)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("n_pairs") != N_PAIRS:
        raise Mst1Error("legacy capture is not a complete n600 capture")
    binding = complete.get("source_binding", {})
    if binding.get("archive_sha256") != EXPECTED["archive_sha256"]:
        raise Mst1Error("legacy capture archive pin drifted")
    if binding.get("tokens_sha256") != EXPECTED["token_sha256"]:
        raise Mst1Error("legacy capture token pin drifted")

    checkpoint_path = store / "MIGRATION_CHECKPOINT.json"
    checkpoint_binding = {
        "schema": "ddm_mst1_local_capture_migration_binding.v1",
        "source_capture_complete": complete_fact,
        "source_store": str(source),
        "destination_store": str(store),
        "archive_sha256": EXPECTED["archive_sha256"],
        "tokens_sha256": EXPECTED["token_sha256"],
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("binding") != checkpoint_binding:
            raise Mst1Error("local migration checkpoint is bound to different sources")
    else:
        if store.exists() and any(store.iterdir()):
            raise Mst1Error("local migration store is non-empty without its checkpoint")
        store.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema": "ddm_mst1_local_capture_migration_checkpoint.v1",
            "binding": checkpoint_binding,
            "completed_chunks": {},
            "completed_provenance_sources": {},
        }
        atomic_json(
            store / "LOCAL_STORAGE_PREFLIGHT.json",
            {
                "schema": "ddm_mst1_local_storage_preflight.v1",
                "tier": "local_disk_explicit_opt_in",
                "store": str(store),
                "free_bytes_before": shutil.disk_usage(LOCAL_RECEIPT_ROOT.parent).free,
                "minimum_free_bytes": MIN_FREE_BYTES,
                "expected_capture_payload_bytes": complete.get("payload_bytes"),
                "routing_reason": "both SSD tiers were measured at 100% on 2026-08-22",
                "source_store_is_read_only": True,
                "policy": "verified copy only; no source deletion and no /Volumes write",
                "argv": sys.argv,
            },
        )
        atomic_json(checkpoint_path, checkpoint)

    local_manifests: list[dict[str, Any]] = []
    expected_pair = 0
    for source_manifest_fact in complete["manifests"]:
        source_manifest_path = Path(source_manifest_fact["path"])
        verify_file(
            source_manifest_path,
            source_manifest_fact["sha256"],
            source_manifest_fact["bytes"],
        )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        start = int(source_manifest["pair_start"])
        stop = int(source_manifest["pair_stop_exclusive"])
        if start != expected_pair or stop > start + CHUNK_PAIRS:
            raise Mst1Error("legacy capture manifests are not contiguous or bounded")
        key = f"{start:04d}_{stop - 1:04d}"
        local_manifest_path = store / "retained/chunks" / key / "MANIFEST.json"
        local_payloads: list[dict[str, Any]] = []
        for payload in source_manifest["payloads"]:
            source_payload = Path(payload["path"])
            if not source_payload.is_relative_to(source):
                raise Mst1Error(f"capture payload escapes source store: {source_payload}")
            local_payload = store / source_payload.relative_to(source)
            copied = atomic_copy_verified(
                source_payload,
                local_payload,
                expected_sha256=payload["sha256"],
                expected_bytes=int(payload["bytes"]),
            )
            local_payloads.append(
                {
                    **payload,
                    "path": str(local_payload),
                    "copied_from": copied["copied_from"],
                }
            )
        local_manifest = {
            **source_manifest,
            "payloads": local_payloads,
            "custody_migration": {
                "source_manifest": source_manifest_fact,
                "destination_tier": "local_disk_explicit_opt_in",
                "source_store_read_only": True,
            },
        }
        if local_manifest_path.exists():
            existing = json.loads(local_manifest_path.read_text(encoding="utf-8"))
            if existing != local_manifest:
                raise Mst1Error(f"local migrated manifest drifted: {local_manifest_path}")
        else:
            atomic_json(local_manifest_path, local_manifest)
        local_manifest_fact = file_fact(local_manifest_path)
        local_manifests.append(local_manifest_fact)
        checkpoint["completed_chunks"][key] = {
            "pair_start": start,
            "pair_stop_exclusive": stop,
            "manifest": local_manifest_fact,
        }
        atomic_json(checkpoint_path, checkpoint)
        expected_pair = stop
        print(f"migrated and verified chunk {key}", flush=True)
    if expected_pair != N_PAIRS:
        raise Mst1Error("legacy capture migration did not cover n600")

    provenance_sources: dict[str, Any] = {}
    for key, fact in _source_file_facts(complete):
        source_path = Path(fact["path"])
        suffix = source_path.suffix or ".bin"
        destination = store / "retained/provenance_sources" / f"{key}{suffix}"
        provenance_sources[key] = atomic_copy_verified(
            source_path,
            destination,
            expected_sha256=fact["sha256"],
            expected_bytes=int(fact["bytes"]),
        )
        checkpoint["completed_provenance_sources"][key] = provenance_sources[key]
        atomic_json(checkpoint_path, checkpoint)
    provenance_sources["archive"] = atomic_copy_verified(
        args.archive,
        store / "retained/provenance_sources/archive.zip",
        expected_sha256=EXPECTED["archive_sha256"],
        expected_bytes=180_368,
    )
    checkpoint["completed_provenance_sources"]["archive"] = provenance_sources["archive"]
    atomic_json(checkpoint_path, checkpoint)

    original_receipts: dict[str, Any] = {}
    for name in ("CAPTURE_COMPLETE.json", "CAPTURE_CHECKPOINT.json", "STORAGE_PREFLIGHT.json"):
        original_path = source / name
        original_fact = file_fact(original_path)
        original_receipts[name] = atomic_copy_verified(
            original_path,
            store / "retained/original_receipts" / name,
            expected_sha256=original_fact["sha256"],
            expected_bytes=original_fact["bytes"],
        )

    local_complete = {
        **complete,
        "schema": "ddm_mst1_stage_capture_complete.local_migration.v1",
        "manifests": local_manifests,
        "retention_tier": "local_disk_explicit_opt_in",
        "retention_store": str(store),
        "original_capture_complete": complete_fact,
        "provenance_sources": provenance_sources,
        "original_receipts": original_receipts,
        "migration": {
            "argv": sys.argv,
            "source_store_read_only": True,
            "source_deleted": False,
            "destination_writes_only": True,
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    atomic_json(store / "CAPTURE_COMPLETE.json", local_complete)
    completion = {
        "schema": "ddm_mst1_local_capture_migration_complete.v1",
        "complete": True,
        "capture_complete": file_fact(store / "CAPTURE_COMPLETE.json"),
        "payload_count": complete["payload_count"],
        "payload_bytes": complete["payload_bytes"],
        "n_pairs": N_PAIRS,
        "local_manifests": local_manifests,
        "provenance_sources": provenance_sources,
        "no_volume_writes": True,
    }
    atomic_json(store / "MIGRATION_COMPLETE.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


def _payload_path(manifest: Mapping[str, Any], name: str) -> Path:
    rows = [Path(row["path"]) for row in manifest["payloads"] if Path(row["path"]).name == name]
    if len(rows) != 1:
        raise Mst1Error(f"chunk manifest does not contain exactly one {name}")
    return rows[0]


def _assemble_argmax(store: Path, manifests: list[dict[str, Any]], source_name: str, output_name: str) -> Path:
    out = store / "retained/assembled" / output_name
    if out.exists():
        array = np.load(out, mmap_mode="r")
        if array.shape != SHAPE or array.dtype != np.uint8:
            raise Mst1Error(f"assembled field drifted: {out}")
        for manifest in manifests:
            start, stop = manifest["pair_start"], manifest["pair_stop_exclusive"]
            source = np.load(_payload_path(manifest, source_name), mmap_mode="r")
            if not np.array_equal(array[start:stop], source):
                raise Mst1Error(f"assembled field content drifted from its retained chunks: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{os.getpid()}.partial")
    if temporary.exists():
        interrupted = store / "retained/interrupted/assembled"
        interrupted.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, interrupted / f"{temporary.name}.{int(time.time())}")
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=SHAPE)
    for manifest in manifests:
        start, stop = manifest["pair_start"], manifest["pair_stop_exclusive"]
        target[start:stop] = np.load(_payload_path(manifest, source_name), mmap_mode="r")
        target.flush()
    del target
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, out)
    return out


def _open_field(path: Path, *, raw: bool = False) -> np.ndarray:
    if raw:
        return np.memmap(path, dtype=np.uint8, mode="r", shape=SHAPE)
    value = np.load(path, mmap_mode="r")
    if value.shape != SHAPE or value.dtype != np.uint8:
        raise Mst1Error(f"field shape/dtype drift: {path} {value.shape}/{value.dtype}")
    return value


def _mask_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(
        (
            "representation_error_support",
            "final_error_support",
            "final_manufactured_support",
            "final_survived_representation_error",
            "final_repaired_representation_error",
        )
    )
    names.extend(f"state_wrong_{stage}" for stage in STAGE_NAMES)
    names.extend(f"final_manufactured_wrong_at_{stage}" for stage in STAGE_NAMES)
    names.extend(f"representation_error_wrong_at_{stage}" for stage in STAGE_NAMES)
    names.extend(f"earliest_manufactured_{stage}" for stage in STAGE_NAMES)
    names.extend(f"earliest_repaired_{stage}" for stage in STAGE_NAMES)
    names.extend(f"gross_manufactured_{stage}" for stage in STAGE_NAMES)
    names.extend(f"gross_repaired_{stage}" for stage in STAGE_NAMES)
    return tuple(names)


def _verify_completed_result(store: Path, result: Mapping[str, Any]) -> dict[str, int]:
    expected_totals = {
        "final_error": EXPECTED["final_errors"],
        "manufactured": EXPECTED["manufactured"],
        "survived": EXPECTED["survived"],
        "representation_errors": EXPECTED["representation_errors"],
        "representation_corrected": EXPECTED["representation_corrected"],
    }
    if result.get("totals") != expected_totals:
        raise Mst1Error("completed MST1 result does not reproduce the MS9 gate")
    stage_rows = result.get("stage_rows", [])
    if [row.get("stage") for row in stage_rows] != list(STAGE_NAMES):
        raise Mst1Error("completed MST1 stage ordering drifted")
    if sum(int(row["earliest_final_manufactured"]) for row in stage_rows) != EXPECTED["manufactured"]:
        raise Mst1Error("completed MST1 manufactured stage charges do not close")
    if sum(int(row["earliest_final_repaired"]) for row in stage_rows) != EXPECTED["representation_corrected"]:
        raise Mst1Error("completed MST1 repaired stage charges do not close")
    class_rows = result.get("class_rows", [])
    if [row.get("class_name") for row in class_rows] != list(CLASS_NAMES):
        raise Mst1Error("completed MST1 class ordering drifted")
    if sum(int(row["gt_pixels"]) for row in class_rows) != N_PIXELS:
        raise Mst1Error("completed MST1 class areas do not partition the n600 field")
    for stage_index, stage_row in enumerate(stage_rows):
        class_manufactured = sum(
            int(row["stages"][stage_index]["manufactured"]) for row in class_rows
        )
        class_repaired = sum(int(row["stages"][stage_index]["repaired"]) for row in class_rows)
        if class_manufactured != int(stage_row["earliest_final_manufactured"]):
            raise Mst1Error(f"completed per-class manufactured charges do not close for {stage_row['stage']}")
        if class_repaired != int(stage_row["earliest_final_repaired"]):
            raise Mst1Error(f"completed per-class repaired charges do not close for {stage_row['stage']}")
    state_errors = result.get("state_errors", {})
    previous_errors = int(state_errors.get("labels", -1))
    for stage_row in stage_rows:
        current_errors = int(state_errors.get(stage_row["stage"], -1))
        if int(stage_row["net_gross_error_change"]) != current_errors - previous_errors:
            raise Mst1Error(f"completed gross transition accounting does not close for {stage_row['stage']}")
        previous_errors = current_errors
    manifests = _load_capture_manifests(store)
    if len(manifests) != (N_PAIRS + CHUNK_PAIRS - 1) // CHUNK_PAIRS:
        raise Mst1Error("completed MST1 result does not retain every bounded capture chunk")
    payload_count = sum(len(manifest["payloads"]) for manifest in manifests)
    payload_bytes = sum(
        int(payload["bytes"])
        for manifest in manifests
        for payload in manifest["payloads"]
    )
    capture_complete = json.loads((store / "CAPTURE_COMPLETE.json").read_text(encoding="utf-8"))
    if payload_count != int(capture_complete.get("payload_count", -1)):
        raise Mst1Error("completed MST1 retained payload count drifted")
    if payload_bytes != int(capture_complete.get("payload_bytes", -1)):
        raise Mst1Error("completed MST1 retained payload byte count drifted")
    for fact in result.get("source_fields", {}).values():
        verify_file(Path(fact["path"]), fact["sha256"], fact["bytes"])
    migration = result.get("migration_complete")
    if not isinstance(migration, dict):
        raise Mst1Error("completed MST1 result lacks the local migration receipt")
    verify_file(Path(migration["path"]), migration["sha256"], migration["bytes"])
    expected_mask_names = set(_mask_names())
    mask_names = {fact["name"] for fact in result.get("mask_manifest", [])}
    if mask_names != expected_mask_names:
        raise Mst1Error("completed MST1 result mask set drifted")
    mask_counts = {fact["name"]: int(fact["count_true"]) for fact in result["mask_manifest"]}
    if mask_counts["final_error_support"] != EXPECTED["final_errors"]:
        raise Mst1Error("completed final-error mask count drifted")
    if mask_counts["final_manufactured_support"] != EXPECTED["manufactured"]:
        raise Mst1Error("completed manufactured mask count drifted")
    if mask_counts["final_survived_representation_error"] != EXPECTED["survived"]:
        raise Mst1Error("completed survived-representation mask count drifted")
    if mask_counts["final_repaired_representation_error"] != EXPECTED["representation_corrected"]:
        raise Mst1Error("completed repaired-representation mask count drifted")
    for stage_row in stage_rows:
        stage = stage_row["stage"]
        if mask_counts[f"earliest_manufactured_{stage}"] != int(stage_row["earliest_final_manufactured"]):
            raise Mst1Error(f"completed earliest-manufactured mask count drifted for {stage}")
        if mask_counts[f"earliest_repaired_{stage}"] != int(stage_row["earliest_final_repaired"]):
            raise Mst1Error(f"completed earliest-repaired mask count drifted for {stage}")
        if mask_counts[f"gross_manufactured_{stage}"] != int(stage_row["gross_right_to_wrong"]):
            raise Mst1Error(f"completed gross-manufactured mask count drifted for {stage}")
        if mask_counts[f"gross_repaired_{stage}"] != int(stage_row["gross_wrong_to_right"]):
            raise Mst1Error(f"completed gross-repaired mask count drifted for {stage}")
    for fact in result["mask_manifest"]:
        verify_file(Path(fact["path"]), fact["sha256"], fact["bytes"])
    return {"payload_count": payload_count, "payload_bytes": payload_bytes}


def analyse(args: argparse.Namespace) -> int:
    store = _ensure_local_store(args.store)
    completed_result_path = store / "MST1_RESULT.json"
    if completed_result_path.exists():
        completed_result = json.loads(completed_result_path.read_text(encoding="utf-8"))
        verified_capture = _verify_completed_result(store, completed_result)
        verification = {
            "schema": "ddm_mst1_completed_verification.v1",
            "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "analysis_source": file_fact(Path(__file__).resolve()),
            "result": file_fact(completed_result_path),
            "capture_complete": file_fact(store / "CAPTURE_COMPLETE.json"),
            "migration_complete": file_fact(store / "MIGRATION_COMPLETE.json"),
            "attribution_mask_manifest": file_fact(store / "ATTRIBUTION_MASK_MANIFEST.json"),
            "verified_payload_count": verified_capture["payload_count"],
            "verified_payload_bytes": verified_capture["payload_bytes"],
            "verified_mask_count": len(completed_result["mask_manifest"]),
            "verified_totals": completed_result["totals"],
            "no_volume_writes": True,
        }
        atomic_json(store / "COMPLETED_VERIFICATION.json", verification)
        print(json.dumps(completed_result, indent=2, sort_keys=True))
        return 0
    manifests = _load_capture_manifests(store)
    native_path = _assemble_argmax(store, manifests, "argmax_native.uint8.npy", "argmax_native_n600.npy")
    pre_path = _assemble_argmax(store, manifests, "argmax_preuint8.uint8.npy", "argmax_preuint8_n600.npy")
    uint8_path = _assemble_argmax(store, manifests, "argmax_uint8.uint8.npy", "argmax_uint8_n600.npy")
    retained_inputs = store / "retained/inputs"
    retained_gt = retained_inputs / "gt_argmax_n600.npy"
    retained_labels = retained_inputs / "tokens_cpu_stage_complete.u8"
    retained_cuda = retained_inputs / "cuda_terminal_argmax_n600.npy"
    sources = {
        "gt": atomic_copy_verified(
            args.gt,
            retained_gt,
            expected_sha256=EXPECTED["gt_sha256"],
            expected_bytes=N_PIXELS + 128,
        ),
        "labels": atomic_copy_verified(
            args.labels,
            retained_labels,
            expected_sha256=EXPECTED["token_sha256"],
            expected_bytes=N_PIXELS,
        ),
        "cuda_terminal": atomic_copy_verified(
            args.argmax,
            retained_cuda,
            expected_sha256=EXPECTED["argmax_sha256"],
            expected_bytes=N_PIXELS + 128,
        ),
        "native": file_fact(native_path),
        "preuint8": file_fact(pre_path),
        "uint8": file_fact(uint8_path),
        "capture_complete": file_fact(store / "CAPTURE_COMPLETE.json"),
    }
    gt = _open_field(retained_gt)
    labels = _open_field(retained_labels, raw=True)
    native = _open_field(native_path)
    preuint8 = _open_field(pre_path)
    uint8 = _open_field(uint8_path)
    cuda = _open_field(retained_cuda)
    states = (native, preuint8, uint8, cuda)

    masks_dir = store / "retained/attribution_masks"
    if masks_dir.exists():
        _preserve_incomplete_directory(
            masks_dir,
            store / "retained/interrupted/analysis",
            "attribution directory existed without an admitted MST1 result",
        )
    masks_partial = store / "retained" / f".attribution_masks.{os.getpid()}.partial"
    if masks_partial.exists():
        raise Mst1Error(f"unexpected existing partial attribution directory: {masks_partial}")
    masks_partial.mkdir(parents=True, exist_ok=False)
    mask_names = _mask_names()
    temp_paths = {name: masks_partial / f"{name}.n600.packbits" for name in mask_names}
    final_paths = {name: masks_dir / f"{name}.n600.packbits" for name in mask_names}
    handles = {name: temp_paths[name].open("wb") for name in mask_names}
    counts = dict.fromkeys(mask_names, 0)
    class_counts = {
        "earliest_manufactured": {stage: [0] * N_CLASSES for stage in STAGE_NAMES},
        "earliest_repaired": {stage: [0] * N_CLASSES for stage in STAGE_NAMES},
    }
    class_area = np.zeros(N_CLASSES, dtype=np.int64)
    state_errors = {"labels": 0, **dict.fromkeys(STAGE_NAMES, 0)}
    state_disagreement_from_labels = dict.fromkeys(STAGE_NAMES, 0)
    terminal_advisory_vs_cuda = 0
    totals = {
        "final_error": 0,
        "manufactured": 0,
        "survived": 0,
        "representation_errors": 0,
        "representation_corrected": 0,
    }
    try:
        for start in range(0, N_PAIRS, CHUNK_PAIRS):
            stop = min(start + CHUNK_PAIRS, N_PAIRS)
            g = np.asarray(gt[start:stop])
            label_chunk = np.asarray(labels[start:stop])
            chunk_states = [np.asarray(state[start:stop]) for state in states]
            for class_id in range(N_CLASSES):
                class_area[class_id] += int(np.count_nonzero(g == class_id))
            final_error = chunk_states[-1] != g
            representation_error = label_chunk != g
            manufactured = final_error & ~representation_error
            survived = final_error & representation_error
            corrected = ~final_error & representation_error
            totals["final_error"] += int(np.count_nonzero(final_error))
            totals["manufactured"] += int(np.count_nonzero(manufactured))
            totals["survived"] += int(np.count_nonzero(survived))
            totals["representation_errors"] += int(np.count_nonzero(representation_error))
            totals["representation_corrected"] += int(np.count_nonzero(corrected))
            state_errors["labels"] += int(np.count_nonzero(representation_error))
            terminal_advisory_vs_cuda += int(np.count_nonzero(chunk_states[-2] != chunk_states[-1]))

            masks: dict[str, np.ndarray] = {}
            masks["representation_error_support"] = representation_error
            masks["final_error_support"] = final_error
            masks["final_manufactured_support"] = manufactured
            masks["final_survived_representation_error"] = survived
            masks["final_repaired_representation_error"] = corrected
            for stage, state in zip(STAGE_NAMES, chunk_states, strict=True):
                masks[f"state_wrong_{stage}"] = state != g
                masks[f"final_manufactured_wrong_at_{stage}"] = manufactured & (state != g)
                masks[f"representation_error_wrong_at_{stage}"] = representation_error & (state != g)
            unassigned_m = manufactured.copy()
            unassigned_r = corrected.copy()
            previous = label_chunk
            for stage, current in zip(STAGE_NAMES, chunk_states, strict=True):
                current_wrong = current != g
                state_errors[stage] += int(np.count_nonzero(current_wrong))
                state_disagreement_from_labels[stage] += int(np.count_nonzero(current != label_chunk))
                first_m = unassigned_m & current_wrong
                first_r = unassigned_r & ~current_wrong
                masks[f"earliest_manufactured_{stage}"] = first_m
                masks[f"earliest_repaired_{stage}"] = first_r
                previous_wrong = previous != g
                masks[f"gross_manufactured_{stage}"] = ~previous_wrong & current_wrong
                masks[f"gross_repaired_{stage}"] = previous_wrong & ~current_wrong
                unassigned_m &= ~first_m
                unassigned_r &= ~first_r
                for class_id in range(N_CLASSES):
                    class_mask = g == class_id
                    class_counts["earliest_manufactured"][stage][class_id] += int(
                        np.count_nonzero(first_m & class_mask)
                    )
                    class_counts["earliest_repaired"][stage][class_id] += int(
                        np.count_nonzero(first_r & class_mask)
                    )
                previous = current
            if np.any(unassigned_m) or np.any(unassigned_r):
                raise Mst1Error("earliest-stage attribution did not exhaust final support")
            for name in mask_names:
                mask = masks[name]
                counts[name] += int(np.count_nonzero(mask))
                handles[name].write(np.packbits(mask.reshape(-1), bitorder="little").tobytes())
        for stream in handles.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        handles.clear()
    finally:
        for stream in handles.values():
            stream.close()

    expected_totals = {
        "final_error": EXPECTED["final_errors"],
        "manufactured": EXPECTED["manufactured"],
        "survived": EXPECTED["survived"],
        "representation_errors": EXPECTED["representation_errors"],
        "representation_corrected": EXPECTED["representation_corrected"],
    }
    if totals != expected_totals:
        raise Mst1Error(f"MS9 four-count drift gate failed: {totals} != {expected_totals}")
    if sum(counts[f"earliest_manufactured_{stage}"] for stage in STAGE_NAMES) != EXPECTED["manufactured"]:
        raise Mst1Error("manufactured earliest-stage counts are not additive")
    if sum(counts[f"earliest_repaired_{stage}"] for stage in STAGE_NAMES) != EXPECTED["representation_corrected"]:
        raise Mst1Error("repair earliest-stage counts are not additive")
    os.replace(masks_partial, masks_dir)
    mask_manifest = []
    expected_mask_bytes = N_PIXELS // 8
    for name in mask_names:
        fact = file_fact(final_paths[name])
        if fact["bytes"] != expected_mask_bytes:
            raise Mst1Error(f"packed attribution mask has wrong size: {name}")
        mask_manifest.append(
            {
                **fact,
                "name": name,
                "count_true": counts[name],
                "format": "numpy.packbits",
                "bitorder": "little",
                "logical_shape": list(SHAPE),
                "logical_dtype": "bool",
            }
        )
    bytes_per_flip = (100.0 / N_PIXELS) / (25.0 / 37_545_489)
    stage_rows = []
    for stage in STAGE_NAMES:
        m = counts[f"earliest_manufactured_{stage}"]
        r = counts[f"earliest_repaired_{stage}"]
        stage_rows.append(
            {
                "stage": stage,
                "earliest_final_manufactured": m,
                "earliest_final_repaired": r,
                "gross_right_to_wrong": counts[f"gross_manufactured_{stage}"],
                "gross_wrong_to_right": counts[f"gross_repaired_{stage}"],
                "net_gross_error_change": counts[f"gross_manufactured_{stage}"] - counts[f"gross_repaired_{stage}"],
                "manufactured_share": m / EXPECTED["manufactured"],
                "byte_equivalent_ceiling": m * bytes_per_flip,
            }
        )
    class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        per_stage = []
        for stage in STAGE_NAMES:
            count = class_counts["earliest_manufactured"][stage][class_id]
            per_stage.append(
                {
                    "stage": stage,
                    "manufactured": count,
                    "rate_per_class_pixel": count / int(class_area[class_id]),
                    "per_million_class_pixels": 1_000_000.0 * count / int(class_area[class_id]),
                    "byte_equivalent_ceiling": count * bytes_per_flip,
                    "repaired": class_counts["earliest_repaired"][stage][class_id],
                }
            )
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_pixels": int(class_area[class_id]),
                "gt_fraction": int(class_area[class_id]) / N_PIXELS,
                "stages": per_stage,
            }
        )
    if int(class_area.sum()) != N_PIXELS:
        raise Mst1Error("contest-CUDA DALI GT class areas do not partition the n600 field")
    for stage_index, stage in enumerate(STAGE_NAMES):
        class_manufactured = sum(row["stages"][stage_index]["manufactured"] for row in class_rows)
        class_repaired = sum(row["stages"][stage_index]["repaired"] for row in class_rows)
        if class_manufactured != counts[f"earliest_manufactured_{stage}"]:
            raise Mst1Error(f"per-class manufactured counts do not close for {stage}")
        if class_repaired != counts[f"earliest_repaired_{stage}"]:
            raise Mst1Error(f"per-class repaired counts do not close for {stage}")
    previous_errors = state_errors["labels"]
    for row in stage_rows:
        current_errors = state_errors[row["stage"]]
        if row["net_gross_error_change"] != current_errors - previous_errors:
            raise Mst1Error(f"gross transition accounting does not close for {row['stage']}")
        previous_errors = current_errors
    result = {
        "schema": "ddm_mst1_manufactured_stage_split.v1",
        "verdict_scope": "INSTANCE:DX2_T4_n600 with macOS-CPU intermediate head observations",
        "score_claim": False,
        "pointer_moved": False,
        "gt_lineage": "contest-CUDA DALI GT from MS9/QS3 on every count",
        "wrong_definition": (
            "At each RGB intermediate, wrong means argmax(frozen upstream SegNet(intermediate)) != contest-CUDA DALI GT. "
            "The same frozen head, weights, batch16, and four CPU threads observe every intermediate; SegNet-forward and argmax "
            "are therefore an explicitly unseparated decision operator, not assigned a fictitious RGB-independent subcharge."
        ),
        "source_fields": sources,
        "totals": totals,
        "stage_rows": stage_rows,
        "class_rows": class_rows,
        "state_errors": state_errors,
        "state_disagreement_from_labels": state_disagreement_from_labels,
        "terminal_advisory_vs_cuda_pixels": terminal_advisory_vs_cuda,
        "terminal_advisory_exactly_equals_cuda": terminal_advisory_vs_cuda == 0,
        "bytes_per_eliminated_flip": bytes_per_flip,
        "mask_manifest": mask_manifest,
        "capture_axis": "[macOS-CPU advisory] frozen CPU-torch SegNet, batch16, 4 threads",
        "final_support_axis": "[contest-CUDA T4 component-only exact field replay]",
        "retention_tier": "local_disk_explicit_opt_in",
        "retention_store": str(store),
        "analysis_argv": sys.argv,
        "migration_complete": file_fact(store / "MIGRATION_COMPLETE.json"),
    }
    atomic_json(store / "MST1_RESULT.json", result)
    atomic_json(store / "ATTRIBUTION_MASK_MANIFEST.json", {"schema": "ddm_mst1_mask_manifest.v1", "masks": mask_manifest})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    migrate_parser = sub.add_parser(
        "migrate-capture",
        help="verify and copy the complete legacy capture into local receipt custody",
    )
    migrate_parser.add_argument("--source", type=Path, default=DEFAULT_LEGACY_CAPTURE)
    migrate_parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    migrate_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    analyse_parser = sub.add_parser("analyse", help="join retained stage fields to MS9 exact fields")
    analyse_parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    analyse_parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    analyse_parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    analyse_parser.add_argument("--argmax", type=Path, default=DEFAULT_ARGMAX)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "migrate-capture":
        return migrate_capture(args)
    if args.command == "analyse":
        return analyse(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
