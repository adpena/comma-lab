#!/usr/bin/env python3
"""Bounded, receiver-closed RJ2 joint-renderer exact-object CPU smoke.

The smoke is deliberately narrow (one pair, one W96 single-FiLM step, float32
CPU), but it does not remove a mechanism stage.  It consumes the WD3
score-native objective, both frozen scorers, MF1 boundary/margin inputs, the
exact DX2 token field, the RJ1 retained initialization, an exact-object carrier
Jacobian solve, and the production CAP1 -> DX2 -> RR5 carrier coder chain.

Every materialized payload is retained under the APDataStore arm root.  The
result is an engineering smoke only: it is not an n600 score or a family
verdict, and it never mutates the canonical pointer or ``upstream/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

sys.dont_write_bytecode = True

import brotli
import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_ap1_residue_purchase_scorer as ap1
from experiments import ddm_po1_t4_error_feedback_pose_compensation as po1
from experiments import ddm_rj1_renderer_joint_move as rj1
from experiments import ddm_sa2_compile_candidate as sa2
from experiments import ddm_wd2_student_receiver as receiver
from experiments import ddm_wd3_scorer_aware_width_distillation as wd3
from tac.scorer import load_differentiable_scorers

OUTPUT_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change")
DX2_RUNTIME: Final = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
DX2_ARCHIVE: Final = DX2_RUNTIME / "archive.zip"
DX2_RAW: Final = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated/0.raw")
DX2_TOKENS: Final = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
RJ1_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1")
RJ1_INIT: Final = RJ1_ROOT / "rungs/film_amortized_flat_w96/renderer_initialization.pt"
MF1_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_mf1_manufactured_seg_repair/measurement_v3")
MF1_BOUNDARY_R9: Final = MF1_ROOT / "retained/boundary_masks/decoded_token_boundary_chebyshev_r9.n600.packbits"
MF1_MANUFACTURED: Final = MF1_ROOT / "retained/native_manufactured_mask.n600.packbits"
GT_CACHE: Final = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
PAIR_COUNT: Final = 600
PAIR_ID: Final = 0
DIMENSIONS: Final = 12
SEED: Final = 20260823
RATE_DENOMINATOR: Final = 37_545_489
RATE_EXCHANGE_S_PER_BYTE: Final = 6.658590e-7
TARGET_BYTES: Final = 137_986
DX2_ZERO_DISTORTION_SHED_BYTES: Final = 150
CARRIER_BROTLI_QUALITY: Final = 9
CARRIER_BROTLI_LGWIN: Final = 16
EMA_DECAY: Final = 0.9
LEARNING_RATE: Final = 1.0e-6
MAX_GRAD_NORM: Final = 1.0
MINIMUM_FREE_BYTES: Final = 4 * 1024**3
AXIS: Final = "[macOS-CPU advisory n1 exact local scorers; engineering smoke]"

PINS: Final = {
    "dx2_archive": (
        DX2_ARCHIVE,
        180_368,
        "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    ),
    "dx2_raw": (
        DX2_RAW,
        3_662_409_600,
        "7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7",
    ),
    "dx2_tokens": (
        DX2_TOKENS,
        117_964_800,
        "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    ),
    "rj1_initialization": (
        RJ1_INIT,
        253_955,
        "e74ba046af251808ef105cf0a2295f6133efa194360148f3110762765b9db434",
    ),
    "mf1_boundary_r9": (
        MF1_BOUNDARY_R9,
        14_745_600,
        "8adbd0f04f66f7527c7245448ff10351b41e2c640b7170cce0d04a794366e501",
    ),
    "mf1_manufactured": (
        MF1_MANUFACTURED,
        14_745_600,
        "cd7b0176e0d6a41d73c9ae539acf9a24304f3ff0a87a96faaa83709673beffb6",
    ),
    "gt_cache": (
        GT_CACHE,
        5_078_017_610,
        "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6",
    ),
}


class RJ2Error(RuntimeError):
    """An exact-object, retention, resume, or mechanism invariant failed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RJ2Error(f"required retained file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_retained_records(
    value: object,
    *,
    allowed_root: Path,
    _cache: dict[Path, dict[str, Any]] | None = None,
) -> int:
    """Validate every nested file record without accepting paths outside custody."""

    cache = {} if _cache is None else _cache
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = Path(str(value["path"])).resolve()
            root = allowed_root.resolve()
            if path != root and root not in path.parents:
                raise RJ2Error(f"retained record escaped custody root: {path}")
            observed = cache.setdefault(path, file_record(path))
            expected = {name: value[name] for name in ("path", "bytes", "sha256")}
            if observed != expected:
                raise RJ2Error(f"retained payload drifted: {path}")
            return 1
        return sum(validate_retained_records(item, allowed_root=allowed_root, _cache=cache) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(validate_retained_records(item, allowed_root=allowed_root, _cache=cache) for item in value)
    return 0


def git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode or not completed.stdout.strip():
        raise RJ2Error(f"cannot resolve git provenance for {root}")
    return completed.stdout.strip()


def local_tree_snapshot(root: Path) -> dict[str, Any]:
    """Fingerprint the exact advisory tree, including bytecode and symlinks."""

    skip = {".git", ".pytest_cache", ".mypy_cache", "node_modules"}
    outer = hashlib.sha256()
    file_count = 0
    symlink_count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if any(part in skip for part in relative.parts):
            continue
        if path.is_symlink():
            payload = os.readlink(path).encode()
            kind = "symlink"
            payload_bytes = len(payload)
            payload_sha256 = sha256_bytes(payload)
            symlink_count += 1
        elif path.is_file():
            kind = "file"
            payload_bytes = path.stat().st_size
            payload_sha256 = sha256_file(path)
            file_count += 1
        else:
            continue
        outer.update(relative.as_posix().encode())
        outer.update(b"\n")
        outer.update(kind.encode())
        outer.update(b"\n")
        outer.update(str(payload_bytes).encode())
        outer.update(b"\n")
        outer.update(payload_sha256.encode())
        outer.update(b"\n")
    return {
        "sha256": outer.hexdigest(),
        "regular_files": file_count,
        "symlinks": symlink_count,
    }


def provenance_record() -> dict[str, Any]:
    upstream_snapshot = local_tree_snapshot(REPO / "upstream")
    return {
        "repo_git_head": git_head(REPO),
        "implementation": file_record(Path(__file__)),
        "implementation_git_state": "untracked before governed serializer landing",
        "upstream_git_head": git_head(REPO / "upstream"),
        "upstream_snapshot_sha256": upstream_snapshot["sha256"],
        "upstream_snapshot_regular_files": upstream_snapshot["regular_files"],
        "upstream_snapshot_symlinks": upstream_snapshot["symlinks"],
        "upstream_snapshot_contract": "ddm_rj2_local_advisory_tree.v1",
        "upstream_snapshot_source_only": False,
        "upstream_snapshot_caveat": (
            "exact local tree includes pre-existing executable bytecode; local advisory only, "
            "never authority or promotion eligible"
        ),
        "seed": SEED,
        "deterministic_algorithms": True,
        "torch_threads": 4,
        "torch_interop_threads": 1,
    }


def atomic_bytes(path: Path, value: bytes) -> dict[str, Any]:
    """Retain immutable bytes atomically; differing existing bytes fail closed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != value:
            raise RJ2Error(f"refusing to overwrite differing retained payload: {path}")
        return file_record(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_record(path)


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload)


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return atomic_bytes(path, buffer.getvalue())


def atomic_torch_once(path: Path, value: object) -> dict[str, Any]:
    """Write a distinct checkpoint once; resumptions never overwrite stages."""

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return atomic_bytes(path, buffer.getvalue())


def require_pins() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, (path, expected_bytes, expected_sha256) in PINS.items():
        record = file_record(path)
        if (record["bytes"], record["sha256"]) != (
            expected_bytes,
            expected_sha256,
        ):
            raise RJ2Error(f"source pin changed: {name}")
        records[name] = record
    return records


def storage_preflight(output: Path, required_free_bytes: int, *, phase: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    resolved = output.resolve()
    expected = OUTPUT_ROOT.resolve()
    if resolved != expected and expected not in resolved.parents:
        raise RJ2Error(f"RJ2 output must stay under {expected}")
    free = shutil.disk_usage(output).free
    result = {
        "schema": "ddm_rj2_storage_preflight.v1",
        "phase": phase,
        "status": "PASS" if free >= required_free_bytes else "FAIL",
        "root": str(resolved),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": free,
        "routing": "APDataStore because Vertigo is read-only/full for this arm",
        "cleanup": "certify-or-block; no retained payload is deleted",
    }
    if result["status"] != "PASS":
        raise RJ2Error("APDataStore storage preflight failed")
    stem = f"STORAGE_PREFLIGHT_{phase.upper()}"
    index = 1
    while (output / f"{stem}_{index:02d}.json").exists():
        index += 1
    result["receipt"] = atomic_json(output / f"{stem}_{index:02d}.json", result)
    return result


def configure_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    major, minor = np.lib.format.read_magic(stream)
    if (major, minor) == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif (major, minor) == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise RJ2Error(f"unsupported NPY version: {(major, minor)}")
    return tuple(shape), bool(fortran), np.dtype(dtype)


def read_stored_npz_pair(
    archive: Path,
    member: str,
    pair: int,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
) -> np.ndarray:
    """Read one row from a ZIP_STORED NPY without materializing its n600 peer."""

    with zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo(f"{member}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise RJ2Error(f"GT member is not ZIP_STORED: {member}")
        with bundle.open(info) as stream:
            shape, fortran, dtype = _read_npy_header(stream)
            if shape != expected_shape or fortran or dtype != np.dtype(expected_dtype):
                raise RJ2Error(f"GT member geometry differs: {member}")
            if not 0 <= pair < shape[0]:
                raise RJ2Error("GT pair index exceeds the retained population")
            row_items = math.prod(shape[1:])
            row_bytes = row_items * dtype.itemsize
            stream.seek(pair * row_bytes, 1)
            payload = stream.read(row_bytes)
            if len(payload) != row_bytes:
                raise RJ2Error(f"truncated GT row: {member}/{pair}")
    return np.frombuffer(payload, dtype=dtype).copy().reshape(shape[1:])


def read_packbits_pair(path: Path, pair: int) -> np.ndarray:
    row_bits = receiver.EVAL_H * receiver.EVAL_W
    if row_bits % 8:
        raise RJ2Error("MF1 mask row is not byte aligned")
    row_bytes = row_bits // 8
    with path.open("rb") as stream:
        stream.seek(pair * row_bytes)
        payload = stream.read(row_bytes)
    if len(payload) != row_bytes:
        raise RJ2Error(f"truncated MF1 mask row: {path}")
    return (
        np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
        .reshape(receiver.EVAL_H, receiver.EVAL_W)
        .astype(bool, copy=False)
    )


def load_source_subset(pair: int) -> dict[str, np.ndarray]:
    raw = np.memmap(
        DX2_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(2 * PAIR_COUNT, receiver.CAMERA_H, receiver.CAMERA_W, 3),
    )
    frame0 = np.asarray(raw[2 * pair]).copy()
    frame1 = np.asarray(raw[2 * pair + 1]).copy()
    del raw
    tokens = np.memmap(
        DX2_TOKENS,
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, receiver.EVAL_H, receiver.EVAL_W),
    )
    pair_tokens = np.asarray(tokens[pair]).copy()
    del tokens
    argmax = read_stored_npz_pair(
        GT_CACHE,
        "lstars",
        pair,
        expected_shape=(PAIR_COUNT, receiver.EVAL_H, receiver.EVAL_W),
        expected_dtype=np.dtype("<i8"),
    )
    margin = read_stored_npz_pair(
        GT_CACHE,
        "margins",
        pair,
        expected_shape=(PAIR_COUNT, receiver.EVAL_H, receiver.EVAL_W),
        expected_dtype=np.dtype("<f4"),
    )
    pose = read_stored_npz_pair(
        GT_CACHE,
        "gt_poses",
        pair,
        expected_shape=(PAIR_COUNT, 6),
        expected_dtype=np.dtype("<f8"),
    )
    boundary = read_packbits_pair(MF1_BOUNDARY_R9, pair)
    manufactured = read_packbits_pair(MF1_MANUFACTURED, pair)
    threshold = float(np.quantile(margin, 0.10))
    selected = boundary | manufactured | (margin <= threshold)
    if not np.any(selected):
        raise RJ2Error("MF1 boundary/margin selection is empty")
    return {
        "frame0": frame0,
        "frame1": frame1,
        "tokens": pair_tokens,
        "original_argmax": argmax,
        "original_margin": margin,
        "original_pose6": pose,
        "mf1_boundary_r9": boundary,
        "mf1_manufactured": manufactured,
        "selected_cells": selected,
        "margin_threshold": np.asarray(threshold, dtype=np.float32),
    }


def retain_subset(root: Path, subset: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: atomic_npy(root / f"{name}.npy", np.asarray(value)) for name, value in subset.items()}


def load_retained_subset(records: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, record in records.items():
        path = Path(str(record["path"]))
        if file_record(path) != dict(record):
            raise RJ2Error(f"retained source subset drifted: {name}")
        result[name] = np.load(path, allow_pickle=False)
    return result


def model_from_state(state: Mapping[str, torch.Tensor]) -> receiver.StudentSemanticRenderer:
    spec = receiver.StudentSpec("film_amortized_flat_w96", "flattened", 96, 4)
    model = receiver.StudentSemanticRenderer(spec)
    model.load_state_dict(OrderedDict(state), strict=True)
    return model


def quantized_master(
    model: receiver.StudentSemanticRenderer,
    tokens: torch.Tensor,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    state = receiver.fake_quantize_state(model)
    low = functional_call(model, state, (tokens.long(), pair_indices))
    camera = F.interpolate(
        low,
        size=(receiver.CAMERA_H, receiver.CAMERA_W),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 255.0)
    rounded = camera.round().clamp(0.0, 255.0)
    return camera + (rounded - camera).detach()


def parsed_packet_master(
    packet: bytes, tokens: torch.Tensor, pair_indices: torch.Tensor
) -> tuple[receiver.StudentSemanticRenderer, torch.Tensor]:
    parsed = receiver.unpack_student(packet)
    if receiver.pack_student(parsed) != packet:
        raise RJ2Error("WD2S packet parse/repack differs")
    master = receiver.camera_uint8(parsed, tokens, pair_indices).float()
    return parsed, master


def frame_pair(frame0_hwc: np.ndarray | torch.Tensor, frame1_chw: torch.Tensor) -> torch.Tensor:
    if isinstance(frame0_hwc, np.ndarray):
        first = torch.from_numpy(np.ascontiguousarray(frame0_hwc)).permute(2, 0, 1).float()
    else:
        first = frame0_hwc.float()
    if first.shape != frame1_chw.shape:
        raise RJ2Error("frame-0 and frame-1 camera geometries differ")
    return torch.stack((first, frame1_chw), dim=0).unsqueeze(0)


def measured_metrics(
    pair: torch.Tensor,
    *,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    original_argmax: torch.Tensor,
    original_pose6: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        pose6, logits = wd3.scorer_forward(pair, posenet, segnet)
    d_seg = float((logits.argmax(dim=1) != original_argmax).float().mean())
    d_pose = float((pose6 - original_pose6).square().mean())
    return (
        {
            "d_seg": d_seg,
            "d_pose": d_pose,
        },
        pose6.detach(),
        logits.detach(),
    )


def contest_arithmetic(*, d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, float]:
    rate = 25.0 * archive_bytes / RATE_DENOMINATOR
    seg = 100.0 * d_seg
    pose = math.sqrt(10.0 * d_pose)
    return {
        "rate_s": rate,
        "seg_s": seg,
        "pose_s": pose,
        "s": rate + seg + pose,
    }


def ema_update(
    shadow: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
) -> OrderedDict[str, torch.Tensor]:
    if not 0.0 <= decay < 1.0:
        raise RJ2Error("EMA decay must be in [0,1)")
    current = model.state_dict()
    if tuple(shadow) != tuple(current):
        raise RJ2Error("EMA and live state keys differ")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in current.items():
        prior = shadow[name].to(dtype=value.dtype)
        result[name] = (
            (decay * prior + (1.0 - decay) * value.detach()).cpu().clone()
            if value.is_floating_point() or value.is_complex()
            else value.detach().cpu().clone()
        )
    return result


def checkpoint_payload(
    *,
    source_checkpoint: str | None,
    step: int,
    model: receiver.StudentSemanticRenderer,
    ema_shadow: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    subset: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "ddm_rj2_joint_checkpoint.v1",
        "source_checkpoint": source_checkpoint,
        "step": step,
        "pair_ids": [PAIR_ID],
        "precision": "float32",
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "ema_decay": EMA_DECAY,
        "live_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "ema_shadow": {name: value.detach().cpu().clone() for name, value in ema_shadow.items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "subset": dict(subset),
        "sources": dict(sources),
        "provenance": dict(provenance),
        "history": list(history),
        "deployment_weights": "ema_shadow",
        "resumable_from_disk": True,
        "atomic": True,
    }


def prepare(output: Path) -> dict[str, Any]:
    completed_path = output / "PREPARE_RESULT.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("schema") != "ddm_rj2_prepare.v1" or completed.get("status") != "READY_FOR_CPU_SMOKE":
            raise RJ2Error("retained prepare receipt is invalid")
        validated = validate_retained_records(
            {
                "birth_checkpoint": completed["birth_checkpoint"],
                "initial_packet": completed["initial_packet"],
                "subset": completed["subset"],
                "storage": completed["storage"],
            },
            allowed_root=output,
        )
        if validated < 2:
            raise RJ2Error("completed prepare receipt has incomplete retained custody")
        return completed
    configure_determinism(SEED)
    storage = storage_preflight(output, MINIMUM_FREE_BYTES, phase="prepare")
    sources = require_pins()
    provenance = provenance_record()
    subset = load_source_subset(PAIR_ID)
    subset_records = retain_subset(output / "retained/source_pair_0000", subset)
    state = torch.load(RJ1_INIT, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RJ2Error("RJ1 initialization is not a state dictionary")
    model = model_from_state(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    ema_shadow = OrderedDict((name, value.detach().cpu().clone()) for name, value in model.state_dict().items())
    birth_path = output / "checkpoints/stage_00_birth.pt"
    checkpoint = checkpoint_payload(
        source_checkpoint=None,
        step=0,
        model=model,
        ema_shadow=ema_shadow,
        optimizer=optimizer,
        subset=subset_records,
        sources=sources,
        provenance=provenance,
        history=[],
    )
    birth = atomic_torch_once(birth_path, checkpoint)
    initial_packet = receiver.pack_student(model)
    initial_packet_record = atomic_bytes(output / "retained/stage_00_birth/semantic.wd2s", initial_packet)
    result = {
        "schema": "ddm_rj2_prepare.v1",
        "status": "READY_FOR_CPU_SMOKE",
        "axis": AXIS,
        "score_claim": False,
        "storage": storage,
        "sources": sources,
        "provenance": provenance,
        "subset": subset_records,
        "selection": {
            "pair": PAIR_ID,
            "selected_cells": int(subset["selected_cells"].sum()),
            "population_cells": int(subset["selected_cells"].size),
            "mf1_boundary_r9_cells": int(subset["mf1_boundary_r9"].sum()),
            "mf1_manufactured_cells": int(subset["mf1_manufactured"].sum()),
            "margin_quantile": 0.10,
            "margin_threshold": float(subset["margin_threshold"]),
            "use": "training-time selective objective only; no shipped mask",
        },
        "birth_checkpoint": birth,
        "initial_packet": initial_packet_record,
        "resumable_from_disk": True,
        "stage_checkpoints_distinct": True,
        "all_payloads_retained": True,
    }
    atomic_json(output / "PREPARE_RESULT.json", result)
    return result


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "ddm_rj2_joint_checkpoint.v1"
        or payload.get("resumable_from_disk") is not True
        or payload.get("deployment_weights") != "ema_shadow"
        or not isinstance(payload.get("provenance"), Mapping)
    ):
        raise RJ2Error("resume checkpoint is incomplete or belongs to another arm")
    return dict(payload)


def restore_rng_state(checkpoint: Mapping[str, Any]) -> None:
    rng = checkpoint.get("rng")
    if not isinstance(rng, Mapping) or set(rng) != {"torch", "numpy", "python"}:
        raise RJ2Error("resume checkpoint RNG state is incomplete")
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def carrier_frame0(
    state: po1.CarrierState,
    codes: np.ndarray,
    *,
    renderer_module: Any,
) -> torch.Tensor:
    raw_basis = torch.from_numpy(
        state.basis_codes.reshape(DIMENSIONS, 3, 24, 32).astype(np.float32) * state.basis_scales[:, None, None, None]
    )
    basis = renderer_module.normalized_basis(raw_basis)
    values = torch.from_numpy(np.asarray(codes, dtype=np.float32))
    coefficients = values[None] * torch.from_numpy(state.coefficient_scales)[None]
    carrier = torch.einsum("bk,kchw->bchw", coefficients, basis) / math.sqrt(DIMENSIONS)
    low = (127.5 + renderer_module.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0)
    low = low + (low.round() - low).detach()
    high = F.interpolate(
        low,
        size=(receiver.CAMERA_H, receiver.CAMERA_W),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 255.0)
    high = high + (high.round() - high).detach()
    return high[0]


def carrier_pose_and_jacobian(
    state: po1.CarrierState,
    codes: np.ndarray,
    master: torch.Tensor,
    *,
    posenet: torch.nn.Module,
    renderer_module: Any,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    code_tensor = torch.from_numpy(np.asarray(codes, dtype=np.float32)).requires_grad_(True)
    raw_basis = torch.from_numpy(
        state.basis_codes.reshape(DIMENSIONS, 3, 24, 32).astype(np.float32) * state.basis_scales[:, None, None, None]
    )
    basis = renderer_module.normalized_basis(raw_basis)
    coefficient = code_tensor[None] * torch.from_numpy(state.coefficient_scales)[None]
    carrier = torch.einsum("bk,kchw->bchw", coefficient, basis) / math.sqrt(DIMENSIONS)
    low = (127.5 + renderer_module.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0)
    low = low + (low.round() - low).detach()
    high = F.interpolate(
        low,
        size=(receiver.CAMERA_H, receiver.CAMERA_W),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 255.0)
    slave = high + (high.round() - high).detach()
    pair = torch.stack((slave[0], master), dim=0).unsqueeze(0)
    output = posenet(posenet.preprocess_input(pair))["pose"][0, :6]
    rows = []
    for dimension in range(6):
        gradient = torch.autograd.grad(
            output[dimension],
            code_tensor,
            retain_graph=dimension < 5,
        )[0]
        rows.append(gradient.detach().numpy().astype(np.float64, copy=False))
    return (
        output.detach().numpy().astype(np.float64, copy=False),
        np.stack(rows),
        slave[0].detach(),
    )


def _runtime_riders(runtime: Path) -> SimpleNamespace:
    prior = rj1._clear_runtime_modules()
    sys.path.insert(0, str(runtime.resolve()))
    try:
        rr5 = importlib.import_module("runtime.rr5_arith_basis")
        dx2 = importlib.import_module("runtime.dx2_cabac_coefficients")
        residual = importlib.import_module("runtime.residual_archive")
        carrier_repack = importlib.import_module("runtime.carrier_repack")
        coefficient_codec = importlib.import_module("runtime.entropy.coefficient_ar1_codec")
        coefficient_predictor = importlib.import_module("runtime.entropy.coefficient_predictor")
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "runtime" or name.startswith("runtime."):
                sys.modules.pop(name, None)
        sys.modules.update(prior)
    return SimpleNamespace(
        rr5=rr5,
        dx2=dx2,
        residual_archive=residual,
        carrier_repack=carrier_repack,
        coefficient_codec=coefficient_codec,
        coefficient_predictor=coefficient_predictor,
    )


def encode_carrier_stream(
    state: po1.CarrierState,
    codes: np.ndarray,
    *,
    runtime: Path,
    retention_root: Path,
    source_predictor_metadata: bytes,
) -> dict[str, Any]:
    """Run the exact production carrier encoder chain and retain all identities."""

    values = np.asarray(codes, dtype=np.int32)
    if values.shape != (PAIR_COUNT, DIMENSIONS):
        raise RJ2Error("carrier lattice geometry differs")
    canonical = po1.encode_canonical_carrier(state, values)
    retained = {"canonical": atomic_bytes(retention_root / "canonical.cpr1", canonical)}
    modules = _runtime_riders(runtime)
    cap1 = ap1.encode_cap1_with_source_predictor(
        canonical,
        values,
        source_predictor_metadata,
        modules,
    )
    cap1_report = {
        "encoder": "fixed exact-DX2 predictor metadata",
        "source_predictor_sha256": sha256_bytes(source_predictor_metadata),
    }
    retained["cap1"] = atomic_bytes(retention_root / "carrier.cap1", cap1)
    if modules.coefficient_codec.decode_cap1(cap1, frames=PAIR_COUNT, dimensions=DIMENSIONS) != canonical:
        raise RJ2Error("CAP1 parse-back differs")
    if not cap1.startswith(b"CAP1\x01\x00\x00\x00"):
        raise RJ2Error("CAP1 prefix differs")
    stripped = cap1[8:]
    bit_counts, predictor = stripped[:6], stripped[6:42]
    scales, lengths = stripped[42:138], stripped[138:170]
    ks, rest = stripped[170:182], stripped[182:]
    canonical_section = bit_counts + scales + predictor + lengths + ks + rest + state.selector[5:]
    plain = sa2.pack_cap1_metadata(canonical_section)
    retained["plain_packed"] = atomic_bytes(retention_root / "carrier.plain_packed.bin", plain)
    # The receiver restores RR5 first and DX2 second, so the encoder applies
    # the two exact transforms in the inverse order.
    dx2_result = modules.dx2.apply_cabac_to_carrier_body(plain)
    dx2_body = bytes(dx2_result["body"])
    retained["dx2_body"] = atomic_bytes(retention_root / "carrier.dx2_body.bin", dx2_body)
    if modules.dx2.restore_carrier_body(dx2_body) != plain:
        raise RJ2Error("DX2 carrier inverse differs")
    rr5_result = modules.rr5.apply_rider_to_carrier_body(dx2_body)
    rr5_body = bytes(rr5_result["body"])
    retained["rr5_body"] = atomic_bytes(retention_root / "carrier.rr5_body.bin", rr5_body)
    if modules.rr5.restore_carrier_body(rr5_body) != dx2_body:
        raise RJ2Error("RR5 carrier inverse differs")
    completed = subprocess.run(
        [
            "brotli",
            "-q",
            str(CARRIER_BROTLI_QUALITY),
            f"--lgwin={CARRIER_BROTLI_LGWIN}",
            "-c",
        ],
        input=rr5_body,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RJ2Error("production carrier Brotli encoder failed: " + completed.stderr.decode(errors="replace"))
    stream = completed.stdout
    retained["stream"] = atomic_bytes(retention_root / "carrier.stream", stream)
    if brotli.decompress(stream) != rr5_body:
        raise RJ2Error("carrier Brotli round-trip differs")
    return {
        "canonical": canonical,
        "cap1": cap1,
        "plain_packed": plain,
        "rr5_body": rr5_body,
        "dx2_body": dx2_body,
        "stream": stream,
        "retained": retained,
        "cap1_report": cap1_report,
        "rr5_basis_bits": int(rr5_result["rider_basis_bits"]),
        "dx2_cabac_bits": int(dx2_result["cabac_bits"]),
    }


_PARSEBACK_PROGRAM: Final = r"""
import hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
runtime, archive, expected_codes, expected_semantic = sys.argv[1:]
sys.path.insert(0, runtime)
sys.path.insert(0, runtime + "/cpr1")
import carrier_codec
from runtime.carrier_repack import materialize_cpr1
from runtime.residual_archive import read_residual_archive
parts = read_residual_archive(Path(archive))
canonical = materialize_cpr1(
    parts.carrier_blob, SimpleNamespace(N=600, CARRIER_DIM=12))
_bs, _bc, _cs, encoded = carrier_codec.decode_compact_carrier(
    canonical, basis_count=12*3*24*32, frames=600, dimensions=12)
delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
codes = np.cumsum(delta, axis=0) & 0xFFF
codes = np.where(codes >= 0x800, codes - 0x1000, codes).astype(np.int32)
wanted = np.load(expected_codes, allow_pickle=False)
report = {
    "codes_exact": bool(np.array_equal(codes, wanted)),
    "semantic_sha256": hashlib.sha256(parts.semantic_blob).hexdigest(),
    "expected_semantic_sha256": expected_semantic,
    "carrier_blob_sha256": hashlib.sha256(parts.carrier_blob).hexdigest(),
    "compensation_blob_bytes": None if parts.compensation_blob is None else len(parts.compensation_blob),
}
if not report["codes_exact"] or report["semantic_sha256"] != expected_semantic:
    raise SystemExit(json.dumps(report, sort_keys=True))
print(json.dumps(report, sort_keys=True))
"""


def fresh_process_parseback(
    *,
    runtime: Path,
    archive: Path,
    expected_codes: Path,
    semantic_sha256: str,
    output: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            "-c",
            _PARSEBACK_PROGRAM,
            str(runtime.resolve()),
            str(archive.resolve()),
            str(expected_codes.resolve()),
            semantic_sha256,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
    )
    transcript = atomic_bytes(
        output,
        (completed.stdout + "\n--- STDERR ---\n" + completed.stderr).encode(),
    )
    if completed.returncode:
        raise RJ2Error("fresh-process receiver parse-back failed")
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    if report.get("codes_exact") is not True:
        raise RJ2Error("fresh-process carrier codes differ")
    return {"status": "PASS", "report": report, "transcript": transcript}


def prepare_runtime_copy(source: Path, destination: Path) -> dict[str, Any]:
    """Atomically copy generic runtime code while excluding the source archive."""

    resumed = destination.exists()
    if not resumed:
        temporary = destination.with_name(f".{destination.name}.copy.tmp")
        if temporary.exists():
            raise RJ2Error(f"incomplete retained runtime copy requires inspection: {temporary}")
        shutil.copytree(
            source,
            temporary,
            ignore=shutil.ignore_patterns("archive.zip", "__pycache__", "*.pyc", ".DS_Store", "._*"),
            copy_function=shutil.copyfile,
        )
        os.replace(temporary, destination)
    if not resumed and (destination / "archive.zip").exists():
        raise RJ2Error("fresh RJ2 runtime copy unexpectedly contains a source archive")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "source_archive_excluded": True,
        "atomic_copy": True,
        "resumed": resumed,
        "archive_present_before_patch": (destination / "archive.zip").is_file(),
    }


def clean_trivial_runtime_residue(root: Path) -> None:
    """Remove only host bytecode caches and AppleDouble metadata from a copied runtime."""

    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and (path.suffix == ".pyc" or path.name.startswith("._"))),
        reverse=True,
    )
    for path in files:
        path.unlink()
    directories = sorted(
        (path for path in root.rglob("__pycache__") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        try:
            path.rmdir()
        except OSError as error:
            raise RJ2Error(f"nontrivial runtime cache directory remains: {path}") from error


def object_repricing_rows() -> list[dict[str, str]]:
    """Typed dependency table; RE-PRICED never means automatically reopened."""

    return [
        {
            "leg": "QS2",
            "disposition": "RE-PRICED",
            "object_change": "model and carrier changed; field unchanged",
            "boundary": "fresh exact-object compensation required; no old delta transfers",
        },
        {
            "leg": "RE1",
            "disposition": "RE-PRICED",
            "object_change": "renderer model changed; base field unchanged",
            "boundary": "scorer effect changes; its old zero-byte field price is not banked",
        },
        {
            "leg": "EC1",
            "disposition": "RE-PRICED",
            "object_change": "renderer model changed; proposal field absent from this smoke",
            "boundary": "realized B/H/W must be remeasured on the moved renderer",
        },
        {
            "leg": "LD1",
            "disposition": "RE-PRICED",
            "object_change": "renderer model changed; token coder and base field unchanged",
            "boundary": "old RC64 rate closure stands; distortion response is newly priced",
        },
        {
            "leg": "AE1",
            "disposition": "UNCHANGED",
            "object_change": "neither token field nor HPAC probability object changed",
            "boundary": "anti-predicted excess closure transfers",
        },
        {
            "leg": "OE1",
            "disposition": "UNCHANGED",
            "object_change": "neither token field nor causal HPAC object changed",
            "boundary": "escape-member closure transfers",
        },
        {
            "leg": "HPAC sharp-optimum rows",
            "disposition": "UNCHANGED",
            "object_change": "HPAC model and categorical field are byte-identical",
            "boundary": "no model-rate leg is reopened by a renderer-only move",
        },
    ]


def smoke(output: Path, resume_from: Path, max_steps: int) -> dict[str, Any]:
    if max_steps != 1:
        raise RJ2Error("the chartered bounded smoke is sealed at exactly one step")
    completed_path = output / "SMOKE_RESULT.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if (
            completed.get("schema") != "ddm_rj2_joint_renderer_smoke.v1"
            or completed.get("status") != "MECHANISM_COMPLETE_ENGINEERING_SMOKE"
            or completed.get("all_payloads_retained") is not True
        ):
            raise RJ2Error("retained completed smoke receipt is invalid")
        if completed["checkpoints"]["resume_from"] != file_record(resume_from):
            raise RJ2Error("completed smoke belongs to a different resume checkpoint")
        validated = validate_retained_records(
            {
                "payloads": completed["payloads"],
                "retained_fields": completed["retained_fields"],
                "checkpoints": completed["checkpoints"],
                "mechanism_gates": completed["mechanism_gates"],
                "storage": completed["storage"],
            },
            allowed_root=output,
        )
        if validated < 2:
            raise RJ2Error("completed smoke receipt has incomplete retained custody")
        return completed
    configure_determinism(SEED)
    storage = storage_preflight(output, MINIMUM_FREE_BYTES, phase="smoke")
    sources = require_pins()
    provenance = provenance_record()
    checkpoint = load_checkpoint(resume_from)
    if checkpoint["sources"] != sources:
        raise RJ2Error("resume source binding differs")
    if checkpoint["provenance"] != provenance:
        raise RJ2Error("resume provenance differs from the current executable surface")
    subset_records = checkpoint["subset"]
    subset = load_retained_subset(subset_records)
    model = model_from_state(checkpoint["live_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ema_shadow = OrderedDict((name, value.detach().cpu().clone()) for name, value in checkpoint["ema_shadow"].items())
    restore_rng_state(checkpoint)
    start_step = int(checkpoint["step"])
    if start_step > max_steps:
        raise RJ2Error("resume checkpoint is beyond the sealed smoke horizon")
    history = list(checkpoint.get("history", []))
    pair_id = torch.tensor([PAIR_ID], dtype=torch.long)
    tokens = torch.from_numpy(np.asarray(subset["tokens"], dtype=np.uint8)[None])
    original_argmax = torch.from_numpy(np.asarray(subset["original_argmax"], dtype=np.int64)[None])
    original_pose6 = torch.from_numpy(np.asarray(subset["original_pose6"], dtype=np.float32)[None])
    selected = torch.from_numpy(np.asarray(subset["selected_cells"], dtype=bool)[None])
    source_frame1 = torch.from_numpy(np.ascontiguousarray(subset["frame1"])).permute(2, 0, 1).float()[None]
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device="cpu")
    posenet.eval()
    segnet.eval()

    source_pair = frame_pair(subset["frame0"], source_frame1[0])
    source_metrics, source_pose, source_logits = measured_metrics(
        source_pair,
        posenet=posenet,
        segnet=segnet,
        original_argmax=original_argmax,
        original_pose6=original_pose6,
    )
    before_master = quantized_master(model, tokens, pair_id)
    before_pair = frame_pair(subset["frame0"], before_master[0])
    before_metrics, before_pose, before_logits = measured_metrics(
        before_pair,
        posenet=posenet,
        segnet=segnet,
        original_argmax=original_argmax,
        original_pose6=original_pose6,
    )
    smoke_payload_root = output / "retained/smoke_pair_0000"
    retained_fields = {
        "source_pair": atomic_npy(smoke_payload_root / "source_pair.float32.npy", source_pair.numpy()),
        "source_pose": atomic_npy(smoke_payload_root / "source_pose.float32.npy", source_pose.numpy()),
        "source_logits": atomic_npy(smoke_payload_root / "source_logits.float32.npy", source_logits.numpy()),
        "before_pair": atomic_npy(smoke_payload_root / "before_pair.float32.npy", before_pair.detach().numpy()),
        "before_pose": atomic_npy(smoke_payload_root / "before_pose.float32.npy", before_pose.numpy()),
        "before_logits": atomic_npy(smoke_payload_root / "before_logits.float32.npy", before_logits.numpy()),
    }

    for step in range(start_step + 1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        master = quantized_master(model, tokens, pair_id)
        pair = frame_pair(subset["frame0"], master[0])
        student_pose, student_logits = wd3.scorer_forward(pair, posenet, segnet)
        calibration = wd3.calibrate_soft_disagreement(student_logits.detach(), original_argmax)
        top2 = source_logits.topk(k=2, dim=1).values
        total, components = wd3.score_native_objective(
            student_logits=student_logits,
            student_pose6=student_pose,
            student_frame1=master,
            teacher_logits=source_logits,
            teacher_argmax=source_logits.argmax(dim=1),
            teacher_margin=top2[:, 0] - top2[:, 1],
            teacher_pose6=source_pose,
            original_argmax=original_argmax,
            original_pose6=original_pose6,
            teacher_frame1=source_frame1,
            selected_cells=selected,
            thresholds=wd3.StageThresholds(
                calibration_scale=calibration["stage_frozen_calibration_scale"],
                margin_ceiling=0.0,
                teacher_kl_ceiling=0.0,
                decode_ceiling=wd3.DECODE_MSE_CEILING,
            ),
            duals=wd3.DualState(),
        )
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM))
        optimizer.step()
        ema_shadow = ema_update(ema_shadow, model, EMA_DECAY)
        row = {
            "step": step,
            "total_loss": float(total.detach()),
            "gradient_norm_before_clip": gradient_norm,
            "components": {name: float(value.detach()) for name, value in components.items()},
            "mf1_selected_cells": int(selected.sum()),
            "precision": "float32",
            "pair_count": 1,
        }
        history.append(row)
        periodic = checkpoint_payload(
            source_checkpoint=str(resume_from.resolve()),
            step=step,
            model=model,
            ema_shadow=ema_shadow,
            optimizer=optimizer,
            subset=subset_records,
            sources=sources,
            provenance=provenance,
            history=history,
        )
        atomic_torch_once(output / f"checkpoints/stage_10_joint_step_{step:04d}.pt", periodic)

    stage_end = checkpoint_payload(
        source_checkpoint=str(resume_from.resolve()),
        step=max_steps,
        model=model,
        ema_shadow=ema_shadow,
        optimizer=optimizer,
        subset=subset_records,
        sources=sources,
        provenance=provenance,
        history=history,
    )
    stage_end_record = atomic_torch_once(
        output / f"checkpoints/stage_20_joint_smoke_end_step_{max_steps:04d}.pt",
        stage_end,
    )
    deployment_model = model_from_state(ema_shadow)
    packet = receiver.pack_student(deployment_model)
    _parsed_model, final_master = parsed_packet_master(packet, tokens, pair_id)
    semantic_record = atomic_bytes(output / "retained/final_object/semantic.wd2s", packet)
    final_before_comp_pair = frame_pair(subset["frame0"], final_master[0])
    trained_metrics, trained_pose, trained_logits = measured_metrics(
        final_before_comp_pair,
        posenet=posenet,
        segnet=segnet,
        original_argmax=original_argmax,
        original_pose6=original_pose6,
    )
    retained_fields.update(
        {
            "trained_pair": atomic_npy(
                smoke_payload_root / "trained_pair.float32.npy",
                final_before_comp_pair.detach().numpy(),
            ),
            "trained_pose": atomic_npy(smoke_payload_root / "trained_pose.float32.npy", trained_pose.numpy()),
            "trained_logits": atomic_npy(smoke_payload_root / "trained_logits.float32.npy", trained_logits.numpy()),
        }
    )

    source_parts, carrier_state = po1.load_carrier(DX2_ARCHIVE, DX2_RUNTIME)
    source_runtime_modules = _runtime_riders(DX2_RUNTIME)
    source_cap1, source_selector = source_runtime_modules.carrier_repack.split_frame0_selector_carrier(
        source_parts.carrier_blob
    )
    if source_selector != carrier_state.selector or len(source_cap1) < 50:
        raise RJ2Error("exact DX2 source predictor surface differs")
    source_predictor_metadata = source_cap1[14:50]
    if int(carrier_state.selector_choices[PAIR_ID]) != 0:
        raise RJ2Error("pair 0 is not on the exact identity-selector carrier surface")
    renderer_module = po1._load_renderer(DX2_RUNTIME)
    reconstructed_frame0 = carrier_frame0(
        carrier_state,
        carrier_state.codes[PAIR_ID],
        renderer_module=renderer_module,
    )
    source_frame0 = torch.from_numpy(np.ascontiguousarray(subset["frame0"])).permute(2, 0, 1).float()
    if not torch.equal(reconstructed_frame0, source_frame0):
        raise RJ2Error("carrier surface does not reproduce the retained DX2 frame 0")
    local_pose, jacobian, local_frame0 = carrier_pose_and_jacobian(
        carrier_state,
        carrier_state.codes[PAIR_ID],
        final_master[0],
        posenet=posenet,
        renderer_module=renderer_module,
    )
    update, solve_diagnostics = po1.solve_damped_least_squares(
        jacobian,
        np.asarray(subset["original_pose6"], dtype=np.float64) - local_pose,
        damping=0.05,
        max_code_step=4.0,
    )
    proposed_codes = po1.quantize_int12_update(carrier_state.codes[PAIR_ID], update).astype(np.int16, copy=False)
    proposed_frame0 = carrier_frame0(carrier_state, proposed_codes, renderer_module=renderer_module)
    proposed_pair = torch.stack((proposed_frame0, final_master[0]), dim=0).unsqueeze(0)
    proposed_metrics, proposed_pose, proposed_logits = measured_metrics(
        proposed_pair,
        posenet=posenet,
        segnet=segnet,
        original_argmax=original_argmax,
        original_pose6=original_pose6,
    )
    proposed_distortion_s = 100.0 * proposed_metrics["d_seg"] + math.sqrt(10.0 * proposed_metrics["d_pose"])
    trained_distortion_s = 100.0 * trained_metrics["d_seg"] + math.sqrt(10.0 * trained_metrics["d_pose"])
    if proposed_distortion_s <= trained_distortion_s:
        selected_codes = proposed_codes
        selected_frame0 = proposed_frame0
        selected_metrics = proposed_metrics
        selected_pose = proposed_pose
        selected_logits = proposed_logits
        compensation_disposition = "PROPOSED_INT12_STEP_ADMITTED"
    else:
        selected_codes = carrier_state.codes[PAIR_ID]
        selected_frame0 = local_frame0
        selected_metrics = trained_metrics
        selected_pose = trained_pose
        selected_logits = trained_logits
        compensation_disposition = "PROPOSED_INT12_STEP_REFUSED_BASE_RETAINED"
    final_codes = carrier_state.codes.astype(np.int32, copy=True)
    final_codes[PAIR_ID] = selected_codes
    final_pair = torch.stack((selected_frame0, final_master[0]), dim=0).unsqueeze(0)
    retained_fields.update(
        {
            "carrier_jacobian": atomic_npy(smoke_payload_root / "carrier_jacobian.float64.npy", jacobian),
            "carrier_update": atomic_npy(smoke_payload_root / "carrier_update.float64.npy", update),
            "carrier_codes_base": atomic_npy(smoke_payload_root / "carrier_codes_base.int16.npy", carrier_state.codes),
            "carrier_codes_proposed_pair": atomic_npy(
                smoke_payload_root / "carrier_codes_proposed_pair.int16.npy", proposed_codes
            ),
            "carrier_codes_final": atomic_npy(smoke_payload_root / "carrier_codes_final.int32.npy", final_codes),
            "proposed_frame0": atomic_npy(
                smoke_payload_root / "proposed_frame0.float32.npy",
                proposed_frame0.detach().numpy(),
            ),
            "proposed_pose": atomic_npy(smoke_payload_root / "proposed_pose.float32.npy", proposed_pose.numpy()),
            "proposed_logits": atomic_npy(smoke_payload_root / "proposed_logits.float32.npy", proposed_logits.numpy()),
            "final_pair": atomic_npy(
                smoke_payload_root / "final_pair.float32.npy",
                final_pair.detach().numpy(),
            ),
            "final_pose": atomic_npy(smoke_payload_root / "final_pose.float32.npy", selected_pose.numpy()),
            "final_logits": atomic_npy(smoke_payload_root / "final_logits.float32.npy", selected_logits.numpy()),
        }
    )

    container = rj1.source_container()
    shipped_stream = atomic_bytes(
        output / "retained/identity_control_q9_w16/shipped.carrier.stream",
        container["carrier"],
    )
    shipped_body = atomic_bytes(
        output / "retained/identity_control_q9_w16/shipped.carrier.rr5_body.bin",
        brotli.decompress(container["carrier"]),
    )
    identity_carrier = encode_carrier_stream(
        carrier_state,
        carrier_state.codes,
        runtime=DX2_RUNTIME,
        retention_root=output / "retained/identity_control_q9_w16",
        source_predictor_metadata=source_predictor_metadata,
    )
    if identity_carrier["stream"] != container["carrier"]:
        raise RJ2Error("production carrier encoder identity control failed")
    encoded_carrier = encode_carrier_stream(
        carrier_state,
        final_codes,
        runtime=DX2_RUNTIME,
        retention_root=output / "retained/final_object/carrier_encoder",
        source_predictor_metadata=source_predictor_metadata,
    )
    carrier_payloads = encoded_carrier["retained"]
    moved_container = dict(container)
    moved_container["carrier"] = encoded_carrier["stream"]
    semantic_stream, member, archive = rj1.build_archive(moved_container, packet)
    repeat = rj1.deterministic_zip(member)
    if archive != repeat:
        raise RJ2Error("primary and repeat archives differ")
    final_root = output / "retained/final_object"
    payloads = {
        "semantic_packet": semantic_record,
        "semantic_stream": atomic_bytes(final_root / "semantic.ck2.brotli", semantic_stream),
        "member": atomic_bytes(final_root / "p", member),
        "archive": atomic_bytes(final_root / "archive.zip", archive),
        "archive_repeat": atomic_bytes(final_root / "archive.repeat.zip", repeat),
        "carrier": carrier_payloads,
    }
    runtime_root = output / "candidate_runtime_rj2"
    runtime_copy = prepare_runtime_copy(DX2_RUNTIME, runtime_root)
    runtime_receipt = rj1.patch_runtime(DX2_RUNTIME, runtime_root, final_root / "archive.zip")
    semantic_parseback = rj1.parse_with_runtime(runtime_root, runtime_root / "archive.zip", packet)
    codes_record = retained_fields["carrier_codes_final"]
    receiver_parseback = fresh_process_parseback(
        runtime=runtime_root,
        archive=runtime_root / "archive.zip",
        expected_codes=Path(codes_record["path"]),
        semantic_sha256=sha256_bytes(packet),
        output=output / "retained/final_object/PARSEBACK_TRANSCRIPT.txt",
    )
    clean_trivial_runtime_residue(runtime_root)

    archive_bytes = len(archive)
    source_s = contest_arithmetic(
        d_seg=source_metrics["d_seg"],
        d_pose=source_metrics["d_pose"],
        archive_bytes=rj1.DX2_ARCHIVE_BYTES,
    )
    final_s = contest_arithmetic(
        d_seg=selected_metrics["d_seg"],
        d_pose=selected_metrics["d_pose"],
        archive_bytes=archive_bytes,
    )
    bytes_shed = rj1.DX2_ARCHIVE_BYTES - archive_bytes
    result = {
        "schema": "ddm_rj2_joint_renderer_smoke.v1",
        "status": "MECHANISM_COMPLETE_ENGINEERING_SMOKE",
        "verdict_scope": "CPU_N1_SINGLE_STEP_FLOAT32_W96_FILM; NO_FAMILY_OR_N600_VERDICT",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "scope_reductions": {
            "pair_count": 1,
            "pair_ids": [PAIR_ID],
            "steps": 1,
            "precision": "float32",
            "representation": "film_amortized_flat_w96 only",
            "carrier_solve": "one pair bounded int12 Gauss-Newton step; full n600 lattice re-encoded",
        },
        "mechanism_gates": {
            "joint_optimization_both_frozen_scorers": {
                "status": "PASS",
                "receipt": stage_end_record,
            },
            "mf1_boundary_margin_training_input": {
                "status": "PASS",
                "selected_cells": int(selected.sum()),
                "receipt": subset_records["selected_cells"],
            },
            "in_compile_compensation_final_packet_render": {
                "status": "PASS",
                "disposition": compensation_disposition,
                "changed_coordinates": int(np.count_nonzero(selected_codes != carrier_state.codes[PAIR_ID])),
                "receipt": retained_fields["carrier_jacobian"],
            },
            "carrier_resolved_and_reencoded_after_move": {
                "status": "PASS",
                "identity_control": True,
                "shipped_stream": shipped_stream,
                "shipped_body": shipped_body,
                "production_chain": "CAP1 -> DX2 -> RR5 -> Brotli-q9-lgwin16",
                "receipt": carrier_payloads,
            },
            "real_coders_real_payloads": {
                "status": "PASS",
                "receipt": payloads,
            },
            "receiver_parseback": {
                "status": "PASS",
                "semantic": semantic_parseback,
                "carrier": receiver_parseback,
                "runtime_copy": runtime_copy,
                "runtime": runtime_receipt,
            },
            "primary_repeat_archive": {
                "status": "PASS",
                "matching_hashes": payloads["archive"]["sha256"] == payloads["archive_repeat"]["sha256"],
                "primary": payloads["archive"],
                "repeat": payloads["archive_repeat"],
            },
        },
        "checkpoints": {
            "resume_from": file_record(resume_from),
            "periodic": file_record(output / "checkpoints/stage_10_joint_step_0001.pt"),
            "stage_end": stage_end_record,
            "ema_shadow_deployed": True,
            "distinct_stage_filenames": True,
            "atomic": True,
        },
        "measurements": {
            "source_dx2_pair": source_metrics,
            "untrained_rj1_pair": before_metrics,
            "trained_precompensation_pair": trained_metrics,
            "compensation_proposal_pair": proposed_metrics,
            "final_selected_pair": selected_metrics,
            "distortion_axis": AXIS,
            "n600_d_seg": "UNMEASURED",
            "n600_d_pose": "UNMEASURED",
            "contest_cpu_score": "UNMEASURED",
            "contest_cuda_score": "UNMEASURED",
            "source_scope_arithmetic": source_s,
            "final_scope_arithmetic": final_s,
            "scope_delta_s": final_s["s"] - source_s["s"],
        },
        "bytes": {
            "source_dx2_archive": rj1.DX2_ARCHIVE_BYTES,
            "candidate_archive": archive_bytes,
            "bytes_shed": bytes_shed,
            "rate_credit_s": bytes_shed * RATE_EXCHANGE_S_PER_BYTE,
            "current_distortion_currency": {
                "target_bytes": TARGET_BYTES,
                "source_demand_bytes": rj1.DX2_ARCHIVE_BYTES - TARGET_BYTES,
                "remaining_candidate_bytes_over_target": archive_bytes - TARGET_BYTES,
            },
            "zero_distortion_currency": {
                "source_shed_required_bytes": DX2_ZERO_DISTORTION_SHED_BYTES,
                "candidate_bytes_shed": bytes_shed,
                "byte_requirement_met": bytes_shed >= DX2_ZERO_DISTORTION_SHED_BYTES,
                "distortion_is_zero": False,
            },
        },
        "object_repricing": object_repricing_rows(),
        "compensation": {
            "pair": PAIR_ID,
            "disposition": compensation_disposition,
            "solve_diagnostics": solve_diagnostics,
            "base_codes": carrier_state.codes[PAIR_ID].astype(int).tolist(),
            "proposed_codes": proposed_codes.astype(int).tolist(),
            "selected_codes": np.asarray(selected_codes).astype(int).tolist(),
        },
        "history": history,
        "retained_fields": retained_fields,
        "payloads": payloads,
        "storage": storage,
        "sources": sources,
        "provenance": provenance,
        "all_payloads_retained": True,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "full_n600_scorer_invocations": 0,
    }
    atomic_json(output / "SMOKE_RESULT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    commands = argument_parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    run = commands.add_parser("smoke")
    run.add_argument("--resume-from", type=Path, required=True)
    run.add_argument("--max-steps", type=int, default=1)
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = prepare(args.output) if args.command == "prepare" else smoke(args.output, args.resume_from, args.max_steps)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
