#!/usr/bin/env python3
"""ddm_js1 Stage 0: retained n600 per-edge decomposition on the resealed objects.

This runner compares the exact CP135 base archive with the T1R1 C1-composed
archive on one matched local CPU receiver/scorer axis.  The pointer F26 entrypoint
is CUDA-locked, so this diagnostic imports its custodied parser, HPAC decoder, and
renderer directly and changes only the execution device to CPU.  It fails closed
unless that local surface reproduces the terminal CP135 Seg row; a mismatched
surface is retained as a blocker diagnostic, never admitted as Stage-0 rho.
Every decoded token field, renderer state, raw frame, SegNet logit field, and
argmax field is retained.

The job is restartable at three durable boundaries:

* token decode for each candidate;
* renderer chunks and selector chunks for each candidate;
* scorer chunks of at most 120 pairs for each candidate.

Only ``summarize`` adjudicates the per-edge result.  It refuses incomplete n600
payloads and emits both directed GT->rendered cells and undirected interfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge"
)
UPSTREAM = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
GT_ARGMAX = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy"
)
C1_TOKENS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/"
    "c1_solved_tokens_n600.u8"
)

N = 600
SEG_H, SEG_W = 384, 512
CAM_H, CAM_W = 874, 1164
SEG_PX = SEG_H * SEG_W
TOTAL_PX = N * SEG_PX
RAW_BYTES = N * 2 * CAM_H * CAM_W * 3
CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
AXIS = "[macOS-CPU frozen-SegNet advisory, n600] NON-PROMOTABLE"
RHO_REQUIRED = 0.827795
TERMINAL_BASE_D_SEG = 0.00029639352578669786
TERMINAL_BASE_FLIPS = 34_964
C1_BATCH16_REFERENCE_FLIPS = 17_926
# ddm_js1b_cuda_custody_adjudication_20260813: measured T4 residual 6 flips
# (batch-shape/tie-break class, et4); tolerance bounds that class, never a
# renderer-drift magnitude (local drift measured +15,425).
CP135_BATCH_SHAPE_TOLERANCE_FLIPS = 20
C1_TARGET_CUSTODY_SHA256 = (
    "a9c4936c41bc6634477f9c060be3d170542bd2a1d4d0cd04d5afcd0912fb3908"
)
MIN_RESERVE_BYTES = 32 * 1024**3
EXPECTED_NEW_BYTES = 2 * RAW_BYTES + 2 * RAW_BYTES + 2 * (N * 5 * SEG_PX * 4)
EXPECTED_NEW_BYTES += 5 * (N * SEG_PX) + 2 * 1024**3


@dataclass(frozen=True)
class Candidate:
    name: str
    archive: Path
    runtime: Path
    archive_bytes: int
    archive_sha256: str
    decoded_spatial: Path
    decoded_spatial_sha256: str


CANDIDATES = {
    "cp135_base": Candidate(
        name="cp135_base",
        archive=Path(
            "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip"
        ),
        runtime=Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime"),
        archive_bytes=186_252,
        archive_sha256="6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6",
        decoded_spatial=Path(
            "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/coders/"
            "hp3_step2/decoded_spatial_tokens.fresh_rc64.bin"
        ),
        decoded_spatial_sha256=(
            "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
        ),
    ),
    "t1r1_c1_composed": Candidate(
        name="t1r1_c1_composed",
        archive=Path(
            "/Volumes/APDataStore/pact/ddm_t1r1/retained/adapted_runtime/archive.zip"
        ),
        runtime=Path("/Volumes/APDataStore/pact/ddm_t1r1/retained/adapted_runtime"),
        archive_bytes=187_046,
        archive_sha256="12a5b181fef4e15ad8a752161c744347beca0b5a1224c5d3d542ab148f6ece80",
        decoded_spatial=Path(
            "/Volumes/APDataStore/pact/ddm_t1r1/retained/receiver_state/"
            "decoded_spatial_tokens.shipped_rc64.bin"
        ),
        decoded_spatial_sha256=(
            "2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5"
        ),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(payload), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def require_file(path: Path, *, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file():
        raise RuntimeError(f"required file is unavailable: {path}")
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {path}")
    if digest is not None and sha256_file(path) != digest:
        raise RuntimeError(f"SHA-256 mismatch for {path}")


def require_receipt_binding(record: dict[str, Any], path: Path) -> None:
    """Require a retained path and size to match its already-computed content receipt."""
    if Path(record["path"]).resolve() != path.resolve():
        raise RuntimeError(f"receipt path mismatch for {path}")
    require_file(path, size=int(record["bytes"]))
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"receipt has no SHA-256 binding for {path}")


def tree_record(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name.startswith("._"):
            continue
        relative = path.relative_to(root).as_posix()
        record = file_record(path)
        rows.append({"relative_path": relative, **record})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(record["sha256"].encode())
        aggregate.update(b"\0")
        aggregate.update(str(record["bytes"]).encode())
        aggregate.update(b"\n")
    return {
        "root": str(root.resolve()),
        "file_count": len(rows),
        "tree_sha256": aggregate.hexdigest(),
        "files": rows,
    }


def candidate_root(output: Path, candidate: Candidate) -> Path:
    return output / "candidates" / candidate.name


def validate_candidate(candidate: Candidate) -> None:
    require_file(
        candidate.archive,
        size=candidate.archive_bytes,
        digest=candidate.archive_sha256,
    )
    require_file(
        candidate.decoded_spatial,
        size=TOTAL_PX,
        digest=candidate.decoded_spatial_sha256,
    )
    for relative in (
        "runtime/f26_inflate.py",
        "runtime/residual_archive.py",
        "runtime/entropy/rc64_backend.c",
        "cpr1/inflate.py",
    ):
        require_file(candidate.runtime / relative)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    consumer = Path("/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810").resolve()
    if not output.is_relative_to(consumer):
        raise RuntimeError("Stage-0 output must remain inside the resealed consumer store")
    for candidate in CANDIDATES.values():
        validate_candidate(candidate)
    require_file(
        GT_ARGMAX,
        size=117_964_928,
        digest="b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d",
    )
    require_file(
        C1_TOKENS,
        size=117_964_800,
        digest="2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5",
    )
    require_file(UPSTREAM / "modules.py")
    require_file(UPSTREAM / "models/segnet.safetensors")
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    required = EXPECTED_NEW_BYTES + MIN_RESERVE_BYTES
    if usage.free < required:
        raise RuntimeError(
            f"storage preflight failed: free={usage.free} required={required}"
        )

    custody = output / "custody"
    custody.mkdir(parents=True, exist_ok=True)
    gt_copy = custody / "gt_argmax_n600.npy"
    if not gt_copy.exists():
        shutil.copyfile(GT_ARGMAX, gt_copy)
    require_file(gt_copy, size=GT_ARGMAX.stat().st_size, digest=sha256_file(GT_ARGMAX))
    c1_copy = custody / "c1_target_argmax_n600.npy"
    if not c1_copy.exists():
        source = np.memmap(C1_TOKENS, mode="r", dtype=np.uint8, shape=(N, SEG_H, SEG_W))
        atomic_npy(c1_copy, source)
    c1 = np.load(c1_copy, mmap_mode="r", allow_pickle=False)
    if c1.shape != (N, SEG_H, SEG_W) or c1.dtype != np.uint8:
        raise RuntimeError("retained C1 target has invalid geometry")

    result = {
        "schema": "ddm_js1_stage0_preflight.v1",
        "status": "PASS",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "git_head_at_preflight": git_head(),
        "storage": {
            "output": str(output),
            "free_bytes": usage.free,
            "expected_new_bytes_upper_bound": EXPECTED_NEW_BYTES,
            "reserve_bytes": MIN_RESERVE_BYTES,
            "required_bytes": required,
        },
        "candidates": {
            name: {
                "archive": file_record(value.archive),
                "retained_receiver_decoded_spatial": file_record(value.decoded_spatial),
                "runtime_tree": tree_record(value.runtime),
            }
            for name, value in CANDIDATES.items()
        },
        "gt_argmax": file_record(gt_copy),
        "c1_target_argmax": file_record(c1_copy),
        "scorer": {
            "modules": file_record(UPSTREAM / "modules.py"),
            "weights": file_record(UPSTREAM / "models/segnet.safetensors"),
        },
        "m37_same_parent_contract": {
            "cp135_base": CANDIDATES["cp135_base"].archive_sha256,
            "t1r1_c1_composed": CANDIDATES["t1r1_c1_composed"].archive_sha256,
            "target_plane": sha256_file(C1_TOKENS),
            "status": "BOUND_BY_CONTENT",
        },
    }
    atomic_json(output / "00_PREFLIGHT.json", result)
    return result


def compile_rc64(runtime: Path, root: Path) -> Path:
    source = runtime / "runtime/entropy/rc64_backend.c"
    target = root / "native/libcpr1_rc64.dylib"
    receipt_path = root / "native/COMPILE_RECEIPT.json"
    if target.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt["source"] == file_record(source) and receipt["library"] == file_record(target):
            return target
        raise RuntimeError("existing RC64 library is not bound to the current source")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    command = [
        shutil.which("cc") or "cc",
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        str(source),
        "-o",
        str(temporary),
    ]
    subprocess.run(command, check=True, cwd=REPO)
    os.replace(temporary, target)
    atomic_json(
        receipt_path,
        {
            "schema": "ddm_js1_rc64_compile.v1",
            "argv": command,
            "source": file_record(source),
            "library": file_record(target),
        },
    )
    return target


def load_runtime(candidate: Candidate):
    for name in tuple(sys.modules):
        if name == "runtime" or name.startswith("runtime.") or name == "_f26_renderer":
            del sys.modules[name]
    sys.path.insert(0, str(candidate.runtime))
    try:
        f26 = importlib.import_module("runtime.f26_inflate")
        residual = importlib.import_module("runtime.residual_archive")
        carrier = importlib.import_module("runtime.carrier_repack")
        weights = importlib.import_module("runtime.entropy.renderer_weight_codec")
        renderer = f26._load_renderer(candidate.runtime / "cpr1")
    except Exception:
        sys.path.pop(0)
        raise
    return f26, residual, carrier, weights, renderer


def release_runtime() -> None:
    if sys.path and any(sys.path[0] == str(value.runtime) for value in CANDIDATES.values()):
        sys.path.pop(0)


def parse_receiver_state(candidate: Candidate, root: Path):
    import torch

    f26, residual, carrier, weights, renderer = load_runtime(candidate)
    parts = residual.read_residual_archive(candidate.archive)
    carrier_blob, selector_blob = carrier.split_frame0_selector_carrier(parts.carrier_blob)
    canonical_carrier = carrier.materialize_cpr1(carrier_blob, renderer)
    semantic_pose = (
        struct.pack("<II", 40_252, len(canonical_carrier))
        + bytes(40_252)
        + canonical_carrier
    )
    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
    records = weights.decode_wans1(parts.semantic_blob)
    semantic = renderer.SemanticTokenRenderer(96)
    semantic.load_state_dict(
        {
            record.schema.name: torch.from_numpy(
                np.ascontiguousarray(record.values, dtype=np.float32)
            )
            for record in records
        },
        strict=True,
    )
    state_root = root / "receiver_state"
    state_root.mkdir(parents=True, exist_ok=True)
    atomic_bytes(state_root / "semantic_pose_adapter_input.bin", semantic_pose)
    atomic_npy(state_root / "pose_basis.float32.npy", basis.detach().cpu().numpy())
    atomic_npy(
        state_root / "pose_coefficients.float32.npy",
        coefficients.detach().cpu().numpy(),
    )
    for index, record in enumerate(records):
        atomic_npy(
            state_root / "semantic" / f"{index:02d}_{record.schema.name}.float32.npy",
            np.ascontiguousarray(record.values, dtype=np.float32),
        )
    return f26, residual, carrier, renderer, parts, selector_blob, semantic, basis, coefficients


def decode_tokens(candidate: Candidate, root: Path) -> dict[str, Any]:
    receipt_path = root / "10_DECODE.json"
    token_path = root / "retained/decoded_tokens_n600.npy"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt["archive"] != file_record(candidate.archive):
            raise RuntimeError("decode checkpoint belongs to another archive")
        if receipt["tokens"] != file_record(token_path):
            raise RuntimeError("decode checkpoint token payload changed")
        return receipt
    source = np.memmap(
        candidate.decoded_spatial,
        mode="r",
        dtype=np.uint8,
        shape=(N, SEG_H, SEG_W),
    )
    started = time.time()
    if token_path.is_file():
        token_array = np.load(token_path, mmap_mode="r", allow_pickle=False)
        if (
            token_array.shape != (N, SEG_H, SEG_W)
            or token_array.dtype != np.uint8
            or hashlib.sha256(token_array.tobytes()).hexdigest()
            != candidate.decoded_spatial_sha256
        ):
            raise RuntimeError("orphan decoded-token checkpoint is not adoptable")
    else:
        atomic_npy(token_path, source)
    receipt = {
        "schema": "ddm_js1_stage0_retained_decode_adoption.v1",
        "axis": AXIS,
        "decode_rerun": False,
        "source_parseback": file_record(candidate.decoded_spatial),
        "source_parseback_raw_sha256": candidate.decoded_spatial_sha256,
        "adoption_reason": (
            "the archive's full-n600 shipped-receiver parse-back is already retained; "
            "reuse avoids re-decoding a settled payload"
        ),
        "contest_runtime_identity_claim": False,
        "archive": file_record(candidate.archive),
        "tokens": file_record(token_path),
        "tokens_raw_sha256": hashlib.sha256(
            np.load(token_path, mmap_mode="r", allow_pickle=False).tobytes()
        ).hexdigest(),
        "wall_seconds": time.time() - started,
    }
    if receipt["tokens_raw_sha256"] != candidate.decoded_spatial_sha256:
        raise RuntimeError("retained token payload differs from decoder digest")
    atomic_json(receipt_path, receipt)
    return receipt


def render_preselector(candidate: Candidate, root: Path) -> None:
    import torch
    from torch.nn import functional

    progress_path = root / "20_RENDER_PROGRESS.json"
    raw_path = root / "retained/0.preselector.raw"
    tokens = np.load(root / "retained/decoded_tokens_n600.npy", mmap_mode="r", allow_pickle=False)
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress["archive_sha256"] != candidate.archive_sha256:
            raise RuntimeError("render progress belongs to another archive")
        if not raw_path.is_file() or raw_path.stat().st_size != RAW_BYTES:
            raise RuntimeError("render progress has no complete-sized raw payload")
        raw = np.memmap(raw_path, mode="r+", dtype=np.uint8, shape=(N * 2, CAM_H, CAM_W, 3))
    elif raw_path.exists():
        raise RuntimeError("orphan preselector raw exists without a progress checkpoint")
    else:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.memmap(raw_path, mode="w+", dtype=np.uint8, shape=(N * 2, CAM_H, CAM_W, 3))
        progress = {
            "schema": "ddm_js1_stage0_render_progress.v1",
            "archive_sha256": candidate.archive_sha256,
            "masters_next": 0,
            "carriers_next": 0,
            "complete": False,
        }
        atomic_json(progress_path, progress)

    try:
        (
            _f26,
            _residual,
            _carrier,
            renderer,
            _parts,
            _selector,
            semantic,
            basis,
            coefficients,
        ) = parse_receiver_state(candidate, root)
        device = torch.device("cpu")
        semantic = semantic.eval().to(device)
        normalized = renderer.normalized_basis(basis.to(device))
        coefficients = coefficients.to(device)
        with torch.inference_mode():
            for index in range(int(progress["masters_next"]), N):
                frame_index = torch.tensor([index], dtype=torch.long, device=device)
                frame_tokens = torch.from_numpy(np.asarray(tokens[index]).copy())[None].long()
                master = (
                    functional.interpolate(
                        semantic(frame_tokens, frame_index),
                        size=(CAM_H, CAM_W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .clamp(0.0, 255.0)
                    .round()
                )
                raw[2 * index + 1] = master[0].to(torch.uint8).permute(1, 2, 0).numpy()
                if (index + 1) % 24 == 0 or index == N - 1:
                    raw.flush()
                    progress["masters_next"] = index + 1
                    atomic_json(progress_path, progress)
                    print(f"[{candidate.name}] masters {index + 1}/{N}", flush=True)
            for index in range(int(progress["carriers_next"]), N):
                code = coefficients[index : index + 1]
                field = torch.einsum("bk,kchw->bchw", code, normalized)
                field = field / math.sqrt(int(renderer.CARRIER_DIM))
                slave = (
                    functional.interpolate(
                        (127.5 + float(renderer.CARRIER_AMPLITUDE) * field)
                        .clamp(0.0, 255.0)
                        .round(),
                        size=(CAM_H, CAM_W),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .clamp(0.0, 255.0)
                    .round()
                )
                raw[2 * index] = slave[0].to(torch.uint8).permute(1, 2, 0).numpy()
                if (index + 1) % 24 == 0 or index == N - 1:
                    raw.flush()
                    progress["carriers_next"] = index + 1
                    atomic_json(progress_path, progress)
                    print(f"[{candidate.name}] carriers {index + 1}/{N}", flush=True)
        progress["complete"] = True
        raw.flush()
        atomic_json(progress_path, progress)
    finally:
        release_runtime()


def apply_selector(candidate: Candidate, root: Path) -> dict[str, Any]:
    progress_path = root / "30_SELECTOR_PROGRESS.json"
    source_path = root / "retained/0.preselector.raw"
    target_path = root / "retained/0.raw"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress["archive_sha256"] != candidate.archive_sha256:
            raise RuntimeError("selector progress belongs to another archive")
        if not target_path.is_file() or target_path.stat().st_size != RAW_BYTES:
            raise RuntimeError("selector progress has no complete-sized target payload")
        target = np.memmap(target_path, mode="r+", dtype=np.uint8, shape=(N * 2, CAM_H, CAM_W, 3))
    elif target_path.exists():
        raise RuntimeError("orphan selected raw exists without progress")
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target = np.memmap(target_path, mode="w+", dtype=np.uint8, shape=(N * 2, CAM_H, CAM_W, 3))
        progress = {
            "schema": "ddm_js1_stage0_selector_progress.v1",
            "archive_sha256": candidate.archive_sha256,
            "next_pair": 0,
            "complete": False,
        }
        atomic_json(progress_path, progress)
    source = np.memmap(source_path, mode="r", dtype=np.uint8, shape=(N * 2, CAM_H, CAM_W, 3))
    try:
        f26, _residual, _carrier, renderer, parts, selector_blob, *_ = parse_receiver_state(
            candidate, root
        )
        if selector_blob is None:
            modes = ()
            indices = np.zeros(N, dtype=np.uint8)
        else:
            modes, indices = f26.decode_selector(selector_blob)
            if indices.size != N:
                raise RuntimeError("selector has the wrong frame count")
        for start in range(int(progress["next_pair"]), N, 24):
            end = min(start + 24, N)
            target[2 * start : 2 * end] = source[2 * start : 2 * end]
            if selector_blob is not None:
                for mode_index, mode in enumerate(modes):
                    local = np.flatnonzero(indices[start:end] == mode_index) + start
                    if local.size:
                        target[2 * local] = f26.apply_pixel_mode(
                            np.asarray(target[2 * local]).copy(), mode
                        )
            target.flush()
            progress["next_pair"] = end
            atomic_json(progress_path, progress)
            print(f"[{candidate.name}] selector {end}/{N}", flush=True)
        progress["complete"] = True
        atomic_json(progress_path, progress)
    finally:
        release_runtime()
    return {
        "selector_present": selector_blob is not None,
        "selector_mode_count": len(modes),
        "selector_payload_bytes": 0 if selector_blob is None else len(selector_blob),
    }


def inflate(args: argparse.Namespace) -> dict[str, Any]:
    candidate = CANDIDATES[args.candidate]
    validate_candidate(candidate)
    preflight_receipt = args.output / "00_PREFLIGHT.json"
    if not preflight_receipt.is_file() or json.loads(preflight_receipt.read_text())["status"] != "PASS":
        raise RuntimeError("run preflight before inflation")
    root = candidate_root(args.output, candidate)
    root.mkdir(parents=True, exist_ok=True)
    archive_copy = root / "retained/archive.zip"
    archive_copy.parent.mkdir(parents=True, exist_ok=True)
    if not archive_copy.exists():
        shutil.copyfile(candidate.archive, archive_copy)
    require_file(archive_copy, size=candidate.archive_bytes, digest=candidate.archive_sha256)
    extracted = root / "retained/archive/p"
    if not extracted.exists():
        with zipfile.ZipFile(candidate.archive) as archive:
            names = archive.namelist()
            if names != ["p"]:
                raise RuntimeError(f"archive members differ: {names}")
            atomic_bytes(extracted, archive.read("p"))
    decode = decode_tokens(candidate, root)
    render_preselector(candidate, root)
    selector = apply_selector(candidate, root)
    raw_path = root / "retained/0.raw"
    receipt = {
        "schema": "ddm_js1_stage0_inflate.v1",
        "complete": True,
        "axis": AXIS,
        "archive": file_record(archive_copy),
        "extracted_payload": file_record(extracted),
        "decode_receipt": file_record(root / "10_DECODE.json"),
        "tokens": decode["tokens"],
        "preselector_raw": file_record(root / "retained/0.preselector.raw"),
        "selected_raw": file_record(raw_path),
        "selector": selector,
        "resumability": {
            "decode_checkpoint": str((root / "10_DECODE.json").resolve()),
            "render_checkpoint": str((root / "20_RENDER_PROGRESS.json").resolve()),
            "selector_checkpoint": str((root / "30_SELECTOR_PROGRESS.json").resolve()),
        },
    }
    if receipt["selected_raw"]["bytes"] != RAW_BYTES:
        raise RuntimeError("selected raw has the wrong byte count")
    atomic_json(root / "40_INFLATE_RESULT.json", receipt)
    return receipt


def load_segnet():
    import torch
    from safetensors.torch import load_file

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import modules

    if Path(modules.__file__).resolve() != (UPSTREAM / "modules.py").resolve():
        raise RuntimeError("imported a non-custodied modules.py")
    torch.set_num_threads(4)
    model = modules.SegNet().eval().cpu()
    model.load_state_dict(
        load_file(str(UPSTREAM / "models/segnet.safetensors"), device="cpu")
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def score(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.chunk > 120 or args.chunk <= 0:
        raise RuntimeError("scorer chunk must be in [1, 120]")
    candidate = CANDIDATES[args.candidate]
    root = candidate_root(args.output, candidate)
    inflate_receipt = root / "40_INFLATE_RESULT.json"
    if not inflate_receipt.is_file():
        raise RuntimeError("inflate candidate before scoring")
    raw_path = root / "retained/0.raw"
    raw = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(N * 2, CAM_H, CAM_W, 3))
    scorer_root = root / "scorer"
    scorer_root.mkdir(parents=True, exist_ok=True)
    progress_path = scorer_root / "SCORER_PROGRESS.json"
    argmax_path = scorer_root / "argmax_n600.npy"
    logits_path = scorer_root / "logits_n600.float32.npy"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress["archive_sha256"] != candidate.archive_sha256:
            raise RuntimeError("scorer progress belongs to another archive")
        if int(progress["chunk"]) != args.chunk:
            raise RuntimeError("scorer resume chunk differs from the checkpoint")
        saved_batch = int(progress.get("inner_batch", 1))
        if saved_batch != args.batch:
            raise RuntimeError("scorer resume batch differs from the checkpoint")
        progress["inner_batch"] = saved_batch
        argmax = np.lib.format.open_memmap(argmax_path, mode="r+")
        logits = np.lib.format.open_memmap(logits_path, mode="r+")
    elif argmax_path.exists() or logits_path.exists():
        raise RuntimeError("orphan scorer payload exists without progress")
    else:
        argmax = np.lib.format.open_memmap(
            argmax_path, mode="w+", dtype=np.uint8, shape=(N, SEG_H, SEG_W)
        )
        logits = np.lib.format.open_memmap(
            logits_path, mode="w+", dtype=np.float32, shape=(N, 5, SEG_H, SEG_W)
        )
        progress = {
            "schema": "ddm_js1_stage0_scorer_progress.v1",
            "archive_sha256": candidate.archive_sha256,
            "next_pair": 0,
            "chunk": args.chunk,
            "inner_batch": args.batch,
            "complete": False,
        }
        atomic_json(progress_path, progress)
    if argmax.shape != (N, SEG_H, SEG_W) or logits.shape != (N, 5, SEG_H, SEG_W):
        raise RuntimeError("scorer payload geometry mismatch")
    model = load_segnet()
    started = time.time()
    with torch.inference_mode():
        for start in range(int(progress["next_pair"]), N, args.chunk):
            end = min(start + args.chunk, N)
            for mini in range(start, end, args.batch):
                mini_end = min(mini + args.batch, end)
                frames = np.asarray(raw[2 * np.arange(mini, mini_end) + 1]).copy()
                value = torch.from_numpy(frames).permute(0, 3, 1, 2).float()[:, None]
                output = model(model.preprocess_input(value))
                output_np = output.cpu().numpy().astype(np.float32, copy=False)
                logits[mini:mini_end] = output_np
                argmax[mini:mini_end] = output_np.argmax(axis=1).astype(np.uint8)
            logits.flush()
            argmax.flush()
            progress["next_pair"] = end
            atomic_json(progress_path, progress)
            print(
                f"[{candidate.name}] scorer {end}/{N} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    progress["complete"] = True
    atomic_json(progress_path, progress)
    result = {
        "schema": "ddm_js1_stage0_scorer.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "archive": file_record(candidate.archive),
        "raw": file_record(raw_path),
        "argmax": file_record(argmax_path),
        "logits": file_record(logits_path),
        "scorer": {
            "modules": file_record(UPSTREAM / "modules.py"),
            "weights": file_record(UPSTREAM / "models/segnet.safetensors"),
            "chunk_pairs": args.chunk,
            "inner_batch_pairs": args.batch,
        },
        "wall_seconds_this_invocation": time.time() - started,
        "checkpoint": file_record(progress_path),
    }
    atomic_json(scorer_root / "SCORER_RESULT.json", result)
    return result


def confusion(gt: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    joint = gt.reshape(-1).astype(np.int64) * len(CLASSES)
    joint += predicted.reshape(-1).astype(np.int64)
    return np.bincount(joint, minlength=25).reshape(5, 5).astype(np.int64)


def matrix_summary(matrix: np.ndarray, *, total_pixels: int = TOTAL_PX) -> dict[str, Any]:
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    total_flips = int(off.sum())
    directed = []
    for gt_class in range(5):
        for rendered_class in range(5):
            if gt_class == rendered_class:
                continue
            flips = int(off[gt_class, rendered_class])
            directed.append(
                {
                    "cell": f"{CLASSES[gt_class]}->{CLASSES[rendered_class]}",
                    "gt_class": CLASSES[gt_class],
                    "rendered_class": CLASSES[rendered_class],
                    "flips": flips,
                    "share_of_all_flips": flips / total_flips if total_flips else 0.0,
                }
            )
    directed.sort(key=lambda row: (-row["flips"], row["cell"]))
    edges = []
    for left in range(5):
        for right in range(left + 1, 5):
            forward = int(off[left, right])
            backward = int(off[right, left])
            total = forward + backward
            edges.append(
                {
                    "edge": f"{CLASSES[left]}<->{CLASSES[right]}",
                    f"{CLASSES[left]}->{CLASSES[right]}": forward,
                    f"{CLASSES[right]}->{CLASSES[left]}": backward,
                    "flips": total,
                    "share_of_all_flips": total / total_flips if total_flips else 0.0,
                    "asymmetry_ratio": (
                        max(forward, backward) / min(forward, backward)
                        if min(forward, backward)
                        else None
                    ),
                }
            )
    edges.sort(key=lambda row: (-row["flips"], row["edge"]))
    road_incident = sum(
        row["flips"] for row in edges if row["edge"].startswith("Road<->")
    )
    return {
        "confusion_matrix_gt_by_rendered": matrix.tolist(),
        "total_flips": total_flips,
        "d_seg": total_flips / total_pixels,
        "seg_score_contribution": 100.0 * total_flips / total_pixels,
        "directed_cells": directed,
        "undirected_edges": edges,
        "road_incident_flips": road_incident,
        "road_incident_share": road_incident / total_flips if total_flips else 0.0,
    }


def decomposition(field: np.ndarray, gt: np.ndarray, per_pair_path: Path) -> dict[str, Any]:
    total = np.zeros((5, 5), dtype=np.int64)
    rows = []
    for pair in range(N):
        matrix = confusion(np.asarray(gt[pair]), np.asarray(field[pair]))
        total += matrix
        row = {"pair": pair, **matrix_summary(matrix, total_pixels=SEG_PX)}
        rows.append(row)
    atomic_bytes(
        per_pair_path,
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows),
    )
    summary = matrix_summary(total)
    summary["per_pair_jsonl"] = file_record(per_pair_path)
    return summary


def edge_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["edge"]: row for row in summary["undirected_edges"]}


def adjudicate_axis(base_flips: int, c1_flips: int) -> dict[str, Any]:
    """Admit rho only when the diagnostic reproduces the resealed reference axis."""
    base_matches = base_flips == TERMINAL_BASE_FLIPS
    target_matches = c1_flips == C1_BATCH16_REFERENCE_FLIPS
    admitted = base_matches and target_matches
    return {
        "status": "ADMITTED" if admitted else "BLOCKED_AXIS_MISMATCH",
        "admitted_for_stage0_rho": admitted,
        "terminal_base": {
            "expected_flips": TERMINAL_BASE_FLIPS,
            "expected_d_seg": TERMINAL_BASE_D_SEG,
            "observed_flips": base_flips,
            "matches": base_matches,
        },
        "c1_batch16_reference": {
            "expected_flips": C1_BATCH16_REFERENCE_FLIPS,
            "observed_flips": c1_flips,
            "matches": target_matches,
        },
    }


def require_downloaded_record(record: dict[str, Any], path: Path) -> None:
    """Bind a downloaded Modal payload by content, independent of its remote path."""
    require_file(path, size=int(record["bytes"]), digest=str(record["sha256"]))


def load_cuda_argmax_bundle(root: Path) -> dict[str, Any]:
    """Load one admitted JS1B T4 field bundle without running a local scorer."""
    root = root.resolve()
    receipt_candidates = [
        path
        for path in (root / "FINAL_RESULT.json", root / "JS1B_FINAL_RESULT.json")
        if path.is_file()
    ]
    if len(receipt_candidates) != 1:
        raise RuntimeError(
            "--from-argmax-fields requires exactly one FINAL_RESULT.json or "
            f"JS1B_FINAL_RESULT.json under {root}"
        )
    receipt_path = receipt_candidates[0]
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema")
        != "ddm_js1b_cuda_argmax_field_materializer_result.v1"
        or receipt.get("execution_status") != "COMPLETE"
        or receipt.get("axis")
        != "[contest-CUDA T4 frozen-SegNet argmax fields, n600, batch=16] COMPONENT-ONLY"
        or int(receipt.get("batch_size", -1)) != 16
        or bool(receipt.get("score_claim"))
        or bool(receipt.get("promotion_eligible"))
    ):
        raise RuntimeError(f"JS1B field receipt contract differs: {receipt_path}")
    # Adjudicated admission (ddm_js1b_cuda_custody_adjudication_20260813):
    # the worker's literal bars were measured on the LOCAL Mac GT instrument;
    # on the T4 instrument the honest contract is (a) CP135-vs-GT within the
    # et4 batch-shape tolerance of the promoted row and (b) the stored C1
    # target field byte-identical to js1 custody. The 17,926 C1 bar was a
    # local-instrument number (T4-instrument value: 27,330) — never a decode
    # control; do not re-impose it here.
    adjudication = receipt.get("axis_adjudication", {})
    cp135_observed = int(
        adjudication.get("cp135_control", {}).get("observed_flips", -1)
    )
    if abs(cp135_observed - TERMINAL_BASE_FLIPS) > CP135_BATCH_SHAPE_TOLERANCE_FLIPS:
        raise RuntimeError(
            f"JS1B CP135 control {cp135_observed} outside the batch-shape "
            f"tolerance ±{CP135_BATCH_SHAPE_TOLERANCE_FLIPS} of the promoted "
            f"{TERMINAL_BASE_FLIPS}; stop and treat as a field/custody question"
        )
    c1_field_path = root / "retained/fields/c1_target_argmax_n600.npy"
    c1_sha = hashlib.sha256(c1_field_path.read_bytes()).hexdigest()
    if c1_sha != C1_TARGET_CUSTODY_SHA256:
        raise RuntimeError(
            f"JS1B c1_target field sha {c1_sha} differs from js1 custody "
            f"{C1_TARGET_CUSTODY_SHA256}; stop and treat as a custody question"
        )
    field_paths = {
        name: root / f"retained/fields/{name}_argmax_n600.npy"
        for name in ("gt", "cp135_base", "t1r1_c1_composed", "c1_target")
    }
    fields = {}
    for name, path in field_paths.items():
        record = receipt.get("fields", {}).get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"JS1B receipt has no {name} field record")
        require_downloaded_record(record, path)
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (N, SEG_H, SEG_W) or value.dtype != np.uint8:
            raise RuntimeError(f"JS1B field shape/dtype differs: {path}")
        fields[name] = value
    actual_flips = {
        name: int(np.count_nonzero(value != fields["gt"]))
        for name, value in fields.items()
        if name != "gt"
    }
    # Adjudicated bars (ddm_js1b_cuda_custody_adjudication_20260813): CP135
    # within the et4 batch-shape tolerance of the promoted row; the C1 target
    # field is custody-checked by sha above — its flips-vs-GT on the T4
    # instrument (27,330) REPLACE the local-instrument 17,926 as reference.
    if (
        abs(actual_flips["cp135_base"] - TERMINAL_BASE_FLIPS)
        > CP135_BATCH_SHAPE_TOLERANCE_FLIPS
    ):
        raise RuntimeError(
            f"downloaded JS1B cp135 field flips {actual_flips['cp135_base']} "
            f"outside ±{CP135_BATCH_SHAPE_TOLERANCE_FLIPS} of "
            f"{TERMINAL_BASE_FLIPS}; stop and treat as a field/custody question"
        )
    receipt_flips = adjudication.get("flips_vs_gt")
    if receipt_flips != actual_flips:
        raise RuntimeError("downloaded JS1B field counts differ from the remote receipt")
    return {
        "root": root,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "field_paths": field_paths,
        "fields": fields,
        "flips_vs_gt": actual_flips,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    cuda_bundle = None
    scorer_receipts: dict[str, dict[str, Any]] = {}
    if args.from_argmax_fields is not None:
        cuda_bundle = load_cuda_argmax_bundle(args.from_argmax_fields)
        fields = dict(cuda_bundle["fields"])
        gt = fields.pop("gt")
        c1 = fields["c1_target"]
        gt_path = cuda_bundle["field_paths"]["gt"]
        c1_path = cuda_bundle["field_paths"]["c1_target"]
        axis = str(cuda_bundle["receipt"]["axis"])
        for name in CANDIDATES:
            scorer_receipts[name] = cuda_bundle["receipt"]["scorers"][name]
        object_receipts = {
            name: {
                "archive": cuda_bundle["receipt"]["receivers"][name]["archive"],
                "raw": cuda_bundle["receipt"]["receivers"][name]["raw"],
                "argmax": file_record(cuda_bundle["field_paths"][name]),
                "scorer_receipt": scorer_receipts[name],
                "retained_scorer_payloads": {
                    "seg_inputs": True,
                    "logits": True,
                    "batch_receipts": scorer_receipts[name]["retained_batch_receipts"],
                },
            }
            for name in CANDIDATES
        }
    else:
        gt_path = args.output / "custody/gt_argmax_n600.npy"
        c1_path = args.output / "custody/c1_target_argmax_n600.npy"
        gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
        c1 = np.load(c1_path, mmap_mode="r", allow_pickle=False)
        fields = {"c1_target": c1}
        axis = AXIS
        for name, candidate in CANDIDATES.items():
            scorer_result = candidate_root(args.output, candidate) / "scorer/SCORER_RESULT.json"
            if not scorer_result.is_file():
                raise RuntimeError(f"incomplete scorer payload for {name}")
            scorer_receipt = json.loads(scorer_result.read_text())
            if not scorer_receipt["complete"]:
                raise RuntimeError(f"incomplete scorer payload for {name}")
            root = candidate_root(args.output, candidate)
            require_receipt_binding(scorer_receipt["archive"], candidate.archive)
            require_receipt_binding(scorer_receipt["raw"], root / "retained/0.raw")
            require_receipt_binding(
                scorer_receipt["argmax"], root / "scorer/argmax_n600.npy"
            )
            require_receipt_binding(
                scorer_receipt["logits"], root / "scorer/logits_n600.float32.npy"
            )
            scorer_receipts[name] = scorer_receipt
            fields[name] = np.load(
                root / "scorer/argmax_n600.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        object_receipts = {
            name: {
                "archive": scorer_receipts[name]["archive"],
                "raw": scorer_receipts[name]["raw"],
                "argmax": scorer_receipts[name]["argmax"],
                "logits": scorer_receipts[name]["logits"],
                "scorer_receipt": file_record(
                    candidate_root(args.output, candidate) / "scorer/SCORER_RESULT.json"
                ),
            }
            for name, candidate in CANDIDATES.items()
        }
    decomp_root = args.output / "decomposition"
    summaries = {
        name: decomposition(field, gt, decomp_root / f"{name}_per_pair.jsonl")
        for name, field in fields.items()
    }
    base = summaries["cp135_base"]
    composed = summaries["t1r1_c1_composed"]
    target = summaries["c1_target"]
    axis_adjudication = adjudicate_axis(base["total_flips"], target["total_flips"])
    denominator = base["total_flips"] - target["total_flips"]
    diagnostic_rho = (
        (base["total_flips"] - composed["total_flips"]) / denominator
        if denominator > 0
        else None
    )
    admitted_rho = (
        diagnostic_rho if axis_adjudication["admitted_for_stage0_rho"] else None
    )
    base_edges = edge_map(base)
    composed_edges = edge_map(composed)
    target_edges = edge_map(target)
    edge_comparison = []
    for edge in sorted(base_edges):
        base_flips = int(base_edges[edge]["flips"])
        composed_flips = int(composed_edges[edge]["flips"])
        target_flips = int(target_edges[edge]["flips"])
        edge_denominator = base_flips - target_flips
        edge_rho = (
            (base_flips - composed_flips) / edge_denominator
            if edge_denominator > 0
            else None
        )
        axis_admitted = axis_adjudication["admitted_for_stage0_rho"]
        edge_comparison.append(
            {
                "edge": edge,
                "base_flips": base_flips,
                "composed_flips": composed_flips,
                "target_flips": target_flips,
                "base_to_composed_flip_gain": base_flips - composed_flips,
                "rho_measured": edge_rho if axis_admitted else None,
                "diagnostic_rho_not_admitted": edge_rho if not axis_admitted else None,
            }
        )
    edge_comparison.sort(
        key=lambda row: (-abs(row["base_to_composed_flip_gain"]), row["edge"])
    )
    result = {
        "schema": (
            "ddm_js1_stage0_per_edge_result.v2"
            if cuda_bundle is not None
            else "ddm_js1_stage0_per_edge_result.v1"
        ),
        "status": axis_adjudication["status"],
        "axis": axis,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "selection_mode": "full population, no sampling, all 600 non-overlapping pairs",
        "denominators": {
            "pairs": N,
            "scorer_pixels_per_pair": SEG_PX,
            "total_scorer_pixels": TOTAL_PX,
            "directed_edge_cells": 20,
            "undirected_interfaces": 10,
        },
        "m94_scope": {
            "claim_scope": "INSTANCE",
            "instrument_capacity_in_claim_units": (
                "117,964,800 frozen-SegNet argmax pixels, all 600 pairs, all 20 "
                "directed cells and 10 undirected interfaces"
            ),
            "object_capacity_in_claim_units": (
                "the complete downloaded contest-CUDA argmax fields for both exact bound archives"
                if cuda_bundle is not None
                else "the complete local-CPU decoded argmax fields for the two exact bound archives"
            ),
            "scope_limit": (
                "contest-CUDA T4 SegNet component field only; no PoseNet, full score, "
                "contest-CPU transfer, family kill, or pointer claim"
                if cuda_bundle is not None
                else "matched macOS CPU diagnostic receiver/scorer only; no CUDA transfer, "
                "family kill, or contest score claim"
            ),
        },
        "objects": object_receipts,
        "gt_argmax": file_record(gt_path),
        "c1_target_argmax": file_record(c1_path),
        "argmax_field_source": (
            file_record(cuda_bundle["receipt_path"])
            if cuda_bundle is not None
            else None
        ),
        "decompositions": summaries,
        "axis_adjudication": axis_adjudication,
        "comparison": {
            "base_to_composed_flip_gain": base["total_flips"] - composed["total_flips"],
            "base_to_composed_d_seg_gain": base["d_seg"] - composed["d_seg"],
            "diagnostic_rho_not_admitted": (
                diagnostic_rho
                if not axis_adjudication["admitted_for_stage0_rho"]
                else None
            ),
            "rho_measured": admitted_rho,
            "rho_required_for_sub015_reseal": RHO_REQUIRED,
            "rho_gate_passed": admitted_rho is not None and admitted_rho >= RHO_REQUIRED,
            "edge_rows": edge_comparison,
        },
        "boundaries": {
            "measured": (
                "downloaded receiver-closed contest-CUDA T4 frozen-SegNet argmax fields "
                "and full-n600 per-pair directed and undirected edge counts"
                if cuda_bundle is not None
                else "receiver-closed local CPU raw, frozen-SegNet logits/argmax, "
                "full-n600 per-pair directed and undirected edge counts"
            ),
            "not_measured": (
                "PoseNet, complete score, contest-CPU, public evaluator, V1-V5 causal "
                "ladder, any trained or optimized realization arm"
                if cuda_bundle is not None
                else "PoseNet, complete score, contest CUDA/CPU runtime, public evaluator, "
                "V1-V5 causal ladder, any trained or optimized realization arm"
            ),
            "blocking_reason": (
                None
                if cuda_bundle is not None
                else "the local CPU renderer does not reproduce the CUDA-locked terminal "
                "CP135 Seg row, and the local GT plane differs by one pixel from the "
                "retained C1 batch-16 reference; promoted CUDA argmax custody is required"
            ),
        },
    }
    atomic_json(args.output / "STAGE0_RESULT.json", result)
    return result


def all_stages(args: argparse.Namespace) -> dict[str, Any]:
    preflight(args)
    for name in CANDIDATES:
        child_values = vars(args).copy()
        child_values["candidate"] = name
        child = argparse.Namespace(**child_values)
        inflate(child)
        score(child)
    return summarize(args)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=("preflight", "inflate", "score", "summarize", "all"))
    value.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    value.add_argument("--candidate", choices=tuple(CANDIDATES))
    value.add_argument("--chunk", type=int, default=120)
    value.add_argument("--batch", type=int, default=1)
    value.add_argument(
        "--from-argmax-fields",
        type=Path,
        help="downloaded admitted JS1B CUDA field root; summarize only, no local forward",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    if args.stage in {"inflate", "score"} and args.candidate is None:
        raise SystemExit(f"{args.stage} requires --candidate")
    if args.from_argmax_fields is not None and args.stage != "summarize":
        raise SystemExit("--from-argmax-fields is valid only with summarize")
    if args.batch <= 0 or args.batch > args.chunk:
        raise SystemExit("--batch must be positive and no larger than --chunk")
    actions = {
        "preflight": preflight,
        "inflate": inflate,
        "score": score,
        "summarize": summarize,
        "all": all_stages,
    }
    result = actions[args.stage](args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
