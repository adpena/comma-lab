#!/usr/bin/env python3
"""Locked-cu128 T4 worker for the VD1 EC1 singleton batch validator.

The worker is intentionally provider-only.  It decodes CP135 once, retains
every materialized source batch and candidate payload, and evaluates only the
unique pairs affected by the supplied singleton events with the exact upstream
models and preprocessing path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import numpy as np

REMOTE_REPO: Final = Path("/workspace/pact")
UPSTREAM: Final = REMOTE_REPO / "upstream"
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
POSE_STACK_BUDGET_GLOBAL: Final = 1.3e-7
POSE_STACK_EQUIVALENT_EVENTS: Final = 44
POSE_PER_EVENT_GLOBAL_BUDGET: Final = POSE_STACK_BUDGET_GLOBAL / POSE_STACK_EQUIVALENT_EVENTS
POSE_PER_EVENT_PAIR_BUDGET: Final = POSE_PER_EVENT_GLOBAL_BUDGET * N_PAIRS
JS7_DAMAGE_SCALE: Final = 0.000216
CP135_ARCHIVE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
CP135_DECODED_TOKEN_SHA256: Final = (
    "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
)
AXIS: Final = "[contest-CUDA T4 exact-upstream affected-pair n600 delta]"
CHECKPOINT_EVERY_EVENTS: Final = 10


class WorkerError(RuntimeError):
    """A remote decode, scorer, resume, or retention invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    staging.write_bytes(payload)
    os.replace(staging, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def checkpoint_once(path: Path, value: Any) -> None:
    """Create one immutable stage checkpoint, or verify its resume identity."""
    payload = canonical_json_bytes(value)
    if path.is_file():
        if path.read_bytes() != payload:
            raise WorkerError(f"resume checkpoint differs: {path}")
        return
    atomic_bytes(path, payload)


def atomic_npy(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
    os.replace(staging, path)
    return file_record(path)


def atomic_npz(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        np.savez(stream, **values)
    os.replace(staging, path)
    return file_record(path)


def extract_zip_once(path: Path, destination: Path, marker_name: str) -> None:
    marker = destination / marker_name
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise WorkerError(f"unsafe ZIP member: {info.filename}")
            if info.is_dir():
                continue
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_bytes(target, archive.read(info))
            os.chmod(target, (info.external_attr >> 16) & 0o777 or 0o644)
    atomic_json(marker, {"source": file_record(path), "complete": True})


def read_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise WorkerError("truncated EC1 varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def decode_event(payload: bytes) -> tuple[int, int, int, int, np.ndarray]:
    if len(payload) < 13 or payload[:8] != b"EC1PROP1":
        raise WorkerError("event header differs")
    frame, source, target, event_type = struct.unpack_from("<HBBB", payload, 8)
    if frame >= N_PAIRS or source >= 5 or target >= 5:
        raise WorkerError("event address differs")
    count, offset = read_uvarint(payload, 13)
    indices = np.empty(count, dtype=np.int64)
    previous = 0
    for position in range(count):
        gap, offset = read_uvarint(payload, offset)
        value = gap if position == 0 else previous + gap
        if value >= HEIGHT * WIDTH or (position and value <= previous):
            raise WorkerError("event coordinate differs")
        indices[position] = value
        previous = value
    if offset != len(payload):
        raise WorkerError("event has trailing bytes")
    return frame, source, target, event_type, indices


def load_events(event_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((event_root / "VD1_EVENT_MANIFEST.json").read_text())
    rows = manifest.get("events")
    if not isinstance(rows, list) or len(rows) != int(manifest.get("selected_events", -1)):
        raise WorkerError("event manifest count differs")
    for expected_ordinal, row in enumerate(rows):
        if int(row["ordinal"]) != expected_ordinal:
            raise WorkerError("event ordinals are not dense")
        path = event_root / str(row["member"])
        if file_record(path)["sha256"] != row["payload_sha256"]:
            raise WorkerError(f"event payload digest differs: {row['proposal_id']}")
    return manifest, rows


def compile_rc64(runtime_root: Path, work_root: Path) -> Path:
    source = runtime_root / "runtime/entropy/rc64_backend.c"
    output = work_root / "rc64_backend.so"
    if output.is_file():
        return output
    command = ["cc", "-O3", "-shared", "-fPIC", str(source), "-o", str(output)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not output.is_file():
        raise WorkerError("RC64 compile returned without a shared library")
    return output


def tree_record(root: Path) -> dict[str, Any]:
    """Hash one retained dependency tree without omitting its materialized files."""
    digest = hashlib.sha256()
    rows = []
    total_bytes = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        record = file_record(path)
        rows.append({"relative_path": relative, **record})
        total_bytes += int(record["bytes"])
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(rows),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
        "files": rows,
    }


def _adapted_brotli_spec(runtime_root: Path) -> str:
    """Read the dependency pin from the staged runtime's own inflate entrypoint."""
    source = (runtime_root / "inflate.sh").read_text()
    matches = sorted(set(re.findall(r"Brotli==[0-9]+(?:\.[0-9]+)+", source)))
    if len(matches) != 1:
        raise WorkerError(f"adapted runtime Brotli pin is not singular: {matches}")
    return matches[0]


def _loaded_brotli_record(expected_spec: str) -> dict[str, Any] | None:
    expected_version = expected_spec.partition("==")[2]
    try:
        module = importlib.import_module("brotli")
        version = importlib.metadata.version("Brotli")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return None
    if version != expected_version or not callable(getattr(module, "decompress", None)):
        return None
    module_path = Path(module.__file__).resolve()
    return {"spec": expected_spec, "version": version, "module": file_record(module_path)}


def _activate_brotli_site(site: Path, spec: str) -> dict[str, Any] | None:
    site_text = str(site)
    if site_text not in sys.path:
        sys.path.insert(0, site_text)
    for name in ("brotli", "_brotli"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return _loaded_brotli_record(spec)


def ensure_adapted_runtime_brotli(
    runtime_root: Path,
    retained_root: Path,
) -> dict[str, Any]:
    """Apply the adapted inflate.sh Brotli bootstrap inside the locked worker Python."""
    spec = _adapted_brotli_spec(runtime_root)
    receipt_path = retained_root / "BOOTSTRAP.json"
    available = _loaded_brotli_record(spec)
    if available is not None:
        receipt = {
            "schema": "ddm_vd1_adapted_runtime_dependency.v1",
            "status": "ALREADY_AVAILABLE",
            "authority": file_record(runtime_root / "inflate.sh"),
            "dependency": available,
            "installed_tree": None,
        }
        atomic_json(receipt_path, receipt)
        return {**receipt, "receipt": file_record(receipt_path)}

    site = retained_root / "site"
    cache = retained_root / "uv_cache"
    if site.is_dir():
        available = _activate_brotli_site(site, spec)
        if available is not None:
            receipt = {
                "schema": "ddm_vd1_adapted_runtime_dependency.v1",
                "status": "REUSED_RETAINED_INSTALL",
                "authority": file_record(runtime_root / "inflate.sh"),
                "dependency": available,
                "installed_tree": tree_record(site),
                "cache_tree": tree_record(cache) if cache.is_dir() else None,
            }
            atomic_json(receipt_path, receipt)
            return {**receipt, "receipt": file_record(receipt_path)}

    uv = shutil.which("uv")
    if not uv:
        raise WorkerError(f"adapted runtime requires {spec}, but uv is unavailable")
    site.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    # Install on LOCAL container disk, then copy to the retained volume. uv's cache
    # and --target both persist temp files via rename, which Modal volumes refuse
    # ("Could not persist temporary file ... Operation not permitted" — measured
    # fc-01KZWKRB00 run ddm_vd1_20260812b BOOTSTRAP.json rc=2). Local tmp has full
    # POSIX semantics; the volume copy is custody, not the install target.
    local_root = Path(tempfile.mkdtemp(prefix="vd1_brotli_"))
    local_site = local_root / "site"
    local_cache = local_root / "uv_cache"
    local_site.mkdir(parents=True, exist_ok=True)
    local_cache.mkdir(parents=True, exist_ok=True)
    command = [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(local_site),
        "--no-deps",
        "--only-binary",
        ":all:",
        spec,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "UV_CACHE_DIR": str(local_cache)},
    )
    if not completed.returncode:
        shutil.copytree(local_site, site, dirs_exist_ok=True)
    install = {
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "uv_cache": str(cache),
    }
    if completed.returncode:
        atomic_json(
            receipt_path,
            {
                "schema": "ddm_vd1_adapted_runtime_dependency.v1",
                "status": "INSTALL_FAILED",
                "authority": file_record(runtime_root / "inflate.sh"),
                "install": install,
            },
        )
        raise WorkerError(f"adapted runtime dependency install failed for {spec}")

    available = _activate_brotli_site(site, spec)
    if available is None:
        raise WorkerError(f"adapted runtime dependency remained unavailable after installing {spec}")
    receipt = {
        "schema": "ddm_vd1_adapted_runtime_dependency.v1",
        "status": "INSTALLED_FROM_ADAPTED_RUNTIME_PIN",
        "authority": file_record(runtime_root / "inflate.sh"),
        "dependency": available,
        "install": install,
        "installed_tree": tree_record(site),
        "cache_tree": tree_record(cache),
    }
    atomic_json(receipt_path, receipt)
    return {**receipt, "receipt": file_record(receipt_path)}


def _load_adapted_f26(runtime_root: Path) -> Any:
    """Import the exact staged module used by adapted_runtime/inflate.py."""
    expected = (runtime_root / "runtime/f26_inflate.py").resolve()
    sys.path.insert(0, str(runtime_root))
    try:
        module = importlib.import_module("runtime.f26_inflate")
    finally:
        sys.path.pop(0)
    actual = Path(module.__file__).resolve()
    if actual != expected:
        raise WorkerError(f"adapted runtime import resolved elsewhere: {actual} != {expected}")
    return module


def load_receiver_state(
    archive_path: Path,
    runtime_root: Path,
    retained_root: Path,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    """Decode CP135 once and retain a complete receiver checkpoint."""
    import torch

    dependency_report = ensure_adapted_runtime_brotli(
        runtime_root,
        retained_root / "runtime_dependencies/brotli",
    )
    f26 = _load_adapted_f26(runtime_root)
    parts = f26.read_residual_archive(archive_path)
    renderer = f26._load_renderer(runtime_root / "cpr1")
    carrier_blob, selector_blob = f26.split_frame0_selector_carrier(parts.carrier_blob)
    canonical_carrier = f26.materialize_cpr1(carrier_blob, renderer)
    semantic_pose = struct.pack("<II", 40_252, len(canonical_carrier)) + bytes(40_252) + canonical_carrier
    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
    records = f26.decode_wans1(parts.semantic_blob)
    semantic = renderer.SemanticTokenRenderer(96)
    state = {
        record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
        for record in records
    }
    semantic.load_state_dict(state, strict=True)
    tokens_path = retained_root / "decoded/base_tokens.uint8.npy"
    report_path = retained_root / "decoded/TOKEN_DECODE_REPORT.json"
    if tokens_path.is_file() and report_path.is_file():
        tokens = torch.from_numpy(np.load(tokens_path, allow_pickle=False))
        token_report = json.loads(report_path.read_text())
    else:
        tokens, token_report = f26.decode_production_tokens(
            parts,
            renderer,
            runtime_root / "cpr1",
            device,
        )
        atomic_npy(tokens_path, tokens.numpy())
        atomic_json(report_path, token_report)
    if tuple(tokens.shape) != (N_PAIRS, HEIGHT, WIDTH):
        raise WorkerError(f"decoded token shape differs: {tuple(tokens.shape)}")
    decoded_token_sha256 = sha256_bytes(tokens.numpy().tobytes())
    if decoded_token_sha256 != token_report["decoded_token_sha256"]:
        raise WorkerError("decoded token digest differs from receiver report")
    if decoded_token_sha256 != CP135_DECODED_TOKEN_SHA256:
        raise WorkerError("decoded token plane differs from the adapted CP135 receiver golden")
    atomic_npz(
        retained_root / "decoded/semantic_weights.float32.npz",
        {name: value.numpy() for name, value in state.items()},
    )
    atomic_npy(retained_root / "decoded/carrier_basis.float32.npy", basis.cpu().numpy())
    atomic_npy(retained_root / "decoded/carrier_coefficients.float32.npy", coefficients.cpu().numpy())
    if selector_blob is None:
        selector_modes, selector_indices = (), np.zeros(N_PAIRS, dtype=np.uint8)
    else:
        selector_modes, selector_indices = f26.decode_selector(selector_blob)
    atomic_npy(retained_root / "decoded/frame0_selector_indices.uint8.npy", selector_indices)
    token_report = {
        **token_report,
        "archive": file_record(archive_path),
        "tokens": file_record(tokens_path),
        "semantic_weights": file_record(retained_root / "decoded/semantic_weights.float32.npz"),
        "basis": file_record(retained_root / "decoded/carrier_basis.float32.npy"),
        "coefficients": file_record(retained_root / "decoded/carrier_coefficients.float32.npy"),
        "selector_indices": file_record(retained_root / "decoded/frame0_selector_indices.uint8.npy"),
        "adapted_runtime_module": file_record(Path(f26.__file__).resolve()),
        "adapted_runtime_dependency": dependency_report,
        "adapted_runtime_token_golden_sha256": CP135_DECODED_TOKEN_SHA256,
        "adapted_runtime_token_golden_match": True,
    }
    return renderer, semantic, basis, coefficients, tokens, (selector_modes, selector_indices), token_report


def render_master(renderer: Any, semantic: Any, tokens: np.ndarray, pair: int, device: Any) -> np.ndarray:
    import torch

    semantic = semantic.eval().to(device)
    with torch.inference_mode():
        value = semantic(
            torch.from_numpy(np.ascontiguousarray(tokens))[None].long().to(device),
            torch.tensor([pair], dtype=torch.long, device=device),
        )
        camera = (
            torch.nn.functional.interpolate(
                value,
                size=(renderer.CAMERA_H, renderer.CAMERA_W),
                mode="bilinear",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
            .to(torch.uint8)
        )
    return camera[0].permute(1, 2, 0).cpu().numpy()


def render_frame0(
    renderer: Any,
    normalized_basis: Any,
    coefficients: Any,
    pair: int,
    selector: tuple[Any, np.ndarray],
    device: Any,
) -> np.ndarray:
    import torch

    modes, indices = selector
    with torch.inference_mode():
        carrier = torch.einsum("bk,kchw->bchw", coefficients[pair : pair + 1].to(device), normalized_basis)
        carrier = carrier / math.sqrt(renderer.CARRIER_DIM)
        frame = (
            torch.nn.functional.interpolate(
                (127.5 + renderer.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round(),
                size=(renderer.CAMERA_H, renderer.CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
            .to(torch.uint8)
        )[0].permute(1, 2, 0).cpu().numpy()
    mode_index = int(indices[pair])
    if mode_index:
        sys.path.insert(0, str(Path(renderer.__file__).resolve().parents[1]))
        try:
            from runtime.frame0_selector import apply_pixel_mode
        finally:
            sys.path.pop(0)
        frame = apply_pixel_mode(frame[None], modes[mode_index])[0]
    return frame


def load_scorer(device: Any) -> Any:
    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path
    finally:
        sys.path.pop(0)
    network = DistortionNet().eval().to(device=device)
    network.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    return network


def score_pair(network: Any, pair: np.ndarray, device: Any) -> dict[str, np.ndarray]:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(pair))[None].to(device)
    with torch.inference_mode():
        pose_input, seg_input = network.preprocess_input(tensor)
        pose_output = network.posenet(pose_input)["pose"][..., :6]
        seg_logits = network.segnet(seg_input)
        seg_argmax = seg_logits.argmax(dim=1)
    return {
        "pose_input": pose_input.cpu().numpy(),
        "seg_input": seg_input.cpu().numpy(),
        "pose_output6": pose_output.cpu().numpy(),
        "seg_logits": seg_logits.cpu().numpy(),
        "seg_argmax": seg_argmax.to(torch.uint8).cpu().numpy(),
    }


def retain_scored(root: Path, prefix: str, scored: dict[str, np.ndarray]) -> dict[str, Any]:
    records = {}
    for name, value in scored.items():
        records[name] = atomic_npy(root / f"{prefix}_{name}.npy", value)
    return records


def decode_and_retain_gt(
    retained_root: Path,
    affected_pairs: set[int],
    network: Any,
    device: Any,
) -> dict[int, dict[str, Any]]:
    """Decode the exact n600 DALI corpus, retaining every materialized batch."""
    import torch

    sys.path.insert(0, str(UPSTREAM))
    try:
        from frame_utils import DaliVideoDataset
    finally:
        sys.path.pop(0)
    names = (UPSTREAM / "public_test_video_names.txt").read_text().splitlines()
    dataset = DaliVideoDataset(
        names,
        data_dir=UPSTREAM / "videos",
        batch_size=16,
        device=device,
        num_threads=2,
        seed=1234,
        prefetch_queue_depth=4,
    )
    dataset.prepare_data()
    selected: dict[int, dict[str, Any]] = {}
    cursor = 0
    batch_rows = []
    for batch_ordinal, (_path, _index, batch) in enumerate(dataset):
        batch_cpu = batch.cpu().numpy()
        batch_record = atomic_npy(
            retained_root / f"gt/n600_batches/batch_{batch_ordinal:04d}.uint8.npy",
            batch_cpu,
        )
        batch_rows.append(
            {
                "ordinal": batch_ordinal,
                "pair_start": cursor,
                "pair_end": cursor + len(batch_cpu),
                "payload": batch_record,
            }
        )
        for offset in range(len(batch_cpu)):
            pair_id = cursor + offset
            if pair_id not in affected_pairs:
                continue
            pair = np.asarray(batch_cpu[offset]).copy()
            pair_root = retained_root / f"gt/affected_pairs/{pair_id:04d}"
            pair_record = atomic_npy(pair_root / "pair.uint8.npy", pair)
            scored = score_pair(network, pair, device)
            selected[pair_id] = {
                "pair": pair_id,
                "pair_payload": pair_record,
                "scorer_payloads": retain_scored(pair_root, "gt", scored),
            }
        cursor += len(batch_cpu)
    torch.cuda.synchronize(device)
    if cursor != N_PAIRS or set(selected) != affected_pairs:
        raise WorkerError(f"GT census differs: decoded={cursor}, selected={sorted(selected)}")
    atomic_json(
        retained_root / "gt/N600_DECODE_MANIFEST.json",
        {
            "schema": "ddm_vd1_gt_n600_decode_manifest.v1",
            "axis": AXIS,
            "decoded_pairs": cursor,
            "affected_pairs": sorted(affected_pairs),
            "batches": batch_rows,
            "all_materialized_gt_batches_retained": True,
        },
    )
    return selected


def load_retained_scored(root: Path, prefix: str) -> dict[str, np.ndarray]:
    return {
        name: np.load(root / f"{prefix}_{name}.npy", allow_pickle=False)
        for name in ("pose_input", "seg_input", "pose_output6", "seg_logits", "seg_argmax")
    }


def base_cache(
    retained_root: Path,
    affected_pairs: set[int],
    renderer: Any,
    semantic: Any,
    basis: Any,
    coefficients: Any,
    tokens: Any,
    selector: tuple[Any, np.ndarray],
    network: Any,
    device: Any,
) -> dict[int, dict[str, Any]]:
    normalized_basis = renderer.normalized_basis(basis.to(device))
    cache: dict[int, dict[str, Any]] = {}
    for pair_id in sorted(affected_pairs):
        root = retained_root / f"base/affected_pairs/{pair_id:04d}"
        result_path = root / "RESULT.json"
        if result_path.is_file():
            cache[pair_id] = json.loads(result_path.read_text())
            continue
        frame0 = render_frame0(renderer, normalized_basis, coefficients, pair_id, selector, device)
        frame1 = render_master(renderer, semantic, np.asarray(tokens[pair_id]), pair_id, device)
        pair = np.stack((frame0, frame1), axis=0)
        payloads = {
            "frame0": atomic_npy(root / "frame0.uint8.npy", frame0),
            "frame1": atomic_npy(root / "frame1.uint8.npy", frame1),
            "pair": atomic_npy(root / "pair.uint8.npy", pair),
        }
        scored = score_pair(network, pair, device)
        result = {
            "pair": pair_id,
            "payloads": payloads,
            "scorer_payloads": retain_scored(root, "base", scored),
        }
        atomic_json(result_path, result)
        cache[pair_id] = result
    return cache


def event_result(
    retained_root: Path,
    event_root: Path,
    row: dict[str, Any],
    renderer: Any,
    semantic: Any,
    base_tokens: Any,
    network: Any,
    device: Any,
) -> dict[str, Any]:
    proposal_id = str(row["proposal_id"])
    root = retained_root / f"events/{int(row['ordinal']):04d}_{proposal_id}"
    result_path = root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    payload = (event_root / str(row["member"])).read_bytes()
    pair_id, source, target, event_type, indices = decode_event(payload)
    tokens = np.asarray(base_tokens[pair_id]).copy()
    flat = tokens.reshape(-1)
    if np.any(flat[indices] != source):
        raise WorkerError(f"event source precondition differs: {proposal_id}")
    flat[indices] = target
    candidate_master = render_master(renderer, semantic, tokens, pair_id, device)
    base_pair_root = retained_root / f"base/affected_pairs/{pair_id:04d}"
    base_pair = np.load(base_pair_root / "pair.uint8.npy", allow_pickle=False)
    candidate_pair = np.stack((base_pair[0], candidate_master), axis=0)
    candidate_scored = score_pair(network, candidate_pair, device)
    base_scored = load_retained_scored(base_pair_root, "base")
    gt_scored = load_retained_scored(
        retained_root / f"gt/affected_pairs/{pair_id:04d}", "gt"
    )
    gt_labels = gt_scored["seg_argmax"]
    base_flips = int(np.count_nonzero(base_scored["seg_argmax"] != gt_labels))
    candidate_flips = int(np.count_nonzero(candidate_scored["seg_argmax"] != gt_labels))
    gt_pose = gt_scored["pose_output6"].astype(np.float64)
    base_pose_error = float(np.mean((base_scored["pose_output6"].astype(np.float64) - gt_pose) ** 2))
    candidate_pose_error = float(
        np.mean((candidate_scored["pose_output6"].astype(np.float64) - gt_pose) ** 2)
    )
    delta_flips = candidate_flips - base_flips
    delta_pose_pair = candidate_pose_error - base_pose_error
    delta_pose_global = delta_pose_pair / N_PAIRS
    payloads = {
        "event": file_record(event_root / str(row["member"])),
        "indices": atomic_npy(root / "event_indices.int64.npy", indices),
        "candidate_tokens": atomic_npy(root / "candidate_tokens.uint8.npy", tokens),
        "candidate_master": atomic_npy(root / "candidate_master.uint8.npy", candidate_master),
        "candidate_pair": atomic_npy(root / "candidate_pair.uint8.npy", candidate_pair),
        "candidate_scorer": retain_scored(root, "candidate", candidate_scored),
    }
    result = {
        "schema": "ddm_vd1_singleton_result.v1",
        "axis": AXIS,
        "n600_denominator_exact": True,
        "proposal_id": proposal_id,
        "ordinal": int(row["ordinal"]),
        "pair": pair_id,
        "source_class": source,
        "target_class": target,
        "event_type_id": event_type,
        "site_count": int(indices.size),
        "base_flips_pair": base_flips,
        "candidate_flips_pair": candidate_flips,
        "delta_flips_candidate_minus_base": delta_flips,
        "net_flip_gain_base_minus_candidate": -delta_flips,
        "base_d_pose_pair": base_pose_error,
        "candidate_d_pose_pair": candidate_pose_error,
        "delta_d_pose_pair": delta_pose_pair,
        "delta_d_pose_global_n600": delta_pose_global,
        "pose_per_event_pair_budget": POSE_PER_EVENT_PAIR_BUDGET,
        "pose_per_event_global_budget": POSE_PER_EVENT_GLOBAL_BUDGET,
        "downstream_selection_eligible": (
            delta_flips < 0 and delta_pose_global <= POSE_PER_EVENT_GLOBAL_BUDGET
        ),
        "payloads": payloads,
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(result_path, result)
    return result


def selection_projection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row["downstream_selection_eligible"])]
    eligible.sort(
        key=lambda row: (
            -int(row["net_flip_gain_base_minus_candidate"]),
            float(row["delta_d_pose_global_n600"]),
            int(row["ordinal"]),
        )
    )
    pose_used = 0.0
    selected = []
    for row in eligible:
        proposed = pose_used + float(row["delta_d_pose_global_n600"])
        if proposed > POSE_STACK_BUDGET_GLOBAL:
            continue
        selected.append(str(row["proposal_id"]))
        pose_used = proposed
    optimistic_flip_gain = sum(
        int(row["net_flip_gain_base_minus_candidate"])
        for row in eligible
        if str(row["proposal_id"]) in set(selected)
    )
    optimistic_seg_score_gain = 100.0 * optimistic_flip_gain / (N_PAIRS * HEIGHT * WIDTH)
    return {
        "schema": "ddm_vd1_downstream_selection_projection.v1",
        "selection_only_not_composed": True,
        "singleton_interactions_unmeasured": True,
        "eligible_count": len(eligible),
        "selected_ids_under_additive_pose_budget": selected,
        "selected_count": len(selected),
        "additive_global_pose_delta": pose_used,
        "pose_stack_budget_global": POSE_STACK_BUDGET_GLOBAL,
        "optimistic_additive_singleton_flip_gain": optimistic_flip_gain,
        "optimistic_additive_seg_score_gain": optimistic_seg_score_gain,
        "js7_six_event_realized_damage_scale": JS7_DAMAGE_SCALE,
        "ec1_gen1_falsifier_fires_on_optimistic_projection": optimistic_seg_score_gain < JS7_DAMAGE_SCALE,
        "falsifier_scope": (
            "projection only until MAIN composes selected events into one retained +<=3B archive and "
            "runs the final exact row"
        ),
    }


def run(run_root: Path, resume_from: str) -> dict[str, Any]:
    import timm
    import torch
    import torchvision

    from tac.contest_compliance import compute_upstream_snapshot_sha256

    started = time.time()
    request = json.loads((run_root / "inputs/REQUEST.json").read_text())
    if resume_from != str(request["resume_from"]) or resume_from != str(request["run_id"]):
        raise WorkerError("resume token differs from the retained run identity")
    final_path = run_root / "FINAL_RESULT.json"
    if final_path.is_file():
        completed = json.loads(final_path.read_text())
        if completed.get("status") != "COMPLETE":
            raise WorkerError("retained final result is not complete")
        return completed
    archive_path = run_root / "inputs/archive.zip"
    runtime_zip = run_root / "inputs/runtime_bundle.zip"
    event_zip = run_root / "inputs/event_bundle.zip"
    if sha256_file(archive_path) != CP135_ARCHIVE_SHA256:
        raise WorkerError("remote archive differs from CP135 custody")
    upstream_snapshot_sha256 = compute_upstream_snapshot_sha256(
        UPSTREAM,
        upstream_subdir=".",
        reject_executable_artifacts=True,
    )
    if not upstream_snapshot_sha256:
        raise WorkerError("remote canonical upstream snapshot is missing")
    work_root = run_root / "work"
    runtime_root = work_root / "runtime_bundle"
    event_root = work_root / "event_bundle"
    extract_zip_once(runtime_zip, runtime_root, "VD1_RUNTIME_EXTRACTED.json")
    extract_zip_once(event_zip, event_root, "VD1_EVENTS_EXTRACTED.json")
    _event_manifest, event_rows = load_events(event_root)
    checkpoint_once(
        run_root / "checkpoints/stage_00_inputs.json",
        {
            "schema": "ddm_vd1_stage_checkpoint.v1",
            "stage": "inputs",
            "archive": file_record(archive_path),
            "runtime_bundle": file_record(runtime_zip),
            "event_bundle": file_record(event_zip),
            "resume_from": resume_from,
            "complete": True,
        },
    )
    rc64 = compile_rc64(runtime_root, work_root)
    os.environ["CPR1_RC64_LIBRARY"] = str(rc64)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    renderer, semantic, basis, coefficients, tokens, selector, token_report = load_receiver_state(
        archive_path, runtime_root, run_root / "retained", device
    )
    checkpoint_once(
        run_root / "checkpoints/stage_10_base_decoded.json",
        {
            "schema": "ddm_vd1_stage_checkpoint.v1",
            "stage": "base_decoded",
            "token_report": token_report,
            "rc64_library": file_record(rc64),
            "complete": True,
        },
    )
    network = load_scorer(device)
    affected_pairs = {int(row["pair"]) for row in event_rows}
    gt_manifest = run_root / "retained/gt/N600_DECODE_MANIFEST.json"
    if not gt_manifest.is_file():
        decode_and_retain_gt(run_root / "retained", affected_pairs, network, device)
    checkpoint_once(
        run_root / "checkpoints/stage_20_gt_n600_decoded.json",
        {
            "schema": "ddm_vd1_stage_checkpoint.v1",
            "stage": "gt_n600_decoded",
            "manifest": file_record(gt_manifest),
            "affected_pairs": sorted(affected_pairs),
            "complete": True,
        },
    )
    base = base_cache(
        run_root / "retained",
        affected_pairs,
        renderer,
        semantic,
        basis,
        coefficients,
        tokens,
        selector,
        network,
        device,
    )
    checkpoint_once(
        run_root / "checkpoints/stage_30_base_pairs_scored.json",
        {
            "schema": "ddm_vd1_stage_checkpoint.v1",
            "stage": "base_pairs_scored",
            "pair_count": len(base),
            "affected_pairs": sorted(base),
            "complete": True,
        },
    )
    results = []
    for ordinal, row in enumerate(event_rows):
        result = event_result(
            run_root / "retained",
            event_root,
            row,
            renderer,
            semantic,
            tokens,
            network,
            device,
        )
        results.append(result)
        completed = ordinal + 1
        if completed % CHECKPOINT_EVERY_EVENTS == 0 or completed == len(event_rows):
            checkpoint_once(
                run_root / f"checkpoints/stage_40_events_{completed:04d}.json",
                {
                    "schema": "ddm_vd1_stage_checkpoint.v1",
                    "stage": "events",
                    "completed_events": completed,
                    "total_events": len(event_rows),
                    "last_proposal_id": result["proposal_id"],
                    "complete": completed == len(event_rows),
                },
            )
    projection = selection_projection(results)
    rows_path = run_root / "EVENT_RESULTS.jsonl"
    atomic_bytes(rows_path, b"".join(canonical_json_bytes(row) for row in results))
    final = {
        "schema": "ddm_vd1_modal_batch_event_validator_result.v1",
        "status": "COMPLETE",
        "axis": AXIS,
        "n600_denominator": N_PAIRS,
        "base_archive": file_record(archive_path),
        "event_count": len(results),
        "affected_pair_count": len(affected_pairs),
        "affected_pairs": sorted(affected_pairs),
        "event_results": file_record(rows_path),
        "net_flip_gain_positive_count": sum(
            int(row["net_flip_gain_base_minus_candidate"]) > 0 for row in results
        ),
        "pose_budget_pass_count": sum(
            float(row["delta_d_pose_global_n600"]) <= POSE_PER_EVENT_GLOBAL_BUDGET
            for row in results
        ),
        "downstream_selection_eligible_count": sum(
            bool(row["downstream_selection_eligible"]) for row in results
        ),
        "downstream_selection_projection": projection,
        "pose_budget": {
            "stack_global_d_pose": POSE_STACK_BUDGET_GLOBAL,
            "equivalent_events": POSE_STACK_EQUIVALENT_EVENTS,
            "per_event_global_d_pose": POSE_PER_EVENT_GLOBAL_BUDGET,
            "per_event_pair_d_pose": POSE_PER_EVENT_PAIR_BUDGET,
        },
        "retention": {
            "all_input_payloads": True,
            "base_token_plane": True,
            "all_materialized_gt_n600_batches": True,
            "all_affected_gt_and_base_scorer_tensors": True,
            "all_per_event_payloads_tokens_frames_scorer_tensors_and_deltas": True,
            "volume_run_root": str(run_root),
        },
        "provenance": {
            "source_git_head": request["source_git_head"],
            "source_git_dirty_at_dispatch": request["source_git_dirty"],
            "source_git_status_sha256": request["source_git_status_sha256"],
            "dispatcher_source_sha256": request["dispatcher_source_sha256"],
            "worker_source_sha256": request["worker_source_sha256"],
            "upstream_snapshot_sha256": upstream_snapshot_sha256,
            "seed": int(request["seed"]),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "timm_version": timm.__version__,
            "numpy_version": np.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_final_exact_row_required": True,
    }
    checkpoint_once(run_root / "checkpoints/stage_50_final.json", final)
    checkpoint_once(final_path, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", required=True)
    args = parser.parse_args(argv)
    result = run(args.run_root.resolve(), args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
