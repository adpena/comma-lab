#!/usr/bin/env python3
"""Reproduce the scorer-free OP1R cache, receiver, and XZ measurements."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import _lzma
import importlib.metadata
import importlib.util
import io
import json
import lzma
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(spec: dict[str, Any], *, key: str = "path") -> Path:
    path = Path(spec[key])
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in spec and path.stat().st_size != spec["bytes"]:
        raise RuntimeError(
            f"size mismatch for {path}: {path.stat().st_size} != {spec['bytes']}"
        )
    expected = spec.get("sha256") or spec.get("archive_sha256")
    if expected and sha256_file(path) != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}")
    return path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_receipt(
    hardware_label: str, dependency_root: Path | None
) -> dict[str, Any]:
    xz = subprocess.run(
        ["xz", "--version"], check=False, capture_output=True, text=True
    )
    constriction_version = None
    if dependency_root is not None:
        sys.path.insert(0, str(dependency_root.resolve()))
        try:
            constriction_version = package_version("constriction")
        finally:
            sys.path.remove(str(dependency_root.resolve()))
    lzma_extension = Path(_lzma.__file__).resolve()
    otool = subprocess.run(
        ["otool", "-L", str(lzma_extension)],
        check=False,
        capture_output=True,
        text=True,
    )
    linked_liblzma_line = next(
        (
            line.strip()
            for line in otool.stdout.splitlines()[1:]
            if "liblzma" in line
        ),
        None,
    )
    linked_liblzma = (
        Path(linked_liblzma_line.split(" (", 1)[0]).resolve()
        if linked_liblzma_line is not None
        else None
    )
    python_executable = Path(sys.executable)
    lzma_module = Path(lzma.__file__).resolve()
    return {
        "hardware_label": hardware_label,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_build": platform.python_build(),
        "python_compiler": platform.python_compiler(),
        "python_executable": str(python_executable),
        "python_executable_resolved": str(python_executable.resolve()),
        "python_executable_sha256": sha256_file(python_executable),
        "numpy": package_version("numpy"),
        "torch": package_version("torch"),
        "constriction": constriction_version,
        "dependency_root": (
            str(dependency_root.resolve()) if dependency_root is not None else None
        ),
        "lzma_module": str(lzma_module),
        "lzma_module_sha256": sha256_file(lzma_module),
        "_lzma_extension": str(lzma_extension),
        "_lzma_extension_sha256": sha256_file(lzma_extension),
        "linked_liblzma": str(linked_liblzma) if linked_liblzma else None,
        "linked_liblzma_sha256": (
            sha256_file(linked_liblzma) if linked_liblzma else None
        ),
        "linked_liblzma_otool_line": linked_liblzma_line,
        "xz_version_stdout": xz.stdout.strip(),
        "seed_policy": {
            "target_cache": "N/A_no_RNG",
            "xz": "N/A_no_RNG",
            "receiver_overwritten_constructor_state": 0,
        },
    }


def load_torch_xz(path: Path) -> tuple[dict[str, Any], bytes]:
    import torch

    raw = lzma.decompress(path.read_bytes())
    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict cache in {path}")
    return obj, raw


def numpy_raw_sha(array: Any) -> str:
    import numpy as np

    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def numeric_summary(values: Any) -> dict[str, float | int]:
    import numpy as np

    values = np.asarray(values)
    return {
        "min": int(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "max": int(values.max()),
    }


def reproduce_target_cache(
    config: dict[str, Any], common: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np
    import torch

    started_at = utc_now()
    started = time.monotonic()
    inputs = config["inputs"]
    dali_path = checked_file(inputs["official_dali_xz"])
    original_path = checked_file(inputs["pr130_original_av_like_xz"])
    local_path = checked_file(inputs["local_macos_cpu_av_npy"])
    m1_path = checked_file(inputs["m1_target_cache"])
    verified_inputs = {}
    for name, path in {
        "official_dali_xz": dali_path,
        "pr130_original_av_like_xz": original_path,
        "local_macos_cpu_av_npy": local_path,
        "m1_target_cache": m1_path,
    }.items():
        expected_input = inputs[name]
        verified_inputs[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": expected_input["sha256"],
            "verified_against_config": True,
        }

    dali_obj, dali_raw = load_torch_xz(dali_path)
    original_obj, original_raw = load_torch_xz(original_path)
    for name, obj in {"DALI": dali_obj, "PR130 original": original_obj}.items():
        seg_tensor = obj.get("seg")
        pose_tensor = obj.get("pose")
        if not isinstance(seg_tensor, torch.Tensor):
            raise RuntimeError(f"{name} segmentation is not a Torch tensor")
        if tuple(seg_tensor.shape) != (600, 384, 512):
            raise RuntimeError(f"{name} segmentation has unexpected stored shape")
        if seg_tensor.dtype != torch.uint8:
            raise RuntimeError(f"{name} segmentation has unexpected stored dtype")
        if not isinstance(pose_tensor, torch.Tensor):
            raise RuntimeError(f"{name} pose is not a Torch tensor")
        if tuple(pose_tensor.shape) != (600, 6):
            raise RuntimeError(f"{name} pose has unexpected stored shape")
        if pose_tensor.dtype != torch.float32:
            raise RuntimeError(f"{name} pose has unexpected stored dtype")
    dali_seg = np.ascontiguousarray(dali_obj["seg"].cpu().numpy(), dtype=np.uint8)
    original_seg = np.ascontiguousarray(
        original_obj["seg"].cpu().numpy(), dtype=np.uint8
    )
    local_seg = np.load(local_path)
    if local_seg.shape != (600, 384, 512) or local_seg.dtype != np.uint8:
        raise RuntimeError("local AV-like array has unexpected shape/dtype")
    local_seg = np.ascontiguousarray(local_seg)
    for name, value in {"DALI": dali_seg, "PR130 original": original_seg}.items():
        if value.shape != (600, 384, 512) or value.dtype != np.uint8:
            raise RuntimeError(f"{name} segmentation has unexpected shape/dtype")

    mismatch = dali_seg != local_seg
    per_pair = mismatch.reshape(600, -1).sum(axis=1, dtype=np.int64)
    boundary = np.zeros(dali_seg.shape, dtype=np.bool_)
    horizontal = dali_seg[:, :, 1:] != dali_seg[:, :, :-1]
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    vertical = dali_seg[:, 1:, :] != dali_seg[:, :-1, :]
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    mismatch_count = int(mismatch.sum(dtype=np.int64))
    sites = int(dali_seg.size)
    boundary_sites = int(boundary.sum(dtype=np.int64))
    on_boundary = int((mismatch & boundary).sum(dtype=np.int64))
    per_class = []
    for class_id in range(5):
        class_mask = dali_seg == class_id
        class_sites = int(class_mask.sum(dtype=np.int64))
        class_mismatches = int((class_mask & mismatch).sum(dtype=np.int64))
        per_class.append(
            {
                "class": class_id,
                "sites": class_sites,
                "mismatches": class_mismatches,
                "fraction": class_mismatches / class_sites,
            }
        )

    dali_pose = np.ascontiguousarray(
        dali_obj["pose"].cpu().numpy(), dtype=np.float32
    )
    original_pose = np.ascontiguousarray(
        original_obj["pose"].cpu().numpy(), dtype=np.float32
    )
    if dali_pose.shape != (600, 6) or original_pose.shape != (600, 6):
        raise RuntimeError("pose cache has unexpected shape")
    pose_delta = dali_pose.astype(np.float64) - original_pose.astype(np.float64)
    pose_pair_mse = np.mean(pose_delta * pose_delta, axis=1, dtype=np.float64)
    pose_mse = float(np.mean(pose_delta * pose_delta, dtype=np.float64))

    m1_obj = torch.load(m1_path, map_location="cpu", weights_only=False)
    m1_tensor = m1_obj.get("seg") if isinstance(m1_obj, dict) else None
    if not isinstance(m1_tensor, torch.Tensor):
        raise RuntimeError("M1 segmentation is not a Torch tensor")
    if tuple(m1_tensor.shape) != (600, 384, 512):
        raise RuntimeError("M1 segmentation has unexpected stored shape")
    if m1_tensor.dtype != torch.int64:
        raise RuntimeError("M1 segmentation has unexpected stored dtype")
    m1_seg = np.ascontiguousarray(m1_tensor.cpu().numpy(), dtype=np.uint8)
    m1_seg_sha = numpy_raw_sha(m1_seg)
    del m1_obj, m1_seg

    results = {
        "segmentation": {
            "sites": sites,
            "official_dali_seg_raw_sha256": numpy_raw_sha(dali_seg),
            "local_macos_cpu_av_seg_raw_sha256": numpy_raw_sha(local_seg),
            "pr130_original_seg_raw_sha256": numpy_raw_sha(original_seg),
            "m1_seg_uint8_raw_sha256": m1_seg_sha,
            "mismatches": mismatch_count,
            "fraction": mismatch_count / sites,
            "pairs_with_any_mismatch": int(np.count_nonzero(per_pair)),
            "per_pair_mismatches": numeric_summary(per_pair),
            "dali_boundary_sites": boundary_sites,
            "mismatches_on_dali_boundary": on_boundary,
            "mismatch_share_on_dali_boundary": on_boundary / mismatch_count,
            "per_dali_class": per_class,
            "pr130_original_mismatches_vs_dali": int(
                np.count_nonzero(original_seg != dali_seg)
            ),
            "pr130_original_mismatches_vs_local_av": int(
                np.count_nonzero(original_seg != local_seg)
            ),
        },
        "pose": {
            "official_dali_pose_raw_sha256": numpy_raw_sha(dali_pose),
            "pr130_original_av_like_pose_raw_sha256": numpy_raw_sha(original_pose),
            "elements": int(pose_delta.size),
            "exact_unequal_elements": int(np.count_nonzero(pose_delta)),
            "pairs_with_any_difference": int(
                np.count_nonzero(np.any(pose_delta != 0.0, axis=1))
            ),
            "mse": pose_mse,
            "mae": float(np.mean(np.abs(pose_delta), dtype=np.float64)),
            "max_abs": float(np.max(np.abs(pose_delta))),
            "per_pair_mse": {
                "min": float(pose_pair_mse.min()),
                "q25": float(np.quantile(pose_pair_mse, 0.25)),
                "median": float(np.median(pose_pair_mse)),
                "mean": float(pose_pair_mse.mean()),
                "q75": float(np.quantile(pose_pair_mse, 0.75)),
                "max": float(pose_pair_mse.max()),
            },
            "root_pose_term_separation_scale": float(np.sqrt(10.0 * pose_mse)),
        },
        "containers": {
            "official_dali_uncompressed_bytes": len(dali_raw),
            "official_dali_uncompressed_sha256": sha256_bytes(dali_raw),
            "pr130_original_uncompressed_bytes": len(original_raw),
            "pr130_original_uncompressed_sha256": sha256_bytes(original_raw),
        },
    }
    expected = config["expected"]
    comparisons = {
        "official_dali_seg_raw_sha256": results["segmentation"][
            "official_dali_seg_raw_sha256"
        ],
        "local_macos_cpu_av_seg_raw_sha256": results["segmentation"][
            "local_macos_cpu_av_seg_raw_sha256"
        ],
        "pr130_original_seg_raw_sha256": results["segmentation"][
            "pr130_original_seg_raw_sha256"
        ],
        "m1_seg_uint8_raw_sha256": results["segmentation"][
            "m1_seg_uint8_raw_sha256"
        ],
        "seg_mismatches": mismatch_count,
        "seg_pairs_with_any_mismatch": results["segmentation"][
            "pairs_with_any_mismatch"
        ],
        "seg_boundary_sites": boundary_sites,
        "seg_mismatches_on_boundary": on_boundary,
        "pr130_original_mismatches_vs_dali": results["segmentation"][
            "pr130_original_mismatches_vs_dali"
        ],
        "pr130_original_mismatches_vs_local_av": results["segmentation"][
            "pr130_original_mismatches_vs_local_av"
        ],
        "official_dali_pose_raw_sha256": results["pose"][
            "official_dali_pose_raw_sha256"
        ],
        "pr130_original_av_like_pose_raw_sha256": results["pose"][
            "pr130_original_av_like_pose_raw_sha256"
        ],
        "pose_exact_unequal_elements": results["pose"][
            "exact_unequal_elements"
        ],
        "pose_pairs_with_any_difference": results["pose"][
            "pairs_with_any_difference"
        ],
        "pose_target_mse": results["pose"]["mse"],
        "pose_target_mae": results["pose"]["mae"],
        "pose_target_max_abs": results["pose"]["max_abs"],
    }
    mismatched = {
        key: {"expected": expected[key], "observed": observed}
        for key, observed in comparisons.items()
        if observed != expected[key]
    }
    if mismatched:
        raise RuntimeError(f"target-cache expected-value mismatch: {mismatched}")
    return {
        **common,
        "seed": "N/A_no_RNG",
        "axis": "scorer-free full-population preserved-cache content comparison",
        "measured_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "measurement_classification": "FRESH_DETERMINISTIC_RERUN",
        "verified_inputs": verified_inputs,
        "results": results,
        "expected_checks": comparisons,
        "verdict": "PASS",
        "scope_boundary": config["scope_boundary"],
    }


def import_inflate(source_path: Path):
    source_dir = source_path.parent
    sys.path.insert(0, str(source_dir))
    spec = importlib.util.spec_from_file_location("op1r_pr130_inflate", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(source_dir))
    return module


def receiver_worker(
    config_path: Path,
    raw_path: Path,
    output_path: Path,
    dependency_root: Path,
    pycache_prefix: Path,
) -> None:
    sys.dont_write_bytecode = True
    worker_started_at = utc_now()
    config = json.loads(config_path.read_text())["cpu_first_pair_receiver"]
    source = config["source"]
    configured_dependency_root = Path(config["runtime"]["dependency_root"]).resolve()
    if dependency_root.resolve() != configured_dependency_root:
        raise RuntimeError("receiver dependency-root mismatch")
    observed_dependency_files = {}
    for relative, expected_sha in config["runtime"]["dependency_files"].items():
        dependency_file = configured_dependency_root / relative
        observed_sha = sha256_file(dependency_file)
        if observed_sha != expected_sha:
            raise RuntimeError(f"runtime dependency hash mismatch: {relative}")
        observed_dependency_files[relative] = observed_sha
    sys.path.insert(0, str(configured_dependency_root))
    configured_pycache_prefix = Path(config["runtime"]["pycache_prefix"]).resolve()
    if pycache_prefix.resolve() != configured_pycache_prefix:
        raise RuntimeError("receiver pycache-prefix mismatch")
    if sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve() != configured_pycache_prefix:
        raise RuntimeError("fresh worker did not start with the configured pycache prefix")
    if any(configured_pycache_prefix.rglob("*")):
        raise RuntimeError("configured pycache prefix is not empty before source import")
    source_path = Path(source["inflate_py_path"])
    if sha256_file(source_path) != source["inflate_py_sha256"]:
        raise RuntimeError("inflate.py hash mismatch")
    source_root = source_path.parent.parent
    observed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_head != source["repo_head"]:
        raise RuntimeError(f"source HEAD mismatch: {observed_head}")
    observed_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if observed_status:
        raise RuntimeError("source checkout is dirty")
    dependency_files = {
        "carrier_codec_sha256": "carrier_codec.py",
        "hpac_integer_sha256": "hpac_integer.py",
        "hpac_integer_sparse_sha256": "hpac_integer_sparse.py",
        "integer_model_io_sha256": "integer_model_io.py",
    }
    observed_dependencies = {}
    for hash_key, filename in dependency_files.items():
        dependency_path = source_path.parent / filename
        observed_sha = sha256_file(dependency_path)
        if observed_sha != source[hash_key]:
            raise RuntimeError(f"source dependency hash mismatch: {filename}")
        observed_dependencies[filename] = observed_sha
    archive_path = checked_file(config["archive"])

    import numpy as np
    import torch

    observed_versions = {
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "constriction": package_version("constriction"),
    }
    required_versions = {
        key: config["runtime"][key]
        for key in ("python", "torch", "numpy", "constriction")
    }
    if observed_versions != required_versions:
        raise RuntimeError(
            "receiver runtime version mismatch: "
            f"observed={observed_versions}, required={required_versions}"
        )

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.set_num_threads(config["runtime"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(
        config["runtime"]["torch_manual_seed_for_fully_overwritten_constructor_state"]
    )
    torch.use_deterministic_algorithms(True)
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("receiver default Torch dtype is not float32")
    module = import_inflate(source_path)
    imported_origins = {}
    expected_source_origins = {
        "op1r_pr130_inflate": source_path,
        "carrier_codec": source_path.parent / "carrier_codec.py",
        "hpac_integer": source_path.parent / "hpac_integer.py",
        "hpac_integer_sparse": source_path.parent / "hpac_integer_sparse.py",
        "integer_model_io": source_path.parent / "integer_model_io.py",
        "constriction": configured_dependency_root / "constriction" / "__init__.py",
    }
    for module_name, expected_origin in expected_source_origins.items():
        imported = sys.modules[module_name]
        observed_origin = Path(imported.__file__).resolve()
        if observed_origin != expected_origin.resolve():
            raise RuntimeError(f"unexpected import origin for {module_name}: {observed_origin}")
        if observed_origin.suffix in {".pyc", ".pyo"}:
            raise RuntimeError(f"bytecode import forbidden for {module_name}")
        cached = getattr(imported, "__cached__", None)
        if cached is not None and not Path(cached).resolve().is_relative_to(
            configured_pycache_prefix
        ):
            raise RuntimeError(f"import cache escaped prefix for {module_name}: {cached}")
        imported_origins[module_name] = {
            "origin": str(observed_origin),
            "cached_path": cached,
            "cached_file_exists": bool(cached and Path(cached).exists()),
        }
    extension = sys.modules.get("constriction.constriction")
    if extension is None:
        raise RuntimeError("constriction extension module was not loaded")
    extension_origin = Path(extension.__file__).resolve()
    expected_extension = configured_dependency_root / "constriction" / "constriction.cpython-314-darwin.so"
    if extension_origin != expected_extension.resolve():
        raise RuntimeError(f"unexpected constriction extension: {extension_origin}")
    imported_origins["constriction.constriction"] = {
        "origin": str(extension_origin),
        "cached_path": None,
        "cached_file_exists": False,
    }
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise RuntimeError("unexpected CPR1 ZIP members")
        payload = archive.read("p")
    models_bytes = struct.unpack_from("<I", payload)[0]
    models_raw = lzma.decompress(payload[4 : 4 + models_bytes])
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_pose_bytes = 8 + semantic_bytes + carrier_bytes
    semantic, basis, coeff = module.unpack_semantic_pose(
        models_raw[:semantic_pose_bytes]
    )
    device = torch.device("cpu")
    hpac = module.load_hpac(models_raw[semantic_pose_bytes:], device)
    if module.N != 600:
        raise RuntimeError("source N changed before full-state load")
    saved_n = module.N
    started = time.monotonic()
    try:
        module.N = 1
        tokens = module.decode_tokens(hpac, payload[4 + models_bytes :], device)
        token_array = np.ascontiguousarray(tokens.numpy(), dtype=np.uint8)
        module.render_video(semantic, basis, coeff, tokens, raw_path, device)
    finally:
        module.N = saved_n
    elapsed = time.monotonic() - started
    if module.N != 600:
        raise RuntimeError("source N was not restored")
    receipt = {
        "schema": "ddm_op1r_receiver_worker_receipt.v1",
        "measured_at_utc": worker_started_at,
        "completed_at_utc": utc_now(),
        "workspace_head": git_head(),
        "producer_path": str(SCRIPT.relative_to(ROOT)),
        "producer_sha256": sha256_file(SCRIPT),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "argv": [sys.executable, *sys.argv],
        "elapsed_seconds": elapsed,
        "token_shape": list(token_array.shape),
        "token_sha256": numpy_raw_sha(token_array),
        "output_bytes": raw_path.stat().st_size,
        "output_sha256": sha256_file(raw_path),
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "source_repo_head": observed_head,
        "source_repo_clean": True,
        "source_dependency_sha256": observed_dependencies,
        "runtime_dependency_root": str(configured_dependency_root),
        "runtime_dependency_sha256": observed_dependency_files,
        "runtime_versions": observed_versions,
        "torch_manual_seed_for_fully_overwritten_constructor_state": config[
            "runtime"
        ]["torch_manual_seed_for_fully_overwritten_constructor_state"],
        "python_random_seed": 0,
        "numpy_random_seed": 0,
        "python_dont_write_bytecode": sys.dont_write_bytecode,
        "python_pycache_prefix": str(configured_pycache_prefix),
        "pycache_prefix_empty_after_import": not any(
            configured_pycache_prefix.rglob("*")
        ),
        "imported_origins": imported_origins,
        "output_path": str(raw_path),
    }
    atomic_json(output_path, receipt)


@contextlib.contextmanager
def exclusive_receiver_lock(common: dict[str, Any]):
    scratch = Path(
        "/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/reproducer_scratch"
    )
    scratch.mkdir(parents=True, exist_ok=True)
    lock_path = scratch / "receiver.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("receiver scratch namespace already has a writer") from error
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "schema": "ddm_op1r_receiver_exclusive_lock.v1",
                "pid": os.getpid(),
                "acquired_at_utc": utc_now(),
                "producer_sha256": common["producer_sha256"],
                "config_sha256": common["config_sha256"],
                "argv": common["argv"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def validate_receiver_worker_receipt(
    run: dict[str, Any],
    *,
    argv: list[str],
    raw_path: Path,
    config: dict[str, Any],
    common: dict[str, Any],
    dependency_root: Path,
    pycache_prefix: Path,
) -> None:
    """Fail closed before trusting a fresh or preserved receiver stage."""
    required = {
        "schema": "ddm_op1r_receiver_worker_receipt.v1",
        "producer_path": common["producer_path"],
        "producer_sha256": common["producer_sha256"],
        "config_path": str((ROOT / common["config_path"]).resolve()),
        "config_sha256": common["config_sha256"],
        "argv": argv,
        "source_repo_head": config["source"]["repo_head"],
        "source_repo_clean": True,
        "runtime_dependency_root": str(dependency_root),
        "runtime_dependency_sha256": config["runtime"]["dependency_files"],
        "runtime_versions": {
            key: config["runtime"][key]
            for key in ("python", "torch", "numpy", "constriction")
        },
        "python_dont_write_bytecode": True,
        "python_pycache_prefix": str(pycache_prefix),
        "pycache_prefix_empty_after_import": True,
        "output_path": str(raw_path),
    }
    observed_source_dependencies = {
        "carrier_codec.py": config["source"]["carrier_codec_sha256"],
        "hpac_integer.py": config["source"]["hpac_integer_sha256"],
        "hpac_integer_sparse.py": config["source"]["hpac_integer_sparse_sha256"],
        "integer_model_io.py": config["source"]["integer_model_io_sha256"],
    }
    required["source_dependency_sha256"] = observed_source_dependencies
    mismatched = {
        key: {"expected": value, "observed": run.get(key)}
        for key, value in required.items()
        if run.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"receiver worker receipt mismatch: {mismatched}")
    worker_head = run.get("workspace_head")
    if (
        not isinstance(worker_head, str)
        or len(worker_head) != 40
        or any(character not in "0123456789abcdef" for character in worker_head)
    ):
        raise RuntimeError("receiver worker workspace HEAD is malformed")
    source_path = Path(config["source"]["inflate_py_path"]).resolve()
    expected_origins = {
        "op1r_pr130_inflate": source_path,
        "carrier_codec": source_path.parent / "carrier_codec.py",
        "hpac_integer": source_path.parent / "hpac_integer.py",
        "hpac_integer_sparse": source_path.parent / "hpac_integer_sparse.py",
        "integer_model_io": source_path.parent / "integer_model_io.py",
        "constriction": dependency_root / "constriction" / "__init__.py",
        "constriction.constriction": (
            dependency_root
            / "constriction"
            / "constriction.cpython-314-darwin.so"
        ),
    }
    origins = run.get("imported_origins")
    if not isinstance(origins, dict) or set(origins) != set(expected_origins):
        raise RuntimeError("receiver worker import-origin closure mismatch")
    for name, expected_origin in expected_origins.items():
        item = origins[name]
        if Path(item.get("origin", "")).resolve() != expected_origin.resolve():
            raise RuntimeError(f"receiver worker origin mismatch for {name}")
        cached = item.get("cached_path")
        if cached is not None and not Path(cached).resolve().is_relative_to(
            pycache_prefix
        ):
            raise RuntimeError(f"receiver worker cache path escaped for {name}")
        if item.get("cached_file_exists"):
            raise RuntimeError("receiver worker consumed or wrote cached bytecode")


def reproduce_receiver_locked(
    config_path: Path,
    config: dict[str, Any],
    common: dict[str, Any],
    dependency_root: Path,
    pycache_prefix: Path,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    archive_path = checked_file(config["archive"])
    for key, expected_sha in {
        "inflate_py_path": config["source"]["inflate_py_sha256"],
    }.items():
        if sha256_file(Path(config["source"][key])) != expected_sha:
            raise RuntimeError(f"source hash mismatch: {key}")
    scratch = Path(
        "/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/reproducer_scratch"
    )
    scratch.mkdir(parents=True, exist_ok=True)
    pycache_prefix.mkdir(parents=True, exist_ok=True)
    if any(pycache_prefix.rglob("*")):
        raise RuntimeError("receiver pycache prefix must be empty")
    runs = []
    subprocess_argvs = []
    for run_id in (1, 2):
        raw_path = scratch / f"receiver_run_{run_id}.raw"
        worker_path = scratch / f"receiver_run_{run_id}.json"
        if raw_path.exists():
            raise RuntimeError(f"refusing to overwrite existing raw scratch for run {run_id}")
        argv = [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--receiver-worker",
            "--scratch-raw",
            str(raw_path),
            "--worker-output",
            str(worker_path),
            "--dependency-root",
            str(dependency_root),
            "--pycache-prefix",
            str(pycache_prefix),
        ]
        subprocess_argvs.append(argv)
        resumed_stage = worker_path.exists()
        if resumed_stage:
            run = json.loads(worker_path.read_text())
        else:
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = "0"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)
            completed = subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"receiver worker {run_id} failed rc={completed.returncode}: "
                    f"{completed.stderr[-4000:]}"
                )
            run = json.loads(worker_path.read_text())
            run["worker_stdout_sha256"] = sha256_bytes(completed.stdout.encode())
            run["worker_stderr_sha256"] = sha256_bytes(completed.stderr.encode())
        validate_receiver_worker_receipt(
            run,
            argv=argv,
            raw_path=raw_path,
            config=config,
            common=common,
            dependency_root=dependency_root,
            pycache_prefix=pycache_prefix,
        )
        run["resumed_from_preserved_stage_receipt"] = resumed_stage
        run["worker_receipt"] = {
            "path": str(worker_path),
            "bytes": worker_path.stat().st_size,
            "sha256": sha256_file(worker_path),
            "preserved": True,
        }
        run["cleanup"] = {
            "path": str(raw_path),
            "bytes": run["output_bytes"],
            "sha256": run["output_sha256"],
            "action": "success_delete_after_hash_and_receipt",
            "executed": True,
            "rebuildable": True,
            "reason": "Exact argv/config/source/input hashes and output digest are retained; the 6.1 MB raw is deterministic scratch.",
        }
        runs.append(run)
        if not resumed_stage:
            raw_path.unlink()

    expected = config["expected"]
    for run in runs:
        for key in (
            "token_shape",
            "token_sha256",
            "output_bytes",
            "output_sha256",
        ):
            if run[key] != expected[key]:
                raise RuntimeError(
                    f"receiver mismatch for {key}: {run[key]} != {expected[key]}"
                )
    repeat_equal = all(
        runs[0][key] == runs[1][key]
        for key in ("token_shape", "token_sha256", "output_bytes", "output_sha256")
    )
    if repeat_equal is not expected["repeat_equal"]:
        raise RuntimeError("receiver repeat-equality mismatch")
    if any(scratch.glob("receiver_run_*.raw")):
        raise RuntimeError("receiver raw scratch cleanup incomplete")
    resumed_stage_count = sum(
        bool(run["resumed_from_preserved_stage_receipt"]) for run in runs
    )
    return {
        **common,
        "axis": "macOS-CPU advisory scorer-free real first-pair TOY-BRACKET",
        "device": "cpu",
        "torch_threads": config["runtime"]["torch_threads"],
        "pythonhashseed": 0,
        "torch_deterministic_algorithms": True,
        "torch_manual_seed_for_fully_overwritten_constructor_state": config[
            "runtime"
        ]["torch_manual_seed_for_fully_overwritten_constructor_state"],
        "seed": 0,
        "measured_at_utc": min(run["measured_at_utc"] for run in runs),
        "completed_at_utc": max(run["completed_at_utc"] for run in runs),
        "aggregate_verified_at_utc": started_at,
        "elapsed_seconds": time.monotonic() - started,
        "measurement_classification": (
            "VERIFIED_PRESERVED_FRESH_STAGE_RECEIPTS"
            if resumed_stage_count
            else "FRESH_DETERMINISTIC_RERUN"
        ),
        "resumed_stage_receipt_count": resumed_stage_count,
        "source": config["source"],
        "dependency_root": str(dependency_root),
        "pycache_prefix": str(pycache_prefix),
        "pycache_prefix_empty_after_runs": not any(pycache_prefix.rglob("*")),
        "archive": {
            **config["archive"],
            "verified_path": str(archive_path),
        },
        "subprocess_argvs": subprocess_argvs,
        "runs": runs,
        "repeat_equal": repeat_equal,
        "scratch_cleanup": "PASS_no_receiver_raw_files_remain; JSON stage receipts preserved",
        "verdict": "PASS",
        "scope_boundary": config["scope_boundary"],
    }


def reproduce_receiver(
    config_path: Path,
    config: dict[str, Any],
    common: dict[str, Any],
    dependency_root: Path,
    pycache_prefix: Path,
) -> dict[str, Any]:
    with exclusive_receiver_lock(common) as lock_path:
        result = reproduce_receiver_locked(
            config_path,
            config,
            common,
            dependency_root,
            pycache_prefix,
        )
        result["exclusive_lock"] = {
            "path": str(lock_path),
            "single_writer_enforced": True,
        }
        return result


def lzma_filter(spec: dict[str, Any]) -> dict[str, Any]:
    mode = {"fast": lzma.MODE_FAST, "normal": lzma.MODE_NORMAL}[spec["mode"]]
    match_finder = {
        "hc3": lzma.MF_HC3,
        "hc4": lzma.MF_HC4,
        "bt2": lzma.MF_BT2,
        "bt3": lzma.MF_BT3,
        "bt4": lzma.MF_BT4,
    }[spec["mf"]]
    return {
        "id": lzma.FILTER_LZMA2,
        "dict_size": spec["dict_size"],
        "lc": spec["lc"],
        "lp": spec["lp"],
        "pb": spec["pb"],
        "mode": mode,
        "nice_len": spec["nice_len"],
        "mf": match_finder,
        "depth": spec["depth"],
    }


def compress_filter(raw: bytes, spec: dict[str, Any]) -> bytes:
    return lzma.compress(raw, format=lzma.FORMAT_XZ, filters=[lzma_filter(spec)])


def grid_one_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for dictionary in config["dict_sizes"]:
        for lc in range(5):
            for lp in range(5 - lc):
                for pb in config["pb"]:
                    specs.append({
                        "dict_size": dictionary,
                        "lc": lc,
                        "lp": lp,
                        "pb": pb,
                        "mode": config["mode"],
                        "nice_len": config["nice_len"],
                        "mf": config["mf"],
                        "depth": config["depth"],
                    })
    if len(specs) != config["valid_rows"]:
        raise RuntimeError(f"grid 1 row mismatch: {len(specs)}")
    return specs


def grid_two_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for dictionary in config["dict_sizes"]:
        for mode in config["modes"]:
            for mf in config["match_finders"]:
                for nice_len in config["nice_len"]:
                    for depth in config["depth"]:
                        specs.append({
                            "dict_size": dictionary,
                            "lc": config["lc"],
                            "lp": config["lp"],
                            "pb": config["pb"],
                            "mode": mode,
                            "nice_len": nice_len,
                            "mf": mf,
                            "depth": depth,
                        })
    if len(specs) != config["valid_rows"]:
        raise RuntimeError(f"grid 2 row mismatch: {len(specs)}")
    return specs


def stage_receipt_path(state_path: Path, stage: str) -> Path:
    return state_path.with_name(f"{state_path.stem}.{stage}_complete.json")


def row_sizes_digest(row_sizes: list[int]) -> str:
    return sha256_bytes(
        json.dumps(row_sizes, separators=(",", ":")).encode("utf-8")
    )


def validate_grid_progress(
    payload: dict[str, Any],
    *,
    identity: dict[str, Any],
    specs: list[dict[str, Any]],
    schema: str,
    allowed_verdicts: set[str],
) -> tuple[int, list[int], int | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if payload.get("schema") != schema:
        raise RuntimeError(f"grid receipt schema mismatch: {payload.get('schema')}")
    if any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError("grid receipt identity mismatch")
    completed_rows = payload.get("completed_rows")
    valid_roundtrips = payload.get("valid_roundtrips")
    if not isinstance(completed_rows, int) or not 0 <= completed_rows <= len(specs):
        raise RuntimeError("grid completed-row count is invalid")
    if valid_roundtrips != completed_rows:
        raise RuntimeError("grid valid-roundtrip count differs from cursor")
    verdict = payload.get("verdict")
    if verdict not in allowed_verdicts:
        raise RuntimeError(f"grid receipt verdict is invalid: {verdict}")
    if completed_rows < len(specs) and verdict != "IN_PROGRESS":
        raise RuntimeError("partial grid does not have IN_PROGRESS verdict")
    if completed_rows == len(specs) and verdict not in {
        "ROWS_COMPLETE_PENDING_STAGE_RECEIPT",
        "PASS",
    }:
        raise RuntimeError("complete grid has an inconsistent verdict")
    row_sizes = payload.get("row_sizes")
    if (
        not isinstance(row_sizes, list)
        or len(row_sizes) != completed_rows
        or any(not isinstance(size, int) or size <= 0 for size in row_sizes)
    ):
        raise RuntimeError("grid row-size ledger is invalid")
    if payload.get("row_sizes_sha256") != row_sizes_digest(row_sizes):
        raise RuntimeError("grid row-size ledger digest mismatch")
    computed_best = min(row_sizes) if row_sizes else None
    computed_specs = [
        specs[index] for index, size in enumerate(row_sizes) if size == computed_best
    ] if computed_best is not None else []
    if payload.get("best_bytes") != computed_best:
        raise RuntimeError("grid best-byte summary disagrees with row ledger")
    if payload.get("best_specs") != computed_specs:
        raise RuntimeError("grid best-spec summary disagrees with row ledger")
    if schema == "ddm_op1r_xz_stage_receipt.v2" and payload.get(
        "best_tie_count"
    ) != len(computed_specs):
        raise RuntimeError("grid best-tie count disagrees with row ledger")
    launch_history = payload.get("launch_history")
    if not isinstance(launch_history, list) or not launch_history:
        raise RuntimeError("grid launch history is missing")
    return completed_rows, row_sizes, computed_best, computed_specs, launch_history


def run_resumable_grid(
    raw: bytes,
    specs: list[dict[str, Any]],
    state_path: Path,
    stage: str,
    common: dict[str, Any],
) -> dict[str, Any]:
    """Run one grid with an atomic cursor after every verified roundtrip."""
    started = time.monotonic()
    raw_sha = sha256_bytes(raw)
    grid_sha = sha256_bytes(
        json.dumps(specs, sort_keys=True, separators=(",", ":")).encode()
    )
    completed_path = stage_receipt_path(state_path, stage)
    identity = {
        "stage": stage,
        "raw_sha256": raw_sha,
        "grid_sha256": grid_sha,
        "total_rows": len(specs),
        "producer_sha256": common["producer_sha256"],
        "config_sha256": common["config_sha256"],
        "xz_runtime_fingerprint": common["xz_runtime_fingerprint"],
    }
    current_launch = {
        "invocation_started_at_utc": common["invocation_started_at_utc"],
        "argv": common["argv"],
        "workspace_head": common["workspace_head"],
        "runtime_fingerprint": common["xz_runtime_fingerprint"],
    }
    if completed_path.exists():
        completed = json.loads(completed_path.read_text())
        completed_rows, _, _, _, _ = validate_grid_progress(
            completed,
            identity=identity,
            specs=specs,
            schema="ddm_op1r_xz_stage_receipt.v2",
            allowed_verdicts={"PASS"},
        )
        if completed_rows != len(specs):
            raise RuntimeError(f"completed receipt is partial: {completed_path}")
        return {**completed, "resumed_from_stage_receipt": True}

    next_row = 0
    best_size = None
    best_specs: list[dict[str, Any]] = []
    row_sizes: list[int] = []
    launch_history = [current_launch]
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("stage") == stage:
            next_row, row_sizes, best_size, best_specs, launch_history = (
                validate_grid_progress(
                    state,
                    identity=identity,
                    specs=specs,
                    schema="ddm_op1r_xz_resume_state.v2",
                    allowed_verdicts={
                        "IN_PROGRESS",
                        "ROWS_COMPLETE_PENDING_STAGE_RECEIPT",
                        "PASS",
                    },
                )
            )
            if launch_history[-1] != current_launch:
                launch_history.append(current_launch)

    resumed_from_row = next_row
    for row_index in range(next_row, len(specs)):
        spec = specs[row_index]
        encoded = compress_filter(raw, spec)
        if lzma.decompress(encoded) != raw:
            raise RuntimeError(f"roundtrip failed at {stage} row {row_index}")
        size = len(encoded)
        row_sizes.append(size)
        if best_size is None or size < best_size:
            best_size = size
            best_specs = [spec]
        elif size == best_size:
            best_specs.append(spec)
        atomic_json(
            state_path,
            {
                **identity,
                "schema": "ddm_op1r_xz_resume_state.v2",
                "completed_rows": row_index + 1,
                "valid_roundtrips": row_index + 1,
                "row_sizes": row_sizes,
                "row_sizes_sha256": row_sizes_digest(row_sizes),
                "best_bytes": best_size,
                "best_specs": best_specs,
                "launch_history": launch_history,
                "updated_at_utc": utc_now(),
                "verdict": (
                    "ROWS_COMPLETE_PENDING_STAGE_RECEIPT"
                    if row_index + 1 == len(specs)
                    else "IN_PROGRESS"
                ),
            },
        )

    result = {
        **identity,
        "schema": "ddm_op1r_xz_stage_receipt.v2",
        "completed_rows": len(specs),
        "valid_roundtrips": len(specs),
        "row_sizes": row_sizes,
        "row_sizes_sha256": row_sizes_digest(row_sizes),
        "best_bytes": best_size,
        "best_specs": best_specs,
        "best_tie_count": len(best_specs),
        "launch_history": launch_history,
        "resumed_from_row": resumed_from_row,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "completed_at_utc": utc_now(),
        "verdict": "PASS",
    }
    atomic_json(completed_path, result)
    atomic_json(
        state_path,
        {
            **identity,
            "schema": "ddm_op1r_xz_resume_state.v2",
            "completed_rows": len(specs),
            "valid_roundtrips": len(specs),
            "row_sizes": row_sizes,
            "row_sizes_sha256": row_sizes_digest(row_sizes),
            "best_bytes": best_size,
            "best_specs": best_specs,
            "launch_history": launch_history,
            "updated_at_utc": utc_now(),
            "verdict": "PASS",
        },
    )
    return result


def deterministic_zip(payload: bytes, config: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo(config["member"], (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)
    return buffer.getvalue()


@contextlib.contextmanager
def exclusive_xz_lock(state_path: Path, common: dict[str, Any]):
    """Hold a fail-closed single-writer lock for one resume namespace."""
    lock_path = state_path.with_name(f"{state_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"XZ resume namespace already has a writer: {lock_path}") from error
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "schema": "ddm_op1r_xz_exclusive_lock.v1",
                "pid": os.getpid(),
                "acquired_at_utc": utc_now(),
                "state_path": str(state_path),
                "producer_sha256": common["producer_sha256"],
                "config_sha256": common["config_sha256"],
                "argv": common["argv"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def validate_xz_runtime(
    expected: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    version_lines = observed["xz_version_stdout"].splitlines()
    normalized = {
        "python": observed["python"],
        "python_build": list(observed["python_build"]),
        "python_compiler": observed["python_compiler"],
        "python_executable": observed["python_executable"],
        "python_executable_resolved": observed["python_executable_resolved"],
        "python_executable_sha256": observed["python_executable_sha256"],
        "python_lzma_module": observed["lzma_module"],
        "python_lzma_module_sha256": observed["lzma_module_sha256"],
        "_lzma_extension": observed["_lzma_extension"],
        "_lzma_extension_sha256": observed["_lzma_extension_sha256"],
        "linked_liblzma": observed["linked_liblzma"],
        "linked_liblzma_sha256": observed["linked_liblzma_sha256"],
        "linked_liblzma_otool_line": observed["linked_liblzma_otool_line"],
        "xz_utils": (
            version_lines[0].removeprefix("xz (XZ Utils) ")
            if len(version_lines) >= 1
            else None
        ),
        "liblzma": (
            version_lines[1].removeprefix("liblzma ")
            if len(version_lines) >= 2
            else None
        ),
    }
    required = {
        "python": expected["python"],
        "python_build": expected["python_build"],
        "python_compiler": expected["python_compiler"],
        "python_executable": expected["python_executable"],
        "python_executable_resolved": expected["python_executable_resolved"],
        "python_executable_sha256": expected["python_executable_sha256"],
        "python_lzma_module": expected["python_lzma_module"],
        "python_lzma_module_sha256": expected["python_lzma_module_sha256"],
        "_lzma_extension": expected["_lzma_extension"],
        "_lzma_extension_sha256": expected["_lzma_extension_sha256"],
        "linked_liblzma": expected["linked_liblzma"],
        "linked_liblzma_sha256": expected["linked_liblzma_sha256"],
        "linked_liblzma_otool_line": expected["linked_liblzma_otool_line"],
        "xz_utils": expected["xz_utils"],
        "liblzma": expected["liblzma"],
    }
    if normalized != required:
        raise RuntimeError(
            f"XZ runtime mismatch: observed={normalized}, required={required}"
        )
    fingerprint = sha256_bytes(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    )
    return {
        "normalized": normalized,
        "fingerprint_sha256": fingerprint,
        "config_match": True,
    }


def reproduce_xz_locked(
    config: dict[str, Any], common: dict[str, Any], resume_from: Path
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    runtime_validation = validate_xz_runtime(config["runtime"], common["runtime"])
    grid_common = {
        **common,
        "xz_runtime_fingerprint": runtime_validation["fingerprint_sha256"],
    }
    input_spec = config["input"]
    archive_path = checked_file(
        {
            "path": input_spec["archive_path"],
            "sha256": input_spec["archive_sha256"],
        }
    )
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != [input_spec["member"]]:
            raise RuntimeError("unexpected source archive members")
        original_payload = archive.read(input_spec["member"])
    model_bytes = struct.unpack_from("<I", original_payload)[0]
    incumbent_xz = original_payload[4 : 4 + model_bytes]
    tokens = original_payload[4 + model_bytes :]
    raw = lzma.decompress(incumbent_xz)
    if len(raw) != input_spec["raw_model_bundle_bytes"]:
        raise RuntimeError("raw model size mismatch")
    if sha256_bytes(raw) != input_spec["raw_model_bundle_sha256"]:
        raise RuntimeError("raw model hash mismatch")

    first_grid = run_resumable_grid(
        raw,
        grid_one_specs(config["grid_1"]),
        resume_from,
        "grid_1",
        grid_common,
    )
    second_grid = run_resumable_grid(
        raw,
        grid_two_specs(config["grid_2_canonical_rerun"]),
        resume_from,
        "grid_2",
        grid_common,
    )
    canonical = config["grid_2_canonical_rerun"]["canonical_rerun_best_filter"]
    canonical_stream = compress_filter(raw, canonical)
    if lzma.decompress(canonical_stream) != raw:
        raise RuntimeError("canonical grid-2 winner roundtrip failed")
    second_grid["canonical_best_filter"] = canonical
    second_grid["canonical_xz_sha256"] = sha256_bytes(canonical_stream)
    candidate_xz = compress_filter(raw, config["candidate_filter"])
    candidate_payload = struct.pack("<I", len(candidate_xz)) + candidate_xz + tokens
    candidate_archive = deterministic_zip(candidate_payload, config["zip_rebuild"])
    expected_xz = config["candidate_xz"]
    expected = config["expected"]
    if first_grid["best_bytes"] != expected["grid_1_best_bytes"]:
        raise RuntimeError("grid-1 best-byte mismatch")
    if second_grid["best_bytes"] != expected["grid_2_best_bytes"]:
        raise RuntimeError("grid-2 best-byte mismatch")
    if (
        second_grid["canonical_xz_sha256"]
        != expected["canonical_grid_2_xz_sha256"]
    ):
        raise RuntimeError("canonical grid-2 stream mismatch")
    checks = {
        "candidate_xz_bytes": len(candidate_xz),
        "candidate_xz_sha256": sha256_bytes(candidate_xz),
        "candidate_archive_bytes": len(candidate_archive),
        "candidate_archive_sha256": sha256_bytes(candidate_archive),
        "member_bytes": len(candidate_payload),
        "member_sha256": sha256_bytes(candidate_payload),
        "raw_parseback_sha256": sha256_bytes(lzma.decompress(candidate_xz)),
        "tokens_parseback_equal": candidate_payload[4 + len(candidate_xz) :] == tokens,
        "raw_models_parseback_equal": lzma.decompress(candidate_xz) == raw,
    }
    required = {
        "candidate_xz_bytes": expected_xz["bytes"],
        "candidate_xz_sha256": expected_xz["sha256"],
        "candidate_archive_bytes": expected["candidate_archive_bytes"],
        "candidate_archive_sha256": expected["candidate_archive_sha256"],
        "member_bytes": expected["member_bytes"],
        "member_sha256": expected["member_sha256"],
        "raw_parseback_sha256": expected_xz["raw_parseback_sha256"],
        "tokens_parseback_equal": expected["tokens_parseback_equal"],
        "raw_models_parseback_equal": expected["raw_models_parseback_equal"],
    }
    mismatched = {
        key: {"expected": required[key], "observed": observed}
        for key, observed in checks.items()
        if observed != required[key]
    }
    if mismatched:
        raise RuntimeError(f"XZ expected-value mismatch: {mismatched}")
    retained = Path(expected["candidate_archive_path"])
    if sha256_file(retained) != checks["candidate_archive_sha256"]:
        raise RuntimeError("retained candidate archive differs from reconstruction")
    grids = (first_grid, second_grid)
    reused_stage_count = sum(
        bool(grid.get("resumed_from_stage_receipt")) for grid in grids
    )
    resumed_row_count = sum(int(grid.get("resumed_from_row", 0)) for grid in grids)
    if reused_stage_count:
        measurement_classification = "VERIFIED_PRESERVED_FRESH_STAGE_RECEIPTS"
    elif resumed_row_count:
        measurement_classification = "RESUMED_DETERMINISTIC_RERUN"
    else:
        measurement_classification = "FRESH_DETERMINISTIC_RERUN"
    return {
        **common,
        "seed": "N/A_no_RNG",
        "axis": "scorer-free exact CPR1 model-bundle compression",
        "measured_at_utc": min(
            grid["launch_history"][0]["invocation_started_at_utc"]
            for grid in grids
        ),
        "completed_at_utc": max(grid["completed_at_utc"] for grid in grids),
        "aggregate_verified_at_utc": started_at,
        "elapsed_seconds": time.monotonic() - started,
        "measurement_classification": measurement_classification,
        "reused_stage_receipt_count": reused_stage_count,
        "resumed_row_count": resumed_row_count,
        "source_archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "member_bytes": len(original_payload),
            "member_sha256": sha256_bytes(original_payload),
            "incumbent_xz_bytes": len(incumbent_xz),
            "incumbent_xz_sha256": sha256_bytes(incumbent_xz),
            "token_bytes": len(tokens),
            "token_sha256": sha256_bytes(tokens),
        },
        "grid_1": first_grid,
        "grid_2": second_grid,
        "runtime_validation": runtime_validation,
        "resume_state": {
            "path": str(resume_from),
            "grid_1_stage_receipt": str(stage_receipt_path(resume_from, "grid_1")),
            "grid_2_stage_receipt": str(stage_receipt_path(resume_from, "grid_2")),
        },
        "candidate_checks": checks,
        "retained_candidate_path": str(retained),
        "verdict": "PASS",
    }


def reproduce_xz(
    config: dict[str, Any], common: dict[str, Any], resume_from: Path
) -> dict[str, Any]:
    with exclusive_xz_lock(resume_from, common) as lock_path:
        result = reproduce_xz_locked(config, common, resume_from)
        result["exclusive_lock"] = {
            "path": str(lock_path),
            "single_writer_enforced": True,
        }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--section",
        choices=("target-cache", "receiver", "xz", "all"),
        default="all",
    )
    parser.add_argument("--hardware-label", default="UNSPECIFIED")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--pycache-prefix", type=Path)
    parser.add_argument("--receiver-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scratch-raw", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if args.receiver_worker:
        if args.scratch_raw is None or args.worker_output is None:
            raise SystemExit("receiver worker requires scratch/output paths")
        if args.dependency_root is None or args.pycache_prefix is None:
            raise SystemExit(
                "receiver worker requires --dependency-root and --pycache-prefix"
            )
        receiver_worker(
            config_path,
            args.scratch_raw,
            args.worker_output,
            args.dependency_root,
            args.pycache_prefix,
        )
        return
    if args.output is None:
        raise SystemExit("--output is required")
    if args.section in ("xz", "all") and args.resume_from is None:
        raise SystemExit("--resume-from is required for the XZ sweep")
    if args.section in ("receiver", "all") and (
        args.dependency_root is None or args.pycache_prefix is None
    ):
        raise SystemExit(
            "--dependency-root and --pycache-prefix are required for the receiver"
        )

    started_at = utc_now()
    config = json.loads(config_path.read_text())
    head = git_head()
    producer_sha = sha256_file(SCRIPT)
    argv = [sys.executable, *sys.argv]
    runtime = runtime_receipt(args.hardware_label, args.dependency_root)
    common = {
        "invocation_started_at_utc": started_at,
        "workspace_head": head,
        "producer_path": str(SCRIPT.relative_to(ROOT)),
        "producer_sha256": producer_sha,
        "argv": argv,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "seed_policy": runtime["seed_policy"],
        "runtime": runtime,
    }
    sections: dict[str, Any] = {}
    selected = (
        ("target-cache", "receiver", "xz")
        if args.section == "all"
        else (args.section,)
    )
    if "target-cache" in selected:
        sections["target_cache"] = reproduce_target_cache(
            config["target_cache_comparison"], common
        )
    if "receiver" in selected:
        sections["receiver"] = reproduce_receiver(
            config_path,
            config["cpu_first_pair_receiver"],
            common,
            args.dependency_root.resolve(),
            args.pycache_prefix.resolve(),
        )
    if "xz" in selected:
        sections["xz"] = reproduce_xz(
            config["xz_filter_race"], common, args.resume_from.resolve()
        )
    section_classifications = {
        value.get("measurement_classification", "FRESH_DETERMINISTIC_RERUN")
        for value in sections.values()
    }
    top_classification = (
        "FRESH_DETERMINISTIC_RERUN"
        if section_classifications == {"FRESH_DETERMINISTIC_RERUN"}
        else "AGGREGATE_DETERMINISTIC_RERUN_WITH_VERIFIED_OR_RESUMED_STAGE_RECEIPTS"
    )
    receipt = {
        "schema": "ddm_op1r_reproduction_receipt.v1",
        "original_one_off_command_provenance": "UNDETERMINED",
        "classification": top_classification,
        "section_classifications": sorted(section_classifications),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "workspace_head": head,
        "producer": {
            "path": common["producer_path"],
            "sha256": producer_sha,
        },
        "argv": argv,
        "config": {
            "path": common["config_path"],
            "sha256": common["config_sha256"],
            "schema": config["schema"],
        },
        "runtime": common["runtime"],
        "sections": sections,
        "verdict": "PASS" if all(v["verdict"] == "PASS" for v in sections.values()) else "FAIL",
        "score_claim": False,
        "pointer_moved": False,
    }
    atomic_json(args.output.resolve(), receipt)


if __name__ == "__main__":
    main()
