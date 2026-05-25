# SPDX-License-Identifier: MIT
"""Receiver/runtime compiler for tensor-factorize materializers.

The tensor-factorize materializer replaces a NumPy ``.npy`` tensor member with
an ``.npz`` packet containing low-rank factors. This module builds the matching
cooperative receiver: a portable inflate-runtime adapter that reconstructs a
shadow ``.npy`` member from the factor packet, then delegates to the original
runtime.

This is receiver evidence, not score authority. A ready adapter proves the
transformed archive is consumed by a runtime path; exact-eval promotion still
belongs to the normal contest-auth gates.
"""

from __future__ import annotations

import ast
import io
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.optimization.family_agnostic_materializers import (
    FALSE_AUTHORITY,
    TENSOR_FACTORIZE_MATERIALIZER_ID,
    TENSOR_FACTORIZE_PROOF_KIND,
    TENSOR_FACTORIZE_RECEIVER_CONTRACT_KIND,
    TENSOR_FACTORIZE_SCHEMA,
    TENSOR_FACTORIZE_TARGET_KIND,
)
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import (
    artifact_dir_transaction,
    read_json,
    sha256_bytes,
    sha256_file,
    tree_sha256,
    write_json_artifact,
)

RUNTIME_MANIFEST_SCHEMA = "tensor_factorize_receiver_runtime.v1"
RUNTIME_PROOF_KIND = TENSOR_FACTORIZE_PROOF_KIND
RUNTIME_COMPILER_ID = "tensor_factorize_receiver_compiler.v1"
RUNTIME_PY = "tensor_factorize_receiver_runtime.py"
WRAPPER_SH = "inflate.sh"
SOURCE_RUNTIME_DIR = "source_runtime"
ARCHIVE_NAME = "archive.zip"

_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cfg",
    ".env",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_EXCLUDED_NAMES = {".DS_Store", "__pycache__", "eval_runs"}
_EXCLUDED_SUFFIXES = {".log", ".pyc", ".pyo", ".zip"}
_SKIPPED_DATA_SUFFIXES = {".bin", ".br", ".mkv", ".pt", ".raw"}


class TensorFactorizeReceiverError(ValueError):
    """Raised when a tensor-factorize receiver cannot be compiled safely."""


def reconstruct_tensor_factorize_archive_bytes(
    candidate_archive: str | Path,
    *,
    member_name: str,
) -> bytes:
    """Return archive bytes with a tensor-factorized member expanded to ``.npy``."""

    archive = Path(candidate_archive)
    with tempfile.SpooledTemporaryFile(max_size=16 << 20) as output:
        with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(output, "w") as target:
            for info in source.infolist():
                if info.is_dir():
                    target.mkdir(_copy_zip_info(info))
                    continue
                payload = source.read(info.filename)
                replacement = (
                    tensor_factorize_packet_to_npy_bytes(payload)["npy_payload"]
                    if info.filename == member_name
                    else payload
                )
                target.writestr(_copy_zip_info(info), replacement)
        output.seek(0)
        return output.read()


def write_reconstructed_tensor_factorize_archive(
    candidate_archive: str | Path,
    output_archive: str | Path,
    *,
    member_name: str,
) -> dict[str, Any]:
    """Write a shadow archive and return its byte identity."""

    payload = reconstruct_tensor_factorize_archive_bytes(
        candidate_archive,
        member_name=member_name,
    )
    target = Path(output_archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {
        "path": target.as_posix(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def tensor_factorize_packet_to_npy_bytes(payload: bytes) -> dict[str, Any]:
    """Decode a tensor-factorize NPZ packet into source-runtime ``.npy`` bytes."""

    import numpy as np

    with np.load(io.BytesIO(payload), allow_pickle=False) as packet:
        for key in ("metadata", "u", "s", "vt"):
            if key not in packet.files:
                raise TensorFactorizeReceiverError(
                    f"tensor factorize packet missing key: {key}"
                )
        metadata = _parse_packet_metadata(packet["metadata"])
        if metadata.get("schema") != TENSOR_FACTORIZE_SCHEMA:
            raise TensorFactorizeReceiverError("tensor factorize packet schema mismatch")
        u = np.asarray(packet["u"], dtype=np.float32)
        s = np.asarray(packet["s"], dtype=np.float32)
        vt = np.asarray(packet["vt"], dtype=np.float32)
    source_shape = tuple(int(dim) for dim in metadata.get("source_shape") or ())
    if len(source_shape) != 2:
        raise TensorFactorizeReceiverError("tensor factorize source_shape must be 2D")
    source_dtype = np.dtype(str(metadata.get("source_dtype") or "float32"))
    reconstruction = ((u * s) @ vt).reshape(source_shape).astype(source_dtype)
    out = io.BytesIO()
    np.save(out, reconstruction, allow_pickle=False)
    return {
        "schema": "tensor_factorize_packet_decode.v1",
        "metadata": metadata,
        "tensor": reconstruction,
        "npy_payload": out.getvalue(),
    }


def build_tensor_factorize_receiver_runtime(
    *,
    source_runtime_dir: str | Path,
    candidate_manifest: str | Path | Mapping[str, Any],
    runtime_dir_out: str | Path,
    runtime_manifest_out: str | Path | None = None,
    repo_root: str | Path | None = None,
    allow_runtime_sidecars: bool = False,
    allow_overwrite: bool = False,
    expected_existing_runtime_tree_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Compile a self-contained receiver runtime for a tensor-factorize candidate."""

    repo = Path(repo_root).resolve(strict=False) if repo_root is not None else Path.cwd()
    source_runtime = _resolve(source_runtime_dir, repo=repo)
    if not source_runtime.is_dir():
        raise TensorFactorizeReceiverError(f"source runtime dir not found: {source_runtime}")
    candidate = _load_candidate_manifest(candidate_manifest, repo=repo)
    _require_tensor_factorize_candidate(candidate)
    runtime_dir = _resolve(runtime_dir_out, repo=repo)
    manifest_path = (
        _resolve(runtime_manifest_out, repo=repo)
        if runtime_manifest_out is not None
        else runtime_dir.with_name(f"{runtime_dir.name}.tensor_factorize_receiver_runtime.json")
    )

    blockers: list[str] = []
    with artifact_dir_transaction(
        runtime_dir,
        allow_overwrite=allow_overwrite,
        expected_existing_tree_sha256=expected_existing_runtime_tree_sha256,
        min_free_bytes=min_free_bytes,
    ) as txn:
        copied_runtime_manifest = _copy_runtime_tree(
            source_runtime,
            txn.staging / SOURCE_RUNTIME_DIR,
            allow_runtime_sidecars=allow_runtime_sidecars,
        )
        blockers.extend(copied_runtime_manifest["blockers"])
        _write_receiver_runtime_py(txn.staging / RUNTIME_PY)
        _write_receiver_wrapper(txn.staging / WRAPPER_SH)

    runtime_tree_sha = tree_sha256(runtime_dir)
    candidate_archive = _candidate_archive_path(candidate, repo=repo)
    selected_member = _selected_member_name(candidate)
    candidate_member = _candidate_member_record(candidate)
    source_archive = _source_archive_path(candidate, repo=repo)
    source_member = _source_member_record(candidate)
    identity = tensor_factorize_reconstruction_identity(
        candidate_archive,
        member_name=selected_member,
        source_archive=source_archive,
        source_member_name=selected_member,
        tolerances=_candidate_tolerances(candidate),
    )
    blockers.extend(identity["blockers"])
    source_entrypoint = _source_runtime_entrypoint(runtime_dir / SOURCE_RUNTIME_DIR)
    blockers.extend(source_entrypoint["blockers"])

    manifest: dict[str, Any] = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "compiler_id": RUNTIME_COMPILER_ID,
        "runtime_adapter_kind": "shadow_archive_tensor_factorize_receiver",
        "runtime_adapter_ready": not blockers,
        "ready_for_receiver_verification": not blockers,
        "candidate_manifest_schema": candidate.get("schema"),
        "candidate_manifest_path": (
            _resolve(candidate_manifest, repo=repo).as_posix()
            if isinstance(candidate_manifest, (str, Path))
            else None
        ),
        "candidate_archive": {
            "path": candidate_archive.as_posix(),
            "bytes": candidate_archive.stat().st_size,
            "sha256": sha256_file(candidate_archive),
        },
        "candidate_member": dict(candidate_member),
        "candidate_archive_sha256": sha256_file(candidate_archive),
        "candidate_member_sha256": candidate_member.get("sha256"),
        "source_archive": {
            "path": source_archive.as_posix(),
            "bytes": source_archive.stat().st_size,
            "sha256": sha256_file(source_archive),
        },
        "source_member": dict(source_member),
        "selected_member_name": selected_member,
        "factorization": dict(candidate.get("factorization") or {}),
        "reconstruction_identity": identity,
        "runtime_dir": runtime_dir.as_posix(),
        "runtime_tree_sha256": runtime_tree_sha,
        "source_runtime_dir": source_runtime.as_posix(),
        "source_runtime_tree_sha256": tree_sha256(source_runtime),
        "copied_source_runtime": copied_runtime_manifest,
        "entrypoint": {
            "inflate_sh": (runtime_dir / WRAPPER_SH).as_posix(),
            "receiver_runtime_py": (runtime_dir / RUNTIME_PY).as_posix(),
            "source_runtime": source_entrypoint,
        },
        "runtime_behavior": {
            "mode": "shadow_archive_tensor_reconstruction_then_delegate",
            "archive_name": ARCHIVE_NAME,
            "loads_scorer": False,
            "network_required": False,
            "external_state_required": False,
        },
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    write_json_artifact(
        manifest_path,
        manifest,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=(
            sha256_file(manifest_path)
            if manifest_path.exists() and allow_overwrite
            else None
        ),
        min_free_bytes=min_free_bytes,
    )
    manifest["runtime_manifest_path"] = manifest_path.as_posix()
    return manifest


def build_tensor_factorize_runtime_consumption_proof(
    *,
    runtime_adapter_manifest: str | Path | Mapping[str, Any],
    candidate_manifest: str | Path | Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the proof consumed by family-agnostic materializer verification."""

    repo = Path(repo_root).resolve(strict=False) if repo_root is not None else Path.cwd()
    runtime_manifest = _load_mapping(runtime_adapter_manifest, repo=repo)
    if candidate_manifest is None:
        candidate_manifest_path = runtime_manifest.get("candidate_manifest_path")
        if not isinstance(candidate_manifest_path, str) or not candidate_manifest_path:
            raise TensorFactorizeReceiverError(
                "candidate_manifest is required when runtime manifest lacks candidate_manifest_path"
            )
        candidate = _load_candidate_manifest(candidate_manifest_path, repo=repo)
    else:
        candidate = _load_candidate_manifest(candidate_manifest, repo=repo)
    _require_tensor_factorize_candidate(candidate)
    blockers: list[str] = []
    try:
        require_no_truthy_authority_fields(
            runtime_manifest,
            context="tensor_factorize_runtime_adapter_manifest",
        )
    except ValueError as exc:
        blockers.append(str(exc))
    if runtime_manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        blockers.append("runtime_adapter_manifest_schema_mismatch")
    if runtime_manifest.get("runtime_adapter_ready") is not True:
        blockers.append("runtime_adapter_not_ready")
    blockers.extend(str(blocker) for blocker in runtime_manifest.get("blockers") or [])

    candidate_archive = _candidate_archive_path(candidate, repo=repo)
    selected_member = _selected_member_name(candidate)
    candidate_member = _candidate_member_record(candidate)
    source_archive = _source_archive_path(candidate, repo=repo)
    identity = tensor_factorize_reconstruction_identity(
        candidate_archive,
        member_name=selected_member,
        source_archive=source_archive,
        source_member_name=selected_member,
        tolerances=_candidate_tolerances(candidate),
    )
    blockers.extend(identity["blockers"])
    archive_sha = sha256_file(candidate_archive)
    if runtime_manifest.get("candidate_archive_sha256") != archive_sha:
        blockers.append("runtime_adapter_candidate_archive_sha_mismatch")
    if runtime_manifest.get("candidate_member_sha256") != candidate_member.get("sha256"):
        blockers.append("runtime_adapter_candidate_member_sha_mismatch")

    passed = not blockers
    return {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "proof_kind": RUNTIME_PROOF_KIND,
        "proof_scope": "tensor_factorize_shadow_archive_runtime_adapter_consumes_transformed_archive",
        "target_kind": TENSOR_FACTORIZE_TARGET_KIND,
        "materializer_id": TENSOR_FACTORIZE_MATERIALIZER_ID,
        "receiver_contract_kind": TENSOR_FACTORIZE_RECEIVER_CONTRACT_KIND,
        "receiver_contract_id": f"{TENSOR_FACTORIZE_TARGET_KIND}.receiver.v1",
        "runtime_adapter_manifest": dict(runtime_manifest),
        "runtime_adapter_manifest_path": (
            _resolve(runtime_adapter_manifest, repo=repo).as_posix()
            if isinstance(runtime_adapter_manifest, (str, Path))
            else None
        ),
        "runtime_adapter_ready": runtime_manifest.get("runtime_adapter_ready") is True,
        "ready_for_receiver_verification": passed,
        "candidate_archive": {
            "path": candidate_archive.as_posix(),
            "bytes": candidate_archive.stat().st_size,
            "sha256": archive_sha,
        },
        "candidate_member": dict(candidate_member),
        "candidate_archive_sha256": archive_sha,
        "candidate_member_sha256": candidate_member.get("sha256"),
        "member_sha256": candidate_member.get("sha256"),
        "selected_member_name": selected_member,
        "factorization": dict(candidate.get("factorization") or {}),
        "runtime_consumption_probe": {
            "schema": "tensor_factorize_runtime_adapter_probe.v1",
            "passed": passed,
            "runtime_adapter_ready": runtime_manifest.get("runtime_adapter_ready") is True,
            "shadow_archive_reconstruction_passed": identity["passed"],
            "reconstruction_identity": identity,
        },
        "receiver_contract_satisfied": passed,
        "runtime_consumption_proof_passed": passed,
        "passed": passed,
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
    }


def run_tensor_factorize_receiver_smoke(
    *,
    runtime_dir: str | Path,
    candidate_archive: str | Path,
    output_dir: str | Path,
    file_list: str = "0.mkv",
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Execute the compiled receiver runtime against a candidate archive."""

    runtime = Path(runtime_dir)
    archive = Path(candidate_archive)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tensor-factorize-smoke-") as tmp:
        archive_dir = Path(tmp) / "archive"
        archive_dir.mkdir()
        shutil.copy2(archive, archive_dir / ARCHIVE_NAME)
        command = [
            str(runtime / WRAPPER_SH),
            str(archive_dir),
            str(out_dir),
            file_list,
        ]
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    return {
        "schema": "tensor_factorize_receiver_smoke.v1",
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "command": command,
        **FALSE_AUTHORITY,
    }


def tensor_factorize_reconstruction_identity(
    candidate_archive: str | Path,
    *,
    member_name: str,
    source_archive: str | Path,
    source_member_name: str,
    tolerances: Mapping[str, float | None],
) -> dict[str, Any]:
    """Measure the receiver reconstruction against the source tensor."""

    import numpy as np

    blockers: list[str] = []
    candidate_path = Path(candidate_archive)
    source_path = Path(source_archive)
    try:
        with zipfile.ZipFile(candidate_path, "r") as zf:
            candidate_payload = zf.read(member_name)
        decoded = tensor_factorize_packet_to_npy_bytes(candidate_payload)
        reconstructed = np.asarray(decoded["tensor"], dtype=np.float32)
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
        return {
            "schema": "tensor_factorize_reconstruction_identity.v1",
            "passed": False,
            "blockers": [f"tensor_factorize_payload_not_parseable:{exc}"],
            "max_abs_error": None,
            "rmse": None,
            "max_relative_error": None,
            "reconstructed_member_sha256": None,
        }
    try:
        with zipfile.ZipFile(source_path, "r") as zf:
            source_payload = zf.read(source_member_name)
        source = np.asarray(np.load(io.BytesIO(source_payload), allow_pickle=False), dtype=np.float32)
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
        return {
            "schema": "tensor_factorize_reconstruction_identity.v1",
            "passed": False,
            "blockers": [f"tensor_factorize_source_tensor_not_parseable:{exc}"],
            "max_abs_error": None,
            "rmse": None,
            "max_relative_error": None,
            "reconstructed_member_sha256": None,
        }
    if source.shape != reconstructed.shape:
        blockers.append("tensor_factorize_reconstruction_shape_mismatch")
    delta = reconstructed - source
    abs_delta = np.abs(delta)
    max_abs_error = float(abs_delta.max(initial=0.0))
    rmse = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
    denom = np.maximum(np.abs(source), np.finfo(np.float32).eps)
    max_relative_error = float(np.max(abs_delta / denom, initial=0.0))
    finite = (
        math.isfinite(max_abs_error)
        and math.isfinite(rmse)
        and math.isfinite(max_relative_error)
    )
    if not finite:
        blockers.append("tensor_factorize_reconstruction_error_nonfinite")
    abs_tol = tolerances.get("max_abs_error_tolerance")
    rel_tol = tolerances.get("max_relative_error_tolerance")
    if abs_tol is None and rel_tol is None and max_abs_error != 0.0:
        blockers.append("tensor_factorize_reconstruction_requires_tolerance")
    if abs_tol is not None and max_abs_error > float(abs_tol):
        blockers.append("tensor_factorize_reconstruction_abs_error_exceeds_tolerance")
    if rel_tol is not None and max_relative_error > float(rel_tol):
        blockers.append("tensor_factorize_reconstruction_relative_error_exceeds_tolerance")
    reconstructed_record = tensor_factorize_packet_to_npy_bytes(candidate_payload)
    return {
        "schema": "tensor_factorize_reconstruction_identity.v1",
        "passed": not blockers,
        "member_name": member_name,
        "source_member_name": source_member_name,
        "source_shape": list(source.shape),
        "reconstructed_shape": list(reconstructed.shape),
        "max_abs_error": max_abs_error,
        "max_abs_error_tolerance": abs_tol,
        "rmse": rmse,
        "max_relative_error": max_relative_error,
        "max_relative_error_tolerance": rel_tol,
        "reconstructed_member_sha256": sha256_bytes(reconstructed_record["npy_payload"]),
        "blockers": ordered_unique(blockers),
    }


def _copy_runtime_tree(
    source: Path,
    target: Path,
    *,
    allow_runtime_sidecars: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    copied: list[dict[str, Any]] = []
    skipped: list[str] = []
    for child in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        rel = child.relative_to(source)
        if any(part in _EXCLUDED_NAMES for part in rel.parts):
            skipped.append(rel.as_posix())
            continue
        if child.is_dir():
            continue
        if child.is_symlink():
            blockers.append(f"runtime_symlink_refused:{rel.as_posix()}")
            continue
        if child.suffix in _EXCLUDED_SUFFIXES:
            skipped.append(rel.as_posix())
            continue
        code_like = child.suffix in _CODE_SUFFIXES or child.name in {"inflate.sh", "config.env"}
        if not code_like and not allow_runtime_sidecars:
            if child.suffix in _SKIPPED_DATA_SUFFIXES:
                skipped.append(rel.as_posix())
                continue
            blockers.append(f"runtime_sidecar_refused:{rel.as_posix()}")
            continue
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, destination)
        copied.append(
            {
                "path": rel.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "executable": os.access(destination, os.X_OK),
            }
        )
    return {
        "schema": "tensor_factorize_receiver_source_runtime_copy.v1",
        "source_runtime_dir": source.as_posix(),
        "copied_file_count": len(copied),
        "copied_files": copied,
        "skipped_paths": skipped,
        "blockers": ordered_unique(blockers),
    }


def _write_receiver_wrapper(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"\n"
        f"exec \"$PYTHON_BIN\" \"$HERE/{RUNTIME_PY}\" \"$@\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_receiver_runtime_py(path: Path) -> None:
    path.write_text(_RECEIVER_RUNTIME_SOURCE, encoding="utf-8")
    path.chmod(0o644)


def _source_runtime_entrypoint(source_runtime: Path) -> dict[str, Any]:
    inflate_sh = source_runtime / "inflate.sh"
    inflate_py = source_runtime / "inflate.py"
    blockers: list[str] = []
    if inflate_sh.is_file():
        entry = inflate_sh
        kind = "inflate_sh"
    elif inflate_py.is_file():
        entry = inflate_py
        kind = "inflate_py"
    else:
        entry = source_runtime / "inflate.sh"
        kind = "missing"
        blockers.append("source_runtime_missing_inflate_entrypoint")
    return {
        "kind": kind,
        "path": entry.as_posix(),
        "sha256": sha256_file(entry) if entry.is_file() else None,
        "blockers": blockers,
    }


def _candidate_archive_path(candidate: Mapping[str, Any], *, repo: Path) -> Path:
    archive = candidate.get("candidate_archive")
    if not isinstance(archive, Mapping):
        raise TensorFactorizeReceiverError("candidate manifest missing candidate_archive")
    path = archive.get("path")
    if not isinstance(path, str) or not path.strip():
        raise TensorFactorizeReceiverError("candidate_archive.path missing")
    resolved = _resolve(path, repo=repo)
    if not resolved.is_file():
        raise TensorFactorizeReceiverError(f"candidate archive not found: {resolved}")
    expected_sha = archive.get("sha256")
    if isinstance(expected_sha, str) and expected_sha and sha256_file(resolved) != expected_sha:
        raise TensorFactorizeReceiverError("candidate archive sha256 mismatch")
    return resolved


def _source_archive_path(candidate: Mapping[str, Any], *, repo: Path) -> Path:
    archive = candidate.get("source_archive")
    if not isinstance(archive, Mapping):
        raise TensorFactorizeReceiverError("candidate manifest missing source_archive")
    path = archive.get("path")
    if not isinstance(path, str) or not path.strip():
        raise TensorFactorizeReceiverError("source_archive.path missing")
    resolved = _resolve(path, repo=repo)
    if not resolved.is_file():
        raise TensorFactorizeReceiverError(f"source archive not found: {resolved}")
    expected_sha = archive.get("sha256")
    if isinstance(expected_sha, str) and expected_sha and sha256_file(resolved) != expected_sha:
        raise TensorFactorizeReceiverError("source archive sha256 mismatch")
    return resolved


def _candidate_member_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    member = candidate.get("candidate_member")
    if not isinstance(member, Mapping):
        raise TensorFactorizeReceiverError("candidate manifest missing candidate member")
    return dict(member)


def _source_member_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    member = candidate.get("source_member")
    if not isinstance(member, Mapping):
        raise TensorFactorizeReceiverError("candidate manifest missing source member")
    return dict(member)


def _selected_member_name(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("selected_member_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    member = _source_member_record(candidate)
    value = member.get("name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise TensorFactorizeReceiverError("candidate manifest missing selected member name")


def _candidate_tolerances(candidate: Mapping[str, Any]) -> dict[str, float | None]:
    factorization = candidate.get("factorization")
    if not isinstance(factorization, Mapping):
        factorization = {}
    return {
        "max_abs_error_tolerance": _optional_float(
            factorization.get("max_abs_error_tolerance")
        ),
        "max_relative_error_tolerance": _optional_float(
            factorization.get("max_relative_error_tolerance")
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed < 0.0 or not math.isfinite(parsed):
        raise TensorFactorizeReceiverError("tensor factorize tolerance must be finite and nonnegative")
    return parsed


def _require_tensor_factorize_candidate(candidate: Mapping[str, Any]) -> None:
    if candidate.get("target_kind") != TENSOR_FACTORIZE_TARGET_KIND:
        raise TensorFactorizeReceiverError("candidate manifest is not tensor_factorize_v1")
    if candidate.get("materializer_id") != TENSOR_FACTORIZE_MATERIALIZER_ID:
        raise TensorFactorizeReceiverError("candidate manifest materializer mismatch")
    try:
        require_no_truthy_authority_fields(candidate, context="tensor_factorize_candidate")
    except ValueError as exc:
        raise TensorFactorizeReceiverError(str(exc)) from exc


def _load_candidate_manifest(
    value: str | Path | Mapping[str, Any],
    *,
    repo: Path,
) -> dict[str, Any]:
    return _load_mapping(value, repo=repo)


def _load_mapping(value: str | Path | Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return dict(read_json(_resolve(value, repo=repo)))


def _resolve(path: str | Path, *, repo: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    return candidate.resolve(strict=False)


def _copy_zip_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    copied = zipfile.ZipInfo(info.filename, info.date_time)
    copied.comment = info.comment
    copied.extra = info.extra
    copied.internal_attr = info.internal_attr
    copied.external_attr = info.external_attr
    copied.create_system = info.create_system
    copied.extract_version = info.extract_version
    copied.create_version = info.create_version
    copied.flag_bits = info.flag_bits
    copied.compress_type = info.compress_type
    return copied


def _parse_packet_metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, Mapping):
            raise TensorFactorizeReceiverError("tensor factorize metadata must be an object")
        return dict(parsed)
    if isinstance(value, Mapping):
        return dict(value)
    raise TensorFactorizeReceiverError("tensor factorize metadata must be text")


_RECEIVER_RUNTIME_SOURCE = r'''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generated tensor-factorize receiver runtime."""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = "tensor_factorize_candidate.v1"


def _metadata(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, dict):
            raise RuntimeError("tensor factorize metadata must be an object")
        return parsed
    raise RuntimeError("tensor factorize metadata must be text")


def _try_decode_tensor_packet(payload: bytes) -> bytes | None:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as packet:
            if not {"metadata", "u", "s", "vt"}.issubset(set(packet.files)):
                return None
            meta = _metadata(packet["metadata"])
            if meta.get("schema") != SCHEMA:
                return None
            u = np.asarray(packet["u"], dtype=np.float32)
            s = np.asarray(packet["s"], dtype=np.float32)
            vt = np.asarray(packet["vt"], dtype=np.float32)
        shape = tuple(int(dim) for dim in meta.get("source_shape") or ())
        dtype = np.dtype(str(meta.get("source_dtype") or "float32"))
        tensor = ((u * s) @ vt).reshape(shape).astype(dtype)
        out = io.BytesIO()
        np.save(out, tensor, allow_pickle=False)
        return out.getvalue()
    except Exception:
        return None


def _expand_archive(source_archive: Path, output_archive: Path) -> None:
    with zipfile.ZipFile(source_archive, "r") as source, zipfile.ZipFile(output_archive, "w") as target:
        for info in source.infolist():
            if info.is_dir():
                target.mkdir(info)
                continue
            payload = source.read(info.filename)
            replacement = _try_decode_tensor_packet(payload)
            target.writestr(info, payload if replacement is None else replacement)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: inflate.py <archive_dir> <output_dir> <file_list>")
    archive_dir, output_dir, file_list = map(Path, argv)
    source_archive = archive_dir / "archive.zip"
    if not source_archive.is_file():
        raise SystemExit(f"archive.zip not found: {source_archive}")
    here = Path(__file__).resolve().parent
    source_runtime = here / "source_runtime"
    with tempfile.TemporaryDirectory(prefix="tensor-factorize-receiver-") as tmp:
        shadow_dir = Path(tmp) / "archive"
        shadow_dir.mkdir()
        _expand_archive(source_archive, shadow_dir / "archive.zip")
        inflate_sh = source_runtime / "inflate.sh"
        inflate_py = source_runtime / "inflate.py"
        if inflate_sh.is_file():
            cmd = [str(inflate_sh), str(shadow_dir), str(output_dir), str(file_list)]
        elif inflate_py.is_file():
            cmd = [sys.executable, str(inflate_py), str(shadow_dir), str(output_dir), str(file_list)]
        else:
            raise SystemExit("source runtime has no inflate.sh or inflate.py")
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''


__all__ = [
    "RUNTIME_MANIFEST_SCHEMA",
    "RUNTIME_PROOF_KIND",
    "TensorFactorizeReceiverError",
    "build_tensor_factorize_receiver_runtime",
    "build_tensor_factorize_runtime_consumption_proof",
    "reconstruct_tensor_factorize_archive_bytes",
    "run_tensor_factorize_receiver_smoke",
    "tensor_factorize_packet_to_npy_bytes",
    "tensor_factorize_reconstruction_identity",
    "write_reconstructed_tensor_factorize_archive",
]
