# SPDX-License-Identifier: MIT
"""Receiver/runtime compiler for packet-member merge materializers.

The packet-member merge materializer lowers several ZIP members into one
self-describing member. This module builds the matching cooperative receiver:
a portable inflate-runtime adapter that reconstructs the original member names
inside a shadow archive, then delegates to the original runtime.

It is deliberately authority-poor. A compiled receiver proves that transformed
bytes are consumed by a runtime path; it does not claim score, promotion
eligibility, or exact-eval readiness.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.family_agnostic_materializers import (
    FALSE_AUTHORITY,
    PACKET_MEMBER_MERGE_MATERIALIZER_ID,
    PACKET_MEMBER_MERGE_PAYLOAD_MAGIC,
    PACKET_MEMBER_MERGE_RECEIVER_CONTRACT_KIND,
    PACKET_MEMBER_MERGE_RUNTIME_ADAPTER_PROOF_KIND,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    parse_packet_member_merge_payload,
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

RUNTIME_MANIFEST_SCHEMA = "packet_member_merge_receiver_runtime.v1"
RUNTIME_PROOF_KIND = PACKET_MEMBER_MERGE_RUNTIME_ADAPTER_PROOF_KIND
RUNTIME_COMPILER_ID = "packet_member_merge_receiver_compiler.v1"
RUNTIME_PY = "packet_member_merge_receiver_runtime.py"
WRAPPER_SH = "inflate.sh"
SOURCE_RUNTIME_DIR = "source_runtime"
MERGED_ARCHIVE_NAME = "archive.zip"

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
_EXCLUDED_NAMES = {
    ".DS_Store",
    "__pycache__",
    "eval_runs",
}
_EXCLUDED_SUFFIXES = {
    ".log",
    ".pyc",
    ".pyo",
    ".zip",
}
_SKIPPED_DATA_SUFFIXES = {
    ".bin",
    ".br",
    ".mkv",
    ".pt",
    ".raw",
}


class PacketMemberMergeReceiverError(ValueError):
    """Raised when a merge receiver/runtime cannot be compiled safely."""


def reconstruct_packet_member_merge_archive_bytes(
    candidate_archive: str | Path,
    *,
    merged_member_name: str,
) -> bytes:
    """Return archive bytes with the merged member expanded into original names."""

    archive = Path(candidate_archive)
    with tempfile.SpooledTemporaryFile(max_size=16 << 20) as output:
        with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(output, "w") as target:
            for info in source.infolist():
                if info.is_dir():
                    target.mkdir(_copy_zip_info(info))
                    continue
                payload = source.read(info.filename)
                if info.filename != merged_member_name:
                    target.writestr(_copy_zip_info(info), payload)
                    continue
                decoded = parse_packet_member_merge_payload(payload)
                rows = decoded["table"].get("members")
                if not isinstance(rows, Sequence) or isinstance(rows, (bytes, bytearray, str)):
                    raise PacketMemberMergeReceiverError("merge table members must be a list")
                members = decoded["members"]
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise PacketMemberMergeReceiverError("merge table row must be an object")
                    name = str(row.get("name") or "")
                    if not name or name not in members:
                        raise PacketMemberMergeReceiverError(
                            f"merge table row missing reconstructed payload: {name}"
                        )
                    out_info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                    out_info.compress_type = zipfile.ZIP_STORED
                    out_info.external_attr = (stat.S_IFREG | 0o644) << 16
                    target.writestr(out_info, members[name])
        output.seek(0)
        return output.read()


def write_reconstructed_packet_member_merge_archive(
    candidate_archive: str | Path,
    output_archive: str | Path,
    *,
    merged_member_name: str,
) -> dict[str, Any]:
    """Write a shadow archive and return its byte identity."""

    payload = reconstruct_packet_member_merge_archive_bytes(
        candidate_archive,
        merged_member_name=merged_member_name,
    )
    target = Path(output_archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return {
        "path": target.as_posix(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def build_packet_member_merge_receiver_runtime(
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
    """Compile a self-contained receiver runtime for a merge candidate."""

    repo = Path(repo_root).resolve(strict=False) if repo_root is not None else Path.cwd()
    source_runtime = _resolve(source_runtime_dir, repo=repo)
    if not source_runtime.is_dir():
        raise PacketMemberMergeReceiverError(f"source runtime dir not found: {source_runtime}")
    candidate = _load_candidate_manifest(candidate_manifest, repo=repo)
    _require_merge_candidate(candidate)
    runtime_dir = _resolve(runtime_dir_out, repo=repo)
    manifest_path = (
        _resolve(runtime_manifest_out, repo=repo)
        if runtime_manifest_out is not None
        else runtime_dir.with_name(f"{runtime_dir.name}.packet_member_merge_receiver_runtime.json")
    )

    blockers: list[str] = []
    copied_runtime_manifest: dict[str, Any]
    with artifact_dir_transaction(
        runtime_dir,
        allow_overwrite=allow_overwrite,
        expected_existing_tree_sha256=expected_existing_runtime_tree_sha256,
        min_free_bytes=min_free_bytes,
    ) as txn:
        source_dst = txn.staging / SOURCE_RUNTIME_DIR
        copied_runtime_manifest = _copy_runtime_tree(
            source_runtime,
            source_dst,
            allow_runtime_sidecars=allow_runtime_sidecars,
        )
        blockers.extend(copied_runtime_manifest["blockers"])
        _write_receiver_runtime_py(txn.staging / RUNTIME_PY)
        _write_receiver_wrapper(txn.staging / WRAPPER_SH)

    runtime_tree_sha = tree_sha256(runtime_dir)
    candidate_archive = _candidate_archive_path(candidate, repo=repo)
    merged_member = _merged_member_name(candidate)
    candidate_member = _candidate_member_record(candidate)
    reconstructed_identity = _reconstruction_identity(
        candidate_archive,
        merged_member_name=merged_member,
        expected_member_sha256s=_source_member_sha256s(candidate),
    )
    blockers.extend(reconstructed_identity["blockers"])
    source_entrypoint = _source_runtime_entrypoint(runtime_dir / SOURCE_RUNTIME_DIR)
    if source_entrypoint["blockers"]:
        blockers.extend(source_entrypoint["blockers"])

    manifest: dict[str, Any] = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "compiler_id": RUNTIME_COMPILER_ID,
        "runtime_adapter_kind": "shadow_archive_packet_member_merge_receiver",
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
        "candidate_merged_member": dict(candidate_member),
        "candidate_member": dict(candidate_member),
        "candidate_archive_sha256": sha256_file(candidate_archive),
        "candidate_member_sha256": candidate_member.get("sha256"),
        "merged_member_name": merged_member,
        "selected_member_names": list(candidate.get("selected_member_names") or []),
        "source_member_sha256s": _source_member_sha256s(candidate),
        "reconstruction_identity": reconstructed_identity,
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
            "mode": "shadow_archive_and_member_tree_reconstruction_then_delegate",
            "archive_name": MERGED_ARCHIVE_NAME,
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


def build_packet_member_merge_runtime_consumption_proof(
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
            raise PacketMemberMergeReceiverError(
                "candidate_manifest is required when runtime manifest lacks candidate_manifest_path"
            )
        candidate = _load_candidate_manifest(candidate_manifest_path, repo=repo)
    else:
        candidate = _load_candidate_manifest(candidate_manifest, repo=repo)
    _require_merge_candidate(candidate)
    blockers: list[str] = []
    try:
        require_no_truthy_authority_fields(
            runtime_manifest,
            context="packet_member_merge_runtime_adapter_manifest",
        )
    except ValueError as exc:
        blockers.append(str(exc))
    if runtime_manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA:
        blockers.append("runtime_adapter_manifest_schema_mismatch")
    if runtime_manifest.get("runtime_adapter_ready") is not True:
        blockers.append("runtime_adapter_not_ready")
    blockers.extend(str(blocker) for blocker in runtime_manifest.get("blockers") or [])

    candidate_archive = _candidate_archive_path(candidate, repo=repo)
    merged_member = _merged_member_name(candidate)
    candidate_member = _candidate_member_record(candidate)
    expected_member_sha256s = _source_member_sha256s(candidate)
    identity = _reconstruction_identity(
        candidate_archive,
        merged_member_name=merged_member,
        expected_member_sha256s=expected_member_sha256s,
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
        "proof_scope": "packet_member_merge_shadow_archive_runtime_adapter_consumes_transformed_archive",
        "target_kind": PACKET_MEMBER_MERGE_TARGET_KIND,
        "materializer_id": PACKET_MEMBER_MERGE_MATERIALIZER_ID,
        "receiver_contract_kind": PACKET_MEMBER_MERGE_RECEIVER_CONTRACT_KIND,
        "receiver_contract_id": f"{PACKET_MEMBER_MERGE_TARGET_KIND}.receiver.v1",
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
        "candidate_merged_member": dict(candidate_member),
        "candidate_archive_sha256": archive_sha,
        "candidate_member_sha256": candidate_member.get("sha256"),
        "member_sha256": candidate_member.get("sha256"),
        "merged_member_name": merged_member,
        "selected_member_names": list(candidate.get("selected_member_names") or []),
        "source_member_sha256s": expected_member_sha256s,
        "reconstructed_member_sha256s": identity["reconstructed_member_sha256s"],
        "runtime_consumption_probe": {
            "schema": "packet_member_merge_runtime_adapter_probe.v1",
            "passed": passed,
            "runtime_adapter_ready": runtime_manifest.get("runtime_adapter_ready") is True,
            "shadow_archive_reconstruction_passed": identity["passed"],
            "reconstructed_member_proofs": identity["member_proofs"],
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


def run_packet_member_merge_receiver_smoke(
    *,
    runtime_dir: str | Path,
    candidate_archive: str | Path,
    output_dir: str | Path,
    file_list: str | Path = "0.mkv",
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Execute the compiled receiver runtime against a candidate archive."""

    runtime = Path(runtime_dir)
    archive = Path(candidate_archive)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="packet-member-merge-smoke-") as tmp:
        tmp_root = Path(tmp)
        archive_dir = Path(tmp) / "archive"
        archive_dir.mkdir()
        shutil.copy2(archive, archive_dir / MERGED_ARCHIVE_NAME)
        file_list_path, file_list_entries, file_list_fixture_kind = _smoke_file_list_path(
            file_list,
            tmp_root=tmp_root,
        )
        command = [
            str(runtime / WRAPPER_SH),
            str(archive_dir),
            str(out_dir),
            str(file_list_path),
        ]
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    return {
        "schema": "packet_member_merge_receiver_smoke.v1",
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "command": command,
        "file_list_entries": file_list_entries,
        "file_list_fixture_kind": file_list_fixture_kind,
        **FALSE_AUTHORITY,
    }


def _smoke_file_list_path(
    file_list: str | Path,
    *,
    tmp_root: Path,
) -> tuple[Path, list[str], str]:
    """Return a contest-contract file-list path for receiver smoke execution."""

    path = Path(file_list)
    if path.is_file():
        entries = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return path, entries, "caller_supplied_file_list_path"
    raw = str(file_list).strip()
    entries = [line.strip() for line in raw.splitlines() if line.strip()]
    if not entries:
        entries = ["0.mkv"]
    fixture = tmp_root / "file_list.txt"
    fixture.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
    return fixture, entries, "generated_file_list_fixture"


def _reconstruction_identity(
    candidate_archive: Path,
    *,
    merged_member_name: str,
    expected_member_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    blockers: list[str] = []
    member_proofs: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(candidate_archive, "r") as zf:
            merged_payload = zf.read(merged_member_name)
        decoded = parse_packet_member_merge_payload(merged_payload)
    except (KeyError, OSError, zipfile.BadZipFile, ValueError) as exc:
        return {
            "schema": "packet_member_merge_reconstruction_identity.v1",
            "passed": False,
            "blockers": [f"packet_member_merge_payload_not_parseable:{exc}"],
            "reconstructed_member_sha256s": {},
            "member_proofs": [],
        }
    reconstructed = {
        name: sha256_bytes(payload)
        for name, payload in decoded["members"].items()
    }
    for name, expected_sha in expected_member_sha256s.items():
        actual_sha = reconstructed.get(name)
        passed = actual_sha == expected_sha
        if not passed:
            blockers.append(f"reconstructed_member_sha_mismatch:{name}")
        member_proofs.append(
            {
                "member_name": name,
                "expected_sha256": expected_sha,
                "reconstructed_sha256": actual_sha,
                "passed": passed,
            }
        )
    return {
        "schema": "packet_member_merge_reconstruction_identity.v1",
        "passed": not blockers,
        "merge_payload_magic": PACKET_MEMBER_MERGE_PAYLOAD_MAGIC.decode("ascii", errors="replace"),
        "reconstructed_member_sha256s": reconstructed,
        "member_proofs": member_proofs,
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
        "schema": "packet_member_merge_receiver_source_runtime_copy.v1",
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
        "exec \"$PYTHON_BIN\" \"$HERE/packet_member_merge_receiver_runtime.py\" \"$@\"\n",
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
        raise PacketMemberMergeReceiverError("candidate manifest missing candidate_archive")
    path = archive.get("path")
    if not isinstance(path, str) or not path.strip():
        raise PacketMemberMergeReceiverError("candidate_archive.path missing")
    resolved = _resolve(path, repo=repo)
    if not resolved.is_file():
        raise PacketMemberMergeReceiverError(f"candidate archive not found: {resolved}")
    expected_sha = archive.get("sha256")
    if isinstance(expected_sha, str) and expected_sha and sha256_file(resolved) != expected_sha:
        raise PacketMemberMergeReceiverError("candidate archive sha256 mismatch")
    return resolved


def _candidate_member_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    member = candidate.get("candidate_merged_member") or candidate.get("candidate_member")
    if not isinstance(member, Mapping):
        raise PacketMemberMergeReceiverError("candidate manifest missing candidate merged member")
    return dict(member)


def _source_member_sha256s(candidate: Mapping[str, Any]) -> dict[str, str]:
    raw = candidate.get("source_member_sha256s")
    if isinstance(raw, Mapping):
        return {str(name): str(sha) for name, sha in raw.items() if str(sha)}
    rows = candidate.get("source_members")
    out: dict[str, str] = {}
    if isinstance(rows, Sequence) and not isinstance(rows, (bytes, bytearray, str)):
        for row in rows:
            if isinstance(row, Mapping) and row.get("name") and row.get("sha256"):
                out[str(row["name"])] = str(row["sha256"])
    if not out:
        raise PacketMemberMergeReceiverError("candidate manifest missing source member hashes")
    return out


def _merged_member_name(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("merged_member_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    member = _candidate_member_record(candidate)
    value = member.get("name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise PacketMemberMergeReceiverError("candidate manifest missing merged member name")


def _require_merge_candidate(candidate: Mapping[str, Any]) -> None:
    if candidate.get("target_kind") != PACKET_MEMBER_MERGE_TARGET_KIND:
        raise PacketMemberMergeReceiverError("candidate manifest is not packet_member_merge_v1")
    if candidate.get("materializer_id") != PACKET_MEMBER_MERGE_MATERIALIZER_ID:
        raise PacketMemberMergeReceiverError("candidate manifest materializer mismatch")
    try:
        require_no_truthy_authority_fields(candidate, context="packet_member_merge_candidate")
    except ValueError as exc:
        raise PacketMemberMergeReceiverError(str(exc)) from exc


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


_RECEIVER_RUNTIME_SOURCE = r'''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generated packet-member merge receiver runtime."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

JSON_MAGIC = b"TAC_PACKET_MEMBER_MERGE_V1\0"
BINARY_MAGIC = b"TAC_PACKET_MEMBER_MERGE_BIN1\0"
DEFLATE_SEQUENCE_MAGIC = b"TAC_PACKET_MEMBER_MERGE_DFL1\0"


def _parse(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    if payload.startswith(DEFLATE_SEQUENCE_MAGIC):
        return _parse_deflate_sequence(payload)
    if payload.startswith(BINARY_MAGIC):
        return _parse_binary(payload)
    if payload.startswith(JSON_MAGIC):
        return _parse_json(payload)
    raise RuntimeError("merged member payload has bad magic")


def _parse_json(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    if not payload.startswith(JSON_MAGIC):
        raise RuntimeError("merged member payload has bad magic")
    if len(payload) < len(JSON_MAGIC) + 8:
        raise RuntimeError("merged member payload is truncated")
    table_len = struct.unpack_from("<Q", payload, len(JSON_MAGIC))[0]
    table_start = len(JSON_MAGIC) + 8
    table_end = table_start + int(table_len)
    table = json.loads(payload[table_start:table_end].decode("utf-8"))
    concatenated = payload[table_end:]
    members = {}
    codec = str(table.get("payload_codec") or "raw_member_payload_v1")
    for row in table.get("members") or []:
        name = str(row["name"])
        offset = int(row["offset"])
        length = int(row["length"])
        encoded = concatenated[offset: offset + length]
        if codec == "raw_member_payload_v1":
            members[name] = encoded
        elif codec in {
            "source_zip_compressed_stream_v1",
            "source_zip_compressed_stream_binary_table_v1",
        }:
            members[name] = _decompress_zip_member(
                encoded,
                int(row["zip_compress_type"]),
                name,
            )
        else:
            raise RuntimeError(f"unsupported packet member merge payload codec: {codec}")
    return table, members


def _parse_deflate_sequence(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    cursor = len(DEFLATE_SEQUENCE_MAGIC)
    count, cursor = _decode_uvarint(payload, cursor, "member count")
    names = []
    for _ in range(count):
        name_len, cursor = _decode_uvarint(payload, cursor, "name length")
        name_end = cursor + int(name_len)
        if name_end > len(payload):
            raise RuntimeError("deflate sequence merge table name extends past payload")
        names.append(payload[cursor:name_end].decode("utf-8"))
        cursor = name_end
    remaining = payload[cursor:]
    offset = 0
    members = {}
    table_rows = []
    for index, name in enumerate(names):
        decoded, consumed = _decompress_next_zip_deflate_stream(remaining, name)
        members[name] = decoded
        table_rows.append(
            {
                "name": name,
                "offset": offset,
                "length": consumed,
                "zip_compress_type": zipfile.ZIP_DEFLATED,
                "uncompressed_length": len(decoded),
            }
        )
        offset += consumed
        remaining = remaining[consumed:]
        if index == len(names) - 1 and remaining:
            raise RuntimeError("deflate sequence merge payload has trailing bytes")
    return (
        {
            "schema": "packet_member_merge_table.v1",
            "payload_codec": "fixed_order_raw_deflate_sequence_v1",
            "table_format": "uleb_name_raw_deflate_sequence_v1",
            "member_count": len(table_rows),
            "members": table_rows,
        },
        members,
    )


def _parse_binary(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    cursor = len(BINARY_MAGIC)
    count, cursor = _decode_uvarint(payload, cursor, "member count")
    rows = []
    for _ in range(count):
        name_len, cursor = _decode_uvarint(payload, cursor, "name length")
        name_end = cursor + int(name_len)
        if name_end > len(payload):
            raise RuntimeError("binary merge table name extends past payload")
        name = payload[cursor:name_end].decode("utf-8")
        cursor = name_end
        compress_type, cursor = _decode_uvarint(payload, cursor, "compress type")
        compressed_length, cursor = _decode_uvarint(payload, cursor, "compressed length")
        uncompressed_length, cursor = _decode_uvarint(payload, cursor, "uncompressed length")
        rows.append(
            {
                "name": name,
                "zip_compress_type": int(compress_type),
                "length": int(compressed_length),
                "uncompressed_length": int(uncompressed_length),
            }
        )
    concatenated = payload[cursor:]
    offset = 0
    members = {}
    table_rows = []
    for row in rows:
        name = row["name"]
        length = int(row["length"])
        encoded = concatenated[offset: offset + length]
        if len(encoded) != length:
            raise RuntimeError(f"binary merge payload truncated for {name}")
        decoded = _decompress_zip_member(encoded, int(row["zip_compress_type"]), name)
        if len(decoded) != int(row["uncompressed_length"]):
            raise RuntimeError(f"binary merge payload length mismatch for {name}")
        members[name] = decoded
        table_rows.append(
            {
                "name": name,
                "offset": offset,
                "length": length,
                "zip_compress_type": int(row["zip_compress_type"]),
                "uncompressed_length": len(decoded),
            }
        )
        offset += length
    if offset != len(concatenated):
        raise RuntimeError("binary merge payload has trailing bytes")
    return (
        {
            "schema": "packet_member_merge_table.v1",
            "payload_codec": "source_zip_compressed_stream_binary_table_v1",
            "table_format": "uleb_name_compressed_stream_table_v1",
            "member_count": len(table_rows),
            "members": table_rows,
        },
        members,
    )


def _decompress_zip_member(payload: bytes, compress_type: int, name: str) -> bytes:
    if compress_type == zipfile.ZIP_STORED:
        return payload
    if compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(payload, -zlib.MAX_WBITS)
    raise RuntimeError(f"unsupported ZIP compression method for {name}: {compress_type}")


def _decompress_next_zip_deflate_stream(payload: bytes, name: str) -> tuple[bytes, int]:
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    decoded = decompressor.decompress(payload)
    decoded += decompressor.flush()
    if not decompressor.eof:
        raise RuntimeError(f"deflate stream did not terminate for {name}")
    consumed = len(payload) - len(decompressor.unused_data)
    if consumed <= 0:
        raise RuntimeError(f"deflate stream consumed no bytes for {name}")
    return decoded, consumed


def _write_member_file(root: Path, name: str, payload: bytes) -> None:
    destination = (root / name).resolve()
    root_resolved = root.resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        raise RuntimeError(f"refusing unsafe reconstructed member path: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _decode_uvarint(data: bytes, offset: int, label: str) -> tuple[int, int]:
    value = 0
    shift = 0
    cursor = offset
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            raise RuntimeError(f"{label} uvarint too wide")
    raise RuntimeError(f"{label} uvarint truncated")


def _expand_archive(source_archive: Path, output_archive: Path) -> None:
    member_root = output_archive.parent
    with zipfile.ZipFile(source_archive, "r") as source, zipfile.ZipFile(output_archive, "w") as target:
        for info in source.infolist():
            if info.is_dir():
                target.mkdir(info)
                (member_root / info.filename).mkdir(parents=True, exist_ok=True)
                continue
            payload = source.read(info.filename)
            if (
                payload.startswith(JSON_MAGIC)
                or payload.startswith(BINARY_MAGIC)
                or payload.startswith(DEFLATE_SEQUENCE_MAGIC)
            ):
                table, members = _parse(payload)
                for row in table.get("members") or []:
                    out = zipfile.ZipInfo(str(row["name"]), (1980, 1, 1, 0, 0, 0))
                    out.compress_type = zipfile.ZIP_STORED
                    target.writestr(out, members[str(row["name"])])
                    _write_member_file(member_root, str(row["name"]), members[str(row["name"])])
            else:
                target.writestr(info, payload)
                _write_member_file(member_root, info.filename, payload)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: inflate.py <archive_dir> <output_dir> <file_list>")
    archive_dir, output_dir, file_list = map(Path, argv)
    source_archive = archive_dir / "archive.zip"
    if not source_archive.is_file():
        raise SystemExit(f"archive.zip not found: {source_archive}")
    here = Path(__file__).resolve().parent
    source_runtime = here / "source_runtime"
    with tempfile.TemporaryDirectory(prefix="packet-member-merge-receiver-") as tmp:
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
    "PacketMemberMergeReceiverError",
    "build_packet_member_merge_receiver_runtime",
    "build_packet_member_merge_runtime_consumption_proof",
    "reconstruct_packet_member_merge_archive_bytes",
    "run_packet_member_merge_receiver_smoke",
    "write_reconstructed_packet_member_merge_archive",
]
