# SPDX-License-Identifier: MIT
"""Deterministic, fail-closed sealing for research-evidence directories.

The sealer deliberately accepts only regular files below a repository-local
directory.  It records a sorted content manifest, stores that manifest plus the
payload in a deterministic gzip-tar stream, validates the staged artifact, and
atomically publishes a content-addressed output directory.  It never mutates or
deletes the source evidence.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_SCHEMA = "tac.research_evidence_manifest.v1"
MANIFEST_MEMBER = "evidence.manifest.json"
PAYLOAD_PREFIX = "payload"
BUNDLE_FILENAME = "bundle.evidence.bundle"
MANIFEST_FILENAME = "manifest.json"
_CHUNK_BYTES = 1 << 20
_HEX = frozenset("0123456789abcdef")


class EvidenceSealError(ValueError):
    """Raised when custody, path safety, or verification requirements fail."""


@dataclass(frozen=True)
class EvidenceFile:
    """One regular source file secured by the evidence manifest."""

    path: str
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class EvidenceManifest:
    """Canonical, path-independent identity of a research-evidence tree."""

    files: tuple[EvidenceFile, ...]
    file_manifest_sha256: str
    tree_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "files": [entry.as_dict() for entry in self.files],
            "file_manifest_sha256": self.file_manifest_sha256,
            "tree_sha256": self.tree_sha256,
        }


@dataclass(frozen=True)
class SealResult:
    """Published paths and independently useful content hashes."""

    publish_dir: Path
    manifest_path: Path
    bundle_path: Path
    manifest: EvidenceManifest
    bundle_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": str(self.bundle_path),
            "bundle_sha256": self.bundle_sha256,
            "file_manifest_sha256": self.manifest.file_manifest_sha256,
            "manifest_path": str(self.manifest_path),
            "publish_dir": str(self.publish_dir),
            "tree_sha256": self.manifest.tree_sha256,
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _is_lower_hex_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def _resolved(path: Path | str, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise EvidenceSealError(f"{label} must not be a symbolic link: {candidate}")
    try:
        return candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvidenceSealError(f"{label} does not exist: {candidate}") from exc


def _require_within(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise EvidenceSealError(f"{label} must be inside repository root: {path}") from exc


def _validate_destination_before_mutation(
    path: Path | str,
    *,
    repo_root: Path,
    label: str,
) -> Path:
    """Resolve and contain a destination without creating any path component."""
    candidate = Path(os.path.abspath(os.fspath(path)))
    cursor = candidate
    missing_components: list[str] = []
    while not cursor.exists():
        if cursor.is_symlink():
            raise EvidenceSealError(f"{label} must not contain a symbolic-link component: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            raise EvidenceSealError(f"{label} has no existing ancestor: {candidate}")
        missing_components.append(cursor.name)
        cursor = parent

    current = Path(cursor.anchor)
    for component in cursor.parts[1:]:
        current /= component
        if current.is_symlink():
            raise EvidenceSealError(f"{label} must not contain a symbolic-link component: {current}")
    if not cursor.is_dir():
        raise EvidenceSealError(f"{label} nearest existing ancestor must be a directory: {cursor}")

    resolved_ancestor = cursor.resolve(strict=True)
    _require_within(resolved_ancestor, repo_root, label=f"{label} nearest existing ancestor")
    resolved_destination = resolved_ancestor.joinpath(*reversed(missing_components)).resolve(strict=False)
    _require_within(resolved_destination, repo_root, label=label)
    return resolved_destination


def _validate_relative_path(path: PurePosixPath | str) -> str:
    raw = str(path)
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or raw.startswith("/"):
        raise EvidenceSealError(f"unsafe evidence path: {raw!r}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise EvidenceSealError(f"unsafe evidence path: {raw!r}")
    if "\\" in raw:
        raise EvidenceSealError(f"unsafe evidence path: {raw!r}")
    return candidate.as_posix()


def _source_files(source: Path) -> tuple[EvidenceFile, ...]:
    files: list[EvidenceFile] = []
    for current, directories, names in os.walk(source, topdown=True, followlinks=False):
        current_path = Path(current)
        safe_directories: list[str] = []
        for directory in directories:
            path = current_path / directory
            if path.is_symlink():
                raise EvidenceSealError(f"evidence source contains symbolic-link directory: {path}")
            if not path.is_dir():
                raise EvidenceSealError(f"evidence source contains non-directory path: {path}")
            safe_directories.append(directory)
        directories[:] = sorted(safe_directories)
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise EvidenceSealError(f"evidence source contains non-regular file: {path}")
            relative = _validate_relative_path(path.relative_to(source).as_posix())
            byte_count, digest = _sha256_file(path)
            files.append(EvidenceFile(path=relative, bytes=byte_count, sha256=digest))
    return tuple(sorted(files, key=lambda entry: entry.path))


def build_manifest(source: Path | str, *, repo_root: Path | str) -> EvidenceManifest:
    """Hash a repository-local regular-file tree into a canonical manifest."""
    root = _resolved(repo_root, label="repository root")
    if not root.is_dir():
        raise EvidenceSealError(f"repository root must be a directory: {root}")
    source_path = _resolved(source, label="evidence source")
    if not source_path.is_dir():
        raise EvidenceSealError(f"evidence source must be a directory: {source_path}")
    _require_within(source_path, root, label="evidence source")
    files = _source_files(source_path)
    rows = [entry.as_dict() for entry in files]
    file_manifest_sha256 = _sha256_bytes(_canonical_json_bytes(rows))
    tree_sha256 = _sha256_bytes(_canonical_json_bytes({"schema": MANIFEST_SCHEMA, "files": rows}))
    return EvidenceManifest(
        files=files,
        file_manifest_sha256=file_manifest_sha256,
        tree_sha256=tree_sha256,
    )


def _parse_manifest(raw: bytes) -> EvidenceManifest:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceSealError("manifest is not valid UTF-8 canonical JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "files",
        "file_manifest_sha256",
        "tree_sha256",
    }:
        raise EvidenceSealError("manifest has an unexpected schema")
    if payload["schema"] != MANIFEST_SCHEMA or not isinstance(payload["files"], list):
        raise EvidenceSealError("manifest schema or files are invalid")
    files: list[EvidenceFile] = []
    for row in payload["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise EvidenceSealError("manifest file entry has an unexpected schema")
        path = _validate_relative_path(row["path"])
        if not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] < 0:
            raise EvidenceSealError(f"manifest byte count is invalid for {path!r}")
        if not _is_lower_hex_sha256(row["sha256"]):
            raise EvidenceSealError(f"manifest SHA-256 is invalid for {path!r}")
        files.append(EvidenceFile(path=path, bytes=row["bytes"], sha256=row["sha256"]))
    if [entry.path for entry in files] != sorted(entry.path for entry in files):
        raise EvidenceSealError("manifest paths must be sorted")
    if len({entry.path for entry in files}) != len(files):
        raise EvidenceSealError("manifest has duplicate paths")
    result = EvidenceManifest(
        files=tuple(files),
        file_manifest_sha256=payload["file_manifest_sha256"],
        tree_sha256=payload["tree_sha256"],
    )
    if not _is_lower_hex_sha256(result.file_manifest_sha256) or not _is_lower_hex_sha256(result.tree_sha256):
        raise EvidenceSealError("manifest aggregate SHA-256 is invalid")
    if _canonical_json_bytes(result.as_dict()) != raw:
        raise EvidenceSealError("manifest JSON is not canonical")
    rows = [entry.as_dict() for entry in result.files]
    if result.file_manifest_sha256 != _sha256_bytes(_canonical_json_bytes(rows)):
        raise EvidenceSealError("manifest file aggregate hash does not verify")
    if result.tree_sha256 != _sha256_bytes(_canonical_json_bytes({"schema": MANIFEST_SCHEMA, "files": rows})):
        raise EvidenceSealError("manifest tree hash does not verify")
    return result


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_bundle(source: Path, manifest: EvidenceManifest, destination: Path) -> None:
    manifest_bytes = _canonical_json_bytes(manifest.as_dict())
    with (
        destination.open("xb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0, compresslevel=9) as gzip_handle,
        tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        archive.addfile(_tar_info(MANIFEST_MEMBER, len(manifest_bytes)), io.BytesIO(manifest_bytes))
        for entry in manifest.files:
            path = source / entry.path
            with path.open("rb") as handle:
                archive.addfile(_tar_info(f"{PAYLOAD_PREFIX}/{entry.path}", entry.bytes), handle)
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def _validate_tar_member(member: tarfile.TarInfo) -> str:
    if not member.isreg():
        raise EvidenceSealError(f"bundle contains non-regular member: {member.name!r}")
    if (member.mtime, member.uid, member.gid, member.mode, member.uname, member.gname) != (0, 0, 0, 0o644, "", ""):
        raise EvidenceSealError(f"bundle member metadata is not deterministic: {member.name!r}")
    return _validate_relative_path(member.name)


def _stream_member_digest(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[int, str]:
    handle = archive.extractfile(member)
    if handle is None:
        raise EvidenceSealError(f"bundle member cannot be read: {member.name!r}")
    digest = hashlib.sha256()
    byte_count = 0
    with handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


@contextmanager
def _open_bundle(bundle: Path) -> Iterator[tarfile.TarFile]:
    with (
        bundle.open("rb") as raw_handle,
        gzip.GzipFile(mode="rb", fileobj=raw_handle) as gzip_handle,
        tarfile.open(fileobj=gzip_handle, mode="r:") as archive,
    ):
        yield archive


def verify_bundle(bundle: Path | str, *, repo_root: Path | str) -> EvidenceManifest:
    """Verify bundle structure, canonical manifest, and every payload byte."""
    root = _resolved(repo_root, label="repository root")
    bundle_path = _resolved(bundle, label="bundle")
    _require_within(bundle_path, root, label="bundle")
    if not bundle_path.is_file():
        raise EvidenceSealError(f"bundle must be a regular file: {bundle_path}")
    try:
        with _open_bundle(bundle_path) as archive:
            members = archive.getmembers()
            names = [_validate_tar_member(member) for member in members]
            if len(names) != len(set(names)) or not names or names[0] != MANIFEST_MEMBER:
                raise EvidenceSealError("bundle members are missing, duplicated, or out of order")
            manifest_member = members[0]
            manifest_handle = archive.extractfile(manifest_member)
            if manifest_handle is None:
                raise EvidenceSealError("bundle manifest cannot be read")
            with manifest_handle:
                manifest_bytes = manifest_handle.read()
            manifest = _parse_manifest(manifest_bytes)
            expected_names = [MANIFEST_MEMBER, *(f"{PAYLOAD_PREFIX}/{entry.path}" for entry in manifest.files)]
            if names != expected_names:
                raise EvidenceSealError("bundle members do not match the manifest")
            for member, entry in zip(members[1:], manifest.files, strict=True):
                byte_count, digest = _stream_member_digest(archive, member)
                if (byte_count, digest) != (entry.bytes, entry.sha256):
                    raise EvidenceSealError(f"bundle payload does not match manifest: {entry.path}")
            return manifest
    except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
        raise EvidenceSealError(f"bundle cannot be read safely: {bundle_path}") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _output_directory(output_dir: Path | str, *, repo_root: Path, source: Path) -> Path:
    resolved = _validate_destination_before_mutation(
        output_dir,
        repo_root=repo_root,
        label="output directory",
    )
    try:
        resolved.relative_to(source)
    except ValueError:
        pass
    else:
        raise EvidenceSealError("output directory must not be inside evidence source")
    if resolved.exists() and not resolved.is_dir():
        raise EvidenceSealError(f"output directory must be a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return _resolved(resolved, label="output directory")


def seal_research_evidence(
    source: Path | str,
    *,
    output_dir: Path | str,
    repo_root: Path | str,
) -> SealResult:
    """Seal ``source`` without overwriting outputs or mutating source evidence."""
    root = _resolved(repo_root, label="repository root")
    source_path = _resolved(source, label="evidence source")
    manifest = build_manifest(source_path, repo_root=root)
    published_parent = _output_directory(output_dir, repo_root=root, source=source_path)
    publish_dir = published_parent / f"{source_path.name}.evidence.{manifest.tree_sha256}"
    if publish_dir.exists() or publish_dir.is_symlink():
        raise EvidenceSealError(f"refusing to overwrite existing evidence bundle: {publish_dir}")
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{publish_dir.name}.stage-", dir=published_parent))
    staged_manifest = stage_dir / MANIFEST_FILENAME
    staged_bundle = stage_dir / BUNDLE_FILENAME
    manifest_bytes = _canonical_json_bytes(manifest.as_dict())
    with staged_manifest.open("xb") as handle:
        handle.write(manifest_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    _write_bundle(source_path, manifest, staged_bundle)
    if _parse_manifest(staged_manifest.read_bytes()) != manifest:
        raise EvidenceSealError("staged manifest failed verification")
    if verify_bundle(staged_bundle, repo_root=root) != manifest:
        raise EvidenceSealError("staged bundle failed verification")
    os.replace(stage_dir, publish_dir)
    _fsync_directory(published_parent)
    manifest_path = publish_dir / MANIFEST_FILENAME
    bundle_path = publish_dir / BUNDLE_FILENAME
    if (
        _parse_manifest(manifest_path.read_bytes()) != manifest
        or verify_bundle(bundle_path, repo_root=root) != manifest
    ):
        raise EvidenceSealError("published evidence bundle failed verification")
    return SealResult(
        publish_dir=publish_dir,
        manifest_path=manifest_path,
        bundle_path=bundle_path,
        manifest=manifest,
        bundle_sha256=_sha256_file(bundle_path)[1],
    )


def restore_bundle(bundle: Path | str, destination: Path | str, *, repo_root: Path | str) -> EvidenceManifest:
    """Restore a verified bundle into a new repository-local directory only."""
    root = _resolved(repo_root, label="repository root")
    bundle_path = _resolved(bundle, label="bundle")
    manifest = verify_bundle(bundle_path, repo_root=root)
    final_target = _validate_destination_before_mutation(
        destination,
        repo_root=root,
        label="restore destination",
    )
    if final_target.is_symlink() or final_target.exists():
        raise EvidenceSealError(f"refusing to overwrite restore destination: {final_target}")
    parent = final_target.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent = _resolved(parent, label="restore destination parent")
    final_target = parent / final_target.name
    if final_target.is_symlink() or final_target.exists():
        raise EvidenceSealError(f"refusing to overwrite restore destination: {final_target}")
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{final_target.name}.restore-", dir=parent))
    try:
        with _open_bundle(bundle_path) as archive:
            members = archive.getmembers()[1:]
            for member, entry in zip(members, manifest.files, strict=True):
                destination_path = stage_dir / entry.path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                source_handle = archive.extractfile(member)
                if source_handle is None:
                    raise EvidenceSealError(f"bundle member cannot be restored: {entry.path}")
                with source_handle, destination_path.open("xb") as output_handle:
                    shutil.copyfileobj(source_handle, output_handle, length=_CHUNK_BYTES)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
        restored = build_manifest(stage_dir, repo_root=root)
        if restored != manifest:
            raise EvidenceSealError("restored tree does not match bundle manifest")
        os.replace(stage_dir, final_target)
        _fsync_directory(parent)
        if build_manifest(final_target, repo_root=root) != manifest:
            raise EvidenceSealError("published restore does not match bundle manifest")
        return manifest
    except BaseException:
        # The source and any already-published evidence are never deleted.  A
        # failed restore stage is intentionally retained for forensic diagnosis.
        raise


def default_output_dir(source: Path | str) -> Path:
    """Return a source-sibling, repository-local default output directory."""
    return Path(source).parent / "evidence_bundles"


__all__ = [
    "BUNDLE_FILENAME",
    "MANIFEST_FILENAME",
    "EvidenceFile",
    "EvidenceManifest",
    "EvidenceSealError",
    "SealResult",
    "build_manifest",
    "default_output_dir",
    "restore_bundle",
    "seal_research_evidence",
    "verify_bundle",
]
