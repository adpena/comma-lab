# SPDX-License-Identifier: MIT
"""Small deterministic IO helpers for local tooling.

These helpers intentionally avoid provider state and process mutation. They are
for repository-local audit, readiness, profiling, and manifest tools that need
stable JSON bytes, SHA-256 digests, and repo-relative path display.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


class ArtifactWriteError(RuntimeError):
    """Raised when an artifact write would clobber state or violate disk budget."""


@dataclass(frozen=True)
class ArtifactWriteResult:
    """Metadata for a guarded artifact write."""

    path: str
    bytes_written: int
    sha256: str
    free_bytes_before: int
    allow_overwrite: bool


@dataclass(frozen=True)
class ArtifactDirTransaction:
    """Staging directory metadata yielded by ``artifact_dir_transaction``."""

    target: Path
    staging: Path
    free_bytes_before: int
    allow_overwrite: bool


def json_text(payload: Any) -> str:
    """Return canonical pretty JSON text with a trailing newline."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def json_line(payload: Any) -> str:
    """Return canonical one-line JSON text with a trailing newline."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def read_json(path: str | Path) -> Any:
    """Read JSON from ``path`` using UTF-8."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    """Write canonical pretty JSON to ``path`` using UTF-8."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json_text(payload), encoding="utf-8")


def write_text_artifact(
    path: str | Path,
    content: str,
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
    min_free_bytes: int = 0,
    encoding: str = "utf-8",
) -> ArtifactWriteResult:
    """Write text through the guarded artifact writer."""

    return write_bytes_artifact(
        path,
        content.encode(encoding),
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_sha256,
        min_free_bytes=min_free_bytes,
    )


def write_json_artifact(
    path: str | Path,
    payload: Any,
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> ArtifactWriteResult:
    """Write canonical JSON as a guarded artifact."""

    return write_text_artifact(
        path,
        json_text(payload),
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_sha256,
        min_free_bytes=min_free_bytes,
    )


def write_bytes_artifact(
    path: str | Path,
    payload: bytes,
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> ArtifactWriteResult:
    """Write bytes without silent clobbering and with a disk-space floor.

    The non-overwrite path uses a temporary file plus a hard-link commit so a
    racing writer cannot slip between a preflight existence check and the final
    artifact creation.
    """

    target = Path(path)
    if min_free_bytes < 0:
        raise ArtifactWriteError("min_free_bytes must be non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not allow_overwrite:
        raise ArtifactWriteError(f"refusing to overwrite existing artifact: {target}")
    if target.exists() and allow_overwrite and expected_existing_sha256 is None:
        raise ArtifactWriteError(
            f"{target}: expected_existing_sha256 is required before overwrite"
        )
    if expected_existing_sha256 is not None:
        if not target.is_file():
            raise ArtifactWriteError(f"{target}: expected existing artifact is missing")
        actual_sha = sha256_file(target)
        if actual_sha != expected_existing_sha256:
            raise ArtifactWriteError(
                f"{target}: existing artifact sha256 mismatch "
                f"expected={expected_existing_sha256} actual={actual_sha}"
            )
    free_bytes_before = _free_bytes_for_write(target)
    required_free = len(payload) + int(min_free_bytes)
    if free_bytes_before < required_free:
        raise ArtifactWriteError(
            f"{target}: insufficient free space before artifact write "
            f"free={free_bytes_before} required={required_free}"
        )

    digest = sha256_bytes(payload)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(target.parent)) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if allow_overwrite:
            temp_path.replace(target)
        else:
            try:
                os.link(temp_path, target)
            except FileExistsError as exc:
                raise ArtifactWriteError(
                    f"refusing to overwrite existing artifact: {target}"
                ) from exc
        _fsync_directory(target.parent)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    return ArtifactWriteResult(
        path=str(target),
        bytes_written=len(payload),
        sha256=digest,
        free_bytes_before=free_bytes_before,
        allow_overwrite=allow_overwrite,
    )


@contextmanager
def artifact_dir_transaction(
    path: str | Path,
    *,
    allow_overwrite: bool = False,
    expected_existing_tree_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> Iterator[ArtifactDirTransaction]:
    """Stage a directory artifact and commit it without deleting first.

    Existing directories are only replaced when the caller supplies the expected
    tree hash. The hash is checked before staging and immediately before the
    rename so a stale worker cannot silently replace newer work.
    """

    target = Path(path)
    if min_free_bytes < 0:
        raise ArtifactWriteError("min_free_bytes must be non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not allow_overwrite:
            raise ArtifactWriteError(f"refusing to overwrite existing artifact dir: {target}")
        if expected_existing_tree_sha256 is None:
            raise ArtifactWriteError(
                f"{target}: expected_existing_tree_sha256 is required before overwrite"
            )
        _require_tree_sha(target, expected_existing_tree_sha256)
    free_bytes_before = _free_bytes_for_write(target)
    if free_bytes_before < min_free_bytes:
        raise ArtifactWriteError(
            f"{target}: insufficient free space before directory artifact write "
            f"free={free_bytes_before} required={min_free_bytes}"
        )
    staging = target.with_name(
        f".{target.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir()
    backup: Path | None = None
    committed = False
    try:
        yield ArtifactDirTransaction(
            target=target,
            staging=staging,
            free_bytes_before=free_bytes_before,
            allow_overwrite=allow_overwrite,
        )
        if target.exists():
            if expected_existing_tree_sha256 is None:
                raise ArtifactWriteError(
                    f"{target}: expected_existing_tree_sha256 is required before overwrite"
                )
            _require_tree_sha(target, expected_existing_tree_sha256)
            backup = target.with_name(
                f".{target.name}.backup.{os.getpid()}.{uuid.uuid4().hex}"
            )
            target.rename(backup)
        try:
            staging.rename(target)
            committed = True
            _fsync_directory(target.parent)
        except Exception:
            if backup is not None and not target.exists():
                backup.rename(target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def tree_sha256(path: str | Path) -> str:
    """Return a deterministic SHA-256 for a file tree."""

    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise ArtifactWriteError(f"{root}: expected file or directory")
    digest = sha256()
    for child in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        rel = child.relative_to(root).as_posix().encode("utf-8")
        if child.is_symlink():
            digest.update(b"L\0" + rel + b"\0")
            digest.update(os.readlink(child).encode("utf-8") + b"\0")
        elif child.is_file():
            digest.update(b"F\0" + rel + b"\0")
            digest.update(sha256_file(child).encode("ascii") + b"\0")
        elif child.is_dir():
            digest.update(b"D\0" + rel + b"\0")
    return digest.hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of one file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of an in-memory byte payload."""

    return sha256(data).hexdigest()


def repo_relative(path: str | Path, repo_root: str | Path) -> str:
    """Return a POSIX path relative to ``repo_root`` when possible."""

    candidate = Path(path)
    root = Path(repo_root)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def _free_bytes_for_write(path: Path) -> int:
    root = path.parent
    while not root.exists() and root.parent != root:
        root = root.parent
    return int(shutil.disk_usage(root).free)


def _require_tree_sha(path: Path, expected: str) -> None:
    actual = tree_sha256(path)
    if actual != expected:
        raise ArtifactWriteError(
            f"{path}: existing artifact tree sha256 mismatch expected={expected} actual={actual}"
        )


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "ArtifactDirTransaction",
    "ArtifactWriteError",
    "ArtifactWriteResult",
    "artifact_dir_transaction",
    "json_line",
    "json_text",
    "read_json",
    "repo_relative",
    "sha256_bytes",
    "sha256_file",
    "tree_sha256",
    "write_bytes_artifact",
    "write_json",
    "write_json_artifact",
    "write_text_artifact",
]
