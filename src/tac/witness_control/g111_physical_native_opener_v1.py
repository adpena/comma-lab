# SPDX-License-Identifier: MIT
"""Content-addressed physical opener for a G111 native-v3 checkpoint.

This module proves only what the physical NPZ can prove without a freshly
constructed trainer runtime:

* the exact regular file at ``path`` was opened without following symlinks;
* the bytes read from that stable inode match a caller-supplied SHA-256;
* the NPZ is pickle-free and contains one canonical transaction manifest;
* all six canonical owners are active, nonempty, and claim every payload leaf
  exactly once (apart from explicit derived-lineage leaves);
* the canonical fourteen-domain coverage matrix is present; and
* every manifest descriptor matches the physical array dtype, shape, byte
  count, and content hash.

Fresh-runtime topology and restore admissibility remain the authority of
``g111_owner_inventory_binder_v1``.  This opener intentionally cannot derive
an expected schema from the checkpoint it is validating.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

from tac.witness_control.trajectory_transaction_v2 import (
    ATOMIC_OWNERS,
    CANONICAL_DOMAIN_COVERAGE,
    LINEAGE_ENVELOPE,
    MANIFEST_KEY,
    PENDING_VERDICT_PREFIX,
    SCHEMA,
    SEMANTIC_DOMAINS,
    EntryDescriptor,
    TransactionManifest,
    TransactionValidationError,
    canonical_owner_semantic_hashes,
    canonical_semantic_hash,
    manifest_from_array,
    stage_arrays,
)

CLAIM_SCOPE: Final = "physical_content_addressed_native_v3_only"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_READ_CHUNK_BYTES: Final = 8 * 1024 * 1024


class G111PhysicalNativeOpenError(TransactionValidationError):
    """Physical native-v3 bytes failed content-addressed validation."""


def _fail(message: str) -> None:
    raise G111PhysicalNativeOpenError(message)


def _canonical_sha256(value: object) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        _fail("expected_sha256 must be exactly 64 lowercase hexadecimal characters")
    return value


def _stable_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


@dataclass(frozen=True, slots=True)
class G111PhysicalOwnerMetadata:
    """Immutable physical metadata for one exact native-v3 owner."""

    owner: str
    active: bool
    claimed_keys: tuple[str, ...]
    payload_keys: tuple[str, ...]
    described_nbytes: int
    semantic_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "active": self.active,
            "claimed_keys": list(self.claimed_keys),
            "payload_keys": list(self.payload_keys),
            "described_nbytes": self.described_nbytes,
            "semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True, slots=True)
class G111PhysicalEntryMetadata:
    """Immutable physical descriptor for one exact NPZ payload leaf."""

    key: str
    owner: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    sha256: str

    @classmethod
    def from_descriptor(
        cls,
        descriptor: EntryDescriptor,
    ) -> G111PhysicalEntryMetadata:
        return cls(
            key=descriptor.key,
            owner=descriptor.owner,
            dtype=descriptor.dtype,
            shape=descriptor.shape,
            nbytes=descriptor.nbytes,
            sha256=descriptor.sha256,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "owner": self.owner,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class G111PhysicalNativeReceipt:
    """Immutable metadata proven from one content-addressed physical NPZ."""

    path: str
    file_bytes: int
    file_sha256: str
    manifest_schema: str
    manifest_semantic_sha256: str
    manifest_array_sha256: str
    entries: Mapping[str, G111PhysicalEntryMetadata]
    entry_count: int
    payload_nbytes: int
    derived_lineage_keys: tuple[str, ...]
    owners: tuple[G111PhysicalOwnerMetadata, ...]
    owner_semantic_sha256: Mapping[str, str]
    domain_coverage: tuple[tuple[str, tuple[str, ...]], ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        hashes = dict(self.owner_semantic_sha256)
        if tuple(hashes) != ATOMIC_OWNERS:
            _fail("receipt owner semantic hashes are not in canonical O1--O6 order")
        entries = dict(self.entries)
        if tuple(entries) != tuple(sorted(entries)):
            _fail("receipt physical entries are not in canonical key order")
        for key, entry in entries.items():
            if key != entry.key:
                _fail(f"receipt entry mapping key {key!r} differs from descriptor key {entry.key!r}")
        object.__setattr__(self, "owner_semantic_sha256", MappingProxyType(hashes))
        object.__setattr__(self, "entries", MappingProxyType(entries))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe copy suitable for a physical-lineage receipt."""

        return {
            "schema": "tac.g111_physical_native_receipt.v1",
            "claim_scope": self.claim_scope,
            "path": self.path,
            "file_bytes": self.file_bytes,
            "file_sha256": self.file_sha256,
            "manifest_schema": self.manifest_schema,
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
            "manifest_array_sha256": self.manifest_array_sha256,
            "entries": {
                key: entry.as_dict()
                for key, entry in self.entries.items()
            },
            "entry_count": self.entry_count,
            "payload_nbytes": self.payload_nbytes,
            "derived_lineage_keys": list(self.derived_lineage_keys),
            "owners": [owner.as_dict() for owner in self.owners],
            "owner_semantic_sha256": dict(self.owner_semantic_sha256),
            "domain_coverage": [
                {"domain": domain, "owners": list(owners)}
                for domain, owners in self.domain_coverage
            ],
        }


@dataclass(frozen=True, slots=True)
class _PhysicalSnapshot:
    path: str
    content: bytes
    sha256: str


def _open_directory_component(parent_fd: int, component: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        _fail("physical native-v3 open requires O_NOFOLLOW and O_DIRECTORY")
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(component, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise G111PhysicalNativeOpenError(
            f"checkpoint path component {component!r} is absent, not a directory, or a symlink"
        ) from exc


def _read_physical_snapshot(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> _PhysicalSnapshot:
    try:
        absolute = os.path.abspath(os.fspath(path))
    except TypeError as exc:
        raise G111PhysicalNativeOpenError("path must be a string or path-like object") from exc
    parts = Path(absolute).parts
    if len(parts) < 2 or parts[0] != os.path.sep or not parts[-1]:
        _fail(f"checkpoint path is not a canonical absolute file path: {absolute!r}")

    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        _fail("physical native-v3 open requires O_NOFOLLOW and O_DIRECTORY")

    directory_fd = os.open(
        os.path.sep,
        os.O_RDONLY | directory | getattr(os, "O_CLOEXEC", 0),
    )
    file_fd: int | None = None
    try:
        for component in parts[1:-1]:
            next_fd = _open_directory_component(directory_fd, component)
            os.close(directory_fd)
            directory_fd = next_fd

        filename = parts[-1]
        flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        try:
            path_before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(path_before.st_mode):
                _fail("checkpoint must not be a symlink")
            if not stat.S_ISREG(path_before.st_mode):
                _fail("checkpoint must be a regular file")
            file_fd = os.open(filename, flags, dir_fd=directory_fd)
        except OSError as exc:
            raise G111PhysicalNativeOpenError(
                "checkpoint is absent, inaccessible, or a symlink"
            ) from exc
        fd_before = os.fstat(file_fd)
        if not stat.S_ISREG(fd_before.st_mode):
            _fail("checkpoint must be a regular file")
        if (path_before.st_dev, path_before.st_ino) != (fd_before.st_dev, fd_before.st_ino):
            _fail("checkpoint path changed while it was opened")

        hasher = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            chunks.append(chunk)
        content = b"".join(chunks)

        fd_after = os.fstat(file_fd)
        try:
            path_after = os.stat(
                filename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise G111PhysicalNativeOpenError(
                "checkpoint path changed while its bytes were read"
            ) from exc
        if _stable_stat_identity(fd_before) != _stable_stat_identity(fd_after):
            _fail("checkpoint inode changed while its bytes were read")
        if (path_after.st_dev, path_after.st_ino) != (fd_after.st_dev, fd_after.st_ino):
            _fail("checkpoint path no longer names the opened inode")
        if len(content) != fd_after.st_size:
            _fail("checkpoint byte count differs from the stable inode size")

        actual_sha256 = hasher.hexdigest()
        if actual_sha256 != expected_sha256:
            _fail(
                "checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        return _PhysicalSnapshot(
            path=absolute,
            content=content,
            sha256=actual_sha256,
        )
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def _load_pickle_free_npz(content: bytes) -> Mapping[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            names = tuple(archive.files)
            if len(names) != len(set(names)):
                duplicates = sorted(
                    name for name in set(names) if names.count(name) > 1
                )
                _fail(f"NPZ contains duplicate member names: {duplicates}")
            raw = {name: archive[name] for name in names}
    except G111PhysicalNativeOpenError:
        raise
    except (EOFError, OSError, ValueError) as exc:
        raise G111PhysicalNativeOpenError(
            "NPZ is malformed, contains an object array, or otherwise requires pickle"
        ) from exc
    return stage_arrays(raw)


def _validate_physical_manifest(
    arrays: Mapping[str, np.ndarray],
) -> tuple[
    TransactionManifest,
    tuple[G111PhysicalOwnerMetadata, ...],
    Mapping[str, str],
    int,
]:
    if MANIFEST_KEY not in arrays:
        _fail("physical native-v3 NPZ lacks its canonical transaction manifest")
    manifest = manifest_from_array(arrays[MANIFEST_KEY])
    if manifest.schema != SCHEMA:
        _fail(f"native-v3 manifest schema must be {SCHEMA!r}")

    payload_keys = set(arrays) - {MANIFEST_KEY}
    pending = sorted(key for key in payload_keys if PENDING_VERDICT_PREFIX in key)
    if pending:
        _fail(f"native-v3 forbids pending-verdict payloads: {pending}")

    entry_keys = tuple(entry.key for entry in manifest.entries)
    if entry_keys != tuple(sorted(entry_keys)) or len(set(entry_keys)) != len(entry_keys):
        _fail("manifest entries must be unique and in canonical key order")
    if set(entry_keys) != payload_keys:
        _fail(
            "manifest descriptor reverse coverage mismatch; "
            f"undescribed={sorted(payload_keys - set(entry_keys))}, "
            f"missing={sorted(set(entry_keys) - payload_keys)}"
        )
    entries = {entry.key: entry for entry in manifest.entries}

    claim_owners = tuple(claim.owner for claim in manifest.owner_claims)
    activity_owners = tuple(row.owner for row in manifest.activity)
    if claim_owners != ATOMIC_OWNERS or activity_owners != ATOMIC_OWNERS:
        _fail("manifest must declare the six canonical owners in O1--O6 order")
    if any(row.active is not True for row in manifest.activity):
        _fail("all six canonical native-v3 owners must be active")

    claimed_by: dict[str, str] = {}
    claims: dict[str, tuple[str, ...]] = {}
    for claim in manifest.owner_claims:
        if not claim.keys:
            _fail(f"active owner {claim.owner!r} must claim at least one physical leaf")
        claims[claim.owner] = claim.keys
        for key in claim.keys:
            if key == MANIFEST_KEY:
                _fail("transaction manifest cannot be owner-claimed")
            previous = claimed_by.setdefault(key, claim.owner)
            if previous != claim.owner:
                _fail(
                    f"payload leaf {key!r} is multiply claimed by "
                    f"{previous!r} and {claim.owner!r}"
                )

    derived = set(manifest.derived_lineage_keys)
    if MANIFEST_KEY in derived:
        _fail("transaction manifest cannot be a derived-lineage leaf")
    overlap = derived & set(claimed_by)
    if overlap:
        _fail(f"derived-lineage leaves are also owner-claimed: {sorted(overlap)}")
    reverse_coverage = set(claimed_by) | derived
    if reverse_coverage != payload_keys:
        _fail(
            "owner claim reverse coverage mismatch; "
            f"unclaimed={sorted(payload_keys - reverse_coverage)}, "
            f"phantom={sorted(reverse_coverage - payload_keys)}"
        )

    coverage = tuple((row.domain, row.owners) for row in manifest.domain_coverage)
    expected_coverage = tuple(
        (domain, CANONICAL_DOMAIN_COVERAGE[domain]) for domain in SEMANTIC_DOMAINS
    )
    if coverage != expected_coverage:
        _fail("manifest semantic-domain coverage differs from the canonical fourteen-domain matrix")

    for key in entry_keys:
        expected_owner = (
            LINEAGE_ENVELOPE if key in derived else claimed_by[key]
        )
        physical = EntryDescriptor.from_array(key, expected_owner, arrays[key])
        if entries[key] != physical:
            _fail(
                f"{key!r}: manifest descriptor differs from physical "
                "dtype, shape, nbytes, owner, or content SHA-256"
            )

    owner_hashes = canonical_owner_semantic_hashes(manifest)
    owner_rows: list[G111PhysicalOwnerMetadata] = []
    for owner in ATOMIC_OWNERS:
        payload_for_owner = set(claims[owner])
        if owner == LINEAGE_ENVELOPE:
            payload_for_owner.update(derived)
        ordered_payload = tuple(sorted(payload_for_owner))
        owner_rows.append(
            G111PhysicalOwnerMetadata(
                owner=owner,
                active=True,
                claimed_keys=claims[owner],
                payload_keys=ordered_payload,
                described_nbytes=sum(entries[key].nbytes for key in ordered_payload),
                semantic_sha256=owner_hashes[owner],
            )
        )
    return (
        manifest,
        tuple(owner_rows),
        MappingProxyType(owner_hashes),
        sum(entry.nbytes for entry in manifest.entries),
    )


def open_g111_native_v3_physical(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> G111PhysicalNativeReceipt:
    """Open and validate one exact physical native-v3 NPZ.

    ``expected_sha256`` is mandatory content-addressed custody, not optional
    metadata.  The returned receipt does not claim fresh-runtime topology,
    restore compatibility, or score authority.
    """

    expected = _canonical_sha256(expected_sha256)
    snapshot = _read_physical_snapshot(path, expected_sha256=expected)
    arrays = _load_pickle_free_npz(snapshot.content)
    try:
        manifest, owners, owner_hashes, payload_nbytes = _validate_physical_manifest(
            arrays
        )
    except G111PhysicalNativeOpenError:
        raise
    except TransactionValidationError as exc:
        raise G111PhysicalNativeOpenError(str(exc)) from exc
    manifest_bytes = arrays[MANIFEST_KEY].tobytes(order="C")
    return G111PhysicalNativeReceipt(
        path=snapshot.path,
        file_bytes=len(snapshot.content),
        file_sha256=snapshot.sha256,
        manifest_schema=manifest.schema,
        manifest_semantic_sha256=canonical_semantic_hash(manifest),
        manifest_array_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        entries=MappingProxyType(
            {
                entry.key: G111PhysicalEntryMetadata.from_descriptor(entry)
                for entry in manifest.entries
            }
        ),
        entry_count=len(manifest.entries),
        payload_nbytes=payload_nbytes,
        derived_lineage_keys=manifest.derived_lineage_keys,
        owners=owners,
        owner_semantic_sha256=owner_hashes,
        domain_coverage=tuple(
            (row.domain, row.owners) for row in manifest.domain_coverage
        ),
    )


__all__ = [
    "CLAIM_SCOPE",
    "G111PhysicalEntryMetadata",
    "G111PhysicalNativeOpenError",
    "G111PhysicalNativeReceipt",
    "G111PhysicalOwnerMetadata",
    "open_g111_native_v3_physical",
]
