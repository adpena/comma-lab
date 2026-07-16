# SPDX-License-Identifier: MIT
"""Deterministic exact-byte rate matching for two already-closed ZIP archives.

Rate matching is experimental custody, not compression.  Each output is a byte
copy of its source with one fixed-metadata, ``ZIP_STORED`` ``rate_match.bin``
member appended.  Payload lengths are chosen from rendered zero-payload copies,
then both final archive sizes are required to agree exactly.  Existing member
payloads are hashed before and after; malformed archives, duplicate names, and
pre-existing/duplicate padding fail closed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "FIXED_ZIP_TIMESTAMP",
    "PADDING_MEMBER",
    "ArchiveBudgetError",
    "EqualArchiveBudgetReceipt",
    "MatchedArchiveReceipt",
    "equalize_archive_budgets",
    "file_sha256",
    "output_tree_hashes",
    "verify_equal_archive_budget_receipt",
    "verify_matched_archive_receipt",
    "verify_original_members_preserved",
    "verify_output_tree_preserved",
    "verify_rate_match_member",
    "zip_member_hashes",
]

PADDING_MEMBER = "rate_match.bin"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZERO_CHUNK = b"\0" * (1024 * 1024)
_MAX_ZIP32_PADDING = (1 << 31) - 1
_RECEIPT_VERSION = "equal_archive_budget_v2"
_PAIR_PUBLISH_REPLACE = os.replace
_ROLLBACK_REPLACE = os.replace


class ArchiveBudgetError(ValueError):
    """An archive cannot support an unambiguous exact-byte comparison."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_member_path_and_type(info: zipfile.ZipInfo) -> None:
    """Reject extraction aliases and non-regular filesystem objects."""
    name = info.filename
    if not name or "\x00" in name:
        raise ArchiveBudgetError("ZIP member name must be nonempty and NUL-free")
    if "\\" in name:
        raise ArchiveBudgetError(f"ZIP member uses forbidden backslash path alias: {name!r}")
    if name.startswith("/"):
        raise ArchiveBudgetError(f"ZIP member path must be relative: {name!r}")
    path_without_directory_suffix = name[:-1] if name.endswith("/") else name
    parts = path_without_directory_suffix.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArchiveBudgetError(f"ZIP member path is non-canonical or traversing: {name!r}")
    first = parts[0]
    if len(first) >= 2 and first[0].isalpha() and first[1] == ":":
        raise ArchiveBudgetError(f"ZIP member uses forbidden drive-qualified path: {name!r}")

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if info.create_system == 3 and file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ArchiveBudgetError(f"ZIP member is a symlink or special entry: {name!r}")
    if file_type == stat.S_IFDIR and not info.is_dir():
        raise ArchiveBudgetError(f"ZIP directory metadata/path mismatch: {name!r}")
    if info.is_dir() and file_type == stat.S_IFREG:
        raise ArchiveBudgetError(f"ZIP regular-file metadata/path mismatch: {name!r}")


def _open_validated_archive(
    path: str | Path, *, require_padding: bool | None
) -> tuple[zipfile.ZipFile, list[zipfile.ZipInfo]]:
    archive = Path(path)
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise ArchiveBudgetError(f"not a well-formed ZIP archive: {archive}")
    try:
        handle = zipfile.ZipFile(archive, mode="r")
        infos = handle.infolist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArchiveBudgetError(f"cannot parse ZIP archive {archive}: {exc}") from exc
    try:
        for info in infos:
            _validate_member_path_and_type(info)
    except ArchiveBudgetError:
        handle.close()
        raise
    names = [info.filename for info in infos]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        handle.close()
        raise ArchiveBudgetError(f"duplicate ZIP member names are ambiguous: {duplicates}")
    canonical_names: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        canonical = info.filename.rstrip("/")
        if canonical in canonical_names:
            handle.close()
            raise ArchiveBudgetError(
                f"ZIP file/directory member paths alias each other: {canonical!r}"
            )
        canonical_names[canonical] = info
    regular_paths = {name for name, info in canonical_names.items() if not info.is_dir()}
    for canonical in canonical_names:
        parts = canonical.split("/")
        for end in range(1, len(parts)):
            parent = "/".join(parts[:end])
            if parent in regular_paths:
                handle.close()
                raise ArchiveBudgetError(
                    f"ZIP regular file aliases a member parent directory: {parent!r}"
                )
    padding_count = names.count(PADDING_MEMBER)
    padding_alias_count = sum(name.rstrip("/") == PADDING_MEMBER for name in names)
    if require_padding is True and (padding_count != 1 or padding_alias_count != 1):
        handle.close()
        raise ArchiveBudgetError(f"matched archive must contain exactly one {PADDING_MEMBER}")
    if require_padding is False and padding_alias_count != 0:
        handle.close()
        raise ArchiveBudgetError(f"source archive already contains reserved {PADDING_MEMBER}")
    return handle, infos


def zip_member_hashes(
    path: str | Path, *, include_padding: bool = False
) -> dict[str, str]:
    """Hash uncompressed member bytes, rejecting malformed or duplicate archives."""
    try:
        handle, infos = _open_validated_archive(path, require_padding=None)
        with handle:
            hashes: dict[str, str] = {}
            for info in infos:
                if not include_padding and info.filename == PADDING_MEMBER:
                    continue
                try:
                    payload = handle.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise ArchiveBudgetError(
                        f"cannot read/CRC-verify member {info.filename!r} from {path}"
                    ) from exc
                hashes[info.filename] = hashlib.sha256(payload).hexdigest()
            return hashes
    except ArchiveBudgetError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArchiveBudgetError(f"cannot validate ZIP archive {path}: {exc}") from exc


def verify_rate_match_member(path: str | Path) -> int:
    """Validate padding name, uniqueness, timestamp, storage mode, and zero payload."""
    handle, infos = _open_validated_archive(path, require_padding=True)
    with handle:
        info = next(item for item in infos if item.filename == PADDING_MEMBER)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ArchiveBudgetError(f"{PADDING_MEMBER} must use ZIP_STORED")
        if info.date_time != FIXED_ZIP_TIMESTAMP:
            raise ArchiveBudgetError(
                f"{PADDING_MEMBER} timestamp {info.date_time} is not {FIXED_ZIP_TIMESTAMP}"
            )
        if info.extra or info.comment:
            raise ArchiveBudgetError(f"{PADDING_MEMBER} may not carry extra fields or comments")
        remaining = info.file_size
        try:
            with handle.open(info, mode="r") as payload:
                while remaining:
                    chunk = payload.read(min(remaining, len(_ZERO_CHUNK)))
                    if not chunk or chunk != b"\0" * len(chunk):
                        raise ArchiveBudgetError(f"{PADDING_MEMBER} payload must be all zero bytes")
                    remaining -= len(chunk)
                if payload.read(1):
                    raise ArchiveBudgetError(f"{PADDING_MEMBER} exceeds declared payload length")
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise ArchiveBudgetError(f"cannot read/CRC-verify {PADDING_MEMBER}") from exc
        return int(info.file_size)


def verify_original_members_preserved(
    source_archive: str | Path, matched_archive: str | Path
) -> dict[str, str]:
    """Prove all and only the source member payloads survive rate matching."""
    source_handle, _ = _open_validated_archive(source_archive, require_padding=False)
    source_handle.close()
    verify_rate_match_member(matched_archive)
    source = zip_member_hashes(source_archive, include_padding=False)
    matched = zip_member_hashes(matched_archive, include_padding=False)
    if source != matched:
        missing = sorted(set(source) - set(matched))
        added = sorted(set(matched) - set(source))
        changed = sorted(name for name in set(source) & set(matched) if source[name] != matched[name])
        raise ArchiveBudgetError(
            f"original ZIP members changed: missing={missing}, added={added}, changed={changed}"
        )
    return source


def output_tree_hashes(root: str | Path) -> dict[str, str]:
    """Return deterministic relative-path hashes for an inflated output tree."""
    base = Path(root)
    if not base.is_dir():
        raise ArchiveBudgetError(f"inflated output root is not a directory: {base}")
    hashes: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ArchiveBudgetError(f"inflated output custody refuses symlink: {path}")
        if path.is_file():
            hashes[path.relative_to(base).as_posix()] = file_sha256(path)
    return hashes


def verify_output_tree_preserved(
    source_output: str | Path, matched_output: str | Path
) -> dict[str, str]:
    """Prove an inflate/replay produced the identical named output bytes."""
    source = output_tree_hashes(source_output)
    matched = output_tree_hashes(matched_output)
    if source != matched:
        missing = sorted(set(source) - set(matched))
        added = sorted(set(matched) - set(source))
        changed = sorted(name for name in set(source) & set(matched) if source[name] != matched[name])
        raise ArchiveBudgetError(
            f"inflated outputs changed: missing={missing}, added={added}, changed={changed}"
        )
    return source


def _write_zeros(handle: Any, size: int) -> None:
    remaining = int(size)
    while remaining:
        chunk = _ZERO_CHUNK[: min(remaining, len(_ZERO_CHUNK))]
        handle.write(chunk)
        remaining -= len(chunk)


def _append_padding_copy(source: Path, destination: Path, padding_bytes: int) -> None:
    if padding_bytes < 0 or padding_bytes > _MAX_ZIP32_PADDING:
        raise ArchiveBudgetError(
            f"padding length {padding_bytes} is outside deterministic ZIP32 range"
        )
    source_handle, _ = _open_validated_archive(source, require_padding=False)
    source_handle.close()
    shutil.copyfile(source, destination)
    try:
        with zipfile.ZipFile(destination, mode="a", allowZip64=False) as archive:
            info = zipfile.ZipInfo(PADDING_MEMBER, date_time=FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            info.extra = b""
            info.comment = b""
            with archive.open(info, mode="w", force_zip64=False) as payload:
                _write_zeros(payload, padding_bytes)
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArchiveBudgetError(f"failed to append deterministic rate padding: {exc}") from exc
    actual_padding = verify_rate_match_member(destination)
    if actual_padding != padding_bytes:
        raise ArchiveBudgetError(
            f"padding parse-back mismatch: wrote {padding_bytes}, read {actual_padding}"
        )


def _temporary_path(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".rate_match_", suffix=".zip", dir=parent)
    os.close(descriptor)
    return Path(name)


def _zero_payload_sha256(size: int) -> str:
    digest = hashlib.sha256()
    remaining = int(size)
    while remaining:
        chunk = _ZERO_CHUNK[: min(remaining, len(_ZERO_CHUNK))]
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _SourceSnapshot:
    original_path: Path
    snapshot_path: Path
    archive_sha256: str
    archive_bytes: int
    member_sha256: dict[str, str]


def _snapshot_validated_source(source: Path, snapshot: Path) -> _SourceSnapshot:
    """Copy one stable source generation and prove path bytes did not race the copy."""
    source_handle, _ = _open_validated_archive(source, require_padding=False)
    source_handle.close()
    before_members = zip_member_hashes(source, include_padding=False)
    before_sha256 = file_sha256(source)
    before_bytes = source.stat().st_size
    shutil.copyfile(source, snapshot)
    snapshot_sha256 = file_sha256(snapshot)
    after_sha256 = file_sha256(source)
    after_bytes = source.stat().st_size
    if not (
        before_sha256 == snapshot_sha256 == after_sha256
        and before_bytes == snapshot.stat().st_size == after_bytes
    ):
        raise ArchiveBudgetError("source archive changed while its immutable snapshot was captured")
    snapshot_handle, _ = _open_validated_archive(snapshot, require_padding=False)
    snapshot_handle.close()
    snapshot_members = zip_member_hashes(snapshot, include_padding=False)
    if snapshot_members != before_members:
        raise ArchiveBudgetError("source member bytes changed while snapshot custody was captured")
    return _SourceSnapshot(
        original_path=source,
        snapshot_path=snapshot,
        archive_sha256=snapshot_sha256,
        archive_bytes=before_bytes,
        member_sha256=snapshot_members,
    )


def _prevalidate_source_without_copy(source: Path) -> None:
    handle, _ = _open_validated_archive(source, require_padding=False)
    handle.close()
    zip_member_hashes(source, include_padding=False)


def _require_source_still_matches_snapshot(snapshot: _SourceSnapshot) -> None:
    try:
        same = (
            snapshot.original_path.stat().st_size == snapshot.archive_bytes
            and file_sha256(snapshot.original_path) == snapshot.archive_sha256
        )
    except OSError as exc:
        raise ArchiveBudgetError("source archive disappeared after snapshot custody") from exc
    if not same:
        raise ArchiveBudgetError("source archive changed after immutable snapshot custody")


def _require_local_snapshot_immutable(snapshot: _SourceSnapshot) -> None:
    if (
        snapshot.snapshot_path.stat().st_size != snapshot.archive_bytes
        or file_sha256(snapshot.snapshot_path) != snapshot.archive_sha256
    ):
        raise ArchiveBudgetError("immutable local source snapshot changed during rate matching")


@dataclass(frozen=True)
class MatchedArchiveReceipt:
    """Custody for one side of an equal-byte comparison."""

    source_archive_sha256: str
    matched_archive_sha256: str
    source_archive_bytes: int
    matched_archive_bytes: int
    padding_bytes: int
    padding_sha256: str
    original_member_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EqualArchiveBudgetReceipt:
    """Exact two-arm rate-matching receipt."""

    version: str
    padding_member: str
    fixed_zip_timestamp: tuple[int, int, int, int, int, int]
    target_archive_bytes: int
    equal_archive_bytes: bool
    left: MatchedArchiveReceipt
    right: MatchedArchiveReceipt
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _arm_receipt(
    source: _SourceSnapshot, matched: Path, padding_bytes: int
) -> MatchedArchiveReceipt:
    _require_local_snapshot_immutable(source)
    members = verify_original_members_preserved(source.snapshot_path, matched)
    if members != source.member_sha256:
        raise ArchiveBudgetError("matched archive members do not bind the source snapshot")
    return MatchedArchiveReceipt(
        source_archive_sha256=source.archive_sha256,
        matched_archive_sha256=file_sha256(matched),
        source_archive_bytes=source.archive_bytes,
        matched_archive_bytes=matched.stat().st_size,
        padding_bytes=int(padding_bytes),
        padding_sha256=_zero_payload_sha256(padding_bytes),
        original_member_sha256=members,
    )


def verify_matched_archive_receipt(
    matched_archive: str | Path, receipt: MatchedArchiveReceipt
) -> bool:
    """Re-derive one published destination against its pre-publication receipt."""
    matched = Path(matched_archive)
    if matched.stat().st_size != receipt.matched_archive_bytes:
        raise ArchiveBudgetError("published matched archive size differs from receipt")
    if file_sha256(matched) != receipt.matched_archive_sha256:
        raise ArchiveBudgetError("published matched archive hash differs from receipt")
    if verify_rate_match_member(matched) != receipt.padding_bytes:
        raise ArchiveBudgetError("published padding length differs from receipt")
    if zip_member_hashes(matched, include_padding=False) != receipt.original_member_sha256:
        raise ArchiveBudgetError("published original member hashes differ from receipt")
    if _zero_payload_sha256(receipt.padding_bytes) != receipt.padding_sha256:
        raise ArchiveBudgetError("published padding hash differs from receipt")
    return True


def _verify_published_pair(
    left: Path,
    right: Path,
    left_receipt: MatchedArchiveReceipt,
    right_receipt: MatchedArchiveReceipt,
    target_bytes: int,
) -> None:
    verify_matched_archive_receipt(left, left_receipt)
    verify_matched_archive_receipt(right, right_receipt)
    if left.stat().st_size != target_bytes or right.stat().st_size != target_bytes:
        raise ArchiveBudgetError("published archive pair is not exactly equal-byte")


def verify_equal_archive_budget_receipt(
    left_archive: str | Path,
    right_archive: str | Path,
    receipt: EqualArchiveBudgetReceipt,
) -> bool:
    """Re-derive the full two-arm receipt from the final published paths."""
    if (
        receipt.version != _RECEIPT_VERSION
        or receipt.padding_member != PADDING_MEMBER
        or receipt.fixed_zip_timestamp != FIXED_ZIP_TIMESTAMP
        or receipt.equal_archive_bytes is not True
    ):
        raise ArchiveBudgetError("equal-archive receipt schema/constants are invalid")
    fields: dict[str, Any] = {
        "version": receipt.version,
        "padding_member": receipt.padding_member,
        "fixed_zip_timestamp": list(receipt.fixed_zip_timestamp),
        "target_archive_bytes": receipt.target_archive_bytes,
        "equal_archive_bytes": receipt.equal_archive_bytes,
        "left": receipt.left.to_dict(),
        "right": receipt.right.to_dict(),
    }
    if _canonical_hash(fields) != receipt.receipt_sha256:
        raise ArchiveBudgetError("equal-archive receipt canonical hash mismatch")
    if (
        receipt.left.matched_archive_bytes != receipt.target_archive_bytes
        or receipt.right.matched_archive_bytes != receipt.target_archive_bytes
    ):
        raise ArchiveBudgetError("equal-archive receipt does not encode exact target equality")
    _verify_published_pair(
        Path(left_archive),
        Path(right_archive),
        receipt.left,
        receipt.right,
        receipt.target_archive_bytes,
    )
    return True


def _backup_existing_destination(destination: Path, scratch: list[Path]) -> Path | None:
    if not destination.exists():
        return None
    if destination.is_symlink() or not destination.is_file():
        raise ArchiveBudgetError(f"output destination must be a regular file path: {destination}")
    backup = _temporary_path(destination.parent)
    scratch.append(backup)
    shutil.copyfile(destination, backup)
    if file_sha256(backup) != file_sha256(destination):
        raise ArchiveBudgetError(f"could not take a stable output rollback backup: {destination}")
    return backup


def _restore_destination(destination: Path, backup: Path | None) -> None:
    if backup is None:
        destination.unlink(missing_ok=True)
    else:
        _ROLLBACK_REPLACE(backup, destination)


def _publish_pair_transactionally(
    left_staged: Path,
    right_staged: Path,
    left_destination: Path,
    right_destination: Path,
    *,
    verify_published: Callable[[], None],
    scratch: list[Path],
) -> None:
    """Publish both paths or restore the exact pre-call destination pair."""
    left_backup = _backup_existing_destination(left_destination, scratch)
    right_backup = _backup_existing_destination(right_destination, scratch)
    try:
        _PAIR_PUBLISH_REPLACE(left_staged, left_destination)
        _PAIR_PUBLISH_REPLACE(right_staged, right_destination)
        verify_published()
    except BaseException as publication_error:
        rollback_errors: list[str] = []
        for destination, backup in (
            (left_destination, left_backup),
            (right_destination, right_backup),
        ):
            try:
                _restore_destination(destination, backup)
                if backup is not None and backup in scratch:
                    scratch.remove(backup)
            except BaseException as rollback_error:  # pragma: no cover - catastrophic filesystem loss
                rollback_errors.append(f"{destination}: {rollback_error}")
        if rollback_errors:
            raise ArchiveBudgetError(
                "pair publication failed and rollback was incomplete: " + "; ".join(rollback_errors)
            ) from publication_error
        raise
    else:
        for backup in (left_backup, right_backup):
            if backup is not None:
                backup.unlink(missing_ok=True)
                if backup in scratch:
                    scratch.remove(backup)


def equalize_archive_budgets(
    left_source: str | Path,
    right_source: str | Path,
    left_output: str | Path,
    right_output: str | Path,
) -> EqualArchiveBudgetReceipt:
    """Add deterministic padding so two final ZIP files have exactly equal size.

    Inputs remain untouched.  Outputs are written through sibling temporary
    files and replaced only after both parse back, preserve every original
    member payload, and have byte-for-byte equal lengths.
    """
    left_src, right_src = Path(left_source), Path(right_source)
    left_dst, right_dst = Path(left_output), Path(right_output)
    resolved_sources = {left_src.resolve(), right_src.resolve()}
    resolved_outputs = {left_dst.resolve(), right_dst.resolve()}
    if len(resolved_outputs) != 2:
        raise ArchiveBudgetError("left_output and right_output must be distinct paths")
    if resolved_sources & resolved_outputs:
        raise ArchiveBudgetError("rate matching never overwrites a source archive")

    # Validate both sides, including CRCs and extraction safety, before the
    # first snapshot copy.  Snapshot capture repeats this under hash custody.
    _prevalidate_source_without_copy(left_src)
    _prevalidate_source_without_copy(right_src)

    scratch: list[Path] = []
    try:
        # Every later read is from these immutable, content-bound snapshots.
        # The original paths are checked again before publication.
        left_snapshot_path = _temporary_path(left_dst.parent)
        right_snapshot_path = _temporary_path(right_dst.parent)
        scratch.extend((left_snapshot_path, right_snapshot_path))
        left_snapshot = _snapshot_validated_source(left_src, left_snapshot_path)
        right_snapshot = _snapshot_validated_source(right_src, right_snapshot_path)

        left_zero = _temporary_path(left_dst.parent)
        right_zero = _temporary_path(right_dst.parent)
        scratch.extend((left_zero, right_zero))
        _append_padding_copy(left_snapshot.snapshot_path, left_zero, 0)
        _append_padding_copy(right_snapshot.snapshot_path, right_zero, 0)
        left_minimum = left_zero.stat().st_size
        right_minimum = right_zero.stat().st_size
        target = max(left_minimum, right_minimum)
        left_padding = target - left_minimum
        right_padding = target - right_minimum

        left_final = _temporary_path(left_dst.parent)
        right_final = _temporary_path(right_dst.parent)
        scratch.extend((left_final, right_final))
        _append_padding_copy(left_snapshot.snapshot_path, left_final, left_padding)
        _append_padding_copy(right_snapshot.snapshot_path, right_final, right_padding)
        if left_final.stat().st_size != target or right_final.stat().st_size != target:
            raise ArchiveBudgetError(
                "exact archive-size equality failed after deterministic padding render"
            )
        left_receipt = _arm_receipt(left_snapshot, left_final, left_padding)
        right_receipt = _arm_receipt(right_snapshot, right_final, right_padding)
        if left_receipt.matched_archive_bytes != right_receipt.matched_archive_bytes:
            raise ArchiveBudgetError("final archive sizes differ by at least one byte")

        fields: dict[str, Any] = {
            "version": _RECEIPT_VERSION,
            "padding_member": PADDING_MEMBER,
            "fixed_zip_timestamp": list(FIXED_ZIP_TIMESTAMP),
            "target_archive_bytes": target,
            "equal_archive_bytes": True,
            "left": left_receipt.to_dict(),
            "right": right_receipt.to_dict(),
        }
        receipt = EqualArchiveBudgetReceipt(
            version=_RECEIPT_VERSION,
            padding_member=PADDING_MEMBER,
            fixed_zip_timestamp=FIXED_ZIP_TIMESTAMP,
            target_archive_bytes=target,
            equal_archive_bytes=True,
            left=left_receipt,
            right=right_receipt,
            receipt_sha256=_canonical_hash(fields),
        )
        _require_source_still_matches_snapshot(left_snapshot)
        _require_source_still_matches_snapshot(right_snapshot)

        left_dst.parent.mkdir(parents=True, exist_ok=True)
        right_dst.parent.mkdir(parents=True, exist_ok=True)
        _publish_pair_transactionally(
            left_final,
            right_final,
            left_dst,
            right_dst,
            verify_published=lambda: verify_equal_archive_budget_receipt(
                left_dst, right_dst, receipt
            ),
            scratch=scratch,
        )
        return receipt
    finally:
        for path in scratch:
            path.unlink(missing_ok=True)
