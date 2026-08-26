"""Lossless content-addressed retention for large evaluator payload trees.

The store is deliberately boring: immutable SHA-256-named byte objects plus a
manifest that records every logical file and its ordered object sequence.  The
chunk boundaries expose two exact repetitions in WD3 evaluation trees without
changing any payload bytes:

* camera RGB files are split at one-frame boundaries; and
* NPZ/ZIP files are split at local-member boundaries.

Everything else is a whole-file object.  A tree can be restored without
symlinks and is accepted only after every reconstructed file matches its
recorded byte count and SHA-256.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import stat
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile

SCHEMA = "tac_content_addressed_tree.v1"
INVENTORY_SCHEMA = "tac_content_addressed_inventory.v1"
CAMERA_FRAME_BYTES = 874 * 1164 * 3
COPY_BLOCK_BYTES = 8 * 1024 * 1024


class ContentAddressedRetentionError(RuntimeError):
    """Retention refused a lossy, ambiguous, or corrupt operation."""


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = _canonical_bytes(value)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _object_path(store: Path, digest: str) -> Path:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContentAddressedRetentionError("invalid SHA-256 object identity")
    return store / "objects" / "sha256" / digest[:2] / digest


def _read_exact(stream: BinaryIO, length: int) -> Iterable[bytes]:
    remaining = length
    while remaining:
        block = stream.read(min(remaining, COPY_BLOCK_BYTES))
        if not block:
            raise ContentAddressedRetentionError("payload ended before its declared chunk boundary")
        remaining -= len(block)
        yield block


def _chunk_ranges(path: Path) -> tuple[str, list[tuple[int, int]]]:
    size = path.stat().st_size
    if path.name == "receiver_pairs.rgb.u8":
        if size == 0 or size % CAMERA_FRAME_BYTES:
            raise ContentAddressedRetentionError(f"camera payload is not frame-aligned: {path}")
        return "camera_frame_v1", [
            (offset, min(CAMERA_FRAME_BYTES, size - offset))
            for offset in range(0, size, CAMERA_FRAME_BYTES)
        ]
    if path.suffix.lower() in {".npz", ".zip"} and size:
        try:
            with ZipFile(path, "r") as archive:
                offsets = sorted({int(info.header_offset) for info in archive.infolist()})
                start_dir = int(archive.start_dir)
        except (BadZipFile, OSError, ValueError):
            return "whole_file_v1", [(0, size)]
        if offsets and offsets[0] == 0 and all(0 <= offset < start_dir for offset in offsets):
            boundaries = [*offsets, start_dir]
            ranges = [
                (start, end - start)
                for start, end in itertools.pairwise(boundaries)
                if end > start
            ]
            if start_dir < size:
                ranges.append((start_dir, size - start_dir))
            if ranges and sum(length for _, length in ranges) == size:
                return "zip_local_member_v1", ranges
    return "whole_file_v1", [(0, size)]


def _scan_file(
    path: Path,
    *,
    relative_path: str,
    store: Path | None,
    materialize: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ContentAddressedRetentionError(f"logical payload is not a regular file: {path}")
    chunking, ranges = _chunk_ranges(path)
    file_hasher = hashlib.sha256()
    chunks: list[dict[str, Any]] = []
    objects: dict[str, int] = {}
    with path.open("rb") as stream:
        expected_offset = 0
        for offset, length in ranges:
            if offset != expected_offset or length < 0:
                raise ContentAddressedRetentionError(f"non-contiguous chunk plan: {path}")
            stream.seek(offset)
            chunk_hasher = hashlib.sha256()
            temporary: Path | None = None
            output: BinaryIO | None = None
            if materialize:
                if store is None:
                    raise ContentAddressedRetentionError("materialization requires a CAS root")
                temporary = store / "tmp" / f"{os.getpid()}-{offset}-{path.name}.partial"
                temporary.parent.mkdir(parents=True, exist_ok=True)
                output = temporary.open("wb")
            try:
                try:
                    for block in _read_exact(stream, length):
                        chunk_hasher.update(block)
                        file_hasher.update(block)
                        if output is not None:
                            output.write(block)
                    if output is not None:
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    if output is not None:
                        output.close()
                digest = chunk_hasher.hexdigest()
                objects.setdefault(digest, length)
                chunks.append({"sha256": digest, "bytes": length})
                if materialize:
                    assert store is not None and temporary is not None
                    destination = _object_path(store, digest)
                    if destination.exists():
                        if not destination.is_file() or destination.stat().st_size != length:
                            raise ContentAddressedRetentionError(f"CAS object identity collision: {destination}")
                        temporary.unlink()
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(temporary, destination)
            except BaseException:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                raise
            expected_offset += length
    size = path.stat().st_size
    if expected_offset != size:
        raise ContentAddressedRetentionError(f"chunk plan did not cover the complete file: {path}")
    return (
        {
            "path": relative_path,
            "bytes": size,
            "sha256": file_hasher.hexdigest(),
            "mode": stat.S_IMODE(mode),
            "chunking": chunking,
            "chunks": chunks,
        },
        objects,
    )


def scan_tree(
    root: Path,
    *,
    store: Path | None = None,
    materialize: bool = False,
    exclude_relative: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, int]]:
    """Scan one tree and optionally materialize its immutable CAS objects."""

    root = root.resolve()
    if not root.is_dir():
        raise ContentAddressedRetentionError(f"logical tree is absent: {root}")
    if materialize:
        if store is None:
            raise ContentAddressedRetentionError("materialization requires a CAS root")
        resolved_store = store.resolve()
        if resolved_store == root or root in resolved_store.parents:
            raise ContentAddressedRetentionError("CAS root must stay outside the logical tree")
    excluded = set(exclude_relative)
    files: list[dict[str, Any]] = []
    objects: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ContentAddressedRetentionError(f"symlink payload is forbidden: {path}")
        if path.is_dir():
            continue
        record, discovered = _scan_file(
            path,
            relative_path=relative,
            store=store,
            materialize=materialize,
        )
        files.append(record)
        for digest, length in discovered.items():
            prior = objects.setdefault(digest, length)
            if prior != length:
                raise ContentAddressedRetentionError("same object hash acquired two byte lengths")
    manifest = {
        "schema": SCHEMA,
        "logical_root": str(root),
        "files": files,
        "logical_file_count": len(files),
        "logical_bytes": sum(int(record["bytes"]) for record in files),
        "logical_allocated_bytes": sum(
            (root / record["path"]).stat().st_blocks * 512 for record in files
        ),
        "unique_objects_within_tree": len(objects),
        "unique_object_bytes_within_tree": sum(objects.values()),
        "all_payloads_recoverable": True,
        "symlinks_used": False,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    return manifest, objects


def retain_tree(
    root: Path,
    *,
    store: Path,
    manifest_path: Path,
    compact: bool,
    exclude_relative: Sequence[str] = (),
) -> dict[str, Any]:
    """Materialize, deep-verify, and optionally compact a logical tree."""

    root = root.resolve()
    store = store.resolve()
    manifest_path = manifest_path.resolve()
    if manifest_path.parent != root:
        raise ContentAddressedRetentionError("tree manifest must be a direct child of its logical root")
    manifest, _ = scan_tree(
        root,
        store=store,
        materialize=True,
        exclude_relative=exclude_relative,
    )
    manifest["cas_root"] = str(store)
    manifest["compacted"] = bool(compact)
    _atomic_json(manifest_path, manifest)
    verify_manifest(manifest_path, deep=True)
    if compact:
        compact_retained_tree(manifest_path, root=root, keep_relative=(manifest_path.name,))
    return manifest


def _validated_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("all_payloads_recoverable") is not True:
        raise ContentAddressedRetentionError(f"retention manifest is incomplete: {path}")
    claimed = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    # cas_root/compacted are added after the source-tree identity is sealed and
    # therefore are custody metadata, not part of the logical-tree digest.
    body.pop("cas_root", None)
    body.pop("compacted", None)
    actual = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if claimed != actual:
        raise ContentAddressedRetentionError(f"retention manifest identity differs: {path}")
    return manifest


def verify_manifest(path: Path, *, deep: bool = False) -> dict[str, Any]:
    manifest = _validated_manifest(path)
    store = Path(manifest["cas_root"])
    checked: set[str] = set()
    for record in manifest["files"]:
        if Path(record["path"]).is_absolute() or ".." in Path(record["path"]).parts:
            raise ContentAddressedRetentionError("manifest contains an unsafe logical path")
        if sum(int(chunk["bytes"]) for chunk in record["chunks"]) != int(record["bytes"]):
            raise ContentAddressedRetentionError("manifest chunks do not cover their logical file")
        for chunk in record["chunks"]:
            digest = str(chunk["sha256"])
            object_path = _object_path(store, digest)
            if not object_path.is_file() or object_path.stat().st_size != int(chunk["bytes"]):
                raise ContentAddressedRetentionError(f"required CAS object is absent: {object_path}")
            if deep and digest not in checked:
                with object_path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != digest:
                    raise ContentAddressedRetentionError(f"CAS object bytes differ: {object_path}")
                checked.add(digest)
    return manifest


def read_logical_bytes(manifest_path: Path, relative_path: str, *, maximum_bytes: int = 16 * 1024 * 1024) -> bytes:
    """Read one small logical file directly from CAS without restoring its tree."""

    manifest = verify_manifest(manifest_path, deep=False)
    matches = [record for record in manifest["files"] if record["path"] == relative_path]
    if len(matches) != 1:
        raise ContentAddressedRetentionError(f"manifest has no unique logical file: {relative_path}")
    record = matches[0]
    if int(record["bytes"]) > maximum_bytes:
        raise ContentAddressedRetentionError(f"logical file exceeds bounded read limit: {relative_path}")
    store = Path(manifest["cas_root"])
    payload = bytearray()
    for chunk in record["chunks"]:
        payload.extend(_object_path(store, str(chunk["sha256"])).read_bytes())
    if len(payload) != int(record["bytes"]) or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ContentAddressedRetentionError(f"logical file identity differs during bounded read: {relative_path}")
    return bytes(payload)


def compact_retained_tree(
    manifest_path: Path,
    *,
    root: Path,
    keep_relative: Sequence[str] = (),
) -> None:
    """Delete only manifest-covered logical copies after their CAS proof passes."""

    manifest = verify_manifest(manifest_path, deep=False)
    root = root.resolve()
    keep = set(keep_relative)
    for record in manifest["files"]:
        relative = str(record["path"])
        if relative in keep:
            continue
        path = root / relative
        if root != path.resolve() and root not in path.resolve().parents:
            raise ContentAddressedRetentionError("compaction path escaped its logical root")
        if path.is_symlink():
            raise ContentAddressedRetentionError(f"symlink appeared during compaction: {path}")
        if path.exists():
            if not path.is_file():
                raise ContentAddressedRetentionError(f"logical file became a non-file: {path}")
            path.unlink()
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def restore_tree(manifest_path: Path, destination: Path) -> dict[str, Any]:
    """Restore a manifest to regular files and verify complete file identity."""

    manifest = verify_manifest(manifest_path, deep=False)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    store = Path(manifest["cas_root"])
    restored = []
    for record in manifest["files"]:
        output = destination / record["path"]
        if destination != output.resolve() and destination not in output.resolve().parents:
            raise ContentAddressedRetentionError("restore path escaped its destination")
        if output.exists():
            raise ContentAddressedRetentionError(f"restore refuses to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        hasher = hashlib.sha256()
        written = 0
        with temporary.open("wb") as stream:
            for chunk in record["chunks"]:
                object_path = _object_path(store, str(chunk["sha256"]))
                with object_path.open("rb") as source:
                    for block in iter(lambda: source.read(COPY_BLOCK_BYTES), b""):
                        stream.write(block)
                        hasher.update(block)
                        written += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if written != int(record["bytes"]) or hasher.hexdigest() != record["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ContentAddressedRetentionError(f"restored logical file identity differs: {output}")
        os.chmod(temporary, int(record["mode"]))
        os.replace(temporary, output)
        restored.append({"path": str(output), "bytes": written, "sha256": hasher.hexdigest()})
    return {
        "schema": "tac_content_addressed_restore.v1",
        "manifest": str(manifest_path.resolve()),
        "destination": str(destination),
        "files": restored,
        "all_files_byte_identical": True,
        "symlinks_used": False,
    }


def restore_logical_file(manifest_path: Path, relative_path: str, output: Path) -> dict[str, Any]:
    """Restore one arbitrarily large logical payload to a regular file."""

    manifest = verify_manifest(manifest_path, deep=False)
    matches = [record for record in manifest["files"] if record["path"] == relative_path]
    if len(matches) != 1:
        raise ContentAddressedRetentionError(f"manifest has no unique logical file: {relative_path}")
    record = matches[0]
    output = output.resolve()
    if output.exists():
        raise ContentAddressedRetentionError(f"restore refuses to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    store = Path(manifest["cas_root"])
    hasher = hashlib.sha256()
    written = 0
    try:
        with temporary.open("wb") as stream:
            for chunk in record["chunks"]:
                object_path = _object_path(store, str(chunk["sha256"]))
                with object_path.open("rb") as source:
                    for block in iter(lambda: source.read(COPY_BLOCK_BYTES), b""):
                        stream.write(block)
                        hasher.update(block)
                        written += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if written != int(record["bytes"]) or hasher.hexdigest() != record["sha256"]:
            raise ContentAddressedRetentionError(f"restored logical file identity differs: {output}")
        os.chmod(temporary, int(record["mode"]))
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(output),
        "bytes": written,
        "sha256": hasher.hexdigest(),
        "symlink": False,
    }


def inventory_trees(roots: Sequence[Path], *, block_size: int = 4096) -> dict[str, Any]:
    """Measure logical and cross-tree unique bytes without creating CAS objects."""

    if block_size < 512 or block_size & (block_size - 1):
        raise ContentAddressedRetentionError("allocation block size must be a power of two >= 512")
    global_objects: dict[str, int] = {}
    trees = []
    manifest_bytes = 0
    for root in roots:
        manifest, objects = scan_tree(root, materialize=False)
        encoded_bytes = len(_canonical_bytes(manifest))
        manifest_bytes += encoded_bytes
        trees.append(
            {
                "root": str(root.resolve()),
                "logical_bytes": manifest["logical_bytes"],
                "logical_allocated_bytes": manifest["logical_allocated_bytes"],
                "logical_file_count": manifest["logical_file_count"],
                "unique_objects_within_tree": manifest["unique_objects_within_tree"],
                "unique_object_bytes_within_tree": manifest["unique_object_bytes_within_tree"],
                "manifest_sha256": manifest["manifest_sha256"],
                "projected_manifest_bytes": encoded_bytes,
            }
        )
        for digest, length in objects.items():
            prior = global_objects.setdefault(digest, length)
            if prior != length:
                raise ContentAddressedRetentionError("same object hash acquired two byte lengths")
    object_bytes = sum(global_objects.values())
    object_allocated = sum(((length + block_size - 1) // block_size) * block_size for length in global_objects.values())
    manifest_allocated = sum(
        ((int(tree["projected_manifest_bytes"]) + block_size - 1) // block_size) * block_size
        for tree in trees
    )
    return {
        "schema": INVENTORY_SCHEMA,
        "chunking": {
            "receiver_pairs.rgb.u8": "camera_frame_v1",
            "npz_and_zip": "zip_local_member_v1",
            "other": "whole_file_v1",
        },
        "block_size": block_size,
        "trees": trees,
        "tree_count": len(trees),
        "logical_bytes": sum(int(tree["logical_bytes"]) for tree in trees),
        "logical_allocated_bytes": sum(int(tree["logical_allocated_bytes"]) for tree in trees),
        "unique_object_count": len(global_objects),
        "unique_object_bytes": object_bytes,
        "unique_object_allocated_bytes": object_allocated,
        "projected_manifest_bytes": manifest_bytes,
        "projected_manifest_allocated_bytes": manifest_allocated,
        "post_dedup_allocated_bytes": object_allocated + manifest_allocated,
        "dedup_saved_allocated_bytes": sum(int(tree["logical_allocated_bytes"]) for tree in trees)
        - object_allocated
        - manifest_allocated,
        "measurement": "source-backed exact SHA-256 inventory; no source payload changed",
        "all_payloads_recoverable_by_manifest": True,
    }


def inventory_cohorts(
    cohorts: dict[str, Sequence[Path]], *, block_size: int = 4096
) -> dict[str, Any]:
    """Measure overlapping tree cohorts while hashing every source tree only once."""

    if not cohorts or block_size < 512 or block_size & (block_size - 1):
        raise ContentAddressedRetentionError("cohorts and allocation block size must be valid")
    resolved = {
        name: tuple(Path(root).resolve() for root in roots)
        for name, roots in cohorts.items()
    }
    if any(not roots for roots in resolved.values()):
        raise ContentAddressedRetentionError("every inventory cohort must contain a tree")
    scanned: dict[Path, tuple[dict[str, Any], dict[str, int], int]] = {}
    for root in sorted({root for roots in resolved.values() for root in roots}):
        manifest, objects = scan_tree(root, materialize=False)
        scanned[root] = (manifest, objects, len(_canonical_bytes(manifest)))
    results: dict[str, Any] = {}
    for name, roots in sorted(resolved.items()):
        if len(set(roots)) != len(roots):
            raise ContentAddressedRetentionError(f"inventory cohort repeats a logical tree: {name}")
        objects: dict[str, int] = {}
        tree_rows = []
        for root in roots:
            manifest, discovered, encoded_bytes = scanned[root]
            tree_rows.append(
                {
                    "root": str(root),
                    "logical_bytes": manifest["logical_bytes"],
                    "logical_allocated_bytes": manifest["logical_allocated_bytes"],
                    "logical_file_count": manifest["logical_file_count"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "projected_manifest_bytes": encoded_bytes,
                }
            )
            for digest, length in discovered.items():
                prior = objects.setdefault(digest, length)
                if prior != length:
                    raise ContentAddressedRetentionError("same object hash acquired two byte lengths")
        object_allocated = sum(
            ((length + block_size - 1) // block_size) * block_size for length in objects.values()
        )
        manifest_allocated = sum(
            ((int(row["projected_manifest_bytes"]) + block_size - 1) // block_size) * block_size
            for row in tree_rows
        )
        logical_allocated = sum(int(row["logical_allocated_bytes"]) for row in tree_rows)
        results[name] = {
            "tree_count": len(tree_rows),
            "trees": tree_rows,
            "logical_bytes": sum(int(row["logical_bytes"]) for row in tree_rows),
            "logical_allocated_bytes": logical_allocated,
            "unique_object_count": len(objects),
            "unique_object_bytes": sum(objects.values()),
            "unique_object_allocated_bytes": object_allocated,
            "projected_manifest_allocated_bytes": manifest_allocated,
            "post_dedup_allocated_bytes": object_allocated + manifest_allocated,
            "dedup_saved_allocated_bytes": logical_allocated - object_allocated - manifest_allocated,
        }
    return {
        "schema": "tac_content_addressed_cohort_inventory.v1",
        "block_size": block_size,
        "chunking": {
            "receiver_pairs.rgb.u8": "camera_frame_v1",
            "npz_and_zip": "zip_local_member_v1",
            "other": "whole_file_v1",
        },
        "source_tree_count": len(scanned),
        "cohorts": results,
        "measurement": "source-backed exact SHA-256 inventory; each source tree hashed once; no source payload changed",
        "all_payloads_recoverable_by_manifest": True,
    }
