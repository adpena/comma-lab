#!/usr/bin/env python3
"""Losslessly certify, compress, verify, and reclaim one closed AP retention tree.

This is the executable safety boundary for ``ddm_sr3``.  It never removes an
original path until all of the following are durable:

* a manifest with SHA-256 for every regular file (plus directory/symlink rows),
* a deterministic tar stream compressed by the recorded zstd CLI,
* a local-scratch extraction whose complete manifest equals the source,
* a machine-readable pre-reclaim certificate, and
* a second full source re-hash immediately before removal.

The stages are resumable from the files written at the tree root.  Incomplete
``*.partial`` outputs are certified as rebuildable scratch before cleanup.
Only explicitly named top-level paths from the original manifest are removed;
the tree root and the SR3 custody artifacts are always retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

AP_ROOT = Path("/Volumes/APDataStore/pact")
LOCAL_SCRATCH_ROOT = Path("/private/tmp")
PROTECTED_TREES = {
    AP_ROOT / "ddm_bs3_born_small_resolved",
    AP_ROOT / "ddm_w96a_aligned_window",
}
CUSTODY_PREFIXES = ("cold_store", "vertigo_coldstore")
RECENT_SECONDS = 24 * 60 * 60
LOCAL_MARGIN_BYTES = 8 * 2**30
AP_ABORT_FLOOR_BYTES = 2 * 2**30
ZSTD_LEVEL = 15
ZSTD_LONG = 31
ZSTD_THREADS = 4

MANIFEST_NAME = "SR3_ORIGINAL_MANIFEST.jsonl"
ARCHIVE_NAME = "SR3_ORIGINAL_TREE.tar.zst"
VERIFY_NAME = "SR3_VERIFICATION_RECEIPT.json"
CERT_NAME = "SR3_RECLAIM_CERTIFICATE.json"
PROGRESS_NAME = "SR3_PROGRESS.json"
FAILURE_NAME = "SR3_FAILURE_RECEIPTS.jsonl"
RESERVED_NAMES = {
    MANIFEST_NAME,
    ARCHIVE_NAME,
    VERIFY_NAME,
    CERT_NAME,
    PROGRESS_NAME,
    FAILURE_NAME,
    f"{MANIFEST_NAME}.partial",
    f"{ARCHIVE_NAME}.partial",
    f"{VERIFY_NAME}.partial",
    f"{CERT_NAME}.partial",
    f"{PROGRESS_NAME}.partial",
    f"._{MANIFEST_NAME}",
    f"._{ARCHIVE_NAME}",
    f"._{VERIFY_NAME}",
    f"._{CERT_NAME}",
    f"._{PROGRESS_NAME}",
    f"._{FAILURE_NAME}",
}
SCHEMA = "ddm_sr3_ap_certify_compress_reclaim.v1"


class CertifyError(RuntimeError):
    """The operation cannot continue without weakening the custody proof."""


@dataclass(frozen=True)
class Entry:
    path: str
    kind: str
    mode: int
    mtime_ns: int
    bytes: int = 0
    sha256: str | None = None
    link_target: str | None = None

    def as_dict(self) -> dict[str, object]:
        row: dict[str, object] = {
            "path": self.path,
            "type": self.kind,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
        }
        if self.kind == "file":
            row.update({"bytes": self.bytes, "sha256": self.sha256})
        elif self.kind == "symlink":
            row["link_target"] = self.link_target
        return row


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def progress(message: str) -> None:
    """Best-effort telemetry; custody work must not die when a caller detaches."""
    try:
        print(message, file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass


def emit_json(value: object) -> None:
    try:
        print(json.dumps(value, indent=2, sort_keys=True))
    except BrokenPipeError:
        pass


def sha256_file(path: Path, chunk_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def fs_bytes(path: Path) -> dict[str, int | str]:
    values = os.statvfs(path)
    return {
        "path": str(path),
        "available_bytes": values.f_bavail * values.f_frsize,
        "free_bytes": values.f_bfree * values.f_frsize,
        "block_size": values.f_frsize,
    }


def allocated_bytes(path: Path) -> int:
    result = subprocess.run(
        ["du", "-sk", str(path)], capture_output=True, text=True, check=True
    )
    return int(result.stdout.split()[0]) * 1024


def validate_tree(raw: str) -> Path:
    raw_tree = Path(raw)
    if not raw_tree.is_absolute():
        raise CertifyError("--tree must be an absolute path")
    if raw_tree.is_symlink():
        raise CertifyError(f"tree must not be a symlink: {raw_tree}")
    tree = raw_tree.resolve(strict=True)
    resolved_parent = tree.parent.resolve()
    if resolved_parent != AP_ROOT.resolve():
        raise CertifyError(f"tree must be a direct child of {AP_ROOT}: {tree}")
    if tree.is_symlink() or not tree.is_dir():
        raise CertifyError(f"tree must be a real directory: {tree}")
    if tree in PROTECTED_TREES:
        raise CertifyError(f"explicitly protected live store: {tree}")
    if tree.name.startswith(CUSTODY_PREFIXES):
        raise CertifyError(f"custody namespace requires separate certificate adjudication: {tree}")
    return tree


def assert_fleet_idle(repo: Path) -> str:
    result = subprocess.run(
        [str(repo / ".venv/bin/python"), str(repo / "tools/codex_arm_queue.py"), "status"],
        capture_output=True,
        text=True,
        check=True,
    )
    first = next((line for line in result.stdout.splitlines() if "codex arms live:" in line), "")
    if "codex arms live: 0/" not in first:
        raise CertifyError(f"fleet is not proven idle: {first or 'status row absent'}")
    return first.strip()


def _path_kind(path: Path, st: os.stat_result) -> str:
    if stat.S_ISREG(st.st_mode):
        return "file"
    if stat.S_ISDIR(st.st_mode):
        return "dir"
    if stat.S_ISLNK(st.st_mode):
        return "symlink"
    raise CertifyError(f"unsupported filesystem object: {path} mode={oct(st.st_mode)}")


def scan_tree(tree: Path) -> list[Entry]:
    """Return every original descendant, excluding only exact SR3 custody names."""
    entries: list[Entry] = []
    for dirpath, dirnames, filenames in os.walk(tree, topdown=True, followlinks=False):
        at_root = Path(dirpath) == tree
        if at_root:
            dirnames[:] = sorted(name for name in dirnames if name not in RESERVED_NAMES)
        else:
            dirnames[:] = sorted(dirnames)
        for name in sorted(dirnames + filenames):
            if at_root and name in RESERVED_NAMES:
                continue
            path = Path(dirpath) / name
            st = path.lstat()
            kind = _path_kind(path, st)
            rel = path.relative_to(tree).as_posix()
            if kind == "symlink" and name in dirnames:
                dirnames.remove(name)
            entries.append(
                Entry(
                    path=rel,
                    kind=kind,
                    mode=stat.S_IMODE(st.st_mode),
                    mtime_ns=st.st_mtime_ns,
                    bytes=st.st_size if kind == "file" else 0,
                    link_target=os.readlink(path) if kind == "symlink" else None,
                )
            )
    entries.sort(key=lambda row: row.path)
    return entries


def _hash_entry(tree: Path, entry: Entry) -> Entry:
    path = tree / entry.path
    before = path.lstat()
    digest = sha256_file(path)
    after = path.lstat()
    if (
        before.st_size != entry.bytes
        or after.st_size != entry.bytes
        or before.st_mtime_ns != entry.mtime_ns
        or after.st_mtime_ns != entry.mtime_ns
    ):
        raise CertifyError(f"file changed while hashing: {path}")
    return Entry(**{**entry.__dict__, "sha256": digest})


def hash_entries(tree: Path, entries: Iterable[Entry], workers: int = 4) -> list[Entry]:
    base = list(entries)
    files = [entry for entry in base if entry.kind == "file"]
    hashed: dict[str, Entry] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_hash_entry, tree, entry): entry.path for entry in files}
        for index, future in enumerate(concurrent.futures.as_completed(pending), start=1):
            row = future.result()
            hashed[row.path] = row
            if index % 100 == 0 or index == len(files):
                progress(f"hashed {index}/{len(files)} files")
    return [hashed.get(entry.path, entry) for entry in base]


def canonical_manifest_bytes(entries: Iterable[Entry]) -> bytes:
    return b"".join(
        (json.dumps(entry.as_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        for entry in sorted(entries, key=lambda row: row.path)
    )


def write_manifest(path: Path, entries: list[Entry]) -> str:
    payload = canonical_manifest_bytes(entries)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return hashlib.sha256(payload).hexdigest()


def read_manifest(path: Path) -> tuple[list[Entry], str]:
    payload = path.read_bytes()
    rows: list[Entry] = []
    for line in payload.splitlines():
        data = json.loads(line)
        rows.append(
            Entry(
                path=data["path"],
                kind=data["type"],
                mode=data["mode"],
                mtime_ns=data["mtime_ns"],
                bytes=data.get("bytes", 0),
                sha256=data.get("sha256"),
                link_target=data.get("link_target"),
            )
        )
    if any(row.kind == "file" and not row.sha256 for row in rows):
        raise CertifyError("manifest contains an unhashed file")
    return rows, hashlib.sha256(payload).hexdigest()


def assert_entry_metadata_equal(
    expected: list[Entry], observed: list[Entry], *, require_mtime: bool = True
) -> None:
    expected_map = {row.path: row for row in expected}
    observed_map = {row.path: row for row in observed}
    if expected_map.keys() != observed_map.keys():
        missing = sorted(expected_map.keys() - observed_map.keys())[:10]
        extra = sorted(observed_map.keys() - expected_map.keys())[:10]
        raise CertifyError(f"path set changed; missing={missing} extra={extra}")
    for rel, want in expected_map.items():
        got = observed_map[rel]
        if (want.kind, want.bytes, want.link_target) != (got.kind, got.bytes, got.link_target):
            raise CertifyError(f"metadata changed for {rel}")
        if require_mtime and want.kind == "file" and want.mtime_ns != got.mtime_ns:
            raise CertifyError(f"mtime changed for {rel}")


def assert_hashes_equal(
    expected: list[Entry], observed: list[Entry], *, require_metadata: bool = True
) -> None:
    want = {row.path: row for row in expected}
    got = {row.path: row for row in observed}
    if want.keys() != got.keys():
        raise CertifyError("verified manifest path set differs")
    for rel, left in want.items():
        right = got[rel]
        left_content = (left.path, left.kind, left.bytes, left.sha256, left.link_target)
        right_content = (right.path, right.kind, right.bytes, right.sha256, right.link_target)
        if left_content != right_content:
            raise CertifyError(f"verified manifest differs at {rel}")
        if require_metadata and (left.mode, left.mtime_ns) != (right.mode, right.mtime_ns):
            raise CertifyError(f"verified source metadata differs at {rel}")


def assert_source_subset_equal(tree: Path, expected: list[Entry], observed: list[Entry]) -> None:
    """Verify a partially retired source after a crash in the reclaim stage."""
    want = {row.path: row for row in expected}
    got = {row.path: row for row in observed}
    extra = sorted(got.keys() - want.keys())[:10]
    if extra:
        raise CertifyError(f"unexpected paths appeared during reclaim resume: {extra}")
    for rel, row in got.items():
        expected_row = want[rel]
        if (row.kind, row.bytes, row.link_target) != (
            expected_row.kind,
            expected_row.bytes,
            expected_row.link_target,
        ):
            raise CertifyError(f"remaining source metadata differs at {rel}")
        if row.kind == "file" and row.mtime_ns != expected_row.mtime_ns:
            raise CertifyError(f"remaining source file mtime differs at {rel}")
    hashed = hash_entries(tree, observed)
    hashed_map = {row.path: row for row in hashed}
    for rel, row in hashed_map.items():
        expected_row = want[rel]
        if (row.bytes, row.sha256, row.link_target) != (
            expected_row.bytes,
            expected_row.sha256,
            expected_row.link_target,
        ):
            raise CertifyError(f"remaining source content differs at {rel}")


def _tar_info(entry: Entry) -> tarfile.TarInfo:
    name = entry.path + ("/" if entry.kind == "dir" else "")
    info = tarfile.TarInfo(name=name)
    info.mode = entry.mode
    info.mtime = entry.mtime_ns // 1_000_000_000
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if entry.kind == "file":
        info.type = tarfile.REGTYPE
        info.size = entry.bytes
    elif entry.kind == "dir":
        info.type = tarfile.DIRTYPE
        info.size = 0
    elif entry.kind == "symlink":
        info.type = tarfile.SYMTYPE
        info.linkname = entry.link_target or ""
        info.size = 0
    else:  # pragma: no cover - Entry validation makes this unreachable.
        raise CertifyError(f"unsupported manifest kind: {entry.kind}")
    return info


def zstd_version() -> str:
    result = subprocess.run(["zstd", "--version"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def cleanup_partial(tree: Path, partial: Path, reason: str) -> None:
    if not partial.exists():
        return
    receipt = {
        "schema": SCHEMA,
        "phase": "PARTIAL_SCRATCH_CLEANUP",
        "written_at_utc": utcnow(),
        "path": str(partial),
        "bytes": partial.stat().st_size,
        "sha256": sha256_file(partial),
        "reason": reason,
        "rebuild": "rerun this certifier from the durable source manifest and original tree",
    }
    append_jsonl(tree / FAILURE_NAME, receipt)
    partial.unlink()


def build_archive(tree: Path, entries: list[Entry], archive: Path) -> dict[str, object]:
    partial = archive.with_name(archive.name + ".partial")
    cleanup_partial(tree, partial, "stale incomplete archive from an interrupted prior stage")
    command = [
        "zstd",
        f"-{ZSTD_LEVEL}",
        f"--long={ZSTD_LONG}",
        f"-T{ZSTD_THREADS}",
        "--no-progress",
        "-f",
        "-o",
        str(partial),
        "-",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise CertifyError("zstd stdin pipe unavailable")
    floor_hit = threading.Event()

    def monitor() -> None:
        while process.poll() is None:
            if int(fs_bytes(AP_ROOT)["available_bytes"]) < AP_ABORT_FLOOR_BYTES:
                floor_hit.set()
                process.send_signal(signal.SIGTERM)
                return
            time.sleep(1.0)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for index, entry in enumerate(entries, start=1):
                info = _tar_info(entry)
                if entry.kind == "file":
                    with (tree / entry.path).open("rb") as handle:
                        tar.addfile(info, handle)
                else:
                    tar.addfile(info)
                if index % 100 == 0 or index == len(entries):
                    progress(f"packed {index}/{len(entries)} paths")
        process.stdin.close()
        process.stdin = None
        stderr = process.communicate()[1].decode(errors="replace")
    except (BrokenPipeError, OSError) as exc:
        process.kill()
        stderr = process.communicate()[1].decode(errors="replace")
        reason = (
            f"AP free-space abort floor {AP_ABORT_FLOOR_BYTES} B reached"
            if floor_hit.is_set()
            else f"archive stage failed: {exc}; zstd={stderr[-1000:]}"
        )
        cleanup_partial(tree, partial, reason)
        raise CertifyError(f"archive blocked; originals retained: {reason}") from exc
    watcher.join(timeout=2)
    if process.returncode != 0 or floor_hit.is_set():
        reason = (
            f"AP free-space abort floor {AP_ABORT_FLOOR_BYTES} B reached"
            if floor_hit.is_set()
            else f"zstd rc={process.returncode}: {stderr[-1000:]}"
        )
        cleanup_partial(tree, partial, reason)
        raise CertifyError(f"archive blocked; originals retained: {reason}")
    os.replace(partial, archive)
    return {
        "command": command,
        "zstd_version": zstd_version(),
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
    }


def _safe_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise CertifyError(f"unsafe archive member: {member.name}")
    if member.issym():
        target = PurePosixPath(member.linkname)
        if target.is_absolute() or ".." in target.parts:
            raise CertifyError(f"unsafe symlink member: {member.name} -> {member.linkname}")


def extract_archive(archive: Path, destination: Path) -> None:
    process = subprocess.Popen(
        ["zstd", f"--long={ZSTD_LONG}", "-d", "-c", str(archive)],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise CertifyError("zstd decode stdout pipe unavailable")
    with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
        for member in tar:
            _safe_member(member)
            tar.extract(member, path=destination, filter="data")
    process.stdout.close()
    rc = process.wait()
    if rc != 0:
        raise CertifyError(f"zstd decode failed rc={rc}")


def verify_archive(
    tree: Path, archive: Path, expected: list[Entry], tree_hash: str
) -> dict[str, object]:
    logical_bytes = sum(row.bytes for row in expected if row.kind == "file")
    local_free = int(fs_bytes(LOCAL_SCRATCH_ROOT)["available_bytes"])
    if local_free < logical_bytes + LOCAL_MARGIN_BYTES:
        raise CertifyError(
            f"local extraction headroom {local_free} < {logical_bytes + LOCAL_MARGIN_BYTES}"
        )
    with tempfile.TemporaryDirectory(prefix="ddm_sr3_verify_", dir=LOCAL_SCRATCH_ROOT) as tmp:
        destination = Path(tmp)
        extract_archive(archive, destination)
        extracted_meta = scan_tree(destination)
        assert_entry_metadata_equal(expected, extracted_meta, require_mtime=False)
        extracted = hash_entries(destination, extracted_meta)
        assert_hashes_equal(expected, extracted, require_metadata=False)
    archive_sha = sha256_file(archive)
    return {
        "schema": SCHEMA,
        "phase": "ROUND_TRIP_VERIFIED",
        "written_at_utc": utcnow(),
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "source_tree_hash": tree_hash,
        "verified_file_count": sum(row.kind == "file" for row in expected),
        "verified_directory_count": sum(row.kind == "dir" for row in expected),
        "verified_symlink_count": sum(row.kind == "symlink" for row in expected),
        "verified_logical_bytes": logical_bytes,
        "verification": "complete path set + per-file SHA-256 equality",
        "scratch_policy": "local TemporaryDirectory removed automatically after success or failure",
    }


def verify_receipt_matches(receipt: dict[str, object], archive: Path, tree_hash: str) -> bool:
    return (
        receipt.get("phase") == "ROUND_TRIP_VERIFIED"
        and receipt.get("source_tree_hash") == tree_hash
        and receipt.get("archive_bytes") == archive.stat().st_size
        and receipt.get("archive_sha256") == sha256_file(archive)
    )


def remove_original_top_level(
    tree: Path, entries: list[Entry], *, allow_already_absent: bool = False
) -> list[str]:
    top_names = sorted({PurePosixPath(row.path).parts[0] for row in entries})
    removed: list[str] = []
    for name in top_names:
        if name in RESERVED_NAMES or name in {"", ".", ".."}:
            raise CertifyError(f"refusing unsafe original top-level name: {name!r}")
        target = tree / name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        elif allow_already_absent and not target.exists():
            continue
        else:
            raise CertifyError(f"original top-level path vanished before reclaim: {target}")
        removed.append(str(target))
    return removed


def original_top_level_paths(tree: Path, entries: list[Entry]) -> list[str]:
    return [
        str(tree / name)
        for name in sorted({PurePosixPath(row.path).parts[0] for row in entries})
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--closure-citation", required=True)
    parser.add_argument("--closure-note", required=True)
    parser.add_argument("--repo", default="/Users/adpena/Projects/pact")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    try:
        tree = validate_tree(args.tree)
        repo = Path(args.repo).resolve()
        closure_path = Path(args.closure_citation)
        if not closure_path.is_absolute():
            closure_path = repo / closure_path
        closure_path = closure_path.resolve()
        if not closure_path.is_file():
            raise CertifyError(f"closure citation is not a file: {closure_path}")
        if len(args.closure_note.strip()) < 20:
            raise CertifyError("a substantive closure note is required")
        closure_record = {
            "path": str(closure_path),
            "bytes": closure_path.stat().st_size,
            "sha256": sha256_file(closure_path),
            "note": args.closure_note.strip(),
        }
        fleet_receipt = assert_fleet_idle(repo)
        cert_path = tree / CERT_NAME
        prior_certificate: dict[str, object] | None = None
        if cert_path.exists():
            prior = json.loads(cert_path.read_text())
            if prior.get("status") == "RECLAIMED_VERIFIED":
                emit_json(prior)
                return 0
            if prior.get("status") == "VERIFIED_READY_TO_RECLAIM":
                prior_certificate = prior
                if prior.get("closure_citation") != closure_record:
                    raise CertifyError("closure citation changed after pre-reclaim certification")

        df_before = (
            prior_certificate["df_before"] if prior_certificate else fs_bytes(AP_ROOT)
        )
        allocated_before = (
            int(prior_certificate["original_allocated_bytes"])
            if prior_certificate
            else allocated_bytes(tree)
        )
        manifest_path = tree / MANIFEST_NAME
        if manifest_path.exists():
            entries, tree_hash = read_manifest(manifest_path)
            observed = scan_tree(tree)
            if prior_certificate:
                archive = tree / ARCHIVE_NAME
                verify_path = tree / VERIFY_NAME
                if not archive.is_file() or not verify_path.is_file():
                    raise CertifyError("pre-reclaim certificate exists without archive/verification")
                prior_verify = json.loads(verify_path.read_text())
                if not verify_receipt_matches(prior_verify, archive, tree_hash):
                    raise CertifyError("pre-reclaim resume archive/verification pins differ")
                assert_source_subset_equal(tree, entries, observed)
                progress("resuming certified reclaim stage")
            else:
                assert_entry_metadata_equal(entries, observed)
                progress("resuming from durable source manifest")
        else:
            observed = scan_tree(tree)
            if not observed:
                raise CertifyError("source tree is empty")
            newest = max(tree.lstat().st_mtime_ns, *(row.mtime_ns for row in observed)) / 1e9
            age = time.time() - newest
            if age < RECENT_SECONDS:
                raise CertifyError(f"newest descendant is only {age / 3600:.2f} h old")
            entries = hash_entries(tree, observed)
            tree_hash = write_manifest(manifest_path, entries)

        logical_bytes = sum(row.bytes for row in entries if row.kind == "file")
        progress_record = {
            "schema": SCHEMA,
            "phase": "SOURCE_MANIFEST_DURABLE",
            "updated_at_utc": utcnow(),
            "tree": str(tree),
            "closure_citation": closure_record,
            "fleet_receipt": fleet_receipt,
            "source_allocated_bytes": allocated_before,
            "source_logical_bytes": logical_bytes,
            "source_tree_hash": tree_hash,
            "source_file_count": sum(row.kind == "file" for row in entries),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "df_before": df_before,
        }
        atomic_json(tree / PROGRESS_NAME, progress_record)
        if not args.apply:
            emit_json(progress_record)
            return 0

        archive = tree / ARCHIVE_NAME
        if archive.exists():
            archive_info = {
                "command": "resumed existing completed archive",
                "zstd_version": zstd_version(),
                "archive_path": str(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": sha256_file(archive),
            }
        else:
            archive_info = build_archive(tree, entries, archive)
        progress_record.update(
            {"phase": "ARCHIVE_DURABLE", "archive": archive_info, "updated_at_utc": utcnow()}
        )
        atomic_json(tree / PROGRESS_NAME, progress_record)

        verify_path = tree / VERIFY_NAME
        verified: dict[str, object]
        if verify_path.exists():
            prior_verify = json.loads(verify_path.read_text())
            if verify_receipt_matches(prior_verify, archive, tree_hash):
                verified = prior_verify
                progress("resuming from matching round-trip receipt")
            else:
                raise CertifyError("existing verification receipt does not match archive/tree")
        else:
            verified = verify_archive(tree, archive, entries, tree_hash)
            atomic_json(verify_path, verified)
        verify_sha = sha256_file(verify_path)

        archive_bytes = archive.stat().st_size
        archive_sha = sha256_file(archive)
        pre_certificate = {
            "schema": SCHEMA,
            "status": "VERIFIED_READY_TO_RECLAIM",
            "written_at_utc": utcnow(),
            "original_tree_path": str(tree),
            "original_allocated_bytes": allocated_before,
            "original_logical_bytes": logical_bytes,
            "original_file_count": sum(row.kind == "file" for row in entries),
            "original_directory_count": sum(row.kind == "dir" for row in entries),
            "original_symlink_count": sum(row.kind == "symlink" for row in entries),
            "source_tree_hash": tree_hash,
            "manifest": {
                "path": str(manifest_path),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            },
            "archive": {
                "path": str(archive),
                "bytes": archive_bytes,
                "sha256": archive_sha,
                "zstd_version": zstd_version(),
                "settings": {
                    "level": ZSTD_LEVEL,
                    "long_window_log": ZSTD_LONG,
                    "threads": ZSTD_THREADS,
                    "determinism_scope": "same manifest, zstd version, settings, and thread count",
                },
            },
            "verification_receipt": {
                "path": str(verify_path),
                "bytes": verify_path.stat().st_size,
                "sha256": verify_sha,
                "result": "complete per-file SHA-256 equality",
            },
            "closure_citation": closure_record,
            "fleet_receipt": fleet_receipt,
            "df_before": df_before,
            "reconstruction_command": (
                f"zstd --long={ZSTD_LONG} -d -c {shlex_quote(str(archive))} | "
                f"tar -xf - -C {shlex_quote(str(tree))}"
            ),
        }
        atomic_json(cert_path, pre_certificate)

        # The second complete source read is the final deletion gate.
        deletion_fleet_receipt = assert_fleet_idle(repo)
        current_meta = scan_tree(tree)
        if prior_certificate:
            assert_source_subset_equal(tree, entries, current_meta)
        else:
            assert_entry_metadata_equal(entries, current_meta)
            current_hashed = hash_entries(tree, current_meta)
            assert_hashes_equal(entries, current_hashed)
        removed = remove_original_top_level(
            tree, entries, allow_already_absent=prior_certificate is not None
        )
        remaining_originals = scan_tree(tree)
        if remaining_originals:
            raise CertifyError(
                f"original path set remains after exact-target removal: "
                f"{[row.path for row in remaining_originals[:10]]}"
            )

        df_after = fs_bytes(AP_ROOT)
        final = {
            **pre_certificate,
            "status": "RECLAIMED_VERIFIED",
            "completed_at_utc": utcnow(),
            "original_top_level_paths": original_top_level_paths(tree, entries),
            "removed_original_top_level_paths_this_attempt": removed,
            "deletion_fleet_receipt": deletion_fleet_receipt,
            "df_after": df_after,
            "measured_available_bytes_delta": int(df_after["available_bytes"])
            - int(df_before["available_bytes"]),
            "retained_tree_allocated_bytes": allocated_bytes(tree),
            "archive_ratio_logical_over_archive": logical_bytes / archive_bytes,
            "reconstruction_verified": True,
        }
        atomic_json(cert_path, final)
        progress_record.update({"phase": "RECLAIMED_VERIFIED", "updated_at_utc": utcnow()})
        atomic_json(tree / PROGRESS_NAME, progress_record)
        emit_json(final)
        return 0
    except (CertifyError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        progress(f"BLOCK: {exc}")
        return 2


def shlex_quote(value: str) -> str:
    """Quote one command argument without importing a shell or executing it."""
    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
