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


def validate_tree(raw: str, *, lift_protection: str | None = None) -> Path:
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
    if tree in PROTECTED_TREES and not lift_protection:
        raise CertifyError(f"explicitly protected live store: {tree}")
    if tree.name.startswith(CUSTODY_PREFIXES):
        raise CertifyError(f"custody namespace requires separate certificate adjudication: {tree}")
    return tree


def validate_keep_paths(tree: Path, raw_keeps: Iterable[str]) -> tuple[str, ...]:
    """Normalise --keep-uncompressed into safe, existing, tree-relative POSIX paths."""
    keeps: list[str] = []
    resolved_tree = tree.resolve()
    for raw in raw_keeps:
        if not raw.strip():
            raise CertifyError("--keep-uncompressed must not be empty")
        if Path(raw.strip()).is_absolute():
            raise CertifyError(f"--keep-uncompressed must be tree-relative: {raw!r}")
        text = raw.strip().strip("/")
        if not text:
            raise CertifyError("--keep-uncompressed must not be empty")
        parts = PurePosixPath(text).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise CertifyError(f"unsafe --keep-uncompressed path: {raw!r}")
        if parts[0] in RESERVED_NAMES:
            raise CertifyError(f"--keep-uncompressed may not name SR3 custody: {raw!r}")
        rel = PurePosixPath(*parts).as_posix()
        target = tree / rel
        if not target.exists() and not target.is_symlink():
            raise CertifyError(f"--keep-uncompressed path does not exist: {target}")
        parent = target.parent.resolve()
        if parent != resolved_tree and resolved_tree not in parent.parents:
            raise CertifyError(f"--keep-uncompressed escapes the tree: {raw!r}")
        if rel not in keeps:
            keeps.append(rel)
    ordered = tuple(sorted(keeps))
    for outer in ordered:
        for inner in ordered:
            if inner != outer and _is_kept(inner, (outer,)):
                raise CertifyError(
                    f"--keep-uncompressed {inner!r} is already covered by {outer!r}"
                )
    return ordered


_REFERENCE_ROOTS = ("tools", "experiments", "src", "scripts", "runtime-rs")
_REFERENCE_SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml"}
_REFERENCE_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-"
)


_REFERENCE_SCAN_METHOD = (
    "literal absolute-prefix scan of executable source. BOUNDS, stated so this "
    "receipt cannot be read as universal coverage: (1) it finds LITERAL path "
    "strings only -- a path composed at runtime from variables is invisible to "
    "it; (2) a path split across implicitly-concatenated string literals is "
    "RESOLVED -- the sibling literals are joined, so the adjudicated path is the "
    "one the program opens, not a prefix that could spuriously match a carve-out "
    "whose name it merely starts with; a continuation this scan cannot read (an "
    "f-string, an unterminated literal) is refused instead, never counted as "
    "covered, and so is a join that stops at a character a path cannot hold "
    "(`{}` slot, `#` fragment, space) WHEN the truncated prefix would otherwise "
    "read as carved out -- refusal is spent only where truncation could change "
    "the verdict; (3) it reads only the "
    "executable roots, because record surfaces (.omx/research memos, .omx/state "
    "ledgers, graph nodes) CITE paths without reading them, and the archive "
    "preserves those files anyway. Bound (1) is the dangerous one -- a "
    "runtime-composed path is not seen AT ALL, so it cannot even be refused -- "
    "which is why a lift is a judgement about THIS scan's denominators, never a "
    "proof of safety on its own."
)


def _literal_continues(text: str, end: int) -> bool:
    """True when the string literal holding this path continues in a sibling literal.

    Python's implicit concatenation (``"a/b/" "c.bin"``) is invisible to a raw-text
    scan: it reads only up to the first closing quote, so the extracted path is a
    PREFIX of the real one.  That matters because a prefix can spuriously match a
    carve-out whose NAME it merely starts with -- ``"…/retained/body" "guard/x"``
    extracts ``retained/body``, which is a carve-out, while the real file
    ``retained/bodyguard/x`` is not under one and would be archived and removed.
    So the prefix is not conservative in general.  Detecting the continuation is
    this function's whole job; ``_resolve_implicit_concatenation`` then READS it
    (a compile-time join is exactly reproducible), and only a continuation that
    cannot be read is refused.

    Quote -> whitespace -> quote is exactly implicit concatenation in Python; any
    other following character (``,`` ``)`` ``:`` ``if`` ...) is a different construct
    and reads as a complete literal.
    """
    if end >= len(text) or text[end] not in "\"'":
        return False
    if text[end : end + 3] == text[end] * 3:
        # A triple-quote DELIMITER closing a docstring that happened to end on a path,
        # not a sibling literal.  Without this the scan would over-refuse on any
        # docstring citing a tree path -- safe, but a lift blocked by a comment.
        return False
    probe = end + 1
    while probe < len(text) and text[probe] in " \t\r\n":
        probe += 1
    # A sibling literal may carry a prefix (f/r/b/u), and an f-string prefix is exactly
    # the case that must be SEEN so the caller can refuse it -- missing it here would
    # read the prefix as a complete literal and let a truncated path pass as covered.
    # Letters count only when a quote follows IMMEDIATELY: `and "x"` is not a literal,
    # `f"{x}"` is.  Python's prefixes are at most 2 characters; 3 is slack, not licence.
    letters = probe
    while probe < len(text) and probe - letters < 3 and text[probe].isalpha():
        probe += 1
    return probe < len(text) and text[probe] in "\"'"


@dataclass(frozen=True)
class ReferenceScan:
    """What the protection-lift scan looked at, and what it found.

    ``files_scanned`` and ``references_found`` are the DENOMINATORS.  Without them
    a scan that examined nothing reports a clean result indistinguishable from a
    genuinely clean one -- the vacuous-pass failure mode.
    """

    files_scanned: int
    references_found: int
    covered: tuple[dict[str, str], ...]
    violations: tuple[dict[str, str], ...]
    absent: tuple[dict[str, str], ...] = ()

    def receipt(self) -> dict[str, object]:
        return {
            "method": _REFERENCE_SCAN_METHOD,
            "reference_roots": list(_REFERENCE_ROOTS),
            "reference_suffixes": sorted(_REFERENCE_SUFFIXES),
            "files_scanned": self.files_scanned,
            "references_found": self.references_found,
            "references_covered_by_carve_out": len(self.covered),
            "references_uncovered": len(self.violations),
            "references_absent_from_tree": len(self.absent),
            "vacuous_scan_no_reference_found": self.references_found == 0,
        }


def _resolve_implicit_concatenation(text: str, end: int) -> tuple[str, int] | None:
    """Join the sibling literals Python would concatenate, so the REAL path is read.

    Refusing every truncated row (the first cure) is correct but far too coarse:
    ``Path("<root>/<dir>/" "<file>")`` is simply how a long path is written under a
    120-column limit, and a repo-wide scan measured 416 such sites.  A gate that
    refuses all of them refuses everything.  Reading them is easy and exact --
    implicit concatenation is a compile-time join, so appending the sibling
    literals reproduces the string the program actually opens, and `_is_kept` then
    adjudicates the true path instead of a prefix.

    Returns ``(joined_tail, new_end)``; ``("", end)`` when nothing continues.
    Returns None only for a continuation this scan genuinely cannot read -- an
    f-string (composed at runtime) or an unterminated literal -- and the caller
    refuses those rows rather than guessing.
    """
    tail: list[str] = []
    while _literal_continues(text, end):
        probe = end + 1
        while probe < len(text) and text[probe] in " \t\r\n":
            probe += 1
        prefix_start = probe
        while probe < len(text) and text[probe].isalpha():
            probe += 1
        if "f" in text[prefix_start:probe].lower():
            return None  # runtime-composed; not readable as a constant
        if probe >= len(text) or text[probe] not in "\"'":
            return None
        quote = text[probe]
        probe += 1
        chunk: list[str] = []
        while probe < len(text) and text[probe] != quote:
            if text[probe] == "\\":
                probe += 1
                if probe >= len(text):
                    return None
            chunk.append(text[probe])
            probe += 1
        if probe >= len(text):
            return None  # unterminated literal
        tail.append("".join(chunk))
        end = probe
    return "".join(tail), end


def scan_live_reference_detail(
    repo: Path, tree: Path, keep_uncompressed: tuple[str, ...]
) -> ReferenceScan:
    """Scan executable source for references INTO the tree, covered and not.

    This is the machine-checked half of a protection lift.  A protected live store
    may be compressed only AROUND the exact sub-paths live code still reads, so the
    reference set is derived from the source instead of asserted by hand -- the same
    check that, run by hand, is what distinguishes a safe reclaim from a silent
    breakage of an open task.  Its bounds are stated in ``_REFERENCE_SCAN_METHOD``
    and carried into the certificate; they are real, and a reader must see them.
    """
    prefix = f"{AP_ROOT}/{tree.name}/"
    covered: list[dict[str, str]] = []
    violations: list[dict[str, str]] = []
    absent: list[dict[str, str]] = []
    files_scanned = 0
    for root_name in _REFERENCE_ROOTS:
        root = repo / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in _REFERENCE_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                raise CertifyError(
                    f"unreadable source during reference scan: {path}: {exc}"
                ) from exc
            files_scanned += 1
            cursor = 0
            while True:
                hit = text.find(prefix, cursor)
                if hit < 0:
                    break
                start = hit + len(prefix)
                end = start
                while end < len(text) and text[end] in _REFERENCE_PATH_CHARS:
                    end += 1
                cursor = max(end, start + 1)
                raw = text[start:end]
                joined = _resolve_implicit_concatenation(text, end)
                if joined is None:
                    # A continuation this scan cannot read (f-string / unterminated),
                    # so `raw` is a PREFIX of the real path -- and a prefix can
                    # spuriously match a carve-out whose name it merely starts with.
                    # Refuse rather than guess; never count it as covered.
                    rel = raw.strip("/")
                    if rel:
                        violations.append(
                            {
                                "source": str(path.relative_to(repo)),
                                "referenced_path": rel,
                                "unresolvable_implicit_concatenation": "true",
                            }
                        )
                    continue
                extra, cont_end = joined
                truncated = False
                for ch in extra:
                    if ch not in _REFERENCE_PATH_CHARS:
                        truncated = True
                        break
                    raw += ch
                cursor = max(cursor, cont_end)
                rel = raw.strip("/")
                if not rel:
                    continue
                row = {"source": str(path.relative_to(repo)), "referenced_path": rel}
                # A reference the scan read EXACTLY (the literal closed on its own
                # quote and every joined character was path-legal) and that names a
                # path the tree does not hold is not a live read: the bytes are
                # already gone, so archiving cannot break a read that is already
                # broken.  Record it -- never silently drop it -- and let the reclaim
                # proceed.  Exactness is the load-bearing half: a row that stopped on
                # an f-string `{` slot is a PREFIX, and an absent prefix says nothing
                # about the real path, so those stay violations.  Measured 2026-08-31
                # over the six terminal-lane trees: 1 absent row of 61 references
                # (ddm_fs3 -> FS3_REOPEN_PRICE.json), and that one row made its tree
                # unarchivable in BOTH directions -- the scan refused the reference
                # while validate_keep_paths refused a carve-out for a path that does
                # not exist.
                exact_read = not truncated and end < len(text) and text[end] in "\"'"
                if (
                    not _is_kept(rel, keep_uncompressed)
                    and exact_read
                    and not (tree / rel).exists()
                    and not (tree / rel).is_symlink()
                ):
                    row["absent_from_tree_cannot_be_a_live_read"] = "true"
                    absent.append(row)
                elif not _is_kept(rel, keep_uncompressed):
                    violations.append(row)
                elif truncated:
                    # The join stopped at a character a path cannot hold (a `{}`
                    # format slot, a `#` fragment, a space before prose), so `rel`
                    # is a PREFIX -- and a prefix matching a carve-out proves
                    # nothing about the real path, which may merely START with the
                    # carve-out's name.  Refuse only here, where the truncation
                    # could change the verdict: an uncovered prefix is already a
                    # violation, and a fully-read path is exact.  Measured over
                    # this corpus, 8 of 412 split sites truncate and none of them
                    # is covered, so this costs nothing today and closes the hole.
                    row["truncated_continuation_cannot_confirm_carve_out"] = "true"
                    violations.append(row)
                else:
                    covered.append(row)
    return ReferenceScan(
        files_scanned=files_scanned,
        references_found=len(covered) + len(violations) + len(absent),
        covered=tuple(covered),
        violations=tuple(violations),
        absent=tuple(absent),
    )


def scan_live_references(
    repo: Path, tree: Path, keep_uncompressed: tuple[str, ...]
) -> list[dict[str, str]]:
    """The uncovered references only -- the ones that block a protection lift."""
    return list(scan_live_reference_detail(repo, tree, keep_uncompressed).violations)


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


def _keep_ancestor_dirs(keep_uncompressed: tuple[str, ...]) -> frozenset[str]:
    """Directories that MUST survive removal because they hold a carve-out."""
    ancestors: set[str] = set()
    for keep in keep_uncompressed:
        parts = PurePosixPath(keep).parts
        for depth in range(1, len(parts)):
            ancestors.add(PurePosixPath(*parts[:depth]).as_posix())
    return frozenset(ancestors)


def _is_kept(rel: str, keep_uncompressed: tuple[str, ...]) -> bool:
    """True when a tree-relative path is a declared carve-out or lives under one."""
    return any(rel == keep or rel.startswith(f"{keep}/") for keep in keep_uncompressed)


def scan_tree(tree: Path, keep_uncompressed: tuple[str, ...] = ()) -> list[Entry]:
    """Return every original descendant, excluding SR3 custody names and carve-outs.

    A carve-out is never manifested, never archived, and never removed.  The tree
    keeps it uncompressed and resolvable so live readers of that exact path keep
    working while the surrounding bulk is reclaimed.  Callers scanning an EXTRACTED
    tree must pass no carve-outs: the extraction contains exactly the archived set.
    """
    entries: list[Entry] = []
    for dirpath, dirnames, filenames in os.walk(tree, topdown=True, followlinks=False):
        at_root = Path(dirpath) == tree
        if at_root:
            dirnames[:] = sorted(name for name in dirnames if name not in RESERVED_NAMES)
        else:
            dirnames[:] = sorted(dirnames)
        if keep_uncompressed:
            dirnames[:] = [
                name
                for name in dirnames
                if not _is_kept(
                    (Path(dirpath) / name).relative_to(tree).as_posix(), keep_uncompressed
                )
            ]
        for name in sorted(dirnames + filenames):
            if at_root and name in RESERVED_NAMES:
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(tree).as_posix()
            if _is_kept(rel, keep_uncompressed):
                continue
            st = path.lstat()
            kind = _path_kind(path, st)
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


def _remove_archived_entries_selectively(
    tree: Path, entries: list[Entry], *, allow_already_absent: bool
) -> list[str]:
    """Remove exactly the manifested entries, deepest first, keeping carve-outs.

    Used when carve-outs exist: a top-level ``rmtree`` would take the kept paths
    with it, so removal walks the manifest bottom-up and leaves any directory that
    still holds a carve-out (or its ancestors) standing.
    """
    removed: list[str] = []
    for row in sorted(entries, key=lambda e: len(PurePosixPath(e.path).parts), reverse=True):
        first = PurePosixPath(row.path).parts[0]
        if first in RESERVED_NAMES or first in {"", ".", ".."}:
            raise CertifyError(f"refusing unsafe original top-level name: {first!r}")
        target = tree / row.path
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            if any(target.iterdir()):
                continue  # a carve-out (or its ancestor) still lives here
            target.rmdir()
        elif allow_already_absent and not target.exists():
            continue
        else:
            raise CertifyError(f"original path vanished before reclaim: {target}")
        removed.append(str(target))
    return removed


def remove_original_top_level(
    tree: Path,
    entries: list[Entry],
    *,
    allow_already_absent: bool = False,
    keep_uncompressed: tuple[str, ...] = (),
) -> list[str]:
    if keep_uncompressed:
        return _remove_archived_entries_selectively(
            tree, entries, allow_already_absent=allow_already_absent
        )
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


def build_keep_records(tree: Path, keep_uncompressed: tuple[str, ...]) -> list[dict[str, object]]:
    """Per-carve-out custody: what each kept path held BEFORE the reclaim ran."""
    records: list[dict[str, object]] = []
    for rel in keep_uncompressed:
        target = tree / rel
        if target.is_symlink():
            records.append({"path": rel, "kind": "symlink", "link_target": os.readlink(target)})
        elif target.is_file():
            records.append(
                {
                    "path": rel,
                    "kind": "file",
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
        else:
            files = sorted(q for q in target.rglob("*") if q.is_file())
            records.append(
                {
                    "path": rel,
                    "kind": "dir",
                    "file_count": len(files),
                    "bytes": sum(q.stat().st_size for q in files),
                    "files": [
                        {
                            "path": q.relative_to(tree).as_posix(),
                            "bytes": q.stat().st_size,
                            "sha256": sha256_file(q),
                        }
                        for q in files
                    ],
                }
            )
    return records


def verify_keep_records(tree: Path, keep_records: list[dict[str, object]]) -> dict[str, object]:
    """Re-verify every carve-out AFTER removal against its pre-removal custody.

    The carve-out exists so live readers keep working; recording its sha256 and
    never checking it again would let a future removal defect damage the exact
    paths the carve-out protects while the certificate still said VERIFIED.
    A live store may legitimately GAIN files here, so extra files are not drift --
    only a vanished or byte-changed carve-out is.
    """
    vanished: list[str] = []
    changed: list[str] = []
    checked = 0
    for record in keep_records:
        rel = str(record["path"])
        target = tree / rel
        if record["kind"] == "symlink":
            if not target.is_symlink() or os.readlink(target) != record["link_target"]:
                (vanished if not target.is_symlink() else changed).append(rel)
            checked += 1
        elif record["kind"] == "file":
            if not target.is_file():
                vanished.append(rel)
            elif sha256_file(target) != record["sha256"]:
                changed.append(rel)
            checked += 1
        else:
            for row in record["files"]:  # type: ignore[union-attr]
                sub = tree / str(row["path"])
                if not sub.is_file():
                    vanished.append(str(row["path"]))
                elif sha256_file(sub) != row["sha256"]:
                    changed.append(str(row["path"]))
                checked += 1
    return {
        "paths_checked": checked,
        "vanished": vanished,
        "changed": changed,
        "clean": not vanished and not changed,
    }


def unreclaimed_originals(tree: Path, keep_uncompressed: tuple[str, ...]) -> list[Entry]:
    """Rows that must be GONE after removal -- the reclaim gate's residue.

    A carve-out's ancestor directories legitimately survive: they are what holds
    the kept path resolvable.  Everything else the manifest named must be gone,
    or the reclaim did not do what the certificate is about to claim.  This is the
    single definition of that predicate -- ``main`` and its controls both call it,
    so the gate cannot drift away from the test that pins it.
    """
    expected_survivors = _keep_ancestor_dirs(keep_uncompressed)
    return [
        row
        for row in scan_tree(tree, keep_uncompressed)
        if not (row.kind == "dir" and row.path in expected_survivors)
    ]


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
    parser.add_argument(
        "--keep-uncompressed",
        action="append",
        default=[],
        metavar="REL_PATH",
        help=(
            "tree-relative path to carve OUT of the archive: never manifested, "
            "never archived, never removed, so live readers keep resolving it. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--lift-protection",
        default=None,
        metavar="RATIONALE",
        help=(
            "operate on an explicitly protected live store. Requires a non-empty "
            "--keep-uncompressed set AND a clean live-reference scan; the lift is "
            "recorded in the certificate."
        ),
    )
    args = parser.parse_args()

    try:
        tree = validate_tree(args.tree, lift_protection=args.lift_protection)
        repo = Path(args.repo).resolve()
        keep_uncompressed = validate_keep_paths(tree, args.keep_uncompressed)
        keep_records = build_keep_records(tree, keep_uncompressed)
        protection_lift: dict[str, object] | None = None
        if args.lift_protection is not None:
            if len(args.lift_protection.strip()) < 20:
                raise CertifyError("a substantive --lift-protection rationale is required")
            if not keep_uncompressed:
                raise CertifyError(
                    "--lift-protection requires at least one --keep-uncompressed carve-out"
                )
        # The scan runs on EVERY tree, protected or not.  Protection marks a store we
        # already KNOW is live; it is not what makes a tree readable, and the tree
        # nobody declared is exactly where an unchecked "nothing reads this" hides.
        # Measured 2026-08-31 over the six un-archived terminal-lane trees, none of
        # them protected: 61 live references, every tree non-zero.  Gating the scan on
        # a declaration would have archived and removed all six unchecked.
        scan = scan_live_reference_detail(repo, tree, keep_uncompressed)
        if scan.violations:
            shown = [
                f"{row['source']} -> {row['referenced_path']}" for row in scan.violations[:10]
            ]
            raise CertifyError(
                f"{len(scan.violations)} live reference(s) into {tree.name} are not "
                f"carved out (add --keep-uncompressed for each path live code reads, "
                f"or leave the tree alone): {shown}"
            )
        live_reference_scan = {
            "reference_scan": scan.receipt(),
            "references_covered": [dict(row) for row in scan.covered],
            "references_absent_from_tree": [dict(row) for row in scan.absent],
        }
        if args.lift_protection is not None:
            protection_lift = {
                "rationale": args.lift_protection.strip(),
                **live_reference_scan,
            }
            progress(f"PROTECTION LIFTED for {tree} around {list(keep_uncompressed)}")
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
            observed = scan_tree(tree, keep_uncompressed)
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
            observed = scan_tree(tree, keep_uncompressed)
            if not observed:
                raise CertifyError(
                    "nothing to archive: source tree is empty"
                    + (" once the carve-outs are excluded" if keep_uncompressed else "")
                )
            # Quiescence is checked over the ARCHIVED set only.  A carve-out is never
            # read into the tar and never removed, so a live writer inside it does not
            # make this archive inconsistent -- scope declared, not assumed.
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
            "keep_uncompressed": keep_records,
            "live_reference_scan": live_reference_scan,
            "protection_lift": protection_lift,
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
        current_meta = scan_tree(tree, keep_uncompressed)
        if prior_certificate:
            assert_source_subset_equal(tree, entries, current_meta)
        else:
            assert_entry_metadata_equal(entries, current_meta)
            current_hashed = hash_entries(tree, current_meta)
            assert_hashes_equal(entries, current_hashed)
        removed = remove_original_top_level(
            tree,
            entries,
            allow_already_absent=prior_certificate is not None,
            keep_uncompressed=keep_uncompressed,
        )
        remaining_originals = unreclaimed_originals(tree, keep_uncompressed)
        if remaining_originals:
            raise CertifyError(
                f"original path set remains after exact-target removal: "
                f"{[row.path for row in remaining_originals[:10]]}"
            )

        keep_verification = verify_keep_records(tree, keep_records)
        df_after = fs_bytes(AP_ROOT)
        final = {
            **pre_certificate,
            "status": "RECLAIMED_VERIFIED" if keep_verification["clean"] else "CARVE_OUT_DRIFT",
            "keep_uncompressed_verification": keep_verification,
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
        progress_record.update({"phase": str(final["status"]), "updated_at_utc": utcnow()})
        atomic_json(tree / PROGRESS_NAME, progress_record)
        emit_json(final)
        if not keep_verification["clean"]:
            # The receipt is written FIRST: removal already happened, so losing the
            # certificate would cost more than the drift itself.  Then fail closed.
            raise CertifyError(
                f"carve-out drift after removal: vanished={keep_verification['vanished'][:5]} "
                f"changed={keep_verification['changed'][:5]}"
            )
        return 0
    except (CertifyError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        progress(f"BLOCK: {exc}")
        return 2


def shlex_quote(value: str) -> str:
    """Quote one command argument without importing a shell or executing it."""
    return shlex.quote(value)


if __name__ == "__main__":
    raise SystemExit(main())
