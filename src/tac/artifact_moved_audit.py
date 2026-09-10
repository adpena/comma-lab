"""Bounded custody checks over explicitly supplied MOVED certificates.

The default checks metadata and the small AppleDouble header only. Full hashing
is opt-in; callers must hold any required host-quiet authorization first. This
module discovers only caller-supplied roots; it never follows MOVED chains or
mutates artifacts.
"""
from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable
from pathlib import Path

from tac.artifact_moved import MovedArtifactError, verify_payload

_CUSTODY_STATUSES = {"MOVED", "RECOVERED", "MOVED_CERTIFIED"}
_NON_CUSTODY_STATUSES = {"MOVED_CERTIFICATE_FALSE", "REFUSED", "BLOCKED", "SKIPPED"}


def discover_certificates(roots: Iterable[str | Path], max_depth: int = 4) -> list[Path]:
    """Discover current per-file certificates within explicit roots, without hashing.

    The root has depth zero, so max_depth=1 includes only its immediate files.
    Include *.MOVED.json, MOVE_LOG.jsonl and bare MOVED.json. Bare certificates
    are excluded only for recognized vertigo_cold_move.v2 or
    ddm_sr2_vertigo_move_v1 directory records with absolute original_path and
    destination and no moved_to; those require the separate full tree audit.
    Unknown or malformed bare certificates remain included for refusal. Skip
    AppleDouble certificate sidecars and directory symlinks. Missing/unreadable
    roots, traversal errors and symlink/special certificate files raise instead
    of silently returning an incomplete census. No default volume is inferred.
    """
    if type(max_depth) is not int or max_depth < 1:
        raise ValueError("INVALID_DISCOVERY_MAX_DEPTH")
    roots = [Path(root) for root in roots]
    if not roots:
        raise ValueError("EMPTY_DISCOVERY_ROOTS")
    found = set()
    for root in roots:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise ValueError(f"DISCOVERY_ROOT_NOT_REAL_DIRECTORY:{root}")
        pending = [(root, 0)]
        while pending:
            directory, depth = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name.startswith("._"):
                        continue
                    path = Path(entry.path)
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if stat.S_ISDIR(mode):
                        if depth + 1 < max_depth:
                            pending.append((path, depth + 1))
                    elif entry.name in {"MOVE_LOG.jsonl", "MOVED.json"} or entry.name.endswith(".MOVED.json"):
                        if not stat.S_ISREG(mode):
                            raise ValueError(f"DISCOVERY_CERTIFICATE_NOT_REGULAR:{path}")
                        if entry.name == "MOVED.json":
                            try:
                                row = _object(path.read_text())
                            except ValueError:
                                row = {}
                            if ("moved_to" not in row
                                    and row.get("schema") in ("vertigo_cold_move.v2", "ddm_sr2_vertigo_move_v1")
                                    and all(isinstance(row.get(key), str)
                                            and Path(row[key]).is_absolute()
                                            for key in ("original_path", "destination"))):
                                continue
                        found.add(path)
    return sorted(found)


def _object(text: str) -> dict:
    """Reject duplicate keys as well as non-object JSON certificates."""
    def unique(pairs):
        row = {}
        for key, value in pairs:
            if key in row:
                raise ValueError(f"DUPLICATE_KEY:{key}")
            row[key] = value
        return row

    row = json.loads(text, object_pairs_hook=unique)
    if not isinstance(row, dict):
        raise ValueError("CERTIFICATE_NOT_OBJECT")
    return row


def _source(row: dict) -> str:
    value = row.get("path", row.get("source"))
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError("INVALID_SOURCE_PATH")
    return value


def _payload(row: dict, *, hash_payloads: bool) -> None:
    destination = row.get("moved_to", row.get("destination"))
    if not isinstance(destination, str) or not destination.strip():
        raise ValueError("MISSING_DESTINATION_OR_UNSUPPORTED_DIRECTORY_SCHEMA")
    if not Path(destination).is_absolute():
        raise ValueError(f"NON_ABSOLUTE_DESTINATION:{destination}")
    size, digest = row.get("bytes"), row.get("sha256")
    if type(size) is not int or size < 0:
        raise ValueError(f"INVALID_EXPECTED_BYTES:{destination}")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError(f"INVALID_EXPECTED_SHA256:{destination}")
    verify_payload(destination, size, digest, hash_payload=hash_payloads)


def _log_row(row: dict, *, hash_payloads: bool) -> None:
    status = row.get("status", row.get("certificate_status"))
    if status in _NON_CUSTODY_STATUSES:
        # Corrections/refusals are evidence, never a waiver of a prior MOVED row.
        _source(row)
        return
    if not isinstance(status, str) or status not in _CUSTODY_STATUSES:
        raise ValueError(f"UNSUPPORTED_LOG_STATUS:{status!r}")
    if "moved_to" not in row and "destination" not in row:
        source = _source(row)
        manifest = Path(source + ".MOVED.json")
        certificate = _object(manifest.read_text())
        # A legacy log's content identity must agree with its destination manifest.
        for key in ("bytes", "sha256"):
            if key in row and row[key] != certificate.get(key):
                raise ValueError(f"LOG_MANIFEST_IDENTITY_MISMATCH:{key}:{manifest}")
        _payload(certificate, hash_payloads=hash_payloads)
    else:
        _payload(row, hash_payloads=hash_payloads)


def certificate_violations(paths: Iterable[str | Path], *, hash_payloads: bool = False) -> list[str]:
    """Return path-qualified failures; an empty input means no coverage, not a census.

    Only file MOVED schemas and MOVE_LOG.jsonl are supported. Historical rows
    remain checked after correction/recovery events, so merely appending a status
    cannot turn a missing destination into valid custody. Restoring the original
    destination makes the historical record valid again. Directory certificates
    lacking a file destination fail explicitly rather than passing vacuously.
    """
    found = []
    for supplied in paths:
        path = Path(supplied)
        try:
            if path.name.startswith("._"):
                raise ValueError("APPLEDOUBLE_CERTIFICATE_NAME")
            if path.name == "MOVE_LOG.jsonl":
                with path.open() as stream:
                    rows = 0
                    for number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        rows += 1
                        try:
                            _log_row(_object(line), hash_payloads=hash_payloads)
                        except (OSError, ValueError, TypeError) as exc:
                            found.append(f"{path}:{number}:{exc}")
                    if not rows:
                        raise ValueError("EMPTY_MOVE_LOG")
            elif path.name == "MOVED.json" or path.name.endswith(".MOVED.json"):
                _payload(_object(path.read_text()), hash_payloads=hash_payloads)
            else:
                raise ValueError("UNSUPPORTED_CERTIFICATE_FILENAME")
        except (OSError, ValueError, TypeError, MovedArtifactError) as exc:
            found.append(f"{path}:{exc}")
    return found
