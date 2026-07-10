"""Canonical fcntl-locked JSONL append helper — the ONE shared implementation.

Per the hardcode/duplication audit (``.omx/research/hardcode_duplication_audit_witness_stack_20260710.md``
finding #4): ``_append_locked_jsonl`` was reimplemented BYTE-IDENTICALLY (same name, same docstring,
same body) in ``src/tac/witness_dsl/activation_ledger.py`` and
``src/tac/witness_dsl/curriculum_candidate_pool.py`` — a copy-paste, not an import. Per CLAUDE.md
"Results must become system intelligence" + the fcntl-locked JSONL store discipline (Catalog #128/#131
atomic-write pattern), this module is the single canonical home so a future correctness fix (e.g.
``BlockingIOError`` retry, non-POSIX fallback hardening) lands in ONE place instead of being
hand-propagated across N copies.

Both former call sites are migrated to import :func:`append_locked_jsonl` from here; their local
``_append_locked_jsonl`` definitions are deleted. Behavior is preserved EXACTLY: fcntl ``LOCK_EX``
around the write, ``flush()`` + ``os.fsync()`` while holding the lock, ``LOCK_UN`` in a ``finally``,
and a best-effort plain-append fallback on platforms without ``fcntl`` (Windows).

Per CLAUDE.md "'Off' is a tracked queue" + the `.omx/state/*.jsonl` canonical-store pattern this
helper backs: callers are expected to serialize one JSON object per line, APPEND-ONLY, with
latest-row-wins semantics resolved by the READER (this helper only writes).

NOTE: there is a repo-wide sister pattern (~85-95 files touch ``fcntl.LOCK_EX`` directly, per the
same audit) that is explicitly OUT OF SCOPE for this landing — see the audit memo's op-routable #4
for the deferred larger sweep. This module canonicalizes only the two byte-identical
``_append_locked_jsonl`` copies named in that finding.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def append_locked_jsonl(p: Path, row: dict, *, sort_keys: bool = True) -> None:
    """fcntl-locked APPEND of ONE json row (canonical .omx/state pattern; best-effort off-POSIX).

    Creates the parent directory if missing, serializes ``row`` as one JSON line (``sort_keys``
    controls key ordering; both former call sites used ``sort_keys=True``, now the default), and
    appends it under an exclusive fcntl lock so concurrent writers never interleave partial lines.
    On platforms without ``fcntl`` (e.g. Windows), falls back to a plain unlocked append.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=sort_keys) + "\n"
    try:
        import fcntl
        with open(p, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - non-POSIX fallback
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)


__all__ = ["append_locked_jsonl"]
