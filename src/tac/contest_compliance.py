# SPDX-License-Identifier: MIT
"""Canonical upstream-snapshot drift-detection helper.

Per the 12-month strategic-foresight premortem
(`.omx/research/12_month_frustration_premortem_and_recommendations_20260516.md`
item #2): every empirical anchor (continual-learning posterior, Modal call_id
ledger, cost-band posterior) MUST carry an ``upstream_snapshot_sha256`` field
so we can structurally detect when an anchor was scored against a different
upstream/ snapshot than the one currently checked out. Without the field, a
silent upstream rotation (e.g. Yousfi swaps SegNet weights) invalidates every
prior score claim with no operator-visible signal.

This module exports ONE canonical entry point — :func:`compute_upstream_snapshot_sha256` —
that walks ``upstream/`` deterministically and returns a stable hex SHA-256 of
the (path, content) ordered tuple. Python bytecode is executable input, not
cache noise: it is included in ordinary digests and authority producers can
request fail-closed rejection so host-local bytecode can never silently alter
the frozen evaluator.

Per CLAUDE.md "Apples-to-apples evidence discipline": anchor consumers can
read the stamped hash from the JSONL row and refuse to compare anchors whose
upstream snapshots differ (or downgrade the cross-snapshot comparison to
advisory). The helper deliberately returns ``None`` when ``upstream/`` is
missing so call sites in ephemeral CI / test environments without the upstream
clone do not break — but a missing hash MUST be surfaced upstream as
``None`` rather than silently falling through.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

# Paths under ``upstream/`` to skip when hashing. Python ``__pycache__`` is
# deliberately absent: cached bytecode can be imported and therefore belongs
# to the executable dependency surface.
_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {".git", ".pytest_cache", ".mypy_cache", "node_modules"}
)

# Bytecode can be executed directly even when no matching source exists. Exact
# authority runs reject it instead of trusting environment-specific cache
# validation; non-authority callers bind it into the digest.
_EXECUTABLE_BYTECODE_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo"})

# macOS writes an AppleDouble sidecar ``._<name>`` beside each file when a tree
# is copied onto a filesystem without native resource-fork support (ExFAT, the
# external SSD tiers). ``._mod.pyc`` therefore has suffix ``.pyc`` while holding
# no bytecode at all. Refusal is unchanged; only the diagnosis is.
_APPLEDOUBLE_PREFIX = "._"


def _iter_upstream_files(
    upstream_dir: Path,
    *,
    reject_executable_artifacts: bool,
) -> Iterable[Path]:
    """Yield every regular file under ``upstream_dir`` in sorted-path order.

    Excludes ``_SKIP_DIR_NAMES`` per the canonical upstream-snapshot contract.
    Symlinks are rejected before any exclusion rule rather than followed or
    skipped: otherwise evaluator code or weights could be consumed through an
    unbound path while the purported full-tree digest remained unchanged.
    Executable bytecode is either hashed or rejected, never omitted.
    """
    if not upstream_dir.exists():
        return
    # Pre-collect to enable stable sorted order. rglob does not guarantee
    # ordering across filesystems.
    candidates: list[Path] = []
    for path in upstream_dir.rglob("*"):
        rel_parts = path.relative_to(upstream_dir).parts
        if path.is_symlink():
            raise ValueError(
                "canonical upstream snapshot cannot contain symlinks: "
                f"{path.relative_to(upstream_dir).as_posix()}"
            )
        if any(part in _SKIP_DIR_NAMES for part in rel_parts):
            continue
        if not path.is_file():
            continue
        if reject_executable_artifacts and path.suffix in _EXECUTABLE_BYTECODE_SUFFIXES:
            rel = path.relative_to(upstream_dir).as_posix()
            # An AppleDouble sidecar (``._name``) carries the .pyc SUFFIX but is
            # macOS resource-fork metadata, not bytecode. Refusing is still
            # correct -- an authority tree must not carry filesystem litter --
            # but the generic bytecode message sends the reader hunting for a
            # compiled module that does not exist. Name the object and the cure.
            if path.name.startswith(_APPLEDOUBLE_PREFIX):
                raise ValueError(
                    "canonical authority snapshot cannot contain an AppleDouble "
                    f"sidecar: {rel} -- this is macOS resource-fork metadata "
                    "(not executable bytecode), written when a tree is copied "
                    "onto a non-HFS filesystem such as ExFAT. Remove it with "
                    f"`find {upstream_dir} -name '._*' -delete` and re-run; if "
                    "a real __pycache__ directory is also present, purge that "
                    "too and export PYTHONDONTWRITEBYTECODE=1 for the producer."
                )
            raise ValueError(
                f"canonical authority snapshot cannot contain executable bytecode: {rel}"
            )
        candidates.append(path)
    candidates.sort(key=lambda p: p.relative_to(upstream_dir).as_posix())
    yield from candidates


def compute_upstream_snapshot_sha256(
    repo_root: Path | str | None = None,
    *,
    upstream_subdir: str = "upstream",
    reject_executable_artifacts: bool = False,
) -> str | None:
    """Compute a deterministic SHA-256 of the ``upstream/`` directory tree.

    Returns the hex digest, or ``None`` when ``upstream/`` does not exist at
    ``repo_root / upstream_subdir``. The digest is invariant under filesystem
    iteration order (paths are sorted). Executable ``*.pyc`` / ``*.pyo``
    artifacts are included by default. Set ``reject_executable_artifacts`` for
    an authority producer or consumer that requires a source-only frozen tree;
    this fails closed rather than allowing host-local bytecode to affect the
    evaluator outside the digest contract.

    Each contributing file appears in the hash as the byte sequence
    ``"<posix-relative-path>\\n<size>\\n<sha256-of-content>\\n"`` so a rename
    that preserves bytes still mutates the snapshot hash (canonical filename
    is part of the contract per ``upstream/`` being the contest's
    source-of-truth for evaluate.py / scorer modules / video bytes).
    """
    # Default to the repository root (this file lives at
    # ``src/tac/contest_compliance.py`` so the repo root is 3 parents up).
    # Test fixtures pass an explicit repo_root.
    repo_root = Path(__file__).resolve().parent.parent.parent if repo_root is None else Path(repo_root)
    upstream_dir = repo_root / upstream_subdir
    if not upstream_dir.exists():
        return None

    outer = hashlib.sha256()
    found_any = False
    for path in _iter_upstream_files(
        upstream_dir,
        reject_executable_artifacts=reject_executable_artifacts,
    ):
        found_any = True
        rel = path.relative_to(upstream_dir).as_posix()
        size = path.stat().st_size
        inner = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                inner.update(chunk)
        outer.update(rel.encode("utf-8"))
        outer.update(b"\n")
        outer.update(str(size).encode("ascii"))
        outer.update(b"\n")
        outer.update(inner.hexdigest().encode("ascii"))
        outer.update(b"\n")
    if not found_any:
        # The directory exists but contains no regular files — distinguish
        # this from "no upstream/" by returning a stable sentinel rather
        # than None so consumers can detect the degenerate case.
        return hashlib.sha256(b"<empty-upstream-tree>").hexdigest()
    return outer.hexdigest()


__all__ = ["compute_upstream_snapshot_sha256"]
