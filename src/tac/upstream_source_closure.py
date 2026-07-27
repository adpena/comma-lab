# SPDX-License-Identifier: MIT
"""Portable identity for the frozen upstream evaluator source closure.

The full ``upstream/`` directory is a host workspace, not a portable
dependency closure: it can contain an environment-local ``.venv`` symlink,
models, videos, caches, or other files that are separately held in custody.
Candidate producers need the four source files recursively imported by the
authoritative evaluator.  Their identity must survive relocation to a clean
main checkout, so the closure digest intentionally excludes absolute paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Final

UPSTREAM_SOURCE_CLOSURE_SCHEMA: Final = "tac.upstream_source_closure.v1"
UPSTREAM_SOURCE_CLOSURE_MEMBERS: Final = (
    "evaluate.py",
    "frame_utils.py",
    "modules.py",
    "public_test_video_names.txt",
)


class UpstreamSourceClosureError(ValueError):
    """The frozen evaluator source closure is absent, aliased, or changed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_member(relative: str) -> PurePosixPath:
    if type(relative) is not str:
        raise UpstreamSourceClosureError("upstream source member must be an exact string")
    member = PurePosixPath(relative)
    if (
        not relative
        or member.is_absolute()
        or member.as_posix() != relative
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise UpstreamSourceClosureError(f"upstream source member is noncanonical: {relative!r}")
    return member


def compute_upstream_source_closure_identity(
    repo_root: Path | str | None = None,
    *,
    upstream_subdir: str = "upstream",
) -> dict[str, Any]:
    """Return a relocation-invariant identity for evaluator source members.

    ``root`` is diagnostic metadata only.  ``closure_sha256`` covers the
    schema and the ordered ``relative_path``/``bytes``/``sha256`` member rows;
    it never covers an absolute path.  Exact member files and every path
    component below ``upstream/`` must be regular and non-symlinked.  Unrelated
    workspace entries are deliberately outside this source-only closure.
    """

    base = Path(__file__).resolve().parent.parent.parent if repo_root is None else Path(repo_root)
    upstream_root = base / upstream_subdir
    if upstream_root.is_symlink() or not upstream_root.is_dir():
        raise UpstreamSourceClosureError(f"upstream source root must be a real directory: {upstream_root}")
    resolved_root = upstream_root.resolve()
    rows: list[dict[str, Any]] = []
    for relative in UPSTREAM_SOURCE_CLOSURE_MEMBERS:
        member = _canonical_member(relative)
        unresolved = upstream_root.joinpath(*member.parts)
        cursor = upstream_root
        for part in member.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise UpstreamSourceClosureError(f"upstream source member cannot traverse a symlink: {relative}")
        if not unresolved.is_file():
            raise UpstreamSourceClosureError(f"required upstream source member is absent: {relative}")
        resolved = unresolved.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise UpstreamSourceClosureError(f"upstream source member escapes its root: {relative}")
        rows.append(
            {
                "relative_path": relative,
                "bytes": int(resolved.stat().st_size),
                "sha256": _sha256_file(resolved),
            }
        )
    closure_payload = {
        "schema": UPSTREAM_SOURCE_CLOSURE_SCHEMA,
        "members": rows,
    }
    return {
        "schema": UPSTREAM_SOURCE_CLOSURE_SCHEMA,
        "root": str(resolved_root),
        "members": rows,
        "closure_sha256": hashlib.sha256(
            b"TAC-UPSTREAM-SOURCE-CLOSURE-V1\0" + _canonical_json(closure_payload)
        ).hexdigest(),
    }


__all__ = [
    "UPSTREAM_SOURCE_CLOSURE_MEMBERS",
    "UPSTREAM_SOURCE_CLOSURE_SCHEMA",
    "UpstreamSourceClosureError",
    "compute_upstream_source_closure_identity",
]
