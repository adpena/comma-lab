# SPDX-License-Identifier: MIT
"""Small shared helpers for contest ``inflate.py`` runtimes.

These helpers are reviewability infrastructure, not score evidence. They keep
the repeated contest-loop and state-dict custody code out of individual
submission runtimes while preserving the exact three-argument inflate contract:
``archive_dir output_dir file_list``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tac.substrates._shared.inflate_runtime import raw_output_path

RenderOneVideoFn = Callable[[Path, Path, str], Any]


@dataclass(frozen=True)
class InflatedVideoRecord:
    """One rendered video produced by ``inflate_loop_per_video``."""

    video_name: str
    raw_output_path: Path
    render_result: Any = None


def iter_file_list_entries(file_list: Path | str) -> list[str]:
    """Read contest file-list entries, stripping blanks but preserving order."""

    path = Path(file_list)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path | str) -> str:
    """Return SHA-256 for a runtime payload file."""

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha256(path: Path | str, expected_sha256: str | None) -> str:
    """Verify a file hash when an expected SHA is supplied.

    Returns the actual SHA so callers can include it in runtime custody records.
    """

    actual = sha256_file(path)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"sha256 mismatch for {Path(path)}: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_per_substrate_state_dict(
    archive_dir: Path | str,
    state_relpath: str | Path,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
) -> Any:
    """Load a runtime state dict from inside the extracted archive directory.

    The helper refuses absolute paths and ``..`` traversal so a submission
    runtime cannot quietly depend on sibling working-tree state.
    """

    rel = Path(state_relpath)
    if rel.is_absolute() or any(part in {"", ".."} for part in rel.parts):
        raise ValueError(f"state_relpath must stay inside archive_dir: {state_relpath!r}")
    path = Path(archive_dir) / rel
    require_sha256(path, expected_sha256)
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        return torch.load(path, map_location=map_location)


def inflate_loop_per_video(
    *,
    file_list: Path | str,
    archive_dir: Path | str,
    output_dir: Path | str,
    render_fn: RenderOneVideoFn,
) -> list[InflatedVideoRecord]:
    """Run a per-video render function for every safe entry in ``file_list``.

    ``render_fn`` receives ``(archive_dir, raw_output_path, video_name)`` and is
    responsible for writing bytes to the supplied raw path. The helper creates
    parent directories and delegates output-path safety to ``raw_output_path``.
    """

    archive_root = Path(archive_dir)
    out_root = Path(output_dir)
    records: list[InflatedVideoRecord] = []
    for video_name in iter_file_list_entries(file_list):
        raw_path = raw_output_path(out_root, video_name)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        result = render_fn(archive_root, raw_path, video_name)
        records.append(
            InflatedVideoRecord(
                video_name=video_name,
                raw_output_path=raw_path,
                render_result=result,
            )
        )
    return records


# Compatibility aliases matching the original OP-1 directive names.
_inflate_loop_per_video = inflate_loop_per_video
_load_per_substrate_state_dict = load_per_substrate_state_dict


__all__ = [
    "InflatedVideoRecord",
    "RenderOneVideoFn",
    "_inflate_loop_per_video",
    "_load_per_substrate_state_dict",
    "inflate_loop_per_video",
    "iter_file_list_entries",
    "load_per_substrate_state_dict",
    "require_sha256",
    "sha256_file",
]

