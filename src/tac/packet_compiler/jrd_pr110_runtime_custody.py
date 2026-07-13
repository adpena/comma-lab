# SPDX-License-Identifier: MIT
"""Canonical byte custody for the executable PR110 submission runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

RUNTIME_RELATIVE_FILES = (
    "inflate.py",
    "src/codec.py",
    "src/codec_ctx.py",
    "src/codec_sidecar.py",
    "src/model.py",
    "src/frame_selector.py",
    "src/fec10_hybrid_decoder.py",
    "encoder/build_pr101_frame_exploit_selector_packet_fec10_hybrid.py",
    "encoder/build_pr101_frame_exploit_selector_packet_markov.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_custody(submission_dir: Path, repo_root: Path) -> dict[str, Any]:
    files = [submission_dir / relative for relative in RUNTIME_RELATIVE_FILES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"submission runtime is incomplete: {missing}")
    rows = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    canonical = (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode()
    return {"files": rows, "tree_sha256": hashlib.sha256(canonical).hexdigest()}
