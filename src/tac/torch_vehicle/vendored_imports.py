# SPDX-License-Identifier: MIT
"""Pristine-source import shim for the vendored PR95 HNeRV-Muon trainer.

The vendored source lives in a gitignored public-PR intake clone and MUST stay
byte-pristine (CLAUDE.md "Forbidden in-place edits to public PR intake clones").
This module is the ONLY place that puts the clone on ``sys.path`` and resolves
the challenge root the vendored ``data.py`` needs (it expects a
``comma_video_compression_challenge/`` sibling; our repo vendors it under
``workspace/upstream/`` and ``upstream/``).

NO vendored file is mutated; we only:
  * prepend the clone's ``src`` (and ``src/stages``) to ``sys.path`` so the
    vendored flat imports (``from model import ...``) resolve;
  * set ``COMMA_CHALLENGE_ROOT`` (if unset) so ``data.py`` finds the frozen
    SegNet/PoseNet + ``frame_utils.yuv420_to_rgb`` GT decode.

Idempotent: repeated calls are no-ops.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

# Repo root: this file is src/tac/torch_vehicle/vendored_imports.py
_REPO_ROOT = Path(__file__).resolve().parents[3]

VENDORED_SRC = (
    _REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src"
)

# Candidate challenge roots (the dir containing frame_utils.py + modules.py +
# the segnet/posenet weights), in preference order.
_CHALLENGE_ROOT_CANDIDATES = (
    _REPO_ROOT / "workspace/upstream/comma_video_compression_challenge",
    _REPO_ROOT / "upstream",
)


def resolve_challenge_root() -> Path:
    """Return the challenge root dir (with frame_utils.py + modules.py).

    Honors an explicit ``COMMA_CHALLENGE_ROOT`` env override; else picks the
    first existing candidate. Raises if none found (fail closed — a missing
    frozen scorer is a hard blocker, not a silent CPU fallback).
    """
    env = os.environ.get("COMMA_CHALLENGE_ROOT")
    if env:
        cand = Path(env).resolve()
        if (cand / "frame_utils.py").exists() and (cand / "modules.py").exists():
            return cand
        raise FileNotFoundError(
            f"COMMA_CHALLENGE_ROOT={env} lacks frame_utils.py/modules.py"
        )
    for cand in _CHALLENGE_ROOT_CANDIDATES:
        if (cand / "frame_utils.py").exists() and (cand / "modules.py").exists():
            return cand.resolve()
    raise FileNotFoundError(
        "No challenge root with frame_utils.py + modules.py found. "
        f"Tried {[str(c) for c in _CHALLENGE_ROOT_CANDIDATES]}. "
        "Set COMMA_CHALLENGE_ROOT."
    )


def _ensure_sys_path() -> None:
    """Prepend the vendored src (+ stages) to sys.path, idempotently."""
    if not VENDORED_SRC.exists():
        raise FileNotFoundError(
            f"Vendored PR95 src not found at {VENDORED_SRC}. "
            "The public-PR intake clone is gitignored; ensure it is present."
        )
    for p in (str(VENDORED_SRC), str(VENDORED_SRC / "stages")):
        if p not in sys.path:
            sys.path.insert(0, p)


def import_vendored(module_name: str) -> ModuleType:
    """Import a vendored module by its flat name (e.g. ``"model"``, ``"codec"``).

    Sets up sys.path + COMMA_CHALLENGE_ROOT first. The vendored ``data.py``
    applies its differentiable-yuv6 monkeypatch at import time (the contest
    pose-gradient fix); importing it here triggers that patch in OUR process —
    that is the vendored author's intended runtime behavior, NOT a source edit.
    """
    _ensure_sys_path()
    if "COMMA_CHALLENGE_ROOT" not in os.environ:
        os.environ["COMMA_CHALLENGE_ROOT"] = str(resolve_challenge_root())
    return importlib.import_module(module_name)


__all__ = ["VENDORED_SRC", "import_vendored", "resolve_challenge_root"]
