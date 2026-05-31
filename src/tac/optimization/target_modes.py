# SPDX-License-Identifier: MIT
"""Shared optimization target-mode contract.

These modes make contest-video overfitting an explicit, portable setting rather
than hidden behavior in queue builders, materializers, or scorer-response
surfaces. Contest mode may optimize directly against the declared challenge
video. Production modes must bind a declared corpus or hybrid target upstream.
"""

from __future__ import annotations

from typing import Final

CONTEST_VIDEO_OVERFIT_MODE: Final[str] = "contest_video_overfit"
CORPUS_GENERALIZATION_MODE: Final[str] = "corpus_generalization"
HYBRID_CONTEST_PLUS_CORPUS_MODE: Final[str] = "hybrid_contest_plus_corpus"

TARGET_OPTIMIZATION_MODES: Final[frozenset[str]] = frozenset(
    {
        CONTEST_VIDEO_OVERFIT_MODE,
        CORPUS_GENERALIZATION_MODE,
        HYBRID_CONTEST_PLUS_CORPUS_MODE,
    }
)

OVERFIT_ALLOWED_TARGET_MODES: Final[frozenset[str]] = frozenset(
    {
        CONTEST_VIDEO_OVERFIT_MODE,
        HYBRID_CONTEST_PLUS_CORPUS_MODE,
    }
)

CORPUS_REQUIRED_TARGET_MODES: Final[frozenset[str]] = frozenset(
    {
        CORPUS_GENERALIZATION_MODE,
        HYBRID_CONTEST_PLUS_CORPUS_MODE,
    }
)


def normalize_target_optimization_mode(
    mode: str,
    *,
    field_name: str = "target_mode",
) -> str:
    """Return a validated target optimization mode."""

    normalized = str(mode or "").strip()
    if normalized not in TARGET_OPTIMIZATION_MODES:
        raise ValueError(
            f"{field_name} must be one of {sorted(TARGET_OPTIMIZATION_MODES)}; "
            f"got {mode!r}"
        )
    return normalized


def target_mode_declares_overfit_allowed(mode: str) -> bool:
    """Return whether ``mode`` permits overfitting to the declared contest video."""

    return normalize_target_optimization_mode(mode) in OVERFIT_ALLOWED_TARGET_MODES


def target_mode_requires_corpus_manifest(mode: str) -> bool:
    """Return whether ``mode`` requires a declared corpus manifest upstream."""

    return normalize_target_optimization_mode(mode) in CORPUS_REQUIRED_TARGET_MODES


__all__ = [
    "CONTEST_VIDEO_OVERFIT_MODE",
    "CORPUS_GENERALIZATION_MODE",
    "CORPUS_REQUIRED_TARGET_MODES",
    "HYBRID_CONTEST_PLUS_CORPUS_MODE",
    "OVERFIT_ALLOWED_TARGET_MODES",
    "TARGET_OPTIMIZATION_MODES",
    "normalize_target_optimization_mode",
    "target_mode_declares_overfit_allowed",
    "target_mode_requires_corpus_manifest",
]
