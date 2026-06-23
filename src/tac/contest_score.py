# SPDX-License-Identifier: MIT
"""tac.contest_score — the SINGLE canonical contest-score formula helper.

THE COMPLIANCE BEDROCK. The contest score is computed by the pinned
upstream evaluator at ``upstream/evaluate.py:92`` (READ IT — never edit
it per CLAUDE.md "Non-Negotiable Upstream Rule"):

    score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate

where, at ``upstream/evaluate.py:63-65``::

    compressed_size   = (submission_dir / 'archive.zip').stat().st_size
    uncompressed_size = sum(f.stat().st_size for f in uncompressed_dir.rglob('*') if f.is_file())
    rate              = compressed_size / uncompressed_size

For the canonical contest video set the uncompressed size is the fixed
reference ``37_545_489`` bytes (the comma2k19 public-test payload total;
sister-pinned in ``src/tac/archive_byte_profile.py::CONTEST_ORIGINAL_BYTES``
and ``src/tac/joint_scorer_aware_training.py::LAMBDA_RATE_CONSTANT``).

Why this module exists (operator NON-NEGOTIABLE, 2026-06-23)
-----------------------------------------------------------
The score formula had been HAND-ROLLED across ~20+ files with the
``37_545_489`` constant scattered everywhere and NO single canonical
helper verified against upstream. A subagent dropped the ``× 25`` on the
rate term and reported a break-even ``d_seg`` of ``1.89e-3`` when the
correct value was ``~8.07e-4`` (off by ~2.3×). This module makes the
formula a single IMPORT so no consumer can slip:

    from tac.contest_score import compute_contest_score, break_even_d_seg

    S = compute_contest_score(d_seg, d_pose, archive_bytes)
    target_d_seg = break_even_d_seg(0.19110, d_pose, archive_bytes)

ADVISORY vs AUTHORITY (NO-FAKE supreme rule)
--------------------------------------------
This helper is the canonical PROXY + decision arithmetic. It is byte-
identical to the upstream formula given the SAME ``(d_seg, d_pose,
archive_bytes)`` inputs, but it is NEVER a substitute for a compliance
claim. The AUTHORITATIVE score is ALWAYS ``upstream/evaluate.py`` run on
the byte-closed archive on contest-compliant hardware (BOTH ``--device
cpu`` and ``--device cuda`` per CLAUDE.md "Submission auth eval — BOTH
CPU AND CUDA"). MPS-derived ``d_seg`` / ``d_pose`` inputs to this helper
yield ``[advisory only]`` numbers per the MPS-noise non-negotiable.

Relationship to sister modules
------------------------------
- :mod:`tac.score_composition` composes per-axis *deltas* into ΔS (the
  marginal / incremental form; pose term is the difference of two
  ``sqrt`` calls). THIS module computes the *absolute* score + the
  decision arithmetic (break-even) operating on absolute values.
- :data:`tac.archive_byte_profile.CONTEST_ORIGINAL_BYTES` is the sister
  rate-term-only home; this module re-exports a numerically identical
  :data:`UNCOMPRESSED_SIZE_BYTES`.
- :data:`tac.joint_scorer_aware_training.LAMBDA_RATE_CONSTANT` (=
  ``25 / 37545489``) is the training-side rate coefficient;
  ``rate_term(bytes) == LAMBDA_RATE_CONSTANT * bytes`` exactly.

Catalog #125 6-hook wire-in declaration
---------------------------------------
- Hook #1 sensitivity-map: ACTIVE — :func:`break_even_d_seg` IS the
  d_seg-axis decision threshold consumers compute sensitivity against.
- Hook #2 Pareto constraint: N/A — absolute-score helper; the per-axis
  Pareto polytope lives in :mod:`tac.score_composition` (delta form).
- Hook #3 bit-allocator: ACTIVE — :func:`rate_term` is the canonical
  byte→score-units conversion the bit-allocator consumes.
- Hook #4 cathedral autopilot dispatch: ACTIVE — dispatch/fire decisions
  (dashboard, slope-gate) consume :func:`break_even_d_seg`.
- Hook #5 continual-learning posterior: N/A — pure arithmetic, no
  posterior state.
- Hook #6 probe-disambiguator: N/A — single canonical formula, no
  competing interpretations.
"""
from __future__ import annotations

import math

__all__ = [
    "POSE_WEIGHT",
    "RATE_WEIGHT",
    "SEG_WEIGHT",
    "UNCOMPRESSED_SIZE_BYTES",
    "break_even_d_seg",
    "compute_contest_score",
    "pose_term",
    "rate_term",
    "seg_term",
]

# === Canonical scorer constants (cite: upstream/evaluate.py:92) ===
# uncompressed_size for the canonical contest video set. Sister-pinned at
# ``src/tac/archive_byte_profile.py::CONTEST_ORIGINAL_BYTES`` (== 37545489).
UNCOMPRESSED_SIZE_BYTES: int = 37_545_489
"""Reference denominator for the rate term (uncompressed_size bytes).

Per ``upstream/evaluate.py:64-65`` ``rate = compressed_size /
uncompressed_size``; for the canonical contest video set
``uncompressed_size == 37_545_489``.
"""

SEG_WEIGHT: float = 100.0
"""Coefficient on ``segnet_dist`` (``100 *`` in upstream/evaluate.py:92)."""

POSE_WEIGHT: float = 10.0
"""Constant inside ``sqrt(posenet_dist * 10)`` (upstream/evaluate.py:92)."""

RATE_WEIGHT: float = 25.0
"""Coefficient on ``rate`` (``25 *`` in upstream/evaluate.py:92)."""


def seg_term(d_seg: float) -> float:
    """SegNet contribution to the contest score: ``100 * d_seg``.

    Per ``upstream/evaluate.py:92`` (``100 * segnet_dist``).
    """
    _require_finite_nonneg("d_seg", d_seg)
    return SEG_WEIGHT * float(d_seg)


def pose_term(d_pose: float) -> float:
    """PoseNet contribution to the contest score: ``sqrt(10 * d_pose)``.

    Per ``upstream/evaluate.py:92`` (``math.sqrt(posenet_dist * 10)``).
    The marginal sensitivity ``d/d(d_pose) sqrt(10 * d_pose) =
    5 / sqrt(10 * d_pose)`` grows without bound as ``d_pose -> 0`` —
    see CLAUDE.md "SegNet vs PoseNet importance — operating-point
    dependent".
    """
    _require_finite_nonneg("d_pose", d_pose)
    return math.sqrt(POSE_WEIGHT * float(d_pose))


def rate_term(
    archive_bytes: int | float,
    *,
    uncompressed_size: int = UNCOMPRESSED_SIZE_BYTES,
) -> float:
    """Rate contribution to the contest score: ``25 * archive_bytes / N``.

    Per ``upstream/evaluate.py:64-65,92``: ``rate = compressed_size /
    uncompressed_size`` and the score adds ``25 * rate``. ``archive_bytes``
    is the size in bytes of the submitted ``archive.zip``.

    THIS is the term whose ``× 25`` a subagent dropped on 2026-06-23 —
    centralizing it here makes that slip impossible for any consumer.
    """
    _require_finite_nonneg("archive_bytes", archive_bytes)
    if not isinstance(uncompressed_size, int) or isinstance(
        uncompressed_size, bool
    ):
        raise TypeError(
            "rate_term: uncompressed_size must be a positive int, got "
            f"{type(uncompressed_size).__name__}"
        )
    if uncompressed_size <= 0:
        raise ValueError(
            "rate_term: uncompressed_size must be > 0 (got "
            f"{uncompressed_size})"
        )
    return RATE_WEIGHT * float(archive_bytes) / uncompressed_size


def compute_contest_score(
    d_seg: float,
    d_pose: float,
    archive_bytes: int | float,
    *,
    uncompressed_size: int = UNCOMPRESSED_SIZE_BYTES,
) -> float:
    """Compute the contest score, byte-identical to upstream/evaluate.py:92.

        score = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / N

    Args:
        d_seg: SegNet distortion (argmax-disagreement rate, in [0, 1]).
        d_pose: PoseNet distortion (6-dim pose MSE; >= 0).
        archive_bytes: ``archive.zip`` size in bytes (>= 0).
        uncompressed_size: rate denominator; defaults to the canonical
            ``37_545_489`` contest reference.

    Returns:
        The contest score as a float. NON-AUTHORITATIVE proxy: a
        compliance/frontier claim REQUIRES ``upstream/evaluate.py`` on
        the byte-closed archive (CPU and CUDA) per CLAUDE.md.
    """
    return (
        seg_term(d_seg)
        + pose_term(d_pose)
        + rate_term(archive_bytes, uncompressed_size=uncompressed_size)
    )


def break_even_d_seg(
    target_S: float,
    d_pose: float,
    archive_bytes: int | float,
    *,
    uncompressed_size: int = UNCOMPRESSED_SIZE_BYTES,
) -> float:
    """The ``d_seg`` at which the contest score exactly equals ``target_S``.

    Inverting ``upstream/evaluate.py:92`` for ``d_seg``::

        target_S = 100 * d_seg + sqrt(10 * d_pose) + 25 * bytes / N
        =>  d_seg = (target_S - sqrt(10 * d_pose) - 25 * bytes / N) / 100
                  = (target_S - pose_term(d_pose) - rate_term(bytes)) / 100

    This is the EXACT decision arithmetic a subagent got wrong on
    2026-06-23 (it dropped the ``× 25`` on the rate term, reporting
    ``1.89e-3`` when the correct break-even was ``~8.07e-4``). Now
    centralized + correct.

    Args:
        target_S: the score to hit (e.g. ``0.19110`` to beat the
            current frontier; ``0.15`` for the sub-0.15 goal).
        d_pose: current PoseNet distortion (operating point).
        archive_bytes: current ``archive.zip`` size in bytes.
        uncompressed_size: rate denominator (default canonical).

    Returns:
        The break-even ``d_seg`` as a float. May be negative if the
        ``target_S`` is already unreachable given the current pose+rate
        budget (i.e. ``pose_term + rate_term > target_S``); a negative
        return is a valid signal that the d_seg axis cannot get there
        alone — the caller must cut pose or rate.
    """
    if not isinstance(target_S, (int, float)) or isinstance(target_S, bool):
        raise TypeError(
            "break_even_d_seg: target_S must be numeric, got "
            f"{type(target_S).__name__}"
        )
    if math.isnan(target_S) or math.isinf(target_S):
        raise ValueError("break_even_d_seg: target_S must be finite")
    return (
        float(target_S)
        - pose_term(d_pose)
        - rate_term(archive_bytes, uncompressed_size=uncompressed_size)
    ) / SEG_WEIGHT


def _require_finite_nonneg(name: str, value: object) -> None:
    """Validate ``value`` is a finite, non-negative real number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    fvalue = float(value)
    if math.isnan(fvalue) or math.isinf(fvalue):
        raise ValueError(f"{name} must be finite (no NaN / inf), got {value}")
    if fvalue < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
