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
  ``rate_term`` follows the evaluator's binary64 operation order exactly:
  divide bytes by the denominator first, then multiply that rate by 25.
  Algebraically equivalent pre-multiplication is not bit-identical.

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
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "POSE_WEIGHT",
    "RATE_WEIGHT",
    "SEG_WEIGHT",
    "UNCOMPRESSED_SIZE_BYTES",
    "UPSTREAM_VIDEOS_DIR",
    "RateDenominatorMismatchError",
    "RateDenominatorVerdict",
    "assert_upstream_videos_clean",
    "break_even_d_seg",
    "compute_contest_score",
    "expected_video_names",
    "pose_term",
    "rate_term",
    "seg_term",
    "verify_upstream_videos_clean",
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


# === Rate-denominator cleanliness guard (task #812, FEED-dg1) =================
# The rate DENOMINATOR is DYNAMIC. ``upstream/evaluate.py:64`` computes it as::
#
#     uncompressed_size = sum(f.stat().st_size
#                             for f in uncompressed_dir.rglob('*') if f.is_file())
#
# It rglobs the WHOLE ``upstream/videos/`` tree and COUNTS DOTFILES — MEASURED
# 2026-07-31: a stray macOS ``._0.mkv`` (AppleDouble) or ``.DS_Store`` is matched
# by ``rglob('*')`` and silently inflates the sum (100 -> 180 in a fixture).
# Our whole stack hardcodes ``UNCOMPRESSED_SIZE_BYTES = 37_545_489`` as that
# denominator. If a stray lands, the REAL contest denominator grows, every
# rate/score number we compute is silently wrong, and nothing warns. Historical
# precedent: old bootstrap scripts carried ``find upstream -name '._*' -delete``.
#
# This guard verifies the hardcoded constant still equals the LIVE filesystem
# sum + inventory whenever the constant is actually used AND the tree is
# observable. It NEVER deletes anything (``upstream/`` is IMMUTABLE per CLAUDE.md
# "Non-Negotiable Upstream Rule") — it raises fail-closed and NAMES the offending
# files for the operator to clear. Sister warn-only preflight:
# ``tac.preflight.check_upstream_videos_dir_clean``.

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

UPSTREAM_VIDEOS_DIR: Path = _REPO_ROOT / "upstream" / "videos"
"""Canonical location of the contest uncompressed-video payload — the rate-term
denominator source per ``upstream/evaluate.py:64`` (``--uncompressed-dir``)."""

_VIDEO_NAMES_FILE: Path = _REPO_ROOT / "upstream" / "public_test_video_names.txt"
_FALLBACK_VIDEO_NAMES: tuple[str, ...] = ("0.mkv",)


class RateDenominatorMismatchError(RuntimeError):
    """Raised when the live ``upstream/videos/`` byte-sum/inventory no longer
    matches the hardcoded canonical denominator (:data:`UNCOMPRESSED_SIZE_BYTES`).

    A mismatch means ``upstream/evaluate.py:64`` would compute a DIFFERENT
    denominator than our stack assumes — every rate/score number is corrupted.
    The message is the cleanliness report NAMING the offending files. NEVER
    auto-remediated: ``upstream/`` is immutable per CLAUDE.md; the operator
    clears the stray (e.g. ``rm upstream/videos/.DS_Store``)."""


@dataclass(frozen=True)
class RateDenominatorVerdict:
    """Structured result of verifying ``upstream/videos/`` cleanliness.

    ``present``/``clean`` are the two decision bits: a fail-closed consumer
    raises iff ``present and not clean`` (an absent/unverifiable tree is never a
    violation — the constant simply stands as the assumption)."""

    present: bool
    clean: bool
    videos_dir: str
    dynamic_sum: int
    expected_sum: int
    expected_files: tuple[str, ...]
    actual_files: tuple[str, ...]
    strays: tuple[str, ...]
    missing: tuple[str, ...]
    sum_matches: bool
    report: str


def expected_video_names() -> tuple[str, ...]:
    """The set of legitimate files that should live under ``upstream/videos/``.

    Sourced from ``upstream/public_test_video_names.txt`` (one name per line —
    the same file ``upstream/evaluate.py:55-56`` reads for the test set). Falls
    back to ``("0.mkv",)`` if the names file is absent/unreadable, so the guard
    still has a reference inventory on hosts without the full upstream tree.

    NOTE the asymmetry that IS the vulnerability: evaluate.py reads this file to
    pick the SCORED videos (line 56) but sums the ENTIRE directory for the
    denominator (line 64). A stray that is not in this list still inflates the
    denominator — hence the inventory check compares against this expected set.
    """
    try:
        text = _VIDEO_NAMES_FILE.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_VIDEO_NAMES
    names = tuple(sorted({line.strip() for line in text.splitlines() if line.strip()}))
    return names or _FALLBACK_VIDEO_NAMES


def verify_upstream_videos_clean(
    videos_dir: Path | str | None = None,
    *,
    expected_sum: int = UNCOMPRESSED_SIZE_BYTES,
    expected_names: tuple[str, ...] | None = None,
) -> RateDenominatorVerdict:
    """Compute a :class:`RateDenominatorVerdict` for ``videos_dir``.

    NEVER raises for cleanliness — it REPORTS (use
    :func:`assert_upstream_videos_clean` to fail-closed). Replicates
    ``upstream/evaluate.py:64`` exactly::

        sum(f.stat().st_size for f in dir.rglob('*') if f.is_file())

    INCLUDING dotfiles (MEASURED: ``rglob('*')`` matches ``._0.mkv`` /
    ``.DS_Store``). An absent or unreadable dir yields
    ``present=False``/``clean=True`` (unverifiable → the constant stands as the
    assumption; the guard never fabricates a violation from a missing surface,
    matching the sibling anti-rot gates in ``tac.preflight``)."""
    root = Path(videos_dir) if videos_dir is not None else UPSTREAM_VIDEOS_DIR
    exp_names = (
        tuple(sorted(expected_names)) if expected_names is not None else expected_video_names()
    )
    exp_set = set(exp_names)

    if not root.is_dir():
        return RateDenominatorVerdict(
            present=False, clean=True, videos_dir=str(root),
            dynamic_sum=0, expected_sum=expected_sum,
            expected_files=exp_names, actual_files=(),
            strays=(), missing=(), sum_matches=True,
            report=(
                f"upstream videos dir absent ({root}) — rate denominator "
                f"UNVERIFIABLE on this host; assuming canonical constant "
                f"{expected_sum}. (No block: nothing to check.)"
            ),
        )

    try:
        entries = [p for p in root.rglob("*") if p.is_file()]
        rels = tuple(sorted(p.relative_to(root).as_posix() for p in entries))
        dynamic_sum = sum(p.stat().st_size for p in entries)
    except OSError as exc:
        return RateDenominatorVerdict(
            present=True, clean=True, videos_dir=str(root),
            dynamic_sum=0, expected_sum=expected_sum,
            expected_files=exp_names, actual_files=(),
            strays=(), missing=(), sum_matches=True,
            report=(
                f"upstream videos dir present but unreadable ({root}: {exc}) — "
                "rate denominator UNVERIFIABLE; not blocking on an I/O error."
            ),
        )

    actual_set = set(rels)
    strays = tuple(sorted(actual_set - exp_set))
    missing = tuple(sorted(exp_set - actual_set))
    sum_matches = dynamic_sum == expected_sum
    clean = not strays and not missing and sum_matches

    if clean:
        report = (
            f"upstream/videos/ CLEAN: {list(rels)} sum={dynamic_sum} "
            f"== canonical denominator {expected_sum}."
        )
    else:
        lines = [f"upstream/videos/ rate-denominator MISMATCH at {root}:"]
        if strays:
            lines.append(
                "  STRAY file(s) inflating the denominator (evaluate.py:64 "
                f"rglob counts these, incl. dotfiles): {list(strays)}"
            )
        if missing:
            lines.append(f"  MISSING expected file(s): {list(missing)}")
        if not sum_matches:
            lines.append(
                f"  BYTE-SUM {dynamic_sum} != canonical {expected_sum} "
                f"(delta {dynamic_sum - expected_sum:+d})."
            )
        lines.append(
            "  upstream/ is IMMUTABLE (CLAUDE.md) — do NOT auto-delete; operator "
            "clears strays (e.g. `rm upstream/videos/.DS_Store`) or restores the "
            "missing payload."
        )
        report = "\n".join(lines)

    return RateDenominatorVerdict(
        present=True, clean=clean, videos_dir=str(root),
        dynamic_sum=dynamic_sum, expected_sum=expected_sum,
        expected_files=exp_names, actual_files=rels,
        strays=strays, missing=missing, sum_matches=sum_matches,
        report=report,
    )


def assert_upstream_videos_clean(
    videos_dir: Path | str | None = None,
    *,
    expected_sum: int = UNCOMPRESSED_SIZE_BYTES,
    expected_names: tuple[str, ...] | None = None,
) -> RateDenominatorVerdict:
    """Fail-closed variant: raise :class:`RateDenominatorMismatchError` iff the
    tree is PRESENT and NOT clean; otherwise return the verdict.

    Absent/unreadable tree → no raise (unverifiable is not a violation)."""
    verdict = verify_upstream_videos_clean(
        videos_dir, expected_sum=expected_sum, expected_names=expected_names
    )
    if verdict.present and not verdict.clean:
        raise RateDenominatorMismatchError(verdict.report)
    return verdict


_DEFAULT_DENOMINATOR_VERDICT: RateDenominatorVerdict | None = None


def _assert_default_denominator_clean_cached() -> None:
    """Cheap per-process guard used by :func:`rate_term` when the caller relies
    on the default canonical denominator. First call does one ``rglob`` of
    ``upstream/videos/`` (a ~1-file dir); the verdict is cached, so subsequent
    calls are O(1). Re-raises on a cached dirty verdict (a dirty tree can never
    yield a valid score, so re-raising is correct)."""
    global _DEFAULT_DENOMINATOR_VERDICT
    if _DEFAULT_DENOMINATOR_VERDICT is None:
        _DEFAULT_DENOMINATOR_VERDICT = verify_upstream_videos_clean()
    verdict = _DEFAULT_DENOMINATOR_VERDICT
    if verdict.present and not verdict.clean:
        raise RateDenominatorMismatchError(verdict.report)


def _reset_denominator_cache() -> None:
    """Test hook: clear the per-process cached default-denominator verdict."""
    global _DEFAULT_DENOMINATOR_VERDICT
    _DEFAULT_DENOMINATOR_VERDICT = None


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
    if not isinstance(uncompressed_size, int) or isinstance(uncompressed_size, bool):
        raise TypeError(f"rate_term: uncompressed_size must be a positive int, got {type(uncompressed_size).__name__}")
    if uncompressed_size <= 0:
        raise ValueError(f"rate_term: uncompressed_size must be > 0 (got {uncompressed_size})")
    # #812 rate-denominator guard (fail-closed, BEFORE any rate arithmetic):
    # when the caller relies on the canonical constant, verify upstream/videos/
    # still sums to it — a stray macOS dotfile would silently change
    # evaluate.py:64's denominator and corrupt every score. No-op when the tree
    # is absent/clean; cached per-process (one rglob of a ~1-file dir). Skipped
    # when the caller passes an explicit non-canonical uncompressed_size (a
    # deliberate hypothetical that is NOT claiming the real contest denominator).
    if uncompressed_size == UNCOMPRESSED_SIZE_BYTES:
        _assert_default_denominator_clean_cached()
    # Preserve upstream/evaluate.py's binary64 DAG exactly.  Upstream first
    # forms ``rate = compressed_size / uncompressed_size`` and only then
    # evaluates ``25 * rate``.  ``25 * compressed_size / uncompressed_size``
    # differs by one ULP for some integer byte counts (including 1), so the
    # tempting algebraic reassociation is forbidden on custody-bearing paths.
    rate = float(archive_bytes) / uncompressed_size
    return RATE_WEIGHT * rate


def compute_contest_score(
    d_seg: float,
    d_pose: float,
    archive_bytes: int | float,
    *,
    uncompressed_size: int = UNCOMPRESSED_SIZE_BYTES,
) -> float:
    """Compute the contest score, byte-identical to upstream/evaluate.py:92.

        rate = archive_bytes / N
        score = 100 * d_seg + sqrt(10 * d_pose) + 25 * rate

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
    return seg_term(d_seg) + pose_term(d_pose) + rate_term(archive_bytes, uncompressed_size=uncompressed_size)


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
        raise TypeError(f"break_even_d_seg: target_S must be numeric, got {type(target_S).__name__}")
    if math.isnan(target_S) or math.isinf(target_S):
        raise ValueError("break_even_d_seg: target_S must be finite")
    return (
        float(target_S) - pose_term(d_pose) - rate_term(archive_bytes, uncompressed_size=uncompressed_size)
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
