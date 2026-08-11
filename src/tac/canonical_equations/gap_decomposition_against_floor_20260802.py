# SPDX-License-Identifier: MIT
"""Per-axis gap decomposition of S against a DEMONSTRATED floor, and the denominator
every ΔS claim must be quoted against.

WHY THIS EXISTS (ddm_cv1, 2026-08-02).  The campaign repeatedly quoted ΔS values with no
denominator, which makes a 0.0077%-of-gap win and a 12%-of-gap win read identically.  This
equation makes the denominator executable: ``fraction_of_gap`` refuses to evaluate unless a
floor has been supplied, so a ΔS cannot be reported as "progress" without saying progress
toward what.

WHAT A "GAP" IS HERE.  Not the raw axis contribution.  The gap is measured against a
DEMONSTRATED floor -- an externally achieved row proving the axis value is reachable.  Our
own axis contribution over-states how much is available (part of it is the floor itself,
which nobody has beaten), and the raw contribution ordering can therefore disagree with the
gap ordering.  On the 2026-08-02 state it does not, but the ordering is a measured output,
never an assumption -- so ``rank_by_gap`` returns it rather than any caller hardcoding it.

RATE DENOMINATOR (Catalog #812 / ddm_us1).  ``upstream/evaluate.py`` sums
``rglob('*')`` over ``upstream/videos/`` DYNAMICALLY; it does NOT use a literal
37,545,489.  A stray ``._*`` or ``.DS_Store`` silently changes it.  This module therefore
REFUSES to default the denominator and requires it as a measured input with its source.

S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/rate_denominator_bytes
    (upstream/evaluate.py:92)
"""
from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

EQUATION_ID = "gap_decomposition_against_demonstrated_floor_v1"

_SEG_COEFF = 100.0
_POSE_COEFF = 10.0
_RATE_COEFF = 25.0

# Explicit opt-out sentinel for the pointer cross-check. A distinct object, never None:
# "use the default pointer" and "deliberately skip the check" must not share a value.
SKIP_POINTER_CROSSCHECK = "__ddm_op3_skip_pointer_crosscheck__"

# 600 scored pairs x the SegNet argmax lattice the flip count is defined over.
# Used only by the seg<->rate exchange rate; never by S itself.
_SCORED_PIXELS = 600 * 512 * 384


def _require_finite_nonneg(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return out


@dataclass(frozen=True)
class MeasuredScoreTriple:
    """A byte-closed (d_seg, d_pose, archive_bytes) row plus its rate denominator.

    ``source_artifact`` is mandatory: an unsourced triple cannot anchor a gap, and a gap
    without a source is exactly the unanchored-ΔS failure this equation exists to stop.
    """

    d_seg: float
    d_pose: float
    archive_bytes: int
    rate_denominator_bytes: int
    source_artifact: str
    axis_tag: str
    status: str = "MEASURED"

    def __post_init__(self) -> None:
        if self.status != "MEASURED":
            raise ValueError("only status=MEASURED is accepted; derived rows cannot anchor a gap")
        _require_finite_nonneg(self.d_seg, "d_seg")
        _require_finite_nonneg(self.d_pose, "d_pose")
        for name in ("archive_bytes", "rate_denominator_bytes"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TypeError(f"{name} must be int (a byte count), got {type(raw).__name__}")
            if raw <= 0:
                raise ValueError(f"{name} must be > 0, got {raw}")
        if not self.source_artifact.strip():
            raise ValueError("source_artifact is required")
        if not self.axis_tag.strip():
            raise ValueError("axis_tag is required (e.g. '[macOS-CPU advisory exact n600]')")

    @property
    def seg_contribution(self) -> float:
        return _SEG_COEFF * float(self.d_seg)

    @property
    def pose_contribution(self) -> float:
        return math.sqrt(_POSE_COEFF * float(self.d_pose))

    @property
    def rate_contribution(self) -> float:
        return _RATE_COEFF * float(self.archive_bytes) / float(self.rate_denominator_bytes)

    @property
    def total(self) -> float:
        """S recomputed from components -- never read a rounded 'Final score' field."""
        return self.seg_contribution + self.pose_contribution + self.rate_contribution


@dataclass(frozen=True)
class GapDecomposition:
    """Our measured row against a demonstrated floor. Both must be MEASURED."""

    ours: MeasuredScoreTriple
    floor: MeasuredScoreTriple

    def __post_init__(self) -> None:
        if self.ours.rate_denominator_bytes != self.floor.rate_denominator_bytes:
            raise ValueError(
                "rate denominators differ between our row and the floor row; the rate axis "
                "is not comparable across different upstream/videos/ contents (Catalog #812)"
            )

    @property
    def seg_gap(self) -> float:
        return self.ours.seg_contribution - self.floor.seg_contribution

    @property
    def pose_gap(self) -> float:
        return self.ours.pose_contribution - self.floor.pose_contribution

    @property
    def rate_gap(self) -> float:
        return self.ours.rate_contribution - self.floor.rate_contribution

    @property
    def total_gap(self) -> float:
        return self.ours.total - self.floor.total

    def per_axis(self) -> dict[str, float]:
        return {"seg": self.seg_gap, "pose": self.pose_gap, "rate": self.rate_gap}

    def shares(self) -> dict[str, float]:
        """Fraction of the TOTAL gap carried by each axis.

        Negative gaps (an axis where we already beat the floor) are reported as-is rather
        than clipped: a negative share is real information -- it says that axis is not
        where the remaining work is, and clipping it would silently inflate the others.
        """
        total = self.total_gap
        if abs(total) < 1e-12:
            raise ValueError("total gap is ~0; shares are undefined (we are AT the floor)")
        return {k: v / total for k, v in self.per_axis().items()}

    def rank_by_gap(self) -> tuple[str, ...]:
        """Axes ordered by remaining gap, largest first. MEASURED OUTPUT, not an assumption."""
        return tuple(
            name for name, _ in sorted(self.per_axis().items(), key=lambda kv: -kv[1])
        )

    def fraction_of_gap(self, delta_s: float) -> float:
        """THE DENOMINATOR. A ΔS expressed as a fraction of the remaining total gap.

        Sign convention matches the campaign: a score-LOWERING move has delta_s < 0 and
        returns a POSITIVE fraction (that much of the gap closed).
        """
        if isinstance(delta_s, bool) or not isinstance(delta_s, (int, float)):
            raise TypeError("delta_s must be numeric")
        value = float(delta_s)
        if not math.isfinite(value):
            raise ValueError("delta_s must be finite")
        total = self.total_gap
        if abs(total) < 1e-12:
            raise ValueError("total gap is ~0; fraction_of_gap is undefined")
        return -value / total

    def bytes_per_percent_of_gap(self) -> float:
        """How many archive bytes one percent of the remaining gap is worth.

        Converts the rate axis into the same currency as seg and pose so a byte saving and
        a distortion saving can be compared without re-deriving the exchange rate by hand.
        """
        total = self.total_gap
        if abs(total) < 1e-12:
            raise ValueError("total gap is ~0; exchange rate is undefined")
        bytes_per_s_unit = float(self.ours.rate_denominator_bytes) / _RATE_COEFF
        return abs(total) * 0.01 * bytes_per_s_unit

    def cross_axis_warning(self) -> str | None:
        """Non-None when our row and the floor were measured on DIFFERENT authority axes.

        The gap is still the best available ranking signal, but a reader who does not know
        the two legs come from different instruments will over-read its last digits. This
        is the instrument coordinate of the stale-fit genus (see ``ARGUMENT_AXES``): the
        number is fine, the missing argument is which evaluator produced each leg.
        """
        if self.ours.axis_tag.strip() == self.floor.axis_tag.strip():
            return None
        return (
            f"CROSS-AXIS: ours={self.ours.axis_tag!r} vs floor={self.floor.axis_tag!r}. "
            "The gap ranks axes correctly but its low-order digits are not a paired "
            "same-instrument comparison."
        )


# ---------------------------------------------------------------------------
# THE STALE-FIT GENUS -- the argument space a quoted scalar is evaluated at.
# ---------------------------------------------------------------------------
# WHY THIS IS HERE (ddm_op3, 2026-08-03).  Six separate arms priced a delta against a
# baseline that had already moved, and the sweep arm's own charter carried the defect.
# Enumerating incidents does not close a class.  The structural statement is:
#
#     A quoted scalar is the value of a function AT A POINT in the argument space below.
#     "A number that was true and silently stopped being true" is exactly the loss of one
#     of those coordinates.  It is ARGUMENT LOSS, not arithmetic error -- which is why
#     every instance passed every arithmetic check it was ever given.
#
# What makes this operational rather than a taxonomy is the third column: under a change
# of each argument a quantity is EXACT_INVARIANT (carry it forward unchanged),
# RESTATEABLE (carry it forward through a stated transformation), or NOT_RESTATEABLE
# (the number is void and must be re-measured).  Collapsing those three into "stale" is
# what made re-pricing look purely destructive; in fact banked POSE deltas are
# UNDER-priced as pose improves, and nobody had swept in that direction.
ARGUMENT_AXES: tuple[str, ...] = (
    "baseline",       # what the delta was measured AGAINST
    "floor",          # what the fraction/share was divided BY
    "operating_point",  # where a nonlinear marginal was evaluated
    "population",     # which samples (n600 vs a prefix vs a verdict subset)
    "scope",          # which files/cases the sweep actually covered (empty scope != PASS)
    "term_set",       # which of {seg, pose, rate} the delta covers
    "formulation",    # the definitional convention (which floor family, which labels)
    "instrument",     # the authority axis (exact CUDA / exact CPU / advisory / proxy)
)

EXACT_INVARIANT = "EXACT_INVARIANT"
RESTATEABLE = "RESTATEABLE"
NOT_RESTATEABLE = "NOT_RESTATEABLE"


@dataclass(frozen=True)
class AxisMarginal:
    """dS per unit of an axis quantity, WITH its behaviour under a move of the point.

    The campaign has been carrying marginals with no invariance label, so an exactly
    invariant exchange rate and a strongly operating-point-dependent one read identically.
    They are not alike: ``W`` is a ratio of two LINEAR terms and has no operating point at
    all, while the pose marginal is the derivative of a CONCAVE term and rises as pose
    improves.
    """

    axis: str
    value: float
    unit: str
    invariance: str
    derivation: str

    def __post_init__(self) -> None:
        if self.invariance not in (EXACT_INVARIANT, RESTATEABLE, NOT_RESTATEABLE):
            raise ValueError(f"unknown invariance class: {self.invariance!r}")


def marginals(triple: MeasuredScoreTriple) -> dict[str, AxisMarginal]:
    """The three dS/d(axis) marginals at ``triple``'s operating point, labelled.

    seg and rate are linear in S, so their marginals do not depend on the point at all.
    pose is not: ``dS/d(d_pose) = 5 / sqrt(10*d_pose)`` rises without bound as d_pose
    falls, so a pose distortion saving banked at a worse operating point is worth MORE
    today, not less.
    """
    pose_contribution = triple.pose_contribution
    if pose_contribution <= 0.0:
        raise ValueError("pose marginal is undefined at d_pose = 0 (the derivative diverges)")
    return {
        "seg": AxisMarginal(
            axis="seg",
            value=_SEG_COEFF,
            unit="S per unit d_seg",
            invariance=EXACT_INVARIANT,
            derivation="S has the linear term 100*d_seg; the coefficient is a constant",
        ),
        "pose": AxisMarginal(
            axis="pose",
            value=(_POSE_COEFF / 2.0) / pose_contribution,
            unit="S per unit d_pose",
            invariance=RESTATEABLE,
            derivation=(
                "d/d(d_pose) sqrt(10*d_pose) = 5/sqrt(10*d_pose) = 5/pose_contribution; "
                "CONCAVE, so the marginal rises as pose improves"
            ),
        ),
        "rate": AxisMarginal(
            axis="rate",
            value=_RATE_COEFF / float(triple.rate_denominator_bytes),
            unit="S per archive byte",
            invariance=EXACT_INVARIANT,
            derivation=(
                "S has the linear term 25*B/DEN. Invariant GIVEN the denominator -- but "
                "DEN is summed dynamically from upstream/videos/ (Catalog #812), so it is "
                "invariant in B, not in the directory contents"
            ),
        ),
    }


def seg_rate_exchange_bytes_per_flip(
    rate_denominator_bytes: int,
    scored_pixels: int = _SCORED_PIXELS,
) -> AxisMarginal:
    """``W`` -- archive bytes that buy one SegNet argmax flip. EXACTLY INVARIANT.

    Setting the two linear legs equal, ``25*dB/DEN = 100*d(d_seg)`` with
    ``flips = d(d_seg)*PX``, gives ``W = (100*DEN/25)/PX = 4*DEN/PX``.  There is no
    archive-size term and no distortion term, so W does NOT move as the archive shrinks.
    Stated explicitly because the opposite was assumed in a live charter.
    """
    if isinstance(rate_denominator_bytes, bool) or not isinstance(rate_denominator_bytes, int):
        raise TypeError("rate_denominator_bytes must be int")
    if rate_denominator_bytes <= 0 or scored_pixels <= 0:
        raise ValueError("rate_denominator_bytes and scored_pixels must be > 0")
    return AxisMarginal(
        axis="seg<->rate",
        value=(_SEG_COEFF * float(rate_denominator_bytes) / _RATE_COEFF) / float(scored_pixels),
        unit="archive bytes per argmax flip",
        invariance=EXACT_INVARIANT,
        derivation="W = 4*DEN/PX; a ratio of two linear terms has no operating point",
    )


def restate_pose_delta_at(
    delta_d_pose: float,
    banked_at: MeasuredScoreTriple,
    now: MeasuredScoreTriple,
) -> dict[str, float]:
    """Re-price a banked pose DISTORTION saving at today's operating point.

    Takes a saving in ``d_pose`` (the axis quantity, NOT a ΔS -- a banked ΔS on a concave
    term cannot be re-pointed, because the ΔS has already folded in the old point).  This
    is the RESTATEABLE case made executable, and it is the one direction in which
    re-pricing FINDS value: as d_pose falls the marginal rises, so an old pose lever is
    UNDER-priced.
    """
    if isinstance(delta_d_pose, bool) or not isinstance(delta_d_pose, (int, float)):
        raise TypeError("delta_d_pose must be numeric")
    value = float(delta_d_pose)
    if not math.isfinite(value):
        raise ValueError("delta_d_pose must be finite")
    then = marginals(banked_at)["pose"].value
    today = marginals(now)["pose"].value
    return {
        "marginal_when_banked": then,
        "marginal_now": today,
        "ratio": today / then,
        "delta_s_when_banked": -value * then,
        "delta_s_now": -value * today,
    }


# ---------------------------------------------------------------------------
# CUSTODY READERS -- the inputs must not be typeable by hand.
# ---------------------------------------------------------------------------
# Every wrong gap figure in this campaign was a correct equation fed a hand-typed input:
# 190,952 for 191,052 (a digit transposition in the one number every ranking divides by),
# and a dc1_fold-era gap carried two frontier moves past its own validity.  So the fix is
# not more arithmetic care -- it is removing the typing opportunity.

_REPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "d_pose": re.compile(r"^\s*Average PoseNet Distortion:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*$", re.M),
    "d_seg": re.compile(r"^\s*Average SegNet Distortion:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*$", re.M),
    "archive_bytes": re.compile(r"^\s*Submission file size:\s*([0-9][0-9,]*)\s*bytes\s*$", re.M),
    "rate_denominator_bytes": re.compile(r"^\s*Original uncompressed size:\s*([0-9][0-9,]*)\s*bytes\s*$", re.M),
}
_REPORT_SAMPLES = re.compile(r"Evaluation results over\s+([0-9][0-9,]*)\s+samples", re.M)


class EvaluatorReportParseError(ValueError):
    """Raised when an ``upstream/evaluate.py`` report cannot be read STRICTLY.

    Fails closed on purpose.  A tolerant parse of this file is how ``353,808`` became
    ``353``: the byte pattern is anchored to end-of-line and demands the ``bytes`` suffix
    so a partial match cannot succeed quietly.
    """


@dataclass(frozen=True)
class ParsedEvaluatorReport:
    """Every scoring input, read from one evaluator receipt. No literal survives."""

    path: str
    d_seg: float
    d_pose: float
    archive_bytes: int
    rate_denominator_bytes: int
    n_samples: int

    @property
    def total(self) -> float:
        return (
            _SEG_COEFF * self.d_seg
            + math.sqrt(_POSE_COEFF * self.d_pose)
            + _RATE_COEFF * self.archive_bytes / self.rate_denominator_bytes
        )


def parse_evaluator_report(path: str | Path) -> ParsedEvaluatorReport:
    """Parse an ``upstream/evaluate.py`` ``report.txt`` strictly, or raise.

    Reads the rate DENOMINATOR from the receipt too ("Original uncompressed size"), so
    Catalog #812's dynamic-denominator hazard is answered by custody rather than by a
    constant that happened to be right on the day it was typed.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="strict")
    except OSError as exc:
        raise EvaluatorReportParseError(f"cannot read evaluator report {p}: {exc}") from exc

    values: dict[str, str] = {}
    missing: list[str] = []
    for name, pattern in _REPORT_PATTERNS.items():
        found = pattern.findall(text)
        if not found:
            missing.append(name)
        elif len(found) > 1 and len(set(found)) > 1:
            raise EvaluatorReportParseError(
                f"{p}: field {name!r} appears {len(found)} times with differing values "
                f"{sorted(set(found))} -- ambiguous receipt, refusing"
            )
        else:
            values[name] = found[0]
    if missing:
        raise EvaluatorReportParseError(
            f"{p}: evaluator report missing required field(s) {sorted(missing)}; a report "
            "that does not carry every scoring input cannot anchor a claim"
        )

    samples_match = _REPORT_SAMPLES.search(text)
    if samples_match is None:
        raise EvaluatorReportParseError(
            f"{p}: no 'Evaluation results over N samples' header -- the POPULATION is one "
            "of the arguments a claim is evaluated at and cannot be assumed"
        )

    archive_bytes = int(values["archive_bytes"].replace(",", ""))
    denominator = int(values["rate_denominator_bytes"].replace(",", ""))
    n_samples = int(samples_match.group(1).replace(",", ""))
    # Round-4 self-review: the regexes accept "0", and ParsedEvaluatorReport.total would
    # then ZeroDivisionError deep inside a caller instead of naming the bad receipt here.
    for name, value in (
        ("archive_bytes", archive_bytes),
        ("rate_denominator_bytes", denominator),
        ("n_samples", n_samples),
    ):
        if value <= 0:
            raise EvaluatorReportParseError(
                f"{p}: {name} parsed as {value}; a receipt with a non-positive {name} "
                "cannot anchor a score"
            )
    return ParsedEvaluatorReport(
        path=str(p),
        d_seg=float(values["d_seg"]),
        d_pose=float(values["d_pose"]),
        archive_bytes=archive_bytes,
        rate_denominator_bytes=denominator,
        n_samples=n_samples,
    )


def triple_from_evaluator_report(
    path: str | Path,
    *,
    axis_tag: str,
    required_samples: int | None = 600,
) -> MeasuredScoreTriple:
    """A ``MeasuredScoreTriple`` whose every numeric field came from a receipt.

    ``required_samples`` defaults to the full 600-pair population: a subset receipt is a
    DIFFERENT population, and a prefix of a temporally-correlated video list is a scene
    block rather than a sample.  Pass ``None`` only when the subset itself is the subject.
    """
    parsed = parse_evaluator_report(path)
    if required_samples is not None and parsed.n_samples != required_samples:
        raise EvaluatorReportParseError(
            f"{parsed.path}: receipt covers {parsed.n_samples} samples, not "
            f"{required_samples}; a subset is a different population, not a smaller "
            "measurement of the same one"
        )
    return MeasuredScoreTriple(
        d_seg=parsed.d_seg,
        d_pose=parsed.d_pose,
        archive_bytes=parsed.archive_bytes,
        rate_denominator_bytes=parsed.rate_denominator_bytes,
        source_artifact=f"{parsed.path} (n={parsed.n_samples})",
        axis_tag=axis_tag,
    )


# The external demonstrated floor.  Kept as named constants WITH an executable
# reproduction check rather than read from a JSON nobody validates: the published row is
# the only thing that can adjudicate them, and 190,952 vs 191,052 is decided precisely by
# whether the recomputation reproduces it.
PR130_FLOOR_D_SEG = 2.9660e-4
PR130_FLOOR_D_POSE = 2.331e-5
PR130_FLOOR_ARCHIVE_BYTES = 191_052
PR130_FLOOR_PUBLISHED_TOTAL = 0.172141
PR130_FLOOR_SOURCE = (
    ".omx/research/pr86_pr130_fullstack_intake_20260728.md:140 (PR130 bot row, official "
    "rail). Bytes CORRECTED 190,952 -> 191,052 by ddm_na1 2026-08-02: 190,952 is the "
    "inner 'p' payload, 191,052 is archive.zip, and evaluate.py:63 charges archive.zip. "
    "190,952 gives 0.1720747, which does not reproduce the published 0.172141."
)


def _default_frontier_pointer_path() -> Path:
    """Locate the canonical pointer from THIS FILE, never from the caller's cwd.

    Round-2 self-review defect: the first draft defaulted to the relative string
    ``.omx/state/canonical_frontier_pointer.json``. Called from any other working
    directory that path does not exist, the loader takes the "pointer absent" branch, and
    the bar cross-check SILENTLY DOES NOT RUN -- a check that skips itself under a
    condition nobody states is the vacuity failure this module was written to attack.
    """
    return Path(__file__).resolve().parents[3] / ".omx" / "state" / "canonical_frontier_pointer.json"


def demonstrated_floor_pr130(
    rate_denominator_bytes: int,
    *,
    frontier_pointer_path: str | Path | None = None,
) -> MeasuredScoreTriple:
    """The PR130 floor triple, cross-checked against BOTH its published row and custody.

    Two independent checks, because this is the number every ranking divides by:

    1. the recomputed total must reproduce the published ``0.172141`` to 6 dp -- this is
       what decided 191,052 over 190,952;
    2. it must agree with ``canonical_frontier_pointer.json``'s ``effective_frontier``
       at the pointer's own display precision -- so if the leaderboard moves, this
       function REFUSES instead of quietly ranking against a superseded bar.

    ``rate_denominator_bytes`` is required, not defaulted: pass the value read from OUR
    receipt so the comparison is on one denominator (Catalog #812).

    ``frontier_pointer_path`` defaults to the repo pointer located from this file. To skip
    check 2 you must say so EXPLICITLY with ``SKIP_POINTER_CROSSCHECK``; there is no way to
    skip it by accident, which was the round-2 defect.
    """
    if frontier_pointer_path is None:
        frontier_pointer_path = _default_frontier_pointer_path()
    triple = MeasuredScoreTriple(
        d_seg=PR130_FLOOR_D_SEG,
        d_pose=PR130_FLOOR_D_POSE,
        archive_bytes=PR130_FLOOR_ARCHIVE_BYTES,
        rate_denominator_bytes=rate_denominator_bytes,
        source_artifact=PR130_FLOOR_SOURCE,
        axis_tag="[contest-CUDA]",
    )
    recomputed = triple.total
    if abs(recomputed - PR130_FLOOR_PUBLISHED_TOTAL) > 5e-7:
        raise ValueError(
            f"PR130 floor recomputes to {recomputed:.7f} but the published row is "
            f"{PR130_FLOOR_PUBLISHED_TOTAL}; the constants no longer reproduce their own "
            "source and must not anchor a gap"
        )
    if frontier_pointer_path is not SKIP_POINTER_CROSSCHECK:
        pointer_score = _effective_frontier_score(frontier_pointer_path)
        if pointer_score is not None and abs(recomputed - pointer_score) > 5e-4:
            raise ValueError(
                f"PR130 floor {recomputed:.6f} disagrees with the canonical pointer's "
                f"effective_frontier {pointer_score}; the bar has moved and this floor is "
                "superseded -- refusing to rank against it"
            )
    return triple


def _effective_frontier_score(path: str | Path) -> float | None:
    """Read ``effective_frontier.score`` from the canonical pointer, or None if absent.

    Returns None only when the pointer file itself is missing -- and that is WARNED, not
    silent, because "the check did not run" and "the check passed" must never look alike.
    A MALFORMED pointer raises outright.
    """
    p = Path(path)
    if not p.exists():
        warnings.warn(
            f"canonical frontier pointer not found at {p}; the PR130 floor's bar "
            "cross-check DID NOT RUN. This is not a pass -- supply the pointer path or "
            "pass SKIP_POINTER_CROSSCHECK to say you meant to skip it",
            stacklevel=3,
        )
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    frontier = obj.get("effective_frontier")
    if not isinstance(frontier, dict) or "score" not in frontier:
        raise ValueError(f"{p}: effective_frontier.score missing -- pointer is malformed")
    return float(frontier["score"])


@dataclass(frozen=True)
class LiveOperatingPoint:
    """The ONE answer to 'what is live, what is the bar, what is the gap, what is 1%'.

    Carries its own scope denominator.  An empty or partial scan is VACUOUS, never a
    pass: ``receipts_scanned`` and ``receipts_parsed`` are reported so a reader can see
    the population the "best" was selected from rather than inferring it.
    """

    decomposition: GapDecomposition
    best_label: str
    receipts_scanned: int
    receipts_parsed: int
    all_totals: tuple[tuple[str, float], ...]

    @property
    def total_gap(self) -> float:
        return self.decomposition.total_gap

    def summary(self) -> dict[str, object]:
        d = self.decomposition
        out: dict[str, object] = {
            "best_label": self.best_label,
            "best_S": d.ours.total,
            "floor_S": d.floor.total,
            "total_gap": d.total_gap,
            "per_axis_gap": d.per_axis(),
            "shares": d.shares(),
            "rank_by_gap": d.rank_by_gap(),
            "bytes_per_percent_of_gap": d.bytes_per_percent_of_gap(),
            "receipts_scanned": self.receipts_scanned,
            "receipts_parsed": self.receipts_parsed,
        }
        warning = d.cross_axis_warning()
        if warning is not None:
            out["cross_axis_warning"] = warning
        return out


def live_operating_point(
    receipt_paths: Iterable[str | Path],
    *,
    axis_tag: str,
    floor: MeasuredScoreTriple | None = None,
    required_samples: int | None = 600,
) -> LiveOperatingPoint:
    """Select the live best from evaluator receipts and decompose its gap to the bar.

    REFUSES on an empty scope.  Seven silent instruments in one day emitted the same
    symbol for "clean full scope" and "nothing was examined"; an operating point derived
    from zero receipts is the most dangerous form of that, because every downstream
    ranking would divide by it.
    """
    paths: Sequence[Path] = [Path(p) for p in receipt_paths]
    if not paths:
        raise ValueError(
            "live_operating_point: EMPTY SCOPE -- zero receipts supplied. An empty scope "
            "is vacuous, never a pass; supply receipts or handle the absence explicitly"
        )
    triples: list[tuple[str, MeasuredScoreTriple]] = []
    for p in paths:
        triple = triple_from_evaluator_report(
            p, axis_tag=axis_tag, required_samples=required_samples
        )
        triples.append((p.parent.name or p.name, triple))

    best_label, best = min(triples, key=lambda item: item[1].total)
    resolved_floor = floor or demonstrated_floor_pr130(best.rate_denominator_bytes)
    return LiveOperatingPoint(
        decomposition=GapDecomposition(ours=best, floor=resolved_floor),
        best_label=best_label,
        receipts_scanned=len(paths),
        receipts_parsed=len(triples),
        all_totals=tuple(sorted(((lbl, t.total) for lbl, t in triples), key=lambda kv: kv[1])),
    )
