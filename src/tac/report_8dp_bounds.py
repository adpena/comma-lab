"""Canonical report-8dp error bounds for scores and for DELTAS between scores.

WHY THIS MODULE EXISTS (rv13 F3 + F9, and round-12 F1 before them).

The contest harness prints its distortion components rounded to 8 decimal
places. Every score recomputed from those printed components therefore carries a
worst-case absolute error bound, and ``experiments/contest_auth_eval.py`` already
PUBLISHES it on every receipt::

    report_8dp_score_worst_case_abs_error_bound
    report_8dp_pose_score_worst_case_abs_error_bound
    report_8dp_seg_score_worst_case_abs_error_bound
    report_component_rounding_abs_bound

Three separate defects landed anyway, all from the same cause -- the bound was
retyped into seal/memo prose by hand instead of being computed:

* **round-12 F1, and again in ck2 (rv13 F2).** A *delta* between two rows was
  divided by ONE row's bound. Bounds ADD for a delta, so the stated margin was
  exactly 2.00x too large. The cure landed in four files and was not carried
  into the file written hours later.
* **rv13 F3.** to1's seal stated a two-row total of ``6.672304e-06`` and then
  listed addends ``5.000000e-07 + 2.836152e-06`` -- which sum to *half* of it.
  Those are the AXIS addends of ONE row; the total is the TWO-ROW sum. A reader
  auditing by addition gets half. The same seal called the two rows' bounds
  "unequal per row" when ``d_pose`` was identical in both, so they were exactly
  equal.
* **rv13 F9.** The pose bound was re-derived from the ROUNDED ``d_pose`` using
  the LINEARIZED form ``5/sqrt(10*d_pose) * eps``, giving ``2.836152e-06``
  against the harness's published ``2.836608e-06`` -- a disagreement in the 4th
  significant figure with the campaign's own receipt.

So this module does three things and refuses to do a fourth:

1. **Prefers the published field.** The receipt is the authority. Derivation is
   the fallback, never the default.
2. **Derives with the EXACT endpoint form**, not the linearization. The
   linearization is a first-order approximation of a concave function; the
   producer takes the max over both rounding endpoints, and only the exact form
   reproduces the published digits (verified: 2.836608391523776e-06 on the live
   to1 receipt, exact match).
3. **Makes a delta bound structurally two-row.** ``DeltaBound`` cannot be built
   from one row, and it CHECKS that its stated addends sum to its stated total
   before it will render a sentence. F3 is unrepresentable here.

It refuses to accept a hand-typed bound. There is deliberately no
``bound=`` parameter anywhere in this module.

Note on conservatism: this bound is worst-case over the printing, not a claim
about the underlying measurement. When two rows share a bit-identical decoded
state their distortion legs cancel EXACTLY and the true bound on the distortion
delta is zero. Quoting the conservative bound understates the margin, which is
the safe direction -- but say which one is being quoted.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_COMPONENT_ROUNDING_ABS_BOUND",
    "DeltaBound",
    "RowBound",
    "delta_bound",
    "derive_pose_score_bound",
    "derive_seg_score_bound",
    "extract_auth_eval_components",
    "row_bound_from_result",
]

# 8 decimal places of printed component -> half-ulp of 1e-8.
DEFAULT_COMPONENT_ROUNDING_ABS_BOUND = 5e-9

_POSE_WEIGHT = 10.0  # score_pose = sqrt(10 * d_pose)
_SEG_WEIGHT = 100.0  # score_seg  = 100 * d_seg


class BoundContractError(ValueError):
    """A bound was requested that cannot be computed or does not self-check."""


def _is_number(value: Any) -> bool:
    """Numeric and NOT a bool. ``bool`` subclasses ``int``, so ``True`` would
    otherwise pass every numeric check in this module as 1.0."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def derive_pose_score_bound(
    d_pose: float,
    *,
    component_rounding_abs_bound: float = DEFAULT_COMPONENT_ROUNDING_ABS_BOUND,
) -> float:
    """Worst-case abs error on ``sqrt(10*d_pose)`` from 8dp component rounding.

    EXACT endpoint form -- the max over both rounding endpoints, matching
    ``experiments/contest_auth_eval.py`` digit for digit. The linearized
    ``5/sqrt(10*d_pose)*eps`` is NOT used: it disagrees with the published field
    in the 4th significant figure (rv13 F9), because ``sqrt`` is concave and the
    lower endpoint moves further than the upper one.

    The bound GROWS as ``d_pose`` falls -- the derivative ``5/sqrt(10*d_pose)``
    diverges at 0 -- so a better pose row carries a *wider* printing bound. That
    is counter-intuitive and is exactly why it must be computed, not eyeballed.
    """
    if not isinstance(d_pose, (int, float)) or isinstance(d_pose, bool):
        raise BoundContractError(f"d_pose must be numeric, got {d_pose!r}")
    if d_pose < 0:
        raise BoundContractError(f"d_pose must be non-negative, got {d_pose!r}")
    eps = float(component_rounding_abs_bound)
    centre = math.sqrt(_POSE_WEIGHT * d_pose)
    lower = math.sqrt(_POSE_WEIGHT * max(d_pose - eps, 0.0))
    upper = math.sqrt(_POSE_WEIGHT * (d_pose + eps))
    return max(abs(lower - centre), abs(upper - centre))


def derive_seg_score_bound(
    *,
    component_rounding_abs_bound: float = DEFAULT_COMPONENT_ROUNDING_ABS_BOUND,
) -> float:
    """Worst-case abs error on ``100*d_seg``. Linear, so it does not depend on d_seg."""
    return _SEG_WEIGHT * float(component_rounding_abs_bound)


def extract_auth_eval_components(result: Any) -> dict[str, Any]:
    """Find the auth-eval component block inside whatever shape was handed over.

    A live ``MODAL_REMOTE_RESULT.json`` is a SUMMARY wrapper: the published
    bounds are not at top level, they sit in
    ``artifacts["contest_auth_eval.json"]`` -- and that value is stored as the
    ``repr`` of a bytes object, so it needs ``ast.literal_eval`` then ``decode``
    then ``json.loads`` to open. Encapsulated here precisely so the next seal
    writer does not hand-decode it (or, likelier, give up and retype the number,
    which is how F9 happened).

    Accepts: the inner block itself, the wrapper, or a JSON/bytes-repr string.
    Returns the block with the component fields, or ``{}``.
    """
    if isinstance(result, (bytes, bytearray)):
        try:
            result = json.loads(bytes(result).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
    if isinstance(result, str):
        text = result.strip()
        if text.startswith(("b'", 'b"')):
            try:
                text = ast.literal_eval(text).decode("utf-8")
            except (ValueError, SyntaxError, AttributeError, UnicodeDecodeError):
                return {}
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if not isinstance(result, dict):
        return {}
    if "avg_posenet_dist" in result and (
        "report_8dp_pose_score_worst_case_abs_error_bound" in result
        or "report_component_rounding_abs_bound" in result
    ):
        return result
    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        inner = extract_auth_eval_components(artifacts.get("contest_auth_eval.json"))
        if inner:
            return inner
    # A wrapper with components but no published bounds is still usable: the
    # bounds can be DERIVED from d_pose. Return it so the caller can say so.
    if "avg_posenet_dist" in result:
        return result
    return {}


@dataclass(frozen=True)
class RowBound:
    """The report-8dp bound of ONE score row, split into its two axis addends."""

    seg: float
    pose: float
    d_pose: float
    source: str  # "published" | "derived"
    label: str = ""

    @property
    def total(self) -> float:
        return self.seg + self.pose

    def describe(self) -> str:
        name = f"{self.label} " if self.label else ""
        return (
            f"{name}row bound {self.total:.6e} "
            f"(seg {self.seg:.6e} + pose {self.pose:.6e}; {self.source})"
        )


def row_bound_from_result(result: Any, *, label: str = "") -> RowBound:
    """The bound of one row: PUBLISHED if the receipt carries it, else DERIVED.

    Raises rather than guessing when neither a published bound nor a ``d_pose``
    is available. A bound nobody can compute must not become a plausible number.
    """
    block = extract_auth_eval_components(result)
    if not block:
        raise BoundContractError(
            "no auth-eval component block found; expected either the inner "
            "contest_auth_eval result or a MODAL_REMOTE_RESULT wrapper carrying it"
        )
    d_pose = block.get("avg_posenet_dist")
    published_pose = block.get("report_8dp_pose_score_worst_case_abs_error_bound")
    published_seg = block.get("report_8dp_seg_score_worst_case_abs_error_bound")
    # ``bool`` is a subclass of ``int``: a stray ``True`` would otherwise be
    # accepted as a published bound of 1.0 and silently dominate every margin.
    if _is_number(published_pose) and _is_number(published_seg):
        return RowBound(
            seg=float(published_seg),
            pose=float(published_pose),
            d_pose=float(d_pose) if _is_number(d_pose) else float("nan"),
            source="published",
            label=label,
        )
    if not _is_number(d_pose):
        raise BoundContractError(
            "receipt publishes no report_8dp_*_bound and carries no numeric "
            "avg_posenet_dist; refusing to invent a bound"
        )
    eps = block.get("report_component_rounding_abs_bound")
    eps = float(eps) if _is_number(eps) else DEFAULT_COMPONENT_ROUNDING_ABS_BOUND
    return RowBound(
        seg=derive_seg_score_bound(component_rounding_abs_bound=eps),
        pose=derive_pose_score_bound(float(d_pose), component_rounding_abs_bound=eps),
        d_pose=float(d_pose),
        source="derived",
        label=label,
    )


@dataclass(frozen=True)
class DeltaBound:
    """The bound on a DELTA between two rows. Structurally two-row.

    There is no single-row constructor. That is the point: rv13 F2/F3 and
    round-12 F1 are all the same mistake -- pricing a delta against one row's
    bound, or listing one row's axis addends under a two-row total.
    """

    base: RowBound
    candidate: RowBound

    @property
    def total(self) -> float:
        return self.base.total + self.candidate.total

    @property
    def rows_are_equal(self) -> bool:
        """Whether the two rows' bounds coincide (they do when d_pose matches)."""
        return math.isclose(self.base.total, self.candidate.total, rel_tol=1e-12, abs_tol=0.0)

    def self_check(self) -> None:
        """Refuse to render unless the stated addends sum to the stated total.

        This is rv13 F3 made unrepresentable: commit ``6e976eeafd`` is titled
        *"the sa3 summed bound must equal its stated addends"*, and two days
        later a seal shipped addends summing to half its total.
        """
        stated = self.base.total + self.candidate.total
        if not math.isclose(stated, self.total, rel_tol=1e-12, abs_tol=0.0):
            raise BoundContractError(
                f"addends {self.base.total!r} + {self.candidate.total!r} do not sum to "
                f"the stated total {self.total!r}"
            )

    def multiple_of(self, net_ds: float) -> float:
        """How many bounds the delta clears. ``abs`` because sign is carried separately."""
        self.self_check()
        if self.total <= 0:
            raise BoundContractError("bound total must be positive to form a multiple")
        return abs(float(net_ds)) / self.total

    def describe(self, net_ds: float | None = None) -> str:
        """The canonical sentence, with BOTH rows' addends shown and summed.

        Every number here is computed. Nothing in this string is hand-typed,
        which is the whole remedy: the three defects this module cures were all
        prose that had drifted from the arithmetic it claimed to report.
        """
        self.self_check()
        equality = "equal" if self.rows_are_equal else "unequal"
        parts = [
            f"report-8dp bound on the DELTA = {self.total:.6e} "
            f"(bounds ADD for deltas: base {self.base.total:.6e} + candidate "
            f"{self.candidate.total:.6e}; the two rows' bounds are {equality} here)",
            f"  base      {self.base.describe()}",
            f"  candidate {self.candidate.describe()}",
        ]
        if net_ds is not None:
            parts.insert(
                0,
                f"net dS {float(net_ds):.6e} is {self.multiple_of(net_ds):.2f}x the "
                f"SUMMED two-row report-8dp error bound {self.total:.6e}",
            )
        return "\n".join(parts)


def delta_bound(
    base_result: Any,
    candidate_result: Any,
    *,
    base_label: str = "base",
    candidate_label: str = "candidate",
) -> DeltaBound:
    """Build the two-row delta bound from two receipts. Self-checked on render."""
    bound = DeltaBound(
        base=row_bound_from_result(base_result, label=base_label),
        candidate=row_bound_from_result(candidate_result, label=candidate_label),
    )
    bound.self_check()
    return bound
