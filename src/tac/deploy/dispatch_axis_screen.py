"""Refuse a paid dual-axis row when every distortion axis is an assertion.

THE MEASURED DEFECT (2026-08-16, ``ddm_ps1u`` r2)
------------------------------------------------
The ps1u candidate existed to move **pose**: a 626 B ``P1D1`` frame-0 carrier
delta over the 60 top-mass pairs, costing a known **+588 B** of rate.  Its
sealed request and its ``POSE_SCREEN_RESULT.json`` carried, verbatim::

    "local_pose_delta": 0.0,
    "pose_unmeasured": true,
    "role": "... pose is UNMEASURED locally by the worker placeholder law"

and its seg leg was asserted, not measured::

    "seg_leg_measured": false,
    "re1t_run_id": "NONE_ps1u_seg_asserted_decode_identical",
    "seg_delta_s_exact_t4_field": 0.0

So both distortion legs were assertions and only the rate leg — a **cost** —
was real.  The row fired, cost ~$0.16, and was REFUSED at +1.686e-02 S because
the unscreened pose leg came back **8.93x worse**
(``.omx/research/ddm_ps1u_r2_dual_axis_pose_verdict_20260816.md``).

THE GENUS
---------
A numeric **placeholder** sat next to a flag saying the value was never
measured, and a consumer read the value while nothing read the flag.  The
worker's arithmetic folds ``pose_delta_s_placeholder = 0.0`` into its mixed sum
(``experiments/ddm_re1t_modal_t4_sign_gate.py:664-676``), so a paid row whose
distortion axes are all placeholders is a coin flip bought with a known cost.

WHY A NUMERIC SCAN CANNOT DECIDE THIS
-------------------------------------
ps1u's evidence payload *does* contain a finite non-zero pose float:
``pre_registered_admission.required_cuda_dpose_after = 6.251198917870592e-06``.
That is a **target**, not a measurement of the candidate.  A target and a
measurement are indistinguishable by looking at the number — which is the same
genus one level up.  So the pose screen is recognised only through an explicit
declaration or an explicit, corpus-measured key vocabulary, never by pattern.

THE RULE
--------
A sealed dual-axis request may not dispatch unless **at least one distortion
axis (seg or pose) carries a local measurement**.  Rate is not a distortion
axis and never satisfies the rule: rate is what the row *spends*.

Resolution ladder — POSITIVE evidence only, first hit wins per axis.  Every rung
answers "what proves a measurement happened?", never "what fails to deny it":
an absent key is never a pass, so deleting a field can never open the gate.

1. ``local_axis_screen`` — the canonical forward declaration new arms must use.
2. request-level pose: ``pose_unmeasured is False``, or a finite non-zero
   ``local_pose_delta`` (LEGACY; the shape ``ddm_qs2``/``ddm_qs5`` already use).
3. request-level seg: ``seg_leg_provenance`` naming a real prior seg run
   (LEGACY; the shape ``ddm_re1`` uses).
4. evidence-payload pose under the corpus-measured key vocabulary
   (LEGACY; the shape ``ddm_qs1``/``ddm_qs4``/``ddm_pk3``/``ddm_mc36`` use).

Nothing hits and no substantive waiver present => refuse, fail-closed.

MEASURED POPULATION: 1 of the 9 sealed ``ddm_qs1_t4_dual_axis_request.v1``
requests retained on the SSD tier trips this rule — ps1u.  The other 8 each
carry a real local distortion measurement somewhere in the ladder.  The gate is
narrow on purpose: it refuses the state where *nothing measured* backs a paid
row, not every honest ``*_unmeasured`` status label in the corpus.

AXIS ``[apparatus — no score, no dispatch of its own]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "CANONICAL_POSE_SCREEN_KEYS",
    "AxisScreenCensus",
    "UnscreenedAxisDispatchError",
    "assert_distortion_axis_locally_screened",
    "census_distortion_axis_screen",
]

WAIVER_KEY: Final = "unscreened_axis_dispatch_waiver"

#: Substantive-rationale rejection, mirroring the repo's same-line waiver
#: discipline (Catalog #287) in the JSON domain this gate actually reads.
#: The gate reads a hash-sealed JSON request, not source text, so the escape
#: is a JSON field rather than a ``# ..._OK:<rationale>`` comment.
_PLACEHOLDER_RATIONALES: Final = frozenset(
    {
        "",
        "-",
        "<rationale>",
        "<reason>",
        "<value>",
        "n/a",
        "na",
        "none",
        "ok",
        "pending",
        "pending_ratification",
        "placeholder",
        "tbd",
        "todo",
        "why",
        "yes",
    }
)
_MIN_RATIONALE_CHARS: Final = 24

#: Keys that declare a MEASURED local pose reading **of the candidate itself**.
#: Every entry was read out of a retained ``POSE_SCREEN_RESULT.json`` on the
#: SSD tier on 2026-08-16 — this vocabulary is measured, not guessed.  Targets
#: (``target_dpose``, ``required_cuda_dpose_after``), placeholders
#: (``local_pose_delta`` at 0.0, ``pose_delta_s_placeholder_not_measurement``)
#: and base-only readings (``base_instrument.dpose_recomputed``) are excluded
#: by construction: they are not a screen of this candidate's own pose delta.
CANONICAL_POSE_SCREEN_KEYS: Final[tuple[tuple[str, ...], ...]] = (
    ("local_pose_screen_delta_s",),  # forward canonical — new arms use this
    ("pose_delta_s",),  # ddm_qs2, ddm_qs4
    ("conservative_residual_pose_bound_s",),  # ddm_qs1
    ("local_pose_advisory", "delta_dpose"),  # ddm_qs5
    ("local_pose_advisory", "delta_pose_score_term"),  # ddm_qs5
    ("local_advisory", "delta_dpose"),  # ddm_mc36
    ("base_sample_dpose",),  # ddm_pk3
    ("model_sample_dpose",),  # ddm_pk3
    ("model_lopo_dpose",),  # ddm_pk3
)


class UnscreenedAxisDispatchError(RuntimeError):
    """Every distortion axis of a paid row was an assertion, not a measurement."""


def _is_real_number(value: Any) -> bool:
    """True for a finite int/float that is not a bool.

    ``bool`` is an ``int`` subclass in Python; a flag is never a measurement.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _is_finite_nonzero(value: Any) -> bool:
    return _is_real_number(value) and float(value) != 0.0


def _dig(payload: Any, path: Sequence[str]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _is_substantive_rationale(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if stripped.casefold() in _PLACEHOLDER_RATIONALES:
        return False
    if stripped.startswith("<") and stripped.endswith(">"):
        return False
    return len(stripped) >= _MIN_RATIONALE_CHARS


@dataclass(frozen=True)
class AxisScreenCensus:
    """Which distortion legs of a sealed request carry a local measurement."""

    pose_measured: bool
    pose_basis: str
    seg_measured: bool
    seg_basis: str
    waived: bool
    waiver_rationale: str | None

    @property
    def any_distortion_axis_measured(self) -> bool:
        return self.pose_measured or self.seg_measured

    @property
    def refused(self) -> bool:
        return not self.any_distortion_axis_measured and not self.waived

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": "tac_dispatch_axis_screen_census.v1",
            "pose_measured": self.pose_measured,
            "pose_basis": self.pose_basis,
            "seg_measured": self.seg_measured,
            "seg_basis": self.seg_basis,
            "any_distortion_axis_measured": self.any_distortion_axis_measured,
            "waived": self.waived,
            "waiver_rationale": self.waiver_rationale,
            "refused": self.refused,
        }


def _explicit_leg(request: Mapping[str, Any], axis: str) -> tuple[bool, str] | None:
    """Read the canonical forward ``local_axis_screen`` declaration for one axis.

    A declaration of ``measured: false`` is not authoritative — the ladder still
    consults the legacy rungs, because real evidence of a measurement elsewhere
    should not be overridden by an under-filled declaration.  Only a POSITIVE
    hit ends the ladder.
    """
    block = request.get("local_axis_screen")
    if not isinstance(block, Mapping):
        return None
    leg = block.get(axis)
    if not isinstance(leg, Mapping):
        return None
    measured = leg.get("measured")
    if measured is not True:
        return (False, f"local_axis_screen.{axis}.measured is not True")
    basis = leg.get("basis")
    if not _is_substantive_rationale(basis):
        raise UnscreenedAxisDispatchError(
            f"local_axis_screen.{axis} claims measured=true without a substantive "
            f"basis string (>={_MIN_RATIONALE_CHARS} chars, no placeholder)"
        )
    if not _is_real_number(leg.get("delta_s")):
        raise UnscreenedAxisDispatchError(
            f"local_axis_screen.{axis} claims measured=true without a finite "
            "numeric delta_s"
        )
    return (True, f"local_axis_screen.{axis}: {str(basis).strip()}")


def _legacy_pose_leg(
    request: Mapping[str, Any], evidence_payload: Mapping[str, Any] | None
) -> tuple[bool, str]:
    """Recognise the pre-gate shapes that DID screen pose locally."""
    # Positive evidence only.  ``is not True`` would let a successor bypass the
    # gate by simply DELETING the flag, which is the defect wearing a hat.
    if request.get("pose_unmeasured") is False:
        return (True, "legacy: request declares pose_unmeasured false")
    local = request.get("local_pose_delta")
    if _is_finite_nonzero(local):
        # ddm_qs5 shape: a real measured value parked next to a stale True flag.
        return (
            True,
            f"legacy: request local_pose_delta={local!r} is a finite non-zero "
            "reading (pose_unmeasured flag is stale and should be corrected)",
        )
    if evidence_payload is not None:
        for path in CANONICAL_POSE_SCREEN_KEYS:
            value = _dig(evidence_payload, path)
            if _is_finite_nonzero(value):
                return (
                    True,
                    f"legacy: evidence payload {'.'.join(path)}={value!r}",
                )
    return (
        False,
        "pose is the placeholder pair (local_pose_delta 0.0 / pose_unmeasured "
        "true) and no canonical local pose screen key carries a reading",
    )


def _legacy_seg_leg(request: Mapping[str, Any]) -> tuple[bool, str]:
    """Recognise the pre-gate shape that DID measure the seg leg (ddm_re1)."""
    provenance = request.get("seg_leg_provenance")
    if not isinstance(provenance, Mapping):
        return (False, "no seg_leg_provenance block")
    if provenance.get("seg_leg_measured") is True:
        return (True, "legacy: seg_leg_provenance.seg_leg_measured is True")
    run_id = provenance.get("re1t_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return (False, "seg_leg_provenance names no prior seg run")
    if run_id.strip().upper().startswith("NONE"):
        return (
            False,
            f"seg_leg_provenance.re1t_run_id={run_id!r} asserts the seg leg "
            "instead of naming a real prior seg run",
        )
    if not _is_real_number(provenance.get("seg_delta_s_exact_t4_field")):
        return (False, "seg_leg_provenance carries no finite seg_delta_s_exact_t4_field")
    return (True, f"legacy: seg leg measured by prior run {run_id.strip()!r}")


def census_distortion_axis_screen(
    request: Mapping[str, Any],
    evidence_payload: Mapping[str, Any] | None = None,
) -> AxisScreenCensus:
    """Build the per-axis local-screen census for a sealed dual-axis request.

    ``evidence_payload`` is the parsed ``POSE_SCREEN_RESULT.json`` that travels
    with the request; pass ``None`` when the transport carries no evidence file.
    """
    pose = _explicit_leg(request, "pose")
    if pose is None or not pose[0]:
        pose = _legacy_pose_leg(request, evidence_payload)
    seg = _explicit_leg(request, "seg")
    if seg is None or not seg[0]:
        seg = _legacy_seg_leg(request)

    waiver = request.get(WAIVER_KEY)
    waived = False
    rationale: str | None = None
    if isinstance(waiver, Mapping):
        candidate = waiver.get("rationale")
        if _is_substantive_rationale(candidate):
            waived = True
            rationale = str(candidate).strip()
        else:
            raise UnscreenedAxisDispatchError(
                f"{WAIVER_KEY} present with a placeholder or too-short rationale "
                f"({candidate!r}); a waiver needs >={_MIN_RATIONALE_CHARS} chars "
                "of real reason naming why this paid row may fire unscreened"
            )
    elif waiver is not None:
        raise UnscreenedAxisDispatchError(
            f"{WAIVER_KEY} must be an object carrying a substantive 'rationale'"
        )

    return AxisScreenCensus(
        pose_measured=pose[0],
        pose_basis=pose[1],
        seg_measured=seg[0],
        seg_basis=seg[1],
        waived=waived,
        waiver_rationale=rationale,
    )


def assert_distortion_axis_locally_screened(
    request: Mapping[str, Any],
    evidence_payload: Mapping[str, Any] | None = None,
) -> AxisScreenCensus:
    """Refuse the dispatch when no distortion axis carries a local measurement.

    Raises :class:`UnscreenedAxisDispatchError` on refusal; returns the census
    otherwise so the caller can record it with the dispatch receipt.
    """
    result = census_distortion_axis_screen(request, evidence_payload)
    if result.refused:
        raise UnscreenedAxisDispatchError(
            "paid dual-axis dispatch refused: every distortion axis is an "
            "assertion, so only the rate leg (a cost) is real, and the row is a "
            f"coin flip on the axis it targets. pose: {result.pose_basis}. "
            f"seg: {result.seg_basis}. Measure one distortion axis locally "
            "before spending (declare it under 'local_axis_screen'), or attach "
            f"{WAIVER_KEY} with a substantive rationale. "
            "Anchor: .omx/research/ddm_ps1u_r2_dual_axis_pose_verdict_20260816.md"
        )
    return result
