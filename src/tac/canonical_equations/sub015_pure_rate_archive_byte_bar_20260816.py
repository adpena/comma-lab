# SPDX-License-Identifier: MIT
"""Canonical equation: the sub-0.15 admission bar for PURE-RATE work, expressed as a
DERIVATION off the live frontier rather than as a literal byte count.

Why this law exists
-------------------
Admission bars keep going stale, and they go stale SILENTLY. Three independent audit
arms found the same defect class on 2026-08-16 alone (``ddm_hv2`` rate, ``ddm_gx1``
rate, ``ddm_pv1`` pose). The mechanism is always identical: a fire order copied a
SNAPSHOT of the frontier -- an absolute byte count, or a delta off one -- and the
pointer then moved underneath it.

The measured cost of one instance: four arms carried ``archive < 186,269 B`` while the
live shipping archive was 182,759 B. A candidate landing exactly at that bar PASSES
while scoring **+0.002337165 WORSE** than what we already ship -- 233.7x the 1e-5
naming bar, in the anti-conservative direction, with no alarm.

The cure is not to re-stamp documents. It is to change what a bar QUOTES. For work
that changes ONLY the rate term, the sub-0.15 target is a byte threshold that follows
from the live pointer by exact arithmetic:

    S            = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes / D
    distortion   = S_base - 25*B_base / D                  (the seg+pose legs, S units)
    bar_bytes    = (0.15 - distortion) * D / 25
    required_cut = B_base - bar_bytes

Feed it any base on the frozen-distortion lineage and it returns the same threshold.
Feed it a base with different distortion and it correctly returns a different one.
That is the whole point: the DERIVATION does not go stale, only literals do.

THE CAVEAT, which must always travel with the number (this is not decoration)
--------------------------------------------------------------------------
``bar_bytes`` is valid **only while the candidate leaves d_seg and d_pose exactly
where the base has them.** The instant a candidate changes distortion -- which is what
the qs / re / wd families do by construction -- the distortion leg moves and this
threshold is wrong. Distortion-changing work must quote the canonical pointer and
re-measure both legs. A single number cannot cover both regimes, and claiming it could
would be the same error class this law exists to retire.

MEASURED, and it corrects the memo that motivated the law
---------------------------------------------------------
``ddm_fb1`` reported the bar as "re-derived off all four bases; identical to four
decimal places from every one of them", naming the lineage
``cp135 -> MC36 -> e480b v2 -> hv1``. Re-derived here at 44-digit precision:

* MC36, e480b v2 and hv1 agree to **7.23e-11 B** -- decode-identical distortion, so
  one shared bar of **168,345.5977 B**;
* **cp135 does NOT.** Its bar is **168,297.5395 B**, lower by **48.058 B**, because
  the cp135 -> MC36 step was a DISTORTION move: the MC36 verdict measures its seg leg
  at -37 flips = -3.136529e-5 S, so cp135's seg+pose sits +3.2e-5 above the frontier's.

So the identity holds over the frozen-distortion SUB-lineage, not over all four bases,
and the counter-example sits inside the very lineage that was cited. That is a sharper
result than the original claim, not a weaker one: the pure-rate caveat is not a
footnote to the invariant, it is the condition that generates it, and cp135 is the
worked proof. Both figures are kept below so nobody can re-broaden the claim.

Axis: exact arithmetic over ``[contest-CUDA T4 n600]`` receipts. The law itself carries
no measurement authority and is never a score claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "sub015_pure_rate_archive_byte_bar_v1"
AXIS = (
    "[DERIVED exact arithmetic over contest-CUDA T4 n600 receipts] — the law carries no "
    "measurement authority and is never a score claim"
)
SOURCE_MEMO = ".omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md"
POINTER_PATH = ".omx/state/canonical_frontier_pointer.json"

#: ``upstream/evaluate.py:63`` rate denominator. Catalog #812 records that evaluate.py
#: sums ``rglob('*')`` over ``videos/`` rather than hardcoding this, so a videos/ change
#: moves it. Never treat it as immortal; it is an input, not a constant of nature.
RATE_DENOMINATOR_BYTES = 37_545_489
SUB015_TARGET_S = 0.15

#: The live frontier at registration: hv1 ep0634, contest-CUDA T4 n600.
FRONTIER_ARCHIVE_SHA256 = (
    "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
)
FRONTIER_S = 0.15959729295498598
FRONTIER_ARCHIVE_BYTES = 182_759

#: Derived at 44-digit precision from the four bases (see the module docstring).
#: Kept as evidence of the SCOPE of the identity, not as a value to copy into a charter
#: -- copying either of these is the exact defect this law retires.
BAR_BYTES_FROZEN_DISTORTION_LINEAGE = 168_345.5977
BAR_BYTES_CP135_SUPERSEDED_DISTORTION = 168_297.5395
FROZEN_LINEAGE_BAR_SPREAD_BYTES = 7.23e-11
CP135_BAR_OFFSET_BYTES = -48.058


def distortion_leg(base_score: float, base_archive_bytes: int, *, denominator: int = RATE_DENOMINATOR_BYTES) -> float:
    """``100*d_seg + sqrt(10*d_pose)`` in S units, recovered from a complete-S row.

    Exact by subtraction, so it does not need the component breakdown -- which matters,
    because published seg legs are often rounded to 6 significant figures while the
    composed S is carried to 17.
    """
    if denominator <= 0:
        raise ValueError("rate denominator must be positive")
    if base_archive_bytes < 0:
        raise ValueError("archive bytes must be non-negative")
    return base_score - 25.0 * base_archive_bytes / denominator


def pure_rate_byte_bar(
    base_score: float,
    base_archive_bytes: int,
    *,
    target_score: float = SUB015_TARGET_S,
    denominator: int = RATE_DENOMINATOR_BYTES,
) -> float:
    """Archive bytes at which a PURE-RATE candidate off this base reaches ``target_score``.

    PURE-RATE ONLY. Valid only while the candidate holds d_seg and d_pose exactly where
    the base has them. Any distortion change invalidates it -- re-measure instead.

    A NEGATIVE return means the base's distortion alone already exceeds the target: no
    amount of rate work reaches it, and the honest read is that the distortion legs are
    the binding constraint. That is the correct arithmetic, not an error.
    """
    return (target_score - distortion_leg(base_score, base_archive_bytes, denominator=denominator)) * denominator / 25.0


def required_cut_bytes(
    base_score: float,
    base_archive_bytes: int,
    *,
    target_score: float = SUB015_TARGET_S,
    denominator: int = RATE_DENOMINATOR_BYTES,
) -> float:
    """Bytes a pure-rate candidate must remove from ``base_archive_bytes`` to hit target.

    Non-positive means the base already clears the target on rate alone.
    """
    bar = pure_rate_byte_bar(
        base_score, base_archive_bytes, target_score=target_score, denominator=denominator
    )
    return base_archive_bytes - bar


def _load_pointer(repo_root: Path | None = None) -> dict:
    root = repo_root or Path(__file__).resolve().parents[3]
    return json.loads((root / POINTER_PATH).read_text(encoding="utf-8"))


def pure_rate_byte_bar_from_pointer(
    *,
    target_score: float = SUB015_TARGET_S,
    repo_root: Path | None = None,
) -> dict:
    """THE stale-proof consumption path: read the live pointer, derive the bar.

    This is what a fire order should call instead of copying a byte count into its own
    text. The returned dict carries the base it used, so a receipt can show WHICH
    frontier the bar was derived from rather than asserting a bare number.

    Fails closed (KeyError / FileNotFoundError / JSONDecodeError propagate) when the
    pointer is absent or malformed. A bar that silently falls back to a hardcoded
    default is the defect wearing the cure's clothes.
    """
    pointer = _load_pointer(repo_root)
    frontier = pointer["effective_frontier"]
    source = frontier["source"]
    if source not in pointer:
        # effective_frontier can select the upstream leaderboard, whose archive we do
        # not hold. There is no byte bar to derive off someone else's archive, and
        # inventing one would be exactly the false-authority class.
        raise KeyError(
            f"effective_frontier.source={source!r} has no local archive record in "
            f"{POINTER_PATH}; a pure-rate byte bar can only be derived off an archive "
            "we hold"
        )
    base_bytes = int(pointer[source]["extra"]["archive_bytes"])
    base_score = float(frontier["score"])
    bar = pure_rate_byte_bar(base_score, base_bytes, target_score=target_score)
    return {
        "bar_bytes": bar,
        "required_cut_bytes": base_bytes - bar,
        "target_score": target_score,
        "base_score": base_score,
        "base_archive_bytes": base_bytes,
        "base_archive_sha256": frontier.get("archive_sha256"),
        "base_axis": frontier.get("evidence_grade"),
        "base_measured_at_utc": frontier.get("measured_at_utc"),
        "distortion_leg": distortion_leg(base_score, base_bytes),
        "valid_only_for": (
            "PURE-RATE candidates that hold d_seg and d_pose exactly where the base has "
            "them; any distortion change invalidates this bar — re-measure both legs"
        ),
        "score_claim": False,
    }


def build_sub015_pure_rate_archive_byte_bar_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        SOURCE_MEMO,
        reactivation_criteria=(
            "re-derive whenever the canonical frontier pointer moves on a DISTORTION "
            "axis, or whenever upstream/evaluate.py's rate denominator changes; a "
            "rate-only pointer move needs no recalibration, which is the point of the law"
        ),
        measurement_axis=AXIS,
        hardware_substrate="linux_x86_64_t4",
        captured_at_utc="2026-08-16T18:00:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="pure_rate_bar_is_frozen_over_the_frozen_distortion_lineage_20260816",
            measurement_utc="2026-08-16T18:00:00Z",
            inputs={
                "bases": {
                    "cp135": {"S": 0.16195513827824176, "archive_bytes": 186252},
                    "MC36_variant_C": {"S": 0.1619344578804448, "archive_bytes": 186269},
                    "e480b_v2": {"S": 0.1600920261571558, "archive_bytes": 183502},
                    "hv1_ep0634": {"S": FRONTIER_S, "archive_bytes": FRONTIER_ARCHIVE_BYTES},
                },
                "axis": "[contest-CUDA T4 n600]",
                "target_score": SUB015_TARGET_S,
                "denominator": RATE_DENOMINATOR_BYTES,
                "precision": "44 significant digits (decimal.Decimal)",
            },
            predicted_output={
                "claim_under_test": (
                    "ddm_fb1: the sub-0.15 byte bar is identical off all four bases "
                    "(cp135 -> MC36 -> e480b v2 -> hv1) to four decimal places"
                )
            },
            empirical_output={
                "frozen_distortion_lineage": {
                    "members": ["MC36_variant_C", "e480b_v2", "hv1_ep0634"],
                    "bar_bytes": BAR_BYTES_FROZEN_DISTORTION_LINEAGE,
                    "max_spread_bytes": FROZEN_LINEAGE_BAR_SPREAD_BYTES,
                    "verdict": "CONFIRMED — decode-identical distortion, one shared bar",
                },
                "cp135_counter_example": {
                    "bar_bytes": BAR_BYTES_CP135_SUPERSEDED_DISTORTION,
                    "offset_vs_live_bar_bytes": CP135_BAR_OFFSET_BYTES,
                    "distortion_leg_offset_S": 3.2e-5,
                    "mechanism": (
                        "cp135 -> MC36 was a DISTORTION move: the MC36 dual-axis T4 "
                        "verdict measures its seg leg at -37 flips = -3.136529e-5 S, so "
                        "cp135's seg+pose sits above the frontier's and its pure-rate bar "
                        "is 48.058 B lower"
                    ),
                    "verdict": (
                        "REFUTED for cp135 — the four-base claim is FALSE at its own "
                        "first base; the identity is scoped to the frozen-distortion "
                        "sub-lineage, and the caveat is the condition that generates it"
                    ),
                },
                "noted_not_resolved": (
                    "the widely-quoted MC36 S 0.1619344578804448 and the value implied by "
                    "cp135 plus MC36's own measured net ΔS (-1.99799e-5) disagree by "
                    "7.005e-7 S — 0.07x the 1e-5 naming bar, so it moves no verdict here, "
                    "but a tight MC36-based seg claim must go to the primary T4 receipt"
                ),
            },
            residual=0.0,
            source_artifact=SOURCE_MEMO,
            measurement_method=(
                "exact decimal arithmetic on published complete-S rows and archive byte "
                "counts; distortion leg recovered by subtracting the rate term, so no "
                "rounded component figure enters the derivation. No scorer forward, no "
                "decode, no dispatch."
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Sub-0.15 pure-rate archive byte bar, derived from the live pointer rather "
            "than copied from a snapshot"
        ),
        one_line_summary=(
            "bar_bytes = (0.15 - (S_base - 25*B_base/D)) * D / 25. Stale-proof for "
            "PURE-RATE work; invalid the moment a candidate moves d_seg or d_pose."
        ),
        latex_form=(
            r"B^{\ast}=\frac{D}{25}\left(0.15-\left(S_{base}-\frac{25\,B_{base}}{D}\right)\right),"
            r"\quad D=37{,}545{,}489"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.sub015_pure_rate_archive_byte_bar_20260816"
            ":pure_rate_byte_bar"
        ),
        domain_of_validity={
            "axis": AXIS,
            "research_only": False,
            "applies_to": (
                "candidates whose only changed score term is rate — a recode, a repack, "
                "a section shrink that is proven distortion-neutral (e.g. by decoded-"
                "token identity plus raw-output byte identity, as hv1 was admitted)"
            ),
            "does_not_apply_to": (
                "any candidate that moves d_seg or d_pose. For those the distortion leg "
                "is no longer the base's, so this threshold is simply wrong; quote "
                f"{POINTER_PATH} and re-measure both legs. The cp135 anchor above is the "
                "worked counter-example: one distortion move shifted the bar 48.058 B."
            ),
            "denominator_is_an_input": (
                "D is read from upstream/evaluate.py's videos/ rglob sum (Catalog #812); "
                "a videos/ change moves the bar and must be passed explicitly"
            ),
        },
        units_in={
            "base_score": "S (contest score, complete-S row)",
            "base_archive_bytes": "bytes of the base archive.zip",
            "target_score": "S",
        },
        units_out={"bar_bytes": "bytes", "required_cut_bytes": "bytes"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"registration": 0.0},
        last_calibration_utc="2026-08-16T18:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "every pure-rate fire order and admission bar: call "
            "pure_rate_byte_bar_from_pointer() instead of copying a byte count, so the "
            "bar cannot go stale when the pointer moves on rate",
            "tools/codex_arm_queue.py retraction reasons — the derivation a retracted "
            "stale-bar row should be re-filed against",
            "rate-route rungs (mp2, rfo2, wd2 successors) that quote a required cut",
        ),
        canonical_producers=(
            SOURCE_MEMO,
            ".omx/research/ddm_mc36_dual_axis_t4_verdict_20260814.md (the -37 flip seg "
            "leg that makes cp135 a distortion counter-example)",
            ".omx/research/ddm_hv1_harvest_compose_ep508_20260815.md (the frozen "
            "distortion legs seg 0.029611 / pose 0.008294576541331089)",
        ),
        provenance=provenance,
    )


def populate_sub015_pure_rate_archive_byte_bar_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_sub015_pure_rate_archive_byte_bar_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "AXIS",
    "BAR_BYTES_CP135_SUPERSEDED_DISTORTION",
    "BAR_BYTES_FROZEN_DISTORTION_LINEAGE",
    "CP135_BAR_OFFSET_BYTES",
    "EQUATION_ID",
    "FRONTIER_ARCHIVE_BYTES",
    "FRONTIER_S",
    "FROZEN_LINEAGE_BAR_SPREAD_BYTES",
    "POINTER_PATH",
    "RATE_DENOMINATOR_BYTES",
    "SUB015_TARGET_S",
    "build_sub015_pure_rate_archive_byte_bar_v1",
    "distortion_leg",
    "populate_sub015_pure_rate_archive_byte_bar_equation",
    "pure_rate_byte_bar",
    "pure_rate_byte_bar_from_pointer",
    "required_cut_bytes",
]
