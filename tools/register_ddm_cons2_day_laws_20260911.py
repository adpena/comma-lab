#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register the 2026-09-10 arms' MEASURED laws into the canonical registries (ddm_cons2).

Two canonical equations, three canonical anti-patterns, and ONE anchor appended to an existing
equation, each carrying the measured numbers, the memo that produced them, and a NAMED consumer --
per the mh1 finding that 68.6 % of findings reach no consumer, the consumer field is not optional.

EQUATIONS (new)
  * ``lane_surprise_atlas_oracle_ladder_v1``  -- ls1's oracle ladder on the SHIPPED move-44 field
    plus ls2's charged realisation. The best RECEIVER-VISIBLE oracle is 8,364.784 B short of the
    25,899 B demand with free tables; the charged full-resolution correction returns 385.550480 B
    (1.49 %). The one rung that comes near (1,035.362 B short) uses geometry the receiver does not
    have.
  * ``token_edit_composition_subadditive_on_pair_overlap_v1`` -- sj1's compose-39: two token arms
    with ZERO cell collisions lost 6 of 80 seg repairs, all four losing pairs inside their 13-pair
    overlap. +5.086e-06 S of a +1.62e-06 miss against the naive sum; confirmed on the exact axis as
    pointer move 40.

ANCHOR APPENDED (existing equation, NOT re-registered)
  * ``rate_directed_predistortion_yield_v1`` gains the 2026-09-10 legs its 09-09 row lacks: sj1's
    ADD-leg asymmetry (a written-in surprise costs 1.279x its modelled bits, against 0.1445-0.2663
    returned on the take-out leg) and rp1's round-2 K curve (neutrality FLAT to rank 256 at 4.26 %;
    only the prize decays, 0.477 -> 0.056 bits/test, 8.5x).

ANTI-PATTERNS (new)
  * ``fitted_scalar_in_receiver_code_is_counted_content_v1``  -- pr8: a fitted SCALAR is content;
    "no per-frame fitted table" is not rule-118 compliance. Move 41's row stands, its archive does
    not qualify, the pointer retracted to move 40.
  * ``derived_listing_inside_identity_digest_v1``            -- pr14: a derived hash listing inside
    a BEHAVIOR digest re-imports the very bytes the normalizer strips. rlc4's candidate was
    raw-identical to move 42 with ZERO executable bytes changed and was refused.
  * ``inherited_timing_leg_reinherited_v1``                  -- ntb2: a decode wall-clock may be
    inherited from a measured leg exactly once; the cure for the second hop is to MEASURE, not to
    patch the contract that blocked you.

WHAT THIS SCRIPT DELIBERATELY DOES NOT REGISTER, and why, because a duplicate law is worse than a
missing one:
  * the container-break one-sample lottery (sd 34.8 B) is ALREADY the third anchor
    ``fe1_container_break_delta_is_a_one_sample_lottery_20260909`` on
    ``model_section_edit_container_break_fee_v1``;
  * the first-order-token-price law is ALREADY ``rate_directed_predistortion_yield_v1`` -- which is
    why the 09-10 legs are APPENDED to it above rather than given a second home;
  * tc2-tc4, bnd2/bnd3 and eb1/eb2 already hold ``lane_boundary_context_map_bound_v1``,
    ``boundary_segment_recode_price_v1`` and
    ``partition_description_rate_distortion_lower_bound_v1``.

CONTAINMENT: pure build + JSONL append. NO scorer, NO Modal, NO Metal, NO launch, NO training, and
no pointer or seal write. The frontier is UNMOVED. Idempotent for the equations and anti-patterns
(re-running appends another event and the query surfaces return the latest payload per id); the
anchor append is NOT idempotent, so it is guarded by ``--skip-anchor`` and by an id check.

Usage:
    .venv/bin/python tools/register_ddm_cons2_day_laws_20260911.py [--dry-run] [--verify]
                                                                  [--skip-anchor]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_anti_patterns.cons2_contract_and_compliance_20260910 import (  # noqa: E402
    build_derived_listing_inside_identity_digest_v1,
    build_fitted_scalar_in_receiver_code_is_counted_content_v1,
    build_inherited_timing_leg_reinherited_v1,
)
from tac.canonical_anti_patterns.registry import register_anti_pattern  # noqa: E402
from tac.canonical_equations.equation import (  # noqa: E402
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    EmpiricalAnchor,
)
from tac.canonical_equations.ls1_lane_surprise_atlas_oracle_ladder_20260911 import (  # noqa: E402
    build_lane_surprise_atlas_oracle_ladder_v1,
)
from tac.canonical_equations.registry import (  # noqa: E402
    get_equation_by_id,
    register_canonical_equation,
    update_equation_with_empirical_anchor,
)
from tac.canonical_equations.sj1_composition_subadditive_on_pair_overlap_20260911 import (  # noqa: E402
    build_token_edit_composition_subadditive_on_pair_overlap_v1,
)
from tac.provenance.builders import build_provenance_for_research_sidecar  # noqa: E402

AGENT = "ddm_cons2_consolidate_20260910_arms_receipts_20260911"

YIELD_EQUATION_ID = "rate_directed_predistortion_yield_v1"
YIELD_ANCHOR_ID = "rp1_sj1_add_leg_asymmetry_and_round2_k_curve_20260910"

EQUATION_BUILDERS = (
    (
        build_lane_surprise_atlas_oracle_ladder_v1,
        "ddm_ls1 + ddm_ls2 on the SHIPPED move-44 field. Registered with the ladder's own "
        "receiver-visible flag, because the charter's framing ('oracle 17,534 B receiver-visible, "
        "1,035 B short') pairs the receiver_lane MM VALUE with the granted_previous_row_lane "
        "SHORTFALL -- two different rows of one table, and the 1,035 B rung uses previous-row "
        "geometry no decoder has. The receiver-visible rung's own shortfall is 8,364.784 B (MM) "
        "or 3,676.476 B against the optimistic plug-in. ls2's refusal is TIMING, not gain.",
    ),
    (
        build_token_edit_composition_subadditive_on_pair_overlap_v1,
        "ddm_sj1 section 31, confirmed on the exact axis as pointer move 40 "
        "(S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]). The seg term is 7.6x the byte "
        "term and it is the whole story: a repair belongs to the RENDER, so a sister's disjoint "
        "token writes on a SHARED pair can un-make it. n = 1 composition, said out loud in the "
        "domain; the NUMBERS do not transfer, the sign and the locus do.",
    ),
)

ANTI_PATTERN_BUILDERS = (
    (
        build_fitted_scalar_in_receiver_code_is_counted_content_v1,
        "ddm_pr8, second family, on the runtime TREE. The producing arms' own compliance line was "
        "'no per-frame fitted table' -- true and beside the point. The exact row stands at "
        "S 0.13758600733559048; the archive does not qualify; cpd1 built the retraction mechanism. "
        "The door is real and re-enters COUNTED: rlc1 put the same constants in a 19-byte rider "
        "for a 60 B net saving.",
    ),
    (
        build_derived_listing_inside_identity_digest_v1,
        "ddm_pr14, cured by ddm_pr18. The frozen pre-fire contract refused its FIRST real producer "
        "with ZERO executable bytes changed. It had not fired earlier only because earlier "
        "manifests were stale; honest regeneration closed it for everyone at once. Cure by SCOPE: "
        "exclude the listing from the BEHAVIOR digest, validate it beside the digest, never widen "
        "into CUSTODY.",
    ),
    (
        build_inherited_timing_leg_reinherited_v1,
        "ddm_ntb2, read-only by cons2. A CONTRACT FACT, not an S-arithmetic law -- registered as an "
        "anti-pattern because the forbidden MOVE (inheriting from an inherited leg, or patching "
        "the contract that blocks you) is what recurs. The refusal's cause is MEASURED by a "
        "pointer-swap control: the same leg passes at 1232.418725255 s with move 44 as the pointer.",
    ),
)


def build_yield_add_leg_anchor() -> EmpiricalAnchor:
    """The 2026-09-10 legs of rp1's yield law: the ADD-leg asymmetry and the round-2 K curve."""
    ledger = ".omx/research/ddm_rp1_round2_move40_20260910.md"
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=ledger,
        reactivation_criteria=(
            "the K curve is measured on move 40's field with 24 pairs and 6,144 realized "
            "proposals; re-measure it on any field a pointer move changes. The stop rule is "
            "pre-registered (continue while a marginal band returns >= 2x the 2e-05 admit bar per "
            "shard-hour) and it is the thing to re-apply, not the band numbers"
        ),
        measurement_axis="[macOS-CPU realized re-encode; contest-CUDA T4 n600 for the exact rows]",
        hardware_substrate="macOS CPU (real re-encodes, frozen CPU SegNet neutrality) + T4 for moves 40/42",
        captured_at_utc="2026-09-10T00:00:00Z",
    )
    return EmpiricalAnchor(
        anchor_id=YIELD_ANCHOR_ID,
        measurement_utc="2026-09-10T00:00:00Z",
        inputs={
            "add_leg": "sj1 pass 5 -- surprise WRITTEN IN to a field, priced by real re-encode",
            "k_curve": "rp1 round 2 on move 40's field, 24 pairs, 6,144 realized proposals, K to 256",
            "why_this_is_a_second_leg": (
                "the 09-09 anchor measured only the TAKE-OUT direction (0.1445 raw, 0.2663 under "
                "per-pair Lagrange selection); the asymmetry needs both"
            ),
        },
        predicted_output={
            "symmetry_assumption": "a first-order -log2 p sum prices both directions alike",
        },
        empirical_output={
            "add_leg_realized_over_modelled": 1.2793,
            "take_out_leg_realized_over_first_order": [0.1445, 0.2663],
            "asymmetry_statement": (
                "under the shipped causal mixer a taken-out surprise pays 0.15-0.27 of its "
                "ranking; an added one costs 1.28x -- both from one object"
            ),
            "neutrality_is_flat_in_rank": {
                "overall_fraction": 0.0426,
                "per_band_range": [0.0312, 0.0521],
                "bands_tested_to_rank": 256,
                "note": "no trend across an 8x K extension -- only the PRIZE decays",
            },
            "prize_decay_bits_per_test": {"first_band": 0.477, "last_band": 0.056, "factor": 8.5},
            "marginal_s_per_shard_hour": {
                "32_to_64": 8.9e-05,
                "64_to_128": 7.0e-05,
                "128_to_192": 4.8e-05,
                "192_to_256": 3.1e-05,
            },
            "pre_registered_stop_rule": (
                "continue while the marginal band returns >= 2x the 2e-05 admit bar per shard-hour "
                "(>= 4e-05 S/h); K = 192 is the last band that clears it, 192 -> 256 is REFUSED"
            ),
            "what_the_stop_rule_bought": {
                "pointer_move": 42,
                "s": 0.1374765052591843,
                "archive_bytes": 180_238,
                "axis": "[contest-CUDA T4 n600]",
                "seg_delta": "exactly 0; the pose landed below base; rate +5 B",
            },
            "in_loop_joint_neutrality_check": (
                "removed round 1's failure entirely -- 0/24 pairs disagreed post-hoc, against "
                "12.9 % needing greedy narrowing before it. Prevention, not detection"
            ),
        },
        # |1.2793 - 1| : how far the ADD direction is from the first-order model that priced it.
        residual=0.2793,
        source_artifact=ledger,
        measurement_method=(
            "realize every proposal against the frozen CPU SegNet for joint (composite) argmax "
            "neutrality, then price the SELECTED subset by a real re-encode through the shipped "
            "coder -- never by its first-order -log2 p sum"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )


def _append_yield_anchor(*, dry_run: bool) -> int:
    existing = get_equation_by_id(YIELD_EQUATION_ID)
    if existing is None:
        print(f"ABSENT   {YIELD_EQUATION_ID} -- cannot append; register it first")
        return 1
    have = {a.anchor_id for a in existing.empirical_anchors}
    print(f"{YIELD_EQUATION_ID}: existing anchors={sorted(have)}")
    if YIELD_ANCHOR_ID in have:
        print(f"  SKIP -- {YIELD_ANCHOR_ID} already present (the append is not idempotent)")
        return 0
    anchor = build_yield_add_leg_anchor()
    print(f"  built anchor {anchor.anchor_id}")
    if not dry_run:
        update_equation_with_empirical_anchor(
            YIELD_EQUATION_ID,
            anchor,
            agent=AGENT,
            notes=(
                "ddm_cons2: the 2026-09-10 legs the 09-09 row lacks -- sj1's ADD-leg asymmetry "
                "(1.2793x modelled) against the take-out leg's 0.1445-0.2663, and rp1's round-2 K "
                "curve showing neutrality FLAT to rank 256 (4.26 %) while the prize decays 8.5x. "
                "Appended rather than given a second equation id: one law, two directions."
            ),
        )
        print("  anchor appended -> .omx/state/canonical_equations_registry.jsonl")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="build + validate, append nothing")
    ap.add_argument("--verify", action="store_true", help="read the registries back and report")
    ap.add_argument("--skip-anchor", action="store_true", help="do not touch the yield equation")
    args = ap.parse_args(argv)

    if args.verify:
        missing = []
        for builder, _ in EQUATION_BUILDERS:
            equation_id = builder().equation_id
            found = get_equation_by_id(equation_id)
            print(f"{'PRESENT ' if found else 'ABSENT  '} {equation_id}")
            if not found:
                missing.append(equation_id)
        yield_row = get_equation_by_id(YIELD_EQUATION_ID)
        have = {a.anchor_id for a in yield_row.empirical_anchors} if yield_row else set()
        ok = YIELD_ANCHOR_ID in have
        print(f"{'PRESENT ' if ok else 'ABSENT  '} {YIELD_EQUATION_ID}::{YIELD_ANCHOR_ID}")
        if not ok:
            missing.append(YIELD_ANCHOR_ID)
        return 1 if missing else 0

    for builder, notes in EQUATION_BUILDERS:
        equation = builder()
        print(f"built {equation.equation_id}: {equation.one_line_summary}")
        print(f"  producers={equation.canonical_producers}")
        print(f"  consumers={equation.canonical_consumers}")
        print(f"  anchors={[a.anchor_id for a in equation.empirical_anchors]}")
        if not args.dry_run:
            register_canonical_equation(equation, agent=AGENT, notes=notes)
            print("  registered -> .omx/state/canonical_equations_registry.jsonl")

    for builder, notes in ANTI_PATTERN_BUILDERS:
        anti_pattern = builder()
        print(f"built {anti_pattern.anti_pattern_id} ({anti_pattern.severity})")
        print(f"  producers={anti_pattern.canonical_producers}")
        print(f"  consumers={anti_pattern.canonical_consumers}")
        print(
            "  falsifications="
            f"{[f.falsification_id for f in anti_pattern.empirical_falsifications]}"
        )
        if not args.dry_run:
            register_anti_pattern(anti_pattern, agent=AGENT, notes=notes)
            print("  registered -> .omx/state/canonical_anti_patterns_registry.jsonl")

    if not args.skip_anchor:
        rc = _append_yield_anchor(dry_run=args.dry_run)
        if rc:
            return rc

    if args.dry_run:
        print("DRY-RUN: nothing registered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
