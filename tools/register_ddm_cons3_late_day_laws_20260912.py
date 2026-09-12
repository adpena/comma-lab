#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register the late-2026-09-11 arms' MEASURED laws into the canonical registries (ddm_cons3).

Six canonical ANTI-PATTERNS, two anchors APPENDED to equations that already own their laws, and
one falsification APPENDED to an anti-pattern a sister consolidation registered from the other
side.  Per the mh1 finding that 68.6 % of findings reach no consumer, every row here carries a
NAMED consumer, and every producer/consumer path is checked to resolve on disk by ``--verify``.

NO NEW EQUATION IS REGISTERED, and that is a decision, not an omission:

  * ddm_tmx1's refit result belongs to ``hpr1_counted_section_refit_debt_v1``, which cons1
    registered at n = 1 with hpr1's own bar written into its reactivation field: "a SECOND
    section's refit".  This is that second section, and it LOSES.  The law becomes n = 2 with one
    positive and one NEGATIVE instance -- which is a weaker claim than it had, correctly.
  * ddm_tmx1's container fact belongs to ``model_section_edit_container_break_fee_v1``, whose
    domain already EXCLUDES sections that are not range-coded under a match-finding compressor.
    tmx1 measured that exclusion (one STORE member, 100 B of ZIP overhead, stream bytes reaching
    archive bytes 1:1 on four paired rows) instead of reasoning it.
  * ddm_pr19's two adjudications produce no arithmetic of S, and pr19 says so itself: "contract
    text over a custody/spend gate, not laws of the S-arithmetic".  Registering an equation for
    them would import an instrument that does not exist, so the risk-gate defect lands as an
    anti-pattern whose UNWIND PATH carries the envelope arithmetic, and the chain refusal is
    appended to the anti-pattern ntb2's side already produced.

ANTI-PATTERNS
  * ``refit_rungs_ranked_by_conditioned_mass_v1``  -- tmx1 fired the rung hpr1's audit ranked #1
    by conditioned mass and it returned the wrong sign (+20 B against a -30.04 B bar) while the
    section ranked below it repaid -887 B on the SAME stream.
  * ``noisy_local_proxy_projects_a_gate_on_a_measured_quantity_v1`` -- pr19: a 2.83x-spread local
    proxy with a one-sided clamp refused a candidate at 1,367.77 s that measures 1,232.42 s.
  * ``contract_code_landed_without_its_ledger_row_v1`` -- pr19's code landed, its rows were held,
    and no producer could emit a pre-fire intent at all (2 of 15 implementation rows drifting).
  * ``undived_reserve_constant_transferred_across_volume_regimes_v1`` -- sr5: a boot-volume swap
    floor enforced on SSDs that host no swap; a 3 KB copy refused with 35.3 GiB free.
  * ``display_unit_label_read_as_the_true_unit_v1`` -- sr5: df -h prints the 10^9 value under a
    'Gi' label, so a target written from the display is 1.0737x off.
  * ``warm_start_init_dropped_the_quantization_state_v1`` -- MAIN 2026-09-12: the refit law's
    warm-start init carries ZERO bit_depth keys while its source checkpoint carries ten, so every
    refit relearns depths from 8 bits (5.888-5.890 against the shipped 4.116) and pays +351 B.
    The CURE's price is ddm_dpi1's and is PENDING; nothing is claimed for it here.

DSL ACTIVATION.  Five of the six ``Lever`` factories in
``tac.witness_dsl.cons3_closed_levers_20260912`` get ``fired`` + ``measured`` + ``retired`` events
so the duty queue carries their verdict instead of an orphan.  The sixth,
``Hpr1ConvADilationRung``, gets NO events on purpose: it was designed and costed and never fired,
so it SHOULD sit in never_fired() and duty_to_measure() until a real encode prices it.  Events key
on the FACTORY name (the AST surface), never on the Lever's name string.

CONTAINMENT: pure build + JSONL append.  NO scorer, NO Modal, NO Metal, NO launch, NO training,
and no pointer or seal write.  The frontier is UNMOVED: composition S 0.13638261682704697 @
179,111 B [contest-CUDA T4 n600] (move 48).  Anti-pattern registration is idempotent (re-running
appends another event and the query surfaces return the latest payload per id); the anchor and
falsification appends are NOT, so each is guarded by an id check.

Usage:
    .venv/bin/python tools/register_ddm_cons3_late_day_laws_20260912.py [--dry-run] [--verify]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_anti_patterns.cons3_custody_units_and_warm_start_20260912 import (  # noqa: E402
    build_display_unit_label_read_as_the_true_unit_v1,
    build_undived_reserve_transferred_across_volume_regimes_v1,
    build_warm_start_init_dropped_the_quantization_state_v1,
)
from tac.canonical_anti_patterns.cons3_refit_and_timing_contract_20260912 import (  # noqa: E402
    build_contract_code_landed_without_its_ledger_row_v1,
    build_noisy_local_proxy_projects_a_gate_v1,
    build_pr19_chain_inheritance_refusal_falsification,
    build_refit_rungs_ranked_by_conditioned_mass_v1,
)
from tac.canonical_anti_patterns.registry import (  # noqa: E402
    append_empirical_falsification,
    get_anti_pattern_by_id,
    register_anti_pattern,
)
from tac.canonical_equations.cons3_refit_and_container_anchors_20260912 import (  # noqa: E402
    CONTAINER_ANCHOR_ID,
    CONTAINER_EQUATION_ID,
    REFIT_ANCHOR_ID,
    REFIT_EQUATION_ID,
    build_tmx1_low_capacity_refit_negative_anchor,
    build_tmx1_store_member_container_scope_anchor,
)
from tac.canonical_equations.registry import (  # noqa: E402
    get_equation_by_id,
    update_equation_with_empirical_anchor,
)
from tac.witness_dsl.activation_ledger import (  # noqa: E402
    duty_to_measure,
    known_levers,
    never_fired,
    record_activation,
)

AGENT = "ddm_cons3_consolidate_late_20260911_arms_20260912"

CHAIN_ANTI_PATTERN_ID = "inherited_timing_leg_reinherited_v1"
CHAIN_FALSIFICATION_ID = "pr19_chain_inheritance_refused_on_the_real_trees_20260911"

ANTI_PATTERN_BUILDERS = (
    (
        build_refit_rungs_ranked_by_conditioned_mass_v1,
        "ddm_tmx1 refuting ddm_hpr1's staleness-audit ranking rule by firing its #1 rung: 40 B of "
        "fitted mixer state on a 118,511 B stream repays +20 B, while 12,262 B of fitted HPAC "
        "state on the SAME stream repaid -887 B. The replacement quantity (fitted capacity) is "
        "registered as the UNWIND PATH at n=2, never as a law -- tmx1's own words: 'a proposal at "
        "n=2 (one positive, one negative), not a law'.",
    ),
    (
        build_noisy_local_proxy_projects_a_gate_v1,
        "ddm_pr19: the local cold-decode instrument spreads 2.83x WITHIN one identity class, "
        "including 1.21x between two runs of identical bytes, and the gate's max(0, ratio-1) clamp "
        "is one-sided. The cure (measured_t4_identity_class_envelope) is stronger in five places "
        "and weaker in exactly one, and its honest residual -- the ceiling is a max over DECLARED "
        "legs -- is carried in the reactivation field rather than hidden.",
    ),
    (
        build_contract_code_landed_without_its_ledger_row_v1,
        "ddm_pr19's landing, measured by ddm_hpr1: code landed at a9fd12704 with its amendment "
        "rows deliberately held, and the frozen-file check then refused EVERY new pre-fire intent "
        "(2 of 15 implementation rows drifting). Both halves of the choice were defensible, which "
        "is why the anti-pattern is about SEQUENCING and names no culprit.",
    ),
    (
        build_undived_reserve_transferred_across_volume_regimes_v1,
        "ddm_sr5 unit 2. cons3 re-derived both constants from source rather than quoting the memo: "
        "storage_tiers.py:26 carries 40.0 with no comment, launch_detached_process.py:396 carries "
        "40.0 with a full boot-volume swap derivation. The memo's ~21/~24 GiB is a PROPOSAL and "
        "nothing was changed; its own concurrency judgement (2 arms) is carried as the "
        "reactivation condition, because 4 concurrent decodes would vindicate 40.",
    ),
    (
        build_display_unit_label_read_as_the_true_unit_v1,
        "ddm_sr5 unit 1: df -h prints the 10^9 value under a 'Gi' label on this host, so a target "
        "written from the display is 1.0737x off (37.97 shown where 35.358 GiB is true). The value "
        "is right and the LABEL is wrong, which is why cross-checking against the same display "
        "always agrees.",
    ),
    (
        build_warm_start_init_dropped_the_quantization_state_v1,
        "MAIN's 2026-09-12 measurement, registered while ddm_dpi1 is LIVE and curing it. The "
        "DEFECT is measured (zero bit_depth keys in the init, ten in its source checkpoint, "
        "5.888-5.890 against 4.116 bits, +351 B packed); the CURE's price is dpi1's and is "
        "PENDING, and this registration claims nothing for it -- tmx1's low-capacity refit is the "
        "standing reminder that a restoration can also lose.",
    ),
)

# factory name -> (verdict_ref, reason) for the five levers that HAVE a verdict.
CLOSED_LEVERS = {
    "Pc3Cap1PredictorRefit": (
        ".omx/research/ddm_pc3_cap1_predictor_refit_move44_contest_cuda_20260911_"
        "pointer_move_45_20260911.md",
        "LANDED as pointer move 45: -160 B at bit-identical codes (51,581 -> 50,270 Rice bits), "
        "cold n600 decode byte-identical to move 44 across 3,662,409,600 bytes; superseded as a "
        "rung only by its own expiring price",
    ),
    "Ntb2HpacFrameQuadRounding": (
        ".omx/research/ddm_hpr1_composition_and_inheritance_wall_20260911.md",
        "FALSIFIED at exact bytes: +58 B (archive 179,417), dS +3.8620e-05, against step 2's "
        "-248 B. It LOCATES the turn between step 2 and step 4; the rounding family is closed at "
        "this resolution because coarser does not pay",
    ),
    "Hpr1QRecalibrationRung": (
        ".omx/research/ddm_hpr1_composition_and_inheritance_wall_20260911.md",
        "CLOSED NEGATIVE: held-out recalibration gain -9,439.97 B on the refit prior against "
        "-183.53 B on the old one (51x harder), because the refit moved q CLOSER to its optimum "
        "-- calibration error -1.21e-06 over 117,964,800 symbols",
    ),
    "Tmx1TailMixerRefit": (
        ".omx/research/ddm_tmx1_refit_tail_mixer_tc1_on_current_field_20260911.md",
        "CLOSED NEGATIVE at exact bytes: +20 B best of four fits on a -30.04 B bar, six 600-frame "
        "encodes, two bases, four held-out folds, DO NOT FIRE. This is the n=2 NEGATIVE anchor of "
        "hpr1_counted_section_refit_debt_v1",
    ),
    "Rbf1PostRenderBoundaryTreatment": (
        ".omx/research/ddm_rbf1_free_post_render_boundary_treatment_20260911.md",
        "CLOSED at formulation scope on two n600 channels: a free post-render pixel edit pays a "
        "quadratic pose tax (~6.6e-3 tau^2, n=24) that symmetric application does not cure, with "
        "2,287,200 token-edge pixels touched to reach 12,196 wrong ones (187.5:1) on the seg side",
    ),
}

# Registered with a factory and deliberately NO events: it must stay in the duty queue.
UNFIRED_LEVERS = ("Hpr1ConvADilationRung",)


def _append_equation_anchor(equation_id: str, anchor_id: str, builder, notes: str, *, dry_run: bool) -> int:
    existing = get_equation_by_id(equation_id)
    if existing is None:
        print(f"ABSENT   {equation_id} -- cannot append; register it first")
        return 1
    have = {a.anchor_id for a in existing.empirical_anchors}
    print(f"{equation_id}: existing anchors={sorted(have)}")
    if anchor_id in have:
        print(f"  SKIP -- {anchor_id} already present (the append is not idempotent)")
        return 0
    anchor = builder()
    print(f"  built anchor {anchor.anchor_id} (residual {anchor.residual})")
    if not dry_run:
        update_equation_with_empirical_anchor(equation_id, anchor, agent=AGENT, notes=notes)
        print("  anchor appended -> .omx/state/canonical_equations_registry.jsonl")
    return 0


def _append_chain_falsification(*, dry_run: bool) -> int:
    existing = get_anti_pattern_by_id(CHAIN_ANTI_PATTERN_ID)
    if existing is None:
        print(f"ABSENT   {CHAIN_ANTI_PATTERN_ID} -- cannot append; register it first")
        return 1
    have = {f.falsification_id for f in existing.empirical_falsifications}
    print(f"{CHAIN_ANTI_PATTERN_ID}: existing falsifications={sorted(have)}")
    if CHAIN_FALSIFICATION_ID in have:
        print(f"  SKIP -- {CHAIN_FALSIFICATION_ID} already present (the append is not idempotent)")
        return 0
    falsification = build_pr19_chain_inheritance_refusal_falsification()
    print(f"  built falsification {falsification.falsification_id}")
    if not dry_run:
        append_empirical_falsification(
            falsification,
            agent=AGENT,
            notes=(
                "ddm_pr19 adjudicated for the second family what ntb2 met as an obstacle: chains "
                "stay REFUSED with no length bound, and the decisive reason is economic (a chain "
                "buys no dispatch) rather than a drift argument -- no per-step bound exists, the "
                "one paired observation is 8.03 % and confounded with T4 host variance, and "
                "compounding it twice already exceeds the 1,260 s policy limit."
            ),
        )
        print("  falsification appended -> .omx/state/canonical_anti_patterns_registry.jsonl")
    return 0


def _record_lever_activations(*, dry_run: bool) -> None:
    known = known_levers()
    before = (len(known), len(never_fired(known)), len(duty_to_measure(known)))
    print(
        f"activation ledger BEFORE: known={before[0]} never_fired={before[1]} "
        f"duty_to_measure={before[2]}"
    )
    for factory, (verdict_ref, reason) in CLOSED_LEVERS.items():
        if factory not in known:
            print(f"  WARN {factory} is not in known_levers() -- check the factory name (AST surface)")
        if dry_run:
            print(f"  would record fired/measured/retired for {factory}")
            continue
        record_activation(factory, "fired", verdict_ref=verdict_ref, reason=reason, agent=AGENT)
        record_activation(factory, "measured", verdict_ref=verdict_ref, reason=reason, agent=AGENT)
        record_activation(factory, "retired", verdict_ref=verdict_ref, reason=reason, agent=AGENT)
        print(f"  recorded fired+measured+retired for {factory}")
    for factory in UNFIRED_LEVERS:
        print(
            f"  NO EVENTS for {factory} on purpose -- designed and costed, never fired; it must "
            "stay in never_fired() and duty_to_measure() until a real encode prices it"
        )
    known_after = known_levers()
    after = (
        len(known_after),
        len(never_fired(known_after)),
        len(duty_to_measure(known_after)),
    )
    print(
        f"activation ledger AFTER:  known={after[0]} never_fired={after[1]} "
        f"duty_to_measure={after[2]}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="build + validate, append nothing")
    ap.add_argument("--verify", action="store_true", help="read the registries back and report")
    args = ap.parse_args(argv)

    if args.verify:
        missing: list[str] = []
        for builder, _ in ANTI_PATTERN_BUILDERS:
            anti_pattern_id = builder().anti_pattern_id
            found = get_anti_pattern_by_id(anti_pattern_id)
            print(f"{'PRESENT ' if found else 'ABSENT  '} {anti_pattern_id}")
            if not found:
                missing.append(anti_pattern_id)
        for equation_id, anchor_id in (
            (REFIT_EQUATION_ID, REFIT_ANCHOR_ID),
            (CONTAINER_EQUATION_ID, CONTAINER_ANCHOR_ID),
        ):
            equation = get_equation_by_id(equation_id)
            have = {a.anchor_id for a in equation.empirical_anchors} if equation else set()
            ok = anchor_id in have
            print(f"{'PRESENT ' if ok else 'ABSENT  '} {equation_id} :: {anchor_id}")
            if not ok:
                missing.append(f"{equation_id}::{anchor_id}")
        chain = get_anti_pattern_by_id(CHAIN_ANTI_PATTERN_ID)
        have_f = {f.falsification_id for f in chain.empirical_falsifications} if chain else set()
        ok = CHAIN_FALSIFICATION_ID in have_f
        print(f"{'PRESENT ' if ok else 'ABSENT  '} {CHAIN_ANTI_PATTERN_ID} :: {CHAIN_FALSIFICATION_ID}")
        if not ok:
            missing.append(CHAIN_FALSIFICATION_ID)
        # Every producer and consumer path must resolve on disk: a registration that points at
        # nothing is an orphan wearing a citation.
        bad_paths: list[str] = []
        for builder, _ in ANTI_PATTERN_BUILDERS:
            anti_pattern = builder()
            for rel in anti_pattern.canonical_producers + anti_pattern.canonical_consumers:
                if not (REPO / rel).exists():
                    bad_paths.append(f"{anti_pattern.anti_pattern_id}: {rel}")
        for rel in bad_paths:
            print(f"BAD PATH {rel}")
        known = known_levers()
        for factory in tuple(CLOSED_LEVERS) + UNFIRED_LEVERS:
            print(f"{'KNOWN   ' if factory in known else 'UNKNOWN '} lever {factory}")
        return 1 if (missing or bad_paths) else 0

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

    rc = _append_equation_anchor(
        REFIT_EQUATION_ID,
        REFIT_ANCHOR_ID,
        build_tmx1_low_capacity_refit_negative_anchor,
        (
            "ddm_cons3: the SECOND section's refit that hpr1's own bar demanded, and it LOSES. "
            "40 B of fitted mixer state on the same 118,511 B stream that 12,262 B of HPAC prior "
            "conditions: Delta model = 0 (the 60 B rider's length is fixed by the shipped schema), "
            "Delta tail = +20 B, six exact encodes on two bases. The law is now n = 2 with one "
            "positive and one NEGATIVE instance and authorises no refit rung by itself."
        ),
        dry_run=args.dry_run,
    )
    rc |= _append_equation_anchor(
        CONTAINER_EQUATION_ID,
        CONTAINER_ANCHOR_ID,
        build_tmx1_store_member_container_scope_anchor,
        (
            "ddm_cons3: the fee law's own EXCLUSION clause, measured rather than reasoned. The "
            "candidate archive holds ONE member ('p', compress_type 0 STORE) with 100 B of ZIP "
            "overhead and a fixed-length rider, so a token-stream length change reaches archive "
            "bytes 1:1 on all four paired rows. The +-34.8 B container-break lottery does NOT "
            "govern this axis; it still governs any edit that moves the brotli'd hpac member."
        ),
        dry_run=args.dry_run,
    )
    rc |= _append_chain_falsification(dry_run=args.dry_run)

    _record_lever_activations(dry_run=args.dry_run)

    if args.dry_run:
        print("DRY-RUN: nothing registered.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
