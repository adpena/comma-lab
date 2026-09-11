#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register the 2026-09-11 closed-arm laws into the canonical registries (ddm_cons1).

Four canonical equations and two canonical anti-patterns, each carrying the measured numbers, the
memo that produced them, and a NAMED consumer -- per the mh1 finding that 68.6 % of findings reach
no consumer, the consumer field is not optional here.

EQUATIONS
  * ``obx2_render_floor_admission_gate_v1``      -- the pre-burn gate: a correction family cannot
    go below the render floor; obx2's object measured 0.00231 = 5.77x the closure memo's ceiling
    and 7.22x the burn spec's PRE-REGISTERED one.
  * ``pc3_rung_price_expires_at_pointer_move_v1`` -- a rung price belongs to the coder state:
    per-dim /2 went 54 -> 75 B and global /2 808 -> 914 B across one pointer move.
  * ``rbf1_post_render_pose_amplitude_tax_v1``   -- Delta S_pose ~= 6.6e-3 * tau^2 (k = 9.0e-6,
    constant to 5 % over tau = 1..8, n = 24 seeded-random pairs): no payable amplitude exists.
  * ``hpr1_counted_section_refit_debt_v1``       -- Delta B = Delta model + Delta tail; the hpac
    prior, 13 moves stale, paid +351 model for -1,238 tail = -887 B. n = 1, said out loud.

ANTI-PATTERNS
  * ``unexercised_certificate_authorizes_deletion_v1`` -- a recorded hash that was never redeemed
    is a promise, not a certificate (sr5 refused 3,780,374,400 B on a 'SKIPPED_LARGE_STREAMED'
    literal, 99.2 % of an already-authorized class).
  * ``sampled_prefilter_decides_equality_v1``          -- a sampled prefilter chooses what to
    hash, it never decides equality (one of two candidate groups refuted; 3.0x over-claim).

The obx2 POSE law (``obx2_pose_vs_scorer_plane_rmse_v1``) was built and tested by ddm_obx2 on the
same day but never appended to the registry -- the built-but-unwired class -- and is registered
separately by this consolidation; it is listed by ``--verify`` here so the set can be checked.

CONTAINMENT: pure build + JSONL append. NO scorer, NO Modal, NO Metal, NO launch, NO training, and
no pointer or seal write. The frontier is UNMOVED. Idempotent: re-running appends another
'registered' event and the query surfaces return the latest payload per id.

Usage:
    .venv/bin/python tools/register_ddm_cons1_day_laws_20260911.py [--dry-run] [--verify]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_anti_patterns.registry import register_anti_pattern  # noqa: E402
from tac.canonical_anti_patterns.sr5_storage_certification_builders_20260911 import (  # noqa: E402
    build_sampled_prefilter_decides_equality_v1,
    build_unexercised_certificate_authorizes_deletion_v1,
)
from tac.canonical_equations.hpr1_counted_section_refit_debt_20260911 import (  # noqa: E402
    build_hpr1_counted_section_refit_debt_v1,
)
from tac.canonical_equations.obx2_render_floor_admission_gate_20260911 import (  # noqa: E402
    build_obx2_render_floor_admission_gate_v1,
)
from tac.canonical_equations.pc3_rung_price_expiry_20260911 import (  # noqa: E402
    build_pc3_rung_price_expires_at_pointer_move_v1,
)
from tac.canonical_equations.rbf1_pose_amplitude_tax_20260911 import (  # noqa: E402
    build_rbf1_post_render_pose_amplitude_tax_v1,
)
from tac.canonical_equations.registry import (  # noqa: E402
    get_equation_by_id,
    register_canonical_equation,
)

AGENT = "ddm_cons1_consolidate_closed_arms_20260911"

EQUATION_BUILDERS = (
    (
        build_obx2_render_floor_admission_gate_v1,
        "ddm_obx2 design-A closure: the render floor is what a boundary-moving correction CANNOT "
        "reach, so it is a NECESSARY pre-burn gate. n600 frozen CPU argmax; the default ceiling is "
        "the burn spec's PRE-REGISTERED 3.1990253844713356e-4, not the closure memo's rounded "
        "4.0e-4 -- a restatement must never loosen a pre-registered gate.",
    ),
    (
        build_pc3_rung_price_expires_at_pointer_move_v1,
        "ddm_pc3 pose corner: the same rungs against the same shipped codes, priced on move 44 and "
        "again on move 45 (pc3's OWN predictor refit). Every price rose, most on the cheapest rung "
        "-- a stale table can rank correctly and still price wrongly, and only the price admits. "
        "cl3's substitutes law measured on a new axis: the PREDICTOR, not the entropy coder.",
    ),
    (
        build_rbf1_post_render_pose_amplitude_tax_v1,
        "ddm_rbf1 declined to register this itself and named the precondition (widen the SUPPORT "
        "arm from 4 pairs to n>=120). ddm_cons1 registers ONLY the amplitude clause -- the one the "
        "arm's own n600 closure line quotes -- with its n=24 support in the domain, the f^1.16 "
        "support clause explicitly EXCLUDED, and the objection carried in the reactivation field.",
    ),
    (
        build_hpr1_counted_section_refit_debt_v1,
        "ddm_hpr1 move 47, read-only by cons1 (hpr1 is a LIVE arm; nothing was written into its "
        "tree). Registered as a measured INSTANCE with n=1 stated in the domain, per the staleness "
        "audit's own bar. NO SECANT is attached: the arm refuses one for a refit ('a refit, not a "
        "capacity rung'), and the only secant it reports, -1.2066, belongs to the FALSIFIED shape "
        "rung. Cross-check: the leg bytes re-derive -887 B independently of the archive totals, "
        "and dS/dB = 6.6584e-07 against the known exchange rate 6.658589531221714e-07.",
    ),
)

ANTI_PATTERN_BUILDERS = (
    (
        build_unexercised_certificate_authorizes_deletion_v1,
        "ddm_sr5: MEASURED on an operator-authorized 4.5 GB class, 99.2 % of which measurement "
        "refused. The cure is a producer that refuses (tools/sr5_certify_delete_derived.py, four "
        "bindings), proven by a synthetic negative control that was refused with the file intact.",
    ),
    (
        build_sampled_prefilter_decides_equality_v1,
        "ddm_sr5: the prefilter proposed two duplicate groups and the full sha256 refuted one of "
        "them; the three refuted files match every sampled window. 3.0x over-claim if the "
        "prefilter had been believed. The measured 3.411 GiB was released by HARDLINK, with "
        "nothing deleted at all.",
    ),
)

SISTER_EQUATION_IDS = ("obx2_pose_vs_scorer_plane_rmse_v1",)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="build + validate, append nothing")
    ap.add_argument("--verify", action="store_true", help="read the registries back and report")
    args = ap.parse_args(argv)

    if args.verify:
        ids = [b().equation_id for b, _ in EQUATION_BUILDERS] + list(SISTER_EQUATION_IDS)
        missing = []
        for equation_id in ids:
            found = get_equation_by_id(equation_id)
            print(f"{'PRESENT ' if found else 'ABSENT  '} {equation_id}")
            if not found:
                missing.append(equation_id)
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

    if args.dry_run:
        print("DRY-RUN: nothing registered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
