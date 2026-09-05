#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register the MODEL-section adaptive-recode ceiling (ddm_rc1) into the canonical JSONL
registry (``.omx/state/canonical_equations_registry.jsonl``) so the EQUATIONS leg carries both
the credit and its ceiling, and the next charter inherits the width-stability rule instead of
re-deriving it from a bits-per-param figure computed on the wrong denominator.

Equation: ``model_section_adaptive_recode_ceiling_v1`` -- an adaptive per-group tree coder over
the two RX1 MODEL sections' packed integer codes lands within 1% of their order-0 entropy and
buys -1,733 container B at zero distortion (179,982 -> 178,249 B).  The credit tracks the
packing's WIDTH STABILITY, not the raw coder win, and the ~1,139 B of measured first-order
structure in the IHS1 rows is real AND unaffordable at this sample size.

Producers: experiments/ddm_rc1_model_section_adaptive_recode.py,
           experiments/ddm_rc1_adaptive_section_codec.py.
Consumer:  .omx/research/ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md.

CONTAINMENT: pure build + JSONL append; NO scorer, NO Modal, NO Metal, NO launch, NO training.
Idempotent: re-running appends another 'registered' event; ``query_equations`` returns the
latest payload per equation_id.

Usage:
    .venv/bin/python tools/register_model_section_adaptive_recode_ceiling_20260905.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.model_section_adaptive_recode_ceiling_20260905 import (  # noqa: E402
    build_model_section_adaptive_recode_ceiling_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build + validate but do NOT append to the registry",
    )
    args = ap.parse_args(argv)

    eq = build_model_section_adaptive_recode_ceiling_v1()
    print(f"built {eq.equation_id}: {eq.one_line_summary}")
    print(f"  producers={eq.canonical_producers}")
    print(f"  consumers={eq.canonical_consumers}")
    print(f"  anchors={[anchor.anchor_id for anchor in eq.empirical_anchors]}")
    print(f"  residuals={dict(eq.predicted_vs_empirical_residual)}")
    if args.dry_run:
        print("DRY-RUN: not registered.")
        return 0
    register_canonical_equation(
        eq,
        agent="ddm_rc1_model_section_adaptive_recode_20260905",
        notes=(
            "ddm_rc1 verdict: ADMIT. -1,733 container B at zero distortion, 5.78x the -300 B "
            "admit bar; the charter's FALSIFIER did NOT fire. Both shipped container streams "
            "re-pack byte-identically before any race, every coded body is decoded by a fresh "
            "decoder, and the candidate decodes to the shipped field. Neither generic baseline "
            "(xz -9e, zstd --ultra -22) beats the shipped Brotli on either body. Scorer-free "
            "exact byte arithmetic; score_claim=false until MAIN fires a T4 row."
        ),
    )
    print(f"registered {eq.equation_id} into .omx/state/canonical_equations_registry.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
