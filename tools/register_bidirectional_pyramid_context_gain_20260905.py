#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register the bidirectional-pyramid context-gain canonical equation (ddm_bd1) into the JSONL
registry (``.omx/state/canonical_equations_registry.jsonl``) so the EQUATIONS leg carries the
closure and the next charter inherits its prior-law prediction instead of re-deriving it.

Equation: ``bidirectional_pyramid_context_gain_v1`` -- the GT SegNet argmax label field decays
so slowly in time (``P(32)/P(1) = 1.069..1.129``) that the NEXT plane is near-redundant with the
PREVIOUS one.  Every decode-order-realizable B-pyramid saves 2.84-5.68% of the 113,419 B RC64
token stream, against an UNATTAINABLE supremum of 7.46-9.03%; the charter's 8% screen falsifier
therefore bites the supremum itself, not merely one layout.

Producer: experiments/ddm_bd1_bidirectional_context_screen.py.
Consumer: .omx/research/ddm_bd1_bidirectional_pyramid_context_20260905.md.

CONTAINMENT: pure build + JSONL append; NO scorer, NO Modal, NO Metal, NO launch, NO training.
The frontier is UNMOVED -- this is a closed door recorded honestly, not a lever.  Idempotent:
re-running appends another 'registered' event; ``query_equations`` returns the latest payload
per equation_id.

Usage:
    .venv/bin/python tools/register_bidirectional_pyramid_context_gain_20260905.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.bidirectional_pyramid_context_gain_20260905 import (  # noqa: E402
    build_bidirectional_pyramid_context_gain_v1,
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

    eq = build_bidirectional_pyramid_context_gain_v1()
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
        agent="ddm_bd1_bidirectional_pyramid_context_20260905",
        notes=(
            "ddm_bd1 screen verdict: falsifier F1 FIRED at the $0 counting-model screen, so no "
            "training was launched and Metal was never requested. Three independent context "
            "ladders agree; the KT and plug-in brackets bound the reading; the GOP sweep "
            "{2,4,8,16,32} plus the unattainable all-pairs-d1 supremum price the whole family. "
            "Scorer-free exact bit/byte arithmetic; NON-PROMOTABLE; pointer unmoved."
        ),
    )
    print(f"registered {eq.equation_id} into .omx/state/canonical_equations_registry.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
