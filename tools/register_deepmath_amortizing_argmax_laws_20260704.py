# SPDX-License-Identifier: MIT
"""Register the PROVEN "Amortizing the Argmax" deep-math laws as canonical equations (task #284 / A2).

Lands the 8 proven laws from the synthesis draft
(``.omx/research/deepmath_amortizing_argmax_paper_draft_20260704.md`` §2) into
``.omx/state/canonical_equations_registry.jsonl`` so the EQUATIONS leg of the triality AGREES with
the DAG (FEED-03y/03z) + the DSL. Idempotent (append-only 'registered' event keyed by equation_id).

The se(3)-screw temporal-sufficiency law (draft §2 law 5) is NOT registered here -- it is ALREADY
``store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`` (REFERENCED, not duplicated).

MEANS != ends: NONE of these move the pointer (contest-CPU 0.19110, UNMOVED); they NAME the geometry
of the #205 witness. The net score is #205-gated.

Usage:
    .venv/bin/python tools/register_deepmath_amortizing_argmax_laws_20260704.py
    .venv/bin/python tools/register_deepmath_amortizing_argmax_laws_20260704.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704 import (  # noqa: E402
    ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402

_SUBAGENT_ID = "deepmath-284-A2-register-amortizing-argmax-laws"
_NOTES = (
    "FEED-03z synthesis: 8 proven 'Amortizing the Argmax' laws (Maslov / Fisher-caustic / MD=NG / "
    "shearlet-rate-upper-bound / tau=eps=hbar / Modica-Mortola / MCF-erasure + anisotropy "
    "correction). MEANS; pointer 0.19110 UNMOVED; net-score #205-gated. se(3) REFERENCED "
    "(store_nothing_pose_carrier_rate_collapse_vs_dpose_v1), not duplicated."
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="build + validate the equations but do NOT write the registry")
    args = ap.parse_args(argv)

    registered: list[str] = []
    for builder in ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS:
        eq = builder()  # __post_init__ validates the full contract
        summary = {
            "equation_id": eq.equation_id,
            "n_anchors": len(eq.empirical_anchors),
            "consumers": list(eq.canonical_consumers),
            "producers": list(eq.canonical_producers),
            "verification_statuses": sorted(
                {a.empirical_verification_status for a in eq.empirical_anchors
                 if a.empirical_verification_status is not None}
            ),
        }
        print(json.dumps(summary, indent=2))
        if not args.dry_run:
            register_canonical_equation(eq, subagent_id=_SUBAGENT_ID, notes=_NOTES)
            registered.append(eq.equation_id)

    print(json.dumps({
        "stage": "dry_run" if args.dry_run else "registered",
        "n_laws": len(ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS),
        "registered": registered,
        "se3_decision": ("REFERENCED store_nothing_pose_carrier_rate_collapse_vs_dpose_v1 "
                         "(NOT duplicated)"),
        "pointer": "0.19110 UNMOVED (MEANS; net-score #205-gated)",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
