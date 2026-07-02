# SPDX-License-Identifier: MIT
"""Register the #219 AS-BUILT triality-reconcile (session 2026-07-02) canonical equations.

Registers into the JSONL registry (``.omx/state/canonical_equations_registry.jsonl``) the
campaign-meta + apparatus laws landed this session so the EQUATIONS leg of the triality matches the
DAG + DSL legs:

  1. ``powerplay_variant_ii_cost_isomorphism_v1``  (REQUIRED) -- S IS a POWERPLAY Variant-II cost;
     axis-9 = the Correctness Demonstration; #216 order = K(T,q|history).
  2. ``oom_verdict_batch_spike_peak_rss_v1``       -- the #205 OOM = batched-verdict spike; always
     chunk full-P scorer forwards (--verdict-batch 32); score-neutral launch-safety law.
  3. ``task_rd_dominates_reconstruction_rd_v1``    -- task R(D) < reconstruction R(D) (arXiv:2602.12866
     / Dobrushin-Witsenhausen); the task-space witness dominating a full-RGB codec is a THEOREM.
  4. ``store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`` -- DRIFT-CLOSE: this session's landing was
     built + tested but never persisted into the JSONL (the equations leg was incomplete). Registering
     it closes the leg drift (idempotent; re-registering just appends a fresh 'registered' event).

CONTAINMENT: pure decision + JSONL append; NO GPU, NO launch, NO trainer edits. Pointer 0.19110
UNMOVED -- this is apparatus / triality maintenance (task #219). Idempotent-safe: re-running appends
another 'registered' event; ``query_equations`` returns the latest-payload-per-equation_id.

Usage:
    .venv/bin/python tools/register_triality_reconcile_session_20260702_equations.py [--dry-run]
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

from tac.canonical_equations.oom_verdict_batch_spike_peak_rss_20260702 import (  # noqa: E402
    build_oom_verdict_batch_spike_peak_rss_v1,
)
from tac.canonical_equations.powerplay_variant_ii_cost_isomorphism_20260702 import (  # noqa: E402
    build_powerplay_variant_ii_cost_isomorphism_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402
from tac.canonical_equations.store_nothing_pose_carrier_rate_dpose_20260702 import (  # noqa: E402
    build_store_nothing_pose_carrier_rate_collapse_vs_dpose_v1,
)
from tac.canonical_equations.task_rd_dominates_reconstruction_rd_20260702 import (  # noqa: E402
    build_task_rd_dominates_reconstruction_rd_v1,
)

_BUILDERS = (
    (build_powerplay_variant_ii_cost_isomorphism_v1,
     "FEED-pp POWERPLAY Variant-II cost isomorphism (S=L(s)+task deficit); axis-9=Correctness Demo; #216 order"),
    (build_oom_verdict_batch_spike_peak_rss_v1,
     "FEED-oom #205 verdict-batch peak-RSS spike; chunk full-P scorer forwards (--verdict-batch 32); score-neutral"),
    (build_task_rd_dominates_reconstruction_rd_v1,
     "FEED-rdd task R(D)<reconstruction R(D) theorem (2602.12866/Dobrushin-Witsenhausen); task-space dominates full-RGB"),
    (build_store_nothing_pose_carrier_rate_collapse_vs_dpose_v1,
     "FEED-snx DRIFT-CLOSE store-nothing pose carrier rate collapse (built+tested this session, JSONL-register now)"),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="build + print, do not append to the registry")
    args = ap.parse_args(argv)

    out = []
    for build, note in _BUILDERS:
        eq = build()
        out.append({"equation_id": eq.equation_id, "anchors": len(eq.empirical_anchors),
                    "well_calibrated": eq.is_well_calibrated, "note": note})
        if not args.dry_run:
            register_canonical_equation(
                eq, subagent_id="triality-reconcile-219-20260702", notes=note)
    print(json.dumps({"stage": "dry_run" if args.dry_run else "registered",
                      "count": len(out), "equations": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
