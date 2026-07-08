"""Register the T5 CRUCIBLE measured laws (2026-07-07) into the canonical-equations JSONL
registry (``.omx/state/canonical_equations_registry.jsonl``) — the EQUATIONS leg of the
crucible's requirement-G triality integration (operator 2026-07-07: "all must be integrated
into triality especially DSL and tasks or it's ephemeral").

Registers 8 NEW equations (S2 anneal-truncation + finisher-budget laws; S4 entropy-floor +
measured-archive-rate rows; S6 pose-second-wall bound; S5 islands-necessity floor +
ledger-truth apparatus law; S3 indefinite-spectrum row [PROVISIONAL K=8]) and appends 3
anchor updates to EXISTING rows (muon finisher cold-quench second receipt; lane-band +
pose-carrier real byte-close marginals) instead of duplicating them.

Idempotent by inspection of prior events (skips equation_ids/anchor_ids already present).
MEANS; pointer contest-CPU 0.19110 UNMOVED.

    .venv/bin/python tools/register_t5_crucible_equations_20260707.py --dry-run
    .venv/bin/python tools/register_t5_crucible_equations_20260707.py
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

from tac.canonical_equations.registry import (  # noqa: E402
    load_registry_events_lenient,
    register_canonical_equation,
    update_equation_with_empirical_anchor,
)
from tac.canonical_equations.t5_crucible_measured_laws_20260707 import (  # noqa: E402
    ALL_BUILDERS,
    ANCHOR_UPDATES,
)

_SUBAGENT_ID = "t5-crucible-triality-integration-tranche1-20260707"
_NOTES = (
    "T5 CRUCIBLE equations leg (requirement G.1): measured seat findings from "
    ".omx/research/t5_crucible/position_S{2,3,4,5,6}_*_20260707.md; all rows "
    "[macOS-CPU advisory]/[macOS-MLX research-signal] NON-PROMOTABLE; pointer 0.19110 UNMOVED"
)


def _existing_state() -> tuple[set[str], set[str]]:
    """Return (registered equation_ids, existing anchor_ids) from the live registry."""
    eq_ids: set[str] = set()
    anchor_ids: set[str] = set()
    for ev in load_registry_events_lenient():
        eq_ids.add(ev.get("equation_id", ""))
        payload = ev.get("equation_payload") or {}
        for a in payload.get("empirical_anchors", []) or []:
            aid = a.get("anchor_id")
            if aid:
                anchor_ids.add(aid)
    return eq_ids, anchor_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eq_ids, anchor_ids = _existing_state()
    results: list[dict[str, object]] = []

    for build in ALL_BUILDERS:
        eq = build()  # builds + validates (raises on schema violation)
        if eq.equation_id in eq_ids:
            results.append({"equation_id": eq.equation_id, "skipped": "already_registered"})
            continue
        if args.dry_run:
            results.append({"equation_id": eq.equation_id, "dry_run": True})
            continue
        register_canonical_equation(eq, subagent_id=_SUBAGENT_ID, notes=_NOTES)
        results.append({"equation_id": eq.equation_id, "registered": True})

    for target_id, build_anchor in ANCHOR_UPDATES:
        anchor = build_anchor()
        if anchor.anchor_id in anchor_ids:
            results.append({"equation_id": target_id, "anchor_id": anchor.anchor_id,
                            "skipped": "anchor_already_present"})
            continue
        if args.dry_run:
            results.append({"equation_id": target_id, "anchor_id": anchor.anchor_id,
                            "dry_run": True})
            continue
        update_equation_with_empirical_anchor(
            target_id, anchor, subagent_id=_SUBAGENT_ID, notes=_NOTES,
        )
        results.append({"equation_id": target_id, "anchor_id": anchor.anchor_id,
                        "anchor_appended": True})

    print(json.dumps({"stage": "register_t5_crucible_equations_20260707",
                      "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
