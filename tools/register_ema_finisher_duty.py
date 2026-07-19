#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register the BUILT-never-fired ``--ema-decay-finisher`` lever in the duty/activation ledger.

ARM-C p0_ema_calibration (SPEC_v10 §13.3): the wider-finisher EMA (``--ema-decay-finisher``,
THETA* TIER-2 MUST-3) is BUILT in the trainer + HELD by the DSL (``EmaDecayFinisher``) but has
NEVER FIRED — exactly the "off is a tracked queue, never a forgotten default" orphan class.
This CLI makes the apparatus (not the operator) hold that memory: it appends an idempotent
relative-significance row for the ``EmaDecayFinisher`` factory so the costate SENSE layer's
``duty_to_measure_ranked`` queue surfaces it (an un-estimated owed lever is itself orphaned
signal; the row is honest UNMEASURED — NO-FAKE: no guessed ΔS).

The activation ledger's ``never_fired`` state needs no write: it is DERIVED from
``lever_factories()`` (which now discovers ``EmaDecayFinisher``) minus fired events — an empty
history honestly reports never-fired. The ONLY append this tool makes is the significance
registration (+ its duty metadata), fcntl-locked, APPEND-ONLY, latest-row-wins.

Usage:
    .venv/bin/python tools/register_ema_finisher_duty.py [--ledger-root /path/to/repo] [--dry-run]

``--ledger-root`` points at the repo whose ``.omx/state`` ledgers the costate reads (default:
this file's repo). Idempotent: an existing identical latest row -> no duplicate append.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

LEVER = "EmaDecayFinisher"
SOURCE_ANCHOR = (
    "SPEC_v10 §13.3/§13.5 (git show claude/p0_521_spec_v10_capstone_20260717:"
    ".omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md) + canonical equation "
    "ema_decay_run_geometry_v1 + trainer --ema-decay-finisher (THETA* TIER-2 MUST-3, built "
    "2026-07-0x, verified never-fired 2026-07-17)"
)
NOTES = (
    "BUILT never-fired wider-finisher EMA (SWA-style late-oscillation averaging). "
    "Duty-to-measure: fire as an A/B arm alongside the shadow-vs-live byte-close comparator "
    "(tools/compare_shadow_vs_live_byte_close.py decides the §13.3 EMA question empirically). "
    "UNMEASURED est_delta_s (honest; no guessed number) — first measurement sets it."
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger-root", type=Path, default=_REPO,
                    help="repo root whose .omx/state ledgers the costate reads")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from tac.witness_dsl.activation_ledger import (
        SIGNIFICANCE_PATH,
        _read_significance,
        activation_status,
        known_levers,
        record_relative_significance,
    )

    sig_path = args.ledger_root / ".omx" / "state" / SIGNIFICANCE_PATH.name
    led_path = args.ledger_root / ".omx" / "state" / "lever_activation_ledger.jsonl"

    if LEVER not in known_levers():
        print(json.dumps({"error": f"{LEVER} not discovered by lever_factories(); the DSL "
                          "factory must land before registration"}), flush=True)
        return 2

    st = activation_status(LEVER, path=led_path)
    existing = _read_significance(sig_path).get(LEVER)
    if existing is not None and existing.get("delta_s_label") != "UNMEASURED":
        print(json.dumps({"skipped": "already has a measured/estimated significance row",
                          "row": existing}), flush=True)
        return 0
    if existing is not None and existing.get("source_anchor") == SOURCE_ANCHOR:
        print(json.dumps({"skipped": "identical registration already present (idempotent)",
                          "state": st.state}), flush=True)
        return 0

    if args.dry_run:
        print(json.dumps({"dry_run": True, "would_register": LEVER, "state": st.state,
                          "sig_path": str(sig_path)}), flush=True)
        return 0

    row = record_relative_significance(
        LEVER, None, label="UNMEASURED", source_anchor=SOURCE_ANCHOR, axis="d_seg",
        notes=NOTES, agent="arm_c_p0_build_skiplever_ema_20260717", path=sig_path)
    print(json.dumps({"registered": LEVER, "activation_state": st.state,
                      "sig_path": str(sig_path), "row": row}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
