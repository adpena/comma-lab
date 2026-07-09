#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seed the canonical lever-relative-significance store from the 2026-07-08 re-audit RE-OPEN table.

The store ``.omx/state/lever_relative_significance.jsonl`` is a canonical ``.omx/state/*.jsonl`` runtime
ledger and is GITIGNORED (per CLAUDE.md "Do not track raw .omx/state/*.json"), exactly like its sibling
``lever_activation_ledger.jsonl``. So the REPRODUCIBLE source of truth for the initial population lives
HERE, in committed code — run this script to (re)populate the store on any checkout.

Rows are the measured/estimated ΔS anchors from ``.omx/research/relative_significance_reaudit_20260708.md``
(the recurring "relative-not-absolute-significance-near-goal" lesson). Each carries its ``source_anchor``
(NO-FAKE: every ΔS cites where it was measured/estimated). Score-neutral apparatus — read/rank/log only.

Usage:
  .venv/bin/python tools/seed_lever_relative_significance.py            # seed (reset then write)
  .venv/bin/python tools/seed_lever_relative_significance.py --append   # append without reset
  .venv/bin/python tools/seed_lever_relative_significance.py --show     # print the ranked store, no write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac.witness_dsl import activation_ledger as al  # noqa: E402

# (lever, est_delta_s, label, axis, source_anchor, notes) — the re-audit RE-OPEN table, verbatim anchors.
SEED_ROWS: tuple[tuple, ...] = (
    ("d_seg_aware_taper_121", 0.03, al.SIG_LABEL_ESTIMATED, "d_seg",
     "#121 / relative_significance_reaudit_20260708",
     "d_seg-aware taper; +18% NO-GO RETRACTED (under-converged ge300/3000); converged anchors flip sign "
     "to -8% ~0.03; RE-VALIDATE at convergence (cheap disk A/B). verdict_scope: instance-under-convergence."),
    ("horizon_weighted_margin_169", 0.018, al.SIG_LABEL_MEASURED, "d_seg",
     "#169 / relative_significance_reaudit_20260708 / DAG FEED-v75Aactuated B.5 / FEED-horizonmargin",
     "0-byte in-training horizon-weighted margin; MEASURED oracle CEILING 0.012-0.024 (midpoint 0.018). "
     "BUILT + FIREABLE: DSL HorizonWeightedMargin factory + trainer loss term (L4953) + reference twin "
     "tac.boundary_math.horizon_weighted_margin + eq horizon_weighted_margin_hinge_v1. NEVER-FIRED: owed "
     "a CONVERGED n600 byte-close A/B (surviving flips must shift to HIGHER GT margin, else terminal)."),
    ("seg_chroma_boundary_276", None, al.SIG_LABEL_UNMEASURED, "d_seg",
     "LEVER-4c / chroma DOF probe a3e9f0bd / #276 chroma-DOF / DAG FEED-chromalever",
     "0-byte annulus chroma-boundary MATCH (w*mean_ann ||chroma(f1)-chroma(GT)||^2 on the SHARED "
     "realized-through-R render; chroma=rgb-BT.601-luma = LUMA-INVARIANT). BUILT + FIREABLE: DSL "
     "SegChromaBoundary factory + trainer loss term (L4933) + reference twin "
     "tac.boundary_math.chroma_boundary_match + eq chroma_boundary_annulus_match_hinge_v1. DOF MEASURED "
     "(removal ablation: 7.54% Lane->Road + 4.38% Movable->Undriv, 93.4% in margin<1 annulus) but the "
     "ADD-BACK score ΔS is UNMEASURED -> duty-to-ESTIMATE inside v7.5, then a CONVERGED n600 byte-close "
     "A/B (surviving annulus flips must shift toward GT chroma, else terminal). NEVER-FIRED."),
    ("StepNativeActivation", 0.013, al.SIG_LABEL_MEASURED, "d_seg",
     "activation screen / FINER -4.5% n600 / relative_significance_reaudit_20260708 #3",
     "step-native / FINER++ vs sine; -18.7% n100 -> -4.5% n600; adopt-verdict OWED (screen is LIVE). "
     "Registered DSL lever — the value-join working on a real factory name."),
    ("seg_down_weight_274", None, al.SIG_LABEL_UNMEASURED, "d_seg",
     "#274 / relative_significance_reaudit_20260708",
     "seg down-weight lever (BUILT, the standing seg play); ΔS not-yet-measured-at-optimal -> "
     "duty-to-ESTIMATE inside v7.5; ensure it is in the optimal-combination set, not default-off."),
    ("latent_table_truncate_d18_k90", 0.001, al.SIG_LABEL_ESTIMATED, "rate",
     "D18 k90 / relative_significance_reaudit_20260708",
     "latent-table TRUNCATE-at-export free-byte cut; near-goal any real byte cut is undiluted S; "
     "sensor ARMED, run at stop-time A/B. 'rate is dead' is a floor claim, not a licence to skip a free cut."),
    ("mod32_neutrality_19_ab", 0.0005, al.SIG_LABEL_ESTIMATED, "rate",
     "19-neutrality / relative_significance_reaudit_20260708",
     "mod-32 rate-saving A/B; non-blocking != never; fold into stop-time byte-close A/B alongside D18."),
)


def seed(path: Path | None = None, *, reset: bool = True, agent: str = "relsig-fold-seed") -> int:
    """(Re)populate the store. ``reset`` clears the file first (latest-wins makes append also correct,
    but reset keeps the committed seed the sole source with no stale accretion). Returns rows written."""
    p = Path(path) if path is not None else al.SIGNIFICANCE_PATH
    if reset and p.exists():
        p.unlink()
    for lever, est, label, axis, src, notes in SEED_ROWS:
        al.record_relative_significance(lever, est, label=label, source_anchor=src, axis=axis,
                                        notes=notes, agent=agent, path=p)
    return len(SEED_ROWS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--append", action="store_true", help="append without clearing the store first")
    ap.add_argument("--show", action="store_true", help="print the ranked store and exit (no write)")
    args = ap.parse_args(argv)
    if args.show:
        for r in al.duty_to_measure_ranked():
            if r.get("est_delta_s") is not None or r.get("in_duty_queue"):
                pct = r.get("rel_sig_pct")
                print(f"{r['lever']:34s} est={r['est_delta_s']} rel_sig={pct}% axis={r.get('axis')}")
        return 0
    n = seed(reset=not args.append)
    print(f"seeded {n} rows -> {al.SIGNIFICANCE_PATH}")
    print("ranked top:")
    for r in al.duty_to_measure_ranked()[:6]:
        print(f"  {r['lever']:34s} {r.get('rel_sig_pct')}% ({r.get('delta_s_label')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
