#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""$0 desk-calc gate for the capacity-RD score-aware-QAT pivot.

Runs ``tac.capacity_rd_qat.run_desk_calc`` and prints the S_native(p) + S_QAT(p)
table + the PROCEED/STOP verdict. Writes a JSON report. NO GPU, NO training, NO
score claim — every number is [advisory] NON-PROMOTABLE.

Usage:
  .venv/bin/python tools/capacity_rd_qat_desk_calc.py
  .venv/bin/python tools/capacity_rd_qat_desk_calc.py --qat-nbits 5 --frac-low 0.6 \
      --json reports/capacity_rd_qat_desk_calc.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from tac.capacity_rd_qat import (
    ANCHOR_BC20,
    ANCHOR_FRONTIER,
    FRONTIER_ARCHIVE_BYTES,
    FRONTIER_DECODER_SECTION_BYTES,
    frontier_qat_rows,
    run_desk_calc,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-chs", type=int, nargs="+", default=[20, 24, 28, 32, 36])
    ap.add_argument("--qat-nbits", type=int, default=4, choices=[4, 5, 6, 7])
    ap.add_argument(
        "--frac-low", type=float, default=0.70, help="fraction of (d_seg-blind) weights pushed to low precision"
    )
    ap.add_argument("--hold-delta", type=float, default=0.0003, help="+delta to d_seg the QAT-hold break-even budgets")
    ap.add_argument("--proceed-threshold", type=float, default=0.30)
    ap.add_argument("--json", type=str, default="reports/capacity_rd_qat_desk_calc.json")
    args = ap.parse_args(argv)

    res = run_desk_calc(
        base_chs=tuple(args.base_chs),
        qat_nbits=args.qat_nbits,
        qat_frac_low_precision=args.frac_low,
        qat_d_seg_hold_delta=args.hold_delta,
        proceed_threshold=args.proceed_threshold,
    )

    print("=" * 92)
    print("CAPACITY-RD SCORE-AWARE-QAT DESK CALC  [advisory] NON-PROMOTABLE")
    print(
        f"QAT grid: int{args.qat_nbits} on {args.frac_low:.0%} d_seg-blind weights | "
        f"d_seg hold +{args.hold_delta:.4g} | proceed if best S_QAT < {args.proceed_threshold}"
    )
    print("=" * 92)
    print(
        f"{'base_ch':>7} {'dec_params':>10} {'d_seg':>9} {'d_seg_src':>12} "
        f"{'B_native':>9} {'S_native':>9} {'B_QAT':>8} {'S_QAT':>9}"
    )
    print("-" * 92)
    for r in res.rows:
        src = "MEAS" if r.d_seg_evidence.startswith("MEAS") else "model"
        print(
            f"{r.base_ch:>7} {r.decoder_params:>10} {r.d_seg:>9.5f} {src:>12} "
            f"{r.native_bytes:>9} {r.native_S:>9.4f} {r.qat_bytes:>8} {r.qat_S:>9.4f}"
        )
    print("-" * 92)
    print(
        f"REFERENCE   bc20 native S = {res.bc20_native_S:.4f}  |  frontier S = {res.frontier_S:.4f}  |  sub-0.15 target"
    )
    print(f"argmin NATIVE: base_ch={res.argmin_native.base_ch}  S={res.argmin_native.native_S:.4f}")
    print(f"argmin QAT   : base_ch={res.argmin_qat.base_ch}  S={res.argmin_qat.qat_S:.4f}")
    print("=" * 92)
    if res.proceed:
        print(
            f"VERDICT: PROCEED — train base_ch={res.chosen.base_ch} "
            f"(modelled S_QAT={res.chosen.qat_S:.4f} beats bc20 native {res.bc20_native_S:.4f})."
        )
    else:
        print(
            f"VERDICT: STOP — no capacity's QAT-shrunk S < {res.proceed_threshold} "
            f"(best is base_ch={res.argmin_qat.base_ch} at {res.argmin_qat.qat_S:.4f}). "
            "The pivot needs a rethink before burning training."
        )
    print("=" * 92)
    for n in res.notes:
        print(f"  · {n}")

    # --- Frontier-QAT path: QAT-shrink the EXISTING 0.19110 frontier decoder section. ---
    fq_rows = frontier_qat_rows(d_seg_hold_delta=args.hold_delta)
    print()
    print("=" * 92)
    print("FRONTIER-QAT PATH — QAT-shrink the EXISTING 0.19110 frontier decoder section")
    print(
        f"frontier archive {FRONTIER_ARCHIVE_BYTES} B | decoder section "
        f"{FRONTIER_DECODER_SECTION_BYTES} B (90.9%, MEASURED) is the QAT-attackable share"
    )
    print("=" * 92)
    print(f"{'grid':>6} {'frac_low':>8} {'dec_B':>8} {'arch_B':>8} {'S(perfect hold)':>16} {'S(+spill)':>11}")
    print("-" * 92)
    for r in fq_rows:
        flag_pf = " <SUB-0.15" if r.qat_S_perfect_hold < 0.15 else ""
        flag_sp = " <SUB-0.15" if r.qat_S_with_spill < 0.15 else ""
        print(
            f"int{r.qat_nbits:>2} {r.frac_low_precision:>8.2f} {r.decoder_section_bytes:>8} "
            f"{r.qat_archive_bytes:>8} {r.qat_S_perfect_hold:>14.4f}{flag_pf:>0} "
            f"{r.qat_S_with_spill:>9.4f}{flag_sp:>0}"
        )
    print("-" * 92)
    best_fq_perfect = min(fq_rows, key=lambda r: r.qat_S_perfect_hold)
    best_fq_spill = min(fq_rows, key=lambda r: r.qat_S_with_spill)
    print(
        f"best S(perfect hold): int{best_fq_perfect.qat_nbits} frac={best_fq_perfect.frac_low_precision} "
        f"-> {best_fq_perfect.qat_S_perfect_hold:.4f}"
    )
    print(
        f"best S(+{args.hold_delta} spill): int{best_fq_spill.qat_nbits} frac={best_fq_spill.frac_low_precision} "
        f"-> {best_fq_spill.qat_S_with_spill:.4f}"
    )
    print("=" * 92)

    out = {
        "tool": "capacity_rd_qat_desk_calc",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer_moved": False,
        "params": vars(args),
        "anchors": {
            "bc20": dataclasses.asdict(ANCHOR_BC20),
            "frontier": dataclasses.asdict(ANCHOR_FRONTIER),
        },
        "rows": [dataclasses.asdict(r) for r in res.rows],
        "argmin_native_base_ch": res.argmin_native.base_ch,
        "argmin_native_S": res.argmin_native.native_S,
        "argmin_qat_base_ch": res.argmin_qat.base_ch,
        "argmin_qat_S": res.argmin_qat.qat_S,
        "bc20_native_S": res.bc20_native_S,
        "frontier_S": res.frontier_S,
        "proceed_threshold": res.proceed_threshold,
        "proceed": res.proceed,
        "chosen_base_ch": res.chosen.base_ch if res.chosen else None,
        "notes": res.notes,
        "frontier_qat_path": {
            "frontier_archive_bytes": FRONTIER_ARCHIVE_BYTES,
            "frontier_decoder_section_bytes": FRONTIER_DECODER_SECTION_BYTES,
            "rows": [dataclasses.asdict(r) for r in fq_rows],
            "best_perfect_hold": dataclasses.asdict(best_fq_perfect),
            "best_with_spill": dataclasses.asdict(best_fq_spill),
        },
    }
    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nJSON -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
