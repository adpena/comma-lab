#!/usr/bin/env python3
"""Thin CLI for tac.witness_autoconfig — the clip -> witness_config dogfood.

Derives a launch-ready :class:`tac.witness_autoconfig.WitnessConfig` from a clip's
GT cache and (with --emit-command) emits the GO-ready, attribution-clean,
from-scratch n600 launch command for
``experiments/train_levelset_witness_realized_through_R_mlx.py`` THEN
FLAG-VALIDATES it: parses the trainer's real argparse and asserts every emitted
flag exists (prints PASS/FAIL per flag). This is the NO-FAKE "never invent a
CLI flag" dogfood.

Pure CPU. No GPU, no training launch, no scoring. means != ends: this emits +
validates a command; only a byte-closed n600 exact row < 0.19110 moves the
pointer, and the LAUNCH awaits operator GO (one GPU; containment/protect/preserve).

Usage:
    .venv/bin/python tools/witness_autoconfig.py \\
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \\
        --num-pairs 600 --emit-command
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac import witness_autoconfig as wac  # noqa: E402

_TRAINER = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"


def real_trainer_flags() -> frozenset[str]:
    """Parse the trainer argparse -> the SET of real --flag names (never-invent guard)."""
    return frozenset(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', _TRAINER.read_text()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-cache", required=True,
                    help="path to the clip's GT cache (e.g. .../gt_n600.npz)")
    ap.add_argument("--num-pairs", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="launch out-dir (default: contained results/levelset_n600_v2_<n>)")
    ap.add_argument("--aggressive", action="store_true",
                    help="overfit=False: emit the aggressive Whitney-floor mod-dim (rate-saving)")
    ap.add_argument("--emit-command", action="store_true",
                    help="emit + FLAG-VALIDATE the GO-ready launch command")
    args = ap.parse_args(argv)

    overfit = not args.aggressive
    cfg = wac.derive_config(args.gt_cache, num_pairs=args.num_pairs,
                            overfit=overfit, epochs=args.epochs)

    gt_exists = (_REPO / args.gt_cache).exists()

    # --- the derived config + provenance ----------------------------------
    print(f"# tac.witness_autoconfig  {wac.ADVISORY_TAG}  pointer 0.19110 UNMOVED")
    print(f"# clip = {args.gt_cache}  (num_pairs={args.num_pairs}, overfit={overfit})")
    if not gt_exists:
        print(f"# NOTE: {args.gt_cache} not found on disk -> gt cache regen required at launch")
    print("# --- DERIVED knobs (generator -> value -> provenance) ---")
    for k, pv in cfg.provenance.items():
        flag = "fallback!" if pv.is_fallback else pv.source
        print(f"#   {k:28s} = {pv.value!r:>8}  [{flag}]  {pv.provenance}")
    print(f"#   surgical_levers_enabled      = {cfg.surgical_levers_enabled}  "
          f"(attribution-clean FIRST launch)")
    print(f"#   dm1_enabled                  = {cfg.dm1_enabled}")
    print(f"# --- portability split (corpus generalization) ---")
    for cls in (wac.Portability.SCORER_FIXED, wac.Portability.DOMAIN, wac.Portability.INSTANCE):
        knobs = sorted(k for k, v in cfg.portability.items() if v == cls)
        print(f"#   {cls:22s}: {', '.join(knobs)}")

    if not args.emit_command:
        return 0

    out_dir = args.out_dir or "experiments/results/levelset_n600_v2_attrclean_<utc>"
    cmd = cfg.to_command(out_dir)

    print("\n# === GO-READY LAUNCH COMMAND (attribution-clean, from-scratch) ===")
    print("# HOLD: await operator GO (one GPU; no autonomous heavy launch).")
    print(cmd)

    # --- the dogfood flag-validation --------------------------------------
    print("\n# === FLAG VALIDATION (every emitted flag vs the trainer argparse) ===")
    real = real_trainer_flags()
    emitted = [flag for flag, _ in cfg.to_trainer_flags(out_dir)]
    all_pass = True
    for flag in emitted:
        ok = flag in real
        all_pass &= ok
        print(f"#   [{'PASS' if ok else 'FAIL'}] {flag}")
    n = len(emitted)
    print(f"# RESULT: {sum(1 for f in emitted if f in real)}/{n} flags PASS  "
          f"-> {'ALL PASS (no invented flag)' if all_pass else 'FAIL — invented flag(s)!'}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
