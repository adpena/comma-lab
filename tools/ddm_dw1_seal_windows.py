"""ddm_dw1 — seal the matched A/B/C window tickets + the pre-fire argv diffs (guard 3).

Compiles the three matched window programs (control / distill / distill_head_relax) via the
DSL, seals each into a ddm_tb1_tr1_sealed_ticket.v1 (the governed launcher recompiles + seal-
checks it), and computes the THREE argv diffs the argv-diff law (warm_start_resume) requires:

  * A-vs-B : must be EXACTLY the 5 distill flags (+ out-dir) — matched-config discipline.
  * C-vs-A : must be EXACTLY --head-range-relax (+ out-dir).
  * B-vs-burn : the intended window deltas vs the sealed burn argv (epochs / basin-handoff off /
    gate cadence / composed-s dropped / ema derived / resume / out-dir) — catches SILENT drift.

score_claim=false; advisory; pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.witness_dsl.spec_tr1_dw1_distill_window_20260730 import dw1_window_program  # noqa: E402

E2 = "/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out/checkpoints/stage_seg_trunk_tau_final.npz"
MASK = "/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/qa24_grid_keep_mask_50.npy"
GT = "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
CACHE = "/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/distill_field_cache/distill_logits.f16.npy"
BURN_RECEIPT = "/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out/launch_receipt.json"
OUT_ROOT = "/Volumes/VertigoDataTier/pact/ddm_dw1_20260730"


def _flag_map(argv: list[str]) -> dict[str, str]:
    return {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}


def _diff(a: list[str], b: list[str]) -> dict:
    fa, fb = _flag_map(a), _flag_map(b)
    changed = {k: {"lhs": fa.get(k), "rhs": fb.get(k)} for k in set(fa) | set(fb)
               if fa.get(k) != fb.get(k)}
    return {"changed_flags": changed,
            "only_in_lhs": sorted(set(fa) - set(fb)),
            "only_in_rhs": sorted(set(fb) - set(fa))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", required=True, choices=("kd_logits", "margin_field", "argmax_ce"))
    ap.add_argument("--attack-temp", type=float, default=0.0)
    ap.add_argument("--window-epochs-ab", type=int, default=60)
    ap.add_argument("--window-epochs-c", type=int, default=40)
    ap.add_argument("--max-wall", type=float, default=75.0)
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    common = dict(mask_path=MASK, gt_cache=GT, resume_from=E2, distill_field_cache=CACHE,
                  distill_form=args.form, distill_weight=100.0, distill_temp=2.0,
                  distill_attack_temp=args.attack_temp, max_wall_minutes=args.max_wall)
    prog_b = dw1_window_program("control", str(out_root / "control"),
                                window_epochs=args.window_epochs_ab, **common)
    prog_a = dw1_window_program("distill", str(out_root / "distill"),
                                window_epochs=args.window_epochs_ab, **common)
    prog_c = dw1_window_program("distill_head_relax", str(out_root / "distill_head_relax"),
                                window_epochs=args.window_epochs_c, **common)
    av_b, av_a, av_c = prog_b.compile_trainer_argv(), prog_a.compile_trainer_argv(), \
        prog_c.compile_trainer_argv()

    burn_argv = json.loads(Path(BURN_RECEIPT).read_text())["argv"]

    diffs = {
        "A_vs_B (must be EXACTLY the 5 distill flags + out-dir)": _diff(av_a, av_b),
        "C_vs_A (must be EXACTLY --head-range-relax + out-dir)": _diff(av_c, av_a),
        "B_vs_burn (intended window deltas)": _diff(av_b, burn_argv),
    }
    tickets_dir = out_root / "tickets"
    tickets_dir.mkdir(parents=True, exist_ok=True)
    for name, prog in (("control", prog_b), ("distill", prog_a), ("distill_head_relax", prog_c)):
        t = prog.sealed_ticket()
        (tickets_dir / f"{name}_ticket.json").write_text(json.dumps(t, indent=2) + "\n")

    # Assertions: A-vs-B changed flags (minus out-dir) == the 5 distill flags; C-vs-A == head-relax.
    a_b = set(diffs["A_vs_B (must be EXACTLY the 5 distill flags + out-dir)"]["changed_flags"]) \
        - {"--out-dir"}
    c_a = set(diffs["C_vs_A (must be EXACTLY --head-range-relax + out-dir)"]["changed_flags"]) \
        - {"--out-dir"}
    exp_ab = {"--distill-field-cache", "--distill-form", "--distill-weight", "--distill-temp",
              "--distill-attack-temp"}
    ok_ab = a_b == exp_ab
    ok_ca = c_a == {"--head-range-relax"}
    report = {"schema": "ddm_dw1_window_seal.v1", "form": args.form, "attack_temp": args.attack_temp,
              "window_epochs_ab": args.window_epochs_ab, "window_epochs_c": args.window_epochs_c,
              "A_vs_B_exactly_distill_flags": ok_ab, "C_vs_A_exactly_head_relax": ok_ca,
              "diffs": diffs, "tickets_dir": str(tickets_dir),
              "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED"}
    (out_root / "window_seal_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"A_vs_B_exactly_distill_flags": ok_ab,
                      "C_vs_A_exactly_head_relax": ok_ca,
                      "A_vs_B_changed": sorted(a_b),
                      "C_vs_A_changed": sorted(c_a),
                      "B_vs_burn_changed": sorted(
                          diffs["B_vs_burn (intended window deltas)"]["changed_flags"])}, indent=2))
    if not (ok_ab and ok_ca):
        print("REFUSE: matched-config diff violated (see window_seal_report.json)", file=sys.stderr)
        return 4
    print(f"SEALED 3 tickets -> {tickets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
