#!/usr/bin/env python3
"""owed-15 / SYNTHESIS_v3_v752 §C item 15 — CLASS-A TRUNK-LEVER ISOLATION LADDER (CONFIGS ONLY).

Author the three FRESH incremental-training isolation arms as COMPILED, VALIDATED, launch-ready
configs (launch.sh files) — but do NOT train them (the multi-hour arms await the operator's
which-to-run GO). Launch-1 composes THREE Class-A levers at ep0; two carry sub-n600 evidence that is
NOT independently n600-confirmed and rides load-bearing in the trunk:

  * #121 d_seg-aware taper — ESTIMATED ~0.03 RANK-1 from ONE under-converged run; "converged flips to
    -8%" is INFERRED, not measured.
  * AA render-IPE — oracle-R lane-recall +0.38 @384; through-training ΔS ASSUMED.
  * self-orient Fourier augmentation — historical -48% n96 direct-partition result is
    UN-REPRODUCED; owed16 n600 warm-start OFF beat ON, so no strong-anchor claim.

R8 LAW (SYNTHESIS §C item 15, CODE-GROUNDED): NEVER inference-toggle a trained-WITH render lever. #121
taper is a trained-WITH curvelet-amplitude reshape; --render-aa ipe reshapes the render path (persisted
as __cfg_render_aa) — the weights ADAPT to the tapered/AA'd render, so toggling either at inference on
ONE composed checkpoint MISMATCHES the render = an INVALID/toy isolation. Each arm must therefore be a
FRESH incremental TRAINING run, trained to the SAME floor, scored at n600 through-R d_seg:

  ARM 1  basis_only   self-orient ON,  taper OFF, --render-aa none   (the incremental baseline)
  ARM 2  plus_taper   self-orient ON,  taper ON,  --render-aa none   (isolates the #121 taper delta)
  ARM 3  plus_aa_ipe  self-orient ON,  taper ON,  --render-aa ipe    (isolates the AA-ipe delta = trunk)

ROLLBACK sign-test: keep a lever IFF its isolated n600 through-R d_seg IMPROVES over the arm below it;
ROLLBACK (drop the lever from the trunk) otherwise — never ship a silently-hurting trunk lever on a
single-run / oracle estimate. $0-local, NON-blocking (the cluster still ships if net-positive).

Base = crucible_v7 (the v7.5.2 substrate; when P7 lands the typed crucible_v752 configs these three arms
become typed WitnessProgram variants — the DSL join point). Every emitted flag is validated against the
trainer's REAL argparse (never-invent-a-flag). CONFIGS ONLY — this tool writes launch.sh files and a
manifest; it NEVER spawns a trainer.

Usage:
    .venv/bin/python tools/build_v752_isolation_arms.py \\
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TAPER_FLAGS = ["--dseg-aware-taper", "--dseg-aware-taper-strength", "1.0",
                "--dseg-aware-taper-scale", "0.0", "--dseg-aware-taper-floor", "0.05"]

_ARMS = [
    {"arm": "arm1_basis_only", "aa": "none", "taper": False,
     "isolates": "the incremental baseline (self-orient directional basis alone)"},
    {"arm": "arm2_plus_taper", "aa": "none", "taper": True,
     "isolates": "the #121 d_seg-aware taper delta (vs arm1)"},
    {"arm": "arm3_plus_aa_ipe", "aa": "ipe", "taper": True,
     "isolates": "the AA render-IPE delta (vs arm2) = the full launch-1 trunk"},
]

_RENDER_AA_RE = re.compile(r"--render-aa\s+\S+")
_FLAG_RE = re.compile(r"(--[a-z0-9][a-z0-9-]*)")


def _validate_flags(text: str, real_flags: frozenset[str]) -> list[str]:
    """Return any emitted --flag token NOT in the trainer's real argparse (never-invent-a-flag)."""
    seen = {m.group(1) for m in _FLAG_RE.finditer(text)}
    return sorted(f for f in seen if f not in real_flags)


def build_arms(gt_cache: str, num_pairs: int, out_root: Path) -> dict:
    """Emit + validate each arm's launch.sh under out_root/<arm>/ (NOT executed). Returns a manifest."""
    import launch_witness_run as L

    real_flags = L.real_trainer_flags()
    arms_out: list[dict] = []
    for spec in _ARMS:
        arm_dir = out_root / spec["arm"]
        cfg = L.derive_named_config("crucible_v7", gt_cache, num_pairs=num_pairs, epochs=None,
                                    overfit=True)
        extra = list(_TAPER_FLAGS) if spec["taper"] else None
        launch = L.write_launch_sh(cfg, arm_dir, extra_flags=extra)
        text = Path(launch).read_text()
        # Set the render-aa mode by REPLACEMENT (crucible_v7 base emits --render-aa ipe; a dup-append
        # would be refused by the C13 duplicate-long-flag guard — so we rewrite the value in place).
        new_text, n_sub = _RENDER_AA_RE.subn(f"--render-aa {spec['aa']}", text)
        if n_sub:
            tmp = Path(launch).with_suffix(f".sh.tmp.{spec['arm']}")
            tmp.write_text(new_text)
            tmp.chmod(0o755)
            tmp.replace(launch)
            text = new_text
        invented = _validate_flags(text, real_flags)
        # confirm the intended lever state is actually present
        aa_ok = f"--render-aa {spec['aa']}" in text
        taper_present = "--dseg-aware-taper" in text.split()
        self_orient = "--self-orient" in text.split()
        ok = (not invented) and aa_ok and (taper_present == spec["taper"]) and self_orient
        arms_out.append({
            "arm": spec["arm"], "isolates": spec["isolates"],
            "launch_sh": str(launch),
            "render_aa": spec["aa"], "taper": spec["taper"], "self_orient": self_orient,
            "flags_all_valid": (not invented), "invented_flags": invented,
            "lever_state_ok": ok,
            "trained": False,
        })

    all_ok = all(a["lever_state_ok"] for a in arms_out)
    manifest = {
        "gate": "owed-15 Class-A trunk-lever isolation ladder (CONFIGS ONLY)",
        "base_config": "crucible_v7 (v7.5.2 substrate; P7 typed crucible_v752 = the DSL join point)",
        "num_pairs": num_pairs, "gt_cache": gt_cache,
        "r8_law": "FRESH incremental TRAINING arms; NEVER inference-toggle a trained-WITH render lever "
                  "(#121 taper + --render-aa reshape the render path → the weights adapt → a one-ckpt "
                  "toggle mismatches the render = a toy isolation).",
        "rollback_sign_test": "keep a lever IFF its isolated n600 through-R d_seg IMPROVES over the arm "
                              "below it; else ROLLBACK from the trunk.",
        "run_order": "arm1_basis_only -> arm2_plus_taper -> arm3_plus_aa_ipe (each FRESH to the SAME "
                     "floor; NON-blocking — launch-1 still ships if net-positive; do NOT train until "
                     "the operator's which-to-run GO).",
        "arms": arms_out, "all_arms_valid": all_ok,
        "note": "CONFIGS ONLY — launch.sh emitted + argparse-validated (never-invent-a-flag); NOT trained.",
        "axis": "[macOS-MLX advisory] NON-PROMOTABLE (MEANS; pointer 0.19110 UNMOVED)",
    }
    return manifest


def format_manifest(m: dict) -> str:
    lines = [f"owed-15 CLASS-A ISOLATION LADDER — base={m['base_config']}", "",
             f"  R8 law: {m['r8_law']}", f"  rollback: {m['rollback_sign_test']}",
             f"  run order: {m['run_order']}", ""]
    for a in m["arms"]:
        mark = "OK " if a["lever_state_ok"] else "!! "
        lines.append(f"  [{mark}] {a['arm']}: self-orient={a['self_orient']} taper={a['taper']} "
                     f"render-aa={a['render_aa']}  (isolates {a['isolates']})")
        lines.append(f"         launch.sh: {a['launch_sh']}  valid={a['flags_all_valid']} trained=False")
        if a["invented_flags"]:
            lines.append(f"         INVENTED FLAGS: {a['invented_flags']}")
    lines.append("")
    lines.append(f"  ALL ARMS VALID: {'YES' if m['all_arms_valid'] else 'NO'}")
    lines.append(f"  {m['note']}")
    lines.append(f"  {m['axis']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--out-root", default="experiments/results/v752_isolation_arms",
                    help="where the arm launch.sh files are written (NOT executed)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = _REPO / out_root
    m = build_arms(args.gt_cache, args.num_pairs, out_root)
    (out_root / "manifest.json").write_text(json.dumps(m, indent=2))
    if args.json:
        print(json.dumps(m, indent=2))
    else:
        print(format_manifest(m))
        print(f"\n  manifest: {out_root / 'manifest.json'}")
    return 0 if m["all_arms_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
