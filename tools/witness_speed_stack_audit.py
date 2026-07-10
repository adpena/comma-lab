#!/usr/bin/env python3
"""owed-4 / SYNTHESIS_v3_v752 §C item 3 — SPEED-STACK COMPOSITION + WALL-CLOCK BUDGET.

Two deliverables, both from the SYNTHESIS §B `speed:` + `wall_clock:` blocks:

  (A) SPEED-STACK AUDIT — verify every score-neutral speed lever the sealed doc names is ON in the
      compiled config (fused-R bit-exact · grouped-backward ~17× · safe-compile fingerprint-certified ·
      async-verdict) OR excluded-with-measured-reason (micro-batch-pairs, #313 batch-dependence). Each
      row carries its per-lever NEUTRALITY RECEIPT (why it does not change the score) so a reviewer can
      audit composition against the doc, not a memory.

  (B) WALL-CLOCK BUDGET TABLE — the §B.wall_clock per-stage budget = MEASURED sec/ep (from the owed-2
      dry-start report, else the 42 s/ep MLX-local anchor) × the DERIVED per-stage floor epochs
      (clamp[150,400]) + the ~11 min GPU head-solve. This is the v7.5.2 half of the #385 dual-chain
      comparison brief (SYNTHESIS §D).

Score-neutral, `[macOS-MLX advisory]` — MEANS, not a score (pointer 0.19110 UNMOVED). READ-ONLY.

Usage:
    .venv/bin/python tools/witness_speed_stack_audit.py --config crucible_v7 \\
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \\
        [--dry-start-report <dir>/dry_start_report.json] [--json]
    .venv/bin/python tools/witness_speed_stack_audit.py --launch-sh <run>/launch.sh [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# SYNTHESIS §B.wall_clock: the MLX-local M5 Max anchor + the DERIVED per-stage floor bracket.
SEC_PER_EP_ANCHOR = 42.0
STAGE_FLOOR_MIN, STAGE_FLOOR_NOM, STAGE_FLOOR_MAX = 150, 250, 400
N_MAIN_STAGES = 3            # CE · tau · Muon (S6 J1–J4)
POSE_EP_MIN, POSE_EP_NOM, POSE_EP_MAX = 50, 100, 150   # §B.wall_clock stage_4bc_pose
HEAD_SOLVE_GPU_MIN = 11.0    # §B.wall_clock stage_4a_solve (~11 min GPU / ~3 h CPU; SOLVE not train)


# --------------------------------------------------------------------------- speed lever audit --
def _has_token(text: str, flag: str) -> bool:
    return flag in text.split()


def _flag_value(text: str, flag: str) -> str | None:
    toks = text.split()
    if flag in toks:
        i = toks.index(flag)
        if i + 1 < len(toks):
            return toks[i + 1]
    return None


def audit_speed_levers(launch_text: str) -> list[dict]:
    """Audit the emitted launch.sh against the SYNTHESIS §B speed block. Each lever: status ∈
    {ON, EXCLUDED_WITH_REASON, BYTE_IDENTICAL_OFF, MISSING} + its neutrality receipt. PURE."""
    levers: list[dict] = []

    # 1. fused-R kernel — bit-exact fixed-order VJP.
    on = _has_token(launch_text, "--fused-r-kernel")
    levers.append({
        "lever": "fused_r_kernel", "flag": "--fused-r-kernel",
        "status": "ON" if on else "MISSING",
        "expected": "ON", "ok": on,
        "neutrality_receipt": "BIT-EXACT: fixed-order VJP → cross-process determinism (MEMORY L70, "
                              "#348 28/28 wall closed) + ~8% faster; not a score change.",
    })

    # 2. grouped-backward — env perf prefix (the ~17× fast path).
    genv = "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in launch_text
    levers.append({
        "lever": "grouped_backward", "flag": "env TAC_MLX_CUSTOM_GROUPED_BACKWARD=1",
        "status": "ON" if genv else "MISSING",
        "expected": "ON", "ok": genv,
        "neutrality_receipt": "BIT-EXACT VJP reorder; ~17× backward (MEMORY L45). Compute-identical by "
                              "construction — perf-env guard (launcher step b-perf) refuses a launch that "
                              "drops it.",
    })

    # 3. safe-compile-regions — fingerprint-certified regions (default 'none' = byte-identical).
    sc = _flag_value(launch_text, "--safe-compile-regions")
    sc_norm = (sc or "none").strip().lower()
    sc_off = sc_norm in ("none", "off", "")
    levers.append({
        "lever": "safe_compile_regions", "flag": "--safe-compile-regions",
        "value": sc, "status": "BYTE_IDENTICAL_OFF" if sc_off else "ON",
        # either state is neutral: OFF = byte-identical; ON = fingerprint-certified on THIS chip
        "expected": "certified region OR none (both score-neutral)", "ok": True,
        "neutrality_receipt": ("per-chip fingerprint-CERTIFIED region (launcher step b2 refuses a stale/"
                               "absent manifest); default 'none' is byte-identical."
                               if not sc_off else "byte-identical (compile OFF)."),
    })

    # 4. async-verdict — neutral by construction (verdict off the training thread, advisory-only).
    av = _has_token(launch_text, "--async-verdict")
    levers.append({
        "lever": "async_verdict", "flag": "--async-verdict",
        "status": "ON" if av else "MISSING",
        "expected": "ON", "ok": av,
        "neutrality_receipt": "NEUTRAL BY CONSTRUCTION: the verdict runs off the training thread "
                              "(GIL-released) and is ADVISORY-only — the trained weights never read it.",
    })

    # 5. micro-batch-pairs — EXCLUDED-WITH-REASON (#313 batch-dependence).
    mbp = _flag_value(launch_text, "--micro-batch-pairs")
    # trainer default is 1 (== off); EXCLUDED means absent or ==1
    excluded = (mbp is None or str(mbp).strip() == "1")
    levers.append({
        "lever": "micro_batch_pairs", "flag": "--micro-batch-pairs",
        "value": mbp, "status": "EXCLUDED_WITH_REASON" if excluded else "PRESENT_NONDEFAULT",
        "expected": "EXCLUDED (default 1)", "ok": excluded,
        "neutrality_receipt": "EXCLUDED (SYNTHESIS §B / W-9): #313 batch-DEPENDENCE (2.26e-2 drift / 11 "
                              "argmax flips) makes bit-identity-at-speedup IMPOSSIBLE → NOT a neutral "
                              "speed lever. Admission = a bounded n600 A/B (a SCORE decision), never a "
                              "neutrality proof.",
    })
    return levers


# --------------------------------------------------------------------------- wall-clock budget --
def sec_per_ep_from_report(report_path: Path) -> tuple[float | None, str]:
    """(sec_per_ep, provenance) from an owed-2 dry_start_report.json — prefers the MARGINAL measure
    (closer to steady-state); falls back to gross. Returns (None, ...) when absent/unmeasured."""
    try:
        rep = json.loads(Path(report_path).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None, "no dry-start report"
    m = rep.get("sec_per_ep_marginal")
    g = rep.get("sec_per_ep_gross")
    if isinstance(m, (int, float)) and m > 0:
        return float(m), f"MEASURED marginal (dry-start {rep.get('config')}, {rep.get('num_pairs')} pairs)"
    if isinstance(g, (int, float)) and g > 0:
        return float(g), f"MEASURED gross-upper-bound (dry-start {rep.get('config')})"
    return None, "dry-start report has no positive sec/ep"


def wall_clock_budget(sec_per_ep: float) -> dict:
    """The §B.wall_clock per-stage budget table from a sec/ep. 3 main stages × clamp[150,400] floor +
    ~11 min GPU head-solve + a 50–150-ep pose finish + polyak backstop (~0). PURE."""
    def _hours(ep: int) -> float:
        return round(ep * sec_per_ep / 3600.0, 2)

    stages = []
    for name in ("stage_1_CE", "stage_2_tau", "stage_3_Muon"):
        stages.append({
            "stage": name, "floor_ep": [STAGE_FLOOR_MIN, STAGE_FLOOR_NOM, STAGE_FLOOR_MAX],
            "wall_h": [_hours(STAGE_FLOOR_MIN), _hours(STAGE_FLOOR_NOM), _hours(STAGE_FLOOR_MAX)],
            "exit": "event (decoupling/plateau) — floors are the DERIVED lower bound, not the target",
        })
    stages.append({
        "stage": "stage_4a_head_solve", "floor_ep": "— (SOLVE, not train)",
        "wall_h": round(HEAD_SOLVE_GPU_MIN / 60.0, 3), "wall_note": "~11 min GPU / ~3 h CPU",
        "exit": "ρ-gated GN/CG of the ~791-param affine head; DELETES a terminal fine-tune stage",
    })
    stages.append({
        "stage": "stage_4bc_pose", "floor_ep": [POSE_EP_MIN, POSE_EP_NOM, POSE_EP_MAX],
        "wall_h": [_hours(POSE_EP_MIN), _hours(POSE_EP_NOM), _hours(POSE_EP_MAX)],
        "exit": "conditioning-gate window; else 0 (ship banked R1)",
    })
    stages.append({"stage": "stage_5_polyak", "floor_ep": "backstop only", "wall_h": 0.0,
                   "exit": "fail-safe; ~0 typical"})

    train_ep_lo = N_MAIN_STAGES * STAGE_FLOOR_MIN + POSE_EP_MIN
    train_ep_nom = N_MAIN_STAGES * STAGE_FLOOR_NOM + POSE_EP_NOM
    train_ep_hi = N_MAIN_STAGES * STAGE_FLOOR_MAX + POSE_EP_MAX
    head_h = HEAD_SOLVE_GPU_MIN / 60.0
    return {
        "sec_per_ep": round(sec_per_ep, 2),
        "stages": stages,
        "total_train_epochs": {"lo": train_ep_lo, "nominal": train_ep_nom, "hi": train_ep_hi},
        "total_wall_h": {
            "lo": round(train_ep_lo * sec_per_ep / 3600.0 + head_h, 2),
            "nominal": round(train_ep_nom * sec_per_ep / 3600.0 + head_h, 2),
            "hi": round(train_ep_hi * sec_per_ep / 3600.0 + head_h, 2),
        },
        "head_solve_gpu_min": HEAD_SOLVE_GPU_MIN,
        "design_wins": "event-exits SAVE epochs vs floors; solve-replaces-train DELETES a stage; "
                       "FRESH-better-config beats warm-recovery (SYNTHESIS §B design_wins)",
    }


# --------------------------------------------------------------------------- driver --
def _emit_launch_text(config: str, gt_cache: str, num_pairs: int) -> str:
    """Emit the compiled config's launch.sh text (with the perf-env prefix) for auditing, via the
    launcher's own writer (never-invent-a-flag; the SAME bytes a real launch would run)."""
    import launch_witness_run as L
    cfg = L.derive_named_config(config, gt_cache, num_pairs=num_pairs, epochs=None, overfit=True)
    tmp = _REPO / ".omx" / "tmp" / f"speed_audit_{config}"
    launch = L.write_launch_sh(cfg, tmp)
    return Path(launch).read_text()


def run_audit(launch_text: str, sec_per_ep: float, sec_prov: str, *, config: str | None) -> dict:
    levers = audit_speed_levers(launch_text)
    budget = wall_clock_budget(sec_per_ep)
    all_ok = all(v["ok"] for v in levers)
    return {
        "gate": "owed-4 speed-stack composition + wall-clock budget",
        "config": config, "speed_levers": levers, "speed_stack_composed_ok": all_ok,
        "sec_per_ep_provenance": sec_prov,
        "wall_clock_budget": budget,
        "feeds": "SYNTHESIS §D #385 dual-chain comparison brief (the v7.5.2 half)",
        "axis": "[macOS-MLX advisory] NON-PROMOTABLE (score-neutral MEANS; pointer 0.19110 UNMOVED)",
    }


def format_report(rep: dict) -> str:
    lines = [f"SPEED-STACK AUDIT + WALL-CLOCK BUDGET — config={rep['config']}", ""]
    lines.append("  SPEED LEVERS (SYNTHESIS §B):")
    for v in rep["speed_levers"]:
        mark = "OK " if v["ok"] else "!! "
        val = f"={v.get('value')}" if v.get("value") is not None else ""
        lines.append(f"    [{mark}] {v['lever']}{val}: {v['status']} (expect {v['expected']})")
        lines.append(f"           receipt: {v['neutrality_receipt']}")
    lines.append(f"  → speed stack composed: {'OK' if rep['speed_stack_composed_ok'] else 'INCOMPLETE'}")
    lines.append("")
    b = rep["wall_clock_budget"]
    lines.append(f"  WALL-CLOCK BUDGET  (sec/ep = {b['sec_per_ep']} — {rep['sec_per_ep_provenance']}):")
    for s in b["stages"]:
        lines.append(f"    {s['stage']:<20} floor_ep={s['floor_ep']}  wall_h={s['wall_h']}")
    t = b["total_wall_h"]
    lines.append(f"    {'TOTAL':<20} train_ep={b['total_train_epochs']}  "
                 f"wall_h lo/nom/hi = {t['lo']} / {t['nominal']} / {t['hi']}  (+~11min GPU head-solve)")
    lines.append("")
    lines.append(f"  feeds: {rep['feeds']}")
    lines.append(f"  {rep['axis']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--launch-sh", help="audit an already-emitted launch.sh")
    src.add_argument("--config", help="emit + audit a named config (crucible_v7, …)")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--dry-start-report", default=None,
                    help="owed-2 dry_start_report.json (dir or file) for the MEASURED sec/ep; absent "
                    f"-> the {SEC_PER_EP_ANCHOR} s/ep MLX-local anchor")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="also write the JSON report to this path")
    args = ap.parse_args(argv)

    if args.launch_sh:
        p = Path(args.launch_sh)
        if not p.is_absolute():
            p = _REPO / p
        launch_text = p.read_text()
        config = "launch.sh"
    else:
        launch_text = _emit_launch_text(args.config, args.gt_cache, args.num_pairs)
        config = args.config

    sec_per_ep, sec_prov = SEC_PER_EP_ANCHOR, f"{SEC_PER_EP_ANCHOR} s/ep MLX-local M5 Max anchor (§B)"
    if args.dry_start_report:
        rp = Path(args.dry_start_report)
        if rp.is_dir():
            rp = rp / "dry_start_report.json"
        m, prov = sec_per_ep_from_report(rp)
        if m is not None:
            sec_per_ep, sec_prov = m, prov

    rep = run_audit(launch_text, sec_per_ep, sec_prov, config=config)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(format_report(rep))
    return 0 if rep["speed_stack_composed_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
