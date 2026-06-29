#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DM1 decisive-smoke EXECUTABLE verdict harvester (recursive-review M1 + M2).

Reads the four DM1 smoke arms' JSONL logs and emits a MACHINE-READABLE verdict
GO / SYMPTOM / MEANS_FALSIFIED / VOID — so the firewall is a RUNNABLE program, not
prose left to the eye (review M1). The PR bar is SELF-CALIBRATING from the A0
baseline's measured trajectory in the faithful regime, NOT a hardcoded l7-era
constant (review M2).

THE FIREWALL (means != ends), made executable
---------------------------------------------
  * DISEASE-PRESENT PRECONDITION (else VOID): the A0 BASELINE must REPRODUCE the
    disease in this window — its (deployed/shadow) PR(M) must DROP meaningfully
    AND its advisory d_seg must DEGRADE vs its first measured value. If A0 does
    NOT reproduce the disease, NEITHER GO nor SYMPTOM is valid (the lever was
    never tested against the pathology) => VOID.
  * SELF-CALIBRATING PR BAR (review M2): bar = a0_pr_end + hold_frac*(pr_start -
    a0_pr_end) — i.e. A3 must RECOVER at least ``hold_frac`` (default 0.5) of the
    A0 collapse. Derived from A0's MEASURED endpoints in the faithful regime, not
    the hardcoded ~3.0 / ~1.2 (an l7-era 1500-epoch endpoint unreachable in the
    100-epoch tau window).
  * GO          = A3 HOLDS PR (>= bar) AND A3 d_seg is >= ``dseg_gain_frac``
    (default 10%) lower than A0's end d_seg. The cure fixes the MEANS *and* moves
    the END. Pursue the optimizer program.
  * SYMPTOM     = A3 HOLDS PR (>= bar) but d_seg does NOT improve. We were
    treating a SYMPTOM; DM1 is not the binding d_seg cause. Pivot to the real
    d_seg lever. (A MAJOR finding either way.)
  * MEANS_FALSIFIED = disease present but A3 does NOT hold PR (< bar). The
    Stiefel+entropy cure is FALSIFIED at implementation; re-open the cure math.
  * VOID        = disease not reproduced by A0 (precondition failed). Re-run the
    smoke faithful to the disease regime (anneal-epochs / lane confound / regime).

The verdict reads the SHADOW (DEPLOYED) PR (``pr_film_M_shadow``) when present —
the EMA shadow is what ships, and an EMA of orthonormal matrices is not itself
orthonormal (review Med1) — falling back to the live ``pr_film_M`` only when the
shadow field is absent (older logs).

Advisory only: d_seg here is ``[macOS-MLX research-signal]`` (NEVER a contest
score). This harvester makes NO score claim; it adjudicates a MEANS-vs-ENDS A/B.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# canonical arm directory names (created by experiments/results/dm1_decisive_smoke/ready_to_fire.sh)
_ARMS = ("A0_baseline", "A1_stiefel", "A2_entropy", "A3_minimal")
_LOG_NAME = "daemon.log"


def _iter_json_rows(log_path: Path):
    """Yield each JSON object on its own line in ``log_path`` (tolerant of daemon
    prefixes / non-JSON lines). The trainer prints one compact JSON dict per row to
    stdout, which the durable daemon appends verbatim to the log."""
    if not log_path.exists():
        return
    for raw in log_path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        # a daemon may prefix a line; locate the first '{' and try from there.
        start = line.find("{")
        if start < 0:
            continue
        frag = line[start:]
        try:
            obj = json.loads(frag)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict) and "stage" in obj:
            yield obj


def _pr_of(row: dict[str, Any]) -> float | None:
    """The DEPLOYED PR to gate on: shadow PR (what ships) when present, else live."""
    v = row.get("pr_film_M_shadow")
    if v is None:
        v = row.get("pr_film_M")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def parse_arm(log_path: Path) -> dict[str, Any]:
    """Extract the PR + d_seg trajectory endpoints from one arm's log."""
    tele: list[tuple[int, float]] = []   # (epoch, deployed PR)
    verd: list[tuple[int, float]] = []   # (epoch, d_seg)
    for row in _iter_json_rows(log_path):
        st = row.get("stage")
        ep = row.get("epoch")
        if ep is None:
            continue
        try:
            ep = int(ep)
        except (TypeError, ValueError):
            continue
        if st == "dm1_telemetry":
            pr = _pr_of(row)
            if pr is not None:
                tele.append((ep, pr))
        elif st == "verdict":
            ds = row.get("d_seg")
            if ds is not None:
                try:
                    verd.append((ep, float(ds)))
                except (TypeError, ValueError):
                    pass
    tele.sort(key=lambda t: t[0])
    verd.sort(key=lambda t: t[0])
    out: dict[str, Any] = {
        "log": str(log_path), "exists": log_path.exists(),
        "n_telemetry": len(tele), "n_verdict": len(verd),
    }
    if tele:
        out["pr_start_epoch"], out["pr_start"] = tele[0]
        out["pr_end_epoch"], out["pr_end"] = tele[-1]
    if verd:
        out["dseg_start_epoch"], out["dseg_start"] = verd[0]
        out["dseg_end_epoch"], out["dseg_end"] = verd[-1]
    return out


def adjudicate(
    arms: dict[str, dict[str, Any]],
    *,
    hold_frac: float = 0.5,
    disease_pr_drop_frac: float = 0.15,
    dseg_gain_frac: float = 0.10,
) -> dict[str, Any]:
    """Compute the GO / SYMPTOM / MEANS_FALSIFIED / VOID verdict from the parsed arms.

    Pure function (no I/O) so it is unit-testable on synthetic arm dicts.
    """
    a0 = arms.get("A0_baseline", {})
    a3 = arms.get("A3_minimal", {})
    reasons: list[str] = []

    # completeness: A0 + A3 must have BOTH telemetry (PR) and verdict (d_seg) endpoints.
    need = ("pr_start", "pr_end", "dseg_start", "dseg_end")
    miss_a0 = [k for k in need if k not in a0]
    miss_a3 = [k for k in need if k not in a3]
    if miss_a0 or miss_a3:
        return {
            "verdict": "INCOMPLETE",
            "reason": f"missing A0{miss_a0} / A3{miss_a3} endpoints (arms not finished or no rows)",
            "arms": arms,
        }

    pr_start = float(a0["pr_start"])            # shared warm-start PR (all arms resume the SAME ckpt)
    a0_pr_end = float(a0["pr_end"])
    a3_pr_end = float(a3["pr_end"])
    a0_dseg_start = float(a0["dseg_start"])
    a0_dseg_end = float(a0["dseg_end"])
    a3_dseg_end = float(a3["dseg_end"])

    # --- disease-present precondition (else VOID) ---
    pr_drop = (pr_start - a0_pr_end) / pr_start if pr_start > 0 else 0.0
    pr_collapsed = pr_drop >= disease_pr_drop_frac
    dseg_degraded = a0_dseg_end > a0_dseg_start
    if not (pr_collapsed and dseg_degraded):
        reasons.append(
            f"A0 PR drop {pr_drop:.3f} (need >= {disease_pr_drop_frac}) collapsed={pr_collapsed}; "
            f"A0 d_seg {a0_dseg_start:.5f}->{a0_dseg_end:.5f} degraded={dseg_degraded}")
        return {
            "verdict": "VOID",
            "reason": "disease NOT reproduced by A0 in this window (re-run faithful to the regime); "
                      + "; ".join(reasons),
            "pr_start": pr_start, "a0_pr_end": a0_pr_end, "a0_pr_drop_frac": pr_drop,
            "a0_dseg_start": a0_dseg_start, "a0_dseg_end": a0_dseg_end,
            "arms": arms,
        }

    # --- self-calibrating PR bar (review M2) ---
    bar = a0_pr_end + hold_frac * (pr_start - a0_pr_end)
    a3_holds_pr = a3_pr_end >= bar

    # --- d_seg gain test ---
    a3_dseg_target = (1.0 - dseg_gain_frac) * a0_dseg_end
    a3_improves_dseg = a3_dseg_end <= a3_dseg_target

    if not a3_holds_pr:
        verdict = "MEANS_FALSIFIED"
        reason = (f"disease present, but A3 PR_end {a3_pr_end:.3f} < bar {bar:.3f} "
                  f"(A0 collapse {pr_start:.3f}->{a0_pr_end:.3f}); Stiefel+entropy FALSIFIED at "
                  "implementation -> re-open the cure math")
    elif a3_improves_dseg:
        verdict = "GO"
        reason = (f"A3 holds PR ({a3_pr_end:.3f} >= bar {bar:.3f}) AND lowers d_seg "
                  f"{a0_dseg_end:.5f}->{a3_dseg_end:.5f} (>= {dseg_gain_frac:.0%} rel); the cure fixes "
                  "MEANS and moves the END -> pursue the optimizer program")
    else:
        verdict = "SYMPTOM"
        reason = (f"A3 holds PR ({a3_pr_end:.3f} >= bar {bar:.3f}) but d_seg flat "
                  f"({a0_dseg_end:.5f}->{a3_dseg_end:.5f}, target <= {a3_dseg_target:.5f}); DM1 is NOT "
                  "the binding d_seg cause -> pivot to the real d_seg lever (MAJOR finding)")

    return {
        "verdict": verdict, "reason": reason,
        "pr_start": pr_start, "pr_bar": bar, "hold_frac": hold_frac,
        "a0_pr_end": a0_pr_end, "a3_pr_end": a3_pr_end, "a3_holds_pr": a3_holds_pr,
        "a0_pr_drop_frac": pr_drop,
        "a0_dseg_start": a0_dseg_start, "a0_dseg_end": a0_dseg_end,
        "a3_dseg_end": a3_dseg_end, "a3_dseg_target": a3_dseg_target,
        "a3_improves_dseg": a3_improves_dseg, "dseg_gain_frac": dseg_gain_frac,
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-dir", type=Path,
                    default=Path("experiments/results/dm1_decisive_smoke"),
                    help="dir containing the A0_baseline/A1_stiefel/A2_entropy/A3_minimal arm subdirs.")
    ap.add_argument("--hold-frac", type=float, default=0.5,
                    help="(M2) fraction of the A0 PR collapse A3 must RECOVER to count as 'holds PR' "
                    "(bar = a0_pr_end + hold_frac*(pr_start - a0_pr_end)). Default 0.5.")
    ap.add_argument("--disease-pr-drop-frac", type=float, default=0.15,
                    help="min relative A0 PR drop (start->end) to count the disease as reproduced "
                    "(else VOID). Default 0.15 (15%%).")
    ap.add_argument("--dseg-gain-frac", type=float, default=0.10,
                    help="min relative A3 d_seg improvement vs A0_end for GO (else SYMPTOM). Default 0.10.")
    ap.add_argument("--json", action="store_true", help="emit the full machine-readable verdict JSON.")
    args = ap.parse_args(argv)

    arms = {name: parse_arm(args.smoke_dir / name / _LOG_NAME) for name in _ARMS}
    result = adjudicate(
        arms, hold_frac=args.hold_frac,
        disease_pr_drop_frac=args.disease_pr_drop_frac, dseg_gain_frac=args.dseg_gain_frac)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"DM1 SMOKE VERDICT: {result['verdict']}")
        print(f"  {result['reason']}")
        for name in _ARMS:
            a = arms[name]
            pr = f"PR {a.get('pr_start','?')}@{a.get('pr_start_epoch','?')}->{a.get('pr_end','?')}@{a.get('pr_end_epoch','?')}"
            ds = f"d_seg {a.get('dseg_start','?')}->{a.get('dseg_end','?')}"
            tag = "" if a.get("exists") else " [NO LOG]"
            print(f"  {name:14s} {pr:42s} {ds}{tag}")
    # rc convention: 0 GO, 3 SYMPTOM, 4 MEANS_FALSIFIED, 5 VOID, 2 INCOMPLETE (so a caller can branch).
    return {"GO": 0, "SYMPTOM": 3, "MEANS_FALSIFIED": 4, "VOID": 5, "INCOMPLETE": 2}.get(
        result["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
