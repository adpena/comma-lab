#!/usr/bin/env python
"""$0-LIGHT n24 STEP-DIAGNOSTIC for the ep100-resume stepping instability (STEP-INSTAB-DIAG).

THE INCIDENT (3 strikes, measured at n600): every resume of the ep100 snapshot
(experiments/results/bd_calib_20260705/snap/resume_state_ep100.npz) reaches a ~5x-elevated loss
state within ~1 epoch of optimizer steps and the (correct) spike guard then deadlocks:
  * v1 20260705T015247Z: organic ep92 jump 6-8.5 -> ~58 (gnorm ~300), guard lockout ep92-100+;
  * v2 20260705T083453Z (resume + bd 0.2, fresh seed @0.70 compose): ep101 ~40-level, ep102 ~199;
  * v3 20260705T095728Z (same + seed compose == 0): ep101 accepted 75/75, ep102+ ~238->250.

THIS PROBE separates the candidate mechanisms on gt_n24 (pairs 0..23 of the n600 set; 203MB)
with the #304-item-4 per-term loss telemetry ON (TAC_LOSS_TERM_PROBE=1) and the spike guard
DISARMED (--spike-factor 1e9) so the evolution is OBSERVED instead of deadlocked:

  arm baseline_v3    : v3 semantics (bd 0.2, seed-anneal-epochs 101 => seed compose 0, Adam
                       moments RESTORED, lr as-scheduled ~9.1e-4) -- does the per-term explosion
                       reproduce, and WHICH term carries it?
  arm fresh_moments  : same, but the sliced snapshot DROPS the optimizer state (fresh AdamW
                       moments) -- if stable, restored-stale-moments is the mechanism.
  arm low_lr         : same as baseline but lr x0.1 -- if stable, step-size/basin-sharpness.
  arm no_bd          : same as baseline but boundary-distance weight 0 -- if stable, the bd term
                       (or its interaction) destabilizes; if it still explodes the instability
                       predates the bd intervention (and v1's organic ep92 was the same thing).
  arm v1_pure        : the UNCHANGED v1 config resumed (no bd, seed-anneal-epochs 300 => fresh
                       structured seed at the v1 compose weight) -- the pure-resume control that
                       isolates the resume-state reconstruction gap (the trained island SEED is
                       NOT persisted in the sidecar; every resume loses it).

AXIS / NO-FAKE: every number here is [n24 advisory -- mechanism probe, NOT n600 evidence]. The
probe reuses the trainer's OWN resume path (subprocess on the real entry point; nothing
reimplemented); the snapshot is sliced per-pair (rows 0..23) which is FAITHFUL for these pairs
(same weights, same per-pair moments). pointer 0.19110 UNMOVED (this is means, not the goal).

Foreground-only; no daemons; ~5-8 min/arm on MLX-GPU. Usage:
    .venv/bin/python experiments/probe_resume_stepping_instability.py \
        --out-dir experiments/results/stepping_instab_diag_20260705 [--arms baseline_v3,no_bd]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
SNAPSHOT = REPO / "experiments/results/bd_calib_20260705/snap/resume_state_ep100.npz"
GT_N24 = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n24.npz"

N600 = 600  # the snapshot's pair count (per-pair arrays keyed by first-dim 600 / 1200)


def slice_snapshot(src: Path, dst: Path, n_pairs: int, *, drop_opt: bool) -> dict:
    """Slice the n600 resume sidecar down to pairs [0..n_pairs) by FIRST-DIM matching:
    dim0==600 -> [:n]; dim0==1200 -> [:2n] (code rows are 2-per-pair). Everything else is copied
    verbatim. __cl_* closed-loop keys are DROPPED (the probe runs without --closed-loop-control).
    ``drop_opt`` omits every optP__ key + zeroes __resume_has_opt (the fresh-moments arm).
    Faithful for the kept pairs: same weights, same per-pair Adam moments."""
    z = np.load(src, allow_pickle=False)
    out: dict[str, np.ndarray] = {}
    stats = {"sliced": 0, "copied": 0, "dropped_cl": 0, "dropped_opt": 0}
    for k in z.files:
        if k.startswith("__cl_"):
            stats["dropped_cl"] += 1
            continue
        if drop_opt and k.startswith("optP__"):
            stats["dropped_opt"] += 1
            continue
        a = z[k]
        if k == "__resume_has_opt" and drop_opt:
            out[k] = np.asarray(0)
            continue
        if a.ndim >= 1 and a.shape[0] == N600:
            out[k] = a[:n_pairs].copy()
            stats["sliced"] += 1
        elif a.ndim >= 1 and a.shape[0] == 2 * N600:
            out[k] = a[: 2 * n_pairs].copy()
            stats["sliced"] += 1
        else:
            out[k] = a
            stats["copied"] += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(dst)
    return stats


def _base_argv(out_dir: Path, snap: Path, *, epochs: int, accum: int, seed_anneal_epochs: int,
               bd_weight: float, lr: float, lr_end: float, tau_start: int, band_start: int,
               persist_warmup: int, extra: list[str] | None = None,
               guard: str = "disarm", eval_every: int = 999) -> list[str]:
    """The v3 launch argv translated to n24 diagnostics: verdicts at the resume point only,
    spike guard DISARMED (observe, don't deadlock), closed-loop/async OFF (no threads), per-term
    telemetry driven by TAC_LOSS_TERM_PROBE=1 in the env.

    (EIK-STAB build 3) ``extra`` appends arm-specific stabilizer flags; ``guard`` selects the
    spike-guard treatment: "disarm" (the original --spike-factor 1e9 observe-don't-deadlock) or
    "rollback" (--spike-guard-mode rollback ARMED at the production --spike-factor 5 — the
    actuator-under-test arms); ``eval_every`` re-enables periodic verdicts so the arbitration can
    read d_seg drift (does the damping DEGRADE boundary sharpness?)."""
    argv = [
        sys.executable, str(TRAINER),
        "--out-dir", str(out_dir),
        "--gt-cache", str(GT_N24),
        "--num-pairs", "24",
        "--mlx-device", "gpu",
        "--seed", "0",
        "--epochs", str(epochs),
        # schedule fidelity at a SHORT probe budget: the lr cosine + hosc-beta + softmax-temp
        # anneals all use --anneal-epochs as denominator (review C2), so the probe's per-epoch
        # schedule VALUES match the n600 runs (epochs=1000) exactly at ep101-104.
        "--anneal-epochs", "1000",
        "--eval-every", str(eval_every),
        "--verdict-pairs", "24",
        "--verdict-batch", "24",
        "--curriculum",
        "--tau-softplus-start-epoch", str(tau_start),
        "--tau-softplus-tau", "0.3",
        "--l7-start-epoch", "1001",
        # Muon finisher flags DROPPED: --muon-start-epoch 726 is validated against --epochs and is
        # inert in the ep101-104 probe window anyway (v3 semantics unchanged for these epochs).
        "--stage-transition-rewarmup-epochs", "20",
        "--stage-transition-rewarmup-floor", "0.1",
        "--stage-transition-rewarmup-shape", "cosine",
        "--stage-transition-reset-moments",
        "--w-seg", "100", "--w-pose", "1.0", "--score-domain-loss",
        "--pose-carrier", "--pose-carrier-residual-mode", "table",
        # skip the startup s_t self-calibration: the n600 runs fit s_t on min(fit_pairs,P)=24
        # pairs == EXACTLY pairs 0..23 (this probe's set) and selected 0.044 (v1+v3 logs agree).
        "--pose-carrier-s-t", "0.044",
        "--mod-dim", "19", "--hidden-dim", "96", "--n-hidden", "4",
        "--activation", "hosc", "--hosc-beta", "1.0", "--hosc-beta-end", "5.134",
        "--hosc-beta-anneal", "linear", "--hosc-omega", "1.0", "--siren-init",
        "--softmax-temp-start", "1.0", "--softmax-temp-end", "1.0",
        "--tau-anneal-shape", "geometric",
        "--self-orient", "--n-dir-freqs", "2", "--freq-across", "32", "--freq-along", "4",
        "--reorient-every", "50", "--max-bank-freq", "64",
        "--chroma", "--palette-anchor",
        "--eikonal-weight", "0.05", "--length-weight", "0.001",
        "--render-h", "384", "--render-w", "512", "--render-aa", "none",
        "--lane-render-band",
        "--lane-band-start-epoch", str(band_start),
        "--lane-band-uncertainty-source", "witness",
        "--lane-band-tau", "0.85", "--lane-band-eps", "0.35", "--lane-band-softness", "1.0",
        "--lane-band-dash-forward-max-m", "55.0", "--lane-band-weight", "1.0",
        "--persistence-loss-weight", "1.0", "--persistence-recall-weight", "1.0",
        "--cldice-iters", "5",
        "--persistence-warmup-epochs", str(persist_warmup),
        "--persistence-classes", "auto",
        "--amplify-weight", "1.0", "--amplify-form", "hinge",
        "--amplify-margin-target", "1.0", "--amplify-persist", "inverse_thickness",
        "--island-dilate-px", "1",
        "--structured-init", "--structured-init-include-lane",
        "--lane-prior-phi1", "--lane-prior-phi1-mode", "paint", "--lane-prior-phi1-dash-gate",
        "--accum-pairs", str(accum),
        "--grad-clip", "1.0", "--ema-decay", "0.997",
        "--lr", f"{lr:g}", "--lr-end", f"{lr_end:g}",
        "--weight-decay", "1e-4", "--adam-beta2", "0.999",
        "--ckpt-every", "0",
        "--seed-islands",
        "--eikonal-weight-end", "0.1",
        "--film-stiefel",
        "--witness-alone-island-loss",
        "--seed-anneal-shape", "cosine", "--seed-anneal-epochs", str(seed_anneal_epochs),
        "--resume-from", str(snap),
        "--resume-allow-lever-drift", "--resume-clear-spike-guard",
        "--boundary-distance-weight", f"{bd_weight:g}",
        "--loss-term-log-every", "1",
    ]
    if guard == "rollback":
        # (EIK-STAB build 3) actuator-under-test: guard ARMED at the production factor, rollback
        # mode; window/frac tuned to the n24 cadence (3 accum-chunks/epoch at accum 8 => window 6
        # = a 2-epoch sustained-runaway trigger, matching the measured descend-then-runaway shape).
        argv += ["--spike-factor", "5", "--spike-guard-mode", "rollback",
                 "--spike-rollback-window", "6", "--spike-rollback-frac", "0.5",
                 "--spike-rollback-lr-cut", "0.5", "--spike-rollback-max", "8"]
    else:
        argv += ["--spike-factor", "1e9"]  # DISARM the guard: observe the evolution, don't deadlock
    if extra:
        argv += list(extra)
    return argv


ARMS: dict[str, dict] = {
    # v3 semantics: bd 0.2, seed compose == 0 at ep>=101, moments restored, lr as-scheduled.
    "baseline_v3": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                        tau_start=400, band_start=450, persist_warmup=275),
    # (b) same but Adam moments zeroed at resume (sliced snapshot omits optP__ keys).
    "fresh_moments": dict(drop_opt=True, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                          tau_start=400, band_start=450, persist_warmup=275),
    # (c) moments restored, lr x0.1.
    "low_lr": dict(drop_opt=False, bd=0.2, lr=1e-4, lr_end=1e-5, seed_anneal=101,
                   tau_start=400, band_start=450, persist_warmup=275),
    # (d) the deadlock-escape config minus the new bd term.
    "no_bd": dict(drop_opt=False, bd=0.0, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                  tau_start=400, band_start=450, persist_warmup=275),
    # (e) the UNCHANGED v1 config resumed (pure-resume control; fresh seed at v1 compose weight).
    "v1_pure": dict(drop_opt=False, bd=0.0, lr=1e-3, lr_end=1e-4, seed_anneal=300,
                    tau_start=300, band_start=350, persist_warmup=300),
    # (f-h) lr-threshold BRACKETING arms (calibrate-don't-guess: 9.1e-4 unstable / 9.1e-5 stable
    # per the 5-arm run; locate the eikonal-direction stability threshold between them).
    "lr_5e4": dict(drop_opt=False, bd=0.2, lr=5e-4, lr_end=5e-5, seed_anneal=101,
                   tau_start=400, band_start=450, persist_warmup=275),
    "lr_3e4": dict(drop_opt=False, bd=0.2, lr=3e-4, lr_end=3e-5, seed_anneal=101,
                   tau_start=400, band_start=450, persist_warmup=275),
    "lr_18e5": dict(drop_opt=False, bd=0.2, lr=1.8e-4, lr_end=1.8e-5, seed_anneal=101,
                    tau_start=400, band_start=450, persist_warmup=275),
    # ── (EIK-STAB build 3) ARBITRATION arms: the two candidate eikonal-runaway CURES + the
    # rollback-guard actuator, ALL at the UNSTABLE lr 1e-3 (the exploding baseline) so a STABLE
    # verdict is attributable to the cure, not the step size. StEik weights log-spaced x10;
    # ViscoReg eps at {0.3, 1.0} (the paper starts eps=1 with |grad u| ~ 1 — our margin's scale);
    # constant eps (no anneal) at this short horizon. d_seg drift read via eval_every=10.
    "steik_005": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                      tau_start=400, band_start=450, persist_warmup=275,
                      extra=["--eikonal-steik-weight", "0.05"]),
    "steik_05": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                     tau_start=400, band_start=450, persist_warmup=275,
                     extra=["--eikonal-steik-weight", "0.5"]),
    "steik_5": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                    tau_start=400, band_start=450, persist_warmup=275,
                    extra=["--eikonal-steik-weight", "5.0"]),
    "visco_03": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                     tau_start=400, band_start=450, persist_warmup=275,
                     extra=["--eikonal-viscosity", "0.3"]),
    "visco_1": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                    tau_start=400, band_start=450, persist_warmup=275,
                    extra=["--eikonal-viscosity", "1.0"]),
    "rollback_guard": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                           tau_start=400, band_start=450, persist_warmup=275,
                           guard="rollback"),
    "steik05_rollback": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                             tau_start=400, band_start=450, persist_warmup=275,
                             guard="rollback", extra=["--eikonal-steik-weight", "0.5"]),
    # ── (V6 SYMPOSIUM #317) LONGER-HORIZON re-entry arbitration. The FEED-05v arbitration ran only
    # 120 steps (40 ep) and called visco_03 STABLE; v5 (n600) then re-entered runaway at ~ep109 =
    # ~600-675 optimizer STEPS (75 steps/epoch at accum 8) — PAST the 120-step arbitration horizon.
    # The eikonal runaway is anti-diffusive accumulation PER OPTIMIZER STEP (StEik ill-posedness),
    # so the surrogate must match STEP COUNT: run these at --epochs-past-resume>=200 (>=600 steps).
    # NOTE (measured from _scheduled_eikonal_weight): the 0.05->0.10 eik-weight ramp STEPS at the
    # TAU onset (ep400) — so eik_w is FLAT 0.05 through the entire pre-tau re-entry window; candidate
    # (c) is therefore LOWERING the base weight (a targeted lr-cut on the anti-diffusive term, NOT a
    # flat-vs-anneal choice which is moot pre-tau).
    # (v5 faithful) EXACT v5 eikonal config: ANNEALED visco 0.3 over 1000 ep (the arbitration used
    # CONSTANT 0.3; at 600 steps annealed eps ~= 0.24 vs constant 0.30 — reproduce v5 exactly).
    "visco_03_anneal": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                            tau_start=400, band_start=450, persist_warmup=275,
                            extra=["--eikonal-viscosity", "0.3", "--eikonal-viscosity-anneal", "1000"]),
    # (b) HIGHER fixed viscosity eps=0.5 (constant): eps=0.3 partial-cure, eps=1.0 exploded => probe
    # the mid-window value for a permanently-dominant fixed viscosity.
    "visco_05": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                     tau_start=400, band_start=450, persist_warmup=275,
                     extra=["--eikonal-viscosity", "0.5"]),
    # (c) LOWER base eikonal weight 0.05->0.02 (targeted anti-diffusive lr-cut) alone, NO viscosity.
    "eik002": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                   tau_start=400, band_start=450, persist_warmup=275,
                   extra=["--eikonal-weight", "0.02", "--eikonal-weight-end", "0.02"]),
    # (b+c COMPOUND) visco 0.3 + lower base eik weight 0.02.
    "visco_03_eik002": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                            tau_start=400, band_start=450, persist_warmup=275,
                            extra=["--eikonal-viscosity", "0.3",
                                   "--eikonal-weight", "0.02", "--eikonal-weight-end", "0.02"]),
    # (b+c COMPOUND, stronger) visco 0.5 + lower base eik weight 0.02.
    "visco_05_eik002": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                            tau_start=400, band_start=450, persist_warmup=275,
                            extra=["--eikonal-viscosity", "0.5",
                                   "--eikonal-weight", "0.02", "--eikonal-weight-end", "0.02"]),
    # (a) NORMALIZED StEik n^T H n (V6 #317 build 1a-N): removes the raw |grad m|^2 self-amplification
    # (measured 575x-1431x NO-GO). Anisotropic: damps ONLY the anti-diffusive normal-direction mode,
    # tangential (dash) curvature FREE. The DE-DERIVATION sibling (#318) predicts this is the OPTIMAL
    # cure on the d_seg-drift axis. Weight 0.5 (n^T H n ~ O(1) near-boundary; O(0.01-0.1) mean).
    "steik_norm_05": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                          tau_start=400, band_start=450, persist_warmup=275,
                          extra=["--eikonal-steik-weight", "0.5", "--eikonal-steik-normalized"]),
    # (a COMPOUND with visco) normalized StEik 0.5 (additive normal-damping) + ViscoReg 0.3 (viscous
    # residual): the two structural cures composed — the principled "belt + braces".
    "steik_norm_05_visco03": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                                  tau_start=400, band_start=450, persist_warmup=275,
                                  extra=["--eikonal-steik-weight", "0.5", "--eikonal-steik-normalized",
                                         "--eikonal-viscosity", "0.3"]),
    # (a, lighter weight) normalized StEik 0.1 standalone — probe the weight sensitivity.
    "steik_norm_01": dict(drop_opt=False, bd=0.2, lr=1e-3, lr_end=1e-4, seed_anneal=101,
                          tau_start=400, band_start=450, persist_warmup=275,
                          extra=["--eikonal-steik-weight", "0.1", "--eikonal-steik-normalized"]),
    # NOTE (V6 #317): a DERIVED-CONFIG slot is reserved for the DE-DERIVATION sibling (#318)'s
    # PREDICTED-optimal arm (e.g. adaptive-eps(t) proportional to local sharpness kappa(t)); added
    # when its message arrives, tested at the SAME >=250-step horizon against this set.
}


def _parse_rows(run_log: Path) -> dict:
    terms_rows, verdicts, spikes, rollbacks = [], [], 0, []
    for line in run_log.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        st = row.get("stage")
        if st == "loss_terms":
            terms_rows.append(row)
        elif st == "verdict":
            verdicts.append({k: row.get(k) for k in ("epoch", "d_seg", "d_pose")})
        elif st == "spike_skip":
            spikes += 1
        elif st == "spike_rollback":
            rollbacks.append({k: row.get(k) for k in
                              ("ep", "restored_from_epoch", "lr_before", "lr_after", "lr_scale")})
    return {"terms_rows": terms_rows, "verdicts": verdicts, "spike_skips": spikes,
            "rollbacks": rollbacks}


def _summarize_arm(name: str, parsed: dict, epochs_run: int) -> dict:
    rows = parsed["terms_rows"]
    if not rows:
        return {"arm": name, "status": "NO_TELEMETRY_ROWS", "verdicts": parsed["verdicts"]}
    first, last = rows[0], rows[-1]
    keys = list(first["terms"].keys())
    growth = {}
    for k in keys:
        a, b = float(first["terms"][k]), float(last["terms"][k])
        if abs(a) > 1e-9 or abs(b) > 1e-9:
            growth[k] = {"first": round(a, 4), "last": round(b, 4),
                         "delta": round(b - a, 4),
                         "ratio": (round(b / a, 3) if abs(a) > 1e-9 else None)}
    tot0, tot1 = float(first["total"]), float(last["total"])
    # "explodes" verdict: TROUGH-to-last ratio > 3 (the measured signature is descend-then-runaway:
    # the restored state is already elevated, the optimizer first recovers, THEN a term runs away).
    totals = [float(r["total"]) for r in rows]
    trough = min(totals)
    trough_i = totals.index(trough)
    peak_after = max(totals[trough_i:])
    explodes = (not np.isfinite(peak_after)) or peak_after > 3.0 * max(trough, 1e-9)
    # per-term trough-to-last attribution (which term carries the runaway).
    runaway = {}
    for k in keys:
        series = [float(r["terms"].get(k, 0.0)) for r in rows]
        seg_after = series[trough_i:]
        t_min, t_max = min(seg_after), max(seg_after)
        if t_max > 1.0 and t_max > 3.0 * max(t_min, 1e-9):
            runaway[k] = {"after_trough_min": round(t_min, 3), "after_trough_max": round(t_max, 3),
                          "ratio": round(t_max / max(t_min, 1e-9), 2)}
    dominant = sorted(growth.items(), key=lambda kv: -abs(kv[1]["delta"]))
    return {
        "arm": name,
        "axis": "[n24 advisory -- mechanism probe, NOT n600 evidence]",
        "status": "EXPLODES" if explodes else "STABLE",
        "total_first": round(tot0, 3), "total_last": round(tot1, 3),
        "total_trough": round(trough, 3), "trough_epoch": rows[trough_i]["ep"],
        "peak_after_trough": round(peak_after, 3),
        "trough_to_peak_ratio": round(peak_after / max(trough, 1e-9), 3),
        "runaway_terms": runaway,
        "total_ratio": round(tot1 / max(tot0, 1e-9), 3),
        "n_rows": len(rows), "epochs_run": epochs_run,
        "spike_skips": parsed["spike_skips"],
        "rollbacks": parsed.get("rollbacks", []),
        "dominant_terms_by_delta": [
            {"term": k, **v} for k, v in dominant[:5]
        ],
        "per_term_growth": growth,
        "verdicts": parsed["verdicts"],
        "epoch_trace": [
            {"ep": r["ep"], "accum_batch": r["accum_batch"], "total": r["total"],
             "gnorm": r.get("gnorm"),
             "top_terms": dict(sorted(((k, v) for k, v in r["terms"].items()
                                       if abs(float(v)) > 1e-6),
                                      key=lambda kv: -abs(float(kv[1])))[:4])}
            for r in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/results/stepping_instab_diag_20260705")
    ap.add_argument("--arms", type=str, default=",".join(ARMS.keys()),
                    help="comma-separated arm subset")
    ap.add_argument("--epochs-past-resume", type=int, default=3)
    ap.add_argument("--accum-pairs", type=int, default=8)
    ap.add_argument("--timeout-s", type=int, default=3600, help="per-arm subprocess timeout")
    ap.add_argument("--eval-every", type=int, default=999,
                    help="(EIK-STAB build 3) trainer --eval-every: <999 re-enables periodic "
                    "verdicts so the arbitration reads d_seg drift under the damping terms.")
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not SNAPSHOT.exists():
        print(f"FATAL: snapshot missing at {SNAPSHOT}", file=sys.stderr)
        return 2

    # sliced snapshots (with + without opt state), built ONCE from the READ-ONLY original.
    snap24 = out_dir / "resume_state_ep100_n24.npz"
    snap24_noopt = out_dir / "resume_state_ep100_n24_noopt.npz"
    s1 = slice_snapshot(SNAPSHOT, snap24, 24, drop_opt=False)
    s2 = slice_snapshot(SNAPSHOT, snap24_noopt, 24, drop_opt=True)
    print(json.dumps({"stage": "snapshot_slice", "with_opt": s1, "no_opt": s2}), flush=True)

    epochs = 100 + args.epochs_past_resume
    summaries = []
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        cfg = ARMS[name]
        arm_dir = out_dir / f"arm_{name}"
        snap = snap24_noopt if cfg["drop_opt"] else snap24
        argv_t = _base_argv(arm_dir, snap, epochs=epochs, accum=args.accum_pairs,
                            seed_anneal_epochs=cfg["seed_anneal"], bd_weight=cfg["bd"],
                            lr=cfg["lr"], lr_end=cfg["lr_end"], tau_start=cfg["tau_start"],
                            band_start=cfg["band_start"], persist_warmup=cfg["persist_warmup"],
                            extra=cfg.get("extra"), guard=cfg.get("guard", "disarm"),
                            eval_every=args.eval_every)
        (out_dir / f"arm_{name}.argv.json").write_text(json.dumps(argv_t, indent=1))
        log_path = out_dir / f"arm_{name}.log"
        print(json.dumps({"stage": "arm_start", "arm": name, "log": str(log_path)}), flush=True)
        env = dict(os.environ)
        env["TAC_LOSS_TERM_PROBE"] = "1"
        env.setdefault("TAC_MLX_CUSTOM_GROUPED_BACKWARD", "1")
        with log_path.open("w") as lf:
            try:
                rc = subprocess.run(argv_t, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=args.timeout_s, env=env, cwd=str(REPO)).returncode
            except subprocess.TimeoutExpired:
                rc = -9
                print(json.dumps({"stage": "arm_timeout", "arm": name}), flush=True)
        parsed = _parse_rows(log_path)
        summ = _summarize_arm(name, parsed, args.epochs_past_resume)
        summ["rc"] = rc
        summaries.append(summ)
        print(json.dumps({"stage": "arm_done", "arm": name, "rc": rc,
                          "status": summ.get("status"),
                          "total_first": summ.get("total_first"),
                          "total_last": summ.get("total_last"),
                          "spike_skips": summ.get("spike_skips")}), flush=True)

    report = {
        "axis": "[n24 advisory -- mechanism probe, NOT n600 evidence]",
        "snapshot": str(SNAPSHOT), "gt_cache": str(GT_N24),
        "epochs_past_resume": args.epochs_past_resume,
        "spike_guard": "DISARMED (--spike-factor 1e9) so evolution is observed",
        "arms": summaries,
        "pointer": "0.19110 UNMOVED",
    }
    rp = out_dir / "step_instability_probe_report.json"
    rp.write_text(json.dumps(report, indent=1))
    print(json.dumps({"stage": "report_written", "path": str(rp)}), flush=True)
    # compact operator table
    print("\narm             status    total first->last   skips  top delta terms")
    for s in summaries:
        if s.get("status") in ("EXPLODES", "STABLE"):
            tops = ",".join(s["runaway_terms"].keys()) or ",".join(
                d["term"] for d in s["dominant_terms_by_delta"][:3])
            print(f"{s['arm']:<15} {s['status']:<9} {s['total_first']:>8} -> trough "
                  f"{s['total_trough']}@ep{s['trough_epoch']} -> peak {s['peak_after_trough']}"
                  f" ({s['trough_to_peak_ratio']}x) skips={s['spike_skips']}  runaway: {tops}")
        else:
            print(f"{s['arm']:<15} {s.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
