"""tac.witness_control.trace_probes — $0 trace-replay instruments for the T5 crucible probes.

DURABLE REUSABLE INSTRUMENTS (requirement Q, T5 crucible orchestration ledger): the three
CONTROL/SCHEDULE recess probes P-CT1 / P-CT2 / P-CT3 as pure functions over ANY completed run's
41-row (or N-row) verdict trace (``levelset_train_result.json`` ``history``). Run-2 re-runs the
SAME instruments on its own trace for free — the campaign ILC error term made cheap.

Laws implemented (provenance: DRAFT_OPTIMAL_STACK_v5_20260707.md §2.2f/§2.5/§7c +
ct_deepresearch_1_training_campaign_control_20260707.md §3.3/§4.2/§10.2):

* **P-CT3 forfeit-matched TAU→FIN arm** (v5 §2.2f, CT-1 §3.3): fire when the per-epoch-normalized
  d_seg slope s < s* = ν·forfeit. The slope estimator is EXACTLY the shipped co-predicate's form
  (v2 §2.2b, reproduced bit-for-bit by :func:`copredicate_backtest` against the mod32cap table:
  eps_rel=5e-3/V=4 → first fire ep625, rel slope −1.369e-3/25ep, n_fires=8): trailing-V-row
  endpoint slope. A least-squares line fit over the same window is reported as the robust
  second estimator (requirement: "run BOTH the shipped form and a robust fit").
* **P-CT1 ν refit per stage** (v5 §7c row 4, CT-1 §10.2): d_seg(t) = a + b·exp(−ν·t) per stage via
  :func:`tac.witness_control.powerlaw_exit.fit_tail_models` (the existing deterministic profile-LS
  fitting machinery); ν = 1/tau_e. The AIC-preferred alternative (power-law, alpha) is reported
  alongside per requirement R (a failed exponential FORMULATION does not kill window-law thinking).
* **P-CT2 self-triggered verdict cadence replay** (v5 §2.5, CT-1 §4.2):
  Δt_next = clamp(floor_S/|Ŝ′(t)|, lo, hi), replayed over the trace's verdict grid using only
  VISITED verdicts for the slope estimate (self-triggered semantics), floor-rounded to the grid.

Everything here is READ-ONLY + advisory (like the whole ``witness_control`` package): it never
mutates a run, never claims a score. All slopes are in d_seg-only S units (S contribution
100·d_seg), per-epoch-normalized (MINOR-9 discipline). Fail-safe: insufficient data raises or
returns explicit ``None`` fields — never a silent verdict.

Pre-registered bands live in the CLI (``tools/witness_trace_probes.py``) with their provenance;
this module computes numbers only, so a future band revision cannot silently re-verdict old
artifacts (the band + verdict are stamped into each artifact at emit time).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tac import witness_run_artifacts as _wra
from tac.witness_control.powerlaw_exit import fit_tail_models

__all__ = [
    "load_history",
    "copredicate_backtest",
    "forfeit_matched_backtest",
    "refit_nu_per_stage",
    "cadence_replay",
]

_S_PER_DSEG = 100.0  # S = 100*d_seg + sqrt(10*d_pose) + rate; d_seg-only S units throughout.


def load_history(run_dir: str | Path) -> list[tuple[int, float]]:
    """Load the verdict trace [(epoch, d_seg), ...] from ``<run_dir>/levelset_train_result.json``.

    The mod32cap control run's ``history`` is the canonical "41-row trace" (epochs 0,25,...,1000).
    """
    p = Path(run_dir) / _wra.TRAIN_RESULT_JSON
    hist = json.loads(p.read_text(encoding="utf-8"))["history"]
    rows = sorted((int(r["epoch"]), float(r["d_seg"])) for r in hist)
    if len(rows) < 4:
        raise ValueError(f"trace too short ({len(rows)} rows) in {p}")
    return rows


def _endpoint_slope_s_per_ep(rows: list[tuple[int, float]], i: int, v_window: int) -> float:
    """Shipped-form per-epoch-normalized IMPROVEMENT slope in S/ep over the trailing V rows.

    s = −100·(d[i] − d[i−(V−1)]) / (ep[i] − ep[i−(V−1)])  (positive while d_seg descends).
    """
    j = i - (v_window - 1)
    (e0, d0), (e1, d1) = rows[j], rows[i]
    return -_S_PER_DSEG * (d1 - d0) / float(e1 - e0)


def _ls_slope_s_per_ep(rows: list[tuple[int, float]], i: int, v_window: int) -> float:
    """Robust second estimator: least-squares line slope over the same trailing-V window."""
    j = i - (v_window - 1)
    xs = [float(e) for e, _ in rows[j:i + 1]]
    ys = [d for _, d in rows[j:i + 1]]
    n = float(len(xs))
    xb, yb = sum(xs) / n, sum(ys) / n
    den = sum((x - xb) ** 2 for x in xs)
    beta = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / den
    return -_S_PER_DSEG * beta


def copredicate_backtest(rows: list[tuple[int, float]], *, eps_rel: float = 5e-3,
                         v_window: int = 4) -> dict:
    """Reproduce the SHIPPED co-predicate backtest (v2 §2.2b) — the estimator-form anchor.

    Predicate: trailing-V-row relative d_seg slope per cadence
    (d[i]−d[j])/d[j]/(V−1) > −eps_rel ⇒ fire. Returns the fire list; against the mod32cap trace
    this must yield first fire ep625, rel slope −1.369e-3, n_fires=8 (the bit-for-bit anchor that
    pins the estimator form used by :func:`forfeit_matched_backtest`).
    """
    fires = []
    for i in range(v_window - 1, len(rows)):
        j = i - (v_window - 1)
        rel = (rows[i][1] - rows[j][1]) / rows[j][1] / (v_window - 1)
        if rel > -eps_rel:
            fires.append({"epoch": rows[i][0], "rel_slope_per_cadence": rel})
    return {"eps_rel": eps_rel, "v_window": v_window, "n_fires": len(fires),
            "first_fire_epoch": fires[0]["epoch"] if fires else None,
            "first_fire_rel_slope": fires[0]["rel_slope_per_cadence"] if fires else None,
            "fires": fires}


def _prefix_best(rows: list[tuple[int, float]], lo: int, hi: int, upto: int) -> tuple[int, float]:
    """EMA-best proxy: the best (min d_seg) verdict row within [lo, min(hi, upto)]."""
    cand = [(d, e) for e, d in rows if lo <= e <= min(hi, upto)]
    d, e = min(cand)
    return e, d


def forfeit_matched_backtest(rows: list[tuple[int, float]], *, s_star: float,
                             v_window: int = 4, stage_lo: int = 300, stage_hi: int = 725,
                             cap_epoch: int = 726,
                             reference_fire_epochs: tuple[int, ...] = (625, 650)) -> dict:
    """P-CT3: replay the forfeit-matched TAU→FIN arm (v5 §2.2f) over a verdict trace.

    Fires (would-fire) at the first stage verdict where the shipped-form windowed slope
    s < s_star; the LS-fit slope is replayed in parallel. For each fire/reference epoch the
    stage-window EMA-best (prefix-min proxy — per-stage ckpt mandate) and its forfeit vs the
    stage's true best are reported at full precision (the v3 §2.2c forfeit-table arithmetic).
    """
    per_verdict = []
    fires_endpoint: list[int] = []
    fires_ls: list[int] = []
    for i in range(v_window - 1, len(rows)):
        e = rows[i][0]
        if not (stage_lo <= e <= stage_hi):
            continue
        s_ep = _endpoint_slope_s_per_ep(rows, i, v_window)
        s_ls = _ls_slope_s_per_ep(rows, i, v_window)
        per_verdict.append({"epoch": e, "slope_endpoint_S_per_ep": s_ep,
                            "slope_lsfit_S_per_ep": s_ls,
                            "fire_endpoint": s_ep < s_star, "fire_lsfit": s_ls < s_star})
        if s_ep < s_star:
            fires_endpoint.append(e)
        if s_ls < s_star:
            fires_ls.append(e)

    stage_best_e, stage_best_d = _prefix_best(rows, stage_lo, stage_hi, stage_hi)

    def _fire_report(fire_epoch: int | None) -> dict | None:
        if fire_epoch is None:
            return None
        eb, db = _prefix_best(rows, stage_lo, stage_hi, fire_epoch)
        d_at = dict(rows)[fire_epoch]
        return {
            "fire_epoch": fire_epoch,
            "d_seg_at_fire": d_at,
            "ema_best_epoch": eb,
            "ema_best_d_seg": db,
            "ema_best_within_cadence_of_stage_best":
                abs(eb - stage_best_e) <= (rows[1][0] - rows[0][0]),
            "forfeit_d_seg_vs_stage_best": db - stage_best_d,
            "forfeit_S_vs_stage_best": _S_PER_DSEG * (db - stage_best_d),
            "forfeit_S_current_theta_no_restore": _S_PER_DSEG * (d_at - stage_best_d),
        }

    return {
        "s_star_S_per_ep": s_star, "v_window": v_window,
        "stage_window": [stage_lo, stage_hi], "cap_epoch": cap_epoch,
        "stage_best": {"epoch": stage_best_e, "d_seg": stage_best_d},
        "per_verdict": per_verdict,
        "first_sustained_fire_endpoint": fires_endpoint[0] if fires_endpoint else None,
        "first_sustained_fire_lsfit": fires_ls[0] if fires_ls else None,
        "all_fires_endpoint": fires_endpoint, "all_fires_lsfit": fires_ls,
        "fire_report_endpoint": _fire_report(fires_endpoint[0] if fires_endpoint else None),
        "fire_report_lsfit": _fire_report(fires_ls[0] if fires_ls else None),
        "reference_fires": {str(e): _fire_report(e) for e in reference_fire_epochs
                            if e in dict(rows)},
    }


def refit_nu_per_stage(rows: list[tuple[int, float]],
                       stages: dict[str, tuple[int, int]], *, n_boot: int = 50) -> dict:
    """P-CT1: refit the contraction constant ν per stage (v5 §7c row 4 / CT-1 §10.2).

    Fits BOTH d(t)=a+b·exp(−t/tau_e) (ν = 1/tau_e) and the power-law alternative via the
    existing :func:`fit_tail_models` machinery; reports AIC preference so an exponential-
    FORMULATION mismatch is visible without killing window-law thinking (requirement R).
    Derived window laws are recomputed from each stage's ν: settle 3/ν, cycle floor 3/ν+150,
    s* = ν·forfeit (forfeit = 5.450779e-4 S, the v3 §2.2c measured transition cost re-verified
    by :func:`forfeit_matched_backtest` on the mod32cap trace).
    """
    forfeit_s = 5.450779e-4
    out: dict = {"stages": {}}
    for name, (lo, hi) in stages.items():
        sub = [(e, d) for e, d in rows if lo <= e <= hi]
        if len(sub) < 4:
            out["stages"][name] = {"error": f"insufficient rows ({len(sub)}) in [{lo},{hi}]"}
            continue
        fits = fit_tail_models([e for e, _ in sub], [d for _, d in sub], n_boot=n_boot)
        tau_e = fits["exponential"]["tau_e"]
        nu = 1.0 / tau_e
        out["stages"][name] = {
            "window": [lo, hi], "n_rows": len(sub),
            "nu_per_ep": nu, "tau_e": tau_e,
            "exp_a": fits["exponential"]["a"], "exp_b": fits["exponential"]["b"],
            "preferred_model_aic": fits["preferred_model"],
            "delta_aic_exp_minus_pow": fits["delta_aic_exp_minus_pow"],
            "powerlaw_alpha": fits["alpha"], "powerlaw_alpha_ci95": fits["alpha_ci95"],
            "powerlaw_a": fits["power_law"]["a"],
            "recomputed_window_laws": {
                "settle_3_over_nu_ep": 3.0 / nu,
                "tail_cycle_floor_ep": 3.0 / nu + 150.0,
                "dwell_lower_bound_ep": 3.0 / nu,
                "s_star_nu_times_forfeit_S_per_ep": nu * forfeit_s,
            },
        }
    return out


def cadence_replay(rows: list[tuple[int, float]], *, floor_s: float = 0.00178,
                   dt_lo: int = 25, dt_hi: int = 100,
                   estimator: str = "window", v_window: int = 4) -> dict:
    """P-CT2: replay Δt_next = clamp(floor_S/|Ŝ′|, lo, hi) (v5 §2.5 / CT-1 §4.2).

    Self-triggered semantics: |Ŝ′| is estimated ONLY from already-visited verdicts —
    ``estimator="window"`` uses the trailing min(V, len(visited)) visited rows (endpoint form,
    per-epoch-normalized); ``estimator="pair"`` uses the last two visited rows. Δt is clamped
    then floor-rounded to the trace's verdict grid. Missed-best analysis: every prefix-best
    row must lie within one BASE cadence of a visited verdict.
    """
    ep = [e for e, _ in rows]
    d = dict(rows)
    grid = ep[1] - ep[0]
    diffs = {b - a for a, b in zip(ep, ep[1:])}
    if diffs != {grid}:
        raise ValueError(f"non-uniform verdict grid {sorted(diffs)}; cadence replay needs a "
                         "uniform trace grid")
    if dt_lo < grid:
        raise ValueError(f"dt_lo {dt_lo} below the trace grid {grid}")
    visited = [ep[0]]
    while True:
        if len(visited) >= 2:
            if estimator == "window":
                w = visited[-min(v_window, len(visited)):]
                slope = abs(_S_PER_DSEG * (d[w[-1]] - d[w[0]]) / (w[-1] - w[0]))
            elif estimator == "pair":
                slope = abs(_S_PER_DSEG * (d[visited[-1]] - d[visited[-2]])
                            / (visited[-1] - visited[-2]))
            else:
                raise ValueError(f"unknown estimator {estimator!r}")
            dt = min(max(floor_s / max(slope, 1e-30), float(dt_lo)), float(dt_hi))
            dt = grid * int(dt // grid)
        else:
            dt = float(dt_lo)
        nxt = visited[-1] + int(dt)
        if nxt > ep[-1]:
            break
        visited.append(nxt)
    skipped = [e for e in ep if e not in set(visited)]
    prefix_best = []
    cur = math.inf
    for e in ep:
        if d[e] < cur:
            cur = d[e]
            prefix_best.append(e)
    missed = [{"best_epoch": b, "dist_to_nearest_visited": min(abs(b - v) for v in visited)}
              for b in prefix_best if min(abs(b - v) for v in visited) > grid]
    global_best = min((v, e) for e, v in rows)[1]
    return {
        "floor_S": floor_s, "dt_clamp": [dt_lo, dt_hi], "estimator": estimator,
        "v_window": v_window, "base_cadence_ep": grid,
        "n_trace_verdicts": len(ep), "n_visited": len(visited),
        "n_skipped": len(skipped), "skipped_epochs": skipped, "visited_epochs": visited,
        "missed_prefix_best_beyond_one_cadence": missed,
        "global_best_epoch": global_best,
        "global_best_dist_to_nearest_visited": min(abs(global_best - v) for v in visited),
        "global_best_visited": global_best in set(visited),
    }
