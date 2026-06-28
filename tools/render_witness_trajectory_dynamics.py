#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Standing READ-ONLY *trajectory-dynamics* instrument for the level-set witness row.

TEMPORAL sister of the SPATIAL stage-diff (``.omx/tmp/stage_compare/``). Where the
stage-diff compares WHERE in the frame two stages disagree, this tool measures HOW
FAST each curriculum stage / loss-form lever does its work over EPOCHS — so every
run automatically TEACHES us each config's strength + per-stage time-constants +
"how many epochs each behavior takes to demonstrate". The A/B campaign calls this
per-arm to BUDGET each arm's length.

It parses the verdict-bearing daemon log(s) of
``experiments/train_levelset_witness_realized_through_R_mlx.py`` — JSON lines of the
form ``{"stage":"verdict","epoch":N,"seg_form":"ce|tau_softplus|l7_softplus",
"d_seg":..,"d_pose":..,"blob_bytes":..,"implied_S":..}`` — groups them by stage
(``seg_form``, in epoch order), and for EACH stage computes:

  * onset / end epoch, #verdicts
  * start d_seg, best d_seg + its epoch, time-to-best (= best_ep - onset)
  * DESCENT RATE in the productive window (start->best): Delta d_seg / epoch — the
    comparable "config strength" metric (the cross-stage ranking IS this ranking)
  * PLATEAU onset: the epoch after which d_seg stops improving for K consecutive
    verdicts (productive->dead transition), K configurable (default 3)
  * POST-PLATEAU "dead tail": length (epochs + verdicts) + its net Delta d_seg
    (surfaces wall-clock waste, e.g. tau's measured ~200-ep dead tail)
  * VOLATILITY band post-best: min / max / std d_seg
  * net Delta d_seg and Delta d_pose across the stage
  * TIME-TO-DEMONSTRATE: smallest #epochs into the stage at which d_seg has dropped
    >= X% (default 50%) of the stage's eventual total drop — how fast the behavior
    becomes visible (l7 ~50ep; tau ~375ep to best)
  * the d_pose analogues (best / descent / time-to-demonstrate) where useful

It then emits a cross-stage summary table (descent-rate ranking = config-strength
ranking), a machine-readable JSON dump (so the A/B campaign / a planner can consume
the time-constants), and a clean stacked trajectory plot (d_seg log-y / d_pose /
implied_S vs epoch) with stage shading + best-point markers + plateau lines + a
per-stage text box.

ALL MEASURED. ADVISORY ONLY ([macOS-MLX training] NON-PROMOTABLE; the verdict log's
realized-through-R MLX d_seg/d_pose is the signal). The exact contest row is
byte-closed CPU/CUDA; the canonical frontier pointer is 0.19110.

READ-ONLY: this tool never touches the GPU, never edits the trainer, never launches
training. It only READS verdict logs and WRITES a PNG + JSON under .omx/tmp/trajectory.

Usage:
    # follow the newest verdict-bearing run log (single-run view):
    render_witness_trajectory_dynamics.py

    # explicit log(s) — repeat --log to MERGE a curriculum split across runs
    # (e.g. CE in one log + tau/l7 in a resume log); later --log wins per epoch:
    render_witness_trajectory_dynamics.py \
        --log .omx/tmp/levelset_amort_decoder_n200_...log \
        --log .omx/tmp/levelset_amort_deconf_taualone_...log \
        --out .omx/tmp/trajectory/witness_trajectory_dynamics.png \
        --json .omx/tmp/trajectory/witness_trajectory_dynamics.json
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

# ── stage canon: seg_form -> (short label, shading color) ─────────────────────
# Colors mirror tools/render_levelset_dashboard.py for visual consistency.
_STAGE_CE = "#1f3b5f"  # blue
_STAGE_TAU = "#3a2a5f"  # purple
_STAGE_L7 = "#5f3320"  # orange
_STAGE_MUON = "#2f5f33"  # green
_STAGE_OTHER = "#3a3a3a"  # neutral

_STAGE_META: "OrderedDict[str, tuple[str, str]]" = OrderedDict(
    [
        ("ce", ("CE", _STAGE_CE)),
        ("tau_softplus", ("tau", _STAGE_TAU)),
        ("l7_softplus", ("l7", _STAGE_L7)),
        ("muon", ("Muon", _STAGE_MUON)),
    ]
)


def _stage_short(label: str) -> str:
    return _STAGE_META.get(label, (label, _STAGE_OTHER))[0]


def _stage_color(label: str) -> str:
    return _STAGE_META.get(label, (label, _STAGE_OTHER))[1]


# ── dark theme palette (clean, calm — matches the dashboard) ──────────────────
_BG = "#14161a"
_PANEL = "#1b1e24"
_FG = "#d8dde6"
_MUTED = "#8b93a3"
_GRIDC = "#2c313b"
_ACC = "#5ab0ff"  # d_seg
_POSE = "#ffb454"  # d_pose
_SVAL = "#ff6b6b"  # implied_S
_GOAL = "#46d369"
_BEST = "#ffffff"  # best-point marker


# ─────────────────────────── parse + merge ───────────────────────────────────
def parse_verdicts(log_path: "str | Path") -> list[dict]:
    """Parse verdict JSON lines from a single log file (matplotlib-free, pure).

    Mirrors the dashboard's ``_parse_verdicts`` pattern: keep only lines that are a
    JSON object with ``stage=="verdict"`` and an ``epoch`` field. Returns rows
    sorted by epoch. Non-verdict / malformed lines are skipped silently."""
    path = Path(log_path)
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if '"stage": "verdict"' not in line and '"stage":"verdict"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("stage") != "verdict" or "epoch" not in d:
            continue
        try:
            d["epoch"] = int(d["epoch"])
        except Exception:
            continue
        rows.append(d)
    rows.sort(key=lambda r: r["epoch"])
    return rows


def _has_verdict(path: Path) -> bool:
    """True iff the file carries >=1 verdict line. Used so the self-follow
    resolver never latches onto a daemon log / non-verdict log that merely shares
    the glob with a newer mtime (the dashboard's confounder guard)."""
    try:
        if not path.is_file():
            return False
        for line in path.read_text(errors="replace").splitlines():
            if '"stage": "verdict"' in line or '"stage":"verdict"' in line:
                return True
    except Exception:
        return False
    return False


def resolve_logs(logs: "list[str] | None", log_glob: "str | None") -> list[Path]:
    """Resolve which log(s) to read.

    Explicit ``--log`` (repeatable) WINS (back-compat with the dashboard's
    ``--log`` override) and is returned VERBATIM in command-line order — repeated
    ``--log`` is how a curriculum split across runs is merged (later wins per
    epoch, see ``merge_rows``). Otherwise return a single-element list: the
    NEWEST-mtime file matching ``--log-glob`` that contains verdict lines (so a
    freshly-started run is auto-followed and we never self-follow a non-verdict
    log). Returns ``[]`` when nothing resolves. Ties on mtime break by filename."""
    if logs:
        return [Path(p) for p in logs]
    if not log_glob:
        return []
    verdict_logs = [Path(p) for p in _glob.glob(log_glob) if _has_verdict(Path(p))]
    if not verdict_logs:
        return []
    verdict_logs.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return [verdict_logs[-1]]


def merge_rows(row_lists: list[list[dict]]) -> list[dict]:
    """Merge verdict rows from multiple logs, keyed by epoch, LATER-LIST-WINS.

    Curriculum-stitch semantics: passing ``--log CE_run --log tau_l7_resume_run``
    yields CE epochs from the first log and tau/l7 epochs from the second; where an
    epoch appears in BOTH (e.g. an overlapping tau window between a confounded run
    and its deconfounded resume), the LAST log's row overrides — so the resume run
    cleanly supersedes the earlier run on the shared epochs. Deterministic + pure;
    returns rows sorted by epoch."""
    by_epoch: "dict[int, dict]" = {}
    for rows in row_lists:
        for r in rows:
            by_epoch[int(r["epoch"])] = r
    return [by_epoch[e] for e in sorted(by_epoch)]


def assign_stage(row: dict, tau: int, l7: int, muon: "int | None") -> str:
    """Canonical stage label for a verdict row.

    Uses the explicit ``seg_form`` when present (the trainer's own stage marker).
    For formless rows (the ep0 random-init baseline, a resume anchor) the stage is
    INFERRED from the curriculum epoch thresholds (``--tau`` / ``--l7``), matching
    the dashboard's ``_stage_short`` convention so a formless row never spawns a
    spurious ``<none>`` stage. When ``--muon`` is set, l7-region rows at
    ``epoch >= muon`` are relabelled ``muon`` (the optimizer sub-phase)."""
    form = row.get("seg_form")
    ep = int(row["epoch"])
    if not form:
        if ep < tau:
            form = "ce"
        elif ep < l7:
            form = "tau_softplus"
        else:
            form = "l7_softplus"
    if muon is not None and form == "l7_softplus" and ep >= muon:
        form = "muon"
    return form


def group_stages(rows: list[dict], tau: int, l7: int, muon: "int | None") -> "OrderedDict[str, list[dict]]":
    """Group rows by canonical stage label, in order of each stage's first epoch."""
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rows:
        label = assign_stage(r, tau, l7, muon)
        groups.setdefault(label, []).append(r)
    for label in groups:
        groups[label].sort(key=lambda r: r["epoch"])
    # order stages by their minimum epoch (chronological curriculum order)
    ordered = OrderedDict(sorted(groups.items(), key=lambda kv: kv[1][0]["epoch"]))
    return ordered


# ─────────────────────────── per-stage dynamics ──────────────────────────────
def _std(xs: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def _series(stage_rows: list[dict], key: str) -> list[tuple[int, float]]:
    """[(epoch, value)] for rows where value is a finite number, in epoch order."""
    out: list[tuple[int, float]] = []
    for r in stage_rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append((int(r["epoch"]), float(v)))
    return out


def _plateau_and_tail(series: list[tuple[int, float]], best_epoch: int, k: int) -> dict:
    """Plateau onset + post-plateau dead-tail metrics for a (epoch,value) series.

    Plateau onset = the best epoch IFF >= K verdicts follow it WITHOUT a new
    running-min (since the best is the series min, none after it can improve) — the
    productive->dead transition. Dead tail = end - plateau_onset (epochs), the
    verdict count after it, and its net Delta value (>=0 = regression/wall-clock
    waste). ``trailing_no_improve_streak`` is exposed for transparency."""
    after = [(e, v) for (e, v) in series if e > best_epoch]
    streak = len(after)  # all after the global-min best fail to improve by definition
    if streak >= k:
        plateau_onset = best_epoch
        end_epoch, end_val = series[-1]
        best_val = next(v for (e, v) in series if e == best_epoch)
        return {
            "plateau_onset_epoch": plateau_onset,
            "trailing_no_improve_streak": streak,
            "dead_tail_epochs": end_epoch - plateau_onset,
            "dead_tail_verdicts": len(after),
            "dead_tail_net_delta": end_val - best_val,
        }
    return {
        "plateau_onset_epoch": None,
        "trailing_no_improve_streak": streak,
        "dead_tail_epochs": 0,
        "dead_tail_verdicts": 0,
        "dead_tail_net_delta": 0.0,
    }


def _time_to_demonstrate(series: list[tuple[int, float]], onset: int, frac: float) -> "int | None":
    """Smallest #epochs into the stage at which value has dropped >= frac of its
    eventual total drop (start - best). None when the stage never improves."""
    if len(series) < 2:
        return None
    start = series[0][1]
    best = min(v for (_, v) in series)
    total_drop = start - best
    if total_drop <= 0:
        return None
    target = start - frac * total_drop
    for e, v in series:
        if v <= target:
            return e - onset
    return None


def compute_stage_dynamics(label: str, stage_rows: list[dict], plateau_k: int,
                           demonstrate_frac: float) -> dict:
    """All trajectory-dynamics metrics for ONE stage. Pure + JSON-friendly."""
    seg = _series(stage_rows, "d_seg")
    pose = _series(stage_rows, "d_pose")
    onset = stage_rows[0]["epoch"]
    end = stage_rows[-1]["epoch"]
    out: dict = {
        "stage": label,
        "stage_short": _stage_short(label),
        "onset_epoch": onset,
        "end_epoch": end,
        "n_verdicts": len(stage_rows),
        "span_epochs": end - onset,
    }

    # ── d_seg dynamics ──
    if seg:
        start_dseg = seg[0][1]
        best_epoch, best_dseg = min(seg, key=lambda t: t[1])
        ttb = best_epoch - onset
        # signed descent rate Delta d_seg / epoch over the productive window
        # (negative = descending = good); None when there is no window (ttb == 0).
        descent = (best_dseg - start_dseg) / ttb if ttb > 0 else None
        improvement = max(0.0, -(descent)) if descent is not None else 0.0
        post = [v for (e, v) in seg if e >= best_epoch]
        out.update(
            {
                "start_d_seg": start_dseg,
                "best_d_seg": best_dseg,
                "best_epoch": best_epoch,
                "time_to_best": ttb,
                "descent_rate_dseg_per_epoch": descent,
                "improvement_per_epoch_dseg": improvement,
                "net_delta_dseg": seg[-1][1] - start_dseg,
                "vol_post_best_min": min(post),
                "vol_post_best_max": max(post),
                "vol_post_best_std": _std(post),
                "time_to_demonstrate_dseg": _time_to_demonstrate(seg, onset, demonstrate_frac),
            }
        )
        out.update(_plateau_and_tail(seg, best_epoch, plateau_k))
        # rename the generic dead-tail keys to make them d_seg-specific in JSON
        out["dead_tail_net_delta_dseg"] = out.pop("dead_tail_net_delta")
    else:
        out.update(
            {
                "start_d_seg": None, "best_d_seg": None, "best_epoch": None,
                "time_to_best": None, "descent_rate_dseg_per_epoch": None,
                "improvement_per_epoch_dseg": 0.0, "net_delta_dseg": None,
                "vol_post_best_min": None, "vol_post_best_max": None,
                "vol_post_best_std": None, "time_to_demonstrate_dseg": None,
                "plateau_onset_epoch": None, "trailing_no_improve_streak": 0,
                "dead_tail_epochs": 0, "dead_tail_verdicts": 0,
                "dead_tail_net_delta_dseg": 0.0,
            }
        )

    # ── d_pose dynamics (analogue, where useful) ──
    if pose:
        start_dpose = pose[0][1]
        best_pose_epoch, best_dpose = min(pose, key=lambda t: t[1])
        ttb_pose = best_pose_epoch - onset
        descent_pose = (best_dpose - start_dpose) / ttb_pose if ttb_pose > 0 else None
        out.update(
            {
                "start_d_pose": start_dpose,
                "best_d_pose": best_dpose,
                "best_d_pose_epoch": best_pose_epoch,
                "time_to_best_dpose": ttb_pose,
                "descent_rate_dpose_per_epoch": descent_pose,
                "net_delta_dpose": pose[-1][1] - start_dpose,
                "time_to_demonstrate_dpose": _time_to_demonstrate(pose, onset, demonstrate_frac),
            }
        )
    else:
        out.update(
            {
                "start_d_pose": None, "best_d_pose": None, "best_d_pose_epoch": None,
                "time_to_best_dpose": None, "descent_rate_dpose_per_epoch": None,
                "net_delta_dpose": None, "time_to_demonstrate_dpose": None,
            }
        )
    return out


def _anchor_check(stages: list[dict]) -> dict:
    """Self-validate against the hand-measured anchors (the tool is the oracle for
    these numbers): tau time-to-best ~375 (best@~675), tau dead-tail ~200, and l7
    descent rate ~2.8x tau's. Returns measured values + within-tolerance booleans.
    Null fields when a stage is absent (e.g. a tau-only or CE-only view)."""
    by = {s["stage"]: s for s in stages}
    tau = by.get("tau_softplus")
    l7 = by.get("l7_softplus")
    res: dict = {
        "tau_time_to_best": None, "tau_best_epoch": None, "tau_dead_tail_epochs": None,
        "l7_over_tau_descent_ratio": None,
        "tau_time_to_best_matches_375": None, "tau_dead_tail_matches_200": None,
        "l7_descent_ratio_matches_2p8": None,
    }
    if tau and tau.get("time_to_best") is not None:
        res["tau_time_to_best"] = tau["time_to_best"]
        res["tau_best_epoch"] = tau["best_epoch"]
        res["tau_dead_tail_epochs"] = tau["dead_tail_epochs"]
        res["tau_time_to_best_matches_375"] = abs(tau["time_to_best"] - 375) <= 50
        res["tau_dead_tail_matches_200"] = abs(tau["dead_tail_epochs"] - 200) <= 50
    if (
        tau and l7
        and tau.get("improvement_per_epoch_dseg")
        and l7.get("improvement_per_epoch_dseg")
    ):
        ratio = l7["improvement_per_epoch_dseg"] / tau["improvement_per_epoch_dseg"]
        res["l7_over_tau_descent_ratio"] = ratio
        res["l7_descent_ratio_matches_2p8"] = abs(ratio - 2.8) <= 0.8
    return res


def compute_dynamics(rows: list[dict], tau: int, l7: int, muon: "int | None",
                     plateau_k: int, demonstrate_frac: float) -> dict:
    """Top-level: group + per-stage dynamics + cross-stage ranking + anchor check."""
    groups = group_stages(rows, tau, l7, muon)
    stages = [
        compute_stage_dynamics(label, srows, plateau_k, demonstrate_frac)
        for label, srows in groups.items()
    ]
    # cross-stage config-strength ranking = descent-rate (improvement/epoch) ranking
    ranked = sorted(
        [s for s in stages if s.get("improvement_per_epoch_dseg")],
        key=lambda s: s["improvement_per_epoch_dseg"],
        reverse=True,
    )
    min_imp = min(
        (s["improvement_per_epoch_dseg"] for s in ranked if s["improvement_per_epoch_dseg"]),
        default=None,
    )
    ranking = [
        {
            "stage": s["stage"],
            "stage_short": s["stage_short"],
            "improvement_per_epoch_dseg": s["improvement_per_epoch_dseg"],
            "descent_rate_dseg_per_epoch": s["descent_rate_dseg_per_epoch"],
            "time_to_best": s["time_to_best"],
            "ratio_to_slowest": (
                s["improvement_per_epoch_dseg"] / min_imp if min_imp else None
            ),
        }
        for s in ranked
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": "[macOS-MLX training advisory] NON-PROMOTABLE",
        "frontier_pointer": 0.19110,
        "curriculum": {"tau": tau, "l7": l7, "muon": muon},
        "plateau_k": plateau_k,
        "demonstrate_frac": demonstrate_frac,
        "n_verdicts": len(rows),
        "epoch_range": [rows[0]["epoch"], rows[-1]["epoch"]] if rows else None,
        "stages": stages,
        "cross_stage_ranking": ranking,
        "anchor_check": _anchor_check(stages),
    }


# ─────────────────────────── text table ──────────────────────────────────────
def _fmt(v, spec: str = ".6f") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        try:
            return format(v, spec)
        except Exception:
            return str(v)
    return str(v)


def render_table(dynamics: dict) -> str:
    """Human-readable per-stage + cross-stage summary table (string)."""
    lines: list[str] = []
    er = dynamics.get("epoch_range")
    lines.append(
        f"Witness trajectory dynamics — {dynamics['n_verdicts']} verdicts"
        + (f", epochs {er[0]}..{er[1]}" if er else "")
        + f"  [{dynamics['axis']}; pointer {dynamics['frontier_pointer']}]"
    )
    lines.append("")
    hdr = (
        f"{'stage':>5} {'onset':>5} {'end':>5} {'n':>3} {'start_dseg':>10} "
        f"{'best_dseg':>10} {'@ep':>5} {'t2best':>6} {'descent/ep':>11} "
        f"{'plateau':>7} {'deadtail':>8} {'t2dem':>6} {'volStd':>8} {'netΔseg':>9} {'netΔpose':>9}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def _i(v) -> str:  # int-or-dash
        return str(v) if v is not None else "—"

    for s in dynamics["stages"]:
        lines.append(
            f"{s['stage_short']:>5} {s['onset_epoch']:>5} "
            f"{s['end_epoch']:>5} {s['n_verdicts']:>3} {_fmt(s['start_d_seg']):>10} "
            f"{_fmt(s['best_d_seg']):>10} {_i(s['best_epoch']):>5} "
            f"{_i(s['time_to_best']):>6} "
            f"{_fmt(s['descent_rate_dseg_per_epoch'],'.3e'):>11} "
            f"{_i(s['plateau_onset_epoch']):>7} "
            f"{s['dead_tail_epochs']:>8} "
            f"{_i(s['time_to_demonstrate_dseg']):>6} "
            f"{_fmt(s['vol_post_best_std'],'.2e'):>8} "
            f"{_fmt(s['net_delta_dseg'],'.2e'):>9} {_fmt(s['net_delta_dpose'],'.2e'):>9}"
        )
    lines.append("")
    lines.append("Cross-stage CONFIG-STRENGTH ranking (descent rate = improvement in d_seg per epoch):")
    for i, r in enumerate(dynamics["cross_stage_ranking"], 1):
        ratio = r.get("ratio_to_slowest")
        lines.append(
            f"  {i}. {r['stage_short']:>5}  {_fmt(r['improvement_per_epoch_dseg'],'.3e')}/ep"
            f"  (t2best {_fmt(r['time_to_best'],'d') if r['time_to_best'] is not None else '—'}"
            f", {_fmt(ratio,'.2f')}x slowest)"
        )
    ac = dynamics["anchor_check"]
    lines.append("")
    lines.append("Anchor check (hand-measured oracle):")
    lines.append(
        f"  tau time-to-best = {_fmt(ac['tau_time_to_best'],'d') if ac['tau_time_to_best'] is not None else '—'}"
        f" (best@{_fmt(ac['tau_best_epoch'],'d') if ac['tau_best_epoch'] is not None else '—'})"
        f"  [~375? {ac['tau_time_to_best_matches_375']}]"
    )
    lines.append(
        f"  tau dead-tail   = {_fmt(ac['tau_dead_tail_epochs'],'d') if ac['tau_dead_tail_epochs'] is not None else '—'}"
        f"  [~200? {ac['tau_dead_tail_matches_200']}]"
    )
    lines.append(
        f"  l7/tau descent  = {_fmt(ac['l7_over_tau_descent_ratio'],'.2f')}x"
        f"  [~2.8? {ac['l7_descent_ratio_matches_2p8']}]"
    )
    return "\n".join(lines)


# ─────────────────────────── plot ────────────────────────────────────────────
def _style_ax(ax) -> None:
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_color(_GRIDC)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(alpha=0.18, color=_GRIDC)


def render_png(rows: list[dict], dynamics: dict, out_path: "str | Path") -> Path:
    """Stacked d_seg(log) / d_pose(log) / implied_S(log) vs epoch, with per-stage
    shading + best-point markers + plateau-onset vlines + a per-stage text box on
    the d_seg panel ({t2best, descent, dead-tail, t2demonstrate}). Calm dark style."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True)
    fig.patch.set_facecolor(_BG)
    ep_all = [r["epoch"] for r in rows]
    xlo, xhi = (min(ep_all), max(ep_all)) if ep_all else (0, 1)
    if xhi <= xlo:
        xhi = xlo + 1
    stages = dynamics["stages"]

    def _shade(ax):
        for s in stages:
            ax.axvspan(s["onset_epoch"], s["end_epoch"], color=_stage_color(s["stage"]),
                       alpha=0.22, lw=0)
            if s.get("plateau_onset_epoch") is not None:
                ax.axvline(s["plateau_onset_epoch"], ls="--", lw=1.0, color=_MUTED, alpha=0.7)

    def _panel(ax, key, color, title, sub, logy=True, hline=None, hlabel=None,
               best_key="best_epoch", best_val_key=None):
        ax.set_xlim(xlo, xhi)
        _style_ax(ax)
        xy = [(r["epoch"], r.get(key)) for r in rows
              if isinstance(r.get(key), (int, float)) and math.isfinite(r.get(key))]
        if xy:
            xs, ys = zip(*xy)
            ax.plot(xs, ys, "-o", ms=3.2, lw=1.5, color=color)
            if logy and any(v > 0 for v in ys):
                ax.set_yscale("log")
        # best-point markers per stage
        for s in stages:
            be = s.get(best_key)
            bv = s.get(best_val_key) if best_val_key else None
            if be is not None and bv is not None and math.isfinite(bv) and bv > 0:
                ax.scatter([be], [bv], s=42, facecolor="none", edgecolor=_BEST,
                           lw=1.4, zorder=5)
        ax.set_title(title, fontsize=10.5, color=_FG, loc="left")
        ax.set_ylabel(sub, fontsize=8.5, color=_MUTED)
        if hline is not None:
            ax.axhline(hline, ls=":", lw=1.2, color=_GOAL, alpha=0.9)
            ax.text(xlo, hline, f" {hlabel or ''}", color=_GOAL, fontsize=8, va="bottom")
        _shade(ax)
        # stage labels along the top
        ylim = ax.get_ylim()
        for s in stages:
            xmid = (s["onset_epoch"] + s["end_epoch"]) / 2.0
            ax.text(xmid, ylim[1], _stage_short(s["stage"]), color=_FG, fontsize=8.5,
                    ha="center", va="top", alpha=0.9)

    _panel(axes[0], "d_seg", _ACC,
           "d_seg — realized SegNet-argmax disagreement (log-y, LOWER better)",
           "d_seg", logy=True, hline=0.00112, hlabel="sub-0.19 goal",
           best_key="best_epoch", best_val_key="best_d_seg")
    _panel(axes[1], "d_pose", _POSE,
           "d_pose — realized PoseNet MSE 6-dim (log-y, LOWER better)",
           "d_pose", logy=True, hline=9e-4, hlabel="existence-proof ~9e-4",
           best_key="best_d_pose_epoch", best_val_key="best_d_pose")
    _panel(axes[2], "implied_S", _SVAL,
           "implied_S — ADVISORY mid-training estimate (NOT the contest score)",
           "implied_S", logy=True, hline=0.19110, hlabel="pointer 0.19110",
           best_key=None, best_val_key=None)
    axes[2].set_xlabel("epoch", fontsize=9, color=_MUTED)

    # per-stage text box on the d_seg panel
    box = dict(boxstyle="round,pad=0.3", fc=_PANEL, ec=_GRIDC, alpha=0.9)
    for s in stages:
        if s.get("time_to_best") is None:
            continue
        descent = s.get("descent_rate_dseg_per_epoch")
        txt = f"{s['stage_short']}\nt2best {s['time_to_best']}"
        if descent is not None:
            txt += f"\nrate {descent:.1e}/ep"
        dead = s.get("dead_tail_epochs", 0)
        t2d = s.get("time_to_demonstrate_dseg")
        txt += f"\ndead {dead}"
        if t2d is not None:
            txt += f"\nt2dem {t2d}"
        xmid = (s["onset_epoch"] + s["end_epoch"]) / 2.0
        axes[0].text(xmid, 0.04, txt, transform=axes[0].get_xaxis_transform(),
                     color=_FG, fontsize=7.5, ha="center", va="bottom", bbox=box, zorder=6)

    fig.suptitle(
        f"Level-set witness trajectory dynamics — {dynamics['n_verdicts']} verdicts"
        f"  ·  [macOS-MLX advisory NON-PROMOTABLE]  ·  pointer 0.19110",
        color=_FG, fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, facecolor=_BG)
    plt.close(fig)
    out.write_bytes(buf.getvalue())
    return out


# ─────────────────────────── main ────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", default=None,
                    help="explicit verdict log (repeatable to MERGE a split "
                         "curriculum; later --log wins per epoch). Overrides --log-glob.")
    ap.add_argument("--log-glob", default=".omx/tmp/levelset_amort_*.log",
                    help="when no --log: read the NEWEST verdict-bearing log matching this glob")
    ap.add_argument("--out", default=".omx/tmp/trajectory/witness_trajectory_dynamics.png",
                    help="output PNG path")
    ap.add_argument("--json", default=".omx/tmp/trajectory/witness_trajectory_dynamics.json",
                    help="machine-readable dynamics dump path")
    ap.add_argument("--tau", type=int, default=300, help="tau_softplus start epoch (stage inference)")
    ap.add_argument("--l7", type=int, default=900, help="l7_softplus start epoch (stage inference)")
    ap.add_argument("--muon", type=int, default=None,
                    help="optional Muon sub-phase start epoch (relabels l7 rows at epoch>=muon)")
    ap.add_argument("--plateau-k", type=int, default=3,
                    help="K consecutive non-improving verdicts that define a plateau (default 3)")
    ap.add_argument("--demonstrate-frac", type=float, default=0.5,
                    help="fraction of the stage's total d_seg drop that counts as "
                         "'behavior demonstrated' (default 0.5)")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG (table + JSON only)")
    args = ap.parse_args()

    logs = resolve_logs(args.log, args.log_glob)
    if not logs:
        print(json.dumps({"stage": "trajectory", "error": "no verdict-bearing log resolved",
                          "log_glob": args.log_glob}))
        return
    rows = merge_rows([parse_verdicts(p) for p in logs])
    if not rows:
        print(json.dumps({"stage": "trajectory", "error": "no verdict rows parsed",
                          "logs": [str(p) for p in logs]}))
        return

    dynamics = compute_dynamics(rows, args.tau, args.l7, args.muon,
                                args.plateau_k, args.demonstrate_frac)
    dynamics["logs"] = [str(p) for p in logs]

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(dynamics, indent=2))

    png_note = None
    if not args.no_plot:
        png = render_png(rows, dynamics, args.out)
        png_note = str(png)

    print(render_table(dynamics))
    print("")
    print(json.dumps({"stage": "trajectory", "rendered": True,
                      "logs": [str(p) for p in logs], "n_verdicts": len(rows),
                      "json": str(json_path), "png": png_note}))


if __name__ == "__main__":
    main()
