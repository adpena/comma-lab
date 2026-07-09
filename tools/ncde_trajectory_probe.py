#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 shadow probe: fit a Linear-NCDE to a witness run's telemetry + emit hit->solve advisories.

READ-ONLY on the run dir (never touches the trainer). Reads ``<run-dir>/run.log`` (JSONL rows:
``{stage: loss_terms}`` per-epoch + ``{stage: verdict}`` periodic), builds two telemetry paths,
fits the Linear-NCDE over sliding windows, and prints/emits the advisory series
``{stage: "ncde_trajectory", ...}`` -- the shape the shadow controller can consume later.

  * DENSE loss path  (per-epoch loss_terms): state = [log total, log gnorm, seg_share,
    island_share]; control = [softmax_temp, hosc_beta]. Many points -> plateau-slope backtest.
  * VERDICT d_seg path (per verdict): state = [log d_seg, log ep_loss]; control =
    [softmax_temp, hosc_beta] forward-filled from the nearest loss_terms row. The SCORE-relevant
    hit->solve on d_seg (the #341 basin consumer).

Advisory-only forever (CONTAINMENT): NEVER actuates. Usage:

    .venv/bin/python tools/ncde_trajectory_probe.py --run-dir experiments/results/<dir>
    .venv/bin/python tools/ncde_trajectory_probe.py --run-dir <dir> --emit    # write advisory jsonl
    .venv/bin/python tools/ncde_trajectory_probe.py --run-dir <dir> --backtest # holdout MAPE

FORMALIZATION_PENDING: see ``tac.witness_control.ncde_trajectory`` module docstring -- the
predicted-vs-empirical residual law needs N>=5 completed runs to anchor honestly.

means != ends: advisory sensor, NOT a score; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tac.witness_control.ncde_trajectory import (
    HitSolveConfig,
    TelemetryPath,
    backtest_holdout,
    fit_sliding_windows,
    sliding_onestep_backtest,
)


def _read_rows(run_log: Path) -> list[dict]:
    rows: list[dict] = []
    for line in run_log.read_text().splitlines():
        line = line.strip()
        if not (line.startswith("{") and '"stage"' in line):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _safe_log(x: float, floor: float = 1e-9) -> float:
    return math.log(max(float(x), floor))


def build_dense_path(rows: list[dict]) -> TelemetryPath | None:
    """State = [log_total, log_gnorm, seg_share, island_share]; control = [softmax_temp, hosc_beta]."""
    eps: list[float] = []
    st: list[list[float]] = []
    ct: list[list[float]] = []
    seen: set[int] = set()
    for r in rows:
        if r.get("stage") != "loss_terms":
            continue
        ep = r.get("ep")
        if ep is None or ep in seen:
            continue
        terms = r.get("terms", {})
        total = r.get("total")
        gnorm = r.get("gnorm")
        if total is None or gnorm is None or not terms:
            continue
        tot = float(total)
        if tot <= 0 or not math.isfinite(tot):
            continue
        seg = float(terms.get("seg", 0.0))
        island = float(terms.get("island_amplify", 0.0))
        seg_share = seg / tot if tot > 0 else 0.0
        island_share = island / tot if tot > 0 else 0.0
        seen.add(ep)
        eps.append(float(ep))
        st.append([_safe_log(tot), _safe_log(gnorm), seg_share, island_share])
        ct.append([float(r.get("softmax_temp", 1.0)), float(r.get("hosc_beta", 1.0))])
    if len(eps) < 8:
        return None
    order = np.argsort(eps)
    return TelemetryPath(
        epochs=np.array(eps)[order],
        state=np.array(st)[order],
        control=np.array(ct)[order],
        state_names=("log_total", "log_gnorm", "seg_share", "island_share"),
        control_names=("softmax_temp", "hosc_beta"),
    )


def build_verdict_path(rows: list[dict]) -> TelemetryPath | None:
    """State = [log_d_seg, log_ep_loss]; control = [softmax_temp, hosc_beta] fwd-filled."""
    # forward-fill control from loss_terms by epoch.
    ctrl_by_ep: dict[int, tuple[float, float]] = {}
    for r in rows:
        if r.get("stage") == "loss_terms" and r.get("ep") is not None:
            ctrl_by_ep[int(r["ep"])] = (
                float(r.get("softmax_temp", 1.0)),
                float(r.get("hosc_beta", 1.0)),
            )
    ctrl_eps = sorted(ctrl_by_ep)

    def nearest_ctrl(ep: float) -> tuple[float, float]:
        if not ctrl_eps:
            return (1.0, 1.0)
        prior = [e for e in ctrl_eps if e <= ep]
        key = prior[-1] if prior else ctrl_eps[0]
        return ctrl_by_ep[key]

    eps: list[float] = []
    st: list[list[float]] = []
    ct: list[list[float]] = []
    seen: set[int] = set()
    for r in rows:
        if r.get("stage") != "verdict":
            continue
        ep = r.get("epoch")
        d_seg = r.get("d_seg")
        if ep is None or d_seg is None or ep in seen:
            continue
        ds = float(d_seg)
        if ds <= 0 or not math.isfinite(ds):
            continue
        ep_loss = r.get("ep_loss")
        log_ep_loss = _safe_log(ep_loss) if ep_loss not in (None, 0) else _safe_log(ds)
        seen.add(ep)
        eps.append(float(ep))
        st.append([_safe_log(ds), log_ep_loss])
        ct.append(list(nearest_ctrl(float(ep))))
    if len(eps) < 8:
        return None
    order = np.argsort(eps)
    return TelemetryPath(
        epochs=np.array(eps)[order],
        state=np.array(st)[order],
        control=np.array(ct)[order],
        state_names=("log_d_seg", "log_ep_loss"),
        control_names=("softmax_temp", "hosc_beta"),
    )


def _run_is_active(run_dir: Path, *, stale_s: float = 600.0) -> bool:
    """True if the run appears LIVE (run.log touched within stale_s). Protects READ-ONLY dirs."""
    import time

    log = run_dir / "run.log"
    if not log.exists():
        return False
    return (time.time() - log.stat().st_mtime) < stale_s


def run_probe(
    run_dir: Path, *, window: int, emit: bool, do_backtest: bool, out: Path | None = None
) -> dict:
    run_log = run_dir / "run.log"
    if not run_log.exists():
        raise FileNotFoundError(f"no run.log at {run_log}")
    rows = _read_rows(run_log)
    dense = build_dense_path(rows)
    verdict = build_verdict_path(rows)

    report: dict = {
        "run_dir": str(run_dir),
        "n_rows": len(rows),
        "dense_points": dense.n if dense else 0,
        "verdict_points": verdict.n if verdict else 0,
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
    }

    advisories: list[dict] = []
    # dense: plateau-slope tracking (context sensor).
    if dense is not None and dense.n >= window:
        wf = fit_sliding_windows(
            dense, window=min(window, dense.n), stride=max(1, (dense.n - window) // 20 or 1),
            detector_cfg=HitSolveConfig(target_state="log_total", min_r2=0.3),
        )
        for w in wf:
            if w.verdict is not None:
                advisories.append(w.verdict.to_row(w.end_epoch, stage="dense_loss"))
        report["dense_windows"] = len(wf)
        if wf:
            report["dense_latest"] = wf[-1].ncde.to_dict()

    # verdict: the SCORE-relevant d_seg hit->solve (the #341 basin consumer).
    if verdict is not None:
        vwin = min(window, verdict.n)
        if verdict.n >= max(8, vwin):
            vwf = fit_sliding_windows(
                verdict, window=vwin, stride=1,
                detector_cfg=HitSolveConfig(target_state="log_d_seg"),
            )
            for w in vwf:
                if w.verdict is not None:
                    advisories.append(w.verdict.to_row(w.end_epoch, stage="verdict_d_seg"))
            report["verdict_windows"] = len(vwf)
            if vwf and vwf[-1].verdict is not None:
                report["verdict_latest_advisory"] = vwf[-1].verdict.to_row(
                    vwf[-1].end_epoch, stage="verdict_d_seg"
                )

    report["n_advisories"] = len(advisories)
    report["n_fires"] = sum(1 for a in advisories if a.get("fire"))

    if do_backtest and dense is not None and dense.n >= 12:
        # Global holdout: STRESS test (one linear CDE across a whole multi-stage run).
        try:
            report["backtest_dense_global_holdout"] = backtest_holdout(
                dense, train_frac=0.6, target_state="log_total"
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            report["backtest_dense_global_holdout_error"] = str(e)
        # Sliding one-step: the sensor's REAL operating accuracy (the faithful backtest).
        try:
            report["backtest_dense_sliding_onestep"] = sliding_onestep_backtest(
                dense, window=min(window, dense.n - 1), target_state="log_total"
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            report["backtest_dense_sliding_onestep_error"] = str(e)
    if do_backtest and verdict is not None and verdict.n >= 12:
        try:
            report["backtest_verdict_sliding_onestep"] = sliding_onestep_backtest(
                verdict, window=min(window, verdict.n - 1), target_state="log_d_seg"
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            report["backtest_verdict_sliding_onestep_error"] = str(e)

    if emit and advisories:
        target = out if out is not None else (run_dir / "ncde_trajectory_shadow.jsonl")
        # READ-ONLY guard: never write a sidecar INTO an active/live run dir (CONTAINMENT).
        writing_into_run_dir = target.resolve().parent == run_dir.resolve()
        if writing_into_run_dir and _run_is_active(run_dir):
            report["emit_refused"] = (
                "run appears ACTIVE (run.log touched < 600s ago); refusing to write into a "
                "read-only live run dir. Pass --out <path> to emit elsewhere."
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(json.dumps(a) for a in advisories) + "\n")
            report["emitted_to"] = str(target)

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--window", type=int, default=12, help="sliding-window length (points)")
    ap.add_argument("--emit", action="store_true", help="write ncde_trajectory_shadow.jsonl")
    ap.add_argument("--out", type=Path, default=None,
                    help="emit target path (default: <run-dir>/ncde_trajectory_shadow.jsonl; "
                         "refused into an ACTIVE run dir -- use --out to emit elsewhere)")
    ap.add_argument("--backtest", action="store_true", help="run holdout MAPE backtest")
    args = ap.parse_args(argv)
    report = run_probe(
        args.run_dir, window=args.window, emit=args.emit, do_backtest=args.backtest,
        out=args.out,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
