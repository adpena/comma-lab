#!/usr/bin/env python3
"""Build the E-AXIS (training-time) convergence dataset by mining trajectory logs.

This tool is a $0, read-only MINER. It does NO training and NEVER touches a GPU/MPS
device. It scans ``experiments/results/**`` for two trajectory-log schemas
(torch_vehicle + capstone), extracts per-run ``d_seg(epoch)`` / ``d_pose(epoch)`` /
``score(epoch)`` curves together with the run's config (base_channels, optimizer,
n_pairs, levers, stage, devices), and writes a single unified JSON dataset.

Authority: every row is [advisory] / NON-PROMOTABLE. d_seg/d_pose come from the
trajectory logs' own evaluated rows (rendered by the run, on its own authority
device); score is recomputed via ``tac.contest_score.compute_contest_score`` from
the logged (d_seg, d_pose, archive_bytes) so the score column is internally
consistent with the canonical contest math. The numbers are mined measurements,
NOT new measurements — no fabrication (NO-FAKE).

Cross-check discipline (per feedback_terminal_conclusion_needs_existence_proof_crosscheck):
a SHORT run's final d_seg is NOT a converged floor. This miner records the raw
curves only; convergence/extrapolation modeling lives in the fitting tool and is
always labelled. PR95 reaches d_seg 5.6e-4 in 29,650 epochs — the existence proof.

Usage:
    .venv/bin/python tools/build_convergence_dataset.py \
        --results-root experiments/results \
        --out .omx/research/eaxis_convergence_dataset_20260623.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from tac.contest_score import compute_contest_score
except Exception:  # pragma: no cover - import guard
    compute_contest_score = None  # type: ignore


# Canonical existence-proof anchors (measured, public). Used by the modeling tool
# for the cross-check; recorded here so the dataset is self-describing.
EXISTENCE_PROOFS = {
    "pr95_8stage_29650ep": {
        "d_seg": 5.6e-4,
        "epochs": 29650,
        "source": "CLAUDE.md L14 + feedback_terminal_conclusion_needs_existence_proof_crosscheck",
        "note": "PR95 8-stage curriculum reaches d_seg 5.6e-4; the canonical d_seg existence proof",
    },
    "frontier_pr101_class": {
        "score": 0.19110,
        "source": ".omx/state/canonical_frontier_pointer.json (pointer-only)",
        "note": "borrowed PR101-class byte-recode; the current exact frontier",
    },
}


@dataclass
class CurvePoint:
    global_epoch: int
    d_seg: float | None
    d_pose: float | None
    score: float | None  # recomputed from contest math when bytes available
    rate: float | None
    archive_bytes: int | None
    wall_clock_s: float | None
    stage_name: str | None
    muon_lr: float | None = None
    adamw_lr: float | None = None


@dataclass
class RunCurve:
    run_id: str
    path: str
    schema: str  # "torch_vehicle" | "capstone"
    config: dict = field(default_factory=dict)
    n_eval_points: int = 0
    epoch_span: tuple[int, int] | None = None
    final_d_seg: float | None = None
    min_d_seg: float | None = None
    final_d_pose: float | None = None
    final_score: float | None = None
    wall_clock_s: float | None = None
    points: list[dict] = field(default_factory=list)


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _safe_int(x) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _config_from_metadata(run_dir: Path) -> dict:
    """Pull config from torch_vehicle_summary.json + PROVENANCE.json if present."""
    cfg: dict = {}
    summ = _read_json(run_dir / "torch_vehicle_summary.json")
    if summ:
        rm = summ.get("run_meta", {}) or {}
        for k in ("base_channels", "device", "ema_decay", "latent_dim", "n_pairs",
                  "total_epoch_budget"):
            if k in rm:
                cfg[k] = rm[k]
        if "best_score" in summ:
            cfg["summary_best_score"] = summ["best_score"]
        if "best_ep" in summ:
            cfg["summary_best_ep"] = summ["best_ep"]
    prov = _read_json(run_dir / "PROVENANCE.json")
    if prov:
        for k in ("levers", "train_device", "authority_device", "async_eval",
                  "total_epoch_budget", "eval_every", "n_pairs", "base_channels",
                  "seed", "command", "experiment"):
            if k in prov:
                cfg[k] = prov[k]
    return cfg


def _recompute_score(d_seg, d_pose, archive_bytes) -> float | None:
    if compute_contest_score is None:
        return None
    if d_seg is None or d_pose is None or archive_bytes is None:
        return None
    try:
        return float(compute_contest_score(d_seg, d_pose, archive_bytes))
    except Exception:
        return None


def parse_torch_vehicle(traj_path: Path) -> RunCurve | None:
    run_dir = traj_path.parent
    run_id = str(run_dir.relative_to(REPO_ROOT / "experiments" / "results"))
    cfg = _config_from_metadata(run_dir)
    points: list[CurvePoint] = []
    last_bytes: int | None = None
    try:
        with traj_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("evaluated"):
                    continue
                d_seg = _safe_float(row.get("d_seg"))
                if d_seg is None:
                    continue
                d_pose = _safe_float(row.get("d_pose"))
                ab = _safe_int(row.get("archive_bytes"))
                if ab is not None:
                    last_bytes = ab
                score = _safe_float(row.get("score"))
                # Prefer a contest-math recompute when bytes are present (internally
                # consistent); fall back to logged score, then to rate-free proxy.
                recomputed = _recompute_score(d_seg, d_pose, ab if ab is not None else last_bytes)
                points.append(CurvePoint(
                    global_epoch=_safe_int(row.get("global_epoch")) or 0,
                    d_seg=d_seg,
                    d_pose=d_pose,
                    score=recomputed if recomputed is not None else score,
                    rate=_safe_float(row.get("rate")),
                    archive_bytes=ab,
                    wall_clock_s=_safe_float(row.get("wall_clock_s")),
                    stage_name=row.get("stage_name"),
                    muon_lr=_safe_float(row.get("muon_lr")),
                    adamw_lr=_safe_float(row.get("adamw_lr")),
                ))
    except OSError:
        return None
    if not points:
        return None
    return _finalize(run_id, traj_path, "torch_vehicle", cfg, points)


def parse_capstone(traj_path: Path) -> RunCurve | None:
    run_dir = traj_path.parent
    run_id = str(run_dir.relative_to(REPO_ROOT / "experiments" / "results"))
    cfg: dict = {}
    points: list[CurvePoint] = []
    try:
        with traj_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "init":
                    for k in ("base_channels", "n_pairs"):
                        if k in row:
                            cfg[k] = row[k]
                    continue
                d_seg = _safe_float(row.get("exact_d_seg"))
                if d_seg is None:
                    continue
                # capstone uses global_epoch or epoch
                ge = row.get("global_epoch")
                if ge is None:
                    ge = row.get("epoch")
                stage = row.get("stage") or row.get("stage_name")
                points.append(CurvePoint(
                    global_epoch=_safe_int(ge) or 0,
                    d_seg=d_seg,
                    d_pose=_safe_float(row.get("mean_d_pose")),
                    score=None,  # capstone has no archive bytes -> no contest score
                    rate=None,
                    archive_bytes=None,
                    wall_clock_s=_safe_float(row.get("elapsed_s")),
                    stage_name=stage,
                ))
    except OSError:
        return None
    if not points:
        return None
    # Inherit any base_channels from dir-name fallback
    if "base_channels" not in cfg:
        for tok in run_id.replace("/", "_").split("_"):
            if tok.startswith("b") and tok[1:].isdigit():
                cfg["base_channels"] = int(tok[1:])
                break
            if tok.startswith("bc") and tok[2:].isdigit():
                cfg["base_channels"] = int(tok[2:])
                break
    return _finalize(run_id, traj_path, "capstone", cfg, points)


def _finalize(run_id, traj_path, schema, cfg, points: list[CurvePoint]) -> RunCurve:
    points.sort(key=lambda p: p.global_epoch)
    d_segs = [p.d_seg for p in points if p.d_seg is not None]
    rc = RunCurve(
        run_id=run_id,
        path=str(traj_path.relative_to(REPO_ROOT)),
        schema=schema,
        config=cfg,
        n_eval_points=len(points),
        epoch_span=(points[0].global_epoch, points[-1].global_epoch) if points else None,
        final_d_seg=points[-1].d_seg,
        min_d_seg=min(d_segs) if d_segs else None,
        final_d_pose=points[-1].d_pose,
        final_score=points[-1].score,
        wall_clock_s=points[-1].wall_clock_s,
        points=[asdict(p) for p in points],
    )
    return rc


def build_dataset(results_root: Path) -> dict:
    runs: list[RunCurve] = []
    for dirpath, _dirnames, filenames in os.walk(results_root):
        for fn in filenames:
            if fn == "torch_vehicle_trajectory.jsonl":
                rc = parse_torch_vehicle(Path(dirpath) / fn)
                if rc:
                    runs.append(rc)
            elif fn == "trajectory.jsonl":
                rc = parse_capstone(Path(dirpath) / fn)
                if rc:
                    runs.append(rc)
    runs.sort(key=lambda r: (-(r.n_eval_points), r.run_id))
    return {
        "schema_version": 1,
        "authority": "[advisory] NON-PROMOTABLE; mined measurements from trajectory "
                     "logs, NOT new measurements; score recomputed via tac.contest_score",
        "generated_by": "tools/build_convergence_dataset.py",
        "existence_proofs": EXISTENCE_PROOFS,
        "n_runs": len(runs),
        "n_total_eval_points": sum(r.n_eval_points for r in runs),
        "runs": [asdict(r) for r in runs],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="experiments/results")
    ap.add_argument("--out", default=".omx/research/eaxis_convergence_dataset_20260623.json")
    ap.add_argument("--summary", action="store_true", help="print a human-readable summary")
    args = ap.parse_args(argv)

    results_root = (REPO_ROOT / args.results_root).resolve()
    if not results_root.exists():
        print(f"results root not found: {results_root}", file=sys.stderr)
        return 2

    ds = build_dataset(results_root)
    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ds, indent=2))
    print(f"wrote {out_path} : {ds['n_runs']} runs, {ds['n_total_eval_points']} eval points")

    if args.summary:
        print("\n=== top runs by eval-point count ===")
        for r in ds["runs"][:25]:
            cfg = r["config"]
            bc = cfg.get("base_channels", "?")
            npairs = cfg.get("n_pairs", "?")
            span = r["epoch_span"]
            print(f"  {r['n_eval_points']:4d}pts  bc={bc} n={npairs}  ep{span}  "
                  f"final_dseg={r['final_d_seg']}  min_dseg={r['min_d_seg']}  {r['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
