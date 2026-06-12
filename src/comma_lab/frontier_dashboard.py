# SPDX-License-Identifier: MIT
"""Realtime browser dashboard for the contest frontier + a live training run.

A small self-contained Flask app (no template files, no JS build) that renders:

* a HEADER bar with the canonical frontier CPU + CUDA scores, each with its
  metadata (when it was measured, the technique/architecture-class, archive
  bytes, evidence tag) and the sub-0.19 / sub-0.15 target ladder;
* a RUN STATUS bar with when the live run started, a tqdm-style progress bar,
  and a dynamic ETA to completion;
* a REALTIME PANEL with the latest authority d_seg / d_pose / score + a d_seg
  sparkline (polled every few seconds);
* a HISTORICAL TABLE of recent eval epochs;
* RUN SUMMARY / META stats across the whole run.

Data sources (read-only, no mutation):
* ``.omx/state/canonical_frontier_pointer.json`` — the frontier SoT
  (CLAUDE.md "Frontier scores are pointer-only").
* the live run's append-only trajectory JSONL — either ``trajectory.jsonl``
  (capstone curriculum daemon) or ``torch_vehicle_trajectory.jsonl``
  (torch-vehicle basin). Both formats are normalized.

Authority discipline: the trajectory d_seg/d_pose are the torch-CPU AUTHORITY
eval the run records (never MPS). This dashboard only DISPLAYS them; it makes no
score claim and never writes anything.

Run it:

    .venv/bin/python -m comma_lab.frontier_dashboard --port 8765
    # then open http://127.0.0.1:8765/

It auto-discovers the most-recently-updated run under ``experiments/results/``;
pin a specific one with ``--run-dir`` or the ``?run=<name>`` query param.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Annotated, Any

from flask import Flask, Response, render_template_string, request
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

RATE_DENOM = 37_545_489  # contest archive-size normalizer
TARGET_FLOOR = 0.19      # T_1 — floor of acceptable
TARGET_GOAL = 0.15       # T_3 — THE target
DEFAULT_TOTAL_EPOCHS = 29650  # faithful full PR95 8-stage curriculum
# (3000+5650+1500+500+9000+2000+3000+5000); the basin runs this when the launcher's
# total_epoch_budget is unset (None). A compressed budget is auto-detected per-run.
DEFAULT_PORT = 8765          # canonical dashboard port (hardcoded)
DEFAULT_HOST = "127.0.0.1"

_TRAJECTORY_NAMES = ("torch_vehicle_trajectory.jsonl", "trajectory.jsonl")


def _repo_root() -> Path:
    # this file is src/comma_lab/frontier_dashboard.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Frontier pointer
# --------------------------------------------------------------------------- #
def load_frontier(root: Path) -> dict[str, Any]:
    p = root / ".omx" / "state" / "canonical_frontier_pointer.json"
    out: dict[str, Any] = {"cpu": None, "cuda": None, "last_refreshed_utc": None}
    try:
        blob = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["last_refreshed_utc"] = blob.get("last_refreshed_utc")
    for key, axis in (("cpu", "our_local_frontier_contest_cpu"),
                      ("cuda", "our_local_frontier_contest_cuda")):
        row = blob.get(axis)
        if not row:
            continue
        extra = row.get("extra") or {}
        out[key] = {
            "score": row.get("score"),
            "evidence_tag": extra.get("evidence_tag") or row.get("evidence_grade"),
            "technique": extra.get("architecture_class"),
            "archive_bytes": extra.get("archive_bytes"),
            "archive_sha256": (row.get("archive_sha256") or "")[:12],
            "hardware": row.get("hardware_substrate"),
            "measured_at_utc": row.get("measured_at_utc"),
        }
    return out


# --------------------------------------------------------------------------- #
# Run trajectory (two formats, normalized)
# --------------------------------------------------------------------------- #
# Ancillary A/B arms / probes — real basin runs are preferred over these when
# auto-selecting the "live run" (they still appear in available_runs and can be
# pinned explicitly via ?run=).
_ANCILLARY_MARKERS = (
    "chaos_control", "descent_ab", "_gate", "/arm_", "_smoke", "gt_targets",
    "descent_equiv",
)


def _is_ancillary(path: Path) -> bool:
    s = str(path.parent).replace("\\", "/")
    return any(m in s for m in _ANCILLARY_MARKERS)


_RUNS_CACHE: dict[str, Any] = {"ts": 0.0, "runs": []}
_RUNS_TTL_S = 15.0


def discover_runs(root: Path, *, ttl: float = _RUNS_TTL_S) -> list[Path]:
    """Trajectory files under experiments/results/, newest mtime first.

    CRITICAL: experiments/results/ holds 200k+ files (inflated frames,
    checkpoints, artifacts). A recursive ``**`` glob walks ALL of them (~3s) and
    would hang the dashboard on every poll. Instead we glob only the BOUNDED
    depths where trajectory files actually live (``results/<run>/`` and one or
    two levels under, e.g. ``.../arm_clean/``) and CACHE the result for ``ttl``
    seconds — the run LIST changes rarely; the live trajectory CONTENT is read
    fresh per request (one small file)."""
    now = time.time()
    if (now - _RUNS_CACHE["ts"]) < ttl and _RUNS_CACHE["runs"]:
        return _RUNS_CACHE["runs"]
    results = root / "experiments" / "results"
    seen: set[Path] = set()
    found: list[Path] = []
    for name in _TRAJECTORY_NAMES:
        for depth in ("*/", "*/*/", "*/*/*/"):  # bounded — NOT a full ** walk
            for p in results.glob(depth + name):
                if p in seen or not p.is_file():
                    continue
                seen.add(p)
                found.append(p)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    _RUNS_CACHE["ts"] = now
    _RUNS_CACHE["runs"] = found
    return found


def pick_live_run(root: Path, want: str | None) -> Path | None:
    """Pin by substring (``want``) across ALL runs; otherwise auto-select the
    newest NON-ancillary run (a real basin), falling back to newest overall."""
    runs = discover_runs(root)
    if want:
        for p in runs:
            if want in str(p.parent):
                return p
    primary = [p for p in runs if not _is_ancillary(p)]
    if primary:
        return primary[0]
    return runs[0] if runs else None


def _f(row: dict, *keys: str) -> float | None:
    """First finite numeric value among ``keys`` (NaN/Inf -> None). Non-finite
    floats must NEVER enter the payload: Python emits literal ``NaN``/``Infinity``
    (invalid JSON) which the browser's strict ``JSON.parse`` rejects."""
    for k in keys:
        v = row.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _sanitize_count(obj: Any) -> tuple[Any, int]:
    """Recursively replace non-finite floats (NaN/Inf/-Inf) with None and COUNT
    how many were replaced — so the conversion is surfaced (``data_warnings``),
    never silent. The browser's strict JSON.parse rejects NaN/Infinity, which is
    what silently broke the dashboard before."""
    if isinstance(obj, float):
        return (obj, 0) if math.isfinite(obj) else (None, 1)
    if isinstance(obj, dict):
        out, n = {}, 0
        for k, v in obj.items():
            out[k], c = _sanitize_count(v)
            n += c
        return out, n
    if isinstance(obj, (list, tuple)):
        out_l, n = [], 0
        for v in obj:
            sv, c = _sanitize_count(v)
            out_l.append(sv)
            n += c
        return out_l, n
    return obj, 0


# ---- Pydantic response schema ------------------------------------------------
# Guarantees an RFC-valid JSON payload (non-finite floats -> null EXPLICITLY via
# FiniteFloat) and FAILS LOUDLY on a structurally-wrong payload (returns a visible
# error, not a silent break). Combined with json.dumps(allow_nan=False) at the
# route, invalid JSON can NEVER ship silently.
def _finite_or_none(v: Any) -> Any:
    return None if isinstance(v, float) and not math.isfinite(v) else v


FiniteFloat = Annotated[float | None, BeforeValidator(_finite_or_none)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate evolving/extra fields


class FrontierAxisModel(_Model):
    score: FiniteFloat = None
    evidence_tag: str | None = None
    technique: str | None = None
    archive_bytes: int | None = None
    archive_sha256: str | None = None
    hardware: str | None = None
    measured_at_utc: str | None = None


class FrontierModel(_Model):
    cpu: FrontierAxisModel | None = None
    cuda: FrontierAxisModel | None = None
    last_refreshed_utc: str | None = None
    error: str | None = None


class EpochRowModel(_Model):
    epoch: int | None = None
    stage: str | None = None
    d_seg: FiniteFloat = None
    d_pose: FiniteFloat = None
    rate: FiniteFloat = None
    archive_bytes: int | None = None
    best_d_seg: FiniteFloat = None
    elapsed_s: FiniteFloat = None
    lr: FiniteFloat = None
    loss: FiniteFloat = None
    pose_mse: FiniteFloat = None
    grad_norm: FiniteFloat = None
    score: FiniteFloat = None
    score_is_full: bool | None = None


class TrainTelModel(_Model):
    epoch: int | None = None
    stage: str | None = None
    loss: FiniteFloat = None
    pose_mse: FiniteFloat = None
    grad_norm: FiniteFloat = None
    lr: FiniteFloat = None
    elapsed_s: FiniteFloat = None


class SummaryModel(_Model):
    n_records: int | None = None
    n_eval_epochs: int | None = None
    init_d_seg: FiniteFloat = None
    d_seg_descent: FiniteFloat = None
    d_seg_min: FiniteFloat = None
    avg_s_per_epoch: FiniteFloat = None
    median_s_per_epoch: FiniteFloat = None
    elapsed_s: FiniteFloat = None
    recent_dseg_per_epoch: FiniteFloat = None
    latest_loss: FiniteFloat = None


class RunModel(_Model):
    run_dir: str
    trajectory_file: str | None = None
    mtime: FiniteFloat = None
    is_live: bool | None = None
    is_done: bool | None = None
    started_unix: FiniteFloat = None
    current_epoch: int | None = None
    total_epochs: int | None = None
    progress: FiniteFloat = None
    eta_s: FiniteFloat = None
    eta_median_s: FiniteFloat = None
    train: TrainTelModel | None = None
    latest_eval: EpochRowModel | None = None
    best: EpochRowModel | None = None
    summary: SummaryModel | None = None
    rows: list[EpochRowModel] = []
    spark: list[FiniteFloat] = []
    spark_kind: str | None = None
    error: str | None = None


class ApiStateModel(_Model):
    frontier: FrontierModel
    targets: dict
    now_unix: FiniteFloat = None
    run: RunModel | None = None
    available_runs: list[str] = []
    data_warnings: list[str] = []
    error: str | None = None


def serialize_state(state: dict) -> tuple[str, int]:
    """Validate + serialize the API state to GUARANTEED-valid JSON. Returns
    (json_body, http_status). Non-finite floats become null (counted into
    ``data_warnings``); a structurally-invalid payload returns a LOUD error body
    (status 200 so the UI renders the message) instead of a silent break."""
    try:
        model = ApiStateModel.model_validate(state)
    except ValidationError as e:
        body = json.dumps({"error": f"payload validation failed: {e}",
                           "frontier": {"cpu": None, "cuda": None},
                           "run": None, "available_runs": []})
        return body, 200
    payload = model.model_dump()
    payload, n_nonfinite = _sanitize_count(payload)   # belt-and-suspenders for any extra field
    if n_nonfinite:
        payload.setdefault("data_warnings", []).append(
            f"{n_nonfinite} non-finite value(s) nulled for JSON validity")
    # allow_nan=False => raises if ANYTHING non-finite slipped through (no silent invalid JSON)
    return json.dumps(payload, allow_nan=False), 200


def _normalize_row(row: dict) -> dict | None:
    """Normalize BOTH formats. The capstone ``trajectory.jsonl`` records only
    eval rows (every row has ``exact_d_seg``); the torch-vehicle
    ``torch_vehicle_trajectory.jsonl`` records EVERY epoch (loss/pose_mse) with
    ``d_seg``/``d_pose`` populated only on eval epochs. Keep ALL rows so the
    realtime per-epoch telemetry is visible; ``d_seg`` is None on non-eval rows."""
    if row.get("event") == "init":
        d = _f(row, "exact_d_seg", "d_seg")
        return {
            "epoch": 0, "stage": "init", "d_seg": d, "d_pose": _f(row, "mean_d_pose", "d_pose"),
            "loss": None, "pose_mse": None, "grad_norm": None,
            "rate": _f(row, "rate"), "archive_bytes": row.get("archive_bytes"),
            "best_d_seg": d,
            "elapsed_s": _f(row, "elapsed_s", "wall_clock_s", "wall_s", "elapsed") or 0.0,
            "lr": _f(row, "lr_scale", "lr"),
        }
    epoch = row.get("global_epoch")
    if epoch is None:
        epoch = row.get("epoch")
    if epoch is None:
        return None
    return {
        "epoch": int(epoch),
        "stage": row.get("stage") or row.get("stage_name") or "",
        "d_seg": _f(row, "exact_d_seg", "d_seg"),  # None on non-eval rows
        "d_pose": _f(row, "mean_d_pose", "d_pose"),
        "loss": _f(row, "loss"),
        "pose_mse": _f(row, "pose_mse"),
        "grad_norm": _f(row, "grad_norm_adamw", "grad_norm_muon", "grad_norm"),
        "rate": _f(row, "rate"),
        "archive_bytes": row.get("archive_bytes"),
        "best_d_seg": _f(row, "best_d_seg"),
        "elapsed_s": _f(row, "elapsed_s", "wall_clock_s", "wall_s", "elapsed"),
        "lr": _f(row, "lr_scale", "lr", "adamw_lr", "muon_lr"),
    }


def _read_run_total_epochs(run_dir: Path) -> int | None:
    """Prefer the run's OWN configured curriculum length so the progress bar/ETA
    track whatever the run is set to (e.g. the full 29,650-epoch PR95 curriculum,
    or an open-ended indefinite run) rather than a fixed dashboard default."""
    try:
        summ = json.loads((run_dir / "torch_vehicle_summary.json").read_text())
        v = (summ.get("run_meta") or {}).get("total_epoch_budget")
        return int(v) if v else None
    except Exception:  # noqa: BLE001
        return None


def _score(d_seg: float | None, d_pose: float | None,
           archive_bytes: int | None) -> tuple[float | None, bool]:
    """Return (S, is_full). Full contest S needs archive_bytes; otherwise a
    distortion-only partial (pre-byte-close) is returned with is_full=False."""
    if d_seg is None:
        return None, False
    s = 100.0 * d_seg
    if d_pose is not None:
        s += math.sqrt(10.0 * max(d_pose, 0.0))
    if archive_bytes:
        return s + 25.0 * archive_bytes / RATE_DENOM, True
    return s, False


def load_run(path: Path, total_epochs: int) -> dict[str, Any]:
    rows: list[dict] = []
    try:
        text = path.read_text()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "rows": []}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial last line being appended — skip
        nr = _normalize_row(raw)
        if nr is not None:
            rows.append(nr)
    rows.sort(key=lambda r: r["epoch"])
    mtime = path.stat().st_mtime
    is_done = any((path.parent / m).exists()
                  for m in ("DONE_MARKER", "done.json", "done.marker", "DONE", "done_marker.json"))

    evals = [r for r in rows if r.get("d_seg") is not None]  # authority-eval rows
    latest_train = rows[-1] if rows else None                # latest ANY epoch (realtime)
    latest_eval = evals[-1] if evals else None               # latest authority eval

    best = None
    for r in evals:
        if best is None or r["d_seg"] < best["d_seg"]:
            best = r

    # started time: latest elapsed_s is seconds-since-start -> start = mtime - elapsed
    if latest_train and latest_train.get("elapsed_s"):
        started_unix = mtime - latest_train["elapsed_s"]
    else:
        started_unix = path.stat().st_ctime

    cur_epoch = latest_train["epoch"] if latest_train else 0
    elapsed_s = (latest_train.get("elapsed_s") if latest_train else None) or max(time.time() - started_unix, 0.0)
    avg_s_per_epoch = (elapsed_s / cur_epoch) if cur_epoch > 0 else None
    run_total = _read_run_total_epochs(path.parent)
    total = max(run_total or total_epochs, cur_epoch)
    eta_s = (avg_s_per_epoch * max(total - cur_epoch, 0)) if avg_s_per_epoch else None
    progress = (cur_epoch / total) if total else 0.0

    def _with_score(r: dict) -> dict:
        s, full = _score(r.get("d_seg"), r.get("d_pose"), r.get("archive_bytes"))
        return {**r, "score": s, "score_is_full": full}

    # per-epoch realtime training telemetry (updates EVERY epoch, not just evals)
    train = None
    if latest_train:
        train = {k: latest_train.get(k) for k in
                 ("epoch", "stage", "loss", "pose_mse", "grad_norm", "lr", "elapsed_s")}

    latest_eval_s = _with_score(latest_eval) if latest_eval else None
    best_s = _with_score(best) if best else None

    # sparkline: prefer the dense per-epoch loss; fall back to eval d_seg
    loss_series = [r["loss"] for r in rows if r.get("loss") is not None][-160:]
    if loss_series:
        spark, spark_kind = loss_series, "loss"
    else:
        spark, spark_kind = [r["d_seg"] for r in evals][-160:], "d_seg"

    d_segs = [r["d_seg"] for r in evals]
    init_d_seg = next((r["d_seg"] for r in rows if r.get("d_seg") is not None), None)
    descent = ((init_d_seg - latest_eval["d_seg"])
               if (init_d_seg is not None and latest_eval) else None)
    # Per-epoch wall-clock deltas -> ROBUST MEDIAN training cadence (excludes the
    # multi-minute eval spikes that inflate the mean). avg_s_per_epoch above is the
    # mean (incl. blocking evals); this median is the true per-training-epoch time.
    _wc = [(r["epoch"], r["elapsed_s"]) for r in rows if r.get("elapsed_s") is not None]
    _dts = [(_wc[i][1] - _wc[i - 1][1]) / (_wc[i][0] - _wc[i - 1][0])
            for i in range(1, len(_wc))
            if _wc[i][0] > _wc[i - 1][0] and _wc[i][1] >= _wc[i - 1][1]]
    median_s_per_epoch = sorted(_dts)[len(_dts) // 2] if _dts else None
    # Prefer the MEDIAN cadence for the ETA: it is robust to RESTARTS (on resume the
    # wall-clock resets while the epoch counter continues, corrupting the mean) AND to
    # eval spikes — so it's the honest steady-state time-to-finish. Fall back to mean.
    if median_s_per_epoch:
        eta_s = median_s_per_epoch * max(total - cur_epoch, 0)
    summary = {
        "n_records": len(rows),
        "n_eval_epochs": len(evals),
        "init_d_seg": init_d_seg,
        "d_seg_descent": descent,
        "d_seg_min": min(d_segs) if d_segs else None,
        "avg_s_per_epoch": avg_s_per_epoch,
        "median_s_per_epoch": median_s_per_epoch,
        "elapsed_s": elapsed_s,
        "recent_dseg_per_epoch": _recent_rate(evals),
        "latest_loss": (latest_train.get("loss") if latest_train else None),
    }

    return {
        "run_dir": str(path.parent.relative_to(_repo_root())),
        "trajectory_file": path.name,
        "mtime": mtime,
        "is_live": (time.time() - mtime) < 180,  # trajectory written in last 3 min
        "is_done": is_done,
        "started_unix": started_unix,
        "current_epoch": cur_epoch,
        "total_epochs": total,
        "progress": progress,
        "eta_s": eta_s,
        # projected ETA at the TRAINING cadence (median, ex-eval) — what async
        # (non-blocking) CPU eval delivers, since the eval no longer stalls training.
        "eta_median_s": (median_s_per_epoch * max(total - cur_epoch, 0)) if median_s_per_epoch else None,
        "train": train,                # realtime per-epoch
        "latest_eval": latest_eval_s,  # periodic authority eval
        "best": best_s,
        "summary": summary,
        "rows": [_with_score(r) for r in evals[-40:]],
        "spark": spark,
        "spark_kind": spark_kind,
    }


def _recent_rate(evals: list[dict]) -> float | None:
    pts = [r for r in evals if r["d_seg"] is not None][-6:]
    if len(pts) < 2:
        return None
    de = pts[-1]["epoch"] - pts[0]["epoch"]
    if de <= 0:
        return None
    return (pts[-1]["d_seg"] - pts[0]["d_seg"]) / de


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
def create_app(*, total_epochs: int = DEFAULT_TOTAL_EPOCHS,
               run_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    root = _repo_root()

    # Warm the run-discovery cache in the background so the server boots instantly
    # AND the first browser request hits a warm cache (no cold-glob stall).
    threading.Thread(target=lambda: discover_runs(root), daemon=True).start()

    def _pick_run() -> Path | None:
        return pick_live_run(root, request.args.get("run") or run_dir)

    @app.route("/api/state")
    def api_state():  # noqa: ANN202
        try:
            total = int(request.args.get("total", total_epochs))
            run_path = _pick_run()
            state = {
                "frontier": load_frontier(root),
                "targets": {"floor": TARGET_FLOOR, "goal": TARGET_GOAL},
                "now_unix": time.time(),
                "run": load_run(run_path, total) if run_path else None,
                "available_runs": [str(p.parent.relative_to(root)) for p in discover_runs(root)[:12]],
            }
            body, status = serialize_state(state)
        except Exception as e:  # noqa: BLE001 — surface the failure LOUDLY, never a blank/silent 500
            body = json.dumps({"error": f"{type(e).__name__}: {e}",
                               "frontier": {"cpu": None, "cuda": None},
                               "run": None, "available_runs": []})
            status = 200
        return Response(body, status=status, mimetype="application/json")

    @app.route("/")
    def index():  # noqa: ANN202
        return render_template_string(_PAGE)

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--total-epochs", type=int, default=DEFAULT_TOTAL_EPOCHS,
                    help="planned total epochs for the progress bar / ETA")
    ap.add_argument("--run-dir", default=None,
                    help="substring of a specific run dir to pin (default: newest live run)")
    args = ap.parse_args(argv)
    app = create_app(total_epochs=args.total_epochs, run_dir=args.run_dir)
    print(f"[frontier-dashboard] http://{args.host}:{args.port}/  "
          f"(total_epochs={args.total_epochs})", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


# --------------------------------------------------------------------------- #
# The single-page UI (inline; polls /api/state)
# --------------------------------------------------------------------------- #
_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>comma-lab · frontier &amp; run telemetry</title>
<style>
:root{--bg:#0b0e14;--panel:#141a24;--panel2:#1b2330;--ink:#e6edf3;--muted:#8b98a9;
--good:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff;--line:#263041;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-x:hidden}
.wrap{max-width:1160px;margin:0 auto;padding:18px}
h1{font-size:15px;margin:0 0 2px;letter-spacing:.04em}
.sub{color:var(--muted);font-size:12px}
.row{display:flex;gap:14px;flex-wrap:wrap}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-wrap:anywhere;word-break:break-word;min-width:0;overflow:hidden}
.hdr .card{flex:1;min-width:300px}
.score{font-size:30px;font-weight:700;letter-spacing:.02em}
.tag{font-size:11px;color:var(--muted)}
.kv{display:flex;justify-content:space-between;gap:10px;font-size:12px;color:var(--muted);margin-top:3px;min-width:0}
.kv b{color:var(--ink);font-weight:600;text-align:right;min-width:0;overflow-wrap:anywhere;word-break:break-word}
.kv span{flex:0 0 auto}
.kv.stack{flex-direction:column;align-items:flex-start;gap:1px}
.kv.stack b{word-break:break-all;text-align:left;line-height:1.3;width:100%}
.metric .v{overflow-wrap:anywhere}
#run-name{overflow-wrap:anywhere;word-break:break-all}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;border:1px solid var(--line)}
.pill.cpu{color:var(--accent);border-color:#234}
.pill.cuda{color:var(--good);border-color:#243}
.section-t{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin:22px 0 8px}
.big{display:flex;gap:18px;flex-wrap:wrap;align-items:stretch}
.metric{flex:1;min-width:150px}
.metric .v{font-size:26px;font-weight:700}
.metric .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.progress{height:20px;background:var(--panel2);border:1px solid var(--line);border-radius:6px;overflow:hidden;position:relative}
.bar{height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff);width:0%;transition:width .6s ease}
.progress .lbl{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:12px;color:#fff;text-shadow:0 1px 2px #000a}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel)}
td.best{color:var(--good)}
.tbl-wrap{max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:10px}
.live{color:var(--good)}.stale{color:var(--warn)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--bad);margin-right:6px;vertical-align:middle}
.dot.on{background:var(--good);box-shadow:0 0 8px var(--good)}
svg{display:block}
.foot{color:var(--muted);font-size:11px;margin-top:18px}
.gx{color:var(--good)}.bx{color:var(--bad)}.wx{color:var(--warn)}
</style></head>
<body><div class="wrap">
  <h1>comma-lab · contest frontier &amp; live run telemetry</h1>
  <div class="sub" id="subline">connecting…</div>

  <div class="section-t">Frontier — source of truth</div>
  <div class="row hdr">
    <div class="card"><div class="tag"><span class="pill cpu">contest-CPU</span> &nbsp;public-leaderboard axis</div>
      <div class="score" id="cpu-score">—</div>
      <div class="kv stack"><span>technique</span><b id="cpu-tech">—</b></div>
      <div class="kv"><span>archive bytes</span><b id="cpu-bytes">—</b></div>
      <div class="kv"><span>measured</span><b id="cpu-when">—</b></div>
      <div class="kv"><span>sha</span><b id="cpu-sha">—</b></div>
    </div>
    <div class="card"><div class="tag"><span class="pill cuda">contest-CUDA</span> &nbsp;T4 axis</div>
      <div class="score" id="cuda-score">—</div>
      <div class="kv stack"><span>technique</span><b id="cuda-tech">—</b></div>
      <div class="kv"><span>archive bytes</span><b id="cuda-bytes">—</b></div>
      <div class="kv"><span>measured</span><b id="cuda-when">—</b></div>
      <div class="kv"><span>sha</span><b id="cuda-sha">—</b></div>
    </div>
    <div class="card" style="min-width:200px;flex:0 0 220px">
      <div class="tag">target ladder</div>
      <div class="kv"><span>T₁ floor</span><b>&lt; 0.19</b></div>
      <div class="kv"><span>T₃ goal</span><b class="gx">&lt; 0.15</b></div>
      <div class="kv" style="margin-top:8px"><span>frontier vs goal</span><b id="gap-goal">—</b></div>
      <div class="tag" id="frontier-refreshed" style="margin-top:8px"></div>
    </div>
  </div>

  <div class="section-t">Live run <span id="live-dot" class="dot"></span><span id="live-word" class="sub"></span></div>
  <div class="card">
    <div class="kv" style="font-size:13px"><span><b id="run-name">—</b></span>
       <span id="run-started" class="sub"></span></div>
    <div class="progress" style="margin:10px 0 4px"><div class="bar" id="bar"></div>
      <div class="lbl" id="bar-lbl"></div></div>
    <div class="kv"><span id="run-epochs" class="sub"></span><span id="run-eta" class="sub"></span></div>
  </div>

  <div class="section-t">Realtime — live training (per-epoch) + authority eval (torch-CPU, periodic)</div>
  <div class="big">
    <div class="card metric"><div class="l">epoch (live)</div><div class="v" id="m-epoch">—</div>
       <div class="tag" id="m-stage"></div></div>
    <div class="card metric"><div class="l">loss (live)</div><div class="v" id="m-loss">—</div>
       <div class="tag" id="m-posemse"></div></div>
    <div class="card metric"><div class="l">d_seg (eval)</div><div class="v" id="m-dseg">—</div>
       <div class="tag" id="m-dseg-best"></div></div>
    <div class="card metric"><div class="l">d_pose (eval)</div><div class="v" id="m-dpose">—</div>
       <div class="tag" id="m-eval-ep"></div></div>
    <div class="card metric"><div class="l" id="m-score-l">score S</div><div class="v" id="m-score">—</div>
       <div class="tag" id="m-score-note"></div></div>
    <div class="card metric" style="flex:2;min-width:260px"><div class="l" id="spark-l">trajectory</div>
       <svg id="spark" width="100%" height="64" viewBox="0 0 600 64" preserveAspectRatio="none"></svg>
       <div class="tag" id="spark-range"></div></div>
  </div>

  <div class="section-t">Run summary / meta</div>
  <div class="row">
    <div class="card metric"><div class="l">eval epochs</div><div class="v" id="s-neval">—</div></div>
    <div class="card metric"><div class="l">d_seg descent</div><div class="v gx" id="s-descent">—</div>
       <div class="tag">init → latest</div></div>
    <div class="card metric"><div class="l">recent Δd_seg/epoch</div><div class="v" id="s-rate">—</div></div>
    <div class="card metric"><div class="l">sec/epoch (incl. eval)</div><div class="v" id="s-spe">—</div>
       <div class="tag" id="s-spe-med"></div></div>
    <div class="card metric"><div class="l">elapsed</div><div class="v" id="s-elapsed">—</div></div>
  </div>

  <div class="section-t">Historical — recent eval epochs</div>
  <div class="tbl-wrap"><table id="tbl"><thead><tr>
    <th>epoch</th><th>stage</th><th>d_seg</th><th>d_pose</th><th>score</th><th>best d_seg</th><th>lr</th><th>elapsed</th>
  </tr></thead><tbody id="tbody"></tbody></table></div>

  <div class="foot" id="foot"></div>
</div>
<script>
const $=id=>document.getElementById(id);
let G=null;  // live-clock state, refreshed each poll; the 1s ticker interpolates from it
const fmt=(x,d=6)=>x==null?'—':(+x).toFixed(d);
const sci=x=>x==null?'—':(Math.abs(x)<1e-3&&x!==0?(+x).toExponential(2):(+x).toFixed(6));
const bytes=x=>x==null?'—':(+x).toLocaleString()+' B';
function dur(s){if(s==null)return'—';s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),ss=s%60;
  return h?`${h}h ${m}m`:(m?`${m}m ${ss}s`:`${ss}s`);}
function ago(unix,now){if(unix==null)return'—';const d=now-unix;if(d<90)return Math.round(d)+'s ago';
  return dur(d).replace(/(\d+)s$/,'')+' ago';}
function whenUTC(s){if(!s)return'—';try{return new Date(s).toISOString().slice(0,16).replace('T',' ')+'Z';}catch(e){return s;}}
function sparkline(vals){const svg=$('spark');svg.innerHTML='';if(!vals||vals.length<2){return;}
  const W=600,H=64,pad=4;const mn=Math.min(...vals),mx=Math.max(...vals);const rng=(mx-mn)||1;
  const pts=vals.map((v,i)=>{const x=pad+(W-2*pad)*i/(vals.length-1);const y=pad+(H-2*pad)*(1-(v-mn)/rng);return [x,y];});
  const path=pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
  const el=n=>document.createElementNS('http://www.w3.org/2000/svg',n);
  const pl=el('path');pl.setAttribute('d',path);pl.setAttribute('fill','none');pl.setAttribute('stroke','#58a6ff');pl.setAttribute('stroke-width','1.5');svg.appendChild(pl);
  const c=el('circle');c.setAttribute('cx',pts[pts.length-1][0]);c.setAttribute('cy',pts[pts.length-1][1]);c.setAttribute('r','2.6');c.setAttribute('fill','#3fb950');svg.appendChild(c);
  $('spark-range').textContent=`min ${sci(mn)} · max ${sci(mx)} · n=${vals.length}`;}
async function tick(){
  let st;
  try{const _r=await fetch('/api/state'+location.search);st=await _r.json();}
  catch(e){$('subline').textContent='reconnecting… (dashboard server restarting; auto-retries every 2.5s)';return;}
  try{
  const now=st.now_unix;
  // frontier
  const fr=st.frontier||{};
  for(const ax of ['cpu','cuda']){const f=fr[ax];if(!f)continue;
    $(ax+'-score').textContent=f.score==null?'—':(+f.score).toFixed(8);
    $(ax+'-tech').textContent=f.technique||'—';
    $(ax+'-bytes').textContent=bytes(f.archive_bytes);
    $(ax+'-when').textContent=whenUTC(f.measured_at_utc);
    $(ax+'-sha').textContent=f.archive_sha256||'—';}
  if(fr.cpu&&fr.cpu.score!=null){const g=(fr.cpu.score-st.targets.goal);
    $('gap-goal').innerHTML=g>0?`<span class="wx">+${g.toFixed(5)} to go</span>`:`<span class="gx">GOAL MET</span>`;}
  $('frontier-refreshed').textContent='pointer refreshed '+ago(fr.last_refreshed_utc?Date.parse(fr.last_refreshed_utc)/1000:null,now);
  // run
  const r=st.run;
  if(!r){$('subline').textContent='no training run found under experiments/results/';G=null;return;}
  $('subline').textContent='watching '+r.run_dir+'  ·  '+(st.available_runs?.length||0)+' runs on disk';
  $('run-name').textContent=r.run_dir+'  ('+r.trajectory_file+')';
  $('run-started').textContent='started '+ago(r.started_unix,now)+'  ·  '+whenUTC(new Date(r.started_unix*1000).toISOString());
  const dot=$('live-dot'),word=$('live-word');
  if(r.is_done){dot.classList.remove('on');word.textContent='done';word.className='sub';}
  else if(r.is_live){dot.classList.add('on');word.textContent='LIVE · last row '+ago(r.mtime,now);word.className='live';}
  else{dot.classList.remove('on');word.textContent='running · last row '+ago(r.mtime,now)+' (slow eval?)';word.className='stale';}
  // store live-clock state for the 1s ticker (elapsed/ETA tick locally between polls)
  G={started:r.started_unix,serverNow:st.now_unix,pollLocal:Date.now()/1000,
     eta:r.eta_s,etaMedian:r.eta_median_s,avg:r.summary.avg_s_per_epoch,total:r.total_epochs,
     epoch:r.current_epoch,progress:r.progress,running:!r.is_done};
  $('run-epochs').textContent='epoch '+r.current_epoch+' / '+r.total_epochs;
  // train (live per-epoch) + latest_eval (periodic authority)
  const T=r.train||{},E=r.latest_eval||{},B=r.best||{};
  $('m-epoch').textContent=T.epoch==null?'—':T.epoch;
  $('m-stage').textContent=T.stage||'';
  $('m-loss').textContent=T.loss==null?'—':(+T.loss).toFixed(4);
  $('m-posemse').textContent=T.pose_mse==null?'':'pose_mse '+(+T.pose_mse).toFixed(5)+(T.grad_norm!=null?' · |g| '+(+T.grad_norm).toFixed(1):'');
  $('m-dseg').textContent=sci(E.d_seg);
  $('m-dseg-best').innerHTML=B.d_seg!=null?`best <span class="gx">${sci(B.d_seg)}</span> @ ep ${B.epoch}`:'awaiting first eval';
  $('m-dpose').textContent=sci(E.d_pose);
  $('m-eval-ep').textContent=E.epoch!=null?'@ ep '+E.epoch:'';
  $('m-score').textContent=E.score==null?'—':(+E.score).toFixed(5);
  $('m-score-l').textContent=E.score_is_full?'score S (full)':'score S (distortion-only)';
  $('m-score-note').textContent=E.score_is_full?'100·d_seg+√(10·d_pose)+25·B/N':'pre-byte-close · rate term not yet added';
  $('spark-l').textContent=(r.spark_kind==='loss'?'loss':'d_seg')+' trajectory';
  sparkline(r.spark);
  // summary
  const s=r.summary;
  $('s-neval').textContent=s.n_eval_epochs+(s.n_records?' / '+s.n_records+' rows':'');
  $('s-descent').textContent=s.d_seg_descent==null?'—':s.d_seg_descent.toFixed(4);
  $('s-rate').textContent=s.recent_dseg_per_epoch==null?'—':s.recent_dseg_per_epoch.toExponential(2);
  $('s-spe').textContent=s.avg_s_per_epoch==null?'—':s.avg_s_per_epoch.toFixed(1);
  $('s-spe-med').textContent=s.median_s_per_epoch!=null?'train ~'+(+s.median_s_per_epoch).toFixed(0)+'s/ep (median, ex-eval)':'';
  liveClock();  // immediate elapsed/ETA paint; the 1s ticker continues it
  // table
  const tb=$('tbody');tb.innerHTML='';
  for(const row of [...r.rows].reverse()){const tr=document.createElement('tr');
    const isBest=B&&row.epoch===B.epoch;
    tr.innerHTML=`<td>${row.epoch}</td><td>${row.stage||''}</td>`+
      `<td class="${isBest?'best':''}">${sci(row.d_seg)}</td><td>${sci(row.d_pose)}</td>`+
      `<td>${row.score==null?'—':(+row.score).toFixed(5)}</td><td>${sci(row.best_d_seg)}</td>`+
      `<td>${row.lr==null?'—':(+row.lr).toFixed(4)}</td><td>${dur(row.elapsed_s)}</td>`;
    tb.appendChild(tr);}
  $('foot').textContent='advisory display only · authority d_seg/d_pose are torch-CPU · frontier is pointer-only · poll ~2.5s · clock 1s';
  }catch(e){$('subline').textContent='render hiccup ('+e+') — retrying';}
}
// 1s local ticker: elapsed + ETA + progress advance smoothly between polls (operator: realtime, regardless of poll rate)
function liveClock(){
  if(!G)return;
  const dLocal=Math.max(0,Date.now()/1000 - G.pollLocal);
  const baseElapsed=(G.serverNow!=null&&G.started!=null)?(G.serverNow-G.started):0;
  const elapsed=baseElapsed+(G.running?dLocal:0);
  $('s-elapsed').textContent=dur(elapsed);
  if(G.eta!=null&&G.running){
    const eta=Math.max(0,G.eta-dLocal);
    $('run-eta').textContent='ETA '+dur(eta)+(G.etaMedian!=null&&G.etaMedian<G.eta?'  ·  ~'+dur(G.etaMedian)+' at train cadence (async eval)':'')+'  ·  '+(G.avg?dur(G.avg)+'/ep':'');
    // nudge the progress bar forward fractionally within the current epoch
    if(G.avg&&G.total){const frac=Math.min(1,dLocal/G.avg);
      const pct=Math.min(100,100*((G.epoch+frac)/G.total));
      $('bar').style.width=pct.toFixed(2)+'%';$('bar-lbl').textContent=pct.toFixed(1)+'%';}
  } else {
    $('run-eta').textContent=G.running?'ETA —':'finished';
    const pct=Math.min(100,100*(G.progress||0));$('bar').style.width=pct.toFixed(1)+'%';$('bar-lbl').textContent=pct.toFixed(1)+'%';
  }
}
tick();setInterval(tick,2500);setInterval(liveClock,1000);
</script>
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
