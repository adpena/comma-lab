#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only milestone diagnostics for the live #205 v7.5.2 witness run.

This tool consumes only artifacts the trainer already emitted.  It never imports MLX,
never loads a scorer, never renders a frame, never runs training/evaluation, and never
writes beneath ``--run-dir``.  All numbers are advisory trajectory telemetry; they do
not move the contest pointer.

The EP450 reversal claim is pre-registered here before the post-transition data exists:

* pre window: verdict epochs exactly {375, 400, 425};
* post window: verdict epochs exactly {475, 500, 525}, giving the ep450 chroma/screw
  repairs at least 25 epochs of exposure and including the ep500 lane-band engagement;
* gated classes: Road (0) and Undrivable (2);
* gated metrics: class ``within_flip`` (the verdict's ``d_seg_by_class``) and the
  emitted threshold-annulus ``per_class_annulus_flip_frac``;
* each class/metric reverses only if BOTH (i) post-window mean <= 0.95 * pre-window
  mean (at least a 5% relative reduction) AND (ii) post-window OLS slope <= -1e-5
  fraction/epoch;
* PASS_REVERSAL requires all four class/metric gates.  Once all six exact verdict
  epochs exist, any missed gate is FAIL_IMPLEMENTATION_FALSIFIED.  Missing data is
  PRE_REGISTERED_PENDING.  The FAIL scope is INSTANCE/FORMULATION: this live run's
  deferred-repair curriculum implementation, never the level-set family/paradigm.

Pose-finish readiness is also pre-registered.  The last eight ``jacobian_basin`` rows
strictly before ep726 are READY only when robust median sigma_min >= 0.100, robust
p10 sigma_min >= 0.025, median condition number <= 125000, at least 6/8 median
sigma_min values are >= 0.080, and no degenerate guard fires.  Degenerate guards are:
non-finite/non-positive values, robust median sigma_min < 0.080, robust p10 < 0.015,
median condition number > 150000, or last-four median sigma_min < 75% of the prior-four
median.  The post-switch outcome (ep725-ish to first verdict >=825) succeeds only if
d_pose falls >=10% while d_seg rises no more than 0.001 absolute; it remains pending
until those rows exist.

The byte-close section is READINESS ONLY: checkpoint schema/finite-data/EMA custody,
canonical tool presence+syntax, and D18 k90 telemetry.  It never builds an archive or
invokes ``upstream/evaluate.py``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO / "experiments/results/levelset_v752_baseline_20260710T185913Z"
POINTER = "0.19108282 [contest-CPU] UNMOVED"
AXIS = "[macOS-CPU/numpy advisory] NON-PROMOTABLE"
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}

EP450_PRE = (375, 400, 425)
EP450_POST = (475, 500, 525)
REVERSAL_CLASSES = (0, 2)
REVERSAL_RELATIVE_MEAN = 0.95
REVERSAL_MAX_SLOPE = -1.0e-5

POSE_BOUNDARY = 726
POSE_WINDOW = 8
POSE_READY_MEDIAN_SIGMA = 0.100
POSE_READY_P10_SIGMA = 0.025
POSE_READY_MAX_COND = 125_000.0
POSE_READY_MIN_GOOD = 6
POSE_GOOD_SIGMA = 0.080
POSE_DEGENERATE_MEDIAN_SIGMA = 0.080
POSE_DEGENERATE_P10_SIGMA = 0.015
POSE_DEGENERATE_MAX_COND = 150_000.0
POSE_DEGENERATE_COLLAPSE_RATIO = 0.75
POSE_POST_EPOCH = 825
POSE_POST_MIN_DPOSE_REDUCTION = 0.10
POSE_POST_MAX_DSEG_RISE = 0.001

REQUIRED_PARAMS = ("code", "in_proj.weight", "out_sdf.weight", "out_tex.weight", "palette")
REQUIRED_CFG = ("__cfg_activation", "__bank_n_scales", "__render_hw", "__epoch")
TRAJECTORY_STAGES = frozenset({
    "verdict", "annulus_convergence", "handoff_readiness", "loss_terms",
    "jacobian_basin", "pose_finish_conditioning_gate", "mod_dim_dynamics",
    "mod_dim_ablation", "checkpoint", "birth_completion", "birth_completion_ramp",
    "stage_transition", "tau_octave_advance", "pose_finish_engaged", "muon_start",
})


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def refuse_run_write(path: Path | None, run_dir: Path, field: str) -> None:
    if path is not None and is_within(path, run_dir):
        raise ValueError(f"{field}={path} is inside sacred read-only run dir {run_dir}")


@dataclass(frozen=True)
class ParsedLog:
    rows: tuple[dict[str, Any], ...]
    malformed_lines: int
    total_lines: int


def load_json_rows(path: Path) -> ParsedLog:
    rows: list[dict[str, Any]] = []
    malformed = 0
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            total += 1
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return ParsedLog(tuple(rows), malformed, total)


def rows_of(rows: Iterable[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("stage") == stage]


def by_epoch(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        ep = row.get("epoch", row.get("ep"))
        if isinstance(ep, (int, float)):
            out[int(ep)] = row
    return out


def ols_slope(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.nan
    xs = np.asarray([p[0] for p in points], dtype=np.float32)
    ys = np.asarray([p[1] for p in points], dtype=np.float32)
    dx = xs - xs.mean(dtype=np.float32)
    denom = np.sum(dx * dx, dtype=np.float32)
    if float(denom) <= 0.0:
        return math.nan
    return float(np.sum(dx * (ys - ys.mean(dtype=np.float32)), dtype=np.float32) / denom)


def _class_metric(
    verdict: dict[str, Any], annulus: dict[str, Any] | None, class_id: int, metric: str,
) -> float | None:
    if metric == "within_flip":
        values = verdict.get("d_seg_by_class")
        return float(values[class_id]) if isinstance(values, list) and len(values) > class_id else None
    if metric == "flip_mass_share":
        values = verdict.get("flip_share_by_class")
        return float(values[class_id]) if isinstance(values, list) and len(values) > class_id else None
    if metric == "annulus_flip_frac" and annulus:
        values = (annulus.get("threshold") or {}).get("per_class_annulus_flip_frac") or {}
        value = values.get(str(class_id), values.get(class_id))
        return float(value) if isinstance(value, (int, float)) else None
    raise ValueError(f"unknown metric {metric!r}")


def ep450_reversal(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    verdicts = by_epoch(rows_of(rows, "verdict"))
    annulus = by_epoch(rows_of(rows, "annulus_convergence"))
    required = EP450_PRE + EP450_POST
    missing = [ep for ep in required if ep not in verdicts or ep not in annulus]
    result: dict[str, Any] = {
        "status": "PRE_REGISTERED_PENDING" if missing else "UNSET",
        "transition_epoch": 450,
        "pre_epochs": list(EP450_PRE),
        "post_epochs": list(EP450_POST),
        "missing_epochs": missing,
        "thresholds": {
            "post_mean_le_pre_mean_times": REVERSAL_RELATIVE_MEAN,
            "post_ols_slope_le_fraction_per_epoch": REVERSAL_MAX_SLOPE,
            "gated_classes": [CLASS_NAMES[c] for c in REVERSAL_CLASSES],
            "gated_metrics": ["within_flip", "annulus_flip_frac"],
            "all_gates_required": True,
        },
        "verdict_scope_on_fail": (
            "INSTANCE/FORMULATION: #205 v7.5.2 deferred-repair curriculum implementation; "
            "NOT family/paradigm"
        ),
        "metrics": {},
    }
    for c in REVERSAL_CLASSES:
        cname = CLASS_NAMES[c]
        result["metrics"][cname] = {}
        for metric in ("within_flip", "annulus_flip_frac", "flip_mass_share"):
            pre = [(ep, _class_metric(verdicts[ep], annulus.get(ep), c, metric))
                   for ep in EP450_PRE if ep in verdicts and ep in annulus]
            post = [(ep, _class_metric(verdicts[ep], annulus.get(ep), c, metric))
                    for ep in EP450_POST if ep in verdicts and ep in annulus]
            pre = [(ep, float(v)) for ep, v in pre if v is not None]
            post = [(ep, float(v)) for ep, v in post if v is not None]
            item: dict[str, Any] = {"pre": pre, "post": post, "gating": metric != "flip_mass_share"}
            if len(pre) == 3 and len(post) == 3:
                pre_mean = float(np.mean(np.asarray([v for _, v in pre], np.float32), dtype=np.float32))
                post_mean = float(np.mean(np.asarray([v for _, v in post], np.float32), dtype=np.float32))
                post_slope = ols_slope(post)
                item.update({
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "post_over_pre": post_mean / pre_mean if pre_mean > 0 else math.inf,
                    "post_slope_per_epoch": post_slope,
                    "mean_gate_pass": post_mean <= REVERSAL_RELATIVE_MEAN * pre_mean,
                    "slope_gate_pass": post_slope <= REVERSAL_MAX_SLOPE,
                })
                item["pass"] = bool(item["mean_gate_pass"] and item["slope_gate_pass"])
            result["metrics"][cname][metric] = item
    if not missing:
        gates = [result["metrics"][CLASS_NAMES[c]][m].get("pass", False)
                 for c in REVERSAL_CLASSES for m in ("within_flip", "annulus_flip_frac")]
        result["status"] = "PASS_REVERSAL" if all(gates) else "FAIL_IMPLEMENTATION_FALSIFIED"
    return result


def _nearest_before(verdicts: dict[int, dict[str, Any]], boundary: int) -> dict[str, Any] | None:
    eps = [ep for ep in verdicts if ep < boundary]
    return verdicts[max(eps)] if eps else None


def _first_after(verdicts: dict[int, dict[str, Any]], boundary: int, exposure: int = 25) -> dict[str, Any] | None:
    eps = [ep for ep in verdicts if ep >= boundary + exposure]
    return verdicts[min(eps)] if eps else None


def _tau_boundaries(rows: Iterable[dict[str, Any]]) -> list[tuple[int, str]]:
    points = sorted(
        ((int(r["epoch"]), float(r["tau"])) for r in rows_of(rows, "mod_dim_dynamics")
         if isinstance(r.get("epoch"), (int, float)) and isinstance(r.get("tau"), (int, float))),
        key=lambda x: x[0],
    )
    out: list[tuple[int, str]] = []
    prev: float | None = None
    for ep, tau in points:
        if prev is not None and not math.isclose(tau, prev, rel_tol=0.0, abs_tol=1e-12):
            out.append((ep, f"tau_octave {prev:g}->{tau:g}"))
        prev = tau
    return out


def stage_attribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    verdicts = by_epoch(rows_of(rows, "verdict"))
    annulus = by_epoch(rows_of(rows, "annulus_convergence"))
    boundaries = [*_tau_boundaries(rows),
        (450, "chroma_boundary + temporal_screw repair"),
        (500, "lane_render_band joins repair stack"),
        (726, "Muon optimizer + terminal pose-finish"),
    ]
    table: list[dict[str, Any]] = []
    for boundary, label in sorted(boundaries):
        before = _nearest_before(verdicts, boundary)
        after = _first_after(verdicts, boundary)
        item: dict[str, Any] = {"boundary_epoch": boundary, "lever_or_stage": label}
        if not before or not after:
            item.update({"status": "PENDING", "before_epoch": before and before.get("epoch"),
                         "after_epoch": after and after.get("epoch")})
            table.append(item)
            continue
        be, ae = int(before["epoch"]), int(after["epoch"])
        item.update({"status": "AVAILABLE", "before_epoch": be, "after_epoch": ae,
                     "delta_d_seg": float(after["d_seg"]) - float(before["d_seg"]),
                     "per_class_delta_within_flip": {}, "per_class_delta_annulus_flip_frac": {}})
        for c, name in CLASS_NAMES.items():
            bv = _class_metric(before, annulus.get(be), c, "within_flip")
            av = _class_metric(after, annulus.get(ae), c, "within_flip")
            ba = _class_metric(before, annulus.get(be), c, "annulus_flip_frac")
            aa = _class_metric(after, annulus.get(ae), c, "annulus_flip_frac")
            item["per_class_delta_within_flip"][name] = None if bv is None or av is None else av - bv
            item["per_class_delta_annulus_flip_frac"][name] = None if ba is None or aa is None else aa - ba
        table.append(item)
    return {
        "authority": AXIS,
        "attribution_scope": "before/after telemetry association; causal isolation requires a matched A/B",
        "existing_deep_tool": "tools/witness_per_stage_annulus_attribution.py",
        "boundaries": table,
    }


def pose_finish_readiness(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    basin = sorted(
        (r for r in rows_of(rows, "jacobian_basin")
         if isinstance(r.get("epoch"), (int, float)) and int(r["epoch"]) < POSE_BOUNDARY),
        key=lambda r: int(r["epoch"]),
    )[-POSE_WINDOW:]
    result: dict[str, Any] = {
        "status": "PRE_REGISTERED_PENDING",
        "boundary_epoch": POSE_BOUNDARY,
        "sample_epochs": [int(r["epoch"]) for r in basin],
        "success_criteria": {
            "window_rows": POSE_WINDOW,
            "median_sigma_min_ge": POSE_READY_MEDIAN_SIGMA,
            "median_p10_sigma_min_ge": POSE_READY_P10_SIGMA,
            "median_condition_number_le": POSE_READY_MAX_COND,
            "count_sigma_min_ge_0p080_ge": POSE_READY_MIN_GOOD,
        },
        "degenerate_guards": {
            "nonfinite_or_nonpositive": True,
            "median_sigma_min_lt": POSE_DEGENERATE_MEDIAN_SIGMA,
            "median_p10_sigma_min_lt": POSE_DEGENERATE_P10_SIGMA,
            "median_condition_number_gt": POSE_DEGENERATE_MAX_COND,
            "last4_over_prior4_lt": POSE_DEGENERATE_COLLAPSE_RATIO,
        },
        "post_switch_criteria": {
            "first_verdict_epoch_ge": POSE_POST_EPOCH,
            "d_pose_relative_reduction_ge": POSE_POST_MIN_DPOSE_REDUCTION,
            "d_seg_absolute_rise_le": POSE_POST_MAX_DSEG_RISE,
        },
    }
    # Require a full window that reaches the final expected pre-boundary cadence neighborhood.
    if len(basin) == POSE_WINDOW and int(basin[-1]["epoch"]) >= 722:
        sig = [float(r["median_sigma_min"]) for r in basin]
        p10 = [float(r["p10_sigma_min"]) for r in basin]
        cond = [float(r["median_cond"]) for r in basin]
        finite_positive = all(math.isfinite(x) and x > 0 for x in sig + p10 + cond)
        med_sig = statistics.median(sig)
        med_p10 = statistics.median(p10)
        med_cond = statistics.median(cond)
        collapse_ratio = statistics.median(sig[4:]) / statistics.median(sig[:4])
        degenerate = (
            not finite_positive or med_sig < POSE_DEGENERATE_MEDIAN_SIGMA
            or med_p10 < POSE_DEGENERATE_P10_SIGMA or med_cond > POSE_DEGENERATE_MAX_COND
            or collapse_ratio < POSE_DEGENERATE_COLLAPSE_RATIO
        )
        ready = (
            not degenerate and med_sig >= POSE_READY_MEDIAN_SIGMA
            and med_p10 >= POSE_READY_P10_SIGMA and med_cond <= POSE_READY_MAX_COND
            and sum(x >= POSE_GOOD_SIGMA for x in sig) >= POSE_READY_MIN_GOOD
        )
        result.update({
            "median_sigma_min": med_sig, "median_p10_sigma_min": med_p10,
            "median_condition_number": med_cond, "last4_over_prior4": collapse_ratio,
            "count_sigma_min_ge_0p080": sum(x >= POSE_GOOD_SIGMA for x in sig),
            "status": "READY" if ready else ("DEGENERATE_NOT_READY" if degenerate else "NOT_READY"),
        })
    verdicts = by_epoch(rows_of(rows, "verdict"))
    pre_eps = [ep for ep in verdicts if ep < POSE_BOUNDARY]
    post_eps = [ep for ep in verdicts if ep >= POSE_POST_EPOCH]
    if pre_eps and post_eps:
        pre, post = verdicts[max(pre_eps)], verdicts[min(post_eps)]
        dpose_reduction = 1.0 - float(post["d_pose"]) / float(pre["d_pose"])
        dseg_rise = float(post["d_seg"]) - float(pre["d_seg"])
        result["post_switch"] = {
            "status": "SUCCESS" if dpose_reduction >= POSE_POST_MIN_DPOSE_REDUCTION
            and dseg_rise <= POSE_POST_MAX_DSEG_RISE else "DEGRADE_OR_NO_GAIN",
            "pre_epoch": int(pre["epoch"]), "post_epoch": int(post["epoch"]),
            "d_pose_relative_reduction": dpose_reduction, "d_seg_absolute_rise": dseg_rise,
        }
    else:
        result["post_switch"] = {"status": "PRE_REGISTERED_PENDING"}
    return result


def _syntax_ok(path: Path) -> tuple[bool, str | None]:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return True, None
    except (OSError, SyntaxError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def inspect_checkpoint(path: Path, expected_epoch: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result["ready"] = False
        return result
    result.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    try:
        with np.load(path, allow_pickle=False) as z:
            missing_params = [k for k in REQUIRED_PARAMS if k not in z.files]
            missing_cfg = [k for k in REQUIRED_CFG if k not in z.files]
            finite = True
            for key in z.files:
                if key.startswith("__") or z[key].dtype.kind not in "fc":
                    continue
                if not bool(np.isfinite(np.asarray(z[key], dtype=np.float32)).all()):
                    finite = False
                    break
            code_shape = list(z["code"].shape) if "code" in z.files else None
            epoch = int(z["__epoch"]) if "__epoch" in z.files else None
            result.update({
                "epoch": epoch, "code_shape": code_shape,
                "n_pairs": code_shape[0] // 2 if code_shape and len(code_shape) == 2 else None,
                "mod_dim": code_shape[1] if code_shape and len(code_shape) == 2 else None,
                "missing_required_params": missing_params, "missing_required_cfg": missing_cfg,
                "all_learned_arrays_finite": finite,
                "expected_epoch_matches": expected_epoch is None or epoch == expected_epoch,
                "ema_custody": "EMA" in path.name or "ema" in path.name,
            })
            result["ready"] = bool(
                not missing_params and not missing_cfg and finite and result["n_pairs"] == 600
                and result["expected_epoch_matches"] and result["ema_custody"]
            )
    except Exception as exc:  # fail closed: malformed npz is not ready
        result.update({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def byte_close_readiness(run_dir: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    best_meta_path = run_dir / "levelset_best.json"
    best_meta = json.loads(best_meta_path.read_text()) if best_meta_path.is_file() else {}
    best_name = best_meta.get("path", "levelset_witness_ema_BEST.npz")
    checkpoints = [inspect_checkpoint(run_dir / best_name, best_meta.get("epoch"))]
    latest = run_dir / "levelset_witness_ema_mlx.npz"
    if latest != run_dir / best_name:
        checkpoints.append(inspect_checkpoint(latest))
    byte_tool = REPO / "tools/levelset_byte_close_and_eval.py"
    d18_tool = REPO / "tools/witness_code_pca_byteclose.py"
    byte_syntax, byte_error = _syntax_ok(byte_tool)
    d18_syntax, d18_error = _syntax_ok(d18_tool)
    d18_rows = sorted(
        (r for r in rows_of(rows, "mod_dim_dynamics") if isinstance(r.get("epoch"), (int, float))),
        key=lambda r: int(r["epoch"]),
    )
    d18: dict[str, Any] = {"status": "PENDING"}
    if d18_rows:
        r = d18_rows[-1]
        full = r.get("code_bytes_full")
        estimate = r.get("k90_truncate_bytes_estimate")
        k90 = (r.get("spectrum") or {}).get("k90")
        mod_dim = r.get("mod_dim")
        ready = all(isinstance(x, (int, float)) for x in (full, estimate, k90, mod_dim))
        if ready:
            ready = int(k90) < int(mod_dim) and int(estimate) < int(full)
        d18 = {
            "status": "READY_FOR_MILESTONE_A_B" if ready else "NOT_READY",
            "epoch": int(r["epoch"]), "k90": k90, "mod_dim": mod_dim,
            "code_bytes_full": full, "k90_truncate_bytes_estimate": estimate,
            "estimated_bytes_saved": int(full) - int(estimate) if ready else None,
            "estimate_only": True,
        }
    return {
        "status": "READY" if all(c.get("ready") for c in checkpoints)
        and byte_syntax and d18_syntax and d18["status"] == "READY_FOR_MILESTONE_A_B" else "NOT_READY",
        "readiness_only": True,
        "exact_eval_launched": False,
        "archive_built": False,
        "best_metadata": best_meta,
        "checkpoints": checkpoints,
        "canonical_byte_close_tool": {"path": str(byte_tool), "syntax_ok": byte_syntax, "error": byte_error},
        "d18_tool": {"path": str(d18_tool), "syntax_ok": d18_syntax, "error": d18_error},
        "d18": d18,
    }


def live_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    verdicts = sorted(rows_of(rows, "verdict"), key=lambda r: int(r.get("epoch", -1)))
    losses = sorted(rows_of(rows, "loss_terms"), key=lambda r: int(r.get("ep", -1)))
    best = min(verdicts, key=lambda r: float(r["d_seg"])) if verdicts else None
    latest = verdicts[-1] if verdicts else None
    latest_loss = losses[-1] if losses else None
    return {
        "latest_training_epoch": latest_loss and int(latest_loss["ep"]),
        "latest_verdict": latest,
        "best_verdict": best,
        "verdict_count": len(verdicts),
        "curriculum_stage": latest and latest.get("seg_form", latest.get("phase")),
    }


def build_report(run_dir: Path, parsed: ParsedLog) -> dict[str, Any]:
    daemon = run_dir / "daemon.log"
    shadow = run_dir / "costate_shadow.jsonl"
    return {
        "schema": "pact.n205_full_run_diagnostics.v1",
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "read_only": True,
        "authority": AXIS,
        "score_claim": False,
        "pointer": POINTER,
        "source": {
            "daemon_log_sha256": sha256_file(daemon),
            "daemon_log_bytes": daemon.stat().st_size,
            "costate_shadow_sha256": sha256_file(shadow) if shadow.is_file() else None,
            "json_rows": len(parsed.rows), "malformed_json_lines": parsed.malformed_lines,
            "total_log_lines": parsed.total_lines,
        },
        "live": live_summary(parsed.rows),
        "ep450_reversal": ep450_reversal(parsed.rows),
        "stage_attribution": stage_attribution(parsed.rows),
        "pose_finish_ep726": pose_finish_readiness(parsed.rows),
        "byte_close_readiness": byte_close_readiness(run_dir, parsed.rows),
        "continual_learning": {
            "ledger": ".omx/research/costate_organ_trajectory_ledger.md",
            "selected_stages": sorted(TRAJECTORY_STAGES),
            "re_run_at": ["ep450+post-window", "ep726+post-window", "terminal ep3000"],
        },
    }


def write_trajectory(path: Path, report: dict[str, Any], rows: Iterable[dict[str, Any]]) -> int:
    # Preserve every parseable trainer row, not merely the current costate feature subset. A future
    # organ may learn from a setup/provenance/memory signal that today's model does not consume;
    # dropping it now would under-harvest an irreproducible ten-day trajectory.
    selected = list(rows)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "record_type": "manifest", "schema": "pact.n205_trajectory.v1",
            "generated_at": report["generated_at"], "run_dir": report["run_dir"],
            "source": report["source"], "authority": AXIS, "score_claim": False,
            "pointer": POINTER, "n_rows": len(selected),
            "current_costate_primary_stages": sorted(TRAJECTORY_STAGES),
        }, sort_keys=True) + "\n")
        for row in selected:
            fh.write(json.dumps({"record_type": "telemetry", **row}, sort_keys=True) + "\n")
    return len(selected)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--trajectory-out", type=Path, default=None)
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory missing: {run_dir}")
    refuse_run_write(args.out_json, run_dir, "--out-json")
    refuse_run_write(args.trajectory_out, run_dir, "--trajectory-out")
    parsed = load_json_rows(run_dir / "daemon.log")
    report = build_report(run_dir, parsed)
    if args.trajectory_out:
        args.trajectory_out.parent.mkdir(parents=True, exist_ok=True)
        report["continual_learning"]["trajectory_path"] = str(args.trajectory_out)
        report["continual_learning"]["trajectory_rows"] = write_trajectory(
            args.trajectory_out, report, parsed.rows)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
