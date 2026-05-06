#!/usr/bin/env python3
"""Predicted-vs-actual reconciliation for apogee_intN dispatches.

Joins:
  predicted: experiments/results/apogee_int*_repack_*/repack_metadata.json
             (predicted_score_band emitted by the producer at compress time)
  actual:    experiments/results/lane_apogee_int*_*/eval/contest_auth_eval.json
             (final_score from CUDA dispatch — when the operator has launched
             the lane via scripts/remote_lane_apogee_intN.sh)

Reports per-bits:
  - historical predicted band [low, high] (forensic byte-only, noncanonical)
  - actual final_score (from contest_auth_eval.json, if present)
  - whether the historical prediction was falsified
  - readiness blockers proving the lane is not exact-eval dispatch-ready
  - device + samples (advisory tag check)

Pairs with:
  tools/apogee_intN_pareto.py  — pre-dispatch decision matrix
  tools/score_dashboard.py      — sorted view of every contest_auth_eval

Usage:
  .venv/bin/python tools/predicted_vs_actual_reconciler.py
  .venv/bin/python tools/predicted_vs_actual_reconciler.py --json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json  # noqa: E402

PR106_BASELINE_SCORE = 0.20945673
APOGEE_RECONCILER_DISPATCH_BLOCKERS = [
    "historical_prediction_is_forensic_byte_only",
    "missing_contest_faithful_distortion_model",
    "missing_scorer_basin_parity_gate",
    "reconciler_table_is_not_a_dispatch_gate",
]


@dataclass
class ReconciledRow:
    bits: int
    predicted_low: float
    predicted_high: float
    archive_size_bytes: int
    distortion_risk: str
    rate_score_delta: float
    prediction_status: str | None
    ready_for_exact_eval_dispatch: bool
    dispatch_blockers: list[str]
    actual_score: float | None
    actual_path: str | None
    in_band: bool | None
    beats_pr106: bool | None
    device: str | None
    samples: int | None


def _parse_band(band_str: str) -> tuple[float, float]:
    inner = band_str.strip().lstrip("[").rstrip("]")
    lo_str, hi_str = inner.split(",")
    return float(lo_str.strip()), float(hi_str.strip())


def _find_actual_for_bits(repo_root: Path, bits: int) -> tuple[float | None, str | None, str | None, int | None]:
    """Find the latest contest_auth_eval.json for a given apogee_intN bits config.

    Looks under experiments/results/ for any directory matching
    `lane_apogee_int{bits}_*` containing an `eval/contest_auth_eval.json`
    OR a top-level `contest_auth_eval.json`. Returns the score from the
    most-recently-modified JSON found.
    """
    candidates: list[Path] = []
    base = repo_root / "experiments" / "results"
    if not base.is_dir():
        return None, None, None, None
    for lane_dir in base.glob(f"lane_apogee_int{bits}*"):
        for sub in lane_dir.rglob("contest_auth_eval*.json"):
            candidates.append(sub)
    if not candidates:
        return None, None, None, None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        data = read_json(latest)
    except (ValueError, OSError):
        return None, str(latest.relative_to(repo_root)), None, None
    score = data.get("final_score") or data.get("score") or data.get("total_score")
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
    device = data.get("device", "?")
    samples = data.get("samples") or data.get("n_samples")
    return score, str(latest.relative_to(repo_root)), str(device), samples


def reconcile(repo_root: Path) -> list[ReconciledRow]:
    rows: list[ReconciledRow] = []
    base = repo_root / "experiments" / "results"
    if not base.is_dir():
        return rows
    for manifest in base.glob("apogee_int*_repack_*/repack_metadata.json"):
        try:
            data = read_json(manifest)
        except (ValueError, OSError):
            continue
        try:
            lo, hi = _parse_band(data["predicted_score_band"])
        except (KeyError, ValueError):
            continue
        bits = int(data["bits"])
        actual, actual_path, device, samples = _find_actual_for_bits(repo_root, bits)
        in_band = (lo <= actual <= hi) if actual is not None else None
        beats_pr106 = (actual < PR106_BASELINE_SCORE) if actual is not None else None
        source_blockers = list(data.get("dispatch_blockers") or [])
        dispatch_blockers = list(dict.fromkeys(source_blockers + APOGEE_RECONCILER_DISPATCH_BLOCKERS))
        if data.get("ready_for_exact_eval_dispatch") is True:
            dispatch_blockers.append("source_manifest_ready_claim_ignored_by_reconciler")
        rows.append(ReconciledRow(
            bits=bits,
            predicted_low=lo,
            predicted_high=hi,
            archive_size_bytes=int(data["archive_size_bytes"]),
            distortion_risk=str(data["distortion_risk"]),
            rate_score_delta=float(data["rate_component_score_delta_vs_pr106"]),
            prediction_status=data.get("prediction_status"),
            ready_for_exact_eval_dispatch=False,
            dispatch_blockers=dispatch_blockers,
            actual_score=actual,
            actual_path=actual_path,
            in_band=in_band,
            beats_pr106=beats_pr106,
            device=device,
            samples=samples,
        ))
    rows.sort(key=lambda r: r.bits)
    return rows


def _format_table(rows: list[ReconciledRow]) -> str:
    if not rows:
        return "(no apogee_intN repack manifests found — run experiments/repack_pr106_with_intN_block_fp.py first)"
    out: list[str] = []
    out.append(f"PR106 baseline: 0.{int(PR106_BASELINE_SCORE * 1e8):08d}, 186,239 bytes")
    out.append("Apogee intN predictions are forensic byte-only and noncanonical until a distortion gate exists.")
    out.append("")
    header = (
        f"{'bits':>4}  {'historical band':<18}  {'actual':>8}  {'status':<11}  "
        f"{'dispatch':<8}  {'risk':<16}  {'device':<6}  evidence"
    )
    out.append(header)
    out.append("-" * len(header))
    n_landed = 0
    n_falsified = 0
    for r in rows:
        actual_str = f"{r.actual_score:.5f}" if r.actual_score is not None else "(pending)"
        if r.in_band is True:
            status = "in-band"
        elif r.in_band is False:
            status = "FALSIFIED"
            n_falsified += 1
        else:
            status = "pending"
        dispatch_status = "READY" if r.ready_for_exact_eval_dispatch else "BLOCKED"
        if r.actual_score is not None:
            n_landed += 1
        # Mark non-CUDA device with asterisk per CLAUDE.md
        device_str = r.device if r.device else "—"
        if device_str not in ("cuda", "—", "?", None) and not device_str.endswith("*"):
            device_str = device_str + "*"
        evidence = r.actual_path or "(no contest_auth_eval.json yet)"
        # Truncate evidence path
        if len(evidence) > 60:
            evidence = "…" + evidence[-59:]
        out.append(
            f"{r.bits:>4}  [{r.predicted_low:.3f}, {r.predicted_high:.3f}]   "
            f"{actual_str:>8}  {status:<11}  {dispatch_status:<8}  "
            f"{r.distortion_risk:<16}  {(device_str or '—'):<6}  {evidence}"
        )
    out.append("")
    out.append(f"Summary: {n_landed}/{len(rows)} bits configs have actual scores; "
              f"{n_falsified}/{n_landed} falsified their historical byte-only band; "
              "0 rows are dispatch-ready without a distortion gate")
    if n_landed == 0:
        out.append("")
        out.append("No actual scores yet. Do not dispatch from this table; build a distortion gate first.")
    out.append("(* = non-CUDA device; treat as advisory only per CLAUDE.md MPS-auth-eval-is-NOISE)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output instead of human table.")
    args = parser.parse_args(argv)
    rows = reconcile(REPO_ROOT)
    if args.json:
        out = {
            "pr106_baseline_score": PR106_BASELINE_SCORE,
            "n_configs": len(rows),
            "n_landed": sum(1 for r in rows if r.actual_score is not None),
            "n_in_band": sum(1 for r in rows if r.in_band is True),
            "n_beats_pr106": sum(1 for r in rows if r.beats_pr106 is True),
            "rows": [asdict(r) for r in rows],
        }
        print(json_text(out), end="")
    else:
        print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
