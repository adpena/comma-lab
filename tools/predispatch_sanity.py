#!/usr/bin/env python3
"""Pre-dispatch sanity gate — 5-check ladder before any paid GPU dispatch.

Council Q3 prescription (`feedback_grand_council_predictor_calibration_no_arbitrariness_20260505.md`).
The apogee_int4 8x miss happened because we dispatched without a sanity ladder.
This tool runs 5 checks in <30s and returns exit 64 on refusal:

  Gate 1 (anchors_sufficient):  predictor has ≥3 calibration anchors for this lane class
  Gate 2 (sanity_lossy_vs_lossless): predicted lossy score > lossless baseline
  Gate 3 (distortion_proxy_local):  for high-rel_err candidates, distortion proxy was run
  Gate 4 (hazard_scan):  tools/check_dispatch_cli_shell_hazards.py shows 0 dispatch_local_path_leak
  Gate 5 (lane_registry_consistent):  tools/lane_maturity.py validate passes

Operator override: `--override-reason <≥40-char-reason>` bypasses but logs to
`.omx/state/predispatch_overrides.log` JSONL for forensic audit.

Usage:
    .venv/bin/python tools/predispatch_sanity.py \\
        --archive experiments/results/<lane>/archive.zip \\
        --predicted-low 0.155 --predicted-high 0.180 \\
        --rel-err-pct 7.09 \\
        --lane-class apogee_intN

Exit codes:
    0   all gates pass
    64  one or more gates failed (no override)
    65  override accepted with logged reason
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANCHORS_DIR = REPO_ROOT / ".omx" / "calibration"
OVERRIDE_LOG = REPO_ROOT / ".omx" / "state" / "predispatch_overrides.log"

# Add src/ to import path so we can use tac.predictor
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.predictor.score_band import (  # noqa: E402
    CalibrationAnchor,
    load_calibration_anchors,
    predict_score_band,
)


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    confidence: str = "high"  # "high" | "medium" | "low"


@dataclass
class SanityResult:
    passed: bool
    gates: list[GateResult] = field(default_factory=list)
    refusal_reasons: list[str] = field(default_factory=list)


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gate_anchors_sufficient(lane_class: str, anchors_dir: Path) -> GateResult:
    """Gate 1: ≥3 calibration anchors for this lane class."""
    anchors_path = anchors_dir / f"anchors_{lane_class}.json"
    anchors = load_calibration_anchors(anchors_path)
    if len(anchors) < 3:
        return GateResult(
            name="anchors_sufficient",
            passed=False,
            detail=(
                f"only {len(anchors)} calibration anchors at {anchors_path} (need ≥3). "
                f"Add more empirical anchors before banding {lane_class} candidates."
            ),
        )
    return GateResult(
        name="anchors_sufficient",
        passed=True,
        detail=f"{len(anchors)} anchors loaded from {anchors_path.name}",
    )


def _gate_sanity_lossy_vs_lossless(
    predicted_low: float,
    predicted_high: float,
    rel_err_pct: float,
    anchors: list[CalibrationAnchor],
) -> GateResult:
    """Gate 2: predicted band must respect lossy-cannot-beat-lossless invariant."""
    if rel_err_pct == 0.0:
        return GateResult(
            name="sanity_lossy_vs_lossless",
            passed=True,
            detail="lossless candidate (rel_err=0); skip sanity gate",
        )
    lossless = [a for a in anchors if a.rel_err_pct_per_weight == 0.0]
    if not lossless:
        return GateResult(
            name="sanity_lossy_vs_lossless",
            passed=True,
            detail="no lossless anchor in calibration set; skip sanity gate (informational)",
            confidence="low",
        )
    lossless_score = lossless[0].contest_cuda_score
    if predicted_high < lossless_score:
        return GateResult(
            name="sanity_lossy_vs_lossless",
            passed=False,
            detail=(
                f"predicted_high={predicted_high:.4f} < lossless baseline {lossless_score:.4f}. "
                "A lossy compression cannot strictly improve every component; this band is incoherent."
            ),
        )
    return GateResult(
        name="sanity_lossy_vs_lossless",
        passed=True,
        detail=f"predicted_high {predicted_high:.4f} ≥ lossless {lossless_score:.4f}",
    )


def _gate_distortion_proxy(
    rel_err_pct: float,
    distortion_proxy_was_run: bool,
) -> GateResult:
    """Gate 3: high-rel_err candidates require local distortion proxy to be run.

    Per council Q1 (Hotz): >1% rel_err needs an empirical distortion estimate.
    """
    HIGH_REL_ERR_THRESHOLD = 1.0
    if rel_err_pct <= HIGH_REL_ERR_THRESHOLD:
        return GateResult(
            name="distortion_proxy_local",
            passed=True,
            detail=f"rel_err {rel_err_pct:.2f}% ≤ {HIGH_REL_ERR_THRESHOLD}% threshold; proxy not required",
        )
    if not distortion_proxy_was_run:
        return GateResult(
            name="distortion_proxy_local",
            passed=False,
            detail=(
                f"rel_err {rel_err_pct:.2f}% > {HIGH_REL_ERR_THRESHOLD}% but --distortion-proxy-ran "
                "not set. Run experiments/distortion_proxy_local.py against the archive first; "
                "the cost is ~30s CPU/MPS forward pass — far cheaper than $0.30 GPU mistake."
            ),
        )
    return GateResult(
        name="distortion_proxy_local",
        passed=True,
        detail=f"rel_err {rel_err_pct:.2f}% > threshold; distortion proxy was run (per --distortion-proxy-ran)",
    )


def _gate_hazard_scan() -> GateResult:
    """Gate 4: dispatch_local_path_leak must be 0 across the repo."""
    scanner = REPO_ROOT / "tools" / "check_dispatch_cli_shell_hazards.py"
    if not scanner.is_file():
        return GateResult(
            name="hazard_scan",
            passed=False,
            detail=f"scanner not found at {scanner}",
        )
    result = subprocess.run(
        [sys.executable, str(scanner), "--strict"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
        check=False,
    )
    # Strict-mode exit nonzero on any hazard; we filter for the dispatch_local_path_leak class.
    leak_lines = [
        line for line in (result.stdout + result.stderr).splitlines()
        if "dispatch_local_path_leak" in line or "remote_script_local_pythonpath_leak" in line
    ]
    if leak_lines:
        return GateResult(
            name="hazard_scan",
            passed=False,
            detail=f"{len(leak_lines)} dispatch-path-leak hazards found:\n  " + "\n  ".join(leak_lines[:5]),
        )
    return GateResult(
        name="hazard_scan",
        passed=True,
        detail="0 dispatch_local_path_leak / remote_script_local_pythonpath_leak hazards",
    )


def _gate_lane_registry() -> GateResult:
    """Gate 5: lane_maturity validate passes (registry is internally consistent)."""
    cli = REPO_ROOT / "tools" / "lane_maturity.py"
    if not cli.is_file():
        return GateResult(
            name="lane_registry_consistent",
            passed=False,
            detail=f"lane_maturity tool not found at {cli}",
        )
    result = subprocess.run(
        [sys.executable, str(cli), "validate"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return GateResult(
            name="lane_registry_consistent",
            passed=False,
            detail=f"lane_maturity validate failed (rc={result.returncode}): {result.stderr[:200]}",
        )
    return GateResult(
        name="lane_registry_consistent",
        passed=True,
        detail="lane_maturity validate clean",
    )


def predispatch_sanity(
    archive_path: Path,
    predicted_low: float,
    predicted_high: float,
    rel_err_pct: float,
    lane_class: str,
    distortion_proxy_was_run: bool = False,
    anchors_dir: Path = DEFAULT_ANCHORS_DIR,
) -> SanityResult:
    """Run all 5 gates and return aggregate result."""
    if not archive_path.is_file():
        return SanityResult(
            passed=False,
            refusal_reasons=[f"archive not found: {archive_path}"],
        )

    anchors_path = anchors_dir / f"anchors_{lane_class}.json"
    anchors = load_calibration_anchors(anchors_path)

    gates = [
        _gate_anchors_sufficient(lane_class, anchors_dir),
        _gate_sanity_lossy_vs_lossless(predicted_low, predicted_high, rel_err_pct, anchors),
        _gate_distortion_proxy(rel_err_pct, distortion_proxy_was_run),
        _gate_hazard_scan(),
        _gate_lane_registry(),
    ]
    passed = all(g.passed for g in gates)
    refusal_reasons = [f"{g.name}: {g.detail}" for g in gates if not g.passed]
    return SanityResult(passed=passed, gates=gates, refusal_reasons=refusal_reasons)


def _log_override(
    archive_path: Path,
    override_reason: str,
    refusal_reasons: list[str],
) -> None:
    OVERRIDE_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts_utc": _utc_now(),
        "archive_path": str(archive_path),
        "override_reason": override_reason,
        "refusal_reasons": refusal_reasons,
        "operator": os.environ.get("USER", "unknown"),
    }
    with OVERRIDE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="archive to sanity-check")
    parser.add_argument("--predicted-low", required=True, type=float)
    parser.add_argument("--predicted-high", required=True, type=float)
    parser.add_argument("--rel-err-pct", required=True, type=float,
                        help="rel_err_pct_per_weight from build metadata")
    parser.add_argument("--lane-class", required=True,
                        help="e.g. apogee_intN, pr106_sidecar — must match anchors_<class>.json filename")
    parser.add_argument("--distortion-proxy-ran", action="store_true",
                        help="set if you ran experiments/distortion_proxy_local.py first")
    parser.add_argument("--override-reason", default="",
                        help="bypass gate failures; reason ≥40 chars required and logged")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of human-readable output")
    args = parser.parse_args(argv)

    result = predispatch_sanity(
        archive_path=args.archive,
        predicted_low=args.predicted_low,
        predicted_high=args.predicted_high,
        rel_err_pct=args.rel_err_pct,
        lane_class=args.lane_class,
        distortion_proxy_was_run=args.distortion_proxy_ran,
    )

    if args.json:
        print(json.dumps({
            "passed": result.passed,
            "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail, "confidence": g.confidence}
                       for g in result.gates],
            "refusal_reasons": result.refusal_reasons,
        }, indent=2))
    else:
        for g in result.gates:
            mark = "PASS" if g.passed else "FAIL"
            print(f"  [{mark}] {g.name}: {g.detail}")
        if result.passed:
            print("[predispatch_sanity] ALL 5 GATES PASS")
        else:
            print(f"[predispatch_sanity] BLOCKED — {len(result.refusal_reasons)} gate(s) failed")

    if result.passed:
        return 0
    if args.override_reason:
        if len(args.override_reason) < 40:
            print(
                f"[predispatch_sanity] override-reason too short ({len(args.override_reason)} < 40 chars). "
                "Required: a substantive reason that future-you would accept reading.",
                file=sys.stderr,
            )
            return 64
        _log_override(args.archive, args.override_reason, result.refusal_reasons)
        print(
            f"[predispatch_sanity] OVERRIDE ACCEPTED: {args.override_reason} "
            f"(logged to {OVERRIDE_LOG.relative_to(REPO_ROOT)})",
            file=sys.stderr,
        )
        return 65
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
