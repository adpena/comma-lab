#!/usr/bin/env python3
"""Pre-dispatch sanity gate — fail-closed ladder before any paid GPU dispatch.

Council Q3 prescription (`feedback_grand_council_predictor_calibration_no_arbitrariness_20260505.md`).
The apogee_int4 8x miss happened because we dispatched without a sanity ladder.
This tool runs fail-closed checks in <30s and returns exit 64 on refusal:

  Gate 1 (anchors_sufficient):  predictor has ≥3 calibration anchors for this lane class
  Gate 2 (sanity_lossy_vs_lossless): predicted lossy score > lossless baseline
  Gate 3 (distortion_model_gate):   high-rel_err candidates have proxy telemetry
          plus explicit non-proxy distortion/parity evidence for the exact bytes
  Gate 4 (hazard_scan):  tools/check_dispatch_cli_shell_hazards.py shows 0 dispatch_local_path_leak
  Gate 5 (lane_registry_consistent):  tools/lane_maturity.py validate passes
  Gate 6 (apogee_evidence_semantics): apogee_intN has explicit non-proxy
          distortion/parity evidence for the exact archive bytes

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
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_line, json_text, read_json, sha256_file  # noqa: E402

DEFAULT_ANCHORS_DIR = REPO_ROOT / ".omx" / "calibration"
OVERRIDE_LOG = REPO_ROOT / ".omx" / "state" / "predispatch_overrides.log"
CONTEST_ORIGINAL_BYTES = 37_545_489
APOGEE_ALLOWED_EVIDENCE_SEMANTICS = {
    "contest_faithful_distortion_model",
    "scorer_basin_parity_gate",
    "contest_cuda_exact_eval_positive",
}
APOGEE_FORBIDDEN_EVIDENCE_MARKERS = (
    "byte_only",
    "byte-only",
    "prediction_only",
    "predicted_band",
    "invalid_predicted_band",
    "proxy_only",
    "distortion_proxy_local",
    "local_distortion_proxy",
    "[distortion-proxy:local]",
)
APOGEE_PASS_STATUSES = {"pass", "passed", "ready", "exact_positive_cuda"}

from tac.predictor.score_band import (  # noqa: E402
    CalibrationAnchor,
    load_calibration_anchors,
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
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    archive_path: Path,
    evidence_json_path: Path | None,
) -> GateResult:
    """Gate 2: predicted band must respect the rate-distortion lower bound.

    A lossy candidate can beat a lossless baseline if it spends fewer charged
    bytes and has exact-byte non-proxy evidence that scorer-visible distortion
    stays inside the trusted basin. It cannot beat the baseline by more than
    the official rate-term delta without claiming distortion improvement.
    """
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
    lossless_bytes = lossless[0].archive_bytes
    if predicted_high < lossless_score:
        candidate_bytes = archive_path.stat().st_size
        official_rate_delta = 25.0 * (candidate_bytes - lossless_bytes) / CONTEST_ORIGINAL_BYTES
        component_penalty = _readiness_evidence_component_score_penalty(evidence_json_path)
        component_penalty_value = component_penalty if component_penalty is not None else 0.0
        rate_distortion_floor = lossless_score + official_rate_delta + component_penalty_value
        if candidate_bytes < lossless_bytes and predicted_high >= rate_distortion_floor - 1e-12:
            evidence_gate = _validate_non_proxy_readiness_evidence(
                archive_path=archive_path,
                evidence_json_path=evidence_json_path,
                gate_name="sanity_lossy_vs_lossless",
            )
            if evidence_gate.passed:
                return GateResult(
                    name="sanity_lossy_vs_lossless",
                    passed=True,
                    detail=(
                        f"predicted_high={predicted_high:.4f} < lossless baseline "
                        f"{lossless_score:.4f}, but candidate is {lossless_bytes - candidate_bytes} "
                        "charged bytes smaller; official rate-distortion floor is "
                        f"{rate_distortion_floor:.4f} "
                        f"(rate_delta={official_rate_delta:.6f}, "
                        f"component_penalty={component_penalty_value:.6f}), "
                        f"and {evidence_gate.detail}"
                    ),
                )
            return GateResult(
                name="sanity_lossy_vs_lossless",
                passed=False,
                detail=(
                    f"lossy predicted_high={predicted_high:.4f} is below lossless baseline "
                    f"{lossless_score:.4f} only within the official rate-distortion floor "
                    f"{rate_distortion_floor:.4f}, but exact-byte non-proxy readiness "
                    f"evidence is missing or invalid: {evidence_gate.detail}"
                ),
            )
        if candidate_bytes < lossless_bytes and predicted_high < rate_distortion_floor:
            return GateResult(
                name="sanity_lossy_vs_lossless",
                passed=False,
                detail=(
                    f"predicted_high={predicted_high:.4f} is below the SHA-tied "
                    f"rate-distortion floor {rate_distortion_floor:.4f} "
                    f"(lossless={lossless_score:.4f}, rate_delta={official_rate_delta:.6f}, "
                    f"component_penalty={component_penalty_value:.6f}). "
                    "Parity/readiness evidence is not score-lowering evidence."
                ),
            )
        return GateResult(
            name="sanity_lossy_vs_lossless",
            passed=False,
            detail=(
                f"predicted_high={predicted_high:.4f} < lossless baseline {lossless_score:.4f}. "
                "A lossy candidate can beat lossless only by the official rate-term delta "
                "and only with exact-byte non-proxy evidence that distortion stays in basin."
            ),
        )
    return GateResult(
        name="sanity_lossy_vs_lossless",
        passed=True,
        detail=f"predicted_high {predicted_high:.4f} ≥ lossless {lossless_score:.4f}",
    )


def _readiness_evidence_component_score_penalty(evidence_json_path: Path | None) -> float | None:
    """Return nonnegative score penalty implied by readiness component deltas.

    The evidence may prove the candidate stays in the same scorer basin without
    proving it lowers score. If it records component deltas, charge positive
    deltas against the official score formula before letting byte savings justify
    a below-lossless predicted score.
    """
    if evidence_json_path is None:
        return None
    try:
        payload = read_json(evidence_json_path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    report = payload.get("parity_report")
    if not isinstance(report, dict):
        report = {}
    seg_delta = _finite_float(payload.get("seg_dist_delta"))
    if seg_delta is None:
        seg_delta = _finite_float(report.get("seg_dist_delta"))
    pose_lossless = _finite_float(report.get("pose_dist_lossless"))
    pose_quantized = _finite_float(report.get("pose_dist_quantized"))
    pose_delta = _finite_float(payload.get("pose_dist_delta"))
    if pose_delta is None:
        pose_delta = _finite_float(report.get("pose_dist_delta"))

    seg_penalty = 100.0 * max(float(seg_delta or 0.0), 0.0)
    pose_penalty = 0.0
    if pose_lossless is not None and pose_quantized is not None:
        pose_penalty = max(
            (10.0 * max(pose_quantized, 0.0)) ** 0.5
            - (10.0 * max(pose_lossless, 0.0)) ** 0.5,
            0.0,
        )
    elif pose_delta is not None:
        # A pose-distance delta is not itself an official score delta because
        # score uses sqrt(10 * pose_dist). Without both absolute endpoints the
        # baseline is unknown, so fail closed instead of pretending baseline=0.
        return None
    return seg_penalty + pose_penalty


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def _gate_distortion_proxy(
    *,
    rel_err_pct: float,
    distortion_proxy_was_run: bool,
    archive_path: Path,
    evidence_json_path: Path | None,
) -> GateResult:
    """Gate 3: high-rel_err candidates require non-proxy distortion evidence.

    The local distortion proxy is useful telemetry but is not a dispatch gate.
    """
    HIGH_REL_ERR_THRESHOLD = 1.0
    if rel_err_pct <= HIGH_REL_ERR_THRESHOLD:
        return GateResult(
            name="distortion_model_gate",
            passed=True,
            detail=f"rel_err {rel_err_pct:.2f}% ≤ {HIGH_REL_ERR_THRESHOLD}% threshold; proxy not required",
        )
    if not distortion_proxy_was_run:
        return GateResult(
            name="distortion_model_gate",
            passed=False,
            detail=(
                f"rel_err {rel_err_pct:.2f}% > {HIGH_REL_ERR_THRESHOLD}% but --distortion-proxy-ran "
                "not set. Run experiments/distortion_proxy_local.py against the archive first; "
                "then attach non-proxy distortion/parity evidence for the exact candidate bytes."
            ),
        )
    evidence_gate = _validate_non_proxy_readiness_evidence(
        archive_path=archive_path,
        evidence_json_path=evidence_json_path,
        gate_name="distortion_model_gate",
    )
    if not evidence_gate.passed:
        evidence_gate.detail = (
            "local distortion proxy was run but is non-promotable by itself; "
            + evidence_gate.detail
        )
        return evidence_gate
    return GateResult(
        name="distortion_model_gate",
        passed=True,
        detail=(
            f"rel_err {rel_err_pct:.2f}% > threshold; proxy telemetry plus "
            f"{evidence_gate.detail}"
        ),
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


def _gate_apogee_evidence_semantics(
    *,
    lane_class: str,
    archive_path: Path,
    evidence_json_path: Path | None,
) -> GateResult:
    """Gate 6: Apogee intN needs explicit non-proxy readiness evidence."""
    if lane_class != "apogee_intN":
        return GateResult(
            name="apogee_evidence_semantics",
            passed=True,
            detail=f"not an apogee_intN lane class ({lane_class}); gate not applicable",
        )
    if evidence_json_path is None:
        return GateResult(
            name="apogee_evidence_semantics",
            passed=False,
            detail=(
                "apogee_intN cannot dispatch from byte-only, proxy-only, or predicted-band evidence. "
                "Provide --readiness-evidence-json with evidence_semantics in "
                f"{sorted(APOGEE_ALLOWED_EVIDENCE_SEMANTICS)} for the exact archive SHA."
            ),
        )
    generic_gate = _validate_non_proxy_readiness_evidence(
        archive_path=archive_path,
        evidence_json_path=evidence_json_path,
        gate_name="apogee_evidence_semantics",
    )
    if not generic_gate.passed:
        return generic_gate
    try:
        payload = read_json(evidence_json_path)
    except (OSError, ValueError) as exc:
        return GateResult(
            name="apogee_evidence_semantics",
            passed=False,
            detail=f"cannot read readiness evidence JSON {evidence_json_path}: {exc}",
        )
    if not isinstance(payload, dict):
        return GateResult(
            name="apogee_evidence_semantics",
            passed=False,
            detail=f"readiness evidence must be a JSON object: {evidence_json_path}",
        )

    blockers: list[str] = []
    actual_sha = sha256_file(archive_path)
    recorded_sha = (
        payload.get("candidate_archive_sha256")
        or payload.get("archive_sha256")
        or payload.get("archive", {}).get("sha256")
    )
    if recorded_sha != actual_sha:
        blockers.append(f"candidate archive SHA mismatch evidence={recorded_sha!r} actual={actual_sha}")

    semantics = str(payload.get("evidence_semantics", "")).strip().lower()
    joined_payload = json_text(payload).lower()
    if not semantics:
        blockers.append("missing evidence_semantics")
    if semantics not in APOGEE_ALLOWED_EVIDENCE_SEMANTICS:
        blockers.append(f"unsupported evidence_semantics={semantics!r}")
    for marker in APOGEE_FORBIDDEN_EVIDENCE_MARKERS:
        if marker in semantics or marker in joined_payload:
            blockers.append(f"forbidden proxy/prediction evidence marker {marker!r}")
            break

    if payload.get("ready_for_exact_eval_dispatch") is not True:
        blockers.append("ready_for_exact_eval_dispatch is not true")
    evidence_grade = str(payload.get("evidence_grade", "")).lower()
    if "negative" in evidence_grade or evidence_grade in {"invalid", "external", "prediction"}:
        blockers.append(f"non-promotable evidence_grade={payload.get('evidence_grade')!r}")

    distortion_status = str(payload.get("distortion_model_status", "")).lower()
    parity_status = str(payload.get("scorer_basin_parity_status", "")).lower()
    exact_positive = payload.get("exact_positive_cuda_evidence") is True
    if semantics != "contest_cuda_exact_eval_positive":
        blockers.append(
            "apogee_intN requires contest_cuda_exact_eval_positive for normal predispatch pass; "
            f"{semantics or 'missing'} is calibration-only and requires an explicit override"
        )
    if semantics == "contest_faithful_distortion_model" and distortion_status not in APOGEE_PASS_STATUSES:
        blockers.append("contest_faithful_distortion_model requires passing distortion_model_status")
    if semantics == "scorer_basin_parity_gate" and parity_status not in APOGEE_PASS_STATUSES:
        blockers.append("scorer_basin_parity_gate requires passing scorer_basin_parity_status")
    if semantics == "contest_cuda_exact_eval_positive" and not exact_positive:
        blockers.append("contest_cuda_exact_eval_positive requires exact_positive_cuda_evidence=true")

    if blockers:
        return GateResult(
            name="apogee_evidence_semantics",
            passed=False,
            detail="; ".join(blockers),
        )
    return GateResult(
        name="apogee_evidence_semantics",
        passed=True,
        detail=f"{semantics} evidence matches exact archive SHA",
    )


def _validate_non_proxy_readiness_evidence(
    *,
    archive_path: Path,
    evidence_json_path: Path | None,
    gate_name: str,
) -> GateResult:
    """Validate minimal exact-byte non-proxy readiness evidence."""

    if evidence_json_path is None:
        return GateResult(
            name=gate_name,
            passed=False,
            detail=(
                "missing --readiness-evidence-json with non-proxy distortion/parity "
                "evidence for the exact archive SHA"
            ),
        )
    try:
        payload = read_json(evidence_json_path)
    except (OSError, ValueError) as exc:
        return GateResult(
            name=gate_name,
            passed=False,
            detail=f"cannot read readiness evidence JSON {evidence_json_path}: {exc}",
        )
    if not isinstance(payload, dict):
        return GateResult(
            name=gate_name,
            passed=False,
            detail=f"readiness evidence must be a JSON object: {evidence_json_path}",
        )

    blockers: list[str] = []
    actual_sha = sha256_file(archive_path)
    recorded_sha = (
        payload.get("candidate_archive_sha256")
        or payload.get("archive_sha256")
        or payload.get("archive", {}).get("sha256")
    )
    if recorded_sha != actual_sha:
        blockers.append(f"candidate archive SHA mismatch evidence={recorded_sha!r} actual={actual_sha}")
    semantics = str(payload.get("evidence_semantics", "")).strip().lower()
    joined_payload = json_text(payload).lower()
    if not semantics:
        blockers.append("missing evidence_semantics")
    if semantics not in APOGEE_ALLOWED_EVIDENCE_SEMANTICS:
        blockers.append(f"unsupported evidence_semantics={semantics!r}")
    for marker in APOGEE_FORBIDDEN_EVIDENCE_MARKERS:
        if marker in semantics or marker in joined_payload:
            blockers.append(f"forbidden proxy/prediction evidence marker {marker!r}")
            break
    if payload.get("ready_for_exact_eval_dispatch") is not True:
        blockers.append("ready_for_exact_eval_dispatch is not true")
    evidence_grade = str(payload.get("evidence_grade", "")).lower()
    if "negative" in evidence_grade or evidence_grade in {"invalid", "external", "prediction"}:
        blockers.append(f"non-promotable evidence_grade={payload.get('evidence_grade')!r}")
    if blockers:
        return GateResult(name=gate_name, passed=False, detail="; ".join(blockers))
    return GateResult(name=gate_name, passed=True, detail=f"{semantics} evidence matches exact archive SHA")


def predispatch_sanity(
    archive_path: Path,
    predicted_low: float,
    predicted_high: float,
    rel_err_pct: float,
    lane_class: str,
    distortion_proxy_was_run: bool = False,
    anchors_dir: Path = DEFAULT_ANCHORS_DIR,
    readiness_evidence_json: Path | None = None,
) -> SanityResult:
    """Run all fail-closed gates and return aggregate result."""
    if not archive_path.is_file():
        return SanityResult(
            passed=False,
            refusal_reasons=[f"archive not found: {archive_path}"],
        )

    anchors_path = anchors_dir / f"anchors_{lane_class}.json"
    anchors = load_calibration_anchors(anchors_path)

    gates = [
        _gate_anchors_sufficient(lane_class, anchors_dir),
        _gate_sanity_lossy_vs_lossless(
            predicted_low,
            predicted_high,
            rel_err_pct,
            anchors,
            archive_path,
            readiness_evidence_json,
        ),
        _gate_distortion_proxy(
            rel_err_pct=rel_err_pct,
            distortion_proxy_was_run=distortion_proxy_was_run,
            archive_path=archive_path,
            evidence_json_path=readiness_evidence_json,
        ),
        _gate_hazard_scan(),
        _gate_lane_registry(),
        _gate_apogee_evidence_semantics(
            lane_class=lane_class,
            archive_path=archive_path,
            evidence_json_path=readiness_evidence_json,
        ),
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
        f.write(json_line(record))


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
                        help="set if you ran experiments/distortion_proxy_local.py first; proxy output is not readiness")
    parser.add_argument("--readiness-evidence-json", type=Path, default=None,
                        help="required for apogee_intN: explicit non-proxy distortion/parity evidence JSON")
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
        readiness_evidence_json=args.readiness_evidence_json,
    )

    if args.json:
        print(
            json_text(
                {
                    "passed": result.passed,
                    "gates": [
                        {
                            "name": g.name,
                            "passed": g.passed,
                            "detail": g.detail,
                            "confidence": g.confidence,
                        }
                        for g in result.gates
                    ],
                    "refusal_reasons": result.refusal_reasons,
                }
            ),
            end="",
        )
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
