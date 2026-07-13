#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Aggregate source-compatible task-455 matched-window regime receipts.

This is a research-evidence reducer, never an evaluator. It preserves each
regime's exact-derived schedule and combines only compatible timing buckets and
fail-closed formulation verdicts. The 1656 ms comparator remains explicitly
operator supplied; it is never inserted into the measured sample population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.repo_io import write_json_artifact  # noqa: E402

SCHEMA: Final[str] = "onpolicy_costate_matched_campaign.v1"
REQUIRED_REGIMES: Final[tuple[str, ...]] = ("early", "boundary", "late")
METRICS: Final[tuple[str, ...]] = ("ce", "d_seg", "d_pose")
# The campaign is a conjunction over requested regimes: any non-anchor measured
# prefix failure rejects the registered formulation. NEEDS-MORE wins only when
# no regime has produced a decisive NO-GO. Full-cadence completeness is reported
# separately and must never mask an observed trajectory failure.
VERDICT_PRECEDENCE: Final[tuple[str, ...]] = ("NO-GO", "NEEDS-MORE", "GO")
RECEIPT_WINDOW_BASIS: Final[str] = (
    "sums of symmetric complete per-step operational timers under one exact-derived common norm "
    "schedule; each includes render, provider, renderer VJP, and candidate update; line-search and "
    "exact validation calls excluded"
)


class CampaignError(RuntimeError):
    """Fail-closed receipt or aggregation error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError(f"{name} must be finite")
    return result


def _load_receipt(path: Path, regime: str) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError(f"missing {regime} receipt: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "onpolicy_costate_matched_window_probe.v1":
        raise CampaignError(f"{regime} receipt schema mismatch")
    if payload.get("status") != "MEASURED":
        raise CampaignError(f"{regime} receipt is not terminal MEASURED")
    if payload.get("config", {}).get("regime") != regime:
        raise CampaignError(f"{regime} receipt regime mismatch")
    if payload.get("score_claim") is not False or payload.get("research_only") is not True:
        raise CampaignError(f"{regime} receipt attempts to broaden authority")
    flags = payload.get("false_authority_flags", {})
    if any(flags.get(name) is not False for name in (
        "score_claim",
        "mps_authority",
        "surrogate_eval_authority",
        "contest_cpu_or_cuda_eval",
    )):
        raise CampaignError(f"{regime} receipt false-authority flags are incomplete")
    accounting = payload.get("teacher_accounting", {})
    if accounting.get("segnet_forward_reconciliation") != "PASS" or accounting.get(
        "posenet_forward_reconciliation"
    ) != "PASS":
        raise CampaignError(f"{regime} teacher-call accounting did not reconcile")
    return payload


def _source_identity(receipt: dict[str, Any]) -> dict[str, str]:
    custody = receipt["run_contract"]["payload"]["source_custody"]
    return {name: row["sha256"] for name, row in sorted(custody.items())}


def _common_config(receipt: dict[str, Any]) -> dict[str, Any]:
    config = dict(receipt["config"])
    config.pop("regime", None)
    return config


def _aggregate_bucket(receipts: list[dict[str, Any]], bucket: str) -> dict[str, Any]:
    count = 0
    total = 0.0
    for receipt in receipts:
        row = receipt["timing"]["measured"][bucket]
        count += int(row["count"])
        total += _finite(row["total_seconds"], name=f"{bucket}.total_seconds")
    return {
        "count": count,
        "total_seconds": total,
        "mean_seconds": total / count if count else None,
    }


def _raw_fidelity(receipt: dict[str, Any]) -> dict[str, Any]:
    exact = receipt["exact_trace"]
    target = receipt["surrogate_trace"]
    if len(exact) != len(target) or not exact:
        raise CampaignError("matched traces are incomplete")
    floor = receipt["deterministic_repeat_noise_floor"]
    comparisons: dict[str, Any] = {}
    for metric in METRICS:
        deltas = [
            abs(_finite(target_row[metric], name=metric) - _finite(exact_row[metric], name=metric))
            for exact_row, target_row in zip(exact, target, strict=True)
        ]
        tolerance = _finite(floor[metric], name=f"noise_floor.{metric}")
        comparisons[metric] = {
            "max_abs_delta": max(deltas),
            "deterministic_repeat_tolerance": tolerance,
            "within_floor_at_every_step": all(delta <= tolerance for delta in deltas),
            "first_failing_step": next(
                (index for index, delta in enumerate(deltas) if delta > tolerance), None
            ),
        }
    return comparisons


def _validated_window_seconds(receipt: dict[str, Any]) -> tuple[float, float]:
    economics = receipt["window_economics"]
    if economics.get("comparison_basis") != RECEIPT_WINDOW_BASIS:
        raise CampaignError("receipt window timing basis is not symmetric complete-step timing")
    measured = receipt["timing"]["measured"]
    observed_steps = int(economics["observed_window_steps"])
    totals: list[float] = []
    for branch, economics_key in (
        ("exact_window_operational_step", "exact_window_operational_seconds"),
        ("surrogate_window_operational_step", "surrogate_window_operational_seconds"),
    ):
        bucket = measured[branch]
        if int(bucket["count"]) != observed_steps:
            raise CampaignError(f"{branch} count does not match observed window steps")
        bucket_total = _finite(bucket["total_seconds"], name=f"{branch}.total_seconds")
        economics_total = _finite(economics[economics_key], name=economics_key)
        if not math.isclose(bucket_total, economics_total, rel_tol=0.0, abs_tol=1e-12):
            raise CampaignError(f"{branch} total does not reconcile to window economics")
        totals.append(bucket_total)
    return totals[0], totals[1]


def _derived_mission_verdict(
    receipt: dict[str, Any], comparisons: dict[str, Any], *, ema_admitted: bool
) -> str:
    economics = receipt["window_economics"]
    skip_fraction = _finite(
        economics["observed_exact_teacher_skip_fraction"], name="observed_teacher_skip_fraction"
    )
    any_trace_failure = any(not row["within_floor_at_every_step"] for row in comparisons.values())
    if skip_fraction > 0.0 and (any_trace_failure or not ema_admitted):
        return "NO-GO"
    if (
        bool(economics["target_cadence_fidelity_validated"])
        and skip_fraction > 0.0
        and ema_admitted
        and not any_trace_failure
    ):
        return "GO"
    return "NEEDS-MORE"


def aggregate(paths: dict[str, Path]) -> dict[str, Any]:
    if tuple(sorted(paths)) != tuple(sorted(REQUIRED_REGIMES)):
        raise CampaignError("exactly early, boundary, and late receipts are required")
    receipts_by_regime = {
        regime: _load_receipt(paths[regime], regime) for regime in REQUIRED_REGIMES
    }
    receipts = list(receipts_by_regime.values())
    source_identity = _source_identity(receipts[0])
    common_config = _common_config(receipts[0])
    for regime, receipt in receipts_by_regime.items():
        if _source_identity(receipt) != source_identity:
            raise CampaignError(f"{regime} receipt source bundle differs")
        if _common_config(receipt) != common_config:
            raise CampaignError(f"{regime} receipt treatment config differs")

    regime_rows: list[dict[str, Any]] = []
    mission_verdicts: list[str] = []
    exact_window_total = 0.0
    target_window_total = 0.0
    for regime, receipt in receipts_by_regime.items():
        declared_verdict = receipt.get("mission_verdict")
        if declared_verdict not in VERDICT_PRECEDENCE:
            raise CampaignError(f"{regime} mission verdict is invalid")
        comparisons = _raw_fidelity(receipt)
        ema_admitted = bool(receipt["collection_fit_rows"][-1]["admitted"])
        verdict = _derived_mission_verdict(
            receipt, comparisons, ema_admitted=ema_admitted
        )
        if declared_verdict != verdict:
            raise CampaignError(
                f"{regime} declared mission verdict does not match raw prefix evidence"
            )
        mission_verdicts.append(verdict)
        economics = receipt["window_economics"]
        exact_seconds, target_seconds = _validated_window_seconds(receipt)
        exact_window_total += exact_seconds
        target_window_total += target_seconds
        regime_rows.append(
            {
                "regime": regime,
                "receipt_path": str(paths[regime].resolve()),
                "receipt_sha256": _sha256(paths[regime]),
                "receipt_bytes": paths[regime].stat().st_size,
                "canonical_verdict": receipt["fidelity_verdict"]["verdict"],
                "mission_verdict": verdict,
                "mission_verdict_derivation": (
                    "recomputed from raw exact/surrogate metric traces, deterministic-repeat floors, "
                    "EMA admission, observed teacher skip, and full-cadence validation"
                ),
                "mission_verdict_reason": receipt["mission_verdict_reason"],
                "raw_exact_metric_trace_comparison": comparisons,
                "ema_final_admitted": ema_admitted,
                "observed_teacher_skip_fraction": economics["observed_exact_teacher_skip_fraction"],
                "target_cadence_fidelity_validated": economics["target_cadence_fidelity_validated"],
                "whole_window_speedup": economics["speedup"],
            }
        )

    mission_verdict = next(
        verdict for verdict in VERDICT_PRECEDENCE if verdict in mission_verdicts
    )
    timing_buckets = (
        "exact_forward_only",
        "exact_costate_forward_backward",
        "anchor_fit",
        "surrogate_inference",
        "renderer_vjp_exact_control",
        "renderer_vjp_surrogate_target",
        "surrogate_anchor_exact_costate",
        "surrogate_nonanchor_operational_step",
    )
    aggregate_timings = {
        bucket: _aggregate_bucket(receipts, bucket) for bucket in timing_buckets
    }
    exact_forward_mean = aggregate_timings["exact_forward_only"]["mean_seconds"]
    surrogate_mean = aggregate_timings["surrogate_inference"]["mean_seconds"]
    if exact_forward_mean is None or surrogate_mean is None or surrogate_mean <= 0.0:
        measured_forward_speedup = None
        operator_reference_speedup = None
    else:
        measured_forward_speedup = exact_forward_mean / surrogate_mean
        operator_reference_speedup = 1.656 / surrogate_mean

    return {
        "schema": SCHEMA,
        "status": "MEASURED",
        "mission_verdict": mission_verdict,
        "verdict_scope": (
            f"tested formulation; pair0, seed{common_config['seed']}, saved early/boundary/late regimes, "
            "up-to-five-step measured prefixes, macOS-CPU advisory training-gradient; "
            "not family or score authority"
        ),
        "regimes": regime_rows,
        "source_identity": source_identity,
        "common_treatment_config": common_config,
        "aggregate_isolated_timings": aggregate_timings,
        "aggregate_whole_matched_window": {
            "exact_seconds": exact_window_total,
            "surrogate_seconds": target_window_total,
            "speedup": exact_window_total / target_window_total,
            "comparison_basis": (
                "sum of symmetric complete per-step operational timers under each regime's "
                "exact-derived schedule; each step includes render, provider, renderer VJP, "
                "and candidate update; line-search and exact validation calls excluded"
            ),
        },
        "forward_replacement_economics": {
            "measured_same-run_exact_forward_over_surrogate_inference_speedup": measured_forward_speedup,
            "derived_operator_1656ms_over_measured_surrogate_inference_speedup": operator_reference_speedup,
            "operator_reference_seconds": 1.656,
            "operator_reference_label": "OPERATOR-SUPPLIED, not measured by these runs",
            "target_k20_full_cycle_fidelity": "UNKNOWN; blocked measured prefixes only",
        },
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "surrogate_eval_authority": False,
            "pointer_delta": "none",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", required=True, metavar="REGIME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths: dict[str, Path] = {}
    for spec in args.receipt:
        regime, separator, raw_path = spec.partition("=")
        if not separator or regime in paths:
            raise CampaignError("--receipt must be unique REGIME=PATH entries")
        paths[regime] = Path(raw_path)
    payload = aggregate(paths)
    payload["reducer_custody"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
        "argv": list(sys.argv if argv is None else [Path(__file__).name, *argv]),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    result = write_json_artifact(args.output, payload)
    print(json.dumps({"output": result.path, "sha256": result.sha256, "verdict": payload["mission_verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
