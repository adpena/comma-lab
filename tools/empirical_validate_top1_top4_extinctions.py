#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Empirical validation of TOP-1 + TOP-4 ARBITRARINESS-EXTINCTION canonical
helpers.

Per ARBITRARINESS-EXTINCTION audit 2026-05-18 commit ``2d042f7e6``:

* TOP-1: ``lambda_seg_pose_rate_multipliers_unprincipled`` -- analytical
  Lagrangian-multiplier helper :mod:`tac.score_lagrangian`. Empirical
  validation runs :func:`tac.score_lagrangian.compute_marginal_multipliers`
  at 5 canonical operating points and verifies CLAUDE.md anchor recovery
  (PR106 frontier ratio ~ 2.71; OLD 1.x ratio < 1).

* TOP-4: ``ema_decay_0.997_hardcoded_all_substrate_trainers`` -- canonical
  EMA-decay-window formula :meth:`tac.training.EMA.decay_from_total_steps`.
  Empirical validation: paired EMA convergence on a 1000-step toy
  optimization comparing the hardcoded ``decay=0.997`` against the
  closed-form formula at five total-step horizons (recovers Quantizr's
  empirical anchor at ``total_steps=1666``).

All persisted results are tagged ``[macOS-CPU advisory]`` per CLAUDE.md
"MPS auth eval is NOISE" + Catalog #192 + Catalog #317 -- never promoted
to ``[contest-CPU]`` without a paired Linux x86_64 anchor. The output
JSON carries canonical Provenance per Catalog #323.

Sister discipline:

* :mod:`tac.optimization.macos_cpu_advisory_signal` -- canonical store for
  research-signal manifests. Not used for the formula validation itself
  (that surface expects contest-archive observations, not math
  verification rows); instead the empirical validation emits an
  experiments/results artifact wrapped with canonical Provenance.

Catalog #125 six-hook wire-in (downstream consumers may chain on the
output JSON):

1. Sensitivity-map: ACTIVE via per-operating-point multiplier rows.
2. Pareto constraint: ACTIVE via the Lagrangian multipliers themselves.
3. Bit-allocator: N/A (above the per-tensor layer).
4. Cathedral autopilot dispatch: ACTIVE via canonical formula output.
5. Continual-learning posterior update: ACTIVE via the EMA-decay tracker.
6. Probe-disambiguator: ACTIVE via the operating-point classification.

Usage::

    python tools/empirical_validate_top1_top4_extinctions.py \\
        --output-dir experiments/results/empirical_validate_top1_top4_extinctions_20260518
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _utc_now_iso() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _detect_hardware_substrate() -> str:
    """Return canonical hardware substrate string per Catalog #190.

    Imports the canonical helper lazily to keep this tool callable in
    environments where ``tac.substrates`` is partially shadowed.
    """
    try:
        from tac.substrates._shared.trainer_skeleton import (
            detect_hardware_substrate,
        )

        return detect_hardware_substrate(axis="cpu")
    except Exception:
        # Fail-safe fallback per "macOS arm64" detection
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "darwin" and machine in {"arm64", "aarch64"}:
            return "macos_arm64"
        return f"{system}_{machine}_unknown_cpu"


# ---------------------------------------------------------------------------
# TOP-1: lambda_seg_pose_rate_multipliers_unprincipled
# ---------------------------------------------------------------------------


def _validate_top1_lambda_multipliers() -> dict[str, Any]:
    """Run :func:`compute_marginal_multipliers` at 5 canonical anchors.

    Returns a dict with per-anchor multipliers + classification, and a
    summary verdict comparing against CLAUDE.md "SegNet vs PoseNet
    importance" table.
    """
    from tac.score_lagrangian import (
        POSE_CROSSOVER_AT_AVG,
        compute_marginal_multipliers,
        empirical_anchor_old_1x_operating_point,
        empirical_anchor_pr106_frontier,
    )

    anchors: list[dict[str, Any]] = [
        {
            "name": "OLD_1x_seg_dominant_per_CLAUDE.md_anchor",
            "seg_avg": 0.07,
            "pose_avg": 0.18,
            "expected_classification": "old_1x_seg_dominant",
            "expected_pose_to_seg_ratio_band": (0.01, 0.20),
        },
        {
            "name": "crossover_at_pose_avg_2.5e-4",
            "seg_avg": 1e-3,
            "pose_avg": POSE_CROSSOVER_AT_AVG,
            "expected_classification": "crossover",
            "expected_pose_to_seg_ratio_band": (0.95, 1.05),
        },
        {
            "name": "PR106_frontier_per_CLAUDE.md_anchor",
            "seg_avg": 6.7e-4,
            "pose_avg": 3.4e-5,
            "expected_classification": "frontier_pose_dominant",
            "expected_pose_to_seg_ratio_band": (2.5, 2.9),
        },
        {
            "name": "extreme_frontier_pose_avg_1e-10",
            "seg_avg": 1e-6,
            "pose_avg": 1e-10,
            "expected_classification": "frontier_pose_dominant",
            "expected_pose_to_seg_ratio_band": (100.0, 1e8),
        },
        {
            "name": "synthetic_noise_pose_avg_2.0_seg_avg_2.0",
            "seg_avg": 2.0,
            "pose_avg": 2.0,
            "expected_classification": "old_1x_seg_dominant",
            "expected_pose_to_seg_ratio_band": (0.005, 0.05),
        },
    ]

    rows: list[dict[str, Any]] = []
    all_classifications_match = True
    all_ratios_in_band = True

    for anchor in anchors:
        mult = compute_marginal_multipliers(
            seg_avg=anchor["seg_avg"], pose_avg=anchor["pose_avg"]
        )
        ratio_band = anchor["expected_pose_to_seg_ratio_band"]
        classification_match = (
            mult.operating_point_classification == anchor["expected_classification"]
        )
        ratio_in_band = ratio_band[0] <= mult.pose_to_seg_ratio <= ratio_band[1]
        if not classification_match:
            all_classifications_match = False
        if not ratio_in_band:
            all_ratios_in_band = False
        rows.append(
            {
                "anchor_name": anchor["name"],
                "seg_avg": anchor["seg_avg"],
                "pose_avg": anchor["pose_avg"],
                "lambda_seg": mult.lambda_seg,
                "lambda_pose": mult.lambda_pose,
                "lambda_rate": mult.lambda_rate,
                "pose_to_seg_ratio": mult.pose_to_seg_ratio,
                "operating_point_classification": mult.operating_point_classification,
                "expected_classification": anchor["expected_classification"],
                "expected_ratio_band": list(ratio_band),
                "classification_match": classification_match,
                "ratio_in_expected_band": ratio_in_band,
            }
        )

    # Direct anchor-helper round-trip check.
    pr106 = empirical_anchor_pr106_frontier()
    old1x = empirical_anchor_old_1x_operating_point()
    helper_round_trip = {
        "pr106_anchor_ratio": pr106.pose_to_seg_ratio,
        "pr106_anchor_classification": pr106.operating_point_classification,
        "old_1x_anchor_ratio": old1x.pose_to_seg_ratio,
        "old_1x_anchor_classification": old1x.operating_point_classification,
        "pr106_ratio_matches_claude_md_2_71": abs(pr106.pose_to_seg_ratio - 2.71) < 0.02,
    }

    return {
        "top1_arbitrariness_id": "lambda_seg_pose_rate_multipliers_unprincipled",
        "canonical_helper": "tac.score_lagrangian.compute_marginal_multipliers",
        "anchors_count": len(anchors),
        "anchors": rows,
        "helper_round_trip": helper_round_trip,
        "summary_verdict": {
            "all_classifications_match_expected": all_classifications_match,
            "all_ratios_in_expected_band": all_ratios_in_band,
            "pr106_anchor_recovers_2_71_within_2_percent": helper_round_trip[
                "pr106_ratio_matches_claude_md_2_71"
            ],
            "overall_pass": (
                all_classifications_match
                and all_ratios_in_band
                and helper_round_trip["pr106_ratio_matches_claude_md_2_71"]
            ),
        },
    }


# ---------------------------------------------------------------------------
# TOP-4: ema_decay_0.997_hardcoded_all_substrate_trainers
# ---------------------------------------------------------------------------


def _ema_run_toy_optimization(
    *, total_steps: int, decay: float, target_value: float = 1.0
) -> dict[str, Any]:
    """Simulate EMA over a synthetic 1-D drift-and-converge optimization.

    Models a scalar parameter ``theta`` whose live value walks toward
    ``target_value`` with additive Gaussian noise (variance 1e-2). The
    EMA shadow tracks the live value. After ``total_steps``, we measure:

    * ``shadow_lag``: distance of shadow from target at the end.
    * ``shadow_variance``: variance of shadow's last 10 percent of steps.

    A well-tuned decay should produce shadow_lag near zero AND low
    shadow_variance simultaneously. A too-narrow decay (high noise
    bandwidth) has low lag but high variance; a too-wide decay (low
    noise bandwidth) has high lag and low variance.

    This is a *formula-validation* experiment, not a contest measurement.
    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192/#317, results
    are tagged ``[macOS-CPU advisory]`` and are non-promotable.
    """
    import random

    # Pin the seed for byte-deterministic replay per the canonical
    # pipeline standard.
    rng = random.Random(20260518)

    live = 0.0
    shadow = 0.0
    last_decile_values: list[float] = []
    n_decile = max(1, total_steps // 10)

    for step in range(total_steps):
        # Drift toward target with additive noise (canonical
        # SGD-like trajectory).
        live += 0.05 * (target_value - live) + 0.01 * rng.gauss(0.0, 1.0)
        # Apply canonical Polyak averaging update: shadow = decay * shadow
        # + (1 - decay) * live.
        shadow = decay * shadow + (1.0 - decay) * live
        if step >= total_steps - n_decile:
            last_decile_values.append(shadow)

    shadow_lag = abs(shadow - target_value)
    mean_last_decile = sum(last_decile_values) / len(last_decile_values)
    var_last_decile = sum(
        (v - mean_last_decile) ** 2 for v in last_decile_values
    ) / len(last_decile_values)

    return {
        "total_steps": total_steps,
        "decay": decay,
        "shadow_final": shadow,
        "shadow_lag": shadow_lag,
        "shadow_variance_last_decile": var_last_decile,
    }


def _validate_top4_ema_decay_formula() -> dict[str, Any]:
    """Paired EMA convergence comparison: hardcoded 0.997 vs formula.

    Five total-step horizons covering canonical training lengths:
    100 / 500 / 1666 / 5000 / 12000 steps. For each horizon, run both
    the hardcoded ``decay=0.997`` and the closed-form
    ``decay_from_total_steps(N)``; compare shadow lag + variance.

    Also verifies the canonical anchor: ``decay_from_total_steps(1666) == 0.997``
    rounded to 3 decimal places (the Quantizr PR101 anchor recovery).
    """
    from tac.training import EMA

    horizons = [100, 500, 1666, 5000, 12000]
    comparisons: list[dict[str, Any]] = []

    for total_steps in horizons:
        hardcoded_decay = 0.997
        formula_decay = EMA.decay_from_total_steps(total_steps)

        hardcoded_run = _ema_run_toy_optimization(
            total_steps=total_steps, decay=hardcoded_decay
        )
        formula_run = _ema_run_toy_optimization(
            total_steps=total_steps, decay=formula_decay
        )

        comparisons.append(
            {
                "total_steps": total_steps,
                "hardcoded_decay": hardcoded_decay,
                "formula_decay": formula_decay,
                "decay_delta": formula_decay - hardcoded_decay,
                "hardcoded_shadow_lag": hardcoded_run["shadow_lag"],
                "formula_shadow_lag": formula_run["shadow_lag"],
                "hardcoded_shadow_variance": hardcoded_run["shadow_variance_last_decile"],
                "formula_shadow_variance": formula_run["shadow_variance_last_decile"],
                "formula_better_lag": formula_run["shadow_lag"] < hardcoded_run["shadow_lag"],
                "formula_better_variance": (
                    formula_run["shadow_variance_last_decile"]
                    < hardcoded_run["shadow_variance_last_decile"]
                ),
            }
        )

    # Quantizr anchor recovery (TOP-4 canonical empirical validation).
    quantizr_anchor_decay = EMA.decay_from_total_steps(1666)
    quantizr_anchor_recovered = round(quantizr_anchor_decay, 3) == 0.997

    # Boundary clamping verification.
    short_run_clamp = EMA.decay_from_total_steps(1)
    huge_run_clamp = EMA.decay_from_total_steps(1_000_000_000)

    return {
        "top4_arbitrariness_id": "ema_decay_0.997_hardcoded_all_substrate_trainers",
        "canonical_helper": "tac.training.EMA.decay_from_total_steps",
        "horizons_compared": len(horizons),
        "comparisons": comparisons,
        "quantizr_anchor_recovery": {
            "decay_at_1666_steps": quantizr_anchor_decay,
            "rounded_3dp": round(quantizr_anchor_decay, 3),
            "matches_claude_md_0_997": quantizr_anchor_recovered,
        },
        "clamping_verification": {
            "decay_at_1_step": short_run_clamp,
            "decay_at_1e9_steps": huge_run_clamp,
            "clamps_at_0_99_floor": short_run_clamp == 0.99,
            "clamps_at_0_9999_ceiling": huge_run_clamp == 0.9999,
        },
        "summary_verdict": {
            "quantizr_anchor_recovered": quantizr_anchor_recovered,
            "clamp_floor_active": short_run_clamp == 0.99,
            "clamp_ceiling_active": huge_run_clamp == 0.9999,
            "overall_pass": (
                quantizr_anchor_recovered
                and short_run_clamp == 0.99
                and huge_run_clamp == 0.9999
            ),
        },
    }


# ---------------------------------------------------------------------------
# Provenance + persistence
# ---------------------------------------------------------------------------


def _wrap_with_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap empirical-validation payload with canonical Provenance per Catalog #323.

    The payload is NOT a contest score claim; it is a math-formula
    validation. Provenance kind = PREDICTED (the multiplier / decay
    values are model-derived predictions); grade = MACOS_CPU_ADVISORY
    (math run on macOS CPU per CLAUDE.md "MPS auth eval is NOISE" rule
    extended to formula validation).
    """
    try:
        from tac.provenance import (
            build_provenance_for_macos_cpu_advisory,
            provenance_to_dict,
        )

        # Use a synthetic archive sha (the validation does not target any
        # contest archive); per CLAUDE.md Catalog #323 the sha is required
        # for canonical Provenance audit.
        synthetic_sha = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        prov = build_provenance_for_macos_cpu_advisory(
            archive_sha256=synthetic_sha,
            source_path=(
                "tools/empirical_validate_top1_top4_extinctions.py"
                ":formula_validation_only_NOT_a_contest_archive_score_claim"
            ),
            captured_at_utc=_utc_now_iso(),
        )
        prov_dict = provenance_to_dict(prov)
    except Exception as exc:
        # Provenance is decorative for this validation (no score claim);
        # fall back to a minimal dict so the artifact still lands.
        prov_dict = {
            "_provenance_construction_warning": (
                f"could not build canonical Provenance: {type(exc).__name__}: {exc}; "
                "this payload is NOT a contest score claim so the failure is non-blocking"
            ),
            "kind": "predicted",
            "evidence_grade": "macOS-CPU-advisory",
            "captured_at_utc": _utc_now_iso(),
        }

    wrapped = {
        "schema": "empirical_validate_top1_top4_extinctions.v1",
        "captured_at_utc": _utc_now_iso(),
        "hardware_substrate": _detect_hardware_substrate(),
        "evidence_grade": "macOS-CPU-advisory",
        "evidence_tag": "[macOS-CPU advisory only]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "claude_md_axis_compliance": (
            "[macOS-CPU advisory] per CLAUDE.md 'MPS auth eval is NOISE' + "
            "Catalog #192/#317; this is a formula validation, NOT a contest "
            "score claim; promotion to [contest-CPU] requires a paired "
            "Linux x86_64 anchor on the same closed-form formula (not "
            "applicable since this is a math derivation)."
        ),
        "provenance": prov_dict,
        "payload": payload,
    }
    return wrapped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Empirical validation of TOP-1 (Lagrangian multipliers) + TOP-4 "
            "(EMA decay formula) ARBITRARINESS-EXTINCTION canonical helpers."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/empirical_validate_top1_top4_extinctions_20260518"),
        help="Directory to write validation_report.json + console summary.",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    top1_result = _validate_top1_lambda_multipliers()
    top4_result = _validate_top4_ema_decay_formula()

    combined_payload = {
        "top1_lambda_multipliers": top1_result,
        "top4_ema_decay_formula": top4_result,
        "lane_id": "lane_arbitrariness_extinction_top1_top4_batched_wave_20260518",
        "audit_source": (
            ".omx/state/arbitrariness_extinction_audit_20260518.jsonl "
            "(commit 2d042f7e6)"
        ),
        "canonical_helpers_validated": [
            "tac.score_lagrangian.compute_marginal_multipliers",
            "tac.score_lagrangian.empirical_anchor_pr106_frontier",
            "tac.score_lagrangian.empirical_anchor_old_1x_operating_point",
            "tac.training.EMA.decay_from_total_steps",
        ],
        "overall_verdict": (
            "PASS"
            if top1_result["summary_verdict"]["overall_pass"]
            and top4_result["summary_verdict"]["overall_pass"]
            else "FAIL"
        ),
    }

    wrapped = _wrap_with_provenance(combined_payload)

    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(wrapped, indent=2, default=str))

    # Console summary
    print()
    print("=" * 78)
    print("ARBITRARINESS-EXTINCTION TOP-1 + TOP-4 empirical validation")
    print("=" * 78)
    print(f"Hardware substrate: {wrapped['hardware_substrate']}")
    print(f"Evidence grade: {wrapped['evidence_grade']}")
    print(f"Captured at: {wrapped['captured_at_utc']}")
    print()
    print("TOP-1 (lambda multipliers):")
    print(f"  canonical helper: {top1_result['canonical_helper']}")
    print(f"  anchors run: {top1_result['anchors_count']}")
    print(f"  PR106 ratio recovers 2.71: {top1_result['helper_round_trip']['pr106_ratio_matches_claude_md_2_71']}")
    print(f"  overall pass: {top1_result['summary_verdict']['overall_pass']}")
    print()
    print("TOP-4 (EMA decay formula):")
    print(f"  canonical helper: {top4_result['canonical_helper']}")
    print(f"  horizons compared: {top4_result['horizons_compared']}")
    print(f"  Quantizr anchor recovered: {top4_result['quantizr_anchor_recovery']['matches_claude_md_0_997']}")
    print(f"  overall pass: {top4_result['summary_verdict']['overall_pass']}")
    print()
    print(f"OVERALL: {combined_payload['overall_verdict']}")
    print(f"Report written: {report_path}")
    print()
    return 0 if combined_payload["overall_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
