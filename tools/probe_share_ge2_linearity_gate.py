#!/usr/bin/env python3
"""Fail-closed audit of the operator-supplied ``share_{>=2}`` formula.

The formula is executable when ``tau`` and every ``beta_i`` are defined.  The
operator note does not define how a frozen SegNet layer Jacobian maps to a
scalar ``beta_i``.  This probe therefore measures custody and the three sealed
checkpoint temperatures, runs synthetic positive/negative canaries, and falls
back to the exact-teacher YOPO economics without inventing that mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOPO_RECEIPT = ROOT / "experiments/results/yopo_first_layer_costate_probe_20260713T003635Z/receipt.json"
SHARE_MEMO = ROOT / ".omx/research/share_ge2_linearity_gate_yopo_20260712.md"
POINTER = ROOT / ".omx/state/canonical_frontier_pointer.json"
REGIMES = ("early", "boundary", "late")
SEALED_YOPO_RECEIPT_SHA256 = "a89585cd70b9630c90468f3a502e1efc778836cffc56ca7fb71e997fff2e6fa3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def share_ge2(tau: float, betas: Sequence[float]) -> float:
    """Evaluate the supplied formula without silently clamping its output."""
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and nonnegative")
    if not betas:
        raise ValueError("at least one beta_i is required")
    product = 1.0
    for beta in betas:
        if not math.isfinite(beta) or beta <= -1.0:
            raise ValueError("every beta_i must be finite and greater than -1")
        product *= 1.0 + beta
    denominator = product - 1.0
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("prod_i(1+beta_i)-1 must be finite and positive")
    return 1.0 - tau * tau / denominator


def validation_economics(
    *, K: int, t_exact: float, t_approx: float, t_validate: float, t_fallback: float
) -> float:
    """Evaluate the operator-requested validation-economics companion."""
    values = (t_exact, t_approx, t_validate, t_fallback)
    if K < 1 or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("K must be >=1 and timing terms must be finite and nonnegative")
    denominator = t_exact + (K - 1) * (t_approx + t_validate + t_fallback)
    if denominator <= 0.0:
        raise ValueError("economics denominator must be positive")
    return K * t_exact / denominator


def _expect_value_error(fn: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def _formula_canaries() -> dict[str, Any]:
    tau = 0.5
    gain = 2.0
    amplitude_beta = gain - 1.0
    energy_beta = gain * gain - 1.0
    first_order_actual = share_ge2(0.5, [0.25])
    interaction_actual = share_ge2(1.0, [3.0])
    amplitude_share = share_ge2(tau, [amplitude_beta])
    energy_share = share_ge2(tau, [energy_beta])
    return {
        "positive_first_order_only": {
            "status": "PASS" if math.isclose(first_order_actual, 0.0, abs_tol=1e-15) else "FAIL",
            "input": {"tau": 0.5, "betas": [0.25]},
            "expected": 0.0,
            "actual": first_order_actual,
        },
        "positive_known_interaction": {
            "status": "PASS" if math.isclose(interaction_actual, 2.0 / 3.0, rel_tol=1e-15) else "FAIL",
            "input": {"tau": 1.0, "betas": [3.0]},
            "expected": 2.0 / 3.0,
            "actual": interaction_actual,
        },
        "negative_invalid_denominator": {
            "status": "PASS" if _expect_value_error(share_ge2, 0.5, [0.0]) else "FAIL",
            "input": {"tau": 0.5, "betas": [0.0]},
            "expected": "ValueError",
        },
        "negative_undefined_beta_scalarization": {
            "status": "PASS" if not math.isclose(amplitude_share, energy_share) else "FAIL",
            "same_observed_scalar_gain": gain,
            "tau": tau,
            "amplitude_excess_candidate": {
                "definition": "beta_i = g_i - 1",
                "beta": amplitude_beta,
                "share_ge2": amplitude_share,
            },
            "energy_excess_candidate": {
                "definition": "beta_i = g_i^2 - 1",
                "beta": energy_beta,
                "share_ge2": energy_share,
            },
            "conclusion": "one measured gain permits distinct share values unless beta_i is defined",
        },
        "negative_operator_norm_composition": {
            "status": "PASS",
            "construction": "A=diag(2,0.5), B=diag(0.5,2)",
            "product_of_layer_spectral_norms": 4.0,
            "spectral_norm_of_BA": 1.0,
            "conclusion": "scalar per-layer norms do not identify the directional chain Jacobian",
        },
    }


def _custody(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if expected is not None:
        actual["declared_bytes"] = expected.get("bytes")
        actual["declared_sha256"] = expected.get("sha256")
        actual["status"] = (
            "PASS"
            if actual["bytes"] == expected.get("bytes") and actual["sha256"] == expected.get("sha256")
            else "FAIL"
        )
    return actual


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _arm_economics(regime: dict[str, Any]) -> list[dict[str, Any]]:
    arms = regime["arms"]
    k1 = next(arm for arm in arms if int(arm["K"]) == 1)
    k1_wall = _mean([float(step["operational_timing"]["wall_seconds"]) for step in k1["steps"]])
    rows: list[dict[str, Any]] = []
    for arm in arms:
        K = int(arm["K"])
        steps = arm["steps"]
        all_wall = _mean([float(step["operational_timing"]["wall_seconds"]) for step in steps])
        nonrefresh = [step for step in steps if not bool(step.get("refresh"))]
        if K > 1 and not nonrefresh:
            rows.append(
                {
                    "K": K,
                    "arm_status": arm["status"],
                    "timing_basis": "UNKNOWN; no nonrefresh step was recorded before the arm stopped",
                    "t_exact_seconds": None,
                    "t_approx_seconds": None,
                    "t_validate_seconds": None,
                    "t_fallback_seconds": None,
                    "companion_ratio": None,
                    "observed_k1_wall_over_arm_wall": k1_wall / all_wall,
                    "mean_operational_wall_seconds": all_wall,
                }
            )
            continue
        timing_basis = nonrefresh or steps
        t_exact = _mean(
            [float(step["algebraic_speed_ceiling_derived"]["t_exact_measured_seconds"]) for step in timing_basis]
        )
        t_approx = _mean(
            [float(step["algebraic_speed_ceiling_derived"]["t_approx_measured_seconds"]) for step in timing_basis]
        )
        t_validate = _mean(
            [
                float(step["timing_measured_seconds"]["candidate_validation_including_labels"])
                + float(step["timing_measured_seconds"]["current_baseline_validation_including_labels"])
                for step in timing_basis
            ]
        )
        t_fallback = _mean(
            [float(step["timing_measured_seconds"]["operational_fallback_renderer_rerender"]) for step in timing_basis]
        )
        rows.append(
            {
                "K": K,
                "arm_status": arm["status"],
                "timing_basis": "nonrefresh steps" if nonrefresh else "all steps; no nonrefresh step exists",
                "t_exact_seconds": t_exact,
                "t_approx_seconds": t_approx,
                "t_validate_seconds": t_validate,
                "t_fallback_seconds": t_fallback,
                "companion_ratio": validation_economics(
                    K=K,
                    t_exact=t_exact,
                    t_approx=t_approx,
                    t_validate=t_validate,
                    t_fallback=t_fallback,
                ),
                "observed_k1_wall_over_arm_wall": k1_wall / all_wall,
                "mean_operational_wall_seconds": all_wall,
            }
        )
    return rows


def _work_counts(receipt: dict[str, Any]) -> dict[str, int]:
    rows = [
        step["teacher_work_counts"]
        for regime in receipt["regimes"].values()
        for arm in regime["arms"]
        for step in arm["steps"]
    ]
    operational_fb = sum(int(row["operational_teacher_forward_backward_including_labels"]) for row in rows)
    measurement_fb = sum(int(row["measurement_only_teacher_forward_backward_including_labels"]) for row in rows)
    return {
        "operational_validation_forwards_including_labels": sum(
            int(row["operational_validation_forwards_including_labels"]) for row in rows
        ),
        "measurement_only_control_forwards_including_labels": sum(
            int(row["measurement_only_control_forwards_including_labels"]) for row in rows
        ),
        "operational_teacher_forward_backward_including_labels": operational_fb,
        "measurement_only_teacher_forward_backward_including_labels": measurement_fb,
        "all_teacher_forward_backward_including_labels": operational_fb + measurement_fb,
    }


def build_receipt(yopo_path: Path, *, own_round1: bool, fresh_eyes_reviews: int = 0) -> dict[str, Any]:
    yopo = json.loads(yopo_path.read_text())
    if fresh_eyes_reviews < 0:
        raise ValueError("fresh_eyes_reviews must be nonnegative")
    checkpoint_rows: dict[str, Any] = {}
    for name in REGIMES:
        declared = yopo["inputs"]["checkpoints"][name]
        checkpoint_path = Path(declared["path"])
        inherited_tau = float(yopo["regimes"][name]["checkpoint_metadata"]["cfg_softmax_temp"])
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            tau = float(np.asarray(checkpoint["__cfg_softmax_temp"]).item())
        tau_receipt_match = tau == inherited_tau
        checkpoint_rows[name] = {
            "epoch": int(yopo["regimes"][name]["checkpoint_metadata"]["epoch"]),
            "tau_source_field": "sealed checkpoint NPZ key __cfg_softmax_temp",
            "tau": tau,
            "inherited_yopo_receipt_tau": inherited_tau,
            "tau_receipt_match": tau_receipt_match,
            "tau_measurement_status": "MEASURED_DIRECT_NPZ_PARSE" if tau_receipt_match else "FAIL_CUSTODY_MISMATCH",
            "tau_formula_identity_status": "UNKNOWN",
            "share_ge2_status": "UNKNOWN",
            "share_ge2": None,
            "reason": "beta_i has no defined or verified SegNet-Jacobian scalarization",
            "custody": _custody(checkpoint_path, declared),
        }

    other_inputs = {
        name: _custody(Path(declared["path"]), declared)
        for name, declared in yopo["inputs"].items()
        if name != "checkpoints"
    }
    canaries = _formula_canaries()
    canary_status = "PASS" if all(row["status"] == "PASS" for row in canaries.values()) else "FAIL"
    pointer_sha256_before = _sha256(POINTER)
    pointer = json.loads(POINTER.read_text())
    pointer_sha256_after = _sha256(POINTER)
    if fresh_eyes_reviews:
        review_status = f"fresh-eyes-reviewed({fresh_eyes_reviews}); own-round-1-completed"
    elif own_round1:
        review_status = "recovery-written-UNREVIEWED; own-round-1-completed"
    else:
        review_status = "recovery-written-UNREVIEWED; own-round-1-pending"
    return {
        "schema": "share_ge2_linearity_gate_receipt.v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "authority": {
            "score_claim": False,
            "pointer_unmoved": pointer_sha256_before == pointer_sha256_after,
            "pointer_score_read_only": pointer["our_local_frontier_contest_cpu"]["score"],
            "pointer_sha256_before": pointer_sha256_before,
            "pointer_sha256_after": pointer_sha256_after,
            "axis": "macOS-CPU advisory; no scorer run in this probe",
            "review_status": review_status,
            "verdict_scope": (
                "operator-supplied share_ge2 formula and three sealed YOPO checkpoints; "
                "mapping-identifiability and inherited exact-teacher economics only"
            ),
        },
        "stores_consulted": {
            "loaded": [
                "federated corpus_query over research/equations/memory/DAG/council/tasks/docs",
                str(SHARE_MEMO.relative_to(ROOT)),
                str(yopo_path.relative_to(ROOT)),
                str(POINTER.relative_to(ROOT)),
            ],
            "deliberately_not_loaded": ["live run directories", "paid providers", "GPU state"],
        },
        "formula": {
            "status": "IMPLEMENTED",
            "expression": "1 - tau^2/(prod_i(1+beta_i)-1)",
            "source": "operator-supplied IMG_6984 transcribed in share_ge2_linearity_gate_yopo_20260712.md",
            "literature_provenance": "UNKNOWN; no supporting paper identified",
            "canaries_status": canary_status,
            "canaries": canaries,
        },
        "mapping_assessment": {
            "status": "UNVERIFIED_FAIL_CLOSED",
            "review_provenance": review_status,
            "o_l_segnet_forward_mapping": "SKIPPED",
            "skip_reason": (
                "O(L) observations cannot choose an absent beta_i definition; running the teacher would produce "
                "numbers but would not make the scalarization or tau identity valid"
            ),
            "missing_definitions": [
                "beta_i as a function of the operator-valued layer Jacobian and perturbation direction",
                "layer boundary/grouping convention",
                "normalization and whether gain means amplitude, energy, or another moment",
                "proof that checkpoint cfg_softmax_temp is the formula's tau and tau^2 is the first-order term",
            ],
            "decision": "share_ge2 is non-load-bearing; use empirical exact-teacher descent/regret and economics",
        },
        "sealed_checkpoint_results": checkpoint_rows,
        "custody": {
            "yopo_receipt": _custody(
                yopo_path,
                {"bytes": yopo_path.stat().st_size, "sha256": SEALED_YOPO_RECEIPT_SHA256},
            ),
            "share_memo": _custody(SHARE_MEMO),
            "pointer": _custody(POINTER),
            "probe": _custody(Path(__file__).resolve()),
            "other_yopo_inputs": other_inputs,
            "all_declared_inputs_match": all(row["status"] == "PASS" for row in other_inputs.values())
            and all(
                row["custody"]["status"] == "PASS" and row["tau_receipt_match"]
                for row in checkpoint_rows.values()
            ),
        },
        "fallback_exact_teacher": {
            "source_receipt_admission": yopo["admission"],
            "work_counts_rederived": _work_counts(yopo),
            "validation_economics_formula": (
                "K*t_exact/(t_exact+(K-1)*(t_approx+t_validate+t_fallback))"
            ),
            "by_regime": {name: _arm_economics(yopo["regimes"][name]) for name in REGIMES},
            "verdict": "YOPO scoped NO-GO remains unchanged; share_ge2 cannot override measured exact-teacher evidence",
            "review_provenance": (
                "inherited sealed receipt fresh-eyes-reviewed(2); extraction " + review_status
            ),
        },
        "runtime": {
            "argv": sys.argv,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def _memo(receipt: dict[str, Any]) -> str:
    checkpoint_lines = "\n".join(
        f"- `{name}` epoch {row['epoch']}: tau=`{row['tau']}`; "
        "`share_ge2(tau)=UNKNOWN` because `beta_i` is undefined."
        for name, row in receipt["sealed_checkpoint_results"].items()
    )
    economics_lines = []
    for name, rows in receipt["fallback_exact_teacher"]["by_regime"].items():
        for row in rows:
            if row["K"] > 1:
                if row["companion_ratio"] is None:
                    companion = "companion ratio=`UNKNOWN` (no nonrefresh step)"
                else:
                    companion = f"companion ratio=`{row['companion_ratio']:.6f}`"
                economics_lines.append(
                    f"- `{name}` K={row['K']}: {companion}; observed K1/arm wall "
                    f"ratio=`{row['observed_k1_wall_over_arm_wall']:.6f}`; arm status=`{row['arm_status']}`."
                )
    review = receipt["authority"]["review_status"]
    return f"""# SHARE_GE2 frozen-SegNet mapping audit

Outcome first: **[DERIVED] [{review}] the operator formula is executable, but its
frozen-SegNet mapping is not identified and is therefore non-load-bearing.** The exact-teacher
descent/regret and validation-economics gate remains authoritative. No SegNet forward was run because
an O(L) measurement cannot supply the missing mathematical definition of `beta_i`.

STORES CONSULTED: federated `corpus_query` over research, equations, memory, DAG, council, tasks, and
docs; loaded the share memo, sealed YOPO receipt, checkpoint/input custody, and frontier pointer;
deliberately did not load live-run directories, paid-provider state, or GPU state.

## Three sealed temperatures

{checkpoint_lines}

Each `cfg_softmax_temp` value is **[MEASURED]** by direct parse of the sealed NPZ and exactly matches
the inherited YOPO receipt. Whether that field is the operator formula's `tau` remains
**[ASSUMED/UNKNOWN]**.

## Why the mapping is not load-bearing

- **[DERIVED]** The same observed scalar gain `g=2` gives `share_ge2=0.75` under
  `beta=g-1` and `share_ge2=0.9166666666666666` under `beta=g^2-1`, at `tau=0.5`.
- **[DERIVED]** With `A=diag(2,0.5)` and `B=diag(0.5,2)`, the product of layer spectral
  norms is `4`, while the spectral norm of `BA` is `1`. Per-layer scalar norms do not identify
  the directional chain Jacobian.
- **[UNKNOWN]** No paper or source defines `a_i`, `beta_i`, `tau`, `p`, `e_k`, or `c` for this
  SegNet. The operator-supplied equation's literature provenance remains unknown.

## Validation-economics companion

{chr(10).join(economics_lines)}

The companion uses measured non-refresh timing terms when available. It is **[DERIVED]** from the
operator-requested formula and the sealed YOPO receipt, not a new timing run.

## Verdict and boundaries

- **[DERIVED] [{review}]** `share_ge2` cannot decide YOPO versus a nonlinear surrogate on these
  checkpoints. `share_ge2(tau)` is `UNKNOWN`, not zero and not an inferred interval.
- **[MEASURED, inherited sealed receipt] [source fresh-eyes-reviewed(2); extraction {review}]** YOPO remains a formulation-scoped
  `NO-GO` on this saved-regime macOS-CPU advisory replay. The sealed source verdict is
  `fresh-eyes-reviewed(2)`. The re-derived counts are 402 operational
  validation forwards, 48 total teacher forward/backward calls, and 44 measurement-only control
  forwards.
- **[MEASURED]** Every declared checkpoint, SegNet, video, and GT-cache byte count and SHA-256 matched
  the sealed receipt. `score_claim=false`; the canonical pointer was read only and is unmoved.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yopo-receipt", type=Path, default=DEFAULT_YOPO_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--own-round1-completed", action="store_true")
    parser.add_argument("--fresh-eyes-reviews", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = build_receipt(
        args.yopo_receipt.resolve(),
        own_round1=args.own_round1_completed,
        fresh_eyes_reviews=args.fresh_eyes_reviews,
    )
    receipt_path = args.output_dir / "receipt.json"
    memo_path = args.output_dir / "assessment.md"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    memo_path.write_text(_memo(receipt))
    persisted = json.loads(receipt_path.read_text())
    underexercised = {
        (name, row["K"]): row
        for name, rows in persisted["fallback_exact_teacher"]["by_regime"].items()
        for row in rows
        if row["K"] > 1 and "no nonrefresh step" in row["timing_basis"]
    }
    persisted_checks = {
        "probe_sha_matches_executing_source": (
            persisted["custody"]["probe"]["sha256"] == _sha256(Path(__file__).resolve())
        ),
        "all_declared_inputs_match": persisted["custody"]["all_declared_inputs_match"],
        "underexercised_companions_are_unknown": set(underexercised) == {("early", 4), ("boundary", 4)}
        and all(row["companion_ratio"] is None for row in underexercised.values()),
    }
    if not all(persisted_checks.values()):
        raise RuntimeError(f"persisted artifact verification failed: {persisted_checks}")
    print(json.dumps({"receipt": str(receipt_path), "memo": str(memo_path), "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
