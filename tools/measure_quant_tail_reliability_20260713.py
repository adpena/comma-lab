#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure cached finite-design tail reliability for Pact interpolators.

Containment is fail closed: this tool reads sealed PRE-SE/organ artifacts, does
NumPy-only refits, and writes one new receipt.  It does not call a scorer, train
a witness, mutate a live run, dispatch work, touch an archive, or move a pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.canonical_equations.control_interpolator_tail_reliability_20260713 import (  # noqa: E402
    loss_tail_summary,
    retained_mass_tail_summary,
    select_tail_lambda,
)
from tac.scorer_surrogate.replace_round4_support_ranking import (  # noqa: E402
    ORDERED_PAIR_COUNT,
    QuadraticStatistics,
    fit_exact_quadratic,
)
from tac.witness_dsl.control_tail_reliability_policy_20260713 import (  # noqa: E402
    ControlTailReliabilityPolicy,
)

LANE_ID = "lane_quant_tail_reliability_20260713"
SCHEMA = "quant_tail_reliability_20260713.v1"
DEFAULT_OUTPUT = REPO / ".omx/research/quant_tail_reliability_receipt_20260713.json"
LOCUS_NAMES = ("block2-pre-se", "block3-pre-se")


class MeasurementError(RuntimeError):
    """Cached evidence or a numerical invariant failed closed."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _custody(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MeasurementError(f"required cached artifact missing: {path}")
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise MeasurementError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rankrls_statistics(
    features: np.ndarray,
    pair_ids: np.ndarray,
    support: np.ndarray,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified row-bootstrap RankRLS spectra and imported MP certificates."""

    generator = np.random.default_rng(seed)
    spectral: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    width = int(features.shape[1])
    for block in range(ORDERED_PAIR_COUNT):
        positive = features[(pair_ids == block) & support].astype(np.float64, copy=False)
        negative = features[(pair_ids == block) & ~support].astype(np.float64, copy=False)
        if positive.size and negative.size:
            positive = positive[
                generator.integers(0, positive.shape[0], positive.shape[0])
            ]
            negative = negative[
                generator.integers(0, negative.shape[0], negative.shape[0])
            ]
            n_positive = int(positive.shape[0])
            n_negative = int(negative.shape[0])
            with np.errstate(all="ignore"):
                sum_positive = positive.sum(axis=0, dtype=np.float64)
                sum_negative = negative.sum(axis=0, dtype=np.float64)
                gram = (
                    n_negative * (positive.T @ positive)
                    + n_positive * (negative.T @ negative)
                    - np.outer(sum_positive, sum_negative)
                    - np.outer(sum_negative, sum_positive)
                )
                rhs = n_negative * sum_positive - n_positive * sum_negative
            gram = np.ascontiguousarray(0.5 * (gram + gram.T), dtype=np.float64)
            stats = QuadraticStatistics(
                gram=gram,
                rhs=np.ascontiguousarray(rhs, dtype=np.float64),
                target_square=float(n_positive * n_negative),
                row_count=n_positive * n_negative,
                state_count=420,
            )
        else:
            n_positive = int(positive.shape[0])
            n_negative = int(negative.shape[0])
            stats = QuadraticStatistics(
                gram=np.zeros((width, width), dtype=np.float64),
                rhs=np.zeros(width, dtype=np.float64),
                target_square=0.0,
                row_count=0,
                state_count=0,
            )
        if not np.isfinite(stats.gram).all() or not np.isfinite(stats.rhs).all():
            raise MeasurementError(f"nonfinite bootstrap sufficient statistics in block {block}")
        mp_fit = fit_exact_quadratic(stats)
        eigenvalues, eigenvectors = np.linalg.eigh(stats.gram)
        maximum = max(0.0, float(eigenvalues[-1])) if eigenvalues.size else 0.0
        cutoff = np.finfo(np.float64).eps * max(1, width) * maximum
        scale = float(np.trace(stats.gram) / max(1, width)) or 1.0
        with np.errstate(all="ignore"):
            projected_rhs = eigenvectors.T @ stats.rhs
        if not np.isfinite(projected_rhs).all():
            raise MeasurementError(f"nonfinite projected RHS in block {block}")
        spectral.append(
            {
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "projected_rhs": projected_rhs,
                "scale": scale,
                "cutoff": cutoff,
                "mp_weights": mp_fit.weights,
            }
        )
        audit.append(
            {
                "block": block,
                "positive_rows": n_positive,
                "negative_rows": n_negative,
                "ridge_scale_mean_gram_diagonal": scale,
                "mp_certificate": mp_fit.certificate,
            }
        )
    return spectral, audit


def _weights_for_lambda(
    spectral: list[dict[str, Any]], *, lambda_value: float, width: int
) -> np.ndarray:
    weights = np.zeros((ORDERED_PAIR_COUNT, width), dtype=np.float64)
    for block, row in enumerate(spectral):
        if lambda_value == 0.0:
            weights[block] = row["mp_weights"]
            continue
        eigenvalues = row["eigenvalues"]
        eigenvectors = row["eigenvectors"]
        denominator = np.maximum(eigenvalues, 0.0) + lambda_value * row["scale"]
        if np.any(denominator <= 0.0):
            raise MeasurementError("positive ridge produced a nonpositive denominator")
        with np.errstate(all="ignore"):
            weights[block] = eigenvectors @ (row["projected_rhs"] / denominator)
    if not np.isfinite(weights).all():
        raise MeasurementError("ridge weights are nonfinite")
    return weights


def _state_retained_mass(
    scores: np.ndarray,
    mass: np.ndarray,
    offsets: np.ndarray,
    *,
    area_fraction: float,
) -> tuple[list[float], float]:
    values: list[float] = []
    retained_total = 0.0
    mass_total = 0.0
    for start, end in pairwise(offsets):
        state_scores = scores[int(start) : int(end)]
        state_mass = mass[int(start) : int(end)].astype(np.float32, copy=False)
        count = max(1, math.ceil(area_fraction * state_scores.size))
        order = np.lexsort(
            (np.arange(state_scores.size, dtype=np.int64), -state_scores)
        )[:count]
        retained = float(state_mass[order].sum(dtype=np.float32))
        total = float(state_mass.sum(dtype=np.float32))
        if total <= 0.0:
            raise MeasurementError("cached dev state has zero exact costate mass")
        values.append(retained / total)
        retained_total += retained
        mass_total += total
    return values, retained_total / mass_total


def _load_pre_se_arrays(
    root: Path, locus: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    chunk_root = root / "loci" / locus / "nonlinear" / "stage_chunks"
    paths = sorted(chunk_root.glob("*.npz"))
    if len(paths) != 3:
        raise MeasurementError(f"{locus} must have exactly three sealed stage chunks")
    chunks: list[dict[str, np.ndarray]] = []
    custody: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            chunk = {name: np.array(archive[name], copy=True) for name in archive.files}
        required = {
            "core_x",
            "core_pair",
            "core_y",
            "dev_x",
            "dev_pair",
            "dev_mass",
            "dev_lengths",
        }
        if not required.issubset(chunk):
            raise MeasurementError(f"cached stage chunk schema drift: {path}")
        chunks.append(chunk)
        custody.append(_custody(path))
    data = {
        "core_x": np.concatenate([row["core_x"] for row in chunks]).astype(np.float32),
        "core_pair": np.concatenate([row["core_pair"] for row in chunks]).astype(np.int16),
        "core_y": np.concatenate([row["core_y"] for row in chunks]).astype(np.bool_),
        "dev_x": np.concatenate([row["dev_x"] for row in chunks]).astype(np.float32),
        "dev_pair": np.concatenate([row["dev_pair"] for row in chunks]).astype(np.int16),
        "dev_mass": np.concatenate([row["dev_mass"] for row in chunks]).astype(np.float32),
        "dev_lengths": np.concatenate([row["dev_lengths"] for row in chunks]).astype(np.int64),
    }
    data["dev_offsets"] = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(data["dev_lengths"], dtype=np.int64))
    )
    if data["dev_lengths"].size != 60:
        raise MeasurementError("cached train-only dev state count is not 60")
    if not all(np.isfinite(data[name]).all() for name in ("core_x", "dev_x", "dev_mass")):
        raise MeasurementError("cached PRE-SE arrays contain nonfinite values")
    return data, custody


def _official_mp_anchor(root: Path, locus: str, *, alpha: float) -> dict[str, Any]:
    records = sorted((root / "heldout").glob("pair_*.json"))
    if len(records) != 120:
        raise MeasurementError("official PRE-SE heldout receipt count is not 120")
    values: list[float] = []
    custody_digest = hashlib.sha256()
    for path in records:
        row = _load_json(path)
        selection = row["loci"][locus]["selections"]["convex-deeper-pair-block-mp"]
        values.append(float(selection["retained_mass"]) / float(selection["total_mass"]))
        custody_digest.update(bytes.fromhex(_sha256(path)))
    return {
        "scope": "official untouched n120 heldout; MP only",
        "summary": retained_mass_tail_summary(values, alpha=alpha),
        "state_values": values,
        "receipt_count": len(records),
        "ordered_receipt_sha256_digest": custody_digest.hexdigest(),
        "new_lambda_evaluable_from_cache": False,
        "blocker": (
            "heldout JSON preserves MP selections and costate/feature hashes, not raw feature "
            "rows or exact mass arrays; new-lambda top masks cannot be reconstructed"
        ),
    }


def _measure_pre_se(policy: ControlTailReliabilityPolicy) -> dict[str, Any]:
    root = REPO / policy.pre_se_cache
    receipt = _load_json(root / "receipt.json")
    if receipt.get("schema") != "pre_se_locus_20260713.v1":
        raise MeasurementError("PRE-SE receipt schema drift")
    accounting = receipt["teacher_call_accounting"]
    if not accounting.get("all_n600_states_completed") or (
        int(accounting["inherited_round5_exact_train_targets"])
        + int(accounting["fresh_exact_heldout_unique_completed_states"])
        != 600
    ):
        raise MeasurementError("PRE-SE n600 exact-call custody does not close")
    output: dict[str, Any] = {
        "custody": {
            "receipt": _custody(root / "receipt.json"),
            "complete": _custody(root / "complete.json"),
            "n600": {"train_inherited": 480, "untouched_heldout": 120, "total": 600},
        },
        "measurement_scope": (
            "MEASURED-CACHED cross-fit curve on 420 core / 60 train-only dev real states; "
            "official n120 heldout remains an MP-only anchor"
        ),
        "bootstrap_scope": (
            "row bootstrap within ordered-pair/support strata; state-cluster bootstrap is "
            "unavailable because core state boundaries were not preserved"
        ),
        "loci": {},
    }
    for locus in LOCUS_NAMES:
        data, chunk_custody = _load_pre_se_arrays(root, locus)
        rows_by_lambda: dict[float, dict[str, Any]] = {
            value: {
                "lambda": value,
                "retained_mass_values": [],
                "seed_summaries": [],
                "mass_weighted_aggregates": [],
                "max_abs_score_fp32_vs_fp64": 0.0,
            }
            for value in policy.lambda_grid
        }
        spectral_audit: dict[str, Any] = {}
        for seed in policy.fit_resample_seeds:
            spectral, audit = _rankrls_statistics(
                data["core_x"], data["core_pair"], data["core_y"], seed=seed
            )
            spectral_audit[str(seed)] = audit
            for lambda_value in policy.lambda_grid:
                weights64 = _weights_for_lambda(
                    spectral, lambda_value=lambda_value, width=data["core_x"].shape[1]
                )
                weights32 = weights64.astype(np.float32)
                selected_weights32 = weights32[data["dev_pair"]]
                scores32 = np.sum(
                    data["dev_x"] * selected_weights32,
                    axis=1,
                    dtype=np.float32,
                )
                scores64 = np.einsum(
                    "ij,ij->i",
                    data["dev_x"].astype(np.float64),
                    weights64[data["dev_pair"]],
                )
                values, aggregate = _state_retained_mass(
                    scores32,
                    data["dev_mass"],
                    data["dev_offsets"],
                    area_fraction=policy.retained_area_fraction,
                )
                row = rows_by_lambda[lambda_value]
                row["retained_mass_values"].extend(values)
                row["mass_weighted_aggregates"].append(aggregate)
                row["seed_summaries"].append(
                    {
                        "seed": seed,
                        "summary": retained_mass_tail_summary(
                            values, alpha=policy.cvar_alpha
                        ),
                        "mass_weighted_aggregate": aggregate,
                    }
                )
                row["max_abs_score_fp32_vs_fp64"] = max(
                    row["max_abs_score_fp32_vs_fp64"],
                    float(np.max(np.abs(scores64 - scores32.astype(np.float64)))),
                )
        curve: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        for lambda_value in policy.lambda_grid:
            row = rows_by_lambda[lambda_value]
            summary = retained_mass_tail_summary(
                row["retained_mass_values"], alpha=policy.cvar_alpha
            )
            curve_row = {
                "lambda": lambda_value,
                "summary": summary,
                "mass_weighted_aggregate_mean_across_seeds": float(
                    np.mean(row["mass_weighted_aggregates"])
                ),
                "seed_summaries": row["seed_summaries"],
                "retained_mass_values": row["retained_mass_values"],
                "max_abs_score_fp32_vs_fp64": row["max_abs_score_fp32_vs_fp64"],
            }
            curve.append(curve_row)
            selection_rows.append(
                {
                    "lambda": lambda_value,
                    "losses": [1.0 - value for value in row["retained_mass_values"]],
                }
            )
        reference = next(row for row in curve if row["lambda"] == 0.0)
        selected = select_tail_lambda(
            selection_rows,
            mean_reference=float(reference["summary"]["shortfall_mean"]),
            alpha=policy.cvar_alpha,
            mean_tolerance=0.0,
            require_positive=True,
        )
        selected_row = next(row for row in curve if row["lambda"] == selected.lambda_value)
        output["loci"][locus] = {
            "feature_width": int(data["core_x"].shape[1]),
            "core_balanced_rows": int(data["core_x"].shape[0]),
            "train_only_dev_rows": int(data["dev_x"].shape[0]),
            "train_only_dev_states": int(data["dev_lengths"].size),
            "fit_resample_count": len(policy.fit_resample_seeds),
            "pooled_costate_seed_samples": int(
                data["dev_lengths"].size * len(policy.fit_resample_seeds)
            ),
            "chunk_custody": chunk_custody,
            "official_mp_anchor": _official_mp_anchor(
                root, locus, alpha=policy.cvar_alpha
            ),
            "curve": curve,
            "tail_selection": selected.to_dict(),
            "tail_selection_summary": selected_row["summary"],
            "mean_gate_reference": "same cached cross-fit MP lambda=0 shortfall mean",
            "spectral_audit": spectral_audit,
        }
    return output


def _fit_organ_mp(model: Any, intervals: list[Any], phis: np.ndarray) -> None:
    """Fit the explicit Moore-Penrose diagnostic for A/P/Q at lambda zero."""

    if hasattr(model, "_rows"):
        phi, target, _occupancy = model._rows(intervals, phis)
        coefficient = np.linalg.pinv(phi) @ target
        model.coef = np.ascontiguousarray(coefficient, dtype=np.float64)
        model.kappa = None
        return
    rows = []
    targets = []
    for interval in intervals:
        occupancy = phis.T @ interval.u_mean
        rows.append(np.concatenate(([1.0], interval.x0, occupancy)))
        targets.append(interval.dxdt())
    phi = np.stack(rows)
    target = np.stack(targets)
    coefficient = np.linalg.pinv(phi) @ target
    state_width = target.shape[1]
    model.a = coefficient[0].copy()
    model.C = coefficient[1 : 1 + state_width].T.copy()
    model.M = coefficient[1 + state_width :].T.copy()


def _measure_organ(policy: ControlTailReliabilityPolicy) -> dict[str, Any]:
    from tac.witness_control.aniso_perclass_lambda import (
        AnisoPriorMeanAdjoint,
        IsoPriorMeanAdjoint,
    )
    from tac.witness_control.lambda_net import (
        RidgeSolveAdjoint,
        _predict_interval,
        build_intervals,
        fit_score_composition,
        lever_features,
        read_trajectory,
    )

    cache_path = REPO / policy.organ_cache
    cached = _load_json(cache_path)
    persistence_mean = float(
        cached["tournament"]["A_ridge_solve"]["walkforward_mae_heuristic"]
    )
    trajectory = read_trajectory(REPO / policy.organ_run_dir)
    intervals = build_intervals(trajectory)
    phis = np.stack([lever_features(name) for name in trajectory.lever_names])
    class_weights32 = np.asarray(
        fit_score_composition(trajectory.verdicts).class_weights, dtype=np.float32
    )
    arms = {
        "A_ridge_solve": RidgeSolveAdjoint,
        "P_priormean_aniso": AnisoPriorMeanAdjoint,
        "Q_priormean_iso": IsoPriorMeanAdjoint,
    }
    result: dict[str, Any] = {
        "custody": {"cached_433_receipt": _custody(cache_path)},
        "trajectory": {
            "run_dir": policy.organ_run_dir,
            "verdict_count": trajectory.n_verdicts,
            "interval_count": len(intervals),
            "walkforward_fold_count": max(0, len(intervals) - 2),
            "read_only": True,
        },
        "loss": (
            "score-law-weighted walk-forward forecast error on observed control; "
            "a proxy for control regret, not counterfactual downstream action regret"
        ),
        "counterfactual_control_regret_measured": False,
        "counterfactual_blocker": (
            "cached trajectory has one realized combined control per interval and no "
            "counterfactual per-action next states"
        ),
        "persistence_reference": {
            "walkforward_mean": persistence_mean,
            "source": "cached #433 deployment-faithful walk-forward heuristic",
            "tail_quantiles_available": False,
        },
        "arms": {},
    }
    for arm_name, constructor in arms.items():
        curve: list[dict[str, Any]] = []
        for lambda_value in policy.lambda_grid:
            fold_rows: list[dict[str, Any]] = []
            for hold in range(2, len(intervals)):
                model = constructor(ridge=lambda_value)
                if lambda_value == 0.0:
                    _fit_organ_mp(model, intervals[:hold], phis)
                    solver = "numpy.linalg.pinv Moore-Penrose diagnostic"
                else:
                    model.fit(intervals[:hold], phis, seed=0)
                    solver = "imported scaled-positive-ridge solver"
                interval = intervals[hold]
                prediction64 = _predict_interval(
                    model, arm_name, interval, trajectory.lever_names
                )
                prediction32 = np.asarray(prediction64, dtype=np.float32)
                measured32 = np.asarray(interval.dxdt(), dtype=np.float32)
                error32 = abs(
                    float(
                        np.dot(
                            class_weights32,
                            prediction32[:5] - measured32[:5],
                        )
                    )
                ) * float(np.float32(interval.dep))
                fold_rows.append(
                    {
                        "fold": hold,
                        "n_train": hold,
                        "ep0": interval.ep0,
                        "ep1": interval.ep1,
                        "forecast_error": error32,
                        "max_abs_prediction_fp32_vs_fp64": float(
                            np.max(
                                np.abs(
                                    prediction64
                                    - prediction32.astype(np.float64)
                                )
                            )
                        ),
                    }
                )
            losses = [row["forecast_error"] for row in fold_rows]
            curve.append(
                {
                    "lambda": lambda_value,
                    "solver": solver,
                    "summary": loss_tail_summary(losses, alpha=policy.cvar_alpha),
                    "folds": fold_rows,
                    "losses": losses,
                }
            )
        default = next(
            row for row in curve if row["lambda"] == policy.organ_mean_reference_lambda
        )
        selection = select_tail_lambda(
            [{"lambda": row["lambda"], "losses": row["losses"]} for row in curve],
            mean_reference=float(default["summary"]["mean"]),
            alpha=policy.cvar_alpha,
            mean_tolerance=0.0,
            require_positive=True,
        )
        selected = next(row for row in curve if row["lambda"] == selection.lambda_value)
        cached_default = float(cached["tournament"][arm_name]["walkforward_mae_model"])
        if not math.isclose(
            float(default["summary"]["mean"]), cached_default, rel_tol=2.0e-6, abs_tol=2.0e-10
        ):
            raise MeasurementError(f"{arm_name} default-lambda reproduction drifted")
        selected_index = policy.lambda_grid.index(selection.lambda_value)
        bracket_closed = selected_index not in (0, len(policy.lambda_grid) - 1)
        result["arms"][arm_name] = {
            "curve": curve,
            "tail_selection": selection.to_dict(),
            "tail_selection_summary": selected["summary"],
            "current_default_lambda": policy.organ_mean_reference_lambda,
            "current_default_summary": default["summary"],
            "cached_default_mean_reproduced": cached_default,
            "bracket_closed": bracket_closed,
            "selected_vs_default_mean_relative": (
                float(selected["summary"]["mean"]) / float(default["summary"]["mean"]) - 1.0
            ),
            "selected_vs_default_cvar_relative": (
                float(selected["summary"]["cvar"]) / float(default["summary"]["cvar"]) - 1.0
            ),
            "selected_mean_vs_persistence_relative": (
                float(selected["summary"]["mean"]) / persistence_mean - 1.0
            ),
        }
    return result


def _interpolator_inventory() -> list[dict[str, Any]]:
    return [
        {
            "surface": "PRE-SE / Round-4/5 RankRLS",
            "path": "src/tac/scorer_surrogate/replace_round4_support_ranking.py",
            "control_role": "costate support selector; fallback/controller evidence",
            "tail_status": "MEASURED here on cached train-only dev; official n120 ridge owed",
        },
        {
            "surface": "Costate organ A and reweighted G/H/I/J/K/N/O arms",
            "path": "src/tac/witness_control/lambda_net.py",
            "control_role": "flow lens driving controller decisions",
            "tail_status": "A MEASURED here; wrappers inherit the same required selection law",
        },
        {
            "surface": "Costate organ prior-mean P/Q/R/S arms",
            "path": "src/tac/witness_control/aniso_perclass_lambda.py",
            "control_role": "per-class response/control forecast",
            "tail_status": "P/Q MEASURED here; R/S owed on record accrual",
        },
        {
            "surface": "LinearNCDE sliding-window ridge",
            "path": "src/tac/witness_control/ncde_trajectory.py",
            "control_role": "trajectory forecast and equilibrium/rollout",
            "tail_status": "GAP: fixed 1e-3 and mean-style diagnostics; must adopt CVaR+mean gate",
        },
        {
            "surface": "Prototype-router local weighted ridge",
            "path": "src/tac/witness_control/prototype_router.py",
            "control_role": "state-dependent response router",
            "tail_status": "GAP: fixed 1e-2 per prototype; report regime tail and support count",
        },
        {
            "surface": "Transient Forge ridge response",
            "path": "src/tac/witness_control/transient_forge.py",
            "control_role": "synthetic transient control disambiguator",
            "tail_status": "research-only; tail rule required before real control consumption",
        },
        {
            "surface": "Rate-law ladder cross-fitted ridge",
            "path": "tools/measure_rate_law_ladder_owed.py",
            "control_role": "receiver/rate planning predictor",
            "tail_status": "GAP: selects mean validation alpha; add byte-regret CVaR by pair block",
        },
        {
            "surface": "OOF scorer-response ridge",
            "path": "tools/fit_scorer_response_oof_predictions.py",
            "control_role": "candidate/spend triage",
            "tail_status": "GAP: report OOF p95/p99 sign/regret before planner consumption",
        },
        {
            "surface": "Jacobian Moore-Penrose terminal inverse",
            "path": "src/tac/research/jacobian_optimal.py",
            "control_role": "terminal inverse research surface",
            "tail_status": "MP-only; require singular-value stress/tail receipt if promoted",
        },
        {
            "surface": "Round-3 RFF costate-mass ridge",
            "path": "src/tac/witness_dsl/replace_round3_fidelity_wall_policy.py",
            "control_role": "fixed-replay surrogate rung, not live controller",
            "tail_status": "settled formulation evidence; do not reopen absent req-R",
        },
    ]


def measure(policy: ControlTailReliabilityPolicy) -> dict[str, Any]:
    contract = policy.compile_measurement_contract()
    source_paths = [
        REPO / "src/tac/canonical_equations/control_interpolator_tail_reliability_20260713.py",
        REPO / "src/tac/witness_dsl/control_tail_reliability_policy_20260713.py",
        Path(__file__).resolve(),
        REPO / "src/tac/scorer_surrogate/replace_round4_support_ranking.py",
        REPO / "src/tac/witness_control/lambda_net.py",
        REPO / "src/tac/witness_control/aniso_perclass_lambda.py",
    ]
    return {
        "schema": SCHEMA,
        "completed_at_utc": _utc_now(),
        "lane_id": LANE_ID,
        "mode": "MEANS; research_only=true; $0 cached local",
        "pointer_delta": "NONE",
        "score_claim": False,
        "promotion_eligible": False,
        "authority": (
            "[macOS-CPU advisory; NumPy-fp32 decision; float64 eigensolve optimization evidence]"
        ),
        "containment": {
            "training": False,
            "scorer_calls": False,
            "paid_or_remote_dispatch": False,
            "live_run_mutation": False,
            "archive_mutation": False,
            "pointer_mutation": False,
        },
        "git_head_at_measurement": _git_head(),
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "compiled_policy": contract,
        "source_custody": [_custody(path) for path in source_paths],
        "pre_se": _measure_pre_se(policy),
        "organ_433": _measure_organ(policy),
        "finite_sample_bound": {
            "derived": True,
            "numeric_bound_evaluated": False,
            "closes_vs_measured": False,
            "blockers": [
                "no cached per-state residual innovation vectors for covariance estimation",
                "official n120 heldout raw features and exact mass arrays were not preserved",
                "row-bootstrap refits do not establish independent whitened sub-Gaussian innovations",
            ],
            "scope": (
                "conditional fixed-design Gaussian/Hanson-Wright law is canonicalized; "
                "distribution-free numeric closure is not claimed"
            ),
        },
        "vrghal_witness_sgd": {
            "quantitative_convergence_residual_measured": False,
            "verdict_scope": (
                "FORMULATION x FROZEN-STAGE/FROZEN-REPLAY/FIXED-LOSS WITNESS-SGD WINDOW"
            ),
            "verdict": "CONDITIONAL-NOT-THEOREM-ADMITTED",
            "req_R": (
                "fixed update map; nonexpansive/contractive trust region; measured gamma, sigma, "
                "kappa_E, oracle unbiasedness, and native-norm residual trace"
            ),
        },
        "interpolator_inventory": _interpolator_inventory(),
        "triality": {
            "equations": [
                "control_interpolator_tail_cvar_mean_gate_v1",
                "fixed_design_correlated_gaussian_ridge_tail_v1",
            ],
            "dsl": "ControlTailReliabilityPolicy (default off; cached read only)",
            "dag": ".omx/research/quant_tail_reliability_DAG_FEED_20260713.md",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    policy = ControlTailReliabilityPolicy()
    payload = measure(policy)
    output = args.output if args.output.is_absolute() else REPO / args.output
    _atomic_json(output, payload)
    final = _custody(output)
    print(json.dumps({"receipt": final, "pointer_delta": "NONE"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
