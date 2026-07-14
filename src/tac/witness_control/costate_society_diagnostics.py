# SPDX-License-Identifier: MIT
"""Behavioural differentiation and local stability for the costate organ society.

The machinery is warm-started from Huot, Kaisers, and Lapata (2026): actors are
compared by held-out behaviour, Hierarchic Social Entropy integrates clustering
entropy over every taxonomic threshold, and router robustness is assignment
agreement under meaning-preserving input variants.  Here actors are organ
mechanisms, prompts are walk-forward folds, and variants are exact one-ULP
surface forms of the same NumPy-fp32 regime-state estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tac.witness_control.costate_requential_curriculum import (
    walkforward_requential_backtest,
)
from tac.witness_control.costate_warmstart_cluster import walkforward_backtest
from tac.witness_control.lambda_net import (
    CampaignTrajectory,
    build_intervals,
    fit_score_composition,
    lever_features,
)
from tac.witness_control.regime_dispatch import (
    PERSISTENCE,
    _model_surprise,
    _standalone_wf,
    backtest_dispatch,
    dispatch_decision,
)
from tac.witness_control.router_stability import certify_fp32_gate

SOCIETY_RECEIPT_SCHEMA = "costate_mechanism_society_diagnostic.v1"
DEFAULT_MECHANISMS = (
    PERSISTENCE,
    "A_ridge_solve",
    "G_ridge_scorerprior",
    "H_smoothed_argmax",
    "J_adv_boundary",
    "P_priormean_aniso",
    "Q_priormean_iso",
    "R_priormean_c10k_scorelaw",
    "T_gp_costate_posterior",
)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def _partition_entropy(uf: _UnionFind, n: int) -> float:
    counts: dict[int, int] = {}
    for i in range(n):
        root = uf.find(i)
        counts[root] = counts.get(root, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def cosine_distance_matrix(behaviour: np.ndarray) -> np.ndarray:
    b = np.asarray(behaviour, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] < 1 or not np.all(np.isfinite(b)):
        raise ValueError("behaviour must be a finite actor-by-fold matrix")
    norm = np.linalg.norm(b, axis=1)
    out = np.zeros((len(b), len(b)), dtype=np.float64)
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            if norm[i] == 0.0 or norm[j] == 0.0:
                distance = 1.0
            else:
                distance = 1.0 - float(b[i] @ b[j] / (norm[i] * norm[j]))
                distance = min(max(distance, 0.0), 1.0)
            out[i, j] = out[j, i] = distance
    return out


def hierarchic_social_entropy(behaviour: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Exact single-linkage HSE integral for nonnegative behaviour vectors."""
    distance = cosine_distance_matrix(behaviour)
    n = len(distance)
    if n == 1:
        return 0.0, 0.0, distance
    uf = _UnionFind(n)
    pairs = [(float(distance[i, j]), i, j)
             for i in range(n) for j in range(i + 1, n)]
    # h=0 merges exact behavioural duplicates before the first interval is integrated.
    for d, i, j in pairs:
        if d <= np.finfo(np.float64).eps:
            uf.union(i, j)
    entropy = _partition_entropy(uf, n)
    previous = 0.0
    area = 0.0
    for threshold in sorted({d for d, _, _ in pairs if d > np.finfo(np.float64).eps}):
        area += entropy * (threshold - previous)
        for d, i, j in pairs:
            if d <= threshold:
                uf.union(i, j)
        entropy = _partition_entropy(uf, n)
        previous = threshold
    normalized = area / math.log2(n)
    return float(area), float(normalized), distance


def errors_to_behaviour(errors: np.ndarray, persistence_errors: np.ndarray) -> np.ndarray:
    """Map same-unit fold errors to bounded skill profiles without cross-fold tuning."""
    e = np.asarray(errors, dtype=np.float64)
    p = np.asarray(persistence_errors, dtype=np.float64)
    if e.ndim != 2 or p.shape != (e.shape[1],):
        raise ValueError("errors must be actor-by-fold and persistence must be fold-length")
    scale = np.maximum(p, np.finfo(np.float64).tiny)
    return np.exp(-e / scale[None, :])


@dataclass(frozen=True)
class SocietyDiagnostic:
    mechanisms: tuple[str, ...]
    errors: dict[str, list[float]]
    behaviour: dict[str, list[float]]
    hse: float
    normalized_hse: float
    pairwise_cosine_distance: dict[str, dict[str, float]]
    greedy_coreset_curve: tuple[dict, ...]
    positive_gain_coreset: tuple[str, ...]
    route_active_coreset: tuple[str, ...]
    route_active_coreset_hse: float
    pareto_coreset: tuple[str, ...]
    routed_system_performance: dict
    representation_sensitivity: dict[str, dict]
    blocked_mechanisms: dict[str, str]

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "mechanisms": list(self.mechanisms),
            "greedy_coreset_curve": [dict(row) for row in self.greedy_coreset_curve],
            "positive_gain_coreset": list(self.positive_gain_coreset),
            "route_active_coreset": list(self.route_active_coreset),
            "pareto_coreset": list(self.pareto_coreset),
        }


def _greedy_coreset(names: tuple[str, ...], behaviour: np.ndarray,
                     mean_errors: np.ndarray) -> tuple[tuple[dict, ...], tuple[str, ...]]:
    start = int(np.argmin(mean_errors))
    selected = [start]
    remaining = set(range(len(names))) - {start}
    curve = [{"size": 1, "added": names[start], "hse": 0.0,
              "normalized_hse": 0.0, "marginal_hse": 0.0}]
    previous = 0.0
    best_prefix = [start]
    best_hse = 0.0
    while remaining:
        candidates = []
        for idx in remaining:
            subset = [*selected, idx]
            hse, normalized, _ = hierarchic_social_entropy(behaviour[subset])
            candidates.append((-hse, mean_errors[idx], names[idx], idx, normalized))
        neg_hse, _, _, chosen, normalized = min(candidates)
        hse = -neg_hse
        gain = hse - previous
        selected.append(chosen)
        remaining.remove(chosen)
        curve.append({"size": len(selected), "added": names[chosen], "hse": hse,
                      "normalized_hse": normalized, "marginal_hse": gain})
        if hse > best_hse + np.finfo(np.float64).eps:
            best_hse = hse
            best_prefix = selected.copy()
        previous = hse
    return tuple(curve), tuple(names[i] for i in best_prefix)


def diagnose_mechanism_society(
    traj: CampaignTrajectory,
    *,
    mechanisms: tuple[str, ...] = DEFAULT_MECHANISMS,
    seed: int = 0,
) -> SocietyDiagnostic:
    intervals = build_intervals(traj)
    comp = fit_score_composition(traj.verdicts)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    errors: dict[str, list[float]] = {}
    blocked: dict[str, str] = {
        "distilled_surrogate": (
            "NO_REAL_WF_PREDICTION_ROWS; quotient-VJP surrogate remains sister-owned")}
    for mechanism in mechanisms:
        try:
            errors[mechanism] = [float(v) for v in _standalone_wf(
                mechanism, intervals, comp, traj.lever_names, phis, seed)]
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            blocked[mechanism] = f"{type(exc).__name__}: {exc}"
    u = walkforward_backtest(traj)
    errors["U_hierarchical_physics_residual"] = [float(f.error) for f in u.folds]
    req = walkforward_requential_backtest(traj, strategy="disagreement")
    errors["R_requential_disagreement_curriculum"] = [float(f.error) for f in req.folds]
    dispatch = backtest_dispatch(traj, seed=seed)
    dispatch_errors = [float(r["dispatcher_err"]) for r in dispatch.fold_rows]

    names = tuple(sorted(errors))
    matrix = np.asarray([errors[name] for name in names], dtype=np.float64)
    persistence = np.asarray(errors[PERSISTENCE], dtype=np.float64)
    behaviour = errors_to_behaviour(matrix, persistence)
    hse, normalized, distance = hierarchic_social_entropy(behaviour)
    pairwise = {
        a: {b: float(distance[i, j]) for j, b in enumerate(names)}
        for i, a in enumerate(names)
    }
    curve, coreset = _greedy_coreset(names, behaviour, matrix.mean(axis=1))
    route_active = (PERSISTENCE, "T_gp_costate_posterior")
    route_indices = [names.index(name) for name in route_active]
    route_hse, _, _ = hierarchic_social_entropy(behaviour[route_indices])
    pareto = []
    for i, name in enumerate(names):
        dominated = any(
            j != i and np.all(matrix[j] <= matrix[i]) and np.any(matrix[j] < matrix[i])
            for j in range(len(names))
        )
        if not dominated:
            pareto.append(name)
    ratio = matrix / np.maximum(persistence, np.finfo(np.float64).tiny)[None, :]
    reciprocal = 1.0 / (1.0 + ratio)
    col_min, col_max = matrix.min(axis=0), matrix.max(axis=0)
    span = col_max - col_min
    minmax = np.where(
        span[None, :] > np.finfo(np.float64).eps,
        (col_max[None, :] - matrix) / np.maximum(span[None, :], np.finfo(np.float64).tiny),
        1.0,
    )
    sensitivity = {}
    key_pairs = (
        (PERSISTENCE, "T_gp_costate_posterior"),
        ("A_ridge_solve", "G_ridge_scorerprior"),
        ("U_hierarchical_physics_residual", "R_requential_disagreement_curriculum"),
    )
    for label, representation in (
        ("exp_error_over_persistence", behaviour),
        ("reciprocal_error_over_persistence", reciprocal),
        ("fold_minmax_skill", minmax),
    ):
        r_hse, r_normalized, r_distance = hierarchic_social_entropy(representation)
        sensitivity[label] = {
            "hse": r_hse,
            "normalized_hse": r_normalized,
            "key_distances": {
                f"{a}__{b}": float(r_distance[names.index(a), names.index(b)])
                for a, b in key_pairs
            },
        }
    return SocietyDiagnostic(
        mechanisms=names,
        errors={name: errors[name] for name in names},
        behaviour={name: [float(v) for v in behaviour[i]] for i, name in enumerate(names)},
        hse=hse, normalized_hse=normalized,
        pairwise_cosine_distance=pairwise,
        greedy_coreset_curve=curve, positive_gain_coreset=coreset,
        route_active_coreset=route_active,
        route_active_coreset_hse=route_hse,
        pareto_coreset=tuple(pareto),
        routed_system_performance={
            "name": "regime_dispatch_436",
            "fold_errors": dispatch_errors,
            "mean_wf_mae": float(np.mean(dispatch_errors)),
            "actor_society_member": False,
        },
        representation_sensitivity=sensitivity,
        blocked_mechanisms=blocked,
    )


@dataclass(frozen=True)
class RouterPerturbationDiagnostic:
    n_folds: int
    perturbations_per_fold: int
    assignment_robustness: float
    stable_fold_count: int
    unstable_fold_count: int
    fold_rows: tuple[dict, ...]
    perturbation_scope: str = "ONE_ULP_NUMPY_FP32_REGIME_STATE_ONLY"

    def to_dict(self) -> dict:
        return {**self.__dict__, "fold_rows": [dict(r) for r in self.fold_rows]}


def _ulp_variants(recent: float, median: float) -> tuple[tuple[str, float, float], ...]:
    r, m = np.float32(recent), np.float32(median)
    r_lo, r_hi = np.nextafter(r, np.float32(-np.inf)), np.nextafter(r, np.float32(np.inf))
    m_lo, m_hi = np.nextafter(m, np.float32(-np.inf)), np.nextafter(m, np.float32(np.inf))
    return (
        ("recent_minus_1ulp", float(r_lo), float(m)),
        ("recent_plus_1ulp", float(r_hi), float(m)),
        ("median_minus_1ulp", float(r), float(m_lo)),
        ("median_plus_1ulp", float(r), float(m_hi)),
        ("toward_plateau_joint", float(r_lo), float(m_hi)),
        ("toward_transient_joint", float(r_hi), float(m_lo)),
    )


def diagnose_router_ulp_robustness(
    traj: CampaignTrajectory,
    *,
    seed: int = 0,
) -> RouterPerturbationDiagnostic:
    intervals = build_intervals(traj)
    comp = fit_score_composition(traj.verdicts)
    rows = []
    total_same = 0
    total = 0
    for hold in range(2, len(intervals)):
        past = intervals[:hold]
        original = dispatch_decision(
            past, comp, traj.lever_names, seed=seed, meta_lambda_guard=True)
        cert = original.classification.gate_certificate
        assert cert is not None and cert.recent_slope_mag is not None
        assert cert.median_slope_mag is not None
        _, surprise_ratio = _model_surprise(past, comp, traj.lever_names, seed=seed)
        variants = []
        for label, recent, median in _ulp_variants(
                cert.recent_slope_mag, cert.median_slope_mag):
            variant = certify_fp32_gate(
                recent_slope_mag=recent,
                median_slope_mag=median,
                n_past_intervals=len(past),
                surprise_ratio=surprise_ratio,
                meta_lambda_guard=True,
                policy={
                    "transient": "T_gp_costate_posterior",
                    "plateau": PERSISTENCE,
                    "uncertain": PERSISTENCE,
                },
            )
            same = variant.selected_tool == original.tool
            total_same += int(same)
            total += 1
            variants.append({"variant": label, "tool": variant.selected_tool,
                             "regime": variant.selected_regime, "same_tool": same})
        rho = sum(v["same_tool"] for v in variants) / len(variants)
        rows.append({
            "hold": hold,
            "ep1": float(intervals[hold].ep1),
            "original_regime": original.classification.regime,
            "original_tool": original.tool,
            "slope_margin_ulps": cert.slope_margin_ulps,
            "rho": rho,
            "variants": variants,
        })
    stable = sum(row["rho"] == 1.0 for row in rows)
    return RouterPerturbationDiagnostic(
        n_folds=len(rows), perturbations_per_fold=6,
        assignment_robustness=total_same / max(total, 1),
        stable_fold_count=stable, unstable_fold_count=len(rows) - stable,
        fold_rows=tuple(rows),
    )


__all__ = [
    "DEFAULT_MECHANISMS", "SOCIETY_RECEIPT_SCHEMA", "RouterPerturbationDiagnostic",
    "SocietyDiagnostic", "cosine_distance_matrix", "diagnose_mechanism_society",
    "diagnose_router_ulp_robustness", "errors_to_behaviour", "hierarchic_social_entropy",
]
