# SPDX-License-Identifier: MIT
"""Meta-Lagrangian search engine for extreme automated Shannon-floor optimization.

Combines the predictor + distortion proxy + predispatch sanity gate into a
single search loop that walks a candidate generator, scores each candidate
via a Boyd-style Lagrangian, and returns a ranked queue suitable for paid-eval
dispatch. Every candidate that escalates to GPU spend goes through:

    1. Distortion proxy ([distortion-proxy:local], CPU only — no MPS auth eval)
    2. Score-band predictor (refuses outside calibrated regime)
    3. Lagrangian aggregation (penalizes constraint violations)
    4. Predispatch sanity gate (5 hardened checks)

Lagrangian formulation (Boyd-style, used to RANK candidates not select winners):

    L(x; λ) = predicted_score(x)
            + λ_R · max(0, rate_unscaled(x) − R_max)
            + λ_P · max(0, pose(x)            − P_max)
            + λ_S · max(0, seg(x)             − S_max)

Lower L = better. Candidates with refused predictions, failed sanity gates, or
positive constraint violations sort to the bottom regardless of nominal score.

CONTRACT: this module makes NO claim about an actual score. Outputs are
``ProxyEvaluation`` records tagged ``[distortion-proxy:local]`` per CLAUDE.md.
The contest-faithful score MUST come from upstream/evaluate.py on the EXACT
archive bytes, validated by the council/predictor pipeline.

Wiring (council Q3 + Q4 prescriptions):
  - tac.predictor.score_band.predict_score_band     — supplies refusal modes
  - experiments.distortion_proxy_local.make_distortion_proxy — supplies (pose, seg)
  - tools/predispatch_sanity.py predispatch_sanity   — final paid-spend gate
"""
from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.predictor.score_band import (  # noqa: E402
    POSE_COEFFICIENT_SQRT_INNER,
    PR106_TOTAL_RATE_DENOM,
    RATE_COEFFICIENT,
    SEG_COEFFICIENT,
    CalibrationAnchor,
    ScoreBand,
    predict_score_band,
)

# ── Contest score reproduction (matches upstream/evaluate.py) ─────────────


def contest_score(avg_pose_dist: float, avg_seg_dist: float, archive_bytes: int) -> float:
    """Compute the contest score `100*seg + sqrt(10*pose) + 25*rate`.

    All coefficients are ``[contest-defined]`` (re-exported from
    ``tac.predictor.score_band`` so PCC10 sees them in their canonical
    location, not duplicated here).
    """
    rate_unscaled = archive_bytes / PR106_TOTAL_RATE_DENOM
    return (
        SEG_COEFFICIENT * avg_seg_dist
        + math.sqrt(POSE_COEFFICIENT_SQRT_INNER * max(avg_pose_dist, 0.0))
        + RATE_COEFFICIENT * rate_unscaled
    )


# ── Lagrangian wiring ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LagrangianConstraints:
    """Per-component upper bounds for the Lagrangian penalty terms.

    Defaults are chosen so a candidate matching PR106 baseline lies on the
    feasible boundary. Tighten constraints as the frontier advances.
    """
    rate_unscaled_max: float = 0.005  # [calibration:.omx/calibration/anchors_apogee_intN.json] PR106 rate
    pose_dist_max: float = 1e-4       # [heuristic:matches A++ contest-T4 lossless class (PR106 3.4e-5 + 3x slack)]
    seg_dist_max: float = 1e-3        # [heuristic:matches A++ contest-T4 lossless class (PR106 6.7e-4 + 50% slack)]
    lambda_rate: float = 1.0          # [heuristic:Boyd ADMM-style equal-weight init; tuneable per search]
    lambda_pose: float = 1.0          # [heuristic:Boyd ADMM-style equal-weight init]
    lambda_seg: float = 1.0           # [heuristic:Boyd ADMM-style equal-weight init]


@dataclass
class CandidateEvaluation:
    """A single candidate's full evaluation through the search pipeline."""
    candidate_id: str
    archive_bytes: int
    rel_err_pct: float
    n_layers: int
    lane_class: str
    archive_path: Path | None = None

    # Proxy outputs (closed-form, [distortion-proxy:local])
    proxy_pose: float = 0.0
    proxy_seg: float = 0.0
    proxy_rate_unscaled: float = 0.0
    proxy_score: float = float("inf")

    # Predictor outputs
    band_low: float = 0.0
    band_high: float = float("inf")
    band_refused: bool = True
    band_refusal_reason: str = "not-yet-evaluated"
    band_method: str = "none"

    # Lagrangian
    lagrangian: float = float("inf")
    rate_violation: float = 0.0
    pose_violation: float = 0.0
    seg_violation: float = 0.0

    # Sanity gate
    sanity_passed: bool = False
    sanity_failures: list[str] = field(default_factory=list)

    # Final ranking key (lower = better; refused/failed sort to the bottom)
    rank_key: float = float("inf")
    eligible_for_dispatch: bool = False


@dataclass(frozen=True)
class CmaEsSearchBounds:
    """Bounded continuous search box for deterministic CMA-ES candidate prep.

    The optimizer runs in normalized ``[0, 1]^3`` coordinates and maps back to
    the strict candidate schema consumed by :meth:`evaluate_candidate`. It is a
    CPU planning generator only; returned candidates still need archives,
    preflight, lane claims, and exact CUDA evidence.
    """

    archive_bytes: tuple[int, int]
    rel_err_pct: tuple[float, float]
    n_layers: tuple[int, int]


@dataclass(frozen=True)
class CmaEsCandidateSuggestion:
    """One deterministic CMA-ES proposal plus its local objective evidence."""

    candidate: dict[str, Any]
    objective: float
    generation: int
    unit_vector: tuple[float, float, float]
    evaluation: CandidateEvaluation
    score_claim: bool = False
    ready_for_exact_eval_dispatch: bool = False
    dispatch_blockers: tuple[str, ...] = (
        "cma_es_candidate_generator_is_cpu_planning_only",
        "candidate_archive_missing",
        "requires_static_preflight",
        "requires_lane_dispatch_claim",
        "requires_exact_cuda_auth_eval",
    )


DistortionProxy = Callable[[int, float, int], tuple[float, float]]


# ── Search engine ─────────────────────────────────────────────────────────


class MetaLagrangianSearch:
    """Boyd-style multi-constraint search over codec parameter candidates.

    Usage::

        from experiments.distortion_proxy_local import make_distortion_proxy
        proxy = make_distortion_proxy()
        anchors = load_calibration_anchors(Path(".omx/calibration/anchors_apogee_intN.json"))
        search = MetaLagrangianSearch(
            calibration_anchors=anchors,
            distortion_proxy=proxy,
            constraints=LagrangianConstraints(),
        )
        ranked = search.evaluate_all(candidates)
        top_k = search.top_k(ranked, k=3)
    """

    def __init__(
        self,
        calibration_anchors: list[CalibrationAnchor],
        distortion_proxy: DistortionProxy,
        constraints: LagrangianConstraints | None = None,
        sanity_gate: Callable[..., object] | None = None,
    ) -> None:
        self.anchors = calibration_anchors
        self.proxy = distortion_proxy
        self.constraints = constraints or LagrangianConstraints()
        self._sanity_gate = sanity_gate or _default_sanity_gate

    # ── Per-candidate evaluation ──────────────────────────────────────────

    def evaluate_candidate(
        self,
        candidate_id: str,
        archive_bytes: int,
        rel_err_pct: float,
        n_layers: int,
        lane_class: str,
        archive_path: Path | None = None,
        sanity_predicted_band: tuple[float, float] | None = None,
    ) -> CandidateEvaluation:
        ev = CandidateEvaluation(
            candidate_id=candidate_id,
            archive_bytes=archive_bytes,
            rel_err_pct=rel_err_pct,
            n_layers=n_layers,
            lane_class=lane_class,
            archive_path=archive_path,
        )

        # 1. Distortion proxy (CPU-only closed-form; tag [distortion-proxy:local])
        ev.proxy_pose, ev.proxy_seg = self.proxy(archive_bytes, rel_err_pct, n_layers)
        ev.proxy_rate_unscaled = archive_bytes / PR106_TOTAL_RATE_DENOM
        ev.proxy_score = contest_score(ev.proxy_pose, ev.proxy_seg, archive_bytes)

        # 2. Score-band predictor (carries refusal modes from council Q1)
        band: ScoreBand = predict_score_band(
            archive_bytes=archive_bytes,
            rel_err_pct_per_weight=rel_err_pct,
            n_quantized_layers=n_layers,
            calibration_anchors=self.anchors,
            distortion_proxy=self.proxy,
        )
        ev.band_low = band.low
        ev.band_high = band.high
        ev.band_refused = band.refused
        ev.band_refusal_reason = band.refusal_reason
        ev.band_method = band.prediction_method

        # 3. Lagrangian penalty aggregation (Boyd-style)
        c = self.constraints
        ev.rate_violation = max(0.0, ev.proxy_rate_unscaled - c.rate_unscaled_max)
        ev.pose_violation = max(0.0, ev.proxy_pose - c.pose_dist_max)
        ev.seg_violation = max(0.0, ev.proxy_seg - c.seg_dist_max)
        ev.lagrangian = (
            ev.proxy_score
            + c.lambda_rate * ev.rate_violation
            + c.lambda_pose * ev.pose_violation
            + c.lambda_seg * ev.seg_violation
        )

        # 4. Predispatch sanity gate (5 hardened checks)
        if archive_path is not None:
            try:
                sanity_low, sanity_high = sanity_predicted_band or (band.low, band.high)
                sanity_result = self._sanity_gate(
                    archive_path=archive_path,
                    predicted_low=sanity_low,
                    predicted_high=sanity_high,
                    rel_err_pct=rel_err_pct,
                    lane_class=lane_class,
                    distortion_proxy_was_run=True,  # the proxy IS this search engine's first stage
                )
                ev.sanity_passed = sanity_result.passed
                ev.sanity_failures = list(sanity_result.refusal_reasons)
            except Exception as exc:
                ev.sanity_passed = False
                ev.sanity_failures = [f"sanity_gate_error: {type(exc).__name__}: {exc}"]
        else:
            # No archive on disk: sanity skipped (deferred until producer runs)
            ev.sanity_passed = False
            ev.sanity_failures = ["sanity_skipped: archive_path not yet produced"]

        # 5. Ranking key — refused / failed candidates always sort below feasible.
        # Use Lagrangian as the primary sort, +inf penalty for refused/failed.
        if ev.band_refused or not ev.sanity_passed:
            ev.rank_key = float("inf")  # bottom of the queue
            ev.eligible_for_dispatch = False
        else:
            ev.rank_key = ev.lagrangian
            ev.eligible_for_dispatch = True

        return ev

    # ── Batch + ranking ───────────────────────────────────────────────────

    def evaluate_all(
        self,
        candidates: Iterable[dict],
    ) -> list[CandidateEvaluation]:
        """Evaluate a sequence of candidate dicts.

        Each dict must carry: candidate_id, archive_bytes, rel_err_pct,
        n_layers, lane_class. Optional: archive_path.
        """
        return [self.evaluate_candidate(**c) for c in candidates]

    def suggest_cma_es_candidates(
        self,
        *,
        lane_class: str,
        bounds: CmaEsSearchBounds,
        candidate_id_prefix: str = "cma_es",
        generations: int = 4,
        population_size: int = 8,
        sigma: float = 0.25,
        seed: int = 0,
    ) -> list[CmaEsCandidateSuggestion]:
        """Generate deterministic planning candidates with CMA-ES.

        This is a proposal generator, not a dispatch gate. It optimizes the
        local Lagrangian over ``archive_bytes``, ``rel_err_pct``, and
        ``n_layers`` using the same proxy/predictor pipeline as normal
        candidates, but it never supplies an archive path and therefore cannot
        become dispatch-eligible by itself.
        """

        _validate_cma_es_bounds(bounds)
        if generations <= 0:
            raise ValueError("generations must be positive")
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if not math.isfinite(float(sigma)) or sigma <= 0:
            raise ValueError("sigma must be a positive finite float")

        try:
            import numpy as np
            from cmaes import CMA
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise RuntimeError(
                "cmaes is required for CMA-ES candidate generation; install "
                "the project dependencies or use --candidates-json/--auto-sweep-bits"
            ) from exc

        optimizer = CMA(
            mean=np.array([0.5, 0.5, 0.5], dtype=np.float64),
            sigma=float(sigma),
            bounds=np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
            seed=int(seed),
            population_size=int(population_size),
        )
        best_by_key: dict[tuple[int, float, int], CmaEsCandidateSuggestion] = {}
        for generation in range(int(generations)):
            solutions = []
            for member in range(int(population_size)):
                unit = optimizer.ask()
                clipped = np.clip(np.asarray(unit, dtype=np.float64), 0.0, 1.0)
                candidate = _candidate_from_unit_vector(
                    clipped,
                    bounds=bounds,
                    lane_class=lane_class,
                    candidate_id=(
                        f"{candidate_id_prefix}_g{generation:02d}_m{member:02d}"
                    ),
                )
                evaluation = self.evaluate_candidate(**candidate)
                objective = _cma_es_objective(evaluation)
                solutions.append((unit, objective))
                key = (
                    int(candidate["archive_bytes"]),
                    round(float(candidate["rel_err_pct"]), 12),
                    int(candidate["n_layers"]),
                )
                suggestion = CmaEsCandidateSuggestion(
                    candidate=candidate,
                    objective=objective,
                    generation=generation,
                    unit_vector=tuple(float(value) for value in clipped.tolist()),
                    evaluation=evaluation,
                )
                previous = best_by_key.get(key)
                if previous is None or suggestion.objective < previous.objective:
                    best_by_key[key] = suggestion
            optimizer.tell(solutions)
        return sorted(
            best_by_key.values(),
            key=lambda item: (
                item.objective,
                int(item.candidate["archive_bytes"]),
                float(item.candidate["rel_err_pct"]),
                int(item.candidate["n_layers"]),
                str(item.candidate["candidate_id"]),
            ),
        )

    @staticmethod
    def top_k(evaluations: list[CandidateEvaluation], k: int = 3) -> list[CandidateEvaluation]:
        """Return the top-k DISPATCH-ELIGIBLE evaluations sorted by rank_key."""
        eligible = [e for e in evaluations if e.eligible_for_dispatch]
        eligible.sort(key=lambda e: e.rank_key)
        return eligible[:k]


# ── Default sanity-gate adapter ───────────────────────────────────────────


def _default_sanity_gate(**kwargs):
    """Lazy-load tools/predispatch_sanity.py and call its predispatch_sanity().

    Bug #3 (root-cause fix 2026-05-05): RAISES FileNotFoundError when the
    helper script is missing. The previous behavior — silently returning
    `passed=True` via SimpleNamespace — was the FORBIDDEN comment-only-contract
    pattern (CLAUDE.md): a missing dependency MUST NOT silently let dispatch
    proceed. Tests that need to bypass the helper must inject a custom
    `sanity_gate` callable into `MetaLagrangianSearch`, not rely on graceful
    degradation.
    """
    helper = REPO_ROOT / "tools" / "predispatch_sanity.py"
    if not helper.is_file():
        # Bug #3 fix: hard-fail rather than graceful pass. A silent skip here
        # is exactly the silent-success-on-missing-dependency pattern that
        # CLAUDE.md non-negotiable forbids. Callers needing tests/sandbox
        # operation must inject a `sanity_gate` callable explicitly.
        raise FileNotFoundError(
            f"predispatch_sanity helper not found at {helper}. "
            "MetaLagrangianSearch refuses to silently bypass the sanity gate; "
            "either restore tools/predispatch_sanity.py or pass a custom "
            "`sanity_gate=` callable to the search constructor."
        )
    spec = importlib.util.spec_from_file_location("_pact_predispatch_sanity_meta", helper)
    if spec is None or spec.loader is None:
        from types import SimpleNamespace
        return SimpleNamespace(passed=False, refusal_reasons=["cannot import predispatch_sanity"], gates=[])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.predispatch_sanity(**{k: v for k, v in kwargs.items() if k != "self"})


def _validate_cma_es_bounds(bounds: CmaEsSearchBounds) -> None:
    for name, pair in (
        ("archive_bytes", bounds.archive_bytes),
        ("rel_err_pct", bounds.rel_err_pct),
        ("n_layers", bounds.n_layers),
    ):
        if len(pair) != 2:
            raise ValueError(f"{name} bounds must have exactly two values")
        lo, hi = pair
        if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
            raise ValueError(f"{name} bounds must be finite")
        if hi < lo:
            raise ValueError(f"{name} upper bound must be >= lower bound")
    if bounds.archive_bytes[0] <= 0 or bounds.n_layers[0] <= 0:
        raise ValueError("archive_bytes and n_layers lower bounds must be positive")


def _scale_unit(value: float, *, lo: float, hi: float) -> float:
    return float(lo) + float(value) * (float(hi) - float(lo))


def _candidate_from_unit_vector(
    unit: Any,
    *,
    bounds: CmaEsSearchBounds,
    lane_class: str,
    candidate_id: str,
) -> dict[str, Any]:
    archive_bytes = round(
        _scale_unit(
            unit[0],
            lo=bounds.archive_bytes[0],
            hi=bounds.archive_bytes[1],
        )
    )
    rel_err_pct = _scale_unit(unit[1], lo=bounds.rel_err_pct[0], hi=bounds.rel_err_pct[1])
    n_layers = round(_scale_unit(unit[2], lo=bounds.n_layers[0], hi=bounds.n_layers[1]))
    n_layers = max(int(bounds.n_layers[0]), min(int(bounds.n_layers[1]), n_layers))
    return {
        "candidate_id": candidate_id,
        "archive_bytes": archive_bytes,
        "rel_err_pct": float(rel_err_pct),
        "n_layers": n_layers,
        "lane_class": lane_class,
    }


def _cma_es_objective(evaluation: CandidateEvaluation) -> float:
    objective = float(evaluation.lagrangian)
    if evaluation.band_refused:
        objective += 1_000_000.0
    if not math.isfinite(objective):
        return 1_000_000_000.0
    return objective
