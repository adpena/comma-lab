"""Predictor score-band module with refusal modes (council 22/0 verdict 2026-05-05).

CALIBRATION SOURCES (each constant tagged per `tools/check_calibration_provenance.py`
recipe):

1. Contest formula constants `100, sqrt(10), 25` are CONTEST-DEFINED — see
   `upstream/evaluate.py` and CLAUDE.md "TRUE score data" section.
2. `PR106_TOTAL_RATE_DENOM = 37545489` is the contest's fixed reference video
   byte count (CONTEST-DEFINED).
3. Distortion curve coefficients are FITTED from empirical anchors, never
   hardcoded — see `fit_distortion_curve` and `.omx/calibration/anchors_*.json`.

REFUSAL MODES (council Q1 prescription) — checked in this exact order:
  1. INSUFFICIENT_ANCHORS:  <3 calibration anchors loaded
  2. LOSSY_ANCHOR_INVALID_NO_RATE_SAVINGS:  any lossy anchor has archive_bytes >= tightest lossless
  3. EXTRAPOLATION:  requested rel_err outside calibration range (with 20% margin)
  4. HIGH_REL_ERR_WITHOUT_PROXY:  rel_err > 1.0% AND no distortion_proxy provided
  5. CURVE_FIT_DEGENERATE:  insufficient lossy anchors to fit power-law curve
  6. CURVE_FIT_DEGENERATE_NUMERICAL_FLOOR:  proxy curve fit hit the numerical floor
  7. CURVE_FIT_NON_MONOTONE:  fitted exponent b ≤ 0 (distortion would decrease with rel_err)
  8. LOSSY_BETTER_THAN_LOSSLESS_INCOHERENT:  predicted_high < tightest lossless baseline AND rel_err > 0

The empirical apogee_int4 fixture (rel_err=7.09%, score 1.4287) is the
acceptance test in `src/tac/tests/test_score_band_predictor.py`.

Adversarial review 2026-05-05 (subagent a0455937b64e0dbb5) added:
  - CalibrationAnchor.__post_init__ validates rate_unscaled vs archive_bytes/denom (M3)
  - sanity gate uses min(lossless score) not list[0] (M2)
  - non-monotone curve guard (M1)
  - prediction_method field distinguishes proxy vs power-law fit (N2)
  - check order aligned with docstring (C1)
  - dead `rate_contribution` removed (C2), `field` import removed (N1)
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# ── Contest-defined constants (NEVER tune these — they are scoring-rule fixed) ──
SEG_COEFFICIENT = 100.0  # [contest-defined] upstream/evaluate.py
POSE_COEFFICIENT_SQRT_INNER = 10.0  # [contest-defined] upstream/evaluate.py
RATE_COEFFICIENT = 25.0  # [contest-defined] upstream/evaluate.py
PR106_TOTAL_RATE_DENOM = 37545489  # [contest-defined] reference video bytes

# ── Defaults derived from process discipline (council Q1) ──
MIN_CALIBRATION_ANCHORS = 3  # [heuristic:Q1-Dykstra "≥3 anchors define a curve"]
HIGH_REL_ERR_THRESHOLD_PCT = 1.0  # [heuristic:Q1-Hotz "above 1%, run local proxy"]


@dataclass(frozen=True)
class CalibrationAnchor:
    """A measured (rel_err_pct, contest_cuda_score) point.

    Persisted in `.omx/calibration/anchors_<lane_class>.json`. The
    `__post_init__` validates that `rate_unscaled` matches
    `archive_bytes / PR106_TOTAL_RATE_DENOM` within tolerance — guards against
    stale JSON state where the byte count was bumped but the cached rate
    wasn't recomputed (M3 from adversarial review 2026-05-05).

    `anchor_role` (added 2026-05-05 tier-1 cleanup) distinguishes:
      - "fit" (default): a real lossy-curve-fitting datapoint (rate-reducing)
      - "compatibility_only": an inflate-path-validation reference whose
        archive is structurally NOT rate-reducing vs lossless (e.g., int8
        anchor at 187731 bytes vs PR106 lossless 186239). Excluded from
        lossy-curve fitting AND excluded from the `lossy_anchor_invalid_no_rate_savings`
        refusal so it can sit alongside real lossy anchors without
        poisoning the predictor.
    """
    lane_id: str  # e.g. "lane_pr106_baseline", "lane_apogee_int8"
    rel_err_pct_per_weight: float  # 0.0 for lossless baseline
    archive_bytes: int
    contest_cuda_score: float  # canonical_score from contest_auth_eval.json
    avg_pose_dist: float
    avg_seg_dist: float
    rate_unscaled: float
    measured_utc: str  # ISO timestamp of contest-CUDA dispatch
    job_id: str  # e.g. "apogee-int8-baseline-confirm-20260505t174500z"
    archive_sha256: str
    notes: str = ""
    anchor_role: str = "fit"  # "fit" | "compatibility_only"

    def __post_init__(self) -> None:
        # Validate rate_unscaled is consistent with archive_bytes/denom (M3).
        # Tolerance 1e-6 allows for round-off from the JSON file's rounded
        # rate_unscaled value but catches any sign error or wrong denominator.
        expected = self.archive_bytes / PR106_TOTAL_RATE_DENOM
        if abs(self.rate_unscaled - expected) > 1e-6:
            raise ValueError(
                f"CalibrationAnchor({self.lane_id}): rate_unscaled={self.rate_unscaled} "
                f"inconsistent with archive_bytes={self.archive_bytes} "
                f"/ PR106_TOTAL_RATE_DENOM={PR106_TOTAL_RATE_DENOM} "
                f"(expected {expected:.10f}; diff {abs(self.rate_unscaled - expected):.2e})"
            )
        if self.anchor_role not in ("fit", "compatibility_only"):
            raise ValueError(
                f"CalibrationAnchor({self.lane_id}): anchor_role={self.anchor_role!r} "
                "must be 'fit' or 'compatibility_only'"
            )


@dataclass(frozen=True)
class ScoreBand:
    """Predicted score band with confidence and refusal handling.

    `prediction_method` distinguishes how the distortion estimate was produced
    so callers can apply different downstream tolerances (N2 from review).
    """
    low: float
    high: float
    confidence: str  # "calibrated_strong", "calibrated_weak", "extrapolation_risk", "none"
    refused: bool
    refusal_reason: str = ""
    distortion_estimate_used: bool = False
    predicted_pose: float = 0.0
    predicted_seg: float = 0.0
    predicted_rate: float = 0.0
    prediction_method: str = "none"  # "proxy" | "power_law_fit" | "none"
    derivation: str = ""  # short trace of how the band was computed

    def as_str(self) -> str:
        if self.refused:
            return f"REFUSED ({self.refusal_reason})"
        return f"[{self.low:.4f}, {self.high:.4f}] (confidence={self.confidence})"


# DistortionProxy is any callable that takes (archive_bytes, rel_err_pct,
# n_layers) and returns (predicted_avg_pose_dist, predicted_avg_seg_dist).
# In production this runs a local inflate + scorer-on-N-frames forward pass
# (see `tools/distortion_proxy_local.py` — to be written; pending Q1a wiring).
DistortionProxy = Callable[[int, float, int], tuple[float, float]]


def load_calibration_anchors(path: Path) -> list[CalibrationAnchor]:
    """Load anchors from a JSON list."""
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"calibration anchors file must be JSON list: {path}")
    return [CalibrationAnchor(**entry) for entry in data]


def fit_distortion_curve(anchors: list[CalibrationAnchor]) -> dict[str, float]:
    """Fit a power-law distortion curve `D ≈ a * rel_err^b + d_baseline`.

    Returns dict with keys `a`, `b`, `d_baseline`, `n_anchors`. With <3 anchors
    returns degenerate fit (a=NaN, b=NaN). The fit is on AVG_POSE_DIST + AVG_SEG_DIST
    summed (the "raw distortion budget"), not on the contest-formula score.

    The fit is intentionally simple (2-parameter) because we have very few
    anchors (3) and over-parameterization would memorize, not generalize.
    Closed-form via log-linear regression on `log(D - d_baseline) ≈ b·log(rel_err) + log(a)`.

    Anchors with `anchor_role="compatibility_only"` are excluded from the fit
    (they are inflate-path-validation references whose bytes don't represent
    real lossy-direction rate savings). They still count toward the
    MIN_CALIBRATION_ANCHORS gate via len(anchors) so callers see the full
    calibration set, but they cannot poison the lossy power-law regression.
    """
    if len(anchors) < MIN_CALIBRATION_ANCHORS:
        return {"a": float("nan"), "b": float("nan"), "d_baseline": float("nan"), "n_anchors": len(anchors)}

    # baseline anchor = the one with rel_err == 0 (lossless reference)
    baseline_anchors = [a for a in anchors if a.rel_err_pct_per_weight == 0.0]
    if not baseline_anchors:
        # No lossless anchor — use min rel_err as the "baseline" approximation.
        baseline = min(anchors, key=lambda a: a.rel_err_pct_per_weight)
        d_baseline = baseline.avg_pose_dist + baseline.avg_seg_dist
    else:
        baseline = baseline_anchors[0]
        d_baseline = baseline.avg_pose_dist + baseline.avg_seg_dist

    # Lossy anchors only (rel_err > 0 AND anchor_role == "fit"); fit log(D - d_baseline)
    # ≈ b·log(rel_err) + log(a). compatibility_only anchors are excluded from the fit.
    lossy = [
        a for a in anchors
        if a.rel_err_pct_per_weight > 0.0 and a.anchor_role == "fit"
    ]
    if len(lossy) < 2:
        return {"a": float("nan"), "b": float("nan"), "d_baseline": d_baseline, "n_anchors": len(anchors)}

    log_rel = [math.log(a.rel_err_pct_per_weight) for a in lossy]
    log_d_minus_base = []
    for a in lossy:
        d = a.avg_pose_dist + a.avg_seg_dist
        excess = max(d - d_baseline, 1e-12)  # numerical floor
        log_d_minus_base.append(math.log(excess))

    n = len(lossy)
    mean_x = sum(log_rel) / n
    mean_y = sum(log_d_minus_base) / n
    var_x = sum((x - mean_x) ** 2 for x in log_rel)
    cov_xy = sum((log_rel[i] - mean_x) * (log_d_minus_base[i] - mean_y) for i in range(n))
    if var_x == 0:
        return {"a": float("nan"), "b": float("nan"), "d_baseline": d_baseline, "n_anchors": len(anchors)}
    b = cov_xy / var_x
    log_a = mean_y - b * mean_x
    a = math.exp(log_a)
    return {"a": a, "b": b, "d_baseline": d_baseline, "n_anchors": len(anchors)}


def _score_from_components(pose: float, seg: float, rate_unscaled: float) -> float:
    """Apply the contest scoring formula. CONTEST-DEFINED, do not modify."""
    return (
        SEG_COEFFICIENT * seg
        + math.sqrt(POSE_COEFFICIENT_SQRT_INNER * max(pose, 0.0))
        + RATE_COEFFICIENT * rate_unscaled
    )


def predict_score_band(
    archive_bytes: int,
    rel_err_pct_per_weight: float,
    n_quantized_layers: int,
    calibration_anchors: list[CalibrationAnchor],
    distortion_proxy: DistortionProxy | None = None,
    band_half_width_score: float = 0.05,  # [heuristic:council-Q1 "0.05 = sane sanity-band width"]
) -> ScoreBand:
    """Predict a contest-CUDA score band for a candidate.

    Refuses (refused=True) when:
      1. INSUFFICIENT_ANCHORS: <3 calibration anchors
      2. EXTRAPOLATION: rel_err outside [min, max] of anchors AND not within 20% of an endpoint
      3. HIGH_REL_ERR_WITHOUT_PROXY: rel_err > 1.0% AND distortion_proxy is None
      4. LOSSY_BETTER_THAN_LOSSLESS_INCOHERENT: predicted_high < lossless baseline AND rel_err > 0

    The empirical apogee_int4 fixture (rel_err=7.09%, score 1.4287) MUST cause refusal
    when calibration only has the PR106 anchor (insufficient) — landing 1.4287 with a
    confident band [0.155, 0.180] is exactly the failure this module prevents.
    """
    # ── Refusal #1: insufficient anchors ──────────────────────────────────
    if len(calibration_anchors) < MIN_CALIBRATION_ANCHORS:
        return ScoreBand(
            low=0.0, high=0.0, confidence="none",
            refused=True,
            refusal_reason=f"insufficient_anchors: have {len(calibration_anchors)}, need {MIN_CALIBRATION_ANCHORS}",
            derivation="Predictor requires ≥3 calibration anchors per council Q1 (Dykstra).",
        )

    # ── Refusal #2 (Bug #2 root-cause fix 2026-05-05) ─────────────────────
    # A "lossy" anchor (rel_err > 0) MUST have archive_bytes strictly less
    # than the tightest lossless anchor — that's the entire point of being
    # lossy. The predictor's curve-fit assumes monotone rate reduction along
    # the lossy direction; an anchor that is BOTH lossy AND larger than
    # lossless violates this assumption and produces a degenerate fit.
    #
    # Concrete failure (canonical anchor file 2026-05-05): int8 anchor at
    # 187,731 bytes > PR106 lossless 186,239 bytes (+1,492 bytes layout
    # overhead). The "lossy" int8 brings no rate savings; treating it as a
    # lossy data point poisons the power-law regression.
    #
    # Anchors with `anchor_role="compatibility_only"` are EXCLUDED from this
    # gate (tier-1 cleanup 2026-05-05): they are inflate-path-validation
    # references that intentionally have no rate savings vs lossless. They
    # are still loaded for context but neither poison the fit nor trigger
    # this refusal.
    lossless = [a for a in calibration_anchors if a.rel_err_pct_per_weight == 0.0]
    if lossless:
        tightest_lossless_bytes = min(a.archive_bytes for a in lossless)
        offending = [
            a for a in calibration_anchors
            if a.rel_err_pct_per_weight > 0.0
            and a.anchor_role == "fit"
            and a.archive_bytes >= tightest_lossless_bytes
        ]
        if offending:
            names = ", ".join(f"{a.lane_id}({a.archive_bytes}B)" for a in offending)
            return ScoreBand(
                low=0.0, high=0.0, confidence="none",
                refused=True,
                refusal_reason=(
                    f"lossy_anchor_invalid_no_rate_savings: anchor(s) [{names}] are labeled "
                    f"lossy (rel_err>0) but have archive_bytes >= tightest lossless ({tightest_lossless_bytes}B). "
                    "A lossy anchor with no rate savings is structurally invalid; relabel it as "
                    "anchor_role=compatibility_only to exclude it from lossy-curve fitting while "
                    "preserving the inflate-path validation reference."
                ),
                derivation="Council Q1 (Dykstra/Shannon): lossy direction must reduce rate.",
            )

    rate_unscaled = archive_bytes / PR106_TOTAL_RATE_DENOM

    # ── Refusal #3: extrapolation outside anchor range ────────────────────
    # Order: extrapolation check before proxy gate per docstring contract (C1).
    rel_err_anchors = [a.rel_err_pct_per_weight for a in calibration_anchors]
    rel_err_min, rel_err_max = min(rel_err_anchors), max(rel_err_anchors)
    rel_err_range = max(rel_err_max - rel_err_min, 1e-9)
    extrapolation_margin = 0.2 * rel_err_range
    if rel_err_pct_per_weight < rel_err_min - extrapolation_margin or rel_err_pct_per_weight > rel_err_max + extrapolation_margin:
        return ScoreBand(
            low=0.0, high=0.0, confidence="none",
            refused=True,
            refusal_reason=(
                f"extrapolation: rel_err={rel_err_pct_per_weight:.2f}% outside calibrated range "
                f"[{rel_err_min:.2f}%, {rel_err_max:.2f}%] (with 20% margin). Add a calibration anchor "
                "in this regime before banding."
            ),
            derivation="Council Q1 (Shannon): extrapolation outside calibration is unsupported.",
        )

    # ── Refusal #3: high rel_err without distortion proxy ─────────────────
    if rel_err_pct_per_weight > HIGH_REL_ERR_THRESHOLD_PCT and distortion_proxy is None:
        return ScoreBand(
            low=0.0, high=0.0, confidence="none",
            refused=True,
            refusal_reason=(
                f"high_rel_err_without_proxy: rel_err={rel_err_pct_per_weight:.2f}% > {HIGH_REL_ERR_THRESHOLD_PCT}% "
                "and no distortion_proxy provided. Run local inflate + scorer to estimate distortion before banding."
            ),
            derivation="Council Q1 (Hotz): >1% per-weight error needs empirical distortion estimate.",
        )

    # ── Predict distortion ────────────────────────────────────────────────
    pred_pose: float
    pred_seg: float
    method: str
    # Bug #7 (root-cause fix 2026-05-05): track proxy fit quality so band
    # confidence + width can be adjusted when the proxy's internal curve fit
    # is degenerate (e.g., one anchor's per-component excess is below the
    # numerical floor — Bug #1). The previous code unconditionally tagged
    # `calibrated_strong` whenever a proxy was supplied, regardless of fit.
    proxy_fit_degenerate = False
    proxy_degenerate_reason = ""
    if distortion_proxy is not None:
        pred_pose, pred_seg = distortion_proxy(archive_bytes, rel_err_pct_per_weight, n_quantized_layers)
        method = "proxy"
        derivation_pieces = ["distortion via local proxy"]
        # Inspect the proxy's exposed curves (if any) for NaN-as-degenerate.
        proxy_curves = getattr(distortion_proxy, "curves", None)
        if isinstance(proxy_curves, dict):
            degenerate_components = []
            for comp_name in ("pose", "seg"):
                comp = proxy_curves.get(comp_name)
                if isinstance(comp, dict):
                    a = comp.get("a", float("nan"))
                    b = comp.get("b", float("nan"))
                    reason = comp.get("degenerate_reason", "")
                    try:
                        is_nan = math.isnan(a) or math.isnan(b)
                    except TypeError:
                        is_nan = True
                    if is_nan or reason:
                        degenerate_components.append(f"{comp_name}({reason or 'NaN'})")
            if degenerate_components:
                proxy_fit_degenerate = True
                proxy_degenerate_reason = "; ".join(degenerate_components)
                derivation_pieces.append(
                    f"proxy_curve_degenerate[{proxy_degenerate_reason}] → confidence downgraded + band widened 50%"
                )
    else:
        # Use fitted curve for low-rel-err regime (≤1%).
        curve = fit_distortion_curve(calibration_anchors)
        if math.isnan(curve["a"]) or math.isnan(curve["b"]):
            return ScoreBand(
                low=0.0, high=0.0, confidence="none",
                refused=True,
                refusal_reason="curve_fit_degenerate: insufficient lossy anchors to fit distortion curve",
                derivation="fit_distortion_curve returned NaN; need ≥2 lossy anchors.",
            )
        # ── Refusal #5: non-monotone fit (M1 from review) ─────────────────
        if curve["b"] <= 0:
            return ScoreBand(
                low=0.0, high=0.0, confidence="none",
                refused=True,
                refusal_reason=(
                    f"curve_fit_non_monotone: fitted exponent b={curve['b']:.4g} ≤ 0. "
                    "Distortion would decrease with rel_err — physically impossible for naive PTQ. "
                    "Likely cause: anchors include a QAT-suppressed point mixed with naive-PTQ points; "
                    "split into separate lane classes before fitting."
                ),
                derivation="Council M1 (Shannon/Dykstra): power-law must be monotone.",
            )
        if rel_err_pct_per_weight == 0.0:
            d_total = curve["d_baseline"]
        else:
            d_total = curve["d_baseline"] + curve["a"] * (rel_err_pct_per_weight ** curve["b"])
        # Split D = pose + seg using nearest-anchor ratio
        nearest = min(calibration_anchors, key=lambda a: abs(a.rel_err_pct_per_weight - rel_err_pct_per_weight))
        anchor_d = nearest.avg_pose_dist + nearest.avg_seg_dist
        pose_ratio = nearest.avg_pose_dist / anchor_d if anchor_d > 0 else 0.0
        pred_pose = d_total * pose_ratio
        pred_seg = d_total * (1.0 - pose_ratio)
        method = "power_law_fit"
        derivation_pieces = [f"power-law fit a={curve['a']:.4g} b={curve['b']:.4g}"]

    point_score = _score_from_components(pred_pose, pred_seg, rate_unscaled)
    # Bug #7 fix: when the proxy fit is degenerate, widen the band by 50%
    # per Boyd interval-inflation under degenerate fit. This ensures downstream
    # consumers see a wider uncertainty range matching the actual fit quality.
    effective_half_width = band_half_width_score
    if proxy_fit_degenerate:
        effective_half_width = band_half_width_score * 1.5
    band_low = point_score - effective_half_width
    band_high = point_score + effective_half_width
    derivation_pieces.append(f"point_score={point_score:.4f} ± {effective_half_width}")

    # ── Refusal #6: lossy-better-than-lossless sanity gate ────────────────
    # Use the TIGHTEST (best) lossless score per M2 from review — multiple
    # lossless anchors should not let list[0] make the gate non-deterministic.
    lossless = [a for a in calibration_anchors if a.rel_err_pct_per_weight == 0.0]
    if lossless and rel_err_pct_per_weight > 0.0:
        lossless_score = min(a.contest_cuda_score for a in lossless)
        if band_high < lossless_score:
            return ScoreBand(
                low=0.0, high=0.0, confidence="none",
                refused=True,
                refusal_reason=(
                    f"lossy_better_than_lossless_incoherent: predicted_high={band_high:.4f} < "
                    f"tightest lossless baseline {lossless_score:.4f}. A lossy compression cannot strictly improve "
                    "every component; this prediction is mathematically incoherent."
                ),
                derivation="Council Q1 (Contrarian): sanity gate fires.",
            )

    # ── Confidence label ──────────────────────────────────────────────────
    # Bug #7 fix: degraded proxy fits do NOT earn `calibrated_strong` —
    # downgrade to `calibrated_weak` so consumers know the proxy did NOT
    # carry meaningful per-component information.
    if method == "proxy":
        confidence = "calibrated_weak" if proxy_fit_degenerate else "calibrated_strong"
    elif rel_err_min <= rel_err_pct_per_weight <= rel_err_max:
        confidence = "calibrated_weak"  # within range, no proxy
    else:
        confidence = "extrapolation_risk"  # within 20% margin

    return ScoreBand(
        low=band_low,
        high=band_high,
        confidence=confidence,
        refused=False,
        distortion_estimate_used=True,
        predicted_pose=pred_pose,
        predicted_seg=pred_seg,
        predicted_rate=rate_unscaled,
        prediction_method=method,
        derivation=" | ".join(derivation_pieces),
    )
