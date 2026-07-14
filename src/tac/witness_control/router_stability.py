# SPDX-License-Identifier: MIT
"""Deterministic router stability for the advisory costate organ.

This module maps three *patterns* from NVIDIA-NeMo/labs-molt onto the #426/#436
costate organ without importing Molt or adding an actuation path:

* the discrete gate is evaluated in deterministic NumPy ``float32`` and emits a
  distance-to-boundary certificate;
* a DECIDE record pins the exact selected expert and APPLY consumes that record,
  while later decisions remain free to use an updated router;
* architecture evaluation can use self-normalized, clipped, support-masked
  live/backtest regime-density ratios, but fails closed when density custody is
  absent.

Every surface is advisory and score-neutral.  A replayed expert name is not an
operator-GO token and cannot launch or mutate a witness run.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

AXIS_TAG = "[macOS advisory] NON-PROMOTABLE"
SCHEMA_VERSION = "pact.costate_router_stability.v1"
BLOCKED_DISTRIBUTION_CUSTODY = "BLOCKED_DISTRIBUTION_CUSTODY"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RouterReplayError(RuntimeError):
    """Base class for replay-ledger contract failures."""


class RouterReplayMismatchError(RouterReplayError):
    """APPLY requested an expert different from the recorded DECIDE expert."""


class DistributionCustodyError(ValueError):
    """Importance weighting was requested without sufficient density custody."""


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _f32(value: float) -> np.float32:
    """Canonical scalar conversion used at every discrete gate comparison."""
    out = np.float32(value)
    if not np.isfinite(out):
        raise ValueError(f"router gate requires finite float32 input; got {value!r}")
    return out


def _ulp_guard(*values: np.float32) -> float:
    """Two-input float32 rounding guard around a subtraction/comparison boundary."""
    spacings = [abs(float(np.spacing(v))) for v in values]
    return 2.0 * max([*spacings, float(np.finfo(np.float32).tiny)])


@dataclass(frozen=True)
class RouterGateCertificate:
    """The selected branch and its distance from every active decision boundary."""

    selected_regime: str
    selected_tool: str
    n_past_intervals: int
    gate_dtype: str
    recent_slope_mag: float | None
    median_slope_mag: float | None
    slope_signed_margin: float | None
    slope_abs_margin: float | None
    slope_relative_margin: float | None
    slope_ulp_guard: float | None
    slope_margin_ulps: float | None
    surprise_ratio: float | None
    surprise_threshold: float | None
    surprise_signed_margin: float | None
    surprise_abs_margin: float | None
    surprise_ulp_guard: float | None
    surprise_margin_ulps: float | None
    stable_beyond_float32_roundoff: bool
    active_boundaries: tuple[str, ...]
    tie_break_rule: str
    policy_sha256: str
    axis_tag: str = AXIS_TAG
    score_claim: bool = False
    actuation: str = "NONE"

    def to_dict(self) -> dict:
        return asdict(self)


def certify_fp32_gate(
    *,
    recent_slope_mag: float,
    median_slope_mag: float,
    n_past_intervals: int,
    surprise_ratio: float,
    meta_lambda_guard: bool,
    policy: Mapping[str, str],
    surprise_threshold: float = 1.5,
) -> RouterGateCertificate:
    """Select the regime/tool using canonical NumPy-fp32 comparisons.

    The tie rules preserve #436 semantics: ``recent == median`` is transient and
    ``surprise == threshold`` does not defer.  The certificate reports margins in
    native units and float32 ULP guards; a deterministic selection can therefore be
    distinguished from a selection that is numerically close to a branch boundary.
    """
    if set(policy) != {"transient", "plateau", "uncertain"}:
        raise ValueError("router policy must cover transient/plateau/uncertain exactly")
    policy_payload = [[str(k), str(policy[k])] for k in sorted(policy)]
    policy_sha = _sha256_json(policy_payload)

    if n_past_intervals < 2:
        regime = "uncertain"
        return RouterGateCertificate(
            selected_regime=regime,
            selected_tool=str(policy[regime]),
            n_past_intervals=int(n_past_intervals),
            gate_dtype="numpy.float32",
            recent_slope_mag=_finite_or_none(recent_slope_mag),
            median_slope_mag=None,
            slope_signed_margin=None,
            slope_abs_margin=None,
            slope_relative_margin=None,
            slope_ulp_guard=None,
            slope_margin_ulps=None,
            surprise_ratio=None,
            surprise_threshold=None,
            surprise_signed_margin=None,
            surprise_abs_margin=None,
            surprise_ulp_guard=None,
            surprise_margin_ulps=None,
            stable_beyond_float32_roundoff=True,
            active_boundaries=("history_count_lt_2",),
            tie_break_rule=("integer history guard; at >=2 intervals slope tie selects "
                            "transient; surprise tie does not defer"),
            policy_sha256=policy_sha,
        )

    recent = _f32(recent_slope_mag)
    median = _f32(median_slope_mag)
    slope_delta = np.float32(recent - median)
    slope_guard = _ulp_guard(recent, median)
    slope_abs = abs(float(slope_delta))
    slope_rel = slope_abs / max(abs(float(median)), float(np.finfo(np.float32).tiny))
    slope_ulps = slope_abs / slope_guard
    plateau = bool(slope_delta < np.float32(0.0))
    regime = "plateau" if plateau else "transient"
    active = ["recent_slope_minus_running_median"]
    stable = slope_abs > slope_guard

    ratio_out: float | None = None
    threshold_out: float | None = None
    surprise_delta_out: float | None = None
    surprise_abs_out: float | None = None
    surprise_guard_out: float | None = None
    surprise_ulps_out: float | None = None
    if meta_lambda_guard and not plateau and math.isfinite(float(surprise_ratio)):
        ratio = _f32(surprise_ratio)
        threshold = _f32(surprise_threshold)
        surprise_delta = np.float32(ratio - threshold)
        surprise_guard = _ulp_guard(ratio, threshold)
        surprise_abs = abs(float(surprise_delta))
        surprise_ulps = surprise_abs / surprise_guard
        active.append("model_surprise_ratio_minus_threshold")
        stable = stable and surprise_abs > surprise_guard
        if surprise_delta > np.float32(0.0):
            regime = "uncertain"
        ratio_out = float(ratio)
        threshold_out = float(threshold)
        surprise_delta_out = float(surprise_delta)
        surprise_abs_out = surprise_abs
        surprise_guard_out = surprise_guard
        surprise_ulps_out = surprise_ulps

    return RouterGateCertificate(
        selected_regime=regime,
        selected_tool=str(policy[regime]),
        n_past_intervals=int(n_past_intervals),
        gate_dtype="numpy.float32",
        recent_slope_mag=float(recent),
        median_slope_mag=float(median),
        slope_signed_margin=float(slope_delta),
        slope_abs_margin=slope_abs,
        slope_relative_margin=slope_rel,
        slope_ulp_guard=slope_guard,
        slope_margin_ulps=slope_ulps,
        surprise_ratio=ratio_out,
        surprise_threshold=threshold_out,
        surprise_signed_margin=surprise_delta_out,
        surprise_abs_margin=surprise_abs_out,
        surprise_ulp_guard=surprise_guard_out,
        surprise_margin_ulps=surprise_ulps_out,
        stable_beyond_float32_roundoff=stable,
        active_boundaries=tuple(active),
        tie_break_rule=("slope tie (recent == median) selects transient; surprise tie "
                        "(ratio == threshold) does not defer; policy keys canonicalized "
                        "lexicographically"),
        policy_sha256=policy_sha,
    )


@dataclass(frozen=True)
class RouterDecisionRecord:
    """Durable DECIDE payload; its digest is the replay token."""

    decision_id: str
    run_ref: str
    decision_epoch: float
    selected_regime: str
    selected_tool: str
    policy_sha256: str
    certificate: dict
    router_learning_frozen: bool = False
    axis_tag: str = AXIS_TAG
    score_claim: bool = False
    actuation: str = "NONE"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RouterReplayOutcome:
    decision_id: str
    selected_tool: str
    requested_tool: str
    status: str
    mismatch_alarm: bool
    router_learning_frozen: bool = False
    axis_tag: str = AXIS_TAG
    score_claim: bool = False
    actuation: str = "NONE"

    def to_dict(self) -> dict:
        return asdict(self)


def make_decision_record(
    *,
    run_ref: str,
    decision_epoch: float,
    selected_regime: str,
    selected_tool: str,
    certificate: RouterGateCertificate,
) -> RouterDecisionRecord:
    """Build a content-addressed DECIDE record (repeat-safe across restart)."""
    if selected_regime != certificate.selected_regime:
        raise ValueError("selected_regime does not match the fp32 gate certificate")
    if selected_tool != certificate.selected_tool:
        raise ValueError("selected_tool does not match the fp32 gate certificate")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_ref": str(run_ref),
        "decision_epoch": float(decision_epoch),
        "selected_regime": str(selected_regime),
        "selected_tool": str(selected_tool),
        "policy_sha256": certificate.policy_sha256,
        "certificate": certificate.to_dict(),
        "router_learning_frozen": False,
        "axis_tag": AXIS_TAG,
        "score_claim": False,
        "actuation": "NONE",
    }
    return RouterDecisionRecord(decision_id=_sha256_json(payload), **{
        key: payload[key] for key in (
            "run_ref", "decision_epoch", "selected_regime", "selected_tool",
            "policy_sha256", "certificate", "router_learning_frozen", "axis_tag",
            "score_claim", "actuation")
    })


def _record_expected_decision_id(record_payload: Mapping[str, object]) -> str:
    basis = {
        "schema_version": SCHEMA_VERSION,
        **{key: record_payload[key] for key in (
            "run_ref", "decision_epoch", "selected_regime", "selected_tool",
            "policy_sha256", "certificate", "router_learning_frozen", "axis_tag",
            "score_claim", "actuation")},
    }
    return _sha256_json(basis)


def _read_events_locked(fh) -> list[dict]:
    fh.seek(0)
    events: list[dict] = []
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RouterReplayError(f"malformed replay ledger JSONL: {exc}") from exc
        if not isinstance(row, dict):
            raise RouterReplayError("replay ledger row must be an object")
        events.append(row)
    return events


def append_decide_record(path: str | Path, record: RouterDecisionRecord) -> RouterDecisionRecord:
    """Append DECIDE exactly once; a conflicting duplicate id fails closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            events = _read_events_locked(fh)
            prior = [r for r in events
                     if r.get("event") == "DECIDE" and r.get("decision_id") == record.decision_id]
            payload = record.to_dict()
            if _record_expected_decision_id(payload) != record.decision_id:
                raise RouterReplayError("DECIDE payload does not match its content address")
            if prior:
                if _canonical_json(prior[-1].get("payload")) != _canonical_json(payload):
                    raise RouterReplayError(
                        f"decision_id collision with different payload: {record.decision_id}")
                return record
            event = {
                "schema_version": SCHEMA_VERSION,
                "event": "DECIDE",
                "written_at_utc": _utc(),
                "decision_id": record.decision_id,
                "payload": payload,
            }
            fh.seek(0, 2)
            fh.write(_canonical_json(event) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return record


def replay_apply(
    path: str | Path,
    decision_id: str,
    *,
    requested_tool: str | None = None,
) -> RouterReplayOutcome:
    """Consume the exact DECIDE selection and append APPLY or MISMATCH_ALARM.

    ``requested_tool`` models the consumer's proposed arm.  Omitting it means the
    consumer accepts the recorded selection.  This function never actuates that arm.
    """
    path = Path(path)
    if not path.exists():
        raise RouterReplayError(f"replay ledger absent: {path}")
    with open(path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            events = _read_events_locked(fh)
            matches = [r for r in events
                       if r.get("event") == "DECIDE" and r.get("decision_id") == decision_id]
            if not matches:
                raise RouterReplayError(f"unknown decision_id {decision_id}")
            payload = matches[-1].get("payload") or {}
            if _record_expected_decision_id(payload) != decision_id:
                raise RouterReplayError("stored DECIDE payload failed content-address verification")
            selected = str(payload.get("selected_tool"))
            requested = selected if requested_tool is None else str(requested_tool)
            mismatch = requested != selected
            outcome = RouterReplayOutcome(
                decision_id=decision_id,
                selected_tool=selected,
                requested_tool=requested,
                status="MISMATCH_ALARM" if mismatch else "REPLAY_MATCH",
                mismatch_alarm=mismatch,
            )
            event = {
                "schema_version": SCHEMA_VERSION,
                "event": outcome.status,
                "written_at_utc": _utc(),
                "decision_id": decision_id,
                "payload": outcome.to_dict(),
            }
            fh.seek(0, 2)
            fh.write(_canonical_json(event) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    if mismatch:
        raise RouterReplayMismatchError(
            f"router replay mismatch: decided {selected!r}, APPLY requested {requested!r}; "
            f"alarm appended for {decision_id}")
    return outcome


def load_replay_events(path: str | Path) -> tuple[dict, ...]:
    """Read the replay ledger for audit/testing without changing it."""
    path = Path(path)
    if not path.exists():
        return ()
    with open(path, encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
        try:
            return tuple(_read_events_locked(fh))
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class RegimeDensityCustody:
    """Explicit source custody for backtest and visited-live regime densities."""

    backtest_density: Mapping[str, float]
    live_density: Mapping[str, float]
    backtest_source_sha256: str
    live_source_sha256: str
    regime_schema_sha256: str
    backtest_source: str
    live_source: str

    def normalized(self) -> tuple[dict[str, float], dict[str, float]]:
        for name, value in (
            ("backtest_source_sha256", self.backtest_source_sha256),
            ("live_source_sha256", self.live_source_sha256),
            ("regime_schema_sha256", self.regime_schema_sha256),
        ):
            if not _SHA256_RE.fullmatch(str(value)):
                raise DistributionCustodyError(
                    f"{BLOCKED_DISTRIBUTION_CUSTODY}: {name} is not a sha256")
        keys = set(self.backtest_density) | set(self.live_density)
        if not keys:
            raise DistributionCustodyError(
                f"{BLOCKED_DISTRIBUTION_CUSTODY}: empty density support")

        def _norm(values: Mapping[str, float], label: str) -> dict[str, float]:
            out = {str(k): float(v) for k, v in values.items()}
            if any((not math.isfinite(v)) or v < 0.0 for v in out.values()):
                raise DistributionCustodyError(
                    f"{BLOCKED_DISTRIBUTION_CUSTODY}: {label} has invalid mass")
            total = sum(out.values())
            if total <= 0.0:
                raise DistributionCustodyError(
                    f"{BLOCKED_DISTRIBUTION_CUSTODY}: {label} has zero mass")
            return {k: out.get(k, 0.0) / total for k in sorted(keys)}

        return _norm(self.backtest_density, "backtest_density"), _norm(
            self.live_density, "live_density")

    def to_dict(self) -> dict:
        back, live = self.normalized()
        return {
            "backtest_density": back,
            "live_density": live,
            "backtest_source_sha256": self.backtest_source_sha256,
            "live_source_sha256": self.live_source_sha256,
            "regime_schema_sha256": self.regime_schema_sha256,
            "backtest_source": self.backtest_source,
            "live_source": self.live_source,
        }


@dataclass(frozen=True)
class ImportanceWeightDiagnostics:
    regimes: tuple[str, ...]
    raw_ratios: tuple[float, ...]
    clipped_ratios: tuple[float, ...]
    support_mask: tuple[bool, ...]
    normalized_weights: tuple[float, ...]
    clip_bounds: tuple[float, float]
    effective_sample_size: float
    n_retained: int
    status: str = "IS_WEIGHTED"

    def to_dict(self) -> dict:
        return asdict(self)


def self_normalized_clipped_masked_weights(
    regimes: Sequence[str],
    *,
    custody: RegimeDensityCustody | None,
    clip_bounds: tuple[float, float] | None,
    support_mask: Sequence[bool] | None = None,
) -> ImportanceWeightDiagnostics:
    """Compute ``w_i ∝ mask_i clip(p_live(g_i)/p_backtest(g_i))``.

    Retained weights are self-normalized to mean one.  Missing density hashes,
    missing target support, or unspecified clip bounds are blockers; uniform weights
    are never silently substituted.
    """
    if custody is None:
        raise DistributionCustodyError(
            f"{BLOCKED_DISTRIBUTION_CUSTODY}: no live/backtest density manifest")
    if clip_bounds is None:
        raise DistributionCustodyError(
            f"{BLOCKED_DISTRIBUTION_CUSTODY}: clip bounds lack value provenance")
    low, high = map(float, clip_bounds)
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high):
        raise ValueError("clip bounds must satisfy 0 < low <= high")
    if not regimes:
        raise ValueError("at least one regime row is required")
    if support_mask is None:
        mask = np.ones(len(regimes), dtype=bool)
    else:
        if len(support_mask) != len(regimes):
            raise ValueError("support_mask length must match regimes")
        mask = np.asarray(support_mask, dtype=bool)

    backtest, live = custody.normalized()
    ratios: list[float] = []
    for regime in regimes:
        if regime not in backtest or regime not in live:
            raise DistributionCustodyError(
                f"{BLOCKED_DISTRIBUTION_CUSTODY}: missing density for regime {regime!r}")
        p_back = backtest[regime]
        p_live = live[regime]
        if p_back <= 0.0 and p_live > 0.0:
            raise DistributionCustodyError(
                f"{BLOCKED_DISTRIBUTION_CUSTODY}: live regime {regime!r} has zero "
                "backtest support")
        ratio = p_live / p_back if p_back > 0.0 else 0.0
        ratios.append(ratio)
    raw = np.asarray(ratios, dtype=np.float32)
    # A zero target-density cell is outside the target estimand and is masked, not
    # raised to the positive lower clip bound.
    mask &= raw > np.float32(0.0)
    clipped = np.clip(raw, np.float32(low), np.float32(high)).astype(np.float32)
    retained = clipped * mask.astype(np.float32)
    n_retained = int(mask.sum())
    total = float(retained.sum(dtype=np.float32))
    if n_retained == 0 or total <= 0.0:
        raise DistributionCustodyError(
            f"{BLOCKED_DISTRIBUTION_CUSTODY}: mask leaves no supported rows")
    weights = retained * np.float32(n_retained / total)
    denom = float(np.square(weights, dtype=np.float32).sum(dtype=np.float32))
    ess = float(weights.sum(dtype=np.float32) ** 2 / denom) if denom > 0.0 else 0.0
    return ImportanceWeightDiagnostics(
        regimes=tuple(str(r) for r in regimes),
        raw_ratios=tuple(float(v) for v in raw),
        clipped_ratios=tuple(float(v) for v in clipped),
        support_mask=tuple(bool(v) for v in mask),
        normalized_weights=tuple(float(v) for v in weights),
        clip_bounds=(low, high),
        effective_sample_size=ess,
        n_retained=n_retained,
    )


@dataclass(frozen=True)
class ImportanceWeightedArchitectureReport:
    status: str
    per_arch_weighted_mae: Mapping[str, float]
    per_arch_unweighted_mae: Mapping[str, float]
    selected_arch: str | None
    diagnostics: dict | None
    blocker: str | None
    verdict_scope: str = "instance: one real trajectory; advisory only"
    axis_tag: str = AXIS_TAG
    score_claim: bool = False
    promotable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SequentialRouterPosteriorStep:
    """One prequential update for the route-match trajectory posterior."""

    epoch: float
    confidence_class: str
    route_matches_oracle: bool
    posterior_alpha: float
    posterior_beta: float
    posterior_match_probability: float
    posterior_std: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RouterCalibrationBin:
    """Beta-Bernoulli reliability posterior for one margin-confidence class."""

    confidence_class: str
    n_folds: int
    matches: int
    mismatch_rate: float
    mean_dispatcher_error: float
    posterior_alpha: float
    posterior_beta: float
    posterior_match_probability: float
    posterior_std: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ForecastComputeAllocation:
    """Advisory test-time compute request; it has no launcher or actuation surface."""

    epoch: float
    selected_tool: str
    confidence_class: str
    requested_k: int
    candidate_tools: tuple[str, ...]
    mode: str
    rationale: str
    actuation: str = "NONE"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RouterForecastCalibrationReport:
    """Sequential route-reliability posterior and margin-calibration diagnosis.

    The posterior is deliberately over the *walk-forward route-match path*, not a
    claim about the full physical training trajectory.  The latter still requires
    independent trajectory and live-density custody.  A data-neutral Beta(1,1)
    prior is explicit and configurable; it is an ASSUMED controller prior, not an
    empirical constant or promotion threshold.
    """

    status: str
    posterior_kind: str
    prior_alpha: float
    prior_beta: float
    prior_provenance: str
    n_folds: int
    terminal_posterior_alpha: float
    terminal_posterior_beta: float
    terminal_posterior_match_probability: float
    terminal_posterior_std: float
    stable_bin: dict
    roundoff_unstable_bin: dict
    high_minus_low_match_rate: float
    high_minus_low_mean_error: float
    sequential_posterior: tuple[dict, ...]
    compute_allocations: tuple[dict, ...]
    allocation_verdict: str
    blocker: str
    verdict_scope: str = (
        "INSTANCE: one real trajectory and hindsight-oracle route labels; advisory only")
    axis_tag: str = AXIS_TAG
    score_claim: bool = False
    promotable: bool = False
    actuation: str = "NONE"

    def to_dict(self) -> dict:
        return asdict(self)


def _beta_summary(alpha: float, beta: float) -> tuple[float, float]:
    total = alpha + beta
    mean = alpha / total
    std = math.sqrt(alpha * beta / (total * total * (total + 1.0)))
    return mean, std


def _calibration_bin(
    rows: Sequence[Mapping[str, object]],
    *,
    confidence_class: str,
    prior_alpha: float,
    prior_beta: float,
) -> RouterCalibrationBin:
    matches = sum(bool(row["route_matches_oracle"]) for row in rows)
    n_folds = len(rows)
    alpha = prior_alpha + matches
    beta = prior_beta + n_folds - matches
    posterior_mean, posterior_std = _beta_summary(alpha, beta)
    return RouterCalibrationBin(
        confidence_class=confidence_class,
        n_folds=n_folds,
        matches=matches,
        mismatch_rate=(float(n_folds - matches) / n_folds if n_folds else float("nan")),
        mean_dispatcher_error=(
            float(np.mean(np.asarray(
                [float(row["dispatcher_err"]) for row in rows], dtype=np.float32)))
            if rows else float("nan")
        ),
        posterior_alpha=alpha,
        posterior_beta=beta,
        posterior_match_probability=posterior_mean,
        posterior_std=posterior_std,
    )


def calibrate_router_forecast(
    fold_rows: Sequence[Mapping[str, object]],
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    prior_provenance: str | None = None,
    fallback_architecture: str = "A_ridge_solve",
) -> RouterForecastCalibrationReport:
    """Check whether fp32 selection margin predicts route correctness on a path.

    ``route_matches_oracle`` is a backtest-only, look-ahead label.  It is never read
    by the dispatcher.  Confidence is not threshold-fitted: the existing gate
    certificate supplies the structural split ``stable_beyond_float32_roundoff``.

    When the confidence direction is uncalibrated, the function requests K=2
    *shadow* comparison against the architectural incumbent for every fold.  It
    does not replace the decided tool: the #205 rows do not contain enough evidence
    to license a new fallback policy, and this module has no actuation surface.
    """
    if not fold_rows:
        raise ValueError("fold_rows must be non-empty")
    prior_alpha = float(prior_alpha)
    prior_beta = float(prior_beta)
    if not (math.isfinite(prior_alpha) and math.isfinite(prior_beta)
            and prior_alpha > 0.0 and prior_beta > 0.0):
        raise ValueError("Beta prior parameters must be finite and > 0")
    if prior_provenance is None:
        if prior_alpha != 1.0 or prior_beta != 1.0:
            raise ValueError(
                "non-default Beta prior requires explicit prior_provenance")
        prior_provenance = "ASSUMED_DATA_NEUTRAL_BETA_1_1; NO_PROMOTION_THRESHOLD"
    if not str(prior_provenance).strip():
        raise ValueError("prior_provenance must be non-empty")
    for row in fold_rows:
        cert = row.get("gate_certificate")
        if not isinstance(cert, Mapping) or "stable_beyond_float32_roundoff" not in cert:
            raise ValueError("each fold row requires a router gate certificate")
        if "route_matches_oracle" not in row or "dispatcher_err" not in row:
            raise ValueError("each fold row requires oracle-match and dispatcher-error labels")

    stable_rows = [
        row for row in fold_rows
        if bool((row["gate_certificate"])["stable_beyond_float32_roundoff"])
    ]
    unstable_rows = [
        row for row in fold_rows
        if not bool((row["gate_certificate"])["stable_beyond_float32_roundoff"])
    ]
    stable = _calibration_bin(
        stable_rows,
        confidence_class="FLOAT32_STABLE",
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    unstable = _calibration_bin(
        unstable_rows,
        confidence_class="WITHIN_FLOAT32_ROUNDOFF",
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )

    alpha = prior_alpha
    beta = prior_beta
    sequential: list[dict] = []
    for row in fold_rows:
        match = bool(row["route_matches_oracle"])
        alpha += float(match)
        beta += float(not match)
        mean, std = _beta_summary(alpha, beta)
        confidence_class = (
            "FLOAT32_STABLE"
            if bool((row["gate_certificate"])["stable_beyond_float32_roundoff"])
            else "WITHIN_FLOAT32_ROUNDOFF"
        )
        sequential.append(SequentialRouterPosteriorStep(
            epoch=float(row["epoch"]),
            confidence_class=confidence_class,
            route_matches_oracle=match,
            posterior_alpha=alpha,
            posterior_beta=beta,
            posterior_match_probability=mean,
            posterior_std=std,
        ).to_dict())

    stable_match_rate = stable.matches / stable.n_folds if stable.n_folds else float("nan")
    unstable_match_rate = (
        unstable.matches / unstable.n_folds if unstable.n_folds else float("nan"))
    match_gap = stable_match_rate - unstable_match_rate
    error_gap = stable.mean_dispatcher_error - unstable.mean_dispatcher_error
    if not stable_rows or not unstable_rows:
        status = "CALIBRATION_UNIDENTIFIABLE"
    elif match_gap > 0.0 and error_gap < 0.0:
        status = "CALIBRATED_DIRECTIONALLY_INSTANCE"
    elif match_gap == 0.0 or error_gap == 0.0:
        status = "CALIBRATION_INCONCLUSIVE_INSTANCE"
    else:
        status = "MIS_CALIBRATED_INSTANCE"

    allocations: list[dict] = []
    for row in fold_rows:
        stable_margin = bool(
            (row["gate_certificate"])["stable_beyond_float32_roundoff"])
        confidence_class = (
            "FLOAT32_STABLE" if stable_margin else "WITHIN_FLOAT32_ROUNDOFF")
        selected = str(row["tool"])
        needs_shadow = status != "CALIBRATED_DIRECTIONALLY_INSTANCE" or not stable_margin
        candidates = ((selected,) if not needs_shadow else
                      tuple(dict.fromkeys((selected, str(fallback_architecture)))))
        allocations.append(ForecastComputeAllocation(
            epoch=float(row["epoch"]),
            selected_tool=selected,
            confidence_class=confidence_class,
            requested_k=len(candidates),
            candidate_tools=candidates,
            mode=("K2_SHADOW_COMPARE_NO_ACTUATION" if len(candidates) > 1
                  else "K1_ROUTER_NO_ACTUATION"),
            rationale=(
                "margin confidence is not directionally calibrated on this instance; "
                "measure the architectural incumbent from the same checkpoint"
                if status != "CALIBRATED_DIRECTIONALLY_INSTANCE"
                else (
                    "gate is within float32 roundoff; measure the architectural incumbent "
                    "from the same checkpoint"
                    if not stable_margin else
                    "directionally calibrated stable gate; retain K=1 advisory route"
                )
            ),
        ).to_dict())

    terminal_mean, terminal_std = _beta_summary(alpha, beta)
    allocation_verdict = (
        "K2_SHADOW_A_RIDGE_SOLVE_OWED_FOR_ALL_FOLDS; CURRENT_ROUTE_UNCHANGED"
        if status != "CALIBRATED_DIRECTIONALLY_INSTANCE"
        else "K2_SHADOW_ONLY_WITHIN_FLOAT32_ROUNDOFF; CURRENT_ROUTE_UNCHANGED"
    )
    return RouterForecastCalibrationReport(
        status=status,
        posterior_kind="SEQUENTIAL_BETA_BERNOULLI_ROUTE_MATCH_PATH",
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        prior_provenance=str(prior_provenance),
        n_folds=len(fold_rows),
        terminal_posterior_alpha=alpha,
        terminal_posterior_beta=beta,
        terminal_posterior_match_probability=terminal_mean,
        terminal_posterior_std=terminal_std,
        stable_bin=stable.to_dict(),
        roundoff_unstable_bin=unstable.to_dict(),
        high_minus_low_match_rate=match_gap,
        high_minus_low_mean_error=error_gap,
        sequential_posterior=tuple(sequential),
        compute_allocations=tuple(allocations),
        allocation_verdict=allocation_verdict,
        blocker=(
            "one trajectory; confidence direction fails on #205; full trajectory posterior "
            "and live-distribution calibration require independent trajectories plus "
            "BLOCKED_DISTRIBUTION_CUSTODY closure"
        ),
    )


def importance_weighted_architecture_eval(
    fold_rows: Sequence[Mapping[str, object]],
    *,
    custody: RegimeDensityCustody | None,
    clip_bounds: tuple[float, float] | None,
    support_mask: Sequence[bool] | None = None,
) -> ImportanceWeightedArchitectureReport:
    """Evaluate per-arm fold error under a visited-live regime distribution.

    Each row must carry ``regime`` and ``per_arm_err``.  The selected architecture is
    the minimum weighted MAE with a lexical tie break, making the discrete result
    deterministic after the fp32 weighting step.
    """
    if not fold_rows:
        raise ValueError("fold_rows must be non-empty")
    regimes = [str(row["regime"]) for row in fold_rows]
    arm_sets = [set((row.get("per_arm_err") or {}).keys()) for row in fold_rows]
    if not arm_sets or not set.intersection(*arm_sets):
        raise ValueError("fold rows have no common per_arm_err architectures")
    arms = sorted(set.intersection(*arm_sets))
    unweighted = {
        arm: float(np.mean(np.asarray(
            [float((row["per_arm_err"])[arm]) for row in fold_rows], dtype=np.float32)))
        for arm in arms
    }
    try:
        diagnostics = self_normalized_clipped_masked_weights(
            regimes, custody=custody, clip_bounds=clip_bounds, support_mask=support_mask)
    except DistributionCustodyError as exc:
        return ImportanceWeightedArchitectureReport(
            status=BLOCKED_DISTRIBUTION_CUSTODY,
            per_arch_weighted_mae={},
            per_arch_unweighted_mae=unweighted,
            selected_arch=None,
            diagnostics=None,
            blocker=str(exc),
        )
    weights = np.asarray(diagnostics.normalized_weights, dtype=np.float32)
    mask = np.asarray(diagnostics.support_mask, dtype=bool)
    weighted: dict[str, float] = {}
    for arm in arms:
        errors = np.asarray(
            [float((row["per_arm_err"])[arm]) for row in fold_rows], dtype=np.float32)
        weighted[arm] = float(np.sum(errors[mask] * weights[mask], dtype=np.float32)
                              / np.sum(weights[mask], dtype=np.float32))
    selected = min(weighted, key=lambda arm: (weighted[arm], arm))
    return ImportanceWeightedArchitectureReport(
        status="IS_WEIGHTED",
        per_arch_weighted_mae=weighted,
        per_arch_unweighted_mae=unweighted,
        selected_arch=selected,
        diagnostics=diagnostics.to_dict(),
        blocker=None,
    )


def regime_density_from_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Count and normalize explicit regime rows (helper; hashes remain caller-owned)."""
    counts: dict[str, int] = {}
    for row in rows:
        regime = str(row["regime"])
        counts[regime] = counts.get(regime, 0) + 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no regime rows")
    return {k: counts[k] / total for k in sorted(counts)}


__all__ = [
    "AXIS_TAG",
    "BLOCKED_DISTRIBUTION_CUSTODY",
    "DistributionCustodyError",
    "ForecastComputeAllocation",
    "ImportanceWeightDiagnostics",
    "ImportanceWeightedArchitectureReport",
    "RegimeDensityCustody",
    "RouterCalibrationBin",
    "RouterDecisionRecord",
    "RouterForecastCalibrationReport",
    "RouterGateCertificate",
    "RouterReplayError",
    "RouterReplayMismatchError",
    "RouterReplayOutcome",
    "SequentialRouterPosteriorStep",
    "append_decide_record",
    "calibrate_router_forecast",
    "certify_fp32_gate",
    "importance_weighted_architecture_eval",
    "load_replay_events",
    "make_decision_record",
    "regime_density_from_rows",
    "replay_apply",
    "self_normalized_clipped_masked_weights",
]
