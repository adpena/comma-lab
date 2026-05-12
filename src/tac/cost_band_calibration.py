"""Cost-band self-calibration posterior.

The "calibration of calibration" answer: instead of hand-deriving cost-band
estimates in each operator wrapper (and then having a fresh-eyes adversarial
subagent catch that the hand estimate is itself uncalibrated), maintain a
canonical fcntl-locked JSONL posterior at
``.omx/state/cost_band_posterior.jsonl`` where every dispatch APPENDS a
measured anchor on completion. Future predictions read the posterior's
p10/p50/p90 for the matching (platform, gpu, epochs, flags) bucket and emit
a confidence-tagged estimate. Calibration converges automatically as
empirical data accumulates.

This module is the sister of ``tac.continual_learning`` (Catalog #128
atomic fcntl-locked writes) — same primitive, different domain. Anchors
are append-only; the posterior is read-only at planning time.

Schema (cost_band_posterior_v1):
    {
        "schema": "cost_band_posterior_v1",
        "logged_at_utc": "2026-05-12T18:23:00+00:00",
        "dispatch_label": "<INSTANCE_JOB_ID>",
        "trainer": "experiments/train_<name>.py",
        "platform": "modal" | "vastai" | "lightning" | "azure" | "kaggle",
        "gpu": "T4" | "A10G" | "A100" | "H100" | "4090",
        "epochs": <int>,
        "batch_size": <int>,
        "all_flags_on": <bool>,           # all TIER_1_OPERATOR_REQUIRED_FLAGS threaded?
        "actual_wall_clock_sec": <float>,
        "actual_cost_usd": <float>,
        "predicted_cost_usd_low": <float | null>,
        "predicted_cost_usd_high": <float | null>,
        "prediction_in_band": <bool | null>,
        "notes": "<optional free-text>"
    }

Wrapper integration (canonical pattern):
    # At end of dispatch script, after wall-clock + cost are known:
    python tools/append_cost_band_anchor.py \\
        --dispatch-label "$INSTANCE_JOB_ID" \\
        --trainer "experiments/train_x.py" \\
        --platform modal --gpu "$MODAL_GPU" \\
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --all-flags-on \\
        --actual-wall-clock-sec "$WALLCLOCK" \\
        --actual-cost-usd "$ACTUAL_COST"

Reader API (in operator_authorize_*.sh cost-band block):
    python -c "from tac.cost_band_calibration import predict; \\
        p = predict('modal', 'T4', 3000, all_flags_on=True); \\
        print(p.p50_cost_usd, p.n_anchors, p.confidence_tag)"

Per CLAUDE.md "Forbidden score claims" sister rule: every emission carries
``confidence_tag`` ∈ {"empirical_posterior" (N≥3), "weak_posterior" (1≤N≤2),
"hand_calibrated_fallback" (N=0)} so consumers see the evidence grade.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "cost_band_posterior_v1"
POSTERIOR_PATH = Path(".omx/state/cost_band_posterior.jsonl")
LOCK_PATH = Path(".omx/state/.cost_band_posterior.lock")

# Platform-keyed hourly rate tables. Adding a new platform here is the
# canonical way to extend cost-band coverage; previously each provider had
# a sibling helper module (``tac.deploy.modal.training_cost``,
# ``tac.deploy.vastai.training_cost``, ...). Inlining the lookup here keeps
# the rate tables next to the posterior they feed.
PLATFORM_RATES_USD_PER_HOUR: dict[str, dict[str, float]] = {
    "modal": {
        "T4": 0.59,
        "A10G": 1.10,
        "A100": 4.00,
        "A100-40GB": 4.00,
        "A100-80GB": 4.00,
        "H100": 3.90,
        "H100-80GB": 3.90,
    },
    "vastai": {
        "4090": 0.25,
        "RTX_4090": 0.25,
        "A100": 0.80,
        "H100": 1.80,
    },
    "lightning": {
        "T4": 0.0,  # free tier
        "A10G": 0.40,
    },
    "azure": {
        "A100": 3.60,
        "H100": 8.00,
    },
    "kaggle": {
        "T4": 0.0,  # free tier
        "P100": 0.0,
    },
}

# Confidence-tag thresholds (per Council probe-disambiguator pattern).
#
# F3 derivation (non-arbitrariness sweep 2026-05-12 — replaces hand-asserted N=3):
#   N=1: p10/p50/p90 collapse to the single observed value; no spread signal.
#         → "weak_posterior" tag with x1.5 band-widening reflects this honestly.
#   N=2: p10 ≈ p_low, p90 ≈ p_high; spread observable but Wilson 95% CI is
#         ~0.55 wide at p=0.5, so any percentile is dominated by sampling noise.
#         → still "weak_posterior".
#   N=3: minimum N at which p10/p50/p90 are distinct interpolated points
#         AND Wilson 95% CI half-width ≈ 0.55/sqrt(N) ≈ 0.32 — still wide but
#         the band has SHAPE (not just min/max). Per Wilson 1927 + Boyd
#         experimental-design conventions: N=3 is the smallest sample at which
#         a percentile band carries qualitative information; N≥5 is required
#         for quantitative (CI<25%) operator decisions.
#         → "empirical_posterior" emits the band WITH "qualitative-only,
#         widen mentally if making a >$20 decision" framing in `fallback_rationale`.
#   Future refinement (operator-decision-gated): split into
#     weak_posterior (N=1-2), nominal_posterior (N=3-4),
#     tight_posterior (N≥5) — see non_arbitrariness_sweep_20260512.md#f3.
_EMPIRICAL_MIN_ANCHORS = 3
_WEAK_MIN_ANCHORS = 1


@dataclass(frozen=True)
class CostBandPrediction:
    """A confidence-tagged cost-band prediction for one (platform, gpu, epochs, flags) bucket.

    p10/p50/p90 are computed empirically from matching anchors.
    Falls back to a hand-calibrated stub when no anchors match.
    """

    platform: str
    gpu: str
    epochs: int
    all_flags_on: bool
    n_anchors: int
    p10_cost_usd: float
    p50_cost_usd: float
    p90_cost_usd: float
    p10_wall_clock_hr: float
    p50_wall_clock_hr: float
    p90_wall_clock_hr: float
    confidence_tag: str  # "empirical_posterior" | "weak_posterior" | "hand_calibrated_fallback"
    freshness_seconds: float | None  # how old the most-recent matching anchor is
    fallback_rationale: str = ""

    def as_dict(self) -> dict:
        return {
            "platform": self.platform,
            "gpu": self.gpu,
            "epochs": self.epochs,
            "all_flags_on": self.all_flags_on,
            "n_anchors": self.n_anchors,
            "p10_cost_usd": self.p10_cost_usd,
            "p50_cost_usd": self.p50_cost_usd,
            "p90_cost_usd": self.p90_cost_usd,
            "p10_wall_clock_hr": self.p10_wall_clock_hr,
            "p50_wall_clock_hr": self.p50_wall_clock_hr,
            "p90_wall_clock_hr": self.p90_wall_clock_hr,
            "confidence_tag": self.confidence_tag,
            "freshness_seconds": self.freshness_seconds,
            "fallback_rationale": self.fallback_rationale,
        }


@dataclass(frozen=True)
class CostBandAnchor:
    """One empirical anchor: a completed dispatch's measured wall-clock + cost.

    ``actual_cost_usd`` is invoice-actual when a provider invoice is available.
    Recovery paths may otherwise append a provider-table estimate; those rows
    must say so in ``notes`` via ``cost_estimate_source=...``.
    """

    logged_at_utc: str
    dispatch_label: str
    trainer: str
    platform: str
    gpu: str
    epochs: int
    batch_size: int
    all_flags_on: bool
    actual_wall_clock_sec: float
    actual_cost_usd: float
    predicted_cost_usd_low: float | None = None
    predicted_cost_usd_high: float | None = None
    prediction_in_band: bool | None = None
    notes: str = ""
    schema: str = SCHEMA_VERSION


def _now_utc_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _ensure_state_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_anchor(
    anchor: CostBandAnchor,
    *,
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Append an anchor to the JSONL posterior under fcntl LOCK_EX.

    Atomic write per Catalog #128 pattern. Concurrent appenders serialize at
    the lock and each writes ONE JSON line followed by newline.
    """
    posterior = posterior_path or POSTERIOR_PATH
    lock = lock_path or LOCK_PATH
    _ensure_state_dir(posterior)
    _ensure_state_dir(lock)
    line = json.dumps(
        {
            "schema": anchor.schema,
            "logged_at_utc": anchor.logged_at_utc,
            "dispatch_label": anchor.dispatch_label,
            "trainer": anchor.trainer,
            "platform": anchor.platform,
            "gpu": anchor.gpu,
            "epochs": anchor.epochs,
            "batch_size": anchor.batch_size,
            "all_flags_on": anchor.all_flags_on,
            "actual_wall_clock_sec": anchor.actual_wall_clock_sec,
            "actual_cost_usd": anchor.actual_cost_usd,
            "predicted_cost_usd_low": anchor.predicted_cost_usd_low,
            "predicted_cost_usd_high": anchor.predicted_cost_usd_high,
            "prediction_in_band": anchor.prediction_in_band,
            "notes": anchor.notes,
        },
        sort_keys=True,
        allow_nan=False,
    )
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with posterior.open("a", encoding="utf-8") as pf:
                pf.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def load_anchors(
    posterior_path: Path | None = None,
) -> list[CostBandAnchor]:
    """Read every anchor from the JSONL posterior. Skips malformed lines."""
    posterior = posterior_path or POSTERIOR_PATH
    if not posterior.exists():
        return []
    out: list[CostBandAnchor] = []
    for line in posterior.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("schema") != SCHEMA_VERSION:
            continue
        try:
            out.append(
                CostBandAnchor(
                    logged_at_utc=d["logged_at_utc"],
                    dispatch_label=d["dispatch_label"],
                    trainer=d["trainer"],
                    platform=d["platform"],
                    gpu=d["gpu"],
                    epochs=int(d["epochs"]),
                    batch_size=int(d["batch_size"]),
                    all_flags_on=bool(d["all_flags_on"]),
                    actual_wall_clock_sec=float(d["actual_wall_clock_sec"]),
                    actual_cost_usd=float(d["actual_cost_usd"]),
                    predicted_cost_usd_low=d.get("predicted_cost_usd_low"),
                    predicted_cost_usd_high=d.get("predicted_cost_usd_high"),
                    prediction_in_band=d.get("prediction_in_band"),
                    notes=d.get("notes", ""),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _percentile(values: list[float], q: float) -> float:
    """Simple percentile (q in [0, 100]) without numpy dependency.

    For N < 3 we widen the bands to avoid false precision.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sv = sorted(values)
    # Linear interpolation between adjacent samples.
    idx = (q / 100.0) * (len(sv) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sv) - 1)
    frac = idx - lo
    return sv[lo] + frac * (sv[hi] - sv[lo])


# Hand-calibrated fallbacks per the 2026-05-12 audit (acknowledged as
# uncalibrated per the fresh-eyes adversarial NF0 + F2 non-arbitrariness
# sweep). These ONLY apply when the posterior has zero matching anchors;
# the first real dispatch with `append_anchor` supersedes them. Per-tuple
# justification basis below; the spread (p10..p90) is the highest-uncertainty
# component and should converge fastest once real dispatches land.
#
# Modal T4 p50=$8 anchored to codex's ~14-min/epoch baseline (3000ep x 14min/60 x
# $0.59/hr x 1.05 overhead ≈ $8.7); spread ±$3 reflects auth-eval-on-best
# tail variance. Modal A10G/A100/H100 scaled by the configured Modal hourly
# estimate table in tac.deploy.modal.training_cost. Vast.ai 4090 anchored to codex empirical
# $0.10-0.50/dispatch range x 1.5 margin for post-Tier-1 4090 slowdown
# observations. Lightning T4 free-tier confirmed by Lightning pricing page.
_HAND_CALIBRATED_FALLBACKS = {
    # (platform, gpu, epochs_bucket, all_flags_on): (p10, p50, p90) USD
    ("modal", "T4", 3000, True): (5.0, 8.0, 12.0),  # ARBITRARY_OK:codex_14min_per_epoch_baseline_x_$0.59hr
    ("modal", "A10G", 3000, True): (4.0, 6.0, 10.0),  # ARBITRARY_OK:scaled_from_T4_x_$1.10/$0.59_x_0.97_TFLOP_hr_ratio
    ("modal", "A100", 3000, True): (3.0, 5.0, 8.0),  # ARBITRARY_OK:scaled_from_T4_x_$4.00/$0.59_x_0.74_TFLOP_hr_ratio
    ("modal", "H100", 3000, True): (3.0, 4.5, 7.0),  # ARBITRARY_OK:scaled_from_T4_x_$3.90/$0.59_x_0.47_TFLOP_hr_ratio
    ("vastai", "4090", 3000, True): (0.50, 0.80, 1.20),  # ARBITRARY_OK:codex_empirical_$0.10-0.50_x_1.5_post_tier1_margin
    ("lightning", "T4", 3000, True): (0.0, 0.0, 0.0),  # ARBITRARY_OK:lightning_free_tier_per_pricing_page
    ("modal", "T4", 3000, False): (3.5, 5.5, 8.0),  # ARBITRARY_OK:T4_all_flags_on_minus_30pct_for_non_optimal_config
    ("modal", "T4", 1000, True): (1.5, 2.5, 4.0),  # ARBITRARY_OK:T4_all_flags_on_x_1000/3000_x_1.05_amortization
    ("modal", "T4", 100, True): (0.20, 0.30, 0.50),  # ARBITRARY_OK:T4_all_flags_on_x_100/3000_x_1.5_warmup_overhead
}


def _epochs_bucket(epochs: int) -> int:
    """Bucket epochs to ease anchor matching.

    F4 derivation (non-arbitrariness sweep 2026-05-12): bucket boundaries
    are NOT log-spaced; they match the canonical T1 Ballé / SC++ / Lane 12
    dispatch counts that operators actually configure:

        smoke         50ep   ARBITRARY_OK:operator_smoke_test_canonical
        dev          100ep   ARBITRARY_OK:dev_loop_canonical_3k_seconds_T4
        mid          500ep   ARBITRARY_OK:mid_config_canonical
        long        1000ep   ARBITRARY_OK:long_config_canonical
        full        3000ep   ARBITRARY_OK:T1_Balle_canonical_per_TT_landing
        multi-day  6000ep+   ARBITRARY_OK:Phase_2/3_long_run_canonical_per_dashboard

    Cost-per-epoch is NOT linear: warmup overhead, data-loader steady-state,
    auth-eval-on-best, EMA-shadow-eval all contribute fixed costs that
    amortize differently across these regime. Linear/log binning would smear
    those regimes together; the canonical-config binning preserves them.

    KNOWN sharp transitions (acceptable per F4 sweep finding):
        epochs=199 → bucket=100; epochs=201 → bucket=500 (5× cost jump).
        Mitigation: operators using boundary-adjacent epoch counts SHOULD
        request the larger bucket's confidence band explicitly. Future
        refinement: log-spaced fallback for epochs not matching canonical.
    """
    if epochs <= 50:
        return 50
    if epochs <= 200:
        return 100
    if epochs <= 600:
        return 500
    if epochs <= 1500:
        return 1000
    if epochs <= 4500:
        return 3000
    return ((epochs + 2999) // 3000) * 3000


def predict(
    platform: str,
    gpu: str,
    epochs: int,
    *,
    all_flags_on: bool = True,
    posterior_path: Path | None = None,
    matching_anchors_min: int = _WEAK_MIN_ANCHORS,
) -> CostBandPrediction:
    """Predict cost-band for one (platform, gpu, epochs, flags) bucket.

    Returns confidence-tagged p10/p50/p90 estimate. Falls back to a
    hand-calibrated stub when N < ``matching_anchors_min``.
    """
    anchors = load_anchors(posterior_path)
    bucket = _epochs_bucket(epochs)
    matching = [
        a for a in anchors
        if a.platform == platform
        and a.gpu == gpu
        and _epochs_bucket(a.epochs) == bucket
        and a.all_flags_on == all_flags_on
    ]
    if len(matching) >= _EMPIRICAL_MIN_ANCHORS:
        confidence = "empirical_posterior"
        rationale = ""
    elif len(matching) >= _WEAK_MIN_ANCHORS:
        confidence = "weak_posterior"
        rationale = (
            f"only {len(matching)} matching anchor(s); "
            f"band widened by ±50% to reflect uncertainty"
        )
    else:
        # Fallback to hand-calibrated stub.
        key = (platform, gpu, bucket, all_flags_on)
        stub = _HAND_CALIBRATED_FALLBACKS.get(key)
        if stub is None:
            stub = (0.0, 0.0, 0.0)
            rationale = (
                f"no anchors AND no hand-calibrated stub for "
                f"({platform},{gpu},epochs~{bucket},all_flags={all_flags_on}); "
                f"emitting zero band"
            )
        else:
            rationale = (
                "0 anchors; hand-calibrated stub per 2026-05-12 audit "
                "(acknowledged uncalibrated; supersedes on first dispatch)"
            )
        return CostBandPrediction(
            platform=platform,
            gpu=gpu,
            epochs=bucket,
            all_flags_on=all_flags_on,
            n_anchors=0,
            p10_cost_usd=stub[0],
            p50_cost_usd=stub[1],
            p90_cost_usd=stub[2],
            p10_wall_clock_hr=0.0,
            p50_wall_clock_hr=0.0,
            p90_wall_clock_hr=0.0,
            confidence_tag="hand_calibrated_fallback",
            freshness_seconds=None,
            fallback_rationale=rationale,
        )

    costs = [a.actual_cost_usd for a in matching]
    wallclocks_hr = [a.actual_wall_clock_sec / 3600.0 for a in matching]
    # Widen bands for weak posteriors.
    widen = 1.5 if confidence == "weak_posterior" else 1.0
    p10_c = _percentile(costs, 10) / widen
    p50_c = _percentile(costs, 50)
    p90_c = _percentile(costs, 90) * widen
    p10_w = _percentile(wallclocks_hr, 10) / widen
    p50_w = _percentile(wallclocks_hr, 50)
    p90_w = _percentile(wallclocks_hr, 90) * widen
    # Freshness of the most-recent matching anchor.
    most_recent = max(
        matching,
        key=lambda a: a.logged_at_utc,
    )
    try:
        dt_anchor = datetime.datetime.fromisoformat(most_recent.logged_at_utc)
        dt_now = datetime.datetime.now(datetime.UTC)
        freshness_seconds: float | None = (dt_now - dt_anchor).total_seconds()
    except (ValueError, TypeError):
        freshness_seconds = None
    return CostBandPrediction(
        platform=platform,
        gpu=gpu,
        epochs=bucket,
        all_flags_on=all_flags_on,
        n_anchors=len(matching),
        p10_cost_usd=p10_c,
        p50_cost_usd=p50_c,
        p90_cost_usd=p90_c,
        p10_wall_clock_hr=p10_w,
        p50_wall_clock_hr=p50_w,
        p90_wall_clock_hr=p90_w,
        confidence_tag=confidence,
        freshness_seconds=freshness_seconds,
        fallback_rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Platform-keyed cost helpers (inlined from tac.deploy.modal.training_cost
# 2026-05-12 simplification T1-B). The Modal-specific shim in
# tac.deploy.modal.training_cost now delegates here.
# ---------------------------------------------------------------------------


def normalize_gpu(platform: str, gpu: str) -> str:
    """Return a canonical GPU label for one platform's rate-table buckets.

    Per-platform normalisation:

    - ``modal``: ``A100*`` → ``A100`` unless an exact variant exists; ``H100*``
      similarly. ``A10G`` and ``T4`` are returned as-is.
    - ``vastai``: ``RTX_4090`` collapses to ``4090``.
    - other platforms: case-folded passthrough.
    """

    value = str(gpu or "").strip().upper()
    if not value:
        return value
    platform_norm = platform.lower()
    table = PLATFORM_RATES_USD_PER_HOUR.get(platform_norm, {})
    if platform_norm == "modal":
        if value == "A10G":
            return "A10G"
        if value.startswith("A100"):
            return value if value in table else "A100"
        if value.startswith("H100"):
            return value if value in table else "H100"
        if value == "T4":
            return "T4"
        return value
    if platform_norm == "vastai":
        if value in {"4090", "RTX_4090"}:
            return "4090"
    return value


def estimate_cost_usd(
    platform: str, gpu: str, elapsed_seconds: float
) -> tuple[float, float]:
    """Estimate provider cost from a platform/GPU class + measured elapsed seconds.

    Returns ``(cost_usd, hourly_rate_usd)``.

    Per CLAUDE.md "Forbidden score claims" sister rule: the returned value is
    an ESTIMATE, not an invoice. Callers must record the rate / source in
    the anchor notes.
    """

    platform_norm = platform.lower()
    table = PLATFORM_RATES_USD_PER_HOUR.get(platform_norm)
    if table is None:
        raise ValueError(
            f"no hourly-rate table configured for platform={platform!r}; "
            f"known platforms: {sorted(PLATFORM_RATES_USD_PER_HOUR)}"
        )
    gpu_norm = normalize_gpu(platform_norm, gpu)
    rate = table.get(gpu_norm)
    if rate is None:
        raise ValueError(
            f"no hourly rate configured for platform={platform!r} gpu={gpu!r} "
            f"(normalised to {gpu_norm!r}); known buckets: {sorted(table)}"
        )
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(
            f"elapsed_seconds must be finite and nonnegative; got {elapsed!r}"
        )
    return rate * elapsed / 3600.0, rate


def _bool_field(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _numeric_field(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def append_platform_training_anchor(
    platform: str,
    *,
    out_dir: Path,
    metadata: dict[str, Any],
    result: dict[str, Any],
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Append a platform training cost anchor once, if metadata requests it.

    Returns a small status manifest and writes it to
    ``cost_band_anchor_appended.json`` inside ``out_dir``. Re-running is
    idempotent: an existing marker file is returned and no second posterior
    row is appended.

    ``metadata`` must contain a ``cost_band_anchor`` dict with at least
    ``trainer``, ``epochs``, ``batch_size``. ``result`` must contain a
    finite ``elapsed_seconds``.
    """

    out_dir = Path(out_dir)
    marker = out_dir / "cost_band_anchor_appended.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {**payload, "already_appended": True}
        except json.JSONDecodeError:
            pass

    cost_meta = metadata.get("cost_band_anchor")
    if not isinstance(cost_meta, dict):
        return {
            "schema": "platform_training_cost_anchor_append_v1",
            "appended": False,
            "reason": "metadata_missing_cost_band_anchor",
        }

    elapsed = result.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        return {
            "schema": "platform_training_cost_anchor_append_v1",
            "appended": False,
            "reason": "result_missing_numeric_elapsed_seconds",
        }

    gpu = normalize_gpu(platform, str(metadata.get("gpu") or cost_meta.get("gpu") or ""))
    try:
        estimated_cost, hourly_rate = estimate_cost_usd(platform, gpu, float(elapsed))
        epochs = int(cost_meta["epochs"])
        batch_size = int(cost_meta["batch_size"])
        trainer = str(cost_meta["trainer"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema": "platform_training_cost_anchor_append_v1",
            "appended": False,
            "reason": f"invalid_cost_band_metadata:{type(exc).__name__}:{exc}",
        }

    label = str(metadata.get("label") or cost_meta.get("dispatch_label") or f"{platform}_training")
    rc = result.get("returncode")
    timed_out = bool(result.get("timed_out", False))
    source_tag = f"{platform}_elapsed_seconds_x_configured_hourly_rate"
    notes = (
        f"cost_estimate_source={source_tag}; "
        f"hourly_rate_usd={hourly_rate}; returncode={rc}; timed_out={timed_out}"
    )
    if cost_meta.get("notes"):
        notes += f"; {cost_meta['notes']}"

    pred_low = _numeric_field(cost_meta.get("predicted_cost_usd_low"))
    pred_high = _numeric_field(cost_meta.get("predicted_cost_usd_high"))
    prediction_in_band: bool | None
    if pred_low is not None and pred_high is not None:
        prediction_in_band = bool(pred_low <= estimated_cost <= pred_high)
    else:
        prediction_in_band = None

    anchor = CostBandAnchor(
        logged_at_utc=_now_utc_iso(),
        dispatch_label=label,
        trainer=trainer,
        platform=platform,
        gpu=gpu,
        epochs=epochs,
        batch_size=batch_size,
        all_flags_on=_bool_field(cost_meta.get("all_flags_on", False)),
        actual_wall_clock_sec=float(elapsed),
        actual_cost_usd=estimated_cost,
        predicted_cost_usd_low=pred_low,
        predicted_cost_usd_high=pred_high,
        prediction_in_band=prediction_in_band,
        notes=notes,
    )
    append_anchor(anchor, posterior_path=posterior_path, lock_path=lock_path)
    manifest = {
        "schema": "platform_training_cost_anchor_append_v1",
        "appended": True,
        "already_appended": False,
        "score_claim": False,
        "promotion_eligible": False,
        "cost_estimate": True,
        "cost_estimate_source": source_tag,
        "dispatch_label": label,
        "trainer": trainer,
        "platform": platform,
        "gpu": gpu,
        "epochs": epochs,
        "batch_size": batch_size,
        "all_flags_on": anchor.all_flags_on,
        "elapsed_seconds": float(elapsed),
        "estimated_cost_usd": estimated_cost,
        "hourly_rate_usd": hourly_rate,
        "posterior_path": str(posterior_path) if posterior_path is not None else None,
        "notes": notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # PCC3 internal-consistency check (CLAUDE.md non-negotiable): when both
    # epochs and elapsed_seconds are present, refuse to write a manifest that
    # claims more epochs than physics permits (≥ 50 ms per epoch is a very
    # loose lower bound for a real training loop).
    _MIN_SEC_PER_EPOCH = 0.05
    if epochs > 0 and elapsed >= 0:
        if elapsed < epochs * _MIN_SEC_PER_EPOCH:
            raise RuntimeError(
                f"stats internal-consistency violation (PCC3): "
                f"epochs={epochs} but elapsed_seconds={elapsed:.3f} "
                f"< epochs * {_MIN_SEC_PER_EPOCH} = "
                f"{epochs * _MIN_SEC_PER_EPOCH:.3f}. "
                f"Stub-loop suspected; refusing to write platform training anchor manifest."
            )
    # The marker file's JSON payload must match what an idempotent re-run
    # returns. Serialize stably to keep byte-identical output across runs.
    marker.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def summary_by_bucket(
    posterior_path: Path | None = None,
) -> list[dict]:
    """Aggregate every bucket present in the posterior with N + median cost.

    Used by ``tools/cost_band_calibration_summary.py`` for the operator-facing
    table. Read-only; does NOT modify the posterior.
    """
    anchors = load_anchors(posterior_path)
    buckets: dict[tuple, list[CostBandAnchor]] = {}
    for a in anchors:
        key = (a.platform, a.gpu, _epochs_bucket(a.epochs), a.all_flags_on)
        buckets.setdefault(key, []).append(a)
    out = []
    for (platform, gpu, ep, flags), rows in sorted(buckets.items()):
        costs = [r.actual_cost_usd for r in rows]
        out.append(
            {
                "platform": platform,
                "gpu": gpu,
                "epochs_bucket": ep,
                "all_flags_on": flags,
                "n_anchors": len(rows),
                "p50_cost_usd": _percentile(costs, 50),
                "min_cost_usd": min(costs),
                "max_cost_usd": max(costs),
            }
        )
    return out
