# SPDX-License-Identifier: MIT
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
        "outcome": "successful_dispatch" | "failed_dispatch" | "timed_out" | "harvested_partial",
        "returncode": <int | null>,       # subprocess returncode (informational; outcome is authoritative)
        "notes": "<optional free-text>"
    }

NV7 anchor-outcome discipline (review-omni 2026-05-12):
    `predict()` excludes anchors with ``outcome != "successful_dispatch"`` by
    default. A failed dispatch (e.g. fc-01KREXK209TRX7ED5ZRVXHY1VT 14.77-sec
    rc=1 from WWW4) measures the CRASH wall-clock, not the training wall-clock;
    folding it into the percentile band underestimates real cost by 400-750x.
    Failed anchors are still retained in the posterior for forensic audit but
    callers must pass ``include_failed=True`` to opt them in. Anchors without
    an explicit ``outcome`` field default to ``"successful_dispatch"`` for
    backward compatibility with pre-NV7 rows; the migration tool
    ``tools/migrate_cost_band_posterior_failed_anchors.py`` tags historical
    failed rows by inspecting ``notes`` for ``returncode=<nonzero>`` markers.

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
import enum
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
    "github": {
        "CPU": 0.0,  # public-repo GHA minutes used for contest-CPU harvests
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


def parse_actual_cost_usd(value: str | None, *, field_name: str = "actual_cost_usd") -> float | None:
    """Parse a measured cost value, returning ``None`` when absent.

    Absence means "do not append a cost-band anchor"; it is not equivalent to a
    measured zero-dollar run.
    """

    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a finite non-negative float; got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative float; got {value!r}")
    return parsed


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


SUCCESSFUL_DISPATCH = "successful_dispatch"
FAILED_DISPATCH = "failed_dispatch"
TIMED_OUT = "timed_out"
HARVESTED_PARTIAL = "harvested_partial"
# FIX-WAVE-2 R2-3 (2026-05-13): explicit tag for pre-NV7 rows that lack
# an ``outcome`` field. Previously the read-side coerced missing
# outcome to SUCCESSFUL_DISPATCH silently, which Ballé flagged as a
# side-information channel corruption. Now read-side tags such rows
# with this explicit outcome so the load is non-destructive but the
# downstream predict() / posterior math knows to exclude them by default
# (legacy_pre_nv7 is NOT in the default-included outcome set).
LEGACY_PRE_NV7 = "legacy_pre_nv7"

VALID_OUTCOMES = frozenset({
    SUCCESSFUL_DISPATCH,
    FAILED_DISPATCH,
    TIMED_OUT,
    HARVESTED_PARTIAL,
    LEGACY_PRE_NV7,
})


@dataclass(frozen=True)
class CostBandAnchor:
    """One empirical anchor: a completed dispatch's measured wall-clock + cost.

    ``actual_cost_usd`` is invoice-actual when a provider invoice is available.
    Recovery paths may otherwise append a provider-table estimate; those rows
    must say so in ``notes`` via ``cost_estimate_source=...``.

    NV7 (2026-05-12): ``outcome`` MUST be one of :data:`VALID_OUTCOMES`. Failed
    or timed-out anchors are retained for forensic audit but excluded from
    ``predict()`` by default. ``returncode`` is informational; the outcome
    field is the authoritative include/exclude key.
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
    outcome: str = SUCCESSFUL_DISPATCH
    returncode: int | None = None
    notes: str = ""
    schema: str = SCHEMA_VERSION
    # R4 finding Z-4.1 (2026-05-13): per-row integer schema_version for
    # future-proof migration. Legacy rows (without schema_version) are
    # implicitly schema_version=1. Subsequent schema bumps increment this
    # integer; the LEGACY_PRE_NV7 outcome sentinel disambiguates the
    # outcome-field shape; schema_version disambiguates ALL field
    # additions/removals/renames. See
    # feedback_review_zeta_r4_LANDED_20260513.md Finding Z-4.1.
    schema_version: int = 1


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
    if anchor.outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"CostBandAnchor.outcome={anchor.outcome!r} not in "
            f"{sorted(VALID_OUTCOMES)} (NV7 anchor-outcome discipline)"
        )
    line = json.dumps(
        {
            "schema": anchor.schema,
            # R4 finding Z-4.1 (2026-05-13): per-row integer schema_version.
            # Legacy rows (pre-2026-05-13) lack this field — read-side defaults
            # missing rows to 1 implicitly. Subsequent schema bumps increment.
            "schema_version": anchor.schema_version,
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
            "outcome": anchor.outcome,
            "returncode": anchor.returncode,
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
            # FIX-WAVE-2 R2-3 (2026-05-13): missing outcome is tagged
            # explicitly as ``LEGACY_PRE_NV7`` rather than silently
            # coerced to ``SUCCESSFUL_DISPATCH``. Per Ballé's review the
            # silent coercion corrupted the posterior side-information
            # channel. ``predict()`` excludes ``LEGACY_PRE_NV7`` rows by
            # default (caller may opt them in via include_legacy=True).
            # The migration tool
            # ``tools/migrate_cost_band_posterior_failed_anchors.py``
            # tags historical failed rows by inspecting ``notes`` for
            # ``returncode=<nonzero>`` markers; rows it cannot
            # classify carry the LEGACY_PRE_NV7 tag.
            outcome_raw = d.get("outcome", LEGACY_PRE_NV7)
            if outcome_raw not in VALID_OUTCOMES:
                # Refuse to materialize anchors with an unknown outcome string;
                # treat as malformed line (skip, do NOT default-coerce).
                continue
            returncode_raw = d.get("returncode")
            returncode_val: int | None = None if returncode_raw is None else int(returncode_raw)
            # R4 finding Z-4.1 (2026-05-13): legacy rows pre-2026-05-13 lack
            # `schema_version`; default to 1 to preserve backward compat.
            schema_version_raw = d.get("schema_version", 1)
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
                    outcome=outcome_raw,
                    returncode=returncode_val,
                    notes=d.get("notes", ""),
                    schema_version=int(schema_version_raw),
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
    include_failed: bool = False,
) -> CostBandPrediction:
    """Predict cost-band for one (platform, gpu, epochs, flags) bucket.

    Returns confidence-tagged p10/p50/p90 estimate. Falls back to a
    hand-calibrated stub when N < ``matching_anchors_min``.

    NV7 (2026-05-12): only ``outcome=successful_dispatch`` anchors contribute
    to the percentile band by default. Failed/timed-out/partially-harvested
    anchors stay in the posterior for forensic audit but are excluded so a
    crash-in-72-seconds doesn't underestimate the real training cost. Pass
    ``include_failed=True`` to override (e.g. for "what's the median wall-
    clock-to-crash" diagnostics).
    """
    anchors = load_anchors(posterior_path)
    bucket = _epochs_bucket(epochs)
    matching = [
        a for a in anchors
        if a.platform == platform
        and a.gpu == gpu
        and _epochs_bucket(a.epochs) == bucket
        and a.all_flags_on == all_flags_on
        and (include_failed or a.outcome == SUCCESSFUL_DISPATCH)
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
    if platform_norm == "vastai" and value in {"4090", "RTX_4090"}:
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

    def _write_marker(manifest: dict[str, Any]) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("appended") is True:
                return {**payload, "already_appended": True}
        except json.JSONDecodeError:
            pass

    cost_meta = metadata.get("cost_band_anchor")
    if not isinstance(cost_meta, dict):
        return _write_marker({
            "schema": "platform_training_cost_anchor_append_v1",
            "appended": False,
            "reason": "metadata_missing_cost_band_anchor",
            "score_claim": False,
            "promotion_eligible": False,
        })

    elapsed = result.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        return _write_marker({
            "schema": "platform_training_cost_anchor_append_v1",
            "appended": False,
            "reason": "result_missing_numeric_elapsed_seconds",
            "score_claim": False,
            "promotion_eligible": False,
        })

    gpu = normalize_gpu(platform, str(metadata.get("gpu") or cost_meta.get("gpu") or ""))
    try:
        estimated_cost, hourly_rate = estimate_cost_usd(platform, gpu, float(elapsed))
        epochs = int(cost_meta["epochs"])
        batch_size = int(cost_meta["batch_size"])
        trainer = str(cost_meta["trainer"])
    except (KeyError, TypeError, ValueError) as exc:
        return _write_marker({
            "schema": "platform_training_cost_anchor_append_v1",
            "appended": False,
            "reason": f"invalid_cost_band_metadata:{type(exc).__name__}:{exc}",
            "score_claim": False,
            "promotion_eligible": False,
        })

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

    # NV7: derive outcome from the dispatch rc + timed_out signal so future
    # predict() calls correctly exclude failed dispatches by default. Caller
    # can override via cost_meta["outcome"] for harvested partial recoveries.
    outcome_override = cost_meta.get("outcome")
    if outcome_override is not None and outcome_override in VALID_OUTCOMES:
        outcome_value = str(outcome_override)
    elif timed_out:
        outcome_value = TIMED_OUT
    elif isinstance(rc, (int, float)) and int(rc) == 0:
        outcome_value = SUCCESSFUL_DISPATCH
    elif rc is None:
        # rc absent => caller did not record subprocess exit; conservatively
        # tag as harvested_partial so the anchor doesn't poison the band.
        outcome_value = HARVESTED_PARTIAL
    else:
        outcome_value = FAILED_DISPATCH
    returncode_value: int | None = int(rc) if isinstance(rc, (int, float)) else None

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
        outcome=outcome_value,
        returncode=returncode_value,
        notes=notes,
    )
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
        "outcome": outcome_value,
        "returncode": returncode_value,
        "posterior_path": str(posterior_path) if posterior_path is not None else None,
        "notes": notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # PCC3 internal-consistency check (CLAUDE.md non-negotiable): when both
    # epochs and elapsed_seconds are present, refuse to write a manifest that
    # claims more epochs than physics permits (≥ 50 ms per epoch is a very
    # loose lower bound for a real training loop).
    #
    # NV7 (2026-05-12): PCC3 only fires on successful_dispatch outcomes. A
    # failed dispatch that crashed in 14.77 seconds is NOT a stub-loop — it
    # is a real crash anchor we want to retain for predict() to exclude.
    # PCC3 catches stub-loops that claim success; the outcome field is the
    # authoritative success signal.
    _MIN_SEC_PER_EPOCH = 0.05
    if (
        outcome_value == SUCCESSFUL_DISPATCH
        and epochs > 0
        and elapsed >= 0
        and elapsed < epochs * _MIN_SEC_PER_EPOCH
    ):
        raise RuntimeError(
            f"stats internal-consistency violation (PCC3): "
            f"epochs={epochs} but elapsed_seconds={elapsed:.3f} "
            f"< epochs * {_MIN_SEC_PER_EPOCH} = "
            f"{epochs * _MIN_SEC_PER_EPOCH:.3f}. "
            f"Stub-loop suspected; refusing to write platform training anchor manifest."
        )
    append_anchor(anchor, posterior_path=posterior_path, lock_path=lock_path)
    # The marker file's JSON payload must match what an idempotent re-run
    # returns. Serialize stably to keep byte-identical output across runs.
    return _write_marker(manifest)


def summary_by_bucket(
    posterior_path: Path | None = None,
    *,
    include_failed: bool = False,
) -> list[dict]:
    """Aggregate every bucket present in the posterior with N + median cost.

    Used by ``tools/cost_band_calibration_summary.py`` for the operator-facing
    table. Read-only; does NOT modify the posterior.

    NV7 (2026-05-12): mirrors :func:`predict` — only successful-dispatch
    anchors contribute to the summary by default; failed/timed-out anchor
    counts are surfaced in ``n_failed`` for transparency. Pass
    ``include_failed=True`` to fold them into ``p50_cost_usd``.
    """
    anchors = load_anchors(posterior_path)
    buckets: dict[tuple, list[CostBandAnchor]] = {}
    failed_buckets: dict[tuple, int] = {}
    for a in anchors:
        key = (a.platform, a.gpu, _epochs_bucket(a.epochs), a.all_flags_on)
        if a.outcome != SUCCESSFUL_DISPATCH:
            failed_buckets[key] = failed_buckets.get(key, 0) + 1
            if not include_failed:
                # Ensure failed-only buckets still surface (with n_anchors=0)
                # so operators see them in the summary report. NV7 transparency.
                buckets.setdefault(key, [])
                continue
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
                "n_failed": failed_buckets.get((platform, gpu, ep, flags), 0),
                "p50_cost_usd": _percentile(costs, 50) if costs else 0.0,
                "min_cost_usd": min(costs) if costs else 0.0,
                "max_cost_usd": max(costs) if costs else 0.0,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Per-class provider routing (council Decision 9, omnibus commit 7872c9f4b)
# ---------------------------------------------------------------------------
#
# Council Decision 9 binding verdict (PROCEED Option B, 11/11):
#
#   smoke (≤30 min, ≤$2)        -> Modal T4 default; Modal A10G fallback
#   full  (1-12h, $2-$15)       -> Vast.ai RTX 4090 default; Modal A100 fallback
#   long_burn (12h+, $50+)      -> Lightning A100 default (subscription);
#                                  Vast.ai H100 race-mode fallback
#   eval  (auth eval)           -> Modal T4 default; Modal A10G fallback
#   cpu   (contest-CPU)         -> GHA Linux x86_64 default (free)
#
# Time-Traveler amendment adopted: cost-band-posterior anchor shifts >25%
# trigger re-routing re-evaluation. Concretely the routing helper consults
# :func:`predict` for the canonical (provider, gpu) tuple AND every fallback;
# if the canonical tuple's empirical p50 is >25% higher than the cheapest
# fallback's p50 (both computed on the SAME confidence_tag floor) the helper
# emits the fallback as the recommended provider with a forensic rationale
# string describing the shift.
#
# Per CLAUDE.md "Subagent coherence-by-default" hook #4 (cathedral autopilot
# dispatch hook): the autopilot ranker can call ``select_provider_for_class``
# directly to pre-compute the provider for every queued candidate before
# dispatch.
#
# Per CLAUDE.md "Forbidden score claims" sister rule: the routing decision
# is informational; it does NOT change archive bytes, scorer outputs, or
# evaluator paths. The provider choice is a cost/wall-clock optimization,
# never a score-affecting axis.

# Dispatch-class enumeration. Adding a new class requires a council
# decision per CLAUDE.md "Design decisions — non-negotiable" because the
# canonical (provider, gpu) is the binding artifact of Decision 9.
DISPATCH_CLASSES = ("smoke", "full", "long_burn", "eval", "cpu")

# Per-class canonical (provider, gpu) tuples per Decision 9 verdict.
# Operators MUST NOT mutate this dict; new classes go through council.
CANONICAL_PROVIDER_PER_CLASS: dict[str, tuple[str, str]] = {
    "smoke": ("modal", "T4"),
    "full": ("vastai", "RTX_4090"),
    "long_burn": ("lightning", "A100"),
    "eval": ("modal", "T4"),
    "cpu": ("github", "ubuntu-latest"),
}

# BOYD-1 self-protection (Catalog #237 2026-05-15): the fallback table is
# semantically OVERLOADED. Two distinct trigger conditions historically
# shared the same data structure:
#
#   1. CHEAPER_ALTERNATIVE — Time-Traveler amendment trigger (cost-band
#      posterior shift >25%). Fallback fires automatically when canonical
#      empirical p50 exceeds fallback p50 by the threshold. Example:
#      `full` class fallback `modal/A100` is genuinely cheaper than
#      canonical `vastai/RTX_4090` for some recipes; auto-routing works.
#
#   2. CAPACITY_OVERFLOW — manual operator escalation when canonical
#      provider is saturated (Lightning A100 queue full / Modal A100 503).
#      Fallback is MORE EXPENSIVE than canonical by design. The
#      Time-Traveler amendment will NEVER fire because the cost-shift
#      inequality fails. Example: `long_burn` class fallback
#      `vastai/H100` ($1.50-1.99/hr) vs canonical `lightning/A100`
#      ($0/hr subscription) — H100 is more expensive but exists for
#      the rare race-mode operator escalation when Lightning queue is
#      saturated.
#
# Per Boyd's R2 verdict + Tao's structural finding: the SAME dict cannot
# carry both semantics without a discriminator. Future operators adding
# new dispatch classes have no signal which trigger to expect. Per
# CLAUDE.md "Beauty, simplicity": every public API must be expressive
# and unambiguous.
#
# Resolution (Option B from R2 ledger): keep the per-class semantics
# explicit via two separate dicts. Time-Traveler auto-routing reads ONLY
# `_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS`. Capacity-overflow fallbacks
# (currently only `long_burn`) live in
# `_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS` and are NEVER auto-routed —
# they require explicit operator escalation via
# :func:`select_provider_for_class(..., capacity_overflow=True)`.
#
# Cross-ref: Catalog #237 STRICT preflight gate refuses any future
# refactor that re-merges the two dicts under a single name, OR adds
# a fallback to the cheaper-alternative set whose static cost class is
# higher than the canonical for that dispatch class.
#
# Sister of Catalog #136 (custody validator concrete-tokens-only) and
# Catalog #233 / #236 (4-gate canonical structural-evidence discipline)
# — all three extinct the same META class: ambiguous data structures
# carrying multiple semantics under one name.

class FallbackReason(enum.Enum):
    """Reason a fallback fires for a dispatch class.

    CHEAPER_ALTERNATIVE: Time-Traveler amendment trigger. Fallback is
        empirically cheaper than canonical (>25% cost-band posterior shift).
        Auto-routed without operator intervention.

    CAPACITY_OVERFLOW: Manual operator escalation. Fallback is more
        expensive than canonical, used only when canonical provider is
        saturated. Time-Traveler amendment will NEVER fire for this
        reason. Requires explicit ``capacity_overflow=True`` flag in
        :func:`select_provider_for_class`.
    """

    CHEAPER_ALTERNATIVE = "cheaper_alternative"
    CAPACITY_OVERFLOW = "capacity_overflow"


# Cheaper-alternative fallbacks: Time-Traveler amendment auto-routes here
# when cost-band posterior shows >25% cost shift. Order matters: the
# routing helper tries fallbacks in order and returns the cheapest.
_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS: dict[str, list[tuple[str, str]]] = {
    "smoke": [("modal", "A10G")],
    "full": [("modal", "A100")],
    "long_burn": [],  # No cheaper alternative — A100 subscription is already free.
    "eval": [("modal", "A10G")],
    "cpu": [],  # GHA is the only canonical free CPU surface.
}

# Capacity-overflow fallbacks: manual operator escalation when canonical
# is saturated. NEVER auto-routed; requires explicit
# ``capacity_overflow=True`` argument. Cost may be HIGHER than canonical.
_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS: dict[str, list[tuple[str, str]]] = {
    "smoke": [],  # smoke recipes can wait for canonical T4 capacity.
    "full": [],  # full recipes can wait for canonical RTX_4090 capacity.
    "long_burn": [("vastai", "H100")],  # Lightning A100 saturation escalation.
    "eval": [],
    "cpu": [],
}

# Backwards-compat alias for legacy callers / tests that grep the old name.
# DEPRECATED: do not reference in new code. New consumers MUST use the
# explicit `_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS` /
# `_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS` dicts. Catalog #237 STRICT
# preflight gate refuses any new write to / re-export of this name.
# The legacy union is computed for the back-compat alias only; mutation
# is explicitly NOT supported.
FALLBACK_PROVIDERS_PER_CLASS: dict[str, list[tuple[str, str]]] = {
    cls: list(
        _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS.get(cls, [])
    ) + list(
        _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS.get(cls, [])
    )
    for cls in DISPATCH_CLASSES
}


def _fallback_reason_for(
    dispatch_class: str, provider: str, gpu: str
) -> FallbackReason | None:
    """Return the :class:`FallbackReason` for one (class, provider, gpu) tuple.

    Returns ``None`` if the tuple is not registered as any fallback for the
    given dispatch class. Used by :func:`select_provider_for_class` to
    classify the chosen fallback for forensic logging.
    """
    cheaper = _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS.get(dispatch_class, [])
    if (provider, gpu) in cheaper:
        return FallbackReason.CHEAPER_ALTERNATIVE
    overflow = _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS.get(dispatch_class, [])
    if (provider, gpu) in overflow:
        return FallbackReason.CAPACITY_OVERFLOW
    return None

# Per-class soft cost ceiling in USD per dispatch (informational; the gate
# is the operator's session-budget envelope, not this number). Used by
# :func:`classify_dispatch` to disambiguate borderline recipes when the
# operator did not declare an explicit dispatch_class.
PER_CLASS_SOFT_COST_CEILING_USD: dict[str, float] = {
    "smoke": 2.0,
    "full": 15.0,
    "long_burn": 100.0,
    "eval": 2.0,
    "cpu": 0.0,
}

# Per-class soft wall-clock ceiling in hours.
PER_CLASS_SOFT_WALLCLOCK_CEILING_HR: dict[str, float] = {
    "smoke": 0.5,
    "full": 12.0,
    "long_burn": 168.0,  # one week
    "eval": 0.5,
    "cpu": 12.0,
}

# Time-Traveler amendment: re-route trigger threshold (fractional cost
# improvement). If a fallback is empirically >25% cheaper than the
# canonical (with matched confidence tag), recommend the fallback.
RE_ROUTING_TRIGGER_FRACTION = 0.25


@dataclass(frozen=True)
class ProviderRoutingDecision:
    """Resolved (provider, gpu) tuple for one dispatch with full forensics.

    Carries the canonical Decision 9 verdict, every fallback considered, the
    cost-band posterior consulted, and a human-readable rationale. Consumers
    (operator_authorize.py, autopilot ranker) read ``provider`` + ``gpu`` for
    the dispatch decision and ``rationale`` for logging.

    BOYD-1 self-protection (Catalog #237 2026-05-15): ``fallback_reason``
    explicitly disambiguates whether the chosen fallback is a Time-Traveler
    cheaper-alternative auto-route OR a manual operator capacity-overflow
    escalation. ``None`` when no fallback was selected (canonical chosen).
    """

    dispatch_class: str
    provider: str  # canonical resolved provider (e.g. "modal", "vastai")
    gpu: str  # canonical resolved GPU (e.g. "T4", "RTX_4090")
    canonical_provider: str
    canonical_gpu: str
    fallback_provider: str | None
    fallback_gpu: str | None
    posterior_consulted: bool
    re_routed: bool
    re_routing_rationale: str
    canonical_cost_p50_usd: float | None
    fallback_cost_p50_usd: float | None
    rationale: str
    fallback_reason: FallbackReason | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dispatch_class": self.dispatch_class,
            "provider": self.provider,
            "gpu": self.gpu,
            "canonical_provider": self.canonical_provider,
            "canonical_gpu": self.canonical_gpu,
            "fallback_provider": self.fallback_provider,
            "fallback_gpu": self.fallback_gpu,
            "posterior_consulted": self.posterior_consulted,
            "re_routed": self.re_routed,
            "re_routing_rationale": self.re_routing_rationale,
            "canonical_cost_p50_usd": self.canonical_cost_p50_usd,
            "fallback_cost_p50_usd": self.fallback_cost_p50_usd,
            "rationale": self.rationale,
            "fallback_reason": (
                self.fallback_reason.value if self.fallback_reason else None
            ),
        }


# Recipe-side label tokens that classify a dispatch.
# Per CLAUDE.md "NEVER invent CLI flags": these tokens come from
# the existing dispatch_label conventions in `.omx/state/cost_band_posterior.jsonl`
# (e.g. `*__smoke__100ep`, `gha_cpu_eval_*`, `*_auth_eval_*`). The
# premise verifier `.omx/tmp/d9_provider_routing_premise_verifier.py`
# audited 38 anchors and confirmed these tokens partition cleanly.
_SMOKE_LABEL_TOKENS = ("__smoke__", "_smoke_", "smoke_v", "_probe_")
_LONG_BURN_LABEL_TOKENS = ("long_burn", "longburn")
_EVAL_LABEL_TOKENS = ("auth_eval", "_cpu_eval_", "_cuda_eval_", "gha_cpu_eval")


def classify_dispatch(
    *,
    dispatch_label: str | None = None,
    epochs: int | None = None,
    estimated_wall_clock_sec: float | None = None,
    estimated_cost_usd: float | None = None,
    explicit_dispatch_class: str | None = None,
) -> str:
    """Classify a dispatch into one of :data:`DISPATCH_CLASSES`.

    Order of resolution:

    1. ``explicit_dispatch_class`` if provided (operator override).
    2. ``dispatch_label`` token match (most specific signal).
    3. Wall-clock + cost soft ceilings (fallback).
    4. ``"full"`` as the safe default for ambiguous recipes.

    Returns one of :data:`DISPATCH_CLASSES`. Never raises; defaults to
    ``"full"`` so the routing helper has a safe per-class canonical.
    """
    if explicit_dispatch_class:
        cls = explicit_dispatch_class.strip().lower()
        if cls in DISPATCH_CLASSES:
            return cls
        # Unknown explicit class: fall through to label-based inference
        # rather than raising, so a typo doesn't crash the routing call.

    label = (dispatch_label or "").lower()
    if any(tok in label for tok in _EVAL_LABEL_TOKENS):
        return "eval"
    if any(tok in label for tok in _LONG_BURN_LABEL_TOKENS):
        return "long_burn"
    if any(tok in label for tok in _SMOKE_LABEL_TOKENS):
        return "smoke"

    # Numeric inference. Per the premise verifier's auditing, label tokens
    # cover 28/38 of the historical posterior; numeric inference covers
    # the rest (recipes that pre-date the smoke-before-full label
    # convention OR whose dispatch_label uses a custom format).
    #
    # BOYD-2 (Boyd LOW R2 finding 2026-05-14, Catalog #239 self-protection
    # 2026-05-15): the long_burn upgrade boundaries use OPEN ``>`` (strict
    # greater-than) because crossing them at the ceiling triggers a 5-10x
    # cost class jump (full $2-15 -> long_burn $50+). At 12.0h exactly the
    # safer routing is "full" (Vast.ai 4090 ~$2-15) — operators must
    # explicitly request long_burn (Lightning A100 subscription, $50+
    # equivalent) rather than fall into it from a borderline wallclock
    # estimate. Per Boyd's convex-feasibility lens: regions whose boundary
    # crossing changes the cost class non-trivially must use OPEN
    # boundaries; CLOSED boundaries (``>=``) create discontinuous cost
    # expectation at the ceiling. The smoke downgrade boundaries keep
    # ``<=`` (CLOSED) because they route DOWNWARD to the cheaper class —
    # at the smoke ceiling exactly the safer routing is "smoke" and there
    # is no cost penalty for over-routing to the cheaper class.
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec > PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:
            return "long_burn"
        if estimated_wall_clock_sec <= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["smoke"] * 3600.0:
            # Sub-30-min dispatches are smokes by default. Epochs <= 200 is a
            # secondary signal that catches recipes whose wall-clock estimate
            # is missing.
            return "smoke"
    if epochs is not None and epochs <= 200:
        return "smoke"
    if estimated_cost_usd is not None:
        if estimated_cost_usd <= PER_CLASS_SOFT_COST_CEILING_USD["smoke"]:
            return "smoke"
        if estimated_cost_usd > PER_CLASS_SOFT_COST_CEILING_USD["long_burn"]:
            return "long_burn"

    return "full"


def _provider_label(provider: str, gpu: str) -> str:
    return f"{provider}/{gpu}"


def _consult_posterior_for_provider(
    provider: str,
    gpu: str,
    *,
    epochs_bucket_value: int,
    posterior_path: Path | None = None,
) -> tuple[float | None, str]:
    """Return ``(p50_cost_usd, confidence_tag)`` for one provider.

    Returns ``(None, "no_anchors")`` when no successful anchors exist;
    callers should NOT compare across confidence tags ("empirical_posterior"
    vs "weak_posterior" vs "hand_calibrated_fallback") because the bands
    are not comparable.
    """
    pred = predict(
        provider,
        gpu,
        epochs_bucket_value,
        all_flags_on=True,
        posterior_path=posterior_path,
    )
    if pred.confidence_tag == "hand_calibrated_fallback":
        return None, pred.confidence_tag
    return pred.p50_cost_usd, pred.confidence_tag


def select_provider_for_class(
    dispatch_class: str,
    *,
    recipe_meta: dict[str, Any] | None = None,
    posterior_path: Path | None = None,
    consult_posterior: bool = True,
    epochs_for_posterior: int = 3000,
    capacity_overflow: bool = False,
) -> ProviderRoutingDecision:
    """Resolve the (provider, gpu) tuple for one dispatch class.

    Implements council Decision 9 binding verdict (PROCEED Option B + Time-
    Traveler amendment). Returns a :class:`ProviderRoutingDecision` with full
    forensic context (canonical, fallback, posterior consulted, re-routing
    rationale, fallback reason).

    Args:
        dispatch_class: One of :data:`DISPATCH_CLASSES`. Unknown classes
            default to ``"full"`` with a rationale note.
        recipe_meta: Optional recipe dict; if it carries an explicit
            ``provider``/``gpu`` pair OR a ``provider: auto`` marker, the
            routing helper honors the operator's choice (auto = use Decision
            9 canonical; explicit = pass-through with a recipe-override
            rationale).
        posterior_path: Override for the posterior JSONL (tests only).
        consult_posterior: Whether to consult the cost-band posterior for
            dynamic re-routing per the Time-Traveler amendment. Default
            True; tests may set False to assert the canonical decision.
        epochs_for_posterior: Epochs bucket to query the posterior with.
            Defaults to 3000 (T1 Balle canonical).
        capacity_overflow: If True, allow the routing helper to escalate
            to the capacity-overflow fallback set (e.g. ``long_burn`` →
            ``vastai/H100`` when Lightning A100 is saturated). Default
            False — capacity-overflow fallbacks are NEVER auto-routed
            without explicit operator opt-in. Per BOYD-1 R2 finding +
            Catalog #237 self-protection: cheaper-alternative and
            capacity-overflow are semantically distinct triggers and
            must not be conflated.

    Returns:
        :class:`ProviderRoutingDecision`. Consumers read ``provider`` +
        ``gpu`` for the dispatch decision; ``rationale`` for forensics;
        ``fallback_reason`` for explicit cheaper-vs-overflow disambiguation.

    Per Catalog #175 + #177 (cost-band outcome discipline) the helper only
    consults SUCCESSFUL_DISPATCH anchors via :func:`predict`. Per Catalog
    #199 + #202 (operator-authorize bypass discipline) the helper does NOT
    bypass any operator-authorize gate; recipe-side overrides are honored
    so the operator's explicit choice always wins.
    """
    # Resolve the dispatch class. Unknown classes -> "full" with rationale.
    if dispatch_class not in DISPATCH_CLASSES:
        return ProviderRoutingDecision(
            dispatch_class="full",
            provider=CANONICAL_PROVIDER_PER_CLASS["full"][0],
            gpu=CANONICAL_PROVIDER_PER_CLASS["full"][1],
            canonical_provider=CANONICAL_PROVIDER_PER_CLASS["full"][0],
            canonical_gpu=CANONICAL_PROVIDER_PER_CLASS["full"][1],
            fallback_provider=None,
            fallback_gpu=None,
            posterior_consulted=False,
            re_routed=False,
            re_routing_rationale="",
            canonical_cost_p50_usd=None,
            fallback_cost_p50_usd=None,
            rationale=(
                f"unknown dispatch_class={dispatch_class!r}; "
                f"defaulted to 'full' canonical per Decision 9 safe-default"
            ),
        )

    canon_provider, canon_gpu = CANONICAL_PROVIDER_PER_CLASS[dispatch_class]

    # Recipe-side override: operator may explicitly set `provider:` and `gpu:`
    # in the recipe to opt out of auto-routing. Per CLAUDE.md "Subagent
    # coherence-by-default" anti-fragmentation primitive: the routing helper
    # NEVER silently overrides the operator's explicit choice. ``provider: auto``
    # means "use the Decision 9 canonical".
    recipe_meta = recipe_meta or {}
    recipe_provider = recipe_meta.get("provider") or recipe_meta.get("platform")
    recipe_gpu = recipe_meta.get("gpu")
    if recipe_provider and str(recipe_provider).strip().lower() not in {"auto", "none"}:
        # Operator explicitly chose this provider; pass-through.
        return ProviderRoutingDecision(
            dispatch_class=dispatch_class,
            provider=str(recipe_provider).strip().lower(),
            gpu=str(recipe_gpu or canon_gpu),
            canonical_provider=canon_provider,
            canonical_gpu=canon_gpu,
            fallback_provider=None,
            fallback_gpu=None,
            posterior_consulted=False,
            re_routed=False,
            re_routing_rationale="",
            canonical_cost_p50_usd=None,
            fallback_cost_p50_usd=None,
            rationale=(
                f"recipe explicitly set provider={recipe_provider!r} gpu={recipe_gpu!r}; "
                f"Decision 9 canonical for class={dispatch_class!r} would be "
                f"{_provider_label(canon_provider, canon_gpu)} (not used)"
            ),
        )

    # Auto-routing path. Consult posterior if requested.
    if not consult_posterior:
        return ProviderRoutingDecision(
            dispatch_class=dispatch_class,
            provider=canon_provider,
            gpu=canon_gpu,
            canonical_provider=canon_provider,
            canonical_gpu=canon_gpu,
            fallback_provider=None,
            fallback_gpu=None,
            posterior_consulted=False,
            re_routed=False,
            re_routing_rationale="",
            canonical_cost_p50_usd=None,
            fallback_cost_p50_usd=None,
            rationale=(
                f"Decision 9 canonical {_provider_label(canon_provider, canon_gpu)} "
                f"for class={dispatch_class!r}; posterior consultation skipped per caller"
            ),
        )

    # Time-Traveler amendment: consult posterior for canonical AND every
    # cheaper-alternative fallback. BOYD-1 self-protection: capacity-
    # overflow fallbacks are NOT auto-routed; only the cheaper-alternative
    # set participates in the >25% cost-shift evaluation. Capacity-overflow
    # escalation requires explicit ``capacity_overflow=True`` opt-in below.
    canon_p50, canon_tag = _consult_posterior_for_provider(
        canon_provider,
        canon_gpu,
        epochs_bucket_value=epochs_for_posterior,
        posterior_path=posterior_path,
    )
    fallbacks = list(
        _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS.get(dispatch_class, [])
    )
    if capacity_overflow:
        # Operator explicitly authorized capacity-overflow escalation; add
        # those fallbacks to the candidate set. The Time-Traveler cost-shift
        # check is still applied (so even when overflow=True, an overflow
        # fallback is only chosen if its empirical cost beats canonical's
        # — which is rare by definition since overflow fallbacks are
        # typically more expensive). The semantic split is preserved via
        # the `fallback_reason` field on the returned decision.
        fallbacks.extend(
            _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS.get(dispatch_class, [])
        )
    best_fallback: tuple[str, str] | None = None
    best_fallback_p50: float | None = None
    best_fallback_tag: str | None = None
    for fb_provider, fb_gpu in fallbacks:
        fb_p50, fb_tag = _consult_posterior_for_provider(
            fb_provider,
            fb_gpu,
            epochs_bucket_value=epochs_for_posterior,
            posterior_path=posterior_path,
        )
        if fb_p50 is None:
            continue
        if best_fallback_p50 is None or fb_p50 < best_fallback_p50:
            best_fallback = (fb_provider, fb_gpu)
            best_fallback_p50 = fb_p50
            best_fallback_tag = fb_tag

    # Re-routing decision per Time-Traveler amendment. Trigger only when
    # BOTH canonical AND fallback have empirical p50s (matched confidence
    # floor) and fallback is >25% cheaper.
    re_routed = False
    re_routing_rationale = ""
    chosen_provider = canon_provider
    chosen_gpu = canon_gpu

    if (
        canon_p50 is not None
        and best_fallback_p50 is not None
        and canon_p50 > 0.0
        and best_fallback is not None
        # Matched confidence floor: both at least "weak_posterior"
        # (not "hand_calibrated_fallback" — those are not real anchors).
        and canon_tag in {"empirical_posterior", "weak_posterior"}
        and best_fallback_tag in {"empirical_posterior", "weak_posterior"}
    ):
        cost_shift = (canon_p50 - best_fallback_p50) / canon_p50
        if cost_shift > RE_ROUTING_TRIGGER_FRACTION:
            re_routed = True
            chosen_provider, chosen_gpu = best_fallback
            re_routing_rationale = (
                f"Time-Traveler amendment trigger: canonical "
                f"{_provider_label(canon_provider, canon_gpu)} p50=${canon_p50:.2f} "
                f"vs fallback {_provider_label(*best_fallback)} p50=${best_fallback_p50:.2f}; "
                f"shift={cost_shift * 100:.1f}% > {RE_ROUTING_TRIGGER_FRACTION * 100:.0f}% threshold; "
                f"re-routed to fallback"
            )

    if re_routed:
        rationale = re_routing_rationale
    else:
        rationale = (
            f"Decision 9 canonical {_provider_label(canon_provider, canon_gpu)} "
            f"for class={dispatch_class!r}; "
            f"posterior canon_p50="
            f"{('$%.2f' % canon_p50) if canon_p50 is not None else 'no_anchors'} "
            f"({canon_tag}); fallback="
            f"{_provider_label(*best_fallback) if best_fallback else 'none'} "
            f"p50="
            f"{('$%.2f' % best_fallback_p50) if best_fallback_p50 is not None else 'no_anchors'}"
        )

    chosen_fallback_reason: FallbackReason | None = None
    if best_fallback is not None:
        chosen_fallback_reason = _fallback_reason_for(
            dispatch_class, best_fallback[0], best_fallback[1]
        )

    return ProviderRoutingDecision(
        dispatch_class=dispatch_class,
        provider=chosen_provider,
        gpu=chosen_gpu,
        canonical_provider=canon_provider,
        canonical_gpu=canon_gpu,
        fallback_provider=best_fallback[0] if best_fallback else None,
        fallback_gpu=best_fallback[1] if best_fallback else None,
        posterior_consulted=True,
        re_routed=re_routed,
        re_routing_rationale=re_routing_rationale,
        canonical_cost_p50_usd=canon_p50,
        fallback_cost_p50_usd=best_fallback_p50,
        rationale=rationale,
        fallback_reason=chosen_fallback_reason,
    )


def select_provider_for_recipe(
    recipe_meta: dict[str, Any],
    *,
    posterior_path: Path | None = None,
    consult_posterior: bool = True,
    capacity_overflow: bool = False,
) -> ProviderRoutingDecision:
    """Convenience wrapper: classify + select for one recipe meta dict.

    The recipe meta is the same dict that ``operator_authorize.Recipe.raw``
    carries: it may contain ``dispatch_class`` (explicit), ``dispatch_label``,
    ``cost_band.epochs``, and the canonical ``platform``/``provider`` +
    ``gpu`` keys. This helper threads them into :func:`classify_dispatch`
    and :func:`select_provider_for_class`.

    Args:
        recipe_meta: Recipe dict with optional ``dispatch_class`` /
            ``dispatch_label`` / ``cost_band.epochs`` keys.
        posterior_path: Override for the posterior JSONL (tests only).
        consult_posterior: Whether to consult the cost-band posterior.
        capacity_overflow: If True, allow capacity-overflow fallback
            escalation (e.g. Lightning A100 saturated → Vast.ai H100).
            BOYD-1 self-protection: capacity-overflow fallbacks are
            NEVER auto-routed without explicit operator opt-in.
    """
    explicit_class = recipe_meta.get("dispatch_class")
    dispatch_label = recipe_meta.get("dispatch_label") or recipe_meta.get("label")
    cost_band = recipe_meta.get("cost_band", {}) or {}
    epochs = cost_band.get("epochs") or recipe_meta.get("epochs")
    if isinstance(epochs, str):
        try:
            epochs = int(epochs)
        except ValueError:
            epochs = None
    estimated_wall_clock_sec = cost_band.get("estimated_wall_clock_sec")
    estimated_cost_usd = cost_band.get("estimated_cost_usd")

    dispatch_class = classify_dispatch(
        dispatch_label=dispatch_label,
        epochs=int(epochs) if isinstance(epochs, int) else None,
        estimated_wall_clock_sec=
            float(estimated_wall_clock_sec) if isinstance(estimated_wall_clock_sec, (int, float)) else None,
        estimated_cost_usd=
            float(estimated_cost_usd) if isinstance(estimated_cost_usd, (int, float)) else None,
        explicit_dispatch_class=str(explicit_class) if explicit_class else None,
    )

    return select_provider_for_class(
        dispatch_class,
        recipe_meta=recipe_meta,
        posterior_path=posterior_path,
        consult_posterior=consult_posterior,
        epochs_for_posterior=int(epochs) if isinstance(epochs, int) and epochs > 0 else 3000,
        capacity_overflow=capacity_overflow,
    )
