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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "cost_band_posterior_v1"
POSTERIOR_PATH = Path(".omx/state/cost_band_posterior.jsonl")
LOCK_PATH = Path(".omx/state/.cost_band_posterior.lock")

# Confidence-tag thresholds (per Council probe-disambiguator pattern).
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
    freshness_seconds: Optional[float]  # how old the most-recent matching anchor is
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
    """One empirical anchor: a completed dispatch's measured wall-clock + cost."""

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
    predicted_cost_usd_low: Optional[float] = None
    predicted_cost_usd_high: Optional[float] = None
    prediction_in_band: Optional[bool] = None
    notes: str = ""
    schema: str = SCHEMA_VERSION


def _now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _ensure_state_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_anchor(
    anchor: CostBandAnchor,
    *,
    posterior_path: Optional[Path] = None,
    lock_path: Optional[Path] = None,
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
    posterior_path: Optional[Path] = None,
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
# uncalibrated per the fresh-eyes adversarial NF0). These ONLY apply when
# the posterior has zero matching anchors. The first real dispatch with
# `append_anchor` supersedes them.
_HAND_CALIBRATED_FALLBACKS = {
    # (platform, gpu, epochs_bucket, all_flags_on): (p10, p50, p90) USD
    ("modal", "T4", 3000, True): (5.0, 8.0, 12.0),
    ("modal", "A10G", 3000, True): (4.0, 6.0, 10.0),
    ("modal", "A100", 3000, True): (3.0, 5.0, 8.0),
    ("modal", "H100", 3000, True): (3.0, 4.5, 7.0),
    ("vastai", "4090", 3000, True): (0.50, 0.80, 1.20),
    ("lightning", "T4", 3000, True): (0.0, 0.0, 0.0),  # free tier
    ("modal", "T4", 3000, False): (3.5, 5.5, 8.0),
    ("modal", "T4", 1000, True): (1.5, 2.5, 4.0),
    ("modal", "T4", 100, True): (0.20, 0.30, 0.50),
}


def _epochs_bucket(epochs: int) -> int:
    """Bucket epochs to ease anchor matching (small training runs vary
    a lot in wall-clock per epoch, so we widen the bucket)."""
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
    posterior_path: Optional[Path] = None,
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
        dt_now = datetime.datetime.now(datetime.timezone.utc)
        freshness_seconds: Optional[float] = (dt_now - dt_anchor).total_seconds()
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


def summary_by_bucket(
    posterior_path: Optional[Path] = None,
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
