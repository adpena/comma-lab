"""Cross-series INTERACTION analyzer for witness training dynamics (task #312 Phase C).

The existing telemetry measures STATES (per-term loss values, per-class verdicts, costate
λ, annulus fraction, memory). This module measures the INTERACTION layer: how the series
move TOGETHER over training time — windowed lagged cross-correlations, lead/lag structure,
and stability — so an agent can see synergy/antagonism and lead/lag between levers, losses,
distortion, schedule, and gradient norm.

It is a PURE, offline, read-only analyzer: it consumes a run dir (``run.log`` +
``costate_shadow.jsonl``) and emits (i) a machine-readable synergy report and (ii) ranked
ADVISORY fine-tune recommendations that carry the evidence chain and the axis discipline
(``[macOS advisory]`` / NON-PROMOTABLE). Recommendation rows match the shadow-controller
schema (``action`` / ``predicted_dS`` / ``rationale`` / ``evidence``) so the costate DECIDE
surface can ingest them. No trainer changes; no MLX; no score claims — the pointer moves
only when a controller pick lands a lower byte-closed exact row.

All correlation math is plain-numpy and unit-tested against synthetic lead/lag series.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── axis / actuation discipline (mirrors shadow_controller) ─────────────────
AXIS_TAG = "[macOS advisory] NON-PROMOTABLE"
POINTER_NOTE = "pointer 0.19110 UNMOVED — advisory synergy analysis is MEANS"
ACTUATION = "ADVISORY — recommendations feed the costate DECIDE surface; no auto-actuation"

# Series names that are LOSS TERMS (from loss_terms rows) vs OBSERVABLES.
_SCHEDULE_SERIES = ("softmax_temp", "hosc_beta", "gnorm", "accepted_frac")
_DISTORTION_SERIES = ("d_seg", "d_pose", "implied_S", "blob_bytes")


# ─────────────────────────── pure correlation math ──────────────────────────
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with fail-safe on constant/degenerate input (returns 0.0)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size != x.size:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    dx = float(np.sqrt(np.sum(xm * xm)))
    dy = float(np.sqrt(np.sum(ym * ym)))
    if dx <= 1e-12 or dy <= 1e-12:  # a constant series has no linear relationship
        return 0.0
    return float(np.clip(np.sum(xm * ym) / (dx * dy), -1.0, 1.0))


def best_lag_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Return ``(lag, corr)`` maximising |corr| over lags in [-max_lag, max_lag].

    Convention: ``lag > 0`` means x LEADS y (x[t] aligns with y[t+lag]); ``lag < 0``
    means y leads x. Overlap shrinks with |lag|; a lag whose overlap < 3 is skipped.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    best = (0, 0.0)
    if n < 3 or y.size != n:
        return best
    ml = int(min(max_lag, n - 3))
    for lag in range(-ml, ml + 1):
        if lag >= 0:
            a, b = x[: n - lag], y[lag:]
        else:
            a, b = x[-lag:], y[: n + lag]
        if a.size < 3:
            continue
        c = pearson(a, b)
        if abs(c) > abs(best[1]):
            best = (lag, c)
    return best


def align_series(series_map: dict[str, list[tuple[int, float]]]
                 ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align every ``{name: [(epoch, value), ...]}`` onto the sorted union epoch grid via
    forward-fill (last-observation-carried-forward). Grid points before a series' first
    observation are NaN; callers mask per-pair on the shared finite support."""
    all_eps: set[int] = set()
    per: dict[str, dict[int, float]] = {}
    for name, pts in series_map.items():
        d: dict[int, float] = {}
        for ep, val in pts:
            if val is None or not np.isfinite(float(val)):
                continue
            d[int(ep)] = float(val)
            all_eps.add(int(ep))
        per[name] = d
    grid = np.array(sorted(all_eps), dtype=int)
    out: dict[str, np.ndarray] = {}
    for name, d in per.items():
        arr = np.full(grid.shape, np.nan, dtype=float)
        last = np.nan
        for i, ep in enumerate(grid):
            if ep in d:
                last = d[ep]
            arr[i] = last
        out[name] = arr
    return grid, out


@dataclass
class PairInteraction:
    a: str
    b: str
    lag: int              # >0: a leads b
    correlation: float    # best-lag correlation on full finite support
    n: int                # finite-overlap sample count
    window: int
    n_windows: int
    windowed_corrs: list[float] = field(default_factory=list)
    windowed_lags: list[int] = field(default_factory=list)
    stability: float = 0.0  # 1 - normalized dispersion of per-window correlation (0..1)

    def to_row(self) -> dict:
        return {
            "pair": [self.a, self.b], "lag": self.lag,
            "correlation": round(self.correlation, 4), "n": self.n,
            "window": self.window, "n_windows": self.n_windows,
            "stability": round(self.stability, 4),
            "windowed_correlations": [round(c, 4) for c in self.windowed_corrs],
            "lead": (self.a if self.lag > 0 else self.b if self.lag < 0 else "simultaneous"),
        }


def _finite_overlap(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def windowed_interaction(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str,
                         *, window: int, max_lag: int, step: int | None = None
                         ) -> PairInteraction:
    """Full-support best-lag correlation PLUS per-window best-lag correlations, and a
    STABILITY score = 1 − (std of per-window correlations), clamped to [0,1] (a pair whose
    sign/strength holds across windows is stable; a flip-flopping pair is not)."""
    fa, fb = _finite_overlap(a, b)
    lag, corr = best_lag_correlation(fa, fb, max_lag)
    wcorrs: list[float] = []
    wlags: list[int] = []
    step = step or max(1, window // 2)
    if fa.size >= window:
        for start in range(0, fa.size - window + 1, step):
            wl, wc = best_lag_correlation(fa[start:start + window], fb[start:start + window],
                                          max_lag)
            wcorrs.append(wc)
            wlags.append(wl)
    stability = 0.0
    if len(wcorrs) >= 2:
        stability = float(np.clip(1.0 - np.std(wcorrs), 0.0, 1.0))
    elif len(wcorrs) == 1:
        stability = float(abs(wcorrs[0]))
    return PairInteraction(a=name_a, b=name_b, lag=lag, correlation=corr, n=int(fa.size),
                           window=window, n_windows=len(wcorrs), windowed_corrs=wcorrs,
                           windowed_lags=wlags, stability=stability)


# ─────────────────────────── run-dir series loading ─────────────────────────
def _read_jsonl_stage(path: Path, want_stage: str | None = None) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if want_stage is None or r.get("stage") == want_stage:
            rows.append(r)
    return rows


def load_series(run_dir: str | Path) -> dict[str, list[tuple[int, float]]]:
    """Build named ``{name: [(epoch, value)]}`` series from run.log + costate_shadow.jsonl.

    Loss terms (keyed by ``ep``): every active term key -> ``term:<name>``; plus ``gnorm``,
    ``softmax_temp``, ``hosc_beta``, ``accepted_frac``. Verdicts (keyed by ``epoch``):
    ``d_seg``, ``d_pose``, ``implied_S``, ``blob_bytes``. Shadow rows (keyed by
    ``state.epoch``): ``lambda_d_pose`` (the state-dependent pose costate). Missing series
    are simply absent (the analyzer degrades gracefully)."""
    run_dir = Path(run_dir)
    log = run_dir / "run.log"
    series: dict[str, list[tuple[int, float]]] = {}

    def _push(name: str, ep, val) -> None:
        if ep is None or val is None:
            return
        try:
            e = int(ep)
            v = float(val)
        except (TypeError, ValueError):
            return
        if np.isfinite(v):
            series.setdefault(name, []).append((e, v))

    for r in _read_jsonl_stage(log, "loss_terms"):
        ep = r.get("ep")
        terms = r.get("terms") or {}
        # only keep terms that are ever nonzero across the run (drop dead levers) -> filtered below
        for k, v in terms.items():
            _push(f"term:{k}", ep, v)
        for k in ("gnorm", "softmax_temp", "hosc_beta", "accepted_frac", "total"):
            if k in r and not isinstance(r[k], str):
                _push(k if k != "total" else "loss_total", ep, r.get(k))

    for r in _read_jsonl_stage(log, "verdict"):
        ep = r.get("epoch")
        for k in ("d_seg", "d_pose", "implied_S", "blob_bytes"):
            _push(k, ep, r.get(k))

    for r in _read_jsonl_stage(run_dir / "costate_shadow.jsonl", None):
        ep = (r.get("state") or {}).get("epoch")
        for c in (r.get("costates") or []):
            if c.get("name") == "lambda_d_pose" and c.get("value") is not None:
                _push("lambda_d_pose", ep, c.get("value"))

    # Drop constant/near-dead series (a term that is 0.0 for the whole run carries no
    # interaction signal; keep only series with >=2 distinct finite values).
    pruned: dict[str, list[tuple[int, float]]] = {}
    for name, pts in series.items():
        vals = {round(v, 12) for _, v in pts}
        if len(pts) >= 3 and len(vals) >= 2:
            pruned[name] = sorted(pts)
    return pruned


# ─────────────────────────── the analysis ───────────────────────────────────
@dataclass
class DynamicsReport:
    run_dir: str
    n_series: int
    n_grid: int
    interactions: list[dict]
    recommendations: list[dict]

    def to_obj(self) -> dict:
        return {
            "run_dir": self.run_dir, "n_series": self.n_series, "n_grid_epochs": self.n_grid,
            "interactions": self.interactions, "recommendations": self.recommendations,
            "axis": AXIS_TAG, "actuation": ACTUATION, "pointer": POINTER_NOTE,
        }


def compute_interactions(series_map: dict[str, list[tuple[int, float]]], *,
                         window: int = 40, max_lag: int = 8, min_abs_corr: float = 0.3
                         ) -> tuple[list[PairInteraction], int]:
    """All-pairs windowed lagged interaction, filtered to |full-support corr| >= min_abs_corr,
    ranked by |corr|·stability descending. Returns (rows, grid_len)."""
    grid, aligned = align_series(series_map)
    names = sorted(aligned)
    rows: list[PairInteraction] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pi = windowed_interaction(aligned[names[i]], aligned[names[j]], names[i],
                                      names[j], window=window, max_lag=max_lag)
            if pi.n >= 3 and abs(pi.correlation) >= min_abs_corr:
                rows.append(pi)
    rows.sort(key=lambda p: abs(p.correlation) * (0.5 + 0.5 * p.stability), reverse=True)
    return rows, int(grid.size)


def _slope(series_map: dict, name: str, tail: int = 12) -> float | None:
    pts = series_map.get(name)
    if not pts or len(pts) < 3:
        return None
    pts = sorted(pts)[-tail:]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if xs.size < 3 or np.ptp(xs) < 1e-9:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def rank_recommendations(interactions: list[PairInteraction],
                         series_map: dict[str, list[tuple[int, float]]]
                         ) -> list[dict]:
    """Turn the strongest, STABLE interactions into ADVISORY fine-tune recommendations
    with the evidence chain. Matches the shadow-controller rec schema (action / predicted_dS
    (None here — synergy is directional, not a measured ΔS) / rationale / evidence). Every
    row is [macOS advisory] NON-PROMOTABLE; the costate DECIDE surface owns ranking-to-action."""
    recs: list[dict] = []
    d_seg_slope = _slope(series_map, "d_seg")
    d_seg_stalled = d_seg_slope is not None and abs(d_seg_slope) < 5e-6

    for pi in interactions:
        a, b = pi.a, pi.b
        pair_terms = {a, b}
        is_term = a.startswith("term:") or b.startswith("term:")
        touches_dseg = "d_seg" in pair_terms
        strong = abs(pi.correlation) >= 0.5 and pi.stability >= 0.4
        if not strong:
            continue
        # ANTAGONISM: a loss term negatively correlated with d_seg improvement while d_seg
        # is stalled is the reweight/recadence candidate.
        if is_term and touches_dseg and pi.correlation > 0.4 and d_seg_stalled:
            term = a if a.startswith("term:") else b
            recs.append({
                "action": f"REWEIGHT_OR_RECADENCE::{term}",
                "predicted_dS": None, "predicted_dS_band": None,
                "rationale": (f"{term} moves WITH d_seg (corr {pi.correlation:+.2f}, "
                              f"lag {pi.lag:+d}, stability {pi.stability:.2f}) while d_seg is "
                              f"stalled (slope {d_seg_slope:+.1e}); this term may be pinning the "
                              f"partition — candidates: down-weight or re-cadence it. ADVISORY."),
                "evidence": [f"windowed xcorr {a}~{b}: r={pi.correlation:+.3f} over "
                             f"{pi.n_windows} windows", f"d_seg tail slope {d_seg_slope:+.2e}"],
                "costate": None, "interaction": pi.to_row(),
            })
        # LEAD/LAG: a schedule/gnorm series that LEADS a distortion series is a timing lever.
        elif pi.lag != 0 and (pi.a in _SCHEDULE_SERIES or pi.b in _SCHEDULE_SERIES):
            lead = pi.a if pi.lag > 0 else pi.b
            follow = pi.b if pi.lag > 0 else pi.a
            recs.append({
                "action": f"TIMING_REVIEW::{lead}->{follow}",
                "predicted_dS": None, "predicted_dS_band": None,
                "rationale": (f"{lead} LEADS {follow} by {abs(pi.lag)} epochs "
                              f"(corr {pi.correlation:+.2f}, stability {pi.stability:.2f}); a "
                              f"schedule/regularizer change here propagates to {follow} with a "
                              f"measurable lag — tune cadence/anneal timing. ADVISORY."),
                "evidence": [f"lagged xcorr {a}~{b}: lag={pi.lag:+d} r={pi.correlation:+.3f}"],
                "costate": None, "interaction": pi.to_row(),
            })
        # TERM-TERM CONFLICT: two loss terms strongly anti-correlated = fighting each other.
        elif is_term and pi.correlation < -0.5 and a.startswith("term:") and b.startswith("term:"):
            recs.append({
                "action": f"CONFLICT_PAIR::{a}|{b}",
                "predicted_dS": None, "predicted_dS_band": None,
                "rationale": (f"{a} and {b} are ANTI-correlated (corr {pi.correlation:+.2f}, "
                              f"stability {pi.stability:.2f}) — the two loss terms are pulling the "
                              f"optimizer in opposing directions; consider rebalancing weights. "
                              f"ADVISORY."),
                "evidence": [f"windowed xcorr {a}~{b}: r={pi.correlation:+.3f}"],
                "costate": None, "interaction": pi.to_row(),
            })
    # de-dup by action, keep the strongest
    seen: dict[str, dict] = {}
    for r in recs:
        k = r["action"]
        if k not in seen:
            seen[k] = r
    return list(seen.values())


def analyze(run_dir: str | Path, *, window: int = 40, max_lag: int = 8,
            min_abs_corr: float = 0.3) -> DynamicsReport:
    run_dir = Path(run_dir)
    series = load_series(run_dir)
    interactions, grid = compute_interactions(series, window=window, max_lag=max_lag,
                                              min_abs_corr=min_abs_corr)
    recs = rank_recommendations(interactions, series)
    return DynamicsReport(run_dir=str(run_dir), n_series=len(series), n_grid=grid,
                          interactions=[pi.to_row() for pi in interactions],
                          recommendations=recs)
