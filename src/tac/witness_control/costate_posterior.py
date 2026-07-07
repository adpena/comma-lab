"""Cross-run costate POSTERIOR — the #247 continual-learning loop.

The costate controller estimates λ_i = ∂S/∂(lever_i) from ONE run's trajectory verdicts. Without a
cross-run memory, every run starts blind: the same costate is re-estimated from scratch, and the
DECIDE ranker cannot seed an UNIDENTIFIABLE in-run costate from what earlier runs already measured.
This module is that memory — the mechanism that makes session N+1 smarter than N (per CLAUDE.md
"Results must become system intelligence").

It is a canonical, APPEND-ONLY, fcntl-locked JSONL at ``.omx/state/costate_posterior.jsonl`` (mirrors
the other ``.omx/state/*.jsonl`` stores). NO-FAKE by construction:

* Only IDENTIFIABLE costates are recorded — an ``UNIDENTIFIABLE`` estimate (the controller's honest
  "not enough data") is NEVER written, so the posterior can never accumulate a guess.
* Combination is inverse-variance (precision) weighting of the per-run estimates — the standard
  Bayesian combination for independent Gaussian estimates λ̂_i ± σ_i. A run with a wide stderr
  contributes little; a tight one dominates. No estimate is invented; the posterior is exactly the
  precision-weighted mean of what was MEASURED.
* The posterior is ADVISORY: it seeds the DECIDE queue ("this lever historically had costate λ") — it
  never fabricates an in-run measurement and never actuates (the controller's ``actuation`` is always
  ``NONE``).

Write cadence: ONE set of observations per run at its byte-close (the CLOSE side), not per shadow tick
— ``record_run_costates`` dedups within a run by (run_ref, costate name), keeping the latest
(most-converged) estimate. Read cadence: every shadow report carries ``costate_prior`` = the current
cross-run posterior, so the live controller SEES the accumulated memory.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
POSTERIOR_PATH = _REPO_ROOT / ".omx" / "state" / "costate_posterior.jsonl"

# tiers that are REAL measurements (identifiable). UNIDENTIFIABLE is deliberately excluded.
_RECORDABLE_TIERS = frozenset({"MEASURED", "ANALYTIC", "PARTIAL"})


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_costate_observation(
    name: str,
    value: float,
    stderr: float,
    *,
    tier: str,
    run_ref: str,
    epoch: int | None = None,
    evidence: str = "",
    path: Path | None = None,
) -> dict | None:
    """Append ONE measured costate observation (fcntl-locked). Returns the written row, or ``None`` if
    the observation is not recordable (NO-FAKE: UNIDENTIFIABLE / non-finite / missing value is dropped,
    never written)."""
    if tier not in _RECORDABLE_TIERS:
        return None
    try:
        v = float(value)
        se = float(stderr)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or not math.isfinite(se) or se < 0.0:
        return None
    if not name or not run_ref:
        return None
    row = {
        "name": name, "value": v, "stderr": se, "tier": tier,
        "run_ref": run_ref, "epoch": epoch, "evidence": evidence, "ts": _utc(),
    }
    p = Path(path) if path is not None else POSTERIOR_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    try:
        import fcntl
        with open(p, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - non-POSIX fallback
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
    return row


def _read(path: Path | None = None) -> list[dict]:
    p = Path(path) if path is not None else POSTERIOR_PATH
    if not p.exists():
        return []
    out: list[dict] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("name") and "value" in row:
            out.append(row)
    return out


def record_run_costates(costates: list, run_ref: str, path: Path | None = None) -> tuple[dict, ...]:
    """CLOSE-side: record a whole run's IDENTIFIABLE costates ONCE (at byte-close). ``costates`` is the
    list of ``CostateEstimate`` from a shadow report; each identifiable one is appended. Returns the
    rows written. Fail-safe: any malformed costate is skipped, never fabricated."""
    written: list[dict] = []
    for c in costates or []:
        try:
            tier = getattr(c, "tier", None)
            name = getattr(c, "name", None)
            value = getattr(c, "value", None)
            stderr = getattr(c, "stderr", None)
            ev = getattr(c, "evidence", None)
            evidence = ev[0] if isinstance(ev, (tuple, list)) and ev else (ev or "")
        except Exception:  # noqa: BLE001
            continue
        r = record_costate_observation(
            str(name) if name else "", value if value is not None else float("nan"),
            stderr if stderr is not None else float("nan"),
            tier=str(tier) if tier else "", run_ref=run_ref, evidence=str(evidence), path=path,
        )
        if r is not None:
            written.append(r)
    return tuple(written)


@dataclass(frozen=True)
class CostatePosterior:
    name: str
    mean: float          # inverse-variance-weighted mean of the per-run estimates
    stderr: float        # posterior stderr = 1/sqrt(sum of precisions)
    n_obs: int
    n_runs: int


def posterior_for(name: str, path: Path | None = None) -> CostatePosterior | None:
    """The cross-run posterior for one costate: inverse-variance (precision) combination of the LATEST
    per-run estimates. Returns ``None`` if no run has measured it. A zero-stderr estimate is floored to
    a tiny epsilon so it does not produce an infinite precision."""
    rows = [r for r in _read(path) if r.get("name") == name]
    if not rows:
        return None
    # latest observation per run (most-converged estimate for that run)
    latest: dict[str, dict] = {}
    for r in rows:
        rr = r.get("run_ref") or ""
        prev = latest.get(rr)
        if prev is None or (r.get("ts") or "") >= (prev.get("ts") or ""):
            latest[rr] = r
    ests = list(latest.values())
    eps = 1e-12
    precisions = [1.0 / max(float(r["stderr"]) ** 2, eps) for r in ests]
    total_prec = sum(precisions)
    if total_prec <= 0.0:
        return None
    mean = sum(p * float(r["value"]) for p, r in zip(precisions, ests)) / total_prec
    return CostatePosterior(
        name=name, mean=mean, stderr=math.sqrt(1.0 / total_prec),
        n_obs=len(rows), n_runs=len(ests),
    )


def all_posteriors(path: Path | None = None) -> list[dict]:
    """The full cross-run posterior — one row per costate name (the READ surface the shadow report
    carries as ``costate_prior`` so the live controller SEES the accumulated memory)."""
    names = sorted({r["name"] for r in _read(path)})
    out = []
    for n in names:
        post = posterior_for(n, path)
        if post is not None:
            out.append({"name": post.name, "mean": post.mean, "stderr": post.stderr,
                        "n_obs": post.n_obs, "n_runs": post.n_runs})
    return out


__all__ = [
    "POSTERIOR_PATH",
    "CostatePosterior",
    "all_posteriors",
    "posterior_for",
    "record_costate_observation",
    "record_run_costates",
]
