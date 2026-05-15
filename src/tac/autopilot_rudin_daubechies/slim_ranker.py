# SPDX-License-Identifier: MIT
"""SLIM (Sparse Linear Integer Model) ranker over Taylor proxies.

Per Rudin's canonical SLIM formulation (see ``Ustun & Rudin 2016, "Supersparse
linear integer models for optimized medical scoring systems"``):

    predicted_score = intercept + sum(integer_coef_i * proxy_value_i)

with the constraint that every coefficient is an INTEGER in
``[-K, K]`` (default K=10) and at most ``S`` coefficients are nonzero
(default S=5). Both constraints are HARD; sparsity is enforced via combinatorial
greedy + coordinate-descent local search rather than L1 (the latter only
APPROXIMATES sparsity and rounds to non-integer coefficients).

The ranker consumes the canonical Taylor proxy panel
(``feedback_taylor_decomposition_contest_rules_into_autopilot_proxies_landed_20260515``);
because that infrastructure is forward-defined, this module ships a MINIMAL
:class:`ProxyPanel` schema aligned with the memo's contract that interoperates
once ``src/tac/autopilot_proxies/`` lands.

Continual learning per operator directive 2026-05-15: every empirical anchor
flows through :meth:`SLIMRanker.update_from_anchor` which refits coefficients
under fcntl-locked posterior write per Catalog #128/#131 sister discipline. The
ranker gets smarter with each spend.

Per CLAUDE.md "Apples-to-apples evidence discipline" predictions carry
``[prediction; first-principles-bound]`` (cold start) or
``[prediction; n=K-anchor-posterior]`` (after K updates).

Per CLAUDE.md "Council conduct — non-conservative bias": the integer-constraint
discipline is NOT a safety hedge; it is the contract that makes ranking
decisions auditable. The Rashomon set guarantees a near-optimal interpretable
SLIM exists if you look hard enough.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_INTEGER_COEFFICIENT_BOUND: int = 10
DEFAULT_SPARSITY_TARGET: int = 5

# Canonical Taylor proxy axis names per the Phase A decomposition memo.
# The 14 proxies = 4 SegNet + 4 PoseNet + 4 Rate + 2 architectural.
_CANONICAL_PROXY_NAMES: tuple[str, ...] = (
    # Component 1: SegNet (coefficient 100, linear)
    "seg_p0",  # canonical SegNet result (when available)
    "seg_p1",  # distilled-SegNet KL surrogate
    "seg_p2",  # boundary-margin change
    "seg_p3",  # total variation at 256x384
    # Component 2: PoseNet (coefficient sqrt(10), nonlinear sqrt)
    "pose_p0",  # canonical PoseNet result
    "pose_p1",  # per-dim MSE summary (mean over 6 hydra dims)
    "pose_p2",  # pairwise pose drift (THE substrate-class discriminator)
    "pose_p3",  # RAFT residual after warp
    # Component 3: Rate (coefficient 25, linear)
    "rate_p0",  # archive bytes / uncompressed bytes
    "rate_p1",  # per-section byte fraction
    "rate_p2",  # Shannon entropy lower bound
    "rate_p3",  # codec-vs-Shannon gap
    # Architectural / cross-component:
    "pose_jacobian_amplifier",  # 5/sqrt(10*pose_p_predicted) — operator-flip discriminator
    "rate_floor_position",  # P_rate^4 — distance to theoretical floor
)

CANONICAL_PROXY_NAMES = _CANONICAL_PROXY_NAMES  # public re-export


# ───────────────────────────────────────────────────────────────────────────
# ProxyPanel schema
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProxyPanel:
    """Minimal Taylor proxy panel schema aligned with the canonical memo.

    Every field is ``float | None``. ``None`` means "not computed for this
    candidate" — the SLIM ranker handles missing proxies by treating their
    contribution as zero (consistent with the integer-coefficient additive
    composition).

    The :attr:`panel_axis` field carries the canonical axis label per
    CLAUDE.md "Apples-to-apples evidence discipline": predictions on a
    ``contest_cuda`` panel cannot be promoted as ``contest_cpu`` evidence,
    and ``macos_cpu_advisory`` panels are non-promotable per Catalog #192.
    """

    seg_p0: float | None = None
    seg_p1: float | None = None
    seg_p2: float | None = None
    seg_p3: float | None = None
    pose_p0: float | None = None
    pose_p1: float | None = None
    pose_p2: float | None = None
    pose_p3: float | None = None
    rate_p0: float | None = None
    rate_p1: float | None = None
    rate_p2: float | None = None
    rate_p3: float | None = None
    pose_jacobian_amplifier: float | None = None
    rate_floor_position: float | None = None
    panel_axis: str = "macos_cpu_advisory"
    candidate_id: str = ""

    def as_feature_vector(self, feature_order: Sequence[str] | None = None) -> list[float]:
        """Return proxy values in canonical order, with ``None`` -> 0.0.

        The integer-coefficient SLIM cannot consume ``None``; the convention is
        "missing proxy contributes nothing" which matches the additive-composition
        semantics (the coefficient times zero is zero).
        """
        order = tuple(feature_order) if feature_order is not None else _CANONICAL_PROXY_NAMES
        out: list[float] = []
        for name in order:
            value = getattr(self, name, None)
            if value is None or not math.isfinite(value):
                out.append(0.0)
            else:
                out.append(float(value))
        return out


# ───────────────────────────────────────────────────────────────────────────
# SLIM coefficient + ranker
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SLIMCoefficient:
    """One named integer coefficient with bound + provenance."""

    proxy_name: str
    integer_coef: int
    bound: int  # the |coef| <= bound constraint that was active

    def __post_init__(self) -> None:
        if not isinstance(self.integer_coef, int) or isinstance(self.integer_coef, bool):
            raise SLIMTrainingError(
                f"integer_coef must be an int, got {type(self.integer_coef).__name__}"
            )
        if not isinstance(self.bound, int) or self.bound < 1:
            raise SLIMTrainingError(f"bound must be int >= 1, got {self.bound!r}")
        if abs(self.integer_coef) > self.bound:
            raise SLIMTrainingError(
                f"integer_coef={self.integer_coef} violates |coef| <= {self.bound}"
            )


class SLIMTrainingError(ValueError):
    """Raised when SLIM training violates an integer / sparsity / bound contract."""


@dataclass
class _AnchorRecord:
    """One empirical anchor for SLIM continual learning."""

    panel: ProxyPanel
    observed_score: float
    axis: str
    written_at_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel": dataclasses.asdict(self.panel),
            "observed_score": float(self.observed_score),
            "axis": self.axis,
            "written_at_utc": self.written_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "_AnchorRecord":
        panel_raw = dict(raw.get("panel", {}))
        panel = ProxyPanel(**panel_raw)
        return cls(
            panel=panel,
            observed_score=float(raw.get("observed_score", 0.0)),
            axis=str(raw.get("axis", "macos_cpu_advisory")),
            written_at_utc=str(raw.get("written_at_utc", "")),
        )


class SLIMRanker:
    """Sparse Linear Integer Model ranker over Taylor proxies.

    Coefficients are fitted to minimize squared error on the (panel, score)
    anchors collected so far, subject to:

    * |coef_i| <= ``integer_bound`` for all i (default 10)
    * at most ``sparsity_target`` coefficients are nonzero (default 5)
    * coefficients are EXACTLY integer (not floats rounded post-fit)

    Training algorithm: greedy forward selection (pick the proxy whose best
    integer coefficient most reduces SSE; repeat ``sparsity_target`` times)
    + coordinate descent local search (one pass over selected proxies, each
    re-tuned to its best integer in ``[-bound, bound]``).

    For low-data regimes (N < 3 anchors) the ranker falls back to first-
    principles bounds: the canonical scorer formula
    ``S = 100*S_seg + sqrt(10*S_pose) + 25*R`` is approximated linearly via
    integer coefficients (100 -> 10*10, sqrt(10) -> ~3, 25 -> 25 capped to
    bound). The ``confidence_tag`` reflects this with
    ``[prediction; first-principles-bound]``.

    Continual learning: every :meth:`update_from_anchor` call appends an
    anchor to the JSONL store under fcntl lock and re-runs the fit. The
    ranker gets smarter with each spend.

    Sister of :class:`tac.continual_learning.ContinualLearningPosterior`
    which holds component-axis-stratified Bayesian posteriors at a different
    granularity (per architecture-class, per CUDA/CPU axis); the SLIM ranker
    is the OPERATOR-FACING interpretable scoring layer that consumes the
    posterior's predictions.
    """

    def __init__(
        self,
        *,
        integer_bound: int = DEFAULT_INTEGER_COEFFICIENT_BOUND,
        sparsity_target: int = DEFAULT_SPARSITY_TARGET,
        feature_order: Sequence[str] | None = None,
        rng_seed: int = 0,
        store_path: Path | None = None,
        lock_path: Path | None = None,
    ) -> None:
        if integer_bound < 1:
            raise SLIMTrainingError(
                f"integer_bound must be >= 1, got {integer_bound}"
            )
        if sparsity_target < 1:
            raise SLIMTrainingError(
                f"sparsity_target must be >= 1, got {sparsity_target}"
            )
        feature_order = (
            tuple(feature_order)
            if feature_order is not None
            else _CANONICAL_PROXY_NAMES
        )
        if not feature_order:
            raise SLIMTrainingError("feature_order must be non-empty")
        if sparsity_target > len(feature_order):
            raise SLIMTrainingError(
                f"sparsity_target={sparsity_target} exceeds feature_count="
                f"{len(feature_order)}"
            )
        self.integer_bound = int(integer_bound)
        self.sparsity_target = int(sparsity_target)
        self.feature_order: tuple[str, ...] = feature_order
        self._rng = random.Random(rng_seed)
        self._anchors: list[_AnchorRecord] = []
        self._coefficients: list[SLIMCoefficient] = []
        self._intercept: int = 0
        self._fitted_at_utc: str = ""
        self._store_path = store_path
        self._lock_path = lock_path or (
            store_path.with_suffix(store_path.suffix + ".lock")
            if store_path is not None
            else None
        )
        self._fit_first_principles()
        if store_path is not None:
            self._load_from_store()

    # ── prediction surface ─────────────────────────────────────────────────

    @property
    def coefficients(self) -> tuple[SLIMCoefficient, ...]:
        return tuple(self._coefficients)

    @property
    def intercept(self) -> int:
        return self._intercept

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    def predict(self, panel: ProxyPanel) -> float:
        """Return predicted score for ``panel``.

        Composition: ``intercept + sum(coef * proxy)`` over selected
        coefficients only. Missing proxies contribute zero.
        """
        feats = panel.as_feature_vector(self.feature_order)
        index = {name: i for i, name in enumerate(self.feature_order)}
        out = float(self._intercept)
        for c in self._coefficients:
            i = index.get(c.proxy_name)
            if i is None:
                continue
            out += float(c.integer_coef) * feats[i]
        return out

    def confidence_tag(self) -> str:
        if self.n_anchors == 0:
            return "[prediction; first-principles-bound]"
        return f"[prediction; n={self.n_anchors}-anchor-posterior]"

    # ── continual-learning surface ─────────────────────────────────────────

    def update_from_anchor(
        self,
        observed_score: float,
        panel: ProxyPanel,
        *,
        axis: str = "macos_cpu_advisory",
    ) -> None:
        """Append a new empirical anchor and refit coefficients.

        Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
        against": the persisted JSONL store is fcntl-locked per Catalog
        #128/#131 sister discipline; concurrent harvesters serialize on the
        lock so distinct anchor updates all survive.
        """
        if not isinstance(panel, ProxyPanel):
            raise TypeError(
                f"panel must be ProxyPanel, got {type(panel).__name__}"
            )
        if not math.isfinite(observed_score):
            raise SLIMTrainingError(
                f"observed_score must be finite, got {observed_score!r}"
            )
        record = _AnchorRecord(
            panel=panel,
            observed_score=float(observed_score),
            axis=str(axis),
            written_at_utc=_utc_now_iso(),
        )
        if self._store_path is not None:
            with _slim_anchor_store_lock(self._lock_path):
                self._append_anchor_to_store(record)
                self._anchors = list(self._read_all_from_store())
                self._refit()
        else:
            self._anchors.append(record)
            self._refit()

    def explain(self, panel: ProxyPanel) -> str:
        """Human-readable rule chain explanation per Rudin's interpretability principle."""
        return explain_slim_prediction(self, panel)

    # ── private: training ──────────────────────────────────────────────────

    def _refit(self) -> None:
        if not self._anchors:
            self._fit_first_principles()
            return
        self._fit_greedy_then_coord_descent()
        self._fitted_at_utc = _utc_now_iso()

    def _fit_first_principles(self) -> None:
        """Cold-start: seed coefficients from the canonical scorer formula.

        ``S = 100*S_seg + sqrt(10*S_pose) + 25*R``

        Approximate via integers within bound:

        * ``seg_p0`` -> integer_bound (cap of 100; the bound limits the cap)
        * ``pose_p0`` -> 3 (sqrt(10) ~ 3.162; nearest integer)
        * ``rate_p0`` -> min(25, integer_bound)

        These are the FIRST-PRINCIPLES seeds; once empirical anchors arrive
        the greedy fit replaces them.
        """
        seeds: list[SLIMCoefficient] = []
        for proxy_name, raw_target in (
            ("seg_p0", 100),
            ("pose_p0", 3),
            ("rate_p0", 25),
        ):
            if proxy_name not in self.feature_order:
                continue
            value = max(-self.integer_bound, min(self.integer_bound, raw_target))
            if value == 0:
                continue
            seeds.append(
                SLIMCoefficient(
                    proxy_name=proxy_name,
                    integer_coef=int(value),
                    bound=self.integer_bound,
                )
            )
        # Respect sparsity_target by trimming if necessary.
        self._coefficients = seeds[: self.sparsity_target]
        self._intercept = 0
        self._fitted_at_utc = _utc_now_iso()

    def _fit_greedy_then_coord_descent(self) -> None:
        """Greedy forward selection + coordinate descent local search."""
        feats_per_anchor = [
            a.panel.as_feature_vector(self.feature_order) for a in self._anchors
        ]
        targets = [a.observed_score for a in self._anchors]
        n_features = len(self.feature_order)

        best_intercept = self._tune_intercept(targets, [], feats_per_anchor)
        active_coefs: dict[int, int] = {}

        # ── Greedy forward selection ──
        for _step in range(self.sparsity_target):
            best_gain = 0.0
            best_choice: tuple[int, int, int] | None = None
            current_sse = self._sse(
                targets, feats_per_anchor, active_coefs, best_intercept
            )
            for fi in range(n_features):
                if fi in active_coefs:
                    continue
                for trial in self._candidate_integers():
                    if trial == 0:
                        continue
                    trial_intercept = self._tune_intercept(
                        targets, list(active_coefs.items()) + [(fi, trial)],
                        feats_per_anchor,
                    )
                    trial_sse = self._sse(
                        targets,
                        feats_per_anchor,
                        {**active_coefs, fi: trial},
                        trial_intercept,
                    )
                    gain = current_sse - trial_sse
                    if gain > best_gain:
                        best_gain = gain
                        best_choice = (fi, trial, trial_intercept)
            if best_choice is None:
                break
            fi, trial, trial_intercept = best_choice
            active_coefs[fi] = trial
            best_intercept = trial_intercept

        # ── Coordinate descent local search ──
        improved = True
        passes = 0
        while improved and passes < 4:
            improved = False
            passes += 1
            for fi in list(active_coefs.keys()):
                best_local_sse = self._sse(
                    targets, feats_per_anchor, active_coefs, best_intercept
                )
                best_local_choice = active_coefs[fi]
                for trial in self._candidate_integers():
                    if trial == active_coefs[fi]:
                        continue
                    trial_active = dict(active_coefs)
                    if trial == 0:
                        # try DROPPING the coefficient
                        trial_active.pop(fi, None)
                    else:
                        trial_active[fi] = trial
                    trial_intercept = self._tune_intercept(
                        targets, list(trial_active.items()), feats_per_anchor
                    )
                    trial_sse = self._sse(
                        targets, feats_per_anchor, trial_active, trial_intercept
                    )
                    if trial_sse < best_local_sse - 1e-12:
                        best_local_sse = trial_sse
                        best_local_choice = trial
                        if trial == 0:
                            active_coefs.pop(fi, None)
                        else:
                            active_coefs[fi] = trial
                        best_intercept = trial_intercept
                        improved = True
                        break
                if best_local_choice == 0 and fi not in active_coefs:
                    continue

        self._intercept = int(best_intercept)
        self._coefficients = [
            SLIMCoefficient(
                proxy_name=self.feature_order[fi],
                integer_coef=int(coef),
                bound=self.integer_bound,
            )
            for fi, coef in sorted(active_coefs.items())
        ]

    def _candidate_integers(self) -> list[int]:
        # Inclusive range [-bound, bound] of trial integer coefficients.
        return list(range(-self.integer_bound, self.integer_bound + 1))

    def _tune_intercept(
        self,
        targets: Sequence[float],
        active: Sequence[tuple[int, int]],
        feats_per_anchor: Sequence[Sequence[float]],
    ) -> int:
        """Pick integer intercept in ``[-bound, bound]`` minimizing SSE."""
        if not targets:
            return 0
        # Optimal continuous intercept = mean(target - linear_part); round to nearest int.
        residuals: list[float] = []
        for j, t in enumerate(targets):
            linear = sum(coef * feats_per_anchor[j][fi] for fi, coef in active)
            residuals.append(t - linear)
        mean_resid = sum(residuals) / len(residuals)
        rounded = int(round(mean_resid))
        return max(-self.integer_bound, min(self.integer_bound, rounded))

    @staticmethod
    def _sse(
        targets: Sequence[float],
        feats_per_anchor: Sequence[Sequence[float]],
        active_coefs: Mapping[int, int],
        intercept: int,
    ) -> float:
        total = 0.0
        for j, t in enumerate(targets):
            pred = float(intercept) + sum(
                coef * feats_per_anchor[j][fi] for fi, coef in active_coefs.items()
            )
            total += (t - pred) ** 2
        return total

    # ── private: persistence ───────────────────────────────────────────────

    def _append_anchor_to_store(self, record: _AnchorRecord) -> None:
        path = self._store_path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.as_dict(), separators=(",", ":")))
            fh.write("\n")

    def _read_all_from_store(self) -> Iterable[_AnchorRecord]:
        path = self._store_path
        if path is None or not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield _AnchorRecord.from_dict(raw)

    def _load_from_store(self) -> None:
        if self._lock_path is None:
            self._anchors = list(self._read_all_from_store())
        else:
            with _slim_anchor_store_lock(self._lock_path):
                self._anchors = list(self._read_all_from_store())
        if self._anchors:
            self._refit()


def explain_slim_prediction(ranker: SLIMRanker, panel: ProxyPanel) -> str:
    """Operator-facing rule chain readback per Rudin's interpretability principle.

    Returns a string of the form::

        predicted_score 0.184 = intercept(2) + 5*proxy_pose_p2(0.012)
                              + 2*proxy_rate_p3(0.025) + ...

    Every term is auditable; the operator can verify the prediction by
    eyeball arithmetic. This IS the rule chain; no docstring promises, no
    hidden state — the prediction equals the visible sum.
    """
    pred = ranker.predict(panel)
    parts: list[str] = [f"intercept({ranker.intercept})"]
    for c in ranker.coefficients:
        value = getattr(panel, c.proxy_name, None)
        value_repr = "None" if value is None else f"{float(value):g}"
        parts.append(f"{c.integer_coef}*{c.proxy_name}({value_repr})")
    return f"predicted_score {pred:g} = " + " + ".join(parts)


# ───────────────────────────────────────────────────────────────────────────
# Locking + persistence helpers
# ───────────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _slim_anchor_store_lock(lock_path: Path | None):
    """fcntl-locked update sister of ``tac.continual_learning._posterior_lock``.

    Per CLAUDE.md "Catalog #131" canonical bare-write discipline; per
    "Catalog #138" strict-load discipline (corrupted lines are ignored at
    read time, never silently coerce the entire store to ``[]``).
    """
    if lock_path is None:
        yield None
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _utc_now_iso() -> str:
    # Fixed ISO format; mirrors continual_learning's discipline.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
