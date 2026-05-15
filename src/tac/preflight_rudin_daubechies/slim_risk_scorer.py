# SPDX-License-Identifier: MIT
"""SLIM (Sparse Linear Integer Model) risk scorer over preflight gate verdicts.

Per Rudin's canonical SLIM formulation (see ``Ustun & Rudin 2016, "Supersparse
linear integer models for optimized medical scoring systems"``):

    predicted_risk_score = intercept + sum(integer_coef_i * gate_violation_i)

with the constraint that every coefficient is an INTEGER in
``[-K, K]`` (default K=50, larger than autopilot's K=10 because preflight
risk is denominated in ~25-point Tier-1 violation increments) and at most
``S`` coefficients are nonzero (default S=8, larger than autopilot's S=5
because the preflight surface has ~270 gates).

The risk scorer consumes a panel of per-gate verdicts (PASSED / VIOLATED /
WAIVED / EXEMPT). VIOLATED gates contribute their integer coefficient to the
risk; PASSED gates contribute zero; WAIVED + EXEMPT gates contribute zero
(operator has explicitly opted out).

Cold-start first-principles bound (per the design memo):

* Tier-1 violation = +25 risk (e.g., Catalog #1, #5, #7, #146, #167)
* Tier-2 violation = +15 risk (e.g., Catalog #117, #125, #126)
* Tier-3 violation = +7 risk (most other catalog gates)
* META-meta violation = +50 risk (e.g., Catalog #118, #176, #185)

These are the FIRST-PRINCIPLES seeds; once empirical preflight outcomes arrive
the greedy fit replaces them.

Continual learning per operator directive 2026-05-15: every preflight outcome
flows through :meth:`PreflightSLIMRiskScorer.update_from_anchor` which refits
coefficients under fcntl-locked posterior write per Catalog #128/#131 sister
discipline. The risk scorer gets smarter with each preflight cycle.

Per CLAUDE.md "Apples-to-apples evidence discipline" predictions carry
``[preflight-risk; cold-start]`` (cold start) or
``[preflight-risk; n=K-anchor-posterior]`` (after K updates).

Per CLAUDE.md "Council conduct — non-conservative bias": the integer-constraint
discipline is NOT a safety hedge; it is the contract that makes risk decisions
auditable. The Rashomon set guarantees a near-optimal interpretable SLIM exists
for the preflight surface.

Self-protection: Catalog #273 enforces integer-coefficient discipline at
SOURCE level — bypassing the canonical helpers re-introduces the bug class.

[verified-against: Ustun & Rudin 2016 §3.2 + canonical autopilot sister
``tac.autopilot_rudin_daubechies.slim_ranker.SLIMRanker``]
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reuse the integer-contract validator from the autopilot sister.
from tac.autopilot_rudin_daubechies.slim_ranker import SLIMCoefficient as _AutopilotSLIMCoefficient

DEFAULT_PREFLIGHT_INTEGER_BOUND: int = 50
DEFAULT_PREFLIGHT_SPARSITY_TARGET: int = 8


# Canonical gate-tier classifications. Per the design memo first-principles
# bound, these tiers map to integer coefficient seeds. Updated by sister
# subagent per the LiveTier classification.
_TIER_1_GATE_NUMBERS: frozenset[int] = frozenset({
    # Tier-1: catastrophic failure modes. Source-level forbidden patterns.
    1, 5, 6, 7, 8, 14, 109, 110, 113, 117, 124, 125, 127, 128, 131, 138,
    146, 151, 152, 153, 157, 167, 174, 176, 185, 186, 192, 205, 206, 220,
    233, 240, 245, 248,
})

_TIER_2_GATE_NUMBERS: frozenset[int] = frozenset({
    # Tier-2: integration / wire-up gates.
    2, 3, 4, 11, 12, 13, 15, 16, 17, 18, 19, 90, 91, 95, 96, 97, 98,
    115, 118, 119, 126, 130, 132, 133, 134, 135, 136, 137, 139, 140, 141,
    142, 143, 144, 145, 147, 148, 150, 154, 156, 158, 159, 161, 162, 163,
    164, 165, 166, 168, 169, 173, 175, 177, 178, 179, 180, 187, 188, 189,
    190, 191, 193, 197, 199, 201, 202, 203, 204, 207, 208, 209, 210, 211,
    213, 215, 218, 219, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
    231, 234, 235, 236, 237, 239, 241, 242, 243, 244, 246, 249,
})

_TIER_3_GATE_NUMBERS: frozenset[int] = frozenset({
    # Tier-3: per-trainer / per-recipe hygiene + scaffolding.
    9, 10, 92, 93, 94, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
    111, 112, 114, 121, 122, 123, 129, 149, 155, 160, 170, 171, 172, 181,
    182, 183, 184, 194, 195, 196, 198, 200, 212, 214, 216, 217, 232, 238,
    247, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262,
    263, 264, 265,
})

_META_META_GATE_NUMBERS: frozenset[int] = frozenset({
    # META-meta: catalog-discipline / strict-text-vs-truth gates.
    118, 159, 176, 185, 235,
})

_TIER_1_RISK_SCORE: int = 25
_TIER_2_RISK_SCORE: int = 15
_TIER_3_RISK_SCORE: int = 7
_META_META_RISK_SCORE: int = 50

DISPATCH_RISK_REFUSAL_THRESHOLD: float = 50.0


# ───────────────────────────────────────────────────────────────────────────
# GateVerdictPanel schema (the preflight analog of the autopilot's ProxyPanel)
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateVerdictPanel:
    """Panel of per-gate verdicts for ONE staged-files snapshot.

    Each entry maps a catalog # (string of the int) to one of:

    * ``"VIOLATED"`` — the gate fired and reported a violation
    * ``"PASSED"`` — the gate ran and emitted no violations
    * ``"WAIVED"`` — the staged files use a same-line waiver to opt out
    * ``"EXEMPT"`` — the gate is structurally not applicable (e.g. test files)
    * ``"NOT_RUN"`` — the gate was not invoked for this snapshot

    Risk computation: VIOLATED contributes the gate's integer coefficient;
    PASSED / WAIVED / EXEMPT contribute zero (operator has explicitly opted
    out where applicable); NOT_RUN contributes zero because we lack evidence.

    The :attr:`scope_axis` field carries the canonical preflight axis label per
    CLAUDE.md "Apples-to-apples evidence discipline":

    * ``"changed-file-subset"`` — fast hook mode (default for tools/preflight_hook.py)
    * ``"all-source-tree"`` — full --scope all sweep
    * ``"release-custody"`` — pre-release sweep with archive custody verification
    """

    verdicts: Mapping[str, str] = field(default_factory=dict)
    scope_axis: str = "changed-file-subset"
    snapshot_id: str = ""

    def violated_gate_numbers(self) -> tuple[int, ...]:
        """Return the catalog # ints that have verdict VIOLATED."""
        out: list[int] = []
        for k, v in self.verdicts.items():
            if str(v).upper() == "VIOLATED":
                try:
                    out.append(int(k))
                except (ValueError, TypeError):
                    continue
        return tuple(sorted(out))

    def as_feature_vector(
        self, gate_order: Sequence[int] | None = None
    ) -> list[float]:
        """Return the ordered violation indicator vector (1.0 if VIOLATED, else 0.0)."""
        order = list(gate_order) if gate_order is not None else _all_canonical_gate_numbers()
        out: list[float] = []
        violated = set(self.violated_gate_numbers())
        for gate_num in order:
            out.append(1.0 if gate_num in violated else 0.0)
        return out


def _all_canonical_gate_numbers() -> tuple[int, ...]:
    """Return every canonical catalog # in sorted order (feature vector axis)."""
    union = (
        _TIER_1_GATE_NUMBERS
        | _TIER_2_GATE_NUMBERS
        | _TIER_3_GATE_NUMBERS
        | _META_META_GATE_NUMBERS
    )
    return tuple(sorted(union))


# ───────────────────────────────────────────────────────────────────────────
# Coefficient + risk scorer
# ───────────────────────────────────────────────────────────────────────────


class PreflightSLIMTrainingError(ValueError):
    """Raised when SLIM training violates an integer / sparsity / bound contract."""


@dataclass(frozen=True)
class _PreflightAnchorRecord:
    """One empirical preflight outcome anchor."""

    panel: GateVerdictPanel
    observed_dispatch_risk: float  # observed downstream risk score
    axis: str
    written_at_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel": {
                "verdicts": dict(self.panel.verdicts),
                "scope_axis": self.panel.scope_axis,
                "snapshot_id": self.panel.snapshot_id,
            },
            "observed_dispatch_risk": float(self.observed_dispatch_risk),
            "axis": self.axis,
            "written_at_utc": self.written_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> _PreflightAnchorRecord:
        panel_raw = raw.get("panel", {}) or {}
        panel = GateVerdictPanel(
            verdicts=dict(panel_raw.get("verdicts", {}) or {}),
            scope_axis=str(panel_raw.get("scope_axis", "changed-file-subset")),
            snapshot_id=str(panel_raw.get("snapshot_id", "")),
        )
        return cls(
            panel=panel,
            observed_dispatch_risk=float(raw.get("observed_dispatch_risk", 0.0)),
            axis=str(raw.get("axis", "preflight")),
            written_at_utc=str(raw.get("written_at_utc", "")),
        )


class PreflightSLIMRiskScorer:
    """Sparse Linear Integer Model risk scorer over preflight gate verdicts.

    Coefficients are fitted to minimize squared error on the (panel, observed
    dispatch risk) anchors collected so far, subject to:

    * |coef_i| <= ``integer_bound`` for all i (default 50)
    * at most ``sparsity_target`` coefficients are nonzero (default 8)
    * coefficients are EXACTLY integer (not floats rounded post-fit)

    Training algorithm: greedy forward selection (pick the gate whose best
    integer coefficient most reduces SSE; repeat ``sparsity_target`` times)
    + coordinate descent local search (one pass over selected gates, each
    re-tuned to its best integer in ``[-bound, bound]``).

    For low-data regimes (N < 3 anchors) the scorer falls back to first-
    principles tier-based bounds. The ``confidence_tag`` reflects this with
    ``[preflight-risk; cold-start]``.

    Continual learning: every :meth:`update_from_anchor` call appends an
    anchor to the JSONL store under fcntl lock and re-runs the fit.

    Sister of :class:`tac.autopilot_rudin_daubechies.slim_ranker.SLIMRanker`
    which holds the dispatch-side ranking layer at a different granularity
    (per-CandidateRow predicted-score; the preflight scorer is the
    OPERATOR-FACING risk-readback layer that consumes the verdict panel).

    [verified-against: autopilot SLIMRanker — same greedy + coord-descent
    integer-projection algorithm; bounds + sparsity scaled for the preflight
    surface size (~270 gates vs 14 Taylor proxies).]
    """

    def __init__(
        self,
        *,
        integer_bound: int = DEFAULT_PREFLIGHT_INTEGER_BOUND,
        sparsity_target: int = DEFAULT_PREFLIGHT_SPARSITY_TARGET,
        feature_order: Sequence[int] | None = None,
        rng_seed: int = 0,
        store_path: Path | None = None,
        lock_path: Path | None = None,
    ) -> None:
        if integer_bound < 1:
            raise PreflightSLIMTrainingError(
                f"integer_bound must be >= 1, got {integer_bound}"
            )
        if sparsity_target < 1:
            raise PreflightSLIMTrainingError(
                f"sparsity_target must be >= 1, got {sparsity_target}"
            )
        feature_order = (
            tuple(feature_order)
            if feature_order is not None
            else _all_canonical_gate_numbers()
        )
        if not feature_order:
            raise PreflightSLIMTrainingError("feature_order must be non-empty")
        if sparsity_target > len(feature_order):
            raise PreflightSLIMTrainingError(
                f"sparsity_target={sparsity_target} exceeds feature_count="
                f"{len(feature_order)}"
            )
        self.integer_bound = int(integer_bound)
        self.sparsity_target = int(sparsity_target)
        self.feature_order: tuple[int, ...] = tuple(feature_order)
        self._rng = random.Random(rng_seed)
        self._anchors: list[_PreflightAnchorRecord] = []
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

    def predict(self, panel: GateVerdictPanel) -> float:
        """Return predicted dispatch-risk score for ``panel``."""
        violated = set(panel.violated_gate_numbers())
        out = float(self._intercept)
        for c in self._coefficients:
            try:
                gate_num = int(c.proxy_name)
            except (ValueError, TypeError):
                continue
            if gate_num in violated:
                out += float(c.integer_coef)
        return out

    def confidence_tag(self) -> str:
        if self.n_anchors == 0:
            return "[preflight-risk; cold-start]"
        return f"[preflight-risk; n={self.n_anchors}-anchor-posterior]"

    # ── continual-learning surface ─────────────────────────────────────────

    def update_from_anchor(
        self,
        observed_dispatch_risk: float,
        panel: GateVerdictPanel,
        *,
        axis: str = "preflight",
    ) -> None:
        """Append a new empirical anchor and refit coefficients.

        Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
        against": the persisted JSONL store is fcntl-locked per Catalog
        #128/#131 sister discipline; concurrent writers serialize on the
        lock so distinct anchor updates all survive.
        """
        if not isinstance(panel, GateVerdictPanel):
            raise TypeError(
                f"panel must be GateVerdictPanel, got {type(panel).__name__}"
            )
        if not math.isfinite(observed_dispatch_risk):
            raise PreflightSLIMTrainingError(
                f"observed_dispatch_risk must be finite, got {observed_dispatch_risk!r}"
            )
        record = _PreflightAnchorRecord(
            panel=panel,
            observed_dispatch_risk=float(observed_dispatch_risk),
            axis=str(axis),
            written_at_utc=_utc_now_iso(),
        )
        if self._store_path is not None:
            with _preflight_anchor_store_lock(self._lock_path):
                self._append_anchor_to_store(record)
                self._anchors = list(self._read_all_from_store())
                self._refit()
        else:
            self._anchors.append(record)
            self._refit()

    def explain(self, panel: GateVerdictPanel) -> str:
        """Human-readable rule chain explanation per Rudin's interpretability principle."""
        return explain_preflight_risk_prediction(self, panel)

    # ── private: training ──────────────────────────────────────────────────

    def _refit(self) -> None:
        if not self._anchors:
            self._fit_first_principles()
            return
        self._fit_greedy_then_coord_descent()
        self._fitted_at_utc = _utc_now_iso()

    def _fit_first_principles(self) -> None:
        """Cold-start: seed coefficients from per-tier first-principles bound.

        Top-K most catastrophic Tier-1 gates get coefficient = +25.
        Top-K META-meta gates get coefficient = +50 (capped at integer_bound).
        Sparsity-budget exhausted before lower tiers are seeded.
        """
        seeds: list[SLIMCoefficient] = []
        # META-meta first (highest per-violation cost)
        for gate_num in sorted(_META_META_GATE_NUMBERS):
            if len(seeds) >= self.sparsity_target:
                break
            value = max(-self.integer_bound, min(self.integer_bound, _META_META_RISK_SCORE))
            seeds.append(
                SLIMCoefficient(
                    proxy_name=str(gate_num),
                    integer_coef=int(value),
                    bound=self.integer_bound,
                )
            )
        # Tier-1 next
        for gate_num in sorted(_TIER_1_GATE_NUMBERS):
            if len(seeds) >= self.sparsity_target:
                break
            if any(int(c.proxy_name) == gate_num for c in seeds):
                continue
            value = max(-self.integer_bound, min(self.integer_bound, _TIER_1_RISK_SCORE))
            seeds.append(
                SLIMCoefficient(
                    proxy_name=str(gate_num),
                    integer_coef=int(value),
                    bound=self.integer_bound,
                )
            )
        self._coefficients = seeds[: self.sparsity_target]
        self._intercept = 0
        self._fitted_at_utc = _utc_now_iso()

    def _fit_greedy_then_coord_descent(self) -> None:
        """Greedy forward selection + coordinate descent local search.

        Performance scoping per the design memo: with ~270 catalog gates and
        K=8 Rashomon members refit per anchor, the naive O(N x bound) inner
        loop is ~270 x 101 ~ 27k trial scans per greedy step. We restrict
        the candidate feature set to gates that have actually been VIOLATED
        in at least one observed anchor — typical empirical density is ~5-10
        gates fire across all anchors, so this scopes the search ~30x and
        keeps refit-per-anchor sub-second.
        """
        feats_per_anchor = [
            a.panel.as_feature_vector(self.feature_order) for a in self._anchors
        ]
        targets = [a.observed_dispatch_risk for a in self._anchors]
        # Scope feature search to gates that have ever been violated in the
        # anchor pool. Gates that never fire have feature vector = 0.0
        # everywhere; their best integer coefficient is 0 (drops from sparsity
        # budget), so search over them is wasteful.
        active_feature_indices = [
            fi for fi in range(len(self.feature_order))
            if any(fa[fi] != 0.0 for fa in feats_per_anchor)
        ]
        n_features = len(active_feature_indices)
        if n_features == 0:
            # No gate has ever fired; intercept-only fit.
            self._intercept = self._tune_intercept(targets, [], feats_per_anchor)
            self._coefficients = []
            self._fitted_at_utc = _utc_now_iso()
            return

        best_intercept = self._tune_intercept(targets, [], feats_per_anchor)
        active_coefs: dict[int, int] = {}

        for _step in range(self.sparsity_target):
            best_gain = 0.0
            best_choice: tuple[int, int, int] | None = None
            current_sse = self._sse(
                targets, feats_per_anchor, active_coefs, best_intercept
            )
            # Only iterate over gates that have actually been violated in
            # the anchor pool — other gates' best integer is 0.
            for fi in active_feature_indices:
                if fi in active_coefs:
                    continue
                for trial in self._candidate_integers():
                    if trial == 0:
                        continue
                    trial_intercept = self._tune_intercept(
                        targets,
                        [*list(active_coefs.items()), (fi, trial)],
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

        improved = True
        passes = 0
        while improved and passes < 4:
            improved = False
            passes += 1
            for fi in list(active_coefs.keys()):
                best_local_sse = self._sse(
                    targets, feats_per_anchor, active_coefs, best_intercept
                )
                for trial in self._candidate_integers():
                    if trial == active_coefs.get(fi, 0):
                        continue
                    trial_active = dict(active_coefs)
                    if trial == 0:
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
                        if trial == 0:
                            active_coefs.pop(fi, None)
                        else:
                            active_coefs[fi] = trial
                        best_intercept = trial_intercept
                        improved = True
                        break

        self._intercept = int(best_intercept)
        self._coefficients = [
            SLIMCoefficient(
                proxy_name=str(self.feature_order[fi]),
                integer_coef=int(coef),
                bound=self.integer_bound,
            )
            for fi, coef in sorted(active_coefs.items())
        ]

    def _candidate_integers(self) -> list[int]:
        return list(range(-self.integer_bound, self.integer_bound + 1))

    def _tune_intercept(
        self,
        targets: Sequence[float],
        active: Sequence[tuple[int, int]],
        feats_per_anchor: Sequence[Sequence[float]],
    ) -> int:
        if not targets:
            return 0
        residuals: list[float] = []
        for j, t in enumerate(targets):
            linear = sum(coef * feats_per_anchor[j][fi] for fi, coef in active)
            residuals.append(t - linear)
        mean_resid = sum(residuals) / len(residuals)
        rounded = round(mean_resid)
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

    def _append_anchor_to_store(self, record: _PreflightAnchorRecord) -> None:
        path = self._store_path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.as_dict(), separators=(",", ":")))
            fh.write("\n")

    def _read_all_from_store(self) -> Iterable[_PreflightAnchorRecord]:
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
                yield _PreflightAnchorRecord.from_dict(raw)

    def _load_from_store(self) -> None:
        if self._lock_path is None:
            self._anchors = list(self._read_all_from_store())
        else:
            with _preflight_anchor_store_lock(self._lock_path):
                self._anchors = list(self._read_all_from_store())
        if self._anchors:
            self._refit()


# Public re-export of the autopilot's integer-contract validator.
SLIMCoefficient = _AutopilotSLIMCoefficient


def explain_preflight_risk_prediction(
    scorer: PreflightSLIMRiskScorer,
    panel: GateVerdictPanel,
) -> str:
    """Operator-facing rule chain readback per Rudin's interpretability principle.

    Returns a string of the form::

        predicted_dispatch_risk 25 = intercept(0) + 25*gate_146(VIOLATED)
                                   + 0*gate_167(PASSED) + ...

    Every term is auditable; the operator can verify the prediction by
    eyeball arithmetic. This IS the rule chain; no docstring promises, no
    hidden state — the prediction equals the visible sum.
    """
    pred = scorer.predict(panel)
    parts: list[str] = [f"intercept({scorer.intercept})"]
    violated = set(panel.violated_gate_numbers())
    for c in scorer.coefficients:
        try:
            gate_num = int(c.proxy_name)
        except (ValueError, TypeError):
            continue
        verdict_label = "VIOLATED" if gate_num in violated else "PASSED"
        parts.append(f"{c.integer_coef}*gate_{gate_num}({verdict_label})")
    return f"predicted_dispatch_risk {pred:g} = " + " + ".join(parts)


def predict_dispatch_risk_score_with_rationale(
    panel: GateVerdictPanel,
    *,
    scorer: PreflightSLIMRiskScorer | None = None,
    refusal_threshold: float = DISPATCH_RISK_REFUSAL_THRESHOLD,
) -> tuple[float, str, str]:
    """Convenience helper: return (risk_score, rule_chain_explanation, verdict).

    Verdict is one of:

    * ``"REFUSE"`` — risk above threshold; operator MUST review before dispatch
    * ``"WARN"`` — risk near threshold; operator review recommended
    * ``"OK"`` — risk below 50% of threshold; routine dispatch path

    Per CLAUDE.md "Operator gates must be wired and used": this is the
    canonical entry point for ``tools/operator_authorize.py`` to consult
    BEFORE any paid GPU dispatch.
    """
    actual_scorer = scorer if scorer is not None else PreflightSLIMRiskScorer()
    risk = actual_scorer.predict(panel)
    explanation = explain_preflight_risk_prediction(actual_scorer, panel)
    confidence = actual_scorer.confidence_tag()
    if risk >= refusal_threshold:
        verdict = "REFUSE"
    elif risk >= refusal_threshold * 0.5:
        verdict = "WARN"
    else:
        verdict = "OK"
    rationale = (
        f"{verdict} {confidence} risk={risk:g} threshold={refusal_threshold:g}\n"
        f"  rule chain: {explanation}"
    )
    return risk, rationale, verdict


# ───────────────────────────────────────────────────────────────────────────
# Locking + persistence helpers
# ───────────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _preflight_anchor_store_lock(lock_path: Path | None):
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
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
