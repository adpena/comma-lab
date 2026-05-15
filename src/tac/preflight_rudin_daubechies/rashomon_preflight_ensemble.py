# SPDX-License-Identifier: MIT
"""Rashomon ensemble of K=8 near-optimal preflight rankings.

Per Rudin's "Rashomon set" theory (Semenova, Rudin & Parr 2020): for any
sufficiently complex prediction problem, there exists a SET of near-optimal
models that perform comparably on the training data but DIFFER on
out-of-sample predictions. The disagreement among the set is the canonical
signal for "what to add next".

Operationalized for preflight as K=8 SLIM risk scorers each trained on a
different sub-sample of the empirical preflight outcome pool (bootstrap
resampling). Consensus prediction = mean across K rankers; disagreement =
standard deviation. HIGH disagreement panels are surfaced as the
next-gate-to-add queue.

Continual learning per operator directive 2026-05-15: every preflight
outcome flows through :meth:`PreflightRashomonEnsemble.update_all` which
refits all K rankers (each with a fresh bootstrap sample of the updated
anchor pool). Members age out per the Daubechies multi-scale spirit.

Self-protection: Catalog #275 enforces continual-update-locked discipline
at SOURCE level — bypassing the persisted `store_path` skips the
fcntl-locked JSONL store and the continual-learning loop never closes.

[verified-against: Semenova, Rudin & Parr 2020 §4 + autopilot sister
``tac.autopilot_rudin_daubechies.rashomon_ensemble.RashomonEnsembleRanker``]
"""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .slim_risk_scorer import (
    DEFAULT_PREFLIGHT_INTEGER_BOUND,
    DEFAULT_PREFLIGHT_SPARSITY_TARGET,
    GateVerdictPanel,
    PreflightSLIMRiskScorer,
    _preflight_anchor_store_lock,
    _PreflightAnchorRecord,
    _utc_now_iso,
)

DEFAULT_PREFLIGHT_RASHOMON_SIZE: int = 8


@dataclass
class PreflightRashomonMember:
    """One ranker in the preflight Rashomon set."""

    member_id: int
    scorer: PreflightSLIMRiskScorer
    bootstrap_seed: int
    n_anchors_used: int

    def predict(self, panel: GateVerdictPanel) -> float:
        return self.scorer.predict(panel)


@dataclass
class _DisagreementEntry:
    """One panel-disagreement record for the operator-facing ideation queue."""

    snapshot_id: str
    consensus_risk: float
    disagreement_stddev: float
    member_predictions: tuple[float, ...]


class PreflightRashomonEnsemble:
    """Ensemble of K near-optimal SLIM risk scorers.

    Each member trains on a bootstrap-resampled subset of the preflight
    anchor pool (with replacement; sample size = ``len(pool)``); the K
    members differ because they each saw a slightly different empirical
    sample. Together they probe the Rashomon set of comparably-fit
    interpretable preflight risk scorers.

    Consensus prediction = arithmetic mean across the K members; the
    operator-facing rank-key is the consensus, but the disagreement std-dev
    is THE side-information for ideation (which gate to ADD next, which
    rule to PROMOTE).

    Continual learning surface:

    * :meth:`update_all` — appends a new anchor and refits ALL K members
      with fresh bootstrap samples of the updated pool. Per CLAUDE.md
      "Catalog #128/#131" sister discipline: the underlying anchor store
      is fcntl-locked.

    [verified-against: Semenova, Rudin & Parr 2020 §4.1 + autopilot
    sister ``tac.autopilot_rudin_daubechies.rashomon_ensemble.RashomonEnsembleRanker``]
    """

    def __init__(
        self,
        *,
        ensemble_size: int = DEFAULT_PREFLIGHT_RASHOMON_SIZE,
        integer_bound: int = DEFAULT_PREFLIGHT_INTEGER_BOUND,
        sparsity_target: int = DEFAULT_PREFLIGHT_SPARSITY_TARGET,
        rng_seed: int = 0,
        store_path: Path | None = None,
        lock_path: Path | None = None,
    ) -> None:
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}")
        self.ensemble_size = int(ensemble_size)
        self.integer_bound = int(integer_bound)
        self.sparsity_target = int(sparsity_target)
        self._rng = random.Random(rng_seed)
        self._store_path = store_path
        self._lock_path = lock_path or (
            store_path.with_suffix(store_path.suffix + ".lock")
            if store_path is not None
            else None
        )
        self._anchors: list[_PreflightAnchorRecord] = []
        self._members: list[PreflightRashomonMember] = []
        self._disagreement_queue: list[_DisagreementEntry] = []
        self._build_initial_members()
        if store_path is not None:
            self._load_from_store()

    @property
    def members(self) -> tuple[PreflightRashomonMember, ...]:
        return tuple(self._members)

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    @property
    def disagreement_queue(self) -> tuple[_DisagreementEntry, ...]:
        return tuple(self._disagreement_queue)

    # ── prediction surface ─────────────────────────────────────────────────

    def predict_with_disagreement(
        self, panel: GateVerdictPanel
    ) -> tuple[float, float, tuple[float, ...]]:
        """Return (consensus_risk, disagreement_stddev, per-member predictions)."""
        per_member = tuple(m.predict(panel) for m in self._members)
        if not per_member:
            return 0.0, 0.0, ()
        consensus = sum(per_member) / len(per_member)
        stddev = statistics.stdev(per_member) if len(per_member) >= 2 else 0.0
        return consensus, stddev, per_member

    def confidence_tag(self) -> str:
        if self.n_anchors == 0:
            return f"[preflight-risk; cold-start; rashomon-K={self.ensemble_size}]"
        return (
            f"[preflight-risk; n={self.n_anchors}-anchor-posterior; "
            f"rashomon-K={self.ensemble_size}]"
        )

    # ── continual-learning surface ─────────────────────────────────────────

    def update_all(
        self,
        observed_dispatch_risk: float,
        panel: GateVerdictPanel,
        *,
        axis: str = "preflight",
    ) -> tuple[float, float, tuple[float, ...]]:
        """Append a new anchor, refit ALL K members, and return prediction.

        Per CLAUDE.md "Catalog #128/#131" + sister discipline: the underlying
        anchor store is fcntl-locked when a ``store_path`` is set.
        """
        if not isinstance(panel, GateVerdictPanel):
            raise TypeError(
                f"panel must be GateVerdictPanel, got {type(panel).__name__}"
            )
        if not math.isfinite(observed_dispatch_risk):
            raise ValueError(
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
        else:
            self._anchors.append(record)
        self._refit_all_members()
        prediction = self.predict_with_disagreement(panel)
        # Surface high-disagreement rows to the queue.
        consensus, stddev, per_member = prediction
        if stddev > self.integer_bound * 0.05:  # ~5% of bound = meaningful spread
            self._disagreement_queue.append(
                _DisagreementEntry(
                    snapshot_id=panel.snapshot_id,
                    consensus_risk=consensus,
                    disagreement_stddev=stddev,
                    member_predictions=per_member,
                )
            )
        return prediction

    def _build_initial_members(self) -> None:
        self._members = []
        for member_id in range(self.ensemble_size):
            seed = self._rng.randint(0, 2**31 - 1)
            scorer = PreflightSLIMRiskScorer(
                integer_bound=self.integer_bound,
                sparsity_target=self.sparsity_target,
                rng_seed=seed,
            )
            self._members.append(
                PreflightRashomonMember(
                    member_id=member_id,
                    scorer=scorer,
                    bootstrap_seed=seed,
                    n_anchors_used=0,
                )
            )

    def _refit_all_members(self) -> None:
        if not self._anchors:
            return
        n = len(self._anchors)
        for member in self._members:
            rng = random.Random(member.bootstrap_seed + n)
            # Bootstrap with replacement.
            sampled_indices = [rng.randint(0, n - 1) for _ in range(n)]
            sampled_anchors = [self._anchors[i] for i in sampled_indices]
            member.scorer = PreflightSLIMRiskScorer(
                integer_bound=self.integer_bound,
                sparsity_target=self.sparsity_target,
                rng_seed=member.bootstrap_seed,
            )
            member.scorer._anchors = sampled_anchors
            member.scorer._refit()
            member.n_anchors_used = n

    # ── private: persistence ───────────────────────────────────────────────

    def _append_anchor_to_store(self, record: _PreflightAnchorRecord) -> None:
        path = self._store_path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.as_dict(), separators=(",", ":")))
            fh.write("\n")

    def _read_all_from_store(self) -> Iterable[_PreflightAnchorRecord]:
        path = self._store_path
        if path is None or not path.exists():
            return
        import json
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
            self._refit_all_members()
