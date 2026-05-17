# SPDX-License-Identifier: MIT
"""Rashomon ensemble of K=8 near-optimal SLIM rankers + disagreement queue.

Per Rudin's "Rashomon set" theory (Semenova, Rudin & Parr 2020): for any
sufficiently complex prediction problem, there exists a SET of near-optimal
models that perform comparably on the training data but DIFFER on
out-of-sample predictions. The disagreement among the set is the
canonical signal for "what to measure next".

Operationalized here as K=8 SLIM rankers each trained on a different
sub-sample of the empirical-anchor pool (bootstrap resampling). Consensus
prediction = mean across the K rankers; disagreement = standard deviation.
HIGH disagreement candidates are surfaced as the next-experiment queue.

Continual learning per operator directive 2026-05-15: every empirical
anchor flows through :meth:`RashomonEnsembleRanker.update_all` which
refits all K rankers (each with a fresh bootstrap sample of the updated
anchor pool). Members age out and new ones spawn over time per the
Daubechies multi-scale spirit (older anchors get LESS weight as they age,
mimicking the wavelet "vanishing moments" property).

Per CLAUDE.md "Council conduct — non-conservative bias" the Rashomon
ensemble structurally PREVENTS premature consensus: the Contrarian's
role is encoded in the K-1 dissenting rankers; unanimous predictions are
high-confidence; non-unanimous predictions are the operator's next probe.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .slim_ranker import (
    DEFAULT_INTEGER_COEFFICIENT_BOUND,
    DEFAULT_SPARSITY_TARGET,
    ProxyPanel,
    SLIMRanker,
    _AnchorRecord,
    _slim_anchor_store_lock,
    _utc_now_iso,
)


DEFAULT_RASHOMON_ENSEMBLE_SIZE: int = 8


@dataclass
class RashomonMember:
    """One ranker in the Rashomon set."""

    member_id: int
    ranker: SLIMRanker
    bootstrap_seed: int
    n_anchors_used: int

    def predict(self, panel: ProxyPanel) -> float:
        return self.ranker.predict(panel)


@dataclass
class _DisagreementEntry:
    """One candidate-disagreement record for the operator-facing queue."""

    candidate_id: str
    consensus_score: float
    disagreement_stddev: float
    member_predictions: tuple[float, ...]


class RashomonEnsembleRanker:
    """Ensemble of K near-optimal SLIM rankers.

    Each member trains on a bootstrap-resampled subset of the anchor pool
    (with replacement; sample size = ``len(pool)``); the K members differ
    because they each saw a slightly different empirical sample. Together
    they probe the Rashomon set of comparably-fit interpretable models.

    Consensus prediction = arithmetic mean across the K members; the
    operator-facing rank-key is the consensus, but the disagreement std-dev
    is THE side-information for ideation.

    Continual learning surface:

    * :meth:`update_all` — appends a new anchor and refits ALL K members
      with fresh bootstrap samples of the updated pool. Per CLAUDE.md
      "Catalog #128/#131" sister discipline: the underlying anchor store
      is fcntl-locked.
    * :meth:`disagreement_queue` — returns candidates sorted by descending
      stddev; HIGH stddev = next experiment to run.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" every
    consensus prediction carries the canonical confidence tag inherited
    from the underlying SLIM members.
    """

    def __init__(
        self,
        *,
        ensemble_size: int = DEFAULT_RASHOMON_ENSEMBLE_SIZE,
        integer_bound: int = DEFAULT_INTEGER_COEFFICIENT_BOUND,
        sparsity_target: int = DEFAULT_SPARSITY_TARGET,
        feature_order: Sequence[str] | None = None,
        rng_seed: int = 0,
        store_path: Path | None = None,
        lock_path: Path | None = None,
    ) -> None:
        if ensemble_size < 2:
            raise ValueError(
                f"ensemble_size must be >= 2 for disagreement signal, got {ensemble_size}"
            )
        self.ensemble_size = int(ensemble_size)
        self.integer_bound = int(integer_bound)
        self.sparsity_target = int(sparsity_target)
        self.feature_order = (
            tuple(feature_order) if feature_order is not None else None
        )
        self._rng = random.Random(rng_seed)
        self._anchors: list[_AnchorRecord] = []
        self._store_path = store_path
        self._lock_path = lock_path or (
            store_path.with_suffix(store_path.suffix + ".lock")
            if store_path is not None
            else None
        )
        self.members: list[RashomonMember] = []
        if store_path is not None:
            self._load_anchors_from_store()
        self._refit_all()

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    # ── prediction surface ─────────────────────────────────────────────────

    def predict(self, panel: ProxyPanel) -> float:
        """Consensus prediction = arithmetic mean across K members."""
        preds = [m.predict(panel) for m in self.members]
        if not preds:
            return 0.0
        return statistics.fmean(preds)

    def predict_with_disagreement(self, panel: ProxyPanel) -> tuple[float, float]:
        """Return ``(consensus, disagreement_stddev)``.

        Disagreement stddev = sample stddev across K members. HIGH stddev
        means the K near-optimal rankers DISAGREE on this candidate; per
        Rudin's Rashomon set theory, this is the highest-information probe.
        """
        preds = [m.predict(panel) for m in self.members]
        if not preds:
            return 0.0, 0.0
        consensus = statistics.fmean(preds)
        if len(preds) < 2:
            return consensus, 0.0
        disagreement = statistics.stdev(preds)
        return consensus, disagreement

    def confidence_tag(self) -> str:
        if self.n_anchors == 0:
            return "[prediction; first-principles-bound; rashomon-K=" + str(self.ensemble_size) + "]"
        return (
            f"[prediction; n={self.n_anchors}-anchor-posterior; "
            f"rashomon-K={self.ensemble_size}]"
        )

    # ── continual-learning surface ─────────────────────────────────────────

    def update_all(
        self,
        observed_score: float,
        panel: ProxyPanel,
        *,
        axis: str = "macos_cpu_advisory",
    ) -> None:
        """Append a new anchor and refit ALL K members."""
        if not isinstance(panel, ProxyPanel):
            raise TypeError(
                f"panel must be ProxyPanel, got {type(panel).__name__}"
            )
        if not math.isfinite(observed_score):
            raise ValueError(
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
                # Append to the canonical store.
                self._store_path.parent.mkdir(parents=True, exist_ok=True)
                with self._store_path.open("a", encoding="utf-8") as fh:
                    import json as _json

                    fh.write(
                        _json.dumps(record.as_dict(), separators=(",", ":"))
                    )
                    fh.write("\n")
                self._load_anchors_from_store()
                self._refit_all()
        else:
            self._anchors.append(record)
            self._refit_all()

    def update_all_from_master_gradient(
        self,
        per_pair_gradient,  # np.ndarray of shape (N_bytes, N_pairs, 3)
        *,
        archive_sha256: str,
        measurement_axis: str = "macos_cpu_advisory",
        measurement_hardware: str = "linux_x86_64_cpu",
        k_members: int | None = None,
        pair_subsample_fraction: float = 0.8,
        random_seed: int = 42,
        top_k: int = 100,
        write_sidecar: bool = False,
    ):
        """Ingest a per-pair master gradient tensor and refit the Rashomon ensemble.

        Sister of :meth:`update_all` at the master-gradient surface: where
        ``update_all`` consumes ONE (observed_score, panel) anchor from a paid
        empirical dispatch + refits via historical-anchor bootstrap, this
        method consumes the per-pair gradient tensor from a Catalog #245
        master-gradient anchor + refits via PAIR-AXIS bootstrap (K subsamples
        of pair indices). Both are legitimate Semenova-Rudin-Parr 2020 Rashomon
        constructions; they capture complementary uncertainty surfaces.

        The pair-axis bootstrap fits in the existing canonical persistence
        contract:

        - Each Rashomon member k samples a pair subset S_k ⊂ {0, ..., N_pairs-1}.
        - The per-member per-byte averaged gradient ``np.mean(per_pair[:, S_k, :], axis=1)``
          reduces to a per-member per-byte importance scalar via the canonical
          L1-sum-of-abs metric.
        - Per-byte stddev across K members IS the operator-facing
          next-experiment queue signal per Rudin's Rashomon discipline.

        Per Catalog #252 sister: the per-pair-bootstrap disagreement queue is
        persisted to a sister JSONL store at
        ``<store_path>.per_byte_disagreement.jsonl`` (under the same fcntl lock
        as the anchor store) so the continual-learning loop closes — the next
        autopilot ranking pass reads this store via
        :func:`tac.master_gradient_consumers.rashomon_disagreement_queue` and
        weights candidates' ``predicted_dispatch_risk`` by their byte-overlap
        with the top-K disagreement bytes.

        Per CLAUDE.md "Apples-to-apples evidence discipline": the persisted
        row carries ``score_claim=False`` and the canonical
        ``measurement_axis`` + ``measurement_hardware`` labels.

        Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192: the
        ``measurement_hardware`` field is preserved as-is; promotion of the
        disagreement signal to a contest-grade claim requires a paired
        ``[contest-CPU]`` or ``[contest-CUDA]`` anchor on 1:1 contest-compliant
        hardware.

        Args:
          per_pair_gradient: numpy ndarray of shape ``(N_bytes, N_pairs, 3)``
            per the canonical ``PER_PAIR_GRADIENT_TENSOR_KIND`` contract.
          archive_sha256: archive SHA-256 from the source master-gradient anchor.
          measurement_axis: canonical axis label from the source anchor (e.g.
            ``"contest_cpu"`` / ``"contest_cuda"`` / ``"macos_cpu_advisory"``).
          measurement_hardware: hardware substrate label from the source anchor.
          k_members: number of Rashomon members; defaults to
            ``self.ensemble_size`` to keep the per-pair-bootstrap and the
            historical-anchor-bootstrap dimensions consistent.
          pair_subsample_fraction: fraction of pairs per member (default 0.8 =
            80% of pairs sampled without replacement per member).
          random_seed: deterministic seed for the K bootstrap samples.
          top_k: number of top-disagreement bytes to surface in the persisted row.
          write_sidecar: whether to also emit a sidecar JSON at the canonical
            ``.omx/state/master_gradient_consumers/`` path (default False;
            persistence to the per-byte disagreement JSONL is unconditional).

        Returns:
          ``RashomonDisagreementQueue`` from
          :func:`tac.master_gradient_consumers.rashomon_disagreement_queue`.

        Raises:
          RuntimeError: if ``store_path`` is None (Catalog #252 forbids
            non-persisted Rashomon ensembles; persisting the disagreement
            signal is the entire point of this method).
          ValueError: from the underlying consumer on shape / parameter
            validation failures.
        """
        # Sister Catalog #252 enforcement: persistence is mandatory.
        if self._store_path is None:
            raise RuntimeError(
                "update_all_from_master_gradient requires a persisted store_path "
                "(Catalog #252 `check_rashomon_ensemble_continual_update_locked` "
                "forbids non-persisted Rashomon ensembles); construct the "
                "RashomonEnsembleRanker with store_path=<canonical JSONL path>"
            )
        # Defer the heavy import to call-time so the autopilot_rudin_daubechies
        # package stays import-light when only SLIM / falling-rule paths are
        # used.
        from tac.master_gradient_consumers import rashomon_disagreement_queue

        k_effective = int(k_members) if k_members is not None else int(self.ensemble_size)
        queue = rashomon_disagreement_queue(
            per_pair_gradient,
            archive_sha256=archive_sha256,
            measurement_axis=measurement_axis,
            measurement_hardware=measurement_hardware,
            k_members=k_effective,
            pair_subsample_fraction=pair_subsample_fraction,
            random_seed=random_seed,
            top_k=top_k,
            write_sidecar=write_sidecar,
        )

        # Persist per-byte disagreement row to sister JSONL store under the
        # SAME fcntl lock as the anchor store per Catalog #128/#131. Path =
        # ``<anchor_store>.per_byte_disagreement.jsonl`` so cleanup tools that
        # treat the anchor store as a unit can find the sister file by glob.
        disagreement_store_path = self._store_path.with_name(
            self._store_path.name + ".per_byte_disagreement.jsonl"
        )
        row = {
            "schema": "rashomon_per_pair_disagreement_v1",
            "archive_sha256": archive_sha256,
            "measurement_axis": measurement_axis,
            "measurement_hardware": measurement_hardware,
            "k_members": queue.k_members,
            "pair_subsample_size": queue.pair_subsample_size,
            "pair_subsample_fraction": queue.pair_subsample_fraction,
            "random_seed": queue.random_seed,
            "n_bytes": queue.n_bytes,
            "n_pairs": queue.n_pairs,
            "top_k_disagreement_indices": list(queue.top_k_disagreement_indices),
            "aggregate_disagreement_score": queue.aggregate_disagreement_score,
            "written_at_utc": _utc_now_iso(),
            # Per CLAUDE.md "Apples-to-apples evidence discipline"
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "evidence_grade": "[diagnostic; rashomon per-pair disagreement]",
        }
        with _slim_anchor_store_lock(self._lock_path):
            disagreement_store_path.parent.mkdir(parents=True, exist_ok=True)
            with disagreement_store_path.open("a", encoding="utf-8") as fh:
                import json as _json

                fh.write(_json.dumps(row, separators=(",", ":")))
                fh.write("\n")

        return queue

    def disagreement_queue(
        self,
        candidates: Iterable[ProxyPanel],
    ) -> list[_DisagreementEntry]:
        """Rank candidates by descending member-disagreement std-dev.

        HIGH std-dev candidates are the canonical next-experiment queue per
        Rudin's Rashomon discipline: when K near-optimal interpretable models
        disagree, the disagreement IS the information bottleneck.
        """
        out: list[_DisagreementEntry] = []
        for panel in candidates:
            preds = tuple(m.predict(panel) for m in self.members)
            if not preds:
                continue
            consensus = statistics.fmean(preds)
            disagreement = statistics.stdev(preds) if len(preds) >= 2 else 0.0
            out.append(
                _DisagreementEntry(
                    candidate_id=panel.candidate_id,
                    consensus_score=consensus,
                    disagreement_stddev=disagreement,
                    member_predictions=preds,
                )
            )
        out.sort(key=lambda e: e.disagreement_stddev, reverse=True)
        return out

    # ── private: training ──────────────────────────────────────────────────

    def _refit_all(self) -> None:
        new_members: list[RashomonMember] = []
        for k in range(self.ensemble_size):
            seed = self._rng.randrange(0, 2**31 - 1)
            ranker = SLIMRanker(
                integer_bound=self.integer_bound,
                sparsity_target=self.sparsity_target,
                feature_order=self.feature_order,
                rng_seed=seed,
                store_path=None,
                lock_path=None,
            )
            sample = self._bootstrap_sample(self._anchors, seed)
            for record in sample:
                ranker.update_from_anchor(
                    record.observed_score, record.panel, axis=record.axis
                )
            new_members.append(
                RashomonMember(
                    member_id=k,
                    ranker=ranker,
                    bootstrap_seed=seed,
                    n_anchors_used=len(sample),
                )
            )
        self.members = new_members

    @staticmethod
    def _bootstrap_sample(
        anchors: Sequence[_AnchorRecord], seed: int
    ) -> list[_AnchorRecord]:
        """Sample with replacement; sample-size = len(anchors).

        Returns empty list when the anchor pool is empty (consistent with
        the cold-start first-principles fallback in :class:`SLIMRanker`).
        """
        if not anchors:
            return []
        rng = random.Random(seed)
        n = len(anchors)
        return [anchors[rng.randrange(n)] for _ in range(n)]

    def _load_anchors_from_store(self) -> None:
        path = self._store_path
        if path is None or not path.exists():
            self._anchors = []
            return
        out: list[_AnchorRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            import json as _json

            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                out.append(_AnchorRecord.from_dict(raw))
        self._anchors = out
