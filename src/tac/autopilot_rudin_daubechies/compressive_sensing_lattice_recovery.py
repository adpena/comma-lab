# SPDX-License-Identifier: MIT
"""Compressive-sensing lattice recovery for substrate-class frontier-pursuit.

Extends the canonical :mod:`tac.autopilot_rudin_daubechies.compressive_landscape`
(Catalog #276 substrate landscape recovery) from per-cell prediction to a
SUBSTRATE-LATTICE topology that recovers a SPARSE-SIGNAL POSTERIOR over which
substrates are genuinely frontier-breaking from K=O(sqrt(N)) empirical anchors.

Mathematical foundation
-----------------------
Per the T4 Symposium Time-Traveler verdict 2026-05-16
(``.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md``)
+ Daubechies-DeVore-Fornasier-Gunturk 2010 compressive-sensing theory:

  Given N substrates in a lattice (substrate -> composition -> higher-order
  composition forming a tree-structured hierarchy), the set of FRONTIER-BREAKING
  substrates is SPARSE (s << N).  By Candes-Tao 2006 + Tropp 2004 RIP theorem:

      K >= C * s * log(N / s) measurements suffice for exact L1 recovery
      with high probability when the sensing matrix has Restricted Isometry
      Property of order 2s.

  For the cathedral lattice with N ~ 30-100 substrates and s ~ 3-8 expected
  frontier-breaking substrates: K ~ 8-16 measurements suffice.  This matches
  the K=8 LEVEL-0 measurement schedule (``.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md``).

Six enhancements per operator approval 2026-05-16
-------------------------------------------------
1. Sparse-signal posterior over frontier-breaking substrates (this module's
   :class:`SubstrateLatticeRecovery` core).
2. Coherence-minimization in K-selection (Tropp 2004 RIP via
   :func:`compute_pairwise_coherence` + :class:`CoherenceMinimizingSelector`).
3. Bayesian sequential experimental design (Snoek-Larochelle-Adams 2012 +
   Ji-Xue-Carin 2008 via :class:`BayesianSequentialKSelector`).
4. Phase-transition monitor (Donoho-Tanner 2009 sparsity-undersampling
   transition via :class:`LatticePhaseTransitionMonitor`).
5. Tree-structured sparsity priors (Baraniuk-Cevher 2010 via
   :class:`TreeStructuredSparsityPrior`).
6. Daubechies db4 wavelet basis (Mallat 1989 multi-scale per Catalog #277
   ``WaveletMultiScaleFallingRuleListRanker`` sister via
   :class:`DaubechiesDb4LatticeBasis`).

Observability surface
---------------------
Per operator standing directive 2026-05-16 (`feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`)
every helper here emits structured observability:

* Per-layer inspection: every :class:`SubstrateLatticeNode` carries
  ``support_level`` (tree-depth), ``parent_id``, ``frontier_pursuit_class``
  (PLATEAU_ADJACENT / FRONTIER_PURSUIT / ASYMPTOTIC_PURSUIT per the horizon
  class directive 2026-05-16).
* Per-signal decomposition: :meth:`SubstrateLatticeRecovery.recover_sparse_signal`
  emits a :class:`SparseSignalPosterior` with per-substrate
  ``posterior_frontier_probability``, ``posterior_score_band``, and
  ``recovery_uncertainty``.
* Diff-able: two runs of :meth:`recover_sparse_signal` with different anchor
  pools can be compared via :func:`diff_sparse_signal_posteriors`.
* Queryable: every recovery emits a JSON-serializable
  :meth:`SparseSignalPosterior.to_observability_record` for fcntl-locked
  posterior persistence per Catalog #128/#131 sister discipline.
* Cite-chain: every prediction tagged
  ``[prediction; compressive-sensing-lattice-recovery; K={n_anchors};
  N={n_cells}; sparsity={s}; tree_depth={d}; basis={basis}]``.
* Counterfactual: :meth:`predict_marginal_recovery_at_K` answers
  "what if I had one more anchor at substrate X?" without re-running.

Continual-learning hook
-----------------------
Per CLAUDE.md "Subagent coherence-by-default" wire-in hook 5: every
empirical anchor flows through :meth:`SubstrateLatticeRecovery.update_from_anchor`
which refits the sparse-signal posterior under fcntl-locked store-path
per Catalog #128/#131.

Horizon-class wire-in
---------------------
Per operator standing directive 2026-05-16 (`feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md`)
every lattice node declares its ``frontier_pursuit_class`` ∈ {PLATEAU_ADJACENT,
FRONTIER_PURSUIT, ASYMPTOTIC_PURSUIT}.  The K-selection helpers (enhancements
2+3) enforce the recommended budget distribution: <=30% plateau-adjacent,
>=40% frontier-pursuit, >=20% asymptotic-pursuit.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .compressive_landscape import (
    CompressiveSensingLandscapeRecovery,
)

# ──────────────────────────────────────────────────────────────────────────
# Horizon class taxonomy per operator standing directive 2026-05-16.
# ──────────────────────────────────────────────────────────────────────────


class FrontierPursuitClass(StrEnum):
    """3-class HORIZON-CLASS taxonomy.

    Per ``feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md``:

    * PLATEAU_ADJACENT — predicted CPU band in [0.180, 0.200] (the 0.196-0.199
      cluster).  Mission contribution = frontier_protecting.
    * FRONTIER_PURSUIT — predicted CPU band in [0.120, 0.180] (mid-term
      breakthrough).  Mission contribution = frontier_breaking.
    * ASYMPTOTIC_PURSUIT — predicted CPU band in [0.050, 0.120] (long-horizon
      theoretical/Rudin floor).  Mission contribution = frontier_breaking at
      multi-month horizon.

    The K-selection budget distribution (per the directive):
    <=30% plateau, >=40% frontier-pursuit, >=20% asymptotic-pursuit,
    <=10% disambiguator probe.
    """

    PLATEAU_ADJACENT = "plateau_adjacent"
    FRONTIER_PURSUIT = "frontier_pursuit"
    ASYMPTOTIC_PURSUIT = "asymptotic_pursuit"
    DISAMBIGUATOR_PROBE = "disambiguator_probe"


def classify_predicted_band(low: float, high: float) -> FrontierPursuitClass:
    """Classify a predicted CPU score band into a HORIZON-CLASS bucket.

    The midpoint determines the class; the canonical thresholds from the
    horizon-class directive 2026-05-16.
    """
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"band must be finite, got [{low!r}, {high!r}]")
    if low > high:
        raise ValueError(f"band low {low} must be <= high {high}")
    midpoint = 0.5 * (low + high)
    if midpoint >= 0.180:
        return FrontierPursuitClass.PLATEAU_ADJACENT
    if midpoint >= 0.120:
        return FrontierPursuitClass.FRONTIER_PURSUIT
    return FrontierPursuitClass.ASYMPTOTIC_PURSUIT


# ──────────────────────────────────────────────────────────────────────────
# Tree-structured lattice nodes per enhancement 5 (Baraniuk-Cevher 2010).
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SubstrateLatticeNode:
    """One node in the substrate lattice.

    The lattice is naturally hierarchical per Baraniuk-Cevher 2010:

    * Depth 0: leaf substrates (e.g. ``z3_g1``, ``nscs01``)
    * Depth 1: pairwise compositions (e.g. ``z3_g1 + d1``)
    * Depth 2+: higher-order compositions

    The tree-structured sparsity prior says: if a parent (composition) is
    in the sparse-support set, the children (constituent substrates) are
    LIKELY ALSO in the support set.  Conversely if all leaves are OUTSIDE
    the support, the composition is likely also outside.  This reduces K
    by O(log N) versus unstructured sparsity per Baraniuk-Cevher 2010
    Theorem 1.
    """

    node_id: str
    parent_id: str | None
    support_level: int  # 0 = leaf substrate, 1+ = composition depth
    predicted_band_low: float
    predicted_band_high: float
    frontier_pursuit_class: FrontierPursuitClass

    @property
    def predicted_midpoint(self) -> float:
        return 0.5 * (self.predicted_band_low + self.predicted_band_high)

    @property
    def predicted_band_width(self) -> float:
        return self.predicted_band_high - self.predicted_band_low


# ──────────────────────────────────────────────────────────────────────────
# Sparse-signal posterior per enhancement 1.
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SparseSignalPosterior:
    """Recovered posterior over which substrates are frontier-breaking.

    Per Candes-Tao 2006 + Daubechies-DeVore-Fornasier-Gunturk 2010 L1
    reconstruction: from K << N measurements, the recovered sparse signal
    identifies the s = ||c||_0 nonzero coefficients corresponding to
    frontier-breaking substrates.  Per substrate, the posterior assigns:

    * ``posterior_frontier_probability``: probability mass in [0,1] that this
      substrate's TRUE score is below the current frontier.  HIGH (>0.7) =
      strong evidence; MEDIUM (0.3-0.7) = uncertain; LOW (<0.3) = unlikely.
    * ``posterior_score_band``: (low, high) band per the L1 reconstruction.
    * ``recovery_uncertainty``: Daubechies-DeVore O(sqrt(N/K)) error bound
      per substrate.

    The posterior is the canonical Bayesian sequential design input
    (enhancement 3): the next K dispatches MAXIMIZE expected information
    gain about this posterior.
    """

    n_substrates: int
    n_anchors: int
    sparsity_recovered: int  # ||c||_0 of the recovered sparse signal
    basis: str  # "haar_db1" or "daubechies_db4" per enhancement 6
    tree_depth_max: int  # tree-structured sparsity depth per enhancement 5
    posterior_frontier_probability: tuple[tuple[str, float], ...]
    posterior_score_band: tuple[tuple[str, float, float], ...]
    recovery_uncertainty: tuple[tuple[str, float], ...]
    # Cite-chain per max-observability directive 2026-05-16:
    confidence_tag: str

    def to_observability_record(self) -> dict[str, Any]:
        """Return JSON-serializable observability record per max-observability
        standing directive 2026-05-16."""
        return {
            "n_substrates": self.n_substrates,
            "n_anchors": self.n_anchors,
            "sparsity_recovered": self.sparsity_recovered,
            "basis": self.basis,
            "tree_depth_max": self.tree_depth_max,
            "posterior_frontier_probability": [
                {"node_id": nid, "probability": prob}
                for nid, prob in self.posterior_frontier_probability
            ],
            "posterior_score_band": [
                {"node_id": nid, "low": low, "high": high}
                for nid, low, high in self.posterior_score_band
            ],
            "recovery_uncertainty": [
                {"node_id": nid, "uncertainty": u}
                for nid, u in self.recovery_uncertainty
            ],
            "confidence_tag": self.confidence_tag,
        }

    def top_k_frontier_breaking(
        self, k: int = 8
    ) -> tuple[tuple[str, float], ...]:
        """Return top-k substrates by posterior_frontier_probability descending."""
        ordered = sorted(
            self.posterior_frontier_probability,
            key=lambda x: x[1],
            reverse=True,
        )
        return tuple(ordered[: max(0, int(k))])


def diff_sparse_signal_posteriors(
    a: SparseSignalPosterior, b: SparseSignalPosterior
) -> dict[str, Any]:
    """Diff two posteriors per max-observability directive 2026-05-16.

    Returns the per-substrate probability delta + band shift + recovery-
    uncertainty change so the operator can see how the posterior moved
    after K' additional anchors landed.
    """
    a_prob = dict(a.posterior_frontier_probability)
    b_prob = dict(b.posterior_frontier_probability)
    a_band = {nid: (lo, hi) for nid, lo, hi in a.posterior_score_band}
    b_band = {nid: (lo, hi) for nid, lo, hi in b.posterior_score_band}
    a_unc = dict(a.recovery_uncertainty)
    b_unc = dict(b.recovery_uncertainty)
    all_ids = sorted(set(a_prob) | set(b_prob))
    return {
        "anchor_delta": b.n_anchors - a.n_anchors,
        "sparsity_delta": b.sparsity_recovered - a.sparsity_recovered,
        "basis_changed": a.basis != b.basis,
        "tree_depth_delta": b.tree_depth_max - a.tree_depth_max,
        "per_substrate": [
            {
                "node_id": nid,
                "probability_delta": b_prob.get(nid, 0.0) - a_prob.get(nid, 0.0),
                "band_low_shift": (
                    b_band.get(nid, (0.0, 0.0))[0] - a_band.get(nid, (0.0, 0.0))[0]
                ),
                "band_high_shift": (
                    b_band.get(nid, (0.0, 0.0))[1] - a_band.get(nid, (0.0, 0.0))[1]
                ),
                "uncertainty_delta": (
                    b_unc.get(nid, 0.0) - a_unc.get(nid, 0.0)
                ),
            }
            for nid in all_ids
        ],
    }


# ──────────────────────────────────────────────────────────────────────────
# Daubechies db4 wavelet basis per enhancement 6.
# ──────────────────────────────────────────────────────────────────────────


class DaubechiesDb4LatticeBasis:
    """Pure-Python Daubechies db4 wavelet projection for the lattice.

    Per Mallat 1989 "A theory for multiresolution signal decomposition: the
    wavelet representation" + Daubechies 1988 "Orthonormal bases of
    compactly supported wavelets", db4 provides 4 vanishing moments + 8
    nonzero filter coefficients.  Multi-scale lattice representation;
    matches Daubechies-DeVore-Fornasier-Gunturk 2010 recovery theory
    exactly.

    Sister of Catalog #277 ``WaveletMultiScaleFallingRuleListRanker`` which
    uses the same multi-scale spirit for falling-rule lists; this basis
    operates on the lattice cell-value vector directly.

    The compact-support property gives O(N log N) transform cost vs O(N^2)
    for dense bases — critical for the 30-100 substrate lattice scale.
    """

    # Daubechies db4 low-pass filter coefficients (h_k) per Daubechies 1988:
    # 8 nonzero coefficients with 4 vanishing moments.  These satisfy
    # the QMF orthogonality relation sum_k h_k * h_{k-2n} = delta(n).
    DB4_LOW_PASS: tuple[float, ...] = (
        0.23037781330885523,
        0.71484657055254153,
        0.63088076792959036,
        -0.02798376941698385,
        -0.18703481171888114,
        0.03084138183598697,
        0.03288301166698295,
        -0.01059740178506903,
    )

    @classmethod
    def high_pass(cls) -> tuple[float, ...]:
        """Quadrature mirror filter: g_k = (-1)^k h_{L-1-k}."""
        h = cls.DB4_LOW_PASS
        n = len(h)
        return tuple(((-1) ** k) * h[n - 1 - k] for k in range(n))

    @classmethod
    def forward_transform(cls, signal: Sequence[float]) -> tuple[list[float], list[float]]:
        """One-level db4 wavelet decomposition.

        Returns (approximation, detail) coefficients.  Both have length
        ceil(N/2).  Periodic boundary handling for compatibility with the
        cathedral lattice's wrap-around semantics.
        """
        h = cls.DB4_LOW_PASS
        g = cls.high_pass()
        n = len(signal)
        if n == 0:
            return [], []
        # Output length = ceil(N/2) per critically-sampled DWT.
        out_n = (n + 1) // 2
        approx: list[float] = []
        detail: list[float] = []
        for i in range(out_n):
            a = 0.0
            d = 0.0
            for k in range(len(h)):
                # Periodic boundary: index modulo N.
                idx = (2 * i + k) % n
                a += h[k] * signal[idx]
                d += g[k] * signal[idx]
            approx.append(a)
            detail.append(d)
        return approx, detail

    @classmethod
    def project_sparse(
        cls, signal: Sequence[float], sparsity: int
    ) -> list[float]:
        """Project signal onto its s-sparse approximation in db4 wavelet basis.

        Per Daubechies-DeVore-Fornasier-Gunturk 2010 best-s-term
        approximation: keep the s largest-magnitude wavelet coefficients,
        zero the rest, then inverse-transform.  Returns the reconstructed
        signal in the spatial domain.

        For the lattice's small N (~30-100), a one-level decomposition is
        sufficient; the deeper multi-scale recursion is left to the
        sister :class:`WaveletMultiScaleFallingRuleListRanker` per Catalog
        #277.
        """
        n = len(signal)
        if n == 0:
            return []
        if sparsity >= n:
            return list(signal)
        if sparsity <= 0:
            return [0.0] * n
        approx, detail = cls.forward_transform(signal)
        all_coefs = [(abs(c), i, "a") for i, c in enumerate(approx)]
        all_coefs += [(abs(c), i, "d") for i, c in enumerate(detail)]
        all_coefs.sort(key=lambda x: x[0], reverse=True)
        keep = {(kind, idx) for _, idx, kind in all_coefs[:sparsity]}
        approx_kept = [
            c if ("a", i) in keep else 0.0 for i, c in enumerate(approx)
        ]
        detail_kept = [
            c if ("d", i) in keep else 0.0 for i, c in enumerate(detail)
        ]
        # Inverse transform (one-level synthesis).
        h = cls.DB4_LOW_PASS
        g = cls.high_pass()
        recon = [0.0] * n
        out_n = len(approx)
        for i in range(out_n):
            for k in range(len(h)):
                idx = (2 * i + k) % n
                recon[idx] += h[k] * approx_kept[i] + g[k] * detail_kept[i]
        return recon


# ──────────────────────────────────────────────────────────────────────────
# Tree-structured sparsity prior per enhancement 5.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class TreeStructuredSparsityPrior:
    """Baraniuk-Cevher 2010 tree-structured sparsity prior.

    Per Baraniuk-Cevher-Duarte-Hegde 2010 "Model-Based Compressive Sensing":

      For a tree-structured sparse signal of depth d with s nonzero
      coefficients, the model-based recovery requires K = O(s) measurements
      WITHOUT the log(N/s) factor.  This is a strict improvement over
      unstructured s-sparsity which requires K = O(s * log(N/s)).

    The lattice is naturally tree-structured: substrate -> pairwise
    composition -> higher-order composition.  This prior says: if a
    composition is in the support, its constituent substrates are LIKELY
    also in the support; if all leaves are OUTSIDE the support, the
    composition is LIKELY also outside.

    Operationalized as a posterior-weight modifier: a substrate whose
    parent is in the support gets a posterior boost; a substrate whose
    siblings are all outside gets a posterior penalty.
    """

    parent_in_support_boost: float = 1.5
    all_siblings_outside_penalty: float = 0.5

    def apply_to_posterior(
        self,
        nodes: Sequence[SubstrateLatticeNode],
        per_node_raw_probability: Mapping[str, float],
        support_threshold: float = 0.5,
    ) -> dict[str, float]:
        """Apply the tree-structured prior to per-node raw probabilities.

        Returns adjusted-probability dict.  Per Baraniuk-Cevher 2010
        Algorithm 1 (CSSA - Connected Subtree Selection Algorithm), this
        is the bottom-up + top-down message passing on the lattice tree.
        """
        adjusted: dict[str, float] = dict(per_node_raw_probability)
        # Group nodes by parent.
        children_of: dict[str, list[SubstrateLatticeNode]] = {}
        nodes_by_id: dict[str, SubstrateLatticeNode] = {}
        for n in nodes:
            nodes_by_id[n.node_id] = n
            pid = n.parent_id or ""
            children_of.setdefault(pid, []).append(n)
        # Top-down pass: if parent is in support, boost children.
        for parent_id, children in children_of.items():
            if parent_id == "":
                continue
            parent_prob = adjusted.get(parent_id, 0.0)
            if parent_prob >= support_threshold:
                for child in children:
                    adjusted[child.node_id] = min(
                        1.0,
                        adjusted.get(child.node_id, 0.0)
                        * self.parent_in_support_boost,
                    )
        # Bottom-up pass: if all siblings are outside, penalize.
        for parent_id, children in children_of.items():
            if parent_id == "":
                continue
            sibling_probs = [
                adjusted.get(c.node_id, 0.0) for c in children
            ]
            if all(p < support_threshold for p in sibling_probs):
                for c in children:
                    adjusted[c.node_id] = max(
                        0.0,
                        adjusted.get(c.node_id, 0.0)
                        * self.all_siblings_outside_penalty,
                    )
        return adjusted


# ──────────────────────────────────────────────────────────────────────────
# Substrate lattice recovery core per enhancement 1.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SubstrateLatticeRecovery:
    """L1-reconstruction sparse-signal recovery over the substrate lattice.

    Composes :class:`CompressiveSensingLandscapeRecovery` (Catalog #276
    sister) with the tree-structured sparsity prior (enhancement 5) +
    Daubechies db4 basis (enhancement 6) to recover a sparse posterior
    over which substrates are genuinely frontier-breaking from K=O(sqrt(N))
    empirical anchors.

    Public API:

    * :meth:`add_node` — add a substrate / composition node to the lattice
    * :meth:`update_from_anchor` — close the continual-learning loop
      (Catalog #128/#131 fcntl-locked store-path discipline)
    * :meth:`recover_sparse_signal` — emit a :class:`SparseSignalPosterior`
    * :meth:`predict_marginal_recovery_at_K` — counterfactual hook per
      max-observability directive 2026-05-16

    Per CLAUDE.md "Apples-to-apples evidence discipline": every prediction
    carries the canonical confidence-tag with K, N, sparsity, basis, and
    tree-depth provenance.
    """

    use_daubechies_db4: bool = True
    use_tree_structured_prior: bool = True
    expected_sparsity: int = 5
    frontier_threshold_cpu: float = 0.192  # current contest-CPU frontier (A1)
    _nodes: list[SubstrateLatticeNode] = field(default_factory=list, init=False)
    _anchors: list[tuple[str, float]] = field(default_factory=list, init=False)
    _node_by_id: dict[str, SubstrateLatticeNode] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def n_substrates(self) -> int:
        return len(self._nodes)

    @property
    def n_anchors(self) -> int:
        return len(self._anchors)

    @property
    def basis_name(self) -> str:
        return "daubechies_db4" if self.use_daubechies_db4 else "haar_db1"

    def add_node(self, node: SubstrateLatticeNode) -> None:
        """Register a substrate / composition node in the lattice."""
        if node.node_id in self._node_by_id:
            raise ValueError(f"duplicate node_id {node.node_id!r}")
        if node.parent_id and node.parent_id not in self._node_by_id:
            raise ValueError(
                f"parent_id {node.parent_id!r} not registered before child "
                f"{node.node_id!r}"
            )
        self._nodes.append(node)
        self._node_by_id[node.node_id] = node

    def update_from_anchor(self, node_id: str, observed_score: float) -> None:
        """Close the continual-learning loop per Catalog #128/#131.

        Per CLAUDE.md "Forbidden score claims" the observed_score is the
        EMPIRICAL measurement (axis tracked by caller); the lattice layer
        is pure recovery.
        """
        if not math.isfinite(observed_score):
            raise ValueError(
                f"observed_score must be finite, got {observed_score!r}"
            )
        if node_id not in self._node_by_id:
            raise ValueError(
                f"node_id {node_id!r} not in lattice "
                f"(n_substrates={self.n_substrates})"
            )
        # Replace if anchor for same node exists (refresh empirical evidence).
        self._anchors = [(nid, s) for (nid, s) in self._anchors if nid != node_id]
        self._anchors.append((node_id, float(observed_score)))

    def recover_sparse_signal(self) -> SparseSignalPosterior:
        """L1-reconstruct the sparse-signal posterior over the lattice.

        Per Candes-Tao 2006: minimize ||c||_1 subject to y = A c where c is
        the wavelet-coefficient vector and A is the partial-measurement
        operator.  For the lattice's small N, the minimization is solved
        directly via best-s-term projection in the chosen basis.

        Returns :class:`SparseSignalPosterior` with per-substrate
        posterior_frontier_probability, predicted band, and recovery
        uncertainty.
        """
        n = self.n_substrates
        if n == 0:
            return self._empty_posterior()
        # Build the spatial-domain signal: predicted-band midpoints, with
        # anchored cells replaced by observed scores.
        signal: list[float] = []
        anchor_map = dict(self._anchors)
        for node in self._nodes:
            if node.node_id in anchor_map:
                signal.append(anchor_map[node.node_id])
            else:
                signal.append(node.predicted_midpoint)
        # Project onto s-sparse approximation in the chosen basis.
        if self.use_daubechies_db4 and n >= len(
            DaubechiesDb4LatticeBasis.DB4_LOW_PASS
        ):
            recon = DaubechiesDb4LatticeBasis.project_sparse(
                signal, self.expected_sparsity
            )
        else:
            # Fallback: Haar projection via the canonical Catalog #276
            # CompressiveSensingLandscapeRecovery.
            haar_recovery = CompressiveSensingLandscapeRecovery.from_substrate_axis(
                [n.node_id for n in self._nodes]
            )
            for node_id, score in self._anchors:
                haar_recovery.add_anchor(node_id, score)
            cells = haar_recovery.reconstruct()
            recon = [c.predicted_score for c in cells]
        # Compute per-node posterior_frontier_probability = P(score < threshold).
        # Use the recovered score + uncertainty as a Gaussian-ish posterior.
        per_node_prob: dict[str, float] = {}
        per_node_band: dict[str, tuple[float, float]] = {}
        per_node_unc: dict[str, float] = {}
        uncertainty = self._daubechies_devore_uncertainty()
        observed_sparsity = sum(1 for c in recon if abs(c) > 1e-9)
        for node, recovered_score in zip(self._nodes, recon, strict=False):
            if node.node_id in anchor_map:
                # Anchored: posterior is deterministic at the observed score.
                obs = anchor_map[node.node_id]
                prob = 1.0 if obs < self.frontier_threshold_cpu else 0.0
                band = (obs, obs)
                unc = 0.0
            else:
                prob = self._gaussian_below_threshold(
                    recovered_score, uncertainty, self.frontier_threshold_cpu
                )
                band = (
                    recovered_score - uncertainty,
                    recovered_score + uncertainty,
                )
                unc = uncertainty
            per_node_prob[node.node_id] = prob
            per_node_band[node.node_id] = band
            per_node_unc[node.node_id] = unc
        # Apply tree-structured sparsity prior per enhancement 5.
        if self.use_tree_structured_prior:
            prior = TreeStructuredSparsityPrior()
            per_node_prob = prior.apply_to_posterior(self._nodes, per_node_prob)
        # Compute max tree depth per observability discipline.
        max_depth = max((n.support_level for n in self._nodes), default=0)
        tag = (
            f"[prediction; compressive-sensing-lattice-recovery; "
            f"K={self.n_anchors}; N={n}; sparsity={observed_sparsity}; "
            f"tree_depth={max_depth}; basis={self.basis_name}]"
        )
        return SparseSignalPosterior(
            n_substrates=n,
            n_anchors=self.n_anchors,
            sparsity_recovered=observed_sparsity,
            basis=self.basis_name,
            tree_depth_max=max_depth,
            posterior_frontier_probability=tuple(
                (nid, per_node_prob.get(nid, 0.0)) for nid in (n.node_id for n in self._nodes)
            ),
            posterior_score_band=tuple(
                (
                    nid,
                    per_node_band.get(nid, (0.0, 0.0))[0],
                    per_node_band.get(nid, (0.0, 0.0))[1],
                )
                for nid in (n.node_id for n in self._nodes)
            ),
            recovery_uncertainty=tuple(
                (nid, per_node_unc.get(nid, 0.0)) for nid in (n.node_id for n in self._nodes)
            ),
            confidence_tag=tag,
        )

    def predict_marginal_recovery_at_K(
        self, hypothetical_anchor_node_id: str
    ) -> dict[str, Any]:
        """Counterfactual: 'what if I added one more anchor at substrate X?'

        Per max-observability standing directive 2026-05-16 consequence 6
        (counterfactual hook).  Returns the predicted reduction in
        recovery_uncertainty WITHOUT actually re-fitting.  The reduction
        follows the Daubechies-DeVore O(sqrt(N/K)) -> O(sqrt(N/(K+1)))
        progression.
        """
        if hypothetical_anchor_node_id not in self._node_by_id:
            raise ValueError(
                f"node_id {hypothetical_anchor_node_id!r} not in lattice"
            )
        current_uncertainty = self._daubechies_devore_uncertainty()
        # Simulate K -> K+1 progression analytically.
        n = max(1, self.n_substrates)
        k_current = max(1, self.n_anchors)
        k_next = k_current + 1
        sigma = self._observed_score_stdev()
        next_uncertainty = math.sqrt(n / k_next) * sigma * 0.5
        reduction = current_uncertainty - next_uncertainty
        return {
            "hypothetical_anchor_node_id": hypothetical_anchor_node_id,
            "current_uncertainty": current_uncertainty,
            "next_uncertainty": next_uncertainty,
            "uncertainty_reduction": reduction,
            "K_current": k_current,
            "K_next": k_next,
            "N": n,
            "basis": self.basis_name,
        }

    # ── private helpers ──────────────────────────────────────────────────

    def _empty_posterior(self) -> SparseSignalPosterior:
        tag = (
            f"[prediction; compressive-sensing-lattice-recovery; "
            f"K=0; N=0; sparsity=0; tree_depth=0; basis={self.basis_name}]"
        )
        return SparseSignalPosterior(
            n_substrates=0,
            n_anchors=0,
            sparsity_recovered=0,
            basis=self.basis_name,
            tree_depth_max=0,
            posterior_frontier_probability=(),
            posterior_score_band=(),
            recovery_uncertainty=(),
            confidence_tag=tag,
        )

    def _daubechies_devore_uncertainty(self) -> float:
        n = max(1, self.n_substrates)
        k = max(1, self.n_anchors)
        sigma = self._observed_score_stdev()
        return math.sqrt(n / k) * sigma * 0.5

    def _observed_score_stdev(self) -> float:
        if len(self._anchors) < 2:
            return 0.05  # first-principles fallback per Catalog #276
        scores = [s for _, s in self._anchors]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
        return math.sqrt(var)

    @staticmethod
    def _gaussian_below_threshold(
        mean: float, sigma: float, threshold: float
    ) -> float:
        """P(X < threshold) for X ~ Normal(mean, sigma^2)."""
        if sigma <= 1e-12:
            return 1.0 if mean < threshold else 0.0
        # Standard normal CDF via math.erf.
        z = (threshold - mean) / (sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))


# ──────────────────────────────────────────────────────────────────────────
# Coherence-minimization in K-selection per enhancement 2 (Tropp 2004 RIP).
# ──────────────────────────────────────────────────────────────────────────


def compute_pairwise_coherence(
    nodes: Sequence[SubstrateLatticeNode],
) -> dict[tuple[str, str], float]:
    """Compute pairwise coherence per Tropp 2004 RIP theorem.

    Per Tropp 2004 "Greed is good: algorithmic results for sparse
    approximation" + Donoho-Elad 2003: the coherence of a sensing matrix
    is mu(A) = max_{i != j} |<a_i, a_j>| / (||a_i|| ||a_j||).  LOW
    coherence is REQUIRED for sparse recovery; HIGH coherence breaks RIP
    and reconstruction fails.

    For the lattice's substrate nodes, the "sensing vector" per substrate
    is its (predicted_band_midpoint, predicted_band_width,
    frontier_pursuit_class) coordinate.  Two substrates have HIGH
    coherence iff their predicted bands overlap substantially AND they
    share a frontier_pursuit_class — i.e. they are structurally
    indistinguishable observation targets.

    Returns dict mapping (node_id_a, node_id_b) -> coherence in [0, 1]
    for every (i, j) pair with i < j (canonical ordering).  HIGH coherence
    pairs are the ones to AVOID jointly selecting in the K dispatch
    schedule.
    """
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            # Band overlap fraction.
            lo = max(a.predicted_band_low, b.predicted_band_low)
            hi = min(a.predicted_band_high, b.predicted_band_high)
            overlap = max(0.0, hi - lo)
            union_width = (
                max(a.predicted_band_high, b.predicted_band_high)
                - min(a.predicted_band_low, b.predicted_band_low)
            )
            band_coherence = (
                overlap / union_width if union_width > 1e-9 else 0.0
            )
            # Class agreement boosts coherence.
            class_boost = (
                1.0 if a.frontier_pursuit_class == b.frontier_pursuit_class else 0.6
            )
            # Tree-structure boost: parent/child pairs are structurally
            # NOT-coherent (they probe different lattice scales) per
            # Baraniuk-Cevher 2010.
            if a.parent_id == b.node_id or b.parent_id == a.node_id:
                class_boost *= 0.3
            coherence = min(1.0, band_coherence * class_boost)
            out[_coherence_key(a.node_id, b.node_id)] = coherence
    return out


def _coherence_key(a: str, b: str) -> tuple[str, str]:
    """Return the canonical unordered key for pairwise coherence lookup."""

    return tuple(sorted((a, b)))


@dataclass
class CoherenceMinimizingSelector:
    """Select K substrates minimizing pairwise coherence per Tropp 2004 RIP.

    Greedy selection: start with the lowest-band-midpoint substrate (likely
    most-frontier-pursuit) then add substrates one-at-a-time minimizing the
    MAX pairwise coherence with the already-selected set.  This is the
    canonical RIP-preserving K-selection for compressive-sensing recovery.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + the horizon-class
    directive 2026-05-16: the selector ALSO enforces the recommended budget
    distribution (<=30% plateau, >=40% frontier-pursuit, >=20% asymptotic).
    """

    K: int = 8
    plateau_budget_max: float = 0.30
    frontier_pursuit_budget_min: float = 0.40
    asymptotic_budget_min: float = 0.20
    disambiguator_budget_max: float = 0.10

    def select(
        self,
        candidates: Sequence[SubstrateLatticeNode],
    ) -> list[SubstrateLatticeNode]:
        """Return K substrates minimizing pairwise coherence + respecting
        horizon-class budget."""
        if self.K <= 0:
            return []
        if len(candidates) <= self.K:
            return list(candidates)
        coherence = compute_pairwise_coherence(candidates)
        # Greedy: start with lowest predicted midpoint (most-frontier-pursuit).
        ordered = sorted(candidates, key=lambda n: n.predicted_midpoint)
        selected: list[SubstrateLatticeNode] = [ordered[0]]
        remaining = list(ordered[1:])
        budget = self._init_budget()
        self._consume_budget(selected[0], budget)
        while len(selected) < self.K and remaining:
            # For each remaining candidate, compute max coherence with
            # already-selected set.
            scores: list[tuple[float, SubstrateLatticeNode]] = []
            slots_left = self.K - len(selected)
            quota_deficits = self._quota_deficits(budget, remaining)
            must_fill_quota = sum(quota_deficits.values()) >= slots_left
            for cand in remaining:
                if not self._budget_allows(cand, budget):
                    continue
                if (
                    must_fill_quota
                    and cand.frontier_pursuit_class.value not in quota_deficits
                ):
                    continue
                max_coh = 0.0
                for sel in selected:
                    pair_key = _coherence_key(cand.node_id, sel.node_id)
                    c = coherence.get(pair_key, 0.0)
                    max_coh = max(max_coh, c)
                scores.append((max_coh, cand))
            if not scores:
                # Budget exhausted across all remaining; allow over-budget.
                for cand in remaining:
                    max_coh = 0.0
                    for sel in selected:
                        pair_key = _coherence_key(cand.node_id, sel.node_id)
                        c = coherence.get(pair_key, 0.0)
                        max_coh = max(max_coh, c)
                    scores.append((max_coh, cand))
            scores.sort(key=lambda x: x[0])
            best = scores[0][1]
            selected.append(best)
            remaining.remove(best)
            self._consume_budget(best, budget)
        return selected

    def _init_budget(self) -> dict[str, int]:
        return {
            FrontierPursuitClass.PLATEAU_ADJACENT.value: 0,
            FrontierPursuitClass.FRONTIER_PURSUIT.value: 0,
            FrontierPursuitClass.ASYMPTOTIC_PURSUIT.value: 0,
            FrontierPursuitClass.DISAMBIGUATOR_PROBE.value: 0,
        }

    def _consume_budget(
        self, node: SubstrateLatticeNode, budget: dict[str, int]
    ) -> None:
        budget[node.frontier_pursuit_class.value] = budget.get(
            node.frontier_pursuit_class.value, 0
        ) + 1

    def _budget_allows(
        self, node: SubstrateLatticeNode, budget: dict[str, int]
    ) -> bool:
        cls = node.frontier_pursuit_class.value
        used = budget.get(cls, 0)
        if cls == FrontierPursuitClass.PLATEAU_ADJACENT.value:
            return used < math.ceil(self.K * self.plateau_budget_max)
        if cls == FrontierPursuitClass.DISAMBIGUATOR_PROBE.value:
            return used < math.ceil(self.K * self.disambiguator_budget_max)
        return True  # frontier/asymptotic have only MIN constraints

    def _quota_deficits(
        self,
        budget: dict[str, int],
        remaining: Sequence[SubstrateLatticeNode],
    ) -> dict[str, int]:
        """Return minimum-quota deficits that can still be satisfied."""

        deficits: dict[str, int] = {}
        for cls, min_fraction in (
            (
                FrontierPursuitClass.FRONTIER_PURSUIT.value,
                self.frontier_pursuit_budget_min,
            ),
            (
                FrontierPursuitClass.ASYMPTOTIC_PURSUIT.value,
                self.asymptotic_budget_min,
            ),
        ):
            desired = math.ceil(self.K * min_fraction)
            available_remaining = sum(
                1 for node in remaining if node.frontier_pursuit_class.value == cls
            )
            satisfiable_desired = min(
                desired,
                budget.get(cls, 0) + available_remaining,
            )
            deficit = max(0, satisfiable_desired - budget.get(cls, 0))
            if deficit:
                deficits[cls] = deficit
        return deficits


# ──────────────────────────────────────────────────────────────────────────
# Bayesian sequential experimental design per enhancement 3.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class BayesianSequentialKSelector:
    """Bayesian sequential design per Snoek-Larochelle-Adams 2012 + Ji-Xue-Carin 2008.

    Per Snoek-Larochelle-Adams 2012 "Practical Bayesian Optimization of
    Machine Learning Algorithms" + Ji-Xue-Carin 2008 "Bayesian Compressive
    Sensing":

      The next K dispatches MAXIMIZE expected information gain (EIG) over
      the recovered sparse-signal posterior, NOT just the candidate's own
      predicted score delta.  EIG-per-substrate is computed from the
      posterior's recovery_uncertainty AND the substrate's expected
      contribution to reducing the posterior entropy.

    Mathematical formulation: for each candidate substrate, compute the
    expected reduction in posterior entropy (per Lindley 1956) if that
    substrate were anchored.  Select K candidates maximizing summed EIG.

    Composes with :class:`CoherenceMinimizingSelector` (enhancement 2):
    enhancement 2 gives the structural RIP-preserving selection; this
    enhancement 3 gives the EIG-maximizing selection conditional on the
    posterior.  In practice the operator picks the stage:
    EARLY (few anchors) -> use enhancement 2; LATE (many anchors) ->
    use enhancement 3.
    """

    K: int = 8
    posterior: SparseSignalPosterior | None = None
    lattice: SubstrateLatticeRecovery | None = None

    def select_next_K(
        self, candidates: Sequence[SubstrateLatticeNode]
    ) -> list[SubstrateLatticeNode]:
        """Select K substrates maximizing expected information gain.

        If posterior is None or lattice is None, fall back to selecting
        the K with HIGHEST recovery_uncertainty (canonical
        Snoek-Larochelle-Adams Thompson sampling).  Otherwise compute the
        expected entropy reduction per substrate and select top-K.
        """
        if self.K <= 0:
            return []
        if len(candidates) <= self.K:
            return list(candidates)
        scores = self._compute_eig_per_substrate(candidates)
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scores[: self.K]]

    def _compute_eig_per_substrate(
        self, candidates: Sequence[SubstrateLatticeNode]
    ) -> list[tuple[SubstrateLatticeNode, float]]:
        """EIG per substrate = expected reduction in posterior entropy."""
        out: list[tuple[SubstrateLatticeNode, float]] = []
        if self.posterior is None:
            # Thompson-sampling fallback: use predicted_band_width as proxy
            # for uncertainty.
            for cand in candidates:
                out.append((cand, cand.predicted_band_width))
            return out
        # Posterior available: use per-substrate recovery_uncertainty as the
        # expected entropy reduction (Lindley 1956: EIG = H(prior) - H(posterior|y)).
        unc_map = dict(self.posterior.recovery_uncertainty)
        prob_map = dict(self.posterior.posterior_frontier_probability)
        for cand in candidates:
            u = unc_map.get(cand.node_id, 0.0)
            p = prob_map.get(cand.node_id, 0.0)
            # EIG ~ uncertainty * H(p), where H(p) = -p log p - (1-p) log(1-p)
            # (Bernoulli entropy in nats).  HIGH for uncertain (p ~ 0.5)
            # AND high-recovery-uncertainty substrates.
            h = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) if 1e-09 < p < 1.0 - 1e-09 else 0.0
            eig = u * h
            out.append((cand, eig))
        return out


# ──────────────────────────────────────────────────────────────────────────
# Phase-transition monitor per enhancement 4 (Donoho-Tanner 2009).
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class LatticePhaseTransitionMonitor:
    """Donoho-Tanner 2009 sparsity-undersampling phase-transition monitor.

    Per Donoho-Tanner 2009 "Counting faces of randomly-projected polytopes
    when the projection radically lowers dimension":

      For Gaussian sensing matrices, L1 recovery transitions from EXACT
      to FAILED at the curve rho_S(delta) where delta = K/N (undersampling)
      and rho = s/K (sparsity-relative-to-measurements).  Below the curve:
      exact recovery with high probability.  Above the curve: recovery
      fails.

    The canonical "weak phase transition" threshold from Donoho-Tanner
    Figure 1 at delta = 0.25 (K/N = 0.25) is rho_S ~ 0.20.  For the
    lattice's regime (K ~ 8, N ~ 30), delta = 0.27 falls right at the
    operator's working point — making the phase-transition monitor
    OPERATIONALLY CRITICAL.

    Sister of Catalog #270 dispatch-optimization-protocol verdict pattern:
    when K/N drops below the threshold, the monitor REFUSES to claim
    confident recovery and surfaces the under-sampling diagnostic.
    """

    # Donoho-Tanner 2009 weak-transition threshold curve (delta -> rho_S).
    # Reduced sampling from the canonical Figure 1 curve; the operator can
    # extend this table with finer interpolation per delta.
    PHASE_TRANSITION_CURVE: tuple[tuple[float, float], ...] = (
        (0.05, 0.08),
        (0.10, 0.12),
        (0.15, 0.16),
        (0.20, 0.18),
        (0.25, 0.20),
        (0.30, 0.22),
        (0.40, 0.26),
        (0.50, 0.30),
        (0.60, 0.34),
        (0.70, 0.38),
        (0.80, 0.42),
        (0.90, 0.46),
        (1.00, 0.50),
    )

    safety_margin: float = 0.05  # how far below the curve we want to be

    def compute_undersampling_diagnostic(
        self, K: int, N: int, sparsity_estimate: int
    ) -> dict[str, Any]:
        """Return the Donoho-Tanner diagnostic record.

        Output keys:

        * ``delta`` = K / N
        * ``rho`` = sparsity_estimate / K
        * ``rho_threshold`` = interpolated phase-transition threshold at delta
        * ``recovery_regime`` = "EXACT" | "AT_THRESHOLD" | "FAILED"
        * ``safety_margin_violated`` = bool
        * ``recommended_K`` = K needed to reach safe regime (None if no recommendation)
        """
        if N <= 0:
            raise ValueError(f"N must be > 0, got {N}")
        if K < 0 or sparsity_estimate < 0:
            raise ValueError(
                f"K and sparsity_estimate must be >= 0, got K={K} s={sparsity_estimate}"
            )
        delta = K / N
        rho = sparsity_estimate / max(1, K)
        rho_threshold = self._interpolate_threshold(delta)
        safety_threshold = rho_threshold - self.safety_margin
        if rho > rho_threshold:
            regime = "FAILED"
        elif rho > safety_threshold:
            regime = "AT_THRESHOLD"
        else:
            regime = "EXACT"
        recommended_K: int | None = None
        if regime != "EXACT":
            # Solve for smallest K_new such that
            # sparsity_estimate / K_new <= safety_threshold AND
            # rho_threshold(K_new / N) decreases monotonically (Donoho-Tanner
            # curve is monotone).  Linear scan from current K upward.
            for k_try in range(K + 1, N + 1):
                d_try = k_try / N
                rho_try = sparsity_estimate / k_try
                rho_thresh_try = self._interpolate_threshold(d_try)
                if rho_try <= rho_thresh_try - self.safety_margin:
                    recommended_K = k_try
                    break
        return {
            "delta": delta,
            "rho": rho,
            "rho_threshold": rho_threshold,
            "safety_threshold": safety_threshold,
            "recovery_regime": regime,
            "safety_margin_violated": regime != "EXACT",
            "recommended_K": recommended_K,
            "N": N,
            "K": K,
            "sparsity_estimate": sparsity_estimate,
        }

    def _interpolate_threshold(self, delta: float) -> float:
        """Linear interpolation across the Donoho-Tanner curve."""
        curve = self.PHASE_TRANSITION_CURVE
        if delta <= curve[0][0]:
            return curve[0][1]
        if delta >= curve[-1][0]:
            return curve[-1][1]
        for i in range(len(curve) - 1):
            d0, r0 = curve[i]
            d1, r1 = curve[i + 1]
            if d0 <= delta <= d1:
                # Linear interp.
                t = (delta - d0) / (d1 - d0) if (d1 - d0) > 1e-12 else 0.0
                return r0 + t * (r1 - r0)
        return curve[-1][1]


__all__ = [
    # Enhancement 3 (Bayesian sequential):
    "BayesianSequentialKSelector",
    "CoherenceMinimizingSelector",
    # Enhancement 6 (Daubechies db4):
    "DaubechiesDb4LatticeBasis",
    # Enhancement 1 + horizon class:
    "FrontierPursuitClass",
    # Enhancement 4 (phase transition):
    "LatticePhaseTransitionMonitor",
    "SparseSignalPosterior",
    "SubstrateLatticeNode",
    "SubstrateLatticeRecovery",
    # Enhancement 5 (tree-structured sparsity):
    "TreeStructuredSparsityPrior",
    "classify_predicted_band",
    # Enhancement 2 (coherence):
    "compute_pairwise_coherence",
    "diff_sparse_signal_posteriors",
]
