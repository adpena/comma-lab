# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #9 - per-pair gradient clustering.

Per RESPAWN-MG-7-BUNDLE 2026-05-20. Consumes per-pair ``M_contest`` via the
producer's ``cluster_pairs_by_gradient_similarity`` helper. Pairs in the same
cluster are perceptually equivalent to the scorer; the encoder can share
codec parameters across cluster members (symmetry-breaking exploit). Auto-
discovered by cathedral autopilot ranker per Catalog #335 canonical contract.

## Canonical-vs-unique decision per layer

- Per-pair clustering: ADOPT canonical
  ``tac.master_gradient_comparison.multi_granularity.cluster_pairs_by_gradient_similarity``
  producer surface (greedy cosine-similarity clustering).
- Provenance contract: ADOPT canonical Provenance per Catalog #323.
- Routing markers: ADOPT Catalog #341 canonical non-promotable markers
  (clustering is CODEC-SHARING guidance).
- Codec-side wire-in: FORK to
  ``src/tac/codec/per_cluster_codec_sharing.py`` (scaffold) because no
  canonical helper currently exposes cluster-based codec parameter sharing.

## Observability surface

Per Catalog #305:

1. Inspectable per layer: ``cluster_pairs_by_gradient_similarity`` returns
   tuple of ``EquivalenceClass`` rows; per-cluster membership is queryable.
2. Decomposable per signal: per-cluster decomposes into representative +
   members + intra-class cosine.
3. Diff-able across runs: cluster assignments tied to M_contest sha256.
4. Queryable post-hoc: operator-facing CLI
   ``tools/identify_pair_equivalence_classes.py``.
5. Cite-able: producer surface invocation cited in provenance.
6. Counterfactual-able: cluster structure lets operator ask "what if we
   shared codec params across cluster N?" without re-running scorer.

## 9-dimension success checklist evidence

1. UNIQUENESS: per-pair gradient clustering is canonically distinct from
   uniform per-pair coding; symmetry-breaking exploit.
2. BEAUTY+ELEGANCE: ~180 LOC consumer; greedy cosine clustering kernel.
3. DISTINCTNESS: distinct from sister exploits.
4. RIGOR: cosine similarity is bounded; threshold-based clustering is
   well-defined.
5. OPTIMIZATION-PER-TECHNIQUE: threshold parameter allows substrate to
   define equivalence empirically.
6. STACK-OF-STACKS-COMPOSABILITY: clustering composes with per-pair
   difficulty atlas (sister MG-4) + bit allocator.
7. DETERMINISTIC-REPRODUCIBILITY: deterministic greedy clustering.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_pairs^2 * D) cosine matrix.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: clustering enables param sharing which
   reduces archive size + improves rate term.

## Cargo-cult audit per assumption

- ASSUMPTION: cosine similarity threshold 0.95 is the correct cluster
  membership criterion. CLASSIFICATION: CARGO-CULTED. The threshold is a
  hyperparameter; per-substrate empirical calibration required. Consumer
  exposes threshold as parameter.
- ASSUMPTION: pairs in same cluster can share codec params without score
  regression. CLASSIFICATION: CARGO-CULTED-PENDING-EMPIRICAL. Cluster
  membership predicts SIMILAR scorer response; codec sharing requires
  per-substrate validation that the shared params do not regress score.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from tac.cathedral.consumer_contract import AxisDecomposition, HookNumber


CONSUMER_NAME = "per_pair_gradient_clustering_consumer"
CONSUMER_VERSION = "1.1.0"  # bumped: per-axis decomposition emission (Dim 3 Step 3.4)
_PROVENANCE_MODEL_ID = "per_pair_gradient_clustering_consumer.predicted_axis_decomposition_v1"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Future contest-CUDA anchors with empirical evidence on cluster-sharing
    score regression SHOULD inform substrate-specific threshold defaults
    via this hook.
    """
    _ = anchor


def _build_per_axis_decomposition(
    equivalence_classes: Any,
    pose_axis_dominant_pair_indices: list[int] | None,
    per_cluster_archive_bytes_saved: int,
    per_cluster_pose_axis_savings: float,
    m_contest_sha: str | None,
) -> AxisDecomposition:
    """Build canonical per-axis decomposition with Provenance.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.4: this consumer is
    pose-axis-dominant for ego-motion pairs (per-pair clustering surfaces
    pairs whose pose-gradient signature is similar; codec-sharing across
    cluster members reduces archive bytes AND can recover pose-axis fidelity
    via shared optimization). Seg-axis contribution defaults to 0 (clustering
    does not move per-class chroma). Rate-axis = archive bytes saved via
    cluster-based parameter sharing (caller-supplied; defaults 0).
    """
    try:
        from tac.provenance.builders import build_provenance_for_predicted
        from tac.provenance.validator import provenance_to_dict
    except ImportError:  # pragma: no cover
        canonical_provenance: Mapping[str, Any] = {
            "artifact_kind": "predicted_from_model",
            "model_id": _PROVENANCE_MODEL_ID,
            "measurement_axis": "[predicted]",
            "evidence_grade": "predicted",
            "promotion_eligible": False,
            "score_claim_valid": False,
        }
    else:
        sha_seed = m_contest_sha or "no_m_contest_sha"
        n_clusters = len(equivalence_classes) if equivalence_classes else 0
        inputs_seed = (
            f"{_PROVENANCE_MODEL_ID}:m_contest_sha={sha_seed}:n_clusters={n_clusters}"
        )
        inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
        prov = build_provenance_for_predicted(
            model_id=_PROVENANCE_MODEL_ID,
            inputs_sha256=inputs_sha256,
            measurement_axis="[predicted]",
            hardware_substrate="cpu_local",
        )
        canonical_provenance = provenance_to_dict(prov)

    return AxisDecomposition(
        predicted_d_seg_delta=0.0,  # clustering does not move seg-class chroma
        predicted_d_pose_delta=float(per_cluster_pose_axis_savings),
        predicted_archive_bytes_delta=int(per_cluster_archive_bytes_saved),
        axis_tag="[predicted]",
        canonical_provenance=canonical_provenance,
    )


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.4: also emits canonical
    ``predicted_axis_decomposition`` (pose-axis-dominant for ego-motion
    clusters; seg=0; rate=archive_bytes_saved via codec sharing). Per-axis
    emission is OBSERVABILITY-ONLY per Catalog #341.
    """
    equivalence_classes = candidate.get("pair_equivalence_classes")
    m_contest_sha = candidate.get("m_contest_array_sha256")
    threshold = candidate.get("similarity_threshold")
    pose_dominant_pairs = candidate.get("pose_axis_dominant_pair_indices")
    per_cluster_bytes_saved = int(candidate.get("per_cluster_archive_bytes_saved") or 0)
    per_cluster_pose_savings = float(
        candidate.get("per_cluster_pose_axis_savings") or 0.0
    )

    rationale_parts = [
        "per-pair gradient clustering consumer (exploit #9 symmetry-breaking)",
        "non-promotable codec-sharing guidance per Catalog #341",
    ]
    if threshold is not None:
        rationale_parts.append(f"similarity_threshold={threshold}")
    if equivalence_classes is not None:
        rationale_parts.append(
            f"upstream pair_equivalence_classes n={len(equivalence_classes)}"
        )
    if m_contest_sha is not None:
        rationale_parts.append(f"M_contest sha256[:12]={str(m_contest_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    decomposition = _build_per_axis_decomposition(
        equivalence_classes=equivalence_classes,
        pose_axis_dominant_pair_indices=pose_dominant_pairs,
        per_cluster_archive_bytes_saved=per_cluster_bytes_saved,
        per_cluster_pose_axis_savings=per_cluster_pose_savings,
        m_contest_sha=m_contest_sha,
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "consumer_signal_kind": "per_pair_codec_sharing_recommendation",
        "pair_equivalence_classes": equivalence_classes,
        "similarity_threshold": threshold,
        "m_contest_array_sha256": m_contest_sha,
        "predicted_axis_decomposition": decomposition.as_dict(),
    }


def cluster_pairs_by_gradient_similarity_consumer_api(
    M_contest_per_pair,
    similarity_threshold: float = 0.95,
):
    """Cluster pairs whose gradient signatures cosine-similar >= threshold.

    Thin wrapper that exposes a consumer-friendly API on top of the
    producer surface. Per Catalog #318: ``M_contest_per_pair`` should be
    derived from canonical
    ``tac.master_gradient_comparison.multi_granularity.extract_M_contest``.

    Args:
        M_contest_per_pair: np.ndarray of shape (N_pairs, ...) - the
            per-pair gradient tensor; can be (N_pairs, 3, H, W) or any
            other rank as long as axis 0 is pairs. Flattened internally
            for cosine similarity.
        similarity_threshold: cosine similarity threshold for cluster
            membership; in [0, 1]. Default 0.95.

    Returns:
        List of clusters; each cluster is a list of pair indices
        (representative first; members in ascending index order).

    Raises:
        ValueError: on invalid threshold or empty input.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for clustering") from exc

    arr = np.asarray(M_contest_per_pair, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError(
            f"M_contest_per_pair must have at least 1 axis; got shape {arr.shape}"
        )
    if not (0.0 <= similarity_threshold <= 1.0):
        raise ValueError(
            f"similarity_threshold must be in [0, 1]; got {similarity_threshold}"
        )
    n_pairs = arr.shape[0]
    if n_pairs == 0:
        return []

    flat = arr.reshape(n_pairs, -1)
    norms = np.linalg.norm(flat, axis=1)
    safe_norms = np.where(norms > 0, norms, 1.0)
    normalized = flat / safe_norms[:, None]

    assigned = np.zeros(n_pairs, dtype=bool)
    clusters: list[list[int]] = []
    for rep_idx in range(n_pairs):
        if assigned[rep_idx]:
            continue
        rep_vec = normalized[rep_idx]
        sims = normalized @ rep_vec
        if norms[rep_idx] == 0:
            assigned[rep_idx] = True
            clusters.append([rep_idx])
            continue
        member_mask = (sims >= similarity_threshold) & (~assigned) & (norms > 0)
        member_mask[rep_idx] = True
        member_indices = [int(i) for i in np.flatnonzero(member_mask)]
        for m_idx in member_indices:
            assigned[m_idx] = True
        # Representative first, then ascending member indices.
        sorted_members = sorted(member_indices)
        if rep_idx in sorted_members:
            sorted_members.remove(rep_idx)
        clusters.append([rep_idx] + sorted_members)
    return clusters


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "cluster_pairs_by_gradient_similarity_consumer_api",
    "consume_candidate",
    "update_from_anchor",
    "_build_per_axis_decomposition",
]
