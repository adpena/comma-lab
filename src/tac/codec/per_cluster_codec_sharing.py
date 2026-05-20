# SPDX-License-Identifier: MIT
"""Codec-side wire-in SCAFFOLD for per-cluster codec parameter sharing.

Per RESPAWN-MG-7-BUNDLE exploit #9 (sister consumer at
``tac.cathedral_consumers.per_pair_gradient_clustering_consumer``). Pairs
clustered by gradient similarity (cosine >= threshold) can SHARE codec
parameters without score regression (symmetry-breaking exploit).

**SCAFFOLD STATUS**: per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY" non-negotiable + Catalog #220 + Catalog #240, this module is
DESIGN-TIME SCAFFOLD ONLY. Production codec adoption requires:

1. Empirical paired contest-CUDA dispatch comparing per-cluster shared
   codec params vs per-pair independent codec params; the shared params
   MUST not measurably regress score.
2. The codec MUST carry research_only=True OR pass Catalog #325 per-
   substrate symposium gate.
3. The codec MUST route per-pair gradient similarity through the
   canonical producer
   ``tac.master_gradient_comparison.multi_granularity.cluster_pairs_by_gradient_similarity``.

## Canonical-vs-unique decision per layer

- Clustering surface: ADOPT canonical producer.
- Codec parameter sharing layer: FORK because no canonical helper currently
  exposes cluster-based codec parameter sharing.
- Provenance contract: ADOPT canonical Provenance per Catalog #323.

## Observability surface

(See sister consumer.)

## 9-dimension success checklist evidence

(See sister consumer.)

## Cargo-cult audit per assumption

(See sister consumer.)
"""
from __future__ import annotations

from collections.abc import Sequence


def build_per_cluster_codec_params_scaffold(
    cluster_assignments: Sequence[Sequence[int]],
    n_total_pairs: int,
    *,
    base_codec_params_per_pair=None,
) -> dict:
    """Build per-cluster codec parameter sharing manifest (SCAFFOLD).

    For each cluster, the representative pair's codec params are designated
    as the shared params; all cluster members inherit them. The output
    manifest maps cluster_idx -> (representative_pair_idx, member_pair_indices,
    shared_codec_params).

    Per Catalog #287 evidence-tag discipline: this scaffold's output is
    [predicted] axis tag; production codec adoption requires paired-axis
    empirical evidence that the shared params do not regress score.

    Args:
        cluster_assignments: list of clusters, each cluster a list of pair
            indices (representative first per the consumer API).
        n_total_pairs: total number of pairs (for validation).
        base_codec_params_per_pair: optional list/dict of per-pair codec
            params; if provided, representative pair's params are used as
            cluster-shared params; if None, params are left as None and
            substrate fills in.

    Returns:
        Dict with structure:
            {
                "schema": "per_cluster_codec_sharing_scaffold_v1",
                "n_clusters": <int>,
                "n_total_pairs": <int>,
                "n_pairs_in_clusters": <int>,
                "clusters": [
                    {
                        "cluster_idx": <int>,
                        "representative_pair_idx": <int>,
                        "member_pair_indices": [<int>, ...],
                        "shared_codec_params": <object or None>,
                    },
                    ...
                ],
                "scaffold_status": <get_scaffold_status() output>,
            }

    Raises:
        ValueError: on invalid input.
    """
    if not isinstance(cluster_assignments, Sequence):
        raise ValueError(
            f"cluster_assignments must be Sequence; got {type(cluster_assignments).__name__}"
        )
    if n_total_pairs < 0:
        raise ValueError(f"n_total_pairs must be >= 0; got {n_total_pairs}")

    assigned_pairs: set[int] = set()
    clusters_out: list[dict] = []
    for cluster_idx, cluster in enumerate(cluster_assignments):
        if not isinstance(cluster, Sequence) or len(cluster) == 0:
            raise ValueError(
                f"cluster {cluster_idx} must be non-empty Sequence; got {cluster!r}"
            )
        members = [int(p) for p in cluster]
        for p in members:
            if p < 0 or p >= n_total_pairs:
                raise ValueError(
                    f"pair index {p} in cluster {cluster_idx} out of range "
                    f"[0, {n_total_pairs})"
                )
            if p in assigned_pairs:
                raise ValueError(
                    f"pair index {p} assigned to multiple clusters "
                    f"(cluster {cluster_idx} duplicates an earlier cluster)"
                )
            assigned_pairs.add(p)
        rep_idx = members[0]  # representative first per consumer API
        if base_codec_params_per_pair is not None:
            try:
                shared_params = base_codec_params_per_pair[rep_idx]
            except (IndexError, KeyError, TypeError):
                shared_params = None
        else:
            shared_params = None
        clusters_out.append(
            {
                "cluster_idx": cluster_idx,
                "representative_pair_idx": rep_idx,
                "member_pair_indices": members,
                "shared_codec_params": shared_params,
            }
        )

    return {
        "schema": "per_cluster_codec_sharing_scaffold_v1",
        "n_clusters": len(clusters_out),
        "n_total_pairs": n_total_pairs,
        "n_pairs_in_clusters": len(assigned_pairs),
        "clusters": clusters_out,
        "scaffold_status": get_scaffold_status(),
    }


def get_scaffold_status() -> dict:
    """Return scaffold status metadata per Catalog #220 / #240 contract."""
    return {
        "scaffold_kind": "codec_per_cluster_parameter_sharing",
        "exploit_id": 9,
        "exploit_name": "per_pair_gradient_clustering",
        "research_only": True,
        "dispatch_enabled": False,
        "production_adoption_blockers": (
            "paired_contest_cuda_dispatch_required",
            "per_substrate_symposium_required",
            "no_score_regression_proof_required",
            "canonical_producer_surface_routing_required",
        ),
        "canonical_helper_invocation": (
            "tac.codec.per_cluster_codec_sharing.build_per_cluster_codec_params_scaffold"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
    }


__all__ = [
    "build_per_cluster_codec_params_scaffold",
    "get_scaffold_status",
]
