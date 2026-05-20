#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for exploit #9 - per-pair equivalence classes.

Per RESPAWN-MG-7-BUNDLE 2026-05-20.

Usage::

    .venv/bin/python tools/identify_pair_equivalence_classes.py \\
        --m-contest <path.npy> [--similarity-threshold 0.95] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-pair equivalence classes per exploit #9",
    )
    parser.add_argument("--m-contest", required=True, type=Path)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required", file=sys.stderr)
        return 2

    from tac.cathedral_consumers.per_pair_gradient_clustering_consumer import (
        cluster_pairs_by_gradient_similarity_consumer_api,
    )

    if not args.m_contest.is_file():
        print(f"FATAL: M_contest not found: {args.m_contest}", file=sys.stderr)
        return 2

    m_contest = np.load(args.m_contest)
    clusters = cluster_pairs_by_gradient_similarity_consumer_api(
        m_contest, similarity_threshold=args.similarity_threshold,
    )

    n_pairs = int(m_contest.shape[0])
    n_clusters = len(clusters)
    n_in_clusters = sum(len(c) for c in clusters)

    payload = {
        "schema": "identify_pair_equivalence_classes_v1",
        "exploit_id": 9,
        "exploit_name": "per_pair_gradient_clustering",
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "n_pairs_in_clusters": n_in_clusters,
        "similarity_threshold": args.similarity_threshold,
        "clusters": [
            {
                "cluster_idx": i,
                "representative_pair_idx": cluster[0],
                "member_pair_indices": list(cluster),
                "size": len(cluster),
            }
            for i, cluster in enumerate(clusters)
        ],
        "axis_tag": "[predicted]",
        "promotable": False,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Per-pair equivalence classes (threshold={args.similarity_threshold}):")
        print(f"  n_pairs={n_pairs}  n_clusters={n_clusters}  n_in_clusters={n_in_clusters}")
        # Show clusters with >1 member; singletons are not interesting for sharing.
        multi_clusters = [c for c in payload["clusters"] if c["size"] > 1]
        print(f"  multi-member clusters: {len(multi_clusters)}")
        for c in multi_clusters[:20]:
            members_preview = c["member_pair_indices"][:10]
            ellipsis = "..." if c["size"] > 10 else ""
            print(f"    cluster {c['cluster_idx']:3d} (size={c['size']:3d}): "
                  f"rep={c['representative_pair_idx']} members={members_preview}{ellipsis}")
        if len(multi_clusters) > 20:
            print(f"    ... ({len(multi_clusters) - 20} more multi-clusters)")
        print(f"\naxis_tag: {payload['axis_tag']} (codec-sharing guidance; non-promotable per Catalog #341)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
