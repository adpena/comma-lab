#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fit Lane WC Curator outlier weights from SegNet feature geometry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.curator_outlier import CuratorOutlierScorer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Cosmos Curator soft-DTW outlier pair weights"
    )
    parser.add_argument("--segnet-features", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-pca-components", type=int, default=3)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--soft-dtw-gamma", type=float, default=0.1)
    parser.add_argument("--outlier-quantile", type=float, default=0.95)
    parser.add_argument("--weight-scale", type=float, default=5.0)
    return parser.parse_args(argv)


def _load_feature_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("features", "segnet_features"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(
        f"{path} must contain a torch.Tensor or a dict with a tensor under "
        "'features'/'segnet_features'"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features = _load_feature_tensor(args.segnet_features)
    if features.ndim != 2:
        raise ValueError(
            f"--segnet-features must be a (N_pairs, D) tensor, got {tuple(features.shape)}"
        )
    if features.shape[0] != args.n_pairs:
        raise ValueError(
            f"--n-pairs={args.n_pairs} does not match feature rows {features.shape[0]}"
        )

    scorer = CuratorOutlierScorer(
        n_pca_components=args.n_pca_components,
        n_clusters=args.n_clusters,
        soft_dtw_gamma=args.soft_dtw_gamma,
        outlier_quantile=args.outlier_quantile,
    )
    scorer.fit(features)
    pair_weights = scorer.derive_pair_weights(
        n_pairs=args.n_pairs,
        weight_scale=args.weight_scale,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "pair_weights.pt"
    scorer_path = args.output_dir / "curator_outlier_scorer.pt"
    torch.save(pair_weights, weights_path)
    scorer.save(scorer_path)

    assignments = scorer.cluster_assignments_
    flags = scorer.outlier_flags_
    thresholds = scorer.cluster_thresholds_
    print("Lane WC Curator outlier fit complete")
    print(f"features: {tuple(features.shape)}")
    print(f"pair_weights: {weights_path}")
    print(f"scorer: {scorer_path}")
    print("per-cluster Q95 thresholds and outlier counts:")
    for cluster_idx in range(thresholds.numel()):
        mask = assignments == cluster_idx
        count = int(mask.sum().item())
        outlier_count = int(flags[mask].sum().item())
        threshold = thresholds[cluster_idx].item()
        print(
            f"  cluster={cluster_idx} size={count} "
            f"q95={threshold:.6f} outliers={outlier_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
