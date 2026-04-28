from __future__ import annotations

import time

import torch

from tac.curator_outlier import CuratorOutlierScorer, soft_dtw_distance


def _synthetic_outlier_features(
    *,
    n_typical: int = 580,
    n_outliers: int = 20,
    dim: int = 32,
    shift: float = 5.0,
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    typical = torch.randn(n_typical, dim)
    outliers = torch.randn(n_outliers, dim) + shift
    features = torch.cat([typical, outliers], dim=0)
    labels = torch.cat([
        torch.zeros(n_typical, dtype=torch.bool),
        torch.ones(n_outliers, dtype=torch.bool),
    ])
    return features, labels


def test_fit_with_synthetic_features():
    features, labels = _synthetic_outlier_features()

    scorer = CuratorOutlierScorer()
    scorer.fit(features)
    scores = scorer.per_pair_typicality_score()

    assert scores[labels].mean().item() > 0.7


def test_pca_preserves_outlier_structure():
    features, labels = _synthetic_outlier_features(dim=100, shift=6.0, seed=99)

    scorer = CuratorOutlierScorer()
    scorer.fit(features)

    projected = scorer.pca_projection_
    typical_center = projected[~labels].mean(dim=0)
    outlier_center = projected[labels].mean(dim=0)
    between = torch.linalg.vector_norm(outlier_center - typical_center)
    within = torch.linalg.vector_norm(projected[~labels] - typical_center, dim=1).mean()

    assert between > 3.0 * within


def test_ts_kmeans_clusters_consistent():
    features, _ = _synthetic_outlier_features(dim=24, seed=2026)

    scorer_a = CuratorOutlierScorer()
    scorer_b = CuratorOutlierScorer()
    scorer_a.fit(features)
    scorer_b.fit(features)

    assert torch.equal(scorer_a.cluster_assignments_, scorer_b.cluster_assignments_)


def test_soft_dtw_differentiable():
    torch.manual_seed(7)
    x = torch.randn(5, 1, dtype=torch.double, requires_grad=True)
    y = torch.randn(5, 1, dtype=torch.double)

    distance = soft_dtw_distance(x, y, gamma=0.2)
    distance.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_per_pair_typicality_in_unit_interval():
    features, _ = _synthetic_outlier_features(dim=16, seed=42)

    scorer = CuratorOutlierScorer()
    scorer.fit(features)
    scores = scorer.per_pair_typicality_score()

    assert scores.shape == (features.shape[0],)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_derive_pair_weights_outliers_get_higher_weight():
    features, labels = _synthetic_outlier_features(dim=16, seed=314)

    scorer = CuratorOutlierScorer()
    scorer.fit(features)
    weights = scorer.derive_pair_weights(n_pairs=features.shape[0], weight_scale=5.0)

    assert weights.shape == (features.shape[0],)
    assert weights.max().item() <= 5.0
    assert weights[labels].mean().item() > weights[~labels].mean().item()


def test_save_load_roundtrip(tmp_path):
    features, _ = _synthetic_outlier_features(dim=20, seed=5150)
    path = tmp_path / "curator_outlier_scorer.pt"

    scorer = CuratorOutlierScorer()
    scorer.fit(features)
    before = scorer.per_pair_typicality_score()
    scorer.save(path)

    loaded = CuratorOutlierScorer.load(path)
    after = loaded.per_pair_typicality_score()

    assert torch.allclose(before, after)


def test_fits_all_600_pairs_in_under_60_seconds_smoke():
    torch.manual_seed(8080)
    features = torch.randn(600, 512)

    scorer = CuratorOutlierScorer()
    started = time.time()
    scorer.fit(features)
    elapsed = time.time() - started

    assert scorer.per_pair_typicality_score().shape == (600,)
    assert elapsed < 60.0
