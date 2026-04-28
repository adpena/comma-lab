"""Cosmos Curator-style soft-DTW outlier scoring for pair weighting.

Lane WC uses only SegNet feature geometry. It intentionally does not accept
or inspect renderer loss, scorer loss, PoseNet loss, or any training telemetry.
"""

from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["CuratorOutlierScorer", "soft_dtw_distance"]

_KMEANS_SEED = 0
_KMEANS_MAX_ITERS = 100


def _as_trajectory(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(-1)
    if x.ndim == 2:
        return x
    raise ValueError(f"soft-DTW expects a 1D or 2D trajectory, got shape {tuple(x.shape)}")


def soft_dtw_distance(x: torch.Tensor, y: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Differentiable soft-DTW distance between two trajectories.

    ``x`` and ``y`` may be shaped ``(T,)`` for scalar trajectories or
    ``(T, F)`` for feature trajectories. The recurrence is the standard DTW
    dynamic program with hard min replaced by ``-gamma * logsumexp(-v/gamma)``.
    """
    if gamma <= 0:
        raise ValueError("soft-DTW gamma must be > 0")
    x = _as_trajectory(x)
    y = _as_trajectory(y).to(device=x.device, dtype=x.dtype)
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(
            f"trajectory feature dims differ: {x.shape[-1]} vs {y.shape[-1]}"
        )

    cost = (x[:, None, :] - y[None, :, :]).pow(2).sum(dim=-1)
    n, m = cost.shape
    inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    prev = [inf for _ in range(m + 1)]
    prev[0] = torch.zeros((), device=x.device, dtype=x.dtype)
    for i in range(1, n + 1):
        curr = [inf]
        for j in range(1, m + 1):
            vals = torch.stack((prev[j], curr[j - 1], prev[j - 1]))
            soft_min = -float(gamma) * torch.logsumexp(-vals / float(gamma), dim=0)
            curr.append(cost[i - 1, j - 1] + soft_min)
        prev = curr
    return prev[m]


def _soft_dtw_pairwise(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Vectorized soft-DTW for ``x=(N,T,F)`` and ``y=(K,U,F)``."""
    if gamma <= 0:
        raise ValueError("soft-DTW gamma must be > 0")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("pairwise soft-DTW expects x=(N,T,F), y=(K,U,F)")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(f"feature dims differ: {x.shape[-1]} vs {y.shape[-1]}")

    y = y.to(device=x.device, dtype=x.dtype)
    cost = (x[:, None, :, None, :] - y[None, :, None, :, :]).pow(2).sum(dim=-1)
    n_items, n_centers, n, m = cost.shape
    shape = (n_items, n_centers)
    inf = torch.full(shape, float("inf"), device=x.device, dtype=x.dtype)
    prev = [inf for _ in range(m + 1)]
    prev[0] = torch.zeros(shape, device=x.device, dtype=x.dtype)
    for i in range(1, n + 1):
        curr = [inf]
        for j in range(1, m + 1):
            vals = torch.stack((prev[j], curr[j - 1], prev[j - 1]), dim=0)
            soft_min = -float(gamma) * torch.logsumexp(-vals / float(gamma), dim=0)
            curr.append(cost[:, :, i - 1, j - 1] + soft_min)
        prev = curr
    return prev[m]


class CuratorOutlierScorer:
    def __init__(
        self,
        n_pca_components: int = 3,
        n_clusters: int = 5,
        soft_dtw_gamma: float = 0.1,
        outlier_quantile: float = 0.95,
    ):
        if n_pca_components <= 0:
            raise ValueError("n_pca_components must be > 0")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        if soft_dtw_gamma <= 0:
            raise ValueError("soft_dtw_gamma must be > 0")
        if not (0.0 < outlier_quantile < 1.0):
            raise ValueError("outlier_quantile must be in (0, 1)")

        self.n_pca_components = int(n_pca_components)
        self.n_clusters = int(n_clusters)
        self.soft_dtw_gamma = float(soft_dtw_gamma)
        self.outlier_quantile = float(outlier_quantile)

    def fit(self, segnet_features: torch.Tensor) -> None:
        """Fit PCA, time-series KMeans, and soft-DTW outlier scores.

        ``segnet_features`` must be ``(N_pairs, D)`` SegNet penultimate-layer
        features. No loss values are accepted by this method.
        """
        if not isinstance(segnet_features, torch.Tensor):
            raise TypeError("segnet_features must be a torch.Tensor")
        if segnet_features.ndim != 2:
            raise ValueError(
                f"segnet_features must be (N_pairs, D), got {tuple(segnet_features.shape)}"
            )
        if segnet_features.shape[0] == 0:
            raise ValueError("segnet_features must contain at least one pair")
        if not torch.isfinite(segnet_features).all():
            raise ValueError("segnet_features contains NaN/Inf")

        features = segnet_features.detach().to(device="cpu", dtype=torch.float32)
        projection, mean, components = self._pca_project(features)
        trajectories = projection.unsqueeze(-1)
        centroids, assignments = self._ts_kmeans(trajectories)
        all_distances = _soft_dtw_pairwise(
            trajectories, centroids, gamma=self.soft_dtw_gamma
        )
        pair_distances = all_distances[
            torch.arange(features.shape[0]), assignments
        ].contiguous()
        global_barycenter = trajectories.mean(dim=0, keepdim=True)
        cluster_isolation = _soft_dtw_pairwise(
            centroids, global_barycenter, gamma=self.soft_dtw_gamma
        ).squeeze(1)
        # A compact shifted minority cluster is still atypical. The score
        # therefore combines within-cluster soft-DTW with the assigned
        # barycenter's soft-DTW isolation from the global feature barycenter.
        outlier_distances = (
            pair_distances + cluster_isolation[assignments]
        ).contiguous()

        thresholds, flags = self._cluster_thresholds(
            outlier_distances, assignments, centroids.shape[0]
        )
        scores = self._normalize_scores(outlier_distances)

        self.pca_mean_ = mean
        self.pca_components_ = components
        self.pca_projection_ = projection
        self.cluster_barycenters_ = centroids.squeeze(-1).contiguous()
        self.cluster_assignments_ = assignments.contiguous()
        self.cluster_thresholds_ = thresholds
        self.cluster_isolation_distances_ = cluster_isolation.contiguous()
        self.pair_soft_dtw_distances_ = pair_distances
        self.pair_outlier_distances_ = outlier_distances
        self.outlier_flags_ = flags
        self.typicality_scores_ = scores

    def per_pair_typicality_score(self) -> torch.Tensor:
        """Return ``(N_pairs,)`` scores in ``[0, 1]``; higher means outlier."""
        self._require_fit()
        return self.typicality_scores_.clone()

    def derive_pair_weights(self, n_pairs: int, weight_scale: float = 5.0) -> torch.Tensor:
        """Map typicality scores to ``[1, weight_scale]`` training weights."""
        self._require_fit()
        if n_pairs != self.typicality_scores_.numel():
            raise ValueError(
                f"n_pairs={n_pairs} does not match fitted scorer length "
                f"{self.typicality_scores_.numel()}"
            )
        if weight_scale < 1.0:
            raise ValueError("weight_scale must be >= 1.0")
        scores = self.typicality_scores_.to(torch.float32)
        return 1.0 + scores * (float(weight_scale) - 1.0)

    def save(self, path: Path) -> None:
        self._require_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n_pca_components": self.n_pca_components,
                "n_clusters": self.n_clusters,
                "soft_dtw_gamma": self.soft_dtw_gamma,
                "outlier_quantile": self.outlier_quantile,
                "pca_mean": self.pca_mean_,
                "pca_components": self.pca_components_,
                "pca_projection": self.pca_projection_,
                "cluster_barycenters": self.cluster_barycenters_,
                "cluster_assignments": self.cluster_assignments_,
                "cluster_thresholds": self.cluster_thresholds_,
                "cluster_isolation_distances": self.cluster_isolation_distances_,
                "pair_soft_dtw_distances": self.pair_soft_dtw_distances_,
                "pair_outlier_distances": self.pair_outlier_distances_,
                "outlier_flags": self.outlier_flags_,
                "typicality_scores": self.typicality_scores_,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "CuratorOutlierScorer":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        scorer = cls(
            n_pca_components=state["n_pca_components"],
            n_clusters=state["n_clusters"],
            soft_dtw_gamma=state["soft_dtw_gamma"],
            outlier_quantile=state["outlier_quantile"],
        )
        scorer.pca_mean_ = state["pca_mean"]
        scorer.pca_components_ = state["pca_components"]
        scorer.pca_projection_ = state["pca_projection"]
        scorer.cluster_barycenters_ = state["cluster_barycenters"]
        scorer.cluster_assignments_ = state["cluster_assignments"]
        scorer.cluster_thresholds_ = state["cluster_thresholds"]
        scorer.cluster_isolation_distances_ = state.get(
            "cluster_isolation_distances",
            torch.zeros_like(scorer.cluster_thresholds_),
        )
        scorer.pair_soft_dtw_distances_ = state["pair_soft_dtw_distances"]
        scorer.pair_outlier_distances_ = state.get(
            "pair_outlier_distances",
            scorer.pair_soft_dtw_distances_,
        )
        scorer.outlier_flags_ = state["outlier_flags"]
        scorer.typicality_scores_ = state["typicality_scores"]
        return scorer

    def _pca_project(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = features.to(torch.float32)
        mean = x.mean(dim=0)
        centered = x - mean
        torch.manual_seed(_KMEANS_SEED)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        n_available = min(self.n_pca_components, vh.shape[0])
        components = vh[:n_available].contiguous()
        projection = centered @ components.T
        if n_available < self.n_pca_components:
            projection = torch.cat(
                [
                    projection,
                    projection.new_zeros(
                        projection.shape[0], self.n_pca_components - n_available
                    ),
                ],
                dim=1,
            )
            components = torch.cat(
                [
                    components,
                    components.new_zeros(
                        self.n_pca_components - n_available, features.shape[1]
                    ),
                ],
                dim=0,
            )
        return projection.contiguous(), mean.contiguous(), components.contiguous()

    def _ts_kmeans(self, trajectories: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_items = trajectories.shape[0]
        n_clusters = min(self.n_clusters, n_items)
        torch.manual_seed(_KMEANS_SEED)
        initial_idx = torch.randperm(n_items, device=trajectories.device)[:n_clusters]
        centroids = trajectories[initial_idx].clone()
        assignments = torch.full(
            (n_items,), -1, dtype=torch.long, device=trajectories.device
        )

        for _ in range(_KMEANS_MAX_ITERS):
            distances = _soft_dtw_pairwise(
                trajectories, centroids, gamma=self.soft_dtw_gamma
            )
            new_assignments = torch.argmin(distances, dim=1)
            new_centroids = centroids.clone()
            nearest = distances.min(dim=1).values
            for k in range(n_clusters):
                mask = new_assignments == k
                if mask.any():
                    new_centroids[k] = trajectories[mask].mean(dim=0)
                else:
                    new_centroids[k] = trajectories[torch.argmax(nearest)]
            if torch.equal(new_assignments, assignments) and torch.allclose(
                new_centroids, centroids, atol=1e-6, rtol=1e-6
            ):
                break
            assignments = new_assignments
            centroids = new_centroids

        final_distances = _soft_dtw_pairwise(
            trajectories, centroids, gamma=self.soft_dtw_gamma
        )
        return centroids, torch.argmin(final_distances, dim=1)

    def _cluster_thresholds(
        self,
        pair_distances: torch.Tensor,
        assignments: torch.Tensor,
        n_clusters: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        thresholds = torch.empty(n_clusters, dtype=pair_distances.dtype)
        flags = torch.zeros(pair_distances.shape[0], dtype=torch.bool)
        for k in range(n_clusters):
            mask = assignments == k
            if not mask.any():
                thresholds[k] = torch.nan
                continue
            vals = pair_distances[mask]
            threshold = torch.quantile(vals, self.outlier_quantile)
            thresholds[k] = threshold
            flags[mask] = vals >= threshold
        return thresholds, flags

    @staticmethod
    def _normalize_scores(pair_distances: torch.Tensor) -> torch.Tensor:
        d_min = pair_distances.min()
        d_max = pair_distances.max()
        denom = d_max - d_min
        if denom.abs().item() <= 1e-12:
            return torch.zeros_like(pair_distances, dtype=torch.float32)
        return ((pair_distances - d_min) / denom).clamp(0.0, 1.0).to(torch.float32)

    def _require_fit(self) -> None:
        if not hasattr(self, "typicality_scores_"):
            raise RuntimeError("CuratorOutlierScorer has not been fit")
