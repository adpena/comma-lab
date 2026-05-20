# SPDX-License-Identifier: MIT
"""Gaussian process surrogate over (substrate_features, predicted_delta) anchors.

Per SLOT MG-5 of the 2026-05-19 master-gradient enhancement wave.

This module wraps :class:`sklearn.gaussian_process.GaussianProcessRegressor`
as the non-linear alternative to the Kalman filter for the
streaming-prediction surface. The Kalman filter is the canonical
1-D online state-space estimator; the GP surrogate is the canonical
multi-dimensional regression surface over historical (features, score)
anchors.

When to use Kalman vs GP:

- **Kalman**: live online posterior over predicted_score for a SINGLE
  in-flight training run. Updates per N-epoch sample. Trivially fast
  per update (single matrix operation in 1-D). The dashboard's live
  posterior comes from here.

- **GP**: historical-anchor regression over the full empirical posterior
  (``.omx/state/continual_learning_posterior.jsonl`` + streaming-sample
  ledger). Fits a surrogate ``y = f(substrate_features) + ε`` once per
  batch of new anchors; predicts predicted_delta for a NEW substrate
  scaffold the operator hasn't dispatched yet. NON-PROMOTABLE per
  Catalog #323 (the GP is a model; its predictions are
  ``evidence_grade=PREDICTED``).

[verified-against:Rasmussen+Williams 2006 "Gaussian Processes for Machine Learning"]
[verified-against:sklearn.gaussian_process.GaussianProcessRegressor canonical API]

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": this is
the canonical sklearn-delegating wrapper. Direct ``sklearn.gaussian_process``
construction outside this module is permitted but discouraged for the
streaming-prediction surface so the audit tool can detect non-canonical
GP usages at the source-text scan.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# sklearn is already a project dependency (used by autopilot Rashomon
# ensemble + Rudin-Daubechies SLIM ranker per Catalog #250).
try:  # pragma: no cover — exercised by the canonical helper path
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    GaussianProcessRegressor = None  # type: ignore[misc,assignment]
    RBF = None  # type: ignore[misc,assignment]
    ConstantKernel = None  # type: ignore[misc,assignment]
    WhiteKernel = None  # type: ignore[misc,assignment]
    _SKLEARN_AVAILABLE = False


# Default kernel composition: ConstantKernel * RBF + WhiteKernel.
# Per Rasmussen & Williams 2006 §2.7 + §4.2 — this is the canonical
# noise-aware RBF kernel for one-shot regression. The WhiteKernel
# component absorbs the observation noise so the predicted variance
# is the SUM of model uncertainty + observation noise, which the
# dashboard surfaces as the confidence band.
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_CONSTANT_VALUE = 1.0
DEFAULT_NOISE_LEVEL = 1e-4
DEFAULT_N_RESTARTS_OPTIMIZER = 5


@dataclass
class GaussianProcessSurrogate:
    """Wrapper around sklearn.gaussian_process.GaussianProcessRegressor.

    Maintains the fitted model + the feature-vector schema + the
    refit-from-anchors helper. The class is NOT frozen because the
    sklearn model is stateful (fitted once, predicts many times).

    Attributes:
        feature_names: ordered tuple of substrate-feature names
            (e.g., ``("predicted_band_lower", "predicted_band_upper",
            "horizon_class_int", "lane_class_int")``).
        n_restarts_optimizer: passed to GP constructor.
        length_scale_init: initial RBF length-scale.
        constant_value_init: initial Constant kernel multiplier.
        noise_level_init: initial WhiteKernel noise level.
        model: the underlying GaussianProcessRegressor (None until fit).
        n_training_samples: number of (X, y) pairs in the most recent fit.
    """

    feature_names: tuple[str, ...]
    n_restarts_optimizer: int = DEFAULT_N_RESTARTS_OPTIMIZER
    length_scale_init: float = DEFAULT_LENGTH_SCALE
    constant_value_init: float = DEFAULT_CONSTANT_VALUE
    noise_level_init: float = DEFAULT_NOISE_LEVEL
    model: Any = None
    n_training_samples: int = 0

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "GaussianProcessSurrogate requires sklearn; install scikit-learn>=1.3"
            )

    def _build_kernel(self) -> Any:
        """Construct the canonical kernel composition."""
        return (
            ConstantKernel(self.constant_value_init, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=self.length_scale_init, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=self.noise_level_init, noise_level_bounds=(1e-8, 1e0))
        )

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "GaussianProcessSurrogate":
        """Fit the GP on (X, y).

        Args:
            X: shape (n_samples, n_features) — substrate feature vectors.
            y: shape (n_samples,) — observed predicted_delta values.

        Returns:
            self (for chaining).

        Raises:
            ValueError: if X / y shapes are inconsistent or empty.
        """
        if len(X) == 0:
            raise ValueError("X must be non-empty")
        if len(X) != len(y):
            raise ValueError(f"len(X)={len(X)} != len(y)={len(y)}")
        for i, row in enumerate(X):
            if len(row) != len(self.feature_names):
                raise ValueError(
                    f"X[{i}] has {len(row)} features but schema expects "
                    f"{len(self.feature_names)} ({self.feature_names})"
                )

        kernel = self._build_kernel()
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn convergence warnings on tiny samples
            gp.fit(list(X), list(y))
        self.model = gp
        self.n_training_samples = len(X)
        return self

    def predict(
        self, X: Sequence[Sequence[float]], *, return_std: bool = True
    ) -> tuple[list[float], list[float] | None]:
        """Predict mean (and optionally std) for new feature vectors.

        Args:
            X: shape (n_samples, n_features).
            return_std: if True, return (mean, std); else (mean, None).

        Returns:
            (mean_list, std_list_or_None).

        Raises:
            RuntimeError: if model has not been fit.
        """
        if self.model is None:
            raise RuntimeError("GaussianProcessSurrogate.predict called before fit()")
        if len(X) == 0:
            return ([], [] if return_std else None)
        for i, row in enumerate(X):
            if len(row) != len(self.feature_names):
                raise ValueError(
                    f"X[{i}] has {len(row)} features but schema expects "
                    f"{len(self.feature_names)}"
                )

        if return_std:
            mean, std = self.model.predict(list(X), return_std=True)
            return (list(map(float, mean)), list(map(float, std)))
        mean = self.model.predict(list(X), return_std=False)
        return (list(map(float, mean)), None)


def fit_surrogate_from_samples(
    samples: Iterable[dict[str, Any]],
    *,
    feature_extractor: Any | None = None,
    feature_names: tuple[str, ...] | None = None,
    target_key: str = "predicted_score",
) -> GaussianProcessSurrogate:
    """Fit a GP surrogate from a list of streaming-prediction sample rows.

    The DEFAULT ``feature_extractor`` extracts a 3-D feature vector::

        (epoch, posterior_mean, m_sample_size)

    The CALLER can provide a custom callable that maps a sample dict to a
    feature vector + the corresponding ``feature_names`` tuple. This is
    the canonical extension point for sister consumers that want
    substrate-specific features (e.g., horizon_class_int + lane_class_int).

    Args:
        samples: iterable of dict rows from the streaming-prediction ledger.
        feature_extractor: callable ``dict -> list[float]`` or None for default.
        feature_names: tuple of feature names matching the extractor output.
            REQUIRED if feature_extractor is provided; else defaults to
            ``("epoch", "posterior_mean", "m_sample_size")``.
        target_key: key in each sample dict that holds the regression target.

    Returns:
        Fitted GaussianProcessSurrogate.

    Raises:
        ValueError: if samples is empty after filtering for valid targets.
    """
    if feature_extractor is None:
        def _default_extractor(row: dict[str, Any]) -> list[float]:
            return [
                float(row.get("epoch", 0)),
                float(row.get("posterior_mean", row.get("predicted_score", 0.0))),
                float(row.get("m_sample_size", 0)),
            ]
        feature_extractor = _default_extractor
        feature_names = feature_names or ("epoch", "posterior_mean", "m_sample_size")
    if feature_names is None:
        raise ValueError("feature_names must be provided when feature_extractor is custom")

    X: list[list[float]] = []
    y: list[float] = []
    for row in samples:
        if not isinstance(row, dict):
            continue
        target = row.get(target_key)
        if target is None or not isinstance(target, (int, float)):
            continue
        if not math.isfinite(float(target)):
            continue
        try:
            features = feature_extractor(row)
        except Exception:  # noqa: BLE001 — robust to malformed rows
            continue
        if len(features) != len(feature_names):
            continue
        X.append([float(f) for f in features])
        y.append(float(target))

    if not X:
        raise ValueError("no valid samples to fit GP surrogate")

    surrogate = GaussianProcessSurrogate(feature_names=feature_names)
    surrogate.fit(X, y)
    return surrogate
