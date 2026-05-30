# SPDX-License-Identifier: MIT
"""Canonical LCLSMR linear steganalysis detector (Yousfi autostego 2026).

Origin: ``github.com/YassineYousfi/autostego/blob/eve/steganalysis/lclsmr.py``
+ sister ``_lclsmr.py`` (LSMR solver). NEW canonical detector NOT YET in our
prior portfolio (Slot FF UNIWARD / Slot YY HILL / Slot AAA MiPOD / Slot CCC
HUGO are COST-FUNCTIONS; SRNet / SRM are HEAVY-DEEP / 34K-FEATURE-CLASSICAL
detectors; LCLSMR is the LIGHTWEIGHT linear-classifier canonical detector).

The CANONICAL insight (Yousfi 2026 autostego):
LCLSMR = **Linear Classifier with LSMR solver** (Least Squares Minimum
Residual, Fong-Saunders 2011). Fits a linear classifier ``y = W @ x + b``
on extracted feature vectors using the LSMR algorithm which is optimal for
ill-conditioned least-squares problems WITHOUT requiring explicit normal
equations ``A^T A`` (which can lose half the digits of precision).

Why LSMR not SGD or AdamW: when training a linear classifier on a fixed
feature extractor (SRM features, SRNet penultimate layer, our SegNet+PoseNet
forward-pass features), the optimization is convex and LSMR converges in
``O(sqrt(condition_number) * iter)`` while SGD takes ``O(condition_number *
iter)`` and may not converge at all on near-singular systems.

Why this matters for the contest:
The CANONICAL adaptation is replacing Eve's "predict cover-vs-stego" with
"predict per-pair pose-axis vulnerability score" -- a LINEAR scoring head on
top of SegNet+PoseNet penultimate features. The LSMR solver is the canonical
optimization for this regression problem; AdamW would NOT converge cleanly
on a near-singular feature matrix.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- JPEG steganalysis -> contest YUV6 lossy compression
* **Axis B (problem space)** -- binary cover-vs-stego -> regression on
  per-pair pose-axis vulnerability score
* **Axis C (math)** -- LSMR solver 1:1 with Fong-Saunders 2011; Krylov
  subspace iteration that bounds residual norm without computing normal eqs
* **Axis D (data)** -- BOSSbase 256x256 -> ``upstream/videos/0.mkv`` per-pair
* **Axis E (video)** -- single-image -> per-pair shared latent

Sister landings
---------------
* ``scipy.sparse.linalg.lsmr`` -- canonical SciPy implementation of LSMR
  that this canonical helper routes through (DOCUMENTED ADAPTATION: we
  leverage SciPy's tested implementation; Yousfi vendors his own ``_lclsmr.py``).
* ``tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_fusion_detector_ensemble``
  -- canonical helper for combining LCLSMR + SRNet + SRM detector scores
  per Yousfi's ``fusion.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

__all__ = (
    "LCLSMRSolverStrategy",
    "LCLSMRConfig",
    "LCLSMRDetectorError",
    "fit_lclsmr_linear_classifier",
    "score_lclsmr_linear_classifier",
)


class LCLSMRDetectorError(ValueError):
    """Raised when LCLSMR fit/score violates a canonical invariant."""


class LCLSMRSolverStrategy(str, Enum):
    """Canonical LSMR solver strategies.

    * ``LSMR_FONG_SAUNDERS_2011`` -- Yousfi's canonical 1:1 (Fong-Saunders
      2011 paper "LSMR: an iterative algorithm for sparse least-squares
      problems"); converges without computing normal equations.
    * ``LSQR_PAIGE_SAUNDERS_1982`` -- sister algorithm (Paige-Saunders 1982
      LSQR); 1:1 numerically equivalent in exact arithmetic but LSMR has
      better residual-norm monotonicity in finite precision.
    * ``DIRECT_NORMAL_EQUATIONS`` -- ablation baseline (solve ``A^T A x =
      A^T b`` directly); included for Catalog #303 cargo-cult audit; NOT
      recommended for ill-conditioned feature matrices.
    * ``QR_DECOMPOSITION`` -- sister direct solver (numpy.linalg.lstsq);
      uses QR decomposition; numerically stable but O(min(m,n)^3) cost
      vs LSMR's O(iter * nnz).
    """

    LSMR_FONG_SAUNDERS_2011 = "lsmr_fong_saunders_2011"
    LSQR_PAIGE_SAUNDERS_1982 = "lsqr_paige_saunders_1982"
    DIRECT_NORMAL_EQUATIONS = "direct_normal_equations"
    QR_DECOMPOSITION = "qr_decomposition"


@dataclass(frozen=True)
class LCLSMRConfig:
    """Canonical LCLSMR config.

    Attributes
    ----------
    solver_strategy
        Which LSMR variant to use.
    damping
        Tikhonov regularization parameter (canonical Yousfi default = 0.0;
        recommend 1e-4 to 1e-2 for ill-conditioned feature matrices).
    atol
        Absolute tolerance for LSMR convergence (canonical = 1e-6).
    btol
        Residual-norm tolerance (canonical = 1e-6).
    max_iter
        Hard cap on LSMR iterations (canonical = 1000).
    """

    solver_strategy: LCLSMRSolverStrategy = LCLSMRSolverStrategy.LSMR_FONG_SAUNDERS_2011
    damping: float = 0.0
    atol: float = 1e-6
    btol: float = 1e-6
    max_iter: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.solver_strategy, LCLSMRSolverStrategy):
            raise LCLSMRDetectorError(
                f"solver_strategy={self.solver_strategy!r} must be LCLSMRSolverStrategy"
            )
        if self.damping < 0:
            raise LCLSMRDetectorError(f"damping={self.damping} must be >= 0")
        if self.atol <= 0:
            raise LCLSMRDetectorError(f"atol={self.atol} must be > 0")
        if self.btol <= 0:
            raise LCLSMRDetectorError(f"btol={self.btol} must be > 0")
        if self.max_iter < 1:
            raise LCLSMRDetectorError(f"max_iter={self.max_iter} must be >= 1")


def fit_lclsmr_linear_classifier(
    feature_matrix: np.ndarray,
    label_vector: np.ndarray,
    config: Optional[LCLSMRConfig] = None,
) -> np.ndarray:
    """Fit a linear classifier ``y = W @ x + b`` via LSMR / sister solver.

    Parameters
    ----------
    feature_matrix
        Shape ``(n_samples, n_features)``; canonical SRM features or
        SRNet penultimate-layer activations or our SegNet+PoseNet forward
        features.
    label_vector
        Shape ``(n_samples,)``; canonical {-1, +1} or {0, 1} labels.
    config
        :class:`LCLSMRConfig`; defaults to ``LSMR_FONG_SAUNDERS_2011``.

    Returns
    -------
    np.ndarray
        Shape ``(n_features + 1,)``; the fitted weight vector with bias
        appended as the last element. The bias is fitted by augmenting
        feature_matrix with a column of ones.

    Raises
    ------
    LCLSMRDetectorError
        Invalid input shapes or solver failure.
    """
    if config is None:
        config = LCLSMRConfig()
    if not isinstance(feature_matrix, np.ndarray):
        raise LCLSMRDetectorError(
            f"feature_matrix must be np.ndarray; got {type(feature_matrix).__name__}"
        )
    if not isinstance(label_vector, np.ndarray):
        raise LCLSMRDetectorError(
            f"label_vector must be np.ndarray; got {type(label_vector).__name__}"
        )
    if feature_matrix.ndim != 2:
        raise LCLSMRDetectorError(
            f"feature_matrix must be 2-D; got ndim={feature_matrix.ndim}"
        )
    if label_vector.ndim != 1:
        raise LCLSMRDetectorError(
            f"label_vector must be 1-D; got ndim={label_vector.ndim}"
        )
    if feature_matrix.shape[0] != label_vector.shape[0]:
        raise LCLSMRDetectorError(
            f"feature_matrix rows {feature_matrix.shape[0]} != "
            f"label_vector length {label_vector.shape[0]}"
        )
    if feature_matrix.size == 0:
        raise LCLSMRDetectorError("feature_matrix empty")

    # Augment with bias column (sister of sklearn fit_intercept=True).
    n_samples = feature_matrix.shape[0]
    augmented = np.concatenate(
        [feature_matrix, np.ones((n_samples, 1), dtype=feature_matrix.dtype)],
        axis=1,
    )

    if config.solver_strategy == LCLSMRSolverStrategy.LSMR_FONG_SAUNDERS_2011:
        try:
            from scipy.sparse.linalg import lsmr
        except ImportError as exc:
            raise LCLSMRDetectorError(
                "scipy required for LSMR solver; install scipy>=1.0"
            ) from exc
        result = lsmr(
            augmented,
            label_vector.astype(np.float64),
            damp=config.damping,
            atol=config.atol,
            btol=config.btol,
            maxiter=config.max_iter,
        )
        weights = result[0]
        return weights
    if config.solver_strategy == LCLSMRSolverStrategy.LSQR_PAIGE_SAUNDERS_1982:
        try:
            from scipy.sparse.linalg import lsqr
        except ImportError as exc:
            raise LCLSMRDetectorError(
                "scipy required for LSQR solver; install scipy>=1.0"
            ) from exc
        result = lsqr(
            augmented,
            label_vector.astype(np.float64),
            damp=config.damping,
            atol=config.atol,
            btol=config.btol,
            iter_lim=config.max_iter,
        )
        return result[0]
    if config.solver_strategy == LCLSMRSolverStrategy.DIRECT_NORMAL_EQUATIONS:
        # Solve A^T A x = A^T b directly; for cargo-cult-audit comparison.
        ata = augmented.T @ augmented
        atb = augmented.T @ label_vector.astype(np.float64)
        if config.damping > 0:
            ata = ata + config.damping * np.eye(ata.shape[0])
        return np.linalg.solve(ata, atb)
    if config.solver_strategy == LCLSMRSolverStrategy.QR_DECOMPOSITION:
        # Use numpy.linalg.lstsq (QR-based).
        weights, _, _, _ = np.linalg.lstsq(
            augmented, label_vector.astype(np.float64), rcond=None
        )
        return weights
    raise LCLSMRDetectorError(f"unhandled solver_strategy {config.solver_strategy!r}")


def score_lclsmr_linear_classifier(
    weight_vector: np.ndarray,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """Score new samples via fitted weight vector.

    Parameters
    ----------
    weight_vector
        Shape ``(n_features + 1,)``; fitted weights from
        :func:`fit_lclsmr_linear_classifier`.
    feature_matrix
        Shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples,)``; predicted scores ``y_pred = W @ x + b``.

    Raises
    ------
    LCLSMRDetectorError
        Invalid input shapes.
    """
    if weight_vector.ndim != 1:
        raise LCLSMRDetectorError(
            f"weight_vector must be 1-D; got ndim={weight_vector.ndim}"
        )
    if feature_matrix.ndim != 2:
        raise LCLSMRDetectorError(
            f"feature_matrix must be 2-D; got ndim={feature_matrix.ndim}"
        )
    if weight_vector.shape[0] != feature_matrix.shape[1] + 1:
        raise LCLSMRDetectorError(
            f"weight_vector length {weight_vector.shape[0]} != "
            f"feature_matrix.shape[1] + 1 = {feature_matrix.shape[1] + 1}"
        )
    n_samples = feature_matrix.shape[0]
    augmented = np.concatenate(
        [feature_matrix, np.ones((n_samples, 1), dtype=feature_matrix.dtype)],
        axis=1,
    )
    return augmented @ weight_vector
