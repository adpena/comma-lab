# SPDX-License-Identifier: MIT
"""Canonical multi-scheme Dirichlet prior (Yousfi 2019 ALASKA Pattern #3).

Origin: ``external/alaska_yousfi/src/tools/tf_utils.py:55-95`` ``gen_train``
``np.random.choice(len(priors), 1, p=priors)`` + the canonical
``stego_schemes=['EBS','JUNI','NSF5','UED']`` /
``priors=[0.15, 0.4, 0.15, 0.3]`` tuple in
``src/notebooks/tf_fine_tune_branch.ipynb`` cell 3.

The CANONICAL insight (Yousfi 2019 ALASKA-#1-winning):
Real-world steganalysis must defend against MULTIPLE embedding schemes
simultaneously. Yousfi formulated this as a 5-class softmax (cover + 4 stego
schemes) trained with a NON-UNIFORM prior weighting the schemes by their
expected operational frequency:

* JUNI=0.4 (J-UNIWARD; most common adaptive scheme)
* UED=0.3 (Uniform Embedding Distortion; second most common)
* EBS=0.15 (Entropy-Based Steganography; less common)
* NSF5=0.15 (Non-Shrinkage F5; least common)

The non-uniform prior empirically beats uniform by ~1.5pp on ALASKA-1
validation because it weights training samples in proportion to
expected-attack frequency, matching the test distribution closely.

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- JPEG steganography schemes -> comma per-pair
  perturbation modes (HILL / MiPOD / HUGO / UNIWARD as the canonical 4
  per the Yousfi-Fridrich 7-axis cascade per CLAUDE.md "Mar 2026 Wave 9")
* **Axis B (problem space)** -- detector training -> generator gradient
  weighting (which perturbation mode receives more attention)
* **Axis C (math)** -- 1:1 categorical Dirichlet
* **Axis D (data)** -- BOSSBase 4-scheme -> comma 4-cost-function
* **Axis E (video)** -- single-image -> per-pair (latent->frame) cost
  weighting

Sister of slot ``tac.composition.alaska_inverse_steganalysis_patterns.canonical_color_separation``
(both extracted from Yousfi 2019 ALASKA codebase; this one is the
**dispatch-prior** surface, color-separation is the **architecture** surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = (
    "MultiSchemeDirichletPrior",
    "MultiSchemePriorConfig",
    "sample_perturbation_scheme",
    "canonical_alaska_1_priors",
    "canonical_comma_pair_perturbation_priors",
    "MultiSchemePriorError",
)


class MultiSchemePriorError(ValueError):
    """Raised when prior tuple violates an invariant."""


def canonical_alaska_1_priors() -> dict[str, float]:
    """Return the canonical Yousfi 2019 ALASKA-#1 prior verbatim.

    Source: ``external/alaska_yousfi/src/notebooks/tf_fine_tune_branch.ipynb``
    cell 3 ``priors = [0.15, 0.4, 0.15, 0.3]`` mapped to
    ``stego_schemes = ['EBS', 'JUNI', 'NSF5', 'UED']``.

    Returns
    -------
    dict[str, float]
        Canonical 4-scheme prior; sums to 1.0 exactly.
    """
    return {
        "JUNI": 0.40,  # J-UNIWARD - most common adaptive scheme
        "UED": 0.30,  # Uniform Embedding Distortion
        "EBS": 0.15,  # Entropy-Based Steganography
        "NSF5": 0.15,  # Non-Shrinkage F5
    }


def canonical_comma_pair_perturbation_priors() -> dict[str, float]:
    """Return the comma-video adaptation of Yousfi's ALASKA prior.

    Maps Yousfi's 4 JPEG steganography schemes to the canonical 4 comma
    cost-function adaptations (per CLAUDE.md "Fridrich inverse steganalysis"
    + sister Catalog slot landings Slot FF/YY/AAA/CCC):

    * UNIWARD (0.40) -- canonical wavelet-directional cost
      (sister of JUNI=0.40 upstream; both UNIWARD-family)
    * HILL (0.30) -- second-order high-pass cost
      (sister of UED=0.30 upstream; both adaptive uniform-style)
    * MiPOD (0.15) -- model-based Fisher-info cost
      (sister of EBS=0.15 upstream; both information-theoretic)
    * HUGO (0.15) -- SPAM-feature Markov-chain cost
      (sister of NSF5=0.15 upstream; both Markov-feature-based)

    Returns
    -------
    dict[str, float]
        Comma 4-scheme prior; sums to 1.0 exactly. The dispatch weights
        reflect the canonical "UNIWARD first / HILL second / MiPOD+HUGO
        diagnostic" attack-ordering hypothesis per the Yousfi-Fridrich
        7-axis cascade.
    """
    return {
        "UNIWARD": 0.40,
        "HILL": 0.30,
        "MIPOD": 0.15,
        "HUGO": 0.15,
    }


@dataclass(frozen=True)
class MultiSchemePriorConfig:
    """Canonical multi-scheme prior config.

    Attributes
    ----------
    scheme_priors
        Mapping scheme_name -> probability. Must sum to 1.0 within 1e-6
        tolerance; all values must be in (0, 1].
    seed
        Optional deterministic seed.
    """

    scheme_priors: Mapping[str, float]
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.scheme_priors:
            raise MultiSchemePriorError("scheme_priors empty")
        total = float(sum(self.scheme_priors.values()))
        if abs(total - 1.0) > 1e-6:
            raise MultiSchemePriorError(
                f"scheme_priors sum to {total}; must sum to 1.0 within 1e-6"
            )
        for name, p in self.scheme_priors.items():
            if not (0.0 < p <= 1.0):
                raise MultiSchemePriorError(
                    f"scheme_priors[{name!r}]={p} must be in (0, 1]"
                )
            if not isinstance(name, str) or not name:
                raise MultiSchemePriorError(
                    f"scheme name {name!r} must be non-empty str"
                )


class MultiSchemeDirichletPrior:
    """Canonical multi-scheme Dirichlet sampler.

    Wraps :class:`MultiSchemePriorConfig` with a stable scheme-name ordering
    and ``numpy`` RNG; sample one scheme name per call per Yousfi's
    ``np.random.choice(len(priors), 1, p=priors)`` upstream pattern.
    """

    def __init__(self, config: MultiSchemePriorConfig) -> None:
        self._config = config
        self._scheme_names: tuple[str, ...] = tuple(config.scheme_priors.keys())
        self._probs: np.ndarray = np.asarray(
            [config.scheme_priors[name] for name in self._scheme_names],
            dtype=np.float64,
        )
        self._rng = np.random.default_rng(config.seed)

    @property
    def scheme_names(self) -> tuple[str, ...]:
        return self._scheme_names

    @property
    def probabilities(self) -> np.ndarray:
        return self._probs.copy()

    def sample(self) -> str:
        """Sample one scheme name per the canonical prior."""
        idx = int(self._rng.choice(len(self._scheme_names), 1, p=self._probs)[0])
        return self._scheme_names[idx]

    def sample_n(self, n: int) -> list[str]:
        """Sample ``n`` scheme names per the canonical prior."""
        if n < 0:
            raise MultiSchemePriorError(f"n={n} must be >= 0")
        if n == 0:
            return []
        idxs = self._rng.choice(len(self._scheme_names), n, p=self._probs)
        return [self._scheme_names[int(i)] for i in idxs]


def sample_perturbation_scheme(
    scheme_priors: Mapping[str, float] | None = None,
    *,
    seed: int | None = None,
) -> str:
    """One-shot canonical sampler with sensible default.

    Parameters
    ----------
    scheme_priors
        Mapping scheme_name -> probability. Defaults to
        :func:`canonical_comma_pair_perturbation_priors` when None.
    seed
        Optional deterministic seed.

    Returns
    -------
    str
        Sampled scheme name.
    """
    priors = (
        dict(scheme_priors)
        if scheme_priors is not None
        else canonical_comma_pair_perturbation_priors()
    )
    cfg = MultiSchemePriorConfig(scheme_priors=priors, seed=seed)
    return MultiSchemeDirichletPrior(cfg).sample()
