# SPDX-License-Identifier: MIT
"""Tests for canonical ALASKA multi-scheme Dirichlet prior."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tac.composition.alaska_inverse_steganalysis_patterns import (
    MultiSchemeDirichletPrior,
    MultiSchemePriorConfig,
    MultiSchemePriorError,
    canonical_alaska_1_priors,
    canonical_comma_pair_perturbation_priors,
    sample_perturbation_scheme,
)


def test_canonical_alaska_1_priors_sum_to_one() -> None:
    p = canonical_alaska_1_priors()
    assert pytest.approx(sum(p.values()), abs=1e-9) == 1.0


def test_canonical_alaska_1_priors_verbatim_yousfi() -> None:
    """Verbatim Yousfi 2019 ALASKA-#1 prior."""
    p = canonical_alaska_1_priors()
    assert p == {"JUNI": 0.40, "UED": 0.30, "EBS": 0.15, "NSF5": 0.15}


def test_canonical_comma_pair_priors_sum_to_one() -> None:
    p = canonical_comma_pair_perturbation_priors()
    assert pytest.approx(sum(p.values()), abs=1e-9) == 1.0


def test_canonical_comma_pair_priors_4_canonical_costs() -> None:
    """Per CLAUDE.md Fridrich inverse steganalysis lineage: 4 canonical
    cost functions UNIWARD / HILL / MiPOD / HUGO."""
    p = canonical_comma_pair_perturbation_priors()
    assert set(p.keys()) == {"UNIWARD", "HILL", "MIPOD", "HUGO"}


def test_config_rejects_non_normalized_priors() -> None:
    with pytest.raises(MultiSchemePriorError, match="must sum to 1.0"):
        MultiSchemePriorConfig(scheme_priors={"A": 0.5, "B": 0.4})


def test_config_rejects_negative_prob() -> None:
    with pytest.raises(MultiSchemePriorError, match=r"must be in \(0, 1\]"):
        MultiSchemePriorConfig(scheme_priors={"A": 1.5, "B": -0.5})


def test_config_rejects_empty_priors() -> None:
    with pytest.raises(MultiSchemePriorError, match="empty"):
        MultiSchemePriorConfig(scheme_priors={})


def test_config_rejects_empty_name() -> None:
    with pytest.raises(MultiSchemePriorError, match="non-empty str"):
        MultiSchemePriorConfig(scheme_priors={"": 1.0})


def test_prior_sampler_seeded_deterministic() -> None:
    cfg = MultiSchemePriorConfig(
        scheme_priors=canonical_alaska_1_priors(), seed=42
    )
    sampler1 = MultiSchemeDirichletPrior(cfg)
    sampler2 = MultiSchemeDirichletPrior(cfg)
    seq1 = [sampler1.sample() for _ in range(20)]
    seq2 = [sampler2.sample() for _ in range(20)]
    assert seq1 == seq2


def test_prior_sampler_empirical_distribution_matches_prior() -> None:
    """Slot EEE substantive-distinctness: with N=10000 samples, the
    empirical frequency of each scheme MUST match the prior within 2pp.

    Verifies the sampler is REAL (not a stub returning a constant).
    """
    cfg = MultiSchemePriorConfig(
        scheme_priors=canonical_alaska_1_priors(), seed=12345
    )
    sampler = MultiSchemeDirichletPrior(cfg)
    samples = sampler.sample_n(10_000)
    counts = Counter(samples)
    expected = canonical_alaska_1_priors()
    for name, p in expected.items():
        emp = counts[name] / 10_000
        assert abs(emp - p) < 0.02, (
            f"empirical freq {name}={emp:.4f} != prior {p:.4f} (delta > 0.02)"
        )


def test_prior_sampler_substantively_distinct_per_scheme() -> None:
    """Verify the sampler distinguishes ALL 4 schemes (no degenerate
    case where one scheme dominates 100%)."""
    cfg = MultiSchemePriorConfig(
        scheme_priors=canonical_alaska_1_priors(), seed=12345
    )
    sampler = MultiSchemeDirichletPrior(cfg)
    samples = sampler.sample_n(1000)
    distinct = set(samples)
    assert distinct == {"JUNI", "UED", "EBS", "NSF5"}


def test_sample_n_rejects_negative() -> None:
    cfg = MultiSchemePriorConfig(
        scheme_priors=canonical_alaska_1_priors(), seed=42
    )
    sampler = MultiSchemeDirichletPrior(cfg)
    with pytest.raises(MultiSchemePriorError, match="must be >= 0"):
        sampler.sample_n(-1)


def test_sample_n_zero_returns_empty() -> None:
    cfg = MultiSchemePriorConfig(
        scheme_priors=canonical_alaska_1_priors(), seed=42
    )
    sampler = MultiSchemeDirichletPrior(cfg)
    assert sampler.sample_n(0) == []


def test_one_shot_sampler_default_comma_prior() -> None:
    """One-shot helper defaults to comma prior + returns a valid name."""
    name = sample_perturbation_scheme(seed=42)
    assert name in {"UNIWARD", "HILL", "MIPOD", "HUGO"}
