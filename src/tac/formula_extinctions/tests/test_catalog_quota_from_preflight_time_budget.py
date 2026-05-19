# SPDX-License-Identifier: MIT
"""Tests for Row #10 — Catalog quota from preflight time budget."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.catalog_quota_from_preflight_time_budget import (
    CatalogQuotaInput,
    DEFAULT_BUDGET_MS,
    DEFAULT_MEAN_GATE_COST_MS,
    canonical_catalog_quota_from_preflight_budget,
)


def test_default_baseline_recovers_400():
    """30s budget / 75ms per gate = 400 (Catalog #299 anchor)."""
    r = canonical_catalog_quota_from_preflight_budget()
    assert r.solved_value == 400


def test_default_constants_pinned():
    """Canonical 30s + 75ms baseline."""
    assert DEFAULT_BUDGET_MS == 30_000.0
    assert DEFAULT_MEAN_GATE_COST_MS == 75.0


def test_halving_gate_cost_doubles_quota():
    """quota = budget/cost; halving cost doubles quota."""
    r = canonical_catalog_quota_from_preflight_budget(
        CatalogQuotaInput(mean_gate_cost_ms=37.5)
    )
    assert r.solved_value == 800


def test_doubling_budget_doubles_quota():
    """60s budget at 75ms = 800."""
    r = canonical_catalog_quota_from_preflight_budget(
        CatalogQuotaInput(budget_ms=60_000.0)
    )
    assert r.solved_value == 800


def test_invalid_inputs_raise():
    """Non-positive inputs raise."""
    with pytest.raises(ValueError, match="budget_ms"):
        CatalogQuotaInput(budget_ms=0.0)
    with pytest.raises(ValueError, match="mean_gate_cost_ms"):
        CatalogQuotaInput(mean_gate_cost_ms=-1.0)


def test_citation_catalog_299():
    """Catalog #299 citation."""
    r = canonical_catalog_quota_from_preflight_budget()
    assert "Catalog #299" in r.literature_citation
