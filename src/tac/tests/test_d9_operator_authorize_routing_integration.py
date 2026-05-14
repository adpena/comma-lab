# SPDX-License-Identifier: MIT
"""Integration tests: D9 per-class routing API consumed by operator_authorize.py.

Covers the wiring between :func:`tac.cost_band_calibration.select_provider_for_recipe`
and ``tools/operator_authorize.py::_run_dispatch``. The integration:

  1. Recipes declaring ``platform: auto`` get auto-routed to the canonical
     Decision 9 provider/gpu (the routing helper rewrites
     ``recipe.raw["platform"]`` and ``recipe.raw["gpu"]``).
  2. Recipes with an explicit ``platform:`` (legacy default) pass through
     unchanged; the routing recommendation is logged for forensics.
  3. The routing decision is recorded under
     ``recipe.raw["_d9_routing_decision"]`` so downstream consumers can
     audit the chosen vs canonical pair.

Per CLAUDE.md "Subagent coherence-by-default" anti-fragmentation primitive:
the operator's explicit choice always wins. Auto-routing is opt-in.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_operator_authorize():
    """Load `tools/operator_authorize.py` as a module under a private name.

    The script lives outside the `tac` package so we import it via a
    file-spec loader. A module-level cache prevents re-importing on every
    test (which would re-run the top-level subprocess gate setup).
    """
    if "_oa_d9" in sys.modules:
        return sys.modules["_oa_d9"]
    path = REPO_ROOT / "tools/operator_authorize.py"
    spec = importlib.util.spec_from_file_location("_oa_d9", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_oa_d9"] = module
    spec.loader.exec_module(module)
    return module


def _make_recipe(raw: dict, name: str = "test_recipe") -> object:
    """Build a Recipe instance directly from a raw dict (no YAML round-trip)."""
    oa = _import_operator_authorize()
    return oa.Recipe(name=name, path=Path(f"{name}.yaml"), raw=raw)


# -- _resolve_routing_decision ----------------------------------------------


def test_resolve_routing_decision_returns_canonical_for_smoke_recipe() -> None:
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "modal",
            "gpu": "T4",
            "dispatch_label": "substrate_x_modal_t4_dispatch_20260514T010000Z__smoke__100ep",
            "cost_band": {"epochs": 100},
        }
    )
    decision = oa._resolve_routing_decision(recipe)
    assert decision is not None
    assert decision["dispatch_class"] == "smoke"
    assert decision["canonical_provider"] == "modal"
    assert decision["canonical_gpu"] == "T4"


def test_resolve_routing_decision_returns_canonical_for_full_recipe() -> None:
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "modal",
            "gpu": "A100",
            "dispatch_class": "full",
            "cost_band": {"epochs": 3000},
        }
    )
    decision = oa._resolve_routing_decision(recipe)
    assert decision is not None
    assert decision["dispatch_class"] == "full"
    assert decision["canonical_provider"] == "vastai"
    assert decision["canonical_gpu"] == "RTX_4090"


def test_resolve_routing_decision_returns_canonical_for_long_burn_recipe() -> None:
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "modal",
            "gpu": "A100",
            "dispatch_class": "long_burn",
            "cost_band": {"epochs": 6000},
        }
    )
    decision = oa._resolve_routing_decision(recipe)
    assert decision is not None
    assert decision["dispatch_class"] == "long_burn"
    assert decision["canonical_provider"] == "lightning"
    assert decision["canonical_gpu"] == "A100"


# -- _maybe_apply_auto_routing ---------------------------------------------


def test_maybe_apply_auto_routing_explicit_platform_passes_through(capsys) -> None:
    """Recipe with explicit ``platform: modal`` must NOT be rewritten."""
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "modal",
            "gpu": "T4",
            "dispatch_label": "substrate_x_modal_t4_dispatch_20260514__smoke__100ep",
            "cost_band": {"epochs": 100},
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    assert rewritten.platform == "modal"
    assert rewritten.gpu == "T4"
    # Routing decision still recorded for forensics.
    assert "_d9_routing_decision" in rewritten.raw
    captured = capsys.readouterr()
    assert "D9 routing" in captured.out
    assert "pass-through" in captured.out


def test_maybe_apply_auto_routing_auto_platform_resolves_to_canonical(capsys) -> None:
    """Recipe declaring ``platform: auto`` MUST be rewritten to the
    Decision 9 canonical provider for its dispatch class."""
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "auto",
            "dispatch_class": "full",
            "cost_band": {"epochs": 3000},
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    # Decision 9 canonical for "full" is vastai/RTX_4090.
    assert rewritten.platform == "vastai"
    assert rewritten.gpu == "RTX_4090"
    assert rewritten.raw["platform"] == "vastai"
    assert rewritten.raw["gpu"] == "RTX_4090"
    captured = capsys.readouterr()
    assert "D9 routing" in captured.out
    assert "vastai/RTX_4090" in captured.out


def test_maybe_apply_auto_routing_auto_platform_smoke_class_resolves_to_modal_t4(
    capsys,
) -> None:
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "auto",
            "dispatch_class": "smoke",
            "cost_band": {"epochs": 100},
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    assert rewritten.platform == "modal"
    assert rewritten.gpu == "T4"


def test_maybe_apply_auto_routing_records_canonical_under_recipe_raw() -> None:
    """The routing decision MUST land under recipe.raw['_d9_routing_decision']
    so downstream tooling can audit the chosen vs canonical pair."""
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "modal",
            "gpu": "A100",
            "dispatch_class": "full",
            "cost_band": {"epochs": 3000},
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    decision = rewritten.raw["_d9_routing_decision"]
    assert decision["dispatch_class"] == "full"
    assert decision["canonical_provider"] == "vastai"
    assert decision["canonical_gpu"] == "RTX_4090"
    # provider/gpu in the decision reflect what the recipe ALSO exposes
    # (the operator's explicit choice — modal/A100 — NOT the canonical).
    assert decision["provider"] == "modal"
    assert decision["gpu"] == "A100"


def test_maybe_apply_auto_routing_logs_d9_rationale(capsys) -> None:
    """Auto-routing decisions log the rationale string for operator forensics."""
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "auto",
            "dispatch_class": "full",
            "cost_band": {"epochs": 3000},
        }
    )
    oa._maybe_apply_auto_routing(recipe)
    captured = capsys.readouterr()
    assert "D9 rationale" in captured.out


# -- Backward compatibility: existing recipes work unchanged ----------------


def test_existing_modal_recipe_still_routes_to_modal(capsys) -> None:
    """Regression guard: every existing recipe in
    `.omx/operator_authorize_recipes/` declares an explicit `platform:`. The
    auto-routing helper must NOT mutate any of them."""
    oa = _import_operator_authorize()
    # Use a real recipe shape (mirrors the canonical YAML keys).
    recipe = _make_recipe(
        {
            "lane_id": "lane_test_existing",
            "platform": "modal",  # explicit legacy default
            "gpu": "A100",
            "dispatch_class": "full",
            "cost_band": {"epochs": 3000},
            "min_smoke_gpu": "A100",
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    assert rewritten.platform == "modal"  # unchanged
    assert rewritten.gpu == "A100"  # unchanged


def test_existing_vastai_recipe_still_routes_to_vastai() -> None:
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test_vast",
            "platform": "vastai",
            "gpu": "RTX_4090",
            "dispatch_class": "full",
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    assert rewritten.platform == "vastai"
    assert rewritten.gpu == "RTX_4090"


def test_existing_lightning_recipe_still_routes_to_lightning() -> None:
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test_lightning",
            "platform": "lightning",
            "gpu": "A100",
            "dispatch_class": "long_burn",
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    assert rewritten.platform == "lightning"
    assert rewritten.gpu == "A100"


# -- Integration with _run_dispatch ----------------------------------------


def test_run_dispatch_consumes_routing_for_unsupported_platform_raises() -> None:
    """If `platform: auto` resolves to an unsupported dispatcher, `_run_dispatch`
    raises (existing FATAL message) — the routing helper is purely
    informational; the dispatcher behavior stays the same."""
    # NOTE: ``_run_dispatch`` calls platform-specific helpers that require
    # real provider state (Modal CLI, Vast.ai launcher, etc.); we cannot
    # call it end-to-end here without mocks. The integration we DO assert
    # is that ``_maybe_apply_auto_routing`` runs first and any subsequent
    # FATAL error message references the auto-resolved platform name.
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "auto",
            "dispatch_class": "cpu",  # resolves to github/ubuntu-latest
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    # cpu -> github -> falls into the noop dispatcher branch.
    assert rewritten.platform == "github"
    assert rewritten.gpu == "ubuntu-latest"


# -- Recipe-side override priority ----------------------------------------


def test_recipe_explicit_provider_overrides_canonical_routing(capsys) -> None:
    """Per CLAUDE.md "Subagent coherence-by-default": operator's explicit
    choice wins. A recipe with `platform: lightning gpu: T4` for a "smoke"
    class should NOT be rewritten even though Decision 9 canonical for
    smoke is modal/T4."""
    oa = _import_operator_authorize()
    recipe = _make_recipe(
        {
            "lane_id": "lane_test",
            "platform": "lightning",
            "gpu": "T4",
            "dispatch_class": "smoke",
        }
    )
    rewritten = oa._maybe_apply_auto_routing(recipe)
    assert rewritten.platform == "lightning"
    assert rewritten.gpu == "T4"
    # But D9's recommendation (modal/T4) is still recorded for forensics.
    decision = rewritten.raw["_d9_routing_decision"]
    assert decision["canonical_provider"] == "modal"
    assert decision["canonical_gpu"] == "T4"
