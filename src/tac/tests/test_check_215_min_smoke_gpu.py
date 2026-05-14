"""Catalog #215 P0 fix — recipe-level smoke GPU minimum.

The SIREN smoke timeout on 2026-05-13 (label
``substrate_siren_modal_a100_dispatch_..._smoke``, rc=124, ~3601s) showed
the smoke wrapper defaults to T4 even when the full-run GPU is A100. T4
is too slow for compute-heavy substrates like SIREN's coordinate-MLP
forward + scorer forward at 100 epochs. The fix: recipes can declare
``min_smoke_gpu: "A100"`` and the smoke wrapper honors it over a CLI
downgrade.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.run_modal_smoke_before_full import (
    _resolve_min_smoke_gpu,
    _resolve_smoke_gpu,
    _smoke_gpu_class_at_least,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# -------- _resolve_min_smoke_gpu --------


def test_resolve_min_smoke_gpu_returns_default_when_field_absent() -> None:
    recipe = "schema_version: 1\nname: x\nlane_id: lane_x\n"
    assert _resolve_min_smoke_gpu(recipe) == "T4"


def test_resolve_min_smoke_gpu_returns_default_when_field_value_invalid() -> None:
    recipe = 'min_smoke_gpu: "BANANA_GPU"\n'
    # Unknown class falls back to default rather than erroring.
    assert _resolve_min_smoke_gpu(recipe) == "T4"


def test_resolve_min_smoke_gpu_a100_quoted() -> None:
    recipe = 'min_smoke_gpu: "A100"\n'
    assert _resolve_min_smoke_gpu(recipe) == "A100"


def test_resolve_min_smoke_gpu_a100_unquoted() -> None:
    recipe = "min_smoke_gpu: A100\n"
    assert _resolve_min_smoke_gpu(recipe) == "A100"


def test_resolve_min_smoke_gpu_h100() -> None:
    recipe = 'min_smoke_gpu: "H100"\n'
    assert _resolve_min_smoke_gpu(recipe) == "H100"


def test_resolve_min_smoke_gpu_l4_l40s_a10g() -> None:
    for cls in ("L4", "L40S", "A10G"):
        recipe = f'min_smoke_gpu: "{cls}"\n'
        assert _resolve_min_smoke_gpu(recipe) == cls


def test_resolve_min_smoke_gpu_alternate_default() -> None:
    recipe = "schema_version: 1\n"
    assert _resolve_min_smoke_gpu(recipe, smoke_gpu_default="A10G") == "A10G"


# -------- _smoke_gpu_class_at_least --------


def test_smoke_gpu_class_at_least_t4_meets_t4() -> None:
    assert _smoke_gpu_class_at_least("T4", "T4") is True


def test_smoke_gpu_class_at_least_a100_meets_t4() -> None:
    assert _smoke_gpu_class_at_least("A100", "T4") is True


def test_smoke_gpu_class_at_least_t4_does_not_meet_a100() -> None:
    assert _smoke_gpu_class_at_least("T4", "A100") is False


def test_smoke_gpu_class_at_least_unknown_class_returns_false() -> None:
    assert _smoke_gpu_class_at_least("BANANA", "T4") is False
    assert _smoke_gpu_class_at_least("T4", "BANANA") is False


def test_smoke_gpu_class_at_least_h100_meets_a100() -> None:
    assert _smoke_gpu_class_at_least("H100", "A100") is True


# -------- _resolve_smoke_gpu (integration) --------


def test_resolve_smoke_gpu_no_recipe_min_honors_cli_default() -> None:
    recipe = "schema_version: 1\n"
    resolved, rationale = _resolve_smoke_gpu(recipe, "T4")
    assert resolved == "T4"
    assert rationale == "cli_smoke_gpu_no_recipe_min"


def test_resolve_smoke_gpu_recipe_min_a100_overrides_cli_t4() -> None:
    recipe = 'min_smoke_gpu: "A100"\n'
    resolved, rationale = _resolve_smoke_gpu(recipe, "T4")
    assert resolved == "A100"
    assert rationale == "recipe_min_smoke_gpu_overrides_cli_downgrade"


def test_resolve_smoke_gpu_cli_a100_meets_recipe_min_a100() -> None:
    recipe = 'min_smoke_gpu: "A100"\n'
    resolved, rationale = _resolve_smoke_gpu(recipe, "A100")
    assert resolved == "A100"
    assert rationale == "cli_smoke_gpu_meets_recipe_min"


def test_resolve_smoke_gpu_cli_h100_meets_recipe_min_a100_upgrade() -> None:
    recipe = 'min_smoke_gpu: "A100"\n'
    resolved, rationale = _resolve_smoke_gpu(recipe, "H100")
    assert resolved == "H100"
    assert rationale == "cli_smoke_gpu_meets_recipe_min"


def test_resolve_smoke_gpu_recipe_t4_no_override_when_at_default() -> None:
    recipe = 'min_smoke_gpu: "T4"\n'
    # Recipe declared T4 == smoke default; rationale should be the no-override
    # path even though the recipe declared explicitly.
    resolved, rationale = _resolve_smoke_gpu(recipe, "T4")
    assert resolved == "T4"
    assert rationale == "cli_smoke_gpu_no_recipe_min"


# -------- Empirical SIREN regression guard --------


def test_siren_recipe_has_min_smoke_gpu_a100_per_catalog_215() -> None:
    """Regression guard: SIREN recipe must declare min_smoke_gpu A100.

    Anchor: substrate_siren_modal_a100_dispatch_20260513T140410Z__smoke
    timed out (rc=124, 3601s wall-clock) because smoke ran on T4 default.
    """
    path = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml"
    )
    recipe_text = path.read_text(encoding="utf-8")
    assert _resolve_min_smoke_gpu(recipe_text) == "A100", (
        f"SIREN recipe must declare min_smoke_gpu: A100 per Catalog #215; "
        f"current resolution: {_resolve_min_smoke_gpu(recipe_text)!r}"
    )


def test_resolve_smoke_gpu_idempotent_double_pass() -> None:
    """Resolving twice yields the same result (no hidden state)."""
    recipe = 'min_smoke_gpu: "A100"\n'
    once, rationale_once = _resolve_smoke_gpu(recipe, "T4")
    twice, rationale_twice = _resolve_smoke_gpu(recipe, "T4")
    assert once == twice == "A100"
    assert rationale_once == rationale_twice
