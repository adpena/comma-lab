# SPDX-License-Identifier: MIT
"""Operator-authorize dispatch-gate tests.

The recipe is the operator-facing dispatch contract. If a recipe declares
``dispatch_enabled: false`` or leaves ``pre_promotion_blockers`` in place,
``tools/operator_authorize.py`` must refuse before confirmation, lane claim, or
provider setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "tools"))

from operator_authorize import (  # noqa: E402
    Recipe,
    _recipe_dispatch_refusal,
    _required_input_flag_values_from_recipe,
)


def _recipe(raw: dict) -> Recipe:
    return Recipe(name="test", path=Path("test.yaml"), raw=raw)


def test_dispatch_enabled_false_refuses_with_blockers() -> None:
    refusal = _recipe_dispatch_refusal(
        _recipe(
            {
                "dispatch_enabled": False,
                "dispatch_blockers": ["remote_driver_missing"],
                "defer_reason": "remote driver missing\nextra detail",
            }
        )
    )
    assert refusal is not None
    assert "dispatch_enabled=false" in refusal
    assert "remote_driver_missing" in refusal
    assert "remote driver missing" in refusal


def test_pre_promotion_blockers_refuse_until_recipe_clears_them() -> None:
    refusal = _recipe_dispatch_refusal(
        _recipe(
            {
                "dispatch_enabled": True,
                "pre_promotion_blockers": [
                    "sane_hnerv_first_anchor_required",
                    "balle_renderer_first_anchor_required",
                ],
            }
        )
    )
    assert refusal is not None
    assert "pre_promotion_blockers still declared" in refusal
    assert "sane_hnerv_first_anchor_required" in refusal
    assert "balle_renderer_first_anchor_required" in refusal


def test_dispatch_blockers_refuse_even_if_dispatch_enabled_omitted() -> None:
    refusal = _recipe_dispatch_refusal(
        _recipe(
            {
                "name": "substrate_cnerv_modal_a100_dispatch",
                "dispatch_blockers": [
                    "tier_manifest_required_for_catalog_151_compliance",
                    "trainer_scaffold_only_non_smoke_systemexit_gate",
                ],
            }
        )
    )
    assert refusal is not None
    assert "dispatch_blockers still declared" in refusal
    assert "tier_manifest_required_for_catalog_151_compliance" in refusal
    assert "trainer_scaffold_only_non_smoke_systemexit_gate" in refusal


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"dispatch_enabled": True},
        {"dispatch_enabled": True, "pre_promotion_blockers": []},
    ],
)
def test_dispatchable_recipe_has_no_refusal(raw: dict) -> None:
    assert _recipe_dispatch_refusal(_recipe(raw)) is None


def test_recipe_required_inputs_become_validator_flag_values() -> None:
    values = _required_input_flag_values_from_recipe(
        _recipe(
            {
                "required_input_files": [
                    {"flag": "--video-path", "default_path": "upstream/videos/0.mkv"},
                    {"flag": "--ignored-no-default"},
                    "bad-shape",
                ]
            }
        )
    )
    assert values == ["--video-path=upstream/videos/0.mkv"]
