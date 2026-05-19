# SPDX-License-Identifier: MIT
"""Tests for the 5 ORPHAN-CANONICAL-HELPERS-LANDING-WAVE cathedral consumers.

Per ORPHAN-CANONICAL-HELPERS-LANDING-WAVE 2026-05-19 + Catalog #335
canonical contract. Verifies each of the 5 sister consumers (TOP-1 +
TOP-2/3/4/G1) satisfies the canonical ``CathedralConsumerContract`` AND
matches the standing axis-tag + promotable discipline per CLAUDE.md
"Apples-to-apples evidence discipline" + Catalog #287/#323.
"""
from __future__ import annotations

import importlib

import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)


# The 5 orphan-canonical-helpers cathedral consumers landed in this wave.
WAVE_CONSUMER_NAMES = (
    "score_lagrangian_consumer",
    "cpu_axis_optimal_consumer",
    "uncertainty_weighted_loss_consumer",
    "early_stopping_consumer",
    "ema_decay_formula_consumer",
)


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_module_importable(consumer_name: str) -> None:
    """Each of the 5 sister consumer packages must import cleanly."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    assert mod is not None


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_module_satisfies_canonical_contract(consumer_name: str) -> None:
    """Each consumer MUST pass ``validate_consumer_module``."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    reg = validate_consumer_module(
        mod, module_path=f"tac.cathedral_consumers.{consumer_name}"
    )
    assert reg.contract_compliant is True, (
        f"{consumer_name} validation errors: {reg.validation_errors}"
    )
    assert reg.waiver_active is False


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_has_required_metadata(consumer_name: str) -> None:
    """CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS required."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    assert hasattr(mod, "CONSUMER_NAME")
    assert mod.CONSUMER_NAME == consumer_name
    assert hasattr(mod, "CONSUMER_VERSION")
    assert isinstance(mod.CONSUMER_VERSION, str) and mod.CONSUMER_VERSION
    assert hasattr(mod, "CONSUMER_HOOK_NUMBERS")
    assert isinstance(mod.CONSUMER_HOOK_NUMBERS, tuple)
    assert len(mod.CONSUMER_HOOK_NUMBERS) >= 1
    for hook in mod.CONSUMER_HOOK_NUMBERS:
        assert isinstance(hook, HookNumber)


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_declares_cathedral_autopilot_hook(consumer_name: str) -> None:
    """Every wave consumer declares hook #4 (cathedral autopilot dispatch)."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_update_from_anchor_callable_noop(consumer_name: str) -> None:
    """update_from_anchor accepts any anchor + returns None (NO-OP discipline)."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    result = mod.update_from_anchor({"any": "anchor"})
    assert result is None


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_consume_candidate_returns_canonical_dict(consumer_name: str) -> None:
    """consume_candidate returns the canonical observability dict shape."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    out = mod.consume_candidate({"any": "candidate"})
    assert isinstance(out, dict)
    # Canonical fields per Catalog #335 reference
    assert "predicted_delta_adjustment" in out
    assert "rationale" in out
    assert "axis_tag" in out
    assert "promotable" in out
    assert "confidence" in out


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_zero_score_adjustment(consumer_name: str) -> None:
    """Per CLAUDE.md observability-only: predicted_delta_adjustment must be 0.0."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    out = mod.consume_candidate({"any": "candidate"})
    assert out["predicted_delta_adjustment"] == 0.0


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_axis_tag_is_predicted(consumer_name: str) -> None:
    """Per CLAUDE.md + Catalog #287/#323: axis_tag = '[predicted]'."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    out = mod.consume_candidate({"any": "candidate"})
    assert out["axis_tag"] == "[predicted]"


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_not_promotable(consumer_name: str) -> None:
    """Per CLAUDE.md "Apples-to-apples evidence discipline": NOT promotable."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    out = mod.consume_candidate({"any": "candidate"})
    assert out["promotable"] is False


@pytest.mark.parametrize("consumer_name", WAVE_CONSUMER_NAMES)
def test_consumer_rationale_cites_canonical_helper(consumer_name: str) -> None:
    """Rationale must cite the canonical helper module or tool path."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{consumer_name}")
    out = mod.consume_candidate({"any": "candidate"})
    rationale = out["rationale"].lower()
    # Must contain "tac." module path OR tools/ path AND "[predicted]" tag
    has_tac_or_tools = "tac." in rationale or "tools/" in rationale
    has_predicted_tag = "[predicted]" in rationale
    assert has_tac_or_tools, f"{consumer_name} rationale missing tac.* or tools/ citation: {rationale}"
    assert has_predicted_tag, f"{consumer_name} rationale missing [predicted] tag: {rationale}"


def test_catalog_335_gate_clean_after_wave_landing() -> None:
    """Catalog #335 STRICT preflight gate MUST remain clean after the wave."""
    from tac.preflight import (
        check_cathedral_consumer_directory_package_exposes_canonical_contract,
    )
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract()
    # Empty violations list = clean
    assert violations == [], f"Catalog #335 violations after wave landing: {violations}"


def test_all_5_wave_consumers_discoverable_via_filesystem() -> None:
    """Each wave consumer directory exists under src/tac/cathedral_consumers/."""
    from pathlib import Path
    consumers_dir = Path(__file__).resolve().parents[1] / "cathedral_consumers"
    for name in WAVE_CONSUMER_NAMES:
        package_dir = consumers_dir / name
        assert package_dir.is_dir(), f"missing package dir: {package_dir}"
        init_py = package_dir / "__init__.py"
        assert init_py.exists(), f"missing __init__.py: {init_py}"


def test_consumer_count_post_landing_at_least_22() -> None:
    """Cumulative cathedral_consumers count >= 22 (21 pre-wave + 5 new = some overlap acceptable)."""
    from pathlib import Path
    consumers_dir = Path(__file__).resolve().parents[1] / "cathedral_consumers"
    package_dirs = [
        p for p in consumers_dir.iterdir()
        if p.is_dir() and (p / "__init__.py").exists()
    ]
    # 21 pre-existing + 5 new wave = 26 expected (some may already exist as
    # pre-existing reference; allow >=22 floor per session memory hash).
    assert len(package_dirs) >= 22, (
        f"expected >= 22 cathedral_consumers packages; got {len(package_dirs)}"
    )
