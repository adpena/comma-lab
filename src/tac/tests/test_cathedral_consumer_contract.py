# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral.consumer_contract canonical Protocol + validators.

Per CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT subagent landing 2026-05-19 + Catalog
#335 self-protection. Tests cover:

- Protocol satisfaction (positive: well-formed consumer)
- Field validation (missing CONSUMER_NAME / wrong type CONSUMER_HOOK_NUMBERS / etc.)
- Waiver discovery (valid rationale / placeholder rejected / outside first 30 lines)
- Validation error messages (explicit field-by-field rationale)
- Frozen dataclass invariants
- HookNumber enum membership

Sister test file: src/tac/tests/test_check_335_cathedral_consumer_directory_contract.py
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from tac.cathedral.consumer_contract import (
    CathedralConsumerContract,
    CathedralConsumerContractError,
    ConsumerRegistration,
    HookNumber,
    WAIVER_TOKEN,
    discover_waiver_in_init,
    validate_consumer_module,
)


# ---------------------------------------------------------------------------
# HookNumber enum
# ---------------------------------------------------------------------------


def test_hook_number_has_six_canonical_members() -> None:
    """Catalog #125 6-hook wire-in non-negotiable: 6 surfaces exactly."""
    members = list(HookNumber)
    assert len(members) == 6
    assert HookNumber.SENSITIVITY_MAP == 1
    assert HookNumber.PARETO_CONSTRAINT == 2
    assert HookNumber.BIT_ALLOCATOR == 3
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH == 4
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR == 5
    assert HookNumber.PROBE_DISAMBIGUATOR == 6


def test_hook_number_intenum_for_compactness() -> None:
    assert int(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH) == 4


# ---------------------------------------------------------------------------
# ConsumerRegistration frozen-dataclass invariants
# ---------------------------------------------------------------------------


def test_consumer_registration_well_formed() -> None:
    reg = ConsumerRegistration(
        consumer_name="example",
        consumer_version="0.1.0",
        consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
        consumer_module_path="tac.cathedral_consumers.example",
        contract_compliant=True,
    )
    assert reg.consumer_name == "example"
    assert reg.contract_compliant is True
    assert reg.waiver_active is False


def test_consumer_registration_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="consumer_name"):
        ConsumerRegistration(
            consumer_name="",
            consumer_version="0.1.0",
            consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
            consumer_module_path="x",
            contract_compliant=True,
        )


def test_consumer_registration_rejects_non_tuple_hooks() -> None:
    with pytest.raises(ValueError, match="consumer_hook_numbers"):
        ConsumerRegistration(
            consumer_name="x",
            consumer_version="0.1.0",
            consumer_hook_numbers=[HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH],  # type: ignore[arg-type]
            consumer_module_path="x",
            contract_compliant=True,
        )


def test_consumer_registration_rejects_non_hook_entries() -> None:
    with pytest.raises(ValueError, match="HookNumber"):
        ConsumerRegistration(
            consumer_name="x",
            consumer_version="0.1.0",
            consumer_hook_numbers=(4,),  # type: ignore[arg-type]
            consumer_module_path="x",
            contract_compliant=True,
        )


def test_consumer_registration_waiver_requires_rationale() -> None:
    with pytest.raises(ValueError, match="waiver_rationale"):
        ConsumerRegistration(
            consumer_name="x",
            consumer_version="0.1.0",
            consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
            consumer_module_path="x",
            contract_compliant=False,
            waiver_active=True,
            waiver_rationale=None,
        )


def test_consumer_registration_waiver_incompatible_with_compliant() -> None:
    with pytest.raises(ValueError, match="waiver_active=True is incompatible"):
        ConsumerRegistration(
            consumer_name="x",
            consumer_version="0.1.0",
            consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
            consumer_module_path="x",
            contract_compliant=True,
            waiver_active=True,
            waiver_rationale="legitimate reason here",
        )


def test_consumer_registration_is_frozen() -> None:
    reg = ConsumerRegistration(
        consumer_name="x",
        consumer_version="0.1.0",
        consumer_hook_numbers=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,),
        consumer_module_path="x",
        contract_compliant=True,
    )
    with pytest.raises(Exception):
        reg.consumer_name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_consumer_module: positive
# ---------------------------------------------------------------------------


def _make_well_formed_module(name: str = "fake_consumer") -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.CONSUMER_NAME = "test_consumer"
    mod.CONSUMER_VERSION = "1.0.0"
    mod.CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)
    mod.update_from_anchor = lambda anchor: None
    mod.consume_candidate = lambda candidate: {
        "predicted_delta_adjustment": 0.0,
        "rationale": "test",
        "axis_tag": "[predicted]",
    }
    return mod


def test_validate_well_formed_module_compliant() -> None:
    mod = _make_well_formed_module()
    reg = validate_consumer_module(mod, module_path="tac.cathedral_consumers.fake")
    assert reg.contract_compliant is True
    assert reg.consumer_name == "test_consumer"
    assert reg.consumer_version == "1.0.0"
    assert reg.consumer_hook_numbers == (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)
    assert reg.validation_errors == ()


def test_validate_satisfies_protocol_runtime_check() -> None:
    mod = _make_well_formed_module()
    # The Protocol is runtime_checkable.
    assert isinstance(mod, CathedralConsumerContract)


def test_validate_multiple_hooks_accepted() -> None:
    mod = _make_well_formed_module()
    mod.CONSUMER_HOOK_NUMBERS = (
        HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
        HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    )
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is True
    assert len(reg.consumer_hook_numbers) == 2


# ---------------------------------------------------------------------------
# validate_consumer_module: missing fields
# ---------------------------------------------------------------------------


def test_validate_missing_consumer_name() -> None:
    mod = _make_well_formed_module()
    delattr(mod, "CONSUMER_NAME")
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("CONSUMER_NAME" in err for err in reg.validation_errors)


def test_validate_missing_consumer_version() -> None:
    mod = _make_well_formed_module()
    delattr(mod, "CONSUMER_VERSION")
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("CONSUMER_VERSION" in err for err in reg.validation_errors)


def test_validate_missing_hook_numbers() -> None:
    mod = _make_well_formed_module()
    delattr(mod, "CONSUMER_HOOK_NUMBERS")
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("CONSUMER_HOOK_NUMBERS" in err for err in reg.validation_errors)


def test_validate_empty_hook_numbers() -> None:
    mod = _make_well_formed_module()
    mod.CONSUMER_HOOK_NUMBERS = ()
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("must not be empty" in err for err in reg.validation_errors)


def test_validate_missing_update_from_anchor() -> None:
    mod = _make_well_formed_module()
    delattr(mod, "update_from_anchor")
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("update_from_anchor" in err for err in reg.validation_errors)


def test_validate_missing_consume_candidate() -> None:
    mod = _make_well_formed_module()
    delattr(mod, "consume_candidate")
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("consume_candidate" in err for err in reg.validation_errors)


# ---------------------------------------------------------------------------
# validate_consumer_module: wrong-type fields
# ---------------------------------------------------------------------------


def test_validate_wrong_type_consumer_name() -> None:
    mod = _make_well_formed_module()
    mod.CONSUMER_NAME = 42  # type: ignore[assignment]
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("CONSUMER_NAME must be str" in err for err in reg.validation_errors)


def test_validate_wrong_type_hook_entry() -> None:
    mod = _make_well_formed_module()
    mod.CONSUMER_HOOK_NUMBERS = (4,)  # raw int instead of HookNumber
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any("HookNumber" in err for err in reg.validation_errors)


def test_validate_non_callable_update_from_anchor() -> None:
    mod = _make_well_formed_module()
    mod.update_from_anchor = "not callable"  # type: ignore[assignment]
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is False
    assert any(
        "update_from_anchor must be callable" in err for err in reg.validation_errors
    )


def test_validate_none_module_raises() -> None:
    with pytest.raises(CathedralConsumerContractError, match="module is None"):
        validate_consumer_module(None)


# ---------------------------------------------------------------------------
# Waiver discovery
# ---------------------------------------------------------------------------


def test_discover_waiver_nonexistent_path(tmp_path: Path) -> None:
    rationale, active = discover_waiver_in_init(tmp_path / "nonexistent.py")
    assert rationale is None
    assert active is False


def test_discover_waiver_no_marker(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text("# regular init\n\nVERSION = '0.1.0'\n")
    rationale, active = discover_waiver_in_init(init)
    assert rationale is None
    assert active is False


def test_discover_waiver_valid_rationale(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text(
        f"# SPDX-License-Identifier: MIT\n# {WAIVER_TOKEN}:Pending Phase 2 wire-in\n"
    )
    rationale, active = discover_waiver_in_init(init)
    assert rationale == "Pending Phase 2 wire-in"
    assert active is True


def test_discover_waiver_placeholder_rejected(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text(f"# {WAIVER_TOKEN}:<rationale>\n")
    rationale, active = discover_waiver_in_init(init)
    assert rationale == "<rationale>"
    assert active is False


def test_discover_waiver_placeholder_reason_rejected(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text(f"# {WAIVER_TOKEN}:<reason>\n")
    rationale, active = discover_waiver_in_init(init)
    assert active is False


def test_discover_waiver_short_rationale_rejected(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text(f"# {WAIVER_TOKEN}:abc\n")  # 3 chars, below min 4
    rationale, active = discover_waiver_in_init(init)
    assert active is False


def test_discover_waiver_outside_first_30_lines(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    lines = ["# filler\n"] * 35
    lines.append(f"# {WAIVER_TOKEN}:legitimate rationale here\n")
    init.write_text("".join(lines))
    rationale, active = discover_waiver_in_init(init)
    # Line 36 is OUTSIDE the 30-line window so waiver not found.
    assert rationale is None
    assert active is False


def test_discover_waiver_at_line_30_boundary_accepted(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    lines = ["# filler\n"] * 29
    lines.append(f"# {WAIVER_TOKEN}:legitimate rationale here\n")
    init.write_text("".join(lines))
    rationale, active = discover_waiver_in_init(init)
    assert active is True
