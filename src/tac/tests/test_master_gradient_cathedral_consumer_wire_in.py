# SPDX-License-Identifier: MIT
"""Tests for the 8 Cable D master-gradient cathedral consumer wrappers.

Per `lane_master_gradient_consumer_cathedral_wire_in_20260519` Phase 1 +
Catalog #335 (cathedral_consumers/* canonical contract) + Catalog #287
(observability-only contribution; promotable=False default) + Catalog #323
(canonical Provenance umbrella).

Covers:
- All 8 wrappers satisfy CathedralConsumerContract Protocol
- All 8 are auto-discovered by discover_and_register_consumers
- Catalog #335 STRICT gate passes (8 new consumers contract-compliant)
- Observability-only contract: promotable=False per Catalog #287/#323
- Each consumer's consume_candidate returns a well-formed dict

Sister-test pattern mirrors `test_cathedral_consumer_contract.py` (canonical
sister) + `test_check_335_cathedral_consumer_directory_contract.py`.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tac.cathedral.consumer_contract import (
    CathedralConsumerContract,
    HookNumber,
    validate_consumer_module,
)


# The 8 NEW Cable D master-gradient consumer wrappers landed in this commit.
NEW_CONSUMER_PACKAGES = (
    "per_pair_pareto_envelope_consumer",
    "per_pair_lagrangian_lambda_bisection_consumer",
    "per_pair_lora_supervision_signal_consumer",
    "per_pair_coding_budget_allocation_consumer",
    "engineered_correction_targeting_consumer",
    "per_pair_kkt_residuals_consumer",
    "per_pair_volterra_cross_terms_consumer",
    "gradient_informed_decoder_pruning_consumer",
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Phase 1: Each wrapper satisfies CathedralConsumerContract Protocol
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pkg_name", NEW_CONSUMER_PACKAGES)
def test_wrapper_satisfies_protocol(pkg_name: str) -> None:
    """Each of the 8 wrappers satisfies CathedralConsumerContract."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    assert isinstance(mod, type(importlib.import_module("tac")))
    # runtime_checkable Protocol check
    assert isinstance(mod, CathedralConsumerContract), (
        f"{pkg_name} fails CathedralConsumerContract Protocol check"
    )


@pytest.mark.parametrize("pkg_name", NEW_CONSUMER_PACKAGES)
def test_wrapper_validates_clean(pkg_name: str) -> None:
    """validate_consumer_module returns contract_compliant=True for each."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    reg = validate_consumer_module(mod, module_path=f"tac.cathedral_consumers.{pkg_name}")
    assert reg.contract_compliant, (
        f"{pkg_name} validation errors: {list(reg.validation_errors)}"
    )
    assert reg.consumer_name == pkg_name
    assert reg.consumer_version == "0.1.0"
    assert len(reg.consumer_hook_numbers) >= 1


@pytest.mark.parametrize("pkg_name", NEW_CONSUMER_PACKAGES)
def test_wrapper_declares_cathedral_dispatch_hook(pkg_name: str) -> None:
    """Every wrapper declares CATHEDRAL_AUTOPILOT_DISPATCH (hook #4) per task spec."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS


# ---------------------------------------------------------------------------
# Phase 2: Auto-discovery loop ingests all 8 new wrappers
# ---------------------------------------------------------------------------

def test_all_8_wrappers_auto_discovered() -> None:
    """discover_and_register_consumers ingests all 8 new packages."""
    # Late import: cathedral_autopilot is a tool, not a tac.* module
    import sys
    tools_dir = REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from cathedral_autopilot_autonomous_loop import discover_and_register_consumers

    regs = discover_and_register_consumers(repo_root=REPO_ROOT)
    discovered_names = {r["consumer_name"] for r in regs}
    for pkg_name in NEW_CONSUMER_PACKAGES:
        assert pkg_name in discovered_names, (
            f"{pkg_name} not in discovered consumers: {sorted(discovered_names)}"
        )


def test_all_8_wrappers_contract_compliant_after_discovery() -> None:
    """All 8 new wrappers report contract_compliant=True post-discovery."""
    import sys
    tools_dir = REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from cathedral_autopilot_autonomous_loop import discover_and_register_consumers

    regs = discover_and_register_consumers(repo_root=REPO_ROOT)
    new_regs = {r["consumer_name"]: r for r in regs if r["consumer_name"] in NEW_CONSUMER_PACKAGES}
    assert len(new_regs) == len(NEW_CONSUMER_PACKAGES)
    for name, reg in new_regs.items():
        assert reg["contract_compliant"], (
            f"{name} not contract_compliant: errors={reg['validation_errors']}"
        )
        assert not reg["waiver_active"], f"{name} unexpectedly waived"


def test_cumulative_cathedral_consumers_count_at_least_20() -> None:
    """Post-landing, cumulative cathedral_consumers count ≥ 20.

    Sister Slot 3 WIRING-REMEDIATION T2 landed 12 consumers + 1 example
    + THIS subagent's 8 new wrappers = ≥ 21 total (some additional consumers
    may have landed in parallel).
    """
    import sys
    tools_dir = REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from cathedral_autopilot_autonomous_loop import discover_and_register_consumers

    regs = discover_and_register_consumers(repo_root=REPO_ROOT)
    # Lower bound: 12 sister + 8 new = 20 (example may be filtered out by name pattern).
    assert len(regs) >= 20, (
        f"Cumulative cathedral_consumers count {len(regs)} < 20; "
        f"discovered={sorted(r['consumer_name'] for r in regs)}"
    )


# ---------------------------------------------------------------------------
# Phase 3: Catalog #335 STRICT gate passes with 8 new consumers
# ---------------------------------------------------------------------------

def test_catalog_335_strict_gate_passes_with_8_new_consumers() -> None:
    """Catalog #335 STRICT preflight gate refuses 0 violations.

    The 8 new wrappers are all contract-compliant + the auto-discovery loop
    sees them cleanly, so Catalog #335 LIVE_COUNT remains 0.
    """
    from tac.preflight import (
        check_cathedral_consumer_directory_package_exposes_canonical_contract,
    )
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    # Live count MUST be 0 — the 8 new wrappers + sister 12 are all contract-compliant.
    assert violations == [], (
        f"Catalog #335 LIVE_COUNT non-zero: {violations}"
    )


def test_catalog_335_strict_mode_does_not_raise() -> None:
    """Catalog #335 strict=True does NOT raise (violations remain 0)."""
    from tac.preflight import (
        check_cathedral_consumer_directory_package_exposes_canonical_contract,
    )
    # strict=True should not raise because live count = 0
    result = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        repo_root=REPO_ROOT, strict=True, verbose=False
    )
    assert result == []


# ---------------------------------------------------------------------------
# Phase 4: Observability-only contract per Catalog #287/#323
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pkg_name", NEW_CONSUMER_PACKAGES)
def test_consume_candidate_returns_well_formed_dict(pkg_name: str) -> None:
    """Each consume_candidate returns a dict with the canonical keys."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    result = mod.consume_candidate({"archive_sha256": "x" * 64})
    assert isinstance(result, dict), f"{pkg_name}.consume_candidate must return dict"
    assert "predicted_delta_adjustment" in result
    assert "rationale" in result
    assert "axis_tag" in result


@pytest.mark.parametrize("pkg_name", NEW_CONSUMER_PACKAGES)
def test_consume_candidate_is_observability_only(pkg_name: str) -> None:
    """Each consumer's contribution is observability-only per Catalog #287/#323.

    predicted_delta_adjustment MUST be 0.0 (no score adjustment) and
    promotable MUST be False (non-promotable) until paired empirical
    anchor lands per Catalog #127.
    """
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    result = mod.consume_candidate({"archive_sha256": "x" * 64})
    assert result["predicted_delta_adjustment"] == 0.0, (
        f"{pkg_name} must be zero-adjustment (observability-only)"
    )
    assert result.get("promotable") is False, (
        f"{pkg_name} must be non-promotable per Catalog #287/#323"
    )


@pytest.mark.parametrize("pkg_name", NEW_CONSUMER_PACKAGES)
def test_consume_candidate_axis_tag_is_predicted(pkg_name: str) -> None:
    """Each consumer carries axis_tag=[predicted] per CLAUDE.md 'Apples-to-apples'."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    result = mod.consume_candidate({"archive_sha256": "x" * 64})
    assert result["axis_tag"] == "[predicted]", (
        f"{pkg_name} axis_tag={result['axis_tag']!r}; must be [predicted] "
        "for observability-only contribution"
    )


@pytest.mark.parametrize("pkg_name", NEW_CONSUMER_PACKAGES)
def test_consume_candidate_rationale_cites_producer_module(pkg_name: str) -> None:
    """Each rationale cites the producer module path for cite-chain.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": rationale
    must be human-readable + cite back to the canonical producer.
    """
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    result = mod.consume_candidate({"archive_sha256": "x" * 64})
    rationale = result["rationale"]
    assert "tac.master_gradient_consumers" in rationale, (
        f"{pkg_name} rationale must cite tac.master_gradient_consumers producer"
    )


@pytest.mark.parametrize("pkg_name", NEW_CONSUMER_PACKAGES)
def test_update_from_anchor_is_callable(pkg_name: str) -> None:
    """update_from_anchor is callable with a synthetic anchor (NO-OP design)."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    # Should not raise on any anchor shape (NO-OP per design).
    mod.update_from_anchor({"evidence_grade": "predicted", "axis": "diagnostic_cpu"})
    mod.update_from_anchor(None)


# ---------------------------------------------------------------------------
# Phase 5: Hook-number declarations match Cable D landing memo
# ---------------------------------------------------------------------------

# Per Cable D landing memo §"6-hook wire-in declaration":
# - per_pair_pareto_envelope     → #2 (Pareto) + #4 (cathedral)
# - per_pair_lagrangian_lambda_bisection → #1 (sensitivity-map) + #4
# - per_pair_lora_supervision_signal → #4 + #5 (CL posterior)
# - per_pair_coding_budget_allocation → #3 (bit-allocator) + #4
# - engineered_correction_targeting → #4 (+ #3 bit-allocator)
# - per_pair_kkt_residuals → #2 + #4 + #6 (probe-disambiguator)
# - per_pair_volterra_cross_terms → #1 + #4 + #2
# - gradient_informed_decoder_pruning → #3 + #4 + #6
EXPECTED_HOOKS = {
    "per_pair_pareto_envelope_consumer": (HookNumber.PARETO_CONSTRAINT,),
    "per_pair_lagrangian_lambda_bisection_consumer": (HookNumber.SENSITIVITY_MAP,),
    "per_pair_lora_supervision_signal_consumer": (HookNumber.CONTINUAL_LEARNING_POSTERIOR,),
    "per_pair_coding_budget_allocation_consumer": (HookNumber.BIT_ALLOCATOR,),
    "engineered_correction_targeting_consumer": (HookNumber.BIT_ALLOCATOR,),
    "per_pair_kkt_residuals_consumer": (
        HookNumber.PARETO_CONSTRAINT,
        HookNumber.PROBE_DISAMBIGUATOR,
    ),
    "per_pair_volterra_cross_terms_consumer": (
        HookNumber.SENSITIVITY_MAP,
        HookNumber.PARETO_CONSTRAINT,
    ),
    "gradient_informed_decoder_pruning_consumer": (
        HookNumber.BIT_ALLOCATOR,
        HookNumber.PROBE_DISAMBIGUATOR,
    ),
}


@pytest.mark.parametrize("pkg_name,required_hooks", list(EXPECTED_HOOKS.items()))
def test_hook_numbers_match_cable_d_landing_memo(
    pkg_name: str, required_hooks: tuple[HookNumber, ...]
) -> None:
    """Each wrapper declares the hook numbers cited in the Cable D landing memo."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{pkg_name}")
    declared = set(mod.CONSUMER_HOOK_NUMBERS)
    for required in required_hooks:
        assert required in declared, (
            f"{pkg_name} missing required hook {required.name}; "
            f"declared={[h.name for h in mod.CONSUMER_HOOK_NUMBERS]}"
        )
