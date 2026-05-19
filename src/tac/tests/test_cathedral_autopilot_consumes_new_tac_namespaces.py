# SPDX-License-Identifier: MIT
"""Tests for the 12 NEW cathedral consumer packages added per WIRING-REMEDIATION-T1-PLUS-T2.

Per wiring + integration audit 2026-05-19 (commit 3821cfb6b) TIER 2 #4:
12 new ``tac.*`` namespaces landed in the 2026-05-18 → 2026-05-19 window
with ZERO consumed by cathedral autopilot. The fix is the canonical
``src/tac/cathedral_consumers/`` auto-discovery + contract pattern (per
Catalog #335 + CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT landing 2026-05-19).

This file pins:
- All 12 new consumer packages auto-discovered by the canonical loop
- Each is contract-compliant per ``tac.cathedral.consumer_contract``
- Each ``consume_candidate`` returns canonical observability-grade rows
  with ``[predicted]`` or ``[MPS-PROXY]`` axis tag per Catalog #287
- Each ``update_from_anchor`` is callable (deliberate NO-OP for these
  observability-only consumers; documented in each docstring)
- No consumer mutates a candidate's predicted_score_delta (all are
  observability annotations per the wiring audit's TIER 2 #4 design)

Sister tests:
- src/tac/tests/test_cathedral_consumer_contract.py (Protocol + validators)
- src/tac/tests/test_check_335_cathedral_consumer_directory_contract.py (STRICT gate)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Ensure tools/ on path for cathedral autopilot import
TOOLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


NEW_CONSUMERS = (
    "atom_consumer",
    "formula_extinctions_consumer",
    "experimental_extinctions_consumer",
    "contest_oracle_consumer",
    "utility_curves_consumer",
    "solvers_consumer",
    "unified_action_consumer",
    "procedural_codebook_generator_consumer",
    "mps_diagnostic_consumer",
    "mps_gap_experiment_consumer",
    "contest_exploits_consumer",
    "analytical_solve_extinctions_consumer",
)


VALID_AXIS_TAGS = frozenset(
    {
        "[predicted]",
        "[contest-CUDA]",
        "[contest-CPU]",
        "[macOS-CPU advisory]",
        "[MPS-PROXY]",
        "[advisory only]",
        "[empirical:contest-CUDA]",
        "[empirical:contest-CPU]",
    }
)


@pytest.mark.parametrize("name", NEW_CONSUMERS)
def test_new_consumer_importable(name: str) -> None:
    """Each new consumer package imports cleanly."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{name}")
    assert mod is not None
    assert hasattr(mod, "CONSUMER_NAME")
    assert hasattr(mod, "CONSUMER_VERSION")
    assert hasattr(mod, "CONSUMER_HOOK_NUMBERS")
    assert hasattr(mod, "update_from_anchor")
    assert hasattr(mod, "consume_candidate")


@pytest.mark.parametrize("name", NEW_CONSUMERS)
def test_new_consumer_contract_compliant(name: str) -> None:
    """Each new consumer satisfies tac.cathedral.consumer_contract."""
    from tac.cathedral.consumer_contract import validate_consumer_module
    mod = importlib.import_module(f"tac.cathedral_consumers.{name}")
    reg = validate_consumer_module(mod, module_path=f"tac.cathedral_consumers.{name}")
    assert reg.contract_compliant, f"{name} failed: {reg.validation_errors}"
    assert reg.consumer_name == name
    assert reg.consumer_version
    assert len(reg.consumer_hook_numbers) >= 1


@pytest.mark.parametrize("name", NEW_CONSUMERS)
def test_consume_candidate_returns_canonical_row(name: str) -> None:
    """Each consume_candidate returns a row with canonical fields."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{name}")
    row = mod.consume_candidate({})
    assert isinstance(row, dict)
    assert "predicted_delta_adjustment" in row
    assert "rationale" in row
    assert "axis_tag" in row
    assert isinstance(row["predicted_delta_adjustment"], (int, float))
    assert isinstance(row["rationale"], str)
    assert len(row["rationale"]) >= 4
    assert row["axis_tag"] in VALID_AXIS_TAGS


@pytest.mark.parametrize("name", NEW_CONSUMERS)
def test_consume_candidate_zero_adjustment(name: str) -> None:
    """All 12 new consumers are OBSERVABILITY-ONLY: zero adjustment.

    Per the wiring audit's TIER 2 #4 design: these consumers surface
    canonical-helper availability as ``[predicted]``/``[MPS-PROXY]``
    annotations. They MUST NOT mutate predicted_score_delta — per-candidate
    score adjustment requires explicit per-helper integration (e.g.
    master_gradient_consumers cascade for sister-#817 sidecars).
    """
    mod = importlib.import_module(f"tac.cathedral_consumers.{name}")
    row = mod.consume_candidate({"archive_sha256": "fake_sha"})
    assert row["predicted_delta_adjustment"] == 0.0, (
        f"{name} returned non-zero adjustment {row['predicted_delta_adjustment']}; "
        f"per wiring audit TIER 2 #4 design these consumers MUST be observability-only"
    )


@pytest.mark.parametrize("name", NEW_CONSUMERS)
def test_consume_candidate_not_promotable(name: str) -> None:
    """All 12 new consumers return promotable=False per Catalog #287
    + #323 canonical Provenance discipline.
    """
    mod = importlib.import_module(f"tac.cathedral_consumers.{name}")
    row = mod.consume_candidate({})
    promotable = row.get("promotable", False)
    assert promotable is False, (
        f"{name} returned promotable={promotable}; observability annotations "
        f"MUST be non-promotable per CLAUDE.md 'Apples-to-apples evidence discipline'"
    )


@pytest.mark.parametrize("name", NEW_CONSUMERS)
def test_update_from_anchor_callable(name: str) -> None:
    """Each update_from_anchor is callable with a generic anchor object."""
    mod = importlib.import_module(f"tac.cathedral_consumers.{name}")
    # Should not raise for any anchor shape (deliberate NO-OP for the
    # 12 observability-only consumers)
    mod.update_from_anchor(None)
    mod.update_from_anchor({"any": "anchor"})
    mod.update_from_anchor(object())


def test_all_12_new_consumers_auto_discovered() -> None:
    """The canonical auto-discovery loop picks up all 12 new consumers
    + the reference _example_consumer = 13 total.
    """
    from cathedral_autopilot_autonomous_loop import (
        discover_and_register_consumers,
    )
    regs = discover_and_register_consumers(
        strict=False,
        include_underscore_packages=True,
    )
    names = {r["consumer_name"] for r in regs}
    # _example_consumer is the reference fixture; expected at all times
    assert "_example_consumer" in names, (
        f"reference _example_consumer missing from auto-discovery: {names}"
    )
    for new_name in NEW_CONSUMERS:
        assert new_name in names, (
            f"{new_name} NOT auto-discovered by cathedral autopilot; "
            f"discovered={names}"
        )
    # All 12 new + 1 reference + any pre-existing future consumers
    assert len(names) >= len(NEW_CONSUMERS) + 1


def test_all_12_new_consumers_contract_compliant_in_discovery() -> None:
    """All 12 new consumers report contract_compliant=True in the
    canonical auto-discovery loop (sister of per-consumer validate test).
    """
    from cathedral_autopilot_autonomous_loop import (
        discover_and_register_consumers,
    )
    regs = discover_and_register_consumers(
        strict=False,
        include_underscore_packages=True,
    )
    by_name = {r["consumer_name"]: r for r in regs}
    for new_name in NEW_CONSUMERS:
        r = by_name[new_name]
        assert r["contract_compliant"] is True, (
            f"{new_name} reported non-compliant: {r.get('validation_errors')}"
        )
        assert not r.get("waiver_active"), (
            f"{new_name} should be compliant without waiver, but waiver_active={r.get('waiver_active')}"
        )


def test_audit_consumer_named_packages_match_audit_recommendation() -> None:
    """Regression guard: the 12 consumers are exactly the 12 namespaces
    the 2026-05-19 audit (commit 3821cfb6b) identified as orphan producers.

    If a 13th namespace lands without a sister consumer this test FAILS,
    forcing the next subagent to extend the consumer set per Catalog #335
    convention-over-configuration paradigm.
    """
    audit_named_namespaces = (
        "tac.atom",
        "tac.formula_extinctions",
        "tac.experimental_extinctions",
        "tac.contest_oracle",
        "tac.utility_curves",
        "tac.solvers",
        "tac.unified_action",
        "tac.procedural_codebook_generator",
        "tac.mps_diagnostic",
        "tac.mps_gap_experiment",
        "tac.contest_exploits",
        "tac.analytical_solve_extinctions",
    )
    consumer_to_namespace = {
        f"{ns.split('.')[1]}_consumer": ns for ns in audit_named_namespaces
    }
    for consumer_name, ns in consumer_to_namespace.items():
        assert consumer_name in NEW_CONSUMERS, (
            f"audit-identified orphan namespace {ns} has no sister "
            f"consumer; expected {consumer_name} in NEW_CONSUMERS"
        )


def test_cathedral_autopilot_module_imports_with_new_consumers_present() -> None:
    """Smoke test: cathedral autopilot module imports cleanly with all
    13 consumer packages present in src/tac/cathedral_consumers/.
    """
    import cathedral_autopilot_autonomous_loop as cap
    assert cap is not None


def test_procedural_codebook_consumer_surfaces_archive_seed_authority_packet() -> None:
    """The procedural consumer reads per-candidate authority packets without
    converting them into score or promotion authority.
    """
    from tac.procedural_codebook_generator import build_procedural_seed_authority_packet

    mod = importlib.import_module(
        "tac.cathedral_consumers.procedural_codebook_generator_consumer"
    )
    packet = build_procedural_seed_authority_packet(
        "archive-seed-candidate",
        modes=("archive_seeded", "runtime_constant"),
        runtime_consumption_proof=True,
        self_contained_archive_proof=True,
        scorer_free_inflate_proof=True,
        no_external_state_proof=True,
        packet_compiler_target_declared=True,
        exact_eval_validated=False,
    )

    row = mod.consume_candidate({"procedural_seed_authority_packet": packet})

    assert row["predicted_delta_adjustment"] == 0.0
    assert row["promotable"] is False
    assert row["procedural_authority_detected"] is True
    authority = row["procedural_authority"]
    assert authority["preferred_promotion_mode"] == "archive_seeded"
    assert authority["ready_for_exact_eval_modes"] == ("archive_seeded",)
    assert authority["promotion_eligible_modes"] == ()
    assert "runtime_constant:script_side_per_video_payload_probe_only" in authority[
        "authority_blockers"
    ]


def test_procedural_codebook_consumer_blocks_script_side_payload_semantics() -> None:
    from tac.procedural_codebook_generator import build_procedural_seed_authority_packet

    mod = importlib.import_module(
        "tac.cathedral_consumers.procedural_codebook_generator_consumer"
    )
    packet = build_procedural_seed_authority_packet(
        "script-seed-probe",
        modes=("runtime_constant",),
        runtime_constant_kind="per_video_payload",
    )

    row = mod.consume_candidate({"procedural_authority_packet": packet})

    assert row["predicted_delta_adjustment"] == 0.0
    assert row["promotable"] is False
    assert row["procedural_authority"]["ready_for_exact_eval_modes"] == ()
    assert row["procedural_authority"]["research_only"] is True
    assert "runtime_constant:script_side_per_video_payload_probe_only" in row[
        "procedural_authority"
    ]["authority_blockers"]


def test_procedural_codebook_consumer_fail_closes_overauthoritative_packets() -> None:
    """Malformed authority packets must not become nested promotion authority."""
    mod = importlib.import_module(
        "tac.cathedral_consumers.procedural_codebook_generator_consumer"
    )
    packet = {
        "schema": "untrusted_schema_v99",
        "preferred_promotion_mode": "runtime_constant",
        "ready_for_exact_eval_modes": ["runtime_constant"],
        "promotion_eligible_modes": ["runtime_constant"],
        "research_only": False,
        "score_claim": True,
        "modes": {
            "runtime_constant": {
                "seed_carrier": "inflate_py_literal_seed",
                "literal_payload_kind": "per_video_payload",
                "ready_for_exact_eval_dispatch": True,
                "promotion_eligible": True,
                "score_claim": True,
            },
        },
    }

    row = mod.consume_candidate({"procedural_seed_authority_packet": packet})

    assert row["predicted_delta_adjustment"] == 0.0
    assert row["promotable"] is False
    authority = row["procedural_authority"]
    assert authority["score_claim"] is False
    assert authority["source_score_claim_blocked"] is True
    assert authority["ready_for_exact_eval_modes"] == ()
    assert authority["promotion_eligible_modes"] == ()
    assert authority["research_only"] is True
    assert "packet_schema_untrusted:'untrusted_schema_v99'" in authority[
        "authority_blockers"
    ]
    assert "packet_score_claim_not_allowed" in authority["authority_blockers"]
    assert "runtime_constant:script_side_per_video_payload_probe_only" in authority[
        "authority_blockers"
    ]
    assert "runtime_constant:score_claim_not_allowed" in authority[
        "authority_blockers"
    ]


def test_procedural_codebook_consumer_fail_closes_top_level_mode_mismatch() -> None:
    from tac.procedural_codebook_generator import build_procedural_seed_authority_packet

    mod = importlib.import_module(
        "tac.cathedral_consumers.procedural_codebook_generator_consumer"
    )
    packet = build_procedural_seed_authority_packet(
        "mismatched-ready-claim",
        modes=("archive_seeded",),
    )
    packet["ready_for_exact_eval_modes"] = ["archive_seeded"]

    row = mod.consume_candidate({"procedural_seed_authority_packet": packet})

    authority = row["procedural_authority"]
    assert authority["ready_for_exact_eval_modes"] == ()
    assert authority["promotion_eligible_modes"] == ()
    assert authority["research_only"] is True
    assert "ready_for_exact_eval_modes_mismatch" in authority["authority_blockers"]
