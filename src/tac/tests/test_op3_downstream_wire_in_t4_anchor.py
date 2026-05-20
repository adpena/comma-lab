# SPDX-License-Identifier: MIT
"""WAVE-3-OP3-DOWNSTREAM-WIRE-IN-T4-ANCHOR: end-to-end T4 anchor consumption tests.

Per `feedback_wave_3_op3_v3_minibatch_redispatch_landed_20260520.md` (predecessor
OP3-V3 SUCCESS) + WAVE-3-OP3-DOWNSTREAM-WIRE-IN-T4-ANCHOR task brief
(2026-05-20). The first authoritative ``[contest-CUDA T4]`` master-gradient
anchor for the FEC6 frontier archive (``fc-01KS370Z9TF4QZMKQ9ND72KH4N``;
sidecar ``master_gradient_fec6_contest_cuda_t4_20260520.npy`` shape
``(178417, 3)`` fp32) landed via OP3-V3; this test file confirms 6-hook wire-in
per Catalog #125, in particular converting hook #3 (bit-allocator) from
"ACTIVE INDIRECTLY (score-axis dominance metadata)" to "ACTIVE DIRECTLY"
via the new canonical helper
:func:`tac.bit_allocator.per_byte.allocate_per_byte_from_master_gradient_anchor`.

Test scope:

1. ``canonical_equation_lookup_consumer`` produces equation-token-match
   annotations for fec6-tagged candidates (Catalog #344).
2. ``allocate_per_byte_from_master_gradient_anchor`` loads the (178417, 3)
   T4 sensitivity tensor + emits top-K byte allocation with anchor metadata
   threaded into the plan's notes (Catalog #125 hook #3 ACTIVE DIRECTLY).
3. ``information_theoretic_floor_consumer`` Tier B branch consumes the
   T4 anchor's operating-point + per_axis_floor signal + bounds the
   solver-derived ``predicted_delta_adjustment`` to the canonical
   ``[-0.05, 0.05]`` safety rail (Catalog #357).
4. ``substrate_fit_diagnostic_consumer`` Tier B branch consumes the T4
   anchor's per-axis residuals + emits ranking-signal annotation (Catalog
   #357).

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + Catalog #341 +
Catalog #357: every consumer return value carries the canonical non-
promotable markers (``promotable=False``, ``axis_tag`` validated, Provenance
threaded when Tier B). The T4 anchor is the empirical bridge — these tests
verify the bridge is structural, not tribal-knowledge.
"""
from __future__ import annotations

import os

import pytest

# Canonical anchors per OP3-V3 landing.
FEC6_FRONTIER_ARCHIVE_SHA = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)
"""PR #110 frontier archive sha256; carries the authoritative T4 master-gradient anchor."""

EXPECTED_CALL_ID = "fc-01KS370Z9TF4QZMKQ9ND72KH4N"
"""Modal call_id of the OP3-V3 T4 dispatch that produced the canonical anchor."""

EXPECTED_TENSOR_SHAPE = (178417, 3)
"""Per-byte sensitivity tensor shape: (N_archive_bytes, 3=[seg,pose,rate])."""


def _anchor_present() -> bool:
    """Return True iff the canonical T4 anchor is present in the live ledger.

    Defensive: tests that exercise the live anchor skip gracefully when the
    repo is checked out fresh without the .npy sidecar (e.g. CI without
    .omx/state synced).
    """
    try:
        from tac.master_gradient import latest_anchor_for_archive
    except ImportError:
        return False
    try:
        anchor = latest_anchor_for_archive(
            FEC6_FRONTIER_ARCHIVE_SHA, axis="[contest-CUDA]"
        )
    except Exception:  # noqa: BLE001
        return False
    if anchor is None:
        return False
    try:
        from pathlib import Path

        gradient_path = Path(anchor.get("gradient_array_path", ""))
        if not gradient_path.is_absolute():
            gradient_path = Path.cwd() / gradient_path
        return gradient_path.is_file()
    except Exception:  # noqa: BLE001
        return False


_ANCHOR_PRESENT = _anchor_present()


# ---------------------------------------------------------------------------
# Test 1: canonical_equation_lookup_consumer consumes fec6 candidate
# ---------------------------------------------------------------------------


def test_canonical_equation_lookup_consumer_returns_axis_tag_predicted_for_fec6():
    """Catalog #344 + Catalog #341: fec6 candidate annotation is observability-only."""
    from tac.cathedral_consumers.canonical_equation_lookup_consumer import (
        consume_candidate,
    )

    candidate = {
        "archive_sha256": FEC6_FRONTIER_ARCHIVE_SHA,
        "lane_id": "lane_brotli_cascade_fec6",
        "consumer_name": "tac.cathedral_consumers.per_byte_sensitivity_consumer",
    }
    result = consume_candidate(candidate)

    # Catalog #341 canonical-routing markers (all 3).
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_canonical_equation_lookup_consumer_matches_per_byte_equation():
    """The fec6 candidate's token-overlap matches per_byte_leverage_uniformly_distributed_v1."""
    from tac.cathedral_consumers.canonical_equation_lookup_consumer import (
        consume_candidate,
    )

    candidate = {
        "archive_sha256": FEC6_FRONTIER_ARCHIVE_SHA,
        "consumer_name": "tac.cathedral_consumers.per_byte_sensitivity_consumer",
    }
    result = consume_candidate(candidate)
    matched = result.get("matched_equations", [])
    matched_ids = {m["equation_id"] for m in matched}
    assert "per_byte_leverage_uniformly_distributed_v1" in matched_ids


def test_canonical_equation_lookup_consumer_no_match_for_unknown_candidate():
    """A candidate with no overlapping tokens degenerates to 'no canonical equation matches'."""
    from tac.cathedral_consumers.canonical_equation_lookup_consumer import (
        consume_candidate,
    )

    candidate = {
        "archive_sha256": "1" * 64,
        "lane_id": "some_unrelated_lane_no_match_xyz",
    }
    result = consume_candidate(candidate)
    assert "no canonical equation matches" in result["rationale"]


# ---------------------------------------------------------------------------
# Test 2: bit_allocator.per_byte from_master_gradient_anchor
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _ANCHOR_PRESENT,
    reason="canonical T4 anchor sidecar not present; skip live-anchor test",
)
def test_allocate_per_byte_from_master_gradient_anchor_loads_t4_shape():
    """The canonical helper loads the (178417, 3) sensitivity tensor."""
    from tac.bit_allocator import (
        PerByteAllocationMethod,
        allocate_per_byte_from_master_gradient_anchor,
    )

    plan = allocate_per_byte_from_master_gradient_anchor(
        total_budget_bits=256,
        archive_sha256=FEC6_FRONTIER_ARCHIVE_SHA,
        method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
        top_k=32,
        per_byte_bit_cap=8,
    )

    assert plan.n_bytes_in_scope == EXPECTED_TENSOR_SHAPE[0]
    assert plan.n_bytes_allocated == 32
    assert plan.total_budget_bits == 256
    assert plan.residual_bits == 0


@pytest.mark.skipif(
    not _ANCHOR_PRESENT,
    reason="canonical T4 anchor sidecar not present; skip live-anchor test",
)
def test_allocate_per_byte_from_master_gradient_anchor_threads_anchor_metadata_into_notes():
    """Plan.notes['master_gradient_anchor'] carries call_id + axis + hardware."""
    from tac.bit_allocator import allocate_per_byte_from_master_gradient_anchor

    plan = allocate_per_byte_from_master_gradient_anchor(
        total_budget_bits=128,
        archive_sha256=FEC6_FRONTIER_ARCHIVE_SHA,
    )

    anchor_notes = plan.notes["master_gradient_anchor"]
    assert anchor_notes["measurement_axis"] == "[contest-CUDA]"
    assert anchor_notes["measurement_hardware"] == "linux_x86_64_t4_modal"
    assert anchor_notes["measurement_call_id"] == EXPECTED_CALL_ID
    assert anchor_notes["n_pairs_used"] == 600
    assert anchor_notes["axis_aggregator"] == "score_weighted_sum"


@pytest.mark.skipif(
    not _ANCHOR_PRESENT,
    reason="canonical T4 anchor sidecar not present; skip live-anchor test",
)
def test_allocate_per_byte_from_master_gradient_anchor_carries_canonical_provenance():
    """Catalog #323: every allocation plan carries canonical Provenance."""
    from tac.bit_allocator import allocate_per_byte_from_master_gradient_anchor
    from tac.provenance import ProvenanceEvidenceGrade, ProvenanceKind

    plan = allocate_per_byte_from_master_gradient_anchor(
        total_budget_bits=64, archive_sha256=FEC6_FRONTIER_ARCHIVE_SHA
    )

    assert plan.score_claim is False
    assert plan.promotion_eligible is False
    assert plan.axis_tag == "[predicted]"
    assert plan.provenance.evidence_grade is ProvenanceEvidenceGrade.PREDICTED
    assert plan.provenance.artifact_kind is ProvenanceKind.PREDICTED_FROM_MODEL


def test_allocate_per_byte_from_master_gradient_anchor_rejects_invalid_aggregator():
    """The helper refuses unrecognized axis_aggregator strings."""
    from tac.bit_allocator import (
        PerByteAllocationError,
        allocate_per_byte_from_master_gradient_anchor,
    )

    with pytest.raises(PerByteAllocationError, match="axis_aggregator"):
        allocate_per_byte_from_master_gradient_anchor(
            total_budget_bits=64,
            archive_sha256=FEC6_FRONTIER_ARCHIVE_SHA,
            axis_aggregator="bogus_method",
        )


def test_allocate_per_byte_from_master_gradient_anchor_rejects_both_inputs():
    """Mutually exclusive: must pass exactly one of archive_sha256 / anchor."""
    from tac.bit_allocator import (
        PerByteAllocationError,
        allocate_per_byte_from_master_gradient_anchor,
    )

    with pytest.raises(PerByteAllocationError, match="exactly one"):
        allocate_per_byte_from_master_gradient_anchor(
            total_budget_bits=64,
            archive_sha256="0" * 64,
            anchor={"gradient_array_path": "/tmp/dummy.npy"},
        )


def test_allocate_per_byte_from_master_gradient_anchor_rejects_neither_input():
    """Mutually exclusive: must pass exactly one of archive_sha256 / anchor."""
    from tac.bit_allocator import (
        PerByteAllocationError,
        allocate_per_byte_from_master_gradient_anchor,
    )

    with pytest.raises(PerByteAllocationError, match="either"):
        allocate_per_byte_from_master_gradient_anchor(total_budget_bits=64)


# ---------------------------------------------------------------------------
# Test 3: information_theoretic_floor_consumer Tier B consumes T4 evidence
# ---------------------------------------------------------------------------


def test_information_theoretic_floor_tier_b_consumes_per_axis_signal(monkeypatch):
    """Tier B solver derives non-zero predicted_delta_adjustment from per_axis_floor."""
    monkeypatch.setenv("CONSUMER_TIER_B_MODE", "tier_b_solver")
    from tac.cathedral_consumers.information_theoretic_floor_consumer import (
        consume_candidate,
    )

    candidate = {
        "archive_sha256": FEC6_FRONTIER_ARCHIVE_SHA,
        "substrate_name": "pr101_fec6",
        "information_theoretic_floor": 0.15,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "a1afce29" + "0" * 56,
        "current_best_empirical_score": 0.19205,
        "per_axis_floor": {"seg": -0.0005, "pose": -0.001, "rate_bytes": -100},
    }
    result = consume_candidate(candidate)

    # Tier B contract: non-zero predicted_delta within safety rail; non-promotable.
    assert result["consumer_branch_kind"] == "tier_b_solver"
    assert result["predicted_delta_adjustment"] != 0.0
    assert -0.05 <= result["predicted_delta_adjustment"] <= 0.05
    assert result["axis_tag"] == "[diagnostic-CUDA]"
    assert result["promotable"] is False
    assert "provenance" in result


def test_information_theoretic_floor_tier_a_baseline_remains_observability_only(monkeypatch):
    """Tier A baseline preserves Catalog #341 invariants (predicted_delta=0.0)."""
    monkeypatch.setenv("CONSUMER_TIER_B_MODE", "tier_a_baseline")
    from tac.cathedral_consumers.information_theoretic_floor_consumer import (
        consume_candidate,
    )

    candidate = {
        "archive_sha256": FEC6_FRONTIER_ARCHIVE_SHA,
        "substrate_name": "pr101_fec6",
        "information_theoretic_floor": 0.15,
        "per_axis_floor": {"seg": -0.0005, "pose": -0.001, "rate_bytes": -100},
    }
    result = consume_candidate(candidate)
    assert result["consumer_branch_kind"] == "tier_a_baseline"
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"
    assert result["promotable"] is False


def test_information_theoretic_floor_paired_comparison_default(monkeypatch):
    """Default mode is paired_comparison (both branches fire; Tier A authoritative)."""
    monkeypatch.delenv("CONSUMER_TIER_B_MODE", raising=False)
    from tac.cathedral_consumers.information_theoretic_floor_consumer import (
        consume_candidate,
    )

    candidate = {
        "archive_sha256": FEC6_FRONTIER_ARCHIVE_SHA,
        "substrate_name": "pr101_fec6",
        "per_axis_floor": {"seg": -0.0005, "pose": -0.001, "rate_bytes": -100},
    }
    result = consume_candidate(candidate)
    assert result["consumer_branch_kind"] == "paired_comparison_authoritative_tier_a"
    # Authoritative payload = Tier A (predicted_delta = 0.0)
    assert result["predicted_delta_adjustment"] == 0.0
    # Tier B payload available for observability
    assert "tier_b_paired_payload" in result
    tier_b_payload = result["tier_b_paired_payload"]
    assert -0.05 <= tier_b_payload["predicted_delta_adjustment"] <= 0.05


# ---------------------------------------------------------------------------
# Test 4: substrate_fit_diagnostic_consumer Tier B consumes T4 evidence
# ---------------------------------------------------------------------------


def test_substrate_fit_diagnostic_tier_a_baseline_observability_only(monkeypatch):
    """Tier A baseline preserves Catalog #341 invariants."""
    monkeypatch.setenv("CONSUMER_TIER_B_MODE", "tier_a_baseline")
    from tac.cathedral_consumers.substrate_fit_diagnostic_consumer import (
        consume_candidate,
    )

    candidate = {
        "archive_sha256": FEC6_FRONTIER_ARCHIVE_SHA,
        "substrate_name": "pr101_fec6",
        "substrate_fit_scores": {
            "pr101_fec6": 0.95,
            "pr106": 0.78,
        },
        "m_contest_array_sha256": "a1afce29" + "0" * 56,
        "per_axis_residuals": {"seg": -0.0003, "pose": -0.0005},
    }
    result = consume_candidate(candidate)
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"
    assert result["promotable"] is False
    assert result["consumer_branch_kind"] == "tier_a_baseline"


def test_substrate_fit_diagnostic_tier_b_solver_emits_signal(monkeypatch):
    """Tier B solver consumes per_axis_residuals + emits non-promotable ranking signal."""
    monkeypatch.setenv("CONSUMER_TIER_B_MODE", "tier_b_solver")
    from tac.cathedral_consumers.substrate_fit_diagnostic_consumer import (
        consume_candidate,
    )

    candidate = {
        "archive_sha256": FEC6_FRONTIER_ARCHIVE_SHA,
        "substrate_name": "pr101_fec6",
        "substrate_fit_scores": {"pr101_fec6": 0.95, "pr106": 0.78},
        "per_axis_residuals": {"seg": -0.0003, "pose": -0.0005},
    }
    result = consume_candidate(candidate)
    assert result["consumer_branch_kind"] == "tier_b_solver"
    # Tier B contract per Catalog #357: non-promotable + diagnostic axis.
    assert result["promotable"] is False
    assert result["axis_tag"] == "[diagnostic-CPU]"
    assert "provenance" in result
    # safety rail bound preserved
    assert -0.05 <= result["predicted_delta_adjustment"] <= 0.05


# ---------------------------------------------------------------------------
# Sister regression: existing canonical helper smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _ANCHOR_PRESENT,
    reason="canonical T4 anchor sidecar not present; skip live-anchor regression",
)
def test_latest_anchor_for_archive_finds_t4_contest_cuda_anchor():
    """tac.master_gradient.latest_anchor_for_archive routes to the T4 anchor."""
    from tac.master_gradient import latest_anchor_for_archive

    anchor = latest_anchor_for_archive(
        FEC6_FRONTIER_ARCHIVE_SHA, axis="[contest-CUDA]"
    )
    assert anchor is not None
    assert anchor["measurement_axis"] == "[contest-CUDA]"
    assert anchor["measurement_hardware"] == "linux_x86_64_t4_modal"
    assert anchor["measurement_call_id"] == EXPECTED_CALL_ID
    assert anchor["n_pairs_used"] == 600
    assert anchor["archive_sha256"] == FEC6_FRONTIER_ARCHIVE_SHA


@pytest.mark.skipif(
    not _ANCHOR_PRESENT,
    reason="canonical T4 anchor sidecar not present; skip live-anchor regression",
)
def test_load_aggregate_gradient_from_anchor_returns_canonical_shape():
    """tac.master_gradient_consumers.load_aggregate_gradient_from_anchor returns (178417, 3)."""
    from tac.master_gradient_consumers import load_aggregate_gradient_from_anchor

    array, anchor = load_aggregate_gradient_from_anchor(
        archive_sha256=FEC6_FRONTIER_ARCHIVE_SHA
    )
    assert array.shape == EXPECTED_TENSOR_SHAPE
    assert str(array.dtype) == "float32"
    assert anchor["measurement_call_id"] == EXPECTED_CALL_ID


# ---------------------------------------------------------------------------
# Catalog #335 / #341 / #357 sister regression
# ---------------------------------------------------------------------------


def test_all_three_consumers_canonical_contract_intact():
    """Catalog #335: all 3 consumers expose the canonical Protocol contract."""
    from tac.cathedral_consumers.canonical_equation_lookup_consumer import (
        CONSUMER_HOOK_NUMBERS as ceq_hooks,
        CONSUMER_NAME as ceq_name,
    )
    from tac.cathedral_consumers.information_theoretic_floor_consumer import (
        CONSUMER_HOOK_NUMBERS as itf_hooks,
        CONSUMER_NAME as itf_name,
        CONSUMER_TIER as itf_tier,
    )
    from tac.cathedral_consumers.substrate_fit_diagnostic_consumer import (
        CONSUMER_HOOK_NUMBERS as sfd_hooks,
        CONSUMER_NAME as sfd_name,
        CONSUMER_TIER as sfd_tier,
    )
    from tac.cathedral.consumer_contract import ConsumerTier, HookNumber

    assert ceq_name == "canonical_equation_lookup_consumer"
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in ceq_hooks

    assert itf_name == "information_theoretic_floor_consumer"
    assert itf_tier is ConsumerTier.TIER_B_SCORE_CONTRIBUTING

    assert sfd_name == "substrate_fit_diagnostic_consumer"
    assert sfd_tier is ConsumerTier.TIER_B_SCORE_CONTRIBUTING


# ---------------------------------------------------------------------------
# Hook #3 wire-in regression: importability + namespace
# ---------------------------------------------------------------------------


def test_bit_allocator_exports_from_master_gradient_anchor_helper():
    """The new helper is exported from the bit_allocator package namespace."""
    from tac.bit_allocator import (
        allocate_per_byte_from_master_gradient_anchor as helper,
    )

    assert callable(helper)


def test_bit_allocator_module_all_includes_new_helper():
    """tac.bit_allocator.__all__ explicitly lists the new helper."""
    import tac.bit_allocator as ba

    assert "allocate_per_byte_from_master_gradient_anchor" in ba.__all__


def test_bit_allocator_per_byte_module_all_includes_new_helper():
    """tac.bit_allocator.per_byte.__all__ explicitly lists the new helper."""
    from tac.bit_allocator import per_byte

    assert "allocate_per_byte_from_master_gradient_anchor" in per_byte.__all__
