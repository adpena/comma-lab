# SPDX-License-Identifier: MIT
"""Tests for the §7.6 namespace per-pair master gradient wire-in shims.

LOW gap closure widened wave 2026-05-17 — `lane_low_gap_closure_widened_8_modules_
plus_autopilot_wire_in_20260517` BUCKET A. Exercises the solver namespace
shims around the canonical
``tac.optimization.per_pair_namespace_wire_in.compose_namespace_per_pair_wire_in``
helper.

Per CLAUDE.md "Apples-to-apples evidence discipline": every outcome carries
`[predicted; namespace per-pair wire-in v1]` and `score_claim=False`.
"""

from __future__ import annotations

import pytest

from tac.boosting import (
    BoostingPerPairWireInOutcome,
    compose_boosting_per_pair_wire_in,
)
from tac.compress_time_optimization import (
    CompressTimeOptimizationPerPairWireInOutcome,
    compose_compress_time_optimization_per_pair_wire_in,
)
from tac.inflate_time_post_processing import (
    InflateTimePostProcessingPerPairWireInOutcome,
    compose_inflate_time_post_processing_per_pair_wire_in,
)
from tac.optimization.per_pair_namespace_wire_in import (
    LEGAL_NAMESPACE_IDS,
    NAMESPACE_WIRE_IN_SCHEMA,
    NamespacePerPairWireInOutcome,
    compose_namespace_per_pair_wire_in,
)
from tac.search import (
    SearchPerPairWireInOutcome,
    compose_search_per_pair_wire_in,
)
from tac.side_information import (
    SideInformationPerPairWireInOutcome,
    compose_side_information_per_pair_wire_in,
)
from tac.training_curriculum import (
    TrainingCurriculumPerPairWireInOutcome,
    compose_training_curriculum_per_pair_wire_in,
)


# ── Shared helper invariants ──────────────────────────────────────────────────


def test_legal_namespace_ids_pinned():
    """LEGAL_NAMESPACE_IDS is the canonical 6-element frozenset."""
    assert isinstance(LEGAL_NAMESPACE_IDS, frozenset)
    assert LEGAL_NAMESPACE_IDS == frozenset({
        "boosting",
        "compress_time_optimization",
        "inflate_time_post_processing",
        "side_information",
        "search",
        "training_curriculum",
    })


def test_schema_constant_pinned():
    """NAMESPACE_WIRE_IN_SCHEMA is the canonical v1 pinned schema."""
    assert NAMESPACE_WIRE_IN_SCHEMA == "tac_namespace_per_pair_wire_in_v1"


def test_compose_namespace_rejects_invalid_namespace_id():
    with pytest.raises(ValueError, match="namespace_id"):
        compose_namespace_per_pair_wire_in(
            namespace_id="not_a_legal_namespace",
            archive_sha256="deadbeef1234",
            total_bit_budget=100,
            auto_load=False,
        )


def test_compose_namespace_rejects_negative_budget():
    with pytest.raises(ValueError, match="total_bit_budget"):
        compose_namespace_per_pair_wire_in(
            namespace_id="boosting",
            archive_sha256="deadbeef1234",
            total_bit_budget=-1,
            auto_load=False,
        )


def test_compose_namespace_rejects_non_int_budget():
    with pytest.raises(ValueError, match="total_bit_budget"):
        compose_namespace_per_pair_wire_in(
            namespace_id="boosting",
            archive_sha256="deadbeef1234",
            total_bit_budget=10.5,  # type: ignore[arg-type]
            auto_load=False,
        )


def test_compose_namespace_rejects_short_sha():
    with pytest.raises(ValueError, match="archive_sha256"):
        compose_namespace_per_pair_wire_in(
            namespace_id="boosting",
            archive_sha256="short",
            total_bit_budget=100,
            auto_load=False,
        )


def test_compose_namespace_rejects_non_hex_sha():
    with pytest.raises(ValueError, match="archive_sha256"):
        compose_namespace_per_pair_wire_in(
            namespace_id="boosting",
            archive_sha256="GGGGGGGGGGGG",  # 12 chars but not hex
            total_bit_budget=100,
            auto_load=False,
        )


# ── Per-namespace shim contract tests ────────────────────────────────────────


def test_boosting_shim_namespace_id():
    out = compose_boosting_per_pair_wire_in(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        auto_load=False,
    )
    assert isinstance(out, NamespacePerPairWireInOutcome)
    assert out.namespace_id == "boosting"
    assert out.score_claim is False
    assert out.promotion_eligible is False
    assert out.ready_for_exact_eval_dispatch is False
    assert out.evidence_grade == "[predicted; namespace per-pair wire-in v1]"
    assert out.schema == NAMESPACE_WIRE_IN_SCHEMA


def test_compress_time_optimization_shim_namespace_id():
    out = compose_compress_time_optimization_per_pair_wire_in(
        archive_sha256="abcdef0123456789abcdef01",
        total_bit_budget=200,
        auto_load=False,
    )
    assert out.namespace_id == "compress_time_optimization"
    assert out.score_claim is False
    assert "compress_time_optimization" in out.rationale


def test_inflate_time_post_processing_shim_namespace_id():
    out = compose_inflate_time_post_processing_per_pair_wire_in(
        archive_sha256="0123456789abcdef01234567",
        total_bit_budget=64,
        auto_load=False,
    )
    assert out.namespace_id == "inflate_time_post_processing"
    assert out.score_claim is False


def test_side_information_shim_namespace_id():
    out = compose_side_information_per_pair_wire_in(
        archive_sha256="fedcba9876543210fedcba98",
        total_bit_budget=512,
        auto_load=False,
    )
    assert out.namespace_id == "side_information"
    assert out.score_claim is False


def test_search_shim_namespace_id():
    out = compose_search_per_pair_wire_in(
        archive_sha256="1111222233334444aaaabbbb",
        total_bit_budget=128,
        auto_load=False,
    )
    assert out.namespace_id == "search"
    assert out.score_claim is False


def test_training_curriculum_shim_namespace_id():
    out = compose_training_curriculum_per_pair_wire_in(
        archive_sha256="2222333344445555aaaabbbb",
        total_bit_budget=96,
        auto_load=False,
    )
    assert out.namespace_id == "training_curriculum"
    assert out.score_claim is False
    assert "training_curriculum" in out.rationale


# ── Cascade path correctness ─────────────────────────────────────────────────


def test_no_signal_path_when_auto_load_false_and_no_reweight():
    """Auto-load disabled + no reweight → aggregate_fallback (signal still
    reaches Hook 3 via the sister #817 cascade's aggregate-fallback path)."""
    out = compose_boosting_per_pair_wire_in(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        auto_load=False,
    )
    # Hook 3 inherits the sister #817 aggregate_fallback because that helper
    # always allocates uniformly when neither plan nor (gradient+reweight) is
    # supplied.
    assert out.cascade_path_used in ("aggregate_fallback", "no_signal")
    assert out.hook_1_sensitivity_reweight_consumed is False


def test_hook_1_consumed_when_reweight_supplied():
    """Supplying a sensitivity_reweight dict marks Hook 1 as consumed."""
    reweight = {0: 2.0, 1: 0.1, 2: 1.0}
    out = compose_boosting_per_pair_wire_in(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        sensitivity_reweight=reweight,
        auto_load=False,
    )
    assert out.hook_1_sensitivity_reweight_consumed is True


def test_zero_budget_produces_empty_allocation():
    out = compose_boosting_per_pair_wire_in(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=0,
        auto_load=False,
    )
    assert out.total_allocated_bytes == 0
    assert out.per_byte_bit_allocation == {}


# ── Schema + rationale tagging ───────────────────────────────────────────────


def test_rationale_mentions_namespace_id():
    out = compose_search_per_pair_wire_in(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        auto_load=False,
    )
    assert "search" in out.rationale
    assert "[predicted; namespace per-pair wire-in v1]" in out.rationale


def test_rationale_mentions_hook_5_preservation():
    """The rationale text MUST mention Hook 5 preservation via sister
    fcntl-locked store (per CLAUDE.md cargo-cult-audit verdict + spec
    'preserve sister's append_stage_outcome_locked HARD-EARNED pattern')."""
    out = compose_boosting_per_pair_wire_in(
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        auto_load=False,
    )
    # Either "Hook 5 preserved" mentions OR explicit append_*_outcome_locked
    rationale_lower = out.rationale.lower()
    assert "hook 5" in rationale_lower or "sister" in rationale_lower


# ── Per-namespace shim outcome type aliases ──────────────────────────────────


def test_boosting_outcome_type_alias_resolves_to_canonical():
    """The namespace-shim outcome aliases all resolve to the canonical type
    (per CLAUDE.md 'Beauty, simplicity, and developer experience' — typed
    re-export pattern preserves the canonical contract)."""
    assert BoostingPerPairWireInOutcome is NamespacePerPairWireInOutcome
    assert CompressTimeOptimizationPerPairWireInOutcome is NamespacePerPairWireInOutcome
    assert InflateTimePostProcessingPerPairWireInOutcome is NamespacePerPairWireInOutcome
    assert SideInformationPerPairWireInOutcome is NamespacePerPairWireInOutcome
    assert SearchPerPairWireInOutcome is NamespacePerPairWireInOutcome
    assert TrainingCurriculumPerPairWireInOutcome is NamespacePerPairWireInOutcome


# ── All 5 namespaces exhaustively covered ─────────────────────────────────────


@pytest.mark.parametrize(
    "namespace_id",
    sorted(LEGAL_NAMESPACE_IDS),
)
def test_every_namespace_id_produces_valid_outcome(namespace_id: str):
    """Parametrized smoke: every namespace in LEGAL_NAMESPACE_IDS produces a
    valid outcome via the shared canonical helper."""
    out = compose_namespace_per_pair_wire_in(
        namespace_id=namespace_id,
        archive_sha256="deadbeef1234567890abcdef",
        total_bit_budget=100,
        auto_load=False,
    )
    assert out.namespace_id == namespace_id
    assert out.schema == NAMESPACE_WIRE_IN_SCHEMA
    assert out.score_claim is False
    assert out.evidence_grade == "[predicted; namespace per-pair wire-in v1]"
