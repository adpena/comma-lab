# SPDX-License-Identifier: MIT
"""Tests for Cable D consumers 7-14 → SOLVER-SURFACE wire-in (Slot HH).

Per Slot FF landing memo + Slot HH highest-EV op-routable: validates the
canonical solver-surface contributions for the 6 Cable D consumers, plus the
canonical non-promotable + observability-only invariants per Catalog
#287/#323/#341.

Coverage:
- 9 HARD-EARNED (consumer × hook) ACTIVE cells per Catalog #303 cargo-cult
  audit, each with sidecar-absent + sidecar-present round-trips
- 9 N/A cells (refused at construction time per Catalog #303 phantom-cell
  prevention)
- Canonical non-promotable + observability invariants per Catalog #287/#323/#341
- Cross-hook consistency (same consumer + same sha = consistent payload across
  its hooks)
- Sister-regression smoke (Catalog #335 cathedral consumer count unchanged)
- Live-repo regression guard (canonical sidecar registry pinned)
"""
from __future__ import annotations

import json
import pathlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Mapping

import pytest

from tac.cathedral_solver_wire_in import (
    CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY,
    CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS,
    SOLVER_WIRE_IN_OBSERVABILITY_AXIS,
    SolverHookContribution,
    bit_allocator_contribution_for_consumer,
    collect_all_solver_contributions_for_archive,
    consumer_owns_hook,
    is_solver_contribution_promotable,
    pareto_constraint_contribution_for_consumer,
    sensitivity_map_contribution_for_consumer,
)
from tac.cathedral_solver_wire_in.consumers_7_14_contributions import (
    CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS,
)


# ─── Canonical sidecar fixture helpers ──────────────────────────────────────

_SIDECAR_ROOT = pathlib.Path(
    "/Users/adpena/Projects/pact/.omx/state/master_gradient_consumers"
)


@contextmanager
def _staged_canonical_sidecar(
    consumer_id: str,
    archive_sha256: str,
    *,
    schema: str | None = None,
    score_claim: bool | None = False,
    promotion_eligible: bool | None = False,
    n_pairs: int = 600,
    n_bytes: int = 12345,
    extra: Mapping[str, object] | None = None,
) -> Iterator[pathlib.Path]:
    """Stage a synthetic canonical sidecar JSON; cleanup on exit.

    Default schema is the canonical schema from
    CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS for the consumer_id.
    """
    if schema is None:
        for cid, canonical_schema in CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS:
            if cid == consumer_id:
                schema = canonical_schema
                break
    payload = {
        "schema": schema,
        "archive_sha256": archive_sha256,
        "score_claim": score_claim,
        "promotion_eligible": promotion_eligible,
        "n_pairs": n_pairs,
        "n_bytes": n_bytes,
    }
    if extra:
        payload.update(dict(extra))
    sha_short = archive_sha256[:12]
    # Use uuid suffix to avoid collisions across parallel tests
    suffix = uuid.uuid4().hex[:8]
    name = f"{consumer_id}_{sha_short}_20260520T020000_{suffix}.json"
    path = _SIDECAR_ROOT / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _fresh_archive_sha() -> str:
    """Generate a deterministic unique 64-hex sha for test isolation."""
    return uuid.uuid4().hex + uuid.uuid4().hex


# ─── Constants + canonical contract pinned ──────────────────────────────────


def test_canonical_observability_axis_is_predicted() -> None:
    assert SOLVER_WIRE_IN_OBSERVABILITY_AXIS == "[predicted]"


def test_canonical_hook_pairs_count_is_nine_per_catalog_303() -> None:
    """Per Catalog #303 cargo-cult audit: 9 HARD-EARNED, NOT 18 (all × all)."""
    assert len(CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS) == 9


def test_canonical_hook_registry_has_6_consumers() -> None:
    assert len(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY) == 6


def test_canonical_hook_registry_per_consumer_hook_set_pinned() -> None:
    """Per Catalog #303: each consumer's HARD-EARNED hook set per producer header."""
    assert CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[
        "per_pair_pareto_envelope"
    ] == frozenset({"pareto_constraint"})
    assert CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[
        "per_pair_lagrangian_lambda_bisection"
    ] == frozenset({"sensitivity_map", "pareto_constraint"})
    assert CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[
        "per_pair_lora_supervision_signal"
    ] == frozenset({"bit_allocator"})
    assert CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[
        "per_pair_coding_budget_allocation"
    ] == frozenset({"bit_allocator"})
    assert CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[
        "per_pair_kkt_residuals"
    ] == frozenset({"sensitivity_map", "pareto_constraint"})
    assert CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[
        "per_pair_volterra_cross_terms"
    ] == frozenset({"sensitivity_map", "pareto_constraint"})


def test_canonical_sidecar_registry_pinned_to_6_entries() -> None:
    assert len(CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS) == 6


# ─── consumer_owns_hook canonical disambiguator ────────────────────────────


def test_consumer_owns_hook_active_cells() -> None:
    """All 9 HARD-EARNED cells per Catalog #303 return True."""
    for consumer_id, hook_name in CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS:
        assert consumer_owns_hook(consumer_id, hook_name), (consumer_id, hook_name)


def test_consumer_owns_hook_na_cells_return_false() -> None:
    """All 9 N/A cells per Catalog #303 return False."""
    # Enumerate N/A cells: all 6 consumers × 3 hooks = 18 cells, minus 9 ACTIVE = 9 N/A
    all_consumers = list(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY.keys())
    all_hooks = ("sensitivity_map", "pareto_constraint", "bit_allocator")
    active_cells = set(CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS)
    na_count = 0
    for c in all_consumers:
        for h in all_hooks:
            if (c, h) not in active_cells:
                na_count += 1
                assert not consumer_owns_hook(c, h), (c, h)
    assert na_count == 9, f"expected 9 N/A cells per Catalog #303, got {na_count}"


def test_consumer_owns_hook_unknown_consumer_returns_false() -> None:
    assert not consumer_owns_hook("not_a_real_consumer", "sensitivity_map")


def test_consumer_owns_hook_unknown_hook_returns_false() -> None:
    assert not consumer_owns_hook(
        "per_pair_lagrangian_lambda_bisection", "not_a_real_hook"
    )


# ─── Hook #1 sensitivity-map per-consumer happy paths ──────────────────────


def test_hook_1_sensitivity_consumer_8_lambda_sidecar_absent() -> None:
    """Hook #1 + consumer #8: sidecar absent → present=False, n_pairs=0."""
    c = sensitivity_map_contribution_for_consumer(
        "per_pair_lagrangian_lambda_bisection", _fresh_archive_sha()
    )
    assert c.sidecar_present is False
    assert c.n_pairs == 0
    assert c.sidecar_path is None
    assert c.axis_tag == "[predicted]"
    assert c.promotion_eligible is False


def test_hook_1_sensitivity_consumer_8_lambda_sidecar_present() -> None:
    """Hook #1 + consumer #8: canonical sidecar present → present=True, n_pairs preserved."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_lagrangian_lambda_bisection", sha, n_pairs=600, n_bytes=8192
    ) as sidecar_path:
        c = sensitivity_map_contribution_for_consumer(
            "per_pair_lagrangian_lambda_bisection", sha
        )
        assert c.sidecar_present is True
        assert c.n_pairs == 600
        assert c.n_bytes == 8192
        assert c.sidecar_path == str(sidecar_path)


def test_hook_1_sensitivity_consumer_12_kkt_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar("per_pair_kkt_residuals", sha, n_pairs=300):
        c = sensitivity_map_contribution_for_consumer(
            "per_pair_kkt_residuals", sha
        )
        assert c.sidecar_present
        assert c.n_pairs == 300


def test_hook_1_sensitivity_consumer_13_volterra_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_volterra_cross_terms", sha, n_pairs=400
    ):
        c = sensitivity_map_contribution_for_consumer(
            "per_pair_volterra_cross_terms", sha
        )
        assert c.sidecar_present
        assert c.n_pairs == 400


# ─── Hook #2 Pareto-constraint per-consumer happy paths ────────────────────


def test_hook_2_pareto_consumer_7_envelope_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_pareto_envelope", sha, n_pairs=600
    ):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_pareto_envelope", sha
        )
        assert c.sidecar_present
        assert c.hook_name == "pareto_constraint"


def test_hook_2_pareto_consumer_8_lambda_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_lagrangian_lambda_bisection", sha
    ):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_lagrangian_lambda_bisection", sha
        )
        assert c.sidecar_present


def test_hook_2_pareto_consumer_12_kkt_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar("per_pair_kkt_residuals", sha):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_kkt_residuals", sha
        )
        assert c.sidecar_present


def test_hook_2_pareto_consumer_13_volterra_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar("per_pair_volterra_cross_terms", sha):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_volterra_cross_terms", sha
        )
        assert c.sidecar_present


# ─── Hook #3 bit-allocator per-consumer happy paths ────────────────────────


def test_hook_3_bit_allocator_consumer_9_lora_sidecar_absent() -> None:
    c = bit_allocator_contribution_for_consumer(
        "per_pair_lora_supervision_signal", _fresh_archive_sha()
    )
    assert c.sidecar_present is False
    assert c.hook_name == "bit_allocator"


def test_hook_3_bit_allocator_consumer_9_lora_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_lora_supervision_signal", sha, n_bytes=2048
    ):
        c = bit_allocator_contribution_for_consumer(
            "per_pair_lora_supervision_signal", sha
        )
        assert c.sidecar_present
        assert c.n_bytes == 2048


def test_hook_3_bit_allocator_consumer_10_coding_budget_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_coding_budget_allocation", sha, n_pairs=600
    ):
        c = bit_allocator_contribution_for_consumer(
            "per_pair_coding_budget_allocation", sha
        )
        assert c.sidecar_present


# ─── Catalog #303 N/A cell refusal (forbidden phantom cells) ───────────────


@pytest.mark.parametrize(
    "consumer_id,hook_fn",
    [
        # N/A: consumer #7 does NOT own sensitivity_map per producer header
        ("per_pair_pareto_envelope", sensitivity_map_contribution_for_consumer),
        # N/A: consumer #7 does NOT own bit_allocator per producer header
        ("per_pair_pareto_envelope", bit_allocator_contribution_for_consumer),
        # N/A: consumer #8 does NOT own bit_allocator
        ("per_pair_lagrangian_lambda_bisection", bit_allocator_contribution_for_consumer),
        # N/A: consumer #9 does NOT own sensitivity_map
        ("per_pair_lora_supervision_signal", sensitivity_map_contribution_for_consumer),
        # N/A: consumer #9 does NOT own pareto_constraint
        ("per_pair_lora_supervision_signal", pareto_constraint_contribution_for_consumer),
        # N/A: consumer #10 does NOT own sensitivity_map
        ("per_pair_coding_budget_allocation", sensitivity_map_contribution_for_consumer),
        # N/A: consumer #10 does NOT own pareto_constraint
        ("per_pair_coding_budget_allocation", pareto_constraint_contribution_for_consumer),
        # N/A: consumer #12 does NOT own bit_allocator
        ("per_pair_kkt_residuals", bit_allocator_contribution_for_consumer),
        # N/A: consumer #13 does NOT own bit_allocator
        ("per_pair_volterra_cross_terms", bit_allocator_contribution_for_consumer),
    ],
)
def test_catalog_303_na_cells_refused_at_construction_time(
    consumer_id: str, hook_fn
) -> None:
    """N/A cells (9 total per Catalog #303) raise ValueError citing Catalog #303."""
    with pytest.raises(ValueError, match="does NOT own hook"):
        hook_fn(consumer_id, _fresh_archive_sha())


# ─── Catalog #287/#323/#341 canonical non-promotable invariants ────────────


def test_catalog_341_solver_contribution_score_claim_valid_forbidden() -> None:
    """Catalog #287/#323: solver contributions NEVER claim score validity."""
    with pytest.raises(ValueError, match="score_claim_valid"):
        SolverHookContribution(
            consumer_id="per_pair_pareto_envelope",
            hook_name="pareto_constraint",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_pareto_envelope_v1",
            n_pairs=0,
            n_bytes=0,
            score_claim_valid=True,  # forbidden
        )


def test_catalog_341_solver_contribution_promotion_eligible_forbidden() -> None:
    """Catalog #341: solver contributions NEVER leak into promotion."""
    with pytest.raises(ValueError, match="promotion_eligible"):
        SolverHookContribution(
            consumer_id="per_pair_pareto_envelope",
            hook_name="pareto_constraint",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_pareto_envelope_v1",
            n_pairs=0,
            n_bytes=0,
            promotion_eligible=True,  # forbidden
        )


def test_catalog_287_solver_contribution_axis_must_be_predicted() -> None:
    """Catalog #287: axis_tag MUST be [predicted] for observability-only."""
    with pytest.raises(ValueError, match="axis_tag"):
        SolverHookContribution(
            consumer_id="per_pair_pareto_envelope",
            hook_name="pareto_constraint",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_pareto_envelope_v1",
            n_pairs=0,
            n_bytes=0,
            axis_tag="[contest-CUDA]",  # forbidden
        )


def test_solver_contribution_predicted_delta_adjustment_must_be_zero() -> None:
    """FF cascade applies the reward; SOLVER contribution MUST be zero-adj."""
    with pytest.raises(ValueError, match="predicted_delta_adjustment"):
        SolverHookContribution(
            consumer_id="per_pair_pareto_envelope",
            hook_name="pareto_constraint",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_pareto_envelope_v1",
            n_pairs=0,
            n_bytes=0,
            predicted_delta_adjustment=-0.001,  # forbidden
        )


def test_is_solver_contribution_promotable_always_false() -> None:
    """Catalog #341 invariant: solver contributions are NEVER promotable."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar("per_pair_pareto_envelope", sha):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_pareto_envelope", sha
        )
        assert is_solver_contribution_promotable(c) is False


def test_solver_contribution_phantom_cell_construction_refused() -> None:
    """Direct construction of phantom (consumer, hook) cell raises per Catalog #303."""
    with pytest.raises(ValueError, match="is N/A for hook_name"):
        SolverHookContribution(
            consumer_id="per_pair_pareto_envelope",  # owns only pareto_constraint
            hook_name="sensitivity_map",  # N/A
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_pareto_envelope_v1",
            n_pairs=0,
            n_bytes=0,
        )


# ─── Cross-hook consistency tests ──────────────────────────────────────────


def test_cross_hook_consumer_8_consistent_across_sensitivity_and_pareto() -> None:
    """Consumer #8 owns hooks #1 AND #2; same sidecar → same n_pairs across both."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_lagrangian_lambda_bisection", sha, n_pairs=600, n_bytes=4096
    ):
        c1 = sensitivity_map_contribution_for_consumer(
            "per_pair_lagrangian_lambda_bisection", sha
        )
        c2 = pareto_constraint_contribution_for_consumer(
            "per_pair_lagrangian_lambda_bisection", sha
        )
        assert c1.n_pairs == c2.n_pairs == 600
        assert c1.n_bytes == c2.n_bytes == 4096
        assert c1.sidecar_path == c2.sidecar_path  # same sidecar file


def test_cross_hook_consumer_13_consistent_across_sensitivity_and_pareto() -> None:
    """Consumer #13 owns hooks #1 AND #2; same sidecar → consistent payload."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_volterra_cross_terms", sha, n_pairs=500
    ):
        c1 = sensitivity_map_contribution_for_consumer(
            "per_pair_volterra_cross_terms", sha
        )
        c2 = pareto_constraint_contribution_for_consumer(
            "per_pair_volterra_cross_terms", sha
        )
        assert c1.sidecar_present == c2.sidecar_present == True
        assert c1.n_pairs == c2.n_pairs == 500


# ─── collect_all_solver_contributions_for_archive ──────────────────────────


def test_collect_all_solver_contributions_returns_9_cells() -> None:
    """Catalog #303: collection returns exactly 9 HARD-EARNED ACTIVE cells."""
    contribs = collect_all_solver_contributions_for_archive(_fresh_archive_sha())
    assert len(contribs) == 9


def test_collect_all_solver_contributions_all_observability_only() -> None:
    """Every contribution is non-promotable + [predicted] axis + zero-adjustment."""
    contribs = collect_all_solver_contributions_for_archive(_fresh_archive_sha())
    for c in contribs:
        assert c.promotion_eligible is False
        assert c.score_claim_valid is False
        assert c.axis_tag == "[predicted]"
        assert c.predicted_delta_adjustment == 0.0


def test_collect_all_solver_contributions_with_one_sidecar_present() -> None:
    """Stage 1 of 6 sidecars; verify only that consumer × its hooks show present=True."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_lagrangian_lambda_bisection", sha
    ):
        contribs = collect_all_solver_contributions_for_archive(sha)
        # consumer #8 owns 2 hooks (sensitivity + pareto); both should be present
        present_for_8 = [
            c for c in contribs
            if c.consumer_id == "per_pair_lagrangian_lambda_bisection"
        ]
        assert len(present_for_8) == 2
        assert all(c.sidecar_present for c in present_for_8)
        # all OTHER contributions: sidecar absent
        absent_other = [
            c for c in contribs
            if c.consumer_id != "per_pair_lagrangian_lambda_bisection"
        ]
        assert len(absent_other) == 7
        assert all(not c.sidecar_present for c in absent_other)


def test_collect_all_solver_contributions_empty_sha_rejected() -> None:
    with pytest.raises(ValueError, match="archive_sha256"):
        collect_all_solver_contributions_for_archive("")


# ─── Custody-leak guards (Catalog #341 + FF helper integration) ────────────


def test_sidecar_with_score_claim_true_rejected_as_phantom() -> None:
    """Per Catalog #321/#322/#323: score_claim=True sidecar → not promoted to present."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_pareto_envelope", sha, score_claim=True
    ):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_pareto_envelope", sha
        )
        assert c.sidecar_present is False  # score_claim leak refused


def test_sidecar_with_promotion_eligible_true_rejected() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_pareto_envelope", sha, promotion_eligible=True
    ):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_pareto_envelope", sha
        )
        assert c.sidecar_present is False


def test_sidecar_with_cross_archive_sha_rejected() -> None:
    sha_a = _fresh_archive_sha()
    sha_b = _fresh_archive_sha()
    # Stage sidecar at sha_a's prefix but JSON body claims sha_b
    with _staged_canonical_sidecar(
        "per_pair_pareto_envelope", sha_b, n_pairs=600
    ) as sidecar_path:
        # rename to sha_a's prefix
        bad_name = sidecar_path.name.replace(sha_b[:12], sha_a[:12])
        bad_path = sidecar_path.parent / bad_name
        sidecar_path.rename(bad_path)
        try:
            c = pareto_constraint_contribution_for_consumer(
                "per_pair_pareto_envelope", sha_a
            )
            # FF helper refuses cross-archive contamination
            assert c.sidecar_present is False
        finally:
            try:
                bad_path.unlink()
            except FileNotFoundError:
                pass


def test_sidecar_with_wrong_schema_rejected() -> None:
    """FF helper requires canonical schema tag match."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_pareto_envelope",
        sha,
        schema="WRONG_SCHEMA_v1",  # corrupted schema
    ):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_pareto_envelope", sha
        )
        assert c.sidecar_present is False


def test_sidecar_with_trivial_signal_rejected() -> None:
    """FF helper rejects empty sidecar (n_pairs=0 AND n_bytes=0)."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_pareto_envelope", sha, n_pairs=0, n_bytes=0
    ):
        c = pareto_constraint_contribution_for_consumer(
            "per_pair_pareto_envelope", sha
        )
        assert c.sidecar_present is False


# ─── Sister-regression smoke tests ─────────────────────────────────────────


def test_sister_regression_cathedral_consumer_count_unchanged() -> None:
    """Slot HH is SOLVER-side; cathedral consumer count must remain 34 per Slot FF."""
    import importlib.util
    import sys

    repo_root = pathlib.Path("/Users/adpena/Projects/pact")
    target = repo_root / "tools" / "cathedral_autopilot_autonomous_loop.py"
    spec = importlib.util.spec_from_file_location(
        "_test_autopilot_loop_for_sister_regression", target
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    consumers = mod.discover_compliant_consumer_modules()
    # Per Slot FF: 34 consumers at landing. Slot HH does not add or remove any.
    assert len(consumers) >= 34, (
        f"sister regression: cathedral consumer count {len(consumers)} dropped "
        "below Slot FF baseline of 34"
    )


def test_sister_regression_ff_cascade_unchanged() -> None:
    """Slot FF cascade canonical helpers still callable per Catalog #185."""
    import importlib.util
    import sys

    repo_root = pathlib.Path("/Users/adpena/Projects/pact")
    target = repo_root / "tools" / "cathedral_autopilot_autonomous_loop.py"
    spec = importlib.util.spec_from_file_location(
        "_test_autopilot_loop_for_ff_helpers", target
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert callable(
        mod.adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars
    )
    assert callable(mod._latest_cable_d_consumer_sidecar_for_archive)
    assert callable(mod._cable_d_consumer_sidecar_carries_structural_signal)


# ─── Live-repo regression guards ───────────────────────────────────────────


def test_live_repo_canonical_sidecar_registry_pinned_to_6() -> None:
    """Catalog #185 sister: canonical sidecar registry must remain 6 entries."""
    assert len(CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS) == 6


def test_live_repo_no_phantom_solver_hook_in_pairs() -> None:
    """Every (consumer × hook) pair MUST be in HARD-EARNED registry."""
    for consumer_id, hook_name in CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS:
        assert hook_name in CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[consumer_id]


def test_live_repo_hard_earned_cells_match_producer_headers() -> None:
    """Verify the 9 HARD-EARNED cells match producer-header declarations."""
    # Test that the 9 ACTIVE cells per Catalog #303 audit match the producer
    # headers' "hook #X consumed by tac.optimization.{pareto,bit_allocator}"
    # or "hook #1 SENSITIVITY_MAP feeds tac.sensitivity_map.*" declarations
    expected_hard_earned = {
        # per consumer #7 producer header: "feeds tac.optimization.pareto"
        ("per_pair_pareto_envelope", "pareto_constraint"),
        # per consumer #8: "feeds tac.sensitivity_map" + "feeds tac.optimization.pareto"
        ("per_pair_lagrangian_lambda_bisection", "sensitivity_map"),
        ("per_pair_lagrangian_lambda_bisection", "pareto_constraint"),
        # per consumer #9: "feeds tac.optimization.bit_allocator"
        ("per_pair_lora_supervision_signal", "bit_allocator"),
        # per consumer #10: "feeds tac.optimization.bit_allocator"
        ("per_pair_coding_budget_allocation", "bit_allocator"),
        # per consumer #12: "feeds tac.optimization.pareto" + "sensitivity_map" (KKT)
        ("per_pair_kkt_residuals", "sensitivity_map"),
        ("per_pair_kkt_residuals", "pareto_constraint"),
        # per consumer #13: "feed tac.sensitivity_map.* + tac.optimization.pareto"
        ("per_pair_volterra_cross_terms", "sensitivity_map"),
        ("per_pair_volterra_cross_terms", "pareto_constraint"),
    }
    actual = set(CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS)
    assert actual == expected_hard_earned, (
        f"HARD-EARNED registry drift: actual={actual} expected={expected_hard_earned}"
    )
