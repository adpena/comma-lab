# SPDX-License-Identifier: MIT
"""Tests for Cable D consumers 7-14 → hooks #5 + #6 FULL Catalog #125 closure.

Slot LL 2026-05-20 — closes the producer → sidecar → ranker (FF) → solver
(HH) → posterior (#5) → disambiguator (#6) FULL 6-hook loop.

Per Catalog #303 cargo-cult audit per assumption: tests assert the
**HARD-EARNED matrix** per producer-header ``CONSUMER_HOOK_NUMBERS``
declarations:

- Hook #5: 1 ACTIVE (consumer #9 ``per_pair_lora_supervision_signal``) + 5 N/A
- Hook #6: 1 ACTIVE (consumer #12 ``per_pair_kkt_residuals``) + 5 N/A

CARGO-CULTED "all × all" cells (10 total) are refused at construction time.

Coverage:
- 2 HARD-EARNED ACTIVE cells (canonical marker + invariant + happy paths)
- 10 N/A cells (refused per Catalog #303 phantom-cell prevention)
- Canonical non-promotable + observability invariants per Catalog #287/#323/#341
- Cross-hook consistency across 6-hook full closure (consumers in hooks #1-#6)
- Sister-regression smoke (HH 51 + FF cascade + consumer contract + Catalog
  #335/#341 + auto-discovery — all still pass)
- Live-repo regression guards (cathedral consumer count unchanged at 34;
  HOOK_5/6_ACTIVE pinned per producer-header declarations)
- Catalog #313 sister probe-outcomes ledger integration (verdict taxonomy
  alignment)
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
    # HH surfaces (must still be importable post-LL landing)
    CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY,
    CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS,
    SOLVER_WIRE_IN_OBSERVABILITY_AXIS,
    SolverHookContribution,
    consumer_owns_hook,
    # LL hook #5 surfaces
    CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE,
    CABLE_D_CONSUMERS_7_14_HOOK_5_PAIRS,
    Hook5PosteriorAnchor,
    collect_all_hook_5_anchors_for_archive,
    consumer_owns_hook_5,
    is_hook_5_anchor_promotable,
    query_posterior_for_consumer,
    # LL hook #6 surfaces
    CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE,
    CABLE_D_CONSUMERS_7_14_HOOK_6_PAIRS,
    DEFAULT_DISAMBIGUATOR_TOP_N,
    HOOK_6_DEFAULT_VERDICT,
    Hook6DisambiguatorVerdict,
    collect_all_hook_6_verdicts_for_archive,
    consumer_owns_hook_6,
    disambiguate_per_pair_stationarity,
    is_hook_6_verdict_promotable,
)
from tac.cathedral_solver_wire_in.consumers_7_14_contributions import (
    CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS,
)


# ─── Canonical sidecar fixture helpers (mirror HH pattern) ──────────────────

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
    """Stage a synthetic canonical sidecar JSON; cleanup on exit."""
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
    suffix = uuid.uuid4().hex[:8]
    name = f"{consumer_id}_{sha_short}_20260520T021500_{suffix}.json"
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
    return uuid.uuid4().hex + uuid.uuid4().hex


# ─── Hook #5: Catalog #303 HARD-EARNED matrix pinned ────────────────────────


def test_hook_5_active_set_is_one_consumer_per_catalog_303() -> None:
    """Per Catalog #303 + producer-header CONSUMER_HOOK_NUMBERS declarations:
    ONLY consumer #9 (per_pair_lora_supervision_signal) declares
    HookNumber.CONTINUAL_LEARNING_POSTERIOR.
    """
    assert CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE == frozenset(
        {"per_pair_lora_supervision_signal"}
    )


def test_hook_5_pairs_count_is_one() -> None:
    assert len(CABLE_D_CONSUMERS_7_14_HOOK_5_PAIRS) == 1


def test_hook_5_pairs_canonical_form() -> None:
    assert CABLE_D_CONSUMERS_7_14_HOOK_5_PAIRS == (
        ("per_pair_lora_supervision_signal", "continual_learning_posterior"),
    )


def test_consumer_owns_hook_5_active_returns_true() -> None:
    assert consumer_owns_hook_5("per_pair_lora_supervision_signal") is True


@pytest.mark.parametrize(
    "consumer_id",
    [
        "per_pair_pareto_envelope",
        "per_pair_lagrangian_lambda_bisection",
        "per_pair_coding_budget_allocation",
        "per_pair_kkt_residuals",
        "per_pair_volterra_cross_terms",
    ],
)
def test_consumer_owns_hook_5_na_returns_false(consumer_id: str) -> None:
    """The 5 N/A consumers per producer-header CONSUMER_HOOK_NUMBERS NO-OP."""
    assert consumer_owns_hook_5(consumer_id) is False


def test_consumer_owns_hook_5_unknown_consumer_returns_false() -> None:
    assert consumer_owns_hook_5("not_a_real_consumer") is False


# ─── Hook #5: happy path (consumer #9 LoRA, ACTIVE) ─────────────────────────


def test_hook_5_consumer_9_lora_sidecar_absent_returns_empty_anchor() -> None:
    """No sidecar → present=False, n_pairs=0, sidecar_sha256=None."""
    anchor = query_posterior_for_consumer(
        "per_pair_lora_supervision_signal", _fresh_archive_sha()
    )
    assert isinstance(anchor, Hook5PosteriorAnchor)
    assert anchor.sidecar_present is False
    assert anchor.n_pairs == 0
    assert anchor.n_bytes == 0
    assert anchor.sidecar_path is None
    assert anchor.sidecar_sha256 is None
    assert anchor.axis_tag == "[predicted]"
    assert anchor.promotion_eligible is False
    assert anchor.score_claim_valid is False


def test_hook_5_consumer_9_lora_sidecar_present_carries_payload() -> None:
    """Sidecar present + schema-valid → present=True, n_pairs+n_bytes preserved."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_lora_supervision_signal", sha, n_pairs=300, n_bytes=4096
    ) as sidecar_path:
        anchor = query_posterior_for_consumer(
            "per_pair_lora_supervision_signal", sha
        )
        assert anchor.sidecar_present is True
        assert anchor.n_pairs == 300
        assert anchor.n_bytes == 4096
        assert anchor.sidecar_path == str(sidecar_path)
        # SHA-256 of sidecar is captured for cite-chain per Catalog #287/#323
        assert anchor.sidecar_sha256 is not None
        assert len(anchor.sidecar_sha256) == 64


def test_hook_5_consumer_9_rationale_cites_producer_header_declaration() -> None:
    """Rationale documents the canonical CONSUMER_HOOK_NUMBERS declaration."""
    anchor = query_posterior_for_consumer(
        "per_pair_lora_supervision_signal", _fresh_archive_sha()
    )
    assert "CONTINUAL_LEARNING_POSTERIOR" in anchor.rationale
    assert "[predicted]" in anchor.rationale


# ─── Hook #5: Catalog #303 N/A cells refused at query time ──────────────────


@pytest.mark.parametrize(
    "consumer_id",
    [
        "per_pair_pareto_envelope",
        "per_pair_lagrangian_lambda_bisection",
        "per_pair_coding_budget_allocation",
        "per_pair_kkt_residuals",
        "per_pair_volterra_cross_terms",
    ],
)
def test_hook_5_query_for_na_consumer_raises_per_catalog_303(
    consumer_id: str,
) -> None:
    """N/A consumers per producer-header CONSUMER_HOOK_NUMBERS NO-OP raise."""
    with pytest.raises(ValueError, match="does NOT own hook #5"):
        query_posterior_for_consumer(consumer_id, _fresh_archive_sha())


def test_hook_5_anchor_construction_for_na_consumer_refused() -> None:
    """Direct Hook5PosteriorAnchor construction for N/A consumer raises."""
    with pytest.raises(ValueError, match="is N/A for hook #5"):
        Hook5PosteriorAnchor(
            consumer_id="per_pair_pareto_envelope",  # N/A per CONSUMER_HOOK_NUMBERS
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_pareto_envelope_v1",
            sidecar_sha256=None,
            n_pairs=0,
            n_bytes=0,
        )


def test_hook_5_query_empty_archive_sha_rejected() -> None:
    with pytest.raises(ValueError, match="archive_sha256"):
        query_posterior_for_consumer("per_pair_lora_supervision_signal", "")


# ─── Hook #5: canonical non-promotable invariants per Catalog #287/#323/#341 ─


def test_hook_5_anchor_score_claim_valid_forbidden() -> None:
    """Catalog #287/#323: anchors NEVER claim score validity."""
    with pytest.raises(ValueError, match="score_claim_valid"):
        Hook5PosteriorAnchor(
            consumer_id="per_pair_lora_supervision_signal",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_lora_supervision_v1",
            sidecar_sha256=None,
            n_pairs=0,
            n_bytes=0,
            score_claim_valid=True,  # forbidden
        )


def test_hook_5_anchor_promotion_eligible_forbidden() -> None:
    """Catalog #341: anchors NEVER leak into promotion."""
    with pytest.raises(ValueError, match="promotion_eligible"):
        Hook5PosteriorAnchor(
            consumer_id="per_pair_lora_supervision_signal",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_lora_supervision_v1",
            sidecar_sha256=None,
            n_pairs=0,
            n_bytes=0,
            promotion_eligible=True,  # forbidden
        )


def test_hook_5_anchor_axis_must_be_predicted() -> None:
    """Catalog #287: axis_tag MUST be [predicted]."""
    with pytest.raises(ValueError, match="axis_tag"):
        Hook5PosteriorAnchor(
            consumer_id="per_pair_lora_supervision_signal",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_lora_supervision_v1",
            sidecar_sha256=None,
            n_pairs=0,
            n_bytes=0,
            axis_tag="[contest-CUDA]",  # forbidden
        )


def test_is_hook_5_anchor_promotable_always_false() -> None:
    """Catalog #341 invariant: never promotable."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar("per_pair_lora_supervision_signal", sha):
        anchor = query_posterior_for_consumer(
            "per_pair_lora_supervision_signal", sha
        )
        assert is_hook_5_anchor_promotable(anchor) is False


def test_hook_5_anchor_provenance_canonical_helper_pinned() -> None:
    """Per Catalog #287/#323: canonical Provenance citation is pinned."""
    anchor = query_posterior_for_consumer(
        "per_pair_lora_supervision_signal", _fresh_archive_sha()
    )
    assert (
        anchor.provenance_canonical_helper
        == "tac.cathedral_solver_wire_in.hook5_continual_learning"
    )


# ─── Hook #5: collection helper ─────────────────────────────────────────────


def test_collect_all_hook_5_anchors_returns_one_per_catalog_303() -> None:
    """Per Catalog #303: collection returns exactly 1 HARD-EARNED ACTIVE cell."""
    anchors = collect_all_hook_5_anchors_for_archive(_fresh_archive_sha())
    assert len(anchors) == 1
    assert anchors[0].consumer_id == "per_pair_lora_supervision_signal"


def test_collect_all_hook_5_anchors_all_observability_only() -> None:
    anchors = collect_all_hook_5_anchors_for_archive(_fresh_archive_sha())
    for a in anchors:
        assert a.promotion_eligible is False
        assert a.score_claim_valid is False
        assert a.axis_tag == "[predicted]"


def test_collect_all_hook_5_anchors_with_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_lora_supervision_signal", sha, n_pairs=600
    ):
        anchors = collect_all_hook_5_anchors_for_archive(sha)
        assert len(anchors) == 1
        assert anchors[0].sidecar_present is True
        assert anchors[0].n_pairs == 600


def test_collect_all_hook_5_anchors_empty_sha_rejected() -> None:
    with pytest.raises(ValueError, match="archive_sha256"):
        collect_all_hook_5_anchors_for_archive("")


# ─── Hook #6: Catalog #303 HARD-EARNED matrix pinned ────────────────────────


def test_hook_6_active_set_is_one_consumer_per_catalog_303() -> None:
    """Per Catalog #303 + producer-header CONSUMER_HOOK_NUMBERS declarations:
    ONLY consumer #12 (per_pair_kkt_residuals) declares
    HookNumber.PROBE_DISAMBIGUATOR.
    """
    assert CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE == frozenset(
        {"per_pair_kkt_residuals"}
    )


def test_hook_6_pairs_count_is_one() -> None:
    assert len(CABLE_D_CONSUMERS_7_14_HOOK_6_PAIRS) == 1


def test_hook_6_pairs_canonical_form() -> None:
    assert CABLE_D_CONSUMERS_7_14_HOOK_6_PAIRS == (
        ("per_pair_kkt_residuals", "probe_disambiguator"),
    )


def test_consumer_owns_hook_6_active_returns_true() -> None:
    assert consumer_owns_hook_6("per_pair_kkt_residuals") is True


@pytest.mark.parametrize(
    "consumer_id",
    [
        "per_pair_pareto_envelope",
        "per_pair_lagrangian_lambda_bisection",
        "per_pair_lora_supervision_signal",
        "per_pair_coding_budget_allocation",
        "per_pair_volterra_cross_terms",
    ],
)
def test_consumer_owns_hook_6_na_returns_false(consumer_id: str) -> None:
    """The 5 N/A consumers per CONSUMER_HOOK_NUMBERS lacking PROBE_DISAMBIGUATOR."""
    assert consumer_owns_hook_6(consumer_id) is False


def test_consumer_owns_hook_6_unknown_consumer_returns_false() -> None:
    assert consumer_owns_hook_6("not_a_real_consumer") is False


# ─── Hook #6: happy path (consumer #12 KKT residuals, ACTIVE) ───────────────


def test_hook_6_consumer_12_kkt_sidecar_absent_returns_empty_verdict() -> None:
    verdict = disambiguate_per_pair_stationarity(_fresh_archive_sha())
    assert isinstance(verdict, Hook6DisambiguatorVerdict)
    assert verdict.sidecar_present is False
    assert verdict.n_pairs == 0
    assert verdict.top_pair_indices == ()
    assert verdict.top_pair_residual_magnitudes == ()
    assert verdict.axis_tag == "[predicted]"
    assert verdict.promotion_eligible is False
    assert verdict.verdict == "PARTIAL"


def test_hook_6_consumer_12_kkt_sidecar_present_extracts_top_n_rank() -> None:
    """Sidecar with per_pair_residual_magnitudes → top-N rank extracted."""
    sha = _fresh_archive_sha()
    residuals = [0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.8, 0.05]
    with _staged_canonical_sidecar(
        "per_pair_kkt_residuals",
        sha,
        n_pairs=10,
        extra={"per_pair_residual_magnitudes": residuals},
    ) as sidecar_path:
        verdict = disambiguate_per_pair_stationarity(sha, top_n=3)
        assert verdict.sidecar_present is True
        assert verdict.n_pairs == 10
        assert verdict.top_n_requested == 3
        assert verdict.sidecar_path == str(sidecar_path)
        # Top 3 by magnitude: 0.9 (idx 3), 0.8 (idx 8), 0.7 (idx 5)
        assert verdict.top_pair_indices == (3, 8, 5)
        assert verdict.top_pair_residual_magnitudes == pytest.approx(
            (0.9, 0.8, 0.7)
        )


def test_hook_6_consumer_12_kkt_alternate_payload_key_residual_magnitudes() -> None:
    """Sidecar with alternate key ``residual_magnitudes`` also recognized."""
    sha = _fresh_archive_sha()
    residuals = [3.0, 1.0, 2.0]
    with _staged_canonical_sidecar(
        "per_pair_kkt_residuals",
        sha,
        n_pairs=3,
        extra={"residual_magnitudes": residuals},
    ):
        verdict = disambiguate_per_pair_stationarity(sha, top_n=2)
        assert verdict.sidecar_present is True
        assert verdict.top_pair_indices == (0, 2)
        assert verdict.top_pair_residual_magnitudes == pytest.approx((3.0, 2.0))


def test_hook_6_consumer_12_kkt_signed_residuals_take_abs() -> None:
    """``per_pair_kkt_residual`` signed payload → absolute value taken."""
    sha = _fresh_archive_sha()
    signed = [-2.5, 1.0, -0.5, 3.0]
    with _staged_canonical_sidecar(
        "per_pair_kkt_residuals",
        sha,
        n_pairs=4,
        extra={"per_pair_kkt_residual": signed},
    ):
        verdict = disambiguate_per_pair_stationarity(sha, top_n=2)
        assert verdict.sidecar_present is True
        # Top by |magnitude|: 3.0 (idx 3), 2.5 (idx 0)
        assert verdict.top_pair_indices == (3, 0)
        assert verdict.top_pair_residual_magnitudes == pytest.approx((3.0, 2.5))


def test_hook_6_consumer_12_no_payload_rank_returns_empty_tuples() -> None:
    """Sidecar present but no per-pair rank payload → empty rank."""
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_kkt_residuals", sha, n_pairs=10
    ):
        verdict = disambiguate_per_pair_stationarity(sha, top_n=5)
        # sidecar present + valid, but no recognizable rank key
        assert verdict.sidecar_present is True
        assert verdict.top_pair_indices == ()
        assert verdict.top_pair_residual_magnitudes == ()


def test_hook_6_consumer_12_top_n_exceeds_pairs_returns_all() -> None:
    """top_n > n_pairs → returns all pairs sorted by magnitude."""
    sha = _fresh_archive_sha()
    residuals = [0.3, 0.1, 0.5]
    with _staged_canonical_sidecar(
        "per_pair_kkt_residuals",
        sha,
        n_pairs=3,
        extra={"per_pair_residual_magnitudes": residuals},
    ):
        verdict = disambiguate_per_pair_stationarity(sha, top_n=10)
        assert len(verdict.top_pair_indices) == 3  # all 3 pairs


def test_hook_6_default_top_n_is_16() -> None:
    assert DEFAULT_DISAMBIGUATOR_TOP_N == 16


def test_hook_6_consumer_12_rationale_cites_producer_header() -> None:
    verdict = disambiguate_per_pair_stationarity(_fresh_archive_sha())
    assert "PROBE_DISAMBIGUATOR" in verdict.rationale
    assert "[predicted]" in verdict.rationale


# ─── Hook #6: Catalog #303 N/A cells refused at query time ──────────────────


@pytest.mark.parametrize(
    "consumer_id",
    [
        "per_pair_pareto_envelope",
        "per_pair_lagrangian_lambda_bisection",
        "per_pair_lora_supervision_signal",
        "per_pair_coding_budget_allocation",
        "per_pair_volterra_cross_terms",
    ],
)
def test_hook_6_disambiguate_for_na_consumer_raises_per_catalog_303(
    consumer_id: str,
) -> None:
    """N/A consumers per producer-header CONSUMER_HOOK_NUMBERS raise."""
    with pytest.raises(ValueError, match="does NOT own hook #6"):
        disambiguate_per_pair_stationarity(
            _fresh_archive_sha(), consumer_id=consumer_id
        )


def test_hook_6_verdict_construction_for_na_consumer_refused() -> None:
    """Direct Hook6DisambiguatorVerdict construction for N/A consumer raises."""
    with pytest.raises(ValueError, match="is N/A for hook #6"):
        Hook6DisambiguatorVerdict(
            consumer_id="per_pair_pareto_envelope",  # N/A per CONSUMER_HOOK_NUMBERS
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_pareto_envelope_v1",
            sidecar_sha256=None,
            n_pairs=0,
            top_pair_indices=(),
            top_pair_residual_magnitudes=(),
            top_n_requested=16,
        )


def test_hook_6_empty_archive_sha_rejected() -> None:
    with pytest.raises(ValueError, match="archive_sha256"):
        disambiguate_per_pair_stationarity("")


def test_hook_6_negative_top_n_rejected() -> None:
    with pytest.raises(ValueError, match="top_n"):
        disambiguate_per_pair_stationarity(_fresh_archive_sha(), top_n=-1)


# ─── Hook #6: canonical non-promotable invariants per Catalog #287/#323/#341 ─


def test_hook_6_verdict_score_claim_valid_forbidden() -> None:
    with pytest.raises(ValueError, match="score_claim_valid"):
        Hook6DisambiguatorVerdict(
            consumer_id="per_pair_kkt_residuals",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_kkt_residuals_v1",
            sidecar_sha256=None,
            n_pairs=0,
            top_pair_indices=(),
            top_pair_residual_magnitudes=(),
            top_n_requested=16,
            score_claim_valid=True,
        )


def test_hook_6_verdict_promotion_eligible_forbidden() -> None:
    with pytest.raises(ValueError, match="promotion_eligible"):
        Hook6DisambiguatorVerdict(
            consumer_id="per_pair_kkt_residuals",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_kkt_residuals_v1",
            sidecar_sha256=None,
            n_pairs=0,
            top_pair_indices=(),
            top_pair_residual_magnitudes=(),
            top_n_requested=16,
            promotion_eligible=True,
        )


def test_hook_6_verdict_axis_must_be_predicted() -> None:
    with pytest.raises(ValueError, match="axis_tag"):
        Hook6DisambiguatorVerdict(
            consumer_id="per_pair_kkt_residuals",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_kkt_residuals_v1",
            sidecar_sha256=None,
            n_pairs=0,
            top_pair_indices=(),
            top_pair_residual_magnitudes=(),
            top_n_requested=16,
            axis_tag="[contest-CUDA]",
        )


def test_hook_6_verdict_invalid_verdict_taxonomy_rejected() -> None:
    """Per Catalog #313: verdict must be in canonical VALID_VERDICTS set."""
    with pytest.raises(ValueError, match="VALID_VERDICTS"):
        Hook6DisambiguatorVerdict(
            consumer_id="per_pair_kkt_residuals",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_kkt_residuals_v1",
            sidecar_sha256=None,
            n_pairs=0,
            top_pair_indices=(),
            top_pair_residual_magnitudes=(),
            top_n_requested=16,
            verdict="NOT_A_VALID_VERDICT",
        )


def test_hook_6_verdict_default_is_partial_per_catalog_313() -> None:
    """Per Catalog #313: default verdict is PARTIAL (informational rank)."""
    assert HOOK_6_DEFAULT_VERDICT == "PARTIAL"
    verdict = disambiguate_per_pair_stationarity(_fresh_archive_sha())
    assert verdict.verdict == "PARTIAL"


def test_hook_6_verdict_parallel_array_invariant() -> None:
    """top_pair_indices and top_pair_residual_magnitudes MUST be parallel."""
    with pytest.raises(ValueError, match="parallel arrays"):
        Hook6DisambiguatorVerdict(
            consumer_id="per_pair_kkt_residuals",
            archive_sha256="a" * 64,
            sidecar_present=False,
            sidecar_path=None,
            sidecar_schema="master_gradient_consumer_per_pair_kkt_residuals_v1",
            sidecar_sha256=None,
            n_pairs=0,
            top_pair_indices=(0, 1, 2),
            top_pair_residual_magnitudes=(0.5, 0.3),  # mismatched length
            top_n_requested=16,
        )


def test_is_hook_6_verdict_promotable_always_false() -> None:
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar("per_pair_kkt_residuals", sha):
        verdict = disambiguate_per_pair_stationarity(sha)
        assert is_hook_6_verdict_promotable(verdict) is False


def test_hook_6_verdict_provenance_canonical_helper_pinned() -> None:
    verdict = disambiguate_per_pair_stationarity(_fresh_archive_sha())
    assert (
        verdict.provenance_canonical_helper
        == "tac.cathedral_solver_wire_in.hook6_probe_disambiguator"
    )


def test_hook_6_verdict_metric_name_pinned() -> None:
    """metric_name maps to canonical Catalog #313 probe-outcomes ledger field."""
    verdict = disambiguate_per_pair_stationarity(_fresh_archive_sha())
    assert verdict.metric_name == "per_pair_kkt_residual_magnitude_l2"


# ─── Hook #6: collection helper ─────────────────────────────────────────────


def test_collect_all_hook_6_verdicts_returns_one_per_catalog_303() -> None:
    verdicts = collect_all_hook_6_verdicts_for_archive(_fresh_archive_sha())
    assert len(verdicts) == 1
    assert verdicts[0].consumer_id == "per_pair_kkt_residuals"


def test_collect_all_hook_6_verdicts_all_observability_only() -> None:
    verdicts = collect_all_hook_6_verdicts_for_archive(_fresh_archive_sha())
    for v in verdicts:
        assert v.promotion_eligible is False
        assert v.score_claim_valid is False
        assert v.axis_tag == "[predicted]"


def test_collect_all_hook_6_verdicts_with_sidecar_present() -> None:
    sha = _fresh_archive_sha()
    residuals = [0.1, 0.5, 0.3]
    with _staged_canonical_sidecar(
        "per_pair_kkt_residuals",
        sha,
        n_pairs=3,
        extra={"per_pair_residual_magnitudes": residuals},
    ):
        verdicts = collect_all_hook_6_verdicts_for_archive(sha, top_n=2)
        assert len(verdicts) == 1
        assert verdicts[0].sidecar_present is True
        assert verdicts[0].top_pair_indices == (1, 2)


def test_collect_all_hook_6_verdicts_empty_sha_rejected() -> None:
    with pytest.raises(ValueError, match="archive_sha256"):
        collect_all_hook_6_verdicts_for_archive("")


# ─── Cross-hook 6-hook FULL CLOSURE consistency tests ──────────────────────


def test_consumer_9_lora_present_in_hooks_3_and_5() -> None:
    """Cross-hook: consumer #9 owns hook #3 (bit_allocator) AND hook #5
    (continual_learning_posterior); both surfaces report consistent
    sidecar presence + structural counts.
    """
    sha = _fresh_archive_sha()
    with _staged_canonical_sidecar(
        "per_pair_lora_supervision_signal", sha, n_pairs=300, n_bytes=4096
    ):
        # Hook #3 (HH surface)
        from tac.cathedral_solver_wire_in import (
            bit_allocator_contribution_for_consumer,
        )

        hook_3 = bit_allocator_contribution_for_consumer(
            "per_pair_lora_supervision_signal", sha
        )
        # Hook #5 (LL surface)
        hook_5 = query_posterior_for_consumer(
            "per_pair_lora_supervision_signal", sha
        )
        assert hook_3.sidecar_present == hook_5.sidecar_present == True
        assert hook_3.n_pairs == hook_5.n_pairs == 300
        assert hook_3.n_bytes == hook_5.n_bytes == 4096
        assert hook_3.sidecar_path == hook_5.sidecar_path


def test_consumer_12_kkt_present_in_hooks_1_2_and_6() -> None:
    """Cross-hook: consumer #12 owns hook #1 (sensitivity) + #2 (pareto) AND
    hook #6 (probe_disambiguator); all surfaces report consistent presence.
    """
    sha = _fresh_archive_sha()
    residuals = [0.5, 0.3, 0.7, 0.1]
    with _staged_canonical_sidecar(
        "per_pair_kkt_residuals",
        sha,
        n_pairs=4,
        extra={"per_pair_residual_magnitudes": residuals},
    ):
        from tac.cathedral_solver_wire_in import (
            pareto_constraint_contribution_for_consumer,
            sensitivity_map_contribution_for_consumer,
        )

        hook_1 = sensitivity_map_contribution_for_consumer(
            "per_pair_kkt_residuals", sha
        )
        hook_2 = pareto_constraint_contribution_for_consumer(
            "per_pair_kkt_residuals", sha
        )
        hook_6 = disambiguate_per_pair_stationarity(sha, top_n=2)
        # All present, same n_pairs
        assert hook_1.sidecar_present is True
        assert hook_2.sidecar_present is True
        assert hook_6.sidecar_present is True
        assert hook_1.n_pairs == hook_2.n_pairs == hook_6.n_pairs == 4
        # Hook #6 extracts top-2 rank: 0.7 (idx 2), 0.5 (idx 0)
        assert hook_6.top_pair_indices == (2, 0)


def test_full_6_hook_loop_aggregate_active_cell_count() -> None:
    """FULL CLOSURE: HH 9 (hooks 1+2+3) + LL 1 (hook 5) + LL 1 (hook 6) = 11
    HARD-EARNED ACTIVE cells across the 6 Cable D consumers per Catalog #303.
    """
    hh_count = len(CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS)
    ll_5_count = len(CABLE_D_CONSUMERS_7_14_HOOK_5_PAIRS)
    ll_6_count = len(CABLE_D_CONSUMERS_7_14_HOOK_6_PAIRS)
    assert hh_count == 9
    assert ll_5_count == 1
    assert ll_6_count == 1
    assert hh_count + ll_5_count + ll_6_count == 11


def test_each_consumer_owns_at_least_one_solver_hook() -> None:
    """Every Cable D consumer 7-14 owns at least one HARD-EARNED hook
    across the FULL solver-wire-in package (hooks #1+#2+#3+#5+#6).
    """
    consumers = sorted(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY.keys())
    for consumer_id in consumers:
        hh_hooks = CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY[consumer_id]
        owns_hh = len(hh_hooks) > 0
        owns_5 = consumer_owns_hook_5(consumer_id)
        owns_6 = consumer_owns_hook_6(consumer_id)
        assert owns_hh or owns_5 or owns_6, (
            f"consumer {consumer_id!r} has ZERO HARD-EARNED hooks across "
            "the FULL solver-wire-in package — this would be an orphan-signal "
            "regression."
        )


# ─── Sister regression tests (HH + FF + Catalog #335/#341 still pass) ──────


def test_sister_hh_canonical_hook_pairs_unchanged() -> None:
    """HH 9-cell registry must be unchanged by LL landing."""
    assert len(CABLE_D_CONSUMERS_7_14_SOLVER_HOOK_PAIRS) == 9


def test_sister_hh_hook_registry_consumers_unchanged() -> None:
    """HH 6-consumer registry unchanged."""
    assert len(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY) == 6
    assert set(CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY.keys()) == {
        "per_pair_pareto_envelope",
        "per_pair_lagrangian_lambda_bisection",
        "per_pair_lora_supervision_signal",
        "per_pair_coding_budget_allocation",
        "per_pair_kkt_residuals",
        "per_pair_volterra_cross_terms",
    }


def test_sister_observability_axis_tag_unchanged() -> None:
    """Shared canonical [predicted] axis tag preserved across HH + LL."""
    assert SOLVER_WIRE_IN_OBSERVABILITY_AXIS == "[predicted]"


def test_sister_canonical_sidecar_registry_unchanged() -> None:
    assert len(CABLE_D_CONSUMERS_7_14_CANONICAL_SIDECARS) == 6


def test_sister_hh_consumer_owns_hook_callable() -> None:
    """HH consumer_owns_hook still works post-LL landing."""
    assert consumer_owns_hook(
        "per_pair_lagrangian_lambda_bisection", "sensitivity_map"
    ) is True
    assert consumer_owns_hook("per_pair_pareto_envelope", "bit_allocator") is False


def test_sister_solver_hook_contribution_construction_still_works() -> None:
    """HH SolverHookContribution still constructible post-LL landing."""
    c = SolverHookContribution(
        consumer_id="per_pair_pareto_envelope",
        hook_name="pareto_constraint",
        archive_sha256="a" * 64,
        sidecar_present=False,
        sidecar_path=None,
        sidecar_schema="master_gradient_consumer_per_pair_pareto_envelope_v1",
        n_pairs=0,
        n_bytes=0,
    )
    assert c.consumer_id == "per_pair_pareto_envelope"


# ─── Live-repo regression guards ────────────────────────────────────────────


def test_live_repo_cathedral_consumer_count_unchanged() -> None:
    """Per HH baseline: cathedral consumer count is 34 at LL landing."""
    from tools.cathedral_autopilot_autonomous_loop import (
        discover_compliant_consumer_modules,
    )

    consumers = list(discover_compliant_consumer_modules())
    assert len(consumers) >= 34, (
        f"cathedral consumer count regressed: was 34 at HH baseline, now "
        f"{len(consumers)}"
    )


def test_live_repo_hook_5_active_pinned_per_producer_header() -> None:
    """Live-repo regression guard: hook #5 ACTIVE set is exactly 1 consumer
    per the canonical producer-header CONSUMER_HOOK_NUMBERS declarations.
    """
    assert CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE == frozenset(
        {"per_pair_lora_supervision_signal"}
    )


def test_live_repo_hook_6_active_pinned_per_producer_header() -> None:
    """Live-repo regression guard: hook #6 ACTIVE set is exactly 1 consumer
    per the canonical producer-header CONSUMER_HOOK_NUMBERS declarations.
    """
    assert CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE == frozenset(
        {"per_pair_kkt_residuals"}
    )


def test_live_repo_no_phantom_solver_hook_in_full_loop() -> None:
    """No (consumer × hook) cell in the FULL solver-wire-in loop is phantom
    per Catalog #303: every ACTIVE cell maps to a producer-header declaration.
    """
    # Hook #5 ACTIVE consumer must declare CONTINUAL_LEARNING_POSTERIOR
    from tac.cathedral_consumers.per_pair_lora_supervision_signal_consumer import (
        CONSUMER_HOOK_NUMBERS as lora_hooks,
    )
    from tac.cathedral.consumer_contract import HookNumber

    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in lora_hooks

    # Hook #6 ACTIVE consumer must declare PROBE_DISAMBIGUATOR
    from tac.cathedral_consumers.per_pair_kkt_residuals_consumer import (
        CONSUMER_HOOK_NUMBERS as kkt_hooks,
    )

    assert HookNumber.PROBE_DISAMBIGUATOR in kkt_hooks


def test_live_repo_na_hook_5_consumers_do_not_declare_posterior() -> None:
    """Live regression: N/A hook #5 consumers MUST NOT declare
    CONTINUAL_LEARNING_POSTERIOR in their CONSUMER_HOOK_NUMBERS.
    """
    from tac.cathedral.consumer_contract import HookNumber
    from tac.cathedral_consumers.per_pair_pareto_envelope_consumer import (
        CONSUMER_HOOK_NUMBERS as pe_hooks,
    )
    from tac.cathedral_consumers.per_pair_lagrangian_lambda_bisection_consumer import (
        CONSUMER_HOOK_NUMBERS as lambda_hooks,
    )
    from tac.cathedral_consumers.per_pair_coding_budget_allocation_consumer import (
        CONSUMER_HOOK_NUMBERS as cb_hooks,
    )
    from tac.cathedral_consumers.per_pair_kkt_residuals_consumer import (
        CONSUMER_HOOK_NUMBERS as kkt_hooks,
    )
    from tac.cathedral_consumers.per_pair_volterra_cross_terms_consumer import (
        CONSUMER_HOOK_NUMBERS as volterra_hooks,
    )

    for hook_tuple, name in [
        (pe_hooks, "per_pair_pareto_envelope"),
        (lambda_hooks, "per_pair_lagrangian_lambda_bisection"),
        (cb_hooks, "per_pair_coding_budget_allocation"),
        (kkt_hooks, "per_pair_kkt_residuals"),
        (volterra_hooks, "per_pair_volterra_cross_terms"),
    ]:
        assert HookNumber.CONTINUAL_LEARNING_POSTERIOR not in hook_tuple, (
            f"consumer {name!r} declares CONTINUAL_LEARNING_POSTERIOR but is "
            "tracked as hook #5 N/A — registry mismatch with producer header"
        )


def test_live_repo_na_hook_6_consumers_do_not_declare_disambiguator() -> None:
    """Live regression: N/A hook #6 consumers MUST NOT declare
    PROBE_DISAMBIGUATOR in their CONSUMER_HOOK_NUMBERS.
    """
    from tac.cathedral.consumer_contract import HookNumber
    from tac.cathedral_consumers.per_pair_pareto_envelope_consumer import (
        CONSUMER_HOOK_NUMBERS as pe_hooks,
    )
    from tac.cathedral_consumers.per_pair_lagrangian_lambda_bisection_consumer import (
        CONSUMER_HOOK_NUMBERS as lambda_hooks,
    )
    from tac.cathedral_consumers.per_pair_lora_supervision_signal_consumer import (
        CONSUMER_HOOK_NUMBERS as lora_hooks,
    )
    from tac.cathedral_consumers.per_pair_coding_budget_allocation_consumer import (
        CONSUMER_HOOK_NUMBERS as cb_hooks,
    )
    from tac.cathedral_consumers.per_pair_volterra_cross_terms_consumer import (
        CONSUMER_HOOK_NUMBERS as volterra_hooks,
    )

    for hook_tuple, name in [
        (pe_hooks, "per_pair_pareto_envelope"),
        (lambda_hooks, "per_pair_lagrangian_lambda_bisection"),
        (lora_hooks, "per_pair_lora_supervision_signal"),
        (cb_hooks, "per_pair_coding_budget_allocation"),
        (volterra_hooks, "per_pair_volterra_cross_terms"),
    ]:
        assert HookNumber.PROBE_DISAMBIGUATOR not in hook_tuple, (
            f"consumer {name!r} declares PROBE_DISAMBIGUATOR but is tracked "
            "as hook #6 N/A — registry mismatch with producer header"
        )


# ─── Catalog #313 sister probe-outcomes ledger integration ─────────────────


def test_hook_6_verdict_uses_catalog_313_canonical_valid_verdicts() -> None:
    """The verdict taxonomy mirrors tac.probe_outcomes_ledger.VALID_VERDICTS."""
    from tac.probe_outcomes_ledger import VALID_VERDICTS

    # The hook #6 verdict must be in canonical Catalog #313 VALID_VERDICTS
    verdict = disambiguate_per_pair_stationarity(_fresh_archive_sha())
    assert verdict.verdict in VALID_VERDICTS


def test_hook_6_verdict_metric_name_consumable_by_probe_outcomes_ledger() -> None:
    """The verdict's metric_name + verdict can be passed to register_probe_outcome."""
    # Smoke: parameter shape compatible with canonical helper signature.
    # We do NOT actually call register_probe_outcome here (would mutate
    # canonical ledger); we verify the parameter shapes are compatible.
    verdict = disambiguate_per_pair_stationarity(_fresh_archive_sha())
    assert isinstance(verdict.metric_name, str) and verdict.metric_name
    assert isinstance(verdict.verdict, str) and verdict.verdict
    assert verdict.sidecar_path is None or isinstance(verdict.sidecar_path, str)
