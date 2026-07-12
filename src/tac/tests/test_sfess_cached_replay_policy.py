# SPDX-License-Identifier: MIT
"""Fail-closed DSL tests for the isolated SFESS cached replay."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from tac.sfess_cached_replay import cached_state_sha256
from tac.witness_dsl.scorer_gradient_policy import ScorerGradientPolicy
from tac.witness_dsl.sfess_cached_replay_policy import (
    SFESSCacheCustody,
    SFESSCachedReplayPolicy,
    SFESSObjectiveContext,
)

SHA = "a" * 64
SOURCE_VIDEO_SHA = "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"


def _custody(path: Path, *, kind: str = "cache") -> SFESSCacheCustody:
    payload = path.read_bytes()
    return SFESSCacheCustody(
        kind=kind,
        path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _policy(tmp_path: Path, **overrides: object) -> SFESSCachedReplayPolicy:
    table = tmp_path / "table.jsonl"
    receipt = tmp_path / "receipt.json"
    manifest = tmp_path / "manifest.json"
    source_video = tmp_path / "0.mkv"
    table.write_text("table\n", encoding="utf-8")
    receipt.write_text("receipt\n", encoding="utf-8")
    manifest.write_text("manifest\n", encoding="utf-8")
    source_video.write_bytes(b"source-video-fixture")
    table_custody = _custody(table)
    receipt_custody = _custody(receipt)
    manifest_custody = _custody(manifest)
    source_video_custody = _custody(source_video, kind="source_video")
    context = SFESSObjectiveContext(
        objective_table_sha256=table_custody.sha256,
        measurement_receipt_sha256=receipt_custody.sha256,
        candidate_manifest_sha256=manifest_custody.sha256,
        fixture_archive_sha256=SHA,
        fixture_authority_sha256="b" * 64,
        source_video_sha256=source_video_custody.sha256,
        source_video_bytes=source_video_custody.size_bytes,
        n_bits=6,
        state_count=64,
        mask_order="little_endian_bit_j_equals_index_shift_j",
        axis="[macOS-CPU advisory . frozen CPU-torch exact cells . NON-PROMOTABLE]",
    )
    payload: dict[str, object] = {
        "mode": "sfess_cached_k_subset",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "produces_costate": False,
        "live_gradient_fallback": "full_teacher",
        "cache_failure_action": "refuse",
        "objective_context": context,
        "objective_context_fingerprint": context.fingerprint(),
        "objective_table_custody": table_custody,
        "measurement_receipt_custody": receipt_custody,
        "candidate_manifest_custody": manifest_custody,
        "source_video_custody": source_video_custody,
        "k_values": (1, 2, 3, 4, 5),
        "include_degenerate_k_controls": True,
        "samples_per_gradient": 5,
        "eval_budget_per_k": 64,
        "seed": 396_400,
        "max_evidence_age_queries": 0,
        "comparison_noise_floor_s": 1.0e-12,
        "initial_mask_rule": "lowest_indices",
        "acceptance_rule": "strict_improvement_beyond_registered_floor",
        "retention_rule": "strict_gated_returned_state",
        "k_selection_status": "post_hoc_exploratory",
        "control_variate_anchor": "wijk_2024_five_sample_leave_one_out",
    }
    payload.update(overrides)
    return SFESSCachedReplayPolicy.model_validate(payload)


def _authorize(compiled, policy, **overrides: object):
    mask = (1, 0, 1, 0, 0, 0)
    value = 0.1908
    payload: dict[str, object] = {
        "table_source_sha256": policy.objective_context.objective_table_sha256,
        "table_n_bits": 6,
        "mask": mask,
        "value": value,
        "declared_state_sha256": cached_state_sha256(mask, value),
        "query_index": 7,
        "evidence_query_index": 7,
        "current_objective_context_fingerprint": policy.objective_context.fingerprint(),
    }
    payload.update(overrides)
    return compiled.authorize_lookup(**payload)


def test_cached_lookup_is_admitted_but_live_gradient_is_never_admitted(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    decision = _authorize(policy.compile(), policy)
    assert decision.admitted_for_cached_lookup is True
    assert decision.live_gradient_admitted is False
    assert decision.fallback_to_full_teacher is True
    assert decision.evidence_age_queries == 0
    assert len(decision.custody_checks) == 4


def test_mutated_cache_fails_closed_without_rehash_laundering(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    compiled = policy.compile()
    Path(policy.objective_table_custody.path).write_text("changed\n", encoding="utf-8")
    decision = _authorize(compiled, policy)
    assert decision.admitted_for_cached_lookup is False
    assert decision.fallback_to_full_teacher is True
    assert any("objective_table custody failed" in reason for reason in decision.reasons)


def test_mutated_source_video_fails_closed_after_full_hash_compile(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    compiled = policy.compile()
    Path(policy.source_video_custody.path).write_bytes(b"changed-source-video")
    decision = _authorize(compiled, policy)
    assert decision.admitted_for_cached_lookup is False
    assert decision.fallback_to_full_teacher is True
    assert any("source_video custody failed" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"evidence_query_index": 6}, "stale"),
        ({"evidence_query_index": 8}, "future"),
        ({"table_n_bits": 5}, "n_bits mismatch"),
        ({"table_source_sha256": "f" * 64}, "provider fingerprint mismatch"),
        ({"current_objective_context_fingerprint": "f" * 64}, "context fingerprint mismatch"),
        ({"declared_state_sha256": "f" * 64}, "state fingerprint mismatch"),
        ({"mask": (1, 0)}, "mask length mismatch"),
        ({"value": float("nan")}, "nonfinite"),
    ],
)
def test_changed_age_nonfinite_and_fingerprint_evidence_fail_closed(
    tmp_path: Path, overrides: dict[str, object], reason_fragment: str
) -> None:
    policy = _policy(tmp_path)
    decision = _authorize(policy.compile(), policy, **overrides)
    assert decision.admitted_for_cached_lookup is False
    assert decision.live_gradient_admitted is False
    assert decision.fallback_to_full_teacher is True
    assert any(reason_fragment in reason for reason in decision.reasons)


def test_scorer_costate_policy_rejects_sfess_mode() -> None:
    with pytest.raises(ValidationError):
        ScorerGradientPolicy.model_validate({"mode": "sfess_cached_k_subset"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"samples_per_gradient": 4},
        {"eval_budget_per_k": 63},
        {"max_evidence_age_queries": 1},
        {"k_values": (1, 3, 2)},
        {"k_values": (0, 1, 2)},
        {"produces_costate": True},
        {"score_claim": True},
        {"promotion_eligible": True},
        {"live_gradient_fallback": "cached"},
    ],
)
def test_control_law_and_false_authority_fields_have_no_loose_defaults(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        _policy(tmp_path, **overrides)


def test_context_refuses_promotable_axis_and_incomplete_state_space() -> None:
    base = {
        "objective_table_sha256": SHA,
        "measurement_receipt_sha256": "b" * 64,
        "candidate_manifest_sha256": "c" * 64,
        "fixture_archive_sha256": "d" * 64,
        "fixture_authority_sha256": "e" * 64,
        "source_video_sha256": SOURCE_VIDEO_SHA,
        "source_video_bytes": 37_545_489,
        "n_bits": 6,
        "state_count": 64,
        "mask_order": "little_endian_bit_j_equals_index_shift_j",
        "axis": "[macOS-CPU advisory]",
    }
    with pytest.raises(ValidationError):
        SFESSObjectiveContext.model_validate({**base, "state_count": 63})
    with pytest.raises(ValidationError):
        SFESSObjectiveContext.model_validate({**base, "axis": "[contest-CPU]"})
