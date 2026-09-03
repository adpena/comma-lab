from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tac.canonical_frontier_pointer import (
    CanonicalFrontierPointer,
    effective_frontier_score,
    load_canonical_frontier_pointer_strict,
    recompute_effective_frontier,
)
from tac.score_geometry import contest_score, target_byte_budget_for_score
from tac.witness_dsl.taskspace_inverse_stack_receipt import (
    RECEIPT_SCHEMA,
    SCHEMA,
    TaskspaceInverseStackReceiptError,
    _strict_max_archive_bytes,
    build_stack_receipt,
    canonical_json_bytes,
    validate_stack_receipt,
)

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_stack_receipt(repo_root=REPO, strict_source_reopen=False)


def _rehash(receipt: dict) -> dict:
    mutated = copy.deepcopy(receipt)
    import hashlib

    mutated["body_sha256"] = hashlib.sha256(canonical_json_bytes(mutated["body"])).hexdigest()
    return mutated


def test_receipt_reopens_p_g_a_optional_t_and_encoder_truth_without_candidate_authority(receipt: dict) -> None:
    assert receipt["schema"] == RECEIPT_SCHEMA
    body = receipt["body"]
    assert body["schema"] == SCHEMA
    assert body["authority"] == {
        "research_only": True,
        "score_claim": False,
        "candidate_score": None,
        "candidate_archive_emitted": False,
        "candidate_payload_eligible": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "originality_claim": False,
        "pointer_moved": False,
        "pointer_delta": None,
        "n600_is_only_decision_surface": True,
        "n64_is_bounded_non_promotable_timing_and_acquisition_diagnostic_only": True,
    }
    assert body["source_reopen"]["teacher_scope_pairs"] == 64
    assert body["source_reopen"]["decision_surface_pairs"] == 600
    assert body["p_g_a_t_ownership"]["T"] == {
        "role": "optional_counted_irreducible_terminal_quotient",
        "admission_rule": (
            "count_only_after_measured_same_object_matched_byte_P_G_A_controls_fail_to_improve_total_score"
        ),
        "instantiated": False,
        "candidate_bytes": None,
        "may_duplicate_preceding_owners": False,
    }
    assert body["p_g_a_t_ownership"]["E"]["candidate_payload_allowed"] is False
    assert body["borrowed_substrate_accounting"]["named_input_inventory_present"] is True
    assert body["borrowed_substrate_accounting"]["complete_candidate_archive_accounted"] is False
    assert body["readiness"]["complete_stack"]["n600_decision_ready"] is False
    assert body["lineage"]["originality_proven"] is False
    assert "receiver_consumption_custody_absent" in body["exact_blockers"]
    validate_stack_receipt(receipt)


def test_frontier_and_c1_ceiling_are_dynamic_prediction_only(receipt: dict) -> None:
    pointer = load_canonical_frontier_pointer_strict(repo_root=REPO)
    target = effective_frontier_score(pointer)
    assert target is not None
    ceiling = receipt["body"]["conditional_c1_n600_byte_ceiling"]
    expected = target_byte_budget_for_score(
        target_score=target,
        d_seg_floor=ceiling["d_seg"],
        d_pose_floor=ceiling["d_pose"],
    )
    assert ceiling["target_score"] == target
    assert ceiling["max_archive_bytes"] == expected.max_archive_bytes
    assert ceiling["max_archive_bytes_is_strict"] is True
    assert ceiling["next_archive_byte_is_not_strict"] is True
    assert ceiling["score_at_max_archive_bytes"] < target
    assert ceiling["score_at_next_archive_byte"] >= target
    assert ceiling["planning_helper_agrees_with_strict_ceiling"] is True
    assert ceiling["conditional_only"] is True
    assert ceiling["candidate_archive_bytes"] is None
    assert ceiling["score_claim"] is False


def test_changed_pointer_recomputes_target_and_conditional_ceiling(tmp_path: Path) -> None:
    pointer_path = tmp_path / "canonical_frontier_pointer.json"
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    changed_target = 0.1
    pointer["upstream_leaderboard_snapshot"]["best_entry"]["score"] = changed_target
    pointer["upstream_leaderboard_snapshot"]["entries"][0]["score"] = changed_target
    recomposed = recompute_effective_frontier(CanonicalFrontierPointer.from_dict(pointer))
    assert recomposed is not None
    pointer["effective_frontier"] = recomposed
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    changed = build_stack_receipt(
        repo_root=REPO,
        frontier_pointer_path=pointer_path,
        strict_source_reopen=False,
    )
    ceiling = changed["body"]["conditional_c1_n600_byte_ceiling"]
    assert changed["body"]["frontier_join"]["effective_frontier"]["score"] == changed_target
    assert ceiling["target_score"] == changed_target
    assert ceiling["max_archive_bytes"] < receipt_max_bytes(
        build_stack_receipt(repo_root=REPO, strict_source_reopen=False)
    )


def test_inconsistent_cached_effective_frontier_is_rejected(tmp_path: Path) -> None:
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    pointer["effective_frontier"] = {**pointer["effective_frontier"], "score": 0.99}
    path = tmp_path / "inconsistent_pointer.json"
    path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(TaskspaceInverseStackReceiptError, match="constituent minimum"):
        build_stack_receipt(repo_root=REPO, frontier_pointer_path=path, strict_source_reopen=False)


def test_stale_pointer_is_rejected_before_planning(tmp_path: Path) -> None:
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text(encoding="utf-8"))
    pointer["last_refreshed_utc"] = "2000-01-01T00:00:00+00:00"
    path = tmp_path / "stale_pointer.json"
    path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(TaskspaceInverseStackReceiptError, match="pointer is stale"):
        build_stack_receipt(repo_root=REPO, frontier_pointer_path=path, strict_source_reopen=False)


def test_exact_integral_byte_boundary_uses_open_sublevel() -> None:
    target = contest_score(0.0, 0.0, 10)
    assert _strict_max_archive_bytes(target_score=target, d_seg=0.0, d_pose=0.0) == 9


def test_float_cancellation_cannot_admit_a_boundary_tie() -> None:
    d_seg = 0.0002589167502929634
    d_pose = 0.0005112747213686085
    target = 0.11152800824355927
    maximum = _strict_max_archive_bytes(
        target_score=target,
        d_seg=d_seg,
        d_pose=d_pose,
    )
    assert maximum == 21224
    assert contest_score(d_seg, d_pose, maximum) < target
    assert contest_score(d_seg, d_pose, maximum + 1) >= target


def receipt_max_bytes(value: dict) -> int:
    result = value["body"]["conditional_c1_n600_byte_ceiling"]["max_archive_bytes"]
    assert isinstance(result, int)
    return result


def test_forged_self_rehashed_c1_v2_is_rejected_before_planning(tmp_path: Path) -> None:
    source = (
        REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/c1_live_target_debt_n600_batch16.json"
    )
    c1 = json.loads(source.read_text(encoding="utf-8"))
    c1["aggregate"]["mean_d_seg"] = 0.0
    c1["aggregate"]["mean_d_pose"] = 0.0
    path = tmp_path / "c1.json"
    path.write_text(json.dumps(c1), encoding="utf-8")
    with pytest.raises(TaskspaceInverseStackReceiptError, match="canonical path and immutable file SHA-256"):
        build_stack_receipt(repo_root=REPO, c1_anchor_path=path, strict_source_reopen=False)


def test_fresh_wrapper_timestamp_cannot_mask_stale_official_snapshot(tmp_path: Path) -> None:
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    pointer["upstream_leaderboard_snapshot_at_utc"] = "2000-01-01T00:00:00+00:00"
    recomposed = recompute_effective_frontier(CanonicalFrontierPointer.from_dict(pointer))
    assert recomposed is not None
    pointer["effective_frontier"] = recomposed
    path = tmp_path / "stale_official_pointer.json"
    path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(TaskspaceInverseStackReceiptError, match="official leaderboard snapshot is stale"):
        build_stack_receipt(repo_root=REPO, frontier_pointer_path=path, strict_source_reopen=False)


def test_lane_dag_body_hash_is_checked_before_consumption(tmp_path: Path) -> None:
    source = REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/roadmap_v4.json"
    roadmap = json.loads(source.read_text(encoding="utf-8"))
    roadmap["body"]["mission"]["strict_authoritative_target"] = 0.01
    path = tmp_path / "roadmap_v4.json"
    path.write_text(json.dumps(roadmap), encoding="utf-8")
    with pytest.raises(TaskspaceInverseStackReceiptError, match="body hash differs"):
        build_stack_receipt(repo_root=REPO, lane_dag_path=path, strict_source_reopen=False)


def test_source_symlink_is_rejected_by_no_follow_custody(tmp_path: Path) -> None:
    source = (
        REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/c1_live_target_debt_n600_batch16.json"
    )
    link = tmp_path / "c1-link.json"
    link.symlink_to(source)
    with pytest.raises(TaskspaceInverseStackReceiptError, match="without following links"):
        build_stack_receipt(repo_root=REPO, c1_anchor_path=link, strict_source_reopen=False)


def test_recomputed_hash_cannot_add_borrowed_candidate_bytes(receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["body"]["borrowed_substrate_accounting"]["rows"][0]["candidate_weights_bytes"] = 1
    with pytest.raises(TaskspaceInverseStackReceiptError, match="substrate authority"):
        validate_stack_receipt(_rehash(mutated))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authority", "candidate_score"), 0.1),
        (("authority", "promotion_eligible"), True),
        (("frontier_join", "roadmap_v4_fixed_target_ignored"), False),
        (("conditional_c1_n600_byte_ceiling", "conditional_only"), False),
        (("conditional_c1_n600_byte_ceiling", "max_archive_bytes"), 1),
        (("conditional_c1_n600_byte_ceiling", "source_axis"), "[contest-CPU]"),
        (("source_custody", "c1_n600_anchor", "path"), "forged-c1.json"),
        (("source_custody", "c1_n600_anchor", "sha256"), "0" * 64),
        (("lineage", "originality_proven"), True),
        (
            (
                "p_g_a_t_ownership",
                "A",
                "reverse_causal_G_to_exact_Y1_to_Y0_given_Y1_reference_grammar_present",
            ),
            False,
        ),
        (("p_g_a_t_ownership", "T", "instantiated"), True),
        (("p_g_a_t_ownership", "E", "candidate_payload_allowed"), True),
        (("borrowed_substrate_accounting", "originality_proven"), True),
        (("readiness", "complete_stack", "n600_decision_ready"), True),
    ],
)
def test_recomputed_hash_cannot_relax_authority(receipt: dict, path: tuple[str, ...], value: object) -> None:
    mutated = copy.deepcopy(receipt)
    cursor = mutated["body"]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(TaskspaceInverseStackReceiptError):
        validate_stack_receipt(_rehash(mutated))


def test_canonical_sources_strictly_reopen_after_regeneration() -> None:
    receipt = build_stack_receipt(repo_root=REPO, strict_source_reopen=True)
    assert receipt["body"]["source_reopen"] == {
        "teacher_census": "STRICT_REOPEN_PASS",
        "prior_signal_harvest": "STRICT_REOPEN_PASS",
        "teacher_scope_pairs": 64,
        "teacher_scope_role": "bounded_non_promotable_timing_and_acquisition_diagnostic_only",
        "decision_surface_pairs": 600,
    }
