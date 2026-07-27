# SPDX-License-Identifier: MIT
"""Implementation tests only; these small fixtures are never scientific evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.c0b_semantic_quotient import PlaneChunk
from tac.witness_dsl.taskspace_conditional_quotient_profiler_v1 import (
    CONFIG_SCHEMA,
    EVIDENCE_AXIS,
    INPUT_BINDING_SCHEMA,
    REPRESENTATION_IDS,
    ConditionalQuotientProfileConfigV1,
    ConditionalQuotientProfilerError,
    profile_conditional_quotient_chunk,
    run_conditional_quotient_profile,
)


def _fresh_v15_custody(digest: str) -> dict[str, object]:
    typed_config_sha = "b" * 64
    checkpoints: list[dict[str, object]] = []
    digest_material: list[str] = []
    for index in range(38):
        start = index * 16
        stop = min(start + 16, 600)
        camera_digest = hashlib.sha256(f"camera-{index}".encode()).hexdigest()
        digest_material.append(camera_digest + camera_digest)
        checkpoints.append(
            {
                "path": f"/fixture/full_p_camera_identity/batch_{start:04d}_{stop:04d}.json",
                "bytes": 429,
                "sha256": hashlib.sha256(f"checkpoint-{index}".encode()).hexdigest(),
                "local_pair_range": [start, stop],
                "typed_config_sha256": typed_config_sha,
                "base_camera_sha256": camera_digest,
                "final_camera_sha256": camera_digest,
                "byte_identical": True,
                "score_claim": False,
            }
        )
    chain = hashlib.sha256("".join(digest_material).encode("ascii")).hexdigest()
    return {
        "schema": "tac.taskspace_fresh_v15_derivation_custody.v1",
        "run_id": "fixture-fresh-v15-run",
        "derivation_proof_separate_from_archive_content_identity": True,
        "historical_path_fallback_allowed": False,
        "compile_receipt": {
            "path": "/fixture/fresh/compile_receipt.json",
            "bytes": 1,
            "sha256": digest,
            "schema": "ddm_v15_scorer_solved_template_receipt.v1",
            "run_id": "fixture-fresh-v15-run",
        },
        "source_config": {
            "path": "/fixture/fresh/source_config.json",
            "bytes": 1,
            "sha256": digest,
            "rfc8785_sha256": typed_config_sha,
        },
        "adjacent_archive": {
            "path": "/fixture/v15.zip",
            "bytes": 133_941,
            "sha256": digest,
            "content_identity_only": True,
        },
        "producer_sources": [
            {
                "path": "fixture_producer.py",
                "resolved_path": "/fixture/fixture_producer.py",
                "bytes": 1,
                "sha256": digest,
                "live_rehashed": True,
            }
        ],
        "receiver_checkpoint": {
            "path": "/fixture/fresh/stage_checkpoints/02_receiver_closed_archive.json",
            "bytes": 1,
            "sha256": digest,
            "schema": "ddm_v15_receiver_closed_archive.v1",
            "typed_config_sha256": typed_config_sha,
            "archive_sha256": digest,
            "score_claim": False,
        },
        "full_p_camera_identity": {
            "pair_count": 600,
            "batch_count": 38,
            "batch_size": 16,
            "typed_config_sha256": typed_config_sha,
            "ordered_checkpoints": checkpoints,
            "receipt_digest_chain_sha256": chain,
            "recomputed_digest_chain_sha256": chain,
            "digest_chain_matches_receipt": True,
            "all_camera_bytes_identical": True,
            "score_claim": False,
        },
    }


def _config(*, pair_count: int = 4, chunk_pairs: int = 2) -> ConditionalQuotientProfileConfigV1:
    return ConditionalQuotientProfileConfigV1(
        pair_count=pair_count,
        chunk_pairs=chunk_pairs,
        scorer_hw=(4, 5),
        channels=3,
        test_only_small_fixture=True,
    )


def _binding(config: ConditionalQuotientProfileConfigV1) -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema": INPUT_BINDING_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_allowed": False,
        "candidate_payload_allowed": False,
        "teacher_payload_serialized": False,
        "scorer_weights_present": False,
        "pair_count": config.pair_count,
        "scorer_hw": list(config.scorer_hw),
        "channels": config.channels,
        "v15_archive_path": "/fixture/v15.zip",
        "v15_archive_bytes": 133_941,
        "v15_archive_sha256": digest,
        "v15_strict_parse": True,
        "v15_current_receiver_source_sha256": digest,
        "fresh_v15_derivation_custody": _fresh_v15_custody(digest),
        "base_coordinate_transform": {
            "camera_hw": [874, 1164],
            "scorer_hw": list(config.scorer_hw),
            "method": "c0b_disjoint_factor2_exact_integer_resize_round_u8",
            "operator_source": "src/tac/optimization/uint8_lattice_feasibility.py",
        },
        "selected_plane_teacher_id": "fixture-selected-preimage",
        "selected_plane_y0_sha256": digest,
        "selected_plane_y1_sha256": digest,
        "selected_plane_origin_scorer_batch_size": 32,
        "selected_plane_geometry_custody": {
            "path": "/fixture/custody.json",
            "sha256": digest,
        },
        "fresh_teacher_scorer_batch_size": 16,
        "fresh_teacher_target_labels_path": "/fixture/labels.u8",
        "fresh_teacher_target_labels_sha256": digest,
        "fresh_teacher_receipt": {
            "path": "/fixture/fresh.json",
            "sha256": digest,
            "sealed_receipt_sha256": digest,
        },
        "upstream_default_scorer_batch_size": 16,
        "current_planning_scorer_batch_size": 16,
        "current_planning_matches_upstream_batch_geometry": True,
        "canonical_batch16_debt_receipt": {
            "path": "/fixture/canonical_batch16.json",
            "sha256": digest,
            "receipt_sha256": digest,
            "axis": "[fixture advisory]",
            "batch_size": 16,
            "decoded_raw_sha256": digest,
            "contest_cpu_same_raw_crosscheck_sha256": digest,
            "authority": "PRIMARY_EXISTING_BATCH16_PLANNING_COORDINATE_NOT_SCORE_AUTHORITY",
            "score_claim": False,
        },
        "independent_batch16_replay_corroboration": {
            "path": "/fixture/g54.json",
            "sha256": digest,
            "receipt_sha256": digest,
            "axis": "[fixture corroboration]",
            "batch_size": 16,
            "decoded_raw_sha256": digest,
            "d_seg": 0.00015196057211142033,
            "d_pose": 0.0001018434704747051,
            "distortion_minus_canonical_primary": -1.805437170927273e-09,
            "authority": "INDEPENDENT_CORROBORATION_ONLY_NOT_PRIMARY_OR_SCORE_AUTHORITY",
            "score_claim": False,
        },
        "planning_coordinate_premise": (
            "PREEXISTING_CANONICAL_BATCH16_PRIMARY_G54_INDEPENDENT_CORROBORATION_NO_NOVELTY"
        ),
        "frontier_pointer": {
            "path": "/fixture/frontier.json",
            "sha256": digest,
            "effective_frontier_score": 0.172,
            "selection_rule": "fixture",
        },
        "current_batch16_planning_coordinate": {
            "effective_frontier_score": 0.172,
            "d_seg": 0.00015196058485243054,
            "d_pose": 0.00010184347386600314,
            "largest_total_archive_bytes_below_effective_frontier": 187_563,
            "largest_total_archive_bytes_below_sub_0_15": 154_523,
            "base_archive_bytes": 133_941,
            "headroom_bytes_to_effective_frontier": 53_622,
            "headroom_bytes_to_sub_0_15": 20_582,
            "authority": "canonical_batch16_planning_arithmetic_only_not_new_eval_or_frontier_authority",
            "score_claim": False,
        },
        "historical_ms1_batch32_counterfactual": {
            "effective_frontier_score": 0.172,
            "d_seg": 0.0001519690619574653,
            "d_pose": 0.00010184327939026322,
            "largest_total_archive_bytes_below_effective_frontier": 187_562,
            "largest_total_archive_bytes_below_sub_0_15": 154_522,
            "base_archive_bytes": 133_941,
            "headroom_bytes_to_effective_frontier": 53_621,
            "headroom_bytes_to_sub_0_15": 20_581,
            "authority": "historical_batch32_coupled_score_arithmetic_only_not_eval_or_frontier_authority",
            "score_claim": False,
        },
        "implementation_sources": {
            "fixture": {
                "path": "fixture.py",
                "bytes": 1,
                "sha256": digest,
            }
        },
    }


def _chunk(
    chunk_index: int,
    pair_ids: tuple[int, ...],
) -> tuple[PlaneChunk, PlaneChunk, np.ndarray]:
    pair_count = len(pair_ids)
    values = np.arange(pair_count * 4 * 5 * 3, dtype=np.uint16).reshape(pair_count, 4, 5, 3)
    base0 = ((values + 7 * pair_ids[0]) % 256).astype(np.uint8)
    base1 = ((values * 3 + 11 + pair_ids[0]) % 256).astype(np.uint8)
    target0 = base0.copy()
    target1 = base1.copy()
    target0[:, 1:3, 2:4, 0] = (target0[:, 1:3, 2:4, 0].astype(np.uint16) + 17).astype(np.uint8)
    target1[:, ::2, 1::2, 2] = (target1[:, ::2, 1::2, 2].astype(np.uint16) - 9).astype(np.uint8)
    labels = np.broadcast_to(
        np.arange(4 * 5, dtype=np.uint8).reshape(1, 4, 5) % 5,
        (pair_count, 4, 5),
    ).copy()
    return (
        PlaneChunk(chunk_index, pair_ids, base0, base1),
        PlaneChunk(chunk_index, pair_ids, target0, target1),
        labels,
    )


def test_profile_chunk_races_all_exact_bases_and_emits_actionable_statistics() -> None:
    base, target, labels = _chunk(0, (0, 1))
    row = profile_conditional_quotient_chunk(
        base,
        target,
        labels,
        run_binding_sha256="b" * 64,
    )
    assert tuple(row["representations"]) == REPRESENTATION_IDS
    assert row["all_representations_exact_roundtrip"] is True
    assert len(row["pair_marginals"]) == 2
    assert len(row["class_conditioned_signed_residual"]) == 30
    assert all(
        representation["codec_sizes"]["zlib9_block_bytes"] > 0 for representation in row["representations"].values()
    )
    assert row["score_claim"] is False
    assert row["candidate_payload"] is False


def test_seg_primary_pose_enhancement_is_an_exact_distinct_layer() -> None:
    base, target, labels = _chunk(0, (0,))
    shared = target.y1.copy()
    shared_target = PlaneChunk(0, (0,), shared, shared.copy())
    row = profile_conditional_quotient_chunk(
        base,
        shared_target,
        labels,
        run_binding_sha256="c" * 64,
    )
    pose = row["representations"]["seg_y1_plus_pose_y0_xor_y1"]["components"]["pose_enhancement_y0_xor_y1"]
    assert pose["nonzero_value_count"] == 0
    assert pose["runs"]["zero_fraction"] == 1.0


def test_run_is_resumable_without_reopening_completed_scientific_chunks(tmp_path: Path) -> None:
    config = _config()
    calls: list[int] = []

    def loader(chunk_index: int, pair_ids: tuple[int, ...]):
        calls.append(chunk_index)
        return _chunk(chunk_index, pair_ids)

    first = run_conditional_quotient_profile(
        config=config,
        input_binding=_binding(config),
        work_root=tmp_path / "profile",
        chunk_loader=loader,
    )
    assert calls == [0, 1]
    assert first["full_population_profiled"] is False
    assert first["conditional_budget_arbitration"]["frontier_feasibility_inference_allowed"] is False
    assert first["conditional_budget_arbitration"]["current_batch16_headroom_bytes_to_effective_frontier"] == 53_622
    assert set(first["downstream_hook_coverage"]) == {"1", "2", "3", "4", "5", "6"}
    assert first["downstream_hook_coverage"]["5"]["status"] == "BLOCKED_NONAUTHORITY_LOCAL_BATCH16"
    operator = first["functional_operator_proposal_surface"]
    assert operator["task_weighted_operator"]["status"] == "BLOCKED_MISSING_SCORER_COSTATE_EFFECTS"
    assert operator["hope_compatibility_fences"]["ph1_batchnorm_closed_forms_allowed"] is False
    assert operator["hope_compatibility_fences"]["static_parameter_count_as_rate_allowed"] is False
    assert "planning coordinate" in first["conditional_budget_arbitration"]["reason_frontier_inference_forbidden"]

    def forbidden_loader(_chunk_index: int, _pair_ids: tuple[int, ...]):
        raise AssertionError("completed chunk was reopened")

    resumed = run_conditional_quotient_profile(
        config=config,
        input_binding=_binding(config),
        work_root=tmp_path / "profile",
        chunk_loader=forbidden_loader,
    )
    assert resumed == first


def test_resume_refuses_tampered_immutable_chunk_checkpoint(tmp_path: Path) -> None:
    config = _config(pair_count=2)
    root = tmp_path / "profile"
    run_conditional_quotient_profile(
        config=config,
        input_binding=_binding(config),
        work_root=root,
        chunk_loader=_chunk,
    )
    stage = root / "stage_checkpoints" / "10_chunk_0000.json"
    value = json.loads(stage.read_bytes())
    value["score_claim"] = True
    stage.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ConditionalQuotientProfilerError, match="checkpoint contract"):
        run_conditional_quotient_profile(
            config=config,
            input_binding=_binding(config),
            work_root=root,
            chunk_loader=_chunk,
        )


def test_typed_configs_refuse_toy_geometry_without_explicit_test_gate_and_extras() -> None:
    with pytest.raises(ConditionalQuotientProfilerError, match="exact n600"):
        ConditionalQuotientProfileConfigV1(pair_count=4, chunk_pairs=2, scorer_hw=(4, 5))
    config = _config()
    mapping = config.as_mapping()
    assert mapping["schema"] == CONFIG_SCHEMA
    mapping["invented"] = True
    with pytest.raises(ConditionalQuotientProfilerError, match="keys/schema"):
        ConditionalQuotientProfileConfigV1.from_mapping(mapping)
    binding = _binding(config)
    binding["invented"] = True
    with pytest.raises(ConditionalQuotientProfilerError, match="keys/schema"):
        run_conditional_quotient_profile(
            config=config,
            input_binding=binding,
            work_root=Path("/not/used"),
            chunk_loader=_chunk,
        )
    binding = _binding(config)
    binding["current_planning_scorer_batch_size"] = 32
    binding["current_planning_matches_upstream_batch_geometry"] = False
    with pytest.raises(ConditionalQuotientProfilerError, match="current planning coordinate"):
        run_conditional_quotient_profile(
            config=config,
            input_binding=binding,
            work_root=Path("/not/used"),
            chunk_loader=_chunk,
        )


def test_input_binding_keeps_archive_identity_separate_from_fresh_derivation_proof() -> None:
    config = _config()
    binding = _binding(config)
    custody = binding["fresh_v15_derivation_custody"]
    assert isinstance(custody, dict)
    assert custody["adjacent_archive"]["sha256"] == binding["v15_archive_sha256"]
    custody["derivation_proof_separate_from_archive_content_identity"] = False
    with pytest.raises(ConditionalQuotientProfilerError, match="derivation custody boundary"):
        run_conditional_quotient_profile(
            config=config,
            input_binding=binding,
            work_root=Path("/not/used"),
            chunk_loader=_chunk,
        )
