from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tac.witness_dsl import g1_prior_signal_harvest as harvest_module
from tac.witness_dsl.g1_prior_signal_harvest import (
    CENSUS_BODY_SCHEMA,
    CENSUS_ENVELOPE_SCHEMA,
    PBR2_SCHEMA,
    SCHEMA,
    V10_LATTICE_SCHEMA,
    V13_SCHEMA,
    V14_SCHEMA,
    V19C_SCHEMA,
    G1SignalHarvestError,
    _write_once,
    build_signal_harvest_body,
    canonical_json_bytes,
    make_envelope,
    validate_envelope,
)


def _documents() -> tuple[dict, dict, dict, dict, dict, dict]:
    pbr2 = {
        "schema": PBR2_SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "candidate_payload_allowed": False,
        "exact_difference": {
            "mismatch_cells": 9,
            "total_cells": 24,
            "mismatch_fraction": 0.375,
        },
        "pbr2": {
            "schema": "tac.progressive_geometry_residual.v3",
            "pbr2_is_target_derived": True,
            "candidate_archive_admissible": False,
            "exact_target_semantic_reconstruction": True,
            "pbr2_reconstructs_exact_gt_argmax": True,
            "target_derived_residual_promotion_admitted": False,
            "promotion_eligible": False,
            "research_only": True,
            "score_claim": False,
            "decode_scorer_dependency": False,
            "artifact_role": "encoder_side_conditional_entropy_measurement",
            "candidate_archive_blocker": "lossless predictor-conditional target-semantic-table encoding",
            "generic_apply_requires_external_predictor_semantics": True,
            "physical_prefix_decode_supported": False,
            "staged_application_requires_complete_packet": True,
            "target_semantic_lineage": "frozen_gt_argmax",
            "predictor_program_sha256": "3" * 64,
            "predictor_semantic_sha256": "4" * 64,
            "target_semantic_sha256": "5" * 64,
            "source_pair_start": 10,
            "source_pair_stop_exclusive": 12,
            "initial_error_cells": 9,
            "final_error_cells": 0,
            "pbr2_event_count": 9,
            "pbr2_event_density_numerator": 9,
            "pbr2_event_density_denominator": 24,
            "reconstructed_target_semantic_bytes": 24,
            "separate_dense_target_table_section_bytes": 0,
            "packet_prefix_header_bytes": 1,
            "header_bytes": 2,
            "crc_bytes": 1,
            "pbr2_target_derived_section_bytes": 19,
            "packet_bytes": 23,
            "packet_sha256": "a" * 64,
            "strata": [
                {
                    "order": 1,
                    "name": "temporal_boundary",
                    "mode": "block_context",
                    "corrected_cells": 7,
                    "errors_before": 9,
                    "errors_after": 2,
                    "record_count": 3,
                    "span_count": 3,
                    "payload_bytes": 11,
                    "payload_sha256": "0" * 64,
                },
                {
                    "order": 2,
                    "name": "component_islands",
                    "mode": "connected_row_spans_8",
                    "corrected_cells": 1,
                    "errors_before": 2,
                    "errors_after": 1,
                    "record_count": 1,
                    "span_count": 2,
                    "payload_bytes": 5,
                    "payload_sha256": "1" * 64,
                },
                {
                    "order": 3,
                    "name": "sparse_tail",
                    "mode": "singleton_delta_events",
                    "corrected_cells": 1,
                    "errors_before": 1,
                    "errors_after": 0,
                    "record_count": 1,
                    "span_count": 1,
                    "payload_bytes": 3,
                    "payload_sha256": "2" * 64,
                },
            ],
        },
        "inputs": {"gt_cache": {"sha256": "f" * 64, "bytes": 1234}},
        "receiver_closure": {
            "candidate_payload_allowed": False,
            "exact_target_recovered_without_gt_cache_at_decode": True,
            "predictor_semantics_rederived_from_counted_program": True,
        },
    }
    v13 = {
        "schema": V13_SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "execution_allowed": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "selected_rung": "islands",
        "composition_ladder": [
            {
                "rung": "islands",
                "archive": {"bytes": 123, "sha256": "b" * 64},
                "bridge": {
                    "segmentation": {"d_seg": "0.1"},
                    "pose": {"d_pose": "2.0"},
                },
            }
        ],
        "natural_production_inventory": {
            "movable_g1": {
                "payload_bytes": 17,
                "payload_sha256": "c" * 64,
                "decoded_mask_errors": 1,
                "decoded_clean_rest_dseg": 0.01,
                "births": 1,
                "deaths": 0,
                "persists": 1,
                "vertices": 4,
            }
        },
        "falsifier": {"binding_mechanism": "receiver_projection"},
        "target_custody": {"cache_sha256": "f" * 64, "cache_bytes": 1234},
    }
    v14 = {
        "schema": V14_SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "execution_allowed": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "selected_candidate": "islands",
        "fixed_ladder": [
            {
                "candidate": "islands",
                "archive_bytes": 130,
                "d_seg": "0.08",
                "d_pose": "2.1",
            }
        ],
        "diagnostics": {
            "measured_mechanism": "projection",
            "lane_windows": [
                {
                    "source_pair_id": 10,
                    "local_pair_id": 0,
                    "delta_d_seg": "-0.01",
                    "fixed_islands_d_seg": "0.1",
                    "fixed_both_d_seg": "0.09",
                },
                {
                    "source_pair_id": 99,
                    "local_pair_id": 89,
                    "delta_d_seg": "1.0",
                    "fixed_islands_d_seg": "1.0",
                    "fixed_both_d_seg": "2.0",
                },
            ],
        },
        "target_custody": {"sha256": "f" * 64, "bytes": 1234},
    }
    v19c = {
        "schema": V19C_SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "execution_allowed": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "curve": {
            "n600_endpoint": {
                "archive_bytes": 140,
                "archive_sha256": "d" * 64,
                "d_seg": "0.07",
                "d_pose": "2.2",
                "joint_delta_vs_v19b": {"joint_delta": -0.2},
            }
        },
        "family_attribution": {
            "n600": {
                "small": {
                    "admitted": 3,
                    "compile_infeasible": 0,
                    "proposals_measured_or_classified": 4,
                    "strict_joint_gain": 0.01,
                },
                "large": {
                    "admitted": 2,
                    "compile_infeasible": 1,
                    "proposals_measured_or_classified": 5,
                    "strict_joint_gain": 0.1,
                },
            }
        },
        "asymptote": {
            "unique_coordinate_inventory": 9,
            "dev_proposals": 20,
            "dev_admissions": 5,
            "n600_admissions": 4,
            "consecutive_failures_at_stop": 8,
            "family_optimum_claimed": False,
        },
        "c1_bucket_attribution": {"v19c_incremental_realized_net_flips": {"residual_fraction": 0.75}},
    }
    v10_lattice = {
        "schema": V10_LATTICE_SCHEMA,
        "axis": "[macOS-CPU advisory subset]",
        "authority": {
            "score_claim": False,
            "pointer_moved": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "subset_non_promotable": True,
            "verdict_scope": (
                "selected real n600-cache pairs, frozen CPU SegNet, uint8 frame1/A factor only; "
                "no PoseNet, receiver archive, contest CPU/CUDA, or full-n600 claim"
            ),
        },
        "configuration": {
            "pair_ids": [1, 2, 3, 4, 5, 6],
            "camera_hw": [874, 1164],
            "scorer_hw": [384, 512],
            "input_hashes": {"gt_n600_npz_sha256": "f" * 64},
        },
        "aggregate": {
            "arms": {
                "exact_uint8_lattice_candidate": {
                    "d_seg": 0.0,
                    "mismatched_pixels": 0,
                    "total_pixels": 1_179_648,
                    "per_class": {
                        "0": {"d_seg": 0.0, "mismatched_pixels": 0, "target_pixels": 1_179_644},
                        "1": {"d_seg": 0.0, "mismatched_pixels": 0, "target_pixels": 1},
                        "2": {"d_seg": 0.0, "mismatched_pixels": 0, "target_pixels": 1},
                        "3": {"d_seg": 0.0, "mismatched_pixels": 0, "target_pixels": 1},
                        "4": {"d_seg": 0.0, "mismatched_pixels": 0, "target_pixels": 1},
                    },
                }
            },
            "exact_search": {
                "aggregate_statuses": ["FEASIBLE_EXACT"],
                "certified_exact_frames": 6,
                "decoded_frames_with_exact_numerator_equality": 6,
                "exact_blocks": 3_538_944,
                "exact_candidate_blocks": 3_538_944,
                "budget_blocks": 0,
                "heuristic_blocks": 0,
                "nonzero_decoded_numerator_residual_cells": 0,
                "nodes_visited": 123,
                "max_abs_decoded_numerator_residual": 0,
            },
        },
        "sidecar": {
            "bytes": 456,
            "sha256": "e" * 64,
            "frame_count": 6,
            "parse_back_all_frame_hashes_match": True,
            "honest_name": "incremental uint8 lattice feasibility sidecar; NOT a contest archive",
            "candidate_payload_allowed": False,
        },
        "remaining_blockers": ["PoseNet/both-frame interaction"],
    }
    census_body = {
        "schema": CENSUS_BODY_SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "input_custody": {"sha256": "f" * 64, "bytes": 1234},
        "measurements": {
            "aggregate": {
                "total_sites": 24,
                "temporal_changed_sites": {"sum": 2},
                "temporal_changed_fraction": 1 / 12,
                "row_runs": {"sum": 8},
            },
            "temporal_interpretation": {"source_frame_stride": 2},
        },
    }
    census = {
        "schema": CENSUS_ENVELOPE_SCHEMA,
        "body": census_body,
        "body_sha256": hashlib.sha256(canonical_json_bytes(census_body)).hexdigest(),
    }
    return pbr2, v13, v14, v19c, v10_lattice, census


def _custody() -> dict[str, dict[str, object]]:
    custody: dict[str, dict[str, object]] = {
        name: {"sha256": hashlib.sha256(name.encode()).hexdigest(), "bytes": len(name)}
        for name in ("pbr2", "v13", "v14", "v19c", "v10_lattice", "target_partition_census")
    }
    custody["pbr2_packet"] = {"sha256": "a" * 64, "bytes": 23}
    return custody


def _producer_custody() -> dict[str, object]:
    return {"module": {"sha256": "b" * 64}, "git_head": "c" * 40}


def _pbr2_accounting(pbr2: dict) -> dict:
    return copy.deepcopy(pbr2["pbr2"])


def _body() -> dict:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    return build_signal_harvest_body(
        pbr2=pbr2,
        v13=v13,
        v14=v14,
        v19c=v19c,
        v10_lattice=v10_lattice,
        census_envelope=census,
        input_custody=_custody(),
        pbr2_packet_accounting=_pbr2_accounting(pbr2),
        producer_custody=_producer_custody(),
        semantic_argv=["fixture"],
    )


def test_harvest_preserves_teacher_boundary_and_ranks_existing_priors() -> None:
    body = _body()
    assert body["schema"] == SCHEMA
    assert body["candidate_payload_emitted"] is False
    assert body["teacher"]["candidate_payload_allowed"] is False
    assert body["composition_decision"]["pbr_packets_in_candidate"] is False
    assert body["composition_decision"]["compact_generator_exact_target_output_allowed"] is True
    assert body["composition_decision"]["independent_component_admission_thresholds"] is False
    assert body["composition_decision"]["archive_byte_delta_is_signed"] is True
    assert body["composition_decision"]["counted_preimage_object"] == ("compact_Y0_Y1_obligation_generator_program")
    assert body["inverse_preimage_prior"]["exact_blocks"] == 3_538_944
    assert body["inverse_preimage_prior"]["dense_measurement_sidecar"]["candidate_payload_allowed_by_harvest"] is False
    rows = body["correction_saturation_prior"]["family_priors_ranked_by_measured_n600_strict_gain"]
    assert [row["family"] for row in rows] == ["large", "small"]
    anchors = body["realization_prior"]["pbr2_window_anchor_rows"]
    assert anchors == [
        {
            "source_pair_id": 10,
            "local_pair_id": 0,
            "delta_d_seg": "-0.01",
            "fixed_islands_d_seg": "0.1",
            "fixed_both_d_seg": "0.09",
        }
    ]


@pytest.mark.parametrize(
    ("document_index", "field", "value"),
    [
        (0, "candidate_payload_allowed", True),
        (1, "pointer_moved", True),
        (2, "score_claim", True),
        (3, "research_only", False),
    ],
)
def test_harvest_refuses_false_authority(document_index: int, field: str, value: object) -> None:
    documents = list(_documents())
    document = copy.deepcopy(documents[document_index])
    document[field] = value
    documents[document_index] = document
    with pytest.raises(G1SignalHarvestError):
        build_signal_harvest_body(
            pbr2=documents[0],
            v13=documents[1],
            v14=documents[2],
            v19c=documents[3],
            v10_lattice=documents[4],
            census_envelope=documents[5],
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(documents[0]),
        )


def test_envelope_hash_is_exact_and_mutation_refuses() -> None:
    envelope = make_envelope(_body())
    validate_envelope(envelope, reopen_sources=False)
    encoded = canonical_json_bytes(envelope)
    assert json.loads(encoded) == envelope
    changed = copy.deepcopy(envelope)
    changed["body"]["pointer_moved"] = True
    with pytest.raises(G1SignalHarvestError, match="body hash mismatch"):
        validate_envelope(changed)

    escalated = copy.deepcopy(envelope["body"])
    escalated["score_claim"] = True
    escalated["promotion_eligible"] = True
    escalated["pointer_moved"] = True
    with pytest.raises(G1SignalHarvestError, match="authority boundary"):
        validate_envelope(make_envelope(escalated), reopen_sources=False)

    nested_axis = copy.deepcopy(envelope["body"])
    nested_axis["worldsheet_prior"]["authority"] = "[contest-CPU]"
    with pytest.raises(G1SignalHarvestError, match="evidence-axis boundary"):
        validate_envelope(make_envelope(nested_axis), reopen_sources=False)

    nested_payload = copy.deepcopy(envelope["body"])
    nested_payload["inverse_preimage_prior"]["dense_measurement_sidecar"]["source_candidate_payload_allowed"] = True
    with pytest.raises(G1SignalHarvestError, match="candidate-lineage boundary"):
        validate_envelope(make_envelope(nested_payload), reopen_sources=False)


def test_census_body_hash_and_pair_window_are_required() -> None:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    broken = copy.deepcopy(census)
    broken["body_sha256"] = "0" * 64
    with pytest.raises(G1SignalHarvestError, match="census body hash"):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=v10_lattice,
            census_envelope=broken,
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(pbr2),
        )

    no_overlap = copy.deepcopy(v14)
    no_overlap["diagnostics"]["lane_windows"][0]["source_pair_id"] = 100
    with pytest.raises(G1SignalHarvestError, match="no measured anchor"):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=no_overlap,
            v19c=v19c,
            v10_lattice=v10_lattice,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(pbr2),
        )


def test_v10_lattice_prior_refuses_false_authority_and_nonexact_evidence() -> None:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    false_authority = copy.deepcopy(v10_lattice)
    false_authority["authority"]["score_claim"] = True
    with pytest.raises(G1SignalHarvestError, match="score_claim=false"):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=false_authority,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(pbr2),
        )

    inexact = copy.deepcopy(v10_lattice)
    inexact["aggregate"]["exact_search"]["heuristic_blocks"] = 1
    with pytest.raises(G1SignalHarvestError, match="does not prove"):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=inexact,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(pbr2),
        )


def test_v10_lattice_prior_refuses_contradictory_frame_closure() -> None:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    contradictory = copy.deepcopy(v10_lattice)
    contradictory["configuration"]["pair_ids"] = []
    contradictory["aggregate"]["exact_search"]["certified_exact_frames"] = 999
    contradictory["aggregate"]["exact_search"]["max_abs_decoded_numerator_residual"] = 17
    with pytest.raises(G1SignalHarvestError, match="pair_ids"):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=contradictory,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(pbr2),
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("aggregate", "exact_search", "certified_exact_frames"), 6.0, "exact integer"),
        (("aggregate", "exact_search", "max_abs_decoded_numerator_residual"), False, "exact integer"),
        (("aggregate", "arms", "exact_uint8_lattice_candidate", "mismatched_pixels"), False, "exact integer"),
        (("aggregate", "arms", "exact_uint8_lattice_candidate", "d_seg"), False, "numeric zero"),
    ],
)
def test_v10_lattice_prior_refuses_json_numeric_type_confusion(
    path: tuple[str, ...], value: object, message: str
) -> None:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    cursor = v10_lattice
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(G1SignalHarvestError, match=message):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=v10_lattice,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(pbr2),
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("configuration", "scorer_hw"), [384.0, 512.0], "exact integer list"),
        (("sidecar", "frame_count"), 6.0, "exact integer"),
        (("authority", "verdict_scope"), "[contest-CPU]", "verdict_scope"),
    ],
)
def test_v10_lattice_prior_refuses_geometry_and_authority_type_laundering(
    path: tuple[str, ...], value: object, message: str
) -> None:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    cursor = v10_lattice
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(G1SignalHarvestError, match=message):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=v10_lattice,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(pbr2),
        )


def test_pbr2_receipt_window_must_match_strict_packet_accounting() -> None:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    accounting = _pbr2_accounting(pbr2)
    pbr2["pbr2"]["source_pair_stop_exclusive"] = 13
    with pytest.raises(G1SignalHarvestError, match="strict packet contents"):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=v10_lattice,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=accounting,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("packet_bytes", 23.0),
        ("final_error_cells", False),
        ("initial_error_cells", 9.0),
    ],
)
def test_pbr2_receipt_comparison_is_json_type_exact(field: str, value: object) -> None:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    accounting = _pbr2_accounting(pbr2)
    pbr2["pbr2"][field] = value
    with pytest.raises(G1SignalHarvestError, match="strict packet contents"):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=v10_lattice,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=accounting,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("synthetic_lineage", "target_semantic_lineage"),
        ("broken_stratum", "stratum error/byte continuity"),
        ("foreign_target_cache", "frozen target custody"),
    ],
)
def test_teacher_lineage_and_arithmetic_are_closed(mutation: str, message: str) -> None:
    pbr2, v13, v14, v19c, v10_lattice, census = _documents()
    if mutation == "synthetic_lineage":
        pbr2["pbr2"]["target_semantic_lineage"] = "synthetic_fixture"
    elif mutation == "broken_stratum":
        pbr2["pbr2"]["strata"][1]["errors_before"] = 3
    else:
        census_body = census["body"]
        census_body["input_custody"]["sha256"] = "9" * 64
        census["body_sha256"] = hashlib.sha256(canonical_json_bytes(census_body)).hexdigest()
    with pytest.raises(G1SignalHarvestError, match=message):
        build_signal_harvest_body(
            pbr2=pbr2,
            v13=v13,
            v14=v14,
            v19c=v19c,
            v10_lattice=v10_lattice,
            census_envelope=census,
            input_custody=_custody(),
            pbr2_packet_accounting=_pbr2_accounting(pbr2),
        )


def test_write_once_is_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "harvest.json"
    payload = canonical_json_bytes(make_envelope(_body()))
    _write_once(output, payload, reopen_sources=False)
    _write_once(output, payload, reopen_sources=False)
    assert output.read_bytes() == payload
    assert not list(tmp_path.glob(".harvest.json.*.tmp"))

    different_body = _body()
    different_body["verdict"] = "different"
    different = canonical_json_bytes(make_envelope(different_body))
    with pytest.raises(G1SignalHarvestError, match="different bytes"):
        _write_once(output, different, reopen_sources=False)
    assert output.read_bytes() == payload


def test_existing_harvest_is_stable_reopened_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "harvest.json"
    payload = canonical_json_bytes(make_envelope(_body()))
    output.write_bytes(payload)
    real_read = harvest_module._read_regular_bytes
    reads = 0

    def racing_read(path: Path, *, max_bytes: int = 64 << 20):
        nonlocal reads
        result = real_read(path, max_bytes=max_bytes)
        if Path(path) == output and reads == 0:
            output.write_bytes(b"peer-replaced-after-compare")
        reads += 1
        return result

    monkeypatch.setattr(harvest_module, "_read_regular_bytes", racing_read)
    with pytest.raises(G1SignalHarvestError, match="changed during validation"):
        _write_once(output, payload, reopen_sources=False)
