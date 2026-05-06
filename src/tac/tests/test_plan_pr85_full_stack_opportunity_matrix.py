from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_pr85_full_stack_opportunity_matrix.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("plan_pr85_full_stack_matrix_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture_paths(tmp_path: Path) -> dict[str, list[str]]:
    bit_budget = _write_json(
        tmp_path / "bit_budget.json",
        {
            "bundle": {
                "segment_lengths": {
                    "mask": 1000,
                    "model": 400,
                    "post": 50,
                    "shift": 10,
                    "frac": 5,
                    "frac2": 5,
                    "frac3": 5,
                    "randmulti": 200,
                }
            },
            "planning_only": True,
            "score_claim": False,
        },
    )
    qma9_context = _write_json(
        tmp_path / "qma9_context.json",
        {
            "charged_baseline": {"mask_segment_bytes": 1000},
            "opportunity_ranking": [
                {
                    "model": "left_plus_up",
                    "break_even_overhead_bytes": -250,
                    "estimated_bytes_saved_lower_bound": -250,
                }
            ],
            "run_length_opportunities": {
                "row_sequences_along_width_axis": {"average_run_length": 20.0}
            },
            "planning_only": True,
            "score_claim": False,
        },
    )
    qma9_residual = _write_json(
        tmp_path / "qma9_residual.json",
        {
            "charged_baseline": {"mask_segment_bytes": 1000},
            "residual_programs": [
                {
                    "predictor": "left_zero_border",
                    "best_lower_bound_bytes": 1200.0,
                    "estimated_bytes_saved_vs_charged_mask": -200.0,
                    "planning_only": True,
                    "score_claim": False,
                }
            ],
            "planning_only": True,
            "score_claim": False,
        },
    )
    qh0 = _write_json(
        tmp_path / "qh0.json",
        {
            "blocker_class": "no_real_byte_win",
            "best_screened_candidate": {
                "candidate_id": "qh0_canonical_source_passthrough",
                "candidate_model_delta_bytes_vs_source": 0,
            },
            "built_candidate_count": 0,
            "planning_only": True,
            "score_claim": False,
        },
    )
    hpac = _write_json(
        tmp_path / "hpac.json",
        {
            "gross_byte_math": {"gross_mask_byte_opportunity": 123},
            "fail_closed_reasons": [
                {"gate": "pr86_full_decode_reencode_token_parity", "status": "failed_closed"}
            ],
            "planning_only": True,
            "score_claim": False,
            "dispatch_performed": False,
        },
    )
    baseline = _write_json(
        tmp_path
        / "experiments/results/lightning_batch/exact_eval_public_pr85_fixture/contest_auth_eval.json",
        {
            "archive_size_bytes": 1500,
            "avg_posenet_dist": 0.0001,
            "avg_segnet_dist": 0.0002,
            "n_samples": 600,
            "score_recomputed_from_components": 0.25,
        },
    )
    minus_randmulti = _write_json(
        tmp_path
        / "experiments/results/lightning_batch/exact_eval_pr85_minus_randmulti_fixture/contest_auth_eval.json",
        {
            "archive_size_bytes": 1300,
            "avg_posenet_dist": 0.0005,
            "avg_segnet_dist": 0.0003,
            "n_samples": 600,
            "score_recomputed_from_components": 0.31,
        },
    )
    randmulti_top001 = _write_json(
        tmp_path
        / "experiments/results/lightning_batch/exact_eval_pr85_randmulti_top001_fixture/contest_auth_eval.json",
        {
            "archive_size_bytes": 1320,
            "avg_posenet_dist": 0.00045,
            "avg_segnet_dist": 0.0003,
            "n_samples": 600,
            "score_recomputed_from_components": 0.30,
        },
    )
    minus_post = _write_json(
        tmp_path
        / "experiments/results/lightning_batch/exact_eval_pr85_minus_post_fixture/contest_auth_eval.json",
        {
            "archive_size_bytes": 1400,
            "avg_posenet_dist": 0.0007,
            "avg_segnet_dist": 0.0003,
            "n_samples": 600,
            "score_recomputed_from_components": 0.32,
        },
    )
    randmulti_summary = _write_json(
        tmp_path / "randmulti_summary.json",
        {
            "candidates": [
                {
                    "policy_id": "waterfill_top001",
                    "byte_delta_vs_source_archive": -180,
                    "score_claim": False,
                }
            ],
            "score_claim": False,
        },
    )
    post_summary = _write_json(
        tmp_path / "post_summary.json",
        {
            "candidates": [
                {
                    "policy_id": "preserve_motion_only",
                    "byte_delta_vs_source_archive": -12,
                    "score_claim": False,
                }
            ],
            "score_claim": False,
        },
    )
    pair_readiness = _write_json(
        tmp_path / "pair_readiness.json",
        {
            "dispatch_unlocked": False,
            "blocker_class": "missing_pair_action_spec",
            "top_pair_opportunities": [
                {
                    "pair_id": "pair_0192",
                    "break_even_bytes": 516.6922114573065,
                    "ranking_score": 0.00034404413500734173,
                }
            ],
            "planning_only": True,
            "score_claim": False,
        },
    )
    pair_action = _write_json(
        tmp_path / "pair_action_specs.json",
        {
            "schema": "pr85_pair_action_candidate_specs_v1",
            "dispatch_unlocked": False,
            "ready_for_exact_eval_after_lane_claim_count": 0,
            "candidate_count": 1,
            "blocker_class": "missing_pair_action_evidence",
            "blockers": [
                {
                    "blocker_class": "missing_pair_action_evidence",
                    "reason": "no grounded pair-action evidence JSON was supplied",
                }
            ],
            "candidates": [
                {
                    "candidate_id": "pr85_pair_0192_unlowered",
                    "selected_pairs": [
                        {
                            "atom_id": "fixture:pair_0192",
                            "pair_index": 192,
                            "component_signal": {
                                "combined_break_even_bytes": 516.6922114573065,
                            },
                        }
                    ],
                    "ready_for_exact_eval_after_lane_claim": False,
                    "blocker_class": "missing_explicit_pair_action",
                }
            ],
            "score_claim": False,
            "dispatch_performed": False,
        },
    )
    correction_recode = _write_json(
        tmp_path / "correction_recode.json",
        {
            "archive_candidate_count": 0,
            "best_byte_delta_vs_source_archive": 0,
            "exact_eval_unlocked": False,
            "result_class": "exact_local_negative_no_byte_winning_recode",
            "planning_only": True,
            "score_claim": False,
        },
    )
    qma9_native = _write_json(
        tmp_path / "qma9_native.json",
        {
            "candidate_count": 0,
            "best_byte_delta": None,
            "blockers": ["no_byte_positive_runtime_supported_qma9_native_grammar_candidate"],
            "planning_only": True,
            "score_claim": False,
        },
    )
    qma9_alt = _write_json(
        tmp_path / "qma9_alt.json",
        {
            "byte_positive_candidate_count": 0,
            "runtime_supported_byte_positive_candidate_count": 0,
            "best_alt_candidate": {
                "candidate_id": "adaptive9up2left2",
                "mode": "adaptive9up2left2",
                "payload_bytes": 161034,
                "delta_bytes_vs_source_qma9": 2023,
                "payload_changed_vs_source_qma9": True,
                "token_parity": {"verified": True},
            },
            "source_qma9": {"segment_bytes": 159011},
            "fail_closed": {
                "emitted": True,
                "reason": "no screened alternate grammar was byte-positive",
            },
            "planning_only": True,
            "score_claim": False,
            "dispatch_unlocked": False,
        },
    )
    qma9_run = _write_json(
        tmp_path / "qma9_run.json",
        {
            "byte_positive_candidate_count": 0,
            "runtime_supported_byte_positive_candidate_count": 0,
            "best_bytes_vs_pr85_qma9_159011B": {
                "best_mode": "row_rle_lzma6",
                "best_payload_bytes": 462176,
                "best_delta_bytes": 303165,
                "reference_bytes": 159011,
            },
            "source_qma9": {"segment_bytes": 159011},
            "planning_only": True,
            "score_claim": False,
            "dispatch_unlocked": False,
        },
    )
    hpac_probability = _write_json(
        tmp_path / "hpac_probability.json",
        {
            "status": "failed_closed",
            "byte_parity_variants": [],
            "source_contract_byte_parity_variants": [],
            "variant_results": [
                {
                    "probability_variant": {"name": "source_float32_perfect_false"},
                    "hpac_decode": {"decoded_symbol_count_before_failure": 30513},
                },
                {
                    "probability_variant": {"name": "source_float64_perfect_false"},
                    "hpac_decode": {"decoded_symbol_count_before_failure": 5951},
                },
            ],
            "score_claim": False,
            "dispatch_unlocked": False,
        },
    )
    escape = _write_json(
        tmp_path / "escape_manifest.json",
        {
            "decision": {"local_screen_negative": True},
            "subset": {"delta_bytes_vs_subset_qma9": 25},
            "dispatch_performed": False,
            "score_claim": False,
        },
    )
    return {
        "pr85_bit_budget": [str(bit_budget)],
        "qma9_context_entropy": [str(qma9_context)],
        "qma9_residual_sufficient_program": [str(qma9_residual)],
        "qh0_model_self_compression": [str(qh0)],
        "pr86_hpac_contract_plan": [str(hpac)],
        "pr85_exact_eval": [str(baseline), str(minus_randmulti), str(randmulti_top001), str(minus_post)],
        "pr85_randmulti_policy_summary": [str(randmulti_summary)],
        "pr85_post_motion_policy_summary": [str(post_summary)],
        "pr85_pair_atom_readiness": [str(pair_readiness)],
        "pr85_pair_action_specs": [str(pair_action)],
        "pr85_correction_recode_summary": [str(correction_recode)],
        "qma9_native_grammar_summary": [str(qma9_native)],
        "qma9_alt_grammar_summary": [str(qma9_alt)],
        "qma9_run_grammar_summary": [str(qma9_run)],
        "pr86_hpac_probability_variants": [str(hpac_probability)],
        "qma9_escape_screen_manifest": [str(escape)],
    }


def _by_family(matrix: dict) -> dict[str, dict]:
    return {row["family_id"]: row for row in matrix["opportunities"]}


def test_discovery_accepts_absolute_glob_patterns(tmp_path: Path) -> None:
    action_dir = tmp_path / "absolute_action"
    action = _write_json(
        action_dir / "candidate_specs.json",
        {
            "schema": "pr85_pair_action_candidate_specs_v1",
            "score_claim": False,
        },
    )

    found = module.discover_inputs(
        tmp_path,
        overrides={
            "pr85_pair_action_specs": [str(action_dir / "*.json")],
        },
    )

    assert found["pr85_pair_action_specs"] == [action]


def test_matrix_is_planning_only_and_marks_refuted_routes(tmp_path: Path) -> None:
    overrides = _fixture_paths(tmp_path)

    matrix = module.build_matrix(repo_root=tmp_path, overrides=overrides)
    records = _by_family(matrix)

    assert matrix["planning_only"] is True
    assert matrix["score_claim"] is False
    assert matrix["dispatch_performed"] is False
    assert matrix["remote_jobs_dispatched"] is False
    assert matrix["baseline_pr85_exact_eval"]["score"] == 0.25
    assert records["qh0_record_level_model_repack"]["already_refuted"] is True
    assert records["qh0_record_level_model_repack"]["blocked"] is True
    assert records["qh0_record_level_model_repack"]["exact_evidence_status"] == (
        "empirical_serializer_screen_no_real_byte_win"
    )
    assert records["qma9_simple_context_entropy_replacement"]["already_refuted"] is True
    assert records["qma9_residual_sufficient_program_density"]["blocked"] is True
    assert records["qma9_residual_sufficient_program_density"]["exact_evidence_status"] == (
        "empirical_residual_sufficient_program_direct_coder_negative"
    )
    assert any(
        note == "best_predictor=left_zero_border"
        for note in records["qma9_residual_sufficient_program_density"]["notes"]
    )
    assert records["qma9_block_copy_escape_screens"]["already_refuted"] is True
    assert records["protected_randmulti_group_waterfill"]["already_refuted"] is True
    assert records["protected_randmulti_group_waterfill"]["blocked"] is True
    assert records["protected_randmulti_group_waterfill"]["exact_evidence_status"] == (
        "exact_cuda_negative_full_600_samples"
    )
    assert records["protected_post_motion_group_policy"]["already_refuted"] is True
    assert records["protected_post_motion_group_policy"]["blocked"] is True
    assert records["protected_post_motion_group_policy"]["exact_evidence_status"] == (
        "exact_cuda_negative_full_600_samples"
    )
    assert records["whole_sidechannel_deletion_routes"]["exact_evidence_status"] == (
        "exact_cuda_negative_full_600_samples"
    )
    assert records["whole_sidechannel_deletion_routes"]["already_refuted"] is True
    assert records["scorer_gradient_pair_atom_policy"]["blocked"] is True
    assert records["scorer_gradient_pair_atom_policy"]["bytes_at_stake"] == 517
    assert records["scorer_gradient_pair_atom_policy"]["exact_evidence_status"] == (
        "fail_closed_pair_action_lowered_missing_grounded_action_evidence"
    )
    assert any(
        note == "pair_action_candidate_count=1"
        for note in records["scorer_gradient_pair_atom_policy"]["notes"]
    )
    assert records["decoded_parity_correction_stream_recode"]["already_refuted"] is True
    assert records["decoded_parity_correction_stream_recode"]["blocked"] is True
    assert records["decoded_parity_correction_stream_recode"]["exact_evidence_status"] == (
        "empirical_decoded_parity_recode_no_byte_win"
    )
    assert records["qma9_native_runtime_supported_grammar_screen"]["already_refuted"] is True
    assert records["qma9_native_runtime_supported_grammar_screen"]["blocked"] is True
    assert records["qma9_alternate_neighbor_table_grammar_screen"]["already_refuted"] is True
    assert records["qma9_alternate_neighbor_table_grammar_screen"]["blocked"] is True
    assert records["qma9_alternate_neighbor_table_grammar_screen"]["exact_evidence_status"] == (
        "empirical_alt_grammar_full_stream_no_byte_win"
    )
    assert records["qma9_qrg1_row_run_grammar_screen"]["already_refuted"] is True
    assert records["qma9_qrg1_row_run_grammar_screen"]["blocked"] is True
    assert records["qma9_qrg1_row_run_grammar_screen"]["exact_evidence_status"] == (
        "empirical_qrg1_full_stream_no_byte_win"
    )
    assert records["pr86_hpac_probability_contract_variants"]["already_refuted"] is True
    assert records["pr86_hpac_probability_contract_variants"]["blocked"] is True
    assert records["pr86_hpac_probability_contract_variants"]["exact_evidence_status"] == (
        "fail_closed_probability_variants_no_full_decode"
    )
    assert all(not row["family_id"].startswith("whole_") for row in matrix["top_stack_plans"])
    assert all(row["family_id"] != "protected_randmulti_group_waterfill" for row in matrix["top_stack_plans"])
    assert all(row["family_id"] != "protected_post_motion_group_policy" for row in matrix["top_stack_plans"])
    assert all(row["family_id"] != "qh0_record_level_model_repack" for row in matrix["top_stack_plans"])
    assert all(row["family_id"] != "decoded_parity_correction_stream_recode" for row in matrix["top_stack_plans"])
    assert all(row["family_id"] != "qma9_native_runtime_supported_grammar_screen" for row in matrix["top_stack_plans"])
    assert all(row["family_id"] != "qma9_alternate_neighbor_table_grammar_screen" for row in matrix["top_stack_plans"])
    assert all(row["family_id"] != "qma9_qrg1_row_run_grammar_screen" for row in matrix["top_stack_plans"])
    assert all(row["family_id"] != "pr86_hpac_probability_contract_variants" for row in matrix["top_stack_plans"])


def test_matrix_digest_is_stable_for_same_inputs(tmp_path: Path) -> None:
    overrides = _fixture_paths(tmp_path)

    first = module.build_matrix(repo_root=tmp_path, overrides=overrides)
    second = module.build_matrix(repo_root=tmp_path, overrides=overrides)

    assert first["stable_matrix_digest_sha256"] == second["stable_matrix_digest_sha256"]
    assert first == second


def test_write_outputs_emit_json_and_markdown(tmp_path: Path) -> None:
    overrides = _fixture_paths(tmp_path)
    matrix = module.build_matrix(repo_root=tmp_path, overrides=overrides)

    outputs = module.write_outputs(matrix, tmp_path / "out")
    json_path = REPO / outputs["json"] if not Path(outputs["json"]).is_absolute() else Path(outputs["json"])
    if not json_path.exists():
        json_path = tmp_path / "out" / "pr85_full_stack_opportunity_matrix.json"
    md_path = tmp_path / "out" / "pr85_full_stack_opportunity_matrix.md"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert payload["planning_only"] is True
    assert payload["score_claim"] is False
    assert "PR85 Full-Stack Opportunity Matrix" in markdown
    assert "score_claim: false" in markdown
