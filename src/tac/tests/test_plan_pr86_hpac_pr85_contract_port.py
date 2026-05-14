# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_pr86_hpac_pr85_contract_port.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("plan_pr86_hpac_pr85_contract_port_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _pr86_anatomy(*, exact_score: bool = False) -> dict:
    replay = {
        "archive_identity": {
            "archive_sha256": module.EXPECTED_PR86_ARCHIVE_SHA256,
            "archive_size_bytes": module.EXPECTED_PR86_ARCHIVE_BYTES,
        },
        "evidence_grade": "invalid",
        "score_claim": False,
        "status": "archive_validator_whitelist_blocked",
    }
    if exact_score:
        replay = {
            "archive_size_bytes": module.EXPECTED_PR86_ARCHIVE_BYTES,
            "avg_posenet_dist": 0.0001,
            "avg_segnet_dist": 0.0002,
            "evidence_grade": "exact_score_json_local_mirror",
            "n_samples": 600,
            "score_claim": False,
            "score_delta_vs_pr85": -0.001,
            "score_recomputed_from_components": module.PR85_T4_BEST_SCORE - 0.001,
            "status": "score_json_present",
        }
    return {
        "archive_member_contract": {
            "promotable_member_contract": True,
            "members": [
                {"file_size": size, "name": name, "sha256": name.replace(".", "")[:4]}
                for name, size in module.EXPECTED_PR86_MEMBER_BYTES.items()
            ],
        },
        "current_exact_replay_status": replay,
        "member_payload_layers": {
            "members": [
                {
                    "encoded_bytes": module.EXPECTED_PR86_MEMBER_BYTES["tokens.bin"],
                    "name": "tokens.bin",
                    "queue_word_dtype": "uint32",
                    "status": "structurally_parseable",
                    "uint32_aligned": True,
                    "uint32_word_count": module.EXPECTED_PR86_TOKEN_WORDS,
                }
            ]
        },
        "public_pr86_reference": {
            "claimed_report_values_external": {"display_score": 0.123456},
            "score_claim_in_this_artifact": False,
        },
        "score_claim": False,
        "source_archive": {
            "bytes": module.EXPECTED_PR86_ARCHIVE_BYTES,
            "expected_identity_match": True,
            "expected_sha256": module.EXPECTED_PR86_ARCHIVE_SHA256,
            "path": "fixture/pr86/archive.zip",
            "sha256": module.EXPECTED_PR86_ARCHIVE_SHA256,
        },
        "token_hpac_decode_contract": {
            "archive_write_tokens_second_arg": "gt",
            "categorical_perfect_false_present": True,
            "inflate_reconstructs_residuals": False,
            "probability_clip_eps": "1e-7",
            "range_decoder_api_present": True,
            "range_encoder_api_present": True,
            "submitted_archive_token_encoding": "raw_tokens",
            "training_objective": "residual_tokens",
        },
    }


def _pr85_profile() -> dict:
    segments = [
        {
            "bytes": module.EXPECTED_PR85_MASK_BYTES,
            "magic_ascii": "QMA9X\\u0002",
            "name": "mask",
            "sha256": "4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179",
        },
        {"bytes": 57074, "name": "model", "sha256": "m" * 64},
        {"bytes": 1487, "name": "pose", "sha256": "p" * 64},
        {"bytes": 1400, "name": "post", "sha256": "o" * 64},
        {"bytes": 226, "name": "shift", "sha256": "s" * 64},
        {"bytes": 106, "name": "frac", "sha256": "f" * 64},
        {"bytes": 149, "name": "frac2", "sha256": "g" * 64},
        {"bytes": 154, "name": "frac3", "sha256": "h" * 64},
        {"bytes": 223, "name": "bias", "sha256": "b" * 64},
        {"bytes": 273, "name": "region", "sha256": "r" * 64},
        {"bytes": 16101, "name": "randmulti", "sha256": "a" * 64},
    ]
    return {
        "archive": {
            "archive_sha256": module.EXPECTED_PR85_ARCHIVE_SHA256,
            "archive_size_bytes": module.EXPECTED_PR85_ARCHIVE_BYTES,
            "member_name": "x",
        },
        "bundle_format": "pr85_v5_micro_24bit_lengths_fixed_bias_region",
        "score_claim": False,
        "segments": segments,
    }


def _parity(*, token_parity: bool = False, pr85_transfer: bool = False) -> dict:
    byte_status = "passed" if token_parity else "not_run"
    pr85_status = "passed" if pr85_transfer else "not_run"
    return {
        "conclusions": {
            "own_stream_decode_status": (
                "full_archive_decode_passed" if token_parity else "bounded_prefix_decodes"
            )
        },
        "decode_probes": [
            {
                "full_archive_decode": token_parity,
                "status": "passed" if token_parity else "skipped",
            }
        ],
        "planning_only": True,
        "required_gates_before_hpac_transfer_to_pr85": [
            {"gate": "byte_exact_reencode", "status": byte_status},
            {"gate": "pr85_transfer_parity", "status": pr85_status},
        ],
        "score_claim": False,
    }


def _pr85_token_source() -> dict:
    return {
        "exactness": {"raw_tensor_exact": True},
        "mask_segment_identity": {
            "bytes": module.EXPECTED_PR85_MASK_BYTES,
            "sha256": "4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179",
        },
        "planning_only": True,
        "score_claim": False,
        "token_source": {
            "bytes": 600 * 512 * 384,
            "dtype": "uint8",
            "extracted": True,
            "invalid_symbol_values": [],
            "observed_range": {"min": 0, "max": 4},
            "sha256": "c" * 64,
            "shape": [600, 512, 384],
        },
    }


def _pr85_hpac_parity(
    *,
    passed: bool = False,
    noop: bool = False,
    candidate_token_sha: str = "c" * 64,
    candidate_mask_sha: str = "d" * 64,
) -> dict:
    if not passed:
        return {
            "dispatch_performed": False,
            "evidence_grade": "local_decode_negative",
            "failure_class": "pr86_hpac_decode_contract_or_dependency_mismatch",
            "observed_error": (
                "AssertionError: Tried to decode from compressed data that is invalid for "
                "the employed entropy model."
            ),
            "pr85_decoded_sha256": "0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45",
            "pr85_decoded_shape": [600, 384, 512],
            "score_claim": False,
            "status": "blocked_entropy_decode_assertion",
        }
    return {
        "candidate": {
            "archive_byte_closed": True,
            "byte_exact_token_parity": True,
            "decoded_token_sha256": candidate_token_sha,
            "decoded_token_shape": [600, 512, 384],
            "digests": {"archive_sha256": "a" * 64},
            "mask_replacement_kind": "hpac_pr85_mask_replacement",
            "mask_segment_sha256": candidate_mask_sha,
            "noop": noop,
            "paths": ["experiments/results/fixture_pr86_hpac_pr85/archive.zip"],
            "runtime_output_parity": True,
        },
        "dispatch_performed": False,
        "score_claim": False,
        "status": "passed",
    }


def _probability_matrix() -> dict:
    return {
        "byte_parity_variants": [],
        "dispatch_unlocked": False,
        "failure_reason": "no_source_contract_variant_full_decode_byte_exact_reencode",
        "source_contract_byte_parity_variants": [],
        "status": "failed_closed",
        "variant_results": [
            {
                "byte_parity_achieved": False,
                "dispatch_unlocked": False,
                "failure_context": {
                    "decoded_symbol_count_before_failure": 5951,
                    "failed_at": {"frame": 0, "group": 10, "symbol_in_group": 191},
                },
                "failure_reason": "hpac_entropy_decode_contract_mismatch",
                "failure_stage": "submitted_tokens_decode",
                "probability_variant": {
                    "name": "source_float64_perfect_false",
                    "source_contract": True,
                },
                "status": "failed_closed",
            }
        ],
    }


def _inputs(
    tmp_path: Path,
    *,
    exact_score: bool = False,
    token_parity: bool = False,
    pr85_transfer: bool = False,
    hpac_parity: dict | None = None,
    probability_matrix: dict | None = None,
) -> dict[str, Path]:
    anatomy = tmp_path / "pr86_anatomy.json"
    profile = tmp_path / "pr85_profile.json"
    parity = tmp_path / "pr86_parity.json"
    token_source = tmp_path / "pr85_token_source.json"
    hpac_parity_path = tmp_path / "pr85_hpac_parity.json"
    probability_matrix_path = tmp_path / "pr86_probability_matrix.json"
    _write_json(anatomy, _pr86_anatomy(exact_score=exact_score))
    _write_json(profile, _pr85_profile())
    _write_json(parity, _parity(token_parity=token_parity, pr85_transfer=pr85_transfer))
    _write_json(token_source, _pr85_token_source())
    _write_json(hpac_parity_path, hpac_parity or _pr85_hpac_parity())
    _write_json(probability_matrix_path, probability_matrix or _probability_matrix())
    return {
        "anatomy": anatomy,
        "profile": profile,
        "parity": parity,
        "token_source": token_source,
        "hpac_parity": hpac_parity_path,
        "probability_matrix": probability_matrix_path,
    }


def test_planner_refuses_score_claims_even_with_external_pr_score_text(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=None,
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
        request_score_claim=True,
    )

    assert plan["score_claim"] is False
    assert plan["score_claim_refusal"]["status"] == "refused"
    assert plan["score_claim_refusal"]["requested_score_claim"] is True
    assert plan["pr85_target_reference"]["score_claim_from_this_plan"] is False
    assert plan["gross_byte_math"]["score_claim"] is False


def test_dispatchable_request_fails_closed_without_exact_score_and_token_parity(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=None,
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
        request_dispatchable=True,
    )

    assert plan["dispatchable"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["dispatchability"]["status"] == "refused_fail_closed"
    refusal_gates = {row["gate"] for row in plan["dispatchability"]["refusal_reasons"]}
    assert "pr86_exact_score_evidence" in refusal_gates
    assert "pr86_full_decode_reencode_token_parity" in refusal_gates
    assert "pr85_hpac_token_parity" in refusal_gates


def test_dispatchable_request_still_refuses_when_score_exists_but_token_parity_is_missing(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path, exact_score=True, token_parity=False, pr85_transfer=False)

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=None,
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
        request_dispatchable=True,
    )

    gates = {row["id"]: row for row in plan["required_parity_gates"]}
    assert gates["pr86_exact_score_evidence"]["passed"] is True
    assert gates["pr86_full_decode_reencode_token_parity"]["passed"] is False
    assert gates["pr85_hpac_token_parity"]["passed"] is False
    assert plan["dispatchable"] is False
    refusal_gates = {row["gate"] for row in plan["dispatchability"]["refusal_reasons"]}
    assert "pr86_exact_score_evidence" not in refusal_gates
    assert "pr86_full_decode_reencode_token_parity" in refusal_gates
    assert "pr85_hpac_token_parity" in refusal_gates


def test_gross_mask_opportunity_and_next_gate_are_recorded(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=None,
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
    )

    assert plan["token_stream_contract"]["stream_dtype"] == "uint32"
    assert plan["token_stream_contract"]["uint32_word_count"] == 28475
    assert plan["gross_byte_math"]["pr85_mask_segment_bytes"] == 159011
    assert plan["gross_byte_math"]["pr86_hpac_tokens_meta_bytes"] == 143642
    assert plan["gross_byte_math"]["gross_mask_byte_opportunity"] == 15369
    assert plan["gross_byte_math"]["gross_saved_bytes_if_same_contract"] == 15369
    assert plan["next_gate"]["id"] == "pr86_full_decode_reencode_token_parity"


def test_pr85_baseline_token_extraction_gate_passes_with_token_source_profile(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=inputs["token_source"],
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
        request_dispatchable=True,
    )

    gates = {row["id"]: row for row in plan["required_parity_gates"]}
    assert gates["pr85_baseline_token_extraction"]["passed"] is True
    assert gates["pr85_baseline_token_extraction"]["status"] == "passed_token_source_profiled"
    assert gates["pr85_baseline_token_extraction"]["evidence"]["token_shape"] == [600, 512, 384]
    refusal_gates = {row["gate"] for row in plan["dispatchability"]["refusal_reasons"]}
    assert "pr85_baseline_token_extraction" not in refusal_gates
    assert "pr85_hpac_token_parity" in refusal_gates


def test_blocker_records_probability_matrix_and_pr85_probe_failure(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=inputs["token_source"],
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
        request_dispatchable=True,
    )

    assert plan["blocker"]["id"] == "pr86_hpac_pr85_mask_contract_port"
    assert plan["blocker"]["status"] == "blocked_fail_closed"
    assert plan["candidate_spec"]["status"] == "not_emitted_fail_closed"
    assert plan["candidate_spec"]["candidate_paths"] == []
    precise = plan["blocker"]["precise_failure"]
    assert precise["pr86_probability_matrix"]["failure_reason"] == (
        "no_source_contract_variant_full_decode_byte_exact_reencode"
    )
    assert precise["pr86_probability_matrix"]["variants"][0]["failed_at"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
    }
    assert (
        precise["pr85_hpac_parity_probe"]["probe_status"]
        == "blocked_entropy_decode_assertion"
    )


def test_noop_pr85_hpac_candidate_is_not_accepted_as_parity(tmp_path: Path) -> None:
    inputs = _inputs(
        tmp_path,
        exact_score=True,
        token_parity=True,
        pr85_transfer=True,
        hpac_parity=_pr85_hpac_parity(
            passed=True,
            noop=True,
            candidate_token_sha="c" * 64,
            candidate_mask_sha="4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179",
        ),
    )

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=inputs["token_source"],
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
        request_dispatchable=True,
    )

    gates = {row["id"]: row for row in plan["required_parity_gates"]}
    assert gates["pr85_hpac_token_parity"]["passed"] is False
    assert gates["pr85_hpac_token_parity"]["status"] == (
        "failed_closed_pr85_hpac_candidate_missing_non_noop_parity_proof"
    )
    assert plan["candidate_spec"]["non_noop_guard"]["passed"] is False
    assert plan["dispatchable"] is False


def test_non_noop_pr85_hpac_candidate_requires_matching_baseline_token_sha(
    tmp_path: Path,
) -> None:
    inputs = _inputs(
        tmp_path,
        exact_score=True,
        token_parity=True,
        pr85_transfer=True,
        hpac_parity=_pr85_hpac_parity(
            passed=True,
            noop=False,
            candidate_token_sha="e" * 64,
            candidate_mask_sha="d" * 64,
        ),
    )

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=inputs["token_source"],
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
        request_dispatchable=True,
    )

    gates = {row["id"]: row for row in plan["required_parity_gates"]}
    assert gates["pr85_hpac_token_parity"]["passed"] is False
    assert plan["candidate_spec"]["token_parity_guard"]["passed"] is False
    assert plan["candidate_spec"]["token_parity_guard"]["candidate_decoded_token_sha256"] == "e" * 64


def test_pr85_hpac_candidate_parity_gate_can_pass_before_runtime_archive_gates(
    tmp_path: Path,
) -> None:
    inputs = _inputs(
        tmp_path,
        exact_score=True,
        token_parity=True,
        pr85_transfer=True,
        hpac_parity=_pr85_hpac_parity(passed=True),
    )

    plan = module.build_plan(
        pr86_anatomy_json=inputs["anatomy"],
        pr85_profile_json=inputs["profile"],
        pr86_parity_json=inputs["parity"],
        pr86_probability_matrix_json=inputs["probability_matrix"],
        pr85_token_source_json=inputs["token_source"],
        pr85_hpac_parity_json=inputs["hpac_parity"],
        pr85_archive=None,
        request_dispatchable=True,
    )

    gates = {row["id"]: row for row in plan["required_parity_gates"]}
    assert gates["pr85_hpac_token_parity"]["passed"] is True
    assert gates["pr85_hpac_token_parity"]["status"] == "passed"
    assert plan["candidate_spec"]["non_noop_guard"]["passed"] is True
    assert plan["candidate_spec"]["token_parity_guard"]["passed"] is True
    assert plan["dispatchable"] is False
    refusal_gates = {row["gate"] for row in plan["dispatchability"]["refusal_reasons"]}
    assert "pr85_runtime_output_parity" in refusal_gates
    assert "candidate_archive_byte_closure" in refusal_gates


def test_cli_writes_json_plan(tmp_path: Path, capsys) -> None:
    inputs = _inputs(tmp_path)
    out = tmp_path / "contract_port_plan.json"

    assert (
        module.main(
            [
                "--pr86-anatomy-json",
                str(inputs["anatomy"]),
                "--pr86-parity-json",
                str(inputs["parity"]),
                "--pr86-probability-matrix-json",
                str(inputs["probability_matrix"]),
                "--pr85-profile-json",
                str(inputs["profile"]),
                "--pr85-token-source-json",
                str(inputs["token_source"]),
                "--pr85-hpac-parity-json",
                str(inputs["hpac_parity"]),
                "--json-out",
                str(out),
                "--request-dispatchable",
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dispatchability"]["status"] == "refused_fail_closed"
    assert payload["gross_byte_math"]["gross_mask_byte_opportunity"] == 15369
    assert '"dispatchable": false' in capsys.readouterr().out


def test_default_pr86_parity_prefers_full_decode_reencode_gate_when_present() -> None:
    full_gate = module.FULL_PR86_DECODE_REENCODE_GATE_JSON
    legacy_gate = module.LEGACY_PR86_PARITY_JSON
    if full_gate.exists():
        assert module.DEFAULT_PR86_PARITY_JSON == full_gate
    else:
        assert module.DEFAULT_PR86_PARITY_JSON == legacy_gate
