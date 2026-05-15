# SPDX-License-Identifier: MIT
"""Tests for Catalog #272 — Distinguishing-Feature Integration Contract.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable + "HNeRV / leaderboard-implementation parity discipline"
lessons 2/7/11. Anchor 2026-05-15: Z3-G1 trained a 1KB SegNet-class CDF
(the "smart" distinguishing thing) but the archive's
``hyperprior_weights_int8`` slot is ``b""`` — the smart thing was
engineered, never wired. Smoke score == Z3 v2 baseline to 5 decimals.

Sister gate of Catalog #220 / #139 / #105 / #226 / #240 / #249.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_272_collect_lane_text,
    _check_272_lane_contract_status,
    _check_272_lane_has_required_fields,
    _check_272_lane_has_waiver,
    _check_272_lane_is_in_scope,
    _check_272_lane_is_research_only,
    _check_272_lane_is_substrate_engineering,
    _CHECK_272_REQUIRED_FIELDS,
    _CHECK_272_IN_SCOPE_ID_SUBSTRINGS,
    check_substrate_distinguishing_feature_integration_contract as check_272,
)


_FIXTURE_SHA = "a" * 64
_FIXTURE_MUTATION_PROOF_PATH = (
    "experiments/results/test/distinguishing_feature_byte_mutation_proof.json"
)
_FIXTURE_RUNTIME_PROOF_PATH = (
    "experiments/results/test/parser_section_runtime_consumption_proof.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_registry(tmp_path: Path, lanes: list[dict]) -> Path:
    """Write a fixture lane_registry.json under tmp_path/.omx/state/."""
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        if lane.get("byte_mutation_smoke_passes") == _FIXTURE_MUTATION_PROOF_PATH:
            targets = lane.get("distinguishing_bytes_path")
            _write_mutation_proof(tmp_path, targets=targets)
        if lane.get("byte_mutation_smoke_passes") == _FIXTURE_RUNTIME_PROOF_PATH:
            targets = lane.get("distinguishing_bytes_path")
            _write_parser_section_runtime_proof(tmp_path, targets=targets)
    registry_dir = tmp_path / ".omx" / "state"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry = {"schema_version": 1, "lanes": lanes}
    path = registry_dir / "lane_registry.json"
    path.write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def _normalize_targets(targets: object | None) -> list[str]:
    if isinstance(targets, str):
        return [targets]
    if isinstance(targets, (list, tuple)):
        return [str(t) for t in targets]
    return ["hyperprior_weights_int8", "w_hat_int8"]


def _write_mutation_proof(
    root: Path,
    *,
    rel_path: str = _FIXTURE_MUTATION_PROOF_PATH,
    targets: object | None = None,
) -> str:
    target_list = _normalize_targets(targets)
    payload = {
        "schema_version": 1,
        "archive_sha256": _FIXTURE_SHA,
        "archive_size_bytes": 1234,
        "inflate_sh_sha256": _FIXTURE_SHA,
        "distinguishing_bytes_paths": target_list,
        "section_results": [
            {
                "section": target,
                "target_basis": "zip_member",
                "member": target,
                "offset": None,
                "length": None,
                "section_size_bytes": 17,
                "mutations_attempted": 2,
                "mutations_changed_output": 1,
                "passed": True,
                "first_changed_inflated_output_sha256": _FIXTURE_SHA,
                "baseline_inflated_output_sha256": _FIXTURE_SHA,
            }
            for target in target_list
        ],
        "overall_passed": True,
        "verdict": "PASSED",
        "elapsed_seconds": 0.01,
    }
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return rel_path


def _write_parser_section_runtime_proof(
    root: Path,
    *,
    rel_path: str = _FIXTURE_RUNTIME_PROOF_PATH,
    targets: object | None = None,
) -> str:
    target_list = _normalize_targets(targets)
    changed_sections = [
        {
            "section_name": target,
            "new_sha256": _FIXTURE_SHA,
            "runtime_consumed": True,
            "offset": 4 + idx * 16,
            "length": 16,
        }
        for idx, target in enumerate(target_list)
    ]
    payload = {
        "schema": "tac_runtime_consumption_proof_v1",
        "proof_kind": "tac_monolithic_runtime_consumption_probe_v1",
        "ready_for_exact_eval_runtime": True,
        "blockers": [],
        "consumed_sections": changed_sections,
        "changed_sections": changed_sections,
        "score_claim": False,
    }
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return rel_path


def _full_contract_lane(lane_id: str, level: int = 2) -> dict:
    """Build a lane that declares all 4 contract fields."""
    return {
        "id": lane_id,
        "level": level,
        "name": "Test substrate",
        "notes": "test substrate fully wired",
        "distinguishing_feature_name": (
            "1KB SegNet-class CDF replacing 50KB Ballé hyperprior"
        ),
        "distinguishing_bytes_path": ["hyperprior_weights_int8", "w_hat_int8"],
        "inflate_consumer_function": [
            "inflate.py::_unpack_hyperprior_cdf",
            "inflate.py::_class_conditional_range_decode",
        ],
        "byte_mutation_smoke_passes": _FIXTURE_MUTATION_PROOF_PATH,
        "gates": {
            "impl_complete": {"satisfied": True, "evidence": "fixture"},
        },
    }


def _no_contract_lane(lane_id: str, level: int = 2) -> dict:
    return {
        "id": lane_id,
        "level": level,
        "name": "Test substrate",
        "notes": "no contract declared",
        "gates": {
            "impl_complete": {"satisfied": True, "evidence": "fixture"},
        },
    }


# ---------------------------------------------------------------------------
# In-scope classifier (helper)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lane_id,expected",
    [
        ("lane_substrate_z3_g1_scorer_softmax_hyperprior_gating_20260515", True),
        ("lane_d1_segnet_margin_polytope_encoder_20260514", True),
        ("lane_a1_inflate_time_bias_correction_sweep", True),
        ("lane_pr106_latent_sidecar_r2_pr101_grammar", True),
        ("lane_12_nerv_mask_codec", True),
        ("lane_wavelet_residual_pr106", True),
        ("lane_yucr_anchor", True),
        ("lane_distinguishing_feature_integration_contract_20260515", False),
        ("lane_unrelated_infrastructure_landing", False),
        ("lane_codex_review_round_3", False),
        ("", False),
    ],
)
def test_check_272_in_scope_classifier(lane_id, expected):
    assert _check_272_lane_is_in_scope(lane_id) is expected


# ---------------------------------------------------------------------------
# Required-fields evaluator (helper)
# ---------------------------------------------------------------------------


def test_required_fields_present_with_full_contract(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    _write_mutation_proof(tmp_path, targets=lane["distinguishing_bytes_path"])
    all_present, missing = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is True
    assert missing == []


def test_required_fields_missing_all_four():
    lane = _no_contract_lane("lane_substrate_test_l2", level=2)
    all_present, missing = _check_272_lane_has_required_fields(lane)
    assert all_present is False
    assert set(missing) == set(_CHECK_272_REQUIRED_FIELDS)


def test_required_fields_partial_three_of_four(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    _write_mutation_proof(tmp_path, targets=lane["distinguishing_bytes_path"])
    del lane["byte_mutation_smoke_passes"]
    all_present, missing = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["byte_mutation_smoke_passes"]


def test_required_fields_byte_mutation_false_rejected(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    lane["byte_mutation_smoke_passes"] = False
    all_present, missing, proof_errors = _check_272_lane_contract_status(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["byte_mutation_smoke_passes"]
    assert "self_attested_bool_rejected" in proof_errors


def test_required_fields_byte_mutation_true_rejected_as_self_attested(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    lane["byte_mutation_smoke_passes"] = True
    all_present, missing, proof_errors = _check_272_lane_contract_status(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["byte_mutation_smoke_passes"]
    assert "self_attested_bool_rejected" in proof_errors


def test_required_fields_byte_mutation_dict_evidence_path_accepted(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    rel_path = _write_mutation_proof(
        tmp_path,
        rel_path="experiments/results/test/custom_proof.json",
        targets=lane["distinguishing_bytes_path"],
    )
    lane["byte_mutation_smoke_passes"] = {
        "evidence_path": rel_path,
    }
    all_present, _ = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is True


def test_required_fields_byte_mutation_str_evidence_path_accepted(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    rel_path = _write_mutation_proof(
        tmp_path,
        rel_path="experiments/results/test/string_proof.json",
        targets=lane["distinguishing_bytes_path"],
    )
    lane["byte_mutation_smoke_passes"] = rel_path
    all_present, _ = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is True


def test_required_fields_byte_mutation_str_without_artifact_rejected(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    lane["byte_mutation_smoke_passes"] = (
        "experiments/results/test/missing_proof.json"
    )
    all_present, missing, proof_errors = _check_272_lane_contract_status(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["byte_mutation_smoke_passes"]
    assert any("proof_artifact_missing" in err for err in proof_errors)


def test_required_fields_inline_proof_object_rejected_as_self_attested(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    lane["byte_mutation_smoke_passes"] = {
        "schema_version": 1,
        "verdict": "PASSED",
        "overall_passed": True,
        "section_results": [],
    }
    all_present, missing, proof_errors = _check_272_lane_contract_status(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["byte_mutation_smoke_passes"]
    assert "inline_proof_object_requires_artifact_path" in proof_errors


def test_required_fields_parser_section_runtime_proof_artifact_accepted(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    lane["distinguishing_bytes_path"] = ["decoder_packed_brotli"]
    lane["byte_mutation_smoke_passes"] = _write_parser_section_runtime_proof(
        tmp_path,
        targets=lane["distinguishing_bytes_path"],
    )
    all_present, missing = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is True
    assert missing == []


def test_required_fields_parser_section_without_offsets_rejected(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    lane["distinguishing_bytes_path"] = ["decoder_packed_brotli"]
    rel_path = _write_parser_section_runtime_proof(
        tmp_path,
        targets=lane["distinguishing_bytes_path"],
    )
    proof_path = tmp_path / rel_path
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    del payload["changed_sections"][0]["offset"]
    proof_path.write_text(json.dumps(payload), encoding="utf-8")
    lane["byte_mutation_smoke_passes"] = rel_path
    all_present, missing, proof_errors = _check_272_lane_contract_status(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["byte_mutation_smoke_passes"]
    assert any("offset_missing" in err for err in proof_errors)


def test_required_fields_empty_list_rejected(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    _write_mutation_proof(tmp_path, targets=lane["distinguishing_bytes_path"])
    lane["distinguishing_bytes_path"] = []
    all_present, missing = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["distinguishing_bytes_path"]


def test_required_fields_empty_string_rejected(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    _write_mutation_proof(tmp_path, targets=lane["distinguishing_bytes_path"])
    lane["distinguishing_feature_name"] = "   "
    all_present, missing = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["distinguishing_feature_name"]


def test_required_fields_none_rejected(tmp_path):
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    _write_mutation_proof(tmp_path, targets=lane["distinguishing_bytes_path"])
    lane["inflate_consumer_function"] = None
    all_present, missing = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert missing == ["inflate_consumer_function"]


def test_required_fields_bool_string_field_rejected(tmp_path):
    """bool values for str-typed fields are treated as missing."""
    lane = _full_contract_lane("lane_substrate_test_l2", level=2)
    _write_mutation_proof(tmp_path, targets=lane["distinguishing_bytes_path"])
    lane["distinguishing_feature_name"] = True
    all_present, missing = _check_272_lane_has_required_fields(
        lane,
        repo_root=tmp_path,
    )
    assert all_present is False
    assert "distinguishing_feature_name" in missing


# ---------------------------------------------------------------------------
# Research-only / substrate-engineering classifiers (helpers)
# ---------------------------------------------------------------------------


def test_research_only_top_level_bool_true():
    lane = {"id": "lane_substrate_x", "research_only": True, "notes": ""}
    assert _check_272_lane_is_research_only(lane, "")


def test_research_only_top_level_str_true():
    lane = {"id": "lane_substrate_x", "research_only": "true", "notes": ""}
    assert _check_272_lane_is_research_only(lane, "")


def test_research_only_notes_token():
    lane = {"id": "lane_substrate_x", "notes": "research_only=true per HNeRV L2"}
    text = _check_272_collect_lane_text(lane)
    assert _check_272_lane_is_research_only(lane, text)


def test_research_only_dash_form():
    lane = {"id": "lane_substrate_x", "notes": "research-only=true variant"}
    text = _check_272_collect_lane_text(lane)
    assert _check_272_lane_is_research_only(lane, text)


def test_research_only_negative():
    lane = {"id": "lane_substrate_x", "notes": "production substrate"}
    text = _check_272_collect_lane_text(lane)
    assert not _check_272_lane_is_research_only(lane, text)


def test_substrate_engineering_top_level():
    lane = {
        "id": "lane_substrate_x",
        "lane_class": "substrate_engineering",
        "notes": "",
    }
    text = _check_272_collect_lane_text(lane)
    assert _check_272_lane_is_substrate_engineering(lane, text)


def test_substrate_engineering_notes_token():
    lane = {
        "id": "lane_substrate_x",
        "notes": "lane_class=substrate_engineering per HNeRV lesson 7",
    }
    text = _check_272_collect_lane_text(lane)
    assert _check_272_lane_is_substrate_engineering(lane, text)


def test_substrate_engineering_negative():
    lane = {"id": "lane_substrate_x", "notes": "production substrate"}
    text = _check_272_collect_lane_text(lane)
    assert not _check_272_lane_is_substrate_engineering(lane, text)


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_waiver_with_real_rationale_accepted():
    text = (
        "# DISTINGUISHING_FEATURE_CONTRACT_OK:operator-reviewed L2 sister "
        "lane shares contract with primary"
    )
    waived, reason = _check_272_lane_has_waiver(text)
    assert waived
    assert "operator-reviewed" in reason


def test_waiver_placeholder_reason_rejected():
    text = "# DISTINGUISHING_FEATURE_CONTRACT_OK:<rationale>"
    waived, reason = _check_272_lane_has_waiver(text)
    assert not waived
    assert reason == ""


def test_waiver_placeholder_reason_alt_rejected():
    text = "# DISTINGUISHING_FEATURE_CONTRACT_OK:<reason>"
    waived, _ = _check_272_lane_has_waiver(text)
    assert not waived


def test_waiver_empty_text():
    waived, reason = _check_272_lane_has_waiver("")
    assert not waived
    assert reason == ""


def test_waiver_no_marker():
    waived, reason = _check_272_lane_has_waiver("notes without any waiver")
    assert not waived


# ---------------------------------------------------------------------------
# End-to-end gate
# ---------------------------------------------------------------------------


def test_gate_no_registry_returns_empty(tmp_path):
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_registry_with_no_lanes_field(tmp_path):
    registry_dir = tmp_path / ".omx" / "state"
    registry_dir.mkdir(parents=True)
    (registry_dir / "lane_registry.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_corrupt_registry_handled_gracefully(tmp_path):
    registry_dir = tmp_path / ".omx" / "state"
    registry_dir.mkdir(parents=True)
    (registry_dir / "lane_registry.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_passes_clean_full_contract(tmp_path):
    _write_registry(tmp_path, [_full_contract_lane("lane_substrate_x", level=2)])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_l1_substrate_skipped(tmp_path):
    """L1 lanes are out of scope (only L2+ promotion is gated)."""
    _write_registry(
        tmp_path, [_no_contract_lane("lane_substrate_x", level=1)]
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_l0_substrate_skipped(tmp_path):
    _write_registry(
        tmp_path, [_no_contract_lane("lane_substrate_x", level=0)]
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_non_substrate_lane_skipped(tmp_path):
    """Lanes whose id doesn't match substrate patterns are out of scope."""
    _write_registry(
        tmp_path,
        [_no_contract_lane("lane_codex_review_round_3", level=2)],
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_flags_l2_substrate_without_contract(tmp_path):
    _write_registry(
        tmp_path, [_no_contract_lane("lane_substrate_x", level=2)]
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "lane_substrate_x" in violations[0]
    assert "missing 4/4" in violations[0]


def test_gate_accepts_research_only_lane(tmp_path):
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["research_only"] = True
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_accepts_research_only_via_notes(tmp_path):
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["notes"] = "research_only=true per HNeRV L2"
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_accepts_substrate_engineering_top_level(tmp_path):
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["lane_class"] = "substrate_engineering"
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_accepts_substrate_engineering_via_notes(tmp_path):
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["notes"] = "lane_class=substrate_engineering"
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_accepts_waiver_in_notes(tmp_path):
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["notes"] = (
        "# DISTINGUISHING_FEATURE_CONTRACT_OK:operator-reviewed sister lane"
    )
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_rejects_placeholder_waiver(tmp_path):
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["notes"] = "# DISTINGUISHING_FEATURE_CONTRACT_OK:<rationale>"
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert len(violations) == 1


def test_gate_aggregates_multiple_violations(tmp_path):
    lanes = [
        _no_contract_lane("lane_substrate_a", level=2),
        _no_contract_lane("lane_substrate_b", level=2),
        _no_contract_lane("lane_substrate_c", level=2),
    ]
    _write_registry(tmp_path, lanes)
    violations = check_272(repo_root=tmp_path, strict=False)
    assert len(violations) == 3


def test_gate_strict_mode_raises(tmp_path):
    _write_registry(
        tmp_path, [_no_contract_lane("lane_substrate_x", level=2)]
    )
    with pytest.raises(PreflightError) as exc_info:
        check_272(repo_root=tmp_path, strict=True)
    err = str(exc_info.value)
    assert "Catalog #272" in err
    assert "Z3-G1" in err


def test_gate_strict_mode_silent_on_clean(tmp_path):
    _write_registry(
        tmp_path, [_full_contract_lane("lane_substrate_x", level=2)]
    )
    # Should NOT raise
    violations = check_272(repo_root=tmp_path, strict=True)
    assert violations == []


def test_gate_message_includes_lane_id(tmp_path):
    _write_registry(
        tmp_path, [_no_contract_lane("lane_substrate_xyz_distinct", level=2)]
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert "lane_substrate_xyz_distinct" in violations[0]


def test_gate_message_includes_remediation_command(tmp_path):
    _write_registry(
        tmp_path, [_no_contract_lane("lane_substrate_x", level=2)]
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert "tools/lane_maturity.py set-field" in violations[0]


def test_gate_message_cites_z3_g1_anchor(tmp_path):
    _write_registry(
        tmp_path, [_no_contract_lane("lane_substrate_x", level=2)]
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert "Z3-G1" in violations[0]
    assert "hyperprior_weights_int8" in violations[0]


def test_gate_string_repo_root_accepted(tmp_path):
    _write_registry(
        tmp_path, [_full_contract_lane("lane_substrate_x", level=2)]
    )
    violations = check_272(repo_root=str(tmp_path), strict=False)
    assert violations == []


# ---------------------------------------------------------------------------
# Substrate fixtures matching real-world lanes (Z3-G1, D1, Z4, A1)
# ---------------------------------------------------------------------------


def test_z3_g1_anchor_satisfied_via_research_only(tmp_path):
    """Z3-G1 is research_only=true today; the gate should accept it."""
    lane = _no_contract_lane(
        "lane_z3_g1_scorer_softmax_hyperprior_gating_20260515", level=2
    )
    lane["research_only"] = True
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_z3_g1_anchor_passes_when_full_contract_declared(tmp_path):
    """Z3-G1 with the contract fully declared (post-wire-in) passes."""
    lane = {
        "id": "lane_z3_g1_scorer_softmax_hyperprior_gating_20260515",
        "level": 2,
        "name": "Z3-G1 scorer-softmax hyperprior gating",
        "notes": "post G1 wire-in: bytes operationally consumed",
        "distinguishing_feature_name": (
            "1KB SegNet-class CDF replacing 50KB Ballé hyperprior"
        ),
        "distinguishing_bytes_path": [
            "hyperprior_weights_int8",
            "w_hat_int8",
        ],
        "inflate_consumer_function": [
            "inflate.py::_unpack_hyperprior_cdf",
            "inflate.py::_class_conditional_range_decode",
        ],
        "byte_mutation_smoke_passes": _FIXTURE_MUTATION_PROOF_PATH,
        "gates": {},
    }
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_d1_segnet_polytope_satisfied_via_substrate_engineering(tmp_path):
    """D1 SegNet polytope encoder lane is lane_class=substrate_engineering."""
    lane = _no_contract_lane("lane_d1_segnet_margin_polytope_encoder_20260514", level=2)
    lane["lane_class"] = "substrate_engineering"
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_pr106_latent_sidecar_l2_without_contract_flagged(tmp_path):
    """A real-world L2 lane that lacks all 3 opt-outs is flagged."""
    lane = _no_contract_lane(
        "lane_pr106_latent_sidecar_r2_pr101_grammar_contest_cpu", level=2
    )
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "lane_pr106_latent_sidecar" in violations[0]


# ---------------------------------------------------------------------------
# Live-repo regression guard + orchestrator wire-in
# ---------------------------------------------------------------------------


def test_live_repo_count_bounded():
    """Live-repo violation count must stay below an upper bound (currently 14).

    The gate is initially WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule"
    because the 32-substrate-trainer audit at landing reveals N >= 14
    substrate L2+ lanes that have not yet declared the contract fields.
    Strict-flip pending the operator-routed backfill sweep.

    This test ensures no regression — the count must not GROW past the
    upper bound. As lanes are backfilled, the upper bound can be lowered.
    """
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_272(repo_root=repo_root, strict=False)
    assert len(violations) <= 14, (
        f"Catalog #272 live count regressed: {len(violations)} > 14 upper "
        f"bound. Backfill substrate L2+ lanes via tools/lane_maturity.py "
        f"set-field. First 3:\n  " + "\n  ".join(v[:200] for v in violations[:3])
    )


def test_orchestrator_wires_warn_only():
    """preflight_all() must wire #272 with strict=False (warn-only)."""
    preflight_py = (
        Path(__file__).resolve().parents[3] / "src" / "tac" / "preflight.py"
    )
    text = preflight_py.read_text(encoding="utf-8")
    assert (
        "check_substrate_distinguishing_feature_integration_contract(\n"
        "            strict=False, verbose=verbose,\n"
        "        )"
    ) in text, (
        "Catalog #272 must be wired in preflight_all() with strict=False "
        "(WARN-ONLY) per CLAUDE.md 'Strict-flip atomicity rule'. The "
        "32-substrate-trainer audit at landing reveals N >= 14 violations; "
        "strict-flip pending sister-wave backfill."
    )


def test_lane_maturity_set_field_allowlist_includes_new_fields():
    """tools/lane_maturity.py set-field must accept the 4 contract fields.

    Per the user prompt: "additive lane_registry.json mutations via
    tools/lane_maturity.py set-field --field (canonical helper preserves
    fcntl per Catalog #131)".
    """
    from tools.lane_maturity import _SET_FIELD_TOP_LEVEL_ALLOWED  # type: ignore

    for field in _CHECK_272_REQUIRED_FIELDS:
        assert field in _SET_FIELD_TOP_LEVEL_ALLOWED, (
            f"Field {field!r} must be in lane_maturity.py "
            f"_SET_FIELD_TOP_LEVEL_ALLOWED so subagents can set it via "
            f"the canonical helper."
        )


def test_in_scope_substrings_match_catalog_220():
    """Catalog #272 must use the SAME in-scope substring set as Catalog #220.

    Both gates target the same surface (substrate L1+/L2+ lanes); divergent
    in-scope sets would create two different "what's in scope" answers.
    """
    from tac.preflight import _CHECK_220_IN_SCOPE_ID_SUBSTRINGS

    assert _CHECK_272_IN_SCOPE_ID_SUBSTRINGS == _CHECK_220_IN_SCOPE_ID_SUBSTRINGS, (
        "Catalog #272 in-scope set diverged from Catalog #220. Both gates "
        "scan the same lane surface; they MUST share the substring set so "
        "operator-facing scope is consistent."
    )


# ---------------------------------------------------------------------------
# Acceptance cascade ordering (research_only beats waiver beats fields)
# ---------------------------------------------------------------------------


def test_research_only_short_circuits_missing_fields(tmp_path):
    """research_only=true accepts even when contract fields are missing."""
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["research_only"] = True
    _write_registry(tmp_path, [lane])
    assert check_272(repo_root=tmp_path, strict=False) == []


def test_substrate_engineering_short_circuits_missing_fields(tmp_path):
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["lane_class"] = "substrate_engineering"
    _write_registry(tmp_path, [lane])
    assert check_272(repo_root=tmp_path, strict=False) == []


def test_waiver_with_partial_contract_accepted(tmp_path):
    """Waiver beats partial contract: even 3/4 fields without the 4th
    is acceptable if the waiver carries a real rationale."""
    lane = _no_contract_lane("lane_substrate_x", level=2)
    lane["distinguishing_feature_name"] = "test"
    lane["distinguishing_bytes_path"] = ["test_section"]
    lane["inflate_consumer_function"] = ["test_fn"]
    # byte_mutation_smoke_passes intentionally missing
    lane["notes"] = (
        "# DISTINGUISHING_FEATURE_CONTRACT_OK:smoke-pending operator review"
    )
    _write_registry(tmp_path, [lane])
    assert check_272(repo_root=tmp_path, strict=False) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_gate_ignores_non_dict_lane_entries(tmp_path):
    _write_mutation_proof(tmp_path)
    registry_dir = tmp_path / ".omx" / "state"
    registry_dir.mkdir(parents=True)
    (registry_dir / "lane_registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lanes": [
                    "not a dict",
                    None,
                    _full_contract_lane("lane_substrate_x", level=2),
                ],
            }
        ),
        encoding="utf-8",
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_handles_lane_without_id_field(tmp_path):
    lane = {"level": 2, "notes": "no id"}
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert violations == []  # empty id => not in scope


def test_gate_handles_lane_level_as_string(tmp_path):
    """level field stored as string (legacy) must coerce to int."""
    lane = _no_contract_lane("lane_substrate_x", level=0)
    lane["level"] = "2"  # string instead of int
    _write_registry(tmp_path, [lane])
    violations = check_272(repo_root=tmp_path, strict=False)
    assert len(violations) == 1


def test_gate_l3_substrate_also_audited(tmp_path):
    """L3 lanes (FULL PRODUCTION HARDENED) are also in scope."""
    _write_registry(
        tmp_path, [_no_contract_lane("lane_substrate_x", level=3)]
    )
    violations = check_272(repo_root=tmp_path, strict=False)
    assert len(violations) == 1
    assert "(L3)" in violations[0]


def test_gate_verbose_output_lists_violations(tmp_path, capsys):
    _write_registry(
        tmp_path,
        [
            _no_contract_lane("lane_substrate_a", level=2),
            _full_contract_lane("lane_substrate_b", level=2),
        ],
    )
    check_272(repo_root=tmp_path, strict=False, verbose=True)
    captured = capsys.readouterr()
    assert "[catalog-272]" in captured.out
    assert "l2plus=2" in captured.out
    assert "violations=1" in captured.out
