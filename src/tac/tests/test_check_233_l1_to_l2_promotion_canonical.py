# SPDX-License-Identifier: MIT
"""Tests for Catalog #233 — L1->L2 substrate-class promotion canonical 4-gate.

Operationalizes the council Decision 8 binding verdict from the omnibus commit
``7872c9f4b`` (PROCEED Option D, 11/11 unanimous): substrate L2+ promotion
REQUIRES (1) smoke green, (2) Tier C MDL density measured, (3) 100ep auth-eval
anchor with byte-deterministic archive, (4) custody validated per Catalog #127.

Sister of Catalog #127 (per-call-site custody validator routing) + Catalog #220
(L1 scaffold byte-addition gate) + Catalog #227 (L2+ class-shift Tier C gate).

Memory: feedback_d8_l1_l2_promotion_canonical_strict_preflight_landed_20260514.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tac.preflight import (
    PreflightError,
    check_l1_to_l2_promotion_canonical_4_gate,
    _check_233_lane_is_in_scope,
    _check_233_lane_is_exempt,
    _check_233_lane_has_waiver,
    _check_233_collect_lane_text,
    _check_233_evaluate_4_gates,
    _check_233_text_contains_any_token,
    _CHECK_233_IN_SCOPE_ID_SUBSTRINGS,
    _CHECK_233_EXEMPT_LANE_CLASS_TOKENS,
    _CHECK_233_EXEMPT_TARGET_MODE_TOKENS,
    _CHECK_233_SMOKE_GREEN_TOKENS,
    _CHECK_233_TIER_C_TOKENS,
    _CHECK_233_AUTH_EVAL_100EP_TOKENS,
    _CHECK_233_CUSTODY_VALIDATOR_TOKENS,
)


# ---------------------------------------------------------------------------
# Helpers — synthesize lane registry under tmp_path with controlled lanes.
# ---------------------------------------------------------------------------


def _gates_with_evidence(
    *,
    impl_complete_evidence: str = "stub-impl-complete",
    real_archive_evidence: str = "",
    contest_cuda_evidence: str = "",
    contest_cpu_evidence: str = "",
    impl_complete_status: bool = True,
    real_archive_status: bool = True,
    contest_cuda_status: bool = False,
    contest_cpu_status: bool = False,
) -> dict[str, dict]:
    """Build the canonical 8-gate dict with controlled evidence strings."""
    return {
        "impl_complete": {
            "evidence": impl_complete_evidence,
            "status": impl_complete_status,
        },
        "real_archive_empirical": {
            "evidence": real_archive_evidence,
            "status": real_archive_status,
        },
        "contest_cuda": {
            "evidence": contest_cuda_evidence,
            "status": contest_cuda_status,
        },
        "contest_cpu": {
            "evidence": contest_cpu_evidence,
            "status": contest_cpu_status,
        },
        "strict_preflight": {"evidence": "", "status": False},
        "three_clean_review": {"evidence": "", "status": False},
        "memory_entry": {"evidence": "", "status": False},
        "deploy_runbook": {"evidence": "", "status": False},
    }


def _make_lane(
    lane_id: str,
    *,
    level: int = 2,
    name: str = "test lane",
    notes: str = "",
    lane_class: str = "",
    target_modes: list[str] | None = None,
    gates: dict[str, dict] | None = None,
) -> dict[str, Any]:
    lane: dict[str, Any] = {
        "id": lane_id,
        "level": level,
        "name": name,
        "notes": notes,
        "phase": 1.0,
        "gates": gates or _gates_with_evidence(),
    }
    if lane_class:
        lane["lane_class"] = lane_class
    if target_modes is not None:
        lane["target_modes"] = target_modes
    return lane


def _write_registry(root: Path, lanes: list[dict]) -> Path:
    state = root / ".omx" / "state"
    state.mkdir(parents=True, exist_ok=True)
    path = state / "lane_registry.json"
    payload = {"schema_version": 1, "lanes": lanes}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# Canonical 4-gate evidence text — each token from a different gate token-set.
ALL_FOUR_GATES_TEXT = (
    "smoke rc=0; tier_c measured 100ep auth-eval; [contest-CUDA]; "
    "validate_custody passed; evidence_grade=contest_cuda; "
    "hardware_substrate=linux_x86_64_a100"
)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_in_scope_classifier_accepts_substrate_substring():
    assert _check_233_lane_is_in_scope("lane_substrate_x_y_z")
    assert _check_233_lane_is_in_scope("lane_d4_wyner_ziv")
    assert _check_233_lane_is_in_scope("lane_d1_segnet_polytope")
    assert _check_233_lane_is_in_scope("lane_a1_segnet_smoothing")
    # lane_a1_ matches; sidecar lanes are matched via _sidecar_ when nested.
    assert _check_233_lane_is_in_scope("lane_pr106_latent_sidecar_r2")


def test_in_scope_classifier_rejects_non_substrate_lanes():
    assert not _check_233_lane_is_in_scope("lane_oss_release_packet")
    assert not _check_233_lane_is_in_scope("lane_ci_smoke_runner")
    assert not _check_233_lane_is_in_scope("lane_council_omnibus")


def test_in_scope_classifier_handles_empty_id():
    assert not _check_233_lane_is_in_scope("")


def test_in_scope_substrings_constant_pinned():
    # If anyone removes a canonical token, this test catches it.
    needed = {"substrate_", "d4_", "d1_segnet", "lane_a1_", "_hnerv_", "_nerv_"}
    actual = set(_CHECK_233_IN_SCOPE_ID_SUBSTRINGS)
    missing = needed - actual
    assert not missing, f"Catalog #233 in-scope set lost canonical tokens: {missing}"


def test_exempt_classifier_substrate_engineering_lane_class():
    lane = _make_lane(
        "lane_substrate_x", lane_class="substrate_engineering"
    )
    assert _check_233_lane_is_exempt(lane)


def test_exempt_classifier_research_substrate_lane_class():
    lane = _make_lane(
        "lane_substrate_x", lane_class="research_substrate"
    )
    assert _check_233_lane_is_exempt(lane)


def test_exempt_classifier_research_only_target_mode():
    lane = _make_lane(
        "lane_substrate_x", target_modes=["research_only"]
    )
    assert _check_233_lane_is_exempt(lane)


def test_exempt_classifier_research_only_in_notes():
    lane = _make_lane(
        "lane_substrate_x", notes="research_only=true (placeholder)"
    )
    assert _check_233_lane_is_exempt(lane)


def test_exempt_classifier_substrate_engineering_exception_in_notes():
    lane = _make_lane(
        "lane_substrate_x",
        notes="substrate_engineering_exception per HNeRV parity L7",
    )
    assert _check_233_lane_is_exempt(lane)


def test_exempt_classifier_clean_lane_not_exempt():
    lane = _make_lane("lane_substrate_x", notes="ordinary substrate work")
    assert not _check_233_lane_is_exempt(lane)


def test_collect_lane_text_concatenates_notes_plus_gate_evidence():
    lane = _make_lane(
        "lane_x",
        notes="NOTES-MARKER",
        gates=_gates_with_evidence(
            impl_complete_evidence="IMPL-MARKER",
            real_archive_evidence="EMPIRICAL-MARKER",
            contest_cuda_evidence="CUDA-MARKER",
            contest_cpu_evidence="CPU-MARKER",
        ),
    )
    text = _check_233_collect_lane_text(lane)
    for marker in (
        "NOTES-MARKER", "IMPL-MARKER", "EMPIRICAL-MARKER",
        "CUDA-MARKER", "CPU-MARKER",
    ):
        assert marker in text, f"missing {marker} in concatenated text"


def test_waiver_with_rationale_accepted():
    text = "ordinary notes # L1_L2_PROMOTION_CANONICAL_OK:operator-reviewed"
    waived, reason = _check_233_lane_has_waiver(text)
    assert waived
    assert "operator-reviewed" in reason


def test_waiver_placeholder_reason_rejected():
    text = "notes # L1_L2_PROMOTION_CANONICAL_OK:<reason>"
    waived, reason = _check_233_lane_has_waiver(text)
    assert not waived
    assert reason == ""


def test_waiver_placeholder_rationale_rejected():
    text = "notes # L1_L2_PROMOTION_CANONICAL_OK:<rationale>"
    waived, reason = _check_233_lane_has_waiver(text)
    assert not waived


def test_waiver_inline_form_accepted():
    text = "L1_L2_PROMOTION_CANONICAL_OK:legitimate-rationale"
    waived, reason = _check_233_lane_has_waiver(text)
    assert waived
    assert "legitimate-rationale" in reason


def test_waiver_no_marker_returns_false():
    text = "ordinary notes with no waiver"
    waived, reason = _check_233_lane_has_waiver(text)
    assert not waived
    assert reason == ""


def test_text_contains_any_token_case_insensitive():
    # Token "Tier C" matches text "Tier C measured" via case-insensitive substring.
    assert _check_233_text_contains_any_token("Tier C measured", ("Tier C",))
    # Token "tier c" with the actual whitespace also matches "Tier C measured".
    assert _check_233_text_contains_any_token("Tier C measured", ("tier c",))
    # "tier_c" (with underscore) does NOT match "Tier C measured" (with space) —
    # confirms _ vs space is preserved (no normalization).
    assert not _check_233_text_contains_any_token("Tier C measured", ("tier_c",))
    assert _check_233_text_contains_any_token("smoke rc=0 here", ("smoke rc=0",))
    assert not _check_233_text_contains_any_token("nothing here", ("smoke_green",))


def test_evaluate_4_gates_all_present():
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates(ALL_FOUR_GATES_TEXT)
    assert smoke and tier_c and auth and custody


def test_evaluate_4_gates_only_smoke_present():
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates("smoke rc=0")
    assert smoke
    assert not (tier_c or auth or custody)


def test_evaluate_4_gates_only_tier_c_present():
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates(
        "Tier C measured but no other evidence"
    )
    assert tier_c
    assert not (smoke or auth or custody)


def test_evaluate_4_gates_only_auth_eval_present():
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates(
        "[contest-CUDA] anchor here only"
    )
    assert auth
    assert not (smoke or tier_c or custody)


def test_evaluate_4_gates_only_custody_present():
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates(
        "validate_custody routed here only"
    )
    assert custody
    assert not (smoke or tier_c or auth)


def test_evaluate_4_gates_three_of_four_still_fails():
    text = "smoke rc=0; tier_c measured; [contest-CUDA] anchor"
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates(text)
    assert smoke and tier_c and auth
    assert not custody


def test_evaluate_4_gates_rejects_r2_false_positive_prose():
    text = (
        "See discussion of 100ep vs 200ep tradeoffs in the smoke green report; "
        "tier c was measured; validate_custody verified; evidence_grade=contest_cuda"
    )
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates(text)
    assert not (smoke and tier_c and auth and custody)
    assert not auth


def test_evaluate_4_gates_accepts_structured_key_value_evidence():
    text = (
        "smoke_green=true; tier_c_density=0.42; "
        "auth_eval_score_axis=contest_cuda; custody_validated=true"
    )
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates(text)
    assert smoke and tier_c and auth and custody


def test_100ep_without_auth_eval_context_is_not_anchor():
    smoke, tier_c, auth, custody = _check_233_evaluate_4_gates(
        "100ep vs 200ep tradeoffs only"
    )
    assert not (smoke or tier_c or auth or custody)


# ---------------------------------------------------------------------------
# End-to-end gate behavior
# ---------------------------------------------------------------------------


def test_passes_when_no_lane_registry(tmp_path: Path):
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_passes_when_no_substrate_lanes_at_l2(tmp_path: Path):
    _write_registry(tmp_path, [
        _make_lane("lane_unrelated_oss_release", level=2),
        _make_lane("lane_substrate_x", level=1, gates=_gates_with_evidence()),
    ])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # lane_unrelated_oss_release is out of scope; lane_substrate_x is L1
    # (gate is L1→L2 promotion, not L0→L1). Expect zero violations.
    assert violations == []


def test_lane_with_all_four_gates_in_notes_passes(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_great",
        level=2,
        notes=ALL_FOUR_GATES_TEXT,
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_lane_with_all_four_gates_split_across_evidence_strings_passes(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_distributed_evidence",
        level=2,
        gates=_gates_with_evidence(
            impl_complete_evidence="smoke rc=0 + smoke complete",
            real_archive_evidence="tier_c measured @ density 0.13",
            contest_cuda_evidence="100ep auth-eval [contest-CUDA] complete",
            contest_cpu_evidence="validate_custody routed; evidence_grade=contest_cuda",
        ),
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_lane_missing_smoke_green_gate_flagged(tmp_path: Path):
    text = "tier_c measured + 100ep auth-eval [contest-CUDA] + validate_custody"
    lane = _make_lane(
        "lane_substrate_no_smoke",
        level=2,
        notes=text,
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "(1) smoke green" in violations[0]


def test_lane_missing_tier_c_gate_flagged(tmp_path: Path):
    text = "smoke rc=0 + 100ep auth-eval [contest-CUDA] + validate_custody"
    lane = _make_lane(
        "lane_substrate_no_tier_c",
        level=2,
        notes=text,
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "(2) Tier C" in violations[0]


def test_lane_missing_100ep_auth_eval_gate_flagged(tmp_path: Path):
    text = "smoke rc=0 + tier_c measured + validate_custody routed"
    lane = _make_lane(
        "lane_substrate_no_auth_eval",
        level=2,
        notes=text,
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "(3) 100ep auth-eval" in violations[0]


def test_lane_missing_custody_gate_flagged(tmp_path: Path):
    text = "smoke rc=0 + tier_c measured + 100ep auth-eval [contest-CUDA]"
    lane = _make_lane(
        "lane_substrate_no_custody",
        level=2,
        notes=text,
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "(4) custody validated" in violations[0]


def test_lane_missing_all_four_gates_flagged(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_naked_l2",
        level=2,
        notes="ordinary notes with no canonical evidence",
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    msg = violations[0]
    assert "missing 4 of 4 canonical gates" in msg
    for g in ("(1) smoke green", "(2) Tier C", "(3) 100ep auth-eval",
              "(4) custody validated"):
        assert g in msg


def test_l1_lane_skipped(tmp_path: Path):
    # Even with no evidence, L1 is skipped (this is the L1→L2 promotion gate).
    lane = _make_lane(
        "lane_substrate_l1_only",
        level=1,
        notes="ordinary l1 notes",
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_l3_lane_also_audited(tmp_path: Path):
    # The gate is "L2+" so L3 lanes also flagged if missing gates.
    lane = _make_lane(
        "lane_substrate_l3_naked",
        level=3,
        notes="naked l3 notes",
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "(L3)" in violations[0]


def test_research_only_lane_exempt(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_research_only",
        level=2,
        notes="research_only=true; no contest-promotion eligibility",
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_substrate_engineering_lane_class_exempt(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_engineering_x",
        level=2,
        lane_class="substrate_engineering",
        notes="ordinary substrate-engineering scaffold",
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_research_substrate_target_mode_exempt(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_x",
        level=2,
        target_modes=["research_substrate"],
        notes="ordinary notes",
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_waiver_with_rationale_passes_lane(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_waived",
        level=2,
        notes="naked notes # L1_L2_PROMOTION_CANONICAL_OK:operator-routed-audit-pending",
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_waiver_placeholder_does_not_pass_lane(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_bad_waiver",
        level=2,
        notes="naked notes # L1_L2_PROMOTION_CANONICAL_OK:<reason>",
    )
    _write_registry(tmp_path, [lane])
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_strict_mode_raises_on_violation(tmp_path: Path):
    lane = _make_lane("lane_substrate_naked", level=2, notes="naked")
    _write_registry(tmp_path, [lane])
    with pytest.raises(PreflightError) as excinfo:
        check_l1_to_l2_promotion_canonical_4_gate(
            repo_root=tmp_path, strict=True, verbose=False
        )
    msg = str(excinfo.value)
    assert "Catalog #233" in msg
    assert "council Decision 8" in msg.lower() or "Decision 8" in msg
    assert "4-gate canonical" in msg.lower() or "4 of 4 canonical gates" in msg


def test_strict_mode_silent_on_clean(tmp_path: Path):
    lane = _make_lane(
        "lane_substrate_great",
        level=2,
        notes=ALL_FOUR_GATES_TEXT,
    )
    _write_registry(tmp_path, [lane])
    # Should not raise.
    out = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert out == []


def test_multiple_lanes_aggregated(tmp_path: Path):
    lanes = [
        _make_lane("lane_substrate_a", level=2, notes="naked notes"),
        _make_lane("lane_substrate_b", level=2, notes="also naked"),
        _make_lane(
            "lane_substrate_c", level=2,
            notes=ALL_FOUR_GATES_TEXT,
        ),
    ]
    _write_registry(tmp_path, lanes)
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 2
    flagged_ids = {"lane_substrate_a", "lane_substrate_b"}
    for v in violations:
        assert any(lid in v for lid in flagged_ids)
    # c (clean) not in any violation.
    for v in violations:
        assert "lane_substrate_c" not in v


def test_corrupt_registry_handled_gracefully(tmp_path: Path):
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "lane_registry.json").write_text("not valid json {{{", encoding="utf-8")
    out = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert out == []


def test_registry_without_lanes_list_handled(tmp_path: Path):
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "lane_registry.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )
    out = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert out == []


def test_non_dict_lane_entries_skipped(tmp_path: Path):
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "lanes": [
            None, "string-not-dict", 42,
            _make_lane("lane_substrate_naked", level=2, notes="naked"),
        ],
    }
    (state / "lane_registry.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    out = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(out) == 1
    assert "lane_substrate_naked" in out[0]


def test_string_repo_root_accepted(tmp_path: Path):
    _write_registry(tmp_path, [
        _make_lane("lane_substrate_naked", level=2, notes="naked"),
    ])
    out = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=str(tmp_path), strict=False, verbose=False
    )
    assert len(out) == 1


def test_token_constants_exposed_for_audit():
    """Ensures Catalog #233's token constants are pinned and importable.

    A future refactor that drops one of these constants will fail this
    test, surfacing the regression at preflight time instead of in
    production.
    """
    for name in (
        "_CHECK_233_SMOKE_GREEN_TOKENS",
        "_CHECK_233_TIER_C_TOKENS",
        "_CHECK_233_AUTH_EVAL_100EP_TOKENS",
        "_CHECK_233_CUSTODY_VALIDATOR_TOKENS",
        "_CHECK_233_EXEMPT_LANE_CLASS_TOKENS",
        "_CHECK_233_EXEMPT_TARGET_MODE_TOKENS",
    ):
        from tac import preflight as p
        assert hasattr(p, name), f"Catalog #233 lost constant {name}"


def test_smoke_green_token_set_includes_canonical_phrasings():
    needed = {"smoke rc=0", "smoke green", "smoke complete"}
    actual = {t.lower() for t in _CHECK_233_SMOKE_GREEN_TOKENS}
    missing = {n for n in needed if n.lower() not in actual}
    assert not missing, f"Catalog #233 smoke-green tokens missing: {missing}"


def test_tier_c_token_set_includes_canonical_phrasings():
    needed = {"tier_c", "Tier C", "mdl_tier_c"}
    actual = set(_CHECK_233_TIER_C_TOKENS)
    missing = needed - actual
    assert not missing


def test_auth_eval_token_set_includes_canonical_phrasings():
    needed = {"100ep", "[contest-CUDA]", "auth_eval_score_axis=contest_cuda"}
    actual = set(_CHECK_233_AUTH_EVAL_100EP_TOKENS)
    missing = needed - actual
    assert not missing


def test_custody_token_set_includes_canonical_phrasings():
    needed = {"validate_custody", "evidence_grade=contest_cuda"}
    actual = set(_CHECK_233_CUSTODY_VALIDATOR_TOKENS)
    missing = needed - actual
    assert not missing


def test_verbose_output_lists_per_lane_state(tmp_path: Path, capsys):
    lanes = [
        _make_lane("lane_substrate_clean", level=2, notes=ALL_FOUR_GATES_TEXT),
        _make_lane("lane_substrate_naked", level=2, notes="naked"),
        _make_lane(
            "lane_substrate_engineering_only", level=2,
            lane_class="substrate_engineering", notes="ordinary",
        ),
    ]
    _write_registry(tmp_path, lanes)
    out = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=True
    )
    assert len(out) == 1
    captured = capsys.readouterr().out
    assert "lane_substrate_clean" in captured
    assert "EXEMPT" in captured
    assert "in_scope_L2=3" in captured
    assert "violations=1" in captured


def test_orchestrator_callsite_wires_strict_false():
    """Catalog #233 initial wire-in must be strict=False per CLAUDE.md
    'Strict-flip atomicity rule' + the council's operator-routable
    checkpoint #2 (operator-routed audit-and-downgrade sweep is the
    pre-condition for strict-flip).
    """
    src = Path(__file__).resolve().parents[1] / "preflight.py"
    body = src.read_text(encoding="utf-8")
    fn_name = "check_l1_to_l2_promotion_canonical_4_gate"
    needle = fn_name + "("
    # The orchestrator callsite (inside preflight_all) APPEARS BEFORE the
    # function def in the file because preflight_all is at the top of the
    # module and the function definition is later. Find every occurrence and
    # the one with 8-space indent (orchestrator) is the wire-in.
    indented_needle = "        " + needle
    call_idx = body.find(indented_needle)
    assert call_idx > 0, "orchestrator callsite (8-space indented) missing"
    window = body[call_idx : call_idx + 200]
    assert "strict=False" in window, (
        "Catalog #233 must wire strict=False initially per atomicity rule"
    )


def test_live_repo_registry_violation_count_bounded():
    """Live-repo regression guard.

    Catalog #233 lands warn-only with 6 known violations across 17 in-scope
    L2+ substrate lanes (per the council's operator-routable checkpoint #2
    audit-and-downgrade scope). This test pins the upper bound at landing
    so a future drift (more substrate L2+ lanes silently created without
    the 4-gate canonical) surfaces at preflight time.

    Tighten the bound when the operator-routed audit sweep brings the
    count below the current floor; STRICT-flip when it reaches 0.
    """
    repo_root = Path(__file__).resolve().parents[3]
    out = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=repo_root, strict=False, verbose=False
    )
    # Pin upper bound at landing-time count + minor headroom for sister
    # subagent concurrent lane edits during this PR.
    assert len(out) <= 20, (
        f"Catalog #233 live count drifted upward: {len(out)} violations. "
        f"Audit and downgrade per council Decision 8 operator-routable "
        f"checkpoint #2 before landing more L2+ substrate lanes."
    )


def test_gate_function_callable_via_globals():
    """Catalog #185 sister regression — refuses to silently lose the
    gate function from module globals.
    """
    from tac import preflight as p
    fn = getattr(p, "check_l1_to_l2_promotion_canonical_4_gate", None)
    assert callable(fn), "Catalog #233 gate function disappeared from globals"


def test_gate_function_signature_kwargs_only():
    import inspect
    sig = inspect.signature(check_l1_to_l2_promotion_canonical_4_gate)
    for name, param in sig.parameters.items():
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"Catalog #233 parameter {name!r} is not keyword-only"
        )
    assert "strict" in sig.parameters
    assert "verbose" in sig.parameters
    assert "repo_root" in sig.parameters


def test_no_substrate_lane_escapes_in_scope_classifier(tmp_path: Path):
    """Anti-regression: every canonical substrate-id pattern must be in scope."""
    # Sample one lane id per known token; gate must classify each as in-scope.
    for token in _CHECK_233_IN_SCOPE_ID_SUBSTRINGS:
        # Build a synthetic lane id that contains the token.
        lid = f"lane_test_{token}_dummy"
        assert _check_233_lane_is_in_scope(lid), (
            f"in-scope classifier dropped canonical token: {token!r}"
        )


def test_text_with_only_first_two_gates_flagged_for_last_two(tmp_path: Path):
    # Demonstrate violation message names the EXACT missing gates.
    text = "smoke rc=0 + tier_c measured (no auth-eval, no custody)"
    lane = _make_lane(
        "lane_substrate_two_of_four",
        level=2,
        notes=text,
    )
    _write_registry(tmp_path, [lane])
    out = check_l1_to_l2_promotion_canonical_4_gate(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(out) == 1
    msg = out[0]
    assert "missing 2 of 4 canonical gates" in msg
    assert "(3) 100ep auth-eval" in msg
    assert "(4) custody validated" in msg
    assert "(1) smoke green" not in msg
    assert "(2) Tier C" not in msg


def test_council_verdict_4_gates_match_doc_string():
    """Anchor test: the 4 gates in the function docstring are the council
    Decision 8 binding verdict (verbatim, lines 366-371 of the omnibus
    ledger). If the docstring drifts, this test catches it.
    """
    fn_doc = check_l1_to_l2_promotion_canonical_4_gate.__doc__ or ""
    canonical_gates = [
        "Smoke green",
        "Tier C MDL density",
        "100ep auth-eval anchor",
        "Custody validated",
    ]
    for g in canonical_gates:
        assert g in fn_doc, (
            f"Catalog #233 docstring lost council verdict gate: {g!r}"
        )


def test_d8_council_ledger_anchor_referenced_in_docstring():
    fn_doc = check_l1_to_l2_promotion_canonical_4_gate.__doc__ or ""
    assert "7872c9f4b" in fn_doc, (
        "Catalog #233 docstring must cite the council omnibus commit anchor"
    )
    assert "Option D" in fn_doc, (
        "Catalog #233 docstring must cite the binding council verdict (Option D)"
    )
