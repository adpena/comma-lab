"""Tests for Catalog #220 — substrate scaffold byte addition without
operational score-improvement mechanism.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable + Catalog #220. Anchor: D1 R3 dispatch 2026-05-14 score
~0.222 vs predicted [0.181, 0.188] because the L1 SCAFFOLD landed a 43 KB
sidecar that the inflate runtime did NOT consume operationally.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_220_extract_byte_addition_kb,
    _check_220_lane_has_operational_signal,
    _check_220_lane_has_waiver,
    _check_220_lane_is_in_scope,
    _check_220_lane_is_pre_build_substrate_engineering,
    _check_220_lane_is_research_only,
    check_substrate_l1_scaffold_no_byte_addition_without_operational_score_improvement_mechanism as check_220,
)


def _write_registry(
    tmp_path: Path, lanes: list[dict], schema_version: int = 1
) -> Path:
    """Helper: write a synthetic lane registry to tmp_path."""
    repo_root = tmp_path
    state_dir = repo_root / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = state_dir / "lane_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "updated_at": "2026-05-14T00:00:00Z",
                "description": "test fixture",
                "gate_definitions": {},
                "level_rules": {},
                "lanes": lanes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return repo_root


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lane_id,expected",
    [
        ("lane_substrate_pr101_lc_v2_clone_20260512", True),
        ("lane_d1_segnet_margin_polytope_encoder_20260514", True),
        ("lane_yucr_substrate_20260514", True),
        ("lane_a1_plus_lapose_20260512", True),
        ("lane_pr106_siren_residual_sidecar_20260513", True),
        ("lane_pr106_wavelet_residual_sidecar_20260513", True),
        # Out-of-scope (infrastructure / META / catalog gates).
        ("lane_lane_registry_consistent", False),
        ("lane_meta_lagrangian_atom_emitter", False),
        ("lane_random_unrelated_lane", False),
        ("", False),
    ],
)
def test_in_scope_classifier(lane_id, expected):
    assert _check_220_lane_is_in_scope(lane_id) is expected


@pytest.mark.parametrize(
    "text,expected_range",
    [
        ("+43 KB", (40.0, 50.0)),
        ("~43 KB", (40.0, 50.0)),
        ("~2.7 KB", (2.0, 3.0)),
        ("overhead ~43 KB", (40.0, 50.0)),
        ("sidecar 2 KB", (1.5, 2.5)),
        ("d1_overhead_bytes=43296", (40.0, 50.0)),
        ("archive_bytes_added=2700", (2.5, 3.0)),
        ("archive_bytes_added=~43 KB", (40.0, 50.0)),
        ("rate +0.029", (40.0, 50.0)),
        ("43,296 bytes", (40.0, 50.0)),
        ("43,296 B", (40.0, 50.0)),
    ],
)
def test_extract_byte_addition_kb_positive(text, expected_range):
    kb = _check_220_extract_byte_addition_kb(text)
    assert kb is not None
    assert expected_range[0] <= kb <= expected_range[1], (
        f"text={text!r} kb={kb} not in {expected_range}"
    )


def test_extract_byte_addition_kb_empty_text():
    assert _check_220_extract_byte_addition_kb("") is None
    assert _check_220_extract_byte_addition_kb("") is None


def test_extract_byte_addition_kb_no_signal():
    """Text with no byte-related tokens returns None."""
    assert _check_220_extract_byte_addition_kb(
        "lane impl_complete: 14 files; 89 tests pass"
    ) is None


def test_extract_byte_addition_kb_picks_largest():
    text = "overhead +43 KB or shrunk +2.7 KB; sidecar 1 KB"
    kb = _check_220_extract_byte_addition_kb(text)
    assert kb is not None
    assert kb >= 40.0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("score_improvement_mechanism_status=OPERATIONAL", True),
        ("L2_INTEGRATION_LANDED", True),
        ("l2_overlay_active", True),
        ("runtime_overlay_consumed=true", True),
        ("operational_overlay=true", True),
        ("score_improvement_mechanism_status=DEFERRED", False),
        ("L1 SCAFFOLD pending L2", False),
        ("", False),
    ],
)
def test_operational_signal(text, expected):
    assert _check_220_lane_has_operational_signal(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("research_only=true", True),
        ("research-only=true", True),
        ("RESEARCH_ONLY=TRUE", True),
        ("research_only=false", False),
        ("", False),
        ("ready_for_exact_eval_dispatch=true", False),
    ],
)
def test_research_only_signal(text, expected):
    assert _check_220_lane_is_research_only(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "lane_class=substrate_engineering; _full_main raises NotImplementedError",
            True,
        ),
        ("lane_class=substrate_engineering; full_path_council_gated", True),
        ("lane_class=substrate_engineering; scaffold_only", True),
        # Only one signal — not enough.
        ("lane_class=substrate_engineering", False),
        ("_full_main raises NotImplementedError", False),
        ("", False),
    ],
)
def test_pre_build_substrate_engineering_signal(text, expected):
    assert _check_220_lane_is_pre_build_substrate_engineering(text) is expected


def test_waiver_with_reason_accepted():
    text = "# SCAFFOLD_DEFERRED_INTEGRATION_OK: operator-approved L1 dispatch for empirical anchor only"
    waived, reason = _check_220_lane_has_waiver(text)
    assert waived is True
    assert "operator-approved" in reason


def test_waiver_placeholder_reason_rejected():
    text = "# SCAFFOLD_DEFERRED_INTEGRATION_OK: <reason>"
    waived, reason = _check_220_lane_has_waiver(text)
    assert waived is False


def test_waiver_empty_text_rejected():
    waived, reason = _check_220_lane_has_waiver("")
    assert waived is False
    assert reason == ""


# ---------------------------------------------------------------------------
# End-to-end gate tests
# ---------------------------------------------------------------------------


def test_gate_no_registry_returns_empty(tmp_path):
    violations = check_220(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_lane_below_l1_skipped(tmp_path):
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_20260514",
                "level": 0,  # below threshold
                "notes": "+43 KB sidecar without operational mechanism",
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert violations == []


def test_gate_lane_out_of_scope_skipped(tmp_path):
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_meta_lagrangian_atom_emitter_20260514",  # not substrate
                "level": 2,
                "notes": "+43 KB metadata sidecar",
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert violations == []


def test_gate_lane_no_byte_addition_skipped(tmp_path):
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_20260514",
                "level": 1,
                "notes": "lane scaffold with no byte addition signal",
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert violations == []


def test_gate_lane_byte_addition_below_threshold_skipped(tmp_path):
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_20260514",
                "level": 1,
                "notes": "tiny probe packet ~0.3 KB",
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert violations == []


def test_gate_refuses_substrate_with_byte_addition_no_mechanism(tmp_path):
    """The D1 R3 bug class: substrate adds ~43 KB but no operational mechanism."""
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_buggy_20260514",
                "level": 1,
                "notes": "L1 SCAFFOLD adds ~43 KB sidecar; overlay no-op-by-default",
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert len(violations) == 1
    assert "lane_substrate_test_buggy_20260514" in violations[0]
    assert "43.0 KB" in violations[0] or "43 KB" in violations[0]


def test_gate_accepts_operational_mechanism_declared(tmp_path):
    """D1 post-L2-INTEGRATION: byte addition + operational mechanism = accept."""
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_operational_20260514",
                "level": 1,
                "notes": (
                    "+43 KB sidecar; score_improvement_mechanism_status=OPERATIONAL "
                    "via tac.substrates.test.overlay; runtime_overlay_consumed=true"
                ),
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert violations == []


def test_gate_accepts_research_only(tmp_path):
    """research_only=true bypasses the gate."""
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_research_only_20260514",
                "level": 1,
                "notes": "+43 KB experimental sidecar; research_only=true",
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert violations == []


def test_gate_accepts_pre_build_substrate_engineering(tmp_path):
    """Pre-build substrate-engineering scaffold gated by NotImplementedError."""
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_pre_build_20260514",
                "level": 1,
                "notes": (
                    "+43 KB sidecar designed; lane_class=substrate_engineering; "
                    "_full_main raises NotImplementedError pending council "
                    "deliberation on 5 design verdicts"
                ),
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert violations == []


def test_gate_accepts_waiver_with_real_reason(tmp_path):
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_waived_20260514",
                "level": 1,
                "notes": (
                    "+43 KB sidecar; "
                    "# SCAFFOLD_DEFERRED_INTEGRATION_OK: operator-approved L1 "
                    "dispatch for empirical proxy-vs-actual band check"
                ),
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert violations == []


def test_gate_rejects_placeholder_waiver(tmp_path):
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_bad_waiver_20260514",
                "level": 1,
                "notes": (
                    "+43 KB sidecar; "
                    "# SCAFFOLD_DEFERRED_INTEGRATION_OK: <reason>"
                ),
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert len(violations) == 1


def test_gate_substrate_engineering_alone_insufficient(tmp_path):
    """lane_class=substrate_engineering without pre-build signal is NOT enough."""
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_engineering_only_20260514",
                "level": 1,
                "notes": "+43 KB sidecar; lane_class=substrate_engineering",
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert len(violations) == 1


def test_gate_strict_mode_raises(tmp_path):
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_strict_raise_20260514",
                "level": 1,
                "notes": "+43 KB sidecar; no operational mechanism",
                "gates": {},
            },
        ],
    )
    with pytest.raises(PreflightError, match="Catalog #220"):
        check_220(repo_root=repo_root, strict=True)


def test_gate_strict_mode_clean_passes(tmp_path):
    """Strict mode does NOT raise when no violations."""
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_clean_20260514",
                "level": 1,
                "notes": (
                    "+43 KB sidecar; "
                    "score_improvement_mechanism_status=OPERATIONAL"
                ),
                "gates": {},
            },
        ],
    )
    # Should not raise.
    assert check_220(repo_root=repo_root, strict=True) == []


def test_gate_aggregates_multiple_violations(tmp_path):
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_bad_a_20260514",
                "level": 1,
                "notes": "+10 KB sidecar without mechanism",
                "gates": {},
            },
            {
                "id": "lane_substrate_bad_b_20260514",
                "level": 1,
                "notes": "+30 KB sidecar without mechanism",
                "gates": {},
            },
            {
                "id": "lane_substrate_good_20260514",
                "level": 1,
                "notes": "+5 KB sidecar; runtime_overlay_consumed=true",
                "gates": {},
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert len(violations) == 2


def test_gate_evidence_strings_in_gates_are_scanned(tmp_path):
    """Byte addition declared in gate evidence (not notes) is also detected."""
    repo_root = _write_registry(
        tmp_path,
        [
            {
                "id": "lane_substrate_test_gate_evidence_20260514",
                "level": 1,
                "notes": "",
                "gates": {
                    "impl_complete": {
                        "status": True,
                        "evidence": "shipped +43 KB sidecar without mechanism",
                    }
                },
            },
        ],
    )
    violations = check_220(repo_root=repo_root, strict=False)
    assert len(violations) == 1


def test_gate_corrupted_registry_returns_empty(tmp_path):
    """Bad JSON does not raise — gate degrades gracefully (consistent with
    sister Catalog #219 behavior)."""
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "lane_registry.json").write_text(
        "not valid json {", encoding="utf-8"
    )
    violations = check_220(repo_root=tmp_path, strict=False)
    assert violations == []


def test_gate_live_repo_live_count_zero():
    """Live-repo regression guard: live count MUST be 0 per Strict-flip
    atomicity rule. Any future regression that introduces a substrate scaffold
    with byte addition >1 KB without operational mechanism will surface here."""
    violations = check_220(strict=False)
    assert len(violations) == 0, (
        f"Live repo Catalog #220 violations: {violations[:3]}"
    )
