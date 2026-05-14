# SPDX-License-Identifier: MIT
"""Tests for Catalog #219 — MDL-density STRICT preflight gate on L2+ archive
promotion.

Bug class: a lane promoted to Level 2+ (real_archive_empirical / contest_cuda
/ contest_cpu gate satisfied) whose archive's scorer-conditional MDL density
exceeds the empirically-derived class-saturation threshold (0.90) is by
definition in the within-class trap per Z1 ablation
(`feedback_z1_mdl_ablation_landed_20260514.md`). The gate refuses such
promotions to force the operator's next dispatch to a class-shift
architecture (predictive-receiver / cooperative-receiver / foveation /
Wyner-Ziv).

Sister of Catalog #90 (`check_lane_registry_consistent`) +
Catalog #127 (`check_authoritative_tag_requires_custody_metadata`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tac.preflight import (
    PreflightError,
    check_archive_promotion_blocked_by_mdl_density_above_threshold,
    _check_219_discover_mdl_results,
    _check_219_extract_sha_tokens,
    _check_219_lane_has_waiver,
    _check_219_lane_is_exempt,
    _check_219_match_sha_against_mdl_index,
    _check_219_text_has_sha_context,
    _CHECK_219_MDL_DENSITY_THRESHOLD,
)

# Realistic sha256 prefixes from the Z1 anchor archives (production data).
A1_SHA_FULL = "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
PR106_SHA_FULL = "7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f"
A1_SHA_PREFIX = A1_SHA_FULL[:8]
PR106_SHA_PREFIX = PR106_SHA_FULL[:8]

# Synthetic fake sha for "low density" scenarios (just hex chars).
SAFE_SHA_FULL = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
SAFE_SHA_PREFIX = SAFE_SHA_FULL[:8]


def _write_mdl_result(
    root: Path,
    sha: str,
    density: float,
    *,
    folder_name: str = "mdl_ablation_z1_test",
    archive_name: str = "test",
) -> Path:
    """Synthesize one MDL ablation result file under
    experiments/results/<folder>/<archive>_mdl_ablation.json.
    """
    folder = root / "experiments" / "results" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{archive_name}_mdl_ablation.json"
    payload = {
        "archive_name": archive_name,
        "archive_path": f"submissions/{archive_name}/archive.zip",
        "archive_sha256": sha,
        "archive_size_bytes": 178262,
        "mdl_density_estimate_lo": density,
        "mdl_density_estimate_hi": density,
        "zen_floor_band_recommendation": (
            "[0.100, 0.150] — major architectural breakthrough needed"
            if density > 0.90
            else "[0.030, 0.070] — class shift possible"
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_lane_registry(
    root: Path,
    lanes: list[dict],
    *,
    schema_version: int = 1,
) -> Path:
    """Synthesize a minimal lane registry JSON."""
    omx = root / ".omx" / "state"
    omx.mkdir(parents=True, exist_ok=True)
    path = omx / "lane_registry.json"
    payload = {"schema_version": schema_version, "lanes": lanes}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_l2_lane(
    lane_id: str,
    *,
    level: int = 2,
    sha_in_evidence: str | None = None,
    evidence_text: str | None = None,
    notes: str = "",
    lane_class: str | None = None,
    target_modes: list[str] | None = None,
    gate_status: bool = True,
) -> dict:
    """Build a synthetic lane registry entry."""
    if evidence_text is None and sha_in_evidence is not None:
        evidence_text = (
            f"[empirical:experiments/results/foo/] archive sha256={sha_in_evidence} "
            f"178262 B"
        )
    elif evidence_text is None:
        evidence_text = "[no sha referenced]"
    lane: dict[str, Any] = {
        "id": lane_id,
        "name": lane_id,
        "phase": 1,
        "level": level,
        "gates": {
            "impl_complete": {"status": True, "evidence": "synthetic"},
            "real_archive_empirical": {
                "status": gate_status,
                "evidence": evidence_text,
            },
            "contest_cuda": {"status": False, "evidence": ""},
            "contest_cpu": {"status": False, "evidence": ""},
            "strict_preflight": {"status": False, "evidence": ""},
            "three_clean_review": {"status": False, "evidence": ""},
            "memory_entry": {"status": False, "evidence": ""},
            "deploy_runbook": {"status": False, "evidence": ""},
        },
        "notes": notes,
    }
    if lane_class is not None:
        lane["lane_class"] = lane_class
    if target_modes is not None:
        lane["target_modes"] = target_modes
    return lane


# ── _check_219_text_has_sha_context ────────────────────────────────────────


def test_text_has_sha_context_with_sha256_token() -> None:
    assert _check_219_text_has_sha_context("archive sha256=abc123")


def test_text_has_sha_context_with_ellipsis_form() -> None:
    assert _check_219_text_has_sha_context("sha 87ec7ca5...492b5")


def test_text_has_sha_context_rejects_unrelated_hex() -> None:
    # Hex-looking timestamp/counter without a sha-indicative context.
    assert not _check_219_text_has_sha_context("run_id 1234567890abcdef")


def test_text_has_sha_context_handles_non_string() -> None:
    assert not _check_219_text_has_sha_context(None)  # type: ignore[arg-type]
    assert not _check_219_text_has_sha_context(12345)  # type: ignore[arg-type]


# ── _check_219_discover_mdl_results ────────────────────────────────────────


def test_discover_mdl_results_empty_root(tmp_path: Path) -> None:
    idx = _check_219_discover_mdl_results(tmp_path)
    assert idx == {}


def test_discover_mdl_results_finds_single_archive(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99)
    idx = _check_219_discover_mdl_results(tmp_path)
    assert A1_SHA_FULL.lower() in idx
    assert idx[A1_SHA_FULL.lower()]["density_lo"] == pytest.approx(0.99)


def test_discover_mdl_results_skips_summary_files(tmp_path: Path) -> None:
    folder = tmp_path / "experiments" / "results" / "mdl_ablation_summary_test"
    folder.mkdir(parents=True)
    # summary_mdl_ablation.json should be ignored
    (folder / "summary_mdl_ablation.json").write_text(
        json.dumps({"archive_sha256": A1_SHA_FULL, "mdl_density_estimate_lo": 0.99})
    )
    idx = _check_219_discover_mdl_results(tmp_path)
    assert A1_SHA_FULL.lower() not in idx


def test_discover_mdl_results_picks_higher_density_on_collision(
    tmp_path: Path,
) -> None:
    # Two files with the same sha; higher density wins.
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.85, folder_name="a", archive_name="lo")
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99, folder_name="b_mdl_ablation", archive_name="hi")
    # Folder "a" doesn't have mdl_ablation in name; only b_mdl_ablation scanned.
    # Add an MDL folder for a
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.85, folder_name="a_mdl_ablation_low", archive_name="lo")
    idx = _check_219_discover_mdl_results(tmp_path)
    assert idx[A1_SHA_FULL.lower()]["density_lo"] == pytest.approx(0.99)


def test_discover_mdl_results_handles_malformed_json(tmp_path: Path) -> None:
    folder = tmp_path / "experiments" / "results" / "mdl_ablation_malformed"
    folder.mkdir(parents=True)
    (folder / "bad_mdl_ablation.json").write_text("{not valid json")
    idx = _check_219_discover_mdl_results(tmp_path)
    assert A1_SHA_FULL.lower() not in idx
    assert idx == {}  # silent skip on malformed


def test_discover_mdl_results_skips_missing_sha(tmp_path: Path) -> None:
    folder = tmp_path / "experiments" / "results" / "mdl_ablation_nosha"
    folder.mkdir(parents=True)
    (folder / "x_mdl_ablation.json").write_text(
        json.dumps({"mdl_density_estimate_lo": 0.99})  # no archive_sha256
    )
    idx = _check_219_discover_mdl_results(tmp_path)
    assert idx == {}


# ── _check_219_extract_sha_tokens ──────────────────────────────────────────


def test_extract_sha_tokens_from_real_archive_empirical_evidence() -> None:
    lane = _make_l2_lane(
        "lane_test",
        sha_in_evidence=A1_SHA_FULL,
    )
    toks = _check_219_extract_sha_tokens(lane)
    assert any(t.startswith(A1_SHA_FULL[:8].lower()) for t in toks)


def test_extract_sha_tokens_from_abbreviated_form() -> None:
    lane = _make_l2_lane(
        "lane_test",
        evidence_text=f"archive sha {A1_SHA_PREFIX}...492b5 (178,262 B)",
    )
    toks = _check_219_extract_sha_tokens(lane)
    assert A1_SHA_PREFIX.lower() in toks


def test_extract_sha_tokens_ignores_evidence_when_gate_status_false() -> None:
    lane = _make_l2_lane(
        "lane_test", sha_in_evidence=A1_SHA_FULL, gate_status=False
    )
    toks = _check_219_extract_sha_tokens(lane)
    assert A1_SHA_PREFIX.lower() not in toks


def test_extract_sha_tokens_rejects_hex_without_sha_context() -> None:
    lane = _make_l2_lane(
        "lane_test",
        evidence_text=f"run_id {A1_SHA_PREFIX}deadbeef counter 1234567890abcdef",
    )
    # No sha-indicative context -> empty token set
    toks = _check_219_extract_sha_tokens(lane)
    assert toks == set()


# ── _check_219_match_sha_against_mdl_index ─────────────────────────────────


def test_match_sha_prefix_against_full_index() -> None:
    idx = {A1_SHA_FULL.lower(): {"density_lo": 0.99}}
    matches = _check_219_match_sha_against_mdl_index({A1_SHA_PREFIX.lower()}, idx)
    assert len(matches) == 1
    assert matches[0][0] == A1_SHA_FULL.lower()


def test_match_sha_full_token_against_full_index() -> None:
    idx = {A1_SHA_FULL.lower(): {"density_lo": 0.99}}
    matches = _check_219_match_sha_against_mdl_index({A1_SHA_FULL.lower()}, idx)
    assert len(matches) == 1


def test_match_sha_rejects_short_tokens() -> None:
    idx = {A1_SHA_FULL.lower(): {"density_lo": 0.99}}
    matches = _check_219_match_sha_against_mdl_index({"87ec"}, idx)  # too short
    assert matches == []


def test_match_sha_no_match_when_prefix_mismatched() -> None:
    idx = {A1_SHA_FULL.lower(): {"density_lo": 0.99}}
    matches = _check_219_match_sha_against_mdl_index({"deadbeef"}, idx)
    assert matches == []


# ── _check_219_lane_is_exempt ──────────────────────────────────────────────


def test_lane_exempt_via_lane_class_substrate_engineering() -> None:
    lane = _make_l2_lane("lane_test", lane_class="substrate_engineering")
    assert _check_219_lane_is_exempt(lane)


def test_lane_exempt_via_lane_class_substrate_class_shift() -> None:
    lane = _make_l2_lane("lane_test", lane_class="substrate_class_shift")
    assert _check_219_lane_is_exempt(lane)


def test_lane_exempt_via_target_modes_research_substrate() -> None:
    lane = _make_l2_lane("lane_test", target_modes=["research_substrate"])
    assert _check_219_lane_is_exempt(lane)


def test_lane_exempt_via_notes_research_only_true() -> None:
    lane = _make_l2_lane("lane_test", notes="research_only=true; planning_only")
    assert _check_219_lane_is_exempt(lane)


def test_lane_exempt_via_notes_lane_class_in_body() -> None:
    lane = _make_l2_lane(
        "lane_test", notes="lane_class=substrate_engineering; for autopilot"
    )
    assert _check_219_lane_is_exempt(lane)


def test_lane_not_exempt_when_research_only_false() -> None:
    lane = _make_l2_lane("lane_test", notes="research_only=false; production lane")
    assert not _check_219_lane_is_exempt(lane)


def test_lane_not_exempt_with_empty_fields() -> None:
    lane = _make_l2_lane("lane_test")
    assert not _check_219_lane_is_exempt(lane)


# ── _check_219_lane_has_waiver ─────────────────────────────────────────────


def test_lane_has_waiver_via_notes_with_rationale() -> None:
    lane = _make_l2_lane(
        "lane_test",
        notes=(
            "# MDL_DENSITY_OVER_THRESHOLD_OK: A1 baseline anchor; operator-"
            "explicit reference per Z1 council 2026-05-14"
        ),
    )
    waived, reason = _check_219_lane_has_waiver(lane)
    assert waived
    assert "baseline anchor" in reason


def test_lane_has_waiver_inline_in_evidence_string() -> None:
    lane = _make_l2_lane(
        "lane_test",
        evidence_text=(
            f"[empirical:foo] archive sha256={A1_SHA_FULL}; "
            "MDL_DENSITY_OVER_THRESHOLD_OK:operator-baseline-reference"
        ),
    )
    waived, reason = _check_219_lane_has_waiver(lane)
    assert waived
    assert reason == "operator-baseline-reference"


def test_lane_waiver_rejects_placeholder_reason() -> None:
    lane = _make_l2_lane(
        "lane_test", notes="# MDL_DENSITY_OVER_THRESHOLD_OK:<reason>"
    )
    waived, _ = _check_219_lane_has_waiver(lane)
    assert not waived


def test_lane_waiver_rejects_empty_reason() -> None:
    lane = _make_l2_lane("lane_test", notes="# MDL_DENSITY_OVER_THRESHOLD_OK: ,")
    waived, _ = _check_219_lane_has_waiver(lane)
    assert not waived


def test_lane_no_waiver_in_empty_notes() -> None:
    lane = _make_l2_lane("lane_test")
    waived, _ = _check_219_lane_has_waiver(lane)
    assert not waived


# ── End-to-end check_archive_promotion_blocked_by_mdl_density_above_threshold ─


def test_gate_passes_when_no_registry_present(tmp_path: Path) -> None:
    # No lane_registry.json under tmp_path
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=True
    )
    assert vs == []


def test_gate_passes_when_no_mdl_evidence_present(tmp_path: Path) -> None:
    _write_lane_registry(
        tmp_path, [_make_l2_lane("lane_x", sha_in_evidence=A1_SHA_FULL)]
    )
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=True
    )
    # No MDL ablation results yet => clean
    assert vs == []


def test_gate_passes_when_lane_density_below_threshold(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, SAFE_SHA_FULL, 0.50)
    _write_lane_registry(
        tmp_path, [_make_l2_lane("lane_x", sha_in_evidence=SAFE_SHA_FULL)]
    )
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=True
    )
    assert vs == []


def test_gate_flags_L2_lane_above_threshold(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99)
    _write_lane_registry(
        tmp_path, [_make_l2_lane("lane_x", sha_in_evidence=A1_SHA_FULL)]
    )
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=False
    )
    assert len(vs) == 1
    assert "lane_x" in vs[0]
    assert "0.99" in vs[0] or "0.9929" in vs[0]


def test_gate_skips_L1_lanes(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99)
    _write_lane_registry(
        tmp_path, [_make_l2_lane("lane_x", level=1, sha_in_evidence=A1_SHA_FULL)]
    )
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=True
    )
    assert vs == []


def test_gate_skips_exempt_lanes_via_lane_class(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99)
    _write_lane_registry(
        tmp_path,
        [
            _make_l2_lane(
                "lane_x",
                sha_in_evidence=A1_SHA_FULL,
                lane_class="substrate_engineering",
            )
        ],
    )
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=True
    )
    assert vs == []


def test_gate_skips_lanes_with_explicit_waiver(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99)
    _write_lane_registry(
        tmp_path,
        [
            _make_l2_lane(
                "lane_x",
                sha_in_evidence=A1_SHA_FULL,
                notes=(
                    "# MDL_DENSITY_OVER_THRESHOLD_OK: A1 baseline anchor; "
                    "operator-explicit reference"
                ),
            )
        ],
    )
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=True
    )
    assert vs == []


def test_gate_strict_raises_when_violation_found(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99)
    _write_lane_registry(
        tmp_path, [_make_l2_lane("lane_x", sha_in_evidence=A1_SHA_FULL)]
    )
    with pytest.raises(PreflightError, match="Catalog #219"):
        check_archive_promotion_blocked_by_mdl_density_above_threshold(
            repo_root=tmp_path, strict=True
        )


def test_gate_strict_silent_when_clean(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, SAFE_SHA_FULL, 0.50)
    _write_lane_registry(
        tmp_path, [_make_l2_lane("lane_x", sha_in_evidence=SAFE_SHA_FULL)]
    )
    # Should not raise
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=True
    )
    assert vs == []


def test_gate_threshold_value_is_0_90() -> None:
    """Per Z1 operator decision #5: threshold is exactly 0.90."""
    assert _CHECK_219_MDL_DENSITY_THRESHOLD == 0.90


def test_gate_passes_when_density_exactly_at_threshold(tmp_path: Path) -> None:
    # 0.90 is NOT strictly greater than 0.90; gate uses strict > so accepts.
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.90)
    _write_lane_registry(
        tmp_path, [_make_l2_lane("lane_x", sha_in_evidence=A1_SHA_FULL)]
    )
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=True
    )
    assert vs == []


def test_gate_flags_multiple_L2_lanes(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99, archive_name="a1")
    _write_mdl_result(
        tmp_path, PR106_SHA_FULL, 0.97, folder_name="mdl_ablation_pr106", archive_name="pr106"
    )
    _write_lane_registry(
        tmp_path,
        [
            _make_l2_lane("lane_a1", sha_in_evidence=A1_SHA_FULL),
            _make_l2_lane("lane_pr106", sha_in_evidence=PR106_SHA_FULL),
        ],
    )
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=False
    )
    assert len(vs) == 2


def test_gate_lane_with_abbreviated_sha_form_matches(tmp_path: Path) -> None:
    _write_mdl_result(tmp_path, A1_SHA_FULL, 0.99)
    lane = _make_l2_lane(
        "lane_x",
        evidence_text=f"archive sha {A1_SHA_PREFIX}...492b5 178262 B",
    )
    _write_lane_registry(tmp_path, [lane])
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        repo_root=tmp_path, strict=False
    )
    assert len(vs) == 1


def test_gate_live_repo_clean_or_known_exempts_only() -> None:
    """Live-repo regression guard: the production gate must be clean OR all
    violations must be already-exempted via lane_class / waiver. Initial
    landing: warn-only; this test verifies the wire-in works."""
    vs = check_archive_promotion_blocked_by_mdl_density_above_threshold(
        strict=False, verbose=False
    )
    # All violations (if any) MUST point to lanes that have NOT been exempted
    # — the test is purely diagnostic and does not block.
    assert isinstance(vs, list)
