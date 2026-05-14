# SPDX-License-Identifier: MIT
"""Tests for Catalog #227 — substrate-class-shift promotion requires Tier C
empirical evidence.

Sister of Catalog #219 (Tier A density gate). Where #219 refuses L2+
promotion when Tier A density > 0.90, #227 refuses substrate-class-shift
PROMOTION (lanes claiming `lane_class` ∈ class-shift tokens OR `notes`
referencing class-shift literature anchors) at L2+ WITHOUT Tier C
empirical evidence.

Bug class: Tier A is brotli-saturated at the byte layer (any fp16-weight +
brotli archive sits at Tier A density ~0.99) and structurally CANNOT
discriminate substrate class. A lane can claim ACROSS-CLASS lineage via
its lane_class / notes / literature_anchor field WITHOUT empirically
demonstrating the class-shift signature via Tier C post-decode
perturbation. The C6 5ep empirical anchor
(`feedback_mdl_ablation_tier_c_ibps1_landed_20260514.md`) showed the
dispositive test IS Tier C — the latent Δscore at σ=1.0 must be sub-0.05
AND state_dict perturbation must show a knee+plateau structural signature.

Memory: feedback_autopilot_tier_c_integration_catalog_227_landed_20260514.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_class_promotion_requires_tier_c_evidence,
    _check_227_lane_is_class_shift_claim,
    _check_227_lane_has_tier_c_evidence,
    _check_227_lane_has_waiver,
    _check_227_discover_tier_c_results,
    _CHECK_227_CLASS_SHIFT_LANE_CLASS_TOKENS,
    _CHECK_227_CLASS_SHIFT_NOTES_TOKENS,
    _CHECK_227_TIER_C_TOKENS,
)


# Synthetic sha for tests.
C6_SHA_FULL = "a27328ce02211f1c8ee0cfb4318ace29c438a62cf09a42358481d0273a204607"
SAFE_SHA_FULL = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"


def _write_mdl_result_with_tier_c(
    root: Path,
    sha: str,
    *,
    folder_name: str = "mdl_ablation_z1_tier_c_test",
    archive_name: str = "test",
    tier_c_rows: list[dict] | None = None,
    tier_c_density: float = 0.13,
    tier_c_verdict: str = "across_class",
) -> Path:
    """Synthesize one MDL ablation result file with a non-empty tier_c list."""
    folder = root / "experiments" / "results" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{archive_name}_mdl_ablation.json"
    if tier_c_rows is None:
        # Canonical 8-row Tier C schema (4 sigmas × 2 targets) matching the
        # C6 5ep anchor shape.
        tier_c_rows = [
            {
                "target": "state_dict",
                "noise_sigma_relative": 0.001,
                "delta_seg": 0.0,
                "delta_pose": -0.003,
                "delta_score_components": -0.005,
                "elapsed_seconds": 23.0,
            },
            {
                "target": "latents",
                "noise_sigma_relative": 0.001,
                "delta_seg": 0.0,
                "delta_pose": 0.0005,
                "delta_score_components": 0.0008,
                "elapsed_seconds": 22.5,
            },
            {
                "target": "state_dict",
                "noise_sigma_relative": 0.01,
                "delta_seg": 0.0,
                "delta_pose": -0.048,
                "delta_score_components": -0.074,
                "elapsed_seconds": 20.9,
            },
            {
                "target": "latents",
                "noise_sigma_relative": 0.01,
                "delta_seg": 0.0,
                "delta_pose": -0.0012,
                "delta_score_components": -0.0018,
                "elapsed_seconds": 21.1,
            },
            {
                "target": "state_dict",
                "noise_sigma_relative": 0.1,
                "delta_seg": 0.0,
                "delta_pose": -0.041,
                "delta_score_components": -0.063,
                "elapsed_seconds": 20.9,
            },
            {
                "target": "latents",
                "noise_sigma_relative": 0.1,
                "delta_seg": 0.0,
                "delta_pose": 0.004,
                "delta_score_components": 0.006,
                "elapsed_seconds": 20.7,
            },
            {
                "target": "state_dict",
                "noise_sigma_relative": 1.0,
                "delta_seg": 0.0,
                "delta_pose": 95.5,
                "delta_score_components": 27.8,
                "elapsed_seconds": 20.9,
            },
            {
                "target": "latents",
                "noise_sigma_relative": 1.0,
                "delta_seg": 0.0,
                "delta_pose": -0.0014,
                "delta_score_components": -0.00213,
                "elapsed_seconds": 20.5,
            },
        ]
    payload: dict[str, Any] = {
        "archive_name": archive_name,
        "archive_path": f"submissions/{archive_name}/archive.zip",
        "archive_sha256": sha,
        "archive_size_bytes": 224481,
        "mdl_density_estimate_lo": 0.99,  # Tier A saturated
        "mdl_density_estimate_hi": 0.99,
        "mdl_tier_c_density_estimate": tier_c_density,
        "mdl_tier_c_substrate_class_verdict": tier_c_verdict,
        "tier_c": tier_c_rows,
        "tier_a": [],
        "tier_b": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_lane_registry(
    root: Path,
    lanes: list[dict],
    *,
    schema_version: int = 1,
) -> Path:
    omx = root / ".omx" / "state"
    omx.mkdir(parents=True, exist_ok=True)
    path = omx / "lane_registry.json"
    payload = {"schema_version": schema_version, "lanes": lanes}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_class_shift_l2_lane(
    lane_id: str,
    *,
    level: int = 2,
    lane_class: str | None = "substrate_class_shift",
    notes: str = "",
    sha_in_evidence: str | None = None,
    evidence_text: str | None = None,
    gate_status: bool = True,
) -> dict:
    """Build a synthetic L2+ lane that CLAIMS substrate-class-shift lineage."""
    if evidence_text is None and sha_in_evidence is not None:
        evidence_text = (
            f"[empirical:experiments/results/foo/] archive sha256={sha_in_evidence} "
            f"224481 B"
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
    return lane


# ── _check_227_lane_is_class_shift_claim ───────────────────────────────────


def test_class_shift_claim_via_lane_class_substrate_class_shift() -> None:
    lane = {"lane_class": "substrate_class_shift", "notes": ""}
    assert _check_227_lane_is_class_shift_claim(lane)


def test_class_shift_claim_via_lane_class_predictive_receiver() -> None:
    lane = {"lane_class": "predictive_receiver", "notes": ""}
    assert _check_227_lane_is_class_shift_claim(lane)


def test_class_shift_claim_via_lane_class_cooperative_receiver() -> None:
    lane = {"lane_class": "cooperative_receiver", "notes": ""}
    assert _check_227_lane_is_class_shift_claim(lane)


def test_class_shift_claim_via_lane_class_foveation() -> None:
    lane = {"lane_class": "foveation", "notes": ""}
    assert _check_227_lane_is_class_shift_claim(lane)


def test_class_shift_claim_via_notes_lane_class_assignment() -> None:
    lane = {
        "lane_class": "",
        "notes": "lane_class=substrate_class_shift",
    }
    assert _check_227_lane_is_class_shift_claim(lane)


def test_class_shift_claim_via_notes_literature_anchor_mdl_ibps() -> None:
    lane = {"lane_class": "", "notes": "Tier B MDL-IBPS substrate per Tishby-Zaslavsky 2015"}
    assert _check_227_lane_is_class_shift_claim(lane)


def test_class_shift_claim_via_notes_literature_anchor_wyner_ziv() -> None:
    lane = {"lane_class": "", "notes": "Wyner-Ziv frame_0 cooperative-receiver"}
    assert _check_227_lane_is_class_shift_claim(lane)


def test_class_shift_claim_via_hafner_dreamer_v3() -> None:
    """Per C1 council `RETAIN` decision, Hafner DreamerV3 is a canonical token."""
    lane = {"lane_class": "", "notes": "C1 world-model per Hafner DreamerV3 2023"}
    assert _check_227_lane_is_class_shift_claim(lane)


def test_class_shift_claim_via_ha_schmidhuber() -> None:
    lane = {"lane_class": "", "notes": "Ha-Schmidhuber 2018 world model"}
    assert _check_227_lane_is_class_shift_claim(lane)


def test_no_class_shift_claim_when_no_token() -> None:
    lane = {"lane_class": "hnerv_lc_v2", "notes": "PR101 sidecar"}
    assert not _check_227_lane_is_class_shift_claim(lane)


def test_no_class_shift_claim_when_empty_lane() -> None:
    assert not _check_227_lane_is_class_shift_claim({})


def test_class_shift_claim_handles_non_string_lane_class() -> None:
    lane = {"lane_class": None, "notes": ""}
    assert not _check_227_lane_is_class_shift_claim(lane)


# ── _check_227_lane_has_tier_c_evidence ─────────────────────────────────────


def test_tier_c_evidence_via_notes_token() -> None:
    lane = {
        "notes": "Tier C verdict: across_class (latent_sigma1=0.00213)",
        "gates": {},
    }
    assert _check_227_lane_has_tier_c_evidence(lane, {})


def test_tier_c_evidence_via_evidence_string_token() -> None:
    lane = {
        "notes": "",
        "gates": {
            "real_archive_empirical": {
                "status": True,
                "evidence": "[mdl_tier_c:experiments/results/c6_tier_c_20260514/]",
            },
        },
    }
    assert _check_227_lane_has_tier_c_evidence(lane, {})


def test_tier_c_evidence_via_sha_match_against_index(tmp_path: Path) -> None:
    """When the lane references an archive sha that has a Tier C ablation,
    the gate accepts that as evidence."""
    _write_mdl_result_with_tier_c(tmp_path, C6_SHA_FULL)
    tier_c_index = _check_227_discover_tier_c_results(tmp_path)
    lane = {
        "notes": "",
        "gates": {
            "real_archive_empirical": {
                "status": True,
                "evidence": (
                    f"archive sha256={C6_SHA_FULL} 224481 B"
                ),
            },
        },
    }
    assert _check_227_lane_has_tier_c_evidence(lane, tier_c_index)


def test_no_tier_c_evidence_when_sha_not_in_index() -> None:
    lane = {
        "notes": "",
        "gates": {
            "real_archive_empirical": {
                "status": True,
                "evidence": f"archive sha256={SAFE_SHA_FULL} 100 B",
            },
        },
    }
    assert not _check_227_lane_has_tier_c_evidence(lane, {})


def test_no_tier_c_evidence_when_notes_empty() -> None:
    lane = {"notes": "", "gates": {}}
    assert not _check_227_lane_has_tier_c_evidence(lane, {})


# ── _check_227_lane_has_waiver ─────────────────────────────────────────────


def test_waiver_accepted_with_rationale() -> None:
    lane = {
        "notes": "# TIER_C_EVIDENCE_PENDING_OK:c6_5ep_pre_converged_anchor",
        "gates": {},
    }
    waived, reason = _check_227_lane_has_waiver(lane)
    assert waived
    assert "c6_5ep_pre_converged_anchor" in reason


def test_waiver_inline_form_accepted() -> None:
    lane = {
        "notes": "TIER_C_EVIDENCE_PENDING_OK:pre_empirical_design_phase",
        "gates": {},
    }
    waived, reason = _check_227_lane_has_waiver(lane)
    assert waived
    assert "pre_empirical_design_phase" in reason


def test_waiver_placeholder_rejected() -> None:
    """The literal `<reason>` placeholder must NOT self-waive."""
    lane = {
        "notes": "# TIER_C_EVIDENCE_PENDING_OK:<reason>",
        "gates": {},
    }
    waived, _ = _check_227_lane_has_waiver(lane)
    assert not waived


def test_waiver_rationale_placeholder_rejected() -> None:
    lane = {
        "notes": "# TIER_C_EVIDENCE_PENDING_OK:<rationale>",
        "gates": {},
    }
    waived, _ = _check_227_lane_has_waiver(lane)
    assert not waived


def test_no_waiver_when_token_absent() -> None:
    lane = {
        "notes": "some other note",
        "gates": {},
    }
    waived, reason = _check_227_lane_has_waiver(lane)
    assert not waived
    assert reason == ""


def test_waiver_in_evidence_field_also_accepted() -> None:
    lane = {
        "notes": "",
        "gates": {
            "real_archive_empirical": {
                "status": True,
                "evidence": "# TIER_C_EVIDENCE_PENDING_OK:design_phase",
            },
        },
    }
    waived, reason = _check_227_lane_has_waiver(lane)
    assert waived
    assert "design_phase" in reason


# ── _check_227_discover_tier_c_results ─────────────────────────────────────


def test_discover_tier_c_results_empty_root(tmp_path: Path) -> None:
    idx = _check_227_discover_tier_c_results(tmp_path)
    assert idx == {}


def test_discover_tier_c_results_finds_archive(tmp_path: Path) -> None:
    _write_mdl_result_with_tier_c(tmp_path, C6_SHA_FULL)
    idx = _check_227_discover_tier_c_results(tmp_path)
    assert C6_SHA_FULL.lower() in idx
    assert idx[C6_SHA_FULL.lower()]["has_tier_c"]
    assert idx[C6_SHA_FULL.lower()]["n_tier_c_rows"] == 8


def test_discover_tier_c_skips_empty_tier_c(tmp_path: Path) -> None:
    """A JSON with tier_c=[] should be excluded from the index."""
    _write_mdl_result_with_tier_c(
        tmp_path, C6_SHA_FULL, tier_c_rows=[],
    )
    idx = _check_227_discover_tier_c_results(tmp_path)
    assert idx == {}


def test_discover_tier_c_extracts_verdict_and_density(tmp_path: Path) -> None:
    _write_mdl_result_with_tier_c(
        tmp_path,
        C6_SHA_FULL,
        tier_c_density=0.13,
        tier_c_verdict="across_class",
    )
    idx = _check_227_discover_tier_c_results(tmp_path)
    rec = idx[C6_SHA_FULL.lower()]
    assert rec["tier_c_density"] == pytest.approx(0.13)
    assert rec["tier_c_verdict"] == "across_class"


# ── End-to-end gate behavior ──────────────────────────────────────────────


def test_gate_passes_when_no_registry(tmp_path: Path) -> None:
    """Missing registry → 0 violations (gate is no-op)."""
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_gate_passes_when_no_class_shift_lanes(tmp_path: Path) -> None:
    """Lanes without class-shift claim are out of scope."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_within_class",
            lane_class="hnerv_lc_v2",  # not a class-shift token
            notes="PR101 sidecar bolt-on",
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_gate_passes_when_l1_class_shift_lane(tmp_path: Path) -> None:
    """L1 class-shift lanes are out of scope (only L2+ promotion gated)."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_l1",
            level=1,
            lane_class="substrate_class_shift",
            gate_status=False,  # L1 = not promoted
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_gate_flags_l2_class_shift_without_tier_c(tmp_path: Path) -> None:
    """L2 lane with substrate_class_shift but NO Tier C evidence → flagged."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_unproven",
            lane_class="substrate_class_shift",
            notes="MDL-IBPS substrate",  # class-shift claim
            sha_in_evidence=SAFE_SHA_FULL,  # no Tier C JSON for this sha
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert len(violations) == 1
    assert "lane_c6_unproven" in violations[0]
    assert "Tier C empirical evidence" in violations[0]


def test_gate_passes_when_tier_c_evidence_in_notes(tmp_path: Path) -> None:
    """L2 class-shift lane with Tier C token in notes → passes."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_with_tier_c",
            lane_class="substrate_class_shift",
            notes="Tier C verdict: across_class (latent_sigma1=0.00213)",
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_gate_passes_when_tier_c_evidence_via_sha_index(tmp_path: Path) -> None:
    """L2 class-shift lane whose archive sha is in the Tier C index → passes."""
    _write_mdl_result_with_tier_c(tmp_path, C6_SHA_FULL)
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_with_tier_c_index",
            lane_class="substrate_class_shift",
            sha_in_evidence=C6_SHA_FULL,
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_gate_passes_when_waiver_present(tmp_path: Path) -> None:
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_waived",
            lane_class="substrate_class_shift",
            notes="# TIER_C_EVIDENCE_PENDING_OK:5ep_pre_converged_anchor",
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_gate_skips_research_only_lanes(tmp_path: Path) -> None:
    """research_only lanes are exempt (out of contest-promotion scope)."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_research_only",
            lane_class="substrate_class_shift",
            notes="research_only=true",
            sha_in_evidence=SAFE_SHA_FULL,
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_gate_skips_substrate_engineering_lanes(tmp_path: Path) -> None:
    """substrate_engineering lanes are exempt (sister of #219)."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_substrate_eng",
            lane_class="substrate_engineering",  # NOT a class-shift token
            sha_in_evidence=SAFE_SHA_FULL,
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    # Not a class-shift claim, so out of scope.
    assert violations == []


def test_gate_flags_multiple_lanes(tmp_path: Path) -> None:
    """Two class-shift lanes both lacking Tier C → both flagged."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_a",
            lane_class="substrate_class_shift",
            notes="MDL-IBPS",
            sha_in_evidence=SAFE_SHA_FULL,
        ),
        _make_class_shift_l2_lane(
            "lane_d4_b",
            lane_class="cooperative_receiver",
            notes="Wyner-Ziv",
            sha_in_evidence=SAFE_SHA_FULL,
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert len(violations) == 2
    ids = "\n".join(violations)
    assert "lane_c6_a" in ids
    assert "lane_d4_b" in ids


def test_gate_strict_mode_raises_preflight_error(tmp_path: Path) -> None:
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_strict",
            lane_class="substrate_class_shift",
            notes="MDL-IBPS",
            sha_in_evidence=SAFE_SHA_FULL,
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    with pytest.raises(PreflightError) as exc:
        check_substrate_class_promotion_requires_tier_c_evidence(
            repo_root=tmp_path,
            strict=True,
        )
    assert "Catalog #227" in str(exc.value)
    assert "Tier C" in str(exc.value)


def test_gate_strict_mode_silent_on_clean_repo(tmp_path: Path) -> None:
    """Strict mode does NOT raise when 0 violations."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_clean",
            lane_class="substrate_class_shift",
            notes="Tier C verdict: across_class",
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    # Should not raise.
    result = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
        strict=True,
    )
    assert result == []


def test_gate_verbose_mode_emits_summary(tmp_path: Path, capsys) -> None:
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_clean",
            lane_class="substrate_class_shift",
            notes="Tier C verdict: across_class",
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "tier-c-class-shift-promotion-gate" in captured.out
    assert "OK" in captured.out


def test_gate_verbose_emits_violation_summary(tmp_path: Path, capsys) -> None:
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_dirty",
            lane_class="substrate_class_shift",
            notes="MDL-IBPS",
            sha_in_evidence=SAFE_SHA_FULL,
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "violation(s)" in captured.out


def test_gate_handles_unreadable_registry(tmp_path: Path) -> None:
    """Malformed JSON → no crash, 0 violations (silent skip)."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / ".omx" / "state" / "lane_registry.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_gate_passes_when_l2_gate_not_satisfied(tmp_path: Path) -> None:
    """L2 with the real_archive_empirical gate status=False → out of scope."""
    lanes = [
        _make_class_shift_l2_lane(
            "lane_c6_unfinished",
            level=2,
            lane_class="substrate_class_shift",
            notes="MDL-IBPS",
            sha_in_evidence=SAFE_SHA_FULL,
            gate_status=False,
        ),
    ]
    _write_lane_registry(tmp_path, lanes)
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        repo_root=tmp_path,
    )
    assert violations == []


def test_constants_include_class_shift_tokens() -> None:
    """Sanity check: the canonical class-shift tokens are pinned."""
    assert "substrate_class_shift" in _CHECK_227_CLASS_SHIFT_LANE_CLASS_TOKENS
    assert "predictive_receiver" in _CHECK_227_CLASS_SHIFT_LANE_CLASS_TOKENS
    assert "cooperative_receiver" in _CHECK_227_CLASS_SHIFT_LANE_CLASS_TOKENS
    assert "foveation" in _CHECK_227_CLASS_SHIFT_LANE_CLASS_TOKENS


def test_constants_include_canonical_literature_tokens() -> None:
    assert "MDL-IBPS" in _CHECK_227_CLASS_SHIFT_NOTES_TOKENS
    assert "Wyner-Ziv" in _CHECK_227_CLASS_SHIFT_NOTES_TOKENS
    assert "Tishby-Zaslavsky" in _CHECK_227_CLASS_SHIFT_NOTES_TOKENS
    # C1 council "RETAIN" decision tokens:
    assert "Hafner" in _CHECK_227_CLASS_SHIFT_NOTES_TOKENS
    assert "DreamerV3" in _CHECK_227_CLASS_SHIFT_NOTES_TOKENS
    assert "Ha-Schmidhuber" in _CHECK_227_CLASS_SHIFT_NOTES_TOKENS


def test_constants_include_tier_c_tokens() -> None:
    assert "tier_c" in _CHECK_227_TIER_C_TOKENS
    assert "Tier C" in _CHECK_227_TIER_C_TOKENS
    assert "mdl_tier_c" in _CHECK_227_TIER_C_TOKENS


# ── Live-repo regression guard ────────────────────────────────────────────


def test_live_repo_zero_violations() -> None:
    """The repo's lane registry should have 0 Catalog #227 violations.

    This is a regression guard: if a sister-subagent promotes a class-shift
    lane to L2 without Tier C evidence, this test catches the drift.
    Initial wire-in is warn-only, so the gate returns the list rather than
    raising — but the live count MUST stay at 0.
    """
    # Use the canonical repo root (not tmp_path).
    violations = check_substrate_class_promotion_requires_tier_c_evidence(
        strict=False,
    )
    assert violations == [], (
        f"Catalog #227 found {len(violations)} live violations in the repo; "
        f"each: {violations[:3]}"
    )


# ── Orchestrator wire-in regression guard ─────────────────────────────────


def test_check_227_wired_into_preflight_all() -> None:
    """Regression guard: Catalog #227 MUST be called from preflight_all().

    Per CLAUDE.md "Strict-flip atomicity rule" the initial wire-in is
    warn-only (strict=False); but the gate MUST be invoked or the
    self-protection is silently broken.
    """
    import inspect
    from tac import preflight as preflight_module

    src = inspect.getsource(preflight_module.preflight_all)
    assert (
        "check_substrate_class_promotion_requires_tier_c_evidence" in src
    ), (
        "Catalog #227 must be wired into preflight_all() but the call site "
        "was not found in the source. The CLAUDE.md 'Strict-flip atomicity' "
        "rule requires the gate to be invoked even at warn-only stage."
    )
