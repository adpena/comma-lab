# SPDX-License-Identifier: MIT
"""Tests for Catalog #323 check_no_score_claim_without_canonical_provenance.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
the gate is the META-class umbrella for phantom-score class instances.
Tests pin the gate's detection across all 5 sister-gate empirical anchors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_score_claim_without_canonical_provenance,
)
from tac.provenance import (
    ProvenanceEvidenceGrade,
    build_provenance_for_archive_member,
    build_provenance_for_research_sidecar,
    provenance_to_dict,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """Create a synthetic repo with .omx/state/ subdir for gate scanning."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / "experiments" / "results").mkdir(parents=True)
    return tmp_path


# -----------------------------------------------------------------------------
# Negative path: clean repo passes
# -----------------------------------------------------------------------------

def test_clean_repo_no_violations(synthetic_repo: Path):
    """Empty repo with no JSON artifacts passes cleanly."""
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert violations == []


def test_repo_with_non_score_artifacts_passes(synthetic_repo: Path):
    """JSON artifacts without score-claim keys are CLEAN."""
    (synthetic_repo / ".omx" / "state" / "config.json").write_text(
        json.dumps({"name": "test", "count": 42})
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert violations == []


# -----------------------------------------------------------------------------
# Positive path: phantom-score class anchors
# -----------------------------------------------------------------------------

def test_catalog_321_research_sidecar_phantom_caught(synthetic_repo: Path):
    """Catalog #321 anchor: deliverable_score_savings_estimate without Provenance."""
    (synthetic_repo / ".omx" / "state" / "pr101_state_dict.json").write_text(
        json.dumps({
            "lane_id": "pr101_state_dict",
            "deliverable_score_savings_estimate": 0.477,
        })
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1
    assert "deliverable_score_savings_estimate" in violations[0]


def test_catalog_319_composition_alpha_phantom_caught(synthetic_repo: Path):
    """Catalog #319 anchor: composition_alpha without DeliverabilityProof."""
    (synthetic_repo / ".omx" / "state" / "fec6_composition.json").write_text(
        json.dumps({
            "pair_key": "fec6_x_pr106",
            "composition_alpha": 1.15,
        })
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1


def test_catalog_823_alpha_savings_ratio_form_phantom_caught(synthetic_repo: Path):
    """Catalog #823 anchor: alpha_savings_ratio_form SIREN byte-identity."""
    (synthetic_repo / ".omx" / "state" / "siren_pair.json").write_text(
        json.dumps({
            "pair_key": "lane_g_v3_renderer__x__siren_renderer",
            "alpha_savings_ratio_form": 4.74,
        })
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1


def test_catalog_287_score_without_evidence_tag_caught(synthetic_repo: Path):
    """Catalog #287 anchor: score without evidence tag (umbrella catch)."""
    (synthetic_repo / ".omx" / "state" / "lane_result.json").write_text(
        json.dumps({"lane_id": "test", "score": 0.5})
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1


@pytest.mark.parametrize(
    "score_key",
    [
        "canonical_score",
        "canonical_score_recomputed",
        "score_recomputed",
        "score_recomputed_from_components",
        "score_recomputed_from_contest_components",
        "score_recomputed_from_public_components",
        "score_contest_cuda",
        "score_contest_cpu",
        "contest_cuda_score_recomputed",
        "contest_cpu_score_recomputed",
        "empirical_score",
        "diagnostic_cpu_score",
        "auth_eval_recomputed_score",
        "score_recomputed_from_auth_eval",
    ],
)
def test_common_auth_eval_score_synonyms_without_provenance_caught(
    synthetic_repo: Path,
    score_key: str,
):
    """Real auth-eval/harvest score spellings require canonical Provenance."""
    (synthetic_repo / ".omx" / "state" / f"{score_key}.json").write_text(
        json.dumps({"lane_id": "test", score_key: 0.5})
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False,
        repo_root=synthetic_repo,
    )
    assert len(violations) == 1
    assert score_key in violations[0]


# -----------------------------------------------------------------------------
# Negative path: artifacts with valid Provenance pass
# -----------------------------------------------------------------------------

def test_score_with_canonical_provenance_passes(synthetic_repo: Path):
    """Score-claim with valid Provenance is CLEAN."""
    prov = build_provenance_for_research_sidecar(
        sidecar_path=synthetic_repo / "experiments" / "results" / "fake.pt",
        reactivation_criteria="test",
    )
    (synthetic_repo / ".omx" / "state" / "with_prov.json").write_text(
        json.dumps({
            "score": 0.0,  # 0 is OK for research sidecar
            "provenance": provenance_to_dict(prov),
        })
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert violations == []


def test_non_zero_score_with_research_provenance_caught(synthetic_repo: Path):
    """Phantom-score class structural detection."""
    prov = build_provenance_for_research_sidecar(
        sidecar_path=synthetic_repo / "experiments" / "results" / "fake.pt",
        reactivation_criteria="test",
    )
    (synthetic_repo / ".omx" / "state" / "phantom.json").write_text(
        json.dumps({
            "score": 0.477,  # phantom — research sidecar can't claim score
            "provenance": provenance_to_dict(prov),
        })
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1


# -----------------------------------------------------------------------------
# Waiver mechanism
# -----------------------------------------------------------------------------

def test_file_level_waiver_with_rationale_accepted(synthetic_repo: Path):
    """File-level # PROVENANCE_CANONICAL_WAIVED:<rationale> accepts."""
    # Note: JSON doesn't support # comments, so the waiver mechanism is
    # for files with a non-JSON header (or as a hash-prefixed first line
    # in JSONL). For pure JSON, the row-level waiver via a value containing
    # the marker is the mechanism. We test the row-level mechanism.
    (synthetic_repo / ".omx" / "state" / "waived.json").write_text(
        json.dumps({
            "score": 0.5,
            "waiver_note": "# PROVENANCE_CANONICAL_WAIVED:legacy-pre-canonical-helper-landing",
        })
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert violations == []


def test_waiver_with_placeholder_rejected(synthetic_repo: Path):
    """# PROVENANCE_CANONICAL_WAIVED:<rationale> placeholder rejected."""
    (synthetic_repo / ".omx" / "state" / "placeholder_waiver.json").write_text(
        json.dumps({
            "score": 0.5,
            "waiver_note": "# PROVENANCE_CANONICAL_WAIVED:<rationale>",
        })
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1


def test_waiver_with_short_rationale_rejected(synthetic_repo: Path):
    """Short row-level waiver rationale is not enough to bypass Catalog #323."""
    (synthetic_repo / ".omx" / "state" / "short_waiver.json").write_text(
        json.dumps({
            "score": 0.5,
            "waiver_note": "# PROVENANCE_CANONICAL_WAIVED:ab",
        })
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1


def test_jsonl_per_line_evaluation(synthetic_repo: Path):
    """JSONL files are scanned row-by-row."""
    jsonl_path = synthetic_repo / ".omx" / "state" / "rows.jsonl"
    lines = [
        json.dumps({"name": "ok-no-score"}),  # CLEAN
        json.dumps({"score": 0.5}),  # VIOLATION (no provenance)
        json.dumps({"count": 1}),  # CLEAN
    ]
    jsonl_path.write_text("\n".join(lines))
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1
    assert "row 1" in violations[0]


# -----------------------------------------------------------------------------
# Strict-mode behavior
# -----------------------------------------------------------------------------

def test_strict_mode_raises_on_violation(synthetic_repo: Path):
    """Strict mode raises PreflightError with Catalog #323 message."""
    (synthetic_repo / ".omx" / "state" / "violator.json").write_text(
        json.dumps({"score": 0.5})
    )
    with pytest.raises(PreflightError) as exc_info:
        check_no_score_claim_without_canonical_provenance(
            strict=True, repo_root=synthetic_repo,
        )
    assert "Catalog #323" in str(exc_info.value)
    assert "tac.provenance" in str(exc_info.value)


def test_strict_mode_silent_on_clean_repo(synthetic_repo: Path):
    """Strict mode does not raise when clean."""
    check_no_score_claim_without_canonical_provenance(
        strict=True, repo_root=synthetic_repo,
    )


# -----------------------------------------------------------------------------
# Exempt markers
# -----------------------------------------------------------------------------

def test_excluded_intake_clones_skipped(synthetic_repo: Path):
    """Vendored intake clones (Catalog #109) are excluded."""
    intake_dir = synthetic_repo / "experiments" / "results" / "public_pr95_intake_codex"
    intake_dir.mkdir(parents=True)
    (intake_dir / "build_manifest.json").write_text(
        json.dumps({"score": 0.5})  # would be violation if scanned
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert violations == []


def test_excluded_archive_state_skipped(synthetic_repo: Path):
    """.omx/state/archive/ paths (Catalog #298) are excluded."""
    archive_dir = synthetic_repo / ".omx" / "state" / "archive"
    archive_dir.mkdir(parents=True)
    (archive_dir / "old_anchor.json").write_text(json.dumps({"score": 0.5}))
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert violations == []


def test_excluded_quarantine_skipped(synthetic_repo: Path):
    """Quarantined artifacts are excluded."""
    q_dir = synthetic_repo / ".omx" / "state" / "quarantine_phantom_pre_catalog_321"
    q_dir.mkdir(parents=True)
    (q_dir / "phantom.json").write_text(json.dumps({"score": 0.5}))
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert violations == []


# -----------------------------------------------------------------------------
# Multi-violation aggregation
# -----------------------------------------------------------------------------

def test_multi_violation_aggregated(synthetic_repo: Path):
    """Multiple violating files all reported."""
    for i in range(5):
        (synthetic_repo / ".omx" / "state" / f"violator_{i}.json").write_text(
            json.dumps({"score": float(i)})
        )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 5


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------

def test_invalid_provenance_dict_caught(synthetic_repo: Path):
    """provenance field is not a dict → flagged."""
    (synthetic_repo / ".omx" / "state" / "bad_prov.json").write_text(
        json.dumps({"score": 0.5, "provenance": "string_not_dict"})
    )
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert len(violations) == 1


def test_malformed_json_silent_skip(synthetic_repo: Path):
    """Malformed JSON doesn't crash the gate."""
    (synthetic_repo / ".omx" / "state" / "bad.json").write_text("{not valid json")
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    # Should not flag (file not parseable as score-claim)
    assert violations == []


def test_empty_file_silent_skip(synthetic_repo: Path):
    """Empty file silent skip."""
    (synthetic_repo / ".omx" / "state" / "empty.json").write_text("")
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=synthetic_repo,
    )
    assert violations == []


def test_string_repo_root_accepted(synthetic_repo: Path):
    """repo_root accepts str or Path."""
    (synthetic_repo / ".omx" / "state" / "x.json").write_text(json.dumps({"score": 0.5}))
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, repo_root=str(synthetic_repo),  # str instead of Path
    )
    assert len(violations) == 1


# -----------------------------------------------------------------------------
# Live-repo regression guard
# -----------------------------------------------------------------------------

def test_live_repo_violations_bounded():
    """Live-repo violations bounded so future regression spikes are detected."""
    violations = check_no_score_claim_without_canonical_provenance(
        strict=False, verbose=False,
    )
    # At landing: ~543 baseline violations across .omx/state + experiments/results
    # Bound at 5000 to catch dramatic regressions.
    assert len(violations) < 5000, (
        f"Catalog #323 violations={len(violations)} unexpectedly high"
    )


# -----------------------------------------------------------------------------
# Orchestrator wire-in regression guard
# -----------------------------------------------------------------------------

def test_orchestrator_wires_warn_only_at_landing():
    """preflight_all() wires Catalog #323 WARN-ONLY at landing."""
    import inspect
    from tac.preflight import preflight_all
    src = inspect.getsource(preflight_all)
    # Wire-in present
    assert "check_no_score_claim_without_canonical_provenance" in src
    # WARN-ONLY at landing
    # We grep for the strict=False pattern in proximity
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if "check_no_score_claim_without_canonical_provenance" in line and "(" in line:
            # Look at next 5 lines for strict=
            window = "\n".join(lines[i : i + 5])
            assert "strict=False" in window, (
                "Catalog #323 should be wired strict=False at landing"
            )
            return
    pytest.fail("Catalog #323 wire-in not found in preflight_all")


def test_gate_callable_via_globals_regression_guard():
    """Catalog #185 sister regression: gate must be importable via module globals."""
    import tac.preflight as preflight_module
    assert hasattr(preflight_module, "check_no_score_claim_without_canonical_provenance")
    assert callable(preflight_module.check_no_score_claim_without_canonical_provenance)
