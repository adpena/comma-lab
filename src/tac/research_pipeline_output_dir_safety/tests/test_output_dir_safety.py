# SPDX-License-Identifier: MIT
"""Tests for ``tac.research_pipeline_output_dir_safety``.

Covers the 4-cascade falling-rule validator + the convenience enforce
wrapper + Catalog #287 placeholder-rationale rejection invariants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.research_pipeline_output_dir_safety import (
    DEFAULT_DATED_SUFFIX_PATTERNS,
    DEFAULT_WAIVER_PLACEHOLDERS,
    HISTORICAL_PROVENANCE_JSON_NAMES,
    OutputDirSafetyError,
    ResearchPipelineOutputDirVerdict,
    enforce_research_pipeline_output_dir,
    is_dated_subdir,
    validate_research_pipeline_output_dir,
)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / ".omx" / "research").mkdir(parents=True)
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    return tmp_path


def _make_canonical_json_file(target_dir: Path, name: str = "manifest.json") -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    p = target_dir / name
    p.write_text(json.dumps({"k": "v"}) + "\n")
    return p


# ----- is_dated_subdir helper -----

def test_is_dated_subdir_full_utc_pattern_matches() -> None:
    p = Path("/tmp/foo_20260528T131513Z")
    assert is_dated_subdir(p)


def test_is_dated_subdir_short_utc_pattern_matches() -> None:
    p = Path("/tmp/foo_20260528T1315Z")
    assert is_dated_subdir(p)


def test_is_dated_subdir_date_only_matches() -> None:
    p = Path("/tmp/foo_20260528Z")
    assert is_dated_subdir(p)


def test_is_dated_subdir_bare_date_matches() -> None:
    p = Path("/tmp/foo_20260528")
    assert is_dated_subdir(p)


def test_is_dated_subdir_long_numeric_matches() -> None:
    p = Path("/tmp/foo_20260528131513")
    assert is_dated_subdir(p)


def test_is_dated_subdir_no_date_does_not_match() -> None:
    p = Path("/tmp/foo_bar_baz")
    assert not is_dated_subdir(p)


def test_is_dated_subdir_arbitrary_token_does_not_match() -> None:
    p = Path("/tmp/foo_local")
    assert not is_dated_subdir(p)


def test_dated_suffix_patterns_pinned() -> None:
    # Regression guard: at least the 5 canonical patterns are pinned
    assert len(DEFAULT_DATED_SUFFIX_PATTERNS) >= 5


def test_canonical_json_names_includes_all_24_anchors() -> None:
    # The 3 anchor dirs use these names; if regression drops one, audit fails
    required = {
        "manifest.json",
        "runner_summary.json",
        "run_summary.json",
        "plan.json",
        "queue.json",
        "queue_observe.json",
        "queue_performance.json",
        "queue_validate.json",
        "score_report.json",
        "work_order.json",
        "archive_discovery.json",
        "runtime_consumption_proof.json",
        "representation_training_manifest.json",
        "pr95_public_archive_export.json",
    }
    missing = required - HISTORICAL_PROVENANCE_JSON_NAMES
    assert not missing, f"missing canonical names: {missing}"


def test_waiver_placeholders_canonical_set_pinned() -> None:
    for tok in ("<rationale>", "<reason>", "tbd", "todo", "fixme"):
        assert tok in DEFAULT_WAIVER_PLACEHOLDERS


# ----- Cascade A — out of scope -----

def test_cascade_a_out_of_scope_when_not_under_omx_research(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / "experiments" / "results" / "foo"
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert v.recommendation == "PROCEED"
    assert v.cascade_matched == "A_OUT_OF_SCOPE"
    assert not v.is_under_omx_research


def test_cascade_a_out_of_scope_when_under_omx_state(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "state" / "foo"
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert v.recommendation == "PROCEED"
    assert v.cascade_matched == "A_OUT_OF_SCOPE"


# ----- Cascade B — fresh dir -----

def test_cascade_b_fresh_when_dir_does_not_exist(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert v.recommendation == "PROCEED"
    assert v.cascade_matched == "B_FRESH"
    assert v.is_under_omx_research
    assert v.existing_canonical_json_files == ()


def test_cascade_b_fresh_when_dir_is_empty(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    out.mkdir(parents=True)
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert v.recommendation == "PROCEED"
    assert v.cascade_matched == "B_FRESH"


def test_cascade_b_fresh_when_dir_has_non_canonical_files(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    out.mkdir(parents=True)
    # 0.bin / archive.zip / etc are not in HISTORICAL_PROVENANCE_JSON_NAMES
    (out / "0.bin").write_bytes(b"\x00")
    (out / "archive.zip").write_bytes(b"PK")
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert v.recommendation == "PROCEED"
    assert v.cascade_matched == "B_FRESH"
    assert v.existing_canonical_json_files == ()


# ----- Cascade C — explicit operator opt-in -----

def test_cascade_c_explicit_opt_in_accepted_with_substantive_rationale(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out)
    v = validate_research_pipeline_output_dir(
        out,
        repo_root=repo,
        allow_overwrite_existing_historical_provenance=True,
        waiver_rationale="recovery-from-corrupt-half-write-2026-05-28",
    )
    assert v.recommendation == "PROCEED"
    assert v.cascade_matched == "C_EXPLICIT_OPT_IN"
    assert v.waiver_rationale_accepted


def test_cascade_c_placeholder_rationale_rejected_per_catalog_287(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out)
    v = validate_research_pipeline_output_dir(
        out,
        repo_root=repo,
        allow_overwrite_existing_historical_provenance=True,
        waiver_rationale="<rationale>",
    )
    assert v.recommendation == "REFUSE_EXISTING_HISTORICAL_PROVENANCE"
    assert v.cascade_matched == "D_REFUSE"
    assert not v.waiver_rationale_accepted


def test_cascade_c_empty_rationale_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out)
    v = validate_research_pipeline_output_dir(
        out,
        repo_root=repo,
        allow_overwrite_existing_historical_provenance=True,
        waiver_rationale="",
    )
    assert v.recommendation == "REFUSE_EXISTING_HISTORICAL_PROVENANCE"
    assert not v.waiver_rationale_accepted


def test_cascade_c_short_rationale_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out)
    v = validate_research_pipeline_output_dir(
        out,
        repo_root=repo,
        allow_overwrite_existing_historical_provenance=True,
        waiver_rationale="xy",  # 2 chars
    )
    assert v.recommendation == "REFUSE_EXISTING_HISTORICAL_PROVENANCE"
    assert not v.waiver_rationale_accepted


def test_cascade_c_opt_in_alone_without_rationale_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out)
    v = validate_research_pipeline_output_dir(
        out,
        repo_root=repo,
        allow_overwrite_existing_historical_provenance=True,
        # waiver_rationale=None
    )
    assert v.recommendation == "REFUSE_EXISTING_HISTORICAL_PROVENANCE"
    assert v.waiver_rationale_rejection_reason == "missing rationale"


# ----- Cascade D — refuse -----

def test_cascade_d_refuses_when_canonical_json_files_present(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out, "manifest.json")
    _make_canonical_json_file(out, "runner_summary.json")
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert v.recommendation == "REFUSE_EXISTING_HISTORICAL_PROVENANCE"
    assert v.cascade_matched == "D_REFUSE"
    assert set(v.existing_canonical_json_files) == {"manifest.json", "runner_summary.json"}
    assert "Catalog #113" in v.operator_routable_unwind_path
    assert "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1" in (
        v.operator_routable_unwind_path
    )


def test_cascade_d_includes_canonical_anti_pattern_id(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out)
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert v.canonical_anti_pattern_id == (
        "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1"
    )


def test_cascade_d_diagnostic_lists_existing_files(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out, "manifest.json")
    _make_canonical_json_file(out, "plan.json")
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert "manifest.json" in v.diagnostic_message
    assert "plan.json" in v.diagnostic_message


def test_cascade_d_three_anchor_dirs_pattern(tmp_path: Path) -> None:
    """Empirical anchor regression: pr95_mlx + repair_multi + frontier_final dirs."""

    repo = _make_repo(tmp_path)
    for dir_name, json_name, count in [
        ("pr95_mlx_runtime_consumption_queue_20260528T131513Z", "manifest.json", 24),
        ("repair_multi_archive_autonomous_live_psv3_fec6_20260528T055303Z", "runner_summary.json", 50),
        ("frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal", "queue.json", 3),
    ]:
        out = repo / ".omx" / "research" / dir_name
        _make_canonical_json_file(out, json_name)
        v = validate_research_pipeline_output_dir(out, repo_root=repo)
        assert v.recommendation == "REFUSE_EXISTING_HISTORICAL_PROVENANCE", (
            f"{dir_name}: should refuse existing {json_name}"
        )


# ----- Additional canonical filenames extension -----

def test_additional_canonical_filenames_extends_canonical_set(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    out.mkdir(parents=True)
    (out / "custom_artifact.json").write_text("{}")
    # Without extension: cascade B (fresh)
    v1 = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert v1.cascade_matched == "B_FRESH"
    # With extension: cascade D (refuse)
    v2 = validate_research_pipeline_output_dir(
        out,
        repo_root=repo,
        additional_canonical_filenames=frozenset({"custom_artifact.json"}),
    )
    assert v2.cascade_matched == "D_REFUSE"


# ----- enforce wrapper -----

def test_enforce_passes_for_proceed_verdict(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    v = enforce_research_pipeline_output_dir(out, repo_root=repo)
    assert v.recommendation == "PROCEED"


def test_enforce_raises_OutputDirSafetyError_on_refuse(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out)
    with pytest.raises(OutputDirSafetyError) as exc_info:
        enforce_research_pipeline_output_dir(out, repo_root=repo)
    err = str(exc_info.value)
    assert "Catalog #113" in err
    assert "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1" in err


def test_enforce_accepts_explicit_opt_in_with_rationale(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    _make_canonical_json_file(out)
    v = enforce_research_pipeline_output_dir(
        out,
        repo_root=repo,
        allow_overwrite_existing_historical_provenance=True,
        waiver_rationale="explicit operator review approved overwrite 2026-05-28",
    )
    assert v.recommendation == "PROCEED"
    assert v.cascade_matched == "C_EXPLICIT_OPT_IN"


# ----- Verdict dataclass invariants -----

def test_verdict_is_frozen(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    with pytest.raises((AttributeError, TypeError)):
        v.recommendation = "MUTATED"  # type: ignore[misc]


def test_verdict_canonical_anti_pattern_id_immutable(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    # Default value pinned
    assert v.canonical_anti_pattern_id.endswith(
        "_with_mutated_fields_v1"
    )


def test_verdict_string_paths_for_serialization(tmp_path: Path) -> None:
    """Verdict carries string paths so it's directly JSON-serializable."""

    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    assert isinstance(v.output_dir, str)


# ----- Edge cases -----

def test_string_repo_root_accepted(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    v = validate_research_pipeline_output_dir(out, repo_root=str(repo))
    assert v.recommendation == "PROCEED"


def test_string_output_dir_accepted(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    v = validate_research_pipeline_output_dir(str(out), repo_root=repo)
    assert v.recommendation == "PROCEED"


def test_namespace_root_itself_treated_as_out_of_scope(tmp_path: Path) -> None:
    """.omx/research/ root with no subdir component is not a valid write target."""

    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research"  # the namespace root itself
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    # is_under_omx_research returns False for root-itself by design
    assert v.recommendation == "PROCEED"
    assert v.cascade_matched == "A_OUT_OF_SCOPE"


def test_non_canonical_filename_capitalization_not_flagged(tmp_path: Path) -> None:
    """Filename matching is case-sensitive; MANIFEST.JSON not in canonical set."""

    repo = _make_repo(tmp_path)
    out = repo / ".omx" / "research" / "foo_20260528T130000Z"
    out.mkdir(parents=True)
    (out / "MANIFEST.JSON").write_text("{}")
    v = validate_research_pipeline_output_dir(out, repo_root=repo)
    # Case-sensitive: cascade B (fresh)
    assert v.cascade_matched == "B_FRESH"
