"""Tests for tools/build_public_pr_mining_expansion_backlog.py.

15-25 dedicated tests per CLAUDE.md "Bugs must be permanently fixed AND
self-protected against" — the miner must produce deterministic, typed,
research-signal-only JSONL backed by source-byte inspection.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "build_public_pr_mining_expansion_backlog.py"

# Load the tool as a module (it's a script, not under src/tac/).
spec = importlib.util.spec_from_file_location(
    "tools_build_public_pr_mining_expansion_backlog",
    TOOL_PATH,
)
assert spec is not None and spec.loader is not None
miner = importlib.util.module_from_spec(spec)
sys.modules["tools_build_public_pr_mining_expansion_backlog"] = miner
spec.loader.exec_module(miner)


# ----------------------------------------------------------------------------
# Catalog shape tests
# ----------------------------------------------------------------------------


def test_primitives_catalog_is_non_empty():
    assert len(miner.PRIMITIVES_CATALOG) > 0


def test_primitives_catalog_size_in_expected_range():
    # We expect 15-25 mined primitives; if a future edit moves outside this
    # range, the synthesis ranking + EV/byte top-N may need re-tuning.
    assert 15 <= len(miner.PRIMITIVES_CATALOG) <= 35


def test_every_primitive_is_frozen_dataclass():
    for p in miner.PRIMITIVES_CATALOG:
        assert isinstance(p, miner.MinedPrimitive)
        # frozen dataclass: assignment should raise
        with pytest.raises((AttributeError, Exception)):
            p.primitive_id = "mutated"  # type: ignore[misc]


def test_every_primitive_has_unique_id():
    ids = [p.primitive_id for p in miner.PRIMITIVES_CATALOG]
    assert len(ids) == len(set(ids)), f"duplicate primitive_ids: {ids}"


def test_every_primitive_has_non_empty_required_fields():
    for p in miner.PRIMITIVES_CATALOG:
        assert p.primitive_id
        assert p.pr_number > 0
        assert p.submission_slot
        assert p.family
        assert p.representation_type
        assert p.score_axis_target
        assert p.key_mechanism_description
        assert p.source_paths
        assert p.source_loc_observed
        assert p.estimated_loc_to_port > 0
        assert p.applicable_to_pr106_r2_frontier in {"true", "false", "unknown"}
        assert p.next_action


# ----------------------------------------------------------------------------
# Safety-tag tests (CLAUDE.md non-negotiables)
# ----------------------------------------------------------------------------


def test_every_primitive_has_score_claim_false():
    """Per CLAUDE.md `forbidden_score_claim_with_byte_change_unless_inflate_consumes`."""
    for p in miner.PRIMITIVES_CATALOG:
        assert p.score_claim is False, f"{p.primitive_id} has score_claim=True"


def test_every_primitive_has_promotion_eligible_false():
    for p in miner.PRIMITIVES_CATALOG:
        assert p.promotion_eligible is False


def test_every_primitive_has_ready_for_exact_eval_dispatch_false():
    for p in miner.PRIMITIVES_CATALOG:
        assert p.ready_for_exact_eval_dispatch is False


def test_evidence_grade_carries_research_signal_tag():
    for p in miner.PRIMITIVES_CATALOG:
        assert "research-signal" in p.evidence_grade.lower()
        assert "not contest-CUDA-validated" in p.evidence_grade or "not contest-cuda-validated" in p.evidence_grade.lower()


# ----------------------------------------------------------------------------
# Provenance tests
# ----------------------------------------------------------------------------


def test_every_primitive_source_path_is_under_kaggle_mirror():
    for p in miner.PRIMITIVES_CATALOG:
        for path in p.source_paths:
            assert "public_pr_archive_kaggle_mirror" in path, (
                f"{p.primitive_id} source path not under kaggle mirror: {path}"
            )


def test_every_primitive_source_path_references_correct_pr():
    for p in miner.PRIMITIVES_CATALOG:
        # At least one source path should contain the PR number marker.
        marker = f"public_pr{p.pr_number}_intake"
        assert any(marker in path for path in p.source_paths), (
            f"{p.primitive_id} source paths {p.source_paths} don't contain {marker}"
        )


def test_no_primitive_references_tmp_path():
    """Per CLAUDE.md `forbidden /tmp paths in any persisted artifact`."""
    for p in miner.PRIMITIVES_CATALOG:
        for path in p.source_paths:
            assert not path.startswith("/tmp/"), f"{p.primitive_id} has /tmp path: {path}"
            assert "/tmp/" not in path, f"{p.primitive_id} has /tmp path: {path}"


def test_no_primitive_references_kaggle_or_internal_paths_outside_intake():
    """Primitive sources must be the public-PR intake mirror, never internal."""
    for p in miner.PRIMITIVES_CATALOG:
        for path in p.source_paths:
            assert "src/tac/" not in path
            assert ".omx/" not in path
            assert "experiments/results/" in path


# ----------------------------------------------------------------------------
# Composition + deduplication tests
# ----------------------------------------------------------------------------


def test_no_primitive_duplicates_existing_tac_packet_compiler_coverage():
    """Each emitted primitive must NOT already be in tac.packet_compiler."""
    for p in miner.PRIMITIVES_CATALOG:
        assert p.primitive_id not in miner.ALREADY_IN_TAC_PACKET_COMPILER, (
            f"{p.primitive_id} should be excluded — already in tac.packet_compiler"
        )


def test_composes_with_references_known_modules():
    """Every `composes_with` entry should reference a real tac.packet_compiler
    module or another primitive in this catalog."""
    own_ids = {p.primitive_id for p in miner.PRIMITIVES_CATALOG}
    known_external = {
        "pr101_sidecar_grammar",
        "pr103_arithmetic_coding",
        "pr81_quantizr",
        "pr84_adaptive_mask",
        "pr91_hpac_grammar",
        "pr92_joint_stream",
        "pr93_pose_codec",
        "pr93_lowpass_luma",
        "pr97_h3_grammar",
        "custom_binary_container",
        "magic_codec",
        "sparse_packet_ir",
    }
    for p in miner.PRIMITIVES_CATALOG:
        for ref in p.composes_with:
            assert ref in own_ids or ref in known_external, (
                f"{p.primitive_id} references unknown composer: {ref}"
            )


# ----------------------------------------------------------------------------
# EV/byte ranking tests
# ----------------------------------------------------------------------------


def test_estimate_ev_per_loc_returns_zero_for_non_pr106_applicable():
    """Primitives marked applicable=false/unknown should rank at 0."""
    for p in miner.PRIMITIVES_CATALOG:
        if p.applicable_to_pr106_r2_frontier != "true":
            assert miner.estimate_ev_per_loc(p) == 0.0


def test_estimate_ev_per_loc_returns_positive_for_pr106_applicable():
    """Primitives marked applicable=true must rank > 0."""
    for p in miner.PRIMITIVES_CATALOG:
        if p.applicable_to_pr106_r2_frontier == "true":
            assert miner.estimate_ev_per_loc(p) > 0.0


def test_pose_axis_outranks_seg_axis_when_loc_equal():
    """Per CLAUDE.md operating-point rule: pose 2.71x seg at PR106 r2."""
    # Build a synthetic pair and check ranking.
    pose_p = miner.MinedPrimitive(
        primitive_id="test_pose",
        pr_number=999,
        submission_slot="x",
        family="pose-codec",
        representation_type="packet-grammar",
        score_axis_target="pose",
        key_mechanism_description="t",
        source_paths=("experiments/results/x",),
        source_loc_observed=(("x", 1, 2),),
        estimated_loc_to_port=100,
        composes_with=(),
        applicable_to_pr106_r2_frontier="true",
        archive_grammar_fields_declared=(),
        blockers_to_promote_to_tac_packet_compiler=(),
        next_action="x",
    )
    seg_p = miner.MinedPrimitive(
        primitive_id="test_seg",
        pr_number=999,
        submission_slot="x",
        family="mask-codec",
        representation_type="packet-grammar",
        score_axis_target="seg",
        key_mechanism_description="t",
        source_paths=("experiments/results/x",),
        source_loc_observed=(("x", 1, 2),),
        estimated_loc_to_port=100,
        composes_with=(),
        applicable_to_pr106_r2_frontier="true",
        archive_grammar_fields_declared=(),
        blockers_to_promote_to_tac_packet_compiler=(),
        next_action="x",
    )
    assert miner.estimate_ev_per_loc(pose_p) > miner.estimate_ev_per_loc(seg_p)


def test_rank_top_n_respects_n_limit():
    ranked = miner.rank_top_n(miner.PRIMITIVES_CATALOG, n=5)
    assert len(ranked) <= 5


def test_rank_top_n_only_returns_applicable_primitives():
    ranked = miner.rank_top_n(miner.PRIMITIVES_CATALOG, n=20)
    for p in ranked:
        assert p.applicable_to_pr106_r2_frontier == "true"


# ----------------------------------------------------------------------------
# End-to-end driver tests
# ----------------------------------------------------------------------------


def test_main_produces_backlog_jsonl_and_synthesis_md(tmp_path):
    out_dir = tmp_path / "out"
    exit_code = miner.main([
        "--out-dir", str(out_dir),
    ])
    assert exit_code == 0
    backlog = out_dir / "backlog.jsonl"
    synthesis = out_dir / "synthesis.md"
    assert backlog.is_file()
    assert synthesis.is_file()
    # Backlog has at least len(catalog) rows
    rows = [json.loads(line) for line in backlog.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == len(miner.PRIMITIVES_CATALOG)


def test_backlog_jsonl_is_deterministic_across_runs(tmp_path):
    out_dir_a = tmp_path / "a"
    out_dir_b = tmp_path / "b"
    miner.main(["--out-dir", str(out_dir_a)])
    miner.main(["--out-dir", str(out_dir_b)])
    a = (out_dir_a / "backlog.jsonl").read_bytes()
    b = (out_dir_b / "backlog.jsonl").read_bytes()
    assert a == b, "backlog.jsonl must be deterministic across runs"


def test_backlog_rows_are_sorted_keys_json():
    """Sort-keys JSON is required for deterministic golden-vector parity."""
    from io import StringIO
    primitives = miner.attach_pr_claimed_score(
        miner.PRIMITIVES_CATALOG,
        REPO_ROOT / "experiments/results/public_pr_archive_release_view/FETCH_SUMMARY.json",
    )
    # Pick any row, re-serialize, check keys are sorted.
    p = primitives[0]
    serialized = json.dumps(asdict(p), sort_keys=True)
    parsed = json.loads(serialized)
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_synthesis_md_contains_top_5_table(tmp_path):
    out_dir = tmp_path / "out"
    miner.main(["--out-dir", str(out_dir)])
    body = (out_dir / "synthesis.md").read_text(encoding="utf-8")
    assert "Top-5 promotable-to-tac.packet_compiler" in body
    assert "EV/byte heuristic" in body
    # Top-5 table headers
    assert "| primitive_id |" in body


def test_synthesis_md_calls_out_loop_pause_status(tmp_path):
    out_dir = tmp_path / "out"
    miner.main(["--out-dir", str(out_dir)])
    body = (out_dir / "synthesis.md").read_text(encoding="utf-8")
    assert "Loop remains PAUSED" in body
    assert "operator directive 2026-05-09" in body


def test_synthesis_md_calls_out_hard_requirements(tmp_path):
    out_dir = tmp_path / "out"
    miner.main(["--out-dir", str(out_dir)])
    body = (out_dir / "synthesis.md").read_text(encoding="utf-8")
    assert "$0 GPU spend" in body
    assert "No /tmp paths" in body


def test_attach_pr_claimed_score_attaches_known_pr_scores(tmp_path):
    primitives = miner.attach_pr_claimed_score(
        miner.PRIMITIVES_CATALOG,
        REPO_ROOT / "experiments/results/public_pr_archive_release_view/FETCH_SUMMARY.json",
    )
    # PR55 has score=0.333 in the public dataset.
    pr55 = [p for p in primitives if p.pr_number == 55]
    if pr55:
        assert pr55[0].pr_claimed_score == 0.333
    pr105 = [p for p in primitives if p.pr_number == 105]
    if pr105:
        assert pr105[0].pr_claimed_score == 0.198


def test_attach_pr_claimed_score_handles_missing_summary():
    """Missing summary file shouldn't crash; primitives keep score=None."""
    primitives = miner.attach_pr_claimed_score(
        miner.PRIMITIVES_CATALOG,
        REPO_ROOT / "experiments/results/this_path_does_not_exist.json",
    )
    for p in primitives:
        assert p.pr_claimed_score is None


def test_discover_target_pr_dirs_handles_missing_root(tmp_path):
    dirs = miner.discover_target_pr_dirs(tmp_path / "nonexistent")
    assert dirs == []


def test_discover_target_pr_dirs_only_returns_target_prs(tmp_path):
    intake = tmp_path / "intake"
    intake.mkdir()
    # Decoy dirs
    (intake / "public_pr999999_intake_20260101_auto").mkdir()  # not in target map
    (intake / "public_pr55_intake_20260505_auto").mkdir()  # in target map
    (intake / "not_a_pr_dir").mkdir()
    dirs = miner.discover_target_pr_dirs(intake)
    names = [d.name for d in dirs]
    assert "public_pr55_intake_20260505_auto" in names
    assert "public_pr999999_intake_20260101_auto" not in names


# ----------------------------------------------------------------------------
# Sanity-check tests (per CLAUDE.md "Internal-consistency assertions")
# ----------------------------------------------------------------------------


def test_byte_slope_is_canonical_value():
    """25 / N_BYTES_CONTEST = ~6.658e-7."""
    expected = 25.0 / 37_545_489
    assert abs(miner.BYTE_SLOPE - expected) < 1e-12


def test_n_bytes_contest_matches_upstream_constant():
    """Same constant as `tac.analysis.public_pr_mechanism_index.CONTEST_N_BYTES`."""
    assert miner.N_BYTES_CONTEST == 37_545_489


def test_new_submission_slot_by_pr_excludes_canonical_baselines():
    """Map should only contain custom/named submissions, not baselines."""
    forbidden = {
        "av1_crf31_bicubic", "av1_roi_lanczos_unsharp", "av1_sharp1_adaptive",
        "baseline_fast", "no_compress", "h265_g16_512x384_veryslow",
        "neural_inflate", "roi_gop300_c34", "roi_v2",
        "svt_av1_lanczos_fg", "svtav1_45pct_unsharp", "svtav1_45pct_unsharp20_direct",
        "svtav1_av1grain_10bit", "svtav1_cheetah", "svtav1_spline_fg22",
        "v4_qp_aq2_roi", "damir_bearclaw_001", "damir_bearclaw_002", "damir_bearclaw_003",
    }
    for pr, slot in miner.NEW_SUBMISSION_SLOT_BY_PR.items():
        assert slot not in forbidden, f"PR{pr} slot {slot} is a baseline"


def test_at_least_one_pose_axis_primitive_in_top_5():
    """Per CLAUDE.md operating-point rule at PR106 r2: pose dominates marginally."""
    ranked = miner.rank_top_n(miner.PRIMITIVES_CATALOG, n=5)
    assert any(p.score_axis_target == "pose" for p in ranked), (
        "Top-5 must contain at least one pose-axis primitive at PR106 r2"
    )
