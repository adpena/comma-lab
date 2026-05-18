"""Tests for Catalog #325 — per-substrate optimal form symposium anchor gate.

Per CLAUDE.md NON-NEGOTIABLE "PER-SUBSTRATE OPTIMAL FORM via adversarial grand
council symposium" 2026-05-18. Covers helper unit tests, end-to-end gate
behavior, waiver semantics, strict-mode raising, and live-repo regression
guards.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from tac.preflight import (
    PreflightError,
    check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor,
    _check_325_extract_substrate_id_from_recipe,
    _check_325_parse_recipe_dispatch_enabled,
    _check_325_find_recent_symposium_memo,
    _CHECK_325_ACCEPTED_VERDICTS,
    _CHECK_325_SYMPOSIUM_WINDOW_DAYS,
    _CHECK_325_WAIVER_TOKEN,
    _CHECK_325_PLACEHOLDER_RATIONALES,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_helper_extract_substrate_id_canonical_modal_t4(tmp_path):
    recipe = tmp_path / "substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    assert _check_325_extract_substrate_id_from_recipe(recipe) == "c6_e4_mdl_ibps"


def test_helper_extract_substrate_id_modal_a100(tmp_path):
    recipe = tmp_path / "substrate_wyner_ziv_cooperative_receiver_modal_a100_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    assert _check_325_extract_substrate_id_from_recipe(recipe) == "wyner_ziv_cooperative_receiver"


def test_helper_extract_substrate_id_modal_smoke(tmp_path):
    recipe = tmp_path / "substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    assert _check_325_extract_substrate_id_from_recipe(recipe) == "z6_v2_candidate_1_multi_layer_film"


def test_helper_extract_substrate_id_non_canonical_returns_rest(tmp_path):
    recipe = tmp_path / "substrate_foo_bar_unknown_suffix.yaml"
    recipe.write_text("")
    # Falls back to full rest if no canonical suffix matches
    assert _check_325_extract_substrate_id_from_recipe(recipe) == "foo_bar_unknown_suffix"


def test_helper_extract_substrate_id_no_substrate_prefix(tmp_path):
    recipe = tmp_path / "not_a_substrate.yaml"
    recipe.write_text("")
    assert _check_325_extract_substrate_id_from_recipe(recipe) is None


def test_helper_parse_dispatch_enabled_true(tmp_path):
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\nresearch_only: false\n")
    enabled, ro, waiver = _check_325_parse_recipe_dispatch_enabled(recipe)
    assert enabled is True
    assert ro is False
    assert waiver is None


def test_helper_parse_dispatch_enabled_false(tmp_path):
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: false\n")
    enabled, ro, _ = _check_325_parse_recipe_dispatch_enabled(recipe)
    assert enabled is False


def test_helper_parse_research_only_true(tmp_path):
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\nresearch_only: true\n")
    enabled, ro, _ = _check_325_parse_recipe_dispatch_enabled(recipe)
    assert enabled is True
    assert ro is True


def test_helper_parse_waiver_accepted_with_rationale(tmp_path):
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(
        "dispatch_enabled: true  # PER_SUBSTRATE_SYMPOSIUM_WAIVED:emergency frontier override per operator session directive\n"
    )
    _, _, waiver = _check_325_parse_recipe_dispatch_enabled(recipe)
    assert waiver is not None
    assert "emergency frontier override" in waiver


def test_helper_parse_waiver_placeholder_rationale_rejected(tmp_path):
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(
        "dispatch_enabled: true  # PER_SUBSTRATE_SYMPOSIUM_WAIVED:<rationale>\n"
    )
    _, _, waiver = _check_325_parse_recipe_dispatch_enabled(recipe)
    assert waiver is None  # placeholder rejected


def test_helper_parse_waiver_reason_placeholder_rejected(tmp_path):
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(
        "dispatch_enabled: true  # PER_SUBSTRATE_SYMPOSIUM_WAIVED:<reason>\n"
    )
    _, _, waiver = _check_325_parse_recipe_dispatch_enabled(recipe)
    assert waiver is None


def test_helper_parse_waiver_short_rationale_rejected(tmp_path):
    recipe = tmp_path / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(
        "dispatch_enabled: true  # PER_SUBSTRATE_SYMPOSIUM_WAIVED:abc\n"
    )
    _, _, waiver = _check_325_parse_recipe_dispatch_enabled(recipe)
    assert waiver is None  # <4 chars


def test_helper_find_symposium_memo_within_window(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    research_dir.mkdir(parents=True)
    memo = research_dir / "council_per_substrate_symposium_c6_e4_mdl_ibps_20260518.md"
    memo.write_text("council_verdict: PROCEED\n")
    found, verdict = _check_325_find_recent_symposium_memo(
        "c6_e4_mdl_ibps", tmp_path, now_utc="2026-05-18T12:00:00Z"
    )
    assert found is True
    assert verdict == "PROCEED"


def test_helper_find_symposium_memo_outside_window(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    research_dir.mkdir(parents=True)
    memo = research_dir / "council_old_c6_e4_mdl_ibps_20260101.md"
    memo.write_text("council_verdict: PROCEED\n")
    found, _ = _check_325_find_recent_symposium_memo(
        "c6_e4_mdl_ibps", tmp_path, now_utc="2026-05-18T12:00:00Z"
    )
    assert found is False  # older than 14 days


def test_helper_find_symposium_memo_no_research_dir(tmp_path):
    found, verdict = _check_325_find_recent_symposium_memo(
        "foo", tmp_path, now_utc="2026-05-18T12:00:00Z"
    )
    assert found is False
    assert verdict is None


def test_helper_find_symposium_memo_latest_wins(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    research_dir.mkdir(parents=True)
    older = research_dir / "council_one_c6_20260513.md"
    older.write_text("council_verdict: DEFER_PENDING_EVIDENCE\n")
    newer = research_dir / "council_two_c6_20260518.md"
    newer.write_text("council_verdict: PROCEED\n")
    found, verdict = _check_325_find_recent_symposium_memo(
        "c6", tmp_path, now_utc="2026-05-18T12:00:00Z"
    )
    assert found is True
    assert verdict == "PROCEED"  # latest wins


def test_helper_find_symposium_memo_no_verdict_in_frontmatter(tmp_path):
    research_dir = tmp_path / ".omx" / "research"
    research_dir.mkdir(parents=True)
    memo = research_dir / "council_x_foo_20260518.md"
    memo.write_text("just markdown no frontmatter\n")
    found, verdict = _check_325_find_recent_symposium_memo(
        "foo", tmp_path, now_utc="2026-05-18T12:00:00Z"
    )
    assert found is True
    assert verdict is None


# ---------------------------------------------------------------------------
# End-to-end gate behavior
# ---------------------------------------------------------------------------


def _setup_repo(tmp_path: Path) -> Path:
    """Create a tmp repo with .omx/operator_authorize_recipes and .omx/research."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / ".omx" / "research").mkdir(parents=True)
    return tmp_path


def test_gate_clean_repo_no_violations(tmp_path):
    _setup_repo(tmp_path)
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=tmp_path
    )
    assert violations == []


def test_gate_recipe_dispatch_enabled_no_symposium_flagged(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert len(violations) == 1
    assert "c6_e4_mdl_ibps" in violations[0]
    assert "NO per-substrate symposium memo" in violations[0]


def test_gate_dispatch_disabled_out_of_scope(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: false\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert violations == []


def test_gate_research_only_opt_out(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\nresearch_only: true\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert violations == []


def test_gate_symposium_with_proceed_verdict_passes(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    memo = repo / ".omx" / "research" / "council_per_substrate_symposium_c6_e4_mdl_ibps_20260518.md"
    memo.write_text("council_verdict: PROCEED\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert violations == []


def test_gate_symposium_with_proceed_with_revisions_passes(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_c6_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    memo = repo / ".omx" / "research" / "council_per_substrate_c6_20260518.md"
    memo.write_text("council_verdict: PROCEED_WITH_REVISIONS\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert violations == []


def test_gate_symposium_with_defer_verdict_flagged(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_c6_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    memo = repo / ".omx" / "research" / "council_per_substrate_c6_20260518.md"
    memo.write_text("council_verdict: DEFER_PENDING_EVIDENCE\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert len(violations) == 1
    assert "verdict='DEFER_PENDING_EVIDENCE'" in violations[0]


def test_gate_symposium_with_refuse_verdict_flagged(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_c6_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    memo = repo / ".omx" / "research" / "council_per_substrate_c6_20260518.md"
    memo.write_text("council_verdict: REFUSE\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert len(violations) == 1
    assert "REFUSE" in violations[0]


def test_gate_waiver_with_rationale_passes(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(
        "dispatch_enabled: true  # PER_SUBSTRATE_SYMPOSIUM_WAIVED:operator frontier override per session directive 2026-05-18\n"
    )
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert violations == []


def test_gate_waiver_placeholder_rejected(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_foo_modal_t4_dispatch.yaml"
    recipe.write_text(
        "dispatch_enabled: true  # PER_SUBSTRATE_SYMPOSIUM_WAIVED:<rationale>\n"
    )
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert len(violations) == 1


def test_gate_strict_mode_raises(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_c6_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    with pytest.raises(PreflightError, match=r"Catalog #325"):
        check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
            strict=True, repo_root=repo, now_utc="2026-05-18T12:00:00Z"
        )


def test_gate_strict_silent_on_clean(tmp_path):
    repo = _setup_repo(tmp_path)
    # No recipes at all
    check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        strict=True, repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )


def test_gate_multiple_violations_aggregated(tmp_path):
    repo = _setup_repo(tmp_path)
    for sid in ("c6_e4_mdl_ibps", "z3_balle", "siren"):
        recipe = repo / ".omx" / "operator_authorize_recipes" / f"substrate_{sid}_modal_t4_dispatch.yaml"
        recipe.write_text("dispatch_enabled: true\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert len(violations) == 3


def test_gate_outside_window_flagged(tmp_path):
    repo = _setup_repo(tmp_path)
    recipe = repo / ".omx" / "operator_authorize_recipes" / "substrate_c6_modal_t4_dispatch.yaml"
    recipe.write_text("dispatch_enabled: true\n")
    # Memo from > 14 days ago
    memo = repo / ".omx" / "research" / "council_old_c6_20260101.md"
    memo.write_text("council_verdict: PROCEED\n")
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=repo, now_utc="2026-05-18T12:00:00Z"
    )
    assert len(violations) == 1
    assert "NO per-substrate symposium memo" in violations[0]


def test_gate_no_recipe_dir(tmp_path):
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=tmp_path
    )
    assert violations == []


def test_gate_string_repo_root_accepted(tmp_path):
    repo = _setup_repo(tmp_path)
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        repo_root=str(repo), now_utc="2026-05-18T12:00:00Z"
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Constants & contract
# ---------------------------------------------------------------------------


def test_constants_pinned():
    assert _CHECK_325_SYMPOSIUM_WINDOW_DAYS == 14
    assert _CHECK_325_ACCEPTED_VERDICTS == frozenset({"PROCEED", "PROCEED_WITH_REVISIONS"})
    assert _CHECK_325_WAIVER_TOKEN == "PER_SUBSTRATE_SYMPOSIUM_WAIVED"
    assert "<rationale>" in _CHECK_325_PLACEHOLDER_RATIONALES
    assert "<reason>" in _CHECK_325_PLACEHOLDER_RATIONALES


def test_orchestrator_callsite_warn_only():
    """Catalog #325 must be wired warn-only (strict=False) at landing per
    CLAUDE.md "Strict-flip atomicity rule" — per-substrate symposium
    discipline is NEW; backfill is multi-subagent wave."""
    preflight_path = REPO_ROOT / "src" / "tac" / "preflight.py"
    text = preflight_path.read_text()
    assert "check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(" in text
    # find the orchestrator call site (not the def)
    idx = text.find("        check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(\n")
    assert idx > 0, "Orchestrator call site not found"
    snippet = text[idx : idx + 500]
    assert "strict=False" in snippet, f"Catalog #325 should be wired warn-only at landing: {snippet[:300]}"


def test_live_repo_regression_guard():
    """Live count at landing is bounded; warn-only baseline acceptable."""
    violations = check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor(
        strict=False, verbose=False
    )
    # At landing: 2 violations (rudin_floor + z6_v2_candidate_1_multi_layer_film)
    # Bound at 20 to allow for new dispatchable substrates being added during
    # the multi-subagent backfill wave.
    assert len(violations) <= 20, f"Live count {len(violations)} exceeds warn-only ceiling 20"


def test_185_sister_gate_callable_via_globals():
    """Per Catalog #185 META-meta-meta drift detection: this gate's function
    MUST be callable via tac.preflight module globals so the META-meta gate
    can verify Live count claims in CLAUDE.md catalog table."""
    import tac.preflight as preflight_mod
    fn = getattr(preflight_mod, "check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor", None)
    assert fn is not None
    assert callable(fn)
