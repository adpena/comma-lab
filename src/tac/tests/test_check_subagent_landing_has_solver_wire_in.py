# SPDX-License-Identifier: MIT
"""Tests for preflight Catalog #125:
``check_subagent_landing_has_solver_wire_in``.

Operationalises CLAUDE.md "Subagent coherence-by-default" non-negotiable
(landed 2026-05-09). The check refuses post-cutover landing memos
(``feedback_*_landed_<YYYYMMDD>.md`` with ``YYYYMMDD >= 20260509``) that do
not declare ALL 6 unified-Lagrangian wire-in hooks:

  1. Sensitivity-map contribution
  2. Pareto constraint
  3. Bit-allocator hook
  4. Cathedral autopilot dispatch hook
  5. Continual-learning posterior update
  6. Probe-disambiguator (if 2+ defensible interpretations)

Opt-outs:
  - ``research_only=true`` declared in the memo body
  - Per-hook ``<HookAlias>: N/A — <rationale>``

This test set verifies:
  1. Pre-cutover memo (date < 20260509) → exempt
  2. Post-cutover memo with all 6 hooks → no warning
  3. Post-cutover memo missing hooks → warns once with hook list
  4. Each hook missing individually → warns naming that hook
  5. ``research_only=true`` opt-out → no warning
  6. Per-hook ``N/A — <rationale>`` opt-out → no warning
  7. Per-hook ``N/A`` WITHOUT rationale → still warns
  8. Strict mode raises with formatted message
  9. Hook-alias detection (multiple spellings accepted)
 10. Memo dir missing → degrades gracefully
 11. Performance: 1000 memo scan in < 1s
 12. Empty memo → warns on all 6 hooks
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tac.preflight import (
    _LANDING_MEMO_HOOK_CUTOVER_YYYYMMDD,
    _UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS,
    PreflightError,
    _landing_memo_date,
    _memo_declares_hook,
    _memo_declares_research_only,
    _strip_code_spans_and_fences,
    check_subagent_landing_has_solver_wire_in,
)


def _all_6_hooks_body() -> str:
    """Return a memo body that declares ALL 6 hooks in the canonical form."""
    return """---
name: T99 example landing
type: feedback
---

# Implementation log

This is an example. The 6 unified-Lagrangian wire-ins are declared
explicitly below per CLAUDE.md "Subagent coherence-by-default":

1. Sensitivity-map contribution: writes per-pixel ∂L_T99/∂θ into tac.sensitivity_map.t99.
2. Pareto constraint: adds rate ≤ R_T99 to the convex feasible set.
3. Bit-allocator hook: register T99 as a per-tensor importance source.
4. Cathedral autopilot dispatch: T99 packs into archive variant 'apogee_t99'.
5. Continual-learning posterior update: T99 anchor calls posterior_update().
6. Probe-disambiguator: T99 has KL-vs-MSE alternatives; ships both modes.
"""


def _make_memo(tmp: Path, date: int, slug: str, body: str) -> Path:
    """Create a feedback_<slug>_landed_<date>.md file in tmp."""
    p = tmp / f"feedback_{slug}_landed_{date}.md"
    p.write_text(body, encoding="utf-8")
    return p


# ── Test 1: pre-cutover memo (date < 20260509) is exempt ────────────────


def test_pre_cutover_memo_no_warning(tmp_path: Path) -> None:
    """A 2026-05-08 memo predates the rule — no enforcement."""
    _make_memo(tmp_path, 20260508, "old_thing", "no hooks declared")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_well_pre_cutover_memo_no_warning(tmp_path: Path) -> None:
    _make_memo(tmp_path, 20260101, "ancient", "no hooks declared")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_cutover_date_itself_is_enforced(tmp_path: Path) -> None:
    """The cutover date is INCLUSIVE: 20260509 is enforced."""
    _make_memo(tmp_path, 20260509, "cutover_day", "no hooks declared")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


# ── Test 2: post-cutover memo with all 6 hooks → no warning ──────────────


def test_all_6_hooks_present_no_warning(tmp_path: Path) -> None:
    _make_memo(tmp_path, 20260510, "complete_landing", _all_6_hooks_body())
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], f"got: {violations}"


# ── Test 3: post-cutover memo missing hooks → warns ──────────────────────


def test_no_hooks_declared_warns_with_all_6(tmp_path: Path) -> None:
    """A bare memo with no hook declarations warns naming all 6."""
    _make_memo(tmp_path, 20260520, "bare_landing", "Just some text. No hooks.")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    msg = violations[0]
    assert "[Check 125]" in msg
    for hook_label, _ in _UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS:
        assert hook_label in msg


# ── Test 4: each hook missing individually → warns naming that hook ──────


@pytest.mark.parametrize(
    "drop_hook",
    [h for h, _ in _UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS],
)
def test_each_hook_missing_individually_warns(
    tmp_path: Path, drop_hook: str,
) -> None:
    """Drop one hook line at a time; check warns naming that hook."""
    body = _all_6_hooks_body()
    # Build a body with one hook removed by stripping every alias
    aliases = next(a for h, a in _UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS if h == drop_hook)
    new_lines: list[str] = []
    for line in body.splitlines():
        line_lower = line.lower()
        if any(a.lower() in line_lower for a in aliases):
            continue
        new_lines.append(line)
    body_stripped = "\n".join(new_lines)
    _make_memo(tmp_path, 20260510, f"missing_{drop_hook}", body_stripped)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert drop_hook in violations[0], (
        f"expected {drop_hook} in violation msg, got: {violations[0]}"
    )


# ── Test 5: research_only=true opt-out ───────────────────────────────────


def test_research_only_opt_out_no_warning(tmp_path: Path) -> None:
    body = "no hooks declared.\nresearch_only=true (rationale: pure research artifact)"
    _make_memo(tmp_path, 20260510, "research_only_landing", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_research_only_with_colon_opt_out(tmp_path: Path) -> None:
    body = "no hooks declared.\nresearch_only: true (rationale: research)"
    _make_memo(tmp_path, 20260510, "research_only_colon", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_research_only_false_does_not_opt_out(tmp_path: Path) -> None:
    body = "research_only=false  ; no hook declarations"
    _make_memo(tmp_path, 20260510, "research_only_false", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


# ── Test 6: per-hook N/A with rationale opts that hook out ───────────────


def test_per_hook_na_with_rationale_no_warning(tmp_path: Path) -> None:
    """Each hook may opt out via '<Alias>: N/A — <rationale>'."""
    body = """
1. Sensitivity-map: N/A — preflight infrastructure, not a score signal
2. Pareto: N/A — no new constraint
3. Bit-allocator: N/A — no per-tensor importance change
4. Cathedral autopilot: N/A — not archive-deployable
5. Continual-learning: N/A — no empirical anchor produced
6. Probe-disambiguator: N/A — both checks are deterministic; no design tension
"""
    _make_memo(tmp_path, 20260510, "all_na", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], f"got: {violations}"


def test_per_hook_na_without_rationale_still_warns(tmp_path: Path) -> None:
    """Bare 'Sensitivity-map: N/A' (no rationale) does NOT opt that hook out."""
    body = "1. Sensitivity-map: N/A\n"  # no rationale — must warn
    _make_memo(tmp_path, 20260510, "bare_na", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    # All 6 hooks should be in the violation (sensitivity wasn't really declared
    # because the N/A had no rationale).
    assert "sensitivity_map" in violations[0]


# ── Test 7: strict mode raises with formatted message ────────────────────


def test_strict_mode_raises(tmp_path: Path) -> None:
    _make_memo(tmp_path, 20260510, "missing", "no hooks here")
    with pytest.raises(PreflightError) as excinfo:
        check_subagent_landing_has_solver_wire_in(
            memory_dir=tmp_path, strict=True, verbose=False,
        )
    msg = str(excinfo.value)
    assert "check_subagent_landing_has_solver_wire_in" in msg
    assert "Required hooks" in msg
    # Should name at least one missing hook
    for h, _ in _UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS:
        if h in msg:
            return
    pytest.fail(f"no hook label appeared in error message: {msg}")


def test_strict_mode_no_violations_does_not_raise(tmp_path: Path) -> None:
    _make_memo(tmp_path, 20260510, "ok", _all_6_hooks_body())
    # Should not raise
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=True, verbose=False,
    )
    assert violations == []


# ── Test 8: hook-alias detection ─────────────────────────────────────────


@pytest.mark.parametrize(
    "alias", [
        "Sensitivity-map", "sensitivity_map", "sensitivity map", "SENSITIVITYMAP",
    ],
)
def test_sensitivity_map_alias_detection(tmp_path: Path, alias: str) -> None:
    """Various aliases for 'sensitivity_map' all count."""
    body = (
        f"{alias} contribution: real declaration\n"
        "Pareto constraint: real declaration\n"
        "Bit-allocator hook: real declaration\n"
        "Cathedral autopilot dispatch: real declaration\n"
        "Continual-learning posterior update: real declaration\n"
        "Probe-disambiguator: real declaration\n"
    )
    _make_memo(tmp_path, 20260510, f"alias_{abs(hash(alias))}", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], f"alias '{alias}' not detected; violations: {violations}"


# ── Test 9: memo dir missing degrades gracefully ─────────────────────────


def test_missing_memory_dir_returns_setup_warning(tmp_path: Path) -> None:
    nonexistent = tmp_path / "not_there"
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=nonexistent, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "memory dir not present" in violations[0]


def test_missing_memory_dir_strict_raises(tmp_path: Path) -> None:
    """A missing memo dir is a setup gap that strict mode must surface."""
    nonexistent = tmp_path / "not_there"
    with pytest.raises(PreflightError, match="memory dir not present"):
        check_subagent_landing_has_solver_wire_in(
            memory_dir=nonexistent, strict=True, verbose=False,
        )


def test_auto_missing_memory_dir_skips_on_github_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repo CI must not fail on an operator-local Claude memory path."""
    monkeypatch.delenv("PACT_MEMORY_DIR", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert check_subagent_landing_has_solver_wire_in(
        strict=True, verbose=False,
    ) == []


def test_env_missing_memory_dir_still_strict_raises_on_github_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit memory-dir contract remains enforceable in CI."""
    monkeypatch.setenv("PACT_MEMORY_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    with pytest.raises(PreflightError, match="memory dir not present"):
        check_subagent_landing_has_solver_wire_in(strict=True, verbose=False)


# ── Test 10: empty memo body warns on all 6 ──────────────────────────────


def test_empty_memo_warns_on_all_6(tmp_path: Path) -> None:
    _make_memo(tmp_path, 20260510, "empty", "")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    for h, _ in _UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS:
        assert h in violations[0]


# ── Test 11: non-landing memo (no _landed_<date>) is ignored ─────────────


def test_non_landing_memo_ignored(tmp_path: Path) -> None:
    """A regular feedback_*.md file is not a landing memo and is ignored."""
    p = tmp_path / "feedback_some_topic.md"
    p.write_text("no hooks", encoding="utf-8")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_project_landed_file_ignored_unless_feedback_prefix(tmp_path: Path) -> None:
    """A project_*_landed_*.md file is NOT scanned by Check 125."""
    p = tmp_path / "project_topic_landed_20260510.md"
    p.write_text("no hooks", encoding="utf-8")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 12: helper unit tests ───────────────────────────────────────────


def test_landing_memo_date_extracts_correctly() -> None:
    p = Path("feedback_foo_landed_20260509.md")
    assert _landing_memo_date(p) == 20260509


def test_landing_memo_date_returns_none_for_non_landing() -> None:
    p = Path("feedback_foo.md")
    assert _landing_memo_date(p) is None


def test_landing_memo_date_returns_none_for_malformed() -> None:
    p = Path("feedback_foo_landed_BADDATE.md")
    assert _landing_memo_date(p) is None


def test_landing_memo_date_handles_long_path() -> None:
    p = Path("/tmp/feedback_long_compound_topic_landed_20260601.md")
    assert _landing_memo_date(p) == 20260601


def test_research_only_detection_case_insensitive() -> None:
    assert _memo_declares_research_only("foo\nresearch_only=true\nbar")
    assert _memo_declares_research_only("RESEARCH_ONLY=TRUE")
    assert _memo_declares_research_only("research_only: true")
    assert not _memo_declares_research_only("research_only=false")
    assert not _memo_declares_research_only("no opt-out here")


def test_memo_declares_hook_canonical_form() -> None:
    aliases = ("sensitivity-map", "sensitivity map", "sensitivity_map")
    text = "Sensitivity-map contribution: declared"
    assert _memo_declares_hook(text, "sensitivity_map", aliases)


def test_memo_declares_hook_with_markdown_bold_label() -> None:
    aliases = ("probe-disambiguator", "probe disambiguator", "probe_disambiguator")
    text = "6. **Probe-disambiguator**: ships both interpretations behind CLI flags."
    assert _memo_declares_hook(text, "probe_disambiguator", aliases)


def test_memo_declares_hook_na_form_with_rationale() -> None:
    aliases = ("pareto", "pareto-constraint")
    text = "Pareto: N/A — no new constraint"
    assert _memo_declares_hook(text, "pareto", aliases)


def test_memo_declares_hook_na_form_without_rationale_rejected() -> None:
    aliases = ("pareto", "pareto-constraint")
    text = "Pareto: N/A"
    assert not _memo_declares_hook(text, "pareto", aliases)


def test_memo_declares_hook_irrelevant_text_rejected() -> None:
    aliases = ("pareto",)
    text = "Some unrelated paragraph mentioning Pareto in passing somewhere."
    assert not _memo_declares_hook(text, "pareto", aliases)


@pytest.mark.parametrize(
    "text",
    [
        "Sensitivity-map: not wired yet",
        "Sensitivity-map hook deferred",
        "Sensitivity-map contribution missing",
        "Sensitivity-map: TODO after trainer lands",
    ],
)
def test_memo_declares_hook_negative_declarations_rejected(text: str) -> None:
    aliases = ("sensitivity-map", "sensitivity map", "sensitivity_map")
    assert not _memo_declares_hook(text, "sensitivity_map", aliases)


# ── Test 13: performance — 1000 memo scan ────────────────────────────────


def test_performance_1000_memos_under_1s(tmp_path: Path) -> None:
    """1000 memo scan completes in < 1s."""
    body = _all_6_hooks_body()  # All 6 hooks present, no violation
    for i in range(1000):
        _make_memo(tmp_path, 20260510, f"perf_{i:04d}", body)
    t0 = time.perf_counter()
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    assert violations == []
    assert elapsed < 1.0, f"scan took {elapsed:.2f}s for 1000 memos (expected < 1.0s)"


# ── Test 14: only first violation per memo ───────────────────────────────


def test_one_violation_per_memo(tmp_path: Path) -> None:
    """A memo missing 6 hooks yields ONE violation entry, not 6."""
    _make_memo(tmp_path, 20260510, "many_missing", "no hooks at all")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


# ── Test 15: multiple memos ──────────────────────────────────────────────


def test_multiple_memos_separate_violations(tmp_path: Path) -> None:
    """Two missing memos yield TWO violation entries."""
    _make_memo(tmp_path, 20260510, "first", "no hooks")
    _make_memo(tmp_path, 20260511, "second", "no hooks")
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 2


def test_multiple_memos_some_pass_some_fail(tmp_path: Path) -> None:
    _make_memo(tmp_path, 20260510, "ok_one", _all_6_hooks_body())
    _make_memo(tmp_path, 20260511, "fail_one", "no hooks")
    _make_memo(tmp_path, 20260512, "ok_two", _all_6_hooks_body())
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "fail_one" in violations[0]


def test_constants_are_well_formed() -> None:
    """Sanity-check that the module-level constants are tuples of expected shape."""
    assert _LANDING_MEMO_HOOK_CUTOVER_YYYYMMDD == 20260509
    assert len(_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS) == 6
    for hook_label, aliases in _UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS:
        assert isinstance(hook_label, str)
        assert isinstance(aliases, tuple)
        assert len(aliases) >= 1


# ── Test 16: code-span / fence stripping & opt-out-in-doc protection ─────


def test_strip_code_spans_inline() -> None:
    """`research_only=true` mentioned inside backticks is stripped before scan."""
    text = "OR add `research_only=true` somewhere; sample code."
    stripped = _strip_code_spans_and_fences(text)
    assert "research_only=true" not in stripped


def test_strip_code_spans_fenced() -> None:
    text = "Sample:\n```\nresearch_only=true\n```\nfollow-up."
    stripped = _strip_code_spans_and_fences(text)
    assert "research_only=true" not in stripped


def test_strip_code_spans_preserves_outside_text() -> None:
    text = "Outside `code` rest"
    stripped = _strip_code_spans_and_fences(text)
    assert "Outside" in stripped
    assert "rest" in stripped


def test_research_only_in_backticks_is_documentation_not_optout(
    tmp_path: Path,
) -> None:
    """A memo that mentions `research_only=true` in a code span is NOT opting out."""
    body = (
        "**OR** add `research_only=true` (with rationale).\n"
        "But this memo declares all 6 hooks below:\n"
        "1. Sensitivity-map: present\n"
        "2. Pareto: present\n"
        "3. Bit-allocator: present\n"
        "4. Cathedral autopilot: present\n"
        "5. Continual-learning: present\n"
        "6. Probe-disambiguator: present\n"
    )
    _make_memo(tmp_path, 20260510, "doc_in_backticks", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    # Memo should pass because the backticked opt-out is documentation
    # AND all 6 hooks are declared. So no violation.
    assert violations == [], f"got: {violations}"


def test_explicit_research_only_false_negates_other_optout_mentions(
    tmp_path: Path,
) -> None:
    """If memo declares research_only=false, in-narrative research_only=true is doc."""
    body = (
        "research_only=false\n"
        "Note: the alternative would be `research_only=true` (omitted here).\n"
        "But because we declare 6/6 N/A below:\n"
        "1. Sensitivity-map: N/A — preflight infrastructure\n"
        "2. Pareto: N/A — no constraint\n"
        "3. Bit-allocator: N/A — no importance\n"
        "4. Cathedral autopilot: N/A — not deployable\n"
        "5. Continual-learning: N/A — no anchor\n"
        "6. Probe-disambiguator: N/A — deterministic\n"
    )
    _make_memo(tmp_path, 20260510, "explicit_false", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], f"got: {violations}"


# ── Test 13: 2026-08-25 detector cure (ddm_wb1) — multi-word hook labels ──
#
# The six CLAUDE.md hook labels are multi-word ("Continual-learning POSTERIOR
# update", "Sensitivity-map CONTRIBUTION", ...). When a memo writes the FULL
# label, the declaration marker sits AFTER the trailing label noun rather than
# adjacent to the matched alias. The trailing nouns of hooks #1-#4 were already
# inside the marker keyword set; hook #5's ("posterior") was not, so 101 of 124
# era memos failed on hook #5 alone while carrying a fully substantive
# "hook #5 continual-learning posterior = ACTIVE (...)" declaration.
#
# POSITIVE controls: the real corpus forms must now be accepted.
# NEGATIVE controls: narrative mention, bare N/A, and explicit absence must
# still be refused — the cure only adds acceptance paths for genuine
# declarations, it does not weaken the gate.


@pytest.mark.parametrize(
    "line",
    [
        # Verbatim forms harvested from the 2026-05/06 landing-memo corpus.
        "* **Hook #5 continual-learning posterior**: ACTIVE via canonical "
        "equation registration; anchors auto-append per Catalog #371.",
        "- **hook #5 continual-learning posterior** = ACTIVE — the A/B "
        "reproduction and this verdict are the anchors.",
        "5. **Continual-learning posterior** — ACTIVE: the canonical equation "
        "gains EmpiricalAnchor #4.",
        "- hook #5 continual-learning posterior = **ACTIVE** (probe outcome "
        "row appended via the canonical helper)",
        "continual_learning_posterior: ACTIVE",
    ],
)
def test_cure_accepts_full_label_hook5_declarations(line: str) -> None:
    """POSITIVE: real corpus declarations of hook #5 are recognised."""
    aliases = dict(_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS)["continual_learning"]
    assert _memo_declares_hook(line + "\n", "continual_learning", aliases)


@pytest.mark.parametrize(
    ("hook", "line"),
    [
        (
            "sensitivity_map",
            "1. **Sensitivity-map** — ACTIVE: the per-op drift table is the "
            "per-op sensitivity contribution.",
        ),
        (
            "probe_disambiguator",
            "6. **Probe-disambiguator** — ACTIVE: the harness IS the "
            "disambiguator between the two readings.",
        ),
    ],
)
def test_cure_accepts_em_dash_declaration_separator(hook: str, line: str) -> None:
    """POSITIVE: '**Label** — STATUS: evidence' is a declaration, not prose."""
    aliases = dict(_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS)[hook]
    assert _memo_declares_hook(line + "\n", hook, aliases)


def test_cure_preserves_table_form_declaration() -> None:
    """POSITIVE (regression): the continuation strip must be ADDITIVE.

    "| #2 Pareto constraint | ACTIVE | ..." is accepted through the UNSTRIPPED
    window via the 'constraint' keyword. An earlier draft of the cure consumed
    'constraint' destructively and broke this form.
    """
    aliases = dict(_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS)["pareto"]
    line = "| #2 Pareto constraint | ACTIVE | rate <= 180_215 B |\n"
    assert _memo_declares_hook(line, "pareto", aliases)


@pytest.mark.parametrize(
    "line",
    [
        # Narrative mention: no declaration marker anywhere after the label.
        "The continual-learning posterior rows accumulate over successive "
        "waves and future agents inherit them.",
        # Bare N/A with no rationale: still refused (CLAUDE.md requires one).
        "- hook #5 continual-learning posterior = N/A",
        # Explicit absence: acknowledging a hook is unwired is not a wire-in.
        "- hook #5 continual-learning posterior — not wired",
        "- hook #5 continual-learning posterior: deferred",
        # Label alone as a heading tail.
        "## Continual-learning posterior",
    ],
)
def test_cure_still_refuses_non_declarations(line: str) -> None:
    """NEGATIVE: the cure must not accept prose, bare N/A, or absence."""
    aliases = dict(_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS)["continual_learning"]
    assert not _memo_declares_hook(line + "\n", "continual_learning", aliases)


def test_cure_still_refuses_hyphenated_continuation() -> None:
    """NEGATIVE: an ASCII hyphen is NOT a declaration separator.

    'sensitivity-map-based' must not be read as '**Sensitivity-map** - ...'.
    """
    aliases = dict(_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS)["sensitivity_map"]
    line = "We used a sensitivity-map-based heuristic to pick the tensors.\n"
    assert not _memo_declares_hook(line, "sensitivity_map", aliases)


def test_cure_end_to_end_memo_with_full_labels_passes(tmp_path: Path) -> None:
    """POSITIVE end-to-end: a memo in the corpus's own convention passes."""
    body = (
        "# Landing\n\n"
        "## 6-hook wire-in (Catalog #125)\n"
        "- **hook #1 sensitivity-map contribution** = ACTIVE (per-op table)\n"
        "- **hook #2 Pareto constraint** = ACTIVE (rate bound tightened)\n"
        "- **hook #3 bit-allocator hook** = ACTIVE (per-tensor importance)\n"
        "- **hook #4 cathedral autopilot dispatch** = ACTIVE (recipe landed)\n"
        "- **hook #5 continual-learning posterior** = ACTIVE (anchor appended)\n"
        "- **hook #6 probe-disambiguator** = ACTIVE (two readings arbitrated)\n"
    )
    _make_memo(tmp_path, 20260530, "full_label_convention", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert violations == [], f"got: {violations}"


def test_cure_end_to_end_memo_without_declarations_still_fires(
    tmp_path: Path,
) -> None:
    """NEGATIVE end-to-end: a memo that only NARRATES the hooks still fires."""
    body = (
        "# Landing\n\n"
        "We touched the sensitivity-map-based heuristic and watched the\n"
        "continual-learning posterior rows accumulate. The Pareto frontier\n"
        "moved. A bit-allocator exists. The cathedral autopilot ran. The\n"
        "probe-disambiguator concept is interesting.\n"
    )
    _make_memo(tmp_path, 20260530, "narration_only", body)
    violations = check_subagent_landing_has_solver_wire_in(
        memory_dir=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1, f"got: {violations}"
    assert "sensitivity_map" in violations[0]
    assert "continual_learning" in violations[0]


# ── Test 14: 2026-08-25 detector cure #2 (ddm_wb1) — status slot vs qualifier ──
#
# A negative token ("deferred" / "missing" / "todo") means "this hook is not
# wired" only when it occupies the STATUS SLOT. When the status slot already
# says ACTIVE and the negative token merely qualifies the evidence downstream,
# the line IS a declaration. Three corpus lines were refused for this reason:
#   "hook #2 Pareto constraint = ACTIVE via Dykstra-feasibility check +
#    canonical equation candidate registration DEFERRED"


@pytest.mark.parametrize(
    ("hook", "line"),
    [
        (
            "pareto",
            "- hook #2 Pareto constraint = ACTIVE via Dykstra-feasibility "
            "check + canonical equation candidate registration DEFERRED",
        ),
        (
            "continual_learning",
            "- Hook #5 continual-learning posterior = ACTIVE (empirical "
            "anchor accumulates for D.1 deferred-registration)",
        ),
        (
            "continual_learning",
            "- Hook #5 continual-learning posterior = ACTIVE (deferred "
            "canonical equation + anti-pattern candidates)",
        ),
    ],
)
def test_cure2_accepts_active_status_with_downstream_qualifier(
    hook: str, line: str,
) -> None:
    """POSITIVE: ACTIVE in the status slot survives a downstream qualifier."""
    aliases = dict(_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS)[hook]
    assert _memo_declares_hook(line + "\n", hook, aliases)


@pytest.mark.parametrize(
    ("hook", "line"),
    [
        # Negative token IS the status -> still refused.
        ("pareto", "- hook #2 Pareto constraint = DEFERRED"),
        ("pareto", "- hook #2 Pareto constraint: TODO"),
        ("bit_allocator", "- hook #3 bit-allocator: not wired"),
        ("bit_allocator", "- hook #3 bit-allocator = missing"),
        ("continual_learning", "- hook #5 continual-learning posterior: unwired"),
        # The 'reactivation' trap: 'active' must not match inside a longer
        # word, or "reactivation criteria missing" would read as a positive.
        (
            "bit_allocator",
            "- hook #3 bit-allocator: reactivation criteria missing",
        ),
        # Positive word appears only AFTER the absence has been stated.
        (
            "pareto",
            "- hook #2 Pareto constraint: not wired here; a sister lane "
            "registered one",
        ),
    ],
)
def test_cure2_still_refuses_absence_in_status_slot(hook: str, line: str) -> None:
    """NEGATIVE: absence in the status slot is refused, cure notwithstanding."""
    aliases = dict(_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS)[hook]
    assert not _memo_declares_hook(line + "\n", hook, aliases)
