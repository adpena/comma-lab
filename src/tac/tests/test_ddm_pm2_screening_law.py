"""Charter declaration regression tests; no scoring or pricing workloads."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue():
    path = Path(__file__).resolve().parents[3] / "tools" / "codex_arm_queue.py"
    spec = importlib.util.spec_from_file_location("pm2_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def charter(tmp_path):
    def write(section, extra=""):
        path = tmp_path / "charter.md"
        path.write_text(
            "# Build renderer\n## MANDATE\nBuild and measure the real renderer.\n"
            "## PRIOR NEGATIVE\nThe earlier global scale screen reversed sign.\n"
            "## OPTIMAL FORM\nReference receipt: renderer.py at abc1234.\n"
            + section + "\n" + extra,
            encoding="utf-8",
        )
        return str(path)
    return write


@pytest.mark.parametrize("scope", [
    "SCOPE reductions: subset of pairs.",
    "SCOPE reductions: n120 screen.",
    "SCOPE reductions: n=32, seeded random.",
    "SCOPE reductions: 12-pair sizing.",
    "SCOPE reductions: screen-vs-verdict labels.",
    "SCOPE reductions: pair subset,\n  seeded selection.",
    "SCOPE reduction: subset.",
    "SCOPE reductions:\nscreen on seeded random pairs.",
    "SCOPE reductions: n<600.",
])
def test_sampled_scope_requires_confinement(queue, charter, scope):
    assert "lacks actuator confinement" in queue.lint_charter_screening_law(charter(scope))[0]


def test_pair_confined_screen_is_admissible(queue, charter):
    assert queue.lint_charter_screening_law(charter(
        "SCOPE: n120 screen. Actuator confinement: pair-confined."
    )) == []


def test_incident_shape_global_scale_screen_is_mechanism_reduction(queue, charter):
    findings = queue.lint_charter_screening_law(charter(
        "SCOPE reductions: 120-pair screen for renderer scale moves.\n"
        "Actuator confinement: global. Full n600 confirmation follows."
    ))
    assert "MECHANISM reduction" in findings[0]
    assert "verdict-invalid" in findings[0]


def test_global_toy_bracket_allows_diagnostic_screen(queue, charter):
    assert queue.lint_charter_screening_law(charter(
        "SCOPE: n120 screen. Actuator: global.\n"
        "TOY-BRACKET: diagnostic-only sampling; no family verdict."
    )) == []


@pytest.mark.parametrize("bracket", [
    "none", "none; full n600 later", "<rationale here>",
    "required before running", "TODO supply diagnostic rationale", "true",
])
def test_placeholder_brackets_do_not_waive_global_screen(queue, charter, bracket):
    assert "MECHANISM reduction" in queue.lint_charter_screening_law(charter(
        "SCOPE: n32 screen. Actuator: global.\nTOY-BRACKET: " + bracket
    ))[0]


def test_no_scope_reduction_does_not_fire(queue, charter):
    assert queue.lint_charter_screening_law(charter(
        "SCOPE reductions: none. MECHANISM: no subset screen."
    )) == []


def test_full_population_screen_does_not_fire(queue, charter):
    assert queue.lint_charter_screening_law(charter("SCOPE: n600 screen.")) == []


def test_scope_deltas_none_does_not_inherit_later_prohibitions(queue, charter):
    assert queue.lint_charter_screening_law(charter(
        "SCOPE deltas allowed: none. A subset screen would be a toy; refuse it."
    )) == []


def test_subset_prohibition_does_not_fire(queue, charter):
    assert queue.lint_charter_screening_law(charter(
        "SCOPE reductions: step budget. MECHANISM: no subset-n verdict."
    )) == []


def test_later_section_confinement_cannot_clear_current_scope(queue, charter):
    assert queue.lint_charter_screening_law(charter(
        "SCOPE: n32 screen.", "## HISTORY\nActuator: pair-confined."
    ))


def test_comments_cannot_supply_confinement(queue, charter):
    assert queue.lint_charter_screening_law(charter(
        "SCOPE: n32 screen. <!-- actuator: pair-confined -->"
    ))


def test_markdown_decoration_does_not_hide_scope(queue, charter):
    assert queue.lint_charter_screening_law(charter(
        "**SCOPE**: `n32` screen. **Actuator confinement**: `global`."
    ))


def test_mixed_actuators_need_global_bracket(queue, charter):
    assert "MECHANISM reduction" in queue.lint_charter_screening_law(charter(
        "SCOPE: n32 screen. Token actuator: pair-confined; renderer actuator: global."
    ))[0]


@pytest.mark.parametrize("confinement", ["not pair-confined", "not a pair-confined actuator"])
def test_negated_confinement_is_not_a_declaration(queue, charter, confinement):
    assert "lacks actuator confinement" in queue.lint_charter_screening_law(charter(
        "SCOPE: n32 screen. Actuator: " + confinement
    ))[0]


def test_confinement_menu_is_not_a_choice(queue, charter):
    assert "lists alternatives" in queue.lint_charter_screening_law(charter(
        "SCOPE: n32 screen. Actuator confinement: pair-confined | global."
    ))[0]


def test_warn_only_default_even_when_optimal_form_is_strict(queue, charter, monkeypatch):
    monkeypatch.delenv("TAC_SCREENING_LAW_STRICT", raising=False)
    monkeypatch.setenv("TAC_CHARTER_LINT_STRICT", "1")
    path = charter("SCOPE: n32 screen.\nRECALL_LINT_NA: this is a unit fixture.")
    assert not any("screening-law:" in p for p in queue.lint_charter_optimal_form(path))
    assert any("screening-law:" in p for p in queue.lint_charter_recall_advisories(path))


def test_strict_mode_routes_to_refusal_not_duplicate_warning(queue, charter, monkeypatch):
    monkeypatch.setenv("TAC_SCREENING_LAW_STRICT", "1")
    path = charter("SCOPE: n32 screen.\nRECALL_LINT_NA: this is a unit fixture.")
    assert any("screening-law:" in p for p in queue.lint_charter_optimal_form(path))
    assert queue.lint_charter_recall_advisories(path) == []


def test_optimal_form_waiver_does_not_override_screening_law(queue, charter, monkeypatch):
    monkeypatch.setenv("TAC_SCREENING_LAW_STRICT", "1")
    path = charter("OPTIMAL_FORM_NA: bounded apparatus audit.\nSCOPE: n32 screen.")
    assert any("screening-law:" in p for p in queue.lint_charter_optimal_form(path))


def test_unreadable_charter_is_visible(queue, tmp_path):
    assert "unreadable" in queue.lint_charter_screening_law(str(tmp_path / "absent.md"))[0]


@pytest.mark.parametrize("strict, expected_rc, expected_rows", [(False, 0, 1), (True, 3, 0)])
def test_real_add_path_warns_or_refuses_before_queue_write(
    queue, charter, monkeypatch, capsys, strict, expected_rc, expected_rows
):
    path = charter("SCOPE: n120 screen.\nRECALL_LINT_NA: bounded fixture.")
    monkeypatch.setenv("TAC_SCREENING_LAW_STRICT", "1" if strict else "0")
    monkeypatch.setenv("TAC_CHARTER_LINT_STRICT", "1")
    monkeypatch.setattr(queue, "charter_file_path", lambda value: (Path(value), None))
    monkeypatch.setattr(queue, "_is_managed_charter", lambda value: False)
    monkeypatch.setattr(queue, "lint_charter_capability_advisories", lambda value: [])
    monkeypatch.setattr(queue, "lint_charter_fm_advisories", lambda value: [])
    rows = []
    monkeypatch.setattr(queue, "append_row", rows.append)
    args = SimpleNamespace(prompt=path, name="screen_fixture", rank=1, owns_scorer=False, note="")
    assert queue.cmd_add(args) == expected_rc
    assert len(rows) == expected_rows
    output = capsys.readouterr().out
    assert "screening-law:" in output
    assert ("REFUSED" if strict else "WARN") in output
