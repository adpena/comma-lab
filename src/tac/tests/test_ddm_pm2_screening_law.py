"""Charter declaration regression tests; no scoring or pricing workloads."""
from __future__ import annotations

import hashlib
import importlib.util
import json
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
    monkeypatch.setenv("TAC_SCREENING_LAW_STRICT", "0")
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


def test_screening_law_is_strict_by_default_after_zero_census(queue, charter, monkeypatch):
    monkeypatch.delenv("TAC_SCREENING_LAW_STRICT", raising=False)
    path = charter("SCOPE: n120 screen.\nRECALL_LINT_NA: bounded fixture.")
    assert any("screening-law:" in p for p in queue.lint_charter_optimal_form(path))
    assert queue.lint_charter_recall_advisories(path) == []


def _write_historical_exemption(queue, charter_path: Path, evidence_path: Path, **changes):
    row = {
        "schema": "ddm_pm2_screening_law_exemption.v1",
        "classification": "historical_finished_before_law",
        "charter_path": str(charter_path.resolve().relative_to(queue._REPO.resolve())),
        "charter_sha256": hashlib.sha256(charter_path.read_bytes()).hexdigest(),
        "evidence_path": str(evidence_path.resolve().relative_to(queue._REPO.resolve())),
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "finished_at_utc": "2026-09-09T22:00:00+00:00",
        "law_cutoff_utc": queue.SCREENING_LAW_CUTOFF_UTC,
        "law_landing_commit": queue.SCREENING_LAW_LANDING_COMMIT,
    }
    row.update(changes)
    queue.SCREENING_LAW_EXEMPTIONS.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return row


def test_exact_historical_exemption_clears_only_its_finished_charter(
    queue, charter, monkeypatch, tmp_path
):
    monkeypatch.setattr(queue, "_REPO", tmp_path)
    exemptions = tmp_path / ".omx/research/exemptions.jsonl"
    exemptions.parent.mkdir(parents=True)
    monkeypatch.setattr(queue, "SCREENING_LAW_EXEMPTIONS", exemptions)
    charter_path = Path(charter("SCOPE: n120 screen."))
    evidence = tmp_path / ".omx/research/terminal.md"
    evidence.write_text("Terminal verdict: complete.\n", encoding="utf-8")
    _write_historical_exemption(queue, charter_path, evidence)
    assert queue.lint_charter_screening_law(str(charter_path)) == []


@pytest.mark.parametrize(
    "change",
    [
        {"charter_sha256": "0" * 64},
        {"evidence_sha256": "0" * 64},
        {"finished_at_utc": "2026-09-10T00:00:00+00:00"},
        {"classification": "historical"},
    ],
)
def test_stale_or_post_cutoff_historical_exemption_fails_closed(
    queue, charter, monkeypatch, tmp_path, change
):
    monkeypatch.setattr(queue, "_REPO", tmp_path)
    exemptions = tmp_path / ".omx/research/exemptions.jsonl"
    exemptions.parent.mkdir(parents=True)
    monkeypatch.setattr(queue, "SCREENING_LAW_EXEMPTIONS", exemptions)
    charter_path = Path(charter("SCOPE: n120 screen."))
    evidence = tmp_path / ".omx/research/terminal.md"
    evidence.write_text("Terminal verdict: complete.\n", encoding="utf-8")
    _write_historical_exemption(queue, charter_path, evidence, **change)
    assert "lacks actuator confinement" in queue.lint_charter_screening_law(
        str(charter_path)
    )[0]


def test_conflicting_duplicate_exemptions_fail_closed(
    queue, charter, monkeypatch, tmp_path
):
    monkeypatch.setattr(queue, "_REPO", tmp_path)
    exemptions = tmp_path / ".omx/research/exemptions.jsonl"
    exemptions.parent.mkdir(parents=True)
    monkeypatch.setattr(queue, "SCREENING_LAW_EXEMPTIONS", exemptions)
    charter_path = Path(charter("SCOPE: n120 screen."))
    evidence = tmp_path / ".omx/research/terminal.md"
    evidence.write_text("Terminal verdict: complete.\n", encoding="utf-8")
    row = _write_historical_exemption(queue, charter_path, evidence)
    with exemptions.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    assert "lacks actuator confinement" in queue.lint_charter_screening_law(
        str(charter_path)
    )[0]
