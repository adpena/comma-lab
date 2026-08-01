# SPDX-License-Identifier: MIT
"""Tests for the vacuity cure (ddm_vc1, 2026-08-01, task #842).

THE LAW under test: an instrument that evaluated an EMPTY SCOPE emits the same
symbol as one that evaluated a full scope cleanly — vacuity is
indistinguishable from PASS.

These tests verify BEHAVIOUR, not constants (CLAUDE.md NO-FAKE forbidden class
#2: "if every test would still pass when the function body is replaced by
`return canonical_markers`, the test suite is verifying constants not
behaviour"). Concretely, the pair that matters is:

* :func:`test_gate_fires_on_planted_vacuous_verdict` — the POSITIVE control.
* :func:`test_gate_silent_on_counted_verdict_over_same_scope` — the NEGATIVE
  control, over the SAME enumerated scope, differing ONLY in whether the
  verdict carries a count.

A guard that cannot distinguish those two reproduces the bug it cures, so both
directions are asserted rather than just the firing one.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tac.confound_gates import (
    _VACUITY_PASS_TOKENS,
    POSITIVE_CONTROLS,
    check_verdict_surfaces_report_examined_count,
)
from tac.scope_ledger import (
    COMPLETE,
    PARTIAL,
    VACUOUS,
    ScopeLedger,
    ScopeVacuityError,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# The two fixtures differ ONLY in the verdict line. Same enumeration, same
# emptiness, same control flow.
_VACUOUS_SRC = (
    "from pathlib import Path\n"
    "def audit(root):\n"
    "    bad = [p for p in Path(root).rglob('*.py') if 'x' in p.name]\n"
    "    if not bad:\n"
    "        print('AUDIT PASSED')\n"
    "    return bad\n"
)
_COUNTED_SRC = (
    "from pathlib import Path\n"
    "def audit(root):\n"
    "    seen = list(Path(root).rglob('*.py'))\n"
    "    bad = [p for p in seen if 'x' in p.name]\n"
    "    if not bad:\n"
    "        print(f'AUDIT PASSED: {len(bad)} of {len(seen)} examined')\n"
    "    return bad\n"
)


def _tree(tmp_path: Path, filename: str, source: str) -> Path:
    tools = tmp_path / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / filename).write_text(source)
    return tmp_path


# ---------------------------------------------------------------------------
# ScopeLedger — the verdict ladder
# ---------------------------------------------------------------------------


def test_zero_examined_is_vacuous_not_pass() -> None:
    led = ScopeLedger(surface="gate", examined=0, declared=27)
    assert led.verdict == VACUOUS
    assert led.is_vacuous
    assert not led.examined_something


def test_vacuous_render_never_contains_an_affirmative_pass_word() -> None:
    """The entire cure is that an empty scope cannot be spelled with a pass word.

    Asserted against the gate's OWN token list rather than a crude "PASS"
    substring, which would flag the render's human-legible *negation* ("this is
    not a pass") — a phrase that makes the report better, not worse.
    """
    for declared in (0, 1, 448):
        led = ScopeLedger(surface="gate", examined=0, declared=declared)
        rendered = led.render().upper()
        for token in _VACUITY_PASS_TOKENS:
            assert token not in rendered, (token, rendered)
        assert "VACUOUS" in rendered
        assert "NOT A PASS" in rendered  # the negation is required, not incidental


def test_empty_declared_and_empty_examined_is_still_vacuous() -> None:
    """"Nothing to look at" and "I looked at nothing" read identically to a
    reader, and the reader is entitled to know rather than be told PASS."""
    assert ScopeLedger(surface="scan", examined=0, declared=0).verdict == VACUOUS


def test_partial_when_something_unexamined() -> None:
    led = ScopeLedger(surface="gate", examined=25, declared=27)
    assert led.verdict == PARTIAL
    assert led.unexamined == 2
    assert "examined 25 of 27 declared" in led.render()


def test_complete_only_when_nothing_left_over() -> None:
    led = ScopeLedger(surface="gate", examined=27, declared=27)
    assert led.verdict == COMPLETE
    assert led.unexamined == 0


def test_skipped_scope_downgrades_complete_to_partial() -> None:
    led = ScopeLedger(
        surface="gate", examined=27, declared=27, skipped_scopes=("codebase",)
    )
    assert led.verdict == PARTIAL
    assert "codebase" in led.render()


def test_population_exposes_a_filter_that_emptied_the_scope() -> None:
    """The codex-findings instance: mtime<3d left 0 of 1260 files in scope.

    Without ``population`` this reads as the innocuous "nothing to do"; with it,
    the filter's damage is legible.
    """
    led = ScopeLedger(surface="findings", examined=0, declared=0, population=1260)
    assert led.filtered_out == 1260
    assert "1260 in population before scope filter" in led.render()
    assert led.verdict == VACUOUS


def test_require_non_vacuous_raises_on_empty_scope() -> None:
    led = ScopeLedger(surface="gate", examined=0, declared=27)
    with pytest.raises(ScopeVacuityError) as exc:
        led.require_non_vacuous(hint="run without --no-codebase")
    assert "VACUOUS" in str(exc.value)
    assert "run without --no-codebase" in str(exc.value)


def test_require_non_vacuous_is_a_noop_when_something_was_examined() -> None:
    led = ScopeLedger(surface="gate", examined=1, declared=27)
    assert led.require_non_vacuous() is led


def test_acknowledgement_suppresses_the_raise_only_never_the_report() -> None:
    """Corollary under test: an override that silences a loud instrument
    reproduces the original silence with extra steps."""
    led = ScopeLedger(surface="gate", examined=0, declared=27, acknowledged=True)
    led.require_non_vacuous()  # must not raise
    assert led.verdict == VACUOUS  # still not a pass
    rendered = led.render().upper()
    for token in _VACUITY_PASS_TOKENS:
        assert token not in rendered, (token, rendered)
    assert "NOT A PASS" in rendered
    assert led.as_dict()["verdict"] == VACUOUS


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        ScopeLedger(surface="gate", examined=-1)


# ---------------------------------------------------------------------------
# The gate — positive AND negative control
# ---------------------------------------------------------------------------


def test_gate_fires_on_planted_vacuous_verdict(tmp_path: Path) -> None:
    """POSITIVE control: enumerate a scope, emit a bare-constant verdict."""
    root = _tree(tmp_path, "planted_vacuous.py", _VACUOUS_SRC)
    found = check_verdict_surfaces_report_examined_count(
        repo_root=root, strict=False, verbose=False
    )
    assert any("planted_vacuous.py" in v for v in found), found


def test_gate_silent_on_counted_verdict_over_same_scope(tmp_path: Path) -> None:
    """NEGATIVE control: same enumeration, same emptiness — but the verdict
    carries a denominator. A guard that cannot tell these apart reproduces the
    bug it cures."""
    root = _tree(tmp_path, "counted_ok.py", _COUNTED_SRC)
    found = check_verdict_surfaces_report_examined_count(
        repo_root=root, strict=False, verbose=False
    )
    assert found == [], found


def test_gate_ignores_a_bare_verdict_with_no_enumeration(tmp_path: Path) -> None:
    """A verdict about someone else's return code is not a claim about a scope,
    so it is deliberately out of this gate's signature (no false alarm)."""
    root = _tree(
        tmp_path,
        "relay.py",
        "import subprocess\n"
        "def fire(cmd):\n"
        "    if subprocess.run(cmd).returncode == 0:\n"
        "        print('dry-run PASSED')\n",
    )
    assert (
        check_verdict_surfaces_report_examined_count(
            repo_root=root, strict=False, verbose=False
        )
        == []
    )


def test_gate_respects_a_real_waiver(tmp_path: Path) -> None:
    root = _tree(
        tmp_path,
        "waived.py",
        _VACUOUS_SRC.replace(
            "    return bad\n",
            "    return bad  # VACUITY_LEDGER_OK: scope is a fixed 3-item literal\n",
        ),
    )
    assert (
        check_verdict_surfaces_report_examined_count(
            repo_root=root, strict=False, verbose=False
        )
        == []
    )


def test_gate_rejects_a_placeholder_waiver(tmp_path: Path) -> None:
    """Catalog #287 sister discipline: the docstring example cannot self-waive."""
    root = _tree(
        tmp_path,
        "fake_waiver.py",
        _VACUOUS_SRC.replace(
            "    return bad\n",
            "    return bad  # VACUITY_LEDGER_OK:<rationale>\n",
        ),
    )
    found = check_verdict_surfaces_report_examined_count(
        repo_root=root, strict=False, verbose=False
    )
    assert any("fake_waiver.py" in v for v in found), found


def test_gate_is_registered_with_a_live_positive_control() -> None:
    """The sibling class guard EXECUTES controls; being registered is what makes
    this gate's detector a live assertion rather than a claim."""
    assert any(
        c.gate == "check_verdict_surfaces_report_examined_count"
        for c in POSITIVE_CONTROLS
    )


def test_gate_live_count_is_zero_on_the_real_repo() -> None:
    """STRICT-from-byte-one precondition. If this ever fails, a new bare verdict
    landed — fix it or waive it; do NOT relax the gate."""
    found = check_verdict_surfaces_report_examined_count(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert found == [], found


def test_gate_guards_the_preflight_cli_ledger_wire_in(tmp_path: Path) -> None:
    """Leg B: if the CLI's ScopeLedger wire-in is deleted, the gate must notice.

    Reproduces the #842 regression by writing a `_preflight_cli_main` with no
    ledger reference at the canonical path.
    """
    target = tmp_path / "src" / "tac" / "preflight.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "def _preflight_cli_main():\n"
        "    run()\n"
        "    print('PREFLIGHT PASSED')\n"
    )
    found = check_verdict_surfaces_report_examined_count(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("_preflight_cli_main" in v and "ScopeLedger" in v for v in found), found


# ---------------------------------------------------------------------------
# The preflight CLI — end to end
# ---------------------------------------------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tac.preflight", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.timeout(200)
def test_cli_refuses_an_empty_scope_instead_of_printing_passed() -> None:
    """The measured #842 anchor: this exact command printed a bare
    "PREFLIGHT PASSED" after running 0 of 27 declared gates in 0.52s."""
    proc = _cli("--no-codebase")
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 3, combined
    assert "PREFLIGHT VACUOUS" in combined, combined
    assert "PREFLIGHT PASSED" not in combined, combined
    assert "examined 0 of" in combined, combined


@pytest.mark.timeout(200)
def test_cli_acknowledged_empty_scope_still_reports_vacuous() -> None:
    proc = _cli("--no-codebase", "--acknowledge-empty-scope")
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "PREFLIGHT VACUOUS" in combined, combined
    assert "PREFLIGHT PASSED" not in combined, combined


def test_cli_passing_run_reports_its_denominator(monkeypatch, capsys) -> None:
    """The NON-vacuous branch, which no end-to-end run can currently reach.

    `--scope dev` is RED on this repo (4 pre-existing custody bypasses), so the
    subprocess tests above only ever exercise the VACUOUS path. That would leave
    the branch every green run takes untested — the shape of bug this whole batch
    is about. Driven in-process with a stub runner that actually invokes a
    recorder-wrapped gate, so `examined` is produced by the real instrument
    rather than asserted.
    """
    from tac import preflight as P

    calls: list[str] = []

    def check_stub_for_vacuity_test(*, strict: bool = False, verbose: bool = True):
        calls.append("ran")
        return []

    # Declare a one-gate scope and inject that gate into the preflight module so
    # the REAL timing recorder wraps it. `examined` is then produced by the same
    # instrument the CLI uses, not stubbed.
    monkeypatch.setitem(
        P.__dict__, "check_stub_for_vacuity_test", check_stub_for_vacuity_test
    )
    monkeypatch.setattr(
        P,
        "_called_preflight_cli_check_names",
        lambda fn: ["check_stub_for_vacuity_test"],
    )

    def fake_runner(**kwargs):
        # Resolve through the module dict so the recorder's wrapper is invoked.
        P.__dict__["check_stub_for_vacuity_test"](strict=False, verbose=False)

    fake_runner.__name__ = "preflight_developer"
    monkeypatch.setattr(P, "preflight_developer", fake_runner)
    monkeypatch.setattr(sys, "argv", ["tac.preflight", "--scope", "dev"])

    with pytest.raises(SystemExit) as exc:
        P._preflight_cli_main()

    assert exc.value.code == 0
    assert calls == ["ran"], "the stub gate must actually have been executed"
    out = capsys.readouterr().out
    assert "PREFLIGHT PASSED" in out, out
    assert "examined 1 of 1 declared" in out, out  # the denominator, not a bare word
    assert "VACUOUS" not in out, out
