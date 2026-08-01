# SPDX-License-Identifier: MIT
"""ddm_gh1 CLASS GUARD — a REFUSE-capable gate needs a POSITIVE CONTROL and a DECLARED DENOMINATOR.

THE CLASS (measured 2026-07-31, five instances in one day): a gate that can REFUSE is trusted
precisely because nobody re-derives it. When its detector is narrowed — by a prefilter, a glob, a
registry that enumerates part of its universe — it keeps printing OK over an almost-empty scan and
everyone reads that as "clean". Measured instances: the CLASS-2 process-guard gate skipped the
entire ``ps -axo command`` family at the FILE level while three live slot guards carried the bug
(#829); the raw-vm gate declared live-count 0 while measuring 6 and silently omitted
``experiments/`` + ``scripts/`` (#830); a lever registry AST'd 1 of 171 modules; a findings gate
scanned 0 of 1,260 files; a duty queue enumerated 116 of 177.

The decisive test below is a MUTATION test: it narrows a real gate the way all five instances were
narrowed and proves the class guard catches it. Without that, the class guard would itself be a
declaration rather than an assertion — the very failure it exists to extinct.
"""
from __future__ import annotations

import pytest

from tac import confound_gates as cg


def test_live_count_is_zero_and_strict_passes():
    assert cg.check_refusal_gates_have_live_positive_control(strict=True, verbose=False) == []


def test_every_registered_control_actually_fires():
    """Runs each control individually so a failure names the gate, not just the count."""
    by_name = {fn.__name__: fn for fn in cg.CONFOUND_GATES}
    for control in cg.POSITIVE_CONTROLS:
        assert control.gate in by_name, f"stale control for {control.gate}"


def test_declared_denominator_is_reported_not_just_a_count():
    coverage = cg.positive_control_coverage()
    assert coverage["covered"] >= cg.MIN_POSITIVE_CONTROL_COVERAGE
    assert coverage["total_refuse_capable_gates"] >= coverage["covered"]
    # The uncovered set must be NAMED — a tracked queue, never a silent default.
    assert isinstance(coverage["uncovered_gates"], list)
    assert (
        len(coverage["uncovered_gates"])
        == coverage["total_refuse_capable_gates"] - coverage["covered"]
    )


def test_mutation_a_gutted_detector_is_caught(monkeypatch):
    """THE DECISIVE TEST. Narrow a real gate exactly the way all five measured instances were
    narrowed — make it scan nothing — and the class guard must refuse."""
    original = cg.check_no_raw_virtual_memory_safety_basis

    def gutted(*, repo_root=None, strict=False, verbose=True):
        return cg._finish(
            name="check_no_raw_virtual_memory_safety_basis",
            tag="raw-vm-safety-basis",
            violations=[],           # a clean OK...
            strict=strict,
            verbose=False,
            ok_detail="0 source file(s) scanned",  # ...over an empty scan
        )

    gutted.__name__ = "check_no_raw_virtual_memory_safety_basis"  # same identity, empty scan
    monkeypatch.setattr(cg, "check_no_raw_virtual_memory_safety_basis", gutted)
    monkeypatch.setattr(
        cg,
        "CONFOUND_GATES",
        tuple(gutted if fn is original else fn for fn in cg.CONFOUND_GATES),
    )
    violations = cg.check_refusal_gates_have_live_positive_control(strict=False, verbose=False)
    assert any("POSITIVE CONTROL NO LONGER FIRES" in v for v in violations), violations
    assert any("check_no_raw_virtual_memory_safety_basis" in v for v in violations)


def test_mutation_coverage_regression_is_caught(monkeypatch):
    """A new REFUSE-capable gate landing with no control drops coverage below the ratchet floor."""

    def brand_new_gate(*, repo_root=None, strict=False, verbose=True):
        return []

    monkeypatch.setattr(cg, "MIN_POSITIVE_CONTROL_COVERAGE", cg.MIN_POSITIVE_CONTROL_COVERAGE + 1)
    monkeypatch.setattr(cg, "CONFOUND_GATES", (*cg.CONFOUND_GATES, brand_new_gate))
    violations = cg.check_refusal_gates_have_live_positive_control(strict=False, verbose=False)
    assert any("coverage REGRESSED" in v for v in violations), violations


def test_mutation_a_stale_control_is_caught(monkeypatch):
    """A control pointing at a gate that no longer exists must not read as a passing control."""
    stale = cg.PositiveControl(
        gate="check_gate_that_was_deleted",
        files={"tools/x.py": "pass\n"},
        must_mention="x.py",
        why="deleted gate",
    )
    monkeypatch.setattr(cg, "POSITIVE_CONTROLS", (*cg.POSITIVE_CONTROLS, stale))
    violations = cg.check_refusal_gates_have_live_positive_control(strict=False, verbose=False)
    assert any("unregistered gate" in v for v in violations), violations


def test_mutation_a_raising_gate_is_a_finding(monkeypatch):
    def exploding(*, repo_root=None, strict=False, verbose=True):
        raise RuntimeError("boom")

    exploding.__name__ = "check_levelset_hosc_requires_beta_end"
    monkeypatch.setattr(
        cg,
        "CONFOUND_GATES",
        tuple(
            exploding if fn.__name__ == "check_levelset_hosc_requires_beta_end" else fn
            for fn in cg.CONFOUND_GATES
        ),
    )
    violations = cg.check_refusal_gates_have_live_positive_control(strict=False, verbose=False)
    assert any("raised" in v for v in violations), violations


def test_class_guard_is_registered_and_strict():
    import inspect

    from tac import preflight

    assert cg.check_refusal_gates_have_live_positive_control in cg.CONFOUND_GATES
    source = inspect.getsource(preflight.preflight_all)
    assert '"check_refusal_gates_have_live_positive_control",' in source, (
        "class guard is not in the _CONFOUND_STRICT set"
    )


@pytest.mark.parametrize("control", cg.POSITIVE_CONTROLS, ids=lambda c: c.gate)
def test_each_control_carries_a_substantive_rationale(control):
    """A control without a stated reason rots into a fixture nobody dares change."""
    assert len(control.why) > 40
    assert control.files and control.must_mention


# ---------------------------------------------------------------------------
# DENOMINATOR-SIDE RATCHET (task #831, added 2026-07-31)
#
# The floor above is on the NUMERATOR (gates that HAVE controls), so it can only fire when a
# control is REMOVED. It is structurally blind to the case its own comment advertised: a new
# REFUSE-capable gate landing WITHOUT a control raises the denominator and leaves the numerator
# alone, so `covered < MIN` stays False and the guard prints OK.
#
# MEASURED: landing check_upstream_pin_no_content_drift took the catalog 23 -> 24 and the
# uncovered set 19 -> 20, and the guard emitted nothing. These two tests are the assertion that
# the closing leg is real rather than another comment.
# ---------------------------------------------------------------------------


def test_ceiling_fires_when_a_bare_refuse_gate_lands(monkeypatch):
    """THE case the numerator floor cannot see. Behaviour, not arithmetic restated."""

    def bare_gate(**_kwargs):
        return []

    bare_gate.__name__ = "check_simulated_new_bare_gate"
    monkeypatch.setattr(cg, "CONFOUND_GATES", (*cg.CONFOUND_GATES, bare_gate))

    violations = cg.check_refusal_gates_have_live_positive_control(strict=False, verbose=False)
    grew = [v for v in violations if "uncovered REFUSE-capable gates GREW" in v]
    assert grew, f"a bare REFUSE-capable gate landed silently: {violations}"
    # The message must NAME the gate, or the reader cannot act on the refusal.
    assert "check_simulated_new_bare_gate" in grew[0]

    # And the NUMERATOR floor must be silent here — proving the two legs are genuinely
    # independent and that this case was previously unreachable.
    assert not [v for v in violations if "coverage REGRESSED" in v]


def test_ceiling_is_tight_against_the_live_uncovered_count():
    """A ratchet slack by even one admits a free bare gate; slack the other way is red-by-default."""
    live = len(cg.positive_control_coverage()["uncovered_gates"])
    assert cg.MAX_UNCOVERED_REFUSE_GATES == live, (
        f"ceiling {cg.MAX_UNCOVERED_REFUSE_GATES} != live uncovered {live}. Lower it when a "
        f"control lands; NEVER raise it to admit a bare gate."
    )
