# SPDX-License-Identifier: MIT
"""BEHAVIOUR tests for the build-completeness axis + the DESIGNED-STUB refusal gate.

ddm_sb2 (task #819). Every test asserts a PROPERTY of the derivation on real repo inputs —
none asserts that a canonical constant equals itself (NO-FAKE forbidden class #2). The
mutation guard at the bottom states explicitly what a marker-returning stub could not do.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.confound_gates import CONFOUND_GATES, check_no_stub_lever_factories
from tac.witness_dsl.activation_ledger import (
    BUILD_DESIGNED_STUB,
    BUILD_GRADE_ORDER,
    BUILD_FIRED,
    BUILD_NEVER_FIRED,
    BUILD_NOT_DESIGNED,
    VALID_BUILD_GRADES,
    build_completeness_report,
    build_grade,
    built_elsewhere_unwired,
    known_levers,
    not_even_designed,
    package_known_levers,
    read_required_components,
    record_required_component,
)
from tac.witness_dsl.lever_registry import (
    build_completeness,
    lever_factories,
    module_trainer_paths,
    package_lever_factories,
    package_lever_modules,
)


# ── the vacuous-scan repair ──────────────────────────────────────────────────────────
def test_package_scan_is_strictly_wider_than_the_single_file_scan() -> None:
    """The repaired scan sees MORE modules and MORE factories than the historical one.

    The measured bug: ``lever_factories()`` ASTs only ``curriculum_dsl.py``, so ~170 sibling
    modules were invisible and stub detection was structurally impossible.
    """
    old = set(lever_factories())
    new = {f.factory for f in package_lever_factories()}
    assert old < new, "package scan must be a strict superset of the curriculum_dsl-only scan"
    assert len(new) > len(old) + 40
    assert len({f.module for f in package_lever_factories()}) > 1
    assert len(package_lever_modules()) > 100


def test_known_levers_is_the_package_wide_universe() -> None:
    """``known_levers()`` IS the package surface — the contract ddm_rg5 (#825) changed.

    ddm_sb2 (#819) landed ``package_known_levers`` BESIDE ``known_levers`` and preserved the
    single-module default "so the historical contract is unchanged". Nothing opted in: the honest
    superset had exactly one grep hit outside its own definition, and that hit was a docstring.
    Meanwhile ``never_fired()`` and ``duty_to_measure()`` — the duty queue itself — both defaulted
    to the narrow set and enumerated 116 of 179 factories, blind to 9 of the 10 designed-stubs the
    sister gate reports. A correct surface nobody consumes is not a repair, so the DEFAULT moved.

    The narrow view survives under an intention-revealing name for the one legitimate question it
    answers, and this test pins BOTH halves: default is package-wide, narrow is a strict subset.
    """
    from tac.witness_dsl.activation_ledger import curriculum_dsl_known_levers

    assert set(known_levers()) == set(package_known_levers())
    assert set(curriculum_dsl_known_levers()) < set(known_levers())
    assert set(curriculum_dsl_known_levers()) == set(lever_factories())


def test_duty_queue_sees_the_designed_stubs() -> None:
    """The orphan tracker must be able to see the orphans its sister gate reports.

    This is the property whose ABSENCE was the bug: every designed-stub is, by construction, a
    lever that has never fired, so it must appear in ``never_fired()``. Under the single-module
    default, 9 of 10 did not — the NO-FAKE forbidden-class-#1 surface was invisible to the very
    queue that exists to surface it, on the day the rule naming that class was written.
    """
    from tac.witness_dsl.activation_ledger import (
        BUILD_DESIGNED_STUB,
        build_completeness_report,
        duty_to_measure,
        never_fired,
    )

    stubs = {r["component"] for r in build_completeness_report()
             if r["grade"] == BUILD_DESIGNED_STUB}
    assert stubs, "fixture guard: expected at least one designed-stub in the live registry"
    missing = stubs - set(never_fired())
    assert not missing, f"designed-stubs invisible to the orphan list: {sorted(missing)}"
    assert not stubs - set(duty_to_measure()), "designed-stubs missing from the duty queue"


def test_trainer_resolution_is_per_module_not_global() -> None:
    """A module declaring its own TRAINER_RELPATH binds to THAT trainer.

    This is what keeps the widened scan honest: without it every tr1 flag would be reported
    missing against the levelset trainer — a false FAIL replacing a vacuous PASS.
    """
    mods = {p.name: p for p in package_lever_modules()}
    tr1 = module_trainer_paths(mods["spec_tr1_renderer_20260728.py"])
    other = module_trainer_paths(mods["curriculum_dsl.py"])
    assert len(tr1) == 1 and tr1[0].name == "train_tr1_partition_renderer_mlx.py"
    assert tr1[0] not in other
    assert all(p.is_file() for p in tr1 + other)


def test_tr1_levers_are_not_falsely_reported_as_stubs() -> None:
    """Regression for the false-FAIL failure mode: tr1 factories resolve against tr1."""
    tr1 = [f for f in package_lever_factories()
           if f.module == "spec_tr1_renderer_20260728.py" and f.flags]
    assert len(tr1) > 15
    assert all(not f.is_stub for f in tr1), (
        "tr1 lever flags exist on the tr1 trainer; reporting them as stubs would mean the "
        "scan is resolving against the wrong trainer")


# ── the grade derivation ─────────────────────────────────────────────────────────────
def test_stub_judgement_is_structural_not_label_based() -> None:
    """A factory is a stub iff its flag is missing — independent of whether it SAYS so."""
    bc = build_completeness()
    assert bc.stubs, "the repo currently carries known DESIGNED-STUB levers"
    for f in bc.stubs:
        assert f.missing_flags, "is_stub must be driven by missing flags, not by the marker"
    # and the class the label cannot catch: stubs that never announce themselves
    silent = bc.silent_stubs
    assert all(not f.stub_marker and f.is_stub for f in silent)
    assert len(silent) >= 1, "the silent-stub class is real in this repo and must be visible"


def test_known_stub_modules_are_detected() -> None:
    """The five fh1 adapted forces — the levers the old registry literally could not see."""
    stubs = {f.factory for f in build_completeness().stubs}
    for name in ("TieLocusEdgeWeighted", "MarginSatisficeCap", "XiAdvectedTokenBase",
                 "BirthPlateauKneeConjunct", "ErfBirthContextCoadapt"):
        assert name in stubs, f"{name} is a known stub and must be graded as one"


def test_build_grade_partitions_every_factory() -> None:
    rows = build_completeness_report()
    assert rows
    assert all(r["grade"] in VALID_BUILD_GRADES for r in rows)
    grades = {r["grade"] for r in rows}
    assert BUILD_DESIGNED_STUB in grades and BUILD_NEVER_FIRED in grades


def test_build_grade_distinguishes_hollow_from_merely_off() -> None:
    """THE bug: a stub and a real default-off lever must NOT report identically.

    Both are ``never-fired`` in the activation ledger; only the build axis separates them.
    """
    idx = {f.factory: f for f in package_lever_factories()}
    stub = next(f.factory for f in idx.values() if f.is_stub)
    real = next(f.factory for f in idx.values() if not f.is_stub)
    assert build_grade(stub) == BUILD_DESIGNED_STUB
    assert build_grade(real) in (BUILD_NEVER_FIRED, BUILD_FIRED)
    assert build_grade(stub) != build_grade(real)


def test_unknown_component_is_not_even_designed() -> None:
    assert build_grade("NoSuchLeverAnywhere") == BUILD_NOT_DESIGNED


def test_report_orders_worst_grade_first() -> None:
    """Debt is read first, never as a footnote under a wall of built levers."""
    rows = build_completeness_report()
    # Use the module's OWN order map: a hand-copied duplicate here went stale the moment a 5th
    # grade landed, which is the drift class this assertion exists to catch elsewhere.
    keys = [BUILD_GRADE_ORDER[r["grade"]] for r in rows]
    assert keys == sorted(keys)
    assert set(BUILD_GRADE_ORDER) == set(VALID_BUILD_GRADES)


# ── the required-component (grade 4) store ───────────────────────────────────────────
def _rec(tmp_path, **kw):
    base = {"component": "X", "needed_by": "path_a",
            "missing_mechanism": "no injection point", "owner": "ddm_sb2",
            "fire_order": 1, "consumer": "tr1 trainer"}
    base.update(kw)
    return record_required_component(path=tmp_path / "req.jsonl", **base)


def test_required_component_round_trips(tmp_path) -> None:
    _rec(tmp_path, component="ResetPriorProvider")
    rows = read_required_components(tmp_path / "req.jsonl")
    assert len(rows) == 1
    assert rows[0]["component"] == "ResetPriorProvider"
    assert rows[0]["grade"] == BUILD_NOT_DESIGNED
    assert rows[0]["owner"] == "ddm_sb2"


def test_required_component_refuses_hollow_charters(tmp_path) -> None:
    """A charter missing owner / mechanism / consumer is the same orphan in a new coat."""
    for bad in ({"owner": ""}, {"missing_mechanism": "x"}, {"consumer": "  "},
                {"needed_by": ""}):
        with pytest.raises(ValueError, match=r"substantive|required"):
            _rec(tmp_path, **bad)
    with pytest.raises(ValueError, match="fire_order"):
        _rec(tmp_path, fire_order=-1)
    with pytest.raises(ValueError, match="invalid grade"):
        _rec(tmp_path, grade="sort-of-built")


def test_declared_component_drops_off_when_a_factory_exists(tmp_path) -> None:
    """The REGISTRY drains this queue, not a human editing a memo.

    Declaring a component that already has a factory yields no live debt — so the only way
    to clear a real row is to BUILD it.
    """
    real = next(iter(package_known_levers()))
    _rec(tmp_path, component=real)
    _rec(tmp_path, component="StillMissingThing")
    live = {r["component"] for r in not_even_designed(tmp_path / "req.jsonl")}
    assert "StillMissingThing" in live
    assert real not in live


def test_latest_row_wins_per_component(tmp_path) -> None:
    _rec(tmp_path, component="Y", owner="first", fire_order=5)
    _rec(tmp_path, component="Y", owner="second", fire_order=5)
    rows = read_required_components(tmp_path / "req.jsonl")
    assert len(rows) == 1 and rows[0]["owner"] == "second"


# ── the refusal gate ─────────────────────────────────────────────────────────────────
def test_gate_is_wired_into_confound_gates() -> None:
    assert check_no_stub_lever_factories in CONFOUND_GATES


def test_gate_reports_every_stub_and_names_the_silent_ones() -> None:
    v = check_no_stub_lever_factories(strict=False, verbose=False)
    bc = build_completeness()
    assert len(v) == len(bc.stubs) + len([f for f in bc.label_drift if not f.is_stub])
    assert any("SILENT" in s for s in v), "silent stubs must be called out as the worst grade"
    for f in bc.stubs:
        assert any(f.factory in s for s in v)


def test_gate_raises_in_strict_mode_while_debt_is_live() -> None:
    from tac.preflight import PreflightError

    with pytest.raises(PreflightError):
        check_no_stub_lever_factories(strict=True, verbose=False)


def test_gate_waiver_is_honoured_and_placeholders_do_not_self_waive(tmp_path) -> None:
    """A real rationale waives; the docstring's own ``<rationale>`` example does not."""
    from tac.confound_gates import _factory_waived

    pkg = tmp_path / "src" / "tac" / "witness_dsl"
    pkg.mkdir(parents=True)
    p = pkg / "m.py"
    p.write_text(
        "def Waived() -> Lever:  # DESIGNED_STUB_OK: owned by ddm_tp1, fires burn-5\n"
        "    return Lever('a', overrides={'--x': True})\n"
        "def Placeholder() -> Lever:  # DESIGNED_STUB_OK:<rationale>\n"
        "    return Lever('b', overrides={'--y': True})\n"
        "def Bare() -> Lever:\n"
        "    return Lever('c', overrides={'--z': True})\n"
    )
    assert _factory_waived(p, "Waived") is True
    assert _factory_waived(p, "Placeholder") is False
    assert _factory_waived(p, "Bare") is False


def test_gate_ok_detail_counts_real_mechanisms() -> None:
    v = check_no_stub_lever_factories(strict=False, verbose=False)
    bc = build_completeness()
    assert len(v) > 0
    assert bc.total - len(bc.stubs) > 100  # most levers ARE real; the gate is not crying wolf


# ── mutation guard ───────────────────────────────────────────────────────────────────
def test_mutation_guard_stub_body_would_fail() -> None:
    """Properties a marker-returning stub could not satisfy.

    (1) the scan's factory set depends on files on disk; (2) grades DIFFER across components;
    (3) the gate's violation list names the actual offending factories and flags.
    """
    facs = package_lever_factories()
    assert len({f.module for f in facs}) >= 10
    assert len({f.factory for f in facs}) == len(facs)
    grades = {build_grade(f.factory) for f in facs}
    assert len(grades) >= 2
    v = check_no_stub_lever_factories(strict=False, verbose=False)
    assert any("--tie-locus-edge-weight" in s for s in v), (
        "the gate must quote the REAL missing flag, not a canned message")
    assert np.isfinite(len(v))


# ── the CONSUMER half of the same class (ddm_rg5 #825) ───────────────────────────────
# The stub gate above refuses a lever that PRESENTS as built with no mechanism. Its sister
# refuses a CONSUMER that reads a PARTIAL registry and so presents a partial universe as
# complete. Live count 0 at landing, hence STRICT from byte one.
def test_legacy_surface_gate_is_wired_and_strict_clean() -> None:
    from tac.confound_gates import check_no_legacy_single_module_lever_surface_consumers as g

    assert g in CONFOUND_GATES
    assert g(strict=True, verbose=False) == []


def test_legacy_surface_gate_flags_a_real_binding(tmp_path) -> None:
    src = tmp_path / "tools"
    src.mkdir(parents=True)
    (src / "ranker.py").write_text(
        "from tac.witness_dsl.lever_registry import lever_factories\n"
        "def rank():\n    return sorted(lever_factories())\n",
        encoding="utf-8")
    from tac.confound_gates import check_no_legacy_single_module_lever_surface_consumers as g

    from tac.preflight import PreflightError

    v = g(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1 and "tools/ranker.py" in v[0]
    with pytest.raises(PreflightError):
        g(repo_root=tmp_path, strict=True, verbose=False)


def test_legacy_surface_gate_ignores_prose_and_honours_waiver(tmp_path) -> None:
    """AST, not regex: the two live TEXT hits at landing were a docstring and an f-string.

    ``tools/register_ema_finisher_duty.py`` mentions ``lever_factories()`` in its module
    docstring and inside an error message. A regex gate reports both, the reader learns the gate
    cries wolf, and the next real binding slides past — the exact dynamic this task exists to
    break. Only an actual call may be refused.
    """
    src = tmp_path / "tools"
    src.mkdir(parents=True)
    (src / "prose.py").write_text(
        '"""Docs mentioning lever_factories() in prose."""\n'
        'MSG = "not discovered by lever_factories(); build it"\n'
        "# comment about lever_factories()\n",
        encoding="utf-8")
    (src / "waived.py").write_text(
        "from tac.witness_dsl.lever_registry import lever_factories\n"
        "def only_curriculum_dsl():\n"
        "    return lever_factories()  # SINGLE_MODULE_LEVER_SURFACE_OK: this reports the "
        "levelset trainer's own module coverage, not the campaign lever universe\n",
        encoding="utf-8")
    from tac.confound_gates import check_no_legacy_single_module_lever_surface_consumers as g

    assert g(repo_root=tmp_path, strict=True, verbose=False) == []


def test_legacy_surface_gate_rejects_placeholder_waiver(tmp_path) -> None:
    src = tmp_path / "tools"
    src.mkdir(parents=True)
    (src / "fake.py").write_text(
        "from tac.witness_dsl.lever_registry import lever_factories\n"
        "def q():\n    return lever_factories()  # SINGLE_MODULE_LEVER_SURFACE_OK: <rationale>\n",
        encoding="utf-8")
    from tac.confound_gates import check_no_legacy_single_module_lever_surface_consumers as g

    assert len(g(repo_root=tmp_path, strict=False, verbose=False)) == 1


# ---------------------------------------------------------------------------
# BUILD_RETIRED — the BUILD axis's dormant-with-reactivation exit (ddm_wr2, #864).
# The ACTIVATION axis has had STATE_RETIRED since the ledger landed; the BUILD axis
# shipped without it, so a wire-or-retire adjudication could not record "retire".
# ---------------------------------------------------------------------------


def test_retired_is_a_valid_build_grade_and_distinct_from_the_other_four() -> None:
    from tac.witness_dsl.activation_ledger import BUILD_RETIRED

    assert BUILD_RETIRED in VALID_BUILD_GRADES
    assert BUILD_RETIRED not in (
        BUILD_FIRED, BUILD_NEVER_FIRED, BUILD_DESIGNED_STUB, BUILD_NOT_DESIGNED)
    # Count pin retained deliberately (ddm_wt1): it forces anyone adding a grade to update this
    # test consciously, which is how the 5th grade's arrival was caught. Bumped 5 -> 6 for
    # built-elsewhere-unwired. Distinctness is asserted separately -- a duplicated string would
    # keep the count right while silently collapsing two grades.
    assert len(VALID_BUILD_GRADES) == 6
    assert len(set(VALID_BUILD_GRADES)) == len(VALID_BUILD_GRADES), "grades must be distinct"


def test_retired_row_leaves_the_not_even_designed_debt_queue(tmp_path) -> None:
    """The whole point: a retired component stops nagging as build debt."""
    from tac.witness_dsl.activation_ledger import BUILD_RETIRED

    p = tmp_path / "req.jsonl"
    record_required_component(
        "Wr2StillOwed", needed_by="cfgA", missing_mechanism="no recipient exists yet",
        owner="someone", fire_order=1, consumer="trainer", path=p)
    record_required_component(
        "Wr2Retired", needed_by="cfgA", missing_mechanism="recipient cannot exist on tr1",
        owner="ddm_wr2", fire_order=4, consumer="n/a", grade=BUILD_RETIRED,
        notes="REACTIVATION: re-open iff tr1 grows the precondition it needs", path=p)

    live = {r["component"] for r in not_even_designed(p)}
    assert "Wr2StillOwed" in live
    assert "Wr2Retired" not in live
    # ...but it is NOT deleted: retirement is dormant-with-reactivation, never a kill.
    assert "Wr2Retired" in {r["component"] for r in read_required_components(p)}


def test_hand_appended_unverified_retirement_cannot_drain_the_debt_queue(tmp_path) -> None:
    """A retirement that does not re-pass the write-path predicate keeps nagging.

    #899 residual (ddm_qd1, 2026-08-03). ``BUILD_RETIRED`` is the only exit from
    ``not_even_designed`` that is a true DRAIN -- a retired row leaves this queue and appears in
    no other one, unlike a grade-5 row which stays visible in ``built_elsewhere_unwired``.

    The sibling test above plants a VALID retirement written through the write path. Nothing
    planted an INVALID one, so this was uncovered: the queue filtered on ``grade`` alone, and a
    row hand-appended to the JSONL with its mandatory reactivation trigger MISSING -- exactly the
    shape ``record_required_component`` refuses on write, asserted by the test below -- still
    exited. Debt could be drained by appending a line the writer would have rejected.
    """
    from tac.witness_dsl.activation_ledger import (
        BUILD_RETIRED,
        RECORD_DECLARED_UNVERIFIED,
        RECORD_VERIFIED,
    )

    p = tmp_path / "req.jsonl"
    record_required_component(
        "QdValidRetired", needed_by="cfgA", missing_mechanism="recipient cannot exist",
        owner="ddm_qd1", fire_order=1, consumer="n/a", grade=BUILD_RETIRED,
        notes="REACTIVATION: re-open iff the recipient lands", path=p)
    # Bypass the writer, exactly as a human editing the ledger by hand would.
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "component": "QdUnverifiedRetired", "needed_by": "cfgA",
            "grade": BUILD_RETIRED, "declared_by": "hand", "declared_at_utc": "2026-08-03T00:00:00Z",
        }) + "\n")

    integrity = {r["component"]: r.get("record_integrity") for r in read_required_components(p)}
    assert integrity["QdValidRetired"] == RECORD_VERIFIED
    assert integrity["QdUnverifiedRetired"] == RECORD_DECLARED_UNVERIFIED

    live = {r["component"] for r in not_even_designed(p)}
    assert "QdValidRetired" not in live, "an earned retirement must still drain"
    assert "QdUnverifiedRetired" in live, (
        "an unverified retirement must NOT drain: it exits every queue, so its exit has to be "
        "re-earned at read time"
    )


def test_unverified_grade5_row_stays_visible_rather_than_being_gated_twice(tmp_path) -> None:
    """The grade-5 exit stays unconditional ON PURPOSE -- it is a hand-off, not a drain.

    Guards against over-correcting the fix above. An unverified grade-5 row leaves
    ``not_even_designed`` but remains visible in ``built_elsewhere_unwired`` (sorted last and
    typed by ``record_integrity``), so gating that exit too would hide the row twice instead of
    once. Dropping rows is signal loss; demoting and typing them is not.
    """
    from tac.witness_dsl.activation_ledger import (
        BUILD_ELSEWHERE_UNWIRED,
        built_elsewhere_unwired,
    )

    p = tmp_path / "req.jsonl"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "component": "QdUnverifiedGrade5", "needed_by": "cfgA",
            "grade": BUILD_ELSEWHERE_UNWIRED, "declared_by": "hand",
            "declared_at_utc": "2026-08-03T00:00:00Z",
        }) + "\n")

    assert "QdUnverifiedGrade5" not in {r["component"] for r in not_even_designed(p)}
    still_visible = {r["component"] for r in built_elsewhere_unwired(p)}
    assert "QdUnverifiedGrade5" in still_visible, (
        "an unverified grade-5 row must remain readable somewhere; it is demoted, never dropped"
    )


def test_retirement_refuses_without_a_reactivation_trigger(tmp_path) -> None:
    from tac.witness_dsl.activation_ledger import BUILD_RETIRED

    p = tmp_path / "req.jsonl"
    for bad_notes in ("", "   ", "x"):
        with pytest.raises(ValueError, match="REACTIVATION TRIGGER"):
            record_required_component(
                "Wr2NoTrigger", needed_by="cfgA", missing_mechanism="recipient absent",
                owner="ddm_wr2", fire_order=1, consumer="none on the live vehicle",
                grade=BUILD_RETIRED, notes=bad_notes, path=p)
    assert read_required_components(p) == []


def test_report_keeps_the_retired_grade_instead_of_reasserting_the_debt(tmp_path) -> None:
    """A retired row reported as not-even-designed silently becomes a build order again."""
    from tac.witness_dsl.activation_ledger import BUILD_RETIRED

    p = tmp_path / "req.jsonl"
    record_required_component(
        "Wr2ReportRetired", needed_by="cfgA", missing_mechanism="absent by construction",
        owner="ddm_wr2", fire_order=4, consumer="n/a", grade=BUILD_RETIRED,
        notes="REACTIVATION: a length/MCF term is added to the live loss", path=p)
    rows = {r["component"]: r for r in build_completeness_report(p)}
    assert rows["Wr2ReportRetired"]["grade"] == BUILD_RETIRED
    assert "REACTIVATION" in (rows["Wr2ReportRetired"]["notes"] or "")
    # retired sorts AFTER every real debt grade — debt stays the first thing read
    grades = [r["grade"] for r in build_completeness_report(p)]
    assert grades.index(BUILD_RETIRED) == len(grades) - 1


def test_mutation_guard_retire_path_actually_branches(tmp_path) -> None:
    """Would fail if BUILD_RETIRED were accepted but treated identically to NOT_DESIGNED."""
    from tac.witness_dsl.activation_ledger import BUILD_RETIRED

    p = tmp_path / "req.jsonl"
    kw = dict(needed_by="cfgA", missing_mechanism="recipient absent", owner="ddm_wr2",
              fire_order=2, consumer="none on the live vehicle", path=p)
    record_required_component("WrTwinOpen", **kw)
    record_required_component("WrTwinShut", grade=BUILD_RETIRED,
                              notes="REACTIVATION: named measured fact", **kw)
    rep = {r["component"]: r["grade"] for r in build_completeness_report(p)}
    # Same charter fields, different grade => the two rows MUST diverge on both surfaces.
    assert rep["WrTwinOpen"] != rep["WrTwinShut"]
    assert len(not_even_designed(p)) == 1


def test_retirement_is_per_charter_key_not_per_component(tmp_path) -> None:
    """The (component, needed_by) key is deliberate: one component can be required by several
    configs. Retiring it for ONE charter must not silently retire the others.

    Regression: ddm_wr2's first pass recorded retirements under a NEW ``needed_by`` and so did
    not supersede the open charters at all -- the queue looked adjudicated and was not.
    """
    from tac.witness_dsl.activation_ledger import BUILD_RETIRED

    p = tmp_path / "req.jsonl"
    common = dict(missing_mechanism="recipient absent on the live vehicle", owner="ddm_wr2",
                  fire_order=1, consumer="none on the live vehicle", path=p)
    record_required_component("WrShared", needed_by="cfgA", **common)
    record_required_component("WrShared", needed_by="cfgB", **common)
    assert len(not_even_designed(p)) == 2

    # Retiring under a THIRD, unrelated key must drain neither.
    record_required_component("WrShared", needed_by="cfgC", grade=BUILD_RETIRED,
                              notes="REACTIVATION: named measured fact", **common)
    assert len(not_even_designed(p)) == 2, "a new key must not supersede an existing charter"

    # Retiring under cfgA's own key drains exactly cfgA.
    record_required_component("WrShared", needed_by="cfgA", grade=BUILD_RETIRED,
                              notes="REACTIVATION: named measured fact", **common)
    remaining = [r["needed_by"] for r in not_even_designed(p)]
    assert remaining == ["cfgB"]


# ---------------------------------------------------------------------------
# BUILD_ELSEWHERE_UNWIRED — the #864 P0 grade (ddm_wt1).
# Operator 2026-08-01: "All built-elsewhere-unwired is p0". The axis could express
# "missing on this vehicle" five ways and could not express "missing NOTHING, wired
# NOWHERE, and beating what is live". Every test below asserts a PROPERTY of the
# refusal/queue behaviour on real inputs, never that a constant equals itself.
# ---------------------------------------------------------------------------


def _harm_kwargs(**over):
    base = {
        "needed_by": "cfgOptimalFromStart",
        "missing_mechanism": "no live call site reaches it from the vehicle entry point",
        "owner": "ddm_wt1",
        "fire_order": 1,
        "consumer": "the live receiver",
        "grade": "built-elsewhere-unwired",
        "live_recipient": "live_incumbent_v1",
        "measured_comparison": "challenger 1.0 vs incumbent 2.0 B [macOS-CPU advisory]",
        # The prose above is a CITATION; these three are the checkable fact (ddm_wd2, #861).
        "live_measured": 2.0,
        "candidate_measured": 1.0,
        "metric_direction": "lower-is-better",
    }
    base.update(over)
    return base


def test_elsewhere_unwired_is_distinct_from_never_fired() -> None:
    """The distinction the P0 is ABOUT: never-fired is dormant, this one is losing score.

    Collapsing them was the second structural reason the class stayed invisible -- not only
    "no detector" but "no vocabulary".
    """
    from tac.witness_dsl.activation_ledger import BUILD_ELSEWHERE_UNWIRED, BUILD_RETIRED

    assert BUILD_ELSEWHERE_UNWIRED in VALID_BUILD_GRADES
    assert BUILD_ELSEWHERE_UNWIRED not in (
        BUILD_FIRED, BUILD_NEVER_FIRED, BUILD_DESIGNED_STUB, BUILD_NOT_DESIGNED, BUILD_RETIRED)


def test_grade_order_covers_every_grade_and_ranks_measured_harm_first() -> None:
    """The order map must stay total, and the only grade carrying a MEASURED present loss reads first."""
    from tac.witness_dsl.activation_ledger import BUILD_ELSEWHERE_UNWIRED

    assert set(BUILD_GRADE_ORDER) == set(VALID_BUILD_GRADES)
    assert len(set(BUILD_GRADE_ORDER.values())) == len(VALID_BUILD_GRADES), "ranks must be distinct"
    assert BUILD_GRADE_ORDER[BUILD_ELSEWHERE_UNWIRED] == min(BUILD_GRADE_ORDER.values())


def test_harm_clause_refuses_a_recipient_less_row(tmp_path) -> None:
    """ddm_wr2 MEASURED 4-of-7 meeting the literal predicate and 0-of-7 meeting the harm clause.

    Without this refusal the grade re-admits those 7 and an adjudicated NEGATIVE-CONTROL set turns
    back into a wiring backlog -- the exact error wr2 spent an arm correcting.
    """
    p = tmp_path / "req.jsonl"
    for missing in ("live_recipient", "measured_comparison"):
        kwargs = _harm_kwargs(**{missing: ""})
        with pytest.raises(ValueError, match="HARM CLAUSE"):
            record_required_component("NoRecipient", path=p, **kwargs)
    # ...and both absent at once is likewise refused, not silently defaulted.
    with pytest.raises(ValueError, match="HARM CLAUSE"):
        record_required_component(
            "NoRecipient", path=p, **_harm_kwargs(live_recipient="", measured_comparison=""))
    assert not p.exists() or p.read_text(encoding="utf-8").strip() == "", "refused rows must not persist"


def test_harm_clause_refusal_names_the_right_alternative(tmp_path) -> None:
    """A refusal that does not say what to record instead just teaches callers to pick any grade."""
    p = tmp_path / "req.jsonl"
    with pytest.raises(ValueError) as exc:
        record_required_component("NoRecipient", path=p, **_harm_kwargs(live_recipient=""))
    assert "built-never-fired" in str(exc.value)


def test_elsewhere_unwired_leaves_the_build_queue_but_enters_the_wiring_queue(tmp_path) -> None:
    """It is BUILT, so "not even designed" is false about it; its debt is WIRING, not building."""
    from tac.witness_dsl.activation_ledger import built_elsewhere_unwired

    p = tmp_path / "req.jsonl"
    record_required_component("WiringDebt", path=p, **_harm_kwargs())
    record_required_component(
        "BuildDebt", needed_by="cfgOptimalFromStart", missing_mechanism="nothing implements it",
        owner="ddm_wt1", fire_order=2, consumer="the live receiver", path=p)

    assert {r["component"] for r in not_even_designed(p)} == {"BuildDebt"}
    assert {r["component"] for r in built_elsewhere_unwired(p)} == {"WiringDebt"}
    # Both remain declared: neither queue deletes, and the row keeps its evidence.
    row = next(r for r in read_required_components(p) if r["component"] == "WiringDebt")
    assert row["live_recipient"] == "live_incumbent_v1"
    assert "incumbent" in row["measured_comparison"]


def test_report_preserves_the_grade_instead_of_coercing_to_not_designed(tmp_path) -> None:
    """Coercion would mis-state a measured WIRING debt as a BUILD debt and drop its harm."""
    from tac.witness_dsl.activation_ledger import BUILD_ELSEWHERE_UNWIRED

    p = tmp_path / "req.jsonl"
    record_required_component("WiringDebt", path=p, **_harm_kwargs())
    row = next(r for r in build_completeness_report(p) if r["component"] == "WiringDebt")
    assert row["grade"] == BUILD_ELSEWHERE_UNWIRED
    assert row["live_recipient"] == "live_incumbent_v1"
    # ...and it sorts to the head of the operator-facing report.
    assert build_completeness_report(p)[0]["component"] == "WiringDebt"


def test_wiring_queue_drains_when_a_factory_lands_not_when_a_memo_says_so(tmp_path) -> None:
    """Same contract as the grade-4 queue: the registry decides, never a human edit."""
    from tac.witness_dsl.activation_ledger import built_elsewhere_unwired

    p = tmp_path / "req.jsonl"
    real = sorted(package_known_levers())[0]  # a component that DOES have a factory
    record_required_component(real, path=p, **_harm_kwargs())
    assert real not in {r["component"] for r in built_elsewhere_unwired(p)}


def test_harm_clause_does_not_leak_onto_the_other_grades(tmp_path) -> None:
    """Grades 1-5 must keep working with no recipient -- the refusal is scoped, not global."""
    from tac.witness_dsl.activation_ledger import BUILD_RETIRED

    p = tmp_path / "req.jsonl"
    common = {"needed_by": "cfgOptimalFromStart", "missing_mechanism": "nothing implements it",
              "owner": "ddm_wt1", "fire_order": 1, "consumer": "the live receiver", "path": p}
    record_required_component("PlainGrade4", **common)
    record_required_component("Retired", grade=BUILD_RETIRED,
                              notes="REACTIVATION: iff tr1 grows the precondition", **common)
    assert len(read_required_components(p)) == 2


# ---------------------------------------------------------------------------
# THE HARM CLAUSE IS SIGNED (ddm_wd2, #864/#861).
# The clause above checked that a comparison was PRESENT; nothing checked which way it
# pointed, while the refusal text promised "showing the component BEATS it" and
# BUILD_GRADE_ORDER ranked the grade 0 on the strength of that promise. These tests assert
# the DIRECTION is now decided from the numbers, on real inputs, not from prose.
# ---------------------------------------------------------------------------


def test_harm_clause_refuses_the_founding_case_a_measured_worse_predecessor(tmp_path) -> None:
    """THE founding case: the P0's own headline instance, which the length-only clause admitted.

    ``p0_864`` records the pose pair as "~39x better, ~38x cheaper, RACED". ddm_wd1 then MEASURED
    that family plateauing at d_pose ~29-30 against a live realized 0.00858133 -- ~3,400x WORSE.
    A detector that cannot refuse its own founding case is not a detector, so this uses the real
    measured numbers rather than a toy pair.
    """
    p = tmp_path / "req.jsonl"
    with pytest.raises(ValueError, match="does not beat"):
        record_required_component("PoseBasisSwap", path=p, **_harm_kwargs(
            live_recipient="warp-pose6 via pfs1_warp_receiver (live in inflate_runner_v4d)",
            measured_comparison="eg1 cosine family plateaus d_pose ~29-30 vs live realized "
                                "0.00858133 [macOS-CPU advisory] (ddm_wd1)",
            live_measured=0.00858133, candidate_measured=29.0,
            metric_direction="lower-is-better"))
    assert not p.exists() or p.read_text(encoding="utf-8").strip() == "", "refused rows must not persist"


def test_harm_clause_accepts_the_same_pair_when_the_candidate_actually_wins(tmp_path) -> None:
    """The negative control for the test above: identical shape, direction reversed => ACCEPTED.

    Without this pair the refusal could be an unconditional reject and every test would still pass.
    """
    p = tmp_path / "req.jsonl"
    row = record_required_component("RealWinner", path=p, **_harm_kwargs(
        live_measured=29.0, candidate_measured=0.00858133, metric_direction="lower-is-better"))
    assert row["harm_advantage"] == pytest.approx(29.0 / 0.00858133)
    assert {r["component"] for r in built_elsewhere_unwired(p)} == {"RealWinner"}


def test_harm_clause_is_directional_both_ways(tmp_path) -> None:
    """higher-is-better inverts the comparison; a metric direction that is absent is undecidable."""
    p = tmp_path / "req.jsonl"
    # Under higher-is-better the SAME numbers that pass lower-is-better must now fail.
    with pytest.raises(ValueError, match="does not beat"):
        record_required_component("HigherLoser", path=p, **_harm_kwargs(
            live_measured=2.0, candidate_measured=1.0, metric_direction="higher-is-better"))
    row = record_required_component("HigherWinner", path=p, **_harm_kwargs(
        live_measured=1.0, candidate_measured=2.0, metric_direction="higher-is-better"))
    assert row["harm_advantage"] == pytest.approx(2.0)
    for bad in ("", "lower", "LOWER-IS-BETTER", None):
        with pytest.raises(ValueError, match="metric_direction"):
            record_required_component("BadDir", path=p, **_harm_kwargs(metric_direction=bad))


def test_harm_clause_refuses_a_tie_because_a_tie_is_not_beating(tmp_path) -> None:
    """Strict inequality: an unwired EQUAL carries no measured present loss, so rank 0 is wrong."""
    p = tmp_path / "req.jsonl"
    with pytest.raises(ValueError, match="does not beat"):
        record_required_component("Tie", path=p, **_harm_kwargs(
            live_measured=1.0, candidate_measured=1.0, metric_direction="lower-is-better"))


def test_harm_clause_refuses_missing_and_non_finite_measurements(tmp_path) -> None:
    """A failed measurement is not a comparison; prose alone can no longer stand in for one."""
    p = tmp_path / "req.jsonl"
    # Absent entirely -- the pre-fix call shape, which used to be accepted on prose alone.
    for missing in ("live_measured", "candidate_measured"):
        with pytest.raises(ValueError, match=missing):
            record_required_component("NoNumbers", path=p, **_harm_kwargs(**{missing: None}))
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="FINITE"):
            record_required_component("NonFinite", path=p, **_harm_kwargs(candidate_measured=bad))
    # A bool is an int in Python; accepting True as a measurement would be a silent coercion.
    with pytest.raises(ValueError, match="numeric"):
        record_required_component("BoolNum", path=p, **_harm_kwargs(candidate_measured=True))
    assert not p.exists() or p.read_text(encoding="utf-8").strip() == ""


def test_harm_advantage_is_none_rather_than_fabricated_when_a_ratio_is_meaningless(tmp_path) -> None:
    """A ratio across zero or a signed quantity has no meaning -- record None, never a number.

    The strict inequality is the gate; the ratio is evidence, so it must be absent when undefined
    instead of inventing a value the caller would later rank on.
    """
    p = tmp_path / "req.jsonl"
    row = record_required_component("SignedDelta", path=p, **_harm_kwargs(
        live_measured=0.0, candidate_measured=-3.0, metric_direction="lower-is-better"))
    assert row["harm_advantage"] is None
    assert row["live_measured"] == 0.0 and row["candidate_measured"] == -3.0
    # ...and it is still a legitimate grade-5 row: the DIRECTION was satisfied.
    assert {r["component"] for r in built_elsewhere_unwired(p)} == {"SignedDelta"}


def test_queue_order_is_fire_order_not_harm_as_the_docstring_now_states(tmp_path) -> None:
    """The docstring used to claim "ranked by quantified harm"; the code never did that.

    Asserted so the claim and the behaviour cannot drift apart again: the row with the SMALLER
    advantage sorts first because its fire_order is lower.
    """
    p = tmp_path / "req.jsonl"
    record_required_component("HugeWinLateFire", path=p, **_harm_kwargs(
        fire_order=9, live_measured=1000.0, candidate_measured=1.0))
    record_required_component("SmallWinEarlyFire", path=p, **_harm_kwargs(
        fire_order=1, live_measured=1.01, candidate_measured=1.0))
    q = built_elsewhere_unwired(p)
    assert [r["component"] for r in q] == ["SmallWinEarlyFire", "HugeWinLateFire"]
    assert q[0]["harm_advantage"] < q[1]["harm_advantage"], "ordering is deliberately NOT by harm"


def test_report_carries_the_signed_evidence_not_just_the_prose(tmp_path) -> None:
    """Found in my own round-1 review: the numbers were recorded and then dropped by the report.

    A measurement the operator-facing surface does not show is built-but-unsurfaced — the class
    this grade exists to name — reproduced inside the module that names it.
    """
    p = tmp_path / "req.jsonl"
    record_required_component("WiringDebt", path=p, **_harm_kwargs(
        live_measured=4.0, candidate_measured=1.0, metric_direction="lower-is-better"))
    row = next(r for r in build_completeness_report(p) if r["component"] == "WiringDebt")
    assert row["live_measured"] == 4.0
    assert row["candidate_measured"] == 1.0
    assert row["metric_direction"] == "lower-is-better"
    assert row["harm_advantage"] == pytest.approx(4.0)


def test_the_exact_pre_fix_call_shape_is_now_refused(tmp_path) -> None:
    """THE regression guard: the historical call shape carried prose and NO numbers.

    Verified against ``git show HEAD~:...`` before this change: that shape was ACCEPTED, entered
    ``built_elsewhere_unwired()``, and sorted to ``build_completeness_report()[0]`` — above every
    live debt row — while describing a ~3,400x REGRESSION. Reproduced here verbatim so the hole
    cannot silently reopen by someone restoring the old signature defaults.
    """
    p = tmp_path / "req.jsonl"
    with pytest.raises(ValueError):
        record_required_component(
            "PoseBasisSwap", needed_by="cfgOptimalFromStart",
            missing_mechanism="no live call site reaches it", owner="ddm_wd2", fire_order=1,
            consumer="the live receiver", grade="built-elsewhere-unwired",
            live_recipient="warp-pose6 (LIVE in inflate_runner_v4d.py:57)",
            measured_comparison="candidate plateaus d_pose ~29-30 vs live realized 0.00858133",
            path=p)
    assert not p.exists() or p.read_text(encoding="utf-8").strip() == ""
    assert built_elsewhere_unwired(p) == ()


# ---------------------------------------------------------------------------
# THE READ PATH RE-RUNS THE WRITE GATE (ddm_ri1, #899).
# ddm_wd2 signed the harm clause on the WRITE path and left the READ path open, deliberately
# and in writing: dropping unverified rows is signal loss, silently trusting them is a false
# claim. The third option is to TYPE what was loaded. MEASURED against the pre-fix module: a
# hand-appended grade-5 row with NO numbers -- a shape record_required_component REFUSES -- was
# read back, entered built_elsewhere_unwired(), and sorted to build_completeness_report()[0],
# ABOVE a genuinely measured row; and a truncated line vanished (3 lines on disk -> 2 rows, no
# signal). Since ddm_gd5 DELETED the grade-5 detector, DECLARATION is the only route into this
# grade, so the read path is the only remaining check on it.
# ---------------------------------------------------------------------------


def _hand_append(p, component, **over):
    """Write a row the way a HUMAN or a partial write would — bypassing the write gate entirely."""
    row = {
        "component": component, "needed_by": "cfgOptimalFromStart",
        "grade": "built-elsewhere-unwired", "missing_mechanism": "declared, never measured",
        "owner": "whoever-edited-the-file", "fire_order": 0, "consumer": "the live receiver",
        "notes": "", "live_recipient": "", "measured_comparison": "", "live_measured": None,
        "candidate_measured": None, "metric_direction": "", "harm_advantage": None,
        "agent": "hand-edit", "ts": "2026-08-02T00:00:00Z",
    }
    row.update(over)
    with Path(p).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def test_hand_appended_grade5_row_is_typed_declared_unverified(tmp_path) -> None:
    """POSITIVE CONTROL: the read path refuses to believe what the write path would have refused."""
    from tac.witness_dsl.activation_ledger import RECORD_DECLARED_UNVERIFIED

    p = tmp_path / "req.jsonl"
    _hand_append(p, "HandAppendedNoEvidence")
    row = next(r for r in read_required_components(p) if r["component"] == "HandAppendedNoEvidence")
    assert row["record_integrity"] == RECORD_DECLARED_UNVERIFIED
    # The REASON travels with the row: a bare flag would make the operator re-derive the defect.
    # (First failing check wins — here the absent recipient; the numbers branch is asserted below.)
    assert "live_recipient" in row["record_integrity_reason"]

    # A row that clears the PROSE fields but carries no numbers — the exact pre-wd2 call shape,
    # which is what a row written before the signed clause landed looks like on disk today.
    _hand_append(p, "PoseBasisSwap", live_recipient="warp-pose6 (live in inflate_runner_v4d)",
                 measured_comparison="candidate plateaus d_pose ~29-30 vs live 0.00858133")
    row2 = next(r for r in read_required_components(p) if r["component"] == "PoseBasisSwap")
    assert row2["record_integrity"] == RECORD_DECLARED_UNVERIFIED
    assert "metric_direction" in row2["record_integrity_reason"]


def test_written_row_verifies_so_the_typing_is_not_an_unconditional_reject(tmp_path) -> None:
    """NEGATIVE CONTROL. Without this, 'everything is declared-unverified' would pass every test."""
    from tac.witness_dsl.activation_ledger import RECORD_VERIFIED

    p = tmp_path / "req.jsonl"
    record_required_component("RealWinner", path=p, **_harm_kwargs())
    row = next(r for r in read_required_components(p) if r["component"] == "RealWinner")
    assert row["record_integrity"] == RECORD_VERIFIED
    assert row["record_integrity_reason"] == ""


def test_unverified_grade5_row_cannot_lead_the_operator_report(tmp_path) -> None:
    """THE defect, end to end: rank 0 asserts a MEASURED present loss, so a declaration cannot hold it.

    Pre-fix this ordering INVERTED — the evidence-free row led and the measured one sat under it.
    """
    from tac.witness_dsl.activation_ledger import BUILD_GRADE_ORDER, BUILD_NEVER_FIRED

    p = tmp_path / "req.jsonl"
    _hand_append(p, "HandAppendedNoEvidence")          # fire_order 0 — would sort first on every key
    record_required_component("RealWinner", path=p, **_harm_kwargs(fire_order=5))

    rep = {r["component"]: r for r in build_completeness_report(p)}
    assert build_completeness_report(p)[0]["component"] == "RealWinner"
    assert rep["RealWinner"]["sort_rank"] == 0
    # Demoted to the rank of the grade the refusal text calls it indistinguishable from...
    assert rep["HandAppendedNoEvidence"]["sort_rank"] == BUILD_GRADE_ORDER[BUILD_NEVER_FIRED]
    # ...but NOT relabelled: swapping one false record for another is not a repair.
    assert rep["HandAppendedNoEvidence"]["grade"] == "built-elsewhere-unwired"


def test_unverified_row_is_kept_not_dropped_and_the_queue_orders_verified_first(tmp_path) -> None:
    """Dropping is the failure mode ddm_wd2 named. It stays visible in every surface, typed."""
    from tac.witness_dsl.activation_ledger import RECORD_DECLARED_UNVERIFIED, RECORD_VERIFIED

    p = tmp_path / "req.jsonl"
    _hand_append(p, "HandAppendedNoEvidence")
    record_required_component("RealWinner", path=p, **_harm_kwargs(fire_order=5))

    q = built_elsewhere_unwired(p)
    assert [r["component"] for r in q] == ["RealWinner", "HandAppendedNoEvidence"]
    assert [r["record_integrity"] for r in q] == [RECORD_VERIFIED, RECORD_DECLARED_UNVERIFIED]
    assert "HandAppendedNoEvidence" in {r["component"] for r in read_required_components(p)}


def test_malformed_lines_are_counted_instead_of_silently_skipped(tmp_path) -> None:
    """A skipped line and an empty file used to emit the same symbol — the vacuity genus."""
    from tac.witness_dsl.activation_ledger import required_component_integrity_summary

    p = tmp_path / "req.jsonl"
    record_required_component("RealWinner", path=p, **_harm_kwargs())
    _hand_append(p, "HandAppendedNoEvidence")
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"component": "TruncatedRow", "grade": "built-elsewhe\n')   # unparseable
        fh.write('{"component": "BadGrade", "grade": "invented-grade"}\n')     # parses, not a row

    s = required_component_integrity_summary(p)
    assert (s["rows_read"], s["verified"], s["declared_unverified"]) == (2, 1, 1)
    assert s["malformed_lines"] == 2
    assert s["declared_unverified_components"] == ["HandAppendedNoEvidence"]
    assert {d["line_no"] for d in s["malformed_detail"]} == {3, 4}   # the LOCATION, not just a count


def test_integrity_summary_on_an_absent_store_reports_zeros_not_a_crash(tmp_path) -> None:
    """The denominator must be readable when there is nothing to read — that IS the useful answer."""
    from tac.witness_dsl.activation_ledger import required_component_integrity_summary

    s = required_component_integrity_summary(tmp_path / "does_not_exist.jsonl")
    assert s["rows_read"] == 0 and s["malformed_lines"] == 0


def test_the_live_store_is_currently_clean_so_the_typing_has_a_real_denominator(tmp_path) -> None:
    """Runs against the REAL ledger: every stored row must re-pass the gate that wrote it.

    This is the regression guard that matters — it fails the moment anyone hand-edits the canonical
    store into a state its own writer would refuse.
    """
    from tac.witness_dsl.activation_ledger import required_component_integrity_summary

    s = required_component_integrity_summary()
    assert s["rows_read"] > 0, "empty scope: this assertion would pass vacuously"
    assert s["declared_unverified"] == 0, s["declared_unverified_components"]
    assert s["malformed_lines"] == 0, s["malformed_detail"]


def test_one_malformed_fire_order_cannot_silence_the_whole_store(tmp_path) -> None:
    """Round-2 self-review catch: a bad sort key used to raise and take the ENTIRE read down.

    One bad row costing the whole corpus is the 2026-08-01 recall-layer failure. The bad row is
    already typed declared-unverified; it must not also be able to hide its healthy neighbours.
    """
    p = tmp_path / "req.jsonl"
    record_required_component("RealWinner", path=p, **_harm_kwargs())
    _hand_append(p, "BadFireOrder", fire_order="not-an-int")

    names = {r["component"] for r in read_required_components(p)}
    assert names == {"RealWinner", "BadFireOrder"}, "the healthy row must survive its bad neighbour"
    assert [r["component"] for r in built_elsewhere_unwired(p)][0] == "RealWinner"
    assert build_completeness_report(p)[0]["component"] == "RealWinner"
