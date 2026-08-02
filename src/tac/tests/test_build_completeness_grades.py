# SPDX-License-Identifier: MIT
"""BEHAVIOUR tests for the build-completeness axis + the DESIGNED-STUB refusal gate.

ddm_sb2 (task #819). Every test asserts a PROPERTY of the derivation on real repo inputs —
none asserts that a canonical constant equals itself (NO-FAKE forbidden class #2). The
mutation guard at the bottom states explicitly what a marker-returning stub could not do.
"""
from __future__ import annotations

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
    assert len(set(VALID_BUILD_GRADES)) == 5


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
