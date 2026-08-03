# SPDX-License-Identifier: MIT
"""ddm_lr2 (2026-08-03) — the lever instruments must report on the vehicle we SHIP.

Three instruments were pointed at a retired vehicle and none of them said so. These tests pin
the repairs, and every one of them is written so it would FAIL if the repair were reverted —
the standard the arm's own charter set, after two single-file detectors shipped the day before
without a working positive control.

The tests are hermetic: nothing here reads the SSD tier or the live ledger, because a test that
passes only when a volume happens to be mounted is the same vacuity class being repaired.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tac.confound_gates import (
    check_lever_module_declares_its_trainer,
    check_no_asserted_packet_ir_readiness_fields,
)
from tac.optimization.inverse_steganalysis_operation_set_compiler import (
    OPERATION_SET_COMPILER_HINT_SCHEMA,
    _byte_closed_operation_count,
    _sequence_is_permutation,
    packet_ir_operation_set_from_compiler_hint,
)
from tac.witness_dsl import activation_ledger as al
from tac.witness_dsl.lever_registry import (
    build_completeness,
    completeness,
    module_declares_trainer,
    module_trainer_paths,
    package_lever_modules,
)

LIVE_TRAINER = "train_tr1_partition_renderer_mlx.py"
RETIRED_TRAINER = "train_levelset_witness_realized_through_R_mlx.py"

# The 8 factories that were graded against a trainer they were never written for. Named
# explicitly rather than counted, so this test says WHICH levers were mis-filed — a bare count
# would still pass if the set silently changed.
REHOMED_FACTORIES = {
    "TieLocusEdgeWeighted",
    "ErfBirthContextCoadapt",
    "MarginSatisficeCap",
    "BirthPlateauKneeConjunct",
    "XiAdvectedTokenBase",
    "Qa80MarginBoundedPhotometric",
    "Qa81LaneCarrierComposite",
    "Ax1Frame0CarriedWarp",
}


def _modules() -> dict[str, Path]:
    return {p.name: p for p in package_lever_modules()}


# ── §1 the re-home ───────────────────────────────────────────────────────────────────────────
def test_tr1_targeted_modules_bind_to_the_live_trainer():
    """The 8 mis-filed factories now resolve to the vehicle their own docstrings name."""
    by_factory = {f.factory: f for f in build_completeness().factories}
    missing = REHOMED_FACTORIES - set(by_factory)
    assert not missing, f"factories vanished from the registry: {sorted(missing)}"
    for name in sorted(REHOMED_FACTORIES):
        fb = by_factory[name]
        assert LIVE_TRAINER in fb.trainer, (
            f"{name} is graded against {fb.trainer!r}, not the live vehicle — the binding "
            "regressed and no TR1-scoped query can surface it"
        )
        assert fb.trainer_declared, f"{name}'s module stopped declaring its trainer"


def test_rehoming_regrades_but_does_not_build():
    """HONEST BOUND: declaring the trainer does not make a stub fireable.

    The flags these factories emit exist on NEITHER trainer (MEASURED), so they remain stubs.
    Pinning this stops the re-home from ever being narrated as "8 levers unblocked".
    """
    by_factory = {f.factory: f for f in build_completeness().factories}
    for name in sorted(REHOMED_FACTORIES):
        assert by_factory[name].is_stub, (
            f"{name} is no longer graded a stub — if its trainer flag was genuinely built that "
            "is good news, but update this test deliberately rather than letting it pass silently"
        )


def test_module_declares_trainer_distinguishes_declared_from_defaulted():
    mods = _modules()
    assert module_declares_trainer(mods["spec_tr1_renderer_20260728.py"])
    assert module_declares_trainer(mods["fh1_adapted_force_levers_20260731.py"])
    # A module with no declaration of either form must report False, not merely resolve a path.
    undeclared = [n for n, p in mods.items() if not module_declares_trainer(p)]
    assert undeclared, "expected some modules to still inherit the default binding"


def test_plural_declaration_is_behaviour_identical_to_the_default_it_replaced():
    """``TRAINER_RELPATHS`` was added so genuine multi-trainer modules could stop relying on the
    silent default WITHOUT changing what they resolve to. If it changed resolution it would
    manufacture false missing-flag grades — a false FAIL replacing a silent PASS."""
    mods = _modules()
    expected = {RETIRED_TRAINER, "train_witness_realized_through_R_mlx.py"}
    for name in ("curriculum_dsl.py", "constants_telemetry_build_wave_20260715.py"):
        resolved = {p.name for p in module_trainer_paths(mods[name])}
        assert resolved == expected, f"{name} resolves to {resolved}, expected {expected}"
        assert module_declares_trainer(mods[name])


def test_verdict_relevance_is_the_narrow_refusable_scope():
    bc = build_completeness()
    for fb in bc.factories:
        assert fb.trainer_binding_is_verdict_relevant == (fb.is_stub and not fb.trainer_declared)
    # A BUILT factory never enters the refusable set: its grade is the same under either trainer.
    assert all(f.is_stub for f in bc.verdict_relevant_undeclared)


def test_gate_live_count_is_zero_and_fires_when_a_declaration_is_removed(tmp_path):
    """Negative AND positive control. A gate that has only ever been seen returning zero has
    not been shown to work."""
    assert check_lever_module_declares_its_trainer(strict=False, verbose=False) == []

    target = _modules()["fh1_adapted_force_levers_20260731.py"]
    original = target.read_text(encoding="utf-8")
    decl = 'TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"'
    assert decl in original, "test fixture drifted: the declaration under test is gone"
    try:
        target.write_text(original.replace(decl, "# declaration removed by test"), encoding="utf-8")
        fired = check_lever_module_declares_its_trainer(strict=False, verbose=False)
        assert len(fired) == 5, f"expected the 5 fh1 factories refused, got {len(fired)}"
        assert all("fh1_adapted_force_levers" in v for v in fired)
        with pytest.raises(Exception):
            check_lever_module_declares_its_trainer(strict=True, verbose=False)
    finally:
        target.write_text(original, encoding="utf-8")
    assert check_lever_module_declares_its_trainer(strict=False, verbose=False) == []


# ── §2 the instruments must state their basis ────────────────────────────────────────────────
def test_completeness_carries_the_vehicle_and_fails_closed_on_unknown():
    default = completeness()
    assert not default.describes_live_vehicle, (
        "the DEFAULT completeness() call describes the RETIRED vehicle; if this flips, the "
        "campaign's default coverage number changed meaning and every citation of it is stale"
    )
    assert "RETIRED" in default.vehicle_label
    live = completeness(f"experiments/{LIVE_TRAINER}")
    assert live.describes_live_vehicle and "LIVE" in live.vehicle_label
    # Unknown vehicle must NOT read as live — an unlabelled number is the defect, not the cure.
    unknown = replace(default, trainer_path="")
    assert not unknown.describes_live_vehicle
    assert unknown.vehicle_label == "[vehicle UNKNOWN]"


def test_unreadable_launch_root_is_UNSCANNED_not_zero(tmp_path):
    """An unmounted SSD returning 'no receipts' must never read like a clean tree."""
    receipts, scanned, unavailable = al.live_launch_receipts(
        roots=(str(tmp_path / "does_not_exist"),)
    )
    assert receipts == [] and scanned == ()
    assert unavailable == (str(tmp_path / "does_not_exist"),)
    cov = al.ledger_coverage(path=tmp_path / "ledger.jsonl", roots=(str(tmp_path / "nope"),))
    assert cov.is_vacuous and "UNKNOWN, not zero" in cov.vacuity_reason


def test_ledger_is_vacuous_when_live_launches_have_no_rows(tmp_path):
    """The measured live state, reproduced hermetically: governed launches exist, ledger blind."""
    run_dir = tmp_path / "runs" / "window_01"
    run_dir.mkdir(parents=True)
    (run_dir / "launch_receipt.json").write_text(
        json.dumps({"schema": al._TR1_RECEIPT_SCHEMA, "argv": [], "ticket_path": ""}),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"lever": "SomeLever", "event": al.EVENT_FIRED,
                    "run_ref": "/somewhere/else", "ts": "2026-07-27T21:17:34Z"}) + "\n",
        encoding="utf-8",
    )
    cov = al.ledger_coverage(path=ledger, roots=(str(tmp_path / "runs"),))
    assert cov.live_launch_receipts == 1
    assert cov.live_receipts_joined_to_ledger == 0
    assert cov.is_vacuous
    assert "NONE has a ledger row" in cov.vacuity_reason
    assert cov.last_write_utc == "2026-07-27T21:17:34Z"


def test_ledger_is_not_vacuous_when_a_live_launch_does_join(tmp_path):
    """POSITIVE CONTROL for the vacuity verdict itself: it must be able to return False."""
    run_dir = tmp_path / "runs" / "window_01"
    run_dir.mkdir(parents=True)
    (run_dir / "launch_receipt.json").write_text(
        json.dumps({"schema": al._TR1_RECEIPT_SCHEMA, "argv": [], "ticket_path": ""}),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"lever": "SomeLever", "event": al.EVENT_FIRED,
                    "run_ref": str(run_dir), "ts": "2026-08-03T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    cov = al.ledger_coverage(path=ledger, roots=(str(tmp_path / "runs"),))
    assert cov.live_receipts_joined_to_ledger == 1
    assert not cov.is_vacuous and cov.vacuity_reason == ""


def test_coverage_fraction_of_an_empty_known_set_is_zero_not_one():
    cov = al.LedgerCoverage(
        known_levers=0, levers_with_any_row=0, rows=0, last_write_utc=None,
        live_launch_receipts=0, live_receipts_joined_to_ledger=0,
        roots_scanned=(), roots_unavailable=(), is_vacuous=True, vacuity_reason="x",
    )
    assert cov.lever_coverage_fraction == 0.0


def test_live_launch_lever_names_measures_the_namespace_join(tmp_path):
    """The join defect is the reason a naive record_activation wire-in is insufficient: the
    ticket records a constructed ``Lever.name``, the DSL universe is keyed by FACTORY name."""
    run_dir = tmp_path / "runs" / "w1"
    run_dir.mkdir(parents=True)
    ticket = tmp_path / "t.json"
    ticket.write_text(
        json.dumps({"levers": [{"name": "tr1_seg_ce"}, {"name": "tr1_renderer_w24"}]}),
        encoding="utf-8",
    )
    (run_dir / "launch_receipt.json").write_text(
        json.dumps({"schema": al._TR1_RECEIPT_SCHEMA, "ticket_path": str(ticket)}),
        encoding="utf-8",
    )
    names, diag = al.live_launch_lever_names(roots=(str(tmp_path / "runs"),))
    assert names == {"tr1_renderer_w24": 1, "tr1_seg_ce": 1}
    assert diag["tickets_read"] == 1 and diag["tickets_missing"] == 0
    assert diag["joinable_to_known_levers"] == 0
    assert diag["namespace_join_is_broken"] is True


# ── §3 the NO-FAKE two-landing ───────────────────────────────────────────────────────────────
def _hint(operations: list[dict]) -> dict:
    return {
        "schema": OPERATION_SET_COMPILER_HINT_SCHEMA,
        "operation_set_id": "t",
        "selected_operations": operations,
    }


def test_byte_closed_count_excludes_blocked_operations():
    """The field previously read ``len(operations)`` — it would have said 2 here."""
    out = packet_ir_operation_set_from_compiler_hint(
        _hint([
            {"unit_id": "u1", "operation_id": "clean", "operation_family": "f", "blockers": []},
            {"unit_id": "u2", "operation_id": "blocked", "operation_family": "f",
             "blockers": ["packetir_operation_not_byte_closed_missing_receipt"]},
        ]),
        source_backlog_key="k",
    )
    assert out["operation_count"] == 2
    assert out["byte_closed_operation_count"] == 1


def test_unrelated_blocker_does_not_reduce_the_byte_closed_count():
    out = packet_ir_operation_set_from_compiler_hint(
        _hint([
            {"unit_id": "u1", "operation_id": "a", "operation_family": "f", "blockers": []},
            {"unit_id": "u2", "operation_id": "b", "operation_family": "f",
             "blockers": ["packetir_operation_set_requires_materializer_contexts"]},
        ]),
        source_backlog_key="k",
    )
    assert out["byte_closed_operation_count"] == 2


def test_permutation_predicate_is_decided_not_asserted():
    ops = [{"operation_id": "a"}, {"operation_id": "b"}]
    assert _sequence_is_permutation([{"operation_id": "b"}, {"operation_id": "a"}], ops)
    assert not _sequence_is_permutation([{"operation_id": "a"}], ops)          # dropped
    assert not _sequence_is_permutation(
        [{"operation_id": "a"}, {"operation_id": "a"}], ops)                   # duplicated
    assert not _sequence_is_permutation([], ops)


def test_byte_closed_helper_on_an_empty_set_is_zero():
    assert _byte_closed_operation_count([]) == 0


def test_asserted_readiness_gate_zero_live_and_fires_on_reintroduction():
    assert check_no_asserted_packet_ir_readiness_fields(strict=False, verbose=False) == []

    target = Path("src/tac/optimization/inverse_steganalysis_operation_set_compiler.py")
    original = target.read_text(encoding="utf-8")
    real = '"byte_closed_operation_count": _byte_closed_operation_count(operations),'
    assert real in original, "test fixture drifted: the computed call is gone"
    try:
        target.write_text(
            original.replace(real, '"byte_closed_operation_count": len(operations),'),
            encoding="utf-8",
        )
        fired = check_no_asserted_packet_ir_readiness_fields(strict=False, verbose=False)
        assert len(fired) == 1 and "byte_closed_operation_count" in fired[0]
        with pytest.raises(Exception):
            check_no_asserted_packet_ir_readiness_fields(strict=True, verbose=False)
    finally:
        target.write_text(original, encoding="utf-8")
    assert check_no_asserted_packet_ir_readiness_fields(strict=False, verbose=False) == []


def test_asserted_readiness_gate_respects_a_real_waiver(tmp_path):
    """And a placeholder rationale must NOT self-waive (Catalog #287 sister)."""
    pkg = tmp_path / "src" / "tac" / "optimization"
    pkg.mkdir(parents=True)

    def _write(marker: str) -> None:
        # The waiver comment must be the LAST thing on the offending line, exactly as a real
        # waiver appears — and the module must stay PARSEABLE, or the gate skips it on
        # SyntaxError and returns a vacuous zero. (This test's first draft did exactly that:
        # the comment swallowed the closing braces and the "waiver worked" result was really
        # "the file never got scanned". Kept as a comment because it is the arm's own genus.)
        (pkg / "m.py").write_text(
            "PACKET_IR_OPERATION_SET_SCHEMA = 'x'\n"
            "def f(operations):\n"
            "    return {\n"
            f"        'byte_closed_operation_count': len(operations),{marker}\n"
            "    }\n",
            encoding="utf-8",
        )

    _write("")
    assert len(check_no_asserted_packet_ir_readiness_fields(
        repo_root=tmp_path, strict=False, verbose=False)) == 1

    _write("  # ASSERTED_READINESS_FIELD_OK:<rationale>")
    assert len(check_no_asserted_packet_ir_readiness_fields(
        repo_root=tmp_path, strict=False, verbose=False)) == 1, "placeholder must not self-waive"

    _write("  # ASSERTED_READINESS_FIELD_OK:fixture emits a single known op")
    assert check_no_asserted_packet_ir_readiness_fields(
        repo_root=tmp_path, strict=False, verbose=False) == []


def test_asserted_readiness_gate_does_not_silently_skip_a_parseable_file(tmp_path):
    """Guards the blind spot the broken fixture above revealed: the scan skips unparseable
    files, so a test whose fixture does not parse would 'pass' while scanning nothing. This
    asserts the fixture shape used above is genuinely scanned."""
    pkg = tmp_path / "src" / "tac" / "optimization"
    pkg.mkdir(parents=True)
    src = (
        "PACKET_IR_OPERATION_SET_SCHEMA = 'x'\n"
        "def f(operations):\n"
        "    return {\n"
        "        'byte_closed_operation_count': len(operations),\n"
        "    }\n"
    )
    (pkg / "m.py").write_text(src, encoding="utf-8")
    import ast as _ast

    _ast.parse(src)  # the fixture must PARSE, else the assertion below is vacuous
    assert len(check_no_asserted_packet_ir_readiness_fields(
        repo_root=tmp_path, strict=False, verbose=False)) == 1
