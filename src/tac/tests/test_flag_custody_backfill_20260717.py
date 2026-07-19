# SPDX-License-Identifier: MIT
"""#332 flag-custody backfill (2026-07-17) — behavior tests.

Covers the NEW surfaces the backfill landed: string LawRef literals (declaration
codec round-trip), the registered non-derivational identity evaluator, custody
byte-neutrality + idempotence, the WitnessProgram custody composition guard, the
volatile run-identity exemption, and identity-record refresh after later-Lever
composition.  The authority for the completed graph itself is
``check_config_flag_provenance_bijection_complete`` (tested in
``test_v9_provenance_gates``).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# LawRef string custody (the #351 "preserves bool/int/float/string bytes" leg)
# ---------------------------------------------------------------------------


def test_inputref_literal_accepts_string_and_roundtrips_declaration() -> None:
    from tac.witness_dsl.lawref import (
        InputRef,
        LawRef,
        lawref_from_declaration,
        lawref_to_declaration,
    )

    ref = LawRef(
        equation_id="dsl_custodied_scalar_identity_v1",
        inputs={"value": InputRef.literal("step_basis", "string custody test")},
        ladder_class="hardcoded_waiver",
        fallback="step_basis",
        fallback_waiver_reason="string custody test waiver",
    )
    rehydrated = lawref_from_declaration(lawref_to_declaration(ref))
    assert rehydrated.inputs["value"].value == "step_basis"
    assert rehydrated.fallback == "step_basis"
    assert rehydrated.equation_id == ref.equation_id


def test_inputref_literal_still_rejects_bool_and_nan() -> None:
    from tac.witness_dsl.lawref import InputRef, LawRefError

    with pytest.raises(LawRefError):
        InputRef.literal(True, "bool stays out (0/1 convention)")
    with pytest.raises(LawRefError):
        InputRef.literal(float("nan"), "NaN refused")


def test_identity_evaluator_is_nonderivational_and_refuses_bad_inputs() -> None:
    from tac.canonical_equations.evaluators import (
        EvaluatorError,
        populate_lawref_evaluators,
        resolve_equation_value,
    )

    populate_lawref_evaluators()
    assert resolve_equation_value("dsl_custodied_scalar_identity_v1", {"value": "x"}) == "x"
    assert resolve_equation_value("dsl_custodied_scalar_identity_v1", {"value": 3}) == 3
    with pytest.raises((EvaluatorError, KeyError)):
        resolve_equation_value("dsl_custodied_scalar_identity_v1", {})
    with pytest.raises(EvaluatorError):
        resolve_equation_value("dsl_custodied_scalar_identity_v1", {"value": float("inf")})


# ---------------------------------------------------------------------------
# attach/strip custody on the real 432 factory (compiled once, ~seconds)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def launch_432():
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config

    return compile_v9_cgauge_432_launch_config()


def test_custody_rollup_is_byte_neutral_and_idempotent(launch_432) -> None:
    from tac.witness_dsl.spec_v9_cgauge import (
        V9_FLAG_CUSTODY_LEVER,
        attach_flag_custody,
        strip_flag_custody,
    )

    typed = launch_432.typed
    names = [lever.name for lever in typed.levers]
    assert names.count(V9_FLAG_CUSTODY_LEVER) == 1
    argv = tuple(typed.to_program().compile_trainer_argv())
    stripped = strip_flag_custody(typed)
    assert V9_FLAG_CUSTODY_LEVER not in [lever.name for lever in stripped.levers]
    assert tuple(stripped.to_program().compile_trainer_argv()) == argv
    reattached, _ = attach_flag_custody(
        typed, launch_432.constants_manifest, program_name="v9_cgauge_432"
    )
    assert tuple(reattached.to_program().compile_trainer_argv()) == argv
    assert [lever.name for lever in reattached.levers].count(V9_FLAG_CUSTODY_LEVER) == 1


def test_custody_guard_refuses_post_custody_base_mutation(launch_432) -> None:
    typed = launch_432.typed
    mutated = typed.model_copy(
        update={"base": {**typed.base, "--annulus-band": 3.0}}
    )
    violations = mutated.validate_program()
    assert any("CUSTODY NON-NEUTRAL" in violation for violation in violations)
    assert any("--annulus-band" in violation for violation in violations)


def test_volatile_out_dir_is_exempt_from_semantic_custody(launch_432) -> None:
    from tac.witness_dsl.spec_v9_cgauge import V9_FLAG_CUSTODY_LEVER

    rollup = next(
        lever for lever in launch_432.typed.levers
        if lever.name == V9_FLAG_CUSTODY_LEVER
    )
    assert "--out-dir" not in rollup.overrides
    assert "--out-dir" not in rollup.lawref_declarations
    # ownership of a volatile value would shadow derived-config out_dir changes;
    # a rebound copy with a different out_dir must therefore stay valid.
    rebound = launch_432.typed.model_copy(update={"out_dir": "experiments/results/__custody_probe__"})
    assert not [
        violation for violation in rebound.validate_program()
        if "CUSTODY NON-NEUTRAL" in violation
    ]


def test_identity_records_refresh_after_later_lever_composition(launch_432) -> None:
    from tac.witness_dsl.spec_v9_cgauge import (
        DSL_IDENTITY_EQUATION_ID,
        refresh_identity_custody_records,
    )

    fd = dict(launch_432.typed.to_program().flag_dict())
    key = "ckpt_every"
    before = launch_432.constants_manifest[key]
    assert before["equation_id"] == DSL_IDENTITY_EQUATION_ID
    fd["--ckpt-every"] = 1  # a launcher-owned bounded delta composes later and wins
    refreshed = refresh_identity_custody_records(launch_432.constants_manifest, fd)
    assert refreshed[key]["value"] == 1
    assert refreshed[key]["equation_id"] == DSL_IDENTITY_EQUATION_ID
    assert refreshed[key]["prior_custody_value_historical_non_authorizing"] == before["value"]
    # untouched flags keep their records byte-for-byte
    assert refreshed["hosc_beta_end"] == launch_432.constants_manifest["hosc_beta_end"]


def test_class4_waiver_custody_is_typed_and_substantive(launch_432) -> None:
    from tac.witness_dsl.spec_v9_cgauge import DSL_IDENTITY_EQUATION_ID

    rows = [
        record for record in launch_432.constants_manifest.values()
        if isinstance(record, dict)
        and record.get("equation_id") == DSL_IDENTITY_EQUATION_ID
    ]
    assert rows, "identity custody records must exist"
    for record in rows:
        custody = record["waiver_custody"]
        for field in ("constant", "value", "reason", "owner",
                      "rederivation_trigger", "battery_arm"):
            assert str(custody[field]).strip()
        assert "#332 flag-custody backfill" in custody["reason"]
        assert record["ladder_class"] == "hardcoded_waiver"
        assert record["fallback_used"] is False
