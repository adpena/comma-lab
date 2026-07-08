"""Tests for tac.witness_dsl.typed_config — the typed DSL authoring schema.

Covers: provenance-ladder validation (types/ranges/extra-forbid/waiver-required),
schedule_governance typing, to_program() -> valid WitnessProgram, manifest emission.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tac.witness_dsl.curriculum_dsl import WitnessProgram
from tac.witness_dsl.typed_config import (
    PROGRAM_MANIFEST_SCHEMA,
    GovernanceClass,
    Provenanced,
    ProvenanceClass,
    ScheduleGovernance,
    TypedAnneal,
    TypedConfigError,
    TypedLever,
    TypedRegularizer,
    TypedStage,
    TypedWitnessConfig,
)


def _anneal() -> TypedAnneal:
    return TypedAnneal(
        start=Provenanced(value=1.0, provenance=ProvenanceClass.MEASURED_ANCHOR, unit="tau"),
        end=Provenanced(value=0.31, provenance=ProvenanceClass.MEASURED_ANCHOR, unit="tau",
                        source="mod32cap ep650-best"),
    )


def _valid_config(**over) -> TypedWitnessConfig:
    kw = dict(
        name="test", out_dir="experiments/results/x", gt_cache="gt.npz",
        num_pairs=600, epochs=1500,
        wall_clock_budget_days=Provenanced(
            value=3.7, provenance=ProvenanceClass.DERIVED_AT_CONFIG, unit="days",
            source="derive_wall_clock_budget_days(epochs)"),
        temp=_anneal(),
        stages=(TypedStage(name="tau_softplus",
                           start_epoch_flag="--tau-softplus-start-epoch", start_epoch=300),),
        regularizers=(TypedRegularizer(
            flag="--eikonal-weight",
            weight=Provenanced(value=0.01, provenance=ProvenanceClass.DERIVED_AT_CONFIG)),),
        base={"--w-seg": 100, "--curriculum": True},
    )
    kw.update(over)
    return TypedWitnessConfig(**kw)


# ── Provenanced (the ladder) ────────────────────────────────────────────────
def test_provenanced_accepts_all_derived_and_measured_classes():
    for pc in (ProvenanceClass.DERIVED_LIVE, ProvenanceClass.DERIVED_AT_CONFIG,
               ProvenanceClass.MEASURED_ANCHOR):
        p = Provenanced(value=1.0, provenance=pc)
        assert p.provenance is pc


def test_provenanced_hardcoded_requires_waiver():
    with pytest.raises(ValidationError):
        Provenanced(value=0.1, provenance=ProvenanceClass.HARDCODED_WITH_WAIVER)


def test_provenanced_hardcoded_with_blank_waiver_refused():
    with pytest.raises(ValidationError):
        Provenanced(value=0.1, provenance=ProvenanceClass.HARDCODED_WITH_WAIVER, waiver="   ")


def test_provenanced_hardcoded_with_waiver_accepted():
    p = Provenanced(value=0.1, provenance=ProvenanceClass.HARDCODED_WITH_WAIVER,
                    waiver="pinned; re-derive on live-m_q law promotion")
    assert p.waiver


def test_provenanced_waiver_on_non_hardcoded_refused():
    with pytest.raises(ValidationError):
        Provenanced(value=1, provenance=ProvenanceClass.DERIVED_LIVE, waiver="oops")


def test_provenanced_extra_key_forbidden():
    with pytest.raises(ValidationError):
        Provenanced(value=1, provenance=ProvenanceClass.DERIVED_LIVE, bogus=True)


def test_provenanced_scalar_only_no_containers():
    with pytest.raises(ValidationError):
        Provenanced(value=[1, 2], provenance=ProvenanceClass.DERIVED_LIVE)


def test_provenanced_is_frozen():
    p = Provenanced(value=1, provenance=ProvenanceClass.DERIVED_LIVE)
    with pytest.raises(ValidationError):
        p.value = 2  # type: ignore[misc]


# ── ScheduleGovernance (the gate's declaration surface) ─────────────────────
def test_governance_event_requires_sensor():
    with pytest.raises(ValidationError):
        ScheduleGovernance(**{"class": "event", "rationale": "r"})


def test_governance_cap_requires_sensor():
    with pytest.raises(ValidationError):
        ScheduleGovernance(**{"class": "cap", "rationale": "r"})


def test_governance_blank_rationale_refused():
    with pytest.raises(ValidationError):
        ScheduleGovernance(**{"class": "event", "sensor": "--s", "rationale": "  "})


def test_governance_sensor_must_be_flag():
    with pytest.raises(ValidationError):
        ScheduleGovernance(**{"class": "event", "sensor": "s", "rationale": "r"})


def test_governance_valid_cap_and_event():
    cap = ScheduleGovernance(**{"class": "cap", "sensor": "--curriculum-event-triggered",
                                "rationale": "fail-safe cap backing the event trigger"})
    assert cap.cls is GovernanceClass.CAP and cap.sensor == "--curriculum-event-triggered"
    ev = ScheduleGovernance(**{"class": "event", "sensor": "--plateau-trigger", "rationale": "r"})
    assert ev.cls is GovernanceClass.EVENT


def test_governance_extra_key_forbidden():
    with pytest.raises(ValidationError):
        ScheduleGovernance(**{"class": "event", "sensor": "--s", "rationale": "r", "x": 1})


# ── typed schedule/lever elements ───────────────────────────────────────────
def test_stage_flag_epoch_must_be_paired():
    with pytest.raises(ValidationError):
        TypedStage(name="s", start_epoch=5)
    with pytest.raises(ValidationError):
        TypedStage(name="s", start_epoch_flag="--tau-softplus-start-epoch")


def test_stage_start_epoch_non_negative():
    with pytest.raises(ValidationError):
        TypedStage(name="s", start_epoch_flag="--tau-softplus-start-epoch", start_epoch=-1)


def test_stage_ce_base_both_none_ok():
    s = TypedStage(name="CE")
    assert s.to_dsl().flags() == {}


def test_regularizer_flag_must_start_with_dashdash():
    with pytest.raises(ValidationError):
        TypedRegularizer(flag="eikonal", weight=Provenanced(
            value=0.01, provenance=ProvenanceClass.DERIVED_AT_CONFIG))


def test_lever_override_keys_must_be_flags():
    with pytest.raises(ValidationError):
        TypedLever(name="lv", overrides={"muon-lr": 0.002})


def test_lever_to_dsl_roundtrip():
    lv = TypedLever(name="Muon", overrides={"--muon-lr": 0.002}, epochs_delta=100)
    d = lv.to_dsl()
    assert d.name == "Muon" and d.overrides == {"--muon-lr": 0.002} and d.epochs_delta == 100


# ── TypedWitnessConfig top-level ────────────────────────────────────────────
def test_config_extra_key_forbidden():
    with pytest.raises(ValidationError):
        _valid_config(bogus=1)


def test_config_num_pairs_and_epochs_positive():
    with pytest.raises(ValidationError):
        _valid_config(num_pairs=0)
    with pytest.raises(ValidationError):
        _valid_config(epochs=0)


def test_config_base_keys_must_be_flags():
    with pytest.raises(ValidationError):
        _valid_config(base={"w-seg": 1})


def test_config_governance_keys_must_be_flags():
    with pytest.raises(ValidationError):
        _valid_config(schedule_governance={
            "tau": ScheduleGovernance(**{"class": "cap", "sensor": "--s", "rationale": "r"})})


def test_config_mlx_device_enum():
    with pytest.raises(ValidationError):
        _valid_config(mlx_device="mps")


def test_to_program_returns_valid_witness_program():
    c = _valid_config()
    prog = c.to_program()
    assert isinstance(prog, WitnessProgram)
    assert c.validate_program() == []  # never-invent-flags + behaviour clauses clean


def test_to_program_flags_all_exist_in_trainer_argparse():
    """The constructed program's emitted flags must all be REAL trainer flags."""
    c = _valid_config()
    viol = c.validate_program()
    invented = [v for v in viol if "INVENTED FLAG" in v]
    assert not invented, invented


def test_to_program_invented_base_flag_is_caught_by_dsl_validate():
    """A base flag not in the trainer argparse is refused by WitnessProgram.validate
    (the typed layer accepts the shape; the DSL validate is the never-invent-flags authority)."""
    c = _valid_config(base={"--totally-invented-flag": 1})
    viol = c.validate_program()
    assert any("INVENTED FLAG" in v and "totally-invented-flag" in v for v in viol)


# ── manifest ────────────────────────────────────────────────────────────────
def test_program_manifest_schema_and_fields():
    man = _valid_config().program_manifest()
    assert man["schema"] == PROGRAM_MANIFEST_SCHEMA
    for k in ("program_name", "typed_config_hash", "argv_sha256", "flag_names",
              "dsl_module", "dsl_qualname", "compiled_at_utc"):
        assert k in man, k
    assert man["dsl_qualname"] == "TypedWitnessConfig.to_program"
    assert isinstance(man["flag_names"], list) and man["flag_names"]


def test_typed_config_hash_is_deterministic():
    c = _valid_config()
    assert c.typed_config_hash() == c.typed_config_hash()


def test_typed_config_hash_changes_on_value_change():
    a = _valid_config().typed_config_hash()
    b = _valid_config(epochs=1600).typed_config_hash()
    assert a != b


def test_emitted_flag_names_sorted_and_stable():
    c = _valid_config()
    names = c.emitted_flag_names()
    assert list(names) == sorted(names)
    assert names == c.emitted_flag_names()


def test_manifest_flag_names_match_emitted_program_flags():
    c = _valid_config()
    man = c.program_manifest()
    assert sorted(man["flag_names"]) == sorted(c.to_program().flag_dict().keys())


def test_typed_config_error_on_broken_program(monkeypatch):
    """to_program surfaces a TypedConfigError (not a raw crash) if the DSL construction fails."""
    import tac.witness_dsl.typed_config as tc

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(tc, "WitnessProgram", _Boom)
    with pytest.raises(TypedConfigError):
        _valid_config().to_program()
