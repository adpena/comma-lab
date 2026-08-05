# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from tac.canonical_equations.jd1_plateau_tail_average_ema_20260805 import (
    build_equation as build_jd1_tail_average_equation,
    plateau_tail_live_weight,
)
from tac.witness_dsl.curriculum_dsl import Lever
from tac.witness_dsl.scope_laws import (
    inertness_violations,
    jd1_tail_average_scope_law_refs,
    jd3_default_scope_law_refs,
    resolve_scope_law,
    scope_law_geometry_hash,
    ticket_payload_hash,
    ticket_scope_law_refs,
    validate_ticket_scope_laws,
)
from tac.witness_dsl.spec_tr1_renderer_20260728 import TR1RendererProgramV1


def test_scope_law_resolution_is_deterministic_for_same_inputs():
    inputs = {
        "remaining_epochs": 8,
        "steps_per_epoch": 150,
        "run_geometry_hash": scope_law_geometry_hash(
            steps_per_epoch=150,
            horizon_epochs=20,
            window_epochs=8,
        ),
    }
    first = resolve_scope_law("jd3_stage_ema_decay", inputs)
    second = resolve_scope_law("jd3_stage_ema_decay", dict(inputs))
    assert first == second
    assert first["resolved_value"] == pytest.approx(0.9966666666666667)
    assert first["tier"] == "T2_SCOPE_LAW"
    assert first["resolution_hash"]
    assert "remaining_epochs" in first["inputs"]
    assert first["inputs"]["run_geometry_hash"] == inputs["run_geometry_hash"]


def test_scope_law_geometry_hash_keys_resolved_values_by_window_geometry():
    h1 = scope_law_geometry_hash(steps_per_epoch=150, horizon_epochs=20, window_epochs=8)
    h2 = scope_law_geometry_hash(steps_per_epoch=150, horizon_epochs=20, window_epochs=9)
    assert h1 != h2
    row1 = resolve_scope_law(
        "jd3_stage_ema_decay",
        {"remaining_epochs": 8, "steps_per_epoch": 150, "run_geometry_hash": h1},
    )
    row2 = resolve_scope_law(
        "jd3_stage_ema_decay",
        {"remaining_epochs": 9, "steps_per_epoch": 150, "run_geometry_hash": h2},
    )
    assert row1["resolution_hash"] != row2["resolution_hash"]


def test_tail_average_scope_law_matches_polyak_update_weight():
    ref = jd1_tail_average_scope_law_refs()[0]
    assert ref["name"] == "jd1_plateau_tail_average_ema"
    assert ref["lawref_declaration"]["equation_id"] == "jd1_plateau_tail_average_ema_v1"
    row = resolve_scope_law("jd1_plateau_tail_average_ema", {"updates_since_anchor": 0})
    assert row["tier"] == "T3_LIVE_ADAPTED"
    assert row["resolved_value"] == pytest.approx(0.5)
    assert row["output_field"] == "ema_tail_live_weight"
    assert plateau_tail_live_weight(8) == pytest.approx(0.1)


def test_tail_average_canonical_equation_builds_from_source_inspection_anchor():
    eq = build_jd1_tail_average_equation()
    assert eq.equation_id == "jd1_plateau_tail_average_ema_v1"
    assert eq.next_recalibration_trigger == "never_auto_operator_only"
    assert eq.empirical_anchors[0].empirical_verification_status == "VERIFIED_VIA_SOURCE_INSPECTION"


def test_scope_law_resolution_round_trips_for_resume_metadata():
    gate = {
        "explicit_margin": 0.0,
        "realized_gate_dseg_per_pair_sd": 0.00072,
        "realized_gate_pair_ids": list(range(36)),
    }
    row = resolve_scope_law("jd3_realized_hold_margin", gate)
    restored = json.loads(json.dumps([row], sort_keys=True))
    assert restored == [row]
    assert inertness_violations(ticket_scope_law_refs(["jd3_realized_hold_margin"]), restored) == []


def test_scope_law_inertness_positive_control_flags_missing_resolution():
    declared = ticket_scope_law_refs([
        "jd3_realized_hold_margin",
        "jd3_realized_hold_floor_latch",
    ])
    resolved = [resolve_scope_law(
        "jd3_realized_hold_margin",
        {
            "explicit_margin": 0.0,
            "realized_gate_dseg_per_pair_sd": 0.00072,
            "realized_gate_pair_ids": list(range(36)),
        },
    )]
    violations = inertness_violations(declared, resolved)
    assert [v["name"] for v in violations] == ["jd3_realized_hold_floor_latch"]
    assert violations[0]["status"] == "INERT"
    assert violations[0]["inertness_alarm"]["alarm_id"] == "jd3_realized_hold_floor_INERT"


def test_ticket_scope_laws_are_schema_validated_and_hashed():
    laws = jd3_default_scope_law_refs()
    validate_ticket_scope_laws(laws)
    payload = {
        "schema": "ddm_tb1_tr1_sealed_ticket.v1",
        "trainer": "experiments/train_tr1_partition_renderer_mlx.py",
        "argv": ["experiments/train_tr1_partition_renderer_mlx.py", "--num-pairs", "600"],
        "levers": [],
        "scope_laws": laws,
        "score_claim": False,
    }
    h1 = ticket_payload_hash(payload)
    payload_without_laws = dict(payload)
    payload_without_laws.pop("scope_laws")
    h0 = ticket_payload_hash(payload_without_laws)
    assert h1 != h0
    payload["scope_laws"] = laws[:-1]
    assert ticket_payload_hash(payload) != h1


def test_tr1_ticket_laws_do_not_change_argv_when_absent_or_declared():
    lever = Lever("window", overrides={"--epochs": "2"})
    base = TR1RendererProgramV1(
        levers=(lever,),
        num_pairs=4,
        out_dir="/tmp/base",
        seed=0,
    ).sealed_ticket()
    with_laws = TR1RendererProgramV1(
        levers=(lever,),
        num_pairs=4,
        out_dir="/tmp/base",
        seed=0,
        scope_laws=tuple(ticket_scope_law_refs(["jd3_stage_ema_decay"])),
    ).sealed_ticket()
    assert base["argv"] == with_laws["argv"]
    assert "scope_laws" not in base
    assert with_laws["scope_laws"][0]["name"] == "jd3_stage_ema_decay"
    assert base["ticket_hash"] != with_laws["ticket_hash"]
