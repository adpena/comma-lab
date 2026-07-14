# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

import tools.schedule_provenance_gate as schedule_gate
from tac.witness_dsl.spec_throughput_component_timer_20260713 import (
    ASYNC_PROGRAM,
    EPOCHS,
    N_PAIRS,
    SOLO_PROGRAM,
    compile_throughput_component_timer_ticket,
)


@pytest.mark.parametrize(
    ("variant", "program", "async_expected"),
    [("async_current", ASYNC_PROGRAM, True), ("solo_control", SOLO_PROGRAM, False)],
)
def test_tickets_are_ce_exact_n24_matched_arms(
    variant: str, program: str, async_expected: bool
) -> None:
    cfg = compile_throughput_component_timer_ticket(variant=variant)
    flags = dict(cfg.to_trainer_flags("OUT"))
    assert cfg.name == program
    assert flags["--num-pairs"] == str(N_PAIRS)
    assert flags["--epochs"] == str(EPOCHS)
    assert flags["--softmax-temp-start"] == "1.0"
    assert flags["--softmax-temp-end"] == "1.0"
    assert "--seg-form-unify-tau" in flags
    assert flags["--component-wallclock-probe-every"] == "1"
    assert flags["--eval-every"] == "1"
    assert flags["--verdict-pairs"] == "2"
    assert flags["--ckpt-every"] == "1"
    assert flags["--loss-term-log-every"] == "-1"
    assert flags["--seg-loss"] == "ce"
    assert flags["--w-pose"] == "0.0"
    for zero_flag in (
        "--persistence-loss-weight",
        "--persistence-recall-weight",
        "--amplify-weight",
        "--seg-chroma-boundary-weight",
        "--seg-temporal-screw-weight",
        "--weight-entropy-penalty-lambda",
        "--logit-adjust-loss-tau",
        "--eikonal-weight",
        "--length-weight",
        "--weight-decay",
    ):
        assert flags[zero_flag] == "0.0"
    assert "--no-witness-alone-island-loss" in flags
    for off_flag in (
        "--no-curriculum",
        "--no-curriculum-event-triggered",
        "--no-curriculum-nucleus-guard",
        "--no-curriculum-reanchor-levers",
        "--no-dseg-aware-taper",
        "--no-seed-islands",
    ):
        assert off_flag in flags
    assert not any(flag.endswith("-start-epoch") for flag in flags)
    assert not any(flag.endswith("-start-event") for flag in flags)
    assert cfg.dsl_levers == (f"throughput_component_timer_{variant}",)
    assert cfg.schedule_governance == {}
    assert "polyak_finisher_start_epoch" not in cfg.constants_manifest
    assert ("--async-verdict" in flags) is async_expected
    assert ("--no-async-verdict" in flags) is (not async_expected)
    assert cfg.dsl_program_manifest["operator_go_required"] is True
    assert cfg.dsl_program_manifest["score_claim"] is False


def test_async_and_solo_are_exactly_matched_except_async_toggle() -> None:
    async_flags = dict(
        compile_throughput_component_timer_ticket(variant="async_current")
        .to_trainer_flags("OUT")
    )
    solo_flags = dict(
        compile_throughput_component_timer_ticket(variant="solo_control")
        .to_trainer_flags("OUT")
    )
    assert async_flags.pop("--async-verdict") is None
    assert solo_flags.pop("--no-async-verdict") is None
    assert async_flags == solo_flags


def test_schedule_provenance_has_zero_naked_epochs() -> None:
    cfg = compile_throughput_component_timer_ticket()
    trainer = Path("experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    verdicts = schedule_gate.classify_launch(
        list(cfg.to_trainer_flags("OUT")),
        registry=schedule_gate.schedule_when_flags(trainer),
        manifest_keys=set(cfg.constants_manifest),
        governance=cfg.schedule_governance,
        event_registry=schedule_gate.event_start_flags(trainer),
    )
    ok, violations, table = schedule_gate.gate_report(verdicts)
    assert ok, table
    assert violations == []


def test_ticket_refuses_silent_resize_or_unknown_variant() -> None:
    with pytest.raises(ValueError, match="sealed to n24"):
        compile_throughput_component_timer_ticket(num_pairs=600)
    with pytest.raises(ValueError, match="unknown timer variant"):
        compile_throughput_component_timer_ticket(variant="invented")
