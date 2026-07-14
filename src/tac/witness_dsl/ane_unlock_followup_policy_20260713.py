# SPDX-License-Identifier: MIT
"""Typed, held policy for the ANE/full-trainer concurrency measurement.

Compilation is pure.  This module cannot launch a trainer or sidecar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

LANE_ID = "lane_ane_unlock_followup_20260713"
N_PAIRS = 24
EPOCHS = 4
ARMS = ("trainer_solo", "trainer_plus_ane_sidecar")
BASE_CONFIG = "throughput_component_timer_solo_20260713"


@dataclass(frozen=True)
class ANETrainerConcurrencyTicket:
    arm: str
    lane_id: str
    base_config: str
    n_pairs: int
    epochs: int
    operator_go_required: bool
    held: bool
    research_only: bool
    sidecar_enabled: bool
    score_claim: bool
    pointer_moved: bool
    resume_required: bool
    per_stage_checkpoints_required: bool
    treatment_invariant: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compile_ane_trainer_concurrency_ticket(
    arm: str,
    *,
    n_pairs: int = N_PAIRS,
    epochs: int = EPOCHS,
    operator_go: bool = False,
) -> ANETrainerConcurrencyTicket:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if int(n_pairs) != N_PAIRS or int(epochs) != EPOCHS:
        raise ValueError("ANE trainer concurrency packet is sealed to n24 and four epochs")
    if operator_go:
        raise PermissionError(
            "this compiler is prepare-only; actuation remains tools/launch_witness_run.py plus explicit operator GO"
        )
    return ANETrainerConcurrencyTicket(
        arm=arm,
        lane_id=LANE_ID,
        base_config=BASE_CONFIG,
        n_pairs=N_PAIRS,
        epochs=EPOCHS,
        operator_go_required=True,
        held=True,
        research_only=True,
        sidecar_enabled=arm == "trainer_plus_ane_sidecar",
        score_claim=False,
        pointer_moved=False,
        resume_required=True,
        per_stage_checkpoints_required=True,
        treatment_invariant=(
            "identical typed trainer config and seed; only the external frozen-SegNet ANE sidecar differs"
        ),
    )


__all__ = [
    "ARMS",
    "BASE_CONFIG",
    "EPOCHS",
    "LANE_ID",
    "N_PAIRS",
    "ANETrainerConcurrencyTicket",
    "compile_ane_trainer_concurrency_ticket",
]
