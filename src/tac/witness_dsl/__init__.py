"""Witness behavior/curriculum DSL (operator riff 2026-06-28, task #189).

A declarative, recursion+math front-end for the level-set witness training program.
It does NOT reimplement training — it COMPILES to the proven trainer CLI
(``experiments/train_levelset_witness_realized_through_R_mlx.py``), VALIDATES every
emitted flag against the trainer's real argparse (never-invent-flags, structural),
and ENFORCES the desired-behavior clauses (preserve / contain / authority) so every
A/B arm and the composed theta* program is auto-checkpointed, memory-bounded, and
authority-tagged BY CONSTRUCTION.

This is Layer 0 ("the rails") of the here->theta* bridge (DAG FEED-gv): every A/B
arm is ``BASELINE.with_lever(...)`` and theta* is the composition of the winning
fragments — each a term/relaxation of the SAME contest energy S.
"""

from tac.witness_dsl.curriculum_dsl import (
    Anneal,
    Authority,
    Contain,
    Freeze,
    Lever,
    Preserve,
    Regularizer,
    Stage,
    WitnessProgram,
    real_trainer_flags,
    BASELINE,
    PoseDecouple,
    Muon,
    DirectionalBasis,
    TauFrozen,
)
from tac.witness_dsl.campaign import (
    Arm,
    ArmResult,
    Cycle,
    compose_theta_star,
    expand_cycles,
    harvest_arm,
    plan_campaign,
    select_winners,
)

__all__ = [
    "Anneal",
    "Authority",
    "Contain",
    "Freeze",
    "Lever",
    "Preserve",
    "Regularizer",
    "Stage",
    "WitnessProgram",
    "real_trainer_flags",
    "BASELINE",
    "PoseDecouple",
    "Muon",
    "DirectionalBasis",
    "TauFrozen",
    # campaign engine
    "Arm",
    "ArmResult",
    "Cycle",
    "compose_theta_star",
    "expand_cycles",
    "harvest_arm",
    "plan_campaign",
    "select_winners",
]
