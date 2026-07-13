# SPDX-License-Identifier: MIT
"""Typed, held n24 CE-equivalent D-A component-timer configurations.

The two variants are a matched throughput A/B, not score arms:

* ``async_current`` keeps the incumbent async CPU verdict worker; and
* ``solo_control`` removes only that async overlap.

Both freeze unified ``L_tau`` at tau=1, which is exactly cross entropy, zero
every non-CE loss/regularizer that could be active in the four-epoch window,
and run four real n24 epochs with two verdict pairs.  They inherit the v7.5.2
event/cap governance and per-stage resume contract, so the launcher-path
schedule-provenance gate has zero naked epochs.  Compilation is pure and
cannot launch; the governed launcher remains the sole actuation surface.
"""

from __future__ import annotations

from dataclasses import replace

from tac.witness_autoconfig import (
    CrucibleV7LaunchConfig,
    _crucible_v7_argv_pairs,
    compile_crucible_v752_launch_config,
)
from tac.witness_dsl.typed_config import TypedLever

ASYNC_PROGRAM = "throughput_component_timer_async_20260713"
SOLO_PROGRAM = "throughput_component_timer_solo_20260713"
VARIANTS = ("async_current", "solo_control")
N_PAIRS = 24
EPOCHS = 4


def compile_throughput_component_timer_ticket(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n24.npz",
    *,
    num_pairs: int = N_PAIRS,
    epochs: int = EPOCHS,
    out_dir: str = "experiments/results/throughput_component_timer_ticket_20260713",
    variant: str = "async_current",
) -> CrucibleV7LaunchConfig:
    """Return a fail-closed, operator-GO-only n24 timing ticket."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown timer variant {variant!r}; expected one of {VARIANTS}")
    if int(num_pairs) != N_PAIRS or int(epochs) != EPOCHS:
        raise ValueError(
            "throughput component timer is sealed to n24 real states and four epochs; "
            "derive a new governed ticket instead of silently resizing it"
        )
    async_on = variant == "async_current"
    program_name = ASYNC_PROGRAM if async_on else SOLO_PROGRAM
    parent = compile_crucible_v752_launch_config(
        gt_cache_path,
        num_pairs=N_PAIRS,
        epochs=EPOCHS,
        out_dir=out_dir,
        self_orient=False,
        amber=True,
    )

    probe_overrides = {
        "--async-verdict": async_on,
        "--component-wallclock-telemetry": True,
        "--component-wallclock-probe-every": 1,
        "--eval-every": 1,
        "--verdict-pairs": 2,
        "--ckpt-every": 1,
        # The standing loss-term observer recomputes terms.  It is score-neutral
        # but would contaminate this component-timing treatment, so the matched
        # A/B disables it explicitly on both arms.
        "--loss-term-log-every": -1,
        # Minimal active objective for the four-epoch timing treatment.  The
        # v7.5.2 parent supplies exact vehicle geometry/perf wiring, but its
        # auxiliary losses are not part of this CE component disambiguator.
        # Zero them explicitly rather than relying on distant start epochs or
        # warm-up arithmetic.  Pose is already event-gated to epoch 726 by the
        # parent, hence its effective weight is zero throughout epochs 1..4.
        "--seg-loss": "ce",
        "--persistence-loss-weight": 0.0,
        "--persistence-recall-weight": 0.0,
        "--amplify-weight": 0.0,
        "--seg-chroma-boundary-weight": 0.0,
        "--seg-temporal-screw-weight": 0.0,
        "--weight-entropy-penalty-lambda": 0.0,
        "--logit-adjust-loss-tau": 0.0,
        "--eikonal-weight": 0.0,
        "--length-weight": 0.0,
        "--weight-decay": 0.0,
        "--witness-alone-island-loss": False,
    }
    probe_lever = TypedLever(
        name=f"throughput_component_timer_{variant}",
        overrides=probe_overrides,
        notes=(
            "MEANS-only D-A measurement protocol; last-wins over operational defaults so "
            "checkpoint/verdict cadence and the async matched arm are explicit"
        ),
    )
    temp = parent.typed.temp.model_copy(update={"end": parent.typed.temp.start})
    purpose = (
        "HELD operator-GO-only n24 real-state D-A component timer; unified L_tau frozen "
        "at tau=1 (CE exact); backward timers inclusive; "
        + ("incumbent async verdict overlap" if async_on else "matched no-async contention control")
        + "; MEANS only, no score claim"
    )
    typed = parent.typed.model_copy(update={
        "name": program_name,
        "out_dir": str(out_dir),
        "purpose": purpose,
        "temp": temp,
        "levers": (*tuple(parent.typed.levers), probe_lever),
    })
    rebound = parent._rebind_typed(typed)
    pairs = dict(_crucible_v7_argv_pairs(typed.to_program().compile_trainer_argv()))
    expected = {
        "--softmax-temp-start": "1.0",
        "--softmax-temp-end": "1.0",
        "--component-wallclock-probe-every": "1",
        "--eval-every": "1",
        "--verdict-pairs": "2",
        "--ckpt-every": "1",
        "--loss-term-log-every": "-1",
        "--seg-loss": "ce",
        "--persistence-loss-weight": "0.0",
        "--persistence-recall-weight": "0.0",
        "--amplify-weight": "0.0",
        "--seg-chroma-boundary-weight": "0.0",
        "--seg-temporal-screw-weight": "0.0",
        "--weight-entropy-penalty-lambda": "0.0",
        "--logit-adjust-loss-tau": "0.0",
        "--eikonal-weight": "0.0",
        "--length-weight": "0.0",
        "--weight-decay": "0.0",
    }
    mismatch = {flag: (pairs.get(flag), value) for flag, value in expected.items()
                if pairs.get(flag) != value}
    for flag in (
        "--seg-form-unify-tau",
        "--component-wallclock-telemetry",
        "--stage-checkpoints",
        "--no-witness-alone-island-loss",
    ):
        if flag not in pairs:
            mismatch[flag] = (pairs.get(flag), "present")
    async_flag = "--async-verdict" if async_on else "--no-async-verdict"
    if async_flag not in pairs:
        mismatch[async_flag] = (pairs.get(async_flag), "present")
    if mismatch:
        raise ValueError(f"component-timer typed actuation REFUSE: {mismatch}")

    constants = dict(rebound.constants_manifest)
    constants["softmax_temp_end"] = {
        "value": 1.0,
        "equation_id": "cross_entropy_is_unified_ltau_at_tau_one_v1",
        "ladder_class": "derived_at_config",
        "fallback_used": False,
        "inputs": {"softmax_temp_start": 1.0, "epochs": EPOCHS},
        "note": "tau start=end=1; L_tau equals cross entropy exactly for the bounded timer",
    }
    constants["throughput_probe_eval_every"] = {
        "value": 1,
        "equation_id": "async_overlap_and_inclusive_vjp_throughput_v1",
        "ladder_class": "operator_sealed_measurement_protocol",
        "fallback_used": False,
        "note": "four n24 epochs, two verdict pairs, matched async/solo arms",
    }
    manifest = dict(rebound.dsl_program_manifest)
    manifest.update({
        "held": True,
        "operator_go_required": True,
        "research_only": True,
        "score_claim": False,
        "pointer_moved": False,
        "measurement_axis": "[n24 real-state macOS-MLX wall-clock advisory] NON-PROMOTABLE",
        "n600_transfer_rule": (
            "component ms/pair may be reported as n24-linear-extrapolation x25 only, never n600 measured"
        ),
        "backward_semantics": (
            "teacher_backward_s and witness_backward_s are inclusive of required forward; "
            "incremental VJP = inclusive - forward"
        ),
        "active_loss_epochs_1_through_4": (
            "seg-form-unify-tau at tau=1 equals CE; persistence/amplify/chroma/screw/"
            "entropy/logit-adjust/eikonal/length/weight-decay are explicit zero; pose "
            "finish remains event-gated beyond the held four-epoch window"
        ),
        "variant": variant,
        "matched_control_program": SOLO_PROGRAM if async_on else ASYNC_PROGRAM,
    })
    return replace(
        rebound,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
    )


__all__ = [
    "ASYNC_PROGRAM",
    "EPOCHS",
    "N_PAIRS",
    "SOLO_PROGRAM",
    "VARIANTS",
    "compile_throughput_component_timer_ticket",
]
