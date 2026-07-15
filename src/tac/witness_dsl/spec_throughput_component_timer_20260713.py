# SPDX-License-Identifier: MIT
"""Typed, held n24 CE-equivalent D-A component-timer configurations.

The two variants are a matched throughput A/B, not score arms:

* ``async_current`` keeps the incumbent async CPU verdict worker; and
* ``solo_control`` removes only that async overlap.

Both freeze unified ``L_tau`` at tau=1, which is exactly cross entropy, disable
curriculum and every inherited stage/controller, zero every non-CE loss or
regularizer, and run four real n24 epochs with two verdict pairs.  The v7.5.2
geometry/performance substrate remains, but its score-moving curriculum levers
do not: this is a one-stage component timer.  Compilation is pure and cannot
launch; the governed launcher remains the sole actuation surface.
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

_NON_CE_BASE_FLAGS = frozenset({
    "--curriculum-min-stage-epochs",
    "--muon-start-event",
    "--seg-chroma-boundary-start-event",
    "--tau-advance-mode",
    "--seed-island-eased",
    "--stage-transition-reset-moments",
    "--stage-transition-rewarmup-epochs",
    "--stage-transition-rewarmup-floor",
    "--stage-transition-rewarmup-shape",
    "--muon-lr",
    "--muon-lr-final-frac",
    "--muon-momentum",
    "--muon-ns-steps",
    "--muon-warm-start-momentum",
    # Pose-carrier value sub-flags: the CE-only timer sets --w-pose 0.0, and the
    # trainer fail-closes ("--pose-carrier requires --w-pose > 0"). Strip the
    # carrier's value sub-flags here and disable the boolean below so the CE-only
    # timing probe boots. (measured-runnability defect, dry-start 2026-07-14.)
    "--pose-carrier-source",
    "--pose-carrier-residual-mode",
})


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
        # Compile the parent at its sealed feasible budget, then derive the
        # four-epoch single-stage child below.  The class guard must reject the
        # impossible parent itself; the timer is not allowed to bypass it.
        epochs=3000,
        out_dir=out_dir,
        self_orient=False,
        amber=True,
    )

    # Keep the v7.5.2 representation/performance substrate, but remove every
    # schedule cap and controller that could engage or fail a four-epoch boot.
    # Defaults are intentionally used for inert stages; no start epoch is
    # emitted by this single-stage program.
    timer_base = {
        flag: value
        for flag, value in parent.typed.base.items()
        if not flag.endswith(("-start-epoch", "-start-event"))
        and flag not in _NON_CE_BASE_FLAGS
    }
    timer_base.update({
        "--curriculum": False,
        "--curriculum-event-triggered": False,
        "--curriculum-nucleus-guard": False,
        "--curriculum-reanchor-levers": False,
        "--dseg-aware-taper": False,
        "--seed-islands": False,
        "--witness-alone-island-loss": False,
        "--w-pose": 0.0,
        # CE-only timer carries no pose objective; the pose carrier requires
        # w_pose > 0, so disable it (BooleanOptional -> emits --no-pose-carrier).
        "--pose-carrier": False,
    })

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
        "--seg-form-unify-tau": True,
        "--w-pose": 0.0,
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
        "epochs": EPOCHS,
        "out_dir": str(out_dir),
        "purpose": purpose,
        "temp": temp,
        "base": timer_base,
        "stages": (),
        "regularizers": (),
        "levers": (probe_lever,),
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
        "--w-pose": "0.0",
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
        "--no-curriculum",
        "--no-curriculum-event-triggered",
        "--no-curriculum-nucleus-guard",
        "--no-curriculum-reanchor-levers",
        "--no-dseg-aware-taper",
        "--no-seed-islands",
        "--seg-form-unify-tau",
        "--component-wallclock-telemetry",
        "--stage-checkpoints",
        "--no-witness-alone-island-loss",
        "--no-pose-carrier",
    ):
        if flag not in pairs:
            mismatch[flag] = (pairs.get(flag), "present")
    emitted_stage_caps = sorted(flag for flag in pairs if flag.endswith("-start-epoch"))
    if emitted_stage_caps:
        mismatch["stage_start_epochs"] = (emitted_stage_caps, "absent")
    emitted_stage_events = sorted(flag for flag in pairs if flag.endswith("-start-event"))
    if emitted_stage_events:
        mismatch["stage_start_events"] = (emitted_stage_events, "absent")
    if len(typed.levers) != 1 or typed.levers[0].name != probe_lever.name:
        mismatch["inherited_levers"] = (
            [lever.name for lever in typed.levers], [probe_lever.name]
        )
    async_flag = "--async-verdict" if async_on else "--no-async-verdict"
    if async_flag not in pairs:
        mismatch[async_flag] = (pairs.get(async_flag), "present")
    if mismatch:
        raise ValueError(f"component-timer typed actuation REFUSE: {mismatch}")

    constants = {
        key: value
        for key, value in rebound.constants_manifest.items()
        if key != "polyak_finisher_start_epoch"
    }
    # (#507 coordinator fix, 2026-07-15) UNBLOCK the timer family under the #406 DSL-compile-hash
    # gate: the v752 parent lineage's custodied LawRef still compiles hosc_beta_end=10.0 (the v6.3
    # pin) while the emitted flag was rephased to 3.177 — the self-recompile catches the mismatch
    # and refuses rc=8 (wallclock_burndown_build_20260715.md §2c). Apply the SAME argv-as-single-
    # scalar-owner reconciliation the v9 lineage uses: the emitted argv value wins; the inherited
    # 10.0 is preserved on the row as ``inherited_manifest_value_replaced`` (the CLAUDE.md #351
    # historical_non_authorizing custody — history, never authority).
    from tac.witness_dsl.spec_v9_cgauge import _derive_manifest_from_emitted_argv
    constants = _derive_manifest_from_emitted_argv(
        constants, tuple(typed.to_program().compile_trainer_argv()))
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
            "entropy/logit-adjust/eikonal/length/weight-decay/pose are explicit zero; "
            "curriculum and all inherited stages/controllers are disabled"
        ),
        "variant": variant,
        "matched_control_program": SOLO_PROGRAM if async_on else ASYNC_PROGRAM,
    })
    return replace(
        rebound,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance={
            flag: governance
            for flag, governance in rebound.schedule_governance.items()
            if flag in pairs
        },
    )


__all__ = [
    "ASYNC_PROGRAM",
    "EPOCHS",
    "N_PAIRS",
    "SOLO_PROGRAM",
    "VARIANTS",
    "compile_throughput_component_timer_ticket",
]
