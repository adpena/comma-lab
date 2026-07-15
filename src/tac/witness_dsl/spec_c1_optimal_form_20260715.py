# SPDX-License-Identifier: MIT
"""#507 C1 OPTIMAL-FORM COMPOSITION — the three C1 legs as ONE DSL-native config.

Composes, on the official leg-A parent (``spec_v9_cgauge.compile_v9_cgauge_ideal_
mod19_sR_launch_config`` — the S_R reachability treatment over the matched
mod19 margin-saliency control):

* **Leg B (throughput)** — the joint wall-clock speed stack. The parent ALREADY
  emits the measured speed core (``--fused-r-kernel`` / ``--cache-gt-skeleton`` /
  ``--async-verdict`` / ``--verdict-batch 32 --verdict-pairs 0`` /
  ``--safe-compile-regions hosc_activation`` / ``--micro-batch-pairs 1``), and the
  ~17x custom grouped-backward + persistence-pool kernels ride the launch-command
  ``PERF_ENV_PREFIX`` (``tac.witness_dsl.typed_config:98`` — the SoT), so the only
  genuine delta this module adds from the throughput leg is the score-neutral
  component-wallclock telemetry (defaults-ON per the "off is a tracked queue" law).
  Under the operator 2026-07-15 relaxed-identity directive ("we don't care about
  drift as long as gradient is good"), the throughput leg's strict-identity
  exclusion of the custom grouped-backward VJP is **REFUTED**: its functional-
  parity proof (cosine + fp32 roundoff) is exactly the gradient-quality criterion,
  and it is the single largest measured wall-clock lever (~17.96x backward /
  ~5.5x n8 e2e) — it therefore stays ON via the perf-env prefix. The whole-step
  megakernel stays EXCLUDED on measured economics, not identity (CPU 0.79-0.83x;
  GPU only 1.12-1.21x — ``witness_fp_reorder_transform_bit_identity_wall_v1``).

* **Leg C (deep math)** — folded where a real trainer consumer exists, typed
  SLOTS where it does not (the directive's own gate (b): consumed-not-inert,
  the #417 proof):
  - FOLDED: ``PoseBlindComputeGate`` (trunk-phase compute saver; trainer flags
    ``--pose-training-compute-gate``/``--verdict-pose-gate``, task #495) and
    ``HeadOffsetSolver(mode="flip_median")`` (the #386 Hamming-optimal advisory
    decode-time head-bias arbiter — consumed at the trainer's EMA-verdict call
    site, NEVER mutates shipped/EMA/resumed weights, so it cannot introduce any
    pixel-flicker pathology by construction).
  - SLOT: the curvelet basis (``curvelet_optimal_form_receipt=...`` folds
    ``WindowedCurveletBasis`` the moment the curvelet_optimal_form_crux arm lands
    its optimal-form receipt; the unproven form is never folded — and per the
    no-Fourier-basis gate the curvelet stays opt-in, never a silent default flip).
  - SLOT (cannot fold honestly): Bregman #504 (its own DAG FEED records "no real
    trainer-consumed swept Bregman/centroid/sigma actuator is evidenced; DSL
    OWED") and the Fisher-natural trust region (``FisherNaturalSolverPolicy`` is
    argv-inert with ``activation='built_not_activated_measurement_owed'``,
    ``research_only=True``). Folding either today would be a counted-but-inert
    #417 fake. The #423 Hessian-preconditioned Newton step is a third fallen
    crack: ``laguerre_logit_offset.py`` exposes ``precondition=`` (opt-in), but
    the trainer's ``solve_head_offsets`` call passes no ``precondition`` kwarg —
    it is NOT argv-reachable; foldable the moment a trainer flag lands.

* **S_R <-> micro-batch reconciliation (leg-B's flagged conflict)** — resolved by
  TRAINER CODE, not assumption: the trainer fail-closes S_R under micro-batch
  ("--margin-saliency-reachability is not supported with --micro-batch-pairs>1
  (the batched LEVER-4 twin does not consume S_R yet)",
  ``experiments/train_levelset_witness_realized_through_R_mlx.py`` — the
  ``msal_reach and _use_micro_batch`` refusal). Under the 2026-07-15 joint
  wall-clock criterion B>1 is now admissible in principle (the batched twin's
  functional-tolerance contract IS the gradient-quality bar), but for THIS
  config it remains code-blocked until the batched LEVER-4 twin consumes S_R —
  the named fallen-crack. This module pins and VERIFIES ``--micro-batch-pairs 1``
  and refuses any parent drift away from it while S_R is emitted.

CONTAINMENT: builds + validates + compiles only ($0, pure). LAUNCH = operator-GO
through the governed launcher. MEANS: pointer moves only through a byte-closed
n600 exact row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

PROGRAM_NAME = "c1_optimal_form"
DEFAULT_OUT_DIR = "experiments/results/c1_optimal_form_20260715"

# The named additions this composition makes over the leg-A parent (fail-closed
# expected-lever accounting; amend ONLY via reviewed amendment).
C1_OPTIMAL_FORM_EXPECTED_ADDITIONS: tuple[str, ...] = (
    "pose_blind_compute_gate",
    "head_offset_solver",
    "c1_component_wallclock_telemetry",
)
C1_CURVELET_SLOT_LEVER = "basis_family::windowed_curvelet"

# Deep-math surfaces that CANNOT be folded today without violating the
# consumed-not-inert (#417) gate — each row cites the custody that blocks it and
# the exact unlock. These are typed SLOTS, not exclusions-by-taste.
C1_DEEP_MATH_SLOTS: dict[str, dict[str, str]] = {
    "bregman_504": {
        "status": "SLOT_NO_TRAINER_CONSUMER",
        "cite": ".omx/research/bregman_all_surfaces_504_DAG_FEED_20260715.md: 'DSL: OWED, "
                "because no real trainer-consumed swept Bregman/centroid/sigma actuator is "
                "evidenced'",
        "unlock": "a real trainer flag consuming the Bregman ground-metric/cheap-form step",
    },
    "fisher_natural_trust_region": {
        "status": "SLOT_BUILT_NOT_ACTIVATED",
        "cite": "tac.witness_dsl.fisher_natural_solver_policy.FisherNaturalSolverPolicy: "
                "activation='built_not_activated_measurement_owed', research_only=True, "
                "flags()=argv-inert",
        "unlock": "the owed measured A/B + a trainer consumer for the H^-1 natural step",
    },
    "hessian_preconditioned_423": {
        "status": "SLOT_NOT_ARGV_REACHABLE",
        "cite": "tac.boundary_math.laguerre_logit_offset._newton_step_from_cov(precondition=...) "
                "is opt-in (default False = legacy pinv), and the trainer's solve_head_offsets "
                "call site passes no precondition kwarg",
        "unlock": "a --head-offset-precondition trainer flag (or a preconditioned solver mode)",
    },
    "curvelet_basis": {
        "status": "SLOT_OPTIMAL_FORM_RECEIPT_OWED",
        "cite": "operator 2026-07-15: fold the moment the curvelet_optimal_form_crux arm lands "
                "its optimal form; do not fold the unproven form. Sister gate: no-Fourier-basis "
                "memory (curvelet opt-in, never a default flip).",
        "unlock": "pass curvelet_optimal_form_receipt=<existing receipt file>",
    },
}

# Leg-B dispositions re-adjudicated under the operator 2026-07-15 relaxed-identity
# + joint wall-clock-to-target directive (supersedes the throughput leg's
# strict-identity split for THIS composition; the historical strict rows are
# preserved in spec_c1_throughput_20260715.EXCLUDED_OR_HELD).
C1_SPEED_DISPOSITIONS: dict[str, dict[str, str]] = {
    "custom_grouped_backward_vjp": {
        "status": "ON_VIA_PERF_ENV",
        "verdict": "throughput-leg strict-identity exclusion REFUTED under the 2026-07-15 "
                   "criterion: functional parity (cosine + fp32 roundoff) IS the gradient-"
                   "quality bar; ~17.96x backward / ~5.5x n8 e2e measured",
        "carrier": "PERF_ENV_PREFIX TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 "
                   "(tac/witness_dsl/typed_config.py:98; launcher perf-env gate enforces)",
    },
    "persistence_pool_kernel": {
        "status": "ON_VIA_PERF_ENV",
        "verdict": "explicit-order custom kernel; part of the #432 all-ON speed set",
        "carrier": "PERF_ENV_PREFIX TAC_MLX_CUSTOM_PERSISTENCE_POOL=1",
    },
    "whole_step_megakernel_356": {
        "status": "EXCLUDED_MEASURED_ECONOMICS",
        "verdict": "stays excluded on MEASURED economics, not identity: CPU 0.79-0.83x, "
                   "GPU only 1.12-1.21x (witness_fp_reorder_transform_bit_identity_wall_v1)",
        "carrier": "n/a",
    },
    "micro_batch_pairs_gt1": {
        "status": "CODE_BLOCKED_FOR_SR",
        "verdict": "B>1 is admissible in principle under joint wall-clock (batched-twin "
                   "functional tolerances = the gradient bar) but the trainer fail-closes "
                   "S_R with micro-batch>1 (batched LEVER-4 twin does not consume S_R) — "
                   "the named fallen-crack; unlock = the batched S_R consumer",
        "carrier": "--micro-batch-pairs 1 (pinned + verified here)",
    },
    "frozen_scorer_one_thread": {
        "status": "TRAINER_NATIVE",
        "verdict": "the operator 1-thread training standard is hard-wired in the trainer via "
                   "canonical_equations.segnet_exact_forward_cpu_thread_law_20260713."
                   "SELECTED_THREADS (torch.set_num_threads at trainer startup); the throughput "
                   "leg's --training-torch-threads flag never landed and is unnecessary",
        "carrier": "trainer torch_thread_standard stage (auth-eval untouched by construction)",
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _telemetry_lever():
    """Score-neutral component-wallclock telemetry (leg B's only genuine argv delta).

    Read-only observability defaults ON per the 'off is a tracked queue' law; all
    three flags verified against the live trainer parser (never-invent-flags).

    # NO_EQUATION_NEEDED: read-only wall-clock observability; adds no loss term,
    # controller law, or score value.
    """
    from tac.witness_dsl.curriculum_dsl import Lever

    return Lever(
        "c1_component_wallclock_telemetry",
        overrides={
            "--component-wallclock-telemetry": True,
            "--component-wallclock-probe-every": 1,
            "--profile-timing": True,
        },
        notes=("leg-B telemetry: per-epoch same-function component decomposition probe + "
               "per-epoch phase split / R micro-bench; read-only, score-neutral, defaults-ON "
               "per the off-is-a-tracked-queue law"),
    )


def compile_c1_optimal_form_launch_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = 3000,
    out_dir: str = DEFAULT_OUT_DIR,
    curvelet_optimal_form_receipt: str | Path | None = None,
):
    """Compile the #507 composed C1 optimal-form config (pure / $0; never launches).

    ``curvelet_optimal_form_receipt`` is the typed curvelet SLOT: ``None`` holds the
    slot (recorded in the manifest); an EXISTING receipt file folds
    :func:`~tac.witness_dsl.curriculum_dsl.WindowedCurveletBasis`; a missing path
    fails closed.
    """
    from tac.witness_autoconfig import _crucible_v7_argv_pairs
    from tac.witness_dsl.curriculum_dsl import (
        HeadOffsetSolver,
        PoseBlindComputeGate,
        WindowedCurveletBasis,
    )
    from tac.witness_dsl.spec_v9_cgauge import (
        _typed_ideal_lever,
        compile_v9_cgauge_ideal_mod19_sR_launch_config,
    )

    # Fail-fast typed curvelet SLOT gate (before the expensive parent compile).
    receipt: Path | None = None
    if curvelet_optimal_form_receipt is not None:
        receipt = Path(curvelet_optimal_form_receipt)
        if not receipt.is_file():
            raise ValueError(
                f"{PROGRAM_NAME} curvelet-slot REFUSE: optimal-form receipt "
                f"{receipt} does not exist — the unproven curvelet form is never folded "
                "(operator 2026-07-15; fold only when curvelet_optimal_form_crux lands).")

    parent = compile_v9_cgauge_ideal_mod19_sR_launch_config(
        gt_cache_path=gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir)
    parent_pairs = dict(
        _crucible_v7_argv_pairs(tuple(parent.typed.to_program().compile_trainer_argv())))

    additions = (
        _typed_ideal_lever(PoseBlindComputeGate()),
        _typed_ideal_lever(HeadOffsetSolver(mode="flip_median", tau=1.0)),
        _typed_ideal_lever(_telemetry_lever()),
    )
    expected = tuple(lv.name for lv in parent.typed.levers) + C1_OPTIMAL_FORM_EXPECTED_ADDITIONS

    curvelet_record: dict[str, Any] = {
        "slot": C1_CURVELET_SLOT_LEVER,
        **C1_DEEP_MATH_SLOTS["curvelet_basis"],
    }
    if receipt is not None:
        additions += (_typed_ideal_lever(WindowedCurveletBasis()),)
        expected += (C1_CURVELET_SLOT_LEVER,)
        curvelet_record = {
            "slot": C1_CURVELET_SLOT_LEVER,
            "status": "FOLDED_WITH_RECEIPT",
            "receipt_path": str(receipt),
            "receipt_sha256": _file_sha256(receipt),
        }

    purpose = (
        "#507 C1 OPTIMAL-FORM COMPOSITION (operator 2026-07-15 relaxed-identity + joint "
        "wall-clock-to-target): the official leg-A S_R treatment (v9_cgauge_ideal_mod19_sR) "
        "+ the leg-B speed stack (parent-carried fused-R/cache-gt-skeleton/async-verdict/"
        "verdict-chunking + PERF_ENV ~17x custom grouped-backward, telemetry folded here) "
        "+ the consumable leg-C deep-math levers (PoseBlindComputeGate trunk-phase compute "
        "saver; flip_median advisory head-offset arbiter) with typed slots for Bregman #504 / "
        "Fisher trust-region / #423 preconditioning / curvelet (each blocked by a cited "
        "missing trainer consumer or owed receipt, never silently dropped). Pose is UNCHANGED: "
        "the R1 two-phase finisher (pose-blind trunk -> pose_finish at sigma_min_plateau); "
        "no parallel pose thread. S_R forces --micro-batch-pairs 1 BY TRAINER CODE (batched "
        "LEVER-4 twin does not consume S_R — the named fallen-crack). CONTAINMENT: compile "
        "only; LAUNCH = operator-GO. MEANS until a byte-closed n600 exact row."
    )
    typed = parent.typed.model_copy(update={
        "name": PROGRAM_NAME,
        "purpose": purpose,
        "out_dir": str(out_dir),
        "levers": tuple(parent.typed.levers) + additions,
    })
    violations = typed.validate_program()
    if violations:
        raise ValueError(
            f"{PROGRAM_NAME} DSL gate: {len(violations)} WitnessProgram.validate "
            f"violation(s): {violations[:4]}")

    got = tuple(lv.name for lv in typed.levers)
    if sorted(got) != sorted(expected):
        raise ValueError(
            f"{PROGRAM_NAME} expected-lever REFUSE: {sorted(got)} != {sorted(expected)} "
            "(a silently-dropped or duplicated leg — amend C1_OPTIMAL_FORM_EXPECTED_ADDITIONS "
            "only via reviewed amendment).")

    argv = tuple(typed.to_program().compile_trainer_argv())
    pairs = dict(_crucible_v7_argv_pairs(argv))

    # (1) Leg-A consumed-proof + the S_R<->micro-batch trainer-code reconciliation.
    if "--margin-saliency-reachability" not in pairs:
        raise ValueError(
            f"{PROGRAM_NAME} leg-A REFUSE: --margin-saliency-reachability not emitted — "
            "the S_R treatment (the leg-A parent's sole scientific delta) was lost.")
    if pairs.get("--micro-batch-pairs") != "1":
        raise ValueError(
            f"{PROGRAM_NAME} S_R/micro-batch REFUSE: --micro-batch-pairs="
            f"{pairs.get('--micro-batch-pairs')!r} with S_R emitted. The trainer fail-closes "
            "S_R under micro-batch>1 (train_levelset_witness_realized_through_R_mlx.py: "
            "'the batched LEVER-4 twin does not consume S_R yet'); B>1 is admissible under "
            "the joint wall-clock criterion ONLY once that batched consumer lands.")

    # (2) Required actuation values (each leg's consumed surface, verified on emitted argv).
    required = {
        "--mod-dim": "19",
        "--margin-saliency-weight": "1.0",
        "--verdict-batch": "32",
        "--verdict-pairs": "0",
        "--safe-compile-regions": "hosc_activation",
        "--head-offset-solver": "flip_median",
        "--head-offset-solver-tau": "1.0",
        "--component-wallclock-probe-every": "1",
    }
    mismatches = {
        flag: (pairs.get(flag), want)
        for flag, want in required.items() if pairs.get(flag) != want
    }
    for flag in (
        "--fused-r-kernel",
        "--cache-gt-skeleton",
        "--async-verdict",
        "--pose-training-compute-gate",
        "--verdict-pose-gate",
        "--component-wallclock-telemetry",
        "--profile-timing",
    ):
        if flag not in pairs:
            mismatches[flag] = ("<ABSENT>", "present")
    if mismatches:
        raise ValueError(f"{PROGRAM_NAME} actuation/custody REFUSE: {mismatches}")

    # (3) Delta-vs-parent contract: the composed argv may differ from the leg-A
    # parent ONLY on flags owned by this module's addition levers (consumed-not-
    # inert both ways: every addition reaches argv; nothing else drifts).
    def _norm(flag: str) -> str:
        # BooleanOptionalAction False compiles to the parser's --no-<flag> form;
        # normalize both directions so ownership accounting cannot miscount.
        return ("--" + flag[len("--no-"):]) if flag.startswith("--no-") else flag

    addition_flags = set()
    for lever in additions:
        addition_flags |= {_norm(flag) for flag in lever.overrides}
    absent = "<ABSENT>"
    diff = {
        flag: (parent_pairs.get(flag, absent), pairs.get(flag, absent))
        for flag in sorted(set(parent_pairs) | set(pairs))
        if flag != "--out-dir" and parent_pairs.get(flag, absent) != pairs.get(flag, absent)
    }
    stray = sorted(flag for flag in diff if _norm(flag) not in addition_flags)
    if stray:
        raise ValueError(
            f"{PROGRAM_NAME} delta-contract REFUSE: argv drifted from the leg-A parent on "
            f"non-addition flags {stray} (diff { {k: diff[k] for k in stray} }).")
    emitted_norm = {_norm(flag) for flag in pairs}
    missing_effect = sorted(
        flag for flag in addition_flags
        if flag not in emitted_norm
    )
    if missing_effect:
        raise ValueError(
            f"{PROGRAM_NAME} inert-addition REFUSE (#417): addition flags {missing_effect} "
            "did not reach the emitted argv.")

    rebound = parent._rebind_typed(typed)
    manifest = dict(rebound.dsl_program_manifest)
    manifest.update({
        "expected_active_levers": list(expected),
        "composition_contract": {
            "task": "#507 C1 optimal-form composition",
            "directive": "operator 2026-07-15 relaxed-identity + joint wall-clock-to-target",
            "leg_a_parent": "spec_v9_cgauge.compile_v9_cgauge_ideal_mod19_sR_launch_config "
                            "(official; sole scientific delta --margin-saliency-reachability)",
            "leg_b_speed": "parent-carried fused-R/cache-gt-skeleton/async-verdict/"
                           "verdict-chunk-32 + PERF_ENV custom kernels + telemetry lever here",
            "leg_c_deep_math": "PoseBlindComputeGate + HeadOffsetSolver(flip_median) folded; "
                               "Bregman/Fisher/#423/curvelet = typed slots (see "
                               "deep_math_slots)",
            "pose": "UNCHANGED R1 two-phase finisher (pose-blind trunk -> pose_finish at "
                    "sigma_min_plateau); PoseBlindComputeGate saves the blind-phase compute",
            "argv_delta_vs_parent": {flag: list(values) for flag, values in diff.items()},
        },
        "sr_microbatch_reconciliation": {
            "sr_requires_micro_batch_pairs": 1,
            "resolved_by": "TRAINER CODE (fail-closed refusal), not assumption",
            "cite": "experiments/train_levelset_witness_realized_through_R_mlx.py: "
                    "'--margin-saliency-reachability is not supported with "
                    "--micro-batch-pairs>1 (the batched LEVER-4 twin does not consume S_R yet)'",
            "b_gt1_reexamination_20260715": "admissible in principle under joint wall-clock "
                                            "(batched-twin functional tolerances = the gradient "
                                            "bar) but code-blocked for S_R configs; fallen-crack "
                                            "= the batched LEVER-4 S_R consumer",
        },
        "speed_dispositions": dict(C1_SPEED_DISPOSITIONS),
        "deep_math_slots": {
            **{k: dict(v) for k, v in C1_DEEP_MATH_SLOTS.items()},
            "curvelet_basis": dict(curvelet_record),
        },
        "held": True,
        "operator_go_required": True,
        "launch_blockers": [
            {"id": "C1_COMPOSED_BENCH_NOT_MEASURED",
             "detail": "the composed-path throughput bench receipt is BLOCKED_INPUT_CUSTODY "
                       "(.omx/research/c1_throughput_composed_bench_20260715.json), not "
                       "MEASURED_PASS — measure sec/epoch + peak RSS on the real cache before GO"},
            {"id": "C1_SR_SIDECAR_CUSTODY",
             "detail": "the 'sR' cache member (or <stem>_sR.npz sidecar) must exist at launch "
                       "(tools/precompute_sR_reachability.py); the trainer fails closed without it"},
        ],
    })
    constants = dict(rebound.constants_manifest)
    constants["head_offset_solver"] = {
        "value": "flip_median",
        "equation_id": "laguerre_ot_head_offset_20260709",
        "ladder_class": "derived_at_config",
        "fallback_used": False,
        "inputs": {
            "objective": "d_seg is Hamming; the L1-optimal 1-D threshold is the per-edge "
                         "flip-margin median (S1), not an OT mass-match",
            "ot_newton_status": "MEASURED-worse at mod32cap ep650 (area objective falsified)",
            "advisory": "decode-time observer; never mutates shipped/EMA/resumed weights",
        },
        "note": "#386 flip_median advisory arbiter folded per the 2026-07-15 directive; the "
                "n600 realized-through-R delta rows it emits ARE the owed A/B instrument.",
    }
    constants["c1_component_wallclock_probe_every"] = {
        "value": 1,
        "equation_id": None,
        "ladder_class": "derived_at_config",
        "fallback_used": False,
        "inputs": {"trainer_default": 1},
        "note": "score-neutral observability cadence (trainer's own designed default; "
                "NO_EQUATION_NEEDED — read-only telemetry).",
    }
    from tac.witness_autoconfig import CrucibleV7LaunchConfig

    return CrucibleV7LaunchConfig(
        typed=rebound.typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance=dict(rebound.schedule_governance),
    )


__all__ = [
    "C1_CURVELET_SLOT_LEVER",
    "C1_DEEP_MATH_SLOTS",
    "C1_OPTIMAL_FORM_EXPECTED_ADDITIONS",
    "C1_SPEED_DISPOSITIONS",
    "DEFAULT_OUT_DIR",
    "PROGRAM_NAME",
    "compile_c1_optimal_form_launch_config",
]
