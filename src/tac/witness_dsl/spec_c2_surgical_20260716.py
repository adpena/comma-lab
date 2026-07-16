# SPDX-License-Identifier: MIT
"""c2_surgical_warm — the doctrine-minimal SURGICAL composition (operator 2026-07-16).

Operator binding (verbatim): "Continue with all and keep Kolmogorov philosophy and
projection and realization in mind as ideal and let's only train the absolute least
amount and most surgical targets possible."
(memory ``train_least_surgical_kolmogorov_projection_realization_doctrine_20260716``)

THE COMPOSITION: warm-start the best held trunk — the FROZEN **mod32cap EMA-best
ep650** checkpoint (verdict d_seg 0.003366; 0.003146 n600 re-measured through the
exact contest R + frozen CPU SegNet, ``c2_witness_own_decomp_20260716.md``) — with
weights-only resume, and train ONLY the two measured surgical targets its own
residual decomposition licenses:

* **(a) Road-Lane sub-pixel boundary appearance-phase** — 90.6 % of the witness
  residual is 1-px edge FLICKER; the Road-Lane pair alone is 66.0 % (witness-own
  decomp §1). Flat-amplitude carriers are MEASURED EXHAUSTED on this trunk (§3:
  every flat band variant net-negative, β→0 bracket ≈ 0) ⇒ only the L86 phase
  stack crosses: T1 ``PhaseAdvectionConsistency`` (#424) + the #360
  ``MarginBandSatisficing`` hinge, both engaging at ONE post-re-anchor boundary.
* **(b) the joint pose finish** — the trunk is pose-blind (w_pose=0 lineage); the
  L68 photometric wall says only JOINT descent crosses; ``PoseFinishConditioningGate``
  (sigma_min_plateau, banked-R1 dxi fallback, never blocks) + ``PoseBlindComputeGate``.

Everything else is SOLVED / SEEDED / DROPPED per the doctrine ledger (decision memo
``.omx/research/c2_surgical_composition_20260716.md``): the argmax partition trunk is
NOT re-trained (warm start holds it), the affine-head finisher stays a SOLVE slot
(#341 full-P GN — build owed; the wired ``HeadOffsetSolver(flip_median)`` advisory
arbiter measures the owed rows), rate carriers stay at the byte-close surface
(#425 ``--phase-carrier`` on the byte-close tool once the phase tail engages).

WHY WARM-START mod32cap AND NOT THE v9 LINE (measured, not assumed):
* the surgical-target map WAS MEASURED ON THIS EXACT CHECKPOINT (witness-own decomp);
* v9/c1 architecture (mod-dim 19, legacy_fourier bank WITHOUT --self-orient) is
  weight-shape-INCOMPATIBLE with the checkpoint (mod-dim 32 + self-orient dir feats
  change code/FiLM/first-layer shapes — DERIVED from the config diff);
* the v752 factory carries a lever stack (lane-band render, ladder islands, dash-comb)
  the checkpoint never trained under — render-side drift would invalidate the held
  d_seg at resume epoch; so the base here is the checkpoint's OWN launch config
  (config of record: ``levelset_n600_witness_mod32cap_20260706T115554Z/launch.sh``),
  flag-for-flag, with each deviation carrying explicit provenance below.

SCHEDULE CONTINUITY (the trainer's own machinery, not a rephasing hack):
``--anneal-epochs 1000`` pins the tau/beta/LR schedules to the ORIGINAL 1000-epoch
plant (the flag's documented warm-start contract), so the resume epoch (651) sees
exactly the checkpoint's schedule state and the extension (ep1001-1400) holds the
end values. ``--epochs 1400`` is the RUN length only.

DEVIATIONS FROM THE CHECKPOINT CONFIG (each provenanced; everything else verbatim):
* ``--l7-start-epoch == epochs + 1`` (was 1001): l7 is a MEASURED DEFECT (CLAUDE.md capstone:
  "demote it"); a naive epoch-extension to 1400 would have RE-ACTIVATED it at 1001
  (adverse finding A3 in the memo). start = epochs + 1 = the trainer's TRUE
  "l7 never runs" form (the loop is range(start, epochs+1) INCLUSIVE — start == epochs
  would run l7 on the final epoch; adversarial-review fix 2026-07-16).
* ``--w-pose 1.0`` (was 0): the pose-finish phase weight; the trunk phase stays
  pose-blind via the two-phase gate + PoseBlindComputeGate (compute saver).
* ``--anneal-epochs 1000`` (new): schedule continuity (above).
* the surgical levers themselves (phase/satisfice/pose/head-offset/telemetry/speed),
  every one a registered ``Lever`` factory — no hand flags.

EVENT-SENSOR HONESTY (adverse finding A2): the c1 ``label_floor`` start-event is
DEAD on this warm path — its floor band [0.00496, 0.00700] sits ABOVE the trunk's
resume d_seg (~0.0034), so the sensor's precondition (approaching the floor from
above in a label-smooth descent) never obtains. The phase stage therefore engages
on an EPOCH boundary (700 = resume+~50 re-anchor epochs) with this rationale
recorded in schedule_governance; this factory REFUSES a config that emits
``--seg-phase-advect-start-event`` (fail-closed against silently re-inheriting the
dead sensor). Warm-path sensor recalibration is a NAMED owed item, not a silent gap.

CONTAINMENT: builds + validates + compiles only ($0, pure). LAUNCH = operator-GO
through the governed launcher (registration in tools/launch_witness_run.py is OWED
at GO time — the launcher is untouchable to this landing). MEANS: pointer 0.19108
moves only through a byte-closed n600 exact row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROGRAM_NAME = "c2_surgical_warm"
DEFAULT_OUT_DIR = "experiments/results/c2_surgical_warm_20260716"

#: the config of record for the warm-start trunk (verdict d_seg 0.003366 @ ep650;
#: 0.003146 n600 through-R re-measure, c2_witness_own_decomp_20260716.md §preamble).
WARM_START_CKPT = ("experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/"
                   "levelset_witness_ema_BEST.npz")
WARM_START_RUN_DIR = "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z"

#: resume epoch of the checkpoint (ep650 best; natural continuation = 651) and the
#: single surgical engage boundary (~50 re-anchor epochs after resume: fresh AdamW
#: moments need a short window before new loss terms land on the trunk).
RESUME_EPOCH = 651
SURGICAL_ENGAGE_EPOCH = 700
POSE_BACKSTOP_EPOCH = 1000
RUN_EPOCHS = 1400
ORIGINAL_SCHEDULE_EPOCHS = 1000  # the checkpoint's plant; --anneal-epochs pins to it

C2_SURGICAL_EXPECTED_LEVERS: tuple[str, ...] = (
    "c2_warm_start_weights_only",
    "phase_advection_consistency",
    "margin_band_satisficing",
    "tie_locus_displacement",
    "pose_finish_conditioning_gate",
    "pose_blind_compute_gate",
    "head_offset_solver",
    "c2_speed_stack",
    "c2_component_wallclock_telemetry",
)

#: doctrine dispositions this config does NOT train (the Kolmogorov test applied);
#: full ledger in the decision memo.
C2_SOLVE_SEED_DROP_SLOTS: dict[str, dict[str, str]] = {
    "terminal_head_gn_341": {
        "status": "SLOT_SOLVE_BUILD_OWED",
        "cite": "solve_dont_train_inventory_20260709 row 1: full-P GN/CG head finisher is the GO "
                "but NOT landed in-trainer; K<P subset solve is a measured FORMULATION NO-GO "
                "(+5.1% n600 overfit)",
        "unlock": "the in-trainer full-P damped Newton-CG head solve (fires at the tau-best basin, "
                  "LM rho in [0.8,1.2] re-verified on the CURRENT checkpoint)",
    },
    "ot_head_offsets_288": {
        "status": "SLOT_BUILT_UNWIRED",
        "cite": "laguerre_logit_offset.py damped-Newton OT solver BUILT but the trainer wires only "
                "the Menon heuristic init; ot_newton mode MEASURED-worse at mod32cap ep650 "
                "(laguerre_ot_head_offset_v1) — flip_median advisory arbiter is what this config "
                "consumes instead",
        "unlock": "a trainer consumer for the solved b_c fold (byte-free decode-time)",
    },
    "flat_band_carriers": {
        "status": "DROPPED_MEASURED_DEAD_ON_THIS_TRUNK",
        "cite": "witness-own decomp §3 flat-amplitude EXHAUSTION: every flat palette-delta band "
                "variant net-negative on this checkpoint (beta->0 bracket ~= 0); "
                "verdict_scope=FORMULATION (flat post-hoc composites)",
        "unlock": "n/a for this config (the palette/necessity vehicle keeps the carrier)",
    },
    "phase_residual_carrier_425": {
        "status": "BYTE_CLOSE_SURFACE_NOT_TRAINING",
        "cite": "tools/levelset_byte_close_and_eval.py --phase-carrier (WIRED, default OFF on the "
                "tool): the archive SHAPE for the phase zero-mode; post-engage checkpoints MUST "
                "byte-close WITH it (c1 manifest byte_close_contract, inherited here verbatim)",
        "unlock": "n/a (contract, not a blocker)",
    },
}


def _file_exists_blocker(path: str, blocker_id: str, detail: str,
                         blockers: list, evidence: dict, key: str) -> None:
    p = Path(path)
    if p.is_file():
        evidence[key] = {"status": "CLEAR", "path": str(p), "bytes": p.stat().st_size}
    else:
        blockers.append({"id": blocker_id, "detail": detail})


def compile_c2_surgical_warm_launch_config(
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = RUN_EPOCHS,
    out_dir: str = DEFAULT_OUT_DIR,
):
    """Compile the c2_surgical_warm config (pure / $0; never launches).

    Returns a :class:`tac.witness_autoconfig.CrucibleV7LaunchConfig` (the launcher-facing
    duck type). Fail-closed: DSL ``WitnessProgram.validate`` (never-invent-flags against
    the REAL trainer parser) + expected-lever accounting + required-actuation argv checks
    + the dead-sensor refusal + compile-time derived launch blockers (checkpoint custody,
    dry-start bench receipt).
    """
    import json as _json

    from tac.local_acceleration.scorer_throughput_gate import derive_wall_clock_budget_days
    from tac.witness_autoconfig import CrucibleV7LaunchConfig, _crucible_v7_argv_pairs
    from tac.witness_dsl.curriculum_dsl import (
        HeadOffsetSolver,
        Lever,
        MarginBandSatisficing,
        PhaseAdvectionConsistency,
        PoseBlindComputeGate,
        PoseFinishConditioningGate,
        TieLocusDisplacement,
    )
    from tac.witness_dsl.typed_config import (
        Provenanced,
        ProvenanceClass as _PC,
        TypedAnneal,
        TypedWitnessConfig,
        build_launch_manifest,
    )

    # ── the checkpoint's own config, flag-for-flag (config of record: launch.sh of
    #    the mod32cap run; every deviation is listed in the module banner + manifest). ──
    base: dict[str, Any] = {
        "--seed": 0,
        "--async-verdict": True,
        "--eval-every": 25,
        "--verdict-pairs": 0,
        "--curriculum": True,
        "--tau-softplus-start-epoch": 300,
        "--tau-softplus-tau": 0.3,
        # DEVIATION: l7 NEVER runs (measured defect). start = epochs + 1 is the TRUE never-runs
        # form (adversarial-review fix 2026-07-16): the trainer's epoch loop is
        # range(start_epoch, epochs+1) INCLUSIVE, so start == epochs RUNS l7 on the final epoch —
        # the trainer's own documented off-by-one (~L15646: "l7_start == epochs is the L1
        # off-by-one (l7 WOULD run on the final epoch) — the fresh config uses epochs+1"). The
        # mod32cap config of record itself parks l7 at 1001 with epochs=1000 ("TRUE never").
        # The DSL epoch-budget gate now exempts EXACTLY the l7 == epochs+1 parking form
        # (same-review NARROW class-fix; any other past-budget value still refuses).
        "--l7-start-epoch": int(epochs) + 1,
        "--muon-start-epoch": 726,
        "--muon-lr": 0.002,
        "--muon-momentum": 0.95,
        "--muon-ns-steps": 5,
        "--stage-transition-rewarmup-epochs": 8,
        "--stage-transition-rewarmup-floor": 0.1,
        "--stage-transition-rewarmup-shape": "linear",
        "--stage-transition-reset-moments": True,
        "--w-seg": 100,
        "--w-pose": 1.0,                        # DEVIATION: pose-finish phase weight
        "--score-domain-loss": True,
        "--mod-dim": 32,
        "--hidden-dim": 96,
        "--n-hidden": 4,
        "--activation": "hosc",
        "--hosc-beta": 1.0,
        "--hosc-beta-end": 4.0,
        "--hosc-beta-anneal": "linear",
        "--hosc-omega": 1.0,
        "--siren-init": True,
        "--softmax-temp-start": 1.0,
        "--softmax-temp-end": 0.05,
        "--self-orient": True,
        "--n-dir-freqs": 4,
        "--freq-across": 32,
        "--freq-along": 8,
        "--reorient-every": 50,
        "--max-bank-freq": 64,
        "--chroma": True,
        "--palette-anchor": True,
        "--eikonal-weight": 0,
        "--length-weight": 0.001,
        "--render-h": 384,
        "--render-w": 512,
        "--accum-pairs": 8,
        "--grad-clip": 1.0,
        "--per-group-grad-clip": True,
        "--ema-decay": 0.997,
        "--structured-init": True,
        "--structured-init-include-lane": True,
        "--ckpt-every": 25,
        "--stage-checkpoints": True,
        # DEVIATION (schedule continuity): pin tau/beta/LR anneals to the ORIGINAL plant.
        "--anneal-epochs": int(ORIGINAL_SCHEDULE_EPOCHS),
    }

    warm_start = Lever(
        "c2_warm_start_weights_only",
        overrides={
            "--resume-from": WARM_START_CKPT,
            "--warm-start-weights-only": True,
        },
        notes=("warm-start the mod32cap EMA-best ep650 trunk (verdict d_seg 0.003366 / "
               "0.003146 n600 through-R): WEIGHTS ONLY (EMA shadow preferred via the flag's "
               "auto resume-model-from=ema), fresh AdamW moments, re-seeded spike guard, "
               "lever drift auto-allowed (an intentional surgical re-treatment). Natural "
               "epoch continuation = 651."),
    )

    speed = Lever(
        "c2_speed_stack",
        overrides={
            "--fused-r-kernel": True,        # bit-identical fused R (L70)
            "--cache-gt-skeleton": True,     # score-neutral GT skeleton cache
            "--verdict-batch": 32,           # the #205 OOM law (forbidden-pattern section)
            "--safe-compile-regions": "hosc_activation",
        },
        notes=("the measured joint wall-clock speed core carried from the c1 composition "
               "(spec_c1_optimal_form C1_SPEED_DISPOSITIONS); the ~17x custom grouped-backward "
               "+ persistence-pool kernels ride PERF_ENV_PREFIX (typed_config SoT; launcher "
               "perf-env gate enforces)."),
    )

    telemetry = Lever(
        "c2_component_wallclock_telemetry",
        overrides={
            "--component-wallclock-telemetry": True,
            "--component-wallclock-probe-every": 1,
            "--profile-timing": True,
        },
        notes=("read-only score-neutral component wall-clock telemetry, defaults-ON per the "
               "off-is-a-tracked-queue law (same lever as c1's; also the instrument that reads "
               "the grad-clip frac_clipped stream against the mod32cap magnitude law — the "
               "autoclip/normalize magnitude-law A/B stays a NAMED open lever, see manifest)."),
    )

    levers = (
        warm_start,
        PhaseAdvectionConsistency(weight=0.4, start_epoch=SURGICAL_ENGAGE_EPOCH),
        MarginBandSatisficing(weight=0.2, start_epoch=SURGICAL_ENGAGE_EPOCH),
        # AMENDMENT (coordinator directive 2026-07-16, lane-channel completeness verdict
        # 477ec610c5 §5/§7): T1 phase_advection is birth-SILENT but leaves a MEASURED coverage
        # gap — 26.3% of candidate straddle pixels (354 lane-adjacent px/frame; GT island churn
        # ~50%/step) get NO phase supervision without Force-3 subpix. Co-emit the ALREADY-BUILT
        # #360 lever at the SAME single engage boundary (factory-default weight/v_band — the
        # #360 build's own designed values, no unmeasured constants folded; the λ_RL≈2x per-pair
        # prior + event-fallback weight variant stay recorded duty-to-measure). pa_flipmass edge
        # weighting reads the MEASURED reports/pa_edge_weights.json artifact (present; the
        # factory falls back uniform+LOUD-WARN if absent, never a guess).
        TieLocusDisplacement(start_epoch=SURGICAL_ENGAGE_EPOCH),
        PoseFinishConditioningGate(backstop_epoch=POSE_BACKSTOP_EPOCH),
        PoseBlindComputeGate(),
        HeadOffsetSolver(mode="flip_median", tau=1.0),
        speed,
        telemetry,
    )

    purpose = (
        "c2_surgical_warm (operator 2026-07-16 train-least/Kolmogorov doctrine): warm-start the "
        "best held trunk (mod32cap EMA-best ep650, d_seg 0.003146 n600 through-R — the exact "
        "checkpoint the witness-own residual decomposition measured) and train ONLY the two "
        "licensed surgical targets: (a) Road-Lane sub-pixel appearance-phase (66% of the residual; "
        "T1 phase-advection #424 + #360 satisficing hinge at one engage boundary ep700), (b) the "
        "joint pose finish (sigma_min_plateau conditioning gate, banked-R1 fallback, backstop "
        "ep1000). Everything else SOLVED/SEEDED/DROPPED per the doctrine ledger "
        "(.omx/research/c2_surgical_composition_20260716.md). Schedule continuity via "
        "--anneal-epochs 1000 (the original plant); l7 excluded (measured defect); run length "
        "1400 (~750 trained epochs, ~1.1-1.9 days at the lineage's measured ~121-180 s/ep vs "
        "12.5 days for the from-scratch c1). CONTAINMENT: compile only; LAUNCH = operator-GO. "
        "MEANS until a byte-closed n600 exact row."
    )

    typed = TypedWitnessConfig(
        name=PROGRAM_NAME,
        out_dir=str(out_dir),
        gt_cache=str(gt_cache_path),
        num_pairs=int(num_pairs),
        epochs=int(epochs),
        wall_clock_budget_days=Provenanced(
            value=round(derive_wall_clock_budget_days(int(epochs)), 3),
            provenance=_PC.DERIVED_AT_CONFIG, unit="days",
            source="scorer_throughput_gate.derive_wall_clock_budget_days"
                   "(anchor RUN1_MEASURED_MIN_PER_EP x epochs x WALL_CLOCK_SLACK_FACTOR); "
                   "NB the budget covers the RUN length; only ~(epochs-651) epochs are actually "
                   "trained on the warm path"),
        mlx_device="gpu",
        temp=TypedAnneal(
            start=Provenanced(value=1.0, provenance=_PC.MEASURED_ANCHOR, unit="tau",
                              source="mod32cap lineage anneal start (config of record)"),
            end=Provenanced(value=0.05, provenance=_PC.MEASURED_ANCHOR, unit="tau",
                            source="mod32cap lineage tau_end (config of record; the checkpoint's "
                                   "weights are conditioned on THIS schedule — the P-TAU2 knee "
                                   "0.31 finding is config-conditional to the v9 lineage and is "
                                   "recorded as adverse finding A4, not silently adopted)"),
        ),
        base=base,
        levers=tuple(  # typed wrappers: reuse the c1 pattern (ideal-lever adapter)
            _typed(lv) for lv in levers
        ),
        purpose=purpose,
    )

    violations = typed.validate_program()
    if violations:
        raise ValueError(
            f"{PROGRAM_NAME} DSL gate: {len(violations)} WitnessProgram.validate "
            f"violation(s): {violations[:4]}")

    got = tuple(lv.name for lv in typed.levers)
    if sorted(got) != sorted(C2_SURGICAL_EXPECTED_LEVERS):
        raise ValueError(
            f"{PROGRAM_NAME} expected-lever REFUSE: {sorted(got)} != "
            f"{sorted(C2_SURGICAL_EXPECTED_LEVERS)}")

    argv = tuple(typed.to_program().compile_trainer_argv())
    pairs = dict(_crucible_v7_argv_pairs(argv))

    # ── required actuation values (consumed-not-inert, verified on emitted argv). ──
    required = {
        "--resume-from": WARM_START_CKPT,
        "--mod-dim": "32",
        "--freq-along": "8",
        "--hosc-beta-end": "4.0",
        "--softmax-temp-end": "0.05",
        "--anneal-epochs": str(ORIGINAL_SCHEDULE_EPOCHS),
        "--l7-start-epoch": str(int(epochs) + 1),
        "--seg-phase-advect-weight": "0.4",
        "--seg-phase-advect-start-epoch": str(SURGICAL_ENGAGE_EPOCH),
        "--seg-margin-satisfice-weight": "0.2",
        "--seg-margin-satisfice-start-epoch": str(SURGICAL_ENGAGE_EPOCH),
        "--seg-subpix-boundary-weight": "0.3",
        "--seg-subpix-boundary-start-epoch": str(SURGICAL_ENGAGE_EPOCH),
        "--seg-subpix-edge-weight-source": "pa_flipmass",
        "--pose-finish-engage-on": "sigma_min_plateau",
        "--pose-finish-start-epoch": str(POSE_BACKSTOP_EPOCH),
        "--w-pose": "1.0",
        "--head-offset-solver": "flip_median",
        "--verdict-batch": "32",
    }
    mismatches = {
        flag: (pairs.get(flag), want)
        for flag, want in required.items() if pairs.get(flag) != want
    }
    for flag in ("--warm-start-weights-only", "--self-orient", "--chroma",
                 "--fused-r-kernel", "--pose-training-compute-gate",
                 "--component-wallclock-telemetry"):
        if flag not in pairs:
            mismatches[flag] = ("<ABSENT>", "present")
    if mismatches:
        raise ValueError(f"{PROGRAM_NAME} actuation/custody REFUSE: {mismatches}")

    # ── the dead-sensor refusal (adverse finding A2): the c1 label_floor start-event
    #    band [0.00496,0.00700] sits ABOVE this trunk's resume d_seg (~0.0034) — the
    #    sensor can never fire on the warm path; emitting it would be a silent dead gate. ──
    if "--seg-phase-advect-start-event" in pairs:
        raise ValueError(
            f"{PROGRAM_NAME} dead-sensor REFUSE: --seg-phase-advect-start-event emitted, but the "
            "label_floor band [0.00496,0.00700] is unreachable from resume d_seg ~0.0034 (the "
            "sensor precondition is a from-scratch label-smooth descent). Warm-path sensor "
            "recalibration is the named owed item; the epoch boundary carries the rationale.")

    # ── compile-time derived launch blockers (the launcher's b1-ticket contract). ──
    blockers: list[dict[str, str]] = []
    evidence: dict[str, Any] = {}
    _file_exists_blocker(
        WARM_START_CKPT, "C2_WARM_START_CKPT_CUSTODY",
        f"the warm-start checkpoint {WARM_START_CKPT} must exist (mod32cap EMA-best ep650)",
        blockers, evidence, "warm_start_ckpt_custody")
    receipt_row = None
    _own_hash = typed.typed_config_hash()
    for rp in sorted(Path("experiments/results").glob("*/dry_start_report.json")):
        try:
            rep = _json.loads(rp.read_text())
        except (OSError, ValueError):
            continue
        # HASH-MATCH tightening (coordinator amendment 2026-07-16): a receipt clears the
        # blocker ONLY for the exact composed config it benched — name+green alone would let a
        # pre-amendment bench green-light an amended config (stale-receipt laundering). A
        # receipt without a recorded hash (older format) NEVER clears the blocker (fail-closed).
        if (rep.get("gate") == "full_config_dry_start"
                and str(rep.get("config")) == PROGRAM_NAME and bool(rep.get("green"))
                and str(rep.get("typed_config_hash", "")) == _own_hash):
            receipt_row = {"status": "MEASURED_GREEN", "report": str(rp),
                           "typed_config_hash": _own_hash,
                           "peak_rss_gib": rep.get("peak_rss_gib"),
                           "sec_per_ep_marginal": rep.get("sec_per_ep_marginal"),
                           "ts": rep.get("ts")}
    if receipt_row is not None:
        evidence["composed_bench_receipt"] = receipt_row
    else:
        blockers.append({
            "id": "C2_COMPOSED_BENCH_NOT_MEASURED",
            "detail": "no GREEN full_config_dry_start report with typed_config_hash "
                      f"{_own_hash[:16]}… exists for config {PROGRAM_NAME!r} — run the "
                      "launcher's bounded --dry-start ON THIS EXACT CONFIG (the receipt "
                      "producer; it also PROVES the warm-start weight-shape load via resume_ok) "
                      "to measure sec/epoch + peak RSS before the real launch",
        })

    emitted_names = sorted({f for f, _ in _crucible_v7_argv_pairs(argv)})
    manifest = build_launch_manifest(
        program_name=PROGRAM_NAME, emitted_flag_names=emitted_names,
        typed_config_hash=typed.typed_config_hash(), typed_validated=True)
    manifest = dict(manifest)
    manifest.update({
        "expected_active_levers": list(C2_SURGICAL_EXPECTED_LEVERS),
        "composition_contract": {
            "task": "c2_surgical_warm composition (train-least doctrine 2026-07-16)",
            "directive": "operator 2026-07-16 Kolmogorov/projection/realization; train the "
                         "absolute least, most surgical targets only",
            "warm_start": {
                "checkpoint": WARM_START_CKPT,
                "trunk_d_seg_verdict": 0.003366,
                "trunk_d_seg_through_R_n600": 0.003146,
                "mode": "weights-only (fresh moments, re-seeded guard, EMA-shadow load, "
                        "natural continuation ep651)",
                "why_not_v9_arch": "mod-dim 19 + no-self-orient bank is weight-shape-"
                                   "incompatible with the mod-dim-32 self-orient checkpoint "
                                   "(DERIVED from the config diff)",
                "why_not_v752_factory": "its lane-band/ladder/dash-comb lever stack is render-"
                                        "side drift the checkpoint never trained under",
            },
            "surgical_targets": {
                "road_lane_appearance_phase": "66.0% of the measured witness residual "
                                              "(witness-own decomp §1); T1 #424 + #360 at ep700",
                "force3_subpix_straddle_coverage": "AMENDMENT 2026-07-16 (lane-channel "
                                                   "completeness verdict 477ec610c5): T1 is "
                                                   "birth-SILENT but 26.3% of candidate straddle "
                                                   "px (354 lane-adjacent px/frame) get NO phase "
                                                   "supervision without Force-3 subpix — "
                                                   "TieLocusDisplacement co-emitted at ep700, "
                                                   "factory defaults, pa_flipmass edge weights "
                                                   "(measured artifact present)",
                "joint_pose_finish": "L68 photometric wall — only joint descent crosses; "
                                     "sigma_min_plateau gate, banked-R1 dxi fallback, "
                                     "backstop ep1000",
            },
            "kolmogorov_per_stage": {
                "re_anchor_651_700": "TRAINED because a weights-only warm start discards "
                                     "optimizer state — no solve reproduces fresh-moment "
                                     "settling on the exact loss; 50 ep bounded",
                "phase_stage_700plus": "TRAINED because the phase zero-mode is the measured "
                                       "irreducible residual (flat carriers EXHAUSTED §3; "
                                       "post-hoc storage DEAD per L68-analogue) — no "
                                       "deterministic generator produces the witness-side "
                                       "sub-pixel phase; the STORED side ships via the #425 "
                                       "carrier (seed), only the render-side coherence trains",
                "pose_finish": "TRAINED because the photometric wall is MEASURED (5 post-hoc "
                               "formulations dead, L68) — pose-legible photometrics exist only "
                               "under joint descent; the stored 7.2KB dxi seed is the fallback",
                "muon_726": "checkpoint's own schedule (lineage-native); NOT re-derived here",
            },
            "not_trained": dict(C2_SOLVE_SEED_DROP_SLOTS),
        },
        "adverse_findings_surfaced": [
            "A1 grad-clip magnitude law: mod32cap lineage uses clip 1.0 per-group WITHOUT "
            "per-param normalize (unlike c1 where the clip is normalize-masked/INERT); the "
            "0.5-saturation finding was a c1-family measurement — on THIS lineage the incumbent "
            "law is kept (changing the magnitude law at warm-start is an uncontrolled "
            "re-treatment); frac_clipped telemetry is ON and the autoclip/normalize A/B stays "
            "a NAMED open lever (perparam_normalize_masks_all_norm_clipping_c0_confound)",
            "A2 label_floor start-event DEAD on the warm path (band above resume d_seg) — "
            "epoch engage + refusal check; warm-path sensor recalibration OWED",
            "A3 naive epoch-extension would re-activate the l7 defect stage at its old 1001 "
            "epoch — excluded via l7-start == epochs + 1 (the TRUE never-runs form; the "
            "adversarial review 2026-07-16 caught the == epochs off-by-one that would have "
            "run l7 on the final epoch)",
            "A4 softmax-temp-end 0.05 is BELOW the v9 P-TAU2 knee band [0.191,0.543]; kept "
            "because the checkpoint's weights are conditioned on this schedule (config-"
            "conditional constant, L18) — a tau re-treatment is a separate measured arm",
            "A5 Muon window ep726-1000 did NOT improve the EMA-best on this lineage's original "
            "run (best stayed ep650, pre-Muon) — the #217 finishing-schedule question is OPEN; "
            "here Muon re-fires at 726 WITH the new phase-shaped gradients (different "
            "treatment), watched via the holistic facet check-ins",
            "A6 warm-start basin lock-in risk (#253 erasure-timing): INFERRED small for a "
            "sub-pixel-phase objective (small-deformation regime around the held trunk), NOT "
            "measured — the ep700 engage re-treats the spike guard; the paired fresh arm is "
            "the c1 run itself (already composed + dry-started)",
            "A7 sec/ep for THIS composed config is NOT yet measured (the mod32cap lineage "
            "anchor is ~121 s/ep incl. verdicts; c1's 325-361 s/ep is the worst-case bound) — "
            "the dry-start receipt blocker holds until measured",
        ],
        "byte_close_contract": {
            "tool": "tools/levelset_byte_close_and_eval.py",
            "phase_carrier": "--phase-carrier REQUIRED on post-engage checkpoints (the trained "
                             "phase zero-mode + its stored section are measured together); "
                             "pre-engage checkpoints byte-close without it",
            "pose": "R1 dxi section (7.2KB) ships when the conditioning gate never fires "
                    "(DISENGAGED, LOUD); joint-finish dxi otherwise",
        },
        "value_provenance_notes": {
            "anneal_epochs": {
                "value": int(ORIGINAL_SCHEDULE_EPOCHS),
                "ladder_class": "derived_at_config",
                "inputs": {"original_run_epochs": 1000,
                           "contract": "--anneal-epochs docstring: a warm-start arm MUST set this "
                                       "to the ORIGINAL schedule length so the resume epoch "
                                       "reproduces the plant"},
                "note": "schedule continuity for the ep651 warm start; tau/beta/LR hold end "
                        "values past ep1000. Lives here (not the constants LawRef surface) "
                        "because no registered ..._vN equation derives it — the #332 gate "
                        "fail-closes on equation_id None.",
            },
            "surgical_engage_epoch": {
                "value": int(SURGICAL_ENGAGE_EPOCH),
                "ladder_class": "derived_at_config",
                "inputs": {"resume_epoch": RESUME_EPOCH,
                           "re_anchor_window": SURGICAL_ENGAGE_EPOCH - RESUME_EPOCH,
                           "rationale": "fresh-moment settling window ~= 2x the stage-transition "
                                        "rewarmup scale before new loss terms engage; label_floor "
                                        "sensor DEAD on warm path (adverse finding A2)"},
                "note": "ONE engage boundary for both surgical terms (single spike-guard "
                        "re-treat); same non-LawRef home as anneal_epochs.",
            },
        },
        "held": True,
        "operator_go_required": True,
        "launcher_registration_owed": "tools/launch_witness_run.py --config c2_surgical_warm "
                                      "(the launcher is untouchable to this landing; 3-line "
                                      "registration at GO time)",
        "launch_blockers": blockers,
        **evidence,
    })

    # NB (the #332 LawRef gate + the schedule-provenance gate, learned from the 2026-07-16 rc=8 +
    # NAKED-epoch refusals on the REAL gate chain): every constants-manifest row MUST cite a
    # registered ``..._vN`` equation, and every emitted positive ``--*-start-epoch`` trigger must
    # be EVENT / DERIVED / CAP. On this warm path NO recognised event sensor is co-emittable (the
    # label_floor band is unreachable from resume d_seg ~0.0034 — adverse finding A2), so EVERY
    # schedule boundary takes the DERIVED form: a constants row citing
    # ``warm_start_schedule_reconstruction_v1`` (registered 2026-07-16; its evaluator RECOMPUTES
    # each value from named inputs at gate time, and the config-of-record reads carry the
    # checkpoint launch.sh SHA as the custody artifact). Non-equation value provenance
    # (anneal_epochs continuity) lives in ``value_provenance_notes`` on the manifest.
    _ws_eq = "warm_start_schedule_reconstruction_v1"
    _record_inputs = lambda v: [  # noqa: E731  (local literal builder, used 2x)
        {"name": "mode", "value": "config_of_record", "source": "literal"},
        {"name": "config_of_record_value", "value": int(v),
         "source": f"{WARM_START_RUN_DIR}/launch.sh",
         "sha256": "dd7921bce67da7fdb0982c5c951cfbc69156e5092b7a454576d5a4b4acfce94a"},
    ]
    constants: dict[str, Any] = {
        "tau_softplus_start_epoch": {
            "value": 300,
            "equation_id": _ws_eq,
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": _record_inputs(300),
            "note": "checkpoint lineage stage boundary (config of record, SHA-verified); already "
                    "PAST at resume ep651 — present for schedule reconstruction, not a new "
                    "trigger",
        },
        "muon_start_epoch": {
            "value": 726,
            "equation_id": _ws_eq,
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": _record_inputs(726),
            "note": "the checkpoint's own lineage schedule (726); re-fires 75 ep after resume "
                    "with phase-shaped gradients; #217 finishing-schedule OPEN (adverse "
                    "finding A5)",
        },
        "l7_start_epoch": {
            "value": int(epochs) + 1,
            "equation_id": _ws_eq,
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": {"mode": "run_length_exclusion", "run_epochs": int(epochs)},
            "note": "l7 NEVER runs (start = epochs + 1, the TRUE never-runs form — the trainer "
                    "loop is range(start, epochs+1) INCLUSIVE so start==epochs would run l7 on "
                    "the final epoch; adversarial-review fix 2026-07-16, matching the mod32cap "
                    "record's own 1001/epochs=1000 parking); l7 is a measured defect — a naive "
                    "extension would have re-fired it at the record's 1001 (adverse finding A3)",
        },
        "seg_phase_advect_start_epoch": {
            "value": int(SURGICAL_ENGAGE_EPOCH),
            "equation_id": _ws_eq,
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": {"mode": "resume_plus_window", "resume_epoch": int(RESUME_EPOCH),
                       "re_anchor_window": int(SURGICAL_ENGAGE_EPOCH - RESUME_EPOCH)},
            "note": "surgical engage boundary = resume 651 + ~50 re-anchor (fresh-moment "
                    "settling ~= 2x the stage-transition rewarmup scale); the label_floor "
                    "sensor is DEAD on the warm path (band above resume d_seg — A2), so the "
                    "boundary is DERIVED, not event-fired; warm-path sensor recalibration OWED",
        },
        "seg_margin_satisfice_start_epoch": {
            "value": int(SURGICAL_ENGAGE_EPOCH),
            "equation_id": _ws_eq,
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": {"mode": "resume_plus_window", "resume_epoch": int(RESUME_EPOCH),
                       "re_anchor_window": int(SURGICAL_ENGAGE_EPOCH - RESUME_EPOCH)},
            "note": "co-engages with the phase term at the single surgical boundary (one "
                    "spike-guard re-treat; the trunk partition is already formed)",
        },
        "seg_subpix_boundary_start_epoch": {
            "value": int(SURGICAL_ENGAGE_EPOCH),
            "equation_id": _ws_eq,
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": {"mode": "resume_plus_window", "resume_epoch": int(RESUME_EPOCH),
                       "re_anchor_window": int(SURGICAL_ENGAGE_EPOCH - RESUME_EPOCH)},
            "note": "Force-3 subpix co-engages at the SAME surgical boundary (amendment "
                    "2026-07-16: closes the 26.3% straddle coverage gap the lane-channel "
                    "completeness verdict measured — T1 alone leaves birth-site straddles "
                    "phase-unsupervised; 477ec610c5)",
        },
        "pose_finish_start_epoch": {
            "value": int(POSE_BACKSTOP_EPOCH),
            "equation_id": _ws_eq,
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": {"mode": "original_plant_end",
                       "original_schedule_epochs": int(ORIGINAL_SCHEDULE_EPOCHS)},
            "note": "fail-safe BACKSTOP for the sigma_min_plateau conditioning gate at the "
                    "original plant end (ep1000 = 300 surgical epochs before run end; a firing "
                    "cap is LOUD)",
        },
        "head_offset_solver": {
            "value": "flip_median",
            "equation_id": "laguerre_ot_head_offset_v1",
            "ladder_class": "derived_at_config",
            "fallback_used": False,
            "inputs": {"objective": "d_seg is Hamming; the L1-optimal 1-D threshold is the "
                                    "per-edge flip-margin median",
                       "ot_newton_status": "MEASURED-worse at mod32cap ep650",
                       "advisory": "decode-time observer; never mutates shipped/EMA weights"},
            "note": "same folded arbiter as c1; its realized-through-R delta rows are the owed "
                    "A/B instrument",
        },
    }

    # Schedule governance is EMPTY BY DERIVATION: every emitted positive --*-start-epoch trigger
    # takes the DERIVED form (a warm_start_schedule_reconstruction_v1 constants row above, which
    # the schedule-provenance gate classifies FIRST — DERIVED precedes any governance tag). The
    # EVENT/CAP forms are structurally unavailable on this warm path (no recognised sensor is
    # co-emittable; label_floor DEAD below resume d_seg — adverse finding A2); the per-boundary
    # rationales live in the constants rows' notes + the manifest.
    governance: dict[str, Any] = {}

    return CrucibleV7LaunchConfig(
        typed=typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance=governance,
    )


def _typed(lever):
    """Adapt a DSL ``Lever`` to the typed-lever wrapper (same adapter the c1/v9 specs use)."""
    from tac.witness_dsl.spec_v9_cgauge import _typed_ideal_lever

    return _typed_ideal_lever(lever)


__all__ = [
    "C2_SOLVE_SEED_DROP_SLOTS",
    "C2_SURGICAL_EXPECTED_LEVERS",
    "DEFAULT_OUT_DIR",
    "PROGRAM_NAME",
    "POSE_BACKSTOP_EPOCH",
    "RESUME_EPOCH",
    "RUN_EPOCHS",
    "SURGICAL_ENGAGE_EPOCH",
    "WARM_START_CKPT",
    "WARM_START_RUN_DIR",
    "compile_c2_surgical_warm_launch_config",
]
