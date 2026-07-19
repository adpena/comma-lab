# SPDX-License-Identifier: MIT
"""v9c3_duty_ab — ep725 weights-only-fork duty-lever A/B programs (task #563 revision).

PURPOSE (vehicle naming lock, operator 2026-07-18): **v9c3** = restart from the v9c2
ep725 banked BEST with corrected events (#518 resume-warmup geometry) — a SIGNAL-HARVEST
vehicle, never v10. This module composes the PAIRED duty-queue A/B fork programs the
2026-07-19 duty-ticket revision owes (charter
``.omx/tmp/codex_prompts/duty_ticket_revision_ep725_fork_20260719.md``):

* ticket 02 ``HorizonWeightedMargin`` (duty 47.3%): OFF twin vs ON
  (derived-live 0.15 single-force share at the ep753 boundary).
* ticket 03 ``StepNativeActivation`` (duty 34.2%): control ``--hosc-beta-end 4.0``
  (the checkpoint's OWN schedule endpoint, config of record) vs treatment ``8.0``.
  NB the composer-declared "3.177 -> 8.0" endpoint belongs to the V9 mod19 LawRef
  custody surface; the c2/mod32 lineage flag value is 4.0 (launch.sh of record) —
  the one-flag contrast on THIS vehicle is 4.0 -> 8.0 (estimand note in the ticket).

FORK GEOMETRY (every constant derived; constants-are-poison):

* base checkpoint: the BANKED v9c2 BEST EMA ep725 (weights-only fork; the bank npz is a
  59-key EMA DEPLOY snapshot with NO optimizer/RNG/event state — a clean pre-Muon
  resumable state does NOT exist, so a fresh-optimizer fork is the ONLY honest form;
  memory ``vehicle_naming_v9c_warm_lineage_v10_reserved_capstone_20260718``).
* RESUME_EPOCH 726 = ckpt epoch 725 + 1 (explicit ``--warm-start-epoch``).
* REWARMUP 27 ep = ceil(2/(1-0.999)/75) via ``adam_v_variance_warmup_length_v1``
  (#518 item 2; composed through :func:`ResumeLRWarmup`, which OVERRIDES the
  config-of-record 8 — the 8 sits UNDER the c=1 memory bound).
* TREATMENT_BOUNDARY 753 = resume + rewarmup window (a treatment engaging INSIDE the
  cold-moment window would confound the contrast with the fresh-AdamW transient; the
  HWM derived-live boundary scan must also see re-conditioned losses).
* PRIMARY window = boundary + 100 ep (4 verdict points at the --eval-every 25 cadence,
  the minimum for a paired-mean verdict with df=3); SECONDARY = boundary -> run end
  (10 points, 247 ep — slightly under the composer memo's conservative 275-ep floor,
  traded for an extrapolation-free plant, see next bullet; that floor came from an
  UNTRUSTED NCDE fit (R^2=0.06), while these windows are sized against the direct
  paired verdict-noise thresholds in the tickets).
* RUN_EPOCHS 1000 = the ORIGINAL plant length exactly: the linear hosc-beta anneal has
  NO clamp past --anneal-epochs (prog=(ep-1)/999 extrapolates), so running past 1000
  would push beta BEYOND its endpoint in both arms (4.084/8.196 @1028); ending at 1000
  keeps every scheduled quantity inside its designed range. ``--anneal-epochs 1000``
  pins schedule continuity (the #517 pre-v0 positioning then loads tau/beta/seg_form
  at their true resume-epoch schedule values before the v0 verdict). l7 parks at 1001,
  byte-matching the mod32cap record's own parking form.
* SCHEDULE-REALIZATION note (measured from the donor): the donor froze beta at 3.1772
  from ep726 on (the FEED-fm Muon freeze — Muon switched at 726), so the donor never
  REALIZED the 3.177->4.0 tail; the control arm here anneals it under AdamW. Both arms
  share this; the contrast stays one-flag. Disclosed in AF5.
* **Muon: OMITTED ENTIRELY** (no ``--muon-start-epoch`` => trainer never switches).
  The donor's absolute ep726 Muon is the MEASURED resume-event confound engine
  (Lane 0.213->0.238 kick + stuck pose gate; memory
  ``warm_start_resume_must_adapt_events_to_resume_epoch_and_geometry_20260718``); a
  duty-lever A/B needs a pure-AdamW measurement plant. #217 finishing-schedule OPEN.
* **pose: pose-blind fork** (``--w-pose 0`` + compute gates, NO conditioning gate):
  the trunk trained ep700-725 with the pose gate never engaged (stuck
  DEGENERATE_GUARD), so pose-blind reproduces the exact loss physics at ep725 AND
  removes the arm-DIVERGENT sigma_min_plateau engagement risk from the A/B window.
* #518 recipe composed: ResumeLRWarmup (27 ep, item 1/2) + ForkEmaClearance (item 7;
  EMA restarts as a point mass at the fork — BEST-banking suppressed ~1/(1-0.997)
  updates) + automatic item-5 (#517) tau/beta/seg_form positioning pre-v0 + item 9
  ramp-end reorient. ForkHeadSolve + MarginStepCap stay OFF in BOTH arms (never-fired
  levers; firing them inside a duty A/B would add a second unmeasured mechanism; the
  measured MarginStepCap cap does not exist yet — named OPEN, not silently defaulted).
* the c2 surgical loss terms (phase/satisfice/subpix at ep700) stay ACTIVE at their
  config-of-record values in BOTH arms — they are the loss the trunk trained under
  ep700-725; dropping them would be a hidden second treatment.

ADVERSE FINDINGS SURFACED PRE-FIRE (the launch-must-surface law): the v9c2 launch.sh
header carries measured adjudications AGAINST both tickets —
``HorizonWeightedMargin=measured WEAK by #141/#169 (dS ceiling 0.012-0.024)`` and
``StepNativeActivation=... step-native is the alternative arm for the NEXT fresh arm``.
Both are quoted in the ticket packages; the A/B is the instrument that converts those
priors into a measured verdict (power analysis in the tickets: the HWM oracle ceiling
is ~5-10x the derived paired noise threshold).

CONTAINMENT: builds + validates + compiles only ($0, pure). LAUNCH = operator-GO
through the governed launcher (--config registration OWED at GO time, same 3-line
pattern as c2). MEANS: pointer 0.1910828242 [contest-CPU] moves only through a
byte-closed n600 exact row; nothing here is a score claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: banked v9c2 BEST EMA ep725 (custody verified 2026-07-19: 460,448 bytes, sha256
#: b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef; sidecar
#: levelset_best.json epoch 725 / d_seg 0.003457972208658854, sha256
#: 66950d15bf8575eb4bae28c39c0276b0960f8e14364ed45da24b30b960f27d9e).
BANK_CKPT = ("experiments/results/banks/v9c2_defensive_bank_20260718/"
             "levelset_witness_ema_BEST.npz")
BANK_CKPT_SHA256 = "b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef"
BANK_CKPT_BYTES = 460448

#: config of record for the fork base = the v9c2 run's own launch.sh (sacred dir,
#: read-only; sha computed 2026-07-19).
CONFIG_OF_RECORD_LAUNCH_SH = ("experiments/results/levelset_n600_witness_20260717T113932Z/"
                              "launch.sh")
CONFIG_OF_RECORD_LAUNCH_SH_SHA256 = (
    "eecbb745e7f7fefbfd9c62c7eb5f16b590e1c49cc10d07ffc8ca6ef96c4fc1ed")

RESUME_EPOCH = 726                     # ckpt ep725 + 1
REWARMUP_EPOCHS = 27                   # adam_v_variance_warmup_length_v1 (beta2 .999, S=75, c=2)
TREATMENT_BOUNDARY = RESUME_EPOCH + REWARMUP_EPOCHS          # 753
PRIMARY_WINDOW_EPOCHS = 100            # 4 verdict points @ eval-every 25 (paired mean, df=3)
ORIGINAL_SCHEDULE_EPOCHS = 1000        # the lineage plant; --anneal-epochs pins to it
RUN_EPOCHS = ORIGINAL_SCHEDULE_EPOCHS  # extrapolation-free: end AT the plant end
SECONDARY_WINDOW_EPOCHS = RUN_EPOCHS - TREATMENT_BOUNDARY    # 247 (10 verdict points)
SURGICAL_ENGAGE_EPOCH = 700            # config of record (already PAST at resume)

#: control endpoint = the checkpoint's OWN schedule (config of record); treatment = 8.0
HOSC_BETA_END_CONTROL = 4.0
HOSC_BETA_END_TREATMENT = 8.0

TICKETS = ("hwm", "step")
ARMS = ("off", "on")

V9C3_DUTY_EXPECTED_LEVERS_COMMON: tuple[str, ...] = (
    "v9c3_warm_fork_ep725_weights_only",
    "resume_lr_rewarmup",
    "fork_ema_clearance",
    "FEED_07b_step_native_activation",   # activation schedule owner (control OR treatment value)
    "phase_advection_consistency",
    "margin_band_satisficing",
    "tie_locus_displacement",
    "pose_blind_compute_gate",
    "head_offset_solver",
    "verdict_live_gap",
    "c2_speed_stack",
    "c2_component_wallclock_telemetry",
)


def _out_dir_for(ticket: str, arm: str) -> str:
    # v9c3_dev prefix: DEV maturity => never pointer-promotable (tac.checkpoint_maturity).
    return (f"/Volumes/VertigoDataTier/pact/experiments/results/"
            f"v9c3_dev_duty_{ticket}_{arm}_n600_seed0")


def compile_v9c3_duty_ab_config(
    ticket: str,
    arm: str,
    gt_cache_path: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    *,
    num_pairs: int = 600,
    epochs: int = RUN_EPOCHS,
    out_dir: str | None = None,
):
    """Compile one arm of one duty-ticket ep725-fork A/B (pure / $0; never launches).

    ``ticket`` in {"hwm", "step"}; ``arm`` in {"off", "on"}. The OFF twin of both
    tickets is physics-identical (distinct run dir + purpose custody); the ON arm
    differs by exactly ONE Lever delta:

    * hwm/on: adds :func:`HorizonWeightedMargin` (derived-live 0.15 share, ep753).
    * step/on: the activation lever endpoint 4.0 -> 8.0 (one-flag argv delta).
    """
    import json as _json

    from tac.local_acceleration.scorer_throughput_gate import derive_wall_clock_budget_days
    from tac.witness_autoconfig import CrucibleV7LaunchConfig, _crucible_v7_argv_pairs
    from tac.witness_dsl.curriculum_dsl import (
        ForkEmaClearance,
        HeadOffsetSolver,
        HorizonWeightedMargin,
        Lever,
        MarginBandSatisficing,
        PhaseAdvectionConsistency,
        PoseBlindComputeGate,
        ResumeLRWarmup,
        StepNativeActivation,
        TieLocusDisplacement,
        VerdictLiveGap,
    )
    from tac.witness_dsl.typed_config import (
        Provenanced,
        ProvenanceClass as _PC,
        TypedAnneal,
        TypedWitnessConfig,
        build_launch_manifest,
    )

    if ticket not in TICKETS:
        raise ValueError(f"ticket must be one of {TICKETS}, got {ticket!r}")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    out_dir = out_dir or _out_dir_for(ticket, arm)
    program_name = f"v9c3_duty_{ticket}_{arm}"

    # ── the checkpoint's own config, flag-for-flag (config of record: the v9c2 run's
    #    launch.sh, SHA-verified above). Deviations, each provenanced in the manifest:
    #    (1) Muon OMITTED (measured resume-event confound engine; #217 OPEN);
    #    (2) pose-blind (--w-pose 0, no conditioning gate; trunk's realized ep700-725
    #        loss physics — the gate never engaged in the donor);
    #    (3) rewarmup trio Lever-owned (ResumeLRWarmup 27 vs record 8; DERIVED > CONFIG);
    #    (4) activation quintet Lever-owned (StepNativeActivation; control == record);
    #    (5) epochs 1000 (plant end, extrapolation-free) / l7 parked at 1001;
    #    (6) resume source = the BANKED ep725 BEST EMA (not the donor's mod32cap ep650). ──
    base: dict[str, Any] = {
        "--seed": 0,
        "--async-verdict": True,
        "--eval-every": 25,
        "--verdict-pairs": 0,
        "--curriculum": True,
        "--tau-softplus-start-epoch": 300,
        "--tau-softplus-tau": 0.3,
        "--l7-start-epoch": int(epochs) + 1,
        "--w-seg": 100,
        "--w-pose": 0.0,
        "--score-domain-loss": True,
        "--mod-dim": 32,
        "--hidden-dim": 96,
        "--n-hidden": 4,
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
        "--stage-transition-reset-moments": True,
        "--anneal-epochs": int(ORIGINAL_SCHEDULE_EPOCHS),
    }

    warm_fork = Lever(
        "v9c3_warm_fork_ep725_weights_only",
        overrides={
            "--resume-from": BANK_CKPT,
            "--warm-start-weights-only": True,
            "--warm-start-epoch": int(RESUME_EPOCH),
        },
        notes=(f"v9c3 fork: BANKED v9c2 BEST EMA ep725 (d_seg 0.003457972208658854; sha "
               f"{BANK_CKPT_SHA256[:12]}…), WEIGHTS ONLY => fresh AdamW moments, re-seeded "
               "spike guard, EMA shadow load, explicit resume epoch 726. The bank npz is a "
               "59-key EMA deploy snapshot (NO optimizer/RNG/event state) — a fresh-state "
               "fork is the only honest resume form (campaign meta-review 2026-07-18)."),
    )

    speed = Lever(
        "c2_speed_stack",
        overrides={
            "--fused-r-kernel": True,
            "--cache-gt-skeleton": True,
            "--verdict-batch": 32,
            "--safe-compile-regions": "hosc_activation",
        },
        notes="config-of-record speed core (identical in every arm).",
    )

    telemetry = Lever(
        "c2_component_wallclock_telemetry",
        overrides={
            "--component-wallclock-telemetry": True,
            "--component-wallclock-probe-every": 1,
            "--profile-timing": True,
        },
        notes="read-only score-neutral wall-clock telemetry (defaults-ON law; identical arms).",
    )

    beta_end = HOSC_BETA_END_TREATMENT if (ticket == "step" and arm == "on") \
        else HOSC_BETA_END_CONTROL
    activation = StepNativeActivation(
        beta_start=1.0, beta_end=float(beta_end), anneal="linear", window=0,
        basis="annealed_hosc", omega=1.0,
        scientific_declaration=(ticket == "step"),
    )

    levers: list[Lever | Any] = [
        warm_fork,
        ResumeLRWarmup(),                 # 27 ep (LawRef-derived; overrides record's 8)
        ForkEmaClearance(),
        activation,
        PhaseAdvectionConsistency(weight=0.4, start_epoch=SURGICAL_ENGAGE_EPOCH),
        MarginBandSatisficing(weight=0.2, start_epoch=SURGICAL_ENGAGE_EPOCH),
        TieLocusDisplacement(start_epoch=SURGICAL_ENGAGE_EPOCH),
        PoseBlindComputeGate(),
        HeadOffsetSolver(mode="flip_median", tau=1.0),
        VerdictLiveGap(every=25),
        speed,
        telemetry,
    ]
    expected = list(V9C3_DUTY_EXPECTED_LEVERS_COMMON)
    if ticket == "hwm" and arm == "on":
        levers.append(HorizonWeightedMargin(
            weight=0.15, target=0.5, margin_lo=0.3, margin_hi=0.5,
            row_lo=96, row_hi=288, start_epoch=int(TREATMENT_BOUNDARY), window=0,
            stage_share_derived_live=True, scientific_declaration=True,
        ))
        expected.append("horizon_weighted_margin")

    purpose = (
        f"v9c3_duty_{ticket}_{arm} (#563 duty-ticket revision 2026-07-19): ep725 "
        "weights-only fork of the banked v9c2 BEST EMA (fresh optimizer, #518 recipe: "
        "27-ep LR rewarmup + fork EMA clearance + #517 pre-v0 schedule positioning; Muon "
        "OMITTED — measured resume-event confound engine; pose-blind plant). "
        + ("PAIRED OFF TWIN — no treatment lever; the shared control for the "
           f"{ticket} duty contrast." if arm == "off" else
           ("TREATMENT: HorizonWeightedMargin derived-live 0.15 single-force share at "
            "the ep753 boundary (receipt: horizon_margin_boundary_receipt.json)."
            if ticket == "hwm" else
            "TREATMENT: hosc beta-end 4.0 -> 8.0 (one-flag step-native sharpening of the "
            "trained trunk; #517 positions beta(726)=6.0801 vs control 3.1772 pre-v0 — the "
            "v0 verdict is the immediate-response receipt).")) +
        " CONTAINMENT: compile only; LAUNCH = operator-GO. MEANS: pointer "
        "0.1910828242 [contest-CPU] moves only through a byte-closed n600 exact row."
    )

    def _typed(lever):
        from tac.witness_dsl.spec_v9_cgauge import _typed_ideal_lever
        return _typed_ideal_lever(lever)

    typed = TypedWitnessConfig(
        name=program_name,
        out_dir=str(out_dir),
        gt_cache=str(gt_cache_path),
        num_pairs=int(num_pairs),
        epochs=int(epochs),
        wall_clock_budget_days=Provenanced(
            value=round(derive_wall_clock_budget_days(int(epochs)), 3),
            provenance=_PC.DERIVED_AT_CONFIG, unit="days",
            source="scorer_throughput_gate.derive_wall_clock_budget_days; only "
                   f"~{int(epochs) - RESUME_EPOCH} epochs actually train on the fork"),
        mlx_device="gpu",
        temp=TypedAnneal(
            start=Provenanced(value=1.0, provenance=_PC.MEASURED_ANCHOR, unit="tau",
                              source="config of record (v9c2 launch.sh, SHA-verified)"),
            end=Provenanced(value=0.05, provenance=_PC.MEASURED_ANCHOR, unit="tau",
                            source="config of record; the ep725 weights are conditioned "
                                   "on THIS schedule (L18 config-conditional)"),
        ),
        base=base,
        levers=tuple(_typed(lv) for lv in levers),
        purpose=purpose,
    )

    violations = typed.validate_program()
    if violations:
        raise ValueError(
            f"{program_name} DSL gate: {len(violations)} WitnessProgram.validate "
            f"violation(s): {violations[:4]}")

    got = sorted(lv.name for lv in typed.levers)
    if got != sorted(expected):
        raise ValueError(f"{program_name} expected-lever REFUSE: {got} != {sorted(expected)}")

    argv = tuple(typed.to_program().compile_trainer_argv())
    pairs = dict(_crucible_v7_argv_pairs(argv))

    # ── required actuation values (consumed-not-inert, verified on emitted argv). ──
    required = {
        "--resume-from": BANK_CKPT,
        "--warm-start-epoch": str(RESUME_EPOCH),
        "--mod-dim": "32",
        "--freq-along": "8",
        "--hosc-beta": "1.0",
        "--hosc-beta-end": str(beta_end),
        "--softmax-temp-end": "0.05",
        "--anneal-epochs": str(ORIGINAL_SCHEDULE_EPOCHS),
        "--l7-start-epoch": str(int(epochs) + 1),
        "--stage-transition-rewarmup-epochs": str(REWARMUP_EPOCHS),
        "--seg-phase-advect-start-epoch": str(SURGICAL_ENGAGE_EPOCH),
        "--seg-margin-satisfice-start-epoch": str(SURGICAL_ENGAGE_EPOCH),
        "--seg-subpix-boundary-start-epoch": str(SURGICAL_ENGAGE_EPOCH),
        "--w-pose": "0.0",
        "--head-offset-solver": "flip_median",
        "--verdict-batch": "32",
    }
    if ticket == "hwm" and arm == "on":
        required.update({
            "--seg-horizon-margin-weight": "0.15",
            "--seg-horizon-margin-start-epoch": str(TREATMENT_BOUNDARY),
        })
    mismatches = {
        flag: (pairs.get(flag), want)
        for flag, want in required.items() if pairs.get(flag) != want
    }
    present_required = ["--warm-start-weights-only", "--self-orient", "--chroma",
                        "--fused-r-kernel", "--pose-training-compute-gate",
                        "--fork-ema-clearance", "--component-wallclock-telemetry"]
    if ticket == "hwm" and arm == "on":
        present_required.append("--seg-horizon-margin-derived-live")
    absent_required = ["--muon-start-epoch", "--pose-finish-start-epoch",
                       "--fork-head-solve", "--margin-step-cap"]
    for flag in present_required:
        if flag not in pairs:
            mismatches[flag] = ("<ABSENT>", "present")
    for flag in absent_required:
        if flag in pairs:
            mismatches[flag] = (pairs.get(flag), "<ABSENT>")
    if mismatches:
        raise ValueError(f"{program_name} actuation/custody REFUSE: {mismatches}")

    # ── schedule constants rows (every positive --*-start-epoch DERIVED; the c2 rc=8
    #    lesson: cite a registered ..._vN equation per row). ──
    _ws_eq = "warm_start_schedule_reconstruction_v1"
    _record_inputs = lambda v: [  # noqa: E731
        {"name": "mode", "value": "config_of_record", "source": "literal"},
        {"name": "config_of_record_value", "value": int(v),
         "source": CONFIG_OF_RECORD_LAUNCH_SH,
         "sha256": CONFIG_OF_RECORD_LAUNCH_SH_SHA256},
    ]
    constants: dict[str, Any] = {
        "warm_start_epoch": {
            "value": int(RESUME_EPOCH), "equation_id": _ws_eq,
            "ladder_class": "derived_at_config", "fallback_used": False,
            "inputs": {"mode": "resume_plus_window", "resume_epoch": int(RESUME_EPOCH),
                       "re_anchor_window": 0,
                       "checkpoint_best_epoch_custody":
                           "experiments/results/banks/v9c2_defensive_bank_20260718/"
                           "levelset_best.json sha256 "
                           "66950d15bf8575eb4bae28c39c0276b0960f8e14364ed45da24b30b960f27d9e"
                           " epoch 725 (resume = 725 + 1)"},
            "note": "resume metadata (not a curriculum stage): explicit fork epoch = banked "
                    "BEST epoch 725 + 1, pinned so the fork is self-documenting and "
                    "deterministic regardless of npz epoch-key presence",
        },
        "tau_softplus_start_epoch": {
            "value": 300, "equation_id": _ws_eq, "ladder_class": "derived_at_config",
            "fallback_used": False, "inputs": _record_inputs(300),
            "note": "lineage stage boundary (config of record); already PAST at resume 726 "
                    "— schedule reconstruction, not a new trigger",
        },
        "l7_start_epoch": {
            "value": int(epochs) + 1, "equation_id": _ws_eq,
            "ladder_class": "derived_at_config", "fallback_used": False,
            "inputs": {"mode": "run_length_exclusion", "run_epochs": int(epochs)},
            "note": "l7 NEVER runs (epochs+1 parking form; l7 is a measured defect)",
        },
        "seg_phase_advect_start_epoch": {
            "value": int(SURGICAL_ENGAGE_EPOCH), "equation_id": _ws_eq,
            "ladder_class": "derived_at_config", "fallback_used": False,
            "inputs": _record_inputs(SURGICAL_ENGAGE_EPOCH),
            "note": "config of record; PAST at resume 726 => active from the fork start in "
                    "BOTH arms (the loss the trunk trained under ep700-725)",
        },
        "seg_margin_satisfice_start_epoch": {
            "value": int(SURGICAL_ENGAGE_EPOCH), "equation_id": _ws_eq,
            "ladder_class": "derived_at_config", "fallback_used": False,
            "inputs": _record_inputs(SURGICAL_ENGAGE_EPOCH),
            "note": "config of record; PAST at resume (see phase row)",
        },
        "seg_subpix_boundary_start_epoch": {
            "value": int(SURGICAL_ENGAGE_EPOCH), "equation_id": _ws_eq,
            "ladder_class": "derived_at_config", "fallback_used": False,
            "inputs": _record_inputs(SURGICAL_ENGAGE_EPOCH),
            "note": "config of record; PAST at resume (see phase row)",
        },
    }
    # NB the HWM ON arm's --seg-horizon-margin-* rows (including start-epoch 753) are
    # custodied by the Lever's OWN scientific declaration (single ownership; a duplicate
    # spec-level row is a LawRef-authority conflict the #332 provenance gate refuses).
    # _merge_lever_constant_manifests below folds them into the launcher constants
    # surface so the schedule-provenance gate classifies the boundary DERIVED.

    from tac.witness_dsl.spec_v9_cgauge import _merge_lever_constant_manifests
    constants = _merge_lever_constant_manifests(constants, tuple(typed.to_program().levers))

    emitted_names = sorted({f for f, _ in _crucible_v7_argv_pairs(argv)})
    manifest = build_launch_manifest(
        program_name=program_name, emitted_flag_names=emitted_names,
        typed_config_hash=typed.typed_config_hash(), typed_validated=True)
    manifest = dict(manifest)

    blockers: list[dict[str, str]] = []
    evidence: dict[str, Any] = {}
    ckpt = Path(BANK_CKPT)
    if ckpt.is_file():
        evidence["warm_fork_ckpt_custody"] = {"status": "CLEAR", "path": str(ckpt),
                                              "bytes": ckpt.stat().st_size}
    else:
        blockers.append({
            "id": "V9C3_BANK_CKPT_CUSTODY",
            "detail": f"the banked ep725 BEST EMA {BANK_CKPT} must exist with sha256 "
                      f"{BANK_CKPT_SHA256} ({BANK_CKPT_BYTES} bytes) at launch time",
        })
    _own_hash = typed.typed_config_hash()
    receipt_row = None
    for rp in sorted(Path("experiments/results").glob("*/dry_start_report.json")):
        try:
            rep = _json.loads(rp.read_text())
        except (OSError, ValueError):
            continue
        if (rep.get("gate") == "full_config_dry_start"
                and str(rep.get("config")) == program_name and bool(rep.get("green"))
                and str(rep.get("typed_config_hash", "")) == _own_hash):
            receipt_row = {"status": "MEASURED_GREEN", "report": str(rp),
                           "typed_config_hash": _own_hash, "ts": rep.get("ts")}
    if receipt_row is not None:
        evidence["composed_bench_receipt"] = receipt_row
    else:
        blockers.append({
            "id": "V9C3_COMPOSED_BENCH_NOT_MEASURED",
            "detail": "no GREEN full_config_dry_start receipt with typed_config_hash "
                      f"{_own_hash[:16]}… exists for {program_name!r} — run the launcher's "
                      "bounded --dry-start ON THIS EXACT CONFIG (it also PROVES the "
                      "warm-fork weight-shape load via resume_ok) before the real launch",
        })

    manifest.update({
        "expected_active_levers": sorted(expected),
        "duty_ticket": {"hwm": "02_horizon_weighted_margin",
                        "step": "03_step_native_activation"}[ticket],
        "arm": arm,
        "paired_off_twin": f"v9c3_duty_{ticket}_off",
        "fork_contract": {
            "checkpoint": BANK_CKPT,
            "checkpoint_sha256": BANK_CKPT_SHA256,
            "checkpoint_bytes": BANK_CKPT_BYTES,
            "resume_epoch": RESUME_EPOCH,
            "mode": "weights-only (fresh AdamW; EMA shadow load; re-seeded guard; "
                    "explicit --warm-start-epoch 726)",
            "no_clean_pre_muon_resume_exists": "the bank npz is a 59-key EMA deploy "
                    "snapshot with NO optimizer/RNG/event state (campaign meta-review "
                    "2026-07-18) — fresh-state fork is the only honest form",
        },
        "resume_event_reanchor": {
            "muon": "OMITTED (no --muon-start-epoch): donor's absolute ep726 Muon is the "
                    "MEASURED confound engine (Lane 0.213->0.238 kick, stuck pose gate); "
                    "pure-AdamW measurement plant; #217 finishing-schedule stays OPEN",
            "pose": "pose-blind (--w-pose 0, compute gates kept, conditioning gate NOT "
                    "composed): reproduces the trunk's realized ep700-725 loss physics and "
                    "removes arm-divergent sigma_min_plateau engagement from the window",
            "lr_rewarmup": f"ResumeLRWarmup: {REWARMUP_EPOCHS} ep beta2-derived window "
                           "(overrides the config-of-record 8; DERIVED > CONFIG)",
            "tau_beta_segform_pre_v0": "#517/#518 item 5: automatic for start_epoch > 1 — "
                    "the v0 verdict reads the true resume-epoch schedule values",
            "ramp_end_reorient": "#518 item 9: automatic when a rewarmup window is set",
        },
        "verdict_windows": {
            "treatment_boundary_epoch": TREATMENT_BOUNDARY,
            "primary": {"epochs": [TREATMENT_BOUNDARY, TREATMENT_BOUNDARY
                                   + PRIMARY_WINDOW_EPOCHS],
                        "verdict_points": [775, 800, 825, 850]},
            "secondary": {"epochs": [TREATMENT_BOUNDARY, RUN_EPOCHS],
                          "verdict_points": [775 + 25 * k for k in range(10)]},
            "note": "step ticket: the treatment is active from resume 726; the "
                    "726-753 rewarmup transient is reported separately (response "
                    "receipt), the paired verdict windows start post-rewarmup",
        },
        "open_518_items": {
            "margin_step_cap": "OFF in BOTH arms — the measured cap does not exist "
                    "(never-fired; #518 R5); deriving it from the OFF twin's rewarmup-"
                    "window margin/update-norm telemetry is the named unlock",
            "fork_head_solve": "OFF in BOTH arms — never-fired lever; firing it inside a "
                    "duty A/B would add a second unmeasured mechanism (its own duty row)",
            "warm_start_restore_boundary_state": "OFF — the bank npz is keyless (nothing "
                    "to restore; #518 item 8 residual R4)",
        },
        "adverse_findings_surfaced": [
            "AF1 (v9c2 launch.sh header, verbatim adjudication): HorizonWeightedMargin "
            "'measured WEAK by #141/#169 (dS ceiling 0.012-0.024; residual d_seg is "
            "label-noise not reconstruction)' — the A/B is the instrument that converts "
            "this prior into a measured verdict; power: the ceiling is ~5-10x the derived "
            "paired noise threshold (see ticket thresholds)",
            "AF2 (v9c2 launch.sh header): StepNativeActivation 'adjudicated at the c2 "
            "crucible seal to annealed-hosc ... step-native is the alternative arm for "
            "the NEXT fresh arm' — the ep725 fork measures a DIFFERENT estimand "
            "(sharpen a TRAINED trunk from ep726) than the fresh-arm disposition (train "
            "under the sharper endpoint from ep0); both are recorded, neither is "
            "silently substituted for the other",
            "AF3 (measured): the donor run's Muon at ep726 regressed Lane and stuck the "
            "pose gate — this fork OMITS Muon entirely (pure-AdamW plant)",
            "AF4 (estimand): the composer-declared step delta '3.177 -> 8.0' is the V9 "
            "mod19 LawRef custody surface; the c2/mod32 vehicle's flag value is 4.0 "
            "(config of record) — the realized one-flag contrast here is beta-end "
            "4.0 -> 8.0. DERIVED consistency: beta(726) on the control schedule = "
            "1 + 3*(725/999) = 3.1772 — the '3.177 sealed endpoint' IS the schedule "
            "value at the fork epoch; the treatment repositions beta(726) to "
            "1 + 7*(725/999) = 6.0801 (#517 positions both pre-v0; the v0 verdict "
            "is the immediate-response receipt)",
            "AF5 (schedule realization, measured): the donor run FROZE beta at 3.1772 "
            "from ep726 (FEED-fm Muon freeze) and never realized the 3.177->4.0 tail; "
            "these Muon-free arms anneal it live under AdamW — shared by both arms, "
            "one-flag contrast preserved",
        ],
        "held": True,
        "operator_go_required": True,
        "launcher_registration_owed": f"tools/launch_witness_run.py --config {program_name} "
                                      "(3-line registration at GO time, the c2 pattern)",
        "launch_blockers": blockers,
        **evidence,
    })

    governance: dict[str, Any] = {}

    return CrucibleV7LaunchConfig(
        typed=typed,
        constants_manifest=constants,
        dsl_program_manifest=manifest,
        schedule_governance=governance,
    )


__all__ = [
    "ARMS",
    "BANK_CKPT",
    "BANK_CKPT_BYTES",
    "BANK_CKPT_SHA256",
    "CONFIG_OF_RECORD_LAUNCH_SH",
    "CONFIG_OF_RECORD_LAUNCH_SH_SHA256",
    "HOSC_BETA_END_CONTROL",
    "HOSC_BETA_END_TREATMENT",
    "PRIMARY_WINDOW_EPOCHS",
    "RESUME_EPOCH",
    "REWARMUP_EPOCHS",
    "RUN_EPOCHS",
    "SECONDARY_WINDOW_EPOCHS",
    "SURGICAL_ENGAGE_EPOCH",
    "TICKETS",
    "TREATMENT_BOUNDARY",
    "V9C3_DUTY_EXPECTED_LEVERS_COMMON",
    "compile_v9c3_duty_ab_config",
]
