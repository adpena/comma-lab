# crucible v7.5 — BIRTH-STACK COUNTER-FORCE (Chan-Vese area constraint + Morse-Smale completion + regime coherence)

**Date:** 2026-07-08 · **Axis:** `[macOS advisory]` $0, **NO launch** · run-1 (pid 63069) + run dirs UNTOUCHED ·
run-1 config UNCHANGED (it stays the island-birth-arm measurement) · **pointer contest-CPU 0.19110 UNMOVED — MEANS.**
The END is a byte-closed n600 exact row < 0.19110 from `upstream/evaluate.py` AFTER a run.

STORES CONSULTED: `road_anomaly_probe_20260708.md` (the diagnosis) · `t5_crucible/{ORCHESTRATION_LEDGER,
LAUNCH_PACKAGE_v7,SYNTHESIS_seal_v73_round2}.md` · FEED-roadfloor + FEED-07a DAG entries · the
DirectionalBasisRebalance / `persistence_classes_for_basis_regime` / LadderIslandHomotopy DSL levers ·
`logit_adjustment_class_prior_20260707` (the sister equation pattern) · CLAUDE.md (NO-FAKE, level-set-flow,
value-provenance ladder, never-invent-flags) · `docs/operating_manual_craft_handoff.md`.

## 0. MEASURE FIRST — which birth pressures are LIVE in v7 (the fix targets what v7 ACTUALLY emits)

Compiled the real v7 argv (`compile_crucible_v7_config(gt_n600, num_pairs=8, epochs=3000)` →
`to_program().compile_trainer_argv()`). The LIVE lane/movable growth drivers, per class:

| driver | flag(s) emitted | lane (1) | movable (3) |
|---|---|---|---|
| island SEED | `--seed-islands --witness-alone-island-loss --seed-island-eased` | ON | ON |
| island-AMPLIFY (ladder) | `--amplify-weight 1.0 --ladder-island-homotopy` (`--ladder-lane-* / -movable-*`, both λ-gate **0.0 = OPEN**) | ON (VP-tangent arm) | ON (dilation-GO arm) |
| logit-adjust | `--logit-adjust-loss-tau 1.0` (Menon offset lane **−5.14** / movable **−4.39**) | ON | ON |
| persistence-recall | `--persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --persistence-classes 3` | **OFF** (A5/M1 already dropped lane) | ON |

So in v7-pre-fix the LANE over-paint drivers were **logit-adjust(−5.14) + ladder-lane amplify** (persistence
already excluded lane via the seal-v7.3 A5 fix); MOVABLE had the FULL stack. **Neither a precision
counter-force nor a completion event existed.** That is exactly the recall-without-precision imbalance the
probe measured (lane 13.8× / movable 4.6× over-paint at ep125, mass-conserved from Road → Road d_seg
pinned ~0.398). The fix below targets THIS emitted config, not run-1's.

## 1. Lever-1 — the CHAN-VESE AREA-CONSTRAINT precision counter-force (the primary, continuous fix)

Formulated as the AREA-CONSTRAINT Lagrange term of the level-set (Chan-Vese) region energy, per the
operator directive ("level set and Morse-Smale are perfect for engineering the precisely desired annealing
behavior"), NOT an ad-hoc penalty:

    E_area,c(φ) = (λ_c / 2) · max(0, A_c(φ) − A_c^GT)²          (one-sided quadratic region energy)
    ∂φ_c/∂t|_area = −λ_c · max(0, A_c − A_c^GT) · δ(φ_c)        (inward retraction ∝ overshoot, boundary-localized)

Composed with the birth force: `∂φ_c/∂t = [F_birth,c − λ_c·max(0,A_c−A_c^GT)]·δ(φ_c)`. **Equilibrium is the
spec** — no ramp schedule: `F_birth,c = λ_c·(A_c* − A_c^GT) ⇒ A_c* = A_c^GT + F_birth/λ_c`. Discrete
differentiable form: `A_c = m_c = mean_px softmax(realized-SegNet-logits)_c` (the SAME realized forward the
island levers already compute — zero extra forward), `A_c^GT = mean_px [lstar==c]`; the softmax Jacobian
`softmax·(1−softmax)` peaks at the annulus = the discrete `δ(φ)`.

**Deriving λ_c (the balance arithmetic):** choose λ so equilibrium overshoot = δ·A_GT (equilibrium ratio
1+δ, uniform across classes): `λ_c = F_birth,c / (δ·A_c^GT)`. With `F_birth = W_birth = 1.0` (the
amplify/recall loss weight, MEASURED-ANCHOR config-conditional) and δ = 0.25:

    λ_lane    = 1.0/(0.25·0.00585) = 683.8
    λ_movable = 1.0/(0.25·0.0124 ) = 322.6

**Dominance at the measured ep125 runaway** (operator: "at 13.8× the retraction MUST dominate the birth
force") = `(r_obs − 1)/δ`: **lane 51× · movable 14×** → retracts hard, settles at 1.25×GT. **Area returned
to Road** at equilibrium: lane 0.0805→0.0073 (+0.073), movable 0.0568→0.0155 (+0.041), total **~0.114 ≥ the
0.1189 measured Road+Undriv deficit** → the ~9% Road-pixel theft is undone, the Road floor lifts.

λ_c is **DERIVED-LIVE** in the trainer from the loaded GT areas (value-provenance gold standard — no frozen
literal). Epistemic tiers (NO-FAKE honest): FORM + balance + arithmetic = DERIVED/VERIFIED; the absolute λ
SCALE that lands the best d_seg = **ASSUMED_AWAITING_VERIFICATION** (owed to the v7.5 A/B, sister of
DirectionalBasisRebalance's √-optimum).

Equation `chan_vese_area_constraint_birth_balance_v1` (REGISTERED, numpy reference `area_penalty`). DSL
`AreaConstraintBirth`. Trainer `--area-constraint-birth` (+ `-birth-force / -tolerance / -classes`),
default-OFF byte-identical, cfg-export + resume-divergence guarded.

## 2. Lever-2 — the MORSE-SMALE PERSISTENCE birth-completion event (defense-in-depth regime hand-off)

Birth-complete per class iff **persistence ≥ τ_persist** (persistence = 1−within_flip, the Morse-Smale
basin prominence above the argmax margin, read from the existing #333/nucleus `within_flip` telemetry) **AND
part_frac ∈ [(1−δ),(1+δ)]·GT** (settled into the Chan-Vese equilibrium band). On fire (LATCHED, resume-safe
`__bc_*` sidecar) the birth stack ramps 1.0→post_level over ramp_epochs and hands off birth→boundary. With
Lever-1 active this is **DEFENSE-IN-DEPTH** (#302 discipline: the Lagrange multiplier self-limits area
CONTINUOUSLY; the event RE-ALLOCATES the freed capacity — two independent mechanisms).

Engine `tac.witness_control.birth_completion` (pure predicate + ramp + per-class controller + resume
roundtrip) is COMPLETE + tested. DSL `BirthCompletionEvent`. Trainer: the DETECTOR + LOUD hand-off telemetry
+ resume state are LIVE (byte-neutral observability, fed the per-class verdict stats). **HONEST SCOPE:** the
birth-stack RAMP APPLICATION to the loss surfaces (modulating amplify/logit-adjust/persistence per-class) is
the **OWED integration** — I did NOT half-wire the per-class multiplier into the three loss surfaces of the
LIVE-LAUNCH trainer under a byte-identity/resume risk I could not fully verify in this unit. Composing the
lever activates the detector (real completion detection → calibration for when the ramp lands); it is
default-OFF byte-identical otherwise. Per NO-FAKE "do LESS but make it REAL": the detector is real, the ramp
is named-and-owed, nothing ships broken.

## 3. Lever-3 — LOGIT-ADJUST REGIME COHERENCE

Under `lane_offloaded` the basis is freq_along≈6 (Candès–Donoho cartoon scale, cannot represent the ~25-cyc
dash comb — lane rides the FREE analytic band), so the v6-inherited −5.14 lane RECALL boost demands lane
skeleton a frequency-starved render physically cannot produce AND fights the band. Companion law
`logit_adjust_classes_for_basis_regime(lane_offloaded)="3"` (movable only) — sister of the A5
`persistence_classes_for_basis_regime`; **both regime-derived subsets now AGREE** (coherent). Trainer
`--logit-adjust-classes` masks the offset vector (movable −4.39 retained, capped by Lever-1). Default "all"
= byte-identical incumbent.

## 4. Composition into crucible_v7 (v7.5)

`_build_crucible_v7`: `--logit-adjust-classes = logit_adjust_classes_for_basis_regime(regime)` (="3");
levers tuple += `AreaConstraintBirth()` + `BirthCompletionEvent()`. `dsl_levers` 5→7; diff-vs-v6 added +11
flags. Compiled argv verified: all 11 new flags emit, `--logit-adjust-classes 3` + `--persistence-classes 3`
agree, `--logit-adjust-loss-tau 1.0` retained. Every emitted flag is DECLARED in the trainer argparse
(never-invent-flags — no argparse crash).

## 5. Hostile round-1 self-review (own it before the coordinator does)

1. **"λ~684 is absurdly large / will explode the loss."** The penalty is one-sided quadratic on relu(m−A_GT)²
   ≈ (0.07)²≈0.005; λ·0.005≈3.4 ≈ the ep125 seg-loss scale (4.15). Sane magnitude. The stiffness lives in
   the tiny A_GT (a rare class needs a stiff spring to cap a small absolute area); `--per-group-grad-clip`
   (ON in v7) + the one-sidedness (zero below A_GT) bound the transient. NOT a red flag — the part_frac
   normalization.
2. **"The area is on the WITNESS soft mass or the SegNet-realized mass?"** REALIZED (`softmax(_slog_wa)`,
   witness-alone = deploy render) — the ACTUALLY-scored partition, reusing the forward already computed when
   island levers are on. If area-on but island-off, `_island_levers_on` is extended so the realized forward
   is still computed (no silent skip).
3. **"F_birth=1.0 is unmeasured — this is a vibes constant."** Declared MEASURED-ANCHOR config-conditional
   (= the amplify/recall weight, which IS 1.0 in the argv) with a re-derive trigger; the absolute λ scale is
   explicitly ASSUMED_AWAITING_VERIFICATION owed to the A/B. The FORM + balance + dominance are derived and
   don't depend on the scale being exactly right (any λ in a wide band caps the runaway; the A/B tunes it).
4. **"Lever-2's ramp isn't wired — is composing it FAKE?"** No: composing activates a REAL detector +
   telemetry (it detects completion + emits LOUD hand-off rows, resume-safe). The loss-surface ramp is named
   OWED, not claimed done. Byte-neutral. Honest per the operating manual.
5. **"Byte-identity when off?"** All three levers default-OFF: `--area-constraint-birth` absent → `_area_lambda`
   None → term skipped; `--birth-completion-event` absent → controller None → no observe; `--logit-adjust-classes`
   default "all" → `allowed_classes` None → incumbent offset. cfg-export + resume-divergence guards added for
   all (a resume that silently drops/changes them fails closed).
6. **"Did you touch run-1?"** No. NO launch. run-1 config unchanged (v7.5 is a NEW config; run-1 stays the
   birth-arm baseline). pid 63069 + run dirs untouched.
7. **"Verdict scope?"** Every claim here is INSTANCE/FORMULATION-level design on the v7.5 config; the
   candidate floor-law is flagged for council, NOT registered as a family kill. The λ scale + Lever-2 ramp
   are pre-registered as owed A/B measurements.

## 6. Commits + tests

39 new tests (`test_v75_birth_counterforce.py`) + 103 crucible/launch tests green; ruff F clean on all
touched files. Equation registered. Files: `chan_vese_area_constraint_birth_balance_20260708.py` (equation),
`birth_completion.py` (engine), `curriculum_dsl.py` (3 DSL factories + companion law), the trainer (Lever-1
loss + Lever-3 mask + Lever-2 detector + argparse + cfg-export), `witness_autoconfig.py` (v7.5 compose),
`canonical_equations/__init__.py`, and the two crucible test files (v7.5 expectations).

**Pointer 0.19110 UNMOVED — this is APPARATUS/MEANS. The next unit launches v7.5 (operator GO) and byte-closes
the n600 row; the Lever-1 λ scale + Lever-2 ramp are the owed A/B measurements.**

## 7. RAMP-LANDED — the owed Lever-2 loss-surface ramp application (2026-07-08)

The §Lever-2 OWED integration (the birth-stack RAMP APPLICATION) is now BUILT. On per-class
birth-completion (detector already live), the birth pressures for THAT class ramp DOWN over a
derived window to their post-birth level, PER-CLASS INDEPENDENTLY (lane + movable complete on their
own latches). New switch `--birth-completion-ramp` (default OFF => DETECTOR-ONLY, byte-identical
loss; requires `--birth-completion-event`, fail-closed). crucible_v7 (v7.5) composes it ON.

### 7.1 What ramps, and how (per-class, exact where separable)

- **logit-adjust offset** (`_LogitAdjustSegAdapter`): a per-epoch mutable offset cell holds
  `base_offset * per_class_multiplier_vector` (`birth_ramp_multiplier_vector`). EXACT per-class
  (the (5,) offset scales element-wise); pre-fire the vector is all-1.0 so the cell == base offset
  (bit-exact fp32 ×1.0) => byte-identical until a class fires.
- **persistence-recall** (`persistence_topology_loss_mlx`, new optional `recall_class_scale`): scales
  each persist class's RECALL contribution by its multiplier (clDice topology UNSCALED — retain
  precision, hand off birth recall). `None`/all-ones => byte-identical (verified). EXACT per-class.
- **island-amplify** (`island_birth_perclass_from_signed_mx`): the combined mean-1 island weight is a
  weighted-MEAN (a ratio), NOT additively per-class-separable. Resolution: split the (ladder-
  maintained) combined weight into DISJOINT lane/movable portions via the self-detected GT masks
  (`movable = movable_mask & ~lane_mask`, lane priority => the two PARTITION the any_mask support) and
  combine the per-class weighted-mean birth terms by their support FRACTION. IDENTITY (measured
  |diff|=0.0): with both multipliers 1.0 this EQUALS the single combined term, so a completed class
  hands off INDEPENDENTLY of the still-growing class (verified: lane→0 keeps movable's share, mov→0
  keeps lane's, and the two shares sum to the combined). The split masks are rebuilt in LOCKSTEP with
  the ladder's per-class radii (captured in both the initial build and `_ladder_build_iw`), so the
  partition tracks the grown support. The combined→split switch is gated on `amp_active` (any island
  class fired) so the OFF/pre-fire path stays the incumbent combined term (byte-identical / ULP-close
  at the fire epoch where the multiplier is still 1.0).

### 7.2 Derived post-birth level + ramp window (provenance)

- **post_level = 1 − τ_persist** (`derive_post_level_from_persistence`; DERIVED-AT-CONFIG, no magic
  literal — crucible reads `round(1 - _CRUCIBLE_V7_BIRTH_COMPLETION_TAU, 6)` = **0.2** at τ=0.8).
  Balance arithmetic (Lever-1): with birth force ramped to `post_level·F_birth` the Chan-Vese
  equilibrium sits at `A* = A_GT·(1 + post_level·δ)`. At completion `persistence ≥ τ_persist` means a
  fraction τ_persist of the class's GT support is FORMED (above the argmax margin); only the unformed
  tail `1−τ_persist` still needs birth pressure => retain exactly that fraction. At τ=0.8, δ=0.25 the
  residual equilibrium overshoot = 0.2·0.25 = **0.05** (A* = 1.05·A_GT) — a tight PRECISION band well
  inside the completion band [0.75,1.25]·GT. Epistemic: the FORM (residual ∝ unformed fraction) is
  DERIVED; the absolute best post_level is ASSUMED_AWAITING_VERIFICATION (owed to the A/B, sister of
  the Lever-1 λ scale). post_level=0 (full hand-off) is the τ→1 limit and remains the engine default.
- **ramp_epochs** (default 50): DERIVED-AT-CONFIG as a sub-stage smoothing timescale ≈
  `curriculum_min_stage_epochs/3` (150/3 = 50) — slow enough not to trip the spike-guard jump
  detector, fast enough to free capacity within a stage. The exact fraction is
  ASSUMED_AWAITING_VERIFICATION.

### 7.3 Resume safety (proof sketch)

The per-class ramp multiplier is a PURE function of `(latched fire epochs, epoch, ramp_epochs,
post_level)`. The ONLY run-varying state is the latched fire-epoch dict, which now rides the resume
registry as the `birth_completion` FunctionResumable (prefix `__bc_`, additive):
`birth_completion_state_arrays` writes `__bc_fired_class/epoch` (+ params); `birth_completion_apply_
restore` restores them INTO the live (argv-derived) controller. A crash-resume therefore reconstructs
the IDENTICAL subsequent multiplier trajectory (verified: `birth_ramp_multiplier_vector` agrees pre-
and post-restore at every epoch). Additive contract: a legacy sidecar (no `__bc_*`, or empty fired)
restores to un-fired => byte-identical PRE-FIRE behavior. The event/ramp/params are cfg-exported
(`__cfg_birth_completion_*`) + F2-divergence-guarded, so a resume that silently drops the ramp or
changes τ/band/ramp/post_level fails closed. OFF (controller None) => write `{}` => no keys => no
manifest (byte-identical, per the FunctionResumable non-event contract).

### 7.4 Tests + gates

`src/tac/tests/test_v75_birth_ramp_application.py` (19 tests): post_level derivation; multiplier
vector identity/per-class-independence/out-of-range; apply_restore round-trip + legacy-unfired +
stale-class + None; DSL ramp_apply flag emit + post_level validation; crucible v7.5 argv carries ramp
+ DERIVED post_level; island per-class identity (hinge + softplus) + partition/independence;
persistence recall-scale None/ones byte-identity + per-class effect + length-mismatch fail-closed;
trainer argparse exposes `--birth-completion-ramp`; resume-registry `__bc_` wiring round-trip. Full
related sweep (crucible/v7.5/resume/autoconfig/DSL) 310 green; ruff F clean; dry-run chain green
(crucible_v7 = 150/150 flags validated, DSL-config gate OK, launch.sh carries `--birth-completion-ramp`
+ `--birth-completion-post-level 0.2`; proven_base OFF-path carries NO ramp flag).

**Pointer 0.19110 UNMOVED — APPARATUS/MEANS. The next unit launches v7.5 (operator GO) and byte-closes
the n600 row; the post_level scale + ramp_epochs fraction are the pre-registered owed A/B measurements.**
