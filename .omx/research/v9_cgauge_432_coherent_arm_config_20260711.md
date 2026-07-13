# Task #432 — the V9·CGauge COHERENT STATE-GATED-SCHEDULE ARM, materialized LAUNCH-READY (2026-07-11)

**Agent:** witness-config architect (P0 #432, operator-directed). **Cost:** $0 — DSL compile +
governed dry-run + dry memory preflight only; **NO launch fired** (CONTAINMENT: heavy GPU launch =
operator-GO). The live #205 run (pid 88030) was untouched (throughput gate deliberately skipped in
the dry-run to avoid GPU contention; it runs on the real launch).
**Pointer 0.19108282 [contest-CPU] UNMOVED — this whole artifact is MEANS.** Every number here is
MEASURED (argv diff / gate output / preflight projection) or DERIVED (cited law) — labeled inline.

**STORES CONSULTED:** `vehicle_v9_cgauge_naming_20260711.md` (design authority) ·
`cgauge_master_action_and_parametrization_20260711.md` (§3 sizing laws, §4 knob table, §6 #299
redesign, §7 SPEC_v9) · `scorer_model_arms_430_schedule_20260711.md` +
`tac.witness_control.schedule_backtest` (the #430 cascade + gates_owed) · the #205 sealed baseline
`experiments/results/levelset_v752_baseline_20260710T185913Z/launch.sh` (the CONTROL) ·
`witness_autoconfig.compile_crucible_v752_launch_config` (the launch-config pattern) ·
curriculum candidate pool (36 rows; the retired `mod_dim_as_capacity_reopen` row reconciled below) ·
SPEC_v75 §8 operating contract · L86 appearance-phase endgame memory.

---

## 1. WHAT WAS BUILT (the deliverable)

**`tac.witness_dsl.spec_v9_cgauge`** now carries the #432 arm as a typed, validated, compiled DSL
program (never-invent-flags; fail-closed):

- `derive_v9_cgauge_432_config()` — the `TypedWitnessConfig` named **`v9_cgauge_432`** =
  the `v9_cgauge` base (crucible_v752 self-orient-OFF + the theory-forced T1 delta) **+ mod-dim 19**.
- `compile_v9_cgauge_432_launch_config()` — the launcher-facing `CrucibleV7LaunchConfig`:
  composes the **#383 `pose_finish_conditioning_gate`** + the **T1 `phase_advection_consistency`
  LEVER** (triality: the lever is the visible owner, riding the expected-active-lever manifest);
  holds the **amber stability values** from the #205 launch (grad-clip 0.5 / coeff-max 25 /
  per-param normalize — the arm may not silently differ from the control on stability);
  enforces `V9_CGAUGE_432_EXPECTED_LEVERS` fail-closed (10 levers); emits the DSL-provenance
  manifest + `expected_stability`; adds the **`seg_phase_advect_start_epoch` DERIVED
  constants-manifest entry** (see §4); reuses v7's LawRef constants + schedule governance.
- `V9_CGAUGE_432_CASCADE_REALIZATION` — the **witness-DSL compile of the #430 bundle** (the
  gates_owed item from the #430 OperatorGoTicket): every `schedule_backtest.CASCADE_STAGES` stage
  mapped to its REAL trainer realization, with every un-realizable bundle element explicitly
  dispositioned (ORGAN-ADVISORY or BUILD-OWED) — cross-checked by test against the #430 module.
- **Launcher wire-in:** `tools/launch_witness_run.py` `--config v9_cgauge_432` (explicit branch;
  fail-loud unknown-name discipline preserved). Also made the pre-existing `crucible_v753` branch
  CLI-reachable (it was in `derive_named_config` but missing from argparse `choices` — a dead
  branch) and added both names to `config_family` so run-identity headers cannot mislabel them.
- **Tests:** `src/tac/tests/test_spec_v9_cgauge.py` — 13 passed (6 new #432 tests: build/validate/
  parse-on-real-parser, argv-delta-exactness, expected levers, cascade↔#430 stage mirror, LawRef
  DERIVED entry, purpose/containment). ruff `--select F` clean on all touched files.
- **Emitted launch-ready artifact (MEASURED, on disk):**
  `experiments/results/v9_cgauge_432_coherent_arm_20260711/{launch.sh, constants_manifest.json}` —
  165/165 flags exist in the trainer argparse; DSL-config gate OK; perf-env prefix
  (GROUPED_BACKWARD + PERSISTENCE_POOL) emitted.

### The exact governed-launcher command (ready to fire on operator-GO — NOT fired)

```bash
.venv/bin/python tools/launch_witness_run.py \
  --config v9_cgauge_432 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --out-dir experiments/results/v9_cgauge_432_coherent_arm_20260711
```

(The real launch runs the full gate chain including the throughput gate + governed admission +
auto-started costate shadow observer. **Pre-launch gates recorded on the config's own purpose
string, all operator-GO: the T1 SEAL + n600 A/B (L86) is owed BEFORE any launch containing the
phase term.** A parallel agent is polishing the memory governor's admission-reservation leak — the
real launch should follow that landing.)

## 2. V9·CGauge-vs-#205 DELTA RECONCILIATION (each choice DERIVED, none guessed)

The full-argv diff vs the banked #205 baseline launch.sh is **MEASURED** (test-pinned:
`test_432_delta_vs_v9_base_is_exactly_the_intended_set`):

| Reconciliation item (prompt) | #205 control | #432 arm | Derivation |
|---|---|---|---|
| **mod-dim 32→17-19** | `--mod-dim 32` | **`--mod-dim 19`** (CHANGED) | `cgauge_whitney_moddim_v1`: 2·8+1 = 17 + 2 gauge margin = 19 (LawRef-evaluator-executable). Support (MEASURED): live #205 eff_rank 16.4 rising < 19; ~10/32 dims ablation-neutral-or-harmful; UU-2 — mod-17-19 valid for SDF-like charts, which THIS trunk is. Rate: latent table ≈ −40% (D18 ~7KB class). Risk bounded by the pre-registered #299 Arm-A rule: revert to 32 if d_seg residual > +2% vs the mod-32 control. **Pool reconcile:** the retired `mod_dim_as_capacity_reopen` row closed mod-dim-as-capacity-INCREASE (#300 island-gradient-starvation was the real wall); #432 moves the OPPOSITE (parsimony/rate) direction — no conflict. |
| **appearance-phase machinery ON** | absent | **T1 `--seg-phase-advect-weight 0.4` + start 726** (+4 explicit companion flags at trainer defaults), composed as the `PhaseAdvectionConsistency` LEVER | Theory-FORCED: flicker-floor law (no label-smooth witness pierces 0.005318; the sub-0.15 need is 4.5–7× below). w=0.4 = Law-5 fraction (measured blink-back 0.418). Start epoch DERIVED (§4). The #425 phase-carrier is STORE-side (`--phase-carrier` in the #406 byte-close tool, not a trainer flag) — terminal intent recorded here, argv-inert by construction. |
| **texture-trunk drop** | absent | absent (HELD) | #395 DROP on both axes (2 kills: #417 build-INERT + covariance residual is events/gauge, not hf-texture; pose mirror: dominance). VERIFIED: no TextureTrunk lever / `--out-tex-*` flags in the composed set. |
| **hidden-dim resize** | `--hidden-dim 96` | 96 (HELD) | Naming-memo spec: hidden sized to rank-8 + gauge margin with NO texture-trunk budget — 96 IS that sizing (master action §4: FREE, margin factor empirical; 2× memory headroom measured live). A resize has no derived target ≠ 96; changing it would be a guess. |
| **pose via banked-R1 dxi** | dxi channel + `--pose-carrier` table/generated + #383 gate | IDENTICAL (HELD) | Pose's +2 DOF are partition-invisible (MEASURED, d_pose mirror) → dxi (6+k), not the trunk; banked R1 floor (d_pose 0.001610 → 0.127) ships; terminal joint finish engages on `sigma_min_plateau` (#383). |
| **coherent/organ-driven schedule vs event-triggers** | events armed | events HELD + the cascade map (§3) | See §3 — the honest reconciliation. |
| **everything else** | — | HELD (MEASURED: zero other adds/drops/changes) | A/B cleanliness — the arm changes ONLY what V9·CGauge + #430 specify. |

## 3. THE COHERENT SCHEDULE — what "#430 on this trainer" honestly IS

The #430 winning shape is a state-gated cascade (island-birth → boundary-form → τ-sharpen⊕repair →
finish) with gates on JOINT state, epoch values demoted to backstops. **The trainer-native
realization of that shape is the event system the sealed config already arms** — and the
schedule-provenance gate now attests it end-to-end (MEASURED, dry-run output): every emitted
schedule token classifies EVENT_TRIGGERED (muon powerlaw_meat / lane-band lane_nucleus / chroma +
screw annulus_plateau / τ event-mode) or FAIL_SAFE_CAP (the epochs, each naming its governing
event) or DERIVED (polyak, T1) — **zero NAKED epochs**. Stage-EXIT is state-gated too
(`--birth-completion-event`); the finish gates ride `sigma_min_plateau` (#383) + the
`tail-stop-marginal-s` costate readout (SELF_DERIVING).

`V9_CGAUGE_432_CASCADE_REALIZATION` (importable, test-mirrored against
`schedule_backtest.CASCADE_STAGES`) records the per-bundle disposition. The elements the flag
space CANNOT realize are stated, not smuggled:

- **Per-class-λ budget shifts** (the #430 `_shift_toward` mechanism, the replay's winning
  intervention): per-verdict loss-weight mutation = FORBIDDEN in-run (loss weights at stage
  boundaries only; live-config mutation = operator-GO). Disposition: **ORGAN-ADVISORY** — the
  score-neutral shadow observer auto-starts with the governed launch; recommendations arrive as
  OperatorGoTickets. The LADDER per-class λ-gates (`--ladder-*-lambda-gate`) stay 0.0 = UNGATED:
  a nonzero per-class λ floor has **no measured derivation** — setting one would be a guess.
- **T1 event engage** (`label_floor` detector as a trainer event): **N7 BUILD-OWED**; the static
  726 stands as the DERIVED approximation (§4).
- **weight-entropy plateau-gating** (the cascade's finish-bundle variant): no trainer start-event
  flag exists, and the always-on form is the MEASURED-BINDING state (~15% direction share,
  91→83KB) — dispositioned ALWAYS-ON, variant organ-advisory.

**Curriculum-pool sweep at finalization (P0 orphan-class duty):** pool = 36 rows
(12 needs-build / 14 built-never-fired / 5 armed / 3 retired / 2 reformulation-queue; 28 owed).
NO never-fired rung was folded blind into this arm (P12: each rung is its own increment — folding
any would also break the A/B). Two rows APPENDED: `phase_advection_consistency_t1` (armed, lever
`PhaseAdvectionConsistency`) + `mod_dim_19_whitney_adoption_432` (armed, config scalar, with the
capacity-reopen reconciliation on the row).

## 4. GATE RECORDS (all MEASURED from the governed dry-run, 2026-07-11)

- **Memory preflight — PASS.** Projected peak **24.48 GiB** (fixed 15.0 + cf_mx_cache 0.07 +
  gt 3.41 + verdict 6.0) vs safe ceiling 108.8 GiB (85% sole-workload) — and vs 89.6 GiB at the
  70% concurrent frac in the standalone run. **System admission: ADMIT even with the live #205 run
  counted** (active_jobs=1, projected system-used 55.0 GiB ≤ adaptive ceiling 106.4 GiB, headroom
  51.3 GiB). The mod-19/no-texture-trunk arm projects BELOW the #205 envelope, as expected — but
  it was PROJECTED, not assumed. Ledger row appended via `witness_memory_preflight` record path.
- **Schedule-provenance gate — PASS (after a real catch).** First dry-run REFUSED:
  `--seg-phase-advect-start-epoch 726` was a NAKED primary epoch (the gate working as designed).
  Fix (the honest form): a **DERIVED** constants-manifest entry `seg_phase_advect_start_epoch` =
  the muon-cap terminal-band co-anchor under the flicker-floor placement law
  (equation_id `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1`), with the
  N7 static-approximation note. A CAP classification was REJECTED as laundering — no sensor
  actually governs T1's engage in the trainer (verified at the engage-gate source, L9141).
- **Expected-active-lever manifest — PASS** (10 levers match the pinned expectation; the T1 lever
  + #383 gate are composition-enforced — built-but-not-composed extinction).
- **DSL-config gate — PASS** (`v9_cgauge_432`, 165 flags, typed-validated; hash in the manifest).
- **Resumability (P0):** fresh start REQUIRED — a mod-19 trunk cannot load mod-32 checkpoints
  (shape mismatch), so no `--resume-from` is emitted; `--ckpt-every 25 --stage-checkpoints`
  verified in the argv (per-stage + periodic checkpoints; EMA-shadow save per the trainer's
  incumbent path); the run resumes from ITS OWN dir on crash.
- **Safe-compile fingerprint — PASS** (hosc_activation manifest matches host).
- **Throughput gate — SKIPPED in the dry-run** (deliberate: it exercises the GPU and pid 88030
  owns the slot). It runs, un-skipped, on the real launch.

## 5. OWN ROUND-1 ADVERSARIAL REVIEW

1. **"Is this V9-optimal or a v7.5.2 reskin?"** The V9 trunk IS the v7.5.2 lineage by explicit
   spec (the naming memo: V9 = the single covariant trunk of v7.5.2 lineage, texture trunk
   dropped, + the phase endgame + the parametrization identity). The MEASURED delta set (T1 block
   + mod-19 + held amber) is exactly what the V9 spec + #430 add beyond the control — anything
   more breaks either A/B cleanliness, never-invent-flags, or the no-guessed-values rule. The
   danger was cosplaying a bigger delta; the memo states the small-but-derived truth instead.
2. **"Does mod-19 have the capacity the phase endgame needs?"** The phase target is low-dim
   (mostly ξ-generated; the zero-mode constants are per-boundary, stored not learned) — it should
   not widen the trunk's mod. But eff_rank was 16.4 AND RISING at ep125; if the terminal
   eff_rank crosses ~19 the arm is capacity-bound. Mitigation: pre-registered Arm-A revert rule
   (+2% residual threshold) + per-class/flip-spectral instrumentation; the #205 mod-32 control is
   banked for exactly this comparison. RISK STATED, bounded, revert path named.
3. **Two deltas in one arm = confounded attribution.** Acknowledged. BUT the master-action §6
   redesign REQUIRES #299 arms to run WITH the phase machinery active (on a smooth witness all
   capacity arms converge to the 0.005318 floor and the A/B reads ≈0) — so mod-19-with-T1 is the
   DESIGNED measurement, not sloppiness. If 3-arm attribution is wanted, the intermediate
   T1-only arm is already compiled and available (`--config`-less: `compile_v9_cgauge_config`,
   mod-32 + T1) — noted as an operator option, not silently launched.
4. **Three treatments co-anchored at ep726** (Muon backstop / pose-finish backstop / T1 static
   start): in the EVENT-fired path (powerlaw_meat, sigma_min) the first two fire earlier and do
   NOT coincide with T1; only the all-backstop worst case stacks them. The trainer's engage does
   spike-guard re-treat + moment-reset + rewarmup at each. WATCH ITEM for the run's holistic
   check-ins (per-class d_seg + island birth + d_pose + rate — facets, never lineage-scoped).
5. **The DERIVED classification of T1's 726** could be attacked as epoch-laundering. Defense: the
   value is COMPILED from `muon-cap co-anchor × placement law` (inputs recorded; changes if the
   muon cap changes), the un-derivable alternative (CAP naming a sensor that does not govern) was
   rejected, and the N7 build-owed event is named on the entry. That is the value-provenance
   ladder working, not a dodge.
6. **What would falsify the arm early:** the pre-registered lane-reversal falsification at
   annulus_plateau engage (harvest 3b) stands unchanged; the #299 Arm-A signature rule gives the
   mod-19 verdict; the T1 SEAL + n600 A/B (L86) gates the launch itself.

## 6. TRIALITY LEGS

- **DSL:** `tac.witness_dsl.spec_v9_cgauge` (`derive/compile_v9_cgauge_432_*`,
  `V9_CGAUGE_432_{DELTA,PROVENANCE,CASCADE_REALIZATION,EXPECTED_LEVERS}`) + the launcher branch +
  2 curriculum-pool rows. The T1 term is composed as its `Lever` factory (not a hand flag).
- **DAG:** FEED-432-coherent-arm appended to the sub015 DAG (same commit).
- **equations:** N/A-with-reason — no new law falls out of a config materialization; the arm
  CONSUMES `cgauge_whitney_moddim_v1` (evaluator → 19), the flicker-floor law (T1 forcing +
  placement), `costate_lambda_marginal_ds_v1` (the organ-advisory boundary), and the #430
  backtest anchor already registered on it. First candidate law = the measured mod-19-vs-32
  neutrality verdict when the arm runs (would anchor #299/#223).

## 7. OWED / WHAT I DID **NOT** DO (honest close)

- **DID NOT FIRE the launch** (CONTAINMENT). The command in §1 is ready; firing = operator-GO,
  and the T1 SEAL + n600 A/B (L86) is owed BEFORE a launch containing the phase term.
- **DID NOT run the throughput gate** (GPU contention with the live #205 run) — it runs at launch.
- **DID NOT build** N7 (label_floor trainer event), N1–N3 (Movable event force / zero-mode
  conditioning hook / Lane-scoped ν_along), weight-entropy plateau-gating, or ladder λ-gate
  values (no measured derivation) — all remain BUILD-OWED / duty-queued, dispositioned in §3.
- **DID NOT re-derive** anything already settled (already-settled table honored: the pose mirror,
  the texture-trunk kills, the flicker floor, the #430 backtest were consumed, not re-measured).
- **Pointer 0.19108282 [contest-CPU] UNMOVED.** A launch-ready config is MEANS; only a
  byte-closed `upstream/evaluate.py` n600 exact row moves it.

---

## PRE-FIRE REVIEW (#441, re-derived by main — codex wrapper truncated the original verdict)

**Verdict: FIRE-WITH-REVISIONS** (revise the READING/plan, not the code). The #439 SEAL is sound — T1 is stable/byte-identical-OFF/premise-valid and CANNOT destabilize; that axis is cleared. Two FIRE-PLAN gaps (attribution, not correctness); #432 is held on #356+operator-GO so there is time:

- **GAP 1 — phase-OFF control described two incompatible ways.** #439 SEAL: control = `#432 w=0.0` twin (clean single-lever T1, mod-19 held both). THIS memo: control = `#205` (confounds T1 + mod-19; and #205 is DEAD). The clean twin is a 2nd multi-day run, not in the fire command/budget.
- **GAP 2 — mod-19 safety net references a dead control.** #299 revert rule ("revert to 32 if d_seg > +2% vs mod-32 control") needs a LIVE mod-32 comparator; #205 (the mod-32 control) is DEAD → net has no comparator at fire time. eff_rank 16.4 is RISING → mod-19 may clip needed capacity.

**Resolution (decisions, $0):** (1) For THE GOAL, fire #432 as a **whole-arm pointer bet vs the frozen 0.19108** — the pointer IS the control; no twin needed. Only T1-*attribution* science needs the phase-OFF twin (budget explicitly as a 2nd run). Stop calling #205 the "matched control." (2) Own the bundled T1+mod-19 bet — a MISS won't say which delta failed; split or ensure a live mod-32 comparator if attribution matters. Neither gap blocks the SEAL. Pointer 0.19108282 UNMOVED (MEANS).

---

## 9. DIVERGENCE POST-MORTEM + FIXED RELAUNCH — the response branch that was OWED (appended 2026-07-12)

**Operator hit (accepted as legitimate):** *"V9 cgauge was supposed to be optimal ... all of that should
have been turned in in the first place."* The fixed relaunch was NOT re-derivable-from-scratch work — the
counter was a COMPLETED, MEASURED lever the arm simply did not turn on. Honest accounting below, all
`[MEASURED]` from the dead run's own artifacts. Pointer 0.19108282 [contest-CPU] UNMOVED (this is MEANS).

### 9.1 What actually happened (MEASURED from `v9_cgauge_432_coherent_arm_20260711/`)
- Best-ever `d_seg = 0.03482 @ ep150` (`levelset_best.json`). Then the `unify_tau` stage **erased it**:
  `d_seg 0.03482 → 0.04092 @ ep275` (RISING, `costate_shadow.jsonl` `classification: diverging_erasing`,
  `d_seg_rel_slope +1.64e-3/ep` over ep175-275, `stage=unify_tau`). Run died at ep275, no live process.
- **"Optimal" was ASPIRATION, not measurement.** Best 0.03482 is **~7× above** the mod32cap control
  (~0.0047), the #205 CE-floor (0.00496), and the flicker oracle floor (0.005318); **~30-43× above** the
  sub-0.15 need (0.0008-0.0012). The arm never reached the KNOWN baseline even at its best — the L87
  Einstein-pass "single covariant trunk optimal both axes" is the design thesis the run did NOT realize.

### 9.2 The precise gap (a KNOWN failure mode + a COMPLETED lever, both omitted)
- `unify_tau` = τ-sharpening = an MCF-like flow. **MCF erases thin structures** — established finding L75
  (`dash_erasure_homogenization`), the exact mechanism. The lane islands are the thinnest structure.
- The dead run launched with **flat `--eikonal-weight 0.01`** and **NO `--eikonal-weight-end`** (default
  `None` = flat). The eikonal term is what HOLDS the interface/SDF against MCF erosion. Flat-0.01 was too
  weak to hold the thin lane through the sharpen stage.
- `--eikonal-weight-end` EXISTS and its help string is literally *"(#292 control-system) STEP the eikonal
  weight from --eikonal-weight up..."* — this IS task **#286** (COMPLETED: *"RAISE eikonal 0.01→0.05
  COUPLED to τ-anneal"*), a MEASURED lever built specifically to counter τ-sharpen erosion. **It was not
  in the arm.** That is the "should have been turned in": a settled counter-lever to the exact wall, left off.
- The divergence RESPONSE was advisory-only (CONTAINMENT): the costate organ DID detect
  `diverging_erasing` and DID emit `ROLLBACK_TO_BEST_CHECKPOINT` + `config_diffs`, but cannot autonomously
  rollback+relaunch — so the run eroded to death and no fixed config was staged. Detection worked;
  actuation + a prepared response did not exist.

### 9.3 FIXED RELAUNCH CONFIG — ready to fire (HELD: operator-GO; do NOT fire during the 95%-kill P0)
Warm-start weights from the ep150 EMA-best + turn ON the eikonal step-up coupled to the τ/MCF onset
(the #286 counter). Same mod-dim 19 / hidden 96 ⇒ same-shape resume is valid.

```bash
.venv/bin/python tools/launch_witness_run.py \
  --config v9_cgauge_432 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --out-dir experiments/results/v9_cgauge_432_eikhold_relaunch_<UTC> \
  --extra-trainer-flags \
    "--resume-from experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz" \
    "--warm-start-weights-only" \
    "--resume-model-from ema" \
    "--eikonal-weight-end 0.10"     # #286: STEP eikonal 0.01→0.10 across unify_tau to hold the thin lane
```

**HONEST caveat (do NOT oversell):** the eikonal-hold stops the EROSION; it does NOT by itself make the
arm optimal. The best-of-0.03482 = 7× above baseline is a SEPARATE, open question — is the covariant-gauge
trunk under-trained (best at ep150 of a run that died at 275), or does it structurally under-perform
mod32cap on d_seg? The relaunch is the experiment that answers "can the trunk descend past ep150-best once
the interface is held," NOT a claimed pointer-mover. Recommend it AFTER the 95%-kill P0 (GPU-timing
contention would corrupt the throughput measurement) and paired with the trunk-vs-baseline question, not
as a reflexive restart.

**Triality:** `[no new equation]` (uses existing #286 eikonal-coupling law + L75 erasure law). DSL: the
relaunch is expressible via the existing `--eikonal-weight-end` gauge (no new lever). DAG FEED appended.
