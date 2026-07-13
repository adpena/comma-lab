# SPEC — steps-dimension 95-kill activation audit and epochs-to-target law — 2026-07-13

**Scope:** new-file-only build and verification. Do not edit the live trainer, shared DSL,
shared equation registry, canonical DAG, scorer-surrogate files, tile-halo/fp16 files, or any
sibling-owned lane. Do not launch training, a scorer, a provider, a GPU, or a paid job from this
spec. Main reviews all bytes uncommitted.

**Authority:** this is a build specification, not evidence. Only an n600 matched realized-through-R
window may populate an epochs-to-target delta. A missing crossing remains `None`/right-censored;
it is never encoded as zero. Local MLX/CPU results remain advisory and cannot move a score pointer.

## Source-truthed activation state

| lever | implementation state | measurement state | required disposition |
|---|---|---|---|
| FreSh `fresh_frequency_shift_init` | **VERIFIED-SOURCE:** existing default-OFF `FreshFrequencyShift` Lever, matched `FreShInitControl`, fixed-quality slice, trainer init path, receipt path, and resume-registry entry | **UNMEASURED n600:** only n8 launch shells exist; durable blocker says governed n600 is owed. Candidate selection is cold-start-only, while a matching FreSh checkpoint restores its selected frequency/bias and persisted state. | Do not add a DSL class. Emit a cold-start matched-init A/B ticket; permit continuation only from the same persisted FreSh arm. |
| `hardness_oversample_lever5` | **VERIFIED-SOURCE false actuator:** existing default-OFF Lever and candidate-pool row exist, but `order` has `P+n_extra` entries while the live loop iterates only to `P` | **UNMEASURED n600 / BLOCKED-BY-WIRING:** current execution neither guarantees every base pair once nor consumes the promised extras | Do not add a DSL class and do not run an A/B. Emit a repair-gated equal-update ticket. |
| `TerminalSolve` / solve-interleave | **VERIFIED-SOURCE designed-only:** existing `ScheduleDisplay` validates n600 but returns no flags and declares a trainer support gap; full-P in-trainer GN/CG does not exist | **UNMEASURED / BLOCKED-BY-WIRING:** K=8 post-run subset solve is a measured formulation NO-GO; full-P family remains open | Do not invent a flag or duplicate the display. Emit a full-P build-and-A/B ticket. |

These three states supersede any prompt shorthand saying all three are already executable.

## Owned files

The implementation worker may create only:

1. `src/tac/canonical_equations/steps_dimension_epochs_to_target_20260713.py`
2. `src/tac/canonical_equations/tests/test_steps_dimension_epochs_to_target_20260713.py`
3. `.omx/research/sub015_DAG_steps_dimension_95kill_20260713.md`
4. `.omx/research/steps_dimension_95kill_20260713.md`

This spec is also lane-owned. No other file may be changed. In particular, do not append the
shared canonical-equation JSONL registry: expose an explicit idempotent population helper for main
review, but leave invocation to the main integration lane.

## Canonical equation module contract

Define `EQUATION_ID = "steps_dimension_epochs_to_target_v1"` and a frozen typed row for each lever.
Each row must carry:

- lever id and existing DSL surface;
- `n_pairs=600` exactly;
- a measurement-authority rule requiring deterministic NumPy-fp32 realization through actual `R`
  plus the frozen CPU-torch scorer on all 600 states (MLX training remains advisory);
- an exact-speed-configuration rule requiring identical arms and
  `all_requested_speed_levers_on=true`, or a fail-closed blocker;
- status from `{AB_TICKET_ONLY, WIRING_NEEDED}`;
- start custody, target rule, maximum window, and censoring rule;
- control and treatment definitions;
- `control_epochs_to_target` and `treatment_epochs_to_target`, both `None` until measured;
- exact optimizer-update counts and actual solver/HVP step counts, kept separate;
- optimizer-update seconds, recurring non-update seconds per epoch, one-time initialization/solve
  overhead, direct elapsed seconds to crossing, and critical-path versus asynchronous service
  seconds, all `None` until measured;
- an explicit wall-composition-admissible flag and refusal reason;
- authority axis, `score_claim=false`, `pointer_moved=false`;
- narrow `verdict_scope` and a reformulation/reactivation queue.

Expose pure functions:

```text
epochs_saved = E_control - E_treatment
step_fraction_saved = 1 - U_treatment / U_control
wall_seconds = U * t_update + E * t_recurring_nonupdate
               + t_one_time + t_terminal_critical_path
wall_fraction = elapsed_treatment_to_crossing / elapsed_control_to_crossing
                # preferred when both direct elapsed receipts exist
              or wall_seconds_treatment / wall_seconds_control
                # only when every recurring critical-path term is allocated
wall_fraction_saved = 1 - wall_fraction
```

Here `U` is optimizer-update count. Nominal epochs are reported too, but a hardness arm with extra
pair visits must not launder more updates into a fake epochs win. FreSh fixed-quality authority
permits a first crossing at epoch zero, so crossing epochs and update counts are non-negative;
`step_fraction_saved` is undefined/`None` when the control has zero updates and permits a treatment
with zero updates when the control denominator is positive. A zero-update arm must not invent a
per-update timing: direct elapsed-to-crossing plus initialization/terminal custody remains the wall
authority. Fail closed on negative counts, negative overhead, n_pairs other than 600, or any
uncensored `MEASURED` claim with missing crossings/timings. A completed right-censored receipt may
be `MEASURED_CENSORED`: it retains its
window/update/timing observations while crossing epochs remain `None`, and exact savings refuse to
evaluate. Asynchronous scorer service is recorded but excluded from the critical path unless a
measured wait proves otherwise. If the wall receipt says `composition_admissible=false`, wall
fraction and wall savings MUST remain `None`; never fold an unallocated epoch residual into
`t_update`. Wall composition defaults to refused and must be admitted explicitly by verified
receipt custody; an unmeasured row can never set it true. Direct-elapsed mode still rejects
negative counts/costs, and allocated fallback refuses a zero control-wall denominator.
Do not multiply per-lever savings unless the inputs come from a measured sequential composition;
the independent product is only a symbolic scenario and must be labelled `ASSUMED`.

Build a `CanonicalEquation` with zero empirical anchors and
`ASSUMED_AWAITING_VERIFICATION`/ticket semantics in its domain. The equation formalizes accounting;
it does not fabricate an empirical anchor. Do not expose or invoke a registry-population helper in
this lane: canonical registration is blocked until one receipt importer verifies durable schema,
n600 cohort, epoch-0 history, matched config/source/checkpoint hashes, target/censor rule, init
receipt hashes, and (for a `WIRING_NEEDED` ticket) explicit wiring-closure evidence. Raw scalar
copy/build helpers are not canonical producers.

## Frozen A/B tickets

### FreSh — cold n600 fixed-quality ticket

- Cohort: **MEASURED authority size** n600.
- Start: cold seed/config only because this A/B measures initialization. The candidate-selection
  sweep is skipped under `--resume-from`; a non-FreSh checkpoint therefore cannot seed this cold
  A/B. Bit-faithful continuation of either arm is allowed only from that same arm's checkpoint;
  matching FreSh checkpoints restore the selected frequency/bias and persisted FreSh state.
- Arms: existing `FreShInitControl` versus existing `FreshFrequencyShift`; compose the same
  `FreShFixedQualitySlice(eval_every=1, ckpt_every=1)` on both.
- Milestone: `d_seg <= 0.040763`, an **MEASURED advisory reference** reached at epoch 50 in the
  existing n600 coherent-arm trajectory. It is a preregistered crossing threshold, not a predicted
  FreSh result and not a matched-control claim. Custody source:
  `experiments/results/v9_cgauge_432_coherent_arm_20260711/run.log`, review-time SHA-256
  `3860bcf20a341f562e1dd402e281a3298a347f60fa94928cb592ee5dcee480e8`; launch SHA-256
  `bd760505c445d51dc51d0b31eadd5a4d2628261220ffa46e2474ca83f358c601`; axis
  `[macOS-CPU advisory verdict from macOS-MLX training; NON-PROMOTABLE]`.
- Receipt-tool bridge: the existing fixed-quality harness accepts a baseline-derived factor rather
  than an absolute threshold. After the deterministic control epoch-0 verdict, derive
  `threshold_factor = 0.040763 / d_seg_control_epoch0` and require it to be strictly in `(0,1)`;
  pass that factor to the existing harness so its frozen threshold is exactly the preregistered
  absolute milestone. Otherwise emit a parameterization blocker rather than changing the target.
- Maximum window: 50 epochs, an **ASSUMED ticket ceiling anchored to the measured epoch-50
  reference**; first emitted crossing, no interpolation, right-censor at 50.
- Accounting: include FreSh's one-time candidate-sweep scorer calls/seconds and training calls.
- Launch status: `AB_TICKET_ONLY`; this is likely governor-heavy and is not launched here.

### Hardness — repaired equal-update ticket

- Cohort: n600.
- Start custody: the existing stage-Octave1 epoch-251 weights may seed a separately labelled
  weights-only re-treatment study; it is not a bit-faithful resume because changing the persisted
  hardness config correctly trips the resume-divergence guard.
- Precondition: live-trainer owner must consume all `len(order)` entries, assert one visit for every
  base pair plus exactly `round(P*oversample)` extras, and preserve RNG/resume state.
- Arms: `oversample=0.5` (**existing DSL default / ASSUMED policy, not a measured optimum**),
  `source=realized`, same seed and exact update count; control has uniform extras
  (`weighted=False`), treatment has hardness-weighted extras (`weighted=True`).
- Milestone: `d_seg <= 0.040915`, the **MEASURED advisory** epoch-275 value from the existing n600
  trajectory after the epoch-251 stage boundary. It has the same run-log custody/hash above.
- Maximum window: 25 nominal epochs from the common start (**ASSUMED bounded-ticket ceiling**);
  report both nominal epochs and exact optimizer updates to first crossing; no interpolation;
  right-censor at the window.
- Launch status: `WIRING_NEEDED`; running current bytes would measure the wrong actuator.

### TerminalSolve / solve-interleave — full-P build ticket

- Cohort: full P=600 only. K<P subset solves are excluded by the measured K=8 transfer failure.
- Start: freeze and clone the existing #341 premise checkpoint
  `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`
  (review-time SHA-256 `6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca`)
  to control and treatment. Its probe receipt is `reports/basin_finisher_probe_20260707.json`
  (SHA-256 `7515cfe7495526e0dcae656477dc2718180d71f77447e69c23159250ca1afbb2`).
- Precondition: real in-trainer full-P HVP/CG stage; typed default-OFF compiler connection; atomic
  pre/post-solve checkpoints; complete solver/resume state; accept/rollback mutation ledger.
- Milestone: `d_seg <= 0.98 * d_seg_start`. The factor `0.98` is an **ASSUMED policy constant**;
  only the resulting numerical threshold is **DERIVED** from it and the common start's n600
  realized-through-R replay before either arm runs. The noise floor is **UNMEASURED**.
- Control: continue the unchanged terminal training schedule, maximum 250 epochs
  (**ASSUMED ticket ceiling**).
- Treatment: one registered full-P damped-GN/CG stage, maximum one LM proposal with 16 CG steps,
  then exact n600 accept/rollback. The 16-step ceiling is inherited from the measured #341 probe,
  not a promised optimum.
- Accounting: solver HVP steps and wall overhead are explicit; do not call a solve "zero epochs."
- Optimizer updates and solver/HVP steps remain separate; no unmeasured equivalence converts a CG
  iteration into a training update.
- Launch status: `WIRING_NEEDED`; the inherited estimate is multi-hour/heavy, so no local launch.

## DAG and memo contract

The DAG feed must encode:

```text
source audit -> {FreSh ticket, hardness repair gate, full-P solve build gate}
              -> n600 matched crossing receipts
              -> epochs/update accounting law
              -> time/step composition
              -> fire/rollback decision
```

Every negative gets the narrowest verdict scope and a reformulation queue. The memo must lead with
the three per-lever deltas as `UNMEASURED / A/B-TICKET` or `WIRING_NEEDED`, state that composed
steps saved is `UNKNOWN`, and identify FreSh as the only runtime/DSL fire-ready mechanism while
separating that from n600 measurement readiness. It must also state that the current CPU one-thread
result is a measured scorer-forward subcomponent factor, not a measured whole-step multiplier.
Consume `experiments/results/cheapen_real95_tilehalo_fp16_20260713/current_wall_receipt.json`
(review-time SHA-256 `c9ec6b2d7154a69b98dddd5c8a6a47455187fcdd3c0f4ea6afbff28554ac3614`): it
**DERIVES** 295.352 seconds/epoch from measured n600 log timestamps on the observed critical path
and records zero measured asynchronous verdict-service wait,
but declares `composition_admissible=false`, `all_requested_speed_levers_on=false`, and the
training residual unallocated. Therefore it is source evidence for an explicit wall-composition
refusal, not a whole-step factor.

## Verification

Required local, non-training checks:

- compile the new module;
- focused tests for row invariants, epoch-zero initialization/solve crossings, right-censor `None`,
  `MEASURED_CENSORED`, epochs/update and solver-step arithmetic, recurring-cost-aware wall
  composition, non-composable-wall refusal, and all three
  exact status rows;
- import existing FreSh, HardnessOversample, and TerminalSolve surfaces and prove the new module did
  not create duplicate DSL owners;
- run triality drift detection scoped to the new equation/DAG/memo if the tool supports a narrow
  path; otherwise record the global result and unrelated sibling drift separately;
- verify `git diff --name-only` contains only lane-owned new files plus the mandated lane registry,
  audit-log, and subagent checkpoint appends.
