# DAG FEED — FLICKER FLOOR IS FORMULATION-SCOPED, NOT HARD (+ the cohesive-package fire)

FEED id: `FEED-507-flicker-floor-rescope-and-fire`
Date: 2026-07-15 · Research-only: `true` · Pointer moved: `false` (0.19108 submittable / 0.18804
banked-borrowed UNMOVED — a law re-scope + a launch are MEANS)
Operator basis (verbatim, 2026-07-15): *"We have also trained dseg beyond the flicker floor in the
past and it's not a hard floor we just need to understand the math and geometry and engineer
optimally instead of being cargo culted forgetful idiots"* + *"stop deferring and build aggressively
and proactively"* + *"Min wall clock."*

## SIGNAL — the hard-floor reading was drifting into cargo-cult

L85-family summaries had begun citing **0.005318** as "the d_seg endgame floor" unconditionally.
The registered law (`gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1`) never said that:
its own `domain_of_validity` scopes the floor to **temporally-smooth-in-LABEL-space witnesses** and
its own anchor carries the pierce existence proof (0.00086 < 0.00532). The drift is the reading,
not the law — exactly the campaign-level forgetting the triality exists to prevent.

## DIAGNOSTIC — the math (DERIVED, each step cites its source)

1. **What the floor IS** (`gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1`, DERIVED +
   n600-MEASURED): for any witness whose per-pair labels are temporally SMOOTHER than GT (cannot
   reproduce GT frame-to-frame argmax flicker), the temporal-majority oracle is optimal and costs
   exactly 1 disagreement per GT spike ⇒ `min d_seg = q_spike = 0.005318` (n600 stride-2 census).
   The LAW transfers; the CONSTANT is a gauge measurement of THIS video/scorer.
2. **Why it binds ONLY that formulation** (`dseg_covariant_gauge_decomposition_v1`): GT sub-pixel
   advection phase on the 291/128 sampling comb makes the per-pair argmax pair-DEPENDENT
   (`d_seg = d_cov + d_gauge`, `d_gauge ≈ q_spike`). The spikes are DETERMINISTIC functions of
   (ξ, R, scene phase) — 94.8% zero-shift in the ξ-comoving frame; blink-back 0.418. A witness
   carrying the **phase zero-mode through (ξ, R)** reproduces the flicker deterministically and the
   majority-oracle bound simply does not apply to it.
3. **Existence proofs — we HAVE trained below it** (both on the SAME GT/scorer):
   - **Phase proof row** (FEED-ma SIGNAL-A): deterministic spike reproduction through R scored
     **d_seg 0.00086** < 0.005318 (6.2× below), n600 cached-authority argmax.
   - **Ancestor bc36 vehicle** (CLAUDE.md §trilemma, MEASURED): PR95-size pair-conditioned net
     reached **d_seg ~6e-4** on the SAME GT. Per L18 the NUMBER does not transfer to the witness
     vehicle — but as an EXISTENCE proof about the GT/floor (a pair-conditioned net is not blocked
     by GT flicker) it is decisive: the floor is a property of the smooth-label FORMULATION, not of
     the GT, the scorer, or the paradigm.
4. **Verdict-ladder placement**: `verdict_scope = FORMULATION` (smooth-label witnesses). Citing
   0.005318 as a paradigm/vehicle-level hard d_seg floor is a FORBIDDEN cargo-cult claim
   (memory `feedback_flicker_floor_not_hard_fire_phase_stack_stop_deferring_20260715`).

## RESPONSE — what the corrected law LICENSES (the config levers)

The only way below the floor is a witness that is NOT temporally smoother than GT — i.e. one that
carries the phase. The corrected reading licenses (all BUILT; grep `phase_primitives`):

- **TRAIN**: `phase_advection_consistency` T1 lever (#424) — `--seg-phase-advect-weight 0.4
  --seg-phase-advect-start-epoch 726` (engages at the terminal band = the flow's finest
  curvelet/persistence scale, where the sub-pixel phase lives; w_p = 0.4·w_subpix DERIVED from
  blink-back 0.418, `label_floor_to_phase_tail_handoff_v1`).
- **CURRICULUM**: law-5 floor→phase-tail hand-off — fire ⇔ label-smooth stage ∧ d_seg ∈
  [0.00496, 0.00700] ∧ flat (the floor is the DERIVED switch to the phase tail, never an
  early-stop green).
- **STORE**: #425 phase-residual carrier (`tac.boundary_math.phase_residual_carrier`) — the archive
  SHAPE for the phase zero-mode; byte-close via `tools/levelset_byte_close_and_eval.py
  --phase-carrier` (wired; default-off flag on the TOOL, contract recorded in the composed config's
  manifest `byte_close_contract`).
- **SENSE**: `tac.witness_control.label_floor_detector` (costate SENSE organ for the hand-off).

## Registry legs landed with this FEED (APPEND-ONLY)

- `domain_refined` event on `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1`:
  explicit `verdict_scope=FORMULATION`, `forbidden_reading=hard_floor`, `licensed_levers=[...]`.
- `anchor_appended` event on the same equation: the ancestor bc36 ~6e-4 existence proof
  (labelled ANCESTOR-VEHICLE EXISTENCE PROOF, non-transferable as a witness number per L18).

## The cohesive-package fire (STEP 2/3 of the same arc)

- Composed config: `--config c1_optimal_form` (spec_c1_optimal_form_20260715; 22 levers; parent
  v9_cgauge_ideal_mod19_sR). Phase stack ON in-config (T1 0.4@726 + event τ-advance + #315
  start-event sensors with epoch backstops + PoseBlindComputeGate #495 + flip_median head arbiter +
  S_R + speed core + PERF_ENV ~17× kernels). Adaptive-ε (#318/#320) = typed SLOT, NOT folded:
  `--eikonal-viscosity-adaptive` is inert without `--eikonal-viscosity>0` (trainer help: "Requires
  --eikonal-viscosity>0"; default 0.0, never in a sealed config) — folding it would be the
  counted-but-inert #417 fake; unlock = the ticket's bounded n24 stability A/B + a sealed
  viscosity term (adaptivization ticket, `adaptive_eps_cfl_edge_tracking_v1`).
- Dry-start gate (owed-2): the c0prime sibling's 2026-07-15 red report is DIAGNOSED, not a wedge —
  boot+epoch-0 n600 verdict took 28.5 min vs a 300 s boot budget; true marginal ≈ **295 s/ep**
  (ep1 68/75 accum batches in ~4.5 min). Fix = budget the boot honestly
  (`--dry-start-boot-budget-s 2400 --dry-start-per-ep-budget-s 600`), not trainer surgery. The
  composed config's own dry-start (this arc) supersedes the c0prime smoke and doubles as the
  `C1_COMPOSED_BENCH_NOT_MEASURED` receipt (sec/ep + peak RSS measured on the real cache).
  `C1_SR_SIDECAR_CUSTODY`: CLEAR (`gt_n600_sR.npz`, 450 MB, present).

## Triality

- **equations**: the two APPEND-ONLY registry events above (this FEED is their source artifact).
- **DSL**: `spec_c1_optimal_form_20260715` (+ this arc's manifest additions: adaptive-ε slot,
  `byte_close_contract` phase-carrier row, flicker-floor licensing cite).
- **DAG**: this FEED + `SPEC_cohesive_v9max_package_20260715.md` (the package ledger).

Pointer 0.19108 UNMOVED (means/apparatus until the run lands a byte-closed n600 exact row).
