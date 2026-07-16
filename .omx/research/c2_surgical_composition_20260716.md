# c2_surgical_warm — the train-least/Kolmogorov composition + warm-start decision (2026-07-16)

**Directive:** operator 2026-07-16 (verbatim): *"Continue with all and keep Kolmogorov philosophy and
projection and realization in mind as ideal and let's only train the absolute least amount and most
surgical targets possible."* (memory `train_least_surgical_kolmogorov_projection_realization_doctrine_20260716`).

**State:** HELD / `operator_go_required` / COMPOSED-NOT-FIRED. **Pointer 0.19108 UNMOVED — a config is
MEANS.** No launch, no dry-start, no paid dispatch, no evaluator call was run by this unit. The live c1
run dir (`levelset_n600_witness_20260716T014623Z`) was not touched.

**The composed object:** `tac.witness_dsl.spec_c2_surgical_20260716.compile_c2_surgical_warm_launch_config`
→ program `c2_surgical_warm`, typed-config hash `be29562e3edffbbba26f4a793d9dbf37e65b7bb99fdde4b40ed68553869ea195`,
85 emitted trainer flags (argv sha16 `ee7933f3c9ef1806`), wall-clock budget 3.88 d DERIVED
(`scorer_throughput_gate.derive_wall_clock_budget_days(1400)`). Compiles + validates clean (DSL
never-invent-flags gate, expected-lever accounting, required-actuation argv checks, dead-sensor refusal).
Launcher registration (`tools/launch_witness_run.py --config c2_surgical_warm`) is OWED at GO time — the
launcher is untouchable to this landing (3-line registration).

---

## 1. Disposition ledger — the doctrine applied per residual bucket

Residual source: the witness-own decomposition (`c2_witness_own_decomp_20260716.md`, MEASURED n600 on the
mod32cap ep650 trunk, d_seg 0.003146 through the exact R + frozen CPU SegNet). Per-stage Kolmogorov test:
*"can a solve, a projection, or a compiled generator produce this instead?"*

| bucket (% of witness residual) | disposition | vehicle | Kolmogorov justification |
|---|---|---|---|
| **Road-Lane boundary flicker (66.0%)** | **TRAIN (surgical stage b)** | T1 `PhaseAdvectionConsistency` (#424, w=0.4) + #360 `MarginBandSatisficing` (w=0.2, msafe auto-resolved 0.03918 from the MEASURED δ_R artifact), ONE engage boundary ep700 | flat-amplitude carriers MEASURED EXHAUSTED on this trunk (decomp §3: every flat band net-negative, β→0 bracket ≈0); post-hoc storage of the phase is the L68-analogue dead end — the render-side sub-pixel phase coherence is produced by NO deterministic generator; the STORED zero-mode side ships as the #425 phase-carrier SEED at byte-close |
| Movable border flicker (21.3%) | DEFER (not this run) | ξ-tracked per-object border carriers (decomp §4 rank-2) | the phase stack's classes are GROUND-only {0,1,2} (the plane homography is wrong for Movable — trainer contract); a Movable-specific carrier is a separate measured arm, not a silent bolt-on |
| Road-Undrivable horizon flicker (15.1%) | TRAIN (rides stage b) | same T1 term — class 2 is in the GROUND set | same phase mechanism, same term; no extra budget |
| Road-MyCar hood rim (9.6%) | SEED + rides stage b partially | static hood-tex seed (1.7 KB) already banked; rim persist 0.054 = flicker → the joint phase term is the only trainable lever; MyCar is NON-ground → NOT in T1's class set | the static component is a compiled generator (rule-118-free code + counted seed); the flicker component on a non-ground class has NO licensed trainable term this run — recorded gap, not a silent claim |
| saddles (0.7%) | DROP (no bytes) | precision annotation on edge carriers (necessity solver) | measure-zero; no training can be justified |
| **pose (d_pose axis)** | **TRAIN (surgical stage c)** | `PoseFinishConditioningGate` (sigma_min_plateau; backstop ep1000) + `PoseBlindComputeGate`; banked R1 dxi (7.2 KB, d_pose 0.001610) as the never-blocks fallback | the photometric wall is MEASURED (5 post-hoc formulations dead, L68) — pose-legible photometrics exist only under joint descent; the stored dxi is the SEED/fallback, not the mechanism |
| argmax partition trunk | **HOLD (warm start — NOT re-trained)** | mod32cap ep650 weights, EMA-shadow load, weights-only | the trunk is at its flat-basis optimum (decomp §3 mechanism) — re-descending it from scratch trains what a checkpoint already realizes; 12.5 d of c1 buys what `--resume-from` gives for free |
| affine head (~791 params) | **SOLVE slot (owed)** | #341 full-P GN/CG in-trainer solve — BUILD OWED (`solve_dont_train_inventory` row 1); the wired `HeadOffsetSolver(flip_median)` advisory arbiter runs meanwhile and emits the owed through-R delta rows | near-quadratic basin CONFIRMED (LM ρ 0.847/0.868); a solve replaces tail epochs; K<P subset solve is a measured FORMULATION NO-GO |
| per-class OT offsets b_c | SOLVE slot (unwired) | `laguerre_logit_offset` damped-Newton BUILT; ot_newton MEASURED-worse at this exact checkpoint → flip_median is the consumed arbiter | a <1 s byte-free solve; wiring is the unlock, not training |
| rate / carriers | SEED + byte-close surface | #425 `--phase-carrier` REQUIRED on post-engage byte-closes; edge floor/openpilot priors stay compiled generators | rate = \|program\|+\|seed\|; nothing rate-side enters the training loop |
| re-anchor window (ep651-700) | TRAIN (bounded, 50 ep) | incumbent loss, fresh AdamW moments | weights-only warm start discards optimizer state; no solve reproduces fresh-moment settling on the exact loss; bounded to ~2× the stage-transition rewarmup scale |

## 2. Warm-start-source decision (MEASURED reasoning)

**Decision: warm-start the mod32cap EMA-best ep650 checkpoint, same-architecture, weights-only.**

| candidate | trunk d_seg | verdict | evidence |
|---|---:|---|---|
| **mod32cap ep650** | **0.003146** (n600 through-R) | **SELECTED** | best held trunk; the surgical-target map was MEASURED on this exact checkpoint; same-architecture resume = full weight-shape compatibility; `--warm-start-weights-only` gives fresh moments + re-seeded spike guard + EMA-shadow load (the trainer's own DE#3 machinery) |
| v9·CGauge coherent ep150 | 0.0348 | REJECTED | 11× worse trunk; warm-starting it still requires the whole descent |
| c1 from-scratch (live, dry-started) | — | NOT DUPLICATED | it IS the from-scratch arm; 12.5 d measured projection; c2 does not re-compose it |

**Why the v9 architecture cannot host this warm start (DERIVED from the config diff, not assumed):**
mod32cap = `--mod-dim 32 --self-orient --n-dir-freqs 4 --freq-across 32 --freq-along 8` (dir feats appended
→ in_feat/first-layer shape) vs v9/c1 = `--mod-dim 19`, no `--self-orient` → per-pair code (600,32)≠(600,19),
FiLM dims and first-layer shapes differ ⇒ weight-shape-incompatible. The v752 factory (which does support
self_orient) carries a lane-band/ladder/dash-comb lever stack the checkpoint never trained under — render-side
drift that would invalidate the held d_seg at the resume epoch (and its self_orient arm pins `--freq-along 6`
vs the checkpoint's 8). Hence the c2 base is the checkpoint's **own launch.sh flag-for-flag** with three
provenanced deviations (l7 exclusion, w_pose 1.0, anneal-epochs 1000).

**What the v9 scientific deltas lose here, honestly:** S_R reachability, CGauge, mod19-Whitney are
descent-efficiency levers for a from-scratch run; the warm path does not descend the trunk, so it forgoes
them by design. The 19-vs-32 family A/B and the S_R treatment remain owned by the c1 arm.

**Basin lock-in (#253/#475) is the INFERRED risk, not measured:** a warm trunk may resist reorganization.
For a sub-pixel-phase objective this is the small-deformation regime around the held optimum — exactly where
staying in the basin is desired. The **deciding A/B is already free**: the c1 from-scratch arm exists
(composed, dry-started); if c2_surgical_warm plateaus without piercing ~0.0031, the fresh arm is the
counterfactual. No additional deciding run needs designing.

**Schedule continuity is the trainer's own contract, not a hack:** `--anneal-epochs 1000` pins τ/β/LR to the
original 1000-epoch plant (the flag's documented warm-start use), so ep651 resumes at exactly the checkpoint's
schedule state (β≈2.95→4.0 by ep1000, τ→0.05, then held flat through ep1400).

## 3. The composed WitnessProgram (stages, event-honest)

1. **Re-anchor** ep651-700 (bounded 50 ep): incumbent loss, fresh moments. Watch: verdict d_seg returns to
   ≈0.0034 (no-regression facet check).
2. **Surgical phase stage** ep700+: T1 phase-advection (w=0.4, GROUND classes, band 2.0, `gt_advected`
   θ-independent target = zero batching change) + #360 satisficing hinge (w=0.2, msafe 0.03918 MEASURED-
   artifact-resolved) at ONE boundary (single spike-guard re-treat). **Epoch gate, not label_floor** — the
   c1 event's floor band [0.00496, 0.00700] sits ABOVE resume d_seg ≈0.0034; the sensor can never fire on
   the warm path. The factory REFUSES a config that emits the dead sensor; warm-path recalibration is a
   NAMED owed item.
3. **Muon** ep726 (the checkpoint's own lineage schedule) — re-fires with phase-shaped gradients (A5).
4. **Joint pose finish**: engages on `sigma_min_plateau` (jacobian-basin σ_min de-noised rolling-slope
   plateau), backstop cap ep1000 (LOUD if it fires); `PoseBlindComputeGate` skips PoseNet compute in the
   blind phase; degenerate/never-fired gate ships the banked R1 dxi (DISENGAGED, LOUD — never blocks).
5. **Solve-interleave**: `HeadOffsetSolver(flip_median)` advisory decode-time arbiter (never mutates
   shipped/EMA weights; its realized-through-R rows are the owed A/B instrument). The #341 full-P GN head
   solve remains a SOLVE slot with its named unlock (build owed).
6. **Compress/rate**: nothing in-loop; byte-close of post-engage checkpoints MUST run
   `tools/levelset_byte_close_and_eval.py --phase-carrier` (the #425 section), pre-engage without it.
   Weight-entropy stays event-gated OFF during descent (#157 law; not composed in-loop here).

Poison dispositions: **grad-clip** — this lineage runs clip 1.0 per-group WITHOUT per-param normalize, so
the c1-family "saturation" finding does not transfer (on c1 the clip is normalize-masked/INERT per
`perparam_normalize_masks_all_norm_clipping_c0_confound`); the incumbent magnitude law is KEPT (changing it
at warm-start is an uncontrolled re-treatment) with frac_clipped telemetry ON and the
autoclip/normalize/fixed magnitude-law A/B a NAMED open lever. **No stage skeleton** — stage count is an
output; the only epoch constants carry recorded rationales in `schedule_governance` (dead-sensor A2, lineage
boundary 726, l7 exclusion). **1-thread standard** — trainer-native (`SELECTED_THREADS`). **Default-off
laws**: every score-affecting addition is a registered `Lever` (activation-ledger visible); observability
(component wall-clock telemetry, probe-every 1) defaults ON.

## 4. Wall-clock projection (MEASURED rates; the doctrine's promise checked with numbers)

| path | epochs trained | rate (s/ep) | projection |
|---|---:|---:|---|
| **c2_surgical_warm** | ~749 (651→1400) | **121 MEASURED** (mod32cap lineage: 650 ep in 78,728 s, launch 2026-07-06T11:55:54Z → BEST ts 2026-07-07T09:48:02Z; includes verdict amortization) | **~25.2 h ≈ 1.05 d** at the lineage rate |
| c2 worst-case bound | 749 | 361 (c1's measured post-ep33 steady 325 + verdict 36 — a HEAVIER config; upper bound) | ~75.1 h ≈ 3.13 d |
| c1 from-scratch (reference) | 3000 | 361 MEASURED | **~12.5 d** (wallclock_burndown, unchanged) |

Honest caveats: the c2 composed rate is NOT yet measured (adverse finding A7 — the phase term adds a cheap
per-pair-local residual vs a θ-independent precomputed target; the pose-finish phase adds PoseNet fwd/bwd
only after engage, gated off before). The `C2_COMPOSED_BENCH_NOT_MEASURED` compile-time blocker fail-closes
the real launch until the launcher's bounded `--dry-start` measures sec/ep + peak RSS **and proves the
warm-start weight-shape load (resume_ok)** — that dry-start is the recommended next step and is
operator-visible, so it was NOT run by this unit. Even at the worst-case bound the doctrine's promise holds:
**3.1 d ≪ 12.5 d; at the lineage rate, ~1.1 d.**

## 5. Adverse findings surfaced against the exact composed config (launch-gate law)

- **A1 grad-clip magnitude law**: lineage-native clip 1.0 per-group kept; the c1 "0.5 saturation POISON"
  finding is (i) a different constant and (ii) re-scoped as INERT-confound on c1 itself; magnitude-law A/B
  named open.
- **A2 label_floor sensor DEAD on the warm path** (band [0.00496,0.00700] above resume ~0.0034): epoch
  engage + factory refusal of the sensor flag; recalibration owed.
- **A3 l7 re-activation trap**: naive epoch extension would have fired the l7 defect at its old 1001;
  excluded via start==epochs (the trainer's documented never-runs pattern).
- **A4 τ_end 0.05 vs the v9 P-TAU2 knee 0.31 [0.191,0.543]**: kept lineage-native (the weights are
  conditioned on this schedule; L18 config-conditional constants); a τ re-treatment is a separate arm.
- **A5 Muon ep726-1000 did NOT improve this lineage's EMA-best originally** (best stayed ep650 pre-Muon);
  #217 finishing-schedule OPEN; here Muon re-fires WITH phase-shaped gradients (a different treatment) —
  holistic facet watch.
- **A6 warm-start basin lock-in**: INFERRED small (small-deformation objective), not measured; the c1
  fresh arm is the standing counterfactual.
- **A7 composed sec/ep + peak RSS unmeasured** → dry-start receipt blocker holds (fail-closed).
- **A8 MyCar rim flicker (9.6%) has no licensed trainable term this run** (non-ground class, T1 excluded
  by the trainer's own homography contract) — a recorded gap, not a claim.

## 6. Triality + stores consulted

- **DSL leg**: `src/tac/witness_dsl/spec_c2_surgical_20260716.py` (the factory IS the leg; every
  score-affecting addition is a registered `Lever` factory reused from `curriculum_dsl` — none rebuilt).
- **DAG leg**: FEED-c2-surgical appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations leg**: NO new equation — no new measured law emerged; the composition cites existing ids
  (`witness_own_residual_decomposition_v1`, `laguerre_ot_head_offset_v1`,
  `margin_band_satisficing_threshold_v1`, `label_floor_to_phase_tail_handoff_v1` (scoped OUT on the warm
  path per A2), `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1` (FORMULATION scope, L85
  re-scope)). Dispositions are POLICY under the 2026-07-16 doctrine, recorded as manifest rows.
- **STORES CONSULTED**: doctrine memory · c2_witness_own_decomp (surgical map) ·
  c2_perclass_stratum_carrier_taxonomy · solve_dont_train_inventory (#342) ·
  c1_optimal_form_composition_DAG_FEED + spec_c1_optimal_form (parent pattern + calibrated values) ·
  v9_missing_signal_constants_audit (§A-1/§B/§C) · wallclock_burndown (12.5 d baseline) ·
  perparam_normalize_masks confound · label_floor_detector source (band semantics) · trainer argparse
  (warm-start/anneal-epochs/phase/satisfice/pose contracts, read not guessed) · mod32cap run dir
  (launch.sh config of record, levelset_best.json, train_result) · L65/L67/L68/L78/L85/L86 · #253/#221.

## 7. The GO question (operator)

**GO/NO-GO requested on:** (1) register `c2_surgical_warm` in the governed launcher (3 lines, owed);
(2) run the launcher's bounded `--dry-start` for `c2_surgical_warm` — it produces the bench receipt
(sec/ep + peak RSS on the real cache) AND proves the warm-start load (`resume_ok`), clearing the last
blocker; (3) schedule decision vs the live c1 arm — same M5 Max GPU: run c2 (~1.1-3.1 d) BEFORE
committing the 12.5 d c1 window, or accept concurrent contention (~0.70 safe-frac halves both).
Recommendation per the doctrine: **c2 first** — it is the train-least path to a byte-closed row and its
failure mode (plateau at ~0.0031) hands the baton back to c1 with the basin question measured.

**Pointer 0.19108 UNMOVED — MEANS.**
