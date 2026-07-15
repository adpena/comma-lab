# SPEC — THE COHESIVE V9-MAX PACKAGE (#507): compose + fire ledger

Date: 2026-07-15 · Arm: cohesive-package COMPOSE+FIRE (#507, respawned) · Research-only: `true`
Pointer: submittable **0.19108 UNMOVED** / banked-borrowed 0.18804 UNMOVED (everything here is MEANS
until a byte-closed n600 exact row).

Operator basis (verbatim 2026-07-15): *"stop deferring and build aggressively and proactively"* ·
*"Min wall clock"* · *"Three stages cargo culted is naive and toy"* · the poison taxonomy
(`constants_are_poison...` + `three_stage_skeleton_cargocult...` +
`poison_taxonomy_event_recompute_and_cuda_drift_20260715`) · the flicker-floor re-scope
(`feedback_flicker_floor_not_hard_fire_phase_stack_stop_deferring_20260715`).

Sibling artifacts: `.omx/research/flicker_floor_formulation_scope_DAG_FEED_20260715.md` (the law
re-scope FEED; committed) · registry events `domain_refined`/`anchor_appended` on
`gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1` (2026-07-15T15:49Z) ·
`src/tac/witness_dsl/spec_c1_optimal_form_20260715.py` (the composed factory, THE config SoT — this
ledger only POINTS at it).

## 1. THE CONFIG

`--config c1_optimal_form` (launcher-registered), parent `v9_cgauge_ideal_mod19_sR`:
**leg A** S_R margin-saliency reachability (sole scientific delta of the parent) · **leg B** joint
wall-clock speed stack (fused-R, cache-gt-skeleton, async-verdict, verdict-batch 32/pairs 0,
safe-compile hosc, PERF_ENV ~17× grouped-backward + persistence-pool, component-wallclock telemetry)
· **leg C** consumable deep-math folds (PoseBlindComputeGate #495, HeadOffsetSolver flip_median #386,
**phase_tail_label_floor_event — NEW this landing**) + typed slots for what cannot be folded
honestly. 23 levers, 236 argv pairs, n600, 3000 epochs, seed 0.

**Paired treatment arm:** `--config c1_optimal_form_curvelet_arm` (NEW this landing) — same seed,
same levers; argv delta = `--basis windowed_curvelet` + bank params + out_dir ONLY. It is the
RECEIPT PRODUCER for the owed `curvelet_through_R_dseg_ab` anchor (no-Fourier-basis doctrine:
curvelet explicit opt-in, never a silent default flip; the main config keeps
`legacy_fourier_ab_control` as the deliberate A/B control).

## 2. THE SKELETON IS DISSOLVED (event continuation; stage count = OUTPUT)

Every transition fires on a wired sensor reading trainer-computed streams; every epoch constant is a
LOUD backstop cap (`cap_fired_before_event` = falsification-relevant, S5):

| transition | sensor (fires) | backstop (cap) | sensor stream (poison-law: READS, never recomputes) |
|---|---|---|---|
| τ octave advance | `--tau-advance-mode event` (per-band relaxation; rung count DERIVED `derive_n_octaves`) | per-octave max-dwell | through-R seg-loss history |
| island-birth EXIT | `--birth-completion-event` (τ-persist 0.8, area band) | ramp windows | birth/area stats already in loop |
| lane-band start | `lane_nucleus` (born ∧ formed) | 500 | per-class nucleus counts at verdict cadence (`_wire_sense["lane_ev"]`) |
| chroma boundary | `annulus_plateau` | 450 | `_wire_sense["annulus_series"]` (shared field, appended once per verdict) |
| temporal screw | `annulus_plateau` (SHARED field with chroma — no recompute) | 450 | same shared series |
| Muon finisher | `powerlaw_meat` (+ REV-B nucleation positive control) | 726 | verdict d_seg `history` (incremental fit) |
| pose finish | `sigma_min_plateau` (#383) | 726 | σ_min series (resumable controller) |
| **T1 phase-advection** | **`label_floor` (NEW — law-5 floor→phase-tail hand-off)** | **726** | **`_wire_sense["labelfloor_series"]` = verdict rows {epoch, d_seg, seg_form} appended under the verdict lock (both async + sync sites); detector = `label_floor_detector.label_floor_reached` (all thresholds DERIVED/measured)** |

β/LR co-annealed on the same continuation: hosc β 1.0→3.177 linear (custodied
`hosc_beta_end=3.177`, `hosc_beta_fireband_pin_v1`), LR 1e-3 cosine→1e-4.

Build receipts (this landing): trainer `--seg-phase-advect-start-event` argparse + 5th
`EventBackstopGate` + engage-block wiring (event path via gate; **OFF path keeps the incumbent
`lever_gate_on_at_epoch` comparison VERBATIM ⇒ byte-identical by construction**) + resume prefix
`__pag_` (`GATE_KEY_PREFIXES`) + `event_wirings.label_floor_event` reader +
`schedule_provenance_gate` sensor registration + governance rows (event=fires / epoch=backstops) +
tests (`test_event_wirings.py` +2, static registry coverage green).

## 3. ON/OFF TABLE (every OFF is TRACKED + REASONED — no orphans)

| lever | state | reason |
|---|---|---|
| S_R reachability (LEVER-4 exact through-R) | ON | leg-A parent's scientific delta; forces `--micro-batch-pairs 1` BY TRAINER CODE (batched twin gap = named fallen-crack) |
| T1 phase-advection 0.4 | ON @ label_floor event (backstop 726) | flicker-floor law licenses it; w=0.4 DERIVED (blink-back 0.418) |
| PoseBlindComputeGate #495 | ON | trunk-phase compute saver; pose UNCHANGED (R1 two-phase finisher) |
| HeadOffsetSolver flip_median #386 | ON (advisory) | Hamming-optimal S1; never mutates shipped/EMA weights |
| component-wallclock telemetry | ON | read-only observability defaults ON ("off is a tracked queue") |
| annulus/jacobian-basin/mod-dim telemetry | ON | same law |
| custom grouped-backward VJP (~17×) | ON via PERF_ENV | relaxed-identity 2026-07-15: functional parity IS the gradient bar |
| whole-step megakernel #356 | OFF | MEASURED economics (CPU 0.79–0.83×), not identity |
| Bregman #504 | SLOT | no trainer consumer evidenced (its own FEED: "DSL OWED") — folding = #417 inert fake |
| Fisher-natural trust region | SLOT | `built_not_activated_measurement_owed`, argv-inert |
| #423 Hessian preconditioning | SLOT | `precondition=` not argv-reachable (no trainer flag) |
| adaptive-ε #318/#320 | SLOT | inert without `--eikonal-viscosity>0` (never sealed) AND CFL-edge cure FALSIFIED_MECHANISM at n600 (FEED-06g); unlock = bounded n24 A/B + sealed viscosity term |
| curvelet basis | SLOT in main + **PAIRED ARM staged** | no-Fourier doctrine: opt-in arm produces the receipt; main keeps the legacy control |
| FreSh #448 / FINER++ #310 | EXPLAINED-ISOLATE | `v9_ideal_config_ab_20260713.md`: fresh-start BASIN treatments — "stacking would confound the core"; HELD with no measured number (`v9_missing_signal_constants_audit` §8). Queued as future paired arms, NOT silently off |
| T2 spike-reweight #274 | OFF (duty-queued) | built default-off; engages the same phase regime T1 owns — measure T1 first (attribution-clean), then A/B T2 |
| MarginBandSatisficing | ON (w 0.2, m_safe DERIVED 0.0392) | the marginband_satisfice_fix custody values |
| grad-clip | INCUMBENT (fixed 0.5 + per-param normalize) | CORRECTED per burn-down final report (`perparam_normalize_masks_all_norm_clipping_c0_confound_20260715`): C0's clip saturation is INERT — the per-param normalize AFTER clip divides out norm scaling; the effective magnitude law is unit-norm×LR (SignSGD-like), so the lr/12 mechanism reading is REFUTED. The owed measurement is the magnitude-LAW A/B (incumbent vs normalize-none+AutoClip vs normalize-none+fixed, anchor `autoclip_descent_speed_effect_n24_ab_owed_20260715`); this config does NOT touch --grad-clip-mode/--grad-normalize — the incumbent stays until that A/B measures |

## 4. THEORY→CONFIG TRACEABILITY (6 clusters, consume-or-explain)

| cluster | law / artifact | config realization | status |
|---|---|---|---|
| **curvelet/basis** | `cgauge_curvelet_parabolic_bank_v1` · no-Fourier doctrine · #508 basis custody | `--basis legacy_fourier_ab_control` (typed custody, deliberate control) + dash-comb C2 term + **curvelet_arm staged** | CONSUMED (control + arm) |
| **INR init/activation** | hosc β fire-band (`hosc_beta_fireband_pin_v1` 3.177) · FreSh #448 · FINER++ #310 · StepNative | hosc β 1.0→3.177 co-annealed; FreSh/FINER/StepNative = ISOLATE basin arms (cite above) | CONSUMED (β) + EXPLAINED (isolates) |
| **level-set forces** | σ_cc′ metric closure (`multiphase_sigma_metric_closure_gamma_admissibility_v1`, fitted 2026-07-07) · #360 forces · eikonal L13 · adaptive-ε ticket | `--length-sigma-matrix fitted-20260707` ON · temporal-screw 0.1 @ annulus_plateau · subpix 0.3 · satisfice 0.2 (DERIVED m_safe) · area-Lagrange birth (1,3) · eikonal 0.01→0.05 closed-loop · adaptive-ε SLOT (cited) | CONSUMED |
| **curriculum-as-continuation** | `curriculum_is_continuation_instabilities_are_bifurcations_20260714` · LSI adiabatic rung law · law-5 hand-off | §2 table: ALL transitions event-fired, rungs DERIVED, unify-τ continuous flow, epoch constants = loud backstops; stage count is an OUTPUT | CONSUMED (this landing closed the last epoch-scripted transition) |
| **dynamics instruments** | HCM attribution boundary (run-level A/B) · confound L1 alarms · liveness | component-wallclock + annulus + jacobian-basin + mod-dim telemetry ON; would-fire rows accrue calibration for every sensor; liveness/confound alarms trainer-native | CONSUMED |
| **phase/flicker** | flicker floor FORMULATION-scoped (`domain_refined` 2026-07-15) · `label_floor_to_phase_tail_handoff_v1` · #424 T1 · #425 carrier | T1 ON @ label_floor event; `byte_close_contract.phase_carrier` manifest row (post-engage checkpoints byte-close WITH `--phase-carrier`); `label_floor_detector` = the SENSE organ | CONSUMED |

## 5. LAUNCH RECORD (filled at fire)

- Dry-start gate: `--dry-start-boot-budget-s 2400 --dry-start-per-ep-budget-s 600` (the c0prime red
  report was an unbudgeted boot, DIAGNOSED not a wedge; true marginal ≈ 295 s/ep).
- Min-wall-clock projection: 3000 ep × ~295 s/ep ≈ **10.2 days** to full; first decisive stage-read
  = the CE/τ floor-approach + label_floor/lane_nucleus would-fire calibration well before ep 500
  (~1.7 d) — under the ~3 d first-read bar, so epochs NOT trimmed (trim would cut the terminal band
  the phase tail needs).
- Memory: single run projected ~67.6 GiB peak (chunked verdict) < 0.85×128 GiB ✓; TWO concurrent
  runs ≈ 135 GiB > 0.70×128 GiB = 89.6 GiB ⇒ **curvelet arm CANNOT fire concurrent** — staged for
  immediate-next (launch.sh compiled, fires when the main run frees the waterfill).
- Launch entries (2026-07-15, all through the governed launcher; each REFUSE = information):
  - r1 REFUSE rc=8: `c1_component_wallclock_probe_every` equation_id None → #332 LawRef gate. FIX
    `22808dc7eb` (registered `laguerre_ot_head_offset_v1`; probe-every row dropped — trainer default,
    provenance in the lever notes).
  - r2 REFUSE (system admission): 121.2 > 117.8 GiB ceiling — 3 sibling bounded dry-starts held
    ~74 GiB active-growth. Not bypassed; waited.
  - r3 REFUSE rc=11 (ticket blockers): static `launch_blockers` refused even the dry-start —
    chicken-and-egg with the bench receipt. FIX `6d39a7dfc9`: blockers now COMPILE-TIME DERIVED
    (sR sidecar custody CLEAR from the real 450 MB file; bench blocker drops on a GREEN
    `full_config_dry_start` report for the config name) + the bounded `--dry-start` PROCEEDS under
    declared blockers (receipt producer; cannot durable-spawn — rc=11 real-launch invariant intact).
  - r4 REFUSE rc=3 (throughput gate): SegNet fwd+bwd 678 ms > 594 ms — the ~17× fast path IS active
    (OFF ≈ 6713 ms) but ~1.7× GPU-contended by the sibling trainers. Transient; not threshold-bumped.
  - r5: DURABLE CHAIN (`.omx/tmp/c1_dry_start/chain_507.sh`, driver log `chain_507_driver.log`) —
    drain-wait → bounded dry-start (writes `dry_start_report.json`, discharging
    C1_COMPOSED_BENCH_NOT_MEASURED on recompile) → on GREEN fires the REAL governed launch
    (`real_launch_507.log`; all gates re-run at spawn). Gate-chain state at arm time: schedule
    provenance 12/12 OK (5 events + backstops incl. `label_floor`), mem-preflight 24.48 GiB
    projected, admission ADMIT, dsl_compile_hash bound.

## 6. FOLD-IN QUEUE (owned elsewhere / next)

1. Batched LEVER-4 S_R consumer (unlocks micro-batch>1 under joint wall-clock) — named fallen-crack.
2. `--head-offset-precondition` trainer flag (#423 unlock).
3. Bregman #504 trainer consumer.
4. Adaptive-ε: bounded n24 stability A/B + sealed viscosity term (`adaptive_eps_cfl_edge_tracking_v1` ticket).
5. FreSh #448 / FINER++ #310 / StepNative: paired basin arms after the core lands its read.
6. T2 spike-reweight #274 A/B on top of a measured T1.
7. Trainer speed internals (adaptive_grad_clip magnitude-law A/B, telemetry_producers real_*
   fields): OWNED by the wallclock burn-down arm — coordinate via inbox, surfaces untouched here.
   (Their `autoclip_grad_clip`/`pose_verdict_gate` DIRECT_CONTROLLER_NAMES coverage gap was
   backfilled this landing — the static test was red at HEAD.) Lane-band cache CORRECTION (their
   measurement): the cache saves only −0.04 s/ep — the band cost is intrinsic θ-dependent
   margin/appearance forwards, NOT static-geometry recompute; the cache stays ON (bit-identical)
   but is NOT a wall-clock lever — struck from the speed ledger.
8. Timer-family #406 custody blocker FIXED this landing (`spec_throughput_component_timer_20260713`
   now derives hosc_beta_end from emitted argv = 3.177 with 10.0 preserved as
   `inherited_manifest_value_replaced` per #351) — unblocks the bounded short-epoch ticket for the
   magnitude-law A/B.

Pointer 0.19108 UNMOVED. Every row above is MEANS until `upstream/evaluate.py` returns a lower
byte-closed n600 exact row.

## Consumer-leg disposition (triality drift-detector, 2026-07-15 main)

The DSL surfaces added in the #507/#509 landings — `compile_c1_optimal_form*` +
`compile_c1_optimal_form_curvelet_arm` factories, the `label_floor` 5th EventBackstopGate,
`AdaptiveGradClip`/`GradNormalizeNone`/`VerdictParallelWorkers`/`LaneBandStaticCache` levers — are
consumed via the GENERIC introspection surfaces: lever_registry completeness, the schedule-provenance
readback (verified 12/12 OK through schedule_readback), launch_manifest/dsl_provenance rendering
(config-name-agnostic), and the governance rows (event=fires / cap=backstops) that the dashboard and
costate digest render schema-driven. No bespoke consumer code is required for these surfaces; any
future consumer that renders per-lever custom UI takes it from `describe()`. [consumers-generic]
