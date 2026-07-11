# #205 LIVE-RUN telemetry harvest for V9 · CGauge — full information-space read (2026-07-11)

**READ-ONLY harvest of the LIVE v7.5.2 baseline run** (`experiments/results/levelset_v752_baseline_20260710T185913Z`,
pid 88030, MLX-GPU, seed 0, git 6a34b66d6 dirty, upstream sha d46d8915…). This run IS the V9·CGauge trunk
lineage (`vehicle_v9_cgauge_naming_20260711.md`). **Pointer 0.19108282 [contest-CPU] UNMOVED** — every number
below is `[macOS-MLX / CPU-torch-verdict advisory] NON-PROMOTABLE`; this is interpretation, not a score.
Run untouched (no signals, no writes into the run dir).

**STORES CONSULTED:** graph_memory_recall ("lane per-class rising early / curriculum defers lane levers" →
FEED-dc/dg/lz/cx; "grad clip saturated" → FEED-be/collapsefix/ca/cb) · costate_digest + witness-checkin ·
`vehicle_v9_cgauge_naming_20260711.md` · `covariance_totality_texture_trunk_verdict_20260710.md` · CLAUDE.md
confound discipline · daemon.log (389 rows, 61 stage kinds) · costate_shadow.jsonl (27 rows) · launch.sh.

---

## 0. LIVENESS / APPARATUS-VALIDITY VERDICT (precondition — MEASURED)

**VALID MEASURING STATE. Harvest admissible.**
- 140/140 `loss_terms` rows: `spike_skipped=false`, `accepted_frac=1.0`, `weights_stepped=true`; ep_loss > 0
  every epoch (421.9 at ep125). Zero `confound_alarm` rows. No frozen-epoch signature; the median-freeze
  deadlock class is structurally absent (spike guard armed, never fired).
- Memory: RSS 20.3→6.2 GiB (drop ~ep60–75 = page-out/compression, not a leak; monotone-flat after),
  `mlx_active` 27.2 GiB, `mlx_peak` 29.0 GiB frozen since ep50, sys avail 65–72 GiB. No leak, huge headroom.
- Verdict authority path: async n600 CPU-torch verdict (`verdict_device=cpu`, `--verdict-pairs 0` = ALL 600),
  ~2190–2320 s each (~37 min, overlaps ~7 epochs of training). EMA-lag: monitor says CONSISTENT /
  `ema_lag_plausible=false` — best==latest at ep125, no EMA-shadow artifact in the headline series.
- fused-R kernel: `forward_bit_identical=true`, `grad_bit_identical=true` — the L70 bit-exactness precondition
  for MLX-GPU holds on this run.
- Cadence (MEASURED from verdict timestamps): **5.14 min/epoch** → Muon+pose-finish gate (ep726) ≈ **+2.1 days**
  from ep142; full 3000 epochs ≈ **10.2 days** wall.

One caveat on scope: only 6 verdict points exist (ep2/25/50/75/100/125); all slope claims below are
5-point-window reads (monitor n_window=5) — shapes, not converged laws.

## 1. WHAT THE RUN IS ACTUALLY RUNNING (config truth ≠ intent, MEASURED from emitted rows)

- **seg_form = `unify_tau` from ep25 on** (`--seg-form-unify-tau`): the discrete CE→tau→l7 curriculum is
  DISSOLVED into continuous L_τ. Consequence surfaced LOUD by the trainer itself:
  `event_curriculum_inert_under_unify` — **`--curriculum-event-triggered` is INERT** on this run (flag retained
  for resume stability only). Muon entry has its own EventBackstopGate.
- **τ ladder**: self-paced 12-octave geometric ladder 1.0→0.31, advance-on-`powerlaw_meat`, min_dwell 250,
  per-octave cap 500. **At ep142, τ = 1.0 still (octave 0)** — softmax_temp and hosc_beta both still at start
  values (β=1.0 despite `--hosc-beta-end 3.177`; anneal appears octave/stage-coupled — INFERRED, mechanism not
  read from source).
- **Budget arithmetic (DERIVED, material):** 12 advances × min_dwell 250 = 3000 = the ENTIRE epoch budget;
  with cap 500 the worst case is 6000. So the terminal τ=0.31 is reached — at best — at the final epoch with
  ZERO dwell, unless the tail controller (`--tail-tau-halving 0.5`, 2 cycles) finishes the descent. The τ
  schedule is structurally back-loaded against the run length.
- **Muon/pose-finish at ep726** (`muon_finisher_WARN` emitted: Muon engages before the l7-form partition —
  allowed by operator design, PR95-placement tension acknowledged in-log). Polyak finisher armed ep2546.
  LR cosine completes at ep1000 (`lr_anneal_epochs_WARN`), tail at 1e-4.
- **Pose is pose-BLIND by design until ep726**: `w_pose` term = 0.0 in every loss row; pose-finish is terminal
  (engage_mode `sigma_min_plateau`, backstop 726); `should_ship_banked_r1=true` throughout. Pose-carrier is
  the STORE-NOTHING warp (s_t=0.044, start d_pose 2.562 from its own s_t fit table).
- Active-now forces (nonzero in loss): seg (~3.5), island_amplify (~0.76), persistence (ramping 0.04→0.35,
  warmup 275), area_constraint (~0.06), weight_entropy (~0.81, λ=15), eikonal (0.028), length (3.5e-5).
- Armed-but-waiting (all 0.0 in loss, event- or epoch-gated): chroma_boundary (annulus_plateau / ep450),
  temporal_screw (annulus_plateau / ep450), lane_band weight fired ep31 (see §3), birth_completion (event),
  margin/horizon/thin-lane/etc. OFF per the sealed clean-baseline arm; HorizonWeightedMargin +
  StepNativeActivation explicitly DEFERRED to terminal-band A/B forks (launch.sh header).
- Event wiring WORKS: `start_event_fired: lane_band` at **ep31 via the lane_nucleus sensor** (cap 500 backstop
  not needed; sensor lag 6 epochs = one verdict cadence). First live proof of the calibrated sensor→transition
  wiring.

## 2. HEADLINE TRAJECTORY (n600 verdict series; advisory)

| ep | d_seg | d_pose | bytes | Road | Lane | Undrv | Mov | MyCar | flip-mass: Road/Lane/Undrv |
|----|-------|--------|-------|------|------|-------|-----|-------|------------------------------|
| 2  | 0.041123 | 6.50 | 91397 | .0916 | .8260 | .0045 | .9998 | .0016 | .52/.12/.05 (Mov .30) |
| 25 | 0.042281 | 17.6 | 85467 | .1474 | .2240 | .0125 | .0334 | .0005 | .81/.03/.15 |
| 50 | 0.033502 | 23.3 | 81558 | .1112 | .3618 | .0105 | .0187 | .0005 | .77/.06/.16 |
| 75 | 0.031089 | 29.6 | 81431 | .1008 | .3731 | .0105 | .0137 | .0004 | .75/.07/.17 |
| 100| 0.031740 | 30.1 | 82993 | .1008 | .3816 | .0119 | .0086 | .0004 | .74/.07/.19 |
| 125| 0.029330 | 33.3 | 82754 | .0925 | .3918 | .0108 | .0073 | .0004 | .73/.08/.18 |

Monitor classification: **CONVERGING / NO_STALL** (d_seg rel-slope −3.3e-3/ep; abs −1.1e-4/ep over ep25–125),
**NO label-floor** (0.0293 vs oracle floor 0.005318, band [0.00496,0.0070] — 5.5× above; the appearance-phase
question is not yet live for THIS run). ep100 blip (+0.00065) recovered by ep125 — oscillation, not a wall.
implied_S (21.2) is **d_pose-dominated transparency** — meaningless as a headline while pose is blind-by-design;
the live d_seg contribution is 100·0.0293 = 2.93.

## 3. THE NUANCED READS (each: signal → meaning → confidence)

**(a) Movable islands are BORN and consolidating — the ladder homotopy works on the live vehicle.**
Movable 0.9998→0.0073 (137× improvement), still falling every verdict; flip-mass share 30%→0.3%.
The islands-unborn failure of mod32cap (L2/L3) is CLOSED on this config: seed-islands + shield containment +
amplify + area-constraint birth did exactly what they were built to do. Seed shield healthy and STRENGTHENING
(mean_abs_seed_on_island 72.0→79.9, monotone). MyCar at 4e-4 (solved, static core). **Confidence: HIGH**
(monotone across 5 verdicts, mechanism matches design).

**(b) THE material warning — Lane is being ERODED while Road/interior are cleaned (the monitor's own
per-class alarm fires).** Lane within-class flip: 0.224@ep25 → 0.392@ep125, rel-slope +4.1e-3/ep; Lane
ABSOLUTE flips 154.7k→270.6k (+75%) while TOTAL flips fell 4.99M→3.46M. Mechanism (DERIVED from part_frac):
at ep25 the witness over-painted lane 4× GT area (part_frac 0.0235 vs GT 0.00585) — a fat blanket that covered
GT-lane cheaply but flipped Road (Road spiked to 0.147@ep25). Bulk pressure then SHRINKS the blanket
(part_frac→0.0134, still 2.3× GT), repairing Road (0.147→0.092) but eroding GT-lane coverage. This is the
L65 lane-erasure/homogenization dynamic in slow motion — the seed shield keeps islands ALIVE but does not stop
band THINNING. Crucially the curriculum KNOWS: every lane-specific repair force (chroma boundary, temporal
screw, dash comb weighting, birth-completion ramp) arms at annulus_plateau/ep450 — lane repair is deliberately
deferred. **The falsifiable in-run prediction: when the annulus_plateau event fires, Lane within-flip must
REVERSE. If it does not, the deferred-lane-stack hypothesis is implementation-falsified on this vehicle.**
**Confidence: HIGH on the trend (monitor-confirmed), MEDIUM on the mechanism attribution.**

**(c) Road is the CURRENT descent axis, not Lane.** 73% of flip mass (2.53M px, d_seg contribution 0.0215 of
0.0293) with within-flip 0.092 still descending. The converged-run picture (Lane = binding residual) does NOT
describe this stage; the early/mid-run binding term is Road bulk. Interior flips falling steadily
(0.0217→0.0132) while annulus flip-mass share RISES 0.50→0.57 — the residual is migrating onto the boundary
annulus exactly as the #333 endpoint (≈97% at convergence) predicts; we are watching the transport happen.
**Confidence: HIGH.**

**(d) d_pose drift 6.5→33.3 is DESIGN, not confound** (w_pose=0 until terminal pose-finish; banked R1 dxi
d_pose 0.001610/7.2KB is the shipped floor either way). BUT one genuinely new signal rides underneath: the
ξ→PoseNet **Jacobian σ_min is IMPROVING under seg-only training** (jacobian_basin median 0.078→0.119@ep138,
p10 0.0096→0.047, cond 95k→71k, basin_frac 1.0). Seg-descent alone is CONDITIONING the pose channel — the
photometric frames are becoming more pose-legible before any pose force exists. This is weak-positive evidence
for the covariance view (the trunk's ξ-structure is shared) and mildly de-risks the ep726 joint pose-finish.
The pose-gate DEGENERATE_GUARD (0/3 flat windows, slope noisy-positive) is correctly REFUSING to fire — the
guard is doing its job, not stuck. **Confidence: MEDIUM (32-pair sensor, provisional plateau).**

**(e) mod-dim spectrum — the live A/B input for the V9 mod-17-19 decision (#223/#299).** Latent-table
effective_rank climbs 8.68→**16.36**@ep125 (k90 23→25, spectral-entropy-norm 0.836→0.911, still rising);
per-dim zero-ablation (k=32 pairs) shows ~8–10 dims with Δd_seg ≤ 0 (removable-or-beneficial: dims
{0,2,3,5,6,10,11,15,20,21} at ep125; dims 2/5/6/10/20 consistently negative across ALL 5 probes — removing
them HELPS). Read: the trunk currently USES ~16 effective dims of the 32 and roughly two-thirds carry positive
utility — **consistent with mod-17-19, NOT with needing 32** — but eff_rank is still RISING at ep142; the
trained-out value (and especially its value after τ descends) is the number that decides #299. k90-truncation
free-rate estimate: 24.6KB vs 31.5KB full (D18 lever, ~7KB ≈ 0.0046 S-rate). ξ-CCA mean 0.55 / max 0.76:
the latent table is MODERATELY ξ-aligned — partial general-covariance in the learned code, room for the V9
explicit-(ξ,R) factoring to absorb more. **Confidence: MEDIUM-HIGH on shape, LOW on final values (early).**

**(f) Amber = SATURATED_ALWAYS_CLIPS is the RegIME, not a bug.** gnorm 3.5–13.6 vs grad_clip 0.5 → 100% of
steps clip-determined (with per-param normalize + per-group clip). This is the stability stack that extincted
the 1070.0802 dead-saturation collapse (FEED-be/collapsefix). Consequence worth stating for V9: **absolute
loss-term scales only steer DIRECTION; step MAGNITUDE is lr×clip** — so w_seg=100 vs weight_entropy λ=15 etc.
act as direction ratios, and lr is the sole magnitude knob (currently cosine→1e-4 by ep1000). Zero
gnorm_hijack alarms. **Confidence: HIGH (definition-level).**

**(g) Rate telemetry:** blob 91.4KB→82.8KB (weight_entropy term ~0.81 flat-active); rate S-term ≈ 0.055.
Byte trend mildly down, noise ±1.5KB across verdicts. Nothing binding here yet.

## 4. BINDING vs INERT on THIS run (the #404 lens, as-measured)

| Lever/force | State | Evidence |
|---|---|---|
| seg (L_τ unified) | **BINDING** (dominant) | ~3.5 of ~5.5 total; d_seg descending |
| island_amplify + seeds + area_constraint | **BINDING** | Movable birth (a); seed magnitude rising |
| persistence (cldice) | **BINDING, ramping** | 0.04→0.35 under 275-ep warmup |
| weight_entropy λ=15 | **BINDING (direction-share ~15%)** | bytes 91→83KB |
| eikonal 0.01 / length 0.001 | active, small (0.028 / 3.5e-5) | regularizer_magnitudes row confirms C3-normalized scale |
| grad-clip 0.5 (amber) | **SATURATED_ALWAYS_CLIPS** — defines the update regime | §3f |
| lane_band (fired ep31) | **ENGAGED via event** — efficacy not yet separable from bulk | first event-wiring live proof |
| dseg_aware_taper #121 | active, ±4.6% spectral reallocation | one-shot row; RE-VALIDATE at convergence (its own note) |
| chroma_boundary | **PENDING** (annulus_plateau/ep450) | term 0.0 all rows |
| temporal_screw #360 | **PENDING** (annulus_plateau/ep450) | term 0.0 all rows |
| curriculum-event-triggered | **INERT by construction** under unify_tau | trainer's own LOUD row |
| pose (all) | **BLIND by design** until ep726 | §3d |
| Muon / polyak / tail | armed ep726 / ep2546 / tail controller | not yet fired |
| appearance-phase endgame (T1 #424 / carrier #425 / Law-5) | **NOT IN THIS RUN** (default-OFF; SEAL+A/B owed) | launch.sh has no phase flags |

## 5. V9 · CGauge CASH-OUT (ranked by EV)

1. **Mod/hidden sizing (#223/#299) — the trajectory SUPPORTS mod-17-19 but hasn't proven it.** eff_rank 16.4
   rising, ~10 dims ablation-neutral-or-harmful, 5 dims consistently negative. OWED: re-read
   `mod_dim_dynamics` at the τ-descended terminal band (the spectrum under τ→0.31 is the decision number);
   the D18 k90-truncate byte-close (~7KB free rate) is the cheapest real S-mover visible in this telemetry.
2. **Lane-erosion watch → the deferred-lane-stack test (§3b).** For V9: either (i) keep the deferral but make
   the annulus_plateau reversal an EXPLICIT gate (auto-alarm if Lane within-flip has not turned within N epochs
   of chroma/screw engage), or (ii) arm a cheap lane-preserving force EARLY (e.g. birth-completion ramp on
   class 1, or the lane-band weight >1) — a run-2 A/B fork, not a live change. The per-class alarm already
   fires; V9 should treat per-class BESTs as first-class (global-best checkpoint selection let Lane's best
   [email protected] go stale 4 verdicts).
3. **Curriculum budget arithmetic (§1): the τ ladder + 3000 epochs are structurally tight.** V9 config should
   either budget epochs ≥ 12×min_dwell + terminal dwell, shorten the ladder, or explicitly rely on the tail
   controller's τ-halving — today it is implicit. Also resolve the muon@726 < l7@800 placement tension the
   trainer itself WARNs about (under unify_tau the l7 stage is dissolved, so the WARN may be vacuous — verify,
   don't assume).
4. **Pose-finish de-risk (§3d): σ_min improving pre-engagement** supports keeping the v7.5.2
   terminal-joint-pose-finish design in V9 unchanged; the basin TRIGGER (would_have_fired@ep138, provisional)
   is a run-2 lever with its first live calibration data now on disk.
5. **Covariance headroom (§3e): ξ-CCA 0.55/0.76** — the learned code is only partially ξ-factored; V9's
   explicit (ξ,R) covariant parametrization has measured room to absorb structure the mod-table is currently
   spending dims on.
6. **Appearance-phase arm stays post-floor:** this run is 5.5× above the 0.005318 oracle floor and the
   label_floor detector correctly says NO_FLOOR — the endgame arm's moment is not yet; no reason from this
   telemetry to change the SEAL+A/B-owed sequencing.
7. **Ops for V9 launches:** 37-min async n600 verdict per 25 epochs is well-amortized (async overlap works;
   zero stalls); 29GiB MLX peak at mod-32/hidden-96/n600 → V9 at same trunk size has ~2× memory headroom on
   this host.

## OWED-TO-MEASURE (from this harvest)
- [ ] Terminal-band `mod_dim_dynamics` re-read (decides #299 mod-17-19; watch eff_rank under τ descent).
- [ ] Lane-reversal check at annulus_plateau engage (falsifies-or-confirms the deferred-lane-stack; §3b).
- [ ] D18 k90-truncate byte-close row (24.6KB vs 31.5KB estimate is advisory until byte-closed).
- [ ] dseg_aware_taper #121 re-validation at convergence (its own emitted note).
- [ ] muon@726-under-unify_tau semantics verify (is the l7 WARN vacuous under the dissolved curriculum?).
- [ ] Jacobian-basin trigger calibration finalize offline vs actual terminal (run-2 lever).

**Triality:** DAG = FEED-v9-harvest-1 (appended this pass) · DSL = N/A-with-rationale (read-only harvest; no
lever built or changed) · equations = N/A-with-rationale (no new law — the lane-erosion read instantiates the
already-registered dash_erasure_homogenization_v1 family; the σ_min-under-seg-only observation is a 32-pair
provisional, below registration bar). Pointer 0.19108282 UNMOVED.
