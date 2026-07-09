# T5 CRUCIBLE-2 — DELTA GROUNDING PACKET (2026-07-09)

**Everything decision-relevant landed SINCE SPEC_v75 (2026-07-08).** Six P1 seats work from THIS pack —
do NOT re-mine the stores; if a claim you need is missing, it is a grounding gap, flag it. Every number
labelled MEASURED / DERIVED / ESTIMATED / ASSUMED with its source. Pointer **0.19110 UNMOVED** —
everything here is [macOS-MLX/CPU advisory] research-signal, NON-PROMOTABLE, MEANS. Only a byte-closed
`upstream/evaluate.py` n600 row moves it. #205 is STOPPED (box free); no live run to protect, but a
v7.5.2 execute-at-n600 smoke is itself governed (operator-GO).

STORES CONSULTED: SPEC_v75 + SPEC_v8 · DAG FEEDs 07-08/07-09 (all named in CONVENING §STORES) ·
activation-ledger duty-to-measure (via FEED-relsigfold seeded top) · memory L68/L79/watch-items.

---

## A. THE ONE-LINE STATE CHANGE SINCE SPEC_v75

SPEC_v75 §1 gate ("v7.5 CANNOT reach sub-0.19; pose ~1.79 → 4.24 of S") is **SUPERSEDED.** Pose is
BANKED at n600 authority (row P-1). **The frontier gap to sub-0.19 is now ENTIRELY d_seg.** run-1 is
STOPPED (row R-1). v7.5's d_seg lever set went from "designed" to "BUILT + fireable + triality-complete
+ default-OFF" (rows L-1..L-8). The junction is now COMPOSITION + collapse-fix + sequencing, not "can it
work."

---

## B. FIREABLE LEVERS — BUILT, triality-complete, DEFAULT-OFF (the composition menu)

All are DSL `Lever` factories, byte-identical when off, held in the activation ledger's duty-to-measure
queue. Ranked by **relative significance = ΔS / (0.19110 − 0.15) = fraction of remaining descent to
sub-0.15** (FEED-relsigfold; the near-goal correction — absolute ΔS orphans them, relative ranks them
high). d_seg is the ENTIRE remaining fight (rate DEAD-at-floor-for-this-vehicle, pose BANKED).

| id | lever | rel-sig (owed queue) | measured basis | verdict_scope | source | implication for v7.5.2 |
|---|---|---|---|---|---|---|
| **L-1** | **#121 d_seg-aware Fourier-feature taper** | **73%** (RANK 1) | +18% was ONE under-converged run; converged anchors FLIP to −8% (ESTIMATED ΔS ~0.03 ≈ 70% remaining) | INSTANCE (the +18% NO-GO was one under-converged run, NOT the formulation) | FEED-dsegtaper 07-09; FEED-relsig | TOP candidate to compose ON — but the "converged A/B" flip must be re-validated; 0-byte, rule-118 FREE, byte-neutral reweight of the curvelet basis toward the d_seg-critical margin band |
| **L-2** | **#169 horizon-weighted margin** (0-byte hinge) | **43.8%** (RANK 2) | oracle ceiling ΔS≈0.024 @ margin≥0.3 / 0.012 @ ≥0.5; 97.8% frontier d_seg in horizon SEG rows 96-288 | FORMULATION (label-noise risk: `<lo` margin = IRREDUCIBLE frozen-SegNet noise, EXCLUDED by construction) | FEED-v75B5built + FEED-horizonmargin 07-09; `dseg_reducibility_gt_margin_verdict_20260623` | one-sided satisficing hinge on the SHARED `_signed` #141 margin; A/B EXIT = surviving flips must shift to HIGHER GT margin, else chasing label-noise → terminal-finding |
| **L-3** | **StepNativeActivation** | **31.6%** (RANK 3) | −4.5% n600 (ΔS ~0.013 ≈ 32%) | INSTANCE→adopt LIVE screen verdict | FEED-relsig/relsigfold | step-native / FINER++ activation; "modest" ≠ orphan near goal |
| **L-4** | **#360 TemporalScrewConsistency** (P0 FORCE 1) | ~50× Undriv-jitter (relative not yet in queue seed) | Undriv-sky 0.082 → target ~0.0016; kills 44% lane-dominated flicker residual | EXIT owed n600 A/B | FEED-v75B4actuated 07-09; `p0_forces_derivation §FORCE 1` | **WIRED into crucible_v7 as 9th lever, EVENT-governed** (fires on annulus_plateau FORMED-boundary sensor); GROUND-class {0,1,2} annulus prob-warp under H(ξ), **ξ=`ground_gt` stop-grad ⇒ PURE seg regularizer, ZERO pose coupling** (L68); cold-start w_t 0.1; + **sky=rotation-only** stratification (B.5): sky at d→∞ warps rotation-only H_rot=K·R·K⁻¹ |
| **L-5** | **#276 LEVER-4c SegChromaBoundary** | UNMEASURED (duty-to-ESTIMATE) | REMOVAL ablation: constant-luma FLIPS 7.54% Lane→Road + 4.38% Movable→Undriv; **93.4% of chroma-flips in margin<1 annulus**; SegNet margin energy 78.8% luma / 21.2% chroma | ADD-BACK ΔS UNMEASURED (ablation ≠ add-back score) | FEED-chromalever 07-09; `chroma_decides_lane_and_movable_at_annulus_v1` | chroma := rgb − BT.601-luma ⇒ LUMA-INVARIANT ⇒ ORTHOGONAL to every luma lever; per-pixel chroma MATCH at annulus (NOT full-RGB recon); the witness under-exploits chroma (converges to near-constant palette) |
| **L-6** | **#220 AACoverageRender + grid≥384** | UNMEASURED (duty-to-ESTIMATE) | oracle-R floor d_seg **0.00091 @384** vs 0.00247 @192; lane-recall +0.38 | through-training ΔS UNMEASURED | FEED-aacoverage220 07-09; `aa_supersample_lane_recall_lift_v1` | coverage-integrated render (supersample AUTHORITY / ipe PROXY); grid≥384 now ENFORCED; **supersample is fail-closed against --self-orient / --dseg-aware-taper** (interaction constraint — the ipe chart is the self-orient-compatible one) |
| **L-7** | **#378 WitnessStability / `--stability-preset amber`** (collapse-fix) | P0-NOW (joint-descent ENABLER) | run-1/v6 collapse DIAGNOSED-BUG (not a wall), 4 causes: (1) no grad-clip; (2) sqrt-pose-eps grad blowup `5/√(10·pose+1e-8)`→5e4 for easy pairs (per-pair batch=1 exposes it); (3) w_seg=100 ⇒ 100× seg-LR; (4) lr 2e-3 × 600 steps/ep | FORMULATION (diagnosed bug, orthogonal to any architecture) | FEED-amber-unblock 07-09; stepping_instability_diagnostic_20260705 | FIX = opt-in stability levers, DEFAULT-PRESERVING (#205 byte-identical): grad-clip + **pose-eps-floor `eps_floor(C)=(5/C)²`** (per-batch-size floor killing the sqrt-pose grad blowup) + **per-param grad-normalize** + stage-boundary LR/w_seg guard; verified vs the existing immune system (gnorm_hijack alarm, spike-guard). **This is what lets the JOINT 7-dim descent CONVERGE** (crux-7dim, row X-1) |
| **L-8** | **#341 terminal head GN/CG solve** | TOP by impact×readiness (solve-don't-train) | head chart near-quadratic CONFIRMED (LM ρ **0.847/0.868** MEASURED); ~791-param AFFINE head (out_sdf.{w,b}+out_tex.{w,b}+palette; FiLM EXCLUDED non-affine) | **FULL-P (P=600) ONLY** — K=8 subset OVERFITS +5.1% n600 (FORMULATION NO-GO, MEASURED) | FEED-#342 solve-don't-train 07-09; `quadratic_head_chart_subset_solve_gap_v1` | replaces the terminal fine-tune leg; ~3 h GPU (NOT $0); fire at TERMINAL tau-best EMA ckpt IFF LM ρ∈~[0.8,1.2] RE-VERIFIED on the current ckpt, all 600 pairs, exact tau-stage loss, --fused-r-kernel bit-identity, verdict through R + frozen CPU SegNet |

**§9 SCOPE (SPEC_v75, binding):** the P0 forces default OFF, activate **ONE per crucible increment** with
measured A/B (≤15% loss-share each / ≤40% total; satisficing sequenced ≥ l7; turning all 3 on = SPEC
VIOLATION, confounds attribution + risks term_domination). This is the tension at the heart of OPEN Q1.

**MEASURED NEGATIVES (do not re-open blind; flip-weighted reformulation where noted):**
- **N-1 #288 OT head offsets — MEASURED NEGATIVE** (FEED-otoffset 07-09): mod32cap ep650, realized-through-R,
  n48+n24 both reproduce order+sign — no_offset **0.00272** < menon 0.00293 < **ot_newton 0.00487 (WORSE)**.
  OT enlarges the rare-Lane cell to hit its 0.59% GT mass (b_Lane≈+28.7) → OVER-predicts Lane → SegNet
  penalises. Both offset arms HURT. verdict_scope: **FORMULATION** (mass-matching to raw GT frequencies as a
  d_seg surrogate) — NEXT reformulation = **flip-weighted target masses** (match argmax to where flips are,
  not raw area). Solver is EXACT (7 Newton iters, mass_err 0.0). Larger-n OWED (probe not resumable-chunked).
- **N-2 lane-ξ ego-transport — MEASURED NO-GO** (FEED-v8-lane-xi 07-09): ξ-advection ENLARGES the Road↔Lane
  stream every arm (best 42,017 B vs identity 41,085 B). WHY: the lane coder stores GROUND-frame coeffs (IPM
  already quotients ego-forward at FIT time) → residual is IRREDUCIBLE curvature evolution, not rigid
  transport. **Corollary (durable law): ego-freeze does NOT transfer to any chart that has already absorbed
  the ego DOF.** Horizon won 14.6× only because its poly is image-frame (removable ego-pitch intercept).
  verdict_scope: FORMULATION (lane rate axis); ξ stays decisive for POSE + image-frame horizon.
- **N-3 #341 subset solve** — K=8 subset overfits +5.1% (row L-8); full-P in-trainer only.

---

## C. POSE — BANKED (superseded the SPEC_v75 §1 launch gate)

| id | finding | MEASURED | source |
|---|---|---|---|
| **P-1** | **R1 dxi BANKED at n600 AUTHORITY** | d_pose **0.001610** → contribution √(10·d_pose) = **0.127**; ξ_eff **7,195 B** (rate 0.004791); **20× over no-dxi** (0.0220→0.469); ~16,000× below naive-calib (26.04). Full advisory S(n600) = 100·0.004549 (**seg 0.455 = THE blocker**) + 0.127 (pose BANKED) + 0.060 (rate) = 0.642 | FEED-238resolved 07-08 (#238 RESOLVED); L68 |
| **P-2** | **pose ⊥ d_seg EXACTLY (structural proof)** — ∂d_seg/∂ξ ≡ 0 (SegNet reads ONLY last frame, modules.py:108; ξ shapes ONLY seg-free frame0) → the ~99.95% seg⊥pose null is a structural exact-zero + measured ~0.05% θ-residual ⇒ pose carrier CANNOT disturb d_seg | FEED-posejac 07-09 |
| **P-3** | **pose = orthogonal/monotone/benign; d_seg = separatrix-bound/stage-sensitive/HARD** — R1: d_pose 97→0.0011 (108ep monotone) while d_seg HELD ~0.0046. STRATEGIC: spend ZERO curriculum complexity on pose (w_pose on + enough epochs, or R1 two-phase); spend ALL curriculum/stage/lever complexity on d_seg | FEED-posesegdynamics 07-08; `pose_seg_convergence_asymmetry_v1` (council-flagged) |
| **P-4** | **D.9 terminal pose-finish stage WIRED** into crucible_v7 (`--pose-finish-start-epoch`, gates effective w_pose to 0 until d_seg converges via `_muon_gate.fired` OR backstop 726, then --w-pose engages for terminal joint pose-descent → #238-serialize dxi at export). Default 0 = pose-BLIND = byte-identical incumbent | FEED-v75D9actuated 07-09 |
| **P-5** | **HONEST POSE FLAG (corrects over-claim):** pose is BANKED-AS-ARTIFACT but **NOT solved-for-v7.5** — whether v7.5's NEW terminal pose-finish (D.9) actually CONVERGES to an R1-class dxi is **UNVALIDATED** (mechanism correct + byte-identical-when-off; efficacy OWED). Memory refutes cheap post-hoc/stored carriers; only JOINT descent from a coherent render crosses the photometric wall | FEED-v75seal 07-09; L68 |
| **P-6** | **σ_min basin sensor** — J_ξ=∂(PoseNet∘R)/∂ξ ∈ ℝ^{6×6}; basin var σ_min; DERIVED coherence↔conditioning (converging d_seg → richer boundary normals → σ_min↑, the Jacobian reason behind the flat 1.2–1.8 floor vs R1-from-converged 0.0011). Telemetry BUILT + observer-only on v7.5 (byte-identical CPU-verified). **σ_min-basin TRIGGER (earlier-engage arm) = a run-2 DEFAULT-OFF lever** — NECESSARY-not-SUFFICIENT (σ_min is observability; content DOF comes from θ) | FEED-posejac/posejacbuild 07-09; `pose_jacobian_basin_conditioning_v1` |
| **P-7** | **#314 pose-carrier source = OPERATOR-ROUTABLE.** crucible_v7 lineage runs the `generated` store-nothing path (structurally pinned → ~1 KB rate). "Restore store-nothing at the next FRESH arm" stays an operator decision. **Rate basis is lineage-tagged:** {fresh_seeded v1→v5: real_keyframe = COUNTED 697,941 B} vs {store_nothing / crucible_v6/v7 / fresh_seeded v6+: generated = ~1 KB}. Any byte-close from the v1→v5 lineage MUST charge the counted-keyframe rate | FEED-drift-d2-fix 07-09 |

---

## D. SCHEDULE / CURRICULUM / CONTROL DELTAS (event-vs-epoch state)

| id | finding | status | source |
|---|---|---|---|
| **S-1** | **v7.5 is ALREADY ~80% event-driven.** Event-fired: CE→tau (#315 plateau-slope), Muon nucleation, birth-completion (Morse-Smale persistence), temporal-screw (annulus_plateau FORMED-boundary), pose-finish (co-fires with muon `_muon_gate.fired`). Remaining EPOCH backstops/fail-safes: min-stage-epochs 250, Polyak finisher start 2546, hosc β frozen 3.177, muon cap 726 | FEED-auditB 07-08; FEED-v75B4actuated | 
| **S-2** | **#270 Muon warm-start + LR-anneal** — the Muon-boundary treatment: warm-start-momentum + LR re-warmup at the Muon boundary (treat as stage boundary) + lr-final-frac 0.1. Muon = −32% d_seg vs AdamW (KEEP Muon, tune the finishing schedule) | SPEC_v75 §2; MEMORY L78/L79 |
| **S-3** | **C.6 handoff_readiness telemetry legibility = default-ON** (score-neutral): stamped `stage_start` / `in_stage_epochs` / `min_stage_epochs` + `plateau_slope`. Fixes the confound that a CE→tau transition can fire between sparse verdict rows so plateau_ok read False on the coarse cadence | FEED-v75C67actuated 07-09 |
| **S-4** | **C.7 `curriculum_min_stage_epochs` 250 = HARDCODED-WITH-WAIVER** on the value-provenance ladder (hand-set safety margin, NOT measured/derived). **OWED derivation NAMED: the CRITICAL-SLOWING relaxation time τ_relax** near a curriculum stage transition (min_stage ≥ τ_relax so a post-transition transient flat isn't misread as convergence) — fit from ep_loss after a fired transition → becomes DERIVED-AT-CONFIG. Did NOT fake a derivation | FEED-v75C67actuated 07-09 |
| **S-5** | **NCDE trajectory detector** (#344) — Linear neural-controlled-DE `dh/dt=Ah+Bu+c` (u = softmax_temp, hosc_beta), closed-form ridge fit; `detect_hit_solve` fires (advisory) on log d_seg when BASIN (remaining within-stage descent < 5% → #341 terminal solve admissible) OR HANDOFF-ETA (#315 plateau within N ep). Shadow-only sense organ, costate-surfaced. Backtest: #205 log_total MAPE 2.6%; mod32cap log_d_seg MAPE 0.77%. FORMALIZATION_PENDING (N≥5 runs owed) | FEED-ncde / ncde-wirein 07-09 |
| **S-6** | **#313 micro-batch OFF** (unchanged from SPEC_v75): scorer forward is batch-DEPENDENT (GPU 2.26e-2 drift / 11 argmax flips) ⇒ bit-identity-at-speedup IMPOSSIBLE; bounded n600 d_seg A/B is the ONLY admission path | SPEC_v75 §2; `frozen_scorer_forward_batch_dependence_v1` |
| **S-7** | **#357 speed bundle / #356 megakernel** — stop-window armed, governed-stop gated (not a d_seg lever; wall-clock only, lexicographic-secondary, never traded vs score) | SPEC_v75 §4 item 8 |

---

## E. RUN-1 (the birth-arm) — STOPPED; what it MEASURED = what v7.5.2 CURES

| id | finding | MEASURED | source |
|---|---|---|---|
| **R-1** | **#205 birth-arm STOPPED clean** (operator-GO "free the box"). Best `levelset_witness_ema_BEST.npz` = **d_seg 0.115102 @ ep325** (improving: 0.1198@ep275→0.1151); **Road 0.312 + Undriv 0.083 DOMINATE** — exactly what v7.5's actuation targets. All 5 islands born, curriculum event-fired CE→tau@257. Resume state + per-stage `resume_stageTau_ep257` preserved. Birth-arm = healthy DIAGNOSTIC baseline, **NOT the pointer-mover** (pre-actuation config lacks the v7.5 fixes) | FEED-205stop 07-09 |
| **R-2** | earlier-run LIVE reads (pre-stop, ep~100-166, EMA-lag caveat): stage-1 unify_tau; ep100 verdict d_seg ~0.034 (drift UP from ~0.032@ep50 = **EMA-shadow lag** read — check EMA lag BEFORE diagnosing pose/plateau); Lane ~0.38 WATCH; Road ~0.108 / 73% flips; MyCar SOLVED; islands born+holding (lane part_frac ~0.014, movable ~0.022); pose-blind by design (co-train ~1.79); basin 100% | run-1 telemetry; SPEC_v75 §5; L-memory EMA-lag |
| **R-3** | **the Road-floor MECHANISM run-1 revealed** (the birth-counter-force reason): birth stack recall-WITHOUT-precision over-paints Lane 13.8× GT / Movable 4.6× INTO GT-Road, mass-conserved with the Road+Undriv deficit (0.1191≈0.1189) ⇒ Road d_seg FLOORED ~0.40. **NOT the analytic band** (falsified 3 ways). Chan-Vese area-constraint (λ_lane 683.8 / λ_movable 322.6 DERIVED-LIVE, equilibrium 1.25×GT returns ~96% of the deficit) is the CURE + the Morse-Smale birth-completion event + per-class ramp — all composed ON in crucible_v7 | FEED-roadfloor/roadfloorfix 07-08; `chan_vese_area_constraint_birth_balance_v1` |
| **R-4** | **⭐ SENSE-GAP / FALSE-GREEN (operator catch 2026-07-09) — a FIRST-CLASS design input for the schedule/control seat.** run-1's VERDICT d_seg ROSE **0.0324@ep50 → 0.0340@ep100** (Lane **0.349→0.381**) while TRAIN LOSS DESCENDED, and the **shadow controller classified CONVERGING** — the FALSE-GREEN mode its OWN source warns about (`shadow_controller.py:~314`). Operator's read: **"more CE isn't going to help" = the same-gradient-won't-cure-it signature** (train↑verdict decoupling ⇒ the current loss is optimizing the wrong thing for the residual). A **verdict-trend / train-verdict-decoupling alarm** is being built IN PARALLEL (defaults ON, per the score-neutral-observability-defaults-ON non-negotiable). **v7.5.2's control/schedule seat (S3) MUST treat train-verdict decoupling as a first-class design input** — a stage that descends train loss while the verdict rises is a CE-exhausted stage (hand off / re-treat / change the term, NOT more of the same gradient). Caveat: partly the EMA-shadow-lag early-verdict artifact (R-2) — the alarm must distinguish decoupling from EMA lag | operator addendum 2026-07-09; `shadow_controller.py`; L-memory EMA-lag |

---

## F. SEAL + STRUCTURAL STATE (what crucible-2 inherits as "launch-eligible")

- **F-1 FEED-v75seal 07-09: STRUCTURAL-SEAL-PROCEED** (1 CRITICAL round → 3 clean structural passes). The
  CRITICAL was **launch-blocking**: B.4's `seg_temporal_screw` EventBackstopGate key prefix was OMITTED from
  `GATE_KEY_PREFIXES` (fail-closes on unknown keys) ⇒ EVERY launch would crash at `run_train` startup; the
  build's "221 tests green" tested CONFIG COMPILATION, never the LAUNCH PATH. **LESSON = the #205-OOM
  "surrogate ≠ authority" class at the launcher surface: passing config tests ≠ passing startup.** FIXED
  additively (fbe39e0da). Verified: 397 tests green, 0 WitnessProgram violations, 0-NAKED schedule-provenance,
  orphans UNCHANGED 120, all 6 new levers DSL factories in the activation ledger, antagonists OFF, D.9 real.
- **F-2 OWED behind the seal (execute-at-n600, all owed a governed run):** (1) execute-at-n600 runnability +
  real peak-RSS (the #205-OOM lesson: a B=8 throughput bench is a SURROGATE, not the authority — must
  execute the REAL n600 config); (2) scored quantities through byte-close (per-class d_seg · **d_pose =
  decisive** · rate); (3) B.5 horizon/sky label-noise-vs-real A/B; (4) A.3 directional −48% A/B (self-orient
  production realization vs the circular-probe measurement).
- **F-3 crucible_v7 as-actuated composition (the incumbent v7.5 config crucible-2 evolves):** counter-force ON
  (Chan-Vese area constraint + birth-completion event + per-class ramp; dsl_levers 5→7); A.1 lane paint-then-
  SDF (`--lane-prior-phi1-mode paint`, drop dead `--structured-init-include-lane`; lane FN 0.00713→0.00211
  ~3×); A.2 dash-comb ON (operator override; `--lane-band-dash-comb`, 8th lever); B.4 temporal-screw wired
  EVENT-governed (9th lever); D.9 pose-finish TypedStage; L-1..L-3 + L-5..L-6 registered-OFF in the duty-to-
  measure queue. The 3 P0 forces default-OFF (one-per-increment). Directional −48% (A.3) is the run's OWN A/B.

---

## M. MODAL / EXACT-EVAL ENVELOPE (operator addendum 2026-07-09)

- **M-1 MODAL ENVELOPE AUTHORIZED: ≤$20 HARD CAP** (operator 2026-07-09) — earmarked for **exact-eval rows**
  (paired contest-CPU + CUDA on byte-closed candidates, the ONLY promotion authority) + the **owed CPU-torch
  n600 measurement queue** (F-2 scored-quantities-through-byte-close). This is the "spend the budget to BUY
  exact rows" mandate at a concrete cap.
- **M-2 witness training itself is MLX** and does NOT run on Modal — the v7.5.2 run trains locally
  (M5 Max, MLX-GPU). Modal is for the byte-closed exact-eval + the CPU-torch verdict authority, not for
  training. ⇒ **OPEN QUESTION (added to the seat list): is a torch-parity twin of the witness worth building
  for Modal A/B fan-out?** (parallel exact-eval of ON/OFF lever arms on paid contest hardware vs the serial
  local byte-close + governed CPU verdict). S4 (rate/byte-close) owns this — weigh the parity-twin build cost
  vs the fan-out throughput within the $20 cap.

## X. THE UNIFYING FRAME (crux + top-AIML re-open — context for the derivation)

- **X-1 crux-7dim (FEED-crux-7dim 07-09):** the boundary-band-flip crux = a JOINT optimal over 7 built
  actuator dims — **scale** (curvelet multi-scale #212), **res** (#149 camera-res sub-pixel placement + LPPN
  arbitrary-res decode), **time** (ξ/se(3) #193/#194 + keyframe-warp #148 + horizon-ξ), **direction**
  (all-class directional basis = the #1 lever −48% + along-tangent #277), **chroma** (#276 + chroma_boundary
  L-5), **luma** (luma carrier + coupling), **place** (margin=Fisher surrogate Pearson 0.978 + annulus #333
  ~97% of d_seg in 4.7% band + d_seg_aware_taper L-1). **The costate controller (#247) is the JOINT optimizer
  OVER the per-dim levers; the collapse-fix (L-7) is precisely what lets the JOINT descent CONVERGE.** The
  AMBER's win was joint through-R descent — the fix unblocks it.
- **X-2 cells2pixels / AMBER top-AIML RE-OPEN (FEED-cells2pixels/amber-unblock 07-09):** NCA+LPPN
  (SIGGRAPH'26) = arbitrary-resolution decode = the #149 camera-res placement machinery (place the flip at
  874×1164 BEFORE the downsample D averages it away = the low-res fragility fix). Our own prior: #146
  continuous-texture NCA = AMBER, the STRONGEST d_seg-core the campaign produced (realized d_seg **0.00337**
  = 1.31× frontier, **boundary_band_flip 0.079 = HALF the polynomial wall**, rate 0.019, S 0.415). Shelved
  only for the (now-diagnosed) training-collapse BUG. **STAGED, not drop-everything** (reactivation: fire
  WHEN the SDF witness walls on realized boundary_band_flip AND the collapse-fix is in). rule-118: NCA/LPPN
  weights = COUNTED; iteration = FREE.

---

## V8. RATE LEDGER (context ONLY — crucible-3's food; d_seg is THE blocker for BOTH vehicles)

- **V8-1 whole-scene rate roll-up (FEED-v8-rollup 07-09, REAL coder bit-exact):** bitmap TOTAL 0.339 ·
  geometric **DOMINANT-only 0.061** (Road/Lane .0275 + horizon .0032 + Movable .00344 + hood .0202 +
  Lane/* .007) · geometric COMPLETE (+lossless residual) **0.140**. vs frontier rate 0.118: dominant-only
  **1.9× BELOW** (v8 rate thesis CONFIRMED on dominant structure); COMPLETE **1.2× ABOVE** (the residual
  sidecar is the whole 0.079 gap). Movable = ONE sparse-site carrier for both edges, 5.04 KB@n600 = S 0.00344
  (vs bitmap 0.0532 = 15.4×). Headroom: de-sharing double-count + curve-relative residual coder (both
  MEASURED-un-exploited, move 0.140→0.061).
- **V8-2 Road↔Lane 0.0275 HELD** (ξ-transport a NO-GO per N-2). Increment-1 draft = revised strong-seed,
  build-ready MODULO P-C (memory-gated post-#205). **d_seg is the true blocker for v8 too.**

---

## OPEN QUESTIONS — the 5 sharpest (each P1 seat takes an INDEPENDENT position)

1. **LEVER COMPOSITION (the central question).** Which fireable default-OFF d_seg levers compose ON in
   the FIRST v7.5.2 launch vs stay one-per-increment A/B? The top-3 by relative significance are
   {#121 taper 73%, #169 horizon-margin 43.8%, StepNative 31.6%}; the P0 forces have a §9 SPEC rule of
   **ONE-per-increment** (≤15%/≤40% loss-share, attribution requires isolation). Tension: **all-top-
   measured-on in synergy order (basis-match BEFORE capacity)** vs **minimal-composed + A/B-drain the
   queue** vs a middle path. Which levers are SYNERGISTIC (compose) vs CONFOUNDING (isolate)? Note the
   MEASURED interaction constraint: #220 supersample is fail-closed against #121 taper / self-orient.
2. **WARM-START vs FRESH + RUN-1 DISPOSITION (explicit crucible question, operator addendum 2026-07-09).**
   v7.5.2 warm-start from a run-1 basin — but run-1 is PRE-actuation (no Chan-Vese counter-force), so its
   Road-floor is the EXACT thing v7.5.2 cures ⇒ warm-starting inherits the floored basin. **OPERATOR
   RECOMMENDATION ON THE TABLE: RE-RESUME run-1 to the stage-1 boundary (~ep250, ≈10 h) then decide
   warm-start-v7.5.2-from-the-stage-1-ckpt vs fresh, USING THE STAGE-1 ISLAND-BIRTH VERDICT as the
   evidence.** (Timing note for the seats to reconcile: FEED-205stop reports run-1 STOPPED at ep325 with a
   preserved `resume_stageTau_ep257` CE→tau-boundary ckpt — the stage-1 boundary at ep257 has ALREADY been
   crossed; the operator recommendation is a disposition framing — re-resume-to-a-clean-stage-1-decision-
   point vs use-the-existing-ep257/ep325-ckpts vs fresh. Reconcile with FEED-205stop.) Alternatives: fresh-
   from-ep0-with-counter-force · warm-start-from-a-DIFFERENT converged basin (mod32cap 0.003366 /
   v2_attrclean 0.004024, per FEED-auditC "v7.5 d_seg base = BEST measured"). Couples to P-7 (#314 store-
   nothing restore at a FRESH arm = operator-routable) + #270 warm-start-momentum. The R-4 false-green/
   decoupling read bears directly: the stage-1 island-birth verdict is the RIGHT evidence precisely because
   the train-loss trend is NOT (it decoupled).
3. **COLLAPSE-FIX ACTIVATION (#378).** Is `--stability-preset amber` composed ON by default in v7.5.2?
   It is default-preserving/byte-identical when off, but the JOINT 7-dim descent it enables (X-1) is the
   convergence ENABLER, and it is a PRECONDITION for arming taper/chroma/AA/temporal-screw together (they
   all sharpen the boundary → richer normals → the very regime where the sqrt-pose-eps blowup + no-grad-
   clip collapse bit run-1). But the fix itself is UN-A/B'd. P0-now says yes; is it a launch-blocking
   dependency or a parallel arm?
4. **EVENT-TRIGGERED vs EPOCH SCHEDULE closure + TRAIN-VERDICT DECOUPLING as a stage-exit signal.** v7.5 is
   ~80% event-driven; the remaining epoch backstops (min-stage 250, Polyak 2546, hosc β 3.177, muon cap
   726) are FAIL-SAFES. Fully-derive the min-stage into the τ_relax critical-slowing law (S-4, DERIVED-AT-
   CONFIG) — or keep the hand-set safety margin as the fail-safe per the §D CONTROL-LAW contract (every
   anneal provably completes before its consumer fires, OR truncation is event-safe)? Which epoch caps
   become events, which stay fail-safes? **NEW (operator addendum, R-4): make train-verdict DECOUPLING a
   first-class stage-exit event** — a stage that descends train loss while the VERDICT d_seg rises is
   CE-exhausted ("more of the same gradient won't cure it") ⇒ that decoupling IS the hand-off/re-treat
   trigger, distinct from (and more informative than) a plateau in train loss. The alarm (defaults-ON, being
   built in parallel) must distinguish genuine decoupling from the EMA-shadow-lag artifact (R-2). (The
   STRUCTURE-BLIND seat S6 answers the stage skeleton independently; S3 owns the decoupling-as-exit control
   law.)
5. **POSE FINISHING + TERMINAL SOLVE composition.** Terminal pose-finish (D.9, fires at muon cap 726) —
   right sequencing, or does the σ_min-basin trigger (P-6, earlier-engage arm) win? Does v7.5.2 SHIP the
   R1 dxi (0.127, 7.2 KB banked) via terminal finish, given P-5's HONEST FLAG (mechanism correct,
   efficacy UNVALIDATED — the v7.5 terminal finish may NOT converge to an R1-class dxi from its own
   basin)? And does #341 terminal head GN/CG solve (L-8, full-P, ~3h, fire IFF ρ re-verify) compose with
   / replace the pose-finish terminal stage — solve-the-head THEN pose-descend, or joint?

6. **TORCH-PARITY TWIN for MODAL A/B FAN-OUT? (operator addendum, M-2).** Witness training is MLX-local;
   Modal (≤$20 hard cap, M-1) is for byte-closed exact-eval + the CPU-torch n600 verdict authority. Is a
   torch-parity twin of the witness worth BUILDING to fan out ON/OFF lever A/Bs in parallel on paid contest
   hardware — vs the serial local byte-close + governed CPU verdict? Weigh the parity-twin build cost +
   MLX↔torch numeric-parity risk (bit-identity is per-{chip,os,mlx,device}; but the byte-closed ARCHIVE is
   the shared invariant, so exact-eval on the archive is parity-safe by construction) against the fan-out
   throughput within the $20 cap. S4 owns this.

---

**Adversarial self-check (per the anti-blind-spot contract):** cross-checked B/C/D/E against the DAG FEED
list + the activation-ledger seeded top (relsigfold: taper 73% · horizon 43.8% · StepNative 31.6% · d18-k90
2.4% · mod32-neutrality 1.2% · seg_down_weight_274 est-owed) + the solve-don't-train inventory (8 rows,
#341/#288 top). Levers explicitly INCLUDED that an absolute-ΔS eyeball would orphan: L-1/L-2/L-3 (the
magnitude-dismissal class). Negatives INCLUDED so a seat does not re-open them blind: N-1/N-2/N-3. The two
lowest-rel-sig owed levers (d18-k90 truncate 2.4% rate A/B, mod32-neutrality 1.2%) + #274 seg-down-weight
(est-owed) are noted here as duty-to-measure tail, not detailed (near-goal any real byte cut = pure S, but
they are rate/formulation not the d_seg blocker). If a seat needs one of these, it is in FEED-relsig.
