# MAX-SIGNAL PARANOIA SWEEP — surface C (TASK LIST + .omx/research + ORPHAN ledger) → the fresh-run lever gate

**UTC** 2026-07-04 · **authority** `[$0 CPU research-consolidation / advisory]` · **score_claim** false ·
**promotable** false · **NO GPU, NO paid, NO launch, #205 READ-ONLY.** **Pointer UNMOVED contest-CPU
0.19109982** (`.omx/state/canonical_frontier_pointer.json`). This is a MEANS — the exhaustive per-lever
verdict + BUILD/CALIBRATE/MEASURE queues to gate the **FRESH SEEDED RUN** (the FEED-04b/04c pivot: preserve
#205 + relaunch with the #208 seed + #286 eikonal + geometric-τ + Ch.6 easing + dynamic controller). No score
moves here; the END is a byte-closed n600 `upstream/evaluate.py` row from the fresh run below 0.19110.
`# ORPHAN_WIN_WAIVED:this-is-the-sweep-C-orphan-ledger-that-DEFINES-the-burndown-not-an-orphaned-mechanism-win`
`# FORMALIZATION_PENDING:sweep-consolidation-memo-no-new-empirical-finding-cites-existing-registered-equations`

## Method + calibration legend
Task SoT: the canonical JSONL ledger (`.omx/state/canonical_task_status.jsonl`) is **SUPERSEDED for witness
tasks #200-225** (self-declared `canonical_task_status_ledger_superseded_by_dag_tasklist_20260701`, last updated
2026-06-18) → SoT = the DAG (`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` FEED-* chain) + live
TaskList + DSL + `tac.canonical_equations`. Every status below cites the DAG FEED / index / code path it rests
on; NO fabricated status. Calibration tags: **EXACT** (byte-closed evaluate.py) · **CPU-adv** (frozen CPU-torch
SegNet argmax) · **MLX-rs** (`[macOS-MLX research-signal]`) · **DERIVED** (math). Sources consulted: the 4
canonical research sub-indices (dseg/pose_curriculum/rate/vehicle_warp/infra_floors 2026-06-29), the measured
lever inventory (`measured_lever_inventory_for_synergy_pass_20260701`), the #225 orphan sweep
(`orphaned_measured_win_bug_class_selfprotect_and_sweep_20260702`), `tools/audit_orphaned_measured_wins.py`,
the #285 next-run config (`deepmath_converged_next_run_config_20260704`), and DAG FEED-03t..04c.

## THE BASELINE (what #205 actually launched with — verified from its `launch.sh`)
`--seed 0 --num-pairs 600 --mlx-device gpu --epochs 1000 --self-orient --n-dir-freqs 2 --freq-across 32
--mod-dim 32 --eikonal-weight 0.01 --length-weight 0.001 --island-dilate-px 1 --tau-anneal-shape cosine
--muon-start-epoch 726 --muon-lr 0.002 --w-seg 100 --w-pose 1.0 --pose-carrier-* --lane-band-* --ema-decay 0.997
--stage-checkpoints`. **CRITICAL FINDING: #205 sits at the SUBOPTIMAL value on EVERY fresh-run lever** — the
along-tangent deficit (`--n-dir-freqs 2`), the un-raised eikonal (0.01), the single-px seed (=1, lane seeded at
ZERO mass = the nucleation failure FEED-04b), the over-embed mod-dim (32 vs measured-optimal 19), the non-geometric
τ-anneal. All are **config-only flags that already exist in the live trainer** (grep-verified). The fresh run's
entire thesis is flipping them; the paranoia is that any get left at the #205 value.

---

## §1. FULL PER-LEVER LEDGER (every named task + the levers found alongside)

Verdict ∈ {INCLUDE · BUILD-then-include · CALIBRATE · MEASURE-first · EXCLUDE(why) · ALREADY-DONE}.

| Lever (task#) | What it does | Status | Cite | Fresh-run verdict |
|---|---|---|---|---|
| **along-tangent freq** `--n-dir-freqs 2→4 --freq-across 32→8` (#277/Ch5-M1) | resolve the ~25 cyc/unit lane-dash on/off modulation ALONG the tangent (basis is sharp across, smooth ≤8 along = 3.2× deficit) | **MEASURED #1 root cause; flag EXISTS in levelset trainer; #205 runs the DEFICIT =2** | FEED-03t (`anisotropic_basis_along_tangent_frequency_deficit_v1`); launch.sh | **INCLUDE (config-only, Nyquist cap `freq_across·2^(n-1)≤64` → 8·2³=64). THE highest-paranoia omission.** |
| **#208 seed (separatrix+asymmetry SDF, lane +2px)** | seed the zero-level-set ON the openpilot lane separatrix, LANE dilated +2px above the critical nucleus (MOVABLE native) | **MEASURED root cause + CALIBRATED spec; the REASON for the fresh run** | FEED-04b/04c (`lane_nucleation_failure_seed_above_critical_nucleus`); `--island-dilate-px 2` | **INCLUDE + confirm the separatrix-SDF-at-init wiring (partial BUILD: +2px flag exists; separatrix seed init path)** |
| **#286 eikonal (raise 0.01→0.05)** | make τ a real Modica-Mortola interface width so the thin lane stays sharp at τ/2; COUPLED with geometric-τ | **config-only (flag exists); Ch.4 τ=ε=ħ derived** | FEED-03y Ch.4 (`tau_eps_hbar_one_dequantization_two_scales_v1`); config memo Tier-1(2) | **INCLUDE (Tier-1 arm, do COUPLED with geometric-τ)** |
| **geometric-τ + τ_end** `--tau-anneal-shape geometric --softmax-temp-end 1.0` | equal-epochs-per-octave = the Fisher-Rao geodesic / constant-info-velocity ADIABATIC schedule | **config-only; facet-4 LANDED (derived)** | FEED-04c facet-4 (992287032); config memo Tier-1(2) | **INCLUDE + CALIBRATE τ_end {0.05,0.1,0.25} (pixel-unit convention measurement-gated)** |
| **#218 head-margin-field (ETF/add-margin/Menon/clDice/Betti/AHA)** | Laguerre logit-offset head + persistence/clDice island-recall | **BUILT** (`laguerre_logit_offset.py`; persistence gated ep300) | canon dseg; FEED-03g | **INCLUDE (persistence tau-gated; the mass-preserving head); the AHA/OT-offset piece → MEASURE-first (#288)** |
| **per-class area/mass hold** `--lane-thin-weight + --persistence-loss-weight` (tau-gated) | pin lane mass≠0 against the MCF erosion (auction-MBO's config sibling) | **BUILT** | FEED-04c facet-4; measured lever §D | **INCLUDE (the MCF-erosion INVERSION; without it a +2px seed still erodes — see paranoia #3)** |
| **#222 β₂-optimizer** `--adam-beta2 0.999` | Adam 2nd-moment; witness anchor | **RESOLVED to 0.999 (0.9999999 = MIS-ANCHOR)** | FEED §2D nexus adjudication (7369-7388) | **ALREADY-DONE (config decided)** |
| **#270 Muon warm-start + LR-anneal** `--muon-warm-start-momentum --muon-lr-final-frac 0.1` | seed Muon momentum from AdamW `m` (kills +0.000357 cold-start spike) + cosine-anneal the flat Muon LR | **BUILT+WIRED into levelset (0b847c96a); ARMED; GO'd for #205 @ep726** | FEED-03q/03r/03s (`muon_finisher_schedule_warmstart_and_lr_anneal_v1`) | **INCLUDE (fresh Muon stage should fire warm+annealed; also the live Tier-2 #205 A/B — watch ep726)** |
| **#268 S_R reachability saliency** `--margin-saliency-reachability` | exact through-R fragility-weighted margin-Jacobian weight (replaces the INERT texture proxy) | **BUILT+WIRED, VERIFIED-ACTIVE; SECONDARY multiplier → MODEST** | FEED-03n/03p (`precompute_sR_reachability.py`) | **MEASURE-first (needs `sR` built into gt_n600 + byte-closed A/B; SECONDARY so not opening-critical)** |
| **#288 OT head-offset (damped-Newton semi-discrete)** | data-aware Laguerre cell-mass rebalance b* == GT-freq; the PRINCIPLED minority-collapse/asymmetry cure REPLACING the Menon −log π heuristic; byte-FREE | **BUILT (8bc91449c, `damped_newton_ot_offsets` in `laguerre_logit_offset.py`, 6 tests); $0-gate OWED (memory-gated by #205 RSS)** | FEED-04a | **MEASURE-first ($0-gate: apply b* to #205 EMA-best → re-render R → d_seg vs Menon, n600; directly attacks the nucleation cause)** |
| **chroma-into-annulus (D9/#227)** | SegNet argmax reads RGB → chroma is an INDEPENDENT d_seg boundary DOF at the lane crux (7.54% Lane→Road on removal, 93.4% in the margin<1 annulus); seg⊥pose frees the seg-frame chroma FREE | **MEASURED GREEN (n96, 100% L*-match); A/B OWED; verdict-BLOCKING (baked into baseline, never A/B'd)** | FEED-03t chroma probe (`chroma_decides_lane_and_movable_at_annulus_v1`) | **MEASURE-first (every pre-chroma verdict is PROVISIONAL; `--chroma` on the lane facet)** |
| **vector-t margin-saliency (VECTOR upgrade)** | t=M_p/(M_p+M_q) = FREE sub-pixel boundary position + flip-direction; upgrades margin-saliency #141 SCALAR→VECTOR | **MEASURED GREEN (self-consistent); VECTOR gauge design-stage; A/B owed** | FEED-03t vector-field probe (`separatrix_asymmetry_t_subpixel_boundary_localizer_v1`) | **BUILD-then-include (scalar exists; VECTOR-t upgrade owed; effect in the 1-2px flip band)** |
| **#274 seg-spike down-weight** `--seg-spike-reweight --seg-spike-downweight` | down-weight the IRREDUCIBLE-flicker pixels (the standing seg play after Lever-D NO-GO) | **BUILT (FlickerTreatmentGauge DOWNWEIGHT_IRREDUCIBLE, measured=True)** | FEED-03w/03x | **MEASURE-first (built, the active flicker seg lever; net-S A/B owed)** |
| **#183 θ*-per-lever-A/B (method)** | isolated-arm A/B per lever for clean attribution | **BUILT (campaign apparatus)** | `thetastar_per_lever_AB` | **ALREADY-DONE (the METHOD for fresh-run arms; the arms themselves are MEASURE-first)** |
| **#188/#216 trajectory-instrument + adaptive stacking** | `decide_next_stage` EXTEND/ADVANCE/RERUN/ROLLBACK; descent-rate lever; de-orphan (not rebuild) | **BUILT (`witness_dsl/campaign.py`); facet-5 de-orphan** | FEED-04c facet-5; canon C16 | **INCLUDE (emit-only monitor; the costate/dynamic-control facet)** |
| **#213 analytic lane band** `--lane-band-*` | analytic centerline render band (RGB-reaching); LBND2 rate codec | **BUILT, default-gated at ep300/350; decode-consistent** | canon vehicle; Wave-F LBND2 | **ALREADY-DONE (baseline element; deconflict start-epoch to 350 per Ch.6-L1)** |
| **#287 dash-comb (tropical max-plus comb)** | period/phase/duty dash modulation ALONG the tangent (complementary: band=centerline, comb=dash); LBND2 COUNTS 3 comb params/line | **BUILT (`analytic_lane_render_band._line_row_params`; `fit_lane_line`; `serialize_lane_band_rd`) — the config memo's "UNBUILT" was a NO-FAKE-corrected error** | FEED-04a NO-FAKE correction; config memo Tier-3(5) | **ALREADY-DONE (rate codec built) + MEASURE-first (net-S A/B of the dash modulation as a d_seg lever)** |
| **#217 leap-residual (post-Muon micro-stage)** | Damian-smoothing on lowest-persistence lane-dash + Muon-Stiefel → exp→poly escape of the slow d_seg tail | **DESIGN (2605.13079); HIGHEST-EV finishing lever; UNBUILT micro-stage** | FEED-03o item(4); DAG 7128 | **BUILD-then-include (finishing micro-stage; net-S #205-gated; sequence AFTER the #270 read)** |
| **#204/#207 sig-proc (deconv/pre-emphasis/matched-filter)** | invert R / sharpen the render | **MEASURED NEGATIVE — R is ALL-PASS to 2px → deconv DEAD ≤+1.25dB "not a d_seg lever"; matched-filter non-binding** | FEED-03t (`contest_r_operator_mtf_allpass_to_2px_v1`); sig-proc memo | **EXCLUDE(measured non-binding; R all-pass; don't build deconvolve-R/matched-filter)** |
| **#204/#207 NTK/multiscale band-pass whitening (Ch5-M2)** | per-scale amplitude ∝ 1/√λ microlocal preconditioner | **UNBUILT flag; dominant SPEED lever ~3-10× + up to −3e-4 d_seg** | FEED-03y Ch.5-M2 (`shearlet_nterm_upper_bounds_task_rate_v1`) | **BUILD-then-include (a build task, not config; EXCLUDED from opening argv; Tier-3)** |
| **#195 MD-decoupling A/B** `--optimizer md` | momentum-decoupled stable transitions | **WIRED in through-R trainer (NOT levelset); stable-by-construction but UNDER-STEPS d_seg at scale** | canon C15; FEED-03k (arXiv 2606.25971) | **EXCLUDE from opening (parallel arm; measured to under-step); MEASURE-first only if reactivated with own LR sweep at scale** |
| **#200 conditional-additive-residual (DM3′ low-rank SDF head)** | rank≈16 per-pair additive SDF correction | **SUPERSEDED for the BULK (G7: bulk needs no trained INR, stratified warp); MAY apply to residual long-tail (open)** | canon vehicle G6/G7; FEED-ir/ja | **EXCLUDE(bulk superseded) / MEASURE-first (residual long-tail only, low-pri)** |
| **#226 margin_conditional_residual (Lever-D flicker coder)** | store+induce the flip-residual (MCR waterline 1.27 B/flip) | **BUILT (`margin_conditional_residual.py` #72); reactivation MEASURED NO-GO — min b=0.876>0.65, net ΔS +0.202 WORSE, band-localization fails (5.2% capture)** | FEED-03x (`leverd_flicker_residual_reactivation_economics_v1`) | **EXCLUDE(measured NO-GO at the converged base; keep #274 down-weight instead — confidently-closed door)** |
| **#242 flat-minima / MDL-weight-compression** | MDL/entropy weight penalty for bytes | **Ballé weight-entropy λ = NET-NEGATIVE at every λ (+2.8 WORSE); no specific flat-minima built lever; rate NOT binding** | measured lever §F (`lever2_lambda_star_sweep`) | **EXCLUDE(MDL rate levers net-negative on this vehicle; seg is binding, not rate)** |
| **#248 pose-carrier-ladder (P-B/P-E/P-F)** | cheap image-space pose carrier below R1 | **RETRACTED (P-F 200× n=3 rate false-positive); P-E existence-proof-only (not shippable); store-nothing R1 (d_pose 0.0011) is the LEGAL floor** | FEED-2026-07-03 pose correction | **EXCLUDE(retracted); pose = store-nothing + #257 coder (ALREADY-DONE)** |
| **#257 store-nothing pose coder** | ξ arith-coded, H derived FREE at decode | **LANDED (2ff654ad8): 0.0347→0.0025 coded (13.9×), d_pose-invariant, bit-exact, 65 tests** | FEED-03b | **ALREADY-DONE (byte-close side; supersedes the aspirational 0.0007)** |
| **#211 meta-init (hypernet H_ψ)** | amortized per-video pre-seed | **DESIGN/horizon only (overfit-XOR-generalize)** | horizon memo | **EXCLUDE from fresh run (design-stage; serves a LATER run)** |
| **#185 θ*-TIER-3 (IPM chart / feature-space-OT)** | trainer-editing Wave-2 levers | **partial; Tier-3 (post-Tier-1)** | DAG 4454 | **BUILD-then-include (Tier-3 bucket; excluded from opening argv)** |
| **#186 l7-maturation** | l7_softplus margin-weight allocation stage | **l7 DEMOTED from default (measured defect `l7_linf_sharpening_defect` > MLX surrogate); tau_softplus is THE primary drop** | FEED nexus adjudication (7388); canon C13 | **EXCLUDE from default curriculum (DEMOTE); optional late-stage arm only** |
| **#86 EMA-warmup** | ramp EMA decay early | **NOT in DAG; EMA-shadow decay 0.997 already DEPLOYED (non-negotiable); warmup-ramp variant unmeasured** | CLAUDE.md EMA; canon C11 | **EXCLUDE(EMA-shadow deployed; warmup micro-variant has no measured EV, not gating)** |
| **#220 AA-coverage + grid≥384** | anti-alias the SDF boundary | **MEASURED: SIGNAL-A real-frame ceiling 0.00086 (floor-PROOF) vs SIGNAL-B brute-supersample WITNESS HURTS −49%; R all-pass to 2px → AA-to-beat-R non-binding at ep200** | `aa_signal_a_vs_signal_b`; FEED-03t | **EXCLUDE(ship `--render-aa none`; NEVER brute supersample) — ALREADY-DONE (verdict = none)** |
| **#260 sg-cache / #261 micro-batch** | speed only (bit-identical / opt-in batched scorer) | **BUILT+WIRED (sg-cache bit-identical; micro-batch B1≈B4 ~1.02× under contention)** | FEED-03g | **INCLUDE sg-cache (free) / CALIBRATE micro-batch B=4 (dedicated-GPU re-measure) — SPEED, no S** |
| **#252 fused-R Metal** `--fused-r-kernel` | 4.69× on the R operator | **BUILT, default-OFF byte-identical; whole-run ~1.02× (Amdahl); mx-compile REJECTED (flips argmax)** | FEED-03c | **EXCLUDE for speed (Amdahl-irrelevant); keep default-OFF** |
| **curvelet coarse→fine scaling curriculum** | multiscale directional bands, N⁻²-optimal for the C² separatrix | **partial (self-orient + max-bank-freq staging built); parabolic SPATIAL support UNBUILT; needs FROM-SCRATCH (shape-break in warm-start) — the fresh run IS from-scratch → ENABLED** | measured lever §A2; FEED-04c facet-2 | **MEASURE-first + BUILD (parabolic-shearlet front-end); the fresh run is its natural home** |
| **facet-1 metric (Fisher-NG preconditioning)** | precondition in the caustic Fisher metric | **LANDED (facet-1)** | FEED-04c | **INCLUDE (folded into synthesis)** |
| **auction-MBO volume-preserving flow** | the PROVEN minority-erasure cure as a SOLVER (not a loss) | **UNBUILT (solver queue); the principled MCF cure** | FEED-04a solver queue; Ch.4 | **BUILD-then-include (net-S-gated; the erasure cure)** |
| **#289 dynamic controller / τ-creep detector** (`tools/witness_control_monitor.py`) | stop-and-re-steer when r̂≥+δ ∧ net_Δd_seg>0 ∧ ep_loss↓ (the LITERAL #205 signature) | **SPEC'd (facet-5 LANDED as design); the TOOL FILE IS MISSING (not built)** | FEED-04c facet-5 | **BUILD-then-include (emit-only monitor; de-orphan the #188/#216 instrument; NEVER launches)** |

---

## §2. ⭐ READY ∧ high-EV BUT UN-INCLUDED (THE KEY OUTPUT — DEFER is FORBIDDEN for these)

Levers whose **code exists** and whose EV is high, that are **currently at the #205-suboptimal value or un-A/B'd**,
and MUST be judged for the fresh-run opening config:

1. **`--n-dir-freqs 4 --freq-across 8` (along-tangent frequency)** — READY (flag in the live trainer), the
   MEASURED #1 root cause of the binding lane-dash residual (3.2× along-tangent deficit). **#205 runs the DEFICIT
   `=2 / freq-across 32`.** Config-only, Nyquist-capped (`8·2³=64`). If the fresh run also launches at `=2`, it
   re-inherits the exact residual that is the whole d_seg wall. **HIGHEST paranoia.**
2. **`--island-dilate-px 2` + separatrix-SDF-at-init (#208 seed)** — READY (dilate flag exists), the CALIBRATED
   nucleation fix (LANE +2px → 98.3% MCF survival vs 44.6% native). **#205 seeds lane at ZERO mass (`=1`) — the
   measured creep cause.** This IS the reason for the fresh run; must land at init (can't retrofit by resume).
3. **`--eikonal-weight 0.05` + `--lane-thin-weight`/`--persistence` area-hold (MCF-erosion inversion)** — READY,
   the pair that makes the +2px seed GROW not erode (length-weight IS the MCF driver; keep small + raise eikonal +
   pin mass). **#205 runs eikonal 0.01, no area-hold.** Omitting = the seed re-erases = #205's failure repeats.
4. **`--muon-warm-start-momentum --muon-lr-final-frac 0.1` (#270)** — READY+WIRED, kills the measured
   +0.000357 cold-start spike + the flat-LR plateau. A fresh Muon stage should fire warm+annealed from the start.
5. **`--chroma` on the lane facet (D9/#227)** — READY, MEASURED GREEN independent d_seg DOF at the lane crux;
   **verdict-BLOCKING** (every pre-chroma verdict is PROVISIONAL) and FREE via seg⊥pose. At risk of being left as
   "baked-in baseline, never A/B'd."
6. **`--margin-saliency-reachability` (#268 S_R)** — READY+WIRED+VERIFIED-ACTIVE; SECONDARY d_seg refine; needs
   only the `sR` cache built into gt_n600 (a $0 build) + the A/B.
7. **OT head-offset b* (#288)** — READY (built `damped_newton_ot_offsets`), the principled cure for the EXACT
   nucleation/asymmetry crux, byte-free, replaces the Menon heuristic. $0-gate owed (memory-gated by #205 RSS).
8. **`--seg-spike-downweight` (#274)** — READY, the standing seg play for irreducible flicker (after Lever-D NO-GO).
9. **`--mod-dim 19` (facet-2)** — READY, MEASURED: `--mod-dim 32` is pure over-embed WASTE (−41% code DOF at equal
   d_seg; intrinsic m≈8 → Whitney 17-19). Rate-save; d_seg-neutrality UNMEASURED at n600 (#223) → CALIBRATE.

## §3. BUILD-then-include queue (code does NOT yet exist; net-S #205-gated, operator-GO-gated)
- **`tools/witness_control_monitor.py`** (#289 facet-5) — spec'd, file MISSING; wraps the existing #188/#216
  instrument, emits decisions+config-diffs ONLY (never launches). De-orphan, don't rebuild.
- **NTK/multiscale band-pass whitening flag (Ch5-M2, #204/#207)** — UNBUILT; SPEED ~3-10× + up to −3e-4 d_seg.
- **auction-MBO volume-preserving solver** (#289/FEED-04a) — the proven-erasure cure as a solver.
- **vector-t VECTOR saliency upgrade** (FEED-03t GREEN) — scalar exists; VECTOR gauge design-stage.
- **#217 post-Muon leap-residual micro-stage** — highest-EV finishing lever; sequence after the #270 read.
- **parabolic-shearlet spatial-support front-end** (curvelet dim facet-2 MISSING piece).
- **`sR` cache into gt_n600** (#268) — the $0 build that unblocks the S_R A/B.

## §4. CALIBRATE queue (built/config; need a $0 or small-n calibration before the launch value is set)
- **geometric-τ `--softmax-temp-end` / τ_end {0.05,0.1,0.25}** — pixel-unit convention measurement-gated (FEED-04c).
- **`--eikonal-weight 0.05`** — the interface-width knob; do COUPLED with geometric-τ (one arm).
- **`--island-dilate-px 2`** — n24 says 98.3% survival; confirm at init/n600 + the critical-nucleus knee ($0 probe).
- **`--mod-dim 19`** — rate-save d_seg-neutrality UNMEASURED at n600 (#223 sweep folds only if d_seg-neutral).
- **micro-batch `--micro-batch-pairs 4`** (#261) — needs a dedicated-GPU re-measure (B1≈B4 under #205 contention).

## §5. MEASURE-first queue ($0 or byte-closed A/B owed BEFORE the lever's launch verdict is load-bearing)
- **OT head-offset $0-gate (#288)** — apply b* to #205 EMA-best → re-render R → d_seg vs Menon vs w=0, n600
  (memory-gated: HELD while #205 at ~61.5 GB; run on #205-free / cached witness output / GO).
- **per-class d_seg attribution on the REAL #205 witness @n600** — confirm the creep IS lane at scale (the
  operator's "more smokes before deciding"; memory-gated).
- **chroma A/B (D9/#227)** — GREEN, verdict-BLOCKING; the seg-frame chroma is FREE via the pose sidecar.
- **S_R reachability A/B (#268)** — after the gt_n600 `sR` build.
- **dash-comb-as-d_seg-lever A/B (#287)** — the rate codec is built; the d_seg net-S is un-A/B'd.
- **#274 seg-spike down-weight net-S A/B** — built; the flicker seg lever's net-S owed.

## §6. STALE / SUPERSEDED / CLOSED (do NOT re-open or re-litigate)
- **`canonical_task_status.jsonl` for #200-225** — SUPERSEDED by the DAG (self-declared, last real update 2026-06-18).
- **#248 P-F cheap image-space pose optimum** — RETRACTED (200× n=3 rate false-positive); #257 supersedes.
- **#222 β₂=0.9999999** — MIS-ANCHOR; superseded by the MEASURED 0.999.
- **#186 l7 as a default curriculum stage** — DEMOTED (`l7_linf_sharpening_defect`); tau_softplus is the primary drop.
- **config memo #285 "#287/#218 UNBUILT" + "M1 needs a build"** — WRONG; dash-comb + Laguerre head + along-tangent
  flag all BUILT (NO-FAKE corrected FEED-04a). Only M2-NTK is genuinely unbuilt.
- **#200 DM3′ additive head for the BULK** — SUPERSEDED (G7 stratified warp; bulk needs no trained INR).
- **#226/#279 Lever-D flicker-residual coder** — MEASURED NO-GO (FEED-03x: b=0.876>0.65, net ΔS +0.202); CLOSED
  (implementation-level, paradigm intact) — keep #274 down-weight.
- **finishing-kit "−0.005 sub-0.19"** — RETRACTED (double-counted already-spent bytes; lossless rate on the borrowed
  0.19110 frontier is EXHAUSTED).
- **"pose is a d_seg-competing lever"** — RULED OUT (pose descends free / rides the stored sidecar; `--w-pose 1.0`
  on the fresh run per nexus adjudication, but pose is not a controllable d_seg lever).
- **deconvolve-R / pre-emphasis / matched-filter (#204/#207) / brute-supersample AA (#220) / mx-compile (#252)** —
  MEASURED NEGATIVE / non-binding (R all-pass; supersample HURTS −49%; mx-compile flips argmax). DON'T build.
- **Ballé weight-entropy λ / MDL weight penalty (#242)** — NET-NEGATIVE at every λ; keep λ=0 OFF.
- **198:1 annulus anisotropy** — DISPUTED; re-measured 9.56:1 / 37.8:1 — do NOT quote 198:1 as settled.
- **§L "level-set has no exact-eval path" (2026-07-01)** — RESOLVED: `tools/levelset_byte_close_and_eval.py` EXISTS.

## §7. TOP-5 PARANOIA FLAGS (levers most at risk of being wrongly OMITTED from the fresh run)
1. **`--n-dir-freqs 4 --freq-across 8` (along-tangent frequency).** #205 runs the DEFICIT `=2/32`; it is the
   MEASURED #1 root cause of the lane-dash residual (the binding wall), a **one-token config change with the
   highest measured EV.** If the fresh run copies #205's argv, it re-launches into the exact wall it was meant to
   escape. Verify the launcher sets `4 / 8` (Nyquist `8·2³=64` — do NOT leave freq-across at 32 with n-dir-freqs 4).
2. **OT head-offset b* (#288, built) is un-run.** It is the PRINCIPLED cure for the EXACT lane-nucleation-failure
   that JUSTIFIES the fresh run, byte-free, replacing a heuristic — yet it sits behind a memory-gated $0 gate. If
   the fresh run relies on the crude +2px dilation + Menon alone, the strongest built cure is orphaned.
3. **The MCF-erosion INVERSION trio (eikonal 0.05 + area/mass-hold + small length-weight) omitted while keeping
   the +2px seed.** FEED-04c is explicit: `--length-weight` IS the MCF driver; a wider seed WITHOUT raised eikonal
   + area-hold still erodes. Shipping the seed alone reproduces #205's creep — a subtle, high-cost half-fix.
4. **`--muon-warm-start-momentum --muon-lr-final-frac 0.1` (#270, built+wired) not carried into the fresh Muon
   stage.** The cold-start +0.000357 spike + flat-LR plateau are MEASURED; a fresh run that cold-starts Muon at
   flat LR re-inherits both. Built and GO'd — easy to forget it is a *fresh-run* setting, not just the #205 restart.
5. **chroma (D9/#227) left "baked-in, never A/B'd."** It is a MEASURED GREEN independent d_seg DOF on the lane
   crux and FREE via seg⊥pose, but it is verdict-BLOCKING: every pre-chroma verdict is PROVISIONAL. Launching the
   fresh run without a chroma-active arm leaves the whole d_seg ledger provisional and forfeits a free lane lever.

---
**NO-FAKE ledger:** every status cites its DAG FEED / index / code path; measured-negatives (Lever-D, deconv,
Ballé-λ, brute-AA) are recorded as CLOSED doors with numbers, not chased. No score moved — this is the MEANS that
gates the fresh seeded run; the END is a byte-closed n600 `upstream/evaluate.py` row below 0.19110. **Pointer
UNMOVED contest-CPU 0.19109982.** #205 READ-ONLY throughout (indices + code grep only; no trainer/run touch).
