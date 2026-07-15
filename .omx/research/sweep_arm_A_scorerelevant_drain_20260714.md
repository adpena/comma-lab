# Sweep Arm A — SCORE-RELEVANT drain (rate/pose/d_seg), 2026-07-14

**Authority:** every number here is `[macOS-CPU advisory research-signal]` on FROZEN caches — NOT a
score claim, NOT a training result. **Pointer UNMOVED: 0.19108 submittable / 0.18804 borrowed-bank.**
Everything here is MEANS; only a byte-closed `upstream/evaluate.py` n600 exact row moves the pointer.
All verdicts scoped on the ladder INSTANCE < FORMULATION < FAMILY < PARADIGM; no naive→binary NO-GO.

Arm A domain: `.omx/research/` (this report + designs) + NEW `experiments/probe_*.py` + tests. Owned by
sibling arms and ROUTED not touched: `src/tac/witness_dsl/` (Arm B), `.omx/state/*ledger*` (Arm C),
`src/tac/preflight.py`, `src/tac/canonical_equations/` code.

---

## A. EXECUTED $0 (measured, optimal-form) — 3 new probes + 1 calibration

### A1. #268 margin-saliency REACHABILITY orthogonality — `experiments/probe_msal_reachability_orthogonality_268.py`
**MEASURED n600 (gt_n600 margins + gt_n600_sR):** Spearman(S_R, fragility-weight `exp(-m/τ)`) on the
fragile annulus = **+0.0088** (p10..p90 −0.022..+0.040) → **orthogonal**; S_R annulus **cv = 0.79**
(NOT flat — 27.4% of annulus pixels carry S_R>0.5) → informative. Verdict **ORTHOGONAL_ADDS_SIGNAL**.
- Meaning: LEVER-4's `sal *= S_R` multiply concentrates on a *fragile-AND-reachable* subset that the
  fragility weight alone does not prioritise, and S_R has real annulus dynamic range (guarded against
  "orthogonal == flat" laundering). This is the missing "is the multiply worth it?" evidence.
- The trainer already MEASURED the sister axis (texture-proxy `1/(1+β·tex)` vs S_R Pearson −0.033 = INERT).
- **DISPOSITION → ROUTE-STRENGTHENED** the #268 training A/B (still operator-GO GPU; see C1). $0 evidence
  now says the reachability weight is orthogonal + informative, so the A/B is worth firing.
- verdict-scope: FORMULATION (a training A/B is the arbiter of realized d_seg).

### A2. #288 flip-weighted (annulus) OT head-offset reformulation — `experiments/probe_ot_flipweighted_mass_288.py`
The bulk mass-matching head-offset was MEASURED NEGATIVE n600 (no_offset 0.0031436 < menon 0.0033119 <
ot_newton 0.0048921). The OPEN reformulation was "match the boundary-ANNULUS mass, not bulk cell mass."
**$0 BOTH-ARMS** via the closed-form Menon offset `b_k=−τ·log(π_k)` (needs only class priors) + the
ALREADY-MEASURED `per_class_1d_curves`.
- **VALIDATION GATE PASSED:** my closed-form bulk-menon offsets reproduce the OT probe's measured menon
  offsets EXACTLY (max|Δ|=0.0) → the method is faithful.
- **MEASURED:** GT bulk priors [0.23,0.006,0.50,0.012,0.25] vs GT annulus priors (band 1.0)
  [0.49,0.16,0.16,0.08,0.11] — dramatically different (Lane 0.6%→16%). Annulus-menon offsets
  {0:−1.11, 1:−0.006, 2:+0.03, 3:+0.68, 4:+0.40} are LARGE on the focus classes (Road −1.11), **outside
  the measured flat ±0.4 grid**, in the same extrapolation region as the measured-WORSE full menon.
- **The decisive structure:** the measured realized-through-R d_seg-vs-per-class-offset surface is
  **minimised at offset=0** (best in-grid offset gives Δ=−3.4e-8; full menon +1.7e-4 WORSE). The
  flip-weighted reformulation only re-selects a non-zero offset on that SAME surface → predicted ≥ baseline.
- **DISPOSITION → PREDICTED_NOGO** (surface minimised at 0). De-prioritise the realized n600 arm — it would
  only confirm. verdict-scope: FORMULATION (flip-weighted target + head-offset MECHANISM; the in-training
  coherence arms remain the live path, sister of #307's "coherence is a training outcome").

### A3. #140 / mod-fold — cross-checkpoint `code` SVD eff-rank — `experiments/probe_code_effrank_cross_ckpt_140.py`
FEED-fl measured (v3_n600 EMA only) code eff-rank 13.5 / 90%@21 → mod-fold 32→21 = −12KB decoder-free, and
flagged "eff-rank measured at h96 only" as an unmeasured coupling. **$0 rate probe** = SVD eff-rank of the
`code` matrix on ALL cached checkpoints (mod 19/26/32).
- **MEASURED:** mod-32 codes carry 90%-energy within rank ≤ 18 across ALL checkpoints → **mod 32→19 fold is
  rate-SAFE** (<10% energy loss), confirming FEED-fl's −12KB win generalises.
- **NEW finding:** eff-rank is **VEHICLE-DEPENDENT (spread 13.5)** — mod32cap ep650 code is eff-rank 1.09
  (rank-1), but the **live V9·CGauge_432 mod-19 code is eff-rank 14.5, 90%@15 (79% of its width) =
  NEAR-SATURATED.** FEED-fl's "13.5" is NOT a universal constant.
- **DISPOSITION → ROUTE the mod-19-vs-32 rate A/B (C4) with two refinements:** (a) 32→19 is rate-safe
  everywhere (bank the −12KB / −40% latent table); (b) on the LIVE V9 vehicle do NOT expect a sub-19 rate
  win — the mod-19 code is already near-saturated, so a mod-16 arm risks energy/d_seg loss. **This directly
  informs Arm B's V9 config (mod-19 is the floor on V9, not over-wide).** verdict-scope: FORMULATION (rate
  structure; d_seg-neutrality of the fold is the run-gated arbiter — a render roundtrip).

### A4. LEVER-4 target/tau calibration (math-optimal loss, $0) — measured inline
FEED-fl flagged `--margin-saliency-target 0.5` as "arbitrary" (should be the measured separatrix margin
0.476). **MEASURED (gt_n600 margins):** median GT margin **within the fragile annulus = 0.483**; mean margin
of the most-fragile 5% of pixels = 0.98; bulk median margin 5.89. → The default **target=0.5 and tau=0.5 are
math-JUSTIFIED** (≈ the measured separatrix scale 0.48), NOT arbitrary. **DISPOSITION → the FEED-fl
"arbitrary target" concern is REFUTED at $0**; keep target≈0.5/tau≈0.5 (a re-tune to 0.476 is within noise,
not worth a config change). Sister to A1 (same LEVER-4).

---

## B. DONE / genuinely-dead (proactive-recall; verdicts already MEASURED — do NOT re-run)

| # | item | verdict (measured) | scope |
|---|------|--------------------|-------|
| #307 | contour-string + digital-straightness flip coder vs 0.65 B/flip | **NO-GO n600: 0.820 B/flip** (beat 0.876 bz2, still >0.65). Residual = fragmented confetti (mean 3.1px comp, 44.6% singletons); anchors 45% of stream. Reformulation "coherence is a TRAINING outcome" → routes to in-training island arms. | FORMULATION |
| requential/MDL (2607.11883) | parent framing of #226/#307 flip coder | routing memo only; the $0 probe IS #307 (NO-GO above). ORGAN n=1 curriculum built but INSTANCE-only. No new $0 probe. | — |
| #242 | MDL/Ballé weight-entropy penalty (rate) | **DOMINATED at current operating point:** weights at i.i.d. entropy floor (FEED-fl: brotli==entropy, AC-coder slack 0.00); sub-int8 bit-alloc measured NO (WF-mb6 +0.114); WeightWatcher trunk near-critical (α→2 headroom ≈0). λ=50 sweep costs d_seg +0.038 (+3.80 S) — real R/D, NOT free. | INSTANCE/regime (reactivate only on a wider net / different vehicle) |
| msal_uni texture proxy | LEVER-4 texture UNIWARD vs S_R | **INERT: Pearson −0.033** (in-trainer comment; superseded by S_R reachability, see A1) | FORMULATION |
| #144 | polynomial-fill lane survival | **DONE: lane SHAPE captured perfectly** (shape-only false-neg d_seg 0.00046 < target 0.00087); full-band residual = 90% DASH-gap false-positives → dash on/off is the residual. Folds into #137 build. | — |
| #288 bulk | OT bulk mass-matching | MEASURED NEGATIVE n600 (see A2). | FORMULATION |
| #301 | focal-γ HOLD | γ*=0 CONFIRMED (island-starvation premise FALSE; non-monotone weight share). Do NOT reopen. | — |
| palette add-back | chroma palette add-back | REFUTED n600 ($0; context-dominated/palette-irreducible). | — |
| iga_ntk_309 | NTK boundary-tangent preconditioning | **HELD-WEAKENED → dead-adjacent:** the 3.2× along-tangent deficit STANDS, but BOTH basis-level cures (owed-16 v1 basis ≈0 zero-shot AND owed16v2 REBALANCE no-benefit) MEASURED ≈0 → negative↔cure points AWAY from the NTK-basis axis. Reactivate only if a fresh derivation names a mechanism surviving both measured negatives. | FORMULATION (basis-preconditioning); the deficit is real, its basis cures are not |
| curvelet_from_scratch_trajectory | curvelet init | reformulation-queue; MEASURED naive-palette ceiling F=0.0337 ≫ directional-direct 0.0037, trained trunk redundant with basis |Δ|≤1.4%. Gate: a fresh mechanism must survive BOTH measured negatives before build. | FORMULATION (re-derivation owed, not a build) |

---

## C. ROUTE (run-gated) — fire-tickets + NAMED measured blockers (operator-GO)

**All of C need a GPU training arm or a byte-closed launch — the diagnostics are $0 but the VERDICT is
run-gated. None is "deferred": each is a concrete launch config with a named blocker.**

- **C1 · #268 S_R reachability training A/B** — BUILT (`--margin-saliency-reachability`, gt_n600_sR ready sha
  d218d07b), NEVER FIRED. Fire-ticket: warm-start #205-class arm, `--margin-saliency-weight >0
  --margin-saliency-reachability`, `--micro-batch-pairs 1` (batched twin doesn't consume S_R yet), A/B on/off
  at fixed steps. **Blocker: operator-GO GPU run.** Evidence from A1: orthogonal+informative → worth firing;
  honest scope SECONDARY multiplier → MODEST d_seg refine.
- **C2 · chroma_rung #227/#276** — registered-off `chroma_annulus_addback_ab` (0-byte, byte-matched). DOF
  receipt MEASURED (constant-luma removal flips 7.54% Lane→Road, 93.4% in annulus). **ADD-BACK ΔS UNMEASURED.**
  Fire-ticket: warm-start byte-closed A/B OFF vs `--seg-chroma-boundary-weight {0.05,0.10}`. **Blocker: operator-GO.**
- **C3 · weight_entropy_penalty_balle #08j** — BUILT (`WeightEntropyPenaltyMLX`), DSL-held, never fired at
  n600 (may be SUPERSEDED — always-on in V9·CGauge). λ* sweep {5,15,30} owed. **Blocker: operator-GO GPU λ*
  sweep** (λ=50 known +0.038 d_seg harm — need the low-λ knee). $0-available: `measure_decoder_weight_symbol_entropy`
  on a cached ckpt confirms current entropy (but that = the archive cost already; the LEVER needs a run).
- **C4 · mod-19-vs-mod-32 rate A/B** — MEASURED −40% latent table / ≈−12KB decoder-free; A3 confirms 32→19
  rate-SAFE all checkpoints. Fire-ticket: matched exact-byte A/B at launch, `#299 Arm-A` revert-rule
  pre-registered. **Blocker: operator-GO fresh-start launch.** A3 warns: on V9 do NOT try sub-19 (near-saturated).
- **C5 · #226 margin_conditional_residual (Lever-D)** — BUILT; net ΔS band EXPECTED −0.048, pessimistic +0.117;
  break-even recovery r is the deciding unknown; rate axis alone 0.90 B/flip > GO bar (consistent with #307's
  0.820). Fire-ticket: Stage-0 `--seg-flip-residual` scaffold on frozen #205 ckpt READ-ONLY, 32-pair subset →
  REAL coded B/flip + net recovery r; GATE proceed only if B/flip<~1.0 AND subset ΔS<0. **Blocker: operator-GO
  small gated run.**
- **C6 · #227 seg⊥pose decoupling** — freeze-then-joint + stored-pose sidecar deletes the MEASURED +0.70
  seg-pose Hessian cross-term + frees decoder capacity + frees seg-frame chroma for d_seg. Fire-ticket: compose
  stored-pose sidecar + freeze-then-joint schedule in the realized-through-R MLX trainer (folds into a witness
  run). **Blocker: operator-GO witness run.**
- **C7 · eikonal-viscosity #316 fair test** — the "viscosity NO-GO" is RETRACTED (spike-guard median-freeze
  confound); n24 fair-test measured ε=0.3 STABLE + d_seg 2.3× better, ε=1.0 explodes. Fire-ticket: rollback-mode
  fair n600 (first non-confounded eikonal measurement), isolates eik-weight 0.05-too-high. **Blocker: operator-GO.**
- **C8 · DashComb #287 n600 A/B** — the C²-optimal dash-erasure cure (dedicated comb/phase term, NOT bank
  rescale; √64=8 bank is C²-optimal). Fire-ticket: GT-conditioned comb-registration audit first, then n600 A/B.
  **Blocker: operator-GO + the audit.**
- **C9 · sensitivity KKT bit-alloc #336 reformulation** — mixed 3–8-bit reverse-waterfill MEASURED REJECTED
  (+8.72; d_pose blew up 152.6). Reformulation owed = **d_pose-aware / per-tensor-freeze** allocation +
  lossless-chart re-derive. **Blocker: design (D-side) then operator-GO n600.** (design in D5.)
- **C10 · built-never-fired levers, mostly composed in v752** (focal_boundary_distance_301, head_geometry_218_etf,
  persistence_topology_224, hardness_oversample_lever5, per_param_grad_normalize, length_sigma_1b,
  dseg_aware_taper, n323_ladder_island_homotopy, seg_form_unify_tau, tail_k_warm_restart, r7_polyak_finisher,
  step_native_finer) — all BUILT + default-off/composed; each verdict needs a training run. **Blocker: fold
  into the next composed launch (operator-GO); isolated single-lever attribution A/Bs are low-priority.**
- **C11 · #400 terminal-band diagonal / MC-finisher** — BUILT, CPU-exact-gated (cheap, not a train arm);
  campaign staged with a MODAL-HOLD candidate (S_authority advisory 0.19081, sha 9c2afa96). **Blocker:
  operator-GO Modal exact eval** (the ONLY thing that would move the pointer from this arm). UGC/REINFORCE
  terminal-polish alternatives MEASURED-LOSE.
- **C12 · post_muon_sgld_217** — gated on the #216 saddle-to-saddle signature test ($0, run first — but a
  training-dynamics probe, not a cache recompute). **Blocker: #216 signature test then operator-GO.**
- **C13 · simpletes_k_gt1_319** — gated behind #315 + BINDING backtest against v1-v5+#205 logs. **Blocker:
  #315 + backtest, then advisory-layer only.**
- **C14 · linear-reparam W_birth,c ∝ (P/A)_c island-birth lever** — occupancy order-param $0-diagnostic done
  (island birth = saddle-node bifurcation); hysteresis UNMEASURABLE at $0 by construction (needs the flow
  sweep). **Blocker: DSL Lever (Arm B) + operator-GO quasi-static EMA-BEST resume.** → routed to Arm B.

---

## D. DESIGN / build-owned — COMPLETE build-tickets (build touches owned modules; ROUTED not touched)

### D1 · fisher_gn_head_full_p_solve (#341-adjacent)
- **Math (DERIVED):** margin = Fisher surrogate (ρ0.978, L1); the quadratic head chart is CONFIRMED (LM ρ
  0.847/0.868). K=8 SUBSET solve OVERFITS (+5.1% net, `quadratic_head_chart_subset_solve_gap_v1`) → ONLY a
  full-P in-trainer Gauss-Newton/CG solve on the per-class head bias is admissible. Solve
  `(J^T J + λI) Δb = −J^T r` where J = ∂(realized margin)/∂(head offset) over ALL P pairs, r = margin residual.
- **Lever + where-it-wires:** an in-trainer CG solve block (NOT a flag) in the terminal band of
  `experiments/train_levelset_witness_realized_through_R_mlx.py`, reusing the LEVER-4 shared realized `_signed`
  margin; ~11min/CG-iter @17× MLX (L77). New DSL `Lever` factory (`HeadFisherGNSolve`) in `witness_dsl/` (Arm B).
- **Measurement gate:** full-P solve d_seg vs baseline through byte-close on frozen EMA; ADOPT only if
  Δd_seg<0 AND rate-neutral (byte-free into out_sdf.bias). **Owed eq anchor:** extend
  `quadratic_head_chart_subset_solve_gap_v1` with the full-P solve result.
- **Route:** build owned by trainer+DSL (Arm B). Blocker: GPU (11min/iter) → operator-GO.

### D2 · md_decoupling_195 (#27, matrix-reparam optimizer)
- **Math:** MD (matrix-decoupling) reparam is stable-by-construction (29× more gnorm-stable, 13/13 tests in
  `src/tac/optimization/md_decoupling.py`) but its realized-descent advantage is UNVERIFIED on THIS vehicle
  (that IS the A/B). Distinct from #227 seg⊥pose decoupling.
- **Lever + where-it-wires:** a `--optimizer {muon,md}` / `--md-base` selection in the levelset trainer
  (TRAINER-GAP: no such flag exists) — wire `md_decoupling.MDOptimizer` as an alternative to Muon in the
  optimizer-construction site; DSL `Lever` (`OptimizerChoice`) in Arm B.
- **Measurement gate:** LR-transfer A/B (Muon vs MD at matched steps) → realized d_seg descent slope. ADOPT
  only if MD's realized d_seg descent ≥ Muon's at equal wall-clock (Muon is −32% d_seg vs AdamW, the bar).
- **Route:** trainer+DSL owned. Blocker: operator-GO A/B run.

### D3 · swa_tail_soup (cross-cycle weight soup)
- **Math:** the finite-τ turnpike produces K basin-endpoint iterates (TAIL cycles); a cross-cycle uniform
  soup (SWA) is the O(1/√n) tail-mean complement to the built PolyakFinisher (in-cycle) and EMA. EMA is NEVER
  replaced.
- **Lever + where:** a checkpoint-space op (byte-close-side tool + a trainer export hook collecting per-cycle
  endpoints), NOT a flag. Build a NEW `experiments/`/`tools/` soup builder consuming the per-cycle EMA npz +
  a trainer hook that saves cycle endpoints. DSL: N/A (export-side).
- **Measurement gate:** soup-EMA vs PolyakFinisher vs plain-EMA d_seg through byte-close on the SAME cycle
  endpoints; ADOPT the argmin. **$0-ADJACENT once cycle endpoints exist** (pure numpy average + one render
  each) — but needs a multi-cycle run to PRODUCE the endpoints first → route.
- **Route:** trainer export hook (Arm B) + a NEW Arm-A soup tool. Blocker: a multi-cycle run to produce endpoints.

### D4 · sam_flat_minima_mdl_242 (PRECONDITION-gated)
- **Math:** SAM (sharpness-aware minimization) finds flat minima = low-MDL weights → smaller archive. But the
  rate-in-loss HALF is ALREADY covered by `WeightEntropyPenaltyMLX` (C3), and #242 weights are DOMINATED
  (entropy-floored, B).
- **Build-ticket WITH PRECONDITION:** build SAM ONLY IF the C3 weight-entropy λ* sweep measures INSUFFICIENT
  rate headroom AND a wider-net/vehicle reopens the #242 regime. Until then it is dominated (not "deferred" —
  gated on a named measured precondition: C3 λ* result + a non-entropy-floored weight regime).
- **Route:** trainer optimizer stage (Arm B). Blocker: C3 result + regime change.

### D5 · #336 sensitivity-KKT bit-alloc reformulation (d_pose-aware)
- **Math:** the mixed 3–8-bit reverse-waterfill MEASURED REJECTED (+8.72; d_pose blew up 152.6 — the alloc
  starved pose-critical tensors). Reformulation: **d_pose-aware allocation** = freeze (keep int8) the tensors
  whose ∂d_pose/∂quant is large (the pose head + pose-coupled trunk), reverse-waterfill only the d_seg-only
  tensors; re-derive the lossless chart.
- **Lever + where:** a per-tensor bit schedule in the byte-close/export path (`tools/` bit-alloc), with a
  d_pose-sensitivity freeze mask precomputed from the cached pose Jacobian. NOT a trainer flag.
- **Measurement gate:** joint archive d_seg + d_pose through byte-close; ADOPT only if d_pose is NOT harmed
  (the failure mode) AND net ΔS<0.
- **Route:** byte-close/export tool (Arm C-adjacent ledger + tools). Blocker: the freeze-mask derivation ($0
  from cached Jacobian) then a byte-closed eval (operator-GO).

### D6 · som_organized_codebooks (representation-side, LOW priority)
- **Math:** SOM magnification law under-allocates rare regions = the measured lane starvation; the conscience
  CURE is already embodied by the LADDER per-class λ + #218. The codebook is the only NEW piece; pays twice IF
  measured (click-polish ±1/±2 locality + temporal-delta rate).
- **Build-ticket WITH CAVEAT:** representation-side chart levers have REPEATEDLY MEASURED ≈0 realized through R
  → EXACT-GATED A/B ONLY, LOW priority. Build only after the higher-value levers (C1/C4) land.
- **Route:** representation/codebook (next-vehicle). Blocker: exact-gated A/B (operator-GO) + priority.

### D7 · pose_inverse_carrier_distill (#248/#366, research_only, BLOCKED)
- **Math:** the free frame-0 inverse drives official pose error to ~2.7e-7 (witness EXISTS); recommended
  carrier `I_{t,0}=W_ξ(I_{t,1}) + P_ψ(F_SDF(φ,I_1), ξ, c_t)` — a decoder-native distilled generator (PoseNet
  Jacobians ENCODER-side only).
- **Named blocker (8 exact, §12 of the ADVISORY):** the RECEIVER-LEGALITY contract — decoder may NOT carry
  scorer/PoseNet/Jacobians/per-pair raster/GT tables (recompute-Jacobian-in-inflate = ILLEGAL; per-pair image
  store = RATE/hide-data). Receiver does NOT yet consume v7.5.3/v8 state; ξ-only floor not remeasured under the
  repaired receiver; no legal generator+codes has a final-ZIP byte receipt. Waterline: sub-0.15 needs d_pose <
  ~4.03e-5 at d_seg=0.0007; R1 realized d_pose 0.001610 is ~40× the waterline → pose fiber + SDF partition must
  improve TOGETHER.
- **Route:** GO-BUILD-ONLY for the offline teacher + analytic-basis prototype (needs-a-build, no run authorized).
  Blocker: receiver-legality gates close FIRST.

### D8 · in_run_click_interleave / #400 diagonal → in-run
- **Math:** the n8 click row MOVED the pointer −1.7e-5 (byte-close terminal polish). The in-run interleave
  needs a trainer-side design; the promotion GATE is the #400 diagonal MEASUREMENT first (terminal-band).
- **Route:** #400 diagonal (C11, operator-GO Modal) THEN design the in-run interleave (trainer, Arm B). Blocker:
  #400 diagonal result.

### D9 · grids_bulk_inr_annulus_308 (RE-SCOPE)
- **Math:** NeurIPS'25 grids-beat-INRs-except-boundary; matched-bytes protocol pre-registered. But v8 per-class
  carriers PARTIALLY embody it.
- **Build-ticket:** RE-SCOPE against the v8 increment-1 carrier BEFORE building (negcure HELD, no matched
  violated fact). Route to a v8-aware re-derivation. Blocker: v8 increment-1 state (Arm B) + re-scope.

### D10 · kd_warm_start_129 (production linchpin, next-vehicle)
- **Math:** Hinton KD from the #205 teacher on the island band; FiLM-v2 trunk-decoupling DONE (867ff3af5,
  ∂d_seg/∂pose=0 proven); actuator pending. Production linchpin spanning export.
- **Route:** production actuator (next-vehicle, bind-all spec). Blocker: production build (operator-GO).

---

## Recursive-drain closure

Every score-relevant item surfaced (curriculum-pool 40, task-ledger 7, DAG owed-reformulations) is at a
TERMINAL state above: **3 executed-$0 (A1–A3) + 1 $0 calibration (A4); 9 done/dead with verdict-scope (B);
14 routed with fire-tickets + named blockers (C); 10 fully-designed build-tickets (D).** No item is left
"deferred"/"parked"/"TODO". Follow-ons surfaced during the drain (A3 → V9 mod-floor warning to Arm B; #288 →
in-training coherence path; C9 → D5 d_pose-aware reformulation) were folded in and driven to terminal states.

**Highest-EV next actions toward the pointer (all operator-GO):** C4 (mod-19 rate, −12KB banked + A3-safe) +
C1 (#268 S_R, A1-strengthened) as a composed launch; C11 (#400 Modal exact eval of the staged 0.19081
candidate — the ONLY listed action that could move the pointer without a fresh train). Pointer UNMOVED
0.19108 / 0.18804 — this drain is MEANS.
