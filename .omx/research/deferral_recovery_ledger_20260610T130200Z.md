# DEFERRAL RECOVERY LEDGER — "what's deferred, why, and how to recover" — 2026-06-10

**Subagent:** `task60_deferral_recovery_ledger_20260610` (READ-ONLY audit; this memo + a task-status row are
the only artifacts). **Operator (2026-06-10, task #60):** *"what has been deferred and why and what needs
to be done to recover and get back on track."* **Evidence grade:** `[macOS-CPU advisory]` / structural
audit. NO score claims, NO dispatch, `promotable=false`. $0 spend.

**Frontier (pointer, never hardcoded — `tools/refresh_canonical_frontier.py`, verified 13:00:15Z):**
- **contest-CPU 0.19109982** (recoded-R3, sha `b46897267ded…`, 177,169 B) — UPPER, ABOVE T_1 → **GOAL
  UNSATISFIED**. Now **both-axis paired** (CUDA 0.22528084, sha-identical, lossless-axis-invariant
  confirmed to 8 decimals — `cuda_pairing_recoded_r3_verdict`). Submission-blocked ONLY on two operator
  dispositions (§D row D1).
- **contest-CUDA frontier 0.20533** (pr106, sha `9cb989ce…`, 186,876 B; the recode does NOT transfer to a
  better CUDA score — R3 CPU archive scored 0.22616/0.22528 on CUDA, WORSE than pr106).
- Score law: `S = 100·d_seg + √(10·d_pose) + 25·B/N`, N=37,545,489. Byte price ≈ **6.66e-7 score/byte**.
- `RACE_MODE_ACTIVE.flag` EXISTS (dated 2026-05-14; currency unverified).

**Scope note (no-duplicative-work):** this ledger UNIFIES five prior partial maps into one cross-axis
deferral picture: `MASTER_ROADMAP_post_exhaustion_map_20260610.md` (the RANK 1-6 exploitation plan),
`orphan_harvest_recovery_ledger_20260610.md` (the R1-R5 ready-made lossless wins + §3 exclusions),
`dedup_consolidation_audit_20260610.md` (CONSUMED-vs-orphan code map), `evaluator_inverse_orphan_inventory_20260609.md`
(103-surface map), and the six named session DEFER verdicts. It does NOT rebuild those; it is the single
operator-facing "get-back-on-track" table.

---

## 0. HEADLINE — the recovery picture in one paragraph

Nothing is rotting from a *premature kill* — the **Catalog #307 audit is CLEAN** (§F: every KILL-flagged
lane is implementation-scoped with pinned reactivation criteria; the 3 lanes a naive grep flags are
decision-needed scaffolds or successor-framing references, NOT paradigm kills). The deferral landscape is
honest and well-documented. The recovery is therefore NOT "un-kill something"; it is **execution
sequencing**: (1) the frontier is a banked-but-unshipped defensive hold blocked on two operator
dispositions; (2) the ONLY ready-made *exact-axis* wins are the three PR-#112-class lossless recode items
(R1/R2/R3, ~90 LOC, ~$0.3, beats the frontier by −0.00092) — these are READY-NOW; (3) every *frontier-
breaking class shift* (lever B score-native carrier, lever C joint amortizer, lever A quotient compiler,
lever D contour coder, aimed retrain) is a real NEEDS-CAMPAIGN training/solver build, NOT a byte transform;
(4) the live crux is **lever B is CONFIRMED + the carrier's blocker is precisely located** — frame1's dual
(seg+pose) fidelity, which the palette/argmax frame1 cannot satisfy, so the live blocker is **#57's
frame1-dual-fidelity = lever C (a JOINT seg+pose frame1 carrier)**. Frozen-bytes distortion is exhausted;
score-native + aimed-retrain are the only live structural doors.

---

## 1. THE TYPED LEDGER (every deferred/orphaned item; ranked within each kind)

Columns: **item** · **kind** · **why_deferred** · **reactivation_criterion** · **current_readiness**
(`READY_NOW` / `BLOCKED_ON:<x>` / `NEEDS_CAMPAIGN`) · **recovery_action** · **priority = value×readiness** ·
**feeds_which_live_lever**.

### A. THE LIVE-LEVER CRUX CHAIN (this session's offensive frontier — the spine of recovery)

| item | kind | why_deferred | reactivation_criterion | readiness | recovery_action | priority | feeds |
|---|---|---|---|---|---|---|---|
| **A1. lever B legal-frame bridge** (`lever_b_score_native_argmax_smoke_verdict`) | verdict→campaign | NOT deferred — **CONFIRMED PROCEED-TO-CAMPAIGN**. d_seg 0.00826 @ 63,802 B (2.54× < frontier seg-share); carrier seg+pose 70,452 B vs 177,169 (−60%, advisory hypothetical S 0.120). The *next build* (legal frame) is unbuilt. | already met (both KILLs negative) | build the min-byte FRAME whose SegNet-argmax==generator argmax AND YUV6 holds the pose tube (`direct_differential_geometric_inverse_solve`); then ONE paired CPU+CUDA eval | **HIGH × NEEDS-CAMPAIGN** | **B (the live class shift)** |
| **A2. #57 frame1-dual-fidelity = lever C joint amortizer** (`score_native_pose_carrier`) | verdict→blocker | DEFER-to-frame1-dual-fidelity. The amortized pose carrier WORKS in isolation (frame0 d_pose 0.0036, 13 KB, beats naive 20×) but the score-native **palette frame1 is pose-blind** (d_pose 12.14 alone). The composition S=11.65 ≫ frontier. **This is THE LIVE BLOCKER on lever B.** | replace palette frame1 with a per-pair RGB carrier trained JOINTLY vs BOTH SegNet (d_seg) AND PoseNet (d_pose) — a convolutional per-pair-latent (HNeRV-class) carrier, NOT coordinate-INR (RD ceiling proven ~0.0036, non-monotone in capacity) | **HIGH × NEEDS-CAMPAIGN** | build the joint frame1 carrier (lever C); converges toward a smaller HNeRV decoder | **HIGH** | **C → unblocks B** |
| **A3. lever D contour/boundary coder** (`boundary_math_seg_core` + `closed_spec_boundary_solver_v1`) | verdict→gate | DEFER. Storing the SegNet argmax partition DIRECTLY is d_seg=0 by construction but costs 524.8 KB under the LZMA baseline (2.96× the whole archive) — boundary entropy too high. The frontier-base correction also declines (95% single-pixel flips, GT-snap net −536). | swap LZMA-over-labels for a margin-aware STC/UNIWARD boundary entropy coder hitting **≲170-250 B/frame** (3.6-5.3× reduction) — OR run the solver on the lever-B base (contiguous residual, repairable) | **MED × NEEDS-CAMPAIGN** | build the contour coder OR run boundary_solver on lever-B base (the #55 reactivation, partially done in `score_native_first_candidate`) | **MED** | **D (composes with B/C)** |
| **A4. lever A evaluator-equivalence quotient compiler** (`innovation_mandate_…`) | direction | the V6 thesis (shortest witness in the oracle's cell); the boundary-math seg-core IS its seg core; no end-to-end compiler exists yet | a working amortizer that emits a byte-closed descending exact score | **HIGH-value × NEEDS-CAMPAIGN (weeks)** | feed boundary_math + lever B/C primitives into Phase-4; live loop waits for a Phase-1 base | **HIGH-value, LOW-readiness** | **A (the moonshot)** |
| **A5. AFSR-1 aimed retrain** (`afsr1_smoke_verdict`) | verdict | KILLED-AT-IMPLEMENTATION (Catalog #307, NOT paradigm). Continuing the memorized frontier decoder with flip-targeted loss DEGRADED both axes monotonically (the memorized point is a sharp optimum with zero slack). | (1) LR/10 + EMA-anchored trust region + freeze most tensors; (2) **train-from-INIT at smaller arch** (fresh basin has slack); (3) **null-space-primary objective** (lever C of GOAL — put error in certified-invisible DOF); (4) multi-video/larger frame set | **MED × NEEDS-CAMPAIGN** | the RANK-1 aimed-retrain campaign (descent-proof 16-pair smoke gate FIRST, $0); use reactivation path 2+3 (fresh-init smaller-arch + null-space-primary), NOT continuation | **MED** | **C/G (the distortion-via-retrain door)** |

### B. THE READY-MADE EXACT-AXIS WINS (the only $-cheap frontier-improving moves — RECOVER FIRST)

| item | kind | why_deferred | reactivation_criterion | readiness | recovery_action | priority | feeds |
|---|---|---|---|---|---|---|---|
| **B1. R1 decoder per-tensor adaptive entropy recode** (the PR-#112 orphan; `orphan_harvest_…` R1) | orphan→ready | BLOCKED as "planning-coordinates only / requires adapter" — but PR #112 just cashed THIS exact win (−1,060 B, byte-identical). Blocker RESOLVED: grammar map (`pr101_split_brotli_codec`), AC codec (`pr103_arithmetic_codec`), `shared_pmf_model`, `constriction` ALL in-tree. | only-missing-primitive = PR#112's adaptive geometric-primed per-tensor (ρ,M,inc,ε) model, ~50 LOC on `shared_pmf_model` | **READY_NOW (SMALL-BUILD ~50 LOC + 1 paired replay ~$0.3)** | extend `byte_range_entropy_recode_materializer.py`; recode 7 brotli streams → re-pack via grammar → byte-close → paired eval | **TOP (value×readiness)** | rate axis (orthogonal to saturated distortion vertex) |
| **B2. R2 latent AR(1)+cross-dim+discrete-Gaussian range recode** (`orphan_harvest_…` R2) | orphan→ready | BLOCKED same as R1. **NUANCE the kill-record misses:** our latent verdict FALSIFIED *2nd-order re-prediction*; it did NOT test PR#112's *1st-order AR + cross-dim LS + range-coder replacing LZMA* (PR#112 measured −317 B with it). The lever is OPEN. | ~40 LOC AR + discrete-Gaussian Q_TABLE range coder; reuse `decode_latents_compact` as inverse | **READY_NOW (SMALL-BUILD ~40 LOC)** | wire into R1 materializer; same re-pack/byte-close/replay | **TOP** | rate axis |
| **B3. R3 canonical `pr110_payload_entropy_recode` materializer** (`orphan_harvest_…` R3; lane L2 exists) | orphan→ready | BLOCKED as "materializer backlog planning-only" — adapters ARE R1+R2; scaffold shell in-tree | R1+R2 land | **READY_NOW (after R1+R2)** | package R1+R2 into the reusable materializer (makes the win durable across R3/PSV3/future lanes) | **TOP** | rate axis (durable home) |
| | | | **R1+R2 combined → ~177,114 B → S ≈ 0.191117, beats PR#112 + beats our frontier by −0.00092, ZERO fidelity risk.** | | | | |
| **B4. R4 inflate-program-bytes-are-RATE-FREE (E1)** (`evaluate_py_fresh_eurekas` E1; `orphan_harvest_…` R4) | orphan→partial | NAMED follow-up never executed; blocker is a *judgment call* (compliance defensibility), not a missing primitive | audit which small sections (sidecar tables, framing constants) are defensibly procedural-as-code (PR110 mode-catalog precedent) | **SMALL-BUILD + JUDGMENT, LOW-byte-EV on current frontier** | low-priority audit; flag as a V6 witness-program subsidy, not an immediate frontier move (big sections are genuine high-entropy payload) | **LOW** | rate (V6 subsidy) |

### C. THE PENDING TASKLIST ITEMS (#30/#31/#40/#51/#54 — why pending, what unblocks)

| item | kind | why_deferred | reactivation_criterion | readiness | recovery_action | priority | feeds |
|---|---|---|---|---|---|---|---|
| **#54 — score-native allocator / runtime (the waterfiller)** | task | The allocator (`lf_payload_rate_distortion` #46 = THE LAW reverse-waterfill) EXISTS and is consumed; #54's job is to feed it the per-component seg/pose marginals from the boundary solver + pose-coupling table. On the FRONTIER base the seg-correction input is EMPTY (all components under-water). | a base with a repairable (contiguous) residual — i.e. the lever-B/C carrier — produces non-empty allocator input | **BLOCKED_ON: a repairable base (lever C)** | run the allocator on the lever-B generator base (per `score_native_pose_carrier` §6 hook #1: pose marginal concentrated in frame1 luma) once lever C exists | MED (downstream of C) | C/D (the allocator IS the integration point) |
| **#51 — #50 phase-2** | task | phase-2 continuation of an earlier unit (#50); pending behind the active score-native crux | resolve the #50 deliverable's phase-1 gate | **BLOCKED_ON: #50 phase-1** | inspect the #50 unit's verdict; sequence after the lever-C build if it feeds the carrier | LOW-MED (context-dependent) | (depends on #50 scope) |
| **#40 — HiNeRV-mechanism** | task | the faithful-HF-vehicle lever (RANK 2): our 3 vehicles share "Shared Mistake A" (skip-free decoder → mean-field → d_seg≈0.5); the F1 bilinear-skip kernel (~15 LOC) is landed but the descent + byte-close is unbuilt | wire F1 `bilinear_skip_residual_canonical` into `pact_nerv_vq`; descent-proof smoke; first byte-closed exact score | **NEEDS-CAMPAIGN (small wire-in + real descent work)** | the RANK-2 parallel vehicle bet (becomes RANK 1 if AFSR-1's smaller-arch smoke also fails — proves architecture is the bug, not objective) | MED (parallel hedge to lever C) | C (alternate carrier vehicle) |
| **#31 — E-composition** | task | composition of the byte-closed carriers; gated on having ≥1 verified composable substrate (the kitchen-sink anti-pattern guard: compose only at L4+ off a verified anchor) | a verified score-aware byte-closed substrate exists (lever B/C carrier or R1/R2/R3) | **BLOCKED_ON: ≥1 verified composable substrate** | defer until lever C lands OR compose R1+R2+R3 (the lossless stack is the safe first composition — orthogonal axes, proven) | MED (R1+R2+R3 is the safe instance) | rate-stack composition |
| **#30 — D-waterfiller** | task | the lever-D contour-coder waterfill (allocate boundary-coding bits at the margin-polytope free-budget); needs lever D's coder to exist | lever D contour coder (≲170-250 B/frame) so the partition carrier crosses the 1.27 B/flip water level | **BLOCKED_ON: lever D coder** | build the margin-aware STC/UNIWARD boundary coder (A3); then the waterfiller allocates its bits | MED (downstream of D) | D (the seg-axis rate lever) |

### D. THE DEFENSIVE BANK (frontier is held but UNSHIPPED — the readiness gap)

| item | kind | why_deferred | reactivation_criterion | readiness | recovery_action | priority | feeds |
|---|---|---|---|---|---|---|---|
| **D1. recoded-R3 contest PR submission** (`cuda_pairing_recoded_r3_verdict`) | submission | Both-axis paired (CPU 0.19110 / CUDA 0.22528, lossless-invariant). DEFER-to-operator: (a) `constriction` import NOT in the compliance allowlist (genuine NEW runtime dep from absorbing PR#112's coder — installed + PASSED on both Modal evals, but the static gate isn't told it's contest-available); (b) `src/codec_ctx.py:8` PR#112 attribution URL trips the no-network-string gate. **ALSO: fails the INNOVATION GATE** — it is a −2.6e-5 absorb-recode of PR#112's codec, within contest reporting precision; a DEFENSIVE bank, NOT the innovative submission. | operator: (1) confirm `constriction` is contest-runtime-available → add to `RUNTIME_ALLOWED_NON_STDLIB_IMPORT_ROOTS`; (2) accept the attribution comment OR relocate to a NOTICE sidecar | **BLOCKED_ON: 2 operator dispositions (allowlist + attribution)** | surface BOTH dispositions to the operator (only the operator may resolve the submission-click + allowlist policy per the autonomy contract); keep banked, ship an *innovative* carrier instead | banked (not shipped) | readiness only — NOT the innovative submission |

### E. CORRECTLY-DORMANT — DEFER-for-a-reason, do NOT touch (the guard rails)

| item | kind | why_deferred (verdict cited) | reactivation_criterion | readiness | recovery_action | feeds |
|---|---|---|---|---|---|---|
| **E1. Frozen-bytes RATE axis** (T1 cross-pair dedup / T8 latent null-proj / S12 / selector recode) | verdict | `t1_s12_lossless_stack_verdict`: T1 FALSIFIED (cross-pair MI=0, k-means net LOSS); T8 FALSIFIED (every 1-code latent step moves ≥3.97 px, latents minimal); S12 INAPPLICABLE (procedural HNeRV stores NO frame pixels); selector at entropy floor (recode = sha no-op). | a frame-storing carrier (would make S12 applicable) | DORMANT | NONE — re-confirming wastes ~$0.3 to learn nothing | — |
| **E2. lossy decoder coarsening** (`lane_lossy_coarsening_analytical` + decoder-axis/QAT verdicts) | verdict | CUDA-confirmed **0.3517 [contest-CUDA A-negative]** (`lossy_coarsening_T0312_retired_*`); decoder-axis c1/c2/c3 +0.0709/+0.0902/+0.1648; QAT-recovery +0.056. Lossy moves d_seg ~10× the rate gain (no redundant precision). **DO NOT REDISPATCH.** | a smaller decoder trained FROM INIT (AFSR-1 path 2) — NOT coarsening the memorized point | DORMANT | NONE for coarsening; the rate lever is fresh-init smaller-arch (A5) | C |
| **E3. Frame-1 seg-repair correction sidecar** (`frontier_seg_repair_pool_verdict`) | verdict | Information-theoretically incapable: 1.525 B/flip position-only floor > 1.27 B/flip break-even; 66,039 flips fully mapped; 3 carriers falsified. A sidecar cannot clear THE LAW. | a better reconstruction (RANK-1 training), not a sidecar | DORMANT (as a sidecar) | the flip-map IS a live aiming surface for lever C training | C |
| **E4. lever G zero-byte deterministic rule** (`closed_spec_boundary_solver_v1` §1 + `lever_g_…`) | verdict | The frontier seg residual is scattered single-pixel boundary noise; any global pixel-space rule pays net-negative collateral (bidirectional symmetry cancels; contour_normal +0.0344; GT-snap net −536). | a base with contiguous repairable residual (lever-B generator) | DORMANT (on frontier base) | run on lever-B base (the boundary_solver reactivation) | D |
| **E5. SNeRV stored-LF representations** (`snerv_branch_b_round2_verdict`) | verdict | Every structured rung 280-530× the 178 KB frontier. | composing the frontier DIRECTLY, not storing LF | DORMANT | NONE for stored-LF; LF entropy-coding front is the live sub-question | (research) |
| **E6. R3 CPU→CUDA promotion** (`cuda_axis_frontier_eval_verdict`) | verdict | NO TRANSFER: 0.22616 CUDA vs 0.20533 control; pose +0.0232 CUDA drift. Kill criterion HIT. | the CUDA axis needs its own seg/pose pool map (CPU-only attacks don't transfer) | DORMANT | a CUDA-axis attack is a separate campaign (RANK 3 residual) | (CUDA axis) |
| **E7. R5 pr106_latent_sidecar / latent 2nd-order re-prediction** (`orphan_harvest_…` R5, X3) | verdict | The frontier ALREADY carries the 607-B L27 sidecar (pays rent ~8×; dropping = +0.0029). 2nd-order entropy 7.52 > 1st-order 7.03 (+977 B). Superseded by frontier evolution. | n/a — exhausted on current frontier | DORMANT | NONE (R2's 1st-order AR is the OPEN sibling, in row B2) | — |

---

## 2. GET-BACK-ON-TRACK PLAN — top 5 recovery actions ranked by value × readiness

**Race-mode posture (`RACE_MODE_ACTIVE.flag` exists):** parallel-dispatch the READY-NOW items while the
NEEDS-CAMPAIGN class-shift builds run; never idle a slot while S > T_1.

| rank | action | readiness | value (derivation) | why now |
|---|---|---|---|---|
| **1** | **Build R1+R2 lossless recode → R3 materializer** (the PR-#112-class harvest) | **READY_NOW** (~90 LOC + 1 paired replay ~$0.3) | **−0.00092 exact** (177,169→~177,114 B; beats PR#112 AND our frontier; ZERO fidelity risk). The ONLY ready-made exact-axis win in the entire deferral landscape. | the operator FLAGGED this exact orphan (PR#112 cashed a win we left blocked); the blocker is RESOLVED (all reuse code in-tree); it makes the frontier durable + improves it. **DO THIS SESSION.** |
| **2** | **Build the lever-C JOINT seg+pose frame1 carrier** (#57's reactivation #1; the live blocker) | NEEDS-CAMPAIGN (descent-proof smoke first, $0) | unblocks the CONFIRMED lever B (−60% bytes, advisory S 0.120 hypothetical); the live class shift. Per-pair conv latent (HNeRV-class), NOT coordinate-INR. | lever B is PROVEN, its ONE blocker is precisely frame1 dual-fidelity; this is the spine of the offensive frontier. Start the descent-proof smoke. |
| **3** | **Run `closed_spec_boundary_solver` on the lever-B generator base** (#55 reactivation; partly done) | READY-ish (the solver + lever-B base exist; `score_native_first_candidate` started it) | the lever-B residual is 74% contiguous (repairable) vs the frontier's 95% single-pixel (unrepairable) — the solver can net d_seg down on this base | $0 local; turns the seg-correction allocator input from EMPTY (frontier base) to non-empty; feeds #54 + #30 |
| **4** | **AFSR-1 reactivation via fresh-init smaller-arch + null-space-primary** (path 2+3, NOT continuation) | NEEDS-CAMPAIGN ($0 descent smoke gate first) | the rate lever (smaller decoder re-memorizes at fewer bytes) + the GOAL's lever-C null-space-primary objective; conservative band −0.02 to −0.03, aggressive −0.02 to −0.05 | the memorized-point continuation is KILLED; a fresh basin has slack the memorized one lacks (the QAT lesson). Run as RANK-1 parallel to lever C. |
| **5** | **Surface the recoded-R3 submission dispositions to the operator** (D1) | BLOCKED_ON operator | banks a both-axis-paired defensive frontier for readiness (NOT the innovative submission — fails the INNOVATION GATE) | the autonomy contract reserves the submission-click + allowlist policy for the operator; 2 clean dispositions (constriction allowlist + attribution) are all that block the bank. Report, don't pause. |

**READY-NOW this session:** #1 (R1+R2+R3 recode) and #3 (boundary-solver on lever-B base) are $0/$0.3 and
can land now. **NEEDS-CAMPAIGN:** #2 (lever C), #4 (AFSR-1 fresh-init) — start their $0 descent-proof smokes
this session, run the full builds as detached daemons. **OPERATOR-BLOCKED:** #5 (report only).

**Correctly-dormant — do NOT touch:** all of §E (frozen-bytes rate, lossy coarsening, seg-repair sidecar,
lever G on frontier base, SNeRV stored-LF, R3→CUDA promotion, R5 latent sidecar). Re-attacking any wastes
spend to re-confirm a documented floor.

---

## 3. PREMATURE-KILL FLAGS (Catalog #307 audit) — NONE FOUND

The registry kill/falsified-flag scan (25 lanes) + the named-verdict review found **zero premature kills**:
- Every KILL-flagged lane is **implementation-scoped** with pinned reactivation criteria (`lane_gp_v4`,
  `lane_7_psd_killed`, `lane_lossy_coarsening_analytical`, `lane_owv3_*`, `lane_apogee_int7`, the
  wave3/wave5 API-crash lanes, `lane_nscs06_*`, `lane_frontier_decoder_qat_recovery`,
  `lane_frontier_seg_repair_pool`, `lane_t1_s12_lossless_stack`).
- **AFSR-1** (`afsr1_smoke_verdict`) header says "KILLED-AT-IMPLEMENTATION" but the body correctly
  classifies it Catalog #307 (paradigm intact, 4 pinned reactivation paths) — **COMPLIANT**, not a violation.
- The 3 lanes a naive grep flags (`lane_psd` = "dispatch-or-kill DECISION needed"; `lane_pfp16` = L3→L1
  *downgrade* with successor framing referencing a DIFFERENT lane's kill; `lane_a1_cuda_axis_refire_scaffold`
  = scaffold surfacing an operator decision) are **NOT kills** — they reference other lanes' kills or are
  decision-needed scaffolds. No reactivation needed.

**Conclusion:** the discipline held. The recovery problem is execution sequencing, not un-killing.

---

## 4. WIRE-IN (Catalog #125)

1. **sensitivity-map — ACTIVE:** this ledger's row A2 (frame1 carries 12.14 of pose debt vs frame0 0.0036)
   + row A3 (95% single-pixel frontier flips, 74% contiguous lever-B flips) are the live aiming inputs.
2. **Pareto — ACTIVE:** §E confirms the frozen-bytes distortion vertex is saturated; rows A1-A5 + B1-B3 are
   the only off-vertex moves (re-synthesis + lossless recode).
3. **bit-allocator — ACTIVE:** #54 (the waterfiller) consumes the boundary-solver per-component marginals;
   on the frontier base they're under-water (empty), on the lever-B base they're fundable (row C/#54).
4. **cathedral-autopilot — ACTIVE:** the recovery plan §2 IS the dispatch order (R1+R2+R3 + lever-C smoke).
5. **continual-learning — ACTIVE:** this ledger reseeds the planner that (a) frozen-bytes is exhausted,
   (b) lossless-recode is OPEN (the one orthogonal axis), (c) the live class shift is lever B blocked on
   frame1 dual-fidelity = lever C, (d) NO premature kills exist.
6. **probe-disambiguator — RESOLVED:** "is anything prematurely killed needing un-kill?" → NO (§3).
   "what is the single live blocker on the offensive frontier?" → #57 frame1-dual-fidelity = lever C.

## 5. Cross-references
`GOAL_standing_v3_20260610.md` · `MASTER_ROADMAP_post_exhaustion_map_20260610.md` (RANK 1-6) ·
`orphan_harvest_recovery_ledger_20260610.md` (R1-R5 + §3 exclusions) ·
`dedup_consolidation_audit_20260610.md` + `evaluator_inverse_orphan_inventory_20260609.md` (CONSUMED map) ·
`lever_b_score_native_argmax_smoke_verdict_20260610.md` (lever B CONFIRMED) ·
`score_native_pose_carrier_20260610T125000Z.md` (#57 the live blocker) ·
`score_native_first_candidate_20260610T112433Z.md` (#56) ·
`closed_spec_boundary_solver_v1_20260610T105830Z.md` + `boundary_math_seg_core_20260610T101618Z.md` (lever D gate) ·
`afsr1_smoke_verdict_20260610.md` · `lever_g_engineered_correction_smoke_20260610T095654Z.md` ·
`t1_s12_lossless_stack_verdict_20260610.md` · `cuda_pairing_recoded_r3_verdict_20260610.md` (D1 bank) ·
`cuda_axis_frontier_eval_verdict_20260610.md` (E6) ·
`lossy_coarsening_T0312_retired_config_do_not_redispatch_20260508_claude.md` (E2) ·
`frontier_{decoder,latent,seg_repair}_*_verdict_20260610.md` · `snerv_branch_b_round2_verdict_20260610.md` (E5).
