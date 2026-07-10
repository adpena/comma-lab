# NEGATIVE-FINDINGS REGISTER — Auditor B (memories + ledgers + canonical-equation negatives)

Task #390 (operator 2026-07-09 "audit all negative findings"). Corpus: `~/.claude/projects/-Users-adpena-Projects-pact/memory/*.md` + `src/tac/canonical_equations/` + `.omx/state/canonical_equations_registry.jsonl` + `.omx/state/lane_registry.json` + `.omx/state/deferral_ledger.md` + `.omx/state/lever_activation_ledger.jsonl`. Auditor A owns the DAG/`.omx/research` verdict memos + t5_crucible seal — no overlap. **REGISTER ONLY — no edits to any memory/ledger (HISTORICAL_PROVENANCE append-only).** Pointer **0.19110 UNMOVED**.

STORES CONSULTED: deferral_ledger.md (D1-D20) · lever_activation_ledger.jsonl · lane_registry.json (1788 lanes, 414 neg-tagged) · canonical_equations_registry.jsonl (negative anchors) · 15 method-negative memory files (batch-read) · CLAUDE.md §verdict-scope-ladder + P9/P10 (design_philosophies_eightfold) + operating_manual + verdict_scope_ladder memo.

Scope ladder applied: **INSTANCE < FORMULATION < FAMILY < PARADIGM** (narrowest the measurement supports). P9 proxy-suspicion: CLEAN / SUSPECT / KNOWN-TAINTED (MPS-derived, <n600 subset, frozen-instrument, borrowed-number → SUSPECT-or-worse **by rule**). P10: what geometric constraint the finding carves, or "instrument failure — carves nothing." Reactivation flag HIGH = (mis-scoped OR proxy-suspect) AND on a **live** v7.5.2/v8/frontier path.

---

## A. LIVE-PATH METHOD NEGATIVES (post-2026-06 witness/level-set/v8 pivot) — the reactivation-relevant bucket

| # | Finding (source) | As-treated scope | Re-grade + MIS-SCOPED? | P9 proxy | P10 constraint carved | Reformulation queued? | React |
|---|---|---|---|---|---|---|---|
| B1 | **Laguerre-OT head-offset mass-matching MEASURED-NEGATIVE** (`laguerre_ot_head_offset_v1`; lever ledger 2026-07-09 HeadOffsetSolver). OT/menon both HURT realized d_seg vs no-offset. | Row-written FORMULATION (memo itself says solver EXACT, only mass-matching objective refuted). | Correct FORMULATION. But the *measurement* is **n24-only** — n96/n600 KILLED by ~5-min background-task limit (probe not resumable-chunked → OWED). | **SUSPECT** — n24 << n600; explicitly non-promotable, verdict on 24 pairs. | Cell-mass-match to RAW GT area over-inflates rare Lane cell → SegNet penalises over-prediction. Carves: target masses must be **flip-weighted**, not area-matched. | YES — "flip-weighted target masses" in duty-to-measure queue. | **HIGH** |
| B2 | **msal_uni texture-multiplier INERT** (L76; `msal_uni_texture_proxy_inert...20260703`; LEVER-4). Texture proxy at CHANCE vs through-R reachability; whole alignment is `exp(-margin/τ)`. | INSTANCE→FORMULATION (a secondary multiplier). | Correct — and it is **not a dead lever, it is a redirect**: verdict = BUILD exact through-R `S_R = |∂(Σ w·margin)/∂x|` (θ-independent, cacheable). | SUSPECT (n96 advisory) but stable n6→n96; the redirect target is exact-through-R. | Texture is RGB-image-space, orthogonal to argmax reachability; S_R concentrates on the fragile small-margin band = where d_seg debt lives. | YES — build S_R + byte-closed A/B as a #205 arm (OWED, unbuilt). | **HIGH** |
| B3 | **DashComb corrector render-side NET-NEGATIVE** (lever ledger 2026-07-07; `dash_erasure_homogenization_v1`). Comb removes 86% of solid-band dash-gap FP but render-composite +0.0038 d_seg. | INSTANCE (render-side post-hoc form). | Mechanism CONFIRMED; the render-side FORM is what failed. Explicit reformulation: "corrector must be IN-TRAINING." | SUSPECT (macOS-CPU advisory) — mechanism-level on frozen ckpt. | Comb matches per-pair fitted gate (~186 floats); post-hoc = OOD to overfit render. | YES — in-training arm `--lane-band-dash-comb` **STILL NEVER FIRED** (duty-to-measure). | **HIGH** |
| B4 | **Render-side post-hoc DEAD — witness overfits its render** (`project_render_side_posthoc_dead...20260701`). ALL render bolt-ons (AA, analytic-lane) RAISE d_seg on frozen witness. | Row-written INSTANCE (frozen l7-best witness). | Correct — an **instrument/overfit** finding, not a lever kill. Verdict = #221 FINE-TUNE not from-scratch; AA belongs in the REAL-geometry TRAINED-IN path. | SUSPECT/instrument — the frozen witness overfit its exact pipeline → carves nothing about the levers, carves the training discipline. | Post-hoc render change = OOD; AA helps undersampled-REAL, hurts oversampled-SYNTHETIC (same principle, content-dependent). | YES — levers TRAINED-IN jointly (render+loss active), never bolted on. | MED |
| B5 | **Lane ground-frame ξ-transport NO-GO** (`lane_groundframe_xi_transport_no_collapse_v1`; 2026-07-09). ξ-advected innovation ≥ identity delta for every predictor; lane ground-frame coder NO-GO. | FORMULATION (explicit, n600). | Correct — a **chart-selection law**, not a ξ kill. ξ stays decisive for POSE + image-frame charts (horizon 14.6× win). | CLEAN (n600, both accounting modes). | Ground-canonicalized chart already quotients ego DOF → ξ only adds error. Carves: ξ candidate ⇔ chart retains removable ego structure. | N/A (the law tells you WHERE ξ applies; horizon is the live win). | LOW (correctly closed) |
| B6 | **Depth-warp / stratified-parallax / true-depth-flow pose REFUTED** (registry: `morse_smale_stratified_parallax_dpose`, true-depth-flow anchors, 2026-07-08/09). Piecewise-geometric flow RAISES d_pose; true mono-depth doesn't beat plane-H on real luma. | FORMULATION (each anchor explicitly formulation-scoped). | Correct. Wall = **PHOTOMETRIC APPEARANCE**, not flow-model crudeness. | CLEAN (n600 per-pair). | Off-plane finite-depth mass ~0.5%; corr(d_pose,\|t\|) NEGATIVE → addressable depth fraction tiny. | YES — cure = **JOINT pose-descent RUN (#238)** (render co-adapts); Option-A stored-depth NOT refuted. | **HIGH** (pose = THE v7.5 blocker, 4.35 of S 17.4) |
| B7 | **Quadratic-head SUBSET-solve gap** (L77; `quadratic_head_chart_subset_solve_gap_v1`; #341). K=8 subset overfits (+5.1% net) → subset tool NO-GO. | INSTANCE (subset K=8). | MIS-SCOPE RISK if read as "head-solve dead" — the full-P in-trainer GPU solve is the LIVE path (chart CONFIRMED, LM ρ 0.847/0.868). | CLEAN. | Subset overfits the sampled pairs; only full-P solve generalizes. | YES — full-P IN-TRAINER GPU solve (~11 min/CG-iter @17×); expensive but live. | MED |
| B8 | **Lever-D flicker residual NO-GO** (L67; `leverd_flicker_residual_reactivation_economics_20260703`; #280). | FORMULATION + has a **reactivation-economics** equation. | Re-grade the economics at the v7.5.2 operating point: near goal, apply RELATIVE-not-absolute significance — a "small" ΔS may be 13-27% of remaining gap. | CLEAN (#205 CE-floor context, n600). | CE-residual = temporal flicker (44% of spikes = Lane). | YES — economics gate; re-price vs remaining-gap-to-target. | MED |
| B9 | **l235 lever stack "overfit"** (`feedback_l235_levers_break_dseg_plateau...20260615`). soft_cosine fast-cool + margin-τ hit d_seg 0.00247 (BELOW plateau) then overfit 37h w/o FiLM-v2. | Written as overfit-negative. | MIS-SCOPED — the **headline is the GO signal** (lever BROKE the d_seg plateau); overfit = FiLM-v2 absence, not lever failure. On ABANDONED base_ch=20 torch basin → RE-MEASURE on v7.5.2 vehicle. | SUSPECT (contest-CPU advisory; abandoned-vehicle basin per L18). | soft_cosine fast-cool + margin-tau bends d_seg below the capacity plateau. | Implicit — re-run the lever stack WITH FiLM-v2 on the live vehicle. | MED |
| B10 | **TTD/FTT NOT-A-LEVER** (L55 papers-checked; `pose_in_training_lever_survey_verdict`). d=2 SVD, no win; axis-vs-directional. | FAMILY (survey verdict). | Correct as surveyed; grain owed → #341/#342 (directional basis, not axis). | CLEAN (paper-checked + survey). | Test-time-descent doesn't add a d_seg DOF here; directional Fourier does. | Partial — directional-basis grain #341/#342. | LOW |

---

## B. ANCESTOR / CONTEST-ERA SUBSTRATE NEGATIVES — PARADIGM/vehicle-superseded (L18); mechanism-lessons only

Per L18 + `ancestor_vehicle_findings_are_lessons_not_transferable_hnerv_pr95_abandoned`: these were measured on the ABANDONED HNeRV/PR95-shrunk substrate (PR101 INT8 symbols, apogee_int4 decoder quant, masks.mkv, full-frame codec). **Reactivation LOW** because the substrate they falsify no longer exists in the pipeline — but several MECHANISM families have already been reactivated on the new vehicle (noted).

| # | Finding | As-treated | Re-grade | React | Note |
|---|---|---|---|---|---|
| B11 | **rel_err² Lagrangian FALSIFIED as score predictor** (3 [contest-CUDA] anchors, 2026-05-08) | FAMILY | CONFIRMED-FORWARD, not reactivation: witness now measures exact d_seg through-R — the discipline this negative *demanded* is the current operating mode. | N/A | Carves: optimize the exact argmax term, never a reconstruction proxy. Still binding. |
| B12 | **Balle ScaleHyperprior FALSIFIED on PR101 symbols** (rel_err 0.98-0.99, no 2D locality; 2026-05-07) | FORMULATION (hyperprior on near-iid symbols) — as-treated read as "hyperprior class exhausted" | MIS-SCOPE RISK. Live SISTER **WeightEntropyPenalty** (Balle rate-in-loss on weights) ALREADY reactivated + MEASURED **−19.6% archive bytes** (ledger). Hyperprior on a *structured* witness `code` table is UN-tested. | MED | Before assuming the witness latent `code` table is hyperprior-dead, run a 2D-locality probe (D18 truncate A/B is adjacent). |
| B13 | **Markov-1 AAC dominated by brotli** on PR101 symbols (+21KB; 2026-05-07) | INSTANCE (PR101 symbols) | Substrate gone. Live sister **LaneBandResCoder** entropy-stage MEASURED **−25.6%** on witness lane payload → entropy-coding family ALIVE on new vehicle. | LOW | The negative carves: adaptive small-sample cost dominates on near-iid; witness lane payload is NOT near-iid. |
| B14 | **UNIWARD standalone KILL** (5/5 council, 2026-04-29; `uniward_standalone_no_op_on_bitstream...v1`) | FORMULATION (no-op on bitstream w/o SLI1 decoder) | Descendant is the msal_uni lever (B2). UNIWARD-as-cost = witness margin/fragility weight, already live. | LOW | Mechanism absorbed; the standalone-encoder form is genuinely dominated. |
| B15 | **AMRC lossless mask codec NEGATIVE** (1.03MB vs AV1 421KB) | INSTANCE | masks.mkv paradigm gone. | LOW | — |
| B16 | **apogee_int4 / naive-PTQ-int4 FALSIFIED** (700× pose collapse; `naive_ptq_int4_block128_falsified_v1`) | FORMULATION (naive PTQ) — CLAUDE.md flags premature-int4-KILL precedent | Substrate (HNeRV decoder) gone; QAT/LSQ reformulation never exhausted on it, but moot post-pivot. | LOW | Lesson: low-bit needs QAT/per-channel; transfers if witness ever quantizes weights. |
| B17 | **int7 packing pareto-dominated by int8** (`non_byte_aligned_int7...v1`); **AC dominated by brotli-q11 small-alphabet** (`ac_dominated_by_brotli_q11...v1`) | INSTANCE | Codec details; LaneBandResCoder already picks best-of-three POST-brotli (consistent — no conflict). | LOW | — |
| B18 | **Preprocessing dead-end for PoseNet** (blur/chroma all +90-105% pose) | FAMILY (preprocessing on full-frame codec) | Witness doesn't preprocess. But the FACT "PoseNet reads whole frame incl chroma" transfers — and CHROMA is now a *d_seg lever* in the capstone. | LOW | Carves: any pixel modification hurts pose → witness composes pose via stored-sidecar, not pixel edits. |
| B19 | **Hybrid CG at inflate DEAD** (scorer weights would count 90MB) | PARADIGM (contest rule) | STILL BINDING — not reactivation. | N/A | Correct; no scorers at inflate time. |
| B20 | **HStack codec dominated by analytical** (`path_b_step4_hstack...dominated`); **hybrid-CG dead**; **AMRC/preproc/rel_err²** family | FORMULATION/INSTANCE | Composition-era details, substrate-superseded. | LOW | — |
| B21 | **414 lane_registry lanes** tagged research_only/deferred/killed/falsified (track1_*, lane_t*_balle/vqvae/siren/wavelet/rust-packet/self-compression/...) | Bulk | Overwhelmingly the ABANDONED HNeRV/PR95/substrate-composition paradigm (no witness/level-set lanes among them). PARADIGM-superseded en masse. | LOW-bulk | Not itemized — carrying value is negligible on the new vehicle; a few entropy/codec mechanisms already re-expressed as witness levers (B12/B13). |

---

## C. DEFERRAL LEDGER (D1-D20) — trigger-liveness re-grade

Key structural finding: **most run-1-stop-gated deferrals (D1/D2/D9/D15/D17/D18/D19) are still correctly ARMED but their gating run has produced NO v7 FINAL CKPT** (D18 note 2026-07-09: run `dry_start`, `best=NONE`, no `levelset_best.json`). Given CURRENT-STATE says v7.5.2 GATE-GREEN + v8 at P4, the "run-1 GOVERNED STOP" trigger may need **re-pointing to the live v7.5.2 run** so these don't silently orphan.

| Row | Trigger still right? | Note |
|---|---|---|
| D1/D2/D9 (GPU-verdict probe, S6-R1 knee, verdict promotion) | ARMED — gated on run-1 stop | Verify the gating run == the live v7.5.2 run, not the dead dry_start. |
| D5/D6/D15/D16/D17/D19 (fp16 cache, async-reclaim, micro-batch, Metal kernels, safe-compile, speed bundle) | QUEUED-W-TRIGGER — need v7 baseline comparator | Same re-point caveat; D17 safe-compile is v2-WIRED (GPU cross-process re-cert deferred to run-1 stop). |
| D7 (#314 pose-carrier inheritance bug) | CHECK-AT-COMPILE | Bug persists for OTHER families even though v7 is safe; keep live. |
| **D10 (marimo contest #347)** | **DEADLINE 2026-07-09 11:59PM PST = TODAY** | Operator go/no-go never given; silence past tonight = lapsed-by-default. FLAG for operator. |
| D18 (latent-table truncate byte-close A/B) | ARMED, blocked on NO FINAL CKPT | Machinery exists (`witness_code_pca_byteclose.py` + k90 sensor); only-missing-wire is auto-feed k90→`--ks`. Fires at v7 stop. |
| D3/D4/D20 | CLOSED (resume registry / serializer / non-gate controllers) | Correctly closed; no action. |

---

## D. EXCLUDED CLASS — process/operational negatives (NOT method verdicts)

~34 memory files are DISCIPLINE, not scientific negatives: bash-harness kills, remote/bootstrap traps, rsync/pipefail/zip/tarball, dead-flag wiring, sandboxed-bash daemon death, subagent stall/false-death, harness SIGURG, log-silence, **spike-guard median-freeze confound** (an instrument-failure, carves nothing about a method), kaggle/vastai ops. Correctly OUT OF SCOPE for reactivation; they carve process constraints, not geometry. No re-grade needed.

---

## TOP-10 REACTIVATION CANDIDATES (why + cheapest decisive re-test)

1. **B1 Laguerre-OT head-offset, flip-weighted masses** — SUSPECT (n24-only verdict, n96/n600 killed by 5-min limit), FORMULATION-scoped, reformulation already queued, live v8 head path. *Cheapest:* resumable-chunked **n600** run of `HeadOffsetSolver` with **flip-weighted** (not area-matched) target masses; the OT solver is already EXACT.
2. **B6 Depth-warp pose → JOINT pose-descent RUN (#238)** — pose is THE v7.5 blocker (4.35 of S 17.4); every stratified/true-depth REFUTAL explicitly names joint pose-descent as the not-refuted cure. *Cheapest:* the #238 joint run (render co-adapts) — already the live thread; ensure it launches.
3. **B2 msal_uni → exact through-R S_R weight** — verdict is a REDIRECT (build S_R), not a kill; S_R is θ-independent + cacheable. *Cheapest:* build the S_R cache + byte-closed d_seg A/B as a #205 arm (OWED, unbuilt).
4. **B3 DashComb in-training arm** — `--lane-band-dash-comb` has NEVER FIRED; render-side was net-negative but the reformulation ("corrector IN-TRAINING") is explicit; live dash-repair. *Cheapest:* fire the in-training flag as a #205 arm.
5. **B12 Balle rate-in-loss on the MLX witness (WeightEntropyPenalty)** — already MEASURED **−19.6% bytes** on the torch vehicle but NOT DSL-holdable (no MLX flag) and net-S n600 A/B OWED. *Cheapest:* port the flag to the MLX levelset trainer OR run the counted-weights arm on the torch vehicle → fold as a `Lever` factory.
6. **B7 Quadratic-head full-P in-trainer GPU solve** — subset NO-GO risks being mis-read as "head-solve dead"; chart CONFIRMED. *Cheapest:* one full-P solve at a converged ckpt (~11 min/CG-iter @17×) to price the real (non-overfit) gain.
7. **B9 l235 lever stack (soft_cosine fast-cool + margin-τ) on v7.5.2** — mis-scoped "overfit" hides the GO signal (broke the d_seg plateau); measured on the abandoned base_ch=20 basin. *Cheapest:* re-run the stack WITH FiLM-v2 decoupling on the live vehicle (the win-vs-overfit A/B).
8. **B8 Lever-D flicker residual — re-price near goal** — NO-GO #280 has a reactivation-economics equation; RELATIVE-not-absolute significance says a "small" ΔS may be 13-27% of the remaining gap at v7.5.2. *Cheapest:* re-evaluate the economics gate at the current operating point (no new run — an arithmetic re-grade against remaining-gap-to-target).
9. **B12′ Hyperprior on the witness latent `code` table** — the FALSIFIED verdict was PR101 near-iid symbols (no 2D locality); the structured witness `code` table is UN-tested. *Cheapest:* a $0 2D-locality probe on the `code` table before assuming hyperprior-dead (adjacent to the D18 truncate A/B).
10. **D10 marimo/molab contest #347** — DEADLINE **tonight**; operator go/no-go never given → lapsing-by-default. *Cheapest:* surface to operator for an explicit go/no-go before 11:59PM PST.

---

## COUNTS

- Corpus scanned: **~80 negatively-named memory files** + **414 neg-tagged lane_registry lanes** + **~20 canonical-equation negative anchors** + **20 deferral-ledger rows** + lever_activation_ledger events.
- **Method/science negatives registered: 21** (B1-B21). Of these:
  - **Live-path (Bucket A): 10** — 4 HIGH, 3 MED, 3 LOW reactivation.
  - **Ancestor/contest-era (Bucket B): 11** — mostly PARADIGM/vehicle-superseded (LOW), **1 MED** mis-scope (B12 Balle, live sister already reactivated).
- **Mis-scoped-or-suspect flags: 6** (B1, B2 redirect, B7, B9, B12, plus B3/B6 formulation-with-live-reformulation).
- **Proxy-SUSPECT (P9): 5** (B1 n24-only, B2/B3/B4 macOS-CPU-advisory frozen-ckpt, B9 abandoned-vehicle). No MPS-derived kills found in this corpus slice (those live in Auditor A's DAG memos).
- **Deferral rows: 20** — 3 CLOSED, 16 ARMED/QUEUED (7 gated on a run-1 stop whose gating-run may need re-pointing to v7.5.2), **1 DEADLINE-TODAY (D10)**.
- **Excluded process/operational: ~34** (discipline, not method — carve process not geometry).
- Pointer **0.19110 UNMOVED** throughout. All Bucket-A verdicts `[macOS-CPU advisory · NON-PROMOTABLE]` or n600-CLEAN as noted.

Follow-ups act on this register (append-only; no memory/ledger edited). Sisters: Auditor A (DAG/research verdict memos), t5_crucible seal agent.
