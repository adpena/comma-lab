# RESURRECTION AUDIT — Historical KILL/DEFER/LESSER-ROLE Substrate Re-evaluation
## Triggered by NSCS06 v7 empirical proof (105.15 → 58.89 in ONE iteration via 4-of-7 cargo-cult unwinds)
## Date: 2026-05-16
## Council tier: T2 (provisional; advances to T3 batched per task #742)
## Operator-prompt anchor: 2026-05-16 resurrection-audit mandate (sister-disjoint to council-hierarchy-v2 landing `a50a92694ac22c9f5`)

---

## Operating-within assumption-statement (per Catalog #292)

The assumption I am operating within for this audit: *"Audit-only enumeration + Tier-classification + reactivation-ranking of historical KILL/DEFER/RESEARCH-ONLY substrates is the lowest-cost intervention to surface re-evaluation candidates without re-opening any lanes — the T3 batched council per task #742 adjudicates."*

HARD-EARNED basis: CLAUDE.md "Forbidden premature KILL without research exhaustion" + "Apples-to-apples evidence discipline" + Catalog #229 premise-verification protocol + the NSCS06 v7 empirical proof (553× outside-band v6 → 58.89 via 4-of-7 cargo-cult unwinds) that DESIGN-time static review surfaces unwind candidates without runtime cost. The Assumption-Adversary seat would challenge: *"Is audit-only sufficient, or does the operator need actionable resurrection NOW given the NSCS06 v7 ROI?"* — Answer: audit-only IS correct per CLAUDE.md "Design decisions — non-negotiable" because reopening a lane is a council-grade tradeoff requiring sextet pact + 30-day deferred-substrate retrospective per mission-alignment directive `feedback_council_apparatus_in_service_of_innovation_rigor_optimization_score_lowering_20260516.md`.

---

## Executive summary

- **N total substrates enumerated:** 31 (deduplicated by lane_id/substrate name across 4 sources: lane registry 144 substrate-tagged candidates, 24 kill memos in Claude memory, 7 retired/falsified memos in `.omx/research/`, NSCS06 v7 anchor + recent v6/v7 memos)
- **Tier 1 (CLEAR CARGO-CULT KILLS — immediate re-evaluation candidate):** 9
- **Tier 2 (PARTIALLY HARD-EARNED — warrant re-evaluation, lower priority):** 12
- **Tier 3 (GENUINE FALSIFICATIONS — keep killed with hard-earned citation):** 10
- **Top 5 priority re-evaluation candidates** (ranked by expected-score-contribution × reactivation-feasibility / cost):
  1. **Lane 17 IMP** — KILL ALREADY WITHDRAWN 2026-04-30 (8-of-10 council vote); STILL UNTRAINED because the canonical re-run via `train_distill` was never executed; 27.8× asymmetric regression (PoseNet 34.8× vs SegNet 1.25×) is empirical evidence of fixable stub-bug not architectural ceiling; cost ~$5-15 Modal A100 100ep
  2. **PR101 CompressAI Ballé hyperprior FULL** — DEFERRED-pending-real-substrate; falsification was on PR101's near-iid INT8 quantized symbols (no 2D locality); cargo-cult was treating PR101 symbol bytes as image; reactivation path: apply to NSCS06-v7 chroma residuals OR ATW codec OR NSCS03 latent stream; cost ~$5 paired smoke
  3. **PR106 Lanes #05 (UNIWARD-delta) + #06 (grayscale-LUT)** — FALSIFIED-as-non-applicable per "PR106 has no mask channel"; cargo-cult was the LANE-DESIGN-VS-PR106-ARCHITECTURE mismatch, NOT the technique class; reactivation path: REFORMULATE per current paradigm ("UNIWARD-delta on PR106 latent stream" / "grayscale-LUT on PR106 latent codebook"); cost $0 design + $5 paired
  4. **Lane MM v2 (encoder-only grayscale-LUT 2.63 [contest-CPU advisory])** — FALSIFIED with axis-tag-cleanup requirement; never received [contest-CUDA] confirm; cargo-cult was the 3ch-trained-renderer + hard-argmax-mask bolt mismatch (NOT the soft-grayscale class); reactivation path: pair with SegMap-trained-from-scratch architecture per Selfcomp's 0.38 anchor; cost ~$5 paired smoke
  5. **Lane SO + Lane SA v3 (task #214/#215 KILLED)** — historical kill memos referenced but not surfaced in current memory dir; presumed pre-CLAUDE.md "KILL is LAST RESORT" (2026-05-05) kills with no 3-section structural compliance; cost $0 audit + reactivation depends on locating original kill verdict

- **Bottom-line operator-facing recommendation:** Pursue Tier 1 re-evaluations IN PARALLEL per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable. Estimated cost $25-75 across top 5 (mostly $0 design-time + paired smoke). Predicted prevention value: re-opening even ONE of these at sub-A1 would close the 0.196-0.199 plateau gap structurally. The NSCS06 v7 anchor (4-of-7 cargo-cult unwinds → 553× → ~52% score reduction in ONE iteration) is the empirical proof that historical kills systematically over-killed by skipping the today's 5 retroactive lenses.

---

## Today's 5 retroactive lenses (applied per substrate below)

1. **Chroma-axis preservation** (NSCS06 lesson; sister Catalog #297): did the kill assume Y=R=G=B / SegNet-blind / signal-axis-destruction without per-class chroma anchor?
2. **Dykstra-feasibility** (D3 helper landed today; sister Catalog #296): was the kill's predicted band an additive sum without Dykstra-intersection check?
3. **9-dim checklist** (operator standing directive `feedback_9_dimension_success_checklist_*`): which of 9 dimensions failed at kill time? Are the missing dimensions now addressable?
4. **Canonical-vs-unique** (PR95 meta-level lesson `feedback_pr95_lesson_now_at_meta_level_*`): was the kill caused by force-canonical-without-evaluation-of-suppression?
5. **7-cargo-cult NSCS06 inventory**: any of (closed-form-CDF / Y=R=G=B / spatial-independent / 2-of-6-pose-warp / NO-neural / symposium-#4-band / PR#56-generalizes) present?

---

## Tier 1 — Clear cargo-cult kills (immediate re-evaluation)

### 1.1 Lane 17 IMP (88K-param Iterative Magnitude Pruning) — KILL ALREADY WITHDRAWN; NEVER RE-RUN

1. **Substrate ID:** `lane_17_imp` / `lane_j_imp` — IMP cycle 0 cycle-fine-tune 88K-param sparse renderer
2. **Current state:** KILL VERDICT WITHDRAWN 2026-04-30 ~22:55 UTC by 8-of-10 council vote (project_lane_17_imp_killed_cycle_0_198_regression_20260430.md), but never properly re-run via the canonical `train_distill` swap; lives as ZOMBIE-WITHDRAWN ever since
3. **Killer-assumption verbatim:** *"1.98 [contest-CUDA] score reflects 88K-param IMP architectural ceiling"* (initial kill verdict)
4. **HARD-EARNED vs CARGO-CULTED:** **CARGO-CULTED** — the killer-assumption mistook a 3.5-second stub-loop "fine-tune" for converged 200-epoch fine-tune; `stats.json: epochs=200, elapsed_sec=3.47` was the smoking gun; the comment promise of "in-script lightweight loop; deploy script swaps in train_distill" was a comment-only contract (CLAUDE.md FORBIDDEN_PATTERNS) that was never satisfied
5. **Today's 5 lenses:**
   - Chroma preservation: N/A (IMP is weight-pruning, not signal-axis)
   - Dykstra-feasibility: N/A (no predicted band; was a measurement-bug kill not a band-failure)
   - 9-dim: Dimension 4 RIGOR failed (no premise-verification of the stats.json internal consistency); Dimension 9 OPTIMAL SCORE never measured (the 1.98 was a stub artifact)
   - Canonical-vs-unique: PARTIAL — IMP is a CLASS-SHIFT (sparse + structured pruning); the kill was NOT canonicalization-induced
   - 7-cargo-cult: Symposium-#4-band-prediction cargo-cult (initial kill verdict treated 1.98 as a real prediction when it was a stub artifact); NO Y=R=G=B / spatial-independent / 2-of-6-pose-warp / NO-neural risk
6. **TIER classification:** **Tier 1** — clear cargo-cult kill; PoseNet 34.8× vs SegNet 1.25× asymmetry is empirical evidence of fixable stub-bug (Frankle 2019 lottery-ticket subnetworks DO recover with proper fine-tune) not architectural ceiling
7. **Reactivation criteria:** Re-run IMP cycle 0 with PROPER `train_distill` fine-tune (10-30 min on L40S, NOT 3.5s stub); compare auth-eval against Lane G v3 anchor; if Δ score still > +0.5 per cycle, THEN structural ceiling is confirmed
8. **Cost / priority:** ~$5-15 Modal A100 100ep — **HIGHEST PRIORITY** because (a) kill is already withdrawn, (b) reactivation criteria are explicit and empirically-grounded per Catalog #229, (c) compositional with NSCS01/NSCS02 (sparse pruning orthogonal to architecture-class)

### 1.2 Lane STC clean-source — FALSIFICATION-AS-SCIENTIFIC-CLAIM WITHDRAWN; never re-run on CUDA

1. **Substrate ID:** `lane_stc_clean_source` — Syndrome-Trellis Coding on clean SegNet argmax masks
2. **Current state:** Original FALSIFIED verdict (~21MB stream, 50.5× regression) was MPS-PROXY — explicitly NOT valid per CLAUDE.md "MPS auth eval is NOISE"; STATUS REVISION 2026-04-29 PM re-tagged to UNDETERMINED pending Modal T4 CUDA re-run (~$0.20, ~10 min); never executed
3. **Killer-assumption verbatim:** *"Clean-source STC produces 21MB stream because boundary-fraction is too high"*
4. **HARD-EARNED vs CARGO-CULTED:** **CARGO-CULTED** at evidence axis — MPS-argmax bytes ≠ CUDA-argmax bytes (23× PoseNet drift per CLAUDE.md MPS rule); the codec has a REAL BUG ("one-majority-plus-exceptions" stores 109M exceptions vs 11.8M boundaries on multi-region masks) but the 21MB measurement specifically applies to MPS-argmax masks, not the contest scorer
5. **Today's 5 lenses:**
   - Chroma preservation: N/A (STC operates on argmax masks)
   - Dykstra-feasibility: not applied (no predicted band)
   - 9-dim: Dimension 4 RIGOR failed (MPS-PROXY treated as `[contest-CUDA]` decision-grade); Dimension 9 NEVER MEASURED on contest substrate
   - Canonical-vs-unique: Filler STC IS substrate-class-shift (steganography-derived codec); kill was MPS-evidence-only
   - 7-cargo-cult: NONE present in design; ALL in evidence collection
6. **TIER classification:** **Tier 1** — kill explicitly tagged UNDETERMINED in source memo; $0.20 Modal T4 CUDA run resolves
7. **Reactivation criteria:** Modal T4 CUDA re-run on clean SegNet argmax masks; if bytes < 421KB AV1 baseline at ≤1% reconstruction error, lane is RESURRECTED; if still 18-50× over AV1, then structural redesign (AV1+STC-residual / temporal-predictor / scanline-RLE / lossy-STC per source memo)
8. **Cost / priority:** ~$0.20 — **HIGH PRIORITY** because cheapest possible re-eval + already documented as council #1 hope

### 1.3 PR101 CompressAI Ballé hyperprior FULL — substrate-mismatch falsification (not technique-class kill)

1. **Substrate ID:** `lane_pr101_compressai_balle_full` — Full CompressAI ScaleHyperprior + MeanScaleHyperprior on PR101 INT8 symbols
2. **Current state:** DEFERRED-pending-research-with-far-stronger-evidence 2026-05-07 (rel_err plateaus 0.98-0.99 across all 8 N/M configs)
3. **Killer-assumption verbatim:** *"Ballé ScaleHyperprior pipeline cannot reconstruct PR101's near-iid quantized symbol distribution because the substrate has no exploitable 2D locality"*
4. **HARD-EARNED vs CARGO-CULTED:** **PARTIALLY HARD-EARNED + CARGO-CULTED**. HARD-EARNED-CORE: the empirical exhaustion of Ballé on PR101-symbols-reshaped-as-1×1×448×512 pseudo-image IS valid as a substrate-mismatch finding. CARGO-CULTED-SHELL: treating "PR101 INT8 symbols" as the target — Ballé hyperprior is canonical for IMAGE/VIDEO domains with spatial locality; applying it to a 1D symbol stream is a substrate misuse
5. **Today's 5 lenses:**
   - Chroma preservation: N/A (PR101 symbols are quantized weights, not images)
   - Dykstra-feasibility: HARD-EARNED (the rel_err plateau IS the empirical feasibility boundary)
   - 9-dim: Dimension 1 UNIQUENESS — the kill applied a CLASS-SHIFT technique (Ballé) to the WRONG class (PR101 symbol stream); reactivation requires applying to the RIGHT substrate
   - Canonical-vs-unique: STRONG cargo-cult — Ballé is canonical for spatially-correlated data; PR101 INT8 symbols are intentionally decorrelated; the canonical was force-applied where it suppresses
   - 7-cargo-cult: NONE in v6-NSCS06 inventory present; this is a different cargo-cult class (canonical-helper-misapplication)
6. **TIER classification:** **Tier 1** — falsification VALID for PR101 substrate; INVALID as a class-kill of Ballé hyperprior
7. **Reactivation criteria:** Apply Ballé hyperprior to a substrate WITH 2D spatial locality: (a) NSCS06-v7 chroma residuals (UV channels after Y=R=G=B unwind); (b) ATW codec latent stream (latent grids with spatial structure); (c) NSCS03 latent stream (Ballé-2018 joint codec ALREADY landed and uses hyperprior — reactivation is OPERATIONAL); (d) PR106 latent stream
8. **Cost / priority:** $0 (NSCS03 already lands Ballé hyperprior; this audit's recommendation is to ENFORCE the substrate-mismatch finding does NOT spill over to NSCS03's frontier potential) + $5 paired smoke for ATW or NSCS06-v7 chroma residual application — **HIGH PRIORITY** because the falsification is being misread as class-kill

### 1.4 PR106 Lanes #05 + #06 (UNIWARD-delta + grayscale-LUT on PR106 mask channel)

1. **Substrate IDs:** revival_plan_05 (UNIWARD-delta) + revival_plan_06 (mask-grayscale-LUT)
2. **Current state:** FALSIFIED-AS-NON-APPLICABLE 2026-05-04 ("PR106 has no separate mask channel")
3. **Killer-assumption verbatim:** *"PR106 has a mask.mkv-style separate stream"* (the WRONG assumption in the lane DESIGN, not the technique)
4. **HARD-EARNED vs CARGO-CULTED:** **CARGO-CULTED** — the killer-assumption was a lane-design-vs-PR106-architecture mismatch (Quantizr-style mask.mkv presumed; PR106 is HNeRV with brotli-decoder + brotli-latents only). The TECHNIQUES (UNIWARD-delta + grayscale-LUT) are HARD-EARNED canonical primitives (Fridrich + Selfcomp respectively)
5. **Today's 5 lenses:**
   - Chroma preservation: N/A here
   - Dykstra-feasibility: kill skipped Dykstra (no feasibility check)
   - 9-dim: Dimension 1 UNIQUENESS — lanes designed within PR106's PRESUMED-Quantizr-paradigm envelope rather than PR106's actual HNeRV envelope; classic within-class refinement applied to wrong class
   - Canonical-vs-unique: Cargo-culted (presumed PR106 architecture)
   - 7-cargo-cult: PR#56-generalizes (assumed PR106 follows PR#56 paradigm; it doesn't) is one of the 7
6. **TIER classification:** **Tier 1** — clear cargo-cult lane-design; techniques are not falsified
7. **Reactivation criteria:** REFORMULATE per current PR106 architecture: (a) "UNIWARD-delta on PR106 LATENT STREAM" (the 15,849-byte brotli-latents per source memo IS analogous to mask.mkv as a sidechannel target); (b) "grayscale-LUT on PR106 LATENT CODEBOOK" (replace per-pair 28-dim continuous latent with discrete LUT-indexed); (c) reformulated lanes get fresh predicted bands derived from Dykstra-feasibility on the actual PR106 archive grammar
8. **Cost / priority:** $0 design rewrite + $5 paired smoke per reformulated lane — **HIGH PRIORITY** because both UNIWARD + grayscale-LUT are leaderboard-proven primitives (PR101/PR103 silver + PR#56 paradigm)

### 1.5 Lane MM v2 (encoder-only grayscale-LUT 2.63 [contest-CPU advisory])

1. **Substrate ID:** `lane_mm_v2` — encoder-only grayscale-LUT mask + Lane A renderer
2. **Current state:** FALSIFIED on `[contest-CPU advisory]` (Modal CPU eval, NOT contest-CUDA); directional verdict stands but exact magnitude (51× PoseNet ratio) needs CUDA confirm; never received [contest-CUDA] paired confirm; ~$0.50 Vast.ai 4090 outstanding
3. **Killer-assumption verbatim:** *"Hard-argmax grayscale-LUT preserves quality bolted onto 3ch-trained renderer"*
4. **HARD-EARNED vs CARGO-CULTED:** **PARTIALLY HARD-EARNED**. HARD-EARNED: the architectural mismatch (3ch-trained renderer + 1ch grayscale-LUT bolt) IS structural per Selfcomp's PR#56 paradigm; SegMap is trained-from-scratch with the LUT, not bolted-on. CARGO-CULTED: the SCORE MAGNITUDE (51× PoseNet ratio, 2.63 score) is on CPU-not-CUDA and could be 1.5-2× different on the contest substrate
5. **Today's 5 lenses:**
   - Chroma preservation: PARTIAL — grayscale-LUT IS chroma-destroying but the falsification was on renderer-architecture-mismatch axis, not chroma axis
   - Dykstra-feasibility: NOT performed
   - 9-dim: Dimension 4 RIGOR — `[contest-CPU advisory]` axis-tag-cleanup never landed
   - Canonical-vs-unique: HARD-EARNED-CORE for technique class (Selfcomp anchor 0.38) + CARGO-CULTED-SHELL for the bolt-on application
   - 7-cargo-cult: Symposium-#4-band-prediction cargo-cult (predicted [0.65, 0.85] vs actual 2.63 — 3× outside-band; same class as NSCS06 v6 553× outside-band)
6. **TIER classification:** **Tier 1** — the falsification is directionally correct but the SCORE MAGNITUDE evidence is not contest-CUDA; reactivation requires (a) $0.50 CUDA-axis confirm of the bolt-on falsification AND (b) pivot to the trained-from-scratch SegMap variant per Selfcomp anchor
7. **Reactivation criteria:** SegMap-trained-from-scratch with grayscale-LUT (NOT bolt-on); reactivate as new lane `lane_mm_v3_segmap_lut_trained_from_scratch`; if reaches sub-1.0 [contest-CUDA] then technique IS viable in correct architectural context
8. **Cost / priority:** $0.50 Vast.ai 4090 CUDA-axis cleanup OR $5-15 Modal A100 trained-from-scratch retry — **MEDIUM-HIGH PRIORITY** (Selfcomp 0.38 anchor is class-shift evidence)

### 1.6 Lane HM-S / WC-S / MAE-V / SAUG (Tasks #216 DEFERRED batch)

1. **Substrate IDs:** Lane HM-S (Hadamard-Mask-Sparse), Lane WC-S (Weight-Cluster-Sparse), Lane MAE-V (Masked-Autoencoder Vision pre-training), Lane SAUG (Steganographic Augmentation V2)
2. **Current state:** DEFERRED batch (task #216 per prompt); Lane SAUG-V2 launched via codex:rescue 2026-04-28 (errno 8 DNS sandbox bug; permanently rerouted to parent shell per `feedback_codex_sandbox_blocks_vastai_dns_20260428.md`); MAE-V referenced in alternative-paradigms research queue; HM-S/WC-S/SAUG never received empirical anchor on contest substrate
3. **Killer-assumption verbatim:** *"DEFER-pending-trainer-recipe-balance"* / *"DEFER-pending-cycle-1-postmortem"* (vague defers, NOT science-grounded falsifications)
4. **HARD-EARNED vs CARGO-CULTED:** **CARGO-CULTED** at evidence axis — the defers are OPERATIONAL (DNS bug, infrastructure not ready, trainer recipe imbalanced) not SCIENTIFIC. None received 3-section structural kill verdict per CLAUDE.md "KILL/FALSIFIED memory verdicts"
5. **Today's 5 lenses:**
   - Chroma preservation: not evaluated
   - Dykstra-feasibility: not evaluated
   - 9-dim: Dimension 4 RIGOR — no empirical receipt; Dimension 1 UNIQUENESS — MAE-V is class-shift (self-supervised pre-training), SAUG is class-shift (steganographic data augmentation)
   - Canonical-vs-unique: Not evaluated
   - 7-cargo-cult: NONE specific to these lanes
6. **TIER classification:** **Tier 1** (MAE-V + SAUG specifically) — these are class-shift candidates DEFERRED for operational reasons only; cargo-cult-of-discipline-blocking-mission per `feedback_council_apparatus_in_service_of_innovation_*` operator directive ("when procedural rigor blocks frontier innovation, the discipline yields")
7. **Reactivation criteria:** (a) MAE-V: pre-train with masked patches on `upstream/videos/0.mkv`; replace renderer init with MAE-pretrained encoder; auth-eval against Lane G v3 anchor; cost ~$15 Modal A100 (b) SAUG: data-augmentation-time steganographic embedding; lane scope already exists at `experiments/results/saug_*`; redispatch when Vast.ai balance allows OR migrate to Modal/Lightning per current dispatch policy; cost ~$10
8. **Cost / priority:** $10-25 total for MAE-V + SAUG re-eval — **MEDIUM-HIGH PRIORITY** (both are CLASS-SHIFT per the abandon-within-class directive)

### 1.7 Lane AL / FC / PA (Tasks #225 DEMOTED to lesser-role)

1. **Substrate IDs:** Lane AL (SGD-Optimized Soft Grayscale Mask), Lane FC (Foveation Codec), Lane PA (Pose-Augmented Renderer)
2. **Current state:** DEMOTED to lesser-role (task #225 per prompt); historical Selfcomp confirmed SGD-optimized soft grayscale IS the correct path per Lane MM v2 falsification (see 1.5); FC/PA never received 3-section structural KILL
3. **Killer-assumption verbatim:** *"Lane MM v2 falsification re-confirmed Lane AL is the correct path"* (Lane MM v2 source memo)
4. **HARD-EARNED vs CARGO-CULTED:** **PARTIALLY HARD-EARNED**. HARD-EARNED-CORE: Lane AL was re-validated by MM v2 falsification (as the correct soft-grayscale path). CARGO-CULTED-SHELL: DEMOTING Lane AL/FC/PA to lesser-role rather than promoting Lane AL to L2+ contradicts the very evidence that validated it
5. **Today's 5 lenses:**
   - Chroma preservation: Lane AL preserves grayscale chroma via soft SGD optimization (NSCS06-v7 lesson applies)
   - Dykstra-feasibility: not evaluated
   - 9-dim: Dimension 1 UNIQUENESS — Lane AL is class-shift (SGD-optimized continuous representation vs hard discrete LUT)
   - Canonical-vs-unique: Lane AL forks soft-grayscale paradigm (UNIQUE, not canonical)
   - 7-cargo-cult: NONE
6. **TIER classification:** **Tier 1 (Lane AL specifically)** — demotion contradicts the empirical re-validation; Lane FC + PA Tier 2 (less direct evidence)
7. **Reactivation criteria:** PROMOTE Lane AL to active dispatch queue; pair with Lane MM v3 (SegMap-trained-from-scratch with grayscale-LUT — see 1.5); auth-eval against Lane G v3 anchor; if reaches sub-1.0 [contest-CUDA], Selfcomp 0.38 paradigm transfer is empirically confirmed
8. **Cost / priority:** $5-15 Modal A100 — **MEDIUM PRIORITY**; composes with Lane MM v3 priority (1.5)

### 1.8 Lane apogee_int4 (DEFERRED-pending-QAT-research at 1.43 [contest-CUDA T4])

1. **Substrate ID:** `lane_apogee_int4` — Naive-PTQ INT4 quantization at 109,996-byte archive
2. **Current state:** Originally tagged FALSIFIED; downgraded to DEFERRED-pending-QAT-research per CLAUDE.md "KILL is LAST RESORT"; specific reactivation criteria documented (QAT, LSQ, per-channel scaling, smaller block sizes, outlier handling/clipping)
3. **Killer-assumption verbatim:** *"INT4 NAIVE-PTQ produces 1.4287 [contest-CUDA T4]"* (the falsified config)
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED-CORE** (single-config empirical receipt is real) + **CARGO-CULTED-SHELL** (treating one config's failure as a class-kill of INT4)
5. **Today's 5 lenses:**
   - Chroma preservation: N/A (quantization)
   - Dykstra-feasibility: HARD-EARNED — the rel_err² objective falsification at rms ≥ 0.04 IS the feasibility-cliff per `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_*`
   - 9-dim: Dimension 1 UNIQUENESS — INT4 IS class-shift (quantization tier); Quantizr (0.33 leader) USES INT4 FP4A → INT4 IS leaderboard-proven viable
   - Canonical-vs-unique: NAIVE-PTQ is canonical default; QAT/LSQ/per-channel are substrate-optimal alternatives
   - 7-cargo-cult: Symposium-#4-band-prediction cargo-cult (predicted sub-0.30, actual 1.43, 4.8× outside-band)
6. **TIER classification:** **Tier 1** — reactivation criteria EXPLICITLY documented + Quantizr 0.33 leader uses INT4 = empirical proof of viability
7. **Reactivation criteria:** QAT (quantization-aware training; Quantizr's winning recipe) → if sub-0.50 [contest-CUDA], INT4 family is RESURRECTED; if still > 1.0, then LSQ → per-channel → outlier clipping cascade
8. **Cost / priority:** $5-15 Modal A100 QAT smoke — **HIGH PRIORITY** (Quantizr 0.33 leader uses INT4; this is direct leaderboard pursuit)

### 1.9 Lane apogee_int7 (DEFERRED Pareto-dominated by int8 at current packer)

1. **Substrate ID:** `lane_apogee_int7` — INT7 quantization at 205,158-byte archive (9.3% LARGER than int8 187,731 due to non-byte-aligned packing)
2. **Current state:** DEFERRED-pending-research-with-byte-aligned-int7-packer; CLAUDE.md kill-as-last-resort compliant (3-section structure present with reactivation criteria); 7-of-10 council members signed off
3. **Killer-assumption verbatim:** *"INT7 archive bytes (205,158) > INT8 archive bytes (187,731) at current packer = Pareto-dominated"*
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED** at current packer + **CARGO-CULTED** if assumed permanent (research paths unexplored: pair-int7 14-bit alignment, arithmetic-coded int7 stream, mixed int7+int8 layer assignment, hyperprior-conditioned int7 codebook)
5. **Today's 5 lenses:**
   - Chroma preservation: N/A
   - Dykstra-feasibility: HARD-EARNED at current packer
   - 9-dim: Dimension 5 OPTIMIZATION-PER-TECHNIQUE — substrate-optimal alternatives unexplored
   - Canonical-vs-unique: 8-bit byte-aligned IS canonical; 7-bit needs UNIQUE packer
   - 7-cargo-cult: NONE
6. **TIER classification:** **Tier 1** — reactivation paths EXPLICITLY enumerated; substrate-engineering deferred
7. **Reactivation criteria:** Build byte-aligned int7 packer (pair-int7 14-bit) AND empirically demonstrate < 187,731 byte archive at same rel_err class
8. **Cost / priority:** ~$0 design + $0 byte-arithmetic verification — **MEDIUM PRIORITY** (compositional path; not standalone score-lowering)

---

## Tier 2 — Mixed signals (warrant re-evaluation, lower priority)

### 2.1 Lane GP v4 (smooth-basis pose-fit structurally infeasible)

1. **Substrate ID:** `lane_gp_v4` — pose-fit via polynomial/B-spline/DCT smooth basis
2. **Current state:** KILLED 2026-04-30 via 4-round adversarial review with 3-section structural compliance; Check 91 (`check_pose_basis_fit_kill_acknowledged`) STRICT @ 0
3. **Killer-assumption verbatim:** *"Smooth-basis pose-fit can reach PoseNet noise floor RMSE < 0.01 below 7KB (Lane PFP16 raw fp16 baseline)"*
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED** — empirical analysis of actual Lane G v3 baseline poses showed white-noise signature (diff_std > signal_std in dims 1-5); DCT energy fraction in top-K coefficients was 27.8% in top-10 (uniform spectrum, NOT smooth)
5. **Today's 5 lenses:**
   - Chroma preservation: N/A (pose representation)
   - Dykstra-feasibility: HARD-EARNED (white-noise signature IS the empirical feasibility wall)
   - 9-dim: Dimension 4 RIGOR satisfied (4-round adversarial); Dimension 5 OPTIMIZATION-PER-TECHNIQUE — alternative ENCODER families (e.g., wavelet decomposition + temporal predictor) not exhausted
   - Canonical-vs-unique: Lane GP was unique-substrate-engineering; kill is structurally sound at this representation class
   - 7-cargo-cult: NONE
6. **TIER classification:** **Tier 2** — KILL is genuinely substrate-class-infeasible at smooth-basis representation; reactivation requires PARADIGM SHIFT (wavelet / Tikhonov / world-model predictive coding) not WITHIN-CLASS refinement (per `feedback_abandon_within_class_*`)
7. **Reactivation criteria:** Already documented: "if PFP16 amortization fails AND a non-polynomial fit family (DCT/B-spline) demonstrates RMSE < 0.5 on Lane G v3 baseline poses"; Time-Traveler L5 predictive-coding world-model (Z6/Z7/Z8) IS the canonical class-shift here
8. **Cost / priority:** $0 (subsumed by Time-Traveler L5) — **LOW PRIORITY** as standalone; HIGH as composition

### 2.2 Lane MM v2 (sister to 1.5; demoted Tier 2 placeholder for the bolt-on falsification specifically)

Already covered in 1.5; the bolt-on-architecture-mismatch finding is HARD-EARNED at substrate-engineering axis.

### 2.3 Lane 7 PSD (PixelShuffle-Downscale; unanimous 10/10 REJECT)

1. **Substrate ID:** `lane_7_psd` — PSD half-res bottleneck with PixelUnshuffle(2)
2. **Current state:** KILLED-DEFERRED 2026-04-30 ~03:00 CDT; unanimous 10/10 council REJECT
3. **Killer-assumption verbatim:** *"PSD half-res bottleneck destroys FastViT-PoseNet's required luma detail (5× empirical PoseNet regression)"*
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED** — historical 1.49 [contest-CUDA equivalent] + Quantizr (0.33 leader) didn't pick PSD despite "sweeping conv dims" + Selfcomp (0.38 leader) explicitly avoided PSD = Bayesian evidence against
5. **Today's 5 lenses:**
   - Chroma preservation: PSD PARTIALLY destroys luma detail (FastViT-PoseNet input)
   - Dykstra-feasibility: HARD-EARNED via 5× PoseNet regression empirical
   - 9-dim: Dimension 1 UNIQUENESS — PSD is within-FastViT-architecture refinement; doesn't class-shift the renderer
   - Canonical-vs-unique: Cargo-culted within-class
   - 7-cargo-cult: Symposium-#4-band-prediction (1.38 "breakthrough" was KL-distill + PSD; KL-distill itself on killed_techniques)
6. **TIER classification:** **Tier 2** — KILL is genuinely architecture-class-infeasible; reactivation requires the explicit criteria documented
7. **Reactivation criteria:** Already documented: "PoseNet-aware luma-skip variant (separate council review) OR floor moves below 0.50 OR Phase 2 Lane 19 demonstrates SegNet improvements transfer architecture-agnostically"
8. **Cost / priority:** $0 — **LOW PRIORITY**

### 2.4 Markov-1 AAC (adaptive arithmetic codec)

1. **Substrate ID:** `pr101_markov1_aac` — Markov-1 adaptive arithmetic codec on PR101's 228,958 INT8 symbols
2. **Current state:** FALSIFIED 2026-05-07 (183,144 vs brotli 162,050 = +21,094 worse; 31,038 bytes above theoretical floor)
3. **Killer-assumption verbatim:** *"Adaptive Markov-1 small-sample cost dominates per-tensor reset on 1.8M conditional bins"*
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED** — round-trip byte-faithful proven + per-tensor reset is the OBVIOUS-FIT canonical pattern; the 30KB small-sample cost is structural
5. **Today's 5 lenses:**
   - Chroma preservation: N/A
   - Dykstra-feasibility: HARD-EARNED (theoretical floor is oracle bound; achievable bound is adaptive)
   - 9-dim: Dimension 1 UNIQUENESS — Markov-1 IS class-shift (context modeling); reactivation paths exist
   - Canonical-vs-unique: Per-tensor reset is canonical; UNIQUE alternatives (shared context across tensors, hierarchical context, ANS-class entropy coders) unexplored
   - 7-cargo-cult: NONE
6. **TIER classification:** **Tier 2** — falsification VALID for this config; alternative configs (shared context, hierarchical, ANS) UNTESTED
7. **Reactivation criteria:** Test shared-context Markov-1 (single context table across all tensors) OR hierarchical-context (token + global context) OR ANS-class coder; if < 162,050 bytes brotli, RESURRECT
8. **Cost / priority:** $0 (analytical implementation) + $0 byte verification — **MEDIUM PRIORITY** (compositional with PR101 / PR103 / Z3 family)

### 2.5 AC bolt-on Alt C (arithmetic-coded per-tensor on PR101 lossy-coarsened K)

Same lineage as 2.4. FALSIFIED-as-measured-config 2026-05-08; reactivation criteria explicit. **Tier 2** — config-retired not class-kill.

### 2.6 PR101 sensitivity-aware Xavier-L2 proxy (DEFERRED 2026-05-08)

1. **Substrate ID:** `pr101_sensitivity_aware_xavier_l2` — Xavier-aware L2 importance proxy + Lagrangian byte allocator
2. **Current state:** DEFERRED-pending-real-Hessian; empirical Xavier-L2 made byte savings WORSE not better (+3,635 B regression at eta=1.0)
3. **Killer-assumption verbatim:** *"Xavier-aware L2 norm `importance(W) = sqrt(mean(W^2))` proxies for true score sensitivity"*
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED-CORE** — the empirical falsification of Xavier-L2 IS valid + **CARGO-CULTED-SHELL** if extended to the sensitivity-aware-quantization CLASS (Track 1 Decision 3 explicitly preserved)
5. **Today's 5 lenses:**
   - Chroma preservation: N/A
   - Dykstra-feasibility: HARD-EARNED
   - 9-dim: Dimension 1 UNIQUENESS — sensitivity-aware quantization IS class-shift
   - Canonical-vs-unique: Xavier-L2 is convenience proxy; UNIQUE alternatives include real Hessian-trace per-tensor importance OR Mallat wavelet-coefficient importance
   - 7-cargo-cult: NONE
6. **TIER classification:** **Tier 2** — proxy config retired; class is preserved
7. **Reactivation criteria:** Real Hessian-trace per-tensor importance (one CUDA forward+backward through scorers) OR Mallat wavelet-coefficient importance (Phase A3-alt council alternative)
8. **Cost / priority:** $5-10 Hessian computation + $5 paired smoke — **MEDIUM PRIORITY**

### 2.7 Wave 3 TCNeRV / BlockNeRV / FFNeRV / DSNeRV / HiNeRV (TERMINATED-API-CRASH 2026-05-13)

1. **Substrate IDs:** 5 NeRV-family substrate trainers
2. **Current state:** TERMINATED-API-CRASH 2026-05-13 (original subagent acbaff01/a802ad34 batch crashed mid-session without commit); DEFERRED-pending-research per CLAUDE.md "Forbidden premature KILL"
3. **Killer-assumption verbatim:** N/A (TERMINATED not killed)
4. **HARD-EARNED vs CARGO-CULTED:** **CARGO-CULTED** — TERMINATED is operational failure (API crash without commit), NOT scientific evidence
5. **Today's 5 lenses:**
   - Chroma preservation: each NeRV-family substrate may handle chroma differently
   - Dykstra-feasibility: not evaluated
   - 9-dim: Dimension 1 UNIQUENESS — NeRV-family IS class-shift (implicit neural representation)
   - Canonical-vs-unique: per-substrate canonical-vs-unique decision pending per `feedback_pr95_lesson_now_at_meta_level_*`
   - 7-cargo-cult: NONE
6. **TIER classification:** **Tier 2** — recoverable via subagent crash-resume per Catalog #206; partial implementations may exist
7. **Reactivation criteria:** Apply Catalog #206 crash-resume protocol; respawn subagent with predecessor checkpoint; commit incremental progress
8. **Cost / priority:** $0 recovery + design lands $0 — **MEDIUM PRIORITY** (DSNeRV + HiNeRV have explicit research_only=true gates pending Phase 2 council)

### 2.8 Lane 12 v2 NeRV-as-renderer (DEFERRED-Phase-B-$40-CUDA-dispatch)

1. **Substrate ID:** `lane_12_v2_nerv_as_renderer`
2. **Current state:** Phase A (design + scaffold) landed; Phase B (deferred): $40 CUDA dispatch; 5 reactivation preconditions documented
3. **Killer-assumption verbatim:** N/A (deferred)
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED** (Phase B explicitly council-gated for cost)
5. **Today's 5 lenses:** Dimension 1 UNIQUENESS — NeRV-as-renderer is class-shift; canonical-vs-unique decision required per design memo
6. **TIER classification:** **Tier 2** — deferred for operational/cost reason; scientific evidence pending
7. **Reactivation criteria:** 5 preconditions documented (operator-routable per cost band)
8. **Cost / priority:** $40 — **MEDIUM PRIORITY**

### 2.9 sane_hnerv first-anchor (4th attempt DEFERRED-pending-trainer-fix 2026-05-12)

Already covered by today's HIGH-RISK audit (Priority 5 there). **Tier 2** — 5 failed attempts is empirical-fragility signal, not class-kill.

### 2.10 PR101 lossy_coarsening T0312 (RETIRED-CONFIG 2026-05-08)

1. **Substrate ID:** `lossy-coarsening-cuda-20260508T0312-noproject`
2. **Current state:** measured_config_retired_exact_cuda_negative; archive 156,404 B → 0.3518 [contest-CUDA A-negative]; DO NOT redispatch this exact config
3. **Killer-assumption verbatim:** *"PR101 lossy_coarsening at rel_err=3.86% lands sub-0.20"* (cargo-culted prediction band)
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED-RETIRED-CONFIG**; cargo-culted prediction band class
5. **Today's 5 lenses:** Symposium-#4-band-prediction cargo-cult (predicted 0.18-0.22, actual 0.3518 = 1.6-1.9× outside-band; sister to NSCS06 v6 553× and apogee_int4 4.8× outside-band)
6. **TIER classification:** **Tier 2** — retired-config-specific; class preserved
7. **Reactivation criteria:** any NEW config (different rel_err, sensitivity-aware allocator, sub-cliff target ≤ 2% rel_err per `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_*`)
8. **Cost / priority:** $5-15 Modal A100 — **MEDIUM PRIORITY**

### 2.11 PR107 lossy_coarsening (substrate-mismatch FALSIFIED 2026-05-08)

1. **Substrate ID:** `pr107_apogee_lossy_coarsening` (3 configs b025/b035/b050)
2. **Current state:** FALSIFIED-AT-SUBSTRATE 2026-05-08; PR101 per-tensor analytical lossy_coarsening does NOT transfer to PR107 apogee architecture; all 3 GHA Linux x86_64 evals returned scores STRICTLY worse than baseline 0.19664
3. **Killer-assumption verbatim:** *"PR101 lossy_coarsening transfer to PR107 apogee"*
4. **HARD-EARNED vs CARGO-CULTED:** **HARD-EARNED** — 3 [contest-CPU] anchors all worse than baseline + per-tensor lossy_coarsening on PR107 violates substrate's PR107-tolerance bound
5. **Today's 5 lenses:** Dimension 1 UNIQUENESS — lossy_coarsening class preserved; PR107-specific application falsified per substrate-specific tolerance
6. **TIER classification:** **Tier 2** — measured-config retired; class preserved
7. **Reactivation criteria:** sensitivity-aware quantization on PR107 substrate (using PR107's actual Hessian, not PR101's analytical proxy)
8. **Cost / priority:** $5-15 Modal A100 — **MEDIUM PRIORITY**

### 2.12 Adversarial-audit-4-FALSIFICATIONS (kalle_fold / tiny_nn / lossy_int4 batch)

1. **Substrate IDs:** kalle_fold / tiny_nn / balle_hyperprior (already covered as 1.3) / lossy_int4
2. **Current state:** ALL FOUR RE-TAGGED DEFERRED-pending-research per CLAUDE.md "KILL is LAST RESORT" 2026-05-07
3. **Killer-assumption verbatim:** *"encoder-saturation finding implies these 4 individual technique-class claims are killed"*
4. **HARD-EARNED vs CARGO-CULTED:** **CARGO-CULTED** in original kill verdicts (single-config failures cannot be KILL); HARD-EARNED only as deferrals with reactivation criteria
5. **Today's 5 lenses:** Dimension 1 UNIQUENESS — all 4 are class-shifts; cargo-cult was treating one config as class-kill
6. **TIER classification:** **Tier 2** — already correctly re-tagged DEFERRED; awaiting reactivation experiments
7. **Reactivation criteria:** explicit per source memo (each technique gets at least one alternative-config test before KILL)
8. **Cost / priority:** $5-15 each = $20-60 total — **MEDIUM PRIORITY** in batch

---

## Tier 3 — Genuine falsifications (kept killed with hard-earned citation)

### 3.1 Lane MM v2 grayscale-LUT bolt-on (architectural mismatch)

**HARD-EARNED CITATION:** Selfcomp's PR#56 paradigm explicitly TRAINS-FROM-SCRATCH with grayscale-LUT; bolt-on application onto 3ch-trained renderer is architecturally invalid by design. **3-section compliance:** council review present (Round 7 retag), internal-consistency check present (`[contest-CPU advisory]` axis-tag-cleanup), reactivation criteria present (Lane MM v3 trained-from-scratch path). **STATUS:** KEEP-KILLED for the bolt-on application specifically; Lane MM v3 reactivation per 1.5.

### 3.2 PR107 lossy_coarsening transfer (substrate-mismatch FALSIFIED with 3 [contest-CPU] anchors)

**HARD-EARNED CITATION:** 3 anchors on GHA Linux x86_64 (the leaderboard's actual hardware) all strictly worse than PR107 baseline 0.19664. **3-section compliance:** council not explicit (single-subagent finding); reactivation criteria present (apply PR107-specific Hessian instead of PR101 analytical proxy). **STATUS:** KEEP-KILLED for the specific PR101→PR107 transfer; class preserved.

### 3.3 Lane OWv3 0120 Arithmetic-Coded Masks (AMRC 2.57× AV1; 7/0 FALSIFY)

**HARD-EARNED CITATION:** Council 7/0 FALSIFY + AMRC at 1,082,649 bytes vs AV1 421,483 bytes = 2.57× regression + EntropyCoder LZMA 1,073,195 bytes = 2.55× regression. Lossless symbol-stream coding cannot beat AV1's spatial redundancy exploitation on 5-class argmax masks. **3-section compliance:** council 7/0 with named members; internal-consistency check present (AMRC round-trip byte-faithful + LZMA round-trip); reactivation criteria implicit (NOT raw-symbol-stream coding). **STATUS:** KEEP-KILLED for raw-symbol-stream AMRC; class shift required (e.g., AV1+STC residual; temporal predictor).

### 3.4 Three lossy anchors (rel_err² Lagrangian objective FALSIFIED at rms ≥ 0.04)

**HARD-EARNED CITATION:** Three independent [contest-CUDA] anchors (apogee_int4 1.4287 + lossy_coarsening_analytical 0.3517 + PR106_UNIWARD_packet 0.3371) all show the rel_err-vs-score CLIFF: rel_err ≥ 0.04 → SegNet/PoseNet discontinuous transitions. **3-section compliance:** explicit empirical anchors + reactivation criteria (sensitivity-aware Lagrangian / sub-cliff targeting). **STATUS:** KEEP-FALSIFIED for the rel_err² Lagrangian objective at rms ≥ 0.04 SPECIFICALLY; sister proxies (Fisher-Info / wavelet-coefficient / real-Hessian) preserved.

### 3.5 PR106 Lanes #05 + #06 as-originally-designed (architectural non-applicability)

**HARD-EARNED CITATION:** PR106 archive has no mask channel (single 0.bin = brotli-decoder + brotli-latents only; verified via extract_pr106_decoder.py); UNIWARD requires signal-vs-cover decomposition; grayscale-LUT requires SegNet-mask discrete-class output. **3-section compliance:** explicit empirical verification + reactivation criteria implicit (REFORMULATE for PR106 architecture per 1.4). **STATUS:** KEEP-KILLED for as-originally-designed (the architectural non-applicability); REFORMULATION ALIVE per 1.4 (different lane id required).

### 3.6 GP v4 smooth-basis pose-fit (white-noise signature)

**HARD-EARNED CITATION:** Empirical analysis of Lane G v3 baseline poses (white-noise signature, DCT energy 27.8% in top-10) + 4-round adversarial review + Check 91 STRICT preflight extinction. **3-section compliance:** explicit (4-round + Check 91 + memory entry + design memo). **STATUS:** KEEP-KILLED at smooth-basis representation class; Time-Traveler L5 predictive-coding world-models is the canonical class-shift reactivation path.

### 3.7 Lane V/V-V2/F-V4/D-V3/J-JBL (5 killed Vast.ai lanes from forensic audit 2026-04-28)

**HARD-EARNED CITATION:** Forensic audit `project_killed_lanes_forensic_audit_20260428.md` categorized each as ENGINEERING / CONFIGURATION / HOST-SIDE / TRULY-BROKEN. Half-frame failures (V/V-V2/D-V3) are NOT paradigm failures (Quantizr ships half-frame at 0.33); BUGS in channel broadcasting, loss-mode validator allowlist, joint warp-expansion training. **3-section compliance:** council audit roster present (Yousfi/Fridrich/Hotz/Quantizr/Contrarian rotating); per-lane diagnostic per CLAUDE.md "Apples-to-apples evidence discipline"; reactivation implicit (engineering bugs fixable). **STATUS:** KEEP-KILLED-AS-MEASURED-CONFIGS (each specific Vast.ai instance configuration); UNDERLYING TECHNIQUES (half-frame, joint warp-expansion, mask-channel broadcasting) ALIVE in canonical leaderboard (Quantizr 0.33).

### 3.8 STCB deflated codec (18× worse than AV1 baseline)

**HARD-EARNED CITATION:** Codec has structural bug — "one-majority-plus-exceptions" stores 109M exceptions vs 11.8M boundaries on multi-region masks. Floor: deflated STCB at 0.259 bpp = 7.6MB, 18× worse than AV1's 0.014 bpp. **3-section compliance:** 22-voice extreme-rigor codex review (45% endorsement); empirical receipts; reactivation criteria explicit (AV1+STC residual / temporal predictor / scanline RLE / lossy STC). **STATUS:** KEEP-FALSIFIED for the original STCB; REDESIGN ALIVE per 4 reactivation paths.

### 3.9 Z3-G1 scorer-softmax-hyperprior-gating (phantom-distinguishing-feature; v1)

**HARD-EARNED CITATION:** Empirical receipt — Z3-G1 v1 archive bytes IDENTICAL to Z3 v2 baseline 0.19869 to 5 decimals; the smart distinguishing feature (1KB SegNet-class CDF) was engineered but NEVER WIRED into archive (empty `hyperprior_weights_int8 = b""` + `w_hat_int8 = b""` slots). **3-section compliance:** explicit operator decision 2026-05-15 (FULL CPU paired ABORTED + DIRECT_RESIDUAL_Z3HV2_REPRODUCTION tag on FULL CUDA harvest); reactivation criteria explicit ([g1_entropy_coded_sigma_table_and_class_index_wire_grammar_landed, no_op_detector_byte_mutation_smoke_proves_g1_bytes_consumed_by_inflate, codex_fix_wave_a038fba_strict_preflight_gate_landed]). **STATUS:** KEEP-research_only-PENDING v2 implementation; v1 KEEP-NOT-PROMOTED (phantom-distinguishing-feature class per Catalog #272).

### 3.10 NSCS06 v6 Carmack-Hotz Strip-Everything (553× outside-band; PROCEED_WITH_REVISIONS not KILL)

**HARD-EARNED CITATION:** Empirical receipt 105.15 [diagnostic_cpu] vs predicted [0.10, 0.20] band = 553× outside band; pose=149.03 (PoseNet NOT translation-invariant); seg=64.59 (SegNet stride-2 cannot recover destroyed chroma). **3-section compliance:** symposium council T4 attendance (20 voices); Assumption-Adversary VETO recorded; reactivation criteria via Path A/B/C/F multi-path with cargo-cult-unwinding mandate; v7 Path A landed 58.89 [diagnostic-CPU] in ONE iteration (4-of-7 cargo-cult unwinds; ~52% score reduction). **STATUS:** v6-as-shipped KEEP-research_only; v7+ ACTIVE per current empirical anchor; this audit's EXISTENCE is triggered by v7 empirical proof.

---

## Cross-substrate patterns

### Common cargo-cult patterns across Tier 1 + Tier 2

**Pattern A — Symposium-#4-band-prediction cargo-cult (kill on band-failure without Dykstra-feasibility check)**
- Present in: apogee_int4 (4.8× outside-band) / Lane MM v2 (3× outside-band) / Lane 17 IMP (initial kill, withdrawn) / PR101 lossy_coarsening T0312 (1.6-1.9× outside-band) / PR107 lossy_coarsening (3 configs all outside-band) / NSCS06 v6 (553× outside-band)
- Root cause: predicted bands were ADDITIVE composition of per-technique-improvement estimates without Dykstra-intersection check of contest polytope constraints
- Today's catalog gate: #296 (`check_substrate_predicted_band_has_dykstra_feasibility_check`) lands this structural protection going forward; sister tool `tools/check_substrate_dykstra_feasibility.py` (proposed per `feedback_high_risk_substrate_cargo_cult_unwind_audit_*`) provides $0 analytical disambiguation

**Pattern B — Substrate-mismatch-as-class-kill (kill ON WRONG-CLASS, kill REPORTED AS KILL-OF-CLASS)**
- Present in: PR101 CompressAI Ballé hyperprior (applied to symbols-as-image; class is PRESERVED) / PR106 Lanes #05 + #06 (applied to PR106-presumed-mask-channel; class is PRESERVED) / PR107 lossy_coarsening (applied across substrates; class is PRESERVED for original PR101 substrate)
- Root cause: subagents conflate "this technique FAILED on this substrate" with "this technique class is KILLED"
- Per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable: 3-section structural compliance with explicit research-path-exhaustion requirement; sister gate Catalog #185 (LIVE_COUNT drift) prevents catalog text from drifting from gate empirical state

**Pattern C — Evidence-grade-cargo-cult (kill on MPS / CPU-not-contest-CPU / single-axis-extrapolation)**
- Present in: Lane STC clean-source (MPS-PROXY kill withdrawn 2026-04-29) / Lane MM v2 (Modal CPU eval treated as decision-grade) / Lane 17 IMP (stub-loop stats.json treated as 200-epoch-converged) / Z3-G1 v1 phantom-CUDA-score (CPU eval routed to `_cuda.json` filename)
- Root cause: violations of CLAUDE.md "Apples-to-apples evidence discipline" + "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
- Today's catalog gate set: #127 (custody validator) / #221 (auth-eval-result-fail-closed) / #226 (canonical helper routing) / #249 (no misleading device-named directories) extinct this structurally going forward

**Pattern D — Within-class-refinement-treated-as-class-shift (kill INFORMS within-class refinement, NOT class-shift abandonment)**
- Present in: Tasks #216 DEFER batch (HM-S/WC-S/MAE-V/SAUG) — MAE-V is CLASS-SHIFT (self-supervised pre-training) but deferred for operational reasons; Lane FC/PA tasks #225 DEMOTED to lesser-role despite class-shift potential
- Root cause: operational defer (DNS bug, Vast.ai balance, trainer recipe) misread as scientific kill
- Today's standing directive `feedback_council_apparatus_in_service_of_innovation_*` mission-alignment + Catalog #291 META-ASSUMPTION cadence extinct this going forward

### META-classes that today's catalog gates now structurally prevent

| META class | Catalog # | What it prevents | Empirical anchor |
|---|---|---|---|
| Symposium-#4-band cargo-cult | #296 (proposed) | Predicted-band-without-Dykstra-feasibility check | NSCS06 v6 553× outside-band |
| Signal-axis-destruction cargo-cult | #297 (proposed) | Y=R=G=B / chroma-collapse / np.roll without reversibility probe | NSCS06 v6 chroma loss seg=64.59 |
| Phantom-distinguishing-feature cargo-cult | #272 | Substrate L2+ claims distinguishing feature but bytes not consumed by inflate | Z3-G1 v1 (0.19869 [diagnostic-CPU] identical to Z3 v2 baseline) |
| Phantom-score directory cargo-cult | #249 | Filename device-token ≠ actual eval device | Z3-G1 v1 CPU eval to `_cuda.json` |
| Evidence-grade-cargo-cult | #127 / #221 / #226 | Tag/axis/hardware-substrate mismatch in promotion | Lane MM v2 + Lane 17 IMP + Lane STC |
| Substrate-mismatch-as-class-kill | #185 | CLAUDE.md catalog text drifts from gate empirical state | PR101 CompressAI Ballé + PR106 #05 + PR106 #06 |
| KILL-without-research-exhaustion | (CLAUDE.md non-negotiable) | KILL verdict without 3-section structural compliance | All adversarial-audit-4-FALSIFICATIONS 2026-05-07 (DEFERRED batch) |
| Within-class-refinement (treated as class-shift) | (`feedback_abandon_within_class_*` standing directive) | Pursuit of within-class refinements that perpetuate 0.196-0.199 plateau | Multiple historical kills mis-categorized as class-shift candidates |
| Canonical-vs-unique-suppression | #290 | Substrate scaffolds force-canonical without evaluation of substrate-optimal alternative | PR101 CompressAI Ballé applied to symbol stream (canonical force-fit) |
| Per-deliberation-assumption-statement | #292 | Council deliberations reach consensus without per-member operating-within assumption-statement | Multiple historical kills with implicit shared assumptions |

---

## Reactivation queue (top 10 ranked by EV/$ × dispatch-readiness urgency)

| # | Substrate id | Tier | Killer-assumption | Today's classification | Reactivation criteria | Cost estimate | Priority |
|---|---|---|---|---|---|---:|---|
| 1 | `lane_17_imp` | 1 | "1.98 is architectural ceiling" | CARGO-CULTED (stub-loop bug) | Run with proper `train_distill` fine-tune | $5-15 | **HIGHEST** |
| 2 | `lane_stc_clean_source` | 1 | "21MB stream because boundary-fraction too high" | CARGO-CULTED (MPS-PROXY evidence) | Modal T4 CUDA re-run on clean SegNet argmax | $0.20 | **HIGH** |
| 3 | `lane_apogee_int4` | 1 | "INT4 NAIVE-PTQ → 1.43" | HARD-EARNED-CORE + CARGO-CULTED-SHELL | QAT (Quantizr 0.33 leader's winning recipe) | $5-15 | **HIGH** |
| 4 | `revival_plan_05_pr106_uniward` | 1 | "PR106 has mask channel" | CARGO-CULTED (lane design vs PR106 arch) | REFORMULATE: "UNIWARD-delta on PR106 latent stream" | $0 design + $5 paired | **HIGH** |
| 5 | `revival_plan_06_pr106_lut` | 1 | "PR106 has SegNet-mask discrete-class output" | CARGO-CULTED (lane design vs PR106 arch) | REFORMULATE: "grayscale-LUT on PR106 latent codebook" | $0 design + $5 paired | **HIGH** |
| 6 | `lane_pr101_compressai_balle_full` | 1 | "Hyperprior cannot reconstruct PR101 symbols" | PARTIAL-HARD-EARNED + CARGO-CULTED-SHELL | Apply to NSCS06-v7 chroma residuals / ATW codec / NSCS03 latent stream | $0 (NSCS03 already lands) + $5 ATW probe | **HIGH** |
| 7 | `lane_mm_v3_segmap_trained_from_scratch` | 1 | "Hard-argmax LUT bolt-on falsified" | PARTIAL-HARD-EARNED + CARGO-CULTED-SHELL | SegMap-trained-from-scratch (per Selfcomp 0.38 anchor) | $5-15 | **MEDIUM-HIGH** |
| 8 | `lane_mae_v` + `lane_saug` (Tasks #216) | 1 | "DEFER-pending-operational-fix" | CARGO-CULTED (operational not scientific) | Pre-train MAE-V on `upstream/videos/0.mkv` + redispatch SAUG on Modal/Lightning | $10-25 total | **MEDIUM-HIGH** |
| 9 | `lane_al_promote` (Task #225 demoted reversal) | 1 | "Lane AL demoted to lesser-role" | CARGO-CULTED (contradicts MM v2 evidence) | PROMOTE Lane AL pair with Lane MM v3 | $5-15 | **MEDIUM** |
| 10 | Wave 3 NeRV-family (TCNeRV/BlockNeRV/FFNeRV/DSNeRV/HiNeRV) | 2 | "TERMINATED-API-CRASH" | CARGO-CULTED (operational not scientific) | Catalog #206 crash-resume protocol respawn | $0 recovery + design | **MEDIUM** |

---

## Recommended next moves (for T3 batched council adjudication per task #742)

1. **Re-run Lane 17 IMP cycle 0 with proper `train_distill` fine-tune** (cost $5-15 Modal A100 100ep). The KILL is already withdrawn; the only missing piece is the empirical receipt with proper fine-tune. PoseNet 34.8× vs SegNet 1.25× asymmetric regression is empirically-supported fixable-stub-bug hypothesis per Frankle 2019.

2. **Run Lane STC clean-source on Modal T4 CUDA** (cost $0.20, ~10 min). Cheapest possible re-eval; explicit UNDETERMINED tag in source memo; council's #1 hope is back on the table if MPS-vs-CUDA delta is structural.

3. **Reformulate PR106 Lanes #05 + #06 to PR106's actual HNeRV architecture** (cost $0 design + $5 per paired smoke). Both UNIWARD + grayscale-LUT are leaderboard-proven primitives; only the lane DESIGN was cargo-culted, not the techniques.

4. **Apply Ballé hyperprior to NSCS06-v7 chroma residuals OR ATW codec OR NSCS03 latent stream** (cost $0 NSCS03 already lands + $5 ATW probe). The PR101-symbol-substrate falsification does NOT transfer to substrates WITH spatial locality.

5. **Pre-train MAE-V on `upstream/videos/0.mkv` and re-deploy SAUG on Modal/Lightning** (cost $10-25 total). Both are CLASS-SHIFT per `feedback_abandon_within_class_*`; deferrals were operational (DNS / balance) not scientific.

**Council deliberation required** (per CLAUDE.md "Design decisions — non-negotiable"):
- Lane 17 IMP re-run priority vs operator-routed Tier 1 substrate backfill sweep (which Tier 1 substrate gets the next Modal slot?)
- Reactivation of Lane AL/FC/PA (Task #225 demotion reversal) — is the directional MM v2 evidence sufficient to promote without dispatch?
- Sequencing of reformulated PR106 lanes vs new substrate-class-shift scaffolds (NSCS01/02/03/04/05 already in flight; reactivation queue contends for sister-subagent slots)

---

## What this audit DOES NOT do

- **Does NOT reopen any lanes** (council T3 decides per task #742). Lane registry mutations are out-of-scope; this audit's deliverable is THIS MEMO ONLY plus checkpoint records.
- **Does NOT change any kill verdict.** Tier 3 "kept killed with hard-earned citation" is the audit's classification; the verdict's source memo remains the source of truth.
- **Does NOT spawn re-validation dispatches.** Cost estimates are operator-decision inputs, not actuator commands. Operator approval + paired-env discipline (per Catalog #199) + smoke-before-full (per Catalog #167) + canonical operator-authorize (per Catalog #176/#243/#271) all apply to any subsequent re-validation dispatch.
- **Does NOT enumerate 100+ "operational" defers** (codex-fix-wave / preflight-gate / state-hygiene lanes). The audit scope is SUBSTRATE / TECHNIQUE / METHOD kills; infrastructure defers are out-of-scope.
- **Does NOT propose NEW kills.** Per CLAUDE.md "Forbidden premature KILL", every lane in this audit either has existing kill compliance (Tier 3 with 3-section structural review) or is a re-evaluation candidate (Tier 1+2). No NEW kill verdicts proposed.

---

## Cross-references

**Empirical anchors:**
- NSCS06 v7 Path A (the triggering anchor): 58.89 [diagnostic-CPU] via 4-of-7 cargo-cult unwinds from v6 105.15 (operator-provided in resurrection-audit prompt)
- NSCS06 v6 falsification anchor: `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md`
- Today's HIGH-RISK 5 unwind audit: `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` (methodological template)
- Today's META-assumption backfill audit: `.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md`

**Standing directives:**
- Mission-alignment: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_council_apparatus_in_service_of_innovation_rigor_optimization_score_lowering_20260516.md`
- Hard-earned-vs-cargo-culted classification: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- Abandon within-class refinements: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_abandon_within_class_refinements_only_substrate_class_shifts_pursue_frontier_20260515.md`
- UNIQUE-AND-COMPLETE-PER-METHOD: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`
- 9-dim success checklist: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_9_dimension_success_checklist_per_substrate_and_stack_of_stacks_standing_directive_20260515.md`

**CLAUDE.md non-negotiables anchoring this audit:**
- "Forbidden premature KILL without research exhaustion (the kill-too-fast trap)" — the structural anchor
- "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS" — 3-section structural requirement
- "Council conduct — non-negotiable" — no-conservative-bias + sextet pact (now sextet) + Assumption-Adversary seat (Catalog #292)
- "META-ASSUMPTION ADVERSARIAL REVIEW" (Catalog #291) — recurring cadence + per-deliberation assumption-statement
- "HNeRV / leaderboard-implementation parity discipline" — 13 inviolable lessons
- "Apples-to-apples evidence discipline" — every score classification carries axis tag

**Catalog gates relevant to audit interpretation:**
- #127 (custody validator) / #221 (auth-eval-result-fail-closed) / #226 (canonical helper) / #249 (no misleading device-named) — Pattern C (Evidence-grade) prevention
- #272 (distinguishing-feature integration contract) — Pattern E (phantom-distinguishing-feature) prevention
- #290 (canonical-vs-unique decision per layer) — Pattern F (force-canonical-suppression) prevention
- #292 (per-deliberation assumption statement) — META-ASSUMPTION discipline at deliberation surface
- #296 (predicted-band Dykstra-feasibility — proposed) — Pattern A (Symposium-#4-band) prevention
- #297 (signal-axis destruction reversibility probe — proposed) — Pattern B prevention for chroma class

---

**END OF RESURRECTION AUDIT LEDGER**
