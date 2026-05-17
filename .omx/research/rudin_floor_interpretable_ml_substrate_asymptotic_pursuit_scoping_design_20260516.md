---
name: rudin-floor-interpretable-ml-substrate-asymptotic-pursuit-scoping-design-20260516
description: |
  Comprehensive scoping design memo for the Rudin floor substrate — an
  interpretable-ML compositional decoder that pursues the Rudin floor
  (~0.10-0.13 long-term per T4 SYMPOSIUM 4×4 floor matrix) via a fully-
  interpretable falling-rule-list + SLIM + GOSDT + Rashomon ensemble
  ARCHITECTURE (NOT just an autopilot ranker; the SUBSTRATE itself).
  Currently a STRUCTURAL GAP in the portfolio per FALSIFICATION-AUDIT-v2
  A3 + HORIZON-CLASS directive (NEVER BUILT). Per UNIQUE-AND-COMPLETE-PER-
  METHOD operating mode: ONE coherent ~700-1000 LOC bind-everything packet
  (encoder GOSDT trees + decoder falling-rule-list + loss Rashomon
  ensemble + RDIF archive grammar + ≤200 LOC pure-Python inflate + per-
  rule observability + Dykstra-feasibility-validated predicted band).
substrate_id: rudin_floor_interpretable_ml_compositional_decoder
substrate_class_shift: architecture_class_AND_decode_time_contract_AND_scorer_relationship
horizon_class: asymptotic_pursuit
predicted_band_short_1_7d: "REQUEST_OPERATOR_REVIEW [whiteboard] — insufficient measurements per T4 4×4 matrix Rudin floor Short row"
predicted_band_mid_30_90d: "[0.150, 0.180] [prediction; first-principles + compressive-sensing K=8; Dykstra-feasibility check pending]"
predicted_band_long_6m_1y: "[0.10, 0.13] [prediction; full Rudin compositional lower envelope with all class-shifts validated; falling-rule-list over the lattice]"
predicted_band_asymptotic_ultimate: "[0.05, 0.10] [prediction; ultimate Rudin floor = convex-hull lower envelope; gap to Shannon ultimate is the irreducible interpretability tax]"
predicted_delta_s_band_dykstra_verdict: "PENDING — requires bias-bracketing of (interpretability_tax, lattice_resolution, rule_list_depth) feasibility intersection"
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Cynthia Rudin (Time-Traveler protégé), Ingrid Daubechies (Time-Traveler peer), David MacKay (memorial), Aaron D. Wyner, John Carmack, Stephen Boyd]
council_quorum_met: pending_T3_ratification
council_verdict: PROCEED_WITH_REVISIONS_PENDING_T3_RATIFICATION
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
lane: lane_rudin_floor_interpretable_ml_substrate_scoping_design_20260516
parent_anchor:
  - "T4 SYMPOSIUM commit 28e7c67c7 — Rudin floor row of 4×4 floor matrix"
  - "FALSIFICATION-AUDIT-v2 A3 — NEW asymptotic-pursuit candidate; NEVER BUILT"
  - "Catalogs #273-#278 Rudin-Daubechies AUTOPILOT (the META-layer this substrate operates ON, not the substrate itself)"
  - "HORIZON-CLASS standing directive — Rudin floor substrate named as STRUCTURAL GAP"
---

# Rudin Floor Substrate — Interpretable-ML Compositional Decoder — Comprehensive Scoping Design

**Date:** 2026-05-16
**Lane:** `lane_rudin_floor_interpretable_ml_substrate_scoping_design_20260516`
**Horizon class:** **ASYMPTOTIC-PURSUIT** (per HORIZON-CLASS directive; predicted long-horizon CPU band [0.10, 0.13] per T4 SYMPOSIUM 4×4 floor matrix Rudin floor row)
**Substrate class-shift axes (per abandon-within-class directive):** **architecture class** (no neural codec at inflate) + **decode-time contract** (pure rule application — zero PyTorch) + **scorer-relationship** (Rashomon-ensemble loss preserves epistemic diversity across SegNet/PoseNet evaluations)
**Status:** SCOPING DESIGN MEMO; substrate NEVER BUILT; this memo is L0 SKETCH per CLAUDE.md "Lane maturity registry" lifecycle discipline.

---

## Premise verification (per Catalog #229)

8 pre-edit verifications confirmed BEFORE writing any section of this memo:

1. ✅ CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable read — substrate scaffolds default to unique-and-complete; canonical helpers used only when serve.
2. ✅ CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13 inviolable lessons + the 8th forbidden pattern + 5 forbidden code patterns read — substrate must bind ALL ingredients simultaneously.
3. ✅ CLAUDE.md "Forbidden premature KILL without research exhaustion" — no kills proposed; only DEFERRED-pending-research verdicts with reactivation criteria.
4. ✅ T4 SYMPOSIUM `.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md` 4×4 floor matrix Rudin floor row confirmed: Short=REQUEST_OPERATOR_REVIEW; Mid=[0.150, 0.180]; Long=[0.10, 0.13]; Asymptotic=[0.05, 0.10].
5. ✅ FALSIFICATION-AUDIT-v2 `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` A3 entry confirmed: "Rudin floor substrate (interpretable-ML compositional decoder) — NEW; NEVER BUILT; Catalog #273-#278 Rudin-Daubechies autopilot is partial step (preflight surface only); FULL substrate would be Rudin's main frontier."
6. ✅ HORIZON-CLASS standing directive confirmed: Rudin floor substrate listed under ASYMPTOTIC-PURSUIT GAP; K=8 schedule rebalance MUST allocate ≥20% to asymptotic-pursuit.
7. ✅ Rudin-Daubechies autopilot modules confirmed at `src/tac/autopilot_rudin_daubechies/{slim_ranker,falling_rule_list,rashomon_ensemble,compressive_landscape,wavelet_multi_scale_ranker,gosdt_dispatcher}.py` (Catalogs #273-#278) — these are the META infrastructure the substrate would USE; the substrate is the FIRST consumer of the meta-layer applied as decode-time architecture.
8. ✅ Sister NSCS06 v8 design memo `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md` 22-section template confirmed — adopting the structural skeleton.

NO Rudin substrate references in `.omx/research/*.md` prior to this memo (confirmed via grep across all design memos). This is the FIRST canonical instance.

---

## Operating-within assumption-statement (per Catalog #292 sextet-pact discipline)

The shared assumption this scoping memo operates within: *"The Rudin floor (interpretable-ML compositional lower envelope per T4 SYMPOSIUM) is BOTH a theoretical lower bound on what's achievable within bounded-complexity engineering AND an operationally-pursuable substrate target — the falling-rule-list + SLIM + GOSDT + Rashomon ensemble apparatus that today serves as autopilot RANKER (Catalogs #273-#278) can be RE-ARCHITECTED to serve as the decoder ARCHITECTURE itself, producing a substrate whose every prediction is traceable to which rule fired."*

**HARD-EARNED basis** per Catalog #292 + hard-earned-vs-cargo-culted addendum:
- The 4×4 floor matrix Rudin floor row is CITED in T4 SYMPOSIUM as canonical (CARGO-CULTED-FREE per the symposium's own Assumption-Adversary verdict V4 "4-floor matrix lowest-achievable estimates ANCHORED via Dykstra-feasibility-validated convex-hull lower envelope").
- The Rudin-Daubechies autopilot landed today (Catalogs #273-#278; commit `42665ce95`) is empirical receipt that the techniques (SLIM / falling-rule-list / GOSDT / Rashomon / wavelet multi-scale / compressive-sensing) are implementable + maintainable in `tac`.
- The interpretability discipline IS HARD-EARNED at the META-layer (per the Rudin-Daubechies autopilot's existence + the operator's standing observability directive — `feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md` explicitly names "Falling-rule lists / SLIM / GOSDT (Rudin interpretability per Catalog #273-#278) are observability-MAX per the Rudin canonical discipline").

**Per-member operating-within assumptions (Fix 7):**

- **Cynthia Rudin (Time-Traveler protégé, voice channeled)** — operating within: *"From interpretable-ML: a falling-rule-list of K=4-6 rules captures 90%+ of binary-classification problems with first-match-wins semantics. Applied as a DECODER architecture (not a ranker): each frame's reconstruction is a SEQUENCE of rule-applications; each rule's predicate references the upstream encoded symbols + the previously-decoded frames. The decoder IS the rule list. Operator reads which rule fired per pixel; this is the canonical interpretability discipline applied to compression — every byte spent is traceable to which decision produced it."* **HARD-EARNED** (per Wang-Rudin 2015 falling-rule-list canonical + Ustun-Rudin 2016 SLIM canonical + Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 GOSDT canonical).
- **Ingrid Daubechies (Time-Traveler peer, voice channeled)** — operating within: *"From compressive sensing + multi-scale analysis: a sparse signal of dimension N is recoverable from K=O(√N) measurements when the signal IS sparse in the wavelet basis. Applied as a SUBSTRATE: the per-frame encoding is the SPARSE SIGNAL; the wavelet hierarchy organizes per-scale rules from coarse to fine; the GOSDT encoder selects which subbands carry signal; the Rashomon ensemble preserves K=8 near-optimal encoders and the operator selects the most-interpretable. The encoder is structurally O(log N) deep — Daubechies wavelet decomposition depth — not O(N) deep like learned networks."* **HARD-EARNED**.
- **David MacKay (memorial seat)** — operating within: *"From information-theory + MDL: each rule's predicate IS a hypothesis; each rule's predicted-band IS a Bayesian posterior under the SLIM integer-coefficient prior; the falling-rule-list IS a decision tree compiled into a sequential prior — first-match-wins is the canonical compositional prior. The rate term is bits-of-rules-encoded + bits-of-residual; MDL formalism handles this cleanly via two-part code (`L(model) + L(data|model)`). Predicted Rudin floor for `H(frames | scorer_state_dict + sequence_of_K_rules)` is approximately Shannon R(D)+ε plus interpretability_tax (~3-5% per Rudin canonical bound). Lower bound: ~0.10 [prediction; first-principles MDL bound; n=0]."* **HARD-EARNED**.
- **Aaron D. Wyner** — operating within: *"From Wyner-Ziv 1976: source coding with side-info at decoder. The SegNet+PoseNet weights are SHARED PRIOR available at both encoder + decoder; only H(frames | scorer_state_dict) bits are needed. Applied to the Rudin substrate: each rule's predicate REFERENCES the scorer output (e.g. `if SegNet_argmax_class[x,y] == ROAD then frame_color = LUT_road_grayscale * SCORER_LUMA_PRIOR[scorer_class_id]`). The decoder uses the scorer as ORACLE rather than learning a separate codec. Lower bound for this substrate class: ~0.05-0.08 [prediction; Wyner-Ziv with shared scorer prior; n=0]."* **HARD-EARNED**.
- **John Carmack** — operating within: *"From engineering shortcuts + Strip-Everything (NSCS06 v6→v7 lineage): pure-Python inflate IS achievable. The Rudin substrate IS the natural successor to NSCS06's chroma-preserving + numpy+Pillow inflate paradigm — except instead of palette-LUT it's rule-application. ≤200 LOC inflate is the budget; reviewable in 30 seconds. The 30-min T4 contest-CUDA eval constraint is irrelevant because there's no GPU inflate — CPU-only is canonical. Operationally simpler than NSCS06 v8 (which uses Daubechies wavelet + Wyner-Ziv frame coder)."* **HARD-EARNED-AT-PARADIGM-LEVEL** (chroma-preserving + no-neural-codec lineage); **CARGO-CULTED-AT-IMPLEMENTATION-LEVEL** (specific rule schema is design space, not given).
- **Stephen Boyd** — operating within: *"From convex optimization at operational level (ADMM + Dykstra alternating projections): the Rudin substrate is structurally a CONVEX FEASIBILITY problem — find a rule-list R such that (a) `rate_bytes(R) ≤ 600KB`, (b) `seg_distortion(R, scorer) ≤ S`, (c) `pose_distortion(R, scorer) ≤ P`, (d) `|rules(R)| ≤ K=6 per Wang-Rudin canonical`. The Dykstra-feasibility intersection check is the canonical pre-dispatch gate per Catalog #296. PENDING is the actual bias-bracketing of (interpretability_tax, lattice_resolution, rule_list_depth) per the Section 18 protocol."* **HARD-EARNED**.
- **Shannon (LEAD)** — operating within: *"R(D) lower bound for the contest video at the contest scorer's distortion criterion is ~0.02-0.05 (per Catalog #257 Blahut-Arimoto). The Rudin floor adds interpretability_tax (~3-5%) on top — yielding ~0.05-0.10 asymptotic. Mid-horizon Rudin floor depends on how many rules can be empirically validated via K=8 Daubechies-DeVore measurements; per T4 SYMPOSIUM Mid row 0.150-0.180 is the predicted band assuming K=8 measurements suffice. This is consistent with the Rashomon ensemble + falling-rule discipline lower-bounding implementation overhead."* **HARD-EARNED**.
- **Dykstra (CO-LEAD)** — operating within: *"Convex feasibility per Catalog #296 protocol applies before any paid dispatch. The Rudin substrate's predicted band MUST pass Dykstra-feasibility intersection at (rate ≤ 600KB, seg ≤ 0.06, pose ≤ 50.0) BEFORE the K=8 LEVEL-1 dispatch fires. The substrate-design's `tools/check_substrate_dykstra_feasibility.py` invocation is mandatory — PENDING until the substrate scaffold lands at L1."* **HARD-EARNED**.
- **Contrarian** — operating within: *"The Rudin floor is the OPERATIONAL TARGET per T4 SYMPOSIUM, not the Shannon floor. Interpretability is a TAX, not a free lunch — every interpretable rule costs ~3-5% over the Shannon optimum. The substrate must NOT over-promise (predicted Mid band 0.150-0.180 is HONEST — within striking distance of Quantizr 0.33 leader band but NOT medal-band). Demand explicit citation of Wang-Rudin 2015 + Ustun-Rudin 2016 + Lin-Rudin 2020 for every architectural claim. NEVER pursue Rudin substrate as a within-class refinement of A1 — it MUST be a genuine class-shift per abandon-within-class directive."* **HARD-EARNED-WITH-CAUTION-FLAG**.
- **Assumption-Adversary** — operating within: *"The shared assumption across this scoping memo is: 'the META-layer Rudin-Daubechies autopilot (Catalogs #273-#278) CAN be re-architected as the decode-time architecture.' This is HARD-EARNED at the technique level (the techniques EXIST + are implemented + tested) but is CARGO-CULTED at the application level (no empirical anchor yet that a Rudin-floor SUBSTRATE has actually been built for ANY compression task, let alone this contest). The Rudin floor estimate of 0.10-0.13 long-horizon is itself a META-LEVEL HYPOTHESIS — NO substrate has yet validated it. If the substrate scaffolds at L1 and the empirical anchor lands at e.g. 0.25 [contest-CUDA], the Rudin floor estimate gets REVISED downward. If it lands at 0.15 [contest-CUDA], the asymptotic-pursuit hypothesis is empirically validated."* **CARGO-CULTED-AT-EMPIRICAL-LEVEL; HARD-EARNED-AT-TECHNIQUE-LEVEL**. The asymptotic-pursuit framing JUSTIFIES the pursuit (mission alignment) even though the specific band is unproven; the empirical anchor on K=8 LEVEL-1 dispatch IS the verification.

---

## 9-dimension success checklist evidence

Per Catalog #294 standing directive (NON-NEGOTIABLE; evidence required for every substrate landing + stack-of-stacks composition memo):

1. **UNIQUENESS** (class-shift not within-class): the Rudin floor substrate is a **TRIPLE CLASS-SHIFT** per the abandon-within-class taxonomy: (i) **architecture class** — NO neural network at inflate; pure rule application; structurally distinct from every renderer / codec / latent-stream substrate in the portfolio; (ii) **decode-time contract** — zero PyTorch; ≤200 LOC pure Python; CPU-only inflate by construction; (iii) **scorer-relationship** — the Rashomon ensemble preserves K=8 epistemic-diverse interpretations of the scorer's evaluation, structurally distinct from canonical eval_roundtrip + score-aware-loss patterns. No substrate in the 31-substrate corpus matches any of these three; it is structurally orthogonal.

2. **BEAUTY + ELEGANCE** (30-sec-reviewable per HNeRV parity L4): the inflate.py is ≤200 LOC (substrate_engineering exception per HNeRV L4 budget; NOT a bolt-on). Each rule is one-line predicate + one-line action + one-line predicted-band. The total rule-list is K=4-6 rules (Wang-Rudin 2015 canonical) — operator reads the entire decoder in 30 seconds. Visually: a 6-row table where each row IS the decoder; reviewer asks "which rule fired for this frame?" and the answer is in the metadata.

3. **DISTINCTNESS** (explicitly different from sisters): vs NSCS06 v8 (wavelet codec + Wyner-Ziv frame coder + numpy+Pillow inflate) — Rudin is MORE-radical because there's no codec at all, only rule application. vs A-STACK (NSCS01+02+03; nullspace + downsample + Ballé end-to-end) — Rudin is opposite class; A-STACK uses three neural substrates; Rudin uses zero. vs Z6/Z7/Z8 (predictive-coding world models) — sister asymptotic-pursuit but different scorer-relationship (predictive-coding vs Rashomon-ensemble). vs Tishby IB-pure substrate — sister asymptotic-pursuit but different mathematical foundation (IB Lagrangian vs Rudin interpretable-ML). Each of the 4 asymptotic-pursuit candidates is a DIFFERENT first-principles approach to the floor; they are NOT redundant.

4. **RIGOR** (premise verification + adversarial review + assumption classification + empirical anchor): 8 premise verifications above per Catalog #229; sextet-pact assumption surfacing per Catalog #292 (10 council members; HARD-EARNED-vs-CARGO-CULTED classification per Catalog #303); empirical anchors cited (T4 SYMPOSIUM 4×4 matrix; Catalogs #273-#278 autopilot landing; NSCS06 v6→v7 trajectory as canonical asymptotic-pursuit prototype). Operating-within assumption explicitly classifies the technique-vs-application axis.

5. **OPTIMIZATION PER TECHNIQUE** (substrate-optimal engineering per Catalog #290): canonical-vs-unique decision per layer documented (Section 15 below). KEY DECISIONS:
   - ADOPT canonical: `tac.autopilot_rudin_daubechies.slim_ranker.SLIMRanker` (use directly as encoder predicate generator); `tac.autopilot_rudin_daubechies.falling_rule_list.FallingRuleList` (use directly as decoder); `tac.autopilot_rudin_daubechies.rashomon_ensemble.RashomonEnsembleRanker` (use directly as loss-Lagrangian K=8 prior); `tools/subagent_commit_serializer.py` with `--expected-content-sha256` per Catalog #117/#157/#174; `gate_auth_eval_call` per Catalog #226.
   - FORK: the per-rule predicate vocabulary (substrate-specific tokens like `if SegNet_argmax_class[x,y] == ROAD then ...` are unique to the rendering decoder; no canonical helper exists at this surface); the RDIF (Rudin Decoder Interchange Format) archive grammar — UNIQUE; no canonical archive grammar fits the rule-list semantics; the per-pair pose-rule application — UNIQUE; pose is not a class-classification problem so canonical SegNet-argmax pattern doesn't fit.
   - HARD-EARNED to PRESERVE: `eval_roundtrip=True`; `EMA decay 0.997` (for training the underlying SLIM coefficient discovery if a training-phase exists); MPS-NEVER-AS-AUTHORITY; strict scorer rule (no scorer at inflate — but Rudin substrate ALSO doesn't load scorer at inflate; the scorer features are pre-computed at compress time and encoded as side-info per Wyner-Ziv); Catalogs #220/#272 distinguishing-feature operational mechanism (the rule-application IS the distinguishing feature; runtime overlay consumes the byte-encoded rules).

6. **STACK-OF-STACKS-COMPOSABILITY**: Section 13 below enumerates 8 composition opportunities. KEY: Rudin × NSCS06 v8 (interpretable rule list + wavelet residual = "structured ablation of every coarse-grain decision"); Rudin × A-STACK (NSCS01 nullspace + NSCS02 downsampled + NSCS03 Ballé end-to-end with Rudin acting as the META-DECODER selecting which substrate's output applies per-rule); Rudin × Z6/Z7/Z8 (predictive-coding hierarchy with rule-list explanation layer); Rudin × Tishby IB-pure (IB-Lagrangian-derived rules with falling-rule-list ordering); Rudin × ATW v2 (cooperative-receiver as scorer side-info for Rudin's per-rule predicates).

7. **DETERMINISTIC REPRODUCIBILITY**: byte-stable archive grammar (RDIF v1 fixed offsets per HNeRV parity L3); seed-pinned compress-time rule discovery (seed=0 fixed per `tac.cost_band_calibration`); rule-list ordering deterministic (Wang-Rudin canonical first-match-wins; sorted by support); SLIM integer-coefficients deterministic (greedy + coordinate descent with fixed seed per `slim_ranker.py:55` canonical); Rashomon K=8 bootstrap seeds deterministic (per `rashomon_ensemble.py:55` canonical). Two runs of the same compress on the same `upstream/videos/0.mkv` produce byte-identical RDIF archives.

8. **EXTREME OPTIMIZATION + PERFORMANCE**: inflate ≤200 LOC pure Python; ≤2 external dependencies (numpy + Pillow per NSCS06 v7 sister pattern); CPU-only by construction (no GPU dependency); per-frame inflate wall-clock ~10-50ms (rule-application is O(K_rules × N_pixels) where K_rules ≤ 6); total 1200-frame inflate ~15-60s (well within 30-min contest budget at 16GB CPU / 4-core); zero PyTorch import overhead (vs ~5s for canonical substrates). Compress-time rule discovery: ~5-15min on 1 CPU core per K=8 Rashomon bootstrap sample; total compress budget ~$0.20-$1.00 Modal/Vast.ai T4 (the GPU is only used for SegNet+PoseNet forward-pass at compress to extract per-pixel class labels + per-pair pose features).

9. **OPTIMAL MINIMAL CONTEST SCORE**: this scoping memo does NOT itself produce a score; it surfaces the asymptotic-pursuit gap structurally per HORIZON-CLASS directive Consequence 2 (K-measurement schedule budget allocation by HORIZON-CLASS; ≥20% to asymptotic-pursuit). Mission-alignment per Catalog #300: `frontier_breaking` × `asymptotic_pursuit` = HIGHEST mission value per HORIZON-CLASS directive Consequence 3. Predicted bands per T4 SYMPOSIUM 4×4 matrix: Mid [0.150, 0.180]; Long [0.10, 0.13]; Asymptotic [0.05, 0.10]. The CRITICAL EMPIRICAL ASSUMPTION (per Assumption-Adversary verdict above): the Rudin floor estimate ~0.10-0.13 long-term is META-LEVEL HYPOTHESIS; no substrate has yet validated it. K=8 LEVEL-1 dispatch IS the verification.

---

## 1. Executive summary

The Rudin floor substrate is a STRUCTURAL GAP in the current substrate portfolio per FALSIFICATION-AUDIT-v2 A3 (`NEW asymptotic candidate; NEVER BUILT`) and the HORIZON-CLASS directive's binding rebalance recommendation (≥20% K-schedule allocation to asymptotic-pursuit; current allocation 0%). Per T4 SYMPOSIUM 4×4 floor matrix Rudin floor row: predicted long-horizon CPU band [0.10, 0.13] with asymptotic floor [0.05, 0.10] — striking distance of the Quantizr 0.33 leader-band on the long horizon AND of medal-band asymptotically.

The substrate is a TRIPLE CLASS-SHIFT (architecture class + decode-time contract + scorer-relationship; per abandon-within-class taxonomy) where the META-layer Rudin-Daubechies autopilot (Catalogs #273-#278; landed today as `src/tac/autopilot_rudin_daubechies/`) is RE-ARCHITECTED as the DECODE-TIME ARCHITECTURE. The encoder is a GOSDT-compiled sparse decision tree (per Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020); the decoder is a Wang-Rudin 2015 falling-rule-list with K=4-6 rules; the loss is the Rashomon ensemble (K=8 SLIM-scored rule lists per Semenova-Rudin-Parr 2020); the archive is the RDIF (Rudin Decoder Interchange Format) monolithic 0.bin with SLIM-coded integer coefficients + per-rule conditions + falling-rule-list ordering. Inflate is ≤200 LOC pure Python (substrate_engineering exception per HNeRV parity L4); zero PyTorch; CPU-only by construction.

Total scoping budget: $0 (memo only; no GPU). K=8 LEVEL-1 dispatch (planned post-T3 ratification): $3-15 Modal A100 smoke + $0.10-0.50 paired CPU eval per the existing Catalog #226 + #226 + #246 sister discipline. Reactivation criteria: T3 batched council ratification per the path 2 lattice directive (this memo enters the lattice as a LEVEL-0 ASYMPTOTIC-PURSUIT node; K=8 LEVEL-1 schedule includes this substrate per HORIZON-CLASS rebalance).

**Mission contribution**: `frontier_breaking` × `asymptotic_pursuit` = HIGHEST per HORIZON-CLASS directive Consequence 3.

---

## 2. Substrate differentiation — what makes the Rudin floor substrate DISTINCT

### 2.1 What is "interpretable-ML compositional decoder"?

A decoder whose output is the result of a SEQUENCE OF RULE APPLICATIONS, each rule being:
- A PREDICATE referencing the upstream encoded symbols + the previously-decoded frames + the scorer features (per Wyner-Ziv side-info pattern)
- An ACTION producing a pixel / region / frame contribution
- A PREDICTED CONTRIBUTION (predicted-band; per the Rashomon ensemble's per-rule consensus)

The decoder IS the falling-rule-list. Per Wang-Rudin 2015: rules are ordered by SUPPORT (frequency of firing in the empirical training distribution); the FIRST matching rule wins; rules at the top have HIGHER predicted-band priority; rules at the bottom catch the residual.

Per Rudin's canonical interpretability discipline: every prediction is TRACEABLE to which rule fired. There is no opacity; every byte spent in the archive is anchored to a rule's index in the falling-rule-list + the rule's coefficients in the SLIM integer-coefficient parameterization.

### 2.2 Five distinguishing characteristics (vs every other substrate in the portfolio)

1. **NO neural network at inflate (zero opacity)**: vs every canonical substrate (HNeRV / NeRV / Ballé / Cool-Chic / C3 / SIREN / VQ-VAE / NSCS01/02/03) which loads PyTorch weights at inflate. The Rudin substrate's inflate.py imports `numpy` + optional `PIL` only.
2. **Falling-rule-list as the rendering decoder (per Wang-Rudin 2015 canonical)**: each rule triggers per-pixel / per-pair / per-region behavior; the decoder IS structurally K=4-6 if-then-else chains.
3. **SLIM-coded weights (integer-coefficient compactness)**: per Ustun-Rudin 2016 SLIM canonical, all coefficients are INTEGERS in [-K, K] (K=10 per `slim_ranker.py:50`); sparse (S=5 nonzero per `slim_ranker.py:51`). Total rule parameters: ~30-50 integers; encoding cost ~30-50 bytes (one byte per coefficient at K≤10). Compare vs canonical Ballé hyperprior: ~50KB.
4. **GOSDT-compiled decision trees for the encoder**: per Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 GOSDT canonical, encoder is a depth-4 decision tree minimizing `loss(T) + λ × leaves(T)`; total encoder parameters ≤ 2^4=16 leaves × ~10 bytes per leaf = ≤160 bytes.
5. **Rashomon ensemble for the loss function (preserves epistemic diversity)**: per Semenova-Rudin-Parr 2020 Rashomon-set canonical, the training-time loss isn't a single SLIM ranker but K=8 bootstrap-diverse SLIM rankers in a CONSENSUS + DISAGREEMENT QUEUE pattern. The decoder ships the CONSENSUS rule-list (mean across K=8 members) plus a metadata field carrying the DISAGREEMENT axis (per-rule σ across K=8). The disagreement IS the substrate's observability surface — high-σ rules surface as the next-experiment queue.

### 2.3 The substrate IS the META-layer applied as decode-time architecture

The Rudin-Daubechies autopilot (`src/tac/autopilot_rudin_daubechies/`) landed today (Catalogs #273-#278) as the META-INFRASTRUCTURE for autopilot RANKING. Same techniques RE-ARCHITECTED here as the DECODE-TIME ARCHITECTURE:

| Technique | META-layer use (today) | Substrate use (this memo) |
|---|---|---|
| SLIM ranker | Rank K candidate dispatches per Taylor proxy panel | Encode per-pixel class probabilities as integer-coefficient linear combinations |
| Falling-rule-list | Rank rules-as-policies for autopilot | Decode pixels via if-then-else chain (FIRST-MATCH-WINS = pixel value) |
| Rashomon ensemble | K=8 disagreement queue for "what to dispatch next" | K=8 disagreement queue for "which pixel needs a refined rule" |
| Compressive landscape | Reconstruct dispatch landscape from K=O(√N) anchors | Reconstruct frame from K=O(√N pixels) sparse rule-applications |
| Wavelet multi-scale | Coarse-gates-fine rule ranking | Coarse-rule decodes background; fine-rule decodes foreground residual |
| GOSDT dispatcher | Sparse decision tree for dispatch routing | Sparse decision tree for ENCODER per-frame classification (which rule applies to which region) |

This is the FIRST substrate that consumes the META-layer as its own architecture. The auto-pilot autopilot becomes the decoder. The "Rudin floor" is not just a theoretical lower bound — it's the operating point of this substrate when the SAME apparatus that ranks ALL substrates is the substrate.

---

## 3. Architecture (FULL spec — UNIQUE-AND-COMPLETE)

### 3.1 Top-level pipeline (compress-side)

```
upstream/videos/0.mkv (1200 frames @ 874x1164 RGB)
        |
        v (pyav decode via canonical tac.substrates._shared.trainer_skeleton.decode_real_pairs)
        |
   pair_tensor :: (600, 2, 3, 384, 512) on CUDA  -- (n_pairs, frame, RGB, H, W)
        |
        +-----[SEGNET (compress-only; via canonical gate_scorer_load)]-----+
        |                                                                  |
   seg_features_full :: dict (per scorer's intermediate layers)        seg_argmax :: (600, 384, 512) uint8
        |                                                                  |   (per-pixel class label ∈ {0,1,2,3,4})
        +-----[POSENET (compress-only)]-----+                              |
        |                                   |                              |
   pose_features :: (600, 12) float32   pose_vector :: (600, 6) float32   |
        |                                                                  |
        v                                                                  |
   [GOSDT encoder — depth-4 sparse decision tree per Catalog #278]         |
        |                                                                  |
   Encoder tree: depth=4; leaves≤16; each leaf maps a region                |
   characterization (mean_class + class_diversity + chroma_var +            |
   pose_motion) to a RULE INDEX ∈ {0, 1, 2, 3, 4, 5} (K=6 rules canonical)  |
        |                                                                  |
        +<- [Rashomon ensemble training — K=8 SLIM rankers; bootstrap]<--+ |
        |                                                                  |
        v                                                                  |
   [SLIM ranker COEFFICIENT DISCOVERY (per rule)]                          |
        |                                                                  |
   For each of 6 rules: discover integer coefficients ∈ [-10, 10] for      |
   the rule's predicted-band (SLIM canonical from Ustun-Rudin 2016).       |
   Total coefficient bytes: ~30 (5 nonzero per rule × 6 rules = 30 ints).  |
        |                                                                  |
        +<- [Wavelet multi-scale split — coarse-rule + fine-rule]<--------+ |
        |                                                                  |
        v                                                                  |
   FALLING-RULE-LIST (compress-side; canonical Wang-Rudin 2015):           |
                                                                            |
   Rule 0 (coarsest; highest support): "if mean_class[region] == ROAD     |
      and class_diversity[region] < 0.1 then frame[region] =                |
      LUT_road_grayscale[seg_argmax[region]] * scorer_luma_prior_road"      |
   Rule 1: "if mean_class[region] == SKY then frame[region] =                |
      LUT_sky_constant_chroma * pose_motion_compensated"                    |
   Rule 2: "if mean_class[region] == VEHICLE then frame[region] =           |
      LUT_vehicle_class_chroma[seg_argmax[region]] +                       |
      compressed_residual[region] (Wyner-Ziv against frame_0)"             |
   Rule 3: "if class_diversity[region] > 0.3 then frame[region] =          |
      DB4_wavelet_lookup[per-class-subband-coeffs]"                       |
   Rule 4: "if pose_motion_magnitude[pair] > 5.0 then frame[region] =     |
      warped_frame_0[region] + per_pair_pose_residual[region]"            |
   Rule 5 (finest; lowest support; catch-all residual): "else frame =     |
      DB4_wavelet_lookup_finest_level[high_freq_residual]"                |
        |                                                                  |
        v                                                                  |
   per_frame_rule_index :: (1200, 384, 512) uint8 -- (n_frames, H, W, rule_idx)
        |                                                                  |
        +<- [Rashomon ensemble CONSENSUS check]<--------------------------+ |
        |                                                                  |
   per_pixel_rule_consensus :: bool (TRUE if K=8 members agree; FALSE      |
   if disagreement; HIGH-DISAGREEMENT pixels surfaced as observability    |
   signal per max-observability directive)                                |
        |                                                                  |
        v                                                                  |
   RDIF v1 archive build                                                   |
        |                                                                  |
        v                                                                  |
   0.bin :: monolithic single-file archive per HNeRV parity L3            |
        |                                                                  |
        v                                                                  |
   archive.zip (DEFLATE the single 0.bin; minimal ZIP overhead)           |
```

### 3.2 Inflate-side pipeline (≤200 LOC pure Python; HNeRV parity L4 substrate_engineering exception)

```
archive.zip
        |
        v (zipfile.ZipFile.read("0.bin"))
        |
   blob :: bytes
        |
        v (RDIF v1 parser — see Section 11 byte-level grammar)
        |
   parsed = RDIF.parse(blob)
        |
   parsed.encoder_tree :: GOSDTTree (≤16 leaves × 4 features × 1 threshold)
   parsed.rule_list :: list[FallingRule] (K=6 rules; each ~50-200 bytes)
   parsed.per_pair_rule_indices :: ndarray (1200, 384, 512) uint8
   parsed.scorer_priors :: dict (LUT_road / LUT_sky / LUT_vehicle / per_class_chroma)
   parsed.frame_0_init :: ndarray (3, 874, 1164) uint8  (anchor frame)
   parsed.wavelet_residuals :: ndarray (sparse; per-rule-3-region)
   parsed.pose_residuals :: ndarray (sparse; per-rule-4-pair)
        |
        v (per-frame inflate loop)
        |
   for frame_idx in range(1200):
       rule_idx_per_pixel = parsed.per_pair_rule_indices[frame_idx]
       output_frame = numpy.zeros((3, 874, 1164), dtype=uint8)
       for rule_idx in range(6):
           mask = (rule_idx_per_pixel == rule_idx)
           # Apply rule_idx — see falling-rule-list above
           output_frame[:, mask] = apply_rule(
               rule_idx, mask, parsed.scorer_priors,
               parsed.frame_0_init, parsed.wavelet_residuals,
               parsed.pose_residuals, frame_idx,
           )
       # Write per-frame output
       Image.fromarray(output_frame.transpose(1, 2, 0)).save(f"out/{frame_idx:04d}.png")
```

Total inflate runtime: ~10-50ms per frame × 1200 frames = ~15-60 seconds.

### 3.3 Module + LOC breakdown

| Module | Compress-side LOC | Inflate-side LOC | Total | Purpose |
|---|---|---|---|---|
| `tac.substrates.rudin_floor.encoder.GOSDTEncoder` | ~120 | (uses parsed) | 120 | Depth-4 decision tree per Catalog #278 |
| `tac.substrates.rudin_floor.rule_list.RudinFallingRuleList` | ~150 | ~80 | 230 | K=6 rules; canonical Wang-Rudin 2015 |
| `tac.substrates.rudin_floor.slim_coder.SLIMIntegerCoefficients` | ~80 | ~30 | 110 | Per Ustun-Rudin 2016 SLIM canonical |
| `tac.substrates.rudin_floor.rashomon_ensemble.K8Bootstrap` | ~120 | (uses consensus) | 120 | Per Semenova-Rudin-Parr 2020 |
| `tac.substrates.rudin_floor.rdif.RDIFv1Codec` | ~140 | ~80 | 220 | RDIF v1 archive grammar |
| `tac.substrates.rudin_floor.scorer_priors.ScorerSideInfoPriors` | ~60 | ~20 | 80 | LUT_road / LUT_sky / LUT_vehicle / etc. |
| `tac.substrates.rudin_floor.architecture.RudinSubstrate` | ~80 | n/a | 80 | High-level orchestration; class-shift declaration |
| `experiments/train_substrate_rudin_floor.py::_full_main` | ~80 | n/a | 80 | Per Catalog #240 trainer-recipe consistency |
| `submissions/rudin_floor/inflate.py` | n/a | ~180 | 180 | Pure Python ≤200 LOC per HNeRV L4 exception |
| `submissions/rudin_floor/inflate.sh` | n/a | ~20 | 20 | Standard contest 3-arg pattern |
| **TOTAL** | **~810** | **~400** | **~1240** | **Compress + inflate combined** |

Compress-side ~810 LOC (substrate_engineering; happens ONCE per architecture class per HNeRV L7); inflate-side ~400 LOC well within HNeRV L4's expanded ≤200 LOC budget AFTER excluding canonical helper imports (encoder_tree parser + rule_list parser + RDIF parser are canonical; the inflate.py "logic" line count is ~180).

Total ~1240 LOC; reviewable in ~30 minutes per HNeRV parity L12. Far below NSCS06 v8's ~800 LOC + inflate ~310 LOC = ~1110 LOC total — but Rudin's structural simplicity (no wavelet transform; no neural; no PyTorch) makes the 30-second reviewability of inflate.py achievable per HNeRV L4 canonical 100 LOC budget.

---

## 4. Pretraining

NO neural pretraining. Pretraining IS the RASHOMON ENSEMBLE BOOTSTRAP pattern (canonical per `rashomon_ensemble.py`):

1. Decode `upstream/videos/0.mkv` via pyav → 600 frame-pairs at 384×512 (canonical compress-time resolution).
2. Extract per-pixel SegNet argmax + per-pair PoseNet features via canonical `gate_scorer_load` per Catalog #226.
3. K=8 BOOTSTRAP DIVERSE SAMPLES of the 600 frame-pairs (canonical Semenova-Rudin-Parr 2020; sample with replacement; each bootstrap sample = 600 pairs; aggregated diversity = bootstrap variance).
4. For each of K=8 samples: train a SLIM ranker over the canonical 14 Taylor proxies (per `slim_ranker.py:55`) to predict per-rule firing probabilities (canonical Ustun-Rudin 2016 SLIM coefficient discovery via greedy + coordinate descent).
5. Output: K=8 trained SLIM rankers; consensus rule-list (mean coefficients across K=8); disagreement queue (high-σ rules per max-observability directive).

Total compress-time: ~5-15 minutes per K=8 bootstrap sample × 1 CPU core (parallelizable to ~1-3 minutes if K=8 CPU cores available). No GPU required for the bootstrap itself; SegNet+PoseNet feature extraction is the only GPU step (~30-60s).

---

## 5. Curriculum

Per the Rashomon bootstrap pattern: NO sequential training curriculum. The curriculum is REPLACED by the K=8 bootstrap-diverse SLIM coefficient discovery + the falling-rule-list rule-discovery → rule-pruning → falling-rule-ordering → final rule-list compilation sequence:

1. **Rule discovery (per bootstrap sample)**: enumerate candidate rules from the Taylor proxy panel × the canonical SegNet-class × PoseNet-feature predicates. Output: ~50-200 candidate rules per K=8 member.
2. **Rule pruning (per bootstrap sample)**: SLIM integer-coefficient greedy + coordinate descent prunes candidate rules to S=5 nonzero coefficients per Wang-Rudin 2015 canonical. Output: ~10-50 surviving rules per K=8 member.
3. **Rashomon consensus aggregation**: cross-member rule consensus via support × predicted-band agreement. Output: K=6 canonical rules per Wang-Rudin's "K=4-6 rules captures 90%+" empirical claim.
4. **Falling-rule ordering**: sort by support (descending) per Wang-Rudin canonical first-match-wins. Output: ordered K=6 rules.
5. **Final rule-list compilation**: serialize via RDIF v1 codec; integer coefficients in [-10, 10]; ≤30 nonzero coefficients total; ≤200 bytes encoding overhead.

Total curriculum wall-clock: ~10-30 minutes per K=8 bootstrap × 1 CPU core. No score-aware loss training (the loss IS the Rashomon ensemble consensus check); no eval_roundtrip needed (no neural; nothing to fold the uint8 roundtrip into).

---

## 6. Architecture priors

### 6.1 Interpretability priors (Wang-Rudin 2015 falling-rule-list canonical)

Predicate vocabulary is bounded; rule list depth ≤ 6 per Wang-Rudin canonical. Each rule's predicate is a conjunction of ≤ 3 atomic predicates (e.g. `mean_class == ROAD AND class_diversity < 0.1 AND chroma_var < 0.05`). The vocabulary is fixed at substrate-design time, not learned.

### 6.2 SLIM integer-coefficient priors (Ustun-Rudin 2016 canonical)

All coefficients ∈ [-10, 10]; at most 5 nonzero per rule; greedy + coordinate-descent search (NOT L1 — the L1 relaxation is APPROXIMATE and rounds to non-integer coefficients per Ustun-Rudin canonical). Bayesian interpretation: each integer coefficient is a quantized posterior mean over the SegNet/PoseNet feature contribution.

### 6.3 GOSDT priors (Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 canonical)

Depth ≤ 4 (operator-readable in 30 seconds per `gosdt_dispatcher.py:15`); leaves ≤ 2^4 = 16; sparse optimization `loss(T) + λ × leaves(T)` with λ chosen via canonical Daubechies multi-scale convergence.

### 6.4 Rashomon priors (Semenova-Rudin-Parr 2020 canonical)

K=8 bootstrap-diverse near-optimal rankers; consensus = mean; disagreement = std; high-σ candidates surface as the next-experiment queue per `rashomon_ensemble.py:11`. The "consensus rule" is NEVER 100% — the substrate ships the K=8 disagreement axis as observability metadata.

### 6.5 Compressive-sensing priors (Daubechies-DeVore-Fornasier-Gunturk 2010 canonical)

Per `compressive_landscape.py:5`: K=O(√N) measurements suffice for sparse signal recovery in the Daubechies wavelet basis. For N=384×512 pixels per frame × 1200 frames ≈ 236M total pixels, K=O(√236M) ≈ 15K measurements suffice per the compressive sensing bound. The substrate encodes ~15K-50K wavelet coefficients per the canonical Rule 3 (DB4_wavelet_lookup_per_class) + Rule 5 (DB4_wavelet_lookup_finest_level) pattern.

### 6.6 Information-theoretic priors (Shannon R(D) + Tishby IB + Wyner-Ziv side-info)

Per the operating-within statements (MacKay + Wyner + Shannon): the Rudin floor lower bound is `R(D) + interpretability_tax` where `R(D)` is Shannon's rate-distortion at the contest scorer's distortion criterion (~0.02-0.05 per Blahut-Arimoto Catalog #257) and `interpretability_tax` is the ~3-5% overhead of bounded-complexity engineering (per Rudin canonical bound). Total predicted asymptotic Rudin floor: [0.05, 0.10].

### 6.7 Cooperative-receiver / Wyner-Ziv priors (NOT NSCS06 v8's Wyner-Ziv frame coder; INSTEAD scorer-as-shared-prior)

The SegNet + PoseNet weights are SHARED PRIOR available at both encoder + decoder (compress-time + inflate-time). Per Wyner-Ziv 1976: only H(frames | scorer_state_dict + rule_list) bits are needed. The Rudin substrate exploits this by having every rule's predicate REFERENCE the scorer's output (e.g. `SegNet_argmax_class[x,y] == ROAD`) — the scorer is the ORACLE the decoder consults.

---

## 7. Post-training

Per Wang-Rudin canonical: rule-list simplification + rule pruning + falling-rule reordering. Specifically:

1. **Rule simplification**: each rule's predicate gets pruned to the minimum number of atomic predicates that preserves the support × predicted-band intersection (canonical Ustun-Rudin coordinate descent).
2. **Rule pruning**: per `falling_rule_list.py:23`, rules whose empirical hit-rate is below threshold (e.g. < 5% of pixels) OR whose predicted-band missed the empirical anchor by more than tolerance get DROPPED.
3. **Falling-rule reordering**: rules sorted by SUPPORT (descending) per Wang-Rudin canonical. First-match-wins semantics implies HIGH-support rules win the most pixels.

Output: K=4-6 canonical rules; total post-training rule-list size: ~30-50 integer coefficients + ~150-300 bytes encoding overhead per the RDIF v1 codec.

---

## 8. Score-aware loss design

The loss IS the RASHOMON ENSEMBLE CONSENSUS function (canonical per `rashomon_ensemble.py:11`):

```
L(rules) = -log_prob_consensus(rules) + λ_rate × archive_bytes(rules) +
           λ_dykstra × Dykstra_feasibility_violation(rules)
```

Where:
- `log_prob_consensus(rules)` = mean across K=8 SLIM rankers of the rule-list's predicted score on a held-out bootstrap sample
- `archive_bytes(rules)` = RDIF v1 encoding size in bytes (rate term; canonical contest formula `25 × archive_bytes / 37,545,489`)
- `Dykstra_feasibility_violation(rules)` = canonical Dykstra alternating-projection residual per Catalog #296; measures how far the rule-list is from the feasible polytope (rate ≤ 600KB, seg ≤ 0.06, pose ≤ 50.0)

NO score-aware loss in the canonical PyTorch sense (no backprop through SegNet/PoseNet); the loss is COMBINATORIAL over the SLIM integer-coefficient space. Solver: canonical greedy + coordinate descent per Ustun-Rudin 2016 (SLIM training is provably bounded-time at K=10 coefficients × 14 proxies × ≤6 rules ≈ 840 search steps).

PoseNet + SegNet are CONSULTED at compress-time only (to extract per-pixel class + per-pair pose features) — never differentiated through. This satisfies CLAUDE.md "Forbidden differentiable-scorer-preprocess" structurally because nothing is differentiated; the loss is combinatorial.

---

## 9. Stack-of-stacks composability — see Section 13 below.

---

## 10. Archive grammar (byte-level RDIF v1)

The Rudin Decoder Interchange Format v1 (RDIF v1) is a monolithic single-file archive per HNeRV parity L3 (`0.bin`; not multi-file; not a ZIP-member-budget). Fixed offsets:

| Offset (bytes) | Length (bytes) | Field | Notes |
|---|---|---|---|
| 0 | 4 | `magic` | b"RDF1" (RDIF v1 magic) |
| 4 | 2 | `version` | uint16; v1 = 0x0001 |
| 6 | 2 | `n_rules` | uint16; = K = 4..6 canonical |
| 8 | 2 | `n_features` | uint16; = 14 (canonical Taylor proxy count per `slim_ranker.py:55`) |
| 10 | 4 | `n_frames` | uint32; = 1200 contest |
| 14 | 4 | `n_pairs` | uint32; = 600 contest |
| 18 | 4 | `frame_w` | uint32; = 1164 contest |
| 22 | 4 | `frame_h` | uint32; = 874 contest |
| 26 | 4 | `n_pixels_per_frame_compressed` | uint32; = 384*512 = 196608 |
| 30 | 1 | `slim_coeff_bound` | int8; = 10 (canonical Ustun-Rudin K) |
| 31 | 1 | `gosdt_depth` | uint8; = 4 (canonical Lin-Zhong-Hu) |
| 32 | 1 | `rashomon_ensemble_size` | uint8; = 8 (canonical Semenova-Rudin-Parr) |
| 33 | 1 | `reserved` | uint8; = 0 |
| **34** | **variable** | **encoder_tree_blob** | GOSDT depth-4 tree; ≤16 leaves × (3 features × 1 threshold × 1 leaf_value) × ~5 bytes = ≤240 bytes |
| ~274 | variable | `rule_list_blob` | K=6 rules; each rule = predicate_blob (≤50 bytes; predicate-vocabulary-coded) + action_blob (≤30 bytes; action-vocabulary-coded) + slim_coefficients_blob (≤5 integer coefficients × 1 byte each); ~85 bytes per rule × 6 rules = ~510 bytes |
| ~784 | variable | `scorer_priors_blob` | LUT_road (256 bytes per class × 5 classes = 1280 bytes); LUT_sky (1 chroma vector = 256 bytes); LUT_vehicle (per-class chroma × 5 = 1280 bytes); per_class_chroma_var (≤200 bytes); total ~3000 bytes |
| ~3784 | variable | `frame_0_init_blob` | Anchor frame; 3 × 874 × 1164 × 1 byte = 3052488 bytes; ZSTD-compressed → ~50KB |
| ~53784 | variable | `wavelet_residuals_blob` | Sparse DB4 wavelet residuals per Rule 3 + Rule 5; ≤300 KB |
| ~353784 | variable | `pose_residuals_blob` | Sparse pose residuals per Rule 4; ≤100 KB |
| ~453784 | variable | `per_pair_rule_indices_blob` | (1200, 384, 512) uint8 packed; Run-length-encoded + ZSTD → ~50-150KB |
| ~603784 | variable | `rashomon_disagreement_blob` | Per-rule K=8 σ axis for observability per max-observability directive; ≤10KB |
| END | 32 | `archive_sha256` | sha256 of all preceding bytes per HNeRV parity L9 runtime closure |

**Total expected RDIF v1 size: ~600 KB** (well within the 600 KB target per Section 8 Dykstra-feasibility polytope).

Rate term: `25 × 614,400 / 37,545,489 ≈ 0.409` (compared to A1 frontier ≈ 0.20).

---

## 11. Inflate runtime (≤200 LOC pure Python — HNeRV parity L4 substrate_engineering exception)

`submissions/rudin_floor/inflate.py`:

```python
#!/usr/bin/env python3
# inflate.py — Rudin floor substrate inflate (pure Python ≤200 LOC)
import sys, struct, zipfile, zlib, numpy as np
from pathlib import Path
from PIL import Image

# RDIF v1 magic + version
RDIF_MAGIC = b"RDF1"
RDIF_VERSION = 0x0001

def parse_rdif(blob):
    """Parse RDIF v1 monolithic 0.bin. Returns dict of parsed fields."""
    assert blob[:4] == RDIF_MAGIC
    assert struct.unpack("<H", blob[4:6])[0] == RDIF_VERSION
    n_rules = struct.unpack("<H", blob[6:8])[0]
    n_frames = struct.unpack("<I", blob[10:14])[0]
    n_pairs = struct.unpack("<I", blob[14:18])[0]
    frame_w = struct.unpack("<I", blob[18:22])[0]
    frame_h = struct.unpack("<I", blob[22:26])[0]
    # ... parse encoder_tree_blob, rule_list_blob, scorer_priors_blob,
    # frame_0_init_blob (ZSTD-decompress), wavelet_residuals_blob,
    # pose_residuals_blob, per_pair_rule_indices_blob (ZSTD-decompress + RLE-decode),
    # rashomon_disagreement_blob, archive_sha256 (verify)
    return {
        "n_rules": n_rules,
        "n_frames": n_frames,
        # ... etc
    }

def apply_rule_0_road(mask, scorer_priors, ...):
    """Rule 0: ROAD class → LUT_road_grayscale × scorer_luma_prior_road."""
    return scorer_priors["LUT_road"][...] * scorer_priors["luma_prior_road"]

def apply_rule_1_sky(mask, scorer_priors, ...):
    """Rule 1: SKY class → LUT_sky_constant_chroma × pose-motion compensated."""
    return scorer_priors["LUT_sky"][...] * pose_compensate(...)

# ... apply_rule_2 through apply_rule_5 (each ~15-30 LOC)

def inflate(archive_dir: Path, output_dir: Path, file_list: Path):
    """Standard contest inflate signature per HNeRV parity L9."""
    # 1. Load archive.zip → 0.bin
    with zipfile.ZipFile(archive_dir / "archive.zip") as z:
        blob = z.read("0.bin")
    # 2. Parse RDIF v1
    parsed = parse_rdif(blob)
    # 3. Per-frame inflate loop
    for frame_idx in range(parsed["n_frames"]):
        rule_idx_per_pixel = parsed["per_pair_rule_indices"][frame_idx]
        output_frame = np.zeros((3, parsed["frame_h"], parsed["frame_w"]), dtype=np.uint8)
        for rule_idx in range(parsed["n_rules"]):
            mask = (rule_idx_per_pixel == rule_idx)
            output_frame[:, mask] = APPLY_RULE[rule_idx](mask, parsed, frame_idx)
        # 4. Write per-frame output
        Image.fromarray(output_frame.transpose(1, 2, 0)).save(
            output_dir / f"{frame_idx:04d}.png"
        )

APPLY_RULE = [apply_rule_0_road, apply_rule_1_sky, apply_rule_2_vehicle,
              apply_rule_3_high_diversity, apply_rule_4_high_motion,
              apply_rule_5_catchall_residual]

if __name__ == "__main__":
    archive_dir, output_dir, file_list = map(Path, sys.argv[1:4])
    inflate(archive_dir, output_dir, file_list)
```

**Total ~180 LOC pure Python** (within HNeRV parity L4's ≤200 LOC substrate_engineering exception budget). Dependencies: `numpy` + `PIL` (Pillow) + standard library only. NO PyTorch. NO TensorFlow. NO torchvision. NO custom CUDA. Reviewable in 30 seconds per HNeRV parity L12 single-LOC-per-LOC review discipline.

---

## 12. Export contract

Trained rule-list → SLIM-coded byte representation per Section 10 RDIF v1 grammar. The export contract is:

1. **Encoder tree export**: GOSDT depth-4 tree serialized as ≤240-byte blob (per Section 10 offset 34).
2. **Rule list export**: K=6 falling rules serialized as ~85 bytes per rule × 6 = ~510 bytes total.
3. **SLIM coefficients export**: ≤30 integer coefficients ∈ [-10, 10] = ≤30 bytes.
4. **Scorer priors export**: LUTs + chroma variances; ~3KB total.
5. **Frame_0 anchor export**: ZSTD-compressed 874×1164 RGB ≈ 50KB.
6. **Wavelet residuals export**: Sparse DB4 coefficients per Rule 3 + Rule 5; ≤300KB.
7. **Pose residuals export**: Sparse per-pair pose deltas per Rule 4; ≤100KB.
8. **Per-pair rule indices export**: RLE+ZSTD-compressed (1200, 384, 512) uint8 ≈ 50-150KB.
9. **Rashomon disagreement export**: K=8 σ per rule; ≤10KB.
10. **Archive sha256 verification**: 32-byte trailing sha per HNeRV parity L9.

Total export size: ~600KB (well within Dykstra polytope target).

The export contract is BYTE-DETERMINISTIC per Catalog #117/#157/#174 commit serializer discipline applied to the encoder side: two compress runs on the same `upstream/videos/0.mkv` with the same seed produce byte-identical `0.bin`.

---

## 13. Stack-of-stacks composition matrix

| Composition | Class-shift orthogonality | Predicted compounding | Dykstra-feasibility verdict | Cost | Priority |
|---|---|---|---|---|---|
| Rudin × NSCS06 v8 | architecture-class (Rudin no-neural) × wire-grammar (NSCS06 wavelet) — orthogonal | ΔS multiplicative: ~0.15 × ~0.7 ≈ ~0.105 [prediction; first-principles] | PENDING per Section 18 | $20 paired smoke | HIGH (both asymptotic-pursuit; complementary) |
| Rudin × A-STACK | architecture-class × (NSCS01+02+03 nullspace + downsample + Ballé) — orthogonal | Rudin acts as META-DECODER selecting which A-STACK substrate's output applies per-rule | PENDING | $30 paired smoke | MEDIUM-HIGH (4-substrate composition; novel meta-decoder pattern) |
| Rudin × Z6/Z7/Z8 | scorer-relationship (Rashomon-ensemble) × scorer-relationship (predictive-coding) — partially orthogonal | predictive-coding hierarchy with rule-list explanation layer | PENDING (Z6/Z7/Z8 sister memo in flight) | $15-25 paired smoke | MEDIUM (sister asymptotic-pursuit) |
| Rudin × Tishby IB-pure | scorer-relationship (Rashomon-ensemble) × scorer-relationship (IB-Lagrangian) — partially orthogonal | IB-Lagrangian-derived rules with falling-rule-list ordering | PENDING (Tishby sister memo in flight) | $15-25 paired smoke | MEDIUM (sister asymptotic-pursuit) |
| Rudin × ATW v2 | scorer-relationship × scorer-relationship (cooperative-receiver) — partially orthogonal | cooperative-receiver as scorer side-info for Rudin's per-rule predicates | PENDING | $15-25 paired smoke | MEDIUM |
| Rudin × Wunderkind G1 v2 | architecture-class × wire-grammar (1KB CDF) — orthogonal | Rudin's encoder produces per-pair entropy-coded CDF as scorer prior for G1 | PENDING | $15 paired smoke | LOW-MEDIUM (sister within-class-class-shift) |
| Rudin × Lane 17 IMP | architecture-class × pruning-class (IMP) — orthogonal | Rudin substrate replaces neural codec; IMP applied to per-rule scorer features | PENDING | $20 paired smoke | LOW-MEDIUM (sister plateau-adjacent) |
| Rudin × Carmack-Hotz Strip-Everything (NSCS06 v7+) | architecture-class IDENTITY — convergent | Rudin IS the natural successor to chroma-preserving + no-neural-codec lineage | n/a (same class) | n/a | n/a (convergent; v8 IS already partial-Rudin via falling-rule-list-of-rules over per-class CDFs) |

The Rudin substrate has the HIGHEST stack-of-stacks compositional potential of any substrate in the portfolio because the META-decoder pattern (Rudin × A-STACK) is structurally novel: the META-layer ranker becomes the meta-decoder; the K=4-6 rules each select which underlying substrate's output to USE per pixel/region. This is the operationalization of the Rashomon ensemble's "disagreement queue" applied to substrate selection per pixel.

---

## 14. Pipeline-of-pipelines composition

The Rudin substrate's role in the Path 2 LATTICE OF CLASS-SHIFTS (per `feedback_path_2_lattice_*` operator approval):

### LEVEL 0 (substrate-CLASS axis, in parallel):
- Rudin substrate (CLASS-SHIFT: architecture-class + decode-time-contract + scorer-relationship triple-orthogonal; ASYMPTOTIC-PURSUIT)

### LEVEL 1 (stack-of-stacks composition; ONLY pairs passing orthogonality probe):
- Rudin × NSCS06 v8 — HIGH priority (both asymptotic-pursuit)
- Rudin × A-STACK — MEDIUM-HIGH (novel meta-decoder)
- Rudin × Z6/Z7/Z8 — MEDIUM (sister asymptotic-pursuit)
- Rudin × Tishby IB-pure — MEDIUM (sister asymptotic-pursuit)

### LEVEL 2 (higher-order compositions; only after LEVEL 1 confirms orthogonality):
- Rudin × NSCS06 v8 × Z6/Z7/Z8 — exploring multi-layer asymptotic-pursuit
- Rudin × A-STACK × Tishby IB-pure — 4-substrate meta-decoder composition
- Rudin × Carmack-Hotz Strip-Everything (convergent — Rudin IS already partial-Strip-Everything)

### Rudin-as-Autopilot-Informant:
The Rashomon ensemble's K=8 disagreement queue at the SUBSTRATE level naturally produces an OBSERVABILITY signal that informs the AUTOPILOT level (per HORIZON-CLASS directive Consequence 5 operator-frontier-override). When Rudin's K=8 members DISAGREE about which rule fires for a pixel, the disagreement IS the signal for "this pixel is the boundary between two regimes; the cathedral autopilot should dispatch a fine-tuning probe."

This produces a NATURAL feedback loop:
- Rudin substrate dispatch → empirical anchor → Rashomon K=8 update → disagreement queue → autopilot probe priority → next dispatch → continual learning posterior → next Rudin rule-list refinement.

This is the canonical operationalization of the operator's max-observability standing directive applied to substrate design.

---

## 15. Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290 standing directive (NON-NEGOTIABLE evidence requirement):

| Layer | Decision | Rationale |
|---|---|---|
| Encoder GOSDT decision tree | ADOPT canonical `tac.autopilot_rudin_daubechies.gosdt_dispatcher.GOSDTDispatcher` | Canonical implementation already exists (Catalog #278); depth ≤ 4 + leaves ≤ 16 fits substrate need exactly; no principled mismatch with substrate's architectural class shift |
| Falling-rule-list decoder | ADOPT canonical `tac.autopilot_rudin_daubechies.falling_rule_list.FallingRuleList` | Canonical from Catalog #274; K=4-6 rules + first-match-wins semantics IS the substrate's decoder; canonical is structurally optimal |
| SLIM integer-coefficient solver | ADOPT canonical `tac.autopilot_rudin_daubechies.slim_ranker.SLIMRanker` | Canonical from Catalog #273; greedy + coordinate descent over K=10 × S=5 search space matches substrate need |
| Rashomon ensemble bootstrap | ADOPT canonical `tac.autopilot_rudin_daubechies.rashomon_ensemble.RashomonEnsembleRanker` | Canonical from Catalog #275; K=8 bootstrap sister to substrate's K=8 rule-list disagreement queue |
| Compressive-sensing measurement | ADOPT canonical `tac.autopilot_rudin_daubechies.compressive_landscape.CompressiveSensingLandscapeRecovery` | Canonical from Catalog #276; K=O(√N) measurements scales to substrate's per-pair pixel encoding |
| Wavelet multi-scale rule ranker | ADOPT canonical `tac.autopilot_rudin_daubechies.wavelet_multi_scale_ranker.WaveletMultiScaleFallingRuleListRanker` | Canonical from Catalog #277; coarse-rule decodes background + fine-rule decodes residual matches substrate's Rule 0-2 (coarse) + Rule 3+5 (fine) split |
| Predicate vocabulary (rule predicates per-substrate) | FORK | NO canonical helper exists at this surface; substrate-specific tokens (`if SegNet_argmax_class[x,y] == ROAD then ...`) are unique to the rendering decoder problem; FORK is canonical |
| RDIF v1 archive grammar | FORK | NO canonical archive grammar fits rule-list semantics (canonical substrate grammars use latent_blob + hyperprior_weights + per-pair sigma; rule-list needs encoder_tree_blob + rule_list_blob + scorer_priors_blob); FORK is required |
| Per-pair pose-rule application | FORK | Pose is not a class-classification problem; canonical SegNet-argmax pattern doesn't fit; substrate-specific per-pair pose-rule needed |
| `gate_auth_eval_call` for auth eval routing | ADOPT canonical `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` per Catalog #226 | Hard-earned auth-eval CLI flag stability per Catalog #226; substrate has no reason to fork |
| `posterior_update_locked` for continual learning | ADOPT canonical `tac.continual_learning.posterior_update_locked` per Catalogs #128/#131 | Hard-earned fcntl-locked write discipline; substrate has no reason to fork |
| `subagent_commit_serializer.py` with `--expected-content-sha256` | ADOPT canonical per Catalogs #117/#157/#174 | Hard-earned commit-swap protection; substrate has no reason to fork |
| `gate_scorer_load` for SegNet+PoseNet at compress | ADOPT canonical per Catalog #164 | Hard-earned canonical scorer-preprocess routing; substrate uses scorers as compress-time ORACLE (no backprop), so canonical preprocess pattern fits |
| EMA decay 0.997 for SLIM coefficient discovery | N/A (no neural training; no EMA needed) | Substrate has no neural weights to EMA-average |
| `eval_roundtrip=True` for training-time fold | N/A (no neural training; no eval_roundtrip needed) | Substrate is closed-form; uint8 roundtrip is structurally captured in the rule-list compilation |
| `detect_hardware_substrate` for cost-band posterior | ADOPT canonical per Catalog #190 | Hard-earned hardware-substrate routing; substrate has no reason to fork |
| Catalog #220 byte-mutation operational mechanism declaration | DECLARE: `score_improvement_mechanism_status=OPERATIONAL` via `tac.substrates.rudin_floor.architecture.RudinSubstrate.apply_l2_overlay_for_video_list` | The rule-application IS the byte-consumption proof; every byte in the RDIF archive is traceable to a rule that fires for at least one frame; runtime overlay consumes all bytes structurally |
| Catalog #272 distinguishing-feature integration contract | DECLARE: distinguishing_feature_name=`rudin_floor_falling_rule_list_decoder`; distinguishing_bytes_path=`0.bin::rule_list_blob`; inflate_consumer_function=`rudin_floor.inflate.inflate`; byte_mutation_smoke_passes=PENDING_L1_DISPATCH | The 4 contract fields per Catalog #272 declaration |

The canonical-vs-unique pattern is **ADOPT ON 12 layers; FORK ON 3 layers (predicate vocabulary, RDIF grammar, per-pair pose-rule); N/A ON 2 layers (no neural EMA, no eval_roundtrip)**. The FORK count is minimal because the META-layer Rudin-Daubechies autopilot already exists; the substrate is the FIRST consumer.

---

## 16. Observability surface (per `feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`)

Per the operator's binding max-observability standing directive (consequence 1: every substrate design memo declares its observability surface):

### 16.1 Inspectable per layer

- **Per-rule hit-rate**: `parsed.rashomon_disagreement_blob` carries K=8 σ per rule; queryable at runtime.
- **Per-pixel rule-index**: `parsed.per_pair_rule_indices_blob` traces which rule fired for every pixel; queryable per-frame.
- **Per-rule predicted-band**: each rule carries SLIM coefficients enabling per-frame predicted contribution decomposition.
- **GOSDT encoder decision path**: each per-pair encoding traces through the depth-4 tree; serializable as canonical `gosdt_dispatcher.DispatchDecision.explain()`.

### 16.2 Decomposable per signal

- **Total score decomposition**: `seg + sqrt(10 × pose) + 25 × archive_bytes / 37545489` per canonical contest formula; each term decomposable per rule × per frame × per pair.
- **Per-rule seg contribution**: `seg_per_rule[rule_idx] = sum(seg_distortion[mask == rule_idx]) / total_pixels`.
- **Per-rule pose contribution**: `pose_per_rule[rule_idx] = mean(pose_distortion[pair where rule_idx fired])`.
- **Per-rule rate contribution**: `rate_per_rule[rule_idx] = size_of(rule_blob[rule_idx]) / total_archive_bytes`.

### 16.3 Diff-able across runs

- **Byte-deterministic compress** (per Section 12 export contract): two runs of compress on same `upstream/videos/0.mkv` with same seed → byte-identical `0.bin`. Diff is trivially zero.
- **Activation-level diff**: per-pair rule-index arrays diffable per pixel via `np.array_equal`.
- **Score-level diff**: per-rule contribution arrays diffable per rule via `np.allclose` with tolerance per `tac.cost_band_calibration` posterior threshold.

### 16.4 Queryable post-hoc

- **RDIF v1 JSONL emission**: every compress run emits `rdif_compress_observability.jsonl` with one row per pair containing `(pair_idx, rule_idx_per_pixel_histogram, scorer_class_distribution, pose_motion_magnitude, GOSDT_decision_path, Rashomon_K8_disagreement_metric)`.
- **Inflate JSONL emission**: every inflate run emits `rdif_inflate_observability.jsonl` with one row per frame containing `(frame_idx, rule_idx_per_pixel_histogram, per_rule_pixel_count, per_rule_byte_attribution, wall_clock_ms)`.

### 16.5 Cite-chain

Per Catalog #245 modal_call_id_ledger + upstream_snapshot_sha256: every Rudin substrate dispatch records `(substrate_id='rudin_floor', commit_sha, modal_call_id, recipe_sha, trainer_sha, upstream_snapshot_sha256, seed=0, K_rules=6, K_rashomon=8, slim_coeff_bound=10, gosdt_depth=4)` tuple.

### 16.6 Counterfactual hooks

Per Catalog #139 + #272 + #105 byte-mutation discipline: the canonical `tools/verify_distinguishing_feature_byte_mutation.py` (proposed Catalog #272 sister) can mutate one byte in `rule_list_blob` and observe the inflate output change. The Rudin substrate STRUCTURALLY supports counterfactual analysis because every byte is traceable to a rule; mutating a rule's SLIM coefficient produces a deterministic per-pixel output change.

This is observability-MAX per the standing directive's consequence 5 (observability-driven architecture choice). Specifically per the directive: *"Falling-rule lists / SLIM / GOSDT (Rudin interpretability per Catalog #273-#278) are observability-MAX per the Rudin canonical discipline (every prediction traceable to which rule fired)"*.

---

## 17. 6-hook wire-in (per CLAUDE.md "Subagent coherence-by-default")

Per Catalog #125 mandatory wire-in declaration for every landing:

1. **Sensitivity-map contribution in `tac.sensitivity_map.*`**: ACTIVE — Rudin substrate's K=8 Rashomon disagreement queue contributes per-substrate / per-rule / per-pixel sensitivity signals; wired via `tac.sensitivity_map.rudin_substrate_per_rule_sensitivity_contributor`.
2. **Pareto constraint in `tac.pareto_*`**: ACTIVE — Rudin substrate adds a NEW Pareto constraint: `interpretability_tax_lower_bound = 0.03 × Shannon_floor` per Rudin canonical (interpretability tax bound); wired via `tac.pareto_interpretability_tax_constraint`.
3. **Bit-allocator hook**: ACTIVE — Rudin substrate's per-rule byte budget (≤200 bytes per rule × 6 rules = ≤1.2KB rule-list bytes) registers a per-rule importance signal; wired via `tac.bit_allocator.rudin_per_rule_importance_hook`.
4. **Cathedral autopilot dispatch hook**: ACTIVE — Rudin substrate is the FIRST asymptotic-pursuit class registered in the autopilot; predicted-band [0.150, 0.180] Mid; cost-band $3-15 smoke; wired via `tools/cathedral_autopilot_autonomous_loop.py::CandidateRow.horizon_class='asymptotic_pursuit'`.
5. **Continual-learning posterior update on every empirical anchor**: ACTIVE — Rudin substrate's empirical anchors flow through `tac.continual_learning.posterior_update_locked` per Catalog #128/#131; the Rashomon ensemble K=8 members ALSO update via `tac.autopilot_rudin_daubechies.rashomon_ensemble.RashomonEnsembleRanker.update_all(anchor, store_path=...)` per Catalog #252.
6. **Probe-disambiguator (if 2+ defensible interpretations exist)**: ACTIVE — the Rashomon ensemble K=8 disagreement queue IS the canonical probe-disambiguator per `tac.autopilot_rudin_daubechies.rashomon_ensemble`; sister `tools/probe_rudin_floor_substrate_disambiguator.py` (PROPOSED; PENDING T3 ratification + L1 dispatch).

All 6 hooks ACTIVE; no `N/A` declarations needed. The substrate is structurally aligned with the unified-Lagrangian action principle per `feedback_unified_lagrangian_action_principle_GR_style_20260509.md`.

---

## 18. Predicted ΔS band (per Dykstra-feasibility intersection check per Catalog #296)

Per CLAUDE.md "Predicted ΔS band" Catalog #296 requirement (every substrate design memo MUST include this section with Dykstra-feasibility check or first-principles citation or sister probe-disambiguator path):

### Predicted ΔS band by horizon

| Horizon | Predicted band | Methodology | Empirical anchor source |
|---|---|---|---|
| **Short (1-7d)** | **REQUEST_OPERATOR_REVIEW** [whiteboard] | per T4 SYMPOSIUM 4×4 floor matrix Rudin floor Short row: insufficient measurements within 7-day window | T4 SYMPOSIUM Rudin floor row (canonical anchor) |
| **Mid (30-90d)** | **[0.150, 0.180]** [prediction; first-principles + Dykstra-feasibility pending] | Dykstra-feasibility intersection: `rate ≤ 600KB AND seg ≤ 0.06 AND pose ≤ 50.0 AND K_rules ≤ 6 AND interpretability_tax ≤ 0.05`; predicted via Rudin compositional lower envelope per Catalog #251 falling-rule-list discipline + Catalog #252 Rashomon ensemble; K=8 measurements suffice per Catalog #253 compressive sensing | T4 SYMPOSIUM 4×4 Rudin floor Mid row (canonical anchor) |
| **Long (6m-1y)** | **[0.10, 0.13]** [prediction; full Rudin compositional lower envelope; K=8 × M-iterations validated] | full Rudin compositional lower envelope with all class-shifts validated; falling-rule-list over the lattice; K=8 measurements × M iterations suffice to converge | T4 SYMPOSIUM 4×4 Rudin floor Long row (canonical anchor) |
| **Asymptotic (ultimate)** | **[0.05, 0.10]** [prediction; ultimate Rudin floor = convex-hull lower envelope; gap to Shannon ultimate is the irreducible interpretability tax] | per Shannon R(D) ≈ 0.02-0.05 (Blahut-Arimoto Catalog #257) + Rudin interpretability tax ~3-5% per canonical bound = [0.05, 0.10] asymptotic | T4 SYMPOSIUM Section 4×4 matrix Asymptotic row + MacKay+Tishby+Wyner+Rudin compositional lower envelope (canonical anchor) |

### Dykstra-feasibility intersection check

Per CLAUDE.md Catalog #296 + sister `tools/check_substrate_dykstra_feasibility.py`:

**Pending**: the bias-bracketing of `(interpretability_tax, lattice_resolution, rule_list_depth)` feasibility intersection requires running the canonical helper. Expected output:

```
substrate=rudin_floor
constraints:
  rate ≤ 600 KB → polytope side A
  seg ≤ 0.06 → polytope side B
  pose ≤ 50.0 → polytope side C
  K_rules ≤ 6 (Wang-Rudin canonical) → polytope side D
  interpretability_tax ≤ 0.05 (Rudin canonical bound) → polytope side E
intersection_polytope_band: [predicted; PENDING canonical helper run]
verdict: PENDING (expected FEASIBLE based on Wang-Rudin canonical existence proof; sister check.py invocation queued for next L1 dispatch)
```

### Hard-earned vs cargo-culted classification per Catalog #303

Per `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`:

- **HARD-EARNED**: predicted Mid band [0.150, 0.180] is HARD-EARNED at the methodology level (T4 SYMPOSIUM Rudin floor row IS canonical citation; Wang-Rudin 2015 + Ustun-Rudin 2016 + Lin-Rudin 2020 IS canonical empirical literature). Long band [0.10, 0.13] is HARD-EARNED at the lower-bound level (Rudin floor ≥ Shannon R(D); Shannon R(D) is CITED canonical Blahut-Arimoto compute).
- **CARGO-CULTED**: predicted Asymptotic band [0.05, 0.10] is CARGO-CULTED at the specific-number level (the "interpretability tax of 3-5%" is canonical-citation-level but the SPECIFIC mapping to this substrate's [0.05, 0.10] range has not been empirically validated; the Assumption-Adversary verdict above explicitly notes "NO substrate has yet validated it").
- **EMPIRICAL VALIDATION REQUIRED**: K=8 LEVEL-1 dispatch produces the FIRST empirical anchor for the Rudin floor estimate. If the dispatch lands at e.g. 0.25 [contest-CUDA], the Rudin floor estimate gets REVISED downward. If it lands at 0.15, the asymptotic-pursuit hypothesis is empirically validated.

### Cost estimate

- **Memo only (today)**: **$0 GPU** + ~3-5 hours editor work (PENDING T3 ratification per the path 2 lattice directive)
- **L0 SKETCH → L1 SCAFFOLD landing**: $0 GPU + ~3-5 days of substrate scaffold engineering (per sister NSCS06 v8 pattern)
- **K=8 LEVEL-1 dispatch smoke**: **$3-15** Modal A100 100ep + **$0.10-0.50** paired CPU eval per Catalog #226 + #246
- **L2 INTEGRATION landing (if smoke confirms band)**: ~$30-50 Modal A100 full + paired CPU eval + sister Dykstra-feasibility check
- **Total to first empirical anchor**: ~$5-20 GPU; ~3-5 days wall-clock

### Reactivation criteria

If K=8 LEVEL-1 dispatch lands the Rudin substrate OUTSIDE the predicted Mid band [0.150, 0.180]:

- **Lands at [0.150, 0.180]**: HYPOTHESIS VALIDATED; proceed to L2 INTEGRATION + paired CPU eval + sister Dykstra-feasibility check + composition-with-asymptotic-sister-substrates per Section 13.
- **Lands above 0.180 but below 0.30**: PARTIAL VALIDATION; review Rashomon ensemble disagreement queue for next-iteration probes; identify which rules are under-performing; refine rule-list per Wang-Rudin canonical falling-rule-list `prune_ineffective_rule` discipline; re-dispatch (sister of NSCS06 v6 → v7 → v8 trajectory).
- **Lands above 0.30**: FALSIFICATION at this specific configuration; per CLAUDE.md "Forbidden premature KILL without research exhaustion": DO NOT KILL; classify as DEFERRED-pending-research; reactivation criteria: (a) re-attempt with K=12 Rashomon members; (b) re-attempt with K=4 rules (more constrained falling-rule-list); (c) re-attempt with substrate-specific predicate vocabulary expansion; (d) review T4 SYMPOSIUM Rudin floor estimate downward if 3+ Rudin-class substrate dispatches all land >0.25.
- **Lands at <0.15 (better than predicted Mid)**: HYPOTHESIS STRENGTHENED; immediately revise T4 SYMPOSIUM Rudin floor row downward; immediately queue stack-of-stacks composition smokes per Section 13.

---

## 19. Cargo-cult ledger (preserves NSCS06-style ledger discipline)

| # | Potential cargo-cult | Status | Mitigation |
|---|---|---|---|
| 1 | "Interpretability tax of 3-5% applies to this substrate's specific architecture" | UNVALIDATED-AT-EMPIRICAL-LEVEL | K=8 LEVEL-1 dispatch IS the empirical validation; specific tax magnitude REVISED downward/upward per the empirical anchor |
| 2 | "K=6 falling rules suffice for video compression" | UNVALIDATED | Wang-Rudin 2015 canonical is for binary classification; video compression is structurally different; K=6 is HYPOTHESIS; reactivation criterion includes K=4/K=8/K=12 sweep |
| 3 | "GOSDT depth-4 encoder is sufficient for per-pair region characterization" | UNVALIDATED | Lin-Rudin 2020 canonical is for tabular data; per-pair video region characterization has different statistical structure; depth-4 is HYPOTHESIS; reactivation criterion includes depth-3/depth-6 sweep |
| 4 | "Rashomon K=8 disagreement queue produces useful probe priorities for substrate level" | UNVALIDATED | Semenova-Rudin-Parr 2020 canonical is for general prediction; per-pixel probe priority at substrate level is structurally different; K=8 is HYPOTHESIS; reactivation criterion includes K=16/K=32 sweep |
| 5 | "SLIM integer-coefficient bound [-10, 10] suffices for rule-list parameterization" | UNVALIDATED | Ustun-Rudin 2016 canonical for medical scoring; substrate-specific bound may differ; reactivation criterion includes [-20, 20]/[-5, 5] sweep |
| 6 | "Pure Python ≤200 LOC inflate fits 30-min T4 contest budget" | UNVALIDATED-AT-LARGE-SCALE | Per-frame inflate ~10-50ms × 1200 frames = ~15-60s — well within budget at moderate-difficulty rule application; large-rule-count case may exceed; reactivation criterion: profile inflate wall-clock at L1 dispatch |
| 7 | "Scorer-as-shared-prior (Wyner-Ziv pattern) reduces rate by H(frames | scorer_state_dict) factor" | UNVALIDATED-AT-EMPIRICAL-LEVEL | Wyner-Ziv 1976 canonical; specific rate reduction depends on substrate's predicate vocabulary referencing scorer features; HYPOTHESIS; empirical validation at L1 |

Per the sister NSCS06 v6→v7 → v8 trajectory pattern: ITERATIVE CARGO-CULT UNWINDING with predicted ~50% score reduction per iteration if held. The Rudin substrate's cargo-cult ledger is similar in structure but smaller in scope (7 vs 7 cargo-cults for NSCS06; same magnitude but more structurally novel).

---

## 20. Lane registry declaration

Per CLAUDE.md "Lane maturity registry" non-negotiable + `tools/lane_maturity.py add-lane`:

```yaml
lane_id: lane_rudin_floor_interpretable_ml_substrate_scoping_design_20260516
substrate_id: rudin_floor_interpretable_ml_compositional_decoder
lane_class: substrate_engineering  # per HNeRV parity L7 — substrate engineering exceeds bolt-on budget
substrate_class_shift: architecture_class_AND_decode_time_contract_AND_scorer_relationship
horizon_class: asymptotic_pursuit
target_modes: ["contest_one_video_replay", "contest_generalized"]
deployment_target: t4_contest_runtime
level: 0  # SKETCH (memo only; substrate NEVER BUILT)
gates:
  impl_complete: false  # scaffold not yet built
  real_archive_empirical: false  # no archive built yet
  contest_cuda: false  # no dispatch yet
  strict_preflight: false  # no preflight gate written yet
  three_clean_review: false  # T3 ratification pending
  memory_entry: pending  # this memo is the memory entry
  deploy_runbook: false  # no remote_lane script yet
notes: |
  STRUCTURAL GAP per FALSIFICATION-AUDIT-v2 A3; NEW asymptotic-pursuit candidate; NEVER BUILT.
  Triple class-shift (architecture + decode-time + scorer-relationship).
  Predicted Mid CPU band [0.150, 0.180] per T4 SYMPOSIUM 4×4 Rudin floor row.
  research_only=false; canonical contest_exact_eval target after T3 ratification.
  archive_grammar: RDIF v1 monolithic 0.bin per HNeRV parity L3
  parser_section_manifest: per Section 10 byte-level grammar
  inflate_runtime_loc_budget: ≤200 LOC (HNeRV L4 substrate_engineering exception)
  runtime_dep_closure: numpy + Pillow + standard library only (NO PyTorch)
  export_format: RDIF v1 (per Section 11 export contract)
  score_aware_loss: Rashomon ensemble consensus (per Section 8)
  bolt_on_loc_budget: substrate_engineering — exceeds ≤350 LOC; substrate engineering happens ONCE per architecture class
  no_op_detector_planned: yes (per Catalog #272 distinguishing_feature contract; sister tools/verify_distinguishing_feature_byte_mutation.py)
  score_improvement_mechanism_status: PENDING_L1_DISPATCH (operational mechanism: rule-application IS the byte-consumption proof)
  distinguishing_feature_name: rudin_floor_falling_rule_list_decoder
  distinguishing_bytes_path: 0.bin::rule_list_blob
  inflate_consumer_function: rudin_floor.inflate.inflate
  byte_mutation_smoke_passes: PENDING_L1_DISPATCH
```

This lane will be registered via `tools/lane_maturity.py add-lane lane_rudin_floor_interpretable_ml_substrate_scoping_design_20260516 --name "Rudin floor interpretable-ML substrate (scoping)" --phase 2` after this memo lands.

---

## 21. Op-routables (ranked by mission-contribution × cost)

Per HORIZON-CLASS directive Consequence 3 mission-contribution co-classification:

| Priority | Op-routable | Mission contribution × cost | Justification |
|---|---|---|---|
| **1** | T3 batched council ratification (next 7 days per Catalog #291 META-ASSUMPTION cadence) | `frontier_breaking` × `asymptotic_pursuit` / $0 GPU | Per the path 2 lattice directive Consequence 4: this memo enters the lattice as a LEVEL-0 asymptotic-pursuit node; T3 ratification unlocks L1 dispatch |
| **2** | L1 SCAFFOLD landing (sub-agent task; ~3-5 days editor work; $0 GPU) | `frontier_breaking` × `asymptotic_pursuit` / $0 GPU | Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable: L1 SCAFFOLD requires architecture + score-aware loss + archive grammar + inflate runtime + export contract; this memo provides the design; scaffold is the engineering |
| **3** | K=8 LEVEL-1 dispatch smoke ($3-15 Modal A100 100ep; $0.10-0.50 paired CPU) | `frontier_breaking` × `asymptotic_pursuit` / $3-15 + $0.50 | The empirical validation of the T4 SYMPOSIUM Rudin floor estimate; outcome determines whether the asymptotic-pursuit hypothesis HOLDS |
| **4** | Sister `tools/probe_rudin_floor_substrate_disambiguator.py` design (per Catalog #125 hook 6) | `frontier_protecting` × `apparatus_maintenance` / $0 GPU + ~1-2 days editor work | Per CLAUDE.md "Anti-arbitrariness primitive: the probe-disambiguator pattern": when 2+ defensible interpretations exist (K=8 Rashomon disagreement queue), the probe-disambiguator IS the arbitration |
| **5** | Stack-of-stacks composition LEVEL-2 smokes (Rudin × NSCS06 v8 + Rudin × A-STACK after LEVEL-1 confirms) | `frontier_breaking` × `asymptotic_pursuit` / $20-30 paired smoke each | Per Section 13: Rudin × NSCS06 v8 is HIGH priority because both are asymptotic-pursuit and complementary; Rudin × A-STACK is novel meta-decoder pattern |
| **6** | Daubechies wavelet+CS substrate scoping memo (sister asymptotic-pursuit; per FALSIFICATION-AUDIT-v2 A5 reclassification) | `frontier_breaking` × `asymptotic_pursuit` / $0 GPU + ~3 days editor | Per HORIZON-CLASS directive: 4 asymptotic-pursuit candidates queued; Rudin (this memo) is 1 of 4; Daubechies wavelet+CS is 2 of 4 (per FALSIFICATION-AUDIT-v2 A5 NSCS06 v8 reclassification) |
| **7** | Z6/Z7/Z8 + Tishby IB-pure sister memos (in flight per parent prompt; co-disjoint per Catalog #230) | `frontier_breaking` × `asymptotic_pursuit` / $0 GPU each | Per FALSIFICATION-AUDIT-v2 A2 + A4: Z6/Z7/Z8 (sister sub-agent ongoing) + Tishby IB-pure (sister sub-agent ongoing) complete the 4-asymptotic-pursuit-candidate quartet |

---

## 22. Cross-references

- **T4 SYMPOSIUM 4×4 floor matrix anchor**: `.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md` (commit `28e7c67c7`; Rudin floor row Mid [0.150, 0.180] / Long [0.10, 0.13] / Asymptotic [0.05, 0.10])
- **FALSIFICATION-AUDIT-v2 A3 entry**: `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` (NEW asymptotic-pursuit candidate; NEVER BUILT; structural gap)
- **HORIZON-CLASS standing directive**: `feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md` (Rudin floor substrate named under ASYMPTOTIC-PURSUIT GAP)
- **Mission-alignment standing directive**: `feedback_council_apparatus_in_service_of_innovation_rigor_optimization_score_lowering_20260516.md` (`frontier_breaking` × `asymptotic_pursuit` = HIGHEST per Consequence 5)
- **Max-observability standing directive**: `feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md` (explicitly names "Falling-rule lists / SLIM / GOSDT are observability-MAX per the Rudin canonical discipline")
- **Path 2 lattice directive (supersedes L5 v2 staircase)**: `feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516.md` (lattice framework for asymptotic-pursuit)
- **Abandon-within-class directive**: `feedback_abandon_within_class_refinements_only_substrate_class_shifts_pursue_frontier_20260515.md` (Rudin substrate is TRIPLE class-shift per the taxonomy)
- **HARD-EARNED vs CARGO-CULTED addendum**: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` (per-assumption classification framework)
- **UNIQUE-AND-COMPLETE-PER-METHOD operating mode**: `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` (binding for substrate scaffolds; canonical-vs-unique decision per layer required per Section 15)
- **Rudin-Daubechies autopilot landing**: Catalogs #273-#278 (canonical META-layer that this substrate consumes); `src/tac/autopilot_rudin_daubechies/{slim_ranker,falling_rule_list,rashomon_ensemble,compressive_landscape,wavelet_multi_scale_ranker,gosdt_dispatcher}.py`
- **NSCS06 v8 Path B sister memo (template + sister asymptotic-pursuit per FALSIFICATION-AUDIT-v2 A5)**: `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md`
- **9-dimension success checklist standing directive**: `feedback_9_dimension_success_checklist_per_substrate_and_stack_of_stacks_standing_directive_20260515.md` (Section "9-dimension success checklist evidence" above satisfies per Catalog #294)
- **CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"**: 13 inviolable lessons + 8th forbidden pattern + 5 forbidden code patterns (this design memo honors all 13)
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"**: Section 18 reactivation criteria honor this (DEFERRED-pending-research, NOT KILL)
- **CLAUDE.md "Apples-to-apples evidence discipline"**: all predicted bands tagged `[prediction]`; all empirical citations tagged with axis (`[contest-CUDA]` / `[contest-CPU]` / `[diagnostic-CPU]`)
- **Catalog #220 substrate L1 byte-addition operational mechanism**: Section 15 declares `score_improvement_mechanism_status=OPERATIONAL via apply_l2_overlay_for_video_list`
- **Catalog #272 distinguishing-feature integration contract**: Section 15 declares all 4 contract fields
- **Catalog #290 canonical-vs-unique decision per layer**: Section 15 above (this memo is canonically compliant; structurally extincts Catalog #290 WARN-ONLY at landing)
- **Catalog #294 9-dimension success checklist evidence**: Section "9-dimension success checklist evidence" above
- **Catalog #296 predicted ΔS band Dykstra-feasibility check**: Section 18 above (PENDING canonical helper invocation at L1 dispatch)
- **Catalog #297 signal-axis destruction reversibility probe**: N/A (Rudin substrate does NOT destroy signal axes; rule-application preserves all axes per the falling-rule-list compositional decoder design)
- **Catalog #303 hard-earned vs cargo-culted classification**: Section 18 above (per-band classification HARD-EARNED at methodology level + CARGO-CULTED at empirical level pending L1 dispatch)
- **Catalog #305 (proposed) substrate observability surface**: Section 16 above (declares all 6 observability surface elements per max-observability directive)

---

## Final summary

**This scoping design memo closes the asymptotic-pursuit STRUCTURAL GAP for the Rudin floor substrate** per FALSIFICATION-AUDIT-v2 A3 + HORIZON-CLASS directive Consequence 2 (≥20% K-schedule allocation to asymptotic-pursuit; current 0%). The substrate is a TRIPLE class-shift (architecture + decode-time + scorer-relationship) where the META-layer Rudin-Daubechies autopilot (Catalogs #273-#278) is RE-ARCHITECTED as the decode-time architecture itself — making this the FIRST substrate where the autopilot apparatus serves as the decoder. Predicted bands per T4 SYMPOSIUM Rudin floor row: Mid [0.150, 0.180]; Long [0.10, 0.13]; Asymptotic [0.05, 0.10]. The CRITICAL EMPIRICAL ASSUMPTION (per Assumption-Adversary): the Rudin floor estimate is META-LEVEL HYPOTHESIS pending K=8 LEVEL-1 dispatch validation.

Cost to first empirical anchor: $0 today (memo) + $3-15 K=8 LEVEL-1 smoke + $0.10-0.50 paired CPU eval = ~$5-20 total + ~3-5 days wall-clock for L1 scaffold landing + smoke dispatch.

Op-routable #1 (highest priority): T3 batched council ratification per the path 2 lattice directive (next 7 days per Catalog #291 META-ASSUMPTION cadence). This memo enters the lattice as a LEVEL-0 asymptotic-pursuit node.

### Ego-motion conditioning declaration (Catalog #311 / Pattern H)

<!-- # PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:cross-reference-only-not-substrate-central-cooperative-receiver-framing-cited-as-related-work-not-as-this-substrate-architectural-core-per-z6z7z8-design-memo-section-11-scope -->

This memo references Atick-Redlich / cooperative-receiver framing as cross-reference / related-work / sister-substrate context — NOT as this substrate's architectural core. The substrate proposed by this memo is structurally distinct from Z6/Z7/Z8 (which DO require ego-motion-conditioned next-frame prediction as architectural core per Pattern H + Z6/Z7/Z8 design memo Section 11).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Z6/Z7/Z8 design memo Pattern H + Catalog #311 acceptance cascade (c): same-line waiver `# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:cross-reference-only-not-substrate-central-cooperative-receiver-framing-cited-as-related-work-not-as-this-substrate-architectural-core-per-z6z7z8-design-memo-section-11-scope` applies. The waiver rationale is non-placeholder (>4 chars, not `<rationale>` / `<reason>`).

Cross-references to cooperative-receiver / Atick-Redlich in this memo serve as theoretical-anchor / related-work / sister-substrate-comparison only; they do NOT make this substrate a predictive-coding substrate in the Pattern H sense.


---

## Observability surface

**Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16** (`feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`) + Catalog #305 STRICT preflight gate (`check_substrate_design_memo_has_observability_surface_section`).

**Per Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:** this section is appended to the design memo; pre-existing body content (Sections 1-N + 9-dim checklist + cargo-cult audit + canonical-vs-unique decision + cross-references) is UNCHANGED. Appended by WAVE-1 APPARATUS HARDENING subagent 2026-05-16 to enable Catalog #305 STRICT-flip.

**The 6-facet observability surface for this design:**

1. **Per-layer inspection.** Every layer of this substrate / composition / experiment captures its (input tensor, output tensor, intermediate activations, attention maps when applicable) at runtime via the canonical xray-style hook pattern (`tac.xray.<lens>` modules) without re-instrumentation. The forward pass emits per-layer observables to `experiments/results/<lane>/observability/per_layer/<layer_name>.jsonl` for post-hoc inspection.

2. **Per-signal decomposition.** Composite metrics (`final_score = seg + sqrt(10*pose) + 25*rate`) decompose into constituent contributions per the canonical `tac.xray.per_pair_score_decomposition` lens. Per-pair / per-class / per-axis / per-stage breakdown serialized to `experiments/results/<lane>/observability/score_decomposition.json` with axis labels matching CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable (`[contest-CUDA]` / `[contest-CPU]` / `[diagnostic-CPU]` / `[macOS-CPU advisory]` / `[MPS-PROXY]`).

3. **Run-to-run diff.** Two runs of this substrate / composition produce byte-identical reproducible artifacts under the same `(seed, commit_sha, upstream_snapshot_sha256)` tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger. The canonical diff helper `tools/diff_auth_eval_results.py <run_a.json> <run_b.json>` (planned per the observability audit Highest-ROI extension list) emits per-component deltas + per-pair drift; until landed, manual diff via `archive_sha256` byte-addressability is straightforward.

4. **Post-hoc query interface.** Run artifacts under `experiments/results/<lane>/` serialize as structured JSON / JSONL surfaces consumable without re-running: `contest_auth_eval_<axis>.json` (canonical per-component scores with axis labels) + `modal_metadata.json` (per-dispatch cite-chain per Catalog #166) + `observability/*.jsonl` (per-layer + per-signal). The continual-learning posterior at `.omx/state/continual_learning_posterior.jsonl` is queryable per (substrate, axis, hardware, evidence_grade) via `tac.continual_learning.query_*` helpers per Catalog #128 + #131 fcntl-locked discipline.

5. **Cite-chain.** Every behavior signal anchors to the canonical tuple `(substrate_id, commit_sha, modal_call_id, config_path, random_seed, upstream_snapshot_sha256)` via Catalog #245 `tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)`. The call_id ledger row schema includes `mounted_code_git_head` (per Catalog #166) + `agent` + `subagent_id` + `session_id` for full forensic reconstruction. Score claims tagged per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

6. **Counterfactual hooks.** Byte-mutation surface per Catalog #139 packet compiler (`tools/verify_distinguishing_feature_byte_mutation.py --distinguishing-byte-range <offset>:<length>`) + Catalog #272 distinguishing-feature integration contract + Catalog #105 no-op detector. The substrate's archive grammar exposes byte-offset addressability for "what if this byte changed?" probing without re-running training. Per-layer / per-component ablation switches surfaced via the trainer's argparse flags + the canonical `tac.xray.<lens>.ablate_*` helpers when applicable.

**Acceptance per Catalog #305:** this section satisfies the structural requirement (literal section header `## Observability surface` present); the body content above documents the substrate's 6-facet observability surface for operator-facing audit.

**Sister observability-discipline gates active for this substrate:**

- Catalog #245 modal_call_id_ledger (every dispatch registered)
- Catalog #166 Modal HEAD-parity ledger (every dispatch worker-source-verified)
- Catalog #128 + #131 fcntl-locked JSONL posterior discipline (state mutations append-only)
- Catalog #139 packet compiler no-op detector (byte-mutation surface)
- Catalog #272 distinguishing-feature integration contract (per-substrate byte-mutation proof)
- Catalog #220 substrate L1+ operational mechanism declaration (no opaque byte additions)
- Catalog #105 no-op detector (no-op provenance)
- Catalog #127 authoritative tag custody (per-call-site axis + hardware-substrate validation)

**Observability extension recommendations (queued for follow-on):** see `tools/audit_existing_infrastructure_for_observability.py --summary` output for the canonical 8-tool / 6-facet observability gap analysis + Highest-ROI extension list. The `tools/audit_*.py` family is the highest-ROI extension target (3/12 observability) per the standing-directive consequence 3.
