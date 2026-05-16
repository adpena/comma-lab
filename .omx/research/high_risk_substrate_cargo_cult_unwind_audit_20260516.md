# HIGH-RISK substrate cargo-cult-unwind DESIGN audit — 2026-05-16

**Author:** HIGH-RISK-SUBSTRATE-CARGO-CULT-UNWIND-AUDIT subagent (CRASH-RESUME respawn `high_risk_cargo_cult_unwind_resp_20260516`)
**Lane:** `lane_high_risk_substrate_cargo_cult_unwind_audit_20260516`
**Foundation:** `.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md` (Top-5 HIGH-RISK list)
**Template:** `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md`
**Cargo-cult framework:** `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`

## Operating-within assumption-statement (Catalog #292)

The assumption I am operating within for this audit: *"Cargo-cult-unwinding via DESIGN-only memos (no trainer mutation) before next paid dispatch is the lowest-cost intervention with the highest NSCS06-class-falsification-prevention EV."* HARD-EARNED basis: Catalog #229 premise-verification protocol + the NSCS06 553× falsification empirical proof that DESIGN-TIME static review catches cargo-cults that runtime smoke does not (e.g. `Y=R=G=B` is a 2-line catch; `np.roll` with 2-of-6 pose dims is a 4-line catch). The Assumption-Adversary seat would challenge: "Why DESIGN-only rather than implementation? — Because implementation is sister-subagent owned per Catalog #230; this audit's scope is DESIGN."

---

## Section 1 — Per-substrate cargo-cult-unwind DESIGN

For each of the 5 HIGH-RISK substrates, the design follows the canonical template: (a) identity → (b) cargo-cult enumeration → (c) UNWIND DESIGN → (d) PROBE-DISAMBIGUATOR spec → (e) post-unwind DISPATCH-READINESS verdict → (f) RECOMMENDED ACTION SEQUENCE.

---

### 1.1 NSCS02 — Downsampled Renderer Inflate Upsample (PRIORITY 1)

**(a) Identity**
- `lane_id`: `lane_nscs02_downsampled_renderer_inflate_upsample_20260515` (L1, research_only)
- design memo: `.omx/research/nscs02_*_design_*` (per audit foundation §2.6)
- trainer: `experiments/train_substrate_nscs02_downsampled_renderer.py` (39.6KB)
- predicted ΔS band: NOT EXPLICITLY DECLARED in design memo (cargo-cult itself; META-CC-1 candidate)

**(b) HIGH-RISK cargo-cult enumeration (4 items; CC-1 + CC-4 HIGH-RISK; CC-2 + CC-3 MEDIUM)**

- **CC-1** "Renderer at downsampled resolution (192×128 / 96×64) + bicubic upsample at inflate matches full-res renderer at lower bytes"
  - Source: `experiments/train_substrate_nscs02_downsampled_renderer.py` `synthesize_frame_*` body
  - HARD-EARNED-vs-CARGO-CULTED: **CARGO-CULTED** — directly analogous to NSCS06's `Y=R=G=B`. Bicubic upsample destroys the high-frequency texture that SegNet's RGB-distinguishing class cues depend on (the same class cues the NSCS06 falsification proved matter)
  - Predicted failure mode: SegNet seg_avg surge (NSCS06 hit seg=64.59 from chroma loss; NSCS02 bicubic upsample is the spatial-frequency analog → predict seg_avg surge to 20-60 range; total score >> 1.0)
  - Severity: HIGH

- **CC-2** "Bicubic interpolation preserves SegNet argmax accuracy at <2× downsample ratio"
  - Source: `experiments/train_substrate_nscs02_downsampled_renderer.py` archive-builder upsample call
  - Classification: **CARGO-CULTED** — SegNet stride-2 stem has 256-pixel effective receptive field; 2× downsample MAY be tolerable but >2× definitely degrades
  - Severity: MEDIUM (boundary-of-tolerance assumption)

- **CC-3** "PoseNet's 12-channel YUV6 input is tolerant to bilinear upsample at (512, 384) preprocess"
  - Source: scorer-preprocess pipeline downstream of the upsample
  - Classification: **CARGO-CULTED** — chroma subsampling means upsampling chroma has 2× tolerance; luma is the dominant pose signal and is more sensitive
  - Severity: MEDIUM

- **CC-4** "Symmetric (compress + inflate) bicubic preserves eval_roundtrip simulation"
  - Source: training loop's pyav decode → renderer → eval_roundtrip → loss chain
  - Classification: **CARGO-CULTED** — eval_roundtrip simulates 384→874→uint8→384, but the trainer's "downsampled renderer" effectively becomes 192→874→uint8→384 if compress applies bicubic too late. Exact cadence match to NSCS06 v6 PV-5 ("compress saw 'what the renderer will produce' but the renderer is structurally incapable")
  - Severity: HIGH (gradient-fidelity bug; trainer optimizes against a fictional eval roundtrip)

**(c) UNWIND DESIGN per cargo-cult**

- **CC-1 unwind**: ADD predicted-band declaration to design memo with EXPLICIT Dykstra-feasibility intersection: "the achievable region for downsampled renderer + bicubic upsample at downsample ratio R intersected with the SegNet argmax-stability polytope at chroma+luma reconstruction radius B(R) is empty for R > 2; predicted ΔS band: NULL pending probe." File: `.omx/research/nscs02_downsampled_renderer_inflate_upsample_design_20260515.md` — INSERT new "## Predicted band + Dykstra-feasibility" section per proposed Catalog #296.

- **CC-2 unwind**: ADD recipe-level `min_downsample_ratio: 2` field (Tier 2 hardware-correctness contract; sister of Catalog #170 min_vram_gb). Refuse archive build at R > 2 unless `# DOWNSAMPLE_RATIO_OVERRIDE_OK:<rationale>` waiver.

- **CC-3 unwind**: ADD scorer-preprocess gradient-reachability annotation in trainer's `_full_main` near the PoseNet forward (similar to Catalog #187 HNeRV parity guard's `patch_upstream_yuv6_globally` requirement). Document in design memo §canonical-vs-unique decision per layer (Catalog #290) that PoseNet's luma-dominant signal is the asymmetric risk axis.

- **CC-4 unwind (HIGHEST PRIORITY)**: REWRITE design memo §training-loop to make the compress-time downsample explicit BEFORE eval_roundtrip simulation. The eval_roundtrip MUST simulate the FULL chain: 384(GT)→192(downsample)→874(scorer)→uint8→384(scorer-resize). Add NEW probe-disambiguator `tools/probe_nscs02_eval_roundtrip_chain_disambiguator.py` that compares the trainer's effective eval_roundtrip output against the actual inflate-time chain on 10 GT pairs; refuses dispatch on >1e-3 RMS divergence.

**(d) PROBE-DISAMBIGUATOR spec (cheapest empirical)**

- Name: `tools/probe_nscs02_paired_downsample_ratio_smoke.py`
- Cost: $5 paired CPU smoke (Linux x86_64 hermetic; macOS-CPU advisory acceptable for first-pass per Catalog #192)
- Method: train NSCS02 trainer at 3 downsample ratios (384×512 = 1×, 192×256 = 2×, 96×128 = 4×) for 25 epochs each on `upstream/videos/0.mkv`; emit `contest_auth_eval` per ratio; compare against PR101 baseline at 0.193 to compute Δ-vs-R knee
- Disambiguates: CC-1 (signal-axis-destruction risk) + CC-2 (SegNet stride-2 tolerance boundary) simultaneously
- Output: `experiments/results/probe_nscs02_downsample_ratio_*/smoke_result.json` with per-ratio score band; updates posterior per Catalog #128

**(e) DISPATCH-READINESS POST-UNWIND**: Would CLEAR Catalog #270 (Tier 1+2+3 protocol) + Catalog #272 (distinguishing feature contract; needs `distinguishing_feature_name=spatial_downsample_ratio_R` + `byte_mutation_smoke_passes`) + NEW Catalog #297 (signal-axis-destruction reversibility probe; the new tools/probe_nscs02_* file satisfies the reversibility-probe acceptance token). Catalog #292 satisfied via design-memo per-section assumption-statements. Catalog #296 satisfied via the §Predicted-band-Dykstra section.

**(f) ACTION SEQUENCE**
- IMMEDIATE ($0): land design-memo §Predicted-band + §Canonical-vs-unique per Catalog #290 + §Probe-disambiguator-required + `min_downsample_ratio: 2` recipe field + EXPLICIT `# DOWNSAMPLE_RATIO_OVERRIDE_OK` waiver semantics. Mark lane `research_only=true; reactivation_criteria=[probe_nscs02_paired_downsample_ratio_smoke_run, dykstra_feasibility_section_landed, eval_roundtrip_chain_disambiguator_passes]`.
- NEAR-TERM ($5): run paired downsample-ratio smoke (above probe spec); produce score-vs-R knee
- DISPATCH-READY ($5-15): once knee found AND empirical band falls within Dykstra-feasibility, paired smoke on Modal A100 100ep for the SELECTED ratio; auth-eval contest-CUDA + contest-CPU on the resulting archive

---

### 1.2 C6 MDL-IBPS (e4 variant) (PRIORITY 2)

**(a) Identity**
- `lane_id`: `lane_c6_e4_mdl_ibps_substrate_campaign_20260514` (L1, dispatch-enabled)
- design memo: `.omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md` + `mdl_ibps_substrate_council_design_round_1_20260513.md`
- trainer: `experiments/train_substrate_c6_e4_mdl_ibps.py` (36.9KB)
- predicted ΔS band: **[0.11, 0.16]** (HIGH-RISK; factor of ~1.45 wide; first-principles MDL+IB lower bound)

**(b) Cargo-cult enumeration (3 items)**

- **CC-1** "MDL × IB × Procedural-Synthesis substrate yields ΔS -0.030 to -0.080 vs A1"
  - Source: design memo §2 (campaign predicted band)
  - Classification: **CARGO-CULTED** — wide band (factor 2.7); not validated empirically beyond 5ep smoke
  - Severity: MEDIUM

- **CC-2** "Procedural decoder + per-pair patches is more bit-efficient than monolithic decoder + latents"
  - Source: trainer architecture choice (Selfridge demon hierarchy + MoE assumption)
  - Classification: **CARGO-CULTED** — HNeRV (content-adaptive embeddings) is the EMPIRICAL counter-example proving content-adaptive >> procedural at the PR101/PR103 frontier
  - Severity: MEDIUM (orthogonal-architecture risk)

- **CC-3** "Predicted post-campaign score band [0.11, 0.16]"
  - Source: design memo §predicted-band
  - Classification: **CARGO-CULTED** — first-principles MDL+IB lower bound but skips Pareto-frontier-consistency check (Dykstra-feasibility); analogous to NSCS06's symposium-#4 prediction failure mode
  - Severity: **HIGH** (band-width-too-narrow risk: if true achievable region is [0.18, 0.22], a paid dispatch could waste $10-30)

**(b-Z1-revision)** Per Catalog #219 Z1 ablation 2026-05-14 + Tier C empirical anchor per Catalog #227: C6 5ep IBPS1 anchor classified `WITHIN-CLASS` per Tier C density posterior. The predicted band [0.11, 0.16] is structurally inconsistent with Z1's within-class plateau finding (the plateau IS the cumulative cost of within-class refinement). The cargo-cult is the band being CARRIED FORWARD POST-Z1.

**(c) UNWIND DESIGN per cargo-cult**

- **CC-1 unwind**: REWRITE design memo §2 predicted band as `[NULL pending Dykstra-feasibility check + Tier C empirical anchor reconciliation]`. Cite Z1 ablation (Catalog #219) as the empirical revision anchor.

- **CC-2 unwind**: ADD §canonical-vs-unique decision per layer (Catalog #290) section to design memo explicitly contrasting procedural-decoder choice against HNeRV's content-adaptive empirical baseline. Document the FORK rationale or pivot to a HYBRID architecture (procedural-decoder + per-pair content-adaptive embeddings).

- **CC-3 unwind (HIGHEST PRIORITY)**: COMPUTE Dykstra-feasibility intersection check ANALYTICALLY (no GPU needed) using `tools/check_substrate_dykstra_feasibility.py` (NEW; proposed Catalog #296 sister tool). Input: MDL constraint `R_MDL >= L(θ) + log|θ|/2` + IB constraint `I(X;T) - β·I(T;Y) ≤ K_IB` + contest rate budget `25*B/N` + Z1 within-class density posterior. Output: achievable Pareto frontier polytope. Revise predicted band to the polytope's distortion-axis intersection.

**(d) PROBE-DISAMBIGUATOR spec**

- Name: `tools/check_substrate_dykstra_feasibility.py` (NEW; analytical; $0 cost; reusable across C6 + ATW + Z4 + TT-L5 + NSCS02)
- Method: alternating projections (Dykstra 1983) onto convex constraint sets; emit feasible polytope vertices + Pareto-frontier-on-distortion-axis
- Disambiguates: CC-3 (predicted band validity) + META-CC-1 (cross-substrate)
- Output: `.omx/state/dykstra_feasibility_<substrate>.json` per substrate; updates cathedral autopilot rate-prediction posterior per Catalog #128

**(e) DISPATCH-READINESS POST-UNWIND**: Would CLEAR Catalog #270 (already passes for C6 per protocol's empirical anchor list) + Catalog #292 (per-section assumption-statements) + NEW Catalog #296 (Dykstra check section in design memo). Catalog #227 Tier C requirement satisfied via existing 5ep anchor.

**(f) ACTION SEQUENCE**
- IMMEDIATE ($0): land Dykstra-feasibility analytical check + design memo §predicted-band revision (cite Z1 ablation Catalog #219); mark `research_only=true; reactivation_criteria=[dykstra_feasibility_polytope_emitted, predicted_band_revised, tier_c_anchor_reconciliation_landed]` UNTIL revision done
- NEAR-TERM ($0): run sister analytical Dykstra check; revise band
- DISPATCH-READY ($5-15): smoke-before-full per Catalog #167 at the REVISED band only

---

### 1.3 ATW Codec V1 (Atick-Tishby-Wyner) (PRIORITY 3)

**(a) Identity**
- `lane_id`: `lane_atw_codec_design_v1_20260515` (L1, research_only; `_full_main` raises NotImplementedError)
- design memo: `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md` (27.4KB)
- trainer: `experiments/train_substrate_atw_codec_v1.py` (14.0KB scaffold)
- predicted ΔS band: **[0.18, 0.21]** (HIGH-RISK; first-principles only)

**(b) Cargo-cult enumeration (3 items)**

- **CC-1** "Three-paper composition into ONE Lagrangian is novel and tractable"
  - Source: design memo §1 (novelty claim)
  - Classification: **HARD-EARNED structurally** (math IS tractable; the three Lagrangian terms compose by convex addition); CARGO-CULTED at the empirical-equivalence axis
  - Severity: LOW

- **CC-2** "Predicted [0.18, 0.21] frontier displacement"
  - Source: design memo §predicted-band (cited from grand reunion symposium Composite #1)
  - Classification: **CARGO-CULTED** — same hand-waved-prediction-band class as NSCS06 symposium-#4
  - Severity: **HIGH**

- **CC-3** "Wyner-Ziv gain estimate for dashcam + scorer is 30-50% conditional entropy reduction"
  - Source: design memo §1 explicitly ("currently a hypothesis, not a measured artifact")
  - Classification: **CARGO-CULTED + HONESTLY ACKNOWLEDGED** — operator-facing disclaimer present
  - Severity: **HIGH** but pre-mitigated by honest disclosure

**(c) UNWIND DESIGN per cargo-cult**

- **CC-1 unwind**: ADD design memo §canonical-vs-unique decision per layer (Catalog #290) section formalizing the 3-paper Lagrangian as the UNIQUE substrate-engineering layer (the FORK from canonical single-paper substrates). Document the math composition (`L_ATW = α·L_AR + β·L_T + γ·L_WZ`) with per-term gradient-fidelity check.

- **CC-2 unwind**: REWRITE design memo §predicted-band as `[NULL pending H(latent|scorer_class) probe + Dykstra-feasibility check]`. Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #287: every numeric prediction MUST carry `[prediction]` axis tag and link to the empirical probe path.

- **CC-3 unwind**: This is ALREADY HONESTLY ACKNOWLEDGED in the design memo. The unwind is just the probe execution (below).

**(d) PROBE-DISAMBIGUATOR spec**

- Name: `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (NEW; SHARED probe across D4 + ATW + Z4 per audit foundation §5 op-routable #5)
- Cost: $3-5 CPU smoke (analytical or one-shot CPU inference)
- Method: compute H(latent | scorer_class) for A1 latent blob conditioned on SegNet argmax classes; estimate Wyner-Ziv gain as max(0, H(latent) - H(latent|class))
- Disambiguates: CC-3 (Wyner-Ziv 30-50% gain hypothesis) directly + CC-2 (predicted band sanity check) by computing the empirical conditional-entropy ceiling
- Output: `.omx/state/h_latent_given_scorer_class_<substrate>.json`; updates ATW + D4 + Z4 simultaneously

**(e) DISPATCH-READINESS POST-UNWIND**: ATW currently CAN'T dispatch (`_full_main raises NotImplementedError`; Catalog #240 recipe-vs-trainer consistency enforces research_only=true). Post-unwind: would CLEAR Catalog #292 + #296 + #287 once probe runs. Implementation (`_full_main` body) is OUT-OF-SCOPE for this audit (sister-subagent territory per Catalog #230).

**(f) ACTION SEQUENCE**
- IMMEDIATE ($0): land §canonical-vs-unique section + §predicted-band rewrite (NULL pending probe) + `[prediction]` axis tags on every numeric claim in design memo
- NEAR-TERM ($3-5): run shared H(latent|scorer_class) probe; emit posterior; update ATW design memo with empirical Wyner-Ziv gain ceiling
- DISPATCH-READY: needs Phase 2 council approval to lift `_full_main NotImplementedError` (independent of cargo-cult-unwind)

---

### 1.4 Time-Traveler L5 (Z6/Z7/Z8 predictive coding world models) (PRIORITY 4)

**(a) Identity**
- `lane_id`: `lane_time_traveler_l5_macos_cpu_smoke_execution_20260513` (L2, dispatch-active macOS CPU advisory) + sister L1 design lanes
- design memo: `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md` + `grand_council_maximize_value_with_time_traveler_seat_20260514.md` (57.6KB)
- trainer: `experiments/train_substrate_time_traveler_l5_autonomy.py` (63.0KB — full PAIR T directive)
- predicted ΔS band: implied TT5L target = 95-110 KB packet; no explicit score band declared (CARGO-CULT itself; META-CC-1)

**(b) Cargo-cult enumeration (3 items)**

- **CC-1** "Single-archive TT5L packet at 95-110 KB target size is achievable"
  - Source: design memo target (composition arithmetic)
  - Classification: **CARGO-CULTED** — target derived from 5-design-move byte-budget composition; not empirically anchored
  - Severity: MEDIUM

- **CC-2** "Five first-principles design moves compose additively for ΔS"
  - Source: design memo §composition arithmetic (Rao-Ballard + Atick-Redlich + foveation + world-model + Tikhonov)
  - Classification: **CARGO-CULTED** — Dykstra-feasibility says composition is at best SUBADDITIVE in convex intersection regime. Z1 ablation 2026-05-14 EMPIRICALLY confirmed within-class saturation. The 5-move composition is mathematically convex-intersection (NOT additive) — the ΔS predicted by summing per-move improvements is structurally an OVER-ESTIMATE
  - Severity: **HIGH**

- **CC-3** "macOS CPU advisory + Linux x86_64 GHA paired is sufficient compute envelope"
  - Classification: **HARD-EARNED** — Catalog #192 + #197 enforce the advisory-vs-promotable contract correctly
  - Severity: LOW

**(c) UNWIND DESIGN per cargo-cult**

- **CC-1 unwind**: ADD design memo §predicted-byte-budget revision: each of the 5 design moves contributes an INDEPENDENT BUDGET ceiling (not floor). The composition byte budget is `max(per_move_budget)` (worst-case dominates) NOT `sum(per_move_savings)` (best-case additive). Revise target from 95-110 KB to `[95 KB ceiling pending paired Dykstra-feasibility]`.

- **CC-2 unwind (HIGHEST PRIORITY)**: REWRITE design memo §composition-arithmetic as Dykstra-feasibility intersection: `achievable_polytope = projection(rate_budget) ∩ projection(distortion_budget) ∩ ⋂_i projection(per_move_constraint_i)`. The 5-move composition is the INTERSECTION not the SUM. ΔS prediction must be the projection of the polytope's centroid onto the distortion axis (or its lower bound). The SUBADDITIVE penalty per move is a 5-move geometric factor; document this explicitly.

- **CC-3 unwind**: No change; existing Catalog #192/#197 contracts hold.

**(d) PROBE-DISAMBIGUATOR spec**

- Name: `tools/check_substrate_dykstra_feasibility.py --substrate time_traveler_l5_5move` (SHARED with C6 + ATW + NSCS02)
- Cost: $0 analytical
- Method: alternating projections onto 5 convex constraint sets (predictive-coding rate ≤ R_pc / Atick-Redlich rate ≤ R_AR / foveation rate ≤ R_fov / world-model rate ≤ R_wm / Tikhonov rate ≤ R_tik) intersected with contest rate budget; emit intersection polytope
- Disambiguates: CC-2 (composition additivity vs subadditivity) + CC-1 (byte budget ceiling) simultaneously
- Output: `.omx/state/dykstra_feasibility_time_traveler_l5.json`

**(e) DISPATCH-READINESS POST-UNWIND**: Would CLEAR Catalog #270 + #292 + #296 + Z1-revision (Catalog #219 within-class density penalty applies; predicted-ΔS gets the within-class haircut). The macOS-CPU advisory anchor REMAINS the only legitimate empirical evidence pending Linux x86_64 paired.

**(f) ACTION SEQUENCE**
- IMMEDIATE ($0): land §Dykstra-feasibility rewrite of composition arithmetic + §predicted-band revision (target = polytope projection; not move-sum); mark sister L1 design lanes `research_only=true; reactivation_criteria=[dykstra_feasibility_5move_polytope_emitted, predicted_target_revised_to_intersection_lower_bound, z1_within_class_haircut_applied]`
- NEAR-TERM ($0): run shared Dykstra check (sister to C6 + ATW + NSCS02)
- DISPATCH-READY ($5-15): once revised band falls within Z1 + Tier-C-class-shift acceptance band, paired Linux x86_64 GHA + macOS-CPU re-eval per Catalog #192

---

### 1.5 sane_hnerv (just-landed canary) (PRIORITY 5)

**(a) Identity**
- `lane_id`: `lane_substrate_sane_hnerv_20260512` (L1) + `lane_sane_hnerv_archive_fix_catalog_161_20260513` (L2 substrate_engineering)
- design memo: implied via HNeRV parity discipline (Catalog #187) + per-substrate trainer comments
- trainer: `experiments/train_substrate_sane_hnerv.py` (51.8KB)
- predicted ΔS band: 0.193-0.195 [contest-CPU] (reproducing PR95 baseline)

**(b) Cargo-cult enumeration (4 items; CC-1 + CC-3 MEDIUM; CC-2 + CC-4 LOW)**

- **CC-1** "Sane HNeRV (cleanup pass on original HNeRV) will reproduce 0.193-0.195 [contest-CPU]"
  - Source: lane registry notes + trainer header
  - Classification: **CARGO-CULTED** — 5 failed first-anchor attempts on 2026-05-12 are empirical fragility signal
  - Severity: **MEDIUM** (empirical-fragility class)

- **CC-2** "Single-canary-first ordering is the right race-mode dispatch pattern"
  - Classification: **HARD-EARNED** — Catalog #173 (canary-first ordering) makes this explicit
  - Severity: LOW

- **CC-3** "HNeRV's `content-adaptive-embedding` is necessary AND sufficient for the score floor"
  - Source: HNeRV parity discipline L1 + sane_hnerv architecture
  - Classification: **CARGO-CULTED on "sufficient"** — necessity is empirically true; sufficiency is contradicted by PR101/PR103/PR102 all adding bolt-ons
  - Severity: MEDIUM

- **CC-4** "Standard `score_pair_components` canonical helper is the optimal scorer-loss routing for HNeRV"
  - Source: trainer's scorer-loss invocation
  - Classification: **CARGO-CULTED** — Catalog #164 hard-earns differentiability invariant; canonical helper passes invariant but may not be HNeRV-OPTIMAL specifically
  - Severity: LOW

**(c) UNWIND DESIGN per cargo-cult**

- **CC-1 unwind**: AUDIT 5-failed-attempt forensics. Pull `.omx/state/modal_call_id_ledger.jsonl` rows for `lane_substrate_sane_hnerv_20260512`; classify each failure (NVML 999 / OOM / archive grammar / scorer-loss routing / harness timeout / etc.). The 5 failures are EMPIRICAL data; the cargo-cult is treating them as integration bugs rather than substrate-design fragility signal. Land design memo §Failure-mode-classification with per-attempt root cause.

- **CC-2 unwind**: No change; Catalog #173 holds.

- **CC-3 unwind**: ADD design memo §necessary-vs-sufficient decomposition. Document that PR101/PR103/PR102 winners all stacked bolt-ons (entropy / sidecar / pose-augmentation) on top of HNeRV-core — therefore the SCORE FLOOR target of 0.193 is the BOLT-ON-MINUS-CORE delta, not the core itself. Revise predicted band to `[0.195 (HNeRV-core only) pending bolt-on candidate selection]`.

- **CC-4 unwind**: ADD paired-comparison probe (sister to FOUNDATION audit op-routable #8).

**(d) PROBE-DISAMBIGUATOR spec**

- Name: `tools/probe_sane_hnerv_scorer_loss_routing_paired.py` (NEW)
- Cost: $5 Modal A100 paired smoke (canonical-helper vs custom-routing variants at 25ep each)
- Method: train 2 variants of sane_hnerv (canonical `score_pair_components` vs HNeRV-custom inline scorer loss with `eval_roundtrip + EMA + diff-yuv6` explicit) on `upstream/videos/0.mkv`; compare contest_auth_eval scores
- Disambiguates: CC-4 directly; CC-1 partially (validates whether the integration scaffold is the failure source vs the substrate design itself)
- Output: `experiments/results/probe_sane_hnerv_scorer_routing_*/paired_result.json`

**(e) DISPATCH-READINESS POST-UNWIND**: Would CLEAR Catalog #270 (already passes per protocol's Tier 1/2/3 wiring), Catalog #292 (per-failure-mode assumption statements), and the META-CC-6 audit gate (smoke-passes ≠ substrate-works) via the 5-failure-mode classification.

**(f) ACTION SEQUENCE**
- IMMEDIATE ($0): land §Failure-mode-classification (5-attempt forensics) + §necessary-vs-sufficient + §predicted-band revision to 0.195 + paired-probe design
- NEAR-TERM ($5): run paired-comparison smoke (canonical vs custom routing)
- DISPATCH-READY ($5-15): post-probe full Modal A100 100ep on the WINNING variant

---

## Section 2 — Cross-substrate aggregate

**Total unwind cost estimate**:

| Substrate | Design-memo unwind ($0) | Probe ($) | Full dispatch ($) | Total |
|---|---|---|---|---|
| NSCS02 | $0 | $5 paired CPU | $5-15 Modal A100 selected ratio | $10-20 |
| C6 MDL-IBPS | $0 | $0 analytical Dykstra | $5-15 paired post-revision | $5-15 |
| ATW V1 | $0 | $3-5 shared probe | (Phase 2 council required first) | $3-5 |
| TT-L5 | $0 | $0 analytical Dykstra | $5-15 paired Linux+macOS | $5-15 |
| sane_hnerv | $0 | $5 Modal A100 paired | $5-15 winning variant | $10-20 |
| **TOTAL** | **$0** | **$13-15** | **$20-60** | **$33-75** |

**Predicted prevention value**:
- NSCS06-class falsifications prevented: 5 substrates × ~$5-15 per falsification = **$25-75 saved**
- META-CC-1 cargo-cult removed across 5 substrates × 1 prediction-class blunder each (cathedral autopilot routing on bad predicted bands) = additional ~$15-30 prevention
- META-CC-2 signal-axis-destruction prevention specifically for NSCS02 = avoids NSCS06-class 553× ratio at 100ep dispatch = **$15-30 prevention single-shot**
- **TOTAL prevention: $55-135**

**ROI**: $55-135 prevention / $33-75 spend = **1.6× to 4.1× ROI** at the lower bound; higher if any single NSCS06-class falsification is prevented at full 100ep dispatch cost.

**Per-substrate priority ranking** (by EV/$ × dispatch-readiness urgency):
1. **NSCS02** — highest priority (HIGH-RISK on 2 axes; structural NSCS06 analog; READY to dispatch on operator gate-flip)
2. **C6 MDL-IBPS** — dispatch-enabled; analytical Dykstra check is $0 and unblocks rev-of-predicted-band
3. **TT-L5** — same analytical Dykstra check tool reusable; macOS-CPU advisory already exists so the unwind is design-only
4. **sane_hnerv** — empirical-fragility signal already paid in 5 failed attempts; paired probe paid-but-cheap
5. **ATW V1** — design-complete; lowest dispatch-readiness urgency (NotImplementedError + Phase 2 council); probe shared with D4+Z4 so cost amortizes

**Shared infrastructure that reduces cost**:
- `tools/check_substrate_dykstra_feasibility.py` (NEW; analytical; $0; reusable across 4 substrates)
- `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (NEW; shared across ATW + D4 + Z4)
- Catalog #296 + #297 STRICT preflight gates (proposed in foundation audit) — make this audit's discipline STRUCTURAL going forward

---

## Section 3 — Operator-decision queue

Per-substrate binary verdict (per CLAUDE.md "Forbidden premature KILL": every verdict is UNWIND or DEFER-pending-redesign, ZERO killed):

| # | Substrate | Verdict | Action | Cost | Urgency |
|---|---|---|---|---|---|
| 1 | NSCS02 | **UNWIND** | land design-memo §predicted-band + §canonical-vs-unique + paired-downsample probe + `min_downsample_ratio:2` recipe field; mark research_only with reactivation criteria | $0 design + $5 probe | HIGH |
| 2 | C6 MDL-IBPS | **UNWIND** | land Dykstra-feasibility analytical check + design memo §predicted-band revision citing Z1 ablation (Catalog #219); within-class haircut required | $0 analytical | HIGH |
| 3 | ATW V1 | **UNWIND** (Phase 2 council DEFER for `_full_main` implementation) | land §canonical-vs-unique section + §predicted-band NULL-pending-probe + run shared H(latent\|scorer_class) probe | $0 design + $3-5 shared probe | MEDIUM |
| 4 | TT-L5 | **UNWIND** | land §Dykstra-feasibility composition rewrite + §predicted-target intersection-lower-bound; mark sister L1 design lanes research_only | $0 analytical | MEDIUM |
| 5 | sane_hnerv | **UNWIND** | land §Failure-mode-classification of 5 failed attempts + §necessary-vs-sufficient + paired scorer-loss-routing probe | $0 design + $5 paired probe | MEDIUM |

**Operator decisions required**:
- (D1) Approve UNWIND for all 5? → recommended YES; total cost ≤$15 in shared analytical+probe work; prevention 1.6-4.1× ROI
- (D2) Approve Catalog #296 + #297 STRICT preflight gate landing (proposed in foundation audit)? → recommended YES (warn-only at landing per atomicity rule)
- (D3) Approve shared `tools/check_substrate_dykstra_feasibility.py` canonical helper? → recommended YES; reusable across 4 substrates immediately + all future substrates
- (D4) Approve shared `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` probe across ATW + D4 + Z4? → recommended YES; $3-5 once disambiguates 3 substrates
- (D5) Sequence the 5 unwinds in parallel (sister-subagent fan-out per CLAUDE.md "Race-mode rigor inversion") or sequential? → recommended PARALLEL per the standing parallel-dispatch directive; each substrate's UNWIND is design-only and disjoint per Catalog #230

**Council review required** (per CLAUDE.md "Design decisions — non-negotiable"):
- C6 MDL-IBPS revised predicted band (Z1 within-class haircut application) is a council-grade tradeoff per "Design decisions" — Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary sextet pact sign-off
- TT-L5 composition rewrite (additive → subadditive) is a council-grade tradeoff for the same reason

**Compliance with this audit's own discipline** (per Catalog #292 self-application): every per-substrate row above states the operating-within assumption explicitly via the cargo-cult classification (HARD-EARNED PRESERVED + CARGO-CULTED ELIGIBLE per addendum). The Assumption-Adversary seat would challenge: *"Is the per-substrate audit itself a within-canonical-template exercise that suppresses substrate-specific creativity?"* — answer: yes, partially; each substrate's UNWIND DESIGN is template-driven, but the per-substrate probe specs ARE substrate-specific (NSCS02 paired-ratio / shared Dykstra / shared H-conditional / scorer-loss routing) so the design preserves the substrate-optimal engineering surface.

---

## Cross-references

- `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` — canonical framework
- `.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md` — foundation audit (this audit's predecessor)
- `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md` — template
- CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" + Catalog #291/#292
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- Catalog #219 (Z1 within-class density gate) + #227 (Tier C substrate-class promotion)
- Catalog #270 (umbrella dispatch optimization protocol)
- Catalog #272 (distinguishing feature integration contract)
- Catalog #287 (no docstring overstatement without evidence tag)

---

**END OF AUDIT LEDGER**
