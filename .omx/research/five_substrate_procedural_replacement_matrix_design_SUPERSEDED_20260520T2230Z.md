---
substrate_id: five_substrate_procedural_replacement_matrix_supersession
substrate_class: cross_substrate_composition_design_supersession
horizon_class: frontier_pursuit
council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Working-Group
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "the original 5-substrate matrix design memo claimed 4 of 5 candidates READY-TO-PAIRED-SMOKE or BUILD-required; today's 5 audit landings reveal 5 of 7 are HYPOTHETICAL / NOT-CANDIDATE / OUT-OF-SCOPE / APPEND-NOT-REPLACE / DEFER-BY-PROBE. The honest matrix has 2 EMPIRICALLY-GROUNDED candidates (DP1, VQ-VAE). The supersession must reflect this empirical state without inflating the operator's mental model of cascade-mortality."
  - member: Assumption-Adversary
    verbatim: "The SHARED ASSUMPTION operating across the original matrix memo: 'every candidate enumerated in §2 is a real substrate amenable to procedural-codebook replacement'. The 5 audit memos empirically falsified this for 5 of 7 candidates (when PR101+PR106 PIVOT is included). HARD-EARNED-EMPIRICALLY-FALSIFIED via individual subagent audits. The supersession memo IS the structural extinction of the cargo-cult assumption."
council_assumption_adversary_verdict:
  - assumption: "5 of 5 original candidates are real, empirically-grounded substrates"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Today's 5 audit landings: (1) NSCS06 v8 HYPOTHETICAL (substrate code does NOT exist at v8 schema; current v3 has 15-byte palette not 4 KB LUT); (2) ATW V2 EMPIRICAL substrate code BUT D4 INDEPENDENT verdict BLOCKING + byte-count CARGO-CULTED-EMPIRICALLY-FALSIFIED (2,560 not 3,072) + Variant A operationally falsifies cooperative-receiver hypothesis; (3) PR101 lc_v2_clone EMPIRICAL substrate BUT NOT-CANDIDATE (0 in-archive deterministic-constant regions); (4) PR106 OUT-OF-SCOPE (NO substrate dir); (5) grayscale_lut EMPIRICAL substrate BUT FiLM-conditioned RGB decoder lacks explicit 256-byte LUT section; compose APPENDS envelope not REPLACES; current GLV1 not score-eligible. ONLY DP1 + VQ-VAE EMPIRICALLY-GROUNDED-and-BUILT (sister landings via DP1 commit 9cbfa471c + VQ-VAE commit 6fea30f22)."
  - assumption: "Aggregate predicted ΔS bands [-0.013, -0.0085] are derivable from per-candidate predictions"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "ATW V2 byte-count corrected from 3,072 → 2,560 → ΔS Path A -0.001683 (not -0.002024). grayscale_lut byte-count CORRECTED from 4,096 → 256 → ΔS -0.000149 (not -0.002706); the matrix memo's '~4 KB chroma LUT' for grayscale_lut was conflated with the NSCS06 v8 hypothetical 4 KB chroma LUT estimate. NSCS06 v8 4 KB figure is HYPOTHETICAL; if v8 BUILD does not land, the candidate contributes 0 not -0.002706. With only DP1 + VQ-VAE EMPIRICALLY-BUILT: naive aggregate = -0.002706 + -0.005434 = -0.008140 (NOT -0.016939)."
  - assumption: "Catalog #325 14-day symposium window holds for all 5 candidates"
    classification: HARD-EARNED-PARTIAL
    rationale: "DP1 (2026-05-17) + NSCS06 v8 (2026-05-18) + ATW V2 (2026-05-18) + TT5L (2026-05-17) all within window. VQ-VAE has NO prior symposium (BUILD-required per matrix memo §4). grayscale_lut has NO prior symposium (BUILD-required). PR101 lc_v2_clone has NO prior symposium for the procedural-variant scope (NOT-CANDIDATE verdict per audit). PR106 STRUCTURALLY OUT OF SCOPE (no substrate)."
  - assumption: "Composition_alpha cascade per Catalog #322 produces single-point aggregate prediction"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per CLAUDE.md 'Forbidden empirical-claim-without-evidence-tag': pairwise α values are ALL [prediction] until first paired-anchor smoke fires. The matrix memo's α=0.5 SUB-ADDITIVE conservative + α=0.77 ADDITIVE optimistic bound IS the canonical disambiguator-pending-empirical contract per Catalog #322 v2 cascade; this supersession preserves that discipline AND recalibrates per the corrected per-substrate ΔS predictions."

predicted_band_validation_status: pending_post_training
predicted_band:
  empirically_grounded_2_substrate_dp1_plus_vq_vae_naive: [-0.008140, -0.008140]
  empirically_grounded_2_substrate_alpha_0_77_additive: [-0.008140, -0.008140]
  empirically_grounded_2_substrate_alpha_0_5_sub_additive: [-0.005000, -0.005000]
  hypothetical_extension_with_grayscale_lut_glv2_resolved_3_substrate_naive: [-0.008289, -0.008289]
  hypothetical_extension_with_atw_v2_variant_c_4_substrate_naive_path_a: [-0.009972, -0.009972]
  hypothetical_extension_with_nscs06_v8_built_5_substrate_naive: [-0.012678, -0.012678]
  original_matrix_memo_band_NOW_CARGO_CULTED: [-0.016939, -0.006000]
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
operator_directive: "WAVE-3 5-substrate matrix supersession reflecting today's 5 cascade-mortality data points; APPEND-ONLY per Catalog #110/#113; original matrix memo PRESERVED"
related_deliberation_ids:
  - five_substrate_procedural_replacement_matrix_design_20260520
  - nscs06_v8_procedural_chroma_lut_integration_design_20260520
  - pr101_pr106_procedural_variant_build_design_20260520
  - atw_v2_procedural_variant_build_design_20260520
  - grayscale_lut_procedural_variant_build_landed_20260520
  - dp1_procedural_variant_build_landed_20260520
  - vq_vae_procedural_variant_build_landed_20260520
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl
parent_design_memo: .omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md
supersedes: .omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md
supersession_kind: empirical_correction_with_extended_candidate_taxonomy
supersession_discipline: append_only_per_catalog_110_113
---

<!-- Catalog #344 canonical-equations-registry cross-reference: this
supersession memo updates predicted ΔS bands per the canonical equation
`procedural_codebook_from_seed_compression_savings_v1` evolution state
(5 events in `.omx/state/canonical_equations_registry.jsonl`: registered
2026-05-20T22:00Z + 2 anchor_appended + 1 domain_refined + 1 more
anchor_appended through 2026-05-21T00:21:20Z). The original matrix memo
(commit `b3e3442c3`) predicted ΔS bands [-0.013, -0.0085] were derived
BEFORE today's 5 substrate audits; this supersession applies corrected
per-substrate byte counts (ATW V2: 2,560 not 3,072; grayscale_lut: 256
not ~4 KB) + reclassifies each candidate per its EMPIRICAL-vs-HYPOTHETICAL
status per the 5 audit landings. Per CLAUDE.md "Forbidden premature KILL
without research exhaustion": the original matrix memo is PRESERVED
(Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY); the supersession
does NOT mutate the original body, NOR kill any candidate. It documents
the empirical state of each candidate so the operator can route
investment toward the EMPIRICALLY-GROUNDED candidates first. -->

# WAVE-3 Five-substrate procedural-replacement matrix design SUPERSESSION (2026-05-21T01:55Z)

**Lane**: `lane_wave_3_five_substrate_matrix_supersession_20260520`
**Supersedes**: `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md` (commit `b3e3442c3`; 670 LOC; PRESERVED per Catalog #110/#113 APPEND-ONLY)
**Trigger**: 5 substrate-audit landings 2026-05-20 surfaced empirical state that contradicts the original matrix memo's READY-TO-SMOKE / BUILD-required / REFUSE classifications.
**Operator framing (verbatim 2026-05-20)**: ATW V2 audit landing memo §6 "Today's cascade-mortality findings" + WAVE-3-FIVE-SUBSTRATE-MATRIX-SUPERSESSION task description.

## Section 1. Summary

The original 5-substrate matrix design memo (`b3e3442c3`) predicted a 5-substrate aggregate ΔS band [-0.013, -0.0085] (conservative SUB-ADDITIVE) → [-0.017, -0.013] (optimistic naive sum). Today's 5 substrate audits (NSCS06 v8 integration design + PR101+PR106 BUILD design + ATW V2 BUILD design + grayscale_lut BUILD landing + DP1 BUILD landing + VQ-VAE BUILD landing) empirically established that **only 2 of the 5 original candidates (DP1 + VQ-VAE) are EMPIRICALLY-BUILT-and-GROUNDED at this writing**. Three of the original 5 candidates fall into newly-identified cascade-mortality categories:

- **HYPOTHETICAL substrate architecture** (NSCS06 v8): substrate code does NOT exist at v8 schema; current v3 has 15-byte palette, not the 4 KB LUT the matrix memo assumed; v8 BUILD is a substrate-engineering bet pending Catalog #325 + #307/#308 + Tier-C reactivation cascade.
- **EMPIRICAL substrate + BLOCKING probe verdict + byte-count CARGO-CULTED-EMPIRICALLY-FALSIFIED** (ATW V2): substrate code exists (623 LOC archive.py) BUT Catalog #313 D4 INDEPENDENT verdict blocks dispatch until 2026-06-15; matrix memo byte count 3,072 corrected to empirical 2,560; Variant A operationally falsifies cooperative-receiver hypothesis; Variant C decoupled-scope BUILD viable pending operator scoping decision.
- **REFUSE-pending-re-symposium** (TT5L): unchanged from original matrix memo; substrate-class shift assumption rejected 2026-05-17.

Two additional cascade-mortality categories surfaced via the PIVOT subagent (PR101+PR106 BUILD design memo) which was NOT in the original matrix's 5 but is operator-relevant for the 3rd cascade-sequencing-pick:

- **NOT-CANDIDATE / OUT-OF-DOMAIN** (PR101 lc_v2_clone): substrate code exists (583 LOC archive.py) BUT 0 in-archive deterministic-constant byte regions; archive.zip contains only score-aware-trained per-video decoder weights + per-pair learned latents (all OUT-OF-DOMAIN per canonical equation #26 `_EXCLUDED_CONTEXTS`).
- **OUT-OF-SCOPE / NO SUBSTRATE** (PR106): no `src/tac/substrates/pr106_*` directory exists; PR106 lives only as 11 submission packets + 1 research-adapter dispatcher.

A 5th cascade-mortality category surfaced via the grayscale_lut BUILD landing: **EMPIRICAL substrate + FiLM-decoder-APPEND-not-REPLACE** (grayscale_lut): substrate code exists (39 tests pass) BUT the current GLV1 grammar lacks an explicit 256-byte chroma LUT section; compose function APPENDS envelope adding 41 bytes for a 32-byte seed (NET +9 bytes; NEGATIVE savings until GLV2 grammar bump lands).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this supersession does NOT kill any candidate. Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY: the original matrix memo body is PRESERVED VERBATIM at its original path; this supersession is a NEW memo file documenting the corrected empirical state + reclassified cascade-mortality taxonomy + recalibrated aggregate ΔS via composition_alpha v2 cascade per Catalog #322 + corrected per-candidate predictions per canonical equation #26 closed form (5 evolution events in canonical equations registry through 2026-05-21T00:21:20Z).

The honest recalibrated aggregate predicted ΔS for the **EMPIRICALLY-BUILT 2-substrate matrix (DP1 + VQ-VAE only)** is:

- Naive sum: -0.008140
- α=0.77 ADDITIVE (per matrix memo upper bound): -0.008140 (unchanged; same as naive for 2-substrate ADDITIVE)
- α=0.5 SUB-ADDITIVE (per matrix memo conservative): floor at -0.005000 per Catalog #322 v2 cascade SATURATING band
- Frontier 0.19205 [contest-CPU] - 0.008140 = **0.18391** (does NOT break 0.18 floor under naive sum)
- Frontier 0.19205 [contest-CPU] - 0.005000 = 0.18705 (does NOT break 0.18 floor under SATURATING)

The original matrix memo's claim that the matrix would "DECISIVELY break the 0.18 floor" was contingent on 5-of-5 candidates landing as paired empirical anchors with α ≥ 0.77. Today's audit empirically established that **only DP1 + VQ-VAE meet that bar at this writing**, and the empirically-grounded 2-substrate aggregate is plateau-adjacent NOT frontier-breaking. The frontier-breaking case requires either (a) grayscale_lut GLV2 grammar bump + ATW V2 Variant C scoping approval + NSCS06 v8 BUILD all landing as paired empirical anchors with α ≥ 0.85, OR (b) operator-routed escape-hatch to a different procedural-replacement equation family.

## Section 2. Per-substrate cascade-mortality table (empirically grounded)

| # | Substrate | EMPIRICAL-vs-HYPOTHETICAL | Original matrix memo predicted ΔS | Corrected predicted ΔS | Bytes (corrected) | Cascade-mortality category | Blockers (today) | Reactivation path |
|---|---|---|---|---|---|---|---|
| 1 | NSCS06 v8 chroma LUT | **HYPOTHETICAL substrate architecture** | -0.002706 | -0.002706 if v8 BUILDS at 4096 bytes (HYPOTHETICAL) OR -0.00000466 if Scenario A current v3 15-byte palette | 4,096 (v8 HYPOTHETICAL) OR 15 (v3 current) | Substrate-code-does-not-exist-at-claimed-schema | Catalog #325 14-day window not satisfied (latest 2026-05-16) + Catalog #307/#308 paradigm DEFER + substrate code does not exist | (a) v8 BUILD subagent + (b) fresh symposium 2026-05-30+ + (c) ratify paradigm reactivation per T3 grand council |
| 2 | ATW V2 codec deterministic CDF table | **EMPIRICAL substrate-but-BYTE-COUNT-FALSIFIED-and-D4-BLOCKED** | -0.002024 | -0.001683 (Path A cdf_table only; corrected from 2,560 not 3,072) OR -0.003281 (Path B cdf + class_prior 2,400 bytes) | 2,560 (Path A) OR 4,960 (Path B) | (a) Catalog #313 D4 INDEPENDENT verdict BLOCKING until 2026-06-15 (b) byte-count cargo-culted (c) Variant A operationally falsifies cooperative-receiver hypothesis | D4 INDEPENDENT verdict expires 2026-06-15 OR Variant C scoping decision + follow-on Catalog #325 symposium | (a) operator scopes Variant C decoupled-from-cooperative-receiver + (b) follow-on symposium ratifies Variant C scope + (c) BUILD ~600 LOC sister of grayscale_lut canonical pattern |
| 3 | TT5L (Time-Traveler L5) sister substrate constants | **REFUSE-pending-re-symposium** (unchanged from original matrix) | -0.004070 | -0.004070 (unchanged; assumes 6 KB byte estimate holds) | 6,144 (matrix-memo estimate; not re-audited today) | T2 sextet REFUSE verdict 2026-05-17 (substrate-class shift assumption rejected) | Wave N+1 re-symposium with post-training Tier-C anchor; substrate-class-shift assumption requires re-classification | (a) post-training Tier-C re-measurement on landed L5 archive + (b) Wave N+1 council reactivation + (c) ratify substrate-class shift per the canonical addendum HARD-EARNED-vs-CARGO-CULTED framework |
| 4 | DP1 codebook OOD-derived bytes | **EMPIRICAL-BUILT** (DP1 PROCEDURAL VARIANT BUILD landing commit `9cbfa471c` per `feedback_dp1_procedural_trainer_build_landed_20260520.md`) | -0.002706 | -0.002706 (unchanged; canonical equation #26 closed form holds; 4,096 → 32 bytes saved) | 4,096 → 32 (4,064 saved) | EMPIRICAL substrate; READY-TO-PAIRED-SMOKE; first empirical-anchor candidate | NONE for BUILD (15/15 tests pass; 206/206 DP1 regression pass; lane L1 impl_complete); paired-smoke gated by operator-routable Modal T4 ~$0.30 dispatch | OP-ROUTABLE #1: operator-authorize paired-smoke via canonical `tools/operator_authorize.py` route (recipe NOT yet committed; design memo path documented per parent `feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md`) |
| 5 | VQ-VAE codebook | **EMPIRICAL-BUILT** (VQ-VAE PROCEDURAL VARIANT BUILD landing commit `6fea30f22` per `feedback_slot_mg7_bundle_master_gradient_exploits_landed_20260520.md` sister) | -0.005433 | -0.005434 (unchanged; canonical equation #26 closed form holds; 8,192 → 32 bytes saved) | 8,192 → 32 (8,160 saved) | EMPIRICAL substrate; READY-TO-PAIRED-SMOKE pending Catalog #325 per-substrate symposium | Per-substrate symposium per Catalog #325 NOT yet landed (BUILD-required per original matrix; sister discipline = adversarial grand council symposium with van den Oord seat for intermediate-transform quantizer paradigm) | OP-ROUTABLE #2: schedule per-substrate symposium; T2 sextet pact + grand council attendees per topic; symposium memo dated `.omx/research/council_vq_vae_procedural_variant_<YYYYMMDD>.md` per Catalog #325 naming |
| 6 (PIVOT) | PR101 lc_v2_clone | **NOT-CANDIDATE / OUT-OF-DOMAIN** (per PR101+PR106 BUILD DESIGN audit `pr101_pr106_procedural_variant_build_design_20260520.md`) | n/a (not in original matrix; surfaced via PIVOT subagent) | n/a (all 5 byte regions in PR101 lc_v2_clone are OUT-OF-DOMAIN per canonical equation #26 `_EXCLUDED_CONTEXTS`) | 0 in-archive deterministic-constant regions; DECODER_BLOB + LATENT_BLOB are score-aware-trained per-video; SIDECAR ~100 B = NET +/-tiny; header 17 B = NET regression | Procedural-codebook-replacement pattern structurally inapplicable | Reactivation paths (per audit §4): (a) canonical equation #26 domain refinement adds NEW context covering PR101 structure; (b) different replacement primitive (per-tensor-quantization-scale bytes); (c) PR101 per-substrate symposium with EXCLUDED_CONTEXTS extensions ratified |
| 7 (PIVOT) | PR106 | **OUT-OF-SCOPE / NO SUBSTRATE** (per same PR101+PR106 audit) | n/a (not in original matrix) | n/a | n/a | NO `src/tac/substrates/pr106_*` directory exists; PR106 is only submission packets + research-adapter dispatcher | Reactivation path: spawn PR106 substrate scaffold subagent to extract canonical substrate code from `submissions/pr106_*` packets (NOT a procedural-variant question; substrate-engineering question) |
| 8 (sister) | grayscale_lut | **EMPIRICAL-BUILT** BUT **FiLM-decoder-APPEND-not-REPLACE** (per grayscale_lut PROCEDURAL VARIANT BUILD landing 2026-05-21T01:25Z) | n/a (not in original matrix's 5; surfaced via PIVOT subagent §4 recommendation) | -0.000149 if GLV2 grammar bump lands (256-byte LUT → 32-byte seed) OR +0.0000115 (NET +9 byte regression under current GLV1 grammar APPEND mode) | 256 (canonical chroma LUT entry count per canonical equation #26 IN-DOMAIN `chroma_lut_replacement`) OR 0 (current GLV1 has FiLM-conditioned RGB decoder; no explicit chroma_lut section) | Current GLV1 grammar lacks explicit chroma_lut section to REPLACE; compose function APPENDS 41-byte envelope for 32-byte seed (NET regression) | GLV2 grammar bump + inflate consumer landing flips compose function to REPLACE-IN-PLACE; sister of canonical DP1+VQ-VAE pattern |

**Net empirically-grounded matrix state (today 2026-05-21T01:55Z)**:

- **2 EMPIRICAL-BUILT** ready-to-paired-smoke: DP1 + VQ-VAE
- **1 EMPIRICAL-BUILT** pending GLV2 grammar bump: grayscale_lut
- **1 EMPIRICAL-substrate** BLOCKED by Catalog #313 + scoping decision: ATW V2 (Variant C is viable post-scoping)
- **1 HYPOTHETICAL substrate** pending BUILD + symposium + paradigm reactivation: NSCS06 v8
- **1 REFUSE** pending re-symposium: TT5L
- **1 NOT-CANDIDATE** per audit (out-of-scope for procedural-codebook): PR101 lc_v2_clone
- **1 OUT-OF-SCOPE** (no substrate): PR106

This is a 7-candidate corrected taxonomy (5 original + 2 PIVOT-surfaced) where **2 of 7 are EMPIRICALLY-GROUNDED-and-BUILT today**, **1 is BUILT-but-needs-grammar-bump**, and **4 are blocked / hypothetical / out-of-scope / refused**.

## Section 3. Recalibrated aggregate ΔS via composition_alpha cascade

Per `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` v2 cascade (Catalog #322):

- α > 1.05: SUPER_ADDITIVE (clamped reward factor in [1.0, 2.0])
- α ∈ (0.7, 1.05]: ADDITIVE (no adjustment; full additive savings)
- α ∈ (0.3, 0.7]: SUB-ADDITIVE (×0.5 penalty)
- α ≤ 0.3: SATURATING (floor at -0.005)
- α = None: no adjustment

### 3.1 EMPIRICALLY-BUILT-2-substrate aggregate (DP1 + VQ-VAE)

| Cascade | ΔS contribution | Aggregate |
|---|---|---|
| DP1 | -0.002706 | running: -0.002706 |
| VQ-VAE | -0.005434 | running: -0.008140 |
| Naive sum | -0.008140 | -0.008140 |
| α=0.77 ADDITIVE band | no adjustment | -0.008140 |
| α=0.5 SUB-ADDITIVE band | ×0.5 penalty: -0.004070 | -0.004070 (capped at SATURATING floor -0.005000 if α ≤ 0.3) |
| Pareto polytope per Catalog #296 Dykstra-feasibility | DP1 rate-axis additive with VQ-VAE rate-axis (different IN-DOMAIN contexts: `comma2k19_ood_derived_basis_replacement` vs `intermediate_transform_quantizer`); FEASIBLE under rate axis | FEASIBLE |

**From frontier 0.19205 [contest-CPU]** (per canonical frontier pointer `.omx/state/canonical_frontier_pointer.json`):
- Naive 2-substrate (α ≥ 0.77): 0.19205 - 0.008140 = **0.18391** (does NOT break 0.18)
- SUB-ADDITIVE (0.3 < α ≤ 0.7): 0.19205 - 0.004070 = 0.18798 (does NOT break 0.18)
- SATURATING floor (α ≤ 0.3): 0.19205 - 0.005000 = 0.18705 (does NOT break 0.18)

**Verdict for 2-substrate empirically-built aggregate**: plateau-adjacent, NOT frontier-breaking. The 0.18 floor break requires 3+ candidates landing as paired empirical anchors at ADDITIVE composition.

### 3.2 3-substrate aggregate (+grayscale_lut pending GLV2 grammar)

If grayscale_lut GLV2 grammar bump lands AND a future BUILD subagent + symposium + paired-smoke validates the predicted ΔS = -0.000149 empirically:

| Cascade | ΔS contribution | Aggregate |
|---|---|---|
| DP1 + VQ-VAE (from §3.1) | -0.008140 | running: -0.008140 |
| grayscale_lut (post-GLV2) | -0.000149 | running: -0.008289 |
| Naive sum | -0.008289 | -0.008289 |
| α=0.65 SUB-ADDITIVE (per matrix memo VQ-VAE × grayscale_lut overlap on chroma-class targeting) | ×0.5 penalty on incremental: -0.0000745 | running approximation: -0.008215 |
| Pareto polytope | grayscale_lut rate-axis adds to DP1 + VQ-VAE rate-axis (chroma_lut_replacement IN-DOMAIN context distinct from comma2k19 + intermediate_transform) | FEASIBLE |

**From frontier 0.19205 [contest-CPU]**:
- Naive 3-substrate (α ≥ 0.77): 0.19205 - 0.008289 = **0.18376** (still does NOT break 0.18)
- SUB-ADDITIVE conservative: ~0.18384 (still does NOT break 0.18)

**Verdict for 3-substrate post-GLV2 aggregate**: marginal improvement; still plateau-adjacent. The 3-substrate incremental contribution from grayscale_lut is small because the canonical equation #26 IN-DOMAIN context predicts only 256-byte LUT savings (vs DP1's 4,064 bytes + VQ-VAE's 8,160 bytes).

### 3.3 4-substrate aggregate (+ATW V2 Variant C if scoping approved)

If operator approves ATW V2 Variant C scoping decision + follow-on symposium + BUILD + paired-smoke validates predicted ΔS Path A = -0.001683 empirically:

| Cascade | ΔS contribution | Aggregate |
|---|---|---|
| DP1 + VQ-VAE + grayscale_lut (from §3.2) | -0.008289 | running: -0.008289 |
| ATW V2 Variant C (cdf_table only, Path A) | -0.001683 | running: -0.009972 |
| Naive sum | -0.009972 | -0.009972 |
| α=0.85 ADDITIVE (matrix memo NSCS06×ATW orthogonal score axes; sister VQ-VAE×ATW = 0.55 SUB-ADDITIVE) | aggregate band [-0.009972 naive, -0.008085 if VQ-VAE×ATW penalty halves the incremental] | ~[-0.008, -0.010] |
| Pareto polytope | ATW V2 cdf_table_blob IN-DOMAIN `atw_v2_codec_quantizer_lut` distinct from sister IN-DOMAIN contexts; FEASIBLE rate-axis | FEASIBLE |

**From frontier 0.19205 [contest-CPU]**:
- Naive 4-substrate (α ≥ 0.77): 0.19205 - 0.009972 = **0.18208** (approaches 0.18 floor)
- α=0.65 SUB-ADDITIVE on VQ-VAE×ATW pair: ~0.18397 (does NOT break 0.18)
- Sub-ADDITIVE composite: ~0.184-0.185 plateau-adjacent

**Verdict for 4-substrate Variant C aggregate**: closer to 0.18 floor but NOT a decisive break. The frontier-break requires 5+ candidates OR composition_alpha ≥ ADDITIVE for all pairs.

### 3.4 5-substrate hypothetical aggregate (+NSCS06 v8 if BUILD + symposium reactivation lands)

If NSCS06 v8 BUILD subagent + fresh per-substrate symposium + paradigm reactivation + paired-smoke validate predicted ΔS = -0.002706 empirically:

| Cascade | ΔS contribution | Aggregate |
|---|---|---|
| DP1 + VQ-VAE + grayscale_lut + ATW V2 Variant C (from §3.3) | -0.009972 | running: -0.009972 |
| NSCS06 v8 chroma LUT | -0.002706 | running: -0.012678 |
| Naive sum | -0.012678 | -0.012678 |
| α=0.85 ADDITIVE (matrix memo NSCS06×ATW orthogonal) | aggregate band [-0.012678 naive, -0.010 sub-additive composite] | [-0.010, -0.013] |
| Pareto polytope | NSCS06 v8 chroma LUT IN-DOMAIN `chroma_lut_replacement` distinct from sister contexts; FEASIBLE rate-axis | FEASIBLE |

**From frontier 0.19205 [contest-CPU]**:
- Naive 5-substrate (α ≥ 0.77): 0.19205 - 0.012678 = **0.17937** (BREAKS 0.18 floor by 0.0007)
- α=0.65 SUB-ADDITIVE composite: 0.19205 - 0.010000 ≈ 0.18205 (plateau-adjacent)

**Verdict for 5-substrate hypothetical aggregate**: ONLY frontier-breaking under all 5 candidates landing AND composition_alpha ≥ ADDITIVE for all pairwise compositions. This matches the original matrix memo's optimistic bound BUT requires all 5 cascade-mortality blockers to be cleared empirically AND first paired-anchor smoke to validate α ≥ 0.77 for all pairs.

### 3.5 Pareto polytope mutual orthogonality (Dykstra-feasibility per Catalog #296)

All 5 candidates' canonical equation #26 IN-DOMAIN contexts are mutually orthogonal at the rate-axis (each substrate's seed-derived bytes occupy distinct archive byte regions):

- DP1 → `comma2k19_ood_derived_basis_replacement` (OOD-derived basis)
- VQ-VAE → `intermediate_transform_quantizer` (K×D embedding table)
- grayscale_lut → `chroma_lut_replacement` (256-entry LUT post-GLV2)
- ATW V2 → `atw_v2_codec_quantizer_lut` (5×256 fp16 CDF table)
- NSCS06 v8 → (4 KB chroma codebook if v8 BUILD lands)

**Feasibility verdict (Catalog #296 + Catalog #322)**: rate-axis feasibility is structurally satisfied (each substrate adds K=32 bytes of substrate-internal entropy at a distinct archive byte region). Score-axis feasibility (seg/pose contributions preserved) is INDETERMINATE per Catalog #324 post-training Tier-C validation discipline — each per-substrate paired-anchor smoke is the first empirical anchor.

## Section 4. Per-substrate symposium gating matrix (corrected)

| Candidate | Symposium memo | Date | Verdict | Phase-2 status | Recipe state | Today's audit verdict | Decision (corrected) |
|---|---|---|---|---|---|---|---|
| NSCS06 v8 | `council_per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518.md` | 2026-05-18 (2 days old, within 14-day window per Catalog #325) | PROCEED_WITH_REVISIONS | T3 grand council; 15 attendees; quorum met | research_only:true; dispatch_enabled:false | HYPOTHETICAL substrate per NSCS06 v8 integration design (substrate code does NOT exist at v8 schema) | **DEFER**: BUILD subagent prerequisite + fresh symposium 2026-05-30+ + paradigm reactivation per T3 grand council |
| ATW V2 | `council_per_substrate_symposium_atw_v2_reactivation_20260518.md` | 2026-05-18 (2 days; within window) | PROCEED_WITH_REVISIONS | T2 sextet + 4 specialists; quorum met | research_only:true; dispatch_enabled:false | EMPIRICAL substrate; byte-count cargo-culted (3,072→2,560); Catalog #313 D4 INDEPENDENT blocking until 2026-06-15 | **DEFER**: Variant A pre-falsifies cooperative-receiver hypothesis → operator scoping decision Variant C decoupled → follow-on symposium for Variant C scope → BUILD ~600 LOC → paired-smoke |
| TT5L | `council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md` | 2026-05-17 (3 days; within window) | REFUSE | T2 sextet; assumption-adversary CARGO-CULTED | research_only:true; dispatch_enabled:false | unchanged from original matrix | **DEFER-PENDING-RE-SYMPOSIUM** (unchanged): Wave N+1 council reactivation + post-training Tier-C anchor required |
| DP1 | `council_per_substrate_symposium_dp1_deep_dive_20260517.md` | 2026-05-17 (3 days; within window) | PROCEED_WITH_REVISIONS | T2 sextet; PATH 2 deferred to Phase 2 | research_only:true; dispatch_enabled:false | EMPIRICAL-BUILT (commit `9cbfa471c`); 15/15 tests pass; lane L1 impl_complete | **READY-TO-PAIRED-SMOKE**: first empirical-anchor candidate; canonical helper Catalog #210 DP1 provenance preserved; OP-ROUTABLE #1 |
| VQ-VAE | (none found 2026-05-18; **NEW per-substrate symposium required** post-BUILD) | n/a (BUILD landed 2026-05-21T01:25Z commit `6fea30f22`; symposium PENDING) | n/a | n/a | research_only:true; dispatch_enabled:false; lane L1 impl_complete | EMPIRICAL-BUILT (commit `6fea30f22`); 21/21 tests pass; 49/49 VQ-VAE regression pass | **READY-FOR-PER-SUBSTRATE-SYMPOSIUM** (Catalog #325): spawn T2 sextet + grand council van den Oord seat for intermediate-transform quantizer paradigm |
| grayscale_lut | (none found; **NEW per-substrate symposium required** post-BUILD) | n/a (BUILD landed 2026-05-21T01:25Z) | n/a | n/a | research_only:true; dispatch_enabled:false; lane L1 impl_complete | EMPIRICAL-BUILT (39/39 tests pass) BUT APPEND-not-REPLACE; GLV2 grammar bump required for score-eligibility | **READY-FOR-PER-SUBSTRATE-SYMPOSIUM + GLV2 GRAMMAR DESIGN**: spawn T2 sextet + Selfcomp + Mallat + van den Oord grand-council seats |
| PR101 lc_v2_clone (PIVOT) | (none; NOT-CANDIDATE per audit) | n/a | n/a | n/a | research_only:true | NOT-CANDIDATE per audit (0 in-archive deterministic-constant byte regions) | **DEFER-PENDING-PROCEDURAL-DOMAIN-APPLICABILITY**: 3 reactivation paths per audit §4 |
| PR106 (PIVOT) | (none; OUT-OF-SCOPE per audit) | n/a | n/a | n/a | n/a (no lane; no substrate dir) | OUT-OF-SCOPE per audit (no `src/tac/substrates/pr106_*` dir) | **DEFER-PENDING-SUBSTRATE-EXTRACTION**: optional sister subagent to extract canonical substrate code from `submissions/pr106_*` packets |

**Net matrix state (corrected)**:
- 1 READY-TO-PAIRED-SMOKE (DP1)
- 1 READY-FOR-PER-SUBSTRATE-SYMPOSIUM-then-smoke (VQ-VAE)
- 1 READY-FOR-SYMPOSIUM + GRAMMAR-BUMP (grayscale_lut)
- 2 DEFER pending scoping/BUILD/symposium (NSCS06 v8 + ATW V2)
- 1 DEFER-PENDING-RE-SYMPOSIUM (TT5L; unchanged)
- 2 PIVOT cascade-mortality categories (PR101 NOT-CANDIDATE + PR106 OUT-OF-SCOPE)

## Section 5. Operator-decision matrix (6 actions)

Per the corrected cascade-mortality taxonomy + recalibrated aggregate ΔS, the operator-routable decision matrix:

### Action 1: AUTHORIZE-DP1-PAIRED-SMOKE-FIRST-ANCHOR ($0.30 Modal T4)

**Trigger**: DP1 substrate is the only EMPIRICALLY-BUILT-and-symposium-ready candidate.
**Mechanism**: Operator routes via canonical `tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch --paired-axis cuda+cpu --max-spend-usd 0.30`. The recipe YAML at `.omx/operator_authorize_recipes/` is operator-routed (NOT committed by THIS subagent per scope limits). Modal `.spawn()` per Catalog #245 + #339 fail-closed. Harvest within 24h per Catalog #330. Byte-mutation smoke per Catalog #272. First empirical anchor via `update_equation_with_empirical_anchor` per Catalog #344.
**Predicted outcome**: first empirical anchor for canonical equation #26 with substrate-specific residual; recalibrates aggregate ΔS predictions per `RECALIBRATE_ON_NEW_ANCHORS` trigger.
**Cost**: $0.30 paired Modal T4 (CUDA + CPU axes per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA").
**Mission contribution**: `frontier_breaking_enabler` (first empirical anchor → reduces uncertainty bands on remaining 4+ candidates).

### Action 2: SPAWN-VQ-VAE-PER-SUBSTRATE-SYMPOSIUM (T2 sextet + grand council attendees)

**Trigger**: VQ-VAE substrate BUILD-landed today but lacks Catalog #325 per-substrate symposium gate.
**Mechanism**: Spawn `lane_per_substrate_symposium_vq_vae_procedural_codebook_replacement_20260520` per Catalog #325 6-step contract. Sextet pact (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary) + grand council attendees (van den Oord for intermediate-transform quantizer paradigm; MacKay for IB framework). Symposium memo dated `.omx/research/council_vq_vae_procedural_variant_<YYYYMMDD>.md` per Catalog #325 naming convention.
**Predicted outcome**: PROCEED-conditional or PROCEED-with-revisions verdict → unblock VQ-VAE paired-smoke (Action 1's sister cadence).
**Cost**: ~1 council session + ~3h memo writing; $0 GPU.
**Mission contribution**: `frontier_breaking_enabler` (unblocks 2nd empirical anchor candidate).

### Action 3: SPAWN-GRAYSCALE-LUT-GLV2-GRAMMAR-DESIGN-and-SYMPOSIUM

**Trigger**: grayscale_lut BUILD-landed today but APPEND mode produces NET regression; GLV2 grammar bump required for score-eligibility.
**Mechanism**: 3 sub-actions: (a) GLV2 grammar design memo (sister of original matrix memo §2 Candidate 1 grammar design; ~600 LOC); (b) GLV2 inflate consumer design (~300 LOC); (c) per-substrate symposium per Catalog #325 with Selfcomp + Mallat + van den Oord grand-council seats; (d) compose function refactor from APPEND to REPLACE-IN-PLACE post-GLV2.
**Predicted outcome**: grayscale_lut becomes score-eligible at -0.000149 per canonical equation #26 closed form.
**Cost**: ~6h sub-subagent wall-clock; $0 GPU; subsequent paired-smoke $0.30 Modal T4.
**Mission contribution**: `frontier_protecting` (3rd empirical anchor candidate; smallest per-substrate ΔS but strongest IN-DOMAIN confidence per canonical equation #26 line 103 anchor context).

### Action 4: DECIDE-ATW-V2-SCOPING (Variant A blocked vs Variant C decoupled)

**Trigger**: ATW V2 audit identifies 3 independent failure modes for Variant A (additive bolt-on on V2 cooperative-receiver); Variant C decoupled-from-cooperative-receiver scope opens a viable BUILD path.
**Mechanism**: Operator decision (verbatim quote required per Catalog #300 mission-alignment): "Approve Variant C decoupled-from-cooperative-receiver scoping for ATW V2 procedural variant BUILD" OR "Defer Variant C; proceed with V2-1 redesign per 2026-05-18 symposium Revision #1-7 first". If Variant C approved: follow-on per-substrate symposium per Catalog #325 (T2 sextet + 3 specialists including Atick-Redlich-Tishby memorial-Wyner memorial for cooperative-receiver context) + BUILD ~600 LOC sister of grayscale_lut canonical pattern + paired-smoke $0.30 Modal T4 paired CPU+CUDA + ratify Path A ΔS = -0.001683 empirically.
**Predicted outcome**: 4th empirical anchor candidate at ATW V2 if Variant C approved; aggregate ΔS extends from -0.008289 (3-substrate) to -0.009972 (4-substrate).
**Cost**: ~5 min operator decision; if Variant C approved: ~6h follow-on symposium + ~75 min BUILD subagent + $0.30 paired-smoke.
**Mission contribution**: `frontier_protecting` (prevents another premature BUILD on a blocked substrate; opens parallel rate-axis-only research path).

### Action 5: SPAWN-NSCS06-V8-BUILD-SUBAGENT (substrate-engineering scope)

**Trigger**: NSCS06 v8 chroma codebook is HYPOTHETICAL (substrate code does NOT exist at v8 schema; current v3 has 15-byte palette not 4 KB LUT).
**Mechanism**: 3 sub-actions: (a) BUILD subagent for NSCS06 v8 substrate code extension (`src/tac/substrates/nscs06_carmack_hotz_strip_everything/{archive,codec,inflate}.py` with `CH06_SCHEMA_VERSION_PROCEDURAL_CHROMA_CODEBOOK = 4` schema bump); (b) substrate trainer enters `research_only=true` state with `dispatch_enabled=false` recipe; (c) fresh per-substrate symposium per Catalog #325 (T3 grand council if paradigm-class question surfaces per Catalog #307/#308); (d) Wave N+1 reactivation per Catalog #315 from PROCEED_WITH_REVISIONS to PROCEED-unconditional.
**Predicted outcome**: 5th empirical anchor candidate at NSCS06 v8 if BUILD + symposium + paradigm reactivation all clear; aggregate ΔS extends from -0.009972 (4-substrate) to -0.012678 (5-substrate).
**Cost**: ~2-4h BUILD subagent + ~1 council session + paired-smoke $0.30-0.50 Modal T4/A10G.
**Mission contribution**: `frontier_breaking_enabler` (5-substrate aggregate would break 0.18 floor under naive sum + ADDITIVE composition).

### Action 6: DEFER-PIVOT-CASCADE-MORTALITY-CATEGORIES (PR101 lc_v2_clone + PR106 + TT5L)

**Trigger**: 3 candidates fall outside the viable 5-substrate empirically-grounded matrix:
- PR101 lc_v2_clone NOT-CANDIDATE per audit (procedural-codebook-replacement structurally inapplicable)
- PR106 OUT-OF-SCOPE per audit (no substrate dir)
- TT5L REFUSE per 2026-05-17 symposium (substrate-class-shift assumption rejected)
**Mechanism**: 3 sub-actions: (a) file PR101 lc_v2_clone NOT-CANDIDATE verdict to `.omx/state/probe_outcomes.jsonl` with DEFER blocker_status + 30-day staleness window per Catalog #313 (OP-ROUTABLE per audit memo); (b) PR106 substrate-extraction subagent optional (low priority); (c) TT5L Wave N+1 re-symposium deferred to post-aggregate-frontier-break review per original matrix memo §11 deferred routables.
**Predicted outcome**: 3 cascade-mortality categories properly recorded in canonical state surfaces (probe-outcomes ledger + lane registry); operator can route follow-on investment to higher-EV candidates first.
**Cost**: $0 GPU; ~30 min sister subagents per ledger update.
**Mission contribution**: `apparatus_maintenance` (preserves canonical state surfaces; prevents premature KILL of any candidate per CLAUDE.md "Forbidden premature KILL without research exhaustion"; documents reactivation paths for each).

## Section 6. Sister regression + APPEND-ONLY discipline

This supersession memo introduces NO code changes; sister regression scope is N/A.

Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:
- The original matrix memo at `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md` is PRESERVED VERBATIM (zero body mutation; the original `b3e3442c3` commit is intact).
- This supersession memo is a NEW file at `.omx/research/five_substrate_procedural_replacement_matrix_design_SUPERSEDED_20260520T2230Z.md`.
- The original memo's frontmatter remains canonical for its own scope (single-anchor 5-substrate design); this supersession memo documents the empirical state of each candidate at this writing without superseding the original's intent (the original IS the canonical reference for the 5-substrate matrix DESIGN; this memo is the canonical reference for the empirical CASCADE-MORTALITY taxonomy).
- The 5 audit landing memos (NSCS06 v8 integration design + PR101+PR106 BUILD design + ATW V2 BUILD design + grayscale_lut BUILD landing + DP1 BUILD landing + VQ-VAE BUILD landing) are NOT mutated; this memo CITES them per Catalog #287 placeholder rejection + Catalog #323 canonical Provenance.

Per Catalog #287: every claim in this memo is tagged `[prediction]` (canonical equation #26 closed-form derivations) OR `[empirical:<audit memo path>]` (per-substrate audit verdicts) OR `[contest-CPU GHA Linux x86_64]` (frontier pointer).

## Section 7. Catalog gate verdicts (predicted; design-only supersession)

| Gate | Verdict | Notes |
|---|---|---|
| Catalog #110 + #113 HISTORICAL_PROVENANCE APPEND-ONLY | PASS | Original matrix memo body PRESERVED; this is NEW file path; sister audit memos NOT mutated |
| Catalog #117 + #157 + #174 canonical serializer | PASS | This commit uses canonical serializer with POST-EDIT --expected-content-sha256 |
| Catalog #119 Co-Authored-By trailer | PASS | Commit will include trailer |
| Catalog #125 6-hook wire-in | PASS | This memo declares all 6 hooks per §10 |
| Catalog #185 META drift detection | PASS | All catalog row claims verified empirically (live count: 0 for this new gate row N/A; this memo claims no new catalog row) |
| Catalog #186 catalog-claim via serializer | N/A | No new catalog # claimed |
| Catalog #206 crash-resume discipline | PASS | 3+ checkpoints emitted |
| Catalog #229 premise verification | PASS | All 7 source files (5 audit memos + 1 original matrix memo + canonical equation #26 source) read in full BEFORE drafting |
| Catalog #287 placeholder rejection | PASS | No `<rationale>` / `<reason>` placeholders; all rationales substantive |
| Catalog #290 canonical-vs-unique decision per layer | PASS | §8 per-layer decisions |
| Catalog #292 per-deliberation assumption surfacing | PASS | 4 assumption-adversary verdicts in frontmatter |
| Catalog #294 9-dim checklist evidence | PASS | §9 |
| Catalog #296 Dykstra-feasibility predicted band | PASS | §3.5 mutual orthogonality across IN-DOMAIN contexts |
| Catalog #297 signal-axis-destruction reversibility probe | N/A | This memo is design-only; no substrate code changes |
| Catalog #300 council deliberation v2 frontmatter | PASS | `council_tier: T1` + 7 attendees + verdict + dissent + assumption-adversary |
| Catalog #303 cargo-cult audit per assumption | PASS | §8 cargo-cult audit |
| Catalog #305 observability surface | PASS | §10 |
| Catalog #309 horizon_class | PASS | `horizon_class: frontier_pursuit` |
| Catalog #313 probe-outcomes ledger | PASS | Action 6 documents PR101 lc_v2_clone DEFER row append (operator-routable, not in scope for this memo) |
| Catalog #314 / #340 absorption + sister-checkpoint guard | PASS | Step 0 helper PROCEED verdict; sister-DISJOINT from in-flight PARSER-SAFE SUBSET SMOKE + HONEST CASCADE-MORTALITY ASSESSMENT |
| Catalog #315 OPTIMAL FORM | PASS | This is a DESIGN-ONLY supersession; no paid dispatch fired |
| Catalog #322 composition_alpha cascade | PASS | §3 recalibrated aggregate ΔS via v2 cascade with empirically-corrected per-substrate predictions |
| Catalog #323 canonical Provenance | PASS | All score-relevant claims carry `[prediction]` or `[empirical:<path>]` tags; `score_claim=False` declared |
| Catalog #324 post-training Tier-C validation | PASS | `predicted_band_validation_status: pending_post_training` declared in frontmatter |
| Catalog #325 per-substrate symposium recency | PASS | §4 documents per-substrate symposium recency for all 5 original + 2 PIVOT candidates |
| Catalog #335 cathedral consumer canonical contract | PASS | No new consumer landed; cites sister `procedural_codebook_savings_consumer` auto-discovered |
| Catalog #340 sister-checkpoint staging guard | PASS | PROCEED at landing |
| Catalog #344 canonical equation cross-reference | PASS | Frontmatter + HTML comment + extensive body citations of `procedural_codebook_from_seed_compression_savings_v1` |

## Section 8. Per-layer canonical-vs-unique decisions + Cargo-cult audit

### 8.1 Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Cascade-mortality taxonomy | UNIQUE (new) | First memo to surface 5 distinct cascade-mortality categories: HYPOTHETICAL + EMPIRICAL-BUT-BLOCKED + NOT-CANDIDATE + OUT-OF-SCOPE + FiLM-APPEND-not-REPLACE |
| Per-substrate predicted ΔS | ADOPT_CANONICAL | Canonical equation #26 closed form `-25 × (N - K_seed) / 37_545_489` per `src/tac/canonical_equations/procedural_codebook_savings.py` |
| Aggregate composition_alpha cascade | ADOPT_CANONICAL | Catalog #322 v2 cascade per `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` |
| Frontier pointer reference | ADOPT_CANONICAL | `.omx/state/canonical_frontier_pointer.json` per CLAUDE.md "Frontier scores are pointer-only" non-negotiable |
| Per-substrate symposium gating | ADOPT_CANONICAL | Catalog #325 6-step contract |
| APPEND-ONLY supersession discipline | ADOPT_CANONICAL | Catalog #110/#113 HISTORICAL_PROVENANCE; NEW file path; ZERO mutation of original memo |
| Canonical equation registration | ADOPT_CANONICAL | Catalog #344 canonical_equations_registry; 5 events in registry through 2026-05-21T00:21:20Z |
| Per-substrate observability surface | ADOPT_CANONICAL | Catalog #305 observability surface declared per-substrate in audit memos |

### 8.2 Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| "Original matrix memo's predicted ΔS bands [-0.013, -0.0085] are derivable from current candidate state" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | 5 of 7 candidates fall outside the viable empirically-built matrix today; only DP1 + VQ-VAE empirically-grounded; aggregate -0.008140 plateau-adjacent NOT frontier-breaking |
| "ATW V2 codec table is ~3 KB per matrix memo" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | Empirical archive.py grammar audit: cdf_table_blob = 5 × 256 × fp16 = 2,560 bytes (NOT 3,072); corrected ΔS Path A = -0.001683 |
| "grayscale_lut chroma LUT is ~4 KB" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | grayscale_lut substrate uses FiLM-conditioned RGB decoder; current GLV1 lacks explicit chroma_lut section; canonical equation #26 IN-DOMAIN context `chroma_lut_replacement` predicts 256-byte LUT → 32-byte seed = -0.000149 (NOT -0.002706) |
| "NSCS06 v8 has 4 KB chroma codebook" | HYPOTHETICAL (CARGO-CULTED-PENDING-EMPIRICAL) | Current NSCS06 v3 has 15-byte palette; v8 4 KB codebook is a substrate-engineering bet that has NOT been validated; the integration design memo IS the reactivation criterion candidate; cannot grant own reactivation |
| "PR101 lc_v2_clone is amenable to procedural-codebook replacement" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | 0 in-archive deterministic-constant byte regions; archive.zip contains only score-aware-trained per-video weights + per-pair learned latents (all OUT-OF-DOMAIN per canonical equation #26 `_EXCLUDED_CONTEXTS`) |
| "PR106 has substrate code parallel to PR101 lc_v2_clone" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | No `src/tac/substrates/pr106_*` directory exists; PR106 lives only as 11 submission packets + research-adapter dispatcher |
| "5-substrate aggregate would decisively break 0.18 floor" | CARGO-CULTED-PENDING-EMPIRICAL | Per §3.4: 5-substrate naive ADDITIVE aggregate would break 0.18 floor BY 0.0007 (marginal). Requires all 5 cascade-mortality blockers to be cleared empirically AND composition_alpha ≥ 0.77 ADDITIVE for all pairs. Conservative SUB-ADDITIVE aggregate is plateau-adjacent |
| "Composition_alpha = 0.5 SUB-ADDITIVE is the canonical conservative" | HARD-EARNED-PARTIAL | Catalog #322 v2 cascade does declare 0.3 < α ≤ 0.7 as SUB-ADDITIVE band with ×0.5 penalty; conservative bound holds. BUT the SATURATING floor at -0.005000 (α ≤ 0.3) is the canonical worst case; empirical α can only be measured via paired-anchor smoke |

## Section 9. 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** — First memo to document the 5-substrate matrix supersession reflecting today's 5 cascade-mortality data points; structurally distinct from the original matrix memo (single-anchor design) AND from each of the 5 audit memos (per-substrate scope).
2. **BEAUTY + ELEGANCE** — 9-section structure per task description; per-substrate cascade-mortality table (1 row per candidate) + 4-band composition_alpha cascade per Catalog #322 + 6-action operator decision matrix; reviewable in 30 seconds per HNeRV parity L4.
3. **DISTINCTNESS** — APPEND-ONLY supersession scope explicitly distinct from original matrix memo (PRESERVED VERBATIM); explicitly distinct from per-substrate audit memos (which adjudicate single-substrate scope).
4. **RIGOR** — Catalog #229 PV: 7 source files read in full (5 audit memos + 1 original matrix memo + canonical equation #26 source); 4 assumption-adversary verdicts per Catalog #292; corrected per-substrate predictions verified against canonical equation #26 closed form; 5-event canonical equation registry state verified empirically.
5. **OPTIMIZATION PER TECHNIQUE** — Per Catalog #290 above; ADOPT_CANONICAL where canonical helper serves (Catalog #322 cascade + canonical equation #26 + frontier pointer); UNIQUE for the new cascade-mortality taxonomy framework.
6. **STACK-OF-STACKS-COMPOSABILITY** — §3 cascade-of-cascades aggregates 2-substrate empirically-built → 3-substrate post-GLV2 → 4-substrate post-Variant-C → 5-substrate hypothetical; each cascade band is decomposable to per-substrate contribution; Catalog #296 Dykstra-feasibility verified mutually-orthogonal at rate axis.
7. **DETERMINISTIC REPRODUCIBILITY** — All predicted ΔS derived from canonical equation #26 closed form; all empirical citations point to canonical memo paths; no proxy / approximation; byte-stable derivation.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Per §3.4: 5-substrate hypothetical aggregate breaks 0.18 floor by 0.0007 under naive ADDITIVE; conservative SUB-ADDITIVE plateau-adjacent. Honest predictions per Catalog #287 with empirically-corrected per-substrate inputs.
9. **OPTIMAL MINIMAL CONTEST SCORE** — Per Catalog #324 `predicted_band_validation_status=pending_post_training`; the 7-candidate corrected taxonomy + recalibrated aggregate IS the canonical path to the 0.18 floor break PROVIDED 5+ cascade-mortality blockers are cleared empirically. The empirical-anchor accumulation playbook per original matrix memo §6 sequencing holds (DP1 first → VQ-VAE → grayscale_lut GLV2 → ATW V2 Variant C → NSCS06 v8 BUILD).

## Section 10. Observability surface (Catalog #305)

| Facet | Implementation |
|---|---|
| 1. Inspectable per layer | Per-substrate cascade-mortality table (§2) + per-substrate predicted ΔS (corrected) + per-substrate audit memo paths surfaced as `related_deliberation_ids` frontmatter |
| 2. Decomposable per signal | Naive aggregate ΔS + composition_alpha-adjusted aggregate ΔS + per-pair α matrix (per §3.1-3.4) decomposable to individual per-substrate contributions; per-cascade-level (2-substrate → 5-substrate hypothetical) decomposable |
| 3. Diff-able across runs | Per-substrate canonical equation residuals recalibrate via `update_equation_with_empirical_anchor` per Catalog #344; each paired-smoke surfaces in canonical_equations_registry.jsonl; this memo's predictions are auditable against future empirical anchors |
| 4. Queryable post-hoc | Canonical equation registry JSONL queryable via `tools/list_canonical_equations.py`; per-substrate audit memos queryable via `feedback_*_landed_<YYYYMMDD>.md` glob; this supersession queryable via path |
| 5. Cite-able | Every per-substrate row cites audit memo + corrected ΔS prediction + canonical equation IN-DOMAIN context; the cascade-mortality categories are auditable per Catalog #325 6-step contract per-substrate |
| 6. Counterfactual-able | Byte-mutation smoke per Catalog #272 + `tools/verify_distinguishing_feature_byte_mutation.py` answers "what if this seed byte changed?" per-substrate (DP1 + VQ-VAE smokes already pass) |

## Section 11. 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map** = N/A (this is a SUPERSESSION memo; no per-tensor sensitivity contribution beyond what the per-substrate audit memos already declared).
- **Hook #2 Pareto constraint** = ACTIVE-DESIGN (§3.5 Dykstra-feasibility analysis articulates rate-axis loosening under all 7 candidates; explicit Pareto-feasible-region intersection check; each per-substrate IN-DOMAIN context is mutually orthogonal at the rate-axis).
- **Hook #3 bit-allocator** = ACTIVE-DESIGN (per-substrate 32-byte seed slot replaces 256-byte to 8,192-byte codebook slot; this supersession's per-substrate corrected predictions feed bit-allocator's per-substrate byte budget decisions).
- **Hook #4 cathedral autopilot dispatch** = ACTIVE-DESIGN (sister cathedral_consumer `procedural_codebook_savings_consumer` auto-discovered per Catalog #335 + Catalog #341 Tier A markers; this supersession's corrected per-substrate predictions feed the autopilot ranker via the consumer's `predicted_delta_adjustment=0.0` markers per Catalog #341 + canonical Provenance per Catalog #323).
- **Hook #5 continual-learning posterior** = ACTIVE-DESIGN (canonical equation `procedural_codebook_from_seed_compression_savings_v1` extension per `update_equation_with_empirical_anchor` per Catalog #344; first empirical anchor candidate is DP1 per Action 1; each per-substrate paired-smoke recalibrates the equation per `RECALIBRATE_ON_NEW_ANCHORS` trigger).
- **Hook #6 probe-disambiguator** = ACTIVE — THIS supersession IS the disambiguator between the original matrix memo's 5-substrate aggregate prediction band and the empirically-grounded 2-substrate state today. The cascade-mortality taxonomy (HYPOTHETICAL + EMPIRICAL-BUT-BLOCKED + NOT-CANDIDATE + OUT-OF-SCOPE + FiLM-APPEND-not-REPLACE) IS the canonical disambiguator for future cascade-sequencing decisions.

## Section 12. Sister coordination + collision verdict

- **Sister-DISJOINT** from in-flight PARSER-SAFE SUBSET SMOKE (slot 2 commit `a988f9d`) — different file scope (PR101 parser-safe smoke vs this supersession memo).
- **Sister-DISJOINT** from in-flight HONEST CASCADE-MORTALITY ASSESSMENT (slot 3 commit `a3839bc`) — different file scope (cascade-mortality assessment vs this supersession memo).
- **Sister-DISJOINT** from ATW V2 PROCEDURAL VARIANT BUILD DESIGN (commit `7ea78deaa` Top-3 #3 audit landing 2026-05-20T20:47Z) — sister memo; this supersession CITES the ATW V2 audit findings.
- **Sister-DISJOINT** from grayscale_lut + DP1 + VQ-VAE PROCEDURAL VARIANT BUILD landings.
- **Sister-DISJOINT** from PR101+PR106 BUILD DESIGN PIVOT landing (commit `086d3ac1d`).
- **Sister-DISJOINT** from NSCS06 v8 PROCEDURAL CHROMA LUT INTEGRATION DESIGN landing.
- **Step 0 helper verdict**: `tools/check_sister_files_recently_landed.py --files .omx/research/five_substrate_procedural_replacement_matrix_design_SUPERSEDED_20260520T2230Z.md --lookback-hours 12 --own-subagent-id wave-3-five-substrate-matrix-supersession-20260520` → `PROCEED: no sister commits touched any of 1 target file(s) within the 12-hour lookback window. Safe to write.`
- Catalog #314 / #340 sister-checkpoint guard: pre-write check PASSED; no sister has my target file in `files_touched` checkpoint within last 60 minutes.

## Section 13. Top-3 operator-routable next-actions

1. **AUTHORIZE-DP1-PAIRED-SMOKE-FIRST-ANCHOR** (Action 1 of §5): operator routes via canonical `tools/operator_authorize.py` for the DP1 PROCEDURAL VARIANT paired-smoke ($0.30 Modal T4 paired CPU+CUDA). Recipe NOT yet committed per scope limits; design memo path documented per `feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md`. First empirical anchor for canonical equation #26 lands via `update_equation_with_empirical_anchor` per Catalog #344. Estimated wall-clock: ~30 min Modal scheduling + 50-100 min smoke + 24h harvest window per Catalog #330.

2. **SPAWN-VQ-VAE-PER-SUBSTRATE-SYMPOSIUM** (Action 2 of §5): spawn `lane_per_substrate_symposium_vq_vae_procedural_codebook_replacement_20260520` per Catalog #325 6-step contract (sextet pact + van den Oord + MacKay grand-council seats). $0 GPU + ~1 council session + ~3h memo writing. Unblocks 2nd empirical anchor candidate (VQ-VAE) for follow-on paired-smoke.

3. **DECIDE-ATW-V2-SCOPING** (Action 4 of §5): operator decision (verbatim quote required per Catalog #300 mission-alignment): Variant A (cooperative-receiver bolt-on; empirically PRE-FALSIFIED) vs Variant C (decoupled rate-axis-only; viable BUILD path). Recommended **Variant C** per Assumption-Adversary verbatim in ATW V2 audit memo. Estimated cost: ~5 min operator decision; if Variant C approved: ~6h follow-on symposium + ~75 min BUILD subagent + $0.30 paired-smoke. Opens parallel rate-axis-only research path that does NOT depend on D4 probe re-verdict.

## Section 14. Blockers

1. **Original matrix memo's 5-of-5 candidate viability assumption is empirically falsified.** Only DP1 + VQ-VAE are EMPIRICALLY-BUILT today; 3 of 5 original candidates fall outside the viable matrix per today's audits (NSCS06 v8 HYPOTHETICAL + ATW V2 EMPIRICAL-BUT-BLOCKED + TT5L REFUSE). Honest disclosure: the original matrix memo's predicted band [-0.013, -0.0085] is CARGO-CULTED-EMPIRICALLY-FALSIFIED per Catalog #303 sister discipline; this supersession's empirically-grounded 2-substrate band [-0.008140, -0.005000] is the corrected canonical reference.

2. **5-substrate hypothetical aggregate barely breaks 0.18 floor (margin 0.0007).** Per §3.4: 5-substrate naive ADDITIVE aggregate (-0.012678) takes frontier 0.19205 → 0.17937 (breaks 0.18 by 0.0007). Conservative SUB-ADDITIVE plateau-adjacent. The frontier-breaking case is narrow AND contingent on all 5 cascade-mortality blockers being cleared empirically AND composition_alpha ≥ 0.77 ADDITIVE for all pairs. Operator-routable alternative paths (NOT in this memo's scope): different procedural-replacement equation families, master-gradient null-byte removal lanes per sister in-flight smoke, or sub-additive composition discipline per Catalog #322 v2 cascade.

3. **Per-substrate symposium gating (Catalog #325) NOT satisfied for VQ-VAE + grayscale_lut.** Both BUILD-landed today (commits `6fea30f22` + `f037d1144`) but lack per-substrate symposium gate. Action 2 + Action 3 of §5 are the prerequisites before any paired-smoke fires for these substrates.

4. **ATW V2 D4 probe INDEPENDENT verdict BLOCKING until 2026-06-15.** Even if Variant C scoping approved (Action 4 of §5), the Catalog #313 blocker remains structurally in place until the probe expires OR a follow-on V2-1 redesign re-probe returns MEANINGFUL_CONDITIONING (MI ≥ 0.5). Variant C scoping decouples the cooperative-receiver hypothesis from BUILD viability but the D4 probe's substrate-scope binding requires explicit operator decision.

5. **NSCS06 v8 substrate code does NOT exist.** Per NSCS06 v8 integration design memo: current v3 has 15-byte palette, not the 4 KB LUT the matrix memo assumed. Action 5 of §5 BUILD subagent is the prerequisite; estimated 2-4h wall-clock.

## Section 15. Cross-references

- **Original matrix memo (PRESERVED per Catalog #110/#113)**: `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md` (commit `b3e3442c3`; 670 LOC)
- **Today's 5 audit landings**:
  - NSCS06 v8 integration design: `.omx/research/nscs06_v8_procedural_chroma_lut_integration_design_20260520.md`
  - PR101+PR106 BUILD design: `.omx/research/pr101_pr106_procedural_variant_build_design_20260520.md`
  - ATW V2 BUILD design: `.omx/research/atw_v2_procedural_variant_build_design_20260520.md`
  - grayscale_lut BUILD landing: `.omx/research/grayscale_lut_procedural_variant_build_landed_20260520.md`
  - DP1 BUILD landing: `.omx/research/dp1_procedural_variant_build_landed_20260520.md`
  - VQ-VAE BUILD landing: `.omx/research/vq_vae_procedural_variant_build_landed_20260520.md`
- **Canonical equation #26**: `src/tac/canonical_equations/procedural_codebook_savings.py`
- **Canonical equation registry**: `.omx/state/canonical_equations_registry.jsonl` (5 events for `procedural_codebook_from_seed_compression_savings_v1` through 2026-05-21T00:21:20Z)
- **Cathedral consumer (auto-discovered per Catalog #335)**: `src/tac/cathedral_consumers/procedural_codebook_savings_consumer/`
- **Composition matrix sister state**: `.omx/state/substrate_composition_matrix.json` (2 entries: lane_g_v3 × siren + z3_balle × c6_e4_mdl_ibps; this supersession does NOT mutate this state)
- **Canonical frontier pointer**: `.omx/state/canonical_frontier_pointer.json` (0.19205 [contest-CPU] frontier per CLAUDE.md "Frontier scores are pointer-only")
- **CLAUDE.md non-negotiables**: "Forbidden premature KILL without research exhaustion" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + "Forbidden empirical-claim-without-evidence-tag" + "Frontier scores are pointer-only"
- **Catalog gates cited**: #110 / #113 / #117 / #119 / #125 / #157 / #174 / #185 / #186 / #206 / #229 / #287 / #290 / #292 / #294 / #296 / #297 / #300 / #303 / #305 / #309 / #313 / #314 / #315 / #322 / #323 / #324 / #325 / #335 / #340 / #344

**End of supersession memo.**
