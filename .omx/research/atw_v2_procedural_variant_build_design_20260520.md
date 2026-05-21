<!-- SPDX-License-Identifier: MIT -->
---
substrate_id: atw_codec_v2_procedural_variant
substrate_aliases: ["atw_v2", "atw_codec_v2"]
substrate_class: procedural_variant_design_for_cooperative_receiver_codec
horizon_class: frontier_pursuit
council_tier: T1
council_attendees:
  - Shannon
  - Yousfi
  - Atick
  - Tishby_memorial
  - Wyner_memorial
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "The Catalog #313 D4 INDEPENDENT verdict (MI=0.006385 << 0.5; expires 2026-06-15) STILL BLOCKS dispatch. A procedural-variant BUILD that ships the CDF table as a 32-byte seed inherits the SAME side-info-channel collapse the D4 probe falsified at MI=0.006385. The 5-substrate matrix memo's claim that ATW V2 is BUILD-ready is CARGO-CULTED. Defer BUILD until either (a) the V2-1 redesign + re-probe per the 2026-05-18 symposium PROCEED_WITH_REVISIONS lands AND clears MI >= 0.5, OR (b) the procedural variant is scoped as a Variant C decoupled from cooperative-receiver hypothesis (per V2 §19 G2-PARTIAL alternative)."
  - member: Assumption-Adversary
    verbatim: "Per Catalog #292 per-deliberation assumption surfacing. The SHARED ASSUMPTION operating across the 5-substrate matrix memo + this design: 'A procedural seed replacing the CDF table is structurally orthogonal to the side-info-channel content question because the seed REPLACES the table bytes deterministically; the seed-derived table is a DIFFERENT byte string than the trained table but information-theoretically equivalent under the canonical equation #26 IN-DOMAIN context atw_v2_codec_quantizer_lut.' I classify this CARGO-CULTED-PENDING-EMPIRICAL. HARD-EARNED basis: canonical equation #26 EXPLICITLY registers atw_v2_codec_quantizer_lut as IN-DOMAIN at line 103 of src/tac/canonical_equations/procedural_codebook_savings.py; the procedural-codebook generator (PCG64) IS deterministic per seed; the byte savings ARE structurally captured. CARGO-CULTED basis: the procedural seed-derived CDF table has ZERO empirical relationship to the trained CDF table's per-class structure. The trained CDF table encodes I(latent_symbol; segnet_class) AS LEARNED from data; a procedural seed produces a uniform-PRNG byte string that DESTROYS that learned conditioning. Per Tishby IB framework: H(T_procedural) = log2(uniform symbol distribution) is irrelevant to I(X;T) for the cooperative-receiver hypothesis. The procedural variant operationally REPLACES the side-info-channel content NOT just its bytes. UNLESS we explicitly scope this as a Variant C (decoupled-from-cooperative-receiver), the procedural variant is OPERATIONALLY ORTHOGONAL TO THE PARADIGM."

council_assumption_adversary_verdict:
  - assumption: "Canonical equation #26 IN-DOMAIN context atw_v2_codec_quantizer_lut applies to ATW V2's cdf_table_blob"
    classification: HARD-EARNED
    rationale: "Line 103 explicit registration; the cdf_table IS a quantizer LUT (5 class × 256 symbol → fp16 probability table consumed at inflate-time by the latent reconstruction loop)."
  - assumption: "The ~3 KB codec table claim in the 5-substrate matrix memo is empirically verified per archive.py:111-115"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "archive.py:111-115 contains DATACLASS FIELD DECLARATIONS (scorer_class_prior_table: torch.Tensor / cdf_table: torch.Tensor field signatures), NOT byte-count assertions. The actual cdf_table_blob byte count from archive.py:40 + line 426 is cdf_classes * cdf_symbols * 2 = 5 * 256 * 2 = 2,560 bytes (NOT 3,072 or the matrix memo's '5 × 41 × fp32 = 820' claim). The matrix memo's predicted ΔS of -0.002024 is therefore based on an INCORRECT byte count. Correct Path A (cdf_table only) ΔS = -25 * (2560 - 32) / 37_545_489 = -0.001683. Correct Path B (cdf + class_prior at prior_dim=2 default = 2400 bytes) ΔS = -0.003281."
  - assumption: "The procedural variant is structurally additive over the cooperative-receiver hypothesis"
    classification: CARGO-CULTED-OPERATIONALLY-FALSIFIED
    rationale: "Per Assumption-Adversary council verbatim: the procedural seed REPLACES the learned CDF table with a uniform-PRNG-derived table, OPERATIONALLY REPLACING the side-info-channel content. The cooperative-receiver hypothesis depends on I(latent; segnet_class) being NON-ZERO via the LEARNED CDF table; a procedural CDF table has uniform conditional distribution. The procedural variant is therefore a VARIANT-C ALTERNATIVE (decoupled from cooperative-receiver) per V2 §19, NOT an ADDITIVE bolt-on on the V2 cooperative-receiver substrate. This is a CRITICAL scoping decision that the BUILD design memo must document EXPLICITLY."
  - assumption: "Catalog #313 D4 INDEPENDENT verdict expires 2026-06-15 and is therefore time-bounded blocking"
    classification: HARD-EARNED
    rationale: "Per probe ledger row probe_id=atw_v2_d4_h_latent_given_scorer_class_20260516 (adjudicated 2026-05-16T22:47:41Z; expires 2026-06-15T22:47:41Z); blocker_status=blocking. The procedural variant DOES NOT replace the D4-tested side-info channel (per-pair SegNet composite class against A1 latents); a procedural CDF table is a DIFFERENT operational substrate. Per Catalog #313 sister discipline + the symposium's Revision #7 'EVIDENCE-BASED not TIME-BASED' reactivation: the D4 blocking verdict does NOT structurally extend to a procedural variant that operates on a DIFFERENT base substrate (eliminating A1 latents + replacing CDF table with seed); the procedural variant is its OWN canonical equation #26 substrate-class-shift requiring its own per-substrate symposium per Catalog #325."
  - assumption: "BUILD is the optimal next-action vs DEFER-PENDING-CLARIFICATION"
    classification: CARGO-CULTED-MILD
    rationale: "Per CLAUDE.md 'Substrate MUST be at OPTIMAL FORM before paid empirical dispatch' + Catalog #315: BUILD is appropriate when the substrate's design memo has council PROCEED-unconditional. The current state: ATW V2 cooperative-receiver SUBSTRATE has 2026-05-18 PROCEED_WITH_REVISIONS (NOT unconditional); the procedural variant SCAFFOLD has no prior council; this is the FIRST council deliberation on the procedural variant. Per Catalog #325 14-day window: the 2026-05-18 symposium memo COVERS atw_codec_v2 substrate but does NOT cover the procedural-variant operational reframing. The honest verdict is DEFER-PENDING-CLARIFICATION until (a) operator decides whether procedural variant is Variant C (decoupled from cooperative-receiver) or merged into V2-1 redesign, AND (b) a follow-on per-substrate symposium ratifies the variant scope."

predicted_band_validation_status: pending_post_training
predicted_band:
  variant_a_cdf_only:
    rate_axis_delta_s: [-0.001683, -0.001683]  # exact per canonical equation #26
    seg_axis_delta_s_under_cooperative_receiver_hypothesis: [+0.005, +0.030]  # uniform-PRNG CDF destroys per-class conditioning; expected score WORSE seg
    pose_axis_delta_s_under_cooperative_receiver_hypothesis: [+0.002, +0.010]  # similar (cooperative-receiver loss affects PoseNet too)
    net_score_delta_s_under_cooperative_receiver_hypothesis: [+0.006, +0.038]  # NET WORSE (rate gain << seg+pose loss)
    net_score_delta_s_under_variant_c_decoupled_hypothesis: [-0.001683, -0.001683]  # rate-axis only; assumes operator scopes as Variant C
  variant_b_cdf_plus_class_prior:
    rate_axis_delta_s: [-0.003281, -0.003281]
    seg_axis_delta_s_under_cooperative_receiver_hypothesis: [+0.008, +0.040]
    net_score_delta_s_under_cooperative_receiver_hypothesis: [+0.005, +0.036]
    net_score_delta_s_under_variant_c_decoupled_hypothesis: [-0.003281, -0.003281]
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
operator_directive: "WAVE-3-ATW-V2-PROCEDURAL-VARIANT-BUILD-DESIGN; per operator-pick highest-EV; design-only (NO BUILD)"
related_deliberation_ids:
  - council_per_substrate_symposium_atw_v2_reactivation_20260518
  - atw_codec_v2_d4_probe_verdict_20260516_codex
  - five_substrate_procedural_replacement_matrix_design_20260520
  - feedback_grayscale_lut_procedural_variant_build_landed_20260520
  - feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520
  - feedback_slot_mg7_bundle_master_gradient_exploits_landed_20260520
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl
parent_design_memo: .omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md
parent_landing_memos:
  - feedback_grayscale_lut_procedural_variant_build_landed_20260520.md
  - feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md
audit_corrections:
  - "Matrix memo claim '~3 KB codec table per archive.py:111-115' refers to DATACLASS FIELD DECLARATIONS, NOT byte counts. Empirical cdf_table_blob = 5 × 256 × fp16 = 2,560 bytes (not 3,072)."
  - "Matrix memo claim '5 × 41 × fp32 = 820 bytes' is WRONG. Empirical grammar: 5 classes × 256 symbols × 2 bytes fp16 = 2,560 bytes."
  - "Corrected predicted ΔS Path A (cdf only) = -0.001683 (NOT matrix memo's -0.002024)."
---

<!-- Catalog #344 canonical-equations-registry cross-reference: this design
memo's predicted ΔS bands ARE derived via the canonical equation
`procedural_codebook_from_seed_compression_savings_v1` registered at
`src/tac/canonical_equations/procedural_codebook_savings.py` and persisted
to `.omx/state/canonical_equations_registry.jsonl`. The IN-DOMAIN context
`atw_v2_codec_quantizer_lut` IS explicitly registered at line 103 of the
canonical equation source. Per-substrate prediction follows
ΔS = -25 * (N_codebook - K_seed) / 37_545_489 with K_seed=32. -->

# WAVE-3-ATW-V2-PROCEDURAL-VARIANT-BUILD-DESIGN (2026-05-20)

**Lane**: `lane_atw_v2_procedural_variant_build_design_20260520`
**Subagent**: `wave-3-atw-v2-procedural-variant-build-design-20260520`
**Parent**: 5-SUBSTRATE PROCEDURAL REPLACEMENT MATRIX (commit `b3e3442c3`) Candidate 2
**Council verdict**: **DEFER_PENDING_EVIDENCE** (NOT PROCEED to BUILD)
**Mission contribution**: `frontier_protecting` (prevents another premature BUILD on a blocked substrate)
**Today's status**: 5-of-5 candidates audit complete; ATW V2 is **4th HONEST-DISCLOSURE** finding (sister of NSCS06 v8, PR101/PR106, grayscale_lut)

## Section 1. Headline

**ATW V2 procedural-variant BUILD is DEFERRED per 3 independent failure modes:**

1. **MATRIX MEMO BYTE-COUNT CLAIM EMPIRICALLY FALSIFIED**: The matrix memo (commit `b3e3442c3`) cites "verified per `src/tac/substrates/atw_codec_v2/archive.py:111-115`" but those lines are dataclass field declarations (`scorer_class_prior_table: torch.Tensor` / `cdf_table: torch.Tensor`), NOT byte-count assertions. Empirical grammar audit (archive.py:40 + line 426): `cdf_table_blob = cdf_classes × cdf_symbols × 2 = 5 × 256 × 2 = 2,560 bytes` (NOT 3,072 as claimed). Corrected predicted ΔS = -0.001683 (not -0.002024).

2. **CATALOG #313 D4 PROBE STILL BLOCKING** (expires 2026-06-15; 26 days from now): The probe-outcomes ledger shows `verdict: INDEPENDENT`, `blocker_status: blocking`, MI=0.006385 bits/symbol vs threshold 0.5 (2 orders of magnitude below). The per-substrate symposium 2026-05-18 (commit `council_per_substrate_symposium_atw_v2_reactivation_20260518.md`) returned `PROCEED_WITH_REVISIONS` with Revision #1-7 (8 binding revisions) requiring V2-1 redesign + new D4 probe BEFORE any paid Modal dispatch.

3. **PROCEDURAL VARIANT OPERATIONALLY REPLACES SIDE-INFO CHANNEL CONTENT**: A procedural seed-derived CDF table destroys the learned `I(latent_symbol; segnet_class)` conditioning that the cooperative-receiver hypothesis depends on. The procedural variant is NOT a bolt-on on V2 cooperative-receiver substrate; it is a Variant C ALTERNATIVE (decoupled from cooperative-receiver hypothesis) per V2 §19. This is a scoping decision requiring explicit operator direction.

**Honest disclosure**: Today's 5-substrate matrix landed 4 HYPOTHETICAL candidates (NSCS06 v8, PR101, PR106, grayscale_lut) + 1 EMPIRICAL (VQ-VAE, DP1). ATW V2 falls into a SIXTH category: **EMPIRICAL substrate code exists BUT a blocking probe outcome prevents BUILD**. The substrate IS implemented (ATW2 archive grammar; cdf_table_blob shipped per V2 design memo §10); the BLOCKER is the cooperative-receiver hypothesis empirical falsification per Catalog #313.

## Section 2. ATW V2 substrate code audit verdict

### 2.1 Empirical existence check (Catalog #229 PV)

- `ls src/tac/substrates/atw_codec_v2/`: **EXISTS** (6 files: `__init__.py`, `architecture.py`, `archive.py`, `inflate.py`, `registered_substrate.py`, `score_aware_loss.py` + `tests/` subdir)
- `archive.py` size: 23.9 KB (623 LOC); ATW2 grammar with 8 sections per design memo §10
- Sister `atw_codec_v1` (atw_codec_v1; ATW1 grammar; predecessor) ALSO EXISTS at 20.4 KB

### 2.2 Codec table byte counts (empirical; corrects matrix memo)

Per `src/tac/substrates/atw_codec_v2/archive.py` actual grammar (lines 22-42 + 426-431):

| Section | Grammar | Empirical bytes (defaults) | Matrix memo claim | Audit verdict |
|---|---|---|---|---|
| `cdf_table_blob` | `cdf_classes × cdf_symbols × 2 (fp16)` | `5 × 256 × 2 = 2,560 bytes` | "~3 KB" / "5 × 41 × fp32 = 820 bytes" | **FALSIFIED-MATRIX-CLAIM** |
| `scorer_class_prior_table_blob` | `num_pairs × prior_dim × 2 (fp16)` | `600 × prior_dim × 2` (prior_dim trainer-config dependent; common values 2-16) | "~1 KB minimum" | **PARTIALLY-CORRECT-DEPENDS-ON-CONFIG** |
| `distill_head_blob` | `brotli(fp16 g1_distill_head)` | trainer-dependent; ~5-50 KB typical | not in matrix memo | **NEW-DISTINGUISHING-FEATURE** (G1 5-way scorer-class distill head) |

**Conclusion**: The matrix memo's "~3 KB CDF + class-prior tables → 32-byte seed" claim is APPROXIMATELY CORRECT in aggregate (Path B = 2560 + 2400 = 4960 bytes ≈ 5 KB) but the specific byte-count attribution at archive.py:111-115 is FALSIFIED (those lines are dataclass field declarations).

### 2.3 Substrate architecture (per V2 design memo §10)

ATW V2 ships an 8-section archive (`ATW2_HEADER_FMT = "<4sBBHHHHHIIIIIIII"`; 48-byte header):

1. `atw2_header` — 48 bytes; control_or_metadata
2. `encoder_blob` — brotli(fp16 encoder); training_provenance_only
3. `decoder_blob` — brotli(fp16 decoder); decoder_weight_stream
4. `wz_head_blob` — brotli(fp16 wz_side_info_head); decoder_weight_stream
5. `distill_head_blob` — brotli(fp16 g1_distill_head); decoder_side_information
6. `latent_residual_blob` — int8 z_residual; latent_stream
7. `class_prior_table_blob` — fp16 scorer_class_prior_table; decoder_side_information
8. `cdf_table_blob` — fp16 cdf_table (B3 scorer-conditional CDF); decoder_side_information
9. `meta_blob` — sorted-keys JSON with `atw_v2_codec_meta` provenance tag

The procedural-variant TARGET sections are #7 (`class_prior_table_blob`) + #8 (`cdf_table_blob`) per the matrix memo. These ARE the "decoder_side_information" sections that operationalize the cooperative-receiver theorem per V2 design memo §3.

### 2.4 EMPIRICAL vs HYPOTHETICAL classification

**Verdict**: **EMPIRICAL substrate code EXISTS** (~$0 to verify), with PARTIAL matrix memo accuracy:
- Substrate code: EMPIRICAL (623 LOC archive.py; 6 files; full ATW2 grammar)
- Byte count claim: PARTIALLY-EMPIRICAL (aggregate ~5 KB correct; specific archive.py:111-115 attribution FALSIFIED)
- Predicted ΔS: REQUIRES CORRECTION (-0.001683 Path A vs -0.002024 claimed)

## Section 3. Catalog #313 D4 re-probe verdict

### 3.1 Probe ledger row (verbatim from `.omx/state/probe_outcomes.jsonl`)

```json
{
  "probe_id": "atw_v2_d4_h_latent_given_scorer_class_20260516",
  "verdict": "INDEPENDENT",
  "blocker_status": "blocking",
  "adjudicated_at_utc": "2026-05-16T22:47:41Z",
  "expires_at_utc": "2026-06-15T22:47:41Z",
  "metric_name": "mutual_information_bits_per_symbol",
  "metric_value": 0.006385502752,
  "threshold": 0.5,
  "threshold_token": "MEANINGFUL_CONDITIONING",
  "next_action": "do_not_dispatch_atw_v2_phase2_from_this_signal",
  "evidence_path": ".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md",
  "recipe_path": ".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml"
}
```

### 3.2 Verdict: **STILL BLOCKING** (does not expire until 2026-06-15; 26 days from now)

Per CLAUDE.md "Catalog #313 dispatch target has no predecessor adjudicated outcome" non-negotiable + Catalog #298 30-day staleness window: the D4 INDEPENDENT verdict explicitly blocks ATW V2 paid Modal dispatch until either:
- The probe expires AND no new probe is run (NOT EVIDENCE-BASED; the symposium Revision #7 explicitly requires evidence-based not time-based reactivation), OR
- The V2-1 redesign per the 2026-05-18 symposium PROCEED_WITH_REVISIONS lands AND a NEW D4 probe on the new side-info channel returns MEANINGFUL_CONDITIONING (MI ≥ 0.5), OR
- The procedural variant scopes as Variant C (decoupled from cooperative-receiver hypothesis), in which case the D4 probe RESULT is structurally irrelevant (procedural CDF table operates on different side-info content than the per-pair SegNet composite the D4 probe tested).

### 3.3 D4 probe reactivation criteria (per ledger)

> "replace SegNet composite class with richer side-information OR rerun on trained ATW v2 residuals"

The procedural-variant BUILD does NEITHER of these:
- It does NOT replace the side-info channel (procedural seed produces uniform-PRNG bytes, not richer scorer signal)
- It does NOT re-run on trained ATW v2 residuals (the BUILD operates at design-time, not on a trained checkpoint)

**Therefore the procedural variant DOES NOT satisfy D4 probe reactivation criteria.**

## Section 4. Per-substrate symposium recency per Catalog #325

### 4.1 Symposium memo present

`.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md`
- Date: 2026-05-18 (2 days old; within 14-day Catalog #325 window)
- Verdict: PROCEED_WITH_REVISIONS (NOT PROCEED-unconditional)
- 8 binding revisions documented (Revision #1-7 + frontier citation)
- T2 sextet + 4 specialists (Atick, Redlich, Tishby memorial, Wyner memorial)
- substrate_aliases mechanism per Catalog #315 supports the `atw_codec_v2` ↔ `atw_v2` matching

### 4.2 Catalog #325 satisfaction status

**SATISFIED for substrate `atw_codec_v2`** (14-day window expires 2026-06-01)

However, per Revision #1 (binding per Contrarian + Atick): "the V2-1 redesign MUST be a council-grade design memo enumerating ≥3 alternative side-info channel hypotheses". This design memo (THIS file) is a **DIFFERENT scope** — it audits whether a PROCEDURAL VARIANT BUILD is viable, NOT whether V2-1 redesign is viable.

**Conclusion**: The 2026-05-18 symposium covers V2 base substrate reactivation; it does NOT pre-authorize a procedural-variant BUILD as a separate substrate path. A follow-on per-substrate symposium per Catalog #325 IS required before any procedural-variant BUILD if operator scopes it as a substrate-class shift.

## Section 5. PROCEDURAL VARIANT module design spec (DEFERRED rationale)

### 5.1 Why NOT BUILD now

Per the 3 failure modes in §1:

1. **Matrix memo byte-count claim EMPIRICALLY FALSIFIED** — the predicted ΔS is -0.001683 (Path A) NOT -0.002024. The 5-substrate aggregate predicted ΔS needs recomputation:
   - Matrix memo aggregate (NAIVE SUM): -0.016939 (using -0.002024 for ATW V2)
   - Corrected aggregate (Path A): -0.016599 (using -0.001683 for ATW V2)
   - Corrected aggregate (Path B): -0.018197 (using -0.003281 for ATW V2 IF both cdf + class_prior replaced)

2. **Catalog #313 D4 blocking** — even if BUILD lands at $0, the substrate cannot dispatch to paid Modal until either V2-1 redesign + new probe OR explicit Variant C scoping.

3. **Cooperative-receiver hypothesis operationally falsified by procedural seed** — per Assumption-Adversary verbatim: a procedural CDF table destroys the learned `I(latent_symbol; segnet_class)` conditioning. The score impact under cooperative-receiver hypothesis is PREDICTED-WORSE (rate-axis gain ≈ -0.0017 << seg-axis loss ≈ +0.005 to +0.030). NET PREDICTED SCORE WORSE by +0.006 to +0.038 if scoped under cooperative-receiver hypothesis.

### 5.2 If operator scopes as Variant C (decoupled from cooperative-receiver hypothesis)

This becomes a viable BUILD design path. Sister of DP1/VQ-VAE/grayscale_lut canonical pattern:

#### 5.2.1 New module spec

`src/tac/substrates/atw_codec_v2/procedural_variant.py` (~600 LOC; sister API surface):
- `ProceduralVariantConfig` dataclass (canonical fields: `seed`, `generator_kind`, `canonical_equation_id`, `in_domain_context="atw_v2_codec_quantizer_lut"`)
- `PROCEDURAL_SEED_SIZE_BYTES = 32` (canonical equation #26 K_seed)
- `PROCEDURAL_CDF_BYTES_DEFAULT = 2560` (5 classes × 256 symbols × fp16; empirical)
- `derive_procedural_cdf_table_from_seed(seed, num_classes=5, num_symbols=256, generator_kind="pcg64")` → torch.Tensor `(5, 256)` fp32
- `compose_with_procedural_cdf_table(canonical_atw2_archive_bytes, seed, ...)` → bytes (REPLACE-IN-PLACE the cdf_table_blob section; preserves all other sections byte-for-byte)
- `verify_procedural_cdf_in_domain` (sister of grayscale_lut `verify_procedural_lut_in_domain`)
- `verify_seed_mutation_changes_cdf_bytes` (Catalog #272 byte-mutation invariant)
- `predicted_archive_bytes_saved(N_cdf=2560, K_seed=32) → 2528`
- `predicted_delta_s → -0.001683`

#### 5.2.2 Sentinel

`PROCEDURAL_CDF_SENTINEL = b"ACPV"` (ATW2 CDF Procedural Variant; sister of grayscale_lut `b"GLPV"` + VQ-VAE `b"VQVP"`)

Distinct from DP1 (no sentinel; in-place codebook section replacement) because ATW V2 has a dedicated `cdf_table_blob` section that CAN be replaced in-place at the canonical grammar layer.

## Canonical-vs-unique decision per layer
<!-- Catalog #290 mandatory section; scope = Variant C BUILD scaffold below per Section 5.2 -->

(See §5.2.3 for the canonical-vs-unique decision table per layer; reproduced here as the canonical Catalog #290 section header.)

#### 5.2.3 Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| `ProceduralVariantConfig` dataclass | ADOPT_CANONICAL_BECAUSE_SERVES | Sister DP1/VQ-VAE/grayscale_lut pattern |
| `PROCEDURAL_SEED_SIZE_BYTES=32` | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical equation #26 K_seed |
| `PROCEDURAL_CDF_BYTES_DEFAULT=2560` | FORK_BECAUSE_PRINCIPLED_MISMATCH | ATW V2 grammar fixes cdf_classes=5 × cdf_symbols=256 × fp16 |
| `CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT="atw_v2_codec_quantizer_lut"` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Equation #26 line 103 explicit registration |
| `PROCEDURAL_CDF_SENTINEL=b"ACPV"` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Substrate-unique sentinel |
| Archive composition (REPLACE-IN-PLACE) | FORK_BECAUSE_PRINCIPLED_MISMATCH | ATW V2 grammar HAS a cdf_table_blob section (unlike grayscale_lut which APPENDS envelope) |
| `derive_codebook_from_seed` canonical helper | ADOPT_CANONICAL_BECAUSE_SERVES | Single source of truth for PRNG-based codebook derivation |
| `validate_context_is_in_domain` sister helper | ADOPT_CANONICAL_BECAUSE_SERVES | Equation #26 explicit IN-DOMAIN registry |

## 9-dimension success checklist evidence
<!-- Catalog #294 mandatory section; scope = Variant C BUILD scaffold below per Section 5.2 -->

(See §5.2.4 for the 9-dimension success checklist evidence; reproduced here as the canonical Catalog #294 section header.)

#### 5.2.4 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** — `atw_v2_codec_quantizer_lut` IN-DOMAIN context is structurally distinct from DP1's `dp1_codebook_bytes` + VQ-VAE's `intermediate_transform_quantizer` + grayscale_lut's `chroma_lut_replacement`; all 4 are canonical equation #26 IN-DOMAIN contexts but operate on different substrate-class sub-surfaces.
2. **BEAUTY + ELEGANCE** — ~600 LOC module reviewable in 30s; sister-identical structure to DP1/VQ-VAE/grayscale_lut canonical pattern.
3. **DISTINCTNESS** — Distinguishing feature: REPLACE-IN-PLACE of `cdf_table_blob` section at the canonical ATW2 grammar layer (vs grayscale_lut APPEND envelope; vs DP1 in-place section replacement; vs VQ-VAE prepend within decoder_blob).
4. **RIGOR** — This design memo's premise verification per Catalog #229 + Catalog #292 assumption-adversary 5 verdicts.
5. **OPTIMIZATION PER TECHNIQUE** — Per §5.2.3 above.
6. **STACK-OF-STACKS-COMPOSABILITY** — Procedural variant replaces 2,560 → 32 bytes at the cdf_table_blob section; preserves all 7 other sections byte-for-byte; composes with DP1 + VQ-VAE + grayscale_lut sister procedural variants additively (rate-axis only) under Variant C scoping.
7. **DETERMINISTIC REPRODUCIBILITY** — `derive_procedural_cdf_table_from_seed` is deterministic per seed + generator_kind + shape + dtype; envelope encoding byte-stable; brotli quality 9 fixed (sister discipline).
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Predicted ΔS = -0.001683 (smaller than DP1 -0.002706 and VQ-VAE -0.005433; larger than grayscale_lut -0.000149). Strongest IN-DOMAIN confidence (canonical equation #26 line 103 explicit registration).
9. **OPTIMAL MINIMAL CONTEST SCORE** — Per Catalog #324, `predicted_band_validation_status=pending_post_training`; the predicted ΔS is HYPOTHESIS not score claim. Per-substrate symposium per Catalog #325 + paired-smoke contest-CUDA + contest-CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" are required gates.

## Cargo-cult audit per assumption
<!-- Catalog #303 mandatory section; scope = Variant C BUILD scaffold below per Section 5.2 -->

(See §5.2.5 for the cargo-cult audit per assumption; reproduced here as the canonical Catalog #303 section header.)

#### 5.2.5 Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| Canonical equation #26 closed-form predicts savings exactly | HARD-EARNED | Closed-form is exact bytes_saved formula; rate term IS `25*bytes/denom` per `upstream/evaluate.py:63` |
| `atw_v2_codec_quantizer_lut` IS strongest IN-DOMAIN fit | HARD-EARNED | Canonical equation #26 line 103 explicit registration |
| Current ATW V2 substrate has a 2,560-byte cdf_table_blob | HARD-EARNED | Empirical archive.py grammar audit (cdf_classes=5 × cdf_symbols=256 × fp16 = 2,560 bytes; verified by direct code reading) |
| Procedural CDF table is information-theoretically equivalent to learned CDF table | CARGO-CULTED — UNWIND-REQUIRED | Per Assumption-Adversary verbatim: procedural seed produces uniform-PRNG bytes; learned CDF table encodes I(latent; class). Variant C scoping decouples this assumption from BUILD viability. |
| Per-pixel-independence of CDF table bytes | CARGO-CULTED — INHERITED | Inherited from canonical equation #26 derivation; the IN-DOMAIN context's domain-of-validity covers it. Per-substrate symposium evaluates whether the inherited assumption empirically holds on ATW V2. |
| Brotli quality 9 is optimal for procedural seed envelope | CARGO-CULTED — INHERITED | Inherited from canonical sister patterns (DP1 + VQ-VAE + grayscale_lut all use quality 9). |
| BUILD before symposium ratification is operator-safe | CARGO-CULTED — REJECT | Per Catalog #325 + symposium PROCEED_WITH_REVISIONS: BUILD without symposium ratification of Variant C scope is forbidden. |

## Observability surface
<!-- Catalog #305 mandatory section; scope = Variant C BUILD scaffold below per Section 5.2 -->

(See §5.2.6 for the observability surface declaration; reproduced here as the canonical Catalog #305 section header.)

#### 5.2.6 Observability surface (Catalog #305)

1. **Inspectable per layer** — `ProceduralVariantConfig`, `derive_procedural_cdf_table_from_seed`, `compose_with_procedural_cdf_table`, `verify_procedural_cdf_in_domain`, `verify_seed_mutation_changes_cdf_bytes` all callable independently; each layer's input + output inspectable at any byte boundary.
2. **Decomposable per signal** — Predicted ΔS decomposes via `predicted_archive_bytes_saved(cdf_bytes, seed_bytes)` × `-25 / 37_545_489`; bytes_saved decomposes into `cdf_bytes - seed_bytes`; envelope decomposes into `SENTINEL(4) + CDF_BYTES(u32) + GEN_KIND_TAG(u8) + SEED_LEN(u16) + seed`.
3. **Diff-able across runs** — Determinism (same seed → same CDF bytes) verified by tests; envelope bytes diff-able byte-for-byte across (seed, generator_kind, num_classes, num_symbols) tuples.
4. **Queryable post-hoc** — Predicted savings + IN-DOMAIN context computable from any (cdf_bytes, seed_bytes) input pair; canonical envelope re-parseable via the sentinel + struct layout.
5. **Cite-able** — Module docstring cites: PR101/PR106 BUILD DESIGN commit `086d3ac1d` + DP1 sister commit `9cbfa471c` + VQ-VAE sister commit `6fea30f22` + grayscale_lut sister + canonical equation #26 source module + this design memo.
6. **Counterfactual-able** — `verify_seed_mutation_changes_cdf_bytes` proves single-byte counterfactuals propagate; the canonical Catalog #272 byte-mutation smoke verifies the operational mechanism.

## Section 6. 5-substrate matrix sequencing impact (5th cascade-mortality data point)

Today's cascade-mortality findings (4 of 5 candidates HYPOTHETICAL + this 5th candidate DEFERRED):

| Candidate | Today's verdict | Reason |
|---|---|---|
| NSCS06 v8 | HYPOTHETICAL substrate architecture | (per sister BUILD HONEST DISCLOSURE memos) |
| PR101 | HYPOTHETICAL substrate architecture | (per sister BUILD HONEST DISCLOSURE memos) |
| PR106 | HYPOTHETICAL substrate architecture | (per sister BUILD HONEST DISCLOSURE memos) |
| grayscale_lut | EMPIRICAL substrate (but APPEND envelope; current GLV1 has no chroma_lut section) | landed `lane_grayscale_lut_procedural_codebook_replacement_variant_20260520` |
| **ATW V2** | **EMPIRICAL substrate (with REPLACE-IN-PLACE) BUT BLOCKED by Catalog #313 + symposium scope question** | **THIS memo** |

**Sister DP1 + VQ-VAE landed EMPIRICAL** (not in matrix memo's 5; sister procedural-variant landings).

### 6.1 Aggregate predicted ΔS recomputation

Corrected per audit:

| Cascade | Path A (cdf only) | Path B (cdf + class_prior) |
|---|---|---|
| ATW V2 ΔS (corrected) | -0.001683 | -0.003281 |
| 5-substrate naive aggregate | -0.016599 | -0.018197 |
| α=0.85 ADDITIVE adjusted | -0.014109 | -0.015467 |
| α=0.5 SUB-ADDITIVE adjusted | -0.008299 | -0.009099 |

### 6.2 Impact on 5-substrate matrix sequencing

Per matrix memo Section 3 (composition-alpha matrix per Catalog #322):

Pair α values involving ATW V2:
- `NSCS06_v8 × ATW_v2`: α=0.85 ADDITIVE (different score-axes; orthogonal)
- `ATW_v2 × DP1`: α=0.70 ADDITIVE/edge (CDF prior compatible with OOD-derived basis but partially redundant)
- `ATW_v2 × VQ_VAE`: α=0.55 SUB-ADDITIVE (both scorer-class targeting; overlap)

If ATW V2 is DEFERRED, the 5-substrate matrix becomes a 4-substrate matrix:
- Removed pair α values: 3 (NSCS06×ATW, ATW×DP1, ATW×VQ_VAE)
- Recomputed aggregate (4-substrate naive): -0.014916
- Recomputed aggregate (4-substrate α=0.85): -0.012679
- Recomputed aggregate (4-substrate α=0.5): -0.007458

The 4-substrate sequencing is STILL frontier-pursuit class per Catalog #309 (would break 0.18 floor if α=0.85 realized).

## Section 7. Composition-alpha estimate (Catalog #322)

Per `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2`:

For the 4 landed sister procedural variants (DP1, VQ-VAE, grayscale_lut + this DEFERRED ATW V2):

| Pair | Predicted α | Band | Rationale |
|---|---|---|---|
| DP1 × VQ-VAE | 0.70 | ADDITIVE/edge | Different IN-DOMAIN contexts (`dp1_codebook_bytes` vs `intermediate_transform_quantizer`) |
| DP1 × grayscale_lut | 0.80 | ADDITIVE | OOD-derived basis × in-domain LUT replacement |
| VQ-VAE × grayscale_lut | 0.65 | SUB-ADDITIVE | Both chroma-class targeting; partial overlap |
| DP1 × ATW V2 (DEFERRED) | 0.70 | ADDITIVE/edge | (per matrix memo) |
| VQ-VAE × ATW V2 (DEFERRED) | 0.55 | SUB-ADDITIVE | (per matrix memo) |
| grayscale_lut × ATW V2 (DEFERRED) | 0.60 | SUB-ADDITIVE | Both chroma-class targeting (new estimate; not in matrix memo) |

**Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag"**: ALL α values above are `[prediction]` only; first paired-anchor smoke is the first empirical anchor.

## Section 8. Sister regression

This memo introduces NO code changes; sister regression scope is N/A.

The audit findings DO suggest a follow-on sister-correction landing on the matrix memo (correct byte-count + corrected predicted ΔS) per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — but that is operator-routable, not in scope for this BUILD design.

## Section 9. Catalog gate verdicts (predicted; design-only)

| Gate | Verdict | Notes |
|---|---|---|
| Catalog #1 `check_no_mps_fallback_default` | N/A | Design only; no code |
| Catalog #229 `check_subagent_landing_includes_premise_verification_evidence` | PASS | This memo IS the premise verifier (audit of matrix memo claim + D4 probe ledger + symposium memo + substrate code) |
| Catalog #287 `check_no_docstring_overstatement_without_evidence_tag` | PASS | All claims tagged `[prediction]` or empirically-sourced |
| Catalog #290 `check_substrate_design_memo_has_canonical_vs_unique_decision_section` | PASS | §5.2.3 |
| Catalog #292 `check_grand_council_deliberation_has_explicit_assumption_statements` | PASS | 5 assumption-adversary verdicts |
| Catalog #294 `check_substrate_landing_memo_has_9_dim_checklist_evidence_section` | PASS | §5.2.4 |
| Catalog #296 `check_substrate_predicted_band_has_dykstra_feasibility_check` | PASS | predicted_band cites canonical equation #26 closed-form (Shannon R(D) source) |
| Catalog #297 `check_substrate_signal_axis_destruction_has_reversibility_probe` | N/A | Procedural variant is rate-axis only (under Variant C scoping); no signal-axis destruction |
| Catalog #300 `check_council_deliberation_declares_tier_in_frontmatter` | PASS | Frontmatter complete with `council_tier: T1` + 7 attendees + verdict + dissent + assumption-adversary |
| Catalog #303 `check_substrate_design_memo_has_cargo_cult_audit_section` | PASS | §5.2.5 |
| Catalog #305 `check_substrate_design_memo_has_observability_surface_section` | PASS | §5.2.6 |
| Catalog #309 `check_substrate_design_memo_declares_horizon_class` | PASS | `horizon_class: frontier_pursuit` |
| Catalog #313 `check_dispatch_target_has_no_predecessor_adjudicated_outcome` | **WOULD-BLOCK-DISPATCH** | D4 INDEPENDENT verdict still blocking until 2026-06-15; design memo only does NOT dispatch |
| Catalog #322 `check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha` | PASS | All α values explicitly `[prediction]` not autopilot-derived |
| Catalog #323 `check_no_score_claim_without_canonical_provenance` | PASS | predicted_delta_s explicitly NOT a score claim |
| Catalog #324 `check_no_predicted_band_without_post_training_tier_c_validation` | PASS | `predicted_band_validation_status=pending_post_training` declared |
| Catalog #325 `check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor` | **PARTIAL** | 2026-05-18 symposium covers V2 base substrate; does NOT cover procedural-variant scope. Follow-on symposium required if operator approves Variant C BUILD. |
| Catalog #340 `check_subagent_commit_serializer_invokes_sister_checkpoint_guard` | N/A | Design memo only |
| Catalog #344 `check_empirical_finding_memo_references_canonical_equation` | PASS | References `procedural_codebook_from_seed_compression_savings_v1` in frontmatter + body |

## Section 10. 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map** = N/A (design memo only; no per-tensor sensitivity contribution).
- **Hook #2 Pareto constraint** = PLANNED (if Variant C BUILD lands; predicted ΔS = -0.001683 contributes additively to rate-axis Pareto polytope).
- **Hook #3 bit-allocator** = PLANNED (32-byte seed slot replaces 2,560-byte CDF slot when Variant C BUILD lands).
- **Hook #4 cathedral autopilot dispatch** = N/A at design surface (BUILD-time wire-in would route via sister consumer `tac.cathedral_consumers.procedural_codebook_generator_consumer` auto-discovered per Catalog #335).
- **Hook #5 continual-learning posterior** = N/A at design surface (first empirical anchor via `update_equation_with_empirical_anchor` would land post-paired-smoke).
- **Hook #6 probe-disambiguator** = ACTIVE — THIS memo IS the disambiguator between Variant A (additive bolt-on on V2 cooperative-receiver — empirically falsified) vs Variant C (decoupled rate-axis-only — viable BUILD path pending operator scoping decision).

## Section 11. Sister coordination + collision verdict

- **Sister-DISJOINT** from in-flight PARSER-SAFE SUBSET SMOKE (slot 2 commit `a988f9d`) — different file scope (PR101 parser-safe vs ATW V2 design memo).
- **Sister-DISJOINT** from in-flight HONEST CASCADE-MORTALITY ASSESSMENT (slot 3 commit `a3839bc`) — different file scope (cascade-mortality vs ATW V2 design memo).
- **Sister-DISJOINT** from grayscale_lut PROCEDURAL VARIANT BUILD (commit `f037d1144` 2026-05-21T01:25Z) — different substrate.
- **Sister-DISJOINT** from DP1 + VQ-VAE PROCEDURAL VARIANT BUILD landings.
- **Step 0 helper verdict**: `PROCEED: no sister commits touched any of 1 target file(s) within the 12-hour lookback window. Safe to write.`
- Catalog #314 / #340 sister-checkpoint guard: pre-write check PASSED; no sister has my target file in `files_touched` checkpoint within last 60 minutes.

## Section 12. Operator-routable Top-3 next-actions

1. **DECIDE PROCEDURAL VARIANT SCOPING**: Variant A (additive bolt-on on V2 cooperative-receiver — empirically PRE-FALSIFIED) vs Variant C (decoupled rate-axis-only — viable BUILD path). Recommended: **Variant C** per Assumption-Adversary verbatim; preserves cooperative-receiver paradigm for V2-1 redesign + opens a parallel rate-axis-only research path that does NOT depend on D4 probe re-verdict. Estimated cost: $0 GPU + ~5 minutes operator decision.

2. **IF Variant C APPROVED**: spawn follow-on per-substrate symposium per Catalog #325 (T2 sextet pact + 3 specialists for procedural-variant scope — recommend Shannon LEAD, Daubechies, Mallat for canonical equation #26 IN-DOMAIN context ratification; ~$0 GPU + ~3h editor + 1 council-grade design memo). Then BUILD landing via sister DP1/VQ-VAE/grayscale_lut canonical pattern (~$0 GPU + ~1 week subagent + ~600 LOC). Then paired-smoke contest-CUDA + contest-CPU (~Modal A10G $0.50) to validate predicted ΔS = -0.001683 empirically.

3. **SISTER-CORRECTION LANDING ON 5-SUBSTRATE MATRIX MEMO**: per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE, append a corrections section to `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md` documenting (a) ATW V2 byte-count corrections (2,560 not 3,072), (b) ATW V2 DEFER per this memo, (c) 4-substrate corrected aggregate ΔS = -0.014916 naive / -0.012679 α=0.85 / -0.007458 α=0.5. Estimated cost: $0 GPU + ~30 min sister subagent.

## Section 13. Blockers

1. **Catalog #313 D4 INDEPENDENT verdict** (expires 2026-06-15; 26 days from now) — STILL BLOCKING ATW V2 paid Modal dispatch. Procedural variant under Variant A scoping inherits this blocker; Variant C scoping decouples but requires explicit operator decision.

2. **Per-substrate symposium scope mismatch** — the 2026-05-18 symposium covers V2 base substrate reactivation; does NOT cover procedural-variant BUILD scope. Follow-on symposium required if operator approves Variant C BUILD.

3. **Cooperative-receiver hypothesis operational falsification** — procedural seed-derived CDF table destroys learned `I(latent; segnet_class)` conditioning under Variant A scoping; this hypothesis-level finding is documented but does NOT prevent Variant C BUILD.

## Section 14. Cross-references

- Parent: `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md` (5-substrate matrix; commit `b3e3442c3`)
- Sister: `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (V2 reactivation symposium)
- Sister: `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md` (D4 probe verdict)
- Sister: `feedback_grayscale_lut_procedural_variant_build_landed_20260520.md` (3rd PROCEDURAL VARIANT BUILD; canonical pattern)
- Sister: `feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md` (DP1 paired-smoke pre-dispatch design)
- Sister: `feedback_slot_mg7_bundle_master_gradient_exploits_landed_20260520.md` (VQ-VAE sister landing)
- Canonical equation: `src/tac/canonical_equations/procedural_codebook_savings.py` (canonical equation #26; line 103 `atw_v2_codec_quantizer_lut` IN-DOMAIN context)
- Substrate code: `src/tac/substrates/atw_codec_v2/` (623 LOC archive.py; 6 files; full ATW2 grammar)
- Probe ledger: `.omx/state/probe_outcomes.jsonl` (D4 verdict row `probe_id=atw_v2_d4_h_latent_given_scorer_class_20260516`)
- CLAUDE.md non-negotiables: "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + "Forbidden premature KILL without research exhaustion"
- Catalog gates cited: #1 / #220 / #229 / #240 / #272 / #287 / #290 / #292 / #294 / #296 / #297 / #298 / #300 / #303 / #305 / #309 / #313 / #315 / #322 / #323 / #324 / #325 / #335 / #340 / #344
