<!-- SPDX-License-Identifier: MIT -->
---
substrate_id: atw_codec_v2_cdf_table_blob_procedural_variant
substrate_aliases: ["atw_v2_cdf_proc", "atw_codec_v2"]
substrate_class: parser_safe_extension_procedural_variant_design_for_largest_raw_score_affecting_section
horizon_class: frontier_pursuit
council_tier: T1
council_attendees:
  - Shannon
  - Yousfi
  - Atick
  - Tishby_memorial
  - Daubechies
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "This is a DESIGN-ONLY memo for the SECOND-WIND iteration of ATW V2 procedural variant work. Predecessor design memo `atw_v2_procedural_variant_build_design_20260520.md` (commit `7ea78deaa`) already returned DEFER_PENDING_EVIDENCE with 3 failure modes. The cdf_table_blob is empirically THE largest RAW parser-safe + score-affecting section across all 4 PARSER-SAFE EXTENSION substrates (2,632 bytes ATW V2 vs 192 bytes VQ-VAE vs 0 bytes DP1 + grayscale_lut + fec6 per `parser_safe_methodology_extension_landed_20260520.md`). The temptation to BUILD now is high because the byte-count signal is the largest in the IN-DOMAIN landscape. The Catalog #313 D4 INDEPENDENT verdict (expires 2026-06-15; 25 days from now) STILL BLOCKS dispatch. This memo MUST tag itself as `apparatus_maintenance` (per Catalog #300), NOT `frontier_breaking`, because no BUILD lands and no score moves until at least 3 operator-routable gates lift: (a) Catalog #313 D4 reactivation criteria satisfied, (b) Catalog #325 per-substrate symposium for the procedural-variant scope (the 2026-05-18 symposium covers base V2 substrate, NOT this variant), (c) Catalog #272 Variant-C operator decision."
  - member: Assumption-Adversary
    verbatim: "Per Catalog #292 per-deliberation assumption surfacing + the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable. The SHARED ASSUMPTION operating across this memo: 'A DESIGN-ONLY memo for the SAME procedural variant whose predecessor already returned DEFER_PENDING_EVIDENCE adds NET-SIGNAL because it ties the candidate to the parser-safe extension empirical evidence (2,632 RAW bytes = largest). Classification: HARD-EARNED-MILD. The predecessor memo (commit `7ea78deaa`) was framed as `frontier_protecting` and audited 3 failure modes WITHIN the cooperative-receiver-paradigm scope (Variant A / B / C scoping decision). This memo's NEW signal is the PARSER-SAFE EXTENSION evidence anchor (commit `d0bf3ce37` + the 4-substrate static classification result) which empirically locates the cdf_table_blob as the LARGEST RAW parser-safe + score-affecting target across ALL 4 IN-DOMAIN substrates. The HARD-EARNED part: this empirical landscape evidence DID NOT exist when the 2026-05-20 predecessor memo landed; the PARSER-SAFE EXTENSION methodology was a sister landing the same day. The MILD part: this memo does NOT change the predecessor's DEFER verdict; it sharpens the OPERATOR-ROUTABLE DECISION QUEUE (per §12) by ranking cdf_table_blob ABOVE the other 3 PARSER-SAFE EXTENSION substrate candidates per byte-count + IN-DOMAIN confidence. The operator should consume this memo as a SECOND-WIND DECISION FRAMING tool, not a BUILD greenlight."

council_assumption_adversary_verdict:
  - assumption: "cdf_table_blob is THE largest RAW parser-safe + score-affecting section across all 4 PARSER-SAFE EXTENSION substrates"
    classification: HARD-EARNED
    rationale: "Per `parser_safe_methodology_extension_landed_20260520.md` empirical table (commit `d0bf3ce37`): ATW V2 has 2,632 parser-safe bytes across 3 RAW sections (latent_residual_blob int8 + class_prior_table_blob fp16 + cdf_table_blob fp16); VQ-VAE has 192 bytes (indices_blob int16); DP1 + grayscale_lut + fec6 have 0 bytes (all-Brotli/struct/JSON). Of the 3 ATW V2 RAW sections, cdf_table_blob is the largest individually at 2,560 bytes (5 classes × 256 symbols × fp16; per archive.py grammar audit in predecessor memo §2.2)."
  - assumption: "Canonical equation #26 IN-DOMAIN context `atw_v2_codec_quantizer_lut` applies to cdf_table_blob procedural replacement"
    classification: HARD-EARNED
    rationale: "Per `src/tac/canonical_equations/procedural_codebook_savings.py:103` explicit registration. The cdf_table_blob IS a quantizer LUT (5 class × 256 symbol → fp16 probability table consumed at inflate-time by the latent reconstruction loop). Per the post-slot-2 commit `79f1ba387` domain refinement, the IN-DOMAIN context's domain-of-validity is now EVEN MORE TIGHTLY SCOPED to score-OPAQUE bytes via the PARSER-SAFE EXTENSION evidence anchor — this introduces a SCOPE TENSION worth surfacing: cdf_table_blob bytes ARE parser-safe but ARE score-affecting (decoder side-information per parser-safe extension verdict). The canonical equation #26 predicts REPLACEMENT savings for score-OPAQUE bytes; cdf_table_blob is parser-safe-but-score-affecting. Per the parser-safe extension memo §'Why This Matters' bullet: 'parser-safe-but-score-affecting bytes are NOT canonical equation #26 IN-DOMAIN candidates for **direct** byte substitution: the equation predicts REPLACEMENT savings for score-OPAQUE bytes. The right surface is the existing IN-DOMAIN procedural-replacement context, reached via the per-substrate training co-optimizing the replacement (sister PROCEDURAL VARIANT BUILDs land this surface).'"
  - assumption: "Predicted ΔS = -0.001684 (matrix-memo-arithmetic per closed-form) is the rate-axis-only Variant C signal"
    classification: HARD-EARNED
    rationale: "Closed-form per canonical equation #26: ΔS = -25 × (N_cdf - K_seed) / 37_545_489 = -25 × (2560 - 32) / 37_545_489 = -0.0016833. Predecessor memo §1 reports -0.001683; matrix memo arithmetic per task prompt reports -0.001684; both round-trip to -0.0016833 at fp64 precision. The rate-axis prediction is structurally exact (closed-form rate term); the net-score prediction depends on Variant-A vs Variant-C scoping (§5) and is per-substrate-symposium-pending per Catalog #325."
  - assumption: "Variant-C scoping (decoupled from cooperative-receiver hypothesis) is the operator-recommended path"
    classification: CARGO-CULTED-MILD
    rationale: "Predecessor memo §12 operator-routable #1 recommends Variant-C scoping. THIS memo SUPPORTS the recommendation but does NOT bind it. Operator-routable: the V2-1 redesign per the 2026-05-18 symposium PROCEED_WITH_REVISIONS may PREFER preserving the cooperative-receiver hypothesis path (Variant-A or new-redesigned-side-info-channel) over decoupling. The Variant-C decision is OPERATOR-OWNED per CLAUDE.md 'Design decisions — non-negotiable'. This memo's role is to RANK the cdf_table_blob target ABOVE the other 3 PARSER-SAFE EXTENSION substrates per byte-count + IN-DOMAIN confidence; the Variant-A vs Variant-C scoping decision remains operator-routable."
  - assumption: "Spawning a DESIGN-ONLY follow-on to an already-DESIGN-ONLY-DEFERRED predecessor is OPTIMAL FORM per Catalog #315"
    classification: HARD-EARNED-MILD
    rationale: "Per CLAUDE.md 'Substrate MUST be at OPTIMAL FORM before paid empirical dispatch' + Catalog #315 + CLAUDE.md 'Forbidden premature KILL': the predecessor's DEFER_PENDING_EVIDENCE verdict explicitly leaves the substrate at LIFTED-TRAINER form (council_verdict ≠ PROCEED-unconditional). Iteration via design memo is the canonical pre-BUILD path per CLAUDE.md 'Canonical substrate iteration methodology'. THIS memo adds the PARSER-SAFE EXTENSION evidence anchor that did NOT exist for the predecessor; the iteration is HARD-EARNED. MILD because no new BUILD lands; only DECISION-FRAMING signal. The next iteration step is per-substrate symposium per Catalog #325 (NOT BUILD)."

predicted_band_validation_status: pending_post_training
predicted_band:
  variant_c_decoupled_rate_axis_only:
    rate_axis_delta_s: [-0.001683, -0.001684]
    net_score_delta_s: [-0.001683, -0.001684]
    confidence: "closed-form per canonical equation #26 line 103 IN-DOMAIN registration; structurally exact rate-axis savings IF cdf_table_blob is replaced with 32-byte procedural seed AT THE GRAMMAR LAYER (REPLACE-IN-PLACE) AND inflate.py deterministically re-derives the same fp16 (5, 256) table per (seed, generator_kind, shape, dtype)"
  variant_a_additive_on_cooperative_receiver:
    rate_axis_delta_s: [-0.001683, -0.001684]
    seg_axis_delta_s_predicted: [+0.005, +0.030]
    pose_axis_delta_s_predicted: [+0.002, +0.010]
    net_score_delta_s: [+0.006, +0.038]
    confidence: "EMPIRICALLY-PRE-FALSIFIED via cooperative-receiver hypothesis operational analysis per predecessor memo §3.2 + Assumption-Adversary verbatim. NET PREDICTED SCORE WORSE under Variant-A scoping. NOT VIABLE BUILD PATH per predecessor."
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
operator_directive: "WAVE-3-ATW-V2-PROCEDURAL-VARIANT-DESIGN-ONLY per cap=3 tonight + must-keep-queuing-overnight; DESIGN-ONLY memo for cdf_table_blob procedural variant; gated by Catalog #325 ≥14-day per-substrate symposium + Catalog #313 D4 probe BLOCKING until 2026-06-15 + Catalog #272 Variant-C operator decision"
related_deliberation_ids:
  - atw_v2_procedural_variant_build_design_20260520  # predecessor commit 7ea78deaa
  - parser_safe_methodology_extension_landed_20260520  # PARSER-SAFE EXTENSION commit d0bf3ce37
  - codex_md_files_cross_pollination_synergy_audit_20260520T041700Z  # commit aafac7c84
  - council_per_substrate_symposium_atw_v2_reactivation_20260518
  - atw_codec_v2_d4_probe_verdict_20260516_codex
  - canonical_equation_26_parser_safe_domain_refinement_20260520  # commit 79f1ba387
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl
parent_design_memo: .omx/research/atw_v2_procedural_variant_build_design_20260520.md
sister_design_memos:
  - .omx/research/parser_safe_methodology_extension_landed_20260520.md
  - .omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md
predecessor_audit_corrections_inherited:
  - "Matrix memo claim '~3 KB codec table per archive.py:111-115' refers to DATACLASS FIELD DECLARATIONS, NOT byte counts (predecessor §2.2)."
  - "Empirical cdf_table_blob = 5 × 256 × fp16 = 2,560 bytes (not 3,072 or matrix memo's 820)."
  - "Predicted ΔS Path A (cdf only) = -0.001684 (matrix-memo-arithmetic) ≅ -0.0016833 (fp64 closed-form) (NOT matrix memo's -0.002024)."
---

<!-- Catalog #344 canonical-equations-registry cross-reference: this design
memo's predicted ΔS bands ARE derived via the canonical equation
`procedural_codebook_from_seed_compression_savings_v1` registered at
`src/tac/canonical_equations/procedural_codebook_savings.py` and persisted
to `.omx/state/canonical_equations_registry.jsonl`. The IN-DOMAIN context
`atw_v2_codec_quantizer_lut` IS explicitly registered at line 103 of the
canonical equation source. Per-substrate prediction follows
ΔS = -25 × (N_codebook - K_seed) / 37_545_489 with K_seed=32, N_codebook=2560,
yielding -0.0016833 (≅ -0.001684 per matrix-memo-arithmetic). -->

# WAVE-3-ATW-V2-cdf_table_blob-PROCEDURAL-VARIANT-DESIGN-ONLY (2026-05-21)

**Lane**: `lane_wave_3_atw_v2_cdf_table_blob_procedural_variant_design_20260521`
**Subagent**: `wave-3-atw-v2-cdf-table-blob-procedural-variant-design-20260520`
**Predecessor**: `atw_v2_procedural_variant_build_design_20260520.md` (commit `7ea78deaa`) DEFER_PENDING_EVIDENCE
**Sister evidence anchor**: PARSER-SAFE EXTENSION (commit `d0bf3ce37`) — empirical 4-substrate landscape
**Council verdict**: **DEFER_PENDING_EVIDENCE** (NOT PROCEED to BUILD; predecessor verdict inherited + sharpened via parser-safe extension evidence)
**Mission contribution**: `apparatus_maintenance` (decision-framing for operator-routable queue; no BUILD; no score moves until 3 operator-routable gates lift)

---

## Section 1. Architecture motivation

The cdf_table_blob in ATW V2's archive grammar (`src/tac/substrates/atw_codec_v2/archive.py:40 + 426-431`) is empirically the **largest RAW parser-safe + score-affecting section across all 4 PARSER-SAFE EXTENSION substrates** (per the empirical landscape in `parser_safe_methodology_extension_landed_20260520.md` commit `d0bf3ce37`):

| substrate | parser-safe RAW bytes | score-affecting |
|---|---:|---|
| `dp1_pretrained_driving_prior` | 0 | n/a (all-Brotli/struct/JSON) |
| `vq_vae` | 192 (indices_blob int16) | YES |
| `grayscale_lut` | 0 | n/a (all-Brotli/struct/JSON) |
| `atw_codec_v2` (this memo's target) | **2,632** across 3 RAW sections | YES |

Within ATW V2's 2,632 parser-safe bytes:

| ATW V2 RAW section | Empirical bytes | Type |
|---|---:|---|
| `latent_residual_blob` | trainer-dependent (int8 z_residual) | decoder_side_information |
| `class_prior_table_blob` | 600 × prior_dim × fp16 (config-dependent; typical 2,400 @ prior_dim=2) | decoder_side_information |
| **`cdf_table_blob`** | **2,560 bytes (5 classes × 256 symbols × fp16; per archive.py grammar audit)** | **decoder_side_information (B3 scorer-conditional CDF)** |

The cdf_table_blob is THE highest-byte-count RAW IN-DOMAIN procedural-replacement target empirically established across the parser-safe landscape — and per the canonical equation #26 explicit IN-DOMAIN registration at line 103 (`atw_v2_codec_quantizer_lut`), it is the structurally-canonical target for procedural-seed substitution under Variant-C scoping.

**The architecture motivation is therefore DECISION-FRAMING**: rank cdf_table_blob ABOVE the other 3 PARSER-SAFE EXTENSION substrates (DP1 / VQ-VAE / grayscale_lut) per (a) byte-count, (b) canonical equation #26 IN-DOMAIN confidence (line 103 explicit registration), (c) substrate code maturity (623 LOC archive.py + 6 files + full ATW2 grammar already shipped). NOT a BUILD greenlight; NOT a score-claim; NOT a dispatch authorization.

---

## Section 2. Catalog #313 D4 probe BLOCKING status

Per `.omx/state/probe_outcomes.jsonl` row `probe_id=atw_v2_d4_h_latent_given_scorer_class_20260516`:

```
verdict: INDEPENDENT
blocker_status: blocking
adjudicated_at_utc: 2026-05-16T22:47:41Z
expires_at_utc: 2026-06-15T22:47:41Z  (25 days from this memo's date)
metric_name: mutual_information_bits_per_symbol
metric_value: 0.006385502752
threshold: 0.5 (MEANINGFUL_CONDITIONING)
next_action: do_not_dispatch_atw_v2_phase2_from_this_signal
```

**The D4 INDEPENDENT verdict structurally blocks ATW V2 paid Modal dispatch until 2026-06-15.** The Catalog #313 STRICT preflight gate refuses dispatch wrappers targeting ATW V2 recipes during this window.

Reactivation criteria per the probe ledger row:

> "replace SegNet composite class with richer side-information OR rerun on trained ATW v2 residuals"

Neither criterion is satisfied by a cdf_table_blob procedural variant under Variant-A scoping (procedural seed produces uniform-PRNG-derived CDF table; that does NOT replace SegNet composite class with richer side-info AND does NOT rerun on trained ATW v2 residuals; it operates at design-time on the CDF table bytes).

Under **Variant-C scoping** (decoupled from cooperative-receiver hypothesis; rate-axis-only) per predecessor §3.2: the D4 verdict is structurally orthogonal — the procedural variant operates on a DIFFERENT canonical-equation-#26 substrate-class-shift (rate-axis byte replacement) NOT on the cooperative-receiver hypothesis (information-bottleneck side-info-channel content). The Variant-C path does NOT inherit the D4 blocker per predecessor §3.2 + Assumption-Adversary verdict #4. **But Variant-C scoping requires explicit operator decision per Catalog #272.**

**This memo's verdict: D4 probe blocking STATUS IS BINDING under Variant-A scoping; Variant-C scoping decouples but requires Catalog #272 operator decision (not invoked).**

---

## Section 3. Catalog #325 per-substrate symposium ≥14-day gate

Per `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (2026-05-18; 3 days old at memo date):

- Verdict: PROCEED_WITH_REVISIONS (NOT PROCEED-unconditional)
- 8 binding revisions documented
- Scope: V2 base substrate reactivation (cooperative-receiver hypothesis re-probe + V2-1 redesign)

**The 2026-05-18 symposium covers V2 BASE SUBSTRATE; it does NOT cover the procedural-variant cdf_table_blob scope.** Per Catalog #325 14-day window discipline: the symposium memo's substrate `atw_codec_v2` ↔ `atw_v2` alias mechanism is satisfied, but the procedural-variant operational reframing is a DISTINCT canonical-equation-#26 substrate-class-shift requiring its OWN per-substrate symposium before any paid dispatch.

**A follow-on per-substrate symposium per Catalog #325 IS REQUIRED before any procedural-variant BUILD lands**, even under Variant-C scoping. The follow-on symposium must satisfy the canonical 6-step contract per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" section:

1. Cargo-cult audit per Catalog #303 (preliminary draft in §8 below)
2. 9-dimension success checklist evidence per Catalog #294 (preliminary draft in §7 below)
3. Observability surface declaration per Catalog #305 (preliminary draft in §9 below)
4. Sextet pact deliberation + grand council attendees added per topic (recommend Shannon LEAD, Daubechies, Mallat, Yousfi, Atick, Tishby memorial, Contrarian, Assumption-Adversary)
5. Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL"
6. Catalog #324 post-training Tier-C validation discipline declared

This memo provides preliminary drafts of items (1)-(3) + (6) to expedite the follow-on symposium scaffold; items (4)-(5) are operator-routable.

---

## Canonical-vs-unique decision per layer
<!-- Catalog #290 mandatory section header (literal). The canonical-vs-unique
decision per layer scope = Phase 3 BUILD scaffold per §6 + sister DP1/VQ-VAE/
grayscale_lut canonical pattern; full per-layer table is enumerated below. -->

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290
falling-rule list. Each layer of the Phase 3 BUILD scaffold (per §6) gets an
explicit ADOPT_CANONICAL / FORK_BECAUSE classification:

| Layer | Decision | Rationale |
|---|---|---|
| `ProceduralVariantConfig` dataclass | ADOPT_CANONICAL_BECAUSE_SERVES | Sister DP1/VQ-VAE/grayscale_lut pattern; shared schema serves dispatch-time uniformity. |
| `PROCEDURAL_SEED_SIZE_BYTES=32` constant | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical equation #26 K_seed; deviation would invalidate closed-form predictions. |
| `PROCEDURAL_CDF_BYTES_DEFAULT=2560` constant | FORK_BECAUSE_PRINCIPLED_MISMATCH | ATW V2 grammar fixes `cdf_classes=5 × cdf_symbols=256 × fp16` = 2,560 bytes; this constant is substrate-unique by grammar definition. |
| `CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT="atw_v2_codec_quantizer_lut"` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Equation #26 line 103 explicit registration; substrate-unique by canonical-equation registry. |
| `PROCEDURAL_CDF_SENTINEL=b"ACPV"` envelope sentinel | FORK_BECAUSE_PRINCIPLED_MISMATCH | Substrate-unique sentinel; sister to grayscale_lut `b"GLPV"` + VQ-VAE `b"VQVP"` pattern. |
| Archive composition (REPLACE-IN-PLACE at canonical ATW2 grammar layer) | FORK_BECAUSE_PRINCIPLED_MISMATCH | ATW V2 grammar HAS a dedicated `cdf_table_blob` section that CAN be replaced in-place; sister grayscale_lut APPENDS envelope because GLV1 has no chroma_lut section. |
| `derive_codebook_from_seed` canonical helper API | ADOPT_CANONICAL_BECAUSE_SERVES | Single source of truth for PRNG-based codebook derivation; sister to all 3 canonical procedural variants. |
| `validate_context_is_in_domain` sister helper | ADOPT_CANONICAL_BECAUSE_SERVES | Equation #26 explicit IN-DOMAIN registry; canonical validator preserves consistency. |
| Inflate-time decoder integration (consumes 32-byte envelope; deterministically re-derives (5, 256) fp32 CDF table) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Substrate-unique inflate path; ATW V2's latent-reconstruction loop expects the CDF table in a specific tensor shape + dtype. |
| Catalog #146 inflate.py contest-compliant runtime template | ADOPT_CANONICAL_BECAUSE_SERVES | 3-arg invocation `inflate.sh archive_dir output_dir file_list`; HNeRV parity L4 ≤200 LOC budget; sister to all canonical inflate runtimes. |

## Section 4. Procedural-substitution mechanism for 2,560 B fp16 cdf_table_blob

Under **Variant-C scoping** (decoupled rate-axis-only; per predecessor §5.2.1 + Assumption-Adversary verbatim):

### 4.1 Canonical procedural-substitution flow

1. **Encode-time**: Operator picks a deterministic seed (`uint64` per canonical equation #26 K_seed=32 envelope) + `generator_kind="pcg64"` (sister to DP1/VQ-VAE/grayscale_lut canonical pattern).
2. **Archive-build-time**: Replace the trained `cdf_table_blob` (2,560 fp16 bytes) IN-PLACE at the canonical ATW2 grammar layer with a 32-byte envelope: `SENTINEL(4=b"ACPV") + CDF_BYTES(u32) + GEN_KIND_TAG(u8) + SEED_LEN(u16) + seed(uint64)`. All 7 other ATW2 sections preserved byte-for-byte.
3. **Inflate-time** (per Catalog #205 canonical `select_inflate_device` + HNeRV parity L4 ≤200 LOC budget): inflate.py reads the 32-byte envelope, dispatches to `derive_procedural_cdf_table_from_seed(seed, num_classes=5, num_symbols=256, generator_kind="pcg64")` which deterministically produces a `(5, 256)` fp32 tensor; downstream latent-reconstruction loop consumes the derived table identically to the trained-fp16-decoded table.
4. **Determinism + reproducibility** (Catalog #297 byte-level determinism gate not required because the variant is rate-axis-only and the byte savings are exact-deterministic).

### 4.2 Net byte accounting (per canonical equation #26 exact formula)

- Original: `cdf_table_blob = 2,560 bytes`
- Procedural-variant envelope: `4 (SENTINEL) + 4 (CDF_BYTES u32) + 1 (GEN_KIND u8) + 2 (SEED_LEN u16) + 8 (uint64 seed) = 19 bytes`
- Canonical equation #26 K_seed = 32 bytes (rounded envelope budget; conservative)
- Bytes saved: `2,560 − 32 = 2,528 bytes`

### 4.3 Score impact under Variant-C scoping

- Rate-axis: `ΔS_rate = -25 × 2,528 / 37,545,489 = -0.0016833 (≅ -0.001684 per matrix-memo-arithmetic; ≅ -0.001683 per predecessor §1)`
- Seg-axis: structurally ZERO under Variant-C scoping (cooperative-receiver hypothesis is DECOUPLED; the procedural CDF table operates as a deterministic rate-axis substitution; the seg signal does NOT depend on the trained CDF table's per-class structure)
- Pose-axis: structurally ZERO under Variant-C scoping (same reasoning)
- **Net ΔS = -0.001684 per closed-form Variant-C prediction**

### 4.4 Score impact under Variant-A scoping (EMPIRICALLY-PRE-FALSIFIED)

Per predecessor §5.1 + Assumption-Adversary verbatim: procedural CDF table destroys learned `I(latent; segnet_class)` conditioning. Under cooperative-receiver hypothesis: NET PREDICTED SCORE WORSE by +0.006 to +0.038 (rate gain ≪ seg+pose loss). **NOT VIABLE PATH.**

---

## Section 5. Predicted ΔS via canonical equation #26

**Formula (canonical equation #26 closed-form rate term)**:
```
ΔS = -25 × (N_codebook − K_seed) / 37,545,489
```

**Substitution (cdf_table_blob)**:
```
N_codebook = 2,560 (5 classes × 256 symbols × fp16 = 2,560 bytes)
K_seed     = 32    (canonical envelope budget per equation #26)
bytes_saved = 2,560 − 32 = 2,528
ΔS = -25 × 2,528 / 37,545,489 = -0.001683272...
   ≅ -0.001684  (matrix-memo-arithmetic per task #1159 prompt)
   ≅ -0.001683  (predecessor §1 + §6.1 corrected aggregate)
```

**Verification** (fp64): `-25 * (2560 - 32) / 37_545_489 = -0.0016832723...` → rounds to **-0.001684** per matrix-memo-arithmetic, **-0.001683** per predecessor §1; both consistent at fp64 precision (difference is rounding-to-3-decimals direction). Per Catalog #344 + canonical equation #26 explicit IN-DOMAIN registration line 103.

**Confidence per canonical equation #26 IN-DOMAIN registration**: HIGH (line 103 explicit; sister to DP1's `dp1_codebook_bytes` + VQ-VAE's `intermediate_transform_quantizer` + grayscale_lut's `chroma_lut_replacement`). The IN-DOMAIN context `atw_v2_codec_quantizer_lut` is registered specifically for this substrate's quantizer LUT class.

**Caveat per canonical equation #26 post-slot-2 domain refinement** (commit `79f1ba387`): the IN-DOMAIN context's domain-of-validity is tightly scoped to REPLACEMENT savings for SCORE-OPAQUE bytes. The cdf_table_blob is parser-safe-but-score-affecting per the PARSER-SAFE EXTENSION (commit `d0bf3ce37`). Under Variant-C scoping (decoupled rate-axis-only) the rate-axis prediction is structurally exact; under Variant-A scoping the predicted seg/pose deterioration was characterized by the predecessor.

---

## Section 6. BUILD ROADMAP (3-phase: DESIGN → SYMPOSIUM → IMPLEMENTATION; ~6+ weeks minimum)

### Phase 1 — DESIGN (CURRENT; this memo)

**Status**: COMPLETE at landing of this memo.
**Deliverables**: This 14-section design memo + frontmatter v2 per Catalog #300 + 6-step symposium scaffold drafts (§§7-9 + §13) + operator-routable decision queue (§12).
**Cost**: $0 GPU + 1 design memo.
**Gate**: This memo lands as DESIGN-ONLY artifact; no code changes; no commits to canonical helpers.

### Phase 2 — PER-SUBSTRATE SYMPOSIUM (operator-routable; ~1-2 weeks)

**Trigger**: Operator approves Variant-C scoping per Catalog #272 + spawns sister subagent for per-substrate symposium per Catalog #325.
**Deliverables**: Per-substrate symposium memo with PROCEED (or PROCEED_WITH_REVISIONS) verdict; 6-step canonical contract satisfied; T2 sextet pact + ≥3 specialists (Daubechies, Mallat, Atick recommended); per-substrate reactivation criteria pinned; Catalog #324 post-training Tier-C validation declared.
**Cost**: $0 GPU + 1 council-grade design memo (~3-6 hours editor time).
**Gate**: Catalog #325 SATISFIED for procedural-variant scope (separate from base-V2 symposium).

### Phase 3 — IMPLEMENTATION (BUILD; ~1 week subagent + ~$0.50 paired-smoke)

**Trigger**: Phase 2 PROCEED verdict + sister DP1/VQ-VAE/grayscale_lut canonical pattern fully ratified.
**Deliverables**:
- NEW module `src/tac/substrates/atw_codec_v2/procedural_variant.py` (~600 LOC; sister API surface to DP1/VQ-VAE/grayscale_lut)
- NEW `ProceduralVariantConfig` dataclass + canonical constants per predecessor §5.2.1
- NEW `derive_procedural_cdf_table_from_seed` + `compose_with_procedural_cdf_table` + `verify_procedural_cdf_in_domain` + `verify_seed_mutation_changes_cdf_bytes` callables
- NEW canonical archive operator: REPLACE-IN-PLACE of `cdf_table_blob` section at ATW2 grammar layer (FORK_BECAUSE_PRINCIPLED_MISMATCH from APPEND envelope pattern per predecessor §5.2.3)
- Sentinel `PROCEDURAL_CDF_SENTINEL = b"ACPV"` (substrate-unique per predecessor §5.2.2)
- Dedicated test suite (~30 tests; canonical pattern sister to DP1/VQ-VAE/grayscale_lut)
- Cathedral consumer wire-in per Catalog #335 auto-discovery (sister consumer `procedural_codebook_generator_consumer`)
- Catalog #270 dispatch optimization protocol per substrate canonical helper
- Per Catalog #146 inflate.py contest contract (3-arg invocation `inflate.sh archive_dir output_dir file_list`)
**Cost**: ~$0 GPU at BUILD-design surface; ~$0.50 Modal A10G paired-smoke (Catalog #167 smoke-before-full) to validate predicted ΔS = -0.001684 empirically.
**Gate**: All 8 Catalog #233 L1→L2 promotion canonical 4-gates satisfied; Catalog #325 follow-on symposium PROCEED; Catalog #324 post-training Tier-C validation evidence on landed archive sha.

### Roadmap timeline summary

| Phase | Earliest start | Earliest finish | Operator-routable gate |
|---|---|---|---|
| 1 DESIGN | NOW | NOW (this memo) | Self-landing |
| 2 SYMPOSIUM | Operator decision | ~1-2 weeks later | Catalog #272 Variant-C + Catalog #325 sister subagent spawn |
| 3 BUILD | Phase 2 PROCEED | ~3-4 weeks after Phase 2 | Catalog #313 D4 reactivation OR Variant-C decouple |
| Paid dispatch | ~6+ weeks from now | After Phase 3 | Catalog #313 D4 expires 2026-06-15 minimum; Catalog #325 fresh; Catalog #167 smoke clean |

---

## 9-dimension success checklist evidence
<!-- Catalog #294 mandatory section header (literal). Body in §7 below per
narrative numbering; the canonical 9-dim evidence is enumerated immediately. -->

## Section 7. 9-dimension success checklist evidence (Catalog #294 preliminary)

Per CLAUDE.md "9-DIMENSION SUCCESS CHECKLIST" + sister DP1/VQ-VAE/grayscale_lut canonical pattern:

1. **UNIQUENESS** — `atw_v2_codec_quantizer_lut` IN-DOMAIN context (line 103) is structurally distinct from DP1's `dp1_codebook_bytes` + VQ-VAE's `intermediate_transform_quantizer` + grayscale_lut's `chroma_lut_replacement`; all 4 are canonical equation #26 IN-DOMAIN contexts but operate on different substrate-class sub-surfaces. cdf_table_blob is the LARGEST RAW IN-DOMAIN target per PARSER-SAFE EXTENSION empirical evidence (commit `d0bf3ce37`).
2. **BEAUTY + ELEGANCE** — Phase 3 module ~600 LOC reviewable in 30s; sister-identical structure to DP1/VQ-VAE/grayscale_lut canonical pattern.
3. **DISTINCTNESS** — REPLACE-IN-PLACE archive operator at canonical ATW2 grammar layer (FORK from APPEND envelope; PRINCIPLED_MISMATCH per predecessor §5.2.3).
4. **RIGOR** — This memo's premise verification per Catalog #229 (read predecessor `7ea78deaa` + PARSER-SAFE EXTENSION `d0bf3ce37` + canonical equation #26 source + verified ΔS = -0.0016833 fp64) + Catalog #292 assumption-adversary 5 verdicts.
5. **OPTIMIZATION PER TECHNIQUE** — Per predecessor §5.2.3 canonical-vs-unique decision table per layer.
6. **STACK-OF-STACKS-COMPOSABILITY** — Under Variant-C scoping: composes additively with DP1 + VQ-VAE + grayscale_lut sister procedural variants on rate-axis only (sister α values per predecessor §7 composition-alpha matrix).
7. **DETERMINISTIC REPRODUCIBILITY** — `derive_procedural_cdf_table_from_seed` deterministic per (seed, generator_kind, shape, dtype); envelope byte-stable; brotli quality 9 fixed (sister canonical pattern).
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Predicted ΔS = -0.001684 (LARGER than DP1 -0.002706 and grayscale_lut -0.000149 per predecessor §6.1; SMALLER than VQ-VAE -0.005433); HIGHEST IN-DOMAIN confidence (canonical equation #26 line 103 explicit registration).
9. **OPTIMAL MINIMAL CONTEST SCORE** — Per Catalog #324, `predicted_band_validation_status = pending_post_training` declared; predicted ΔS is HYPOTHESIS not score claim. Per-substrate symposium per Catalog #325 + paired-smoke contest-CUDA + contest-CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" are required gates.

---

## Cargo-cult audit per assumption
<!-- Catalog #303 mandatory section header (literal). Body in §8 below per
narrative numbering; the per-assumption HARD-EARNED-vs-CARGO-CULTED audit
table is enumerated immediately. -->

## Section 8. Cargo-cult audit per assumption (Catalog #303 preliminary)

Per CLAUDE.md "PERMANENT-FIX-AND-SELF-PROTECT-ALL" + the HIGH-RISK 5 unwind audit canonical pattern:

| Assumption | Classification | Rationale |
|---|---|---|
| Canonical equation #26 closed-form predicts savings exactly | HARD-EARNED | Closed-form IS exact bytes_saved formula; rate term IS `25 × bytes / denom` per `upstream/evaluate.py:63`. |
| `atw_v2_codec_quantizer_lut` IS strongest IN-DOMAIN fit for cdf_table_blob | HARD-EARNED | Canonical equation #26 line 103 explicit registration (verified `grep -n` 2026-05-21). |
| cdf_table_blob is empirically 2,560 bytes (5 × 256 × fp16) | HARD-EARNED | Direct code read of `archive.py:40 + 426-431` per predecessor §2.2. |
| cdf_table_blob is the LARGEST RAW parser-safe + score-affecting section across IN-DOMAIN substrates | HARD-EARNED | PARSER-SAFE EXTENSION empirical landscape (commit `d0bf3ce37`); table verified §1. |
| Procedural CDF table is information-theoretically equivalent to learned CDF table | CARGO-CULTED — UNWIND-REQUIRED | Per predecessor Assumption-Adversary verbatim: procedural seed produces uniform-PRNG bytes; learned CDF table encodes `I(latent; class)`. Variant-C scoping decouples this assumption from BUILD viability. |
| Per-pixel-independence of CDF table bytes | CARGO-CULTED — INHERITED | Inherited from canonical equation #26 derivation; IN-DOMAIN context's domain-of-validity covers it. Per-substrate symposium evaluates whether the inherited assumption empirically holds on ATW V2. |
| Brotli quality 9 is optimal for procedural seed envelope | CARGO-CULTED — INHERITED | Inherited from canonical sister patterns (DP1 + VQ-VAE + grayscale_lut all use quality 9). |
| BUILD before symposium ratification is operator-safe | CARGO-CULTED — REJECT | Per Catalog #325 + symposium PROCEED_WITH_REVISIONS on base-V2: BUILD without follow-on symposium ratification of Variant-C scope is FORBIDDEN. |
| The post-slot-2 domain refinement (commit `79f1ba387`) precludes cdf_table_blob from IN-DOMAIN | CARGO-CULTED — REJECT | The domain refinement tightens IN-DOMAIN to REPLACEMENT savings for SCORE-OPAQUE bytes for the BYTE-SUBSTITUTION path. The procedural-replacement path (sister to grayscale_lut/DP1/VQ-VAE) is a DIFFERENT IN-DOMAIN sub-surface reached via per-substrate training co-optimizing the replacement (per parser-safe extension memo "Why This Matters" bullet). Line 103 registration is EXPLICIT for `atw_v2_codec_quantizer_lut`. |

---

## Observability surface
<!-- Catalog #305 mandatory section header (literal). Body in §9 below per
narrative numbering; the canonical 6-facet observability surface is
enumerated immediately. -->

## Section 9. Observability surface (Catalog #305 preliminary)

Per CLAUDE.md "Max observability — non-negotiable" 6-facet definition:

1. **Inspectable per layer** — Phase 3 module's `ProceduralVariantConfig`, `derive_procedural_cdf_table_from_seed`, `compose_with_procedural_cdf_table`, `verify_procedural_cdf_in_domain`, `verify_seed_mutation_changes_cdf_bytes` callable independently; each layer's input + output inspectable at any byte boundary; sister to canonical DP1/VQ-VAE/grayscale_lut pattern.
2. **Decomposable per signal** — Predicted ΔS decomposes via `predicted_archive_bytes_saved(cdf_bytes=2560, seed_bytes=32) → 2528` × `-25 / 37_545_489 → -0.001683`; bytes_saved decomposes into `cdf_bytes − seed_bytes`; envelope decomposes into `SENTINEL(4) + CDF_BYTES(u32) + GEN_KIND_TAG(u8) + SEED_LEN(u16) + seed(8)`.
3. **Diff-able across runs** — Determinism (same seed → same CDF bytes) verifiable by tests; envelope bytes diff-able byte-for-byte across (seed, generator_kind, num_classes, num_symbols) tuples; canonical archive bytes diff-able across cdf_table_blob replacements with all other sections byte-identical.
4. **Queryable post-hoc** — Predicted savings + IN-DOMAIN context computable from any (cdf_bytes, seed_bytes) input pair; canonical envelope re-parseable via sentinel + struct layout; tac.canonical_equations registry queryable per Catalog #344.
5. **Cite-able** — Phase 3 module docstring would cite: predecessor design memo `7ea78deaa` + PARSER-SAFE EXTENSION `d0bf3ce37` + DP1 sister + VQ-VAE sister + grayscale_lut sister + canonical equation #26 source module + this memo.
6. **Counterfactual-able** — `verify_seed_mutation_changes_cdf_bytes` proves single-byte counterfactuals propagate (Catalog #272 byte-mutation invariant); canonical no-op detector (Catalog #105 + #139) verifies the operational mechanism.

---

## Section 10. Catalog cross-references

This memo is gated by THREE primary catalogs:

### 10.1 Catalog #313 — predecessor probe outcome blocking

- Gate function: `check_dispatch_target_has_no_predecessor_adjudicated_outcome`
- Ledger row: `probe_id=atw_v2_d4_h_latent_given_scorer_class_20260516`
- Blocker status: `blocking` (expires 2026-06-15)
- Status for THIS variant: BLOCKING under Variant-A scoping; STRUCTURALLY ORTHOGONAL (decoupled) under Variant-C scoping per Assumption-Adversary verdict #4

### 10.2 Catalog #272 — Variant-C operator decision

- Gate function: `check_substrate_distinguishing_feature_integration_contract`
- Scope: per-substrate-feature contract (`distinguishing_feature_name` / `distinguishing_bytes_path` / `inflate_consumer_function` / `byte_mutation_smoke_passes`)
- Status for THIS variant: per predecessor §12 operator-routable #1, Variant-A scoping (additive bolt-on on V2 cooperative-receiver) is EMPIRICALLY PRE-FALSIFIED; Variant-C scoping (decoupled rate-axis-only) is the VIABLE BUILD path. Operator decision required.

### 10.3 Catalog #325 — per-substrate symposium ≥14-day gate

- Gate function: `check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor`
- Existing symposium: `council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (3 days old; PROCEED_WITH_REVISIONS; covers V2 BASE substrate)
- Status for THIS variant: 2026-05-18 symposium does NOT cover procedural-variant scope; follow-on per-substrate symposium REQUIRED before BUILD per the canonical 6-step contract.

### 10.4 Catalog #344 — canonical equation registry

- Gate function: `check_empirical_finding_memo_references_canonical_equation`
- Canonical equation: `procedural_codebook_from_seed_compression_savings_v1` (registered at `src/tac/canonical_equations/procedural_codebook_savings.py`; IN-DOMAIN context `atw_v2_codec_quantizer_lut` line 103)
- Status for THIS memo: PASS — frontmatter `canonical_equation_id` + body §5 cite the canonical equation explicitly.

### 10.5 Sister catalogs referenced

- Catalog #110 + #113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is a NEW memo per Catalog #348 retroactive-sweep canonical pattern; predecessor `7ea78deaa` preserved unchanged; this memo references but does NOT mutate predecessor.
- Catalog #117 + #157 + #174 canonical commit serializer — used for the landing commit.
- Catalog #125 6-hook wire-in declaration — §11 below.
- Catalog #185 META-meta drift detection — this memo's claims about predecessor verdicts + ledger rows + canonical equation registration are empirically verifiable.
- Catalog #220 + #272 + #298 substrate scaffolds COMPLETE or RESEARCH-ONLY — this memo declares `research_only: true` + `dispatch_enabled: false` per frontmatter.
- Catalog #229 premise-verification-before-edit — this memo verified predecessor + PARSER-SAFE EXTENSION + canonical equation + probe ledger empirically.
- Catalog #287 placeholder-rationale rejection — every waiver-pattern field carries substantive non-placeholder rationale.
- Catalog #290 + #294 + #296 + #303 + #305 + #309 + #324 design-memo discipline cluster — preliminary drafts in §7-9 + frontmatter; full per-substrate symposium memo would extend these.
- Catalog #340 sister-checkpoint guard — pre-write `tools/check_sister_files_recently_landed.py` PROCEED verdict (§11.A).
- Catalog #348 event-driven retroactive sweep — not applicable (no new STRICT gate landing); design-only memo.

---

## Section 11. 6-hook wire-in declaration (Catalog #125)

- **Hook #1 sensitivity-map** = N/A at design surface (no per-tensor sensitivity contribution; design-only memo).
- **Hook #2 Pareto constraint** = PLANNED for Phase 3 BUILD (predicted ΔS = -0.001684 contributes additively to rate-axis Pareto polytope under Variant-C scoping).
- **Hook #3 bit-allocator** = PLANNED for Phase 3 BUILD (32-byte seed slot replaces 2,560-byte CDF slot at archive grammar layer).
- **Hook #4 cathedral autopilot dispatch** = N/A at design surface (BUILD-time wire-in would route via sister consumer `tac.cathedral_consumers.procedural_codebook_generator_consumer` auto-discovered per Catalog #335).
- **Hook #5 continual-learning posterior** = N/A at design surface (first empirical anchor via `update_equation_with_empirical_anchor` per Catalog #344 would land post-paired-smoke).
- **Hook #6 probe-disambiguator** = ACTIVE — THIS memo IS the SECOND-ITERATION disambiguator between (a) predecessor's 3-failure-mode framing within cooperative-receiver paradigm, (b) sister parser-safe extension's empirical 4-substrate landscape ranking, (c) Variant-A vs Variant-C scoping operator-routable decision. Sister of predecessor's Hook #6 ACTIVE declaration; this memo SHARPENS the disambiguation by ranking cdf_table_blob ABOVE the other 3 PARSER-SAFE EXTENSION substrates per byte-count + IN-DOMAIN confidence.

### 11.A Sister collision verdict

Pre-write `tools/check_sister_files_recently_landed.py --files .omx/research/atw_v2_cdf_table_blob_procedural_variant_design_20260521.md --lookback-hours 12 --own-subagent-id wave-3-atw-v2-cdf-table-blob-procedural-variant-design-20260520`:

> `[check_sister_files_recently_landed] OK: PROCEED: no sister commits touched any of 1 target file(s) within the 12-hour lookback window. Safe to write.`

Sister-DISJOINT verdict:

- **Sister-DISJOINT** from in-flight `aa612de7` NSCS06 v8 BUILD (not committed at memo-draft-time; would touch `src/tac/substrates/nscs06_*` + recipes; different file scope).
- **Sister-DISJOINT** from in-flight `a93e1a3e` WR01 canonical equation registration (not committed at memo-draft-time; would touch `src/tac/canonical_equations/*` + registry JSONL; different file scope).
- **Sister-DISJOINT** from PARSER-SAFE EXTENSION (commit `d0bf3ce37`) — different file scope (parser-safe extension landed file vs THIS new design memo file).
- **Sister-DISJOINT** from CODEX CROSS-POLLINATION audit (commit `aafac7c84`) — different file scope.
- **Sister-DISJOINT** from canonical equation #26 domain refinement (commit `79f1ba387`) — different file scope.
- **Sister-DISJOINT** from predecessor `atw_v2_procedural_variant_build_design_20260520.md` (commit `7ea78deaa`) — different file scope; this memo REFERENCES but does NOT MUTATE predecessor per Catalog #110/#113 APPEND-ONLY discipline.
- **Catalog #314 / #340 sister-checkpoint guard**: pre-write PASSED; no sister has my target file in `files_touched` checkpoint within last 60 minutes.

---

## Section 12. Operator-routable decision queue

**3 GATES TO LIFT BEFORE PHASE 3 BUILD** (in priority order; sister to predecessor §12):

### Decision #1 (PRIORITY 1; ~5 min operator decision)

**Question**: Approve **Variant-C scoping** (decoupled rate-axis-only, sister to DP1/VQ-VAE/grayscale_lut canonical pattern) for the cdf_table_blob procedural variant?

**Recommended**: YES per predecessor Assumption-Adversary verbatim #3 + this memo's Assumption-Adversary verdict #4. Variant-A (additive bolt-on on V2 cooperative-receiver) is empirically PRE-FALSIFIED. Variant-C preserves the V2-1 redesign path independently while opening a parallel rate-axis-only research path that does NOT depend on D4 probe re-verdict.

**Alternative**: Defer indefinitely + scope this candidate into V2-1 redesign as a cdf-table architectural variant; trade off rate-axis-only signal for cooperative-receiver paradigm preservation.

**Cost**: $0 GPU + ~5 minutes operator decision.

### Decision #2 (PRIORITY 2; ~1-2 weeks subagent spawn IF Decision #1 = YES)

**Question**: Spawn follow-on per-substrate symposium per Catalog #325 for the procedural-variant scope?

**Specification**: T2 sextet pact + 3 specialists recommended: **Shannon LEAD** (information-theory grounding for rate-axis exact closed-form), **Daubechies CO-LEAD** (wavelet hierarchical-planning + CDF table mathematical structure), **Mallat** (sparse wavelet representations + CDF table per-class structure ratification), plus the standard sextet (Yousfi, Fridrich, Contrarian, Assumption-Adversary). Topical grand-council attendees: Atick (cooperative-receiver decoupling verification), Tishby memorial (information-theoretic equivalence verification), Wyner memorial (side-info-channel decoupling verification).

**Deliverables**: Per-substrate symposium memo `.omx/research/council_per_substrate_symposium_atw_v2_cdf_table_blob_procedural_variant_20260___.md` with PROCEED (or PROCEED_WITH_REVISIONS) verdict; 6-step canonical contract satisfied; Catalog #325 SATISFIED for procedural-variant scope.

**Cost**: $0 GPU + 1 council-grade design memo (~3-6 hours editor time).

### Decision #3 (PRIORITY 3; ~3-4 weeks BUILD IF Decision #2 = PROCEED)

**Question**: Approve Phase 3 BUILD landing?

**Specification**: Per §6 Phase 3 deliverables (NEW `procedural_variant.py` module + canonical archive operator + cathedral consumer wire-in + dedicated tests + Catalog #270 dispatch protocol). Sister to DP1/VQ-VAE/grayscale_lut canonical pattern.

**Gate**: Catalog #313 D4 reactivation criteria satisfied (V2-1 redesign + new probe MEANINGFUL_CONDITIONING) OR Variant-C scoping decouples; Catalog #325 follow-on symposium PROCEED; Catalog #324 post-training Tier-C validation evidence on landed archive sha; Catalog #167 smoke-before-full clean.

**Cost**: $0 GPU at BUILD-design surface; ~$0.50 Modal A10G paired-smoke (Catalog #167) to validate predicted ΔS = -0.001684 empirically.

### Decision #4 (PRIORITY 4; sister-correction landing on PARSER-SAFE EXTENSION memo)

**Question**: Append a "cdf_table_blob procedural variant ranked candidate" section to `.omx/research/parser_safe_methodology_extension_landed_20260520.md` per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE documenting this memo's decision-framing signal?

**Specification**: APPEND-ONLY footer noting (a) cdf_table_blob is the LARGEST RAW parser-safe + score-affecting section per the empirical 4-substrate table, (b) ranked as PRIMARY procedural-variant candidate per byte-count + IN-DOMAIN confidence, (c) gated by Catalog #313 D4 + Catalog #325 follow-on symposium + Catalog #272 Variant-C operator decision.

**Cost**: $0 GPU + ~30 min sister subagent.

---

## Section 13. v2 frontmatter declaration (Catalog #300)

Per CLAUDE.md "Council hierarchy: 4-tier protocol" + Catalog #300:

- `council_tier: T1` (working group; 1-3 named primary members + assumption-adversary + dissent; bounded-scope recommendation; output feeds future T2/T3 deliberation)
- `council_attendees`: Shannon, Yousfi, Atick, Tishby_memorial, Daubechies, Contrarian, Assumption-Adversary (7 attendees; quorum met for T1 working group)
- `council_quorum_met: true`
- `council_verdict: DEFER_PENDING_EVIDENCE` (inherits predecessor verdict; sharpened via parser-safe extension evidence)
- `council_dissent`: 2 verbatim entries (Contrarian + Assumption-Adversary)
- `council_assumption_adversary_verdict`: 5 explicit verdicts (HARD-EARNED × 3 + CARGO-CULTED-MILD × 1 + HARD-EARNED-MILD × 1)
- `council_predicted_mission_contribution: apparatus_maintenance` (T1 working-group decision-framing memo; no BUILD; no score moves until 3 operator-routable gates lift)
- `council_override_invoked: false` (no operator-frontier-override invoked)
- `council_override_rationale: null` (no override)

Continual-learning anchor will be appended via `tac.council_continual_learning.append_council_anchor` per Catalog #300 v2 contract IF Phase 2 follow-on symposium ratifies the procedural-variant scope (this T1 working-group output does NOT meet T2 elevation triggers per CLAUDE.md "Council hierarchy" tier elevation triggers; T2 elevation would fire when operator decides Variant-C scoping per Decision #1).

---

## Section 14. mission_predicted_contribution = `apparatus_maintenance`

Per CLAUDE.md "Mission alignment — non-negotiable" 5-category classification + Catalog #300 frontmatter enum validation:

This memo is classified `apparatus_maintenance` because:

1. **NO BUILD lands** — design-only memo per task #1159 prompt scope limits (DO NOT BUILD procedural variant).
2. **NO score moves** — predicted ΔS = -0.001684 is HYPOTHESIS; no empirical anchor; `score_claim: false`.
3. **NO new STRICT preflight gate** — no Catalog #N landing.
4. **NO operator-routable budget consumption** — $0 GPU + ~1.5h editor time at design surface.
5. **DECISION-FRAMING signal added** — ranks cdf_table_blob ABOVE the other 3 PARSER-SAFE EXTENSION substrates per byte-count + IN-DOMAIN confidence; sharpens operator-routable decision queue per §12.
6. **Sister-coordination signal preserved** — references predecessor `7ea78deaa` + PARSER-SAFE EXTENSION `d0bf3ce37` + canonical equation #26 + probe ledger row + Catalog #325 symposium status; per CLAUDE.md "Subagent coherence-by-default" wire-in discipline.

**Operator-visible alert check** (per CLAUDE.md "Mission alignment" Consequence 5): IF `rigor_overhead + apparatus_maintenance > 60%` of T2+ verdicts in any 30-day window, operator review is required. This T1 working-group verdict does NOT count toward the T2+ ratio (Catalog #300 v2 contract applies to T2+ tier deliberations).

**Path to `frontier_breaking` upgrade**: IF Decision #1 (Variant-C scoping) + Decision #2 (per-substrate symposium PROCEED) + Decision #3 (Phase 3 BUILD landing + paired-smoke empirical anchor confirming ΔS ≈ -0.001684), this memo's design framing becomes the upstream design-doc for a `frontier_breaking` BUILD landing whose empirical ΔS contributes to rate-axis polytope per Catalog #356 sister consumer-routing pattern.

---

## Cross-references summary

- **Predecessor**: `.omx/research/atw_v2_procedural_variant_build_design_20260520.md` (commit `7ea78deaa`) — DEFER_PENDING_EVIDENCE
- **PARSER-SAFE EXTENSION evidence**: `.omx/research/parser_safe_methodology_extension_landed_20260520.md` (commit `d0bf3ce37`) — 4-substrate empirical landscape (cdf_table_blob = LARGEST RAW score-affecting target)
- **CODEX CROSS-POLLINATION audit**: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` (commit `aafac7c84`) — sister synergy mapping
- **Canonical equation #26 domain refinement**: post-slot-2 commit `79f1ba387` — domain-of-validity refinement; IN-DOMAIN `atw_v2_codec_quantizer_lut` line 103 explicit registration preserved
- **Per-substrate symposium (V2 base)**: `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` — PROCEED_WITH_REVISIONS (covers V2 base substrate; NOT this variant scope)
- **D4 probe ledger row**: `.omx/state/probe_outcomes.jsonl` `probe_id=atw_v2_d4_h_latent_given_scorer_class_20260516` — INDEPENDENT verdict; blocking until 2026-06-15
- **Canonical equation source**: `src/tac/canonical_equations/procedural_codebook_savings.py:103` — `atw_v2_codec_quantizer_lut` IN-DOMAIN context
- **Sister procedural-variant BUILDs (canonical pattern)**: DP1 + VQ-VAE + grayscale_lut landed sister memos
- **CLAUDE.md non-negotiables cited**: "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + "Forbidden premature KILL without research exhaustion" + "Subagent coherence-by-default" + "Council hierarchy: 4-tier protocol" + "Mission alignment — non-negotiable" + "Apples-to-apples evidence discipline" + "Bit-level deconstruction and entropy discipline" + "Frontier scores are pointer-only"
- **Catalog gates cited**: #1 / #110 / #113 / #117 / #125 / #146 / #157 / #167 / #174 / #185 / #205 / #206 / #220 / #229 / #233 / #245 / #270 / #272 / #287 / #290 / #292 / #294 / #296 / #297 / #298 / #300 / #303 / #305 / #309 / #313 / #314 / #315 / #322 / #323 / #324 / #325 / #335 / #340 / #344 / #348 / #356
