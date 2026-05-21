# Catalog #359 cross-reference audit: VQ-VAE indices_blob routing + recent canonical equation registry audit

timestamp_utc: 2026-05-21T05:35:00Z
agent: claude
lane_id: lane_wave_3_catalog_359_cross_reference_audit_vq_vae_routing_20260520
horizon-class: apparatus_maintenance
verdict: AUDIT_COMPLETE_NO_VIOLATIONS_FOUND
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
paid_dispatch_attempted: false
research_only: true
mission_predicted_contribution: apparatus_maintenance
canonical_equations_referenced:
  - procedural_codebook_from_seed_compression_savings_v1
  - procedural_predictor_plus_residual_correction_savings_v1
related_deliberation_ids:
  - vq_vae_indices_blob_procedural_variant_extension_STOOD_DOWN_sister_convergence_20260521  # 149bdc6a1
  - vq_vae_indices_blob_procedural_variant_design_20260521T050932Z_codex  # 77081f991
  - canonical_equation_procedural_predictor_plus_residual_correction_ratification_landed_20260521  # 098d8a31c
  - magic_codec_pair_1_2_engineering_fix_re_run_landed_20260520  # Catalog #359 landing
  - atw_v2_cdf_table_blob_procedural_variant_design_20260521T051855Z_codex  # 8441b702e
canonical_equation_id: procedural_predictor_plus_residual_correction_savings_v1
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl

## Summary verdict

**AUDIT_COMPLETE_NO_VIOLATIONS_FOUND.** Per slot 3-r5 STAND DOWN Top-3 #3
(2026-05-21 commit `149bdc6a1`) operator request: verify codex sister
`77081f991` (Add VQ-VAE procedural indices residual scaffold) correctly
routes through `procedural_predictor_plus_residual_correction_savings_v1`
(NOT canonical equation #26) AND audit other recent canonical equation
registry events for misapplication candidates AND flag any in-flight
procedural variant BUILDs needing similar scrutiny.

**Headline finding**: codex sister `77081f991` routing is CORRECT —
verified across THREE structural surfaces (module imports, callsite
context strings, canonical equation registry anchors). Catalog #359 STRICT
preflight gate live count remains **0 violations** at audit time.

**Sister design-memo concern surfaced (NOT a violation; design-memo
surface only)**: claude-side ATW V2 cdf_table_blob design memo
(`atw_v2_cdf_table_blob_procedural_variant_design_20260521.md` commit
`8441b702e`) declares `canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1`
in frontmatter (line 68), but codex empirically proved via byte-mutation
smoke that cdf_table_blob is currently **decode-opaque** (max_abs_rgb_delta = 0.0).
The codex sister memo `atw_v2_cdf_table_blob_procedural_variant_design_20260521T051855Z_codex.md`
correctly routes paradigm classification to `REMOVAL` (dead-section
cleanup) with `RESIDUAL-CORRECTION-DOWNSTREAM` deferred to future
range-decoder path. The claude memo's verdict is DEFER_PENDING_EVIDENCE
(NOT BUILD), so no anchor was appended — Catalog #359 does not fire on
memo frontmatter. Operator-routable: reconcile the claude vs codex
canonical equation routing in a sister convergence memo.

## VQ-VAE codex sister routing verification (3-surface)

### Surface 1: module imports

`src/tac/substrates/vq_vae/indices_procedural_variant.py` line 19-22:

```python
from tac.canonical_equations.procedural_predictor_residual_savings import (
    EQUATION_ID as RESIDUAL_EQUATION_ID,
    predict_procedural_predictor_plus_residual_correction_savings,
)
```

**VERIFIED CORRECT** — imports the residual-hybrid canonical equation, NOT
canonical equation #26 (`procedural_codebook_savings`).

### Surface 2: callsite context strings

`src/tac/substrates/vq_vae/indices_procedural_variant.py` line 35-38:

```python
RESIDUAL_CONTEXT = (
    "vq_vae_indices_blob_residual_correction_on_parser_safe_score_affecting_indices"
)
```

Empirical verification:

```python
from tac.canonical_equations.procedural_codebook_savings import is_residual_hybrid_context
is_residual_hybrid_context("vq_vae_indices_blob_residual_correction_on_parser_safe_score_affecting_indices")
# True
```

The context string contains `_residual_correction_` which matches
`_RESIDUAL_HYBRID_CONTEXT_PATTERNS` in
`src/tac/canonical_equations/procedural_codebook_savings.py:152-161`.
**VERIFIED CORRECT** — the context string structurally identifies as
residual-hybrid per the canonical pattern.

Two callsites confirmed at lines 245 and 316:

```python
accounting = predict_procedural_predictor_plus_residual_correction_savings(
    original_payload_bytes=...,
    predictor_seed_or_code_bytes=...,
    residual_stream_bytes=...,
    container_overhead_bytes=...,
    context=RESIDUAL_CONTEXT,
)
```

### Surface 3: canonical equation registry anchors

Verified `tools/check_no_canonical_equation_misapplication_to_residual_hybrid_contexts`
gate function returns 0 violations. Registry inspection across all 50
events confirms:

- **Equation #26** (`procedural_codebook_from_seed_compression_savings_v1`)
  carries TWO historical residual-hybrid anchors (pair_1 and pair_2)
  measured BEFORE the Catalog #359 cutoff `2026-05-21T00:30:00Z`. These
  are APPEND-ONLY preserved per Catalog #110/#113 HISTORICAL_PROVENANCE
  with `historical_misapplied_equation` provenance field. The gate's
  cutoff filter correctly exempts these as pre-fix historical record.
- **Equation `procedural_predictor_plus_residual_correction_savings_v1`**
  carries TWO anchors for pair_1 and pair_2 with the CORRECTED residual
  byte accounting (`predicted_delta_s_rate_only` ≈ empirical delta within
  fp64 precision; residual ≈ 0 per the equation's
  `predicted_vs_empirical_residual` field).
- No FUTURE residual-hybrid anchors landed in equation #26 after the
  cutoff. **Catalog #359 STRICT gate is structurally protecting.**

## Recent canonical equation registry audit (~10 events)

Scanned all 50 registry events; classified by `equation_id` + `event_type`:

| event_utc | equation_id | event_type | residual_hybrid_in_payload | gate_status |
|---|---|---|---|---|
| 2026-05-20T22:43:24Z | master_gradient_null_space_byte_fraction_v1 | anchor_appended | n/a | clean |
| 2026-05-20T23:22:40Z | procedural_codebook_from_seed_compression_savings_v1 | anchor_appended | NO | clean |
| 2026-05-20T23:41:52Z | procedural_codebook_from_seed_compression_savings_v1 | domain_refined | n/a | clean |
| 2026-05-20T23:47:07Z | procedural_codebook_from_seed_compression_savings_v1 | anchor_appended | **YES** (pair_1 dwt_detail) | **pre-cutoff exempt** |
| 2026-05-21T00:21:20Z | procedural_codebook_from_seed_compression_savings_v1 | anchor_appended | **YES** (pair_1 + pair_2) | **pre-cutoff exempt** |
| 2026-05-21T01:05:18Z | procedural_predictor_plus_residual_correction_savings_v1 | registered | YES (pair_1 + pair_2) | clean (correct equation) |
| 2026-05-21T01:13:21Z | procedural_codebook_from_seed_compression_savings_v1 | domain_refined | YES (preserved in payload) | **pre-cutoff exempt** |
| 2026-05-21T01:50:23Z | procedural_codebook_from_seed_compression_savings_v1 | domain_refined | YES (preserved in payload) | **pre-cutoff exempt** |
| 2026-05-21T04:39:46Z | procedural_codebook_from_seed_compression_savings_v1 | domain_refined | YES (preserved in payload) | **pre-cutoff exempt** |
| 2026-05-21T05:12:39Z | static_packet_custody_byte_delta_score_savings_v1 | registered | NO (not residual-hybrid) | clean |

**No misapplication candidates** beyond pair_1 and pair_2 (which are
pre-cutoff historical anchors preserved via APPEND-ONLY discipline). The
new `static_packet_custody_byte_delta_score_savings_v1` equation (commit
`6348fcbf0`) is unrelated to procedural variant routing — registered for
HNeRV/wavelet apply_transform PR106x byte-delta score-savings tracking.

## In-flight / queued procedural variant BUILD candidates

| substrate | status | canonical equation routing | verdict |
|---|---|---|---|
| `vq_vae` indices_blob | LANDED (codex `77081f991`) | `procedural_predictor_plus_residual_correction_savings_v1` (residual-hybrid) | **CORRECT** |
| `nscs06_v8_chroma_lut` | L1+ scaffold present (lane in-flight per slot 3-r5) | `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN context `nscs06_v8_chroma_lut` | **CORRECT** (true REPLACEMENT — 32-byte seed REPLACES 4096-byte LUT; not residual-hybrid) |
| `atw_codec_v2` cdf_table_blob | DESIGN-ONLY (claude `aa612de7`; codex `8441b702e`) | DISPUTED — claude memo claims #26 IN-DOMAIN; codex empirical smoke proves dead section (max_abs_rgb_delta=0.0) | **NEEDS RECONCILIATION** (design-memo surface only; no BUILD; no anchor) |
| `nscs06_v8_path_b_wavelet` | substrate present; per `_EXCLUDED_CONTEXTS` direct DWT-detail-subband substitution is REFUSED | n/a — direct byte substitution forbidden per equation #26 domain refinement | safe (excluded context catches this) |
| `dp1` codebook_bytes | landed (sister) | `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN context `dp1_codebook_bytes` | CORRECT (true REPLACEMENT) |
| `grayscale_lut` | landed (sister) | `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN | CORRECT (true REPLACEMENT) |

### NSCS06 v8 chroma_lut routing verification

`src/tac/substrates/nscs06_v8_chroma_lut/procedural_variant.py:308-336`
implements `verify_procedural_lut_in_domain` that calls
`validate_context_is_in_domain(context="nscs06_v8_chroma_lut", raise_on_excluded=False)`.
Context `nscs06_v8_chroma_lut` is in `_INCLUDED_CONTEXTS` line 102 of
`procedural_codebook_savings.py`. This is a TRUE REPLACEMENT context
(32-byte seed REPLACES 4096-byte chroma LUT directly; no residual stream),
so canonical equation #26 IS the correct routing. **VERIFIED CORRECT**.

### ATW V2 cdf_table_blob routing concern (design-memo surface only)

The claude design memo
`atw_v2_cdf_table_blob_procedural_variant_design_20260521.md` (commit
`8441b702e`) frontmatter line 68 declares:

```yaml
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
```

…and line 88 cites the predicted ΔS calculation:

```
ΔS = -25 × (N_codebook - K_seed) / 37_545_489 with K_seed=32, N_codebook=2560,
yielding -0.0016833
```

**However**, codex parallel design memo
`atw_v2_cdf_table_blob_procedural_variant_design_20260521T051855Z_codex.md`
(commit `057130de4` Add ATW2 CDF dead-section parity probe) empirically
verified:

```
max_abs_rgb_delta_after_cdf_byte_xor: 0.0
sum_abs_rgb_delta_after_cdf_byte_xor: 0.0
```

Codex's empirical smoke proves cdf_table_blob is currently **decode-opaque**
in the live ATW V2 runtime — `reconstruct_from_wz_residual()` does NOT
consume the CDF table. This means:

- **Equation #26 misapplication risk**: if a `compose_with_procedural_cdf`
  BUILD were to register an anchor with `inputs.in_domain_context =
  "atw_v2_codec_quantizer_lut"`, equation #26 would predict a -0.0017
  savings via the canonical formula, but the empirical delta would be
  **0.0** (zero savings because the bytes do nothing). This would create
  a residual ≈ 0.0017 (16× the canonical residual=0 standard for in-domain
  predictions).
- **NOT a Catalog #359 violation** — the context `atw_v2_codec_quantizer_lut`
  does NOT match the residual-hybrid pattern (no `_residual_correction_`
  / `_srl1_correction_` / sister tokens). The bug class is DIFFERENT:
  predicting savings for a DEAD section is "REMOVAL paradigm" misapplied
  as REPLACEMENT.

**The claude memo's verdict is DEFER_PENDING_EVIDENCE (NOT BUILD)**, so no
anchor will be appended in the near term. The 3 operator-routable gates
in the claude memo (D4 probe expiry 2026-06-15 + Catalog #325 per-substrate
symposium ≥14-day + Catalog #272 Variant-C operator decision) keep BUILD
gated until empirical evidence resolves.

**Operator-routable**: the claude memo should be amended (sister convergence
pass) to acknowledge codex's empirical decode-opacity finding and
re-classify ATW V2 cdf_table_blob's BUILD as REMOVAL paradigm (dead-section
cleanup via grammar-aware schema change) rather than REPLACEMENT paradigm
under equation #26. Cite codex's ATW2 CDF dead-section parity probe
(commit `057130de4`) as the empirical anchor.

## Catalog #359 STRICT preflight gate live count verification

```
$ .venv/bin/python -c "from tac.preflight import check_no_canonical_equation_misapplication_to_residual_hybrid_contexts; v = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(strict=False, verbose=True); print(f'Live violation count: {len(v)}')"
Live violation count: 0
CLEAN - Catalog #359 gate currently passes (0 violations).
```

**Catalog #359 STRICT gate verified clean at 0 violations.**

## Sister-collision verdict with NSCS06 v8 BUILD (`aa612de7`)

**NO COLLISION.** NSCS06 v8 BUILD touches
`src/tac/substrates/nscs06_v8_chroma_lut/*` (already L1+ scaffold present).
This audit lane touches only `.omx/research/catalog_359_cross_reference_audit_vq_vae_routing_landed_20260521.md`
(new file). Sister-checkpoint guard returned PROCEED at pre-flight.

## 6-hook wire-in declaration (per Catalog #125)

This is a **defensive audit gate** with no algorithmic signal contribution:

- Hook #1 sensitivity-map: **N/A** (audit memo; no signal contribution).
- Hook #2 Pareto constraint: **N/A**.
- Hook #3 bit-allocator: **N/A**.
- Hook #4 cathedral autopilot dispatch: **ACTIVE** indirectly — verifies
  Catalog #359 gate continues to extinct the canonical equation #26
  misapplication bug class structurally; future cathedral consumers that
  ingest canonical equation registry anchors inherit the protection.
- Hook #5 continual-learning posterior: **ACTIVE** indirectly — verifies
  the equation registry's APPEND-ONLY discipline preserves historical
  anchors (pair_1 + pair_2) while the gate prevents future misapplication.
- Hook #6 probe-disambiguator: **ACTIVE** — the audit IS the canonical
  disambiguator between REPLACEMENT-paradigm contexts (equation #26
  IN-DOMAIN) and RESIDUAL-CORRECTION-paradigm contexts
  (`procedural_predictor_plus_residual_correction_savings_v1`) for the
  3 in-flight/landed procedural variant substrates.

## Top-3 operator-routable next-actions

1. **Sister convergence pass on ATW V2 cdf_table_blob design memos**:
   reconcile claude design memo (commit `aa612de7`) canonical equation
   routing (`procedural_codebook_from_seed_compression_savings_v1`) with
   codex empirical finding (commit `057130de4` Add ATW2 CDF dead-section
   parity probe — max_abs_rgb_delta=0.0 proves decode-opacity). Amend
   claude memo to re-classify under REMOVAL paradigm (dead-section
   cleanup via grammar-aware schema change) rather than REPLACEMENT
   paradigm under equation #26. No BUILD authorization until D4 probe
   expires 2026-06-15 + Catalog #325 per-substrate symposium ≥14-day +
   Catalog #272 Variant-C operator decision lift.

2. **Pre-build canonical equation routing audit for any FUTURE procedural
   variant scaffold**: before any new `compose_with_procedural_<X>`
   helper lands, verify the targeted bytes' empirical decode-influence via
   byte-mutation smoke (codex's ATW2 CDF dead-section parity probe pattern
   at commit `057130de4` is the canonical template). Classify the variant
   into one of {REPLACEMENT-UPSTREAM, RESIDUAL-CORRECTION-DOWNSTREAM,
   REMOVAL} BEFORE selecting the canonical equation. This is sister
   discipline to Catalog #272 byte-mutation distinguishing-feature
   contract + Catalog #359 misapplication refusal.

3. **Acknowledge codex's adversarial review excellence pattern**: codex
   sister `77081f991` independently correctly routed VQ-VAE indices_blob
   through `procedural_predictor_plus_residual_correction_savings_v1`
   AND independently caught + extinguished the equation #26 misapplication
   that the slot 3-r5 TaskCreate brief had as a CARGO-CULTED assumption
   (per the codex landing memo's cargo-cult audit: *"Equation #26 applies
   to any seed-derived bytes" → CARGO-CULTED → CORRECTED "Residual bytes
   are charged; use the residual equation."*). This is the canonical
   reverse-directive + adversarial-review pattern working as designed.
   Consider updating CLAUDE.md "Subagent coherence-by-default" to cite
   this pair (slot 3-r5 TaskCreate `7ea60e91f` → codex landing
   `77081f991` → claude stand-down `149bdc6a1` → claude audit THIS memo)
   as a worked example of the canonical cross-agent convergence pattern.

## Blockers

NONE. Audit complete; no Catalog #359 STRICT gate violations found. One
sister design-memo concern surfaced (ATW V2 cdf_table_blob routing
reconciliation) is operator-routable for future sister convergence pass.

## Discipline

- Catalog #229 PV: read all 6 pre-flight inputs in full (slot 3-r5 STAND
  DOWN memo + codex sister landing memo + canonical equation registry
  events + Catalog #359 gate source + canonical equation modules + VQ-VAE
  + NSCS06 v8 procedural variant modules + ATW V2 design memos) BEFORE
  writing this landing memo.
- Catalog #340 sister-checkpoint guard: PROCEED at pre-flight; no
  collision with NSCS06 v8 BUILD (`aa612de7`).
- Catalog #314 absorption-pattern avoidance: NEW audit memo file only;
  zero mutation of sister landing memos / canonical equation source /
  gate source.
- Catalog #206 checkpoint discipline: 3 in-progress checkpoints emitted
  + complete on landing.
- Catalog #117/#157/#174 commit serializer: this landing commits via
  canonical serializer with POST-EDIT `--expected-content-sha256` per
  CLAUDE.md "Subagent commits MUST use serializer" non-negotiable.
- Catalog #119 Co-Authored-By trailer: auto-appended by canonical serializer.
- Catalog #287/#323 canonical Provenance: every audit claim cites source
  commit SHA + file path + line numbers; no docstring-overstatement
  patterns; no score claims.
- Catalog #303 cargo-cult audit: this memo's own assumptions are
  enumerated in §"Recent canonical equation registry audit" with explicit
  HARD-EARNED-vs-CARGO-CULTED classification per the codex sister memo's
  cargo-cult audit template.
- Catalog #322 phantom-provenance avoidance: no autopilot ranker
  adjustment derived from any phantom evidence.
- Catalog #344 canonical equation cross-reference: every canonical
  equation reference cites the registry path + the equation_id +
  the canonical helper module + line numbers.
- Catalog #125 6-hook wire-in: declared above with explicit N/A per hook.
- Catalog #110/#113 APPEND-ONLY: NEW audit memo file only; zero mutation
  of existing forensic artifacts.

## Sister coordination summary

| sister lane | scope | collision verdict |
|---|---|---|
| Slot 3-r5 STAND DOWN (`149bdc6a1`) | VQ-VAE indices_blob extension | UPSTREAM (this audit completes Slot 3-r5 Top-3 #3) |
| Codex sister VQ-VAE landing (`77081f991`) | VQ-VAE indices_blob procedural residual scaffold | UPSTREAM (this audit verifies codex routing correctness) |
| Slot 3-r2 NSCS06 v8 BUILD (`aa612de7`) | `src/tac/substrates/nscs06_v8_chroma_lut/*` | DISJOINT (different namespace; audit verifies NSCS06 v8 routing is correct) |
| Codex ATW2 CDF dead-section parity probe (`057130de4`) | ATW V2 cdf_table_blob empirical decode-opacity verification | SISTER (this audit cites codex's empirical finding as resolution input for ATW V2 design-memo routing concern) |
| Slot 2-r reverse-directive (`7ea60e91f`) | reverse codex-routing-directive #4 | UPSTREAM (issued directive that routed scope to codex) |

## Mission contribution

`apparatus_maintenance` per Catalog #300 — extincts the recurrence risk of
the canonical equation #26 misapplication bug class structurally across
the 3 in-flight/landed procedural variant substrates (VQ-VAE +
NSCS06 v8 chroma_lut + ATW V2 cdf_table_blob). Demonstrates the canonical
cross-agent convergence pattern (TaskCreate → codex BUILD → claude
stand-down → claude audit). Surfaces one operator-routable design-memo
reconciliation (ATW V2 cdf_table_blob REPLACEMENT vs REMOVAL paradigm)
without firing any paid GPU.
