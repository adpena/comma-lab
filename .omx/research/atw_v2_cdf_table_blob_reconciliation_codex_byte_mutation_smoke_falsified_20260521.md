<!-- SPDX-License-Identifier: MIT -->
---
substrate_id: atw_codec_v2_cdf_table_blob_procedural_variant
substrate_aliases: ["atw_v2_cdf_proc", "atw_codec_v2"]
reconciliation_class: codex_byte_mutation_smoke_empirically_falsifies_claude_design_memo_replacement_paradigm_routing
horizon_class: frontier_pursuit
council_tier: T1
council_attendees:
  - Shannon
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "This is the 7th codex sister convergence pattern today (slot 3-r5 STAND DOWN at commit 88af2b2c5 + slot 3-r6 cross-reference audit at commit a4ad7027b + this reconciliation pass). The audit memo (commit a4ad7027b) ALREADY surfaced the cdf_table_blob reconciliation concern under §'ATW V2 cdf_table_blob routing concern (design-memo surface only)' and tagged it operator-routable for future sister convergence. This reconciliation memo is the convergence — it does NOT advance score; it does NOT spawn a BUILD; it does NOT mutate the predecessor design memo (Catalog #110/#113 APPEND-ONLY discipline). It DOES make the implementation-level falsification structurally visible per Catalog #307 paradigm-vs-implementation classification + Catalog #344 canonical equation cross-reference. Tagged apparatus_maintenance per Catalog #300; the structural value is preventing a future Phase 3 BUILD from registering an empirical anchor that would land predicted_delta_s = -0.001684 vs empirical_delta_s = 0.0 = residual 0.0017 (16x the canonical residual=0 standard for in-domain predictions per the audit memo §174-176 verbatim)."
  - member: Assumption-Adversary
    verbatim: "Per Catalog #292 per-deliberation assumption surfacing. The SHARED ASSUMPTION operating across the predecessor design memo (commit 8441b702e): 'cdf_table_blob is parser-safe AND score-affecting AND canonical equation #26 IN-DOMAIN context atw_v2_codec_quantizer_lut applies for REPLACEMENT savings prediction.' Classification: PARTIALLY-CARGO-CULTED. The HARD-EARNED part: cdf_table_blob IS parser-safe (verified via codex's analyze_atw2_cdf_section + the predecessor memo §1 grammar audit). The IS-SCORE-AFFECTING part WAS HARD-EARNED at the PARSER-SAFE EXTENSION evidence anchor commit d0bf3ce37 (static classification + decoder_side_information typing). The CARGO-CULTED part: the assumption that 'score-affecting (per static classification) IMPLIES decode-influential (per empirical byte-mutation smoke)' was a TRANSITIVE LEAP — static parser-safe-plus-score-affecting classification CANNOT substitute for empirical byte-mutation proof of decode influence. The codex byte-mutation smoke (commit 057130de4) is the canonical empirical disambiguator + it returned max_abs_raw_byte_delta=0 across all 2,560 cdf_table_blob bytes. This is the canonical IMPLEMENTATION-LEVEL falsification per Catalog #307: the PARADIGM (procedural-substitution of decoder side-information via canonical equation #26) is INTACT; the IMPLEMENTATION (assuming cdf_table_blob bytes have decode influence) is FALSIFIED. The reclassification per §4 routes the empirical bytes through REMOVAL paradigm (Catalog #110/#113 APPEND-ONLY for the predecessor design memo + this APPEND reconciliation per HISTORICAL_PROVENANCE)."
  - member: Shannon
    verbatim: "Per CLAUDE.md 'Meta-Lagrangian/Pareto solver - NON-NEGOTIABLE' + canonical Lagrangian discipline. The audit memo's residual calculation (predicted -0.0017 savings via canonical equation #26 IN-DOMAIN context atw_v2_codec_quantizer_lut vs empirical 0.0 savings per codex byte-mutation smoke) IS 16x the canonical residual=0 standard for IN-DOMAIN predictions per the post-slot-2 domain refinement (commit 79f1ba387). A future Phase 3 BUILD that registers an empirical anchor with this 16x residual would TRIGGER the canonical equation #26 recalibration loop per Catalog #344 + sister update_equation_with_empirical_anchor. The structural value of this reconciliation memo is to PREVENT that misregistration: Catalog #359 (which catches residual-hybrid context misapplication) does NOT catch THIS misclassification because atw_v2_codec_quantizer_lut is NOT a residual-hybrid context (no _residual_correction_ / _srl1_correction_ token); the bug class is DIFFERENT (predicting savings for a DEAD section is 'REMOVAL paradigm misapplied as REPLACEMENT paradigm'). The proposed Catalog #344 operator-decision to ADD a NEW EXCLUDED context 'direct_byte_substitution_on_decode_opaque_raw_sections' to canonical equation #26's _EXCLUDED_CONTEXTS would extinct THIS bug class structurally."

council_assumption_adversary_verdict:
  - assumption: "cdf_table_blob is canonical equation #26 IN-DOMAIN context atw_v2_codec_quantizer_lut for REPLACEMENT paradigm prediction"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Per codex byte-mutation smoke commit 057130de4 test_cdf_table_xor_preserves_current_inflate_raw_output: proof.raw_equal is True AND proof.max_abs_raw_byte_delta == 0 across all 2,560 cdf_table_blob bytes mutated. The current ATW V2 runtime's reconstruct_from_wz_residual() does NOT consume the CDF table; the bytes are DECODE-OPAQUE despite being parser-visible. The canonical equation #26 IN-DOMAIN registration (line 103 atw_v2_codec_quantizer_lut) was registered under the implicit assumption that cdf_table_blob bytes have decode influence; that assumption is FALSIFIED. Per Catalog #307 paradigm-vs-implementation: this is IMPLEMENTATION-LEVEL (a specific bytes-have-decode-influence assumption falsified), NOT PARADIGM-LEVEL (procedural-substitution of true decoder side-information via canonical equation #26 remains a valid pursuit for substrates whose decoder DOES consume the substituted bytes — e.g. NSCS06 v8 chroma_lut per the audit memo §158 VERIFIED CORRECT routing)."
  - assumption: "Predicted ΔS = -0.001684 (matrix-memo-arithmetic per canonical equation #26 closed-form rate-axis prediction) is empirically realizable via cdf_table_blob procedural substitution under Variant-C scoping"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Per audit memo §174-176 verbatim: 'if a compose_with_procedural_cdf BUILD were to register an anchor with inputs.in_domain_context = atw_v2_codec_quantizer_lut, equation #26 would predict a -0.0017 savings via the canonical formula, but the empirical delta would be 0.0 (zero savings because the bytes do nothing).' The canonical equation #26 closed-form ΔS = -25 × (2560 - 32) / 37,545,489 = -0.0016833 PRESUMES the bytes contribute to inflate output; codex's empirical smoke proves they do NOT. The 16x canonical residual (0.0017 vs 0.0) would trigger canonical equation #26 recalibration if registered as an empirical anchor — which Catalog #344 + Catalog #185 + Catalog #176 + Catalog #185 sister discipline would surface. The reclassification per §4 routes the empirical bytes through REMOVAL paradigm (per parser-safe-extension precedent commit d0bf3ce37 where bytes are removed entirely from the archive at the grammar layer rather than substituted via procedural seed)."
  - assumption: "REMOVAL paradigm is the correct routing for decode-opaque parser-visible bytes (vs REPLACEMENT paradigm via canonical equation #26)"
    classification: HARD-EARNED
    rationale: "Per the PARSER-SAFE EXTENSION precedent (commit d0bf3ce37 + commit 79f1ba387 sister) + sister memo parser_safe_methodology_extension_landed_20260520.md: parser-safe-but-DECODE-OPAQUE bytes are NOT canonical equation #26 IN-DOMAIN candidates for direct byte substitution; the equation predicts REPLACEMENT savings for bytes the decoder CONSUMES. For DECODE-OPAQUE bytes, the right surface is grammar-layer REMOVAL (delete the section entirely from the archive grammar; recover the full byte budget; no procedural seed envelope needed). For ATW V2 cdf_table_blob: under current runtime, removing the section saves 2,560 bytes (vs canonical equation #26 procedural-substitution which saves 2,528 bytes). For a future runtime where reconstruct_from_wz_residual() ACTUALLY consumes the CDF table (e.g. real range-decoder implementation), the classification flips back to REPLACEMENT paradigm + the codex byte-mutation smoke would FAIL (max_abs_raw_byte_delta > 0) and the design-memo routing would be re-validated."

predicted_band_validation_status: empirically_falsified_via_codex_byte_mutation_smoke
predicted_band:
  predecessor_claude_design_memo_replacement_paradigm:
    rate_axis_delta_s: [-0.001683, -0.001684]
    net_score_delta_s: [-0.001683, -0.001684]
    validation_status: PROSPECTIVE_ANCHOR_NEVER_REGISTERED_PER_CATALOG_110_113_HISTORICAL_PROVENANCE_PREDECESSOR_DESIGN_MEMO_PRESERVED
    confidence: "PROSPECTIVE only — predecessor design memo verdict DEFER_PENDING_EVIDENCE; no anchor in canonical equations registry; this reconciliation memo prevents the prospective registration via implementation-level falsification per Catalog #307"
  reclassified_removal_paradigm:
    rate_axis_delta_s: [-0.001705, -0.001705]
    net_score_delta_s: [-0.001705, -0.001705]
    calculation: "-25 × 2560 / 37_545_489 = -0.0017045... (full 2,560 byte removal; no envelope overhead)"
    validation_status: pending_post_training_or_post_grammar_change_per_catalog_324
    confidence: "STRUCTURALLY EXACT closed-form rate term IF cdf_table_blob section is REMOVED from ATW V2 archive grammar AND inflate.py is updated to omit the parsing step (sister to grayscale_lut's GLV1-no-chroma-section grammar pattern). NOT VALID under current ATW V2 runtime because parse_atw2_archive_bytes() expects the section — grammar-layer change required."
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
operator_directive: "WAVE-3-ATW-V2-CDF-TABLE-BLOB-RECONCILIATION per operator blanket-approved + must-keep-queuing-overnight + slot 3-r6 audit (commit a4ad7027b) operator-routable future-sister-convergence-pass; sister-DISJOINT from in-flight NSCS06 v8 BUILD (aa612de7); APPEND reconciliation only per Catalog #110/#113 (NO mutation of predecessor 8441b702e OR codex 057130de4)"
related_deliberation_ids:
  - atw_v2_cdf_table_blob_procedural_variant_design_20260521  # predecessor commit 8441b702e
  - codex_atw2_cdf_dead_section_parity_probe_20260521  # codex empirical anchor commit 057130de4
  - catalog_359_cross_reference_audit_vq_vae_routing_20260521  # audit anchor commit a4ad7027b
  - parser_safe_methodology_extension_landed_20260520  # PARSER-SAFE EXTENSION evidence anchor commit d0bf3ce37
  - canonical_equation_26_parser_safe_domain_refinement_20260520  # commit 79f1ba387
  - atw_codec_v2_d4_probe_verdict_20260516_codex  # Catalog #313 D4 INDEPENDENT verdict
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl
predecessor_design_memo: .omx/research/atw_v2_cdf_table_blob_procedural_variant_design_20260521.md
codex_empirical_anchor_commit: 057130de4
codex_empirical_anchor_test_name: test_cdf_table_xor_preserves_current_inflate_raw_output
codex_empirical_anchor_file: src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py
codex_empirical_anchor_helper_file: src/tac/substrates/atw_codec_v2/cdf_dead_section.py
codex_empirical_anchor_cli: tools/probe_atw2_cdf_dead_section.py
empirical_anchor_max_abs_raw_byte_delta: 0
empirical_anchor_mutated_byte_count: 2560
empirical_anchor_raw_equal: true
audit_cross_reference_section: "audit memo §172-225 'ATW V2 cdf_table_blob routing concern (design-memo surface only)'"
---

<!-- Catalog #344 canonical-equations-registry cross-reference: this
reconciliation memo's primary claim — that the predecessor design memo
(commit 8441b702e) routes cdf_table_blob procedural substitution through
canonical equation #26 IN-DOMAIN context `atw_v2_codec_quantizer_lut` for
predicted ΔS = -0.001684 BUT the codex byte-mutation smoke (commit
057130de4) empirically proves max_abs_raw_byte_delta=0 across all 2,560
cdf_table_blob bytes mutated — IS a structural falsification of that
specific routing's bytes-have-decode-influence assumption. Per Catalog
#307 paradigm-vs-implementation: IMPLEMENTATION-LEVEL falsification
(the specific bytes-have-decode-influence assumption is falsified); NOT
PARADIGM-LEVEL refutation of canonical equation #26 procedural-
substitution paradigm (sister NSCS06 v8 chroma_lut routing remains
correct per audit memo §158 VERIFIED CORRECT). Per Catalog #344
canonical equation #26 operator-decision protocol: §5 proposes a NEW
EXCLUDED context `direct_byte_substitution_on_decode_opaque_raw_sections`
to extinct THIS bug class structurally. -->

# WAVE-3-ATW-V2-CDF-TABLE-BLOB-RECONCILIATION-CODEX-BYTE-MUTATION-SMOKE-FALSIFIED (2026-05-21)

**Lane**: `lane_wave_3_atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521`
**Subagent**: `wave-3-atw-v2-cdf-table-blob-reconciliation-20260520`
**Predecessor design memo**: `atw_v2_cdf_table_blob_procedural_variant_design_20260521.md` (commit `8441b702e`) DEFER_PENDING_EVIDENCE; declared canonical equation #26 IN-DOMAIN REPLACEMENT routing with predicted ΔS = -0.001684
**Codex empirical anchor**: commit `057130de4` "Add ATW2 CDF dead-section parity probe" — proves `max_abs_raw_byte_delta=0` across all 2,560 cdf_table_blob bytes mutated
**Audit cross-reference**: `catalog_359_cross_reference_audit_vq_vae_routing_landed_20260521.md` (commit `a4ad7027b`) §172-225 surfaced this concern operator-routable
**Sister convergence count today**: 7 (slot 3-r5 STAND DOWN at commit `88af2b2c5` + slot 3-r6 cross-reference audit at commit `a4ad7027b` + this reconciliation pass)
**Council verdict**: **PROCEED** (T1 reconciliation memo per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE; predecessor design memo + codex empirical smoke BOTH preserved byte-for-byte; this APPEND makes the implementation-level falsification structurally visible)
**Mission contribution**: `apparatus_maintenance` (extincts a future Phase 3 BUILD's prospective canonical equation #26 misregistration risk; structural protection for cathedral autopilot ranker consumers)

---

## §1. Falsification anchor: codex empirical byte-mutation smoke commit `057130de4`

The codex sister commit `057130de4` ("Add ATW2 CDF dead-section parity probe") lands a research-only helper + test + CLI that empirically probes whether cdf_table_blob bytes influence ATW V2 inflate output. The canonical test asserts:

```python
# src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py
def test_cdf_table_xor_preserves_current_inflate_raw_output() -> None:
    archive = _make_archive()
    proof = prove_atw2_cdf_decode_influence(archive, device="cpu")
    assert proof.raw_equal is True
    assert proof.max_abs_raw_byte_delta == 0
    assert proof.mutated_byte_count == 2560
    assert proof.source_archive_sha256 != proof.mutated_archive_sha256
    assert proof.source_raw_sha256 == proof.mutated_raw_sha256
    assert proof.score_claim is False
```

The test passes empirically. The canonical interpretation per the helper module's own docstring:

> "These helpers are research-only. They test the current ATW2 runtime fact
> that ``cdf_table_blob`` is parsed and copied into the model but is not
> consumed by ``reconstruct_from_wz_residual``. Any future range-decoder
> implementation that uses the table should make these probes fail, forcing
> the lane to move from byte-only cleanup to residual-correction or
> co-trained replacement accounting."

**Empirical receipts** (per `Atw2CdfDecodeInfluenceProof` dataclass + test assertions):

| Quantity | Value | Implication |
|---|---|---|
| `proof.raw_equal` | `True` | Inflate output bytes IDENTICAL after CDF mutation |
| `proof.max_abs_raw_byte_delta` | `0` | ZERO bytes changed across entire inflate output |
| `proof.mutated_byte_count` | `2560` | All 5 × 256 × fp16 = 2,560 cdf_table_blob bytes mutated |
| `proof.source_archive_sha256 != proof.mutated_archive_sha256` | True | Archive bytes DO differ (mutation succeeded at the archive layer) |
| `proof.source_raw_sha256 == proof.mutated_raw_sha256` | True | Inflate output bytes IDENTICAL (cdf_table_blob is decode-opaque) |
| `proof.score_claim` | `False` | Per CLAUDE.md "Forbidden score claims" + canonical Provenance per Catalog #323 |

**Conclusion**: cdf_table_blob is empirically **DECODE-OPAQUE** in the current ATW V2 runtime. The bytes are parsed (per `parse_atw2_archive_bytes`) and stored on the model (per `model.cdf_table`), but `reconstruct_from_wz_residual()` does NOT consume the table for inflate output.

---

## §2. Predecessor claude design memo `8441b702e` REPLACEMENT framing

The predecessor design memo (commit `8441b702e`; landed 2026-05-21T00:21:10 -0500 ≅ 2026-05-21T05:21:10 UTC) declared:

**Frontmatter line 68** (verbatim):
```yaml
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
```

**Section 4 (verbatim §4.3 "Score impact under Variant-C scoping")**:
> "Rate-axis: `ΔS_rate = -25 × 2,528 / 37,545,489 = -0.0016833 (≅ -0.001684 per matrix-memo-arithmetic; ≅ -0.001683 per predecessor §1)`
> Seg-axis: structurally ZERO under Variant-C scoping (cooperative-receiver hypothesis is DECOUPLED; the procedural CDF table operates as a deterministic rate-axis substitution; the seg signal does NOT depend on the trained CDF table's per-class structure)
> Pose-axis: structurally ZERO under Variant-C scoping (same reasoning)
> **Net ΔS = -0.001684 per closed-form Variant-C prediction**"

**Section 4.1 (verbatim)**:
> "Encode-time: Operator picks a deterministic seed (`uint64` per canonical equation #26 K_seed=32 envelope) + `generator_kind="pcg64"` (sister to DP1/VQ-VAE/grayscale_lut canonical pattern).
> Archive-build-time: Replace the trained `cdf_table_blob` (2,560 fp16 bytes) IN-PLACE at the canonical ATW2 grammar layer with a 32-byte envelope... Inflate-time...: inflate.py reads the 32-byte envelope, dispatches to `derive_procedural_cdf_table_from_seed(seed, num_classes=5, num_symbols=256, generator_kind="pcg64")` which deterministically produces a `(5, 256)` fp32 tensor; downstream latent-reconstruction loop consumes the derived table identically to the trained-fp16-decoded table."

**The predecessor's PRESUMED bytes-have-decode-influence assumption** is structurally embedded in §4.1's claim that "downstream latent-reconstruction loop consumes the derived table identically to the trained-fp16-decoded table" — but the codex byte-mutation smoke (§1) proves the loop does NOT consume the table at all in the current runtime. The trained CDF table is parsed and stored, but never read during reconstruction. Therefore "identical consumption" with a procedural-derived table is structurally a NO-OP at decode time; the rate-axis savings prediction is structurally indistinguishable from removing the section entirely (REMOVAL paradigm) under the current runtime.

**Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: the predecessor design memo's frontmatter, body text, sections, and verdict (DEFER_PENDING_EVIDENCE) are PRESERVED byte-for-byte. This reconciliation memo is an APPEND-only sister artifact that surfaces the implementation-level falsification without mutating the predecessor.

---

## §3. CASCADE COMPRESSION symposium discipline + Catalog #307 paradigm-vs-implementation classification

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 (`check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification`):

**This is IMPLEMENTATION-LEVEL falsification, NOT paradigm-level kill.**

| Classification axis | Verdict |
|---|---|
| Paradigm (canonical equation #26 procedural-substitution of decoder side-information via deterministic seed) | **INTACT** — sister NSCS06 v8 chroma_lut routing (audit memo §158) remains VERIFIED CORRECT; TT5L transformer tokens routing remains correct; DP1 codebook bytes routing remains correct; grayscale_lut chroma_lut_replacement routing remains correct |
| Implementation (predecessor design memo `8441b702e`'s specific assumption that ATW V2 cdf_table_blob bytes have decode influence in the current runtime) | **FALSIFIED** — codex byte-mutation smoke `057130de4` proves max_abs_raw_byte_delta=0 |
| Reactivation path (per CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS") | **DEFERRED-PENDING-NEW-RUNTIME-EVIDENCE** — if future range-decoder implementation makes `reconstruct_from_wz_residual()` actually consume the CDF table (per codex helper module's docstring), the bytes-have-decode-influence assumption flips back to TRUE + the predecessor design memo's REPLACEMENT routing becomes VALID again; the codex byte-mutation smoke would FAIL (`max_abs_raw_byte_delta > 0`) and serve as the canonical disambiguator for the routing flip |
| Sister Catalog #307 verdict template usage | `# RATIFY-FALSIFICATION-OF-THE-SPECIFIC-IMPLEMENTATION (predecessor design memo bytes-have-decode-influence assumption FALSIFIED via codex byte-mutation smoke) + REQUEST-REINVESTIGATION-OF-ALTERNATIVES (REMOVAL paradigm reclassification per §4; future range-decoder runtime reactivation per codex helper docstring)` |

**Per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable**: this reconciliation memo's verdict line uses `DEFERRED-PENDING-NEW-RUNTIME-EVIDENCE` (NOT KILL). The structural value is preventing a future Phase 3 BUILD from registering a canonical equation #26 empirical anchor with predicted_delta_s=-0.001684 vs empirical_delta_s=0.0 = 16x canonical residual (per audit memo §174-176 verbatim) which would corrupt the canonical equation #26 recalibration loop per Catalog #344 + sister `update_equation_with_empirical_anchor`.

---

## §4. Reclassification: cdf_table_blob → REMOVAL paradigm candidate

Per the PARSER-SAFE EXTENSION precedent (commit `d0bf3ce37` + sister memo `parser_safe_methodology_extension_landed_20260520.md`):

**Three structurally distinct paradigms for parser-visible bytes**:

| Paradigm | Routing | Bytes accounting | Valid when |
|---|---|---|---|
| REPLACEMENT (canonical equation #26 IN-DOMAIN) | Procedural seed substitution at archive-build time; inflate-time deterministic re-derivation | Save (N - K_seed) bytes; e.g. 2,528 for ATW V2 cdf_table_blob | Bytes ARE consumed at decode time (decoder reads + uses them) |
| REMOVAL (grammar-layer section deletion) | Update archive grammar to omit the section entirely; update inflate parser to skip the parse step | Save N bytes (full section removal; no envelope overhead) | Bytes are DECODE-OPAQUE in current runtime (parser reads them but decoder never consumes them) |
| RESIDUAL-CORRECTION (canonical equation #26 EXCLUDED; sister equation `procedural_predictor_plus_residual_correction_savings_v1` pending) | Procedural seed predicts; residual stream encodes (empirical - predictor) delta | ADDS bytes via residual encoding; rate-axis savings only if predictor approximates empirical well | Bytes are consumed at decode time AND distributional mismatch between procedural-seed-derived bytes vs trained bytes is small enough that residual cost < replacement cost |

**ATW V2 cdf_table_blob (current runtime) verdict per the 3 paradigms**:

| Paradigm | Verdict | Rationale |
|---|---|---|
| REPLACEMENT (predecessor design memo's routing) | **EMPIRICALLY-FALSIFIED via codex byte-mutation smoke** | Decoder does NOT consume the bytes; substituting with procedural-derived bytes is a structural NO-OP at decode time; the routing's bytes-have-decode-influence assumption is structurally false in the current runtime |
| REMOVAL (this memo's reclassification) | **CANDIDATE-PENDING-GRAMMAR-CHANGE** | Save 2,560 bytes (full section removal; no envelope overhead) BY updating `src/tac/substrates/atw_codec_v2/archive.py` grammar to omit `cdf_table_blob` section + updating `parse_atw2_archive_bytes()` to skip the parse step + updating `pack_archive()` to omit the section emission + updating `inflate.py` to skip the section read |
| RESIDUAL-CORRECTION | **OUT-OF-SCOPE for current runtime** | Per Catalog #359 + the residual-hybrid context EXCLUDED set in canonical equation #26; only applicable if a future runtime makes the bytes decode-consumed AND the procedural predictor's distributional mismatch is small |

**REMOVAL paradigm rate-axis calculation** (sister to canonical equation #26 closed-form but with K_seed=0 because there is no envelope; the section is deleted entirely):

```
ΔS_rate_removal = -25 × N_removed / 37_545_489
                = -25 × 2560 / 37_545_489
                = -0.0017045...
                ≅ -0.001705
```

**REPLACEMENT vs REMOVAL byte budget delta**: REMOVAL saves 32 more bytes (2,560 vs 2,528) than REPLACEMENT because there is no envelope overhead. Predicted ΔS difference: `-25 × 32 / 37_545_489 ≅ -2.13e-5` (0.000021 score points) — empirically indistinguishable from REPLACEMENT under any single-archive measurement but structurally cleaner because the bytes are removed entirely rather than substituted with a no-op envelope.

**Sister parser-safe-extension precedent** for grammar-layer REMOVAL: the grayscale_lut substrate's GLV1 grammar has NO `chroma_lut` section (the chroma LUT is replaced via the GLPV envelope per canonical equation #26 chroma_lut_replacement context). For ATW V2 cdf_table_blob, the analogous grammar-layer change is to update the ATW2 grammar from V1 (with cdf_table_blob section) to a sister V1.1 / V2 (without cdf_table_blob section) — preserving all 6 other ATW2 sections byte-for-byte.

**Per Catalog #233 L1→L2 promotion canonical 4-gate**: a Phase 3 BUILD landing the REMOVAL paradigm reclassification would require: (1) smoke green (the codex byte-mutation smoke IS the smoke evidence; max_abs_raw_byte_delta=0 PROVES the removal is safe in the current runtime); (2) Tier C MDL density measured (sister to canonical equation #26 IN-DOMAIN density anchors; sister-routable); (3) 100ep auth-eval anchor with byte-deterministic archive (sister-routable); (4) Catalog #127 custody validated. Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #315 + Catalog #325 per-substrate symposium ≥14-day window: REMOVAL paradigm Phase 3 BUILD remains operator-routable.

---

## §5. Open question: should canonical equation #26 add NEW EXCLUDED context `direct_byte_substitution_on_decode_opaque_raw_sections`?

Per Catalog #344 canonical-equation-formalization protocol + the existing `_EXCLUDED_CONTEXTS` set in `src/tac/canonical_equations/procedural_codebook_savings.py:119-124`:

**Current EXCLUDED contexts** (verbatim from source):
```python
_EXCLUDED_CONTEXTS = (
    "direct_dwt_detail_subband_byte_substitution",
    "direct_byte_substitution_on_wavelet_decomposition_coefficients",
    "master_gradient_null_byte_removal_with_constant_reconstruction",
    "master_gradient_null_byte_replacement_with_arbitrary_constant",
)
```

**Proposed NEW EXCLUDED context** (operator-decision per Catalog #344):
```python
"direct_byte_substitution_on_decode_opaque_raw_sections"
```

**Definition** (proposed): Refuses canonical equation #26 REPLACEMENT savings prediction for any context where (a) the bytes are parser-visible (per static parser-safe extension classification), AND (b) the bytes are NOT consumed at decode time per an empirical byte-mutation smoke (sister to the codex `prove_atw2_cdf_decode_influence` canonical helper pattern). The empirical byte-mutation smoke is the canonical disambiguator: if `max_abs_raw_byte_delta == 0` across all mutated bytes, the section is decode-opaque + REPLACEMENT paradigm is structurally a no-op + the right routing is REMOVAL paradigm (grammar-layer section deletion) OR await a future runtime that consumes the bytes.

**Rationale for ADDING this EXCLUDED context** (operator-decision):

1. **Structural extinction of the META bug class**: per Catalog #344 + sister Catalog #185 + Catalog #176 META-meta discipline, ANY canonical equation #26 IN-DOMAIN registration for a context that is empirically decode-opaque produces a residual ≅ predicted_ΔS_magnitude / canonical_residual_standard (per audit memo §174-176 the 0.0017 residual for ATW V2 cdf_table_blob is 16x the canonical residual=0 standard). Adding the EXCLUDED context routes such contexts through the canonical refusal helper at construction time (sister to the existing `is_residual_hybrid_context()` + `refuse_residual_hybrid_context_misapplication()` helpers at line 164+) rather than discovering the misclassification empirically after the BUILD lands.

2. **Sister to PARSER-SAFE EXTENSION precedent**: the parser-safe extension methodology (commit `d0bf3ce37`) empirically established the static parser-safe classification across all 4 IN-DOMAIN substrates; the empirical byte-mutation smoke (codex commit `057130de4`) extends the methodology to runtime decode-opacity. The EXCLUDED context is the canonical equation #26 surface that consumes the runtime evidence.

3. **Backward-compatible**: existing IN-DOMAIN contexts (DP1 / VQ-VAE / grayscale_lut / NSCS06 v8 chroma_lut / TT5L / sister) are NOT decode-opaque in their respective runtimes (per the sister design memos' explicit decoder-consumption claims + sister codex empirical verifications); the new EXCLUDED context refuses only the specific NEW class of contexts that fail the empirical byte-mutation smoke. The ATW V2 `atw_v2_codec_quantizer_lut` IN-DOMAIN registration (line 103) remains valid for FUTURE runtimes where the CDF table IS decode-consumed (per codex helper docstring's "Any future range-decoder implementation that uses the table should make these probes fail" forward statement); the new EXCLUDED context catches the CURRENT-RUNTIME misclassification specifically.

**Operator-decision queue per Catalog #344**:
- (a) APPROVE-ADD: land the new EXCLUDED context in same commit batch as a sister wave; update `_EXCLUDED_CONTEXTS` set + `domain_of_validity_excluded` field in canonical equations registry + sister cathedral consumer auto-discovery wire-in per Catalog #335.
- (b) DEFER-PENDING-FURTHER-EVIDENCE: keep the existing EXCLUDED set; rely on the predecessor design memo's DEFER_PENDING_EVIDENCE verdict + this reconciliation memo to surface the routing concern to future sister convergence passes.
- (c) REJECT-NOT-APPLICABLE: argue that the empirical byte-mutation smoke is per-runtime + per-archive specific and cannot generalize as a canonical EXCLUDED context.

**This reconciliation memo's recommendation**: Option (a) APPROVE-ADD per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. The bug class (predicting canonical equation #26 REPLACEMENT savings for decode-opaque bytes) is structural + recurring + extincted by the proposed EXCLUDED context. Sister-routable to a follow-on subagent: `lane_canonical_equation_26_add_decode_opaque_excluded_context_20260521` with operator approval gate.

---

## §6. Cascade-coherence: 7th codex sister convergence pattern today

Per the audit memo (commit `a4ad7027b`) §306-308 verbatim:

> "Catalog #313 + #325 + #324 cascade-coherence-after-canonical-rewrite verified clean; sister design-memo concern surfaced (ATW V2 cdf_table_blob routing reconciliation) is operator-routable for future sister convergence pass."

**Today's 7 codex sister convergence patterns**:

| # | Event | Sister artifact | Catalog # / Verdict |
|---|---|---|---|
| 1 | NSCS06 v8 chroma_lut routing verification | audit memo §158 VERIFIED CORRECT | #344 + #359 cross-check |
| 2 | VQ-VAE indices_blob routing (codex `77081f991`) | audit memo §161-170 — routed through `procedural_predictor_plus_residual_correction_savings_v1`, NOT #26 | #359 (residual-hybrid scope correct) |
| 3 | Canonical equation #26 registry audit | audit memo (50 events scanned, 0 misapplication violations beyond pre-cutoff anchors) | #344 audit |
| 4 | Catalog #359 STRICT gate live count verification | audit memo verified live count = 0 | #359 + #185 |
| 5 | ATW V2 cdf_table_blob reconciliation concern surfaced (THIS memo's input signal) | audit memo §172-225 operator-routable for future sister convergence pass | THIS memo's trigger |
| 6 | Slot 3-r5 STAND DOWN (commit `88af2b2c5`) | sister codex memo recommended STAND DOWN on a related procedural-variant attempt; sister convergence acknowledged | per slot 3-r5 |
| 7 | **THIS reconciliation memo** (commit pending) | APPEND reconciliation per Catalog #110/#113; converges claude design memo + codex byte-mutation smoke + audit memo into structural reclassification | #307 + #344 + #359 + #185 |

**Cascade-coherence verdict**: the 7-pattern sequence demonstrates the canonical cascade-compression discipline at scale + the sister-DISJOINT-iteration pattern + the per-substrate canonical equation #26 routing-verification cycle. Per CLAUDE.md "Subagent coherence-by-default" non-negotiable: every sister subagent's primary deliverable (claude design memo + codex empirical smoke + audit memo + THIS reconciliation) lands on a DISTINCT artifact surface without overlap; the cathedral autopilot ranker per Catalog #335 + #336 + #337 auto-discovery consumes the converged signal across all 4 surfaces.

---

## §7. Top-3 operator-routable next-actions

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Catalog #229 premise-verification + Catalog #292 per-deliberation assumption surfacing:

1. **APPROVE-OR-DEFER Catalog #344 NEW EXCLUDED context proposal (§5)**: operator approves adding `direct_byte_substitution_on_decode_opaque_raw_sections` to canonical equation #26 `_EXCLUDED_CONTEXTS` + sister cathedral consumer auto-discovery wire-in per Catalog #335. If APPROVED, spawn `lane_canonical_equation_26_add_decode_opaque_excluded_context_20260521` sister subagent for the BUILD. Sister-routable; operator-decision per Catalog #344 + sister CLAUDE.md "Design decisions — non-negotiable".

2. **AUDIT-OR-RECLASSIFY ATW V2 base substrate Catalog #325 per-substrate symposium (2026-05-18 PROCEED_WITH_REVISIONS)**: per audit memo §172-225 + this reconciliation memo §4, the 2026-05-18 symposium covers V2 BASE substrate reactivation (cooperative-receiver hypothesis re-probe + V2-1 redesign); it does NOT cover the cdf_table_blob REMOVAL paradigm reclassification (this memo's §4). A follow-on per-substrate symposium per Catalog #325 IS REQUIRED before any cdf_table_blob REMOVAL BUILD lands (separate from the procedural-variant Phase 3 BUILD path per predecessor design memo §6 which is now structurally falsified per §3). Sister-routable; operator-decision.

3. **DEFER-OR-SPAWN sister REMOVAL paradigm Phase 3 BUILD design memo**: per §4 + Catalog #233 L1→L2 promotion canonical 4-gate, a Phase 3 BUILD landing the cdf_table_blob REMOVAL paradigm reclassification would update `src/tac/substrates/atw_codec_v2/archive.py` grammar (omit cdf_table_blob section) + `parse_atw2_archive_bytes()` (skip section parse) + `pack_archive()` (skip section emission) + `inflate.py` (skip section read) — saving 2,560 bytes for predicted ΔS ≅ -0.001705 (sister to canonical equation #26 closed-form but with K_seed=0 because there is no envelope). The Phase 3 BUILD is gated by: (a) Catalog #325 follow-on symposium per item 2 above, (b) Catalog #313 D4 INDEPENDENT verdict expiry 2026-06-15 (sister to predecessor design memo §2 which gated the procedural-variant under Variant-A scoping; REMOVAL paradigm IS DECOUPLED from D4 per the codex byte-mutation smoke's max_abs_raw_byte_delta=0 evidence — the bytes have ZERO information content for the scorer in the current runtime), (c) Catalog #272 Variant-C operator decision (REMOVAL paradigm is structurally simpler than REPLACEMENT paradigm; it bypasses the cooperative-receiver hypothesis entirely). Sister-routable; operator-decision.

---

## §8. v2 frontmatter per Catalog #300 (T1 reconciliation memo)

All required fields per Catalog #300 STRICT preflight gate present in frontmatter above:

- `council_tier: T1`
- `council_attendees: [Shannon, Contrarian, Assumption-Adversary]` (T1 reconciliation pattern; T2+ would add Yousfi + Atick + Tishby + sister; T1 sufficient per the structural-protection-only scope of this memo — no BUILD, no anchor registration, no dispatch)
- `council_quorum_met: true`
- `council_verdict: PROCEED` (the reconciliation itself; per CLAUDE.md "Forbidden premature KILL" — does NOT KILL the predecessor design memo; instead structurally reclassifies + preserves predecessor per Catalog #110/#113 APPEND-ONLY)
- `council_dissent` (Contrarian + Assumption-Adversary + Shannon verbatim per Catalog #292)
- `council_predicted_mission_contribution: apparatus_maintenance` (extincts a future Phase 3 BUILD's prospective canonical equation #26 misregistration risk; structural protection for cathedral autopilot ranker consumers)
- `council_override_invoked: false`
- `council_assumption_adversary_verdict` (3 assumptions classified: 2 CARGO-CULTED-EMPIRICALLY-FALSIFIED + 1 HARD-EARNED per Catalog #292 + the hard-earned-vs-cargo-culted addendum)
- `related_deliberation_ids` (6 sister anchors cite-chain per Catalog #292 cite-chain detection)

---

## §9. 6-hook wire-in declaration per Catalog #125

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125 mandatory wire-in declaration:

| Hook | Status | Wire-in surface |
|---|---|---|
| #1 sensitivity-map contribution | N/A | This reconciliation memo is a structural-protection artifact + decision-framing memo; it does NOT emit a sensitivity signal. Predecessor design memo §10 already declared sensitivity-map hook as `N/A — DESIGN-ONLY memo, no contribution to `tac.sensitivity_map.*`; sensitivity-map hook lands ONLY at Phase 3 BUILD`. THIS memo inherits the N/A status. |
| #2 Pareto constraint | N/A | No Pareto-relevant signal; predecessor design memo §10 declared N/A. THIS memo inherits. |
| #3 bit-allocator hook | N/A | No per-tensor importance change; predecessor design memo §10 declared N/A. THIS memo inherits. |
| #4 cathedral autopilot dispatch hook | **ACTIVE (READ-ONLY)** | The audit memo (commit `a4ad7027b`) + THIS reconciliation memo + the predecessor design memo are all CONSUMED by cathedral autopilot ranker via Catalog #335 + #336 + #337 auto-discovery + invocation. The cathedral autopilot's per-iteration ranker now has structural evidence that ATW V2 cdf_table_blob REPLACEMENT paradigm is implementation-level-falsified + REMOVAL paradigm reclassification is the operator-routable next-action. Per Catalog #341 canonical-routing markers: this is a READ-ONLY signal (no `predicted_delta_adjustment` mutation; observability-only per the canonical Tier A contract). |
| #5 continual-learning posterior update | **ACTIVE** | Per Catalog #300 v2 frontmatter + `tac.council_continual_learning.append_council_anchor` canonical helper invocation pattern: this T1 reconciliation memo's `council_verdict=PROCEED` + `council_assumption_adversary_verdict` are append-only signals into the canonical council deliberation posterior at `.omx/state/council_deliberation_posterior.jsonl` per Catalog #128 + #131 fcntl-locked JSONL discipline. Sister subagents querying the posterior via `query_anchors_by_topic('atw_v2_cdf_table_blob_reconciliation')` see the structural reclassification. |
| #6 probe-disambiguator | N/A | The canonical probe-disambiguator for THIS reconciliation IS the codex empirical byte-mutation smoke `tools/probe_atw2_cdf_dead_section.py` (commit `057130de4`); the disambiguator is sister-owned by codex per Catalog #110/#113 HISTORICAL_PROVENANCE. THIS memo does NOT spawn a NEW probe-disambiguator; it consumes the codex sister artifact + cites it as the canonical empirical anchor. |

---

## §10. Catalog cross-references (#110+#113 APPEND-ONLY / #307 / #344 / #359)

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable + sister catalog cite-chain:

- **Catalog #110 + #113 HISTORICAL_PROVENANCE APPEND-ONLY** (CRITICAL discipline observed by THIS memo): zero mutation of predecessor design memo `8441b702e` + zero mutation of codex byte-mutation smoke commit `057130de4` + zero mutation of audit memo `a4ad7027b` + zero mutation of canonical equations registry `.omx/state/canonical_equations_registry.jsonl` + zero mutation of sister memos. THIS memo is a NEW append-only artifact; the convergence is achieved via reading + citing + reclassifying, NOT via mutating predecessor surfaces.

- **Catalog #307** (`check_kill_verdict_distinguishes_paradigm_vs_implementation_falsification`): THIS memo's §3 explicitly classifies the falsification as IMPLEMENTATION-LEVEL (predecessor design memo's specific bytes-have-decode-influence assumption FALSIFIED) NOT PARADIGM-LEVEL (canonical equation #26 procedural-substitution paradigm remains valid for sister contexts NSCS06 v8 chroma_lut / DP1 / VQ-VAE / grayscale_lut / TT5L). The reactivation criterion is pinned (DEFERRED-PENDING-NEW-RUNTIME-EVIDENCE — if future range-decoder runtime makes CDF table decode-consumed, REPLACEMENT routing flips back to VALID).

- **Catalog #344** (`check_empirical_finding_memo_references_canonical_equation`): THIS memo references canonical equation `procedural_codebook_from_seed_compression_savings_v1` explicitly in frontmatter (`canonical_equation_id`) + body §2 + §4 + §5 + §10. Per Catalog #344 the empirical-finding-memo discipline is satisfied; the canonical-equation cross-reference is structural; the operator-decision proposal for adding NEW EXCLUDED context (§5) follows the canonical Catalog #344 protocol.

- **Catalog #359** (`check_no_canonical_equation_misapplication_to_residual_hybrid_contexts`): per audit memo §177-179 verbatim "**NOT a Catalog #359 violation** — the context `atw_v2_codec_quantizer_lut` does NOT match the residual-hybrid pattern (no `_residual_correction_` / `_srl1_correction_` / sister tokens). The bug class is DIFFERENT: predicting savings for a DEAD section is 'REMOVAL paradigm misapplied as REPLACEMENT paradigm'." — THIS memo confirms the audit's classification + proposes the NEW Catalog #344 EXCLUDED context as the structural extinction surface (§5).

- **Catalog #185** (`check_strict_flipped_catalog_entries_have_live_count_zero`): verified during preflight — Catalog #359 live count = 0 (no canonical equation #26 misapplication violations in registry). This reconciliation memo PRESERVES that clean state by preventing prospective registration of an ATW V2 cdf_table_blob anchor with the implementation-falsified routing.

- **Catalog #229** (premise verification before edit): every claim in THIS memo is grounded in a verified-via-read predecessor artifact (predecessor design memo `8441b702e` read in full + codex byte-mutation smoke `057130de4` read in full including test assertions + audit memo `a4ad7027b` read in full §172-225 + canonical equation #26 source `src/tac/canonical_equations/procedural_codebook_savings.py` read in full lines 90-161). No hypothetical claims; every assertion has a citable source.

- **Catalog #292** (per-deliberation assumption surfacing): 3 assumptions classified in `council_assumption_adversary_verdict` per the hard-earned-vs-cargo-culted addendum.

- **Catalog #300** (v2 frontmatter): all required fields present (§8 audit).

- **Catalog #125** (6-hook wire-in declaration): §9 declares all 6 hooks (1 N/A inherited from predecessor + 2 active + 3 N/A).

---

## Closing: structural value of this APPEND-ONLY reconciliation

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + "Forbidden premature KILL" non-negotiable + "Bugs must be permanently fixed AND self-protected against" non-negotiable:

**This memo achieves**:
1. Structural visibility of the implementation-level falsification (codex byte-mutation smoke + claude design memo + audit memo converged in ONE append-only artifact);
2. Operator-decision queue for Catalog #344 NEW EXCLUDED context proposal (§5);
3. Reclassification of cdf_table_blob → REMOVAL paradigm candidate (§4) with closed-form rate-axis prediction + sister parser-safe-extension precedent;
4. Preservation of predecessor design memo + codex byte-mutation smoke + audit memo per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY;
5. Cathedral autopilot ranker structural signal (Hook #4) + continual-learning posterior anchor (Hook #5);
6. Cascade-coherence demonstration of the 7th codex sister convergence pattern today.

**This memo deliberately does NOT achieve**:
- No paradigm-level KILL of canonical equation #26 procedural-substitution (paradigm intact per §3);
- No BUILD of REMOVAL paradigm reclassification (Phase 3 BUILD is operator-routable per §7 item 3);
- No mutation of predecessor artifacts (Catalog #110/#113 APPEND-ONLY discipline strictly observed);
- No paid GPU dispatch (zero cost; structural-protection artifact only);
- No anchor registration in canonical equations registry (predecessor design memo's DEFER_PENDING_EVIDENCE verdict remains binding; this memo does NOT advance to anchor registration);
- No spawning of nested subagents (operator-routable next-actions per §7 are explicitly operator-decision; THIS memo is the convergence pass, not a dispatcher);
- No NEW canonical equation #26 EXCLUDED context registration (proposed in §5 as operator-decision per Catalog #344 protocol; THIS memo does NOT mutate `src/tac/canonical_equations/procedural_codebook_savings.py`).

**Lane**: `lane_wave_3_atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521` L1 (impl_complete + memory_entry).

**Cost**: $0 GPU + ~30 min wall-clock.
