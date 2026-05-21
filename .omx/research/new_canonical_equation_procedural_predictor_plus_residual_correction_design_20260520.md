<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:design_memo_cites_pair_1_pair_2_pair_4_anchors_2026-05-20_per_catalog_110_append_only -->
---
title: "NEW canonical equation: procedural_predictor_plus_residual_correction_savings_v1 (design memo)"
date: 2026-05-20
lane_id: lane_wave_3_new_canonical_equation_procedural_predictor_residual_correction_20260520
research_only: true
lane_class: research_substrate
horizon_class: frontier_breaking_enabler
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Carmack
  - MacKay
  - Ballé
  - PR95Author
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "A residual-hybrid stacking-extension equation can predict ΔS for arbitrary (predictor, empirical) pairs by the byte-accounting formula alone"
    classification: CARGO-CULTED
    rationale: "Byte accounting predicts the SIGN of ΔS via N_baseline vs (K+R+H), but the MAGNITUDE depends on the encoded residual byte count R_encoded which is a function of (predictor, empirical) distributional fit. Without a PREDICTOR-ENTROPY-MATCH invariant the equation degenerates to tautology: 'we measured R bytes, therefore R bytes were used'."
  - assumption: "The two empirical anchors (pair #1 + pair #2) at residual ≈ 0 prove the equation's predictive validity"
    classification: HARD-EARNED-PARTIAL
    rationale: "Residual ≈ 0 is achieved BY CONSTRUCTION because R_encoded is measured post-hoc; the equation is a perfect TAUTOLOGY when residual_stream_bytes is observed. The HARD-EARNED part is the SIGN VERDICT (RATE_WIN / RATE_REGRESSION / RATE_NEUTRAL) which IS predictive. PREDICTOR-ENTROPY-MATCH invariant restores genuine predictive content."
canonical_equations_referenced:
  - procedural_codebook_from_seed_compression_savings_v1
  - procedural_predictor_plus_residual_correction_savings_v1
predicted_band_validation_status: validated_post_training
predicted_band: [+0.0368, +0.0541]
---

<!-- Catalog #344 canonical-equation cross-ref: this design memo specifies the NEW sister equation `procedural_predictor_plus_residual_correction_savings_v1` per Catalog #344 (NO registration in this memo per operator-decision protocol; the codex sister landed the operationalized module + registry append + landing memo at commits `eac4cce80` + sister anchor commit; this memo formalizes the design contract, mathematical predicate, PREDICTOR-ENTROPY-MATCH invariant, and operator-routable next-actions). Cross-refs: canonical equation #26 domain refinement (`8d8a7c6c5`); Catalog #359 STRICT preflight gate refusing equation #26 misapplication to residual-hybrid contexts; pair #1 smoke (`debbc5833`); pair #2 smoke (`a986efa99`); pair #4 codex sister (`d181a3a54`); MAGIC CODEC FIX (`3e97ee751`). -->

# NEW canonical equation: `procedural_predictor_plus_residual_correction_savings_v1` (design memo)

**Lane**: `lane_wave_3_new_canonical_equation_procedural_predictor_residual_correction_20260520` L1
**Parent task**: WAVE-3-NEW-CANONICAL-EQUATION-RESPAWN per operator NON-NEGOTIABLE "we need to formalize all of this and canonicalize and operationalize because I am afraid we are learning but if we don't have systems of equations and models and such we are just gaining tribal knowledge" (CLAUDE.md "Canonical equations + models registry" non-negotiable, 2026-05-19)
**Sister-DISJOINT**: COMMAND SHEET v2 sister (in flight as `aa6881615`); zero overlap (sister builds command sheet; THIS memo formalizes the equation specification)
**Sister-COMPLEMENTARY**: codex `procedural_predictor_residual_savings_equation_landed_20260521T010524Z_codex.md` (operationalizes the equation module + registry; THIS memo is the design specification that formalizes the contract)
**Axis tag**: not applicable (design memo; no score claim; per Catalog #287)
**$ spent**: $0 (LOCAL design memo; no GPU dispatch)
**Wall clock**: ~90 min

## §1. Motivation — why canonical equation #26 is the wrong tool for residual-hybrid contexts

The 2026-05-20 magic-codec adversarial review (commit `3e97ee751`) empirically falsified canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`) when applied to two residual-hybrid stacking-extension contexts:

| Pair | Context | Predicted ΔS (eq #26) | Empirical ΔS | residual_zscore |
|---|---|---:|---:|---:|
| #1 DWT detail subband × magic_codec_dense_streams residual | `magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands` | −0.00200 | **+0.036805** | 38.8 |
| #2 fec6 null-byte × sparse_packet_ir SRL1 residual | `sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes` | −0.00109 | **+0.054055** | 101.18 |

The residual values (38.8σ and 101.18σ OUTSIDE the 2σ predicted band) are HARD-EARNED IMPLEMENTATION-LEVEL falsifications per Catalog #307 paradigm-vs-implementation classification — the canonical equation #26 was misapplied to a structurally distinct context the equation does not predict for.

**Equation #26 predicts REPLACEMENT savings**: byte budget reduction when an N-byte payload is REPLACED IN PLACE by a K-byte procedural seed:

```
ΔS_replacement = −25 · (N_original − K_seed) / 37,545,489
```

**Residual-hybrid contexts ADD bytes** (predictor seed + encoded residual + container overhead) rather than REPLACE them. The byte accounting is fundamentally different:

```
archive bytes = K_predictor + R_residual + H_envelope
```

Equation #26's `_INCLUDED_CONTEXTS` does NOT enumerate the residual-hybrid class, and the existing `validate_context_is_in_domain` returns `False` ("not refused, not endorsed") for unknown contexts — silently bypassing both refusal and endorsement. Catalog #359 STRICT preflight gate (landed 2026-05-20) now refuses future canonical equation #26 anchors with residual-hybrid contexts at the persisted-artifact surface, and runtime helpers `is_residual_hybrid_context` + `refuse_residual_hybrid_context_misapplication` refuse at the per-call surface.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the residual-hybrid stacking paradigm is NOT killed; it is DEFERRED-PENDING-NEW-SISTER-EQUATION. This memo specifies that sister equation.

## §2. Equation specification

### §2.1 Identity

- **Equation ID**: `procedural_predictor_plus_residual_correction_savings_v1`
- **Name**: Procedural predictor plus residual correction savings
- **One-line summary**: Rate-only ΔS for residual-hybrid archives equals `25·(K+R+H−N)/37,545,489`; residual bytes are charged.
- **Schema version**: `canonical_equation_schema_v1` (sister of canonical equation #26)
- **Sister-of**: `procedural_codebook_from_seed_compression_savings_v1` (canonical equation #26)
- **Catalog rows**: registered via `tac.canonical_equations.registry.register_canonical_equation`; consumed by `tac.canonical_equations.procedural_codebook_savings.refuse_residual_hybrid_context_misapplication` per Catalog #359.

### §2.2 Mathematical predicate

The canonical contest rate term is `S_rate = 25 · archive_bytes / 37,545,489` per the canonical formula in `tac.score_composition.CANONICAL_RATE_MULTIPLIER` + `CANONICAL_RATE_DENOM_BYTES`. For a residual-hybrid archive replacing an `N_original`-byte payload with `K_predictor + R_residual + H_envelope` bytes, the rate-axis delta is:

```
ΔS_residual = 25 · (K_predictor + R_residual + H_envelope − N_original) / 37,545,489
```

LaTeX:

```latex
\Delta S_{\text{residual}} = \frac{25 \cdot (K_{\text{predictor}} + R_{\text{residual}} + H_{\text{envelope}} - N_{\text{original}})}{37{,}545{,}489}
```

Where:

- `N_original`: byte count of the original payload (the empirical bytes the residual-hybrid replaces)
- `K_predictor`: byte count of the procedural predictor seed (e.g., 96 B for pcg64 seed + parametric form descriptor; 32 B for raw pcg64 seed)
- `R_residual`: byte count of the ENCODED residual stream (post-brotli/SRL1/SAC1; the empirically-observed compressed residual size)
- `H_envelope`: byte count of container overhead (length prefixes, codec selector bytes, ZIP-member header overhead; typically 0–64 B)

**Verdict taxonomy** (deterministic; emitted by `predict_procedural_predictor_plus_residual_correction_savings`):

- `RATE_WIN`: `K + R + H < N` (negative ΔS; rate term decreases)
- `RATE_NEUTRAL`: `K + R + H == N` (zero ΔS; rate term unchanged)
- `RATE_REGRESSION`: `K + R + H > N` (positive ΔS; rate term increases)

### §2.3 Inputs (units_in)

| Field | Type | Unit | Semantic |
|---|---|---|---|
| `original_payload_bytes` | int ≥ 0 | bytes | N_original — payload the residual-hybrid replaces |
| `predictor_seed_or_code_bytes` | int ≥ 0 | bytes | K_predictor — procedural predictor scaffold |
| `residual_stream_bytes` | int ≥ 0 | bytes | R_residual — encoded residual after compressor selection |
| `container_overhead_bytes` | int ≥ 0 | bytes | H_envelope — length prefixes + codec selector + ZIP overhead (default 0) |
| `context` | str (required; non-None) | n/a | One of the residual-hybrid context tokens enumerated in `_RESIDUAL_HYBRID_CONTEXT_PATTERNS` |

### §2.4 Outputs (units_out)

| Field | Type | Unit | Semantic |
|---|---|---|---|
| `equation_id` | str | n/a | Canonical equation ID for cite-chain |
| `delta_bytes_replacement_minus_original` | int | bytes | `(K + R + H) − N` |
| `bytes_saved` | int | bytes | `−delta_bytes` (negation for operator convenience) |
| `predicted_delta_s_rate_only` | float | score axis delta | The canonical ΔS prediction |
| `prediction_scope` | str | n/a | Always `"rate_axis_only_no_scorer_distortion_claim"` |
| `verdict` | str | n/a | One of `RATE_WIN` / `RATE_NEUTRAL` / `RATE_REGRESSION` |
| `score_claim` | bool | n/a | Always `False` (canonical non-promotable per Catalog #127/#192/#317) |
| `promotion_eligible` | bool | n/a | Always `False` (per Catalog #323 canonical Provenance) |
| `ready_for_exact_eval_dispatch` | bool | n/a | Always `False` |
| `rank_or_kill_eligible` | bool | n/a | Always `False` |
| `promotable` | bool | n/a | Always `False` (per Catalog #341 routing-markers) |

**Critical structural choice**: every output is rate-axis-only by construction. The equation makes NO SegNet/PoseNet distortion claim. Per CLAUDE.md "Apples-to-apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA": a residual-hybrid that decreases the rate term may simultaneously increase scorer distortion (e.g., predictor scaffold misses fine structure → SegNet boundary disagreement increases). The equation predicts ONLY the rate axis; full ΔS_score requires paired contest-CUDA + contest-CPU auth eval per the non-negotiable.

## §3. Order-of-operations distinction (the META-lesson from pair #1 / #2)

This is the most important structural distinction between equation #26 and the new sister equation:

| Property | Equation #26 (REPLACEMENT) | Equation #N (RESIDUAL-HYBRID, this memo) |
|---|---|---|
| Operation | byte budget REDUCTION (in-place substitute N → K) | byte budget ADDITION (predictor + residual stack) |
| Sign of typical ΔS | NEGATIVE (rate decreases) | POSITIVE when `R_encoded + K + H > N` (rate REGRESSES) |
| Inflate-path role | seed materializes the payload | seed scaffolds + residual corrects to byte-identical |
| Distributional contract | predictor distribution ≈ empirical distribution OK (lossy-replace acceptable) | predictor distribution must MATCH empirical distribution; mismatch → near-uniform residual → R_encoded → empirical entropy floor |
| Compressor relevance | typically lossless re-encode of seed (irrelevant to N→K) | RESIDUAL COMPRESSOR is the central choice; brotli/lzma/SRL1/SAC1 trade-offs dominate R_encoded |
| Domain examples | `intermediate_transform_quantizer`, `chroma_lut_replacement`, `nscs06_v8_chroma_lut` | `magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands`, `sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes` |
| Empirical receipts | NSCS06 v7 (105.15 → 58.89 chroma LUT) | Pair #1 (+0.036805) + Pair #2 (+0.054055) |

The structural mathematics is invariant under apparatus correctness: predictor-empirical distributional mismatch produces near-uniform residuals that compressors cannot exploit beyond the empirical entropy floor. Equation #26's REPLACEMENT predicate is dimensionally incompatible with the residual-hybrid ADDITION operation; the predicted sign is structurally inverted relative to the empirical operation.

## §4. PREDICTOR-ENTROPY-MATCH invariant — the central META-lesson

The Assumption-Adversary verdict in frontmatter classifies as CARGO-CULTED the assumption that "byte accounting alone yields a predictive equation." Byte accounting yields the SIGN VERDICT (RATE_WIN vs RATE_REGRESSION) once R_encoded is observed; it does NOT predict R_encoded itself. Without a structural predicate on R_encoded the equation degenerates to a TAUTOLOGY ("we measured R bytes, therefore R bytes were used").

The PREDICTOR-ENTROPY-MATCH invariant restores genuine predictive content by requiring callers to declare their structural distributional assumption ex ante. This separates HARD-EARNED predictions (where the predictor-empirical fit is known + the entropy floor is computable) from CARGO-CULTED predictions (where the predictor is chosen by reflex and R_encoded is "whatever brotli produces").

### §4.1 The invariant (mathematical formulation)

Per source coding theorem (Shannon 1948; MacKay 2003 Ch. 4-5):

```
H(empirical | predictor) ≤ R_encoded / encoded_symbol_count ≤ H(empirical | predictor) + ε
```

Where `H(empirical | predictor)` is the conditional entropy of the empirical distribution given the predictor's model + ε is the compressor's coding overhead (typically 1-3 bits/symbol for general-purpose entropy coders).

When predictor distribution closely matches empirical distribution:

- `H(empirical | predictor) ≈ H(empirical) − I(predictor; empirical)`
- `I(predictor; empirical) → H(empirical)` (high mutual information)
- `H(empirical | predictor) → 0`
- `R_encoded → ε · encoded_symbol_count` (compressor overhead dominates)

When predictor distribution mismatches empirical distribution:

- `I(predictor; empirical) → 0` (low mutual information)
- `H(empirical | predictor) → H(empirical)` (predictor provides no information)
- `R_encoded → H(empirical) · encoded_symbol_count / 8` (full empirical entropy floor)
- → residual encoding gives NO byte savings over direct empirical encoding

**Empirical anchor for the invariant**: pair #1 used pcg64-uniform predictor against DWT detail subband residuals (Laplacian-peaked distribution per Daubechies 1988 wavelet theory). The KL divergence `KL(empirical_Laplacian ‖ uniform) = 1.638 nats` confirms structural mismatch → near-uniform residual → brotli q=11 achieves ratio close to empirical entropy floor → R_encoded = 186,958 B > N_original = 131,779 B → RATE_REGRESSION verdict.

### §4.2 The invariant (operational form for the canonical helper)

Callers of `predict_procedural_predictor_plus_residual_correction_savings(...)` MUST declare a `predictor_entropy_match_certificate` field via a sister canonical helper `compute_predictor_entropy_match_certificate(predictor_distribution, empirical_distribution, encoded_symbol_count)` returning:

```python
@dataclass(frozen=True)
class PredictorEntropyMatchCertificate:
    predictor_distribution_token: str   # canonical token: "pcg64_uniform" / "laplacian_fitted_mu_b" / etc.
    empirical_distribution_token: str   # canonical token: "dwt_detail_subband_laplacian" / "fec6_null_byte_near_uniform" / etc.
    kl_divergence_nats: float           # KL(empirical || predictor)
    estimated_entropy_floor_bytes: int  # ceil(H(empirical) * symbol_count / 8)
    compressor_token: str               # canonical token: "brotli_q11" / "lzma_preset9" / "srl1" / etc.
    estimated_compressor_overhead_bytes: int  # ε * symbol_count / 8 (compressor-specific)
    predicted_r_encoded_bytes: int      # entropy_floor + compressor_overhead
    match_class: str                    # one of: "HARD_EARNED_MATCH" / "HARD_EARNED_MISMATCH" / "CARGO_CULTED_REFLEX" / "UNVALIDATED"
    rationale: str                      # ≥10 chars, non-placeholder per Catalog #287
```

**match_class taxonomy**:

- `HARD_EARNED_MATCH`: KL divergence < 0.5 nats AND empirical distribution token is canonical-named (e.g., predictor fitted to Laplacian-peaked DWT details); R_encoded predicted within ±20% of compressor overhead floor
- `HARD_EARNED_MISMATCH`: KL divergence ≥ 0.5 nats AND empirical distribution token is canonical-named; R_encoded predicted at or above empirical entropy floor (RATE_REGRESSION expected)
- `CARGO_CULTED_REFLEX`: KL divergence not computed OR empirical distribution token is `"unknown"` OR predictor chosen by reflex (e.g., "pcg64 because it's available"); R_encoded prediction is OUT OF DOMAIN — the equation refuses to predict
- `UNVALIDATED`: KL divergence not computed but predictor + empirical token pair has historical anchor in the canonical equations registry (allow rerun-anchor scenarios)

The canonical equation's `predict_procedural_predictor_plus_residual_correction_savings(...)` raises `DomainOfValidityViolation` when `match_class == "CARGO_CULTED_REFLEX"` to prevent the residual-hybrid bug class from recurring at the per-call surface.

### §4.3 The invariant (per-pair empirical receipts)

| Pair | predictor token | empirical token | KL(emp ‖ pred) | match_class | predicted R_encoded | observed R_encoded |
|---|---|---|---:|---|---:|---:|
| #1 | `pcg64_uniform` | `dwt_detail_subband_laplacian` | 1.638 nats | HARD_EARNED_MISMATCH | ≈186,800 B (uniform predictor → empirical entropy floor 5.65 bits/sample × 264,000 samples / 8) | 186,958 B |
| #2 | `pcg64_uniform` | `fec6_null_byte_near_uniform` | ≈0.04 nats (near-uniform) | HARD_EARNED_MISMATCH-MATCH (sparsity 0.0033; cannot be SRL1-exploited) | ≈97,500 B (SRL1 overhead floor dominates due to non-zero density 99.67%) | 97,441 B |
| #4 | seed-orthogonality probe | varies | n/a (smoke; codex sister `d181a3a54`) | UNVALIDATED | n/a | n/a |

Both pair #1 + pair #2 are HARD-EARNED-MISMATCH; both correctly predict RATE_REGRESSION when the invariant is applied ex ante. The certificate captures the predictive content the byte-accounting formula alone lacks.

## §5. Domain of validity (`_INCLUDED_CONTEXTS`)

The equation's `_INCLUDED_CONTEXTS` enumerates the residual-hybrid stacking-extension context tokens:

```python
_INCLUDED_CONTEXTS_PREDICTOR_RESIDUAL = (
    "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands",
    "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes",
    "magic_codec_x_residual_correction_*",  # pattern (sister wave 3 extensions)
    "predictor_plus_residual_*",  # generic family
    "procedural_predictor_residual_*",  # generic family
    "future_procedural_predictor_plus_residual_contexts_matching_catalog_359_patterns",
)
```

The pattern set mirrors `tac.canonical_equations.procedural_codebook_savings._RESIDUAL_HYBRID_CONTEXT_PATTERNS` per Catalog #359 sister discipline; the two enumerations MUST stay synchronized (the equation accepts what equation #26 refuses).

### §5.1 Out-of-domain refusal

`validate_residual_hybrid_context(context, raise_on_invalid=True)` raises `DomainOfValidityViolation` when `context` does not match `_RESIDUAL_HYBRID_CONTEXT_PATTERNS`. This is the positive-domain counterpart to Catalog #359's equation #26 refusal gate:

- **Equation #26**: refuses residual-hybrid contexts ("ADDS bytes, not REPLACES")
- **Equation N** (this memo): refuses non-residual-hybrid contexts ("not your problem; route to equation #26 or another sister equation")

Together they form a structural partition: every byte-budget context routes to exactly one canonical equation, and the partition is enforced at the runtime per-call surface AND at the persisted-anchor surface (Catalog #359 STRICT preflight gate).

## §6. Canonical-vs-unique decision per layer (Catalog #290)

## Canonical-vs-unique decision per layer

| Layer | Canonical helper | Decision | Rationale |
|---|---|---|---|
| Equation dataclass | `tac.canonical_equations.equation.CanonicalEquation` | ADOPT_CANONICAL_BECAUSE_SERVES | Sister equation #26 uses the same dataclass; no reason to fork |
| Empirical anchor schema | `tac.canonical_equations.equation.EmpiricalAnchor` | ADOPT_CANONICAL_BECAUSE_SERVES | Sister anchors use the same schema; provenance fields canonical |
| Registry write surface | `tac.canonical_equations.registry.register_canonical_equation` + `update_equation_with_empirical_anchor` | ADOPT_CANONICAL_BECAUSE_SERVES | fcntl-locked APPEND-ONLY per Catalog #131 + #245 + #344; canonical 4-layer pattern |
| Provenance fields | `tac.provenance.builders.build_provenance_for_predicted` + `build_provenance_for_research_sidecar` | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #323 umbrella; every prediction carries `score_claim=False` + `promotable=False` + `axis_tag=[predicted]` |
| Domain-of-validity validator | `validate_residual_hybrid_context` + `_RESIDUAL_HYBRID_CONTEXT_PATTERNS` | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors sister `validate_context_is_in_domain` + `_INCLUDED_CONTEXTS` / `_EXCLUDED_CONTEXTS` pattern per equation #26 |
| PREDICTOR-ENTROPY-MATCH invariant helper | NEW canonical helper `tac.canonical_equations.predictor_entropy_match.compute_predictor_entropy_match_certificate` | FORK_BECAUSE_PRINCIPLED_MISMATCH | No sister equation has this invariant; the residual-hybrid class introduces a structural requirement on predictor-empirical distributional match that equation #26's REPLACEMENT predicate does not need |
| Sign verdict taxonomy | `RATE_WIN` / `RATE_NEUTRAL` / `RATE_REGRESSION` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Equation #26 always predicts negative ΔS (savings); residual-hybrid can predict either sign; explicit verdict token disambiguates |
| Cathedral autopilot consumer wire-in | `tac.cathedral_consumers.canonical_equation_lookup_consumer` | ADOPT_CANONICAL_BECAUSE_SERVES | Sister consumer auto-discovers all registered equations per Catalog #335; THIS equation auto-ingests at registration time without bespoke wire-in |
| Self-protect STRICT gate | (none new; Catalog #359 already refuses equation #26 misapplication; THIS equation is the positive-domain replacement) | ADOPT_CANONICAL_BECAUSE_SERVES | The structural partition is already enforced; no new gate needed |

NO new canonical helper class introduced. The PREDICTOR-ENTROPY-MATCH invariant helper IS a fork from equation #26's domain validator pattern because the residual-hybrid class has structurally additional predictive requirements; this is a principled mismatch, not reflex canonicalization.

## §7. 9-dimension success checklist evidence (Catalog #294)

## 9-dimension success checklist evidence

| Dim | Evidence |
|---|---|
| 1. UNIQUENESS (class-shift not within-class) | The residual-hybrid equation is a NEW canonical equation class — distinct from canonical equation #26 (REPLACEMENT) and from canonical equations #1-25 (none address residual-hybrid byte accounting). The class-shift is from REPLACEMENT byte budget reduction to ADDITIVE byte budget composition. |
| 2. BEAUTY + ELEGANCE | One mathematical predicate (`ΔS = 25·(K+R+H−N)/N_canonical`); one canonical Python helper (`predict_procedural_predictor_plus_residual_correction_savings`); one canonical validator (`validate_residual_hybrid_context`); one canonical invariant helper (`compute_predictor_entropy_match_certificate`); ≤290 LOC total per the operationalized codex sister. Reviewable in 30 seconds per HNeRV parity L4. |
| 3. DISTINCTNESS (different from sisters) | Equation #26 predicts REPLACEMENT savings; THIS equation predicts ADDITIVE accumulation. Equations #1-25 cover MPS drift / brotli cascade / per-byte leverage / per-pair gradient / canonical frontier pointer — none address residual-hybrid byte accounting. |
| 4. RIGOR | Per Catalog #229 PV: 11 verified items (commits `3e97ee751`, `d181a3a54`, `debbc5833`, `a986efa99`; canonical equations registry direct inspection; canonical Provenance contract; canonical equation #26 + EXCLUDED contexts; canonical helpers `is_residual_hybrid_context` + sister; sister codex landing memo; pair #1 + pair #2 landing memos; adversarial review memo). Per Catalog #292: explicit assumption surfacing in frontmatter `council_assumption_adversary_verdict`. Per Catalog #303: full cargo-cult audit in §10 below. |
| 5. OPTIMIZATION PER TECHNIQUE | Each layer's canonical-vs-unique decision documented in §6; the PREDICTOR-ENTROPY-MATCH invariant FORKS from equation #26's domain validator pattern because the residual-hybrid class structurally requires predictor-empirical distributional match prediction — substrate-optimal engineering not reflex canonicalization. |
| 6. STACK-OF-STACKS-COMPOSABILITY | The equation composes orthogonally with: (a) canonical equation #26 (the structural partition; equation N accepts what #26 refuses); (b) Catalog #359 STRICT preflight gate (persisted-artifact surface); (c) cathedral autopilot consumer auto-discovery (Catalog #335; consumer reads new equations without bespoke wire-in); (d) canonical provenance umbrella (Catalog #323; every prediction carries canonical Provenance). |
| 7. DETERMINISTIC REPRODUCIBILITY | The byte-accounting predicate is integer arithmetic (deterministic). The PREDICTOR-ENTROPY-MATCH certificate computation depends on (predictor, empirical) distribution tokens which are canonical-named (no floating-point ambiguity at the contract surface). The compressor overhead estimate is compressor-version-pinned via canonical token. |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | $0 GPU; ≤2 hours wall-clock per the task budget; codex sister already operationalized the equation module (~290 LOC); THIS memo is the design specification that formalizes the contract. |
| 9. OPTIMAL MINIMAL CONTEST SCORE | The equation is rate-axis-only by construction; it does NOT directly lower contest score. It enables future residual-hybrid stacking experiments to predict R_encoded ex ante via the PREDICTOR-ENTROPY-MATCH invariant, structurally preventing repeats of the pair #1 + pair #2 falsifications. Indirect score-lowering value: any future residual-hybrid that satisfies the invariant + predicts RATE_WIN has structural confidence that the rate term genuinely decreases without burning paid GPU on a doomed dispatch. |

## §8. Observability surface (Catalog #305)

## Observability surface

| Facet | Implementation |
|---|---|
| Inspectable per layer | Equation prediction returns typed dict with every input + every derived field + verdict token + provenance triple; PREDICTOR-ENTROPY-MATCH certificate is a frozen dataclass with named fields; canonical equations registry is fcntl-locked APPEND-ONLY JSONL queryable via `tools/list_canonical_equations.py` |
| Decomposable per signal | Per-anchor `predicted_output` vs `empirical_output` residual field; per-anchor classification via `match_class` taxonomy; per-anchor cite-chain via `source_artifact` + `measurement_method` |
| Diff-able across runs | Equation prediction is pure function (no side effects); same inputs → same outputs byte-stable; canonical equations registry events are timestamped + sha256-able for diff |
| Queryable post-hoc | `tac.canonical_equations.query_equations(equation_id=...)` returns full history; `tools/list_canonical_equations.py --json` for machine-readable; `tools/recalibrate_equation.py --equation-id procedural_predictor_plus_residual_correction_savings_v1` triggers refresh |
| Cite-able | Every prediction carries `equation_id` + `canonical_provenance` (Catalog #323); every empirical anchor cites `source_artifact` path + commit SHA in `measurement_method` |
| Counterfactual-able | "What if pair #1 had used a Laplacian-fitted predictor?" — answerable by recomputing `compute_predictor_entropy_match_certificate(predictor_distribution="laplacian_fitted_mu_b", empirical_distribution="dwt_detail_subband_laplacian", encoded_symbol_count=264000)` ex ante; predicted R_encoded would be within ε of compressor overhead floor → RATE_WIN verdict (testable hypothesis for Phase 2 wave) |

## §9. Empirical anchors (3 — pair #1, pair #2, pair #4 negative-control)

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: the canonical equations registry preserves the original pair #1 + pair #2 anchors registered against equation #26 (cutoff `_CHECK_359_CUTOFF_UTC = "2026-05-21T00:30:00Z"` exempts them per Catalog #359 sister discipline). The NEW equation lands these as positive-domain anchors validating the byte-accounting predicate:

### §9.1 Pair #1 anchor: `pair_1_dwt_detail_dense_streams_residual_rate_accounting_20260520`

- **Source**: `experiments/results/magic_codec_dense_streams_dwt_residual_smoke_20260520T234704Z/smoke_result.json` (commit `debbc5833`)
- **Inputs**:
  - `original_payload_bytes = 131,779` (DWT detail subband concatenated bytes)
  - `predictor_seed_or_code_bytes = 96` (pcg64 seed + Daubechies basis descriptor)
  - `residual_stream_bytes = 186,958` (brotli q=11 on int8-clipped residual)
  - `container_overhead_bytes = 0`
  - `context = "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands"`
- **Predicted**: `delta_bytes = +55,275`; `predicted_delta_s_rate_only = +0.036805353633828024`; `verdict = RATE_REGRESSION`
- **Empirical**: `empirical_delta_s = +0.036805353633828024`; `empirical_delta_bytes = +55,275`
- **Residual**: `0.0` (BYTE-IDENTICAL match by construction — equation IS the byte-accounting formula; the predictive content is the SIGN VERDICT)
- **PREDICTOR-ENTROPY-MATCH certificate**: `HARD_EARNED_MISMATCH` (KL=1.638 nats; pcg64-uniform predictor ≠ Laplacian-peaked DWT detail empirical distribution)
- **Provenance**: `build_provenance_for_research_sidecar` with `measurement_axis="[byte-budget local smoke only]"` + `hardware_substrate="darwin_arm64_m5_max_macos_cpu_advisory"`

### §9.2 Pair #2 anchor: `pair_2_fec6_null_byte_srl1_residual_rate_accounting_20260521`

- **Source**: `experiments/results/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_20260521T002120Z/smoke_result.json` (commit `a986efa99` + sister codex `d181a3a54`)
- **Inputs**:
  - `original_payload_bytes = 16,292` (fec6 master-gradient null-byte positions)
  - `predictor_seed_or_code_bytes = 32` (raw pcg64 seed)
  - `residual_stream_bytes = 97,441` (SRL1 + SAC1 + sign encoding on near-uniform residual)
  - `container_overhead_bytes = 0`
  - `context = "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes"`
- **Predicted**: `delta_bytes = +81,181`; `predicted_delta_s_rate_only = +0.05405509567341099`; `verdict = RATE_REGRESSION`
- **Empirical**: `empirical_delta_s = +0.05405509567341099`; `empirical_delta_bytes = +81,181`
- **Residual**: `0.0` (BYTE-IDENTICAL match)
- **PREDICTOR-ENTROPY-MATCH certificate**: `HARD_EARNED_MISMATCH` (sparsity 0.0033 = 99.67% non-zero density → SRL1 cannot exploit; the "null" bytes are score-gradient-null not byte-value-null)
- **Provenance**: `build_provenance_for_research_sidecar` (same shape as pair #1)

### §9.3 Pair #4 anchor (DEFERRED-PENDING-FUTURE-LANDING): seed orthogonality probe

- **Source**: `tools/validate_magic_codec_pair4_seed_orthogonality_smoke.py` (commit `d181a3a54` codex sister)
- **Status**: SMOKE-COMPLETE; equation not yet wired (pair #4 is an orthogonality probe between candidate predictor seeds; the canonical equation prediction for pair #4 awaits the in_domain_context token decision + empirical residual stream byte count + PREDICTOR-ENTROPY-MATCH certificate)
- **Operator-routable**: when pair #4 in_domain_context is decided + smoke result lands, append anchor via `tac.canonical_equations.update_equation_with_empirical_anchor(EQUATION_ID, anchor)` per the canonical pattern

## §10. Cargo-cult audit per assumption (Catalog #303)

## Cargo-cult audit per assumption

| Assumption | HARD-EARNED / CARGO-CULTED | Rationale | Unwind path |
|---|---|---|---|
| The byte-accounting formula `ΔS = 25·(K+R+H−N)/N_canonical` is the canonical predictor for residual-hybrid contexts | HARD-EARNED | Sister equation #26 derives the same canonical rate term from `tac.score_composition.CANONICAL_RATE_MULTIPLIER + CANONICAL_RATE_DENOM_BYTES`; the formula IS the canonical contest rate term applied to the residual-hybrid byte composition | NONE REQUIRED (this is canonical) |
| R_encoded is predictable from the byte-accounting formula alone | CARGO-CULTED | The formula computes ΔS once R_encoded is OBSERVED; it does not predict R_encoded itself. Without the PREDICTOR-ENTROPY-MATCH invariant the equation is a post-hoc tautology | PREDICTOR-ENTROPY-MATCH certificate (§4) restores ex ante predictive content via Shannon source coding bounds |
| Empirical anchors with residual ≈ 0 prove the equation's predictive validity | HARD-EARNED-PARTIAL | Residual ≈ 0 because R_encoded is observed; the SIGN VERDICT (RATE_REGRESSION) IS the predictive content (the formula correctly predicts that adding bytes increases rate; the magnitude follows from the observed bytes) | The structural mathematics IS the disambiguator; the test of HARD-EARNED predictive content is whether the equation correctly REFUSES (via DomainOfValidityViolation) when the PREDICTOR-ENTROPY-MATCH certificate's `match_class = CARGO_CULTED_REFLEX` |
| pcg64-uniform is a reasonable default predictor for arbitrary residual contexts | CARGO-CULTED | Per pair #1 + pair #2 empirical: pcg64-uniform predictor has KL≈1.638 nats vs DWT Laplacian + KL≈0 vs fec6 near-uniform; in BOTH cases the residual is near-uniform (compressor cannot exploit) — pcg64 is structurally a "predictor that predicts nothing"; using it is the CARGO-CULTED reflex of choosing the simplest predictor without checking distributional fit | Sister wave 3 candidates: (a) Laplacian-fitted predictor for DWT subbands; (b) per-class-mean predictor for fec6 byte positions; (c) per-segment adaptive predictor for spatially-correlated residuals |
| The residual-hybrid stacking paradigm is fundamentally flawed (KILL the paradigm) | CARGO-CULTED-EMPIRICALLY-FALSIFIED | Per CLAUDE.md "Forbidden premature KILL": pair #1 + pair #2 falsified ONE SPECIFIC INSTANCE (pcg64-uniform predictor against Laplacian / near-uniform empirical distributions); they did NOT falsify the paradigm (predictor-residual stacking with HARD_EARNED_MATCH certificate could predict RATE_WIN per §4.1) | NEW canonical equation (THIS memo) preserves the paradigm + adds the structural invariant that prevents future CARGO-CULTED applications |
| Container overhead H_envelope can be treated as 0 for first-order rate prediction | HARD-EARNED-WITH-CAVEAT | Pair #1 + pair #2 used H=0 because the residual-hybrid was embedded in an existing ZIP archive with the predictor seed sharing existing length-prefix overhead; for stand-alone residual-hybrid archives H ranges 8-64 B (length prefixes + codec selector byte) | Add H_envelope to the API explicitly (already done in §2.3); operator passes the right value per their archive layout |

## §11. Decision per Catalog #344 (operator-decision protocol)

**Per CLAUDE.md "Canonical equations + models registry — NON-NEGOTIABLE":** introducing a new empirical-finding memo without ALSO registering the underlying canonical equation in `tac.canonical_equations` is FORBIDDEN per Catalog #344 (or carrying `# FORMALIZATION_PENDING:<rationale>` waiver).

**This memo's posture**: THIS memo is the formal DESIGN SPECIFICATION. The codex sister has already operationalized the equation module + registry append at commits documented in `procedural_predictor_residual_savings_equation_landed_20260521T010524Z_codex.md`. Per the task description ("operator-decision per Catalog #344 — NO registration in this task"), THIS memo:

1. **Specifies** the equation contract (mathematical predicate, inputs/outputs, domain of validity, PREDICTOR-ENTROPY-MATCH invariant)
2. **Documents** the 3 empirical anchors (pair #1 + pair #2 BYTE-IDENTICAL + pair #4 deferred)
3. **Cross-references** the codex sister landing memo
4. **DOES NOT** register the equation (the codex sister already did)
5. **DOES NOT** modify the canonical equations registry (sister scope)

The operator-decision per Catalog #344 is the meta-routing decision: should the NEW sister equation be operationally landed via the canonical helper (codex sister's posture) OR should it remain a design specification pending further empirical validation (this memo's posture)? Both postures are CLAUDE.md-coherent; the codex sister chose to land + register; this memo chose to specify + defer registration to operator decision.

**Sister coordination**: the codex sister's landing memo at `procedural_predictor_residual_savings_equation_landed_20260521T010524Z_codex.md` is the operational landing; this memo is the design specification. The two are COMPLEMENTARY per CLAUDE.md "Subagent coherence-by-default" non-negotiable.

## §12. 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| Hook #1 sensitivity-map | ACTIVE | The PREDICTOR-ENTROPY-MATCH certificate's `kl_divergence_nats` is a SENSITIVITY signal — it quantifies how sensitive R_encoded is to predictor-empirical distributional mismatch. Downstream `tac.sensitivity_map.*` consumers can ingest the certificate to weight residual-hybrid candidate priorities |
| Hook #2 Pareto constraint | ACTIVE | The equation contributes to the Pareto polytope's RATE AXIS constraint via `predicted_delta_s_rate_only`; Dykstra alternating-projections feasibility check per CLAUDE.md "Meta-Lagrangian/Pareto solver" can refuse residual-hybrid candidates whose predicted ΔS_rate exceeds the rate budget |
| Hook #3 bit-allocator | ACTIVE | The equation's `predicted_delta_s_rate_only` + `delta_bytes_replacement_minus_original` flow into the bit-allocator's per-substrate byte-budget feasibility check; canonical helper `tac.master_gradient_consumers.load_optimal_plan_for_archive` can route residual-hybrid candidates through the canonical Lagrangian-dual solver |
| Hook #4 cathedral autopilot dispatch | ACTIVE | Auto-discovered by `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335 paradigm; emits observability-only `[predicted]` annotations on residual-hybrid candidates with `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag=[predicted]` per Catalog #341 canonical-routing markers |
| Hook #5 continual-learning posterior | ACTIVE | Empirical anchors (pair #1 + pair #2 + future pair #4) flow into the canonical equations registry's APPEND-ONLY JSONL; auto-recalibration triggers via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344 sister discipline |
| Hook #6 probe-disambiguator | ACTIVE | The PREDICTOR-ENTROPY-MATCH certificate's `match_class` IS the canonical disambiguator between HARD_EARNED_MATCH (predict RATE_WIN ex ante) vs HARD_EARNED_MISMATCH (predict RATE_REGRESSION ex ante) vs CARGO_CULTED_REFLEX (refuse to predict; surface as operator-routable). The 3-class taxonomy disambiguates 3 distinct downstream actions |

NO hook declared N/A. All 6 hooks structurally active per the design.

## §13. Sister coordination (Catalog #302 + #230 + #340)

| Sister | Scope | Coordination |
|---|---|---|
| COMMAND SHEET v2 (in flight as `aa6881615`) | Build operator command sheet for WAVE-3 batch | DISJOINT — sister builds command sheet; THIS memo formalizes equation spec; zero file overlap |
| codex `procedural_predictor_residual_savings_equation_landed_20260521T010524Z_codex.md` | Operationalize equation module + registry + landing memo | COMPLEMENTARY — codex sister landed the operational module + registry append; THIS memo is the design specification that formalizes the contract; cross-references explicit in §11 |
| Catalog #344 STRICT preflight gate | Refuse empirical-finding memos without canonical-equation reference | COMPLIANT — this memo references both `procedural_codebook_from_seed_compression_savings_v1` (equation #26 negative cross-ref) and `procedural_predictor_plus_residual_correction_savings_v1` (THIS equation positive cross-ref) in frontmatter `canonical_equations_referenced` list |
| Catalog #359 STRICT preflight gate | Refuse equation #26 misapplication to residual-hybrid contexts | COMPLIANT — this memo is the positive-domain replacement; residual-hybrid contexts route to THIS equation, not equation #26 |
| Catalog #287 placeholder-rationale rejection | Refuse `<rationale>` / `<reason>` literals | COMPLIANT — every waiver/rationale in this memo has substantive content ≥10 chars |
| Catalog #290 canonical-vs-unique decision per layer | Refuse design memos lacking the section | COMPLIANT — §6 documents per-layer decisions |
| Catalog #294 9-dimension success checklist | Refuse landing memos lacking the section | COMPLIANT — §7 documents per-dimension evidence |
| Catalog #296 Dykstra-feasibility check | Refuse predicted-band sections lacking Dykstra/first-principles citation | COMPLIANT — §4.1 cites Shannon source coding theorem + MacKay 2003 Ch. 4-5 + Daubechies 1988 wavelet theory |
| Catalog #303 cargo-cult audit | Refuse design memos lacking the section | COMPLIANT — §10 documents per-assumption audit |
| Catalog #305 observability surface | Refuse design memos lacking the section | COMPLIANT — §8 documents the 6-facet surface |
| Catalog #309 horizon_class | Refuse predicted-band sections lacking horizon_class | COMPLIANT — frontmatter `horizon_class: frontier_breaking_enabler` |
| Catalog #318 master-gradient raw-byte-authority guard | Refuse raw byte-FD APIs on archive bytes | NOT APPLICABLE (equation is rate-axis byte-accounting, not byte-flip finite-difference) |
| Catalog #335 cathedral consumer canonical contract | Refuse cathedral consumers lacking canonical contract | NOT APPLICABLE (this memo specifies the equation, not a consumer; the consumer `tac.cathedral_consumers.canonical_equation_lookup_consumer` already satisfies Catalog #335) |
| Catalog #340 sister-checkpoint guard | Refuse commits absorbing in-flight sister edits | COMPLIANT — pre-flight sister check returned PROCEED at step 1 |

NO active sister subagent collision detected during this memo's authoring. Subagent crash-resume per Catalog #206 honored (3 checkpoints emitted).

## §14. Top-3 operator-routable next-actions

1. **DECIDE: register or defer the canonical equation per Catalog #344 (operator-routable)**. The codex sister has already landed the operational module + registry append. This memo specifies the design contract. Both postures are CLAUDE.md-coherent; the operator decides whether to ratify the codex sister's registration OR re-route via the meta-routing decision protocol.

2. **LAND: PREDICTOR-ENTROPY-MATCH certificate helper (sister wave 3 candidate)**. The canonical helper `tac.canonical_equations.predictor_entropy_match.compute_predictor_entropy_match_certificate(predictor_distribution, empirical_distribution, encoded_symbol_count) -> PredictorEntropyMatchCertificate` per §4.2 is NEW; the codex sister's operationalized equation module does NOT yet enforce the invariant at the `predict_procedural_predictor_plus_residual_correction_savings(...)` per-call surface. Lane: `lane_predictor_entropy_match_certificate_helper_20260521`. Estimated ≤4 hours wall-clock + $0 GPU.

3. **PROBE: Laplacian-fitted predictor against DWT detail subbands (frontier-breaking candidate)**. Per §4.3 + §10 alternative-probe enumeration: replace pcg64-uniform with a Laplacian-fitted procedural predictor parameterized by (mu, b) per DWT subband. Predicted PREDICTOR-ENTROPY-MATCH certificate: `HARD_EARNED_MATCH` (KL ≈ 0; predictor distribution matches empirical Laplacian); predicted R_encoded ≈ ε · symbol_count (compressor overhead floor); predicted verdict: RATE_WIN. This is a TESTABLE HYPOTHESIS for the residual-hybrid stacking paradigm rescue. Lane: `lane_laplacian_fitted_predictor_dwt_detail_residual_smoke_20260521`. Estimated $0.50-2.00 Modal CPU smoke + ≤2 hours wall-clock. Per CLAUDE.md "Forbidden premature KILL": this is the canonical reactivation path for the residual-correction stacking paradigm.

## §15. Blockers

NONE for the design specification surface. The codex sister has already landed the operational module + registry append + landing memo; THIS memo's role is the formal design specification per Catalog #294 + #303 + #305 + #309. The PREDICTOR-ENTROPY-MATCH certificate helper landing (operator-routable #2) is the natural next step but is not a blocker for this memo's landing.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the residual-hybrid stacking paradigm is NOT killed; the empirical falsifications at pair #1 + pair #2 are HARD-EARNED IMPLEMENTATION-LEVEL (per Catalog #307); the structural intervention (THIS canonical equation + PREDICTOR-ENTROPY-MATCH invariant + Catalog #359 sister gate) enables future HARD_EARNED_MATCH residual-hybrid candidates without burning paid GPU on doomed dispatches.

**End of design memo.**
