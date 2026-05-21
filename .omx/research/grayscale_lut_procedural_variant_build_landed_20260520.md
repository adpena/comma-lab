# WAVE-3-GRAYSCALE-LUT-PROCEDURAL-TRAINER-BUILD landing memo

**Date (UTC):** 2026-05-21T01:25:00Z
**Lane:** `lane_grayscale_lut_procedural_codebook_replacement_variant_20260520` (L1; impl_complete + strict_preflight)
**Subagent:** `wave-3-grayscale-lut-procedural-trainer-build-20260520`
**Predicted ΔS:** `-0.000149` per canonical equation #26 closed-form `-25 × (256 - 32) / 37_545_489` `[prediction]`
**Mission contribution per Catalog #300:** `frontier_protecting`

## Headline

BUILD landed the 3rd canonical PROCEDURAL VARIANT module — `tac.substrates.grayscale_lut.distillation_procedural_variant` — following the canonical pattern established by DP1 (commit `9cbfa471c`) and VQ-VAE (commit `6fea30f22`). Sister API surface; 30/30 dedicated tests PASS; 39/39 grayscale_lut regression tests PASS; Catalog #272 byte-mutation smoke 32/32 PASS. Canonical equation #26 IN-DOMAIN context = `chroma_lut_replacement` (strongest IN-DOMAIN fit per CANONICAL EQUATION #26 DOMAIN REFINEMENT commit `8d8a7c6c5`).

## 9-dimension success checklist evidence

1. **UNIQUENESS** — Variant context `chroma_lut_replacement` is structurally distinct from DP1's `comma2k19_ood_derived_basis_replacement` (PCA basis) + VQ-VAE's `intermediate_transform_quantizer` (K×D embedding table). All three are members of canonical equation #26 `_INCLUDED_CONTEXTS` but address different LUT-target sub-surfaces.
2. **BEAUTY + ELEGANCE** — Sister-structured module matching DP1/VQ-VAE (Config + public functions + one envelope helper + canonical constants). Convenience wrapper `compose_procedural_archive` in `archive.py` delegates to full API for byte-stability + canonical defaults.
3. **DISTINCTNESS** — Distinguishing feature: `PROCEDURAL_LUT_SENTINEL = b"GLPV"` envelope appended to GLV1 archives (vs DP1's in-place codebook section replacement + VQ-VAE's `b"VQVP"` sentinel-prefix inside the decoder blob). The architectural reality of the current grayscale_lut substrate (FiLM-conditioned RGB decoder rather than explicit chroma LUT section) is documented honestly per CLAUDE.md "Apples-to-apples evidence discipline".
4. **RIGOR** — Premise verification per Catalog #229: read full DP1 + VQ-VAE canonical reference patterns (sister files in full) + grayscale_lut substrate code + canonical equation #26 `_INCLUDED_CONTEXTS` + `validate_context_is_in_domain` helper BEFORE writing the variant. After Codex adversarial correction, all 30 dedicated tests pass; canonical equation predicted ΔS = -0.000149 verified analytically + computationally; byte-mutation smoke 32/32 distinct.
5. **OPTIMIZATION PER TECHNIQUE** — Per Catalog #290 canonical-vs-unique decision per layer:
   - Forked: archive grammar (GLPV sentinel envelope is grayscale_lut-unique; DP1 replaces dp1_codebook section, VQ-VAE prepends VQVP inside decoder_blob — neither pattern fits the current GLV1 grammar which lacks a dedicated chroma_lut section).
   - Adopted canonical: ProceduralVariantConfig dataclass shape; PROCEDURAL_SEED_SIZE_BYTES=32 constant; PCG64 default generator_kind; `derive_codebook_from_seed` helper; `verify_*_in_domain` slot-3-aware pattern with fallback set; `verify_seed_mutation_changes_*_bytes` Catalog #272 invariant; `predicted_archive_bytes_saved` + `predicted_delta_s` canonical equation #26 closed-form.
   - Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": forks were principled (architectural reality of current substrate) not cargo-culted.
6. **STACK-OF-STACKS-COMPOSABILITY** — Procedural variant is structurally additive over the canonical GLV1 archive (compose function preserves all original bytes byte-for-byte + appends envelope; sister `parse_archive` on the GLV1 prefix still returns canonical archive structure, while `parse_archive` correctly rejects the appended scaffold archive as non-GLV1). Future LUT-aware variant lands as GLV2 schema bump per the design memo §honest disclosure.
7. **DETERMINISTIC REPRODUCIBILITY** — `derive_procedural_lut_replacement` is deterministic per seed + generator_kind + shape + dtype; envelope encoding is byte-stable; brotli quality 9 fixed; sorted-keys pattern inherited from DP1/VQ-VAE sister discipline.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Smallest ΔS per substrate in the 5-substrate procedural matrix (-0.000149 vs DP1 -0.002706 vs VQ-VAE -0.005434) but strongest IN-DOMAIN fit confidence: `chroma_lut_replacement` is the canonical equation #26 anchor context per the DOMAIN REFINEMENT commit. The structural value is the IN-DOMAIN confidence multiplier, not absolute byte savings.
9. **OPTIMAL MINIMAL CONTEST SCORE** — Per Catalog #324, `predicted_band_validation_status=pending_post_training`; the predicted ΔS is HYPOTHESIS not score claim. Operator-routed per-substrate symposium per Catalog #325 + paired-smoke contest-CUDA + contest-CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" are required gates before promotion.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| `ProceduralVariantConfig` dataclass | ADOPT_CANONICAL_BECAUSE_SERVES | Sister DP1/VQ-VAE pattern; same invariants apply |
| `PROCEDURAL_SEED_SIZE_BYTES=32` | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical equation #26 K_seed term |
| `PROCEDURAL_LUT_BYTES_DEFAULT=256` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Canonical chroma_lut_replacement N=256 (vs DP1 4096 / VQ-VAE 8192) |
| `CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT="chroma_lut_replacement"` | FORK_BECAUSE_PRINCIPLED_MISMATCH | grayscale_lut's canonical context per equation #26 _INCLUDED_CONTEXTS |
| `PROCEDURAL_LUT_SENTINEL=b"GLPV"` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Substrate-unique sentinel (DP1 has no sentinel; VQ-VAE uses VQVP) |
| Archive composition (APPEND envelope) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Current GLV1 lacks chroma_lut section to replace; honest L0 scaffold |
| `derive_codebook_from_seed` canonical helper | ADOPT_CANONICAL_BECAUSE_SERVES | Single source of truth for PRNG-based codebook derivation |
| `validate_context_is_in_domain` sister helper | ADOPT_CANONICAL_BECAUSE_SERVES | Slot 3 sister helper already landed; we delegate to it |
| `predicted_delta_s` closed-form formula | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical equation #26 exact formula |
| Test structure | ADOPT_CANONICAL_BECAUSE_SERVES | Sister VQ-VAE 21-test pattern adapted to 26 grayscale_lut-specific tests |

## Predicted ΔS band

Predicted ΔS = `-0.000149` (closed-form from canonical equation #26 IN-DOMAIN context `chroma_lut_replacement`).

Dykstra-feasibility check per Catalog #296: the predicted band sits within the canonical equation #26 domain-of-validity (`seed_size_bytes_range=[8, 256]` + IN-DOMAIN context). The rate-axis savings ΔS = -0.000149 contributes additively to the Pareto polytope; the per-substrate symposium per Catalog #325 must validate whether `predicted_d_seg_delta` + `predicted_d_pose_delta` remain ≈0 (the canonical-helper-only-bytes-changed assumption per the bytes-saved structural delta). First-principles citation: canonical equation #26 derives from Shannon R(D) bound on the procedural-PRNG output distribution + the contest rate formula `25 * archive_bytes / 37_545_489`.

`predicted_band_validation_status: pending_post_training` per Catalog #324; reactivation criterion = post-training Tier-C re-measurement on landed paired-smoke archive sha256 via `tools/mdl_scorer_conditional_ablation.py --tier c`.

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| Canonical equation #26 closed-form predicts savings exactly | HARD-EARNED | Closed-form is exact bytes_saved formula; rate term IS `25*bytes/denom` per `upstream/evaluate.py:63` |
| `chroma_lut_replacement` IS strongest IN-DOMAIN fit for grayscale_lut | HARD-EARNED | CANONICAL EQUATION #26 DOMAIN REFINEMENT commit `8d8a7c6c5` registered this context explicitly |
| Current grayscale_lut substrate has a 256-byte chroma LUT | CARGO-CULTED — UNWOUND | Honest audit: current substrate uses FiLM-conditioned RGB decoder; no explicit 256-byte LUT exists yet. Compose function APPENDS envelope rather than replacing (L0 scaffold honest disclosure). Future GLV2 grammar lands the chroma_lut section. |
| Predicted ΔS = score CLAIM | CARGO-CULTED — UNWOUND | predicted_delta_s is HYPOTHESIS per Catalog #287; symposium + paired-smoke required before promotion |
| Per-pixel-independence of chroma LUT bytes | CARGO-CULTED — INHERITED | Inherited from canonical equation #26 derivation; the IN-DOMAIN context's domain-of-validity covers it. Future per-substrate symposium per Catalog #325 evaluates whether the inherited assumption empirically holds on the grayscale_lut substrate. |
| Brotli quality 9 is optimal for procedural seed envelope | CARGO-CULTED — INHERITED | Inherited from canonical sister patterns (DP1 + VQ-VAE both use quality 9). Per-substrate optimization is a future engineering pass; the L0 scaffold matches sister discipline. |

## Observability surface

1. **Inspectable per layer** — `ProceduralVariantConfig`, `derive_procedural_lut_replacement`, `compose_with_procedural_lut`, `_build_procedural_seed_envelope`, `verify_procedural_lut_in_domain`, `verify_seed_mutation_changes_lut_bytes` all callable independently; each layer's input + output inspectable at any byte boundary.
2. **Decomposable per signal** — Predicted ΔS decomposes via `predicted_archive_bytes_saved(lut_bytes, seed_bytes)` × `-25 / 37_545_489`; bytes_saved decomposes into `lut_bytes - seed_bytes`; envelope decomposes into `SENTINEL(4) + LUT_BYTES(u16) + GEN_KIND_TAG(u8) + SEED_LEN(u16) + seed`.
3. **Diff-able across runs** — Determinism (same seed → same LUT bytes) verified by `test_derive_procedural_lut_replacement_deterministic`; envelope bytes diff-able byte-for-byte across (seed, generator_kind, lut_bytes) tuples.
4. **Queryable post-hoc** — Predicted savings + IN-DOMAIN context computable from any (lut_bytes, seed_bytes) input pair; canonical envelope re-parseable via the sentinel + struct layout documented in the module docstring + `_build_procedural_seed_envelope` body.
5. **Cite-able** — Module docstring cites: PR101/PR106 BUILD DESIGN commit `086d3ac1d` + DP1 sister commit `9cbfa471c` + VQ-VAE sister commit `6fea30f22` + canonical equation #26 source module + CANONICAL EQUATION #26 DOMAIN REFINEMENT commit `8d8a7c6c5`.
6. **Counterfactual-able** — `verify_seed_mutation_changes_lut_bytes` proves single-byte counterfactuals propagate; the canonical Catalog #272 byte-mutation smoke (32/32 bytes produce distinct outputs) verifies the operational mechanism.

## Sister regression

| Test suite | Before | After |
|---|---|---|
| `src/tac/substrates/grayscale_lut/tests/` (9 existing tests) | 9 PASS | 9 PASS |
| `src/tac/substrates/grayscale_lut/tests/test_procedural_variant.py` (NEW; 30 tests) | n/a | 30 PASS |
| **Total grayscale_lut** | 9 PASS / 1 skip | **39 PASS** |

Sister regression of DP1 + VQ-VAE substrates not run by this subagent (out of scope). The new variant module is structurally additive: no canonical helper modified; no sister substrate touched.

## 6-hook wire-in declaration per Catalog #125

* **Hook #1 sensitivity-map** = N/A (variant is single archive-build path; no per-tensor sensitivity contribution).
* **Hook #2 Pareto constraint** = ACTIVE via `procedural_codebook_savings_v1` predicted ΔS contribution to rate-axis Pareto polytope. Smallest per-substrate ΔS in the 5-substrate procedural matrix; strongest IN-DOMAIN confidence.
* **Hook #3 bit-allocator** = PLANNED (32-byte seed slot replaces 256-byte LUT slot only when GLV2 lands; current GLV1 scaffold appends bytes and is not score-eligible).
* **Hook #4 cathedral autopilot dispatch** = ACTIVE via sister consumer `tac.cathedral_consumers.procedural_codebook_generator_consumer` (auto-discovered per Catalog #335).
* **Hook #5 continual-learning posterior** = ACTIVE (first empirical anchor via `update_equation_with_empirical_anchor` post-paired-smoke).
* **Hook #6 probe-disambiguator** = ACTIVE (PROCEDURAL vs canonical chroma LUT contrast IS the probe disambiguator for whether the grayscale_lut substrate's chroma-axis rate can be procedurally substituted within the canonical equation #26 IN-DOMAIN context).

## Sister coordination

* **Sister-DISJOINT** from in-flight PR101 GOLD NULL-BYTE REMOVAL SMOKE (`a3dfc84c`) — different substrate file scope (PR101 vs grayscale_lut).
* **Sister-DISJOINT** from in-flight PACT-NERV-DistilledScorer × Codex LL INTEGRATION DESIGN (`ad501b52`) — different substrate file scope (PACT-NERV vs grayscale_lut).
* **Lane registry shared file** (`.omx/state/lane_registry.json`) modified via canonical `tools/lane_maturity.py` fcntl-locked CLI per Catalog #131 sister discipline. The 18 sister commits touching the registry in the 12-hour lookback all used the canonical mutation path.

## Catalog gate verdicts

| Gate | Verdict | Notes |
|---|---|---|
| Catalog #1 `check_no_mps_fallback_default` | PASS | No MPS fallback in new code |
| Catalog #272 `check_substrate_distinguishing_feature_integration_contract` | PASS | Byte-mutation smoke 32/32 distinct |
| Catalog #287 `check_no_docstring_overstatement_without_evidence_tag` | PASS | All claims tagged `[prediction]` or canonical-source-cited |
| Catalog #297 `check_substrate_signal_axis_destruction_has_reversibility_probe` | PASS | No signal-axis destruction (procedural variant is rate-axis only) |
| Catalog #323 `check_no_score_claim_without_canonical_provenance` | PASS | predicted_delta_s explicitly NOT a score claim per docstring |
| Catalog #324 `check_no_predicted_band_without_post_training_tier_c_validation` | PASS | `predicted_band_validation_status=pending_post_training` declared |

## Operator-routable next-actions (Top-3)

1. **Per-substrate symposium per Catalog #325** (estimated ~1 hour operator-attended; T2 sextet pact) — the gate is required before paid GPU dispatch on this lane. Canonical 6-step contract per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable.
2. **Paired-smoke contest-CUDA + contest-CPU after symposium PROCEED** — Modal A10G ~$0.50 estimated; sister to the canonical DP1 paired-smoke pre-dispatch design (`feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md`). Verifies the predicted ΔS = -0.000149 empirically + populates `predicted_band_validation_status=validated_post_training` per Catalog #324.
3. **3-substrate aggregate sequencing** — DP1 + VQ-VAE + grayscale_lut as a coherent procedural-variant batch. Per the META-Lagrangian solver: aggregate predicted ΔS = -0.002706 + -0.005434 + -0.000149 = -0.008289 (rate-axis only; assumes additivity per Pareto polytope). The 3-substrate symposium + 3-paired-smoke batch is the canonical operator-routed cadence.

## Blockers

No blocker to committing as `research_only=true` scaffold. All 30 dedicated tests pass; 39/39 grayscale_lut regression pass; 0 new catalog violations from my work. Module is L1 SCAFFOLD per impl_complete + strict_preflight gates marked.

Score/promotion blocker remains explicit: current GLV1 has no chroma-LUT section, so the compose function APPENDS the envelope and increases bytes by 41 B for a 32 B seed. A score-eligible archive requires GLV2 LUT-explicit grammar + inflate consumer before the `-0.000149` rate-axis hypothesis can be tested.

The compose function APPENDS the envelope rather than REPLACING (current GLV1 grammar lacks chroma_lut section; honest L0 scaffold disclosure). A future LUT-aware variant lands at GLV2 schema bump + the compose function flips to REPLACE-IN-PLACE matching the canonical DP1 + VQ-VAE pattern. This is documented in the module docstring under "Architectural note" + the compose function docstring.

## Cross-references

* PR101/PR106 BUILD DESIGN landing commit `086d3ac1d` (Top-3 #1 PIVOT recommendation rationale)
* DP1 PROCEDURAL TRAINER BUILD landing memo `feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md` + commit `9cbfa471c`
* VQ-VAE PROCEDURAL VARIANT BUILD landing memo `feedback_slot_mg7_bundle_master_gradient_exploits_landed_20260520.md` sister + commit `6fea30f22`
* Canonical equation #26 source: `src/tac/canonical_equations/procedural_codebook_savings.py`
* CANONICAL EQUATION #26 DOMAIN REFINEMENT commit `8d8a7c6c5` (`chroma_lut_replacement` IN-DOMAIN context registration)
* CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable
* CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable (`research_only=true` declared in lane notes)
* CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable
* Catalog #220 / #240 / #272 / #287 / #294 / #303 / #305 / #309 / #323 / #324 / #325 / #335 / #344
