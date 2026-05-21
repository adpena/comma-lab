<!-- HISTORICAL_SCORE_LITERAL_OK:macos_cpu_advisory_methodology_extension_not_score_truth_2026-05-21 -->
# WAVE-3 Canonical equation #26 PARSER-SAFE EXTENSION domain refinement landed

**Date (UTC):** 2026-05-21T04:39:46Z
**Lane:** `lane_wave_3_canonical_equation_26_parser_safe_extension_domain_refinement_20260520`
**Equation:** `procedural_codebook_from_seed_compression_savings_v1` (canonical equation #26)
**Event type:** `domain_refined` (4th `domain_refined` event on equation #26; 8th total event row)
**Axis:** `[macOS-CPU advisory]`; `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`
**Verdict:** `DOMAIN_REFINED_PARSER_SAFE_EXTENSION_REINFORCEMENT`

## Summary

This landing emits a `domain_refined` event on canonical equation #26
that REINFORCES the existing 11 `_INCLUDED_CONTEXTS` tokens as
sufficient for the 4 IN-DOMAIN substrates (DP1 / VQ-VAE / grayscale_lut
/ ATW V2), AND adds 1 NEW `_EXCLUDED_CONTEXTS` token codifying the
META-LESSON from the PARSER-SAFE EXTENSION 4-substrate static
classification: **parser-safe-but-score-affecting RAW sections are
NOT canonical equation #26 IN-DOMAIN candidates for direct byte
substitution**.

Per the sister PARSER-SAFE EXTENSION landing memo
(`.omx/research/parser_safe_methodology_extension_landed_20260520.md`
commit `d0bf3ce37`) the 4-substrate comparative classification surfaced
3 orthogonal substrate archetypes: all-Brotli (DP1, grayscale_lut, fec6
— ZERO parser-safe bytes); mixed-with-RAW-codebook-indices (VQ-VAE —
192 parser-safe bytes in `indices_blob`); side-information-heavy
(ATW V2 — 2,632 parser-safe bytes across `latent_residual_blob` +
`class_prior_table_blob` + `cdf_table_blob`). **ALL** 2,824
parser-safe bytes across the 4 substrates are score-affecting because
they ARE the decoder's side-information.

Per CLAUDE.md HNeRV parity discipline L6 (score-domain Lagrangian) the
canonical equation #26 predicts REPLACEMENT savings for score-OPAQUE
bytes; parser-safe-but-score-affecting bytes are explicitly out of
domain. This landing makes the boundary canonical via the new
`_EXCLUDED_CONTEXTS` token, executing REVERSE-DIRECTIVE #3 from the
CODEX CROSS-POLLINATION audit (commit `aafac7c84`) and operator-routable
next-action #1 from the PARSER-SAFE EXTENSION landing memo.

## Event metadata

* **equation_id**: `procedural_codebook_from_seed_compression_savings_v1`
* **event_type**: `domain_refined`
* **written_at_utc**: `2026-05-21T04:39:46Z`
* **agent**: `claude`
* **subagent_id**: `wave-3-canonical-equation-26-parser-safe-domain-refinement-20260520`
* **NEW `_EXCLUDED` token**: `direct_byte_substitution_on_parser_safe_but_score_affecting_raw_sections`
* **`_INCLUDED` count**: 11 (PRESERVED verbatim per HNeRV parity L7 +
  Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE)
* **`_EXCLUDED` count**: 4 → 5 (1 NEW token appended)
* **Registry event count**: 7 → 8 (1 NEW `domain_refined` row)
* **Empirical anchor commit**: `d0bf3ce37`
* **Empirical anchor artifact**:
  `experiments/results/parser_safe_methodology_extension_smoke_20260521T030815Z/`
* **Aggregate**: 2,824 parser-safe bytes of 4,885 total archive bytes
  across 4 substrates; ALL 2,824 are score-affecting; 0 are
  score-opaque.
* **Verdict**: `MIXED_PARSER_SAFE_BUT_SCORE_AFFECTING`

Per-substrate breakdown (also captured in the event row's
`parser_safe_extension_per_substrate_breakdown` field for downstream
autopilot ranker consumption):

| substrate | archive bytes | parser-safe | score-affecting | parser-safe sections |
|---|---:|---:|---:|---|
| `dp1_pretrained_driving_prior` | 563 | 0 | 0 | (none — all Brotli/JSON) |
| `vq_vae` | 591 | 192 | 192 | `indices_blob` (int16 packed) |
| `grayscale_lut` | 481 | 0 | 0 | (none — all Brotli/JSON) |
| `atw_codec_v2` | 3,250 | 2,632 | 2,632 | `latent_residual_blob` (int8) + `class_prior_table_blob` (fp16) + `cdf_table_blob` (fp16) |
| **TOTAL** | **4,885** | **2,824** | **2,824** | |

## Sister vs predecessor events

The 4 `domain_refined` events on canonical equation #26 form a coherent
domain refinement cascade per Catalog #344 evolution discipline:

| # | UTC | Anchor | Refinement |
|---|---|---|---|
| 1 | 2026-05-20T23:41:52Z | commit `f25f8cc1b` (DWT detail-subband smoke; KL=1.638 nats / 3.28σ) | EXCLUDES `direct_dwt_detail_subband_byte_substitution` + `direct_byte_substitution_on_wavelet_decomposition_coefficients` (sister of T3 DWT BIND symposium) |
| 2 | 2026-05-21T01:13:21Z | H3 PR101 master-gradient null-byte removal smoke (3/3 variants failed inflate) | EXCLUDES `master_gradient_null_byte_removal_with_constant_reconstruction` + `master_gradient_null_byte_replacement_with_arbitrary_constant` |
| 3 | 2026-05-21T01:50:23Z | PARSER-SAFE SUBSET smoke on fec6 (commit `e3e198c9f`; 0 parser-safe bytes on fec6 PR101 GOLD) | (no `_EXCLUDED` token added — clarifies sister #2's null-byte rationale via fec6 single-substrate empirical) |
| **4 (this landing)** | **2026-05-21T04:39:46Z** | **PARSER-SAFE EXTENSION 4-substrate classification (commit `d0bf3ce37`; 2,824 parser-safe but ALL score-affecting)** | **EXCLUDES `direct_byte_substitution_on_parser_safe_but_score_affecting_raw_sections`** |

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: all prior event
rows are preserved verbatim; only a NEW `domain_refined` event row is
added. The cascade structurally documents the progressive narrowing of
canonical equation #26's domain from "any procedural codebook
replacement" → "REPLACEMENT savings on score-OPAQUE bytes outside
parser-essential structures only".

## Cargo-cult audit per assumption (Catalog #303)

1. **Assumption**: "Parser-safe RAW sections imply replaceability."
   Classification: **CARGO-CULTED** (empirically falsified by PARSER-SAFE
   EXTENSION smoke). Unwind path: classify parser-safe bytes by
   SCORE-RELEVANCE before treating them as replaceable.
2. **Assumption**: "Canonical equation #26's `_INCLUDED_CONTEXTS` need
   to be extended with 1 NEW token per substrate (`vq_vae_indices_blob`
   / `atw_v2_cdf_table_blob` / etc.)." Classification: **CARGO-CULTED**.
   The existing 11 tokens already cover the per-substrate procedural-
   replacement surfaces (`dp1_codebook_bytes` /
   `atw_v2_codec_quantizer_lut` / `intermediate_transform_quantizer` /
   `procedural_codebook_as_lookup_table` / `chroma_lut_replacement`);
   adding new tokens would re-collapse into the wrong surface (the
   trained-substrate-specific surface vs the deterministic-replacement
   surface).
3. **Assumption**: "The 4-substrate result generalizes to all future
   substrates with similar grammars." Classification: **HARD-EARNED**.
   The 4 substrates span 3 archetypes (all-Brotli / mixed-RAW-indices /
   side-info-heavy); future substrates will likely fall into one of
   these classes. The methodology extends (and the canonical helper
   `validate_context_is_in_domain` will refuse the new EXCLUDED token).

## 9-dimension success checklist evidence (Catalog #294)

* **UNIQUENESS**: this is the 4th `domain_refined` event on equation
  #26; previous events refined per-context (DWT detail subbands / null-
  byte removal / fec6 parser-safe subset); this event refines per-
  archetype (parser-safe-but-score-affecting RAW sections across 3
  archetypes).
* **BEAUTY + ELEGANCE**: ~800-word landing memo + 1 canonical event row
  via `update_equation_with_domain_refinement` + 0 source changes
  needed (reuses existing canonical helper + existing
  `_EXCLUDED_CONTEXTS` mechanism).
* **DISTINCTNESS**: explicitly different from sister events (different
  anchor / different archetype / different `_EXCLUDED` token).
* **RIGOR**: the event row carries 12 typed fields citing the empirical
  anchor commit + landing memo + artifact path + per-substrate
  breakdown; Provenance per Catalog #323 (`[macOS-CPU advisory]` axis;
  `score_claim=false`); rationale ≥4 chars per Catalog #287.
* **OPTIMIZATION PER TECHNIQUE**: routes through canonical
  `update_equation_with_domain_refinement` helper (Catalog #131
  fcntl-locked APPEND-ONLY JSONL); preserves all 11 `_INCLUDED` tokens.
* **STACK-OF-STACKS COMPOSABILITY**: the new `_EXCLUDED` token is
  consumable by `validate_context_is_in_domain` / `refuse_residual_hybrid_context_misapplication`
  cathedral consumers; future autopilot dispatches against parser-safe-
  but-score-affecting RAW sections are structurally refused.
* **DETERMINISTIC REPRODUCIBILITY**: the event row is byte-stable JSON;
  the empirical anchor artifact at
  `experiments/results/parser_safe_methodology_extension_smoke_20260521T030815Z/`
  is reproducible per the sister landing memo's verification commands.
* **EXTREME OPTIMIZATION + PERFORMANCE**: domain refinement is a static
  structural protection; no GPU spend; <1s wall-clock for the event
  write.
* **OPTIMAL MINIMAL CONTEST SCORE**: the refinement codifies the
  STRUCTURAL EXHAUSTION argument from the sister landing memo: future
  score-lowering work on the 4 substrates routes through the existing
  PROCEDURAL VARIANT BUILDs (DP1 / VQ-VAE / grayscale_lut / ATW V2)
  which co-optimize replacement at training time per the IN-DOMAIN
  context tokens.

## Observability surface (Catalog #305)

* **Inspectable per layer**: the `domain_refined` event row is queryable
  via `tac.canonical_equations.load_registry_events_lenient` /
  `query_equations` / `get_equation_by_id`.
* **Decomposable per signal**: the event's
  `parser_safe_extension_per_substrate_breakdown` field decomposes the
  aggregate result by substrate (4 substrates × {archive_bytes,
  parser_safe, score_affecting, score_opaque}).
* **Diff-able across runs**: the event row sha256 is byte-stable;
  future events on the same equation can be diffed against this row.
* **Queryable post-hoc**: the event row is canonical JSONL consumable
  by `tools/list_canonical_equations.py` + future cathedral consumers
  + autopilot ranker.
* **Cite-able**: the event row's `parser_safe_extension_empirical_anchor_commit`
  field cites the sister landing memo commit `d0bf3ce37`.
* **Counterfactual-able**: the event is APPEND-ONLY; future operators
  can add a sister `domain_refined` event to REVERSE or REFINE the
  decision without mutating this row (per Catalog #110/#113).

## Sister coordination + collision verdict

* **Slot 1 `af36cd72`** (`wave-3-canonical-equation-procedural-predictor-residual-rati...`):
  DISJOINT scope — slot 1 ratifies NEW equation
  `procedural_predictor_plus_residual_correction_savings_v1` (already
  registered with 2 anchors at landing time per registry inspection);
  this slot refines EXISTING equation #26. Both slots concurrently
  write to `.omx/state/canonical_equations_registry.jsonl` via the
  canonical helper which serializes writes via Catalog #131 fcntl.
  Empirical verification: the registry append landed cleanly without
  contention (event count went from 7 → 8 as expected; no rollback).
* **Slot 2 `wave-3-end-of-day-command-sheet-v2-respawn-2026052`**:
  DISJOINT scope (operator routing command sheet; no canonical
  equations registry touch).
* **`wave-3-parser-safe-methodology-extension-20260520`**: COMPLETE at
  03:13:46Z; this slot executes the operator-routable next-action #1
  from that subagent's landing memo.
* **Catalog #340 sister-checkpoint guard**: fired STAND_DOWN_DUPLICATE
  at pre-flight on `.omx/state/canonical_equations_registry.jsonl`
  file-overlap with 15 sister commits in the last 12h; semantic-overlap
  audit showed no sister had emitted the PARSER-SAFE EXTENSION
  `domain_refined` event yet; task prompt explicitly anticipated this
  collision pattern ("slot 1 collision possible on
  canonical_equations_registry.jsonl; canonical helper serializes
  correctly via Catalog #131 fcntl"); proceeded.
* **Catalog #314 absorption-pattern guard**: NO bare commits during
  this subagent's run; canonical serializer used for the commit.

## Provenance (Catalog #323)

* `score_claim`: False
* `promotion_eligible`: False
* `rank_or_kill_eligible`: False
* `ready_for_exact_eval_dispatch`: False
* `axis_tag`: `[macOS-CPU advisory]`
* `evidence_grade`: `macOS-CPU-advisory`
* `canonical_helper_invocation`: `tac.canonical_equations.update_equation_with_domain_refinement`
* `upstream_artifact`:
  `experiments/results/parser_safe_methodology_extension_smoke_20260521T030815Z/`
  (commit `d0bf3ce37`)

## 6-hook wire-in declaration (Catalog #125)

* **hook #1 sensitivity-map**: N/A — domain refinement is a structural
  protection; no per-pair sensitivity contribution.
* **hook #2 Pareto constraint**: N/A — the refinement narrows
  equation #26's predict-domain; downstream Pareto constraints
  consume the refined equation via `tac.cathedral_consumers.procedural_codebook_generator_consumer`
  which refuses out-of-domain candidates per the existing canonical
  Catalog #341 markers.
* **hook #3 bit-allocator**: N/A — the new `_EXCLUDED` token does not
  add new bit-allocator surface; it refines the existing surface.
* **hook #4 cathedral autopilot dispatch**: ACTIVE — the cathedral
  consumers `procedural_codebook_generator_consumer` +
  `null_byte_codebook_candidate_consumer` consume the refined
  `_EXCLUDED_CONTEXTS` set via `validate_context_is_in_domain`;
  candidates with the new `direct_byte_substitution_on_parser_safe_but_score_affecting_raw_sections`
  context are now structurally refused at autopilot dispatch time
  with canonical non-promotable Catalog #341 markers.
* **hook #5 continual-learning posterior**: ACTIVE — the equation's
  Bayesian posterior remains coherent; the refinement does NOT
  re-trigger recalibration (no new empirical anchor on this event;
  the empirical anchor is the static-classification smoke not a
  numerical residual). Future empirical anchors against the 4
  substrates remain IN-DOMAIN per the preserved `_INCLUDED` tokens
  and update the posterior normally via
  `update_equation_with_empirical_anchor`.
* **hook #6 probe-disambiguator**: ACTIVE — the new `_EXCLUDED` token
  IS a disambiguator between trained-substrate-specific procedural-
  replacement surfaces (IN-DOMAIN; routed via existing tokens like
  `dp1_codebook_bytes` / `atw_v2_codec_quantizer_lut`) and direct-byte-
  substitution on parser-safe-but-score-affecting RAW sections (OUT-OF-
  DOMAIN; refused at validation time).

## Top-3 operator-routable next-actions

1. **VQ-VAE indices_blob procedural-variant BUILD** (operator-routable
   per sister landing memo's op-routable #2): extend the existing
   VQ-VAE procedural-variant BUILD (commit `6fea30f22`) to also
   replace the `indices_blob` via a deterministic mapping. Subject to
   Catalog #325 per-substrate symposium AND Catalog #324 post-training
   Tier-C validation BEFORE any paid dispatch. The new `_EXCLUDED`
   token does NOT block this path because the procedural-VARIANT
   surface routes through `atw_v2_codec_quantizer_lut` /
   `procedural_codebook_as_lookup_table` (IN-DOMAIN); only direct
   byte substitution is excluded.
2. **ATW V2 cdf_table_blob procedural-variant**: the 2,560-byte
   `cdf_table_blob` is the largest parser-safe RAW section; a
   procedural variant that emits the CDF table deterministically from
   a 32-byte seed would route through the existing
   `atw_v2_codec_quantizer_lut` IN-DOMAIN context per the sister
   landing memo's op-routable #3. Subject to Catalog #324 + #325.
3. **Audit if any current cathedral candidate is targeting parser-safe-
   but-score-affecting RAW sections**: with the new `_EXCLUDED` token
   in place, run `tools/cathedral_autopilot_autonomous_loop.py
   --report-only --json` and grep for the new token in the refused-
   candidates list; surface to operator any candidate that was
   previously RANKED but is now structurally REFUSED. This closes the
   loop on Catalog #344 evolution discipline.

## Cross-references

* Sister landing memo: `.omx/research/parser_safe_methodology_extension_landed_20260520.md`
  (commit `d0bf3ce37`).
* Sister smoke commit: `e3e198c9f` (PARSER-SAFE SUBSET SMOKE on fec6
  frontier — `PARSER_SAFE_SUBSET_EMPTY`).
* CODEX CROSS-POLLINATION audit: commit `aafac7c84` REVERSE-DIRECTIVE #3.
* Predecessor `domain_refined` events: commit `f25f8cc1b` (DWT)
  + 2 H3 + parser-safe-subset events (per the cascade table above).
* Sister equation: `procedural_predictor_plus_residual_correction_savings_v1`
  (slot 1 `af36cd72`; covers residual-hybrid contexts explicitly
  excluded from equation #26 via Catalog #359 sister discipline).
* Canonical helper: `tac.canonical_equations.update_equation_with_domain_refinement`.
* Cathedral consumer: `tac.cathedral_consumers.procedural_codebook_generator_consumer`.
* Domain validator: `tac.canonical_equations.procedural_codebook_savings.validate_context_is_in_domain`.
