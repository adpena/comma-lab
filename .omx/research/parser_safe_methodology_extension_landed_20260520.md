<!-- HISTORICAL_SCORE_LITERAL_OK:macos_cpu_advisory_methodology_extension_not_score_truth_2026-05-20 -->
# WAVE-3 PARSER-SAFE METHODOLOGY EXTENSION landed

**Date (UTC):** 2026-05-21T03:10:00Z
**Lane:** `lane_wave_3_parser_safe_methodology_extension_20260520`
**Artifact:** `experiments/results/parser_safe_methodology_extension_smoke_20260521T030815Z/`
**Verdict:** `METHODOLOGY_EXTENSION_MIXED_PARSER_SAFE_BUT_SCORE_AFFECTING`
**Axis:** `[macOS-CPU advisory]`; `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`

## Summary

Extension of the PARSER-SAFE SUBSET SMOKE methodology (sister commit
`e3e198c9f`) from the single-substrate fec6 PR101 frontier to 4 canonical
IN-DOMAIN substrates: DP1 (pretrained_driving_prior) + VQ-VAE +
grayscale_lut + ATW V2. Sister smoke established the META-LESSON that
null-gradient is NECESSARY but NOT SUFFICIENT for byte replaceability:
replaceability also requires the byte be downstream of parser dispatch
(NOT inside a Brotli/LZMA/Huffman bitstream or a struct-packed wrapper
field). This extension generalizes the methodology to a comparative
4-substrate static classification.

The comparative result:

| substrate | archive bytes | parser-safe bytes | score-affecting | score-opaque |
|---|---:|---:|---:|---:|
| `dp1_pretrained_driving_prior` | 563 | 0 | 0 | 0 |
| `vq_vae` | 591 | 192 | 192 | 0 |
| `grayscale_lut` | 481 | 0 | 0 | 0 |
| `atw_codec_v2` | 3,250 | 2,632 | 2,632 | 0 |
| **TOTAL** | **4,885** | **2,824** | **2,824** | **0** |

The aggregate verdict is **MIXED-PARSER-SAFE-BUT-SCORE-AFFECTING**:
* DP1 and grayscale_lut have ZERO parser-safe bytes (matches fec6's
  empirical anchor; all sections Brotli/struct/JSON).
* VQ-VAE has 192 parser-safe bytes in `indices_blob` (RAW int16
  packed codebook indices).
* ATW V2 has 2,632 parser-safe bytes across 3 RAW sections:
  `latent_residual_blob` (int8 z_residual), `class_prior_table_blob`
  (fp16), and `cdf_table_blob` (fp16 scorer-conditional CDF).
* **ALL** parser-safe bytes are score-affecting (decoder
  side-information; codebook indices select decoder outputs; latent
  residuals feed Wyner-Ziv reconstruction; CDF tables feed B3
  entropy-coding decoder).

## Why This Matters

The empirical finding is structurally meaningful because it generalizes
the parser-safe taxonomy from a single fec6 anchor into a 4-substrate
comparative landscape. The classification surfaces three orthogonal
substrate archetypes:

1. **All-Brotli substrates (DP1, grayscale_lut, fec6)**: every byte in
   every section is parser-essential. Direct byte substitution is
   IMPOSSIBLE without grammar redesign. Per HNeRV parity discipline L4
   (≤200 LOC inflate budget) the substrates already declare this:
   inflate parses brotli streams + JSON metadata.
2. **Mixed substrates with RAW codebook indices (VQ-VAE)**: the
   `indices_blob` is parser-safe but every index feeds the decoder's
   codebook lookup, so direct mutation IS score-affecting.
3. **Side-information-heavy substrates (ATW V2)**: 3 RAW sections (int8
   latents, fp16 class prior, fp16 CDF) sit outside compressed streams.
   ALL are decoder side-information consumed during inflate-time
   reconstruction; mutation breaks the Wyner-Ziv reconstruction or B3
   entropy-coding path.

The engineering consequence is precise:

- The canonical equation #26 `_INCLUDED_CONTEXTS` already enumerates
  the relevant substrate-specific tokens: `dp1_codebook_bytes` /
  `atw_v2_codec_quantizer_lut` / `intermediate_transform_quantizer` /
  `procedural_codebook_as_lookup_table` / `chroma_lut_replacement`.
- Per CLAUDE.md HNeRV parity discipline L6 (score-domain Lagrangian)
  parser-safe-but-score-affecting bytes are NOT canonical equation #26
  IN-DOMAIN candidates for **direct** byte substitution: the equation
  predicts REPLACEMENT savings for score-OPAQUE bytes. The right
  surface is the existing IN-DOMAIN procedural-replacement context,
  reached via the per-substrate training co-optimizing the replacement
  (sister PROCEDURAL VARIANT BUILDs land this surface).
- No NEW canonical equation #26 IN-DOMAIN context is registered. The
  4-substrate result REINFORCES the existing domain refinement.
- No NEW `_EXCLUDED_CONTEXTS` token is registered either; the existing
  `master_gradient_null_byte_removal_with_constant_reconstruction` +
  `direct_byte_substitution_on_wavelet_decomposition_coefficients`
  exclusions already cover the bug class.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile tools/run_parser_safe_methodology_extension_smoke.py
.venv/bin/python tools/run_parser_safe_methodology_extension_smoke.py --dry-run
.venv/bin/python tools/run_parser_safe_methodology_extension_smoke.py
.venv/bin/python -m pytest -q src/tac/tests/test_parser_safe_methodology_extension_smoke.py
.venv/bin/python -m pytest -q src/tac/tests/test_parser_safe_subset_smoke.py src/tac/canonical_equations/tests/test_procedural_codebook_savings_domain_refinement.py
.venv/bin/python -c "from tac.preflight import check_strict_flipped_catalog_entries_have_live_count_zero; r = check_strict_flipped_catalog_entries_have_live_count_zero(strict=False, verbose=False); print(len(r))"
.venv/bin/python tools/lane_maturity.py validate
git diff --check
```

Dedicated tests pass (15/15). Sister tests pass (37/37 across
`test_parser_safe_subset_smoke.py` + canonical equation #26 domain
refinement). Catalog #185 META-meta drift detector clean (0
violations).

## Cargo-cult audit per assumption (Catalog #303)

1. **Assumption**: "RAW byte sections imply parser-safe-AND-replaceable."
   Classification: **CARGO-CULTED**. Empirically falsified: every RAW
   byte section identified on VQ-VAE + ATW V2 is parser-safe but
   score-affecting (decoder side-information). Unwind path: classify
   parser-safe bytes by SCORE-RELEVANCE before treating them as
   replaceable.
2. **Assumption**: "The PARSER-SAFE SUBSET SMOKE result generalizes to
   all substrates with similar archive grammars." Classification:
   **HARD-EARNED**. The fec6 result was substrate-specific (PR101
   inner-grammar with Brotli/LZMA/Huffman bitstreams); this extension
   empirically verifies the result with synthesized archives for each
   of the 4 IN-DOMAIN substrates.
3. **Assumption**: "Canonical equation #26 `_INCLUDED_CONTEXTS` are
   sufficient for the 4 IN-DOMAIN substrates." Classification:
   **HARD-EARNED**. The 4-substrate classification confirms that the
   already-included context tokens (`dp1_codebook_bytes` /
   `atw_v2_codec_quantizer_lut` / `intermediate_transform_quantizer` /
   `procedural_codebook_as_lookup_table` / `chroma_lut_replacement`)
   cover the substrate-specific procedural-replacement surfaces; no
   new context token is needed.

## 9-dimension success checklist evidence (Catalog #294)

- **UNIQUENESS**: the methodology extends the sister smoke to a
  4-substrate comparative classification; the architecture is unique
  per-substrate (per-grammar region map + per-section RAW vs essential
  classification + per-section score-relevance verdict).
- **BEAUTY + ELEGANCE**: ~700 LOC smoke tool + 200 LOC tests + 1 MD +
  1 JSON; reviewable in 30 seconds per section. Mirrors the sister
  smoke's structure for ease of comparative reading.
- **DISTINCTNESS**: explicitly different from sister fec6 smoke
  (different substrates, different region maps, different verdict
  surface).
- **RIGOR**: STATIC analysis only (no paid GPU); all 4 archives
  synthesized via canonical `pack_archive` calls; archive shapes
  verified against `parse_*_archive_bytes` round-trip; 15 dedicated
  tests covering canonical constants + per-substrate classification +
  aggregate verdict taxonomy + Provenance.
- **OPTIMIZATION PER TECHNIQUE**: smoke tool routes through canonical
  `pack_archive` / `parse_*_archive_bytes` per substrate; canonical
  Provenance per Catalog #323; canonical kind taxonomy + score
  relevance taxonomy.
- **STACK-OF-STACKS COMPOSABILITY**: smoke participates in the
  cathedral autopilot via the canonical `[macOS-CPU advisory]` axis
  (observability-only; never promoted).
- **DETERMINISTIC REPRODUCIBILITY**: synthesizers produce byte-stable
  archives (verified via `test_smoke_synthesizer_determinism`);
  identical sha256 across re-runs.
- **EXTREME OPTIMIZATION + PERFORMANCE**: static analysis only; no
  brotli decompression; runs in <1s wall-clock.
- **OPTIMAL MINIMAL CONTEST SCORE**: the methodology informs the
  STRUCTURAL EXHAUSTION argument: no NEW IN-DOMAIN canonical equation
  #26 contexts surface from these substrates beyond the already-
  included set. Future score-lowering work on these substrates routes
  through the existing PROCEDURAL VARIANT BUILDs (DP1 / VQ-VAE /
  grayscale_lut / ATW V2) which co-optimize replacement at training
  time.

## Observability surface (Catalog #305)

- **Inspectable per layer**: each substrate's region map is a typed
  `SubstrateRegion` frozen dataclass with byte ranges + parser kind +
  parser-essential flag + score-relevance + role + rationale.
- **Decomposable per signal**: aggregate parser-safe total decomposes
  into per-substrate + per-region + per-score-relevance counters.
- **Diff-able across runs**: byte-stable synthesizers + canonical
  sha256 enable byte-level diff between runs.
- **Queryable post-hoc**: smoke_result.json schema
  `parser_safe_methodology_extension_smoke_v1_20260520` is canonical
  JSON consumable by autopilot ranker + future cathedral consumers.
- **Cite-able**: each region's rationale field cites the relevant
  parser source (e.g. `parse_pr101_frame_selector_archive`,
  `decompress_brotli_streams`, `parse_atw2_archive_bytes`).
- **Counterfactual-able**: the SubstrateRegion + SubstrateClassification
  dataclasses are explicit enough to support "what-if" future region
  reclassification (e.g. if VQ-VAE adds a CDF table section).

## Sister coordination + collision verdict

- **NEW canonical equation REGISTRATION** sister: DISJOINT scope (that
  sister registers NEW equations; this smoke confirms NO new equation
  is needed for the 4 substrates).
- **END-OF-DAY OPERATOR ROUTING COMMAND SHEET v2** sister: DISJOINT
  scope (that sister surfaces command-sheet output; this smoke is a
  static analysis artifact).
- Catalog #340 sister-checkpoint guard: cleared at every checkpoint.
- Catalog #314 absorption-pattern guard: NO bare commits during this
  subagent's run (canonical serializer only).

## Provenance (Catalog #323)

- `score_claim`: False
- `promotion_eligible`: False
- `rank_or_kill_eligible`: False
- `ready_for_exact_eval_dispatch`: False
- `axis_tag`: `[macOS-CPU advisory]`
- `evidence_grade`: `macOS-CPU-advisory`
- `canonical_helper_invocation`:
  `tac.provenance.builders.build_provenance_for_macos_cpu_advisory`

## 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map**: N/A — defensive static-analysis artifact
  with no per-pair sensitivity contribution.
- **hook #2 Pareto constraint**: N/A — no Pareto-relevant signal at
  this layer; future per-substrate dispatch consumes the static result
  as side info for the procedural-variant Pareto bound.
- **hook #3 bit-allocator**: N/A — the byte-level region map is
  consumed by future bit-allocator design but not by the current
  fixed-grammar substrates.
- **hook #4 cathedral autopilot dispatch**: ACTIVE — smoke_result.json
  is canonical JSON consumable by autopilot ranker for ranking
  procedural-variant candidates.
- **hook #5 continual-learning posterior**: ACTIVE — Catalog #344
  cross-ref to canonical equation #26 means future empirical anchors
  on the 4 substrates close the prediction-vs-empirical loop on this
  static analysis.
- **hook #6 probe-disambiguator**: ACTIVE — the 3-verdict taxonomy
  (ALL_EMPTY / MIXED_PARSER_SAFE_BUT_SCORE_AFFECTING /
  PARSER_SAFE_AND_SCORE_OPAQUE) IS the canonical disambiguator between
  structural-exhaustion vs new-canonical-context vs known-mixed-state.

## Top-3 operator-routable next-actions

1. **REINFORCE Catalog #344 domain refinement memo**: append a
   structurally-grounded note to canonical equation #26 documentation
   citing this 4-substrate result as additional evidence that the
   `_INCLUDED_CONTEXTS` are sufficient and that direct-byte-
   substitution on parser-safe-but-score-affecting RAW sections (VQ-VAE
   indices / ATW V2 side info) is NOT in the equation's domain.
2. **Route VQ-VAE indices_blob to procedural-codebook-VARIANT**: the
   existing VQ-VAE procedural-variant BUILD (commit `6fea30f22`)
   replaces the codebook tensor inside the decoder state_dict; the
   indices_blob bytes remain trained-codebook-specific. A sister
   landing could extend the procedural variant to also replace the
   indices via a deterministic mapping (subject to Catalog #325
   per-substrate symposium).
3. **Future ATW V2 cdf_table_blob procedural variant**: the 2,560-byte
   `cdf_table_blob` is the largest RAW section; a procedural variant
   that emits the CDF table deterministically from a 32-byte seed
   would route through the existing `atw_v2_codec_quantizer_lut`
   IN-DOMAIN context. Subject to Catalog #324 post-training Tier-C
   validation BEFORE any paid dispatch.

## Sister smoke cascade context

- Sister smoke commit: `e3e198c9f` (PARSER-SAFE SUBSET SMOKE on fec6
  frontier — `PARSER_SAFE_SUBSET_EMPTY`).
- Sister META-LESSON: null-gradient is NECESSARY but NOT SUFFICIENT for
  byte replaceability.
- This extension's role: static structural classification across 4
  IN-DOMAIN substrates per the sister smoke landing memo's
  "Next Action" §1-3.
