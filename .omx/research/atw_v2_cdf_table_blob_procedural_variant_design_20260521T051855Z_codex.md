# ATW V2 cdf_table_blob procedural variant design audit

timestamp_utc: 2026-05-21T05:18:55Z
agent: codex
lane_id: lane_codex_atw_v2_cdf_table_blob_procedural_variant_design_only_20260521
horizon-class: parser_pursuit
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
paid_dispatch_attempted: false
research_only: true
canonical_equations_referenced:
  - procedural_codebook_from_seed_compression_savings_v1
  - procedural_predictor_plus_residual_correction_savings_v1
  - static_packet_custody_byte_delta_score_savings_v1

## Summary verdict

Verdict: PROCEED_TO_REVISED_DESIGN_ONLY, not BUILD and not paid dispatch.

The directive's high-level premise is only partly correct. The ATW2
`cdf_table_blob` is parser-visible and empirically 2,560 bytes:

- Header declares `CDF_TABLE_BLOB_LEN` as `cdf_classes * cdf_symbols * 2`.
- Parser enforces `expected_cdf_table_bytes = cdf_classes * cdf_symbols * 2`.
- Default shape is 5 classes x 256 symbols x fp16 = 2,560 bytes.

However, the stronger claim that these bytes are currently score-affecting is
not supported by the live runtime. `inflate.py` copies `arc.cdf_table` into
`model.cdf_table`, but frame reconstruction calls
`reconstruct_from_wz_residual()`, which uses only `z_residual`,
`scorer_class_prior_table`, and `wz_side_info_head`. The copied CDF table is not
read by the current reconstruction path.

Local empirical smoke on a synthetic ATW2 archive confirmed the code reading:
xoring every byte of `cdf_table_blob` changed neither reconstructed RGB values
nor aggregate output:

```text
cdf_len: 2560
max_abs_rgb_delta_after_cdf_byte_xor: 0.0
sum_abs_rgb_delta_after_cdf_byte_xor: 0.0
```

This makes `cdf_table_blob` a current-runtime parser-visible but decode-opaque
dead section, not the same class as VQ-VAE `indices_blob`. The design path is
dead-section cleanup through a grammar-aware schema change or sentinel envelope
with exact inflate-output parity. It should not be treated as a
residual-correction problem unless a future ATW2 range-decoder path actually
consumes the CDF.

## Ground truth

| claim | evidence | verdict |
|---|---|---|
| ATW2 has a dedicated CDF table section | `src/tac/substrates/atw_codec_v2/archive.py:26-40` | VERIFIED |
| CDF table byte count is 2,560 at default shape | `archive.py:426-431`; `5 * 256 * 2` | VERIFIED |
| CDF table is copied during inflate | `src/tac/substrates/atw_codec_v2/inflate.py:150-156` | VERIFIED |
| CDF table affects current reconstructed frames | `reconstruct_from_wz_residual()` reads class prior and WZ head, not CDF; mutation smoke delta is 0 | INCORRECT |
| D4 predecessor verdict still blocks base ATW V2 paid dispatch | `.venv/bin/python tools/check_predecessor_probe_outcome.py --substrate atw_codec_v2` returns blocking INDEPENDENT verdict | VERIFIED |

## Paradigm classification

| paradigm | verdict | reason |
|---|---|---|
| REPLACEMENT-UPSTREAM | NOT_SELECTED_FOR_CURRENT_RUNTIME_CDF | There is no upstream representation to replace because current reconstruction does not consume the CDF. If a future trained/range-decoded ATW path consumes it, revisit as co-trained upstream replacement. |
| RESIDUAL-CORRECTION-DOWNSTREAM | DEFER_FOR_FUTURE_RANGE_DECODER_ONLY | If a future ATW2 implementation actually uses `cdf_table` to entropy-decode latent symbols, then direct replacement becomes score-affecting and must route through `procedural_predictor_plus_residual_correction_savings_v1` or a co-trained upstream CDF generator. |
| REMOVAL | SELECTED_AS_DEAD_SECTION_CLEANUP_WITH_PARITY_PROOF | Current CDF bytes are decode-opaque. Raw deletion is still refused because it breaks the ATW2 parser/header contract; the valid removal path is a typed schema bump or sentinel/envelope with byte-for-byte inflate-output parity. |

## Canonical-vs-unique decision per layer

| layer | decision | rationale |
|---|---|---|
| byte surface | UNIQUE | This is not VQ-VAE-like decoder addressing. It is currently an unused ATW2 side-info table. |
| archive grammar | FORK | Needs a typed ATW2 CDF sentinel or schema extension; raw deletion is invalid. |
| predictor | CANONICAL | Use `tac.procedural_codebook_generator.derive_codebook_from_seed` only to rehydrate a deterministic shape-valid table. |
| byte accounting | CANONICAL_WITH_GUARD | Prefer `static_packet_custody_byte_delta_score_savings_v1` after exact output parity proves byte-only custody. Use equation #26 only if the scaffold retains a seed-derived replacement table for parser compatibility. If parity fails, route to the residual equation. |
| dispatch | DEFER | Existing D4 blocker still prevents base ATW V2 paid dispatch; no provider dispatch is enabled by this memo. |

## Dykstra feasibility check and predicted band

The only defensible current-runtime rate term is byte-only custody after exact
inflate-output parity:

```text
bytes_saved = 2560 - 32 = 2528
delta_s_rate_only = -25 * 2528 / 37_545_489 = -0.0016832723
```

If the implementation uses a pure schema shrink rather than a 32-byte
seed/envelope, recompute the byte delta with the actual candidate archive byte
count under `static_packet_custody_byte_delta_score_savings_v1`. The 32-byte
seed form above is a conservative procedural envelope budget, not a required
payload.

This is a rate-only prediction, not a score claim. It is feasible only if the
grammar replacement preserves raw inflated frames byte-for-byte. The local RGB
smoke supports that feasibility for the current code path, but a real scaffold
must prove it through archive parse, inflate parity, and exact scorer custody.

No seg/pose band is claimed. The earlier "seg/pose structurally zero under
Variant-C" framing is too broad; zero seg/pose is a property to prove by
inflate-output parity for this specific current runtime, not a property of
score-affecting procedural replacements in general.

## 9-dimension success checklist evidence

| dimension | status | evidence |
|---|---|---|
| uniqueness | PASS | CDF is copied but unused in current inflate reconstruction. |
| elegance | PASS | Valid implementation would be a small typed parser/runtime adapter plus parity tests. |
| distinctness | PASS | Different from VQ-VAE residual indices and different from DWT residual correction. |
| rigor | PASS | Code read plus synthetic all-byte CDF xor smoke. |
| optimization per technique | PARTIAL | Rate term is closed form; no real archive scaffold yet. |
| composability | PASS | Composes as a rate-only parser shrink if parity holds; does not consume D4 side-info hypothesis. |
| deterministic reproducibility | PLANNED | Seed-derived table must be deterministic across CPU/CUDA runtime. |
| performance | PASS | Table derivation is negligible relative to frame reconstruction. |
| optimal contest score | DEFER | Needs real archive scaffold plus exact CPU/CUDA scorer proof before promotion. |

## Cargo-cult audit per assumption

| assumption | classification | correction |
|---|---|---|
| Parser-safe means removable | CARGO-CULTED | Parser contract still has lengths and offsets; use typed grammar replacement. |
| Parser-safe means score-opaque | CARGO-CULTED | VQ-VAE indices are parser-safe and score-affecting. Classify per consumer. |
| ATW2 cdf_table_blob is score-affecting because comments call it decoder side information | CARGO-CULTED-FALSIFIED | Current reconstruction path does not read `cdf_table`; all-byte CDF xor smoke produced zero RGB delta. |
| Equation #26 applies to every seed-derived substitution | CARGO-CULTED | It applies here only if the chosen grammar keeps a seed-derived table for parser compatibility and exact output parity proves score-opacity. Pure schema shrink should use static byte-custody accounting instead. |
| D4 blocker is cleared by shrinking CDF bytes | CARGO-CULTED | D4 still blocks base ATW V2 cooperative-receiver paid dispatch. The CDF shrink is a narrower parser/runtime hygiene lane. |

## Observability surface

The eventual L0 scaffold should expose:

- `analyze_atw2_cdf_section(archive_bytes)` returning section offset, length,
  shape, dtype, and consumer classification.
- `encode_procedural_cdf_blob(seed, shape, dtype, generator_kind)` returning a
  typed sentinel/envelope.
- `decode_procedural_cdf_blob(...)` returning a shape-valid tensor.
- `compose_with_procedural_cdf(...)` returning new archive bytes plus
  `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`.
- `prove_cdf_output_parity(source_archive, candidate_archive)` comparing
  inflated raw outputs before any score claim.

The scaffold must route through typed archive sections or
`CandidateModificationSpec` / `grammar_aware_operator` rows. It must not expose
raw `byte_modifications: Mapping[int, float]` as authority.

## Alternative probe methodologies

1. Exact output-parity probe: replace CDF with deterministic seed table, inflate
   source and candidate, compare raw bytes. This is the highest-signal next
   proof for the current runtime.
2. Consumer-trace probe: instrument `ATWv2Codec` to assert whether
   `cdf_table` is read during `inflate_one_video()` and fail if future code
   starts consuming it without updating the procedural variant classification.
3. Residual-correction probe: if consumer-trace becomes positive, encode
   `cdf_table - procedural_predictor(seed)` with the residual equation and
   compare residual bytes against the original 2,560-byte table.
4. Co-trained upstream probe: train ATW2 with a generated CDF table from the
   start and compare against the same trainer without the generated table.

## Operator-gated deferrals

- D4: still blocking for base `atw_codec_v2` paid dispatch. The existing
  predecessor outcome remains `INDEPENDENT` with metric 0.006385502752 against
  threshold 0.5.
- Variant-C: reframed. The CDF shrink is not a broad Variant-C score-affecting
  substitute; it is a narrow current-runtime score-opaque grammar replacement
  candidate.
- Catalog #325: still required before treating this as a new paid substrate
  dispatch. A local L0 parser/parity scaffold can be prepared only as
  research-only infrastructure and must not claim score.

## Recommended next action

Do not dispatch ATW V2. Do not trust the untracked broad Variant-C memo as a
build greenlight.

The next useful artifact is a small L0 parity scaffold for `cdf_table_blob`:
grammar-aware sentinel replacement, exact parse/inflate-output parity tests,
and a consumer-trace guard that fails if future ATW2 code begins reading
`cdf_table` in reconstruction. If output parity holds on a real ATW archive,
the lane can then seek operator approval for a proper build gate.
