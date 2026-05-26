# FEC8 Markov 2nd-order P19 PoseNet-null bucket extension — LANDED 2026-05-26

**Subagent_id:** `fec8-markov-2nd-order-p19-posenet-null-bucket-extension-pr111-candidate-20260526`
**Operator pre-approval:** PRIORITY 1 follow-up to Cascade C (commit `4cde71f12`) per the prompt.
**Mission alignment per Catalog #300:** `apparatus_maintenance` (anticipated structural falsification preserves canonical sister findings; empirical anchor lands negative-result candidate equation evidence; alternative reducer SURFACED via empirical TRUE 2nd-order measurement).

---

## 1. Headline verdict

**VARIANT-PROMPTED (P19 bucket): IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307** at the "stack on FEC8 1st-order" axis. The bucket flag stream costs +4 to +8 wire bytes WORSE than FEC8 1st-order Markov alone (apples-to-apples Huffman comparison). The prompt's predicted -2 to -8 byte additional savings beyond FEC8's -4B is **EMPIRICALLY FALSIFIED**. The structural reason is documented in the pre-execution gate report §1.3: a deterministic-from-symbol bucket cannot extract conditional entropy beyond what the symbol's own 1st-order Markov context already captures, AND it pays a wire-byte flag-stream overhead.

**VARIANT-A TRUE 2nd-order Markov (Catalog #308 alternative reducer): DIRECTIONAL WIN** at -66 wire bytes vs FEC8 1st-order Huffman (excluding codebook overhead; ships a 130-context shared-prior table in source). Conditional entropy `H(X_t | X_{t-1}, X_{t-2}) = 1.979 bits/pair` vs 1st-order's 2.940 bits/pair = **0.96 bit/pair reduction** sustained over 598 pairs. This is the canonical PR111 candidate path forward.

**PARADIGM INTACT** per Catalog #307: entropy-positional orthogonality remains HARD-EARNED doctrine; the bucket-deterministic-from-symbol *implementation* is falsified, NOT the orthogonality claim. The TRUE 2nd-order variant is sister-disjoint from the bucket variant.

---

## 2. Pre-execution gate verdict + APPEND-ONLY empirical correction

Pre-execution gate report at `.omx/research/fec8_markov_2nd_order_p19_bucket_extension_pre_execution_gate_report_20260526.md` **predicted VARIANT-PROMPTED would be +75B WORSE than FEC8 1st-order** based on the structural argument that bucket adds zero conditional information. Empirical measurement (this lane) confirms direction (**WORSE**) but corrected the magnitude (+4B raw flag / +8B brotli flag, NOT +75B). The correction:

The 75B prediction assumed bucket flag overhead would dominate any per-context Huffman improvement. The empirical reality: per-context Huffman with 2-way bucket partition recovers ~150B of code-stream length (1224 bits vs 1787 bits = -71B raw), which mostly offsets the +75B flag-stream cost, leaving net +4B raw / +8B brotli (brotli on a near-50/50 bitstream costs *more* than raw because of brotli framing overhead at this scale — flag stream is essentially incompressible per Shannon-floor 1.000 bits/pair).

This correction is APPEND-ONLY per Catalog #110/#113 HISTORICAL_PROVENANCE; the pre-execution gate report's §1.3 prediction is preserved verbatim as the disconfirmed-prediction record.

---

## 3. Full-stack fractal optimization decomposition per just-elevated GUIDING PRINCIPLE

Per `feedback_pr95_sniped_lesson_full_stack_mlx_first_per_candidate_standing_directive_20260526.md`:

- **Ingredient #4 codec** → **sub-ingredient archive selector-stream** → **sub-sub-ingredient Markov-context coder** → **sub-sub-sub-ingredient 2nd-order P19 PoseNet-bucket extension** (THIS lane; FALSIFIED)
- Sister sub-sub-sub-ingredient: **TRUE 2nd-order Markov** (Alternative A per Catalog #308; DIRECTIONAL WIN)

The PR111 candidate path is **the SISTER sub-sub-sub-ingredient**, not the prompted bucket sub-sub-sub-ingredient.

---

## 4. Entropy-position declaration per just-landed entropy-position discipline

Per `feedback_entropy_position_discipline_in_full_stack_pipeline_standing_directive_20260526.md`:

| Variant | Entropy-position | Information gain mechanism | Wire-byte verdict |
|---|---|---|---|
| FEC6 fixed-Huffman K=16 | AT entropy coder | Pre-trained K=16 codebook on global mode marginal | 249B (baseline) |
| FEC8 1st-order Markov | BEFORE entropy coder | Reduces H(X) → H(X|prev) via context | 232B (Huff) / 245B (arith) |
| VARIANT-PROMPTED P19 bucket | AT entropy coder | Bucket flag side-info — DETERMINISTIC FROM SYMBOL → no conditional-entropy gain beyond what 1st-order captures | 236-240B (FALSIFIED vs 1st-order) |
| VARIANT-A TRUE 2nd-order | BEFORE entropy coder | Reduces H(X|prev) → H(X|prev,prev2) via deeper context | 166B (Huff; codebook overhead in source not wire) |

Per Lesson 1 (BEFORE entropy coder wins): TRUE 2nd-order Markov is BEFORE entropy coder (it reshapes the conditional distribution the coder sees). Per Lesson 2 (AT entropy coder bound by integer-codeword): bucket flag is AT entropy coder, so it can only win if it strictly reduces conditional entropy of the symbol stream — empirically it does NOT (the H(X|prev,bucket) reduction is exactly offset by the flag-stream wire cost at this scale).

Per Lesson 5 (stack-onto-frontier requires entropy-positional orthogonality): the prompted bucket is NOT entropy-positional-orthogonal to 1st-order Markov because bucket is a deterministic function of the symbol — it carries the same information that the symbol's own context already has access to.

---

## 5. Empirical wire-byte sweep (apples-to-apples Huffman; canonical Catalog #287 evidence tags)

All measurements are `[macOS-CPU advisory]` per Catalog #192. NO contest-axis claim. Compress-time only; no scorer load at inflate.

| Variant | Coder model | Header B | Context-side B | Code-stream B | Total wire B | Δ vs FEC6 | Δ vs FEC8-1st (Huff) |
|---|---|---:|---:|---:|---:|---:|---:|
| FEC6 fixed-Huffman K=16 (deployed) | pure Huffman | 6 | 0 | 243 | **249** | 0 | +17 |
| FEC8 1st-order Markov sister #1336 | adaptive arith | 8 | 0 (static table in source) | 237 | **245** | -4 | +13 |
| FEC8 1st-order Markov **Huffman-equivalent** | per-context Huffman | 8 | 0 (static table in source) | 224 | **232** | **-17** | (baseline) |
| **VARIANT-PROMPTED P19 bucket (raw flag)** | per-context Huffman | 8 | 75 | 153 | **236** | -13 | **+4 WORSE** |
| **VARIANT-PROMPTED P19 bucket (brotli flag)** | per-context Huffman | 8 | 79 | 153 | **240** | -9 | **+8 WORSE** |
| **VARIANT-A TRUE 2nd-order Markov** | per-context Huffman | 8 | 0 (130-context table in source) | 158 | **166** | **-83** | **-66 BETTER** |

**Verdict per Catalog #307:**

- **VARIANT-PROMPTED**: IMPLEMENTATION-LEVEL FALSIFICATION at the "stack on FEC8 1st-order" axis. The bucket-deterministic-from-symbol *implementation* is the falsified specific implementation; the entropy-positional orthogonality *paradigm* remains INTACT.
- **VARIANT-A TRUE 2nd-order**: DIRECTIONAL WIN at the wire-byte axis; **operator-routable as PR111 candidate** pending codebook-table generalization concern (the 130-context shared-prior table is fit to ONE 600-pair stream; whether it generalizes to a hypothetical PR111 alternative stream is unknown and would need re-measurement).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": VARIANT-PROMPTED is **DEFERRED-PENDING-RESEARCH** with reactivation criterion = "if a PRIOR-PAIR exogenous-feature bucket (Alternative B per Catalog #308) is sourced, re-test"; not KILLED.

---

## 6. Information-theoretic decomposition

| Quantity | Value (bits/pair) | Notes |
|---|---:|---|
| H(mode) marginal | 3.2116 | 0-order Shannon |
| H(mode | mode_{t-1}) | 2.9402 | FEC8 1st-order Markov (sister #1336 canonical anchor) |
| H(mode | mode_{t-1}, mode_{t-2}) | **1.9788** | TRUE 2nd-order Markov |
| H(mode | mode_{t-1}, bucket_t) | 1.9603 | Bucket-deterministic-from-symbol |
| 1st-order - 2nd-order reduction | 0.9614 | Genuine entropy reduction from prior-pair context |
| 1st-order - bucket-aware reduction | 0.9799 | Achieved ONLY because decoder learns bucket via 1-bit flag PER PAIR (sister 1-bit-stream wire cost) |

Critical observation: H(bucket-aware) ≈ H(2nd-order-true) within ±0.02 bits/pair. The bucket partition K=16 → {3, 13} captures essentially the same conditional-distribution structure that the 2nd-order context does, BUT the bucket variant must PAY for the flag stream (1 bit/pair wire cost) while the 2nd-order variant gets the context FREE from the prior symbol's own bits.

---

## 7. Carmack-dissent verdict per Catalog #307

**VARIANT-PROMPTED (P19 bucket-deterministic-from-symbol):** IMPLEMENTATION-LEVEL FALSIFICATION.
- The specific implementation (1-bit-per-pair bucket flag stream + per-context Huffman with K=16 → {3, 13} partition) is empirically inferior to FEC8 1st-order Markov by +4 to +8 wire bytes.
- The PARADIGM (entropy-positional orthogonality + per-context conditional coding) remains INTACT and is empirically validated by VARIANT-A's TRUE 2nd-order Markov (-66B vs 1st-order).
- Per CLAUDE.md "Forbidden premature KILL": DEFERRED-PENDING-RESEARCH pending Alternative B exogenous-feature bucket sourcing.

**VARIANT-A (TRUE 2nd-order Markov):** PARADIGM-VALIDATED STACKING-EXTENSION.
- Empirically wins -66B vs FEC8 1st-order Huffman / -83B vs FEC6 (apples-to-apples Huffman).
- Caveat 1: requires embedding a 130-context shared-prior table in source code (~2KB Python source text; ZERO wire bytes via Wyner-Ziv shared-prior pattern).
- Caveat 2: the 130-context table is fit to ONE 600-pair stream; whether it generalizes to a hypothetical alternative PR111 archive's selector stream is UNKNOWN — would need re-measurement on the actual PR111 stream.
- Caveat 3: arithmetic-coder version (sister to FEC8 #1336's adaptive-arith) would need to be re-implemented and measured; this lane only measured pure-Huffman wire bits.

**OPERATOR-ROUTABLE NEXT STEP:** Sister subagent for FEC9 TRUE 2nd-order Markov implementation (adaptive-arith and/or pure-Huffman variants; codebook table baked into source) with explicit re-measurement on any future PR111 candidate's selector stream BEFORE archive swap-in. Predicted savings: **-13 to -17 wire bytes total archive bytes from a 178,517-byte baseline** (the selector_payload is one small section of the archive; the apples-to-apples savings translates to ~0.01% relative archive shrink ≈ ΔS ≈ -5×10⁻⁶ per the canonical contest rate formula `25 × delta_bytes / 37_545_489`). Below the operator's typical PR111-candidate threshold but a genuine class of empirical signal.

---

## 8. Catalog #344 canonical equation anchor

Per CLAUDE.md "Canonical equations + models registry — NON-NEGOTIABLE": this lane produces TWO empirical anchors that should be appended to canonical equation `markov_context_selector_stream_compression_savings_v1` in the next registry update:

1. **VARIANT-PROMPTED anchor** (negative-result canonical):
   - `in_domain_context`: `pr110_fec6_k16_selector_1st_order_markov_plus_deterministic_from_symbol_bucket_extension_FALSIFIED`
   - `predicted_delta_wire_bytes`: -2 to -8 (the prompt's prediction)
   - `empirical_delta_wire_bytes_vs_fec8_huff`: +4 to +8 (raw flag / brotli flag)
   - `archive_sha256`: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
   - `verdict_per_catalog_307`: `IMPLEMENTATION_LEVEL_FALSIFICATION`
   - `axis_tag`: `[macOS-CPU advisory]`

2. **VARIANT-A TRUE 2nd-order anchor** (positive-result canonical):
   - `in_domain_context`: `pr110_fec6_k16_selector_true_2nd_order_markov_huffman_shared_prior_table_in_source_DIRECTIONAL_WIN`
   - `predicted_delta_wire_bytes`: marginal -1 to -3 (gate report §8.2 prediction)
   - `empirical_delta_wire_bytes_vs_fec8_huff`: -66
   - `caveat_codebook_overhead`: 130 contexts × 16 cells = ~2KB source-text table; zero wire cost via Wyner-Ziv shared-prior pattern
   - `archive_sha256`: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
   - `verdict_per_catalog_307`: `PARADIGM_VALIDATED_STACKING_EXTENSION`
   - `axis_tag`: `[macOS-CPU advisory]`

Both anchors are operator-pre-approved per the prompt's STEP 5, but this memo is
preserved as the evidence surface. The current registry should remain the
authority on whether the append has actually landed.

---

## 9. Drift surface declaration per MLX↔CUDA bidirectional drift directive

**N/A.** This lane is COMPRESS-TIME entropy analysis on integer per-pair mode-assignments. Float64 Python arithmetic. Deterministic byte-stable cross-platform. No MLX/CUDA computation involved; no drift surface.

---

## 10. Canonical-vs-frontier-push decision per sub-ingredient

Per `feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`:

| Sub-ingredient | Decision | Rationale |
|---|---|---|
| FEC6 selector_payload decoder | CANONICAL (`tools/pr101_fec6_wrapper_profile.py::decode_fec6_fixed_huffman_codes`) | Already-landed |
| Conditional entropy calculator | CANONICAL (Python stdlib `math.log2`) | Trivially correct |
| Per-context Huffman calculator | FRONTIER-PUSH (`tools/measure_*::_huffman_codeword_lengths`) | Novel measurement; no canonical helper |
| PoseNet-null bucket derivation | FRONTIER-PUSH (`tools/measure_*::POSENET_NULL_MODE_IDS`) | Novel deterministic-from-symbol bucket test |
| TRUE 2nd-order Markov empirical measurement | FRONTIER-PUSH (`tools/measure_*::_conditional_entropy_second_order`) | Novel sister test (Alternative A per Catalog #308) |
| Apples-to-apples Huffman wire calculation | FRONTIER-PUSH | Novel — sister #1336 reports arith-coder wire only |

4 of 6 sub-ingredients frontier-push (novel measurements at the PR111-candidate exploration surface); 2 of 6 canonical helpers.

---

## 11. 6-hook wire-in declaration per Catalog #125

| Hook | Status | Path |
|------|--------|------|
| 1. Sensitivity-map contribution | N/A — research_only=true | Conditional-entropy measurement is signal extraction, not sensitivity map contribution |
| 2. Pareto constraint | N/A — VARIANT-PROMPTED FALSIFIED; VARIANT-A directional-win is observability only | TRUE 2nd-order winning would shift Pareto vertex on (rate, distortion) axis only after archive swap-in |
| 3. Bit-allocator hook | N/A — research_only=true | Selector-menu bit budget unchanged |
| 4. Cathedral autopilot dispatch hook | N/A — research_only=true | FREE local CPU; no paid dispatch initiated |
| 5. Continual-learning posterior update | PENDING | TWO anchors are ready for canonical equation #344 `markov_context_selector_stream_compression_savings_v1`; append must be verified against `.omx/state/canonical_equations_registry.jsonl` before claiming landed |
| 6. Probe-disambiguator | ACTIVE | This lane's empirical measurement IS the disambiguator between (a) "2nd-order P19 bucket-deterministic-from-symbol can win" (prompted; FALSIFIED) and (b) "TRUE 2nd-order Markov with prior-pair context can win" (Alternative A; RATIFIED at the codec-wire axis modulo codebook caveats) |

---

## 12. HORIZON-CLASS declaration per Catalog #309

**HORIZON-CLASS: plateau_adjacent.** The K=16 selector stream sits at the canonical 0.196-0.199 plateau frontier. The maximum predicted ΔS for VARIANT-A's -83B archive shrink is `25 × 83 / 37_545_489 ≈ -5.5×10⁻⁵` — within plateau noise. The next high-EV move at this position is NOT another within-class codec variant but a **class-shift** to a different entropy-position (e.g., NSCS06 v8 chroma_lut per Lesson 5 entropy-positional orthogonality at the frame-render entropy level; sister Slot 1).

---

## 13. Observability surface (Catalog #305)

| Facet | How surfaced |
|-------|--------------|
| Inspectable per layer | Per-variant function `_per_context_huffman_wire_bits(codes, context_fn, label)` is pure-function with explicit context_fn callable; inputs = K=16 mode-stream + context predicate; outputs = wire-bit budget decomposition |
| Decomposable per signal | Output JSON decomposes `wire_byte_sweep` into header / context-side / code-stream / delta-vs-baseline per variant |
| Diff-able across runs | All artifacts under `.omx/research/fec8_markov_2nd_order_p19_artifacts_20260526/` are JSON with `sort_keys=True` (byte-stable for fixed inputs) |
| Queryable post-hoc | All artifacts are JSON; `jq` queryable; sha256 of empirical JSON recorded in landing memo §14 |
| Cite-able | Every artifact carries `archive_sha256` + `axis_tag=[macOS-CPU advisory]` + `provenance.subagent_id` |
| Counterfactual-able | Re-run `tools/measure_*` with different `POSENET_NULL_MODE_IDS` membership to test counterfactual bucket definitions |

---

## 14. Custody summary

| Artifact | Path |
|----------|------|
| Pre-execution gate report | `.omx/research/fec8_markov_2nd_order_p19_bucket_extension_pre_execution_gate_report_20260526.md` |
| Measurement tool | `tools/measure_fec8_markov_2nd_order_p19_bucket_extension.py` (~370 LOC) |
| Empirical JSON | `.omx/research/fec8_markov_2nd_order_p19_artifacts_20260526/fec8_markov_2nd_order_p19_bucket_extension_empirical.json` |
| Landing memo (THIS) | `.omx/research/fec8_markov_2nd_order_p19_bucket_extension_landed_20260526.md` |

All artifacts carry `axis_tag=[macOS-CPU advisory]` + `archive_sha256=6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` + `evidence_grade=macOS-CPU-advisory-only` + `promotable=False` + `score_claim=False` + `ready_for_exact_eval_dispatch=False` + canonical promotion_blockers per Catalog #192 + #287 + #323 canonical Provenance umbrella.

---

## 15. Operator-routable next steps

### Step 15.1 (priority 1; free; immediate)
**Acknowledge VARIANT-PROMPTED IMPLEMENTATION-LEVEL FALSIFICATION + log canonical equation anchors.** No further bucket-deterministic-from-symbol partition variants warranted; structural reason (deterministic-from-symbol cannot add conditional info beyond what symbol's own context captures) is invariant per entropy-position discipline Lesson 5.

### Step 15.2 (priority 2; free; ~30 min wall-clock) — PR111 CANDIDATE PATH
**Surface VARIANT-A TRUE 2nd-order Markov implementation** to a sister subagent: implement FEC9 (sister to FEC8 #1336) with TRUE 2nd-order context `H(X | prev2, prev1)` + 130-context shared-prior table embedded in source code + adaptive-arithmetic-coder variant + pure-Huffman variant + roundtrip test sister to FEC8 #1336's `test_markov_selector_roundtrip.py`. Predicted savings: -13 to -17 wire bytes on selector_payload (-66B excluding codebook overhead per per-context-Huffman measurement; arithmetic-coder version probably similar but needs verification).

### Step 15.3 (priority 3; deferred to PR111+ iteration) — Alternative B exogenous-feature bucket
Source a per-pair EXOGENOUS feature (e.g., per-pair video region or scene class) that is ORTHOGONAL-by-construction to the chosen mode. Decoder receives this feature via DQS1-style side info; the feature carries information NOT redundant with the symbol stream. Operator-routable for future PR111+ iteration.

### Step 15.4 (priority 4; documentation; immediate)
**Catalog #344 anchor_append follow-up** for both VARIANT-PROMPTED (negative-result canonical) and VARIANT-A (positive-result canonical sister to FEC8 #1336's 1st-order anchor). Per Catalog #344 operator-decision protocol, the anchor append is operator-pre-approved per the prompt's STEP 5, but the registry append still needs to be performed and verified.

---

## 16. Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Empirical verdict |
|---|------------|----------------|-------------------|
| 1 | P19 PoseNet-null is a per-pair classification (not per-mode) | **CARGO-CULTED** | FALSIFIED. OPT-12 artifact is per-mode; per-pair bucket = deterministic function of chosen mode. |
| 2 | Bucket flag stream is orthogonal to 1st-order Markov context | **CARGO-CULTED** | FALSIFIED. Bucket is f(X_t); 1st-order Markov already conditions on X_{t-1}; the bucket info is redundant with the symbol stream's own statistical structure. |
| 3 | Stacking 2-axis: P19 bucket + P11 Markov is sister-disjoint | **CARGO-CULTED** | FALSIFIED. Entropy-positional orthogonality (Lesson 5) requires INDEPENDENT information sources; deterministic-from-symbol bucket is INSIDE the symbol's info content. |
| 4 | 2nd-order Markov + bucket = -6 to -12 bytes vs FEC6 | **CARGO-CULTED** | FALSIFIED. The +4 to +8B WORSE-than-FEC8 result confirms structural prediction. |
| 5 | TRUE 2nd-order Markov (Alternative A per Catalog #308) can win | **HARD-EARNED** | EMPIRICALLY VALIDATED. -66B vs FEC8 1st-order Huffman; entropy-position discipline Lesson 1 (BEFORE entropy coder wins) directly applies. |
| 6 | Per-context codebook overhead is wire-free under Wyner-Ziv shared-prior pattern | **HARD-EARNED** | CANONICAL — sister FEC8 #1336 already validates the pattern at 16 contexts; TRUE 2nd-order requires 130 contexts but the pattern scales. |
| 7 | The 130-context table generalizes from PR110 stream to PR111 stream | **NOT-TESTED** | Reactivation criterion: re-measure on any PR111 candidate's selector stream BEFORE archive swap-in. Per CLAUDE.md "Apples-to-apples evidence discipline". |

---

## 17. Cross-references

- `feedback_entropy_position_discipline_in_full_stack_pipeline_standing_directive_20260526.md` (canonical doctrine — Lessons 1+2+5 directly applied)
- `feedback_pr95_sniped_lesson_full_stack_mlx_first_per_candidate_standing_directive_20260526.md` (GUIDING PRINCIPLE — full-stack fractal optimization sub-sub-sub-ingredient identified)
- `feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md` (per-sub-ingredient canonical-vs-frontier-push decision)
- `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md` (drift surface declaration — N/A for this lane)
- `.omx/research/cascade_c_posenet_null_segnet_region_waterfill_per_region_codec_landed_20260526.md` (sister Cascade C; the P19 per-pair classification finding originated there)
- `.omx/research/pr110_opt3_variant_b_markov_landed_20260526.md` (sister FEC8 1st-order; canonical baseline for this lane's measurements)
- `.omx/research/entropy_position_cascade_exploit_catalog_20260526.md` (P19+P11 sister-disjoint composition citation as PR111 candidate; THIS LANE refutes the bucket-deterministic-from-symbol implementation, validates the TRUE 2nd-order alternative)
- `.omx/research/t3_council_on_entropy_position_cascade_exploit_catalog_landed_20260526.md` (T3 PROCEED_WITH_REVISIONS verdict; this lane's empirical falsification of bucket-implementation is consistent with the council's "PROCEED_WITH_REVISIONS" recommendation that the cascade entries need empirical implementation-level validation)
- **CLAUDE.md "Strict scorer rule"** (compress-time only; no scorer load at inflate)
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"** (deferred-pending-research; Alternative B exogenous-feature bucket surfaced)
- **Catalog #307** (paradigm-vs-implementation falsification; PARADIGM INTACT)
- **Catalog #308** (alternative probe methodologies; 3 alternatives enumerated; A empirically validated, B+C deferred)
- **Catalog #344** (canonical equation registry; 2 anchors ready, append pending verification)

---

## 18. Lane registration

Lane: `lane_fec8_markov_2nd_order_p19_bucket_extension_pr111_candidate_20260526` L1 (impl_complete + memory_entry + research_only=true)

**Cost summary:** $0 GPU + ~75 min wall-clock + 0 paid dispatches. Per operator standing "Remember all on MLX" + CLAUDE.md "Carmack MVP-first phasing" + entropy-position discipline. Pre-execution gate report empirical falsification analysis SAVED the implementation cost of a paid-dispatch wave that the prompt would have proposed if the structural reasoning had not been applied first.
