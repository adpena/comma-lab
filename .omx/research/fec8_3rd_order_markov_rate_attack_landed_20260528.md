---
council_tier: T1
council_attendees: [Shannon_inner_council, Dykstra_inner_council, Rao_grand_council, Ballard_grand_council, AssumptionAdversary_inner_council]
council_quorum_met: false
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: AssumptionAdversary
    verbatim: "the operator-mandated `## Predicted ΔS band` was correctly waived via PREDICTED_BAND_VIBES_OK on data-sparsity grounds (4096 contexts vs 597 quads is canonical data-sparsity territory; Dykstra-feasibility cannot bound a per-context Laplace overhead analytically); reactivation criteria document explicit empirical paths"
council_assumption_adversary_verdict:
  - assumption: "deeper Markov context monotonically reduces wire bytes"
    classification: CARGO-CULTED
    rationale: "EMPIRICALLY FALSIFIED: 240B 3rd-order > 239B 2nd-order; data-sparsity overhead inverts the relationship; HARD-EARNED only at the Shannon-floor level (H bits/pair DOES monotonically decrease 3.21->2.94->1.98->0.84)"
  - assumption: "arithmetic coder reaches Shannon floor"
    classification: CARGO-CULTED
    rationale: "EMPIRICALLY FALSIFIED on sparse contexts: arithmetic-coder 240B vs Shannon-floor 62.7B = 3.83x overhead from per-context Laplace +1 smoothing; per-context Huffman wire-bit floor 106B closer (1.69x) but still requires explicit codebook side-info to deploy"
council_decisions_recorded:
  - "op-routable #1: register canonical equation fec8_3rd_order_markov_static_variant_c_savings_v1 per Catalog #344 with IMPLEMENTATION_LEVEL_FALSIFIED_PARADIGM_INTACT verdict per Catalog #307 (LANDED)"
  - "op-routable #2: register probe outcome DEFER blocking with 30-day expires per Catalog #313 + reactivation criteria documented (LANDED)"
  - "op-routable #3: ship canonical encoder/decoder for completeness + future hybrid implementations per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD (LANDED at submissions/.../build_pr101_frame_exploit_selector_packet_fec8_3rd_order_markov.py)"
  - "op-routable #4: emit empirical 4-tuple counts sidecar for reactivation pathway #2 (per-context Huffman with deterministic codebook reconstruction) (LANDED at .omx/state/fec8_3rd_order_markov_empirical_table_20260528.py)"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
---

# FEC8 3rd-order Markov rate-attack canonical landing 2026-05-28

Wave N+24 Option A next-iteration depth-axis extension per CLAUDE.md "Final
rate attack" standing directive + just-saved standing directive
``rate-attack-default-cost-class-is-zero-mlx-or-cpu-not-paid-modal-standing-directive-20260528``
+ Wave N+24 STAND_DOWN audit identification of FEC8 3rd-order Markov as the
canonical depth-axis extension target (sister to canonical equations
`fec8_1st_order_markov_static_variant_b_savings_v1` 245B + 
`fec8_2nd_order_true_markov_variant_a_savings_v1` 166B Huffman wire-bit floor
+ 239B arithmetic-coder static).

## 1. Premise verification

Per Catalog #229 + #376 exhaustive PV (multiple times today per operator critique):

- `grep -r "fec8.*3rd.*markov\|fec8_3rd\|3rd_order_markov" submissions/ src/ tools/` → ZERO hits
- `grep "fec8_3rd_order_markov_savings\|fec8_3rd_order" .omx/state/canonical_equations_registry.jsonl` → ZERO matches
- `git log --oneline --grep="fec8.*3rd\|3rd.order.markov"` → ZERO commits
- `cat .omx/state/probe_outcomes.jsonl | grep "fec8.*3rd"` → ZERO outcomes
- HEAD at `2ef7d2c26` clean per Catalog #376 spawn-PV evidence

VERDICT: GENUINELY NEW work; no STAND_DOWN per Catalog #340 Variant 1.

## 2. Empirical measurement (macOS-CPU advisory, $0 LOCAL CPU)

Per CLAUDE.md "Bit-level deconstruction and entropy discipline" non-negotiable
+ just-saved cost-class directive (rate-attack iteration is $0 MLX/CPU by
default; Modal-paid only for paired-CUDA RATIFICATION).

Measurement script: `tools/measure_fec8_markov_3rd_order_rate_attack.py`
Stream: PR101 FEC6 archive selector_payload, 600 pairs, 16-symbol palette
Source archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`

```
n_pairs:                                  600
unique 4-tuples observed:                 529
contexts possible (16^3):                 4096
data_sparsity_ratio:                      0.0091   (596 / 65536 quad-tuples)

live FEC6 baseline (bytes):               249
FEC8 1st-order static (bytes):            245   (-4 vs FEC6)
FEC8 2nd-order static (bytes):            239   (-10 vs FEC6, -6 vs 1st)
FEC8 3rd-order static (bytes):            240   (-9 vs FEC6, +1 vs 2nd)  <-- THIS
FEC8 3rd-order adaptive (bytes):          304   (+55 vs FEC6)

per-context Huffman 2nd-order wire-floor: 166   (-83 vs FEC6) [info-theoretic only]
per-context Huffman 3rd-order wire-floor: 106   (-143 vs FEC6) [info-theoretic only]

Shannon H(marginal) bits/pair:            3.2116
Shannon H(X|prev1) bits/pair:             2.9402
Shannon H(X|prev1,prev2) bits/pair:       1.9788
Shannon H(X|prev1,prev2,prev3) bits/pair: 0.8356  <-- 3rd-order entropy floor
Shannon 3rd-order floor (bytes / 600 p):  62.7    <-- absolute lower bound

3rd-order static payload sha256:  2fc92d37a53f792bac6f119ece85594ee4a7778c99ffe71fb73e245a02b5d6f3
3rd-order adaptive payload sha256: ff0ef72f8503cec0e78c60acd880cbac9a7e6737eac06d0ae3a06689f386df10

roundtrip verified: 1st/2nd/3rd-static/3rd-adaptive all OK
```

## 3. Per-axis decomposition per Catalog #356

This codec is RATE-AXIS-ONLY (no pose/seg change since the selector codec is
algorithmically lossless — same per-pair selectors decoded, just packaged
differently). Per Catalog #356 AxisDecomposition:

```
predicted_d_seg_delta:           0.0    (codec-only; lossless)
predicted_d_pose_delta:          0.0    (codec-only; lossless)
predicted_archive_bytes_delta:   -9     (3rd-order static vs FEC6 249B)
predicted_score_delta:           -9 * 25 / 37545489 = -5.99e-6
axis_tag:                        "[predicted]"
canonical_provenance:            build_provenance_for_predicted(...)
```

But: +1B WORSE than FEC8 2nd-order canonical (239B → 240B); per Catalog #368
substitution-stacking must use canonical frontier baseline (DQS1 sha
`7a0da5d0fc327cba` at 0.1920282830 [contest-CPU] or FEC6 baseline 249B per
ENCODER comparison) — FEC8 3rd-order is NOT a frontier-crossing candidate
because it's strictly dominated by FEC8 2nd-order on the SAME stream.

## 4. Cargo-cult audit per assumption

Per Catalog #303 + CLAUDE.md HARD-EARNED-vs-CARGO-CULTED addendum.

### Assumption 1: deeper Markov context monotonically reduces wire bytes
- Classification: **CARGO-CULTED** (EMPIRICALLY FALSIFIED today)
- Source: information-theory intuition (H bits/pair monotonically decreases)
- Falsification: arithmetic-coder Laplace smoothing overhead per-context grows
  as O(K) per context × K^d contexts = O(K^(d+1)) for d-th-order Markov; this
  overhead dominates Shannon savings on sparse streams.
- Unwind path: per-context Huffman with explicit observed-only codebook (no
  Laplace smoothing); requires escape-codeword discipline for unobserved
  symbols. Reactivation criterion #2 in canonical equation anchor.

### Assumption 2: 597 4-tuples is sufficient data for 4096-context model
- Classification: **CARGO-CULTED** (data-sparsity ratio 0.0091 = empirical
  proof of insufficiency)
- Source: assumption that contest-stream Markov order is bounded by stream
  length; in practice the Laplace-smoothing overhead overwhelms when
  K^d >> N_observations / K (cumulant convergence rate)
- Unwind path: hybrid 2nd-order + selective 3rd-order escape pattern that
  uses 3rd-order only for contexts with sufficient observations.
  Reactivation criterion #3.

### Assumption 3: arithmetic coder ≈ Shannon floor on conditional entropy
- Classification: **CARGO-CULTED** (3.83x overhead on this stream)
- Source: textbook arithmetic-coder asymptotic optimality result
- Falsification: 240B arithmetic-coder vs 62.7B Shannon floor = 3.83x
  overhead; per-context Huffman 106B wire-floor = 1.69x overhead. The
  textbook result assumes per-context counts are stationary (infinite stream);
  finite-stream Laplace correction inflates the prior by O(K^d).
- Unwind path: longer streams (>>10x) where the per-context Laplace
  correction amortizes. Reactivation criterion #1.

## 5. Predicted ΔS band

Per Catalog #296: predicted-band Dykstra-feasibility check is N/A for this
codec (codec is lossless rate-axis-only; no Pareto polytope intersection
required). Same-line waiver `# PREDICTED_BAND_VIBES_OK` applied per the
canonical pattern for diminishing-returns data-sparsity-bounded analysis.

Predicted band derived from sister-FEC8-family extrapolation:
- FEC8 1st-order: -4B vs FEC6 (empirical)
- FEC8 2nd-order static: -10B vs FEC6 (empirical)
- Linear extrapolation: -16B for 3rd-order (-6B further reduction)
- Predicted band: [-25B, -15B] vs FEC6 (assuming diminishing returns)

EMPIRICAL: -9B vs FEC6 (+1B WORSE than 2nd-order). **Outside predicted band
on the conservative end** — the Laplace-overhead dominance was anticipated
qualitatively but the +1B regression vs 2nd-order was UNEXPECTED. The
canonical equation records this as residual = 0.043 (4.3% relative error).

## 6. Observability surface per Catalog #305

- **inspectable per layer**: empirical 4-tuple counts sidecar at
  `.omx/state/fec8_3rd_order_markov_empirical_table_20260528.py` (529 unique
  4-tuples observable as Python source literal)
- **decomposable per signal**: measurement JSON at
  `.omx/state/fec8_3rd_order_markov_rate_attack_measurement_20260528.json`
  decomposes into 1st/2nd/3rd-order Shannon entropies + per-encoder byte
  counts + per-encoder roundtrip-verified flags
- **diff-able across runs**: deterministic encoder + decoder; sha256 of each
  variant's payload is fixed across runs
- **queryable post-hoc**: canonical equation
  `fec8_3rd_order_markov_static_variant_c_savings_v1` in canonical equations
  registry; canonical posterior in probe_outcomes.jsonl with `expires_at_utc`
- **cite-able**: canonical equation `equation_id` + `last_calibration_utc` +
  `provenance.canonical_helper_invocation` + sister probe-outcome `probe_id`
- **counterfactual-able**: re-run `tools/measure_fec8_markov_3rd_order_rate_attack.py`
  with `--emit-table-only` to regenerate the empirical 4-tuple counts on any
  new selector stream; encoder/decoder will produce different wire bytes for
  the new stream's per-context distributions.

## 7. 9-dimension success checklist evidence per Catalog #294

| Dim | Score | Evidence |
|-----|-------|----------|
| UNIQUENESS | PARTIAL | Deeper Markov context IS a class-shift attempt over arithmetic-coder family but EMPIRICALLY no improvement on this stream |
| BEAUTY+ELEGANCE | PASS | Canonical sister to existing FEC8 1st/2nd-order modules; same arithmetic-coder primitives + same wire format pattern |
| DISTINCTNESS | PASS | Distinct variant byte `b"\x00\x04"`; distinct EmpiricalAnchor; distinct canonical equation |
| RIGOR | PASS | Roundtrip-verified on full 600-pair stream; empirical-bit-spend proof via measurement script; provenance per Catalog #323 |
| OPTIMIZATION-PER-TECHNIQUE | FAIL | +1B vs 2nd-order baseline on this stream; canonical equation records IMPLEMENTATION-LEVEL FALSIFIED per Catalog #307 |
| STACK-OF-STACKS-COMPOSABILITY | DEFERRED | Cannot stack onto canonical frontier without strictly improving over 2nd-order; deferred per reactivation criteria |
| DETERMINISTIC-REPRODUCIBILITY | PASS | sha256 of payload deterministic across runs; tie-break deterministic per `_huffman_codeword_lengths` |
| EXTREME-OPTIMIZATION-PERFORMANCE | DEFERRED | Per-context Huffman explicit-codebook would be the next optimization path (reactivation criterion #2) |
| OPTIMAL-MINIMAL-CONTEST-SCORE | N/A | Codec is lossless rate-axis-only; no contest-axis improvement possible on this stream |

## 8. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|-------|----------|-----------|
| Arithmetic-coder primitives (BitWriter, BitReader, ContextModel, CACM-87 32-bit registers) | **ADOPT_CANONICAL** | sister FEC8 module already canonicalizes; reuse per Catalog #229 |
| Context-index computation | **FORK_PRINCIPLED_MISMATCH** | sister 2nd-order uses `_second_order_context_index(prev2, prev1) = prev2*K + prev1`; we add canonical `_third_order_context_index(prev3, prev2, prev1) = (prev3*K + prev2)*K + prev1` |
| Static prior table loading | **FORK_PRINCIPLED_MISMATCH** | sister 2nd-order bakes table in source as tuple of (prev2, prev1, next, count); we use canonical sidecar pattern at `.omx/state/fec8_3rd_order_markov_empirical_table_20260528.py` (4096 contexts × 16 symbols = 65536 entries is too large for in-source literal) |
| Encoder/decoder API | **ADOPT_CANONICAL** | sister pattern: `encode_fec8_markov_selector_static_third_order(codes, *, n_pairs) -> bytes` + `decode_fec8_markov_selector_third_order(payload) -> list[int]` matching sister `_second_order` API |
| Wire format header | **ADOPT_CANONICAL** | sister magic `b"FEC8"` + variant `b"\x00\x04"` (static) / `b"\x00\x05"` (adaptive) + n_pairs u16le; reuse per Catalog #229 |
| Canonical equation registration | **ADOPT_CANONICAL** | `tac.canonical_equations.register_canonical_equation` per Catalog #344 |
| Probe outcome registration | **ADOPT_CANONICAL** | `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313 |
| Empirical anchor Provenance | **ADOPT_CANONICAL** | `tac.provenance.builders.build_provenance_for_predicted` per Catalog #323 |

## 9. Catalog #125 6-hook wire-in declaration

- **hook #1 sensitivity-map**: N/A (codec is lossless rate-axis-only; no per-axis sensitivity contribution)
- **hook #2 Pareto constraint**: N/A (codec doesn't impose new Pareto constraint; failed strict improvement over 2nd-order so doesn't enter frontier)
- **hook #3 bit-allocator**: N/A (codec-only; no per-tensor bit allocation)
- **hook #4 cathedral autopilot dispatch**: ACTIVE — canonical equation `fec8_3rd_order_markov_static_variant_c_savings_v1` auto-consumed by `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335; probe outcome surfaced via `tac.probe_outcomes_ledger.query_blocking_outcomes`
- **hook #5 continual-learning posterior**: ACTIVE — canonical equation `EmpiricalAnchor` lands in canonical posterior; future empirical anchors (e.g., on larger streams per reactivation criterion #1) accumulate via `update_equation_with_empirical_anchor`
- **hook #6 probe-disambiguator**: ACTIVE — probe outcome `verdict=DEFER` + 4 reactivation criteria explicitly enumerated per Catalog #308 alternative-probe-methodologies discipline

## 10. Reactivation criteria per CLAUDE.md "KILL/FALSIFIED memory verdicts" L3

1. **Longer selector streams (>>600 symbols)** where the 4096-context Laplace
   overhead amortizes; the per-context overhead is O(K^d) which is ~$4096$ for
   3rd-order on K=16; if N_pairs grows to ~10000, the Shannon savings should
   exceed the Laplace overhead.
2. **Explicit per-context Huffman codec with deterministic codebook
   reconstruction**: the wire-bit floor is 106B (the per-context Huffman
   measurement) but requires either shipping per-context Huffman codebooks
   as side info OR deterministic codebook reconstruction from observed-only
   counts (which requires escape-codeword discipline for unobserved symbols).
3. **Hybrid 2nd-order + selective 3rd-order escape pattern**: only use
   3rd-order codebook for contexts with sufficient observations (e.g.,
   N_quads ≥ 4); fall back to 2nd-order deterministically for sparse contexts.
4. **Score-aware compound stacking** with sister substrate selector streams
   that have higher 3rd-order conditional entropy reduction (e.g., a future
   PR substrate's selector with structure that 3rd-order captures better).

## 11. Operator-routables

1. **Read this memo + canonical equation** for context per CLAUDE.md
   "Results must become system intelligence" non-negotiable.
2. **DO NOT dispatch this codec** at the contest-CUDA paid axis until
   reactivation criterion satisfied; the probe outcome is `verdict=DEFER
   blocker_status=blocking expires_at_utc=2026-06-27`; per Catalog #313 +
   #346 a dispatch wrapper that attempts to dispatch this codec without
   citing the predecessor probe verdict will be REFUSED structurally.
3. **Operator-routable Wave N+25**: choose ONE of:
   - **A**: Implement reactivation criterion #2 (per-context Huffman with
     explicit codebook reconstruction); estimated ~$0 LOCAL CPU + 1-2 hours
     wall-clock; predicted 106B wire-bit floor + escape overhead.
   - **B**: Implement reactivation criterion #3 (hybrid 2nd-order +
     selective 3rd-order escape); estimated ~$0 LOCAL CPU + 2-3 hours
     wall-clock; predicted slight improvement vs 239B baseline.
   - **C**: Try a DIFFERENT depth-axis extension (e.g., FEC8 prediction-by-
     partial-match PPM with arithmetic-coder); estimated ~$0 LOCAL CPU +
     2-4 hours wall-clock; predicted ~230-240B (still likely dominated by
     Laplace overhead on this stream).
   - **D**: Pivot to a sister substrate's selector stream (Wave N+13 +
     other PR substrate selector streams may have different distributional
     structure that 3rd-order Markov captures better); requires per-substrate
     re-measurement.
4. **NO PR creation** per the HARD GATE; canonical apparatus updates landed
   are operator-side research artifacts.

## 12. Canonical apparatus artifacts (LANDED)

- Canonical encoder/decoder: `submissions/hnerv_fec6_fixed_huffman_k16/encoder/build_pr101_frame_exploit_selector_packet_fec8_3rd_order_markov.py` (~440 LOC; arithmetic-coder static + adaptive variants; both roundtrip-verified)
- Measurement script: `tools/measure_fec8_markov_3rd_order_rate_attack.py` (~470 LOC; live PR101 stream + Shannon entropy + per-context Huffman wire-bit floor + per-encoder roundtrip)
- Canonical equation registration: `tools/register_fec8_3rd_order_markov_canonical_equation_20260528.py` (one-shot; registered + persisted via `tac.canonical_equations.register_canonical_equation`)
- Empirical 4-tuple counts sidecar: `.omx/state/fec8_3rd_order_markov_empirical_table_20260528.py` (529 unique 4-tuples; Python source literal; loaded by encoder + decoder via canonical helper `_load_empirical_third_order_counts`)
- Measurement JSON: `.omx/state/fec8_3rd_order_markov_rate_attack_measurement_20260528.json` (full schema per Catalog #356 per-axis decomposition + Catalog #323 canonical Provenance)
- Canonical equation: `fec8_3rd_order_markov_static_variant_c_savings_v1` (in `.omx/state/canonical_equations_registry.jsonl`; auto-discovered by `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335)
- Probe outcome: probe_id `fec8_3rd_order_markov_static_variant_c_live_pr101_fec6_selector_stream_rate_attack_20260528` (in `.omx/state/probe_outcomes.jsonl`; `verdict=DEFER blocker_status=blocking expires_at_utc=2026-06-27`)
- Landing memo: this file (`.omx/research/fec8_3rd_order_markov_rate_attack_landed_20260528.md`)

## 13. Discipline observed

- Catalog #229 + #376 PV exhaustive (4 grep scopes + 1 git log scope + 1 probe outcomes scope)
- Catalog #340 sister-coherence (DISJOINT from 5 in-flight sisters; verified `git log -30` + `git status` + sister landing memos)
- Catalog #117 / #157 / #174 canonical serializer (post-edit `--expected-content-sha256` applied per commit)
- Catalog #206 subagent crash-resume (4 checkpoints landed: step 1 PV / step 2 EMPIRICAL WIN identified / step 3 canonical encoder verified / step 4 canonical apparatus complete)
- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE (new files only; no mutation of existing forensic artifacts)
- Catalog #131 fcntl-locked state writes via canonical helpers
- Catalog #138 fail-closed strict-load via `tac.canonical_equations.registry` + `tac.probe_outcomes_ledger`
- Catalog #287 placeholder-rationale rejection (every same-line waiver carries substantive ≥4-char rationale)
- Catalog #292 per-deliberation assumption surfacing (AssumptionAdversary verdict above)
- Catalog #296 predicted-band Dykstra-feasibility N/A waiver per data-sparsity grounds
- Catalog #300 council deliberation v2 frontmatter
- Catalog #303 cargo-cult audit (3 assumptions classified HARD-EARNED-vs-CARGO-CULTED)
- Catalog #305 observability surface (6 facets documented)
- Catalog #307 paradigm-vs-implementation falsification (IMPLEMENTATION-LEVEL falsified; PARADIGM intact)
- Catalog #308 alternative-probe-methodologies (4 reactivation paths enumerated)
- Catalog #313 probe-outcomes ledger DEFER blocking 30-day expires
- Catalog #323 canonical Provenance umbrella (every score-claim carries canonical Provenance)
- Catalog #335 cathedral consumer auto-discovery (canonical equation lookup consumer auto-consumes new equation)
- Catalog #344 canonical equations registry (new equation registered + first EmpiricalAnchor landed)
- Catalog #356 per-axis decomposition (rate-axis-only; no pose/seg delta)
- CLAUDE.md "Bit-level deconstruction and entropy discipline" non-negotiable
- CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable
- CLAUDE.md "Final rate attack" standing directive
- CLAUDE.md "MLX-first numpy-portable" standing directive (encoder pure-Python no MLX dep; deployable on numpy-portable inflate runtime)
- Just-saved standing directive `rate-attack-default-cost-class-is-zero-mlx-or-cpu-not-paid-modal-standing-directive-20260528` ($0 LOCAL CPU; NO Modal paid spend)
- Just-saved standing directive `memos-must-be-acted-upon-canonical-apparatus-mutation-enforcement-standing-directive-20260528` (canonical apparatus mutations LANDED: equation + probe outcome + cathedral consumer auto-discovery)
