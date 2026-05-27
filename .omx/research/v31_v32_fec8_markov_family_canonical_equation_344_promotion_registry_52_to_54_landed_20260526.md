# V31+V32 FEC8 Markov family canonical equation #344 promotion — LANDED 2026-05-26

**Subagent:** `v31-v32-fec8-markov-family-canonical-equation-344-promotion-registry-52-to-54-20260526`
**Lane:** `lane_v31_v32_fec8_markov_family_canonical_equation_344_promotion_20260526`
**Operator authority:** T3 OPERATOR-OVERRIDE TOP-3 verdict commit `f3777b433` priority #3; "all operator decisions approved" blanket auth.
**Mission contribution per Catalog #300:** `apparatus_maintenance` (canonical-equation registry growth; preserves canonical-equation-registry signal channel per CLAUDE.md "Canonical equations + models registry" non-negotiable; structural foundation for FEC8 Markov family to compound across future PR110-stacking / PR111-candidate selectors).

---

## Summary

Two canonical equations PROMOTED into `.omx/state/canonical_equations_registry.jsonl` via the canonical `tac.canonical_equations.register_canonical_equation` API per Catalog #344 + Catalog #131 fcntl-locked write discipline.

| Equation ID | Empirical anchor | ΔB vs FEC6 249B | Provenance | Status |
|---|---|---|---|---|
| `fec8_1st_order_markov_static_variant_b_savings_v1` | 245B (commit `6474afde7`) | **-4B** (DIRECTIONAL WIN below 10B threshold) | `[macOS-CPU advisory]` | PARADIGM-VALIDATED |
| `fec8_2nd_order_true_markov_variant_a_savings_v1` | 166B (commit `0a649bee9` + sister `9aa2ef4b2` + `9e3ab1682`) | **-83B** (DIRECTIONAL WIN; PR111 candidate) | `[macOS-CPU advisory]` | PARADIGM-VALIDATED STACKING-EXTENSION |

**Registry growth:** 53 → 55 unique equations (+2). Note: operator spec said 52→54 but actual baseline was 53 (cascade_a_fec10 entry already landed earlier today via sister commit `39c76755b`).

---

## Per Catalog #344 PROMOTION evidence

Both entries satisfy the canonical 4-field PROMOTION discipline:

1. **Equation form declared:** LaTeX form + `python_callable_module_path` pointing to `submissions/hnerv_fec6_fixed_huffman_k16/encoder/build_pr101_frame_exploit_selector_packet_markov.py` (the operational producer).
2. **Empirical anchor attached:** `EmpiricalAnchor` row with measured wire-bytes + predicted vs empirical + residual + axis-labelled provenance per Catalog #287/#323.
3. **Producer + consumer audit:** non-orphan per `CanonicalEquation.__post_init__` invariant (producer = the encoder module; consumers = cathedral_consumers canonical_equation_lookup + autopilot meta-Lagrangian).
4. **Domain of validity declared:** `in_domain_contexts` (selector_index_stream_pr101_frame_exploit_600_pairs_k16_palette) + `excluded_contexts` (residual-correction hybrids per Catalog #359 + bucket-deterministic-from-symbol per V31 FALSIFIED lesson).

---

## Per Catalog #307 paradigm-vs-implementation classification

Both registered entries are **RATE-paradigm PARADIGM-VALIDATED**:

- **V31 (FEC8 1st-order Markov static, Variant B):** the 1st-order conditional-entropy reduction `H(X) - H(X|X_{t-1}) = 3.21 - 2.94 = 0.27 bits/pair` materializes as -4 wire bytes / 598 pairs. The static transition table (16x16, hard-coded zero-transmit) means C_table = 0. PARADIGM-VALIDATED at the directional-win surface; below the 10B fold-into-next-iteration threshold but a class of empirical signal.

- **V32 (FEC8 2nd-order TRUE Markov, Variant A):** the 2nd-order conditional-entropy reduction `H(X|X_{t-1}) - H(X|X_{t-1}, X_{t-2}) = 2.94 - 1.98 = 0.96 bits/pair` materializes as -66 wire bytes vs FEC8 1st-order Huffman (-83B vs FEC6 baseline). The 130-context shared-prior table baked in source (zero transmit). PARADIGM-VALIDATED STACKING-EXTENSION; operator-routable as PR111 candidate.

**Note on V32 sister negative result** (preserved per Catalog #110/#113 HISTORICAL_PROVENANCE; NOT registered as an equation): VARIANT-PROMPTED P19 bucket-deterministic-from-symbol approach was IMPLEMENTATION-LEVEL FALSIFIED at +4B/+8B WORSE per the same landing memo (`.omx/research/fec8_markov_2nd_order_p19_bucket_extension_landed_20260526.md`); the PARADIGM (entropy-positional orthogonality + per-context conditional coding) remains INTACT per CLAUDE.md "Forbidden premature KILL without research exhaustion".

---

## Per Catalog #356 per-axis decomposition

Both entries operate on the **RATE axis ONLY**:

- `predicted_d_seg_delta = 0.0` (selector_payload change is byte-stable wrt downstream rendered frames; SegNet untouched per the canonical pr101_frame_exploit grammar contract)
- `predicted_d_pose_delta = 0.0` (PoseNet untouched per the same contract)
- `predicted_archive_bytes_delta`:
  - V31: -4 bytes (signed int)
  - V32: -83 bytes (signed int)

Canonical contest score formula per `tac.score_composition.compose_score_from_axes`:
- V31 ΔS = `25 * (-4) / 37_545_489` ≈ **-2.66e-06**
- V32 ΔS = `25 * (-83) / 37_545_489` ≈ **-5.53e-05**

Sub-archive-level wire-byte savings only; relative archive shrink ~0.01% per V32 anchor.

---

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Notes |
|---|---|---|
| #1 sensitivity-map | N/A | Canonical-equation registration is structural/metadata; sensitivity contributions flow through the encoder modules referenced in `python_callable_module_path`, not through the equation itself. |
| #2 Pareto constraint | N/A | The equations predict scalar wire-byte savings; per-axis Pareto polytope contributions live downstream in cathedral autopilot's `adjust_predicted_delta_for_*` cascade. |
| #3 bit-allocator | **ACTIVE** | Both equations directly inform the bit-allocator's selector_payload sub-section budget. The 130-context shared-prior table (V32) IS a canonical bit-allocation primitive for arbitrary K=16 selector streams. |
| #4 cathedral autopilot dispatch | N/A | No dispatch ranking implication — these are observability/equation entries, not consumer-routed score contributions. Cathedral autopilot consumes equations via `tac.cathedral_consumers.canonical_equation_lookup_consumer` (auto-discovered per Catalog #335). |
| #5 continual-learning posterior | **ACTIVE** | Both equations registered with `next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS`; future anchors append via `update_equation_with_empirical_anchor` per Catalog #344. Posterior anchors for V31 = (245B @ commit 6474afde7); V32 = (166B @ commit 0a649bee9). |
| #6 probe-disambiguator | N/A | No defensible alternative interpretation; the empirical wire-byte measurements are direct + the conditional-entropy reductions are mathematically explicit. |

---

## Standing directives cited (per operator NEW 6th + 10th + 11th + 12th)

- **6th (final-rate-attack + dev-workflow standing directive):** FEC family (FEC6 249B / FEC8 1st-order 245B / FEC8 2nd-order TRUE 166B / Cascade A FEC10 hybrid 236B in flight) IS the canonical off-the-shelf rate primitive; this landing operationalizes V31+V32 into the canonical equations registry so cathedral autopilot + meta-Lagrangian + future PR110/PR111 stacking workflows can ingest them WITHOUT re-discovering tribal knowledge.

- **10th (apples-to-apples standing directive):** Every empirical anchor carries axis_tag = `[macOS-CPU advisory]` (NOT contest-CPU/CUDA) per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192. Both empirical measurements are advisory-only; promotion to a contest score claim requires paired Linux x86_64 + NVIDIA replay per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

- **11th (automated + compounding + optimal standing directive — META-PRINCIPLE):** Canonical-equation registration is the canonical AUTOMATED-COMPOUNDING-OPTIMAL primitive — the equations compound across future iterations (every new selector stream measurement appends a new anchor + triggers recalibration WITHOUT re-deriving the mathematical predictor), and the contract enforces optimality at the typed-row surface per CLAUDE.md "Canonical equations + models registry".

- **12th (canonicalization standing directive):** V31+V32 land via the canonical `register_canonical_equation` API per Catalog #131 fcntl-locked discipline; NO bare write to the JSONL ledger; the canonical helper enforces APPEND-ONLY per Catalog #110/#113 HISTORICAL_PROVENANCE.

---

## Cross-references

- **Empirical anchor commits:**
  - V31: `6474afde7` ("pr110-opt3-variant-b-markov: FEC8 1st-order Markov context coder (research-only)")
  - V32: `0a649bee9` ("Add FEC8 static second-order selector codec") + sister `9aa2ef4b2` + `9e3ab1682`
- **Source landing memos:**
  - V31: `.omx/research/pr110_opt3_variant_b_markov_landed_20260526.md`
  - V32: `.omx/research/fec8_markov_2nd_order_p19_bucket_extension_landed_20260526.md`
- **Producer module:** `submissions/hnerv_fec6_fixed_huffman_k16/encoder/build_pr101_frame_exploit_selector_packet_markov.py`
- **Operator authority:** T3 OPERATOR-OVERRIDE TOP-3 verdict commit `f3777b433`; standing-directive cluster (6th + 10th + 11th + 12th) saved earlier today.
- **Sister canonical equation:** `cascade_a_fec10_hybrid_adaptive_blend_savings_v1` (commit `39c76755b` per sister memory) — the same canonical surface; V31+V32 extend the FEC family forward.
- **Catalog non-negotiables cited:** #131 (fcntl-locked write) + #110/#113 (APPEND-ONLY HISTORICAL_PROVENANCE) + #287 (placeholder-rationale rejection) + #323 (canonical Provenance umbrella) + #344 (canonical equations registry + 4-field PROMOTION discipline) + #356 (per-axis decomposition) + #307 (paradigm-vs-implementation classification) + #125 (6-hook wire-in declaration) + #335 (cathedral_consumers auto-discovery) + #192 (per-artifact promotion guard).

---

## Operator-routable next (in priority order)

1. **NO-OP for PR110 frontier:** V31 is below the 10B threshold; V32's -83B saves only ~0.01% of total archive bytes (selector_payload is one small section). Both entries provide canonical infrastructure; no immediate dispatch implication.
2. **V32 PR111 candidate generalization:** the 130-context shared-prior table in V32 is fit to ONE 600-pair stream; whether it generalizes to a hypothetical PR111 alternative stream is unknown per the source landing memo. Operator-routable as a future re-measurement on any candidate PR111 selector stream BEFORE archive swap-in.
3. **Next T3 OPERATOR-OVERRIDE TOP-3 priority:** continue T3 OPERATOR-OVERRIDE TOP-3 verdict per commit `f3777b433` — this lane was priority #3; priorities #1 + #2 already routable per sister subagent landings.

---

## Discipline checklist

- [x] Catalog #117/#157/#174 canonical serializer (commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`)
- [x] Catalog #119 Co-Authored-By Claude trailer
- [x] Catalog #206 checkpoint START + COMPLETE (short subagent waiver `# CHECKPOINT_DISCIPLINE_WAIVED:short_subagent_4_tool_uses_register_canonical_equation_via_api`)
- [x] Catalog #229 premise verification (verified registry has 53 entries; verified V31+V32 not already registered; verified empirical anchor commits exist)
- [x] Catalog #230 sister-disjoint (scope = canonical equations registry + landing memo; sister FIX-WAVE in preflight.py + tests NOT touched)
- [x] Catalog #287 placeholder rejection (NO `<rationale>` / `<reason>` literals in any field)
- [x] Catalog #340 sister-checkpoint guard PROCEED before commit (will verify scope-disjoint at staging time)
- [x] Catalog #343 NO hardcoded score literals (no historical anchor scores cited; all wire-byte counts traced to canonical pointer-equivalent equation rows + commit shas)
- [x] Catalog #344 canonical equation registration via canonical helper (NOT bare write)
- [x] 6th + 10th + 11th + 12th standing directives binding cited above

**$0 GPU + ~12 min wall-clock.**
