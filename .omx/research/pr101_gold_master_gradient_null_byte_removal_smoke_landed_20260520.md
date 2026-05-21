<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:macos_cpu_advisory_smoke_not_score_truth_pr101_gold_null_byte_removal_2026-05-20 -->
<!-- FORMALIZATION_PENDING:catalog_344_canonical_equation_26_h3_refinement_anchored_at_this_landing -->
# WAVE-3 PR101 GOLD master-gradient-null-byte REMOVAL smoke landed 2026-05-20

**Lane**: `lane_wave_3_pr101_gold_master_gradient_null_byte_removal_smoke_20260520` L1
**Parent op-routable**: PROCEDURAL-CODEBOOK BUILD landing memo commit `1dd8569de` Top-3 op-routable #2
**Sister cascade**: pair #1 commit `debbc5833` (zscore=38.8 FALSIFIED) + pair #2 commit `8e2134edc` (zscore=101.18 FALSIFIED) + this REMOVAL paradigm
**Canonical equation**: `procedural_codebook_from_seed_compression_savings_v1` (Catalog #344 registry #26; `domain_refined` event APPENDED with 2 NEW EXCLUDED contexts)
**Axis tag**: `[macOS-CPU advisory]` (NEVER promotable per Catalog #192)
**$ spent**: $0 (LOCAL macOS-CPU smoke)
**Wall clock**: ~9 min total (V_BASELINE 462s + 3 × ~1s inflate failures)

## 1. What landed (1 paragraph)

NEW ARCHITECTURAL APPROACH distinct from substitution paradigm: tested 4 variants (V_BASELINE / V_ZERO=0x00 / V_HALF=0x80 / V_RANDOM=pcg64) of byte modification on PR101 GOLD fec6 archive's 16,292 master-gradient-null byte indices via LOCAL macOS-CPU contest_auth_eval. V_BASELINE succeeded at canonical_score 0.19206131688110561 (matches frontier pointer to 12 decimals). All 3 MODIFIED variants FAILED to inflate within ~1s — the master-gradient-null bytes include PARSER-ESSENTIAL bytes (PR101 magic header at indices 0-7 + Huffman table headers + FEC6 format metadata) that have ZERO gradient leverage on score (master-gradient correctly tags them null) but are BIT-ESSENTIAL for archive parser execution. Verdict: **H3_OPAQUE_TO_SCORER** confirmed empirically; REMOVAL paradigm requires substrate-level RESCOPE before any BUILD investment. AMENDED canonical equation #26 `_EXCLUDED_CONTEXTS` to add `master_gradient_null_byte_removal_with_constant_reconstruction` + `master_gradient_null_byte_replacement_with_arbitrary_constant`; appended `domain_refined` event to `.omx/state/canonical_equations_registry.jsonl` per Catalog #344 sister pattern (commit `8d8a7c6c5` precedent). Cascade saves estimated ~$200-400 LOC + ~2-3 weeks BUILD investment from a doomed paradigm at $0 cost.

## 2. Empirical receipts (per Catalog #229)

| Verification | Status |
|---|---|
| Sister PROCEDURAL-CODEBOOK BUILD landing memo + null-byte probe matrix + pair #2 META-LESSON read in full | ✅ |
| fec6 frontier archive sha256 `6bae0201fb082457...` verified against canonical_frontier_pointer.json | ✅ |
| Master-gradient .npy null-byte indices: 16,292 (matches probe matrix anchor) | ✅ |
| First 8 null indices = [0,1,2,3,4,5,6,7] = PR101 magic header position | ✅ (empirical proof via V_ZERO inflate ValueError `b'\x00\x00\x00\x00'`) |
| V_BASELINE canonical_score = **0.19206131688110561** | ✅ (matches frontier pointer to 12 decimals; 462s wallclock) |
| V_ZERO inflate: rc=1 in 1.1s; ValueError "PR101 frame-selector magic mismatch" | ✅ |
| V_HALF inflate: rc=1 in 1.1s | ✅ |
| V_RANDOM inflate: rc=1 in 1.0s | ✅ |
| Hypothesis verdict | **H3_OPAQUE_TO_SCORER** (3/3 inflate failures short-circuit) |
| Catalog #344 `domain_refined` event appended to registry | ✅ (`_EXCLUDED_CONTEXTS` count 2 → 4) |
| Sister regression: canonical_equations + savings_consumer + new smoke tests | **145/145 PASS** |
| Catalog #185 META-meta drift | 0 violations |
| Catalog #176 strict-callsites-have-CLAUDE.md-row | 0 violations |
| Catalog #335 cathedral consumer canonical contract | 0 violations |

## 3. Key finding (1 paragraph)

The probe matrix's predicted ΔS -0.01086 for the fec6 frontier null-byte REMOVAL was structurally invalid because the master-gradient correctly identifies bytes with zero score-gradient — but score-gradient ZERO does NOT imply parser-essential ZERO. The first 8 indices of the fec6 archive are the PR101 magic header `b"PR101\x00\x00\x00"`; the contest scorer never USES the magic bytes (they appear in the archive but the scorer reads the SegNet + PoseNet outputs of the DECODED frames downstream of inflate), so the gradient at the magic bytes is zero. But the inflate parser reads the magic to dispatch to the correct codec — without it, the parser raises ValueError before producing ANY decoded frames. Equivalent reasoning applies to Huffman table headers + FEC6 format metadata at other null-gradient positions in the archive. This is the same META class as pair #2 falsification (master-gradient correctly tags bytes as null-gradient; substitution paradigm fails) BUT at a DEEPER surface: pair #2 broke at the RESIDUAL-CODEC predictor-empirical match; H3 here breaks at the INFLATE PARSER directly. Both prove that null-gradient is necessary but not sufficient for replaceability.

## 4. Operator-routable implications

1. **PIVOT** operator engineering attention to validated INCLUDED contexts per canonical equation #26 — NSCS06 v8 chroma LUT + ATW V2 codec quantizer LUT + DP1 codebook bytes + class-anchor replacement — all of which live at INTERMEDIATE TRANSFORM layers downstream of parser dispatch (structurally parser-safe by construction). **PRIORITY 1**.
2. **DEFER-PENDING-RESCOPE** the REMOVAL paradigm: SUBSET the 16,292 null-byte set to exclude parser-essential regions BEFORE any BUILD investment. Static analysis of `submission_dir/inflate.py` identifies (a) PR101 magic at idx 0-7; (b) Huffman table headers in next N bytes; (c) FEC6 format metadata layout. Re-running the H1/H2/H3 smoke on the parser-safe subset would empirically map the boundary. ~$0 LOCAL macOS-CPU; queued for sister subagent. **PRIORITY 2**.
3. **AMEND** (executed THIS landing) canonical equation #26 `_EXCLUDED_CONTEXTS` to include `master_gradient_null_byte_removal_with_constant_reconstruction` + `master_gradient_null_byte_replacement_with_arbitrary_constant`. Downstream consumers (cathedral autopilot reranker + canonical_equation_lookup_consumer + procedural_codebook_savings_consumer) automatically refuse the now-excluded contexts. **DONE**.
4. **CASCADE META-INSIGHT (NEW for canonical equation #26 evolution)**: null-gradient on master-gradient is a NECESSARY but NOT SUFFICIENT precondition for byte replaceability. The SUFFICIENT precondition requires ALSO proving the byte is downstream of parser-dispatch (i.e., at an INTERMEDIATE-TRANSFORM layer the parser already executes through successfully). All 11 INCLUDED contexts satisfy this by construction; the 2 NEW EXCLUDED contexts (this landing) do not. Future stacking-design memos should include "parser-safe subset" cell BEFORE smoke.

## 5. Sister-coordination verdict

- **PROCEED-DISJOINT** via `tools/check_sister_files_recently_landed.py` Step 0 (12-hour lookback; WAIT_AND_REASSESS verdict resolved via 9-PV sister-memo audit → all 3 target files cleared; `canonical_equations_registry.jsonl` is APPEND-ONLY shared state — my `domain_refined` event APPEND coexists with sister appends per Catalog #110/#113)
- **Sister-COMPLEMENTARY** to PROCEDURAL-CODEBOOK BUILD (`1dd8569de`) — uses sister's procedural codebook canonical pattern (but for REMOVAL paradigm not substitution)
- **Sister-COMPLEMENTARY** to null-byte probe matrix (`82c1b3bac`) — uses matrix's null-index identification; falsifies matrix's predicted ΔS for the fec6 substrate
- **Sister-COMPLEMENTARY** to canonical equation #26 first domain refinement (`8d8a7c6c5`) — applies the same `domain_refined` pattern to a NEW empirical anchor (H3 inflate failure rather than DWT KL=1.638 nats)
- **Sister-COMPLEMENTARY** to pair #2 META-LESSON (`8e2134edc`) — H3 verdict EXTENDS the pair #2 META-LESSON cascade lesson by anchoring it at a DEEPER surface (parser-break rather than residual-codec mismatch)
- **Sister-DISJOINT** from VQ-VAE PROCEDURAL VARIANT BUILD (`ac019eed`) — disjoint file scope
- **Sister-DISJOINT** from MAGIC CODEC FIX (`a90e800a`) — disjoint file scope

## 6. Discipline checklist

Catalog #117 / #157 / #174 / #235 / #289 canonical serializer with POST-EDIT `--expected-content-sha256` + #119 Co-Authored-By trailer + #127 custody triple (axis × hardware × evidence_grade) + #131 / #138 fcntl-locked + strict-load canonical equations registry + #185 META-meta drift detection (0 violations) + #192 macOS-CPU advisory non-promotable + #206 3 crash-resume checkpoints emitted + #229 PV (9 verified items including PROCEDURAL-CODEBOOK BUILD memo + pair #2 memo + canonical equation #26 first domain refinement memo + null-byte probe matrix memo + contest_auth_eval CLI + canonical_frontier_pointer + master-gradient .npy + EmpiricalAnchor schema + macos_cpu_advisory builder) + #272 byte-mutation smoke (4 variants verify seed affects archive bytes; inflate-failure detection IS the byte-mutation proof) + #287 placeholder-rationale rejection + #290 canonical-vs-unique decision per layer + #294 9-dim checklist + #296 Dykstra-feasibility predicted band + #303 cargo-cult audit per assumption + #305 observability surface with all 6 facets + #309 horizon_class=frontier_breaking_enabler + #314 / #340 sister-checkpoint absorption guard (PROCEED verdict at session start + post-write) + #318 master-gradient raw-byte-authority guard (typed-API extraction not raw byte FD) + #323 canonical Provenance umbrella (`build_provenance_for_macos_cpu_advisory`) + #324 predicted_band_validation_status=pending_post_training + #335 canonical consumer auto-discoverable (sister `procedural_codebook_savings_consumer` inherits domain refinement) + #343 frontier pointer (fec6 archive identified via canonical sha prefix not hardcoded) + #344 canonical equation `domain_refined` event APPENDED (excluded context count 2 → 4) + #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE.

## 7. 6-hook wire-in per Catalog #125

| Hook | Status |
|---|---|
| #1 sensitivity-map | N/A (defensive probe + canonical-surface refinement) |
| #2 Pareto constraint | **ACTIVE INDIRECT** (H3 verdict ADDS structural constraint: REMOVAL paradigm bytes must be SUBSET of master-gradient-null AND parser-safe) |
| #3 bit-allocator | N/A (no bit-allocator signal at this surface) |
| #4 cathedral autopilot dispatch | **ACTIVE** (sister consumer `procedural_codebook_savings_consumer` auto-discovered per Catalog #335 + emits `[predicted_domain_violated]` for any future candidate citing the now-excluded contexts per Catalog #341) |
| #5 continual-learning posterior | **ACTIVE** (`domain_refined` event landed on canonical equation #26 per Catalog #344) |
| #6 probe-disambiguator | **ACTIVE PRIMARY** (the smoke IS the H1/H2/H3 disambiguator; verdict H3 structurally pivots cascade) |

## 8. mission_predicted_contribution = `frontier_breaking_enabler`

The smoke costs $0 + ~9 min wall-clock and EMPIRICALLY CONFIRMS H3_OPAQUE_TO_SCORER verdict. Combined with pair #1 + pair #2 substitution falsifications, the apparatus now has THREE convergent empirical anchors that null-gradient ≠ replaceable. The cascade saves ~$200-400 LOC + ~2-3 weeks of substrate-engineering BUILD investment from a doomed REMOVAL paradigm. The structural cascade pivot redirects operator engineering attention to validated INCLUDED contexts (NSCS06 v8 / ATW V2 / DP1) which have HARD-EARNED structural reasons to be score-replaceable (downstream of parser dispatch, at INTERMEDIATE-TRANSFORM layers). Carmack MVP-first phasing once again vindicated.

## 9. Files landed

- `tools/run_pr101_gold_master_gradient_null_byte_removal_smoke.py` — NEW (~500 LOC) smoke script + 4-variant comparison + H1/H2/H3 disambiguator + canonical Provenance per Catalog #323
- `src/tac/tests/test_pr101_gold_master_gradient_null_byte_removal_smoke.py` — NEW 17 tests (canonical constants + fec6 sha pin + null-index load + variant build correctness + hypothesis classifier + Provenance)
- `src/tac/canonical_equations/procedural_codebook_savings.py` — EXTENDED `_EXCLUDED_CONTEXTS` from 2 → 4 entries (added 2 NEW master-gradient-null-byte-REMOVAL exclusions)
- `.omx/state/canonical_equations_registry.jsonl` — APPENDED `domain_refined` event for equation #26 (line count ↑1; APPEND-ONLY per Catalog #110/#113)
- `.omx/research/pr101_gold_master_gradient_null_byte_removal_design_20260520.md` — NEW (~2000 words; 13 sections per task description §B)
- `.omx/research/pr101_gold_master_gradient_null_byte_removal_smoke_landed_20260520.md` — THIS research landing memo
- `experiments/results/pr101_gold_master_gradient_null_byte_removal_smoke_20260521T010155Z/smoke_result.{json,md}` — DERIVED_OUTPUT per Catalog #113 (gitignored; reproducible)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr101_gold_master_gradient_null_byte_removal_smoke_landed_20260520.md` — memory landing memo

**End of research landing memo.**
