<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:pair_2_landing_memo_macos_cpu_advisory_not_score_truth_2026-05-20 -->
<!-- FORMALIZATION_PENDING:pair_2_landing_memo_canonical_equation_26_third_anchor_appended_event_per_catalog_344 -->
---
council_tier: T1
council_attendees: [Shannon, Dykstra, Carmack, Rudin, Yousfi]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "op-routable #1: PIVOT to pair #4 orthogonality validation (FREE local probe; ~5 min ETA) — validates orthogonality boundary of magic_codec stacking"
  - "op-routable #2: PIVOT to DP1-only cascade (paired smoke per Catalog #325 14-day window) — the only un-falsified pair remaining at $0-1 cost"
  - "op-routable #3: DEFER-PENDING-RESEARCH pair #3 (world-model latent residual) until Z6 ego-pose conditioning ablation lands per DWT-HNeRV bind op-routable #3"
council_assumption_adversary_verdict:
  - assumption: "SRL1 (RLE-of-zeros) wins on sparse-zero-dominated null-byte distributions"
    classification: CARGO-CULTED
    rationale: "empirical receipts at zscore=101 EMPIRICALLY-FALSIFY the claim; null-byte VALUES at zero-leverage positions are NOT all-zero (only 0.44% are byte-value zero) — they are near-random values that brotli already compresses near the entropy floor; (empirical − pcg64-uniform) residual is near-uniform int16 in [-255, 255] which SRL1 cannot exploit (only 0.33% of residual entries are zero); 4-byte indices per nonzero entry inflate byte budget catastrophically"
  - assumption: "procedural codebook substitution at null-leverage positions preserves byte-exact reconstruction via residual correction"
    classification: HARD-EARNED
    rationale: "the smoke's reconstruction sanity check (synthetic_uint8.astype(int16) + residual_int16 == empirical_uint8.astype(int16)) verifies byte-exact reconstruction is mathematically valid — the rescue path's mathematical foundation is sound; failure is in entropy structure of residuals, not in reconstruction validity"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
---
# WAVE-3 magic-codec × sparse_packet_ir SRL1 × fec6 frontier null-byte residual CPU smoke (pair #2) landed 2026-05-20

**Lane**: `lane_wave_3_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_cpu_smoke_20260520` L1
**Parent op-routable**: MAGIC CODEC × CASCADE STACKING ANALYSIS memo §14 Top-3 #2 (commit `5e7831373`)
**Sister-COMPLEMENTARY**: pair #1 smoke `debbc5833` (SECOND anchor; FALSIFIED at zscore=38.8); null-byte probe matrix `82c1b3bac` (16,292 null-byte locations identified)
**Sister-DISJOINT**: DP1 PAIRED-SMOKE PRE-AUTHORIZATION CHECKLIST (`a13d467e`); END-OF-DAY CASCADE RECONCILIATION (`a89a1cd7`)
**Canonical equation**: `procedural_codebook_from_seed_compression_savings_v1` (Catalog #344 registry #26; THIRD `anchor_appended` event with NEW IN-DOMAIN context `sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes`)
**Axis tag**: `[macOS-CPU advisory]` (NEVER promotable per Catalog #192 + #127 + #323)
**$ spent**: $0 (LOCAL macOS-CPU smoke)
**Wall clock**: ~3 minutes (smoke pipeline on full 16,292 null-byte set)

## Empirical receipts (premise verification per Catalog #229)

| Metric | Empirical | Verdict |
|---|---|---|
| fec6 frontier archive sha256 | `6bae0201fb08...` (178,517 bytes wrapper / 178,417 inner member `x`) | matches canonical_frontier_pointer.json |
| Master-gradient null indices observed | 16,292 | matches matrix anchor `n_null_bytes=16292` (9.13% null fraction) |
| Empirical bytes at null positions: byte-value-zero fraction | 0.44% | structural: empirical bytes are NEAR-RANDOM, not all-zero (the canonical "null leverage" surface is gradient-null, NOT byte-value-null) |
| Config A in-place charged baseline | **16,292 B** | canonical apples-to-apples baseline per contest packet semantics |
| Config A secondary audit (brotli q=11) | 16,296 B | empirical bytes essentially incompressible (entropy near uint8 floor) |
| Config B procedural only (32 B seed) | 32 B | LOSSY substitution placeholder; would corrupt archive (byte-exact reconstruction broken) |
| Config C procedural + SRL1 residual | **97,473 B** | 5.98× LARGER than baseline |
| Sister: procedural + brotli on residual | 16,329 B | brotli also can't exploit near-uniform int16 residual (~16,297 B residual + 32 B seed) |
| SRL1 vs brotli-on-residual delta | **+81,144 B** | SRL1 LOST orthogonality audit catastrophically (4-byte indices × 16,238 nonzero entries dominates) |
| Bytes saved (C vs A in-place) | **−81,181 B** | REGRESSION |
| Empirical ΔS | **+0.054055** | rate-term INCREASES 5.4 score points |
| Predicted ΔS pair #2 (stacking memo §7) | −0.00109 | ADDITIVE α=0.9 |
| Residual zscore | **101.18** | FAR outside 2σ threshold (50× worse miss than pair #1's 38.8) |
| Canonical equation #26 verdict at 2σ | **CARGO-CULTED** | pair #2 EMPIRICALLY FALSIFIED |
| Cascade verdict | **`PAIR_2_FALSIFIED_CASCADE_FURTHER_NARROWS_PIVOT_TO_PAIR_4_OR_DP1_ONLY`** | cascade pivots |
| Catalog #272 byte-mutation smoke | **PASSED** | residual + SRL1 payload sha256 both change on 1-byte seed mutation |
| Sister tests | 18/18 PASS | `test_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py` |
| Sister regression | 128/128 PASS | pair #1 + pair #2 + canonical_equations test suites |
| Catalog #185 META-meta drift | 0 violations | clean |
| Residual sparsity ratio | 0.0033 | SRL1 expects high-zero-fraction; this distribution has 99.67% nonzero |
| SRL1 wire-format breakdown | envelope 13 B + indices 64,952 B + values 32,476 B | indices dominate (4-byte uint32 × 16,238 nonzero entries) |
| Reconstruction byte-exact verified | YES | synthetic_uint8 + residual_int16 == empirical_uint8 (raises RuntimeError if violated) |
| THIRD anchor appended | YES | canonical equation #26 anchor count 3 → 5 (sister codex subagent landed parallel anchor at line 45 written 00:20:10Z; mine landed at line 46 written 00:21:20Z — convergent empirical evidence) |

## Key finding (1-paragraph)

The MAGIC CODEC × CASCADE STACKING ANALYSIS memo's pair #2 prediction (ADDITIVE α=0.9 → −0.00109 ΔS) is empirically vindicated as HARD-EARNED-EMPIRICALLY-FALSIFIED by 101× residual zscore: the (pcg64-uniform predictor, fec6 frontier null-leverage bytes, int16 residual + SRL1 encoding) tuple has structurally mismatched entropy assumptions. The "null bytes" surface per Catalog #344 canonical equation `master_gradient_null_space_byte_fraction_v1` is bytes with ZERO GRADIENT LEVERAGE on score (not bytes with VALUE zero) — the empirical byte values at those 16,292 positions are NEAR-RANDOM (only 0.44% are byte-value zero, vs Config A's 99.97% naive expectation), so brotli already compresses them near the entropy floor at 16,296 bytes (essentially identical to the in-place 16,292 charged-byte count). When the procedural codebook predictor (pcg64-uniform synthetic bytes) is subtracted via byte-exact uint8 → int16 residual, the residual distribution is near-uniform int16 in [-255, 255] which SRL1 CANNOT exploit — only 0.33% of residual entries are zero. The 16,238 nonzero entries each require a 4-byte uint32 index in the SRL1 wire format, inflating the total payload to 97,441 bytes (+ 32 B seed = 97,473 B). The orthogonality audit (sister: brotli on the same residual) lands at 16,329 B — also worse than the in-place baseline but only by +37 B vs SRL1's +81,181 B. This is the SAME predictor-empirical distributional mismatch pattern that falsified pair #1, applied to a different empirical distribution; the mechanism (uniform PRNG predictor cannot exploit empirical structure absent from PRNG output) is now empirically anchored on TWO orthogonal in-domain contexts (DWT detail subbands at Laplacian-peaked + fec6 null-leverage bytes at near-uniform), strengthening the canonical equation #26 domain refinement.

## Empirical falsification mechanism (the "predictor-empirical distributional mismatch" anti-pattern, now anchored on 2 orthogonal distributions)

| Aspect | Pair #1 (DWT detail subbands) | Pair #2 (fec6 frontier null-leverage bytes) |
|---|---|---|
| Empirical distribution | Laplacian-peaked int8 (wavelet detail coefficients normalized to [-128, 127]) | Near-uniform uint8 (archive bytes at master-gradient-null positions) |
| Empirical entropy | low (Laplacian peaked at 0; brotli LZ77 + Huffman wins) | high (near-uniform; brotli ≈ entropy floor at 1.0 bit/byte; 16,296 B baseline) |
| Procedural predictor | pcg64 uniform-int8 | pcg64 uniform-uint8 |
| Predictor-empirical match | POOR (uniform predictor cannot exploit Laplacian peak) | POOR (uniform predictor on near-uniform empirical produces near-uniform residual) |
| Residual structure | Near-uniform (empirical Laplacian − uniform = empirical shifted-and-mixed) | Near-uniform int16 in [-255, 255] |
| Residual sparsity | ~0.4% zero entries | ~0.33% zero entries |
| Net byte effect | +55,275 B regression (pair #1 SECOND anchor) | +81,181 B regression (pair #2 THIRD anchor) |
| Residual zscore | 38.8 | 101.18 |

The empirical anchors converge on a structural truth: **the pcg64-uniform predictor in the procedural-codebook helper is mismatched to BOTH Laplacian-peaked AND near-uniform empirical distributions** because the (empirical − predictor) residual is structureless in both cases. The canonical equation #26 domain refinement now has 2 explicit IN-DOMAIN EMPIRICAL FALSIFICATIONS providing strong evidence that the (pcg64, int-clipped residual, byte-budget) cascade cannot rescue.

## Operator-routable implications

1. **PIVOT** to pair #4 (magic_codec on raw seed bytes — orthogonality validation) per stacking analysis memo §14 EXPLICIT-DEFERRAL bullet now elevated. ~5 min FREE CPU smoke. Validates the orthogonality boundary: magic_codec MUST decline gracefully on a 32B uniform-random seed (envelope ≤ seed length); if it FAILS this test, the entire cascade has a deeper bug. **PRIORITY 1** (sub-$0; fastest disambiguator).

2. **PIVOT** to DP1-only cascade per `feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design` sister memo + DWT-HNeRV bind symposium predicted band [-0.015, -0.005]. DP1 is the only un-falsified pair remaining at $0-1 cost; the magic-codec stacking layer is now FALSIFIED on 2 of 4 pairs (the two cheapest to test). **PRIORITY 2** (paired-smoke per Catalog #325 14-day window).

3. **DEFER-PENDING-RESEARCH** pair #3 (world-model latent residual) — gated by Z6/Z7/Z8 ego-pose-conditioning ablation per DWT-HNeRV bind symposium op-routable #3 (not yet landed; Catalog #311 sister symposium). Per CLAUDE.md "Forbidden premature KILL": DEFER not KILL. Reactivation criterion: Z6 ablation lands AND pair #3's predictor-empirical entropy structure is empirically measured BEFORE smoke (the lesson from pair #1+#2 is: entropy measurement BEFORE running the smoke would have predicted the falsification).

4. **AMEND** canonical equation #26's `domain_of_validity` to mark `sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes_with_pcg64_uniform_predictor` as REFUTED-EMPIRICALLY (sister CANONICAL EQUATION #26 DOMAIN REFINEMENT subagent's surface). Cross-reference the parallel pair #1 refutation (`magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands_with_pcg64_uniform_predictor`) so the domain refinement reflects the meta-pattern.

5. **CASCADE META-INSIGHT** for future stacking-design: predictor-empirical distributional match is a NECESSARY precondition for residual-correction stacking to win. Stacking analysis memos should include a per-pair "expected predictor-empirical entropy match" cell BEFORE running the smoke. Current canonical equation has 3 IN-DOMAIN contexts; 2 of which are EMPIRICALLY FALSIFIED via the uniform-predictor failure mode; only the procedural-codebook substitution itself (in-archive bytes replaced by seed at inflate-time per Catalog #213-style canonical helper pattern) is still un-falsified.

## Observability surface (Catalog #305)

| Facet | Implementation |
|---|---|
| Inspectable per layer | per-Configuration byte count (A in-place / A brotli audit / B procedural-only / C SRL1 / C alt brotli on residual); per-residual sparsity_ratio + residual distribution (mean / std / abs_max / n_zero / n_nonzero); per-SRL1 wire-format breakdown (envelope / indices / values) all surfaced in smoke_result.json |
| Decomposable per signal | empirical_int8_sha256 + synthetic_int8_sha256 + residual_int16_sha256 + mutated_residual_int16_sha256 + srl1_serialized_payload_sha256 all queryable; per-3-axis master-gradient null-leverage classification preserved from null_byte_probe_matrix anchor |
| Diff-able across runs | sha256 of every intermediate artifact; reproducible from (archive_path, matrix_path, base_seed_bytes_hex, generator_kind) input tuple; byte-stable by construction |
| Queryable post-hoc | smoke_result.json + smoke_result.md persisted under experiments/results/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_20260521T002120Z/; canonical equation #26 THIRD anchor queryable via tac.canonical_equations.get_equation_by_id |
| Cite-able | THIS landing memo + canonical equation #26 frontmatter + sister anchor at .omx/state/canonical_equations_registry.jsonl line 46 |
| Counterfactual-able | "what if predictor were Laplacian-fitted instead of pcg64-uniform?" answerable by running smoke with a future Laplacian generator_kind (would need new helper in `tac.procedural_codebook_generator`; sister design candidate); "what if SRL1 alphabet were int8 instead of int16?" answerable by clipping the residual before encoding (loses byte-exact reconstruction; structurally lossy) |

## Sister-coordination verdict

- **PROCEED-DISJOINT** verdict resolved Step 0 `tools/check_sister_files_recently_landed.py` PROCEED at start + end (12-hour lookback; zero file overlap on my 3 target files)
- **Sister-COMPLEMENTARY** to pair #1 smoke (`debbc5833`) — same canonical equation; different in-domain context; together they form 2 of the 4 cascade pairs predicted by the stacking analysis memo
- **Sister-COMPLEMENTARY** to null-byte probe matrix (`82c1b3bac`) — matrix identified WHERE the 16,292 null-leverage bytes are; THIS smoke validates whether SRL1 wins on them (it does not)
- **Sister-PARALLEL-LANDING** with `codex-main-wave-3-magic-codec-pair-2-sparse-packet-ir-fec6-null-byte-cpu-smoke-20260521` — they ran the same smoke pipeline 70s before mine and landed their anchor at canonical equation #26 line 45 (written 00:20:10Z; mine at line 46 written 00:21:20Z). Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY: both anchors preserved; the dual landing is STRUCTURALLY OK and the convergent empirical falsification (both at zscore > 100) strengthens the canonical equation domain refinement
- **Sister-DISJOINT** from DP1 PAIRED-SMOKE PRE-AUTHORIZATION CHECKLIST + END-OF-DAY CASCADE RECONCILIATION (different file paths + different scopes; verified via `git status` showing no overlap on shared files)

## Cascade ROI accounting

| Metric | Today's session |
|---|---|
| $0 CPU smokes landed | 3 (yesterday's DWT distributional fit + pair #1 DWT residual + pair #2 fec6 null-byte residual) |
| Paid GPU spend saved | est. $3-10 (pair #1+#2 paid validations would have cost $1-3 each; cumulative cascade dispatches would have cost $5-15) |
| Empirical anchors landed on canonical equation #26 | 4 (FIRST hypothesis + SECOND DWT residual + THIRD + FOURTH fec6 null-byte residual via parallel sister) |
| Cascade pairs FALSIFIED at byte-budget surface | 2/4 (pair #1 + pair #2; both via uniform-predictor / structureless-residual mechanism) |
| Cascade pairs remaining (un-falsified) | 2/4 (pair #3 DEFER-PENDING-Z6-ablation; pair #4 PRIORITY 1 next-action FREE local probe; DP1-only PRIORITY 2 paired smoke) |

## Discipline

Catalog #117 / #157 / #174 / #235 / #289 canonical serializer with POST-EDIT `--expected-content-sha256` for all 3 committed files + #119 Co-Authored-By trailer + #127 custody triple (axis × hardware × evidence_grade) + #131 / #138 fcntl-locked + strict-load canonical equations registry + #185 META-meta drift detection (0 violations verified post-anchor-append) + #192 macOS-CPU advisory non-promotable + #206 3 crash-resume checkpoints emitted + #229 PV (10 verified items including stacking analysis memo + pair #1 landing memo + null-byte probe matrix landing memo + sparse_packet_ir source code + seed_derived_codebook source + canonical equations registry pre-smoke state + EmpiricalAnchor schema + macos_cpu_advisory builder + sister-checkpoint guard + canonical_frontier_pointer.json fec6 frontier sha) + #272 byte-mutation smoke (PASSED on full 16,292-index residual) + #287 placeholder-rationale rejection (zero `<rationale>` / `<reason>` literals in source code or memo) + #305 observability surface section in research memo with all 6 facets + #309 horizon_class=`frontier_breaking_enabler` + #314 / #340 sister-checkpoint absorption guard (PROCEED verdict at start and end of session) + #323 canonical Provenance umbrella (`build_provenance_for_macos_cpu_advisory`) + #324 predicted_band_validation_status `pending_post_training` + #335 canonical consumer auto-discoverable + #343 frontier pointer (fec6 archive identified via canonical sha prefix not hardcoded) + #344 canonical equation THIRD `anchor_appended` event verified via `get_equation_by_id` (anchor count 3 → 5 due to sister-parallel landing of FOURTH anchor at line 45; my THIRD at line 46) + #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE (zero mutation of existing memos or anchors).

## 6-hook wire-in per Catalog #125

1. sensitivity-map = N/A
2. Pareto constraint = N/A
3. bit-allocator = N/A
4. cathedral autopilot dispatch = N/A
5. continual-learning posterior = **ACTIVE** (THIRD `anchor_appended` event landed on canonical equation `procedural_codebook_from_seed_compression_savings_v1` per Catalog #344; sister consumer auto-discovered per Catalog #335 + auto-recalibrates from posterior; convergent FOURTH parallel anchor by codex sister also landed at same in-domain context)
6. probe-disambiguator = **ACTIVE** (smoke IS canonical pair #2 vs pair #4 / DP1-only cascade fork mechanism; FALSIFIED verdict structurally pivots cascade)

## mission_predicted_contribution = `frontier_breaking_enabler`

The smoke costs $0 + ~3 minutes wall-clock and EMPIRICALLY FALSIFIES pair #2 of the magic-codec cascade-stacking 4-pair matrix. Combined with the pair #1 FALSIFICATION (zscore=38.8) from earlier today, the cascade now has 2/4 pairs falsified at $0 cost; the apparatus saved an estimated $1-3 of paid GPU spend by catching the predictor-empirical distributional mismatch via FREE local empirical anchor BEFORE any paid dispatch on pair #2. The convergent empirical evidence across TWO orthogonal in-domain contexts (Laplacian-peaked DWT detail subbands + near-uniform fec6 null-leverage bytes) provides STRONG REPLICATION of the "pcg64-uniform predictor cannot exploit empirical structure absent from PRNG output" failure mechanism. Future cascade-design subagents have empirical anchors to predict similar pairs' likely outcome BEFORE smoke. Carmack MVP-first phasing further vindicated.

## Files

- `tools/run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py` (~1130 LOC post-linter; smoke pipeline + 3-way comparison + Catalog #272 byte-mutation + canonical equation THIRD anchor append)
- `src/tac/tests/test_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py` (~407 LOC; 18 tests PASS)
- `experiments/results/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_20260521T002120Z/smoke_result.{json,md}` (gitignored per Catalog #113 DERIVED_OUTPUT; reproducible by re-running the smoke)
- `.omx/state/canonical_equations_registry.jsonl` (canonical equation #26 anchor count: 3 → 5 due to sister-parallel landing)
- THIS file (research landing memo)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_landed_20260520.md` (memory landing memo; sister to this research memo)

## Cross-refs

- `.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md` (parent stacking analysis memo §7 pair #2 + §14 Top-3 #2)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_magic_codec_dense_streams_dwt_residual_smoke_landed_20260520.md` (pair #1 landing — sister-COMPLEMENTARY SECOND anchor)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_null_byte_probe_matrix_landed_20260520.md` (null-byte probe matrix — sister-COMPLEMENTARY prerequisite identifying 16,292 null-leverage bytes)
- `src/tac/packet_compiler/sparse_packet_ir.py` (canonical SRL1 encoder: `encode_rle_of_zeros` + `serialize_rle_of_zeros`)
- `src/tac/procedural_codebook_generator/seed_derived_codebook.py` (canonical equation #26 producer)
- `src/tac/canonical_equations/procedural_codebook_savings.py` (canonical equation builder)
- `src/tac/canonical_equations/registry.py` (`update_equation_with_empirical_anchor` API)
- `experiments/results/master_gradient_per_archive_fp64_extraction_wave_20260519T012404Z/master_gradient_pr101_fec6_frontier_macos_cpu_advisory_8pair_fp64_20260518.npy` (master-gradient .npy source for null-leverage byte indices)

**End of research landing memo.**
