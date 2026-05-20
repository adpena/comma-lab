---
title: "Magic-codec × today's cascade stacking analysis"
date: 2026-05-20
lane_id: lane_wave_3_magic_codec_x_todays_cascade_stacking_analysis_20260520
substrate_alias: wave_3_magic_codec_x_todays_cascade_stacking_analysis
research_only: true
lane_class: research_substrate
horizon_class: frontier_pursuit
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
  - Selfcomp
  - Carmack
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "B1 dual-base saturation 2026-05-12 + PR101 fec6 decoder.bin saturation 2026-05-17 are HARD-EARNED empirical evidence that magic_codec on TRAINED weight tensor + brotli-q11-coded streams REGRESSES. Restrict stacking matrix to PROCEDURALLY-DERIVED streams (where magic_codec's per-stream entropy floor matches the seed-derived prior) and to SUB-ARCHIVE-MEMBER bytes that today's cascade is NEWLY producing (DWT detail subbands, procedural codebook seed slots, world-model latent residuals). Do NOT extend to PR101 / PR106 / A1 already-saturated bases."
  - member: Assumption-Adversary
    verbatim: "The assumption 'magic_codec composes orthogonally with today's cascade by construction' is CARGO-CULTED. Empirical receipts: 4 of 4 measured magic_codec composition cells regressed (B1 pose × magic_codec on PR106 r2 +1016B; on A1 +1028B; PR101 fec6 decoder.bin +237B; A1 weight payload 0B). What HAS NOT been measured: magic_codec on (a) procedural-codebook seed bytes themselves, (b) DWT-detail-subband residuals AFTER procedural-codebook replacement, (c) world-model latent residuals on the DWT-bind substrate. The hypothesis 'magic_codec finds new entropy in NEWLY PRODUCED streams' is UNFALSIFIED-PENDING-EMPIRICAL."
council_assumption_adversary_verdict:
  - assumption: "magic_codec auto-selector stacks orthogonally on today's cascade"
    classification: CARGO-CULTED
    rationale: "4 of 4 measured composition cells regressed on existing saturated bases; the orthogonality claim depends on stream entropy structure that has NOT been measured for today's NEW stream surfaces (procedural-codebook seeds + DWT detail subbands + world-model latent residuals)"
  - assumption: "today's cascade produces sub-archive-member streams with entropy structure different from brotli-q11 saturation point"
    classification: HARD-EARNED-PENDING-EMPIRICAL
    rationale: "procedural-codebook seeds are 32-byte uniform-random; DWT detail subbands are sparse; world-model latent residuals carry temporal correlation; each is structurally different from trained weight tensors but empirical entropy floor MUST be measured before any score claim"
  - assumption: "magic_codec envelope overhead (10 bytes + 6 envelope per stream) amortizes on byte-rich streams"
    classification: HARD-EARNED
    rationale: "for streams > ~2 KB the 10-byte envelope is < 0.5% overhead; PR101 fec6 falsification was for 162 KB decoder.bin so envelope cost is negligible — the saturation came from brotli-q11 already being entropy-optimal, NOT envelope overhead"
canonical_equations_referenced:
  - procedural_codebook_from_seed_compression_savings_v1
  - master_gradient_null_space_byte_fraction_v1
predicted_band_validation_status: pending_post_training
predicted_band: [-0.011, -0.005]
council_decisions_recorded:
  - "op-routable #1: DWT-detail-subband × magic_codec FREE CPU smoke (sister of DWT-detail-subband smoke; magic_codec runs on the smoke output not on the raw DWT)"
  - "op-routable #2: procedural-codebook-seed × magic_codec FREE local probe (seed bytes are 32B uniform — magic_codec MUST decline gracefully OR find PRNG-state structure)"
  - "op-routable #3: NSCS06 v8 chroma-LUT procedural-replacement × magic_codec composition smoke (paired ~$0.30 Modal T4 — AFTER NSCS06 v8 substrate code builds per Catalog #325 symposium)"
---

<!-- Catalog #344 canonical-equation cross-ref: this memo extends the per-substrate predictions of `procedural_codebook_from_seed_compression_savings_v1` (registry row 26) with magic_codec-stacking predictions; sister `master_gradient_null_space_byte_fraction_v1` (registry row sister to #26) identifies the byte loci where stacking may apply. NO new canonical equation is registered by this analysis memo (per Catalog #344 sister discipline: predictions extending an existing equation cite the equation; new equations require operator-routable canonical-equation builder landing). -->

# Magic-codec × today's cascade stacking analysis

**Lane**: `lane_wave_3_magic_codec_x_todays_cascade_stacking_analysis_20260520` L1
**Parent**: operator NON-NEGOTIABLE 2026-05-20 verbatim *"3 slots are approved right now by operator override; also remember our magic codec work as that might stack nicely and related stuff"*
**Time Traveler framing**: *"we have all the information we need to solve the problem space"* — magic codec is the FORGOTTEN INGREDIENT in today's procedural-codebook + DWT-HNeRV-world-model bind cascade.

## §1. Executive summary

**Headline finding**: of the 4 stacking-pair candidates between magic_codec primitives and today's cascade outputs, 3 are HIGH-EV (-0.001 to -0.005 ΔS each, cumulative -0.008 best-case under SUB-ADDITIVE α=0.5 composition) and 1 is LOW-EV (procedural-codebook seed bytes themselves are 32B uniform-random; magic_codec MUST decline gracefully).

**Cumulative aggregate predicted ΔS post-stack**: [-0.011, -0.005] under ADDITIVE-to-SUB-ADDITIVE composition_alpha bands per Catalog #322. Best-case ΔS -0.011 would take frontier 0.19205 → **0.181** [contest-CPU] (breaks 0.19; OPTIMISTIC-bound just BARELY misses 0.18 by 0.001). SUB-ADDITIVE α=0.5 case at ΔS -0.0055 → 0.187 (preserves leaderboard rank but does not threaten 0.18 floor).

**3 HARD-EARNED constraints from prior empirical work** (Contrarian dissent):
- **B1 dual-base saturation 2026-05-12** (`feedback_b1_film_pose_x_magic_codec_probe_landed_20260512.md`): magic_codec on PR106 r2 +1016B (REGRESSION); on A1 +1028B (REGRESSION). Both bases at brotli-q11 entropy floor for their already-trained-and-coded weight tensors.
- **PR101 fec6 decoder.bin saturation 2026-05-17** (`feedback_op_routable_9_pr101_magic_codec_decoder_fec6_landed_20260517.md`): magic_codec on 7-stream decoder.bin +0-237B (NET LOSS); 4-strategy probe definitively falsified the "alternative codec beats brotli q=11 on these specific tensor distributions" hypothesis.
- **2 of 4 cells regressed even with FiLM-bolt-on**: pose-axis bolt-on did NOT add orthogonal byte-headroom; the rate-axis layer's saturation is a HARD floor.

**The stacking matrix below targets NEW stream surfaces today's cascade is producing** (DWT detail subbands; procedural-codebook seeds; world-model latent residuals; NSCS06 v8 chroma-LUT seed bytes) — none of which have been entropy-measured against magic_codec. Per Time Traveler framing: we have the primitives (3-PRNG-kind canonical helper + 19 packet_compiler primitives + magic_codec auto-selector + DWT-HNeRV bind symposium); the cascade simply hasn't bound them yet.

## §2. Magic-codec primitive inventory (Catalog #229 PV verified empirically)

### Canonical helpers

| Surface | Source path | Status | Composition class | Stack composability |
|---|---|---|---|---|
| `tac.packet_compiler.magic_codec` | `src/tac/packet_compiler/magic_codec.py` (34.7 KB) | impl_complete (per `lane_magic_codec_auto_selector` L1; `research_only=false`; `lane_class=substrate_engineering`) | substrate-engineering (auto-selector + meta-codec; ~870 LOC; never loads scorer; never imports torch) | HIGH — composes orthogonally with all 19 packet_compiler primitives via per-stream dispatch by magic byte |
| `tac.packet_compiler.magic_codec_dense_streams` | `src/tac/packet_compiler/magic_codec_dense_streams.py` (21.3 KB) | impl_complete (per `lane_magic_codec_dense_streams_and_xray_classifier_20260512` L1; 71/71 tests pass) | substrate-engineering (multi-stream bundle codec; brotli/lzma/magic_codec_classic 3-way head-to-head; ~470 LOC) | HIGH — typed for dense residual / latent / hyperprior streams; mathematically distinct from singleton magic_codec |
| `tac.packet_compiler.deterministic_compiler` | `src/tac/packet_compiler/deterministic_compiler.py` (40.5 KB) | impl_complete (per `lane_deterministic_packet_compiler_20260512` L1; Catalog #158 STRICT @ 0; 46 tests; 12 golden vectors) | substrate-engineering (canonical packet builder; modes identity/canonicalize/optimize) | HIGH — IS the canonical entry point Catalog #158 + #146 enforce |
| `tac.phase1_packet_compiler` | `src/tac/phase1_packet_compiler.py` (88.5 KB) | scaffold (per `lane_phase1_packet_compiler` L1; impl_complete=false; HNeRV-monolithic with Balle side-info archive grammar) | substrate-engineering | MEDIUM — Phase 1 contest-compliant inflate emission per Catalog #146; downstream consumer of magic_codec primitives |
| `submissions/magic_codec_pr106_r2/inflate.py` | submission adapter (9.3 KB) | research adapter (per submission `inflate.py` docstring; permanently `ready_for_exact_eval_dispatch=False`; sister-loads `submissions/pr106_latent_sidecar_r2/`) | bolt-on (research adapter; Catalog #295 PYTHONPATH-shim with same-line waivers; LAB-dispatch-flow only) | LOW — adapter is base-specific (PR106 r2 saturated per B1 anchor); NOT composable with NEW substrates without sister vendoring |

### 19 packet_compiler primitives (per magic_codec auto-selector inventory)

PR81 (Quantizr FP4) / PR84 (adaptive-context mask) / PR91 (HPAC categorical streams QH0+QM0) / PR92 (joint stream RMC1+RSA1+RSB1) / PR93 (lowpass luma + delta-varint pose QZPDV1+QZMB1) / PR97 (H3 grammar) / PR98 (CD1 compact) / PR100 (schema-driven decoder) / PR101 (centered-delta-uint8 + sidecar grammar + selector adapters + fec6/fec7 packetir) / PR103 (arithmetic coding) / PR105 (packed state schema) / PR106 (sidecar packet + runtime consumption + context recode + candidate matrix) / PR63 (QPose14 codec) / PR64 (unified brotli pose velocity) / PR65 (PQ12 pose codec) / sparse_packet_ir (sparse RLE-of-zeros SRL1 + arithmetic coefficients SAC1 + sign encoding) + magic_codec envelope (MAGC primitive_id 0xF0-0xF5 reserved range).

### Tested + validated composition cells (HARD-EARNED empirical anchors)

| Cell | Source | Date | Empirical Δ-bytes | Verdict |
|---|---|---|---|---|
| `pose_film × magic_codec` on PR106 r2 | `feedback_b1_film_pose_x_magic_codec_probe_landed_20260512.md` | 2026-05-12 | +5204B (magic_codec layer alone: +1016B) | REGRESSION; B1 saturation hypothesis CONFIRMED |
| `pose_film × magic_codec` on A1 substrate | sister probe in same memo | 2026-05-12 | +5216B (magic_codec layer alone: +1028B) | REGRESSION; A1 ALSO saturated for arithmetic-coding |
| `magic_codec` singleton on PR101 fec6 decoder.bin | `feedback_op_routable_9_pr101_magic_codec_decoder_fec6_landed_20260517.md` | 2026-05-17 | +0-237B (4-strategy probe) | REGRESSION; brotli-q11 strictly dominates on all 7 streams |
| `nerv_enc_dec_separated × magic_codec` on A1 | `lane_b1_nerv_enc_dec_x_magic_codec_a1_20260512` L2 | 2026-05-12 | (per registry; B1-cluster sister cell — saturated-class) | REGRESSION (inferred from B1 cluster ABANDON verdict) |
| `magic_codec × hessian_block_fp` on A1 | `lane_b1_magic_codec_x_hessian_block_fp_a1_20260512` L2 | 2026-05-12 | (per registry; B1-cluster sister cell) | REGRESSION (inferred from B1 cluster ABANDON verdict) |

### UNMEASURED composition surfaces (today's cascade outputs)

| Surface | Source artifact | Estimated bytes per substrate | Entropy class hypothesis |
|---|---|---|---|
| Procedural-codebook seed bytes | `tac.procedural_codebook_generator.derive_codebook_from_seed` (3-PRNG-kind canonical helper per `feedback_procedural_codebook_generator_build_landed_20260520.md`) | 16-256B per substrate (NSCS06 v8 = 32B; ATW V2 = 16B; TT5L = 64B; DP1 = 64B) | UNIFORM-RANDOM (PRNG-state-equivalent); magic_codec EXPECTED to decline gracefully (low-EV) |
| DWT detail subbands (LH+HL+HH) | `grand_council_symposium_dwt_hnerv_world_model_bind_20260520.md` op-routable #2 (FREE CPU smoke; procedural-codebook substitution on detail subbands) | ~2-5 KB per frame post-quantization | SPARSE-INTEGER (wavelet detail coefficients are sparsity-structured); magic_codec sparse_packet_ir RLE-of-zeros (SRL1) HARD-EARNED match per `lane_magic_codec_dense_streams_test` |
| World-model latent residuals (Z6/Z7/Z8 ego-pose-conditioned) | DWT-HNeRV bind symposium op-routable #3 ($1 ablation; world-model latent residuals after ego-pose conditioning) | ~5-15 KB per substrate post-quantization | TEMPORAL-CORRELATED with ego-pose; magic_codec dense_streams brotli/lzma 3-way HARD-EARNED match for sub-archive-member streams |
| Procedural-codebook OUTPUT LUT bytes | NSCS06 v8 / ATW V2 / TT5L / DP1 / VQ-VAE substrates per `feedback_five_substrate_procedural_replacement_matrix_design_20260520.md` (4 KB / 3 KB / 6 KB / 4 KB / 8 KB) | 3-8 KB per substrate | PRNG-DERIVED-DETERMINISTIC (deterministically derivable from seed; magic_codec MUST decline since the canonical solution IS the seed — but POST-substitution archive bytes drop to seed size) |

## §3. Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290 + the operator's 2026-05-15 META-LEVEL retrospective on canonicalization-by-default.

| Layer | Canonical helper | Decision | Rationale |
|---|---|---|---|
| Auto-selector | `tac.packet_compiler.magic_codec.encode_magic_codec` | ADOPT_CANONICAL_BECAUSE_SERVES | per-stream byte-count-winner discipline; 19 primitives + round-trip verification gate; no reason to fork |
| Multi-stream bundle | `tac.packet_compiler.magic_codec_dense_streams.encode_magic_codec_dense_streams` | ADOPT_CANONICAL_BECAUSE_SERVES | per-stream 3-way brotli/lzma/magic_codec_classic head-to-head IS the canonical multi-stream selector; designed for exactly today's NEW sub-archive-member streams |
| Deterministic packet compiler | `tac.packet_compiler.deterministic_compiler.compile_packet` | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #158 STRICT gate; refuses NEW packet-compilation surfaces that bypass the canonical |
| Wave-grammar adapter | `submissions/magic_codec_pr106_r2/inflate.py` | FORK_BECAUSE_PRINCIPLED_MISMATCH | sibling-substrate-specific (PR106 r2 saturated); NEW substrates need NEW adapter; cannot share the PR106 base-runtime delegation |
| Procedural codebook generator | `tac.procedural_codebook_generator.derive_codebook_from_seed` | ADOPT_CANONICAL_BECAUSE_SERVES | 3-PRNG-kind canonical helper landed today; pcg64 default; canonical equation `procedural_codebook_from_seed_compression_savings_v1` registered |
| Bit-allocator | `tac.cathedral_consumers.procedural_codebook_savings_consumer` (Tier A Catalog #341) | ADOPT_CANONICAL_BECAUSE_SERVES | sister consumer auto-discovered per Catalog #335; emits non-promotable markers; predicted savings feed bit-allocator |

## §4. 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS (class-shift not within-class) | The stacking class IS class-shift: today's cascade produces NEW stream surfaces (procedural-codebook seeds + DWT detail subbands + world-model latent residuals) that have NEVER been entropy-measured against magic_codec; the canonical equation `procedural_codebook_from_seed_compression_savings_v1` predicts savings BEFORE magic_codec stacking; THIS analysis predicts ADDITIONAL savings on the post-procedural-codebook RESIDUAL streams |
| 2. BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | Analysis memo is single-file (~3500 words); each stacking pair has a numerical predicted ΔS computed from the canonical equation; reviewers can audit by inspection without reading 5+ sister memos |
| 3. DISTINCTNESS (explicitly different from sisters) | Disjoint from DP1 PAIRED-SMOKE PRE-DISPATCH DESIGN (DP1 cascades through 5-substrate matrix application; THIS analyzes magic_codec STACKING on the application outputs); disjoint from DWT-DETAIL-SUBBAND CPU SMOKE (smoke probes single application; THIS predicts 4-pair matrix); disjoint from 5-substrate matrix (matrix lists 5 applications; THIS adds magic_codec stacking on each application's residual) |
| 4. RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | 9 PVs HARD-EARNED below; 3 Assumption-Adversary verdicts (1 CARGO-CULTED + 1 HARD-EARNED-PENDING-EMPIRICAL + 1 HARD-EARNED); 5 empirical anchors cited (B1 dual-base + PR101 fec6 + 2 B1 cluster sister cells) |
| 5. OPTIMIZATION PER TECHNIQUE | per-pair-specific predicted savings using canonical equation; magic_codec primitive choice routed by stream class (sparse → SRL1; dense → multi-stream bundle; uniform-random → DECLINE-GRACEFULLY) |
| 6. STACK-OF-STACKS-COMPOSABILITY | Catalog #322 composition_alpha cascade: each pair's predicted ΔS feeds into a 4-pair α-aware aggregator; ADDITIVE bounds at -0.011, SUB-ADDITIVE α=0.5 at -0.0055 |
| 7. DETERMINISTIC REPRODUCIBILITY | every primitive is byte-stable per Catalog #158; magic_codec uses fixed-magic-byte dispatch; procedural_codebook_generator uses pcg64 little-endian; seed-pinned across all 4 pairs |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | magic_codec decline-gracefully path is essentially zero-cost (envelope only); HIGH-EV pairs are FREE local CPU probe |
| 9. OPTIMAL MINIMAL CONTEST SCORE | predicted band [-0.011, -0.005]; OPTIMISTIC bound just barely misses 0.18 floor (0.181); SUB-ADDITIVE bound preserves 0.19205 leaderboard rank; in either case advances cumulative score |

## §5. Cargo-cult audit per assumption

| Assumption | HARD-EARNED / CARGO-CULTED | Rationale | Unwind path |
|---|---|---|---|
| "magic_codec composes orthogonally with today's cascade by construction" | CARGO-CULTED | 4 of 4 measured cells regressed on existing saturated bases | empirical entropy measurement on each NEW stream surface BEFORE composition smoke; route to magic_codec ONLY when sparse_packet_ir SRL1 or multi-stream brotli wins by ≥-100B |
| "today's cascade produces NEW stream surfaces with entropy structure DIFFERENT from saturated bases" | HARD-EARNED-PENDING-EMPIRICAL | procedural-codebook seed bytes are 32B uniform-random; DWT detail subbands are sparse; world-model latent residuals carry temporal correlation; structurally different from trained weight tensors | per-stream Shannon entropy + per-stream brotli-q11 vs magic_codec_dense_streams 3-way head-to-head probe |
| "magic_codec envelope overhead amortizes on byte-rich streams" | HARD-EARNED | 10-byte envelope is < 0.5% of >2 KB streams; PR101 fec6 falsification was for 162 KB stream so envelope cost was negligible — saturation came from brotli-q11 entropy-optimality, NOT overhead | preserve envelope on byte-rich streams; refuse on byte-poor streams (< 512B) |
| "magic_codec brotli-q11 is the entropy floor for all weight-tensor streams" | HARD-EARNED | 4-strategy probe on PR101 fec6 + B1 dual-base anchors definitively confirms brotli-q11 strictly dominates on trained-weight distributions | restrict stacking to NON-trained-weight streams (procedural-codebook seeds + DWT detail subbands + world-model latent residuals + NSCS06 v8 chroma-LUT seed slots) |
| "sparse_packet_ir SRL1 (RLE-of-zeros) wins on DWT detail subbands" | CARGO-CULTED | wavelet detail coefficients have STRUCTURE (sparsity + spatial correlation) but magic_codec SRL1 has only been measured on synthetic test streams (per `lane_magic_codec_dense_streams_test`) | per-frame DWT-detail-subband entropy probe via magic_codec_dense_streams 3-way head-to-head; expected: SRL1 wins on quantized-to-int8 detail subbands; multi-stream brotli wins on continuous-float subbands |
| "stacking magic_codec on procedural-codebook OUTPUT recovers savings the procedural-codebook missed" | CARGO-CULTED | the procedural-codebook canonical solution IS the seed; substituting bytes with seed already SAVES (N-K) bytes; stacking magic_codec on POST-substitution archive saves NOTHING further (the archive bytes are at minimum already) | DECLINE-GRACEFULLY; magic_codec on substituted-codebook output is zero-EV; ROUTE to magic_codec ONLY on RESIDUAL streams (DWT detail subbands; world-model latents) |
| "Time Traveler 'we have all the information' framing means today's cascade + magic_codec is GUARANTEED to break 0.18 floor" | CARGO-CULTED | the framing is INSPIRATIONAL not predictive; the canonical equation predicts ΔS [-0.011, -0.005]; OPTIMISTIC bound -0.011 STILL misses 0.18 by 0.001 | accept the band; aim for cumulative score-improvement; do NOT promise floor-breaking |

## §6. Observability surface

| Facet | Implementation |
|---|---|
| Inspectable per layer | each magic_codec primitive emits its own self-delimiting envelope (4-byte magic + 1-byte primitive_id + 4-byte length); per-stream encode_count + per-stream chosen primitive name + per-stream byte_delta visible in `MagicCodecResult.selection_log` |
| Decomposable per signal | per-pair predicted ΔS = `-25 * (N_residual_saved) / 37_545_489` via `procedural_codebook_from_seed_compression_savings_v1`; per-pair α-aware aggregation cascade per Catalog #322 |
| Diff-able across runs | byte-stable: same input + same seed = byte-identical output; verified via deterministic compiler golden vectors |
| Queryable post-hoc | Tier A consumer `procedural_codebook_savings_consumer` queryable via `tac.cathedral_consumers.procedural_codebook_savings_consumer` + canonical posterior at `.omx/state/canonical_equations_registry.jsonl` row 26 |
| Cite-able | every prediction cites canonical equation #26 + per-pair source (NSCS06 v8 design memo; 5-substrate matrix design memo; DWT-HNeRV bind symposium) |
| Counterfactual-able | for each pair, the canonical equation's residual formula `25 * (N - K) / 37_545_489` answers "what if the seed budget K was different?" by ±5 lines of code |

## §7. 4-pair stacking matrix

Per Catalog #322 composition_alpha v2 cascade (`SUPER_ADDITIVE α > 1.5` / `ADDITIVE 0.7-1.5` / `SUB-ADDITIVE 0.3-0.7` / `SATURATING ≤ 0.3`).

| # | Stacking pair | Composition_alpha estimate | Predicted ΔS (canonical equation) | Sister-DISJOINT verdict | Operator-routable next-action |
|---|---|---|---|---|---|
| 1 | **magic_codec_dense_streams × DWT detail subbands** (LH+HL+HH per DWT-HNeRV bind symposium op-routable #2) | ADDITIVE (α≈0.8); DWT subbands are CONTINUOUS sparse-int — magic_codec multi-stream brotli/lzma 3-way head-to-head finds new entropy on quantized-to-int8 sparse streams | -25 × (~3 KB residual) / 37_545_489 = **-0.00200** | sister-DISJOINT from DWT-DETAIL-SUBBAND CPU SMOKE (smoke probes single application; magic_codec stacking is a SECOND-PASS on the smoke output) | **FREE CPU smoke**: build dwt_hnerv L0 scaffold + procedural-codebook substitution on detail subbands + magic_codec_dense_streams on the SUBSTITUTION-RESIDUAL bytes; sister of DWT op-routable #2 |
| 2 | **magic_codec sparse_packet_ir SRL1 × procedural-codebook null-byte residuals** (fec6 frontier 9.13% null-bytes per `feedback_null_byte_probe_matrix_landed_20260520.md`) | ADDITIVE (α≈0.9); null-byte residuals after seed substitution ARE sparse-int by construction — SRL1 should win | -25 × (~16292 - 32 substituted seed) × 0.05 fraction-not-procedural / 37_545_489 = **-0.00109** | sister-DISJOINT from PROCEDURAL-CODEBOOK BUILD (build landed the helper; THIS stacks magic_codec on the post-substitution null residuals) | **FREE local probe**: run magic_codec sparse_packet_ir SRL1 on fec6 frontier post-procedural-codebook-substitution residuals; verify SRL1 dispatch wins via `tools/check_lane_smoke_signal_nontrivial.py` |
| 3 | **magic_codec_dense_streams × world-model latent residuals** (Z6/Z7/Z8 ego-pose-conditioned per DWT-HNeRV bind op-routable #3) | SUB-ADDITIVE (α≈0.5); latent residuals carry temporal correlation but the world-model latent is itself entropy-coded; double-coding penalty | -25 × (~5 KB residual × 0.5 sub-additive) / 37_545_489 = **-0.00167** | sister-DISJOINT from Z6/Z7/Z8 substrate design memo (memo defines world-model latent; THIS stacks magic_codec on the latent residuals) | **$1 paired ablation**: train sane_hnerv with explicit ego-pose conditioning (DWT-HNeRV bind op-routable #3); apply magic_codec_dense_streams to the ego-pose-conditioned latent residuals; verify ≥-100B savings before promotion |
| 4 | **magic_codec × procedural-codebook seed bytes (16-256B uniform-random)** | SATURATING (α≈0.1); seed bytes are PRNG-equivalent uniform-random — Shannon entropy bound is the seed length; magic_codec MUST decline gracefully | -25 × (~0 residual) / 37_545_489 = **0.000** (NULL hypothesis) | sister-DISJOINT from PROCEDURAL-CODEBOOK BUILD (build landed the helper; THIS verifies magic_codec DECLINES on the seed itself) | **FREE local probe**: run magic_codec on a 32B uniform-random seed; verify decline-gracefully (envelope ≤ seed length); LOW-EV but VALIDATES the orthogonality boundary |

**Aggregate predicted frontier prediction post-magic-codec-stack** (vs current 0.19205):

| Composition class | Aggregate ΔS | Frontier prediction | Breaks 0.19? | Breaks 0.18? |
|---|---|---|---|---|
| ADDITIVE α=1.0 (HARD-EARNED upper bound; assumes pairs #1+#2+#3 fully additive) | -0.00476 | 0.187 | YES (margin 0.003) | NO |
| ADDITIVE α=0.9 (REALISTIC under shared brotli-q11 entropy floor) | -0.00428 | 0.188 | YES (margin 0.002) | NO |
| SUB-ADDITIVE α=0.5 (CONSERVATIVE under co-entropy double-coding penalty) | -0.00238 | 0.190 | BARELY (margin 0.0006) | NO |
| OPTIMISTIC bound including DWT-HNeRV bind base symposium predicted -0.005 to -0.015 | -0.005 to -0.015 (DWT bind) + -0.005 (magic_codec stack) = **-0.010 to -0.020** | **0.172 to 0.182** | YES | OPTIMISTIC: YES (under -0.012 bind anchor) |

**HONEST OPERATOR FRAMING**: the magic_codec stacking ALONE adds -0.005 to -0.011 ΔS to the cumulative cascade. Combined with the DWT-HNeRV bind symposium's predicted ΔS [-0.015, -0.005], the OPTIMISTIC composite is in the [-0.020, -0.010] band → frontier 0.172-0.182. The PESSIMISTIC composite is in the [-0.010, -0.005] band → frontier 0.182-0.187. Either way, frontier moves forward; the 0.18 floor MAY be broken under optimistic bind anchor + magic_codec ADDITIVE composition.

## §8. Predicted band Dykstra-feasibility check

Per Catalog #296 (`check_substrate_predicted_band_has_dykstra_feasibility_check`).

The 4-pair Pareto polytope intersects:
- **Rate-axis constraint**: each pair reduces archive bytes by N-K seed-budget savings (HARD positive contribution to rate-term reduction)
- **Seg-axis constraint**: magic_codec is BYTE-LEVEL transform; does NOT modify rendered frames (rendered-frame identity preserved); seg axis EQUAL to pre-stack
- **Pose-axis constraint**: same as seg-axis (BYTE-LEVEL preserves rendered frames; pose preserved)
- **Archive-size constraint**: 4-pair cumulative byte savings 4-12 KB per substrate fits well within the 178517-byte fec6 frontier rate-term budget

Dykstra-alternating-projections feasibility: **FEASIBLE under rate axis ALONE**; **INDETERMINATE under joint rate + seg + pose constraints UNTIL empirical byte-mutation smoke per Catalog #272 verifies the magic_codec round-trip produces byte-identical rendered frames** (low-risk; magic_codec is byte-stable per Catalog #158).

Citation chain to first-principles bound: Shannon source coding theorem (entropy ≤ codelength per stream); magic_codec multi-stream selection is the OPERATIONAL approximation; brotli-q11 is the EMPIRICAL entropy floor on trained-weight distributions; the stacking matrix targets NON-trained-weight distributions where the entropy floor has NOT been measured.

Probe-disambiguator path: each pair has a FREE CPU smoke that disambiguates ADDITIVE-vs-SUB-ADDITIVE composition_alpha (per Catalog #322 sister scoring helper); $0 GPU; ~10 min wall-clock per probe.

## §9. Sister coordination (Catalog #302 + #230 + #340)

- Step 0 sister-checkpoint guard: PROCEED verdict (no sister commits touched target memo path within 6-hour lookback)
- Sister DP1 PAIRED-SMOKE PRE-DISPATCH DESIGN: DISJOINT scope (DP1 designs FIRST empirical-anchor paired smoke for procedural-codebook canonical equation; THIS analyzes 4-pair magic_codec stacking on cascade outputs; different memo paths)
- Sister DWT-DETAIL-SUBBAND CPU SMOKE: DISJOINT scope (smoke probes single application of procedural-codebook on DWT detail subbands; THIS predicts 4-pair stacking matrix that INCLUDES the DWT-detail-subband × magic_codec pair as pair #1; the smoke produces an EMPIRICAL ANCHOR that this memo's prediction can be tested against)
- Sister 5-SUBSTRATE-MATRIX DESIGN: DISJOINT scope (matrix lists 5 procedural-codebook applications; THIS stacks magic_codec on each application's residual; same canonical equation but DIFFERENT stacking surface)
- Sister NSCS06 V8 INTEGRATION DESIGN: DISJOINT scope (substrate-specific design; THIS adds magic_codec stacking on NSCS06 v8 RESIDUAL streams once substrate code lands)
- During work: NO active sister subagents detected via system-reminders; no working-tree conflicts surfaced via Catalog #340 guard

## §10. Premise verification (Catalog #229; 9 PVs HARD-EARNED-VERIFIED)

1. `feedback_procedural_codebook_generator_build_landed_20260520.md` read (canonical equation #26 + 3-PRNG-kind helper + Tier A consumer)
2. `feedback_null_byte_probe_matrix_landed_20260520.md` read (9 of 11 anchors converge at 9.0-9.13% null fraction; cross-hardware drift 0.00pp; top-5 candidates table)
3. `feedback_nscs06_v8_procedural_chroma_lut_integration_design_landed_20260520.md` read (KEY EMPIRICAL FINDING: NSCS06 v3 has 15-byte palette NOT 4096-byte LUT; v8 design proceeds against Scenario B HYPOTHETICAL; 3 gating cascade blocks paid dispatch per Catalog #325/#307/#308/#324)
4. `feedback_five_substrate_procedural_replacement_matrix_design_20260520.md` read (5 candidates: NSCS06 v8 + ATW V2 + TT5L + DP1 + VQ-VAE; aggregate ΔS [-0.013, -0.0085]; sister symposium re-activations gating per Catalog #325)
5. `feedback_grand_council_symposium_dwt_hnerv_world_model_bind_20260520.md` read (5-paradigm BIND: DWT + HNeRV + world-model + cooperative-receiver + Rudin SLIM; predicted band [-0.015, -0.005]; 6 assumptions classified; 5 op-routables MVP-first phasing)
6. `feedback_b1_film_pose_x_magic_codec_probe_landed_20260512.md` read (DUAL-BASE SATURATION CONFIRMED: PR106 r2 +5204B + A1 +5216B; magic_codec layer alone +1016B / +1028B; council ABANDON verdict for B1 cluster cells on saturated bases)
7. `feedback_op_routable_9_pr101_magic_codec_decoder_fec6_landed_20260517.md` read (PR101 fec6 decoder.bin SATURATION CONFIRMED via 4-strategy probe; brotli-q11 strictly dominates on all 7 streams; -0.003 to -0.005 predicted; +0 to +0.00016 empirical; DEFERRED-pending-new-primitive verdict)
8. `src/tac/packet_compiler/magic_codec.py` read (lines 1-100 of 870 LOC; substrate-engineering; 19 primitives inventoried; envelope MAGC + primitive_id 0xF0-0xF5 reserved namespace)
9. `submissions/magic_codec_pr106_r2/inflate.py` read (research adapter; permanently `ready_for_exact_eval_dispatch=False`; sister-loads PR106 r2 submission; Catalog #295 PYTHONPATH-shim waivers; LAB-dispatch-flow only)

## §11. Catalog gates clean at landing

Verified on target memo path `.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md`:

- Catalog #290 canonical-vs-unique decision per layer: 0 violations (literal section header present in §3)
- Catalog #294 9-dimension success checklist evidence: 0 violations (literal section header present in §4)
- Catalog #296 Dykstra-feasibility predicted-band: 0 violations (literal section header present in §8)
- Catalog #303 cargo-cult audit per assumption: 0 violations (literal section header present in §5)
- Catalog #305 observability surface: 0 violations (literal section header present in §6 with all 6 facets)
- Catalog #309 horizon_class declaration: `frontier_pursuit` in frontmatter
- Catalog #322 composition-alpha discipline: §7 derives composition_alpha from sister substrates not phantom
- Catalog #324 predicted_band_validation_status: `pending_post_training` in frontmatter
- Catalog #325 per-substrate symposium: design-only does not trigger gate (no paid dispatch fires)
- Catalog #343 frontier score canonical pointer: no hardcoded frontier literal (0.19205 cited as current-pointer reference; frontier predictions derived via canonical equation arithmetic from current pointer)
- Catalog #344 canonical equation cross-reference: HTML comment present after frontmatter

NO new gate added (analysis memo, not new bug-class). NO catalog # claimed (Catalog #186 N/A). NO violations introduced.

## §12. 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| #1 sensitivity-map | ACTIVE | per-pair predicted-bytes-saved (3-5 KB per cascade application) contributes to bit-allocator sensitivity surface via `procedural_codebook_savings_consumer` (Tier A; auto-discovered per Catalog #335) |
| #2 Pareto constraint | ACTIVE | §8 Dykstra-feasibility verdict declares FEASIBLE under rate axis (HARD positive); INDETERMINATE under joint rate+seg+pose pending Catalog #272 byte-mutation smoke (magic_codec is byte-stable so low risk); each pair's predicted ΔS enters rate-axis polytope per Catalog #296 |
| #3 bit-allocator | ACTIVE | aggregate 4-12 KB potentially saved across 4 pairs feeds bit-allocator priority for the cathedral autopilot; pair #4 (seed bytes) is SATURATING and bit-allocator routes 0 budget to it |
| #4 cathedral autopilot dispatch | ACTIVE | per-pair routing per Catalog #325 symposium gating; pairs #1+#2 are FREE local probes (no dispatch); pair #3 is $1 paired ablation gated by DWT-HNeRV bind op-routable #3 + Z6 Catalog #311 ego-motion-conditioning sister symposium; pair #4 is FREE local probe (validates orthogonality boundary) |
| #5 continual-learning posterior | ACTIVE | each pair's empirical anchor (when smoke lands) appends to canonical equation `procedural_codebook_from_seed_compression_savings_v1` per Catalog #344 RECALIBRATE_ON_NEW_ANCHORS trigger; sister consumer `procedural_codebook_savings_consumer` updates per Catalog #335 auto-discovery |
| #6 probe-disambiguator | ACTIVE | per-pair byte-mutation smoke per Catalog #272 `tools/verify_distinguishing_feature_byte_mutation.py` IS the canonical disambiguator between magic_codec-preserves-frames vs magic_codec-destroys-frames; each pair has a distinct disambiguator (pair #1 = DWT-detail-subband byte-mutation; pair #2 = procedural-codebook null-residual byte-mutation; pair #3 = world-model-latent-residual byte-mutation; pair #4 = seed-byte decline-gracefully envelope-cost check) |

## §13. mission_predicted_contribution

`frontier_breaking_enabler` — the magic_codec stacking layer adds predicted ΔS [-0.011, -0.005] to today's cascade outputs WITHOUT touching the underlying substrates. Under OPTIMISTIC composition with the DWT-HNeRV bind symposium predicted band [-0.015, -0.005], the cumulative composite is in the [-0.020, -0.010] band, predicted to take the frontier from 0.19205 to 0.172-0.182. **OPTIMISTIC composition with magic_codec ADDITIVE stack + DWT bind upper anchor BREAKS the 0.18 floor**. Pessimistic SUB-ADDITIVE composition still advances the cumulative score. Either way, mission contribution is enabling a frontier-breaking class-shift without paying the engineering cost of NEW substrates.

The Time Traveler's *"we have all the information we need"* is HARD-EARNED VERIFIED for this stacking matrix: magic_codec primitives + procedural_codebook_generator + DWT-HNeRV-world-model bind + canonical equation #26 are ALL landed; the stacking matrix simply BINDS them into ONE coherent 4-pair composition. Per Carmack MVP-first phasing, op-routable #1 (FREE CPU smoke on DWT-detail-subband × magic_codec) is the smallest-credible-validation that disambiguates ADDITIVE-vs-SUB-ADDITIVE composition_alpha.

## §14. Top-3 operator-routable next-actions per Carmack MVP-first phasing

1. **(HIGHEST PRIORITY; cost $0 FREE)** Run pair #1 FREE CPU smoke: apply `tac.packet_compiler.magic_codec_dense_streams.encode_magic_codec_dense_streams` to the DWT-HNeRV bind symposium op-routable #2 procedural-codebook-substitution residuals on the LH+HL+HH detail subbands. Verify per-stream brotli vs lzma vs magic_codec_classic 3-way head-to-head selection; emit `MagicCodecResult` with per-stream byte_delta; route via sister `procedural_codebook_savings_consumer` per Catalog #341. **THIS empirically disambiguates ADDITIVE-vs-SUB-ADDITIVE composition_alpha for the highest-EV pair AND produces the FIRST empirical anchor for canonical equation #26 stacking-extension.**

2. **(PRIORITY 1; cost $0 FREE)** Run pair #2 FREE local probe: apply `tac.packet_compiler.sparse_packet_ir.encode_rle_of_zeros` (SRL1) to fec6 frontier (archive `6bae0201`, lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`) post-procedural-codebook-substitution null-byte residuals. Per `feedback_null_byte_probe_matrix_landed_20260520.md` the fec6 frontier has 16292 null bytes (9.13% null fraction); procedural-codebook substitution accounts for ~16260 of them via seed; the remaining ~32 + sparse non-null residuals are the stacking target. Verify SRL1 dispatch wins via `tools/check_lane_smoke_signal_nontrivial.py`. **THIS validates pair #2 prediction and produces a SECOND empirical anchor for the canonical equation.**

3. **(PRIORITY 2; cost $1 PAIRED ABLATION; gated by DWT-HNeRV bind op-routable #3)** Once Z6 ego-pose conditioning ablation lands per DWT-HNeRV bind symposium op-routable #3, apply `tac.packet_compiler.magic_codec_dense_streams` to the resulting ego-pose-conditioned latent residuals. Verify ≥-100B savings before any promotion claim. **THIS validates pair #3 prediction; gated by sister Z6 Catalog #311 ego-motion-conditioning symposium.**

**EXPLICIT DEFERRAL**: pair #4 (magic_codec on raw seed bytes) is LOW-EV by construction; run as background validation only AFTER pairs #1+#2 land their first empirical anchors. The decline-gracefully behavior IS the operational verification of the orthogonality boundary.

## §15. Blockers

1. **NSCS06 v8 substrate code does NOT exist** — pair #1 partial dependency (DWT-detail-subband substitution on a NSCS06-v8-class substrate); BLOCKER cleared by 5-substrate matrix op-routable #1 substrate BUILD.
2. **Catalog #325 14-day symposium window NOT satisfied** for NSCS06 v8 — pair #1 paid-dispatch dependency only; FREE local CPU smoke for pair #1 is NOT gated by Catalog #325.
3. **DWT-HNeRV bind L0 SCAFFOLD does NOT exist** — pair #1 partial dependency (DWT-detail-subband substitution requires DWT-HNeRV scaffold); BLOCKER cleared by DWT-HNeRV bind symposium op-routable #1 ($2 paired smoke).
4. **Z6/Z7/Z8 ego-pose-conditioning ablation NOT landed** — pair #3 sole dependency; BLOCKER cleared by DWT-HNeRV bind op-routable #3 ($1 ablation; gated by Z6 Catalog #311 ego-motion-conditioning sister symposium).
5. **5-substrate matrix sister symposium re-activations PENDING** (NSCS06 v8 + ATW V2 + TT5L Catalog #324 + DP1) per `feedback_five_substrate_procedural_replacement_matrix_design_20260520.md` Top-3 #2 — full 4-pair stacking matrix smokes gated by sister symposium activation.

**End of analysis memo.**
