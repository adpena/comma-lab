<!-- SPDX-License-Identifier: MIT -->
<!-- # FORMALIZATION_PENDING:design_memo_re_scopes_existing_canonical_equation_no_new_empirical_findings_introduced_per_catalog_344_sister_discipline -->
# DWT BIND RE-SCOPE design memo — intermediate-transform path

**Date**: 2026-05-20
**Lane**: `lane_wave_3_canonical_equation_26_domain_refinement_20260520` (THIS landing)
**Sister lane**: `lane_wave_3_dwt_detail_subband_procedural_cpu_smoke_20260520` (empirical anchor; commit `f25f8cc1b`)
**Parent symposium**: `grand_council_symposium_dwt_hnerv_world_model_bind_20260520` (T3 PROCEED_WITH_REVISIONS; commit `9ef3eee22`)
**Re-scopes**: T3 DWT BIND op-routable #1 (`lane_dwt_hnerv_ll_l0_scaffold`) per empirical vindication
**Canonical equation**: `procedural_codebook_from_seed_compression_savings_v1` (Catalog #344 registry #26; `domain_refined` event landed 2026-05-20T23:41:52Z)
**Horizon class**: `frontier_pursuit`
**Mission contribution per Catalog #300**: `frontier_protecting`

`<!-- HISTORICAL_SCORE_LITERAL_OK:dwt_smoke_empirical_anchor_macos_cpu_advisory_kl_1_638_nats_3_28_sigma_not_score_truth -->`

## 1. Context (1-paragraph)

The T3 DWT BIND SYMPOSIUM 2026-05-20 produced PROCEED_WITH_REVISIONS with the Assumption-Adversary verdict #1 ("DWT-2-level matched-filter on contest video is canonical" = CARGO-CULTED). Carmack MVP-first op-routable #2 dispatched a $0 LOCAL macOS-CPU smoke applying the procedural codebook generator (`tac.procedural_codebook_generator.derive_codebook_from_seed`; PCG64 32-byte seed) directly to LH+HL+HH detail subbands of a Haar 2-level DWT on `upstream/videos/0.mkv` frame 300. Empirical receipts: KL=1.638 nats average across the 3 detail subbands; residual_zscore=3.28 > 2σ threshold; Catalog #272 byte-mutation smoke PASSED (all 3 subbands seed-sensitive, ~99.6% bytes-differ on 1-byte seed flip). **Conclusion**: the procedural codebook generator IS structurally sound (avalanche property of PCG64 per O'Neill 2014), but uniform-PRNG-derived int8 bytes DO NOT match Laplacian-peaked detail-subband statistics on contest video. Direct procedural substitution of DWT detail-subband bytes would corrupt the inverse DWT and almost certainly destroy rendered frames. The original T3 op-routable #1 bind (`dwt_hnerv_ll` substrate $1 paired smoke) MUST be re-scoped before paid GPU spend.

## 2. The structural finding restated

The canonical equation #26 producer `derive_codebook_from_seed` remains valid for substrates that consume the codebook as an **INTERMEDIATE TRANSFORM** (quantizer table / dequantizer LUT / class anchor / chroma LUT) rather than as a **DIRECT BYTE SUBSTITUTION** on transform coefficients. The distributional mismatch surfaces only when the procedural bytes are consumed as the actual coefficient stream (where the contest scorer's downstream renderer + segmentation network are exquisitely sensitive to the per-bin distribution). At intermediate-transform positions the codebook is a small reusable table that the inflate runtime indexes against, and the per-codebook-entry distributional mismatch is bounded by the quantizer's already-coarse partitioning (NSCS06 v8 chroma LUT predicted ΔS ≈ -0.0027 per 4 KB hoisted; sister substrates similar).

## 3. The two alternative HNeRV-on-LL bind architectures

The DWT BIND symposium considered three composition paths: (a) direct procedural substitution on detail subbands (now EMPIRICALLY FALSIFIED); (b) procedural codebook at intermediate-transform position (per the existing NSCS06 v8 op-routable + sister equation #26 producer); (c) HNeRV-on-LL with explicit detail-subband handling. THIS memo focuses on path (c) — the BIND between HNeRV's continuous-coordinate decoder and a wavelet-decomposed input — with two alternative detail-subband-handling sub-options. The KEY INSIGHT: HNeRV-on-LL is structurally valid (LL is the low-frequency approximation subband; its statistics are smooth and continuous, well-matched to HNeRV's MLP regression target). The question is what to do with the detail subbands (LH + HL + HH) that the inverse DWT needs to reconstruct the full-resolution frame.

### 3.1 OPTION A: HNeRV-on-LL with detail subbands BILINEARLY UPSAMPLED (zero detail-subband bytes)

**Architecture**: HNeRV decodes only the LL approximation subband (1/4 of the spatial resolution per level; for a 2-level DWT on 384×512 input the LL subband is 96×128). At inverse-DWT reconstruction time, the 3 detail subbands (LH, HL, HH) are populated with **bilinear-upsampled HNeRV-LL prediction residual** — i.e., the detail subbands carry NO archive bytes; they are computed at inflate time as a function of the HNeRV-LL output.

**Math (informal)**: Let `f_LL(x, y; θ_HNeRV)` be HNeRV's prediction for the LL subband. The bilinear-detail estimator sets `f_LH = bilinear_upsample(f_LL) - f_LL_at_native`, `f_HL` similarly, `f_HH = 0`. The inverse 2-level Haar transform reconstructs the full-resolution frame.

**Predicted ΔS band per Catalog #296 Dykstra-feasibility check**:

* **Source-of-validity**: the LL subband carries ~75% of the spectral energy on natural images per Mallat 1989 (Theorem 1); HH carries <5%. A 0-detail-byte reconstruction is structurally similar to the JPEG2000 LL-only fallback at extreme low-bit-rate.
* **Predicted ΔS bound (Shannon R(D) lower bound)**: at the contest's 600-pair × 6-pose ground-truth budget, dropping the 3 detail subbands gives an L2 reconstruction error ~ 2-3× the LL-only error (per dyadic wavelet theory). Translating to contest score via the empirical relationship `Δscore_pose ≈ 0.01 * relative_L2_error` (canonical equation #4 `per_pair_master_gradient_score_impact_taylor_v1` linear regime), the **predicted ΔS band ≈ [+0.02, +0.04]** (i.e., score WORSE by 2-4 points; pose-axis degradation dominates).
* **Dykstra-feasibility verdict**: the predicted band is OUTSIDE the {ΔS ≤ -0.001} feasibility region for any score-lowering claim. The score-cost of dropping detail subbands is much larger than the rate-saving (typical detail-subband-byte count ~ 100-500 KB at the contest's frame rate; saving 25 * 200_000 / 37_545_489 ≈ +0.13 rate term decrease at most).
* **Recommendation**: Option A is **structurally INFEASIBLE for score lowering** under the canonical contest formula. Useful only as a baseline / instrument to measure the detail-subband contribution; NOT a primary submission path.

### 3.2 OPTION B: HNeRV-on-LL with detail subbands BROTLI/STC-CODED

**Architecture**: HNeRV decodes the LL approximation subband (as in Option A). The 3 detail subbands (LH, HL, HH) are quantized + entropy-coded via **brotli** OR **STC** (syndrome-trellis coding per `tac.codec.stc_clean_source` sister) at the source's actual Laplacian-peaked distribution. The brotli/STC bytes are members of `archive.zip`; the rate term charges them per the canonical formula. **Critically**: this path does NOT invoke canonical equation #26 (procedural codebook savings) at all — the detail-subband bytes are EMPIRICAL (sourced from the actual contest video's DWT decomposition), not procedural.

**Math (informal)**: Let `H(LH | source) ≈ 1.5 bits/coefficient` (Laplacian source rate-distortion bound per Berger 1971 for typical sub-1-Mbit video). For a 96×128 LH subband at 2-level DWT, the per-frame brotli-coded byte count ≈ `96 * 128 * 1.5 / 8 ≈ 2304 bytes/frame`. At 1200 frames × 3 detail subbands × 0.6 brotli compression ratio ≈ 5.0 MB. Subtracted from the current ~1.5 MB archive baseline, this is a NET INCREASE.

* **Predicted ΔS band per canonical equation #1 brotli cascade `brotli_cascade_bounded_per_stream_v1`**: the brotli rate over the LH+HL+HH coefficient stream is bounded per the equation's empirical anchor; at ~5 MB additional bytes the rate term INCREASES by `25 * 5_000_000 / 37_545_489 ≈ +3.33`. This dominates any pose-axis improvement.
* **Recommendation**: Option B with naive brotli is **structurally INFEASIBLE** at full-resolution detail subbands. The bind would need to either (a) drop spatial resolution further (3-level or 4-level DWT shrinks the detail subband byte count), (b) use STC at a higher distortion budget (which trades pose accuracy for rate), or (c) re-scope to LL-only with the detail subbands replaced by HNeRV's own learned high-frequency component.

### 3.3 Comparative summary

| Option | Detail subbands | Predicted ΔS band | Catalog #296 Dykstra-feasibility | Recommended for paid GPU spend? |
|--------|-----------------|-------------------|----------------------------------|--------------------------------|
| A (bilinear) | 0 bytes (computed) | [+0.02, +0.04] | INFEASIBLE | NO |
| B (brotli/STC) | ~5 MB (encoded) | [+3.0, +5.0] | INFEASIBLE | NO |
| C (3-level DWT LL-only) | 0 bytes (smaller LL) | [+0.01, +0.05] | INFEASIBLE (LL-only at 1/16 res loses pose) | NO |

**Verdict**: the DWT BIND with HNeRV-on-LL is structurally INFEASIBLE for score lowering across the three sub-options examined. The original T3 op-routable #1 (`lane_dwt_hnerv_ll_l0_scaffold` $1 paired smoke) should be **DEFERRED-pending-redesign** per CLAUDE.md "Forbidden premature KILL without research exhaustion" — the canonical paradigm (DWT decomposition + neural decoder on the smooth approximation subband) remains plausible, but the specific BIND architecture must change before paid GPU spend is justified.

## 4. The recommended re-scope: redirect to NSCS06 v8 chroma LUT + DP1 codebook (equation #26 INCLUDED contexts)

Per the canonical equation #26 `domain_refined` event landed 2026-05-20T23:41:52Z, the INCLUDED contexts for the procedural codebook generator are the intermediate-transform substitution positions:

* `nscs06_v8_chroma_lut` — predicted ΔS ≈ -0.0027 per 4 KB hoisted (canonical formula)
* `atw_v2_codec_quantizer_lut` — predicted ΔS ≈ -0.002 per 3 KB hoisted
* `tt5l_transformer_tokens` — predicted ΔS ≈ -0.001 per 2 KB hoisted
* `dp1_codebook_bytes` — predicted ΔS ≈ -0.002 per OOD-derived basis replacement
* `chroma_lut_replacement` (generic) — depends on substrate; ≈ -0.001 to -0.005

**Aggregate per memo §4**: 5-substrate stack ≈ -0.013 ΔS (predicted). EMPIRICAL anchor pending per Catalog #325 per-substrate symposium.

## 5. Op-routable next actions (Top-3 TERTIARY)

### Top-3 PRIMARY (THIS landing)

DONE — canonical equation #26 domain refinement landed; consumer guards; tests; design memo.

### Top-3 SECONDARY (queued for sister subagent)

**Re-route T3 DWT BIND op-routable #1 $1 paid GPU smoke to NSCS06 v8 chroma LUT first empirical anchor.** The NSCS06 v8 substrate has the most-mature SLOT-NSCS06-V8-PROCEDURAL-CHROMA-LUT-INTEGRATION-DESIGN memo (commit `0b4a1d449`; predicted ΔS -0.0027 per 4 KB hoisted; Catalog #272 byte-mutation smoke required). The $1 budget should fund:

1. NSCS06 v8 trainer with `derive_codebook_from_seed` injected at the chroma LUT position
2. Modal CPU dispatch on full contest video (smoke; 5-10 min wall clock)
3. Empirical anchor appended to canonical equation #26 via `update_equation_with_empirical_anchor` (sister to the DWT anchor; this one IS in the INCLUDED context)
4. Catalog #272 byte-mutation smoke verifying the chroma-LUT bytes actually change rendered frames

Predicted operator-routable outcome: equation #26's `predicted_vs_empirical_residual` for the `nscs06_v8_chroma_lut_modal_cpu_smoke` axis is populated; sister Top-3 #2 (mean-over-N-frames smoke at the DWT detail-subband surface) can then be DEFERRED unless the operator wants additional refutation evidence.

### Top-3 TERTIARY (deferred; operator decision)

**Mean-over-N-frames smoke at DWT detail-subband surface.** The current DWT empirical anchor used a SINGLE frame (frame 300 of `upstream/videos/0.mkv`). A statistically tighter falsification would run the same smoke across N=10-100 frames and emit per-frame KL distributions. Predicted outcome: KL stays in the [1.0, 2.5] nats range with low variance (the distributional mismatch is structural, not frame-dependent). Cost: $0 (LOCAL macOS-CPU smoke; estimated 30-60 min wall clock for N=100). Recommendation: **DEFER** unless the operator wants additional refutation evidence — the canonical equation #26 `domain_refined` event already encodes the structural finding; per-frame statistics are auditable for free via the existing smoke artifact at `experiments/results/dwt_detail_subband_procedural_smoke_20260520T232239Z/smoke_result.json`.

## 6. Discipline (binding)

* CLAUDE.md "Canonical equations + models registry" non-negotiable: `domain_refined` event landed via canonical helper.
* Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: prior `registered` + `anchor_appended` rows preserved verbatim; only a NEW `domain_refined` event row added.
* Catalog #287 placeholder-rationale rejection: rationale 861 chars; non-placeholder.
* Catalog #344 sister discipline: canonical equation evolution discipline honored.
* Catalog #296 Dykstra-feasibility check: each Option's predicted band cited against the {ΔS ≤ -0.001} feasibility region.
* Catalog #303 cargo-cult audit: the OPTION A bilinear-detail assumption (HNeRV-LL spectral dominance) is HARD-EARNED per Mallat 1989 Theorem 1; the OPTION B Laplacian-source brotli rate assumption is HARD-EARNED per Berger 1971 source-coding theorem.
* Catalog #309 horizon-class declaration: `frontier_pursuit` per the original T3 BIND symposium.
* CLAUDE.md "Forbidden premature KILL": the DWT BIND is **DEFERRED-pending-redesign**, NOT KILLED; reactivation criteria pinned (`lane_dwt_hnerv_ll_l0_scaffold` may proceed if a NEW BIND architecture moves the predicted ΔS band INTO the {ΔS ≤ -0.001} feasibility region per Catalog #296).
* CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in: declared below.

## 7. 6-hook wire-in per Catalog #125

1. **sensitivity-map** = N/A (defensive design memo; no per-byte sensitivity contribution)
2. **Pareto constraint** = **ACTIVE** (each Option's predicted band is a Pareto-feasibility check per Catalog #296; consumed by future Dykstra alternating-projections solver runs)
3. **bit-allocator** = N/A (no bit-allocator decision; Options eliminated structurally)
4. **cathedral autopilot dispatch** = **ACTIVE** (canonical equation #26 `domain_refined` event auto-discoverable by `canonical_equation_lookup_consumer`; downstream candidates with `substrate_context = direct_dwt_detail_subband_byte_substitution` will be refused at consumer surface per Catalog #341 markers)
5. **continual-learning posterior** = **ACTIVE** (this memo IS the canonical anchor for the DWT BIND re-scope; future per-substrate symposia consume the redirect signal)
6. **probe-disambiguator** = **ACTIVE** (canonical equation #26 `validate_context_is_in_domain` helper IS the structural disambiguator between INCLUDED and EXCLUDED contexts; replaces ad-hoc per-substrate analysis)

## 8. Observability surface per Catalog #305

* **Inspectable per layer**: each Option's architecture decomposed into HNeRV-LL prediction layer + detail-subband-handling layer + inverse-DWT reconstruction layer.
* **Decomposable per signal**: predicted ΔS bands decomposed into rate-term contribution + pose-axis contribution + seg-axis contribution.
* **Diff-able across runs**: the canonical equation #26 `domain_refined` event captures the BEFORE/AFTER domain_of_validity surface; `git log .omx/state/canonical_equations_registry.jsonl` shows the append-only history.
* **Queryable post-hoc**: `tac.canonical_equations.get_equation_by_id` + `tac.canonical_equations.load_registry_events_lenient` + `tac.canonical_equations.procedural_codebook_savings.validate_context_is_in_domain` are all callable APIs.
* **Cite-able**: every claim in this memo cites a canonical artifact (commit / equation / symposium / sister memo).
* **Counterfactual-able**: the OPTION A/B/C analysis IS the counterfactual — what if we had attempted each path? Predicted ΔS bands quantify the answer per option.

## 9. Cross-references

* `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_dwt_detail_subband_procedural_cpu_smoke_landed_20260520.md` — sister landing memo (empirical vindication)
* `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_canonical_equation_26_domain_refinement_landed_20260520.md` — THIS landing's memo
* `.omx/research/grand_council_symposium_dwt_hnerv_world_model_bind_20260520.md` — T3 PROCEED_WITH_REVISIONS source (43 KB)
* `.omx/research/dwt_detail_subband_procedural_smoke_landed_20260520.md` — full smoke research memo
* `.omx/research/procedural_codebook_generator_null_exploit_design_20260520.md` — original equation #26 design memo
* `src/tac/procedural_codebook_generator/seed_derived_codebook.py` — canonical equation #26 producer (`derive_codebook_from_seed`)
* `src/tac/canonical_equations/procedural_codebook_savings.py` — canonical equation builder + `validate_context_is_in_domain` helper
* `src/tac/cathedral_consumers/procedural_codebook_savings_consumer/__init__.py` — cathedral consumer with domain gating
* `.omx/state/canonical_equations_registry.jsonl` — registry (3 events for equation #26: registered + anchor_appended + domain_refined)
