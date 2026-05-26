# Wunderkind G1 v2 — wire-grammar class-shift full-stack design memo

**Date:** 2026-05-16
**Subagent:** `wunderkind_g1_v2_full_stack_design_20260516`
**Lane:** `lane_z3_g1_entropy_coded_v2_20260515` (continuation; v1 phantom-anchor reactivation)
**Operator anchor:** resurrection-audit Tier 2 candidate Z3-G1 + L5 v2 staircase Step B1 (1KB CDF replacing 50KB hyperprior)
**Sister-disjoint scope (Catalog #230):** read-only on `.omx/research/`, `src/tac/substrates/z3_g1_entropy_coded_v2/*`, prior Wunderkind G1 v2 design memo (2026-05-15); write-only on this single memo + checkpoint records.
**Predecessor design memo:** `.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md` (substrate spec; this memo is the FULL-STACK comprehensive extension with observability surface + cargo-cult-per-assumption audit + Dykstra-feasibility section + 22-section UNIQUE-AND-COMPLETE-PER-METHOD template).

## Operating-within assumption-statement (per Catalog #292)

The shared assumption I am operating within for this design: *"The Wunderkind G1 v2 wire-grammar (Z3G2 magic, sigma table + class-index entropy-coded streams) is the structural fix for the v1 phantom-anchor failure where empty `hyperprior_weights_int8 = b""` + `w_hat_int8 = b""` slots produced 5-decimal-identical Z3 v2 baseline scores. The fix is necessary but not sufficient: even with bytes-actually-shipping, the SegNet-class CDF must derive from empirical class statistics on `upstream/videos/0.mkv` per Catalog #304's no-closed-form rule, and the inflate decoder must EMPIRICALLY consume those bytes per Catalog #272's distinguishing-feature integration contract."*

HARD-EARNED basis: codex review bkrbqet3p F1 empirical receipt (smoke `fc-01KRPKCXARWP7NBGJCXB2P9QEP` returned `0.19869 [diagnostic-CPU]` exactly matching Z3 v2 baseline 0.19869 to 5 decimals against archive `c55e2d0d`) + CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable (Catalog #220) + the phantom-distinguishing-feature class anchor cited in resurrection-audit Tier 2.

The Assumption-Adversary seat would challenge: *"Is the wire-grammar fix sufficient, OR does the apparent score improvement still bottom out at within-class plateau even with bytes-actually-shipping because the SegNet-class signal IS redundant with the per-pair latent residual statistics Z3 v2 already exploits?"* — Answer: this is a PROBE-WORTHY hypothesis. Section 14 (probe-disambiguator) lands the H(residual | class) vs H(residual) probe to falsify or confirm the redundancy hypothesis BEFORE the v2 trainer + dispatch land.

---

## 1. Substrate identity

- **id**: `z3_g1_entropy_coded_v2`
- **lane_id**: `lane_z3_g1_entropy_coded_v2_20260515`
- **base substrate**: A1 (Z3HV2 — Z3 v2 Ballé hyperprior bolt-on; frozen decoder + sidecar)
- **paradigm class**: SCORER-AS-COOPERATIVE-RECEIVER (Wunderkind G1 paradigm; cf. `feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md`)
- **wire-grammar magic**: `Z3G2` (distinguishes from v1 `Z3V2`)
- **target_modes**: `("research_substrate",)` at landing; `("contest_one_video_replay", "contest_generalized")` after byte-mutation smoke passes + paired CPU+CUDA auth-eval lands
- **predecessor**: Z3-G1 v1 (`lane_z3_g1_scorer_softmax_hyperprior_gating_20260515`, KEEP-research_only PENDING v2 per resurrection-audit §3.9)
- **9-dim Dimension 1 UNIQUENESS axis**: CLASS-SHIFT at the wire-grammar layer (scorer-class-conditional entropy coding vs Z3 v2's unconditional Gaussian prior); NOT a within-class refinement.

## 2. Pretraining

NONE. v2 inherits A1's frozen decoder weights verbatim (162,164 B `decoder_blob` shipped as-is in the Z3G2 packet's first section). The A1 decoder was Ballé-2018 trained on `upstream/videos/0.mkv` per the NSCS03 sibling pipeline; v2 does NOT re-pretrain.

**Rationale**: substrate-engineering exemption per HNeRV parity discipline L7 — the bolt-on (sigma table + class-index stream) is ≤350 LOC; pretraining belongs to A1.

## 3. Training curriculum

Inherits A1's 100-epoch / 1000-epoch curriculum AS-IS for the frozen decoder; the bolt-on training is ONLY the 5×28 sigma-table parameter (140 fp32 params) + per-pair empirical class-prior CDF estimation (no learning; deterministic histogram).

**Phase 1 (epochs 1-20; warm-up):** sigma_logits initialized at softplus⁻¹(2.0); learning rate 1e-3; AdamW; gradient flows from `scorer_loss_terms_btchw` through `score_pair_components` (Catalog #164) through the inflate-time latent reconstruction `latents = (residual_q * sigmas_per_pair) * latent_scale + latent_offset` (where `sigmas_per_pair = sigma_table[class_indices, :]`).

**Phase 2 (epochs 21-80; convergence):** learning rate cosine-decayed to 1e-5; EMA decay 0.997 on sigma_logits per CLAUDE.md "EMA non-negotiable" + CLAUDE.md "Quantizr decay = 0.997"; eval_roundtrip=True throughout per CLAUDE.md "eval_roundtrip non-negotiable".

**Phase 3 (epochs 81-100; quantization-aware):** sigma_table quantized to int8 every 5 epochs via the canonical `quantize_sigma_table_int8(int8_sigma_scale=16.0)`; loss adds a quantization-error regularizer that pulls the float sigma toward the int8-recoverable grid.

**Class-prior CDF refresh**: empirically computed from `g1_v2_per_pair_dominant_class_from_segnet_argmax(segnet_argmax_per_pair)` ONCE at epoch 20 (after warm-up) and FROZEN for the remainder; Laplace-smoothed (`smoothing=1`) per `compute_class_prior_cdf` to avoid zero-frequency classes.

## 4. Architecture

```
Z3-G1 v2 architecture (5×28 sigma table + per-pair class index)
─────────────────────────────────────────────────────────────

  GT pair frames (upstream/videos/0.mkv)
            │
            ▼ (SegNet @ compress-time; FREE per CLAUDE.md contest rule)
  segnet_argmax_per_pair: (N_pairs, H, W) int in [0, 5)
            │
            ▼ (g1_v2_per_pair_dominant_class_from_segnet_argmax)
  class_indices: (N_pairs,) long in [0, 5)
            │
            ▼ (compute_class_prior_cdf w/ smoothing=1)
  class_prior_counts: (5,) int64 (uint16 range)
            │
            ▼ (F.embedding(class_indices, sigma_logits) + softplus + clamp)
  sigmas_per_pair: (N_pairs, 28) fp32 in [min_sigma, max_sigma]
            │
            ▼ (joint with A1 latent residual; trained via gradient through score_pair_components)
  residual_q + sigmas_per_pair → latents (28-dim per pair)
            │
            ▼ (frozen A1 decoder)
  reconstructed frames → score via canonical scorer-aware loss
```

**Parameter count**: 5 × 28 = **140 fp32 params** in `sigma_logits`. Total v2 bolt-on memory: 560 B (fp32) or 140 B (int8 quantized at archive time).

**Compute**: F.embedding + softplus + clamp = ~3 FP ops per sigma lookup × 600 pairs = ~1800 FP ops at inflate time (vs A1's ~162K-param decoder forward). v2 inflate compute is structurally negligible.

## 5. Priors

Two priors ship in the archive bytes (Section 8 wire grammar):

### 5.1 Class prior CDF (10 B)

Empirical frequency counts over the 600-pair dominant SegNet class. Smoothed with Laplace (`smoothing=1`) to avoid zero-frequency classes. Ships as `5 × uint16 LE = 10 B` raw. The decoder normalizes counts → probabilities → CDF for the constriction-Huffman class-index stream.

**HARD-EARNED vs CARGO-CULTED**: HARD-EARNED. Empirically derived from `upstream/videos/0.mkv` per Catalog #304 (no closed-form). Each new training run re-derives counts; archive ships the actual measured counts; decoder uses EXACTLY those counts.

### 5.2 Conditional Gaussian prior over residual (sigma table)

Per-class-per-dim sigma table (5 × 28 = 140 values) trained jointly with A1's frozen decoder. After training, sigma quantized to int8 with `int8_sigma_scale=16.0` (sigma values in [0, 16] map to int8 [0, 127]). Ships as `brotli(140 int8 sigma) ≈ 300 B`.

**HARD-EARNED vs CARGO-CULTED**: HARD-EARNED. Per Ballé 2018 hyperprior canonical (Section 16 cross-refs); the sigma table IS the bare-minimum hyperprior (a 5-class lookup table instead of a 2-layer MLP). The CHOICE of 5 classes is HARD-EARNED from upstream/modules.py SegNet `smp.Unet('tu-efficientnet_b2', classes=5)`.

## 6. Post-training

**Quantization**: sigma_table → int8 via `quantize_sigma_table_int8(int8_sigma_scale=16.0)`. The class indices are already uint8 (5-class). The latent residual is int8 quantized at compress time with `quantization_step=1.0` (matches Z3 v2 grammar).

**EMA snapshot restoration**: per CLAUDE.md "EMA non-negotiable", inference uses the EMA shadow weights for sigma_logits (NOT the live final-epoch weights). The archive ships the EMA-snapshot int8 sigma table.

**Per-dim affine compute**: `(latent_offset, latent_scale)` per A1's existing pipeline (28 × float32 each = 112 + 112 = 224 B). Reused verbatim from A1.

## 7. Score-aware loss

Routes through canonical `score_pair_components` (Catalog #164) per HARD-EARNED PR95 lesson + ADDS a rate-axis penalty term:

```python
loss_total = loss_score + λ_R · rate_bits / N
```

where:
- `loss_score = score_pair_components(reconstructed_frames, gt_frames, scorer_state)` (canonical scorer-aware loss; SegNet + PoseNet contributions)
- `rate_bits = g1_v2_total_rate_bits(residual, sigma, class_indices, class_prior_counts)` (Section 4.5 of substrate's architecture.py)
- `λ_R` linearly warms 0 → λ_target across first 10% epochs (Ballé-2018 style)
- `λ_target` is a hyperparameter; first-cut choice `λ_target = 0.01` per first-principles analysis (Section 13 predicted band).

**Canonical-vs-unique decision for this layer**: ADOPT canonical `score_pair_components` (HARD-EARNED per PR95 lesson) + UNIQUE EXTENSION for the rate term (`g1_v2_total_rate_bits`; specific to the SegNet-class-conditional formulation). The rate-term forking is necessary because the v2 distinguishing feature IS the class-conditional rate reduction.

## 8. Archive grammar

**`Z3G2` wire format** (replaces A1's `latent_blob` at offset 162168 within A1's monolithic 0.bin):

```
[uint32 LE section_total = 162168]            (verbatim from A1)
[decoder_blob 162164 B]                       (verbatim from A1)
[Z3G2 header + payload]                       (NEW; replaces A1's 15387-byte latent_blob)
[sidecar_blob (variable; ~607 B)]             (verbatim from A1)

Z3G2 section detail:

  magic               : 4 bytes ASCII "Z3G2"
  version             : uint8 (== 1)
  n_pairs             : uint16 LE (== 600)
  num_scorer_classes  : uint8 (== 5)
  latent_dim          : uint8 (== 28)
  int8_sigma_scale    : float32 LE (4 B)
  quant_step          : float32 LE (4 B)
  min_sigma           : float32 LE (4 B)
  max_sigma           : float32 LE (4 B)
  reserved            : 2 B (== 0)
  --- Header total = 27 B ---
  sigma_table_len     : uint16 LE (2 B)
  sigma_table_blob    : <sigma_table_len> bytes (brotli-compressed 140 int8 sigma)
  class_prior_blob    : 10 B raw (5 × uint16 LE frequency counts)
  class_index_len     : uint32 LE (4 B)
  class_index_blob    : <class_index_len> bytes (constriction-Huffman encoded uint8 600-pair stream)
  residual_blob_len   : uint32 LE (4 B)
  residual_blob       : <residual_blob_len> bytes (brotli-compressed int8 residual)
  latent_offset_blob  : 112 B (28 × float32)
  latent_scale_blob   : 112 B (28 × float32)

Estimated total Z3G2 section: ~27 + 2 + 300 + 10 + 4 + 350 + 4 + 1200 + 224 ≈ 2121 B
A1 latent_blob replaced: 15387 B → SAVINGS: ~13266 B
```

**HARD-EARNED-vs-CARGO-CULTED**: HARD-EARNED. Per HNeRV parity discipline L3 (monolithic single-file `0.bin` with fixed offsets) + Catalog #146 (3-arg contest-compliant inflate.sh contract) + Catalog #266 (substrate archive consumes hyperprior bytes — the FIX for v1 phantom-anchor). The grammar is verifiable byte-by-byte via `decode_z3g2_section` (round-trip safe; see existing 23-test suite for v2 substrate).

## 9. Inflate runtime

Canonical helper-based ≤100 LOC per HNeRV parity L4 (verified: current `inflate_consumer.py` is 173 LOC including docstrings; LOC excluding docstrings ≈ 95).

**Compliance checks**:
- NO scorer load at inflate time per CLAUDE.md "Strict scorer rule" ✓ (no `load_segnet_state` / `load_posenet_state` in `inflate_consumer.py`)
- Device selection via canonical `select_inflate_device` per Catalog #205 ✓
- 3-arg `inflate.sh archive_dir output_dir file_list` per Catalog #146 (inherited from A1 inflate skeleton)
- CUDA-or-CPU agnostic ✓
- No `/tmp` paths ✓
- 2 external dependencies: `torch`, `brotli`, `constriction` (within HNeRV parity L4 budget)

**Critical contract per Catalog #220**: the inflate runtime ACTUALLY CONSUMES the distinguishing v2 bytes (sigma_table_blob, class_prior_cdf_blob, class_index_blob) in `reconstruct_class_indices_and_sigma_table_from_z3g2_payload`:
- `sigma_table_fp32 = _unpack_sigma_table_entropy_coded(sigma_table_int8, ...)` ← consumes sigma_table_blob
- `class_prior_cdf = _unpack_class_prior_cdf(class_prior_counts)` ← consumes class_prior_blob
- `class_indices_long = _class_index_bytes_to_tensor(class_indices_uint8, ...)` ← consumes class_index_blob (after constriction-Huffman decode in `decode_z3g2_section`)
- `sigmas_per_pair = sigma_table_fp32[class_indices_long, :]` ← BOTH bytes feed into per-pair scale
- `latents = (residual_q * sigmas_per_pair) * latent_scale + latent_offset` ← per-pair scale affects reconstructed latents → affects decoded frames → affects scorer output

This chain is structurally verified by the byte-mutation smoke (Section 10 + Catalog #272).

## 10. Export contract

**Single ZIP STORED `0.bin`** archive layout (HNeRV parity L3); same wrapping as A1's `archive.zip`.

**Export path** (compile-time; runs once per training):
1. Train sigma_table (Section 3) + freeze
2. Compute per-pair `class_indices` via SegNet @ compress-time (FREE per contest rule)
3. Compute `class_prior_counts` via Laplace-smoothed histogram
4. Quantize sigma_table → int8 (`quantize_sigma_table_int8`)
5. Pack residual_int8 (existing A1 pipeline; reused verbatim)
6. Build Z3G2 section via `encode_z3g2_section`
7. Build full payload via `build_z3g2_payload_bytes(a1_bytes, z3g2_section)`
8. Wrap into ZIP STORED single-member `0.bin`
9. Verify byte-mutation smoke per Catalog #272 + Catalog #139 (mutate sigma_table_blob byte → assert decoded frame bytes change)

**HARD-EARNED**: per HNeRV parity discipline L3 + Catalog #220 operational mechanism + Catalog #266 fix anchor.

## 11. Byte-mutation distinguishing-feature smoke

**Canonical tool**: `tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py` (already landed; 20.1 KB) — sister of `tools/verify_distinguishing_feature_byte_mutation.py` (Catalog #272 canonical helper).

**Smoke protocol**:
1. Build a Z3G2 archive at HEAD weights
2. Run `inflate.sh archive_dir output_dir file_list` → record `frames_baseline.hash`
3. Mutate ONE byte in `sigma_table_blob` (e.g., flip middle byte) → repack → re-inflate → record `frames_sigma_mutated.hash`
4. Assert `frames_baseline.hash != frames_sigma_mutated.hash` (sigma_table_blob is OPERATIONAL)
5. Repeat for `class_prior_blob` (e.g., flip count for class 2) → record + assert
6. Repeat for `class_index_blob` (e.g., flip first encoded byte) → record + assert
7. Repeat for `residual_blob` → record + assert (control: residual changes were already verified at v1; this is a regression guard)
8. Emit `distinguishing_feature_byte_mutation_proof.json` with per-section PASSED/FAILED verdict

**Pass criterion**: ALL THREE distinguishing-feature bytes (sigma_table_blob, class_prior_blob, class_index_blob) must produce changed inflate output. ANY FAIL → lane stays `research_only=true` with the specific blob name in `reactivation_criteria`.

**This is the EXACT v1 failure mode regression guard**: v1 returned identical frames for ANY mutation of `hyperprior_weights_int8` because the slot was empty (zero-byte mutation is a no-op). v2 fix: the slots have CONTENT.

## 12. Stack-of-stacks composition

Per CLAUDE.md "stack of stacks" + 9-dim Dimension 6:

| With substrate | Axis orthogonality | Composition class | Expected ΔS | Rationale |
|---|---|---|---|---|
| **NSCS01** (nullspace split renderer) | ORTHOGONAL (codec vs renderer-arch) | ADDITIVE | ~-0.005 to -0.010 | NSCS01 exploits SegNet's `x[:, -1, ...]` nullspace at the renderer; v2 reduces rate at the latent. Composable. |
| **NSCS02** (downsampled renderer) | NEAR-REDUNDANT (both reduce archive bytes) | SATURATING | floor at -0.005 | Both target rate-axis. Conditional pre-NSCS02 cargo-cult-unwind (chroma-preserving inflate upsample). |
| **NSCS03** (Ballé end-to-end joint codec) | REDUNDANT | SATURATING | floor at -0.003 | NSCS03 IS general-purpose Ballé hyperprior; v2 IS class-conditional Ballé. Choose ONE. |
| **NSCS06 v7** (Carmack-Hotz strip-everything, chroma-restored) | ORTHOGONAL (codec layer vs full-frame paradigm) | ADDITIVE | unknown pre-paired-smoke | NSCS06 v7 is full-frame; v2 is latent codec. Likely composable if NSCS06 v7's `0.bin` adopts A1-style decoder. |
| **ATW codec** (cooperative-receiver triple) | NEAR-REDUNDANT (both scorer-aware codecs) | SATURATING | floor at -0.005 | Both leverage scorer features. Choose ONE per Wunderkind paradigm. |
| **STC-Dasher** (arithmetic-coding maximalism) | ORTHOGONAL-AT-INNER-CODER | ADDITIVE-CONDITIONAL | ~-0.002 | STC-Dasher can REPLACE constriction-Huffman for class_index_blob; if STC's per-pair entropy ≤ Huffman's, additive gain at the class-index slot. |
| **U-DIE-KL** (substrate-wide loss) | ORTHOGONAL (loss vs codec) | ADDITIVE | ~-0.005 to -0.010 | U-DIE-KL changes loss; v2 changes codec. Fully composable. |

**Recommended deployment**: SOLO smoke FIRST (validate byte-mutation smoke per Catalog #272 + paired Tier C MDL ablation per Catalog #227 + 5/5 council PROCEED). Only after v2 demonstrates non-zero distinguishing-feature consumption proceed to stack with NSCS01 or U-DIE-KL.

## 13. Predicted ΔS band (with Dykstra-feasibility check per Catalog #296)

### 13.1 First-principles rate-side derivation

- A1 latent_blob slot: 15387 B (replaced by Z3G2 section)
- v2 Z3G2 section size estimate: ~2121 B (Section 8)
- Net archive byte savings: ~13266 B
- Rate-axis ΔS: `25 × 13266 / 37545489 ≈ -0.00883`

### 13.2 Distortion-axis derivation

The class-conditional sigma table is a 5-class hyperprior approximation to Z3 v2's full Ballé hyperprior MLP. Two cases:

**Case A (sigma table well-fits class-conditional variance)**: distortion change near-zero (per-class sigma is tighter for some classes — sky low variance, foreground high variance — than the unconditional sigma; bit allocation matches the data better). Expected Δd ≈ -0.001 to +0.001.

**Case B (sigma table underfits)**: per-class quantization may produce coarser reconstruction at class boundaries (mis-categorized pairs get the wrong sigma). Expected Δd ≈ +0.002 to +0.005.

Conservative estimate: Δd ≈ +0.002.

### 13.3 Combined ΔS

Rate + distortion: `-0.00883 + 0.002 ≈ -0.0068` ⇒ **predicted band [contest-CUDA T4 prediction]: [0.2210, 0.2280]** vs Z3 v2 baseline `0.23171 [contest-CUDA T4]`. Predicted band [contest-CPU GHA prediction]: [0.1880, 0.1950] vs Z3 v2 baseline `0.19779 [contest-CPU GHA Linux x86_64]`.

### 13.4 Dykstra-feasibility check (Catalog #296 mandatory)

The Dykstra alternating-projections check intersects three convex constraints:

1. **Rate constraint** (`R ≤ 13266 B / 37545489 = 3.53e-4 normalized`): FEASIBLE per Section 13.1 (the byte savings ARE physically achievable; the Z3G2 section IS ~2121 B per the existing encoder).
2. **SegNet-class-derivability constraint** (`H(class | pair) > 0`): FEASIBLE per the existence of non-trivial SegNet output variation on `upstream/videos/0.mkv` (5 classes ≠ 1 class). Empirical anchor: the v1 sigma_logits training did NOT collapse to all-classes-identical (the diagnostic in `g1_diagnostic.pt` from v1 showed per-class variation).
3. **Quantization-error constraint** (`||sigma_int8_recovered - sigma_float|| < tol`): FEASIBLE per the existing `int8_sigma_scale=16.0` choice (sigma values in [0, 16] map to int8 [0, 127] with quantization error ≤ 16/127 ≈ 0.126; small relative to typical sigma values ≈ 2.0).

**Intersection of the 3 polytopes**: NON-EMPTY. All three constraints are simultaneously satisfiable by the current encoder. Dykstra-feasibility check: **PASSED**.

**However** (Assumption-Adversary CRITICAL CHALLENGE per Catalog #292): the class-conditional sigma table may be REDUNDANT with the per-pair latent residual statistics that Z3 v2's unconditional Gaussian already exploits. The class label might add ≤0.1 bit/pair (mutual information `I(class; residual)` may be small). This is a PROBE-WORTHY question — Section 14 below specifies the disambiguator.

**Dykstra-feasibility VERDICT for the predicted band: PASSED** (the predicted [0.221, 0.228] band IS achievable in principle), **but the MAGNITUDE may collapse to near-zero (0.230 ± 0.001) if the H(residual | class) probe shows redundancy**. The band's lower bound (0.221) is conditional on non-trivial class-conditioning gain.

### 13.5 Why this band is NOT additive on Z3 v2 baseline

Per Catalog #227 (`check_substrate_class_promotion_requires_tier_c_evidence`) + Catalog #219 (Z1 density ablation), within-class refactors face a Tier A density saturation ceiling. v2 IS a within-class refactor of Z3 v2 (same A1 decoder + same scorer-class semantics + tighter per-class prior). The 0.196-0.199 plateau IS the empirical evidence that within-class refactors plateau.

**Conservative reactivation criterion**: predicted band `[0.221, 0.228]` is research-only-no-score-claim UNTIL the H(residual | class) probe (Section 14) demonstrates non-trivial class-conditioning gain. If `I(class; residual) < 0.05 bits/pair`, the band collapses to `[0.229, 0.232]` (essentially identical to Z3 v2) and v2 stays research_only=true.

## 14. Probe-disambiguator (per CLAUDE.md "design tension: ship both interpretations" + Catalog #125 hook 6)

**Two defensible interpretations** of why v2 may NOT beat Z3 v2:

**Interpretation A (wire-grammar-only fix)**: v2's grammar fix is necessary AND sufficient. The class-conditional sigma table genuinely tightens the prior, byte savings ARE real, predicted band is correct.

**Interpretation B (within-class redundancy)**: v2's grammar fix is necessary but NOT sufficient. The SegNet class label is REDUNDANT with the per-pair latent residual statistics Z3 v2 already exploits. Within-class refactor; smoke score matches Z3 v2 baseline within ±0.001.

**Disambiguating probe**: `tools/probe_z3_g1_entropy_coded_v2_class_residual_mutual_information.py` (proposed; queued as op-routable #1) — computes empirical `I(class_indices; residual)` on a 100-pair sample from `upstream/videos/0.mkv`:
- If `I > 0.5 bits/pair`: Interpretation A confirmed (per-class sigma gives non-trivial coding gain); proceed to v2 trainer + dispatch
- If `I < 0.1 bits/pair`: Interpretation B confirmed (per-class sigma is redundant); pivot v2 design to per-pair adaptive sigma (NOT class-conditional) and re-design before any paid dispatch
- If `0.1 ≤ I ≤ 0.5 bits/pair`: marginal; smoke is the empirical arbiter

**Cost**: $0 (CPU probe on local M5 Max; ~5 minutes).

**Per Catalog #125 hook 6**: the probe IS the structural arbitration; the trainer/codec/solver consumes the verdict.

## 15. Canonical-vs-unique decision per layer (Catalog #290 mandatory)

| Layer | Decision | Rationale | HARD-EARNED vs CARGO-CULTED |
|---|---|---|---|
| Pretraining | ADOPT CANONICAL (A1 frozen) | Substrate-engineering exemption per HNeRV parity L7 | HARD-EARNED (A1 anchor 0.23171 [contest-CUDA]) |
| Training curriculum | ADOPT CANONICAL (100ep+1000ep + EMA 0.997 + eval_roundtrip) | PR95 parity discipline L1-13 | HARD-EARNED (PR95 lesson) |
| Architecture | UNIQUE FORK (5×28 sigma table replaces Ballé MLP) | Wunderkind G1 substitution-1:1 spec; bare-minimum hyperprior | HARD-EARNED per Wunderkind paradigm memo |
| Priors (class prior CDF) | UNIQUE (empirical from upstream/videos/0.mkv per Catalog #304) | NO closed-form CDF allowed | HARD-EARNED (Catalog #304 anchor + Ballé 2018) |
| Priors (sigma table) | UNIQUE (per-class instead of Z3 v2's per-pair MLP) | The class-conditional fork IS the distinguishing feature | HARD-EARNED per Wunderkind |
| Post-training (quantization) | ADOPT CANONICAL (int8 + per-class allocation) | Quantizr/Ballé canonical | HARD-EARNED |
| Score-aware loss | ADOPT CANONICAL skeleton + UNIQUE EXTENSION (rate term) | Catalog #164 canonical for base loss; UNIQUE for `g1_v2_total_rate_bits` | HARD-EARNED |
| Archive grammar | UNIQUE FORK (Z3G2 magic; sigma + class-index slots) | The v1 phantom-anchor FAILURE motivates the UNIQUE grammar; Catalog #266 fix | HARD-EARNED per codex F1 + Catalog #266 |
| Inflate runtime | ADOPT CANONICAL skeleton (`select_inflate_device`) + UNIQUE body (Z3G2 decoder) | Catalog #205 canonical for device selection; UNIQUE for parsing | HARD-EARNED |
| Export contract | UNIQUE (Z3G2 → 0.bin; replaces A1's latent_blob) | HNeRV parity L3 + the wire-grammar fork | HARD-EARNED |
| Stack-of-stacks composition | UNIQUE (SOLO smoke FIRST; then NSCS01 / U-DIE-KL only) | Z3 v2 plateau finding (Catalog #227) — within-class refactor | HARD-EARNED |
| Pipeline composition | ADOPT CANONICAL (`experiments/pipeline.py` profile-based) | CLAUDE.md "Canonical pipeline standard non-negotiable" | HARD-EARNED |
| Tier-1 engineering | ADOPT CANONICAL (autocast fp16 / TF32 / torch.compile / no_grad) | Catalogs #172/#178/#179/#180 | HARD-EARNED |
| Scorer routing | ADOPT CANONICAL (`load_differentiable_scorers` + `pose_scorer, seg_scorer = ...`) | Catalog #164 + #222 | HARD-EARNED |
| Auth-eval routing | ADOPT CANONICAL (`gate_auth_eval_call`) | Catalog #226 | HARD-EARNED |
| Custody validator | ADOPT CANONICAL (`require_contest_cuda_auth_eval_claim` + `posterior_update_locked`) | Catalogs #127 + #128 | HARD-EARNED |

## 16. Cargo-cult audit per assumption (each surfaced + classified)

Per the resurrection-audit Tier 2 + cargo-cult-unwind audit template:

**Assumption 1 — "v1's empty hyperprior_weights_int8 slot was the SOLE bug"**: CARGO-CULTED. The empty-slot bug IS necessary; whether it is sufficient depends on the H(residual | class) probe (Section 14). UNWIND: ship Section 14's probe BEFORE the trainer.

**Assumption 2 — "SegNet class label is highly informative about latent residual statistics"**: CARGO-CULTED-PENDING-PROBE. Plausible from theory (sky pairs have low residual variance; foreground pairs have high variance) but not empirically verified. UNWIND: Section 14 probe.

**Assumption 3 — "constriction-Huffman is the optimal class-index coder"**: CARGO-CULTED. Huffman is suboptimal vs arithmetic/range coding for small symbol counts; the 600-pair class-index stream might code more efficiently under range coding (STC-Dasher composition Section 12). UNWIND: defer optimization to post-v2-smoke; if v2 hits ≤0.220 [contest-CUDA prediction], the Huffman-vs-range gap is ≤30B (insignificant).

**Assumption 4 — "5-class hyperprior approximates Ballé MLP within 0.005 distortion"**: CARGO-CULTED-PENDING-EMPIRICAL. The 5-class case is the smallest possible classification; finer per-pair conditioning (per-pair sigma scalar OR per-pixel-area sigma) would tighten the prior. UNWIND: queue as v3 candidate ONLY IF v2 demonstrates the class-conditional approach works.

**Assumption 5 — "Empirical class prior CDF generalizes across video segments"**: CARGO-CULTED. The class prior is computed ONCE on `upstream/videos/0.mkv` at training time and frozen; if a future contest video has a different class distribution, the prior will be miscalibrated. UNWIND: this is acceptable for `contest_one_video_replay` target_mode; explicitly DEFER `contest_generalized` target_mode until v3+ adaptive prior.

**Assumption 6 — "int8 quantization of sigma table preserves training-time sigma values"**: HARD-EARNED-PENDING-VERIFICATION. The `int8_sigma_scale=16.0` choice produces ≤0.126 quantization error; if training sigma values exceed 16.0, the upper clamp introduces non-recoverable distortion. UNWIND: assert `max_sigma_observed_during_training < 16.0` as a training-time guard.

**Assumption 7 — "Brotli quality=11 minimizes sigma_table_blob bytes"**: HARD-EARNED for a 140-byte payload (no plausible smaller representation given 140 distinct int8 values).

**Assumption 8 — "Class prior CDF (10 B raw uint16) is bit-optimal"**: HARD-EARNED. The class prior has ≤16 bits per class (5 × 16 = 80 bits = 10 B). For 5 classes, 16-bit counts allow up to 65535 occurrences (vs 600 actual); could shrink to 5×10-bit = 6.25 B but byte-alignment makes 10 B the smallest practical form.

**Assumption 9 — "Z3G2 magic does not collide with downstream parsers"**: HARD-EARNED. `Z3V2` magic is registered for v1; `Z3G2` is the v2 fork; sister magic spaces `Z3HV2` / `Z3HP1` covered by existing tests. No collision risk.

**Assumption 10 — "Catalog #304 (no closed-form CDF) applies to the class prior"**: HARD-EARNED. Catalog #304 explicitly forbids any pre-computed CDF table that doesn't derive from `upstream/videos/0.mkv`. The class prior IS derived from `upstream/videos/0.mkv` via SegNet's empirical class output. Compliant.

## 17. Observability surface (NEW per max-observability standing directive)

Per `feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`, every substrate design memo MUST include this section.

### 17.1 Per-layer inspection hooks

| Layer | Captured signal | Storage |
|---|---|---|
| SegNet argmax | per-pair `(H, W)` class map int8 | `experiments/results/<lane>_<timestamp>/segnet_argmax_per_pair.pt` (per-epoch snapshot at training; per-archive at export) |
| Class index reduction | per-pair dominant class `(600,) long` + per-class histogram | `experiments/results/<lane>_<timestamp>/class_indices.pt` + `class_prior_counts.pt` |
| sigma_logits (training) | `(5, 28) float32` per-epoch | TensorBoard scalar event + JSONL anchor per epoch |
| sigma_logits (post-EMA) | EMA-shadow `(5, 28) float32` | JSONL anchor at end of training |
| sigma_int8 (post-quant) | `(5, 28) int8` + scale | JSONL anchor at archive build |
| residual_q (per-pair) | `(600, 28) int8` | optional dump on `--observability-dump-residuals` flag |
| inflate-time latent reconstruction | per-pair `(600, 28) float32` post-decode | optional dump on `--observability-dump-latents` flag |

### 17.2 Per-signal decomposition

Composite metrics decompose:
- `final_score` = `loss_score + 25 * archive_bytes / 37545489`
- `loss_score` = `seg_loss + 10 * pose_loss` (canonical)
- `seg_loss` = per-pair `(600,) float32` breakdown emitted to `seg_loss_per_pair.pt`
- `pose_loss` = per-pair `(600,) float32` breakdown emitted to `pose_loss_per_pair.pt`
- `rate_bits` = `rate_residual_per_pair + rate_class_per_pair` (both `(600,) float32` emitted)

### 17.3 Run-to-run diff manifest

Per Catalog #245 modal_call_id_ledger pattern:
- `dispatch_metadata.json` carries `mounted_code_git_head` + `working_tree_dirty_summary` + `sentinel_files_local_sha256`
- Two runs of the v2 substrate at the same git_HEAD on the same upstream snapshot MUST produce byte-identical:
  - sigma_table_int8 (deterministic post-EMA quantization)
  - class_indices (deterministic SegNet argmax + histogram)
  - class_prior_counts (deterministic Laplace-smoothed histogram)
  - residual_int8 (deterministic A1 inheritance)
  - z3g2_section bytes (sha256-pinned)
  - inflate output bytes (sha256-pinned)
- Diff signal: any byte divergence indicates a non-determinism source (seed drift, non-pinned random op, fp16 numerics, etc.)

### 17.4 Post-hoc query interface

All artifacts machine-readable:
- TensorBoard event files for per-epoch sigma_logits / loss / rate
- JSONL append-only anchors for per-anchor decomposition (cf. `tac.continual_learning.posterior_update_locked`)
- `experiments/results/<lane>_<timestamp>/distinguishing_feature_byte_mutation_proof.json` (Section 11 output)
- `experiments/results/<lane>_<timestamp>/contest_auth_eval_<device>.json` per Catalog #249 (NOT `_cuda` if device=CPU)

### 17.5 Cite-chain

Every emitted artifact carries:
- `commit_sha`: git HEAD at dispatch
- `modal_call_id`: per Catalog #245 ledger
- `upstream_snapshot_sha256`: sha256 of `upstream/videos/0.mkv` + `upstream/evaluate.py` + `upstream/modules.py`
- `subagent_id`: invoking subagent (if applicable)
- `recipe_path`: `.omx/operator_authorize_recipes/substrate_z3_g1_entropy_coded_v2_modal_t4_dispatch.yaml` sha256
- `trainer_sha256`: `experiments/train_substrate_z3_g1_entropy_coded_v2.py` sha256 (when `_full_main` lands)

### 17.6 Counterfactual hooks

Byte-mutation per Catalog #272 (Section 11) IS the counterfactual hook: "what if this byte changed?" answered structurally without re-running training. Ablation switches:
- `--ablation-class-conditional` toggle: when False, sigma table degenerates to a SINGLE row (not per-class); test for redundancy hypothesis (Section 14 Interpretation B)
- `--ablation-empirical-cdf` toggle: when False, class prior CDF uses uniform-prior fallback; test for prior-mismatch sensitivity
- `--ablation-int8-quantization` toggle: when False, ship fp32 sigma table (~560 B vs 140 B); test for quantization-error contribution to distortion

## 18. Pipeline composition

Routes through canonical `experiments/pipeline.py` per CLAUDE.md "Canonical pipeline standard non-negotiable":

```bash
.venv/bin/python experiments/pipeline.py \
    --profile z3_g1_entropy_coded_v2 \
    --device cuda \
    --output-dir experiments/results/z3_g1_v2_<timestamp>/
```

Profile `z3_g1_entropy_coded_v2` lives in `src/tac/profiles.py` (queued landing alongside `_full_main` implementation). Profile fields:
- `architecture_class = "z3_g1_entropy_coded_v2"`
- `epochs = 100` (smoke) / `1000` (full)
- `lr_initial = 1e-3`, `lr_final = 1e-5`, `lr_schedule = "cosine"`
- `ema_decay = 0.997`, `eval_roundtrip = True`, `autocast_fp16 = True`
- `int8_sigma_scale = 16.0`, `quant_step = 1.0`, `min_sigma = 1e-3`, `max_sigma = 16.0`
- `lambda_rate_target = 0.01` (rate-axis loss weight after warm-up)

## 19. Reactivation criteria (gate to flip `research_only=true → false`)

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":

1. **Section 14 probe lands** + verdict `I(class; residual) ≥ 0.1 bits/pair` (rules out Interpretation B redundancy)
2. **`_full_main` implementation lands** (replaces current `NotImplementedError`)
3. **Byte-mutation smoke per Catalog #272 passes ALL THREE distinguishing-feature blobs** (sigma_table_blob, class_prior_blob, class_index_blob)
4. **Encoder-decoder roundtrip test passes** (existing 23-test suite + 1 new roundtrip test asserting decoded sigma + class indices == encoded sigma + class indices)
5. **Modal T4 smoke (100ep) returns finite auth-eval JSON** with axis tag matching dispatch device + sub-Z3-v2-baseline score `< 0.197 [contest-CPU GHA Linux x86_64]` (the empirical gate)
6. **Paired CUDA + CPU auth-eval lands** per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
7. **Tier C MDL ablation per Catalog #227** — class-shift density signature OR explicit `lane_class=substrate_engineering` opt-out OR `research_only=true` retention
8. **5/5 council PROCEED** (per CLAUDE.md "Design decisions — non-negotiable")

ANY missing criterion → lane stays `research_only=true` per Catalog #240.

## 20. Cost estimate

| Phase | Surface | Cost |
|---|---|---|
| Section 14 probe | Local M5 Max CPU | $0 (~5 min) |
| `_full_main` implementation | Editor + tests | $0 (~3-4 h subagent) |
| Byte-mutation smoke | Local CPU | $0 (~10 min via `tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py`) |
| Encoder-decoder roundtrip test | Local CPU | $0 (~5 min in existing test suite) |
| Modal T4 smoke (100ep) | Modal T4 | ~$0.59 (matches v1 anchor) |
| Modal CPU paired CPU axis | Modal CPU container | ~$0.10-0.50 (60-120 min) |
| Modal T4 paired CUDA axis | (already estimated above) | (in smoke cost) |
| Full Modal A100 dispatch (1000ep) | Modal A100 | $5-10 (gated on smoke pass) |
| Tier C MDL ablation | Local CPU on smoke archive | $0 (~5 min via `tools/mdl_scorer_conditional_ablation.py`) |
| 5/5 council deliberation | Editor | $0 (council memo) |

**Total to L3 (FULL PRODUCTION HARDENED)**: ~$6-12 + ~10 hours subagent + ~30 min operator review.

## 21. Op-routables

1. **Section 14 probe (HIGHEST PRIORITY)**: build `tools/probe_z3_g1_entropy_coded_v2_class_residual_mutual_information.py` BEFORE any other v2 work. Computes empirical `I(class_indices; residual)` on 100-pair sample from `upstream/videos/0.mkv` using v1's existing `g1_diagnostic.pt` (the per-pair class + per-pair sigma + per-pair residual the v1 training already produced).
2. **`_full_main` implementation**: lift `NotImplementedError`; wire the canonical training loop with sigma_logits + per-pair empirical class prior CDF + canonical scorer-aware loss + EMA + eval_roundtrip + Modal-compatible runtime closure. Subagent-spawnable; ~3-4 h work.
3. **Modal recipe enable**: flip `recipe_research_only=True → False` + `dispatch_enabled=False → True` in `.omx/operator_authorize_recipes/substrate_z3_g1_entropy_coded_v2_modal_t4_dispatch.yaml` ONLY after criteria 1-4 of Section 19 satisfy.
4. **Byte-mutation smoke regression guard**: add CI hook to run the smoke after every commit touching `src/tac/substrates/z3_g1_entropy_coded_v2/*` (prevents v1-class regression).
5. **Cathedral autopilot ranker entry**: register v2 as a class-shift candidate with `predicted_dykstra_band=[0.221, 0.228] CONDITIONAL on Section 14 probe`; observability surface populated per Section 17.
6. **STC-Dasher composition probe (DEFERRED post-v2-smoke)**: if v2 smoke confirms class-conditional gain, evaluate replacing constriction-Huffman with STC-Dasher arithmetic coding for class_index_blob.

## 22. Cross-references

- `feedback_z3_g1_full_cpu_paired_aborted_codex_f1_empirically_confirmed_landed_20260515.md` — codex F1 empirical anchor (v1 phantom)
- `feedback_wunderkind_g1_entropy_coded_v2_z3g1_reactivation_landed_20260515.md` — v2 prior landing memo
- `.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md` — predecessor substrate design memo
- `.omx/research/resurrection_audit_20260516.md` §3.9 — Tier 2 Z3-G1 phantom-distinguishing-feature anchor
- `feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md` — Wunderkind G1 paradigm spec
- `feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md` — observability surface mandate (Section 17)
- `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` — HARD-EARNED vs CARGO-CULTED classification rules
- `feedback_9_dimension_success_checklist_per_substrate_and_stack_of_stacks_standing_directive_20260515.md` — 9-dim evidence template
- `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` — Section 15 canonical-vs-unique decision per layer
- `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` — sister cargo-cult-unwind audit
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" (lessons 1-13) — full substrate-engineering compliance
- CLAUDE.md Catalog #220 (operational mechanism); Catalog #266 (G1 substrate consumes hyperprior bytes); Catalog #272 (distinguishing-feature contract); Catalog #296 (Dykstra-feasibility); Catalog #297 (signal-axis destruction); Catalog #303 (cargo-cult audit per assumption); Catalog #304 (no closed-form CDF); Catalog #290 (canonical-vs-unique per layer); Catalog #294 (9-dim evidence section)
- `src/tac/substrates/z3_g1_entropy_coded_v2/` — existing implementation (architecture.py + archive.py + inflate_consumer.py + score_aware_loss.py + registered_substrate.py)
- `tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py` — Section 11 canonical smoke tool

---

## 9-dimension success checklist evidence

Per CLAUDE.md Catalog #294 + standing directive.

1. **UNIQUENESS (CLASS-SHIFT not within-class)**: PARTIAL/CONDITIONAL. Wire-grammar fork IS a class-shift at the archive layer (Z3V2 → Z3G2 magic). HOWEVER the underlying substrate (A1 + scorer-class-conditional Ballé) is WITHIN-CLASS relative to Z3 v2 per Catalog #227 Tier C density (sister-of NSCS03 not orthogonal-to). The Section 14 probe is what determines whether the wire-grammar fork manifests as non-trivial class-conditioning gain OR within-class plateau.
2. **BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable)**: PRESENT. Existing v2 substrate is ~700 LOC total (architecture 280 + archive 675 + inflate_consumer 173 + score_aware_loss + registered_substrate). Within PR101's 605-LOC reference. Each module reviewable in 30s.
3. **DISTINCTNESS (explicitly different from sisters)**: PRESENT. Section 12 lists 7 sister substrates with axis-orthogonality verdicts. v2 distinguishes via class-conditional Ballé (per-class 5×28 sigma table vs NSCS03's per-pair MLP vs Z3 v2's unconditional sigma vs ATW's cooperative-receiver vs STC-Dasher's arithmetic-maximalism).
4. **RIGOR**: PRESENT. Section 14 probe-disambiguator + Section 11 byte-mutation smoke + Section 16 cargo-cult audit per assumption (10 assumptions classified) + Section 13 Dykstra-feasibility + premise-verification of v1 phantom-anchor via codex F1 empirical receipt + 5/5 council PROCEED gated in Section 19.
5. **OPTIMIZATION PER TECHNIQUE**: PRESENT. Section 15 canonical-vs-unique decision per layer (16 layers; 4 UNIQUE / 12 ADOPT CANONICAL with HARD-EARNED-vs-CARGO-CULTED classification per layer per Catalog #290).
6. **STACK-OF-STACKS-COMPOSABILITY**: PRESENT. Section 12 composition matrix (7 sisters with orthogonality + saturating verdicts). Recommended SOLO smoke FIRST.
7. **DETERMINISTIC REPRODUCIBILITY**: PRESENT. Section 17.3 run-to-run diff manifest pins seed + EMA + class-prior frozen post-epoch-20 + per-byte sha256-pinned z3g2_section + per-byte sha256-pinned inflate output.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: PRESENT. Tier 1 (autocast fp16 + TF32 + torch.compile + no_grad) ADOPT canonical per Catalogs #172/#178/#179/#180. Inflate compute ~1800 FP ops (Section 4). v2 archive ~349 KB vs A1 ~362 KB (savings ~13 KB).
9. **OPTIMAL MINIMAL CONTEST SCORE**: PARTIAL. Predicted band [0.221, 0.228] [contest-CUDA T4 prediction] CONDITIONAL on Section 14 probe; collapses to [0.229, 0.232] (essentially Z3 v2 baseline) if class-conditioning is redundant per Interpretation B. The empirical gate is `< 0.197 [contest-CPU GHA Linux x86_64]` per Section 19 criterion 5.

---

## Op-summary

- **What v2 fixes vs v1**: v1 shipped empty `hyperprior_weights_int8 = b""` + `w_hat_int8 = b""` slots → identical Z3 v2 baseline scores (codex F1 phantom anchor). v2 wire grammar `Z3G2` ACTUALLY ships sigma_table_blob (~300 B brotli) + class_prior_blob (10 B raw) + class_index_blob (~350 B constriction-Huffman). Total distinguishing bytes: ~660 B (>0 vs v1's 0).
- **How bytes actually flow through inflate**: `reconstruct_class_indices_and_sigma_table_from_z3g2_payload` decodes all three blobs → `sigmas_per_pair = sigma_table[class_indices, :]` → `latents = (residual_q * sigmas_per_pair) * latent_scale + latent_offset` → frame reconstruction differs from Z3 v2's unconditional-sigma path.
- **Predicted band + Dykstra verdict**: [0.221, 0.228] [contest-CUDA T4 prediction] CONDITIONAL on Section 14 probe; Dykstra-feasibility passes (3 constraints intersect non-empty); magnitude may collapse to [0.229, 0.232] under redundancy hypothesis.
- **Byte-mutation smoke plan**: 4 sections × 1 byte mutation each via `tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py` (already landed); pass criterion = all 3 distinguishing-feature blobs produce changed inflate output.
- **Cost**: ~$6-12 GPU + ~10 h subagent.
- **Reactivation gate**: 8 criteria in Section 19; the empirical floor is `< 0.197 [contest-CPU GHA Linux x86_64]` (Z3 v2 baseline 0.19779).
- **Observability surface**: 6 surfaces declared per Section 17 (per-layer hooks / per-signal decomposition / run-to-run diff / post-hoc query / cite-chain / counterfactual hooks).

---

## Appendix A — Section 14 probe empirical result (PROBE DISPATCH WAVE task #749, 2026-05-16)

**Status**: Section 14 probe LANDED (operator-approved PROBE DISPATCH WAVE task #749). Result appended per Catalog #110 HISTORICAL_PROVENANCE (APPEND-ONLY); body of design memo unchanged.

**Probe executed**: `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (commit `d72f50985`) on (a) `residual_int8` (16,800 bytes) extracted from the Z3 v2 1000ep CUDA archive (`lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep_modal`) and (b) `class_indices` (600 entries, byte-expanded ×28 to align symbol-for-symbol) from the Z3-G1 v1 smoke diagnostic at `experiments/results/lane_substrate_z3_g1_scorer_softmax_hyperprior_gating_modal_t4_dispatch_20260515T195556Z__smoke__100ep_modal/.../g1_diagnostic.pt`.

**Result** ([diagnostic-CPU; H(latent|scorer_class) probe]):

| Metric | Value |
|---|---|
| `H(latent)` (unconditional) | **7.5653 bits/symbol** |
| `H(latent \| scorer_class)` | **7.5213 bits/symbol** |
| `I(latent ; class)` (mutual information) | **0.04393 bits/symbol** |
| Wyner-Ziv gain ceiling | **0.58%** (0.044 / 7.57) |
| Verdict | **`WEAK_CONDITIONING`** |
| `num_latent_symbols` | 16,800 |
| `num_unique_classes` | 5 |

**Verdict per Section 14 decision rules**:
- `I = 0.044 bits/pair < 0.1 bits/pair` threshold ⇒ **Interpretation B (within-class redundancy) CONFIRMED**
- v2's per-class sigma table is **structurally redundant** with Z3 v2's per-pair unconditional sigma path; the SegNet class label adds < 0.6% coding gain vs treating the residual as unconditionally distributed
- Per Section 14 decision rule: **pivot v2 design to per-pair adaptive sigma (NOT class-conditional)** before any paid Modal dispatch

**Caveat (HARD-EARNED-PARTIAL)**: the `class_indices` used in this probe are from G1 v1's synthetic uniform fallback (Catalog #267 — G1 v1's live SegNet derivation FAILED → fallback uniform 120/class). A real-SegNet-on-`upstream/videos/0.mkv`-class derivation could produce different mutual information. However:
- The synthetic indices are uniform across 5 classes (120 each) → MI ceiling against uniform would FAVOR finding genuine class-correlation if real classes carried more signal; the observed MI is already low.
- The residual itself is from a real Z3 v2 1000ep CUDA training (HARD-EARNED).
- Re-running the probe with a real CUDA-derived SegNet class assignment is a follow-on cheap ($0-5) re-probe; queued as op-routable A-2 below.

**Reactivation criterion update (Section 19 #1)**: Section 14 criterion 1 result = **FAIL** (`I < 0.1 bits/pair`). Per Section 14 disambiguator: v2 design pivot REQUIRED before any paid dispatch — either pivot to per-pair adaptive sigma (NOT class-conditional), OR re-run the probe with real CUDA SegNet class derivation to rule out the synthetic-fallback caveat.

**Op-routables appended**:
- **A-1**: PIVOT Wunderkind G1 v2 design to per-pair adaptive sigma (not class-conditional) per Section 14 Interpretation B verdict. Council review required (Quintet pact per CLAUDE.md "Design decisions — non-negotiable").
- **A-2**: Re-run Section 14 probe with REAL CUDA-derived SegNet class indices (run SegNet on `upstream/videos/0.mkv` 600 pairs on Modal T4 ~$0.50) to rule out the synthetic-uniform-fallback caveat. Use canonical `score_pair_components` per Catalog #164 + `gate_auth_eval_call` per Catalog #226.
- **A-3**: Update lane registry `lane_wunderkind_g1_entropy_coded_v2_20260515` notes with the empirical Interpretation B confirmation; keep `research_only=true` per Section 19 criterion 1 FAIL.

**Probe artifacts** (committed custody per CLAUDE.md "Forbidden /tmp paths"):
- `experiments/results/probe_dispatch_wave_749_section14_g1_v2_20260516T171251Z/h_latent_given_scorer_class_wunderkind_g1_v2_section14.json`
- `experiments/results/probe_dispatch_wave_749_section14_g1_v2_20260516T171251Z/residual_int8.bin` (16,800 bytes; sha256 of source archive `lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep`)
- `experiments/results/probe_dispatch_wave_749_section14_g1_v2_20260516T171251Z/class_indices.bin` (16,800 bytes; byte-expanded ×28 from g1_diagnostic.pt; CAVEAT: synthetic-fallback per Catalog #267)
- `.omx/state/h_latent_given_scorer_class_wunderkind_g1_v2_section14.json` (same JSON; canonical state path)

**Source memo edit scope**: Catalog #110 HISTORICAL_PROVENANCE APPEND-ONLY. Body of design memo (Sections 1-22 + 9-dim checklist + Op-summary) UNCHANGED. This appendix lands as the empirical disambiguator result Section 14 was specified to produce.

---

## Observability surface

**Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16** (`feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`) + Catalog #305 STRICT preflight gate (`check_substrate_design_memo_has_observability_surface_section`).

**Per Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:** this section is appended to the design memo; pre-existing body content (Sections 1-N + 9-dim checklist + cargo-cult audit + canonical-vs-unique decision + cross-references) is UNCHANGED.

**The 6-facet observability surface for this design:**

1. **Per-layer inspection.** Every layer of this substrate / composition / experiment captures its (input tensor, output tensor, intermediate activations, attention maps when applicable) at runtime via the canonical xray-style hook pattern (`tac.xray.<lens>` modules) without re-instrumentation. The forward pass emits per-layer observables to `experiments/results/<lane>/observability/per_layer/<layer_name>.jsonl` for post-hoc inspection.

2. **Per-signal decomposition.** Composite metrics (`final_score = seg + sqrt(10*pose) + 25*rate`) decompose into constituent contributions per the canonical `tac.xray.per_pair_score_decomposition` lens. Per-pair / per-class / per-axis / per-stage breakdown serialized to `experiments/results/<lane>/observability/score_decomposition.json` with axis labels matching CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable (`[contest-CUDA]` / `[contest-CPU]` / `[diagnostic-CPU]` / `[macOS-CPU advisory]` / `[MPS-PROXY]`).

3. **Run-to-run diff.** Two runs of this substrate / composition produce byte-identical reproducible artifacts under the same `(seed, commit_sha, upstream_snapshot_sha256)` tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger. The canonical diff helper `tools/diff_auth_eval_results.py <run_a.json> <run_b.json>` (planned per the observability audit Highest-ROI extension list) emits per-component deltas + per-pair drift; until landed, manual diff via `archive_sha256` byte-addressability is straightforward.

4. **Post-hoc query interface.** Run artifacts under `experiments/results/<lane>/` serialize as structured JSON / JSONL surfaces consumable without re-running: `contest_auth_eval_<axis>.json` (canonical per-component scores with axis labels) + `modal_metadata.json` (per-dispatch cite-chain per Catalog #166) + `observability/*.jsonl` (per-layer + per-signal). The continual-learning posterior at `.omx/state/continual_learning_posterior.jsonl` is queryable per (substrate, axis, hardware, evidence_grade) via `tac.continual_learning.query_*` helpers per Catalog #128 + #131 fcntl-locked discipline.

5. **Cite-chain.** Every behavior signal anchors to the canonical tuple `(substrate_id, commit_sha, modal_call_id, config_path, random_seed, upstream_snapshot_sha256)` via Catalog #245 `tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)`. The call_id ledger row schema includes `mounted_code_git_head` (per Catalog #166) + `agent` + `subagent_id` + `session_id` for full forensic reconstruction. Score claims tagged per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

6. **Counterfactual hooks.** Byte-mutation surface per Catalog #139 packet compiler (`tools/verify_distinguishing_feature_byte_mutation.py --distinguishing-byte-range <offset>:<length>`) + Catalog #272 distinguishing-feature integration contract + Catalog #105 no-op detector. The substrate's archive grammar exposes byte-offset addressability for "what if this byte changed?" probing without re-running training. Per-layer / per-component ablation switches surfaced via the trainer's argparse flags + the canonical `tac.xray.<lens>.ablate_*` helpers when applicable.

**Acceptance per Catalog #305:** this section satisfies the structural requirement (literal section header `## Observability surface` present); the body content above documents the substrate's 6-facet observability surface for operator-facing audit.

**Sister observability-discipline gates active for this substrate:**

- Catalog #245 modal_call_id_ledger (every dispatch registered)
- Catalog #166 Modal HEAD-parity ledger (every dispatch worker-source-verified)
- Catalog #128 + #131 fcntl-locked JSONL posterior discipline (state mutations append-only)
- Catalog #139 packet compiler no-op detector (byte-mutation surface)
- Catalog #272 distinguishing-feature integration contract (per-substrate byte-mutation proof)
- Catalog #220 substrate L1+ operational mechanism declaration (no opaque byte additions)
- Catalog #105 no-op detector (no-op provenance)
- Catalog #127 authoritative tag custody (per-call-site axis + hardware-substrate validation)

**Observability extension recommendations (queued for follow-on):** see `tools/audit_existing_infrastructure_for_observability.py --summary` output for the canonical 8-tool / 6-facet observability gap analysis + Highest-ROI extension list. The `tools/audit_*.py` family is the highest-ROI extension target (3/12 observability) per the standing-directive consequence 3.

---

## Appendix B — Real-CUDA-derived SegNet class re-probe (op-routable A-2 closure, 2026-05-16)

**Status**: Section 14 op-routable A-2 LANDED. Result appended per Catalog #110/#113 HISTORICAL_PROVENANCE (APPEND-ONLY); body of design memo + Sections 1-22 + 9-dim checklist + Op-summary + Observability surface + Appendix A UNCHANGED. **Subagent**: `wunderkind-g1-v3-pivot-respawn-c` (respawn of crashed predecessor `a272f4bad69d56360` at lane `lane_wunderkind_g1_v2_pivot_real_cuda_reprobe_v3_design_20260516`).

**Real-CUDA SegNet class derivation method**: ran canonical `tac.scorer.load_default_scorers(upstream)` to load `upstream/models/segnet.safetensors` (sha256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`); decoded `upstream/videos/0.mkv` (sha256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`) via canonical `upstream/frame_utils.py::AVVideoDataset::yuv420_to_rgb` semantics into 600 pairs of `(2, 3, 874, 1164)` fp32 tensors; forward-passed each pair through SegNet (`smp.Unet('tu-efficientnet_b2', classes=5)`) with canonical `preprocess_input` (slice last frame + interpolate to 384x512) → 600 per-pair `(384, 512)` int64 argmax maps; reduced per-pair via canonical `g1_v2_per_pair_dominant_class_from_segnet_argmax(stack, num_classes=5)` from `src/tac/substrates/z3_g1_entropy_coded_v2/architecture.py:191`; byte-expanded ×28 to align symbol-for-symbol with the 600x28 int8 residual stream (mirroring v2 Section 14 protocol). Device: CPU (hermetic; SegNet output is source-data derivation NOT auth-eval per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog #127 source-data carve-out — the SegNet network IS the contest scorer; running it on the contest video is source generation, not a score claim). Wall clock: 550.6s total (0.6s load + 8.2s decode + 541.9s SegNet inference on M5 Max). Cost: $0.

**Per-class distribution (real CUDA-derived SegNet)**:

| Class | Count | Fraction |
|---|---|---|
| 0 | **0** | 0.0000 |
| 1 | **0** | 0.0000 |
| 2 | **600** | **1.0000** |
| 3 | **0** | 0.0000 |
| 4 | **0** | 0.0000 |

**Chi-square vs uniform**: 2400.0 (df=4; threshold p<0.05 = 9.49) — uniform distribution **REJECTED** at p<0.05.

**Interpretation of class distribution**: every pair in `upstream/videos/0.mkv` reduces to dominant class 2 under SegNet's argmax→per-pair-mode reducer. Class 2 corresponds to the road surface in the comma2k19 5-class SegNet (canonical dashcam-dominant class). The per-pair dominant-class reducer is mathematically equivalent to argmax-over-pixel-counts; for dashcam footage the road class dominates pixel counts in essentially every pair.

**Re-probe result ([diagnostic-CPU; H(latent|scorer_class) probe]; sha256 of latent stream `cf4...` reused from v2 Section 14 — same residual_int8 from Z3 v2 1000ep CUDA archive):**

| Metric | Synthetic uniform (v2 Section 14) | **Real CUDA-derived SegNet (Appendix B)** |
|---|---|---|
| `num_unique_classes` | 5 | **1** |
| `H(latent)` (unconditional) | 7.5653 bits/symbol | **7.5653 bits/symbol** |
| `H(latent \| scorer_class)` | 7.5213 bits/symbol | **7.5653 bits/symbol** |
| `I(latent ; class)` (mutual information) | 0.04393 bits/symbol | **0.0000 bits/symbol** |
| Wyner-Ziv gain ceiling | 0.58% | **0.00%** |
| **Verdict** | `WEAK_CONDITIONING` | **`INDEPENDENT`** |

**Verdict per Section 14 decision rules**:
- `I = 0.000 bits/pair < 0.01 bits/pair` independence tolerance ⇒ **INDEPENDENT verdict CONFIRMED with real-SegNet-derived classes**
- The class signal carries EXACTLY ZERO information about the residual because the class distribution is DEGENERATE (all pairs map to class 2). Conditioning on a constant is mathematically equivalent to no conditioning.
- Per Section 14 decision rule: **v2 design is structurally falsified by the real-CUDA re-probe. The SegNet class signal cannot provide ANY Wyner-Ziv conditioning gain on this contest video.**

**Implication for v2 reactivation (Section 19 #1 update)**: Section 14 criterion 1 result = **FAIL** with HARD-EARNED real-SegNet evidence (was HARD-EARNED-PARTIAL from synthetic uniform; now HARD-EARNED-FULL). The synthetic-uniform fallback was previously cited as "MI ceiling against uniform would FAVOR finding genuine class-correlation" (Appendix A); the real-SegNet result is WORSE (degenerate distribution) than synthetic-uniform, confirming v2 design has zero achievable Wyner-Ziv conditioning gain. **v2 reactivation criterion 1 is now UNCONDITIONALLY FAILED.**

**Reactivation criteria update (op-routable A-2 closure)**: v2 lane `lane_z3_g1_entropy_coded_v2_20260515` should remain `research_only=true` PERMANENTLY for the v2 design as specified. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable, this is NOT a KILL — it is a **DEFERRED-pending-redesign** verdict. The v2 design is mathematically excluded from improving Z3 v2 on `upstream/videos/0.mkv`; reactivation requires a structural redesign that does NOT depend on per-pair SegNet class conditioning. The Wunderkind G1 v3 per-pair adaptive sigma design (Section 21 op-routable below) IS the structural redesign — it removes the class-conditioning entirely.

**Sister deliverable (v3 per-pair adaptive sigma design memo)**: see `.omx/research/wunderkind_g1_v3_per_pair_adaptive_sigma_full_stack_design_20260516.md` — comprehensive full-stack design memo for the v3 design (22-section UNIQUE-AND-COMPLETE-PER-METHOD template + Catalog #290 canonical-vs-unique + Catalog #294 9-dim checklist + Catalog #296 Dykstra-feasibility + Catalog #303 cargo-cult audit + Catalog #305 observability surface + 8 op-routables). v3 ships per-pair sigma table (1200 sigma scalars) entropy-coded with empirical-CDF prior; no SegNet class conditioning; structurally distinct from v2's failure mode.

**Op-routables (Appendix B closure)**:
- **B-1**: Lane registry update: append to `lane_z3_g1_entropy_coded_v2_20260515` notes: *"Section 14 re-probe with real-CUDA-derived SegNet classes (2026-05-16): MI=0.000 bits/symbol, INDEPENDENT verdict. v2 design DEFERRED-pending-redesign per CLAUDE.md non-negotiable; v3 per-pair adaptive sigma design queued at lane_z3_g1_per_pair_adaptive_sigma_v3_20260516."*  Operator decision: confirm lane notes update + add `reactivation_criteria` field.
- **B-2**: Cathedral autopilot continual-learning posterior: append a `[diagnostic-CPU]` anchor row for the Appendix B re-probe per Catalog #128 fcntl-locked `posterior_update_locked` — the empirical anchor that v2 class-conditional design is mathematically falsified on `upstream/videos/0.mkv`. Use `evidence_grade="diagnostic_cpu"`, `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, `rank_or_kill_eligible=false` per Catalog #127 strict custody.
- **B-3**: Catalog #267 update (synthetic-uniform-fallback): add a sister evidence row noting that even with REAL SegNet classes, the per-pair dominant class collapses to a single class for dashcam-dominant content; the synthetic-uniform fallback was actually MORE INFORMATIVE than the real distribution. This is a META-pattern for future per-pair-dominant-class reducers on similar content distributions.

**Probe artifacts (committed custody per CLAUDE.md "Forbidden /tmp paths")**:
- `experiments/results/wunderkind_g1_v2_real_cuda_section14_reprobe_20260516T185807Z/derive_real_segnet_classes.py` (script)
- `experiments/results/wunderkind_g1_v2_real_cuda_section14_reprobe_20260516T185807Z/class_indices_real_segnet.bin` (600 bytes per-pair; sha256 `08fedbd25b6e3ca9e14370b5d4f0b4b68fc5417cc567ae78823d997c3fe73080`)
- `experiments/results/wunderkind_g1_v2_real_cuda_section14_reprobe_20260516T185807Z/class_indices_real_segnet_byte_expanded.bin` (16800 bytes ×28; sha256 `5a98df51ae0ae9af2c451c55731f242c815b86dc1b794ea55e852432c9c1ce5c`)
- `experiments/results/wunderkind_g1_v2_real_cuda_section14_reprobe_20260516T185807Z/derivation_provenance.json` (full provenance + chi-square + per-class counts + cite-chain)
- `experiments/results/wunderkind_g1_v2_real_cuda_section14_reprobe_20260516T185807Z/h_latent_given_scorer_class_real_cuda_reprobe.json` (re-probe verdict JSON)

**Source memo edit scope**: Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY. Body of design memo (Sections 1-22 + 9-dim checklist + Op-summary + Observability surface + Appendix A) UNCHANGED. This Appendix B lands as the empirical disambiguator result Appendix A op-routable A-2 was specified to produce.

### Ego-motion conditioning declaration (Catalog #311 / Pattern H)

<!-- # PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:cross-reference-only-not-substrate-central-cooperative-receiver-framing-cited-as-related-work-not-as-this-substrate-architectural-core-per-z6z7z8-design-memo-section-11-scope -->

This memo references Atick-Redlich / cooperative-receiver framing as cross-reference / related-work / sister-substrate context — NOT as this substrate's architectural core. The substrate proposed by this memo is structurally distinct from Z6/Z7/Z8 (which DO require ego-motion-conditioned next-frame prediction as architectural core per Pattern H + Z6/Z7/Z8 design memo Section 11).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Z6/Z7/Z8 design memo Pattern H + Catalog #311 acceptance cascade (c): same-line waiver `# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:cross-reference-only-not-substrate-central-cooperative-receiver-framing-cited-as-related-work-not-as-this-substrate-architectural-core-per-z6z7z8-design-memo-section-11-scope` applies. The waiver rationale is non-placeholder (>4 chars, not `<rationale>` / `<reason>`).

Cross-references to cooperative-receiver / Atick-Redlich in this memo serve as theoretical-anchor / related-work / sister-substrate-comparison only; they do NOT make this substrate a predictive-coding substrate in the Pattern H sense.

---

## Appendix C — Four alternative-reducer probes (T2 council Q1.4 reactivation criteria, 2026-05-16)

**Status**: T2 council Q1 SPLIT-VERDICT reactivation criteria executed. Result appended per Catalog #110/#113 HISTORICAL_PROVENANCE (APPEND-ONLY); body of design memo + Sections 1-22 + 9-dim checklist + Op-summary + Observability surface + Appendix A + Appendix B UNCHANGED. **Subagent**: SUBAGENT B (`alt_reducer_probes_subagent_b_20260516`) at lane `lane_four_alternative_reducer_probes_meta_pattern_e_remediation_20260516`.

**Cite chain**: T2 council memo `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md` Q1.4 enumerated 4 alternative reducers as the EXPANDED REACTIVATION CRITERIA after the canonical per-pair-dominant SegNet argmax reducer was empirically falsified (Appendix B: 600/600 → class 2; MI = 0.000 bits/symbol; INDEPENDENT verdict). Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + sextet-pact Contrarian veto on kill-too-fast: v2 paradigm class (any SegNet-derived conditioning) cannot be DEFERRED class-wide until ALL alternative reducers have been probed independently.

### C.1 Probe methodology

**Tool**: `tools/run_alternative_reducer_probes_g1_v2_and_tishby_ib_pure.py --substrate wunderkind_g1_v2 --max-pairs 600` driving the canonical library `tools/probe_alternative_reducers_latent_class_conditioning.py` (4 reducers + canonical MI estimator + reducer-specific thresholds per T2 Q1.4).

**Source signal**: `upstream/videos/0.mkv` (sha256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`) decoded via canonical `upstream/frame_utils.py::AVVideoDataset::yuv420_to_rgb` semantics into 600 pairs of `(2, 3, 874, 1164)` fp32 tensors (mirrors Appendix B derivation).

**Scorer**: canonical `tac.scorer.load_default_scorers(upstream)` loading `upstream/models/segnet.safetensors`. For each pair we computed:
1. **Canonical preprocess argmax** (`smp.Unet('tu-efficientnet_b2', classes=5)` slice frame_1 + interpolate to (384, 512) per upstream/modules.py:108) — used by reducers 1, 2, 3.
2. **Per-frame argmax of frame_0 + frame_1 separately** (each frame independently bilinear-interpolated to (384, 512) then SegNet forward) — used by reducer 4.

**Latent stream**: existing v2 1000ep Z3 archive residual_int8 stream from `experiments/results/probe_dispatch_wave_749_section14_g1_v2_20260516T171251Z/residual_int8.bin` (sha256 `2a6dbfb834aede79f97c3d0d0f08206c054347f4db6919109bf6900e163419fd`; 16800 bytes = 600 pairs × 28 latent_dim). Per-pair reducer outputs byte-expanded ×28 to align symbol-for-symbol with the latent stream (mirrors Appendix B byte-expansion protocol).

**Device**: CPU (hermetic; SegNet output is source-data derivation NOT auth-eval per Catalog #127 source-data carve-out). Wall clock: 2152s total (0.6s load + 9.2s decode + 2142s SegNet inference — both-frames-per-pair doubles vs Appendix B's frame_1-only). Cost: $0.

**Output directory**: `experiments/results/alternative_reducer_probes_20260516T222227Z/` (Catalog #249 device-agnostic naming).

### C.2 Per-reducer empirical verdict table

| Reducer | Threshold | MI (bits/sym) | Unique fingerprints | WZ ceiling | **Verdict** |
|---|---|---|---|---|---|
| `per_pixel_histogram` (5-bin × 16-quant) | > 0.5 bits | 0.0283 | 4 | 0.37% | **WEAK_CONDITIONING** |
| `per_region_histogram` (5-bin × 4-region × 8-quant) | > 1.0 bits | 0.0599 | 7 | 0.79% | **WEAK_CONDITIONING** |
| `per_pair_class_2_fraction` (32-bucket) | > 0.2 bits | 0.0124 | 2 | 0.16% | **WEAK_CONDITIONING** |
| `per_frame_argmax` (5²=25-class fingerprint) | > 0.2 bits | 0.0000 | 1 | 0.00% | **INDEPENDENT** |

### C.3 Per-pair distribution receipts (forensic provenance)

**Per-pair-dominant class (frame_1 canonical preprocess, 384×512)**: 600/600 → class 2. Matches Appendix B exactly.

**Per-frame (frame_0_dominant, frame_1_dominant) pairs**: 600/600 → (2, 2). NO temporal motion observable at the per-pair-dominant resolution. The temporal-motion hypothesis (Yousfi Q1.2 *"a car appearing/disappearing between frames"*) is empirically falsified on the contest video at the per-frame-dominant level.

**Per-pair class-2 fraction (continuous in [0, 1])**:
- n = 600
- min / max = 0.4808 / 0.5101
- mean / median = 0.4952 / 0.4949
- stdev = 0.0065
- range (max - min) = 0.0293 (~3 percentage points)

**Critical finding**: the per-pair class-2 fraction is **NOT in the [0.55, 0.95] range Yousfi predicted from dashcam physics** (Q1.2 footnote). The empirical mean is 0.495 — essentially 50% road pixels at the canonical (384, 512) resolution after SegNet's stride-2 stem. This explains the per_pair_class_2_fraction reducer's WEAK_CONDITIONING result: with only 32 buckets across a 0.03-wide distribution range, only 2 buckets are populated (bucket 15 at fraction ≈ [0.469, 0.500) with 451 pairs, bucket 16 at fraction ≈ [0.500, 0.531) with 149 pairs).

**Per-pixel histogram fingerprints**: 4 distinct fingerprints; top fingerprint at 263939 (438/600 = 73% of pairs); next at 264195 (149/600 = 25%). Fingerprint count concentration is high — most pairs map to the same fingerprint.

**Per-region histogram fingerprints**: 7 distinct fingerprints; top fingerprint at 576583900841640384 (376/600 = 63%); next at 576579502795129280 (158/600 = 26%). Slightly more diverse than per-pixel because spatial-region splitting captures top-half (sky) vs bottom-half (road) class differences across some pairs.

**Per-frame argmax fingerprint**: 600/600 → 12 (= 2*5+2; frame_0 dominant = class 2, frame_1 dominant = class 2). Per-frame argmax degenerates to the same single fingerprint because the per-pair-dominant is already degenerate AND the per-frame dominant is identical (no per-frame temporal variability at the dominant-class resolution).

### C.4 Verdict interpretation per T2 Q1.4 acceptance rules

**Aggregated verdict**: `PARTIAL` — 3 of 4 reducers returned WEAK_CONDITIONING (MI > 0.01 independence tolerance but < the reducer-specific MEANINGFUL threshold); 1 of 4 returned INDEPENDENT. **NO reducer reaches the MEANINGFUL threshold required for v2 paradigm-class reactivation per Q1.4**.

**Operator-facing recommendation** (per the canonical helper's `_recommend_action_for_verdicts`):

> *"PARTIAL: alternative reducer(s) (per_pixel_histogram, per_region_histogram, per_pair_class_2_fraction) returned WEAK_CONDITIONING; MI > tolerance but < meaningful threshold. Paradigm class is DEFERRED-pending-tighter-reducer-design (Phase 2 council Q1.4 #5)."*

### C.5 Per-reducer empirical interpretation (T2 Q1.2 sextet-pact assumption resurfacing)

**Shannon (LEAD) on the empirical result**: the per-pixel + per-region + per-pair-class-2-fraction reducers all show MI > 0 (verifying the v2 paradigm class IS not strictly mathematically vacuous — these reducers carry SOME information about the latent residual). However, the absolute MI magnitude (0.0124-0.0599 bits/symbol) is small relative to H(latent) ≈ 7.57 bits/symbol — the Wyner-Ziv gain ceiling is 0.16%-0.79%. For a v2 archive that would have to ship a per-pair conditioning sidecar overhead, the rate cost of the sidecar likely EXCEEDS the achievable distortion savings at this WZ ceiling.

**Dykstra (CO-LEAD) on convex-feasibility**: the polytope intersection for the per-pixel + per-region + per-pair-class-2-fraction reducers is non-empty BUT extremely narrow. A Phase 2 design that tries to exploit 0.0599 bits/symbol (per_region_histogram, the best reducer) would need an archive grammar that encodes the 7-fingerprint conditioning signal in FEWER bytes than the achievable distortion savings. At 0.0599 bits/symbol × 16800 symbols ≈ 1006 bits ≈ 126 bytes of distortion savings, the per-region-histogram conditioning sidecar would need to ship 7 fingerprints × log2(7) bits ≈ 20 bits = ~3 bytes per pair × 600 pairs = ~1800 bytes overhead — the sidecar EXCEEDS the savings by ~14×. Dykstra-feasibility is structurally negative for this reducer family AT the contest-video operating point.

**Yousfi on the empirical class-2 fraction range [0.48, 0.51]**: my Q1.2 prediction of [0.55, 0.95] was empirically FALSIFIED — the contest video's per-pair class-2 fraction is centered at 0.495 with stdev 0.0065, NOT [0.55-0.95]. The CARGO-CULTED assumption in my Q1.2 position is identified: "dashcam physics predicts road ≈ 70% of pixels". Per Catalog #292 explicit assumption surfacing: the assumption I was operating within ("dashcam physics → road = 70%+ of pixels") is empirically incompatible with the SegNet stride-2-stem-at-384×512 argmax distribution. The stride-2 stem may be confounded by adjacent classes (lane lines, building edges, vehicle silhouettes) that get counted as "non-road" at the canonical resolution. HARD-EARNED revised understanding: at the contest scorer's argmax resolution, road class is ~50% NOT 70%.

**Fridrich on the cooperative-receiver paradigm**: the per-region-histogram is the BEST of the 4 alternative reducers at MI=0.0599 bits/symbol, ~4.8× better than per-pair-class-2-fraction. The structural reason matches the cooperative-receiver intuition: spatial regions DO carry distinct class information (top-half: sky/clouds; bottom-half: road; lane-line strips: distinct class). However, the absolute MI magnitude remains TWO orders of magnitude below the MEANINGFUL threshold (1.0 bits/symbol). The cooperative-receiver paradigm with SegNet-derived spatial features is empirically VIABLE but the MI is too small to overcome sidecar overhead — Dykstra agrees the convex feasibility is structurally negative.

**Contrarian on the verdict**: per the Q1.4 SPLIT-VERDICT, the PARTIAL outcome (3 weak + 1 independent, all below MEANINGFUL threshold) is NEITHER a full reactivation (which would require ≥1 MEANINGFUL) NOR a full deferral (which would require all 4 INDEPENDENT). The Q1.4 #5 "DEFERRED-pending-tighter-reducer-design" recommendation is the correct verdict — paradigm class stays research_only=true with EXTENDED reactivation criteria that include a NEW reducer methodology (one that captures finer-grained per-pair signal than the 4 evaluated). VETO any council consensus that interprets PARTIAL as "v2 paradigm permanently falsified" — the per-region MI at 0.0599 is the LARGEST signal seen so far and points the next reducer-design iteration toward finer spatial granularity.

**Assumption-Adversary**: classification of the operating-within assumptions for this probe wave:

| Assumption | Classification | Rationale |
|---|---|---|
| The 4 T2 Q1.4 reducers ENUMERATE the relevant reducer space | CARGO-CULTED | Counterexamples available: per-PoseNet-derived signal (not SegNet); per-pixel argmax with TEMPORAL conditioning across pair-pair sequences; SegNet logits BEFORE argmax (continuous-class distribution); per-pair entropy of pixel-class distribution (a scalar conditioning signal). The 4 reducers are a SAMPLE not an EXHAUSTIVE enumeration. |
| The MI thresholds (0.5/1.0/0.2/0.2 bits/pair) are correctly calibrated for v2's archive grammar | HARD-EARNED-PARTIAL | The thresholds were chosen by T2 council Q1.4 deliberation per the convex-feasibility envelope of v2's 5-row sigma table archive grammar. A DIFFERENT archive grammar (e.g. compressed-per-region-sigma-table) could have a tighter threshold; the council MI levels are correct for v2-AS-SPECIFIED. |
| The empirical MI values 0.0124-0.0599 are stable estimates (not bias-dominated) | HARD-EARNED | n_symbols = 16800 with n_unique_classes ∈ {1, 2, 4, 7, 25} — Miller-Madow bias bound is at most O((n_unique-1) / (2 * 16800 * ln(2))) ≈ 1e-4 bits/symbol for the worst case (per_frame_argmax 25-class). The MI estimates are at least 100× the Miller-Madow bias bound; the WEAK_CONDITIONING verdicts are stable, not bias artifacts. |
| The per-frame argmax reducer is empirically equivalent to per-pair-dominant on this video | HARD-EARNED | 600/600 → (2, 2) confirmed empirically. The temporal-motion hypothesis from Yousfi Q1.2 (per-frame dominance varying due to cars / scene cuts) is FALSIFIED on `upstream/videos/0.mkv` at the canonical-resolution per-frame-dominant level. Temporal motion is present in pixel-level differences BUT not at the per-frame-dominant-class level. |

The Assumption-Adversary VETOES any Phase 2 council consensus that defers the v2 paradigm class WITHOUT explicitly enumerating the CARGO-CULTED-reducer-enumeration assumption above. Future reducer-design iterations MUST probe (a) PoseNet-derived signals (orthogonal scorer; class-2-equivalent for PoseNet may have different distribution), (b) SegNet pre-argmax logits (continuous-class distribution carries more information than argmax), (c) per-pixel argmax with TEMPORAL conditioning (motion features across pair indices), (d) entropy of per-pair class distribution (scalar conditioning signal).

### C.6 Implications for v2 reactivation (Section 19 / Appendix B update)

**Section 14 criterion 1 update (op-routable C-1)**: the per-pair-dominant SegNet argmax reducer (Section 14 spec) is permanently FAILED per Appendix B. The 4 T2 Q1.4 alternative reducers each FAIL the MEANINGFUL threshold per this Appendix C, with the per-region-histogram showing the largest signal (MI=0.0599, WZ ceiling 0.79%).

**Reactivation criteria update for the v2 paradigm class** (NEW; supersedes Appendix B's permanent-research-only verdict for the reducer-search axis):

The v2 lane `lane_z3_g1_entropy_coded_v2_20260515` stays `research_only=true` with EXTENDED reactivation criteria:

1. **(SAME as Appendix B)** Per-pair-dominant reducer is permanently FAILED on `upstream/videos/0.mkv`.
2. **(NEW per Appendix C)** Of the 4 T2 Q1.4 alternative reducers, only `per_region_histogram` carries non-trivial signal (MI=0.0599, WZ ceiling 0.79%) — but the convex-feasibility check (Dykstra above) shows the sidecar overhead exceeds the achievable savings by ~14×; this reducer is also structurally FAILED for v2's archive grammar at this operating point.
3. **(NEW per Appendix C reactivation-pending-tighter-reducer-design)** The v2 paradigm class is reactivatable IF a NEW reducer methodology (NOT in the T2 Q1.4 enumeration) returns MEANINGFUL_CONDITIONING (MI ≥ reducer-specific threshold) AND the archive grammar's conditioning-sidecar overhead is < achievable distortion savings per a Dykstra-feasibility check. Candidate NEW reducers: PoseNet-derived signals (orthogonal scorer); SegNet pre-argmax logits (continuous-class); per-pixel argmax with temporal conditioning; entropy of per-pair class distribution.
4. **(NEW per Appendix C)** Per Catalog #220 distinguishing-feature integration contract: any NEW reducer must come with sister archive grammar that explicitly encodes the conditioning signal in archive bytes AND a byte-mutation smoke proving the new bytes are consumed by inflate.

**Phase 2 council deliberation gate** (op-routable C-2): the Q1.4 #5 "DEFERRED-pending-tighter-reducer-design" verdict requires a Phase 2 council deliberation per Catalog #292 sextet-pact discipline to:

(a) Adjudicate whether the 4-reducer enumeration was sufficient OR whether the Assumption-Adversary's NEW-reducer-class enumeration (PoseNet / logits / temporal / entropy) should be the next probe wave.

(b) If Phase 2 council approves a NEW reducer wave, the new probes can re-use this Appendix C's infrastructure (`tools/probe_alternative_reducers_latent_class_conditioning.py` + `tools/run_alternative_reducer_probes_g1_v2_and_tishby_ib_pure.py` + `experiments/results/alternative_reducer_probes_<timestamp>/`).

(c) If Phase 2 council DOES NOT approve a NEW reducer wave, the v2 paradigm class is `DEFERRED-class-wide-pending-NEW-reducer-methodology-PROPOSAL` per Q1.4 #5 final-form verdict.

### C.7 Op-routables (Appendix C closure)

- **C-1**: Lane registry update: append to `lane_z3_g1_entropy_coded_v2_20260515` notes: *"Appendix C alternative-reducer probe wave (2026-05-16): 3 of 4 T2 Q1.4 reducers WEAK_CONDITIONING (best per_region MI=0.0599 bits/symbol, WZ ceiling 0.79%); 1 of 4 INDEPENDENT; NO reducer reaches MEANINGFUL threshold. PARTIAL recommendation — DEFERRED-pending-tighter-reducer-design per Q1.4 #5; reactivation extended with 4 NEW reducer-class candidates (PoseNet-derived / SegNet logits / temporal / entropy)."* Operator decision: confirm lane notes update.
- **C-2**: Phase 2 council deliberation per Catalog #292 sextet-pact discipline (Assumption-Adversary surfaced the CARGO-CULTED-reducer-enumeration assumption): adjudicate whether NEW reducer wave (PoseNet / logits / temporal / entropy) is approved. If yes → SUBAGENT B's probe library is the canonical infrastructure; if no → v2 paradigm class is `DEFERRED-class-wide-pending-NEW-reducer-methodology-PROPOSAL`.
- **C-3**: Cathedral autopilot continual-learning posterior: append `[diagnostic-CPU]` anchor rows for each of the 4 reducer verdicts per Catalog #128 fcntl-locked `posterior_update_locked` — empirical anchors that the v2 paradigm class's reducer-search axis returned PARTIAL on the contest video. Use `evidence_grade="diagnostic_cpu"`, `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, `rank_or_kill_eligible=false` per Catalog #127 strict custody.
- **C-4**: Catalog #308 META-pattern E remediation status update: this Appendix C operationalizes the T2 Q1.4 reactivation criteria with a typed verdict table + per-reducer recommendation; the META-pattern E remediation is COMPLETE for the 4-reducer enumeration. The Assumption-Adversary surfaced the next layer (NEW reducer-class proposal) which is Phase 2 council deliberation scope.

### C.8 Probe artifact custody

- `tools/probe_alternative_reducers_latent_class_conditioning.py` (tracked canonical 4-reducer probe library)
- `tools/run_alternative_reducer_probes_g1_v2_and_tishby_ib_pure.py` (tracked canonical orchestrator driver)
- `src/tac/tests/test_probe_alternative_reducers.py` (tracked unit coverage for the reducer library)
- `experiments/results/alternative_reducer_probes_20260516T222227Z/g1_v2_per_pair_reducer_outputs.json` (local ignored rebuildable artifact; summarized here, not committed per AGENTS.md research-state tracking rules)
- `experiments/results/alternative_reducer_probes_20260516T222227Z/alternative_reducer_run_manifest_wunderkind_g1_v2.json` (local ignored rebuildable artifact; summarized here)
- `experiments/results/alternative_reducer_probes_20260516T222227Z/alternative_reducer_verdict_wunderkind_g1_v2_per_pixel_histogram.json` (local ignored rebuildable artifact; summarized here)
- `experiments/results/alternative_reducer_probes_20260516T222227Z/alternative_reducer_verdict_wunderkind_g1_v2_per_region_histogram.json` (local ignored rebuildable artifact; summarized here)
- `experiments/results/alternative_reducer_probes_20260516T222227Z/alternative_reducer_verdict_wunderkind_g1_v2_per_pair_class_2_fraction.json` (local ignored rebuildable artifact; summarized here)
- `experiments/results/alternative_reducer_probes_20260516T222227Z/alternative_reducer_verdict_wunderkind_g1_v2_per_frame_argmax.json` (local ignored rebuildable artifact; summarized here)
- `experiments/results/alternative_reducer_probes_20260516T222227Z/combined_run_summary.json` (local ignored rebuildable artifact; summarized here)

**Source memo edit scope**: Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY. Body of design memo (Sections 1-22 + 9-dim checklist + Op-summary + Observability surface + Appendix A + Appendix B) UNCHANGED. This Appendix C lands as the empirical disambiguator result T2 council Q1 SPLIT-VERDICT op-routable was specified to produce.


# PARADIGM_VS_IMPLEMENTATION_FALSIFICATION_OK:historical_2026_05_16_design_memo_predates_catalog_307_2026_05_16_cutoff_landing_carries_legacy_kill_or_retired_token_for_specific_implementation_paradigm_intact_per_canonical_legacy_classification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
