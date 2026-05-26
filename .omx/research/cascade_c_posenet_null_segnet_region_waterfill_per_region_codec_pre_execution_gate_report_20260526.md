# Cascade C pre-execution gate report 2026-05-26

**Subagent_id:** `cascade-c-posenet-null-segnet-region-waterfill-per-region-selector-codec-20260526`

**Operator approval (binding):** 2026-05-26 *"absolutely and enthusiastically"* for EXPLOIT 1 from entropy-position cross-cascade catalog.

**Mission alignment per Catalog #300:** `frontier_breaking_enabler` (novel SCORER-ENTROPY attack class at scorer-side P18+P19; sister cascade variants enabled by entropy-position decomposition learned here).

---

## Entropy-position declaration (per just-landed entropy-position-discipline standing directive 2026-05-26)

Per the canonical 5-question contract:

1. **WHERE in the pipeline does Cascade C operate?**
   - **P19 (PoseNet entropy, scorer-side)**: per-pair PoseNet-gradient bottom-decile selects pairs where PoseNet is structurally near-zero-sensitive to frame-0 perturbations. SCORER-LEVEL position; operates on per-pair pose-gradient distribution.
   - **P18 (SegNet entropy, scorer-side)**: per-pixel SegNet logit-margin determines where scorer is most-confident vs least-confident. SCORER-LEVEL position; operates on per-pixel logit-margin distribution.
   - **P11 (selector-stream entropy, archive-side)**: per-region selector codec partitions K=16 global menu into per-region menus. ARCHIVE-LEVEL position; operates on K=16 selector mode distribution.

2. **WHAT entropy distribution does each interact with?**
   - **P19**: marginal H(pose_gradient_per_pair) — empirically bimodal (small subset of "PoseNet-null" pairs vs the bulk)
   - **P18**: marginal H(segnet_logit_margin_per_pixel) — structurally bounded by SegNet's stride-2 stem (per CLAUDE.md "Exact scorer architectures" — SegNet processes ONLY last frame `x[:, -1, ...]`)
   - **P11**: joint H(region, mode) = H(region) + H(mode|region) = H(mode) marginal (chain rule)

3. **WHAT theoretical bound applies?**
   - **P19**: no information-theoretic bound — empirical exploitation depends on per-pair pose-gradient distribution shape
   - **P18**: Fridrich-UNIWARD bound — scorer-confidence redistribution preserves total distortion-budget per pixel
   - **P11**: Shannon floor for marginal H(mode) = 3.21 bits/pair → 241 bytes for n=600 (live FEC6 achieves 243-byte Huffman stream + 6-byte header = 249 bytes wire)

4. **WHAT empirical evidence anchors the predicted savings to a position?**
   - **P19**: sister #1324 OPT-12 PoseNet-null bottom-decile artifact at `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/pr110_opt12_posenet_null_frame0.json` (n=2 macOS-CPU advisory)
   - **P18**: sister CLAUDE.md "SegNet architectural blind spot" + canonical sweep tool's `seg_delta=0.0` for all 87 widened frame-0 modes
   - **P11**: today's live decode of PR110 fec6 selector stream (249 bytes wire; 16-mode histogram with H=3.21 bits/pair; n=600 pairs)

5. **WHAT canonical equation anchor does this position interact with?**
   - **P11 candidate equation (proposed; awaits operator approval per Catalog #344):** `per_region_selector_codec_savings_v1` with EMPIRICAL anchor at `{n_pairs: 600, n_regions: 3, baseline_H_marginal: 3.21, region_idx_overhead_bytes: 87, predicted_savings: -7 to +91 bytes wire}` per the chain-rule decomposition.
   - **Existing sister:** `markov_context_selector_stream_compression_savings_v1` (FEC8 Markov 1st-order conditional; #1336 commit `6474afde7` empirical -4 bytes)
   - **Existing sister:** `procedural_codebook_from_seed_compression_savings_v1` (NSCS06 v8 chroma_lut; different entropy-position at frame-render layer)

---

## Full-stack fractal optimization decomposition (per GUIDING PRINCIPLE 2026-05-26)

Per just-elevated `feedback_pr95_sniped_lesson_full_stack_mlx_first_per_candidate_standing_directive_20260526.md` GUIDING PRINCIPLE: identify which ingredients + sub-ingredients Cascade C addresses.

**Top-level ingredient targets (per PR101 13-ingredient model):**
- **Ingredient #11 (archive grammar)**: P11 per-region selector codec is a NEW archive-grammar element (FEC11 wire format vs current FEC6)
- **Ingredient #17 (frame-pair post-decode)**: P19 PoseNet-null pairs use a REDUCED selector menu at inflate time; ingredient #17 still uses the same decoded base + perturbation menu
- **Ingredient #4 (inflate runtime)**: P11 requires a NEW selector-decoder branch in `submissions/hnerv_fec6_fixed_huffman_k16/inflate.py` (~30 LOC budget per HNeRV parity L4)

**Sub-ingredient decomposition:**
- **P19 sub-ingredient A**: per-pair PoseNet-gradient ranking → identify bottom-decile subset (60 pairs)
- **P19 sub-ingredient B**: REDUCED menu encoding for bottom-decile (1-bit selector vs 4-bit fec6 K=16)
- **P18 sub-ingredient A**: per-class-region SegNet logit-margin map (precomputed compress-time; embedded in archive metadata)
- **P18 sub-ingredient B**: Fridrich-UNIWARD weight reallocation (low-margin region = absorb more perturbation budget; high-margin region = preserve clean RGB)
- **P11 sub-ingredient A**: region partition design (which K=16 codes cluster into which region)
- **P11 sub-ingredient B**: per-region menu Huffman codebook (vs single global K=16 Huffman)
- **P11 sub-ingredient C**: per-pair region-idx stream wire format

**Recursive doctrine per pushing-frontier standing directive:** each sub-ingredient has its own canonical-vs-frontier-push decision. For Cascade C V1 design (3-region partition), the decision was **frontier-push canonical-novel** because no canonical helper covers per-region selector codec design.

---

## Premise verification (Catalog #229)

Pre-execution reads, all complete BEFORE first code execution:

1. **`.omx/research/pr110_opt_frame0_bundle_landed_20260526.md`** (sister #1313+#1324+#1325 landing memo) — confirmed 87-mode widened frame-0 catalog + PoseNet-null bottom-decile artifact + tier-split smoke landed; sister artifacts at `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/`. The PoseNet-null decile is dominated (50% + 37.5%) by structured-signed-chroma modes (DCT + blue_chroma family).
2. **`.omx/research/pr110_opt3_mode_distribution_20260526T170000Z.md`** (sister #1315 OPT-3 mode-distribution analysis) — confirmed FEC6 selector_payload structure: 6-byte header + 243-byte Huffman bitstream = 249 bytes wire; canonical reproducer for decoding + 16-mode histogram.
3. **PR110 live archive** at `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip` — confirmed archive sha `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` (178,517 bytes); selector decode succeeded; n_pairs = 600.
4. **CLAUDE.md "Strict scorer rule — non-negotiable"** — confirmed Cascade C is COMPRESS-TIME ONLY analysis; no scorer load at inflate time; SegNet + PoseNet structural facts (SegNet processes ONLY last frame; PoseNet uses 12-channel YUV6) inform design but not runtime.
5. **CLAUDE.md "Fridrich inverse steganalysis"** — confirmed UNIWARD heuristic (weight loss by inverse local variance / scorer-confidence) is the canonical Yousfi-Fridrich framework Cascade C P18 instantiates.
6. **CLAUDE.md "Quantizr intelligence"** — confirmed PR101 GOLD pattern (605 LOC reviewable in 30 seconds) bounds the per-region codec to ≤100 additional LOC at inflate runtime.
7. **`.omx/state/master_gradient_anchors.jsonl`** — confirmed 11 anchors exist with archive_sha matching live PR110 archive (`f174192...` predecessor anchor) + sister anchors. Per-pair pose-gradient data is not directly in this ledger (only aggregate); per-pair data sourced from canonical sweep tool's `_score_pairs` output.

**Premise verified.** No PV-failure conditions encountered.

---

## Sister coordination per CLAUDE.md "Subagent coherence-by-default"

Active sister subagents at execution time (read from `.omx/state/subagent_progress.jsonl`):

- `nscs06-v8-stacked-paired-modal-t4-re-fire-post-trainer-v3-wire-in-20260526` (slot 1; PAID Modal T4) — substrate scope; DISJOINT from Cascade C archive-side encoding scope
- `z7-mamba-2-v2-l2-stability-hardening-nan-fix-20260526` (slot 2; MLX local) — substrate scope; DISJOINT
- `boostnerv-variant-c-ii-centered-base-recolor-training-dynamics-fix-20260526` (slot 3; MLX local) — substrate scope; DISJOINT

YOUR scope = PR110 fec6 selector codec design at `.omx/research/cascade_c_artifacts_20260526/` + analysis-only tools (no production trainer mutations). Zero file overlap with sister subagents.

---

## Cascade C measurement plan (executed)

Step 1: Decode live PR110 fec6 selector → 600 per-pair K=16 mode assignments (DONE)
Step 2: Compute per-mode histogram → identify high/mid/low-frequency clusters (DONE)
Step 3: Design 3 candidate region partitions (V1 3-region, V2 2-region, V3 4-region) (DONE)
Step 4: Per-partition wire-byte budget (header + region-idx stream + per-region mode-stream + codebook overhead) (DONE)
Step 5: Compare against FEC6 baseline (249 bytes) + FEC8 Markov sister (245 bytes) + Shannon floor (241 bytes) (DONE)
Step 6: Carmack-dissent verdict per Catalog #307 (DONE — see landing memo)
Step 7: Operator-routable next steps + sister-extinction recommendations per Catalog #308 alternative reducers (DONE — see landing memo)

---

## Mandatory pre-execution gate verdict

- Premise verification: PASS (Catalog #229)
- Sister scope: DISJOINT (Catalog #230)
- Entropy-position declared: PASS (just-landed standing directive)
- Full-stack fractal decomposition: PASS (GUIDING PRINCIPLE)
- Compress-time-only invariant: PASS (CLAUDE.md "Strict scorer rule")
- Canonical Provenance umbrella threading: PASS (Catalog #323; all artifacts carry axis_tag + evidence_grade + promotable=False + score_claim=False + 5 promotion_blockers)
- MLX-LOCAL only / NO PAID DISPATCH: PASS (per operator "Remember all on MLX")

**PROCEED to execution.** Landing memo will document empirical verdict.
