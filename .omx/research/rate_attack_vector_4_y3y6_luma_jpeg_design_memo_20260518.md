---
schema: pact_design_memo_v1
memo_id: rate_attack_vector_4_y3y6_luma_jpeg_design_memo_20260518
review_date: "2026-05-18"
lane_id: lane_rate_attack_y3y6_luma_jpeg_substrate_20260518
parent_master_memo: rate_attack_43_vectors_meta_paradigm_deep_research_20260518
meta_paradigm_anchor: structural_information_not_shipped_meta_paradigm_unification_20260518
vector_ids: ["Y3", "Y6"]
vector_name: "Luma-only encoding + JPEG quantization-table steganography (Quantizr PR101 + Fridrich PhD canonical)"
horizon_class: frontier_breaking
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
research_only: true
write_scope: ".omx/research only"
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU]"
predicted_delta_band_contest_cpu: "[-0.015, -0.004]"
council_tier_assignment: T3_full_grand_council
target_modes:
  - contest_exact_eval
  - contest_generalized
deployment_target: t4_contest_runtime
hardware_substrate: linux_x86_64_t4
---

# TOP-4 Design Memo — Vectors Y3+Y6: Luma-Only + JPEG Quantization-Table Steganography

**Master memo**: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
**META-paradigm**: SINS
**Lane**: `lane_rate_attack_y3y6_luma_jpeg_substrate_20260518` L0

## 0. Executive Summary

**HARD-EARNED canonical anchors**:
- Y3 (luma-only): Quantizr PR101 GOLD-MEDAL (0.193) uses GRAYSCALE-LUT analog mask paradigm = LUMA-ONLY encoding for masks. 4× compression vs RGB per CLAUDE.md "Quantizr intelligence". **EMPIRICALLY PROVEN.**
- Y6 (JPEG quant-table): Fridrich PhD thesis (Binghamton DDE Lab; DDELab/deepsteganalysis canonical repo per CLAUDE.md "Yousfi's repos"). JPEG quantization-table steganography is the CANONICAL Fridrich work.

**Y3+Y6 composition**: encode masks via Quantizr-style grayscale-LUT (Y3); additionally embed bits in JPEG-coefficient parity within YUV blocks (Y6). Compounding savings.

**Predicted ΔS**: [-0.015, -0.004] [contest-CPU].

**Council verdict**: T3 PROCEED_WITH_REVISIONS (Fridrich binding: cite his canonical PhD thesis + verify against actual upstream JPEG-aware preprocessing path).

## 1. Canonical-vs-unique Decision Per Layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Y3: Quantizr-style grayscale-LUT | ADOPT_CANONICAL | Quantizr PR101 is gold-medal canonical |
| Y6: Fridrich UNIWARD/HUGO embedding | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | NEW canonical helper at `tac.substrates.rate_attack_y6_jpeg_quant_table_steganography`; Fridrich's MATLAB codes need Python translation |
| YUV color-space conversion | ADOPT_CANONICAL (`tac.differentiable_eval_roundtrip::rgb_to_yuv6`) | Pinned upstream helper |
| JPEG coefficient quantization | **FORK_BECAUSE_SUBSTRATE_OPTIMAL** | NEW: PIL/Pillow JPEG quantization with custom quant tables |
| Archive grammar | FORK_BECAUSE_SUBSTRATE_OPTIMAL | NEW: Y3 luma + Y6 JPEG-coefficient stream |
| Inflate runtime | ADOPT_CANONICAL + extension | ~60 LOC for Y3 luma reconstruction + ~40 LOC for Y6 JPEG decode |
| Score-aware loss | ADOPT_CANONICAL (`score_pair_components`) | Existing canonical helper |
| EMA + eval_roundtrip + Tier-1/2 engineering | ADOPT_CANONICAL | All existing canonical disciplines |

## 2. 9-Dim Checklist (per Catalog #294) — SUMMARIZED

- Dim 1 UNIQUENESS: Y3+Y6 is class-shift composition of two YUV-native exploits
- Dim 2 BEAUTY: ~300 LOC substrate; ~100 LOC inflate extension; reviewable
- Dim 3 DISTINCTNESS: Y3 luma-only is PROVEN; Y6 JPEG quant-table is NEW composition
- Dim 4 RIGOR: Premise verification + Fridrich PhD citation + Quantizr PR101 anchor
- Dim 5 OPTIMIZATION: Quantizr-canonical 88K-param SegMap baseline; PIL/Pillow JPEG with custom quant tables
- Dim 6 COMPOSABILITY: Y3+Y6 SUB (sub-additive); composes ORTHO with F1, SUB with G1
- Dim 7 REPRODUCIBILITY: PIL JPEG is deterministic; luma-LUT is deterministic
- Dim 8 OPTIMIZATION: ~5ms per frame Y3 luma decode; ~10ms per frame Y6 JPEG decode
- Dim 9 OPTIMAL SCORE: predicted [0.177, 0.188] [contest-CPU] = 1.5-7.8% improvement

## 3. Observability Surface (per Catalog #305)

1. Inspectable: per-frame luma channel + JPEG quantization-table dump-able
2. Decomposable: per-mask Y3 byte cost; per-block Y6 bit allocation
3. Diff-able: pre/post grayscale-LUT byte mapping; pre/post JPEG-coeff parity
4. Queryable: per-frame luma entropy; per-block JPEG quant table
5. Cite-able: (archive_sha, luma_lut_sha, quant_table_sha)
6. Counterfactual-able: modify one JPEG coefficient parity; observe bit recovery

## 4. Cargo-Cult Audit (per Catalog #303)

| Assumption | Verdict |
|---|---|
| Quantizr grayscale-LUT is 4× compression | **HARD-EARNED** (PR101 gold-medal empirical) |
| Y6 JPEG quant-table steganography is exploit-compatible with contest scorer | **CARGO-CULTED** (Fridrich's PhD work targets steganalysis detection; here we're using it for compression; needs PROBE that PoseNet/SegNet don't detect the JPEG-coefficient perturbations) |
| YUV color space is the canonical pose input | **HARD-EARNED-VERIFIED** (upstream/modules.py:73 `rgb_to_yuv6`) |
| Custom JPEG quant tables don't break PIL/Pillow decode | **HARD-EARNED** (PIL canonical supports custom tables) |
| Y3+Y6 composition is sub-additive (α 0.5-1.5) | **CARGO-CULTED** (needs paired-comparison empirical) |

## 5. Dykstra-Feasibility (per Catalog #296)

- (R) Y3 saves ~25% of mask bytes; Y6 saves ~10% of frame bytes
- (S) Y3 luma preserves SegNet argmax (PR101 proven); Y6 JPEG perturbations on Fridrich-low-cost coefficients minimize SegNet impact
- (P) YUV-native encoding preserves PoseNet pose (canonical pipeline)
- (L) ~100 LOC inflate extension; PR101 inflate.py is 150 → total ~250 (REQUIRES HNeRV parity L4 waiver OR helper extraction)
- (D) PIL JPEG + luma-LUT are deterministic across CPU/CUDA

## 6. Predicted Band (per Catalog #324)

Derivation:
- Y3: ~25% of mask bytes saved = ~75 KiB × 0.25 = ~18.7 KiB saved per archive = -0.0125 ΔS
- Y6: additional ~10% of remaining frame bytes = ~22 KiB × 0.1 = ~2.2 KiB = -0.0015 ΔS
- Composition (sub-additive α=0.7): -0.014 to -0.005

Range: [-0.015, -0.004]

## 7. 6-Hook Wire-In (per Catalog #125)

All 6 ACTIVE per master memo §11 with Y3+Y6-specific producer/consumer wiring.

## 8. Routing Directive Sketch

Full directive: `.omx/research/codex_routing_directive_rate_attack_vector_4_y3y6_luma_jpeg_20260518.md` (DEFERRED in this wave — not in TOP-3 routing directive list; per master memo §6 TOP-3 routing directives are F1+G1+B1).

### Phase 1 (PROBE; $0.30):
1. `tools/probe_y6_jpeg_quant_table_segnet_invariance.py` — verify Fridrich-canonical low-cost coefficient embedding doesn't perturb SegNet argmax

### Phase 2 (SUBSTRATE BUILD; $2-6):
1. Build Y3+Y6 substrate; Modal T4 smoke

## 9. Cross-References

- Master memo + META-paradigm
- Quantizr PR101 anchor: CLAUDE.md "Quantizr intelligence" section
- Fridrich PhD thesis: Binghamton DDE Lab + DDELab/deepsteganalysis GitHub
- UNIWARD canonical: Holub-Fridrich 2012 + DOI 10.1186/1687-417X-2014-1
- PR#56 Selfcomp gold-medal canonical (grayscale-LUT empirical)

## 10. Closeout

Y3+Y6 builds on the PROVEN Quantizr canonical + Fridrich PhD legitimacy. Predicted [-0.015, -0.004].

**Next action**: Phase 1 probe per Codex `019de465` (DEFERRED in this wave; routing directive TOP-3 = F1+G1+B1).
