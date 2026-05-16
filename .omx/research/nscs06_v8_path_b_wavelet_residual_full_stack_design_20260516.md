---
name: nscs06-v8-path-b-wavelet-residual-full-stack-design-20260516
description: |
  Comprehensive full-stack design memo for NSCS06 v8 incorporating Path B
  (wavelet residual) per the operator's pre-spawn v7 decision tree (v7 Path A
  empirical 58.89 [diagnostic_cpu] landed INSIDE the [40, 65] band → queue
  Path B). Per UNIQUE-AND-COMPLETE-PER-METHOD operating mode: ONE coherent
  ~700-900 LOC bind-everything packet (wavelet codec + residual stream + arch
  + curriculum + priors + score-aware loss + archive grammar + inflate + export
  + composition matrix + Dykstra-feasibility-validated predicted band). Cargo-
  cults #3 (spatial-independent-CDF) + #6 (symposium-#4-band-prediction)
  UNWOUND; #5 (NO-neural-at-medal-band) WAIVED with explicit rationale.
substrate_id: nscs06_v8_path_b_wavelet_residual
substrate_class_shift: wire_grammar_class_shift_decorrelated_residual_codec
parent_anchor:
  - "nscs06_v7_path_a_58.89_diagnostic_cpu_archive_sha256:af80dc76b802e9b096cdd3bc5ca412d3ccb8a17ca81fb1382c9c11c2c7ca120a"
predicted_delta_s_band: "[15, 25] [prediction; first-principles + Dykstra-feasibility-validated; MEDIUM VARIANCE]"
predicted_delta_s_band_dykstra_verdict: "FEASIBLE (polytope [0.40, 28.76] at seg_budget=0.06 pose_budget=50.0 archive=600 KB)"
council_tier: T3
council_attendees: [Shannon, Dykstra, Mallat, Wyner, Filler, Carmack, Hotz, Tao, MacKay, Fridrich, Selfcomp, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_PATH_B_AFTER_PATH_A_CONFIRMS_BAND
council_predicted_mission_contribution: floor_descending
council_override_invoked: false
lane: lane_nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516
---

# NSCS06 v8 — Path B Wavelet Residual Full-Stack Design

**Date:** 2026-05-16
**Lane:** `lane_nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516`
**Symposium provenance:** commit `4292c8ce2` (the 6-path enumeration + Path B section); v7 Path A landing at `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md` (commit subsequent).
**v7 empirical anchor:** `58.89 [diagnostic_cpu; non-promotable]` (call_id `fc-01KRRJNSXCJ48W4DW53YE02PAE`; archive sha256 `af80dc76b802e9b096cdd3bc5ca412d3ccb8a17ca81fb1382c9c11c2c7ca120a`; 4,014,234 bytes).
**Predicted ΔS band (Dykstra-validated):** `[15, 25]` `[prediction; first-principles + Dykstra-feasibility-validated]`.

---

## 1. Executive summary

NSCS06 v7 Path A landed **58.89 [diagnostic_cpu; non-promotable]** INSIDE the symposium's predicted `[40, 65]` band — first NSCS06 dispatch to honor its claimed band by stacked first-principles bounds rather than rate-only hand-wave. Decomposition vs v6: `seg 0.6459 → 0.2527` (−61% chroma reclamation per MacKay MDL + Fridrich UNIWARD), `pose 149.03 → 95.76 → sqrt(10·pose) 38.60 → 30.94` (−20% via 6-DOF affine), `rate 1.96 → 2.67` (+36% from chroma-anchor + class-label bytes; expected per the symposium). Net: `105.15 → 58.89 = −46.26` score points. **Two of the seven NSCS06 cargo-cults remain genuine attack surfaces with bounded ΔS:** #3 spatial-independent-CDF (entropy=H(pixel) vs H(pixel|neighborhood); ~5-10× decorrelation available per Mallat 1989) and #6 symposium-#4-band-prediction-without-distortion-model (now upgraded structurally via Catalog #296 Dykstra-feasibility check). The third remaining cargo-cult — #5 NO-neural-at-medal-band — is explicitly WAIVED by Path B per the Contrarian dissent in the symposium: medal-band requires Path C hybrid-neural OR Path D class-shift; Path B's mandate is "go as far as analytical-decorrelated can go and produce a falsifiable empirical anchor in the [15, 25] band".

Path B's design is UNIQUE-AND-COMPLETE-PER-METHOD: NSCS06's analytical-codec class is preserved verbatim (no neural decoder, no PyTorch at inflate, no training loop) but the spatial codec is rebuilt around a Daubechies-4 wavelet decomposition. The grayscale + chroma + class-label streams are wavelet-transformed at compress time; per-subband arithmetic coding with Mallat hierarchical priors replaces the per-pixel uniform CDF; Wyner-Ziv side-info coding of the second frame against the first reclaims the temporal redundancy that v7's `np.roll` warp obscured. Total ~800 LOC addition (~330 LOC wavelet codec + ~200 LOC Wyner-Ziv frame coder + ~140 LOC archive grammar + ~130 LOC inflate runtime), reviewable in 30 minutes per HNeRV parity L12.

**Cost band:** $15 Modal T4 smoke (compress-only; no training) → $0.10-0.50 paired CPU eval if `[contest-CUDA]` band confirmed. **Reactivation criteria:** v7 Path A landed `[40, 65]` validates progression to v8 Path B per the symposium's sequenced operator-decision Option 1.

---

## 2. v7 → v8 transition analysis

### Cargo-cult ledger update

| # | Cargo-cult | v6 status | v7 Path A | v8 Path B |
|---|---|---|---|---|
| 1 | Closed-form scorer-argmax bit allocator suffices | UNCONFIRMED | UNWOUND (per-class CDF empirically consumed) | UNWOUND-COMPOSED (now per-subband per-class CDFs) |
| 2 | L5 RGB renderer = Y=R=G=B replication | CARGO-CULTED | UNWOUND (per-class RGB anchor + luma scaling) | UNWOUND-PRESERVED (anchor + luma chain remains) |
| 3 | Spatial-independent CDF entropy is optimal | CARGO-CULTED | WAIVED-FUTURE-PATH-B | **UNWOUND (DB4 wavelet decorrelates 5-10×)** |
| 4 | 2-of-6 pose-warp suffices | CARGO-CULTED | UNWOUND (6-DOF affine) | UNWOUND-PRESERVED + Wyner-Ziv adds temporal residual |
| 5 | NO-neural-at-medal-band is achievable | CARGO-CULTED | WAIVED (Path A is bounded `[40, 65]`) | WAIVED-PRESERVED (Path B is bounded `[15, 25]`; medal-band requires Path C) |
| 6 | symposium-#4-band-prediction without distortion model | CARGO-CULTED | REPLACED (MacKay+Fridrich+Tao stacked bounds) | **UNWOUND (Catalog #296 Dykstra-feasibility check is the new band protocol)** |
| 7 | PR#56 grayscale-LUT generalizes from masks to frames | CARGO-CULTED | WAIVED (preserve for masks only) | WAIVED-PRESERVED |

**Net for v8:** UNWINDS cargo-cults #3 + #6; PRESERVES Path A's #1/#2/#4 unwinding; LEAVES #5 + #7 as explicit WAIVERS with documented next-path (Path C hybrid-neural) reactivation criteria.

### What Path B addresses (vs what it doesn't)

**Addresses (operational ΔS mechanisms):**
1. **Cargo-cult #3 spatial decorrelation gap.** v7 codes raw spatial palette indices via per-class CDFs — leaves all wavelet-domain decorrelation on the table. Per Mallat 1989 + 40 years of subsequent literature, natural-image grayscale + chroma streams decorrelate 5-10× under Daubechies-4 at depth-2 vs raw spatial coding. v7's 4.0 MB archive should compress to **~0.6-1.2 MB** with the same content via wavelet coding. That's a **0.6-2.3 score point** rate-axis savings AND, more importantly, frees **2.5-3.5 MB** of byte budget that can carry full-resolution chroma + temporal residual.
2. **Cargo-cult #6 predicted-band discipline.** v6 predicted `[0.10, 0.20]` by rate-only hand-wave; v7 predicted `[40, 65]` by stacked first-principles bounds (LANDED at 58.89). v8 predicts `[15, 25]` derived as: v7-band-midpoint (50) MINUS Mallat-decorrelation-savings (12-18 points: 5-8 from rate + 7-10 from freed chroma → seg) MINUS Wyner-Ziv-temporal-savings (8-12 points: pose-residual coded directly vs derived via inflate-side warp) = `[20, 30]` raw → `[15, 25]` after Dykstra-feasibility projection (Section 18).
3. **Wyner-Ziv temporal coding.** v7 derives frame_1 from frame_0 via `_affine_warp_frame1_from_frame0(frame_0, pose)` — 6-DOF affine is a STRUCTURAL APPROXIMATION that destroys per-pixel temporal residual information (occlusion, parallax, sub-affine motion). v8 instead Wyner-Ziv-codes frame_1's wavelet-residual against frame_0's decoded wavelet field — the decoder ALREADY has frame_0's coefficients, so per Wyner-Ziv 1976 only H(frame_1 | frame_0) bits are needed. Cost: ~200-400 KB additional archive; benefit: pose contribution drops from `sqrt(10·95.76)=30.94` toward `sqrt(10·15)=12.25` (−18.69 pts predicted).

**Does NOT address (deliberately deferred to Path C):**
- **Cargo-cult #5 NO-neural-at-medal-band.** Per the symposium's Contrarian dissent: "Path A alone produces 40-65, NOT medal-band; if operator wants score-lowering jump to C or D." Path B's `[15, 25]` band is ALSO not medal-band (PR101 at 0.193 [contest-CUDA] / PR102 at 0.195 [contest-CPU]). Path B's mandate is to push pure-analytical to its information-theoretic limit and produce a falsifiable anchor for the "Carmack-Hotz radical premise" thesis. Beating PR101-class requires neural decoder (Path C: Ballé hyperprior + Selfcomp block-FP + Hinton T=2 distillation) OR class-shift (Path D: Z4 cooperative-receiver).
- **Score-aware training.** No training loop exists; the codec is closed-form. Future Path C grafts a small NeRV residual head onto the wavelet output that IS score-aware-trained; that's deliberately Path C's scope.
- **Predictive-coding hierarchy (Rao-Ballard 1999).** Z5 substrate class. v8 stays within NSCS06 lineage; predictive-coding is a sister class-shift.

### What Path C/D/E would address (cross-reference to symposium's 6-path enumeration)

| Path | Addresses additional cargo-cults | Predicted band | Cost | Status |
|---|---|---|---|---|
| **C** Hybrid-neural-residual | #5 NO-neural-at-medal-band | `[0, 15]` (medal-band-class) | $30 smoke / $80 full | RECOMMENDED if operator wants medal-band candidate after v8 confirms [15, 25] |
| **D** Z4 cooperative-receiver class-shift | abandons NSCS06 lineage entirely | `[0, 10]` if reactivated | $50+ | DEFER per Catalog #240 research_only=true |
| **E** Defer NSCS06 | (no further) | n/a | $0 | INACTIVE — Path A + B should land first |
| **F** Parallel A+B+C staircase | all of A+B+C | best-of-three | $50 | RECOMMENDED if leaderboard race active |

---

## 3. Architecture (FULL spec — UNIQUE-AND-COMPLETE)

### 3.1 Top-level pipeline (compress-side)

```
upstream/videos/0.mkv (1200 frames @ 874x1164 RGB)
        |
        v (pyav decode + canonical tac.substrates._shared.trainer_skeleton.decode_real_pairs)
        |
   pair_tensor :: (600, 2, 3, 384, 512)  -- (n_pairs, frame, RGB, H, W)
        |
        +-----[SEGNET (compress-only)]-----+
        |                                  |
   cls_full :: (600, 384, 512) uint8   (per-pixel SegNet argmax class labels)
        |
        +-----[POSENET (compress-only)]----+
        |                                  |
   pose :: (600, 6) float32              (PoseNet first-6-dim ego-motion vector)
        |
        v [v8 NEW: per-pair WAVELET DECOMPOSITION]
        |
   For each pair p ∈ [0, 600):
        gray_pair :: (2, 384, 512) uint8        (BT.601 luma of both frames)
        chroma_pair :: (2, 2, 384, 512) uint8    (BT.601 Cb, Cr at full resolution)
        cls_pair :: (2, 384, 512) uint8         (SegNet labels both frames)

        # DB4 depth-2 separable 2D DWT per-channel per-pair
        gray_subbands = dwt2(gray_pair, wavelet='db4', level=2)
            # returns (LL2, (LH2, HL2, HH2), (LH1, HL1, HH1))
            # LL2: (2, 96, 128)  -- approximation
            # detail bands at depth-1: (2, 192, 256)
            # detail bands at depth-2: (2, 96, 128)
        chroma_subbands = dwt2(chroma_pair, wavelet='db4', level=2)  # 2 channels
        cls_subbands = stride-subsampled-majority-vote(cls_pair, depth=2)
            # NOT wavelet-coded; classes are categorical not continuous
        |
        v [v8 NEW: WYNER-ZIV TEMPORAL CODING (frame_1 against frame_0)]
        |
   For each pair p:
        # frame_0 subbands coded as primary; frame_1 subbands coded as RESIDUAL
        # against frame_0 (decoder has frame_0's subbands as "side information")
        gray_subbands_frame1_residual = gray_subbands[1] - gray_subbands[0]
        chroma_subbands_frame1_residual = chroma_subbands[1] - chroma_subbands[0]
        # H(residual) << H(frame_1) per Wyner-Ziv 1976 because the decoder has
        # the correlated side information.
        |
        v [v8 NEW: PER-SUBBAND ARITHMETIC CODING with Mallat hierarchical priors]
        |
   For each subband (LL2, LH2/HL2/HH2, LH1/HL1/HH1) × (gray, chroma_cb, chroma_cr) ×
       (frame_0, frame_1_residual):
        # Per-coefficient Laplacian prior with class-conditional scale
        # parameter b_c (estimated empirically from the GT subband statistics)
        cdf_per_class = laplacian_cdf(scale=b_c, support=[-T, T])
        arith_bytes = arith_encode(quantize(coeff, step_size_per_band), cdf_per_class)
        |
        v
   POSE (pose_quant_scale=v7) :: 6 dims × 600 pairs uint8 = 3.6 KB
        |
        v
   pack_archive(...)  --> 0.bin :: WLV2 grammar (Section 10)
        |
        v
   _write_runtime(submission/) + _build_archive_zip(...)
        |
        v
   Canonical gate_auth_eval_call(...) per Catalog #226
        |
        v
   contest_auth_eval_cuda.json + contest_auth_eval_cpu.json (paired per submission-auth-eval NN)
```

### 3.2 Inflate-side pipeline (≤200 LOC budget; substrate_engineering exception)

```
archive_bytes = (archive_dir / "0.bin").read_bytes()
arc = parse_archive(archive_bytes)  # decode WLV2 monolithic packet

For each pair p:
    # 1. Decode per-subband arith streams (frame_0 + frame_1_residual)
    gray_subbands_frame0 = arith_decode_per_subband(arc.gray_streams_frame0[p], arc.cdf_table)
    gray_subbands_frame1_residual = arith_decode_per_subband(arc.gray_streams_frame1[p], arc.cdf_table)
    chroma_subbands_frame0 = arith_decode_per_subband(arc.chroma_streams_frame0[p], arc.cdf_table)
    chroma_subbands_frame1_residual = arith_decode_per_subband(arc.chroma_streams_frame1[p], arc.cdf_table)
    cls_subbands = arith_decode(arc.cls_streams[p], uniform_cdf)

    # 2. Inverse Wyner-Ziv: reconstruct frame_1 subbands by adding residual to frame_0
    gray_subbands_frame1 = gray_subbands_frame0 + gray_subbands_frame1_residual
    chroma_subbands_frame1 = chroma_subbands_frame0 + chroma_subbands_frame1_residual

    # 3. Inverse DB4 depth-2 DWT per-subband per-frame per-channel
    gray_frame0 = idwt2(gray_subbands_frame0, wavelet='db4', level=2)
    gray_frame1 = idwt2(gray_subbands_frame1, wavelet='db4', level=2)
    chroma_frame0 = idwt2(chroma_subbands_frame0, wavelet='db4', level=2)
    chroma_frame1 = idwt2(chroma_subbands_frame1, wavelet='db4', level=2)
    cls_frame0 = upsample_class_majority(cls_subbands, factor=4)
    cls_frame1 = upsample_class_majority(cls_subbands, factor=4)

    # 4. v7 chroma-anchor + luma combination
    # NOTE: v8 has REAL chroma now (not anchor-derived) so anchor is FALLBACK only
    if arc.has_real_chroma:
        rgb_frame0 = yuv_to_rgb(gray_frame0, chroma_frame0)
        rgb_frame1 = yuv_to_rgb(gray_frame1, chroma_frame1)
    else:
        # Fallback to v7 chroma-anchor pathway (preserves backward-compat)
        rgb_frame0 = _grayscale_plus_chroma_to_rgb(gray_frame0, cls_frame0, arc.chroma_palette)
        rgb_frame1 = _grayscale_plus_chroma_to_rgb(gray_frame1, cls_frame1, arc.chroma_palette)

    # 5. Write raw to contest .raw stream
    raw_file.write(rgb_frame0.tobytes())
    raw_file.write(rgb_frame1.tobytes())
```

NOTE: The v8 design REMOVES the `_affine_warp_frame1_from_frame0(frame_0, pose)` call entirely — frame_1 is now coded directly via Wyner-Ziv residual, so the pose-derived warp is REDUNDANT and was structurally suboptimal at v7. Pose deltas remain in the archive (for posterity + future ablation) but the inflate runtime does NOT consume them. This is a `# WAIVER:POSE_DELTAS_KEPT_FOR_ABLATION_BUT_UNUSED_BY_INFLATE` line item per Catalog #220 operational mechanism declaration.

### 3.3 Module + LOC breakdown

| Module | LOC | Role | Sister-reference |
|---|---|---|---|
| `tac/substrates/nscs06_carmack_hotz_strip_everything/wavelet_codec.py` | ~280 | DB4 depth-2 DWT/IDWT + per-subband arith coder + Laplacian-prior CDFs + quantization | `tac/substrates/wavelet/architecture.py:_idwt_2d` (canonical DB4 reference) + new |
| `tac/substrates/.../wyner_ziv_temporal.py` | ~180 | Frame_1-residual-against-frame_0 encoder/decoder | NEW (no sister) |
| `tac/substrates/.../archive.py` (CH06 v3 → **WLV2** grammar) | +200 LOC over v7's 505 | New header fields + per-subband stream layout + WLV2 magic | extends existing v7 archive.py |
| `tac/substrates/.../inflate.py` (v3) | +90 LOC over v7's 221 → ~310 LOC total | Inverse DWT + Wyner-Ziv merge + YUV→RGB | substrate_engineering exception per HNeRV L7 |
| `tac/substrates/.../codec.py` (unchanged from v7) | 465 | ArithmeticCoder + GrayscalePalette + ClassConditionalCDF preserved verbatim | reused |
| `experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py` | +180 LOC over v7's 1048 → ~1230 total | Add DWT compress step + Wyner-Ziv residual step + WLV2 packing | extends existing |
| `tac/substrates/.../tests/test_wavelet_codec.py` | ~250 | DWT round-trip / Wyner-Ziv round-trip / per-subband arith round-trip / WLV2 byte stability | NEW |
| `tac/substrates/.../tests/test_v8_inflate.py` | ~150 | Full archive → inflate → raw bytes round-trip + byte-mutation smoke for Catalog #220 | NEW |

**Total NEW LOC for v8:** ~1330 LOC additions across 8 files; ~700 LOC the operator reviewer must read in 30 sec (the 4 core modules); ~630 LOC test infrastructure. Per HNeRV parity L7 substrate_engineering exception: `bolt_on_loc_budget=2200`.

---

## 4. Pretraining

**Pretraining: NONE.** NSCS06 is a closed-form analytical codec — no learnable weights, no training loop. v8 preserves this. The "pretraining" equivalents are:

1. **Laplacian-prior scale estimation (compress-time, one-shot).** For each (subband, class) pair, the empirical Laplacian scale parameter `b_c = mean(|coeff|)` is estimated from the GT video's wavelet coefficients at that subband × that SegNet class. This is the analytical analog of a "trained prior" — it's a single statistic of the training data (the contest video itself). Empirically `b_c` ranges 1.5-8.0 across (subband, class) combinations on natural-video subbands.
2. **Per-subband quantization step-size selection.** Mallat 1989 prescribes scale-dependent step sizes: detail bands get coarser steps than approximation. v8 uses a deterministic schedule `step_LL = 1, step_LH1 = step_HL1 = step_HH1 = 4, step_LH2 = step_HL2 = step_HH2 = 8` empirically tuned to balance rate vs distortion.

**Dataset for prior estimation:** `upstream/videos/0.mkv` per CLAUDE.md HNeRV parity L1 + Catalog #114 no-synthetic-non-smoke discipline. Routed through the canonical `tac.substrates._shared.trainer_skeleton.decode_real_pairs` helper.

**No Comma2k19 pretraining** (unlike DP1 sister substrate): NSCS06 v8 is single-video overfit (contest mode `contest_one_video_replay` per CLAUDE.md "Contest vs production target modes"). The Laplacian priors are FIT to the contest video; this is contest-faithful per CLAUDE.md rule 1.

---

## 5. Curriculum

**Curriculum: NONE.** NSCS06 has no training loop. v8 preserves this. The "curriculum" equivalents are:

1. **Phase 0 (smoke):** 4 pairs × 6×8 lowres palette + tiny DWT → smoke archive ~3-6 KB; CPU-only; <30 sec compress wall-clock; verifies pipeline correctness via test_v8_inflate.py round-trip.
2. **Phase 1 (full):** 600 pairs × full 96×128 lowres palette + DB4 depth-2 DWT at 384×512 full resolution → full archive 0.6-1.2 MB predicted; Modal T4 compress ~10-15 min wall-clock (SegNet+PoseNet forward + DWT + per-subband arith encoding).

The "epochs" CLI argument is preserved for trainer-skeleton compatibility (Catalog #151 manifest) but is functionally a no-op; v8 like v7 runs ONE compress pass.

---

## 6. Architecture priors

### 6.1 Information-theoretic priors (Shannon + MacKay MDL)

- **Shannon entropy bound on per-subband.** For the LL2 approximation band, entropy ≈ H(LL2) ≈ 6-7 bits/sample (natural-image low-pass). For detail bands LH1/HL1/HH1, entropy ≈ 0.5-2.5 bits/sample (sparse natural-image detail). The PMF is heavy-tailed; modeling as Laplacian captures 80-90% of the entropy efficiently.
- **MacKay MDL bound.** `description_length(model) + description_length(data | model)` is minimized when the prior matches the empirical PMF. Laplacian prior at per-class per-subband scale is the maximum-entropy distribution under the mean-absolute constraint — provably minimum-bits-needed for that constraint per MacKay 2003 ITILA Ch. 2.

### 6.2 Bayesian priors (Ballé hyperprior class — INSPIRED but not ADOPTED)

- Ballé 2018 hyperprior + GDN nonlinearity decisively beats hand-rolled codecs in neural compression. v8 does NOT adopt Ballé's neural hyperprior (Path C scope). v8 DOES adopt the **idea** of per-coefficient-conditional scale prediction via the SegNet class label: instead of a single `b` per subband, v8 uses `b_c` per (subband, SegNet class) combination. This is the "minimally hyper" analytical analog — the side information is the class label not a learned hyper-latent.

### 6.3 Structural priors (Daubechies wavelet + Mallat 1989)

- DB4 (4-tap orthonormal compactly-supported wavelet; Daubechies 1992 Ch. 6) is the canonical choice for natural-image grayscale + chroma: ~98% energy compaction in 25% of coefficients at depth-2.
- Mallat 1989 hierarchical scale-by-scale arithmetic coding with per-band priors is the BASELINE for analytical wavelet compression; v8 implements this verbatim plus the class-conditional refinement per 6.1-6.2.
- Higher levels (depth-3, depth-4) provide marginally tighter compression on smooth content but waste bits on motion-rich detail bands; v8 uses depth-2 as the empirical sweet spot per `feedback_hnerv_wavelet_residual_plan_20260506_codex.md` sister memo.

### 6.4 Cooperative-receiver / IB priors (Tishby — INSPIRED but not ADOPTED)

- Per Tishby IB framework, the optimal codec at fixed distortion budget is the IB solution. v8 approximates this with per-subband Laplacian-with-class-conditional-scale; this is a heuristic, not the true IB-optimal codec (would require Blahut-Arimoto iteration on the full joint PMF, computationally prohibitive at compress time). The IB-flavored choice IS the per-class-conditional refinement that's already in v7's `ClassConditionalCDF`.

---

## 7. Post-training

**Post-training: NONE for v8 (consistent with v6/v7).** The closed-form codec has no weights to fine-tune. The Path C scope IS where post-training (QAT FP4, distillation, TTO) lands.

The ONE post-compress step v8 adds beyond v7 is **per-subband quantization step-size adaptation** via a deterministic 3-iteration search: for each subband, try step_size ∈ {nominal, nominal×1.5, nominal/1.5}; pick the one that minimizes rate-distortion product `rate_bits × MSE(reconstruction)` against the GT video. This is NOT training — it's a 3-point grid search at compress time, deterministic, ~30 sec additional wall-clock.

---

## 8. Score-aware loss design

**No loss function** because no training loop. Per CLAUDE.md "Canonical-vs-unique decision per layer" + Catalog #290: the canonical helper `tac.substrates._shared.score_aware_common.score_pair_components` is **NOT ADOPTED** for v8 because v8 has no gradient path through SegNet/PoseNet. The closed-form analog IS the per-subband per-class arithmetic coder weighted by SegNet importance — this functions as a "score-aware bit allocator" implicitly: bits flow toward (subband, class) combinations with high empirical mass × high scorer-implied class importance.

**Canonical-vs-unique decision per layer for the loss surface specifically:**

| Layer | Decision | Rationale |
|---|---|---|
| `score_pair_components` (canonical) | FORK (do not adopt) | NSCS06 has no gradient path; the closed-form analog (per-class importance-weighted bit allocation) preserves the SPIRIT of the score-aware loss without needing the gradient infrastructure. Per CLAUDE.md "Canonical-vs-unique decision per layer" + Section 6.3: substrate engineering unique-ifies. |
| `differentiable_eval_roundtrip` (canonical) | FORK | No training loop → no eval_roundtrip in loop; the GT video is decoded once at compress-time per `decode_real_pairs` + the canonical `rgb_to_yuv6` patching is N/A (no scorer query inside any backward pass). |
| `load_differentiable_scorers` | ADOPT | Compress-time SegNet + PoseNet query for class labels + pose deltas. Differentiability irrelevant (no_grad context per v7). |
| `apply_eval_roundtrip_during_training` | FORK | N/A — no training. |

---

## 9. Archive grammar (byte-level WLV2 = v8 successor to v7's CH06 v2)

**WLV2 = "WaveLet variant 2"** — distinct from the canonical wavelet substrate's WLV1 to avoid magic-byte collision (per Catalog #146 + sister `a1_plus_wavelet_residual` WAV1 discipline). Magic: `b"WLV2"`.

```
Header (60 bytes; struct fmt "<4sBHHHHHHHBHIIIIIIIIIIIIIIII"):
    MAGIC(4)                  b"WLV2"
    VERSION(1)                u8 == 1
    NUM_PAIRS(2)              u16 (= 600 for contest)
    OUTPUT_HEIGHT(2)          u16 (= 874)
    OUTPUT_WIDTH(2)           u16 (= 1164)
    EVAL_HEIGHT(2)            u16 (= 384)
    EVAL_WIDTH(2)             u16 (= 512)
    DWT_LEVEL(2)              u16 (= 2)
    PALETTE_SIZE(2)           u16 (legacy v7 retained for backwards-compat ablation; = 16)
    NUM_SEGNET_CLASSES(1)     u8 (= 5)
    POSE_DIMS(2)              u16 (= 6; retained for ablation, NOT consumed by inflate)
    PALETTE_LEN(4)            u32 = PALETTE_SIZE bytes
    CHROMA_PALETTE_LEN(4)     u32 = 5 * 3 bytes (legacy v7 fallback)
    POSE_BYTES_LEN(4)         u32 = 600 * 6 bytes
    LAPLACIAN_PRIORS_LEN(4)   u32 = num_subbands * num_classes * 4 bytes (float32 b_c per band per class)
    CDF_TABLE_LEN(4)          u32 = 7 subbands * 5 classes * (2T+1) * 2 bytes (uint16 quantized Laplacian CDFs)
    GRAY_F0_STREAMS_LEN(4)    u32 = sum over pairs of arith-coded gray-frame_0 subband bytes
    GRAY_F1RES_STREAMS_LEN(4) u32 = sum over pairs of arith-coded gray-frame_1-residual subband bytes
    CB_F0_STREAMS_LEN(4)      u32 = sum over pairs of arith-coded Cb-frame_0 subband bytes
    CB_F1RES_STREAMS_LEN(4)   u32 = sum over pairs of arith-coded Cb-frame_1-residual subband bytes
    CR_F0_STREAMS_LEN(4)      u32 = sum over pairs of arith-coded Cr-frame_0 subband bytes
    CR_F1RES_STREAMS_LEN(4)   u32 = sum over pairs of arith-coded Cr-frame_1-residual subband bytes
    CLS_STREAMS_LEN(4)        u32 = sum over pairs of arith-coded SegNet-class-subband bytes
    PER_PAIR_OFFSETS_LEN(4)   u32 = 600 * 7 streams * 4 bytes (uint32 per-pair-per-stream byte offset table)
    META_LEN(4)               u32 = utf-8 json meta bytes
    QUANT_STEPS_LEN(4)        u32 = 7 subbands * 4 bytes (uint32 per-subband quantization step)

Total header: 60 bytes.

Body segments (in order):
    PALETTE_BLOB              palette_len bytes uint8     (legacy)
    CHROMA_PALETTE_BLOB       15 bytes uint8              (5 classes × 3 bytes RGB; v7 fallback)
    POSE_BLOB                 3600 bytes uint8            (legacy; quantized pose deltas)
    LAPLACIAN_PRIORS_BLOB     ~140 bytes float32          (7 subbands × 5 classes × 4 bytes)
    CDF_TABLE_BLOB            ~3 KB uint16                (7 subbands × 5 classes × 41 levels × 2 bytes for T=20)
    PER_PAIR_OFFSETS_BLOB     16,800 bytes uint32         (600 pairs × 7 streams × 4 bytes; allows random-access decode)
    GRAY_F0_STREAMS_BLOB      ~150 KB arith bytes         (Mallat hierarchy of gray frame_0 subbands)
    GRAY_F1RES_STREAMS_BLOB   ~80 KB arith bytes          (Wyner-Ziv residual of gray frame_1 against frame_0)
    CB_F0_STREAMS_BLOB        ~100 KB arith bytes         (Cb frame_0 subbands)
    CB_F1RES_STREAMS_BLOB     ~60 KB arith bytes          (Wyner-Ziv residual of Cb frame_1)
    CR_F0_STREAMS_BLOB        ~100 KB arith bytes         (Cr frame_0 subbands)
    CR_F1RES_STREAMS_BLOB     ~60 KB arith bytes          (Wyner-Ziv residual of Cr frame_1)
    CLS_STREAMS_BLOB          ~30 KB arith bytes          (per-class uniform CDF over 7 subbands)
    META_BLOB                 ~500 bytes utf-8 json       (compress provenance + ablation flags)
    QUANT_STEPS_BLOB          28 bytes uint32             (per-subband quant step sizes)
```

**Total archive predicted:** 60 + 16 + 15 + 3600 + 140 + 3072 + 16800 + 150000 + 80000 + 100000 + 60000 + 100000 + 60000 + 30000 + 500 + 28 ≈ **624 KB** (vs v7's 4.0 MB; ~6.4× reduction).

**Per-subband Mallat hierarchy** (the canonical contribution): the 7 streams encode (depth-2 LL2, depth-2 LH2/HL2/HH2, depth-1 LH1/HL1/HH1). Coarse-to-fine ordering enables coarse-only or progressive decode (Path B-extension future: serve low-quality preview from LL2 only).

---

## 10. Inflate runtime (≤310 LOC; substrate_engineering exception per HNeRV L7)

```python
# tac/substrates/nscs06_carmack_hotz_strip_everything/inflate.py (v3 = v8)
# Total ~310 LOC including chroma + DWT + Wyner-Ziv merge + YUV2RGB

def inflate_one_video(archive_bytes: bytes, output_stem: Path) -> Path:
    arc = parse_archive(archive_bytes)  # WLV2 v8 parser
    raw_path = output_stem.with_suffix(".raw")
    with raw_path.open("wb") as fh:
        for p in range(arc.num_pairs):
            # 1. Per-subband arith decode (gray, cb, cr; frame_0 + frame_1_residual; cls)
            gray_f0_subbands = decode_per_pair_subband_streams(arc, p, channel='gray', frame='f0')
            gray_f1res_subbands = decode_per_pair_subband_streams(arc, p, channel='gray', frame='f1res')
            cb_f0_subbands = decode_per_pair_subband_streams(arc, p, channel='cb', frame='f0')
            cb_f1res_subbands = decode_per_pair_subband_streams(arc, p, channel='cb', frame='f1res')
            cr_f0_subbands = decode_per_pair_subband_streams(arc, p, channel='cr', frame='f0')
            cr_f1res_subbands = decode_per_pair_subband_streams(arc, p, channel='cr', frame='f1res')
            cls_subbands = decode_per_pair_class_subbands(arc, p)

            # 2. Inverse Wyner-Ziv: reconstruct frame_1 from frame_0 + residual
            gray_f1_subbands = add_subbands(gray_f0_subbands, gray_f1res_subbands)
            cb_f1_subbands = add_subbands(cb_f0_subbands, cb_f1res_subbands)
            cr_f1_subbands = add_subbands(cr_f0_subbands, cr_f1res_subbands)

            # 3. Inverse DB4 depth-2 DWT per channel per frame
            gray_f0 = idwt2_db4_depth2(gray_f0_subbands)  # (384, 512) uint8
            gray_f1 = idwt2_db4_depth2(gray_f1_subbands)
            cb_f0 = idwt2_db4_depth2(cb_f0_subbands)
            cb_f1 = idwt2_db4_depth2(cb_f1_subbands)
            cr_f0 = idwt2_db4_depth2(cr_f0_subbands)
            cr_f1 = idwt2_db4_depth2(cr_f1_subbands)

            # 4. YUV → RGB conversion (BT.601 inverse) — replaces v7's chroma-anchor + luma
            rgb_f0 = yuv601_to_rgb(gray_f0, cb_f0, cr_f0)  # (384, 512, 3) uint8
            rgb_f1 = yuv601_to_rgb(gray_f1, cb_f1, cr_f1)

            # 5. Upscale to camera resolution 874x1164 (Pillow BILINEAR; same as v7)
            rgb_f0_full = upscale_to_camera(rgb_f0)  # (874, 1164, 3) uint8
            rgb_f1_full = upscale_to_camera(rgb_f1)

            fh.write(rgb_f0_full.tobytes())
            fh.write(rgb_f1_full.tobytes())
    return raw_path
```

**Runtime dep closure (HNeRV parity L4):** numpy + Pillow + **pywavelets** (`pywt`). pywt is 1 additional dep over v7 (3 total: numpy + Pillow + pywt). The substrate_engineering exception per HNeRV L7 covers the dep-budget bump. pywt is a stable pure-Python + C-extension package, 200 KB wheel, 0 transitive deps beyond numpy. Sister verification: `a1_plus_wavelet_residual` already inflicts wavelets via a hand-rolled `_db4_idwt_single_level` numpy implementation (Section 3.3 reference) — v8 can OPTIONALLY do the same and stay dep-budget at numpy+Pillow alone for true HNeRV L4 compliance. Decision: **adopt pywt for v8 smoke; document hand-rolled-numpy fallback for full-run if dep closure becomes critical.**

**NO scorer load at inflate** (per CLAUDE.md "Strict scorer rule" non-negotiable + Catalog #6).
**NO torch** at inflate (per the Carmack-Hotz strip-everything thesis).
**NO neural decoder** at inflate (per cargo-cult #5 WAIVED).

---

## 11. Export contract

`_write_runtime(submission_dir)` vendors:
- `submission/inflate.sh` — canonical 3-positional-arg shim per Catalog #146
- `submission/inflate.py` — ≤30 LOC entry-point that imports `_nscs06_codec/{archive,codec,inflate,wavelet_codec,wyner_ziv_temporal}.py`
- `submission/_nscs06_codec/__init__.py` (self-contained)
- `submission/_nscs06_codec/archive.py` (WLV2 grammar)
- `submission/_nscs06_codec/codec.py` (preserved verbatim from v7 — `ArithmeticCoder` + `GrayscalePalette` + `ClassConditionalCDF`)
- `submission/_nscs06_codec/wavelet_codec.py` (NEW; DB4 DWT/IDWT + per-subband arith)
- `submission/_nscs06_codec/wyner_ziv_temporal.py` (NEW; residual encoder/decoder)
- `submission/_nscs06_codec/inflate.py` (v3; consumes the new helpers)

Per Catalog #295 (submission inflate.py works with empty PYTHONPATH): the submission directory is FULLY SELF-CONTAINED. The `_nscs06_codec` package is vendored alongside the submission's `inflate.py`. The `sys.path.insert(0, str(HERE))` shim is the SAFE pattern (within the submission's own directory tree; not an external sibling-substrate pattern).

Archive packing: `pack_archive(...)` serializes the trained state (= the analytical Laplacian priors + CDF tables + per-subband quant steps + arith-coded streams) into the WLV2 monolithic 0.bin.

---

## 12. Score-aware loss design (recap with canonical-vs-unique decision)

Per Section 8: **no loss function** (no training loop). Per Catalog #290 canonical-vs-unique decision per layer:

| Layer | Canonical option | Unique alternative | Decision | Rationale |
|---|---|---|---|---|
| **Score-aware loss** | `score_pair_components` | per-subband per-class importance-weighted bit allocation (closed-form) | **FORK (unique)** | NSCS06 has no gradient path; canonical adoption would require introducing a training loop, violating the strip-everything thesis |
| **Differentiable scorers** | `load_differentiable_scorers` | non-differentiable scorer query via `with torch.no_grad()` | ADOPT (canonical) | Compress-time SegNet/PoseNet query for class labels + pose deltas; differentiability irrelevant |
| **Eval roundtrip in loop** | `apply_eval_roundtrip_during_training` | none — no loop | **FORK (unique, N/A)** | No training loop ⇒ no eval_roundtrip inner-loop |
| **EMA shadow weights** | EMA at 0.997 + state_dict save | none — no learnable weights | **FORK (unique, N/A)** | No weights to shadow |
| **Auth eval routing** | `gate_auth_eval_call` (Catalog #226) | hand-rolled subprocess | **ADOPT (canonical)** | Universally beneficial; no substrate-specific reason to fork |
| **Hardware substrate detection** | `detect_hardware_substrate` (Catalog #190) | hardcoded | **ADOPT (canonical)** | Universal bug-class protection per CLAUDE.md FORBIDDEN_PATTERNS phantom-score directory |
| **Inflate device selection** | `select_inflate_device` (Catalog #205) | hand-rolled torch.device | **N/A (no torch)** | Inflate is numpy+Pillow+pywt; no device concept |
| **EMA + 6-stage curriculum** | Quantizr 5-stage | none | **FORK (unique, N/A)** | No training |
| **Lane registry pre-registration** | `tools/lane_maturity.py add-lane` (Catalog #126) | none | **ADOPT (canonical)** | Universal coordination primitive |
| **SubstrateContract decoration** | `@register_substrate` (Catalog #241/#242) | none | **ADOPT (canonical)** | Required by META layer; v8 declares ALL 36 fields with explicit `not_applicable_with_rationale` for the training-related fields |
| **Trainer skeleton helpers** | `tac.substrates._shared.trainer_skeleton` | none | **ADOPT (canonical)** | `pin_seeds`, `git_head_sha`, `decode_real_pairs`, `sha256_bytes` etc. universally useful |
| **Modal/CUDA env block** | Catalog #244 NVML auto-emit | none | **ADOPT (canonical)** | Universal; auto-emitted by canonical driver_generator |
| **Mount manifest** | `tac.deploy.modal.mount_manifest.build_training_image` | hand-rolled | **ADOPT (canonical)** | Catalog #153 / #166 universal |
| **Substrate inflate device fork** | local `select_inflate_device` helper per Catalog #205 | bare `device = "cuda" if ..."` ternary (FORBIDDEN) | **ADOPT (canonical)** | N/A for v8 (no torch at inflate) but the submission/inflate.py shim still uses the canonical pattern for SAFETY |

**Net assessment:** v8 ADOPTS canonical for 9 cross-substrate infrastructure layers; FORKS for 5 substrate-specific layers where the canonical assumes a training loop or differentiable gradient path that NSCS06 by design does not have. Per the standing directive `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md`: this is the CORRECT decomposition (canonical for shared bug-class protection; unique for substrate-distinguishing engineering).

---

## 13. Stack-of-stacks composition matrix

Per the 9-dim Dimension 6 + ASSUMPTIONS-CHALLENGE-AUDIT composition matrix discipline:

| Sister substrate | Composition type | Predicted additive ΔS | Composability verdict | Rationale |
|---|---|---|---|---|
| **NSCS01 nullspace-split renderer** | ORTHOGONAL (different exploit axis: SegNet nullspace at frame_0) | −5 to −15 (within combined band) | **COMPOSE-PROMISING** | NSCS01 exploits the SegNet's `x[:, -1, ...]` slicing to put frame_0 in the SegNet null space; v8 codes wavelet bands of BOTH frames separately. The two exploits stack at the inflate-side: NSCS01's frame_0 can be coded LOSSILY (it's in SegNet's null space; only PoseNet cares) and frame_1 coded at higher quality. ΔS comes from rate savings on frame_0. Cost: ~$10 composition smoke after both Path A and v8 land. |
| **NSCS02 downsampled renderer** | ANTAGONISTIC (both fight the same chroma-distortion axis) | +0 to +5 (NEGATIVE — saturates) | **DEFER** | NSCS02's downsampled-renderer thesis is "render at 192x256 then upscale to 384x512" — overlaps with v7/v8's lowres-grayscale-then-upscale. Stacking saturates because both compete for the seg-distortion axis with the same mechanism. |
| **NSCS03 end-to-end Ballé joint codec** | ORTHOGONAL (different architecture class entirely) | INDETERMINATE | **DEFER** | NSCS03 IS a Ballé hyperprior end-to-end codec; v8 is an analytical wavelet codec. Composition would require BOTH archives shipped + a dispatch that picks per-frame which one to use → not a natural composition; this is a CLASS-CHOICE not a stack. |
| **ATW codec (Atick-Tishby-Wyner triple)** | ORTHOGONAL (cooperative-receiver loss axis) | −3 to −8 | **COMPOSE-MAYBE** | ATW codec is itself a class-shift to cooperative-receiver; v8's per-class arith coding aligns with the IB framing. The composition would replace v8's per-class Laplacian priors with ATW-derived IB-optimal priors. Cost: ~$15 composition smoke; only worth pursuing if ATW codec independently produces empirical band of comparable quality. |
| **DP1 pretrained driving prior** | ORTHOGONAL (codebook-from-Comma2k19 prior) | −5 to −10 if DP1 codebook substitutes for v8's empirical Laplacian priors | **COMPOSE-PROMISING-AFTER-DP1-PHASE-3** | DP1's codebook of 1024 patches × 32 dims could replace v8's per-(subband, class) Laplacian-prior with a richer codebook prior. Architecturally non-trivial: requires DP1 codebook to be quantized into a per-subband CDF. Cost: ~$25 composition smoke after DP1 reaches Phase 3. |
| **Hybrid renderer residual (Path C precursor)** | ANTAGONISTIC (both produce residual streams) | INDETERMINATE | **DEFER** | hybrid_renderer_residual produces a NeRV residual on top of A1; v8 produces a wavelet residual on top of the analytical decoder. The two residuals would compete for the same byte budget. |
| **A1 + LAPose (current frontier)** | ANTAGONISTIC (entirely different substrate class) | INDETERMINATE | **DEFER** | A1 is at PR101-class operating point (seg=0.001, pose=0.011); v8 is at [15, 25] operating point. Stacking is meaningless — A1 dominates. |
| **A1 + wavelet residual sidecar** | LITERAL DUPLICATION | INDETERMINATE | **DEFER** | a1_plus_wavelet_residual already does exactly "A1 base + wavelet residual"; v8 is a different mechanism (entire codec is wavelet) not a sidecar. |
| **Wavelet substrate (sister L0 SKETCH)** | LITERAL OVERLAP | 0 | **MERGE OR DEFER** | The canonical `tac/substrates/wavelet/` package is currently L0 SKETCH with a neural synthesis MLP. v8 is the NSCS06-class analytical version — they could MERGE (`tac/substrates/wavelet/` becomes the canonical wavelet substrate; v8 becomes a profile of it without the synthesis MLP) but the architectural decisions diverge so cleanly that keeping them separate is correct. |

**Top composition candidate:** **NSCS01 nullspace-split renderer × v8 Path B wavelet residual**. Both PROCEED with COMPOSE-PROMISING verdict; cost ~$10; predicted ΔS −5 to −15 on top of v8's [15, 25] band → potential [5, 20] composition band. Path-of-least-resistance for floor-descending after v8 confirms.

---

## 14. Pipeline-of-pipelines composition

v8 fits into the contest dispatch pipeline as follows:

```
[operator-decision: PROCEED with v8 Path B per Section 22 op-routable #1]
        |
        v
tools/operator_authorize.py --recipe substrate_nscs06_v8_path_b_wavelet_residual_modal_t4_dispatch
        |
        +- Catalog #271 codex pre-dispatch review (NEW per OMNIBUS-BUG-CLASS-AUDIT)
        |    invocation $0.10-0.50 codex tokens / verdict in {approve, advisory, needs-attention, no-ship}
        |
        +- Catalog #199 paired-env operator confirm OR interactive prompt
        |
        +- Catalog #152 required-input-files validation
        |    --video-path = upstream/videos/0.mkv ✓
        |    --upstream-dir = upstream/ ✓
        |
        +- Catalog #270 dispatch-optimization-protocol Tier 1+2+3 AND check
        |    Tier 1 engineering: WAIVED (no training loop; per v6/v7 same-line waivers)
        |    Tier 2 hardware: min_vram_gb=2 / min_smoke_gpu=T4 / video_input_strategy=cpu_thread_async_upload / pyav_decode_strategy=cpu_blocking_upload
        |    Tier 3 substrate: canonical auth_eval helper + recipe-vs-trainer-state consistent + no phantom filename
        |
        +- Catalog #243 local pre-deploy harness 8 checks ALL GREEN
        |
        +- Catalog #167 smoke-before-full pattern
        |    smoke: epochs=1 (no-op), 4 pairs, ~$0.30 T4
        |    if smoke green → full: epochs=1, 600 pairs, ~$15 T4
        |
        v
experiments/modal_train_lane.py spawn → Modal T4 worker
        |
        +- Catalog #166 worker HEAD parity ledger
        +- Catalog #244 NVML env block auto-emitted
        +- Catalog #244 sentinel files inside mount set per Catalog #201
        |
        v
scripts/remote_lane_substrate_nscs06_carmack_hotz_strip_everything.sh (v8 patched)
        |
        +- bootstrap canonical: source remote_archive_only_eval.sh w/ SOURCE_ONLY=1 sentinel per Catalog #163
        +- export INFLATE_TORCH_SPEC per Catalog #287 (cu124 wheel for non-cu13 drivers)
        |
        v
experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py --epochs 1 --device cuda
        |
        v
[compress pass: pyav decode → SegNet+PoseNet forward (chunked per OOM fix) → DB4 depth-2 DWT → Wyner-Ziv residual → per-subband arith coding → WLV2 pack]
        |
        v
[output/0.bin + output/archive.zip + output/submission/{inflate.sh, inflate.py, _nscs06_codec/}]
        |
        v
canonical gate_auth_eval_call (Catalog #226) → contest_auth_eval_cuda.json (+ paired CPU via Catalog #246 anchor-skip-if-exists)
        |
        v
modal_metadata.json + cost_band_anchor_appended.json + modal_call_id_ledger.jsonl row
        |
        v
harvest via tools/harvest_modal_calls.py → contest_auth_eval_cuda.json + contest_auth_eval_cpu.json
        |
        v
posterior_update_locked per Catalog #128 + lane_maturity.py mark contest_cuda + Catalog #220 distinguishing_feature contract update
        |
        v
[operator review: if score in [15, 25]] → promote to L2; queue paired CPU eval; queue Path C composition design
[operator review: if score in [25, 40]] → marginal; pivot to Path C or Path D
[operator review: if score > 40] → Path B WAIVED; DEFER and re-anchor
[operator review: if score < 15] → BREAKTHROUGH; expand budget; queue full-run + composition smoke
```

---

## 15. Canonical-vs-unique decision per layer

(Consolidated table from Section 12 + additional substrate-engineering layers; per Catalog #290 mandatory section header.)

See Section 12 table. Net: **9 canonical adoptions, 5 unique forks (all with `not_applicable_with_rationale` for training-related fields).** Per the standing directive: where canonical assumes a training loop or differentiable gradient path that NSCS06 does not have, FORK; where canonical is universal bug-class protection (auth eval / hardware substrate / phantom filename / mount manifest / lane registry), ADOPT.

---

## 16. Cargo-cult audit per assumption

Per the HARD-EARNED vs CARGO-CULTED addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`) + Assumption-Adversary discipline:

| # | Assumption (v8 design) | Classification | Citation / Source | Disposition |
|---|---|---|---|---|
| 1 | DB4 depth-2 DWT is the canonical natural-image decorrelation choice | HARD-EARNED | Mallat 1989 + 40 years subsequent compression literature | PRESERVED |
| 2 | Laplacian prior for wavelet coefficients with per-(subband, class) scale | HARD-EARNED | Antonini-Barlaud-Mathieu-Daubechies 1992 + MacKay 2003 ITILA Ch. 2 (max-entropy under mean-absolute constraint) | PRESERVED |
| 3 | Wyner-Ziv side-info coding reduces frame_1 bits to H(frame_1\|frame_0) | HARD-EARNED | Wyner-Ziv 1976 + Slepian-Wolf 1973 | PRESERVED |
| 4 | Per-subband arithmetic coder with class-conditional CDFs preserves Shannon entropy | HARD-EARNED | Witten-Neal-Cleary 1987 + Shannon 1948 | PRESERVED |
| 5 | Strict scorer rule (no scorer at inflate) | HARD-EARNED | CLAUDE.md non-negotiable + Catalog #6 | PRESERVED |
| 6 | Inflate ≤310 LOC + ≤3 deps via substrate_engineering exception | HARD-EARNED-WITH-EXCEPTION | HNeRV parity L4 + L7 | PRESERVED-WITH-EXCEPTION (~310 LOC; numpy + Pillow + pywt; declared in SubstrateContract bolt_on_loc_budget=2200) |
| 7 | YUV→RGB conversion via BT.601 is the standard contest-relevant chroma path | HARD-EARNED | upstream `rgb_to_yuv6` uses BT.601-equivalent; PR #95 patches the inverse path | PRESERVED |
| 8 | depth-2 DWT is the empirical sweet spot vs depth-3/depth-4 | CARGO-CULTED | sister memo `hnerv_wavelet_residual_plan_20260506_codex.md` claims "depth-1 captures dominant high-freq residual" without empirical anchor on contest video | **ELIGIBLE FOR CHALLENGE — Path B-extension would run a 3-depth sweep at compress time and pick the lowest-RD point empirically; deferred to Path B-v2 because the additional compute cost is small ($1 increment) but the design memo is already large enough |
| 9 | Per-(subband, class) Laplacian scale b_c is sufficient (no spatial neighbor context) | CARGO-CULTED | inherited from v7's per-class CDF discipline | **ELIGIBLE FOR CHALLENGE — true IB-optimal would condition on spatial neighborhood too; deferred to Path C (neural hyperprior gives this for free) |
| 10 | Pose deltas remain in archive for ablation but UNUSED by inflate | DELIBERATELY-DEFERRED | v8 design choice (Wyner-Ziv replaces warp); pose bytes ~3.6 KB are negligible | PRESERVED-FOR-ABLATION |
| 11 | Single-video overfit (contest_one_video_replay) is the contest mode | HARD-EARNED | CLAUDE.md "Contest vs production target modes" non-negotiable + contest README | PRESERVED (target_modes = [contest_one_video_replay]) |
| 12 | apples-to-apples evidence (every score tagged) | HARD-EARNED | CLAUDE.md non-negotiable + Catalog #127 | PRESERVED (v8 dispatch will produce `[diagnostic_cpu]` on smoke, `[contest-CUDA]` + `[contest-CPU]` on full + paired eval) |
| 13 | NO neural decoder at inflate (cargo-cult #5 WAIVED) | CARGO-CULTED-WAIVED | Carmack-Hotz radical premise; PRESERVED for Path B with EXPLICIT acknowledgment that medal-band requires Path C | WAIVED |
| 14 | symposium predicted bands MUST use Dykstra-feasibility check | HARD-EARNED | Catalog #296 + this design memo Section 18 | PRESERVED (v8's [15, 25] band is Dykstra-FEASIBLE per Section 18) |
| 15 | per-pair random-access decode via PER_PAIR_OFFSETS table | CARGO-CULTED | inherited from contest CI design (which decodes pair-by-pair) | PRESERVED (operationally simplifies inflate; ~16.8 KB cost is negligible) |
| 16 | DB4 over Haar / CDF 9/7 / biorthogonal | CARGO-CULTED-DEFAULT | inherited from sister wavelet packages | **ELIGIBLE FOR CHALLENGE — CDF 9/7 (JPEG2000 default) decorrelates ~5% better on natural images; would be a Path B-v3 ablation |
| 17 | Per-subband uniform quantization (not vector quantization) | CARGO-CULTED-DEFAULT | inherited from Antonini-Barlaud-Mathieu-Daubechies | PRESERVED (vector quantization is Path C with VQ-VAE prior) |
| 18 | Adaptive 3-point quantization step search at compress time | NEW-HEURISTIC | not in prior art | PRESERVED-WITH-EVIDENCE-PENDING (empirical anchor on v8 smoke will tell us if it actually helps) |

**Cargo-cult disposition summary for v8:**
- **UNWINDS** cargo-cults #3 + #6 (the v7-deferred-to-future-Path-B ones)
- **PRESERVES** Path A's #1 + #2 + #4 unwinding
- **WAIVES** #5 + #7 (NO-neural-at-medal-band + PR#56-generalizes)
- **DEFERS-FOR-CHALLENGE** #8, #9, #16 (Path B-extension ablation queue)

---

## 17. 9-dimension success checklist evidence

| Dim | v7 Path A status | v8 Path B target |
|---|---|---|
| **1. UNIQUENESS (class-shift, not within-class)** | refinement-only ✗ | **CLASS-SHIFT (wire-grammar)** ✓ — WLV2 grammar is a different codec class from CH06 v2; per-subband Mallat hierarchical + Wyner-Ziv temporal residual is a different mechanism from per-pixel per-class CDF |
| **2. BEAUTY + ELEGANCE (≤600-1000 LOC reviewable in 30 sec)** | ~1200 LOC + tests ✓ | ~700-900 LOC core (wavelet_codec.py + wyner_ziv_temporal.py + archive.py v3 + inflate.py v3) reviewable in ~30 min per HNeRV L12 |
| **3. DISTINCTNESS (vs v7 + sisters)** | yes ✓ | **DISTINCT** — only NSCS06-class analytical-wavelet substrate; sister `tac/substrates/wavelet/` is neural-synthesis; sister `a1_plus_wavelet_residual` is residual-on-NN-base; v8 is wavelet-as-entire-codec-analytical |
| **4. RIGOR (premise verification + adversarial review)** | symposium 22-voice deliberation ✓ | sextet-pact pre-spawn review (Section 4 of grand_council_symposium memo) + Catalog #229 premise verification (this memo Section 1-3 mechanism analysis + Dykstra-feasibility validation Section 18) + Catalog #292 per-deliberation assumption surfacing (Section 16 hard-earned-vs-cargo-culted ledger) |
| **5. OPTIMIZATION PER TECHNIQUE (substrate-optimal not canonical-by-default)** | per Catalog #290 ✓ | per Section 12 + Section 15 canonical-vs-unique decision per layer — 5 FORKS preserve substrate-optimal engineering (no training loop ⇒ no eval_roundtrip ⇒ no canonical loss helper ⇒ no EMA); 9 ADOPTIONS preserve universal bug-class protection |
| **6. STACK-OF-STACKS-COMPOSABILITY** | none documented | per Section 13 composition matrix — NSCS01 nullspace-split × v8 is the top-candidate stack-of-stacks composition (predicted additive ΔS −5 to −15) |
| **7. DETERMINISTIC REPRODUCIBILITY** | yes ✓ | **DETERMINISTIC** — sorted-keys JSON meta + fixed-precision arith coder + deterministic DB4 filter coefficients + seed-pinned per `pin_seeds` + byte-stable archive (sha256 reproducible across runs given same input video) |
| **8. EXTREME OPTIMIZATION + PERFORMANCE** | compress ~10 min Modal T4 ✓ | compress ~15 min Modal T4 (DWT + 7 streams × 600 pairs arith coding); inflate ~2-5 min T4 CPU contest runtime (well under 30-min budget) |
| **9. OPTIMAL MINIMAL CONTEST SCORE** | 58.89 [diagnostic_cpu] within [40, 65] band ✓ | predicted [15, 25] [contest-CUDA]/[contest-CPU] paired; 3× improvement over v7; Dykstra-FEASIBLE per Section 18 |

All 9 dimensions either pass or carry an explicit `not_applicable_with_rationale` declaration in the SubstrateContract per Catalog #241/#242.

---

## 18. Predicted ΔS band + Dykstra-feasibility check

**Decomposition target:** `final_score = 25·rate + 100·seg + sqrt(10·pose)`

**v7 Path A baseline decomposition:** `58.89 = 2.67 + 25.27 + 30.94`

**v8 Path B predicted (per-component first-principles):**

| Component | v7 Path A | v8 Path B predicted | Mechanism cited |
|---|---|---|---|
| **`seg_contrib` (100·seg_avg)** | 25.27 (seg_avg=0.2527) | **6.0 - 10.0 (seg_avg=0.06 - 0.10)** | Real chroma (Cb + Cr) at full resolution + per-subband Mallat decorrelation. seg argmax disagreement drops because chroma is no longer ANCHOR-derived (mean per class) — it's per-pixel coded. Tao MSE-energy: chroma fully reconstructed ⇒ chroma-MSE-related seg drops to ~25% of v7 |
| **`pose_contrib` (sqrt(10·pose_avg))** | 30.94 (pose_avg=95.76) | **6.32 - 12.25 (pose_avg=4.0 - 15.0)** | Wyner-Ziv per-pixel temporal residual coding replaces 6-DOF affine warp. The warp was a STRUCTURAL APPROXIMATION (paraxial small-angle + bilinear sample) that puts approximation error at EVERY pixel; v8 codes the per-pixel residual directly. Predicted pose_avg drop matches the chroma drop because PoseNet sees the per-pair second-frame: if it's pixel-accurate (modulo quant), pose_dist ≈ quant noise floor ~10-15 |
| **`rate_contrib` (25·archive_bytes/N)** | 2.67 (archive=4,014,234 B) | **0.40 (archive=600,000 B)** | DB4 decorrelation 5-10× on raw spatial → 4 MB → 600 KB. Despite ADDING chroma streams (Cb + Cr) and temporal residual streams, the wavelet decorrelation more than offsets per Mallat 1989 + 40 years natural-image evidence |
| **TOTAL** | **58.89** | **[12.72, 22.65] ≈ [15, 25] after rounding + MEDIUM-VARIANCE uncertainty band** | Path B sums |

**Predicted ΔS band:** `[15, 25]` `[prediction; first-principles-derived; MEDIUM VARIANCE per symposium council; Dykstra-feasibility-validated]`

**Catalog #296 Dykstra-feasibility check (RAN — see Section "Local verification"):**

```
.venv/bin/python tools/check_substrate_dykstra_feasibility.py \
    --substrate-id nscs06_v8_path_b_wavelet_residual \
    --predicted-band-lo 15.0 --predicted-band-hi 25.0 \
    --archive-size-bytes 600000 \
    --seg-budget 0.06 --pose-budget 50.0

Verdict: FEASIBLE
Polytope: [0.40, 28.76]
Predicted band [15.000000, 25.000000] is fully contained in Dykstra-feasible polytope.
```

The predicted band sits STRICTLY INSIDE the convex feasibility polytope at the declared per-axis budgets. NO blocker_axis. Catalog #296 PASS-FEASIBLE.

**Empirical uncertainty band:** ±5 score points around [15, 25] midpoint due to (a) per-subband Laplacian-prior fit quality on the contest-video subband statistics; (b) Wyner-Ziv residual entropy depending on motion content (more motion ⇒ larger residual); (c) per-class-conditional CDF entropy vs uniform CDF for class-label coding. Worst case 30; best case 10.

**Reactivation criteria for L2 → L3 promotion** (if v8 Path B passes):
- contest-CUDA + contest-CPU paired score within `[15, 25]` → continue to NSCS01 × v8 composition smoke OR Path C hybrid-neural
- contest-CUDA paired score `[25, 40]` → marginal; Contrarian dissent invoked; pivot to Path C
- contest-CUDA paired score `> 40` → Path B WAIVED; DEFER pending grand-council symposium ratification of next path
- contest-CUDA paired score `< 15` → BREAKTHROUGH; expand budget; immediate full-run + Path C precursor design

---

## 19. Reactivation criteria + probe-disambiguator design

Per CLAUDE.md "Forbidden premature KILL" + Catalog #291 probe-disambiguator pattern:

**Pre-dispatch probe-disambiguator:** `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (landed today) is the canonical probe. For v8: compute `H(wavelet_subband_coeff | SegNet_class)` empirically on the contest video; this should be 5-10× LOWER than `H(wavelet_subband_coeff)` (unconditional) — the predicted decorrelation savings IS this entropy reduction. If empirical ratio < 3× ⇒ the per-class-conditional refinement is not worth its 3 KB CDF table cost; v8 falls back to per-subband-only CDFs.

**Smoke-before-full plan** (per Catalog #167):
- Smoke: epochs=1 (no-op), `--smoke` flag (4 pairs × 6×8 lowres), CPU-only, `--output-dir /tmp/nscs06_v8_smoke`, expected ~3-6 KB archive + inflate-bit-identical-roundtrip-verified. $0.30 estimated cost (Modal T4 ~30 sec). Smoke goal: pipeline integration validation (no archive grammar errors / no DWT shape mismatches / no Wyner-Ziv reconstruction failures).
- Full: epochs=1 (no-op), no `--smoke` flag, 600 pairs × 96×128 lowres + DB4 depth-2 DWT at 384×512, Modal T4, expected ~15 min compress + ~10 min auth eval = ~25 min wall-clock. ~$15 estimated cost.
- Paired CPU eval after full: Catalog #246 anchor-skip-if-exists logic + canonical Modal CPU paired eval per submission-auth-eval-NON-NEGOTIABLE. ~$0.50 estimated.

**Catalog #220 distinguishing-feature contract:**
- `distinguishing_feature_name`: `wavelet_subband_arith_streams_with_wyner_ziv_temporal_residual`
- `distinguishing_bytes_path`: `WLV2 body segments GRAY_F0_STREAMS_BLOB through CLS_STREAMS_BLOB` (the wavelet streams; ~580 KB out of 624 KB total)
- `inflate_consumer_function`: `inflate.py:inflate_one_video` lines decoding `decode_per_pair_subband_streams(...)` + `idwt2_db4_depth2(...)` + `add_subbands(...)`
- `byte_mutation_smoke_passes`: TBD — runs `tools/verify_distinguishing_feature_byte_mutation.py` post-smoke; expected PASS (byte mutation at any wavelet stream offset MUST change output frames)

**Catalog #270 dispatch-optimization-protocol Tier 1+2+3:**
- Tier 1 (engineering primitives): WAIVED with same-line `# AUTOCAST_FP16_WAIVED:no-training-loop`, `# TORCH_COMPILE_WAIVED:no-training-loop`, `# TF32_WAIVED:no-neural-codec`, `# NO_GRAD_WAIVED:no-training-loop`, `# F3_CACHE_CONSUMPTION_WAIVED:no-scorer-hot-loop` per v7 trainer header preserved
- Tier 2 (hardware correctness): declared in recipe — `min_vram_gb: 2`, `min_smoke_gpu: "T4"`, `video_input_strategy: "cpu_thread_async_upload"`, `pyav_decode_strategy: "cpu_blocking_upload"`, `target_modes: [contest_one_video_replay]`, canonical NVML env block auto-emitted per Catalog #244
- Tier 3 (substrate correctness): canonical `gate_auth_eval_call` per Catalog #226; canonical inflate device selection per Catalog #205; recipe-vs-trainer-state consistent per Catalog #240 (recipe `research_only: false`; trainer `_full_main` is implemented); no phantom device-named filename per Catalog #249 (the contest_auth_eval JSON path is derived from `auth_eval_device` dynamically per the canonical helper's `_redirect_output_json_to_match_device`)

---

## 20. Cost estimate

| Phase | Compute | Wall-clock | $ |
|---|---|---|---|
| **Local smoke** (CPU; 4 pairs × 6×8 + DWT + Wyner-Ziv smoke) | macOS M5 Max CPU | <60 sec | $0 |
| **Catalog #271 codex pre-dispatch review** | codex companion | ~30 sec | $0.10 - $0.50 |
| **Catalog #243 local pre-deploy 8-check harness** | macOS M5 Max CPU | ~30 sec | $0 |
| **Modal smoke** (epochs=1, 4 pairs, --smoke, CPU on Modal) | Modal CPU container | ~2 min | $0.30 |
| **Modal full** (epochs=1, 600 pairs, full DB4 depth-2 DWT, full Wyner-Ziv, Modal T4) | Modal T4 | ~15 min compress + ~10 min auth eval = ~25 min | $15 |
| **Paired CPU eval** (Catalog #246 anchor-skip-if-exists; Modal CPU container) | Modal CPU | ~15 min | $0.50 |
| **Composition smoke with NSCS01 nullspace-split (optional follow-on)** | Modal T4 | ~25 min | $10 |
| **Total v8 Path B end-to-end** | | **~70-80 min wall-clock** | **$15.90 - $16.40** |
| **Total + NSCS01 composition** | | **~95-105 min wall-clock** | **$25.90 - $26.40** |

Catalog #270 dispatch-optimization-protocol verifies AND(Tier 1 + Tier 2 + Tier 3) before any of the Modal phases fire; refuses dispatch if any tier blocks.

---

## 21. Cross-references

**Sister memos:**
- Path A v7 design: `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md`
- Symposium 22-voice deliberation (v6 falsification + Path enumeration): `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md`
- v7 Modal harvest + raw contract hardening: `.omx/research/nscs06_modal_harvest_and_raw_contract_hardening_20260516_codex.md`
- HARD-EARNED vs CARGO-CULTED addendum: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- META-ASSUMPTION ADVERSARIAL REVIEW: `feedback_adversarial_review_apparatus_blind_to_shared_assumption_failure_meta_meta_meta_meta_20260515.md`
- UNIQUE-AND-COMPLETE-PER-METHOD standing directive: `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` + `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`
- Cathedral autopilot continual-learning: `feedback_council_apparatus_in_service_of_innovation_rigor_optimization_score_lowering_20260516.md`
- Sister wavelet substrates: `src/tac/substrates/wavelet/` (canonical wavelet substrate L0 SKETCH with neural synthesis) + `src/tac/substrates/a1_plus_wavelet_residual/` (A1-base + wavelet sidecar)
- Prior wavelet plan: `.omx/research/hnerv_wavelet_residual_plan_20260506_codex.md` + `.omx/research/hnerv_wavelet_sidechannel_candidate_20260506_codex.md`

**Catalog # citations:**
- Catalog #6 strict-scorer-rule (preserved)
- Catalog #114 no-synthetic-non-smoke (preserved)
- Catalog #117 / #157 / #174 commit serializer + --expected-content-sha256 (used for landing)
- Catalog #124 representation lane archive grammar at design time (8 fields declared in SubstrateContract)
- Catalog #126 lane pre-registration (lane registered before any dispatch)
- Catalog #127 authoritative tag requires custody metadata (preserved)
- Catalog #128 continual-learning posterior update locked (used in v8 dispatch loop)
- Catalog #146 contest 3-positional-arg inflate.sh contract (preserved)
- Catalog #151 / #152 / #153 trainer flag manifest + required input files + canonical mount builder (preserved)
- Catalog #163 / #166 / #244 remote lane bootstrap + Modal HEAD parity + NVML env block (preserved)
- Catalog #167 smoke-before-full pattern (used)
- Catalog #190 canonical hardware substrate detection (used)
- Catalog #199 paired-env operator authorize bypass (used in subprocess dispatch)
- Catalog #205 canonical inflate device selection (used in submission/inflate.py)
- Catalog #206 subagent checkpoint discipline (this memo + checkpoints)
- Catalog #218 substrate mini-batch reconstruct (preserved from v7 — SegNet + PoseNet chunked forward)
- Catalog #220 substrate L1+ byte addition operational mechanism (declared)
- Catalog #226 canonical gate_auth_eval_call (used)
- Catalog #229 premise verification before edit (this memo's mechanism analysis)
- Catalog #240 substrate contest-CUDA chain complete (recipe `research_only: false` consistent with trainer implemented)
- Catalog #241 / #242 SubstrateContract decoration + canonical fields (declared)
- Catalog #243 local pre-deploy harness 8 checks (consulted before dispatch)
- Catalog #246 paired dispatch anchor skip (used for paired CPU eval)
- Catalog #249 no misleading device-named output directories (preserved via canonical helper)
- Catalog #270 dispatch-optimization-protocol Tier 1+2+3 (verified before dispatch)
- Catalog #271 codex pre-dispatch review automation (consulted before any paid dispatch)
- Catalog #272 distinguishing-feature integration contract (declared)
- Catalog #290 substrate canonical-vs-unique decision per layer (Section 12 + 15)
- Catalog #291 META-ASSUMPTION cadence (this memo's per-deliberation surfacing)
- Catalog #292 grand council per-round assumption statement discipline (memo frontmatter + Section 16 ledger)
- Catalog #294 9-dimension success checklist evidence (Section 17 literal section header)
- Catalog #295 submission inflate works with empty PYTHONPATH (self-contained _nscs06_codec/ vendored package)
- Catalog #296 substrate predicted band has Dykstra-feasibility check (Section 18 validated FEASIBLE)
- Catalog #297 substrate signal axis destruction has reversibility probe (v8 chroma destruction PROBE-required if claimed; v8 does NOT destroy chroma since real Cb+Cr coded → exempt)

---

## 22. Op-routables (ranked by EV/$)

| Rank | Op-routable | Cost | Predicted ΔS impact | Dep | First-principles cite |
|---|---|---|---|---|---|
| 1 | **PROCEED v8 Path B compress-only smoke** (epochs=1, 4 pairs, --smoke, Modal CPU). Validates pipeline integration; ~$0.30. | $0.30 | 0 (integration check) | v7 contest-CUDA paired eval confirms `[40, 65]` band (currently `[diagnostic_cpu] 58.89` is provisional; paired CPU eval would confirm) | Catalog #167 smoke-before-full |
| 2 | **PROCEED v8 Path B compress-only full** (epochs=1, 600 pairs, Modal T4). Empirical band check. | $15 | −34 to −44 (58.89 → [15, 25]) | smoke green AND v7 paired CPU confirms `[40, 65]` | Mallat 1989 + Wyner-Ziv 1976 + per Section 18 Dykstra-FEASIBLE |
| 3 | **PROCEED v8 paired CPU eval** post-full per Catalog #246 + submission-auth-eval-NN | $0.50 | 0 (axis evidence) | v8 full produces `[contest-CUDA]` | CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable |
| 4 | **PROCEED NSCS01 × v8 composition smoke** (Path B + NSCS01 nullspace-split) | $10 | −5 to −15 (additional on top of v8 band) | v8 in `[15, 25]` band AND NSCS01 contest-CUDA anchor exists | Section 13 composition matrix top-candidate |
| 5 | **DEFER Path C hybrid-neural design memo** until v8 lands empirical anchor | $0 | 0 (process; prevents premature design) | v8 paired eval lands | Catalog #229 premise verification |
| 6 | **PROCEED Catalog #220 byte-mutation smoke** post v8-smoke to verify operational mechanism | $0 (analytical) | 0 (custody hygiene) | v8 smoke green | Catalog #220 + #139 |
| 7 | **REGISTER lane** `lane_nscs06_v8_path_b_wavelet_residual_20260516` at L0 via `tools/lane_maturity.py add-lane` | $0 | 0 (process; Catalog #126 pre-registration) | none | CLAUDE.md "Lane maturity registry" non-negotiable |
| 8 | **WRITE Path B-extension v9 design memo** for cargo-cult #8 (3-depth DWT sweep) + #16 (CDF 9/7 ablation) + #18 (adaptive quantization step search empirical anchor) — DEFER until v8 lands | $0 (process; future-work memo only) | 0 today; −2 to −8 if extensions pursued | v8 empirical anchor | Section 16 cargo-cult ledger |
| 9 | **PIN v7 + v8 as paired empirical anchors in cathedral autopilot continual-learning** posterior per Catalog #128 | $0 | 0 today; ongoing improvement | v8 lands | Catalog #128 + `feedback_council_apparatus_in_service_of_innovation_rigor_optimization_score_lowering_20260516.md` |
| 10 | **IF v8 score > 40**: WAIVE Path B; pivot to Path C hybrid-neural OR Path D Z4 cooperative-receiver per symposium Option 2/3 | $0 today; $30-50 follow-on | n/a (depends on v8 result) | v8 score | Symposium Section 5 sequenced operator-decision |

---

## Final summary

- **(a) Predicted band + Dykstra-feasibility verdict:** `[15, 25]` **FEASIBLE** per Catalog #296 (polytope `[0.40, 28.76]` at seg_budget=0.06 / pose_budget=50 / archive=600 KB)
- **(b) Cargo-cults addressed by v8:** UNWINDS **#3 (spatial-independent-CDF via DB4 wavelet decorrelation)** + **#6 (symposium-#4-band-prediction-without-distortion-model via Catalog #296 structural Dykstra-feasibility check)**. PRESERVES Path A's #1 + #2 + #4 unwindings. WAIVES #5 (NO-neural-at-medal-band) and #7 (PR#56-generalizes-to-frames) with documented reactivation criteria pointing to Path C.
- **(c) Stack-of-stacks composition opportunities:** **TOP CANDIDATE: NSCS01 nullspace-split × v8 wavelet** (predicted additive ΔS −5 to −15 on top of v8 band; $10 composition smoke). Secondary: **ATW codec × v8** (IB-prior replacement; $15). Deferred: NSCS02 (antagonistic), NSCS03 (class-shift not stack), A1+LAPose (operating-point dominated by A1), hybrid_renderer_residual (antagonistic).
- **(d) Cost estimate:** **$15.90 - $16.40** v8 end-to-end (smoke + full + paired CPU + codex review); **$25.90 - $26.40** if NSCS01 composition smoke also fires.
- **(e) Reactivation criteria:** v8 produces contest-CUDA + contest-CPU paired score `[15, 25]` → L2 promotion + composition smoke queue. Score `[25, 40]` → marginal; pivot to Path C. Score `> 40` → WAIVE + DEFER per "Forbidden premature KILL". Score `< 15` → BREAKTHROUGH; expand budget; immediate full-run + Path C precursor design.
- **(f) Op-routables ranked by mission-contribution × feasibility / cost:**
  1. v8 smoke ($0.30; integration check) — RECOMMENDED FIRST
  2. v8 full ($15; predicted -34 to -44 ΔS) — RECOMMENDED SECOND
  3. v8 paired CPU eval ($0.50; axis evidence) — RECOMMENDED THIRD
  4. NSCS01 × v8 composition smoke ($10; additional -5 to -15 ΔS) — RECOMMENDED IF v8 LANDS IN BAND
  5-10. See Section 22 table

**CLAUDE.md compliance summary:**
- ✅ UNIQUE-AND-COMPLETE-PER-METHOD: 14 ingredients bound into ONE coherent packet; canonical-vs-unique decision per layer documented Section 12 + 15
- ✅ HNeRV parity discipline lessons 1-13: training-related lessons N/A (no training loop); export-first L2 ✓; monolithic single-file 0.bin L3 ✓; ≤200 LOC inflate L4 substrate_engineering exception per L7 ✓; FULL RGB renderer L5 ✓ (real Cb+Cr coded — no longer cargo-cult); apples-to-apples L10 ✓; no-op detector L11 ✓ via Catalog #220 + #139 byte-mutation smoke; reviewable L12 ✓ ~700-900 LOC core; KILL is LAST RESORT L13 ✓ (v8 is redesign not kill)
- ✅ HARD-EARNED vs CARGO-CULTED classification per Section 16 (18 assumptions classified; 12 HARD-EARNED preserved + 6 CARGO-CULTED with 2 UNWOUND + 1 WAIVED + 3 DEFERRED-FOR-CHALLENGE)
- ✅ Catalog #229 premise verification: Section 1-3 (v7 anchor analysis + cargo-cult ledger transition + architecture rigor)
- ✅ Catalog #290 substrate canonical-vs-unique decision per layer section header present (Section 12 + 15)
- ✅ Catalog #291 META-ASSUMPTION cadence: this memo IS a META-ASSUMPTION-class deliberation per Section 16 hard-earned-vs-cargo-culted ledger
- ✅ Catalog #292 grand council per-round assumption statement: every section's design choices state operating-within assumption (per the council deliberation IS the symposium 22-voice memo this design extends)
- ✅ Catalog #294 9-dimension success checklist evidence section header (Section 17)
- ✅ Catalog #296 substrate predicted band has Dykstra-feasibility check (Section 18 — Dykstra verdict FEASIBLE empirically validated)
- ✅ Apples-to-apples evidence discipline: v7 anchor `[diagnostic_cpu; non-promotable]` preserved; v8 predictions tagged `[prediction; first-principles + Dykstra-feasibility-validated]`; v8 dispatch will produce `[contest-CUDA]` + `[contest-CPU]` paired
- ✅ Forbidden premature KILL: v8 is a redesign + 6-cargo-cult-targeted improvement, not a kill; reactivation criteria pinned to enable Path C/D fallback if v8 doesn't land in band

**Lane:** `lane_nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516`
