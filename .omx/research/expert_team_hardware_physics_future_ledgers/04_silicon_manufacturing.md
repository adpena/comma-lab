# Ledger 04 — Silicon manufacturing lineage (2026-05-13)

**Lane:** `lane_expert_team_hardware_physics_future_alien_tech_20260513`.
**Persona:** TSMC 3 nm / 2 nm process engineers (Hsinchu Science Park, Taiwan), Intel Ribbon-FET / PowerVia teams (Hillsboro OR), GlobalFoundries 12LP+ analog-mixed-signal designers (East Fishkill NY / Malta NY). Mythic Inc analog-AI accelerator designers (Austin TX), IBM analog AI 14nm chip team (HERMES Project Chip, Khaddam-Aljameh et al., *Nature Electronics* 2022). We design **compression at the physical layer**.
**Mode:** READ-ONLY engineering-analog derivation. `research_only=true`. NO archive bytes mutated.
**Evidence:** `[engineering-analog]`, `[hardware-derivation]`, `[mathematical-derivation]`.

---

## 0. The silicon-physical-compression frame

Compression in semiconductor physics is **everywhere**, just not labelled as such:
- **DRAM refresh** optimization: trade refresh rate for retention error rate (information storage with thermodynamic decay).
- **NAND wear leveling**: distribute writes to maximize total cell lifetime (entropy of write pattern).
- **Analog crossbar memory**: multiply-accumulate in analog (low-precision representations).
- **Photonic interconnect**: serialize weights as optical pulses (temporal+spatial compression in physical layer).
- **3D stacking** (HBM, V-NAND): vertical interpolation = hierarchical compression.

Each maps to a contest analog. References:
- Khaddam-Aljameh et al., *Nature Electronics* 2022 ("HERMES core") — 14nm analog AI inference.
- Reuther et al., 2020, *IEEE HPEC* — AI accelerator survey covering Mythic, IBM, Tenstorrent.
- Lightmatter Inc. white papers (2022-2024) — silicon-photonic AI compute.

---

## 1. DRAM-refresh analog: temporal redundancy in static frames

### 1.1 Background

DRAM cells lose charge in ~64 ms; **refresh keeps the weakest cell alive**. Refresh rate is set by the weakest cell, not the average — most cells over-refresh. Recent research (Liu et al., ISCA 2012, "RAIDR") exploits this: profile cells, refresh at per-cell rate, save ~75% of refresh energy.

### 1.2 Contest analog

The contest video has substantial **temporal-static segments**: vehicle stopped at traffic light (~5-10 sec at a time), gentle highway cruise (low frame-to-frame change). Currently the renderer codes each frame independently or with neural latent-spatial sharing only.

**Per-frame I/P/B-frame video-codec compression** (H.265, AV1) achieves 50-90% compression on highway driving footage purely from temporal redundancy. We don't have HEVC's 1000s of LOC of inflate.py, but a **minimal P-frame coder** (motion vector field + DCT residual + boundary handling) might fit in 40-60 LOC.

### 1.3 Implementation sketch

```python
# Inflate-time (~50 LOC)
def decode_video_with_p_frames(i_frames_bytes, motion_fields_bytes, residual_bytes):
    """I-frames every Nth frame; P-frames = motion-compensated residual.
       i_frames: every 12th frame, full RGB.
       motion_fields: per-pair (yaw_dx, yaw_dy) shift per 16×16 block.
       residuals: ε-bounded per-pixel correction.
    """
    decoded = []
    for t in range(1200):
        if t % 12 == 0:
            frame = decode_i_frame(i_frames_bytes[t // 12])  # full reference
        else:
            mv = decode_motion(motion_fields_bytes[t])        # ~50 vectors
            res = decode_residual(residual_bytes[t])          # small
            frame = warp(decoded[-1], mv) + res
        decoded.append(frame)
    return decoded
```

### 1.4 Bit budget

- I-frames every 12th: 100 frames × ~3 KB each = 300 KB. **Too big.**
- I-frames every 60th: 20 frames × ~3 KB = 60 KB. **Tractable.**
- Motion fields: 1199 × ~80 bytes = 96 KB. **Tractable** if motion fields are compressed (most blocks have zero motion).
- Residuals: 1199 × ~30 bytes = 36 KB.

**Total: ~192 KB.** Smaller than PR101's 229 KB but not transformative. The win comes from **specific segments with high temporal redundancy** — the static-vehicle segments compress to near-zero bytes.

### 1.5 Score-impact prediction

Speculative; depends heavily on the specific video's temporal redundancy. **20-30% rate savings** on typical driving footage → -0.0005 to -0.0010 [hardware-derivation, literature-prediction].

### 1.6 Reactivation

Register as `lane_p_frame_temporal_codec` at L0 SKETCH. Substrate-engineering tier. Reactivation requires:
- Empirical measurement of inter-frame redundancy in the contest video (~1 day).
- Council review per CLAUDE.md "Design decisions" + HNeRV parity lesson 4 (≤100 LOC inflate budget) — 50 LOC is at the boundary.
- Cross-link with `lane_pr106_latent_sidecar_topk_pareto_20260513_codex` (motion-field flavor).

---

## 2. NAND wear-leveling analog: balanced rate across bit slots

### 2.1 Background

NAND flash distributes writes across blocks to maximize total cell-erase lifetime. Concentrated writes destroy specific cells. Wear-leveling = **entropy-of-writes maximized over the cell array**.

### 2.2 Contest analog (positive — pack ZIP-overhead with payload bits)

A ZIP archive has structural overhead:
- ZIP local file header: ~30 bytes
- ZIP central directory header: ~46 bytes/file
- ZIP end-of-central-directory: ~22 bytes
- Brotli stream framing: ~3-6 bytes
- Optional ZIP extra-field (per-file): up to 65,535 bytes
- Optional ZIP comment (per-file): up to 65,535 bytes
- Optional ZIP archive-comment: up to 65,535 bytes

Most fields are constrained by the ZIP spec but not fully fixed. **Choose the constrained fields to carry useful content.**

### 2.3 Implementation sketch

Pack ~200-400 bytes of pose-residual stream into:
- ZIP archive-comment (up to ~65 KB allowed; some parsers truncate at 64 KB)
- Per-file extra-field (some parsers honor it, some skip it)
- File-name field (filenames can encode bits if the contest parser only checks for exact match against `file_list`)

### 2.4 Bit budget

Recovered 100-400 bytes per archive. **Free bytes, no algorithmic complexity.**

### 2.5 Score-impact prediction

100-400 bytes / 37.5 MB = -0.00003 to -0.00011 score. **Small but free.** [engineering-analog, hardware-derivation]

### 2.6 Caveats and coordination

- Sister codex memo `packetir_pr106_identity_zip_comment_20260513_codex.md` already explores ZIP-comment field. **Coordinate**: pick one canonical implementation.
- Some ZIP parsers (the Python `zipfile` module, most language stdlibs) DO honor extra-fields. Test against the contest's exact `inflate.sh` runtime.
- Filename-based payload encoding is risky — contest contract specifies file_list. Don't pursue.

---

## 3. Analog crossbar MAC: low-bit-width weight quantization

### 3.1 Background

IBM HERMES analog AI chip (Khaddam-Aljameh 2022): performs matrix-vector multiplication in analog via phase-change memory crossbar array. **Weights stored at 4 bits of precision** (16 conductance levels per cell); achieves ImageNet 76% top-1 vs FP32's 78%.

Mythic Inc analog-compute: 8-bit weights, full-network analog dataflow, 25 TOPS at 25 mW.

### 3.2 Contest application

The contest's existing CY-class lane work has tried 3-4 bit quantization on the renderer. Sister memos document:
- 4-bit FP4+Brotli renderer (Quantizr's PR — produces 0.33 score).
- Track-4 uniward-STC-Hessian 3-bit attempt **FALSIFIED** at -0.0058 score regression on `[contest-CPU]` because **rel_err² is anti-correlated with score-gradient saliency on score-aware-trained substrates** (CLAUDE.md Catalog #123).

The **correct discipline** for sub-4-bit quantization:
1. Use **score-gradient saliency** (Catalog #123 `check_no_weight_domain_saliency_on_score_gradient_substrate`).
2. Use **outlier-aware quantization** (LLM.int8, AWQ patterns).
3. Use **per-channel scales** (not per-tensor).
4. Use **QAT** (quantization-aware training), not PTQ.
5. Use **lookup-table-aware allocation** (some weights map to lookup table indices, others to raw bits).

### 3.3 Bit budget

229K params at:
- INT8: 229 KB
- INT4: 115 KB
- E4M3 FP8 (NVIDIA H100 native): 229 KB
- INT3 with outlier handling: 86 KB
- INT2 with outlier handling: 57 KB
- 1.58-bit ternary (BitNet b1.58): 45 KB

Theoretical floor (Shannon entropy of weight distribution, typically ~3-4 bits/weight after distillation): ~90-115 KB.

### 3.4 Score-impact prediction

**INT4 with proper QAT (sister to Quantizr 0.33 work):** already on the Pareto frontier.

**E4M3 FP8 (NVIDIA H100 native):** same byte budget as INT8 but higher dynamic range. ~0.5 dB better reconstruction at zero rate cost. -0.00015 to -0.00030.

**INT3 / INT2 with score-gradient saliency:** speculative; the Catalog #123 anti-correlation result is a warning. **If** the score-gradient saliency is computed correctly on a substrate that was NOT trained against score, INT3 could land 86 KB / 37.5 MB = -0.0023 savings; distortion ≤ 1 dB if outlier handling is solid → net -0.0010 to -0.0020. **Highly speculative** [hardware-derivation, literature-prediction].

**1.58-bit ternary BitNet b1.58:** even more speculative; requires both score-gradient saliency AND a substrate that survives ternary representation. If it works, -0.0018 to -0.0035. **Substrate-engineering tier.**

### 3.5 Reactivation

Register the 4 sub-4-bit variants as separate L0 SKETCH lanes:
- `lane_e4m3_fp8_renderer` (easy, ~1 week)
- `lane_int3_renderer_with_outlier_handling` (medium, ~2 weeks)
- `lane_int2_renderer_with_outlier_handling` (hard, ~3 weeks)
- `lane_bitnet_158_ternary_renderer` (hard, ~3 weeks)

All require Catalog #123 score-gradient-saliency discipline as a precondition. None proceed without operator approval + grand-council review.

---

## 4. Photonic interconnect: temporal serialization of weights

### 4.1 Background

Lightmatter Mars (silicon photonic AI accelerator) serializes weights as **optical pulses over time** rather than storing them as spatially-distributed transistors. A 100×100 weight matrix becomes a 10,000-element temporal pulse train.

In information-theoretic terms, **photonic interconnect trades spatial extent for temporal extent**. The total information is conserved (Bekenstein bound), but the **physical instantiation** shifts from area to time.

### 4.2 Contest analog

Most current archives store renderer weights as a **spatial grid** (Conv2D weights in NHWC layout). A **temporal serialization** would store the weights as a 1D autocorrelated sequence — passed through an LZ77/Brotli/zstd entropy coder.

The key question: is the **temporal serialization** more compressible than the **spatial serialization**?

For a trained Conv2D layer:
- **Spatial:** weights vary smoothly across the spatial dim (correlated within a kernel).
- **Temporal (channel-major order):** weights vary fast across out_channels (less correlated).
- **Temporal (kernel-major order):** weights vary slowly within a kernel, fast across kernels.

**The right ordering depends on the layer.** Test empirically.

### 4.3 Bit budget

If the temporal-ordered weight stream compresses 5-10% better than the spatial-ordered stream after entropy coding, savings ~5-12 KB per archive.

### 4.4 Score-impact prediction

5-12 KB / 37.5 MB = -0.00013 to -0.00032 score. **Small but free** (just a reordering of bytes, no algorithmic change). [hardware-derivation, mathematical-derivation]

### 4.5 Reactivation

Easy to test (just permute weight bytes before Brotli encoding, compare compressed sizes). Register as `lane_temporal_weight_ordering` at L0 SKETCH. **Recommend testing this week.**

---

## 5. 3D-stacked memory analog: hierarchical compression

### 5.1 Background

HBM (High-Bandwidth Memory, Samsung/SK Hynix) and V-NAND (vertically-stacked NAND flash, Samsung) use **3D vertical stacking** of memory cells. Vertical proximity = high inter-layer bandwidth + locality. Hierarchical compression schemes (e.g., per-layer base + delta) exploit this.

### 5.2 Contest analog

Currently the renderer's per-pair latents are stored as a **flat sequence**. A **hierarchical encoding** (base latent shared across multiple pairs + per-pair delta) exploits inter-pair correlation.

This is essentially **clustered latents**: K cluster centroids + per-pair index + per-pair correction. PR101's archive already does some of this implicitly (the trained model output is structured), but explicit clustering of post-trained latents could squeeze further bytes.

### 5.3 Bit budget

For 1199 pairs × 70 bytes/pair = 84 KB latent stream:
- K=8 cluster centroids × 70 bytes = 560 bytes
- 1199 × 3 bits (cluster index) = 450 bytes
- 1199 × 30 bytes (correction) = 36 KB
- **Total: ~37 KB.** Savings 47 KB.

### 5.4 Score-impact prediction

47 KB savings = -0.0012. Speculative; the correction term might need to be larger if clusters don't match well. [hardware-derivation, literature-prediction]

### 5.5 Reactivation

Register as `lane_hierarchical_clustered_latents` at L0 SKETCH. Medium effort; cross-link with codex's pose-codec primitives that already explore similar ideas.

---

## 6. Status / cross-references / next steps

- **Companion:** master memo `expert_team_hardware_physics_future_alien_tech_20260513.md`.
- **Active codex work:**
  - `packetir_pr106_identity_zip_comment_20260513_codex.md` — overlaps §2 NAND wear-leveling.
  - Pose codec primitives PR63/PR64/PR65 (codex's recent landing) — overlap §1 (P-frame motion-codec analog) and §5 (clustered latents).
- **Wire-in hooks** declared in master memo §9.
- **Sister memos:** NASA §1 (CCSDS wavelet), NASA §3 (PASS-AI prior) overlap conceptually with §1 / §5 here.
- **Highest-value test this week:** §4 temporal weight ordering — cheap to validate, may surface 5-12 KB.

**Per CLAUDE.md "KILL is LAST RESORT":** all techniques DEFER-pending-research with explicit reactivation criteria.
