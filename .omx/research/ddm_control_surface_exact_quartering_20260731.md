# The CONTROL SURFACE, read from source — and the exact DOF quartering

**Date:** 2026-07-31 · **Author:** MAIN · **Axis:** `[macOS-CPU advisory]`, `score_claim=false`,
`promotable=false` · **Pointer:** 0.1910828242 `[contest-CPU]` **UNMOVED** — this document measures
structure, not score. No number here is a score claim.

**Why this exists.** Operator, verbatim (2026-07-31): *"We may control more than that. Make sure to be
precise and exhaustive."* + *"Never assume anything and never only recall from working memory. We need
to be extremely rigorous and pay attention to all details. Micro to macro."* + *"They can also use
different number formats, including FP or int … and different precision levels."*

Everything below is read from `upstream/` source this session and, where numeric, verified by execution
against the real upstream functions with degenerate/generic controls in the same run. Where a thing is
DERIVED rather than MEASURED it says so.

---

## 1. THE SCORING PATH, END TO END (every line cited)

| # | step | source |
|---|---|---|
| 1 | rate = `archive.zip` bytes / Σ `rglob('*')` over `videos/` | `evaluate.py:63-65` |
| 2 | our side loaded by `TensorVideoDataset` — **`format='raw'`** | `evaluate.py:67`, `frame_utils:218-221` |
| 3 | filename = test name `.with_suffix('.raw')` | `frame_utils:84` |
| 4 | `N = file_size // frame_bytes`, `frame_bytes = 874·1164·3 = 3,052,008` | `frame_utils:227-230` |
| 5 | `np.memmap(..., dtype=np.uint8, shape=(N,H,W,C))` | `frame_utils:231` |
| 6 | GT side loaded by `AVVideoDataset`→`yuv420_to_rgb` (CPU) / `DaliVideoDataset` (CUDA) | `evaluate.py:42,58` |
| 7 | `dl = zip(dl_gt, dl_comp)` — **truncates to shorter** | `evaluate.py:71` |
| 8 | `rearrange 'b t h w c -> b t c h w'` + **`.float()`, NO /255** | `modules.py:145` |
| 9 | PoseNet: `interpolate→(384,512)` **then** `rgb_to_yuv6`; normalize `(x−127.5)/63.75` | `modules.py:73-74, 77` |
| 10 | SegNet: **`x[:, -1, ...]`** (frame_1 only) then same interpolate; **no normalization at all** | `modules.py:107-109` |
| 11 | d_seg = mean argmax disagreement; d_pose = MSE on first 6 of 12 | `modules.py:112-113, 84` |
| 12 | `S = 100·d_seg + √(10·d_pose) + 25·rate` | `evaluate.py:92` |

**Two corrections to things I had been carrying loosely.** (a) The GT and our side use **different
loaders** but **identical preprocessing** (`DistortionNet.preprocess_input`, `modules.py:143-148`, is
applied to both). The asymmetry is that GT values live on the BT.601-from-YUV420 output set while ours
are arbitrary uint8. (b) The resize precedes yuv6 for pose and is the *same* interpolate SegNet uses —
so both scorers factor through one shared `z ∈ ℝ^(384×512×3)`.

## 2. WHAT WE CONTROL — exhaustive

1. **`archive.zip` bytes.** The only sized artifact. `inflate.sh`/`inflate.py`/runtime tree are unsized.
2. **The full uint8 cube per frame:** 874×1164×3 = 3,052,008 independent values, **unconstrained** — no
   codec, no YUV420 subsampling, no BT.601 round-trip. This is *more* freedom than the GT decoder has.
3. **Number format and precision everywhere upstream of the final cast — as a PER-ELEMENT variable.**
   The only fixed numeric interface in the chain is `dtype=np.uint8` at `frame_utils:231`. Upstream of
   that last cast nothing is constrained. Three levels, each strictly more general than the last:
   - *formats*: fp64/fp32/fp16/bf16, int8/16/32, fixed-point, block-FP, mixed.
   - *sub-byte*: the archive is a byte STRING and the receiver is free code, so there is **no alignment
     constraint** — int5/int4/int3/ternary/binary and arbitrary bit-packing are all available, and
     **different elements may carry different widths**. Already in our lineage: L38 nibble + 3-bit
     Huffman-length sub-byte sidecars, `experiments/block_fp_int4_codec_sketch.py`,
     `src/tac/experiments/benchmark_int4.py`, `repack_pr106_with_int4_block_fp.py`, our own HiNeRV
     `int4_mixed` runs, and task #147 (int5 per-channel + LSQ + outlier handling).
   - *the real object*: with entropy coding (Brotli/LZMA/SMEVR/arithmetic, all already in the stack) the
     **effective rate per element is a code length — a positive REAL, not an integer bit count.** So
     "int5" is one point on a continuum. **DERIVED: bit-width is a discrete proxy for a continuous
     per-element rate allocation `r_i ≥ 0`**, chosen by reverse-waterfilling against per-element
     sensitivity, priced against `W = 1.2731082153320312` B/flip. This is task #157's KKT bit-allocation
     object, and #336 is its apply-pass.

   **Unification with §3 — the quartering IS the sensitivity structure the allocation waterfills
   against, and the four quarters are four DIFFERENT allocation problems:**

   | quarter | sensitivity | optimal allocation | character |
   |---|---|---|---|
   | Q1 | **exactly 0 to both** | **r = 0 — do not transmit at all** | degenerate (this is #401's blind fill) |
   | Q2 | pose only | reverse-waterfill on a smooth quadratic | classic R-D, no plateau |
   | Q3 | seg only (**pose exactly 0**) | **margin-thresholded**, not smooth | argmax plateau ⇒ r=0 suffices below margin |
   | Q4 | both | joint allocation | the only coupled case |

   So "different precision for different tokens" is not a heuristic here — **which allocation law
   applies is DERIVED from which quarter the element occupies**, and two of the four quarters have an
   exactly-zero sensitivity direction, where the allocation is free rather than merely cheap.
4. **Frame count N** — inferred from our file size, integer division (trailing partial-frame bytes are
   silently ignored). See §5: hazard, not lever.
5. **Per-pair DOF placement across the four quarters of §3.**

**What we do NOT control:** GT videos; the rate denominator (guarded, task #812); the frozen scorers;
the resize; the yuv6 map; `seq_len=2`; `camera_size`; the eval device and its numerics.

## 3. THE EXACT QUARTERING (MEASURED against the real functions, with controls)

`rgb_to_yuv6` (`frame_utils:51-78`) keeps **all four luma phases** (`y00,y10,y01,y11`, no averaging) and
**box-means chroma 2×2** (`×0.25`, non-overlapping). So per 2×2 block of `z`, 12 RGB DOF in → PoseNet
sees 6 → a **6-dim exact linear kernel per block**. SegNet sees only frame_1 (`modules.py:108`).

Pair has 2·3·384·512 = **1,179,648** z-DOF; blocks/frame = 192·256 = 49,152; kernel = 6·49,152 = 294,912.

| quarter | dim | reaches | verification (max abs Δ at the scorer input) |
|---|---:|---|---|
| Q1 frame_0 yuv6-null | 294,912 | **NEITHER** | pose 5.684e-14 · seg 0.000e+00 |
| Q2 frame_0 non-null | 294,912 | POSE only | pose 1.940e+01 · seg 0.000e+00 |
| **Q3 frame_1 yuv6-null** | **294,912** | **SEG only** | pose **5.684e-14** · seg **6.000e+00** |
| Q4 frame_1 non-null | 294,912 | BOTH | pose 2.015e+01 · seg 2.998e+01 |

5.684e-14 is float64 machine-eps at this magnitude for a perturbation of **RMS 1.47/255**; the generic
same-norm controls in the same run land at 4.855 (luma) / 2.114 (chroma). Exactly 25.0% each.

### 3b. THE QUARTERING IS A DIMENSION COUNT — the coarsest cut through a graded spectrum

Operator, 2026-07-31: *"Much of your framing has been too coarse."* §3 counts DIMENSIONS. That is the
crudest measure of a subspace — a 25%-dimensional subspace can carry 1% or 99% of the sensitivity mass.
Sister precedent: #580 replaced a **22.6969% axis-aligned dimension count** with **80.6742% real-linear
nullity** — the coarse instrument under-counted by 3.55×.

Exact SVD of the per-block yuv6 map (12→6), computed this session:

| σ₀ | σ₁₋₃ | σ₄ | σ₅ | cond | naive 6-of-12 energy | **actual `tr(MᵀM)/12`** |
|---|---|---|---|---|---|---|
| 0.691312 | 0.668555 ×3 | **0.321916** | **0.265430** | **2.6045** | 50.00% | **0.166074 = 0.3321×** |

Two consequences the quartering hides:
1. **The dimension read overstates pose reach by ~3× in energy.** The map is anisotropic (cond 2.60).
2. **The pose-VISIBLE half is itself graded.** σ₀–σ₃ are the four LUMA directions (luma content
   1.02–1.33); σ₄,σ₅ are the CHROMA directions (luma content **0.037, 0.143**). So the truth is a
   **three-tier ladder, not a binary**:
   `luma → pose gain ~0.67 · chroma-mean → ~0.27–0.32 (2.1–2.6× ATTENUATED) · chroma-zero-mean → EXACTLY 0`
   Q3 is the **exact-zero level set** of that ladder — real and exact, but one cut through a continuum.

3. **And `tr(MᵀM)` is ITSELF coarse.** It is an L2-energy read, which our own g3 instrument explicitly
   rejects (*"flip/margin-weighted, never L2 energy"*). The honest instrument ladder is:
   **dimension count (§3) → L2 spectrum (here) → margin/Fisher-weighted measure (THE object, UNMEASURED)**.
   Any Q3 allocation decision must be made on rung 3, not rung 1. The ms3/ms4 metric-custody bundle is
   the existing producer for rung 3.

**Q3 is the object.** A correction placed in Q3 costs **exactly zero** d_pose — a linear kernel, not a
suppression ratio, so it holds at any amplitude and any base. This reframes `cb1`'s measured
**+22.7 d_pose from one Lane repaint**: that was the price of editing in **Q4**, not evidence that
corrections are pose-vetoed.

## 4. THE HONEST LIMITS ON Q3 (both open, both measurable)

1. **Reaching SegNet's input ≠ moving its argmax.** Q3 arrives with amplitude 6.0/255; whether the
   argmax moves is the open measurement. Mechanically it is not excluded: SegNet's stem is a *learned
   overlapping 3×3 stride-2 conv*, whose kernel is unrelated to the *uniform non-overlapping 2×2 box
   mean* that defines Q3. (An earlier reflex — "Q3 is sub-2px structure, therefore seg-blind" — merged
   two different operators and is withdrawn.)
2. **Realizability.** We control camera-res uint8, not `z`. Since bilinear resize `R` is per-channel and
   linear, luma commutes: `δY(z) = R(δY_cam)`, so Q3 at camera res requires `δY_cam ∈ ker(R)` (#580
   measured 80.6742% resize nullity) plus zero block-mean on resized chroma. #532 measured that uint8
   breaks range(R) exactness (Δ=62.74 vs 1.7e-13). **The uint8-reachable fraction of Q3 is UNMEASURED.**
   A lower bound that needs no ker(R) at all: the best *pointwise*-isoluminant integer step within ±12
   is `(−6,+5,−10)` — `|δchroma| = 7.0834` at `δY = +0.001`, **leverage 7083:1**. Enumeration also shows
   **no exact-zero-luma integer step exists** in ±12 (min |δY| = exactly 1/1000, since 299a+587b+114c
   is an integer). DERIVED, exact arithmetic.

## 5. COMPLIANCE HAZARD — defend, do not exploit

`N = file_size // frame_bytes` + `zip()` truncation + `batch_sizes` counting only zipped batches means a
`.raw` short by one partial batch passes the `:78` shape assert and scores over **fewer samples**. We do
not do this. It is recorded here for two reasons: it is the same class as the #812 rate-denominator
guard, and it is a **self-protection need** — our exporter must emit exactly N frames or we silently
mis-score ourselves, and trailing bytes are silently absorbed by the integer division.

## 6. PRECISION FACTS (verified from the safetensors headers this session)

- `segnet.safetensors` 38,502,892 B — 484×**F32** + 78×I64 (BN `num_batches_tracked`).
- `posenet.safetensors` 55,835,560 B — 510×**F32**.
- Eval constructs at default **fp32**, no autocast (`evaluate.py:52`), under `inference_mode` (`:73`).
- `modules.py:152` carries the upstream author's `# TODO run in bfloat16?` — eval dtype was considered
  variable by its own authors; `--device` (`:16`) additionally varies kernels/accumulation order.

**Coupling to Q3, and it is binding:** d_seg is an **argmax** (precision-robust except inside ties);
d_pose is **MSE on continuous outputs** (precision-sensitive everywhere). A Q3 edit placed at a near-tie
margin is therefore not a score gain but a device-dependent coin flip. **Any Q3 correction must carry a
margin floor**, which is the same discipline as the CPU/CUDA-axis separation, applied at design time
rather than at reporting time.

## 7. PLACEMENT AGAINST THE RECORD (recall before claim)

- `frozen_scorer_exact_factorization_20260715.md:57` already derived *"PoseNet is chroma-blind below
  2px … a fine-scale boundary-RGB carry is pose-safe by construction."* **This is not new.** It was
  filed as a **rate/pose-safety note** (row B4, *"stated as rate"*) rather than as a seg actuator, and
  MEMORY.md compresses it to `"chroma <2px INVISIBLE"`, which reads as *invisible to both*. The signal
  was mis-homed, not missing.
- `ddm_j11r_366_g1_verdict_memo_20260727.md:38-50` measured pose-null/seg-null projectors as exactly
  `[0]` — but on **four sealed 1-D rays in receiver-parameter space W against learned Jacobians**,
  where a rank-1 map forces nullity 0 structurally. That memo scopes itself INSTANCE and states:
  *"NOT a family negative for higher-dimensional proposal families … that family is open."* **Q3 is
  that open family**, and is stronger: an exact algebraic kernel of a fixed map, independent of base,
  linearization, and amplitude.
- Sister realization constraint: `erf_collateral_law_no_posthoc_injection_on_textured_renders_20260731`
  — post-hoc injection on textured renders measured net-worse (+0.30 S, ~85px ERF). Q3 removes the
  **pose** veto; it says nothing about ERF collateral on the seg side. Both must be paid.

## 8. THE ONE MEASUREMENT THIS NAMES

Perturb frame_1 within Q3 ∩ range(R) ∩ uint8-realizable, at a margin floor derived from §6, and measure
**Δd_seg through the real R→uint8→SegNet path** (and confirm Δd_pose = 0 to machine precision as a
positive control). Both outcomes are informative: nonzero Δd_seg = a seg actuator at exactly zero pose
cost, which is the direct answer to the campaign's binding pose-veto problem; zero Δd_seg = Q3 is shared
blind space, which closes "chroma as a free seg actuator" with a mechanism instead of a magnitude.

**Sisters:** `frozen_scorer_exact_factorization_20260715.md` · `ddm_j11r_366_g1_verdict_memo_20260727.md`
· #580 (resize nullity) · #532 (uint8 breaks exactness) · #401 (blind-coordinate fill — Q1's consumer) ·
#812 (rate-denominator guard — §5's sister) · `ddm_surface_correction_economics_20260731.md`.
