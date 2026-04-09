# speculative lanes

These lanes are explicitly secondary unless evidence forces promotion.

## 2026-04-09 — Mathematical investigation results (new evidence)

A sequence of closed-form and spectral experiments has dramatically updated
what we understand about the problem. The 1.845 CNN floor is NOT "the model
that happened to train well" — it is the unique solution class that works
given how PoseNet is structured.

### Jacobian single-step — FALSIFIED

- `experiments/jacobian_optimal.py`: Moore-Penrose minimum-norm delta fails
  catastrophically. Applying the closed-form optimal correction under the
  linear model pushed pose from 0.074 to 0.235 (3x WORSE).
- Root cause measured in `experiments/trust_region_sweep.py`:
  **PoseNet's honest linear trust radius is ~0.0001 pixels**. Even at
  alpha=0.0001 the relative linearization error is 80%.
- Implication: **any single-step or Newton-style closed-form method is dead
  on arrival**. Iterative Newton would need 10^4 steps to move 1 pixel.
  Only SGD with implicit learned priors navigates this terrain.

### Jacobian rank is ~1 — SURPRISING

- `experiments/jacobian_svd_analysis.py`: per-pair J = dPose/dPixel has
  effective rank ~1.008 across 30 sampled pairs.
- Top singular value is 45x larger than the second.
- 98% of PoseNet's pixel sensitivity lies along ONE direction per frame.
- Condition number ~399 explains the error amplification.
- **This means PoseNet's 6-dim output is effectively 1-dim at our operating
  point** — the CNN is solving a scalar regression problem.

### CNN residual characterization — VALIDATES KARPATHY

- `experiments/karpathy_cnn_residual_analysis.py`:
  - CNN moves 56.6% of pixels (Jacobian moves 0.0024%) — 24,000x denser
  - CNN mean |δ| = 0.83 (Jacobian 0.0044) — 189x larger total mass
  - **CNN puts 90.3% of residual energy in mid-frequency band** — a stunning
    concentration that matches PoseNet's early-conv response bandwidth
  - Jacobian is roughly uniform across bands
- Karpathy's inverse-rendering hypothesis is validated on 4 of 5 predictions.
- The CNN spreads corrections densely in mid-frequency space, which is the
  only strategy that stays inside every ReLU region it touches.

### Next experimental priorities (panel + insights)

1. **2D-DCT basis filter** (strongest empirical backing) — explicit
   parametrization of the mid-frequency structure the CNN learned implicitly
2. **CMA-ES gradient-free finetuning** on the 1.845 winner — gradient-free
   search to escape the current basin, sidesteps Jacobian ill-conditioning
3. **Rate-distortion information bound via MINE** — research-grade provable
   minimum pose distortion given current archive bytes. Tells us when to
   stop grinding pose and pivot to SegNet
4. **SegNet direct attack** already in flight — Tao's math says SegNet is
   14x more leveraged than PoseNet per unit distortion

### PixelShuffle+Dilated hybrid (HIGHEST priority — COUNCIL CONSENSUS)

The full expert council (Tao, LeCun, Karpathy, Collier, Jensen, Von Neumann)
unanimously selected this as the single best experiment to run next.

Architecture: PixelUnshuffle(2) → conv1(3x3) → conv2(3x3,**dilation=2**) → 
conv3(3x3) → conv4(3x3) → PixelShuffle(2). Same params as plain PixelShuffle
but with 24×24 effective RF at full-res.

**Tao's projection**: seg=0.004 + pose=0.035 + rate_penalty=0.060 → **score 1.583**
**Conservative**: 1.62. **Optimistic**: 1.56.
**Training launched** as `psd_h64_long1000`.

### Pair-aware 6-channel input (HIGH priority — DEEP RESEARCH #1)

PoseNet scores PAIRS of frames but our filter processes each frame independently.
Feed frame t AND frame t-1 as 6-channel input. Expected delta: -0.05 to -0.15.
The single highest-EV change from the transdisciplinary research. Queued.

### LSQ Learned Step Size Quantization (MEDIUM priority — DEEP RESEARCH #2)

Make the int8 scale a learnable parameter instead of fixed max/127. 5-line change.
Expected: -0.05 to -0.12 from closing quantization gap. Queued.

### SegNet-biased saliency rebalancing (MEDIUM priority — DEEP RESEARCH #3)

Current alpha=20 may over-allocate to PoseNet. Sweep alpha={5,10} on PSD arch.
SegNet has 11.5× leverage with 98.4% headroom. Lower alpha redirects CNN budget.

### PixelShuffle half-resolution processing (HIGH priority)

PixelUnshuffle(2) converts 3-channel full-resolution (1164×874) to 12-channel
half-resolution (582×437), processes with 4 conv layers, then PixelShuffle(2)
reconstructs full-res. Each 3×3 conv at half-res covers 6×6 at full-res,
matching the scorer models' internal resolution. This alignment is why a similar
architecture in the field achieved SegNet distortion 0.00434 — 24% below our
current 0.00576.

Combined with our proven QAT+EMA+saliency+best-checkpoint training recipe,
this architecture could deliver seg≈0.004 while preserving our pose≈0.033,
yielding estimated score **~1.58**. Training launched as `pixelshuffle_h64_long1000`.

88K params at h=64 (~90KB int8) — rate penalty +0.06. Net gain if SegNet
drops to 0.004: -0.116 → score ~1.61.

### Dead lanes (do not resurrect without new evidence)

- **Single-step closed-form pixel correction** — trust radius killed it
- **Test-time Newton or trust-region methods** — step size would need to be
  under 0.0001 pixel, making 10^4+ steps required to move 1 LSB
- **Higher-order Taylor expansion** — Hessian is just as local as Jacobian
- **Karpathy closed-form LUT alongside CNN** — original suggestion falsified

## JAX

Good for fast local surrogate training, differentiable parameter search, vectorized experiment code, and batched teacher-cache analysis.
Not the first choice for the shipped inflator.

## Mojo

Potentially useful for performance experiments or tiny kernels, but not a required dependency.
Treat it as a bounded side lane.

## WASM in model weights / generative overfit codec

Interesting in principle, but high-risk for one month of work.
Main dangers:

- bytes shift from the archive into model artifacts
- harder packaging and reproducibility
- higher runtime uncertainty
- easier to fool yourself with a flashy idea that does not beat a simpler metric-aware codec

Only explore this if the mainline stalls or if a tiny scene-specific model clearly outperforms the safer hybrid path.

## TurboQuant-style ideas

Potentially relevant as inspiration for:

- quantizing tiny side models
- compressing per-frame latents or codebooks
- lowering memory use in surrogate runs

Not a direct mainline solution to the challenge by itself.

## Cross-platform runtime optimization lane

Potentially useful after the codec-only sweeps flatten out.

Candidate ingredients:

- Rust-side hot-path work in `runtime-rs/`
- cross-platform SIMD kernels
- Rayon parallelism for CPU-bound preprocessing or decode-side work
- ffmpeg pipeline tightening where it clearly reduces wall-clock cost without adding byte burden
- end-to-end pipeline profiling to remove avoidable copies, stalls, and format churn

Why this is still speculative now:

- the current highest-leverage unknown is still codec operating point, not runtime
- runtime work is easier to overbuild before the compression floor is stable
- any heavier lane must earn its complexity with measured wall-clock wins and no hidden byte cost

Promotion rule:

Only promote this lane after the codec-only queue is measured and a real runtime bottleneck shows up in scorer-facing runs.

### Current availability note

bat00 is now reachable over SSH at `192.168.1.216` as a Windows 11 Pro box for speculative side work.
Use it for:

- runtime profiling
- CUDA-adjacent scouting if a compatible path is present
- remote throughput experiments that do not replace official local CPU claims

Do **not** use bat00 results as the sole basis for claimed competition scores unless the exact official-style eval path is reproduced and clearly labeled.

### Approved bat00 sequence

- First: runtime profiling benchmark suite on the currently promoted floor.
- Second: JAX/CUDA surrogate setup for research-time ranking and analysis.
- Keep both explicitly non-authoritative unless they reproduce the official-style path and are labeled accordingly.

## Segmentation-guided two-pass lane

Potentially high-upside if the scorer keeps rewarding semantic preservation more than pixel fidelity.

### Refined 2026-04-05 direction

The next experiment in this lane is **dynamic main ROI first**, not generic segmentation for its own sake.

Guiding rules:

- the **main ROI** remains mandatory and central
- any auxiliary ROI is additive only
- dynamic/staticness or learned analysis should stay on the **compression side** first
- do not ship heavy model weights in the inflate path unless the measured gain clearly justifies the byte/runtime burden
- use `uv` for Python tooling and keep BAT00 explicitly non-authoritative for score claims


Potentially high-upside if the scorer keeps rewarding semantic preservation more than pixel fidelity.

Idea:

- use a first pass to separate semantically critical regions from less-critical regions
- allocate more bitrate or a cleaner reconstruction path to the critical mask
- compress the complementary region more aggressively
- possibly stabilize/aggregate the critical mask across time instead of treating each frame independently

Why it might fit this competition:

- the scorer is based on SegNet and PoseNet distortions, not pure PSNR/SSIM
- dashcam data often contains semantically important foreground structure (road edges, lane markings, cars, signs) mixed with less-critical texture/background detail
- a two-pass or ROI-aware codec could outperform uniform treatment if the byte overhead stays small

Why it stays speculative now:

- the mask source itself has byte/runtime cost
- naive masking can introduce hard edges or temporal inconsistency that hurt the scorer
- the current mainline still has cheaper codec-only lanes available

Promotion rule:

Only promote after a tiny prototype shows scorer gains that clearly beat the simpler uniform-codec path at similar byte burden.

### 2026-04-05 measured status

The first executed dynamic main-ROI prototype was rejected at `4.47` with about `2.66 MB` current-workflow bytes.

What this means:

- keeping the main ROI explicit was **not** enough by itself
- multi-window / multi-stream overhead is currently too expensive
- do not revisit this lane unless the side-information and protected-stream burden gets much cheaper


## Teacher-first ROI roadmap

If the ROI / semantic-compression lane is revisited later, the safer next version should follow a teacher-first path:

- use SegNet/PoseNet sensitivity and task-loss ablations first
- compare that against cheap priors like temporal variance, optical flow, motion vectors, and coarse geometry priors
- only add heavier segmentation/tracking tools later as offline mask proposal or stabilization aids
- keep heavy learned masking tools out of `inflate.sh` unless the byte/runtime burden is clearly justified

This remains speculative until a cheaper side-information story proves it can beat the current honest floor.

## AV1 + ROI parity lane

This lane is explicitly speculative until it is implemented end-to-end and scorer-backed.

Canonical deep-dive / partner-agent handoff:

- `docs/av1_roi_roadmap.md`

Required plan before any promotion:

1. codec-agnostic ROI encode abstraction
2. AV1 params for base/ROI/ROI2 streams
3. matching AV1-aware metadata ROI path
4. matching inflate/smoke/scorer parity checks
5. fresh scorer-backed evidence that it actually helps

Why it is still speculative:

- the current ROI implementation is intentionally x265-only
- silent codec drift would corrupt comparisons and undermine compliance claims
- AV1 + ROI needs full branch parity, not partial support

Promotion rule:

Do not promote or even present this lane as live support until all five requirements above are complete and a scorer-backed run beats the current canonical honest floor.


## Learned post-filter follow-on lanes (2026-04-07)

The learned post-filter base lane is no longer speculative. It is now the promoted honest floor at **`2.05`** and the highest-upside active family in the repo.

Current verified status after promotion:
- local isolated smoke: `861,986` bytes, semantic MAE mean `5.355835021839174`
- BAT00 smoke: `861,436` bytes, semantic MAE mean `5.450389819860672`
- authoritative local scorer: **`2.05`** at `861,986` bytes
- grain-mask comparison lane is a verified reject at `2.30`
- the first post-filter variant was a verified reject at `2.35` because it was trained on the wrong archive distribution

### Byte-cost note

At this archive scale, post-filter weight growth is surprisingly cheap in score terms. Roughly:
- +10 KB payload ~= +0.0067 score from the `25 * rate` term
- +20 KB payload ~= +0.0133 score
- +50 KB payload ~= +0.0333 score

That means a slightly larger model is worth exploring if it buys a measurable PoseNet/SegNet improvement. The post-filter search should therefore expand in **both** directions: smaller **and** somewhat larger.

### Recommended three-lane next cycle

1. **Slightly larger post-filter**
   - Goal: test whether a moderate capacity increase beats the tiny 7.5 KB model by more than the tiny byte penalty
   - First candidates:
     - hidden channels `16 -> 24`
     - hidden channels `16 -> 32`
     - keep the same 3-layer residual topology first before changing anything else
   - Why: model-byte cost is tiny relative to potential distortion wins; this is the direct opposite-direction check that the current plan was missing

2. **Smaller post-filter**
   - Goal: keep most of the gain while cutting runtime and shipped bytes
   - First candidates:
     - hidden channels `16 -> 8`
     - two-layer residual CNN instead of three-layer
     - luma-only filter (predict Y correction, keep chroma untouched)
   - Why: this is the most Rick-Rubin-style version — remove everything that is not clearly earning its keep

3. **Cheaper architecture / hardware-friendly post-filter**
   - Goal: preserve the gain with a more CPU-friendly operator
   - First candidates:
     - depthwise-separable residual block
     - per-channel 3x3 kernels + tiny pointwise mix
     - BSConv / blueprint-separable style block
   - Why: aligns with efficient-SR literature and Chris-Lattner/NVIDIA-style deployment thinking: fewer MACs, easier vectorization, lower runtime overhead

### More speculative follow-ons

- **LUT-like post-filter**
  - inspiration: SepLUT (`arXiv:2207.08351`), LUT-GCE (`arXiv:2306.07083`)
  - idea: replace some CNN work with a tiny learned LUT or curve stage for near-zero-cost color/contrast correction
  - risk: likely too global unless paired with a local edge filter

- **Residual patch + post-filter hybrid**
  - keep the post-filter tiny, then spend a few extra bytes only on ROI-local residual corrections
  - attractive if the post-filter mainly fixes broad structure while a tiny patch fixes the highest-value semantic edges

- **Post-filter + ROI map**
  - only worth revisiting after the standalone post-filter score is known
  - do not stack complexity blindly; require one-axis evidence first

### Promotion rule

The base learned post-filter lane is already promoted. Only the follow-on variants above remain speculative, and none should be promoted without a full authoritative scorer run.
