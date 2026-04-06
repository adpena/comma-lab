# speculative lanes

These lanes are explicitly secondary unless evidence forces promotion.

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
