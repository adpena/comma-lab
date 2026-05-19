# 5. Production Deployment

## Historical Apogee Deployment Note

This note is a historical orientation snapshot, not a live leaderboard or
production-readiness claim. For current authority, prefer the root README,
`reports/latest.md`, and evidence-grade rows with exact archive bytes, runtime
custody, component distances, and explicit `[contest-CPU]` / `[contest-CUDA]`
axis labels. The Apogee packet referenced here is not the older
mask-renderer/TTO system described in the historical sections below. It is a
route-specialized HNeRV/Muon archive with a single charged payload member, a
deterministic adapter runtime, and exact Tesla T4 custody:

- exact score: `0.22826947142244708` `[A++]`
- archive bytes/SHA-256:
  `178981`,
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- runtime tree SHA-256:
  `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`
- exact eval elapsed time: about `60.97s`
- inflate elapsed time: about `22.56s`

For production and OSS purposes, the deployable idea is the disciplined packet
contract: charged neural payload, deterministic decoder, fail-closed wrapper,
runtime tree hashing, exact replay tests, and public-release hygiene. The
sections below remain useful as research and deployment context for the broader
`tac` toolkit, but they should not be read as the submitted Apogee architecture.

The comma.ai video compression challenge is not an academic exercise. It models a real engineering problem: how to compress dashcam video while preserving the signals that a self-driving system depends on. This section maps our techniques to comma.ai's actual production pipeline and analyzes deployment feasibility across three scenarios.

## 5.1 comma.ai System Architecture

### Hardware and capture

The comma three runs a Qualcomm Snapdragon 845 SoC with an ON Semiconductor AR0231AT automotive image sensor. The camera captures 1164x874 frames at 20 fps. Video is encoded on-device as H.265 (HEVC) using the Snapdragon's hardware encoder and stored locally on a 32 GB eMMC module. Routes (driving segments) are uploaded over cellular to comma's servers when bandwidth is available.

The comma four uses a newer Snapdragon SoC with the same camera configuration. Both devices run openpilot, comma's open-source driver assistance software, which processes camera frames through a monolithic "supercombo" model that handles lane detection, lead vehicle tracking, path planning, and driver monitoring.

### On-device perception

Two perception networks are relevant to the challenge and to production:

- **SegNet** (EfficientNet-B2 U-Net, ~13M parameters): semantic segmentation into 6 road-scene classes (road, lane markings, undrivable, movable, ego car, sky). Trained on comma10k, a 10,000-image subset of fleet data. No backup sensor provides equivalent information --- if compression degrades SegNet input, scene understanding degrades with no fallback.

- **PoseNet** (convolutional, ~2M parameters): estimates 6-DOF ego-motion from consecutive YUV frame pairs. The driving stack fuses PoseNet output with IMU and wheel odometry through a Kalman filter. PoseNet is therefore partially redundant: degraded PoseNet input causes accuracy loss but not total failure, because the filter can rely on inertial sensors.

This asymmetry explains the scoring formula's structure. SegNet distortion is weighted 100x linearly (any regression is directly harmful). PoseNet distortion is passed through a square root (diminishing returns, reflecting sensor fusion's ability to compensate).

### Data pipeline

The full pipeline is:

```
Camera (AR0231AT, 20fps, 1164x874)
  -> H.265 encode (Snapdragon hardware encoder, CRF ~23)
  -> Local storage (32 GB eMMC, ~20 hours of driving)
  -> Cellular upload to comma servers (when bandwidth permits)
  -> Server-side decode + processing
  -> Training data for supercombo model updates
  -> OTA model deployment back to fleet
```

Compression enters at two points: on-device encoding for local storage and upload, and server-side re-encoding of the training corpus. Both are offline operations --- there is no real-time decode constraint on the compression side, only on the playback/inference side.

## 5.2 Production Deployment Scenarios

### Scenario A: Fleet data compression (server-side)

> Status: illustrative draft scenario. Numeric ratios, route costs, and quality
> claims in this section require evidence tags before public/paper use.

The most natural deployment. comma's servers hold petabytes of driving video from a fleet of 250,000+ devices. Re-compressing this corpus with a task-aware compression system reduces storage costs and speeds up training data loading, without degrading downstream model training quality.

**How it works.** The renderer runs on server GPUs (A100, H100, or commodity 4090s). For each route:

1. Decode the existing H.265 video
2. Run SegNet to extract 5-class semantic masks
3. Encode masks as AV1 monochrome video (~60--80 KB per 60 seconds at 1/8 scale)
4. Train or fine-tune the renderer against the specific route's scorer outputs (or use a fleet-wide pretrained renderer)
5. Store: masks (~79 KB at 1/8 scale) + renderer weights (150 KB) + metadata

**Compression ratio.** The original H.265 video for 60 seconds at CRF 23 is approximately 2.5 MB. Our archive (masks + renderer weights) is 184 KB (183,780 bytes measured). This is a ~14x compression ratio over the already-compressed H.265, or roughly 200x over raw video (1200 frames x 1164 x 874 x 3 bytes = ~2.9 GB uncompressed).

**Reconstruction quality.** Auth score 0.87 (renderer only) or 0.43 (renderer + TTO). The 0.87 figure uses only a forward pass through the renderer --- no iterative optimization --- and runs in approximately 2 seconds on a T4 for 1200 frames. This means the perception models' outputs on reconstructed frames differ from outputs on original frames by less than 1% (SegNet cross-entropy 0.0022, PoseNet L2 0.031).

**Cost estimate.** SegNet inference on 1200 frames: ~3 seconds on T4. Mask encoding: ~1 second. Renderer forward pass: ~2 seconds. Total: approximately 6 seconds per 60-second route on a T4 ($0.59/hr), or $0.001 per route. For 1M routes, the re-compression cost is approximately $1,000. The storage savings at 17x compression dwarf this.

**Limitation.** This scenario requires a trained renderer. A fleet-wide renderer (trained on diverse routes) would generalize across clips but with higher distortion than a per-route model. The per-route training cost (~30 minutes on a 4090, ~2 hours on T4) makes per-route training feasible for archival but not for real-time ingestion.

### Scenario B: Training data pipeline (server-side, with TTO)

A more aggressive variant of Scenario A. Instead of storing video, store compact representations that reconstruct scorer-optimal frames on demand.

**How it works.** At ingestion time:

1. Compress each route as in Scenario A (masks + renderer weights): 150 KB
2. Run TTO for 500 steps per batch, pre-computing the pixel perturbations: ~3 hours on T4, ~40 minutes on 4090
3. Store the perturbation deltas (quantized to int8, sparse): ~50 KB additional

At training time, reconstruct frames with a single forward pass through the renderer + apply stored deltas. No iterative optimization at serve time.

**Quality.** Auth score 0.43 with pre-computed TTO. The reconstruction is optimized specifically for the scorer networks that will consume it during training.

**Cost.** The TTO step is expensive: $1.75 per route on T4, or $0.17 per route on a 4090. At scale (1M routes), this is $170K on 4090s. This is justified only if the downstream training quality improvement (from cleaner data) exceeds the compression cost --- a question that requires empirical validation on fleet-scale training runs.

**Key insight.** TTO pre-computation is an instance of asymmetric compute: invest heavily at compress time (once), benefit at every training read (many times). The amortized cost per training read approaches zero as the training pipeline iterates over the data.

### Scenario C: On-device decoder (edge inference)

The renderer runs on the comma device's Snapdragon SoC to decompress stored routes for local replay, debugging, or fallback visualization.

**Feasibility analysis.** The renderer has 287K parameters in FP4 (150 KB). The Snapdragon 845's Hexagon DSP (NPU) handles int8/int4 inference natively, with the entire model fitting in the NPU's on-chip SRAM.

Estimated per-frame latency:

| Operation | T4 GPU | Snapdragon 845 NPU (est.) | Snapdragon 8 Gen 2 NPU (est.) |
|-----------|--------|---------------------------|-------------------------------|
| Mask decode (AV1) | 0.8 ms | 2--5 ms | 1--3 ms |
| Renderer forward | 1.7 ms | 8--15 ms | 3--7 ms |
| Upscale (384x512 -> 1164x874) | 0.5 ms | 2--4 ms | 1--2 ms |
| **Total per frame** | **3.0 ms** | **12--24 ms** | **5--12 ms** |

*Note: Snapdragon NPU latency estimates are analytical projections based on FLOP counts and published Snapdragon 845/8 Gen 2 NPU throughput specifications; no on-device measurements were performed.*

At 20 fps, the frame budget is 50 ms. Even the older Snapdragon 845 has sufficient projected headroom for real-time renderer-based playback at full frame rate. The newer Snapdragon 8 Gen 2 (comma four candidate SoC) would leave 38--45 ms of budget for other tasks.

**Limitation.** TTO is not feasible on-device. The 500-step optimization requires scorer forward and backward passes per step, totaling ~3 hours on T4. On mobile, this would take days. On-device reconstruction is renderer-only (auth 0.87 quality), not renderer+TTO (auth 0.43).

## 5.3 Scalability Story

Our system scales across four tiers, each adding compute to improve quality:

| Tier | Hardware | Time budget | Approach | Score | Status |
|------|----------|-------------|----------|-------|--------|
| Contest | T4 16 GB | 30 min | Renderer-only | 0.87 | Validated (auth) |
| Contest + TTO | T4 16 GB | 30 min | Renderer + partial TTO (5 batches) | ~0.70 | Validated (auth) |
| Research | RTX 4090 24 GB | Unlimited | Renderer + full TTO (60 batches) | 0.43 | Validated (auth) |
| Fleet | Multi-GPU | Unlimited | Per-route renderer + TTO + ensemble | < 0.40 | Projected |

**Contest tier (T4, 30 minutes).** The renderer inflates 1200 frames in ~3 minutes. The remaining 27 minutes allow TTO on approximately 5 of 60 frame-pair batches (at ~181 seconds per batch on T4). Prioritizing the hardest batches (highest initial distortion) yields diminishing but measurable improvement. This is the contest-compliant configuration.

**Research tier (4090, unlimited).** Full TTO across all 60 batches takes ~40 minutes on a single 4090. The 4090 is 4--5x faster than T4 for scorer forward/backward passes, at roughly the same hourly cost ($0.25/hr on Vast.ai vs. $0.59/hr on Modal). This tier produced our best validated score of 0.43.

**Fleet tier (multi-GPU, unlimited).** TTO is embarrassingly parallel across batches. Distributing 60 batches across 4 GPUs reduces wall time to ~10 minutes. Ensemble techniques (averaging TTO results from multiple random initializations) could push below 0.40, though this is currently unvalidated.

The critical observation: the renderer is the fixed cost (~150 KB, ~3 minutes), and TTO is the variable cost that trades compute for quality. In production, the decision of how much TTO to run is a cost-quality knob that operators set based on their budget and quality requirements.

## 5.4 Technical Innovations for Production

### Mask-based compression

Semantic masks are a natural intermediate representation for driving video. A 5-class segmentation at 48x64 (1/8 scale), encoded as AV1 monochrome video at CRF 20, compresses 60 seconds (1200 frames) into approximately 79 KB. At full 384x512 resolution, the mask video is approximately 2 MB. The 1/8 scale encoding is 30--40x smaller than the equivalent H.265 video while preserving sufficient spatial structure for the renderer. The masks retain the scene's spatial structure --- road layout, lane boundaries, vehicle positions --- while discarding texture, lighting, and other perceptually rich but task-irrelevant detail.

For fleet data pipelines, masks have utility beyond compression: they serve directly as training labels for SegNet updates, as input features for route analysis and scene classification, and as a compact scene descriptor for retrieval. Storing masks alongside or instead of video serves multiple downstream consumers.

### Scorer-aware codec training

Standard codecs minimize pixel-level distortion (MSE, SSIM). Our renderer minimizes task distortion: the cross-entropy between SegNet outputs on generated vs. original frames, and the L2 distance between PoseNet predictions. This end-to-end optimization through the downstream task is the codec analog of end-to-end training in NLP --- removing a hand-engineered intermediate representation (pixel quality) in favor of optimizing the actual objective.

The practical consequence: our renderer produces frames that look unrealistic to humans but score well on the task metrics. A road surface with incorrect texture but correct semantic boundaries gets near-zero SegNet distortion. A frame pair with wrong color temperature but correct relative geometry gets near-zero PoseNet distortion. The codec has learned what the perception stack actually uses.

### Asymmetric compute model

Our system is asymmetric by design: compression is expensive, decompression is cheap.

| Operation | Compute | Time (T4) |
|-----------|---------|-----------|
| Compress: SegNet inference + mask encode | Moderate | 4 s |
| Compress: Renderer training (per-route) | Heavy | 2 hr |
| Compress: TTO pre-computation | Heavy | 3 hr |
| **Decompress: Mask decode + renderer forward** | **Light** | **3 s** |
| Decompress: Apply stored TTO deltas | Light | 0.1 s |

This asymmetry is well-suited to fleet data pipelines, where each route is compressed once and read many times. The heavy compress-time investment amortizes over every downstream read.

### TTO as offline quality enhancement

Test-time optimization is not a deployment-time operation; it is a compress-time investment. The 500 gradient steps per batch refine the renderer's output against the exact scorers that will evaluate it. The resulting pixel perturbations are quantized and stored alongside the archive. At decompress time, they are applied as a single additive correction --- no optimization loop, no scorer loading, no GPU required.

This reframes TTO from "expensive inference" to "offline quality enhancement." The analogy in video coding is multi-pass encoding: the encoder runs multiple passes over the input to find optimal rate allocation. Our TTO is a multi-pass optimization over the output, finding optimal pixel perturbations for the downstream task.

## 5.5 Limitations and Future Work

### Not addressed: real-time encoding

The challenge is offline compression. Our renderer training takes hours per route. For real-time encoding (the on-device capture path), the system would need an amortized encoder --- a feed-forward network that predicts masks + renderer parameters in a single pass, trained across many routes. This is feasible in principle (neural codec encoders exist) but is not part of the current work.

### Not addressed: fleet diversity

Our renderer is trained and evaluated on a single 60-second clip. A production system would need to handle diverse driving conditions: night, rain, snow, highways, urban intersections, tunnels. Whether a single renderer generalizes or whether per-condition models are needed is an open question. The mask-based representation provides some invariance (masks abstract away lighting and weather), but the renderer's learned texture generation is likely condition-dependent.

### Decoder computational cost

The renderer's 287K parameters make it small enough for mobile deployment (150 KB, 8--15 ms per frame on Snapdragon 845 NPU). However, TTO-enhanced reconstruction requires either pre-computed deltas (stored alongside the archive, adding ~50 KB per route) or on-device optimization (infeasible on mobile hardware). The quality gap between renderer-only (0.87) and renderer+TTO (0.43) is substantial, meaning mobile playback operates at roughly half the quality of server-side reconstruction.

### Generalization to other cameras and models

The renderer is trained against specific scorer architectures (EfficientNet-B2 SegNet, convolutional PoseNet) processing specific input from a specific camera (AR0231AT at 1164x874). Changing any of these --- different camera intrinsics, different model architectures, different training data --- requires re-training. The framework transfers (the training procedure, TTO, mask representation), but the weights do not.

comma.ai updates their perception models periodically. Each model update would require re-training the renderer. This is a maintenance cost, though a modest one: renderer training takes 2 hours on a 4090 and is fully automated.

### Perceptual quality

Our generated frames are not intended for human viewing. They satisfy the perception stack but contain visible artifacts: incorrect textures, color shifts, geometric distortions in low-sensitivity regions. For use cases requiring both machine and human consumption (e.g., fleet dashcam review, accident reconstruction), the renderer would need an additional perceptual loss term, likely at the cost of task-metric performance.
