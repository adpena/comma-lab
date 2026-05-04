---
name: Hardware + Geometry + Chroma + Scorer Internals (Full Knowledge Base)
description: Verified facts about the comma EON camera, openpilot scorer architectures, hardware blind spots, dim zones, radial zoom, FoE geometry, YUV6 chroma plane behavior. Anchor everything we do against this.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
CAMERA + GEOMETRY (verified from upstream + project_comma_hardware.md):
- comma EON (2018), AR0231AT sensor, fx=910 px focal, principal point pp=(582, 437)
- Native resolution 1164×874, 20 fps, 1200 frames per clip
- Camera mount: rigid, behind windshield, fixed angle. The Field of Expansion (FoE) for forward driving is at approximately (256, 174) at SegNet input resolution (512×384)
- This means: radial zoom from FoE is the dominant ego-motion signal. Per the PoseNet rank-1 discovery (project_posenet_rank1_discovery.md), pose dim 0 = 99.8% variance — it IS scalar zoom. Dims 1-5 are pose noise.
- Implication: storing 6-D poses per pair is 6x wasteful. Store 1 scalar per pair (2.4KB instead of 7KB).

POSENET ARCHITECTURE (verified from upstream/modules.py):
- FastViT-T12 backbone (transformer-style, NOT EfficientNet)
- Input: 12 channels = 2 frames × YUV6 (4 luma planes + 2 subsampled chroma planes per frame)
- Frame pair resized to (512, 384), normalized (mean=127.5, std=63.75)
- Output: vision (2048) → summary (512) → ResBlock → 12-dim pose, FIRST 6 DIMS USED
- Distortion = MSE on first 6 pose dims
- KEY: rgb_to_yuv6 is a fixed transform. PoseNet sees YUV6 not RGB. Errors that cancel in luminance can persist in chroma — and vice versa.
- KEY: temporal derivative = frame_t to frame_t+1 motion. PoseNet is essentially a "what changed?" detector. Spatial fidelity per frame is irrelevant if the SAME error pattern persists across frames (no perceived motion). Conversely, frame-to-frame jitter looks like real ego-motion.

SEGNET ARCHITECTURE (verified from upstream/modules.py):
- smp.Unet('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)
- EfficientNet-B2 (NOT B4), VANILLA stride-2 stem (no Yousfi surgery in our scorer)
- Input: LAST frame only x[:, -1, ...], bilinear resize to (512, 384)
- Output: 5-class logits, distortion = argmax disagreement rate
- 5 classes (per comma2k19 segmentation labels): road, lane markings, obstacles, vehicles, undriveable
- KEY BLIND SPOT: stride-2 stem loses half resolution immediately → patterns below (256, 192) effective resolution are invisible to argmax
- KEY: only argmax flips count. 100 logit-margin perturbations that don't flip argmax = 0 distortion. 1 boundary pixel that flips = 1 unit of distortion.
- IMPLICATION: bits should flow toward sub-stride-2 detail (FREE) and AVOID class boundaries (EXPENSIVE).

CHROMA / YUV6 BEHAVIOR:
- YUV6 = 4 luma planes (full resolution) + 2 chroma planes (half resolution per axis, BT.601 limited range)
- PoseNet's chroma sensitivity: smaller spatial extent (256×192 effective for chroma) but high temporal sensitivity
- Implication for compression: chroma errors propagate to PoseNet via 4x downsampled signal — a single chroma error covers a 2x2 pixel region in scorer input
- Pixel-shift artifacts (e.g., AV1 quantization at edges) appear as chroma-plane motion to PoseNet → phantom pose

ORTHOGONALITY OF DETECTORS (Fridrich council):
- SegNet = SPATIAL argmax detector. Argmax stability matters. Class-boundary pixel reconstruction is critical. Texture-region pixel reconstruction is FREE.
- PoseNet = TEMPORAL chroma derivative detector. Frame-to-frame pixel CONSISTENCY matters. Per-frame absolute fidelity is partial.
- The two have OPPOSITE signs to the same noise:
  - Mask quantization smoothing → SegNet sees fewer argmax flips (BETTER) → PoseNet sees structured edge motion (WORSE)
  - Persistent low-amplitude noise → SegNet sees same argmax (NEUTRAL) → PoseNet sees no motion change (NEUTRAL)
  - Per-frame random noise → SegNet may see flips (WORSE) → PoseNet sees random pose noise (WORSE)
- Cannot satisfy both detectors by tuning ONE input. Need to optimize against BOTH simultaneously, with awareness of which axes are orthogonal.

HARDWARE EXPLOITS:
- Contest scorer runs on T4 (Turing sm_75) + DALI + CUDA + Linux. NOT MPS, NOT CPU.
- T4 supports tf32, fp16, int8 hardware-accelerated. FP4 is software-only on T4 → SLOW at inflate.
- Inflate budget: 30 min total (decode + render + score). Our renderer must inflate 1200 frames in <5 min for the scorer to have time.
- MPS (Apple Silicon) ≠ CUDA T4 for numerical precision. Verified: FP4 dequant differs by ~0.0147 between MPS and CPU (320× larger than MPS-MPS noise). Implication: every score we measure on MPS may differ from contest by 0.05-0.5.
- DALI (vs AVVideoDataset) for GT decode uses NVDEC + bilinear chroma upsampling + BT.601 limited range. Our local AVVideoDataset matches DALI per `frame_utils.yuv420_to_rgb` (verified pixel-for-pixel).

DIM ZONES + REGIONS:
- Sky region (top of frame, ~y < 200): low-information, mostly uniform luminance. Compression errors here are invisible to all detectors.
- Road region (center, y in 200-700): high information. Lane markings, obstacles, motion features. Errors here are EXPENSIVE.
- Hood region (bottom, ~y > 750): static, vehicle-mounted. Errors here are partial — SegNet may see hood as undriveable.
- Lane marks (specifically): MUTCD 3m × 15cm at known intervals → can be EXPLOITED for analytical zoom estimation per project_lane_marking_speed_estimation.md (still unimplemented).

HARDWARE-AWARE COMPRESSION STRATEGY:
- T4 INT8 inference: hardware accelerated, faster than FP4 dequant
- Trade-off: ship slightly bigger archive (INT8 ~80KB vs FP4 ~40KB for our renderer) → faster inflate → can use the saved time for compress-time TTO that spans more steps
- Asymmetric compute budget: 100% at compress (unlimited), 30 min at inflate (T4)
- Move expensive ops to compress side: pose TTO with 2000 steps, gradient corrections, Fridrich UNIWARD optimization
- Move cheap ops to inflate: single forward through renderer, mask decode, lane-mark zoom estimation (analytic, ~1ms per pair)

THE CHROMA-PLANE EXPLOIT WE HAVE NOT TRIED:
- PoseNet sees YUV6 with 2 chroma planes at half-resolution
- If we render in chroma space with deliberate consistency (e.g., chroma plane = function of pose alone, not of mask quantization noise), PoseNet would see ONLY ego-motion, not compression artifacts
- This requires a YUV6-output renderer (currently RGB) — architectural change
- Estimated impact: PoseNet 0.245 → 0.05 if chroma is decoupled from mask noise

DOMAIN-SPECIFIC TASKS THE SCORERS ARE ACTUALLY TRAINED FOR:
- SegNet: comma2k19 dataset, 5-class semantic segmentation for openpilot's policy network's perception input
- PoseNet: ego-motion estimation from 2-frame stack for openpilot's lateral/longitudinal control
- BOTH are pre-trained openpilot components. The contest measures: "does your compressed video drive openpilot the same way the original would?"
- This means: errors that confuse the policy network (lane edge shifts, phantom motion) are penalized; errors that don't (texture noise, color shifts in uniform regions) are free
