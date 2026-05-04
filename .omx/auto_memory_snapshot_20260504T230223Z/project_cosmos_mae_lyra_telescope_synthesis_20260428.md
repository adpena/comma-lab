---
name: 2026-04-28 Cosmos+MAE+Lyra2.0+Telescope deep research synthesis — 6 new lane proposals
description: Recovery synthesis after molt-OOM crash. Read full PDFs of MAE / Lyra 2.0 (arXiv 2604.13036) / Telescope (arXiv 2604.06332) + cosmos-cookbook GitHub. Key findings: Cosmos park (wrong scale 1000×); MAE asymmetric encoder/decoder = our compress/inflate split; Lyra 3 transferable sub-mechanisms; Telescope hyperbolic foveation REVIVES dead Lane M+N. 3 high-EV lanes (SAUG, MAE-V, HF). Cycle 1 ~$13/12h.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TL;DR — 4 resources synthesized

1. **NVIDIA Cosmos cookbook** — almost entirely irrelevant. 14B-param models at 720p × 93 frames, 1000× wrong scale. Beamr CABR is just `ffmpeg -hwaccel cuda -vcodec h264_nvenc -rc maxq` — hardware NVENC tuning, not a learned codec. **Council verdict: PARK.** Single salvageable recipe = LoRA pattern (rank=32, alpha=32) but doesn't fit single-clip task.

2. **MAE (He et al. 2111.06377)** — directly applicable as conditional generator architecture. The key insight is **asymmetric encoder/decoder where decoder operates on a different (larger) token set than encoder**. Maps 1:1 onto compress→inflate split. **2 lane proposals: Lane MAE, Lane MAE-Q.**

3. **Lyra 2.0 (arXiv 2604.13036, NVIDIA, 2026-04-15)** — long-horizon video synthesis on Wan-2.1-14B-DiT (too big for us), but 3 transferable sub-mechanisms: (a) **FramePack variable-kernel temporal compression**, (b) **self-augmentation** (corrupt history latents during training so model learns to denoise from imperfect context — DIRECTLY ATTACKS our 100-350× proxy/auth gap problem), (c) **canonical-coordinate warping conditioning** (warp position channels not RGB, removes warp artifacts as crutch). **3 lane proposals: Lane FP, Lane SAUG, Lane CCW.**

4. **Telescope (Princeton + Torc Robotics, arXiv 2604.06332)** — learnable hyperbolic foveation. Math: `Φ(x) = (1-w(r))·x + w(r)·h(x)` where `h(x) = o + tanh(α·r)/r · (x-o)` is Poincaré-disk-like radial contraction, blended by `w(r) = (1 - min(r/R, 1))^p`. **Per-image learnable (only 4 params α,R,p,o), invertible via Newton-Raphson, differentiable.** Applied to high-res input → inverted on detections. **REVIVES our DEAD Lane M+N (radial zoom NEGATIVE)** with the right invertible math. 2 lane proposals: Lane HF, Lane HFM.

---

## TOP 3 HIGH-EV PROPOSALS (council-ready)

### Lane SAUG — Lyra self-augmentation (HIGHEST EV)
- **Premise**: corrupt the proxy training signal during renderer training so the model learns to denoise from imperfect context. Directly attacks our largest documented blocker (proxy/auth gap of 100-350× even on CUDA-CUDA per `feedback_proxy_auth_math_useless`, `project_lane_b_pose_tto_proxy_auth_gap`).
- **Mechanism**: during training, randomly perturb the GT poses/masks fed to the renderer with the SAME distribution of noise we observe in the proxy-auth gap (Gaussian on poses with σ matching empirical drift; argmax flips on masks at empirical SegNet error rate). Loss computed against UNPERTURBED targets — model must learn to compensate.
- **Predicted band**: [0.75, 1.05] — first sub-1.0 candidate
- **Cost**: ~$5 / 18h on Vast.ai 4090
- **Composability**: Orthogonal to all renderer-arch lanes. Can stack with Lane W, Lane S, Lane Ω-V2.
- **Risk**: medium — implementation is small but tuning σ schedule requires sweep

### Lane MAE-V — joint mask-aug from epoch 0 (revives Quantizr trick)
- **Premise**: the half-frame trick that BROKE our PoseNet (17.55 score, memory `feedback_half_frame_breaks_posenet`) failed because it was bolted onto a baseline checkpoint. MAE Table 1f shows the model MUST be trained on the masking pattern from the start. Train dilated-h64 with `mask_half_sim_prob=1.0` from epoch 0.
- **Predicted band**: [0.80, 1.10]
- **Cost**: ~$4 / 16h (full retrain)
- **Composability**: stacks with Lane W, Lane SAUG. Conflicts with Lane I (different arch).
- **Risk**: low — well-validated by Quantizr (88K replica)

### Lane HF — Telescope hyperbolic foveation (revives Lane M+N)
- **Premise**: Lane M+N (radial zoom 1-DOF) scored 2.35 [contest-CUDA] = +0.06 vs baseline 2.29 (memory `project_lane_mn_radial_zoom_negative`). Council determined the rank-1 PoseNet sensitivity ≠ rank-1 renderer input space. Telescope's `Φ(x) = (1-w(r))·x + w(r)·h(x)` provides the proper invertible math: per-frame learnable (4 params: α, R, p, o), differentiable, Newton-Raphson invertible. Applied to high-res rendered frame → inverted on the upscale.
- **Predicted band**: [0.85, 1.10]
- **Cost**: ~$4 / 16h
- **Composability**: orthogonal to renderer-quant lanes. Conflicts with Lane HM (different motion math).
- **Risk**: medium — the invertibility is the load-bearing claim; needs gradcheck

---

## RECOMMENDED CYCLE 1 PROPOSAL

Per CLAUDE.md "≤3 experiments per cycle":
- Cycle 1 = **SAUG + MAE-V + HF in parallel** on 3× Vast.ai 4090
- Estimated cost: ~$13
- Estimated wallclock: ~12h
- Composition cycle (Cycle 2) targets [0.55, 0.85] band — **first-ever sub-1.0**

---

## SECONDARY PROPOSALS (lower EV but worth tracking)

- **Lane MAE** — 75% mask reduction on masks.mkv (4× rate cut on dominant component). [0.80, 1.10] but high risk — Quantizr 50% trick already broke. Only do AFTER MAE-V validates.
- **Lane FP** — FramePack variable-kernel temporal compression. Recent frames small kernel, distant frames large kernel. [0.95, 1.15] modest.
- **Lane CCW** — canonical-coordinate warping (warp position channels not RGB). [0.90, 1.15]. Removes warp artifacts as a crutch — could pair with Lane HM.
- **Lane HFM** — Telescope foveation applied to MASKS (not frames) — increases mask-encoding density in PoseNet-critical regions. [0.95, 1.20].
- **Lane MAE-Q** — MAE asymmetry pattern applied to renderer architecture (heavy compress-time encoder, light inflate-time decoder). Architectural, requires significant refactor. [0.85, 1.15].

---

## DEAD-ON-ARRIVAL (dispositioned)

- **Lane LR-V3 (LoRA-as-renderer-delta)** — useless single-clip
- **Lane V-DMD (adversarial distillation)** — Lane W is higher EV
- **Beamr CABR for masks** — we already use libsvtav1 AV1, switching costs rate
- **Cosmos Reason VLM 7B for compression** — absurd at our budget

---

## Cross-references
- `project_council_eurekas_driving_geometry_20260428` — 8 council EUREKA lanes (GP/FL/GE/HM/CG/SG/DI/SI-V3)
- `project_outstanding_work_and_stacks_20260428` — TIER 3 never-deployed lanes
- `feedback_proxy_auth_math_useless` — the gap Lane SAUG attacks
- `feedback_half_frame_breaks_posenet` — why Lane MAE-V must train from epoch 0
- `project_lane_mn_radial_zoom_negative_20260427` — what Lane HF revives
- `project_quantizr_full_intel_20260421` — Lane V Quantizr replica context

## Output artifact
- Full synthesis: `/Users/adpena/Projects/pact/.omx/research/cosmos_mae_2604_telescope_synthesis.md` (384 lines, 35.7KB)
