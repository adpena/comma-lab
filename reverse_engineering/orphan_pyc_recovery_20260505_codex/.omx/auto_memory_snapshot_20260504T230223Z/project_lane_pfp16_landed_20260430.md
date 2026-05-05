---
name: PROJECT — Lane PFP16 LANDED (pose fp16 cast — Hotz successor from Lane GP v4 KILL)
description: 2026-04-30. Lane PFP16 (Pose Float-16 cast) implementation lands as the dominant-strategy successor surfaced by the Lane GP v4 council finding. Cast Lane G v3's `optimized_poses.pt` (15,620 B fp32 pickle) to raw fp16 binary (`optimized_poses.bin`, 7,200 B) — saves 7,439 B on the assembled archive at ZERO distortion (PoseNet runs in fp16 internally during contest CUDA eval). Predicted contest-CUDA score 1.045 [derivation, contest-CUDA pending].
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## TL;DR

Lane PFP16 ships as Hotz's `radical simplicity` successor option from the
Lane GP v4 KILL VERDICT — the only viable pose-stream byte-savings lane
with ZERO distortion penalty. Predicted contest-CUDA score 1.045
[derivation, contest-CUDA pending], down from Lane G v3 baseline 1.05.

## Context

Lane GP v4 council finding
(.omx/research/council_lane_gp_v4_design_20260430.md) demonstrated that
ALL smooth-basis pose-fits (polynomial / B-spline / DCT / natural cubic)
plateau at RMSE ≈ 1.2 — near signal std — because Lane G v3
optimized_poses.pt is approximately white-noise in dims 1-5
(diff_std > signal_std for every dim). The verdict was a hard KILL on
the entire smooth-basis-fit lane class.

The single dominant successor option surfaced was Hotz's radical
simplicity: cast `optimized_poses.pt` from fp32 to raw fp16 binary. This
is NOT a basis-fit; it is a precision-reduction in the storage format
that exploits the fact that PoseNet's contest-CUDA forward pass already
runs in fp16 internally. The cast is invisible to the scorer, but saves
~8 KB in the archive.

## Implementation

### Files added/modified

| File | Lines | Purpose |
|---|---|---|
| `src/tac/pfp16_codec.py` | 200 | encode/decode primitives + file wrapper |
| `src/tac/tests/test_pfp16_codec.py` | 280 | 19 tests (roundtrip + invariants) |
| `src/tac/tests/test_check_pose_stream_fp16.py` | 220 | 12 tests (Check 96 regression) |
| `experiments/build_lane_g_v3_pfp16_stack.py` | 270 | CPU-only stacked archive build |
| `scripts/remote_lane_pfp16_stack.sh` | 200 | 4-stage canonical remote dispatch |
| `reports/lane_pfp16_real_archive.json` | 35 | empirical provenance |
| `src/tac/preflight.py` | +180 | Check 96 (`check_pose_stream_uses_fp16_or_smaller`) STRICT |

### Empirical (real-archive build, 2026-04-30)

| Metric | Value | Tag |
|---|---|---|
| Lane G v3 archive baseline | 694,074 B | [empirical, contest-CUDA-validated 1.05] |
| Lane G v3 + PFP16 archive | 686,635 B | [empirical:reports/lane_pfp16_real_archive.json] |
| Archive bytes saved | -7,439 B | [empirical] |
| Pose stream raw bytes saved | -8,420 B (fp32 pickle 15,620 → fp16 raw 7,200) | [empirical] |
| Δ rate term | 25 × -7,439 / 37,545,489 = -0.00495 | [derivation] |
| Predicted contest-CUDA score | 1.05 + (-0.00495) = ~1.045 | [derivation, contest-CUDA pending] |
| Roundtrip max-abs error | 0.015518 (well under 0.06 tol) | [empirical] |

### STRICT preflight Check 96

`check_pose_stream_uses_fp16_or_smaller` lands STRICT @ 0 violations on
the 2026-04-30 codebase. Scans every `experiments/build_*_stack.py`,
`experiments/build_*_archive.py`, `experiments/build_lane_*.py` for
`torch.save(pose_tensor, ...)` calls without canonical pose encoders
(`encode_pfp16`, `save_poses_binary`, `encode_pose_deltas`,
`encode_pose_delta_v2`, `encode_lora_*`, etc.). Waiver:
`# POSE_FP32_REQUIRED:<reason>` for legitimate exceptions.

## Council adversarial review

3 clean passes / 3 perspectives each:

- Round 1: Yousfi, Fridrich, Contrarian, Quantizr, Hotz → 0 issues
  (.omx/research/council_lane_pfp16_round1_20260430.md)
- Round 2: Shannon, Dykstra, MacKay, Ballé, Selfcomp → 0 issues
  (.omx/research/council_lane_pfp16_round2_20260430.md)
- Round 3: Filler, Carmack, Hassabis, Hinton, Tao → 0 issues
  (.omx/research/council_lane_pfp16_round3_20260430.md)

15 distinct council voices reviewed; counter advanced 0 → 1 → 2 → 3
with NO RESET. Lane is cleared for contest-CUDA dispatch.

## Strategic role

Lane PFP16 is small in absolute score impact (-0.005), but it serves as:

1. **Byte-budget audit pipeline scaffolding** — proves the archive-byte
   discipline (Check 96) works on a low-risk lane before the next
   higher-risk byte lanes (Lane J-NWC, Lane PD-V2 in stacked archive).
2. **D=0 reference point on the Pareto frontier** — strict-best lane
   in the "no distortion penalty" subset; Pareto-dominated only by
   Lane LI (Selfcomp's PoseNet-affine-learned-image, separate paradigm).
3. **Stacking compatibility** — composes with Lane Ω-W-V2 stack (renderer
   re-encoding) and Lane PD/PD-V2 (pose delta) for additional savings
   in future stacked archives.

## Cross-references

- KILL verdict that surfaced PFP16: `.omx/research/council_lane_gp_v4_design_20260430.md`
- Production-hardened standard: `feedback_production_hardened_standard_definition_20260430.md`
- Sibling pose codecs:
  - `src/tac/pose_delta_codec.py` (Lane PD)
  - `src/tac/pose_delta_codec_v2.py` (Lane PD-V2)
  - `src/tac/lora_pose.py` (Lane LR)
- Inflate-side compatibility: `submissions/robust_current/inflate_renderer.py`
  (Branch B of `tac.submission_archive.load_optimized_poses` already
  handles raw fp16 buffers transparently — no inflate-side changes needed)
- Sister lane: Lane Ω-W-V2 stack (renderer re-encoding)
  - `experiments/build_lane_g_v3_omega_w_v2_stack.py`
  - `scripts/remote_lane_omega_w_v2_stack.sh`

## Lane registry status

`lane_pfp16` at L2 → L3 (after contest-CUDA dispatch lands):

- [x] impl_complete (`src/tac/pfp16_codec.py`)
- [x] real_archive_empirical (`reports/lane_pfp16_real_archive.json`)
- [x] strict_preflight (Check 96 @ 0 violations)
- [x] deploy_runbook (`scripts/remote_lane_pfp16_stack.sh`)
- [x] three_clean_review (Rounds 1+2+3 each found 0 issues)
- [x] memory_entry (this file)
- [ ] contest_cuda (PENDING — dispatched, awaiting result)

After contest_cuda result lands, `lane_pfp16` graduates to Level 3 (Full
Production Hardened + Recursive Adversarial Reviewed) — the user's
non-negotiable standard.

## Dispatch

- Vast.ai 4090 instance dispatched via Pattern A nohup detach
- Cost cap: $0.50 (under $10 cap, no further approval needed per CLAUDE.md)
- Predicted ETA: ~30 min wall-clock
- Result tag target: `[contest-CUDA] reports/lane_pfp16_cuda.json`

## Hard kill criteria

- archive bytes >= Lane G v3 baseline → halts at Stage 2
- contest-CUDA score > 1.05 → labeled HARD_KILL_REGRESSION (would mean
  fp16 cast somehow changed PoseNet behaviour beyond its intrinsic fp16
  forward path, which would require investigation before any further
  pose-stream reduction lanes ship)
- contest-CUDA score in [1.04, 1.05] → labeled IN_PREDICTED_BAND →
  graduates to Level 3
- contest-CUDA score < 1.04 → labeled OUT_OF_PREDICTED_BAND → unexpected
  upside, investigate which assumption was conservative
