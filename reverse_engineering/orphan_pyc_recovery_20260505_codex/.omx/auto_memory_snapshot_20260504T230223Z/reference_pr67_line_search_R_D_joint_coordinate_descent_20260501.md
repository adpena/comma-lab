---
name: PR #67 line_search.py is R(D)-joint-optimal pose coordinate descent — CONTRADICTS prior council "no leader uses scorer at compress"
description: 2026-05-01 ~13:55Z. After Fields-medal council flagged pr67_line_search.py as MAJOR INTEL the prior council missed, full reverse-engineering of the 194-LOC tool reveals: PR #67 EthanYangTW runs greedy coordinate descent on the trained pose-0 col values, scoring each candidate via FULL forward-through-generator + forward-through-PoseNet + MSE-vs-target-PoseNet-output, AND adding the encoded-archive byte cost. They accept ONLY changes that lower the joint score. This is the contest's exact objective being directly optimized at compress time. The prior Grand Council's claim "all leaders use vanilla MSE/CE/KL training, none apply UNIWARD-style score-aware embedding" was WRONG — at minimum, PR #67 does score-aware POSE refinement post-training.
type: reference
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## What pr67_line_search.py actually does

Source: `reports/raw/leaderboard_intel_20260501/pr67_line_search.py` (194 LOC).

### Setup
1. Loads existing archive (`p` blob), extracts mask + model + pose_q sections via the hard-coded MASK_BYTES=219472, MODEL_BYTES=56093 constants
2. Decodes QP1 to get current `col0` (uint16, length=600)
3. Loads JointFrameGenerator + PoseNet on CUDA
4. Loads GT video frames via DaliVideoDataset
5. Forward-passes GT through PoseNet to get the TARGET pose tensor (shape (600, 6))

### Coordinate descent loop
For each radius in `[1, 2, 3, 5, 8]`:
  For each pass in `range(2)`:
    For each frame index `i`:
      - Generate `cand` = `col0[i] + arange(-radius, radius+1)` (clamped to [0, 65535])
      - For each candidate value `v`:
        - Compute `pose_v = pose_from_col0([v])` → shape (1, 6) with pose[:, 1:] = 0
        - Generate frame pair: `(f1, f2) = JointFrameGenerator(mask[i], pose_v)`
        - Bilinear upsample to (874, 1164) like contest inflate
        - Forward through PoseNet → `pred_pose`
        - Compute `dist = (pred_pose - target[i]).pow(2).mean(dim=1)`
      - Keep the candidate `v*` with minimum `dist`
      - Update `cur[i] = v*`
    - Compute new `obj = sqrt(10 · mean(dist)) + 25 · archive_size / 37545489`
    - If `obj < best_obj`: accept the entire pass; else revert and break

### The R(D) joint objective
Line 140:
```python
return math.sqrt(10.0 * pose) + 25.0 * archive_size / ORIGINAL_SIZE, pose, pose_bytes, archive_size
```

This is EXACTLY the contest's pose+rate objective (minus the 100·seg term, which is held fixed because masks aren't being re-quantized in this loop).

The candidate's `archive_size = MASK_BYTES + MODEL_BYTES + pose_bytes + 100`. So a candidate that produces a smaller QP1 brotli-compressed delta (because it's closer to its neighbor in the col0 sequence → smaller VLQ → smaller brotli output) reduces archive_size, even if its PoseNet distortion is slightly worse. The greedy keeps the rate-distortion-joint-optimal value.

### Computational cost
For each frame: (2·radius+1) candidates × forward(JointFrameGenerator) + forward(PoseNet). For radius=8 that's 17 candidates per frame × 600 frames × 2 passes × 5 radii = **102,000 forward passes**.

On an RTX 4090 with batch_size=16 candidate_chunk=32, this should complete in ~30-60 minutes.

## Implications for OUR Wave-1 + Wave-Ω stack

### Wave-1 implication
The QZS3 packer subagent (`a3a932ac907d660b9`) should be aware that AFTER they assemble a candidate archive with the trained Q-FAITHFUL renderer, we should ALSO run a port of `pr67_line_search.py` to refine the pose stream against the same renderer. This is a CHEAP +0.001 to +0.005 score improvement.

### Wave-Ω implication
SJ-KL basis subagent (`ab7bce53c1e4a7a32`) is computing the Score-Jacobian eigenbasis for residual encoding. The line-search technique is COMPLEMENTARY: SJ-KL projects the residual into the optimal subspace; line-search refines the input (pose) to minimize the residual's projected energy. Both can stack.

### Strategic implication
The Fields-medal council's finding "PR #67 does score-aware compression" is now CONFIRMED with actual code. The prior Grand Council #1's statement "all leaders use vanilla MSE/CE/KL training, none apply UNIWARD-style score-aware embedding" was wrong. EthanYang already does score-aware POSE refinement; it's reasonable to assume he'll move to score-aware MASK refinement and score-aware MODEL refinement next.

This is competitive intel: **the contest is moving toward score-aware-everything**. Wave-Ω-1 (SJ-KL) is exactly that direction; we're catching up to where the leaders are heading.

## Things our existing line-search candidate codec at `experiments/build_renderer_packed_payload_archive.py:332` (`encode_pose_qpose14_col_delta`) may NOT have

Looking at our existing pose codec (which the QZS3 subagent is verifying against pr67's QP1 byte-for-byte), it MAY only do the encoding step — NOT the score-aware coordinate descent search FIRST. So we likely need TWO things:

1. **Encoder verification** (handled by QZS3 subagent `a3a932ac907d660b9`): does our `encode_pose_qpose14_col_delta` produce bytes EXACTLY matching pr67's `encode_qp1` for the same input?
2. **Pre-encoder line-search** (NEW work, not yet assigned): run pr67-style coordinate descent on the (renderer, pose) tuple to find the col0 values that minimize the joint objective. Then encode the searched values via our pose codec.

These are SEPARATE optimizations.

## Predicted byte/score impact

PR #67's archive ships pose_q at ~951 bytes (per their archive: 276,464 - 219,472 - 56,093 = 899 bytes), which is unlikely to be the raw QP1 output — it includes brotli wrapping. The fact that it's only ~0.9KB AFTER score-aware refinement suggests the coordinate descent finds significantly tighter pose-0 sequences than naive encoding.

Without their tool, our naive QP1 of the same trained col0 might be 1.5-2.5 KB. Score-aware refinement could trim to ~0.9 KB → -0.6 to -1.6 KB on rate → -0.0004 to -0.001 score.

Plus the distortion improvement (smaller pose error after the candidate that minimizes BOTH distortion AND rate): +0.001 to +0.003 score in distortion savings.

**Total: ~0.001-0.005 score improvement** on our Wave-1 archive when this is added.

## Implementation reference

Our future implementation should be a thin port of pr67_line_search.py with our renderer + pose codec substituted in. ~200 LOC. Cost: $0.50 GPU on Vast 4090, 30-60 min wall clock.

NOT urgent — wait for Wave-1 anchor (Q-FAITHFUL + QZS3 archive lands). Then this is a stack-on-top with deterministic +0.001-0.005 score gain.

## Cross-refs

- `reports/raw/leaderboard_intel_20260501/pr67_line_search.py` (the reference implementation, 194 LOC)
- `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md` (Council #2 surfaced this gap)
- `project_grand_council_shannon_floor_eureka_session_20260501.md` (Council #1 incorrectly claimed "no leader uses scorer at compress")
- `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md` (sibling reverse-engineering memo)
- `experiments/build_renderer_packed_payload_archive.py:332` (`encode_pose_qpose14_col_delta` — our existing QP1-style encoder, NOT score-aware)
- AGENTS.md "Build Discipline" — score-aware-everything is the paper-publishable narrative arc
