# phi3 S2SBS — Stride-2-Stem Byte-Stuffing Blindspot Audit

- Lane: `lane_s2sbs_blindspot_audit_20260513` (Phase 2)
- Council source: commit 896f1d79 (TRIPLET phi, O3)
- Evidence grade: **[macOS-CPU advisory]** — research signal only
- `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`
- Timestamp UTC: 2026-05-13T17:55:36Z
- Host: Primary
- Repo HEAD: `3074f7f6e6b6e859df6b2574fa39b1b613525fcc`
- Video: `/Users/adpena/Projects/pact/upstream/videos/0.mkv` sha256=`2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- SegNet weights sha256=`68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`
- PoseNet weights sha256=`0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`
- Sampled pairs: 6 (12 frames)
- JSON: `experiments/results/lane_s2sbs_blindspot_audit_20260513_20260513T175535Z/blindspot_capacity.json`

## Architectural derivation

- SegNet `tu-efficientnet_b2` conv_stem: `Conv2d(k=3, stride=2)` -> post-stem feature map at HALF resolution.
- SegNet `preprocess_input` resizes to (384, 512) -> stem out (192, 256). Nyquist preserved up to (192/2, 256/2) at the input resolution.
- PoseNet `fastvit_t12` stem: TWO consecutive stride-2 MobileOneBlocks -> total stride 4. Plus PoseNet preprocess does rgb_to_yuv6 with 2x chroma subsampling on U/V.
- Stems/resampling attenuate and alias high spatial frequency; they do NOT erase it. The audit measures leakage empirically rather than treating the FFT construction as proof.

## Scorer contract check

```json
{
  "frame_utils_seq_len": 2,
  "frame_utils_camera_size_wh": [
    1164,
    874
  ],
  "frame_utils_segnet_model_input_size_wh": [
    512,
    384
  ],
  "segnet_preprocess_output_bchw": [
    1,
    3,
    384,
    512
  ],
  "posenet_preprocess_output_bchw": [
    1,
    12,
    192,
    256
  ],
  "segnet_first_conv2d": {
    "path": "encoder.model.conv_stem",
    "kernel_size": [
      3,
      3
    ],
    "stride": [
      2,
      2
    ],
    "padding": [
      1,
      1
    ],
    "in_channels": 3,
    "out_channels": 32
  },
  "posenet_vision_first_conv2d": {
    "path": "stem.0.conv_kxk.0.conv",
    "kernel_size": [
      3,
      3
    ],
    "stride": [
      2,
      2
    ],
    "padding": [
      1,
      1
    ],
    "in_channels": 12,
    "out_channels": 64
  },
  "contract_source": "upstream/modules.py + upstream/frame_utils.py"
}
```

## Method

1. Decode the first N non-overlapping pairs from `upstream/videos/0.mkv` with `frame_utils.AVVideoDataset` / `yuv420_to_rgb`. seq_len=2, HxW=874x1164. This matches `upstream/evaluate.py` CPU-side frame semantics but is NOT a stratified full-video sample.
2. For each pair: build TWO independent HF perturbations (one for frame0, one for frame1) at camera scale. The perturbation is FFT-band-limited to lie OUTSIDE the camera-scale LF window whose bilinear-resampled image equals the SegNet stem Nyquist (192, 256). Amplitude swept across `--delta-amps`.
3. Run SegNet on frame1 (which is what the scorer uses) BEFORE and AFTER perturbation. Measure pixelwise argmax disagreement fraction.
4. Run PoseNet on the full pair BEFORE and AFTER perturbation. Measure first-6-pose MSE and max-abs delta.
5. Per delta amplitude, compute `bits_per_pixel_safe = log2(2*delta+1) * (1 - changed_frac)` for SegNet only AND for joint (SegNet + PoseNet MSE < 1e-5) safety. Project to bytes/frame at SegNet input resolution (384*512*3).
6. PRBS-31 stuffing demo at the largest joint-safe delta: encode +/-delta into Hermitian HF FFT coordinate pairs, integer-round to uint8 channel, decode via FFT, measure BER + scorer drift.

## Results

| delta_amp | changed_frac | pose_mse | bpp_seg_only | bytes/frame_seg | bpp_joint | bytes/frame_joint |
|----------:|-------------:|---------:|-------------:|----------------:|----------:|------------------:|
| 0.25 | 0.000011 | 1.078e-06 | 0.5850 | 43127.6 | 0.5850 | 43127.6 |
| 0.50 | 0.000020 | 4.300e-06 | 1.0000 | 73726.5 | 1.0000 | 73726.5 |
| 0.75 | 0.000031 | 9.676e-06 | 1.3219 | 97460.1 | 1.3219 | 97460.1 |
| 1.00 | 0.000042 | 1.717e-05 | 1.5849 | 116851.2 | 0.0000 | 0.0 |
| 1.50 | 0.000073 | 3.841e-05 | 1.9999 | 147445.2 | 0.0000 | 0.0 |
| 2.00 | 0.000100 | 6.785e-05 | 2.3217 | 171174.0 | 0.0000 | 0.0 |
| 4.00 | 0.000189 | 2.629e-04 | 3.1693 | 233668.0 | 0.0000 | 0.0 |
| 8.00 | 0.000383 | 9.847e-04 | 4.0859 | 301245.0 | 0.0000 | 0.0 |

## Aggregate

```json
{
  "stable_pixel_fraction_baseline_at_margin_tau": 0.9542872309684753,
  "margin_tau": 2.0,
  "max_bytes_per_frame_segnet_only_advisory": 147445.25,
  "max_bytes_per_frame_joint_advisory": 97460.14024164196,
  "n_pairs": 6,
  "n_frames": 12,
  "posenet_baseline": {
    "mean_norm": 33.721038818359375
  }
}
```

## PRBS-31 stuffing demo

```json
{
  "target_delta": 0.75,
  "n_bits_encoded": 65536,
  "n_bytes_encoded": 8192,
  "bit_error_rate_after_uint8_roundtrip": 0.4230804443359375,
  "binary_symmetric_channel_capacity_bits_per_encoded_bit": 0.01713973470498198,
  "effective_payload_bytes_before_ecc_upper_bound": 140.40870670321237,
  "hf_conjugate_pairs_available_one_channel": 381282,
  "hermitian_symmetric_fft_payload": true,
  "segnet_argmax_disagree_frac": 0.0,
  "posenet_pose_mse_first6": 6.401923879906235e-09,
  "posenet_pose_max_abs_first6": 0.000194549560546875,
  "single_pair_only": true,
  "channel_used_for_stuffing": "R",
  "notes": "Single-pair PRBS-31 stuffing demo at camera-scale HF band using Hermitian FFT coordinate pairs. BER includes uint8 quantization loss; BSC capacity is an optimistic pre-ECC upper bound. macOS-CPU advisory."
}
```

## Caveats and limits

- **[macOS-CPU advisory]** only. SegNet/PoseNet inference on macOS CPU is not 1:1 with [contest-CUDA] or Linux x86_64 [contest-CPU]. PoseNet MSE drift on MPS is 23x per CLAUDE.md, but here we used CPU (not MPS). Still: these numbers are research-signal only and CANNOT be used to promote, kill, or claim a score.
- Bilinear resize is information-lossy but the HF-band-zeroed pert leaks small amounts of energy into the LF band via the resize kernel. The empirically measured `changed_frac` is the ground-truth, not the theoretical FFT support.
- The PoseNet MSE threshold 1e-5 for joint safety is research-pick (PR106-era pose_avg is ~3.4e-5; 1e-5 is well below). The actual contest impact would need [contest-CPU] / [contest-CUDA] verification on a byte-closed archive.
- The byte-budget number is THEORETICAL (info-theoretic upper bound per pixel from amplitude alphabet). A real S2SBS codec also needs (a) a uint8 quantization channel — the PRBS demo measures BER under round-to-uint8, (b) an inflate-time recovery path, (c) error correction below uint8-rounding noise floor.
- The `bits_per_pixel_safe = log2(2*delta+1) * (1 - frac_changed)` formula is a heuristic upper bound (integer-amplitude alphabet size discount by argmax-flip fraction). Shannon-faithful capacity would use `0.5 * log2(1 + SNR)` after measuring the actual channel SNR (HF perturbation amplitude / leakage noise amplitude). The PRBS-31 BER at the largest joint-safe delta is the realized-channel lower bound on raw bytes after uint8 quantization — ECC reduces effective bytes further.
- The PRBS demo is single-pair only. Its BSC-capacity field is an optimistic pre-ECC upper bound; if BER approaches 0.5, the theoretical byte budget is not a usable payload channel. Cross-pair generalization needs a wider sweep before any codec is built.
- PoseNet visibility is real: `rgb_to_yuv6` preserves four luma subpixel channels (Y00/Y10/Y01/Y11), so 2x2 luma checkerboards remain visible to PoseNet even when U/V chroma averages cancel. S2SBS should prefer perturbations that are pair-consistent and pose-small, not merely SegNet-argmax-safe.
- This is not byte-closed: no archive bytes changed, no `inflate.sh` carries a payload, no decoder recovers bytes from inflated raw frames, and no exact evaluator consumed an S2SBS archive. Therefore every capacity number remains `score_claim=false` and `ready_for_exact_eval_dispatch=false`.
- Exact-eval blockers before contest relevance: build an archive pass that (1) embeds an ECC-coded payload into real Hermitian HF coefficients after renderer/video generation, (2) decodes the payload from uint8 inflated `.raw` frames with a manifest SHA, (3) proves payload byte recovery and frame count over all 600 pairs, (4) runs `scripts/pre_submission_compliance_check.py --contest-final --strict`, and only then (5) queues claimed [contest-CUDA] / paired [contest-CPU] eval.

## Go / no-go verdict

- **GO-FOR-PROTOTYPE**: joint-safe theoretical budget ~97460 bytes/frame and single-pair PRBS BSC upper bound ~140.4 bytes at BER=0.4231. Next step is a byte-closed ECC/Hermitian-HF prototype, not a score claim.

## Worst-case scenarios for the technique

- Bilinear-resize energy leakage: the FFT-band-limited HF perturbation can still leak ~5-15% of its peak amplitude into the LF band after the (cam_size -> 384x512) bilinear resize. This explains why high-delta perturbations show non-zero `changed_frac` even when the FFT is band-limited.
- PoseNet rgb_to_yuv6 + chroma subsampling integrates U/V over 2x2 blocks, but the four luma subpixel channels preserve checkerboard structure. PoseNet is the binding visibility risk for frame-pair perturbations.
- Argmax decision boundary: pixels with logit margin < tau will flip even from a small noise floor. The `stable_pixel_fraction` aggregate quantifies how much of the frame is safe vs boundary.
- uint8 quantization channel: an HF perturbation of amplitude < 0.5 has ~50% chance of being rounded away. The PRBS demo's BER under uint8-roundtrip is the realistic codec lower bound.

## Wire-in hooks (per CLAUDE.md Subagent coherence-by-default)

1. Sensitivity-map contribution: this audit provides a per-pixel (logit-margin, HF-blindspot) sensitivity that the renderer trainer / archive builder can consume to prioritize byte-spend AWAY from these pixels.
2. Pareto constraint: an additional rate-axis constraint `bytes <= rate_budget - s2sbs_recovered_bytes` becomes available when a codec ships.
3. Bit-allocator hook: per-tensor importance UNCHANGED (this is a frame-bytes side channel, not a tensor allocation).
4. Cathedral autopilot dispatch hook: N/A — research-only audit; no archive bytes change yet. A future S2SBS codec lane would register here.
5. Continual-learning posterior update: N/A — no empirical anchor produced (advisory only).
6. Probe-disambiguator: N/A — single defensible interpretation (architectural HF blindspot empirically measured).
