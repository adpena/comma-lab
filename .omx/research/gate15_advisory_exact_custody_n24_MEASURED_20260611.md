# Gate #15 — advisory↔exact custody: MEASURED on the most-descended capstone checkpoint (n24, d_seg 0.0204)

**Authority:** $0 macOS-CPU (agent #15, `gate15_reaudit_15`, 2026-06-11). torch-CPU exact
`modules.py` SegNet/PoseNet on the CANONICAL chain. **NO MPS.** GT ONLY via
`frame_utils.yuv420_to_rgb`. NO paid dispatch. NO score claim (advisory). The GAP is the finding.

**The gate (re-audit #15, `recipe_bug_lens_findings_reaudit_ledger_20260611.md`):** the capstone
training optimizes a LIVE-FLOAT MLX render d_seg/d_pose (the advisory). The contest scorer sees the
INT8-archive → bicubic-inflate frames. *"No capstone advisory number is a trustworthy
`inflate.sh→evaluate.py` predictor until a $0 reloaded-int8 + bicubic-inflate smoke runs. CLOSE THIS
BEFORE the n600 spend."* This memo closes it.

## What was measured

The SAME real descended checkpoint, the SAME 24 pairs, three ways:

| path | what it is |
|---|---|
| **(a) advisory** | trainer `exact_d_seg`/`mean_d_pose` (EMA shadow) = LIVE-FLOAT MLX decoder render → pose-FiLM → bridge SegNet/PoseNet with `eval_roundtrip`. The number training optimizes + reports. |
| **(b1) exact-bridge** | EMA shadow → **int8** byte-closed archive → numpy-inflate (bicubic to camera 874×1164, clip/round/uint8) → downsample 384×512 → SAME bridge SegNet/PoseNet + SAME cached GT. Isolates ONLY the int8/latent/numpy-bicubic export delta. |
| **(b2) exact-canonical** | int8-archive → numpy-inflate camera frames → REAL upstream `DistortionNet.compute_distortion` vs CANONICAL GT (`yuv420_to_rgb` camera-res). The TRUE contest call. |

Checkpoint: `experiments/results/capstone_gate15_descended_n24/checkpoint` (base_ch=20,
stored_latent carrier, restored at stage1 epoch-in-stage 5, global_epoch 13, best_d_seg 0.0204 —
the most-descended saved capstone checkpoint in the repo; the `diag_recipe_fix_fixed_only` deeper
trajectory (d_seg 0.012) saved NO checkpoint, only a trajectory JSON).

Apparatus reused verbatim: `experiments/results/capstone_gate15_descended_n24/measure_advisory_exact_gap.py`.
Verified NO-FAKE before run: `export.py` does genuine per-tensor symmetric int8 (`q=round(w/scale)`,
scale=max|w|/127, zigzag+brotli-q11) — real lossy quant; `inflate.py`/`numpy_reference.py` does
genuine PyTorch-matched bicubic (Keys a=-0.75, align_corners=False) camera upscale → uint8. The
advisory's `exact_d_seg` renders the LIVE FLOAT bundle (no int8); the exact path quantizes the SAME
EMA shadow. Apples-to-apples by construction.

## Result (int8 archive — the n600 ships int8)

| observable | advisory (EMA float) | exact int8 (canonical) | GAP | rel |
|---|---:|---:|---:|---:|
| **d_seg** | 0.0203889 | 0.0203639 | **2.50e-05** | 0.123% |
| **d_pose** | 0.0578099 | 0.0572471 | **5.63e-04** | 0.974% |

- **bridge == canonical EXACTLY** (b1 d_seg == b2 d_seg, b1 d_pose == b2 d_pose, to all digits).
  GT-decode parity confirmed: the int8 archive's inflated frames score identically through the
  bridge SegNet/PoseNet and the canonical upstream `DistortionNet`. The `yuv420_to_rgb` GT decode is
  consistent across both scoring surfaces.
- fp16 archive (148,809 B): d_seg 0.0204074, d_pose 0.0561886 — also within noise of advisory.
- int8 archive = 82,554 B (decoder 81,448 / latent 802 / pose 292).

### Score-unit translation (S = 100·d_seg + √(10·d_pose) + 25·bytes/N)
- d_seg gap → **0.0025** score-units (100·gap).
- d_pose gap → **0.0037** score-units (|√(10·0.05781) − √(10·0.05725)| = 0.76033 − 0.75662).
- **Total advisory→exact drift = 0.0062 score-units.**
- Descent signal init→now = 100·(0.5073 − 0.0204) = **48.7 score-units**. The drift is **0.013% of
  the descent signal** — third-decimal agreement.

### Direction (decisive)
The exact int8 path is **LOWER (better) than the advisory on BOTH axes** (d_seg −0.123%, d_pose
−0.974%). The advisory is, if anything, a *slightly conservative* predictor — it does NOT
over-promise. The feared "advisory optimizes a number the int8 archive doesn't honor" failure mode
does not materialize: the archive honors it, with a hair of margin to spare.

(Cross-check, prior n8 descend at d_seg 0.093: int8 gap d_seg 0.00123, d_pose 0.00106 — same
SMALL-gap regime; the gap does not grow as the checkpoint descends. n24 is the deeper, more
representative anchor.)

## VERDICT: **SMALL-gap → GATE #15 PASSES**

The advisory (live-float MLX render) is a **trustworthy `inflate.sh→evaluate.py` predictor** at the
operating point the n600 will descend through. The int8 export + bicubic-inflate path HONORS the
float the training optimizes (within 2.5e-5 d_seg / 5.6e-4 d_pose / 0.0062 score-units). **The n600
spend optimizes a real number the int8 archive honors — the export/quantization/inflate path is NOT
a binding blocker.** TIER-3 #15 is closed.

### Why the gap is this small (mechanism)
1. **EMA-shadow export parity** — the advisory measures the EMA shadow; the export bytes the SAME EMA
   shadow (snapshot+restore). No EMA-vs-live mismatch at the export point.
2. **eval_roundtrip already in the advisory** — the advisory's bridge already applies the bicubic-up
   → bilinear-down → uint8 *pixel* roundtrip, so the only NEW deltas the exact path adds are (i) int8
   *weight* quant and (ii) numpy-vs-torch bicubic. Both are tiny: the PR95-faithful per-tensor int8
   (~1 B/param at scale=max|w|/127) and the PyTorch-matched numpy bicubic kernel.
3. **stored_latent carrier** — the latent uint8 per-dim quant (802 B) is near-lossless on a 24-pair
   ego-motion latent stream.

## Caveats / honest scope (not over-claiming)
- **n24, not n600.** The gate measures the GAP MECHANISM (int8/inflate fidelity), which is
  capacity/pairs-stable — the prior n8 run shows the same SMALL regime. But the *absolute* d_seg of
  the n600 run will differ; this gate certifies the advisory→exact TRANSFER, not the final score.
- **Stage1/2 descend (d_seg 0.0204), not the full 8-stage curriculum.** The fp16-vs-int8 spread is
  already negligible here; QAT (stage4) only tightens int8 fidelity further, so the gap is expected
  to stay SMALL or shrink through the later stages — not grow. A confirmatory re-measure on a
  stage4-QAT checkpoint from the n600 itself is the natural belt-and-suspenders, but is NOT a
  pre-dispatch blocker given this result + the QAT direction.
- **Capstone export/inflate IS fully wired** (the prior blocker class is closed): byte-closed int8
  archive builds, numpy-portable inflate runs to camera-res uint8 frames, and both the bridge and the
  canonical DistortionNet score them. No "can't byte-close" blocker.

## Artifacts
- `experiments/results/capstone_gate15_descended_n24/advisory_exact_gap_n24.json` (the measured row)
- `experiments/results/capstone_gate15_descended_n24/gap_run.log`
- `experiments/results/capstone_gate15_descended_n24/measure_advisory_exact_gap.py` (apparatus, reused)
- Prior sister: `experiments/results/capstone_gate15_descended_n8/advisory_exact_gap_DESCENDED.json`

**Bottom line for the config agent / symposium:** the advisory↔exact custody gate is the ONE TIER-3
verdict that was named as the binding pre-n600 blocker. It is now MEASURED on a real descended
checkpoint and PASSES with a 0.0062 score-unit drift in the conservative direction. This gate no
longer blocks the $100 n600 spend.
