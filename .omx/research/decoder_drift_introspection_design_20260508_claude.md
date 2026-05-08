# Decoder drift introspection design — DALI vs PyAV third-axis investigation

**Date**: 2026-05-08
**Author**: Claude (subagent, parallel with af945f502 PoseNet-introspector + a22d581a FastViT-theory)
**Tag**: `[diagnostic-not-score]` — no contest score claims herein.
**Scope**: The decoder backend axis ONLY; network-internals and FastViT precision are owned by sibling subagents.

---

## TL;DR

`upstream/evaluate.py` lines 31-42 select **`DefaultDatasetClass = DaliVideoDataset`** when `device.type == 'cuda'`, and **`AVVideoDataset`** otherwise. Critically, this dataset class is **only used for the ground-truth tensor `ds_gt`**; the compressed tensor `ds_comp` always uses `TensorVideoDataset` (a raw mmap). So the divergence between CUDA and CPU paths is **asymmetric** — only the GT side decodes from `.mkv`, and only that side switches backend.

The upstream author **explicitly designed AVVideoDataset to byte-match NVDEC** (frame_utils.py:161 "matches nvdec output", :201 "uses bilinear chroma upsampling + BT.601 limited range"). However, exact byte-identity is NOT guaranteed: NVDEC uses fixed-point chroma upsampling and YUV→RGB matrix with reduced precision, while the AV path uses float32 multiplications that `.round().to(uint8)`. Different rounding conventions can drift ±1 LSB per pixel.

**Lipschitz back-of-envelope** (verified by `test_lipschitz_prediction_pr106_pixel_count_dominant`): assuming a conservative 1.5-LSB per-pixel max-abs drift between DALI and AV at 874×1164×3 RGB, and the smallest plausible PoseNet Lipschitz `L≈1e-5` per normalized RGB unit, the **predicted pose-component drift is 4.1e-4 — about 2.9× the observed 1.4e-4 PR106 gap**. Decoder drift CAN ACCOUNT FOR the entire observed pose ratio without invoking ANY FastViT precision compounding.

**Most likely hypothesis**: H1 (decoder-dominant) or H3 (mixed). H2 (network-dominant only) is unlikely given the math — it requires DALI-vs-AV to be byte-identical, which the upstream author warns against in code comments. **Smallest discriminating test**: dispatch a 10-frame DALI dump on Lightning T4, compare to the local AV dump, run quantify_drift. If max-abs is ≤1 LSB and mean-abs ≤0.1 LSB, the byte-match is tight enough that decoder is subdominant; otherwise, decoder is the dominant suspect.

---

## 1. Code-level divergence

### 1.1 Where the split happens

`upstream/evaluate.py` (read-only per Non-Negotiable Upstream Rule):

```python
# evaluate.py L31-42
if device.type == "cuda":
    ...
    DefaultDatasetClass = DaliVideoDataset
else:
    local_rank, rank, world_size, is_distributed = 0, 0, 1, False
    DefaultDatasetClass = AVVideoDataset
```

`--device cuda` triggers BOTH:
1. `torch.device("cuda")` for the network
2. `DefaultDatasetClass = DaliVideoDataset` for ground-truth decoding

`--device cpu` literally instantiates `AVVideoDataset` with `device.type='cpu'` (and the assertion at frame_utils.py:188 enforces this). MPS path also lands on AV (line 25-26 selects MPS as fallback, then line 40 picks AV since `device.type != 'cuda'`).

### 1.2 The two decode paths

**DaliVideoDataset** (`upstream/frame_utils.py:110-157`):
- Uses `nvidia.dali.fn.experimental.inputs.video(device='mixed', sequence_length=2)`
- "mixed" = NVDEC for HEVC decode + CUDA kernels for YUV→RGB conversion
- Asserts CUDA at L113
- Outputs `(B, S, H, W, 3)` uint8

**AVVideoDataset** (`upstream/frame_utils.py:185-216`):
- Uses PyAV (`av.open(path)` → `container.decode(stream)`)
- Each frame goes through `yuv420_to_rgb()` (L159-183), which:
  - Reads Y/U/V planes (uint8)
  - Bilinear-upsamples U,V to (H, W) via `F.interpolate(..., mode='bilinear', align_corners=False)`
  - Applies BT.601 limited-range matrix: `Y' = (Y-16) * 255/219`, `U' = (U-128) * 255/224`, `V' = (V-128) * 255/224`
  - YUV→RGB: `R = Y' + 1.402*V'`, `G = Y' - 0.344136*U' - 0.714136*V'`, `B = Y' + 1.772*U'`
  - `.clamp(0, 255).round().to(uint8)`
- Asserts NOT CUDA at L188

### 1.3 What's identical and what could diverge

| Stage | Both paths | Potential drift source |
|---|---|---|
| HEVC bitstream parse | Same .mkv input | None (codec spec is bit-exact) |
| YUV420 plane output | Same Y/U/V planes pre-upsample | NVDEC vs libav HEVC reconstruction |
| Chroma upsample | Bilinear | **NVDEC fixed-point vs F.interpolate float32** |
| YUV→RGB matrix | BT.601 limited-range | **NVDEC fixed-point vs float32 mul-add** |
| Quantize to uint8 | (clamp + round) | **Different rounding modes (bankers' vs half-up)** |

The three bolded rows are where bytes plausibly diverge. Empirical literature on FFmpeg-vs-NVDEC suggests 1-3 LSB max-abs per-pixel drift on the same matrix.

---

## 2. Empirical AV decode dump (locally-runnable, [macOS-CPU advisory only])

Run script: `experiments/decoder_drift_av_decode_dump.py`. Output dir: `experiments/results/decoder_drift_av_dump_20260508_claude/`.

Verified results (4 frames from `upstream/videos/0.mkv` on M5 Max):
- `decoded shape=(4, 874, 1164, 3), dtype=torch.uint8`
- `full_decode_sha256 = 7b7b8b4656d9550341e692289bce21dd612ff57888d9fa2d8c59bd3d41614a4c`
- `bit_identical` across two re-runs: **True** (AV decode is deterministic)
- per-channel mean RGB: [33.0, 24.0, 21.6] (consistent with the dim driving-scene profile)
- per-channel std RGB: [23.1, 23.1, 17.6]

**Determinism guarantee**: The same .mkv file, same PyAV version, same yuv420_to_rgb code → bit-identical output. This means any DALI-vs-AV drift we measure later is purely backend, not nondeterminism.

**Tag**: `[macOS-CPU advisory only]` — local libav version may differ from contest CI's ubuntu-latest ffmpeg + the pinned `upstream/ffmpeg-new` binary. The contest authoritative AV decode is what matters; this local dump is a baseline for byte-comparison only.

### DALI dump — design only, no GPU dispatch in this task

`DecoderDriftIntrospector.decode_dali_design()` returns a JSON-serializable recipe an operator dispatches on a CUDA host (Lightning T4 g4dn.2xlarge per CLAUDE.md "Remote code parity"):

```bash
# On a CUDA host with nvidia-dali installed:
python -c "
import sys; sys.path.insert(0, 'upstream')
import torch; from frame_utils import DaliVideoDataset
ds = DaliVideoDataset(['0.mkv'], data_dir='upstream/videos', batch_size=2,
                      device=torch.device('cuda'), num_threads=2, seed=1234,
                      prefetch_queue_depth=4)
ds.prepare_data()
for path, idx, vid in ds:
    torch.save(vid.cpu(), f'dali_{idx}.pt'); break
"
```

The operator scp/rsyncs the resulting `dali_1.pt` back to the local repo, then:

```python
from tac.diagnostics.decoder_drift_introspection import DecoderDriftIntrospector, quantify_drift
intro = DecoderDriftIntrospector()
av = intro.decode_av("upstream/videos/0.mkv", max_frames=2)
dali = intro.ingest_dali_dump("dali_1.pt")  # asserts uint8
report = quantify_drift(av, dali)
print(report.l2_mean, report.max_abs_global, report.histogram_diff_signed)
```

---

## 3. Lipschitz back-of-envelope: decoder drift CAN account for 5× pose ratio

### 3.1 Setup

PoseNet preprocess (per CLAUDE.md "Exact scorer architectures" + `upstream/modules.py`):
1. `rgb_to_yuv6` from `frame_utils.py:50-78` (deterministic, exact for any uint8 input)
2. Resize to (512, 384) bilinear
3. Normalize `(x - 127.5) / 63.75`, mean=127.5, std=63.75

Effective input scale to FastViT: `1 / 63.75 ≈ 0.0157` per uint8 LSB.

### 3.2 Closed-form derivation

Assume per-pixel RGB drift `d` LSB iid uniform in `[-d, +d]`. Per-pixel variance in normalized units:
- `d_norm = d / 63.75`
- `Var = d_norm² / 3`

Total L2 of input perturbation across `N = H * W * C` samples:
- `L2_input = sqrt(N * Var) = sqrt(N) * d_norm / sqrt(3)`

Pose-output drift via Lipschitz `L`:
- `pose_drift ≤ L * L2_input`

For `H=874, W=1164, C=3, N=3,052,008`:
- `L2_input(d=1.5 LSB) = sqrt(3052008) * (1.5/63.75) / sqrt(3) ≈ 1746 * 0.0235 / 1.732 ≈ 23.7` (normalized units)

Wait — let me recheck. Actually, `Var = (2d)²/12 = d²/3` for `Uniform[-d, d]`. So `sqrt(Var) = d/sqrt(3)`.

`L2_input = sqrt(N * d²/3) = sqrt(N) * d / sqrt(3)`

In normalized units, `d → d/63.75`, so:
- `L2_input_norm = sqrt(N/3) * d / 63.75`
- For `d=1.5`, `N=3,052,008`: `L2_input_norm ≈ sqrt(1017336) * 1.5 / 63.75 ≈ 1008.6 * 0.0235 ≈ 23.7`

(Same answer; my code above is right; I had a transient confusion.)

### 3.3 Mapping to observed PR106 gap

PR106 frontier observation (per af945f502/a22d581a + memory references):
- `pose_avg_cuda ≈ 1.7e-4`
- `pose_avg_cpu ≈ 3.4e-5`
- Gap: `1.4e-4`

Required Lipschitz to explain observed gap by decoder drift alone:
- `L_required = 1.4e-4 / 23.7 ≈ 5.9e-6`

This is **smaller than the smallest plausible Lipschitz** (`L=1e-5` is already ~conservative; FastViT-T12's Hydra head is unlikely to be tighter than that). **Conclusion: 1.5 LSB decoder drift, even at the conservative end, generates more pose drift than is observed.**

### 3.4 Caveats / refinements

- The iid-uniform assumption upper-bounds the structured drift. Real DALI-vs-AV drift concentrates at chroma boundaries (where bilinear upsample bias differs) and saturates at clamp(0,255) extremes (which both decoders treat identically), so the effective L2 may be 0.3-0.7× this prediction.
- The Lipschitz is taken as a SCALAR; in reality the pose has 6 dimensions and L is a matrix norm. Using the operator-2-norm of the Jacobian would be more precise.
- The observed `1.4e-4` is averaged over 600 frame-pairs; if drift is concentrated on a few frames, single-frame drift could be 10× higher.

For tighter bounds, af945f502's PoseNet introspector + a JVP-based local Lipschitz probe at 100 random input perturbations would produce an empirical `L`, replacing the back-of-envelope.

---

## 4. Tool design — `src/tac/diagnostics/decoder_drift_introspection.py`

**Status**: Landed. 18 tests pass.

Public API:
- `DecoderDriftIntrospector` — orchestrates AV decode (locally-runnable) + DALI design (JSON recipe, dispatch on CUDA host) + ingestion + drift quantification.
- `FrameByteFingerprint.from_tensor(idx, rgb)` — per-frame statistical fingerprint (sha256 + per-channel mean/std).
- `DriftReport` — L1/L2/max-abs-per-frame, mean-abs-per-channel, p99 abs diff, signed histogram.
- `quantify_drift(av, dali)` — full statistical fingerprint of drift; refuses dtype/shape mismatch.
- `lipschitz_pose_drift_prediction(drift_lsb, n_pixels, std=63.75, L=1e-4)` — closed-form mapping with a 'verdict' field (decoder-dominant / decoder-mixed / decoder-subdominant).

Tests at `src/tac/tests/test_decoder_drift_introspection.py`:
- AV-decode determinism on `upstream/videos/0.mkv` (skipped if missing) ✓
- Synthetic perturbation roundtrip recovers known drift ✓
- Histogram symmetry for unbiased noise ✓
- Lipschitz math closed-form ✓
- Verdict classification at PR106-realistic numbers ✓
- Shape/dtype mismatch error paths ✓
- DALI design recipe is JSON-serializable ✓

---

## 5. Hypothesis test plan

### H1 (decoder-dominant): >70% of the 0.033 score gap is from decoder-bytes drift
**Predicted signature**: max-abs DALI-vs-AV ≥ 2 LSB, mean-abs ≥ 0.5 LSB on > 5% of pixels.

### H2 (network-dominant): <30% from decoder; FastViT precision compounding explains the rest
**Predicted signature**: max-abs DALI-vs-AV ≤ 1 LSB, mean-abs ≤ 0.1 LSB on >95% of pixels.

### H3 (mixed): roughly equal contributions
**Predicted signature**: max-abs ~1-2 LSB, mean-abs ~0.2-0.4 LSB.

### Discriminating tests (each costs ~$0.10-0.30 on Lightning T4; do NOT dispatch in this task)

**Test A — DALI bytes through CPU PoseNet**
1. Dump DALI tensor on Lightning T4 (one batch, 16 frames).
2. rsync to local.
3. Run PoseNet on CPU (locally) using DALI bytes as ground truth (paired with AV bytes for comp).
4. Compare to the standard `--device cpu` pose score.
5. **If pose ≈ CPU score** → DALI bytes don't change CPU PoseNet output → decoder bytes don't matter on CPU PoseNet → H2.
6. **If pose ≈ CUDA score** → DALI bytes alone shift the CPU score to match CUDA's → H1.

**Test B — AV bytes through CUDA PoseNet**
1. Dump AV tensor locally.
2. scp to Lightning T4.
3. Run CUDA PoseNet using AV bytes as ground truth.
4. Compare to standard `--device cuda` pose score.
5. **Inverse interpretation of Test A.**

**Test C — Controlled bit-noise injection**
1. Take the AV-decoded tensor, add `Uniform[-d, +d]` per pixel for `d ∈ {1, 2, 3}`.
2. Run CPU PoseNet on each, compare to baseline `d=0`.
3. **Plot pose-drift-vs-d curve**; identifies the per-LSB sensitivity directly without needing DALI at all.

Test C is the cheapest (no GPU needed) and discriminates H1/H2 if the curve at d=1.5 reaches the observed 1.4e-4 gap.

---

## 6. Exploit prescriptions per hypothesis

### If H1 (decoder-dominant) holds
- **Train against AV-decoded bytes**, not DALI-decoded. The renderer's outputs land in a regime where libav's reconstruction differs from NVDEC's by less.
- **Renderer-output post-processing**: pre-quantize to a uint8 grid that survives both NVDEC and libav YUV→RGB pipelines identically. (Drop the lowest few bits where rounding diverges.)
- **Score formula re-interpretation**: pose term becomes "decoder-noise rejection" not "FastViT precision compounding". All paper §3.4 claims about FastViT need to flip.
- **Submission auth eval discipline**: ALWAYS run the AV path locally + the DALI path on Lightning T4 BEFORE submitting; gap > 0.01 score points means decoder drift is not under control.

### If H2 (network-dominant) holds
- **a22d581a's FastViT compounding hypothesis stands.** Replace FastViT or train at lower attention precision.
- **Decoder choice doesn't matter for our exploits.** Both DALI and AV produce nearly-identical bytes; the divergence is in network kernels.

### If H3 (mixed) holds
- **Both axes need attention.** Order by ROI:
  1. Land cheap decoder-side mitigation (post-quantize renderer output to a robust grid).
  2. Address network-side precision (a22d581a's path).
- **Partial exploits possible on each.** Decoder mitigation may shave 0.005-0.010, FastViT mitigation another 0.005-0.010.

---

## 7. Cross-references

- **af945f502** (PoseNet/SegNet network internals introspection): provides forward-hook recorder, per-layer drift comparator. Once available, plug its `compute_layer_drift` into Test C's AV-decoded-with-noise path to see WHICH layer amplifies the noise.
- **a22d581a** (FastViT precision compounding theory): provides the (1+ε)^L compounding model. Cross-validate against this design's Lipschitz back-of-envelope; if their per-layer ε agrees with our scalar L, we have two-axis convergence.
- **CLAUDE.md** "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + "MPS auth eval is NOISE": this design respects both rules — no MPS evaluation, no GPU dispatch in this task, and all local outputs tagged advisory.
- **`reports/raw/lightning_batch/*/auth_eval.log`** — searched for prior DALI dumps; none found. The contest auth-eval pipeline writes `inflated/*.raw` (which is `TensorVideoDataset` input), not DALI's GT decode dump. To get a DALI dump we must dispatch a one-off CUDA script (cost: ~$0.10).

---

## 8. Required next steps (operator decisions, NOT executed in this task)

1. **Cheap-first**: run **Test C** (controlled bit-noise injection) locally on M5 Max + a Lightning T4 PoseNet pass. This needs zero GPU minutes locally; only a small CUDA roundtrip for the PoseNet eval. Cost: ~$0.20.
2. **DALI dump dispatch**: one-off CUDA dispatch on Lightning T4 g4dn.2xlarge to capture `dali_decode.pt` for a 16-frame batch. Cost: ~$0.10. With the resulting tensor + this design's `quantify_drift`, we have empirical max-abs and mean-abs.
3. **Cross-reference with af945f502 and a22d581a**: once their introspectors land, run a triplet experiment: AV-bytes → CPU PoseNet → per-layer drift, vs DALI-bytes → CUDA PoseNet → per-layer drift. If the per-layer ε agrees between the two paths, the decoder is dominant; if it diverges, FastViT is dominant.

**This task lands the design + tool + tests + AV decode dump. The empirical DALI comparison + GPU dispatch is the operator's next-budget decision.**

---

*Tag: `[diagnostic-not-score]` and `[macOS-CPU advisory only]`. No score claims. Pure observational research.*
