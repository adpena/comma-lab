# commaVQ lossless compression at the entropy ceiling

**Score 4.0009** on the full commaVQ challenge eval set — all 5,000 minutes of
driving-video tokens, official submission format, and **every segment verified
bit-exact by the shipped decompressor**. The frozen leaderboard's winning entry
displays as 4.0; this repo lands on top of it and shows, with measurements,
why nobody goes meaningfully higher with the free model.

| | |
|---|---|
| **Score (official format)** | **4.0009** — 239.9 MB zip for 960 MB of tokens |
| **Bits per token** | 2.4994 achieved · 2.5119 measured model ceiling |
| **Losslessness** | 5,000 / 5,000 segments bit-exact, 0 mismatches |
| **Scale** | 768M tokens · 109 GPU-h per pass on one RTX 5070 Ti 16 GB |

## What's being compressed

Each minute of driving video is 1,200 frames, each encoded by comma's VQ-VAE
into 128 tokens on an 8×16 grid (10 bits each). Consecutive frames share about
a third of their tokens verbatim — the temporal structure a world model
predicts, and prediction is exactly what compression spends:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/dataset_example_dark.svg">
  <img alt="A decoded driving frame, its 8 by 16 grid of token IDs, and the change map to the next frame showing 35 percent of tokens unchanged" src="assets/dataset_example_light.svg">
</picture>

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/leaderboard_dark.svg">
  <img alt="Leaderboard: this repo at 4.0009 above the winner's 4.0, mune-io 3.7, self-compressing NN 3.4, AC+GPT 3.0, zpaq 2.3, lzma 1.6" src="assets/leaderboard_light.svg">
</picture>

## The two numbers that explain everything

**1. Context policy is worth 0.4 bits/token — the least-discussed lever.**
gpt2m predicts the next token from up to 20 frames of history, but its learned
absolute positions force periodic context restarts. The first frame after a
restart costs ~8 bits/token; the model saturates by ~8 frames of context:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context_saturation_dark.svg">
  <img alt="Cross-entropy vs frame position in a fresh context window: 8.1 bits/token with no context, saturating to about 2.46 by frame 8" src="assets/context_saturation_light.svg">
</picture>

Restart windows average **2.908** bits/token; overlapped windows at stride 5
(every coded token sees ≥15 frames of context) reach **2.4993** — the
difference between score 3.44 and score 4.00. In the lockstep codec design the
overlap is nearly free: sequential decode steps dominate and their count is
independent of stride.

**2. The dataset-wide mean is 2.5119 bits/token — and only the full dataset
can tell you that.** Per-segment cross-entropy spans 1.1 to 4.6 with a heavy
right tail (std 0.57), so 30-segment samples mislead by ±0.15 score in either
direction. Published sample-based estimates range 2.415–2.92 on the same
distribution; the full measurement pins the achievable score at 3.98–4.00
before a single byte is coded:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/segment_distribution_dark.svg">
  <img alt="Histogram of gpt2m cross-entropy across all 5,000 segments: heavy-tailed, mean 2.512 bits/token" src="assets/segment_distribution_light.svg">
</picture>

## Why not a better model? (measured, not vibes)

Every published "improvement" lever, falsified three independent ways:

| Lever | Result | Who |
|---|---|---|
| LoRA fine-tune on eval data | +0.04% (noise), train loss flat | ykstorm |
| Test-time training (causal, online) | −2.5% to −3.6% (worse) | meetr1912 |
| **Full fine-tune on public splits 2–3** | **−0.9% (worse), train loss flat** | **this repo** |
| Context mixing (adaptive filters / histograms) | −1.5% to 0% | ykstorm, meetr1912 |
| Temperature calibration | T=1.0 already optimal | meetr1912 |
| Longer context | saturates by ~8 frames | meetr1912, this repo |
| Cross-segment dedup | 0 duplicate frames in 72k | meetr1912 |
| fp32/fp16/bf16 | fp16 == fp32 CE exactly; bf16 +0.06% | this repo |

gpt2m was trained on 3,000,000 minutes of this exact distribution; the
5,000-minute eval set holds no residual signal a cheap adapter can capture.
Compression *is* prediction, and the prediction is already as good as the free
model gets.

## Method

1. **Windows:** 20-frame contexts starting every 5 frames, positions restarted
   per window (BOS at multiples of 129, as trained). The first 15 frames of
   each window are known context, prefilled in one batched forward; the last 5
   are coded token-by-token. BOS tokens are deterministic and never coded.
2. **Probabilities:** fp16 forward → float32 logits → float64 softmax over the
   1,024 content symbols (+2⁻¹⁶ uniform floor) → constriction range coder.
   Feeding raw float64 probabilities costs +0.0005 bits/symbol vs entropy; a
   16-bit pre-quantization costs 25× more (measured), so there isn't one.
3. **Losslessness by construction:** the decoder runs the *identical* batched
   op sequence — same batch sizes, same prefill/step shapes, same kernels —
   so its logits, and therefore its probability tables, match the encoder's
   bit-for-bit. 44 segments decode in lockstep per batch. Verified end-to-end:
   the shipped zip's own `decompress.py` reproduced all 5,000 segments
   exactly. Scope: same GPU/driver/library stack (pinned), the standard
   contract for neural entries.
4. **Runtime is memory-bandwidth physics, not FLOPs:** each decoded token
   reads its sequence's entire KV cache (24 layers × 2,584 positions ≈ 253 MB;
   ~11 GB/step at batch 44). Measured 18 ms/step ≈ 75% of the theoretical
   bandwidth floor; torch.compile is a no-op here, and batch size only
   amortizes weight reads. 109 h per pass, both passes run.
5. **Accounting:** zip carries the range-coded payload (stored), a 21 KB
   manifest, and `decompress.py`. Model weights download from HuggingFace at
   decode time, as the winner's entry did and as `commavq/utils/gpt.py` does.
   File names are recovered from the dataset, exactly as comma's baseline
   decompressor does.

## Files

- `measure_ce.py` — teacher-forced CE measurement (ceiling + context curves)
- `codec.py` — lockstep batched encode/decode core (round-trip self-test)
- `encode_all.py` / `package.py` — resumable full-run driver + zip assembly
- `submission_decompress.py` — official-format decompressor (ships in the zip)
- `verify_sample.py` / `compare_all.py` — bit-exact verification tooling
- `finetune_probe.py` — the falsified public-splits fine-tune probe
- `profile_step.py` — where the milliseconds go
- `make_figures.py` / `make_dataset_figure.py` — regenerate the figures above
- `results/` — measurement logs + full per-segment CE data

The 239.9 MB submission zip is attached to the GitHub release (v1.0).

## Reproduce

```bash
# environment: CUDA GPU (16 GB), python 3.13, pinned deps
pip install -r requirements.txt
git clone https://github.com/commaai/commavq ../commavq  # model def + eval scripts

# 1. the ceiling measurement (~6 h)
python measure_ce.py --segments 5000 --stride 10 --dtype fp16

# 2. round-trip self-test (minutes)
python codec.py --selftest --segments 2 --stride 10 --frames 60

# 3. full encode (~109 h, resumable) + package
python encode_all.py --batch 44 --stride 5 --out encoded
python package.py --batch 44 --stride 5

# 4. verify: decode the actual zip and compare everything (~109 h, resumable)
mkdir -p verify_full/out && (cd verify_full && unzip ../compression_challenge_submission.zip)
OUTPUT_DIR=verify_full/out python verify_full/decompress.py
OUTPUT_DIR=verify_full/out python compare_all.py
```
