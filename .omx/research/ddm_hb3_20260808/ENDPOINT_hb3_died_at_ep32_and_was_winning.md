# hb3 endpoint: the run died at ep32/60 — one epoch INTO the phase that was winning

Settles roadmap step 1 (MATCH). Answer: **not closed — partially BEATEN and abandoned mid-descent.**
`[macOS-MLX/MPS research-signal]` for the training; byte figures are the trainer's estimates, not an
eval row. `score_claim=false`.

## 1. Status

`ddm_hb3_20260808/run` reached **epoch 32 of 60** at 1,397 s and stopped. No `.done` receipt
(`.omx/tmp/codex_runs/gt_hpac_stage2_mps.done.done` absent), no error in the log, no surviving process.
**Cause NOT diagnosed.** The sister `cpu_control` died after epoch 0 (7 min) — both died, which points
at a systematic killer rather than an OOM (the CPU control had barely allocated).

`PR130_BASE_ROADMAP.md` quotes this run at **ep16, tokens 116,560** as MEASURED-IN-PROGRESS. That row
is real but it is on the wrong side of a phase transition (§3).

## 2. Full trajectory (MEASURED, from the run's own log)

PR130 reference for the same two sections: tokens 116,980 + hpac weights 20,179 raw = **137,159 B**.

| ep | phase | bpp | token B | model B | joint B | vs PR130 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | — | 0.0079927 | 117,858 | 27,026 | 144,884 | +7,725 |
| 8 | continuous | 0.0079231 | 116,832 | 23,207 | 140,039 | +2,880 |
| **16** | continuous | 0.0079047 | 116,560 | 21,385 | 137,945 | **+786** ← roadmap's row |
| 22 | continuous | 0.0079242 | 116,847 | 20,631 | 137,478 | +319 |
| 30 | continuous | 0.0079626 | 117,414 | 20,206 | 137,620 | +461 |
| **32** | **discrete_qat** | **0.0078306** | **115,468** | 20,360 | **135,828** | **−1,331** |

- **ΔS at ep32 = 25 · (−1,331) / 37,545,489 = −0.0008863.**
- bpp 0.0078306 vs PR130's 0.0079332 = **1.29% better** on the token stream.

## 3. Why the ep16 row understates it: the QAT phase is where the work happens

`qat_start = max(1, floor(epochs·(1−qat_fraction)) + 1)` = 31 for `epochs=60, qat_fraction=0.5`.

- 30 `continuous` epochs bought **−7,264 B** total (144,884 → 137,620), non-monotonically.
- The FIRST `discrete_qat` epoch bought **−1,792 B alone** (137,620 → 135,828) — more than the previous
  14 epochs combined.

The run was killed with **~28 QAT epochs unrun**, at the steepest part of its descent. The bit-depth
histogram shows the mechanism: tensors migrating out of 8-bit (`"8": 344 → 4 → 6`) and redistributing
across 3–7 bits under the discrete constraint.

`top1_error` is flat at ~0.00196 the whole way. That is the AR model's PREDICTION error, not a
reconstruction error — HPAC codes the exact residual, so the coding is lossless throughout and every
byte gained here costs **zero** d_seg.

## 4. Why the comparison is now SAFE (it was flagged UNSAFE)

The ADDENDUM flagged exactly this comparison — ours vs 137,159 B — as UNSAFE until re-derived on DALI
labels, because hb3 trained on our AV-decoded GT cache. `ADDENDUM2` measured the rate spread across
provenances at **192 B / 0.047%** on a 410 KB field. Scaled to this 137 KB comparison that is ~65 B of
uncertainty against a **−1,331 B** lead — **20×**. The rate half of the comparison is safe.
(The distortion half remains unsafe; that is where the 61.25% seg delta lives.)

## 5. Correction to my own earlier read

I wrote that this run had "loop-end-only saving," forbidden by the per-stage-checkpoint rule. **Wrong.**
`train_hpac_self_compress.py:212` writes `<save>.latest.pt` inside the eval block, i.e. every
`eval_every=2` epochs. The discipline held, and that is exactly why the ep32 state survived to be
recovered. What actually failed is that the process died and nobody read the endpoint for a day — a
WATCHING failure, not a checkpointing one.

## 6. Resume fired

`resume_qat/`, pid 23657: `--init <latest.pt> --epochs 28 --qat-fraction 1.0` → `qat_start=1`, so QAT is
active from the first resumed step, completing the originally-planned 60. Saved to a DISTINCT
stage-encoded path (`..._qat_e28_from_ep32.pt`) so the ep32 state is preserved, not overwritten.

**Positive control PASSED bit-identically.** The resume's epoch-0 eval reproduced
`bpp 0.007830649759668974`, `joint 135,828 B`, `top1_error 0.0019337802463107638` and the identical
bit-depth histogram — the 9 learned `bit_depth` tensors restored rather than resetting to the 8.0 init.

## 7. What is NOT checked

- **Why it died.** Not diagnosed. Both the MPS run and the CPU control died; no receipt, no traceback.
- The joint-byte figures are the trainer's own estimates (`bpp × 117,964,800 / 8` + packed model size),
  not a packed archive measurement. A real archive needs the hb2 pack path.
- No score claim. These are rate-side estimates on a lossless coder; d_seg is untouched by construction
  but has not been re-measured on this checkpoint.
