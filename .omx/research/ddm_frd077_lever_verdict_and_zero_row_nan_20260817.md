# The FiLM-row lever is seg-neutral — and zeroing a FiLM row makes the TRAINER's quantiser emit NaN

> ⚠ **CORRECTED 2026-08-17, same day, after the corpus recall I owed before writing this.**
> The original headline said "the **deployed** quantiser" and §"Who else this blocks" claimed this
> NaN blocked the FiLM-row-sparsity rate family. **Both are refuted.** The defect is real and its
> arithmetic is unchanged, but it lives on the TRAINER path only; the shipping receiver never
> touches it, and `ddm_sf1` (landed the same day, same directory) measured the whole family
> through that receiver. Corrections are inline below, marked ⚠. See §"Who else this blocks".

**Status:** MEASURED (2026-08-17, MAIN, $0 local Metal).
**Axis:** `[macOS-MPS training-signal]` `quantized_exact_seg`, EMA shadow, n600, eval-batch 8.
**No score claim. Frontier untouched.** Payloads under `/Volumes/APDataStore/pact/ddm_ce1/`.

## ANSWER FIRST

Two results, one of them mine to own.

1. **Drain lever 1 (`--film-row-dropout 0.077`) is SEG-NEUTRAL.** Matched single-variable A/B
   against `EF3000` (identical init 33,757 flips, identical config except the one flag):

   | | endpoint flips vs init | per-eval lever cost |
   |---|---:|---|
   | `EF3000` control | −2,286 | — |
   | `FRD077` | −2,128 | mean **+119**, stdev 873, n=30 |

   **+158 endpoint = 0.18σ** against the 855-flip two-run difference band (605 × √2).
   `packed_parameter_bytes` **40,252 on BOTH arms** — byte-neutral by itself.
   Wall clock 1,451 s vs 1,445 s (+0.4%). The lever passes its gate: it costs nothing.

2. **The lever's PAYOFF cannot be measured the way I measured it, and the reason is a real
   defect.** Zeroing *any* FiLM row **in the fp32 weights, then quantizing**, makes the trainer's
   lifted quantiser emit **NaN** frames. ⚠ The shipping receiver does not do this — see below.

## The defect, exactly

`src/tac/pr130_lift/lifted/train_semantic_quantized.py:50-54`:

```python
scale = source.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / limit
scale = scale.to(torch.float16).float()          # <-- destroys the guard above
normalized = (source / scale).clamp(-limit, limit)
```

An all-zero row has `amax = 0`. The `clamp_min(1e-8)` is written to stop the division by zero.
At `bits=4`, `limit = 7`:

```
clamp_min(1e-8) / 7      = 1.428571e-09
fp16 smallest subnormal  = 5.960464e-08          <-- 1.43e-9 is BELOW this
.to(torch.float16)       -> 0.0                  (MEASURED, flush-to-zero)
source / scale           = 0.0 / 0.0 = NaN       (MEASURED)
```

**The guard and the cast are adjacent lines, and the cast undoes the guard.** Verified end to
end: `render_quantized(..., exact_path=True)` returns `nan=False` at k=0 and `nan=True` at k=2.

## What that did to my measurement — the correction, at source

I built `tools/film_row_ablation_curve.py` to price the lever by dropping the k lowest-norm rows
per FiLM tensor and scoring d_seg. The curve came back:

| k | EF3000 flips | FRD077 flips |
|---:|---:|---:|
| 0 | 31,471 | 31,628 |
| 2 … 96 | **59,551,382** (flat) | **59,551,382** (flat) |

59,551,382 / 117,964,800 = **50.48%**, which is exactly "predict Undrivable everywhere"
(Undrivable = 49.5% of pixels). Identical across every k AND across two independently-trained
checkpoints — because a NaN frame gives SegNet the same degenerate argmax regardless of what
produced it. **A constant wearing a measurement's clothes.** It would have read as a clean,
reproducible, flat ablation curve.

What made it catchable: I zeroed the **smallest-norm** rows (min 0.0037 vs 0.48 at the top).
Those are near-no-ops by construction. A total collapse there is not a property of the model, so
the number had to be the instrument. That is the only reason I looked.

**Two prior errors of mine on this lever, both corrected here:**

1. **Row norms were the wrong instrument.** I first measured FiLM row norms to test whether the
   lever created droppable rows (min 0.0037 both arms; 306 vs 291 rows under 1e-2; median
   0.0099 → 0.0097) and read "barely moved." But `_row_dropout` uses **inverted** dropout —
   `scale = 1/(1-p)` keeps the expectation unchanged, *by explicit comment*. Inverted dropout buys
   ROBUSTNESS TO REMOVAL, not sparsity of any row. Row-norm sparsity is a quantity the mechanism
   never claimed to move.
2. **The ablation was the right *question* with the wrong *referent*.** "Drop a row" in the
   receiver's world means *don't ship it, reconstruct zero at decode*. In my harness it meant
   *zero it before the deployed quantiser sees it* — which is the NaN path. Same words, different
   object. The payoff test needs the receiver's own reconstruction, not a pre-quantiser zero.

## Who else this blocks — ⚠ NOBODY. Corrected at source.

**The original claim here was: this NaN blocks the FiLM-row-sparsity rate family, because `mz2`'s
6 retained candidates (−130..−2,051 B) were "bytes measured, distortion never measured" and sit on
this path. Both halves are REFUTED.**

**1. The receiver never touches the defect.** Read at source
(`experiments/ddm_mp2_semantic_receiver.py::_decode_row_prune`, lines 188-211): the SM3R path
quantizes **only the kept rows** into a compact `(expected_keep, columns)` tensor, then scatters
them into a `torch.zeros((rows, columns))` buffer. A dropped row is materialized as zeros
**after** dequantization and is never handed to a quantiser at all. There is no zero-`amax` row on
the shipping path, so there is no zero scale and no NaN.

The two operations wear the same three words:

| path | order | zero-`amax` row reaches `fake_quantize`? |
|---|---|---|
| receiver `_decode_row_prune` | prune → quantize kept rows → scatter into zeros | **No** |
| my `film_row_ablation_curve.py` | zero the fp32 row → quantize the whole tensor | **Yes** → NaN |

**2. The family was measured — the day before my charter, and again the same day as this memo.**
`#1058` closed the FiLM-row sparsity family at **FAMILY scope on 2026-08-16** on three measured
n600 rows. `ddm_sf1` (2026-08-17, same directory, "film" in the filename) then mapped **all 576
FiLM rows** in 32-row groups through the shipped renderer at batch=1 **with a pose channel**, and
re-priced `mz2`'s candidates against authority-tracking GT: the best nets **+0.062227 S — 6.5× the
whole remaining gap, the wrong way**. It also measured that the −2,874 B sum is **undecodable**
(the mixed-q3/q4 credit rides `SD1M`, the row credit rides `SM3R`, `unpack_variant_semantic_or_none`
dispatches on ONE magic, no combined format exists); honest ceiling −2,051 B.

So the family did not die of my NaN. It died on **pose price**, measured twice, through the real
receiver, before and independent of this memo.

**What the defect's blast radius actually is:** the TRAINER, whenever a training-time actuator
drives a row to exactly zero and the next step quantizes it — e.g. `--fixed-zero-mask` pinning a
full row, or any future ablation harness that zeroes before `fake_quantize`. That is a narrow,
real, worth-fixing class. It is not a rate-family blocker.

**verdict_scope: INSTANCE** — `film_row_ablation_curve.py`'s pre-quantiser zeroing at `bits=4` on
`train_semantic_quantized.py`. The FiLM-row-sparsity FAMILY verdict is **not mine**; it belongs to
`#1058` (FAMILY, 08-16) and `ddm_sf1` (08-17, partition-wide, pose-priced).

**Genus:** `measured_object_vs_named_object_20260816`. "Drop a row" named the receiver's
reconstruct-as-zero on one side of the quantiser and my zero-then-quantize on the other. I measured
the second and wrote a conclusion about the first.

## Two-landing

* **Guard (landed here):** `film_row_ablation_curve.py` now renders a probe frame per (arm, k) and
  **REFUSES** on NaN rather than reporting a degenerate d_seg. The failure that cost this run is
  now loud on the first eval instead of silent across all 24.
* **Defect (owed):** the fp16-cast-after-clamp in `fake_quantize`. The clamp bound must survive its
  own cast — `1e-8` cannot, at any `limit ≥ 1`, since `1e-8` is itself below fp16 subnormal after
  division. Fix belongs on the trainer's quantiser with a regression pinning `amax == 0 → finite`.
  **NOT applied here:** this function is on the live training path for the frontier lineage;
  changing it is a training-semantics change that needs its own A/B, not a drive-by.

## What is measured and what is not

* **MEASURED:** the lever is seg-neutral (0.18σ) and byte-neutral. The NaN, its exact arithmetic,
  and its invariance across k and across checkpoints.
* **NOT MEASURED:** whether the lever bought droppability **on its own trained checkpoint**. ⚠ But
  the instrument I called for **already exists**: `ddm_sf1` built exactly the harness this bullet
  demanded — receiver-semantics drop (`_decode_row_prune`), shipped renderer at batch=1, proven
  bit-identical at zero perturbation — and ran it over all 576 rows. It ran on the `hv1` base, not
  on `FRD077`, so the lever-specific question is open; but it is open for want of a *run*, not for
  want of a *harness*, and the family that run would feed is closed (below).
* ⚠ **The pose channel is BUILT, and it is what killed the family.** I wrote that
  `FILM_ROW_FAMILY` **is** `POSE_CRITICAL_TENSORS` (ns1: ~94× spread vs `frame_embed`) and that
  "any future row-drop row owes a pose channel." `sf1` paid that debt and the answer is the
  verdict: the best candidate prices **+0.062227 S**, pose-dominated. The seg-only harness was the
  right worry about the wrong outcome — I expected a missing number; the number exists and it is
  disqualifying.

## NEXT

1. **Cure the zero-scale path** with its regression, then re-run this curve — the tool is built and
   its k=0 positive control reproduces `EF3000`'s endpoint to all 17 digits.
2. **Levers 2–12 of the #1092 drain** are unaffected: `--carrier-rank-penalty` + `--carrier-tensors`
   (ra2 head, #1079) → `--distill-weight` + `--distill-max-seg` (wd3, #1069). None of them zero a
   quantised row.
3. `--fixed-zero-mask` (lever 12) is **not** the ablation actuator — it pins *already-zero* weights
   to stay zero during training (`train_semantic_quantized_resumable.py:1031-1035`), a sparsity
   *preservation* lever. Its consumer is the init mask, not a drop.
