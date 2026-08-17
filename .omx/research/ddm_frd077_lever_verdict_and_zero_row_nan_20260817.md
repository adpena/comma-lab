# The FiLM-row lever is seg-neutral — and zeroing a FiLM row makes the deployed quantiser emit NaN

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
   defect.** Zeroing *any* FiLM row makes the deployed lifted quantiser emit **NaN** frames.

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

## Who else this blocks

`mz2` (5c073e915) retains **6 FiLM-row sparsity candidates at −130..−2,051 B**, explicitly
**"RETAINED unscored"** — bytes measured, distortion never measured. Those candidates sit on this
exact path. Anything that realises them by zeroing rows ahead of the deployed quantiser produces
NaN, so the byte win was never scoreable as implemented. That is not an mz2 error — it measured
what it said it measured — but the family cannot be priced until the zero-row path is cured or
routed around.

**verdict_scope: FORMULATION** for the row-sparsity family *as realised by pre-quantiser zeroing*
at `bits=4` on this trainer. Not a verdict on FiLM-row sparsity as a rate lever: an all-zero row
that the receiver simply omits and reconstructs never reaches this divide.

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
* **NOT MEASURED:** whether the lever bought droppability. That test is unrun — it needs a harness
  that reconstructs a dropped row the way the receiver would, and it will not be an argument about
  norms.
* **NOT MEASURED:** the pose cost of dropping FiLM rows. `FILM_ROW_FAMILY` **is**
  `POSE_CRITICAL_TENSORS` (ns1: ~94× sensitivity spread vs `frame_embed`), and this harness is
  seg-only. Any future row-drop row owes a pose channel.

## NEXT

1. **Cure the zero-scale path** with its regression, then re-run this curve — the tool is built and
   its k=0 positive control reproduces `EF3000`'s endpoint to all 17 digits.
2. **Levers 2–12 of the #1092 drain** are unaffected: `--carrier-rank-penalty` + `--carrier-tensors`
   (ra2 head, #1079) → `--distill-weight` + `--distill-max-seg` (wd3, #1069). None of them zero a
   quantised row.
3. `--fixed-zero-mask` (lever 12) is **not** the ablation actuator — it pins *already-zero* weights
   to stay zero during training (`train_semantic_quantized_resumable.py:1031-1035`), a sparsity
   *preservation* lever. Its consumer is the init mask, not a drop.
