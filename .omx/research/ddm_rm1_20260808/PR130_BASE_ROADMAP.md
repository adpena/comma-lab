# THE ROADMAP — PR130 as the BASE, our levers on top

Operator directive 2026-08-08: *"take fresh eyes to pr130 and then optimizing further
with the stuff we have and they don't — that is the roadmap."* Plus: *"KT is dead"*
and *"don't be too obsessed with the rate ... all of that should match pr130 or better."*

Frame correction this encodes: our own lineage (tq1c / IX2TOK01 / PP1-KT) is NOT the
base. PR130 is the base. We start AT 0.1721417 and go down.

## PR130 measured triple — THEIR OWN official CUDA report

`repro_repo/evidence/source_archive_official_ada_report.txt` (600 samples, device cuda):

| axis | measured | S contribution | share of their score |
|---|---:|---:|---:|
| d_seg  | 0.00028609 | 0.028609 | 16.6% |
| d_pose | 0.00001967 | 0.014025 |  8.1% |
| rate (191,052 B) | — | 0.127214 | 73.9% |
| **S** | | **0.1721417** | |

(The quoted report row is the `landslide` variant at 194,380 B / S 0.17; the
`cpr1` release at 191,052 B is the 0.1721417 bar.)

**Both distortion axes are already crushed on their vehicle.** Three quarters of
their residual score is RATE. Seg/pose are therefore a PRECONDITION, not the lever:
no byte cut is real unless d_seg <= 2.86e-4 AND d_pose <= 1.97e-5 hold through the
real `upstream/evaluate.py` path.

## PR130 archive anatomy — MEASURED by parsing the shipped bytes

`releases/cpr1/archive.zip`, one ZIP_STORED member `p`:

```
p = u32(len(models)) || LZMA-XZ( semantic_blob || carrier_blob || hpac_blob ) || tokens
```

| object | bytes | share | what it is | OUR lever surface |
|---|---:|---:|---|---|
| tokens (raw HPAC-coded) | 116,980 | 61.3% | the semantic label stream | #869 adaptive per-cell waterfill · CR1 edge-conditioned contexts · SMEVR race · granularity re-race |
| semantic_blob | 40,252 | 21.1% | renderer weights | #311 TropNNC · #336 sensitivity bit-alloc · #157 KKT reverse-waterfill · low-rank |
| carrier_blob | 23,054 | 12.1% | pose basis + coeffs | #140 low-rank pose codec (2.7x) · od9 cheap carriage |
| hpac_blob | 20,179 | 10.6% | the AR prior model | already self-compressed by their own machinery |
| ZIP overhead | 100 | 0.05% | | minimal already |

models decompress 83,493 -> 73,968 xz (ratio 0.886). Their LZMA filters are tuned
(`lc=0, lp=1, pb=0`, dict 1<<16, BT4, nice_len 273) for 16-bit-aligned data. Tokens
are appended RAW, deliberately outside the LZMA — already entropy-coded, so
re-compressing them would lose. That is a correct design choice, not an oversight.

## Reproduction check — GT labels under their HPAC prior

`ddm_hb3` (LIVE, MPS, GT `lstars`, epoch 16 of 60, still descending):

| | tokens | prior model | total |
|---|---:|---:|---:|
| PR130 (their trained semantic field) | 116,980 | 20,179 | 137,159 |
| ours (GT labels), ep16/60 | 116,560 | 21,385 | 137,945 |

Within 0.6% at epoch 16. The label stream costs ~137 KB either way; that is this
vehicle's design floor.

**The GT advantage is on the DISTORTION axis, not the rate axis.** Their tokens are
a trained+quantized semantic field, so their 2.86e-4 d_seg contains label error PLUS
realization error. GT labels carry zero label error by construction — only
realization error remains. Same bytes, strictly less error to explain.

## The roadmap

1. **MATCH.** Reproduce the PR130 base triple end to end. Everything needed is handed
   to us: `repro_repo/scripts/e2e.py` (49 resumable stages) + `code/`.
2. **BEAT, per object.** Apply our levers to the four objects above, each gated on
   the precondition (d_seg <= 2.86e-4, d_pose <= 1.97e-5 through real evaluate).
3. **VELOCITY is the asset they cannot copy.** Their chain needs CUDA + DALI. Our
   MLX/Metal stack (custom kernels, fused-R, grouped-backward) lets us iterate the
   race locally. Wall-clock is not scored, but it multiplies every lever by letting
   us run more races per day.

## What is NOT the roadmap

- tq1c / IX2TOK01 latent-token polish (family yield ~-5.2e-4 S per 3h arm)
- PP1-KT context-arithmetic as a coder baseline — dead, superseded by HPAC
- any byte claim that has not held both distortion axes through real evaluate

## Honesty labels

MEASURED: every byte figure above (parsed from the shipped archive) and the PR130
triple (their own 600-sample CUDA report). MEASURED-IN-PROGRESS: the GT ep16 row
(run live, not final). PROJECTED: all "OUR lever surface" values — none has been
measured on a PR130-class object yet. That measurement IS the roadmap.
