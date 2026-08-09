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

---

# STATUS UPDATE 2026-08-09 (APPEND-ONLY; the rows above are preserved as written)

## S1. The ep16 row is SUPERSEDED — the run reached ep32 and was WINNING

The "ours (GT labels), ep16/60 → 137,945" row above was quoted from a live run. That run
**died at epoch 32 of 60**, and ep32 is on the far side of a phase transition the ep16 row
cannot see:

| ep | phase | joint B | vs PR130 137,159 |
|---:|---|---:|---:|
| 16 | continuous | 137,945 | +786 (the row above) |
| 30 | continuous | 137,620 | +461 |
| **32** | **discrete_qat** | **135,828** | **−1,331** (ΔS −0.0008863) |

`qat_start=31` for `epochs=60, qat_fraction=0.5`. The 30 continuous epochs bought −7,264 B
total; the FIRST QAT epoch bought **−1,792 B alone**. The run was killed with ~28 QAT epochs
unrun, at the steepest part of its descent. Full trajectory + cause analysis:
`.omx/research/ddm_hb3_20260808/ENDPOINT_hb3_died_at_ep32_and_was_winning.md`.

**Roadmap step 1 (MATCH) is therefore NOT closed — it is partially BEATEN and was abandoned
mid-descent.** Resume fired 2026-08-09 (`resume_qat/`, `--epochs 28 --qat-fraction 1.0` →
`qat_start=1`), positive control PASSED bit-identically (epoch-0 eval reproduced ep32's
bpp/joint/top1/bit-histogram exactly).

## S2. The "UNSAFE until re-derived on DALI" flag on this comparison is DISCHARGED for RATE

The ADDENDUM flagged the 135,732-vs-137,159 comparison UNSAFE because our HPAC trained on
AV-decoded labels. `ADDENDUM2_platform_leg_and_rate_rederivation.md` measured the same generic
coder on all three label provenances: **410,392 / 410,548 / 410,584 B — a 192 B, 0.047% spread.**
Scaled to a 137 KB stream that is ~65 B of provenance uncertainty against a −1,331 B lead: **20×.**

**The decoder confound is a DISTORTION confound, not a RATE confound.** 20,671 differing labels
(61.25% of PR130's seg term) move coded size by essentially nothing. The rate half of every
ours-vs-theirs byte comparison is safe; the distortion half is not.

Also closed there: the **platform leg** (listed above as STILL UNMEASURED) — Modal-AV vs
LOCAL-macOS-AV = **2 labels in 117,964,800 (1.695e-08)**. Our local decode is faithful; the
entire delta is decoder, never platform.

## S3. Share-arithmetic correction to the anatomy table

The per-object column above sums to **200,565 B** but the archive is **191,052 B**. The model
sub-blobs are listed RAW (83,485) while they ship jointly LZMA'd to **73,968**. Those are
**CONTENT shares, not ARCHIVE shares**. Archive shares are tokens **61.26%** / models-compressed
**38.74%**. This matters for waterfilling: a lever on `semantic_blob` buys COMPRESSED bytes at
the joint-stream margin, not raw bytes.

## S4. Two PROJECTED cells are now MEASURED

- **Our coder lineage on the token axis → MEASURED-NEGATIVE.** SMEVR / CAE-INTER / KT-backoff /
  brotli / lzma / rANS all lose to HPAC by **2.2–3.6×** on the dense partition. Their edge is the
  LEARNED PRIOR, not the packing. Do not spend more effort racing our coders there.
  (`CODER_LINEAGE_VS_HPAC.md`)
- **Lossless model recode → MEASURED −903 B** (split-stream + per-section brotli q11; −234 B in a
  brotli-free variant). Real, free, and small: 2.7% of the −33,254 B needed for sub-0.15 by rate
  alone. Bank it; do not narrate it as the path. (`RATE_AXIS_LOSSLESS_RACE.md`)
- Also measured there: the **pose carrier is incompressible** (23,054 → 23,058 under every coder)
  and the **token stream is at HPAC's model entropy** (+5 B under brotli). Both foreclose coder
  families rather than opening them — which is worth more than a small win.
