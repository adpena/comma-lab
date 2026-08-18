# The section-coding rate axis is measured shut — four mechanisms, −5 B available

`verdict_scope`: **INSTANCE** — the sz1 pointer archive (sha
`debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a`, 179,930 B), measured
byte-exact through the real coders. Receipts + all payloads:
`/Volumes/APDataStore/pact/ddm_bp1/` (P0 ALWAYS KEEP THE PAYLOAD — the three shipped streams
and their decoded bodies were persisted before any measurement ran).

Successor to `ddm_xs1_cross_section_joint_coding_20260818.md`, which closed joint coding and
left two sections **admittedly unprobed** because its instrument was uncalibrated. This memo
calibrates the instrument, probes them, and closes the axis.

## 1. The instrument, finally calibrated

xs1 could not reproduce the shipped `hpac` stream (off by 40 B) and correctly refused to draw a
conclusion. The cause is now read at source: `runtime/residual_archive.py:161` unpacks
`RX1_MODEL_HEADER` and the live archive carries **`codec=2` = `RX1_CODEC_BROTLI`** — the
`lzma.FORMAT_XZ` branch at `:179` is dead code on this lineage. All three model streams are
Brotli.

Sweeping `mode × lgwin × lgblock` at quality 11 **reproduces the shipped `semantic` stream to
the byte** (34,243) at `mode=GENERIC, lgwin=16, q11`. That is the calibration xs1 lacked.

| section | shipped | our best brotli | gap |
|---|---:|---:|---:|
| `semantic` | 34,243 | 34,243 | **0 — exact** |
| `carrier` | 22,161 | 22,160 | −1 |
| `hpac` | 13,515 | 13,554 | **+39** |

The `hpac` +39 is not a parameter we failed to find: the brotli **CLI** also lands 13,555, and
lgblock sweeps do not move it. The shipped stream was produced by an encoder that found a
better parse than python-brotli 1.2.0 can. **Consequence: on `hpac` any candidate must beat
−39 on our instrument merely to tie the bytes already shipped.** Stating that bar is what makes
the probe honest.

## 2. Four mechanisms, measured

| mechanism | result | closes? |
|---|---|---|
| brotli parameter sweep (3 modes × 15 lgwin × 10 lgblock) | shipped is at-or-beyond our encoder on all 3 | **yes** |
| cross-section joint coding (xs1, all 24 orders) | +205 B; every ordered conditional pair negative | **yes** |
| global byte-plane deinterleave, k ∈ {2,3,4,8} | +31…+509 B — loses on all 3 sections at every k | **yes** |
| **region** byte-plane (the sz1 mechanism, generalized) | hpac **+7** · carrier **−5** · semantic **0** | **yes** |

Total honestly available on this axis: **−5 B** on a 179,930 B archive = **3.3e-6 S**, below any
admission bar and not worth a receiver change. Semantic returns exactly 0 because sz1 already
extracted its win; the greedy search rediscovers nothing further, which is a clean consistency
check on both measurements.

## 3. The finding that nearly became a fake win

The first composed hpac result read **−215 B** — a clear win over the −39 bar. It was an
artifact of a **lossy** transform: the compositor read each region from the *original* buffer
while writing into a copy, so overlapping regions clobbered each other and destroyed
information. It compressed better because it was throwing bytes away.

The cure was ordering, not cleverness: **prove invertibility before believing the number.** A
verified forward/inverse pair (200 random op-sequences, overlaps included, all round-tripping
byte-identical) plus a greedy search constrained to that pair converges to **−32**, i.e. **+7
worse than shipped**. Same grid, same instrument, honest transform — the win evaporates.

Recorded as a live instance of the NO-FAKE class: *a transform that improves the metric by
being lossy is not a rate win.* Any future byte-permutation proposal on this vehicle must ship
its inverse and its round-trip proof in the same measurement.

## 4. What this routes

The lossless **coding** of the archive's sections is finished. Concordant with #996 (coder axis
closed vs each section's own memoryless bound) and #1060 (all 38/38 semantic tensors
receiver-required, every exact recoding +340 B). Three independent closures now agree.

Rate progress must come from **representation** — fewer or smaller symbols — not from coding
them better. Per `ddm_asym1`, rate is the linear, additive, honest axis and needs 11,584 B
(6.4% of the archive) to close the gap alone; none of it will come from the coder.

`token_stream` (109,801 B, 61.0% of the archive) remains unprobed for byte-planing. It is an
rc64 arithmetic-coded bitstream and expected null by construction — stated here as a
**derivation, not a measurement**, exactly as xs1 stated it.
