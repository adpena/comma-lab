# ddm_rr1 — the rate rung, closed on measurement; and one new free supplier worth 10.7% of it

**Date:** 2026-08-17
**Base:** hv1 ep0634, `S = 0.15959729295498598` @ 182,759 B `[contest-CUDA T4, n600]`,
archive sha256 `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`
(receipt `experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json`).
**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]`. `score_claim: false`,
`promotable: false`. No Modal, no dispatch, no new archive, no exact eval. **Pointer UNMOVED.**
**Store:** `/Volumes/APDataStore/pact/ddm_rr1/` — 52 files, 1,849,426 B, manifest
`RETENTION_MANIFEST.json`.

---

## ANSWER

**This archive cannot supply −14,414 B on the rate axis. Total measured, composable, pure-rate
supply is now 1,549 B — 10.75% of the bar — and 89.25% of the bar has no measured supplier at
all.** The one thing that can supply the rest is not a rate lever: it is `ddm_rc4`'s token drop,
already measured at −1.19754e-2 S on rate and −3.243e-3 S on rate+seg (125% of the bar in bytes),
**gated entirely on whether the frame-0 Schur compensator can absorb its +0.174319 S pose leg.**

**The new number this arm produced: 1,549 B of the token stream are recoverable at ZERO archive
cost and ZERO distortion, by a decode-time model that ships in `inflate.py` for free under rule
118.** Measured at n600 on the real field, with the shipping decoder's own probability tables
reproduced byte-identically. That is `ΔS = −1.0314e-3`, **10.75% of the gap**, clearing the
standing −3.5e-6 admission bar by **294.7×**, and it is **4.35× the previous free-side ceiling**
(`ddm_hm1`'s 356.1 B) and **119× the largest rule-118 arbitrage ever realised in this repo**
(`ddm_m6`'s 13 B).

The mechanism is proven, not asserted: **HPAC sees exactly one frame of history, and frames
t−2…t−4 are sitting in the decoder's own output buffer for free.** Verified at source
(`prepare_frame_context(model, idx, previous_raw)` takes one frame; `previous_raw` is overwritten
each frame; the FiLM carries only the frame index). The negative control confirms it: adding a
context term HPAC *already* sees (agreement with t−1) changes the result by **−18.13 B**, while
adding run length over t−1…t−4 adds **+1,108.60 B**.

**It is NOT byte-closed.** No archive was built. 1,549 B is a MEASURED code length converted to a
PROJECTED archive byte count through a MEASURED constant coder overhead. Realising it needs a
receiver change and a matching encoder. That build is the fire-order below — and it is a build
order, not a T4 order, because there is no candidate archive to dispatch.

---

## 1. The verified rung table — every row re-read at source, not inherited

My charter carried four stale premises. All four are corrected here.

| # | charter said | source says | correction |
|---|---|---|---|
| 1 | "rung 4 … LEAST explored" | `ddm_rc4` (2026-08-16) priced it to three legs at n600 | **rung 4 is fully adjudicated and REFUSED** |
| 2 | bar = −14,415 B | my own 40-digit decimal on the live pointer | **14,414 B strict** (14,413 leaves S=0.150000268); ceiling ≤ **168,345 B** |
| 3 | "coder refit … least explored" | `ddm_dc1` measured rc64 at its floor on hv1 | **coder axis CLOSED, ceiling ≤ 7.8 B** |
| 4 | "-14,415 B … is your bar" | continuous form is 14,413.402 B | strict/continuous distinction matters at 1 B |

Verified independently by me on the archive bytes: `archive.zip` = **182,759 B**; single ZIP
member `p` = **182,659 B**, `compress_type 0` — **STORED, not deflated**, so a byte off any
section is a byte off `archive.zip` 1:1; framing = **100 B**. `RX1M` header parsed directly:
`hpac 13,515 | semantic 34,763 | carrier 22,161 | header 14`, remainder **112,206 B**
(RCF1 96 + token 112,110). Rate term recomputed `25·182759/37545489 = 0.12169171641365491`.

| rung | object | verdict | scope | vehicle | measured |
|---|---|---|---|---|---|
| **1. mixed precision** | semantic q3/q4 + FiLM row sparsity | **REFUSED** | INSTANCE ×3 scored candidates; family closure UPHELD by `sf1` on the corrected GT axis | hv1 | −823 B buys **+0.066619 S**; best of three nets **+0.062227 S = 6.5× the gap the wrong way** |
| **2. carrier rank / refit** | 22,161 B carrier | **REFUSED** | **FAMILY** (6 treatments: rank truncation, α=0, keep-set refit, pose-metric subspace, trust-region refit, sphere-wide) | hv1 | rank-4 returns 14,709 B = 102.1% of the bar; the score functional misses by **1,498×–3,139×**; best realised (ra3) still **35.5×** |
| **3. nested-width distillation** | 34,763 B semantic | **REFUSED** | **FAMILY** (fresh-init scorer-free birth @65 ep, this teacher); warm-lineage/longer-budget explicitly untested | hv1 + students | ~16 KB projected byte credit against Δd_seg **~6× over its own bar** |
| **4. token drop** | 112,110 B token stream | **REFUSED uncompensated** | **FORMULATION** | hv1 | 17,985 B; rate −1.19754e-2, seg +8.7325e-3, **pose +0.174319 = 517.5× over budget** |
| **4b. coder refit** | all 4 sections | **CLOSED** | measured on hv1 | hv1 | rc64 is **+0.42 B** over HPAC's cross-entropy on the shipped stream; any coder swap ceiling **≤ 7.8 B** |
| **4c. free correction table** | RCF1 axis | **CLOSED** | 9 rungs, 5 context families | hv1 | free-table ceiling **356.1 B**; derivative collapses **210× in one step** past the shipped 25 cells |
| **4d. free DECODE-TIME model** | 112,110 B token stream | **OPEN — supplier found** | this arm | hv1 | **1,549 B at zero bytes, zero distortion** |

Rungs 1–3 ran on hv1 (rung 3's students are a separate object, as `wd3` states). `ddm_mz2`'s
byte census ran on **e480b (183,502 B)** against a **15,153 B** bar — a different archive and a
different bar; `ddm_rfo2`'s "−15,157 B rung" is likewise the e480b figure. Neither transfers.

---

## 2. What I measured — the free decode-time model

### 2.1 The question, and why it was still open

Rule 118 makes `inflate.py` an unsized free interpreter: a GENERIC algorithm costs zero archive
bytes; only VIDEO-DERIVED content is counted. Two arms had already pushed on this and both left
the same door open:

* `ddm_dc1` measured a table-free adaptive context model used **INSTEAD of** HPAC: 177,109 B (KT)
  / 144,167 B (oracle) against a shipped 112,110 B. Standalone replacement is dead by +32,057 B
  at best. dc1 wrote plainly that a mixing/augmentation coder **"was not built."**
* `ddm_hm1` measured **COUNTED** correction tables and, generously, the ceiling with the table
  given away free: **356.1 B**. hm1 wrote that a correction on richer context than its summaries
  **is not bounded by its rows.**

Nobody had measured an **AUGMENTATION**: keep every shipped HPAC probability exactly, and correct
it with a table that is never transmitted because both sides rebuild it from symbols already
decoded.

### 2.2 The hypothesis has a source-level warrant

`decode_tokens` sets `previous_raw = current.clone()` each frame;
`prepare_frame_context(model, idx, previous_raw)` is the only history input; `conv_past` runs over
a one-hot of that single frame; `frame_embed` carries the frame **index**, not history. **HPAC
receives exactly one frame of past.** Frames t−2, t−3, … are in the decoder's own output buffer,
cost nothing, and are provably outside the model's input. In a driving segmentation, per-pixel
temporal persistence beyond one frame should carry information. It does.

### 2.3 Legality (rule 118), stated precisely

The correction is a per-context log2-odds shift estimated from **strictly past frames** by a fixed
generic rule: fixed bin edges, fixed KT smoothing α=0.5, fixed cold-context floor (32
observations → shift exactly 0), fixed clamp, fixed stable tie-break. Encoder and decoder run the
identical rule over the identical already-decoded symbols and obtain identical tables. **Zero
bytes are transmitted. No video-derived table enters `inflate.py`** — only generic constants,
which is the same status the existing decode logic already has. This is not the hide-data-in-code
fake: there is nothing to hide, because nothing is stored.

The decoded token field is **bit-identical** under every rung — the probability model enters only
the arithmetic coder. So `d_seg` and `d_pose` do not move at all. **Pure rate, no scorer needed.**

### 2.4 Positive control — fail-closed, four ways

| control | result |
|---|---|
| `corrected_quantized_logit_sha256` | `562ac652b372faa020d0fc5e2ed9b7b61625169e0f5c2041d4fe99196055b8c7` — **MATCH** |
| `corrected_cdf_input_sha256` | `dd48843b021763e78524caf3dcd01e944045e7bd0ffd93b451dec83548f083b7` — **MATCH** |
| HPAC cross-entropy n600 | **112,109.57757858819 B** vs the 112,109.57757858852 B measured independently by `dc1`, `hm1` and `rc4` — 2.9e-15 relative |
| stage 2 reproduces stage 1 | `bits_per_frame_G1_hit_rl4_n600.npy` sha `7120c292ad7eb484` is **byte-identical** to stage 1's `bits_per_frame_F4_…_n600.npy`, per frame — and the two stages' base ledgers share sha `e5b495f243f6dbcd` |

Every probability priced below is bit-identical to the one the shipping RC64 decoder consumed.
Both scripts refuse to emit a verdict if any control fails.

### 2.5 Stage 1 — the ladder, and the negative control that identifies the mechanism

n600, 117,964,800 positions, 66.0 s. Shipped cross-entropy 112,109.578 B.

| family | contexts | warm | code B | saved B | ΔS | % of gap |
|---|---:|---:|---:|---:|---:|---:|
| F1 (cls, ubin) — recalibration only | 320 | 224 | 112,046.07 | **63.51** | −4.229e-05 | 0.44 |
| F2 (+ t−1 agreement) | 640 | 442 | 112,064.20 | **45.38** | −3.022e-05 | 0.31 |
| F3 (+ run length t−1…t−4) | 2,560 | 1,597 | 110,955.60 | **1,153.98** | −7.684e-04 | 8.01 |
| F4 (+ boundary bucket) | 12,800 | 5,643 | 110,660.78 | **1,448.79** | −9.647e-04 | 10.05 |
| F5 (cls, ubin, bnd) — no temporal | 1,600 | 1,097 | 112,003.79 | **105.78** | −7.044e-05 | 0.73 |

Three readings, and they agree:

1. **HPAC is already well calibrated.** Pure recalibration (F1) returns **63.51 B** — consistent
   with hm1's ladder collapsing past the shipped table. There is no miscalibration to harvest.
2. **The negative control fires exactly as it should.** F2 adds agreement with frame t−1 — which
   HPAC already receives as an input. It returns **−18.13 B relative to F1**: information the
   model already has costs context dilution and returns nothing. This is what rules out
   "more contexts always help."
3. **The gain is the new information.** F3−F2 = **+1,108.60 B**; F4−F5 = **+1,343.01 B**. Two
   independent reads of the same term — run length over frames t−1…t−4 — agree in magnitude.

### 2.6 Stage 2 — optimal form, and the branch split nobody had

Stage 1 declared three scope reductions. Stage 2 removes two of them and measures the third.
n600, 77.2 s.

| rung | code B | saved B | ΔS | % of gap | admission × |
|---|---:|---:|---:|---:|---:|
| G1 hit, run length 4 (= stage 1 F4) | 110,660.78 | 1,448.79 | −9.647e-04 | 10.05 | 275.6 |
| G2 run length 8 | 110,585.28 | 1,524.29 | −1.015e-03 | 10.58 | 290.0 |
| **G3 + prev2 agreement — BEST** | **110,559.71** | **1,549.87** | **−1.032e-03** | **10.75** | **294.9** |
| G4 = G3 + adaptive miss branch (M1) | 110,590.32 | 1,519.26 | −1.012e-03 | 10.54 | 289.0 |
| G5 = G3 + richer miss branch (M2) | 110,610.54 | 1,499.04 | −9.981e-04 | 10.40 | 285.2 |
| M2 miss branch alone | 112,160.41 | **−50.84** | +3.385e-05 | −0.35 | — |

Attribution: run length 4→8 **+75.50 B**; explicit `prev2 == argmax` **+25.58 B**; adaptive miss
branch **−30.61 B (M1) / −50.84 B (M2)**.

**The miss branch is a measured negative, and the reason is a decomposition nobody had made.**
`ddm_rc4` measured that **70.011%** of all code bits sit in the 223,694 positions where HPAC's
argmax is wrong. That is true, and it has been read as "70% of the stream is the miss branch."
It is not. I split the shipped code length exactly:

| branch | bytes | share |
|---|---:|---:|
| hit/miss binary decision | **110,862.39** | **98.89%** |
| 4-ary refinement — *which* non-argmax class | **1,247.19** | **1.11%** |

Almost the whole cost at a miss is paying `−log2(1−p_max)` for being surprised, not identifying
the class. There are only 1,247 B in the refinement branch, so an adaptive correction there
cannot repay its own learning cost. Stage 1's choice to model only the hit/miss event was the
right target by 89×. `verdict_scope: INSTANCE` — these two miss-context forms, this vehicle. The
branch itself is 1.11% of the stream and is not worth another arm.

### 2.7 The byte arithmetic, and exactly what is measured vs projected

* **MEASURED** — code length under G3: **110,559.71 B**; shipped HPAC cross-entropy
  **112,109.578 B**; shipped token stream **112,110 B**, so the realised coder overhead is
  **+0.42242 B** (`dc1` independently measured this as a constant flush, +7.74 B at an 8-frame
  prefix vs +7.80 B at n600 on the decoder's bit position — a constant, not a rate).
* **PROJECTED** — new token stream `110,559.71 + 0.42242 = 110,560.13 → 110,561 B`;
  saving **1,549 B**; archive **181,210 B**; `ΔS = −0.00103142`; **S = 0.15856588**.
* The projection carries sub-byte uncertainty **only** because the ZIP member is STORED and the
  coder overhead is a measured constant. It is still a projection: **no archive was built and no
  `archive.zip` was stat'd.** Label it PROJECTED wherever it is quoted.

### 2.8 Decode cost

The corrector is a table gather plus a sigmoid over 196,608 positions per frame. Stage 2
evaluated **five** models plus the full base reconstruction over all 600 frames in **77.2 s** of
numpy on this machine; stage 1 evaluated five in 66.0 s. A single corrector is therefore well
under ~15 s over the whole field — DERIVED from my own runtime, not from a receiver. Against
hv1's own measured `[contest-CUDA T4]` inflate of **364.111 s** inside the 1,800 s budget
(**4.944× headroom**, receipt `MODAL_REMOTE_RESULT.json`), that is a few percent. **A
receiver-integrated decode-time row is owed before this ships** — my figure is an instrument
timing, not a decode timing.

*(Correcting an inherited number: the 831.5 s / 2.17× headroom figure belongs to **MC36**
(186,269 B, `[contest-CPU]`), not to hv1. hv1 has no contest-CPU decode row at all.)*

---

## 3. What this archive cannot supply, and what can

Pure-rate supply on hv1, every row measured:

| source | bytes | % of 14,414 B bar | status |
|---|---:|---:|---|
| **free decode-time model (this arm)** | **1,549** | **10.75%** | MEASURED, zero distortion, needs a build |
| `ra2` CPR1 inner coder + `ra1` basis_scales gauge | ~278 | 1.93% | raw MEASURED / realised PROVISIONAL (117 B uncertainty), unowned, unfired |
| any coder swap | ≤ 7.8 | 0.05% | ceiling, CLOSED |
| free correction table | (356.1) | (2.47%) | CLOSED — and subsumed by row 1, not additive |
| ZIP framing | 0 | 0% | structural floor, 30+1+46+1+22 |
| **total measured composable** | **~1,827** | **~12.7%** | |
| **residual with no measured supplier** | **~12,587** | **~87.3%** | |

The lossy representation families are all refused above. The only unmeasured multi-KB rate cell
is **HPAC network capacity growth** (`HPAC_CHANNELS`/`HPAC_PATCH`), which is training-blocked and
bounded unfavourably: the often-quoted 3.810 B per counted byte is an **average** over `[0, M0]`
of a convex-decreasing curve and is therefore a hard **upper** bound on the marginal; the only
measured marginal anywhere is **1.15**, at which the whole family is worth ~9.45% of the bar.

**So: no.** The rate axis alone cannot close this gap, and after this arm the honest statement is
that ~87% of the bar has no measured supplier of any kind.

**What can.** `ddm_rc4`'s token drop already buys **17,985 B = 125% of the bar** and is a measured
**−3.243e-3 S** win on rate+seg together. It dies on one leg: **pose, +0.174319 S, 517.5× over
budget.** The single unentered cell in the whole rate programme is `rc4` NEXT #1 / `ws4` §12 —
whether the frame-0 Schur compensator can absorb that pose leg (it must cancel 99.807% and the
re-coded carrier must grow by < 4,873 B). That is a pose-owner cell, not a rate cell, and it is
where the remaining bar lives.

**My result composes with it.** The free model re-codes whatever token field ships; a
Schur-compensated drop changes that field. The two are not additive and the credit must be
re-measured on any new field — a dropped field substitutes the model argmax and is *more*
temporally stable, so the free credit could move either way. Unmeasured; stated as owed.

---

## 4. Fire order — a BUILD order, not a T4 order

**No T4 fire-order is emitted. There is no candidate archive to dispatch, and emitting one would
be a fake.** What exists is a measured code length and the exact build that realises it.

```
TARGET (pre-registered, byte-closed):  archive.zip == 181,210 B  (+/- 2 B)
                                       token stream == 110,561 B
                                       every other section byte-identical to 80d9c8c6...
FALSIFIED IF:                          the built archive exceeds 181,300 B, or the decoded
                                       token field is not bit-identical to sha 9ba2e52b...
```

Build, in order:

1. **Receiver** — add the adaptive corrector to `runtime/residual_archive.decode_production_tokens`,
   immediately after `corrected = base_logits + parts.table.values[feature]`: maintain the G3
   context tables, apply `delta` before `_probability_table`, update from the decoded frame. All
   constants generic and hard-coded. `inflate.py` is unsized; no archive byte changes.
2. **Encoder** — mirror the identical rule when producing the token stream. The two must be one
   function used twice, not two implementations, or they will drift.
3. **Parse-back** — decode the rebuilt archive and assert the token field reproduces sha
   `9ba2e52b…` bit-identically. This is the whole distortion proof: if the field is identical,
   `d_seg` and `d_pose` cannot have moved.
4. **Byte-close** — `stat` the rebuilt `archive.zip` against the pre-registered target above.
5. **Only then** does a T4 row make sense, and only bundled with other credit — 10.75% of the bar
   does not justify a solo dispatch.

Owner: rate owner. Cost: $0 local build + a rebuild. Prerequisite: none.

---

## 5. Honest limits

1. **Not byte-closed.** Stated four times above because it is the one caveat that matters.
2. **Frame-granularity statistics.** Every number is a LOWER bound: a per-group updater (190
   groups/frame, 114,000 update steps instead of 600) adapts strictly faster. Unmeasured.
3. **Context space unswept.** I measured 8 forms. There was no systematic sweep, and the ladder
   was still climbing at G3 (+25.58 B from the last term added). The 1,549 B is not a ceiling.
4. **Decode cost is an instrument timing**, not a receiver timing (§2.8).
5. **Composition with any future token field is unmeasured** (§3).
6. `verdict_scope` on the negatives: the miss-branch correction is **INSTANCE** (two context
   forms, this vehicle); the t−1-agreement term is **INSTANCE**. Neither closes a family. The
   free-augmentation axis is **OPEN and positive** — this arm did not close it, it opened it.

---

## NEXT_IF_RESUMED

Bars must be read from
`tac.canonical_equations.sub015_pure_rate_archive_byte_bar_20260816.pure_rate_byte_bar_from_pointer()`,
never from a literal.

| # | row | owner | fire condition | ready? |
|---|---|---|---|---|
| 1 | **Build the free corrector** into receiver + encoder and byte-close against the 181,210 B target in §4. Zero distortion by construction; the parse-back is the proof. | rate owner | immediate, $0 local | design READY, build owed |
| 2 | **Per-group statistics.** Re-run the G3 ladder updating at group boundaries (190×/frame). Pure upside on a measured lower bound; ~1 h of CPU. | ddm_rr1 successor | with row 1 | instrument READY (`--frames`/tag already parameterised) |
| 3 | **Context sweep.** G3 was still climbing. Add depth-3/4 run-length, previous-frame neighbourhood pattern, and a mixed second predictor. | ddm_rr1 successor | after row 1 lands the plumbing | READY |
| 4 | **Re-measure the free credit on any new token field.** Binding on the Schur-compensated drop, on any HPAC capacity change, and on any semantic-width change. Not additive with them. | whoever moves the field | at that arm's harvest | — |
| 5 | **Retire `ra2`'s vacuous gate.** The CPR1 inner-coder row (~230–278 B, zero distortion, clears admission by 44–53×) is blocked only by its own "fire when a ≥2 KB rung is in flight" clause. **Row 1 above is a 1,549 B rung in flight — the gate's condition is now satisfied.** | ra2 / rate owner | row 1 charters | needs repack-layer re-measure (117 B uncertainty) |
| 6 | **The bar is 14,414 B strict, ceiling ≤ 168,345 B.** The e480b-lineage `−15,157 B` and the `<186,269 B` MC36 bar are both stale; this memo is the eighth site to correct them. | MAIN | standing | — |

**Retracted / not claimed:** no fire-order, no candidate archive, no exact eval, no byte-closed
row. The pointer did not move and this arm did not move it.

---

## Artifacts (ALWAYS KEEP THE PAYLOAD)

Store root `/Volumes/APDataStore/pact/ddm_rr1/`. 52 files, **1,849,426 B**, every one with
sha256 + byte count in `RETENTION_MANIFEST.json`. Nothing was measured and discarded.

| result | bytes | sha256 (16) |
|---|---:|---|
| `FREE_MODEL_HEADROOM_n600.json` | 6,180 | `6f13c45ab67561be` |
| `FREE_MODEL_OPTIMAL_FORM_n600.json` | 6,992 | `4afb1e1885f041e5` |
| `retained/optimal_form/delta_hit_G3_rl8_t2_n600.npy` (the winning table) | 409,728 | `a8e1cb741ba2d12b` |
| `retained/optimal_form/bits_per_frame_G3_hit_rl8_t2_n600.npy` | 4,928 | `156788c97555ef49` |
| `retained/free_model/bits_per_frame_base_n600.npy` | 4,928 | `e5b495f243f6dbcd` |
| `retained/optimal_form/bits_per_frame_G1_hit_rl4_n600.npy` (≡ stage 1 F4) | 4,928 | `7120c292ad7eb484` |

**Landed instruments** — both carry the fail-closed byte-identity control and both refuse to emit
a verdict without it:

* `experiments/ddm_rr1_free_decode_model_headroom.py` — stage 1, the ladder and the negative
  control.
* `experiments/ddm_rr1_free_model_optimal_form.py` — stage 2, optimal form, the branch split, and
  the cross-stage reproduction gate.

Both consume `ddm_hm1`'s retained `base_logits_int16_n600.i16` (1,179,648,000 B) and price a
probability-side change in ~70 s instead of a ~950 s HPAC forward. hm1 predicted that asset would
pay for itself; it did, twice, in this arm.
