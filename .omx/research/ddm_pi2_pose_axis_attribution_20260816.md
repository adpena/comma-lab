---
arm: ddm_pi2
title: "The 21.4x advisory-vs-CUDA d_pose offset is 99.996% GT-DECODE-PATH drift, and the root cause is a GT-LINEAGE SPLIT INSIDE OUR OWN INSTRUMENT: the seg half reads a cached DALI/nvdec-lineage argmax (3 sites of 117,964,800 from authority) while the pose half decodes its own GT with PyAV every run (21.43x). So this is a FIX, not a caveat -- point advisory pose at the retained DALI pose table and it tracks the authority at 1.0008x, measured. Seg is NOT axis-stable either (PyAV GT costs 1.4425x, reproducing the contest-CPU seg row from local files); it only looked sound because its cache was already the authority decode."
utc: 2026-08-16
parent: ".omx/research/ddm_rn1_render_boundary_mechanism_20260816.md"
sister: ".omx/research/ddm_ps1u_uncapped_pose_solve_20260816.md"
axis: "[macOS-CPU advisory] frozen CPU-torch PoseNet + SegNet -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "FORMULATION for the mechanism, the lineage split and the rule (properties of upstream's two GT decode paths and of our own caches, not of any archive); INSTANCE for every hv1-specific coefficient"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_pi2 — what the 21.4x pose offset actually is

STORES CONSULTED: parent `ddm_rn1_render_boundary_mechanism_20260816.md` §3.2b (the defect) ·
sister `ddm_ps1u_uncapped_pose_solve_20260816.md` §5/§5b (device-dependent decode, measured
today) · `ddm_pr130_reproduce_20260809/FX4_GT_LINEAGE.md` + `FX4_GT_LINEAGE_RECEIPT.json` ·
`experiments/modal_dali_av_gt_cache_diff.py` (job #906) ·
`cpu_vs_cuda_drift_engineering_analysis_20260520T173724Z.md` (the A/B split it labels *a fit,
not a measurement*) · `a1_pr106_cpu_cuda_axis_validation_20260513_codex.md` ·
`codex_30k_strategy_review_cpu_gpu_loader_drift_20260508.md` (the 2x2 design, never completed) ·
`ddm_rm1_20260808/chroma_siting_sensitivity.json` · `upstream/{evaluate.py,frame_utils.py,modules.py}`
· `ddm_mt1_t4_sign_gate_verdict_20260814.md` · the #1054 mc36 paired rows · memories [[m96]]
[[m88]] [[et4]], CLAUDE.md "Apples-to-apples evidence discipline".

## ANSWER FIRST

**The advisory instrument is not broken and the offset is not an axis of nature. It is a
GT-LINEAGE SPLIT inside our own tooling, and it is fixable today at $0.**

| term | measured | share of the offset |
|---|---:|---:|
| **(A) GT-decode path**, PyAV `yuv420_to_rgb` vs DALI/nvdec | **1.4061324889e-04** | **99.9960%** |
| (B) scorer-forward + platform, local macOS-CPU vs T4 | 3.572e-12 | 0.0000025% |
| (C) our own inflate's device dependence + all remainder | **<= 5.576e-09** | **<= 0.0040%** |
| total offset (advisory n600 − contest-CUDA) | 1.405861e-04 | 100% |

1. **THE ROOT CAUSE, and it is embarrassing in the useful way.** Our advisory instrument reads
   **two different ground truths**. The seg half loads the cached
   `gt_argmax_n600.npy`, which I measured to be **DALI/nvdec lineage — 3 differing
   sites out of 117,964,800 (2.54e-08) from the authority GT**, and 20,672 from the PyAV GT. The
   pose half has no cache, so `rn1`'s `decode_gt` (and my `n600`) decode GT fresh with PyAV every
   run. **One instrument, two lineages.** Seg was silently reading the authority decode; pose was
   silently reading the other one. That is the whole 21.4x.

2. **So the deliverable is a FIX, not a caveat.** Scoring our frames against the retained
   DALI-lineage pose table gives **d_pose = 6.885576e-06** against the contest-CUDA authority
   **6.88e-06** — **ratio 1.00081**, and the authority row is quoted to 3 s.f. (halfwidth ±5e-09),
   so the residual is indistinguishable from zero at the authority's own precision. **The advisory
   pose instrument becomes authority-tracking by changing which GT table it reads.** No CUDA host,
   no nvdec knowledge, no conversion factor. §0 is the one-line change.

3. **The additive closure proves the mechanism independently.** `d_pose` is an MSE between two
   nearly equal 6-vectors, so a reference perturbation adds incoherently. Predicting the advisory
   baseline as `6.88e-06 + 1.40613e-04 = 1.474932e-04` lands on my own n600 measurement of
   **1.4746613e-04** — a **0.018%** closure. Nothing was fitted.

4. **The same constant appears on a second, unrelated archive.** #1054: contest-CPU 1.4741e-04 vs
   contest-CUDA 6.88e-06 on mc36 (`f0ba4bb4`, 186,269 B) → offset **1.40530e-04**. hv1
   (`80d9c8c6`, 182,759 B) → **1.40586e-04**. Two archives, one constant, because the term belongs
   to `0.mkv` and the two decoders — not to us.

5. **SEG IS NOT AXIS-STABLE EITHER, and I had this backwards before I measured it.** Score the
   same frames against the **PyAV** GT argmax and d_seg is **4.271444e-04 = 1.4425x** the
   authority — against #1054's measured contest-CPU seg ratio of **1.442x**. So I reproduced BOTH
   contest-CPU axes from local files, to 4 significant figures on pose (21.434 vs 21.426) and 4 on
   seg (1.4425 vs 1.4423), using nothing but GT lineage. rt1's "advisory seg is right to 0.021%"
   is **correct and reproduced exactly** (my authority-cache leg: 1.00021x) — but it is a fact
   about *the cache*, not about the seg axis.

6. **rn1's candidate (B) is dead by nine orders of magnitude, and my own intended falsification of
   it was wrong.** I planned to kill (B) by magnitude: invert the offset for the required relative
   forward drift. Measured `|P|_rms = 12.795`, which makes that drift **7.5e-04** — entirely
   plausible for cuDNN-vs-oneDNN on a deep conv net. **The argument fails.** (B) is instead killed
   by direct measurement: local macOS-CPU-AV vs T4-AV pose MSE **3.572e-12**, plus `ddm_mt1`'s T4
   arm at ratio 0.9999 with bit-identical GT pose vectors.

7. **A third cause existed that neither rn1 nor its charter named, and I bounded it.** Our inflate
   is device-dependent (ps1u, today: `80d9c8c6` → raw `e5539653…` on cpu, `9a6b75e5…` on cuda).
   My opening reasoning — *"the compressed side is a raw mmap on both axes, so our frames cannot
   contribute"* — was **wrong**; the mmap reads whatever `inflate.sh` wrote. §3 bounds that
   defect's pose cost at **<= 5.58e-09 d_pose = +3.4e-06 S**. Real compliance defect; not the
   cause here.

8. **No decoder convention I could construct identifies nvdec's, and the amplitude ladder I built
   to price it does not transfer.** Six legal conventions at n=96; none descends (best 17.6x). And
   at matched amplitude a structured convention change costs **37x less** d_pose than white noise,
   so no LSB-equivalent can be read off the ladder. I drafted such a number and withdrew it (§5).

**Pointer UNMOVED.** hv1 ep0634 remains S 0.15959729295498598 @ 182,759 B [contest-CUDA T4].
This unit repaired an instrument. It did not lower the score.

## §0 THE RULE — what advisory numbers may and may not be used for

**Binding on rc4, ra3, hg1 and the wd3 line, and on anything that measures pose or seg advisory.**

### 0.1 POSE — FIX FIRST (preferred; makes the instrument authority-tracking)

> Score against the **DALI-lineage** GT pose table, not a freshly PyAV-decoded GT:
> `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt["pose"]`
> (sha `a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994`, 117,980,732 B,
> `(600,6) float32`). MEASURED: this reproduces the contest-CUDA authority at **1.00081x** on the
> live frontier object. `experiments/ddm_pi2_pose_axis_attribution.py crossaxis` does exactly this.

Any pose measurement that calls `frame_utils.yuv420_to_rgb` on `0.mkv` to build its own reference
is on the **contest-CPU** axis and is **21.43x** the authority. Label it or fix it.

### 0.2 POSE — FALLBACK CONVERSION (only when the fix is impossible)

> Advisory pose carries a **fixed additive floor of `d_pose = 1.4061e-04`**. Quote it **only as an
> absolute `delta d_pose`**, and convert with
>
> ```
> dS_pose = sqrt(10 * (6.88e-06 + delta_d_pose_abs)) - sqrt(10 * 6.88e-06)
> ```
>
> **Never** rescale an advisory `dS_pose`. **Never** quote an advisory `d_pose` ratio.

The floor cancels exactly in the absolute delta. It does not cancel in the ratio, and it does not
cancel in the `sqrt`, which the naive reading evaluates at a baseline 21.4x too high where the
marginal `dS/d(d_pose) = 5/sqrt(10 d_pose)` is `sqrt(21.4) = 4.63x` too small.

### 0.3 SEG

> Keep using the cached `gt_argmax_n600.npy` — MEASURED **DALI lineage, 3 sites of 117,964,800
> from authority, d_seg ratio 1.00021**. **Never decode GT yourself with PyAV for a seg
> measurement**: that is the contest-CPU axis and costs **1.4425x**.

### 0.4 Why a single multiplier would have been wrong

The implied pose correction factor is **not constant**. Applying §0.2 to rn1 §3.2's four rows (its
ratios × its n=24 baseline give the absolute deltas; my `addit` stage re-measured two of them
independently and agrees to **0.06%** and **0.01%**):

| operator | rn1 `dS_pose` as published | **converted `dS_pose`** | factor | **× the 0.0095973 gap** |
|---|---:|---:|---:|---:|
| `dither:amp=1` | +0.002406 | **+0.027545** | 11.45x | **2.87x** |
| `gain:g=0.98` | +0.006598 | +0.056052 | 8.50x | 5.84x |
| `gamma:gamma=1.02` | +0.008846 | +0.069535 | 7.86x | 7.25x |
| `gain:g=1.02` | +0.010394 | +0.078499 | 7.55x | 8.18x |

A flat multiplier is wrong by up to 52% across these four rows alone. **rn1's own independent
absolute-delta cross-check gave +0.0276 for the dither row; the conversion gives +0.027545 — a
0.2% match.** rn1's absolute reading was right, its ratio-based worked-through (+0.0217) was not,
and its OPTIMISTIC-bound caveat now has an exact coefficient. Strengthened form of rn1's own
conclusion: **a one-LSB dither is 2.87x the entire remaining gap**, not "25% of it".

### 0.5 Two limits that survive even after the fix

1. **RESOLUTION, if you do NOT take the fix.** The floor is 20.4x the authority baseline, so a
   *perfect* pose carrier moves the advisory number only 4.7% (1.4747e-04 → 1.4061e-04), inside
   the n600 pair-to-pair spread of ±12%. Advisory-with-floor measures pose DAMAGE well and pose
   GAIN barely. Use paired per-pair differencing, never two independently measured means.
2. **Difference against the CANONICAL floor only.** My pre-registered additivity bar (`< 20%`
   spread across floors) **FAILED**, instructively: the same operator reads `1.2164e-04` against
   the canon floor, `1.3342e-04` against a same-magnitude floor (9.7% apart, fine), and
   `3.893e-04` against an 11.9x floor — and a second operator **sign-flips** there. Differencing
   two large numbers to recover a small one destroys the estimate.

## §1 Prior-law prediction lines — stated BEFORE the measurements

Per the anti-re-anchor law, and because four of these were wrong.

1. **PREDICTION: the offset is ~1 LSB of GT-decode difference, and one of the legal decode
   conventions will bring advisory `d_pose` DOWN to ~6.88e-06, identifying nvdec's convention and
   yielding a one-line decode cure.** **WRONG on both halves, and I withdrew a number over it.**
   No convention descends (best 17.6x, §5). The amplitude claim does not stand either: the
   white-noise ladder I built to price it is **not transferable** — at matched amplitude a
   structured convention change costs 37x less d_pose than white noise (§5). I had drafted "~0.85
   LSB, squarely inside the measured envelope" into this memo and removed it.
2. **PREDICTION: (B) scorer-forward drift is falsifiable by magnitude, because `|P|` is O(1) and
   the required relative drift will be absurd.** **WRONG.** Measured `|P|_rms = 12.795`, so the
   required drift is 7.5e-04 — plausible, not absurd. My falsification collapsed; (B) had to be
   killed by direct measurement instead.
3. **PREDICTION (implicit in my opening reasoning): our own frames are byte-identical across axes
   because `TensorVideoDataset` is a raw mmap, so a third cause cannot exist.** **WRONG** — ps1u
   measured the same archive inflating to two different raws. I carried (C) explicitly afterwards
   and bounded it at <= 0.0040%.
4. **PREDICTION: the additivity bar `< 20%` across floors spanning ~7x will HOLD.** **FALSIFIED**
   at 125% and 195% spread. Additivity is sound as physics — the 0.018% closure proves it — but
   the *estimator* collapses when the floor dwarfs the signal. Turned into limit 2 of §0.5.
5. **PREDICTION: seg is axis-stable BY CANCELLATION — the GT-decode change is an undirected
   perturbation at the label boundary, so rn1's `rho(0.01)=0.985` fair coin makes it invisible to
   argmax while MSE sees it in full.** **WRONG, and I wrote it into a draft of §4 before measuring
   it.** Seg is not axis-stable at all (1.4425x on the PyAV GT); it only looked stable because its
   cache is authority-lineage. And the fair coin does not apply: of the 20,671 GT sites that move,
   **74.8% create a disagreement and only 25.2% cancel one** (§4). rho governs perturbations of
   OUR render, where flips and correct pixels are both dense at the boundary. It does not govern a
   perturbation of the REFERENCE, where we agree with the old reference 99.97% of the time so
   almost any change creates a disagreement. rho is not falsified; I mis-applied it.
6. **PREDICTION (from #1054's 21.4x sitting next to rn1's 18.2x): the advisory instrument is
   simply reproducing the contest-CPU axis.** **HELD**, and sharpened past what I expected: it
   reproduces contest-CPU on **both** axes to 4 s.f. from GT lineage alone (§4).

## §2 The separator — one host, one scorer, two GT decode paths

The decisive measurement needed no dispatch: it was already retained. Job #906
(`experiments/modal_dali_av_gt_cache_diff.py`, 2026-08-09) ran PR130's own
`build_gt_cache_official.py` **twice in one container on one Tesla T4** — `--dataset av`, then
`--dataset dali` — so the caches differ **only** by the GT decode path, with scorer, host, driver
and clock fixed. That is the 2x2 cell the repo has wanted since 2026-05-08 and never filled (the
two probes built for it both lost their CUDA leg; the one real T4 attempt died on `nvml error 999`).

Re-derived here, not inherited:

| quantity | measured |
|---|---:|
| pose MSE, AV GT vs DALI GT, n600, same T4/scorer | **1.4061324889363773e-04** |
| components exactly equal | **0 / 3,600** |
| per-dim MSE, dim 0 | 8.4322e-04 (**99.9%** of the total) |
| `|P|_rms`, DALI cache | 12.773398 |
| `|P|_rms`, my local instrument | 12.772428 (**0.008%** apart) |
| seg argmax sites moved, AV vs DALI | **20,671 / 117,964,800 = 1.7523e-04** |

`gt_cache_av.pt` sha `837b5852…` (117,980,720 B) · `gt_cache_dali.pt` sha `a91d9825…`
(117,980,732 B), both on VertigoDataTier, read-only.

## §3 The closure — three independent pose legs agree

| leg | d_pose | vs contest-CUDA | source |
|---|---:|---:|---|
| advisory n600, my instrument, PyAV GT + cpu-decoded frames | **1.4746613e-04** | 21.434x | `PI2_N600_base` |
| `crossaxis` control, T4-AV GT + same frames | **1.4746785e-04** | 21.434x | `PI2_CROSSAXIS_base` |
| **`crossaxis` authority, T4-DALI GT + same frames** | **6.885576e-06** | **1.00081x** | `PI2_CROSSAXIS_base` |
| contest-CUDA authority (3 s.f.) | 6.88e-06 | 1x | hv1 T4 receipt |

**The control matters as much as the result.** The T4-AV leg reproduces my own locally computed
advisory baseline to **0.0012%** (1 part in 85,000). That one number validates three things at
once: the cache's pose convention is upstream's, the local-vs-T4 scorer forward is negligible
here, and my instrument is rn1's instrument (its n=96 `canon` row also reproduces rn1's
1.251833e-04 exactly).

**Residual after attributing (A):** `6.885576e-06 − 6.88e-06 = 5.576e-09`, i.e. **0.0040%** of the
offset — **below the authority row's own ±5e-09 rounding halfwidth**. In score units the
device-dependent inflate therefore costs **<= +3.4e-06 S** on pose. Custody: raw sha verified
`e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` in-tool before the run, on
archive `80d9c8c6…` whose inflate output is the bit-identity reference reproduced 4/4 by wc1.

## §4 The seg leg — the control everyone leaned on was reading a different GT

`PI2_SEGAXIS_base`, n600, our full argmax field computed and retained, scored against three
references:

| reference | flips / 117,964,800 | d_seg | vs contest-CUDA |
|---|---:|---:|---:|
| **T4-DALI GT (authority lineage)** | 34,935 | **2.961477e-04** | **1.00013x** |
| local cached `gt_argmax_n600.npy` | 34,938 | 2.961731e-04 | **1.00021x** |
| T4-AV GT (PyAV lineage) | 50,388 | **4.271444e-04** | **1.44252x** |

And the lineage test that names the root cause:

| comparison | differing sites / 117,964,800 |
|---|---:|
| `gt_argmax_n600.npy` vs **T4-DALI** | **3** (2.54e-08) |
| `gt_argmax_n600.npy` vs T4-AV | 20,672 |
| T4-AV vs T4-DALI | 20,671 |

**The cached GT argmax our whole seg line uses is DALI/nvdec lineage.** That is why rt1's
advisory-vs-CUDA seg check read 0.021% — I reproduce it exactly at 1.00021x — and it is a fact
about the cache, not about the axis. Swap in the PyAV GT and seg is 1.4425x, which is #1054's
measured contest-CPU seg ratio of 1.442x. **Both contest-CPU axes are now reproducible from local
files with no CPU dispatch**: pose 21.434 (vs 21.426 measured) and seg 1.4425 (vs 1.4423).

**Why seg is only 1.44x while pose is 21.4x** — not cancellation, which is what I predicted and
got wrong. Of the 20,671 GT sites that move between decode paths, **15,453 net become new
disagreements: 74.8% create, 25.2% cancel.** The asymmetry is forced: we agree with the reference
at 99.97% of sites, so moving the reference almost always creates a disagreement. Seg is milder
than pose only because `d_seg` counts a *bounded* quantity — one flip per changed site, capped by
the 20,671 — while `d_pose` accumulates a *squared* error with no such cap and 99.9% of it lands
in a single pose dimension.

## §5 The decoder-convention race — the negative that closes the cheap cure

`PI2_ATTRIB_n96`, n=96 seeded-random (rn1's seed and pair set), six legal conventions plus a
white-noise ladder, each scored on pose AND seg. Two positive controls were paid first: `canon` is
**byte-identical to upstream's `yuv420_to_rgb`** (max abs diff 0), and it reproduces rn1's n=96
baseline at **1.2518e-04 vs 1.251833e-04**.

| variant | d_pose | /CUDA | /canon | px mean-abs | px max | d_seg | seg % vs canon |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chroma_align_corners` | 1.2088e-04 | 17.6 | 0.966 | 0.483 | 39 | 4.5872e-04 | +10.73 |
| **`canon` (= contest-CPU)** | **1.2518e-04** | **18.2** | 1.000 | 0.000 | 0 | 4.1427e-04 | 0.00 |
| `chroma_nearest` | 1.4537e-04 | 21.1 | 1.161 | 0.878 | 81 | 5.2770e-04 | +27.38 |
| `round_trunc` | 4.1230e-04 | 59.9 | 3.294 | 0.483 | 1 | 4.3880e-04 | +5.92 |
| `matrix_bt709` | 3.9468e-03 | 573.7 | 31.528 | 0.745 | 33 | 1.2753e-03 | +207.85 |
| `range_full` | 6.4779e-03 | 941.6 | 51.747 | 13.003 | 21 | 8.8692e-04 | +114.09 |

**Nothing descends.** The only variant below `canon` is `chroma_align_corners`, by 3.4%, still at
17.6x. nvdec's convention is not in the obvious set; a one-line decode fix does not exist.

**And the ladder cannot price it — a negative against my own design.** I built the white-noise
ladder to convert "LSB of decoder disagreement" into d_pose. It does not transfer: `chroma_nearest`
at **0.878** LSB mean-abs costs `Δd_pose = 2.0e-05`, while white noise at **0.747** LSB mean-abs
costs `7.5e-04` — **37x more d_pose at 15% less amplitude**. No LSB-equivalent for the GT-decode
difference can be read off this ladder, and I withdrew the estimate I had drafted from it. The
structural reason is in §2: **99.9% of the AV-vs-DALI pose difference lives in dim 0 alone** — a
specific low-order perturbation, not broadband noise. The ladder's surviving value is the *limit*
it established for §0.5.

**Cure direction:** §0.1 needs no knowledge of nvdec's convention at all, because it works from
the measured scorer-output difference rather than from pixels. That is why the fix is available
today and the pixel-level identification is not needed for the score.

## §6 What this unit did NOT establish

- **No score.** Every row is `[macOS-CPU advisory]`. The pointer is unmoved; this unit was not
  permitted to move it and did not try.
- **nvdec's RGB convention is still unidentified at the PIXEL level.** Nobody in this repo has
  ever compared nvdec RGB bytes against `yuv420_to_rgb` RGB bytes. I closed the question at the
  **scorer-output** level, which is what the score depends on, and left the pixel level open.
- **The residual is bounded, not resolved.** 5.576e-09 sits under the authority row's 3-s.f.
  rounding, so I bound (C) but cannot measure it. Resolving it needs a CUDA row quoted to more
  digits, or the per-frame hash diff ps1u staged to ride free on the next T4 fire.
- **The 1.4061e-04 floor and the 1.4425x seg factor are `0.mkv`-specific.** They are properties of
  one clip and two decoders. Mechanism is `verdict_scope: formulation`; the *coefficients* must be
  re-measured if the clip changes. Both retained caches cover video 0 only.
- **Within-DALI drift is nonzero and not folded in.** Retained-Ada vs fresh-T4 DALI is 1,644 seg
  flips and pose MSE 1.065e-06. My authority leg used the fresh-T4 DALI cache, matching the axis
  the hv1 row was measured on; a different DALI version would move the floor by ~0.8%.
- **I did not re-open rn1's seg findings.** rho, the exchange rate, the 5.95x-sharper witness
  measurement and the +0.1-logit hinge bound are untouched. The correction lands on rn1 §3.2's
  pose column only, in the direction rn1 itself flagged.
- **I did not verify `gt_argmax_n600.npy`'s documented provenance.** I established its lineage
  *empirically* (3 sites from the DALI cache). No producer receipt was located that says so, which
  means the seg line has been relying on an authority-grade cache **by luck, undocumented** — a
  standing fragility, and item 2 of §9.

## §7 My own round-1 adversarial review (a fix is unreviewed new code)

1. **Is the 0.018% closure luck?** Partly. The cross-term `2<e,d>` is zero-mean but fluctuates by
   `~2|e||d|/sqrt(6*600) = 1.04e-06`, i.e. **0.7%** of the offset. Honest claim: "attributed to
   within ~1%". The `crossaxis` leg does not depend on the cross-term at all — it scores against
   the authority GT directly — and it is the leg I lead with.
2. **Did I validate the cache convention or assume it?** Assuming would have been a fake. The
   `leg_av` control exists to test it and reproduces my independent local n600 to 0.0012%.
3. **Am I manufacturing the fudge factor the charter forbade?** No: the mechanism is identified by
   direct measurement, so §0.2 is that mechanism's algebra, and §0.1 removes the need for it
   entirely. Evidence it is not a fudge: the implied factor *varies* 7.55x–11.45x across four
   rows, so no single multiplier exists to fit.
4. **Does 3 s.f. on the authority value undermine the result?** It caps *residual* resolution at
   ±5e-09 = 0.004% of the offset. It cannot touch the other 99.996%.
5. **I wrote a wrong §4 before measuring it.** My rho-cancellation story for seg was elegant and
   false. I had already noticed the tension while drafting (we agree with the reference 99.97% of
   the time, so reference changes should mostly *create* flips) and ran the measurement instead of
   publishing the story. 74.8% create / 25.2% cancel. Recorded as prediction 5.
6. **Did I check the sister arm before contradicting it?** I do not contradict ps1u — I complete
   it. ps1u eliminated the instrument-gap reading and proved device-dependent decode; I supply the
   term it named but could not measure (the GT decode path) and bound what device-dependent decode
   costs on pose (<= 0.0040%). Its §4 caveat stands unchanged.
7. **Does the fix make the frontier row look better or worse?** Neither — it changes no shipped
   byte and no authority row. It changes what our local instrument reports. The gap to 0.15 is
   still 0.0095973 and rn1's frozen-field conclusion is *strengthened*, not weakened.

## §8 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_pi2_pose_axis_attribution_20260816/` (240 GiB free;
VertigoDataTier is at 893 MiB and was used read-only).

| artifact | what it is |
|---|---|
| `PI2_CROSSAXIS_base.{json,npz}` | the decisive pose leg: our 600 pose vectors + both GT pose tables + both per-pair d_pose series |
| `PI2_SEGAXIS_base.{json,npz}` | the seg legs + the full retained n600 argmax field (673,829 B compressed) |
| `PI2_N600_base.{json,npz}` | the advisory identity baseline, all 600 per-pair d_pose + GT pose vectors |
| `PI2_ATTRIB_n96.{json,npz}` | the six-convention race + white-noise ladder, per-pair rows |
| `PI2_ADDIT_n24.{json,npz}` | the additivity test and its falsified 20% bar |
| `PI2_SCALE_n24.{json,npz}` | `|P|` magnitudes and the forward-drift inversion |
| `logs/`, `logs_attrib/`, `logs_seg/` | launcher manifests + run logs |

Consumed unmodified and custody-verified in-tool: the wc1 retained decode `0.raw`
(3,662,409,600 B, sha `e5539653…`), `gt_cache_av.pt` (sha `837b5852…`), `gt_cache_dali.pt`
(sha `a91d9825…`), the qs3 `gt_argmax_n600.npy`, and `upstream/videos/0.mkv`.

Tool: `experiments/ddm_pi2_pose_axis_attribution.py`
(stages `scale` / `attrib` / `addit` / `n600` / `crossaxis` / `segaxis`).

## §9 NEXT_IF_RESUMED

| # | work | owner | fire-condition |
|---|---|---|---|
| 1 | Take §0.1: point every advisory pose measurement at `gt_cache_dali.pt["pose"]`. Recompute any already-quoted advisory `dS_pose` (or convert via §0.2 where the fix is impossible) | rc4 · ra3 · hg1 · wd3 | **NOW** — before any advisory pose coefficient is published |
| 2 | Build a **GT-lineage gate**: every GT cache carries a recorded lineage (`dali` / `av`) and every scorer instrument asserts which one it needs, fail-closed. The bug was two lineages in one instrument and nobody could see it; `gt_argmax_n600.npy` is authority-grade **by luck, undocumented** | ddm_pi2 successor | next unit touching a GT cache — this is the structural cure, the §0 rule is the procedural one |
| 3 | Publish the local contest-CPU predictor: seg ×1.4425 / pose +1.4061e-04 reproduce both #1054 axes to 4 s.f. from local files, so contest-CPU rows may be estimated without a CPU dispatch | any arm needing a CPU-axis estimate | when a CPU-axis number is wanted; validate on a second archive before trusting it |
| 4 | Per-frame hash diff to localize the device-dependent decode (ps1u's `CUDA_DECODE_DISPATCH_SPEC.json`) — bounded at <= +3.4e-06 S on pose, so it is a COMPLIANCE item, not a score item; do not price it as a prize | ps1u | rides free on the next T4 row; never its own dispatch |
| 5 | nvdec-vs-`yuv420_to_rgb` at the RGB PIXEL level — the 2x2 cell open since 2026-05-08 | unassigned | only if a CUDA host is already up for another reason; the score question is closed without it |
