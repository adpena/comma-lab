# V10 M2 live target selection — n600 byte-closed receipt (2026-07-20)

**Verdict: MEASURED POSITIVE DISTORTION RECOVERY, RATE-DEAD DIRECT-TARGET FORMULATION.**
`[macOS-CPU advisory]`, `score_claim=false`, pointer `0.1910828242 [contest-CPU]`
**UNMOVED**. MAIN review/merge is required before this landing has repository authority.

The corrected target closes the full capstone nonrate gap: the unmodified official CPU-Torch
evaluator measured `d_seg=0.0`, `d_pose=0.0` over n600, recovering
`0.04710838004286111 S`. The real counted packet is `1,717,172,741 B`, however, so the measured
triple versus the capstone spine is:

`(delta_d_seg, delta_d_pose, delta_bytes) = (-1.5196e-4, -1.0184e-4, +1,307,645,816 B)`.

This does not restore the claimed zero-byte 216,300→264,320 B budget widening. It proves the
target-selection distortion hypothesis and falsifies only the **direct exact-source-preimage
payload** formulation: the source-dependent fractional target is not derivable from the rounded-Y
packet for free.

## What ran

`tools/measure_m2_live_target_selection.py` consumed the sha-pinned n600 GT cache, interleaved both
camera frames, and used the source four-pixel block as the constructive bounded-Diophantine witness
for every scorer cell. Per 12-pair resumable chunk it measured all four canonical #49 tier-1 fill
policies, retained the smallest real Brotli-q5 section, proved exact resize-numerator equality,
packed the 50 selected sections into a counted ZIP, ran the generated scorer-free inflate path, and
then invoked unmodified `upstream/evaluate.py --device cpu --seed 1234`.

- archive SHA-256: `0fee1b74b315f368cec8f009b8d1aec91b03972e70aa299b60d25338f25cfb6b`
- inflated raw SHA-256: `a7192f9387856c849d406a322a08ff77080502751ac200cc63fe80a704989dd5`
- GT cache SHA-256: `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
- full receipt: `.omx/research/m2_live_target_selection_20260720T1548Z.json`
- SSD custody: `/Volumes/VertigoDataTier/pact/evidence/m2_live_target_selection_20260720T1528Z/`

The first inflate attempt failed closed because bare host `python3` lacked Brotli. The receiver was
made runtime-explicit through `M2_PYTHON`, the 50 preserved chunks resumed without recomputation,
and the official oracle completed. No failed-run number is used.

## Required decomposition

### Encode versus realize

- `d_A=0.0`: all `707,788,800 / 707,788,800` exact resize numerator values match the unrounded
  source target.
- `d_B=0.0`: hard CPU-Torch replay has zero Seg flips and zero Pose MSE.
- Because both official distortions are means of nonnegative terms, aggregate exact zero implies
  zero flips in every source class and margin stratum and zero MSE in all six pose dimensions.

Class denominators were `[27,407,046, 690,639, 58,413,281, 1,460,325, 29,993,509]`; every class
has zero flips. Margin-stratum denominators from `<1e-6` through `>=1e-2` were
`[3, 33, 299, 3,018, 29,859, 117,931,588]`; every stratum has zero flips. Thus recovery is not a
tie-only effect: exact numerator custody removes the plane-quantization gap across the full scorer
surface.

### Null geometry and bytes

- Full real-linear resize nullity is `80.6742315%`, but #49 currently implements only the
  `22.6969261%` axis-aligned integer-exact zero-weight mask. This arm changed `725,830,931` camera
  values in that mask without changing a numerator.
- Direct source sections: `1,989,784,190 -> 1,717,131,154 B`, a measured
  `-272,653,036 B` (`-13.7026%`) min-description gain.
- Winner split: horizontal predictor `41 chunks / 492 pairs / 221,963,112 B freed`; constant
  `6 / 72 / 33,760,257 B`; neighbor mean `3 / 36 / 16,929,667 B`; vertical `0`.
- ZIP/manifest overhead is `41,587 B`. The **source-dependent target sections** remain the
  dominating byte term; the null fill is helpful but cannot erase target custody.

The recovered `0.04710838 S` pays for only `70,748.29` extra bytes at the contest waterline. The
direct formulation spends `1,307,645,816` extra bytes—about `18,483x` the break-even allowance—and
therefore worsens total S by about `870.66` despite perfect distortion.

## Scoped signal and reformulation

The negative is **FORMULATION-scoped**: direct exact-source preimage sections are rate-dead. It is
not a negative on target selection, the bounded-uint8 family, or the v10 plane descriptor.

The measured signal names the next coordinate: do not carry every source numerator. Starting from
the compact rounded-Y plane, select only floor/ceil (or small numerator-residual) decisions whose
hard-oracle margin/pose benefit pays rent, and entropy-code those decisions **inside** the existing
plane descriptor. Admit only if the full charged fractional-selection stream is `<=70,748 B` and
the n600 byte-closed hard oracle retains the gain; reverse-waterfill and stop at the KKT rate
break-even. This converts the present `1.307 GB` custody wall into a sparse target-decision problem
rather than pretending the lost source fractions are a receiver-side zero-byte property.

## Triality and stores consulted

- **DSL:** N/A; this is a deterministic measurement/receiver arm, not a trainer lever.
- **Equations:** appends the n600 empirical anchor to
  `bounded_uint8_resize_preimage_cell_feasibility_v1`.
- **DAG:** `FEED-M2-LIVE-TARGET-SELECTION-20260720` appended to the canonical research DAG.
- **Stores consulted:** SPEC_v10 ADDENDUM-B/box C, FEED-M2 and FEED-M2recon, canonical equations
  registry (#49/#532/#547), graph-memory recall, capstone byte-closed receipt, lane registry,
  active dispatch claims, subagent progress, and live delegation inbox.

No paid dispatch occurred. No contest pointer or promotion state changed.
