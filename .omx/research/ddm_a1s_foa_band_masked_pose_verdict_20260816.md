---
arm: ddm_a1s_foa
title: "FO-A answered at n600: the A1 pose damage is NOT escapable by masking. Band-only drift rms is 0.018162 -- 6.92x the entire incumbent pose error, 2.19x the pre-registered CLOSED bar, and 1.27x above the drift at which the actuator loses even if it recovered ALL 33,743 manufactured round-trip flips. FAMILY_CLOSED. The mechanism is the finding: pose damage tracks perturbation ENERGY, not band membership -- band and interior are within 6% of each other per unit energy -- and the two halves partially CANCEL (cos -0.483), so restricting the actuator to either half makes it WORSE per unit of perturbation it keeps. Interior-only drift (0.023995) exceeds the whole actuator's own (0.022019). The seg leg confirms from the other side: masking keeps 40.1% of the seg win for 82.5% of the pose cost, cost/benefit 189.7x vs the unmasked 98.7x -- strictly dominated on both axes. Charter positive control passed exactly (34,938 flips, argmax sha 2aeb1e6b...)"
utc: 2026-08-17
parent: ".omx/research/ddm_a1s_alpha_sign_verdict_20260816.md"
fire_order: "ddm_a1s section 8 FO-A"
axis: "[macOS-CPU advisory] -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "FAMILY on the hv1 ep0634 vehicle, as FO-A pre-registered; narrower scopes named inline"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_a1s FO-A — is the A1 pose damage band-driven or interior-driven?

STORES CONSULTED: the parent `ddm_a1s_alpha_sign_verdict_20260816.md` — §4 (the pose ladder),
§6.1 (clipping), §7 (the two-GT-cache hazard) and §8 (the FO-A text) read verbatim before any
code was written · its retained receipts `SR1_A1SIGN.json` / `SR1_A1POSE.json` and every payload
in §9 · the committed tool `experiments/ddm_sr1_manufactured_seg_recovery.py` (its real argparse
grepped, never guessed) · `experiments/ddm_rt1_seg_roundtrip_decomposition.py` (`nn_lift_index`,
imported not re-typed) · `docs/operating_manual_craft_handoff.md` · CLAUDE.md
(NO-FAKE, ALWAYS KEEP THE PAYLOAD, n600-scale-or-not-evidence, the verdict_scope ladder,
"SegNet sees REGIONS, not pixels") · memories
`prefix_bias_sign_inverts_between_seg_and_pose_20260803` ([[m96]], read in full — pose prefixes
measure 2.54–4.21× HARDER, the reason nothing here is drawn from a prefix) and [[m88]] [[m91]].

## ANSWER FIRST

**The pose damage is band-driven, and that closes the family rather than opening it.**

Masking `Δ_cam` to the lifted label band leaves the pose drift at **rms 0.018162410533085427**
at α = 0.25, n600 — **82.48% of the unmasked actuator's 0.022019**, from **2.158% of the camera
pixels**. Against the two bars FO-A pre-registered:

| bar | value | measured band-only drift | verdict |
|---|---:|---:|---|
| LIVE below (the entire incumbent pose error `√d_pose`) | 0.0026240 | **0.018162** | not met, by **6.92×** |
| CLOSED at or above (3.2× the incumbent) | 0.0083 | **0.018162** | **FIRED, by 2.19×** |

**Verdict: `FAMILY_CLOSED`.** No bucket was forced; the measured value cleared the CLOSED bar by
more than a factor of two.

**A DERIVED bar makes it unarguable.** The largest seg win physically available to *any* actuator
on this axis is recovering rt1's entire manufactured round trip — all 33,743 flips —
= **0.028604 S**. Setting `ΔS_pose = 0.028604` and inverting the GT-free bound gives a
**break-even drift of 0.014294**. The measured band-only drift is **0.018162, which is 1.27×
above it.** So the band-restricted de-blur is a net S loss **even in the counterfactual where it
recovers every single manufactured flip.** That is why this row does not owe a seg measurement to
reach its verdict: no seg number in the achievable range can change the sign.

**The mechanism is the real finding, and it is stronger than the pre-registration anticipated.**

| leg, α = 0.25, n600 | pose drift rms | % of full | % of camera px | % of ‖Δ‖² energy |
|---|---:|---:|---:|---:|
| unmasked (the parent's A1) | 0.022019 | 100.00 | 100.000 | 100.00 |
| **band only** | **0.018162** | **82.48** | **2.158** | **35.11** |
| interior only | **0.023995** | **108.97** | 97.842 | 64.89 |

**The band geometry FO-A assumed is confirmed, with the figures refined** (n600 means, measured,
where FO-A stated them as approximations): the band is **2.158%** of camera pixels (FO-A: ~2.2%)
carrying `max|Δ|` of **20.03 levels** (FO-A: 22.5), and the interior is **97.842%** of pixels at
**1.69 levels** (FO-A: ~1.8). The premise was sound; the conclusion drawn from it is not.

Two things fall out that no single-leg measurement could have shown:

1. **The interior-only drift EXCEEDS the whole actuator's own drift.** Removing 2.158% of the
   pixels made the pose damage *worse*. The two halves partially **cancel**: the implied
   `cos(angle)` between the band and interior pose responses is **−0.4828**, and
   `hypot(band, interior)/full = 1.3667`. Restricting the actuator to either half destroys the
   cancellation that the full lever gets for free.
2. **Pose damage tracks perturbation ENERGY, not band membership.** Per unit of `‖Δ‖²` retained,
   the band scores **1.9379** and the interior **1.8300** — within **5.9%** of each other. The
   band looks like the culprit only because it is amplitude-dense: **26.0× the interior's pose
   damage per PIXEL**, but essentially the same per unit energy.

So there is no spatial subset of `Δ_cam` that is pose-cheap. The only lever that lowers the pose
cost is lowering the total perturbation energy — which is exactly lowering α, and the parent
already measured that the seg gain peaks at α = 0.25 and reverses above it. **The post-hoc
de-blur of `A` has no pose-null direction to hide in.**

**The seg leg confirms it from the other side (§6).** The band-restricted actuator keeps only
**40.1%** of the parent's seg win (−254 flips vs −634) while keeping **82.5%** of its pose cost.
Its cost/benefit is **189.7×**, nearly double the unmasked lever's 98.7×. **Masking is strictly
dominated on both axes at once** — and the charter's literal positive control passed exactly on
that same leg (34,938 flips, argmax sha `2aeb1e6b…`).

Pointer UNMOVED. No dispatch, no Modal, `[macOS-CPU advisory]` throughout, $0.

## §1 Positive controls — every one passed, and one is stronger than asked

| control | required | measured | verdict |
|---|---|---|---|
| **α = 0 seg flips (the charter's literal control)** | **exactly 34,938** | **34,938** | **PASS** |
| **α = 0 argmax field vs rt1 `argmax_base.npy`** | sha `2aeb1e6b…` | **`2aeb1e6be0f7c6ab8191b790204d8df0ae5fdce7ef2ecc5b2d18a715f1a674c4` — bit-identical** | **PASS** |
| default-off pose6 vs the parent's retained payload | — | **sha-identical**, `max abs diff 0.0` | **PASS (see §2)** |
| α = 0 camera frame vs the shipped frame | — | **bit-identical, asserted every pair, in code** | **PASS**, all 3 legs |
| rebuilt `d@u` vs sr1's retained `A_row` / `A_col` | exact | **max abs diff 0.000e+00**, both axes | **PASS**, all 3 legs |
| separable operator vs the real `F.interpolate` chain | rel < 1e-9 | **3.366119099208506e-16** | **PASS** (reproduces the parent's §1 exactly) |
| `A⁻¹A − I` residual, row / col | — | 1.7878039890728559e-06 / 1.7903853531953118e-06 | matches the parent's §1 |
| every input sha before use | parent §9 | **all 7 match** (§8) | **PASS** |
| emitted FO-A verdict vs the adjudicator re-applied to the drift | agree | `FAMILY_CLOSED` == `FAMILY_CLOSED` | **PASS** |

**On the charter's literal control.** It names a *seg* control, and FO-A's verdict comes from a
**pose** leg, which computes no argmax — so I ran the band-masked **seg** leg at n600 as well
(§6). It **passed in its strongest form**: 34,938 flips exactly, and the α = 0 argmax field is
bit-identical to rt1's base, same sha. The pose legs independently discharge the same control in
the form available to them — the α = 0 pose vector is byte-identical to the parent's retained
`pose6_by_alpha.npy` (§2), pinning today's instrument to the exact one that produced the parent
row: frozen CPU PoseNet, batch = 1 pair, the 8-thread pin, the operator rebuild. Both controls
reproduced; the instrument is sound and the verdict is admissible.

## §2 Byte-identity of the default-off flag — proven at n600, not argued

The flag added is exactly one: `--delta-mask {off,band,interior}`, default `off`.

I re-ran the **full n600 pose leg** with the flag at its default and compared to the parent's
retained payload:

```
retained parent  /...20260816/a1sign/pose6_by_alpha.npy       97e2c89969b6ccc0d3ad612d14714f009824d35698cb1f469311b69af34b7a70
FO-A re-run      /...20260816/foa/a1sign/pose6_by_alpha.npy   97e2c89969b6ccc0d3ad612d14714f009824d35698cb1f469311b69af34b7a70
sha_identical = true · array_bit_identical = true · max_abs_diff = 0.0 · shape (600, 5, 6) both
```

The default path is inert **by construction as well as by measurement**: with `off` the stage
never opens the token field, never builds a lift index, and never touches `Δ_cam` — the mask code
is unreachable, not a mask that happens to be all-ones. `_a1_apply_delta_mask` **refuses** the
string `"off"` outright, so a future edit that routed the default through it fails closed rather
than silently re-deriving the parent row. Two structural tests pin this (§7).

**What this does NOT claim:** the *receipt* `SR1_A1POSE.json` differs between the two runs, because
the FO-A work directory holds no seg receipt to join against, so the ladder rows there carry no
`delta_S_seg` field, and `wall_s` differs. The byte-identity claim is on the **payload** — the
measured object — exactly as the charter framed it.

## §3 The three legs — n600, frozen CPU PoseNet, `[macOS-CPU advisory]`

Instrument, identical across all three: frozen CPU-torch PoseNet from
`upstream/models/posenet.safetensors`, `PoseNet.preprocess_input` verbatim, first 6 pose dims per
`compute_distortion`, **batch = 1 pair**, `torch.set_num_threads(8)` (enforced in code; the stage
refuses to start if violated, per `et4` — batch shape and thread count are part of the instrument).

| α | full drift | band-only | interior-only |
|---:|---:|---:|---:|
| 0.00 | 0.000000 | 0.000000 | 0.000000 |
| **0.25** | **0.022019** | **0.018162** | **0.023995** |
| 0.50 | 0.050930 | 0.034018 | 0.054907 |
| 0.75 | 0.085918 | 0.047427 | 0.088932 |
| 1.00 | 0.128471 | 0.058693 | 0.127228 |

The interior leg exceeds the full actuator at **every** α (109.0% / 107.8% / 103.5% / 99.0% as α
rises), so the cancellation is real across the ladder and strongest where the verdict is taken.

The band leg **saturates** with α (0.0182 → 0.0587 over a 4× strength increase) while the full and
interior legs stay near-linear. That is the parent's §6.1 clipping mechanism seen from the pose
side: at α = 1, **24.9% of band camera pixels clip** against ~0.9% off the band, so the band leg
stops receiving the perturbation it is asked to apply. It does not rescue anything — the band leg
is already 2.19× over the CLOSED bar at α = 0.25, its cheapest rung above zero.

### The GT-free bound, and why it is the primary read

With `p_α = p_0 + δ`, the reverse triangle inequality gives
`√d_pose(α) ≥ | rms‖δ‖ − √d_pose(0) |` with **no GT pose target anywhere**. Using the
authoritative `d_pose(hv1 ep0634) = 6.885642960696714e-06` (`√d_pose = 0.0026241`):

| leg, α = 0.25 | drift rms | ÷ incumbent | `d_pose` LOWER bound | ΔS_pose LOWER bound |
|---|---:|---:|---:|---:|
| full | 0.022019 | 8.391× | 3.7617e-04 (54.6×) | **+0.053035** |
| **band** | **0.018162** | **6.922×** | **2.4144e-04 (35.1×)** | **+0.040839** |
| interior | 0.023995 | 9.144× | 4.5673e-04 (66.3×) | +0.059284 |

Every one of these is a **lower** bound — the best case the geometry permits for the actuator.
This matters and travels with the numbers: the true costs are at least this large.

The parent's §7 GT-cache hazard is fully avoided here. All three legs plus the byte-identity
control read the **same** instrument and the same inputs, so every leg-to-leg difference above is
free of any GT term. No cached GT pose target was consulted at all (`--gt-pose` unset on every
leg), so the parent's advisory-secondary read has no successor here and is not cited.

## §4 The decisive arithmetic — the bar no seg row can clear

Exact contest arithmetic, re-derived, not copied:

```
SEG_DS_PER_FLIP     = 100 / 117,964,800        = 8.477105035e-07   S per flip
largest seg win available = 33,743 flips       = 0.028604 S        (rt1's ENTIRE round trip)
solve  sqrt(10*d_pose) - sqrt(10*6.885643e-06) = 0.028604
   =>  break-even band-only drift rms          = 0.014294
measured band-only drift rms                   = 0.018162          -> 1.27x ABOVE
```

| what the band-restricted lever would have to win to break even | flips | achievable? |
|---|---:|---|
| at the measured drift 0.018162 (ΔS_pose ≥ +0.040838630) | **48,175** | **no — 1.43× more flips than the entire manufactured round trip, and 1.38× more than ALL 34,938 scored flips** |
| the parent's actual measured seg win at α = 0.25 | 634 | — |

The band-restricted actuator would need to fix **more flips than exist**. The pre-registered
CLOSED bar is calibrated to about the same place from the other direction: a drift of exactly
0.0083 costs **+0.009651 S**, which is **1.01× the entire remaining −0.0095973 gap.**

## §5 Verdict

**A1 restricted to the lifted label band — and, by §3's interior leg, A1 restricted to the
complement — is CLOSED. The post-hoc de-blur family has no pose-null spatial restriction.**

- **verdict:** `FAMILY_CLOSED`.
- **verdict_scope:** **FAMILY** — the *post-hoc* de-blur of `A` applied to frame_1 on the hv1
  ep0634 vehicle, under **any** spatial restriction of `Δ_cam`. This is the FAMILY escalation FO-A
  pre-registered, and it is earned by three n600 legs, not one: band, interior, and unmasked.
- **what is NOT in scope, stated plainly:** this closes *post-hoc* de-blurring. It does **not**
  close (a) FO-C, a **render-side** fix that changes frame_1 with the pose term in the training
  loop — nothing here measures that; (b) FO-B, a clipping-aware projection — but §4 now prices
  FO-B's ceiling, and it is unreachable (see §6); (c) any actuator on **frame_0**, which this row
  never perturbed; (d) actuators built from a *different* operator than `A`.
- **honest limit on the mechanism claim:** "pose damage tracks energy, not location" is measured
  on **one partition** (band vs interior) at **one α** (0.25), n600. Two points define a line only
  if you already believe it is a line. A third partition would test it; I did not run one.

## §6 What this does to the parent's other follow-ons

**FO-B (clipping-aware de-blur) is now dominated, and §4 is why.** FO-B's entire premise is that
solving for the best in-range camera field recovers the realisation lost to clipping. But its
best conceivable outcome is to realise **more** of the intended perturbation — i.e. to *increase*
`‖Δ‖`, which §3 measures as monotonically *increasing* pose drift. FO-B can only move the actuator
away from the break-even bar. The parent already gated FO-B on FO-A ("only worth building if FO-A
says the pose damage is escapable"); FO-A says it is not. **FO-B: do not build.**

**FO-C (fix the render, not its output) is the survivor, and it is the only one.** Its distinguishing
property is exactly the one every actuator in this family lacks: a render-side change puts frame_1
under a **joint** objective, so the pose term is *trained against* rather than *damaged blindly*.
Everything measured here is a statement about perturbing a finished frame; none of it transfers to
a frame that was shaped with pose in the loop. That is consistent with the standing vehicle law
that only joint descent crosses the photometric wall.

**The seg row FO-A said a LIVE branch would owe: NOT owed, but measured anyway — and it makes the
verdict worse, not closer.** The branch did not fire, and §4 shows no seg value could change the
sign. I ran it regardless, because the charter's literal positive control lives on the seg leg
and because the band-restricted lever's seg benefit is a real input to FO-C. Band-masked seg
ladder, n600, same frozen CPU SegNet and pre-registered bands:

| α | flips vs GT | Δ vs control | ΔS_seg |
|---:|---:|---:|---:|
| 0.00 | **34,938** | +0 | 0.000000 |
| **0.25** | **34,684** | **−254** | **−0.000215** |
| 0.50 | 35,281 | +343 | +0.000291 |
| 0.75 | 36,336 | +1,398 | +0.001185 |
| 1.00 | 37,816 | +2,878 | +0.002440 |

Same shape as the parent's unmasked ladder — non-monotone with an interior optimum at α = 0.25,
so the seg sub-verdict is again `INDETERMINATE_MIXED`, 0.75% of the round trip, 2.24% of the gap.
**The mask is strictly dominated on both axes at once:**

| at α = 0.25 | unmasked (parent) | **band-masked (this row)** |
|---|---:|---:|
| seg win | −634 flips, −0.000537 S | **−254 flips, −0.000215 S — only 40.1% retained** |
| pose cost, LOWER bound | +0.053035 S | **+0.040839 S — 82.5% retained** |
| **net, best case** | +0.052497 S | **+0.040623 S** |
| pose cost ÷ seg gain | 98.7× | **189.7×** |

Restricting the actuator to the band **throws away 60% of the seg benefit to buy back 18% of the
pose cost**, nearly doubling the cost/benefit ratio. The masked lever is 4.23× the whole remaining
gap, the wrong way. This is the same conclusion as §4 reached without it, arrived at from the
other side.

## §7 Apparatus landed with this row

- **One flag**, `--delta-mask {off,band,interior}`, default `off`, on
  `experiments/ddm_sr1_manufactured_seg_recovery.py`, consumed by `a1sign` **and** `a1pose`.
- **Both stages mask through ONE shared helper** (`_a1_apply_delta_mask`); a second inline copy in
  `stage_a1sign` was caught in self-review and removed, because a duplicate would drift from the
  helper the tests actually pin.
- **The flag fails closed where it cannot act.** `--delta-mask band` on `roperator` / `sign` /
  `emphasis` / `waterfill` / `ledger` is **refused**, not silently ignored — an inert flag is a
  config-orphan.
- **Masked runs cannot overwrite the parent's payloads.** Every artifact carries a `_bandmask` /
  `_interiormask` suffix; only `off` may claim the parent filenames.
- **The masked seg `verdict_scope` no longer claims the global actuator.** The pre-registered
  bands never move with the mask — the bar is the bar — but the label now says
  `band-restricted linear de-blur of A`.
- **A docs nit fixed:** the `SEG_DS_PER_FLIP` comment read `8.477116e-07`; the true value is
  `8.477105035e-07`. **No published number changes** — the constant is computed as
  `100.0 / SCORED_PX`, and it reproduces the parent's ΔS_seg to the last digit
  (`634 × SEG_DS_PER_FLIP = 0.0005374484592013889`, exactly the parent's §4 figure). Only the
  comment was wrong.
- **Tests:** `src/tac/tests/test_ddm_a1s_foa_band_masked_pose.py`, **36 tests** pinning the
  pre-registered thresholds, every adjudicator branch (including "at the bar is not below it"),
  the exact band/interior partition, the fail-closed refusals, the one-shared-helper invariant,
  and the structural default-off guard. The parent's 8 tests still pass.

## §8 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_sr1_manufactured_seg_recovery_20260816/foa/`
(VertigoDataTier holds 893 MiB free and is read-only — read from, never written to).

| artifact | bytes | sha256 (prefix) | what it is |
|---|---:|---|---|
| `a1sign/pose6_by_alpha.npy` | 144,128 | `97e2c89969b6ccc0…` | default-off re-run — **identical sha to the parent's retained payload**; the byte-identity proof |
| `a1sign/pose6_by_alpha_bandmask.npy` | 144,128 | `4c569efaddd2dfc3…` | (600,5,6) pose6, band-masked — **the FO-A primary** |
| `a1sign/pose6_by_alpha_interiormask.npy` | 144,128 | `6318bf8be6385a4d…` | (600,5,6) pose6, interior-masked — the complement |
| `SR1_A1POSE.json` | 3,951 | `d3a87f1bb5702302…` | off receipt |
| `SR1_A1POSE_BANDMASK.json` | 195,907 | `2c5db5383157fc8e…` | band receipt + 600 per-pair geometry rows |
| `SR1_A1POSE_INTERIORMASK.json` | 195,580 | `6f3c3b91e6bf25cf…` | interior receipt + 600 per-pair geometry rows |
| `FOA_VERDICT.json` | 7,304 | `6c2e3d84f03b6b83…` | the adjudicated record: byte-identity, all three pose legs, the derived bars |
| `SR1_A1SIGN_BANDMASK.json` | 464,386 | `e9d6fff3f777cd39…` | band-masked seg receipt: the positive control + the §6 ladder + 600 per-pair rows |
| `a1sign/A1SIGN_BANDMASK_PER_PAIR.jsonl` | 388,433 | `6e0eebd71a0ae795…` | per-pair seg journal (band px, clip, realisation, energy share) |
| `a1sign/delta_cam_pair33_bandmask_f32.npy` | 12,208,160 | `9030112e857125de…` | the band-masked camera perturbation itself, one sample pair |
| `a1sign/argmax_alpha_a0_bandmask.npy` | 117,964,928 | `2aeb1e6be0f7c6ab…` | α = 0 argmax — **identical sha to rt1's `argmax_base.npy`**: the charter's control |
| `a1sign/argmax_alpha_a0p25_bandmask.npy` | 117,964,928 | `e41a06f5ac5fc11b…` | α = 0.25 argmax (the band-masked seg best) |
| `a1sign/argmax_alpha_a0p5_bandmask.npy` | 117,964,928 | `269d8101192da822…` | α = 0.50 argmax |
| `a1sign/argmax_alpha_a0p75_bandmask.npy` | 117,964,928 | `4709c1209caaba38…` | α = 0.75 argmax |
| `a1sign/argmax_alpha_a1_bandmask.npy` | 117,964,928 | `30891c02cb021e74…` | α = 1.00 argmax |

Run custody: `/Volumes/APDataStore/pact/ddm_a1s_foa_20260817/run/` — `launch_manifest.json`
(git HEAD `fde2290364c42b46e4a446e98d8be90a8d4bdd55`, tool sha at launch, python/platform, the
thread pin, every input sha), the four leg scripts, four full logs,
`pre_registered_prediction.json`, and `smoke_certify_or_block.json`.

**Consumed unmodified, every sha verified BEFORE use** (all seven match the parent's §9):

| input | bytes | sha256 |
|---|---:|---|
| wc1 `0.raw` | 3,662,409,600 | `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` |
| qs3 `gt_argmax_n600.npy` | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` |
| sr1 `A_row_384x384.npy` | 1,179,776 | `d884e8ecb9abea65ecb59c20f6ed2dc00b8e54d4e2da2323ac2221aac7621e82` |
| sr1 `A_col_512x512.npy` | 2,097,280 | `1a0fd4c49d253367c7fc4b401ef6f383fb90aed936b838a2206eb013830d80f0` |
| rt1 `argmax_base.npy` | 117,964,928 | `2aeb1e6be0f7c6ab8191b790204d8df0ae5fdce7ef2ecc5b2d18a715f1a674c4` |
| hv1 ep0634 `decoded_spatial_tokens.rc64.bin` | 117,964,800 | `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` |
| a1s `pose6_by_alpha.npy` (identity reference) | 144,128 | `97e2c89969b6ccc0d3ad612d14714f009824d35698cb1f469311b69af34b7a70` |

**Certified and removed:** a 4-pair flag smoke (8 files, 3,296,224 B), certified rebuildable in
`run/smoke_certify_or_block.json` with its rebuild command. Per [[m96]] a 4-pair prefix is a
different population — **no number from it is cited anywhere in this memo.**

## §9 What I did NOT establish

- **No exact-eval row.** Nothing here is a score. The pointer is unmoved and this axis is
  `[macOS-CPU advisory]`.
- **No third partition.** The "energy not location" mechanism (§ANSWER, item 2) rests on the
  band/interior split at α = 0.25 only.
- **No frame_0 measurement.** Every leg perturbed frame_1 alone.
- **No render-side evidence.** FO-C is untouched by this row; §6 argues it survives on a
  structural distinction, which is an argument, not a measurement.
- **No mechanism for the −0.4828 cancellation.** I measured that the band and interior pose
  responses partially oppose; I did not establish why. Plausible candidates I did not separate:
  PoseNet's own spatial pooling, the YUV6 preprocess, or the sign structure of `A⁻¹`.
- **No repeat run of any leg.** Each n600 leg was measured once. The instrument is deterministic
  by construction (frozen weights, fixed thread count, fixed batch shape, seeded nothing), and the
  α = 0 rungs reproduce the parent bit-for-bit on both axes, but I did not run a determinism
  repeat of an α > 0 rung.

**My own pre-registered prediction, recorded before the number landed** (`run/pre_registered_prediction.json`):
band-only drift in **[0.013123, 0.014863]**, bucket **FAMILY_CLOSED**. The bucket was right; the
value came in at **0.018162, 22–38% above my range.** I under-predicted the band's share because I
extrapolated from the 4-pair smoke's band/full ratio (0.675) and the isotropic-Jacobian estimate
(0.596); the true n600 ratio is **0.825**. Recording the miss, not just the hit.

**Own-vehicle frontier: hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]` —
UNMOVED by this unit.**
