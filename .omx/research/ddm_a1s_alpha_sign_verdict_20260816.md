---
arm: ddm_a1s
title: "FO-1 answered on both axes: the zero-byte de-blur IS favourable on seg at low strength (-634 flips = 5.6% of the gap at alpha 0.25, zero archive bytes) but frame_1 feeds PoseNet, and the same edit moves the pose vector 8.39x the entire incumbent pose error -- so d_pose rises at least 54.6x and A1 costs at least +0.052497 S, 98.7x more than it wins. CLOSED as a net loss at every alpha. The order's literal FO-1 formula was separately refuted scorer-free, before any SegNet second was spent: it puts A.x at the scorer at alpha 0 (14.98 levels off the shipped decode) and the status quo at alpha 1, so its ladder never reaches the de-blur"
utc: 2026-08-16
parent: ".omx/research/ddm_sr1_manufactured_seg_recovery_20260816.md"
fire_order: "sr1 FO-1"
axis: "[macOS-CPU advisory] -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "INSTANCE on the hv1 ep0634 vehicle; family verdicts only where named"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_a1s — FO-1: the sign of the zero-byte de-blur

STORES CONSULTED: the sealed order `ddm_sr1_manufactured_seg_recovery_20260816.md` §FO-1, read
verbatim before any code was written, plus its retained receipts · the parent
`ddm_rt1_seg_roundtrip_decomposition_20260816.md` and `RT1_INSTRUMENT_CHECK.json` ·
`ddm_wc1_decode_wallclock_verdict_20260816.md` (the retained decode) · the LIVE decoder
`experiments/results/public_pr130_intake_20260725_fable/source/submissions/
semantic-pose-HPAC_CPR1/inflate.py::render_video` · `upstream/modules.py` — **both** SegNet and
PoseNet `preprocess_input`, read at source · `upstream/frame_utils.py` (the canonical sizes) ·
`ddm_wc2_hpac_mps_port_20260814.md` (the hv1 `d_pose`) · `ddm_rx1_rate_representation_attack_
20260814.md` (the hv1 rate row) · memories [[m88]] [[m96]] (a prefix is a different population),
[[m91]] (seg is one graph with one hub), `et4` (batch shape is part of the forward instrument),
and `the_counted_byte_is_not_fungible_placement_beats_amount_20260816`.

## ANSWER FIRST

**The positive control passed in its strongest possible form.** α = 0 gave **34,938 flips —
exactly** the required count, and the α = 0 argmax field is **byte-identical to rt1's
`argmax_base.npy`** (sha `2aeb1e6be0f7…`, all 600 frames). The synthesised camera frame was
asserted bit-identical to the shipped frame at every one of the 600 pairs. The row is admissible.

**On seg alone, the de-blur is real and favourably signed at low strength — but no pre-registered
band fired.** Best is α = 0.25 at **34,304 flips, −634 vs the control**: 1.88% of the manufactured
round trip, **5.60% of the gap**, at **zero archive bytes**. That is 2.66× short of the LIVE bar
(1,687 flips), while α = 0.75 and α = 1.00 land above the neutral band. The response is
**non-monotone with an interior optimum near α ≈ 0.25** — a shape none of the three pre-registered
bands describes, so the seg verdict is `INDETERMINATE_MIXED` and I am reporting it as shaped
rather than forcing it into a bucket.

**The pose leg closes it anyway, decisively, and this is the finding.** `PoseNet.preprocess_input`
keeps **both** frames of the pair, so an actuator that edits frame_1 moves `d_pose` too — sr1's
§3 treats frame_0 as "the pose carrier only" and prices A1 as a free seg lever. It is not. At
α = 0.25 the pose 6-vector moves **rms 0.022019**, which is **8.39× the entire incumbent pose
error** (`√d_pose = 0.0026240`). By the reverse triangle inequality — no GT needed — `d_pose`
must rise to at least **3.7617e-04, 54.6× the incumbent 6.8856e-06**:

| at α = 0.25, n600 | value |
|---|---:|
| seg gain | **−0.000537 S** |
| pose cost, GT-free LOWER bound | **+0.053035 S** |
| **net, the best case the geometry permits** | **+0.052497 S** |
| as a multiple of the −0.0095973 gap | **5.5× — the wrong way** |
| pose cost ÷ seg gain | **98.7×** |

**A1 is CLOSED: a net S loss at every α > 0, by two orders of magnitude.** The zero-byte de-blur
buys 634 seg flips and pays for them ninety-nine times over on an axis FO-1 did not count.

**Separately, the order's literal FO-1 formula could not have answered the question, and I
established that for $0 before spending one SegNet second** (§2). Pointer UNMOVED, no dispatch,
no Modal, `[macOS-CPU advisory]` throughout.

## §1 The positive control — bit-exact, and it passed

The order made one thing a hard gate: α = 0 must reproduce rt1's **34,938** flips exactly, and a
miss invalidates the row. Every control passed.

| control | required | measured | verdict |
|---|---|---|---|
| α = 0 flips vs GT, n600 | exactly 34,938 | **34,938** | **PASS** |
| α = 0 argmax field vs rt1 `argmax_base.npy` | — | **bit-identical**, sha `2aeb1e6be0f7…` | **PASS (stronger than asked)** |
| synthesised camera frame at α = 0 vs shipped frame | — | **bit-identical, asserted all 600 pairs** | **PASS** |
| rebuilt `d@u` vs sr1's retained `A_row`/`A_col` | exact | **max abs diff 0.000e+00**, both axes | **PASS** |
| separable operator vs the real `F.interpolate` chain | rel < 1e-9 | **3.366e-16** (reproduces sr1) | **PASS** |
| `A⁻¹A − I` residual, row / col | — | 1.788e-06 / 1.790e-06 | recorded |
| input custody: `0.raw` | `e5539653…`, 3,662,409,600 B | **matches** | **PASS** |
| input custody: `gt_argmax_n600.npy` | `91d3ff11…` | **matches** | **PASS** |
| input custody: `A_row` / `A_col` | `d884e8ec…` / `1a0fd4c4…` | **match** | **PASS** |

Two of these deserve a word.

**The custody tie is exact.** sr1 retained only the SQUARE composite `A` (384×384 and 512×512).
FO-1 needs `D(cam)`, and `D` maps 874 → 384, so `A` cannot express it — the order's input list
("`x = D(cam_f1)` using the retained `A_row/A_col`") is not executable as written. I rebuilt all
four one-axis operators from sr1's own deterministic `resample_matrix`, then required `d @ u` to
equal the retained matrices. It does, at **max abs diff 0.000e+00 on both axes**. The rebuild is
sr1's operator, not a lookalike.

**The scorer is rt1's scorer, not a copy of it.** The stage imports `SegInstrument` and
`nn_lift_index` directly from `experiments/ddm_rt1_seg_roundtrip_decomposition.py`, because the
FO-1 pin is that this leg differs from rt1's base leg *only in the camera frame*. Batch = 1 pair,
`torch.set_num_threads(8)`, `SegNet.preprocess_input` verbatim, frozen CPU torch SegNet from
`upstream/models/segnet.safetensors`. The thread pin is enforced in code (`A1_PIN_THREADS`) and
the stage refuses to start if violated.

## §2 The order's literal formula measures the wrong side — refuted scorer-free, then corrected

**The FO-1 text cannot pass its own gate.** It says: `x = D(cam_f1)`; `xs = x + α(A⁻¹x − x)`;
`cam' = round(clamp(U(xs)))`. But the scorer's first act is to apply `D`, and `D∘U = A` — the
entire finding sr1 built. So the literal form hands the scorer `A·xs`, not `xs`:

| α | what the LITERAL form puts at the scorer | measured band deviation from the shipped `x` |
|---:|---|---:|
| 0 | `A·x` — one EXTRA blur | **14.98 levels** (reference `‖Ax − x‖` = 14.97) |
| 1 | `x` — the status quo, exactly | **0.06 levels** |

Its ladder runs from *double-blurred* to *baseline* and never reaches the de-blur; its α = 0
control would have missed by 15 RGB levels and the row would have been declared invalid. This is
a transcription slip, not a defect in sr1's physics: its §5 prose ("the actuator writes
`U(A⁻¹m)`") is right, and `A⁻¹·D(cam) = m_est` is the *shipped* frame reconstructed, not the
post-fix one.

**The correction, preserving every pre-registered element.** The actuator is
`m ← m + α(A⁻¹m − m)` on the renderer master and the decoder writes `round(clamp(U(m_α)))`.
Written as a perturbation of the retained frame:

```
m_est   = A⁻¹ x                                   (x = D(cam); m_est recovers m)
Δ_cam   = U(A⁻¹ m_est − m_est)
cam'(α) = round(clamp(cam + α·Δ_cam, 0, 255))      →  the scorer sees x + α(A⁻¹x − x) = xs
```

because `D(cam + α·Δ_cam) = x + α·A(A⁻¹m_est − m_est) = x + α(A⁻¹x − x)`. Same object `xs`, same
ladder, same gate value, same bands. Three things improve:

| property | order's literal form | the delta form (used) |
|---|---|---|
| scorer input at α = 0 | `A x` — 14.98 levels off | `x` — **0.0 levels** |
| camera frame at α = 0 | not bit-identical to `cam` | **bit-identical, all 600 pairs** |
| vs a fresh `round(clamp(U(A⁻¹m_est)))` render | — | agrees to **0.002 levels** |

That last row **retires the order's stated pessimism caveat.** FO-1 warned that `A⁻¹·D(cam)`
"inherits one round of uint8 noise the real decoder-side fix would not," so a CLOSED reading would
need re-checking against true renderer output. The delta form does not inherit that round: before
the uint8 stage it reproduces the intended `xs` to **0.0001 RGB levels at the band** (against a
21.70-level signal), and after it, it matches a fresh render of the same actuator to 0.002 levels.
The realisation is faithful, not pessimistic, so the verdict below does **not** carry the re-check
obligation the order attached to it. What it does carry is §5.1.

## §3 The ladder — n600, frozen CPU SegNet, `[macOS-CPU advisory]`

| α | flips vs GT | Δ vs control | seg S | flips vs label | band clip % | realised % of intent |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | **34,938** | +0 | 0.029617 | 33,743 | 0.00 | 100.00 |
| **0.25** | **34,304** | **−634** | 0.029080 | 33,103 | 6.27 | 92.70 |
| 0.50 | 34,662 | −276 | 0.029383 | 33,476 | 11.75 | 91.66 |
| 0.75 | 35,603 | +665 | 0.030181 | 34,434 | 18.95 | 88.43 |
| 1.00 | 36,947 | +2,009 | 0.031320 | 35,802 | 24.88 | 84.21 |

Bands: LIVE < 33,251 · neutral [34,589, 35,287] · harmful > 35,287. Clip shares are n600 means;
"realised % of intent" is the band-mean fraction of the intended de-blur that survives to the
scorer, over 12 **seeded-random** pairs (`[33, 66, 81, 89, 280, 299, 322, 353, 410, 438, 474,
538]` — never a prefix, per [[m88]]/[[m96]]).

**The shape has a measured mechanism.** The intent grows linearly in α (5.64 → 22.56 levels at
the band) but the realisation degrades, because clipping bites exactly where the axis lives:
band camera-pixel clipping rises 0 → 6.27 → 11.75 → 18.95 → 24.88%, against ~0.9% off the band.
The optimum near α ≈ 0.25 is where marginal de-blur benefit meets marginal clipping damage. This
is measured, not narrated: the realisation residual at α = 1 is 3.56 levels, of which 3.13 is
clipping and 0.12 is rounding.

**Where the seg flips move**, α = 0 → 0.25, charged to the GT class:

| class | α = 0 | α = 0.25 | Δ |
|---|---:|---:|---:|
| Road | 13,786 | 13,081 | **−705** |
| Lane | 8,712 | 8,206 | **−506** |
| Undrivable | 6,297 | 6,887 | +590 |
| Movable | 4,750 | 4,674 | −76 |
| MyCar | 1,393 | 1,456 | +63 |

The de-blur wins on the Road↔Lane hub — the edge [[m91]] and rt1 both name — and loses on
Undrivable (sky) and MyCar (the static ego hood). That is a coherent picture: sharpening helps
where the argmax boundary is a real image edge and hurts where the region is flat and large.

## §4 The pose leg — the axis FO-1 did not count, and the one that decides

`PoseNet.preprocess_input` (`upstream/modules.py:69-73`) rearranges the **whole pair**
`b t c h w -> (b t) c h w`, so its 12 input channels are 6 from frame_0 **and 6 from frame_1**.
MEASURED directly — perturb frame_1 only by +5 levels and read the per-channel change:

```
PoseNet input channels 0-5  (frame_0): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PoseNet input channels 6-11 (frame_1): [5.0, 5.0, 5.0, 5.0, 0.0, 0.0]
```

(the two zeros in the frame_1 half are chroma, which a uniform RGB shift cannot move). **A1 edits
frame_1, therefore A1 moves `d_pose`.**

So I measured it, n600, on the SAME synthesised frames, with the frozen CPU PoseNet
(`upstream/models/posenet.safetensors`, batch = 1 pair, 8 threads, first 6 pose dims per
`compute_distortion`). The primary result is **GT-FREE**: with `p_α = p_0 + δ`,
`√d_pose(α)` lies within `rms‖δ‖` of `√d_pose(0)`, so the incumbent `d_pose` plus the measured
drift bounds the rise in **both** directions with no GT target at all.

| α | pose drift rms | drift ÷ incumbent pose rms | `d_pose` LOWER bound | ΔS_pose LOWER bound | ΔS_seg | **net, best case** |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000000 | — | 6.886e-06 | 0 | 0 | 0 |
| **0.25** | **0.022019** | **8.39×** | **3.762e-04 (54.6×)** | **+0.053035** | −0.000537 | **+0.052497** |
| 0.50 | 0.050930 | 19.41× | 2.334e-03 (339×) | +0.144460 | −0.000234 | +0.144226 |
| 0.75 | 0.085918 | 32.74× | 6.938e-03 | +0.255099 | +0.000564 | +0.255663 |
| 1.00 | 0.128471 | 48.96× | 1.584e-02 | +0.389666 | +0.001703 | +0.391369 |

Reference: `d_pose(hv1 ep0634) = 6.885642960696714e-06`, so `√d_pose = 0.0026240` and the pose
contribution is `√(10·d_pose) = 0.008298`. The **entire** incumbent pose error is 0.0026240; at
α = 0.25 the actuator moves the pose vector **8.39× that much**. There is no direction the drift
could point that rescues this — the lower bound is what the table reports.

A secondary read against a cached GT PoseNet target agrees on the *rise* (α = 0.25:
6.269e-04 measured vs its own α = 0 baseline of 1.475e-04, a rise of 4.79e-04, bracketing the
GT-free lower bound of 3.762e-04) but its **absolute baseline is 21× the authoritative hv1
`d_pose`**, which is exactly the §6 hazard. That is why the GT-free bound is primary here and the
cached number is labelled advisory-secondary.

## §5 Verdict

**A1 — the zero-byte decoder-side de-blur `m ← m + α(A⁻¹m − m)` — is CLOSED as a net S loss.**

- **verdict:** CLOSED_NET_LOSS.
- **verdict_scope:** FORMULATION — the *global linear* de-blur of `A`, applied uniformly to
  frame_1, on the hv1 ep0634 vehicle. Measured n600 on **both** scored axes.
- **the number:** at its own best strength the actuator wins 634 seg flips (−0.000537 S) and pays
  **at least +0.053035 S** on pose. Net **+0.052497 S at best — 5.5× the whole gap, the wrong
  way.** Cost/benefit **98.7×**.
- **the seg-only sub-verdict**, recorded separately because the order pre-registered it:
  `INDETERMINATE_MIXED` — favourably signed and non-monotone, best −634 flips at α = 0.25, which
  is 2.66× short of the LIVE bar and outside the neutral band in both directions. No
  pre-registered band fired and I did not force one.

**What is NOT closed.** The order's own framing — "a zero-byte actuator has no rate side, so any
recovery is pure profit" — is what this row refutes. The profit was not free; it was charged to
pose. Three things survive and are named in §8: a **pose-null-constrained** de-blur, a
**clipping-aware** one, and moving the fix into the **renderer** instead of post-processing its
output. None is tested here.

**The order's pessimism caveat does not apply** (§2): the delta form matches a fresh render of the
same actuator to 0.002 levels, so this CLOSED reading needs no re-check against true renderer
output.

## §6 Two corrections to sr1 that hold regardless of the verdict

### 6.1 sr1's clipping feasibility number priced the wrong field — the real figure is ~12× larger and sits on the band

sr1 §3 reports "Clipping outside [0,255] touches **0.0431%** of pixels" as A1's feasibility line.
That is computed on `m_est = A⁻¹x` — the field the scorer should *see*. But pre-compensation does
not write what the scorer should see; it writes a field sharp enough that `A` blurs it back into
that. The decoder must write `U(A⁻¹m)`, which is **twice** deconvolved. Measured on sr1's own
seeded pairs:

| field | what it is | out-of-range share |
|---|---|---:|
| `m_est = A⁻¹x`, scorer res | what sr1 priced; what the scorer should SEE | **0.035–0.047%** (reproduces sr1's 0.0431%) |
| `A⁻¹ m_est`, scorer res | what the actuator must WRITE | **1.49–1.85%** (~40×) |
| `U(A⁻¹ m_est)`, camera res | what the decoder actually writes | **0.48–0.54%** (~12×) |

And the clipping is concentrated where the axis lives: at α = 1, **24.9% of BAND camera pixels
clip** against ~0.9% off the band. This is a physical bound on A1, not an instrument artifact —
the real decoder must also write uint8 in [0,255] — and it is the measured cause of the
non-monotone ladder in §3.

### 6.2 A1 is not a pure-seg lever

Covered in §4. sr1 §3's "frame_0 is the 12-dimensional pose carrier only" is right that frame_0
carries the pose *signal*; it is incomplete in the way that decides this row, because PoseNet
reads the pair. Any future frame_1 actuator on this vehicle owes a pose leg before a ΔS claim.
The marginal is `5/√(10·d_pose) = 602.6 per unit d_pose`, so a `d_pose` rise of just **+34.5%
relative** erases a 5%-of-round-trip seg win. The measured rise at the best α is **+5,363%**.

## §7 A caveat for whoever reuses a GT cache: two GT argmax fields differ by 1.8× the whole gap

The 2026-06-10 `lever_b` targets directory holds a `gt_segnet_argmax.u8` at exactly 117,964,800
bytes. Against the qs3 `gt_argmax_n600.npy` this unit used (sha `91d3ff11…`):

**agreement 117,944,127 / 117,964,800 = 99.982475% — they differ on 20,673 pixels.**

That is 0.017525 S units: **59.2% of the entire scored seg term** and **1.83× the whole
−0.0095973 gap**. Two independently built GT references for the same video are not
interchangeable. My row is unaffected — every leg, including the α = 0 control and rt1's base,
reads the single qs3 GT, so leg-to-leg differences carry no GT term, which is exactly what the
pinned instrument buys. The sister pose cache in that directory shows the same hazard
quantitatively (§4: its α = 0 `d_pose` baseline is 21× the authoritative hv1 value). I did not
establish the mechanism — tie-breaking under a different batch shape per `et4`, a different decode
path, or a different thread count are all live candidates — and I am not claiming one.

## §8 What this unit did NOT establish, and the follow-ons

**Not established.**
- **No exact-eval row.** Advisory α = 0 reads 34,938 against the contest-CUDA 34,930.6, ratio
  1.000213; that offset is rt1's, inherited unchanged. Nothing here is a score.
- **No mechanism for the GT-cache disagreement** in §7 — only its size and the rule that follows.
- **No test of any non-uniform de-blur.** The ladder tests `A⁻¹` applied globally. §8's follow-ons
  are different actuators, not this one at another α.
- **No claim that the pose damage is irreducible** — only that the *global* form incurs it.

**Follow-ons, in the order I would fire them.**

1. **FO-A — is the pose damage band-driven or interior-driven?** Mask `Δ_cam` to the lifted label
   band and re-run the α = 0.25 pose leg, n600. The band is ~2.2% of pixels but carries the 22.5-
   level perturbation; the interior is 97.8% of pixels at ~1.8 levels. Pre-registered threshold:
   **if band-only drift rms < 0.0026240** (the incumbent pose error) the pose-null branch is LIVE
   and owes its own seg row; **if ≥ 0.0083** (3.2×, enough to erase any plausible seg win) the
   whole post-hoc de-blur family is CLOSED, not just this formulation. One stage flag on the
   committed tool, ~15 min, $0. This is the single measurement that turns a FORMULATION verdict
   into a FAMILY verdict.
2. **FO-B — clipping-aware de-blur.** §6.1 shows 24.9% of band camera pixels clip at α = 1 and
   that clipping is the entire realisation residual. Solving for the best in-range camera field
   (a projection onto `[0,255]^n`, not a naive clamp) is a strictly better actuator. It is only
   worth building if FO-A says the pose damage is escapable.
3. **FO-C — fix the render, not its output.** hg1 measured that hv1's token trainer contains no
   SegNet at all, so the round trip is renderer realisation error. Training the renderer against
   `A` in the loop targets the same 33,743 flips without a post-hoc perturbation, and a
   render-side fix changes frame_1 in a way the pose term can be trained against jointly rather
   than damaged blindly.

**Not a follow-on:** more α resolution on the global de-blur. §4 makes the sign of the total
unambiguous at every α on the ladder, and the pose cost grows monotonically in α while the seg
gain peaks at 0.25 and reverses. There is no α where this actuator pays.

## §9 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_sr1_manufactured_seg_recovery_20260816/a1sign/`
(VertigoDataTier is full and READ-ONLY; APDataStore had 230 GiB free).

| artifact | bytes | sha256 (prefix) | what it is |
|---|---:|---|---|
| `argmax_alpha_a0.npy` | 117,964,928 | `2aeb1e6be0f7…` | α = 0 argmax — **identical sha to rt1's `argmax_base.npy`**, the control |
| `argmax_alpha_a0p25.npy` | 117,964,928 | `4cba58b4f7eb…` | α = 0.25 argmax (the seg-best leg) |
| `argmax_alpha_a0p5.npy` | 117,964,928 | `1e98ba792ce3…` | α = 0.50 argmax |
| `argmax_alpha_a0p75.npy` | 117,964,928 | `baf24c4c907f…` | α = 0.75 argmax |
| `argmax_alpha_a1.npy` | 117,964,928 | `de3f497f0f50…` | α = 1.00 argmax |
| `pose6_by_alpha.npy` | 144,128 | `97e2c89969b6…` | (600, 5, 6) PoseNet scored dims, every pair × every α |
| `delta_cam_pair33_f32.npy` | 12,208,160 | recorded in receipt | the actuator's camera perturbation, one sample pair |
| `A1SIGN_PER_PAIR.jsonl` | — | — | per-pair journal (band px, clip and realisation diagnostics), 600 rows |

Receipts: `SR1_A1SIGN.json` (seg ladder + controls + per-pair rows) · `SR1_A1POSE.json` (pose
ladder + the GT-free bounds). Run custody: `/Volumes/APDataStore/pact/
ddm_a1s_alpha_sign_20260816/{run,pose_run}/` (launch manifest + full log).
Tool: `experiments/ddm_sr1_manufactured_seg_recovery.py`, stages `a1sign` and `a1pose`.
Tests: `src/tac/tests/test_ddm_a1s_alpha_sign_verdict.py` (8 tests pinning the pre-registered
bands and every verdict branch).
Consumed unmodified: wc1 `0.raw` (`e5539653…`, 3,662,409,600 B), qs3 `gt_argmax_n600.npy`
(`91d3ff11…`), sr1 `A_row_384x384.npy` (`d884e8ec…`) and `A_col_512x512.npy` (`1a0fd4c4…`),
rt1 `argmax_base.npy` (`2aeb1e6b…`), hv1 ep0634 `decoded_spatial_tokens.rc64.bin`.

**Own-vehicle frontier: hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]` —
UNMOVED by this unit.**
