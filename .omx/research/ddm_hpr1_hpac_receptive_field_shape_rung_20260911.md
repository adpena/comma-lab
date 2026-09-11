# ddm_hpr1 — the HPAC prior's RECEPTIVE-FIELD SHAPE, priced on move 45

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false ·
`# FORMALIZATION_PENDING: the shape statistic is a conditional-information RANKING, not a coder charge; it is registered as a canonical equation only when a real byte price confirms or falsifies its sign, per the standing "first-order token price is a ranking not a charge" law.`

**STATE (Opus, 2026-09-11) — COMPLETE.** All four deliverables are measured. The headline is not the
one the charter expected: **the pre-registered shape rung is FALSIFIED by the real coder (+812 B
against its own control), and the experiment's CONTROL is a −887 B candidate with no receiver change**
(§4e). One step is owed before any seal — the cold public parse-back — and §5 says exactly why I
stopped rather than stamp a false label into a receipt. No frontier move, no Modal dispatch, no scorer
run, no seal, no candidate claim. This file is crash-resumable.

Frontier line: `composition S 0.1371383667388406 @ 180,246 B [contest-CUDA T4 n600] (move 45)` — UNMOVED
by this arm.

---

## 1. The shipped receptive field, reconstructed from the receiver — MEASURED

Producer `experiments/ddm_hpr1_surprise_atlas.py reconstruct`; receipt
`probe/RECEPTIVE_FIELD.json`. Bound to archive `145e02e21f9a1cbc…` @ 180,246 B, IHS1 body
`817281908d993d89…` @ 17,770 B, hpac member 11,911 B.

The prior is `cpr1.hpac_integer.IntegerHPAC(patch=64, delta=2, channels=64, frame_dim=8,
num_classes=5)`. It reads **four** neighbourhoods, and every one of them is defined by
`patch_group_mask(kernel, delta, type)` plus a dilation:

| module | kernel | dilation | taps | reads | causal rule | stored values |
|---|---:|---:|---:|---|---|---:|
| `conv_a` | 7 | 1 | **23** of 49 | CURRENT plane, one-hot + 2 coord channels, inside the 64×64 patch | type A: `col−c + 2(row−c) < 0` (strictly before) | 64 × 7 × 23 = 10,304 |
| `conv_b1` | 5 | 2 | 14 of 25 | hidden features, depthwise | type B: `≤ 0` (same group allowed) | 64 × 14 = 896 |
| `conv_b2` | 3 | 4 | 5 of 9 | hidden features, depthwise | type B | 64 × 5 = 320 |
| `conv_past` | 3 | 1 | **9 of 9, dense** | **PREVIOUS plane**, one-hot, NO patch restriction | none — the previous plane is fully known | 64 × 5 × 9 = 2,880 |

plus `spm_dw`/`spm_pw` (a 64×64-patch MEAN of `past`, i.e. a patch-level temporal summary), `head`,
and `frame_shift`/`frame_scale` (FiLM from the 8-dim frame embedding). Total stored weight values
20,416, packed to 11,155.1 B at the shipped depths; `conv_a` alone is 50.5 % of the values and
6,238.8 B of the pack.

**Two structural facts that decide how a shape rung can be built and sealed:**

1. **Not one shape bit ships in `archive.zip`.** The masks are `register_buffer(..., persistent=False)`;
   they are reconstructed at decode time from `HPAC_PATCH`, `HPAC_DELTA` and the three hard-coded
   kernel/dilation constants in `cpr1/inflate.py`. The archive carries only weights. **Every shape
   rung is therefore a RECEIVER CHANGE and routes to the first-measurement chain — none can take the
   normal seal path.** cl2's "must change the SHAPE, not the size" is, in the shipped object's terms,
   "must change constants in `inflate.py`".
2. **The serializer stores only masked taps** (`integer_model_io._weight_rows` returns
   `weight[index][mask[index]]`). So tap COUNT is model bytes and tap POSITION is free. A rung that
   MOVES taps without adding any holds the model leg of cl2's secant at ≈ 0 by construction, which is
   exactly the regime cl2's +0.446 slope does not govern.

3. **A shape rung is a receiver change in THREE places, not one, and the third is a silent-divergence
   hazard.** `cpr1/inflate.py` carries the constants; `runtime/f26_hpac_native.c` RE-IMPLEMENTS
   `conv_past` in C with the dilation-1 stencil baked into its source-coordinate arithmetic
   (`source_row = global_row + kernel_row - 1`, ~lines 448/455); and the two are selected at run time
   by `F26_TOKEN_DECODER` (shipped default `python`, `native-hpac` opt-in). A rung that patched only
   `inflate.py` would decode one field in the default mode and a DIFFERENT field in native mode — an
   archive that is output-lossless on one decoder and wrong on the other, with nothing to announce it.
   This arm's producer therefore moves the constant in both files or refuses on a non-unique anchor,
   and records both before/after by sha. **The hazard is pre-existing and applies to any future
   geometry rung on this object, not only to this one.**

**Rule 118.** A dilation, a kernel size and `delta` are geometric parameters of the same kind as the
constants the receiver already ships as code. A tap-window CENTRE OFFSET fitted to this video's
ego-motion is not: it is a fitted value and would be content. This arm therefore restricts its
fireable rungs to geometric parameters and records the offset family as measured-but-refused. (The
measurement below independently prefers the clean option, so nothing is lost by the restriction.)

---

## 2. Pre-registered rungs and their predicted signs — MEASURED statistic, ranking only

Statistic: on the REAL shipped 600-frame field (`a92e7d90…`, 112,008,208 usable observations), with a
fixed causal base set B = {(0,−1), (−1,0), (−1,1), (−1,−1)} (four current-plane taps, `H(X|B) =
0.016602` bits/symbol), measure the conditional mutual information a candidate tap SET adds, and the
two-part MDL code length that charges its extra context cells. Every candidate holds the SAME number
of positions, so capacity is held and only shape varies. **This is a ranking, never a charge.**

### 2a. Why a set statistic and not a tap atlas

The full 17×17 temporal displacement atlas (`probe/ATLAS_past_r8.json`) shows a broad ANISOTROPIC
plateau leaning toward +dy/+dx. Its single-tap peak is (+1,+2) at 2.898 milli-bits/symbol; the best
tap inside the shipped 3×3 window is (+1,+1) at 2.843. **On single taps the shipped window is 98.1 %
of the best available anywhere — a single-tap reading closes this door.** It is the wrong reading: the
shipped window's nine taps are mutually REDUNDANT (adjacent tokens in a piecewise-constant field are
usually equal), and a spread window buys non-redundant information. Only a joint set statistic sees
that. This is the arm's methodological receipt.

### 2b. The window table (geometry-blind rule: each 3×3 window reduced to its own best four taps)

`probe/WINDOW_ATLAS_past.json`.

| rung | window | best-4 taps | CMI (mbits/sym) | MDL net (B, rank) | Δ vs shipped (B, rank) | predicted sign | rule 118 |
|---|---|---|---:|---:|---:|---|---|
| **R1** | `d2_c00` — conv_past dilation 1→**2**, centre held | (0,2),(2,−2),(2,0),(2,2) | **3.7696** | **21,232** | **−3,625** | **NEGATIVE (tail falls)** | clean (geometric) |
| **R2** | `d3_c00` — dilation 1→3, centre held | (0,3),(3,−3),(3,0),(3,3) | 3.7237 | 20,780 | −2,983 | NEGATIVE, weaker | clean (geometric) |
| R3 | `d1_c11_offset` — centre shifted to (+1,+1) | (0,2),(1,0),(1,1),(1,2) | 3.6634 | 19,826 | −2,138 | NEGATIVE, weaker | **CONTENT** (fitted offset) |
| R4 | `d2_c11_offset` | (1,−1),(1,1),(1,3),(3,3) | 3.5915 | 17,158 | −1,132 | NEGATIVE, weakest | **CONTENT** |
| — | `shipped_d1_c00` (the incumbent) | (0,1),(1,−1),(1,0),(1,1) | 3.5107 | 17,220 | 0 | — | shipped |
| R5 | `d3_c22_offset` | (2,−1),(2,2),(2,5),(5,5) | 3.1517 | 13,169 | **+5,025** | **POSITIVE (refused)** | CONTENT |

**Pre-registered order and the pre-registered falsifier:** R1 > R2 > R3 > R4 > shipped > R5. If the
exact byte price does not order R1 ahead of the shipped prior, the statistic is falsified as a rung
predictor on this object and the shape family closes at the measured level.

**The finding the table carries:** the shipped temporal window is too COMPACT, not mis-centred. Pure
dilation beats every fitted offset, and the best-ranked rung is also the only rule-118-clean one.

### 2c. Spatial rungs (current plane), same statistic

`probe/SET_ATLAS_current.json`, four causal taps beyond B, varying only the dilation of the second
ring:

| set | CMI (mbits/sym) | Δ vs dilation 1 (B, rank) |
|---|---:|---:|
| `a_mixed_1_2` | 1.7695 | −6,740 |
| `a_dil3_ring2` | 1.6578 | −5,220 |
| `a_dil2_ring2` | 1.5172 | −3,309 |
| `a_dil1_ring2` (shipped spacing) | 1.2737 | 0 |
| `a_dil1_far3` | 1.1099 | +2,226 |

Same sign as the temporal axis: at held tap count, a SPREAD cone carries more. Two differences from
the temporal case, both measured: (i) the MDL net of every spatial set is NEGATIVE (the four base taps
already carry most spatial information, so extra spatial context does not repay its parameters),
whereas every temporal set's net is POSITIVE — **the unexploited information is on the temporal axis**;
(ii) `conv_a` reaches its taps through geometry-general code, so a spatial rung is CHEAPER than a
temporal one, not dearer. **I first recorded the opposite and it was wrong — the correction belongs in
the record.** `cpr1/hpac_integer_sparse._conv_a` does hard-code `F.pad(inputs, (3,3,3,3))` and a `+3`
offset, i.e. dilation-1 reach, but `residual_archive.py:679` calls `optimize_sparse_evaluator(sparse)`
UNCONDITIONALLY before the decode loop, and that rebinds `selected_logits` to
`hpac_inference._selected_logits`, whose `_conv_a_features` builds its gather from `sparse.a_offsets`
with a general bounds check and no padding. The native export is general too
(`f26_hpac_native.c:957`: `hidden_row = source_row - model->a_offsets[tap*2]`, bounds-checked, with
the offsets read off the module). The pad-3 routine is dead in the shipped path. **R6 (`conv_a`
dilation) therefore needs ONE edit — set `model.conv_a.dilation` in `cpr1/inflate.py` — and nothing
in C. It is pre-registered with a NEGATIVE predicted sign and left unfired by this arm for
wall-clock, and it is the cheapest unfired rung on the board.**

`delta` (the causal partition) was examined and NOT pre-registered as a rung: delta 2→1 removes two
`conv_a` taps (23→21, −472 raw B, a size change, not a shape change) and delta 2→3 multiplies the
coding groups 190→253 (+33 % group iterations) against a 27.581 s strict slack. Recorded, not fired.

---

## 3. The exact-byte pricing rail — control PASSES

Producer `experiments/ddm_hpr1_shape_price.py`, this arm's store. It is ntb2's law re-rooted on move
45 and owned here: the shipping receiver loop runs unchanged and only the arithmetic DECODE call is
replaced by the landed native ENCODER fed with known source symbols. The decoded plane is asserted
equal to the shipped field frame by frame inside the loop — **that assertion IS the output-lossless
proof, so no scorer is needed and none ran.**

**THE LIVE-LOOP CONTROL PASSES — MEASURED.** `price/control/PRICE.json`:

| object | this arm's re-encode | move 45 | verdict |
|---|---|---|---|
| archive sha256 | `145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a` | same | identical |
| archive bytes | 180,246 | 180,246 | Δ = 0 |
| `hpac` member | 11,911 | 11,911 | identical |
| RLC1 token stream | 119,749 | 119,749 | identical |
| decoded field sha256 | `a92e7d902a449896…` | the shipped field | output-lossless |
| twin encodes | agree | — | deterministic |

Both twins agree and every unrelated archive component (`semantic`, `carrier`, `tc1_weights`,
`residual_payload`) is asserted unchanged. **The rail is real; rung prices measured on it are
admissible.**

### 3a. Decode-time cost of a dilation rung — MEASURED

`experiments/ddm_hpr1_shape_timing.py`, receipt `probe/TIMING_conv_past.json`. `conv_past` runs once
per frame OUTSIDE the per-group loop, so its delta is the whole decode-time delta of the rung.
Interleaved A/B/A/B on the same host, on the module the shipping loader builds from move 45's bytes:

| dilation | ns/symbol | projected over 600 frames | Δ vs shipped | fraction of the 27.581 s strict slack |
|---:|---:|---:|---:|---:|
| 1 (shipped) | 102.089 | 12.043 s | — | — |
| **2 (R1)** | 105.389 | 12.432 s | **+0.389 s** | **1.41 %** |
| 3 (R2) | 111.864 | 13.196 s | +1.153 s | 4.18 % |

**Neither pre-registered rung is a timing blocker.** The 600-frame figure is a PROJECTION; a decode
budget's authority is a full public decode.

## 4. R1 — the exact price

R1's 60-epoch retrain under cl2's law, warm-started from the shipped prior itself, is COMPLETE.
`train/dil2/result.json` terminal epoch 60.

### 4a. The MODEL leg — MEASURED exactly

`price/past_dil2/INPUTS.json`:

| quantity | R1 | move 45 | Δ |
|---|---:|---:|---:|
| IHS1 body | 18,959 B | 17,770 B | +1,189 B |
| **`hpac` archive member** | **12,274 B** | **11,911 B** | **Δmodel = +363 B** |
| rows | 517 | 517 | 0 |
| stored values | 20,416 | 20,416 | **0 — the rung moved taps and added none** |
| mean row depth | 5.888 bits | 4.116 bits | +1.772 |

The layout invariant holds exactly: same rows, same per-row counts. **Δmodel = +363 B, well inside
cl2's +1,500 B bar.** For the joint to be negative the rung must buy Δtail < −363 B, i.e. a secant
Δtail/Δmodel < −1 — the same break-even cl2 measured at **+0.446** for capacity.

**A confound I must name before the tail lands.** The shipped prior's depths average 4.116 bits after
634 + 60 epochs of QAT; R1's average 5.888 after 60. Most of the +1,189 B body growth is plausibly
UNDER-TRAINING, not geometry — a dilated `conv_past` inherits weights meant for a different spacing
and has 60 epochs to re-fit them and re-compress their depths. The shape-CONTROL (shipped geometry,
identical init, identical 60-epoch law) is running to separate the two; without it the model leg
cannot be attributed to shape, and I will not attribute it.

### 4b. The training curve, with the control's anchor — MEASURED

At epoch 0 both runs hold the SAME weights (the shipped prior) and differ only in geometry, so the
gap is the pure cost of reading those weights at the wrong spacing:

| run | epoch 0 token estimate | epoch 0 model estimate | terminal (epoch 60) token estimate |
|---|---:|---:|---:|
| shipped geometry (CONTROL) | **124,038 B** | 27,026 B | in flight |
| R1, dilation 2 | **127,911 B** (+3,873) | 27,026 B (identical) | **122,900 B** |

R1 paid +3,873 B on contact with the new geometry, recovered 5,011 B over 60 epochs, and finished
**1,138 B below the shipped geometry's own epoch-0 value**. Whether it finishes below the CONTROL's
epoch-60 value is the shape question, and the control answers it.

**Calibration, and why the encode is still the authority.** The trainer's token figure is an ideal-code
estimate, not the real coder: the shipped prior estimates 124,038 B and its REAL RLC1 stream is
119,749 B, so the shipped stack beats the estimate by 4,289 B (3.46 %). Carrying that ratio onto R1
would project a stream near 118,650 B and a joint near −738 B — **a DERIVED projection, not a price.**
The ratio is a property of the corrector/mixer stack's interaction with a particular prior's errors
and there is no reason it transfers; the 600-frame encode now running is the only number that counts.

### 4c. R1's EXACT PRICE — MEASURED, and the pre-registered sign is CONFIRMED

`price/past_dil2/PRICE.json`, 600 frames, real coder, real receiver loop, twins agreeing:

| leg | R1 (`past_dil2`) | move 45 | Δ |
|---|---:|---:|---:|
| `hpac` member | 12,274 B | 11,911 B | **Δmodel = +363 B** |
| RLC1 token stream | 119,311 B | 119,749 B | **Δtail = −438 B** |
| **archive** | **180,171 B** | 180,246 B | **ΔB = −75 B** |
| archive sha256 | `581c636c365c0a5ad09b3d756285e0b7bfd856ef75d9eaf931105934fcd86ecb` | — | — |
| decoded field | equals the shipped field sha | — | **output-lossless** |

**Secant Δtail/Δmodel = −1.2066**, against cl2's **+0.446** for capacity and the **−1** break-even.
**The receptive field's SHAPE repays itself where its SIZE does not** — the first rung on this object
to cross cl1's break-even, and the pre-registered sign from the window atlas is confirmed by the real
coder. ΔS = −75 × 6.658589531221714e-7 = **−4.9939e-5**; projected S **0.13708842731735646**.
Decode-time cost +0.389 s, 1.41 % of the strict slack. Receiver change: YES (three edits, §1.3), so
the first-measurement chain, never a normal seal.

**Two honesty notes on the margin.** (i) −75 B is only ~2.2× the 34.8 B standard deviation the
container-break lottery carries, so the WIN is real and reproducible for these exact bytes but its
margin over zero is not comfortable; the −438 B tail leg is the robust half, the +363 B model leg is
the noisy half. (ii) The fire rule (net ΔS < −2e-5) is met by 2.5×, but firing R1 would be premature
until §4d resolves, because a cheaper candidate may be hiding inside the same experiment.

### 4d. The attribution that must land before anything fires

The CONTROL — shipped geometry, same init, same 60-epoch law — is running, and its early telemetry
says the question is live: at **epoch 8** it already estimates **122,707 B** of tokens, BELOW R1's
epoch-60 terminal of 122,900, with 52 epochs left to improve. If the control's exact archive also
beats 180,246 B, then part or all of R1's −75 B belongs to **retraining the prior on the CURRENT
field** — which the shipped prior has never had, since its weights descend from cl2's move-26 fit and
the field moved at move 32 (sj1's token pre-distortion) [DERIVED from the store, not measured here].

That would matter more than R1 itself: **a pure retrain is NOT a receiver change**, so it takes the
normal seal path with no first-measurement chain and no three-file patch.

**The control landed, and it changes the reading. MEASURED:**

| terminal, epoch 60 | token estimate | model estimate | joint estimate |
|---|---:|---:|---:|
| CONTROL (shipped geometry) | **121,791 B** | 18,924 B | **140,715 B** |
| R1 (dilation 2) | 122,900 B | 18,955 B | 141,855 B |

and, from its own packed body (`price/retrain/INPUTS.json`, `receiver_change: false`):

| model leg | CONTROL | R1 | shipped |
|---|---:|---:|---:|
| `hpac` member | 12,262 B (+351) | 12,274 B (+363) | 11,911 B |
| mean row depth | **5.890 bits** | **5.888 bits** | 4.116 bits |

**The depth inflation is identical across the two geometries to within 0.002 bits.** That settles the
§4a confound by measurement: the +1,150-ish B of body growth is a 60-epoch TRAINING BUDGET effect,
not a shape effect. Both retrained priors sit ~1.77 bits/value above the shipped prior's 634+60-epoch
QAT compression, whatever their geometry.

**So R1's naive secant of −1.2066 is computed against a model leg that shape did not cause.** The
shape-attributable legs are R1 minus the CONTROL, not R1 minus move 45: Δmodel(shape) = 12,274 −
12,262 = **+12 B**, and Δtail(shape) is R1's stream minus the control's — which the control's encode,
now running, measures exactly. The estimator says the control's tokens are ~1,109 B BELOW R1's, which
would make Δtail(shape) POSITIVE and the shape rung a LOSS against its own control.

### 4e. THE VERDICT — the shape rung is FALSIFIED, and the control is a −887 B candidate

Both encodes are complete on the same rail, same 600 frames, real coder, real receiver loop, twins
agreeing, decoded field equal to the shipped field sha in both cases.

| | `hpac` | RLC1 stream | **archive** | Δ vs move 45 | ΔS | receiver change |
|---|---:|---:|---:|---:|---:|---|
| move 45 | 11,911 | 119,749 | 180,246 | — | — | — |
| **R1 `past_dil2`** | 12,274 | 119,311 | 180,171 | **−75 B** | −4.994e-5 | **YES** (3 edits) |
| **CONTROL `retrain`** | 12,262 | **118,511** | **179,359** | **−887 B** | **−5.906e-4** | **NO** |

`retrain` sha256 `d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9`;
`past_dil2` sha256 `581c636c365c0a5ad09b3d756285e0b7bfd856ef75d9eaf931105934fcd86ecb`.

**The shape leg, isolated against its own control:**

| leg | value |
|---|---:|
| Δmodel(shape) = 12,274 − 12,262 | **+12 B** |
| Δtail(shape) = 119,311 − 118,511 | **+800 B** |
| **ΔB(shape)** | **+812 B** |
| ΔS(shape) | **+5.407e-4** |

**R1's pre-registered NEGATIVE sign is FALSIFIED at the exact-byte level.** Dilating `conv_past` to
stride 2 costs **+812 B** against the identically-trained shipped geometry. R1's −75 B against move 45
was the retrain's −887 B minus the shape's +812 B. The window atlas ranked `d2_c00` first by −3,625
ranking bytes and the real coder reversed the sign; the pre-registered falsifier written in §2b fires,
and **the conditional-information set statistic is refuted as a rung predictor on this object.** The
gap it could not see is that the neural prior's 64 channels already de-correlate the compact window's
redundancy, so the redundancy the statistic priced was not the prior's to buy — spreading the taps
only moved them further from where the information is densest. Verdict scope: FORMULATION — one
statistic, one geometry family, one 60-epoch budget; the shape paradigm is not closed, but no shape
rung may cite this statistic as a predictor again without a re-derivation.

**And the control is the real find.** `retrain` is the shipped geometry's prior re-fit to the CURRENT
token field under cl2's own law — a pure model-bytes change under the shipped receiver code, with
`receiver_change: false`. It is **−887 B / ΔS −5.906e-4**, projecting
**S = 0.1365477498474212**, and it clears the charter's −2e-5 fire bar by **30×**, at 25× the 34.8 B
container-break noise floor. Distortion is unchanged by construction (the decoded field is the shipped
field, byte for byte) and the semantic and carrier members are asserted untouched. Decode time is
unchanged — the geometry did not move.

**Why nobody had this.** The shipped prior's weights descend from cl2's λ=1.0 fit at move 26, and the
token field has moved since (sj1's pre-distortion at move 32 and after). The prior has been coding a
field it was never fit to. The −887 B is the cost of that staleness, and it was hiding in plain sight
as *the control* of a shape experiment. [The lineage is DERIVED from the store; the −887 B is MEASURED.]

## 5. Fire verdict, and the ONE step owed

**Fire verdict on the charter's rule (net ΔS < −2e-5 at exact bytes against 180,246 B):**

- **R1 `past_dil2` — DO NOT FIRE.** It clears the arithmetic bar (−4.994e-5) but it is dominated by
  its own control on every axis: 812 B larger, a three-file receiver change against none, and the
  first-measurement chain against a normal seal. Firing it would bank the retrain's win while paying
  the shape's loss and the receiver-change cost. Retained, priced, closed.
- **CONTROL `retrain` — FIRE-ELIGIBLE, ONE STEP OWED.** 179,359 B, ΔS −5.906e-4, projected
  S 0.1365477498474212, `receiver_change: false`, output-lossless by in-loop assertion, twins agreeing.

**The one step owed is the cold public parse-back**, and I stopped rather than fake it. The landed
prover, `experiments/ddm_ntb2_public.py`, already takes `--candidate-archive`, `--price-receipt`,
`--promoted-root` and `--public-root`, so it can read this arm's candidate — but its `--treatment`
choices are ntb2's own, and it writes that string into every receipt it emits. Running it as
`--treatment frame_even` over `retrain`'s bytes would stamp a FALSE LABEL into a custody receipt,
which is the exact class this campaign extincts, so I did not. I also did not edit a sister arm's live
producer while it runs, per this arm's boundaries. **The minimal honest unblock is a one-line change
to that prover — a `--label` override, or adding this arm's treatment names to its `choices` — after
which the parse-back is mechanical.** No seal exists and none is claimed until the raw comes back
byte-identical.

**Unfired, with pre-registered signs intact but their predictor now refuted:** R2 (`past_dil3`),
R6 (`cone_dil2` / `cone_dil3`), and the two offset rungs. Each is one trainer command plus one
`--treatment` on this rail. A successor should note that §4e refutes the statistic that ranked them,
so their signs are now UNSUPPORTED, not merely unfired — re-derive before spending Metal on them.

## 6. Apparatus landed

- `tools/train_ddm_cl1_hpac_capacity.py` gains ONE additive profile, `hpr1_shape_rungs`: cl2's config
  dict byte-for-byte, with TWO free geometry axes, `--past-dilation` {1,2} and `--conv-a-dilation`
  {1,2,3}. Verified additive per profile — `cl1` / `rx2_mc36` / `jf1_joint_refit` /
  `cl2_shipped_ladder` all PASS at dilation 1 on both axes and all REFUSE 2; 36 trainer tests pass.
- `experiments/ddm_hpr1_shape_price.py` — the move-45 rail, control-falsified, with `past_dil2`,
  `past_dil3`, `cone_dil2`, `cone_dil3` treatments, idempotent resume-safe receiver patches, and a
  layout invariant that admits depth movement (the rung's model leg) while refusing any change to the
  stored value count.
- `experiments/ddm_hpr1_shape_inputs.py` derives both trainer inputs from move 45 alone.
- `experiments/ddm_hpr1_shape_timing.py` — the interleaved decode-time instrument.
- `experiments/ddm_hpr1_surprise_atlas.py` — reconstruction, displacement atlas, set atlas, window
  atlas.

**A successor inherits a launchable board, not prose:** every rung above is one trainer command plus
one `--treatment` on a rail that already reproduces move 45 byte-identically.

## 7. Boundaries honoured

$0 spent. No Modal, no scorer run, no candidate claim, no pointer touch. `upstream/`, the PR tree, the
sealed promoted tree and the sister arms' directories (`ddm_ntb2_*`, `ddm_pc3_*`, `ddm_sr5`) were read
and copied, never written. MAIN's 2026-09-11 storage re-route was applied mid-arm: every payload
written after it is under `/Volumes/VertigoDataTier/pact/ddm_hpr1/`, the reserve in this arm's
producers was RAISED from 8 to 40 GiB to match the sister HPAC producers, and the earlier payloads
under `/Volumes/APDataStore/pact/ddm_hpr1/` were left in place as instructed. Two governor REFUSALs
(memory ceiling 116.0 GiB, projected 144.3 GiB) were recorded as information and waited out, never
bypassed and never lowered.
