# ddm_hpr1 — the HPAC prior's RECEPTIVE-FIELD SHAPE, priced on move 45

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false ·
`# FORMALIZATION_PENDING: the shape statistic is a conditional-information RANKING, not a coder charge; it is registered as a canonical equation only when a real byte price confirms or falsifies its sign, per the standing "first-order token price is a ranking not a charge" law.`

**STATE (Opus, 2026-09-11).** Deliverables 1 and 2 are COMPLETE and measured. Deliverable 3 (exact
bytes per rung) is in flight: the move-45 live-loop pricing rail PASSED its falsifying control,
and the winning rung is training. No frontier move, no Modal dispatch, no scorer run, no candidate
claim. This file is crash-resumable.

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
   (`source_row = global_row + kernel_row - 1`, lines ~448/455); and the two are selected at run time
   by `F26_TOKEN_DECODER` (shipped default `python`, `native-hpac` opt-in). A rung that patched only
   `inflate.py` would decode one field in the default mode and a DIFFERENT field in native mode — an
   archive that is output-lossless on one decoder and wrong on the other, with nothing to announce it.
   This arm's producer therefore moves the constant in both files or refuses on a non-unique anchor,
   and records both before/after by sha. **The hazard is pre-existing and applies to any future
   geometry rung on this object, not only to this one.**

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

## 4. What is owed

The per-rung price table (Δmodel, Δtail, ΔB exact, secant vs cl2's +0.446, ns/symbol against the
27.581 s slack, receiver-change verdict) and the fire verdict. R1's 60-epoch retrain is running on
Metal under the shipped `cl2` law from a warm start that is the shipped prior itself, so the CONTROL
and the rung differ only in geometry.

## 5. Apparatus landed

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

## 6. Boundaries honoured

$0 spent. No Modal, no scorer run, no candidate claim, no pointer touch. `upstream/`, the PR tree, the
sealed promoted tree and the sister arms' directories (`ddm_ntb2_*`, `ddm_pc3_*`, `ddm_sr5`) were read
and copied, never written. MAIN's 2026-09-11 storage re-route was applied mid-arm: every payload
written after it is under `/Volumes/VertigoDataTier/pact/ddm_hpr1/`, the reserve in this arm's
producers was RAISED from 8 to 40 GiB to match the sister HPAC producers, and the earlier payloads
under `/Volumes/APDataStore/pact/ddm_hpr1/` were left in place as instructed. Two governor REFUSALs
(memory ceiling 116.0 GiB, projected 144.3 GiB) were recorded as information and waited out, never
bypassed and never lowered.
