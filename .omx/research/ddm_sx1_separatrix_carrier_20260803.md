# `ddm_sx1` — Is there a description whose byte cost scales with the SEPARATRIX (1-D) rather than the REGION (2-D)?

**arm:** `ddm_sx1` · **date:** 2026-08-03 · **axis:** `[macOS-CPU advisory]` — **NO scorer run; NO score claim.**
`score_claim=false` · `promotion_eligible=false` · `rank_or_kill_eligible=false`.
All measurements below are on **cached GT SegNet argmax** (`lstars`, n600, full population) and on
**`upstream/models/segnet.safetensors` weights read directly**. Zero contest-scorer forwards were run.

---

## §0 ANSWER FIRST

**YES — and it is measured, exactly, losslessly, on the full n600 population.**

The exact target label field `L*` for all 600 scored frames costs **253,341 B** (order-4 causal
context model, incl. model cost), of which **89.69 % of the bits sit on the 2.16 % of pixels that
are the separatrix**: **0.6984 bits/boundary-px vs 0.001775 bits/interior-px — a 393× concentration**.
The dimensional reduction is real and it is not a hope; it falls out of a plain context model with
no new machinery.

**But the headline finding is the consequence, not the ratio:**

> `L*` costs **0.5349 B per flip-to-kill** against a break-even of **W = 1.27310821533203125 B/flip**.
> **Describing the entire answer perfectly consumes only 42 % of the seg budget.**
> The remaining **58 % — 0.738 B/flip = 5.90 bits/flip — is the budget available for REALIZATION.**

So the seg axis is **not description-capacity-bound**. `ddm_sg2`'s verdict ("seg is a
DESCRIPTION-EFFICIENCY problem") is **refined, not contradicted**: the *where* is nearly free; the
whole difficulty is the *what-RGB*. That reframing is the deliverable, and it comes with a number:
**realization has 5.90 bits/flip to work in.**

**Load-bearing caveat, welded on:** 0.5349 B/flip is a **FLOOR**, attained only if perfect knowledge
of `L*` produced perfect realization. It does not. The decoder ships **RGB**, not labels, and it has
no SegNet. See §6 — this is why S3 below is a reference point and not a carrier.

**Second headline, from a cross-axis question I was asked to test rather than assume (§8.4):**
**seg and pose DECOUPLE at the per-pair mass channel.** `ddm_pu1`'s top-6 pose pairs hold **62.0 %**
of pose mass and **1.15 %** of seg mass against a 1.00 % uniform — a 1.15× enrichment.
`Pearson r = +0.085`, `r² = 0.007`. Pose is a 6-pair spike (max/mean 185×); seg is nearly uniform
(max/mean 2.3×). **`pu1`'s numbers are not lower bounds by way of this channel, and my byte budget
is exactly what it appears to be.** The structural reason was already in my §2.1 derivation:
`SegNet` reads `x[:, -1]` only, so a separatrix description is a *single-frame* object and supplies
one endpoint of a displacement, not the displacement.

---

## §1 THE DENOMINATOR RECONCILIATION (charter §2) — resolved, and it resolves the other way

The charter asked me to suspend every `pc2` share until the `458,738` vs `50,863,944` discrepancy
resolved. **It resolves as a 100× arithmetic slip in the charter, not a defect in `pc2`.**

| quantity | value | derivation |
|---|---:|---|
| `cx1` `d_seg` (per-pair mean) | `0.004311794704861111` | `ddm_pz1_dseg_n600_cx1_20260803.json` → `d_seg_base_mean` |
| `cx1` seg TERM | `0.4311794704861111` | `= 100 · d_seg` (verified exactly) |
| PX | `117,964,800` | `600 × 512 × 384` (`segnet_model_input_size`) |
| **`cx1` flips** | **`508,640`** (exact integer) | `d_seg × PX` |
| charter's figure | `50,863,944` | `seg_TERM × PX` — **uses `100·d_seg` where `d_seg` belonged** |

`pc2`'s `458,738` is therefore **90.19 %** of `cx1`'s flip population — **not 0.9 %**.

And it is not a subset at all: `458,738 / PX = 0.00388877`, against `tb1 ep399`'s quoted
`d_seg = 0.0038892` — agreement to **1.1 × 10⁻⁴ relative**, pure rounding.
**`pc2`'s atlas is the COMPLETE n600 flip population of the `tb1 ep399` vehicle.**

**Verdict: `pc2`'s shares are ANCHORED. Nothing built on them is suspended.**
The honest residual caveat is a *vehicle* caveat, not a *denominator* caveat:

> `pc2` measured on `tb1 ep399` (`d_seg 0.0038888`). `cx1` has **10.87 % MORE flips**
> (`0.0043118`). Edge shares transfer only if edge composition is stable across that growth.
> Untested. Every `pc2` share quoted below carries this label.

Note in passing, and it is not small: **the live best vehicle `cx1` has 10.9 % WORSE seg than the
`tb1` burn endpoint.** The burn was seg-specialised; every win since has been pose or rate.

---

## §2 DERIVED FROM `upstream/` ALONE, BEFORE READING OUR RECEIPTS (charter §0)

Sources: `upstream/modules.py`, `upstream/evaluate.py`, `upstream/models/segnet.safetensors`.

### 2.1 What `d_seg` actually is — VERIFIED_VIA_SOURCE_INSPECTION

```python
# modules.py :: SegNet
def preprocess_input(self, x): x = x[:, -1, ...]        # LAST FRAME ONLY
def compute_distortion(self, out1, out2):
    diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
    return diff.mean(...)
```

Three consequences, all exact:

1. `d_seg` is a **disagreement rate between two argmaxes**, both produced by the same frozen net.
   The target `L* = argmax SegNet(GT)` is a fixed field, computable offline, **600 frames** — not 1200.
2. **`frame_0` carries no seg obligation at all** (`x[:, -1]`). *(Converges with a known result.)*
3. The loss is **0-1 on a 5-way label field**. Everything in the preimage of the correct argmax cell
   is FREE. We are not asked to reproduce the image — only to land in the right cell.

### 2.2 The head is *literally* a power diagram — VERIFIED numerically

`smp.Unet(..., classes=5, activation=None)` ends in `segmentation_head.0 = Conv2d(16, 5, k=3)`.
Read from the checkpoint: weight `(5,16,3,3)`, bias `(5,)`. So the logits are **affine in R^144**
(16 channels × 3×3 tap):  `z_c(p) = ⟨w_c, f(p)⟩ + b_c`.

Setting `s_c = w_c/2` and `r²_c = |s_c|² + b_c`:

```
argmax_c ( ⟨w_c,f⟩ + b_c )  ≡  argmin_c ( |f − s_c|² − r²_c )
```

**Verified on 20,000 random features: argmax agreement = 1.000000.** (The two forms differ by the
class-independent `−|f|²`, which cancels in the argmax — hence the 219.5 raw offset.)

That is **exactly a 5-site Laguerre/power diagram** in R^144. The `c↔c'` separatrix is the radical
hyperplane with normal `w_c − w_c'`.

**Scope of what I verified — three names, not four.**
*Power/Laguerre* is what I checked numerically (agreement 1.000). *Bregman Voronoi* with the
squared-Euclidean divergence **is** that same statement, so it is licensed. *Tropical regular
subdivision* follows structurally: `max_c(⟨w_c,f⟩+b_c)` is a tropical (max-plus) polynomial and its
regions are its regular subdivision — licensed but **not separately verified here**.
**Morse–Smale is NOT licensed and I do not use it.** It needs a potential/flow hypothesis that the
head's affineness does not supply; nothing in `upstream/` provides one. *(Independently reached by
sister arm `ddm_de1` — see §8.3.)* No persistence-ordering or separatrix-as-flow-object argument
anywhere in this memo rests on it.

**No free class-adjacency sparsity, either.** Full quotient rank makes all five cells
full-dimensional, so **every class pair is generically adjacent**; you may not infer a sparse edge
set from the geometry. **Accordingly I MEASURED the edge mass (§3) rather than inferring it** — and
the measurement finds **9 of the 10 pairs carry nonzero interface length** (only Undriv↔MyCar is
absent), with the mass extremely unequal (top edge 50.25 %, bottom four 0.51 % combined). The
sparsity is *empirical*, not structural.

### 2.3 The rank-4 fact — the sharpest thing I derived

The five centred class weights `w_c − w̄` have singular values

```
3.128e+00   2.154e+00   2.025e+00   1.796e+00   3.730e-16
```

**Numerical rank exactly 4 of ambient 144.**

> **`d_seg` depends on the 144-dim decoder window ONLY through a 4-dimensional projection.
> 140 of 144 dimensions (97.2 %) are EXACTLY invisible to the seg term.**

Energy is spread `45 % / 21 % / 19 % / 15 %` — no dominant direction, so there is **no rank-1/2
shortcut**. Per-channel discriminative energy is likewise spread (max 20.65 %; 9 of 16 channels
needed for 80 %) — **no cheap channel pruning either.** Both are honest negatives that close
otherwise-tempting doors.

*(Usability caveat, stated because it is the whole difficulty: this invisibility lives in DECODER
FEATURE space. Exploiting it requires inverting the encoder. It is not directly a byte lever. It is
listed because it bounds what any feature-side scheme could ever hope for.)*

### 2.4 The derivation this predicted — and it is REFUTED

If flips were governed by the head's own geometry, flip density per unit interface length should
track `‖w_i − w_j‖` (a larger normal ⇒ a smaller feature perturbation flips the sign).

| edge | flip % *(pc2, tb1)* | **interface-length %** *(sx1, GT)* | flips/len | ‖wᵢ−wⱼ‖ | near-3px *(pc2)* |
|---|---:|---:|---:|---:|---:|
| Road↔Lane | 49.23 | **50.25** | 0.980 | **3.954** | 2.6 % |
| Road↔Undriv | 16.26 | **17.91** | 0.908 | 2.602 | 2.1 % |
| Undriv↔Movable | 11.85 | **6.14** | **1.929** | 2.946 | **19.7 %** |
| Road↔Movable | 11.47 | **5.58** | **2.057** | 2.942 | **17.9 %** |
| Road↔MyCar | 10.89 | **19.61** | 0.555 | 2.705 | 0.8 % |

**Pearson(‖wᵢ−wⱼ‖, flips/len) = −0.0042, n=5. REFUTED.**
The frozen head's linear geometry does **not** predict where we fail. What does is the *encoder
Jacobian* — i.e. how easily an RGB perturbation moves the decoder feature — which is content- and
texture-dependent. **This independently corroborates the established result that the MARGIN field
(Fisher-correlated at 0.978), not the head weights, is the right cost surrogate.**

### 2.5 …but the interface-length denominator bought something `pc2` could not see

The `flips/len` column is new — it required my GT interface lengths (§3) joined to `pc2`'s flip
shares. It splits the residual into **two mechanisms that need two different carriers**:

| mechanism | edges | share of flips | signature |
|---|---|---:|---|
| **boundary-PRECISION** | Road↔Lane, Road↔Undriv, Road↔MyCar | **76.4 %** | `flips/len` 0.55–0.98; **97–99 % ON the GT boundary** |
| **object-DISPLACEMENT** | Undriv↔Movable, Road↔Movable | **23.3 %** | `flips/len` **1.9–2.1** (2× denser); **18–20 % >3 px OFF** the GT boundary |

**A separatrix carrier addresses the 76.4 %. It cannot address the 23.3 %** — you cannot refine the
edge of an object that is in the wrong *place*; that needs a positional/motion DOF. This is the same
"per-pair POSITIONAL DOF" deficit `pc2` §5 names, now with a share attached.

---

## §3 M1 — SEPARATRIX GEOMETRY (MEASURED, n600, full population)

`experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz` → `lstars (600,384,512)`.
Artifact: `.omx/research/ddm_sx1_separatrix_geometry_n600.json`.

```
boundary pixels (4-conn)   B = 2,551,382      = 2.163 % of area      4,252.3 / frame
crack length (adj. pairs)      1,619,917                             2,699.9 / frame
AREA / BOUNDARY                    46.24×
flips / boundary px             0.19936   <-- only 19.94 % of the separatrix is actually wrong
```

**Per-edge interface length** (this is the new denominator):

| edge | crack length | share |
|---|---:|---:|
| Road↔Lane | 814,066 | **50.25 %** |
| Road↔MyCar | 317,679 | 19.61 % |
| Road↔Undriv | 290,167 | 17.91 % |
| Undriv↔Movable | 99,530 | 6.14 % |
| Road↔Movable | 90,322 | 5.58 % |
| *(4 more)* | 8,153 | 0.51 % |

**Independent corroboration of `pc2` from pure geometry, no vehicle and no scorer:**
Road↔Lane is **50.25 %** of interface length; `pc2` measured **49.23 %** of flips. One percentage
point apart, from two entirely different instruments. Road-incident edges: **93.35 %** of length vs
`pc2`'s **87.85 %** of flips.

> **First-order relation (n=5, NOT a law — round-2 downgrade): flip mass tracks interface LENGTH
> to within a factor of ~2, and the DEVIATIONS are the informative part.**
> Two edges run 1.9–2.1× denser than length predicts and one runs 0.55×; §2.5 shows the deviation
> is exactly the precision-vs-displacement split. Calling this a "law" (as my first draft did)
> overstates an n=5 share comparison across two different vehicles. What it *does* license is
> treating the crack — not the cell — as the natural unit of description, which is all §5 needs.

**Independent check that this cache is the right object (assumption A3).** The class-area shares
measured here on n600 reproduce the canonical n96 anchor in `CLAUDE.md` to within sampling noise:
Road 23.23 % (vs 22.9), Lane 0.585 % (vs 0.59), Undriv 49.52 % (vs 49.3), MyCar 25.43 % (vs 25.6).
Two independent caches agree. Further, the class **order is self-detected, not assumed**, exactly
as `CLAUDE.md` mandates: index 4 has 1.0 connected components per frame in 600 of 600 frames and
sits in the bottom rows ⇒ MyCar/ego-hood; index 2 is the largest area and occupies the top ⇒
Undrivable. **This matches the canonical comma10k order and NOT the forbidden luma-sort.**

---

## §4 M2 — CELL OCCUPANCY vs CELL SIZE (MEASURED)

Fraction of `c × c` cells that the separatrix touches. Artifact:
`.omx/research/ddm_sx1_cell_occupancy_n600.json`.

| c | cells (n600) | boundary cells | occupancy | dense/sparse |
|---:|---:|---:|---:|---:|
| 1 | 117,964,800 | 2,551,382 | 2.16 % | **46.24×** |
| 4 | 7,372,800 | 397,530 | 5.39 % | 18.55× |
| 8 | 1,843,200 | 168,504 | 9.14 % | 10.94× |
| **16** | **460,800** | **74,606** | **16.19 %** | **6.18×** |
| 32 | 115,200 | 32,565 | 28.27 % | 3.54× |

**The `A/(L·c)` scaling law is confirmed empirically**: the sparsity dividend *shrinks* as cells
coarsen, because a curve of length `L` crosses `~L/c` cells of size `c` while the dense grid has
`A/c²`. At the incumbent `c=16`, **the maximum harvestable sparsity is 6.18×**, not 46×.

**This is the honest ceiling on the pure-sparsity play, and it is well short of the 14.537× d_seg
factor required** — but note those are different currencies (§5).

---

## §5 M3 — THE MDL FLOOR (MEASURED, exact, lossless)

Artifact: `.omx/research/ddm_sx1_label_field_mdl_n600.json`.

| model | total | bits/px | per frame |
|---|---:|---:|---:|
| H0 order-0 marginal | 23,821,632 B | 1.6155 | 39,703 B |
| **H1 causal spatial (W,N,NW,NE)** | **248,341 B** (+5,000 B model) | **0.0168** | **414 B** |
| H2 + temporal (prev pair) | 212,136 B (+25,000 B model) | 0.0144 | 354 B |
| *(MEASURED, off-the-shelf)* lzma-9e on raw labels | ~428,120 B | 0.0290 | 714 B |

**Bit localisation under H1 — this is the dimensional reduction, measured:**

```
boundary   2,551,382 px ( 2.16 % of area)  ->  222,729 B  =  89.69 % of bits   0.6984  bits/px
interior 115,413,418 px (97.84 % of area)  ->   25,612 B  =  10.31 % of bits   0.001775 bits/px
                                                                              ------- 393× -------
```

Three things worth naming:

- **We land INSIDE the published contour-coding band, not below it — my first reading was
  apples-to-oranges.** `0.6984 bits/boundary-PIXEL` is not comparable to the literature's
  `1–1.5 bits/contour-PIXEL`, because the boundary *pixel set* (2,551,382) is ~1.6× the boundary
  *crack length* (1,619,917) — a straight edge contributes two pixels per step. Per actual contour
  step: `222,729 B × 8 / 1,619,917 = **1.10 bits/crack-step**`, which sits at the good end of the
  published 1–1.5 band. **Corroboration, not a beat.** *(Caught in round-2 self-review; the earlier
  "beats the floor" claim was wrong and is withdrawn.)*
- The **off-the-shelf** number (lzma, zero new code) is only **1.69×** off the context-model floor.
  Nothing exotic is required to approach this.
- **Estimator hygiene:** H1/H2 are *in-sample* empirical conditional entropies. Miller–Madow bias
  for 625 contexts over 118 M samples is ≈225 B total — negligible against 248,341 B — and I added
  a deliberately generous 5,000 B model cost on top. Raster/column padding with class 0 affects
  ~0.46 % of pixels. Neither materially moves the result.

### 5.1 The currency conversion — where the 14.537× goes

**W = 4·DEN/PX = 1.27310821533203125 B/flip = 10.1849 bits/flip.** *(Derived: setting
`25·B/DEN = 100·flips/PX`.)* A carrier costing `X` bytes and killing `F` flips wins iff `X/F < W`.

```
cx1        S 0.8264972   seg 0.4311795   pose 0.1597320   rate 0.2355862   353,808 B
PR130 bar  S 0.172141    seg 0.02966                                       191,052 B
gap 0.6543562 ; seg gap 0.4015195 = 61.36 % of gap ; flips to kill 473,652
value of the seg residual at W = 603,010 B
required d_seg factor = 14.5374×   <-- a DISTORTION factor, not a byte factor
1 % of gap = 9,827.2 B
```

**Two disciplines on `W` and on 603,0xx, both owed:**
- **`W` is DERIVED, not measured.** Only `DEN = 37,545,489` (frozen source size) is measured; `PX`
  is fixed by `segnet_model_input_size`. **A codec cannot change `W`** — it can only change its own
  achieved bytes per *realized* correction. Nothing below claims otherwise.
- **603,008 / 603,009 / 603,010 B are the same number under different roundings of `seg_bar`.**
  Mine is 603,010 from `seg_bar = 0.02966`; `sg2` reports 603,009; the strict fixed-pose integer
  spend cap is 603,008. The spread is 2 B ≈ 0.0000013 S. **None of the three predicts the
  residual's description length — that is what §5 measures, and it is 253,341 B.**

> **`L*` at 253,341 B / 473,652 flips = 0.5349 B/flip = 42.0 % of W.**
> **Realization budget = 0.7382 B/flip = 5.91 bits/flip.**

**Round-3 discipline on that split.** Description and realization are **not architecturally
separable** — the token field does both at once. The 42 %/58 % split is a **conceptual budget**
that bounds what realization may cost, not a two-section construction. It is useful because it is a
*bound*; do not read it as a design.

**Absolute-archive sanity check** (same claim, different units, to show the magnitude is not a
marginal-analysis artifact): shipping `L*` outright would take the archive to
`353,808 + 253,341 = 607,149 B` → rate term `25·607,149/DEN = 0.4043`. With seg at the PR130 bar
and pose unchanged: `0.02966 + 0.15973 + 0.4043 = 0.5937` — **0.233 better than `cx1`'s 0.8265**,
*even paying full freight for the labels and buying nothing back on rate*. **The description is
affordable in absolute terms, not merely at the margin.** (Counterfactual — it assumes perfect
realization, which §6 says we do not have. It is a magnitude check, not a proposal.)

**Constants verified from source this round** (`upstream/frame_utils.py`):
`segnet_model_input_size = (512, 384)` · `camera_size = (1164, 874)` · `seq_len = 2`
⇒ `PX = 600·512·384 = 117,964,800` ⇒ `W` above is derived from verified inputs.

---

## §6 THE REALIZATION WALL — why §0's ratio is a floor and not a carrier

This is the attack that survives, and it must travel with every number above.

`evaluate.py` scores `argmax SegNet(x̂)` where `x̂` is **RGB produced by `inflate.sh`**. Therefore:

1. **The decoder has no SegNet, and cannot afford one.** DERIVED, not inherited:

   ```
   decode-side seg-apparatus ceiling = W · (all current flips) = 4·DEN·d_seg = 647,553.8 B
   segnet.safetensors  = 38,502,892 B   ->  59.46× over the ceiling
   shipping it: ΔS_rate = 25·38,502,892/DEN = +25.6375 S ; best possible seg gain = −0.4312 S
                                                              net = +25.2063 S
   ```
   *(My first draft of this line was WRONG — I used 94,338,452 B, which is BOTH scorers
   (`posenet` 55,835,560 + `segnet` 38,502,892), and got 145×. Recomputing from `os.path.getsize`
   reproduces the charter's 59.5× and +25.21 S exactly. Caught in round-1 self-review.)*

   Any distilled decode-side separatrix predictor must therefore fit under **647,553 B** — and that
   figure is the ceiling for saving **every** flip, so a predictor recovering fraction `φ` gets
   `φ · 647,554 B`.
2. **Knowing `L*` does not produce `L*`.** A flip-set is unusable decode-side: the decoder cannot
   evaluate its own argmax, so it cannot know which pixels to repair.
3. **SegNet sees REGIONS.** Stride-2 stem + deep conv ⇒ the argmax at `p` is a function of a
   receptive field, not of `p`. This is the documented reason linear "store-the-flip-pixels"
   sidecars NO-GO'd three times.

**But (3) cuts both ways, and the derivation says the second edge is ours.** A receptive field of
radius `R` means one perturbation blob influences ~`R` consecutive contour pixels. It forbids
*independent per-pixel control*; it **amortises** *contour-coherent repair*. And §7 shows the
residual **is** contour-coherent. Whether the amortisation is realisable is the open question —
but it is the *right* open question, and it now has a 5.90 bits/flip budget attached.

---

## §7 M4 — IS THE RESIDUAL CONTOUR-COHERENT? (the charter's "test, do not assume")

Artifact: `.omx/research/ddm_sx1_contour_coherence_n600.json`. 8-connected components, per class,
per frame, n600.

```
total components 18,839  =  31.4 per frame   (all five classes together)
component size: median 16 px, mean 6,262 px, max 100,287 px
components <=16 px: 50.0 % of components  but only 0.045 % of pixels
```

| class | components | comp/frame | mean size |
|---|---:|---:|---:|
| Road | 1,088 | 1.8 | 25,190 px |
| **Lane** | **14,323** | **23.9** | **48 px** |
| Undriv | 631 | 1.1 | 92,573 px |
| Movable | 2,197 | 3.7 | 665 px |
| MyCar | 600 | **1.0** | 49,989 px |

**VERDICT: strongly coherent.** A frame is ~5 large regions plus ~24 tiny Lane dashes.
**MyCar is exactly 1.0 components/frame** — one static ego-hood blob, 600 of 600, which is the
in-vehicle existence proof that a per-class carrier pays (`pc2`: MyCar 0.23× BELOW the area fit).

**76 % of ALL components are Lane** — this is the "lane dashes = lowest-persistence birth–death
pairs" claim, now quantified: 14,323 components carrying 0.585 % of area and sitting on the edge
that owns 50.25 % of interface length and 49.23 % of flips.

---

## §8 THE DELIVERABLE — ranked separatrix-native description classes

Every `B/flip` compared against **W = 1.27311 B/flip**. Incumbent token section = **341,295 B**
(`state/tokens.dr7t`, deflated, from `ddm_cx1_pj2_container_compose_20260802.md`) = 96 % of archive.

| # | class | mechanism | derived cost | B/flip vs W | falsifier | consumer |
|---|---|---|---|---|---|---|
| **S1** | **distance-to-separatrix RATE ALLOCATION** | keep dense tokens, allocate PRECISION by distance to separatrix: full in boundary cells, reduced (factor `q`) in interior cells. **NOT deletion** (§8.1). | see §8.2 — **`−106,221 B` to `−177,730 B`** across both candidate carrier scales | frees rate at **0 B/flip**; buys **83,434–139,604 flips** of headroom | **one n600 scorer pass**: must not add more flips than the headroom column *and* must not move `d_pose` | rate lane / `#766` waterfill |
| **S2** | **generic separatrix predictor + counted placement residual** | `inflate.py` runs a GENERIC contour extractor on its own decoded base (rule-118 FREE); only the placement residual is counted. Direct application of the `gt1` BASIS/COEFFICIENT law. | **≤ 253,341 B** (hard upper bound = shipping `L*` outright); actual `= 253,341·(1−ρ)` | ≤ **0.5349** = **42 % of W** | measure hit-rate `ρ` of a concrete generic extractor on decoded frames — **scorer-free** | makes S1's occupancy map free |
| **S3** | **direct `L*` + decode-side paint** (MPEG-4-object archetype) | ship the label field, paint regions | **253,341 B** (H1) / 237,136 B (H2) | **0.5349** floor | flat paint is out of SegNet's training distribution; PoseNet needs photometric structure | **REFERENCE POINT, not a carrier** (§6) |
| **S4** | **two-rate composite S1∘S2** | boundary-banded token precision, band located by the free generic predictor | S1 savings with S2's map at ~0 B | best available | both above | the shippable synthesis |
| ~~S5~~ | ~~head-geometry-weighted allocation~~ | ~~allocate ∝ ‖wᵢ−wⱼ‖~~ | — | — | **REFUTED §2.4**, Pearson −0.0042 | — |

### 8.1 S1 is rate allocation, NOT interior deletion — my own round-1 kill

My first formulation of S1 was "drop interior cells: 6.18× fewer cells, −0.19045 S". **I refuted it
myself before reporting.** The token field is not a seg-correction sidecar — it is the **image
content** (96 % of the archive; the renderer decodes it to RGB). Deleting interior tokens deletes
the picture in the interior, which (a) destroys `d_pose` (PoseNet consumes the whole frame) and
(b) plausibly *raises* `d_seg` via SegNet's receptive field pulling interior content into boundary
decisions. The surviving formulation allocates **precision**, and keeps every cell present.

*This is the single most important correction in this memo; the deleted version had a headline
ΔS four times larger and was wrong.*

### 8.2 S1's arithmetic, and the carrier-scale uncertainty I could not close

My M2 occupancy is indexed by **cell size in SCORER-grid pixels**. `pc2` reports the carrier's
`render_grid_default: [192,256]` — exactly **half** the 384×512 scorer grid — and `tokens_delta` as
"4 numbers per 16×16 cell". If those 16×16 cells are on the *render* grid they are **32×32 in
scorer pixels**. I could not pin this down from the artifacts I hold, so **I give both**:

| scorer-px cell | occ | q | tokens B | +map B (iid UB) | total B | **saved B** | **ΔS_rate** | flip headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16.19 % | 0.50 | 198,276 | 36,798 | 235,074 | **106,221** | **−0.07073** | 83,434 |
| 16 | 16.19 % | 0.25 | 126,767 | 36,798 | 163,565 | **177,730** | **−0.11834** | **139,604** |
| 32 | 28.27 % | 0.50 | 218,887 | 12,371 | 231,257 | **110,038** | **−0.07327** | 86,432 |
| 32 | 28.27 % | 0.25 | 157,682 | 12,371 | 170,053 | **171,242** | **−0.11402** | 134,507 |

**The scale question turns out not to matter much** — the two scales land within 4 % of each other
on ΔS, because coarser cells lose sparsity but also cheapen the map. That is a useful robustness
result, and it is why I did not block on resolving the geometry.

*Round-3 addendum:* the camera→scorer map is **non-integer** — `1164/512 = 2.2734`,
`874/384 = 2.2760` (verified from `frame_utils.py`) — so a carrier cell defined in camera or render
pixels does **not** land on an integer scorer-cell boundary at all. This is a further reason the
exact geometry must be pinned before G1 is built. The ±4 % robustness above spans a full 2× in cell
size, which brackets the plausible range, so the *conclusion* survives; the *exact* ΔS does not
until the geometry is read off the real archive.

Three honest deductions from the numbers:

- **The map cost is an iid upper bound and is the loosest term.** The occupancy map is a *dilated
  contour* — highly structured — so its real cost is far below `H(occ)·N_cells`. A **two-pass
  decode** removes it entirely: decode coarse everywhere → run the free generic contour predictor
  (S2) on the coarse image → apply the refinement only where it predicts boundary. Then the map is
  *derived*, not transmitted. This is the tightest coupling between S1 and S2 and it is why S4 is
  the shippable form.
- **`q` is not a free real parameter.** The section is "cell-major nibble" — 4-bit codeword indices.
  You cannot scale precision continuously; you re-quantise the codebook or subsample cells. `q` is
  a *modelling* parameter here; the achievable set is discrete and format-constrained.
- **The `saved B` column is a BUDGET, not a gain.** It says how many new flips the freed rate could
  absorb at break-even. Whether interior coarsening actually costs fewer flips than that is exactly
  what G1 measures, and it is assumption **A7**.

### 8.3 DIFF against the independent sister derivation (`ddm_de1`, codex, different model family)

Two arms, two model families, deliberately uncoordinated: `de1` derived from `upstream/` alone;
I measured. **Where we converge, that is two-path corroboration. Where we differ is the finding.**

**CONVERGED — independently, same conclusion:**

| claim | `de1` (derivation) | `sx1` (this memo) |
|---|---|---|
| **the 4-dim quotient** | "exact four-coordinate decision quotient, **140** local feature-patch null directions" | rank **exactly 4** of ambient 144, σ₅ = 3.7e-16; **140** invisible (§2.3) |
| power/Laguerre + tropical exact at terminal head | derived | verified numerically, agreement 1.000 (§2.2) |
| **Morse–Smale NOT an identity** | derived; no licensing hypothesis in corpus | reached independently in round-1; struck from §2.2 |
| no free class-adjacency sparsity | derived from full quotient rank | **measured**: 9/10 pairs carry mass (§3) |
| answer to the question | "**YES for the assignment, CONDITIONAL for a legal witness**" | "**YES**; description is 42 % of budget, **realization is the wall**" (§0, §6) |
| the missing object | "**a legal fixed RGB section**" | "the decoder ships RGB, not labels" (§6); token field is image content (§8.1) |

**The agreement on 140 null directions, from two model families on two different paths, is the
single strongest corroboration in this pair.**

**DIVERGED — and it is complementary, not contradictory:**

1. **`de1` has the SHAPE; I have the SCALE.** It gives `O(L·log(E₀/L) + R + topology)` and a
   coherent-chain subclass at `O(L + K·log A + R)`. I measured every symbol in that expression on
   this video: `L = 2,551,382`, `K = 18,839`, `R = 5`, and the realized code length **253,341 B**.
   The two results compose exactly; neither is complete alone.

2. **`de1` says "the dimensional claim FAILS for area-scale topology." On THIS video it does not
   bind — and that is my measurement to contribute.** Area-scale topology would mean `K ~ 10⁵`
   components per frame. **M4 measures `K = 31.4` per frame** (§7). The topology term is
   `K·log₂A ≈ 18,839 × 17.6 bits ≈ 41 KB`, comfortably inside the 253 KB total (and already
   implicit in H1). **`de1`'s caveat is correct in general and inactive here.** I would not have
   known to check it without `de1`'s formulation; `de1` could not have known it was inactive
   without the measurement.

3. **`de1`'s second failure condition IS my §6/§8.1, reached from the opposite side.** It says the
   claim fails "whenever the fixed contour→RGB section needs area-scale video-specific texture."
   I killed my own interior-deletion formulation for exactly that reason: the tokens *are* the
   texture. **Same wall, two approaches, no coordination.** This is the most decision-relevant
   convergence in the pair and it should be treated as established.

4. **Where I go further:** `de1` stops at "a legal fixed RGB section is missing." I quantify the
   space that section has to live in: **0.7382 B/flip = 5.91 bits/flip** (§5.1). That converts an
   open design question into a budgeted one.

**Adopted from `de1` without independent verification** (labelled INFERRED, not measured here):
the `B(F) ∝ F^(−1/2)` smooth-lossy-contour bound; the `O(·)` forms above.

### 8.4 SEG↔POSE COUPLING — tested inside my own derivation. **They DECOUPLE.**

`ddm_pu1`'s round-3 flagged an unresolved assumption that would, if true, mean its numbers are
lower bounds: *"by Chasles the same `se(3)` twist serves both; a shared `ξ` means the tail's bytes
are already paid for by the seg side."* I did not adopt it — I tested it, on per-pair `d_seg` and
per-pair `d_pose` from the **same live `cx1` base**, n600, full population.
Artifact: `.omx/research/ddm_sx1_segpose_coupling_n600.json`.

| probe | pose | **seg** | uniform | seg enrichment |
|---|---:|---:|---:|---:|
| `pu1`'s top-6 pose pairs `[74,67,21,523,16,71]` | **62.01 %** | **1.15 %** | 1.00 % | **1.15×** |
| block `[60,80)` | 45.60 % | 4.07 % | 3.33 % | 1.22× |
| pair 74 alone | 30.92 % | 0.22 % | 0.167 % | 1.32× |

```
Pearson(d_seg, d_pose)  r = +0.0846  (p=0.038)   ->  r^2 = 0.0072
Spearman                r = +0.0908  (p=0.026)
concentration   pose: top1 30.92%  top6 62.01%  top60 78.48%   max/mean = 185.50x
                seg : top1  0.38%  top6  2.10%  top60 14.71%   max/mean =   2.30x
```

**Verdict: at the per-pair MASS channel the two axes are decoupled.** The correlation is
statistically detectable and *practically negligible* — it explains **0.7 %** of variance. The
distributions are of a different kind: **pose is a 6-pair spike (185× max/mean), seg is nearly
uniform (2.3× max/mean).** Where pose is hardest, seg is ordinary.

**Consequence for `pu1`: its numbers are NOT lower bounds by way of this channel.** And consequence
for me: **my byte budget is exactly what it appears to be.** A separatrix carrier priced against seg
alone is priced correctly.

**Structural argument that predicts the same thing, from §2.1** — reached independently and before
the test: `SegNet.preprocess_input` takes `x[:, -1, ...]`, so **the seg description is a
single-frame object over `frame_1`**; PoseNet consumes the *pair* (12 channels = 2 frames × YUV6).
A `frame_1` boundary description supplies **one endpoint of a displacement, not the displacement** —
and `frame_0`, half of PoseNet's input, carries zero seg obligation. There is no mechanism by which
a static per-frame separatrix description pays pose's bill.

**How much motion information does my description actually carry? Measured: 14.6 %.**
H1 (spatial, per-frame, static by construction) = 248,341 B; H2 (adds the previous pair's label
field) = 212,136 B. The **36,205 B** difference is precisely the part of the separatrix description
explained by inter-frame prediction — real, but 14.6 % of the object, and the correlation table
says it is not landing where pose needs it.

**Scope of this negative, stated per the verdict ladder:** this refutes coupling **through the
per-pair mass channel**, which is the channel that would have changed the byte arithmetic. It does
**not** refute Chasles, and it does **not** refute that a *shared representation* could exist. A
shared `ξ` whose seg and pose benefits land on *disjoint* pairs is entirely consistent with
everything above — it would just mean the shared object buys two things in two places rather than
one thing twice. **FORMULATION-level negative, not FAMILY-level.**

*(Independent read in flight from `ddm_pu2` on the pose side; deliberately uncoordinated.)*

**One thing I take from `pu1` and respect rather than test:** `W` is invariant (no state variable)
while `W_pose` carries `pose_contribution` in its denominator and therefore *diverges* as pose
falls (78,352 → 819,699 B/unit). **Seg's byte-generosity is frozen; pose's grows without bound.**
Any joint allocation must respect that asymmetry — and since the `√`-after-mean makes pose's
allocation exactly linear/greedy while seg's is not, **the two axes do not obey the same allocation
law and must not be waterfilled together.** Nothing in §8 attempts a joint allocation.

---

## §9 WHAT I DID NOT DO

- **No scorer pass.** `ddm_pu1` holds the slot. S1's and S3's falsifiers both need one — queued in §10.
- **I did NOT use `src/tac/boundary_math/contour_codec.py`** and nothing here rests on it. Recording
  the flag raised against it: despite its name and module prose, it **serializes every uint8 label
  in raster order and LZMA-compresses the dense array** — it is a dense-raster LZMA baseline, not an
  explicit boundary-edge codec, and it does **not** establish a 1-D wire representation. Anyone
  reading this memo as licence to cite it as "separatrix coding already built" would be building on
  a false premise. *(Name/mechanism mismatch is an open NO-FAKE adjudication, not mine to close.)*
  Incidentally my own **measured lzma-9e baseline (428,120 B, §5)** is the honest reference for what
  that module's mechanism can actually deliver — **1.69× worse** than the context-model floor.
- **`ru1` atlas absent from this tree** (`.omx/research/ddm_ru1_20260729/atlas_flat.npz` not present).
  Routed around: I used `pc2`'s published per-edge table as the flip-share source and supplied the
  interface-length denominator myself. Per-flip joins (dist_bin, flicker) were not re-derivable here.
- **S2's `ρ` unmeasured** — needs decoded frames (`inflated/0.raw`, 3,492.7 MB); scorer-free but
  large. Highest-value scorer-free follow-on.
- **`pc2` vehicle-transfer untested** (`tb1 ep399` → `cx1`, +10.87 % flips).
- **Temporal H2 assumes pair `i−1` predicts pair `i`.** All 600 pairs are from a single video
  (`upstream/videos/` = `0.mkv` only, `public_test_video_names.txt` = 1 line), so there is no
  cross-video boundary to corrupt it — but pairs are non-overlapping, so the predictor is 2 frames
  back, not 1. The 14.6 % H1→H2 gain is real but modest and I did not pursue it.

---

## §10 QUEUED GATES (exact commands owed, not run)

**G1 — S1 falsifier, one n600 scorer pass.** Build `cx1` with interior-cell token precision at
`q=0.25` (boundary cells = the 74,606 at `c=16` from
`.omx/research/ddm_sx1_cell_occupancy_n600.json`), byte-close, then:

```
.venv/bin/python -m tac.contest_score --archive <cand>.zip --n-pairs 600 --device cpu
```
PASS iff `Δflips < 139,584` **and** `Δd_pose ≈ 0`. Predicted `ΔS_rate = −0.11831` at `q=0.25`.

**G2 — S2 hit-rate `ρ`, scorer-FREE.** Run a generic contour extractor on decoded `inflated/0.raw`
frames; score against `lstars`; report `ρ` = fraction of the 2,551,382 boundary px located within
±1 px. `ρ > 0.5` makes S1's occupancy map effectively free.

**G3 — vehicle transfer.** Re-derive the per-edge flip table on `cx1` (not `tb1 ep399`) and
re-join to §3's interface lengths. Confirms or breaks the `flips ∝ length` law at the live point.

---

## §11 ASSUMPTION LEDGER (charter §3)

| # | assumption | class | status |
|---|---|---|---|
| A1 | `d_seg` = argmax disagreement, last frame only | VERIFIED_VIA_SOURCE_INSPECTION | `modules.py` quoted |
| A2 | head is affine ⇒ power diagram, rank 4 | VERIFIED_VIA_SOURCE_INSPECTION + numeric | argmax agreement 1.000; σ₅=3.7e-16 |
| A3 | `lstars` cache == live `argmax SegNet(GT)` | **upgraded R2 →** VERIFIED_VIA_EMPIRICAL_ANCHOR *(cross-cache)* | class areas reproduce the independent n96 anchor to <0.3 pp on 4 of 5 classes; class order **self-detected** and matches canonical comma10k, not luma-sort (§3). Not a live scorer re-run, so short of VIA_SOURCE_INSPECTION. |
| A4 | `pc2` edge shares transfer `tb1`→`cx1` | ASSUMED_AWAITING_VERIFICATION | +10.87 % flips; G3 |
| A5 | tokens = 341,295 B and are image content | VERIFIED_VIA_SOURCE_INSPECTION | `ddm_cx1_pj2` §; drives §8.1 |
| A6 | context-model entropy ≈ achievable code length | INFERRED_FROM_DOMAIN_LITERATURE | but bracketed by MEASURED lzma at 1.69× |
| A7 | interior precision can be reduced without seg/pose harm | ASSUMED_AWAITING_VERIFICATION | **this is S1's whole risk**; G1 |

**Verdicts resting on A3/A4/A7 are PROVISIONAL.** S1's ΔS is PROVISIONAL pending G1.
§0's floor (0.5349 B/flip) rests only on A1/A2/A3/A6 and is the most solid claim here.

**Never prematurely kill:** S3 is refuted as a *formulation* (flat paint), not as a *family* —
a learned decode-side painter conditioned on `L*` is untested and stays open. Likewise S5 is
refuted as a *formulation* (allocate ∝ ‖wᵢ−wⱼ‖); the *family* "allocate by an exact head-derived
quantity" survives via the margin/tie-distance route, which is the one the data supports.

### 11.1 Review rounds — counter did NOT reach 3 clean

Per charter §3 the counter resets on any round that finds an issue. It reset twice; I report the
honest state rather than declaring a seal.

| round | findings | reset? |
|---|---|---|
| **R1** | (a) **my own S1 was interior-DELETION — wrong, tokens are image content** (§8.1); (b) §6 apparatus ceiling used *both* scorers (94.3 MB) not segnet (38.5 MB) → I had 145×, truth is 59.46× and the charter was right; (c) occupancy-map cost mis-added; (d) carrier cell-scale unresolved (§8.2) | **YES** |
| **R2** | (e) **"beats the published contour floor" was apples-to-oranges** — per crack-step we are at 1.10 bits, *inside* the band (§5); (f) "Law: flips ∝ length" overstated an n=5 cross-vehicle share comparison → downgraded (§3); (g) A3 upgradeable via cross-cache class-area agreement → upgraded | **YES** |
| **R3** | (h) description/realization split is a *conceptual budget*, not architecturally separable (§5.1); (i) `PX`/`W` were inherited, not derived → verified from `frame_utils.py`; (j) camera→scorer stride is non-integer (§8.2) | **YES** |

**Counter stands at 0 clean passes.** Three rounds, three finding-sets — which by the project's own
rule ("a round that FOUND things is evidence more exist") means this memo should be read as
**round-3 output, not sealed work.** The claims most likely to still be wrong, in order:
S1's ΔS (rests on unpinned carrier geometry + A7), then the `flips ∝ length` relation (n=5,
cross-vehicle), then §2.5's precision/displacement shares (inherit `pc2`'s A4).

The claim I would defend hardest, and the one that should survive: **§5's 253,341 B / 393×
concentration / 0.5349 B/flip.** It rests only on A1, A2, A3, A6 — three of which are now verified
by source inspection or cross-cache anchor — and it is the answer to the question that was asked.

---

## NEXT-IF-RESUMED

1. **G2 first** — it is scorer-free and it is the gate on whether S2 (and therefore S4) is real.
2. **G1** the moment the scorer slot frees. Single largest derived rate win in this memo.
3. **The 23.3 % displacement class (§2.5) has no carrier in this table.** It needs a per-pair
   positional DOF. That is a genuinely open design slot and it is not separatrix-shaped.
4. Diff against sister arm `ddm_de1` (codex, independent derivation from `upstream/` alone).
   Convergence on §2.2/§2.3 = corroboration; divergence = the finding.
