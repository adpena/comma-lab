---
schema: ddm_br1_basis_race_and_drop_surface.v1
date_utc: 2026-08-03
arm: ddm_br1 (rate axis — race the gt1 basis/coefficient law; measure the drop surface)
lane_id: "lane_ddm_br1_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: SEE-PER-ROW
axis: "[byte-closed rate, scorer-free]. NO SegNet or PoseNet forward/backward was fired.
  The evaluator slot (held by ddm_pu1) was not requested and not touched. Every byte figure
  is produced by running the real encoder/decoder on the real live-best archive bytes."
consumes:
  - upstream/modules.py, upstream/evaluate.py, upstream/frame_utils.py  (primary, §0)
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_cx1_pj2ix2_archive.zip
  - .omx/research/ddm_gt1_upstream_gt_unmined_inventory_20260803.md
  - .omx/research/ddm_ix2_renderer_split_and_decoder_20260802.md
  - .omx/research/ddm_cx1_pj2_container_compose_20260802.md
consumers: [MAIN]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_br1 — the basis race closes the transform axis; the drop surface was already measured

**Headline in one line:** BR1-1's falsifier **FIRED** (every reversible re-expression of the
coefficient lattice is within **0.83%** of doing nothing — basis is not the lever), and my
charter's premise that the drop surface *"has never been measured at ANY drop level"* is
**FALSE** — the live base **is** `cell_drop50`, and both directions off it are measured and
dominated. Two negatives that close work, one new byte leg that unblocks half of `na1`'s P0-2.

**NEXT-IF-RESUMED** — see §9. Written incrementally; every section stubbed at step 1.

---

## §0 — DERIVED FROM PRIMARY SOURCES BEFORE READING OUR RECEIPTS

**Honesty about ordering (required, because it bounds what this section is worth).** The depth
amendment reached me after I had already read `ddm_gt1` §0–§4, `ddm_ix2` §0–§3 and `ddm_cx1`
§0–§1. So this is **not** a prior-free derivation and I will not present it as one. What I did
do is derive D1–D6 below from `upstream/{modules,evaluate,frame_utils}.py` and the score
arithmetic alone, writing each down *before* opening the vehicle's own encoder, and then diff.
Items already in hand before deriving are marked **[PRIOR]**; items reached here are marked
**[DERIVED]**; the diff against our receipts is §0.7.

### D1 — the sufficient statistic, read off the two `preprocess_input` bodies [DERIVED]

`upstream/modules.py`:

- `SegNet.preprocess_input` (`:107-109`): `x = x[:, -1, ...]` then
  `interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')`.
- `PoseNet.preprocess_input` (`:71-74`): `rearrange` → the **identical**
  `interpolate(..., size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')`
  → `rgb_to_yuv6` → `rearrange`.

Two consequences that fall out immediately and that I checked rather than assumed:

1. **Both scorers apply the SAME operator `D`** — `interpolate` to `(384, 512)`, bilinear, and
   `segnet_model_input_size` is literally the target of both calls. There is no
   "seg-resize vs pose-resize" difference to exploit. (This is the 2026-08-03 correction; it is
   confirmed at source here, independently.)
2. **`rgb_to_yuv6` is applied AFTER `D`, not before** (`:73` interpolates, `:74` converts).
3. **`frame_0` is exactly invisible to SegNet.** SegNet consumes `x[:, -1, ...]` only.

Therefore the score depends on the entire 3.66 GB of delivered frames **only through**, per pair:

| quantity | domain | what the score does with it |
|---|---|---|
| `L = argmax SegNet(D(f1))` | `{0..4}^(384×512)` | counts disagreements with `L*` |
| `p = PoseNet(yuv6(D(f0)), yuv6(D(f1)))[:6]` | `R^6` | squared distance to `p*` |

**Scale of the latitude this buys.** Delivered payload is
`600 × 2 × 874 × 1164 × 3 = 3,663,417,600 B`. The *scored* statistic is `600 × 2` scalars.
Even the *pre-reduction* statistic (`L` and `p` per pair) is `600 × (196,608 5-ary symbols + 6
floats)`. This is why the problem is an **indirect / CEO rate–distortion problem** and not a
video-coding problem: we are not coding the video, we are coding a description whose
reconstruction is scored through a frozen non-invertible operator. Classical R(D) intuitions
about "how many bits does this image need" are the wrong curve.

### D2 — the two-part code, and precisely where "free" enters [DERIVED, law is PRIOR]

MDL: `L(archive) = L(model) + L(data | model)`. Rule-118 sets `L(model) = 0` **iff the model is
generic** — a property of the frozen operators, not of this clip. So the whole design objective
of the rate axis is: *move as much of the description as possible into the generic model, and
keep only the projection onto it counted.* That is `ddm_gt1`'s law restated as an objective
rather than an observation, and it is the thing this arm races.

### D3 — the counted stream is ONE object, and I can name its shape [DERIVED from the receiver]

Reading `render_frame1_float` (`src/tac/optimization/ddm_tr1_runtime.py:1300`):

```
decode_token_grid(parsed, p)  ->  latent
  conv2d + bias + gelu
  4 × [ repeat ×2 in H and W ; conv2d + bias ; gelu ]      (24→384, 32→512, i.e. ×16 = 2^4)
  conv2d + bias ; sigmoid × 255                            ->  [384, 512, 3]
```

and the decoded token array measured directly out of the live archive is
**`(600, 24, 32, 4)` uint8, values `0..15`, all 16 symbols used.**

So the vehicle is *already* exactly the shape of the gt1 law:

| part | bytes in `cx1` | counted? | why |
|---|---:|---|---|
| conv weight **bank** (the basis) | **0** | FREE | regenerated from `selector.lotto_seed`; generic algorithm |
| lottery **mask** (which basis vectors) | 2,781 | counted | fitted to this clip |
| gain/bias table | 492 | counted | fitted |
| **tokens** (the coefficients) | **341,295** | counted | this clip's projection |
| decode program + pose | 9,323 | counted | — |

`600 × 24 × 32 × 4 = 1,843,200` symbols at 4 bits raw = **921,600 B**; shipped **341,295 B** =
**1.4812 bits/symbol**. The coder is already extracting 2.52 bits/symbol of structure.

**The coefficients are 96.5% of the archive.** Every other member together is 12,405 B = 3.5%.
Any rate finding that does not touch the token lattice is arithmetically capped at 3.5%.

### D4 — the pre-registered claim this arm exists to test [DERIVED]

> Entropy is a property of the **(stream, model)** pair, not of the stream. A general-purpose
> coder supplies a *generic* model (order-k context + LZ matches). Our layout already encodes
> the model we have — which is why swapping coders returns ~0. **The remaining lossless lever is
> therefore a change of BASIS on the coefficient lattice, not a change of coder.**

Crucially this can be done **exactly and scorer-free**: a *reversible integer* transform `T` on
the 4-bit lattice is bijective, so `T⁻¹` at decode reproduces the tokens bit-for-bit, the
rendered frames are byte-identical, and `d_seg`/`d_pose` are invariant **by construction** —
the same argument that made the `ix2` container safe. `T` is a fixed algorithm ⇒ generic ⇒ free.

Candidate generic bases, all reversible on integers, all recomputable by `inflate.py` at zero
counted bytes:

| id | basis | rationale |
|---|---|---|
| `IDENT` | shipped cell-major nibble | the incumbent / control |
| `TDPCM` | temporal difference across the 600 pairs | pairs are consecutive in time |
| `HAAR-S` | reversible Haar (S-transform) lifting on the 24×32 grid | spatial smoothness |
| `LG53` | Le Gall 5/3 integer lifting (JPEG2000 reversible) | better spatial decorrelation |
| `CHAN` | reversible transform across the 4 channels | channel correlation |
| products | separable combinations of the above | — |

**Falsifier (from the charter, pre-registered before measuring):** if capture-per-coefficient-byte
lands **within 2×** across bases, basis choice is NOT the lever at our operating point, and the
gt1 law — while true — is not actionable here.

### D5 — the drop threshold is exactly `W`, and that makes every drop candidate one number [DERIVED]

```
ΔS = −25·Δb/DEN + 100·Δf/(196,608·600)
   ⇒ a drop is profitable  ⇔  Δb/Δf  >  4·DEN/(196,608·600)  =  W
W = 4 × 37,545,489 / 117,964,800 = 1.2731082153320312 B/flip   (exact; pure constants)
```

So the drop surface is fully priced by **bytes saved per flip caused**, and the byte half is
scorer-free. For any drop level I can state the *exact* bytes saved and hence the **flip budget**
`Δb / W` — the number of flips below which that level is profitable. That is a complete
pre-registration of the gate without holding the scorer slot.

### D6 — pre-registered prediction on the drop surface [DERIVED]

Naive reasoning says dropping the token LSB (16 → 8 levels) saves 1 of 4 bits = 25% of the
stream. **I predict markedly less than 25%,** because the shipped stream is already at 1.4812
bits/symbol — below 3 bits — so the coder has already spent most of the width. Written down
before measuring. Answered in §3.

### §0.7 — DIFF against our receipts

| derived here | our receipts | verdict |
|---|---|---|
| shared `D`, `yuv6` after `D`, `f0` seg-invisible | `m86`, `pz1`, `ddm_gt1` §3 | **independent corroboration** |
| entropy is a (stream, model) property ⇒ coder is not the lever | `ix2`: brotli **+5 B WORSE** than stored on tokens; layout beat coder **5,184-to-5** | **corroboration, and ours is the stronger form** — ours is measured, mine was predicted |
| the lever is a reversible change of basis on the lattice | **not present in any receipt I can find** — `ix2` raced *layouts* (SoA/AoS, nibble, bit-plane, delta+zigzag+varint, RLE, adaptive context, colex) but every row is a **permutation or a re-framing**, never an invertible *transform* of the values | **THE GAP.** §2 |
| drop threshold `= W` | ~~new framing~~ **REFUTED on round 3**: `ja1` carries `rate_break_even.tokens_B_per_flip = 1.273`; `ddm_costate_organ.py:1293,1325` gates on it live; `ba31` prices a surface against it | **prior, not mine** |

**The gap is the finding of §0 — stated with a correction I owe (round-1 self-review).** My first
draft of this row said *"`ix2` has no transform row at all."* **That is wrong and I am correcting
it rather than deleting it.** The incumbent **is itself a transform**: `_factor_mode_delta`
subtracts a per-cell temporal **mode** and codes the mod-16 residual (the r7 factorisation,
`ddm_ix2_archive_container.py:208`). What is true, and is the actual gap, is narrower: **`ix2`'s
raced matrix explores *layout* (permutation) and *coder* at a FIXED transform; the transform axis
itself was inherited, never raced.** That is what §2 races.

---

## §1 — THE VEHICLE, EXACTLY (every figure cross-checked against an absolute)

`/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_cx1_pj2ix2_archive.zip`,
**353,808 B**, sha `1d3ab694…`. One ZIP member `0.bin`, STORED, 353,700 B.

| part | bytes | share | cross-check |
|---|---:|---:|---|
| **token member** (the coefficients) | **341,295** | **96.46%** | equals `ddm_cx1`'s table exactly; re-encoding the lattice through the live encoder reproduces `341295` |
| joint container (config 36 + renderer 3,266 + selector 535 + pose 8,752, shared coder) | 12,405 | 3.51% | 353,700 − 341,295 = 12,405 ✓ |
| ZIP framing | 108 | 0.03% | 353,700 + 108 = **353,808** ✓ |

**Live constants, every one recomputed here, never re-typed:**

| symbol | value | provenance |
|---|---|---|
| `S` (`cx1`) | 0.8264972 | recomputed from components |
| gap to the PR130 bar (0.172141) | **0.6543562** | 0.8264972 − 0.172141 |
| 1% of gap | **9,827.3 B** | `0.006543562 × DEN/25`; charter's 9,827.2 reproduced |
| `W` | **1.2731082153320312 B/flip** | **DERIVED**, not measured: `4·DEN/PX` from contest weights alone (`ddm_de1` DE1-1 — a codec cannot move `W`, only its own achieved B/flip) |
| `d_seg` | **0.004311794704861111** | `ddm_pz1_dseg_n600_cx1_20260803.json`, positive control passes |
| total flips `F` | **508,640** | `d_seg × 117,964,800` |
| `d_pose` | 0.00255143 | pose term 0.1597320 = √(10·d_pose) ✓ |
| `dS/d(d_pose)` | **31.30** | `5/√(10·d_pose)` |

> **ROUND-1 CATCH, and the most important line in this memo.** `ddm_cx1`'s row reads
> `seg 0.4311790`. **That is the score TERM `100·d_seg`, not `d_seg`.** I spent a working
> paragraph on the hypothesis *"43% of pixels are already wrong, so degrading them further is
> free"* before cross-checking against `pz1`'s absolute. **0.43% are wrong, not 43%** — the
> hypothesis is dead and the whole drop surface reprices by 100×. Caught only by checking a
> derived quantity against an independently-measured absolute, which is exactly the discipline
> the charter names.

**The arithmetic cap this puts on everything else:** the token member is 96.46% of the archive.
Any rate finding that does not touch the coefficient lattice is capped at **3.5%**.

---

## §2 — BR1-1: THE BASIS RACE. **The incumbent wins. The falsifier fires.**

### What was raced, and the denominator

**11 predictors × 6 layouts × 4 coders = 264 evaluations** (reported as 66 predictor×layout
bests). Every candidate is a **reversible mod-16 change of basis** on the (600,24,32,4) lattice,
so the tokens decode bit-for-bit, the rendered frames are byte-identical, and `d_seg`/`d_pose` are
invariant **by construction** — the same argument that made the `ix2` container safe. Every
candidate got the **same** 4-coder race (stored / deflate / brotli / lzma) the incumbent gets, so
this is basis-vs-basis, not coder-vs-coder. Code: `scratchpad/br1_basis_race.py`.

| basis | H₀ (b/sym) | zero frac | best bytes | best layout | vs incumbent |
|---|---:|---:|---:|---|---:|
| **`MODE*` (incumbent)** | 2.1888 | 62.58% | **341,295** | cell-major RCKP | — |
| `MEDIAN` (temporal median) | 2.1767 | 61.50% | 341,996 | RCKP | +701 |
| **`IDENT` (no transform at all)** | 3.4565 | 30.17% | **344,129** | KRCP | **+2,834** |
| `MODE+TPREV` | 2.2731 | 60.86% | 367,466 | RCKP | +26,171 |
| `TPREV` (temporal DPCM) | 2.2772 | 60.81% | 368,650 | RCKP | +27,355 |
| `MODE+CHAN` | 2.4723 | 58.27% | 387,846 | RCPK | +46,551 |
| `CHAN` | 3.7409 | 24.13% | 396,077 | KRCP | +54,782 |
| `MODE+SW` | 2.5999 | 55.36% | 425,727 | RCKP | +84,432 |
| `SW` (spatial west) | 3.7374 | 24.03% | 429,328 | KRCP | +88,033 |
| `SN` (spatial north) | 3.8267 | 20.47% | 438,336 | RCKP | +97,041 |
| `MED` (JPEG-LS LOCO-I) | 3.8194 | 20.14% | 456,488 | RCKP | +115,193 |

### THE FALSIFIER, ANSWERED EXPLICITLY

> Pre-registered: **if capture-per-coefficient-byte lands within 2× across bases, basis choice is
> NOT the lever.**

**Full spread best→worst = 456,488 / 341,295 = 1.337×. WITHIN 2×. THE FALSIFIER FIRES.**

And the honest form is far sharper than 2×: **the entire transform family spans 0.83%** between
the incumbent and *doing nothing at all* (`IDENT`, 344,129 B). Every transform that is not
mode-subtraction makes the stream **worse**, several catastrophically. There is no basis in this
family worth switching to, and the incumbent's own transform — the thing `ix2` inherited without
racing — is worth **2,834 B = 0.83% of the token member = 0.80% of the archive = ΔS 0.0018870 =
0.29% of the gap**. *(Round-2 catch: an earlier draft of this line said "0.29% of the archive",
conflating the archive share with the gap share. 0.80% and 0.29% are different denominators.)*

**Verdict scope: FORMULATION.** This closes *reversible transforms of the shipped token lattice*.
It does **not** close the gt1 law's strongest reading, where the "basis" is the **decoder CNN
itself** (free-from-seed). Racing that requires retraining and the scorer slot; it is queued in §6.
**INSTANCE < FORMULATION < FAMILY: nothing here kills the basis family.**

### Two mechanism findings that explain *why* the falsifier fired

**M1 — the lattice has no spatial correlation to exploit.** Every spatial predictor **raises** H₀
(3.4565 → 3.74/3.83) and every spatial residual codes worse than the raw tokens. Neighbouring
cells of the 24×32 grid are effectively independent. The only structure in the lattice is
**temporal**, and mode-subtraction already takes it.

**M2 — the compressibility is MATCH structure, and it is nearly all free-riding zeros.** Measured
exactly (`scratchpad/br1_refine.py`):

| | symbols | shipped | order-0 bound | ratio |
|---|---:|---:|---:|---|
| ALL residual symbols | 1,843,200 | 341,295 B | 504,291 B | coder beats order-0 by **1.4834×** |
| **LIVE symbols only** | 916,800 | **339,956 B** | 377,100 B | coder beats order-0 by **1.1093×** |

**1,544 of 3,072 units (50.3%) are exactly DEAD** — temporally constant across all 600 pairs. They
contribute **zero** payload: dropping every one of them saves **0 B** (measured: least-active-first
at 50% → 341,295 B unchanged). Strip them and the coder's apparent 1.48× advantage over order-0
collapses to **1.11×**. **The live coefficients are within 9.85% of their own order-0 entropy.**

**This refutes an idea before it was built.** An explicit 3,072-bit support map (≈384 B) coding
only the 1,528 live units would **cost ~384 B and save ~0**, because brotli already LZ-copies the
dead zeros for free. Measured, not argued; zero extra runs.

**M3 — the incumbent's own transform is the reason `IDENT` is close, not far.** `MODE*` drops H₀
from 3.4565 to 2.1888 — a 37% cut in symbol entropy — and buys only **0.83%** of bytes. That is
the cleanest possible statement of the depth amendment's point: *entropy is a property of the
(stream, model) pair*, and the coder's LZ model was already capturing what mode-subtraction makes
explicit. The two models overlap almost completely.

**Diff against `ddm_ix2` (independent, not adopted):** `ix2` measured `346,478 raw → 346,483
brotli` = coder redundancy closed **for that serialization**, and layout beating coder
**5,184-to-5**. I reach the same conclusion by a different route and extend it: not only is the
coder axis closed, the **transform axis is closed too**, and the residual leverage `ix2` pointed
at ("representation/modeling") is **not reachable by any reversible re-expression of this lattice**.

---

## §3 — THE DROP SURFACE

> ## ROUND-3 CATCH — **MY CHARTER'S PREMISE IS FALSE, AND SO WAS MY HEADLINE**
>
> My charter states: *"the coder × drop surface has never been measured at ANY drop level"* and
> called it *"embarrassing that it is open."* **It is not open. It was measured on 2026-07-30 and
> the result is banked and ADOPTED.** Found by `tools/corpus_query.py` on round 3 — after I had
> already written this section as a first measurement.
>
> **1. The live base IS a drop level.** I reported "1,544 of 3,072 units (50.3%) are dead" as a
> structural discovery. It is not a discovery. Grouped by cell the histogram is
> **384 cells with 0 live channels, 377 with all 4, 7 partial** — i.e. **exactly 384/768 = 50.00%
> of cells fully dropped**. That is `ddm_gr1`'s **`cell_drop50`**, already selected as the seg+rate
> **knee** and already in the shipped archive. I re-derived an adopted decision and nearly shipped
> it as a finding. This is the *re-anchor ≠ discovery* failure my own index warns about.
>
> **2. Both directions off the knee are MEASURED and DOMINATED** (`ddm_ba31` §, from `gr1`):
> restore **+0.047 S @ +80,615 B**; drop-more **+0.052 S @ −81,406 B**.
>
> **3. Repricing `ba31`'s drop-more row into B/flip — the number my §4 needed and could not
> measure.** Rate gain at −81,406 B is 0.054205 S; net is +0.052 S; so the distortion cost was
> **0.106205 S**. Attributed entirely to seg that is Δd_seg 0.0010620 = **125,284 flips**, i.e. a
> realized **0.6498 B/flip against `W` = 1.2731 — DOMINATED BY 1.96×.** To break even the flip
> cost must fall **49.0%**, to 63,943.
> *Caveat carried from `ddm_na1`: `ba31`'s row **does not split seg from pose**. Since
> `dS/d(d_pose) = 31.30`, an unknown share of that 0.106205 S may be pose, which would make the
> seg-only B/flip better than 0.6498. The 1.96× domination is therefore an **upper bound on how
> dominated it is**, and splitting it is `na1`'s open P0.*
>
> **4. Axis B is also already measured, and negatively.** `ddm_gc6` S2: *"measured
> `nu_max_tolerable_q=0` (no token certified free) and uniform snapping hurts monotonically."*
> Uniform alphabet coarsening is exactly my Axis B. **It has a measured monotone-harm verdict.**
>
> **What survives, and it is the honest remainder:** the basis race (§2) is genuinely new and its
> falsifier fired; the live/dead *entropy decomposition* (M2) is new and explains why the coder
> axis is shut; S1/S2 are new and are direct inputs to `#766`; the support-map refutation is new;
> and §3.4 below delivers the **byte leg of `na1`'s open P0-2** scorer-free. The rest of this
> section is a **re-measurement of the byte side of a known surface**, and is labelled as such.

### §3.0 — what was actually new here, stated once

| finding | new? | why |
|---|---|---|
| basis race, 264 evaluations, falsifier fired (§2) | **NEW** | no receipt races *transforms* of the lattice; `ix2` raced layout+coder at a fixed transform |
| live-vs-dead **entropy decomposition** (M2) | **NEW** | the 1.483× coder-vs-order-0 advantage is 1.109× once free-riding dead zeros are removed |
| S1 byte-yield flat & activity-uncorrelated | **NEW** | direct input to `#766`; retroactively justifies `gr1` ranking by seg-sensitivity not bytes |
| S2 negative individual marginals | **NEW** | brotli non-monotone in the drop |
| support-map refutation (70 B cost, 0 B saved) | **NEW** | closed before building |
| `cell_drop63/70/80` **byte leg** (§3.4) | **NEW** | supplies half of `na1` P0-2 scorer-free |
| `cell_drop35` is unmeasurable from the archive | **NEW** | hard limit on P0-2's restore leg |
| "50.3% of units are dead" | **NOT NEW** | = `gr1` `cell_drop50`, adopted |
| unit-drop / alphabet-drop being profitable | **NOT NEW, and NEGATIVE** | `ba31` + `gc6` S2 |

**Both drop axes are FORMAT-FREE and RECEIVER-FREE — verified, not assumed.**
`decode_token_grid` (`ddm_tr1_runtime.py:1213-1227`) computes `codes/(levels−1)·2−1` with
`token_quant_levels` from the selector. Using a *subset* of the 16-symbol alphabet, or making a
unit temporally constant, produces a valid lattice that re-encodes through the **unchanged**
encoder and decodes through the **unchanged** decoder to exactly the array handed in. Proved on
the real bytes for the control, a 320-unit drop and an 8/16 alphabet — **all three exact**
(`scratchpad/br1_verify.py`). **No container change, no receiver change, no new format gate.**
This matters because the sister `−2,781 B` SMEVR row is *blocked* on a format change; this surface
is not.

**Axis A — unit drop** (the charter's 768 cells × 4 rungs = 3,072 waterfill units): replace a
unit's 600 temporal values with its own mode.
**Axis B — level drop**: coarsen the alphabet to a sublattice of {0..15}.

### The measured curves

| candidate | saved B | ΔS(rate) | % of gap | flip budget | % of `F` | or Δd_pose budget |
|---|---:|---:|---:|---:|---:|---:|
| A drop 5 units | 1,387 | −0.00092 | 0.14% | 1,089 | 0.21% | 2.96e−05 |
| A drop 20 units | 4,957 | −0.00330 | 0.50% | 3,894 | 0.77% | 1.07e−04 |
| A drop 80 units | 20,117 | −0.01340 | 2.05% | 15,801 | 3.11% | 4.46e−04 |
| A drop 160 units | 40,615 | −0.02704 | 4.13% | 31,902 | 6.27% | 9.37e−04 |
| **A drop 320 units** | **80,078** | **−0.05332** | **8.15%** | 62,900 | 12.37% | 1.99e−03 |
| B alphabet 12/16 | 51,586 | −0.03435 | 5.25% | 40,520 | 7.97% | 1.22e−03 |
| **B alphabet 8/16** | **109,553** | **−0.07295** | **11.15%** | 86,052 | 16.92% | 2.86e−03 |
| B alphabet 4/16 | 212,806 | −0.14170 | 21.65% | 167,155 | 32.86% | 6.53e−03 |
| B alphabet 2/16 | 269,916 | −0.17973 | 27.47% | 212,013 | 41.68% | 8.97e−03 |

Read the budget columns as the **pre-registered gate**: the last two columns are the *entire*
error budget, spent on seg alone **or** on pose alone. The joint condition is

```
100·Δd_seg  +  [√(10·(d_pose+Δd_pose)) − √(10·d_pose)]  <  25·Δb/DEN
```

and because `dS/d(d_pose) = 31.30`, pose can eat the budget fast: `Δd_pose = 1e−4` alone costs
**0.00313 S**, which is the whole yield of a 20-unit drop. **A token change moves BOTH frames**
(`frame_0 := a·warp(frame_1)+b`), so no drop candidate is seg-only.

### Three structural findings on the drop surface

**S1 — byte yield per unit is ~flat and nearly uncorrelated with activity.** Exact individual
marginals across the live range: min **−58**, median **196**, mean **211**, max **472** B. The
most-active unit (activity 542) yields **186 B**; a unit at rank 664 (activity 490) yields
**472 B**. **Consequence for the #766 waterfill: rank units by FLIP damage, not by bytes — the
byte side is essentially a constant ≈211 B/unit and carries almost no ranking information.**

**S2 — two live units have NEGATIVE marginals.** Dropping them makes the stream **bigger**
(−58 B, −19 B): removing their values destroys match structure elsewhere. Brotli is not monotone
in the drop. A greedy one-at-a-time waterfill will wrongly reject profitable units.

**S3 — drops are mildly SUPERadditive (measured on the SAME units, 1.0206×).** Sum of the 20
individual marginals = 4,857 B; the group drop of those same 20 = **4,957 B**. So greedy pricing
slightly *under*states group yield — additivity is safe to within ~2%.

> **ROUND-1 CATCH #2.** I first wrote S3 as *subadditive*, from comparing a sample stratified over
> all 1,528 ranks against a top-320 group. **That is the prefix-vs-population trap** — two
> different populations. Re-measured on the identical 20 units, the sign flips. Recorded because
> the trap caught me inside the very memo that cites it.

### §3.4 — the byte leg of `ddm_na1`'s open P0-2, delivered scorer-free

`na1` P0-2 asks for *"n600 realized d_seg for `cell_drop35` and `cell_drop63` — resolve the knee
that selected the live base"*, worth up to 10.11%. **The byte half of that gate is measured here,
exactly, on the real lattice, using `gr1`'s own cell grain (768 cells, not my 3,072 units):**

| level | cells dropped | token member | **archive** | saved B | ΔS(rate) | flip budget |
|---|---:|---:|---:|---:|---:|---:|
| `cell_drop50` (the live base) | 384 | 341,295 | 353,808 | 0 | — | — |
| **`cell_drop63`** | 484 | 268,751 | **281,264** | 72,544 | **−0.04830** | 56,982 |
| `cell_drop70` | 538 | 222,211 | 234,724 | 119,084 | −0.07929 | 93,538 |
| `cell_drop80` | 614 | 152,415 | 164,928 | 188,880 | −0.12577 | 148,361 |

> **`cell_drop35` CANNOT be measured from the shipped archive, and this is a hard limit, not an
> omission.** `cell_drop35` is the *restore* direction: it needs token values for cells the live
> base has already zeroed. **That information is not in the archive** — the dropped cells carry
> only their mode. Measuring the restore leg requires the pre-drop lattice or a retrain. Whoever
> runs P0-2 should know that its two halves have **different costs**: drop-more is scorer-only
> (byte leg supplied above), restore is scorer **plus** a source artifact we may no longer hold.

**`cell_drop63` is the cheapest decisive P0-2 read**: −72,544 B (archive would fall to 281,264 B),
so it is admitted iff the joint distortion cost is under **56,982 flips-equivalent**. `ba31`'s
adjacent −81,406 B point realized ≈125,284 → **if the surface is locally linear, `cell_drop63`
lands ≈111,600 flips ≈ 1.96× over budget.** That is a pre-registered prediction; measuring it
either confirms the knee or overturns it.

### What the budget means physically (the honest prior, not a verdict)

One cell of the 24×32 grid maps to ≈16×16 = 256 scorer pixels/pair ⇒ ≈153,600 pixels over 600
pairs. A ≈211 B unit buys **≈166 flips**, i.e. the drop must flip **< 0.108%** of its own
footprint, against an ambient flip rate of **0.431%**. **Required rate = 0.25× ambient.** Given
the measured concentration of flips on the codim-1 separatrix (interiors ≈0), units whose
footprint is interior are plausibly far under this; units on the separatrix are plausibly far
over. **That is a hypothesis with a number attached, not a result.** It is exactly what the
queued gate decides.

---

## §4 — THE PRICED TABLE (the deliverable), in B/flip against `W`

Every row is priced as an **exchange rate** so it is comparable to every other lever, per the
charter. `W = 1.2731 B/flip` is **derived**, and a codec cannot move it — only its own achieved
B/flip.

| candidate | bytes | status | **exchange rate vs `W` = 1.2731** | falsifier | consumer |
|---|---:|---|---|---|---|
| **BR1-1 basis switch** (best non-incumbent) | **+701** (worse) | **rate EXACT, lossless, no scorer needed** | — (costs bytes, buys nothing) | **FIRED**: spread 1.337x < 2x => basis is not the lever | `#766`: stop racing lattice transforms |
| **explicit live-unit support map** | **+70** (worse) | **rate EXACT** | — | **FIRED**: measured 70 B cost, **0 B** saved | CLOSED, do not build |
| `ba31` drop-more | **-81,406** | **MEASURED n600**; seg/pose NOT split (`na1`) | **0.6498 B/flip = 0.51x W -> DOMINATED 1.96x** | needs flips cut **49.0%** to 63,943 | spec, not a kill |
| `ba31` restore | +80,615 | **MEASURED n600** | distortion gain 0.006678 S vs 0.053678 S rate cost | dominated ~8x | spec, not a kill |
| **`cell_drop63`** (`na1` P0-2) | **-72,544** | **byte leg EXACT here; flips OWED** | admitted iff **< 56,982 flips**; linear prediction ~111,600 => ~1.96x over | measure it | P0-2 gate, §6 |
| `cell_drop70` / `cell_drop80` | -119,084 / -188,880 | byte leg EXACT; flips OWED | < 93,538 / < 148,361 flips | measure it | §6 |
| `cell_drop35` (restore leg) | n/a | **UNMEASURABLE from the archive** | — | dropped cells retain only their mode | needs pre-drop lattice |
| B alphabet coarsening (any level) | -51,586 … -269,916 | byte leg EXACT; **`gc6` S2: uniform snapping harms MONOTONICALLY** | no level certified free (`nu_max_tolerable_q = 0`) | already negative | CLOSED at FORMULATION |

**Reading the table:** the two rows that are mine *and* decisive are the top two, and both are
**negative** — they CLOSE work rather than open it. Every drop row is a **specification**
("needs the flip cost cut 49%"), never a kill.

**Typing discipline (per the `#307` correction I was sent):** every row above is a **physical**
change to the delivered frames priced against `W`. **None is a target/mismatch map**, and none of
the B/flip figures above is a coder-of-where-we-are-wrong result — in particular the 0.6498 B/flip
row is a *realized physical* rate, which is the same type as `W` and legitimately comparable to it,
unlike `#307`'s 0.8201 B/flip which codes WHERE the witness is wrong and is **not** a candidate.
The `cell_drop63/70/80` rows have an **exact** rate side measured here and an **owed** flip side.

---

## §5 — WHAT I DID NOT DO, AND THE ONE CLAIM I WILL NOT MAKE

- I did **not** hold or request the scorer slot. No SegNet/PoseNet forward or backward was fired.
- I therefore **cannot** say whether any drop row is profitable. The rate side is exact; the
  distortion side is a pre-registered gate, §6.
- I did **not** race the decoder CNN (the gt1 law's strongest "basis"). Out of scope without the
  scorer.
- **Negative-existence claims — I made two and BOTH were REFUTED by my own round-3 search.**
  Recorded in full because negative-existence is the campaign's #1 false-claim class and I am a
  fresh instance of it:
  1. *"`ix2`'s matrix has no transform row"* — **refuted by source**: the incumbent
     `_factor_mode_delta` **is** a transform. Corrected in §0.7 to the narrower true claim
     (the transform was inherited, never *raced*).
  2. *"no receipt applies `W` as a drop threshold"* — **refuted by corpus**: `ja1` carries
     `rate_break_even.tokens_B_per_flip = 1.273`; `ddm_costate_organ.py:1293,1325` gates on it
     live; `ba31` prices a whole surface against it. Retracted entirely.
  The surviving scoped negative is only: *I found no receipt that measures the byte side of
  `cell_drop63/70/80`* — scoped to `.omx/research/**` via `tools/corpus_query.py`
  (**denominator: index ≈76%, 7,398 of 9,706 documents**; stores consulted research 7,429 /
  equations 869 / memory 2,061 / dag 915 / council 292 / tasks 417 / docs 96) plus direct reading
  of `ddm_ix2_archive_container.py` and `ddm_tr1_runtime.py`. **Not exhaustive.**
- `src/tac/boundary_math/contour_codec.py` was **not** used in the coder race (it is a dense-raster
  LZMA baseline, not a boundary codec — flagged by `ddm_de1`).

## §6 — THE QUEUED GATE (exact commands, for whoever holds the scorer slot)

The four drop candidates need **one** n600 pass each through the real evaluator. Build is
scorer-free and already proved format-free; only the gate needs the slot.

**Honest state of the build path (round-2 correction).** `scratchpad/br1_drop_surface.py`
*measures* bytes; it does **not** persist the dropped lattices, and no emitter yet writes a
dropped lattice back into a `cx1`-shaped archive. The missing piece is small and scorer-free: the
drop is `tokens.reshape(600,-1)[:, units] = base.reshape(-1)[units]` (Axis A) or the `coarsen`
map (Axis B), then re-pack through the existing builder. That builder **does exist and is
verified present**: `tools/cx1_build_ix2_container_archive.py` (+ `tools/cx1_verify_frame_parity.py`).

```bash
# 0. WRITE THE PREDICTION FIRST (pre-registration), then:
# 1. emit the dropped lattice  -- ~20 LOC on top of scratchpad/br1_drop_surface.py, scorer-free
# 2. re-pack through the LIVE encoder cx1 used (grep its argparse before emitting any flag)
.venv/bin/python tools/cx1_build_ix2_container_archive.py --help
# 3. gate n600 through the REAL receiver -- NOT the SSD copy: ddm_cx1 §5 measured that
#    stage_v4d_realized_gate.sh stages a PRE-ix2 receiver that cannot parse a container archive.
#    upstream/evaluate.py --device cpu --submission-dir <dir> --report report.txt
```

**PRE-REGISTERED, write before running:** each candidate is admitted iff
`100·Δd_seg + [√(10·(d_pose+Δd_pose)) − √(10·d_pose)] < 25·Δb/DEN`, with `Δb` from the §3 table.
Report the realized **B/flip against `W = 1.2731`**, and state the baseline (`cx1`, S 0.8264972,
353,808 B) on the delta. A dominated row is a **specification** ("needs the flip cost cut to X"),
not a kill.

## §7 — ASSUMPTION LEDGER (per the 3-clean-pass contract)

| assumption | classification | note |
|---|---|---|
| both scorers share `D`; `yuv6` after `D`; `f0` seg-invisible | **VERIFIED_VIA_SOURCE_INSPECTION** | `modules.py:71-74,107-109` read here |
| token member = 341,295 B = 96.46% of archive | **VERIFIED_VIA_SOURCE_INSPECTION** | parsed from the real archive; sums to 353,808 |
| `d_seg` = 0.004311794 (term 0.4311795) | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | `pz1` n600, positive control passes |
| every raced basis is reversible ⇒ lossless | **VERIFIED_VIA_SOURCE_INSPECTION** | round-trip proved for each headline predictor |
| drop axes are format/receiver-free | **VERIFIED_VIA_SOURCE_INSPECTION** | encode→decode exact on 3 real lattices |
| flip cost of `cell_drop63/70/80` | **ASSUMED_AWAITING_VERIFICATION** | ⇒ those verdicts are **PROVISIONAL** |
| drop-more realizes 0.6498 B/flip (0.51× `W`) | **VERIFIED_VIA_EMPIRICAL_ANCHOR**, re-derived here from `ba31`'s n600 row | but seg/pose **not split** (`na1`) ⇒ 1.96× is an **upper bound on domination** |
| the live base is `cell_drop50` | **VERIFIED_VIA_SOURCE_INSPECTION** | 384/768 cells fully dead, measured on the lattice |
| decoder-CNN basis has no better option | **not tested** | out of scope; no claim made |
| **my charter's premise "never measured at any drop level"** | **REFUTED** | `gr1`/`ba31`/`gc6` measured it 2026-07-30/31 |

**Shared assumption this work operates within, and whether violating it unlocks breakthrough:**
*that the coefficient lattice's shape (600×24×32×4, 16 levels) is fixed.* Every row above is a
re-expression or a decimation **within** that shape. Violating it — a different grid, a different
code width, a different decoder — is precisely the untested direction, and §2's result
(**every reversible re-expression is within 0.83% of doing nothing**) is the strongest available
argument that **the shape, not the coding, is where the remaining rate lives.**

## §9 — NEXT-IF-RESUMED

1. **Do not re-race lattice transforms or coders.** Both axes are closed at FORMULATION scope
   (§2, 264 evaluations; `ix2`'s coder race). The next rate byte is not there.
2. **Do not re-measure the drop surface's ΔS at the `cell_drop50` knee.** `gr1`/`ba31` did it;
   both directions are dominated. **Recall before measuring** — I lost most of an arm to this.
3. **The one gate worth firing is `cell_drop63`** (§3.4): byte leg supplied (−72,544 B, archive
   → 281,264 B), flip budget 56,982, pre-registered linear prediction ≈111,600 flips ⇒ ≈1.96×
   over. It closes `na1` P0-2's drop-more half. Its restore half (`cell_drop35`) is
   **unmeasurable from the archive** — say so in the P0-2 plan rather than discovering it there.
4. **Split seg from pose on `ba31`'s rows** (`na1`'s open P0). At `dS/d(d_pose) = 31.30` the split
   materially changes the 1.96×; it is the cheapest way to learn whether the drop surface is
   dominated by 2× or by much less.
5. **Carry S1 into `#766`:** rank waterfill units by flip damage, never by bytes (byte yield is
   ≈211 B/unit flat, min −58, max 472, uncorrelated with activity); price groups, not singles
   (S2 negatives, S3 superadditivity 1.0206×).
6. **The live direction §2 points at:** every reversible re-expression of the lattice is within
   0.83% of doing nothing, so **the lattice SHAPE — not its coding — is where the remaining rate
   lives.** That needs a retrain and the scorer; it is the successor charter, not a rate-arm task.
7. Artifacts: `/Volumes/VertigoDataTier/pact/ddm_br1_20260803/{br1_basis_race,br1_drop_surface,br1_refine,br1_priced_table}.json`,
   `br1_unit_activity.npy`, `cx1_tokens.npy`. Scripts: `scratchpad/br1_{basis_race,drop_surface,refine,verify}.py`.
