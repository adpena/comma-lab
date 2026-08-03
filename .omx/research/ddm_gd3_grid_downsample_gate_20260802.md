# ddm_gd3 — `grid_downsample`: the rate row survives and is now byte-measured end-to-end; the d_seg half is **unmeasurable at $0 on this vehicle**, and the slot proved it

**Arm:** `ddm_gd3`, resuming `ddm_gd2` (killed mid-Phase-A).
**Axis:** `[macOS-CPU advisory — real evaluator, real archive bytes, n600]`. `score_claim=false`,
`promotion_eligible=false`, `pointer_moved=false`.
**Source row:** `ddm_mt1` §0/§3.1/§5 #1 (`.omx/research/ddm_mt1_menu_triage_79_20260802.md`, commit
`90050890db`). Repo HEAD at run: `efbeb38ce3`.

---

## §0 POINTER HONESTY + THE ANSWER

**Exact contest pointer UNMOVED at `0.1910828242 [contest-CPU]`. No gate fired on a candidate. No
training. No paid dispatch. No `upstream/` edit.**

Anchoring: **LIVE BEST = `dc1_fold`**, archive 360,309 B, sha `9fb9f4e9e460f91c…`,
components seg 0.4311790 / pose 0.2272830 / rate 0.2399150 → **S 0.8983770**. Bar = PR130
**0.172141** @ 191,052 B. **Gap 0.7262365**; 1% of gap = 10,907 B = 0.0072624 S.

**`ddm_gd2`'s question was: "does a no-train ds=32 archive yield a meaningful d_seg?" The answer is
NO — and I spent the slot proving it is NO for the entire class, not just for the literal archive.**

| what | verdict |
|---|---|
| **`grid_downsample` 16→32 RATE row** | **STANDS, and is now MEASURED as an `archive.zip` byte count** rather than tokens + a derivation: **101,636 B**, `ΔS_rate = −0.1722397`, **23.72% of the gap**. Two of mt1's inputs corrected (both in our favour). |
| **`grid_downsample` 16→32 SEG row** | **STILL UNMEASURED, and now proven unmeasurable at $0.** The strongest legal no-train probe returns d_seg **0.524275 = 121.6× the base = 301.9× over the break-even**. |
| **the class** | **Every post-hoc token-field surgery on this vehicle is vacuous as a d_seg instrument.** That closes the $0 d_seg read for mt1 §3.1 rows **1, 2 AND 3** (`grid_downsample`, `code_width`, `token_quant_levels`) at once. |
| **mt1's "ΔS pose = 0" on all three top rows** | **FALSIFIED.** Probe d_pose 0.00517 → **76.19**. The pose carrier is a *fitted artifact of the decoded frame_1*; any token-side change invalidates it. The row costs **1 training run + 1 pose re-solve**, not 1 training run. |

---

## §1 THE BLOCKER `ddm_gd2` DIED HOLDING — recovered, confirmed, and then generalized

### 1.1 The structural half (DERIVED from source, `ddm_tr1_runtime.py`)

`grid_downsample` does **not** re-parameterize the shipped renderer. It **changes the architecture.**

`_conv_shapes` (`src/tac/optimization/ddm_tr1_runtime.py:274-289`) sets
`n_upsample = round(log2(grid_downsample))` and emits one `up{k}` conv per upsample. MEASURED by
calling it on the shipped selector and on a ds=32 selector:

```
ds=16  conv0(24,3,3,4) up0 up1 up2 up3 (24,3,3,24) head(3,3,3,24)   — 6 convs
ds=32  conv0(24,3,3,4) up0 up1 up2 up3 up4 (24,3,3,24) head(3,3,3,24) — 7 convs
```

The extra `up4` has **no trained mask, gain, or bias anywhere in our custody.** So a no-train ds=32
archive cannot be *built* without inventing a whole layer, and its d_seg would measure "trained
decoder plus one random layer" — damage that is unbounded and unrelated to grid resolution.
**gd2 was right to stop.**

### 1.2 The empirical half (MEASURED — this is what the slot bought)

The structural argument only kills the *literal* ds=32 archive. So I built the **strongest legal
no-train probe that exists** and gated it at n600:

> **Decoder-fixed coarse-description probe.** Take the shipped codes `(600,24,32,4)`, 2×2
> median-pool to `(600,12,16,4)` — the exact information content of a ds=32 description — then
> **block-replicate back onto the legal 24×32 ds=16 lattice.** Geometry legal, selector unchanged,
> **the trained 6-conv renderer untouched**, lossless round-trip through the real SMEVR coder,
> decodes through the real `inflate_runner_v4d.Decoder`.

I argued *a priori* this was a conservative upper bound on Δd_seg: it omits the two effects that
help a native run (re-optimizing the coarse field; the free extra conv) and adds one that hurts
(input-distribution shift). **The direction of that argument was right. The magnitude made it
vacuous.**

**MEASURED, n600, real `evaluate.sh`, real archive bytes, device=cpu:**

| | d_seg | d_pose | archive B | seg | pose | rate | **S** |
|---|---:|---:|---:|---:|---:|---:|---:|
| base `dc1_fold` | 0.00431179 | 0.00516576 | 360,309 | 0.431179 | 0.227283 | 0.239915 | **0.8983770** |
| **probe (this gate)** | **0.52427500** | **76.19126892** | 257,056 | 52.427500 | 27.602766 | 0.171163 | **80.2014290** |

Report: `…/eval_root/submissions/v4d_gd3_coarse12x16_pooled/report.txt` (the printed
`Final score: 80.20` is rounded; every number above is **recomputed from components**).
Probe archive: `v4d_composed_gd3_coarse12x16_pooled_archive.zip`, 257,056 B, sha `5db01cb36edbe76c…`.

**A bound 301.9× above the decision threshold carries zero decision information.** The probe is
REFUTED as an instrument for this lever.

### 1.3 Build-path control — the 121.6× is the substitution, not a bug

Before reporting the number I ran the identity arm through the **exact same build path**:

```
identity re-encode byte-identical to shipped state/tokens.dr7t : True
control archive bytes                                          : 360309   (dc1_fold 360309)
```

`gd3_CONTROL_identity_rebuild.zip`. So encode/decode/zip/manifest are exact and the entire d_seg
move is attributable to the token substitution. **Perturbation magnitude, MEASURED:** 72.92% of code
slots changed, mean |Δcode| 2.139 on a 0..15 range (14.3% of range), p95 = 7, max = 15; decoded
frame_1 |Δ| mean 14.93 / p95 54 / max 239 on the uint8 scale.

### 1.4 The reusable law (worth more than the row)

> **On this vehicle the decoder is not robust to token-field perturbation: a 72.9%-slot,
> structure-preserving change of the token field moves d_seg by 121.6×. Therefore no post-hoc
> surgery on the shipped token tensor can be scored against the shipped renderer.**

That is a *class* result, and it lands on three live rows at once. mt1 §3.1 rows 1–3
(`grid_downsample`, `code_width`, `token_quant_levels`) are all carried as "d_seg NOT measured,
pre-registered break-even" — and the measurement each of them implies is exactly the surgery just
falsified. **All three need a training run; none has a $0 d_seg read.** It also retroactively scopes
`ddm_bs2`'s post-hoc re-quantization arm for `token_quant_levels` (mt1 §3.4's "fold it into the §5 #1
training run as a free side-read" remains valid **only** because that read happens *inside* a
trained model, not on the shipped tensor).

**I put my own probe in this class.** I refuted gd2's version of the mistake (bolt an untrained layer
onto a trained net) and then built a subtler version of the same genus (feed a trained decoder an
input it was never trained against). The slot bought a methodological result, not the lever's
verdict. Naming that is the report.

---

## §2 THE RATE ROW: REPRODUCED, then TIGHTENED from a derivation to a byte count

### 2.1 mt1's ladder reproduces byte-exact on the LIVE base

`state/tokens.dr7t` is **byte-identical** across `pw1` / `ms8` / `dc1_fold` (sha `305a2be96a29…`,
346,478 B), so mt1's pw1-anchored ladder transfers to the live best exactly. Re-run on
`dc1_fold` (`gd3_rate_ladder_dc1fold.json`):

```
shipped codes (600,24,32,4) levels=16 -> 346478 B (lossless=True)
grid_downsample=32 [CONTROL: decimated]  87065 B  saved 259413   dS_rate -0.172732
grid_downsample=32 [pooled]              73317 B  saved 273161   dS_rate -0.181887
code_width=2                            170340 B  saved 176138   dS_rate -0.117283
```

Identical to mt1's table to the last digit. **mt1's instrument is sound.**

### 2.2 Two corrections, both in our favour

**(a) The renderer delta is +744 B MEASURED, not +778 B DERIVED.** mt1 scaled the shipped 3,341 B
section linearly by param count. I built a ds=32 renderer section through the real
`ddm_tr1_runtime._encode_renderer` and measured it:

```
shipped renderer.sec  3341 B
ds=32 renderer.sec    4085 B    delta = +744 B     (mt1 DERIVED +778)
```

Structural check: masks 22,248 → 27,432 bits = +648 B packed; gains+biases 246 → 294 fp16 = +96 B;
+648+96 = +744, and Brotli is inert on a near-random mask. mt1 over-charged the row by 34 B —
**conservative direction, immaterial (0.013% of the row), and it is now measured rather than scaled.**

**(b) The row is now an `archive.zip` byte count, not tokens-plus-arithmetic.** I built both complete
ds=32 archives (synthetic `up4` — **byte-count artifacts only, NOT renderable**, named as such) and
stat'd them:

| ds=32 arm | tokens B | renderer B | **archive.zip B** | rate | **ΔS_rate** | break-even Δd_seg | as % of live d_seg |
|---|---:|---:|---:|---:|---:|---:|---:|
| **decimated (adversarial CONTROL — quote this)** | 87,065 | 4,085 | **101,636** | 0.0676752 | **−0.1722397** | **1.722397e-03** | 39.9% |
| pooled (optimistic) | 73,317 | 4,085 | 87,888 | 0.0585210 | −0.1813940 | 1.813940e-03 | 42.1% |

`gd3_ds32_BYTECOUNT_ONLY_decimated_synthetic_up4.zip` (sha `ee2501f5171b4fec…`) and
`…_pooled_…zip` (sha `dd09c33bc8b55577…`), both under `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/`.

**The row is 23.72% of the gap and it is real.** mt1 quoted −0.172214; measured is **−0.1722397**.

### 2.3 The residual rate caveat mt1 already named, now with a measured magnitude

mt1 §6: the ds=32 *content* is derived from a tensor trained at 24×32; a native run emits different
content and therefore different bytes. **Magnitude of that risk, MEASURED:** `cr2_ep854`'s token
section is **271,505 B** against ep945's **346,478 B** — **21.6% spread from training time alone at
identical (ds, cw, levels)**. So the ±20% band on the ds=32 token bytes is real. Even at the
pessimistic end (87,065 × 1.216 ≈ 105,871 B tokens → archive ≈ 120,442 B) the row is still
`ΔS_rate ≈ −0.1597` = **22.0% of the gap.** **The row does not depend on the optimistic end.**

---

## §3 mt1's "ΔS pose = 0" IS FALSIFIED — and it changes the cost of the row

mt1 §5 ranks its top 3 "by measured |ΔS_seg+rate| … pose effect is zero on all three", justified as
"tokens carry no pose term". **The bytes carry no pose term; the decoded frames do.**

MEASURED: probe d_pose **0.00516576 → 76.19126892**.

Mechanism (DERIVED from `experiments/inflate_runner_v4d.py:123-160`, `Decoder.f0`): frame_0 is
*reconstructed from the decoded frame_1* — `f0 := a·warp(f1) + b`, with the per-pair `(a,b)`,
`s_t` index, plane selector and rolling-shutter `beta` index all read from `state/pose_warp.stp`.
**Every one of those was SOLVED at encode time against the original decoded frame_1.** Substitute the
token field and the entire fitted pose carrier is stale. PoseNet then reads a pair whose frame_0 was
warped by parameters fitted to a different image.

Two binding consequences:

1. **`grid_downsample` is a three-axis row, not a two-axis row.** The seg+rate decision column is
   still correct as the *acceptance rule* (operator 2026-08-02), but the pose axis is not
   structurally zero and must be watched.
2. **The cost line is wrong.** mt1: "1 training run". Actual: **1 training run + a full pose
   re-solve**, because the ds=32 run produces a different frame_1 and every archive on this line
   inherits a pose carrier fitted to the ds=16 frame_1. I do **not** claim a magnitude for d_pose
   after a re-solve — my 76.19 is a stale-carrier number and says nothing about the re-solved value.

---

## §4 HYPOTHESES I CHECKED AND REFUTED (reporting the negatives, not only the positives)

1. **"stride-2 decimation samples one phase of a periodic cell-drop mask."** The live base is
   `cell_drop50`: **exactly 384/768 = 50.00% of cells are constant across all 600 pairs.** If that
   drop set were a checkerboard, mt1's `codes[:, ::2, ::2, :]` would sample a single phase and the
   87,065 B would be an artifact. **MEASURED: the static set is spatially contiguous** (top grid rows
   fully static, lower rows fully dynamic — the sky/road split), and the four stride-2 phases carry
   static fractions 0.5104 / 0.5000 / 0.4896 / 0.5000. mt1's decimation samples 0.5104 against a
   population 0.5000. **REFUTED — no phase artifact. mt1's control is representative.**

2. **"the SMEVR coder leaves large bytes on the table; a spatial-redundancy coder would dominate
   `grid_downsample` at zero d_seg cost."** The seed was a real measurement: the block-replicated
   field carries the information of 87,065 B but codes to **309,729 B** — the coder is
   **count-driven, not information-driven** (3.56× off ideal on a maximally redundant field). But
   generalizing that to the *real* field is the error. **MEASURED on the real field:** residual
   order-0 entropy 2.1888 bits/code; left-neighbour innovation **2.6178**; up-neighbour **2.6397**;
   previous-frame innovation 2.2736. **Spatial differencing makes the field LESS predictable, not
   more.** Shipped smevr already codes at 1.5038 bits/code — 31% below the residual's order-0
   entropy. **REFUTED: there is no unpriced spatial redundancy on the real field.** (Consistent with
   mt1/bs2's exhaustive 9-codec race, which smevr won by 51,546 B.)

   **This refutation strengthens the row's falsifier**: near-independent neighbouring cells means a
   4:1 decimation discards ~3/4 of *independent* information, so Δd_seg is unlikely to be small.

3. **Manifest `tokens_sha256` is a lying, unread field.** MEASURED: the shipped value
   `85e2b15d28bd…` matches **neither** the bare `state/tokens.dr7t` (`305a2be96a29…`) **nor** the
   TR1-framed payload (`ac478bdaf3c6…`). `Decoder.__init__` never reads it. This is the #417
   counted-but-inert class, sister of `ddm_dc1` §3's `beta_idx_counts` / `selector_num_two` finding.
   **Declared:** I wrote a (different, also unread) value into it in my probe and byte-count
   artifacts; no decode step consumes it, so no result here depends on it.

---

## §5 WHAT THE ROW LOOKS LIKE NOW

| | mt1 §5 #1 | **gd3 (this arm)** |
|---|---|---|
| ΔS_rate | −0.172214 (tokens measured + renderer derived) | **−0.1722397** (whole `archive.zip` measured, 101,636 B) |
| renderer cost | +778 B DERIVED | **+744 B MEASURED** |
| break-even Δd_seg | 1.722e-03 | **1.722397e-03** (39.9% of live d_seg) |
| % of gap | 23.71% | **23.72%** (23.7% → 22.0% at the pessimistic content end) |
| pose | "0 — tokens carry no pose term" | **NOT zero; carrier is fitted to frame_1 and goes stale** |
| cost | 1 training run | **1 training run + 1 pose re-solve** |
| d_seg readable at $0? | implied yes (pre-registered break-even) | **NO — proven vacuous, 301.9× over threshold** |

**Recommendation: the row is still the largest single measured rate object on the vehicle and the
training run is still the right next spend — but it is the ONLY way to learn the seg half, and it
now carries a pose re-solve.** Fire it as a *training* decision, not as a cheaply-gateable one.

---

## §6 NEXT-IF-RESUMED (the point of putting this in the memo and not in a final message)

1. **Widen the menu and train.** `experiments/train_tr1_partition_renderer_mlx.py:1692`
   `--grid-downsample choices=(8,16)` → `(8,16,32)`. **VERIFIED ds=32-ready otherwise:** the config
   derives `grid_h/grid_w` from `SEG_H//D`, `SEG_W//D` (`:355-361`), `n_upsample` from `log2(D)`
   (`:364-370`, and 384/32=12, 512/32=16 are integers), and the module builds
   `for k in range(cfg.n_upsample)` (`:473`, `:644`). **No hardcoded 24×32 anywhere in the trainer.**
   The argparse `choices` tuple is the whole blocker — mt1's "the capability was built; only the menu
   pinned it" is confirmed by source read.
2. **Budget the pose re-solve into the same unit** (§3). A ds=32 composed S is meaningless until the
   pose carrier is re-fitted against the ds=32 frame_1.
3. **Do NOT attempt any further post-hoc token surgery as a d_seg instrument** (§1.4). If a cheap
   d_seg read on the coarse grid is wanted, the only honest one is a *short* ds=32 training run
   (decoder co-adapted), not a re-grid of the shipped tensor.
4. **The pessimistic-content band is the row's remaining rate risk** (§2.3): if the native ds=32 run
   emits token bytes ≥ 105,871 B the row drops from 23.7% to ~22.0% of the gap. Still fires.
5. **Owed and NOT done:** the *decimated* (adversarial) seg arm was never gated — I spent the single
   slot on the pooled arm because it is the cheaper kill, and it killed the whole instrument instead.
   Given §1.4 there is no point gating the decimated arm either; it is superseded, not owed.

---

## §7 FALSIFIERS FOR THIS UNIT

1. **§1.4's class law:** if any post-hoc token-field modification is found whose realized d_seg lands
   within an order of magnitude of the break-even, the "no $0 d_seg read" law is too strong and the
   §3.1 rows 1–3 regain a cheap instrument. (My probe's 301.9× overshoot is one point, not a curve.)
2. **§2.2's rate row:** if a native ds=32 run emits an `archive.zip` ≥ 187,000 B, the row is under
   half its claimed size and the ordering against `code_width` must be re-derived.
3. **§3's pose coupling:** if a ds=32 run with a **re-solved** pose carrier lands d_pose within
   `0.00516576 ± 20%`, then mt1's "pose effect ≈ 0" was right *in effect* even though it is wrong
   *in mechanism*, and the extra re-solve cost I added to the row is overstated.
4. **§4.2:** if a coder with an explicit spatial context model beats shipped smevr on the *real*
   tensor by > 5,000 B, my "no unpriced spatial redundancy" refutation is wrong and the
   zero-d_seg-cost rate lever reopens.

---

## §8 CUSTODY

| artifact | bytes | sha256 (16) |
|---|---:|---|
| `…/ddm_v4d_20260731/v4d_composed_dc1_fold_archive.zip` (base) | 360,309 | `9fb9f4e9e460f91c` |
| `…/ddm_v4d_20260731/v4d_composed_gd3_coarse12x16_pooled_archive.zip` (probe, GATED) | 257,056 | `5db01cb36edbe76c` |
| `…/ddm_v4d_20260731/gd3_CONTROL_identity_rebuild.zip` | 360,309 | build-path control |
| `…/ddm_v4d_20260731/gd3_ds32_BYTECOUNT_ONLY_decimated_synthetic_up4.zip` | 101,636 | `ee2501f5171b4fec` |
| `…/ddm_v4d_20260731/gd3_ds32_BYTECOUNT_ONLY_pooled_synthetic_up4.zip` | 87,888 | `dd09c33bc8b55577` |
| `…/ddm_v4d_20260731/gd3_rate_ladder_dc1fold.json` | 4.9K | ladder re-run on the live base |
| gate report | — | `…/eval_root/submissions/v4d_gd3_coarse12x16_pooled/report.txt` |
| gate log | — | `.omx/tmp/gd3/gate_pooled.log` |

The two `BYTECOUNT_ONLY` archives contain a **synthetic `up4`** and are **not renderable**; they exist
solely to make the §2.2 rate row an `archive.zip` stat instead of an arithmetic sum. They are named
so they cannot be mistaken for candidates. The probe archive is **not a candidate** either — its rate
is not the ds=32 rate; only its d_seg / d_pose were ever the point.

**No gate was fired on a candidate. No pointer was moved. `score_claim=false` on every row above.**
