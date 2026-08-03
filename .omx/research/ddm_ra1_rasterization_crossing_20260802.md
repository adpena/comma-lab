# ddm_ra1 — the rasterization crossing: we are not quantizing, we are RESAMPLING

- **arm:** `ddm_ra1` · **date:** 2026-08-02 · **axis:** `[macOS-CPU advisory]` NON-PROMOTABLE.
  `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`. **Pointer UNMOVED.**
- **operator scope (verbatim):** *"It kinda feels like we're quantizing and rasterizing. The
  continuous geometry and deep math we've discovered represents the world space."* /
  *"It's interesting that we keep brushing up against dithering and anti-aliasing as well."*
- **method:** $0. **No scorer pass fired** — the n600 slot is held (`ddm_gd3` → `pj2`). Every number
  below is either read from a custody receipt or MEASURED by running the **exact shipped receiver on
  the exact shipped bytes** (`/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_dc1_fold/`).
  Pairs are sampled **STRIDED across all 600**, never a prefix (`m88`).
- **denominator for every ΔS:** `tac.canonical_equations.gap_decomposition_against_floor_20260802`.
  Gap to PR130 = **0.7262358** (seg 0.4015190 = 55.3% · pose 0.2120155 · rate 0.1127679).
  1% of gap = **10,907 B**. PR130 floor = 191,052 B.
- **live base:** `dc1_fold` S = 0.8983775 (seg 0.4311790 / pose 0.2272835 / rate 0.2399150), 360,309 B.

---

## §0 Headline

**Four things, three of which correct something — including two of my own hypotheses and one line of
my own charter.**

**(1) The charter's premise is wrong in our favour: R2 was already fired, and against SEG.** The
brief says ll1's window solve *"was derived for pose only"* and asks whether it transfers to seg.
It does not need to transfer — `ddm_ll1` (commit `6331c9d83e`) measured it **through the canonical
DistortionNet/SegNet path on `d_seg`**: 88 flips → 3 flips, `d_seg 0.0001492 → 0.0000051`,
**ΔS −0.01441** (n=3 smoke, ideal targets), and landed it as a **production module wired into the
shipping receiver, DEFAULT OFF**. It is an **UNWIRED-BUT-BUILT P0** (`m56`), not an unfired idea.

**(2) The mechanism is not quantization. It is resampling — and that inverts the operator's
emphasis.** MEASURED on our own shipped renders, decomposing the camera-raster debt
`D(camera) − r` in exact float:

| term | rms (scorer-plane LSB) | max | share of variance |
|---|---:|---:|---:|
| **`D(U(r)) − r` with NO rounding, NO clip, pure float** | **0.7737** | **20.330** | **93.5%** |
| + `rint` | 0.7978 | 20.411 | — |
| + `clip` (the shipped raster) | 0.8003 | 20.411 | 100% |
| *isolated quantization term* (`shipped − float`) | *0.1955* | — | *6.0%* |

**94% of the debt exists in exact float arithmetic.** `U` (bicubic ↑874) followed by `D` (bilinear
**point** sampling at stride 2.276, `antialias=False`) is **not the identity** — `D` undersamples
`U(r)` and aliases. uint8 is 6% of the problem. *(Independent corroboration: ll1's own
`08ad268d4d` says "mp1's dither ask attacks only the 6% rounding term" — same 6%, reached from a
different direction.)*

This is why **dithering is the wrong family and the window solve is the right one**: dither/AA
kernels attack the 6%; the solve attacks 100% because it does not try to improve `U` — it directly
**inverts `D`**.

**(3) On our own bytes the solve is stronger than ll1's own headline, and my "our renders are
smoother so the debt is smaller" hypothesis is REFUTED.** Live `dc1_fold`, 4 strided pairs:

| | plane rms | plane max | scorer px off by >0.5 LSB | blind px touched |
|---|---:|---:|---:|---:|
| shipped `clip(rint(U(r)))` | 0.7994 | 20.411 | **12.99%** | — |
| + ll1 window solve | **0.0298** | **0.474** | **0.00%** | **0** |
| | **26.8×** | **43×** | — | verified, not asserted |

Our debt is **1.65× worse** than ll1's real-frame baseline (0.7994 vs 0.4841), not better. After the
solve the scorer's input is within **half an LSB everywhere** — below the granularity of the signal
itself.

**(4) The gate is pose, and it is large.** frame_0 is warped **from frame_1's camera pixels**
(`inflate_runner.py:172,185`) and the warp resamples **across** the private windows, so the solve's
intra-window redistribution is fully visible to it: **frame_0 delta rms 2.75 LSB, max 178, 58.8% of
pixels changed.** Seg-invisible ≠ pose-invisible. **d_pose remains UNMEASURED and gates the win.**

**R1 verdict.** Camera-raster realization is bounded at roughly **0.014–0.035 S** = **3.5–8.7% of
the 0.4015 seg gap**. **≥91% of the seg gap is DESCRIPTION error, not rasterization.** The operator's
"quantizing and rasterizing" instinct correctly identifies a real, zero-byte, already-built lever —
but it is a **~2–5%-of-gap** lever, not the seg axis itself.

---

## §1 Apparatus validity (checked before anything was read off it)

| claim | source | kind |
|---|---|---|
| ll1 window solve is a production module, wired DEFAULT OFF | `src/tac/optimization/ddm_ll1_window_solve.py`; `ddm_tr1_runtime.py:1378-1416` | READ (source) |
| shipping receiver calls it **without** `window_solve=True` | `v4d_dc1_fold/inflate_runner.py:172` | READ (shipped bytes) |
| ll1 measured on `d_seg`, ΔS −0.01441, n=3 | commit `6331c9d83e` + module docstring L38-45 | READ (custody receipt) |
| D disjoint 2×2; 230,904 blind px (22.70%); 786,432 = 196,608×4 | `08ad268d4d`; `window_geometry()` | MEASURED (ll1); blind-set invariance **re-verified here** |
| `probe_PA` floor path contains `round/uint8` | `probe_PA_paintfloor_perclass_20260708.md:20-21` | READ |
| oracle-R@384 floor 0.09100 S; live 0.38878 S; 4.27× | `ddm_pc2_perclass_road_edges_20260802.md:31` | READ |
| render grid **is** 384×512 live (192→384 win already taken) | `ddm_pc2…:83-90` | READ |
| shipped render = tokens → CNN → `384×512×3` float | `ddm_tr1_runtime.py:1300-1323` | READ (source) |

**Positive controls run before any conclusion.** (a) The blind-set invariance ll1 *asserts by
construction* was **re-verified independently**: across every variant and every pair, `blind_px_touched
= 0` against `blind_mask()`. (b) `d_seg = flips / (n_frames × 196,608)` was checked against ll1's own
receipt: 88 / (3 × 196,608) = 1.4919e-4 vs the reported 0.0001492 — **exact**, so the flip↔d_seg
conversion used in §5 is theirs, not mine. (c) Every variant reproduces the same shipped baseline
`0.7994` rms before diverging, so the variants differ only where intended.

**Artifacts.** `.omx/research/ddm_ra1_realization_debt_dc1_fold_20260802.json` ·
`…_min_norm_window_solve_20260802.json` · `…_init_choice_overshoot_20260802.json`.
Probes: `experiments/ddm_ra1_{realization_debt_on_live_vehicle,minimum_norm_window_solve,init_choice_and_overshoot}.py`.

---

## §2 The crossing, measured on our own bytes

The shipped chain (`ddm_tr1_runtime.py:1409-1416`, exercised through the vendored receiver):

```
tokens --CNN--> r (384,512,3) float --U bicubic--> (874,1164,3) float
       --clip(rint)--> camera uint8 --D bilinear point-sample--> (384,512,3) = what the scorers read
```

`D` is the operator **both** frozen scorers read through (`SegNet.preprocess_input` and
`PoseNet.preprocess_input` both interpolate to 384×512 *before* anything else), so a single
measurement covers both. Per-pair, strided:

| pair | baseline rms | baseline max | solved rms | solved max | camera px moved | blind touched |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.80029 | 20.411 | 0.02992 | 0.477 | 57.56% | 0 |
| 150 | 0.79681 | 20.411 | 0.02971 | 0.473 | 57.47% | 0 |
| 300 | 0.79810 | 20.411 | 0.02989 | 0.473 | 57.50% | 0 |
| 450 | 0.80239 | 20.411 | 0.02981 | 0.473 | 57.66% | 0 |

**The max is 20.411 to five figures on every pair.** That is not content — it is a property of the
`U∘D` geometry, and §3 shows why.

**Population check (`m88`).** The governing quantity here is the plane debt, and its subset spread
is 0.7968–0.8024 (±0.35%) across pairs 200 apart in a temporally-correlated video. Unlike `d_pose`
(right-skewed, 5.1× prefix/population ratio), this quantity is **mechanically flat** — it is a
property of the resampling geometry, not the scene. The 4-pair subset is representative *of this
quantity*; it says nothing about how many **flips** result, which is scene-dependent and unmeasured.

---

## §3 Where the debt is born — and two of my hypotheses, refuted by measurement

I proposed two mechanisms. **Both are wrong.** Recording them because each would have shipped a
wrong headline.

**H1 — "the debt is clipped bicubic overshoot." REFUTED.** Our render is `sigmoid×255`, so it
saturates, and PyTorch bicubic (A = −0.75) has negative side lobes that overshoot past [0,255]
against a saturated edge — a clean story. Measured: only **0.123%** of camera pixels clip, and the
debt **with** clip (0.8012) vs **rounding-only** (0.7988) differs by **0.3%**. Clipping is
negligible.

**H2 — "the debt is quantization." REFUTED, and this is the important one.** The float-only
decomposition in §0(2) shows `D(U(r)) − r` is already **rms 0.7737, max 20.330 with no rounding at
all**. The isolated quantization term is rms 0.1955 = **6.0% of the variance**.

**The actual mechanism: `D` aliases `U(r)`.** `D` is bilinear **point** sampling with
`antialias=False` at stride 2.276 > 2 — it reads 4 camera pixels per scorer pixel and ignores the
other 22.70% entirely. `U` spreads `r` across all 1,017,336 camera pixels with a 4-tap bicubic
kernel. The 4 points `D` happens to read do not reconstruct `r`: the pair `U`, `D` are not inverse,
and the mismatch is largest exactly where `r` has an edge — i.e. **on the separatrix**, which is
where `pc2` measured 99.94% of our flips to live.

**This is the anti-aliasing problem, but the classical cure is unavailable and unnecessary.** We
cannot prefilter `D` (it is frozen), and we do not need to: we control the *signal being sampled*.
The correct move is **pre-compensation, not prefiltering** — choose the camera raster so that an
aliasing sampler reads the intended value. That is precisely what ll1's solve does, and it is why it
recovers ~97% of a debt that dithering could only have touched 6% of.

**H3 — "the init is a free null-space lever." REFUTED, and ll1's choice CORROBORATED on an axis
ll1 never measured.** The solve hits the target from any starting raster, so the init is a free
zero-byte choice. ll1 rejected the r-broadcast init on *plane* residual. But after the solve the
plane residual is ~0.0298 for **every** init — the discriminating quantity is the frame_0 warp
delta, which nobody had measured. Ranked on it:

| init | plane after solve | camera move rms | **frame_0 delta rms** | frame_0 max |
|---|---:|---:|---:|---:|
| **bicubic (shipped)** | 0.02990 | 3.46 | **2.90** | 183 |
| bilinear (no overshoot) | 0.02934 | 3.94 | 3.38 | 182 |
| nearest / r-broadcast | 0.02993 | 6.15 | 5.45 | 181 |

I set out to overturn ll1's init on a new axis and instead **confirmed it**: bicubic is best on the
pose axis too, by 1.17× over bilinear and 1.88× over nearest.

**H4 — "spread, don't concentrate" (my min-norm variant). REFUTED.** CLAUDE.md's Fridrich rule says
*"spread small errors (L∞ penalty), don't concentrate large ones"*, and ll1's greedy allocation dumps
each window's whole correction on the single highest-weight tap — an apparently textbook violation
producing a 178-LSB tail. I built the minimum-L2 allocation (`s = w·err/‖w‖²`, spread over all four
taps in weight proportion, then one closing pass). Result:

| variant | plane rms | camera move rms | camera move **max** | frame_0 rms | frame_0 **max** |
|---|---:|---:|---:|---:|---:|
| greedy (ll1) | 0.029833 | 3.4467 | 188 | 2.7496 | 178.25 |
| min-norm (mine) | 0.029788 | 3.4358 | **188** | 2.7367 | **178.25** |

Differences of 0.15–0.5% and **identical maxima**. **The large moves are FORCED by the delivery
constraint, not chosen by the tie-break.** Where `U∘D` mismatches badly, *any* raster satisfying
`Σ w_k c_k = r` must move that far. The square-root law has no purchase here because there is no
slack to spend — the null space is 3-dimensional but the *magnitude* is set by the constraint.

**What this rules out.** The null-space-choice family (dither pattern, error diffusion order,
allocation norm, init kernel) is **measured shut** for reducing the pose coupling: four independent
attempts moved it by <5%. **verdict_scope: FORMULATION** — null-space *re-allocation* at fixed
per-window constraint. It does **not** kill the family of changing the *constraint* (§5.3).

---

## §4 R3 — where AA can and cannot act in the shipping vehicle

**The shipping vehicle has no continuous-geometry rasterization stage.** `render_frame1_float`
(`ddm_tr1_runtime.py:1300-1323`) is:

```
decode_token_grid --> conv2d+bias --> GELU --> [np.repeat x2 --> conv2d+bias --> GELU] xN
                  --> conv2d+bias --> sigmoid*255 --> (384,512,3)
```

There is **no SDF, no level set, no Laguerre cell, no coverage integral, no sub-pixel boundary
placement** anywhere in the receiver that scores. The continuous geometry the operator is describing
lives in the **witness / level-set line**, which is not the vehicle that produces our frontier.
Consequently:

- **#149 (sub-pixel boundary placement at camera resolution before D averages) — NOT APPLICABLE
  as written.** It presumes camera-resolution geometry to place. We have none; we have a 384×512
  plane. Its *premise* is nevertheless vindicated in a stronger form: §3 shows the `U∘D` mismatch is
  concentrated at edges, which is exactly the effect #149 anticipated — but the actionable cure is
  ll1's exact `D`-inverse, not sub-pixel placement.
- **#220 / #283 (AA coverage-integrated render, AA-SDF rasterizer) — NOT APPLICABLE to TR1.** Both
  operate on the level-set renderer. Confirmed against `aa_feasibility_reconciliation_20260702.md`,
  which describes them at the *witness* render grid.
- **The render-grid lever is already taken.** `pc2` §2 establishes the live path is `--render-h 384
  / --render-w 512` (not the base trainer's 192), i.e. we already render **at the scorer's own
  resolution**. There is no cheap resolution win left on this axis.

**§4.2 — the one genuinely unexamined AA surface, and why it is not free.** The decoder upsamples by
`np.repeat(…, 2, axis=1/2)` — **nearest-neighbour replication, the crudest possible kernel** — three
times, inside the network. That is a real aliasing source at exactly the class boundaries, and
replacing it is a **receiver-side, zero-counted-byte** change. But it is **not a free swap**: the
LOTTO renderer weights and every shipped token were searched against this exact decoder, so changing
the kernel invalidates the description. It is a **re-race item** (new tokens + re-search), not a
receiver patch. Filed, not claimed.

---

## §5 The actionable rung, its price, and what it needs from the held slot

### 5.1 What turning the flag on is worth

`d_seg = flips / (n_frames × 196,608)` ⇒ `ΔS = 100 × flips_per_frame / 196,608`. ll1 measured 29.3
flips/frame of realization debt on real-frame planes (ΔS 0.01441) at a plane debt of their
baseline; ours is **1.65× larger in rms**. Flip count is **not** linear in rms — it depends on the
SegNet margin distribution at the separatrix — so I give a bracket, not a point:

| | ΔS | % of the 0.7262 gap | byte-equivalent @ 10,907 B/% |
|---|---:|---:|---:|
| ll1's measured value, transferred as-is | 0.0144 | 1.98% | ~21,600 B |
| linear-in-rms scaling (1.65×) — **EXTRAPOLATED, not measured** | 0.0238 | 3.27% | ~35,700 B |

**Both are labelled INFERRED.** The honest statement is: **a zero-byte, already-built receiver
change is worth something in the 2–3%-of-gap range on seg, and the exact number needs one scorer
pass.** For scale, `dc1_fold`'s entire measured win over `ms8` was 0.000056 S; `pw1`→`ms8` was
0.049 S. This sits between them — a real rung, not a headline.

### 5.2 Why it cannot be turned on today

The frame_0 warp reads frame_1's camera raster **across** window boundaries, so the solve's
±LSB redistribution is fully visible to PoseNet: **rms 2.75 LSB, max 178, 58.8% of frame_0 pixels
changed**. And the v4d pose sidecar (`p_best`, `s_t`, `sel`, `ab`, `beta_idx`) was **fitted against
the baseline raster**. Changing the warp source under a frozen fitted pose is generically a
**degradation**. Expected sign on d_pose: **negative**. Magnitude: **unmeasured**.

Given pose is 0.2120 of the gap and 1% of gap = 0.00726 S, a d_pose regression of only ~3% of the
pose gap erases the entire seg win. **This is why the flag is off, and the docstring's caution is
correct.**

### 5.3 What §3 rules out, and the one cure it leaves standing

§3's four refutations close the *null-space re-allocation* family: you cannot make the solve
pose-quiet by choosing a different dither, order, norm, or init. The moves are constraint-forced.

The family still standing is to **change the constraint**: solve for the pair `(f1_raster)` under a
**joint** objective — deliver `r` through `D` (seg) **while** the warped `f0` stays close to the
raster the pose fit was made against — or, equivalently and more cheaply, **re-fit the pose sidecar
against the solved raster**. The second is strictly the better first move: it is the *same* bytes,
costs no rate, and `m85`'s matched-base rule already demands it (*"matched-base ctrl ≥32 pairs BEFORE
composing"*). **Composing a seg-only win onto a stale pose fit is the exact error `m85` names.**

### 5.4 The one thing that needs the held n600 slot — an exact, falsifiable request

**Do not self-fire.** Requesting through MAIN:

> Render n600 frame_1 with `window_solve=True` on `v4d_dc1_fold`, holding **everything else
> byte-identical**, and report `d_seg` and `d_pose` separately against the `window_solve=False`
> control on the **same** pairs.

- **Falsification threshold (seg):** if `Δd_seg × 100 > −0.010 S`, the realization debt is smaller on
  our renders than the plane-space debt implies, the flip↔rms transfer is refuted, and this rung is
  **retired** (not deferred).
- **Falsification threshold (pose):** if `Δ√(10·d_pose) > +0.010 S` with the **stale** pose fit, the
  bolt-on is dead and the item converts to "re-fit the sidecar against the solved raster, then
  re-measure" — which is a different, larger unit of work and must be ranked against `#766`
  waterfill and the granularity re-race, not assumed.
- **Cost:** two n600 passes (control + treatment). Decode cost of the solve is 0.07 s/frame ⇒
  ~1.3 min single-core for n600, well inside the 30-min budget (ll1, measured).

---

## §6 What this arm did NOT find (stated so nobody re-runs it)

- **No new seg lever.** The seg gap (55.3% of total, and *constant* to 7 dp across v4d→pw1→ms8→
  dc1_fold) is **description error**. Nothing in the rasterization crossing addresses it.
- **No dithering lever.** Dither/AA attacks the 6% quantization term. Measured shut.
- **No error-diffusion-into-the-blind-set lever (R4).** The blind set is blind to `D` but **not** to
  the warp (`bp2`), and every null-space re-allocation I tried moved the pose coupling by <5%.
  `#532`'s warning (uint8 breaks range(A) exactness) never even became binding — the family failed
  earlier, at the constraint.
- **Negative-existence scope.** Claims of the form "X is not in the vehicle" (§4) are scoped to:
  `ddm_tr1_runtime.py`, `inflate_runner.py`, `pfs1_warp_receiver.py` in the shipped
  `v4d_dc1_fold` submission directory — read in full. Corpus claims are scoped to
  `tools/corpus_query.py`'s index (~76%, 7,398 of 9,706).

---

## NEXT-IF-RESUMED

1. **Highest value:** get the §5.4 n600 pair through MAIN. It is the only unknown that ranks this rung.
2. If pose regresses: the unit becomes *re-fit `p_best`/`s_t`/`ab`/`beta` against the solved raster*.
   Scope that against `#766` waterfill and the granularity re-race before building.
3. **Do not** re-attempt null-space re-allocation (dither order, allocation norm, init kernel) —
   §3 measured it shut at FORMULATION scope, four independent ways.
4. §4.2 (`np.repeat` nearest upsampling inside the decoder) is a live, unexamined, zero-byte AA
   surface — but it is a **re-race** (invalidates the token search), not a patch.
