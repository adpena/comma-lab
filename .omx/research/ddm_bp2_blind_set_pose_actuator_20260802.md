# ddm_bp2 — the scorer-blind 22.70% IS a real, exactly-seg-free pose actuator, and it does not pay

**UTC** 2026-08-03 · **arm** `ddm_bp2_blind_set_pose_actuator` · **axis** `[macOS-CPU advisory]`
frozen CPU-torch scorers on decoded camera rasters — **NOT** `upstream/evaluate.py` on an archive.
`score_claim=false`, `promotion_eligible=false`. **Pointer UNMOVED.**

**Vehicle under test:** `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pb2_bestof_archive.zip`
— 360,339 B, sha256 `6e1b80e90109edd3c06f29fdfe37937dfb78eac7480c4e65adbc364a10e1e764`
(the archive `v4d_verify_receipt.json` certified). ⚠ I did **not** verify that this exact build is the
one behind the `S = 0.9639878` headline; the v4d directory holds several near-identical pose payloads.
Everything below is scoped to **this** archive.

**Code landed:** `604f7180b3` — `src/tac/optimization/ddm_bp2_blind_pose_actuator.py` (exact warp tap
decomposition + adjoint), `tools/ddm_bp2_blind_warp_reach.py` (3 modes), 26 tests.
**Receipts:** `reports/ddm_bp2/{blind_verify,overlap_n600,reach_n600,incidental_n24,index_cost,proxy_index_n12}.json`.

---

## Verdict against the pre-registered falsifier

| | threshold | MEASURED | verdict |
|---|---|---|---|
| **F1** blind-set overlap with the frame_0 warp read-set | `< 5%` closes | **14.80%** (n600) | **REFUTED — does not close** |
| **F2** achievable `abs(delta d_pose)` from a blind-only perturbation | `< 1e-4` closes | **0.628** (mean over pairs, per-pair max over arms) | **REFUTED by ~3.8 orders of magnitude** |

**Both legs of the charter's claim are confirmed.** The 230,904 blind pixels are exactly invisible to
both scorers, they ARE read by v4d's frame_0 warp, and perturbing them moves `d_pose` violently at
literally zero seg cost.

**And the family still does not pay.** The steering direction is video-derived, so it must be shipped,
and at **every** measured operating point the rate cost of naming the coordinates exceeds the pose gain
— by **7.3x at the cheapest fixed arm** and, at best, by **2.9x** at the per-pair argmin envelope. This is a
**FORMULATION**-scope closure (explicit shipped blind-coordinate correction index), not a family kill;
§6 names the one substitution that flips the sign and the measurement that refutes the two cheapest
candidates for it.

---

## 1. The blind set, re-derived rather than inherited (MEASURED)

I did not trust the recorded 230,904. `D` is linear with non-negative weights, so a camera pixel is
blind **iff** its column sum is exactly zero — one autograd pass through the REAL
`F.interpolate(x,(384,512),'bilinear')` gives every column sum with no hand re-derivation of the
bilinear geometry:

```
blind (column sum == 0.0) : 230,904 px = 22.6969%
read                      : 786,432 px = 4 x 196,608 EXACTLY
max column sum            : 0.9934998  (<= 1 => no camera pixel is read twice)
min READ column sum       : 1.0173e-05
sum of all column sums    : 196,608.0  (= the scorer pixel count)
agrees with ddm_ll1_window_solve.blind_mask() : bit-identical
```

**Empirical invisibility, through the canonical `DistortionNet.preprocess_input`.** Perturbing
461,649 blind pixels by uniform ±127 leaves BOTH `posenet_in` and `segnet_in` **bitwise identical**
(`torch.equal`).

**Positive controls fired, at both ends of the read-weight range** — a guard never shown to fire is
untrusted:

| control | D weight | posenet_in identical | segnet_in identical |
|---|---|---|---|
| read px (688,12) — **heaviest** in the frame | 0.9934998 | **False** | **False** |
| read px (689,13) — **lightest** in the frame | 1.0173e-05 | **False** | **False** |
| blind px (437,4) | 0.0 | True | True |

The lightest read pixel in the entire frame still moves both scorer inputs under a +40 change, while a
blind pixel four columns away does not. The blind/read boundary is an **exact structural invariance**,
not a numerical tolerance.

## 2. The warp operator, decomposed exactly (MEASURED)

`v4d_pair_taps` composes precisely what `inflate_runner_v4d.Decoder.f0` does — the per-pair selector
(single-plane vs static two-plane far/ground row split), the rung-A rolling-shutter row blend
`(1-alpha)*W(1-beta/2) + alpha*W(1+beta/2)`, and the photometric `a`. Verified:

- forward reconstruction vs the vendored `pfs1_warp_receiver.warp_rgb` on real decoded frames:
  **max abs error 8.5e-14** (fp64 round-off), tap row sums `1 ± 1.1e-16`;
- adjoint verified by the inner-product identity `<Mx, v> == <x, M^T v>`;
- the receiver's INVALID branch (`out = flat`) is reproduced as a unit identity tap `q -> q` — it is
  **not** a warp read, which matters because it routes read-pixels to read-pixels and contributes
  exactly zero blind mass.

## 3. F1 — the overlap, n600 (MEASURED)

For each pair I compute the exact column mass of the composed operator `D . (a*M)` on blind columns:
*what fraction of frame_0's scorer-visible signal is SOURCED from blind frame_1 pixels.*

```
blind mass fraction   mean 0.14800   min 0.03354   max 0.22208     (n = 600 pairs)
blind px with nonzero influence            mean 63.20%
warp valid fraction                        mean 66.94%
closure check (total mass == |a| * 196608) PASSED on all 600
```

**Internal consistency (DERIVED):** `0.14800 / 0.66935 = 0.2211`, against the frame's blind fraction
`0.22697`. Inside the valid warp region the blind pixels carry **essentially their proportional
share** — independent confirmation the adjoint is right. The ~31% invalid region contributes zero
blind mass by construction (identity path), which is the whole of the gap.

## 4. F2 — the reach, and the mechanism (MEASURED)

⚠ **Sampling caveat, stated once and carried everywhere below.** The reach measurement streams pairs in
video order and had reached **75 of 600** when this memo was written (the run is resumable and still
going; `--resume` carries completed rows). This prefix is **~5x harder than the population**: prefix
mean `d_pose` **0.044373** vs the v4d refine receipt's n600 `mean_d_final = 0.00858414`. The prefix
distribution is heavily skewed (median 0.000886, p90 0.128526, max 0.774989). **Every `d_pose` mean and
every `delta S` in §4–§5 is a prefix number, not an n600 number.** F1 is n600; F2's verdict is 3.8
orders of magnitude clear of its threshold and does not turn on the sample.

**d_seg is EXACTLY unchanged, in every arm, on every pair measured** — including under a full ±1 LSB
gradient-aligned sign step over all 692,712 blind coordinates, the largest structured 1-LSB
perturbation the channel admits. Not "within tolerance": bit-identical `d_seg`. This is the
zero-seg-cost claim, confirmed through the real frozen SegNet rather than argued from the geometry.

**The channel's gain is enormous and direction-selective:**

```
mean d_pose base                                    0.0390414
full +-1 LSB gradient-sign step, descent direction   0.28573    (x 7.3)
full +-1 LSB gradient-sign step, ascent direction    0.64762    (x 16.6)
mean max|delta d_pose| over all arms                 0.6281     (F2 threshold 1e-4)
RANDOM-sign step at the SAME coordinate count       0.0390592  (+0.05%)  <- control
```

The random-sign control is the load-bearing guard: at the **same number of touched coordinates**,
random signs are inert (+0.05%) while gradient-aligned signs move `d_pose` by 7x–17x. The steering is
real, not a perturbation-magnitude artifact.

**A full 1-LSB step is far too coarse.** The gradient L1 mass over the blind set is already ~1.0–1.3x
`d_pose`, so the full step lands at or past the minimum — and it makes `d_pose` ~300x worse in **both**
directions (0/6 pairs improved in the first formulation). Descent therefore requires *sparsifying*:
touch only the top-k blind coordinates by `|gradient|`, with k chosen so the cumulative gradient mass
predicts a first-order decrease of `t * d_pose`.

**Two mechanism findings that are worth more than the headline:**

1. **`d_pose` is a RELATIVE-geometry quantity between the two delivered frames, not a per-frame
   fidelity quantity.** Control: replacing the decoded `f0` with the **true GT frame_0** gives `d_pose`
   **3.05 / 6.29 / 7.96 / 16.66** on pairs 2/1/0/3 — versus **0.0002–0.0009** for the decoded pair.
   v4d's `(warp(f1), f1)` is internally consistent: the warp *is* the intended motion, so PoseNet reads
   it correctly even though both frames are photometrically far from GT. Mixing a real frame into a
   rendered pair destroys that consistency catastrophically. **Corollary: any future "make frame_0 look
   more like GT" objective is aimed at the wrong target on this vehicle.**
2. **Non-adversarial (incidental) sensitivity is mild** (n=24, random signs, whole blind set):
   `±1 LSB -> x1.0070 · ±2 -> x1.0050 · ±4 -> x1.0276 · ±8 -> x1.0338 · ±16 -> x1.2614 · ±64 -> x13.43`.
   The violence is entirely in the ALIGNMENT, not the magnitude. **This bounds the BLIND-channel risk
   only.** I re-verified `ddm_ll1`'s claim rather than repeat it: its window solve on a real frame
   changes **567,524 camera pixels, of which ZERO are blind** (blind pixels byte-identical), i.e. it
   moves **72.2% of the READ set**. So ll1's frame_0 coupling is a different and much larger channel
   than the one measured here, and nothing in this memo prices it — the #897 n600 gate still owes that.

## 5. The economics — the actual verdict (MEASURED gain, MEASURED index cost, DERIVED table)

To realize a per-pair top-k blind correction the receiver must be told **which** coordinates and
**which sign**. Both are video-derived (they come from the true pose target), so both are counted bytes.

**I attacked my own bound before using it.** The combinatorial floor `log2 C(692712, k)` assumes an
unstructured subset. Measured on the real selected sets (brotli q11 / zlib-9 on the selection bitmap):

| k | comb floor B | best measured index B | structure gain | signs-only B |
|---:|---:|---:|---:|---:|
| 200 | 329 | 322 | 1.02x | 29 |
| 1,000 | 1,359 | 1,027 | 1.32x | 129 |
| 5,000 | 5,344 | 3,518 | 1.52x | 629 |
| 20,000 | 16,339 | 10,206 | 1.60x | 2,504 |

The selected coordinates **are** spatially structured — the honest floor is up to 1.6x below the
combinatorial bound. The table below uses the **measured** cost (log-interpolated in k), not the bound.

| target t | mean k | mean d_pose | ΔS_pose | B/pair | ΔS_rate | **NET ΔS** |
|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 0.0390414 | — | — | — | — |
| 0.002 | 5 | 0.0389610 | −0.00064 | 12 | +0.00465 | **+0.00400** |
| 0.01 | 29 | 0.0386820 | −0.00288 | 60 | +0.02388 | +0.02100 |
| 0.05 | 241 | 0.0372846 | −0.01422 | 398 | +0.15883 | +0.14461 |
| 0.15 | 1,222 | 0.0351669 | −0.03181 | 1,351 | +0.53983 | +0.50801 |
| 0.35 | 5,138 | 0.0342468 | −0.03962 | 4,233 | +1.69128 | +1.65166 |
| 0.7 | 23,505 | 0.0427030 | +0.02864 | 14,490 | +5.78886 | +5.81750 |
| 1.0 | 78,582 | 0.0556551 | +0.12119 | 37,398 | +14.94092 | +15.06211 |
| **per-pair argmin** | 3,546 | 0.0105952 | **−0.29933** | 2,189 | +0.87448 | **+0.57515** |

98.6% of pairs improve under the per-pair argmin, and the envelope cuts `d_pose` by **73%**
(0.0390 -> 0.0106, ΔS_pose −0.299). The arm index itself is negligible (3 bits/pair = 225 B total).
**But NET ΔS is positive at every single operating point.** Cost/gain ratios: 7.27x (k=5) · 8.29x
(k=29) · 11.17x (k=241) · 16.97x (k=1,222) · 42.69x (k=5,138). **The tightest deficit anywhere is
2.92x, at the per-pair argmin envelope** — the envelope's much larger pose gain outruns its byte cost
faster than any fixed arm does, so the cheapest arm is NOT the best-value arm.

Per-pair argmin over a deterministic score is a *realizable encoder-side choice*, not selection on
noise — d_pose is deterministic on fixed pairs and the choice is reproducible. The grid was widened
mid-run (from a top target of 0.05 to 1.0) precisely because a partial run showed the top arm binding;
a grid whose top arm is selected reports a LOWER bound on the achievable gain, and the verdict turns on
that number. With the widened grid the top two arms are now *worse* than base, so the interior optimum
is bracketed and the bound is no longer loose on that side.

## 6. What would flip the sign — and the measurement that refutes the two cheapest candidates

**The entire question reduces to: can the INDEX be made receiver-computable?** If it can, only the k
sign bits are video-derived, and every arm flips sign at once:

| arm | k | B/pair (signs only) | ΔS_rate | ΔS_pose | **NET ΔS** |
|---|---:|---:|---:|---:|---:|
| cheapest fixed | 5 | 0.6 | +0.00025 | −0.00064 | **−0.00039** |
| mid | 241 | 30.1 | +0.01204 | −0.01422 | **−0.00218** |
| **per-pair argmin** | 3,546 | 443.2 | +0.17709 | −0.29933 | **−0.12224** |

The argmin envelope is the one that matters: a free index turns a +0.575 loss into a **−0.122 win**,
because the sign bits are ~5x cheaper than naming the coordinates. **The whole verdict rests on this one
substitution**, and the prize is ~0.12 S, not the ~0.0004 S the cheapest arm suggests.

**So I measured it (n=12).** Two receiver-computable rankings (both need only `f1` and the homography
the receiver already builds), each given the SAME true signs, scored on the canonical scorer through
the REAL re-rendered `f0`:

| k | TRUE index d_pose | INFL index d_pose | INFL top-k capture | GRAD index d_pose | GRAD capture | chance |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.010945 | 0.011904 | 1.38% | 0.011906 | 0.46% | 0.03% |
| 1,000 | 0.009895 | 0.011788 | 3.17% | 0.011764 | 0.70% | 0.14% |
| 5,000 | 0.012318 | 0.011366 | 6.85% | 0.011207 | 2.59% | 0.72% |

(base 0.011944; INFL = warp→D influence column mass, GRAD = |Sobel(f1)| x influence.)

There **is** signal — INFL runs **47.8x / 22.0x / 9.5x** above chance at k=200/1,000/5,000 and GRAD
**15.9x / 4.8x / 3.6x** — but absolute capture is ≤6.9%, and the proxy-selected arms are **effectively
inert** (0.0119 -> 0.0114–0.0119) where the true index delivers −17% at k=1,000. Lift is not capture:
a ranking that is 48x better than chance still misses 98.6% of the coordinates that matter.
**REFUTED at FORMULATION scope: two proxies, not the proxy family.**

This independently re-derives `ddm_ll1`'s finding on a different surface — *"the free render-gradient
proxy captures only 18.68% of margin<2.0 px at top-5% (3.74x lift) — no free gate exists even if gating
worked."* Two unrelated surfaces, same wall: **the scorer's own sensitivity ordering is not cheaply
predictable from the render.**

## 6b. The zero-byte escape, measured and REFUTED (n=12)

The other way to flip the sign is to buy more `d_pose` at the SAME byte cost: fix the coordinate set
(the expensive part of the payload) from the initial gradient and **re-solve only the k one-bit
signs**, iterating the gradient at the perturbed point. Signs are always a single ±1 step from the
ORIGINAL `f1` (never accumulated), so the payload stays exactly 1 bit/coordinate — every extra
iteration is free.

| k | one-shot signs | best of 5 re-solves | **extra gain from re-solving** | already a fixed point at iter 1 | beats base |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0119042 (−4.00e-05) | 0.0119042 (−4.00e-05) | **+0.00e+00** | **10/12** | 12/12 |
| 200 | 0.0109449 (−9.99e-04) | 0.0108617 (−1.08e-03) | **−8.32e-05** | 2/12 | 4/12 |

(n=12, mean base `d_pose` 0.0119441.)

**REFUTED.** At the economically relevant small k the one-shot gradient sign is *already the optimum*
— 10 of 12 pairs are a fixed point after a single step, and re-solving buys exactly zero. At k=200 the
trajectory 2-cycles (a Jacobi all-coordinates-at-once artifact) and the extra gain is 8.3e-05, ~0.7% of
base — nowhere near the 2.9x-to-7.3x deficit it would have to close. **The cheapest owed measurement
came back negative; the deficit stands.** A Gauss-Seidel (one coordinate at a time) re-solve is
untested and is k-times more expensive to run, but it would have to find far more gain than the Jacobi
version to matter (at k=200 the Jacobi re-solve buys 8.3e-05 against a deficit measured in whole ΔS units).

## 7. Verdict scope and what is owed

**CLOSED at FORMULATION scope (this vehicle, this steering):** shipping an explicit per-pair
blind-coordinate correction index. Net ΔS positive at every measured k; best case (the per-pair argmin
envelope) loses by 2.92x, the cheapest fixed arm by 7.27x.

**NOT closed:**
- **The family.** The actuator is real, exactly seg-free, direction-selective, and only ~2x under water.
- **A receiver-computable index beyond the two proxies tested.** ≥~50% top-k capture would flip the
  sign at small k. Untested: margin/curvature-style rankings, or a ranking derived from the shipped
  pose payload itself.
- **A Gauss-Seidel sign re-solve.** The Jacobi version is refuted (§6b); a one-coordinate-at-a-time
  re-solve is k-times more expensive and would need ~90x the Jacobi gain to matter.
- **n600 for §4–§5.** The reach run is resumable (`--resume`) and was at 109/600 when this section was
  written; the economics table is a prefix number on a ~5x-harder-than-population sample. The run keeps
  being killed after ~40 pairs by something outside the process (it now runs under a self-restarting
  supervisor), so the prefix is a wall-clock artifact, not a design choice.

**Cross-cutting finding worth more than the verdict:** `d_pose` on this vehicle is a *relative* quantity
between the two delivered frames (§4.1). The frame_0 objective is not "resemble GT frame_0" — a real GT
frame_0 scores 3–17 against the decoded pair's 0.0008.

---

*STORES CONSULTED:* `D_is_disjoint_2x2_sampling_not_area_average_20260802` (direct input),
`ddm_ll1_window_solve.py` (blind mask + the index-does-not-rescue-a-gate precedent),
`inflate_runner_v4d.py` + `pfs1_warp_receiver.py` (the vehicle), `upstream/{modules,frame_utils,evaluate}.py`
(scorer authority), `v4d_verify_receipt.json` + `refine_receipt.json` (archive custody + the n600 pose
population), task #401 (blind-coordinate exploit, previously recorded but never composed with pose).
