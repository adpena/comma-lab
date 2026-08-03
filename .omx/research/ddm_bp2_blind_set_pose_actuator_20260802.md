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
| **F2** achievable `abs(delta d_pose)` from a blind-only perturbation | `< 1e-4` closes | **1.7158** (n600, mean over pairs of the per-pair max over arms) | **REFUTED by 4.2 orders of magnitude** |

**Both legs of the charter's claim are confirmed.** The 230,904 blind pixels are exactly invisible to
both scorers, they ARE read by v4d's frame_0 warp, and perturbing them moves `d_pose` violently at
literally zero seg cost.

**And the family still does not pay.** The steering direction is video-derived, so it must be shipped,
and at **every** measured operating point the rate cost of naming the coordinates exceeds the pose gain
— by **17.8x at the cheapest fixed arm** and, at best, by **3.89x** at the per-pair argmin envelope
(n600). **And it does not pay even if the index were FREE** (best arm NET +0.00001 S, argmin +0.152 S),
which removes the one substitution that could have flipped it. This closes the shipped-correction
formulation firmly; §7 scopes what remains open.

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

**§4–§5 are now n600 — and the prefix I reported earlier was materially misleading.** The run
completed as 6 resumable shards (600/600 unique pairs, zero duplicates, all 6 residues). The earlier
prefix (n=73–181) had mean `d_pose` 0.0390 against the true population's **0.0076425** — it was
**5.1x harder than the population**, and it *flattered the family*: it made the free-index escape look
like a −0.122 S win when at n600 it is a **+0.152 S loss** (§6). The population is heavily skewed
(median 0.0008154, p90 0.0049012, max 0.7749889), so a video-order prefix is not a sample of it.
**This is the whole reason the n600 bar exists, and it changed a conclusion, not just a decimal.**

Robustness at n600, on four disjoint quarters (argmin cost/gain, `>1` = does not pay):
**2.47 · 17.57 · 25.67 · 4.78**, all 600 together **3.89**. The magnitude still swings an order of
magnitude between quarters; the sign never does.

**d_seg is EXACTLY unchanged, in every arm, on all 600 pairs** — including under a full ±1 LSB
gradient-aligned sign step over all 692,712 blind coordinates, the largest structured 1-LSB
perturbation the channel admits. Not "within tolerance": bit-identical `d_seg`. This is the
zero-seg-cost claim, confirmed through the real frozen SegNet rather than argued from the geometry.

**The channel's gain is enormous and direction-selective:**

```
mean d_pose base                                    0.0076425   (n600; pose term 0.276450)
full +-1 LSB gradient-sign step, descent direction  0.88914     (x 116)
full +-1 LSB gradient-sign step, ascent direction   1.17851     (x 154)
mean max|delta d_pose| over all arms                1.7158      (F2 threshold 1e-4)
RANDOM-sign step at the SAME coordinate count       0.0076892   (+0.61%)  <- control
gradient L1 mass carried by the blind set            13.07%     of the whole frame's
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

| target t | mean k | mean d_pose | ΔS_pose | B/pair | ΔS_rate | **NET ΔS** | cost/gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0 | 0.0076425 | — | — | — | — | — |
| 0.002 | 6 | 0.0076253 | −0.00031 | 14 | +0.00552 | **+0.00521** | 17.78x |
| 0.01 | 43 | 0.0075956 | −0.00085 | 86 | +0.03430 | +0.03345 | 40.40x |
| 0.05 | 397 | 0.0078952 | **+0.00453** | 575 | +0.22978 | +0.23432 | ∞ (no gain) |
| 0.15 | 2,315 | 0.0117751 | +0.06670 | 2,249 | +0.89858 | +0.96528 | ∞ |
| 0.35 | 19,330 | 0.0304178 | +0.27507 | 12,361 | +4.93823 | +5.21330 | ∞ |
| 0.7 | 96,172 | 0.0880964 | +0.66215 | 43,432 | +17.35175 | +18.01390 | ∞ |
| 1.0 | 181,337 | 0.1514290 | +0.95411 | 67,494 | +26.96507 | +27.91918 | ∞ |
| **per-pair argmin** | 5,350 | 0.0026076 | **−0.11497** | 1,120 | +0.44738 | **+0.33241** | **3.89x** |

82.2% of pairs improve under the per-pair argmin, and the envelope cuts `d_pose` by **65.9%**
(0.0076425 -> 0.0026076, ΔS_pose −0.115). The arm index itself is negligible (3 bits/pair = 225 B).
**NET ΔS is positive at every single operating point.** At n600 only the two smallest arms produce any
pose gain at all — from t=0.05 upward the "descent" step overshoots and *raises* `d_pose`. The tightest
deficit anywhere is **3.89x**, at the argmin envelope.

Per-pair argmin over a deterministic score is a *realizable encoder-side choice*, not selection on
noise — d_pose is deterministic on fixed pairs and the choice is reproducible. The grid was widened
mid-run (from a top target of 0.05 to 1.0) precisely because a partial run showed the top arm binding;
a grid whose top arm is selected reports a LOWER bound on the achievable gain, and the verdict turns on
that number. With the widened grid the top two arms are now *worse* than base, so the interior optimum
is bracketed and the bound is no longer loose on that side.

## 6. What would flip the sign — and the measurement that refutes the two cheapest candidates

**The obvious escape — make the INDEX receiver-computable so only the k sign bits ship — DOES NOT
WORK AT n600.** This is the conclusion the prefix got wrong, and it is worth stating plainly: on the
n=73 prefix this substitution turned a +0.575 loss into a −0.122 *win*, and I wrote it up as "the whole
verdict rests on one substitution." At n600 it does not:

| arm | k | B/pair (signs only) | ΔS_rate | ΔS_pose | **NET ΔS** |
|---|---:|---:|---:|---:|---:|
| t=0.002 | 6 | 0.79 | +0.00032 | −0.00031 | **+0.00001** |
| t=0.01 | 43 | 5.39 | +0.00215 | −0.00085 | +0.00130 |
| **per-pair argmin** | 5,350 | 668.81 | +0.26720 | −0.11497 | **+0.15223** |

**Even with a perfectly free index the family does not pay.** The best arm lands at NET **+0.00001 S** —
indistinguishable from zero, on an arm that touches 6 coordinates. The argmin still loses by 1.6x. The
prefix was optimistic because a 5.1x-harder sample has far more pose headroom per shipped bit; the real
population's `d_pose` is already small (median 8.2e-4), so there is very little left for the actuator to
take. **The escape is closed by arithmetic, not by the proxy measurement below — which now serves as
independent corroboration rather than the load-bearing leg.**

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

**CLOSED at FORMULATION scope (this vehicle, this steering), now on n600 evidence:** using the blind set
as a shipped per-pair pose correction. NET ΔS positive at every k; best case 3.89x under water. **The
free-index escape is closed too** — at n600 even a zero-cost index leaves the best arm at NET +0.00001 S.
Two independent legs now agree: the arithmetic (§6) and the proxy measurement (§6, ≤6.9% capture).

**Why it fails, in one line:** the actuator's reach is enormous (116x-154x) but almost entirely
DESTRUCTIVE, and the population's `d_pose` is already small (median 8.2e-4) — there is very little left
for a steering channel to take, and every bit of steering must be shipped.

**NOT closed:**
- **The family.** The actuator is real, exactly seg-free, and direction-selective. What is refuted is
  every *pricing* of it tried here.
- **A cheaper description than per-coordinate.** Everything measured prices the correction per
  coordinate. A *parametric* blind-set perturbation — one whose k coordinates and signs are generated
  from a handful of shipped scalars — was never tested and is the only untried shape that could beat
  the arithmetic. It needs a generator whose output correlates with the gradient far better than the
  two rankings in §6 did.
- **The Gauss-Seidel sign re-solve** (§6b refutes only the Jacobi form).
- **The same actuator on a vehicle with LARGER `d_pose`.** The deficit scales with how much pose error
  is available to remove; on the 5.1x-harder prefix the argmin was only 2.31x under water. A future
  vehicle whose pose term is materially worse would move this verdict, and the measurement is cheap to
  repeat — the tool takes an `--archive`.

**Cross-cutting finding worth more than the verdict:** `d_pose` on this vehicle is a *relative* quantity
between the two delivered frames (§4.1). The frame_0 objective is not "resemble GT frame_0" — a real GT
frame_0 scores 3–17 against the decoded pair's 0.0008.

---

*STORES CONSULTED:* `D_is_disjoint_2x2_sampling_not_area_average_20260802` (direct input),
`ddm_ll1_window_solve.py` (blind mask + the index-does-not-rescue-a-gate precedent),
`inflate_runner_v4d.py` + `pfs1_warp_receiver.py` (the vehicle), `upstream/{modules,frame_utils,evaluate}.py`
(scorer authority), `v4d_verify_receipt.json` + `refine_receipt.json` (archive custody + the n600 pose
population), task #401 (blind-coordinate exploit, previously recorded but never composed with pose).

---

## 8. Provenance of the n600 run

Completed as **6 resumable shards** (`--pair-stride 6 --pair-offset J`), merged to 600 unique pairs,
zero duplicates, all six residues present. Per-pair RNG seeding makes a sharded run bit-identical to an
unsharded one (verified: 35/35 numeric keys on a shared pair). The run had to be re-fired four times —
three launch strategies (`nohup`+`disown`, a self-restarting supervisor, `os.setsid` double-fork) were
all killed after ~30-40 pairs; `--resume` never repeats a measured pair, so each re-fire only added work.

**Guards, all 600 pairs:** `d_seg` bit-identical under the full 1-LSB blind step — **600/600**;
cached-GT fast path equals the authority — **600/600**; gradient surrogate within 1e-5 relative of the
authority — **518/600** (max relative deviation **5.46e-04**, i.e. the surrogate agrees to ~3-4
significant figures; the 1e-5 gate is tighter than fp32 graph-order noise warrants, and the surrogate is
only ever a search direction — every reported number comes from the unpatched authority path).

**Instrument warning — the SYMPTOM is real, and my first stated MECHANISM was wrong.** Twice,
`pgrep -f shard_worker.sh` reported live workers when every worker was dead. I wrote that up as "pgrep
matches the shell running it." **That is REFUTED by control, in two independent harnesses:**

- MEASURED (this arm's context): `pgrep -fl "ZZUNIQ_bp2_$$_selftest"` with the token literally in the
  command line returns **rc=1**, and the probing shell's own pid does not appear in any match list.
- MEASURED (MAIN's harness, same machine): a unique-token probe likewise returns **rc=1**.

The actual mechanism, MEASURED here by reading the matched command lines: the loose pattern returned
**10** matches = **6 real** `/bin/bash <path>/shard_worker.sh 6 J` workers **+ 4 of MY OWN leftover
`zsh -c` monitor/waiter shells**, each of which carries `pgrep -f shard_worker.sh` inside its own loop
body and therefore matches the pattern. Stale monitors accumulate; they are indistinguishable from
workers under a loose pattern. Anchoring on the exec form
(`pgrep -f '^/bin/bash /private.*shard_worker\.sh [0-9]'`) returns exactly **6**, the right answer.

**LAW (survives both measurements, and needs no mechanism claim):** use **row-count / receipt-existence**
for liveness, not a pattern-based process probe. A pattern probe can match processes that merely
*mention* the pattern — most insidiously your own monitors — and from inside the probe you cannot tell
which case you are in. If a process probe is unavoidable, anchor it on the executable form and print
the matched command lines rather than a count. *(Scope: the self-match claim is refuted in these two
contexts; whether some other shell invocation self-matches is untested and not relied on.)*
