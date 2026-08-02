# ddm_mq1 — the v4d pose menus, priced as codebooks: the rate axis is degenerate, and `s_t` is a normalizer, not a quantizer

**Axis:** `[macOS-CPU frozen-PoseNet advisory]` for every new measurement · `score_claim=false` ·
`promotion_eligible=false` · pointer `0.1910828242 [contest-CPU]` UNMOVED · own-vehicle live best
**S = 0.9476091** (ddm_pw1 exact-eval row, n600, archive 360,323 B).
**Gap to the bar (PR130 0.172141) = 0.7754681.** Every ΔS below carries its fraction of that gap;
1% of the gap = 0.0077547 S = 11,646 archive bytes.

**REVIEW STATUS:** pre-registered-only (own round-1 adversarial review applied; no fresh-eyes pass).

**STORES CONSULTED:** `tools/corpus_query.py "pose menu codebook quantizer resolution s_t translation
lattice"` — loaded `ddm_gd1_generic_default_census_20260731.md` (row P5, the pose chart),
`ddm_deferral_queue_ledger_20260729.md` (QA43 tail-targeted pose, QA79 warp kernel, QA82 census),
`ddm_ph3_realization_hybrid_adaptive_convocation_20260731.md` (expert-menu generalization),
`ddm_pw1_pose_menu_saturation_20260801.md` (the parent finding),
`comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md` (the
marginal-coefficient cargo-cult row). Deliberately NOT loaded: the HNeRV/PR-lineage intake corpus
(banned as a vehicle/calibration source), and the token/seg-side coder corpus (out of scope — this
memo touches only the pose payload).

---

## §0 Verdict, first

**The menu was never the binding constraint. Every STORAGE-FORMAT lever on the v4d pose payload —
menu placement, codebook size, quantization lattice, conditional entropy coding — is collectively
worth ≤0.056% of the gap. The SEARCH over the same variables is worth ≥1.82% on 48 pairs alone.
Format loses to search by 33×.**

MEASURED, n=48 mass-ordered pairs = 86.5% of the population `d_pose` mass, canary exactly 0.0,
`[macOS-CPU frozen-PoseNet advisory]`:

| pose coordinate | a finer LATTICE would buy | a better SEARCH would buy |
|---|---|---|
| `p0` forward — **negative control** (pw1 already bracketed it) | 0.0213% of gap | 0.1412% of gap |
| `p1` lateral | 0.0128% | **0.4694%** |
| `p2` vertical | 0.0107% | **0.8743%** |
| `beta` rolling-shutter | — (see §4) | **0.3358%** |
| all three per-pair INDEX streams at their conditional-entropy floor | 0.0106% (123 B) | — |
| **TOTAL** | **≤0.056%** | **≥1.82%** |

The exchange rate that makes this arithmetic possible: `dS/dB = 25/37,545,489 = 6.6586e-07` and
`dS/d(d_pose) = 5/sqrt(10·d_pose) = 18.083`, so **1 archive byte ≡ 3.68e-08 of mean `d_pose`**.
Distortion outweighs the entire pose rate axis by **48×**. An entropy-constrained quantizer design
(ECVQ/Lloyd–Max with a rate term) is therefore **degenerate here**: λ is too small to select
between designs, and the correct move is to spend bytes freely and search harder.

Four findings, each measured below:

1. **`s_t` is EXACTLY scale-degenerate with the pose translation columns** (§2) — zero added degrees
   of freedom, machine-epsilon identity. Its 189 bytes are duplicate data in the DOF sense. **And
   deleting them is still a NET LOSS**, because the redundancy is buying resolution: `s_t` is not a
   quantizer of a physical quantity, it is a **per-pair exponent in a two-level (coarse predictor +
   f16 residual) code**. Judged as a codebook it looks 64% wasted; judged as what it is, it works.
2. **The translation LATTICE is closed** (§3). The obvious cure for `p1`/`p2` — the QA65 mean-offset
   device that was worth `0.009196 S` on `p0` — buys **1.01× and 1.00×** and is dead.
3. **The 13-entry `beta` "menu" is not a codebook at all** (§4): every shipped value is a seed or a
   point on the Swann bracket's own doubling orbit `g₀ ± 0.5·(2^k − 1)`. Its occupancy histogram is a
   picture of the SEARCH'S REACHABLE SET, not of a solution density. Freeing it is worth 0.336%.
4. **The over-resolution reference was built and its positive control FAILED in the way that matters**
   (§5): objective agreement across starts is good (median |Δ| 0.35%), but **argmin agreement is only
   5/16**. The objective is flat and multi-modal, so no unbiased reference density over these
   parameters exists — **fitting any codebook to it is ill-posed**, which is the second, independent
   reason the codebook framing does not apply here.

---

## §1 Seed audit — what survived re-derivation

Every dispatch claim was re-derived at the primary artifact before being used as a premise.

| seed claim | source re-derived at | verdict |
|---|---|---|
| `rs_beta_mags` read from the archive manifest | `inflate_runner_v4d.py:127` | **CONFIRMED** |
| applied as `beta_mags[idx] * yaw_sign` | `inflate_runner_v4d.py:177,180` | **CONFIRMED** |
| `beta_idx` ships as `uint8` | `inflate_runner_v4d.py:114-115` | **CONFIRMED** |
| `dim0` is an f16 residual off a manifest offset | `inflate_runner_v4d.py:126,140-143` | **CONFIRMED** |
| pw1's menu extension cost +85 B | archive `stat`: 360,323 − 360,238 | **CONFIRMED**, and newly DECOMPOSED: **+39 B** deflated manifest table, **+46 B** widened index stream |
| `s_t` is an 11-point grid, occupied indices 6–9 only, zero at 0–5 and 10 | `pfs1_warp_receiver.py:18` + `pw1_arms.jsonl` n600 | **CONFIRMED EXACTLY**: occupancy `[0,0,0,0,0,0,22,364,156,58,0]` |
| `dim0`: 124 pairs at the bound, 37.4% of mass, 2.3× the interior mean | `refine.partial.jsonl` + `final_refine.jsonl` | **CONFIRMED**: 124 at `|move| ≥ 0.0475`, **37.37%** of mass, **2.29×** (0.01552 vs 0.00677) |
| beta shipped `[459, 65, 76]`, 76 at top entry, 26.4% of mass | `final_refine.jsonl` | **CONFIRMED** |
| beta's dominant win needs BOTH sign freedom and magnitude > 1.0 (29 pairs, 0.2196 d) | `pw1_receipt.json:arm_b_decomposition` | **CONFIRMED** |
| pw1 post-fix occupancy `[5,5,1,10,15,420,66,52,13,1,7,1,4]` | `final_pw1.jsonl` | **CONFIRMED EXACTLY** |
| `dim0` interior histogram `103,93,67,39,51,34,46,28,15` | both candidate reconstructions | **NOT REPRODUCED** — see below |

**The one seed defect.** The parent memo's interior histogram, and its reading that it *"decays
monotonically and then jumps at the bound"*, does not reproduce. On the memo's own bin edges I get
`[104,95,66,46,42,34,36,42,21]` using `final_refine.jsonl:p[0]` as the refined value, and
`[61,104,73,48,49,36,37,43,25]` using `refine.partial.jsonl:dim0_fine`. Both sum correctly
(486+114 and 476+124), and the **load-bearing** numbers — the at-bound count, the 37% mass share,
the 2.3× conditioning — reproduce under both. But **neither reconstruction decays monotonically**:
both rise from bin 0 to bin 1. The clipping signature in that finding rests on the terminal spike
and the mass concentration, which stand; the "monotone decay" clause does not, and should not be
repeated. This does not disturb the pw1 verdict or its exact-eval row.
`verdict_scope: INSTANCE` (one presentational histogram in one memo).

**S recomputed from components**, never from the rounded evaluator field:
`S = 100·0.00431179 + sqrt(10·0.00764541) + 25·360323/37545489 = 0.431179 + 0.276503 + 0.239924
= 0.947606`, against MAIN's measured `0.9476091` — the +2.5e-06 byte-close prediction error the
parent law already records. The pre-pw1 row recomputes the same way to 0.9639858 vs 0.9639878.

---

## §2 `s_t` is exactly degenerate — and deleting it still loses

`pose_to_homography` (`pfs1_warp_receiver.py:44-49`) uses the pose ONLY as

```
t = s_t * [p[2], p[1], p[0]]        R = expmap(s_r * [p[3], p[4], p[5]])
```

so scaling `(p0,p1,p2)` by `k` and `s_t` by `1/k` leaves the homography invariant. The `beta` sign
reads `p[5]`, a rotation component, which is untouched. **MEASURED: max relative homography
difference `5.98e-16` (machine epsilon) over 200 pairs × 4 scale factors × 3 rotation scales.**
And `_two_plane_static_gn` (`ddm_v4c_resolve.py:264-290`) runs damped Gauss–Newton over the **full
6-vector** — all three translation components are free. Therefore **`s_t` adds exactly zero degrees
of freedom**, and its 189-byte index stream is duplicate data in the sense of design-philosophy
clause A ("every byte names ONE geometric home").

**But deleting it loses.** The quantum of the reconstructed effective translation is
`s_t · ulp16(residual)`, and it is **scale-invariant** — `c·ulp(x/c)` does not depend on `c`, so no
choice of constant `s_t` recovers anything. What sets the quantum is the RELATIVE SPREAD of the
column the offset device sees, and that is exactly what the per-pair `s_t` is removing:

| design | quantum of effective `t0` | index cost |
|---|---|---|
| **A — shipped** (per-pair `s_t` + offset-f16 on `p0`) | **1.0046e-04** | 189 B |
| B — fold `s_t` into the pose, `s_t=1`, offset on the folded column | 4.9060e-04 (**4.9× coarser**) | 0 B |
| C — any CONSTANT `s_t` + offset on the rescaled column | 4.7406e-04 (**4.7× coarser**) | 0 B |

`p0` raw has relative spread 0.075; the effective `t0` has 0.266. The per-pair `s_t` is a coarse
speed predictor that strips 3.5× of the variance *before* the f16 residual is taken.

The trade, priced with the MEASURED coarsening→distortion transfer from `qa72a` (14.5× coarser
lattice ⇒ 5.92× the lattice gap): 4.9× coarser costs roughly 2.8× the current `p0` lattice gap,
whose contribution to the population mean is `80 × 8.063e-05 / 600 = 1.075e-05`. Extra
`Δd̄ ≈ 1.97e-05` ⇒ **+3.6e-04 S**, against a byte gain of `189 B = −1.26e-04 S`.
**Net ≈ +2.3e-04 S — a LOSS of about 0.03% of the gap.** The redundant menu is correctly retained.

*Census note:* `ddm_gd1_generic_default_census_20260731.md` row P5 marks the pose chart RACED
(warp vs cosine) and folds "s_t translation menu" into that row with no open item. The chart was
raced; the `s_t` normalizer inside it was never separately examined. That is a census gap, now
closed by this section.

---

## §3 The resolution allocation is lopsided — and the obvious cure is dead

Relative resolution each translation column receives at the effective translation (exact, no scorer):

| column | stored as | ulp(effective t) | RELATIVE ulp |
|---|---|---|---|
| `p0` forward | offset + f16 residual (QA65) | 1.00e-04 | **3.28e-05** |
| `p1` lateral | plain f16 | 8.63e-06 | 7.33e-04 (**22.4× coarser**) |
| `p2` vertical | plain f16 | 1.84e-05 | 7.05e-04 (**21.5× coarser**) |

Under the waterfill principle that disparity is only correct if the objective is ~22× less sensitive
to lateral/vertical translation than to forward — which nobody had measured.

**The QA65 cure does not transfer. MEASURED: applying the mean-offset device to `p1`/`p2` shrinks
their f16 residual by 1.01× and 1.00× — nothing.** The device works on `p0` (19.24× finer, and
worth a MEASURED `dim0_precision_gain_S = 0.009196`) precisely because `p0` has a large mean
(31.55) and a small spread; `p1`/`p2` are already zero-centred (mean −0.0409 / −0.0240 against std
0.226 / 0.487), so subtracting their mean moves nothing into f16's fine region.
`verdict_scope: FORMULATION` — mean-offset residual coding on the zero-centred translation columns.
Still open in the family: a shared per-pair exponent across `p1`/`p2`, or a wider mantissa.

---

## §4 The reformulation that survives — lattice vs search, and the beta ORBIT

The two dead ends above share a premise: that the loss is in how values are CODED. `qa72a` already
contradicted it for `p0` — offset-f16 sits `8.063e-05` above the continuous optimum, **0.12% of `d`**.

`tools/mq1_pose_lattice_resolution_probe.py` generalises that test to every translation column,
decomposing the recoverable distortion into two disjoint parts while holding all other shipped
variables fixed and accepting only strict decreases at the realized scorer:

- `gap_lattice = d(nearest shippable point) − d(continuous optimum)` — what a FINER LATTICE could
  buy; unreachable by more search.
- `gap_search  = d(shipped) − d(nearest shippable point)` — what a BETTER SEARCH could buy at
  TODAY's lattice; independent of the storage format.

Column `p0` is the **negative control**: `ddm_pw1` already ran a self-terminating bracket on it, so
a large `gap_search` there would mean the instrument finds floating-point noise, not real basins.
The canary (CTRL re-scoring the shipped solution) reproduces `d_final` to **0.0 exactly** on all 48
pairs, and 44/48 gaps clear that floor. The measured table is §0's; `p0`'s search gap is the
smallest of the three, as the control predicts, while `p2` is **6.2× larger** than it.
`moved_ulps` for `p2` reaches 2,985 — the solve left that coordinate thousands of lattice cells
from a better point. **That is a basin problem, not a precision problem.**

**The beta menu is the search's reachable set, not a codebook.** DERIVED FROM SOURCE:
`bracket_out` (`tools/pw1_pose_menu_saturation_ab.py:75-107`) probes `x₀ ± step₀` then doubles, so
from a seed it reaches only `g₀ ± BETA_STEP0·(2^k − 1)`. With `BETA_STEP0 = 0.5`
(`ddm_v4d_resolve.py:71`) and the seed sweep `(0.0, 0.5, 1.0)`, **every one of the 13 shipped
`rs_beta_mags` values is a seed or an orbit point, with no exceptions** (verified exhaustively).
Its spacing DOUBLES with distance from the seed, so it is coarsest exactly where `ddm_pw1` measured
its largest wins. `tools/mq1_beta_overfine_reference.py` re-runs the same search at a step **10×
finer** (0.05) plus a golden-section polish: **37/48 pairs improve, all above the noise floor,
gain 2.17% of the probe's `d_pose`, ΔS = −0.002604 = 0.336% of the gap.** The orbit's holes are
where the value was: pair 42 moved `−3.500 → −5.7891` (the orbit jumps −3.5 → −7.5, a 4.0-wide
hole) for a gain of 2.26e-02; pair 71 moved `+1.000 → +1.2661` for 2.72e-02.

---

## §5 The over-resolution reference — built, and its positive control is the real finding

Per the operator directive: never design a quantizer from quantized data; establish the reference by
solving at 10–100× the deployment resolution, then design down. Done (§4, 10× finer + polish). The
directive also requires the control that makes the reference admissible: **over-resolution removes
MENU censoring; it does not remove SOLVER bias.** ARM W restarts the identical search from a
deliberately wrong initialisation (`g = −7.5`, the far end of the shipped table), on every 3rd pair.

| positive control, n=16 | measured |
|---|---|
| median `\|recovery_rel\|` (objective agreement) | **0.00354** |
| objective agrees within 1% | **11/16** |
| **argmin agrees within 2 fine steps** | **5/16** |
| wrong-init found a STRICTLY BETTER optimum | **3/16** |
| wrong-init trapped (worse) | 2/16, worst `+3399×` |

**Verdict: objective-trustworthy, argmin-UNIDENTIFIED.** The objective values largely agree across
starts, so the measured GAIN is real. But the LOCATIONS do not agree, and in 3 of 16 cases the
from-shipped reference was itself the trapped one. The objective is flat and multi-modal in these
coordinates.

Two consequences, and they are the point of this memo:

1. **The gain is bankable.** Every arm is a monotone-safe continuation from the shipped solution
   accepting only strict decreases at the realized scorer, so the improvement is realized, not
   inferred. This is the same construction `ddm_pw1` banked.
2. **The emitted value distribution is NOT a reference density, and no codebook may be fitted to
   it.** An argmin that is not identified has no density to fit. Combined with the degenerate rate
   axis (§0), there are now **two independent reasons** the codebook framing does not apply to this
   payload — one economic, one statistical. Reporting a Lloyd–Max or objective-weighted-ECVQ menu
   derived from these values as "optimal" would be a fake-optimality claim.

**On the directive's point 3 (objective-weighted, not probability-weighted).** The correction is
right in general and is exactly what `ddm_pw1`'s 2.3× conditioning fact was signalling. It is
nonetheless **moot for this payload**, and the arithmetic says why: a codebook exists to trade rate
for distortion, and here the entire rate budget in play is 123 B = 0.0106% of the gap. With λ that
small the optimal codebook is "as many codepoints as you like" — i.e. ship the value — for ANY
weighting. The objective-weighted criterion would change WHICH menu wins only in a regime where
rate binds; it does not here. Recorded as the correct criterion for the next payload where it does.

---

## §6 What was actually built, and the byte-closed path

`tools/mq1_joint_pose_refine_emit.py` banks the search half as a chained (not independent)
coordinate refine in measured `gap_search` order `p2 → p1 → beta`, updating the pose between steps
so the reported joint gain is realized rather than assumed additive, and emitting a merged n600
JSONL in which unvisited pairs keep their shipped solution verbatim.

**Why this is close to rate-free.** `p1` and `p2` already ship as plain f16 columns
(`ddm_v4d_build_composed_archive.py:176-184`), so better values in them cost **ZERO additional
archive bytes**. `beta` ships as an index into the manifest table `rs_beta_mags`, which accepts any
float and needs **no receiver change at all** — only a widened table (~+150 B at ~50 distinct
values) and slightly higher index entropy.

**The one condition that would make a codebook necessary after all — found in round-2 review.**
`derive_beta_table` (`ddm_v4d_build_composed_archive.py:134`) **fails closed above 256 entries**
because `beta_idx` is `uint8`. A continuous per-pair beta therefore cannot scale past ~256 refined
pairs without quantisation. So the codebook this memo refuses on RD grounds becomes REQUIRED at
n600 — **forced by the storage format, not chosen by rate-distortion**. At that point REFUSAL 2
(unidentified argmin, §5) still applies, so the correct construction is not a fitted Lloyd–Max menu
but a plain 256-level quantiser over the realized values, which needs no density to exist. The
emit receipt carries `beta_table_uint8_headroom` so this is measured, never assumed; the staged
top-150 run leaves ample headroom.

**Byte-closed prediction, and how to check it.** With summed refined gain `G` over the visited
pairs, the predicted composed score is

```
S_pred = 100·0.00431179 + sqrt(10·(0.007645410 − G/600)) + 25·B_new/37545489
```

with `d_seg` bit-identical to the pw1 row by construction (frame_1 is never modified; SegNet reads
`x[:, -1]` only, `upstream/modules.py:108`), and `B_new` read from the built archive by `stat`.
The receipt `mq1_emit/mq1_emit_receipt.json` carries `delta_S_distortion_only` and the exact
`d_pose_mean_refined`; the byte term must be added from the built archive, never assumed.

**Staged, NOT self-fired** (MAIN owns the single n600 scorer slot; `stage_v4d_realized_gate.sh:3`
forbids self-firing):

```bash
# 1. build the candidate from the refined JSONL (encoder-side, no scorer)
.venv/bin/python experiments/ddm_v4d_build_composed_archive.py \
    --final-jsonl /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/mq1_emit/final_mq1.jsonl \
    --dim0-offset auto --tag mq1
# 2. MAIN fires the exact composed row when the slot is idle
bash experiments/stage_v4d_realized_gate.sh cpu mq1
```

Accept/verify: realized `d_seg` UNCHANGED at ~0.004312 (same tokens); realized `d_pose` per the
`mq1_emit` receipt; realized `S` below the pw1 row `0.9476091`.

---

## §7 Circularity — what I broke and what I could not

**Broken.** (a) The `s_t` degeneracy (§2) is an ALGEBRAIC identity in `pose_to_homography`,
independent of any observed distribution. (b) The quantum comparisons (§2) and the relative-ulp
audit (§3) are format-intrinsic — properties of the storage format, not fits to data. (c) The
lattice/search decomposition (§4) and the beta over-fine reference (§4-§5) find their optima with
self-terminating brackets whose reach is not bounded by the old menu, so they leave its support
entirely — pair 42's `−5.7891` is not expressible in the shipped table at all.

**NOT broken, and named rather than assumed away.** (i) Every probe holds the other variables at
their shipped values, and the shipped pose was solved at the shipped `s_t`; a from-scratch joint
re-solve under a different menu could land in a different basin, and nothing here bounds that.
(ii) The positive control (§5) proves the residual bias is REAL and not merely possible: the search
is start-dependent in argmin. **What would break both:** a from-scratch joint GN re-solve run under
two menu designs with identical seeds, schedules and starting frames (design-philosophy P5) — a full
re-solve, not a continuation, and not affordable inside this unit.
(iii) The occupancy histograms (§1) ARE conditioned on the old menus; they are used descriptively
only, and no design here is fitted to them.

**Cost of the reference, as the directive requires it be reported.** The over-fine beta reference
cost 48 pairs × ~13 s = ~10 min wall-clock; the lattice decomposition 48 pairs × ~20 s = ~16 min;
the chained refine ~20 s/pair. At n600 the chained refine is ~3.3 h — affordable but not free, which
is why the emit runs a mass-ordered top-150 (the 48-pair prefix already covers 86.5% of the mass)
and every unvisited pair keeps its shipped row. That subset is MEASURED and stated, never silent.

---

## §8 Verdict scopes

- `INSTANCE` — the parent memo's `dim0` interior histogram and its "monotone decay" reading do not
  reproduce (§1). Its load-bearing numbers are unaffected.
- `FORMULATION` — mean-offset residual coding on `p1`/`p2` is dead (1.01×/1.00×, §3). Untested in
  the family: a shared exponent across `p1`/`p2`; a wider mantissa; per-pair block-FP.
- `FORMULATION` — deleting or constant-ing the `s_t` stream is a net loss (§2). Untested: a
  normalizer DERIVED at decode from already-transmitted state (zero bytes, rule-118 free). Its
  ceiling is 0.016% of the gap, so it is correctly ranked below anything on the search axis.
- `FORMULATION` — entropy coding of the three index streams is closed (123 B ceiling). The streams
  are near-memoryless: first-order context moves `s_t` only 1.4434 → 1.4050 bits/pair.
- `FORMULATION` — **codebook/Lloyd–Max/ECVQ design over the v4d pose parameters** is refused on two
  independent measured grounds (degenerate λ; unidentified argmin). NOT a family kill: in a payload
  where rate binds AND the argmin is identified, the objective-weighted criterion of the operator
  directive stands and should be used.
- `NOT A NEGATIVE` — the search axis is OPEN and is where the remaining pose distortion lives
  (≥1.82% of the gap on 48 pairs, ≥33× the entire format axis).

---

## §9 The byte-closed candidate — BUILT and VERIFIED, staged for MAIN

The chained refine (§6) was run on the mass-ordered pairs and a candidate was **built and
decode-verified**, not merely predicted. Numbers below are MEASURED except the composed `S`, which
is a byte-closed PREDICTION until MAIN fires the gate.

| | pw1 (live best) | mq1 candidate |
|---|---|---|
| archive bytes | 360,323 | **360,702** (+379) |
| rate term | 0.239924296 | 0.240176656 |
| pose contribution `sqrt(10·d_pose)` | 0.276503 | **0.263923** |
| `d_seg` | 0.00431179 | 0.00431179 (bit-identical — same tokens) |
| **composed S** | **0.9476091** (exact-eval) | **0.9352782** (byte-closed prediction) |

**Predicted ΔS = −0.0123309 = 1.590% of the gap**, from 37 replaced pairs. The `+379 B` is
entirely the widened `rs_beta_mags` table plus its index entropy; the `p1`/`p2` improvements —
the larger share of the gain — cost **zero bytes**, because those columns already ship as plain
f16. Distinct beta values 44, uint8 headroom 212.

**Decode verification (`ddm_v4d_verify_decode.py`, archive sha `dbab7eb2da62d0db…`, 360,702 B) —
every leg PASSES:**

- `A_ok` — the #417 receiver-consumption bijection over all 600 pairs (every byte consumed).
- `B_ok` — `pose_reconstruct_exact`, `ab_bit_exact`, `selector_exact`, **`beta_exact`**: the
  receiver reconstructs the encoder's solution exactly, including the extended 44-entry table,
  with **no receiver change of any kind**.
- `C_ok` — `recompute_byte_exact` on 24 sampled pairs, with `two_plane_does_work` and
  `beta_path_exercised` both true (the changed paths are actually taken, not dead).

**Staged for MAIN — NOT self-fired** (`stage_v4d_realized_gate.sh:3` forbids self-firing; MAIN owns
the single n600 scorer slot). The archive already exists, so the gate is a one-liner:

```bash
bash experiments/stage_v4d_realized_gate.sh cpu mq1_partial
```

Accept/verify: `d_seg` UNCHANGED at ~0.004312; realized `S` below the pw1 row `0.9476091`;
prediction error against `0.9352782` expected at the `~2.5e-06` scale the parent law records.

**Still running at hand-off:** `tools/mq1_joint_pose_refine_emit.py --pairs 150` (resumable,
caches per pair, writes `mq1_emit/final_mq1.jsonl` on completion). Gains are already deep into
diminishing returns — the last 5 pairs added 0.0013 of 0.409 summed — so the fuller run is
expected to improve the candidate only marginally. To rebuild from it:

```bash
.venv/bin/python experiments/ddm_v4d_build_composed_archive.py \
    --final-jsonl /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/mq1_emit/final_mq1.jsonl \
    --dim0-offset auto --tag mq1
bash experiments/stage_v4d_realized_gate.sh cpu mq1
```

**Honest limit on this candidate.** It is a monotone-safe CONTINUATION of the pw1 solution, not a
re-solve: every arm started from the shipped point and accepted only strict decreases, so the
improvement is realized. It is NOT the optimum — the positive control (§5) proved the search is
start-dependent in argmin, and 3/16 wrong-init restarts beat the from-shipped search. The gain
banked here is a floor, not a ceiling.
