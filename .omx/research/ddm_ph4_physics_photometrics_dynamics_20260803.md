# ddm_ph4 — PHYSICS / PHOTOMETRICS / DYNAMICS / INTERACTIONS: the follow-on

- **arm:** `ddm_ph4` · **date:** 2026-08-03 · **axis:** `[macOS-CPU advisory]` NON-PROMOTABLE.
  `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`. **Pointer UNMOVED.**
- **cost:** $0. **ZERO scorer forward passes** — `ddm_pu2` holds the n600 slot. Every number below is
  either read from a custody receipt (cited) or MEASURED here by driving the **exact shipped
  receiver on the exact shipped bytes**. Pairs are **STRIDED across all 600**, never a prefix (`m88`).
- **live best (the baseline every ΔS below is named against):** `ddm_pu2`, **S = 0.7910689**,
  353,805 B — seg **0.4311790** · pose **0.1243037** · rate **0.2355842** (recomputed from bytes:
  25 × 353,805 / 37,545,489; components re-sum to 0.7910669, **−2.0e-06** against the quoted
  0.7910689 — quoted-component rounding, recorded rather than smoothed).
- **target:** the PR130 bar **0.172141** (seg 0.02966 · pose 0.015268 · rate 0.127214).
  **Gap = 0.6189279.** 1% of the live gap = **9,295 B** (= 0.006189279 × 37,545,489/25).
  Live gap decomposition, recomputed at the `pu2` row: **seg 0.4015190 = 64.87% · pose 0.1090357
  = 17.62% · rate 0.1083702 = 17.51%.** Seg is now **≈2/3** of everything left, because both of
  today's wins were pose.

Operator directive 2026-08-03 (P0, verbatim): *"Over the past forty eight hours, you found a lot of
very interesting physics and photometrics and dynamics and interactions and synergy, and we need to
pursue all related follow on work as p zero."*

---

## §0 Headline

**One measurement, and it inverts a "do not spend" verdict that was 12 hours old.**

> **The 230,904 `D`-blind camera pixels are an EXACTLY seg-free POSE actuator with 692,712 dimensions
> per pair, and nothing in the vehicle has ever written to them.**
>
> MEASURED, live `cx1`, 4 strided pairs × 5 amplitudes = 20 cells, all four controls passing:
> a step confined to the blind set moves SegNet's input by **`0.0e+00` — exactly zero, every cell** —
> while moving PoseNet's frame_0 scorer-plane input by a gain that converges to **0.2231 LSB per LSB**
> against the blind set's population share of **0.226969** (agreement **1.7%**).
>
> At amplitude 8 the blind actuator delivers **scorer-plane f0 rms 1.7905** — the *same* authority over
> PoseNet's input that `ll1`'s entire window solve delivered (**1.6986**, `pz1` §2) — but at **exactly
> zero** seg cost instead of that rung's measured **+0.000394**.

`ddm_rz1` §3.6 ranks this set **PREDICTED NULL — "read by neither scorer … do not spend on these"**
and §1 R1(c) says it **"must receive zero bits."** `ddm_ra1` §6 says the opposite in one clause —
*"the blind set is blind to `D` but **not** to the warp"* — and never followed it up. **`ra1` is
right, `rz1`'s R1(c) is refuted, and the consequence is an actuator, not a correction to a footnote.**

**Three further results, each a follow-on nobody had run:**

**(2) `rz1`'s rank-2 attack is not what it claims.** A2 ("pose-free chroma steering") is ranked #2
on being *"the only attack that is exactly pose-free by construction."* It is not — on **this**
vehicle. `rz1` §5 assumption #13 lists *"whether our vehicle emits frame_0 independently or as a warp
of frame_1"* as **`UNKNOWN` — blocks A3**, and applied that caveat only to A3 (rank 4). **`ra1`
already resolved it — frame_0 IS a warp of frame_1's camera raster — and the answer is the bad
branch, which lands on A2 exactly as hard as on A3** (§3.2). MEASURED here: a camera-plane edit on the
`D`-**visible** set (the set A2 writes to) delivers **0.8902 LSB per LSB** into PoseNet's frame_0
input — **4.0×** the blind set's, and five-to-six orders of magnitude above the ≤0.9-LSB uint8 residue
`rz1` pre-registered as the expected break. **A2 is re-scoped, not killed**: a real seg actuator that
must PAY a real pose cost rather than assume it away.

**(3) The seg-gap accounting is now exact, and rasterization is on the wrong side of zero.**
`ra1` §0 concluded *"≥91% of the seg gap is DESCRIPTION error, not rasterization"* from a **claimed**
0.014–0.035 realization bound. `pz1` then measured that bound at **−0.000394** — removing the
rasterization makes seg **worse**. Updated: **100.0% of the seg gap is description error;
rasterization contributes −0.098% of it (it HELPS).** Nobody had written the corrected number.

**(4) The FITTED-THROUGH law — `pz1`'s stale-carrier law, which was found on the pose side, applies
to seg, and `pz1`'s `+0.000394` is its first price tag.** The tokens and the LOTTO renderer were
searched *through* the raster. The raster is therefore not an error term to be removed; it is part of
the operator the description was fitted through. This is what `43.3% improve / 49.3% worsen` looks
like, and it is now a number `ra1` §4.2's `np.repeat` re-race can be scoped against (§3.4).

**And one refutation of my own, by my own instrument.** My first 6-pair run gave a blind-set
"leverage" of **0.3213** against a population share of 0.226969 and I began writing up a **1.416×
over-weighting**. The amplitude sweep killed it: the gain decays monotonically to **0.2231** and the
excess at amp 1 is a **uint8-round threshold amplification of 1.204×**, not geometry. §6.

---

## §1 Receipt-verification ledger — checked before anything was built on it

Every load-bearing premise re-verified at its own source, not inherited from the memo that cited it.

| # | claim | verified how | status |
|---|---|---|---|
| 1 | **SegNet reads frame_1 ONLY, then `D`** | `upstream/modules.py:107-109` read directly: `x = x[:, -1, ...] # Use only last frame` then `interpolate(..., size=(384,512), mode='bilinear')` | `VERIFIED_VIA_SOURCE_INSPECTION` |
| 2 | **PoseNet and SegNet share the identical `D`** | `modules.py:73` vs `:109` — identical call; `pz1` §0(4) found CLAUDE.md's stated order backwards | `VERIFIED_VIA_SOURCE_INSPECTION` (independently re-read here) |
| 3 | **frame_0 is a WARP of frame_1's camera raster** | `v4d_cx1_pj2ix2/inflate_runner.py:307-327` read in full: `_warp_pair(f1_f, pose, s_t, sel, …)`, then `a*f0f + b`, then `_to_uint8` | `VERIFIED_VIA_SOURCE_INSPECTION` — **this is the fact `rz1` listed as UNKNOWN** |
| 4 | blind set = 230,904 px = 22.696926% | `tac.optimization.ddm_ll1_window_solve.blind_mask()` executed; matches `rz1` R1 and `ra1` independently | `VERIFIED_VIA_EMPIRICAL_ANCHOR` |
| 5 | `D` is a disjoint partition, ≤1 scorer px per camera px | `rz1` §1 R1 (200-site brute force) — **inherited, not re-measured**; my §2 result is consistent with it but does not re-prove it | inherited |
| 6 | `ll1` window solve: ΔS(seg) +0.000394, ΔS(pose) +0.0310208, n600 | `pz1` §3.2/§5, positive controls 1.2e-5 and 1.2e-6 rel vs `report.txt` | inherited (custody receipt) |
| 7 | camera-raster debt is 93.5% float / 6.0% quantization | `ra1` §0(2) — **inherited, not re-measured** | inherited |
| 8 | live-best S/bytes/legs | `.omx/state/main_hot_state.md` POINTER_LINE (`pu2`, archive sha `c72ef357`) | READ |

---

## §2 THE MEASUREMENT — the blind set is seg-free by proof and pose-active by measurement

**Instrument:** `experiments/ddm_ph4_blind_set_pose_reach.py`.
**Artifact:** `.omx/research/ddm_ph4_blind_set_pose_reach_cx1_20260803.json`.
**Equations leg:** `src/tac/canonical_equations/ddm_ph4_blind_set_seg_free_pose_actuator_20260803.py`
— `seg_delta_per_lsb` · `passthrough_gain` (refuses any support it has not MEASURED, so the
non-additivity trap in §7 A5 cannot be repeated) · `null_space_claim_survives` (`pz1` §7.1 as an
executable gate, refusing an empty consumer list as VACUOUS) · `actuator_overdetermination`.
**Base:** `v4d_cx1_pj2ix2` (the newest submission directory on the SSD; the `pu2` row differs from it
only in pose knobs on 6 tail pairs, so the *geometry* measured here is the live geometry — scope
stated in §6).

### 2.1 Why the seg half is a PROOF, not a statistic

SegNet has exactly **one** path to the frames: `x[:, -1, ...]` → `D`. `D` is bilinear point-sampling
at stride 2.276 > 2, so it reads 768 of 874 rows and 1024 of 1164 columns and never touches the other
230,904 camera pixels. **An edit confined to those pixels cannot change SegNet's input** — not
approximately, not to within a quantization residue, but exactly, and with **no lattice caveat**,
because there is no second lattice for SegNet to read through.

Measured, and it is the literal `0.0`:

| | value |
|---|---|
| `max abs( D(f1 + δ_blind) − D(f1) )`, **all 4 pairs × all 5 amplitudes = 20 cells** | **`0.0e+00`** |
| **C3** — the same edit on a **cardinality-matched** (230,904 px) **`D`-visible** random subset | **1.000**, every pair |

**C3 is the control that makes the zero mean something.** Without it, a broken warp call and a
genuine null are the same symbol (`m50`). The treatment and the control differ only in *which*
pixels move, never how many.

### 2.2 The pose half — the DYNAMICS law

`ΔS`-relevant quantity: the delta in the literal tensor PoseNet's frame_0 half consumes,
`D(f0(f1+δ)) − D(f0(f1))`. Mean over 4 strided pairs (0 / 200 / 399 / 599):

| blind step (LSB) | camera f0 rms | camera f0 %chg | **scorer f0 rms** | **GAIN (rms/amp)** | scorer f0 %chg | **seg delta max** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4700 | 22.1% | 0.2686 | **0.2686** | 25.0% | **0.0e+00** |
| 2 | 0.8631 | 29.3% | 0.4728 | **0.2364** | 31.4% | **0.0e+00** |
| 4 | 1.6840 | 33.7% | 0.9057 | **0.2264** | 35.5% | **0.0e+00** |
| 8 | 3.3452 | 36.6% | **1.7905** | **0.2238** | 38.5% | **0.0e+00** |
| 16 | 6.6781 | 38.3% | 3.5702 | **0.2231** | 40.2% | **0.0e+00** |

**Two laws fall out, and they are different laws.**

**(a) The asymptotic gain IS the population share.** 0.2231 against 0.226969 — agreement to **1.7%**.
The homography is an area-preserving interpolation, so each camera pixel contributes its area
fraction of the warp's attention and the blind set is neither privileged nor penalised. Control
**C4** anchors the scale: an **all-pixel** unit step gives scorer f0 rms **0.9809** (≈1.0, as an
interpolation with unit-sum weights must).

**(b) The small-signal regime is ROUNDING-MEDIATED and 1.204× MORE efficient per LSB.** At amp 1 the
camera f0 delta is rms 0.4700 with 22.1% of pixels changed — and
`sqrt(0.221 × 1²) = 0.4701`, i.e. **the changed pixels moved by exactly ±1 LSB and nothing else moved
at all.** The sub-LSB warp response is being *thresholded* by `_to_uint8`'s round. That threshold is
why the amp-1 gain (0.2686) sits 20.4% above the linear asymptote (0.2231). **The cheapest actuator
step is the most efficient one per LSB** — which is the good direction for a byte-limited actuator,
and is a fact about `_to_uint8`, not about the scene.

### 2.3 The comparison that prices it

Same base, same chain, same scorer plane, same instrument family:

| perturbation of frame_1's camera raster | **scorer-plane f0 rms** | **Δd_seg (n600)** | Δd_pose (n600) |
|---|---:|---:|---:|
| `ll1` window solve (`pz1` §2, §5) | 1.6986 | **+0.000394** | +0.00108726 (ratio 1.4261) |
| **`ph4` blind step, amp 8** | **1.7905** | **EXACTLY 0** | **UNMEASURED** — needs one scorer pass |

**Equal authority over PoseNet's input; the seg cost is replaced by an exact zero.** The window solve
paid a measured seg loss for an *unaimed* pose perturbation that went the wrong way. The blind set
gives the same-sized pose perturbation for free on seg, and the direction is ours to choose.

**Capacity.** 230,904 px × 3 channels = **692,712 dims per pair**, against **6** pose scalars per
pair — over-determined **115,452×**. For scale, the live vehicle's entire pose grammar is **11 knobs**
(`s_t`, `sel`, `ab`(2), `beta_idx`, `p_best`(6)), and `pj2` measured it **already at its discrete-grammar
optimum on 95.0% of pairs** (`pz1` §4). This is not a refinement of a saturated actuator — it is a
disjoint one.

---

## §3 THE IMPLICATION TABLE — what each measured fact implies that nobody has run

Ranked by whether a **$0 scorer-free** measurement settles it. Facts are cited, never re-derived.

| # | measured fact (receipt) | **implication nobody had run** | settleable at $0? | status |
|---|---|---|---|---|
| **I1** | `D` is a disjoint partition; blind set 230,904 px (`rz1` R1) **×** frame_0 is a warp of frame_1 (`ra1`, re-verified §1) | the blind set is an **exactly seg-free pose actuator**, not dead bits | **YES** | **DONE, §2.** `rz1` R1(c)/§3.6 REFUTED |
| **I2** | shared `D` **×** lattice law (`pz1` §7.1) **×** frame_0-is-a-warp | **`rz1`'s rank-2 A2 is not "exactly pose-free by construction"** on this vehicle; its falsifier fires on the WARP, not the uint8 lift it named | **YES** | **DONE + MEASURED, §3.2** — leak gain **0.8902** |
| **I3** | `pz1` ΔS(seg) = **+0.000394** vs `ra1`'s claimed −0.014…−0.035 | **100.0% of the seg gap is description error**; rasterization is worth **−0.098%** of it (it helps) | **YES** | **DONE, §3.3** |
| **I4** | 43.3% improve / 49.3% worsen; the description was searched *through* the raster | **the FITTED-THROUGH law** — a delivery-operator change is a distribution shift on a fitted description, and `+0.000394` is its first price tag | **YES** | **DONE, §3.4**; scopes `ra1` §4.2 |
| **I5** | Q3 = 294,912 pose-free dims (`rz1` R2) **×** I1 | the blind set is the natural **compensator** for a Q3 chroma edit's warp leakage: Q3 moves seg with pose leakage, blind pays pose with exactly zero seg. **They compose.** | no — needs the scorer | **§4, the composed law** |
| **I6** | `hg1` barrier integral spans **5.10 → 58.39** (11.4×) **×** `W` = 1.273108215332031 B/flip is UNIFORM | the **effective** exchange rate is **per-edge**; a waterfill allocating by flip count is misallocated by up to 11.4× | yes, desk join | **ROUTED to `wf2`** (owns the PRICE law) — cite, do not duplicate |
| **I7** | `hg1` **directed** barriers: Road→Lane **51.26** vs Lane→Road **5.10** **×** `d_seg` prices both at 1 flip | **a flip-count objective is PAID to erase lanes, at a 10.05× discount** — a mechanism for the corpus's standing LANE-ERASURE failure mode, from two measured tables never multiplied | yes, desk join | **DONE (desk), §3.5** → **ROUTED to `as1`** (owns ASYMMETRY-synergy) |
| **I8** | asymptotic gain = population share (§2.2a) | the warp is **area-preserving in its attention** ⇒ a blind-set actuator's authority is a fixed 22.70% tax, **scene-independent**, so its capacity can be budgeted before any scorer pass | **YES** | **DONE, §2.2** |
| **I9** | rounding-mediated small-signal gain 1.204× (§2.2b) | the **±1 LSB** blind step is the most byte-efficient one; an actuator quantized to ±1 loses nothing to the nonlinearity and gains 20% | **YES** | **DONE, §2.2** |

### 3.1 — I1 in one paragraph (the full consumer enumeration `pz1` §7.1 demanded)

`pz1` §7.1's operational rule is: *"before claiming a receiver-side lever is free because it lives in
a scorer's null space, enumerate every consumer of the modified bytes and re-check the null-space
claim against each consumer's own operator and lattice."* Executed, exhaustively, for an edit to
frame_1's camera raster — **three** consumers, **two** lattices:

| consumer | operator | blind-set edit visible? |
|---|---|---|
| SegNet | `D(f1)` | **NO** — exactly, `0.0e+00` (§2.1) |
| PoseNet, frame_1 half | `yuv6(D(f1))` | **NO** — same `D`, same lattice, same proof |
| PoseNet, frame_0 half | `yuv6(D(W(f1)))` | **YES** — gain 0.2231 (§2.2) |

That is the complete list (the receiver writes frames; `evaluate.py` feeds exactly these two networks).
The asymmetry — invisible to two consumers, visible to the third — is the actuator.

### 3.2 — I2: `rz1`'s rank-2 attack, re-ranked

`rz1` §3.2 ranks **A2 (pose-free chroma steering)** at **#2**, on this basis: *"It is the only attack
that is **exactly pose-free by construction** (`Q3`: pose max|Δ| = 5.684e-14)."* Its pre-registered
falsifier: *"Kill the exactness claim if |Δd_pose| > 1e-6 — which simultaneously tests R4's prediction
that the uint8 camera lift breaks exact pose-nullity"* at the **≤0.9 LSB** scale.

**The exactness claim is already dead, and not for that reason.** `Q3` is exact for a consumer reading
through `D` at `D`'s own lattice. A2's realization path — `rz1` §3.2's own prescription — is *"design
in the scorer plane, then lift by setting all four private camera pixels to the same target"*, i.e.
the edit is **emitted in the camera plane**, which is the only thing an archive can emit. `W`
resamples that camera plane at sub-pixel offsets onto a different lattice, where `yuv6`'s
block-mean-preserving structure is not preserved. **This is `pz1` §7.1 verbatim, applied to A2 instead
of to the window solve.**

`rz1` knew the blocking fact was missing — §5 assumption **#13**, *"whether our vehicle emits frame_0
independently or as a warp of frame_1"*, marked **`UNKNOWN`** — but attached it only to **A3** (rank 4).
**It lands on A2 identically.** And `ra1` had already answered it from the shipped bytes.

**Magnitude — MEASURED, not inferred, and it is not a rounding residue.** A2's edit lives on the
`D`-**visible** complement (786,432 px = 77.303074%). Measured directly, same instrument, same 4
strided pairs, `--edit-set visible`
(`.omx/research/ddm_ph4_visible_set_warp_passthrough_cx1_20260803.json`):

| step (LSB) | scorer-plane f0 rms | **GAIN** | f0 %chg | seg-plane delta max |
|---:|---:|---:|---:|---:|
| 1 | 0.9033 | **0.9033** | 97.9% | 1.0 |
| 8 | 7.1249 | **0.8906** | 100.0% | 8.0 |
| 16 | 14.2427 | **0.8902** | 100.0% | 16.0 |

> **A camera-plane edit on the `D`-visible set delivers ~89% of its amplitude into PoseNet's frame_0
> input.** That is **4.0×** the blind set's 0.2231, and it is **five to six orders of magnitude above**
> the ≤0.9-LSB uint8 residue `rz1` pre-registered as the expected break. A2's exactness claim does not
> survive contact with this vehicle's warp.

*(The seg-plane column is a second free control: a visible edit of amplitude `A` moves SegNet's input
by exactly `A` — `D`'s weights sum to 1 within each private window — which is the exact mirror of
§2.1's `0.0e+00` and confirms both from the same run.)*

**A2 is NOT killed.** It is re-scoped: it remains a real seg actuator with a real (55.0% isoluminant
efficacy) direction, but it is a **pose-COSTED** one, and its cost must be paid rather than assumed
away. §4 names the payer.

### 3.3 — I3: the seg-gap accounting, corrected

`ra1` §0 R1: *"Camera-raster realization is bounded at roughly 0.014–0.035 S = 3.5–8.7% of the 0.4015
seg gap. **≥91% of the seg gap is DESCRIPTION error, not rasterization.**"* That bracket was
`INFERRED` from `ll1`'s ΔS measured against the **ideal render**, not GT — `pz1` §5.1's finding.

Measured (`pz1` §5, n600, positive-controlled to 1.2e-6 rel): removing the rasterization costs
**+0.000394**.

| | seg term | vs seg gap 0.4015190 |
|---|---:|---:|
| live (`pu2`, rasterized) | 0.4311790 | — |
| rasterization **removed** (`pz1`, on `cx1`) | +0.000394 | **+0.098%** |

> **Corrected: 100.0% of the seg gap is description error. Rasterization contributes −0.098% — it is
> a small NET HELP, not a debt.** `ra1`'s "≥91%" was correct in direction and conservative in
> magnitude; the true figure is the whole thing.

### 3.4 — I4: the FITTED-THROUGH law, and what it prices

> **THE FITTED-THROUGH LAW (`ph4`).** The tokens and the LOTTO renderer weights were **searched
> through** the delivery chain `clip(rint(U(·)))` → `D`. The rasterization is therefore not an error
> term sitting between us and a better answer — it is **part of the operator the description was
> fitted through**. Removing it is a distribution shift on a fitted object, and its expected sign is
> **adverse**, not neutral.
>
> **Signature:** improvements and regressions are near-balanced with a small adverse bias
> (`pz1`: 43.3% / 49.3%) and the aggregate lands **wrong-signed but cleanly resolved** — `+0.000394`
> at **788×** the instrument's 5.0e-07 reproducibility floor.
>
> **This is `pz1` §7.3's stale-carrier law, which was derived on the POSE side, holding on the SEG
> side — where nobody had applied it.** The regime is the same one `pz1` §4 diagnosed for pose:
> **FLOOR-RAISED**, not STALE (a matched partial re-fit recovered only 35.7%; the base was already at
> its optimum on 95.0% of pairs).

**What it prices.** `ra1` §4.2 identifies the decoder's three `np.repeat(…, 2)` nearest-neighbour
upsamples as *"a live, unexamined, zero-byte AA surface"* and correctly flags it as a **re-race**
(*"it invalidates the token search"*). The law now supplies the missing magnitude: a delivery-operator
change of scorer-plane rms **0.77 LSB** cost **+0.000394** on a stale description. The `np.repeat`
swap is a far larger operator change, so **its stale-description cost is ≥ that**, and the re-race is
**mandatory, not optional** — exactly as `ra1` said, now with a number behind it. `pz1` §7.2 binds the
rest: any claimed gain must be measured against **GT**, never against the ideal render.

### 3.5 — I7: the barrier is DIRECTED, and the asymmetry pays for lane erasure

**I had this backwards in my own first draft and the receipt corrected me.** I wrote that Road↔Lane
was *"simultaneously the most numerous and among the most expensive"* flip family. `hg1`'s table
(`ddm_hg1_negatives_as_geometry_20260803.md:221-233`, n600, 11 major directed sides) says the
opposite for one of the two directions — and the fact that it is **directed** is the finding:

| directed side | barrier ∫ |
|---|---:|
| **1→0 Lane→Road** (erase a lane) | **5.10** ← the minimum of all 11 |
| **0→1 Road→Lane** (create a lane) | **51.26** |
| 0→2 Road→Undrivable | 33.26 |
| 0→4 Road→MyCar | **58.39** ← the maximum |

> **The same undirected edge is 10.05× asymmetric in repair cost, and `d_seg` prices both
> directions at exactly 1 flip.** `hg1` §:74-77 states the blindness; nobody had multiplied it by
> the *direction* to get the consequence:
>
> **A flip-count objective is PAID to erase lanes, at a 10.05× discount.** Descending `d_seg` (or
> any surrogate that is a flip count) finds lane erasure ~10× cheaper per unit of loss than lane
> creation. That is a **mechanism**, from measured geometry, for the corpus's long-standing
> LANE-ERASURE failure mode ("the measured error = the LANE long-tail = ERASURE (not shift)") —
> which until now has been described as an observed spectral-bias symptom rather than derived.

**Honest limit, from `hg1` against itself.** `hg1` reports the barrier integral as the **weakest** of
its four Spearman predictors of directed flip *rate* (`:266`, ρ = −0.2619) and pre-registers falsifier
**F4** — *"ranking `#766` units by barrier integral beats ranking by flip count at matched bytes"* —
with the honest note that it may show **no separation**. **My claim is about repair COST, not flip
rate**, so F4 does not test it; but the two must not be conflated, and `as1`/`wf2` should carry both.

**Corroboration from a third arm.** `ddm_ax1` (a pre-registration, not a findings memo) independently
predicts *"Lane sub-cell microstructure — 40–50% of f … Lane = 38.7% of endpoint flips and 69.5× over
its exact floor (renderer-REACH-limited)"*. Three arms, three methods, same object: Lane is
geometry-limited, not classifier-limited.

---

## §4 THE COMPOSED INTERACTION LAW — the one constraint on any carrier

The charge asks for these as one object, not a list. They are.

> ## THE CARRIER CONSTRAINT (`ph4`, 2026-08-03)
>
> On a vehicle where **frame_0 is manufactured by warping frame_1's camera raster**, any edit `E` to
> frame_1's camera raster is read by **three consumers at two lattices** (§3.1):
> `Seg(D(f1+E))` · `yuv6(D(f1+E))` · `yuv6(D(W(f1+E)))`.
>
> **1. `D`-null and `yuv6`-null constructions kill the first two and NOT the third.** `W` resamples at
> sub-pixel offsets, so `D∘W ≠ D` and neither null space annihilates it (`pz1` §7.1). **Measured**
> pass-through (asymptotic gain, scorer-plane f0 rms per LSB of camera step): **0.2231** blind set ·
> **0.8902** `D`-visible complement (the set that carries `Q3`) · **0.9809** all pixels.
> *(They are not additive — 0.2231 + 0.8902 = 1.1133 ≠ 0.9809 — because rms does not add across
> spatially correlated response fields. That non-additivity is exactly what broke my first estimate;
> §7 A5.)*
>
> **2. The pose leakage cannot be absorbed by the existing grammar.** The vehicle's 11 pose knobs are
> at their discrete optimum on **95.0%** of pairs; a matched re-fit against a perturbed raster
> recovers only **35.7%** of the penalty — the **FLOOR-RAISED** regime, whose cure is *"none — a
> re-fit chases a worse optimum"* (`pz1` §7.3).
>
> **3. d_seg is luma-led and luma is what the shared `D` hands PoseNet.** Road↔Lane is **50.25%** of
> boundary mass and **76.6% luma-parallel in energy** (`rz1` §2.1), so the strongest seg direction is
> the most pose-expensive one. `pz1` measured that collision at a **79× pose:seg penalty** for an
> unaimed luma edit.
>
> **4. The isoluminant escape is real but only half-strength**: projecting off the luma normal retains
> **55.0%** of the discriminative direction (48.4% on Road↔Lane), cross-checked at **46.0%** by an
> independent n96 gradient split (`rz1` §2.1). Paying **2×** directional loss to avoid a **79×** pose
> penalty is a good trade — **but per (1) it does not actually buy pose-freedom on this vehicle.**
>
> **5. ⇒ THE CONSEQUENCE. A seg actuator on this vehicle needs a pose COMPENSATOR, and the
> compensator must live where the seg actuator cannot be seen — which, per §2, is the `D`-blind set:**
>
> | | seg effect | pose effect | dims/pair |
> |---|---|---|---:|
> | `Q3` isoluminant chroma, `D`-visible | **real** (55.0% efficacy, 2.73e-3 authority at n96) | leaks, **0.8902 pass-through** (MEASURED) | 294,912 |
> | **`D`-blind step** | **EXACTLY ZERO** (measured, 20/20 cells) | **0.2231 gain**, freely aimable | **692,712** |
>
> The two subspaces are **disjoint by construction** (one is `D`-visible, the other is `D`-blind), so
> they compose with **no interaction term on seg** — the blind compensator cannot undo the chroma
> edit's seg gain, because it cannot touch seg at all. **This is the first pair of actuators on this
> vehicle that satisfies the `#383` "pose AFTER frozen seg" staging law exactly rather than
> approximately.**
>
> **6. The exchange rate between them is 3.99×** (0.8902 / 0.2231): cancelling the *full field* of a
> chroma edit at amplitude `A` would need a blind step at ~`4A`. Available — blind edits are seg-free
> at every amplitude and the range is 0–255 — but note this is an **upper bound, not the requirement**:
> `d_pose` is an MSE over **6 scalars**, so only the leak's *projection onto the 6-dim pose response*
> must be cancelled, not the whole field. The true amplitude requirement is `≤ 4A` and is measured by
> O1, not by rms.

**Honest limit on this composition.** It is a *feasibility* argument, not a *sufficiency* one. Whether
692,712 blind dims can be aimed to cancel a specific 6-scalar pose residual depends on the alignment
of PoseNet's Jacobian with the blind subspace, and that requires the scorer. §5 names the falsifier.

---

## §5 OWED MEASUREMENTS — pre-registered, with kill thresholds

Named, not run. **`ph4` fired no scorer pass** (`pu2` holds the slot).

### O1 — the decisive one. Can the blind set be AIMED? (needs ONE scorer holder)

**Probe.** On the live base, for strided pairs: compute `∂d_pose/∂(blind pixels)` by finite difference
through the frozen PoseNet (the `experiments/ddm_pz1_dpose_window_solve_paired.py` chain already
carries every piece except the perturbation), then take one gradient step confined to the blind set at
amplitude ±1, and re-measure `d_pose` at **n600**.

- **Pre-registered kill.** If a single aimed ±1 blind step does not reduce n600 `d_pose` by ≥ the
  instrument's reproducibility floor (`pz1` §3.1: 1.2e-5 rel), the subspace is **misaligned with the
  pose residual** and the actuator is retired — capacity without alignment is not an actuator.
- **Pre-registered kill (seg).** `d_seg` must come back **bit-identical**. If it does not, my §2.1
  proof is wrong and everything here falls. *(This is a free, exact positive control — use it.)*
- **Pre-registered scope.** Report `d_pose` **and** `d_seg`; a pose-only A/B is forbidden (`uv1`
  measured a 3,019× d_pose separation between bases under an otherwise identical solver).

### O2 — the byte question, conditional on O1

A blind-set edit is only aimable if its parameters come from the archive (`inflate.py` cannot run
PoseNet — CLAUDE.md "no scorers at inflate time"). A rank-`k` correction over a **generic,
deterministically-generated** spatial basis (free in `inflate.py` per rule 118) costs `k` coefficients
× 600 pairs.

| `k` | bytes @1 B/coef | **% of the live 0.6189279 gap** (at 9,295 B/%) |
|---:|---:|---:|
| 1 | 600 | 0.065% |
| 6 | 3,600 | 0.387% |

**Break-even:** the pose gap is **17.62%** of the live gap, so a `k=6` correction must close **2.2%
of the pose gap** to pay for itself. That is a low bar — and it is the right one to pre-register.

### O3 — ~~measure, don't infer, the `Q3` warp pass-through~~ **DONE, and it refuted my inference**

I first `INFERRED` **≈0.76** for the `D`-visible complement by endpoint interpolation between 0.2231
and 0.9809, then ran it: `--edit-set visible`, same instrument, same pairs, ~4 min, $0. **Measured
0.8902** — my inference was low by **17%**, because rms does not add across correlated response
fields. §3.2 now carries the measurement. **Artifact:**
`.omx/research/ddm_ph4_visible_set_warp_passthrough_cx1_20260803.json`.

### O4 — routed, not duplicated

- **`wf2`** (owns the PRICE law): the per-edge **effective** exchange rate — `W` × barrier — from
  `hg1`'s barrier field × `pc2`'s per-edge flip shares × `rz1`'s per-edge boundary mass (I6, I7).
  Three measured tables, never joined.
- **`as1`** (owns ASYMMETRY-synergy): the Road↔Lane double-bind (most numerous **and** most expensive).

---

## §5.5 CONSUMPTION LEDGER — the three items the charge said to consume rather than re-run

**`#890` — DID NOT FIND. Scoped negative-existence claim, and it is a known named confound.**
The charge describes *"`#890` is OPEN with 3 censored items — 'PHYSICS vs PHOTOMETRICS is a
scorer-readout asymmetry', on the v4c/v4d photometric stage."* Searched exhaustively:
all 423 rows of `.omx/state/canonical_task_status.jsonl` (800-series ids present: 800–822, 824–828,
850, 871, 873, 882 — **no 890**), its three `.jsonl.corrupt.*` siblings,
`.omx/research/canonical_task_status_fold_quarantine_20260731.jsonl`, all of `.omx/{research,state,
tmp,logs,status,plans,specs}`, plus a repo-wide `rg --hidden` for `#890`, `"task_id":"890"`,
`PHYSICS vs PHOTOMETRICS`, `scorer-readout asymmetry`, and `scorer readout`.
**Zero hits for the described item in `/Users/adpena/Projects/pact`.** The only `#890` in the repo is
a 2026-05-19 master-gradient wire-in audit (`codex_routing_directive_…_20260519T072000Z.md:51`) —
a different task.

> This is `m89` (**TASK-LEDGER SPLIT**) firing live: the harness TaskList and the repo's
> `canonical_task_status.jsonl` are **different stores**, and an arm sees only the repo. A bare id in
> a charge sends the arm hunting nothing. **I did not fabricate a consumption.** If `#890` has
> content, it needs to be quoted into the repo — **cite CONTENT, never a bare id.**

**`pm1` — CONSUMED, and its two FIRED rungs are ALREADY IN THE SHIPPED VEHICLE.** `ddm_pm1` §1
measured *"PoseNet reads auto-exposure and rolling-shutter — the operator's 'Also, photometric' is a
MEASURED, family-level pose carrier"*: rung **B auto-exposure**, ~4 B/pair, **improves 17/17,
degrades 0**; rung **A rolling-shutter**, ~0 B, improves 12/17, degrades 0. **Both are shipped**, and
I verified it at source rather than inferring it — `v4d_cx1_pj2ix2/inflate_runner.py`:

- `:325-326` — `if a != 1.0 or b != 0.0: f0f = a * f0f + b` ⇒ **rung B is the `ab` knob** (2 scalars/pair).
- `:316-322` — `# rung A: rolling-shutter row-shear, sign from the pose yaw dim (5).` ⇒ **rung A is `beta_idx`.**

> **This closes the loop and it strengthens §2.3's argument rather than competing with it.** The
> photometric physics `pm1` mined is not un-spent headroom — it is **2 of the 11 pose knobs that
> `pj2` measured at their discrete optimum on 95.0% of pairs.** The vehicle has already absorbed the
> photometric family. The `D`-blind subspace is **orthogonal to all 11**, which is the entire
> argument for it.

**`ax1` — consumed as a PRE-REGISTRATION, not as findings.** `ddm_ax1` §0 is an explicit
pre-registration of `f ∈ [7e-4, 2.0e-3]`, not a results memo; it has no "THE ANSWER" section.
Its mechanism #1 — *"Lane sub-cell microstructure, 40–50% of f; Lane = 38.7% of endpoint flips and
**69.5× over its exact floor (renderer-REACH-limited)**"* — corroborates `rz1` §2.2 and §3.5 above
from a third independent direction. **Do not cite `ax1` as a measured result until its fork resolves.**

---

## §6 HONEST NON-REACTIVATIONS — closures are results

**Do not resurrect these. Each is closed with a reason, not with a shrug.**

| closed | why | scope |
|---|---|---|
| **Pre-compensation / inverse-filtering the resize** (the `ll1` window solve) | **RETIRED** on `ra1`'s own pre-registered falsifier. n600, both axes: ΔS(seg) **+0.000394**, ΔS(pose) **+0.0310217**, net **+0.0314155** = 4.80% of `cx1`'s gap (**5.08%** of the live gap) in the wrong direction. `pz1` §6. | the lever as built, on this vehicle |
| **The pose RE-FIT conversion** of the above (~11.2 CPU-h) | **NOT TRIGGERED.** There is no seg win to defend, and a matched re-fit recovers only 35.7% — **FLOOR-RAISED**, whose cure is *none*. `pz1` §6/§7.3. | the conversion, not re-fitting in general |
| **Null-space RE-ALLOCATION** (dither order, allocation norm, init kernel) | **measured shut four independent ways**, <5% movement each; §7.1 explains *why* — the problem is the lattice, not the allocation. `ra1` §3. | FORMULATION |
| **Dither / anti-alias the raster** | attacks the **6.0%** quantization term of a debt that is **93.5% float resampling**. `ra1` §0(2). | the family |
| **`rz1` §3.6 "blind set ⇒ zero bits, do not spend"** | **REFUTED, §2** — but note this is the *ranking* that is refuted, not `rz1`'s R1 partition geometry, which my measurement is consistent with and which stands. | the rank, not the geometry |
| **`rz1` A2 "exactly pose-free by construction"** | **REFUTED as stated, §3.2.** A2 itself is **re-scoped, not killed** — a real seg actuator that must pay a real pose cost. | the exactness claim |

**And one thing I explicitly did NOT close.** I did not test whether the blind-set actuator *works*,
only that it *exists and is seg-free*. Existence is not efficacy. O1 is the falsifier; if it fires,
this arm produced a clean negative and that is the correct outcome.

---

## §7 Round-1 adversarial self-review

**A1 — I refuted my own headline number with my own instrument.** The 6-pair run gave a blind-set
"leverage" of **0.3213** against a population share of 0.226969, and I had started writing up a
**1.416× over-weighting of the blind set by the warp** — a tidy, interesting, *wrong* claim. The
amplitude sweep killed it: the gain decays monotonically 0.2686 → 0.2364 → 0.2264 → 0.2238 → 0.2231
and converges on the population share. **The excess was `_to_uint8`'s round acting as a threshold at
small signal, not geometry.** The corroborating detail is that `sqrt(0.221) = 0.4701` reproduces the
amp-1 camera rms of 0.4700 exactly — the changed pixels each moved by exactly ±1 LSB and nothing else
moved at all, which is the signature of a threshold, not of a linear response. Recorded because a
refuted hypothesis that changes a number is the finding.

**A2 — can my probe return the negative?** Yes, and it is the whole design. **C2**: a null delta
reports exactly `0.0` through the *same* f0 chain. **C3**: a cardinality-matched `D`-**visible** edit
moves `D(f1)` by 1.000 on every pair — so the `0.0e+00` in §2.1 is a property of *which* pixels moved,
not of a dead code path. **C4**: an all-pixel step moves `D(f0)` by 0.9809 ≈ 1. Without C3 in
particular, "the warp call silently no-op'd" and "the blind set is genuinely invisible to `D`" emit
the identical symbol (`m50`).

**A3 — is my base the live base?** No, and I am flagging it rather than smoothing it. I measured on
`v4d_cx1_pj2ix2` (S 0.8264972); the live best is `pu2` (S 0.7910689), which differs by pose knobs on
6 tail pairs. **What transfers:** §2.1's seg-freedom is a property of `D`'s geometry and `SegNet`'s
single path — base-independent, and provable from `modules.py` alone. §2.2's *asymptotic* gain is a
property of area-preserving interpolation — also base-independent, and the measured 1.7% agreement
with the population share is the evidence for that. **What does not transfer:** the per-pair
`%changed` figures, which `pz1` §A5 already showed are the base-sensitive statistic (58.8% on
`dc1_fold` vs 34.9% on `cx1`, same chain).

**A4 — is 4 pairs a population?** For **this quantity**, the `m88` guard is satisfied by mechanism
rather than by sampling: the seg result is `0.0` exactly in 20/20 cells (a proof, not a mean), and the
gain is a property of the resampling geometry — its spread across pairs 200 apart is
0.2231±small, the same "mechanically flat" character `ra1` §2 established for the plane debt. **It
says nothing about how much `d_pose` moves**, which is scene-dependent and is exactly what O1 measures
at n600. I have not claimed otherwise.

**A5 — I refuted a second number of my own, and this one was load-bearing.** §3.2's `Q3` pass-through
started as **≈0.76**, `INFERRED` by endpoint interpolation between 0.2231 (blind) and 0.9809
(all-pixel). It was the magnitude §3.2's "orders of magnitude, not degree" rested on, so I measured it
instead of shipping it: **0.8902** — my inference was low by **17%**. The reason is worth keeping:
**rms does not add across spatially correlated response fields**, so blind + visible (1.1133) exceeds
all-pixel (0.9809) and no endpoint interpolation between them is valid. The correction *strengthens*
§3.2 (the leak is larger than I guessed) and *tightens* §4.6 (the compensator exchange rate is 4.0×,
not 3.4×). **Two hypotheses of mine refuted by my own instrument in one arm** (A1, A5) — which is the
argument for building the amplitude sweep and the `--edit-set` flag rather than reasoning about them.

**A5b — a THIRD number of mine, refuted by a receipt rather than by my instrument.** I drafted I7 as
*"Road↔Lane is simultaneously the most numerous and among the most expensive flip families."*
`hg1`'s directed table says **Lane→Road is the CHEAPEST of all 11 sides (5.10)** while Road→Lane is
51.26. I had the sign backwards, and correcting it produced a *better* finding than the one I set out
to write (§3.5): the asymmetry is a **mechanism** for lane erasure, not a double-bind. **I did not
open `hg1` before drafting I7 — I reasoned from a summary line.** That is the failure mode the
corpus-first rule (`m44`) names, caught here only because I went and fetched the receipt.

**A6 — what I did NOT establish.** (i) That the blind actuator can be **aimed** (O1). (ii)
`dim(null(D) ∩ null(D∘W))` — still uncomputed, `pz1` §A6(i) named it first and it is still open; my
result partly *dissolves* the question (the blind set gives an exactly-seg-free subspace without
needing that intersection) but does not answer it. (iii) Whether a blind-set edit survives the
**inflate→disk→evaluate** round trip bit-identically — the receiver writes uint8 frames, so it should,
but I drove the `Decoder` in-process and did not write frames to disk. **O1 must do the disk round
trip, not the in-process one.**

---

## NEXT-IF-RESUMED

1. **O1 is the single highest-value next action and it needs one scorer holder, not a new arm.** Every
   piece except the perturbation already exists in
   `experiments/ddm_pz1_dpose_window_solve_paired.py`; the free exact positive control (`d_seg` must
   return **bit-identical**) makes it self-validating.
2. **O3 first if no scorer is free** — 4 minutes, $0, one-line change, and it retires the only
   `INFERRED` number this memo leans on.
3. **`rz1`'s attack table needs re-ranking with assumption #13 resolved** (frame_0 IS a warp, `ra1`,
   re-verified §1). A2's "exactly pose-free" premise is gone; A3's blocking unknown is answered in the
   bad branch. Whoever owns that table should re-run its ranking, not re-run its measurements — they
   are sound.
4. **Do not re-open** anything in §6. Each closure names its scope.
5. **`#890` needs its content quoted into the repo, or it cannot be consumed** (§5.5). An arm sees the
   repo ledger only; a bare id routed from the harness TaskList resolves to nothing. `m89`.

---

## Reproduce

```bash
SUB=/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_cx1_pj2ix2

# the seg-freedom proof + the pose-reach dynamics law   (~10 min, $0, no scorer)
.venv/bin/python experiments/ddm_ph4_blind_set_pose_reach.py \
    --submission-dir $SUB --pairs 4 --amps 1 2 4 8 16 --edit-set blind \
    --out .omx/research/ddm_ph4_blind_set_pose_reach_cx1_20260803.json

# the Q3-carrying complement's warp pass-through (O3)   (~5 min, $0, no scorer)
.venv/bin/python experiments/ddm_ph4_blind_set_pose_reach.py \
    --submission-dir $SUB --pairs 4 --amps 1 8 16 --edit-set visible \
    --out .omx/research/ddm_ph4_visible_set_warp_passthrough_cx1_20260803.json
```

All four controls (`C1`–`C4`) print before any result. `SEG EXACTLY FREE` must read `True` for
`--edit-set blind` and `False` for `--edit-set visible`; if either flips, the instrument is wrong,
not the vehicle.
