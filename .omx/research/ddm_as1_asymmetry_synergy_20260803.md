# ddm_as1 — Leveraging the asymmetry findings: reconciliation, synergy, and actuators

**Arm:** `ddm_as1` · **Date:** 2026-08-03 · **Cost:** $0, **scorer-free** (0 scorer forwards;
`ddm_pu2` holds the slot). Every measurement here is a reduction over argmax planes that
`ddm_pu2` had **already cached** — a read, never a re-run.
**Axis:** `[macOS-numpy advisory · NON-PROMOTABLE]` · `score_claim=false` ·
`promotion_eligible=false` · `rank_or_kill_eligible=false`. Pointer UNMOVED.

**Operator directive 2026-08-03, verbatim:** *"We want to fully leverage our asymmetry findings and
look for synergy and other dynamics that we can optimize against."*

**BASELINE for every ΔS below:** live best `S = 0.7910689`, **353,805 B**, seg leg **0.4311790**,
gap to target **0.6189279**, cx1 flips **508,639**. **TARGET** = PR130 floor `S = 0.172141`.

**Exchange rate, RE-DERIVED not re-typed:** `S/flip = 100/(600·384·512) = 8.477105e-7` ·
`S/byte = 25/37,545,489 = 6.658590e-7` · **`W = 1.2731082153 B/flip`** ✓ (matches the live
invariant to 10 digits). Cross-check: `508,639 × 8.477105e-7 = 0.4311794` vs the stated seg leg
`0.4311790` — agree to `4e-7`. **`W` restated as a bit budget: 8·W = 10.185 bits per flip.**

**POSITIVE CONTROL for everything in §1–§4:** the n600 confusion reduction reproduces
`d_seg = 0.004311795` against cx1's evaluator row `0.00431179` — **absdiff `4.70e-09`**. The
argmax planes are the right ones; every count built on them is admissible.

---

## §0 THE ANSWER

**One law, measured six times on six unrelated instruments, and it is a NEGATIVE that is worth
more than the ranked list it replaces:**

> **Every asymmetry we hold is real and large, and every attempt to CASH it through an aggregate
> — a pooled statistic, a class-level scalar, a blended predictor, a proxy screen — has failed.
> The asymmetry's value is not as an ACTUATOR. It is as a PRIOR on an actuator that still has to
> address elements individually. Selectivity, not magnitude, is the binding quantity.**

Receipts, each an independent instrument: `hg1` pooled-ρ Simpson reversal · `sx2` free-extractor
negative marginal value · `pu2` the `|pose − t_p|` screen at `r² = 0.000` · `#900` neither entry
rule dominating · `mf1` `τ_end = m_q/ln5` pooling suspect · **and this arm's new one, §3:** the
class-level area actuator that the decomposition seems to license is **MEASURED DEAD**
(`grow_Lane_into_Road_1px` = **+0.2459 S**, and every other unconditional area move is worse too).

**What that law BUYS, quantified (§4).** The asymmetry pays as a coding prior on a carrier that
already addresses elements: **84,520 direction bits = 10,565 B = 0.007035 S = 1.14% of the gap**,
measured, and it composes with *any* such carrier because it is a prior on an existing field, not
a new field.

**Four new measured facts** (n600, cx1's own argmax, positive control passed):
1. The seg residual splits **exactly and exhaustively** into **33.92% net-area imbalance /
   10.82% area-neutral circulation / 55.27% symmetric positional jitter**.
2. `hg1`'s registered falsifier **F1 is CONFIRMED**: `Lane→Road` share **0.7851** (predicted
   ≥ 0.60, falsifier band [0.45, 0.55]). The witness-vehicle Lane asymmetry **transfers and
   strengthens**.
3. `hg1`'s claim 4 — *"the asymmetry does NOT generalize past Lane; `Road↔Movable` 1.03×,
   `Road↔Undrivable` 1.05× are symmetric"* — is **REFUTED on the live vehicle**. On cx1 those
   sides are **1.606×** and **1.681×**. **Every major edge is asymmetric.** The near-symmetry was
   a witness-vehicle property, not a separatrix property.
4. `hg1`'s ≈15.2%-of-gap hypothesis is **closed exactly** (§2.3), and the closure exposes a
   **quantity conflation** that would otherwise have propagated.

**One sign correction (§5.3):** "`Road↔Lane` costs 299,369 B = 84.6% of the archive ⇒ not buyable
as corrections" reads a **CEILING ON SPEND** as a cost. It says a correction carrier **may spend
up to 10.185 bits per flip**. `hs1`'s static top-128 index (62 B for 91.22%) sits three orders of
magnitude under that ceiling. The edge is not unbuyable; it is unbuyable *at 1 B/flip of naive
addressing*, which nothing proposes.

---

## §1 THE INSTRUMENT (and why it is not a double-count)

`ddm_pu2` cached, and left behind, at
`/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/`:
`gt_argmax_n600.npy` + `cx1_argmax_n600.npy` (600 × 384 × 512 uint8) + `per_pair_directed.jsonl`
(600 rows, each a 5×5 directed confusion). The run reached **600/600** while this arm was working;
an earlier read at 464 pairs is superseded and is *not* used for any conclusion below.

**Prefix discipline (m88), applied and documented.** The 464-pair prefix had
`mean d_seg / population = 0.9407` — a *different population* by the strict test, so the prefix
conclusions were discarded rather than reported. All §2–§4 numbers are the **full n600
population**. Where a sub-sample was unavoidable (§3, §4 morphology at 100 frames) the frames are
**SPREAD** via `linspace(0, 599)`, never a prefix, and the subset/population governing-quantity
ratio is reported inline (**0.9807**).

**Round-1 lineage check — the failure I was warned about.** The likeliest way to fake synergy is
to find "two facts" that are one measurement seen through two instruments sharing a cache. Three
lineages are in play and they are **disjoint**:

| lineage | source array | vehicle | total flips |
|---|---|---|---:|
| `ddm_hg1` §1–3 | `gt_n96.npz` witness cache | witness | 75,987 (n96) |
| `ddm_pc2` | `ddm_ru1_20260729/atlas_flat.npz` | tb1 ep399 | 458,738 |
| **this arm** | `ddm_pu2_20260803/argmax_cache` | **cx1 (live best)** | **508,640** |

Different arrays, different vehicles, different totals. Where they agree (below) the agreement is
**evidence**; where they disagree it is a **vehicle** difference, and I say which.

**Cross-lineage agreement (unforced, therefore load-bearing):**

| quantity | `pc2` (ru1/tb1) | **this arm (cx1)** | agreement |
|---|---:|---:|---|
| `Road↔Lane` share of all flips | 49.23% | **46.23%** | 1.06× |
| Road node participation | 87.85% | **89.62%** | 1.02× |
| Road net area bias | **+118,775 px** | **+138,461 px** | 1.17×, **same sign** |
| `Road↔Lane` asymmetry | 3.60× | **3.653×** | **1.01×** |

---

## §2 (a) THE RECONCILIATION

### 2.1 Rows 1 and 2 are THE SAME FACT, and the synthesis is sharper than either

They *look* like they conflict — `hg1` says the Lane asymmetry is **EXTENT not DENSITY**, `mf1`
says the **margin-depth leg is REFUTED** (transverse profiles symmetric to ~2%). Both are right,
about different objects:

- `mf1` measured the **potential** across the separatrix: the margin rises at essentially the same
  rate on both sides — **symmetric to ~2%**.
- `mf1` also measured the **domain**: *"Road side reach = 7 px, Lane side reach = 1 px"* at ≥20%
  support, with comparable seed counts (629,474 vs 514,023).
- `hg1` measured the same domain from the other side: Lane shell **75.04%**, mean depth
  **1.134 px**, the **only truncating** side, barrier **5.10 vs 33.26–58.39**.
- `rz1` measured it a third way: **Lane has NO INTERIOR** — 6.92% ≥ 2 px deep vs Road's 63.64%.

> **SYNTHESIS (three instruments, one fact): the Lane asymmetry is ENTIRELY GEOMETRIC — a
> property of the DOMAIN — and NOT AT ALL energetic. The potential is symmetric; the domain it
> lives on is 7:1.**

That synthesis is not cosmetic; it has a **binding design consequence** and it is the *positive
dual* of `hg1`'s negative:

> **Any per-side weight must be a function of the AVAILABLE DEPTH (a domain quantity), never of
> the margin field (an energy quantity) — because the margin field is symmetric and therefore
> carries no side information at all.** `hg1`'s "do not density-weight" is the *negative* form of
> this. The positive form is: **depth-weight, don't margin-weight.** `mf1`'s independent suspicion
> of `τ_end = m_q/ln5` follows immediately — it pools an energy quantity over a domain that is 7:1
> asymmetric, so it is a mixture in the *domain*, which is exactly why the sign-based defence of
> it ("the profile is symmetric") does not rescue it.

### 2.2 Rows 3 and 6 are NOT the same phenomenon — and testing that mattered

The tempting composition was "`sx2`'s negative-marginal-value blend and `hg1`'s pooling reversal
are one wrongly-aggregated superposition." **They are not**, and the distinction is mechanical:

- `hg1` row 3 is a **Simpson reversal**: pooled-all `ρ = −0.150` **exceeds every within-class
  value** (max `|ρ| = 0.110` at Lane). An aggregate outside the range of its parts can only arise
  from sub-population mixing.
- `sx2` row 6 is **not** outside the range of its parts: static prior 57.2%, canny 28.1%,
  union **49.1%**, weighted blends 20.7 / 25.2 / 47.4% — every blend lies *between* the parts or
  below the better one. That is the ordinary "the weak predictor's true positives are a near-subset
  of the strong one's while its false positives are additive" signature. No Simpson needed.

**They share a GENUS, not a mechanism**, and this arm's §3 supplies a **third independent
instance** of that genus, so the genus is now measured on three unrelated instruments:

> **GENUS: a quantity that is correct in aggregate fails when acted on, because the underlying
> effect is SELECTIVE and the aggregate discards the selection.**
> `hg1`: pooled ρ mixes sub-populations of opposite alignment. `sx2`: a union blends a coarse
> predictor into a precise one. **`as1` §3: the class-level area deficit is real (+134,786 Lane
> px) but the pixel-level precision of acting on it is 0.208, so the aggregate loses 3:1.**

### 2.3 `hg1`'s 15.2%-of-gap: closed exactly, and it exposes a quantity conflation

`hg1` registered A13 as a **HYPOTHESIS** (*"mixes `ddm_pc2` (cx1) with the witness split. Closed
only by F1."*). F1 is now run. The closure, with every factor named:

```
hg1's chain :  pc2 Road<->Lane HEADROOM 0.16041 S  x  0.6866 (witness dir split)
            =  0.11014 S  =  15.17% of the then-gap 0.7262358      <- reproduces "15.2%"
correction 1 : directional split, witness -> cx1   0.7851/0.6866 = 1.1435x
correction 2 : gap denominator moved 0.7262358 -> 0.6189279       = 1.1734x
=> corrected, ON A HEADROOM BASIS :                 20.35% of the CURRENT gap
```

**hg1's arithmetic was correct given its inputs; it understated by 1.34× for two reasons, both
identified.** But the more important finding is the **conflation risk** it exposes:

> **The `Road↔Lane` edge has TWO different "% of gap" numbers and they are not comparable.**
> **Full live S** of the edge = `235,148 × 8.477105e-7 = 0.199337 S` = **32.21%** of the gap.
> **Headroom above the oracle-R floor** (`pc2`'s quantity) = `0.16041 S` = **25.92%** of the gap.
> `Lane→Road` alone, full live S = `0.156498 S` = **25.29%** of the gap, **36.30% of all flips**.
> Quoting 32.21% against `hg1`'s 15.2% compares a *live-S* number to a *headroom* number and
> makes the understatement look 2.1× worse than it is. This is `m90` live — **the floor you
> divide by decides the answer** — and `m66` — **a ΔS without its baseline is unanchored.**

### 2.4 The genuine conflict, and which side is wrong

`hg1` claim 4 vs the live vehicle. Apples-to-apples (I recomputed `hg1`'s rates as **count**
ratios so the two instruments measure the same thing; `hg1`'s headline 5.85× is a *rate* ratio,
its count ratio is 2.191×):

| edge | `hg1` witness n96 (count ratio) | `pc2` ru1/tb1 | **cx1 n600 (this arm)** |
|---|---:|---:|---:|
| `Road↔Lane` | 2.191× | 3.60× | **3.653×** |
| `Road↔MyCar` | 1.548× | 3.71× | **3.020×** |
| `Undriv↔Movable` | 1.296× | 2.28× | **2.344×** |
| `Road↔Undriv` | **1.038×** | 1.31× | **1.681×** |
| `Road↔Movable` | **1.106×** | 1.47× | **1.606×** |

**Verdict: `hg1`'s "asymmetry is localized to Lane" is a WITNESS-VEHICLE fact and does not
transfer.** Two independent live-vehicle lineages agree that every major edge is asymmetric
(≥ 1.31× on `pc2`, ≥ 1.606× on cx1). **`hg1`'s design consequence — "only Lane-touching negatives
earn per-side re-review" — is refuted: all five major edges earn it.** `hg1`'s *other* design
consequence (do not density-weight) is untouched and stands.

---

## §3 (c) THE ACTUATOR TEST — and the negative that reframes the whole search

### 3.1 The exact, exhaustive decomposition

For `C[gt][rendered]`, split the off-diagonal into a symmetric and an antisymmetric part. On the
**complete** class graph the minimum L1 flow realising a divergence vector `d` is exactly
`½·Σ|d_i|`, so the split is L1-consistent and exhaustive:

| component | flips | % | S | break-even budget |
|---|---:|---:|---:|---:|
| **net area imbalance** (divergence) | **172,519** | **33.92%** | 0.146246 | 219,635 B |
| **circulation** (area-neutral cycles) | 55,015 | 10.82% | 0.046637 | 70,040 B |
| **symmetric jitter** (position) | **281,106** | **55.27%** | 0.238297 | 357,878 B |
| TOTAL | 508,640 | 100.00% | 0.431179 | — |

Per-class net deficit (px, `+` = under-painted): **Lane +134,786** · Movable +37,733 ·
Undriv −2,127 · MyCar −31,931 · **Road −138,461**. Conservation `Σ = 0` ✓.

> **ROUND-1 SELF-REVIEW, a real error caught and fixed.** My first pass computed a Hodge
> decomposition and summed the **L1** masses of the gradient and circulation parts, getting
> `246,463 + 274,968 = 521,431 > 227,534 = |A|`. Hodge is orthogonal in **L2**, not L1, so L1
> masses do not add. The table above uses the min-cost-flow identity instead, which *is*
> L1-exact. The wrong version would have inflated the "per-class scalar can fix it" headline by
> 1.43×. Recorded rather than quietly corrected.

### 3.2 F-ACT-1: the actuator the decomposition seems to license is DEAD

33.92% of the residual being "the wrong AMOUNT of each class" appears to license the cheapest
conceivable actuator: a per-class prior/bias/area move, ~5 numbers, essentially free. **Measured
on cx1's own argmax, 100 SPREAD frames (subset/population ratio 0.9807):**

| variant | Δflips | **ΔS at n600** |
|---|---:|---:|
| `grow_Lane_into_Road_1px` | +48,336 | **+0.245850** |
| `grow_Lane_into_Road_2px` | +131,294 | **+0.667796** |
| `shrink_Road_where_MyCar_1px` | +45,453 | **+0.231186** |
| `grow_Movable_1px` | +2,735 | **+0.013911** |

**F-ACT-1 FIRES on every variant.** The class-level deficit is real and its sign is right, but
acting on it without position is 3:1 destructive. The 172,519-flip area imbalance is **not**
reachable by any position-free actuator.

### 3.3 WHY it fails, and the exact specification it hands the next builder

Per directed side, `N` = candidate pixels (rendered-`Y` adjacent to rendered-`X`), `K` = those
whose GT is `X`. Net = `N − 2K`; **break-even precision `p* = 0.500` on every side.**

| grow X into Y | cands N | true K | **prec p** | naive ΔS(n600) | oracle ΔS(n600) | max gate bytes |
|---|---:|---:|---:|---:|---:|---:|
| Lane into Road | 82,797 | 17,198 | **0.208** | +0.246180 | −0.087474 | 131,369 |
| Undriv into Road | 42,549 | 9,228 | **0.217** | +0.122543 | −0.046936 | 70,489 |
| **Movable into Road** | 11,740 | 4,377 | **0.373** | +0.015188 | −0.022263 | 33,434 |
| **Movable into Undriv** | 11,192 | 4,019 | **0.359** | +0.016042 | −0.020442 | 30,700 |
| Road into Movable | 11,454 | 2,795 | 0.244 | +0.029826 | −0.014216 | 21,350 |
| Road into Lane | 66,588 | 8,055 | 0.121 | +0.256744 | −0.040970 | 61,529 |
| Road into MyCar | 50,409 | 7,492 | 0.149 | +0.180181 | −0.038106 | 57,229 |
| Road into Undriv | 42,559 | 5,055 | 0.119 | +0.165044 | −0.025711 | 38,613 |
| MyCar into Road | 50,399 | 2,436 | 0.048 | +0.231562 | −0.012390 | 18,608 |

**No side reaches `p* = 0.5`**, which is why every unconditional move loses. **But the asymmetry
is exactly the structure in this table**, and it is 5-for-5 consistent with the measured
divergence:

| edge | favoured direction | precision ratio | matches sign of net deficit? |
|---|---|---:|---|
| `Road↔Lane` | grow **Lane** | 1.72× | ✓ Lane +134,786 |
| `Road↔Undriv` | grow **Undriv** | 1.82× | ✓ (Undriv under on this edge) |
| `Road↔Movable` | grow **Movable** | 1.53× | ✓ Movable +37,733 |
| `Undriv↔Movable` | grow **Movable** | 1.76× | ✓ Movable +37,733 |
| `Road↔MyCar` | grow **Road** | 3.10× | ✓ MyCar −31,931 |

> **SPECIFICATION HANDED FORWARD.** A boundary-correction gate needs **precision > 0.500**. It
> starts at **0.208–0.373** on the four best sides. The asymmetry does not close that gap — it
> tells the gate which way to default, worth **1.53×–3.10×** on the prior. **`Movable into Road`
> (p = 0.373) and `Movable into Undriv` (p = 0.359) are the closest to break-even and the
> cheapest to reach** — and they are *not* the biggest edges, which is why a mass-ranked list
> would have missed them.

---

## §4 (b) THE SYNERGY SEARCH — what composes, what cancels, priced in `W`

### S1 — asymmetry × any direction-coding carrier: **COMPOSES. +10,565 B, measured.**

The directional prior's value is exactly the mutual information between "a flip is here" and
"which way it goes", and that is a hard saving on any carrier that must code direction:

| edge | flips | p_major | H (bits) | uniform bits | with prior |
|---|---:|---:|---:|---:|---:|
| `Road↔Lane` | 235,148 | 0.7851 | 0.7508 | 235,148 | 176,539 |
| `Road↔Undriv` | 89,545 | 0.6270 | 0.9530 | 89,545 | 85,332 |
| `Road↔MyCar` | 63,027 | 0.7513 | 0.8093 | 63,027 | 51,006 |
| `Undriv↔Movable` | 61,892 | 0.7010 | 0.8801 | 61,892 | 54,469 |
| `Road↔Movable` | 57,225 | 0.6163 | 0.9606 | 57,225 | 54,970 |
| **TOTAL** | **506,837** | — | **0.8332** | **506,837** | **422,317** |

**84,520 bits = 10,565 B = 0.007035 S = 1.137% of the gap**, at **0.0208 B/flip** against a
budget of `W = 1.2731` B/flip. Modest, but it **composes with everything**: it is a prior on an
existing field, not a new field, so it does not compete for bytes with any other carrier.

**Open question I could not close, flagged not asserted:** whether the 5 per-edge priors are
**free** (a fixed property of *our encoder's* systematic bias → generic → lives in `inflate.py`)
or **counted** (~10 B). Per the three-way test, an operator-property is free and a this-clip
property is counted; a systematic encoder bias is arguably the former. **This is `wf2`'s call
(price law), not mine.**

### S2 — displacement carrier × area imbalance: **COMPLEMENTARY, and it caps `sx2`/`mf1`**

A pure translation is area-preserving, so it lives entirely in the **symmetric** part. The
divergence part is therefore **structurally invisible** to any displacement carrier:

| class | edge-flip involvement | net area deficit | **displacement-BLIND fraction** |
|---|---:|---:|---:|
| Movable | 119,933 | +37,733 | **31.46%** |
| Road | 444,945 | −138,461 | 31.12% |
| **Lane** | 236,816 | +134,786 | **56.92%** |

> **This caps `sx2`/`mf1`'s Movable displacement carrier.** Its "at full effectiveness
> −0.0992 S" is bounded above by the area-preserving fraction: **≤ 68.5% of Movable's flip mass
> is reachable by translation at all.** The carrier is still strongly profitable (1,854 B against
> a 1,456-flip break-even) — this is a **correction to the ceiling, not a kill.** Composition
> with an area actuator would be additive **if** an area actuator existed; §3 says none does.

### S3 — pose-free chroma (row 4) × the steepest-recovery side (row 1): **MILD ANTAGONISM, 1.25×**

`rz1` measured the exactly-pose-free chroma subspace retaining **55.0%** globally, **48.4%** on
`Road↔Lane`. With `Road↔Lane` now measured at **46.23%** of all flips on cx1, the implied
retention on every other edge is **60.67%**:

> **Chroma is weakest exactly where the mass is — but the penalty is only `0.6067/0.484 =
> 1.254×`.** A naive reading ("the dominant edge is the worst for chroma, so chroma is dead") is
> **wrong by construction**: against `pz1`'s measured **79× pose:seg penalty**, a 1.25×
> edge-concentration penalty on top of `rz1`'s 2× directional loss is still a large win.
> **This CONFIRMS `rz1`'s reversal and prices it per-edge for the first time.**

### S4 — pose two-regime (row 5) × `#900` non-dominant entry rules (row 8): **THE SAME FACT**

`pu2`: pair 74 yields **54.2%** to direct search, pair 523 **resists (−7.7%)**, and the frame_1
proxy called them **backwards**; `|pose − t_p|` as a screen is **REFUTED at `r² = 0.000`**.
`#900`: ASYMMETRY_PRICED but **−3.745e-05 composed (0.0048% of gap)** and **neither entry rule
dominates**. These are not two findings:

> **No scalar screen ranks the pose tail. Both are negative-existence results about scalar
> selectors on the same population, from different directions.** The operational consequence is
> single and sharp: **the pose tail is selected by DIRECT PER-PAIR TRIAL, never by a proxy** —
> which is exactly the procedure that produced `pu2`'s measured **−0.0354261 S on 6 pairs**.
> And it is the **same law as §0**: selectivity beats magnitude, on the pose axis too.

### S5 — the four-way Lane law (the coordinator's compose)

Four arms, four instruments, one mechanism:

| receipt | what it says about Lane |
|---|---|
| `hg1` | EXTENT not density: shell 75.04%, depth 1.134 px, only truncating side, barrier 5.10 vs 33.26–58.39 — **yet steepest margin recovery of all 14 sides** |
| `rz1` | **no interior** — 6.92% ≥ 2 px deep vs Road 63.64% |
| `pu2` / this arm | **erasure, not displacement** — `Lane→Road` 0.7851 of the edge; net **−134,786 px**; **56.92% displacement-blind** |
| `hs1` | seg × PAIR is dead (Gini 0.0858); seg concentrates in **SPACE**, and space is **STATIC** — top-128 at 62 B captures 91.22% |

> **THE LAW.** *Lane is a 1-px ribbon with no interior, so it has no region to paint and no
> depth to displace; the vehicle therefore does not misplace it, it **deletes** it — 134,786 px
> net — and because deletion is area-changing it is invisible to a displacement carrier and
> because the ribbon has no interior it is invisible to a region-paint primitive. It survives
> only where it is **addressed**, and it is addressable cheaply because it is **spatially static**.*
>
> **NEGATIVE DESIGN CONSTRAINT (binding, the dual of `hg1`'s):** **do not build a
> region-paint or a displacement carrier for Lane.** Both are structurally blind to > half its
> flip mass — 56.92% measured. Lane needs a **presence/existence** carrier on a **static spatial
> index**, which is a different primitive from anything currently ranked. `dd1` owns whether the
> Lane component is decomposable; this is the constraint its answer has to satisfy.

### What CANCELS (equally a result)

- **Area actuator × everything: cancels.** There is nothing to compose it with — §3 kills it
  standalone, so it cannot contribute to any stack. Removed from the ranked list.
- **`sx2`'s free generic extractor × the counted static prior: cancels, measured** (union 49.1%
  < 57.2% alone). `sx2`'s own conclusion, independently re-derived as an instance of the §2.2
  genus, and it **generalises**: do not blend a coarse free predictor into a precise counted one.
- **Displacement × Lane: cancels** (56.92% blind, S2/S5).

---

## §5 (c) THE DYNAMICS — actuators that exist, and what plainly does not

### 5.1 Named actuators

| asymmetry | actuator | status |
|---|---|---|
| S1 directional prior | needs a **direction-coding field** in the coder's entropy model | **pending** — an `Explore` sweep of `src/tac/`, `experiments/`, `tools/` and the submission dirs was dispatched for exactly this; it had not returned when this memo was written. **I do not assert either way.** |
| S2 displacement | per-component 2-vector, ~1.25 B/component, `sx2`/`mf1` | **cited, owned by `sx2`/`mf1`** — this arm contributes only the ≤68.5% ceiling |
| S3 chroma | `Q3` frame_1-yuv6-null (`d_pose` EXACTLY 0) | exists; `ph4` owns physics/photometrics |
| S4 pose tail | direct per-pair re-solve | exists and has a **measured row** (`pu2`, −0.0354261 S) |
| S5 Lane presence | **NONE. Plainly: no shipped primitive addresses an area-changing defect on a no-interior class.** | this is the gap the law names |
| §3 area imbalance | **NONE, and none should be built** — F-ACT-1 | measured dead |

### 5.2 What I did NOT measure (stated, not hidden)

- **The gate that would reach p > 0.5.** I measured what precision is *required* (0.500) and
  where it *starts* (0.208–0.373). I did **not** build or price a gate that closes it.
- **Whether the S1 prior is free or counted.** `wf2` owns the price law.
- **Anything requiring a scorer forward.** 0 scorer forwards; `pu2` held the slot throughout.
- **Lane component decomposability** — `dd1` owns it; §5.5's law is a constraint on its answer,
  not an answer.
- The `Explore` sweep for direction-coding fields in shipped code (§5.1) — dispatched, unreturned.

### 5.3 A sign correction on a load-bearing framing

> *"fixing the whole edge as a correction stream costs `235,148 × W` = 299,369 B = 84.6% of the
> entire archive ⇒ `Road↔Lane` is NOT BUYABLE as corrections"*

`235,148 × 1.2731082153 = 299,369 B` is arithmetically right and it is **84.6% of 353,805 B** ✓.
But `N × W` is the **break-even BUDGET** — the *maximum* a carrier may spend and still profit —
not a cost. Restated in the natural unit: **a correction carrier may spend up to
`8 × W = 10.185 bits per flip`.** `hs1`'s static top-128 index captures **91.22% at 62 B**, three
orders of magnitude under that ceiling. **The edge is not unbuyable; it is unbuyable only at
naive per-flip addressing, which nothing on the table proposes.** The conclusion *"it must come
from the base representation"* may still be right for the **erasure** reason in §S5 — but it does
not follow from the byte arithmetic, and the two justifications must not be merged.

---

## §6 CLAIM LEDGER (verdict scope per claim)

| # | claim | status | scope |
|---|---|---|---|
| A1 | n600 confusion reproduces `d_seg` to 4.70e-09 | **MEASURED** | positive control, fail-closed |
| A2 | 33.92% / 10.82% / 55.27% L1 split | **MEASURED** | cx1 n600, exact & exhaustive |
| A3 | `hg1` F1 confirmed, `Lane→Road` 0.7851 | **MEASURED** | cx1 n600, closes `hg1` A13 |
| A4 | `hg1` claim 4 refuted on the live vehicle | **MEASURED** | INSTANCE→VEHICLE. `hg1` is correct *for the witness*; the claim does not transfer. Two live lineages agree. |
| A5 | F-ACT-1: position-free area actuator dead | **MEASURED** | FORMULATION — kills *unconditional morphological* area moves on cx1, all 4 variants + all 10 directed sides at p<0.5. Does **not** kill a *gated* area move. |
| A6 | per-side precision 0.048–0.373, `p* = 0.500` | **MEASURED** | 100 SPREAD frames, ratio 0.9807 |
| A7 | directional prior = 10,565 B | **MEASURED** (the bits) / **DERIVED** (that a carrier can absorb them) | absorption is unverified until §5.1 returns |
| A8 | displacement blind to 31.5% Movable / 56.9% Lane | **DERIVED** from A2 + area-preservation | the derivation was **falsified once** (see A11) and then bounded |
| A9 | chroma penalty on the dominant edge = 1.254× | **DERIVED** from `rz1`'s 48.4%/55.0% + my 46.23% | arithmetic only; `rz1`'s retentions are its own |
| A10 | `hg1` 15.2% ⇒ 20.35% headroom-basis; 32.21% is a different quantity | **MEASURED + DERIVED** | chain reproduces `hg1` to 15.17% |
| A11 | translation is area-preserving ⇒ purely symmetric | **PARTIALLY FALSIFIED, corrected** | holds for **compact interior** components (antisym 0.74–2.29%) and for **horizontal** shifts (1.65%); **fails** for frame-touching strata (Road 98.71%, MyCar 99.43%) and for compact objects crossing a background transition (Movable vertical 48.77%). A8 uses only the *global* area-conservation form, which survives. |
| A12 | "84.6% of archive ⇒ not buyable" is a ceiling, not a cost | **DERIVED** | arithmetic; the *erasure* argument for the same conclusion is untouched |

## §7 REPRODUCTION

```
# all $0, scorer-free; reduces ddm_pu2's cached planes
CACHE=/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache
#   per_pair_directed.jsonl -> 5x5 directed confusion, split, Hodge/min-flow (§2,§3.1,§4-S1,S2)
#   gt_argmax_n600.npy + cx1_argmax_n600.npy -> morphology (§3.2,§3.3) and the A11 controls
```
Scratch drivers: `as1_orthogonality.py` (A11 round 1) · `as1_ortho2.py` (A11 round 2) ·
`as1_hodge.py` (the L1/L2 error, kept) · `as1_actuator.py` (F-ACT-1) · `as1_gate.py` (§3.3).

**Sources:** `.omx/research/ddm_hg1_negatives_as_geometry_20260803.md` ·
`ddm_mf1_margin_morse_licence_20260803.md` · `ddm_rz1_realization_attack_table_20260803.md` ·
`ddm_sx2_g2_gate_and_displacement_carrier_20260803.md` ·
`ddm_pc2_perclass_road_edges_20260802.md` · `ddm_pu2_pose_tail_floor_probe_20260803.md` ·
`ddm_rs2_flip_damage_rerank_and_drop_seg_leg_20260803.md` · `hs1` (`9da1e4afa1`) · `#900`.
