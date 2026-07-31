---
schema: ddm_ba29_negative_response_surfaces.v1
date_utc: 2026-07-31
window_audited: 2026-07-29
arm: ddm_ba29 (audit arm; READ + DERIVE only — no scorer job, no launch, no training)
lane_id: "lane_ddm_ba29_negative_response_surfaces_20260729"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[$0 cached-receipt re-derivation; every number carries its receipt path or file:line]"
operator_verbatim: "I've told you numerous times no binary results ever. We are proceeding Einsteinian and according to our design and guiding philosophies and principles, which include unification, completeness, and more."
verdict_scope: "N/A — this memo issues NO verdicts. It replaces 07-29's verdicts with the surfaces they were points on."
---

# ddm_ba29 — the 2026-07-29 negatives, re-read as FIVE response surfaces

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Bars 0.172141 / 0.15.
Nothing here is a score. This is a $0 re-derivation from receipts already on disk.

**The one-line result.** The 34 verdict-shaped statements dated 07-29 are measurements at
**5 surfaces**, not 34 findings — and the surfaces separate cleanly into two families with a
**~100× leverage difference that no verdict on either side mentions**: the *HOW* family (code
the object better) is measured **saturated within 1.381× across 19 coordinates**, while the
*WHERE* family (which support / which cells / how many planes) moved **39.85× (bytes)**,
**145.8× (support cost)** and **417,000× (pose)** inside the same 24 hours. Every 07-29
negative on the HOW side is sound and near-final **because 19 coordinates were measured**, not
because a falsifier fired. Every 07-29 negative on the WHERE side was taken at a coordinate we
had already left by end of day.

---

## §0 DENOMINATOR (stated first, per the honesty bar)

| | count | note |
|---|---:|---|
| Verdict-shaped statements found, dated 07-29 | **34** | across 20 `.omx/research/*20260729*` memos, 48 commits, 11 JSON receipts, 2 SSD receipt trees |
| Placed as points on a surface below | **31** | |
| Found but NOT placeable | **3** | listed in §7 with the reason |
| Re-derived from primary artifact (not quoted) | **11** | QA03 JSONL, QA11 sweep, of1 θ-sweep, r7 coder table, xi1 variant table, vae1 frame bytes, wr1 k-sweep, pfs1 D1 members, water level, the exchange arithmetic, the coder ratio |

**Search scope for the "34" (so the count is auditable, not asserted):** case-insensitive grep
for `REFUTED|NO-GO|DEAD|DOMINATED|FAILED|EXHAUSTED|SEALED|INFEASIBLE|EMPTY|NEGATIVE|KILL|RULED
OUT|CLOSED|REJECT|LOST|falsifier` over every `.omx/research/*20260729*` file, plus every 07-29
commit subject, plus the `ddm_deferral_queue_ledger_20260729.md` QA-row table (95 rows).
I did **not** exhaustively search `.omx/state/*.jsonl`, task rows outside the ledger, or
`reports/`; a negative recorded ONLY in those three places would not appear here.

---

## §1 THE INVARIANT THAT TIES TWO SURFACES TOGETHER (derive it once, it is exact)

Everything on the seg axis trades against bytes at ONE rate, and that rate is a property of
`upstream/evaluate.py`, not of anything we build:

```
S per flip  = 100 / (600·384·512)  = 8.477105e-07      [600 scored frames, 384×512 argmax]
S per byte  = 25  / 37,545,489     = 6.658590e-07
WATER       = 8.477105e-07 / 6.658590e-07 = 1.273108 B/flip     ← EXACT, IMMOVABLE
```

`1.2731` appears in of1, gc6, ru1 and the pp1 band lemma as "the water". **It is the only
truly fixed level set in this window.** Every other level set quoted on 07-29 is a function of
one of OUR knobs. That distinction is the whole audit.

---

## §2 SURFACE A — THE BYTE↔FLIP EXCHANGE (both signs of one object)

Spending bytes to fix flips and saving bytes at the cost of flips are **the same surface read
in opposite directions**. On 07-29 they were reported as five unrelated verdicts across three
arms. Placed together (all MEASURED unless marked):

| arm / point | coordinate | B per flip | vs water 1.2731 |
|---|---|---:|---|
| of1 SW-DERIVE (c) θ=0.05 optimistic | static flicker map, 27 comps | **0.0753** | **16.9× under** ✔ |
| of1 SW-DERIVE (c) θ=0.10 | | 0.0870 | 14.6× under ✔ |
| of1 SW-DERIVE (c) θ=0.20 | | 0.0894 | 14.2× under ✔ *(reported "falsifier fired" — on RECALL 0.206, not on price)* |
| of1 SW-DERIVE (c) θ=0.02 optimistic | 60 comps, recall 0.924 | 0.0983 | 12.9× under ✔ |
| of1 SW-DERIVE (c) θ=0.05 conservative | per-pair component pricing | 0.134 | 9.5× under ✔ |
| of1 SW-DERIVE (c) θ=0.02 conservative | | 0.141 | 9.0× under ✔ |
| of1 SW-DERIVE (a) rim R⊕1px | oracle, compliance-blocked | 0.079 | 16.1× under (ORACLE) |
| of1 SW-DERIVE (a) strict | recall 0.054 — falsifier FIRED | 0.937 | 1.36× under, reach negligible |
| **sb1 QA03 aimed token edits** | **tr1 re-encode price** | **1.4518** | **1.140× OVER** ✖ |
| sb1 QA11 bulk \|g\| snap q=50 | save side | 0.4300 saved/flip | 2.96× short ✖ |
| sb1 QA11 q=90 | save side | 0.3347 | 3.80× short ✖ |
| sb1 QA11 q=25 | save side | 0.0588 | 21.66× short ✖ |
| of1 P2C-OF offset field | — | **NO POINT** | never priced in bytes (§6 gap) |

Receipts: `/Volumes/VertigoDataTier/pact/ddm_of1_20260729/swderive_support_derivability_receipt.json`
· `.../ddm_sb1_20260729/qa03/qa03_receipt.json` (`byte_delta_tr1_reencode=2709`, `net_flips_total=1866`)
· `ddm_sb1_seg_batch_20260729.md:80-81` (QA11 curve).

### WHERE THE LEVEL SET IS, AND WHAT MOVES IT

The level set is fixed at 1.273108. **Position on the surface moves with three of our knobs,
all measured on 07-29:**

**(i) SUPPORT — the largest lever in the window, 145.8×.** of1's flicker channel was priced
DEAD at 1,415,927 B when the region supports were transmitted, and ADMISSIBLE at 9,711 B when
the same supports were derived from a static frequency map (`of1:139` vs `of1:217`). Same
channel, same coherence (0.869), same flips — **the entire verdict is a statement about where
the support description lives.**

**(ii) CODER — measured on Surface C, UNMEASURED here, and it is exactly where the only
above-water verdict of the day sits.** QA03's 2,709 B is a **tr1 re-encode**, flagged advisory
in its own receipt (`byte_delta_note: "true shipping price = r7 SMEVR coder (xi1 handoff)"`).
The same token object costs **767,812 B under tr1** (`qa03_receipt.json:base_archive_bytes`,
sha `b9a7983b…`) and **557,253 B under SMEVR** (`ddm_pfs1_d1_build_receipt_20260729.json:
members/state/tokens.dr7t`), non-token floor 12,743 B (`wr1:§0`) — a whole-object ratio of
**1.355×**. Scaling the marginal by the average (DERIVED, labelled, see the caveat):

```
break-even budget for 1,866 flips = 1866 × 1.273108 = 2,375.6 B   (7.29 B per accepted quantum)
measured  tr1  price               = 2,709   B   = 1.140× water   netS = +0.000222  (a LOSS)
average-scaled SMEVR estimate      = 1,999   B   = 0.842× water   netS = −0.000251  (a GAIN)
```

**The sign of the QA03 action flips on the coder, and the coder used to price it is not the
coder we ship.** DERIVED, not measured — the caveat that bounds it: a marginal cost need not
scale like an average cost, and a single-quantum edit that breaks a run in the occupancy model
can cost more than its share. The counter-evidence pointing the same way: SMEVR's mean delta
symbol is 555,836 B / 1,843,200 symbols = **0.302 B/symbol (2.41 bits)**, so the 7.29 B/quantum
break-even is **24× the mean symbol cost** — a very wide corridor for one changed L16 level.
The ONE measurement that resolves it: re-price the 326 accepted quanta through the landed
SMEVR coder ($0, the r7 harness already exists, xi1 already imports it).

**(iii) AIM DEPTH — the comparator is a function of k, see Surface B.**

### WHERE WE STAND, AND WHICH WAY IT FALLS

Six of the seven priced correction actions in this window are **9–17× under water**. The one
above water is above by **1.140×** and is priced by a coder 1.355× worse than the one we ship.
The surface falls toward *cheaper support descriptions*, and we are standing at the one
coordinate on it where support was free (the atlas) but the coder was wrong.

---

## §3 SURFACE B — SEG YIELD PER UNIT OF REPRESENTATIONAL FREEDOM

of1's offset-field verdict (`of1:95-103`, "5–21× BELOW ru1's +24 flips/quantum") is a **ratio
against a comparator**, and I re-derived the comparator from the same day's primary artifact.

`/Volumes/VertigoDataTier/pact/ddm_sb1_20260729/qa03/qa03_instances.jsonl` (120 rows) records
`accepted_steps` per instance. Summing: **326 accepted quanta, 1,866 net flips.** MEASURED:

| comparator | flips/quantum | offset-field deep (per-band DOF, 4.44) is |
|---|---:|---|
| ru1 predicted, best-of-8 at 17/18 hotspot cells (`ru1:111`) | **24.00** | 5.41× below |
| QA03 realized, rank-1 instance | 13.00 | 2.93× below |
| QA03 realized, top-24 | 8.92 | 2.01× below |
| QA03 realized, top-60 | 6.78 | 1.53× below |
| **QA03 realized, top-120 (the population that was run)** | **5.72** | **1.29× below** |
| P2c round-1, top-24 single-quantum (155 flips / 24) | 6.46 | 1.45× below |

**The comparator is not a constant — it is a decaying function of aim depth, and it is 4.20×
lower at the depth actually run than at the value the falsifier used.** Against the realized
comparator the offset field is **1.29×–5.02× below** (not 5–21×), depending on which DOF
accounting you use.

**Where the level set actually is, honestly.** of1's *structural* criterion is independent of
this comparator and it fired hard and cleanly: within-band thickness autocorrelation
**L = 1 px** (falsifier ≤3 px), conditional entropy 0.85 vs marginal 1.09 b/node (**MI only
0.24 b/node**), and the expensive `m_def ≥ 1.0` tail has **NEGATIVE lag-1 autocorrelation**
(`of1:71-74`). At the DOF accounting consistent with its own measured L=1, the offset field
yields 1.14–1.18 flips/DOF = **4.85–5.02× below** realized. **That reading is sound.** What is
not sound is the *generous* accounting (one constant offset per whole band, 4.44) being
reported at 5.41× when it is 1.29× — because that is the accounting a smoothing/merging
substrate would deliver, and its precondition tag ("reopens if autocorr > 3 px") is calibrated
against the inflated number. Against the realized comparator, the deep band at **L = ∞** (the
representation's absolute ceiling) is still 1.29× short — so the reopen condition is not
"longer autocorrelation," it is **"~29% more flips per connected band,"** which is a different
substrate requirement and a nearer one.

**What moves this level set:** (a) aim depth k — the comparator falls 13.00 → 5.72 from rank 1
to rank 120, so *every* "beats aimed editing" verdict is k-dependent; (b) band merging — the
offset field's yield is directly proportional to mean band arclength (deep: 3.77 px);
(c) the line-search budget — **51 of 120 QA03 instances saturated at the 4-quantum cap**
(`sb1:47`), i.e. the comparator itself was truncated, exactly the defect §5 shows destroyed the
pose number by five orders of magnitude.

---

## §4 SURFACE C — TOKEN CODING (the HOW family; measured saturated)

Three arms (r7, xi1, vae1) independently coded **the same** `[600,24,32,4]` L16 token object.
Placed on one axis, complete-frame bytes (r7 endpoint column `ddm_r7…:136-152`;
xi1 `:32-38`; vae1 `VAE1-E1/E3`):

```
554,219  ideal plug-in H(delta|base,prev_coloc,warp_bwd)   −0.54%   ← unreachable ceiling
557,238  SMEVR                                              0.00%   ← INCUMBENT (best of 19)
557,253  SMEVR reproduced same-object (xi1)                 +0.00%
569,515  xi1 A warp-context expert (backward)               +2.20%
569,775  xi1 A forward                                      +2.25%
581,771  vae1 V2 static pooled AR prior                     +4.40%
607,356  KT-prev1  ·  611,500 KT O8+prev5                   +9.0% / +9.7%
631,309  r7 CAE identity-INTER                             +13.29%
642,363  Brotli-Q11 · 644k logistic 4-expert · 652,879 LZMA2-64K
656,921  LZMA1 · 656,968 LZMA1x pool · 658,280 Bayes 4-expert · 658,414 G4 prefix
662,667  rANS o0 · 663,030 rANS pool · 675,086 canonical Huffman
723,124  xi1 B warp innovation (backward) · 757,604 forward
769,613  rANS o0 on adjacent innovation                    +38.1%
-----
190,334  the 0.172 bar target                              −65.8%
157,294  the 0.15 target                                   −71.8%
```

**19 measured coordinates. Total span 1.381×. Best-known ideal −0.54%. Target −66% to −72%.**

**Credit where it is due — the conclusion WAS drawn in this window, qualitatively.** gc6:219
states *"lossless coding alone is DEAD for the ≤200 KB regime (already effectively measured)"*
and gc6:36 surfaces *"the 130 KB token target is reachable by coding alone"* as an explicit
assumption; gc8 row C re-scopes r7's rate verdict to *"the floor is ALPHABET/CONTEXT-conditional,
not a rate wall; verdict TRUE only on this alphabet."* What is NOT anywhere in the window is
the **quantitative axis** — each of the three arms reported only its own distance from SMEVR
(xi1 +12,262 B; vae1 +24,533 B; r7 a rank-ordered table), so the fact that all 19 coordinates
fall inside **1.381×** while the target is **−66%** was never put on one line. Read together
they are not three failures; they are **one exhaustive measurement of a coder-quality axis, and
it is saturated at the top.** The value added here is the span and the ceiling, not the verdict.

**What would move the level set:** not the coder. xi1 measured the ceiling directly — the
warp expert's IDEAL plug-in gain over SMEVR's realized delta is **1,617 B (0.29%)** while its
adaptive learning tax is **12,262 B**; the information is smaller than the dilution
(`xi1:99-107`). And the plug-in estimate is biased *downward* with more contexts (fewer counts
per context), so 1,617 B is an over-estimate of the available gain. The only named coordinate
with reach on this surface is **granularity/alphabet** (QA24, currently BLOCKED behind two
falsifiers) — xi1 §5 says the warp needs ≥48×64 before displacement exceeds one cell; gc8 row C
says the same thing from the other side: *"the floor is ALPHABET/CONTEXT-conditional, not a rate
wall; verdict TRUE only on this alphabet."*

**This is where the coordinator's pp1 template INVERTS, and that inversion is information.**
In pp1, coder quality was an unexploited axis that could move the band edge. Here the identical
axis has been measured to exhaustion on this object, so on *this* object it can move the edge by
at most 0.29%. The two statements are consistent and together they say: **coder quality is a
real degree of freedom whose remaining travel must be measured per-object, and on the 07-29
token object it is spent.**

---

## §5 SURFACE D — OBJECT MEMBERSHIP (the WHERE family; same units, 39.85× span)

wr1 measured 12 coordinates of a *different* question — not "how well do we code the object"
but "how much of the object does the scorer read" — in **the same units**
(`ddm_wr1…:§2`, real SMEVR re-run per tranche, byte-closed archives):

| k cells dropped | archive B | rate term | vs Surface C's best |
|---:|---:|---:|---|
| 0 | 569,996 | 0.3795 | — |
| 100 | 482,742 | 0.3214 | already −15.3% |
| 300 | 346,671 | 0.2308 | −39.2% |
| **486 (Knee A)** | **274,333** | **0.1827** | **−51.9%** ← 53.1% of the token member |
| **600 (Knee B)** | **174,578** | **0.1162** | **−69.4%**, inside the sub-0.15 budget (≤179,467 B) |
| 768 | 14,303 | 0.0095 | −97.5% |

**Surfaces C and D are in the same units and differ in leverage by ~29× (1.381 vs 39.85).**
gc6:219 / gc8 row C reach this conclusion qualitatively (§4); what the numbers add is the
*size* of the gap and the fact that **it is measured on both sides** — 19 coordinates bounding
C, 12 bounding D — so it is a mapped ratio, not an inference. The unification it forces: the
07-29 rate negatives (xi1 ×2, vae1 ×1, r7 ×15) are not evidence that rate is hard — they are
evidence that **rate is not on the coding surface at all.**

### The Knee-A "REJECT" and where its level set actually lives

QA06 measured Knee-A standalone at **S 2.4097 vs ref 2.2566 = +0.153 net REJECT**
(`ck1:23-24`, real `evaluate.py` rc=0, full n600): rate −0.197 won, but d_seg +0.165 and
d_pose +0.185 lost. Within the same day ck1 re-solved the pose **on the base the candidate
actually ships** and the reject flipped:

- Knee-base two-plane tail best_mean d_pose **0.3609** ≈ full-base two-plane **0.3692** —
  **0.98× recovery parity.** The 486 dropped cells cost the pose axis ~nothing once re-solved
  (`ck1:49-53`). The +0.185 S pose regression was **entirely stale parameters**, not a
  capability loss.
- Composed S with only the tail-112 re-solved and non-tail left STALE: **1.9863 = −0.270 vs
  ref** (`ck1:80`).

**The verdict "+0.153 REJECT" was measured at a coordinate — pose-params-solved-on-a-different-
base — that was not part of the surface as drawn.** Adding that coordinate moves the point by
−0.42 S. The single most transferable lesson in my window: *a composed candidate must be
re-solved on the frames it ships, and any composed verdict that skipped that step is measuring
staleness, not the candidate.*

ck1 §2 also records a coupling worth carrying forward on its own: the dropped sky
cells FROZE the far field, which is precisely what makes the **single**-plane homography
Jacobian degenerate (single-plane ran ~1.7–2.0 on the Knee-A tail; pair 19: 2.0549 vs 0.4245)
**and** exactly what the **two**-plane far→H∞ branch exploits. The same physics is the defect
and the cure depending on generator structure.

---

## §6 SURFACE E — POSE GENERATOR STRUCTURE (the widest travel measured in one day)

Every 07-29 pose number is a point on one surface whose coordinates are
`(start · basis · GN budget · plane count · per-pair vs shared · quantization · which base)`.
Contribution = √(10·d_pose):

```
38.06    P3 naive: start=zeros, rank-6 cosine basis, ~2 relinearizations   ← "the photometric wall"
~15      rank-6 cosine, run to convergence   (the basis alone plateaus here)
10.22    start = stored render
 1.9827  p3v2 s3 warp on ct1 frames (STALE frames — max_abs 255 vs shipped)
 1.4884  pfs1 D1 warp on the archive's OWN f1  (7,200 B → 6,864 B coded)
 1.4383  rank-1 int8 e_p                       (702 B)
 1.3159  rank-4 int8 e_p                       (2,004 B)   ← best ≤4KB point
 1.2630  6-DOF per-pair f16, ONE plane         (7,200 B)   ← "post-hoc saturates here"
 1.2499  ck1 Knee-A composed, non-tail stale
 0.9127  P0 + TWO-PLANE selection, tail-112    (+~75 B)    ← same day, +75 bytes
 0.9040  ck1 optimistic bracket
─────────  0.5  ← the pre-registered falsifier bar
 0.0302  free-frame_0 unpriced ceiling (n24, 100% of pairs ≤1e-3, median 5.8e-5)
 0.0153  PR130-grade
```

**Two measurements of the same knob, hours apart:**

1. **Start · basis · budget: 38.06 → 9.123e-5 d_pose, a factor of ~417,000.** p3v2 §1 names all
   three and says plainly: *"None of those three choices is a vehicle property."* The rank-6
   cosine basis was RANK-DEFICIENT — this is the same-day
   `generic_basis_metric_never_optimal_cosine_fourier_euclid` law, measured.
2. **Plane count: 1.2630 → 0.9127 at +75 B** (qa43 tail-112 final, 95/112 wins, 77 >2×, 41
   >10×; `qa43:67-70`). pfs1 D2 had written *"post-hoc warp+e_p saturates at contribution ~1.26
   (mean-tail-bound)"* and routed **v10 SPEC row-12 pose-in-burn back to REQUIRED** on that
   basis (`pfs1:§6.1`). The saturation claim was falsified the same day by moving one
   coordinate the surface had been drawn with fixed at 1. qa43 §2 says it directly: the
   *"content limit"* is **partly a GENERATOR limit**.

**Where the level set is:** the 0.5 bar is crossed by nothing measured; nearest approach 0.9127
= **1.83× above**. But the *known floor* on this surface is 0.0302 — **30× below the bar** — so
the bar is not near a wall, it is in the middle of a region we have measured on both sides.
The gap between 0.9127 and 0.0302 is entirely generator/carrier-price, not reachability.

**Carried caveats that travel with these numbers (both from the arms themselves):** qa43/ck1
two-plane masks are GT `lstars` — an UPPER BOUND until a rule-118-legal mask source (static
geometric prior or decoded partition) is raced (`ck1:64-68`). And p3v2's own Contrarian booking:
the free-frame_0 win is **basis-adversarial** — it does not survive cheap generic coding
(`p3v2:§0`). Both bound the reach; neither restores the "wall."

---

## §7 UNIFICATION — the shared coordinates

The five surfaces are not independent. Four coordinates appear on more than one, so moving one
knob moves two surfaces:

| shared coordinate | surfaces it sets | measured travel on 07-29 | state |
|---|---|---|---|
| **Coder quality** | **A** (sets B/err → position vs water) and **C** (sets frame bytes) | 1.381× across 19 coords on C; **UNMEASURED on A** | exhausted on C, **untouched on A — and A is where the day's only above-water verdict sits** |
| **Granularity / alphabet** (24×32 L16) | **C** (xi1 §5: warp needs ≥48×64) and **D** (cell count = membership grain) | not measured — QA24 BLOCKED behind two falsifiers | the only named coordinate with reach on C |
| **Solver quality** (start · basis · budget · DOF class) | **E** (pose) and **B** (seg yield) | 417,000× on E; on B the comparator was **truncated at 4 quanta in 51/120 instances** | turned hard on pose, **demonstrably unturned on seg** |
| **Support / aim description** | **A** (where the bytes go) and **B** (what the comparator is) | 145.8× on A (1,415,927 → 9,711 B); 4.20× on B (24 → 5.72) | the largest lever in the window |

**The synthesis.** Sort the whole window by *which family the knob belongs to*:

- **HOW (code the same object better)** — total measured travel **1.381×**, 19 coordinates,
  saturated at the top, ideal remaining 0.29%. Every negative here is sound and near-final.
- **WHERE (which support · which cells · which planes · which start)** — measured travel
  **39.85×** (bytes), **145.8×** (support cost), **417,000×** (pose), **4.20×** (comparator) —
  all inside the same 24 hours, all at ~zero marginal bytes.

On the rate axis the two families WERE compared qualitatively (gc6:219, gc8 row C); what was
not done is comparing them **numerically**, or noticing that the same split recurs on the other
two axes. **They are one object: the archive is a description, and on this vehicle the
description's CONTENT (what is described, and in whose coordinates) has 100–1000× the leverage
of its ENCODING.** The rate axis is the sharpest instance — three arms bounded coding at 1.381×
on the same day one arm measured membership at 39.85× — but the identical shape holds on pose
(plane count, +75 B for −27.7%) and on seg (support description, 145.8×). That recurrence
across three independent axes is what makes it a property of the vehicle rather than a fact
about token coders.

### Sound and near-final, and why (stated plainly, per the honesty bar)

- **Surface C's saturation.** Sound because 19 independent coordinates were measured on the
  same object with byte-exact roundtrip closure, *and* because the ceiling was measured
  separately (1,617 B ideal, itself a downward-biased estimate). This is not a falsifier
  firing; it is a mapped surface.
- **of1 Probe 1's structural criterion.** Sound because L=1 px and MI 0.24 b/node are direct
  measurements of the field with no comparator in them. Only the *ratio* attached to them is
  inflated (§3).
- **QA11's ν=0 on the S-exchange, not just the tolerance.** The arm tested a d_seg tolerance
  (+2e-4); I re-tested on the real economics and it still does not cross — but the **closest
  approach is q≈50 at 2.96× short, not q=25 at 6.9× tolerance**. Same conclusion, different
  coordinate, and the two level sets cross at ΔB ≈ 30,036 B, so on any bulkier lever the
  tolerance test would be **up to 23× stricter than the economics** and would refuse a
  profitable action. The instrument, not the verdict, is what needs fixing.

### The three I could not place

1. **gc8 row A — the seg plateau "endpoint FLAT, last window worsening."** Its own memo
   RE-SCOPES it to "an unfalsified-form suspect" (335/400 epochs ran CE under a tau label).
   There is no measured tau counterfactual anywhere, so it has no coordinate. Placing it
   requires the QA22 A/B; anything else would be manufacturing a location.
2. **uh1 row 4 — QA11 prereg `baseline_full_dseg` 0.013833 vs measured q=0 0.0038892 (3.56×).**
   Flagged unresolved by its own arm and routed to sb1. It is either a stale constant or a
   protocol-lineage confound; I could not determine which from receipts and did not want to
   guess. Note it is the *same* number the QA11 executor's `nu=0.7` artifact is hardcoded
   against (`sb1:76-78`), so one root cause plausibly explains both.
3. **cl1 CliffordNet dispositions.** A crosswalk table with named falsifiers but no measured
   points on our object — it is a queue of unmeasured coordinates, not measurements.

---

## §8 MY OWN ROUND-1 ADVERSARIAL REVIEW

- **"Did I assume a unit?"** The water 1.273108 assumes d_seg is normalized over 600×384×512.
  I did not re-read `evaluate.py` this session; I took the denominator from the same
  `flips/(384·512·600)` formula of1 states as DERIVED (`of1:288`) and it reproduces the 1.2731
  the arms use to 5 digits, which is a consistency check, not an independent verification.
  **Labelled: DERIVED, cross-checked against three arms' usage, not re-read from source.**
- **"Is my strongest claim a class-fix or a point-fix?"** The coder-scaling estimate
  (1,999 B) is a POINT estimate on ONE action. The class statement — "an advisory re-encode
  price was used to adjudicate an action against an exact water level" — is the real finding
  and it is structural. I flagged the specific number as DERIVED and named the $0 measurement
  that settles it.
- **"Would my conclusion survive if QA03's `accepted_steps` meant something else?"** If
  `accepted_steps` recorded *attempted* rather than *accepted* steps, the population rate 5.72
  would be a lower bound and the of1 comparator gap would narrow further — my §3 direction is
  robust to that ambiguity, and 12 instances have zero accepted steps and zero net flips, which
  is consistent with "accepted."
- **"Am I replacing one binary with another?"** The risk is real: saying "HOW is saturated,
  WHERE is not" could become a new label. It is not a label — it is a **measured leverage
  ratio per axis** (1.381 vs 39.85 in identical units; 24 vs 5.72; 1,415,927 vs 9,711), and
  each ratio names the coordinate that produced it and can be re-measured on a new object. On
  a finer lattice the HOW/WHERE ratio would have to be re-derived, not assumed.
- **What I did not check:** I did not re-run any coder, did not verify wr1's byte-closed
  archives, and did not verify that QA03's base archive `b9a7983b` has the same non-token
  members as pfs1 D1's (the 12,743 B floor subtraction assumes it does; if the p2c packaging
  carries a different floor, the 1.355× ratio shifts by that amount).

## §9 STORES CONSULTED

CLAUDE.md · AGENTS.md · `docs/operating_manual_craft_handoff.md` · `tac.subagent_contract.standard_contract()` ·
memory `boolean_flags_are_a_ui_over_a_continuum_never_binary_judgment_20260731` (the diagnostic),
`negative_existence_claims_are_the_days_dominant_error_class_20260731`,
`verdict_scope_ladder_formulation_level_one_failure_not_family_dead_20260708`,
`generic_basis_metric_never_optimal_cosine_fourier_euclid_20260729`,
`pose_is_the_largest_axis_on_the_own_vehicle_1_24_S_20260731`,
`opportunity_pools_non_additive_rate_distortion_reachable_20260718`.
07-29 memos: ng1 · uh1 · of1 · xi1 · sb1 · wr1 · ck1 · qa43 · pfs1 · p3v2 · r7 · vae1 · gc6 · gc7r ·
gc8 · ru1 · stl1 · cl1 · pb1 · deferral ledger. Receipts:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260729/qa03/{qa03_receipt.json,qa03_instances.jsonl}` ·
`.../ddm_of1_20260729/` · `.../ddm_xi1_20260729/` · `.omx/research/ddm_pfs1_d1_build_receipt_20260729.json`.
**Deliberately not loaded:** `.omx/state/*.jsonl`, `reports/`, task rows outside the ledger (named as a
coverage gap in §0).

## §10 POINTER-DELTA HONESTY (last, as first)

**`0.1910828242 [contest-CPU]` UNMOVED.** This memo is a MEANS — it moved no number. It converts
34 verdicts into 5 surfaces and names, for each, the knob that moves its level set. The one
action it makes cheap and decisive: **re-price QA03's 326 accepted quanta through the landed
SMEVR coder** ($0, existing harness) — that single measurement settles the sign of the entire
aimed-correction family on the seg axis.

[no-triality] [p0-ledger-ok] — audit arm; no DSL lever, no canonical-equation surface, no
trainer change. `score_claim=false · promotion_eligible=false · pointer_moved=false`.
