# ddm_bs2 — born-small is REAL and distinct, its rate case is MEASURED and wins, and its distortion case is where every prior points against it

`date_utc: 2026-08-24` · `axis: [macOS-CPU advisory, scorer-free exact field measurement]` ·
`score_claim: false` · `promotion_eligible: false` · `pointer_moved: false`

`verdict_scope`: **FORMULATION** — the born-small route as named by sy2 rank-3 / wq1 D5, priced on
the retained HG1 generator field against the dx2 object. This does NOT close a learned implicit
evaluator-cell carrier, and it makes no nonexistence claim over born-small representations in
general. It closes nothing; it ranks one route and names its deciding measurement.

---

## 0. Answer first

**Born-small is genuinely distinct from trained-small. It is not touched by nt1's reframe. And it is
still not the next thing to fire.**

Three findings, in the order that matters:

1. **The distinction holds.** "Trained-small" picks a smaller point on dx2's *existing*
   rate-distortion curve. "Born-small" is a *different curve*. nt1 retired the first and proposed a
   third thing (flatten the curve in place). All three are different objects, and nt1's reframe —
   *"minimize `dD/dB`, not `B`"* — is an argument about how bad dx2's curve is, which is an argument
   **for** a different curve, not against one.

2. **The rate case is MEASURED and it wins outright.** HG1's analytic generator vocabulary costs
   **47,667 B** against the 127,292 B it replaces (token stream + HPAC model) — a **62.55% cut**,
   79,625 B saved. Dropping the exactness residual gives a **101,128 B** container: **36,858 B under
   the sub-0.12 cap**, clearing sy2's own 26.80% admission bar with **37,243 B of margin**. This is
   the only route measured on this vehicle whose rate leg is not merely adequate but surplus.

3. **The distortion case is where it dies or lives, and I measured the structure myself.** The
   residual-dropped container has **0.024543 S** of distortion headroom — `d_seg` may rise 2.219×,
   or `d_pose` 16.61×. It clears only if **≤ 3.9484%** of the 1,334,939 missed tokens become argmax
   flips. My measurement says that miss is concentrated exactly where argmax is most fragile:
   **23.38× boundary-enriched**, **Lane over-represented 40.81×**, and its dominant mode is
   **Lane→Road erasure (309,429 positions, 97.0% of the Lane miss)** — the campaign's signature
   failure. **17.0% of the Lane erasures alone would exhaust the entire budget.**

**My own prediction's falsifier FIRED.** I predicted prior basis work existed but was *never priced
against the current demand*. False: `et1`, `hg1`, and `hr3` priced exactly this basis against
exactly this demand on **2026-08-23, the day before this charter**, and refused it. The assets are
not stranded. They are measured, and byte-dead by 240,428–397,775 B *before distortion is charged*.
What survives is narrower than I predicted and I grade it as a partial refutation in §6.

**And the charter contains one factual error I must return: `ddm_tv1` does not exist.** It has never
been spawned, chartered, or run. It is `wq1` SPEC B, written yesterday, unclaimed. The gate I was
told to name as a live dependency has no arm behind it (§1.4).

---

## 1. RECALL — what sy2 proposed, and what is actually built

### 1.1 sy2's rank-3 row, at source

`ddm_sy2_composition_synergy_deep_pass_20260823.md` @ `fe2ba12dc2`, §"Candidate hybrid" table row 3
and §"Disposition". Verbatim, the object:

> **Born-small edge/topology carrier + new-alphabet HPAC refit + implicit traversal + terminal exact
> pose solve** … "The edge/generator basis removes dense interior and duplicated edge obligations;
> the new alphabet then gives the context model and traversal local runs and implicit addresses they
> did not have on the raster token field."

Its **admission bar**, verbatim: *"Holding carrier/residual/framing at 22,220 B, renderer+HPAC+tokens
must fall from 158,148 B to **≤115,766 B, a 26.80% cut**. No old leg is booked toward it."*
Honesty label: **`DERIVED admission bar + CONJECTURED representation`**. Disposition:
`QUEUED-BEHIND-JF1/W96`.

sy2 also wrote, and this is the claim I was sent to check:

> "The v8/v9 corpus was checked before naming this basis. The edge-centric carrier is not a new SY2
> invention: it is already derived and partially built. The honest new work is current-DX2/W96 live
> integration, actual coding, receiver consumption, and scorer closure."

### 1.2 Adjudicating that claim

**"Already derived" — ACCURATE.** `SPEC_v8_perclass_decomposition_20260708.md` derives it bindingly,
including the exact cure sy2 cites: *"the decomposition is **EDGE-CENTRIC, not class-naive.**
Separatrix information is SHARED between adjacent classes; one field per adjacency-graph EDGE …
never two region fields paying for the same curve twice."* §8(2) makes a class-naive build *"a spec
violation, not a variant."* The partition form is `P(x)=argmax_c(φ_c(x)+b_c)` with separatrices
*"DERIVED never represented."*

**"Partially built" — OVERSTATED, in the one place it mattered.** The artifact sy2's row credits —
the V9 DecisionCarrierBundle — **has no source code anywhere in the repository or its history.**
`decision_carrier_bundle.py`, `decision_palette_chroma.py`, `decision_carrier_policy.py` survive
only as orphaned `.pyc` bytecode. The codex session that wrote them recorded its serializer failing
at `git add` with rc 128 (`Operation not permitted`), *"no commit SHA exists"*. sy2 inherited a
"built default-OFF" label describing files that had been unrecoverable for six weeks.
`grep`/`git log --all` over those three names: empty. MEASURED.

**"The honest new work is … actual coding, receiver consumption, and scorer closure" — SUPERSEDED BY
MEASUREMENT WITHIN HOURS.** Three arms did precisely that work on 2026-08-23.

### 1.3 What is actually built, graded

| surface | grade | run on dx2? |
|---|---|---|
| `boundary_math/{partition,region_merge,dense_raster_lzma_baseline}.py` — RAG, MDL merge solve at λ*=1.27 B/flip | built+tested | no |
| `boundary_math/{power_diagram_witness,laguerre_logit_offset}.py` — Laguerre machinery + synthetic recovery control | built+tested | no (one byte-close attempt BLOCKED, never produced a byte count) |
| per-class fitters `{lane_sdf_component, analytic_lane_render_band, movable_site_coder, road_horizon_component, hood_static_component}` | built+tested | **`analytic_lane_render_band` YES** — executed by HG1 |
| `boundary_math/road_undriv_bulk_field.py` — the named v8 increment-1 carrier | **scaffold**, self-declared `research_only=True`, *"NOT the increment-1 build"*; gating probe **P-C never run** | no |
| `boundary_math/contour_codec.py` | decoy — its own docstring: *"It is not a boundary-edge, chain-code, or contour codec."* | no |
| `experiments/ddm_et1_…py`, `experiments/ddm_hg1_…py` | **built+tested+BYTE-CLOSED** | **yes, 2026-08-23** |
| V9 DCB | **source destroyed** | never |

### 1.4 The measured record this route must answer to

**On dx2, receiver-closed, exact parse-back, scorer-free `[macOS-CPU advisory]`:**

| arm | container | vs 137,986 B cap | vs 180,218 B zero-distortion cap |
|---|---:|---:|---:|
| ET1 implicit space-time BSP generator tree | 535,761 B | +397,775 | +355,543 |
| **HG1 V8 heterogeneous analytic roster** | **460,408 B** | +322,422 | +280,190 |
| HG1 BL1+MS9 protected bracket (`d_seg` UNMEASURED) | 420,646 B | +282,660 | +240,428 |
| HR3 width-8/16 Fourier residual-action INR | 463,601 B | +325,615 | — |

Both closures are `verdict_scope: FORMULATION`. HG1's guard, verbatim: *"It is not a mathematical
nonexistence claim over every possible learned implicit carrier."*

**The July prior that predicted this, and that sy2 did not surface.** The v8 corpus measured its own
rate ledger on the GT `L*` object (2026-07-09/10, real bit-exact coders, `[macOS-CPU advisory ·
NON-PROMOTABLE]`, registered `v8_geometric_rate_decomposition_v1`): geometric **dominant-only 0.061 S
= 91,611 B**; geometric **complete-lossless 0.140 S = 210,255 B**; **residual enemy 0.074 S =
111,135 B**. The complete-lossless price already exceeded dx2's entire archive — on an easier object,
counting rate only. HYPOTHESIS for dx2, not a transfer. It pointed exactly where HG1 later measured.

**The one v8 carrier ever byte-closed WITH distortion** (`ddm_cb1`, 2026-07-25, n600 through a real
emitted runtime + composite receiver + frozen CPU-torch scorers, `[macOS-CPU frozen-scorer
advisory]`, `verdict_scope=INSTANCE`) — and it splits by class:

- MyCar static mask: **+319 B, ADMIT** (Δ`d_seg` −1.05e-5, Δ`d_pose` −0.179) — helped both axes.
- polished v13 Lane band: **+1,530 B, REJECT** (Δ`d_seg` **+0.0366**, Δ`d_pose` **+22.7**).

That Lane figure is **149.1× this route's entire distortion headroom** (§4). Different vehicle,
INSTANCE scope — a hypothesis here, not a transfer. It is still the most relevant measured prior.

**The structural fact the whole inventory produces.** Alternatives split into two disjoint failure
modes and **nothing has ever been built in the middle**:

- **Lossy alternatives win on bytes, die on distortion.** RC1 = **113,006 B** and NR1 = **122,250 B**
  are both *below* the 137,986 B cap — and score **S = 17.306** and **S = 27.798**.
- **Exact alternatives win on distortion by identity, die on bytes.** ET1/HG1/HR3/WS1 at
  460,408–918,904 B. WS1 is the one true `FAMILY_NO_GO`; it killed the *explicit sampled-boundary
  worldsheet*, which is precisely what the v8 edge basis is defined against.

**Born-small has never been built.** SEARCH SCOPE: `.omx/research/` (10,710 files), `charters/`,
`arm_final_messages/`, `.omx/state/codex_arm_*`, `git log --all`. It exists only as sy2 rank-3 and
wq1 SPEC C (`ddm_bs1`).

**`ddm_tv1` does not exist.** Same scope, plus `grep -rn "ddm_tv1"` over the repo excluding `.git/`.
One hit total, and it is the proposal: `ddm_wq1…:198` **SPEC B**, written yesterday, unclaimed. The
charter's statement that a sister arm "is measuring right now" is **incorrect**. The question is
open and unowned (§5.2).

**Charter citation correction.** The charter cites *"W96 was measured and refused (rf1, 478.7×)"*.
At source, `ddm_rf1_renderer_film_rung_20260824.md:6` records the refusal at **2.7749× the matched
base**; 478.7× is that rung's **exchange ratio** (damage ÷ credit), a different quantity at a
different level. rf1 `:148` records a prior reader making this exact substitution and names it the
wrong route. Genus: `the-instruments-own-units-level-and-aggregation-are-part-of-the-claim`.

---

## 2. The adjudication — born-small vs trained-small, made operational

### 2.1 There are three objects, not two

| | what it changes | nt1's verdict | status |
|---|---|---|---|
| **(a) trained-small** | the operating point `B` on dx2's **existing** curve `D_R(B)` | **RETIRED** — *"aimed at the wrong constraint"*; and rate was already minimized under a mandatory λ | closed |
| **(b) cheap-to-shrink** | the **local shape** of `D_R` — a second-order flatness penalty, same representation class | **PROPOSED** (nt1 §3c) | unbuilt, unmeasured |
| **(c) born-small** | the **curve itself** — a different `D_{R'}` | not addressed | never built |

nt1's reframe retires (a) and proposes (b). It is silent on (c) — and its content is a statement
that dx2's curve is steep, which is a reason to want a different curve, not a reason to doubt one.
**The charter's hypothesis is upheld: born-small is a basis change, trained-small is a point
change.** They are not the same object.

### 2.2 But the reframe reshapes what born-small is allowed to claim

Born-small may **not** claim victory by being small. tri1's property 3 ("movable mass must be
structural") only counts mass that is movable *at acceptable distortion cost* — the 1,089.17 B figure
is already `dD/dB`-conditioned, since granting every rung its bytes at **zero** distortion yields
**152.40%** of demand. So property 3 and nt1's reframe are consistent, and neither is a claim about
size. Any born-small proposal is a claim about `dD/dB` and must be priced as one.

### 2.3 A synthesis neither arm made: tri1 property 2 ≡ nt1 §3c

tri1 requires a new object to *"separate its rate-bearing structure from its frame-bearing
structure."* Taken literally that is **impossible for any representation**: both scorers read the
decoded frames, so every counted byte exists to influence frames — otherwise deleting it would be
free rate with zero distortion. No rate move is frame-neutral **in any basis**.

Read correctly, property 2 means: *rate-reduction directions should map to frame perturbations lying
in the scorers' joint low-sensitivity subspace.* That is exactly `dD/dB`. **tri1's property 2 and
nt1's §3c are the same requirement in two vocabularies.** Both landed 2026-08-24; neither cites the
other. The consequence is the decision-relevant part: **born-small and cheap-to-shrink target the
identical quantity by different means**, so the cheaper instrument tests the shared hypothesis first.

### 2.4 The operational separator

> **Is dx2's high `dD/dB` a property of its BASIS, or of its OBJECTIVE?**

The objective reading is live and cheap. `grep -c -i pose` on the live trainer returns **0**: pose is
**5.38472%** of S at **6.2647×** seg's marginal sensitivity and has never entered a gradient. Pose is
also 93–100% of the damage in every measured refusal (dg2 93.3–93.4%, w72 65.3%, ap1 100%, rf1
97.59%). An object never shaped to protect pose may be **gratuitously** pose-fragile.

- **Objective-induced** ⇒ (b) fixes it on the existing representation ⇒ born-small is **premature**.
- **Structural** ⇒ (b) cannot ⇒ born-small is the answer.

That is tri1's E1 vs E2′ split, decided by one already-specified measurement (tri1 rank 2): retrain
with pose in the loss and re-measure the realized cost of ap1 `carrier_l1` — today **0.306332 S at
exactly zero seg cost** for 2,742 B, 100% pose-dominated. This is why born-small ranks behind: **not
because a gate is closed, but because a cheaper instrument tests the same quantity.**

---

## 3. The rate case — MEASURED, and it wins

All figures recomputed from components; S never read from a display (`#877`). Exchange rate
**6.658590e-07 S/B** CITED from `ddm_tx1_toolbox_crosswalk_20260819.md` §0, not re-derived. My
independent `25/N = 6.658590e-07` agrees to 7e-8 relative.

HG1's container, verified at source (`:201`, *"zero unassigned bytes"*):
`52,962 + 47,667 + 359,280 + 499 = 460,408 B`, where 52,962 = dx2's inherited renderer 30,856 +
carrier 22,010 + compact residual 96.

| quantity | value |
|---|---:|
| generators replace tokens+HPAC | 47,667 B vs 127,292 B — **62.55% cut, 79,625 B saved** |
| sy2's admission bar (renderer+HPAC+tokens 158,148 → ≤115,766) | **26.80% cut required** |
| HG1 same leg (renderer 30,856 + generators 47,667) | **78,523 B = 50.35% cut** — **37,243 B under the bar** |
| container with the residual DROPPED | **101,128 B** |
| headroom under the 137,986 B sub-0.12 cap | **36,858 B** |

That 36,858 B independently reproduces the budget figure `hr3` cites against its w128 model
(39,481 B > 36,858 B) — two arms, same number, arrived at separately.

**The generator vocabulary is not the problem and never was.** HG1 `:106` says so: *"the residual,
not the 47,667 B generator vocabulary, dominates the exact [container]."* **78.04% of HG1's container
is the exactness residual.**

**Is dropping it legal?** Yes. The unique-home law requires every byte to have exactly one home, not
every position to be reproduced. With no residual section there are no residual bytes to home.
Exactness was bought for two reasons, both recorded: the accounting law, and **measurement economy**
— HG1 `:39`, *"categorical field exact; inherited current distortion by identity, no new scorer
claim."* Exactness is the price of **not firing the scorer**. It is not a contest requirement (the
contest scores archive bytes, `d_seg`, `d_pose`) and not a scorer requirement (`db1:120`:
*"scorer-cell equivalence is a **weaker requirement than exact token identity**"*). `wq1` D3 grades
it **NAMED, NEVER MEASURED**; `vf1` measured the evaluator-visible credit denominator as
**0 / 117,964,800 positions, 0 / 113,777 coded bytes = 0%**.

**So the whole route reduces to one trade: pay 359,280 B to inherit distortion by identity, or pay
one scorer fire to measure it.**

---

## 4. The distortion case — my measurement, and it points the other way

### 4.1 The budget

At 101,128 B: rate = 0.067336984, leaving **0.052663016** for distortion against dx2's 0.028120228 —
**0.024542788 S of headroom.**

- pose held ⇒ `d_seg` may reach **4.4682e-04** (**2.219×** dx2's 2.0139e-04)
- seg held ⇒ `d_pose` may reach **1.0578e-04** (**16.61×** dx2's 6.37e-06)

Both are real budgets. The route is not obviously infeasible on arithmetic.

### 4.2 The transfer threshold

The generator-only field misses **1,334,939 / 117,964,800 = 1.131642%** of positions — **56.2×**
dx2's entire pixel-error rate. It clears **iff ≤ 3.9484%** of missed tokens become argmax flips
(≤ 52,708 flips), with pose held.

### 4.3 What I measured — `experiments/ddm_bs2_generator_miss_structure.py`

Diffed the retained generator field (sha `2884c570…`) against dx2's retained categorical field (sha
`cc10a7b0…`, pin-verified against HG1 `:65` before use). **Reproduces HG1's 1,334,939 exactly**, and
its class shares to 4 dp.

**Positive control:** my independently computed class areas reproduce CLAUDE.md's n600 values —
Road 23.233%, Lane 0.586%, Undrivable 49.518%, Movable **1.238%**, MyCar 25.425%. The Movable figure
is the n600 value, not the n96 prefix's 1.56% ([[m88]]). Field, geometry, and class order all check.

| true class | area | miss | over-rep | dominant confusion |
|---|---:|---:|---:|---|
| Road | 23.233% | 639,336 | 2.06× | → Movable 264,776 (41.4%) |
| **Lane** | **0.586%** | **319,147** | **40.81×** | **→ Road 309,429 (97.0%)** |
| Undrivable | 49.518% | 262,741 | 0.40× | → Movable 229,282 (87.3%) |
| Movable | 1.238% | 1,331 | 0.08× | → Undrivable 813 (61.1%) |
| MyCar | 25.425% | 112,384 | 0.33× | → Road 111,929 (99.6%) |

- **Boundary enrichment 23.38×.** Class boundaries are 2.17% of the field and hold **50.65%** of the
  miss. The miss sits on the codim-1 locus where the frozen head decides.
- **Lane over-represented 40.81×**, and its mode is **erasure into Road** — the campaign's signature
  failure (CLAUDE.md: *"the measured error = the LANE long-tail = ERASURE (not shift)"*).
- Per-frame Gini **0.2111** — temporally spread, not a few bad frames. Independently reproduces
  `hr3`'s per-frame mismatch Gini 0.211095.

### 4.4 Why the structure is adverse

The seg head is **exactly rank-4 linear** (MEASURED from frozen weights: singular values
[3.128, 2.154, 2.025, 1.796, **0**], rank-4 reconstruction error 5.96e-8 = fp32 floor). The whole
5-class partition is decided by ten class-pair hyperplanes — and **all four Lane normals are the four
largest (3.75–4.01 vs 2.60–2.95 for every non-Lane pair)**. Per unit perturbation, Lane boundaries
move most. The miss is concentrated **on boundaries** and **in Lane**: both the fragile locus and the
amplified class.

Converting the threshold into the measured structure:

- of the **676,210** boundary-adjacent misses, at most **7.79%** may flip;
- **17.0% of the Lane→Road erasures alone** would exhaust the entire budget.

**Counter-evidence in the route's favour, and it is real.** `msr1` measured **63.4%** of manufactured
pixels had a 17×17 token window exactly equal to GT — the renderer is a contextual CNN, so isolated
token errors need not change rendered RGB. SegNet re-segments the **frames**, not the tokens, so
token→flip transfer is genuinely < 1 and has never been measured. And `cb1`'s per-class split shows
generators are not uniformly bad: the MyCar mask **ADMITTED** and improved both axes.

**Counter-evidence against, and it is heavier.** `cb1`'s Lane band measured Δ`d_seg` **+0.0366** =
**149.1×** this route's entire headroom. `mf1` measured that even a **perfect oracle** seg repair
raised `d_pose` by 47× the seg gain it bought (Δ`d_seg` −4.13e-6 ⇒ ΔS_seg −0.000413; Δ`d_pose`
+6.66e-5 ⇒ ΔS_pose +0.019466), joint ΔS ≥ **+0.04300274393**. And probe **P-C** — the *"interiors
near-free"* go/no-go that gates the entire v8 paint stage — has **never been run**.

### 4.5 Against tri1's three properties

| property | born-small at HG1's price |
|---|---|
| 1. address implicit, not payload | **SATISFIED** — cells are entailed by generators; no address is ever coded. This is the property that makes the route worth testing at all, given tba1's +9.45 B and mf1's +35,969 B addressing losses. |
| 2. rate-bearing separate from frame-bearing | **NOT SATISFIED** — generators drive the field, which drives the renderer, which makes the frames. Impossible in the literal reading (§2.3); in the correct reading it is exactly what §4.2–4.4 leaves open. **This is the route's real risk.** |
| 3. movable mass structural | **SATISFIED** — generator parameters are few and continuous, movable by construction. |

Two of three. Property 2 is open and is the whole question.

---

## 5. Verdict and the deciding measurement

### 5.1 ALIVE, distinct, strongly disfavoured, correctly ranked third

The route is **not closed.** Generator-only distortion has never been measured on dx2 — HG1 inherited
distortion by identity and never fired a scorer, and its relaxed bracket is explicitly
`d_seg=UNMEASURED`. Its rate leg is the best measured on this vehicle. But every structural fact I
measured, and the one relevant scored prior, point against it.

### 5.2 The first decisive measurement — and it needs no build

> **Score HG1's already-materialized `generated_tokens.u8` through dx2's retained sections.**

`/Volumes/APDataStore/.../retained/generators/generated_tokens.u8` — 117,964,800 B, sha
`2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b` — exists now. dx2's
`source_semantic_renderer.bin` (30,856 B), `source_pose_carrier.bin` (22,010 B),
`source_compact_residual.bin` (96 B) are retained beside it. The field goes through the inherited
renderer and carrier to frames, then the frozen scorers.

- **Cost:** one advisory n600 row. No training, no new object, no Modal, no Metal.
- **Bar:** `d_seg ≤ 4.4682e-04` **and** `d_pose ≤ 1.0578e-04`, jointly `ΔS_distortion ≤ +0.024543`.
- **Falsifier:** either bound exceeded ⇒ the residual was an honest price on this generator roster,
  and born-small at HG1's vocabulary is closed at INSTANCE scope.
- **Conservative by construction:** the inherited carrier was fitted to dx2's *old* field, so it is
  mis-targeted here. If the container clears with a stale carrier it clears with a re-solved one.
  A refusal does **not** close the carrier-re-solve variant.
- **Report per class.** `cb1` already showed MyCar admits while Lane rejects; a composite verdict
  would destroy the one signal most likely to route the successor.

### 5.3 Ordering — and the two gates that are genuinely open

Born-small should fire **third**, and the reason is not a closed gate:

1. **tri1 rank 2 / E2′** — add pose to the objective, retrain the existing representation, re-measure
   ap1 `carrier_l1`. Decides basis-vs-objective (§2.4) and could make this whole route unnecessary.
   Cost: one training run + one advisory row.
2. **wq1 SPEC B (`ddm_tv1`, UNSPAWNED)** — reassign `k` random token positions, push through the real
   receiver, measure tolerance with **no addressing and no coding**. Fills `vf1`'s empty denominator
   and decides whether exact reproduction is over-strict. Cost: ~5 advisory rows, $0 local.
3. **This route's §5.2 measurement.**

**§5.2 does depend on the D3/tv1 question, and I name it as a gate.** But it depends on it *weakly*
and in a useful direction: §5.2 **is** a tolerance measurement — a single, structured, already-built
sample of it. If tv1 runs first it calibrates the prior; if §5.2 runs first it is one high-value
point on tv1's curve. They are complementary, not sequential.

**Honest limit on the tolerance thesis.** Relaxing exactness has so far bought single-digit
percentages: WS0 269,921 → 265,930 B (−1.5%), WS1 918,904 → 885,750 B (−3.6%), HG1's protected
bracket 460,408 → 420,646 B (−8.6%, still +240,428 B over the zero-distortion cap). Born-small does
not need tolerance to shave the residual — **it needs to eliminate 89.74% of it** (322,422 of
359,280 B). Only *dropping* the residual outright reaches the budget. That is why §5.2 is the test
and a relaxed-residual variant is not.

---

## 6. PRIOR-LAW PREDICTION — adjudicated, falsifier FIRED

> **Predicted:** born-small is genuinely distinct from trained-small **AND** substantial prior basis
> work exists that was never priced against the current demand — the route is alive but its assets
> are stranded rather than absent.
> **Falsifier:** recall shows born-small collapses to trained-small under nt1's reframe, **OR** the
> prior basis work was already priced against a demand of this size and refused.

**SPLIT: first clause CONFIRMED, second clause FALSIFIED. The falsifier's second branch FIRED.**

- Distinct: **CONFIRMED** (§2.1). Basis change ≠ point change; nt1's reframe does not touch it.
- Prior work exists: **CONFIRMED** (§1.3). SPEC_v8 derives it; real fitters and RAG/merge
  infrastructure are built and tested.
- *"Never priced against the current demand"*: **FALSE.** `et1` (535,761 B), `hg1` (460,408 B) and
  `hr3` (463,601 B) priced this exact basis against this exact demand on **2026-08-23 — the day
  before my charter** — receiver-closed with exact parse-back, and refused it. My framing of
  "stranded assets" was wrong. The assets are measured and byte-dead before distortion is charged.

I pinned this prediction to recall precisely so my own derivation could not decide it, and recall
went against me. What survives is narrower and better-grounded than what I predicted: the route is
alive **only** in the residual-dropped variant that those three FORMULATION-scoped closures
explicitly did not test, and its rate leg is stronger than I expected while its distortion leg is
worse.

---

## 7. NOT CLAIMED

- **No scorer ran.** `d_seg` and `d_pose` for the generator-only field are **UNMEASURED**. Every
  distortion number here is a budget, a bound, or a prior from another vehicle.
- **No claim that born-small clears sub-0.12.** §3 shows its rate leg clears with margin. §4 shows
  the distortion question is open and adversely structured. The two do not compose into a verdict.
- **No transfer of `cb1`'s Δ`d_seg` +0.0366 or Δ`d_pose` +22.7 to dx2.** Different vehicle,
  `verdict_scope=INSTANCE`. Cited as the most relevant measured prior, never as a dx2 number.
- **No transfer of the July v8 GT-object ledger** (91,611 / 210,255 / 111,135 B). Different object.
- **No family closure.** ET1/HG1/HR3 are `FORMULATION`; this memo adds no nonexistence claim. A
  learned implicit evaluator-cell carrier and curve-relative residual coding remain open — HG1 `:291`
  names them itself.
- **No claim the token→flip transfer is above or below 3.9484%.** It has never been measured. That
  is the point of §5.2.
- **`pointer_moved: false`.** No candidate archive, no promotion, no Modal, no Metal, no dispatch.

---

## 8. STORES CONSULTED

`.omx/research/` full-text queries for `born.small`, `laguerre|power diagram|bregman`,
`edge.centric|generator|DCB`, `unique.home|residual`, `ddm_tv1`, `curvelet|shearlet|self.orient`,
`cool.chic|C3|COIN`, `RC1|NR1|codebook`, `worldsheet` · `ddm_sy2_composition_synergy_deep_pass_20260823.md`
@ `fe2ba12dc2` · `ddm_tri1_triple_composition_and_pair_closure_20260824.md` @ `da6255c46a` ·
`ddm_nt1_trained_at_target_rate_20260824.md` @ `940059b42f` · `ddm_wq1_what_was_never_asked_20260824.md`
@ `1cc670031c` · `ddm_hg1_heterogeneous_analytic_generator_gate_20260823.md` @ `1eb31298ec` ·
`ddm_hr3_residual_implicit_carrier_20260823.md` · `ddm_et1_edge_topology_container_gate_20260823.md` ·
`ddm_rf1_renderer_film_rung_20260824.md` @ `310f2cd6aa` · `ddm_mf1_manufactured_seg_repair_20260823.md` ·
`ddm_tba1_token_bit_attribution_20260823.md` · `ddm_bl1_per_position_bit_allocation_20260822.md` ·
`ddm_vf1_evaluator_visible_floor_20260822.md` · `ddm_db1_decode_boundary_families_20260822.md` ·
`ddm_msr1_manufactured_seg_reduction_20260823.md` · `ddm_ap1_residue_purchase_scorer_20260823.md` ·
`ddm_ws0`/`ddm_ws1` · `ddm_cb1` · `ddm_rr9_reorder_refit_20260824.md` @ `3ebdee9657` ·
`ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange rate **CITED, not re-derived**) ·
`SPEC_v8_perclass_decomposition_20260708.md` · `recursive_fractal_optimal_representation_design_503_20260715.md` ·
`segnet_recursive_fractal_factorization_20260715.md` (rank-4 head, MEASURED) ·
`codex_findings_recursive_fractal_optimal_representation_v9_20260714_codex.md` ·
`v8_geometric_rate_decomposition_v1` (canonical equations) · `CLAUDE.md` · `AGENTS.md` ·
`docs/operating_manual_craft_handoff.md` · `.omx/state/main_hot_state.md` ·
sources `src/tac/boundary_math/*`, `experiments/ddm_{et1,hg1}_*.py`,
`tools/train_ddm_cl1_hpac_capacity.py` · retained payload
`/Volumes/APDataStore/pact/ddm_{dx2/r7,hg1_heterogeneous_analytic_generator_gate}/` ·
memories `[[m88]]`, `[[m53]]`, `[[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]`,
`[[perfect-localization-is-worthless-the-address-is-the-tax]]`,
`[[object-change-not-jointness-is-the-composition-law]]`.

**Receipts:** `/Volumes/APDataStore/pact/ddm_bs2_born_small/` — `bs2_generator_miss_structure.json`,
`bs2_per_frame_miss_counts.i64`, `SHA256SUMS.txt`. APDataStore 197 GiB free; **Vertigo at 100% was
neither read nor written.**

---

`dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]` — gap to 0.12 = 0.028220 ⇒ shed
42,382 B at fixed distortion, or 150 B at zero distortion.
