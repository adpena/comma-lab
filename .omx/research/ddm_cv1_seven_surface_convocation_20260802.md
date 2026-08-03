# ddm_cv1 — the seven-surface convocation (codebook · menu · solver · resolver · selector · realization · BASE)

- arm: MAIN (no subagent; corpus-built, not recalled)
- date: 2026-08-02
- operator scope: the standing convocation question, EXTENDED by operator 2026-08-02
  verbatim: *"Base is within the convocations domain as well."*
- method: `tools/corpus_query.py` over 9,704 research docs + the task ledger. Every prior
  verdict's SCOPE re-checked at its source artifact rather than recalled from working memory
  (operator standing directive *"Don't recall from working memory"*).
- axis label: all numbers below `[macOS-CPU advisory exact n600]` unless marked otherwise.
  NOT contest-CPU authority.

---

## §0 The arithmetic that disciplines every row

Gap to the PR130 demonstrated floor (seg 0.029660 / pose 0.015268 / rate 0.127214 =
bar 0.172141), computed from the live `dc1_fold` row 0.8983775:

| axis | our value | floor | **gap** | share |
|---|---:|---:|---:|---:|
| seg  | 0.4311790 | 0.029660 | **0.4015190** | **55.3%** |
| pose | 0.2272835 | 0.015268 | **0.2120155** | 29.2% |
| rate | 0.2399150 | 0.127214 | **0.1127010** | 15.5% |
| | | | **0.7262365** | |

**The load-bearing fact:** seg is CONSTANT to 7 dp across v4d (0.9639878) → pw1 (0.9476092)
→ ms8 (0.8984335) → dc1_fold (0.8983775). **Four consecutive pointer moves, all pose or
rate, ZERO seg movement.** This is what justifies the operator's addition of BASE.

Denominator honesty: **1% of the gap = 10,908 B of rate**, or 7.26e-5 of d_seg. Any ΔS
quoted without this denominator is unanchored.

*(CORRECTED same-session: I first wrote 11,892 B here and in MEMORY.md. The registered
equation `gap_decomposition_against_floor_20260802.bytes_per_percent_of_gap` returned
10,908 B on its first run and is right —* `0.007263 × 37,545,489 / 25` *. Pinned as a
regression test. Making the denominator executable caught its author's own arithmetic
within minutes, which is the entire argument for registering it.)*

Table caveat: this table is computed from PR130's ROUNDED published contributions; the
equation, fed PR130's measured **190,952 B**, returns total gap **0.7263025** and rate gap
**0.1127679**. The equation's figures supersede the table's — it derives the rate term from
bytes rather than inheriting a rounded value.

---

## §1 BASE — 55.3% of the residual, and the surface no post-base move has touched

### Measured (re-checked at source, not recalled)

| receipt | measurement |
|---|---|
| `ddm_cr2r` matched control | same solver, 74 matched pairs: ep854 d_pose **11.5904** vs celldrop50 **2.5308**; ep854 better on **1 of 74**; **46× over** at the floor |
| `ddm_uv1_ep854_pose_illegibility_reject` | ep854 REJECTED; `gr1 cell_drop50` (the base the pose WAS solved against) reproduces at **0.000709** — PASS in-lineage |
| `#889` | measured law: **seg-only training SPENDS pose legibility**; the −0.0866 S composition stranded on it |
| `bp2` (this session) | **d_pose is RELATIVE between the two delivered frames.** True-GT frame_0 spliced into our pair ⇒ d_pose **3.05–16.66** vs the decoded pair's **0.0008**. Four orders WORSE for being MORE correct. |

### The mechanism, stated once

`d_pose = MSE(PoseNet(our_pair)[:6], PoseNet(gt_pair)[:6])`. PoseNet reads a **pair** and
returns the ego-motion **between** its two frames. Our delivered frames are jointly
consistent (same carrier, same warp), so the *relation* matches GT even though neither
frame individually resembles its GT counterpart.

**Therefore: seg is a per-frame functional; pose is a PAIR functional.** A sum of
per-frame objectives is structurally blind to the relation the pose scorer reads.

### Scope re-check of the prior negative

The measured-DEAD verdict on joint pose is scoped to **post-hoc / stored correction
families** (5 formulations, photometric wall — frames trained on seg alone carry no
pose-legible photometric signal). It is **NOT** scoped to a base trained under a
**pair-coherence term**. That family has never been built or measured. **OPEN.**

### The unexhausted move

Not "train a better seg base, then re-solve pose." A **pair-coherence loss term** binding
the two frames as one carrier under one warp — which is exactly why `f0 := a·warp(f1)+b`
works at 194 B. **No burn config in the campaign has ever carried this term.**

---

## §2 RESOLVER — the gate between BASE and the pointer

| receipt | measurement |
|---|---|
| `ddm_sf1` | STALE-FIT GENUS swept (15 quantities) + structurally fixed. Law: *any SOLVED or FITTED quantity has a PARTNER it was solved against.* |
| `ddm_uv1` | capability limit at source: `BASES` was a **hardcoded 2-entry dict** — the solver structurally could not resolve against an arbitrary base. Fixed via `resolve_base()` + `--base-archive`. |
| `ddm_sf1` | `mq1_joint_pose_refine_emit.py` refines `{p0,p1,p2,beta}` and **never** re-solves `(a,b)` ⇒ CARRY, not STAMP |
| `ddm_v4d_resolve.py` | step 2 re-fits `(a,b)` at refined dim0; step 3 selects beta FROZEN |
| `#891` | fiber-transport the stale `(a,b)`: **1 extra FD for ~185 pairs** instead of full re-solve, + a $0 clip-fraction falsifier that may retire most of them |

**The unasked counting question (backcast):** how many seg wins were silently discarded
over the past month because the resolver could not be pointed at their base? The `BASES`
2-entry dict is a *capability* limit, so this is answerable from the corpus.

---

## §3 REALIZATION — a seg-neutral gauge orbit, spent once out of three possible times

### Measured exactly (`src/tac/optimization/ddm_ll1_window_solve.py`)

`F.interpolate(x,(384,512),'bilinear')` with torch defaults `align_corners=False`,
**`antialias=False`**. Strides `874/384 = 2.2760` and `1164/512 = 2.2734`, **both > 2** ⇒
consecutive 2×2 read-windows **cannot overlap**. Reads-per-camera-pixel has exactly two bins:

- **0×: 230,904 px (22.70%)** — read by NEITHER scorer
- **1×: 786,432 px** = 196,608 scorer px × 4, **EXACTLY**

Each scorer pixel owns a **private, disjoint** 2×2 window ⇒ the uint8 preimage problem
**decouples exactly** into 196,608 independent 4-variable problems. Geometry verified
against torch fp64 at **6.45e-12**. PoseNet reads through the IDENTICAL operator
(`preprocess_input` interpolates BEFORE `rgb_to_yuv6`) ⇒ same windows, same blind set.

Solve-don't-search, measured: window solve **0.07 s/frame rms 0.0282** vs enumeration
**2.04 s/frame rms 0.0715** — the search is both slower AND worse. Realized flips
**88 → 3**; ΔS **−0.0144** at n=3.

### The question the corpus posed and nobody answered (`SPEC_g17`)

> *"For a fixed received semantic partition, many RGB/chroma/luma realizations lie in the
> same frozen-SegNet argmax cell after R. They are not equivalent for PoseNet or code length."*

**This is a THIRD CURRENCY:** a gauge orbit that is **seg-neutral by construction** and
**pose-active and rate-active**. Exploited once (ll1, for seg). **Zero times for pose or
rate**, against 29.2% + 15.5% of the gap.

Compounding fact: the **22.70% blind set is invisible to both scorers but IS READ by
v4d's frame_0 warp** ⇒ a pose channel that costs zero seg by construction. `bp2` is the
first direct n600 test.

---

## §4 MENU — a validated triage rule against a 79-of-84 unmeasured denominator

| receipt | measurement |
|---|---|
| `ddm_bs2` | **84 discrete choice points** over 10 live-chain files: 30 discrete menus + 12 boolean flags + 26 all-or-nothing accept/reject rules. **5 measured.** |
| `ddm_pw1` | the discriminator: occupancy piled **AT A BOUND** ⇒ freeing pays. Measured 0.9639878 → 0.9476091. |
| `FEED-pb2` | the counterexample that SHARPENS it: a menu that is a **DIRECTION** rather than a bound got **WORSE** when freed. "Free the menu" is NOT automatically an improvement. |
| `ddm_dc1` | swept 8 menus for the dead-codeword defect: **exactly one** (`st_grid`) over the 10% threshold, and its `gap_lattice` is **identically zero** |

### Verdict-scope correction (load-bearing, MAIN's own claim corrected by dc1)

`t = s_t·[p2,p1,p0]` with `s_r=0 ⇒ R=I`, so scaling `s_t` by *c* and the pose triple by
*1/c* gives a **bit-identical** homography (max rel diff **4.539e-16**, n600, two
independent derivations). `s_t` is **exactly multiplicatively degenerate** with a
continuous coordinate that already ships.

⇒ ms8's −0.049 sits in mq1's **SEARCH** bucket, not FORMAT; mq1's "format ≤0.056%" ceiling
was never contradicted. **"placement beats selection 5.7×" is an `s_t` INSTANCE fact from
the UNIQUE degenerate menu and does NOT generalize.** The real discriminator is
**DEGENERACY**, not dead-codeword fraction.

**Move:** 79 unmeasured points; two-case triage rule (bound vs direction) validated on
both a positive and a negative; **the triage itself needs no scorer.**

---

## §5 CODEBOOK — closed by degeneracy on one axis, entirely unraced on the axis that matters

`ms8` won on ST_GRID dead-codeword refit; `dc1` proved that axis closed **by degeneracy**,
not exhaustion. But **tokens are 99.0% of 504,736 B** — the actual rate axis — and their
codebook family has **never been raced**: QA13 VQ-codebook · QA12 token-LOTTO · QA08
context-MIX · QA09 Cl(2), all sitting unfired in the exporter round-2 pool.

Two sharp, cheap, unanswered questions already in the corpus:

1. **Chou–Lookabaugh–Gray (ECVQ), `ddm_gc6` seat T4:** *"fixed lattice + entropy coder vs
   entropy-constrained codebook at the 0.004-distortion operating point — which side of the
   crossover are we on?"* We sit at d_seg **0.00431179**. One measurement answers it.
2. **The rule-118 FREE codebook** (`gaussian_quant_2512.06609` deepdive): a codebook
   `~N(0, I_m)` generated **from a seed** costs **ZERO archive bytes** — only the token
   **indices** are counted. On a 99%-of-bytes token stream this is not a marginal lever.

---

## §6 SELECTOR — 535 B of *program*, the highest rule-118 leverage per byte

TR1 ships 2 ZIP members / 4 sections; the **selector is 535 B and IS the decode program**.
Smallest section, only one with program semantics ⇒ improvements are paid in **free
interpreter work**, not counted bytes.

The historical selector tension was recorded as *"selector weights compete with mask
bytes"* (`charged_mask_grammar_ego_foveation_greenup_20260502`). At **535 B against
504,736 B of tokens that tension has INVERTED**, and the selector has not been re-derived
under the new ratio.

---

## §7 SOLVER — a measured, unclaimed, free win

| receipt | measurement |
|---|---|
| `#850` | the pose GN solve is **hard-capped at 2–3 relinearizations with NO convergence test**, and is still descending **13–23% per iteration when it stops** |
| `#897` / `ll1` | the REALIZATION solver is closed (88→3 flips, ΔS −0.0144 n=3). This is a different solver from the pose one — do not conflate. |
| `pantheon_synergy_crux_synthesis` | trust-region radius: **"any fixed radius"** named as the cargo-cult; **Fisher-ball radius from local margin curvature + adaptive LM-λ** named as the derived form. **NOT BUILT.** |
| `ddm_fd1` / `ddm_fd2` | zero-accept confound classified `BLOCK_LOCALITY_OR_REALIZATION_GAP` — a **diagnosis, never cured** |

We are stopping a converging solver because of a hardcoded count, on an axis worth 0.2120.

---

## §8 What the operator is not asking about, that they should

1. **BASE and RESOLVER are a COUPLING, not a sequence.** No resolver recovers legibility a
   base never had. The ask is a pair-coherence **loss term**, not a better re-solve.
2. **The realization gauge as a SPENDING currency** — seg-neutral by construction,
   pose- and rate-active; two of three currencies never spent.
3. ~~**The class decomposition of the LIVE seg gap** has never been done.~~ **CORRECTED
   SAME-SESSION — I over-stated this; the corpus partially refutes it.** `ddm_fl1` DID
   perform the join, at the **ep641/r1c** endpoint (`ddm_xp1_20260731/xp1_verdict.json`),
   and already named "residual-above-floor = Undriv/Movable". What is genuinely missing is
   only the **re-join on the LIVE base** (d_seg 0.00431179 vs ep641's 0.004264052 — a
   different endpoint; fl1 itself flags cross-endpoint use as *labeled structure transfer,
   not a number transfer*). See §11 for the ratio table the existing join implies, which
   had not been computed.
4. **Denominators, everywhere.** 79 of 84 menu points unmeasured · 502 of 502 preflight
   gates skipped on a normal commit (#842) · 31 of 31 capped files unable to report why
   they stopped (#874). Today `grep -rilE` over `.omx/research` **timed out at 2 minutes** —
   the corpus has outgrown its own default instrument. Fifth "vacuity == pass" instance of
   the day.
5. **The one to bet on.** Every seg improvement we have produced has been pose-hostile, and
   four consecutive pointer moves avoided seg entirely. That is not a coincidence to route
   around — it is the campaign's central measured fact, with one honest reading: **the base
   objective is missing the term the pose scorer actually reads.**

---

## §9 Verdict-scope ladder for every claim in this memo

| claim | scope | basis |
|---|---|---|
| seg constant across 4 pointer moves | **MEASURED**, INSTANCE(this lineage) | 4 byte-closed n600 rows |
| seg-only training spends pose legibility | **MEASURED**, FORMULATION | cr2r matched control (74 pairs) + uv1 + #889 |
| d_pose is relative between delivered frames | **MEASURED** INSTANCE, **FORMULATION-general** | bp2, 4 disjoint splits, sign-robust; mechanism is a property of the frozen scorer's pair input |
| D is disjoint 2×2 sampling | **MEASURED EXACT** | fp64 parity 6.45e-12; histogram has exactly 2 bins |
| `s_t` multiplicative degeneracy | **MEASURED EXACT** | rel diff 4.539e-16 n600, 2 independent derivations |
| "placement beats selection 5.7×" | **INSTANCE(`s_t`) ONLY** — does NOT generalize | dc1 correction of MAIN's own ms8 framing |
| pair-coherence base term | **UNBUILT — CONJECTURE** | derived from the two MEASURED rows above; no receipt |
| free seed codebook / ECVQ crossover | **UNMEASURED** | corpus-posed, never raced |
| pose GN truncation | **MEASURED** | #850 |

---

## §10 Named next measurements (ranked by gap-share × cost-to-falsify)

1. **Class decomposition of the live 0.00431179 vs `fl1` per-class floors** — $0 given
   cached lstars; converts 55.3% from a number into a mechanism. **No scorer slot needed
   beyond one cached join.**
2. **Menu triage over the 79 unmeasured points** — scorer-free triage (occupancy from
   existing receipts), bound-vs-direction rule already validated both ways.
3. **Pose GN convergence test** (#850) — remove the hardcoded relin cap, add a real
   convergence criterion; re-solve. Axis worth 0.2120.
4. **Realization gauge spent for POSE** — `bp2`'s blind-set actuator is rung 1 of this.
5. **Pair-coherence base term** — the largest and the most speculative; design first,
   falsifier pre-registered against the cr2r matched control (74 pairs, same solver).

Cross-refs: `ddm_dc1_menu_sweep_and_ms8_mq1_reconciliation_20260802` ·
`ddm_ms8_menu_selector_solver_st_codebook_20260802` · `ddm_uv1_ep854_pose_illegibility_reject_20260802` ·
`ddm_sf1_stale_fit_genus_sweep_and_structural_fix_20260802` · `ddm_bs2_lane_guard_schedule_and_binary_occupancy_sweep_20260801` ·
`ddm_fl1` · `src/tac/optimization/ddm_ll1_window_solve.py` · `SPEC_g17_unified_production_envelope_20260726`

---

## §11 The residual/floor RATIO table — computed here, not previously stated

Joining `ddm_xp1` (ep641 r1c per-class residual, S-units) against `ddm_fl1` (per-class
GT-flicker floor, S-units). Both vectors already existed; the RATIO had not been taken.

| class | residual (ep641) | flicker floor | ratio | reading |
|---|---:|---:|---:|---|
| **Road** | 0.18845 | 0.1889 | **1.00** | sitting EXACTLY on its floor |
| Lane | 0.12589 | 0.2316 | 0.54 | already BELOW floor (floor pierced) |
| **Undriv** | 0.05574 | 0.0394 | **1.41** | ABOVE floor — ours to take |
| **Movable** | 0.03792 | 0.0285 | **1.33** | ABOVE floor — ours to take |
| MyCar | 0.01840 | 0.0434 | 0.42 | already BELOW floor |
| **total** | **0.42640** | 0.53180 | 0.80 | aggregate already pierces |

**What this changes about "seg is 55.3%".** The seg residual is NOT one homogeneous
attackable mass:

- **Road alone is 44% of the seg residual and sits at ratio 1.00** — exactly at its
  GT-flicker floor. Attacking Road means piercing a floor, not closing a gap.
- **Lane and MyCar are already BELOW their floors** (0.54, 0.42) — 34% of the residual is
  in classes where we have already beaten the smooth-label reference.
- **Only Undriv + Movable sit above floor** (1.41, 1.33), and together they are
  **0.09366 S = 22% of the seg residual**, or **12.9% of the total remaining gap**.

**Scope, stated honestly.** The floor is **FORMULATION-scoped, not a hard bound** — fl1
records it as pierced by phase-faithful PR130 (2.966e-4) and by ep641 itself in aggregate
(0.004264 total < 0.005318 total floor). So ratio ≥ 1.00 does not mean "unreachable"; it
means "reaching it requires phase-faithfulness the smooth-label reference does not have."
The floor's live value is a **RANKING of phase-faithfulness debt**, which is exactly how
fl1 labelled it (binding debt: Lane #1 at 13.1× corner-C).

**The correction this forces to §1.** BASE remains 55.3% of the gap and remains the
surface no post-base move has touched. But "attack seg" is now two different asks:
(a) the **22% above-floor** portion (Undriv/Movable) is ordinary optimization; (b) the
**44% Road-at-floor** portion is a phase-faithfulness problem, which is the SAME
pair-coherence axis §1 identified from the pose side. Two independent lines of evidence —
the pose-legibility law and the Road-at-floor ratio — point at the same missing property.

**Owed:** the same join on the LIVE base (needs one per-class argmax pass; scorer slot
currently held). Cross-endpoint transfer is LABELED STRUCTURE, never a number.

---

## §12 THE THIRD CURRENCY — first n600 verdict (ddm_bp2, same session)

§3 conjectured the realization gauge as a **third currency** (seg-neutral by construction,
pose- and rate-active) and named the 22.70% blind set as the first testable instance.
`ddm_bp2` ran it to n600. **The mechanism is CONFIRMED; the economics are REFUTED.**

| leg | prediction | MEASURED n600 | verdict |
|---|---|---|---|
| seg-neutral **by construction** | d_seg unchanged | **bit-identical on 600/600 pairs** under a full ±1 LSB gradient-sign step over all 692,712 blind coordinates | **CONFIRMED** |
| F1 — disjoint from the frame_0 warp read-set | <5% overlap closes the family | **14.80%** | REFUTED |
| F2 — achievable pose reach | <1e-4 closes the family | **1.7158** (4.2 orders over) | REFUTED |
| does it PAY? | — | NET ΔS **positive at every arm**: 17.8× at the cheapest, **3.89× at the best** (per-pair argmin, which cuts d_pose 65.9%) | **DOES NOT PAY** |

Reach is enormous and **direction-selective**: d_pose 0.00764 → 0.889/1.179 (116×/154×),
while a random-sign step at the *same coordinate count* is inert (+0.61%). So this is a
genuine actuator, not noise — it is priced out, not absent.

**Correction to §3, stated plainly.** The sentence "a pose channel that costs zero seg by
construction" is MEASURED TRUE. The implied "therefore spendable" is MEASURED FALSE at
per-coordinate pricing. The third-currency framing survives and is now sharper: the gauge
orbit is real and large, and the binding constraint is **description cost per coordinate**,
not reachability. **FORMULATION scope, not family** — the untried shape is a **PARAMETRIC**
blind-set perturbation (k coordinates + signs generated from a few shipped scalars), which
is the only form that can beat the per-coordinate arithmetic. Also untried: Gauss-Seidel
sign re-solve (only Jacobi was refuted), and the same actuator on a vehicle with materially
larger d_pose.

### The methodological finding — worth more than the verdict

bp2's own §6 prefix conclusion was **overturned by its own n600**. On a video-order prefix
(n=73–181) a free receiver-computable index looked like a **−0.122 S WIN**; at n600 the same
arm is a **+0.152 S LOSS**, and the best arm lands at NET **+0.00001 S** — indistinguishable
from zero.

**Cause, measured:** the prefix's mean d_pose was **0.0390** against the population's
**0.0076425** — **5.1× harder** — on a distribution whose median is **8.2e-4**. A
video-order prefix of a heavily skewed population is not a small sample of it; it is a
sample of a different, harder population, and it **flattered the family**.

This is the measured instance of the SCOPE-censoring class (task #875): a subset default
that silently under-samples the verdict. It generalizes past this arm — any prefix-scoped
verdict on a skewed per-pair quantity inherits the same defect, and the campaign has many.

*(Registered separately as a memory; #875 gains its first measured anchor.)*

**Custody:** `604f7180b3` (module + 26 tests + tool), `378ff3fba9` (n600 memo + receipt),
8 receipts under `reports/ddm_bp2/`. Guards: fast-path == authority on 600/600; gradient
surrogate within 1e-5 on 518/600 (max deviation 5.46e-04 — used only as a search
direction, never as authority). Pointer UNMOVED. `[macOS-CPU advisory]`, `score_claim=false`.
