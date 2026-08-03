# ddm_na1 — negative-results audit under four lenses (operator 2026-08-02 "Audit all negative results")

`date_utc: 2026-08-02` · arm `ddm_na1` · axis `[macOS-CPU advisory / $0 corpus+code audit]` ·
`score_claim: false` · `promotion_eligible: false` · **0 scorer forwards, 0 launches, 0 n600 jobs**
(another arm holds the slot). **Exact pointer UNMOVED.** This audit moved no number and does not
claim to.

**Operator scope, verbatim, three parts:**
1. *"Audit all negative results."*
2. *"don't judge moves that produce significant wins on segment and rate if they hurt pose because
   pose can be fixed later using techniques and methods that we already know."*
3. *"Review all of the formulation and instant scopes that produce negative results. Each of those
   suggested more optimal follow ons that we must pursue is p zero, including parametric blind set
   perturbation."* · *"Rate and SEG are extremely important, and pose falls out if everything else
   is done the right way in the right order."*

**Method:** four lenses, three of which did not exist at the prior negative-audits
(#390/#440/#630/#755/#498, and `ddm_ub1` 08-01). Lens 1 run by this arm; lenses 2–4 by three
read-only sweep agents over disjoint query surfaces. Every number below was read from a named
artifact or re-derived here; nothing is recalled.

---

## §0 THE HEADLINE, AND A BLOCKER THAT OUTRANKS EVERY ROW BELOW

**The single largest live re-open is the `ep854 × cell_drop50` composition: `seg+rate = −0.0865743 S`,
MEASURED n600, currently stranded behind a pose debt.** Under the new acceptance rule that is not a
rejection — it is **16.84 % of the seg+rate decision gap**, on a live shippable vehicle, and its
repair path is named in the corpus and **has never been built**.

**But first, the blocker.** The follow-ons this audit is asked to mark P0 are being written against
a task-ID namespace **the canonical ledger does not contain**.

**MEASURED, exhaustive over `.omx/state/canonical_task_status.jsonl` (409 rows, 148 distinct ids,
42 numeric):** in the 840–905 band the ledger holds **only** `850, 871, 873, 882`.

| id | in canonical ledger? |
|---|---|
| #827 · #850 · #871 · #882 | **PRESENT** |
| #846 · #862 · #865 · #875 · #881 · #888 · #889 · #894 · #896 · #900 | **ABSENT** |

**10 of the 14 checked ids — including 7 of the 7 this audit was instructed to sweep — do not exist
as ledger rows.** They exist only in memo prose and as bare `ref:#NNN` entity nodes in
`.omx/state/graph_memory/nodes.jsonl` with empty `summary`/`title` (i.e. scraped from prose, not
written by a ledger). `ddm_cv1`'s own graph note already records that `#900` is absent and that
`ddm_pb2` had to land its charter measurement under `#871` instead.

**Cross-instrument confirmation (independent of my parser):** `tools/corpus_query.py --stores tasks`
(which reports `tasks(409)` — the same denominator) was queried for `#889` and returned
`827 / 603 / 816 / 818 / 871`, never `889`. Two instruments reading the store by different means
agree.

*Scope of this negative-existence claim: the canonical task ledger only. A wider filesystem sweep was
launched and timed out; I do not claim these ids exist nowhere.*

**Consequence:** a P0 follow-on named against `#894` has no row to fire from. This is the
DEFER-AT-SOURCE law failing at the ledger, not at the memo. **Every follow-on in §6 must be given a
ledger row before it can be scheduled**; otherwise this audit produces text, not action.

### Denominators used for every ranking below (re-derived, not inherited)

| axis | live (`dc1_fold`) | PR130 floor | gap | share |
|---|---:|---:|---:|---:|
| seg | 0.4311790 | 0.029660 | **0.4015190** | 55.29 % |
| pose | 0.2272835 | 0.015268 | 0.2120155 | 29.19 % |
| rate | 0.2399150 | 0.127214 | 0.1127013 | 15.52 % |
| **total** | | | **0.7262358** | |

**DECISION gap (seg + rate, per operator directive 3) = 0.5142203 S = 70.81 % of the total.**
1 S unit = 1,501,820 archive bytes. 1 % of the total gap = 10,907 B.

**Correction to a load-bearing denominator (MEASURED).** PR130's byte count is **191,052 B**, not
190,952 B. Recomputing from components: `100(2.9660e-4) + √(10·2.331e-5) + 25·191052/37545489 =
0.172141` — reproduces the published row **exactly**. `MEMORY.md` and `ddm_cv1` §0 both carry
**190,952**, a digit transposition, which understates the floor's rate term by `25·100/37545489 =
6.66e-5 S` and inflates the computed gap by the same amount (cv1's equation returned 0.7263025;
the correct value is 0.7262358). Immaterial to any decision; it is a transcription error in the one
number every ranking divides by, and it propagated into MEMORY. Source: `pr86_pr130_fullstack_intake_20260728.md`.

---

## §1 LENS 1 — PREFIX-SKEW (this arm)

### 1a. The lens is not new. It was minted TWICE and the instrument was never fixed.

**MEASURED at source, `.omx/research/adversarial_review_round1_measurement_20260630T003333Z.md`
(2026-06-30), finding #1 of 2 CRITICALs:**

> *"**[CRITICAL] Contiguous-prefix subsets are systematically EASY → every n24/n96 headline is
> OPTIMISTIC vs n600.** `decode_gt_frame1_pairs` yields the FIRST n contiguous pairs (start of the
> clip), not a representative sample. MEASURED swing in the SAME quantity (R0-flat d_seg_bulk):
> **0.00512 @n24 vs 0.0242 @n96 = 4.7×** … This is the dominant 'the numbers are an artifact' risk
> for this lens."*

Eleven days **earlier**, `.omx/research/recursive_adversarial_review_recent_negatives_20260619T024605Z.md`:

> *"The curve / flat-NCA / texture-NCA / polynomial gates all measure on the **first 3 frames**…
> **n=3 here is effectively n≈1**: not 3 independent samples… Cheap fix for any future GREEN-seeking
> gate: sample non-consecutive frames spanning ≥2 videos."*

**MEASURED at source today, `src/tac/boundary_math/seg_core.py:80-105`:** `decode_gt_frame1_pairs`
still decodes from the container head and breaks on `yielded >= n_pairs`. **No stride. No
representative sampler. 44 days after the first mint, 33 after the CRITICAL.** The proposed cure
(`gt_strided_n200.npz`) did not reach the shared sampler.

**Blast radius, MEASURED:** `rg -l decode_gt_frame1_pairs --glob '*.py'` → **28 files**. Defaults
read from argparse at source:

| gate | default sample |
|---|---|
| `probe_nca_dseg_feasibility_gate.py:426` | `--n-frames 3` |
| `probe_curve_core_dseg_feasibility_gate.py:573` | `--n-frames 3` |
| `probe_nca_texture_dseg_feasibility_gate.py:306` | `--n-frames 3` |
| `probe_polynomial_fill_survival_gate.py:349` | `--n-frames 3` |
| `probe_yousfi_road_lane_geometric_solve.py:765` | `--n-pairs 4`, `--survival-pairs 2` |
| `sg_drf_single_frame_feasibility_probe.py:415` | `--pair-idx 0` (**one** pair) |

An entire family of surfaces named `*_feasibility_gate` / `*_survival_gate` — surfaces whose only
product is a GO/NO-GO verdict — samples 1–4 contiguous clip-start pairs.

### 1b. NEW LAW (DERIVED, with a measured anchor) — the direction is not uniform

Prior treatments assumed one direction. The 06-19 review argued prefix bias makes REDs *more*
trustworthy (*"a wall on easy near-identical frames only worsens on diverse scenes"*). **That is
correct, and it is only half the law:**

> **Prefix bias is CONSERVATIVE for an absolute-threshold WALL verdict** (candidate fails to reach a
> fixed target on the easy sample ⇒ it fails worse on the population) **but UNCONTROLLED for a
> COMPARATIVE verdict between two candidates** (the bias is a per-archive quantity and does not
> cancel in the difference).

**The measured anchor for the differential**, both rows from the same harness, same session
(`ddm_gr1_granularity_rerace_20260730.md` §correctness-gates and `ja1` atlas 9 via `ddm_ba31`):

| archive | d_seg n48 | d_seg n600 | prefix bias |
|---|---:|---:|---:|
| gr1 baseline / REF | 0.0038626 | 0.0038892 | **+0.69 %** |
| gr1 `cell_drop50` | 0.003947 | 0.004310379 | **+9.21 %** |

**The bias differs by 13× between two archives measured by the same instrument.** In d_seg the
differential is `0.0003368`; **as a screening threshold, any n48-based comparative verdict whose
claimed effect is below ≈ 0.034 S is not load-bearing.**

### 1c. HONEST NON-REACTIVATION — gr1 verdict 3 survives this lens

`ddm_ub1` graded gr1 verdict 3 (*"QA07 nested-rung DOMINATED… No middle ground pays"*) **INSTANCE**
on allocation grounds. I attacked it independently on measurement grounds and **failed to break it.**

Bracketing `cell_rung_a` (n48 only, 354,946 B @ 0.004681) against `cell_drop50` at n600 under both
observed bias magnitudes:

| assumption for rung_a's bias | rung_a seg+rate vs REF(n600) | drop50 seg+rate vs REF(n600) |
|---|---:|---:|
| carries REF's +0.69 % | −0.0608 | **−0.0982** |
| carries drop50's +9.21 % | −0.0209 | **−0.0982** |

`drop50` wins under **both** bracketings. The gap narrows (published 0.071 → 0.037–0.077) but does
not flip. **gr1 verdict 3's ORDERING stands.** ub1's INSTANCE grade stands on its own separate
grounds (a self-declared poor proxy used to allocate the rung arm); the two are independent and only
one of them breaks the verdict.

### 1d. WHAT DOES break — the KNEE LOCATION, and it selected the live base

The same table's knee determination is a three-point comparison:

| candidate | archive B | seg+rate vs REF (n48) |
|---|---:|---:|
| `cell_drop35` | 439,836 | −0.085 |
| **`cell_drop50`** | **359,221** | **−0.132  ← selected as the knee; the LIVE base** |
| `cell_drop63` | 277,815 | −0.080 |

Spacings **0.047** and **0.052 S** against a **measured 0.0337 S** prefix-bias differential:
**1.39× and 1.54×**. `ba31` independently called this *"a flat-bottomed minimum sampled at three
points."* And `drop50`'s own n600 correction shrank its win from **−0.132 → −0.0982 (−26 %)**;
`drop35` and `drop63` have **no n600 measurement at all**.

**GRADE: the knee LOCATION is INSTANCE-scoped and not resolved by the measurement that chose it.**
Every subsequent row in the campaign is built on `cell_drop50`.

**RESOLVING MEASUREMENT (named, cheap):** n600 realized d_seg for `cell_drop35` and `cell_drop63`
through the existing `experiments/ddm_gr1_granularity_rerace.py` harness (`drop50` already has its
n600 row). **No training. Two scorer renders.** Falsifier: if either non-selected point's n600
seg+rate beats `drop50`'s −0.0982, the live base is mis-placed and every downstream row inherits it.

---

## §2 LENS 2 — FORMULATION-SCOPED FLOORS CONSUMED AS HARD BOUNDS

`ddm_fl1` itself is **clean** — verified verbatim: *"this floor is FORMULATION-scoped… and is
pierced by phase-faithful renderers"* and *"Citing 0.005318 as a HARD d_seg floor is a FORBIDDEN
cargo-cult claim."* The defect is downstream, exactly as ub1's law predicts.

| # | floor | scope at mint | how cited now | pierced? | grade |
|---|---|---|---|---|---|
| **F1** | fl1 above-floor pool **0.0258 S** used as the seg **runway bound** | FORMULATION | `ddm_gc14_first_descent_20260731.md:93,194,250` — prose keeps the scope, **the default branch does not** | **YES, by gc14's own numbers** | **OVER-SCOPED at the decision layer** |
| **F2** | fp1 receiver floor **0.008305** = *"the **FUNDAMENTAL, head-independent** bound"* | FORMULATION (flat-prototype-paint), stated `:60-61` | escalated to "FUNDAMENTAL" **inside the same document** `:99-100`; hardens to *"the fp1 **wall** that routes the carrier tier"* (`ddm_burn4_charter_skeleton:31`) | **YES — 0.00494 / 0.0049411, 1.68× below, stated in the same doc `:78`** | **OVER-SCOPED (one word)** |
| **F3** | rate axis "DEAD" at the i.i.d. entropy floor | FORMULATION | `main_frontier_lineage_and_roadmap_crossref_20260801.md:61` *"Plane-storage family measured RATE-DEAD"*, untagged, family-worded, **on the live roadmap** | YES (PR130 rate 0.1272 vs our 0.2399) | **confirms ub1's UB1-B; not re-minted** |
| **F4** | ja1 `saturated_do_not_spend` (3 pools) | FORMULATION, *"at THIS base (v4c)"* | `gc16:472` *"Do not spend an arm here"* — **v4c base scope not restated** | no | **MILD over-scope / thin denominator (3 RD points)** |

### F1 is the one that is deciding something today

`ddm_gc14:250` — *"No further continuation windows. The seg axis is handed to the terminal/solve
family and **the slot goes to the rate axis**… This is the default branch on current arithmetic and
I expect it to fire."* The prior that sets that default is a **FORMULATION-scoped pool**. gc14's own
numbers: ep641 → ep834 consumed **−0.0279 S against a 0.0258 S pool** — **over-drawn ~8 % with 112
epochs of the window still to run.** Under its own drain accounting the runway should already be
exhausted; it is not.

**Caveat welded on:** ep834 is the 36-pair `a1_gate` estimator, which
`ddm_cf1_coarse_framing_audit_20260731.md` flags as a **NEVER-DECIDED estimand** — and note this is
a Lens-1 case too (a 36-pair subset). **This is a candidate pierce, not a pierce.**

**RESOLVING MEASUREMENT:** the per-class n600 join on the **live** base, already listed as *"Owed"*
in `ddm_cv1` §11 (`$0`-ish, cached lstars, one argmax pass). It settles F1 and simultaneously
converts the 55.3 % seg share from a number into a mechanism. **This is the highest-value unclaimed
measurement found by any lens.**

### Floors attacked and NOT broken (honest survivors)

- **FLOOR-384 = 1.875e-4** — family-scoped and refuses the hard read itself (*"NOT an absolute floor
  that blocks all decoders"*, verdict CAPACITY-LIMITED, 11–13× headroom). **SOUND.**
- **`S_floor = 0.11797`** — mint explicitly refuses hardness (*"it is **not a hard floor**"*).
  Not pierced. **SOUND.**
- **Smooth-perturbative-only floor `S ≥ 0.2373`** — family scope welded on; the live vehicle is not
  in that family. **SOUND** (worth a rider: it is phrased "CANNOT cross" at exactly our pointer).
- **`SPEC_v10_capstone_cold_start_seeded`** — self-files its own flicker-floor over-claim as F5 and
  inverts it: *"the ONE true intrinsic floor is RATE."* **SOUND.**
- **`ddm_cv1` §11 and `ddm_ba31` Family 2** — both state the formulation scope inline. **Model citations.**
- **`TERMINAL_FINDING_representation_axis_sub015_exhausted_20260619`** (*"sub-0.15 is not reachable
  for any known representation family; the frontier ~0.19110 is near the real achievable floor"*) —
  the largest negative in the corpus and **already RETRACTED at source** on 2026-06-19 by its own
  banner, with the retraction correctly propagated to `SESSION_SYNTHESIS_SoT_20260617_20260618.md`.
  **SOUND — the retraction machinery worked.** One rider: its own stated reactivation criterion
  (*"reactivate on… a measured byte-closed S<0.19110"*) was met on **2026-07-25** by PR130 at
  **0.172141 [contest-CUDA]**, and no formal reactivation row exists. Cross-axis inference is
  forbidden (that row is CUDA; our pointer is CPU), so this is a bookkeeping gap, not a new claim.

---

## §3 LENS 3 — DEGENERACY-MASKED "EXHAUSTION"

`ddm_dc1` found `s_t` exactly multiplicatively degenerate with the shipped pose triple. **The class
is larger than dc1 found, and dc1's derivation is over-conditioned.**

**CORRECTION TO dc1 (strengthens it).** dc1 derives the fold from *"`s_r = 0 ⇒ R = I`"*. That
premise is unnecessary. From `src/tac/optimization/pfs1_warp_receiver.py:44-53`,
`M = R − outer(t,n)/CAMERA_HEIGHT_M` with `t = s_t·[p2,p1,p0]` — `t` is invariant under
`(s_t → c·s_t, triple → triple/c)` for **any** `R`. **The degeneracy therefore holds on the live
v4d path**, where `inflate_runner_v4d.py:196-201` passes `rot = 1.0` or `1 ∓ β/2`, never 0.

**Three degeneracy classes, not one:**
- **(A)** `s_t` ↔ `[p2,p1,p0]` ↔ `CAMERA_HEIGHT_M = 1.22` — a **three-member** class.
- **(B)** `s_r` ↔ `[p3,p4,p5]` — `_expmap_so3` reads only the product `s_r·ω`.
- **(C)** on the `R = I` ground path only: `NATIVE_FX/FY = 910` ↔ `p0`.

| # | verdict re-scoped | coordinate | partner | exactness |
|---|---|---|---|---|
| D1 | `ms8:193` *"Eleven codewords, correctly placed, already **exhaust this instrument**"* | `st_grid` size K | class A | **EXACT** |
| D2 | `pw1:91-104` *"a third menu that does **NOT saturate**… strictly interior"* | `s_t` / `ST_GRID` | class A | **EXACT** |
| D3 | `lg2:124` *"neither v4c nor v4d ever re-picks `s_t`"* | cost of re-solving `s_t` | class A | **EXACT** |
| D4 | `lg2:127` *"a **pinned continuum parameter**, not a menu"* | `s_r` common factor | class B | **EXACT** |
| D5 | `dc1:305` *"`s_t` is the **unique** exactly-degenerate menu"* | the sweep's own uniqueness claim | `CAMERA_HEIGHT_M`, literal base `rot`, `NATIVE_FX` | EXACT (A,B) |

**The sharpest single item is D2, and it is a control-validity failure.** `pw1`'s saturation
discriminator — the campaign's validated bound-vs-direction triage rule — was **validated against
`s_t`**, a coordinate with zero degrees of freedom that **could not have failed the test**. A
coordinate that is exactly degenerate cannot be exhausted, cannot saturate, and cannot have dead
codewords in the quantizer sense. All of D1–D4 must be re-scoped from **AXIS** to
**INSTANCE-of-search-reach** — dc1's own `gap_lattice ≡ 0 / gap_search = −0.049177` split.

**Note on dc1's uniqueness claim:** it is scoped to *menus* (`:45-46`) and therefore structurally
cannot see `CAMERA_HEIGHT_M`, the literal base `rot`, or `NATIVE_FX`. Not wrong — scoped.

### Exhaustion verdicts attacked and NOT broken (honest survivors)

- **`ddm_ba29` 19-coordinate token-coder sweep, span 1.381×, *"saturated at the top."*** Coders are
  bijections of the same symbol stream to bytes; **no shipped continuous coordinate can absorb a
  byte count.** **SOUND** — and it names its own escape (alphabet/context, QA24), a different
  coordinate rather than a degenerate partner.
- **dc1's own NOT-degenerate rulings on `rs_beta_mags`, `selector`, `token_quant_levels`** — each
  re-derived at the decode path and confirmed. (`selector` builds its second homography with a
  **literal `0.0`** for `s_t`; a literal zero survives any fold.) **SOUND.**
- **`ddm_sv1`/`ddm_os1` `converged = 0/600`** — a solver-termination census, not a coordinate
  exhaustion verdict. **SOUND as scoped.**
- **`ba31`'s three `saturated_do_not_spend` byte pools** — byte pools with measured exchange rates.
  **SOUND** (see F4 for the separate scope-transit nit).

**Best remaining non-degenerate pose-side target, surfaced by this lens:** `pitch = 0.0`, hardcoded
at every call site. `n` is unit-norm by construction, so pitch rotates the plane-normal *direction*
and no scale can absorb it. **Genuinely unswept AND genuinely non-degenerate.**

---

## §4 LENS 4 — JOINT-ΔS VETO (the operator's priority lens)

### The flagship, and the only fully-measured live-vehicle re-open

**`ep854 × cell_drop50`** — `ddm_uv1` (#881×#882 / #827), `ddm_cr2r`, `ddm_cr1`. MEASURED n600:

| column | arithmetic | S units |
|---|---|---:|
| Δseg | `100 × (0.00394407 − 0.00431179)` | **−0.0367720** |
| Δrate | `25 × (285,529 − 360,323) / 37,545,489` | **−0.0498023** |
| **seg+rate (the decision column)** | | **−0.0865743 (a WIN)** |
| Δpose | `√(10·37.87713242) − √(10·0.00764555)` | +19.185544 |

- **16.84 % of the seg+rate decision gap · 11.92 % of the total gap.**
- **57 % of the win is RATE, not seg** (`cr1` §3a) — it is not a seg-only artifact.
- Triple-measured and consistent: −0.0866789 predicted (`cr2`), −0.0867981 at matched codec (`cr1`),
  −0.0865743 realized (`uv1`).
- `cr1` §3a: **89.1 % of the 0.097465 S total known inventory. The largest single item.**
- Pose debt is scale-dependent on what is assumed re-solved: **+19.19 S** (naive transplant, uv1),
  **≥ +4.04 S** (74-pair matched-control floor, cr2r), **+3.367 S** (61-pair matched re-solve, cr1).
- Both source arms already say the win survives: uv1 *"The seg+rate win is real and stays stranded"*;
  cr2r *"The −0.0867 S seg+rate half remains real. It needs a pose-carrying base, not a better solver."*
- **Verdict scope at source is INSTANCE** — uv1: *"this kills the ep854-base composition, not the
  base-swap family."*

**ROOT CAUSE, MEASURED:** `window_03/tr1_config.json` carries `w_seg 100.0`, `w_rate 0.05`, and
**no pose term of any kind** (`cr1` §4). The base was burned with the pose objective absent; uv1
measured the mechanism (`corr(f1_gr1, f1_ep854) = +0.119`; d_seg *improved* while photometric
correspondence was destroyed) and named the law: **seg-only training actively SPENDS pose
legibility.**

**REPAIR PATH — named, never built.** `ddm_cv1` §1: *a **pair-coherence loss term** binding the two
frames as one carrier under one warp — **no burn config in the campaign has ever carried this
term***. cr2r: *"Joint/in-loop pose descent during the burn is untested on this base and is the
named alternative."* Post-hoc solving is **measured dead on this base** (uv1 proved it by
arithmetic: a full re-solve on 4 pairs already exceeds the whole n600 break-even budget by 8.1 %) —
so the repair must be **in-loop**, which is precisely a *"technique we already know"* (CLAUDE.md
§"Pose is SOLVED": only JOINT descent crosses the photometric wall).

### The precedent that the debt is payable is already MEASURED, not argued

**`ddm_ck1`, QA06 Knee-A.** Gate result: `S 2.4097 vs ref 2.2566 = +0.153 net REJECT` — rate −0.197
(the win), d_seg +0.165, d_pose +0.185. Re-solving pose **on the base the candidate actually ships**
flipped it: **composed S 1.9863 = −0.270 vs ref**, with non-tail pose left stale and seg damage
uncured; optimistic bracket −0.616. Verbatim: *"the gate's **+0.185 S pose regression was ENTIRELY
stale parameters**, not a capability loss of the knee base."*

**This is the operator's rule, already validated by measurement, 4 days before it was issued.**
(Honest note: ck1 is a *mixed* case — rate won, seg **lost** — so it is not a clean instance of
"wins on segment and rate"; its value is as proof that pose debt from a rate/seg move is repairable.)

### Ranked re-opens under the new rule

| rank | row | seg+rate (S) | % of decision gap | pose debt | repair named? | shippable vehicle? |
|---|---|---:|---:|---:|---|---|
| **1** | **ep854 × cell_drop50** (#827/#881) | **−0.0866** | **16.84 %** | +3.37…+19.19 | **YES — pair-coherence burn term** | **YES (live)** |
| 2 | gr1 **knee location** (§1d) | up to −0.052 | 10.11 % | n/a | n/a — measurement | **YES (live)** |
| 3 | `frame0` **`pose_minimal`** *"DOMINATED"* | −150.71 | — | +13.42 | YES (trained f0 carrier) | **NO — 332 MB probe** |
| 4 | `source → spatial_stride16` *"Pose rejects"* | −440.22 | — | +3.19 | no | **NO — 1.1 MB/pair probe** |
| 5 | **QA66** per-pair rung-A beta member | −0.0134 | 2.61 % | *"B-control post-hoc pose term 14.29"* | YES — ledger already says *"Reopen post pose-conditioning"* | **YES (v4d grammar slot)** |
| 6 | Ω-W-V2 renderer-weight codec | −0.034 (seg column unmeasured) | **n/a — see note** | +0.052 | no | **NO — 2026-04-30 vehicle** |
| 7 | "Region-cheapen" (*"pose VETOES flat-fill"*) | **UNDECOMPOSABLE** — ratio only | — | *"explodes"* | no | unknown |

**Rank 6 carries no gap-percentage deliberately.** Its −0.034 S was measured on a 2026-04-30
vehicle; expressing it as a fraction of *today's* gap would be the borrowed-number fake. Its value
here is historical: it is the artifact where the pose-sensitivity veto was **instituted as family
policy** (*"This is enough to enforce PoseNet-sensitivity weighting for every renderer-weight
codec"*) — i.e. the origin of the habit the operator has now voided.

**Ranks 3 and 4 are arithmetically real and I report them at face value as instructed — but their
vehicles do not exist.** They must not displace rank 1 in scheduling. Rank 3 is separately notable:
`pose_minimal`'s own numbers give net **−137.3 S** even under the OLD joint test, so its
*"DOMINATED"* label was **wrong when it was written**, apparently read off the pose column alone.

### STANDS — pose was NOT the veto (the control group)

| row | why it stands |
|---|---|
| `store-f0-paid` REJECTED | Δrate = **+573 S** (860.4 MB of native keyframes). Pose (+5.98 S) was ~1 % of the kill. **Stands hard.** |
| `ddm_fd1` family-d GN | seg transfer **zero at realized precision** (d_seg identical to 10 digits in 5/6 candidates; the exception moved d_seg the WRONG way). No byte change ⇒ **there was no seg/rate prize for pose to veto.** |
| `ddm_fd2` zero-accept | its own arm disambiguated it: **seg-REALIZATION-GAP, not pose veto.** |
| `seg_secant` `precision_drop3→2`, `stride16→8` | the higher-distortion point uses **more** bytes — loss on both axes. |
| `frame0` `blur_16` leg | worse than `pose_minimal` on rate **and** pose. |
| `gr1` token-granular corrections, W1-COH phase carrier | priced in B/flip against the 1.2731 water line — a pure rate-vs-seg exchange; **pose does not appear in the test.** |
| **QA92** oracle/flat paint | JOINT +0.300 / +0.225 driven by **seg collateral** (*"every class worse incl Lane +0.034"*), not pose. |
| `ba31` `cell_drop50` restore/drop-more | asserted on the **seg+rate projection itself**; **flagged: the artifact does not split seg from pose** — the one STANDS I could not fully verify. |

### REINFORCED-CLOSED — pose wins killed by rate get *more* closed, not less

**`ddm_bp2` blind-set pose actuator** is the exact inverse case and I verified it independently
before the sweep agent reported. d_seg **bit-identical on 600/600**; a real **−65.9 % d_pose** win;
killed by **Δrate +0.44738 S** at the best arm. **seg+rate half = +0.447 S (a LOSS).** Under
"rate and SEG are extremely important," this is *more* dead.

**However, its FOLLOW-ON is P0 by direct operator naming** — see §6.

**`nonrgb_capstone_reopen_verdict`** — verdict is GATED-GO/DEFER, and its own summary already reads
*"pose closed; survival = flat-store-only; rate won… leaving generator-d_seg as the lone open term."*
**Already aligned with the new rule. Not a re-open.**

### OPEN HAZARD — actionable before the next scorer slot is spent

`ddm_mp1_lsb_misplacement_margin_join_20260802.md:247` contains a **pre-registered falsifier that
has not yet fired**: *"if d_pose worsens by more than the seg gain in S units, the frame_0-warp
coupling kills it."* **That is the old pose-veto tiebreak, written into a gate that is still
pending.** Under the new rule it must be rewritten **before** the n600 slot is spent on it,
otherwise it will manufacture a fresh instance of exactly the class this audit is cataloguing.

---

## §5 THE CROSS-LENS PATTERN

Four independent lenses converged on **one mechanism, at three different layers**:

1. **The instrument** (Lens 1) — a prefix sampler flagged CRITICAL twice and never fixed; the cure
   landed in a memo, not in `seg_core.py`.
2. **The scope** (Lenses 2, 3) — correctly-minted FORMULATION scopes that harden into "FUNDAMENTAL",
   "wall", "family", "exhausts" as they are cited forward. This is `ub1`'s law — *a verdict's scope
   is a property of the citation, not of the verdict* — reproduced on four fresh instances (F1, F2,
   F4, D1–D4).
3. **The acceptance test** (Lens 4) — a joint-ΔS tiebreak that let a pose column veto seg+rate wins,
   now voided by operator directive.

And a fourth layer this arm found underneath all of them: **the ledger** (§0). The follow-on
conditions attached to these negatives are written against task ids that **do not exist as rows**.
A scope correction that reaches the memo but not the registry row has not landed (ub1); a follow-on
that reaches the memo but not the ledger cannot fire.

---

## §6 P0 FOLLOW-ONS (operator directive 2), RANKED BY seg/rate LEVERAGE

Every FORMULATION- or INSTANCE-scoped negative below yields exactly one named follow-on with a
falsifier and a cost class. **A row whose follow-on I could not name is marked INCOMPLETE rather
than closed**, per directive.

| P0 | follow-on | from | seg/rate leverage | cost | falsifier |
|---|---|---|---|---:|---|
| **P0-1** | **Burn a base with a PAIR-COHERENCE loss term** (two frames as one carrier under one warp), preserving the −0.0866 S seg+rate win | uv1/cr2r/cv1 §1 (INSTANCE) | **16.84 % of decision gap** | **heavy** (burn) | pre-registered against the cr2r matched control: 74 pairs, same solver — if post-burn mean d_pose on the ep854-class base does not fall below the 0.0131903 break-even, the pair-coherence term is not the cure |
| **P0-2** | **n600 realized d_seg for `cell_drop35` and `cell_drop63`** — resolve the knee that selected the live base | gr1 §B (INSTANCE, §1d) | up to **10.11 %** | **scorer-slot ×2** (no training) | if either beats drop50's n600 −0.0982, the live base is mis-placed |
| **P0-3** | **Per-class n600 join on the LIVE base** vs fl1 floors — settles F1's candidate pierce AND converts the 55.3 % seg share into a mechanism | fl1/gc14/cv1 §11 (FORMULATION) | gates the **55.3 %** seg axis branch | **$0-ish** (cached lstars, one argmax pass) | if the live per-class residual/floor ratios show Undriv+Movable still above 1.0, the seg runway is NOT drained and gc14's default stop is wrong |
| **P0-4** | **PARAMETRIC blind-set perturbation** — k coordinates and signs generated from a handful of shipped scalars (operator named this explicitly) | bp2 §7 (FORMULATION) | rate-side; the only untried shape that can beat per-coordinate pricing | medium (needs a generator) | bp2's own test: NET ΔS must go negative at some k; the generator's output must correlate with the gradient far better than bp2 §6's two rankings did |
| **P0-5** | **Re-run pw1's bound-vs-direction discriminator on a NON-degenerate control** | pw1/dc1/lg2 (D2, AXIS→INSTANCE) | validates the triage rule gating **79 of 84** unmeasured menu points | **$0** (scorer-free triage) | if the rule's verdict changes on a control with real DOF, every menu verdict it produced is INSTANCE-scoped |
| **P0-6** | **Rewrite `ddm_mp1`'s pre-registered pose-veto falsifier** before its n600 slot fires | mp1:247 (hazard) | prevents a NEW instance of the vetoed class | **$0** (edit) | n/a — apparatus |
| **P0-7** | **QA66 per-pair rung-A beta member**, re-priced post pose-conditioning | deferral ledger QA66 (DOMINATED-by-pose) | **2.61 %** | ~0 (data measured; grammar+build) | ledger's own condition: fires once the base carries pose |
| **P0-8** | **Fix the sampler**: give `decode_gt_frame1_pairs` a stride/representative mode and re-run any comparative verdict below the 0.034 S threshold | seg_core.py:80 (instrument) | protects **all** future comparative verdicts | **$0** (code) + re-runs | the 06-30 cure, 33 days owed |
| **P0-9** | **Sweep `pitch`** (hardcoded 0.0, genuinely unswept AND non-degenerate) | Lens 3 survivor | pose-side; best remaining non-degenerate target | cheap | if realized d_pose is flat in pitch, it is a true null lever |
| **P0-10** | **Fisher-weighted Jacobian spectrum for #535** ($0, from existing W1-COH receipts) | ub1 UB1-A | collapses a band straddling the SKIP threshold (0.0123 vs 0.1423 S) | **$0** | ub1's, unchanged |
| **INCOMPLETE** | "Region-cheapen" (*"pose VETOES flat-fill"*, 40× d_seg improvement claimed) | grand-council 06-09 | **unknown** | — | **cannot be graded: the artifact gives a ratio and "explodes" with no absolute d_seg, d_pose, or byte figures.** Needs a numbers re-measurement before it can be re-opened or closed. Not closed here. |

**Registry/ledger actions owed (from §0):** each of P0-1…P0-10 needs a canonical ledger row before
it can be scheduled. Additionally, carried forward from `ub1` §6 and unexecuted: amend the
`exact_plane_storage_rate_dead_family_20260719` anchor's `verdict_scope`, and either register
`post_hoc_stored_corrections_dead_joint_descent_required_law_20260718` **with its three qualifiers**
or stop citing it as "law."

---

## §7 HONEST SURVIVORS — negatives that pass all four lenses

Consolidated so the operator can see what is genuinely closed:

- **`is1 training_necessary_residual = EMPTY_ON_CURRENT_EVIDENCE`** — ships its own scope welded on;
  ub1 called it *"the model of how a negative should be written."* Unbroken.
- **A2-08 exact reversible plane storage on its tested object** — 1,549× over box; the largest coder
  gain ever measured here is 1.381×. Un-closable by any entropy stage. (Only the NOUN over-reaches.)
- **`store-f0-paid`** — +573 S of rate. Pose was ~1 % of the kill.
- **`ddm_fd1` / `ddm_fd2`** — no seg/rate prize existed for pose to veto; fd2 self-disambiguated.
- **`ddm_ba29`'s 19-coordinate coder sweep** — coders are bijections; nothing absorbs bytes.
- **`ddm_bp2`** — a real 65.9 % pose win at exactly zero seg cost, killed 3.89× over by rate.
  **Correctly and now more firmly closed** (its *follow-on* is P0-4; the *verdict* is not re-opened).
- **`gr1` verdicts 1, 2, 4 and verdict 3's ORDERING** — verdict 3's ordering survived my own
  bracketing (§1c); only the KNEE LOCATION breaks.
- **FLOOR-384, `S_floor` 0.11797, smooth-perturbative 0.2373, SPEC_v10's own F5 self-correction** —
  all four refuse the hard read at their own mint.
- **`TERMINAL_FINDING` sub-0.15-exhausted** — retracted at source and the retraction propagated.
  The machinery worked.
- **`nonrgb_capstone_reopen_verdict`** — already ordered rate → pose → seg, matching the new rule.
- **`dc1`'s NOT-degenerate rulings** on `rs_beta_mags`, `selector`, `token_quant_levels` — re-derived
  at the decode path and confirmed.
- **`sv1`/`os1` solver-termination census** — sound as scoped.

---

## §8 WHAT I DID NOT REACH (coverage claims are the dominant false-claim class)

- **Denominators.** Lens 1 (this arm): 28 caller files enumerated, **8 gate files opened**; 4 corpus
  queries. Lens 2: ~95 candidates surfaced, **27 opened**. Lens 3: ~120 titles surfaced,
  **18 research docs + 5 source files opened**. Lens 4: ~140 files surfaced, **19 opened, 12
  candidates isolated, 8 fully decomposable**. Corpus total is **9,704 docs**; the union opened here
  is **under 1 %**. This audit is DEPTH over COVERAGE by construction.
- I did **not** re-audit the 07-29 → 07-31 window (ba29/ba30/ba31's denominator) and make no claim
  about it; nor the 07-19 → 07-28 window beyond ub1's three graded rows.
- **Not searched at all:** `.omx/research/**/*.json` verdict payloads · the equations registry beyond
  two hits · `t5_crucible2/` and `t5_crucible3/` beyond query snippets · `.claude/worktrees/*` ·
  the mask/seg and token-renderer coordinate families (Lens 3) · `src/` generally (Lens 2).
- **The registry is the exact place a formulation floor silently re-hardens** (ub1's law). The row
  `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1` was seen only as a query snippet; its
  live `verdict_scope` / `domain_of_validity` fields are **unread** and are the highest-value single
  next target for Lens 2.
- **One STANDS I could not verify:** `ba31`'s `cell_drop50` restore/drop-more domination is asserted
  on the seg+rate projection but the artifact does not split seg from pose.
- Every negative-existence claim here is scoped inline (§0's is scoped to the canonical task ledger,
  409 rows; §1a's sampler claim is scoped to `seg_core.py` and the 28 `rg` callers).

---

## §9 NEXT IF RESUMED

1. **Read the equations-registry rows** for the flicker floor and the plane-storage anchor and check
   whether their `verdict_scope` fields carry the formulation qualifier. This is where ub1's law says
   a re-scope must land, and neither ub1 nor this audit executed it.
2. **Grade `codex_findings_ddm_ra1_…20260724`** — 32 untagged verdict lines, the densest single
   document; ub1 read only its header and named it the highest-value single next target. Still owed.
3. **Decompose "Region-cheapen"** (the INCOMPLETE row in §6) from its source receipts.
4. **Split seg from pose** on `ba31`'s `cell_drop50` restore/drop-more rows (§8).
5. **Run the transit-decay census over ALL registry anchors** — ub1 named it, it is $0, and this
   audit added four fresh instances (F1, F2, F4, D1–D4) confirming the mechanism generalizes.
6. **Lens 1 over the un-swept families** — the mask/seg and token-renderer coordinate families were
   not examined by any lens.

---

## §10 TRIALITY

- **DAG:** this file. FEED block owed by MAIN on consumption.
- **equations:** two actions proposed, **neither executed here** (this arm has no registry
  authority): (i) the PR130 byte-count correction 190,952 → **191,052** in the
  `gap_decomposition_against_floor_20260802` inputs and in MEMORY; (ii) ub1's two unexecuted
  registry amendments, carried forward unchanged.
- **DSL:** none. No lever, no flag, no config touched.
- **tasks:** **BLOCKED — see §0.** Ten of the fourteen ids this audit was asked to work against do
  not exist as canonical ledger rows. P0-1…P0-10 need rows minted before any of them can fire.
- **code:** zero edits. `$0`, scorer-free, pointer UNMOVED.
