# ddm_ss1 — SELECTION vs SEARCH: the audit, the reconciliation, and the free diagnostic

**Arm:** `ddm_ss1` · **Date:** 2026-08-03 · **Axis:** `[macOS-CPU advisory]` reconstruction from
receipts already on disk. `score_claim=false`, `promotable=false`, **0 new scorer evaluations**.
Exact contest pointer **0.1910828242 [contest-CPU] UNMOVED** — this arm did not move it and does
not claim to.

**Baseline named on every ΔS below:** live best **S = 0.7910689** (= `ddm_pu2`'s banked 6-pair row),
archive **353,805 B**, seg leg **0.4311790**, gap-to-floor **0.6189279**,
`W = 1.273108215332031` B/flip, `cx1` flips **508,639**. **Target:** PR130 floor **S = 0.172141**.

**STORES CONSULTED:** `ddm_sv1_solver_termination_sweep_20260801.md` ·
`ddm_os1_optimization_sweep_termination_census_20260802.md` ·
`ddm_pg1_pose_gn_convergence_20260802.md` · `ddm_uv1_ep854_pose_illegibility_reject_20260802.md` ·
`ddm_dc1_menu_sweep_and_ms8_mq1_reconciliation_20260802.md` ·
`ddm_pu2_pose_tail_floor_probe_20260803.md` ·
`ddm_rs2_flip_damage_rerank_and_drop_seg_leg_20260803.md` · the registered law
`tac.canonical_equations.ddm_os1_termination_census_from_cost_proxy_v1` ·
`experiments/ddm_pfs1_ep_warp_pose_solve.py` at HEAD **and** in the working tree ·
`experiments/ddm_wr1_reverse_waterfill.py` · `experiments/ddm_v4c_resolve.py` ·
`src/tac/optimization/ddm_tr1_runtime.py` · the `d2`/`ps1` receipts on `/Volumes/VertigoDataTier/pact`.
**Deliberately NOT loaded:** the MLX/witness training lane, `dd1`'s displacement surface,
`mg1`'s margin-floor surface, `pu2`'s in-flight control group — cited, not duplicated.

---

## §0 ANSWER FIRST

**"Truncation" is the wrong diagnosis for the live pose initializer, and I can prove it for free.
The iteration cap provably NEVER binds. The LINE SEARCH refuses. That inverts the cure from
*more iterations* to *re-initialization* — which is exactly the lever `pu2` measured and banked.**

1. **The relinearization cap is provably never the reason a solve stops.**
   `bound_relin_exhausted = 0 / 600` at **every** relinearization bound in 2…6, under **both**
   candidate parameter sets, on **both** n600 receipts (`d2`, `ps1`) — 20 configurations, one
   exception at an infeasibly-low bound of 2. MEASURED, $0 (§3.2). What stops the solve is
   `if not accepted` — the damping ladder finding no descent within 4 levels × 2 step scales.

2. **The registered law `ddm_os1_…_v1` has the WRONG solver parameters in its anchor, and the
   receipt's own content proves it — three independent ways, all unanimous at n=600** (§3.1).
   Under the corrected parameters the census stops refusing: **`infeasible 88 → 0`** and the
   verdict upgrades from os1's `UNDETERMINED` / *"85.3 % is a LOWER BOUND, not a census"* to
   **`ALL_STOPPED_ON_A_BOUND` — 600/600, a real census.** os1's own load-bearing reading
   (**`converged 0/600`**) is untouched and independently reconfirmed here.

3. **My charter's §1 seed claims are 2-of-5 STALE, and two prior arms had already said so.**
   `#850` ("hard-capped at 2–3 relins with NO convergence test") is stale twice over —
   `terminal_pose_gn.py` was cured AND is off the live chain (`sv1`, re-checked by `pg1`).
   "13–23 %/iter when it stops" is measured at **1.2 %/relin** by `pg1`. I re-derived both
   rather than inheriting them (§1).

4. **The reconciliation: PLACEMENT, SELECTION and MORE-SEARCH are not three competitors — they
   are one ladder of SCOPE, and which rung binds is diagnosable for free** (§4). `dc1` already
   settled the arithmetic (*"both halves are search: SELECTION is a better index at today's
   reachable set, PLACEMENT is a larger reachable set"*). **`#873`'s 5.7× does NOT generalize** —
   `dc1` measured `st_grid` as the ONLY menu of 5 with any dead codeword (63.6 %; every other
   menu **exactly 0 %**), so on this vehicle *no placement refit is even available* anywhere
   else, and the residual selection move is worth **−0.000748 = 0.096 % of the gap**. The 5.7×
   is the reading of a diagnostic, not a law. **Carry the diagnostic; discard the ratio.**

5. **The cheapest actionable fix is a ROUTING change to an EXISTING solver, and it is free to
   compute** (§5): the census already partitions the 600 pairs into *provably line-search-stalled*
   (→ re-init pays) vs *ambiguous* (→ more relins might). Spending `pu2`'s 6-start budget only on
   the stalled set costs **2.34–4.33×** the shipped pass against uniform multi-start's **6.00×**
   — i.e. **1.4–2.6× cheaper**, computed from the receipt's own `Σ n_forwards = 19,677` (§5.1,
   after I corrected a first draft that overstated the saving as 5.3×). Falsifier pre-registered
   in §5.2. **The routing itself costs 0 scorer evaluations.**

6. **HONEST NEGATIVE / SCOPE.** I fired **no** scorer job and **moved nothing**. Every number
   here is reconstruction from receipts or source inspection. The routing fix in §5 is
   **priced and pre-registered, NOT paid** — paying it needs the n600 scorer slot, which `pu2`
   holds for its control group. It is handed over as a runnable recipe, not as a result.

7. **A REPRODUCIBILITY DEFECT, found while re-deriving (§3.3):** the live chain executes
   `experiments/ddm_pfs1_ep_warp_pose_solve.py` **from the working tree**
   (`ddm_v4c_resolve.py:61` imports the module directly), and that file is **uncommitted —
   36 insertions / 22 deletions vs HEAD**, a *different solver* (6-DOF fixed-`s_t` with
   f16-acceptance vs HEAD's 7-DOF co-optimized). **The code that produced the live archive's pose
   initialization is not in git.** This is the exact confound that mis-anchored os1, still live a
   day later. Sister to `dc1` §3's finding that `pfs1_warp_receiver.py` is not in the repo at all.

---

## §1 ROUND-1 CATCH — my charter's evidence base, re-derived at source

The charter's §1 lists five prior results. I checked each against the primary artifact before
building on any of them. **Two are stale and one needed a scope correction.**

| charter claim | status | evidence |
|---|---|---|
| `#850` — pose solve hard-capped at 2–3 relins with **no convergence test** | **STALE twice over** | `sv1` (2026-08-01) and `pg1` (2026-08-02) independently re-derived: `src/tac/optimization/terminal_pose_gn.py:490-497` documents a stop-on-rejection proof and `:1085-1092` honours it across resume; its only consumers are `tools/pb1_*` and `tools/rehearse_terminal_pose_gn.py`, so it is **off the live chain**. |
| `#850` — "still descending 13–23 %/iter when it stops" | **wrong magnitude** | `pg1` MEASURED **1.2 %/relin** at the shipped bound on ep854. The 30.7 % / 19.9 % descents happen in relins 1–2, which the shipped bound already buys. |
| `pu2` — model vs search → **SEARCH**, multi-start is the mechanism | **CONFIRMED, and it is my baseline** | `pu2` §5.7: end-to-end `upstream/evaluate.py` n600 on the rebuilt archive, `S 0.8264972 → 0.7910689`, `d_seg` bit-exact, bytes exactly 353,805. |
| `ms8`/`dc1_fold` — the win is in the SEARCH bucket, not FORMAT | **CONFIRMED** | `dc1` §1: `s_t` is exactly multiplicatively degenerate with the shipped translation triple (max rel. homography difference **4.539e-16** over n600), so `gap_lattice ≡ 0`. |
| `#873`/`#882` — PLACEMENT beats SELECTION **5.7×** | **CONFIRMED as arithmetic, REFUTED as a law** | `dc1` §3/§6, measured with a denominator — see §4.2. |

*(Two arms in a row before me — `rs2` §1.1 and `br1` round-3 — also found their charter's premise
false at the primary. That is now three consecutive arms. The cure that worked all three times was
reading the primary artifact before building. It is the reason this section is first.)*

---

## §2 THE SWEEP — the three defects, with the denominator

**Scope of the sweep:** the surfaces reachable from the live archive-producing chain
(`ddm_v4c_resolve` → `ddm_v4d_resolve` → `ddm_v4d_build_composed_archive` → `inflate_runner_v4d`)
plus the two upstream producers it consumes (`ddm_pfs1_ep_warp_pose_solve`, `ddm_wr1_reverse_waterfill`)
and the menu inventory `dc1` enumerated. **Denominator = 11 surfaces on the live chain**, below.
Where a prior arm already established a row I cite it rather than re-deriving; where I re-derived,
the evidence column says so.

| # | surface | axis | **truncation** (bound with no reachable criterion) | **starts** | **objective** |
|---:|---|---|---|---|---|
| 1 | `ddm_pfs1_ep_warp_pose_solve.solve_pair_gn` (D2 — the pose **initializer**) | pose | **criterion PRESENT but unreachable-in-practice** (`cur<1e-6`: **0/600**); the operative exit is `not accepted` (bounded line search). Relin cap **provably never binds** (§3.2). | **SINGLE_START** — `p0=tp; p0[3:]=0` (source, working tree). Comment records exactly **one** alternative tried and rejected (raw `t_p` rotation, 74× worse). | **REAL** — frozen CPU-torch PoseNet, f16-rounded acceptance |
| 2 | `ddm_v4c_resolve.ab_damped_gn` (rung-B photometric a,b) | pose | `pg1`: the live uncured site; a longer ladder measured **not** to be the cure | `uv1`: **MULTI_START(n=2)** for the (a,b) re-fit | REAL, but `pg1` measured the two solvers in this chain **disagree on what an accepted step is** (f16 vs f64) |
| 3 | `pj2` pose refine (as shipped into `v4d`) | pose | `pu2`: **under-converged** on 2/4 probed pairs (pair 67 `0.157367 → 0.003968` by continuing GN alone) | **SINGLE_START** → `pu2`'s fix is 6 starts; **no single start dominates** | REAL |
| 4 | `ddm_v4d_resolve` `DIM0_MAX_DOUBLINGS` (:235) | pose | **CRITERION_PRESENT and slack** — `sv1`: max observed 9 of bound 23, **0/600 at the bound**, `CLOSED_INTERIOR_OPTIMUM` | N/A (bracket expand) | REAL |
| 5 | `ddm_v4d_resolve` `BETA_MAX_DOUBLINGS` (:333) | pose | same — max 7 of 17, **0/600 at the bound** (`sv1`) | N/A | REAL |
| 6 | `ddm_wr1_reverse_waterfill` drop ordering (`#766`) | seg/rate | **EXHAUSTIVE over 10 fixed prefixes** of ONE order; the ORDER itself is never re-searched (`:93` single `lexsort`, `:220` hardcoded checkpoints) | **SINGLE_START** — one ordering, no swap/restart move exists | **PROXY, and the proxy is measured wrong** — `rs2`: flip mass on a 16×16 tile = **4.13 %** of the real receptive field (**24.19×** too small); **144/486** cells wr1 certifies "zero-flip" have ambient flips in the region their own drop perturbs; byte tie-break `residual_mass` correlates only **ρ 0.513** with the true per-cell byte marginal |
| 7 | `sb1_seg_batch` `--max-quanta` (QA03) | seg | **CURED** — `sv1`: default 32, `for/else` `stop_reason`, emits `n_cap_saturated` | N/A | REAL |
| 8 | `st_grid` menu (`s_t`) | pose | EXHAUSTIVE (K=11) | N/A | REAL |
| 9 | `selector` menu (1-plane / 2-plane) | pose | EXHAUSTIVE (K=2) | N/A | REAL |
| 10 | `rs_beta_mags` menu | pose | EXHAUSTIVE (K=13), **0 dead by construction** | N/A | REAL |
| 11 | `token_quant_levels` (**96.2 % of the archive**) | seg/rate | EXHAUSTIVE (K=16) — a **hardcoded generic uniform lattice**, `v = 2·code/15 − 1` (`ddm_tr1_runtime.py:1214-1227`), never fitted | N/A | trained through an STE ⇒ in-loop REAL |

### §2.1 Counts

| classification | count / 11 | which |
|---|---:|---|
| a cap that **provably never binds** | **1** | #1 (measured here, §3.2) |
| bound reached, **cure measured NOT to be more iterations** | **1** | #2 (`pg1`) |
| **under-converged** (continuing the same solve improves it) | **1** | #3 (`pu2`, n=4 probed) |
| bound present and **measured slack** (`0/600` at it) | **2** | #4, #5 (`sv1`) |
| **cured** before this arm | **1** | #7 (`sv1`) |
| **SINGLE_START** | **3** | #1, #3, #6 |
| **MULTI_START** | **1** | #2 (n=2) |
| **PROXY objective, and the proxy measured wrong** | **1** | #6 (`rs2`) |
| **EXHAUSTIVE menu, no search defect available** | **4** | #8–#11 |

**The honest shape of the sweep.** Of 11 surfaces, **4 are exhaustive menus** where none of the
three defects can exist, **2 have measured slack**, **1 is cured** — so the live defect surface is
**4 rows: #1, #2, #3, #6**. Three of those four carry SINGLE_START; **not one of the four is a
capacity or format limit**. **Single-start is the single most common live defect on this chain
(3 of 4), and truncation-without-a-criterion is present on exactly ZERO of them.**

**Denominator honesty.** This is 11 surfaces on the **live archive-producing chain**, which is a
deliberately narrow scope. It is NOT a repo-wide census: a keyword scan for iterative surfaces
across `src/tac` + `tools` + `experiments` returns **1,381 files** with a bounded loop, and `dc1`
§3 records that an independent repo-wide sweep of the *receiver + builder import closure alone*
found **36 candidate menus, 19 reaching the archive**, against the 8 its own tool enumerated.
**"Not on the live chain" is not "swept."** Anything outside these 11 rows is **UNKNOWN**, not clean.

---

## §3 THE DECISIVE FREE MEASUREMENT — the registered law's anchor is mis-parameterized

### §3.1 Which solver produced the receipts? The receipt answers, and it contradicts the anchor

`ddm_os1` reconstructs a termination census from a cost proxy `n = init + fd·R + Σ L_i`. Its
anchor records a **CORRECTION**: an initial fit with `init=1, fd=6` (read off the *working tree*)
was discarded in favour of `init=2, fd=7` (read off the *revision git says produced the receipt*),
on the reasoning that the first fit's zero-infeasible result was *"an artifact of wrong parameters
coincidentally fitting."*

**The correction went the wrong way.** os1 trusted the git revision over the receipt's own content.
The receipt discriminates the two solver versions **three independent ways**, and all three are
unanimous at n=600 on **both** receipts:

| discriminator | HEAD (`init=2, fd=7`) predicts | working tree (`init=1, fd=6`) predicts | **MEASURED** |
|---|---|---|---|
| `d_pose_shipped_f16` vs `d_pose_solved` | **differ** — HEAD recomputes it from a *separate* forward (`d_shipped = oracle.d_pose_shipped(...)`, `+1 n_fwd`) | **exactly equal** — working tree returns `"d_pose_shipped_f16": cur` | **600/600 EXACT equality**, both receipts |
| distinct `s_t` values | ~600 — HEAD **co-optimizes** `s_t` continuously (`cst = clip(theta_st + scale·step[6], 0, 0.32)`) | only ST_GRID values — working tree holds `s_t` fixed | **4 values: `0.06, 0.08, 0.12, 0.16`** — all on grid, both receipts |
| minimum `n_forwards` | `2 + 7 + 8 = 17` | `1 + 6 + 8 = 15` | **15**, both receipts |

The third row is the one os1 itself surfaced without following: os1 wrote *"48 of them record fewer
than the **17** forwards a ladder-exhausted single relinearization costs, which is the signature of
the `LinAlgError` break."* Those 48 rows are not a `LinAlgError` signature — **they are `n = 15, 16`,
which is exactly what a ladder-exhausted single relinearization costs under `init=1, fd=6`.**

**Verdict: the anchor's parameters are wrong; `init=1, fd=6` is correct.**
`VERIFIED_VIA_SOURCE_INSPECTION` (both revisions read) **+** `VERIFIED_VIA_EMPIRICAL_ANCHOR`
(three discriminators, n=600, two receipts, $0).

### §3.1b What changes when the parameters are corrected — the law stops refusing

`tools/os1_termination_census_report.py`, `--relin-bound 4 --ladder-levels 4 --line-search-points 2`,
receipt `ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl`, **600 rows, 0 new scorer evaluations**:

| | os1's anchor (`init=2, fd=7`) | **corrected (`init=1, fd=6`)** |
|---|---:|---:|
| converged (`d_pose_solved < 1e-6`) | **0 / 600** | **0 / 600** |
| provably ladder-exhausted | 222 | 114 |
| provably relin-exhausted | **0** | **0** |
| ladder-OR-relin (ambiguous) | 290 | 486 |
| **infeasible under model** | **88 (14.7 %, 33.2 % of mass)** | **0** |
| verdict | `UNDETERMINED` — *"a LOWER BOUND, not a census"* | **`ALL_STOPPED_ON_A_BOUND`** |

**What survives unchanged is the load-bearing reading — `converged = 0/600`** — because it never
depended on the cost model (it reads `d_pose_solved` against the tolerance directly). os1 said
exactly this about its own correction, and it protects the finding through mine too. What changes
is that **os1's census is no longer a refusing lower bound: it is a complete census of 600/600.**

*Same result on the sister receipt* `ddm_ps1_20260730/ps1_ladder.partial.jsonl`: infeasible
`141 → 0`, verdict `UNDETERMINED → ALL_STOPPED_ON_A_BOUND` (at `--relin-bound 3`).

### §3.2 The cap provably never binds — 20 configurations, one line

The question my charter posed is *"does the cap actually bind?"* Answered exhaustively over the
parameter uncertainty rather than at one point:

| receipt | params | relin bound 2 | 3 | 4 | 5 | 6 |
|---|---|---:|---:|---:|---:|---:|
| `d2` | `init=2,fd=7` | **40** | 0 | 0 | 0 | 0 |
| `d2` | `init=1,fd=6` (correct) | 0 | 0 | **0** | 0 | 0 |
| `ps1` | `init=2,fd=7` | **43** | 0 | 0 | 0 | 0 |
| `ps1` | `init=1,fd=6` (correct) | 0 | **0** | 0 | 0 | 0 |

*(cells = `bound_relin_exhausted` count out of 600)*

**`provably relin-exhausted = 0/600` in 18 of 20 configurations, and in ALL 10 under the correct
parameters.** The two non-zero cells sit at a relinearization bound of 2, which the observed cost
range (`max n_forwards = 51`, requiring `R ≥ 4` under the correct parameters) rules out.

**Therefore: raising the iteration cap on the live pose initializer is measured to be a no-op.**
That is a **negative** and it is the useful half of this arm — it closes the lever my charter
proposed spending on, and it does so for **zero scorer evaluations** where a direct A/B would have
cost a full n600 re-solve. It also independently corroborates `pg1`'s finding by a completely
different route (`pg1` measured the descent rate; I measured the exit reason).

### §3.3 The live pose initializer is not in git — a reproducibility defect

While re-deriving §3.1 I established that the **working-tree** version is the one that ran.
`ddm_v4c_resolve.py:61` does `import ddm_pfs1_ep_warp_pose_solve as d2m` off `experiments/`, so the
live chain executes whatever is on disk. `git status` reports that file **modified: 36 insertions,
22 deletions** vs HEAD, and the diff is not cosmetic — it changes the solver's DOF (7 → 6, `s_t`
fixed), its start (`p0[3:] = 0`), and its acceptance quantization (f64 → f16 realized-acceptance).

Its own docstring carries measured justifications (*"pair 0, 0.146 f64-solved → 10.22 f16-shipped"*;
*"raw `t_p` rotation dims through expmap are 74× WORSE"*), so this is real measured work — **it is
simply unlanded.** Consequences, in order of severity:

1. **The code that produced the live archive's pose initialization is not in version control.**
   This is a direct hit on CLAUDE.md's deterministic-reproducibility non-negotiable (provenance:
   *"git hash, seed, config"*): the recorded git hash does not describe the executed code.
2. **It is an active confound generator.** It mis-anchored a *registered canonical law* (§3.1) and
   it is still live a day later. I walked into it myself ten minutes before catching it — my first
   read of the solver was off the working tree and produced os1's discarded parameters.
3. **Sister instance, same class:** `dc1` §3 recorded that `pfs1_warp_receiver.py` **is not in the
   git repo at all** — it exists only on the external volume. The live pose path is *partly outside
   version control*, which is a class, not an instance.

**I did not commit it.** It is another arm's uncommitted work in a file I was told not to disturb,
and landing someone else's in-flight solver rewrite from an audit arm is exactly the collision
CLAUDE.md's serializer discipline forbids. Recorded as OWED with a named owner request in §7.

---

## §4 THE RECONCILIATION — one ladder of scope, and a free diagnostic

### §4.1 The two results do not conflict; one is a special case of the other

`dc1` §1 already did the hard half and it should be quoted, not re-derived:

> ms8's "PLACEMENT vs SELECTION" split is real as arithmetic — the two arms differ by 5.7× — but
> **both halves are search**: SELECTION is a better index at today's reachable set, PLACEMENT is a
> *larger reachable set*.

So "placement beats selection 5.7×" and "the defect is search" are **the same statement**. What
remains — and what my charter actually asks — is *when does each rung bind*. Naming the ladder:

| rung | what it changes | the question it answers |
|---|---|---|
| **SELECTION** | the index chosen within a fixed reachable set | "is a different element of the current menu better?" |
| **MORE SEARCH** | how far the solve descends inside its current basin | "did we stop too early?" |
| **PLACEMENT** | which set / basin is reachable at all (refit the menu, re-initialize the solve) | "are we in the wrong neighbourhood?" |

The rungs are **strictly ordered by cost** (selection ≈ free re-index; more-search ≈ +k
evaluations; placement ≈ ×k whole solves) and **strictly ordered by reach** (each contains the one
above). So the correct move is always *the cheapest rung that is not already exhausted* — which
makes "which rung binds" the only question worth asking, and it is diagnosable.

### §4.2 The 5.7× does NOT generalize — measured, with a denominator

The charter asks me to test this before letting it steer anything. **It fails the test, and `dc1`
had already measured the reason with a denominator:**

- `st_grid` is the **ONLY** menu of the 5 occupancy-measured with any dead codeword (**63.6 %**).
  `selector` **0/2**, `rs_beta_mags` **0/13** (zero *by construction*), `token_quant_levels`
  **0/16**, SMEVR mode-base **0/16** — **exactly 0 % dead, all four.**
- Therefore **no placement refit is available anywhere else on this vehicle** — there is no dead
  region to reclaim. `dc1` §6 states it: *"the family is NOT CLOSED, but it is NARROW … no
  placement refit is available anywhere because no other menu has a dead codeword."*
- The residual **selection** move that does exist (selector re-selection) is worth
  **ΔS −0.000748 at ZERO bytes, 6/600 pairs = 0.096 % of the gap.** Tidy-up scale.
- And `s_t` was never a quantizer at all: `gap_lattice ≡ 0` (homography rel. difference
  **4.539e-16**, n600), so ms8's whole −0.049 was reachable **at the incumbent menu** by folding
  the scale into the pose that already ships — `dc1`'s FOLD recovers **100.02 %** of it for
  **−65 bytes**.

**Conclusion: `#873`'s 5.7× is a measurement of ONE menu's diagnostic state, not a law about
placement vs selection. Do not let the ratio steer anything. The DIAGNOSTIC generalizes; the
ratio does not.** *(Memory `m66`/`m88` discipline: the ratio without its denominator is unanchored,
and its denominator is a population of one.)*

### §4.3 The diagnostic — DERIVED, and each branch already has a measured instance

The binding rung is decidable, per surface, and in every case below the deciding evidence is
**free** (a receipt or an occupancy count), never a new scorer job:

| observation | binding rung | measured instance |
|---|---|---|
| solve stops because the **line search refuses** (no descent at any damping) | **PLACEMENT** (re-initialize) — the point is stationary *within the searched neighbourhood*, so more iterations is provably a no-op | **D2: relin-exhausted 0/600, ladder-exhausted ≥114** (§3.2); `pu2` pair 74 — the shipped point does not move, a start **8.2× worse** descends past it |
| solve stops because the **iteration cap** binds while still descending | **MORE SEARCH** | `pu2` pairs 67, 21 — plain GN continuation `0.157367 → 0.003968` (−97.5 %) |
| menu has a **large dead fraction** | **PLACEMENT** (refit the menu) | `ms8`/`st_grid` — 63.6 % dead, ΔS −0.049 |
| menu has **0 dead** | only **SELECTION** remains, and it is small | `dc1` — 4 menus at exactly 0 %; selector re-selection **−0.000748** |
| ranking key's **support is wrong** | **PLACEMENT** (re-order), not longer prefixes | `rs2`/`wr1` — key sees **4.13 %** of the real footprint; **144/486** "certified safe" cells are not |

**And the diagnostic makes a prediction that was independently confirmed.** Applying the row-5
rule to `wr1`: its false-safe fraction (**29.6 %**) is far over the 10 % threshold `dc1` used, so
the diagnostic says *re-ordering (placement) should beat choosing a different prefix length
(selection)*. `rs2` measured exactly that afterwards and by a different route: at **equal bytes**
(arm B 310 B cheaper), the corrected-key ordering perturbs **11.25 % fewer scorer pixels** at
n600 (600 pairs, sha `cd857c69`), with **27.9 % lower ambient flip mass** in the measured support.
**LABEL: the confirmation is on a PROXY** (perturbed-pixel count and flip mass, not realized
`d_seg`), and it decays to **3.2 %** at the >8 LSB end — `rs2` reports the whole curve, and so do I.
`INFERRED` → the realized-`d_seg` A/B is `rs2`'s owed row, not mine.

**The reconciliation, in one line:**
> **Placement dominates selection exactly when the reachable set has a measurable dead / unreachable
> / mis-certified region; more-search dominates both exactly when the solve stops on its iteration
> cap while still descending. Both are decidable for free — from occupancy counts and from the
> solve's own recorded cost — and on the live chain the free evidence says PLACEMENT on three of
> the four defect rows and MORE-SEARCH on none of them.**

---

## §5 THE CHEAPEST ACTIONABLE FIX — routing, not building

**The fix is not a new search surface** (`m54`: built-instead-of-paid is poison). It is a **routing
change to `pu2`'s existing multi-start recipe**, using a partition that is already computable at
$0 from a receipt already on disk.

### §5.0 What the fix is

`pu2` established that multi-start is the mechanism and banked **ΔS −0.0354283** on 6 tail pairs
(baseline `S 0.8264972` → **0.7910689**, −3 bytes, authority-measured). Extending it means paying
6 starts × R relins on more pairs. **Uniformly, that is ~6× a full n600 solve.**

But §3.2 shows the population is not uniform: pairs that stopped because the **line search refused**
are the ones a new start can help (their incumbent is stationary in the searched neighbourhood);
pairs still bounded by the relin cap are not. **Route the starts by the free census:**

```
partition = os1_termination_census_report.py --receipt <d2 receipt> \
              --cost-field n_forwards --objective-field d_pose_solved --tolerance 1e-6 \
              --relin-bound 4 --fd-per-relin 6 --ladder-levels 4 --line-search-points 2 \
              --init-cost 1 --json           # <-- CORRECTED parameters, per §3.1
  provably ladder-exhausted  -> spend the 6-start budget      (re-init pays)
  ambiguous / relin-bounded  -> spend +2 relins only          (cheap continuation)
```

### §5.1 The price — COMPUTED from the receipt, after I got it wrong once

Measured on the `d2` receipt (600 rows): **Σ `n_forwards` = 19,677**, mean **32.795**,
min **15** / median **33** / max **51**. Under the corrected parameters the partition is
**114 ladder-exhausted / 486 ambiguous** at `--relin-bound 4`, **378 / 222** at bound 5.

Routed cost `= [n_ladder·6 + n_amb·(1 + Δ/32.795)] / 600` where `Δ` = the cost of 2 extra
relinearizations `= 2·(6 + L̄)` and `L̄ = (32.795 − 1)/R − 6` is the measured mean line-search
count per relinearization:

| plan | relin bound 4 | relin bound 5 |
|---|---:|---:|
| shipped single-start pass (Σ = 19,677 forwards) | 1.00 × | 1.00 × |
| **uniform** 6-start over all 600 pairs | **6.00 ×** | **6.00 ×** |
| **census-routed** (6 starts on the ladder-exhausted set; +2 relins on the rest) | **2.34 – 2.47 ×** | **4.29 – 4.33 ×** |
| **saving vs uniform** | **2.4 – 2.6 × cheaper** | **1.39 – 1.40 × cheaper** |

*(Range = the `R ∈ {bound−1, bound}` bracket, since the census cannot pin `R` per pair.)*

**CORRECTION, recorded rather than quietly rewritten.** My first draft of this table claimed
**≈1.14× / "5.3× cheaper"** from a hand-derivation that treated the 6 starts as *replacing* the
base pass instead of *multiplying* it, and that understated the extra-relin cost. Recomputed from
the receipt it is **2.34–4.33×, i.e. 1.4–2.6× cheaper than uniform, not 5.3×.** The saving is real
but roughly **half** what I first wrote, and it **shrinks as the relin bound rises** because more
pairs land in the expensive bucket. Anyone acting on the first number would have under-budgeted the
job by ~2×.

**The routing itself is free** — it costs **0 scorer evaluations**; the partition is read off a
receipt already on disk.

**SCOPE CAVEAT, measured and load-bearing.** D2 is the **initializer**, not the shipped pose: its
own output is `mean d_pose_solved = 0.1595089` against a shipped population mean of `0.0076425`
(**21× worse**), because `v4c` rung-B, `pw1` and `pj2` refine it downstream. So a D2 improvement
**propagates but is not 1:1 with final `d_pose`**, and the ΔS of this fix is **UNKNOWN** until the
falsifier runs. Note also that `d_pose_solved` is extremely right-skewed — **median 0.0027379 vs
mean 0.1595089 (58×)** — which is the tail structure `pu2` worked on and the reason §5.2's sampling
discipline is not optional.

### §5.2 PRE-REGISTERED FALSIFIER — written before any measurement is taken

> **The routing is REFUTED if:** on a matched sample of **≥ 32 pairs** drawn from the
> **ambiguous/relin-bounded** bucket, the 6-start multi-start recovers **≥ 50 %** of the mean
> `d_pose` reduction it recovers on an equally-sized sample from the **provably-ladder-exhausted**
> bucket. That would mean the census carries no routing information and the budget should be spent
> uniformly.
>
> **The routing is CONFIRMED if** that ratio is **< 25 %**, i.e. the ladder-exhausted bucket
> yields ≥ 4× the reduction per start.
>
> **Between 25 % and 50 %:** INCONCLUSIVE — report the ratio, do not route on it.
>
> **Sample discipline (`m88`/`#875`, binding):** the buckets must be sampled **at random within
> bucket**, and the memo must report each sample's mean `d_pose_solved` against the **population**
> mean `d_pose_solved` (0.0076425 population reference). A prefix of a temporally-correlated video
> order is a **different population**; if the sample/population ratio of the governing quantity
> differs from 1, the sample is not admissible.
>
> **Cost of running the falsifier:** 64 pairs × 6 starts ≈ **0.64 ×** a single n600 pass — cheaper
> than the shipped solve itself.

### §5.3 What I did NOT do, stated as a limit

**I did not pay it.** Running it needs the n600 scorer slot, which `pu2` holds for its control
group — and `pu2`'s control group answers an *adjacent* question ("is the win tail-specific or
population-wide?") whose answer this routing depends on. **Sequencing: `pu2`'s control group first,
this second.** Firing both would collide on the same slot and the same pairs.

**Second-order note for whoever pays it:** `pu2` measured that **no single start dominates**
(winners came from `gt_target_rot0` ×2, `shipped_knobs`, a *random* restart, `stageA_best` ×2).
The live D2 start (`p0 = tp; p0[3:] = 0`) is **structurally the same family as `gt_target_rot0`** —
`t_p` translation with rotation dims zeroed. So the incumbent already occupies `pu2`'s
most-frequent winning start, which means **the marginal value of multi-start comes from the OTHER
five starts**, and `shipped_knobs` / random are the ones carrying it. Do not drop the random
restarts as "unprincipled" — one of the six wins came from one. `DERIVED` from `pu2` §4.1 + the
D2 source; **owed:** it is not verified that `pu2`'s `gt_target_rot0` and D2's `p0` are numerically
identical (D2 quantizes to f16 at the start; `pu2`'s start may not).

---

## §6 ROUND-1 ADVERSARIAL SELF-REVIEW

**The failure my charter predicted:** *"finding truncation everywhere because you looked for it —
check that each cap actually binds before calling it a defect."*

**It fired, and the check caught it in the opposite direction.** My §2 sweep initially had row #1
flagged as a truncation defect on the strength of os1's "0/600 converged". §3.2 then measured that
**the cap provably never binds on any of 600 pairs at any bound** — so I moved it out of the
truncation column entirely and the sweep now reports **truncation-without-a-criterion on ZERO of
11 surfaces.** Rows #4 and #5 likewise have caps with measured slack (`0/600` at the bound, `sv1`).
**The charter's central hypothesis is REFUTED on the live chain**, and I am reporting that instead
of the finding I was sent to get.

**Attack 2 — "what did I assume?"** I assumed `relin_bound = 4` for D2. Answered by exhaustive
sweep rather than assertion (§3.2, bounds 2–6, both parameter sets, both receipts). The one
sensitivity that remains: at `--relin-bound 4` the split is 114 provably-ladder-exhausted vs 486
ambiguous; at bound 5 it is 378/222. **The §5.1 price is therefore bound-sensitive** — at bound 5
the routed plan costs **4.29–4.33×** — still cheaper than uniform 6.0×, but only by **1.4×**
instead of 2.6×. I report the whole range rather than the convenient end: **routed plan is
2.34×–4.33× vs uniform 6.00×**, and the saving *shrinks* as the bound rises. A third failure mode
of this same table (the 5.3× arithmetic error) is recorded in §5.1 rather than silently fixed —
**two of my own numbers in one section were wrong before checking, and both were wrong in the
flattering direction.**

**Attack 3 — "does my fix repair the class or the instance?"** §3.1 repairs one law's anchor
(instance). The class is *"a receipt was attributed to a git revision instead of being fitted to
its own content."* The class fix is one line in os1's own tool — **emit the parameter set that
minimizes `n_infeasible` and refuse when two sets tie** — and it is not landed. **OWED, §7.**

**Attack 4 — "would my test still pass if the code were broken?"** The three discriminators in
§3.1 are independent and have *opposite* failure modes (an equality, a cardinality, a minimum), so
a single bug cannot produce all three. The `min n_forwards = 15 = 1+6+8` identity is the strongest:
it is an exact arithmetic coincidence under the correct parameters and an impossibility under the
wrong ones.

**Attack 5 — where I am weakest.** (a) **My denominator is the live chain, not the repo** (§2.1) —
1,381 files carry a bounded loop and I inspected 11 surfaces. Everything else is UNKNOWN, not
clean. (b) **`pu2`'s under-convergence evidence is n=4 probed pairs**, and its own control group has
not run; I inherited that scope and did not widen it. (c) **§4.3's `wr1` confirmation is on a
proxy**, and I labelled it. (d) **I ran no scorer job**, so nothing here is an authority row.

**Attack 6 — did I duplicate a live arm?** Checked: `pu2` owns the scorer slot and the pose control
group (I cite, do not re-run); `rs2` owns the waterfill re-rank and DRIVE (I cite §1.2/§1.5, add
nothing); `dd1` owns displacement/dimensionality and `mg1` the margin-floor cure (neither touched);
`pt2`/`hs1`/`as1` files untouched. My only new *measurement* is the receipt-fit and census sweep,
which no arm was running.

---

## §7 OWED — each row with an owner, per `m45` (nothing exits unowned)

1. **Re-anchor `ddm_os1_termination_census_from_cost_proxy_v1` to `init=1, fd=6`**, APPEND-ONLY
   (Catalog #110/#113 — os1's row is not mutated; a new row supersedes it with the three
   discriminators as evidence). The law's *mechanism* is unchanged and vindicated — its guard
   behaved correctly by refusing under wrong parameters. **Owner: whoever next touches
   `src/tac/canonical_equations/ddm_os1_termination_census_from_cost_proxy_20260802.py`; I am
   handing it to MAIN to route rather than editing another arm's registered law from an audit arm.**
2. **Class fix for the mis-attribution (§6 attack 3):** `os1_termination_census_report.py` should
   FIT its parameters to the receipt (minimize `n_infeasible`) and refuse on ties, instead of
   taking them from the caller's reading of git. ~15 lines in an existing tool. **Owner: same.**
3. **Land or explicitly abandon the uncommitted `ddm_pfs1_ep_warp_pose_solve.py` rewrite (§3.3).**
   The live archive's pose initializer is not in git. **Owner: the arm that wrote it** — the diff
   carries measured justifications, so it has one. If unclaimed within one session, MAIN should
   land it with a provenance note; leaving it is a standing reproducibility violation.
   Sister: `pfs1_warp_receiver.py` is not in the repo at all (`dc1` §3).
4. **Run the §5.2 falsifier**, sequenced **after** `pu2`'s control group. **Owner: `pu2` or its
   successor** (it owns the recipe and the slot).
5. **Realized-`d_seg` A/B of `wr1`'s corrected ordering** — §4.3's confirmation is proxy-only.
   **Owner: `rs2`** (already named as its own owed row; recorded here so it is not re-derived).
6. **UNKNOWN, not clean:** the ~1,370 bounded loops off the live chain, and the 11 archive-reaching
   menus `dc1` could not give a per-pair occupancy for. **Owner: nobody yet — this is the honest
   state, recorded so it is not mistaken for a swept population.**

---

## §8 TRIALITY

- **DAG:** this memo. Verdict: *truncation REFUTED on the live chain; single-start is the live
  defect (3 of 4 rows); placement/selection/search is a scope ladder with a free diagnostic.*
- **equations:** OWED §7.1/§7.2 — the `ddm_os1` anchor re-parameterization. Not landed by this arm
  (another arm's registered law; routed to MAIN).
- **DSL:** N/A — this arm added no lever and no trainer flag.
- **tasks:** cite by CONTENT not by id (`m89`): *"os1 termination census anchor re-parameterization"*,
  *"pfs1 solver uncommitted on the live chain"*, *"census-routed multi-start falsifier"*.
