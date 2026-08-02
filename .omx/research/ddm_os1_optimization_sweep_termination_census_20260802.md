# ddm_os1 — the optimization sweep: which capped solves stopped on a bound, ranked by cost-to-falsify

`verdict_scope` tokens are attached per row. Exact contest pointer **0.1910828242 UNMOVED**.
Own-vehicle rows are `[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`,
`promotable=false`. **No n600 scorer job was fired** — MAIN owns the slot, `ddm_uv1` is ahead
of me in the queue, and every number below is either static or reconstructed from a receipt
that already existed.

STORES CONSULTED: `tools/corpus_query.py` (research 7370 · equations 867 · memory 2044 ·
dag 912 · council 292 · tasks 397 · docs 96); `.omx/research/ddm_sv1_solver_termination_sweep_20260801.md`;
the registered law `ddm_pw1_menu_saturation_discriminator_v1`; `.omx/state/current_focus.md`;
the DAG FEED-fo1/censor1 block (`sub015_DAG_*:27100-27200`); the pfs1/ps1 D2 receipts on
`/Volumes/VertigoDataTier/pact`. Deliberately NOT loaded: the MLX/witness training lane; the
seg token path (this arm is solver-side).

---

## 0. Answer first

**The single highest-value site is `experiments/ddm_pfs1_ep_warp_pose_solve.py:183`
(`solve_pair_gn`), and I measured it at n600 for zero scorer evaluations: `converged`
**0 / 600 = 0.0%**, stopped-on-a-bound **600 / 600 = 100.0%**, 100% of d_pose mass.**
It is on the live chain — `ddm_v4c_resolve.py:61` imports it and `:68` consumes its D2 solve
— so this is the solve that produces the STARTING POINT for v4c's rung-B (a,b) GN, which is
exactly where `sv1` measured restarts beating bound-freeing 1.70×. The ordering I inherited
("the solver's STARTING POINT dominates both its BOUNDS and its MENUS") points at this site,
and the site is censored.

**The method that got it for free is the deliverable.** A damped-GN solve's terminal state
occupies a known interval in its own recorded forward count, so wherever a receipt logged a
cost proxy the census is recoverable retroactively. Landed as
`tac.canonical_equations.ddm_os1_termination_census_from_cost_proxy_v1` + `tools/os1_termination_census_report.py`.

**And one measured negative that changes how this class should be attacked:** loop-SHAPE
analysis cannot rank these sites. Details in §2 — it is the most useful thing I found.

---

## 1. Seeds re-derived first — three of five were stale or wrong

Per the never-recall-from-working-memory rule, every seed fact was re-derived at file:line
before it became a premise.

| seed (from the charter / sv1) | status | evidence |
|---|---|---|
| sv1's join: 98 files carry a cap default, 35 SEARCH / 62 FORMAT | **FLAG-SET DEPENDENT, not wrong** | My own AST join over the same scope (6,045 files, 0 parse failures) with a broader token set returns **831** files with a numeric cap default, **173** SEARCH-ish. sv1 said this would happen and it did. **The count is not the signal**; §3 replaces it with a decidable one. |
| `experiments/ddm_pfs1_ep_warp_pose_solve.py:193` is OWED, live | **CONFIRMED and SHARPENED** | Live via `ddm_v4c_resolve.py:61`. The outer loop is `:183`; `:193` is its inner ladder. Sharper than recorded: the exits are **FUSED** — `if not accepted or cur < 1e-6: break` (`:212`). |
| `tools/codec_op_cma_search.py --max-evals 30` | **CONFIRMED** | `:972`. Note the docstring at `:48` shows `--max-evals 50`; reading the first grep hit would have given the wrong number. |
| `experiments/multi_pass_inflate_optimizer.py --max-iters 5` is un-audited | **WRONG — already cured** | `:411` sets `history[-1].converged = True` and `:468` emits `"converged"` in the receipt. It has a genuine convergence flag. Remove from the queue. |
| `tools/click_polish_local.py --max-rounds 1` is an un-audited search cap | **CONFIRMED, and worse than recorded** | `:126` sets `default=1` against the library's own `max_rounds: int = 40` (`src/tac/click_polish.py:755`) — a **40× wrapper clip**. But the loop at `:850` has three genuine criteria that each LOG their reason, so this is a *bound* defect, not a *criterion* defect. |

---

## 2. MEASURED NEGATIVE — loop-shape analysis cannot rank these sites `verdict_scope: family`

This is the finding I would most want a successor to inherit, and it is a negative.

I built a static classifier over loop SHAPE: for each bounded loop, does any exit exist that
is not the bound? (`NO_EXIT` / `ACCEPTANCE_OVER_BOUNDED_INNER` / `HAS_TOLERANCE_TEST` / …).
Over 6,045 files it classified 6,545 bounded loops. Then I ran it against the only pair of
loops in the repo with **two-sided ground truth**:

* **positive control** — sv1's verbatim pre-cure copy, kept at
  `src/tac/tests/test_ddm_sv1_ab_gn_termination.py:189`, where sv1 MEASURED `converged`
  unreachable (0/60);
* **negative control** — its post-cure replacement `ab_damped_gn`
  (`experiments/ddm_v4c_resolve.py:539`), where a criterion demonstrably exists.

**Both returned `ACCEPTANCE_OVER_BOUNDED_INNER__CONVERGED_UNREACHABLE`. Discrimination
ZERO: 1 true positive, 1 false positive.**

The reason is structural, not a tuning failure. The exit CONDITION is byte-identical across
the cure:

```
pre-cure   if not accepted:            post-cure   if not accepted:
               break                                   stop = ("singular" if singular
                                                          else "converged" if _step_below_f16_resolution(...)
                                                          else "damp_cap")
                                                       break
```

sv1's cure did not add an exit. It added a **proof attached to the stop reason** inside the
break body. **A convergence CRITERION and a convergence LABEL are different objects, and the
loop's exit condition carries neither.**

Per the standing rule that a failed control on a known case is a STOP and not a patch, I did
not tune the classifier to force agreement — there is nothing to tune toward, the shapes are
genuinely identical. I kept one defect it exposed (my first cut attributed an *inner* loop's
`break` to the *outer* loop, which would have made almost any solver look criterion-bearing)
and then abandoned the approach.

**Consequence for the queue:** rank by what a solve DID (its recorded cost), not by what its
source could have done. That is §3.

A second false positive from the same class, worth recording because it would mislead a
successor: my stop-reason token scan reported `src/tac/optimization/terminal_pose_gn.py`
SILENT. It is not — it has a genuine stop-on-rejection proof, breaking at `:1210` with
`admitted` recorded in the trace. The scan missed it because `ast.unparse` drops comments and
because the stop reason is named `admitted`, not `stop_reason`. **There is no canonical
stop-reason vocabulary in this repo**, so a cured site is only auditable by someone who
already knows its local name. That is a real cause of the class recurring: sv1 named it
`stop_reason`, `sb1_seg_batch` names it `stop_reason` + `n_cap_saturated`, `terminal_pose_gn`
names it `admitted`, `multi_pass_inflate_optimizer` names it `converged`.

---

## 3. The decidable reading — and the measurement it bought, at zero cost

`n = init + fd·R + Σ L_i`. An ACCEPTING relinearization costs `L_i ∈ [1, L_max]`; the FINAL
relinearization of a ladder-exhausted solve costs **exactly** `L_max = ladder_levels ×
line_search_points`, because that is what "the ladder ran out" means. Each terminal state
therefore occupies a known interval in the recorded forward count, and an observed count
admits only the states whose interval contains it.

Applied to `d2_ep_solve.partial.jsonl` (600 rows, `relins=4`, `fd=6`, ladder 4, line-search 2),
**zero new scorer evaluations**:

| state | pairs | % | d_pose mass |
|---|---:|---:|---:|
| **converged** (`d_pose_solved < 1e-6`) | **0** | **0.0%** | **0.0%** |
| provably ladder-exhausted (`damp_cap`) | 114 | 19.0% | 46.7% |
| provably relin-exhausted | 0 | 0.0% | 0.0% |
| ladder OR relin — either way a bound | 486 | 81.0% | 53.3% |
| **stopped on a BOUND** | **600** | **100.0%** | **100.0%** |
| infeasible under the model | 0 | 0.0% | 0.0% |

Three things make this load-bearing rather than suggestive:

1. **The `converged = 0/600` leg is EXACT, not inferred.** It reads `d_pose_solved` against
   the literal `1e-6` in the source. It is also not a near-miss: the closest pair sits
   **15.5×** above tolerance.
2. **`0 infeasible` is a positive control on the cost model itself.** The reachable set is
   NOT an interval — a solve of this shape can record 15 or ≥22 but **nothing in 16–21**,
   since a second relinearization costs 6 forwards plus a line search. 600 real observations
   landed 0 rows in that hole. A wrong cost model would not have managed that. (Regression:
   `test_the_pfs1_cost_lattice_has_its_measured_gap`.)
3. **A second, independent n600 instance agrees.** `ddm_ps1_pose_stage.py` calls the same
   `solve_pair_gn`; its receipt gives converged **0/600**, bound-stopped **600/600**,
   0 infeasible. Its `relins` is INFERRED (the receipt does not record the config — itself an
   instrumentation gap), so I ran the sensitivity: **the headline is invariant** under
   `relin_bound ∈ {3,4}`; only the ladder/either split moves (98 vs 271).

This reproduces sv1's (a,b) result — 0% converged, 100% bound-stopped — on the
**six-parameter pose** GN, at **600 pairs instead of 60**, for **0 scorer evaluations where
sv1 spent 1,385**. Same genus, one stage upstream, on the larger axis.

**The sharpest form of the defect at this site is the FUSED exit**
(`if not accepted or cur < 1e-6: break`, `:212`). Convergence and ladder-exhaustion write the
same absence, so no reader of the receipt — and no reader of the source — can separate them.
Fusing a criterion and a bound into one predicate destroys the census at the point of
WRITING, not at the point of reading.

**REACH, with the denominator — the honest limit.** **0 of 21,700** `.omx/research` JSON
receipts and **19 of 8,204** SSD receipts (0.23%) carry any iteration/evaluation count. The
method applies where the proxy exists and nowhere else. That is precisely why sv1 had to buy
the same answer with scorer evaluations. The cheap cure is not this law; it is the habit it
depends on — **record the cost proxy and the census stays recoverable forever.**

---

## 4. THE RANKED TABLE — capped solver/search sites, ranked by COST-TO-FALSIFY

Ranking key is **cost-to-falsify, never predicted ΔS** — `gc17` audited 86 convocation
recommendations and found all six #1-ranked levers refuted by the measurement they ordered,
with bookings ~100× optimistic. A predicted ΔS is an unfalsified belief ordering work.

Scope and denominators: AST over **6,045 files, 0 parse failures**; **221** capped loops that
are also numeric-solve or candidate-search shaped, in **173** files; **176 (79.6%) emit no
stop reason from the enclosing function**. Live-chain membership is DERIVED, not asserted:
import/`import_module` closure from the six v4d entry points = **15 files**, containing
**11** capped solver/search loops.

`ddm_v4c_resolve.py` / `ddm_v4d_resolve.py` are **EXCLUDED — owned by `ddm_uv1`**; listed for
completeness only, not claimed.

| # | site | what the cap bounds | `converged` reachable? | occupancy verdict | cost-to-falsify | next measurement |
|---|---|---|---|---|---|---|
| **1** | `experiments/ddm_pfs1_ep_warp_pose_solve.py:183` `solve_pair_gn` **[LIVE]** | relins (4) × damping ladder (hard-coded `range(4)`) × line-search (2) on the 6-param pose GN | criterion EXISTS (`cur<1e-6`) but **FIRED 0/600** and is FUSED with the bound | **ALL_STOPPED_ON_A_BOUND 600/600, 100% mass** (MEASURED, this memo) | **ZERO — already done** | **Split the fused exit and emit `stop_reason`; then free the ladder and re-measure.** The ladder is the binding half (114/600 provably, 46.7% mass) |
| **2** | same site, **restart policy** | the START, not a cap | n/a | sv1 MEASURED restarts 1.70× bound-freeing on the DOWNSTREAM (a,b) solve that consumes this output | LOW ($0 local, ~33 fwd/pair/start) | Derive a start set (temporal-neighbour `p_star`, per-`s_t` median) rather than a generic displacement; this is the score-moving lever per sv1 §2b |
| **3** | `experiments/ddm_ps1_pose_stage.py` → same `solve_pair_gn` | relins (3, INFERRED — receipt omits config) | same fused exit | **600/600 bound-stopped**, invariant to the inferred bound (MEASURED, this memo) | **ZERO — already done** | Record the solver config IN the receipt; inherits fix #1 |
| **4** | `experiments/ddm_p3v2_optimal_form_pose_resolve.py:212` `s0_cosine6_solve` | relins × line-search `range(8)` | **NO criterion at all** — pure `if not accepted: break` | UNMEASURED (24-row receipt lacks a per-solve cost field; `free_iters_used` belongs to the sister function) | LOW | The cure is already written **10 lines below it**: `s1d_free_solve:246` documents *"Run to CONVERGENCE (not budget-truncate — the exact P3 mistake)"* with a `tol`. Apply the neighbour's own fix |
| **5** | `tools/click_polish_local.py:126` `--max-rounds 1` | polish rounds; clips `src/tac/click_polish.py:755`'s own default of **40** | **YES** — three criteria at `:850`, each logs its reason | UNMEASURED | LOW (logs exist; parse a past run) | Grep a past run's log for which of the three fired. If the `for` exhausted silently, the 40× clip bound a still-improving search |
| 6 | `tools/click_polish_exact_search.py:156` `--max-rounds 3` (`:165` = 40) | same library loop | YES (same loop) | UNMEASURED | LOW | Same as #5; note the two subcommands disagree 3 vs 40 |
| 7 | `tools/codec_op_cma_search.py:972` `--max-evals 30` | CMA evaluation budget | UNKNOWN — not inspected | UNMEASURED | MED (needs a run) | Emit evals-used + best-eval-index; a best found at index ≈30 is a saturated budget |
| 8 | `tools/pose_frame0_inverse_solve_probe.py:226` `max_iter=8` (`:332` `max_iter=6`) | inverse-solve iterations | UNKNOWN | UNMEASURED | MED | Off the live chain; audit only if reactivated |
| — | `experiments/ddm_v4c_resolve.py:284` `_two_plane_static_gn`, `:337` `_single_plane_static_gn` **[LIVE]** | RELINS × ladder | SILENT (no stop reason) — sv1's OWED rows, independently re-found here | UNMEASURED | LOW | **`ddm_uv1` OWNS — not claimed by me** |
| — | `experiments/ddm_v4d_resolve.py:219` `_refine_dim0`, `:317` `_beta_select` **[LIVE]** | doubling brackets | no stop reason, but occupancy MEASURED by pw1/sv1 | **CLOSED_INTERIOR_OPTIMUM, 0/600 at bound** | — | **CLOSED** — pw1's cure terminates by proof |
| — | `src/tac/optimization/terminal_pose_gn.py:1093` | relinearizations | **YES — already cured** (`admitted`, break at `:1210`) | — | — | **FALSE POSITIVE of my own scan** (§2); off the live chain per sv1 |
| — | `experiments/multi_pass_inflate_optimizer.py:366` | `max_iters=5` | **YES — already cured** (`:411`, emitted `:468`) | — | — | **Remove from the queue** (stale seed) |

**Why #1 outranks everything by this key:** its cost-to-falsify was ZERO and the falsification
already came back positive at n600, on the live chain, on the largest axis. Nothing else on
the list has been measured at all.

---

## 5. What I did NOT do, and why

* **No n600 scorer job.** MAIN owns the slot; `uv1` is ahead of me. Every number here is
  static or reconstructed.
* **No edit to `ddm_v4c_resolve.py` / `ddm_v4d_resolve.py`.** `ddm_uv1` owns them.
* **No cure landed at pfs1.** The census is the finding; the fix (split the fused exit, emit
  `stop_reason`, then free the ladder) is unreviewed new code on a live-chain solver and is
  STAGED, not taken — sv1's own discipline, and the fix changes the shipped solve.
* **No ΔS predicted for any row.** Per §4's ranking key. sv1's staged numbers
  (bound-freeing ≈ −0.005, restarts ≥ −0.0086) are its own, carry its own upper-bound
  caveats, and were measured on the (a,b) solve, not this one.

## 6. Falsifiers

* If an instrumented re-run of `solve_pair_gn` reports `converged > 0` on any pair, my cost
  model is wrong (the `converged = 0/600` leg is read from the objective and would survive,
  but the ladder/relin split would not).
* If freeing the pfs1 ladder does not move `d_pose_solved` on the 114 provably
  ladder-exhausted pairs, the ladder was at a genuine local optimum and the census is
  correct-but-inert — `damp_cap` would then mean "converged without a proof", and the cure is
  sv1's f16-resolution test rather than a longer ladder.
* If `ddm_ps1`'s true `relins` was neither 3 nor 4, the split numbers in §3 move; the
  bound-stopped headline does not (measured invariant).
* If a successor finds any capped numeric solver in this repo that names its stop reason with
  a token I did not scan for, my "79.6% silent" is an over-count — it is an upper bound on
  silence, and `terminal_pose_gn` already proved the scan has false positives.

---

**Landed:** `src/tac/canonical_equations/ddm_os1_termination_census_from_cost_proxy_20260802.py`
(registered, registry 419 → 420) · `src/tac/tests/test_ddm_os1_termination_census.py` (29 tests)
· `tools/os1_termination_census_report.py`.
