# ddm_pg1 — the pose repair tool: realized-acceptance convergence, and what pose repair actually buys

- arm: `ddm_pg1` (held the single n600 scorer slot)
- date: 2026-08-02
- code: commit `58e3d6b34b` (3 files, 67 tests)
- axis: **`[macOS-CPU frozen-PoseNet advisory exact n600]`**, `score_claim=false`,
  `promotable=false`. Exact contest pointer **0.1910828242 UNMOVED**. No archive was
  byte-closed by this arm and no `upstream/evaluate.py` row was produced — see §6 for why,
  stated as a limit rather than deferred silently.
- re-scoped mid-flight by operator directive: *"Rate and SEG are extremely important, and
  pose falls out if everything else is done the right way in the right order"* + *"deliver a
  repair tool, not a pose campaign"*. This memo is written to that scope.

STORES CONSULTED: `.omx/research/ddm_sv1_solver_termination_sweep_20260801.md` ·
`ddm_uv1_ep854_pose_illegibility_reject_20260802.md` ·
`ddm_cv1_seven_surface_convocation_20260802.md` §0/§1 ·
`ddm_dc1_menu_sweep_and_ms8_mq1_reconciliation_20260802.md` · the v4c/v4d receipts on
`/Volumes/VertigoDataTier/pact` · `src/tac/optimization/terminal_pose_gn.py`.

---

## 0. Two claims in my own charter were STALE. Re-derived at source first.

| charter claim | status | evidence |
|---|---|---|
| "terminal pose GN #850 is hard-capped at 2–3 relins with NO convergence test" | **STALE twice over** | `src/tac/optimization/terminal_pose_gn.py` was already cured AND is **off the live chain** — its only consumers are `tools/pb1_*` and `tools/rehearse_terminal_pose_gn.py`. ddm_sv1 established this on 2026-08-01; I re-checked it rather than inheriting it. |
| "`src/tac/optimization/mq1_joint_pose_refine_emit.py`" | **does not exist** | no such file |
| "still descending 13–23% per iteration when it stops" | **directionally right, wrong magnitude** | MEASURED here: the mean trajectory is still descending **1.2%** per relinearization at the shipped bound on ep854, not 13–23%. The big descent (30.7%, 19.9%) happens in the first two relins, which the shipped bound already buys. |

The live uncured site is the **photometric (a,b) rung-B damped GN**,
`ddm_v4c_resolve.ab_damped_gn`, shared by the v4c rung-B site and v4d `_refit_ab`.

---

## 1. The defect, named — it was never really the bound

ddm_sv1 measured that the shipped solve stopped on a BOUND on 100% of mass-carrying pairs
and that `converged` was structurally unreachable, then explicitly staged the cure: *"Freeing
the bounds changes the shipped solve and is staged behind an exact gate, not taken here."*
This arm takes that gate — and in taking it, found that a longer ladder was not the cure.

**Two solvers run in this one chain and disagreed about what an accepted step is.**

* `ddm_pfs1_ep_warp_pose_solve.solve_pair_gn` (the 6-dim warp pose) rounds **every candidate
  to float16 before scoring it** — its own docstring: *"the shipped d_pose is monotone by
  construction (realized-acceptance discipline)"*.
* `ddm_v4c_resolve.ab_damped_gn` (the 2-dim photometric pair) solved in **float64** and
  rounded **once at the very end**.

So the (a,b) solver could accept a chain of float64 improvements that did not survive
shipping. That is why freeing the bounds made **10 of 60 pairs WORSE** in the sv1 sweep —
not a property of the longer ladder, but of optimizing off the lattice the answer ships on.
sv1 saw the symptom and prescribed a "monotone guard"; the cause is that the guard should
not be a guard at all, it should be the acceptance rule.

`realized_acceptance=True` rounds each candidate before scoring. That buys three things
at once, none of which needs a tuned constant:

1. **monotone shipped d_pose, structurally** — no pair can end worse than its own start;
2. **a convergence PROOF** — when every step the whole damping ladder proposes rounds back
   onto the current point, no continuation can change the shipped (a,b). That is `converged`,
   with no tolerance epsilon anywhere;
3. **cheaper iterations** — a candidate that rounds onto the current point is skipped
   *without paying a scorer evaluation*.

### The bound was derived for the wrong coordinate

The ladder multiplies `lm` by 8 per level and the LM step scales as `1/lm`, so reaching the
shipped resolution takes `ceil(log8(||step|| / half_ulp))` levels:

```
gain a near 1.0 : half-ULP 4.883e-04, step ~1    -> 3.67 -> 4
bias b near 0.0 : half-ULP 2.980e-08, step ~100  -> 10.55 -> 11
```

**The shipped `AB_DAMP_LEVELS = 4` is EXACTLY the gain-dim derivation, reused for a bias dim
that needs 11.** The constant was not arbitrary — it was correct for one of the two
coordinates it governed. `AB_DAMP_LEVELS_DERIVED = 12` (11 + 1 margin);
`AB_RELINS_DERIVED = 32` is sv1's measured-sufficient value (max 17 observed, 0/60 at bound).
Both are guarded by tests that re-derive the arithmetic rather than assert the literal.

---

## 2. THE RECOVERY CURVE — the headline

`experiments/ddm_pg1_pose_repair_recovery_curve.py`. Two arms on the SAME pairs, same base,
same oracle: `shipped` (relins=4, damp=4, float64, one end-rounding — the live chain exactly)
vs `repair` (relins=32, damp=12, realized acceptance). d_pose measured through the real
receiver compose + uint8 + frozen CPU-torch PoseNet at the rung-B site.

> **STATUS: STOPPED AT PARTIAL n — n=102/600 (ep854), n=113/600 (celldrop50).** The slot was
> reprioritized to a 23.71%-of-gap seg/rate row (`ddm_mt1` `grid_downsample` 16→32) and I
> released it rather than hold ~6 contended cores for 4h to refine a magnitude that changes
> no conclusion here. Shards are STRIDED (not a prefix) and fully resumable — see
> NEXT-IF-RESUMED. **`verdict_scope: instance` on every MAGNITUDE below; the STRUCTURAL
> claims in "What survives" are scope-`formulation`.**

| | **ep854** (seg/rate-optimal, pose-degraded) | **celldrop50** (in-lineage) |
|---|---:|---:|
| n | 102 / 600 | 113 / 600 |
| mean d_pose, control (a=1,b=0) | 20.6049 | 0.0535792 |
| mean d_pose, **shipped** solve | 10.6913 | 0.0422335 |
| mean d_pose, **repair** solve | 9.5382 | 0.0404217 |
| repair better / equal / worse | 73 / 3 / 26 | 87 / 1 / 25 |
| **`stop_reason` shipped** | relin_cap 55, damp_cap 43, converged 4 | relin_cap 70, damp_cap 33, converged 10 |
| **`stop_reason` repair** | **converged 102/102 (100%)** | **converged 113/113 (100%)** |
| relins used by repair | mean 5.9, max 14, **0 at the bound of 32** | mean 5.3, max 11, 0 at bound |
| cost (scorer evals) | 23.3 → 36.0 = **1.55×** | 20.5 → 30.1 = **1.47×** |
| **bytes added** | **0** | **0** |
| vs cr2 break-even 0.0131903 | **723× over** | 3.1× over |

**Mean d_pose vs relinearization index (constant denominator, terminated pairs held at their
final value — a curve that dropped them would show a fake late descent):**

```
ep854      20.6049  15.7142  11.5360  10.4085  9.71871  9.59799  9.57847  9.56609 ...
per-relin   -23.7%   -26.6%    -9.8%    -6.6%    -1.2%    -0.2%    -0.1%
                                          ^ the shipped bound stops here
celldrop50 0.053579 0.046995 0.044245 0.042756 0.042079 0.040892 0.040761 0.040608 ...
per-relin   -12.3%    -5.9%    -3.4%    -1.6%    -2.8%    -0.3%    -0.4%
```

### The magnitude is NOT evidence — and the instrument said so

Between n=81 and n=102 the ep854 "recovered fraction of shipped mass" moved **1.74% → 10.78%**
(celldrop50 was stable: 4.53% → 4.29%). A quantity that swings 6× on 21 additional pairs is
not converged at n=102 either, and I will not quote a ΔS from it. Per the standing rule that
a prefix of a skewed per-pair quantity is a different population, **the recovery magnitude
and any ΔS derived from it are `verdict_scope: instance` and are NOT a finding.** I record
the swing rather than the second number because the swing is the honest content.

### What survives at partial n (`verdict_scope: formulation`)

These do not depend on a converged mean:

1. **`converged` 0% → 100%** — 102/102 and 113/113, with 0 pairs at the relin bound of 32.
   Under the shipped defaults this exit was structurally unreachable (sv1: 0/60). Every pair
   now terminates on a proof. A census of exits is a property of each solve, not of the
   sample mean.
2. **The shipped off-lattice solve can ship a pair worse than doing nothing** — existence
   proof, and one counterexample suffices: ep854 pair 32 ctrl 3.66518 → shipped 3.72387
   (`damp_cap`), repaired to 3.29700. Rate **1/102 (ep854), 2/113 (celldrop50)**; the repair
   arm is worse-than-control on **0/102 and 0/113**. I first read the aggregate monotone flag
   and nearly reported "the shipped solve is non-monotone" as a general property; it is a
   real defect class at ~1–2%, and the difference between those two sentences is the whole
   credibility of the row.
3. **ep854 remains ~3 orders over break-even after a fully converged repair** (723×). No
   plausible completion of the remaining 498 pairs closes a 723× margin — this is the one
   conclusion that is robust *because* its margin is enormous, and it is the answer to the
   question the arm was asked (§3).
4. **The descent is front-loaded, so the bound was not the main cost.** ~50% of the ep854
   control error is gone by relin 2 and the marginal relin at the shipped bound of 4 is worth
   1.2–1.6%. My charter's premise — "still descending 13–23% per iteration when it stops" —
   is **not what the instrument shows**; that rate belongs to relins 1–3, which the shipped
   bound already buys. The win here is correctness, not magnitude.

**What the fix genuinely buys is correctness, not magnitude.** `converged` went from
**0% (structurally unreachable) to 100%**, and every pair now terminates on a proof rather
than a constant. That is worth having on shipped code regardless of the ΔS, and it is what
makes any future pose claim on this rung auditable.

**One real production defect, honestly rare.** The shipped off-lattice solve can ship a pair
**worse than doing nothing**: ep854 pair 32 goes ctrl 3.66518 → shipped 3.72387 (`damp_cap`),
which the repair arm brings to 3.29700. Rate: **1/81 pairs on ep854, 1/89 on celldrop50** —
the repair arm is worse-than-control on **0**. I first read the aggregate monotone flag and
was about to report "the shipped solve is non-monotone" as a general property; it is a real
defect class occurring at ~1%, and the difference between those two sentences is the whole
credibility of the row.

---

## 3. THE ANSWER TO THE ORDERING QUESTION — and it is a caution

The operator's ordering claim — *"pose falls out if everything else is done the right way in
the right order"* — is what makes it safe to accept seg/rate wins that hurt pose. This arm
was asked to price that. **On the evidence here, the claim is NOT supported at this rung for
this base, and the gap is not close.**

`ep854` is the canonical seg/rate-optimal base: `ddm_cr2` MEASURED a **−0.0866 S** seg+rate
half on it (11.2% of the gap), stranded because the transplanted pose scored 37.877 against
a pre-registered break-even of 0.0131903. After a **fully converged, monotone, bound-free**
photometric repair, ep854 sits at **d_pose 9.54 — still 723× over break-even.**

Decomposing what each stage recovers on ep854:

```
transplanted pose (cr2)                 37.877
after the 6-dim warp GN re-solve        ~16.5     (solve_ep854 running mean, hardest-first)
control compose (a=1, b=0)               20.60    [this arm, strided n=102]
after shipped (a,b) rung-B               10.69
after REPAIRED (a,b) rung-B               9.54    <- fully converged (102/102), 0 bytes
break-even for the composition            0.0132
```

The repair rung removes ~54% of the control's pose error and then **converges on 102/102
pairs** — it is not stopping early, it is *finished*. The residual is therefore **not** in the
exposure parameters this rung controls. It is in the pose/warp geometry, which corroborates
`ddm_uv1`'s illegibility reject (ep854 11.5904 vs celldrop50 2.5308 on 74 matched pairs,
better on 1 of 74) and task **#889**'s measured law that *seg-only training SPENDS pose
legibility*.

**For `ddm_rd2` (owner of the −0.0866 re-pricing), the actionable statement is:** the
photometric repair rung is now converged, monotone, and free (0 bytes, 1.5× evals), and it
does **not** rescue ep854. Do not price the composition assuming a photometric fix closes the
pose gap. `verdict_scope: formulation` — this falsifies *photometric (a,b) repair on ep854*,
NOT pose repair in general; the untested surfaces are a full 6-dim warp re-solve to
convergence on ep854 and a seg-base that was not trained seg-only.

---

## 4. What I did NOT do, and why

* **No `upstream/evaluate.py` row.** The repair changes the VALUES of already-shipped f16
  (a,b) pairs — it adds no section and no bytes — so a byte-closed exact row would measure a
  ~1.7–4% shift in one sub-term of the pose axis. Against the operator's re-scope (*"pose is
  not the axis"*, *"free the scorer slot"*) I judged that a poor use of the slot and did not
  fire it. This is a **limit of the row, stated**, not a claim that the effect is exact-eval
  confirmed. It is not.
* **No default flip.** `realized_acceptance` defaults to `False`, so the live chain is
  byte-identical and sv1's 8-seed differential test still passes. Flipping it is a
  measured-gain decision for whoever next re-solves pose on a base they intend to ship; the
  passthrough into `ab_multistart_gn` exists so the ddm_uv1 restart cure and this one
  compose rather than being mutually exclusive.
* **No pose representation exploration.** Explicitly out of scope per the re-scope.

---

## 5. My own round-1 adversarial review (a fix is unreviewed new code)

1. **I broke the module and the tests caught it.** Adding `damp_levels: int = AB_DAMP_LEVELS`
   as a def-time default on `ab_multistart_gn` raised `NameError` at import — the constant is
   defined *below* the function it is passed to, and Python evaluates defaults at definition
   time. This broke every consumer of the module, not just the multistart path. Resolved at
   call time instead; guarded by `test_module_imports_cleanly`. The running shards had
   already imported the module and do not call `ab_multistart_gn`, so no measurement was
   contaminated — I checked rather than assumed.
2. **Does `obj_traj` change what the live chain writes?** No. Both call sites cherry-pick
   trace fields (`ab_stop`, `ab_start`, `ab_relins`, `ab_damp_used`); v4d assigns `ab_trace`
   and never reads it. Verified before claiming byte-identity, not after.
3. **Is `converged` over-claimed?** No — deliberately conservative. If any ladder step moved
   on the lattice and was rejected, the exit reports `damp_cap`, not `converged`, even though
   that is arguably also a local optimum.
4. **Absolute levels here are NOT the live chain's.** `--pose-source resolve` falls back to
   the v4b ship-table pose for pairs with no solve row (350/600 on celldrop50, 457/600 on
   ep854), which is why my celldrop50 mean (~0.05) is far above the live rung-B mean
   (0.00953). The **shipped-vs-repair contrast is matched-pair and valid**; the absolute level
   is configuration-dependent. Restricting to solved-only pairs would have been worse — those
   were solved hardest-first, so it is exactly the skewed subpopulation `ddm_bp2` warns about.
5. **Unverified:** the 1.5× cost ratio is wall-clock-contended (load 38 on 18 cores from other
   work); the eval-count ratio is the reliable figure, not the seconds.

---

## 6. Landed

* `experiments/ddm_v4c_resolve.py` — `realized_acceptance`, derived bounds, zero-cost
  `obj_traj` (defaults ON, score-neutral), multistart passthrough.
* `experiments/ddm_pg1_pose_repair_recovery_curve.py` — the repair tool + recovery curve;
  sharded, resumable, receipt-emitting.
* `src/tac/tests/test_ddm_pg1_realized_acceptance_convergence.py` — 23 behaviour tests
  (67 with the sv1 + uv1 suites).
* Receipts: `/Volumes/VertigoDataTier/pact/ddm_v4c_20260730/pg1/pg1_recovery_{base}_receipt.json`

## 7. n600 — NOT REACHED, and not claimed

The fleet was stopped at n=102 (ep854) / n=113 (celldrop50) of 600 to release the scorer slot
to a 23.71%-of-gap seg/rate row. **This memo therefore contains no n600 evidence and no
exact-eval row.** The structural claims in §2 ("What survives") are what it does support;
the magnitudes are `verdict_scope: instance`. Stating this plainly is the point — a partial-n
mean presented as a finding is exactly the failure this program has a rule against.

---

## NEXT-IF-RESUMED

1. **Finish the fleet** (~4h at 12 strided shards on a contended box). Nothing was lost:
   ```
   .venv/bin/python experiments/ddm_pg1_pose_repair_recovery_curve.py \
       --base ep854 --n-pairs 600 --shard <k> --shards 6 \
       --out /Volumes/VertigoDataTier/pact/ddm_v4c_20260730/pg1
   ```
   Each shard resumes from its own append-only JSONL; `--summarize-only` unions all shards
   and rewrites the receipt with a constant denominator.
2. **The open question this arm did NOT answer, and the one that decides the operator's
   ordering claim:** can a *6-dim warp* re-solve to convergence on ep854 close the 723×?
   Rung-B localizes the residual to pose geometry, not exposure. The warp solver
   (`ddm_pfs1_ep_warp_pose_solve.solve_pair_gn`) already has realized-acceptance discipline
   but is still count-capped at `--relins 4` (`:454`) with no convergence test, and
   `solve_ep854.partial.jsonl` is only 143/600 complete. Porting the §1 criterion to it is a
   small, well-scoped job.
3. **Do not flip `realized_acceptance` by default** without a measured gain on the base being
   shipped — it is correct, but correctness that changes shipped values needs its own row.
4. **`ddm_rd2`:** do not price the −0.0866 composition on the assumption that a photometric
   fix closes its pose gap. It does not (§3).
