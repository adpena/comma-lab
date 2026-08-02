# ddm_sv1 — solver termination sweep: which solves stop on a criterion, which on a constant, which cannot say

`verdict_scope` tokens are attached per row. Exact contest pointer **0.1910828242 UNMOVED**.
Own-vehicle rows are `[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`,
`promotable=false`. No n600 scorer job was fired.

STORES CONSULTED: `tools/corpus_query.py` (research 7366 · equations 865 · memory 2044 ·
dag 909 · council 292 · tasks 397 · docs 96); the registered law
`tac.canonical_equations.ddm_pw1_menu_saturation_discriminator_v1`; the v4c/v4d receipts on
`/Volumes/VertigoDataTier/pact`; `.omx/state/canonical_frontier_pointer.json`. Deliberately
NOT loaded: the MLX/witness training lane, the seg token path — this arm is pose-side.

---

## 0. The three seed instances re-derived first — two were STALE

| seed claim | status | evidence |
|---|---|---|
| terminal pose GN #850 hard-capped at 2–3 relins with NO convergence test | **STALE — already cured, and off the live chain** | `src/tac/optimization/terminal_pose_gn.py:490-497` documents the stop-on-rejection proof (`None` = "runs to exact convergence … the floor=0 limit"); `:1085-1092` honours it across resume. Consumers are only `tools/pb1_p5_byte_close_and_eval.py`, `tools/pb1_terminal_pose_gn_600.py`, `tools/rehearse_terminal_pose_gn.py` — the live `ddm_v4c_resolve → ddm_v4d_resolve → build → inflate_runner_v4d` chain never calls it. |
| QA03 `--max-quanta` default 4 outranked a correct convergence test | **STALE — already cured** | `tools/sb1_seg_batch.py:507` default is **32**; `:249-269` carries `stop_reason` with a `for/else`; `:295-297` emits `n_cap_saturated` + `cap_saturated_frac`. |
| pw1: two v4d menus saturated; freeing realized ΔS −0.0163787 | **CONFIRMED** | Re-derives exactly from `ddm_pw1_menu_saturation_discriminator_20260801.py` (`S_BEFORE 0.9639878`, `S_AFTER 0.9476091`, `BYTES_ADDED 85`). |

All three named instances were closed. The live defect was elsewhere, and was found by
scanning the live chain rather than the seed.

---

## 1. Does pw1's own replacement bracket saturate? — **NO** `verdict_scope: instance`

The charter names this arm's likeliest defect: raising a cap and declaring victory without
measuring that the NEW stopping behaviour terminates on the criterion. Answered from
`/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/pw1_arms.jsonl`, **600 pairs, zero new
scorer evals**.

A unit error I made first, and corrected: the per-pair `arm_*_probes` field is a LIST of
probe records with a `phase` in {`probe`, `expand`}; comparing total probe count against a
*doublings* bound is a unit mismatch. Decomposed by phase and padded to the true bound:

| arm | doubling bound | max observed | at the bound | verdict |
|---|---|---|---|---|
| A (dim0) | 23 | 9 | **0 / 600** | `CLOSED_INTERIOR_OPTIMUM`, terminal mass 0.0 |
| B (beta) | 17 | 7 | **0 / 600** | `CLOSED_INTERIOR_OPTIMUM`, terminal mass 0.0 |

61% and 59% of the bound never reached. Modal cost is exactly **2 probes** (504/600 and
527/600) — precisely the "2 extra evaluations on a pair whose bound did not bind" the law
predicts. **pw1's cure terminates by proof, not by a bigger constant.** Welcome negative.

---

## 2. The uncured live site — the photometric (a,b) damped GN `verdict_scope: formulation`

Byte-identical copies at `experiments/ddm_v4c_resolve.py:521` (rung B) and
`experiments/ddm_v4d_resolve.py:250` (`_refit_ab`). Three ways to stop, all collapsed onto
one `if not accepted: break`; the per-pair receipts record none of them.

**MEASURED**, hardest 60 pairs (**86.60%** of the n600 rung-B pose mass, exact from the
receipt: sum 6.230633 over 600), canary **EXACT (max abs err 0.000e+00 on all 60)** against
the shipped `d_rungB`:

```
converged    0 / 60 =  0.0%   ( 0.0% of d_pose mass)
damp_cap    34 / 60 = 56.7%   (67.0%)
relin_cap   26 / 60 = 43.3%   (33.0%)
-> stopped on a BOUND: 60/60 = 100.0%
```

Occupancy, via the registered discriminator, both `sufficient_for_verdict=True`:

* `n_relin` `[0, 5, 11, 9, 35]` — 35/60 at the bound (51.1% of mass) → `SATURATED_MEASURE_BEYOND_BOUND`
* max damping level `[0, 0, 0, 18, 42]` — 42/60 (71.7% of mass) → `SATURATED_MEASURE_BEYOND_BOUND`
  (upper bound: a level-4 that then *succeeded* is counted here; `stop_reason` is the clean signal)

### The structural finding

Under the shipped bounds **`converged` was unreachable**. `not accepted` and `not singular`
together imply the ladder ran every level, so every non-accepting relin was by construction
a `damp_cap`. The loop had no convergence state at all — only bounds wearing one. This is
the genus in its sharpest form: not "a bound outranked a criterion" but "there IS no
criterion".

### Why a bigger constant is not the cure — MEASURED

Raising the ladder 4 → 12 and relins 4 → 32 on the same 60 pairs: the relin bound stops
binding (max 17, `CLOSED_INTERIOR_OPTIMUM`, 0/60 at bound) but **`damp_cap` is still
60/60 = 100%**. The same unfalsified question, one notch out.

### The cure that landed (commit `d7d11ef96f`)

`_step_below_f16_resolution`: (a,b) are quantized to float16 before they are stored and
re-scored, so a non-improving step below half a float16 ULP cannot change the shipped values
however the ladder continues — a proof of local optimality *on the shipped lattice*. It
reads `step` only, costs zero scorer evaluations, and makes `damp_cap` mean exactly "the
ladder was too short", with the required length bounded by `log8(step/resolution)`.
Both call sites now share one `ab_damped_gn`; `ab_stop_census` reports the run-level
denominator and counts an untraced row as `unrecorded`, never as converged.
Byte-identical by construction; guarded by a differential test that keeps a verbatim copy of
the pre-extraction loop and asserts bit-identity over 8 seeds. 25 behaviour tests.

### What the censoring costs — UPPER BOUND, staged not claimed

| quantity | base | freed (32/12) |
|---|---|---|
| sum d_pose over the 60 | 5.395594 | 5.204249 |
| pairs improved / unchanged / **worse** | — | 30 / 20 / **10** |
| scorer evals | 1,385 | 4,196 (3.03×) |

Propagated to the n600 rung-B mean (0.01038439 → 0.01006385, monotone-guarded):
**ΔS = −0.005012 = 0.65% of the 0.7754681 gap.**

**The caveat travels with the number:** this is measured at the v4c rung-B stage IN
ISOLATION. The live chain then runs the v4d dim0 refine, a second `_refit_ab`, and beta
select on top, all of which re-optimize — so downstream absorption is unquantified and this
is an **UPPER BOUND**, possibly a large overestimate. 10 of 60 pairs got *worse* after f16
quantization, so any adoption needs the monotone guard. Single seed, no noise floor.

---

## 3. The lattice solve — hypothesis REFUTED `verdict_scope: instance`

**Its large rate is a property of the problem, not a censored solve.** The 1.52e-4 object has
**no iteration loop and no budget**: `DisjointResizeOperator.realize_factor2_uint8`
(`src/tac/optimization/uint8_lattice_feasibility.py:358`) is a closed-form integer
construction — four whole-plane numpy assignments over a 2×2 tap support, ~3.5 s for all 600
pairs. Its floor is uint8 plane quantization plus fp32 resize roundoff, corroborated by the
un-rounded variant of the *same* solver reaching d_seg 9.66e-7. The cap-shaped flags on that
chain (`--stop-after-chunks`, `--stop-after-pairs`) default to `None`; the budgeted paths in
that module (`max_nodes=4096`, `max_iterations=16`) are not on the C1 chain
(`repair_with_hard_oracle`: 0 hits across all four chain files). The only `while`-shaped loop
is a byte-count fixed point that raises if it fails to converge
(`tools/measure_v10_two_plane_receiver_timing.py:798-805`).

Its RATE-DEAD verdict therefore does **not** need re-grading on censoring grounds. It was
already re-graded on *scope* grounds five days ago:
`.omx/research/ddm_ub1_untagged_verdict_scope_audit_20260801.md:89-119` grades it
**FORMULATION, not FAMILY** — "mis-scoped by one noun". RATE-DEAD covers *storing* the
plane, not *generating* one.

---

## 4. The reframe — ADOPTED, with one qualification

*A solve with distortion pinned at unbounded rate is a REFERENCE DISTRIBUTION, not only a
failed candidate.* For the lattice solve this holds, and better than expected:

* the exact scorer planes Y0/Y1 are byte-closed and sha-pinned
  (`tools/measure_v10_two_plane_receiver_timing.py:112-113`), 353,894,400 B per plane;
* they are **regenerable in one closed-form pass** (~3.5 s / 600 pairs) from
  `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` via `exact_operator_round_u8` — so the
  reference costs no storage at all;
* that cache also carries `margins` `(600, 384, 512)` f32 = the per-cell top1−top2 slack
  field, which is exactly what a smaller codebook would be designed against;
* a genuine per-cell error/rate reference exists at
  `.../ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/stage_checkpoints/02_scorers/scorer_measurement.json`
  (600 pairs × {q1,q4,q8} × 5 strata, q1 = the uncensored step-1 column).

**Qualification (measured, do not re-derive):** it does NOT serve the v4d pose menus — 8 of
6,934 `.md` files mention `beta_mags`/`s_t_idx`/`_refine_dim0`, all dated today, none from
the ms/lattice lineage. The consumer it *does* serve is the seg/token side.

**Trap:** `/Volumes/VertigoDataTier/pact/ddm_b2p_20260731/qa75_solve_frames/` (3.9 GB) is
filed as "EXACT solve frames" but its `materialize_receipt.json` names the 291,205,400 B box
solve at d_seg 1.16e-3 — **not** q1. The q1 frames themselves were never persisted
(`.omx/research/ddm_rp1_rangeA_cell_realized_probe_20260728.md:25-31`).

---

## 5. The swept inventory, with denominators

Scope `src/ tools/ experiments/`, AST over **16,835 files, 0 parse failures**, 9,297
bounded-iteration loops classified. The raw counts are dominated by `experiments/results/**`
vendored artifact copies and by data-walk loops, so the count is not the signal — these two
joins are:

**A. argparse cap defaults (scope `tools/ experiments/ src/tac`, results/worktrees/tests
excluded: 6,042 files, 0 parse failures).** Matching the SPECIFIC flag's own `default=`
keyword, reporting the SET:

* **98** files carry an argparse cap default (not 31 — the count is flag-set dependent; mine
  is 26 flags, MAIN's was 8)
* **4** mention a stop-reason token; **3 are genuine** —
  `tools/sb1_seg_batch.py`, `tools/run_dqs1_local_first_autopilot.py`
  (`stop_reason="max_candidates_reached"`), `tools/run_dqs1_local_first_tranche.py`
  (`stop_reason="round_limit_reached"`) — and
  `experiments/build_owv3_0120_admm_stack.py` is a **false positive** (the word appears only
  in a docstring at `:46`)
* all 3 genuine emitters are **orchestration** loops, not numeric solvers

So MAIN's conclusion holds for solvers, but "31 files / 0 emitters / 100% censored" should be
read as **98 / 3**, and the 3 are the wrong kind of loop.

**B. the damping-ladder idiom (the class actually fixed here).** **5 of 8,631 files** in
scope, 8 sites:

| site | on live chain? | status |
|---|---|---|
| `experiments/ddm_v4c_resolve.py:521` → now `ab_damped_gn` | yes | **FIXED + MEASURED** |
| `experiments/ddm_v4d_resolve.py:250` → now `ab_damped_gn` | yes | **FIXED + MEASURED** |
| `experiments/ddm_pfs1_ep_warp_pose_solve.py:193` | **yes** | **OWED** — same structure, 6-param pose GN; has an extra genuine `cur < 1e-6` criterion but `damp_cap` still masquerades as convergence |
| `experiments/ddm_v4c_resolve.py:293` (two-plane static GN) | **yes** | **OWED** — same structure, `lm *= 4.0` |
| `experiments/ddm_v4c_resolve.py:346` (single-plane static GN) | **yes** | **OWED** |
| `experiments/ddm_ck1_pose_resolve_kneeA.py:199` | no (probe) | owed |
| `experiments/ddm_qa43_two_plane_parallax_probe.py:149` | no (probe) | owed |
| `experiments/ddm_qa44_photometric_rungs_probe.py:226,:274` | no (probe) | owed |

**My fix is a POINT FIX** on 2 of 5 live-chain sites. The three owed live sites are the
6-parameter *pose* GN — the largest axis — and are in the class by source inspection with
occupancy **UNMEASURED**. Ranked first for a successor because the probe harness already
exists and the receipts already record `n_fwd`.

**C. the silent-instrument class.** Every site in B except `sb1_seg_batch.py` could not
report why it stopped before this landing. That is the finding — not "8 bugs", but that no
capped numeric solver in the repo could answer the question.

**D. SEARCH-cap vs FORMAT-cap re-ranking** (folding `ddm_mq1`'s measurement that on a live
payload, storage FORMAT is worth ≤0.056% of the gap while SEARCH over the same variables is
worth ≥1.82% — **33× apart**). Of the 98 files in join A:

* **35 files carry a SEARCH-bounding cap** (iterations / restarts / relinearizations /
  candidates examined / evals) — the side worth prioritizing
* **62 carry only a FORMAT-bounding cap** (`--top-k`, `--limit` producing a table or listing)
* 1 carries both

The site fixed here is a **SEARCH** cap (relinearizations × damping levels = iterations), and
the three owed live sites are also SEARCH caps, so the queue was already correctly ordered.
Notable un-audited SEARCH caps with small defaults on real optimizers:
`tools/click_polish_local.py --max-rounds 1`, `tools/click_polish_exact_search.py
--max-rounds 3`, `tools/codec_op_cma_search.py --max-evals 30`,
`tools/pose_frame0_inverse_solve_probe.py --max-iter 8`,
`experiments/multi_pass_inflate_optimizer.py --max-iters 5`.

**E. fossil-menu check** (mq1: an occupancy histogram that looks like a menu can be a fossil
of the search's reachable set, e.g. beta's `g₀ ± 0.5·(2^k−1)` doubling orbit). Not a risk for
the site fixed here — `n_relin` and damping-level histograms are iteration counts, not tables
— and every discriminator call in this memo passed `objective_mass=`, never counts alone.

---

## 6. Staged for MAIN (not fired — MAIN owns the n600 slot)

1. Re-run the v4c photo stage with the trace live to get the n600 census
   (`rungB_ab_stop_census` now lands in `photo_*_receipt.json` automatically).
2. If the census confirms the 60-pair reading at n600, raise `AB_DAMP_LEVELS` until
   `damp_cap` → `converged` (bounded by `log8(step/resolution)`, not a guess), keep
   `GN_RELINS_PHOTO` freed to 32 (measured non-binding at 17), and add the monotone guard —
   10/60 pairs regressed under f16 quantization.
3. Predicted, as an **upper bound with downstream absorption unquantified**:
   ΔS ≈ −0.005 (0.65% of the gap) at ~3× the rung-B scorer cost.

## 7. Falsifiers

* If the n600 census shows `converged` > 0% at materially the same rate as the 60-pair
  sample shows 0%, my sample was mass-biased and the reading is INSTANCE-scoped to the
  hardest 60.
* If freeing the bounds on the full chain realizes < 20% of the −0.005, downstream
  re-optimization absorbs the gain and the rung-B stage is not the place to spend.
* If the resolution proof reports `converged` on pairs the freed arm still improves, the
  f16-ULP test is too loose and must move to the actual quantization applied at store time.
