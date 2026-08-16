# ddm_pv1 — the pose GN solve is CONVERGED, and it converges ABOVE what the shipping decode already delivers

- arm: `ddm_pv1` (pose axis). Charter target: "uncap the pose GN solve, measure what it converges
  to, price it."
- date: 2026-08-16
- axis: `[macOS-CPU advisory frozen CPU-torch PoseNet]` for the floor; `[contest-CUDA T4 n600]`
  for every frontier component. `score_claim=false`, `promotable=false`. **Pointer UNMOVED.**
  This arm dispatched nothing, launched nothing heavy, and produced no archive.
- cost: $0. All results re-derived from retained primary artifacts.

## ANSWER FIRST

1. **#850 is closed and the cap was worth ~nothing — 0.1549%.** Re-derived from primary shards at
   **n=50** (the memo of record quotes an n=29 snapshot). 49/50 pairs stop on a convergence
   *proof*, 0/50 on any cap. The charter's "still descending 13–23% per iteration when it stops"
   is **FALSE on this vehicle by two orders of magnitude**.
2. **NEW — the converged floor is `1.285917e-05`, which is `1.87×` WORSE than the shipping CUDA
   decode's `6.88e-06`.** The uncapped exact solve, given 12 free int12 dims/pair and 350–2,200
   scorer evals/pair, converges to a d_pose nearly twice what the CUDA decode delivers for free.
   **The pose carrier has no measured headroom to transport onto the shipping axis.**
3. **NEW — the frontier's d_pose is `6.88e-06`, not `6.885642960696714e-06`.** The 16-digit value
   is the **CP135 base at 186,252 B**, carried onto hv1's **182,759 B** row. Worth **3.4009e-06 S**.
   It is pinned as a constant in 13+ modules and is the base of a **live sealed admission bar**.
4. **NEW — that live bar has a false-admit window.** `ddm_ps1u`'s sealed r2 order states an
   "equivalent" d_pose rule that is not equivalent: a T4 row in
   **[6.245822e-06, 6.251199e-06)** ADMITS on the sealed rule while **RAISING** S.
5. **PREDICTION (pre-registered, falsifiable): ps1u r2 REFUSES on the CUDA axis** — while its
   advisory leg admits easily. Mechanism in §4.

STORES CONSULTED: `ddm_ps1u_uncapped_pose_solve_20260816.md` ·
`ddm_errata_8dp_band_instrument_mixing_20260813.md` · `ddm_gx1_gap_closure_composition_table_20260816.md` ·
`ddm_mc36_dual_axis_t4_verdict_20260814.md` · `ddm_qs5_verdict_and_no_toy_enforcement_20260813.md` ·
`ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md` · `.omx/state/canonical_task_status.jsonl` (#850 rows) ·
`experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json` ·
`/Volumes/APDataStore/pact/ddm_ps1u_uncapped_pose_20260816/{n64/rows_shard*.jsonl,dual_axis_pose/SEALED_REQUEST.json}`.

---

## 0. My charter's premise was stale in every particular. Corrected at source.

| charter claim | status | evidence |
|---|---|---|
| "#850 … MEASURED and never acted on" | **FALSE — closed twice** | completed 2026-08-03 by `ddm_pj2` (`tools/pj2_pose_scale_joint_solve.py`, exits split + convergence proof); residual closed 2026-08-16 by `ddm_ps1u` |
| "hard-capped at 2–3 relinearizations, no convergence test" | **TRUE only for v4d-lineage solvers; NEVER true for this vehicle's actuator** | `ddm_qs1_frame0_schur_coupled_solve.py:490` is `while True` — uncapped already (census in the ps1u memo §1) |
| "still descending 13–23% per iteration when it stops" | **FALSE here — 0.155%** | §1, re-derived at n=50 |
| "#1074 P3 uncapped pose solve into js8" is mine | **owned by `ddm_ps1u`, live** | pid 44287 mid-decode; sealed r2 fire order staged |

**My charter was ~100% duplicative of a live sibling.** I did not re-run it. I re-derived its
primary artifacts, found three errors and one un-drawn inference, and stopped. The rest of this
memo is only what ps1u's own numbers had not yet been asked.

---

## 1. #850 ANSWERED at n=50 — the cap is worth 0.1549%

Re-aggregated directly from `n64/rows_shard{00..03}of04.jsonl` (50 solved pairs, seeded-random,
**never a prefix** — pose prefixes measure 2.5–4.2× anti-conservatively, m88/m96):

| quantity | value |
|---|---:|
| mean base d_pose | `1.607196e-04` |
| mean final d_pose | **`1.285917e-05`** |
| mass-weighted reduction (scorer convention) | **91.999%** |
| **mass-weighted gain forfeited by stopping at 3 sweeps** | **0.1549%** |
| stop reason | **`sweep_no_improvement` 49/50** · `sweep_relative_gain_below_tol` 1/50 · **cap 0/50** |
| sweeps used | mean 2.70, max 8 |
| pairs worse than base | **0** |

Aggregation is **mean-of-d_pose**, never mean-of-ratios (rt1: the sign flips, ×1.809 vs ×0.431).

Every pair terminated on a **convergence proof**, none on a bound. That is exactly the
instrument my charter asked for — and it already existed. The 8 pairs that ran past 3 sweeps show
5.23% mass-weighted gain past cap-3, but that sub-population figure moved 0.91%→0.72%→3.16%→5.23%
as heavy pairs entered: `verdict_scope: instance`, **do not quote it**. The population figure
(0.1549%) is the robust one.

> **#850 verdict: the relinearization cap is a PRICED-OUT cap-artifact.** Uncapping buys ~0.15%
> of the achievable reduction. `verdict_scope: **formulation**` — relinearization-budget
> uncapping on per-pair pose GN solves. This is a real negative and it closes the question.
> Three solvers (pg1, pj2, ps1u), three vehicles, one law: **the pose GN descent is front-loaded
> and the iteration budget is not where the pose prize lives.**

---

## 2. THE UN-DRAWN INFERENCE — the solve converges ABOVE the shipping decode

Put ps1u's converged floor next to the shipped frontier component. Nobody had.

| object | d_pose | provenance |
|---|---:|---|
| advisory CPU-decode, **base** | `1.607196e-04` (n=50 sample) / `1.4747e-04` (n600) | ps1u shards / advisory `contest_auth_eval.json` |
| advisory CPU-decode, **uncapped exact solve CONVERGED** | **`1.285917e-05`** | ps1u shards, n=50, proof-terminated |
| population-scaled converged floor | `1.179907e-05` | n600 base × (1 − 0.91999) |
| **shipped CUDA-decode, base (no carrier optimization at all)** | **`6.88e-06`** | hv1 r2 T4 receipt, `avg_posenet_dist` |

**Ratio: the CUDA decode is `1.87×` better than the solver's converged answer** (`1.71×` on the
population-scaled floor). And ps1u's own admission threshold (`6.2458e-06`, §3) sits **`2.06×`
below the solver's converged floor on its own object**.

### What this does and does not license

**Claims (bounded, defensible):**
- (a) Given full uncapped freedom over the 12 int12 codes/pair, the solver **cannot** push the
  advisory object below ~`1.3e-05`. That residual is imposed by ingredients the carrier does not
  control — the R operator, the uint8 round, the shipped lattice, the frozen net — **plus** the
  advisory decode.
- (b) The CUDA decode reaches `6.88e-06` with **zero** carrier optimization. Its frames are
  intrinsically ~2× closer to the GT pose manifold than anything the carrier reaches on the
  advisory decode.
- (c) Therefore the solved deltas encode a correction for error that is **largely absent on the
  object that ships**.

**Does NOT claim:** that the CUDA floor is `1.286e-05`. The CUDA-decode object is a *different
object* (ps1u proved device-dependent decode: raw sha `e5539653…` CPU vs `9a6b75e5…` CUDA, same
archive). Its floor is **genuinely unmeasured** and only a CUDA-side solve could measure it —
which needs CUDA-decoded frames and ~350–2,200 scorer evals × 600 pairs on CUDA. That is the
honest limit of this section.

---

## 3. THE FRONTIER'S d_pose IS `6.88e-06` — and the 16-digit value belongs to a different archive

The only authoritative receipt
(`ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json`) carries:

```
avg_posenet_dist                  = 6.88e-06
avg_segnet_dist                   = 0.00029611
score_recomputed_from_components  = 0.15959729295498598
archive_size_bytes                = 182759
```

Rebuilding S from the receipt reproduces the canonical value **exactly**:

```
0.029611 + sqrt(10*6.88e-06) + 25*182759/37545489 = 0.15959729295498598   (exact match)
```

So the frontier's pose contribution is **`0.008294576541`**, not `0.008297977441`.

`6.885642960696714e-06` traces to the **po1/pz4r worker measurement of the CP135 base at
186,252 B** (`ddm_mc36_dual_axis_t4_verdict_20260814.md:15`,
`ddm_qs5_verdict_and_no_toy_enforcement_20260813.md:7`). It is a **real** measurement — of a
**different archive**. Carried onto hv1's 182,759 B row it is worth **+3.4009e-06 S**.

`ddm_errata_8dp_band_instrument_mixing_20260813.md` already named this exact class
("the canonical knows only 6.88e-06") and its correction restored the value's standing *as a
worker-family measurement*. **The class recurred anyway**: the value is now pinned as
`BASE_D_POSE` / `CP135_DPOSE` / `CUDA_BASE_DPOSE` in at least 13 modules —
`ddm_ps1u_uncapped_pose_solve.py:100`, `ddm_ps1u_carrier_delta_codec.py:274`,
`ddm_js8_seg_stack_compensated_rerun.py:64`, `ddm_qs5_resolve_compensation.py:76`,
`ddm_re1_realization_engineered_candidate.py:87`, `ddm_ec1/ec2`, `ddm_mc35`, `ddm_qs3`, `ddm_qs4`,
`ddm_pz4r`, `ddm_re1x` — and into MEMORY m04 and my own charter.

This is the **cross-regime constant transfer** genus: a latched element re-used outside the window
it was derived in. **The cure is scope, not deletion:** the value is correct *for CP135 at
186,252 B*; every hv1-based bar must use `6.88e-06`.

### 3b. The false-admit window in the LIVE sealed order

`ddm_ps1u`'s sealed r2 fire order states its bar twice:

- **primary:** "ADMIT iff the T4 row's recomputed S < 0.15959729295498598" — **CORRECT and
  instrument-consistent** (both sides are evaluate.py 8dp-component canonicals; the candidate's
  expected |ΔS| is ~112× the ±3.51e-6 band, so it is well resolved).
- **restated:** "Equivalently: CUDA d_pose must fall below **6.251198917870592e-06** (>9.21%)" —
  **NOT equivalent.**

Derived from the receipt base instead:

```
pose_base    = 0.15959729295498598 - 0.029611 - 25*182759/37545489 = 0.008294576541
rate_delta   = 588 B * 6.658590e-07 S/B                            = 0.000391525064
pose_max_new = 0.008294576541 - 0.000391525064                     = 0.007903051478
d_pose_max   = pose_max_new^2 / 10                                 = 6.245822264645601e-06
required cut from 6.88e-06                                         = 9.2177%
```

**FALSE-ADMIT WINDOW: `[6.245822e-06, 6.251199e-06)`, width `5.377e-09`.** Worked probe:

| probe d_pose | S under the sealed rate delta | sealed d_pose rule | truth |
|---:|---:|---|---|
| `6.2500e-06` | `0.159599936` | **ADMIT** | **HIGHER than 0.15959729 — REFUSE** |
| `6.2470e-06` | `0.159598038` | **ADMIT** | **HIGHER — REFUSE** |

The window is ~half an 8dp ULP, so an 8dp-reported d_pose can barely land inside it — **but the
dual-axis worker reports float-precision d_pose** (errata operating law 1), so it *is*
exercisable.

> **RECOMMENDATION TO MAIN (narrow, no re-seal needed): adjudicate ps1u r2 on
> `score_recomputed_from_components` vs `0.15959729295498598`. Ignore the d_pose restatement.**
> If a d_pose threshold is used at all it must be **`6.245822264645601e-06`** off the hv1
> canonical base **`6.88e-06`**. The candidate archive, request sha and fire command are
> unaffected — this is an adjudication-rule correction only.

Related, not blocking: `.omx/state/canonical_frontier_pointer.json` still records e480b
(`0.1600920261571558`, sha `e3e6f440…`, 183,502 B) as `our_local_frontier_contest_cuda` — the hv1
move is not in the pointer file. Flagging for MAIN; the pointer is MAIN-owned.

---

## 4. PRE-REGISTERED PREDICTION — ps1u r2 REFUSES on CUDA, ADMITS on advisory

**Both budgets, as MAIN's relay requires. ΔB the screen is evaluated at: `+588 B` (a COST, not a
saving) = `+0.000391525064` S at the portable constant `6.658590e-07` S/B.**

| axis | base d_pose | candidate | screen | verdict |
|---|---:|---:|---|---|
| **advisory** (CPU decode) | `1.4747e-04` | `8.5810e-05` (41.8% cut, 60/600 pairs) | needs 9.22%-equivalent | **ADMITS, easily** |
| **contest-CUDA** (ships) | `6.88e-06` | unmeasured | needs **>9.2177%** cut, to `<6.245822e-06` | **predicted REFUSE** |

**Mechanism.** The carrier deltas are a compensation **solved on the CPU-decode object** (base
`1.6e-04`) and **transported to the CUDA-decode object** (base `6.88e-06`). That is precisely the
`qs4` cross-regime-constant-transfer anti-pattern MAIN's relay cites, which cost **+2.396e-4 S**;
`qs5` then proved only the **in-compile** joint re-solve form. Here the error being cancelled is
**21.4× smaller on the target object**, and the solver's converged floor on its own object
(`1.286e-05`) sits `2.06×` above the bar it must clear on the other. Applying a correction
calibrated to remove error that is largely absent should **add** error, not remove it.

**FALSIFIER (states what would prove me wrong):** if the T4 row returns
`d_pose < 6.245822e-06`, the floor argument is wrong — the CUDA object has carrier-addressable
headroom the advisory object does not, and §2(c) fails. **Either outcome is worth the $0.16:**
ADMIT moves the pointer; REFUSE buys the first real number on the CPU→CUDA pose-carrier transfer
question and routes the axis to `js8` with it in hand.

**Note against MAIN's relay point 2.** The general rule (advisory screens ~4.6× *stricter* than
CUDA at a gap-sized credit) **inverts here**, because these are not two budgets on one object but
two *different objects*: the advisory object carries 21.4× more removable error, so it is far more
**permissive**. An advisory ADMIT on a device-dependent-decode vehicle is not weak evidence for
CUDA — it is close to no evidence. Carry both, and say which object each was measured on.

---

## 5. WHAT I DID NOT DO

- **No re-run of the solve.** It exists, it is uncapped, it converged. Re-running it would have
  been the duplication my charter warned about in the wrong direction.
- **No dispatch, no heavy local launch.** ps1u's decode held the machine throughout (pid 44287,
  ~206% CPU); everything here is JSONL aggregation and arithmetic.
- **No n600 pose solve.** n=50 of a seeded-random 64. The floor (`1.286e-05`) is
  `verdict_scope: **instance**`; the cap answer (§1) and the receipt discrepancy (§3) are
  `verdict_scope: **formulation**`.
- **No CUDA-side floor measurement.** The honest gap in §2, and the reason §4 is a *prediction*.
- **PAYLOAD:** none generated. ps1u's per-pair code vectors are retained by ps1u at
  `n64/retained/codes/pair_*_final_codes.int32.npy`; nothing was measured-and-discarded here.

## 6. MY OWN ROUND-1 ADVERSARIAL REVIEW

1. **Is the floor comparison apples-to-apples?** No — and §2 says so in its own header rather than
   burying it. The two objects differ. I narrowed the claim to what survives (a/b/c) and named the
   measurement that would settle it. The tempting headline — "the pose axis is closed" — is **not**
   supported and I did not write it.
2. **Is the n=50 sample representative?** Its base (`1.607e-04`) is 9.0% *heavier* than the n600
   mean (`1.4747e-04`), and heavy pairs solve *better* (mass-weighted 92.0% > per-pair mean 78.4%).
   So a representative sample would converge slightly *worse* — the floor estimate is if anything
   **optimistic**, which strengthens the conclusion. Reported the population-scaled figure too.
3. **Is 0.1549% tautological?** Partly: pairs converging in ≤3 sweeps contribute 0 by construction.
   Split out the 8 pairs that ran past (5.23%) and flagged that sub-figure as still moving —
   without the split the number would be a fake.
4. **Am I sure `6.88e-06` is not itself a rounding of the 16-digit value?** Checked: the canonical
   S rebuilds from `6.88e-06` to **exact** float equality, and `6.885642960696714e-06` lies
   *outside* the rounding interval of `6.88e-06`. It cannot be the unrounded source. It is the
   CP135 archive's value.
5. **Am I over-reading the false-admit window?** It is `5.377e-09` wide — sub-8dp-ULP and narrow.
   I said so, and rested the recommendation on the *primary* S-vs-S rule being correct anyway. The
   fix costs nothing and the fire is unaffected.
6. **Unverified:** whether the ps1u shards grew past 50 rows after my read (10:20 local); whether
   the advisory n600 base `1.4747e-04` and the sample share an identical instrument pin.

## NEXT_IF_RESUMED

1. **MAIN: adjudicate ps1u r2 on recomputed S, not the d_pose restatement** (§3b). Zero cost.
2. **Scope-fix the pinned constant.** `6.885642960696714e-06` is CP135@186,252 B; hv1 bars need
   `6.88e-06`. 13+ modules affected — a two-landing candidate (fix + a gate refusing an
   archive-unqualified `BASE_D_POSE` pin), sister of the errata's law.
3. **The only measurement that reopens this axis:** a CUDA-side floor — solve the carrier against
   CUDA-decoded frames. Blocked on the device-dependent-decode cure (ps1u §5b: portable native
   decode preserving the CUDA-favorable frames). Until then the CUDA pose floor is unmeasured and
   the axis ceiling stays `−0.008294576541` (86.4% of the `−0.0095973` gap) — **less than the gap,
   so SEG and RATE remain load-bearing**.
4. **Do NOT re-open the relinearization cap.** Closed three times on three vehicles.
