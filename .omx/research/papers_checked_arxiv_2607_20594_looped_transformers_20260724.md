# Papers-checked — arXiv 2607.20594 "When Does Recurrence Become an Algorithm? Convergence Selection in Weight-Tied Looped Transformers" (Zhang, Hu, Peng, Xie)

UTC: 2026-07-24 · Harvested by: MAIN (Fable, inline) · $0
Evidence class: MEASURED_EXTERNAL (group word problems, their benchmarks — never our contest
axis). Lessons-only.

## What the paper measures

Weight-tied looped transformer blocks on group word problems. Four findings: (1) budget law —
processing speed v ~ n_train/T_train (measured exponent 0.98±0.04) with principled test-time
halting T* = ceil(n/v̂); (2) ARCHITECTURE constraint (weight tying), not capacity, selects the
algorithm class (serial vs parallel); (3) circuit-complexity barriers don't match theory —
NC¹-complete A5 learns trivially while S5's 120×120 operator DEADLOCKS, and the deadlock is
REVERSIBLE BY CURRICULUM; (4) learned mechanisms TRANSFER through warm-starting but CANNOT be
imposed via input scheduling. Instrument: convergence-time scaling τ(n,i) via head analysis +
causal damage-cone validation, better OOD predictor than conventional metrics.

## Crosswalk vs live surfaces (4 rows)

| # | Their lesson | Our surface | Disposition |
|---|---|---|---|
| 1 | Mechanisms transfer through WARM-STARTING but cannot be imposed via input SCHEDULING | The ws-chain (#518 resume-warmup laws · ws1/ws2/ws3 arbitration · j4→j5 lesson: schedule reform alone did not cure the smoke ascent, warm-start STATE + realized acceptance did) + warm_start_derived_schedule law | **CORROBORATION, sharpest row**: external evidence that the mechanism lives in the STATE, not the schedule — validates arbitrating warm-start candidates (ws3) as first-class and treating schedule as continuation of state, never a substitute. No action; strengthens the sealed doctrine |
| 2 | Optimization deadlock (S5 120×120) is reversible by CURRICULUM, not capacity | curriculum=continuation / instabilities=bifurcations (#318/#344) · event-driven schedule (#686/#688) · the measured "capacity ALONE does nothing/HURTS until basis-match" ranking | **CORROBORATION**: independent instance of curriculum-resolves-what-capacity-cannot. No action |
| 3 | Convergence-time scaling τ(n,i) + halting rule T*=ceil(n/v̂) as a MEASURED stop criterion | #688 DDMEventContinuationV1 event exits · #344 NCDE hit→solve detector · organ plateau detection (AWAITING_MEASURED_PLATEAU) | **NOTE-FOR-CAMPAIGN**: if the #366 campaign loop's event exits ever under-determine a stage stop, a measured convergence-time scaling row (per-stage τ vs residual size) is the shape of the missing telemetry — Class-E row under the DDM-366 dimension contract, gated like all campaign telemetry on the run existing (post-J8F). No build now |
| 4 | Weight tying (architecture) selects serial algorithms regardless of capacity | Our iterative realized-acceptance solves (v17/v19) are hand-DESIGNED serial algorithms; no learned looped module in the live line | **N/A-WEAK**: no learned recurrence in the counted path; recorded for completeness |

## Verdict

`LESSONS_HARVESTED_INLINE; ZERO_ADOPTIONS_REQUIRING_BUILD; TWO_STRONG_CORROBORATIONS
(warm-start-carries-mechanism · curriculum-cures-deadlock) + ONE_CONDITIONAL_TELEMETRY_SHAPE
(convergence-time row for the #366 loop, post-J8F, only if event exits under-determine)`.
Coherence check: NOVELTY — the scheduling-cannot-impose-mechanism result is a new external
datum on a sealed internal doctrine; DERIVATION — rows map to named existing laws, none spawn
hunts; DISTANCE — nothing new on the critical path (j8f unchanged). Pointer 0.1910828242
[contest-CPU] UNMOVED — this is means.

STORES CONSULTED: #518 warm-start laws memory, j4/j5 receipts (FEED-603 rows), #686/#688
schedule spec, #344 NCDE detector, DDM-366 dimension contract, curriculum-continuation memory,
papers_checked_* precedent.
