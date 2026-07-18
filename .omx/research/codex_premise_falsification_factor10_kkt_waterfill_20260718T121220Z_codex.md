# Factor 10 / Task #536 premise falsification — governed KKT waterfill

Date: 2026-07-18 UTC  
Lane: `pool_channel_rd_harness_20260718`  
Receipt: `.omx/research/factor10_kkt_waterfill_blocked_receipt_20260718.json`  
Verdict scope: `IMPLEMENTATION_CUSTODY_AND_GOVERNED_EXECUTION_CONTRACT`  
Status: `BLOCKED_FAIL_CLOSED`; no launch, scorer forward, scientific cell, score claim, or pointer move  
Landing: `MAIN_LANDING_REVIEW_REQUIRED`

## Outcome

The premise that the current governed harness is ready to produce the requested
48 byte-closed component curves is **FALSIFIED for this harness revision**.
This is not a negative verdict on joint waterfilling, coherent correction, or
the representation family.

The system-memory governor **MEASURED ADMIT** a conservative `25 GiB`
projection with `57.122 GiB` projected headroom. Execution still could not
legally start:

1. The harness resolves its canonical root to MAIN and requires the active claim
   ledger and operator authorization under MAIN. This isolated delegation is
   explicitly forbidden to touch MAIN. A shadow worktree authority would be
   false canonical custody, so none was created.
2. The exact governed invocation therefore refused before writes with
   `BLOCKED_GOVERNED_CLAIM_RECEIPT_REQUIRED`; no output directory appeared.
3. Independently of that boundary, source re-derivation proves that `measure`
   always writes only unsplit full-path geometry, sets `finding_eligible=false`,
   and then raises
   `BLOCKED_EXACT_SKIP_DEEP_HOOK_AND_RANGE_KERNEL_INTERVENTION`. Exact
   skip/deep and full orthogonal range/kernel interventions do not exist.
4. The current renewable-authority design is incompatible with P0 resumability.
   The predecessor lease expires after one hour, while `run_contract.json`
   embeds the authority-wrapper SHA and requires byte identity on resume.
   Refreshing the lease invalidates the contract; retaining it fails freshness.

The branch-local claim was closed terminally. The sacred c2 directory and MAIN
were read-only and unchanged.

## What is and is not measured

- **MEASURED:** bank/cache/stop/resume custody hashes; timestamped dead c2 PID
  probes; governor admission; branch-local claim/terminal closure; captured exact
  harness preflight refusal; timestamped absence of the resolved output path.
- **DERIVED FROM SOURCE:** the unconditional 48-cell blocker, false
  `finding_eligible` state, zero scorer forward after the early governed refusal,
  canonical-root/delegation conflict, and renewable-lease/resume deadlock.
- **UNMEASURED:** all 48 pool x head-direction x path x resize cells, every
  byte-closed Seg/Pose/rate marginal, and every KKT operating point.
- **NOT CLAIMED:** an implied minimum score versus `0.19108`. The current
  pointer snapshot read `0.1880443979880752`; neither value moved.

## Named confound hunt: the wrong Gram

Question: is `G_act` the stacked cross-pixel collateral object or merely the
local final-head Gram?

The implementation formula is structurally correct: its intended authority is
the full parsed-bank-code perturbation through renderer, uint8 roundtrip,
resize, nonlinear SegNet, and stacked spatial rows. The local centered-head Gram
is explicitly rejected. But the scored Jacobian uses continuous sub-bin
variants: the separate byte-closed plus/minus-one-bin surface records rendered
changes without scoring those lattice variants. The required deployable
positive control was therefore not executable even apart from authorization.
The result is `FORMULATION_INSPECTION_ONLY`, not an L3 empirical closure. MAIN
must not promote the source shape into a measured or byte-closed `G_act` result.

## Additional helper inconsistency

The canonical claim helper correctly treats every `refused_*` status as
terminal. `export_active_lane_claim_json.py` recognizes only the literal
`refused_dispatch` token, so it can export this lane's
`refused_main_boundary_and_component_hooks` row as `active:true`. The harness
would still reject the `refused_` status, but the exporter payload itself is
false-active custody and must not be used. MAIN should align the terminal-status
predicate before constructing any future execution wrapper.

## Exact discrete rule, KKT relaxation, and why no number is legal

The exact byte-closed authority is finite, not differential. For every feasible
receiver-closed archive move `m` (increment, decrement, or reallocation),

```text
delta_S(m)
  = 100*delta_d_seg
  + delta_sqrt(10*d_pose)
  + (25 / 37_545_489)*delta_B.
```

A move is admitted only when `delta_S(m) < 0`; at a discrete local optimum no
feasible neighboring move has negative `delta_S`.

For a continuous or convexified relaxation and any active receiver-closed lever
`b_j`, the general candidate stationarity law is

```text
-[100 * d(d_seg)/d(b_j)
  + (5 / sqrt(10*d_pose)) * d(d_pose)/d(b_j)]
  = (25 / 37_545_489) * d(B)/d(b_j).
```

At a lower-bound inactive lever, the forward derivative of the relaxed full
score is nonnegative; at a saturated upper bound, the backward derivative is
nonpositive. At a convexified kink, zero belongs to the full relaxed-score
subgradient, equivalently the rate price lies between the negative one-sided
distortion-score slopes. Only when Seg and Pose byte streams are separable, the
cross derivatives are zero, and each allocated byte adds exactly one counted
archive byte does the smooth interior special case become

```text
-100 * d(d_seg)/d(b_seg)
  = -(5 / sqrt(10*d_pose)) * d(d_pose)/d(b_pose)
  = 25 / 37_545_489.
```

Both distortions must remain positive; the raw-`d_pose` marginal diverges near
zero. The calculus is
`DERIVED_CONTINUOUS_RELAXATION_FROM_SCORE_OBJECTIVE_NOT_EXACT_DISCRETE_AUTHORITY`;
separability remains `ASSUMED_AWAITING_BYTE_CLOSED_CURVE_VERIFICATION`. Applying
the exact finite rule or solving the relaxation requires receiver-closed
residual R-D curves on the same artifact. None exists here, so the operating
point and implied minimum score are `null`, not estimated.

Global geometry remains the direct sum `sum_i trace(K_i^2)`, never
`trace((sum_i K_i)^2)`. Pools A/B/C are saddle-first exclusive; isolated pool
ceilings must not be added.

## Exact reactivation contract

MAIN may requeue Task #536 only after all five conditions are true together:

1. MAIN owns a canonical claim, content-bound operator authorization, and fresh
   c2 terminal receipt for the exact harness bytes.
2. Execution authority is split into immutable scientific run custody plus a
   renewable continuation lease that does not alter `run_contract.json`.
3. Exact same-point skip/deep recomposition and full orthogonal range/kernel
   recomposition exist and make all 48 rows `finding_eligible=true`.
4. Each admitted point rebuilds and parses the exact archive, reruns the real
   receiver and frozen CPU Torch scorers, counts bytes, and includes Pose on the
   same artifact before the three-axis solve.
5. The active-claim exporter and claim ledger share one terminal-status
   predicate, including every `refused_*` closure.

## Round-1 adversarial self-review

1. **Attack:** memory admission was described as permission to run.  
   **Resolution:** the receipt calls it necessary-only and records no launch.
2. **Attack:** the branch claim was presented as canonical.  
   **Resolution:** it is explicitly branch-local and terminal; the missing MAIN
   claim remains a blocker.
3. **Attack:** source inspection was promoted to a measured Gram.  
   **Resolution:** the named confound remains formulation-only and its positive
   control is `BLOCKED_NOT_EXECUTED`.
4. **Attack:** a KKT point was inferred from prior pool opportunity totals.  
   **Resolution:** every curve and operating-point field is null; non-additivity
   is restated literally.
5. **Attack:** the refusal killed the waterfill family.  
   **Resolution:** verdict scope is the current execution and harness contract,
   never family or paradigm.
6. **Attack:** a one-hour receipt could simply be refreshed.  
   **Resolution:** the immutable run-contract SHA makes that workaround
   incompatible with resume; the debt is explicit.
7. **Attack:** a stacked-spatial continuous Jacobian was called byte-closed.  
   **Resolution:** the continuous and deployable-lattice surfaces are separated;
   the latter was not scored and no empirical `G_act` is claimed.

## Pointer delta and triality

- Pointer delta: exactly `0`; pointer not mutated.
- DSL/consumer: none landed; current harness remains non-consumable for Factor
  10 and no trainer argv was changed.
- DAG: `.omx/research/factor10_kkt_waterfill_DAG_FEED_20260718.md`.
- Equation candidate: `.omx/research/canonical_equation_candidates_factor10_kkt_waterfill_20260718.jsonl`;
  status is blocked pending byte-close parity, not canonical adoption.
- Completeness matrix: Factor 10 remains `MISSING` with the exact new blocker
  receipt.

## MAIN landing review

Re-derive the three source blockers from the harness, verify the governor and
claim rows, confirm zero output/zero scorer work, preserve the null result
fields, and do not reinterpret this branch as a measurement or KKT solution.
