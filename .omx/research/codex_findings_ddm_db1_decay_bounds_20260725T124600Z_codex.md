---
schema: codex_findings_ddm_db1_decay_bounds.v1
date_utc: 2026-07-25T12:46:00Z
lane_id: ddm_db1_decay_bounds
delegation_checkpoint_key: codex_delegate:ddm_db1_decay_bounds:20260725T121605Z
research_only: true
execution_allowed: false
paid_dispatch: false
score_claim: false
pointer_moved: false
main_landing_review_required: true
gate: G2
verdict: INDETERMINATE
---

# DB1 — $0 descent-decay bounds before any burn

## One-line verdict

`INDETERMINATE`: the hash-bound V19C accepted-proposal analogue cannot reach
`d_seg=8.684e-4` within 450 more admissions, and the fixed SN1 boundary atlas cannot carry
W_seg-class to that target without boundary replenishment; neither artifact identifies the
transport/replenishment law or E7 residual entropy of live scorer-recursive descent.

`verdict_scope=FORMULATION: current V19C accepted proposal-order curve plus the fixed SN1/AT1
predicted-boundary atlas. V19C-instance in-horizon target reach is excluded. Live
scorer-recursive descent, boundary replenishment, E7 context-coded residuals, solve-derived
knee states, and broader description families remain open.`

The current card's E7 amendment binds: **do not REFUSE the campaign from the descent analogue
alone**. G2 remains blocking because the optimal stop `D*` is not identifiable from present
custody.

## Authority and stores consulted

- delegated authority file: 5,195 bytes, SHA-256
  `f93713114a5c1fc8befbc8ab2e2c9946aa3e9d14e6fc7e8a78deaed595aa047d`;
- binding card read from the `main` Git object without entering the main worktree:
  `.omx/research/optimal_start_card_366_refoundation_20260725.md`, 6,162 bytes,
  SHA-256 `8acf356a1ab8809f160fdf633403a36860bad6d281ee94e007b27466bcaae9cf`;
- MacKay/Pantheon position and op-routable 4:
  `.omx/research/feedback_grand_council_blind_spot_hunt_frontier_lowering_20260725.md`;
- V19C 104-admission curve:
  `.omx/research/ddm_v19c_correction_saturation_20260723T063500Z/stage_checkpoints/02_n600_saturation_curve.json`,
  SHA-256 `b23873f45ed001e9e02a54e0e5dc071b3374293d9fd081ff98a4037f76c8b979`;
- SN1 receipt and all 38 certified SSD margin shards; 38/38 byte counts and SHA-256s verified;
- tracked SN1 n600 telemetry, SHA-256
  `e8d26acd3b312e8213d04c48313b755de05e17268cc594e1d5b94927c443116b`;
- AT1 manifest, file SHA-256
  `251cc1e4268fb909a9f9a3ac2af845614c98aab15948f90d33d43a8c1542a1d9`,
  and canonical contraction payload SHA-256
  `046b6a9f80e510d627881682d9a3e4a3f5aa2ca4657bd4620bcf19257e4d2cad`;
- current exact W_joint history `[0,1,2,3,4]`, SHA-256
  `dca72cdc8c15a46be1a1d2e053bf813a79491d99c2793e57ec247f698be87e62`;
- canonical lane/task/equation state, latest sister memos, current directives, and both Codex
  inboxes through broadcast UTC `2026-07-24T23:09:25Z`.

No scorer, optimizer, evaluator, launcher, campaign, paid provider, or pointer mutation ran.
Quarantined/older rows were used as measured harvest signal only.

## 1. V19C decay fits

Primary data are all 104 admitted full-n600 curve rows. Both models have three fitted
parameters and non-negative physical asymptote:

| model | equation | exponent/rate | asymptote `c` | AIC | AICc | `d_seg` at admission `104+450` | target projection |
|---|---|---:|---:|---:|---:|---:|---|
| exponential | `c+a exp(-k n)` | `k=0.01827294057` | `0.02440933351` | `-1864.9806` | `-1864.7406` | `0.02440941333` | no finite crossing (`c > target`) |
| power | `c+a n^-p` | `p=0.01912548974` | boundary fit `0` | `-1798.5183` | `-1798.2783` | `0.02404140372` | `1.4217e78` total admissions |

`ΔAICc = AICc_power - AICc_exponential = +66.4624`, so the exponential is decisively
preferred inside this conditional model comparison.

The confidence interval is a deterministic 1,000-replicate circular moving-block residual
bootstrap (`block=8`, seeds `603366/603367`). It is **conditional on the finite observed
proposal ordering**; it does not cover unseen families, descent transport, or selection-process
uncertainty.

| model | asymptote 95% CI | horizon `d_seg` 95% CI | target behavior |
|---|---|---|---|
| exponential | `[0.02357781545, 0.02469934625]` | `[0.02359265423, 0.02469934696]` | 100% of replicates have asymptote above target |
| power | `[0, 0.02349579265]` | `[0.02385028444, 0.02450412844]` | 52.5% asymptote-above-target; finite subset has `log10(admissions)` CI `[66.2984, 89.9437]` |

Tail-window sensitivity does not rescue the horizon: windows of 70, 52, and 35 admissions give
fitted asymptotes from `0.02405655` to `0.02478698`, and every local-tail fit puts the target
below its asymptote. The full-curve power model is the only zero-asymptote fit, but its formal
crossing is roughly `10^78` admissions.

**Disposition:** `UNREACHABLE_IN_450_FOR_V19C_PROPOSAL_ORDER_FORMULATION`. This is not the G2
verdict for live descent.

## 2. SN1/AT1 margin mass `N(delta)`

The raw SN1 shards contain 2,569,387 ordered winner-neighbour boundary incidences. The tracked
telemetry contains 2,551,382 unique predicted-boundary pixels. Therefore the complete duplicate
budget is exactly 18,005 incidences.

Coordinates were not preserved in the shards, so exact unique `N(delta)` cannot be reconstructed.
The strongest lawful interval is:

`max(0, I(delta) - 18,005) <= N(delta) <= min(I(delta), 2,551,382)`.

AT1 converts each positive logit margin to exact linear-head distance
`d2 = margin / ||w_winner - w_rival||_2`.

| `delta` in AT1 `d2` | ordered incidences `I(delta)` | unique-pixel `N(delta)` interval | global-site fraction interval |
|---:|---:|---:|---:|
| `1e-4` | 1,087 | `[0, 1,087]` | `[0, 9.2146e-6]` |
| `1e-3` | 10,639 | `[0, 10,639]` | `[0, 9.0188e-5]` |
| `1e-2` | 105,922 | `[87,917, 105,922]` | `[7.4528e-4, 8.9791e-4]` |
| `0.1` | 1,005,354 | `[987,349, 1,005,354]` | `[0.00836986, 0.00852249]` |
| `0.2` | 1,835,430 | `[1,817,425, 1,835,430]` | `[0.01540650, 0.01555913]` |
| `0.5` | 2,560,864 | `[2,542,859, 2,551,382]` | `[0.02155608, 0.02162833]` |

At `delta=infinity`, the interval collapses exactly to 2,551,382.

### Opening-rate calibration

The first exact W_joint interval improves
`0.0705192311605 -> 0.0703088972304`, or 24,812 net errors. The current broadcast requires
this history to remain labelled `[naive-menu upper bound]`.

If every net correction came from the fixed boundary atlas, its effective `d2` lies only in
the order-statistic interval `[0.00233047, 0.00402171]`. Even under that hypothetical radius,
the one-step beneficial `d_seg` bound at W_joint, W_seg, and the card targets is
`[0, 0.0003629642]`: zero is the honest lower bound because the atlas lacks target-error
membership. The observed `0.0002103339` lies inside this capacity interval, but it does not
identify head displacement, gross beneficial/harmful flips, or a decay exponent.

The now-available four exact intervals contain only two strict Seg improvements:
`[0.07051923116, 0.07030889723, 0.07030889723, 0.07021567451, 0.07021567451]`.
They remain too short and proposal-source-confounded to identify a terminal decay law.

## 3. MacKay fixed-atlas integration and reconciliation

Integrating all unique mass in the **fixed initial boundary atlas** gives this oracle band:

| operating point | exact errors | all-beneficial fixed-atlas terminal `d_seg` band |
|---|---:|---:|
| W_joint | 8,318,787 | `[0.04889089796, 0.07051923116]` |
| W_seg-class (card) | 2,845,843 | `[0.00249617683, 0.02412451002]` |
| V19C terminal | 2,923,991 | `[0.00315864563, 0.02478697883]` |

The W_seg fixed-support oracle still leaves 294,461 errors, whereas the 130,789-byte card target
allows 102,441. Thus the complete initial boundary support is short by 192,020 beneficial pixels
even before collateral.

This reconciles rather than contradicts the V19C fit:

- V19C stops near `0.0248` after using only a small fraction of the available fixed margin mass,
  so its stop is proposal/Jacobian/receiver support limited, not explained by exhaustion of all
  near-flip pixels.
- The fixed-support oracle floor near `0.00250` is much lower than the V19C asymptote but still
  above the card target.
- Live descent can move margins and create new boundaries; therefore the fixed-atlas lower
  endpoint is **not** a universal descent floor.

MacKay's requested live integrated decay curve requires a state-transition kernel, not just one
static CDF. The current artifacts do not provide it.

## 4. Gate G2 and the one resolving measurement

**G2 verdict: `INDETERMINATE`.**

The cheapest resolving measurement is not a new arm. Augment the already-owed bounded J10
re-smoke: for 10 accepted exact n600 verdict intervals from the card-selected start, preserve
target-error-conditioned `(pair,y,x,winner,rival,margin,d2)` before/after rows, gross beneficial
and harmful flips, realized step seconds, and context-coded residual bits. One same-parent trace
then identifies both:

1. the live margin-transport/replenishment decay curve; and
2. E7's marginal equality between step cost and residual byte cost, hence optimal stop `D*`.

Until that receipt exists, `fire_authority=MAIN_ONLY_AFTER_REVIEW_AND_ALL_CARD_GATES`.

## 5. Triality and system intelligence

- **DSL/data:** `src/tac/analysis/ddm_db1_decay_bounds.py` and
  `tools/analyze_ddm_db1_decay_bounds.py` hash-verify all source shards, fit both model modes,
  emit conditional CIs, and refuse incidence-to-pixel coercion.
- **DAG:** `.omx/research/FEED-603-db1_20260725T124600Z.md` routes the indeterminate gate,
  E7 blocker, and single resolving measurement without authorizing a launch.
- **Equations:** exact fixed-atlas law
  `ddm_sn1_margin_mass_duplicate_budget_bounds_v1` is registered. V19C models remain
  receipt-local and unregistered.
- **Sensitivity map:** the `N(delta)` table may rank collateral sentinels after MAIN review;
  it is not an actuator or benefit predictor.
- **Pareto/bit allocator:** no measured residual byte price enters allocation because
  `H(flip-field | free decoder context)` is absent.
- **Cathedral/autopilot:** G2 stays blocking; no candidate, dispatch, promotion, score, or
  pointer edge is emitted.
- **Continual learning:** the exact duplicate-budget identity is a typed empirical equation
  anchor. No transferable descent posterior is updated.
- **Probe disambiguation:** both exponential and power families remain callable; the 10-interval
  same-parent trace is the registered arbitration evidence, not model preference alone.

## Durable receipt

Final receipt:
`.omx/research/ddm_db1_decay_bounds_20260725T124600Z/ddm_db1_decay_bounds_receipt.json`,
SHA-256 `f36fccd5e6fb1d072de01fb0ae90273398925559186d42a90ed964cb77345984`.

The earlier append-only preflight receipt at `20260725T124120Z`, SHA-256
`42f165fc0f222dc48f8ba8543f0e4f6e3a8409a9fe9206b53f7f9a88c5c14a7c`, has identical
analytical values but non-durable worktree labels. It is retained solely because the first
historical canonical-equation registration cites its exact bytes. The latest registration and
all operator-facing routes cite the path-normalized final receipt.

Verification:

- 5 focused tests pass;
- all five Python files compile;
- a fresh 1,000-bootstrap rerun is structurally identical to the final receipt after excluding
  process-id provenance;
- `git diff --check` passes;
- three clean review-tracker passes cover math/scope, custody, and reproducibility;
- the new lane is internally consistent at research-only L1 (`impl_complete` only).

Repository-wide `lane_maturity.py validate` still reports 110 historical missing-evidence paths
from unrelated lanes. None names `ddm_db1_decay_bounds`; this pre-existing global debt is not
silently relabelled as a DB1 failure or repaired from this delegated branch.

Pointer: `0.1910828242 [contest-CPU] UNMOVED`. MAIN must independently review the model domain,
E7 card amendment, equation registry events, hot-ledger merge, and serializer diff before landing.
