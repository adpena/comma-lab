# FEC6 Writeup Pose-Marginal Correction - 2026-05-17

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- touched partner WIP with a surgical math-only correction:
  - `docs/pr_writeups/cpu_frontier_fec6_20260517.md`

## Trigger

The latest partner WIP plan
`.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md`
explicitly routed op-routable #7: fix the FEC6 writeup's pose marginal from
`922` to about `292`. Live inspection showed the writeup still contained the
stale value in three places.

Adversarial review also found a second-order issue below the headline typo:
after correcting the marginal, the writeup's "spend 1,000 bytes to reduce
`d_pose` by `1e-6`" example is not a net score win.

## Exact Correction

FEC6 CPU operating point:

```text
d_pose = 0.00002943271901344679
pose_term = sqrt(10 * d_pose) = 0.017155966604492673
dS/d(d_pose) = 5 / sqrt(10 * d_pose) = 291.4437941777433
pose/seg marginal ratio = 2.9144379417774333
pose/rate marginal ratio = 437695990.736749
```

So the writeup now says `291.44`, `2.91x`, and `4.38 x 10^8`, not `922`,
`9.22x`, and `1.4 x 10^9`.

The finite-change example:

```text
added_bytes = 1000
byte_score_cost = 25 * 1000 / 37,545,489 = 0.0006658589531221713
pose_delta = 1e-6
pose_score_saving =
  sqrt(10 * d_pose) - sqrt(10 * (d_pose - 1e-6))
  = 0.00029396227124486515
net_score_delta = +0.00037189668187730617  # worse, not better
exact break-even pose_delta for 1000 bytes = 2.24035397806799e-6
```

Therefore the corrected writeup states that the specific `1e-6` / `1000 byte`
trade is not worth taking; a packet must either reduce pose by at least about
`2.24e-6` for those 1000 bytes or find a cheaper byte encoding.

## Code-Backed Guard

Reusable helpers added:

- `src/tac/score_geometry.py::pose_score_saving_from_delta`
- `src/tac/score_geometry.py::pose_byte_tradeoff`
- `src/tac/score_geometry.py::PoseByteTradeoff`

Focused tests:

- `src/tac/tests/test_score_geometry.py::test_pose_byte_tradeoff_bounds_fec6_writeup_example`
- `src/tac/tests/test_score_geometry.py::test_pose_score_saving_from_delta_matches_exact_sqrt_formula`

This converts the correction from prose into a reusable closed-form audit for
future FEC6/L5/Rule #6 decisions that spend bytes to move PoseNet.

## Dispatch Consequence

The qualitative conclusion still holds: FEC6 is in a pose-sensitive CPU basin.
The corrected quantitative consequence is sharper:

- pose moves are valuable;
- byte spend is not free;
- marginal intuition must be checked against finite-change arithmetic before a
  packet is called net score-lowering.

For the next build queue, this means PR101/FEC6 bolt-ons should carry an exact
pose-vs-byte tradeoff row before dispatch. The helper above is the canonical
calculation for that row.
