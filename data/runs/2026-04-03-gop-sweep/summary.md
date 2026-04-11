# 2026-04-03 GOP sweep summary

## goal
Evaluate the next approved Track B sweep around the promoted 3.56 CPU floor using the official scorer path.

## results

| variant | keyint | current_workflow score | archive bytes | local rule_faithful estimate | verdict |
|---|---:|---:|---:|---:|---|
| keyint32 reference | 32 | 3.56 | 1,978,141 | 3.562551302986074 | prior floor |
| keyint24 | 24 | 3.64 | 2,018,006 | 3.646726000239659 | reject |
| keyint48 | 48 | 3.56 | 1,901,606 | 3.5617568220853544 | promote |
| keyint64 | 64 | 3.61 | 1,862,992 | 3.613706858354431 | reject |

## takeaways

- Shorter GOP at 24 hurt score enough to reject it.
- Longer GOP at 64 reduced bytes further but not enough to offset the distortion increase.
- `keyint 48` tied the best rounded score while reducing bytes enough to edge out the prior `keyint 32` floor.

## decision

Promote `keyint 48` into `submissions/robust_current/config.env` and use it as the new Track B floor for the next cycle.

## promotion gate evidence

- Packaging succeeded for all three GOP variants via `comma-lab eval-submission robust_current --package ...`.
- Inflation succeeded inside the upstream evaluator path before scoring.
- The scorer completed for all three variants on `cpu`, so the promoted result is backed by the full competition path rather than a proxy.
