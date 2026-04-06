# 2026-04-03 resolution sweep summary

## goal
Evaluate the next approved Track B sweep around the promoted 3.62 CPU floor using the official scorer path.

## results

| variant | scale | current_workflow score | archive bytes | local rule_faithful estimate | verdict |
|---|---:|---:|---:|---:|---|
| 512x384 reference | 512x384 | 3.62 | 2,819,374 | 3.6183889310448754 | prior floor |
| 448x336 | 448x336 | 3.56 | 1,978,141 | 3.562551302986074 | promote |
| 576x432 | 576x432 | 4.26 | 3,868,552 | 4.263703892867531 | reject |

## takeaways

- Moving down to 448x336 improved score again by cutting bytes enough to outweigh the added distortion.
- Moving up to 576x432 regressed badly; the extra bitrate cost was not recovered by the distortion change.
- Resolution is a high-leverage honest knob in this scorer region.

## decision

Promote `448x336` into `submissions/robust_current/config.env` and use it as the new Track B floor for the next cycle.

## promotion gate evidence

- Packaging succeeded for both new resolution variants via `comma-lab eval-submission robust_current --package ...`.
- Inflation succeeded inside the upstream evaluator path before scoring, implying the expected raw output was produced and accepted by `evaluate.sh`.
- The scorer completed for both new variants on `cpu`, so the promoted result is backed by the full competition path rather than a proxy.
