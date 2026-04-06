# 2026-04-03 CRF/preset sweep summary

## goal
Evaluate the first approved Track B sweep around the 4.06 CPU floor using the official scorer path.

## results

| variant | preset | crf | current_workflow score | archive bytes | local rule_faithful estimate | verdict |
|---|---:|---:|---:|---:|---:|---|
| baseline | medium | 22 | 4.06 | 3,735,828 | 4.065705339595394 | prior floor |
| medium21 | medium | 21 | 4.74 | 5,005,390 | 4.740559607331766 | reject |
| medium23 | medium | 23 | 3.62 | 2,819,374 | 3.6183889310448754 | promote |
| slow22 | slow | 22 | 4.13 | 3,812,776 | 4.128791779915672 | reject |

## takeaways

- Lowering CRF from 22 to 21 was too expensive in bitrate; the score got worse.
- Raising CRF from 22 to 23 was a strong win: lower bytes dominated the modest distortion increase.
- `slow` at the old CRF did not beat `medium/23`, so preset complexity alone did not pay off here.

## decision

Promote `medium/23` into `submissions/robust_current/config.env` and use it as the new Track B floor for the next cycle.

## promotion gate evidence

- Packaging succeeded for all three sweep variants via `comma-lab eval-submission robust_current --package ...`.
- Inflation succeeded inside the upstream evaluator path before scoring, which implies the expected raw output was produced and accepted by `evaluate.sh`.
- The scorer completed for all three variants on `cpu`, so the promoted result is backed by the full competition path rather than a proxy.
