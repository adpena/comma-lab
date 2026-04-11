# 2026-04-04 tiny resolution revisit summary

## goal
Evaluate a very narrow local resolution revisit around the promoted 3.54 floor using the official scorer path.

## results

| variant | scale | current_workflow score | archive bytes | local rule_faithful estimate | verdict |
|---|---:|---:|---:|---:|---|
| reference | 448x336 | 3.54 | 1,901,606 | 3.546277389901901 | prior floor |
| 432x324 | 432x324 | 3.33 | 1,781,129 | 3.330344634381239 | promote |
| 464x348 | 464x348 | 3.44 | 2,139,211 | 3.443342361041948 | reject |

## takeaways

- The tiny downward step to 432x324 is a real win.
- The tiny upward step to 464x348 loses clearly.
- The local optimum had not actually settled at 448x336; one more narrow resolution pass was worth it.

## decision

Promote `432x324 / medium / 23 / keyint48 / lanczos+lanczos` as the new Track B floor.
