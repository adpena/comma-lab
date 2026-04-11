# 2026-04-04 B-frame / ref sweep summary

## goal
Evaluate the next approved Track B sweep around the promoted 3.54 CPU floor using the official scorer path.

## results

| variant | bframes | ref | current_workflow score | archive bytes | local rule_faithful estimate | verdict |
|---|---:|---:|---:|---:|---:|---|
| reference | 4 | 4 | 3.54 | 1,901,606 | 3.546277389901901 | prior floor |
| bframes3-ref4 | 3 | 4 | 3.57 | 2,021,782 | 3.568904302033391 | reject |
| bframes5-ref4 | 5 | 4 | 3.71 | 1,897,819 | 3.715850178661784 | reject |
| bframes4-ref5 | 4 | 5 | 3.55 | 1,894,366 | 3.5520846472034564 | reject |

## takeaways

- Reducing B-frames to 3 hurt score enough to reject it.
- Increasing B-frames to 5 was a clear regression.
- Raising refs to 5 came close, but did not beat the promoted floor.
- At this point, B-frame/ref tuning appears weaker than the earlier CRF, resolution, GOP, and filter levers.

## decision

Keep the promoted `448x336 / medium / 23 / keyint48 / lanczos+lanczos / bframes4 / ref4` floor unchanged.
