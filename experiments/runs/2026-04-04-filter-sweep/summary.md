# 2026-04-04 filter sweep summary

## goal
Evaluate the next approved Track B sweep around the promoted 3.56 CPU floor using the official scorer path.

## results

| variant | filters | current_workflow score | archive bytes | local rule_faithful estimate | verdict |
|---|---:|---:|---:|---:|---|
| lanczos/bicubic reference | lanczos / bicubic | 3.56 | 1,901,606 | 3.5617568220853544 | prior floor |
| bicubic/bicubic | bicubic / bicubic | 3.67 | 1,829,217 | 3.6692647636539166 | reject |
| lanczos/lanczos | lanczos / lanczos | 3.54 | 1,901,606 | 3.546277389901901 | promote |

## takeaways

- `bicubic/bicubic` cut bytes slightly but paid too much in distortion, producing a clear regression.
- `lanczos/lanczos` improved the promoted floor without changing bytes, suggesting the upscale kernel still matters at this resolution.
- Filter choice remains a real lever even after CRF, resolution, and GOP are tuned.

## decision

Promote `lanczos/lanczos` into `submissions/robust_current/config.env` and use it as the new Track B floor for the next cycle.

## promotion gate evidence

- Packaging succeeded for both new filter variants via `comma-lab eval-submission robust_current --package ...`.
- Inflation succeeded inside the upstream evaluator path before scoring.
- The scorer completed for both new variants on `cpu`, so the promoted result is backed by the full competition path rather than a proxy.
