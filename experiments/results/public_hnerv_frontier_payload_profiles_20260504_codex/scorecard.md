# HNeRV Frontier Scorecard

| label | grade | score | bytes | seg | pose | rate | largest section | archive sha |
|---|---:|---:|---:|---:|---:|---:|---|---|
| PR106x | A++ | 0.209451236806 | 186231 | 0.067142000 | 0.018305737 | 0.124003500 | `n/a` | `d25bca80057e8b53` |
| PR106 | A++ | 0.209456736806 | 186239 | 0.067142000 | 0.018305737 | 0.124009000 | `decoder_packed_brotli:170278` | `3fefbe5dfdd73817` |
| PR101 | A++ | 0.226353314440 | 178258 | 0.066304000 | 0.041354564 | 0.118694750 | `decoder_compact_brotli_streams:162164` | `b83bf3488625dbd7` |
| PR103 | A++ | 0.227764971422 | 178223 | 0.067623000 | 0.041470471 | 0.118671500 | `merged_range_coded_weights_and_hi_latents:153856` | `31881b2d23d027e6` |
| PR105x | A++ | 0.230431829870 | 177849 | 0.070456000 | 0.041553580 | 0.118422250 | `n/a` | `692a46931f66416a` |
| PR105 | A++ | 0.230437329870 | 177857 | 0.070456000 | 0.041553580 | 0.118427750 | `decoder_packed_brotli:161891` | `597ba0732810eba0` |

Interpretation: score truth remains the exact CUDA replay JSON. Payload
sections are forensic signals for the next compression action; they do
not imply score deltas without a new exact archive eval.
