---
name: Inflate Time Strategy — Pre-compute Heavy, Refine Light at Inflate
description: Pre-compute full TTO at compress time (unlimited), store results, do light refinement on hard pairs at inflate time within 5-12 min budget.
type: feedback
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
The optimal inflate strategy is: heavy compute at compress time, light refinement at inflate time.

**Why:** The 30-min eval budget leaves only 5-12 min for inflation (setup ~3-5 min, scoring ~15-20 min). Full constrained gen doesn't fit. But pre-computed TTO + light refinement DOES.

**How to apply:**
- Compress time (unlimited): run full TTO (500+ steps, hours on 4090) to get optimal frames
- Archive: renderer weights (~150KB) + pre-computed refinement hints (pose targets, hard-pair list)
- Inflate time (5-12 min): renderer forward pass (~2s) + load pre-computed TTO results OR light refinement on hardest 20% of pairs (120 pairs × 20 steps × 200ms = 480s = 8 min)
- The 80/20 rule: 80% of pairs converge quickly, 20% are hard. Only refine the hard ones at inflate time.
- Can still push sub-0.43: v5b/v5c experiments may show better configs, archive compression saves 0.02, inflate-time hard-pair refinement saves 0.02-0.05

**What NOT to do:**
- Don't try full constrained gen at inflate time on T4 (30+ min, doesn't fit)
- Don't assume 30 min = inflation budget (it's the ENTIRE job timeout)
- Don't store raw TTO frames in archive (707MB, way too large)
