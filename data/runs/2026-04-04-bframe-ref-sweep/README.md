# 2026-04-04 B-frame / ref sweep

Goal: test the next approved Track B queue against the promoted 3.54 CPU floor.

Reference promoted floor:
- preset: medium
- crf: 23
- scale: 448x336
- keyint: 48
- bframes: 4
- ref: 4
- filters: lanczos downscale / lanczos upscale
- score: 3.54
- archive bytes: 1,901,606
- published README baseline: 4.39

Variants in this cycle:
1. bframes 3 / ref 4
2. bframes 5 / ref 4
3. bframes 4 / ref 5

Scoring rule:
- current_workflow score is the scorer authority
- rule_faithful score is a local estimate only
