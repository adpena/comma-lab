# 2026-04-04 filter sweep

Goal: test the next approved Track B queue against the promoted 3.56 CPU floor.

Reference promoted floor:
- preset: medium
- crf: 23
- scale: 448x336
- keyint: 48
- bframes: 4
- ref: 4
- filters: lanczos downscale / bicubic upscale
- score: 3.56
- archive bytes: 1,901,606
- published README baseline: 4.39

Variants in this cycle:
1. lanczos / bicubic (reference)
2. bicubic / bicubic
3. lanczos / lanczos

Scoring rule:
- current_workflow score is the scorer authority
- rule_faithful score is a local estimate only
