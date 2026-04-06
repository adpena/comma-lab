# 2026-04-03 resolution sweep

Goal: test the next approved Track B queue against the promoted 3.62 CPU floor.

Reference promoted floor:
- preset: medium
- crf: 23
- scale: 512x384
- score: 3.62
- archive bytes: 2,819,374
- published README baseline: 4.39

Variants in this cycle:
1. 448x336
2. 512x384 (reused promoted reference)
3. 576x432

Scoring rule:
- current_workflow score is the scorer authority
- rule_faithful score is a local estimate only
