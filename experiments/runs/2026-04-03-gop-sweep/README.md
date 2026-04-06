# 2026-04-03 GOP sweep

Goal: test the next approved Track B queue against the promoted 3.56 CPU floor.

Reference promoted floor:
- preset: medium
- crf: 23
- scale: 448x336
- keyint: 32
- bframes: 4
- ref: 4
- score: 3.56
- archive bytes: 1,978,141
- published README baseline: 4.39

Variants in this cycle:
1. keyint 24
2. keyint 48
3. keyint 64

Scoring rule:
- current_workflow score is the scorer authority
- rule_faithful score is a local estimate only
