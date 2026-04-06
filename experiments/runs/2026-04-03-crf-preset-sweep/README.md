# 2026-04-03 CRF/preset sweep

Goal: test the first approved Track B queue against the measured 4.06 CPU baseline.

Baseline:
- preset: medium
- crf: 22
- score: 4.06
- archive bytes: 3,735,828
- published README baseline: 4.39

Variants in this cycle:
1. medium / 21
2. medium / 23
3. slow / 22

Scoring rule:
- current_workflow score is the scorer authority
- rule_faithful score is a local estimate only
