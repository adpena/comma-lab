# 2026-04-04 BAT00 surrogate v2

Goal: see whether a less-naive BAT00 surrogate becomes useful once the labeled set is larger.

Changes from v1:
- remove rule-faithful-score leakage from the feature set
- keep archive bytes + config features only
- include a bias term in the linear model
- evaluate leave-one-out MAE and rank behavior

Still research-only.
