# SPDX-License-Identifier: MIT
"""Dead code from the comma.ai video compression challenge.

These modules were conclusively killed during competition research:
- adaptive.py: Hinton T² correction was vacuous (T² cancels in derivation)
- scorer_distill.py: KL distillation caused PoseNet collapse (confirmed 2x)
- test_eureka_smoke.py: One-off smoke tests, superseded by proper experiments

Preserved for reference and potential use in other contexts.
"""
