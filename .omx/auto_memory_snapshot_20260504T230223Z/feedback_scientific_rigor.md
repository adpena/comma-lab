---
name: Scientific Rigor — Experiment Design, Contest Compliance, and Execution Standards
description: 100% scientific rigor, full contest rule compliance, pre-registered experiments, council design review. No shortcuts ever.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Experimental Process (Non-Negotiable)

### Before ANY experiment:
1. **Pre-registered hypothesis**: "We expect X because Y"
2. **Success criteria**: "If metric > threshold, PROMOTE"
3. **Kill criteria**: "If metric < threshold after N steps, KILL"
4. **Concern criteria**: "If metric is between thresholds, INVESTIGATE with deeper test"
5. **Council design review**: Yousfi + Fridrich sign off on config, resolution, step count, conditioning, initialization
6. **Resource estimate**: GPU hours, VRAM, expected runtime
7. **Replicability record**: all params saved BEFORE running

### During experiment:
8. **Monitor**: checkpoints at intervals, early stopping if diverging
9. **Log everything**: loss curves, per-component scores, timing, memory

### After experiment:
10. **Results record**: full score breakdown, comparison to baseline, archive size
11. **Council interpretation**: what does this MEAN? promote/kill/investigate?
12. **Decision**: documented, with reasoning, saved to experiment record

### Bias:
- Keep lanes OPEN until conclusively closed
- A negative result on an underspecified test means "test was wrong," not "technique is dead"
- Yousfi and Fridrich have final say as domain experts

### Quality bar:
- Experiments must be faithful to the actual proposed design
- No toy configs that can't produce conclusions
- Representative resolution, step count, conditioning, initialization
- Test the ACTUAL hypothesis, not a strawman

## Contest Compliance (Non-Negotiable)

Every score claim, every submission artifact, every pipeline step must be 100% compliant with contest rules.

- Every score must come from the actual contest evaluation pipeline (inflate.sh → evaluate.py), not proxy or bypassed auth eval
- Every archive must contain ALL artifacts used at inflate time (Yousfi PR #35 rule: scorer weights, models, masks — if your code loads it, your archive includes it)
- Every code path must be tested end-to-end through the contest-defined pipeline
- No reliance on "the eval environment probably has X" — if your code needs it, your archive provides it
- When rules are ambiguous, take the STRICTER interpretation
- Document every compliance decision with the specific rule it satisfies
