---
name: GPU tiering strategy — A100 for long runs, 4090 for quick/non-controversial, H100 nuclear
description: Use A100 ($0.67/hr, 2x speed) for long-running training. 4090 ($0.27/hr) for quick experiments. H100 (~$2/hr, 3.5x) only for final nuclear run with validated winning config.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The Tiering

| GPU | $/hr | Speed | Use Case |
|-----|------|-------|----------|
| RTX 4090 | $0.27 | 1x | Quick experiments, smoke tests, non-controversial runs |
| A100 SXM4 | $0.67 | 2x | Long training runs (WILDE/SHIRAZ 1680ep), A/B tests |
| H100 SXM5 | ~$2.00 | 3.5x | Nuclear: final winning config, perfect training, deadline crunch |

## Why A100 for long runs

42h on 4090 = $11.30. 20h on A100 = $13.40. Only $2 more but results a FULL DAY earlier.
With 9 days to deadline, wall-clock time > dollar cost. A100 buys time.

## Why 4090 for quick experiments

Zoom optimization (30 min), smoke tests (1h), auth eval (30 min) — too short for
A100 premium to matter. 4090 is fine.

## Why H100 only for nuclear

H100 at $2/hr means $48 for a full run — exceeds budget unless we're confident.
Save for: "we found the winning config, train it perfectly one last time."
This is desperation/perfection land, not iteration land.

**How to apply:**
- Default to A100 for any run > 8 hours
- Default to 4090 for any run < 4 hours
- H100 only with explicit human approval for the final submission run
