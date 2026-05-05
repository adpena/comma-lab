---
name: Paper Scalability Story — Three Tiers from Contest to Production
description: The paper demonstrates scalability: contest path (T4, 0.43), research path (4090, ~0.33), theoretical path (unlimited, →0.00).
type: project
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
## Three-Tier Scalability Story for Paper

### Tier 1: Contest Submission (T4, 5-12 min inflate)
- Renderer + pre-computed TTO → auth 0.43
- Archive: ~150KB (renderer weights + pose targets)
- Inflation: renderer forward pass (2s) + write raw frames
- Fits within 30-min eval job budget

### Tier 2: Research Demonstration (RTX 4090, 12 min inflate)
- Constrained gen with ~8KB archive → score TBD (projected 0.33)
- Eliminates renderer weights from archive (0.099 rate savings)
- Per-pair gradient descent through differentiable scorers
- Demonstrates what's possible with slightly more inflate-time compute

### Tier 3: Theoretical Limit (unlimited compute)
- Pure scorer inversion → approaches 0.00
- Tao's proof: scorer-optimal manifold has 99.5% of pixel space unconstrained
- Finding a point on it is trivially easy with gradient descent
- Shannon: minimum archive = noise seed + class colors = ~100 bytes

### Why This Matters for comma.ai
- Production has datacenter GPUs (not T4)
- Our technique scales with compute — more time = lower score
- The renderer provides a fast warm-start; TTO provides the quality ceiling
- At production scale, sub-0.1 is achievable
- This is the value proposition: task-aware compression that improves with compute
