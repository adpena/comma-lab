---
name: Knowledge Distillation — Train Large, Deploy Small
description: User eureka: train a large renderer for quality, distill into a small one for rate. Standard teacher-student approach. Phase 2 optimization after paradigm validated.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Idea (user insight, 2026-04-12)

Train at much higher capacity (2M params, depth=2, channels 64/128) for maximum quality, then compress/distill into the deployment-sized model (276K params, depth=1, channels 36/60).

## Why It Makes Sense
- The large model learns the "right" pixel values with more capacity
- The small model learns to approximate the large model's outputs (teacher-student)
- This decouples training quality from deployment rate
- Quantizr may already do this ("slightly different architecture gets 10% better")

## Implementation Path
1. Validate paradigm at target size (276K params) — CURRENT PHASE
2. If paradigm works: train 2M-param teacher model
3. Distill teacher → student (276K params) using MSE on teacher outputs
4. Student learns from teacher outputs + scorer loss (multi-task)

## When to Apply
- AFTER the asymmetric warp paradigm is validated at the target model size
- NOT before — distillation adds complexity and we need the base paradigm to work first
- The teacher training costs 2-3x GPU hours but the student training is faster (warm start)

**Why:** The rate budget allows ~135KB at FP4. A 276K-param model at FP4 is ~135KB. But quality might benefit from training a larger model first. This is a standard technique in model compression.
