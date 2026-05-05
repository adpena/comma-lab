---
name: The Great Gradient Bug — Postmortem of the Competition-Changing Discovery
description: PoseNet gradients were DEAD in ALL TTO experiments due to upstream @torch.no_grad on rgb_to_yuv6. Discovered 2026-04-15 by skunkworks council.
type: project
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
## The Great Gradient Bug

### What happened
Every TTO (test-time optimization) experiment we ever ran — v1, v3, v4, all smoke tests,
all constrained generation, all sensitivity maps, all gradient rank analysis — had ZERO
PoseNet gradients. The optimizer was completely blind to PoseNet for the entire project.

### Root cause
The upstream `frame_utils.py:rgb_to_yuv6()` function (line 50) is decorated with
`@torch.no_grad()`. PoseNet's `preprocess_input()` calls this function. Any code that
backpropagates through PoseNet's preprocessing gets zero gradients at the YUV conversion
boundary. The autograd graph is silently detached.

### Why it wasn't caught sooner
1. The TRAINING pipeline had its own fix (`_patch_scorers_for_training` in training.py)
   that replaced `preprocess_input` with a differentiable version. So training worked.
2. The TTO pipeline loaded scorers through a DIFFERENT code path (`load_scorers()`) that
   didn't apply the patch. Two code paths, one patched, one not.
3. PoseNet loss still CHANGED during TTO — because SegNet gradients and compressibility
   gradients moved pixels, which incidentally affected PoseNet output. It LOOKED like
   PoseNet was being optimized, but it was random walk driven by other losses.
4. TTO v1 showed 6.3% improvement — we celebrated this as PoseNet optimization working.
   It was actually SegNet spillover.
5. The `@torch.no_grad` decorator is in UPSTREAM code we don't own — not in our codebase.
6. No gradient validation check existed until we added one.

### How it was discovered
The skunkworks council's adversarial review session. The Contrarian demanded: "if 50 steps
of gradient descent make PoseNet WORSE, something is fundamentally wrong." Hotz traced
the call chain and found the decorator. The council voted 13-0 to fix immediately.

### Impact
- Every TTO experiment was invalid (PoseNet optimization was blind)
- Every sensitivity map was zeros (gradient analysis was meaningless)
- Every gradient rank analysis was wrong (Jacobians were zero)
- PR #35 (tensor_inversion, score 0.75) likely had the same bug
- The "embedding loss failed" conclusion was invalid (both v1 and v3 had zero gradients)
- Weeks of GPU time wasted on blind optimization

### The fix
1. Extracted differentiable YUV conversion into `tac.scorer.make_scorers_differentiable()`
2. Created `load_differentiable_scorers()` convenience function (impossible to forget)
3. Added runtime gradient validation check in `coupled_trajectory_optimize()` (~1ms cost)
4. Deployed to ALL 8 experiment scripts that compute gradients
5. The bug class is now structurally impossible

### Novel aspects (paper-worthy)
- Demonstrates how a single `@torch.no_grad` decorator in an upstream dependency can
  silently invalidate an entire optimization pipeline
- The fix was invisible to naive testing (losses still changed, just not for the right reason)
- Only adversarial review with explicit "trace the gradient flow" caught it
- The runtime validation check pattern (1ms to prevent hours of waste) is generalizable
- Shows the danger of loading frozen models through different code paths with different patches

### Lessons
1. ALWAYS validate gradient flow end-to-end before trusting optimization results
2. A loss that changes is NOT proof that gradients are flowing correctly
3. Adversarial review > unit tests for this class of bug
4. Convenience functions that make bugs structurally impossible > documentation that says "remember to call X"
5. When upstream code has @torch.no_grad, grep for it before building gradient pipelines

### Timeline
- 2026-04-12: First TTO experiments (blind)
- 2026-04-14: TTO v1 "succeeds" with 6.3% (SegNet spillover)
- 2026-04-14: TTO v3 embedding loss "fails" (invalid comparison)
- 2026-04-15 ~00:30: Council discovers the bug
- 2026-04-15 ~00:45: Fix committed (10093136)
- 2026-04-15 ~01:00: Fix deployed to all scripts (74853e63, a6e31e92)
- 2026-04-15 ~01:XX: TTO v5 (first REAL PoseNet optimization) — pending
