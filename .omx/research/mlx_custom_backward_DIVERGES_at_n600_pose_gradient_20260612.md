# NO-FAKE catch: the MLX custom grouped-conv backward DIVERGES at n600 — its n8 validation did not generalize (2026-06-12)

**Authority:** `[macOS-MLX research-signal]`, measured. Frontier UNMOVED 0.19109982. This is an APPEND to the
`lane_mlx_custom_grouped_backward_20260612` finding (the kernel is fast + d_seg-descent-equivalent at n8) —
NOT a mutation of it. It records a NEW failure mode discovered when the kernel was first used for a real
n600 training run.

## What happened

The custom MLX grouped/depthwise-conv backward (`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`, af4b4e2b, commits
c5e8ddda5/f0258ec45/b80013055) was validated at **n8 / 40 epochs**: gradient cosine ~1.0 vs the reference,
descent-equivalent on **d_seg** (reached 0.0107 vs torch 0.0110), 5.5× faster. On that basis it was wired
into the n600 basin-depth daemon (muon_throughout, mlx_gpu). The measured trajectory:

| epoch | d_seg | d_pose |
|---|---|---|
| 10 | 0.028 | 0.835 |
| 20 | 0.096 | 6.94 |
| 30 | **0.494** (→ random) | **36.46** (exploding) |

**DIVERGENCE, not oscillation.** d_seg climbs back toward init (0.5); d_pose explodes 0.8 → 7 → 36 — and
the **pose path blows up first and worst.**

## Root cause (diagnosed, not asserted-blindly)

1. **The n8 validation covered d_seg (SegNet) only — NOT d_pose (PoseNet).** The kernel is wired into the
   PoseNet adapters too, and the pose-specific explosion (the architecture STORES pose + FiLM, so d_pose
   should stay ~3e-4) points squarely at a wrong **PoseNet** gradient from the custom backward.
2. **n8 descent-equivalence does not generalize to n600.** A ~0.07% per-step gradient error that stays
   bounded over 40 n8 epochs compounds over n600's ~75× more steps/epoch into divergence.
3. Muon is a secondary suspect, but muon perturbs all params uniformly — the pose-SPECIFIC explosion
   implicates the kernel's pose gradient, not the optimizer.

## Why this matters (the NO-FAKE save)

A 5.5× kernel that diverges is WORSE than useless: had it run unwatched, it would have produced a fake
**"base_ch=20 cannot reach the basin"** capacity verdict that is really just a broken gradient — the exact
"surrogate-optimized-but-not-exact-authority-verified" fake-implementation class. Catching it on the d_pose
explosion (3 eval points) before it contaminated the verdict is the discipline working.

## Disposition

- **The basin measurement fell back to the CORRECT gradient: torch-CPU authority + faithful PR95 recipe**
  (`capstone_n600_correct_faithful_20260612T0101...`), fresh (the per-epoch checkpoint was the diverged
  state). Slower (~19 min/epoch) but trustworthy — it measures the architecture, not a kernel bug.
- **The custom backward is NOT n600-safe until its PoseNet/pose gradient is validated + fixed.** Reactivation
  for the kernel lane: (a) add a d_POSE descent-equivalence test (not just d_seg) at n8 AND at a larger n;
  (b) diagnose the strided-grouped pose-adapter gradient (the same stride×group indexing class the kernel
  already fixed for the forward — the backward may have a residual pose-shape bug); (c) re-validate a full
  n600 run stays bounded BEFORE using it for the basin measurement.
- **Operating rule reinforced:** only ONE mlx_gpu daemon at a time (two contend on the single Metal device),
  and never trust a throughput kernel for a real run until it is descent-validated on BOTH score terms at
  the real n.

## Bottom line

The "build the mlx thing we need" instinct was right and the kernel is real (fast + d_seg-correct at n8) —
but it has a pose-gradient bug that diverges at n600. The local-fast path is OUT until that's fixed; the
basin measurement continues on the correct (slow) gradient. Frontier UNMOVED.
