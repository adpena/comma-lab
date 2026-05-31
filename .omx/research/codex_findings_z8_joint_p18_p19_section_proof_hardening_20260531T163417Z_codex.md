# Codex Findings: Z8 Joint P18/P19 + Section-Proof Hardening

UTC: 2026-05-31T16:34:17Z

## Findings

1. The prior Z8 runtime-custody surface had become too coarse after the WZ and
   stack-context pixel-driver fixes. A single
   `mamba_dreamer_wyner_ziv_pixel_consumption_proven` flag could not distinguish
   section-level receiver pixel consumption from full trained-renderer export or
   exact-axis score authority.

2. The 600-pair Z8 advisory result changes the binding-axis interpretation:
   distortion is already good enough to be useful, while rate dominates. The
   gradient/water-fill path must therefore be a rate-axis bit allocator for
   wavelet detail-band dead-zoning / coarsening, guarded jointly by P18/P19. A
   SegNet-only P18 weight can dead-zone PoseNet-sensitive pairs and reproduce the
   PR97-style tradeoff. The mathematically grounded score-space weight is:

   `w_i = 100 * abs(dL_seg/dx_i) + 5 / sqrt(10 * d_pose) * ||J_pose,i||_{Sigma^-1}`

   P18 supplies the dense SegNet argmax-flip boundary surface. P19 supplies the
   PoseNet null-subset and Mahalanobis / AIL pair weighting guard. Low joint
   weight atoms are eligible for rate attack; high joint weight atoms are
   distortion-protected.

## Landed Integration

- Added `tac.optimization.joint_p18_p19_waterfill` as the reusable P18/P19
  weight primitive, including `rate_attack_deadzone_mask` and
  `distortion_protect_mask` outputs for Z8 rate-bound wavelet quantization.
- Added Z8 joint-driver metadata requiring the P18/P19 contract and explicitly
  forbidding SegNet-only water-fill.
- Added per-section Z8 pixel-consumption proof records and routed the shared
  archive-bound contract through those records instead of relying only on the
  broad legacy flag.
- Added repair-cascade campaign metadata so P18 SegNet waterfill is paired with
  P19 PoseNet null/Mahalanobis guards before any budget spend can be considered.

## Authority

All new surfaces remain `[macOS-MLX research-signal]` / local acquisition
routing only. They do not grant score authority, promotion authority, budget
spend authority, or exact CPU/CUDA dispatch authority.
