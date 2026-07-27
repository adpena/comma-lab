# Codex findings — G95 pair-0 full rank ladder

UTC: `2026-07-27T09:09:00Z`  
Lane: `lane_g95_population_pose_inverse_control_20260727`  
Verdict scope: exact G94 non-final Y1 with the static shared linear 48x64 chart

## Whole-object verdict

The real pair-0 ladder is a useful inverse-control oracle, not a candidate
direction to scale as-is:

| treatment | exact d_pose | Pose term | counted wire |
|---|---:|---:|---:|
| copy Y1 | 195.384920745 | 44.202366537 | 0 |
| rank 6 | 184.804071794 | 42.988844110 | 55,932 B |
| rank 12 | 113.500784318 | 33.689877459 | 111,288 B |
| rank 24 | 71.805344591 | 26.796519287 | 222,000 B |

At the live `0.172` target, even with zero Seg and zero rate, the entire n600
sum of per-pair Pose MSE may be at most `1.77504`. Rank 24 leaves pair 0 alone
at `71.805344591`, or `40.45x` that entire idealized population budget. This
falsifies blind rank-48/grid escalation for this exact conditioning state
without imposing an arbitrary independent Pose threshold.

The ladder nevertheless harvests real signal: rank 24 removes 63% of pair-0
MSE relative to copy-Y1, and all retained costates plus 42 exact direction
ablation rows are durably checkpointed. They are useful as a proposal
dictionary or teacher for a final-Y1-conditioned nonlinear/feature-modulated
carrier.

## Sharp edges

- Several rank-24 ablations improve exact d_pose, demonstrating that SVD order
  is not the optimal nonlinear subset after quantization. The ultimate carrier
  needs byte-aware subset arbitration and joint descent, not monotone rank.
- The LM dual solve emitted overflow/invalid matrix-multiply warnings when the
  6xR Gram became extremely small. The implementation now uses the
  algebraically equivalent damped SVD filter
  `V diag(s/(s^2 + lambda*s_max^2)) U^T r`, which avoids the unstable huge
  dual. A tiny-Jacobian regression and a direct-equivalence test were added.
- G101 subsequently closed the typed G88/G95 ownership/product seam. That does
  not rescue this chart: G95 must be refit after a substitutive final semantic
  Y1 exists.

## Decision

Stop rank 48, grid 64x96, and full-n600 fitting on the current G94 base. Preserve
all checkpoints. Build a materially better substitutive semantic Y1, then use
the typed G94 V2 product to fit exactly one conditional Y0 owner on that final
whole state. Perform byte-aware pruning/joint descent only after the complete
archive exists.

Pointer delta: none.

STORES CONSULTED: G95 raw receipt and all SSD stage checkpoints; live canonical
pointer custody embedded in the receipt; committed G94 V2 product; G95 runner,
receiver, and focused tests.

HISTORICAL_PROVENANCE: first adversarial classification of the real full
pair-0 G95 rank ladder; converts an apparent monotone local improvement into a
coupled frontier falsifier while retaining its costate signal.
