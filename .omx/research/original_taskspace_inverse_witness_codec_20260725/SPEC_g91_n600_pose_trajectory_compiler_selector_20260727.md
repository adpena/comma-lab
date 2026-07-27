# G91 — fresh n600 pose-trajectory compiler/selector

Date: 2026-07-27  
Lane: `lane_g91_n600_pose_trajectory_compiler_selector_20260727`  
Authority: `[macOS-CPU frozen-torch local research-signal]`  
Status: operator-aborted macro-dominated formulation; no candidate, public
runtime closure, upstream exact row, promotion, or pointer movement

## Outcome first

G91 compiled a fresh, exact-EOF XIP2 trajectory from the current 600x6 source
PoseNet target table, proved that this calibrated trajectory is strongly
low-rank, measured four fixed-warp treatments on all 600 pairs through the
actual frozen PoseNet, and built a byte-exact population selector over the G88
successor seam.

It did **not** solve inverse control. Source PoseNet output coordinates are not
renderer SE(3) controls. G16 already measured affine xi-to-PoseNet calibration
at `R^2=-0.215` and forbids this category error. G91's trajectory is therefore
initializer/factorization evidence only.

The full v3 prefix ZIP race was stopped by operator macro judgment with exit
130. This is not a runner failure. No v3 partial file was written, and stages
00–04 remain intact. The family was already decisively outside the competitive
cell:

```text
current G85/G88 d_seg                     0.0274712
100 * d_seg                               2.74712
unchanged-base rate term                  0.08615682166238399
zero-pose current-seam floor              2.833276821662384
competitive target                        0.172
```

No nonnegative `d_pose` can cross 0.172 on this exact seam. At the provisional
132,132-byte fixed-warp archive the zero-pose floor is even higher:
`2.835101275193939`.

On the more favorable operator teacher-seam comparison, the required pose basin
was approximately `d_pose <= 0.000474`. The best executable
single-treatment PASS/XIP2 per-pair oracle is `4.781669243334157`, 10,087.91x
too large. Even a non-executable oracle allowed to choose independently among
PASS and all four treatments for each pair bottoms out at
`0.9080311069153633`, 1,915.68x too large.

The exact frontier pointer is unchanged.

## What was built

The compiler accepts only the current float64 `[600,6]` target table. It:

1. applies the source-Pose calibration to produce fresh numeric xi;
2. takes a centered population SVD and a preregistered rank projection;
3. quantizes one canonical int16 `[600,6]` table with six fp32 scales;
4. races `none`, `delta_ar`, `spline_residual`, and `delta_res`;
5. requires every coder to parse to the same q/scales;
6. binds the exact XIP2 to the G85 member through the G88 operand;
7. races exact outer STORE and DEFLATE archives; and
8. measures PASS and received XIP2 decoded frames through frozen PoseNet.

The measured source trajectory is highly factorable:

```text
centered rank 1 explained energy   0.9986276536814037
centered rank 2 explained energy   0.9994494228162201
centered rank 3 explained energy   1.0
expanded int16 table + scales      7224 bytes
```

That is a compact initializer proof, not an inverse-renderer proof.

## Full n600 measured treatments

All numbers below are local frozen-PoseNet measurements on 600 pairs:

```text
PASS                              163.0613202823172
st0020 rank3 q4096                  5.744202348694083
st0044 rank2 q4096                  5.803632774771892
st0044 rank3 q4096                  5.8068364782709905
st0080 rank3 q4096                  4.7841469259991705
st0080 + exact PASS exceptions      4.781669243334157
```

For `st0080`, XIP2 is better on 597 pairs and PASS is better on pairs 313,
319, and 327. The per-pair minimum vector SHA-256 is
`117980dd7cfbffe5a128e5b59ea1f6c764ef2565b8f24243f394f2cd7994142e`.

The diagnostic all-treatment per-pair union has SHA-256
`f4c0c187cf9829dce573db0c799f82259cb5dccbbb1f59293261325560ea755c`.
It is deliberately labeled non-executable: a G88 operand carries one XIP2
trajectory, not a per-pair bank of four.

## Selector universe and adversarial hardening

For each treatment the implemented v3 selector covers:

```text
default XIP2, PASS exceptions   K=0..599   600 executable rows
default PASS, XIP2 activations  K=1..600   600 executable rows
unchanged base default PASS     K=0          1 decision row
```

Thus there are 1,200 executable G91 prefixes per treatment plus the exact base
decision. Both stable benefit orders include prefixes beyond the zero local
benefit boundary because DEFLATE price is non-additive. This is exhaustive over
prefix length, not a false claim about all `2^600` subsets.

The unchanged base wins every tie or regression. It is never represented by a
fake empty-XIP2 operand.

Adversarial review found and fixed two defects before sealing:

- the first implementation truncated each order at positive local benefit;
- the coder-row dataclass did not independently bind every claimed receipt
  field to its nested parsed bytes.

The landed implementation now binds XIP2, operand, successor member, STORE
archive, DEFLATE archive, selected archive, hashes, lengths, encoding, and
nested parse equality. Tests include an exact-zlib price counterexample whose
optimal prefix crosses the zero-benefit boundary, forged receipt-field
rejection, the 1,200-row universe, and the all-worse preserve-base decision.

The implementation is complete. The full four-treatment v3 byte race was not
completed because its answer cannot alter the macro frontier verdict.

## Exact provisional byte custody

Stage 04 predates the hardened v3 selector and is retained only as provisional
fixed-family evidence:

```text
archive.zip  132132 bytes  de16c751fc5f27dd46ab0606e263c44e70d4cbef71f917848cc2ade4ac059a28
member       136436 bytes  9e036102b7d4ce14a8a4a1a02723900cb46bf0bdd68f63e6c94de654e0306608
operand        2988 bytes  032db740818fab18241f909985aab4ea1e0582aaee479c79c7dd62746bae2727
XIP2           2714 bytes  0c04be92e3f2abac256f95e332cf1e420239f5a43f6b5b8b0b51722116ce0da0
```

Its local `d_pose=4.781669243334157` and combined formula value
`9.750062767013077`. It is not a v3 winner, public archive, exact upstream row,
or candidate.

The selected provisional bytes were decoded twice in bounded batches:

```text
batch count                         75
maximum pairs per batch              8
double decode every batch         true
exact Y1 preserved every batch    true
camera digest  0b55ce1670a63ff57d17beb76cf6e9c10258583d39435e15c5327096ccc6c195
Y1 digest      ffceeebc6fa71236bd7f2da13859abdf0721bd87b50d9ea0c737bef7fb280652
```

The exact unchanged base remains the preserved decision object:

```text
archive.zip  129392 bytes  b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd
member       133363 bytes  d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31
```

## G16 and G90 linkage

G16 is binding settled evidence, not background prose:

```text
affine xi-to-PoseNet R^2                       -0.215
inverse_control_solved                         false
direct_pose_target_as_warp_control_admissible  false
```

G90 was not serialized or used to order or admit G91. At the stop it had six
immutable batches and no aggregate/admission. It is a
`COARSE_FAMILY_COSTATE_ATLAS_ONLY`. Its exact Seg fields transfer only for each
exact measured Y1 byte pair; G91 preserves Y1, so they provide no pose actuator.
Its Pose fields and costates do not transfer across Y0/Y1 changes and are not
PoseNet-inverse decoder-control authority.

## The only admissible successor

G95 must:

1. freeze the final semantic Y1;
2. invoke actual frozen PoseNet in the optimization loop;
3. solve directly over received decoder-control coordinates;
4. use G91 only as a population initializer/factorization;
5. factor and quantize the solved controls after the inverse solve; and
6. bind the result through G94/public runtime before any promotion claim.

This is a new mechanism, not permission to reskin source Pose coordinates as
controls.

## Triality

DSL:

```text
source Pose6 -> calibrated xi -> centered rank-r -> int16 q + fp32 scales
q -> exact XIP2 coder -> strict G88 operand -> exact STORE/DEFLATE bytes
decoded pair -> frozen PoseNet -> per-pair d_pose
prefix allocation -> sqrt(10 * mean(d_pose)) + exact rate
```

DAG:

```text
exact G85/G88 base custody
  -> fresh source-target identity
  -> factorization + quantization
  -> exact XIP2/G88/ZIP parse-back
  -> full n600 PASS and four XIP2 measurements
  -> macro feasibility gate
  -> operator stop, preserve base
  -> G95 PoseNet-in-loop inverse solve
```

Equations:

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37545489

S_floor(base, d_pose=0)
  = 100*0.0274712 + 25*129392/37545489
  = 2.833276821662384
  > 0.172

best one-treatment prefix oracle d_pose
  = mean(min(PASS, st0080))
  = 4.781669243334157
```

## Open blockers and pointer honesty

- `G91_FIXED_SOURCE_POSE_WARP_FAMILY_MACRO_DOMINATED`
- `G16_AFFINE_XI_TO_POSENET_SETTLED_NEGATIVE`
- `G91_POSENET_IN_LOOP_INVERSE_CONTROL_SOLVE_OWED_AS_G95`
- `G91_PUBLIC_RUNTIME_AND_UPSTREAM_EXACT_N600_EVAL_OWED`
- `G88_PUBLIC_INFLATE_RUNTIME_GRAPH_LINK_OWED`

Pointer unchanged. This unit landed real full-n600 formulation evidence and a
hardened selector implementation, but it did not land a lower exact score.
