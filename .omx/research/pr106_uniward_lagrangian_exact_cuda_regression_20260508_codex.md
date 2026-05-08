# PR106 UNIWARD-Lagrangian exact CUDA regression - 2026-05-08

## Artifact

- Lane: `pr106_uniward_lagrangian_runtime_packet`
- Job: `pr106-uniward-rms005-exact-20260508T083555Z`
- Archive: `experiments/results/lightning_batch/pr106-uniward-rms005-exact-20260508T083555Z/archive.zip`
- Archive bytes: `150511`
- Archive SHA-256: `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`
- Exact result: `experiments/results/lightning_batch/pr106-uniward-rms005-exact-20260508T083555Z/contest_auth_eval.adjudicated.json`
- Adjudication: `experiments/results/lightning_batch/pr106-uniward-rms005-exact-20260508T083555Z/adjudication_provenance.json`
- Hardware: Tesla T4, CUDA, `n_samples=600`

## Result

The rms=0.05 PR106 UNIWARD-Lagrangian runtime packet is an exact CUDA
regression:

- Score: `0.3371617511972341`
- Baseline score: `0.20454`
- Delta vs baseline: `+0.13262175119723413`
- SegNet distance: `0.0019625`
- PoseNet distance: `0.00016559`
- Archive delta: `-35728` bytes vs the 186239-byte PR106 source archive
- Adjudicated status: `REGRESSION_REVIEW_REQUIRED`
- Paper claim grade: `A-negative scoped forensic`
- Promotion eligible: `false`

## Interpretation

This does not kill UNIWARD, Lagrangian allocation, lossy coarsening, or PR106
runtime packets as families. It retires only this measured implementation and
configuration pending review: PR106 single-packet K-coarsening selected to
`rms_target=0.05` by tensor-space relative error.

The byte win was real, but the score did not map. The regression is consistent
with a scorer cliff: tensor-space RMS allowed distortions that SegNet noticed.
That makes the user's "pixel is the weight is deterministic" direction more
important, not less important. The next allocator should transport pixel or
boundary importance back through the deterministic decoder instead of trusting
global tensor RMS.

## Composition review

- Additive signal: this result gives an exact trust-region boundary for PR106
  lossy K-coarsening. It should calibrate future Jacobian, Fisher, boundary,
  and UNIWARD allocators even though this archive is not promotable.
- Antagonism: uniform tensor-space RMS at `0.05` is antagonistic with SegNet
  fidelity on PR106. Any stack that uses K-coarsening must constrain boundary
  or scorer-sensitive tensors before reclaiming the 35728-byte rate win.
- Orthogonality: entropy packer work remains orthogonal after a safe K vector
  exists. This exact negative says the distortion allocation failed, not that
  ZIP, brotli, range, or ANS packing failed.
- VStack rescue path: `score/boundary pullback -> protected K allocation ->
  no-dead-K pack -> entropy pack` is still open. Smaller targets such as
  `0.01`, `0.02`, and `0.03` should be considered only after breakeven math
  and finite-difference calibration.
- HStack status: PR106 is a monolithic archive, so HStack means logical
  tensor/section routing inside one packed payload, not separate mask/pose
  files. Do not assume an external mask budget.

This measured archive/config is regression evidence. It is not a broad family
kill, and it should remain as a calibration/rescue artifact for stack design.

## Immediate design consequence

The next PR106 lossy allocator should not spend another exact eval on uniform
RMS targets without a scorer-aware/boundary-aware trust-region check.

Required next experiment class:

1. Build per-weight or per-tensor importance from a pixel-space target:
   boundary mass, component-response sensitivity, Fisher map, or UNIWARD-like
   spatial inverse variance.
2. Pull that importance back through the HNeRV decoder using vector-Jacobian
   products or blockwise finite differences. Avoid materializing the full
   pixel-by-weight Jacobian.
3. Feed the resulting weights into the existing Lagrangian K allocator and
   sweep tighter trust regions before dispatch.
4. Exact CUDA dispatch only after the candidate has byte closure, payload-change
   proof, and an explicit "not this rms=0.05 cliff" diagnostic.

## Claim closure

The dispatch claim was closed with:

`completed_regression_review_required`

Notes: exact CUDA T4 harvested; score `0.3371617511972341`, seg
`0.0019625`, pose `0.00016559`, bytes `150511`, SHA
`0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`;
A-negative scoped to rms=0.05 PR106 UNIWARD runtime packet; no promotion and
no family kill.
