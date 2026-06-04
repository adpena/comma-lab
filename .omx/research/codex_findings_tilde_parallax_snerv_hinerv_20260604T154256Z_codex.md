# Codex Findings: Tilde/Parallax Intake For SNeRV/HiNeRV

UTC: 2026-06-04T15:42:56Z

## Verdict

The crux is still upstream `evaluate.py`, not generic visual fidelity. Pact
should treat full PR95 fidelity plus exact evaluator fidelity as the control
arm to beat: `archive.zip` byte price, deterministic inflate runtime, SegNet
last-frame class disagreement, and PoseNet two-frame YUV6 pose MSE.

The xhigh sidecar found that Parallax is not an official `tilde-research`
video/parallax component. It is an LLM local-linear-attention repo
(`Yifei-Zuo/Parallax`) with Torch/Triton/Hopper runtime debt, so it is useful
only as concept-level optimizer/architecture co-design signal. It must not be
imported into receiver/runtime paths.

## Landed Integration

- Added `nerv_upstream_evaluate_priority_contract.v1` to the HiNeRV/SNeRV
  long-training campaign plan.
- Added row-local `nerv_row_upstream_evaluate_binding.v1` so each campaign row,
  experiment metadata row, and launch-authority contract carries the official
  scorer geometry:
  - SegNet direct weight is frame 1 only.
  - PoseNet direct weight is both frames.
  - Pose marginal is `5/sqrt(10*d_pose)`.
  - Rate price is `25/37_545_489` per `archive.zip` byte.
  - Inflated raw bytes are explicitly not the rate denominator.
- Added `nerv_tilde_oss_leverage_policy.v1` and row-local
  `nerv_row_tilde_oss_binding.v1`:
  - Aurora is allowed only as an optimizer timing/convergence smoke.
  - Wall Attention is allowed only as a Pact-native, byte-charged SNeRV LF/TUB
    temporal-gate inspiration.
  - Parallax direct runtime import is forbidden.
  - Direct Wall kernel import is forbidden.

## Next Work Orders

1. Reuse or refresh exact PR95 archive/runtime identity under current upstream
   `evaluate.py`, with CPU and CUDA axes kept separate.
2. Run the existing `aurora_like` timing-smoke row against PR95/HiNeRV matrices;
   do not grant archive or score authority from this local signal.
3. Build a Pact-native SNeRV Wall-style LF/TUB temporal gate side smoke only
   after source-forward/export authority is clean or the row is explicitly
   labeled side-smoke. Every gate parameter must be receiver-visible,
   byte-charged, SHA-recorded, and compared by frame-1 SegNet delta, two-frame
   PoseNet delta, and archive byte delta.
4. Delay any Parallax-like feature-grid probe for HiNeRV until official
   HiNeRV feature-grid/patch/trilinear/QuantNoise parity and measured section
   bytes are closed.

## Authority

This memo and the campaign plan are false-authority planning surfaces. They do
not claim score, rank, promotion, or exact-eval readiness. Promotion still
requires byte-closed `archive.zip` plus deterministic runtime through the exact
upstream evaluator axis.
