# Hinton MLX Real-Teacher Refire, Corrected 0..255 Scale

Generated at: 2026-05-26T00:16:10Z
Lane: lane_hinton_mlx_real_teacher_refire_20260525
Evidence axis: macOS-MLX research-signal

## Verdict

The corrected real-SegNet teacher refire remains `SUB_PARADIGM`.

The clean-provenance 600-frame, 100-epoch run produced:

- initial loss: `3.4114346504211426`
- final loss: `3.0555782318115234`
- loss reduction: `10.431283465028084%`
- teacher provider: `real_segnet`
- teacher cache device: `cpu`
- effective student/teacher downsample: `1`
- real-teacher logits: `600 x 384 x 512 x 5`
- teacher-cache time: `204.75886487960815s`
- MLX training time: `4.279500961303711s`
- source video sha256: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

The report records clean repository provenance:

- git sha: `221a768053fdd254d71896220c1c4d41601d77dc`
- git branch: `main`
- git dirty: `false`
- command sha256: `c2de636d781ef24fea6962b9b66bbd67eda617126c3e8714ac1d8918c6e421b2`

## Supersession

This memo supersedes the pre-scale-correction real-teacher verdict in
`.omx/research/hinton_mlx_bundle_landed_20260525.md` for any decision that
depends on real-SegNet teacher convergence. The earlier `17.09%` real-teacher
loss-reduction number was produced before the 0..255 teacher-cache scale guard
landed and must remain historical scratch, not queue authority.

The corrected verdict does not kill the Hinton path. It says the exact current
student/loss/training setup is not yet queue-ready as a productive local
substrate trainer. The next step is optimizer/training-design work, not paid
dispatch or numpy parity promotion.

## Artifact

Local ignored result artifact:

`experiments/results/hinton_mlx_real_teacher_refire_0_255_20260526T0015Z/corrected_real_teacher_verdict.json`

The artifact is intentionally not a score claim and not a promotion signal:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Engineering Signals

The bottleneck is not MLX training throughput; the 100-epoch MLX stage took
about four seconds. The CPU real-SegNet teacher cache took about 205 seconds.
For future local substrate sweeps, cache reuse or persistent teacher-logit
artifacts should be first-class queue inputs so repeated optimizer/config
passes do not rebuild identical teacher logits.

Next concrete training-design probes:

1. Reuse the cached real-SegNet logits across longer 1000-epoch and 3000-epoch
   smokes.
2. Sweep learning rate, KL weight, temperature, and student-head capacity under
   the same clean-provenance report schema.
3. Add a queue-owned local MLX training surface that refuses to run real-teacher
   sweeps unless a teacher-cache artifact or explicit cache-build step is
   registered.
4. Keep all outputs on the macOS-MLX research-signal axis until paired CPU and
   CUDA auth eval anchors exist.

