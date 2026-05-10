# Phase A1 Best-Proxy Modal Harvest - 2026-05-09

## Result

The Modal best-proxy checkpoint-selection refire completed and was recovered.

- Lane: `track1_phase_a1_score_gradient`
- Job label: `track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex`
- Modal call id: `fc-01KR5MQ0WYS8TQWDN7HCYBZZ3B`
- Recovery stage: `completed`
- Recovery return code: `0`
- Recovery elapsed: `335.703398914 s`
- Harvested artifacts: `31`
- Dispatch claim terminal status: `completed_modal_recovered`

## Exact CUDA Evidence

- Evidence tag: `[contest-CUDA]`
- Score: `0.2263520234784395`
- Archive bytes: `178,262`
- Archive SHA-256: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- Runtime tree SHA-256: `6658b5749ccd62adb0d46b70fc02e4e984d3507f463b39f9e95a96971437615c`
- Samples: `600`
- SegNet distance: `0.00066299`
- PoseNet distance: `0.00017103`
- Rate term: `0.11869725`
- Auth eval JSON:
  `experiments/results/track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex/harvested_artifacts/eval_work/contest_auth_eval.json`
- Harvest summary:
  `experiments/results/track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex/harvest_summary.json`
- Best-proxy manifest:
  `experiments/results/track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex/harvested_artifacts/train/checkpoint_best_proxy_manifest.json`

## Checkpoint Selection

The selected checkpoint was `checkpoint_best_proxy.pt`, selected at epoch `29`
by minimum epoch-end `weighted_proxy=2.243459679643235`.

Selected proxy metrics:

- `seg_dist=0.022260800004005432`
- `pose_dist=3.0200593755580485e-05`
- `aux_kl=0.1955309957265854`
- `aux_pixel_l1=0.4824162423610687`
- `loss=2.921405553817749`

## Classification

This is a duplicate CUDA confirmation of the existing A1 latent-aligned archive,
not a new frontier or a reason for another CPU spend. The archive SHA-256 and
exact CUDA score match the current A1 paired-anchor archive. The runtime tree
SHA recorded here belongs to this Modal CUDA run; the earlier CPU evidence row
does not by itself prove identical runtime-tree custody.

- Existing paired `[contest-CPU]` Linux x86_64 score:
  `0.19284757743677347`
- Existing `[contest-CUDA]` Modal T4 score:
  `0.2263520234784395`

The selected best-proxy checkpoint converged to the same score-affecting
archive as the prior A1 anchor. This validates the checkpoint-selection
plumbing and closes the active dispatch, but it does not improve the score
frontier or upgrade A1 to submission-ready without the normal paired-axis and
policy gates.

## Decision

- Do not spend a fresh CPU eval on this archive; it is already paired.
- Do not relaunch this exact `lr=2e-6`, `kl=0.2`, `pixel_l1=0.01`, `40x8`,
  best-proxy configuration.
- Future A1 work should use true score-domain validation, SegNet-boundary
  validation, or q-bit-noise-in-training rather than another blind checkpoint
  selector in the same basin.

## Reactivation Criteria

Reopen A1 exact dispatch only if one of the following is true:

1. The training loop emits a byte-different archive with score-domain or
   SegNet-boundary validation evidence better than the current A1 anchor.
2. A retrain/fine-tune changes the runtime-consumed archive SHA-256 and passes
   the advisory collapse screen.
3. A new optimizer objective proves, with exact component evidence, that the
   selected checkpoint moves SegNet or PoseNet in the intended direction rather
   than only lowering the training proxy.
