# L5-v2 Z6 PoseNet Ego-Proxy Probe

Date: 2026-05-17

Lane: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`

## Purpose

This ledger preserves the Z6 full-FiLM ego-conditioning semantic check after
the paired-control initialization fix. The question was whether a PoseNet-
derived ego proxy provides harder evidence than cheap controls before paid
Z6-v1 full-FiLM dispatch.

## Command

```bash
.venv/bin/python tools/probe_z6_real_video_ego_proxy_sweep.py \
  --video-path upstream/videos/0.mkv \
  --device cpu \
  --epochs 3 \
  --seed 0 \
  --include-posenet-proxy
```

## Evidence

- Artifact JSON: `.omx/research/l5_v2_z6_real_video_ego_proxy_sweep_20260516_codex.json`
- Artifact MD: `.omx/research/l5_v2_z6_real_video_ego_proxy_sweep_20260516_codex.md`
- Derived surface: `.omx/research/l5_v2_asymptotic_candidate_surface_20260516_codex.{json,md}`
- Evidence grade: `real_video_smoke_proxy_no_scorer`
- Score claim: `false`
- Promotion eligible: `false`
- Ready for paid dispatch: `false`

## Result

The full-FiLM Z6 arm still beats identity under matched shared initialization,
but the best proxy is `random_control`, not `posenet_pose`.

Key rows:

| proxy_id | full_film_proxy_wins | identity_minus_full_loss_proxy |
|---|---:|---:|
| `random_control` | `true` | `5.304813385009766e-06` |
| `posenet_pose` | `true` | `5.21540641784668e-06` |

The result is predictor-capacity liveness, not hard-earned ego semantics.

## Classification

Class: measured no-score proxy blocker.

Active blockers:

- `ego_proxy_semantics_not_hard_earned`
- `posenet_pose_proxy_not_best`

L5-v2 surface blocker:

- `z6_full_film_paid_dispatch_blocked_posenet_pose_proxy_not_best`

## Decision

Do not paid-dispatch Z6-v1 full-FiLM from this probe. The PoseNet-derived
compress-time proxy did not outperform random/zero-style controls, so the
semantic claim is not strong enough for promotion.

Valid next actions:

- Run a true scorer-bearing paired probe if Z6 remains the priority.
- Redesign the ego-conditioning objective before full_main.
- Advance Z7/Z8 only as new measured configurations, not as automatic Z6-v1
  promotion.

Reactivation criterion: a scorer-bearing or semantically stronger paired probe
must show the Z6 ego-conditioned arm winning for a semantic ego source, with the
same paired-control initialization discipline, before paid Z6-v1 full-FiLM
dispatch is reconsidered.
