# Full-Pipeline Self-Compression Nextwave Worker - 2026-05-03

Evidence grade: `empirical_planning_only`.

Score claim: false. Promotion eligible: false. Remote dispatch: none.

## Scope

Worker-local planning pass for deterministic, contest-faithful
self-compression, packing, and transcoding opportunities against the current
C089 frontier. This pass did not modify scorer code, did not use hidden
sidecars, did not build remote jobs, and did not open a dispatch claim because
no training/eval/remote-GPU job was launched.

New artifacts:

- planner:
  `experiments/plan_full_pipeline_self_compression_nextwave.py`
- tests:
  `src/tac/tests/test_plan_full_pipeline_self_compression_nextwave.py`
- result directory:
  `experiments/results/full_pipeline_self_compression_nextwave_worker_20260503/`
- plan JSON:
  `experiments/results/full_pipeline_self_compression_nextwave_worker_20260503/nextwave_plan.json`
- plan Markdown:
  `experiments/results/full_pipeline_self_compression_nextwave_worker_20260503/nextwave_plan.md`
- C089 byte profile:
  `experiments/results/full_pipeline_self_compression_nextwave_worker_20260503/c089_archive_bit_budget_profile.json`

## C089 Anchor

Source:

- archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- eval JSON:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/contest_auth_eval.adjudicated.json`
- archive bytes: `276342`
- archive SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- exact T4 recomputed score: `0.3154707273953505`
- SegNet: `0.00061038`
- PoseNet: `0.00049601`
- samples: `600`

Break-even math:

- target score: `0.314`
- current gap: `0.0014707273953504796`
- rate score per byte: `25 / 37545489 = 6.658589531221714e-7`
- strict byte-only savings needed at unchanged distortion: `2209`
- target archive bytes at unchanged distortion: `274133`
- score after saving `2209` bytes only: `0.3139998449679036`

## Byte Anatomy

Planner/profile anatomy for the current C089 single-member `p` archive:

| stream | bytes | generic nested savings | solo-crossing target |
| --- | ---: | ---: | ---: |
| `masks.mkv` | `219472` | `0` | `217263` |
| `renderer.bin` | `55965` | `0` | `53756` |
| `seg_tile_actions.bin` | `116` | `0` | impossible |
| `optimized_poses.qp1` | `677` | `0` | impossible |
| P6 payload header | `12` | `0` | impossible |
| ZIP/container overhead | `100` | `0` | impossible |

Interpretation: generic recompression is exhausted on every charged stream.
The only byte-only stream surfaces that can plausibly close `2209` bytes are
the mask stream and renderer stream. Action, pose, payload header, and ZIP
overhead are valuable polish surfaces only when stacked behind a larger move.

## Top 5 Build/Dispatch Recommendations

1. `renderer_trained_self_compression_c089_transplant`
   - Evidence: existing trained renderer export scan contains a byte-crossing
     candidate at `272986` bytes, `3356` bytes below C089.
   - Byte-only projected score if components were unchanged:
     `0.3132361047486725`.
   - Action: rebuild the best trained/self-compressed renderer candidate
     against the exact C089 P6/QP1 slices, then run renderer-transplant
     pose-safety and byte-closure preflights.
   - Dispatch: no dispatch from this worker. If C089-local transplant preflight
     passes, claim the lane before exact CUDA eval.

2. `renderer_zero_fp4_recovery_training`
   - Evidence: byte-crossing renderer shrink candidates exist; the strongest
     local candidate saves `4273` bytes versus C089-equivalent scale.
   - Blocker: raw shrink fails local renderer output parity, so exact eval of
     the raw candidate is not warranted.
   - Action: use the zero-FP4 masks as initialization for a short recovery/QAT
     pass, then rebuild only after pose-safety passes.
   - Dispatch: do not exact-eval the raw shrink candidate; treat it as a
     training/repair seed.

3. `mask_exact_lossless_transcoder_target_217263`
   - Evidence: mask stream dominates the archive at `219472` bytes, but generic
     recompression saves `0`.
   - Action: build a decoded-mask-lossless transcoder that proves the PR75 mask
     decoded SHA before packaging; target `<=217263` charged mask bytes if it
     must close the full target alone.
   - Dispatch: none until decoded-mask SHA parity, archive byte closure, and a
     non-noop payload proof exist.

4. `pr75_p6_action_dictionary_v2_micro_stack`
   - Evidence: the top parser-closed action candidate is not a no-op and is
     `1` byte smaller than C089, with planning estimate
     `0.31546827267490696`.
   - Limitation: after rate, it still needs more than `0.00147` component-score
     improvement to cross `0.314`.
   - Action: use only as a cheap component probe or stack member with a larger
     renderer/mask byte move.
   - Dispatch: local CUDA optional only; no remote dispatch as a standalone
     sub-`0.314` attempt.

5. `renderer_pose_safe_micro_shrink`
   - Evidence: largest pose-safe renderer shrink fixture saves `442` bytes.
   - Byte-only projected score: `0.3151764177380705`.
   - Action: keep as a regression fixture and stack only behind a larger byte
     move.
   - Dispatch: do not dispatch yet; safe but too small.

## Lower-Priority Packaging Base

`p6_lossless_stream_resweep_pack_base` remains useful as a decoded-stream
preserving packaging base, but not as a standalone frontier dispatch. The
available C088/C082-style lossless repack row is `276333` bytes, which would be
only `9` bytes below C089 if components were identical, and known exact evidence
did not supersede C089 by score. Use this only when stacking with a larger
not-noop transform.

## Verification

Commands run:

```bash
.venv/bin/python experiments/archive_bit_budget_profiler.py \
  experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip \
  --output-json experiments/results/full_pipeline_self_compression_nextwave_worker_20260503/c089_archive_bit_budget_profile.json \
  --output-md experiments/results/full_pipeline_self_compression_nextwave_worker_20260503/c089_archive_bit_budget_profile.md

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_full_pipeline_self_compression_nextwave.py -q

.venv/bin/python -m py_compile \
  experiments/plan_full_pipeline_self_compression_nextwave.py \
  src/tac/tests/test_plan_full_pipeline_self_compression_nextwave.py

.venv/bin/python experiments/plan_full_pipeline_self_compression_nextwave.py \
  --profile-json experiments/results/full_pipeline_self_compression_nextwave_worker_20260503/c089_archive_bit_budget_profile.json \
  --output-dir experiments/results/full_pipeline_self_compression_nextwave_worker_20260503
```

Results:

- focused tests: `4 passed`
- compile check: passed
- planner emitted deterministic JSON/Markdown with
  `score_claim=false`, `promotion_eligible=false`, and
  `remote_gpu_dispatch_performed=false`.

