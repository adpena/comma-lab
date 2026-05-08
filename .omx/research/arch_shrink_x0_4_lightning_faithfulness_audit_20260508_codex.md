# Arch Shrink x0.4 Lightning Faithfulness Audit - Codex - 2026-05-08

generated_at_utc: 2026-05-08T10:55:00Z
lane_id: `arch_shrink_x0.4_lightning`
active_job: `arch-shrink-x0-4-lightning-20260508T024304Z`
scope: adversarial half-frame / contest-faithfulness audit; no relaunch; no
terminal claim mutation; no score claim

## Current Status

- Harvester command:
  `.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --job-name arch-shrink-x0-4-lightning-20260508T024304Z --teamspace comma-lab --user adpena --ssh-target s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai --once`
- Harvester result: `status=running` at `2026-05-08T10:46:34Z`.
- Latest remote heartbeat observed: `2026-05-08T10:45:44Z`, GPU `100 %`,
  memory `3655 MiB`.
- Latest train log observed: epoch `290/3000`, Phase 1, about `95.0s/ep`,
  ETA `71.5h`.
- Terminal score artifacts remain absent: no final `archive.zip`, no
  `contest_auth_eval.json`, no `auth_eval.log`.

Classification: `running_training_stage_2_no_score_artifact`.
Evidence grade: `in_progress / invalid for scoring`.

## Half-Frame Audit

No real half-frame violation was found in the active telemetry.

Evidence:

- Stage 1 build log decoded `1200` frames, extracted masks shape
  `(1200, 384, 512)`, then kept odd-indexed half-frame masks with shape
  `(600, 384, 512)`.
- Encoded `masks.mkv` size is `246,549` bytes.
- Training log shows `hf_fires=150/150 (1.00)`,
  `hf_warp_diff` around `0.027`, and `hf_target_prob=1.000`.
- Training pose contract is active with
  `experiments/results/lane_a_landed/optimized_poses.pt` SHA-256
  `e0cd8ccb29bb3cd8613285e1633a6466f8a01f5df101d80243ec52d50d2fb85b`
  and shape `(600, 6)`.

The log string `NO motion, NO warp, single-mask + FiLM-pose` is expected for
Q-FAITHFUL. `JointFrameGenerator` consumes the odd-frame mask (`mask_t1`) plus
the deployed pose stream; it intentionally discards `mask_t`. For this renderer
family, lack of `zoom_scalars` is not itself a violation because the scored
runtime passes the duplicated/paired odd mask as `mask_t1`, and QFAI ignores
the duplicated `mask_t` fallback.

## Contest-Faithfulness Risks

- The active job was staged from the older launcher snapshot with source
  manifest SHA for `experiments/arch_shrink_x0.4_lightning_full.py`
  `e052cfe1bd1e957c1727e5bc3d7a43267f2ab6eb3b4cc2ceea7ed23be30f308c`.
  Current local HEAD has a later launcher with auth-eval environment and JSON
  capture fixes; changing it now does not change the active job.
- The active staged launcher does not export a job-local
  `UV_PROJECT_ENVIRONMENT` before `contest_auth_eval.py` and parses
  `RESULT_JSON` from the log instead of reading the structured work JSON.
  If the job reaches Stage 6, terminal evidence must therefore be accepted only
  after the harvester validates exact CUDA custody, archive bytes/SHA, device,
  sample count, runtime manifest, and structured JSON. If those checks fail,
  classify as `failed_invalid_auth_eval_custody` or a narrower infra failure,
  not as a method result.
- Runtime cap risk remains: at about `95s/epoch`, the 3000-epoch T4 job is
  unlikely to reach Stage 5 archive build and Stage 6 exact CUDA eval inside
  the active 18h cap. That is a scheduling/custody risk, not half-frame
  invalidity.

## Guard Added For Future Dispatch

Changed future launcher/test surfaces only; the active job is already staged
and running.

- `experiments/arch_shrink_x0.4_lightning_full.py` now decodes the extracted
  `masks.mkv` immediately after Stage 1 and fails closed unless it contains
  exactly `600` frames before training starts. The audit tries PyAV first and
  falls back to `ffprobe` so future dispatches do not depend on one Python
  video binding being installed.
- The launcher writes
  `arch_shrink_half_frame_contract.json` with `score_claim=false`,
  `promotion_eligible=false`, `runtime_consumes=mask_t1_plus_optimized_poses`,
  `zoom_scalars_required=false`, and the frame-count backend used.
- `src/tac/tests/test_lossy_coarsening_lightning_tools.py` pins this guard so
  future launcher edits cannot silently remove it or move it after training.

Reactivation / terminal criteria: harvest only the active job when terminal;
do not relaunch a duplicate while the claim is open. A score row can be
considered only after final `archive.zip` and exact CUDA `contest_auth_eval.json`
pass strict harvester custody validation.
