# Test Spec: cold-start dual-track verification and first measured progress

## Goal
Prove the repo can be trusted as a dual-track lab from cold start and produce the first scored result when prerequisites are already present.

## Checks

### 1. Environment readiness
- Command: `source .venv/bin/activate && comma-lab doctor`
- Pass: upstream root and snapshot exist; required binaries for the chosen execution path are reported.
- Record: tool availability, especially `ffmpeg`, and any external blocker.

### 2. Upstream snapshot integrity
- Command: compare `git rev-parse HEAD` plus SHA-256 digests for pinned files against `workspace/upstream_snapshot.json`.
- Pass: commit matches `ec82c291ffeae5212e9a38253791d58995518a80` and all pinned files match.
- Record: exact commit and per-file match table.

### 3. Explicit upstream sync
- Command: `source .venv/bin/activate && comma-lab install-submission exact_current && comma-lab install-submission robust_current`
- Pass: both upstream submission dirs exist and reflect the repo copies used for the current cycle.
- Record: sync command + destination paths.

### 4. Track A current-workflow smoke/eval
- Command: `source workspace/upstream/comma_video_compression_challenge/.venv/bin/activate && bash scripts/eval_exact_current.sh`
- Pass: upstream evaluator completes, returning a score or clearly logged failure mode.
- Record: packaging view `current_workflow`, score/failure, runtime note, archive size.

### 5. Track B package smoke
- Command: `bash scripts/package_robust_current.sh`
- Pass: `submissions/robust_current/archive.zip` is produced.
- Record: archive size, config, and post-package upstream re-sync evidence.

### 6. Track B first scored eval (conditional)
- Command: `source workspace/upstream/comma_video_compression_challenge/.venv/bin/activate && bash scripts/eval_robust_current.sh`
- Pass: evaluator completes **if prerequisites are already satisfied**; otherwise the blocker is explicitly recorded.
- Record: score, rate, runtime note, and the accounting label used in the report; or blocker evidence.

### 7. Durable-state persistence
- Pass: required state/report files are updated with the evidence and next 3 experiments.

## Regression/decision rules
- If Track A fails materially, demote it in the report and move optimization priority to Track B.
- If Track B packaging fails, fix only within allowed mutation frontier and rerun package smoke before any tuning.
- Do not promote a candidate without packaging + inflation + shape/frame-count + promising proxy/full eval evidence.
- Count package/archive measurements as cold-start progress; reserve “measured score” language for evaluator results.

## Next 3 experiments template
1. CRF/preset sweep around current x265 floor.
2. Resolution sweep for 448p/512p/576p-equivalent downscale targets.
3. Sparse residual or keyframe cadence variant only after the x265 floor is measured.
