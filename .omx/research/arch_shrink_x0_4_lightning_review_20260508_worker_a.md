# Arch Shrink x0.4 Lightning Review - Worker A - 2026-05-08

generated_at_utc: 2026-05-08T06:18:01Z
lane_id: `arch_shrink_x0.4_lightning`
job_name: `arch-shrink-x0-4-lightning-20260508T024304Z`
worker: Worker A
scope: custody/review polling only; no relaunch; no harvester code edits

## Operator Constraints Applied

- Branch source of truth: `main`; current local HEAD observed:
  `27b7ed97b1f8bf490847021317aa486f7fea67ae`.
- Worktree was already dirty before this review. Relevant pre-existing dirty
  file: `reports/cathedral_autopilot_evidence.jsonl`.
- Write scope honored:
  - wrote this ledger only;
  - did not append `reports/cathedral_autopilot_evidence.jsonl` because no
    terminal exact CUDA JSON was harvested or parsed;
  - did not edit `.omx/state/lightning_active_jobs.json`;
  - did not edit `.omx/state/active_lane_dispatch_claims.md`;
  - did not edit harvester code.
- Existing arch-shrink harvester behavior noted: on terminal status it mutates
  dispatch claim state and active-job state. Because this review's write scope
  was narrower, status was checked via the harvester's SDK resolution helpers
  and remote artifacts were inspected read-only.

## Commands Run

```bash
git status --short --branch
sed -n '1,220p' AGENTS.md
sed -n '1,220p' CLAUDE.md
sed -n '1,180p' PROGRAM.md
rg -n "adversarial custody review|result-review|archive bytes|runtime-tree SHA|payload-consumption|CUDA auth eval|Evidence Grades|Lightning|dispatch claim|cathedral_autopilot_evidence" AGENTS.md CLAUDE.md PROGRAM.md
rg -n "arch_shrink_x0\.4_lightning|arch-shrink-x0-4-lightning-20260508T024304Z|arch-shrink-x0-4" .omx/state/active_lane_dispatch_claims.md .omx/state .omx/research reports -g '!reports/raw/**'
jq 'map(select((.job_name // .name // "") == "arch-shrink-x0-4-lightning-20260508T024304Z" or (.lane_id // "") == "arch_shrink_x0.4_lightning"))' .omx/state/lightning_batch_jobs.json
jq '.[] | select(.job_name=="arch-shrink-x0-4-lightning-20260508T024304Z")' .omx/state/lightning_active_jobs.json
sed -n '1,620p' experiments/arch_shrink_x0.4_lightning_harvest.py
sed -n '1,260p' src/tac/deploy/lightning/round3_harvest.py
LIGHTNING_DISABLE_VERSION_CHECK=1 .venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
mod_path = Path('experiments/arch_shrink_x0.4_lightning_harvest.py').resolve()
spec = importlib.util.spec_from_file_location('arch_harvest_status', mod_path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
job_name = 'arch-shrink-x0-4-lightning-20260508T024304Z'
job = mod._resolve_lightning_job(name=job_name, teamspace='comma-lab', user='adpena')
print({'job_name': job_name, 'sdk_name': getattr(job, 'name', None), 'status': mod._job_status_lower(job), 'raw_status': str(getattr(job, 'status', None))})
PY
ssh -o BatchMode=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes -o ConnectionAttempts=3 s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai '<remote read-only artifact inspection>'
date -u +%Y-%m-%dT%H:%M:%SZ
jq '{generated_at_utc, git_head, file_count, total_bytes, artifact_paths}' experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z/source_manifest.json
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short -- .omx/research/arch_shrink_x0_4_lightning_review_20260508_worker_a.md reports/cathedral_autopilot_evidence.jsonl .omx/state/lightning_active_jobs.json .omx/state/active_lane_dispatch_claims.md
```

## Local State

### `.omx/state/lightning_active_jobs.json`

- Row found for the target job.
- `submitted_at_utc`: `2026-05-08T02:43:10Z`
- `machine`: `g4dn.2xlarge`
- `profile`: `q_faithful_dilated_88k`
- `target_elements`: `88000`
- `evidence_tag_pending`: `[contest-CUDA]`
- expected local artifact dir:
  `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z`
- expected local archive:
  `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z/archive.zip`
- expected local auth-eval JSON:
  `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z/contest_auth_eval.json`
- `submit_result.name`: `arch-shrink-x0-4-lightning-20260508t024304z`
- `submit_result.status_at_submit`: `Pending`

### `.omx/state/lightning_batch_jobs.json`

- JSON shape is an array.
- No row matched this Studio-backed arch-shrink job. Current source of truth is
  `.omx/state/lightning_active_jobs.json`.

### Dispatch Claim

Active claim row remains open:

| submitted_at_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |
|---|---|---|---|---|---|---|---|
| 2026-05-08T02:43:09Z | claude_lab | arch_shrink_x0.4_lightning | lightning | arch-shrink-x0-4-lightning-20260508T024304Z | 2026-05-08T20:43:09Z | active_dispatching | FULL Lightning T4 train+archive+auth-eval via experiments/arch_shrink_x0.4_lightning_full.py 2026-05-08T02:43:09Z; force-claim: round-3 fix: train_renderer ego_flow+pose forward + force-stage optimized_poses.pt past EXCLUDED_PREFIXES; prior dispatch ended terminal failed_artifact_rsync_rc_23 |

No terminal claim row was appended in this review because the job is still
running.

## Provider / Remote Status

- SDK lookup via `experiments/arch_shrink_x0.4_lightning_harvest.py` helper:
  - SDK job name: `arch-shrink-x0-4-lightning-20260508t024304z`
  - status: `running`
  - raw status: `Running`
- SSH target used:
  `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`
- teamspace/user used for SDK lookup: `comma-lab` / `adpena`
- remote pact root used: `/teamspace/studios/this_studio/pact`

## Remote Artifact Inspection

Remote roots checked:

| root | exists | interpretation |
|---|---:|---|
| `/teamspace/studios/this_studio/pact/experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z` | no | live Studio checkout artifact path is absent from this SSH view |
| `/teamspace/jobs/arch-shrink-x0-4-lightning-20260508t024304z/artifacts/pact/experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z` | yes | persisted job artifact mirror contains live in-progress outputs |

Files observed in the persisted job artifact mirror:

| path | bytes | note |
|---|---:|---|
| `archive_masks_seed.provenance.json` | 1661 | Stage 1 provenance |
| `archive_masks_seed.zip` | 523851 | half-frame mask seed, not the final scored archive |
| `build_masks.log` | 936 | Stage 1 complete |
| `extracted/` | directory | seed extraction output |
| `heartbeat.log` | 15275 | current heartbeat |
| `provenance.json` | 580 | GPU/provenance |
| `run.log` | 375 | Stage 0/1/2 markers |
| `train/` | directory | training output dir |
| `train.log` | 3653 | training progress |

Key terminal artifacts were absent:

- `archive.zip`: missing
- `contest_auth_eval.json`: missing
- `auth_eval_work/contest_auth_eval.json`: missing
- `auth_eval.log`: missing

## Training / Runtime Evidence

- Stage 0 GPU check passed; provenance reports:
  - GPU: `Tesla T4`
  - driver: `580.126.09`
  - torch: `2.11.0+cu130`
  - CUDA: `13.0`
  - `cuda_available=true`
  - intended inflate wheel: `torch==2.5.1+cu124`
- Stage 1 half-frame mask seed completed:
  - decoded 1200 frames
  - extracted 1200 masks
  - kept 600 odd-indexed half-frame masks
  - `masks.mkv`: 246,549 bytes
  - seed archive: 523,851 bytes
- Stage 2 training is in progress:
  - profile: `q_faithful_dilated_88k`
  - model: `JointFrameGenerator`, 87,836 params
  - total schedule: 3000 epochs
  - latest train log line observed: epoch `120/3000`, Phase 1
  - latest epoch speed/ETA line: about `94.8s/ep`, ETA `75.8h`
- Latest heartbeat observed:
  - `2026-05-08T06:17:35Z`
  - GPU utilization: `100 %`
  - memory used: `3655 MiB`

## Result-Review Standard Status

No terminal result-review packet can be completed yet because no exact CUDA
`contest_auth_eval.json` or final `archive.zip` exists.

Current fields:

| required field | status |
|---|---|
| exact archive bytes/SHA | missing; no final `archive.zip` yet |
| runtime tree SHA | missing; no auth-eval JSON/inflate runtime manifest yet |
| command | in source manifest/remote command context; train stage in progress |
| hardware | provisional remote provenance says Tesla T4, CUDA available |
| sample count | missing; auth eval not reached |
| JSON/log paths | logs present; score JSON missing |
| score recomputation | not possible; component fields missing |
| dispatch claim state | active claim open, no terminal row |
| payload closure | not established; final archive/inflate path not emitted |
| failure class | none; job is running, not terminal |
| reactivation criteria | not applicable while running |

## Classification

Current status: `running_training_stage_2_no_score_artifact`.

Evidence grade: `in_progress / invalid for scoring`.

This is not a score, not a negative, not a promotion, and not a family
classification. The only admissible conclusion is that the active Lightning job
is still consuming GPU and has not reached final archive packaging or exact CUDA
auth eval.

## Next Action

Poll again later with the same job name and SSH target. If status becomes
terminal, harvest the persisted artifact mirror, then run strict custody review
before any evidence append:

1. verify final `archive.zip` bytes and SHA-256;
2. parse exact CUDA `contest_auth_eval.json`;
3. require CUDA/T4/full-sample/runtime-tree custody via
   `src/tac/deploy/lightning/round3_harvest.py`;
4. recompute score from components;
5. classify the result narrowly;
6. only append `reports/cathedral_autopilot_evidence.jsonl` if the terminal
   JSON passes exact evidence semantics.
