---
name: 2026-04-30 ~3:30pm CDT — FOURTH quota incident, recovery state inventory
description: Fourth quota cap of the day (resets 3:30pm — rolling window pattern). 3 agents killed mid-flight: #310 RECOVERY-AGENT-3, #311 Lightning bootstrap + IMP migration, #312 Azure dispatch wiring. User mandated "recover and respawn and proceed with all ensure no signal loss". HM-S investigation result CRITICAL: HM-S is HEALTHY, NOT STUCK — at epoch 504/600 (84%), ~2h ETA. Earlier "5+ days uptime" was HOST machine uptime bleed-through, NOT instance uptime.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Quota incidents today (4 total)

1. **3:30am CDT** (resets 4:50am) — 7 swarm agents killed mid-Phase-3
2. **7am CDT** (resets 10am) — 12+ subagents + Modal harvester retry-storm
3. **1:30pm CDT** (resets 1:30pm rolling) — 7 paradigm-shift agents
4. **3:30pm CDT** (resets 3:30pm rolling) — 3 recovery agents (#310, #311, #312)

Pattern: rolling 5h windows. Spawn 3+ heavy agents → quota burns within 60-120 min.

## HM-S investigation (CORRECTING earlier misread)

**Earlier claim**: "HM-S 5+ days uptime, suspicious, harvest-or-kill candidate"

**WRONG**. The "6 days, 3:09" string was from `uptime` command output = HOST machine uptime (Vast.ai bare-metal), NOT the instance/dispatch uptime.

**Truth (verified via SSH at 16:26 UTC)**:
- Process `train_segmap.py --variant kl_distill --arch segmap_homography` started at 06:59 UTC today
- Runtime: ~9h 27min
- Latest log line in `/workspace/pact/lane_hm_s_segmap_homography_results/train.log` at 16:25 UTC: `[train_segmap] epoch=504 loss=1.820433 seg=0.015667 pose=0.008181`
- **Progress: 504/600 epochs = 84%, ETA ~1.7 hours**
- Loss curve: 229 (epoch 36) → 1.82 (epoch 504) — monotonic decline, healthy
- PoseNet distortion: 0.006 (excellent — Lane G v3 anchor was 0.0034)
- SegNet distortion: 0.016 (converged-ish — Lane G v3 anchor was 0.004)
- GPU util: 12% at snapshot (between epochs / eval phase)
- Heartbeat file present, last updated 16:26 UTC (alive)
- No FATAL/ERROR markers in log

**Verdict: LET IT FINISH.** ETA ~2h, total cost ~$2.40, stage 5 contest_auth_eval will land [contest-CUDA] result.

This is exactly the kind of "hey is this stuck?" misread that costs sessions. **Lesson: always SSH for ground truth, never trust API uptime fields without context.**

## What's on Vast.ai right now (verified)

- 35885106 (HM-S, 84% epoch progress, $0.26/hr) — **HEALTHY, leave running**
- 35899850 (lane_19_logit_margin retry b_a4, 49% util, $0.26/hr) — running
- 35906669 (?, 80% util, $0.25/hr) — running, identity TBD
- 35907873 (?, 77% util, $0.27/hr) — running, identity TBD
- Vast credit: $55.57

## What's GONE from Vast.ai (lost)

- 35899275 IMP cycle 0 — never made it to cycle 1 checkpoint, $1-2 sunk

## Lightning.ai status

- SSH `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai` verified working
- Tesla T4 attached by default
- Need to switch to H100 via Lightning UI (or lightning_sdk if available)
- $240 annual credits available, $0 spent yet

## What was the dead agents' scope (so re-spawn knows)

### #310 RECOVERY-AGENT-3 (DIED mid-work)
Phase 1: Commit all orphan work in batches of 3-8 files (Pattern A nohup wrapped serializer)
Phase 2: Dispatch Lane PFP16 (~30 min on Vast.ai 4090)
Phase 3: Lane Pint12-PCA impl + dispatch (Lane GP forensic recommended)
Phase 4: SC++ V5 recovery (block_fp_codec tolerance fix + Stages 3-5 redispatch)
Phase 5: Wire Lightning.ai dispatch infrastructure
Phase 6: Stale Vast.ai cleanup + state update

What it likely got partially done:
- Some orphan commits landed
- Lightning wiring partial
- PFP16 / Pint12-PCA / SC++ recovery — likely DID NOT run (those are Phase 2-4 after Phase 1 commits)

### #311 Lightning bootstrap + IMP migration to H100 (DIED)
- SSH bootstrap of Studio
- Lightning dispatch infra (`src/tac/deploy/lightning/lightning_dispatch.py`)
- Lane 17 IMP migration: original plan was wait-for-Vast.ai-cycle-1, but Vast.ai 35899275 is GONE
- Pivot plan in `/tmp/imp_migration_pivot_for_311.md` (not delivered before quota)

### #312 Azure dispatch wiring (DIED)
- Wire `src/tac/deploy/azure/azure_dispatch.py` analog
- VM spot pattern (NC24ads_A100_v4 ~$1.10/hr spot)
- Infrastructure-only (no actual Azure dispatches)

## Critical findings preserved

- **Lane Ω-W-V2 stack = 1.07 [contest-CUDA] regression** (rate save -0.034 + PoseNet pay +0.052)
- **Lane G v3 = 1.05** [contest-CUDA] STILL THE FRONTIER
- **Lane GP v4 KILL** — pose dims 1-5 white-noise; **Lane PFP16 dominates**
- **Lane GP class HAS narrow merit** (#297) — Lane Pint12-PCA RECOMMEND IMPLEMENT NOW
- **All-scores forensic** — 60% of killed lanes are engineering bugs, not approach failures
- **Grand Council paradigm shift** — predicted 0.20 final-stack [prediction]
- **HM-S CURRENTLY TRAINING and HEALTHY** at epoch 504/600 — DO NOT KILL

## Re-spawn plan (LIGHTER pattern)

Lessons from 4 quota incidents:
- Spawn ONE long-scoped agent at a time (not 3+ in parallel)
- Build defensive checkpointing into agent prompts
- Use Pattern A nohup-wrapped commits (SIGURG-immune)

Plan:
1. Spawn ONE focused agent: orphan commits + IMP H100 dispatch + Lightning wiring (combined)
2. Wait for completion or crash
3. Then spawn next batch: Modal queued investigation OR Phase 2 GPU dispatches

## Cross-refs

- project_quota_incident_3_recovery_state_20260430_1330.md
- feedback_lightning_ai_ssh_credentials_20260430.md
- feedback_full_six_month_plan_aggressive_no_shortcuts_20260430.md
- feedback_no_monetary_commit_20260430.md
- feedback_priority_time_to_floor_with_final_approval_20260430.md
- /tmp/imp_migration_pivot_for_311.md (the pivot doc never delivered)
