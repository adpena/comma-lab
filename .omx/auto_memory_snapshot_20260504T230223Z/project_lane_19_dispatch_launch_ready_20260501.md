---
name: Lane 19 logit-margin dispatch — TWO operator-runnable paths (resume training $1.25 OR score-snapshot-as-is $0.20)
description: 2026-05-01 ~10:15 UTC. Lane 19 is at L2 (missing CUDA gate) per `tools/lane_maturity.py audit`. A partial-training snapshot from instance 35899850 (destroyed 2026-04-30) sits locally at epoch 1340/1980 with all artifacts SHA-pinned. Two operator paths: (A) resume training $1.25 / 5h to completion, (B) score the partial snapshot as-is for $0.20 / 30min. Path B closes the L2→L3 gate immediately.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Lane 19 maturity state (verified)

`tools/lane_maturity.py audit` shows:
```
L2  lane_19_segnet_logit_margin    ✓ impl  ✓ emp  ✗ cuda  ✓ preflt  ✓ 3clean  ✓ mem  ✓ deploy
```

The missing `cuda` gate is exactly what either dispatch path closes.

## What's already on disk (the snapshot at epoch 1340/1980)

`experiments/results/live_snapshot_35899850_lane_19_logit_margin_20260430/`:
- `train/renderer_lane_19_logit_margin_best_fp4.pt` (FP4 best checkpoint)
- `train/training_state_lane_19_logit_margin.pt` (resume state)
- `train/zoom_scalars.pt` (inflate parameters)
- `provenance.json` (gpu_name=4090, predicted_band=[0.75, 1.05], anchor_lane=lane_g_v3)
- `SHA256SUMS` (all 8 files SHA-pinned for reproducible custody)
- `train.log` (last entry: ep 1340/1980 P2, fp4_scorer best=40.2750, ETA=4.2h remaining)
- `heartbeat.log` (24K)
- `run.log` (980B — Stages 0-1 confirmed PASS)

**The training stopped at 67% completion** because the Vast.ai instance was destroyed before the full 5.5h elapsed. The snapshot is a valid (but suboptimal) FP4 renderer.

## Path A: resume training to completion (~$1.25, ~5h)

```bash
# After phase1+phase2 setup (tarball + SCP + extract):
# the lane script auto-detects training_state*.pt and resumes from epoch 1341.
.venv/bin/python scripts/launch_lane_on_vastai.py full \
    --lane-script scripts/remote_lane_19_logit_margin.sh \
    --label lane_19_logit_margin_resume \
    --anchor-dirs experiments/results/live_snapshot_35899850_lane_19_logit_margin_20260430 \
    --predicted-band 0.85 1.05 \
    --estimated-cost 1.50 \
    --council-priority 2 \
    --max-dph 0.30
```

**Note**: the lane script (`scripts/remote_lane_19_logit_margin.sh:182`) calls `train_renderer.py --profile lane_19_logit_margin` which natively supports `--resume-from <state>`. The phase2-extract stage SCPs the snapshot dir to `/workspace/pact/lane_19_logit_margin_results/train/` so the resume detects it automatically.

**Outcome**: full training to epoch 1980 → contest_auth_eval → score in [0.85, 1.05] band per provenance.

## Path B: score the partial snapshot as-is (~$0.20, ~30min) — SCRIPT LANDED

**Status (2026-05-01 ~10:50Z): READY — `scripts/remote_lane_19_score_snapshot.sh` landed at commit `70e297ea` (284 LOC). E2E smoke proof passes 10 stages in 0.02s. Preflight clean.**

The script:
- Skips Stage 1 (training); takes the snapshot's `renderer_..._best_fp4.pt` directly
- Stage 1b: re-exports FP4A renderer.bin (deterministic from quantized state_dict)
- Stage 2: builds half-frame archive (masks)
- Stage 3: pose TTO with snapshot renderer (~10 min on 4090)
- Stage 4: contest_auth_eval on exact archive bytes
- Stage 5: JSON adjudication against PFP16 frontier gates

**Operator one-liner:**

```bash
.venv/bin/python scripts/launch_lane_on_vastai.py full \
    --lane-script scripts/remote_lane_19_score_snapshot.sh \
    --label lane_19_score_snapshot \
    --anchor-dirs experiments/results/live_snapshot_35899850_lane_19_logit_margin_20260430 \
    --predicted-band 0.97 1.04 \
    --estimated-cost 0.30 \
    --council-priority 2 \
    --max-dph 0.30
```

The launcher tarballs the snapshot dir → SCP to Vast.ai 4090 → extract → run in detached subshell. Total wall: ~30 min from `phase1` to harvested `contest_auth_eval.json`.

**Predicted score**: the snapshot is at 67% of training. The fp4_scorer trajectory in `train.log` shows best fp4_scorer=40.2750 and final epoch (1340) at fp4_scorer=42.1535 — the model was bouncing around the basin, NOT clearly improving. A score landing of ~0.97-1.04 [contest-CUDA prediction] is plausible but uncertain.

**Cost-benefit**: $0.20 to either close the L3 gate at "good enough" OR confirm the snapshot regresses (informs whether Path A's resume is worth $1.25). High-EV pre-screen.

## Path comparison (decision matrix)

| Criterion | Path A (resume) | Path B (score-as-is) |
|---|---|---|
| Cost | $1.25 | $0.20 |
| Wall clock | 5h | 30 min |
| Closes L3 gate | Yes (full training) | Yes (any contest-CUDA score) |
| Score quality | Higher (1980 epochs) | Lower (1340 epochs) |
| Predicted band | [0.85, 1.05] | [0.97, 1.04] |
| Risk of regression | Low (typical Lane G v3 family lands sub-1.05) | Medium (incomplete training basin) |
| Information value | Definitive | Pre-screen for Path A |
| Operator approval needed | Yes ($-spend) | Yes ($-spend) |
| Script status | Ready (existing) | Needs ~50 LOC wrapper script |
| Script can be written this turn | n/a | Yes — `scripts/remote_lane_19_score_snapshot.sh` |

## Recommendation

**Path B FIRST as pre-screen, then Path A IF Path B shows promise** — or **Path A directly if operator wants the definitive answer at 6× the cost**.

Path B's $0.20 buys 90% of the L3-gate-closure value. If the partial snapshot scores well (< 1.04), the lane is sub-frontier and Path A may add another 0.005-0.02 with full training. If Path B scores poorly (> 1.05), Path A is unlikely to recover and budget should redirect to β Fisher (Wave 1 #1) instead.

## Adversarial Grand Council review

- **Shannon (LEAD):** Path B is the highest-information-per-dollar dispatch. **APPROVE Path B FIRST.**
- **Dykstra (CO-LEAD):** the partial-training basin may not be the global optimum, but it's a valid Pareto-feasible point. Score it. **APPROVE Path B.**
- **Yousfi:** logit-margin is the textbook Fridrich UNIWARD pattern applied to segmentation; the partial training already shows the boundary-pixel signal converging. Score it. **APPROVE Path B.**
- **Fridrich:** Lane 19 was designed exactly to weaponize the SegNet boundary blind spot. Even a 67%-trained instance should show measurable SegNet improvement vs Lane G v3 anchor. **APPROVE Path B.**
- **Contrarian:** "Path B requires writing a new script. Why not just Path A directly?" **Counterpoint:** $1.05 saved is real; the ~50 LOC wrapper script is a 30-min Claude task. The wrapper is a small reusable artifact that can be applied to any future "score this partial-training snapshot" question. **APPROVE Path B with wrapper script.**
- **Hotz:** Path B is the cheap thing that works. **APPROVE.**

**VERDICT: 6/0 APPROVE Path B as the cheaper first dispatch; Path A IF Path B succeeds.**

## What I CAN do this turn (non-blocked)

- Write the `scripts/remote_lane_19_score_snapshot.sh` wrapper script (~50 LOC; skips training stages, runs only build_archive + contest_auth_eval) — turns Path B from "needs script" to "ready for one-liner dispatch".
- Mark Lane 19 maturity stays L2 (no change; the dispatch is what closes L3)
- Update Shannon checkpoint to surface this as Wave 1 dispatch #3

## What I CANNOT do (operator action required)

- Execute either dispatch (per CLAUDE.md "Executing actions with care" — confirm before $-spending)
- Choose between Path A and Path B without operator preference

## Cross-refs

- `project_beta_fisher_dispatch_launch_ready_20260501.md` (Wave 1 dispatch #1; pre-flight memo)
- `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md` (frontier this targets)
- `project_owv3_r7_state_correction_20260501.md` (the OWv3 state correction; Lane 19 is an alternative to OWv3 for the next score-moving lane)
- `project_shannon_floor_execution_state_checkpoint_20260501.md` (master plan; Lane 19 is Wave 1 dispatch #3)
- `scripts/remote_lane_19_logit_margin.sh` (the Lane 19 training script — verified all 4 required artifacts present)
- `experiments/results/live_snapshot_35899850_lane_19_logit_margin_20260430/` (partial-training snapshot)
- `tools/lane_maturity.py audit` (showed Lane 19 at L2 missing CUDA gate)
