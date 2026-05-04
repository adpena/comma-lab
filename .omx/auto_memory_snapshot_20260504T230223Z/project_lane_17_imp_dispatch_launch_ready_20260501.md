---
name: Lane 17 IMP cycle 0 dispatch — TWO operator paths (Vast.ai 4090 NOW vs Lightning L40S after GPU switch)
description: 2026-05-01 ~11:15 UTC. Lane 17 IMP cycle 0 = real verdict on the 88K-param sparse renderer (KILL retracted; current measurement is the 1.98 [contest-CUDA] STUB-LOOP bug). Pre-flight verified all 10 required artifacts present. Both `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` (Vast.ai 4090) and `scripts/lightning_lane_j_imp_iterative_magnitude_pruning.sh` (Lightning L40S) ready. Vast.ai path is OPERATIONAL NOW; Lightning gated on operator GPU mode switch. PCC1 dispatch swap (train_distill instead of stub) wired locally per fef1b61c.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Why Lane 17 cycle 0 is Wave 1 dispatch #2

Per Shannon checkpoint: "Lane 17 IMP cycle 0 with proper train_distill (~$1, ~30min on Lightning L40S after GPU switch): real verdict on the 88K-param sparse renderer. Currently held by the 1.98 [contest-CUDA] measurement bug retraction. Even one validated cycle gives KILL-or-promote signal worth the budget."

Per `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md`: the prior "1.98 score" was a STUB-LOOP — `stats.json: epochs=200, elapsed_sec=3.47` — the dispatch script's "in-script lightweight loop; deploy script swaps in train_distill" pattern was a stub that the swap never happened. KILL retracted; needs real cycle 0 with PCC1-enforced train_distill swap (landed at fef1b61c per `feedback_grand_council_imp_permanent_fix_review_20260430.md`).

If validated cycle 0 lands at score < 1.05, Lane 17 promotes. If > 1.10, kill confirmed. Anything in between is "promising; spend $25 for full 10-cycle".

## Pre-flight artifact verification (2026-05-01 ~11:15Z, all present)

```
✓ experiments/results/lane_g_v3_landed/iter_0/renderer.bin
✓ experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt
✓ experiments/results/lane_g_v3_landed/iter_0/masks.mkv
✓ experiments/train_imp_cycle.py
✓ experiments/train_distill.py  (the PCC1-required swap target)
✓ experiments/contest_auth_eval.py
✓ submissions/robust_current/inflate.sh
✓ upstream/videos/0.mkv
✓ upstream/models/segnet.safetensors
✓ upstream/models/posenet.safetensors
```

Both dispatch scripts:
- `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` (615 LOC) — Vast.ai 4090 path
- `scripts/lightning_lane_j_imp_iterative_magnitude_pruning.sh` — Lightning L40S path

Both have e2e smoke proofs in `.omx/state/lane_e2e_smoke_proofs.json`.

## Path A: Vast.ai 4090 — OPERATIONAL NOW

**Cycle 0 only** (full 10-cycle is $25/60h; cycle 0 alone is the validation gate):

```bash
# Cycle 0 cost: ~$1.65 / 6h on RTX 4090 ($0.25-0.30/hr).
# Set IMP_AUTH_EVAL_CYCLES=0 to score after cycle 0 only.
IMP_AUTH_EVAL_CYCLES="0" \
.venv/bin/python scripts/launch_lane_on_vastai.py full \
    --lane-script scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh \
    --label lane_17_imp_cycle_0 \
    --anchor-dirs experiments/results/lane_g_v3_landed \
    --predicted-band 0.95 1.10 \
    --estimated-cost 2.00 \
    --council-priority 1 \
    --max-dph 0.30
```

Wall: ~6.5h (200 epochs train_distill + ~30 min auth eval). Cost: ~$1.65.

## Path B: Lightning L40S — GATED on operator action

L40S is ~3× faster than 4090 for this workload. Cost: ~$1.30 for cycle 0 (~30 min wall).

**Operator action required**: Lightning Studio "Switch to GPU" button + select L40S/H100.

After GPU mode active, dispatch via:

```bash
# scripts/lightning_lane_j_imp_iterative_magnitude_pruning.sh runs in-Studio
# (no Vast.ai launcher; the script is invoked directly inside the Studio shell).
# Operator runs:
ssh <studio-alias> 'bash /teamspace/studios/this_studio/pact/scripts/lightning_lane_j_imp_iterative_magnitude_pruning.sh'
```

(See `reference_lightning_workspace_rename_20260501.md` for the exact Studio alias and mount paths after the comma-lab/lossy-compression-challenge rename.)

## Path comparison

| Criterion | Path A (Vast.ai 4090) | Path B (Lightning L40S) |
|---|---|---|
| Cost (cycle 0 only) | ~$1.65 | ~$1.30 |
| Wall clock | ~6.5h | ~30 min |
| Operator gates | $-spend approval | $-spend + Lightning GPU mode switch + SSH re-pair |
| Available NOW | Yes | No |
| Script status | LANDED + smoke proof | LANDED + smoke proof |
| L40S available | No | Yes (after switch) |
| Risk | Low (proven Vast.ai 4090 path) | Medium (Lightning SSH key denial issue per task #318) |

## Recommendation

**Path A (Vast.ai 4090) FIRST** because:
1. Available NOW (no operator-gated infrastructure)
2. ~$0.35 more expensive but $0 risk of Lightning SSH issues
3. The 6h wall is unattended (background dispatch)
4. Lightning path remains an option if operator wants the faster turnaround

If operator approves $-spend AND has Lightning Studio GPU mode active AND has re-paired SSH (3 conditions vs 1 for Path A), Path B saves ~6h wall.

## Adversarial Grand Council review

- **Shannon (LEAD):** the KILL retraction stands; cycle 0 with real training is the canonical test. **APPROVE Path A FIRST.**
- **Dykstra (CO-LEAD):** cycle 0 is the entry point of the iterative pruning sequence; without it, the whole 10-cycle path is blocked. **APPROVE.**
- **Yousfi:** the 88K-param sparse renderer is exactly the Quantizr-class architecture; if it converges to sub-frontier, this is the lane that beats 1.044. **APPROVE.**
- **Fridrich:** PoseNet preservation matters — IMP at 89% sparsity may eliminate critical PoseNet channels. Cycle 0 is the screening test. **APPROVE.**
- **Contrarian:** "What if Path A's 6h burns budget without producing useful signal?" Counterpoint: $1.65 for definitive KILL/PROMOTE/CONTINUE signal on the highest-EV lane in the queue. The information value is 1-2 orders of magnitude higher than the budget. **APPROVE.**
- **Hotz:** Path A is the cheap thing that works without infrastructure babysitting. **APPROVE.**

**VERDICT: 6/0 APPROVE Path A FIRST; Path B IF Lightning unblocks separately.**

## What I CAN do right now

- Verify both scripts are syntax-clean (already verified via existing smoke proofs in `.omx/state/lane_e2e_smoke_proofs.json`)
- Cross-check the `IMP_AUTH_EVAL_CYCLES="0"` env var works (verified at scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:217 — `IMP_AUTH_EVAL_CYCLES=${IMP_AUTH_EVAL_CYCLES:-"0 2 4 6 8 9"}`)
- Surface this as a launch-ready dispatch in the dispatch-readiness inventory

## What I CANNOT do (operator action required)

- Execute either dispatch (per CLAUDE.md "Executing actions with care")
- Re-pair Lightning SSH (separate operator UI action per task #318)
- Set Lightning Studio to GPU mode (operator UI action)

## Cross-refs

- `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` (KILL retraction; smoking gun was 3.47s elapsed for "200 epochs")
- `feedback_grand_council_imp_permanent_fix_review_20260430.md` (PCC1 train_distill swap landed at fef1b61c)
- `feedback_imp_local_backport_landed_20260430.md` (local backport of IMP fixes)
- `project_beta_fisher_dispatch_launch_ready_20260501.md` (Wave 1 dispatch #1 — not gated on Lightning)
- `project_lane_19_dispatch_launch_ready_20260501.md` (Wave 1 dispatch #3 — Path B uses partial snapshot)
- `project_shannon_floor_execution_state_checkpoint_20260501.md` (master plan)
- `reference_lightning_workspace_rename_20260501.md` (Lightning Studio aliases after rename)
- `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` (Vast.ai dispatch, 615 LOC)
- `scripts/lightning_lane_j_imp_iterative_magnitude_pruning.sh` (Lightning dispatch)
