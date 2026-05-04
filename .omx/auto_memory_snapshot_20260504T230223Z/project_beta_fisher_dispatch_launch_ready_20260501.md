---
name: β sensitivity-map Fisher dispatch — LAUNCH-READY operator one-liner ($2, ~30min Vast.ai 4090)
description: 2026-05-01 ~09:50 UTC. Pre-flight verified all 13 dispatch artifacts present locally. Lane G v3 anchor (694KB, sha 9b20bdfca246...) + Fisher script (327 lines) + helper experiments + upstream models + contest_auth_eval.py — every required file exists. Launcher accepts arbitrary --lane-script. One-liner ready for operator approval. This dispatch unblocks R7 OWv3 selection + Ω-W-V3 + Lane 12 NeRV alpha redesign.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Why this dispatch is the canonical Wave 1 unblock

Three downstream lanes are gated on per-channel Fisher sensitivity:

1. **OWv3 R7** — `experiments/sweep_owv3_byte_plan.py:455-544` `select_r7_pose_balanced_candidates` policy text: "Empty output means wait for component-balanced PoseNet/SegNet sensitivity rather than spending exact eval on another blind threshold." (See `project_owv3_r7_state_correction_20260501.md` for the R5+R6 failure context.)

2. **Ω-W-V3 sensitivity-weighted bit allocation** — `src/tac/owv3_sensitivity_weighted.py` currently uses uniform sensitivity prior; with Fisher weights, low-sensitivity channels get aggressive bit-budget cuts, high-sensitivity channels get protected → eliminates the R5+R6 failure mode (wrong-channel byte savings).

3. **Lane 12 NeRV α-redesign** — currently retired due to PoseNet collapse 49.78 [contest-CUDA]; redesign requires scorer-preserving objective which needs sensitivity-driven channel selection.

One $2 dispatch, three lanes unblocked.

## Pre-flight artifact verification (2026-05-01 ~09:50Z, all present)

```
✓ experiments/results/lane_g_v3_landed/iter_0/renderer.bin
✓ experiments/results/lane_g_v3_landed/iter_0/masks.mkv
✓ experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt
✓ experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip  (694074 bytes, sha 9b20bdfca246...)
✓ upstream/videos/0.mkv
✓ upstream/models/segnet.safetensors
✓ upstream/models/posenet.safetensors
✓ submissions/robust_current/inflate.sh
✓ submissions/robust_current/inflate_renderer.py
✓ experiments/profile_hessian_per_weight.py
✓ experiments/convert_fisher_to_owv3_sensitivity_map.py
✓ experiments/build_lane_g_v3_owv3_stack.py
✓ experiments/contest_auth_eval.py
```

The `scripts/remote_lane_g_v3_owv3_fisher_stack.sh` runbook (327 lines) executes:
- Stage 0: CUDA preflight (`torch.cuda.is_available()` required) + NVDEC probe
- Stage 1: Fisher per-weight profile via `profile_hessian_per_weight.py` (`--device cuda`)
- Stage 2: Fisher → OWV3 per-output-channel sensitivity map via `convert_fisher_to_owv3_sensitivity_map.py`
- Stage 3: OWV3-swapped Lane G v3 archive build via `build_lane_g_v3_owv3_stack.py`
- Stage 4: contest_auth_eval.py on exact archive bytes (`--device cuda`)

All 4 stages STRICT-fail-loud per CLAUDE.md non-negotiables (CUDA-required, NVDEC probe, expected-archive validation).

## Operator one-liner (β Fisher dispatch on Vast.ai 4090)

```bash
.venv/bin/python scripts/launch_lane_on_vastai.py full \
    --lane-script scripts/remote_lane_g_v3_owv3_fisher_stack.sh \
    --label lane_g_v3_owv3_fisher_beta \
    --anchor-dirs experiments/results/lane_g_v3_landed \
    --predicted-band 0.95 1.05 \
    --estimated-cost 2.00 \
    --council-priority 1 \
    --max-dph 0.30
```

Phase breakdown (per `launch_lane_on_vastai.py --help`):
- **phase1** (~10-30s): create Vast.ai 4090 instance + register to `.omx/state/vastai_active_instances.json`
- **phase2-wait** (~3-5 min): wait for instance SSH-ready
- **phase2-scp** (~2-3 min): build tarball (anchor 694KB + workspace) + SCP to remote
- **phase2-extract** (~30s): extract + CUDA probe on remote
- **phase2-launch** (~10s): subshell-detach the lane script
- **In-script execution** (~30 min): Stages 0-4 (Fisher profile + map convert + archive build + contest_auth_eval)

Total wall: ~35-40 min from `phase1` invocation to harvested `contest_auth_eval.json`.

## Cost accounting

- Vast.ai 4090 max-dph cap: $0.30/hr → 40 min ≈ $0.20 baseline
- Fisher profile + archive build dominate compute (~25 min) → $0.13
- contest_auth_eval (~10 min) → $0.05
- SCP + setup overhead (~5 min) → $0.02-0.03
- **Conservative cap: $2.00** (per `--estimated-cost`); typical actual: $0.20-0.40
- **Hard budget cap (per CLAUDE.md):** $24 of $25 Vast.ai credit budget. This dispatch consumes 0.8-1.6% of the budget.

## What the dispatch produces

Output directory: `experiments/results/lane_g_v3_owv3_fisher_lightning_20260430_codex_r2/` (or successor — script writes to `${LOG_DIR}/` which defaults to `${LANE_ID}_results`).

Key artifacts:
- `provenance.json` — full dispatch context (gpu_name, driver, knob values, anchor SHAs)
- `heartbeat.log` — every-N-second progress markers
- `owv3_sensitivity_map.pt` — the per-channel Fisher sensitivity (the foundational artifact for downstream lanes)
- `archive_lane_g_v3_owv3_fisher.zip` — the OWV3-swapped archive
- `contest_auth_eval.json` — score_recomputed_from_components, posenet/segnet/rate components
- `paired_pfp16_*` fields if `paired_pfp16_baseline_path` is supplied

## Predicted-score band

Per Shannon checkpoint score derivative:
- The Fisher dispatch produces a sensitivity map; the Ω-W-V3 archive built atop the same map is the score-bearing artifact
- Expected band: [0.95, 1.05] (better than R5+R6's 1.0374/1.0393 because Fisher correctly identifies high-PoseNet-sensitivity channels)
- True frontier (PFP16 A++): [1.037, 1.044] eval-noise band per `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md`
- Promotion threshold: candidate score < 1.037 (the LOW edge of paired PFP16 noise band) confirms sub-frontier

## What I CANNOT do without operator action

- Execute the one-liner (per CLAUDE.md "Executing actions with care" — confirm before $-spending; `--max-dph 0.30` flag is NOT a self-approval)
- Modify the budget cap or acquire additional Vast.ai credits
- Re-pair Lightning Studio SSH (separate operator action)

## What I CAN do this turn (non-blocked, surfaced for next /loop fire)

- Verify the harvest script exists and is wired (`tools/harvest_modal_calls.py` for Modal; for Vast.ai the `phase2-launch` writes `${LOG_DIR}` which is rsync'd back via `vastai copy` after the in-script run)
- Pre-stage operator approval flow: this memo IS the staging — the one-liner is now documented and can be copy-pasted
- Continue OWv3 design refinement: with R5+R6 failed, the next plan is **NOT** to dispatch β Fisher then immediately try owv3_0134 (most aggressive); it's to use Fisher to identify a SPECIFIC mid-aggressive candidate where the sensitivity prior says the OWV2-coded channels are PoseNet-low

## Adversarial Grand Council review

- **Shannon (LEAD):** Fisher diagonal IS the optimal local approximation to the loss curvature; using it for bit allocation is the textbook information-geometry approach. **APPROVE.**
- **Dykstra (CO-LEAD):** the Pareto frontier we want is on the rate-vs-distortion curve; Fisher gives us the exact direction-of-steepest-distortion. **APPROVE.**
- **Yousfi:** the foundational evidence-grade artifact for everything sensitivity-driven. Without it we're flying blind. **APPROVE.**
- **Fridrich:** PoseNet asymmetry is captured by Fisher — high-Fisher channels in PoseNet head get protection; low-Fisher channels in renderer body get aggressive cuts. **APPROVE.**
- **Contrarian:** "What if Fisher diagonal is the wrong sensitivity proxy?" — the alternative (full Hessian or block-diagonal) is 10-100x more expensive. The deferral note suggests the value of perfect sensitivity over Fisher diagonal is at most a few thousand bytes; Fisher is the right sweet spot. **APPROVE WITH NOTE:** if the dispatch lands and OWV3 still regresses, the next step is NOT bigger Fisher pipelines, it's a different paradigm (Lane 19 logit-margin or Lane 17 IMP).
- **Hotz:** "Just dispatch and see." This is the simplest unblock. **APPROVE.**

**VERDICT: 6/0 APPROVE — operator action required to execute.**

## Cross-refs

- `project_owv3_r7_state_correction_20260501.md` (the R5+R6 failure context that necessitates Fisher)
- `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md` (the frontier this targets)
- `project_shannon_floor_execution_state_checkpoint_20260501.md` (master plan; β Fisher is Wave 1 #1)
- `scripts/remote_lane_g_v3_owv3_fisher_stack.sh` (the dispatch script — verified all 13 required artifacts exist)
- `scripts/launch_lane_on_vastai.py` (the launcher; phase1 + phase2 invocations)
- `experiments/sweep_owv3_byte_plan.py:52-83` (R5+R6 failed-CUDA reference data)
