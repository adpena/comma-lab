---
name: 2026-04-28 session recovery from molt-OOM crash — full state map
description: Computer crashed during overnight wave (molt repo OOM, not pact). All local subagents died; Vast.ai survived. Full audit of what landed pre-crash, what was respawned, what's still in flight. Critical: 8 council EUREKA lanes + 3 new high-EV lanes (SAUG/MAE-V/HF) all dispatched for full implementation + recursive review.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Crash context (NOT us)

Computer crashed at ~12:00 UTC due to **molt repo OOM** (separate skunkworks compilation pipeline running concurrent to pact). NO instability in pact code. All Vast.ai instances continued training (crash-immune by design).

## Surviving artifacts on disk (recovered intact)

### Clean landings from pre-crash subagents
- `src/tac/pose_gaussian_process.py` (122 lines) — Lane GP module, tests pass
- `src/tac/raft_pose.py` (142 lines) — Lane FL module, tests pass
- `experiments/fit_pose_gp.py` — Lane GP compress-time tool
- `experiments/derive_poses_from_raft.py` — Lane FL compress-time tool
- 13 test files for Lanes GP/FL/SG/GE/HM/CG/DI/SI-V3/EC

### Partial landings (mid-rename / mid-implementation when crashed)
- `src/tac/learnable_bit_quant.py` — Round 10 fixer water-fill bracket fix incomplete
- `src/tac/learnable_class_weights.py` — Round 10 dual-ascent migration incomplete (17 broken tests)
- `src/tac/learnable_pair_weights.py` — same
- `src/tac/se3.py` — SE(3) left-Jacobian fix in progress
- 5 test files passing, 17 failing

### Vast.ai instances (8 active, all training)
| ID | Lane | GPU util | Duration |
|----|------|----------|----------|
| 35733155 | lane_g_v3_overnight | 81% | 13.7K s |
| 35733831 | lane_i_overnight | 34% | 12.7K s |
| 35733832 | lane_v_overnight | 47% | 12.7K s |
| 35736027 | lane_m_v2_overnight_v2 | running | running |
| 35739770 | lane_w_iceland | running | running |
| 35739771 | lane_k_denmark | running | running |
| 35739773 | lane_os_v2_nc | running | running |
| (8th) | (additional) | running | running |

## Recovery actions taken (2026-04-28 ~12:30 UTC)

### Subagents respawned (all running in background)
1. **Round 10 fixer** (codex:rescue, agentId a0becfe306f1f0db2) — finishing 5 math bugs (Lane W weight-suppression, PS class-equalization, SE(3) translation, bit STE upstream sign, water-fill bracket). Codex CLI session id `019dd3d8-095f-7572-b08e-2498a01740c3` runs independently.
2. **Lane GE+HM+CG+DI+SI-V3 source modules** (codex:rescue, agentId affb8c86aca06fb81) — **LANDED**: 5 modules created, 29 tests pass.
3. **Cosmos+MAE+Lyra+Telescope research** (general-purpose, agentId aedc72ed5f91624ea) — **LANDED**: 384-line synthesis at `.omx/research/cosmos_mae_2604_telescope_synthesis.md`. Proposed 6 new lanes; 3 high-EV (SAUG/MAE-V/HF).
4. **Lane SAUG implementation** (codex:rescue, agentId ac96ad79fc4852b70) — in flight
5. **Lane MAE-V implementation** (codex:rescue, agentId aad00f697049d7f36) — in flight
6. **Lane HF (Telescope) implementation** (codex:rescue, agentId a5b8d39aea1e6552a) — in flight
7. **8 council EUREKA deploy scripts** (codex:rescue, agentId a62673ed2e665a8a2) — in flight

### Direct fixes by parent agent (clean, ready to commit independently)
- **Preflight Check 39** (`check_undeployed_archive_artifact_producers`) added + STRICT, 0 live violations after exemption pass for kaggle_kernels + library files + 2 dead lanes (mini_tto_inflate, optimize_embedding). 8-test regression suite passes.
- **Lane SG `get_protected_patterns` helper** added to `src/tac/self_compress.py` with new `SC_SEGNET_PROTECTED_NAME_PATTERNS`. All 7 tests pass.
- **`--protected-pattern-set` flag** wired into both `src/tac/experiments/train_renderer.py` and `experiments/qat_finetune.py` (choices=['posenet_prior', 'segnet_prior']) + actually wired through to `swap_renderer_convs_with_self_compress` via `extra_protected_patterns` (segnet patterns are added on top of posenet defaults when set).

## What still needs to happen

1. **Wait for 5 in-flight codex:rescue landings** (~15-30 min each based on prior sessions)
2. **Verify**: pytest must be GREEN on all new tests after each landing
3. **Commit semantic chunks**: (a) Check 39 + Lane SG, (b) Round 10 fixes, (c) GE/HM/CG/DI/SI-V3 modules + tests, (d) SAUG/MAE-V/HF impls + deploy scripts, (e) 8 council EUREKA deploy scripts
4. **Recursive adversarial review**: codex:adversarial-review on cumulative diff; cluster findings, parallel-fix subagents until 3 clean passes
5. **Vast.ai harvest** (~6-8h from now): G v3, I, V, M-V2 v2, W, K, OS-V2 results

## Lanes inventory (post-recovery)

### TIER 1: Validated — Lane A 1.15 [contest-CUDA] frontier

### TIER 2: In-flight Vast.ai (overnight wave, 8 instances)
G v3, I, V, M-V2 v2, W, K, OS-V2 + 1 more

### TIER 3: Code shipped, deploy script in flight
- Lane GP, FL — modules done, deploy script in 8-script subagent
- Lane SG — helper done, deploy script in 8-script subagent
- Lane GE, HM, CG, DI, SI-V3 — modules done, deploy script in 8-script subagent
- Lane SAUG, MAE-V, HF — full implementation + deploy in 3 codex:rescue subagents

### TIER 4: Code shipped, never deployed (still need attention)
- Lane Ω-V2, SI-V2, LR-V2, LM-V2, MOS, S-V2, W-V2, F-V4, M-V3, D-V3 — per `project_outstanding_work_and_stacks_20260428` TIER 3
- These will be caught by Check 39 going forward if any add a __main__ entry

## Cross-references
- `project_council_eurekas_driving_geometry_20260428` — 8 lane premises
- `project_cosmos_mae_lyra_telescope_synthesis_20260428` — 3 high-EV lane premises  
- `project_outstanding_work_and_stacks_20260428` — TIER 3 catalog
- `project_lane_ec_engineered_corrections_20260428` — what Check 39 catches
- `feedback_oneshot_vastai_subagent_failure_pattern` — why I dispatched MULTIPLE codex:rescue rather than ONE big one
- `feedback_subagent_recursive_skill_invocation_stall` — why subagents do NOT call codex:adversarial-review themselves; parent does it on staged diff
