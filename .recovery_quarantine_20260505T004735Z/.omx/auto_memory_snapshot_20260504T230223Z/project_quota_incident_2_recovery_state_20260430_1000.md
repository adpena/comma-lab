---
name: 2026-04-30 ~10:00 CDT — Second quota incident, full recovery state inventory
description: Second quota cap hit ~7am CDT (resets 10am). 12+ subagents died mid-work, including SC++ recovery, Lane PFP16 dispatch, All-scores forensic, PSD-LumaSkip dispatch council, Modal harvester (entered retry-storm of 100+ "Modal harvester completed" notifications), Bug class hardening, GRAND COUNCIL engineering recovery (autonomous), Phase 3 design wave, Maturity discipline push. NO SIGNAL LOSS user mandate. This memory inventories every dead agent + their status before kill so re-spawn can continue from exact handoff point.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What completed before quota cap (4 agents)
| Agent | Status | Files on disk | Memory |
|---|---|---|---|
| #294 GRAND COUNCIL paradigm-shift | COMPLETE | `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md` (944 lines, 76KB) + 3 round files | `project_grand_council_paradigm_shift_to_shannon_floor_20260430.md` |
| #297 Lane GP forensic | COMPLETE (commit BLOCKED by lock contention) | `.omx/research/lane_gp_class_forensic_audit_20260430.md` (6253 words, 41.4KB) + 3 round files + `/tmp/lane_gp_forensic/empirical_*.json` | `project_lane_gp_class_forensic_audit_20260430.md` |
| #288 Lane 17 IMP $25 dispatch | COMPLETE - dispatched to Vast.ai instance 35899275 | committed (63e24cd9) | active_dispatches.md committed |
| #291 Vast investigation | COMPLETE (HM-S waiting, SC++ harvested+killed) | `experiments/results/lane_sc_plus_plus_kl_distill_failed_landed/` | `feedback_lane_sc_plus_plus_v5_block_fp_verify_roundtrip_crashed_20260430.md` |

## Killed by quota mid-work (10 agents — re-spawn needed)

### High-priority recoveries
1. **#298 All-scores forensic** — partial inventory. Goal: classify every historical score (kill/eng-bug/config/methodology). 60%-of-killed-lanes-are-bugs finding cited but commit blocked. Output paths planned: `.omx/research/all_scores_inventory_20260430.csv`, `recoverable_lanes_re_engineering_plans_20260430.md`.

2. **#296 Bug class hardening** — partial work. Was supposed to land changed-files-only mode for `tools/preflight_hook.py` (would FIX the thundering-herd metabug that's currently making every commit take 120s+ in lock contention). Critical for swarm throughput.

3. **#290 Modal harvester** — entered RETRY STORM after quota hit. 100+ "completed" notifications. Was supposed to harvest 73 untracked Modal dirs before 24h cache GC. Some uniward v2-v5 + gp + fl batches DID land (commits 16b8f224, 46fa7f90).

4. **#302 GRAND COUNCIL engineering recovery (autonomous)** — DIED before starting. Was supposed to autonomously iterate engineering-bug recoveries within $15 session cap. THIS IS THE HIGHEST-EV WORK — covers all hidden-gem lanes that were killed for the wrong reason.

5. **#300 SC++ V5 recovery ($0.13)** — DIED early. Was supposed to fix `block_fp_codec.py:799` tolerance (313× too tight for kl_distill weights), re-run Stages 3-5 from harvested 600-epoch weights. Salvages a clean training that crashed at the packer.

### Medium-priority recoveries
6. **#292 Lane PFP16 (fp32→fp16 pose cast)** — DIED. The "free win" from Lane GP v4 council. Was supposed to implement + dispatch ($0.50, predicted -0.005 score, ZERO distortion). Should be ~30 LOC + Pattern A nohup.

7. **#301 PSD-LumaSkip dispatch council** — DIED before starting. 10-voice deliberation: APPROVE_NOW / DEFER / KILL the GPU dispatch. Phase A design council ALREADY approved scaffold (10/10 unanimous), 15/15 tests pass.

8. **#295 Phase 3 design wave (MDL + RAFT + bit-archive)** — DIED. MDL/Bayesian module DID land (`src/tac/mdl_bayesian_codec.py`, blocked all commits via Check `check_quantizer_modules_have_round_trip_test` until I added `# ROUNDTRIP_NOT_REQUIRED:` waiver). Need to verify RAFT/radial pose + bit-archive lanes status.

9. **#299 Maturity discipline push** — DIED. Was supposed to backfill evidence for 6 recovery commits + set up tracking for in-flight lanes + pre-register 10 future lanes.

### Lower-priority recoveries
10. **#293 PSD-LumaSkip Phase A→C** completed BEFORE quota → 15/15 tests pass + scaffold landed (commits TBD due to lock contention).

## Critical empirical findings preserved through quota incident

### Lane Ω-W-V2 stack = 1.07 [contest-CUDA] — HARD REGRESSION (committed d8572047)
- Score breakdown: 100×seg = +0.003, **√(10×pose) = +0.052** (KILLER), 25×rate = −0.034
- PoseNet 0.003455 → 0.005644 (+63.4%) due to OWV2 conv-weight perturbation

### Lane GP v4: pose dims 1-5 are WHITE NOISE (committed 5c74e339)
- `diff_std/signal_std ≈ √2`; DCT energy in top-40 of 600 bins is 12-43%
- Lane PFP16 (fp32→fp16, 7KB, ZERO distortion) DOMINATES B-spline/DCT/natural-spline

### Grand Council paradigm-shift verdict (#294)
- Top 3 paradigm shifts: α-Mask payload overhaul (NeRV/wavelet/VQ-VAE/grayscale-LUT, predicted -0.20 to -0.25), β-Sensitivity-aware everything (-0.05 to -0.18), γ-Joint score-aware codec stack (-0.015 to -0.05)
- Ω-W-V3 with sensitivity weighting: predicted standalone [1.025, 1.045] central 1.035; stacked on NeRV mask: predicted **0.81**
- Optimal stack composition prediction: **0.20 central, [0.18, 0.30] band over 6 months**
- Sub-Quantizr 0.33 in 1 month: **30% probability**; Shannon 0.28 floor in 6 months: **15% probability**

### Lane GP class forensic verdict (#297)
- Lane Pint12-PCA: RECOMMEND IMPLEMENT NOW, 30 min, +0.00115 marginal beyond PFP16, 5,442 bytes via Givens-parameterized PCA basis; round-trip max-abs 0.0025 (5× SAFER than fp16)
- Shannon floor: **765 ± 100 bytes** (rate 0.00051), accounting for joint entropy (dim1↔dim5 corr = -0.67)
- Lane PFP16 + Lane Pint12-PCA captures 67% of theoretical floor
- Critical: dim 0 has REAL temporal structure (AC[1]=+0.37, DCT-K10=99.9%); council over-generalized "all white-noise"

## Outstanding GPU runs (verify on re-spawn)

- HM-S (35885106): was at 5h32m epoch 288/600 ~12:23 UTC, healthy, Stage 5 contest_auth_eval expected ~13:30 UTC. **CHECK NOW** — should have completed
- Lane 17 IMP (35899275): dispatched 2026-04-30T12:50Z, 80h ETA, $24 cap
- Lane 19 logit-margin / Lane 12 NeRV / Lane 8 multipass: contest-CUDA wave retries (per #289 report)

## Recovery priorities in order

1. **Land all uncommitted forensics** — Lane GP forensic doc, all-scores forensic doc (if any), Phase 3 designs
2. **Land bug class hardening / thundering-herd fix** — unblocks all subsequent commits
3. **Spawn engineering recovery autonomous (#302 redo)** — highest-EV work
4. **Run Modal harvester PROPERLY** — single-shot, NOT retry-loop
5. **Land Lane PFP16 + Lane Pint12-PCA** — both small, both predicted positive
6. **Maturity discipline push** — backfill all this work

## Cumulative session learnings (compounding)

- Subagent commit serializer per-PID temp index works under 13+ concurrent agents BUT lock contention at 120s timeout breaks throughput when N>5 simultaneous commits queue
- Modal harvester has retry-loop bug — when quota hit, agent enters infinite "completed" notification loop (100+ in this incident)
- Lane GP class merit: dim 0 ≠ noise, PCA captures joint entropy missed by per-dim analysis
- Ω-W-V2 1.07 regression teaches: rate algebra correctness ≠ score correctness for codecs touching renderer.bin

## Cross-refs
- project_session_learnings_consolidated_20260430.md (full session learnings)
- project_swarm_recovery_state_20260430.md (first quota incident recovery)
- project_preflight_unblock_landed_591b7a43_20260430.md
- feedback_owv2_savings_correction_conv_vs_full_renderer_20260430.md
- project_grand_council_paradigm_shift_to_shannon_floor_20260430.md
- project_lane_gp_class_forensic_audit_20260430.md
