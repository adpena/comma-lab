# Modal Artifact Harvest 2026-04-30

## Run summary

Harvest invocation: `.venv/bin/python tools/harvest_modal_calls.py`

| State | Count | Notes |
|-------|-------|-------|
| Already harvested (no-op) | 30 | Persisted from prior harvest (2026-04-29) |
| Newly harvested this pass | 0 | All non-already-harvested were still queued or cancelled |
| Still queued / not_ready | 7 | sa_v4, sa_v5_post_oom_fix, sc_plus_plus_v4, so_v3_post_oom_fix, mae_v_v2, q_faithful_v3, stc_cuda, sz_phase2_v2 |
| Cancelled by user | 1 | lane_so_v3 (RemoteError: cancelled) |
| Expired (>24h GC'd) | 0 | None lost |
| Total dispatched | 39 | All 39 modal_metadata.json files iterated |

## Harvested lanes — per-lane status

All artifacts already on disk from prior harvest. Re-running was idempotent.

| Lane dir | rc | elapsed | n_artifacts | crash kind | total bytes |
|---|---|---|---|---|---|
| lane_lane_fl_modal | 137 | 33s | 55 | RC_137 (SIGKILL) | 17,324,671 |
| lane_lane_fl_v2_modal | 137 | 25s | 55 | RC_137 (SIGKILL) | 17,324,682 |
| lane_lane_gp_modal | 1 | 12s | 55 | RC_1 | 17,325,488 |
| lane_lane_gp_v2_modal | 0 | 824s | 70 | OK | 20,193,562 |
| lane_lane_gp_v3_modal | 0 | 793s | 70 | OK | 20,218,220 |
| lane_lane_mae_v_modal | 1 | 7s | 55 | ERROR | 17,317,038 |
| lane_lane_mm_modal | 3 | 4s | 50 | RC_3 (dead-flag) | 16,580,958 |
| lane_lane_mm_v2_modal | 0 | 832s | 61 | OK | 20,663,553 |
| lane_lane_omega_hessian_modal | 1 | 45s | 52 | RC_1 | 16,603,806 |
| lane_lane_s_modal | 1 | 157s | 54 | RC_1 | 17,780,803 |
| lane_lane_sa_modal | 1 | 129s | 52 | RC_1 (T4 OOM) | 16,586,032 |
| lane_lane_sa_v2_modal | 1 | 126s | 52 | RC_1 (T4 OOM) | 16,586,033 |
| lane_lane_sa_v3_modal | 1 | 112s | 52 | RC_1 (T4 OOM) | 16,587,627 |
| lane_lane_sc_plus_plus_modal | 1 | 129s | 52 | RC_1 (A10G OOM) | 16,586,444 |
| lane_lane_sc_plus_plus_v2_modal | 1 | 160s | 52 | RC_1 (A10G OOM) | 16,586,444 |
| lane_lane_sc_plus_plus_v3_modal | 1 | 140s | 52 | RC_1 (A10G OOM) | 16,588,033 |
| lane_lane_so_modal | 1 | 128s | 52 | RC_1 | 16,586,499 |
| lane_lane_so_v2_modal | 1 | 132s | 52 | RC_1 | 16,586,499 |
| lane_lane_uniward_modal | 1 | 4s | 50 | ERROR | 16,580,821 |
| lane_lane_w_modal | 1 | 274s | 55 | RC_1 | 16,641,726 |
| lane_lane_w_v2_modal | 124 | 28800s | 59 | TIMEOUT (8h cap) | 22,861,935 |
| lane_q_faithful_modal | 2 | 64s | 58 | RC_2 | 18,031,640 |
| lane_q_faithful_v2_modal | 1 | 81s | 58 | RC_1 | 18,018,666 |
| lane_uniward_v2_modal | 1 | 3s | 52 | RC_1 | 16,581,506 |
| lane_uniward_v3_modal | 1 | 12s | 53 | ERROR | 20,653,014 |
| lane_uniward_v4_modal | 1 | 14s | 53 | ERROR | 20,656,383 |
| lane_uniward_v5_modal | 1 | 15s | 53 | ERROR | 20,656,677 |
| lane_uniward_v6_modal | 1 | 15s | 54 | ERROR | 20,656,590 |
| lane_uniward_v7_modal | 0 | 848s | 65 | OK | 22,162,960 |
| lane_uniward_v8_modal | 0 | 779s | 65 | OK | 23,542,965 |

**Totals**: 30 lane dirs, ~559 MB of harvested artifacts on disk.

## Notable scores surfaced

All scores from `experiments/contest_auth_eval.py` with `device=cpu` on Modal T4 worker
→ tagged `[contest-CPU advisory]` per CLAUDE.md non-negotiable. Need CUDA re-eval for
strategy decisions. Compare to Lane G v3 = 1.05 [contest-CUDA] anchor.

| Lane | score | archive bytes | PoseNet | SegNet | Tag | Notes |
|------|-------|---------------|---------|--------|-----|-------|
| lane_uniward_v8 | **1.14** | 694,045 | 0.00450 | 0.00461 | [contest-CPU advisory] | **SUPERSEDED**: Council B audit (`council_uniward_v8_fridrich_shannon_audit_20260429.md`) found this is a NO-OP — shipped masks.mkv is bit-identical (SHA `c07bd465...`) to Lane A's. The 1.14 IS Lane A measured on CPU (Lane A = 1.15 [contest-CUDA], -0.01 = ~9.5% PoseNet CPU drift). Stage 4 `cp $ANCHOR_DIR/masks.mkv` discards SLI1 payload. NOT a Phase 1 contender. See `project_lane_uniward_v8_NO_OP_finding_20260429.md`. |
| lane_lane_mm_v2 | 2.63 | 1,133,750 | 0.17367 | 0.00560 | [contest-CPU advisory] | Documented as FALSIFIED (encoder-only grayscale-LUT). PoseNet 51× worse than baseline. |
| lane_uniward_v7 | 53.61 | 346,214 | 62.69 | 0.28339 | [contest-CPU advisory] | AV1 mask artifacts catastrophe. Documented. |
| lane_lane_gp_v2 | 89.66 | 680,611 | 149.95 | 0.50482 | [contest-CPU advisory] | Runge phenomenon polynomial pose-fit failure. Documented. |
| lane_lane_gp_v3 | 89.67 | 692,568 | 149.95 | 0.50482 | [contest-CPU advisory] | Runge phenomenon polynomial pose-fit failure (Fix A applied, still fails). Documented. |

**Action items**:
- No new sub-Lane-G-v3-comparable result emerges from this harvest. The earlier optimism on `lane_uniward_v8 = 1.14` is retracted (Council B no-op finding). Lane G v3 at 1.05 [contest-CUDA] remains the floor anchor.
- 7 lanes still queued (`sa_v4`, `sa_v5_post_oom_fix`, `sc_plus_plus_v4`, `so_v3_post_oom_fix`, `mae_v_v2`, `q_faithful_v3`, `stc_cuda`, `sz_phase2_v2`) — re-run harvester within 24h to recover before result-cache GC.

## Cleanup candidates

See `reports/modal_cleanup_candidates.md`.
