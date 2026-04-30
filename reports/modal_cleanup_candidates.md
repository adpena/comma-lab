# Modal Cleanup Candidates 2026-04-30

**STATUS: DO NOT DELETE WITHOUT EXPLICIT USER APPROVAL.**

This file lists harvested Modal artifact directories that are known-failed or that
the operator may consider deleting to free disk space. Items are listed in order
of cleanup priority; nothing is auto-deleted.

## Total disk usage of harvested artifacts

~559 MB across 30 directories (`experiments/results/lane_*_modal/harvested_artifacts/`).

## Known-failed runs (rc != 0, no useful artifacts beyond crash logs)

These all exited non-zero on Modal worker. Each holds ~16-23 MB of mostly-redundant
test fixtures, infra logs, and crash tails. The `_stdout_tail.txt` and crash classification
in `_harvest_summary.json` are the load-bearing parts; the bulk is environment dump.

### Quick crashes (< 60s) — likely dead-flag, missing-input, or import errors

| Lane | rc | elapsed | bytes | Likely cause |
|------|----|---------| ------|--------------|
| lane_lane_mm_modal | 3 | 4s | 16,580,958 | RC_3 (dead-flag bug per memory) |
| lane_lane_uniward_modal | 1 | 4s | 16,580,821 | ERROR |
| lane_uniward_v2_modal | 1 | 3s | 16,581,506 | RC_1 |
| lane_lane_mae_v_modal | 1 | 7s | 17,317,038 | ERROR (likely import) |
| lane_uniward_v3_modal | 1 | 12s | 20,653,014 | ERROR |
| lane_uniward_v4_modal | 1 | 14s | 20,656,383 | ERROR |
| lane_uniward_v5_modal | 1 | 15s | 20,656,677 | ERROR |
| lane_uniward_v6_modal | 1 | 15s | 20,656,590 | ERROR |
| lane_lane_gp_modal | 1 | 12s | 17,325,488 | RC_1 |
| lane_lane_omega_hessian_modal | 1 | 45s | 16,603,806 | RC_1 |
| lane_lane_fl_modal | 137 | 33s | 17,324,671 | RC_137 (SIGKILL/OOM) |
| lane_lane_fl_v2_modal | 137 | 25s | 17,324,682 | RC_137 (SIGKILL/OOM) |

Approximate combined size: ~213 MB.

### OOM crashes on T4 / A10G (documented bug class — keep for forensics)

| Lane | rc | elapsed | bytes | Status |
|------|----|---------| ------|--------|
| lane_lane_sa_modal | 1 | 129s | 16,586,032 | T4 OOM (post-OOM-fix lane re-dispatched) |
| lane_lane_sa_v2_modal | 1 | 126s | 16,586,033 | T4 OOM |
| lane_lane_sa_v3_modal | 1 | 112s | 16,587,627 | T4 OOM |
| lane_lane_sc_plus_plus_modal | 1 | 129s | 16,586,444 | A10G OOM (21+GB allocation) |
| lane_lane_sc_plus_plus_v2_modal | 1 | 160s | 16,586,444 | A10G OOM |
| lane_lane_sc_plus_plus_v3_modal | 1 | 140s | 16,588,033 | A10G OOM |
| lane_lane_so_modal | 1 | 128s | 16,586,499 | (Hessian-fallback bug; killed per council) |
| lane_lane_so_v2_modal | 1 | 132s | 16,586,499 | (same) |

Approximate combined size: ~133 MB.

### Slow failures — partial signal, may have informative logs

| Lane | rc | elapsed | bytes | Notes |
|------|----|---------| ------|-------|
| lane_lane_s_modal | 1 | 157s | 17,780,803 | RC_1 mid-training |
| lane_lane_w_modal | 1 | 274s | 16,641,726 | RC_1 (lane W) |
| lane_lane_w_v2_modal | 124 | 28800s | 22,861,935 | TIMEOUT @ 8h cap (max_seconds) |
| lane_q_faithful_modal | 2 | 64s | 18,031,640 | RC_2 |
| lane_q_faithful_v2_modal | 1 | 81s | 18,018,666 | RC_1 |

Approximate combined size: ~93 MB.

### Failed-but-instructive (KEEP — documented in memory)

These have been written about in memory and may be re-referenced:

| Lane | score | Memory ref |
|------|-------|------------|
| lane_uniward_v7_modal | 53.61 | AV1 mask catastrophe (failure mode is itself a teaching artifact) |
| lane_lane_gp_v2_modal | 89.66 | project_lane_gp_v3_landed_runge_phenomenon_20260429.md |
| lane_lane_gp_v3_modal | 89.67 | project_lane_gp_v3_landed_runge_phenomenon_20260429.md |
| lane_lane_mm_v2_modal | 2.63 | project_lane_mm_v2_landed_2_63_falsified_20260429.md |

### Successful runs (KEEP — score signal)

| Lane | score | Note |
|------|-------|------|
| lane_uniward_v8_modal | 1.14 | project_lane_uniward_v8_harvested_1_14_advisory_20260429.md (needs CUDA re-eval) |

## Recommendation

**Keep everything for now.** The dispatched failures are < 1 GB total and they form
the empirical evidence chain for several "killed lane" memory documents. Cleanup is
better deferred until either (a) the contest deadline passes and we can do
post-mortem, or (b) disk pressure becomes an issue.

If user explicitly approves cleanup later: start with the quick-crash group
(~213 MB freed) since the `_harvest_summary.json` already captures the
crash classification and `_stdout_tail.txt` captures the failure signature —
the bulk of the 16 MB is duplicated test fixtures and infra logs.
