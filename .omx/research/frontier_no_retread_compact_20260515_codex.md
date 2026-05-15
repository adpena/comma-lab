# Frontier No-Retread Compact - 2026-05-15

Purpose: prevent repeated loops around the same HNeRV/PR101 byte-polish basin.
This is a routing ledger, not a score claim.

## Fixed Evidence

| Item | Axis | Status | Action |
| --- | --- | --- | --- |
| Public PR #101 | public CPU leaderboard | winner; displayed `0.193` | use as control, not destination |
| Local PR101/FEC6 | `[contest-CPU]` | `0.1920513168811056`, bytes `178517`, sha `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` | valid CPU near-miss |
| Local PR101/FEC6 paired | `[contest-CUDA/T4]` | `0.22621002169349796` | not promotion-ready; CUDA component gap dominates |
| PR101/FEC7 selector-only | no-score byte prototype | best charged candidate `268 B` vs FEC6 selector `249 B` | blocked; do not rerun selector-only entropy |
| PR106 format0C | `[contest-CUDA]` | `0.2063163866158099`, bytes `186327`, sha `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7` | live orthogonal component signal |
| PR106 format0C paired | `[contest-CPU]` | `0.22776488386973992` | CPU not always better; pair every axis |
| Kaggle PR106 score-table v3/v4 | provider advisory producer | v3 missing `constriction`; v4 provider-local claim mirror bug | infrastructure fixed; method untested |
| PR106 public-r2 latent component planner | `[provider-CUDA:kaggle advisory score-table]` | `28,442` net-improving cells after a 2-byte cell charge; source is public PR106 `0.bin` payload `7f2cc905...`, not format0C `x`; best cell pair `545`, `dim=24`, `delta_q=2`, net `-0.007331343441071264` | materialize only against the matching source archive; no format0C reuse without a matching format0C table |

## Stop Rules

- Stop pure PR101 byte polishing unless the charged byte saving is at least `79 B`
  and preserves decoded/runtime custody.
- Stop selector-only FEC7/range/adaptive work unless model bytes are charged and
  total payload beats FEC6 selector by at least `79 B`.
- Stop using CPU results as CUDA frontier language, or CUDA results as CPU
  frontier language. Every promoted packet needs paired CPU/CUDA evidence.
- Stop any source-bundled provider dispatch while uncommitted `src/tac/**`
  files exist unless they are intentionally committed into the runtime closure.

## Live Nonlocal Moves

1. PR106 score-table retry from the hardened provider-local claim/source-bundle
   path. Goal: component-moving latent/y-shift table, not another archive repack.
2. PR106 public-r2 component-moving cell materialization from
   `.omx/research/pr106_component_moving_cells_20260515_codex.{json,md}`.
   Goal: isolate single-cell transfer on the matching public-r2 source. Do not
   apply this table to format0C `x`; the materializer correctly rejects that
   source mismatch.
3. CUDA-in-loop xray/waterfill over hard pairs and hard frames. Goal: modify
   PoseNet/SegNet components, not only rate bytes.
4. Time-traveler / predictive-receiver / C1 / C6 across-family lanes, but only
   when each lane has archive grammar, runtime effect, and byte-closed export.
5. Parallel actuator first for any sweep: candidate generator without dispatch
   consumer is incomplete during race mode.

## Default Next Action

If asked to "continue" from this state:

1. ensure worktree source-bundle cleanliness;
2. claim `lane_pr106_latent_sidecar` or sister y-shift lane;
3. launch/harvest the PR106 score-table producer;
4. materialize only byte-closed candidates with explicit member `x`;
5. run paired exact CPU/CUDA eval before frontier language.

## Source Ledgers

- `.omx/research/public_frontier_refresh_post_deadline_20260515_codex.md`
- `.omx/research/sub_0192_frontier_gate_and_pair_xray_20260515_codex.md`
- `.omx/research/pr101_fec7_selector_entropy_blocker_20260515_codex.md`
- `.omx/research/pr106_format0c_paired_cpu_cuda_auth_eval_20260515_codex.md`
- `.omx/research/pr106_format0c_latent_score_table_kaggle_dispatch_20260515_codex.md`
- `.omx/research/pr106_component_moving_cells_20260515_codex.md`
