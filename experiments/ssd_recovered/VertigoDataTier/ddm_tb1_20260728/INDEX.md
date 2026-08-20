# ddm_tb1 SSD receipts — SPEC_tr1 renderer build (2026-07-28)

Worktree: `/Users/adpena/Projects/pact/.claude/worktrees/agent-a356e3df5f7421e3e` (off main `e4bacb5d39`)
Commits: `db105b6f51` (T0) · `ea9a982fe8` (T0.1 forces) · `58835f6f29` (T0.2 topology) · `d72081f209` (T3 DSL).
Memo: `.omx/research/ddm_tb1_renderer_build_20260728.md` + DAG FEED sibling.
Evidence axis everywhere: `[macOS-CPU/MLX advisory]`, score_claim=false. Pointer 0.1910828242 UNMOVED.

- `t0_microsmoke_plain/`, `t0_microsmoke_lotto/`, `t0_microsmoke_v2/`, `t0_microsmoke_v3/` — n4 real-GT
  end-to-end micro-smokes (T0/T0.1/T0.2 code states).
- `t1_smoke_plain/`, `t1_smoke_lotto/` — T1 n24×60ep race windows: sealed_ticket.json, tr1_config.json,
  telemetry.jsonl (epoch + a1_gate rows), checkpoints/ (stage-encoded EMA-shadow npz),
  tr1_window_receipt.json. NOTE: arms ran skewed code versions (plain pre-T0.1, lotto post-T0.1) —
  T1 ordering not load-bearing.
- `t1_detached/`, `t2_detached/` — launch manifests + run logs (tools/launch_detached_process.py).
- `t2_memory_preflight/` — 1-epoch REAL n600 config window + memory_preflight_receipt.json
  (peak RSS 12.8 GiB MEASURED via /usr/bin/time -l; SAFE).
- `t2_n600_plain/`, `t2_n600_lotto/` — T2 bounded n600 windows (40ep/50min cap, identical code
  58835f6f29, sealed tickets d6eeefd7/0a6eba28, fd2 36-pair gate geometry, full-confirm chunked).

Rebuild: every window is deterministic from its sealed_ticket.json argv + the gt cache
(`experiments/results/mlx_fleet_gt_cache/gt_n600.npz` on MAIN) + the commit sha in the receipt;
checkpoints reload via `--resume-from` (custody proven: reload reproduced gate d_seg bit-for-bit).

## T2 outcome (2026-07-28, final)
- plain full-confirm n600 d_seg 0.014088 @ 549,927 B; lotto 0.013833 @ 534,597 B (renderer 3,284 vs 20,214 B).
- ADJUDICATION (pre-registered Pareto): WINNER = G1-LOTTO; plain ckpt retained as fallback; Lane-pool race first burn item.
- T3 sealed ticket: t3_long_burn_lotto_sealed_ticket.json (ticket 007d8eac…, sealed 99b13a53…, code 17166ee9c4).
- op1 $0 row-foveation gate receipt: op1_row_foveation_gate.json (PASSED 72.1% >= 50%).
- t2_n600_plain_aborted_emawarmup_gate_artifact/ = the #85 confound custody (first launch, named reason).
