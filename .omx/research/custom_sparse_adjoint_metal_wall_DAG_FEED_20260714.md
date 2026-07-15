# DAG FEED-d43-metal-wall — 2026-07-14

**Node FEED-d43-metal-wall** — D43 custom sparse-adjoint Metal micro-bench: BLOCKER CLEARED, wall MEASURED.

- SIGNAL: predecessor `custom_sparse_adjoint_kernel_20260713.md` left the Metal wall `UNMEASURED/BLOCKED`
  (`No Metal device available`), so the DERIVED `2.208577x` arithmetic ceiling read as banked headroom.
- DIAGNOSTIC: Metal device is live this session → ran the staged optimal-form bench
  (`tools/bench_custom_sparse_adjoint_kernel.py`, 10 cross-process parity + 125-shape wall + rank2/K2).
- RESPONSE (MEASURED, advisory `[macOS-MLX/Metal research-signal]`, NON-score):
  - parity `GREEN_BIT_IDENTICAL` (10 procs, 1 hash; max-abs 6.68e-6 vs #212 dense Metal),
  - whole-network wall **0.7078x (SLOWDOWN)**, η=**0.3205** of the 2.2086x ceiling (dense 65.36ms vs sparse 92.34ms),
  - per-family: seg-head 1.63x / decoder 1.06x / encoder 0.76x (114/125 shapes are the 98.3%-support encoder → no sparsity),
  - rank-2/K=2 basis-fusion **0.4923x** (also a slowdown).
- VERDICT: whole-network sparse-adjoint FORMULATION **DOMINATED** on this M5/MLX substrate; scope INSTANCE/FORMULATION.
  Family OPEN only via **hybrid layer-routing** (sparse on decoder/head, dense #212 on encoder), gated behind the
  unbuilt oracle-mask predictor (0.514 rel-L2 gap). Net EV LOW — one fewer phantom lever.
- OWED (arm-owned, routed): eq anchor append to `custom_sparse_adjoint_achieved_vs_ceiling_v1`;
  ledger D43 `METAL-WALL-OWED`→`METAL-WALL-MEASURED 0.708x SLOWDOWN`; DSL stays `REFUSED_NOT_FIREABLE`.
- Memo: `custom_sparse_adjoint_metal_wall_MEASURED_20260714.md`. Pointer 0.19108/0.18804 UNMOVED.
