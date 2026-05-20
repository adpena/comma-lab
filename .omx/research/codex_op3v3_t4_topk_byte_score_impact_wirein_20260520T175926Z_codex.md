# Codex OP3-V3 T4 Top-K Byte Score-Impact Wire-In

Generated: 2026-05-20T17:59:26Z
Author: Codex

## Input Anchor

The OP3-V3 T4 master-gradient anchor is present and verified:

- gradient tensor: `.omx/state/master_gradient_fec6_contest_cuda_t4_20260520.npy`
- gradient SHA-256: `a1afce293533fbe1c1be67b626db9e532700e4ed66d84c62ed6d0bb67d15a1bc`
- archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- inner `x` SHA-256: `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`
- axis: `[contest-CUDA]`
- hardware: `linux_x86_64_t4_modal`
- pairs used: `600`
- score marginals: `dS/d_seg=100.0`, `dS/d_pose=25.59381904871778`, `dS/d_byte=6.658589531221714e-07`

## Tooling Landed

- `src/tac/optimization/byte_score_impact.py`
  - applies operating-point score marginals before ranking bytes;
  - ranks aggregate top-K and per-axis top-K;
  - summarizes cumulative score-impact mass and axis share.
- `tools/plan_topk_byte_score_impact_targets.py`
  - loads a pinned master-gradient anchor;
  - verifies the gradient SHA;
  - emits top-K target sets for byte-level/codec-level candidate builders;
  - validates Catalog #356 `AxisDecomposition` via the existing top-K cathedral consumer.
- `src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer/__init__.py`
  - adds `rank_archive_bytes_by_score_impact(...)` as the score-aware variant of the older raw-norm ranker.

Focused tests:

```text
.venv/bin/python -m pytest -q src/tac/tests/test_byte_score_impact.py src/tac/tests/test_master_gradient_exploits_end_to_end.py src/tac/atom/tests/test_contest_granularity.py src/tac/tests/test_contest_oracle_search.py
35 passed, 1 skipped in 0.87s
```

## Generated Artifact

`experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/topk_byte_score_impact_targets_op3v3_t4_20260520_codex.json`

Command:

```text
.venv/bin/python tools/plan_topk_byte_score_impact_targets.py --gradient-sha256 a1afce293533fbe1c1be67b626db9e532700e4ed66d84c62ed6d0bb67d15a1bc --k-values 32,64,128,256,512,1024 --top-axis-k 32 --top-record-limit 32 --output experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/topk_byte_score_impact_targets_op3v3_t4_20260520_codex.json
```

Key output:

- global score-impact share: SegNet `0.8911468355285032`, PoseNet `0.10885316447149691`, rate `0.0`;
- aggregate top-5 byte indices: `[79944, 79945, 113039, 113040, 113041]`;
- K=32 contiguous target runs: `[79944..79945]` and `[113039..113068]`;
- SegNet top-8: `[79944, 79945, 113039, 113040, 113041, 113042, 113043, 113044]`;
- PoseNet top-8: `[35781, 35782, 79944, 79945, 113384, 113385, 113386, 113387]`.

Top-K cumulative mass:

| K | total score-impact share | Seg share within K | Pose share within K | dominant counts |
|---:|---:|---:|---:|---|
| 32 | `0.005648030695880184` | `0.9454942662161467` | `0.054505733783853254` | seg 32 / pose 0 |
| 64 | `0.010894922648439316` | `0.9462266556331929` | `0.05377334436680713` | seg 64 / pose 0 |
| 128 | `0.02138870655355757` | `0.946613453955799` | `0.053386546044200955` | seg 128 / pose 0 |
| 256 | `0.04237627436379408` | `0.9468123453339871` | `0.05318765466601301` | seg 256 / pose 0 |
| 512 | `0.07257439200238489` | `0.9423767670005607` | `0.05762323299943936` | seg 512 / pose 0 |
| 1024 | `0.09247155122593079` | `0.9358485774305003` | `0.06415142256949964` | seg 1022 / pose 2 |

## Interpretation

This OP3-V3 target surface says the highest-impact byte interventions on the
actual contest-CUDA T4 FEC6 anchor are mostly SegNet-weighted, with a small
PoseNet overlap. That does not make blind byte edits promotable: raw archive
byte gradients remain diagnostic, and packet-valid mutations still need codec
grammar, inflate/raw controls, and exact eval.

The immediate frontier use is to replace visibility-only candidate selection
with a two-stage gate:

1. candidate must touch OP3-V3 top-K score-impact byte neighborhoods or a
   documented codec-grammar symbol mapped to those neighborhoods;
2. candidate must emit Catalog #356 axis decomposition and then survive
   advisory/full exact-eval residual checking.

Do not rank PR110 successors from raw byte count alone or raw gradient L2 norm
when the OP3-V3 score marginals are available.
