Executed and committed as `5c073e91507345315d8a2b2170b24ff0f8638888`.

Key findings:

- Receiver consumption: all 38/38 semantic tensors are required.
- Derive-at-decode: 0/38 tensors are free structural zero/one constants.
- Exact representations: dense, sparse, row-dictionary, and hybrid archives all measured 183,842 B, which is 340 B worse than e480b.
- Mixed q3/q4 candidate: 182,679 B, saving 823 B. Retained but not score-admitted.
- Six Film-row sparsity candidates: savings from 130 to 2,051 B. Retained but unscored.
- No candidate approaches the 15,153 B fixed-distortion saving needed for sub-0.15.
- T4 fire order: `FOLDED`; no paid dispatch occurred.
- Live e960 state was untouched.

Artifacts:

- [MZ2 memo](/Users/adpena/Projects/pact/.omx/research/ddm_mz2_frozen_section_representation_attack_20260815.md)
- [Runner](/Users/adpena/Projects/pact/experiments/ddm_mz2_frozen_section_representation_attack.py)
- [Final receipt](</Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation/FINAL_RESULT.json>)
- Retention inventory: 167 nonrecursive files, 8,183,925 B, SHA-256 `156112d0a0b8caeec0f0a6eaedd3bc1d24e2d389b199dad2495324ebd6c2dbcc`.

Verification: 4 tests passed, Ruff clean, payload-retention gate clean, two review passes completed, and resume behavior verified. The repository-wide developer preflight still has eight pre-existing red gates; their violation lists contained zero MZ2 references.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN`; consumer store: `SCORE_GATE_RESULT.json`; fire trigger: once no full-n600 scorer is active and a strict SD1M shipping receiver parse-backs archive `b3b38b…`, run advisory n600 and admit only below the `−3.5e-6` net threshold.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN`; consumer store: `SCORE_GATE_RESULT.json`; fire trigger: after the mixed-bit verdict, install one strict SM3R receiver and score the six retained sparsity cells sequentially until dominated.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN`; consumer store: `SCORE_GATE_RESULT.json`; fire trigger: after e960 completes and the governed trainer/scorer slot is free, launch deterministic resumable width distillation with per-stage checkpoints and retained learned payloads.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN`; consumer store: `FINAL_RESULT.json`; fire trigger: if all semantic queues terminate above 0.15, attack the exact CAP1 carrier using a same-decode representation not already covered by PK2, PK4, PS135B, or MZ1.

## LIVE-HYPOTHESES

- Mixed q3/q4 may be a small real win because it saves 823 actual bytes and the same allocation survived the prior lineage screen; current receiver-closed Seg/Pose scoring is decisive.
- Light Film sparsity may expose a small Pareto point because keep-87 changes only 548 parameters for 130 B, though it lacks prior n600 support.
- Real renderer distillation is the remaining semantic mechanism capable of multi-kilobyte savings because it changes architecture instead of recoding full-rank matrices.
- CAP1 may still admit a structural same-decode representation because it explicitly stores 27,648 basis symbols and 7,200 coefficients.

## DEAD-ENDS

- Unread-tensor deletion: all 38/38 keys are receiver-required.
- Structural zero/one derivation: 0/38 tensors qualify.
- Exact zero-sparse and row-dictionary storage: 0/64 selectors beat dense; every archive was +340 B.
- Another low-rank/VQ sweep: closed by consumed SM3/SM4/CP2/SV3 evidence plus 16/16 full-rank current matrices.
- Lossless F12 recoding: already closed by MZ1.
- Immediate T4 dispatch: no new archive is both shipping-receiver-closed and current-score-admitted.

Own-vehicle frontier remains **S = 0.1600920261571558 @ 183,502 B `[contest-CUDA T4, n600]`**; pointer unmoved.