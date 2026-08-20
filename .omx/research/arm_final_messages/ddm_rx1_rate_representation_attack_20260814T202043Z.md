RX1 is complete and committed, but it did not improve the frontier.

- Best candidate: 191,746 B, **+5,477 B** versus MC36.
- Exact identity: all tokens and the 3,662,409,600-byte raw output match MC36.
- Projected score: 0.1655813674 `[arithmetic projection; no scorer]`.
- T4 disposition: `DO_NOT_FIRE`.
- Retained: 2,781 files / 6,638,286,404 B.
- Verification: 11 tests passed, Ruff clean, payload-retention gate clean, two review passes.
- Commit: `27bef73f03`.

The arbitrage fallback kept this as one local code unit after its managed goal helper was sandbox-blocked.

Artifacts: [decision memo](/Users/adpena/Projects/pact/.omx/research/ddm_rx1_rate_representation_attack_20260814.md:1), [runner](/Users/adpena/Projects/pact/experiments/ddm_rx1_rate_representation_attack.py:1), [final receipt](/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/FINAL_RESULT.json:1), [retention inventory](/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/RETENTION_INVENTORY.json:1).

Own-vehicle frontier unchanged: **S = 0.1619344578804448 @ 186,269 B `[contest-CUDA] n600 T4`**.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN rate-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/` and `.omx/state/main_hot_state.md`; fire trigger: a seeded, resumable current-label pre-proof predicts a complete archive below 186,269 B after counted model bytes; action: train HPAC directly on MC36 labels and repeat the identity-closed archive race.
- `FOLDED-INTO-RX2` — owner: RX2 trainer owner; consumer store: RX2 per-stage candidate ledger; fire trigger: the first current-label checkpoint passes exact IHS1 round-trip and comes within 5,477 B of MC36; action: fit a checkpoint-specific residual table.

## LIVE-HYPOTHESES

- Direct MC36-label HPAC training may reverse the 5,462-B token penalty because tq1c reached 97,928 B on its own training distribution.
- Joint current-label training and a checkpoint-specific table may approach the sub-0.15 rate target; the transplanted MC36 table already saves 337 B over neutral.
- ANS/RC64 ordering may change after the prior changes; existing same-state margins are only single-digit bytes.

## DEAD-ENDS

- Frozen tq1c IHS1 transfer: closed because its best archive is 5,477 B larger.
- Removing the residual table: closed because it adds another 337 B.
- Further lossless IHS1 compression alone: closed because Brotli saves only 204 B versus XZ.
- Transferring tq1c’s historical 97,928-B stream: invalid because it encodes different labels.
- Re-running settled F26 ANS/RC64, CAP1 packing, or PZ4A allocation: closed by existing exact receipts.
- T4 evaluation of RX1: closed because identical distortion plus higher rate cannot improve the frontier.