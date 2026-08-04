# SCORER BATCH — pending n600 verdicts (ONE batched pass when sg4 frees the slot)
<!-- Owner: MAIN. Arms APPEND sections; MAIN fires the batch. One full-n600 scorer job fleet-wide. -->
Baseline for ALL deltas (m46): LIVE BEST S = 0.7541459 @ 358,084 B [macOS-CPU advisory]
(fz4 sub_final, receipt .omx/research/ddm_fz4_20260804/, commit 09eadc0033).

## B — rt1 rate receipts, scorer-gated (from 5f54e74c1e / eb3d47ce90; measured byte side, owed seg verdict)
| candidate | Dbytes (measured) | pre-registered seg bound | predicted dS if bound holds |
|---|---|---|---|
| #869 adaptive margin [16,12,8,4] on sub_final tokens | -113,555 B net | d_seg delta < 7.561e-4 | ~ -0.0756 rate, bounded seg risk |
| derived-activity fallback [16,12,8,4] | -62,502 B net | d_seg delta < 4.162e-4 | ~ -0.0416 rate |
| L=14 live ix2 | -24,605 B | (composes; re-verify jointly) | ~ -0.0164 rate |
NOTE: rt1 sweeps were measured on the pre-fz4 token base — REMEASURE bytes on sub_final before
the verdict (baselines move; a stale byte delta is not a receipt).

## (arms append below)
