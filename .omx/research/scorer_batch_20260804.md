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

## FZ5 — cr2_ep854 + partial F0PR k6 repair (byte-closed negative; do not spend slot for pointer movement)

Status: **FOLDED for frontier movement before n600**. This section is the owed
verdict spec from fz5, not a recommendation to spend the single scorer slot.

Baseline for delta: LIVE BEST `S = 0.7541459 @ 358,084 B [macOS-CPU advisory]`.

Candidate:

| field | value |
|---|---:|
| archive | `/Volumes/VertigoDataTier/pact/ddm_fz5_20260804/cr2_ep854_f0pr_k6_partial/archive.zip` |
| archive bytes | 292,026 |
| archive sha256 | `b2527b2294f8f817a369a465766677728e448ea2c31b95cf5b9e223e08344fbc` |
| d_seg used | 0.00394407 (from existing `v4d_cr2_ep854` n600 report) |
| d_pose predicted | 35.4509349783 (runner-true repaired pairs + fz5 n600 component census) |
| recomputed S_pred | 19.4172738012 |
| delta vs live best | +18.6631279012 |
| byte ledger | closes=true, residual=0, payload_reencodes_identically=true |
| inflate custody | rc=0, 1 raw, 3,662,409,600 B, 202.66 s |

Why folded: repairing the 27 available k6 pairs only moved mean d_pose
`37.877063 -> 35.450935`; the pose mass remains population-wide. This cannot
beat the live best and should not consume the scorer slot unless MAIN wants a
negative calibration row after all crossing candidates are handled.

If an explicit negative calibration is still requested after sg4, fire:

```bash
.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \
  --sub-dir /Volumes/VertigoDataTier/pact/ddm_fz5_20260804/cr2_ep854_f0pr_k6_partial \
  --out /Volumes/VertigoDataTier/pact/ddm_fz5_20260804/cr2_ep854_f0pr_k6_partial/fz5_n600_eval_receipt.json \
  --inflate-out /Volumes/VertigoDataTier/pact/ddm_fz5_20260804/cr2_ep854_f0pr_k6_partial/fz5_n600_eval_inflated \
  --device cpu --batch-size 16 --num-threads 4
```

Fire condition for a scorer-worthy successor: an all-600 repaired cr2_ep854 row
must first produce a byte-closed archive with predicted `S < 0.7541459`. Current
section format is single-k; the fz1/fz4 estimate called for mixed k6/k8, so the
successor must either choose one k globally or land a receiver-closed mixed-k
section before claiming that estimate.
