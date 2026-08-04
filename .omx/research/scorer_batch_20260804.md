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

## C — qo1 pair-bitpack F0PR1 repair stream, queued n600 verdict

Candidate archive:
`/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip`
(`d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`).

Pre-score ledger: 357,836 B, byte ledger closes with residual 0 and payload re-encodes identically.
Predicted components with seg held constant: d_seg 0.00431179, d_pose 0.0007145917, rate term
`25*357836/37545489`; predicted S 0.75398083, delta -0.00016503 vs live own-vehicle baseline.

Fire order when the scorer slot frees: run the exact n600 upstream/evaluate.py row on the archive
above, recompute S from components, and compare against fz4 `sub_final` 0.7541459 @ 358,084 B
[`macOS-CPU advisory`]. Do not promote from this queued note alone.

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

## R4/#935 - sq1 realizer uncap and convergence test (cap-artifact class)

Status: **QUEUED MEASUREMENT SPEC**, not a promoted candidate. This section is
the na3/gc16 cap-artifact append for the single scorer batch after sg4 releases
the slot.

Source receipts:

- `ddm_sm1_seg_search_transfer_20260803.md` source-verifies
  `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32.json`:
  the `truth` start won 0/32, while 31/32 winners were `dec@25`, the terminal
  25-step iterate.
- `gc16_full_stack_convocation_20260804.md` R4 orders: uncap, run to a
  convergence test, and join the scorer batch.
- `ddm_et1_eta_on_the_priced_band_20260803.md` shows the same paint-solve
  budget axis is not free: eta rises with more steps and d_pose rises with it.

Fire order when the scorer slot frees:

1. Re-run the sq1 realizer with the step cap removed or raised behind an
   explicit convergence criterion, recording per-pair stop reasons. A terminal
   safety bound is allowed only as a safety bound, not as the quoted stop reason.
2. Build the receiver-closed candidate from the converged sq1 realizer output.
3. Run the exact n600 scorer row against the current own-vehicle baseline,
   reporting d_seg, d_pose, bytes, and recomputed S. The R8 pose-bank guard
   applies: refuse composition if the pose term erodes the bank by more than
   the batch threshold.

Do not promote from this queued note. If the convergence-tested candidate cannot
produce a byte-closed row with a plausible `S < 0.7541459` pre-score, fold it
before spending the n600 slot.
