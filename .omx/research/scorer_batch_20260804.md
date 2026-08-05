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

## ED1 - Road/Lane per-edge separatrix carrier, queued n600 verdict

Status: **QUEUED MEASUREMENT SPEC**, not a promoted candidate. ED1 built a
receiver-consumed byte-closed candidate, but did not run SegNet/PoseNet because
the single full-n600 scorer slot is owned by sg4/sb1 under this batch contract.

Candidate archive:
`/Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline/archive.zip`
(`a18c1a8c1fe4cab5fe675f661f3433b4b0013c2b4f51e764119d819b2fd86b89`).

Byte ledger:
`/Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline/ed1_byte_ledger.json`.
Receiver smoke:
`/Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline/ed1_receiver_smoke.json`.

Measured byte side on `sub_final` base:

| field | value |
|---|---:|
| base archive bytes | 358,084 |
| candidate archive bytes | 527,435 |
| archive delta | +169,351 |
| ED1 section bytes | 169,149 |
| Road/Lane cached target cells | 235,148 |
| centerline-band captured target cells | 191,005 |
| cache capture fraction | 0.8122756732 |
| own byte-closed break-even survival, no collateral | 0.6964303814 |
| charter sg3 falsifier survival | 0.3956 |
| projected S at sg3 survival, no collateral | 0.8028554362 |
| projected S at 100% survival, no collateral | 0.7049928349 |

Fire order when the scorer slot frees:

```bash
.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \
  --sub-dir /Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline \
  --out /Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline/ed1_n600_eval_receipt.json \
  --inflate-out /Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline/ed1_n600_eval_inflated \
  --device cpu --batch-size 16 --num-threads 4
```

Verdict fields owed by the scorer row: exact d_seg, d_pose, bytes, recomputed S,
realized Road/Lane survival against both the charter falsifier `0.3956` and the
actual byte-closed break-even `0.6964303814`, per-class flip deltas
Road->Lane/Lane->Road, and any collateral flips outside the Road/Lane target
set. The R8 pose-bank guard applies; do not promote from this queued note.

## C RESULT — qo1 pair-bitpack F0PR1 repair stream

Status: **MEASURED n600 row, PASS small rate win**.

| field | value |
|---|---:|
| archive | `/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip` |
| archive sha256 | `d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a` |
| archive bytes | 357,836 |
| axis | `[macOS-CPU advisory]` |
| n | 600 |
| d_seg | 0.00431179 |
| d_pose | 0.00071459 |
| rate term | 0.00953073 |
| recomputed S | 0.7539807296911207 |
| delta vs fz4 baseline | -0.0001651330203744 |
| R8 pose-bank guard | PASS (`sqrt(10*d_pose)=0.084533425342`, erosion +0.000033425342 vs 0.0845) |

Receipt:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/C_qo1_pairbit_n600_eval_receipt.json`.

Boundary: first two fires failed before scoring because the upstream evaluate
path contract was mis-bound (`submission_dir/inflated/0.raw` is mandatory for
compressed raw; `--uncompressed-dir` is the GT video directory). The measured
row is the corrected candidate-local inflated run. Score claim remains false;
contest-CPU/CUDA pointer unmoved.

## B RESULT - rt1 adaptive-margin [16,12,8,4] on fz4 sub_final

Status: **MEASURED n600 row, FORMULATION NEGATIVE**.

Byte receipt:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_subfinal_adaptive_byte_receipt.json`.
Eval receipt:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_margin_16_12_8_4_n600_eval_receipt.json`.
Per-class decomposition:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_margin_16_12_8_4_per_class_seg_decomp.json`.

| field | value |
|---|---:|
| archive | `/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_margin_coupled_16_12_8_4_sub_final/archive.zip` |
| archive sha256 | `bc34987c711cdce91b33708bb98998f86832e3bc3db25ab2b4aada5e58711060` |
| archive bytes | 244,436 |
| axis | `[macOS-CPU advisory]` |
| n | 600 |
| d_seg | 0.00515854 |
| d_pose | 0.16815221 |
| rate term | 0.00651040 |
| recomputed S | 1.9753490686354727 |
| delta vs fz4 baseline | +1.2212032059239775 |
| decomp delta d_seg | +0.0008467525906032986 |
| pre-registered bound | `< 0.0007561161342178816` |
| R8 pose-bank guard | FAIL (`sqrt(10*d_pose)=1.2967351695701015`, erosion +1.2122351695701015 vs 0.0845) |

Per-class SegNet-only decomposition of the collateral:

| class | delta errors | global delta d_seg |
|---|---:|---:|
| Road | +63,053 | +0.0005345069037543403 |
| Lane markings | +19,516 | +0.0001654391818576389 |
| Undrivable | +14,284 | +0.00012108696831597222 |
| Movable | +3,186 | +0.000027008056640625 |
| MyCar | -152 | -0.0000012885199652777778 |

Verdict: the byte saving is real, but this rt1 adaptive-margin formulation on
the fz4 base violates the pre-registered seg bound and destroys the R8 pose
bank. The derived fallback was byte-closed but not scored because its
pre-registered break-even bound is tighter and the selected margin row already
failed. Score claim remains false; contest-CPU/CUDA pointer unmoved.

## Q3 BLOCKER - pose-null projected reach

Status: **BLOCKED_BY_SINGLE_WRITER_INCOMPLETE_Q3_RECEIPT**.

SB1 did not relaunch Q3 because `.omx/state/main_hot_state.md` marks q3x as a
continuing single-writer lane and says not to relaunch. Durable state found by
SB1: `.omx/tmp/codex_runs/q3x_canonical.log` contains only an n=2 strided smoke
with retained fractions 0.30612244897959184 and 0.25075528700906347 against
threshold 0.9073878056818256; it wrote the smoke JSON to `/tmp`, which is not
acceptable persisted evidence. The expected durable
`.omx/research/ddm_q3x_q3_convergence_measurement_20260803.json` and
`.omx/tmp/codex_runs/q3x_canonical.last.txt` were absent in the searched scope.

Disposition: no n600 row. This is a custody/ownership blocker, not a Q3 family
negative. Fire order: recover or terminally close the q3x single-writer lane,
then run the >=32 matched-base Q3 reach receipt to durable non-`/tmp` custody
before any n600 spend.

## sq1 BLOCKER - uncap convergence test

Status: **BLOCKED_CAP_ARTIFACT_NOT_CONVERGED**.

Existing durable receipts consumed:

- `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32.json`:
  25-step solved paint, `eta_net_pooled=0.7895095948827292`, no explicit
  convergence stop reason.
- `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32_uncap50_cw1.json`:
  50-step convergence-patience run, `eta_net_pooled=0.8620042643923241`,
  `d_pose_after_mean=0.04925062965230609`, cap census `31/32`
  `iteration_cap_best_at_cap` and `1/32` `iteration_cap_before_plateau`.

Disposition: no n600 row. The uncap50 receipt improves eta but is still
cap-bound, so it does not satisfy the charter's convergence-tested stop
requirement and is not a plausible R8-safe scorer spend. Fire order: rerun the
same 32 pairs to an explicit non-cap convergence stop, aggregate, then build
and score only if the pose-bank-accounted pre-score can beat the current
own-vehicle best.

### sq2 follow-up, uncap100 n32

Status: **MEASURED n32 FLOOR, INSTANCE NEGATIVE**.

Receipts:

- `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap100_sq2.json`
  (`dc7ecfe5c1578cc6a7f2668c070f04251b7e570a3e288d2789364d4e8ecead0b`, 188,487 bytes)
- `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32_uncap100_sq2.json`
  (`f6d5ef091fd574d34fbc06cf4230c13a4b1654db94600b6fcf822d221f1c113a`, 6,285 bytes)
- `.omx/research/ddm_sq2_20260804/sq2_gate_verdict.json`

Measured eta curve, same selected n32 pairs:

| cap | eta_net_pooled | stop census |
|---:|---:|---|
| 25 | 0.7895095948827292 | no explicit stop reason; legacy cap artifact |
| 50 | 0.8620042643923241 | 31/32 `iteration_cap_best_at_cap`; 1/32 `iteration_cap_before_plateau` |
| 100 | 0.9112579957356077 | 21/32 `iteration_cap_best_at_cap`; 11/32 `iteration_cap_before_plateau`; 0/32 converged |

Gate arithmetic:

- DERIVED pre-pose delta S from A3 repricing: `-0.13744489822327935`.
- MEASURED subset pose mean after solved paint: `0.07768548923741037`, so `sqrt(10*d_pose)=0.8813937215422536`.
- R8 guard: FAIL. Erosion vs the `0.0845` pose bank is `+0.7968937215422536`, far above the `+0.005` allowance.
- Pose-accounted projected S vs prompt baseline `0.7539807296911207`: `1.413429553010095`.

Disposition: no receiver-closed n600 build and no full-n600 scorer row. The
uncap100 result is a higher floor, not a convergence receipt, and it fails the
R8 pose-bank gate on the measured subset. Optional continuation is a 200-step
rung with the same selected pairs and a new receipt, but only as another floor
measurement; no n600 spend before non-cap convergence and R8-safe pose
accounting both pass.
