# ddm_qo1 repair-stream optimal form

Status: BYTE-CLOSED CANDIDATE QUEUED; full n600 scorer NOT fired.

Axis: `[macOS-CPU frozen-scorer advisory]` for changed-pair forwards. This is not a contest
score and not a promoted frontier row until the fleet scorer owner runs the queued n600 verdict.

## Baseline

Live own-vehicle baseline from fz4:

| row | S | archive bytes | d_seg | d_pose | axis |
|---|---:|---:|---:|---:|---|
| `sub_final` | 0.7541459 | 358,084 | 0.00431179 | 0.00071459 | `[macOS-CPU advisory]` |

Source archive:
`/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/archive.zip`
(`ad5dd0e4fbe5b13ab53a5995a6d77cc558c25f40b63f894ea50ad336bd50fb66`).

The shipped F0PR1 section is `k=6`, 21 repaired pairs, LZMA1-coded, 4,312 section bytes.

## Coder Race

Measured over the real shipped 21-pair F0PR1 coefficient body:

| coder | section bytes | body bytes | receiver-supported here | result |
|---|---:|---:|---|---|
| pair-bitpack | 4,081 | 3,980 | yes | best |
| Brotli q11 | 4,122 | 4,021 | yes | +41 B |
| LZMA1 raw | 4,312 | 4,211 | yes | current, +231 B |
| SMEVR nibble | 4,543 | 4,442 | no | real token-coder round trip, but loses |
| raw int16 | 4,637 | 4,536 | yes | loses |

The pair-bitpack coder is lossless signed bit-width packing per repaired pair. It is now consumed by
the same F0PR1 parser as coder id 3; no video-derived table moved into inflate code.

## Selective Pair Verdict

Waterfill was rerun from the saved `pu2_tail_solve_{a,b}` candidates and the fz4 damage table. The
best selection remains the shipped 21 pairs:

`74, 523, 18, 44, 16, 22, 42, 9, 275, 94, 539, 10, 58, 164, 67, 142, 197, 90, 106, 129, 144`

First rejected pair:

| pair | d_pose gain | trial archive bytes | trial S | verdict |
|---:|---:|---:|---:|---|
| 108 | 0.0012719061 | 358,048 | 0.75399655 | rejected |

Verdict scope: formulation/current saved k6 tail-solve candidates on the pu2 `sub_final` base.

## Adaptive k Result

Uniform cropped-current-coefficient `k<=4` variants are real receiver-compatible probes, but they are
negative on score:

| k | section bytes | archive bytes | changed-pair d_pose mean | predicted S | delta vs live |
|---:|---:|---:|---:|---:|---:|
| 1 | 227 | 353,973 | 0.02729205 | 0.79519190 | +0.04104603 |
| 2 | 591 | 354,346 | 0.03507617 | 0.80565015 | +0.05150429 |
| 3 | 1,181 | 354,944 | 0.14262210 | 0.90591359 | +0.15176772 |
| 4 | 1,998 | 355,769 | 0.09328090 | 0.86697120 | +0.11282533 |
| 5 | 3,043 | 356,794 | 0.03326754 | 0.80497618 | +0.05083031 |
| 6 | 4,312 | 358,084 | 0.00066552 | 0.75414596 | +0.00000010 |

Scope: cropped existing k6 coefficients, not a fresh quant-aware lower-k solve. This falsifies the
cheap truncation form, not the family of newly solved lower-k coefficients.

## Byte-Closed Candidate

Candidate:
`/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip`
(`d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`).

Ledger closes: `residual_bytes=0`, `payload_reencodes_identically=true`.

| row | archive bytes | d_seg held | d_pose predicted | predicted S | delta vs live |
|---|---:|---:|---:|---:|---:|
| `qo1_pairbit` | 357,836 | 0.00431179 | 0.0007145917 | 0.75398083 | -0.00016503 |

Arithmetic:

`100*0.00431179 + sqrt(10*0.0007145917) + 25*357836/37545489 = 0.75398083`

Seg-invisibility control: 8/8 changed pairs matched the zero-frame0 SegNet control through the actual
receiver path and frozen CPU SegNet. All 21 changed pairs were re-forwarded through the actual receiver
for PoseNet; the d_pose mean matches the shipped row because the new coder is lossless.

## Boundaries

- Full n600 was NOT run; queued in `.omx/research/scorer_batch_20260804.md`.
- MPS was not used as authority.
- `upstream/` was not edited.
- The exact-score pointer did NOT move in this unit. This is a byte-closed candidate plus queued
  verdict, not a promoted row.

Receipt JSON:
`.omx/research/ddm_qo1_repair_stream_optimal_form_20260804.json`.
