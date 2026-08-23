The LD1 scorer-free scope is complete and committed as `5e8d6011ba`.

| Lane→Road edits | Real archive ΔB | ΔS_rate |
|---:|---:|---:|
| 2,500 | +196 B | +0.0001305084 |
| 5,000 | +279 B | +0.0001857746 |
| 10,000 | +824 B | +0.0005486678 |
| 20,000 | +1,528 B | +0.0010174325 |
| 40,000 | +598 B | +0.0003981837 |
| 60,000 | +21 B | +0.0000139830 |

Every registered rung increased rate. The unedited 180,368 B control remains the rate optimum; this ladder contributes zero bytes toward the 42,382 B demand.

The control reproduced the shipped 113,777-byte RC64 stream byte-for-byte. All payloads are retained locally, and all 123 manifest entries, six ZIPs, rate calculations, two review passes, and payload-retention checks verified successfully.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_ld1_lane_lossy_drop_exchange_20260822.md:7)
- [Measured rate table](/Users/adpena/Projects/pact/.omx/research/ddm_ld1_lane_lossy_drop_exchange_20260822.md:93)
- [Implementation](/Users/adpena/Projects/pact/experiments/ddm_ld1_lane_lossy_drop_exchange.py:1)
- [RATE_CURVE.json](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/measurement_v1/RATE_CURVE.json)
- [SCORER_FIRE_ORDER.json](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/measurement_v1/SCORER_FIRE_ORDER.json)

The real-path per-class d_seg, final flips, absorption ratio, and joint rate+seg ΔS were not measured because LD1 did not own the exclusive n600 scorer lane. No absorption value was projected, no shipping candidate was built, and pose remains outside this charter’s currency. The MS9 mask counts reproduced, but its live replay-JSON hash differs from the hash named in the committed MS9 memo; LD1 relies only on the validated manifest-backed masks.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / exclusive n600 scorer-lane custodian; consumer store: `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/measurement_v1/scorer/`; action: score all six rungs serially through `tools/fire_local_advisory.py`, retain per-class final fields, and compute absorption plus joint ΔS; fire trigger: MAIN explicitly transfers the lane, no other n600 scorer is active, candidate hashes validate, and the local storage preflight passes.

## LIVE-HYPOTHESES

- Deliberate Lane errors may still receive receiver-path absorption resembling MS9’s 571/1,907 Lane ratio, because the same realization path repairs many naturally occurring representation errors. Only the queued scorer join can test transfer.
- A rung could improve final d_seg enough to offset its small positive rate cost, especially the +21 B 60,000-edit rung, because downstream errors are predominantly manufactured and the curve is highly nonlinear.

## DEAD-ENDS

- The six registered highest-cost Lane→Road instances as a rate-saving ladder: closed at `verdict_scope: INSTANCE`; every real re-encode grew.
- BL1 incumbent cost sums as byte predictions: closed; adaptive context changed both magnitude and sign.
- Lossless probability-only controls as a distortion exchange: closed by LX2 because decoded symbols do not change.
- Transferring MS9’s 24.66% or 29.94% absorption into LD1 without scoring: closed as an unsupported proxy.
- Selecting a byte-only winner, shipping these measurement archives, or claiming a full score: closed by the missing scorer and pose axes.
- Frontier movement: none. Own-vehicle frontier remains **S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, archive `976f706d…f6de6674`.

