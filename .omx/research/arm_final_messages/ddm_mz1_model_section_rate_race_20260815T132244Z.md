MZ1 is complete and landed in commit `31711735fa`. It found no recoverable serialization gap and produced no smaller archive.

The claimed 52,566-byte gap compared an HPAC-only trainer estimate against a wrapper containing three objects:

`70,557 = 13,619 HPAC + 34,763 semantic + 22,161 carrier + 14 wrapper`

Raw HPAC is 17,996 bytes, only 5 bytes above the 17,991-byte estimate, and compresses to 13,619 bytes. The existing split-Brotli q10/q11/q11 representation won all 8 complete lossless rows:

| Best alternative | Bytes | Loss vs current |
|---|---:|---:|
| Current RX1M | 70,557 | 0 |
| Per-section Brotli q11 | 70,598 | +41 |
| Raw LZMA2 | 70,688 | +131 |
| Whole raw LZMA2 | 70,759 | +202 |
| Per-tensor mixed | 71,038 | +481 |
| SMEVR r7 | 72,611 | +2,054 |
| Adaptive RC64 | 83,753 | +13,196 |

All recoded payloads were retained. The rebuilt archive and deterministic repeat are byte-identical to RX2: 183,502 bytes, SHA-256 `e3e6f440…91c19d3`. The existing CPU token/raw-identity receipt transfers to these identical bytes; no new CPU decode or scorer run was performed.

Artifacts:

- [Receipt memo](/Users/adpena/Projects/pact/.omx/research/ddm_mz1_model_section_rate_race_20260815.md)
- [FINAL_RESULT.json](/Volumes/VertigoDataTier/pact/ddm_mz1_model_section_rate_race/FINAL_RESULT.json)
- [Runner](/Users/adpena/Projects/pact/experiments/ddm_mz1_model_section_rate_race.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_mz1_model_section_rate_race.py)

Verification: Ruff and compilation passed; `4 passed`; payload-retention census found 0 violations; both Python files received two review passes. The repository-wide developer preflight remains red on 8/25 gates, but bounded adjudication found zero MZ1 references in any violation.

T4 is `FOLDED`: evaluating the identical archive would duplicate authority call `fc-01M02QMN3SQ9SNHXZMRWXYEJEW`. Own-vehicle frontier remains `S=0.1600920261571558 @ 183,502 B [contest-CUDA T4, n600]`. MZ1 did not move it.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN-assigned representation successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mz1_model_section_rate_race/FINAL_RESULT.json`; fire trigger: a representation-changing candidate saves at least 15,153 complete archive bytes, is shipping-receiver-closed, and proves exact token plus decoded-raw identity before T4 dispatch.

## LIVE-HYPOTHESES

- Changing the semantic/carrier representation remains plausible because those objects and framing occupy 56,938 bytes; MZ1 tested their lossless recoding, not alternative representations.
- Joint model/token retraining remains plausible because MZ1 held decoded tokens fixed.
- Shared-context tensor framing might recover a small number of bytes because some tensor-local trials won before framing, though nothing measured suggests it can reach 15,153 bytes.

## DEAD-ENDS

- The 52,566-byte serialization gap is false attribution, not recoverable coder slack.
- The nine training-only fp32 bit-depth buffers are already absent.
- The learned 259-byte depth vector cannot legally be derived from config or seed.
- The exact container is split Brotli, not XZ.
- All seven tested same-decoded-section replacements lose on this e480b instance.
- Re-firing T4 on the rebuilt winner would evaluate the already-authoritative byte object.