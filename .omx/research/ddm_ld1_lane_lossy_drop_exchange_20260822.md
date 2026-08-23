# DDM LD1 Lane-lossy drop exchange — real DX2 re-encode curve; joint verdict withheld behind the exclusive scorer lane

`date_utc: 2026-08-22` · `arm: ddm_ld1_lane_lossy_drop_exchange` ·
`axis: [macOS-CPU advisory / scorer-free exact full-stream re-encode]` ·
`score_claim: false` · `promotion_eligible: false`

## ANSWER FIRST

LD1 materialized six nested, genuinely lossy n600 Lane→Road fields and priced every one through
the unchanged shipped DX2 19-member HPAC/RC64 law. **All six made the archive larger:** +196, +279,
+824, +1,528, +598, and +21 B. The 60,000-edit rung came closest to rate neutrality at 180,389 B,
but the unedited 180,368 B control is the measured rate optimum. Thus the registered high-BL1-cost
Lane→Road ladder buys **zero bytes** toward the 42,382 B demand; every rung consumes additional bytes.
The real byte curve is the result this arm owns.

The joint Lane-drop exchange is **not adjudicated**: this charter does not grant the sole n600 scorer
lane, so the real-path per-class d_seg, final-argmax fields, introduced-error survival ratios, joint
rate+seg ΔS, and optimum remain queued. A byte-only minimum is not a score minimum and is not called
one here. Pose is outside the charter formula as well, so no full-contest-score claim is made.

The contest pointer did **not** move. No shipping candidate was built; the retained ZIPs are
measurement containers emitted by the exact re-encoder.

## 1. SOURCE REPRODUCTION AND DENOMINATORS

All joins are n600 over `600 × 384 × 512 = 117,964,800` pixels. GT lineage is the pinned
contest-authority DALI/NVDEC field inherited through the DX2/MS9 custody chain. The arm verified the
exact source bytes before materializing a rung:

| source | bytes | SHA-256 |
|---|---:|---|
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| shipped RC64 token stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` |
| TO2/BL1 decoded token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| BL1 per-position integer-frequency cost field | 943,718,400 | `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86` |
| DALI GT argmax field (`.npy` container) | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` |

Direct replay of the retained fields, not transcription from the memos, reproduced:

| item | numerator | denominator | fraction / derived value |
|---|---:|---:|---:|
| GT Lane area | 690,754 px | 117,964,800 px | 0.5855594211% |
| GT Lane shipped-symbol cost | 305,463.969473 bits | 910,209.280609 bits | 33.5597511452% |
| GT Lane byte equivalent | 38,182.996184 B | 42,382 B sub-0.12 demand | 90.09% |
| Lane cost per position | 305,463.969473 bits | 690,754 px | 0.4422181695 bpp |
| Lane enrichment | 0.4422181695 bpp | 0.0077159397 mean bpp | 57.31228964× |
| MS9 final errors | 23,757 | 117,964,800 | d_seg = 0.0002013905843 |
| MS9 transmitted-label errors | 9,182 | 117,964,800 | — |
| MS9 representation errors surviving final argmax | 2,264 | 9,182 | 24.65693749% |
| MS9 realization repairs | 6,918 | 9,182 | 75.34306251% |
| MS9 manufactured final errors | 21,493 | 23,757 | 90.47017721% |
| MS9 GT-Lane representation errors surviving final argmax | 571 | 1,907 | 29.94231778% |

The two MS9 additive identities were replayed from the retained packed masks:
`9,182 = 2,264 + 6,918` and `23,757 = 2,264 + 21,493`.
The Lane-specific 571/1,907 survival row was also replayed from MS9's packed class masks. It is a
more local prior than the body-wide 24.66%, but it still describes naturally occurring baseline
errors, not these deliberate high-cost Lane→Road edits, so neither ratio is transferred as a result.

## 2. THE LOSSY FORMULATION

The reference field contains 688,847 positions where both DALI GT and the shipped token are Lane.
LD1 ranks those positions by BL1's exact incumbent shipped-symbol cost, descending, with flat raster
index as the deterministic tie-break. The nested rungs change the highest-cost 2,500, 5,000, 10,000,
20,000, 40,000, and 60,000 correct Lane tokens from class 1 (Lane) to class 0 (Road).

This is a class-merge formulation, not a generic Lane-family theorem. It is lossy on its face:
each changed symbol was correct before the edit and wrong after it, so transmitted errors added are
exactly the rung count, all on the GT-Lane row; the four other GT-class transmitted-error rows are
zero by construction. Final-argmax collateral is not assumed zero and is specifically queued for
measurement.

| rung | transmitted errors added | incumbent cost selected, B equivalent | retained token-field SHA-256 |
|---:|---:|---:|---|
| 1 | 2,500 | 3,785.040 | `c45979acb7a87bdae41fe23d67c9efd10661d5320e5e0c84f9d863a743b3831e` |
| 2 | 5,000 | 6,671.602 | `6c210dd19eefb2b67dad5c5f93ee8008a625b8aea50e685553ee5335f179f000` |
| 3 | 10,000 | 11,148.952 | `297cee64f3e1438b985f9b242d6405ad5521b5cf320865390bc0ca105fe8351d` |
| 4 | 20,000 | 16,987.471 | `7251367a078796a12c2302d726d2d5b1941c9d35d5755745fc664f29de0344fb` |
| 5 | 40,000 | 23,411.881 | `03ce7bd8a8498ea2a1fc61a0191d0c9eeab3e5ff729e7d522dc07f64add08093` |
| 6 | 60,000 | 27,319.051 | `15018481bd8007dd9099d1b67d5e8014283465d062a34ba3f06b3450758b5878` |

The incumbent-cost column is descriptive only. Autoregressive context changes after each edit, so
it is never used as a byte estimate or score input.

## 3. EXACT RE-ENCODER CONTROL AND REAL BYTE CURVE

`experiments/ddm_jg2_tail_reencode.py` imports the shipped runtime and mirrors its actual group order,
previous-frame conditioning, table, 19-member adaptive corrector, probability quantizer, and RC64
coder. Its n600 unedited control must reproduce all 113,777 shipped stream bytes before a rung delta
is trusted. Every encode checkpoints the full adaptive state every 20 frames and retains the stream,
per-frame bit ledger, measurement archive, and receipt.

<!-- RATE_TABLE_BEGIN -->
The unedited control reproduced the shipped 113,777-byte token stream byte-for-byte at SHA-256
`e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5`. That makes all six archive
deltas below trustworthy. Negative demand share means the rung consumed bytes instead of serving the
42,382 B demand.

| Lane→Road edits / transmitted errors added | archive B | ΔB vs shipped | bytes saved | share of 42,382 B demand | ΔS_rate | d_seg by class | final flips added | survival vs 24.66% | rate+seg ΔS |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 2,500 | 180,564 | +196 | −196 | −0.46246% | +0.0001305084 | QUEUED | QUEUED | QUEUED | QUEUED |
| 5,000 | 180,647 | +279 | −279 | −0.65830% | +0.0001857746 | QUEUED | QUEUED | QUEUED | QUEUED |
| 10,000 | 181,192 | +824 | −824 | −1.94422% | +0.0005486678 | QUEUED | QUEUED | QUEUED | QUEUED |
| 20,000 | 181,896 | +1,528 | −1,528 | −3.60530% | +0.0010174325 | QUEUED | QUEUED | QUEUED | QUEUED |
| 40,000 | 180,966 | +598 | −598 | −1.41098% | +0.0003981837 | QUEUED | QUEUED | QUEUED | QUEUED |
| 60,000 | 180,389 | +21 | −21 | −0.04955% | +0.0000139830 | QUEUED | QUEUED | QUEUED | QUEUED |

The curve is strongly non-monotone. It does not support interpolation between rungs, and the near-return
to baseline at 60,000 is not a saving. The charter's prior-law prediction required a rung freeing more
than 8,000 B; its rate precondition is falsified for every registered instance before any absorption
assumption is applied. This does not prove all Lane-lossy formulations rate-negative.
<!-- RATE_TABLE_END -->

Rate arithmetic uses the pinned score law, never `evaluate.py`'s rounded display:
`ΔS_rate = 25·Δbytes/37,545,489`, so one archive byte is `6.658589531e-7 S`.

## 4. JOINT COLUMNS WITHHELD, NOT ESTIMATED

The shipped allocation is 180,368 B, 9,182 transmitted-label errors, and 23,757 final errors
(`d_seg = 23,757/117,964,800`). For each lossy rung, the following required columns remain null until
the governed scorer consumer writes them:

- terminal frozen-SegNet argmax field after actual render → R → uint8;
- final errors and d_seg with Lane separate from Road, Undrivable, Movable, and MyCar;
- final flips added relative to the exact DX2 baseline;
- introduced-error survival fraction = final flips added / transmitted errors added;
- comparison with MS9's body-wide 24.65693749%;
- `ΔS = 25·Δbytes/37,545,489 + 100·Δd_seg` and the measured optimum or all-positive close.

The prior-law prediction therefore remains untested. In particular, the arm does not use 24.66% to
manufacture a final-flip count and does not interpolate d_seg between rungs.

## 5. RECALL EVIDENCE

Searches covered the full `.omx/research/` memo/receipt corpus, canonical equation registry, canonical
research index, `sub015_DAG_*` FEED text, design/SPEC surfaces, and canonical task ledger. Content
queries included `lane lossy`, `lane drop`, `class-aware lossy`, `token drop`, `token-granular`,
`per-position bit`, `BL1`, `MS9`, `FS2`, and `FS3`.

Beyond the charter seeds, six results changed the execution discipline:

1. `ddm_fs2_rc4_drop_carrier_resolve_20260820.md` measured only 1,022 B of a modeled 11,716.7 B
   confidence-drop credit (8.72%) and a second sparse threshold cost 37 B. That made the real n600
   re-encode mandatory for every LD1 rung; no BL1 cost sum was promoted to bytes.
2. `ddm_fs3_jg5_real_price_reopen_20260820.md` measured 664 B over 997 removed tokens but then refused
   the candidate on a +0.03579520 joint loss from the stale pose carrier. That reinforced the rule that
   a positive rate leg cannot select a rung.
3. `ddm_gr1_granularity_rerace_DAG_FEED_20260730.md` found every token-granular drop candidate dominated
   on a different SMEVR vehicle while cell-drop won. Its scope is INSTANCE/FORMULATION and its vehicle
   differs, so it did not close the GT-Lane DX2 merge; it did forbid any family-level claim here.

4. The canonical `ddm_cf2` equation carries the direction-dependent re-encode warning (~0.09× modeled
   credit toward argmax versus ~0.92–0.93× actual price away from argmax) and explicitly forbids transfer
   across thresholds. LD1 measures rather than applies either factor.
5. `ddm_rc4_rung4_token_drop_verdict_20260816.md` measured severe pose harm for a token-drop arm on
   the older HV1 vehicle. That result cannot be transferred to this DX2 Lane→Road instance, but it
   prevents the charter's rate+seg currency from being described as a full contest-score delta: pose
   is excluded here and remains an explicit risk for any later shipping proposal.
6. `ddm_rl1_roadlane_interface_price_20260803.md` found a real Brotli-q11 price of 1.1604128 B/flip
   for a losslessly represented Lane-mask crop, but only as a 600×mean(n32 evenly-strided) projection.
   That is a different receiver surface and is neither an n600 temporal price nor evidence that dropping
   shipped DX2 Lane tokens has the same marginal cost. It was retained as a representation lead, not
   substituted for LD1's real full-stream re-encode.

One source-custody discrepancy is bounded rather than hidden. The live MS9 `MASK_MANIFEST.json`
validates every packed-mask byte and reproduces the two additive identities plus the 571/1,907 Lane
row, but the committed MS9 memo names `872e0b19…18cc` for `MS9_FIELD_REPLAY.json` while the currently
retained file is `c1449419…ddd5`; no file with the memo's older hash was found in the bounded MS9
store. LD1 therefore relies on the live manifest-backed masks for the numeric join and does not claim
that the replay JSON has the memo-published identity.

## 6. CUSTODY, STORAGE, AND COMMANDS

All current LD1 outputs are retained under the charter's explicit local-disk opt-in:
`/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/measurement_v1/`.
Neither full SSD tier was used for current-run outputs; the APDataStore and Vertigo paths below are
pinned, read-only inputs. Three scalar receipts from the aborted, pre-charter-storage-corrected attempt
remain under `/Volumes/VertigoDataTier/pact/ddm_ld1_lane_lossy_drop_exchange/measurement_v1/`; they are
not inputs to this result, contain no retained rung payloads, and were preserved rather than deleted.
The current materializer preserved the six 117,964,800-byte token fields, six re-encoder
edit payloads, the cost ranking, selected indices, each rung's full transmitted-error field, and five
GT-class transmitted-error masks. The rate stage retains every emitted RC64 stream, per-frame ledger,
adaptive-state checkpoint, and measurement archive. `MANIFEST.json` hashes the completed store.

The exact re-encoder invocation family was:

```text
TAC_JG2_RC64_SOURCE=/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/rc64_backend.c \
  .venv/bin/python experiments/ddm_jg2_tail_reencode.py \
  --stage {control|encode} \
  --store /Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/measurement_v1/rate \
  --runtime-root /Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2 \
  --pointer-archive /Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/archive.zip \
  --expect-pointer-sha256 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674 \
  --tokens /Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/measurement_v1/retained/fields/decoded_tokens_instrumented.u8 \
  --edits <retained rung edits.npz> --tag <rung tag> --frames 600 --checkpoint-every 20 --resume
```

The control omits `--edits`, `--tag`, and the optional pointer-SHA argument because its stage does not
splice a measurement archive; the source archive and parsed stream are nevertheless pinned above.

## 7. DISPOSITION AND FIRE ORDER

- **FIRED AND RETAINED:** six nested Lane→Road decoded-token fields, owner LD1, consumer the exact
  re-encoder and future scorer join, store `measurement_v1/retained/`.
- **FIRED AND RETAINED:** unedited n600 re-encoder control plus six real full-stream encodes, owner
  LD1, consumer `RATE_CURVE.json`, store `measurement_v1/rate/`.
- **QUEUED-WITH-A-FIRE-ORDER:** owner MAIN / exclusive n600 scorer-lane custodian; consumer LD1 joint
  exchange adjudication; consumer store `measurement_v1/scorer/`; fire only after MAIN explicitly
  transfers the unique scorer lane, all other n600 scorer jobs are terminal, the six candidate hashes
  validate, and a fresh local-disk preflight admits each retained final-argmax field plus lossless custody
  of any raw output. Score the rungs serially through `tools/fire_local_advisory.py`, retain the five
  class rows and final fields, reproduce the exact DX2 23,757-error positive control, then write the
  survival and joint-ΔS joins. `SCORER_FIRE_ORDER.json` is the machine-readable order.

`verdict_scope: INSTANCE` — the n600 byte curve is MEASURED for the six registered Lane→Road
cost-ranked instances on DX2. Its narrow rate verdict is all-positive Δbytes, with the unedited control
best. No joint negative, joint optimum, FORMULATION close, or FAMILY close exists before the queued
scorer join.

## 8. VERIFICATION

- Re-ran materialization against all source pins after the implementation was final; it reproduced the
  BL1/MS9 joins and validated every retained rung artifact. The receipt binds implementation SHA-256
  `40e63c8af4a2b6fc15423b503d737690691548ab6d48ce7bd5b795873a6bfb58`.
- Independently rehashed all 123 entries in `MANIFEST.json` (1,331,700,135 retained bytes), verified all
  six measurement archives with `unzip -tqq`, and recomputed all six archive deltas and rate terms.
- `ruff check`, Python bytecode compilation, and the ALWAYS-KEEP payload audit passed. The payload audit
  discovered 41,893 Python files, parsed 3,184 producer candidates, and found zero LD1 findings.
- Two genuine whole-file review passes cover all 22 Python entities:
  `ld1_adversarial_pass_1` and `ld1_adversarial_pass_2`. No review override was used.

**Own-vehicle frontier: S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`, archive
`976f706d…f6de6674` — UNMOVED by LD1.**
