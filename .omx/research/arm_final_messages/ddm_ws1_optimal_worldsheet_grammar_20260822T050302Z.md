The optimal-form explicit worldsheet is a `FAMILY_NO_GO`: **918,904 B lossless** and **885,750 B at the 0.00116 tolerance**, both receiver-closed over all 600 fields.

| Leg | Selected bytes | Gate | Excess | Verification |
|---|---:|---:|---:|---|
| Lossless | **918,904** | 130,000 | +788,904 | 0 mismatches |
| q2 tolerance | **885,750** | 110,000 | +775,750 | 136,839 / 117,964,800 changed |

Both deterministic repeats are byte-identical:

- Lossless SHA-256: `046926e6f2ee7d8532d48a96d14b31aef30791007643296326c618ea5e55fc75`
- Tolerance SHA-256: `2998625473f23674a432a62aa56a8d722a9e5869425f6fa2159dc1286b6d7243`

### Mechanism attribution

| Form | Lossless B | q2 B |
|---|---:|---:|
| ξ without persistent IDs | 1,047,602 | 1,003,771 |
| Persistent IDs, static predictor | 930,767 | 890,369 |
| Persistent IDs + ξ, generic | 939,390 | 907,901 |
| Winning specialists | **918,904** | **885,750** |

- Persistent identities saved 108,212 B losslessly and 95,870 B under q2.
- ξ advection added 8,623 B and 17,532 B versus static prediction.
- Specialists saved 20,486 B and 22,151 B; only the nonempty shared hood curve won.
- Generic persistent ξ was 939,390 B versus ws0 ordinal’s 318,885 B.

### Per-stratum price

Values are lossless/q2 bytes.

| Stratum | ws0 | WS1 generic | WS1 selected |
|---|---:|---:|---:|
| Road↔Lane | 144,092 / 142,462 | 538,668 / 521,916 | 538,668 / 521,916 |
| Road↔Undrivable | 22,893 / 21,674 | 147,598 / 137,758 | 147,598 / 137,758 |
| Road↔Movable | 21,442 / 21,282 | 69,498 / 68,355 | 69,498 / 68,355 |
| Road↔MyCar + Movable↔MyCar | 16,128 / 15,678 | 86,406 / 86,053 | **65,839 / 63,820** |
| Lane↔Undrivable | 350 / 350 | 2,390 / 2,353 | 2,390 / 2,353 |
| Lane↔Movable | 1,426 / 1,443 | 3,629 / 3,716 | 3,629 / 3,716 |
| Lane↔MyCar | 1,067 / 1,050 | 4,482 / 4,487 | 4,482 / 4,487 |
| Undrivable↔Movable | 25,918 / 25,386 | 85,358 / 81,904 | 85,358 / 81,904 |
| Undrivable↔MyCar | 52 / 52 | 44 / 44 | 44 / 44 |
| Shared framing | 36,553 / 36,553 | 1,317 / 1,315 | 1,398 / 1,397 |
| **Total** | **269,921 / 265,930** | **939,390 / 907,901** | **918,904 / 885,750** |

The lane polynomial, horizon, and movable-template specialists all lost their real coder races. The hood’s empty 0.75 form was rejected as mechanism-invalid; the adopted 0.5 form contains 156 actual shared-curve sites.

Held-out diagnostics were genuine: the lane fit used even rows and tested 345,200 odd-row pixels, giving 0.04340 m median and 0.26105 m p95 lateral error. The horizon interleaved holdout gave 0.32258 px median and 2.0 px p95.

Artifacts:

- [Governed memo](/Users/adpena/Projects/pact/.omx/research/ddm_ws1_optimal_worldsheet_grammar_20260821.md)
- [Harness](/Users/adpena/Projects/pact/experiments/ddm_ws1_optimal_worldsheet_grammar.py)
- [Focused tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_ws1_optimal_worldsheet_grammar.py)
- [Retained result](/Volumes/VertigoDataTier/pact/ddm_ws1_optimal_worldsheet_grammar/retained/FINAL_RESULT.json)

Landed as commits `019dc8c08a` and `9d2522a411`. Tests pass 5/5; both Python files received two review passes; payload-retention findings are zero. Developer preflight is 18/25 green: the WS1 lane-registration finding was cured, while seven remaining reds are pre-existing out-of-scope repository findings.

**GESTALT-DELTA:** persistent identities help, but carried proxy ξ hurts, the real lane polynomial loses, and only the hood specialist wins. The explicit partition remains nearly 0.9 MB, so nr1 should favor an implicit sufficient statistic.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN / #1187 nr1; consumer store: the governed memo and retained `FINAL_RESULT.json`; fire trigger: the next nr1 harvest or resume must consume `FAMILY_NO_GO` and exclude an explicit worldsheet from the 90K body.

## LIVE-HYPOTHESES

- An implicit task-space sufficient statistic remains plausible because the explicit curve grammar’s topology is cheap, but its exact innovations dominate by hundreds of kilobytes.
- A genuinely banked cross-pair trajectory could outperform the present ξ operator because this experiment only had the documented nearest-target-pair Pose6 proxy; reopening requires that new motion receipt, not another proxy retune.

## DEAD-ENDS

- The registered explicit-worldsheet family is closed: both full-n600 gates fail by more than 775 KB.
- Current carried-proxy ξ advection is closed as a rate lever; it costs bytes versus static persistence.
- The coherent-slot LBND2 lane polynomial, step-16/32/64 horizon models, and g8/g16 movable templates all lose their per-stratum races.
- The empty 0.75 hood template is mechanism-invalid and must not be retried as a “shared curve.”
- Own-vehicle frontier: UNMOVED — dx2 S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`; WS1 measured only `[macOS-CPU advisory, scorer-free n600 coder]` bytes.