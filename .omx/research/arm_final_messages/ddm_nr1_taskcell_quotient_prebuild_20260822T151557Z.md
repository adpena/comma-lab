Built, measured, retained, tested, and landed the executable NR1 prebuild in commit `b3647a73ca`.

| Row | Packet | Token agreement | Token-only ceiling |
|---|---:|---:|---:|
| K32 | **69,004 B** | 98.6786% | **2,391 B below** |
| K64 | 79,876 B | 98.7467% | 8,481 B above |
| K128 | 91,630 B | 98.8848% | 20,235 B above |

K32 projects to 135,595 B if every non-token DX2 byte and distortion remain fixed. Thus NR1 is not killed by rate, but matched distortion is unmeasured: K32 changes 1,558,833 tokens. No scorer ran, r9 was untouched, full raw receiver closure was not claimed, and the frontier did not move.

Artifacts:

- [Governed memo](/Users/adpena/Projects/pact/.omx/research/ddm_nr1_taskcell_quotient_prebuild_20260822.md)
- [Quotient implementation](/Users/adpena/Projects/pact/src/tac/optimization/nr1_taskcell_quotient.py)
- [Retained runner](/Users/adpena/Projects/pact/experiments/ddm_nr1_taskcell_quotient_prebuild.py)
- [Adversarial tests](/Users/adpena/Projects/pact/experiments/ddm_nr1_taskcell_quotient_prebuild_test.py)
- [K32 result](/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k32_e8192_v1/RESULT.json)

All four surfaces are real and exact-once consumed. K32 attribution is QPARAM 239 B, QCTX 152 B, QPAIR 52,124 B, and QEVENT 16,489 B. All coder losers, repeats, decoded fields, and renderer counterfactuals were retained.

Verification: 45 targeted tests passed, Ruff/compilation/payload-retention checks passed, two final review passes completed, and commit hooks passed. Broad developer preflight remains red on seven unrelated existing repository gates.

Own-vehicle frontier: `S=0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]`; archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; unchanged.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER / ARMED-NOT-FIREABLE`. **Owner:** MAIN. **Consumer store:** `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/retained/`. **Fire trigger:** r9 is terminal, MAIN has frozen and verified its endpoint archive/raw/logit/margin/argmax/Pose6/fresh-dxi receipts, and no competing NR1/scorer lane is active. **Action:** refit K32/K64/K128 to that primary endpoint, integrate the production active receiver and fresh carrier, then withhold scoring until full receiver and Rule-118 closure pass.

## LIVE-HYPOTHESES

- A task-weighted K32 dictionary may improve agreement without K64’s byte cost because the present dictionary uses raw frequency rather than evaluator-cell value.
- QPAIR may compress below 52,124 B because the current previous/default/direct predictor does not model structured pair motion.
- K64 may work as a joint token+HPAC replacement because it clears that combined ceiling by 5,034 B while preserving more tokens than K32.
- A fresh terminal carrier may compose better with the new field because the actual semantic renderer demonstrably consumes and transforms NR1 output.

## DEAD-ENDS

- Specification-only NR1 work is closed: a real executable packet now exists.
- The July quotient ABI is not an NR1 shortcut because it lacks genuine QCTX, uses a synthetic rank-one renderer, and relies on n24/no-op fixtures.
- K128 in its present frequency-born form is closed by rate; it exceeds the combined token+HPAC ceiling by 6,720 B.
- Integrity-only mutations or four named no-op consumers are closed as receiver evidence.
- These three rows are closed as score claims because none used the frozen primary endpoint, produced a complete raw, integrated a fresh carrier, or ran the scorer.
- Full explicit worldsheet carriage remains closed by its 885,750–918,904 B measured size.