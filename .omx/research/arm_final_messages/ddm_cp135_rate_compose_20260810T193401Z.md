Built and landed a receiver-closed PR135 composition candidate:

- **186,252 B**, SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`
- **−472 B** versus PR135’s 186,724 B
- Derived score if PR135 distortion holds: **0.16195513827824176**
- VP1: −177 B; CAP1 pack: −79 B; HP3: −216 B
- All 117,964,800 RC64 symbols decoded identically. Semantic, carrier, residual, and intended HP3 state passed receiver parse-back.
- Every intermediate and candidate payload is retained on the SSD.

This arm ran no renderer, scorer, Modal job, or exact evaluator. The score is derived, not an authority row, and sub-0.15 was not reached.

Artifacts:

- [archive.zip](/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip)
- [durable memo](/Users/adpena/Projects/pact/.omx/research/ddm_cp135_rate_compose_20260810.md)
- [final receipt](/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/FINAL_RESULT.json)
- [parse-back receipt](/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/PARSEBACK_RESULT.json)
- [exact-eval fire order](/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/EXACT_EVAL_FIRE_ORDER.json)

Verification passed: six tests, strict payload-retention gate, lint, formatting, syntax checks, two review passes, deterministic archive repeat, and serializer commit `7b79fc1c7d6a63dde394a8025d4be009b9f7adee`.

Borrowed substrate: PR135’s learned state, F26 receiver, and CUDA renderer are codexblack’s. Ours-original work is the VP1/CAP1/HP3 composition, exact recodes, coder races, and receiver-equality harness.

Own-vehicle frontier remains **lc2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN exact contest-row owner; consumer store: `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/exact_eval`; fire trigger: claim the sole exact-eval lane, verify the 186,252 B archive SHA and adapted-runtime tree, pass locked-T4 `Brotli==1.2.0` preflight, then run exactly one n600 contest-CUDA `upstream/evaluate.py` row.

## LIVE-HYPOTHESES

- The exact CUDA row will retain PR135’s distortion and score about `0.16195514`: the output-driving semantic, carrier, residual, and all decoded tokens are identical, but literal T4 rendering remains untested.

## DEAD-ENDS

- Same-state ANS: +6 B on control and +9 B on HP3 versus RC64.
- SMEVR: won 0 of 14 exact section races.
- LOTTO renderer recodes: +136 B and +254 B versus selected WANS+Brotli.
- Further RC64 tuning: already only about 0.54 B above its model ideal.
- Direct residual-byte reclaim: PR135 stores no camera-resolution residual field.
- Linear addition of inherited lever savings: disproved by complete-container recounts.