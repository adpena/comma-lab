The exact pointer did not move. ddm_js3 is **BOUNDED EXISTENCE / NOT ADMITTED**: learned conditioning crossed the robustness bar, but not while satisfying the pose guard.

| Screen | Projected robust flips | Projected total flips | Pose delta | Coded bytes |
|---|---:|---:|---:|---:|
| 20-step live | **−377** | +1,467 | +0.010799, fail | 751 |
| 8-step pose-control EMA | 0 | −57 | **−1.012e-6, pass** | 819 |

Thus the gradient is alive through the real receiver delta, camera uint8 lattice, and frozen scorers, but no candidate passes all three T4 gates. F1/F2/F3 were not eligible, and no long burn or T4 dispatch occurred.

Implemented and landed:

- [Runner](/Users/adpena/Projects/pact/experiments/ddm_js3_learned_implicit_conditioning.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_js3_learned_implicit_conditioning.py)
- [Research receipt](/Users/adpena/Projects/pact/.omx/research/ddm_js3_learned_implicit_conditioning_20260812.md)
- [Compact handoff](/Volumes/VertigoDataTier/pact/ddm_js3_20260812/FINAL_HANDOFF.json)
- [Sealed MAIN recipe](/Volumes/VertigoDataTier/pact/ddm_js3_20260812/SEALED_MAIN_RECIPE.json)
- [Retained payload root](/Volumes/VertigoDataTier/pact/ddm_js3_20260812/)

Commits: `eb450d1281` and payload-retention fix `387adcbfdd`. Six tests, Ruff, compilation, two review passes per Python file, serializer hooks, parse-back, custody validation, and the targeted payload-retention census passed. The broader developer preflight still reports eight unrelated existing codebase gates; isolation found zero js3 hits.

Effective frontier remains cp135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. Own-vehicle frontier remains lc2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.

Queued fire order, verbatim:

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js3_20260812/main_burn`. **Fire trigger:** MAIN owns the training leg, no full-n600 scorer job is active, the target-host memory preflight passes, and js2b `ROUTE.json` is consumed as a prohibition on direct two-W4 continuation.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js3_20260812/main_burn`. **Fire trigger:** MAIN owns the training leg, no full-n600 scorer job is active, the target-host memory preflight passes, and js2b `ROUTE.json` is consumed as a prohibition on direct two-W4 continuation. Run the sealed pose-guarded 25/100/300-step recipe; admit T4 only after receiver integration and an n600 projection passes every gate.

## LIVE-HYPOTHESES

- A longer pose-guarded schedule may turn the control EMA’s tie-fragile gains into robust repairs: the two screens separately established robust reachability and pose-safe improvement.
- Hidden width 8 remains plausible because its real initial payload is 1,282 B, leaving 218 B below the gate.
- Locking the pose-safe EMA basin before increasing the δ-hinge may resolve the observed Seg/Pose tradeoff.

## DEAD-ENDS

- Direct two-W4 continuation: js2b already closed it for zero robust movement.
- Standalone edge or pose probability tables: sr1 measured −2 B and +43 B.
- Scoring float-QAT weights instead of their coded parse-back: the receiver consumes coded weights.
- Treating ordinary flip gains as robust progress: the pose-control gains were entirely tie-fragile.
- Dispatching either retained module to T4: neither passes the combined robust, byte, pose, and receiver-integration gates.