Executed and landed as commit `80be0ebce0`.

The bounded `[macOS-CPU advisory, stratified n32]` result is an INSTANCE negative:

- α=1 and α=1/2 retained robust improvements of −305 and −112 projected flips, but realized pose deltas were `+8.479e-4` and `+1.273e-4`, far above the `2e-6` gate.
- At α≤1/4, robust movement fell to zero while pose still failed. Continuous leakage scaled approximately as α^2.511, but post-uint8 leakage flattened to α^0.957.
- The deterministically seeded hidden-4/8/12 ladder accepted `0/15` proposals. The best proposal reached `−7.798e-6` against the stricter stage budget `−8.585e-6`.
- Bare modules were 801/1,286/1,821 B. All moved zero robust flips, so B/robust-flip is undefined and none was admitted. The selected 801 B null module passed pose at `−9.548e-6` but was correctly rejected as useless.
- F1, F2, and F3 are all ineligible—not fired. F1 has only 15/200 required proposals.
- No archive, full-n600 scorer run, exact score, or long burn was produced. Sub-0.15 was not achieved.

Artifacts: [receipt](/Users/adpena/Projects/pact/.omx/research/ddm_js5_projector_distilled_conditioning_20260812.md), [runner](/Users/adpena/Projects/pact/experiments/ddm_js5_projector_distilled_conditioning.py), [tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_js5_projector_distilled_conditioning.py), [final JSON](/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/FINAL_RESULT.json), [sealed recipe](/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/SEALED_MAIN_RECIPE.json), and [queue annex](/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/QUEUE_ANNEX.md).

Verification: 45 integrated tests passed; both Python files received two review passes; 179/179 retained custody records verified; payload-discard findings `0`. The payload store is 7.5 GiB. Resume reproduced the α and rung receipts byte-identically.

Own-vehicle frontier unchanged: `lc2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Action:** content-distinct realized-acceptance proposal extension. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200`. **Fire trigger:** MAIN provides a representation-level conditioning proposal source that is not another replay of this five-shrink first-step ladder; retain every realized payload and stop at the first nonzero useful bare admission or after 200 unique proposals for the F1 decision.
- **Action:** projector-distilled conditioning MAIN burn. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/main_burn`. **Fire trigger:** MAIN first obtains a nonzero bare pose-gated robust-improving checkpoint from a content-distinct proposal extension, seeds it into the consumer store, verifies that the >=200-proposal F1 gate has not fired, owns the training leg, observes the sole n600 scorer slot free, and passes storage/memory preflights.

## LIVE-HYPOTHESES

- A representation-level proposal source may reach overlap that scalar LR shrinking missed because the 15 proposals varied width and amplitude, not correction grammar.
- Receiver-lattice-aware curvature conditioning may preserve the continuous α^2.511 behavior through uint8.
- Distillation may work after a nonzero projected stage exists; this run never reached that prerequisite.

## DEAD-ENDS

- Amplitude-only shrinking of the inherited JS4 module is closed at INSTANCE scope: robust movement vanished before pose passed.
- Replaying the same five-shrink first-step ladder as a long burn is closed: every rung restored to zero.
- The 801 B null module is closed as an admission candidate: it moves no robust flips.