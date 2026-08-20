SA1 is `READY_TO_FIRE`; the exact pointer did not move.

- Built and retained three ≤2,048-byte forms.
- Selected an 814 B conditioner in a deterministic 187,178 B archive, +926 B over CP135.
- Local ordering only: −37 projected flips, zero robust repairs, pose Δ +1.16e−6, joint ΔS +0.001284.
- Preserved CP135’s `p` member byte-identically.
- Prepared the resumable candidate-only T4 gate without dispatching Modal.
- T4 admission requires ≤33,417 flips versus CP135’s 34,970.
- Landed commit `36b8b9ab14`.
- Verification: 12 focused tests passed, Ruff passed, two review-tracker passes completed.

Artifacts: [receipt](/Users/adpena/Projects/pact/.omx/research/ddm_sa1_shipping_axis_seg_actuator_20260813.md), [result](/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/FINAL_RESULT.json), [fire ticket](/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/READY_TO_FIRE.json).

A broader JS1B regression run had two pre-existing failures caused by its later C1 custody-SHA addendum drifting from older tests; SA1 did not modify those files.

Current custodial frontier remains CP135 `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. Own-vehicle frontier remains LC2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/t4_sign_gate_v3/`; fire trigger: reconcile active claims and Modal single-flight, then execute the command in `READY_TO_FIRE.json`.

## LIVE-HYPOTHESES

- The weak local candidate may improve materially on T4 because prior local/T4 Seg signs inverted and this module was trained against exact T4 target fields.
- If the hybrid gate fails, true CUDA-in-loop training remains plausible because it removes the measured axis mismatch directly.

## DEAD-ENDS

- Stage-8 EMA as a Seg candidate: closed at INSTANCE scope because it worsened local Seg by 57 projected flips despite pose-driven negative joint pricing.
- Live step-1/2/4 cells: closed at INSTANCE scope because pose harm exceeded the gate.
- Local-only promotion: closed; the local forward is ordering-only and cannot establish a shipping-axis verdict.
- Concurrent resumes into one output tree: closed operationally with a fail-closed exclusive run lock.