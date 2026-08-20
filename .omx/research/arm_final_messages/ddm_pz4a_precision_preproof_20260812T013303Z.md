Verdict: `REFUTED` at INSTANCE/FORMULATION scope.

verdict_scope: INSTANCE/FORMULATION — machine-readable restatement of the prose
verdict on the line above. Appended by ddm_oc2 at consolidation time so the
negative-verdict gate can parse the scope the arm already declared; the recorded
finding, its numbers, and its original wording are unchanged (append-only per
the HISTORICAL_PROVENANCE discipline).

- Exact inner CPR1 baseline: 9,945 B; best rung grew by 12 B.
- Held-basis Brotli-q9 baseline: 23,054 B; best gross saving: 500 B.
- Counted depth-map wire: 2,732 B.
- Best net saving: **−2,232 B**, versus the required **+2,000 B**.
- No compensation build, scorer run, or evaluator row is owed.

The raw int16 projection suggested 8,164 B, demonstrating exactly why raw-width arithmetic was non-authoritative.

Artifacts:

- [Durable handoff](/Users/adpena/Projects/pact/.omx/research/ddm_pz4a_precision_preproof_20260811.md)
- [Pre-proof runner](/Users/adpena/Projects/pact/experiments/ddm_pz4a_precision_preproof.py)
- [Focused tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_pz4a_precision_preproof.py)
- [Final retained result](/Volumes/APDataStore/pact/ddm_pz4a/retained/preproof_v2/FINAL_RESULT.json)
- [Retention audit](/Volumes/APDataStore/pact/ddm_pz4a/retained/preproof_v2/RETENTION_AUDIT.json)

Verification: 6 tests passed, zero payload-retention findings, two review passes, 108 artifact records independently rehashed and five rungs reparsed. Commits: `d0c222fc0e`, `5105a60639`.

No scorer, GPU, Modal, evaluator, upstream mutation, or frontier movement occurred.

## LIVE-HYPOTHESES

- None remain within the chartered PZ4A formulation. Even zero allocation metadata would leave its 500-B gross ceiling below the 2,000-B gate.

## DEAD-ENDS

- Absolute-coefficient sensitivity coarsening on this pass-03 CPR1 state is closed: best counted net is −2,232 B.
- Raw int16-width savings are closed as a gate proxy.
- The superseded coarse-to-fine v1 allocator is closed because it overshot the tolerance band; its payloads remain retained.
- Standing frontier unchanged: own-vehicle LC2 **S = 0.16959899569230852 @ 187,226 B `[contest-CUDA T4, n600]`**; effective floor CP135 **S = 0.16195513827824176 @ 186,252 B `[contest-CUDA T4, n600]` (ours)**.