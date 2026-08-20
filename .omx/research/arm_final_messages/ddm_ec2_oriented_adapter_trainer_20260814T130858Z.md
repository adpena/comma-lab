Implemented, sealed, tested, and committed the EC2 trainer as `93d4dab7ef8c4c2d6e731af2200443249b694489`.

The trainer now:

- Runs the counted adapter through exact int8/float16 QAT before CP135’s four TokenBlocks.
- Uses the real camera uint8 roundtrip and frozen SegNet.
- Reloads serialized adapter bytes for the full-n600 endpoint.
- Retains every training/endpoint payload and distinct live/EMA checkpoints.
- Restricts dispatch 1 to `oriented`; controls are hard-gated on oriented break-even.

Key artifacts:

- [Handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_ec2_oriented_adapter_trainer_20260814.md)
- [T4 worker](/Users/adpena/Projects/pact/experiments/ddm_ec2_oriented_adapter_trainer_worker.py)
- [Modal dispatcher](/Users/adpena/Projects/pact/experiments/ddm_ec2_modal_oriented_adapter_trainer.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_ec2_oriented_adapter_trainer.py)
- [Sealed fire order](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/main_cuda/SEALED_FIRE_ORDER.json), SHA-256 `f5f7782591639ce55d6b70af99c0e0cac37c42f1bdb6aef6c275766081f6834c`

Verification: 12 tests passed, Ruff clean, payload-retention scan 0 findings across both implementation files, dispatch-shell guard passed, two review passes recorded, and post-commit hashes match the seal.

No Modal job, T4 scorer, candidate package, or exact evaluation was run. `modal_dispatched=false`; no score or pointer moved. Own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole Modal scorer-lane router; consumer store: `main_cuda/`; fire trigger: no duplicate EC2 oriented lane, Modal available, and all sealed hashes verified; execute `first_dispatch.exact_argv`.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN EC1 packager and RE1T/JS1B consumer; consumer store: `main_cuda/`; fire trigger: oriented run recovered with valid archive repeat and full-n600 endpoint; execute sealed harvest/package/measurement chain.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole Modal scorer-lane router; consumer store: `main_cuda/dispatch_controls/`; fire trigger: oriented `SELECTED_RESULT.json` proves `clears_oriented_break_even=true`; then fire the sealed controls dispatch.

## LIVE-HYPOTHESES

- Oriented conditioning remains plausible because its 8,380-error targeting mass is well above the 1,340-flip price bar.
- Exact QAT and parsed-byte endpoint measurement may preserve enough selectivity to clear break-even.
- Learned coupled amplitudes may outperform the already weak fixed semantic edits.
- If oriented wins, equal-parameter controls can distinguish orientation information from generic capacity.

## DEAD-ENDS

- Local CPU or MPS authority: closed by the measured local/T4 mismatch.
- Prefix endpoint verdicts: closed; the endpoint is full n600.
- Float-EMA endpoint evidence: closed; the endpoint consumes serialized parse-back bytes.
- Post-render overlays: closed for this route by prior T4 evidence.
- Seeded nonzero modules as trained candidates: closed; they remain structural controls only.
- Firing controls before an oriented win: structurally blocked by the dispatcher.