Blocked before provider acceptance. Three connectivity probes and the sealed Modal dispatch failed with `Could not connect to the Modal server.` No call ID, lane claim, remote payload, or charge was created; reconciliation remains clean at zero live Modal calls.

Completed locally:

- Landed dispatcher/consumer and tests in `0585dd55bd`.
- Landed the [evidence handoff](/Users/adpena/Projects/pact/.omx/research/ddm_js1c_cuda_custody_stage0_20260814.md) in `7a203f8e95`.
- Retained the [sealed fire order](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814/SEALED_FIRE_ORDER.json) and [dispatch-attempt receipt](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814/DISPATCH_ATTEMPT.json).
- Verification: 19 tests passed, payload-retention check passed, and both Python files received two review passes.

No fresh T4 field or rho was measured. The prior same-object T4 fields imply `rho = -2.727356`, but they were not misrepresented as the required fresh JS1C result. Both exact and own-vehicle frontiers remain unmoved.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814`; fire trigger: Modal connectivity returns and reconciliation still reports zero live calls/claims; action: execute `SEALED_FIRE_ORDER.json.exact_command_argv` once.
- **QUEUED-AT-HARVEST** — owner: MAIN; consumer store: the same JS1C store; fire trigger: the call returns `measurement_complete: true`; action: recover, download the retained candidate field, and run `consume` without another scorer pass.

## LIVE-HYPOTHESES

- The fresh field will reproduce the prior field byte-for-byte because archive, runtime, worker, seed, scorer, and batch shape are pinned.
- T1R1 will fail the `0.827795` rho gate because the earlier same-object T4 measurement yielded `−2.727356`.
- Trained-receiver, coupled multi-token, and implicit joint-conditioning routes remain plausible because they change the realization map; this fixed T1R1 instance does not test them.

## DEAD-ENDS

- Local Mac rho as CUDA authority: the instruments differ by 15,431 CP135 flips.
- Treating T4 C1’s 27,330 flips as mismatched against local C1’s 17,926: they are different instruments.
- Reusing the earlier JS1B field as a fresh JS1C result: that would fabricate dispatch and determinism custody.
- Firing V0–V5 without a fresh rho at or above `0.827795`: forbidden by the charter.

