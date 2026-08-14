# DDM JS1C — CUDA custody Stage-0

Date: 2026-08-14  
Status: **BLOCKED BEFORE PROVIDER ACCEPTANCE; SEALED AND QUEUED**  
Verdict scope: one exact T1R1 C1 candidate, archive
`12a5b181fef4e15ad8a752161c744347beca0b5a1224c5d3d542ab148f6ece80`, through
runtime tree `3bb8da9ffed161566458dd9bcd5ffc38bb6f7aa7c54b5f102df9f5e31c2e78d4` on
the retained contest-CUDA T4 batch-16 SegNet field instrument.

## Conclusion

The JS1C apparatus is landed and sealed, but this session did **not** produce a
fresh T4 field or a fresh Stage-0 rho. Three read-only Modal connectivity probes
and the exact sealed dispatch all stopped with `Could not connect to the Modal
server.` The failure happened before the local entrypoint: the provider issued
no call ID, no lane claim was created, the remote run namespace was not
materialized, and no paid run was accepted. Pre- and post-attempt dual-ledger
reconciliation both reported zero live call IDs and zero active Modal claims.

The retained earlier JS1B T4 fields remain strong recall evidence, not a new
JS1C measurement. On those exact fields the base, T1R1 candidate, and C1 target
were respectively `34,970`, `55,807`, and `27,330` flips, giving
`rho = (34,970 - 55,807) / (34,970 - 27,330) = -2.7273560209424086`. That is
far below the charter gate `0.827795` and predicts that the fresh repeat will
fold V0–V5, but JS1C does not promote that prediction into a completed result.

## Landed apparatus

Commit `0585dd55bd7e53d98542f9b8c96220a6bd1c6343` adds:

- `experiments/ddm_js1c_cuda_custody_stage0.py`, SHA-256
  `29c733a0287eea6fa997ed9d60a3c20784b3611f4e815880963473a5baf605ff`;
- `experiments/tests/test_ddm_js1c_cuda_custody_stage0.py`, SHA-256
  `0672cf752846d54a5d384e1c97d3323bcc62e8f95005f8dbc69db12eb73e0bdc`.

The dispatcher reuses the exact proven RE1 function
`b00f3ffc1eb5e8f4680eb8f301bd5c83921728f0c90d68e90c23799157983ec9`, is
modeled on seal
`486cb7e92083bee0c1a7cc078654518da689a672129a6390a33e960cddf6ca63`, and pins
the SegNet-only legacy worker
`03dc9e81a21409f5881cff642d5dc334a8f04deae5b008f31cd2719bba4a14fb`.
The request retains `local_pose_delta: 0.0` only as an explicit placeholder
with `pose_unmeasured: true`; no Pose or complete-score claim is made.

The remote leg is designed as fresh-run, same-run-ID resumable, and
stage-checkpointed; when accepted, it retains the exact receiver raw bytes,
all Seg inputs, logits, batch receipts, and candidate argmax field. The harvest
consumer refuses source or axis drift,
copies GT, CP135 base, C1 target, and candidate fields into the new store,
writes all 600 per-pair decompositions, computes all 20 directed cells and 10
undirected interfaces including the Road hub, and only then evaluates rho.
V0–V5 is fireable only at `rho >= 0.827795`. The task-1043 trigger and failed-rho
reroutes to tasks 982 and 978 are emitted only after a fresh retained result.

Focused and predecessor regression coverage: **19 passed**. The payload
retention detector, Python compilation, Ruff, and whitespace checks passed.
Both Python files received two review passes before serialization.

## Retained custody

Consumer store:
`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814`

- `JS1C_T4_REQUEST.json`: `12,237 B`, SHA-256
  `c393da9698f4440b686638d3d9dd7632abc93ac301ede881dd689f89e49aeaef`;
- `SEALED_FIRE_ORDER.json`: `2,797 B`, SHA-256
  `7487a931fc74b66f1d8339e8ee401aab703b7198a894f37ac61139b3cc158516`;
- `DISPATCH_ATTEMPT.json`: `1,662 B`, SHA-256
  `aa1c06752010f94fd69a411ef3340262ef0b372463de0bf62a9f26d863ec45d1`;
- all three upload payloads are retained under `fire_inputs/`, including the
  exact `187,046 B` T1R1 archive;
- local storage preflight passed with `50,233,614,336 B` free against
  `1,008,999,413 B` required, explicitly accounting for all four harvested
  fields plus reserve;
- the intended remote volume is `comma-ddm-js1b-argmax-retained`, fresh path
  `ddm_js1c_cuda_custody_stage0_20260814_r1/`.

No generated payload was discarded. The fresh remote payloads do not yet
exist because the provider never accepted the call.

## Measured and not measured

**Measured in this unit:** source/archive/runtime hashes; committed apparatus;
sealed request and fire order; local disk capacity; zero-live dual-ledger state
before and after the attempt; provider connectivity failure before call
acceptance.

**Not measured in this unit:** a fresh T4 candidate argmax field, fresh rho,
fresh Road-hub decomposition, V0–V5, PoseNet, complete score, contest-CPU, or a
new exact archive row. The charter's main empirical gate therefore remains
open, not completed.

No new empirical or formal law was introduced, so no equation registry entry
is owed. The pending consumer applies only the chartered fraction
`rho = (base_flips - candidate_flips) / (base_flips - target_flips)` to retained
same-axis fields.

## RECALL EVIDENCE

All-store corpus queries covered research (8,555 documents), equations (890),
memory (2,118), DAG (916), council (297), tasks (547), and docs (96). Exact
queries included:

- `js1 stage0 per edge t1r1 c1 cuda rho 0.827795`;
- `task 1043 implicit joint distortion conditioning trained receiver coupled multi-token 982 978`;
- `Road hub directed per-edge decomposition m91 cp135 t4 argmax`.

Direct custody reads covered the charter/common contract, live hot state,
canonical pointer, JS1/JS1B/JS8 memos, prior local and CUDA result receipts,
the re1 dispatcher/worker/seal sources, the Stage-0 consumer, relevant tests,
lane claims, call-ID ledger reconciliation, and the exact T1R1 archive/runtime.
The decisive recalled fact was the already-retained same-object T4 field SHA
`7c3750d9e44b44fb6cbd8dc9c9907714532d13364bb705c8183f5cff6b9a184a`;
it makes a fresh byte-identical repeat plausible but does not substitute for
the chartered new run.

## Frontier truth

The effective exact frontier remains CP135
`S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`, archive SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
The own-vehicle frontier remains LC2
`S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`. JS1C produced no
archive or score and moved neither pointer. The exact pointer is still above
`T_3 = 0.15`, so this unit did not achieve the goal.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814`; fire trigger: Modal API connectivity is available and `tools/claim_lane_dispatch.py reconcile` still reports zero live Modal calls and zero active Modal claims; action: execute `SEALED_FIRE_ORDER.json.exact_command_argv` once.
- **QUEUED-AT-HARVEST** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814`; fire trigger: the sealed call returns `measurement_complete: true`; action: run the sealed recovery, download the retained candidate field with `HARVEST_REQUEST.json.candidate_field_download_argv`, then run `consume` without another scorer pass.

## LIVE-HYPOTHESES

- **The fresh T4 candidate field will be byte-identical to the prior JS1B field.** This is plausible because the exact archive/runtime, seed, batch shape, frozen scorer, retained controls, and worker source are pinned; the earlier field already exists under custody.
- **T1R1 will fail the matched-axis rho gate by a wide margin.** This is plausible because the prior same-object T4 field regressed from 34,970 to 55,807 flips and yielded rho −2.727, far from 0.827795; only the chartered fresh custody repeat remains untested.
- **Implicit joint distortion conditioning, a trained receiver, or coupled multi-token realization can still escape the frozen-receiver T1R1 failure.** This is plausible because those routes change the realization map or interaction support; the retained negative covers one fixed T1R1 instance, not those families.

## DEAD-ENDS

- **Local Mac rho as CUDA authority:** closed because the Mac renderer moves CP135 by 15,431 flips relative to the promoted T4 row, so it is not the same instrument.
- **Treating the T4 C1 value 27,330 as an axis mismatch against the local 17,926 value:** closed because 17,926 is a local-instrument number; the retained T4 base, target, candidate, and GT fields are internally matched.
- **Silently reusing the earlier JS1B field as the chartered fresh JS1C result:** closed because that would fake a new dispatch, determinism receipt, and completed trigger.
- **Retrying V0–V5 from the recalled T1R1 result:** closed unless a fresh retained JS1C rho reaches 0.827795; the recalled same-axis rho is strongly negative.
