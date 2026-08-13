# DDM VH2 vehicle harvest drain — v lineage

Date: 2026-08-13  
Axis: `[scorer-free local corpus harvest; no scorer forward; no score claim]`

## Result

The declared v-first frame is dry: **121 of 121** top-level `.omx/research/ddm_v*`
roots were structurally inventoried and content-hashed, and the final message plus primary
memo/receipt for each of the 22 v arm groups was read. **Zero** roots remain in this frame.

No unconsumed v-lineage candidate, measured current-floor improvement, or banked current-receiver
archive was found. Every v finding is **FOLDED** into an existing consumer in the routed-artifact
ledger. The highest-value apparent reopenings were already consumed: V13/V14/V19c by G1 and the
current correction grammar, and V4d's dim-0 pose lattice by the current pose stack. Minting new arms
for them would duplicate live ownership.

The machine-readable drain is
`.omx/research/ddm_vh2_vehicle_harvest_drain_20260813_routed_artifacts.jsonl`, SHA-256
`de939c5c07c602c4c8d15f0be763f72b9606b0a4ad19088011e4e9d5fc517f76`. It contains 121
artifact rows and 22 group-route rows. Each artifact row records path, file or deterministic tree
hash, byte count, primary-source hash, final-message hash, read boundary, and group disposition.

## Declared frame and boundary

- Denominator: top-level entries under `.omx/research` whose basename matches `^ddm_v\d`.
- Selection mode: exhaustive, lexicographically ordered, no sampling and no early stop.
- Read denominator: 121/121 roots structurally inventoried and hashed; 19 `.done` final messages
  read in full; the three no-`.done` arms V4b/V4c/V4d used their primary memos as the final surface;
  all 22 primary memos/receipts read in full.
- Payload boundary: retained `*.receipt-bytes` and other binary run payloads were included in their
  directory tree hashes but were not decoded, replayed, rendered, or rescored. No payload was
  materialized by this arm, no payload was discarded, and no existing byte was moved or deleted.
- Directory hash: SHA-256 over sorted entries, each encoded as
  `kind NUL relative_path NUL payload_length NUL payload`; file roots use SHA-256 of file bytes.

## Current floor and comparison rule

The live authority read from `.omx/state/main_hot_state.md`, SHA-256
`622d7d0de8a61a4a7423d7162276cf881b226af4028ef351e129f05753263dde`, is:

- effective/local floor: CP135 `S=0.16195513827824176 @ 186,252 B`
  `[contest-CUDA T4, n600]`;
- own-vehicle frontier: LC2 `S=0.16959899569230852 @ 187,226 B`
  `[contest-CUDA T4, n600]`.

Historical v measurements are not promoted across receiver, vehicle, hardware, or evaluator axes.
This arm ran no scorer and moved neither pointer. It is a means-only harvest, not progress toward
sub-0.15.

## Ranked actionable findings

Ranks are by relevance to the current `0.16195513827824176` floor. “Actionable” here means the
disposition is explicit; all rows are folded because the corresponding action already has a consumer.

| rank | source signal | evidence and current-floor relevance | disposition | owner and consumer |
|---:|---|---|---|---|
| 1 | V19/V19b/V19c realized proposals and saturation | V19c reached historical advisory `d_seg=0.024786978828` at 137,827 B on the old receiver, but G1 already consumed the exact V13/V14/V19c receipts. V19c receipt `.omx/research/ddm_v19c_correction_saturation_20260723T063500Z/ddm_v19c_correction_saturation_receipt.json`, SHA `506fb1dfed849beb06358d3a30d624fa8cbdad3c6e0da6cf1bf1ec14960472ae`; G1 receipt `.omx/research/original_taskspace_inverse_witness_codec_20260725/g1_prior_signal_harvest_v1.json`, SHA `59b47749cca21732a4d166e6b2eb92b93c9c9dea21959cb82679cff3050ae55a`. | **FOLDED.** Do not replay the old bank or create a second v19 successor. | MAIN; `src/tac/witness_dsl/g1_prior_signal_harvest.py` SHA `8fa6efba748432f79a17f89500b5a05e703b9ebd97011edebea0f035c49d87e6`, `src/tac/witness_dsl/generative_taskspace_correction.py` SHA `fa0c230865aecd7085bc1f8292f7a84e852879580272d74f15382d2876a4397b`, and existing task 1029. |
| 2 | V13/V14 worldsheet and receiver projection | These establish useful primitive priors, not a current candidate. V13 DAG SHA `2da23cf3904dc659c9944946ccef86dbbb26cfcc6dde8ae9dd6a50da05c3adfb`; V14 DAG SHA `8b55e046194ff36f3a9eb0fb6d6e9db265bb3ce761a49545e7a32a41b564b8b3`. | **FOLDED.** Existing G1 import is the consumption proof. | MAIN; `src/tac/canonical_equations/ddm_describe_line_rate_distortion_bracket_20260722.py` SHA `44521a1f71ed9bd215c2c5558882b04a99c1a329460a9e05f83a37506462d4f2` and the G1 stores above. |
| 3 | V4d composed gate and dim-0 pose offset | The exact local advisory gate was `S=0.9639878179 @ 360,238 B`, far above CP135. Its lattice mechanism is already wired into current pose work. Source `.omx/research/ddm_v4d_adaptive_hybrid_20260731.md`, SHA `8b3b9cc2117a7b123faaef667b3593f967e10edc24ec0d0934797788d054c4bd`. | **FOLDED.** No current-row reopening. | MAIN pose-stack owner; `tools/cp1_compose_pose_stack.py` SHA `2d5bec58c368af029f1908530ef4b0c6a4576e06aa343fc8eafa1eb7872c3cd3` and `tools/pj2_pose_scale_joint_solve.py` SHA `0c9c327c5268dfc1830800e9fa33ce119f4d92fe06615ecf168d25d90b80b11c`. |
| 4 | V17 realized-validity ratio | The basis-conditioned local model did not survive exact uint8 realization. Source `.omx/research/ddm_v17_iterative_realized_trust_region_DAG_FEED_20260723.json`, SHA `44b55d8b44400645d0c438fcfcffdaafaa2594bc255e58cbd6206471cee0e7ae`. | **FOLDED** at formulation scope. | MAIN admission-discipline owner; `src/tac/canonical_equations/ddm_v17_validity_radius_law_20260723.py` SHA `1dd0e308e7cd6f2700921389a70eb00db5ea07ea65f451babe7f035a4f0fc5e3`. |
| 5 | V18b common-master pricing | Three common-master rounds admitted no negative columns; it closes that correction formulation, not richer representations. Receipt SHA `0d7e3535905cd48d42d7caeb6cfa8f56486a781bf16bbcb58cbe34afab014f55`. | **FOLDED.** V19 and later G1 work already supersede it. | MAIN; canonical sub-0.15 DAG SHA `fa5bac3824363c9a188701bc7f07e27157b3c2ee64744f248e5c09f99e9bc1e9`. |
| 6 | V6–V12 rate/distortion ladders | Rate reductions and bounded correction drains were real on their historical receiver, but Pose remained about 158–163 and the n600 correction family plateaued. Representative primary SHAs: V6 `7f247daf9db0fd09a52a9eebb96ffef036c767d5c41001d30fd35a9afa49d6d7`, V9 `dac48d90fa572e297d36c76cfee0d92bfd25835e439a3553ad24908e9dbadf97`, V12 `0484f65eec6f630772bd936a35810b6705680563533685f68737f26e16798312`. | **FOLDED.** No transfer of their numbers to CP135. | Task 603 in `.omx/state/canonical_task_status.jsonl`, SHA `d955b789afe17ccc144051ec1ab8c7f4eefcab3ca4fad0a07c2c3eeacd1c203d`, plus the describe-line equation. |
| 7 | V15/V16/V18 local solve and column-generation forms | These scoped negatives and apparatus lessons are already in the DAG; they do not contain a current-receiver archive. Primary SHAs: V15 `ca65660e23e59e18746ad360d041bf3f0316258edea5f65c4022beb00c55eea3`, V16 `fe291a2c9633b0ffcd9cb5ceae169ca7b61df6da81c161f5ee8544bc5ee19cb4`, V18 `d98f2a2545d462611e43b84273e22e06cc2cf974f2aa2acd6d480c943ce264a8`. | **FOLDED** at each source’s INSTANCE/FORMULATION scope. | MAIN; canonical sub-0.15 DAG. |
| 8 | V4/V5 structured members and route repair | Built historical receiver components; no current-floor score row or latent CP135 archive. Complete path hashes are in the routed-artifact ledger. | **FOLDED.** | Task 603 canonical status store. |

## Recall evidence

Before grading the v roots, this arm read the governing charter and common contract, PROGRAM,
AGENTS/CLAUDE, the operating handoff, live hot state, task-status ledger, canonical-equation query,
canonical research index, sub-0.15 DAG, prior VH2 apparatus, G1 receipt/code, and the CN5 exclusion
table. The CN5 table at `.omx/research/ddm_cn5_arc_consolidation_20260813.md`, SHA
`2bea2134870459d738fc6cffdb740b418c00bd3ed7039a6fee9fefa7225f6d6c`, was treated as an
exclusion set: the 2026-08-11 through 2026-08-13 arc was not reharvested.

The decisive joins were:

- task 603 already carries V9–V12 landing notes and completion state;
- `ddm_describe_line_rate_distortion_bracket_v1` consumes the V7–V14 describe line;
- `ddm_v17_realized_validity_ratio_uint8_v1` consumes V17’s validity failure;
- G1’s semantic argv explicitly names the exact V13, V14, and V19c receipts and says to reuse the
  original primitives as priors, never as score authority;
- task 1029, owner `ddm_js6_successor`, already owns the only live representation-changing TF1/v19
  join; CN5’s fire order requires a retained proposal bank before any scorer fire;
- current pose tools explicitly reconstruct `pose_dim0_offset`, so V4d did not expose an unused pose
  lever.

## Coverage and ledger custody

The current read-only `coverage()` snapshot is 1,325 top-level `ddm_*` artifacts, 10 harvested,
10 routed, and 1,315 unharvested. This is four artifacts newer than the charter’s 1,311-unharvested
snapshot; the current filesystem denominator is reported rather than back-fitting the seed.

| family | roots read this arm | roots remaining in declared family | canonical `coverage()` before safe ingestion |
|---|---:|---:|---:|
| v | 121 | 0 | 0 harvested / 121 unharvested |
| ms | 0 | 49 | 0 harvested / 49 unharvested |
| j | 0 | 45 | 0 harvested / 45 unharvested |

The dedicated routed-artifact ledger is complete for v, so the logical post-drain corpus count is
1,194 unharvested (`1,315 - 121`). The canonical `coverage()` result intentionally remains unchanged:
`.omx/state/probe_outcomes.jsonl` had pre-existing unrelated changes of **66 additions and 1 deletion**,
plus an earlier uncommitted VH2 append. Its current SHA is
`4608d8f06b0e8545105e3e26fcf9e859db54e3c8c9a09b1e6489bd8ec9a1bd12`; HEAD’s SHA is
`eed6afa8140366ea9de23dc698a88b48f76fe9f45afa125ae491d8aa96a8ae81`. Absorbing that whole
file would violate dirty-index custody. The new JSONL is the lossless ingestion payload; canonical
ingestion is queued only after that overlap is separately adjudicated.

## What was and was not measured

- Measured now: filesystem denominator, per-root bytes, per-root file/tree hashes, read-surface hashes,
  group membership, consumer joins, and canonical ledger dirtiness.
- Recalled from hash-pinned historical receipts: old-vehicle d_seg/d_pose/bytes and scoped mechanism
  verdicts. They remain labeled historical advisory unless their source says otherwise.
- Not measured now: RGB output, R survival, SegNet, PoseNet, complete S, contest CPU/CUDA replay,
  decoder timing, or any candidate archive. No scorer lane, Modal job, trainer, or evaluator ran.
- Conclusion scope: complete absence only within the 121-root top-level v frame and the named current
  consumer joins. This is not a claim that no future representation can use v-family ideas.

## Commit custody

The required serializer was first run against the shared checkout with both post-edit SHA guards and
the required tags. Git refused before staging because its object database is read-only in this managed
sandbox: `unable to create temporary file: Operation not permitted`; the shared index remained empty.

The same two-file patch was then committed through the serializer, with `--no-co-author`, in an
independently clean clone based at `51b3802191d18355587f56e20bd81fbde206ddb8`. Initial custody
commit: `1f88ec16c12d5eb4b486eaf282f2de3e5a5f59e0`. Its prerequisite-thin durable bundle is
`/Volumes/APDataStore/pact/ddm_vh2_20260813/1f88ec16c1.bundle`, 18,777 B, SHA-256
`61be4d1062fad1ecd9b3ed06c34175835809013fd8c8ec3fb92073a048df5fea`. `git bundle verify`
passed and identified exactly that commit as `HEAD` with base `51b3802191d18355587f56e20bd81fbde206ddb8`
as its sole prerequisite. Creation command: `git bundle create <path> HEAD ^51b3802191d18355587f56e20bd81fbde206ddb8`.
The shared working-tree copies remain uncommitted solely because of the Git-write sandbox boundary.

Custody correction: the first memo-only amendment landed as `f7ba96161d`, but the caller had mistyped
its expected SHA as `09be95de…`; the serializer therefore returned its post-commit mismatch code while
keeping the commit. Direct comparison found the committed memo and workspace byte-identical at SHA-256
`95226331e8a4049f40005da4f0d7396d6652ab40958beefdc5a54ece339dd362`. This note is the
reconciliation; no sibling content or unrelated file was present in that commit.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: VH2 ms-family successor. Consumer store: a hash-pinned ms routed-artifact JSONL beside this receipt. Fire trigger:** this v receipt is committed, the frame is declared as all 49 current top-level `ms` roots, and CN5 exclusions plus existing ms consumers are loaded before grading.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: VH2 j-family successor. Consumer store: a hash-pinned j routed-artifact JSONL beside this receipt. Fire trigger:** the ms frame is terminal or explicitly skipped for a recorded resource reason, then all 45 current top-level `j` roots are declared before reading.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN probe-outcomes ledger custodian. Consumer store: `.omx/state/probe_outcomes.jsonl`. Fire trigger:** the pre-existing 66-add/1-delete overlap and earlier 48 VH2 rows are separately custodied, after which the 121 artifact routes can be semantically ingested without staging unrelated changes.

## LIVE-HYPOTHESES

- A representation-level TF1/v19 join may escape the old V19c saturation law because task 1029 changes
  categorical support and coordinates rather than appending more atoms from the saturated vocabulary.
  It is plausible because G1 proved the primitives are receiver-compilable and RE1/CN5 found cheap
  categorical carriage; it remains untested because no retained current-receiver proposal bank exists.
- The ms or j families may contain a genuinely unconsumed current-floor row even though v does not.
  They are plausible next harvests because they are the next two largest named vehicle lineages
  (49 and 45 roots) and have not yet been reconciled by this ledger.

## DEAD-ENDS

- Replaying V19/V19b/V19c as a fresh scorer arm is closed: G1 already consumed the exact receipts, the
  current DSL carries the primitives, and the saturated vocabulary remains formulation-scoped.
- Treating V13/V14 worldsheet receipts as unread current candidates is closed: the canonical equation
  and G1 import prove downstream consumption, while their historical Pose axis is not CP135’s axis.
- Reopening V4d’s dim-0 offset as a new pose idea is closed: current pose composition and joint-solve
  code already reconstruct it, and the archived exact local row is far above the live floor.
- Appending directly to the dirty canonical probe ledger is closed for this arm: it would absorb
  unrelated user work. The dedicated 121-row ingestion payload preserves the result without that loss.
