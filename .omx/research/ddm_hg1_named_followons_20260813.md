# DDM HG1 named follow-ons — structural fixes and PZ4R component routing

## Result

All three named `$0` follow-ons were executed without a scorer, Modal, training, or payload
materialization.

| Item | Disposition | Result |
|---|---|---|
| PO1 recover idempotence | **FIRED** | Both terminal ledgers now use check-before-append keyed by `run_id` and terminal status while a per-output recovery lock prevents concurrent re-entry races. A double recovery appends one outcome and one terminal claim. |
| Queue inline-charter guard | **FIRED** | `add` and `spawn` validate the charter file boundary before lint/stat use. A 5,000-character inline value returns a typed nonzero refusal containing “charters must be files” and no traceback or raw `OSError`. |
| PZ4R semantic salvage | **FIRED, VERIFIED WITH NARROWER SCOPE** | The LC2-derived semantic/HPAC/temporal/token sections are byte-identical and all 600 Seg-scored frame-1 blocks are byte-identical to LC2. The reusable component was appended to the re1 store without changing `ddm_re1_20260813`. Cross-lineage CP135 compatibility is not established. |

The exact pointer and own-vehicle frontier did not move. This was a scorer-free hygiene and routing
unit, not goal completion.

## Item 1 — PO1 recovery idempotence

The charter named `experiments/ddm_po1_t4_error_feedback_pose_compensation.py`, but bounded source
inspection found that the terminal writes actually occur in
`experiments/ddm_po1_modal_t4_pose_feedback.py::recover`. The fix therefore lands at the real writer;
the existing compensation test module holds the regression.

The structural cure has three parts:

1. Query the canonical call-ID ledger before append and match the exact nested `run_id` plus
   `harvested`/`failed` status.
2. Parse the terminal claim ledger before append and match the exact run identity plus the exact
   `completed_cuda_pose_feedback_recovered`/`failed_cuda_pose_feedback_recovered` status.
3. Hold `terminal_recovery.lock` across both decisions so two pollers sharing the output directory
   cannot race through the checks.

**TESTED `[local CPU unit test]`:** calling `recover()` twice against one fake completed Modal return
wrote the returned artifact twice safely, but appended exactly **1/1** call-ID terminal row and
**1/1** terminal claim row. This closes future duplication; historical duplicate rows remain
append-only provenance and were not rewritten.

## Item 2 — queue file-path contract

`tools/codex_arm_queue.py::charter_file_path` now centralizes the path contract for both queue
admission and spawn. It rejects empty, multiline, NUL-containing, unstatable, and non-file inputs
before any charter lint or keeper write. `ENAMETOOLONG` and related `Path.is_file()` failures become:

`REFUSED <name>: charters must be files; --prompt expects a file path, not inline text`

**TESTED `[local CPU subprocess]`:** a 5,015-character inline value returned nonzero, named the
file-path contract, emitted neither `Traceback` nor `OSError`, and appended no queue row.

## Item 3 — PZ4R semantic salvage

### Verdict

The salvage verifies for the **LC2-derived transport component**, but the charter's proposed
CP135-relative attribution needed correction.

**MEASURED `[scorer-free archive parse, n600]`:** LC2 and PZ4R direct-v6 contain exactly equal
semantic raw/wire, HPAC raw/wire, temporal packed, and token wire sections. **MEASURED
`[retained-raw byte comparison, n600]`:** every byte of all 600 Seg-scored frame-1 blocks is equal
between LC2 retained `0.raw` and PZ4R retained `0.raw`: **1,831,204,800 / 1,831,204,800 B equal**,
with **0 / 600 changed pairs**. This is direct output attribution, not inference from a small net
flip delta.

PZ4R is not a CP135-format child. Parsing the CP135 member with the PZ4R/LC2 receiver grammar was
**REFUSED `[scorer-free parser negative]`** with `ReceiverFormatError: combined payload has no
complete token section`. No section identity or measured score transfers from LC2 into CP135/F26
or re1 without an explicit adapter.

### Exact reusable sections

All values below are **MEASURED `[scorer-free real archive parse]`** and equal between the LC2 and
PZ4R archives.

| Section | Representation | Bytes | SHA-256 |
|---|---|---:|---|
| semantic | raw | 40,252 | `9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99` |
| semantic | wire | 34,547 | `5ccee3cbe0e56924bca876e3a1d5d9910e8ebaf7a1dc03bf71d5d3f5881c2843` |
| HPAC | raw | 20,179 | `b07fff73fac41c5fec2d8acbfd7c43c518852696f18d95cf7465fc6ed7510b58` |
| HPAC | wire | 14,977 | `ee576e6001d2badd36811df582911448373239c93a0b38ec7de398bbe41e8a6b` |
| temporal | raw packed | 39 | `f920f7be8108b83831971a8d07c9ef522eadb18abed095cf395bf3a6f871e796` |
| tokens | wire | 114,528 | `85d6c199ffb93ddab0fe1631448882a255e9fea1f6858bab5a04cea2310a7331` |

The exact fixed charged portion is **164,168 B**: 100 B ZIP overhead + 4 B outer header + 12 B
split header + 34,547 B semantic wire + 14,977 B HPAC wire + 114,528 B token wire. Headers are
transport overhead, not semantic content.

Archive custody:

- LC2: **187,226 B**, SHA-256
  `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`.
- PZ4R direct-v6: **183,137 B**, SHA-256
  `c408adf9101bb19a363039a5e0f7185aabce8f31edb6787e2deaf6d0fe6738f4`.

### Cancellation check

The net CP135-to-PZ4R `+18` flips is not itself attribution proof. **MEASURED
`[macOS-CPU advisory retained argmax arrays, n600]`** over 117,964,800 labels:

- 116 labels changed across 105 pairs;
- 49 CP135 errors were cured;
- 67 CP135-correct labels were broken;
- 0 errors changed to a different wrong class;
- 50,345 wrong labels were unchanged;
- net delta = 67 - 49 = **+18**.

Thus the `+18` contains cancellation. The salvage remains verified because the actual parent LC2's
semantic sections and Seg-scored output bytes are independently identical, not because the
cross-lineage CP135 net is small.

The PZ4R direct-v6 **INSTANCE remains FOLDED**: the already-retained matched advisory row is
50,412 versus 50,394 Seg flips but `d_pose` moves from `0.00014746535453014076` to
`0.6310142278671265`; complete matched advisory S moves from `0.20513830286735124` to
`2.676677850377788`. Those are **REPORTED AND RECOMPUTED `[macOS-CPU advisory, frozen CPU-torch
SegNet+PoseNet, n600] NON-PROMOTABLE`**, not new scorer work in HG1.

### re1 routing receipt

A typed append-only row was written to:

`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/candidate_components.jsonl`

Receipt: **1 JSONL row**, **3,133 B**, SHA-256
`c2fea6aeed99ac4114beb78e6e3f79ae4f667b74adb5666c277546d25d5259be`, schema
`ddm_re1_candidate_component.v1`, component
`ddm_hg1_pz4r_lc2_semantic_transport_20260813`. It records the LC2-only scope, every section pin,
the output-byte proof, the cancellation audit, and `re1_live_work_mutated=false`. The live
`probability_object_race/ddm_re1_20260813/` directory was neither entered nor modified.

## RECALL EVIDENCE

The recall pass searched the full required corpus rather than only the charter seeds:

- `.omx/research/` and arm final messages by content with
  `po1 recover idempotence|duplicate terminal|PZ4R|direct_v6|PGQ1|semantic|HPAC|LC2|re1`;
- `.omx/state/codex_arm_queue.next_if_resumed.jsonl`, the modal call-ID ledger, active lane claims,
  task/P0 ledgers, and `.omx/state/main_hot_state.md` with the same terms and exact run IDs;
- `.omx/research/CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` FEED surfaces with
  `PZ4R|PGQ1|realization|receiver|semantic`;
- v7.5/v8 SPEC and design surfaces with `receiver|semantic|HPAC|pose gauge`;
- the complete equation registry via
  `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  `PZ4|receiver|pose semantic|realization`.

Beyond the seeds, recall found:

1. HV1's exact follow-on correctly identifies the real dispatcher source and requires the poller
   to stop before this fix; the terminal store shows the poller stopped.
2. The canonical `realization_breakeven_bytes_v1` entry already corrects PZ4's score recovery to
   **UNMEASURED** and preserves only its **4,089 B** LC2-relative receiver-closed rate yield. This
   prevented importing a zero score-recovery placeholder.
3. The newer `receiver_pose_semantic_preservation_ratio_v1` records the 4,279.0676x PZ4R/base pose
   ratio and explicitly excludes semantic preservation inferred only from archive/decode hashes.
   This changed the salvage proof to exact section identity plus exact Seg-scored output identity.
4. RE1's current store is CP135/F26-shaped and has a pending separate whole-candidate scorer order.
   This prevented editing the live run or pretending that LC2 sections are immediately composable.

## Verification and landing boundary

- Focused suites: **46 passed** in 33.87 s.
- Ruff: all checks passed on the dispatcher, PO1 regression, and queue regression; queue source
  passed `E9,F,I`. Its broader file-level Ruff debt is pre-existing on unchanged lines and was not
  expanded.
- CPython compile: all four changed Python files passed.
- Git diff whitespace check: passed.
- Two explicit review-tracker passes (`hg1-correctness`, `hg1-adversarial`) were recorded for each
  changed Python file with no remaining HG1 finding.
- No existing staged entry was touched during implementation or verification.

The serializer disposition is finalized after this memo's post-edit hash is known.

## Frontier

Effective frontier remains CP135 at **S = 0.16195513827824176 @ 186,252 B
`[contest-CUDA T4, n600]`**. Own-vehicle frontier remains LC2 at
**S = 0.16959899569230852 @ 187,226 B `[contest-CUDA T4, n600]`**.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN/RE1 probability-object owner. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/candidate_components.jsonl`. Fire trigger: the existing `ddm_re1_20260813` run is terminal and a candidate explicitly supplies an LC2-compatible receiver or a measured CP135/F26 adapter.** Consume `ddm_hg1_pz4r_lc2_semantic_transport_20260813`; prove receiver parse-back and output attribution again before any scorer fire.

## LIVE-HYPOTHESES

- The pinned LC2 semantic/HPAC/token package may lower the realization cost of a future joint
  representative because it already reproduces every Seg-scored LC2 frame-1 byte while leaving the
  carrier replaceable. It is plausible only inside LC2 grammar until an adapter is proved.
- A jointly learned pose carrier may compose with the salvaged component because direct-v6's Seg
  output was exactly inherited from LC2 and its catastrophic loss is isolated to the replacement
  carrier. This requires a new counted archive and retained full-population Pose vectors; it is not
  a retry of the PZ4R direct-v6 carrier.

## DEAD-ENDS

- Treating CP135 and PZ4R as section-compatible is closed: their receiver grammars differ and the
  PZ4R parser rejects the CP135 member before a complete token section.
- Using the net `+18` CP135-relative flips as semantic attribution is closed: 49 cures and 67 breaks
  cancel inside that number.
- Retrying exact PZ4R direct-v6 is closed at INSTANCE scope: `d_pose=0.6310142278671265` overwhelms
  the 3,115 B CP135-relative and 4,089 B LC2-relative rate savings.
- Appending another PO1 terminal row on every poll is closed: recovery now detects the exact
  run/status pair in both terminal ledgers under a recovery lock.
- Passing inline charter text to the queue's file argument is closed: it is now a typed refusal at
  both admission and spawn boundaries.
