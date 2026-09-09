# VR5 — 17 certified raw candidates, 57.985 GiB queued for MAIN

Owner: ddm_vr5. Tokens: `[no-triality] [p0-ledger-ok]`.
Axis: `[filesystem custody / scorer-free]`; `score_claim=false`.
**Disposition: PLAN COMPLETE; APPLY QUEUED-WITH-A-FIRE-ORDER for MAIN. Reclaimed: 0 B.**

## Measured result and prediction

The 20-row input contains 73,248,192,000 B (68.217695 GiB). All 20 MAIN rehash values
match the retained decode receipts. All five historical raw SHA values match; the other
15 rows now have a current raw SHA from MAIN's receipt. Current archive/runtime manifests,
exact provenance argv, inflate script, decode and eval receipt hashes match VR4's full
retained certificates on all 20 rows. No archive drift or raw SHA mismatch was found.

| Drive | Input rows | Admitted rows | Admitted B | Admitted GiB | Blocked rows |
|---|---:|---:|---:|---:|---:|
| APDataStore | 15 | 12 | 43,948,915,200 | 40.930617 | 3 |
| VertigoDataTier | 5 | 5 | 18,312,048,000 | 17.054424 | 0 |
| Total | 20 | 17 | 62,260,963,200 | 57.985041 | 3 |

The prior prediction that all 20 rows would admit is **falsified at this 20-path INSTANCE
scope** by consumer references, not by missing bytes or broken reproducers. The existing
reference gate stays conservative: a reference blocks even if its consumer may be old.

| Blocked owner | Evidence preserved in its ledger row | Reason |
|---|---|---|
| WD2 | WD3 `IMPLEMENTATION_SPEC.md:36` | Names this raw as an input. |
| HV1 | 24 hits in live repository tools, recovered source and a test | Names the raw or its inflated directory; no consumer retirement proof. |
| JG4 | JG5 pose runner and recovered JG4 source | Names this raw as an input. |

The exact paths, aliases and every hit are in the plan ledger. No reference was waived
because a file looked old. Ten exact VR4 inventory/certificate/reference observation files
are excluded by explicit SHA-pinned data in `admission_policy.json`; any edit to one fails
closed at apply. The existing VR3 observation exclusions remain. The current plan and its
apply journal are self-observation exclusions. No broad family exemption was added.

## Executable handoff

Run from `/Users/adpena/Projects/pact`, **by MAIN outside the sandbox**:

```sh
.venv/bin/python experiments/ddm_vr3_certified_raw_reclaim.py apply --ledger .omx/research/ddm_vr5_reclaim_plan_20260909.jsonl --expected-ledger-sha256 44164e45610e410b4d59abaadb0bc6b03b5611b267fa558fb274aedcdd5635d6 --journal .omx/research/ddm_vr5_apply_journal_20260909.jsonl --target-bytes 62260963200
```

Plan SHA-256: `44164e45610e410b4d59abaadb0bc6b03b5611b267fa558fb274aedcdd5635d6`.
MAIN rehash receipt SHA-256: `035ead599d3135e0f1187f93c809e295eaeabe72864f3dea794822a5ae4c2a6b`.
VR4 source ledger SHA-256: `2fbe57b7a7bcbe410b01abbf7b7214812553c37e7e4288423914f71decc73495`.

The command targets the entire admitted set. The historical VR3 minimum of 60 GiB still
applies to legacy plans; generalized plans instead require `--target-bytes` to equal the
full admitted logical-byte sum. This prevents the three blocked rows from being smuggled
back in to meet an obsolete target. Two-drive free-space samples and per-drive values are
retained; apply requires both attributed logical deletion and the measured combined free-space
gain to meet its target. Concurrent writes can reduce net free space, so a nonzero apply
result is not permission to delete blocked rows. Inspect the journal and final ledger.

Apply keeps the prior executor sequence: revalidate the retained certificate, raw identity
and full SHA; repeat repository/protected-tree reference clearance; check owning-arm processes
and open handles; fsync PRE_DELETE; unlink only the named regular `0.raw`; fsync its parent;
append/fsync DELETED; atomically replace/fsync the ledger. Process visibility must work at
apply entry and again per row: denied/malformed `ps`, invisible init/self, unavailable/erroring
`pgrep`, an active owner, or unavailable/warning-bearing `lsof` refuses. The raw stat identity
is checked again after the full hash and liveness checks. No apply or unlink was run on SSDs.

The plan reuses VR4's rows, deriving the family from each owner/path join and a pinned closure
memo. It adds no new admission-family constants. AP1/JF2 legacy selection remains functional.
Protected arm names, seals, symlinks and submission trees refuse. Reference scans retain the
original repository scopes, explicitly include the possibly gitignored live pointer/hot-state/
lane/dispatch files, and scan text metadata in protected trees on both drives with ignores
disabled. This is a bounded reference audit of these scopes, not proof of no outside consumers.

MAIN's charter reports `pgrep -fl` per owner and `lsof` on all 20 paths clear at approximately
**2026-09-09 23:50Z**. That is a MAIN-reported receipt, not a VR5 observation. Its stated time
was later than the tool clock when consumed; no freshness inference is made from it.
The live apply checks are mandatory regardless. MAIN's rehash file has no per-row timestamp
or stat identity; the plan preserves that limitation, compares current stat identity to VR4's
inventory, and rehashes again at apply. The full 20-row artifact was awaited before admission.

## RECALL EVIDENCE

Read the charter/common contract, PROGRAM, relevant CLAUDE/AGENTS no-fake/storage/review rules,
the operating handoff, live hot state and canonical pointer. The memory registry query
`reclaim|raw.*certif|certif.*raw|storage|cleanup` found no relevant entry; no memory-derived
fact is used here.

Independent corpus searches cover research memos/receipts by content, the canonical research
index and main DAG, design docs and canonical task rows. Queries and bounded results are in
`recall_searches.json`: `certify.or.block|certified.reclaim|cold.store|storage.*reclaim` and
`certif.{0,30}reclaim|reclaim.{0,30}certif|sandbox.{0,40}lsof`. The canonical equations CLI returned
480 rows; the custody/reclaim query found no applicable equation. Its count/query/output digest
are recorded in `equations_recall.json`.

Beyond the charter seeds: DK2 documented the danger of whole-tree reclamation and contention
from simultaneous movers; SR3 retained live-consumer protections. This kept the change at
single-raw granularity and ruled out consumer waivers. NI1's old memo was queued, so its
2026-08-30 NI1R measured closure was recovered before accepting the closure chain. JF2's
terminal distortion addendum closes its stale scorer/WJ1 follow-ons. RR8's measured successor
and identity receipts establish completion of the older JG5/CD1 advisory lineage. These are
pinned in the policy rather than copying a stale early verdict. The WD3/HV1/JG4 references
changed the admission prediction from 20 to 17. No scientific frontier or family verdict is
being changed by this storage task.

## Validation and landing

**38 tests passed; ruff clean; two visible review passes registered for all three final Python
files.** `review_pass1.json`, `review_pass2.json`, `review_findings.json` and `validation.json`
retain the commands, source hashes and review findings. Tests exercise real unlink/journal
behavior on disposable local fixtures, certificate drift, missing reproduction, live trees,
process blindness, SHA changes with restored stat metadata, ignored pointer references, AP
admission and the legacy executor. Fixture tests establish code behavior, not real reclaimed
bytes. `plan_validation.json` checks all 20 input/output rows and the 17/3 admission split.

Only the executor, its tests and this arm's local artifacts changed. No SSD file was deleted,
moved or rewritten; upstream, live payloads, seals, pointer rows and unrelated staged work
were untouched. No scorer, training, archive materialization or paid job ran. Scientific
DAG/DSL/equation integration is N/A for this storage-custody change (`FORMALIZATION_PENDING`).
The serializer receives post-edit hashes and no attribution trailer. Its exact outcome is
recorded separately in `serializer_status.json`; a fallback bundle is not a main landing.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `.omx/research/ddm_vr5_reclaim_plan_20260909.jsonl` and `.omx/research/ddm_vr5_apply_journal_20260909.jsonl`; fire trigger: harvest this reviewed handoff in a process-visible, Git-writable session, consume any serializer fallback first, then run the exact pinned apply command above. Harvest its journal and final ledger to measure actual reclaim; do not infer success from the plan.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN with WD3/HV1/JG5 consumer owners; consumer store: the three BLOCKED rows in `.omx/research/ddm_vr5_reclaim_plan_20260909.jsonl`; fire trigger: explicit retirement or rebind evidence for every named consumer. Only then build a new hashed plan and repeat every certificate/reference/process gate.

## LIVE-HYPOTHESES

- The 17 admitted raws can return 57.985 GiB of logical storage because all bytes and retained
  reproduction chains match; actual free-space gain and live process clearance remain untested.
- Some of the three blocked consumers may be retired, since their paths belong to older arms;
  that is unproven and cannot authorize their deletion.

## DEAD-ENDS

- All-20 admission is closed for this plan: three rows have unresolved consumer references.
- Empty sandbox lsof as deletion authority is closed: independent host visibility is required.
- Size-only admission, ignoring SHA drift, or adding family-name exceptions bypasses the certificate.
- Drive-to-drive moves create no combined capacity and are outside this deletion-only handoff.

Existing own-vehicle frontier, unchanged by VR5: S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600].

<!-- # FORMALIZATION_PENDING: storage-custody implementation and plan; no new scientific score law -->
