# ddm_vp1 serializer blocker receipt

Status: deliverables validated but uncommitted. The required serializer was invoked; no direct Git
command or serializer bypass was used.

## Attempt

- Last explicitly observed repository HEAD before the attempt:
  `34929f68a9dc8c9d35607984b817c6bd013e012c`. HEAD subsequently advanced through unrelated CP2
  and contract commits; the vp1 files remained untracked.
- Commit message: `ddm_vp1: re-score vehicle inventory on PR130 [no-triality] [p0-ledger-ok]`.
- Serializer options included `--no-co-author`, `--base-content-sha256 <path>=new`, and one
  post-edit `--expected-content-sha256` per intended file.
- The shared staged index was empty before the attempt.

## Failure

The serializer exited 128 during `git add`, before staging or commit:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_vp1_20260810/VP1_RESCORING_REPORT.md: failed to insert into database
error: unable to index file '.omx/research/ddm_vp1_20260810/VP1_RESCORING_REPORT.md'
fatal: adding files failed
```

The managed sandbox exposes Git metadata read-only. Repeating or bypassing the serializer would not
be a valid landing. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** next Git-capable vp1
custodian. **Consumer store:** repository history for `.omx/research/ddm_vp1_20260810/`.
**Fire trigger:** Git-object writes are permitted and the staged index is empty; rehash every file
and invoke `tools/subagent_commit_serializer.py` again.

After the failure, vp1 incorporated the concurrently landed CP2 receipt. The current files therefore
require fresh post-edit hashes; the failed attempt's earlier hashes are not reusable.

## Intended scope

- `.omx/research/ddm_vp1_20260810/VP1_RESCORING_REPORT.md`
- `.omx/research/ddm_vp1_20260810/charters/VP1_TOP1_ANS_REALIZATION.md`
- `.omx/research/ddm_vp1_20260810/charters/VP1_TOP2_SPLIT_MODEL_BANK.md`
- `.omx/research/ddm_vp1_20260810/charters/VP1_TOP3_TEMPORAL_REVERSION.md`
- `.omx/research/ddm_vp1_20260810/COMMIT_BLOCKED.md`

No sister-arm file, protected file, upstream file, or staged-index entry belongs to this unit.
