# ddm_ab1 Commit Intent

status: SERIALIZER_BLOCKED_BY_SANDBOX

Actual serializer attempt: 143 explicit files, `--no-co-author`, `--triality-legs none`, and one `--expected-content-sha256` per file.

Message:

`ddm_ab1 drain #254 admission backlog [no-triality] [p0-ledger-ok]`

Failure:

```text
[subagent-commit-serializer] git add failed (rc=128):
error: unable to create temporary file: Operation not permitted
error: experiments/train_cnerv_as_renderer.py: failed to insert into database
error: unable to index file 'experiments/train_cnerv_as_renderer.py'
fatal: updating files failed
```

Post-failure index check:

`git diff --cached --name-status` returned empty output.

Current replay set: 145 files = 140 trainer entrypoints, `src/tac/tests/test_admission_coverage_gate.py`, and the 4 files under `.omx/research/ddm_ab1_20260807/`.

`POST_EDIT_SHA256SUMS.txt` records post-edit hashes for every replay file except itself (self-hash would be recursive); recompute its hash at replay time if the serializer requires it.
