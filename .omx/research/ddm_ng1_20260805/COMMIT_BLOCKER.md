# NG1 commit blocker

Date: 2026-08-05.

The required serializer commit was attempted with explicit `base=new` and post-edit SHA-256 guards for all NG1 artifacts, using:

```bash
REVIEW_GATE_OVERRIDE=1 .venv/bin/python tools/subagent_commit_serializer.py --message "ddm_ng1: negative-results audit [no-triality] [p0-ledger-ok]" --files .omx/research/ddm_ng1_20260805/negative_verdict_ledger.jsonl .omx/research/ddm_ng1_20260805/cap_artifact_sweep.jsonl .omx/research/ddm_ng1_20260805/pose_rerun_fire_orders_923.jsonl .omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md --base-content-sha256 .omx/research/ddm_ng1_20260805/negative_verdict_ledger.jsonl=new --base-content-sha256 .omx/research/ddm_ng1_20260805/cap_artifact_sweep.jsonl=new --base-content-sha256 .omx/research/ddm_ng1_20260805/pose_rerun_fire_orders_923.jsonl=new --base-content-sha256 .omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md=new --expected-content-sha256 .omx/research/ddm_ng1_20260805/negative_verdict_ledger.jsonl=2a5764ae1529f3d2945ae31a243ae0d45fbeb770dd831ea38462a2be0deec72b --expected-content-sha256 .omx/research/ddm_ng1_20260805/cap_artifact_sweep.jsonl=0fb6b769d9d008ca018ec736daf6e5115b270693139c71dd72f1dff99dea45a4 --expected-content-sha256 .omx/research/ddm_ng1_20260805/pose_rerun_fire_orders_923.jsonl=55043450b63faefeb494c695b0d6eebcfe823cc8a9e33ff5fb07bf2ec3c661b5 --expected-content-sha256 .omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md=857f32168b798995d62d21ee6f7e9e1a0c51f576626c64714ea4702d941eec81 --triality-legs none --triality-reason "report-only negative audit; no DAG/DSL/equation change" --no-co-author
```

Serializer result: blocked at `git add` with return code 128:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_ng1_20260805/cap_artifact_sweep.jsonl: failed to insert into database
error: unable to index file '.omx/research/ddm_ng1_20260805/cap_artifact_sweep.jsonl'
fatal: adding files failed
```

Post-failure index check: no NG1 paths are staged.

Artifact hashes before this blocker was added:

| path | sha256 |
|---|---|
| `.omx/research/ddm_ng1_20260805/negative_verdict_ledger.jsonl` | `2a5764ae1529f3d2945ae31a243ae0d45fbeb770dd831ea38462a2be0deec72b` |
| `.omx/research/ddm_ng1_20260805/cap_artifact_sweep.jsonl` | `0fb6b769d9d008ca018ec736daf6e5115b270693139c71dd72f1dff99dea45a4` |
| `.omx/research/ddm_ng1_20260805/pose_rerun_fire_orders_923.jsonl` | `55043450b63faefeb494c695b0d6eebcfe823cc8a9e33ff5fb07bf2ec3c661b5` |
| `.omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md` | `857f32168b798995d62d21ee6f7e9e1a0c51f576626c64714ea4702d941eec81` |

Disposition: artifacts are durable in the working tree but uncommitted because the managed sandbox refused Git object writes.
