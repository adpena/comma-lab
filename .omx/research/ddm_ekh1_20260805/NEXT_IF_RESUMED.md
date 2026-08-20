# NEXT IF RESUMED

## Current State

EKH1 completed the live residue audit for the target worktree:

```text
.omx/tmp/codex_worktrees/einstein_kolmogorov_crux_20260719T212159Z
```

Normal Git status in that worktree is empty. The original charter claim of about 17 uncommitted files is stale.

The target branch remains provenance-only. Do not merge it. Main already has the core code/hardening work via `57e4c4e52b` and the EK2 merge-debt closure via `01f2115062`.

## If More Work Is Needed

1. Treat the target branch as read-only provenance unless the operator explicitly reverses the `01f2115062` never-merge disposition.
2. If handling ignored run-output directories, start from the two observed roots:
   - `.omx/research/einstein_kolmogorov_crux_closure_v2_20260720/` in the target worktree, about 1.5M.
   - `.omx/research/einstein_kolmogorov_crux_runs_final_20260719/` in the target worktree, about 28M.
3. Before moving or deleting any ignored artifacts, write a machine-readable certify-or-block manifest with path, bytes, SHA-256 or tree hash, and rebuild reason.
4. Do not run scorers or launches from this custody path.

## Boundaries

The live EK canonical equation is now queryable:

```text
PYTHONPATH=src .venv/bin/python tools/list_canonical_equations.py --equation-id einstein_kolmogorov_crux_action_rate_contract_v1
```

Expected result: one entry, two anchors, producers and consumers present, `well-calibrated: True`.

The pre-existing uncommitted registry row `pose_null_subspace_is_ac_only_v1` was not EKH1 work and remains outside this run's commit scope.

EKH1 did not create a commit. Serializer patch-mode and isolated whole-file staging both failed on Git write operations for `.omx/state/canonical_equations_registry.jsonl` with `Operation not permitted`. A docs-only serializer attempt also failed on `.omx/research/ddm_ekh1_20260805/.done`, confirming a general Git object/index write blocker in this sandbox. If resumed in a Git-writable context, commit the EK registry row plus this receipt package through `tools/subagent_commit_serializer.py` with fresh post-edit SHA declarations, and keep the pre-existing `pose_null_subspace_is_ac_only_v1` registry row out of the EKH1 commit unless its owner explicitly asks to land it.

No follow-on arm was named by EKH1. No scorer slot was used.

Required frontier line remains:

```text
S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]
```
