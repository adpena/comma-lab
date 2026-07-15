# FEED-C4 — mod-19 witness rate byte-close

**UTC:** 2026-07-15T08:37:16Z  
**Lane:** `lane_c4_mod19_rate_byteclose_20260715`  
**Status:** `BLOCKED_NO_ELIGIBLE_WITNESS_CHECKPOINT`  
**verdict_scope:** INSTANCE — C4 in this isolated worktree plus the connected SSD artifact tier. The mod-19 family remains open.  
**Pointer:** submittable 0.19108 UNMOVED; borrowed-bank 0.18804 UNMOVED. This is a MEANS row, not a pointer mover.

## STORES CONSULTED

- `experiments/results/` in the isolated worktree: no run directories, `levelset_best.json`, clean best EMA, or complete stage checkpoint.
- `/Volumes/VertigoDataTier/pact`: no C1/V9 witness checkpoint. Only `det_gpu_348_smoke/**` level-set artifacts were found.
- `/Volumes/APDataStore/pact`: tier absent.
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, `.omx/state/canonical_task_status.jsonl`.
- `src/tac/witness_dsl/spec_v9_cgauge.py`, `tools/levelset_byte_close_and_eval.py`, and the v7.5/v8 operating-contract specs.
- Live inbox through broadcast cursor `2026-07-14T20:32:37Z`; no C4-specific override or stop directive.

## Artifact ruling

The newest checkpoint-like object found was
`/Volumes/VertigoDataTier/pact/det_gpu_348_smoke/timing/fused200/levelset_witness_ema_BEST.npz`
(SHA-256 `b7d54b01dd34a553a37f6150628bab2426308e341efcab0c7c865dc48900e7b3`). It is ineligible:
`d_seg=0.4939778645833333` at epoch 200, `hidden_dim=32`, `n_hidden=2`,
`git_sha=unknown`, and `upstream_snapshot_sha256=unknown`. Its path and configuration identify it as
a determinism/timing smoke, not the clean C1/V9 witness checkpoint requested by C4. Byte-closing it would
manufacture a scientifically irrelevant row.

Therefore the raw measurement fields are intentionally `null`: baseline bytes, mod-19 bytes, exact
rate delta, both advisory `d_seg`/`d_pose` rows, and net advisory delta. No archive or evaluation was run.
The machine-readable receipt is `c4_mod19_rate_byteclose_20260715.json`.

## Triality

- **DSL:** `compile_v9_cgauge_ideal_mod19_launch_config` and
  `compile_v9_cgauge_ideal_mod32_launch_config` are the held named presets. Their compiler states the
  family A/B is identical in scientific argv except `--mod-dim`, with separate output custody paths.
- **DAG:** this FEED records the fail-closed custody result and routes the missing producer to
  `C1-WITNESS-CLEAN-STAGE-EMA-20260715`.
- **Equations:** `cgauge_whitney_moddim_v1` is DERIVED support for chart-valid capacity
  (Whitney 17 plus gauge margin 2 = 19). It does not prove `d_seg` neutrality, compressed bytes, or a
  score win. Those remain UNMEASURED until C4 runs on the produced checkpoint.

## Named downstream tickets

1. `C1-WITNESS-CLEAN-STAGE-EMA-20260715` — produce and custody the eligible checkpoint without
   authorizing any new launch in this C4 lane.
2. `AUTH-C4-MOD19-LINUX-X86_64-20260715` — replay the exact C4 archives with
   `upstream/evaluate.py` on Linux x86_64 after the local receipt exists. No authority number is
   inferred here.

## Round-1 review

`PASS_HONEST_BLOCKER`: exact byte fields were not filled from an estimate; advisory fields were not
promoted; the absent checkpoint is explicit; the negative is instance/custody-scoped; DSL/DAG/equation
legs are separated; both the producer and authority follow-on are named.

## Landing custody

The serializer attempt failed at `git add` with rc=128: `unable to create temporary file: Operation not
permitted` while inserting `.omx/state/canonical_task_status.jsonl`. No commit SHA exists. The files remain
in the isolated working tree for main's uncommitted-diff harvester. This is
`GIT_OBJECTS_READ_ONLY_SERIALIZER_RC128`; no force, bypass, or destructive git operation was attempted.
