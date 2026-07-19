# Task #537 resumability fix, round 2 — Codex handoff (2026-07-19)

## Verdict and scope

**IMPLEMENTATION READY FOR MAIN METAL VERIFICATION; MERGE BLOCKED UNTIL MAIN'S NEW
RECEIPT HAS `all_pass: true`.** This is a durability-proof verdict only. It is
not score evidence, does not authorize a launch, and does not change the
frontier pointer. The canonical pointer remains `0.1910828242 [contest-CPU Linux
x86_64]` in `reports/latest.md`.

The governing craft contract is `docs/operating_manual_craft_handoff.md`; the
probe also follows the resumability and per-stage-checkpoint contract in
`AGENTS.md` and the v7.5 operating contract in
`.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`.

## Stores consulted

- delegated authority file and its verified SHA-256;
- `docs/operating_manual_craft_handoff.md`, `AGENTS.md`, `CLAUDE.md`, the v7.5
  and v8 vehicle specifications;
- `/Volumes/VertigoDataTier/pact/evidence/probe537_metal_20260719_receipt.json`
  and the preserved files beneath its sibling run directory;
- prior Task #537 implementation, tests, and round-1 handoff;
- `reports/latest.md`, the lane registry, subagent progress ledger, and current
  task-specific/broadcast inboxes.

No task-specific inbox directive superseded this authority.

## Re-derived failures in the prior Metal receipt

MEASURED:

1. The prior receipt says `BLOCKED_ENVIRONMENT_NO_METAL_DEVICE`, but its captured
   error is actually the resume arm's missing relative interpreter:
   `launch.sh: line 12: .venv/bin/python: No such file or directory`.
2. That same preserved directory contains completed continuous, crash, and
   resumed checkpoint files. In particular, crash has exactly one immutable
   `levelset_periodic_resume_stage_unify_tau_ep3.npz` and its EMA mate; continuous
   and resumed each contain the final stage EMA/resume pair at epoch 4.
3. The old final-pair check mixed the two final stage files with the two periodic
   files, so a valid preserved pair was reported as false.

DERIVED:

- Killing only the `bash launch.sh` wrapper does not prove that its trainer child
  was killed. The proof must create and kill a process group.
- Independent Metal processes need not be bit identical. A GPU replay must first
  measure the same-host control-to-control floor and then report both that floor
  and resumed-to-nearest-control distance. CPU-locked restoration remains the
  strict bit-identity verdict.
- A no-Metal environment status is truthful only when the exception names that
  condition and no checkpoint was created. Once checkpoint bytes exist, the
  receipt must report the executed proof error instead of an environment label.

## Fix landed in this worktree

- Two uninterrupted controls measure host nondeterminism before crash/resume.
- `--mlx-device cpu` requires bit identity for live/EMA, optimizer, RNG, event,
  multigrid/controller, and stage state.
- `--mlx-device gpu` uses the exact per-array control A/B maximum absolute delta
  as its measured envelope, with no guessed multiplier, while still requiring
  structural state identity.
- The crash arm starts a new process session and sends `SIGKILL` to the entire
  process group after the immutable epoch-3 checkpoint event.
- Resume binds directly to the immutable periodic epoch-3 resume file. The
  receipt records `crash_epoch` from that file.
- Final pair discovery excludes periodic files and separately requires exactly
  one final EMA and one final resume checkpoint in both controls and the resumed
  arm.
- Seed `537`, pair count, device, exact argv, typed-DSL hashes, comparisons, and
  checkpoint custody are recorded in receipt schema
  `tac.resumability_537_real_crash_resume.v2`.
- Assertion failures retain their evidence and return nonzero without being
  overwritten by a generic environment receipt.

## Local CPU-locked attempt

MEASURED: the governed `n=24`, seed-537, CPU-selected command reached DSL
admission, then failed at `import mlx.nn` before epoch zero because this sandbox
exposes no Metal device. It wrote no checkpoint. The durable receipt is
`.omx/research/resumability_537_fix2_cpu_receipt_v2_20260719.json` with status
`BLOCKED_ENVIRONMENT_NO_METAL_DEVICE`, an empty `checkpoint_files` list, and the
actual exception. Its explicit `crash_epoch: null` and
`final_pair_preserved: false` are truthful because execution stopped before the
first checkpoint. The earlier same-turn receipt without those two explicit
fields is preserved as a superseded development receipt. Therefore local CPU
restoration is **NO VERDICT — ENVIRONMENT ONLY**, not a negative on checkpoint
restoration.

The small failed-at-import launch/provenance directory remains preserved at the
receipt's `cleanup.base_preserved` path. The 5,078,017,610-byte source GT cache
was read-only and was not copied or deleted.

## Exact MAIN Metal command

Run this from MAIN after reviewing the worktree diff; do not merge on a failed
or incomplete receipt:

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/probe_resumability_537_real_smoke.py --python /Users/adpena/Projects/pact/.venv/bin/python --gt-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz --base /Volumes/VertigoDataTier/pact/evidence/probe537_metal_fix2_20260719 --receipt /Volumes/VertigoDataTier/pact/evidence/probe537_metal_fix2_20260719_receipt.json --mlx-device gpu --num-pairs 24 --seed 537 --timeout 900
```

MAIN acceptance requires all of the following in that new receipt:

- `all_pass: true`, `comparison_mode: "gpu_measured_host_floor"`;
- two completed controls plus a whole-process-group kill at crash epoch 3;
- exactly one preserved final EMA/resume pair at epoch 4 in both controls and
  the resumed arm;
- measured control floor and resumed comparisons for model and optimizer state;
- structural state identity, restored optimizer, byte-close loadability, and no
  source-cache deletion.

## Verification and round-1 self-review

Fresh verification:

- new probe unit tests: `6 passed`;
- checkpoint preflight and resume registry: `30 passed`;
- focused trainer resume tests: `18 passed, 38 deselected`;
- full existing trainer resume module: `52 passed, 4 failed`; the four failures
  are the already-recorded main-entry tests that stop at DSL admission before
  their target assertions, not regressions introduced by this patch;
- Ruff, Python compilation, and `git diff --check`: clean.

Adversarial self-review round 1 (four checks):

1. Re-derived final-pair cardinality from filenames rather than trusting the old
   receipt; fixed the periodic/final glob collision.
2. Traced the crash signal to the wrapper/child boundary; fixed whole-process-
   group termination.
3. Re-derived seed emission from the compiled argv; added the explicit typed
   override because the configuration field alone did not emit it.
4. Forced failure classification to inspect preserved checkpoint bytes, then
   exercised both no-checkpoint/no-Metal and executed-proof-error cases.

## Triality and pointer delta

- DSL: the proof remains compiled through the typed witness DSL and records its
  launch/provenance hashes.
- DAG: no scientific readiness edge or launch authority changed; MAIN's
  `all_pass: true` receipt is the remaining merge gate.
- Equations: GPU acceptance uses the measured envelope
  `delta_resume(k) <= delta_control_A_B(k)` for every numeric state array `k`;
  CPU acceptance requires byte identity.
- Pointer delta: exactly zero.

## MAIN landing review required

MAIN must review the two-control/floor semantics, immutable checkpoint binding,
and process-group kill before landing. Then MAIN must execute the command above
on the Metal-capable host and merge only when its newly generated receipt has
`all_pass: true`. A failed receipt is a blocker artifact, not merge authority.
