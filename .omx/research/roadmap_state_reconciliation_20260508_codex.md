# Roadmap State Reconciliation - Codex - 2026-05-08

Owner: codex
Scope: reconcile the pasted comprehensive roadmap against current local
artifacts, exact evidence rules, and launch-ready dispatch state.

## Current Anchored Frontier

The strongest exact local HNeRV anchor found in the current checkout is:

- Artifact:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
- Score: `0.20898105277982337`
- Strict formula score from rounded components: `0.2089810755823297`
- Archive bytes: `185578`
- SegNet distance: `0.00067082`
- PoseNet distance: `0.0000336`
- Samples: `600`
- Evidence: A++ local contest T4 custody per
  `.omx/research/pr103_pr106_ac_repack_exact_eval_20260507_codex.md`

The roadmap line `PR106 contest frontier = 0.20454 [contest-CUDA]` was not
found in the harvested `contest_auth_eval*.json` artifacts scanned under
`experiments/results/lightning_batch`. It remains unanchored in this checkout
until a concrete JSON path, archive SHA, bytes, and runtime tree are provided.

## Active Paid Signal

`arch-shrink-x0-4-lightning-20260508T024304Z` is still running as of the latest
harvest poll:

- Command:
  `experiments/arch_shrink_x0.4_lightning_harvest.py --job-name arch-shrink-x0-4-lightning-20260508T024304Z --teamspace comma-lab --user adpena --ssh-target s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai --once`
- Status: `running`
- Last reported time: `2026-05-08T07:50:42Z`

No score, archive, or promotion can be claimed from this lane yet.

## Tier-A Candidate Readiness

### `admm_x_lossy_coarsening_path_b_step6_no_dead_k`

- Archive: `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T064711Z/archive.zip`
- Bytes: `153671`
- SHA-256: `b7b09089e852872bd67b4b8aa04c1b4d46168bb89343acff81796c5551d63d05`
- Manifest: `ready_for_exact_eval_dispatch=false`
- Main blockers:
  - CPU rel_err proxy, not score evidence
  - no exact CUDA auth-eval JSON
  - apogee_int6 contest-CUDA anchor required first by manifest policy

### `unified_winners_stack` Stage 1+2

- Archive:
  `experiments/results/unified_winners_stack_20260508T071803Z/stage_1_2_uniward_no_dead_k/archive.zip`
- Bytes: `148378`
- SHA-256: `fc539f935641e049f5cae443af930fbdbbec703103439abc45b2abf3a602ed13`
- Manifest: `ready_for_exact_eval_dispatch=false`
- Main blockers:
  - CPU byte proxy, not score evidence
  - no exact CUDA auth-eval JSON
  - UNIWARD variance proxy substitutes for wavelet residual
  - no iterative primal-dual consensus
  - score-aware per-tensor distortion weights are not in-loop

### `apogee_int6`

The latest preserved attempt was refused before score:

- Job: `claude_apogee_int6_override_20260507_101520Z`
- Status: `REFUSED at submit step`
- Failure: Lightning SDK API 400, accelerator T4 unavailable on the AWS cluster
- Candidate remains predicted-band only.

If Lightning capacity is now available, this is the cleanest prerequisite
dispatch to retry after re-running its predispatch sanity gates and making a
fresh dispatch claim.

## Planner Outputs Generated This Turn

Reconciled planning-only artifacts were generated from the A++ `0.2089810528`
anchor, not from the unverified `0.20454` roadmap row:

- `reports/dispatch_advice_pr103_pr106_current_reconciled_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_019_reconciled_20260508.json`
- `reports/cathedral_autopilot_plan_pr103_pr106_to_0155_reconciled_20260508.json`
- `reports/cathedral_meta_lagrangian_ranking_pr103_pr106_to_019_reconciled_20260508.json`

Key planner conclusions:

- Current score decomposition from strict formula:
  `seg=0.067082`, `pose=0.01833030277982336`,
  `rate=0.12356877280250632`, total `0.2089810755823297`.
- To reach `0.190` by bytes alone at fixed components requires archive bytes
  `157071`, or `28507` bytes saved.
- To reach `0.155` by bytes alone at fixed components requires archive bytes
  `104508`, or `81070` bytes saved.
- The operating point is pose-dominated by marginal value: `dS/d(d_pose)` is
  about `2.73x` `dS/d(d_seg)`.
- Meta-Lagrangian bridge produced `n_eligible_for_dispatch=0`, because all
  top candidates are either missing an archive path, CPU/proxy only, exact
  negative, or planning-band only.

## Dispatch Safety Patch Integrated

Worker D patched `tools/parallel_dispatch_top_k.py` so `[CPU-build]`,
`cpu-only`, and `local-only` evidence markers are dispatch blockers alongside
proxy, MPS, predicted, forensic, CPU-prep, and research-signal rows. The
focused test prevents a spoofed CPU-build row with
`ready_for_exact_eval_dispatch=true` from entering exact-eval fanout.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_dispatch_command_builder_shapes.py -q`
  passed with `18 passed`.
- `git diff --check -- tools/parallel_dispatch_top_k.py src/tac/tests/test_dispatch_command_builder_shapes.py`
  was clean.

An adversarial greenup worker was launched to search for remaining bypasses in
the fanout filter.

Adversarial greenup result:

- The first patch only scanned `evidence_semantics` and `evidence_grade`.
- Greenup found that unsafe markers could hide in `evidence_marker`, source /
  provenance text, or `contest_dispatch_verdict`.
- `tools/parallel_dispatch_top_k.py` now scans the broader evidence text
  surface and fails closed when evidence semantics are missing.
- Verification:
  `.venv/bin/python -m pytest src/tac/tests/test_dispatch_command_builder_shapes.py src/tac/tests/test_lightning_dispatch_pr106_stack.py src/tac/tests/test_predispatch_sanity.py src/tac/tests/test_dispatch_cli_shell_hazards.py -q`
  passed with `62 passed`.

## PR106 UNIWARD Runtime Gap Closure

The earlier `PR106 UNIWARD-Lagrangian` row was not dispatch-ready because it
had CPU sweep bytes but no byte-closed runtime packet. This turn built and
smoke-verified the packet:

- Tool: `tools/build_pr106_uniward_runtime_packet.py`
- Verifier: `tools/verify_pr106_uniward_runtime_packet_sha256.py`
- Archive:
  `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip`
- Archive bytes: `150511`
- Archive SHA-256:
  `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`
- Decoder packed brotli: `134550` bytes versus PR106 published `170278`
- Int8 rel_err: `0.046567`
- PR106 decoder weight-identity smoke rel_err: `1.852e-08`
- Deterministic rebuild verifier: passed byte-identical rebuild.

This remains `[CPU-build]` and non-promotable. The state changed from
`no byte-closed archive/runtime` to `byte-closed CPU-build exists; exact CUDA
auth eval requires claim + explicit operator promotion decision`.

## Near-Term Ordering

1. Keep harvesting `arch_shrink_x0.4_lightning` until terminal.
2. Consider PR106 UNIWARD exact CUDA auth eval only after a fresh dispatch
   claim and explicit promotion decision for the CPU-built packet.
3. Retry or relaunch `apogee_int6` only after current predispatch sanity passes
   and a fresh active claim is recorded.
4. Do not dispatch `153671` no-dead-K or `148378` unified winners until the
   manifest blockers are intentionally resolved or superseded by a reviewed
   operator override.
5. Treat `0.20454` as unanchored until its exact JSON artifact is found.
6. Continue building lower-score work, but keep active ranking and exact
   fanout isolated from CPU/MPS/proxy evidence.
