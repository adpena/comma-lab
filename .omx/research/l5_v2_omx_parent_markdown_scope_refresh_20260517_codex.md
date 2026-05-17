# L5 v2 OMX Parent Markdown Scope Refresh - 2026-05-17

## Why This Exists

The operator flagged that fresh L5, cargo-cult, and score-lowering directives
may sit outside `.omx/research`. I widened the control-plane scan to every
Markdown file under `.omx`, then refreshed the stale state surfaces that future
agents are most likely to read first.

## Scan Commands

```bash
rg --files .omx -g '*.md'
find .omx -path .omx/research -prune -o -type f -name '*.md' -print
rg -n -i 'l5|staircase|time[-_ ]?traveler|tt5l|cargo[-_ ]cult|class-shift|score[-_ ]lower|frontier|current focus|next experiment' .omx --glob '*.md' --glob '!.omx/research/**'
```

## Corpus Result

- `.omx/**/*.md` count observed: 1769 files.
- Non-research Markdown exists under `.omx/state`, `.omx/context`,
  `.omx/plans`, `.omx/specs`, `.omx/interviews`, `.omx/tmp`, and
  `.omx/auto_memory_snapshot_*`.
- Current L5/TT5L authority outside `.omx/research` was concentrated in:
  `.omx/state/current_focus.md`, `.omx/state/next_experiments.md`, and
  `.omx/state/active_lane_dispatch_claims.md`.
- `.omx/notepad.md` and early `.omx/context` files are useful historical AV1
  Track-B memory, but they are not current authority for L5-v2 decisions.
- `.omx/release_manifest_v0.2.0-rc1.md` is release-hygiene context, not a
  current score or dispatch authority.

## Findings

1. `.omx/state/current_focus.md` was stale: it still described the May 15
   TT5L rebaseline and did not carry the May 17 T4 symposium pivot.
2. `.omx/state/next_experiments.md` was stale: it still listed the May 9
   A1-sidecar/Phase-1 queue instead of the May 17 Rule #6/L5-v2 work order.
3. `.omx/state/dispatch_queue.md` is older HTD state and should not override
   the current Rule #6/L5-v2 queue.
4. `.omx/state/active_lane_dispatch_claims.md` remains live and authoritative
   for dispatch conflicts. Recent TT5L paired CPU/CUDA diagnostic rows are
   terminal and non-promotional.
5. The May 17 T4 symposium changes the active score-lowering order:
   Rule #6 bolt-ons on A1 first, TT5L/L5-v2 side-info/probe evidence in
   parallel, high-risk per-pair-conditioning substrates behind
   SCORER-AWARENESS probes.

## Changes Made

- Rewrote `.omx/state/current_focus.md` as the 2026-05-17 current focus:
  A1 anchor, Rule #6 priority, TT5L/L5-v2 status, parent-scope scan record,
  and dispatch discipline.
- Rewrote `.omx/state/next_experiments.md` as the 2026-05-17 queue:
  A1 Ballé hyperprior bolt-on, A1 PR101-style entropy bolt-on, A1 VQ-codebook
  bolt-on, TT5L side-info effect curve, SCORER-AWARENESS probe wave, and Z6
  per-frame-renderer-axis replacement.
- Refreshed the TT5L side-info Lightning paired-axis plan chain against
  current `main` commit `9b926ab6e099585d64a76726db91c8af6be0f181`.

## TT5L Artifact Refresh Result

Commands run:

```bash
.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py
.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_lightning_execution_preflight.py
.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py
.venv/bin/python tools/verify_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py
.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py
.venv/bin/python tools/build_l5_v2_tt5l_lightning_route_unblock_packet.py
.venv/bin/python tools/build_l5_v2_tt5l_lightning_doctor_plan.py
.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py
```

Observed state after refresh:

- Paired-axis source commit:
  `9b926ab6e099585d64a76726db91c8af6be0f181`.
- `source_relevant_paths_match=true`.
- Execution preflight: `10/10` cells ready for operator claiming.
- Execution bundle dry-run verification: `10/10` cells passed.
- Harvest cells: `0/10` exact-eval artifacts harvested, so
  `ready_for_effect_curve_build=false`.
- Route-unblock packet: zero artifact blockers; remaining route work is the
  expected non-dry-run work, not a stale-source blocker.
- Required doctor plan: `ready_for_operator_doctor=true`.
- Architecture lock: `architecture_lock_allowed=false` with blockers
  `requires_all_l5_v2_gate_evidence_valid`,
  `requires_c1_z5_tt5l_probe_gate_evidence`, and
  `requires_paired_cpu_cuda_sideinfo_effect_curve`.

## Authority

- No provider dispatch was launched.
- No lane claim was opened.
- No archive was built.
- No score, promotion, rank, kill, or architecture-lock authority is created.

## Next Gate

Regenerate any L5/TT5L custody artifacts whose source-commit fields are stale
against current `main`, then re-run focused tests. If the stale fields are only
historical and the architecture-lock packet still fails closed for the right
reasons, leave spend blocked and move to the first Rule #6 byte-closed
prototype.

## Current-Main Addendum

The operator specifically re-raised the possibility that OMX/Claude signal may
sit outside `.omx/research`. I rechecked the parent-scope Markdown surface:
top-level `.omx` Markdown is only `.omx/notepad.md` and
`.omx/release_manifest_v0.2.0-rc1.md`; active non-research authority remains
concentrated in `.omx/state/current_focus.md`,
`.omx/state/next_experiments.md`, and `.omx/state/active_lane_dispatch_claims.md`.
The ignored `.omx/auto_memory_snapshot_20260504T230223Z` and `.omx/tmp`
Markdown remain forensic inputs only per the no-ignore follow-up ledger.

The L5/TT5L route artifacts were then regenerated against `main` commit
`9b926ab6e099585d64a76726db91c8af6be0f181`. This was a custody refresh, not a
dispatch or score claim: the route packet still has zero artifact blockers, the
doctor plan remains `ready_for_operator_doctor=true`, dry-run verification
remains `10/10`, harvest remains `0/10` exact-eval artifacts, and architecture
lock remains forbidden until paired CPU/CUDA side-info evidence exists.
