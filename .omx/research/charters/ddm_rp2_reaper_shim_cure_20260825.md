# ddm_rp2_reaper_shim_cure — root-cause cure + two-landing guard for the fleet-reaper shim-PATH kill class (harness task #1189, owning finding recorded in the jo1 r9 arc / task-ledger row 1189)

## MANDATE

Operator 20260825: *"I want you to do whatever it takes and work for as long as it takes
autonomously with full authority and stand and go to accomplish frontier score lowering"*.
Task #1189 (pending, filed 08-22): MAIN's own python-shim PATH is what made the fleet
launchd reaper kill three detached jo1 r9 daemons. The class is NAMED but UNCURED. The
stage-A → ON-15 chain and every future long local burn launches detached daemons; an
uncured reaper class is a standing threat to the campaign's only live route. Cure it at the
true side + land the two-landing guard so the class cannot recur.

## SCOPE

1. RE-DERIVE the root cause at source (never from the task row alone): read the #1189
   finding's evidence — the jo1 r9 daemon kill receipts (grep .omx/research for the r9
   reaper postmortem rows; the fleet reaper is the launchd job the codex-keeper docs name
   as killing hand-rolled processes at ~5-6 min) and the shim at issue (the python
   exec-wrapper shim per memory `python-shim-must-be-exec-wrapper-never-symlink`). Establish
   the EXACT mechanism: which PATH entry routes a daemon's interpreter through the shim, and
   why that makes the reaper classify it as reapable.
2. CURE at the true side: make detached daemons launched via
   `tools/launch_detached_process.py` (and any other canonical launcher) immune — the
   interpreter resolution must not route through the reapable shim surface (absolute venv
   interpreter path, scrubbed PATH, or the shim exec-wrapper carrying the reaper-exempt
   marker — whichever the mechanism derivation shows is the real predicate).
3. TWO-LANDING: (a) the fix; (b) a guard — a launch-time check in the canonical launcher
   that REFUSES (or loudly warns) when the resolved interpreter/env would be
   reaper-classified, with an EXECUTED positive control (a deliberately shim-routed dry
   spawn that the guard catches) and a negative control (the canonical launch passes).
4. Verify the LIVE r7 launch lineage (pid 44616 manifest at
   /Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/training/launch_off_sequential_r7/launch_manifest.json)
   against the derived predicate: state plainly whether r7 was exposed or already safe, with
   the receipt line.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_rp2_reaper_shim_cure/`.
- Do NOT touch the live r7 process, its run dirs, or the fleet reaper's launchd plist
  itself (system-state mutation is operator-gated); the cure lives in OUR launchers/shims.
- Do NOT edit experiments/ddm_wd3_scorer_aware_width_distillation.py (live builder pin
  0b976d0d0a; any edit invalidates the running birth contracts).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- Task #1189's own record — the kill was misread as capacity/permission until the shim PATH
  was identified; do not re-litigate the misreads, verify the shim mechanism at source.
- `python-shim-must-be-exec-wrapper-never-symlink` (memory) — the symlink form was already
  refused once; the cure must not reintroduce a symlink shim.
- `foreground-bash-total-wallclock-is-the-reaper-trigger` (memory, m-index) — the HARNESS
  reaper (rc144 at ~3 min foreground) is a DIFFERENT reaper than the fleet launchd reaper;
  do not conflate the two — the cure targets the fleet reaper only.
- `codex-arm-blocked-git-sandbox-class-main-handoff-cure` (memory) — arm git-block class is
  unrelated; if the arm cannot commit, hand the diff to MAIN per the established handoff.

## OPTIMAL FORM

- Family exemplar: the #1163 kill-doesn't-reach-the-tree class cure (reference:
  tac.process_group_kill migration + STRICT gate, task #1177, landed) — same shape: named
  process-lifecycle class → mechanism derivation → canonical-surface cure → executed-control
  guard. Sister exemplar commit 26f17f2f04 (the #1216 guard-pair cure: fix at BOTH call
  sites, never one).
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN (no "documented the
  workaround" closure — the guard must execute its positive control).
- **PRIOR-LAW PREDICTION (falsifiable):** the shim-PATH mechanism predicts the r7 launch
  (canonical launcher, absolute .venv interpreter in the driver script) was ALREADY SAFE —
  its 1.7 h+ survival is explained by interpreter resolution bypassing the shim. FALSIFIER:
  if the manifest/process tree shows r7's interpreter DID route through the reapable shim,
  the survival is luck not safety, the class is live on the critical path, and the cure
  escalates to before-ON-15 priority — count it plainly.

## DELIVERABLE

`.omx/research/ddm_rp2_reaper_shim_cure_20260825.md` — mechanism derivation with receipts ·
the cure diff (committed, 2 review passes) · guard with BOTH executed controls · the r7
exposure verdict line. Commit via the serializer. End with the own-vehicle frontier line.
