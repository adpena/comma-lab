# s1a seed-2 external SIGKILL — adjudication + the wd3 resumability cure (2026-08-25)

**STORES CONSULTED:** safe_run receipt (`launch_off_sequential_r7/resource_safe_run_status.json`),
run.log, reaper log `/tmp/com.vertigo.claude-code-reaper.log` + plist, unified log
(`log show`, memorystatus/jetsam window), COMPILED_CONFIG.json, wd3 trainer source
(`load_checkpoint`, `_verify_launch_sources`), r7 launch manifest.

## What happened

The r7 sequential OFF burn (#1270 stage A, MAIN_LAUNCH_ORDER sha 708eae6a) ran seed
20260815 to a clean PASS (rc=0, `wd3_stage_end_epoch_0065.pt`, 17:13:01Z). Seed
20260816's trainer (pid 63596) took a bare SIGKILL at 18:40:23Z, ~87 min in.

## Killer adjudication: UNATTRIBUTED-EXTERNAL-SIGKILL (three exonerations)

1. **safe_run exonerated by its own receipt** — `kill_action: null`, peak RSS 8.1 GiB
   vs 116 GiB limit, elapsed 11,885 s vs 18,000 s timeout,
   `receipt_status_disagrees_with_exit: true` (the receipt saying "I did not kill it"
   while the child exited 137).
2. **Fleet reaper exonerated by its log** — it logs every kill unconditionally
   (script line 113/124); `/tmp/com.vertigo.claude-code-reaper.log` has no entry
   13:35–13:45 local and no mention of any s1a pid. Last kill 12:52 (a codex helper).
3. **Kernel memory pressure exonerated** — no memorystatus/jetsam rows in the
   unified log for the window; system had >96 GiB free pages at inspection.

Correlation noted, mechanism unproven: the kill landed at exactly the 1:40pm CT
usage-limit reset boundary; the r7 launch was `start_new_session=True` (own session),
so agent-teardown process-group kills should not reach it. Recurrence on r9 would
discriminate; non-recurrence leaves it a one-off.

## Payload: INTACT (no ALWAYS-KEEP-THE-PAYLOAD incident)

The empty `wd3_output` symlink targets were a red herring (planned root, unused).
Real artifacts: `training/off_seed_*/W96_flattened/checkpoints/` — seed-1 complete
through stage end; seed-2 checkpointed through `wd3_epoch_0030.pt` (13:31, 9 min
before the kill).

## The resumability defect (P0 surface) — TWO self-referential fields

The relaunch surfaced that the trainer's crash-resume path could never succeed,
despite `resumable_from_disk: true` receipts:

1. `load_checkpoint` demanded FULL config equality vs the checkpoint's stored
   as-run config — but `resume_from` must point at the checkpoint itself, a file
   that did not exist when the config was stored. Unsatisfiable by construction.
2. After cure 1, `_verify_launch_sources` refused on `expected_builder_sha256` —
   the trainer's SELF-hash, which necessarily differs when the crash's cure edited
   the trainer file. Same genus: a checkpoint cannot pin the future code that
   resumes it.

**Cure (commits a5fd9ace0b + bfa780756e):** `_resume_config_identity` masks exactly
`resume_from` and `expected_builder_sha256` in the RESUME-identity comparison;
every other field (seed, arm, output, epochs, receiver/adapter pins, optimizer)
stays strict. Source custody is NOT weakened: `_verify_launch_sources` still
refuses unless the LIVE trainer matches the continuation config's pin, so a repin
is an explicit, committed, auditable act. 7 refusal-bound tests
(`test_wd3_resume_config_identity.py`): the masked fields pass, any other drift
refuses.

## Recovery state

r9 LIVE (pid 1860/1872, counter 544, receipt `s1a_off_seed2_resume_r9`): pre-launch
proof PASSED against the real ep30 checkpoint (identity match + live builder pin
match), trainer confirmed past both refusal points and training (42.6% CPU,
6 GiB RSS at t+5.5 min). Continuation config
`off_seed_20260816/COMPILED_CONFIG_resume_ep30.json` (original sealed config
untouched). At the both-OFF endpoint: `tools/s1a_off_floor_adjudicator.py` →
fb1 falsifier → ON-15 per the sealed order, or ENTERED_AND_REFUSED.

verdict_scope: INSTANCE (this trainer's resume contract; the two-field mask is the
general law for any trainer whose checkpoint stores its as-run config).
