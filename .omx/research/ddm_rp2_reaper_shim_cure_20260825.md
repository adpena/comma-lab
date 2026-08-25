# ddm_rp2 reaper shim cure — source verdict, cure, and guard

## Verdict

**CURED at both canonical launcher surfaces.** The three jo1 r9 deaths were fleet-reaper kills, but the causal token was not `tools/host_shims/python` and not interpreter resolution through that shim. The stable `safe_run.py` command line retained the entire injected `PATH=...` as an argument; that PATH contained many standalone `claude` and `codex` components. The reaper matched those words after 300 seconds and killed PIDs 28505, 37877, and 54539. The host shim by itself contains neither word and is an exec-wrapper to the absolute repository venv.

This materially corrects task #1189's shorthand “python-shim PATH”: the shim was first in the polluted PATH, but it was not the matching substring and it did not cause the long-lived interpreter to remain a shim process. The exact class is **a detached stable argv that retains a `claude`/`codex`-bearing environment string without one of the fleet source's persistent exemptions**.

The requested two-landing shape is atomic in this landing:

- **Landing A — true-side cure:** `spawn_durable_daemon.py` no longer relies on a transient `/usr/bin/env REAPER_KEEPALIVE=1` prefix. `launch_detached_process.py` gives its long-lived supervisor a persistent `REAPER_KEEPALIVE` argument while permitting only a guarded child argv.
- **Landing B — permanent guard:** `fleet_reaper_guard.py` models the fleet source's exact name and exemption regexes against the stable post-`env` argv. Both canonical launchers call it before `Popen`; the detached launcher calls it again inside the supervisor. The legacy `--allow-reaper-name-match` flag no longer bypasses refusal.

## Mechanism derivation

The read-only fleet source snapshot, SHA-256 `06cd6a4f...`, establishes the predicate:

1. snapshot `ps -eo pid,ppid,tty,etime,rss,command` (`claude-code-reaper.sh:138-139`);
2. exclude command lines matching `codex_runs/|REAPER_KEEPALIVE|/Applications/[^ ]*\.app/` (`:141-154`);
3. select lines containing `\b(claude|codex)\b` (`:156-182`);
4. require no TTY, plus PPID 1 or dead stdin, and age at least 300 seconds (`:167-177`);
5. send TERM, then KILL if needed (`:107-129`).

The durable evidence receipt is `/Volumes/APDataStore/pact/ddm_rp2_reaper_shim_cure/evidence_receipt.json`. It records the volatile source log's bytes/hash before extracting these six facts:

| PID | Candidate receipt | Reaped receipt |
|---:|---|---|
| 28505 | 2026-08-21 23:59:58, PPID 1, age 344 s | 2026-08-22 00:00:00 |
| 37877 | 2026-08-22 00:08:03, PPID 1, age 359 s | 2026-08-22 00:08:05 |
| 54539 | 2026-08-22 00:23:12, PPID 1, age 354 s | 2026-08-22 00:23:14 |

The matching process in every receipt was the absolute venv Python running `tools/safe_run.py`. The corresponding r9 launch rows (`train.log:1,6,11`) retained `env PATH=tools/host_shims:...` in that stable process's argv. The PATH then contained `/opt/homebrew/Caskroom/codex/...`, `.codex/...`, `codex.system/...`, and `.claude/plugins/...`. Any of those standalone words satisfied the phase-1 regex. By contrast, `tools/host_shims/python:1-6` is an ordinary shell file ending in `exec .../.venv/bin/python "$@"`; it is not a symlink, it contains no matching path component, and after `exec` it is not the stable process image.

Commit `b5738f4b1984af4a0bf2f00a2fe2af86b8d2190e` had already attempted a cure before this charter by prefixing all durable spawns with `/usr/bin/env REAPER_KEEPALIVE=1`. Its unit tests inspected the pre-exec list only. Because `env` installs the variable and execs the target, the marker is not necessarily present in the long-lived target's visible argv. This arm does not count that transient prefix as a durable exemption; it supersedes the implementation and the tests with stable-process reasoning.

## Cure and guard surfaces

- `tools/fleet_reaper_guard.py` owns the source-equivalent regexes, strips simple leading `env KEY=VALUE` wrappers recursively, suppresses exemptions for unresolved `env` option shapes, returns manifest-ready facts, and refuses before spawn.
- `tools/spawn_durable_daemon.py::spawn_detached_verified` uses the shared assertion for every no-PTY detached spawn. A real controlling PTY remains the distinct, source-valid exemption for intentionally named Codex/Claude sessions from task #525.
- `tools/launch_detached_process.py` assesses the post-resource-wrap child argv, refuses before creating an output directory, writes the assessment into successful manifests, and rechecks inside `_supervisor_main` immediately before the child `Popen`.
- The supervisor receives `--fleet-reaper-keepalive REAPER_KEEPALIVE`; unlike an `env` prefix, this remains in the supervisor's argv for its lifetime and therefore protects it from matching words carried in its child-command tail.
- The existing exec-wrapper shim is unchanged. No symlink was introduced and no PATH was globally scrubbed.

## Executed controls

Both commands used the real canonical detached launcher and the real `safe_run.py` outer shape. They were scorer-free `[macOS-CPU apparatus]` controls.

**Positive, deliberately polluted shim-routed dry spawn:** `PATH` began with `tools/host_shims` and also carried `.claude` and `codex` components. The launcher returned **rc 5**, reported `matched_tokens=[claude,codex]`, `refused=true`, and created no positive output directory.

**Negative, canonical shim-only dry spawn:** the same shape used `PATH=tools/host_shims:/usr/bin:/bin`. It returned **rc 0** with `matched_tokens=[]`, `refused=false`. Durable manifest:

`/Volumes/APDataStore/pact/ddm_rp2_reaper_shim_cure/controls/executed_20260825/negative_passed/launch_manifest.json`  
2795 B · SHA-256 `d12d28480630c80cb04da4deede960fae2b9b6a6924d42cf4f5bd03f267f604c`

Automated verification: **46 passed, 2 deselected** across the new guard controls, watched detached launcher, durable-daemon core, and SIGURG launch guard. The two deselections are the pre-existing macOS TTY tests whose in-test `ps` calls are prohibited by this managed sandbox; the non-`ps` process tests passed. `py_compile`, `git diff --check`, and Ruff (excluding two pre-existing unrelated warnings in `spawn_durable_daemon.py`) passed.

## Live r7 exposure verdict

**PREDICTION CONFIRMED: r7 was already safe.** Read-only manifest receipt:

`launch_manifest.json generated_utc=2026-08-25T15:22:17Z pid=44616 effective_argv[0]=/Users/adpena/Projects/pact/.venv/bin/python reaper_predicate_hits=[] reaper_name_match_allowed=false done_receipt_path=/Users/adpena/Projects/pact/.omx/tmp/codex_runs/s1a_off_sequential_r7.done`

At the 2026-08-25T17:21Z read, its existing safe-run receipt still said `status=running`, `elapsed_s=7137.358`, `child_pid=44627`, `kill_reason=null`, `peak_rss_mib=8132.969`, and no done receipt existed. This is more than 23 reaper grace windows. The absolute venv interpreter bypassed the host shim; the effective child argv had no name-predicate hit; and the supervisor's done-receipt argument carried the fleet source's exact `codex_runs/` exemption. No r7 file or process was mutated.

## RECALL EVIDENCE

Queries covered the full required corpus:

- research and receipts: `reaper|REAPER_KEEPALIVE|host_shims|jo1|r9|28505|37877|54539` across `.omx/research/`, the jo6 r9 scratch receipts, launcher tests, git history, and the fleet source/log;
- canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json` filtered by `reaper|daemon|detached|process lifecycle|spawn`;
- research graph: `reaper|#525|#1189|task 1189` across `CANONICAL_RESEARCH_INDEX*`, every `sub015_DAG_*`, `.omx/research/harness_tasklist_bridge_20260803.jsonl`, and `.omx/state/canonical_task_status.jsonl`;
- implementation census: `start_new_session=True|os.setsid|setsid|spawn_detached_verified|launch_detached_process.py` across `tools/`, `experiments/`, and `scripts/`.

Beyond the charter seeds, recall found (a) task #525's PTY cure and verified 10-minute control in DAG FEED-reaper525, (b) the fleet source's exact persistent exemptions, and (c) the pre-charter `b5738f4b` transient-prefix implementation. These findings changed the plan: I preserved the valid PTY path, did not modify the shim, corrected the incomplete `env` prefix, and centralized the stable-argv guard across the two launchers the repository itself names canonical. No reaper-specific canonical equation was found. No #1189 row was found in the bounded committed index/DAG/bridge/status scope; the charter and jo6 arc are the task authority.

## Boundaries and integration disposition

- No scorer ran, no archive or video payload materialized, no Modal dispatch fired, and no score component was measured. `upstream/` remained read-only.
- The live r7 process/run directory, the fleet LaunchAgent plist, and `experiments/ddm_wd3_scorer_aware_width_distillation.py` were not touched.
- This is production apparatus, not a candidate representation. Sensitivity-map, Pareto, bit-allocation, cathedral candidate-dispatch, and continual-learning posterior hooks are N/A because the landing changes no score action, tensor importance, candidate, or empirical score anchor.
- The probe-disambiguator obligation is satisfied by the executable stable-argv control matrix: polluted retained PATH refuses, shim-only PATH passes, top-level transient `env` is assessed post-exec, a persistent exemption passes, and the deprecated bypass refuses.

Own-vehicle frontier: **gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**, unchanged by this scorer-free apparatus arm.
