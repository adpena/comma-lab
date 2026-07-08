# Hardening sweep 2026-07-08 (respawn) — incremental memo

STORES CONSULTED: git log/status (predecessor + sibling landings: ec660ca41 #348 trainer/registry, 6bd8ddd86 wave-B instruments, a38eee199 cross-platform shebang, f1dd0cc2f C4 per-group-grad-clip); mod32cap run dir (READ-ONLY: run.log tail, launch.sh bytes + mtimes); tools/launch_witness_run.py (current + f1dd0cc2f historical via git show); src/tac/witness_autoconfig.py to_command; src/tac/tests/test_launch_witness_run.py; CLAUDE.md non-negotiables (two-landing rule, NEVER weaken a test, cross-platform-by-default L23); MEMORY.md CURRENT-STATE.

CONTRACT: foreground only; NO launches; run dirs read-only; serializer commits with post-edit shas.

## Item 1 — launcher continuation bug (GATES RUN-1) — status: DIAGNOSED, fix in progress

**Verdict scope: mod32cap launch.sh failure (`line 60: --ckpt-every: command not found`), measured from file mtimes + run.log timeline; applies to the launch.sh write path, not the trainer.**

- Timeline (MEASURED): run started 2026-07-06T11:56; `launch.sh` mtime 2026-07-06T17:24 (rewritten ~5.5h INTO the live run); trainer exited 2026-07-07T16:06; error printed at that moment.
- Root cause (DERIVED from bash semantics + inode truncation): `write_launch_sh` used `Path.write_text` = in-place truncate on the SAME inode. bash reads scripts incrementally from an open fd; when the long trainer command finished, bash resumed reading at its saved byte offset inside the REWRITTEN (shifted) file and executed the orphaned flag line `--ckpt-every 25 \` as a command. The builder's continuation-joining itself is correct (both f1dd0cc2f and HEAD join all flags with ` \` + final line bare; `bash -n` passes on generated output).
- The failing test `test_build_launch_sh_structure` asserts `#!/bin/bash` — STALE: commit a38eee199 (operator cross-platform-by-default 2026-07-07) deliberately moved the builder to `#!/usr/bin/env bash` and did not update the test. The BUILDER is right; the test assertion is updated to the deliberate landed contract (this is not weakening-to-match-a-bug; it is aligning with an operator-directed change the commit should have carried).
- Fix: (a) `write_launch_sh` → atomic tmp + `os.replace` in the same dir (new inode every write; a live bash keeps its fd on the old content — the class is structurally extinct); (b) test updated to `#!/usr/bin/env bash`.
- Class-guard tests: generated script passes `bash -n`; every command-block line except the last ends with `\` (no orphaned arg lines); rewrite replaces the inode (st_ino changes) instead of truncating.

## Item 1 — LANDED e28ff371e (atomic write + shebang test alignment + class-guards; 41/41 launcher tests).

## Item 2 — #265 — LANDED f105ff114
**Verdict scope: the custom Metal grouped-backward under non-GPU default device; measured via reproduction + parity tests.**
- Crash REPRODUCED: adapter installed under process-default gpu, run under cpu → `ValueError: [metal_kernel] Only supports the GPU.`
- Fix (a): levelset trainer converts scorer INSIDE `temporary_mlx_device(args.mlx_device)` (base trainer already correct). Fix (b): library VJP re-checks device at CALL time, fails soft to native VJP (correct on CPU; parity vs reference 3e-8 MEASURED) with LOUD once-only warning.
- Sibling sweep (order-of-check class): `metal_fused_r_operator` = fail-closed at RESOLVED config (`--fused-r-kernel` refused unless `--mlx-device gpu`; availability asserts call-time) — NOT vulnerable. AA-SDF / margin / curvelet / clDice / island-birth: no `mx.fast.metal_kernel` (mx.compile/numpy, device-portable) or FLAGGED_NOT_BUILT. Only grouped-backward had the bug.
- Guards: 3 new tests (`test_metal_grouped_conv_backward_device_fallback.py`); kernel suite 60/60; fused-R 25/25.
- Task #265 status: fix landed — parent/operator should mark the TaskList row done (subagent has no TaskList write surface).
## Item 3 — harness-ledger dashboard_false_FAIL_at_init closure — DONE (763e2bea7 + ledger rows)
**Verdict scope: the two 2026-07-07 recurrence rows' named class-cures; sweeps measured by grep + node --check + test suites.**
- Sweep (a) log-path-split: ALL log→run-dir consumers route through `resolve_run_dir_for_log` (dashboard_server.py:78, render_levelset_dashboard.py:97, schedule_readback = home). Every OTHER run.log consumer verified DIR-FIRST (immune by construction): costate_observer_loop / costate_shadow_report / costate_digest / witness_checkin take run_dir and derive the log; dashboard_up / dashboard_supervisor / dashboard_fm_events / dashboard_control_telemetry take explicit --log/--run-dir; build_run_log_timeline reads `.ralph/run_log.md` (different artifact).
- Sweep (b): all 9 bare `catch(e){}` in dashboard JS now log (console.error on handler/render paths: __flowReady/__flowActivate/__whyhowActivate/activateOracle; console.debug on expected-failure probes: matchMedia/localStorage×2/ws.close/poll-retry). NEW null-guarded `setTxt()` converts all 8 direct `$(id).textContent` writes (missing markup → console.error ONCE, render continues). node --check clean; 53/53 dashboard tests.
- Ledger: 2 resolution rows appended closing the empty-resolution recurrences; PLUS new failure class `launch_sh_inplace_rewrite_under_live_bash` recorded (opened, causal_status=measured) + gate-landed resolution (item 1's fix).
- dashboard_server.py uncommitted diff TRIAGED: it was the SANDBOX deep-math tab (companion of `docs/sandbox_pontryagin_lie_deepmath_context.md`; author credit-died Jul 7). Verified functionally complete (generic tab wiring, honest MEASURED/DERIVED/ANALOGY tags) → recovery-committed in 763e2bea7 with review_status flag.
## Item 4 — v5.1 errata fold — SUPERSEDED-BY-SIBLING (no edit made; correct outcome)
**Verdict scope: the 9 seal_round1 MINORs vs DRAFT v5/v6; verified by reading v6's frontmatter + spot-checking incorporation, not taking the claim on faith.**
- Since the prompt was written, the v6 synthesizer LANDED (`DRAFT_OPTIMAL_STACK_v6_20260708.md`, feec6e7af + b5771b531 + fold 31ac6b7c1). Its frontmatter states verbatim: "seal_round1_v5_verdict_20260707.md (9 editorial minors — no v5.1 errata file exists; all 9 incorporated here)" and PINS v5 append-only at b241cf466.
- Spot-checks CONFIRM incorporation: MINOR-A1 flag spellings `--persistence-warmup-epochs`/`--persistence-classes` at v6:153-154 + incorporation table v6:532; Q1 fine-τ conjunct at v6:228. v5 file byte-unchanged since its pin commit (git diff empty).
- Editing v5 into a v5.1 NOW would (a) duplicate the fold, (b) break v6's b241cf466 pin on the preserved v5 — the append-only preservation v6 declares. LEFT deliberately; convergence pattern Variant 3 (completed-by-sister).

## Item 5 — orphan triage — DONE
- `modular_theory_deepmath_review_20260707.md`: complete report-only deep-math review (verdict "nothing to build", sources cited, proactive-recall section present). Author = deep-math subagent, credit-died Jul 7. Recovery-committed b47fb4b98 with `review_status: recovery-committed` flag (L81 discipline: recovery verdicts = UNREVIEWED beyond sanity-read).
- `docs/sandbox_pontryagin_lie_deepmath_context.md` + the `tools/dashboard_server.py` SANDBOX tab: ONE body of work (tab links the doc); complete + honestly tagged. Recovery-committed 763e2bea7 with review-status comment.
- Sibling-owned τ-confirm files (`tools/witness_tau_mq_confirm.py`, probe memo, artifact json) + the annulus-metrics diffs: LEFT untouched; sibling landed them itself (f4dcad1a6, e826f1be9) mid-sweep. Working tree clean at sweep end.

## Item 6 — sweep — DONE
- Test subsets: witness_control tests + launch (41) + dashboard ×3 (53) + confound_gates + kernel-fallback (3) = **205 passed, 0 failed** (plus kernel suite 60/60, fused-R 25/25 run under item 2).
- Owned-file lint: launch_witness_run / dashboard_server / metal_grouped_conv_backward / both test files = ruff CLEAN (dashboard driven to zero incl. an fd-leak fix at the 3 spawn sites, 0d373685f). Levelset trainer: F-class clean; ~90 pre-existing style findings across the 8.6k-line file with ZERO in my diff region — LEFT (bulk style churn on the live launch-path trainer in a hardening commit = risk without benefit; the sealed-argv byte-identity + live-sibling context makes minimal blast radius the right call).
- Repo-wide `ruff check src/ tools/` = 7,565 pre-existing findings (mostly RUF100/SIM115 classes) — OUT OF SCOPE for this sweep; noted for a dedicated lint-debt campaign if the operator wants one.
- In-flight-owned failures: none observed (sibling's suites not run by me; sibling landed green per its own commits).

## Commits landed this sweep
1. e28ff371e — item 1 (atomic launch.sh write + class-guards + shebang test alignment)
2. f105ff114 — item 2 (#265 two-layer fix + sibling sweep + 3 tests)
3. 763e2bea7 — item 3(b) sweep + SANDBOX tab recovery
4. b47fb4b98 — item 3 ledger closure (2 resolutions + new launch_sh class opened/gate-landed) + modular-theory recovery + this memo
5. 0d373685f — dashboard lint-to-zero + spawn fd-leak fix
(fc0113215 DAG FEED-09b for #265 was appended by the triality hook/sibling — not mine.)
