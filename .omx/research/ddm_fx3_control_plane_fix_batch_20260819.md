# ddm_fx3 — control-plane fix batch (tasks #1121, #1138-F7, #905)

Arm: `ddm_fx3`. Date: 2026-08-19. Operator directive 2026-08-19: *"Fix the most
necessary using opus subagents."* Three live-hazard control-plane rows, each
verified at source before any edit per the charter-recall clause.

**Headline verdicts**

| row | task | premise | verdict |
|---|---|---|---|
| 1 | #1121 orphaned sleep waiters | **LANDING 1 STALE** (shipped 08-18), **LANDING 2 LIVE AND OWED** | **FIXED** — structural guard landed |
| 2 | #1138-F7 check_330: "2 Modal harvesters missing ledger writes" | **HALF STALE** — 1 real harvester, 1 detector false positive | **FIXED** (check_330: 2 → 0) |
| 3 | #905 hook blocker = missing code branch | **VERIFIED LIVE** at `tools/preflight_hook.py:788-807` | **FIXED** (third `--scope dev` branch + timeout) |

Nothing in this batch fired Modal, moved the pointer, or made a score claim.

---

## §1 — Row 1: task #1121, orphaned sleep waiters (two-landing)

### Premise check: landing 1 already shipped; landing 2 was the real hazard

**Landing 1 exists.** Commit `5d4b1818f5` (2026-08-18) added `WAITER_DISCIPLINE`
at `src/tac/subagent_contract.py:145-162`, wired into `standard_contract()`.
Its commit message claims the cure is *"now structural in every composed
subagent prompt rather than volitional."*

**That claim is wrong, and the gap is exactly the hazard.** A clause injected
into a prompt is text an arm can ignore. I verified no mechanical enforcement
existed:

* `check_no_orphan_prone_daemon_launch` (Catalog #389, STRICT) requires `nohup`
  AND `bash -c` AND `| tee` AND backgrounding on one logical line, and its own
  comment **explicitly exempts this shape**: *"A short one-shot backgrounded
  command or a `run_in_background` Bash-tool invocation does NOT match"*
  (`src/tac/preflight.py:81722`). Measured: 0 violations, and 0 coverage of #1121.
* `tools/launch_guard_hook.py` fired only on `nohup`+`disown` and on known
  long-runners — never on a bare backgrounded `while … sleep … done`.

So: **landing 1 = behavioural, done. Landing 2 = structural, owed.** That is
precisely what the task's `in_progress` / two-landing status encoded.

**The class is not extinct — proof measured this session.** A live orphan on
this box right now:

```
pid 34844  ppid 34843  elapsed 07-04:36  bash …/ddm_js2b_20260812/js2b_done_poller.sh
pid 34843  ppid 1      (detached fork+setsid shim)
```

It polls for `.omx/state/codex_arms/…​.done` — **a directory that does not
exist**, so it can never fire. Its alert log is 0 bytes, mtime Aug 12. It has
spun for 7 days 4 hours, orphaned to init, invisible to `codex_arm_queue.py
status`. It is harmless only by routing luck: it is artifact-bound (the RULE 2
cure) and writes to an external volume rather than `.omx/tmp/codex_runs/`. Had
its author written the receipt into the notify directory, it would be a spurious
MAIN re-invoker. **Not killed** — provably inert, and another arm's process is
MAIN's call. One-line disposal if wanted: `kill 34843 34844`.

**Repeat genus, not a novel finding:** 2026-04-26 (`while pgrep -f train_distill`
matched its own `bash -c` argv, looped forever, GPU idle) → 2026-07-17
(`launcher_chain_death_postmortem`, whose next_action was literally *"on v2 exit:
waiter fires v3"* — an operator-blessed latent actuator) → 2026-08-12 (the live
orphan above) → 2026-08-18 (ddm_iv1). Roughly one instance every few weeks.

### Root cause, stated precisely

**No waiter in the repo checks PARENT liveness.** Every liveness check is
*subject* liveness. The re-decide-at-fire-time gate exists in exactly two files
(`tools/supervise_ddm_r1c_rung1.py:419-424`, `tools/supervise_ddm_b4s_burn4.py:948`),
both one-off arm supervisors — **there is no shared waiter helper for arms to
call**, which is why every arm hand-rolls one and the bug re-lands.

And the incident's four zero-signal notifications came from **no repo file at
all**: they were ad-hoc Bash `run_in_background` shells typed inline, which the
harness re-invokes on exit. There is no artifact to patch — only the typing
moment. That is why the guard is the right surface.

### Fix (landing 2): structural refusal at the typing moment

Extended `tools/launch_guard_hook.py` (the `PreToolUse` Bash hook) with two
detectors on the pure, unit-tested `decide()` surface, placed with the other
orphan/kill-class blocks and **above** the trainer safe-tokens — same lesson as
the 2026-08-04 codex-spawn ordering bug: a trainer token must not buy a bypass
of an orphan violation.

* **`_is_latent_actuator_waiter`** (RULE 1, the harm-bearing half) — a
  process-table poll loop that **launches** when it drains. Checked first, and
  **not** gated on `run_in_background`: the harm is the duplicate launch over an
  adjudicated receipt, not the notification.
* **`_is_orphan_prone_waiter`** (RULE 2, the noise half) — a *backgrounded* wait
  bound to a clock or the process table.

**The distinguishing test is the PREDICATE**, in the contract's own words:
*"Wait on an artifact's existence, never on a clock."* An artifact-bound wait is
the CORRECT pattern and stays allowed. Only waits that **can fire when nothing
happened** are refused.

### Not breaking the live apparatus — the load-bearing check

The fleet-watcher chain is running on this box right now (`pid 75034`
`launch_detached_process.py _supervise` → `pid 75035` `codex_arm_watch.py`,
`_watcher.alive` reading `channel=fifo`). A false positive here severs MAIN's
notifications. Executed a 16-case matrix, **16/16 correct**:

| allowed (must not break) | blocked (the bug) |
|---|---|
| `until [ -f "$DONE" ]; do sleep 30; done` | `while pgrep -f job; do sleep 30; done` |
| `while [ ! -f …/x.done ]; do sleep 30; done` | `while kill -0 12345; do sleep 5; done` |
| `launch_detached_process.py --done-receipt …` | `sleep 600` (backgrounded) |
| `bash -c "exec … codex_arm_watch.py"` | `sleep 300; echo done` (backgrounded) |
| `codex_arm_queue.py saturate --spawn` | **the verbatim iv1 actuator** |
| `grep -rn "while true; do sleep 60; done"` | `until ! pgrep jobA; …; bash launch_next.sh` |
| artifact-wait-then-act in the foreground | |
| `while read l; do …; done` (no sleep) | |

Also verified as a **real subprocess** on the actual PreToolUse path: the iv1
actuator returns `permissionDecision: deny` with an actionable message; the
canonical artifact waiter passes silently; malformed stdin still exits **rc=0**
with no output (the fail-open contract is intact).

**Explicitly out of class and untouched:** the 205 `scripts/remote_*.sh`
heartbeat subshells (`( while true; do …; sleep 60; done ) &`). They run on
rented GPU hosts, are paired with `trap … EXIT` by
`tools/canonical_lane_template.py:69-75`, and have no channel into the session.
Confusing that population with #1121 would have been the expensive mistake.

### Class guard

The guard **is** the class fix (landing 1 was the instance/behavioural half).
9 new tests in `src/tac/tests/test_launch_guard_hook.py`, all executed; the
first four pin the *allow* side, because that is what breaks the box:
canonical artifact waiter · canonical launcher + live watcher chain · inspection
commands (`grep`/`echo` about waiters) · foreground and non-sleep loops · then
process-table waits · bare clock waits · **the measured iv1 actuator verbatim**
· the explicit override · and trainer tokens not bypassing the block.
**Suite: 44 passed** (9 new, 35 pre-existing, no regressions).

Ladder placement: **ORPHANED-CURE family** — landing 1 was a cure with no
consumption gate and no named enforcement surface, so it could be bypassed by
simply not reading it. Meta-bug **M3 (warn-only purgatory)** in its purest form:
a protection that shipped and was described as structural while remaining
volitional. Sister of **M1** — the class-fix stopped at the prompt surface.

### CLASS-POPULATION line (meta-bug M1)

Swept, not assumed. Across `tools/ scripts/ .claude/` (worktrees excluded):

| class | count |
|---|---|
| raw detach-primitive sites | **69 sites / 38 files** |
| local sites where the spawned thing IS a sleep/poll waiter | **18 files** |
| …of those, able to re-invoke MAIN | **3** |
| …of those, firing on a clock/pid rather than a completion artifact | **1** |
| detach primitives carrying non-waiter payloads | 14 files |
| docstrings that TEACH the `nohup` pattern | 7 files |
| remote-host heartbeat subshells (**out of class**) | 205 sites |
| live orphaned waiters observed on this box | **1** (7d 4h) |

The guard covers the **typing moment** for all future hand-rolled waiters
(Channel C, the vector that caused the measured incident). It does **not**
retroactively rewrite the one in-tree offender — see debt below.

### Named debt (row 1)

1. **`tools/codex_companion_spawn.sh:96-100` is the one in-tree offender.** It
   runs `while kill -0 "$node_pid"; do sleep 5; done` and then writes
   `rc=unknown_detached` into `.omx/tmp/codex_runs/*.done` — the MAIN-notify
   channel. Its own comment concedes the defect: *"detached watcher cannot
   wait(2) for non-child pid"*. It cannot distinguish "arm finished" from "arm
   was reaped", so it is a zero-information re-invocation by construction. **Not
   fixed here:** it is a live codex spawn path, codex is walled until Aug 20, and
   I cannot exercise it end-to-end. Owner: whoever re-opens the codex lane.
2. **No shared waiter helper exists.** The correct re-decide-at-fire-time gate
   is implemented twice, in two one-off supervisors. Until it is a callable
   helper, arms will keep hand-rolling. The guard now refuses the wrong shapes
   but does not yet *hand them the right one* beyond a message.
3. **Task-number ambiguity.** `#1121` is overloaded: the graph-memory node
   `ref:#1121` resolves to **"DP1 PROCEDURAL TRAINER BUILD"**
   (`.omx/research/parent_tasklist_delegation_stale_close_batch_survey_20260521.md:94`),
   stale-closed. I proceeded on the charter's **content** (waiter discipline),
   never the bare id — per [[m89]]. Flagging so the number is not trusted alone.

---

## §2 — Row 2: check_330, the two "missing ledger writes" (task #1138, F7 genus)

### Premise check: HALF STALE, and the stale half is the more interesting one

The rv14f memo (`.omx/research/ddm_rv14f_rv13_fix_batch_20260819.md:306`) queued
this as *"check_330: 2 unmirrored Modal harvesters (F7 genus)"*. Reproduced red
at the start of this arm:

```
check_330 offenders: 2
  experiments/modal_ot_offset_n600_gate.py:33
  tools/modal_dispatch.py:216
```

Reading both at source: **only one of the two is a harvester.**

**`experiments/modal_ot_offset_n600_gate.py:33` — DETECTOR FALSE POSITIVE.**
Line 33 sits inside the module **docstring** (it closes at line 34). The file
dispatches with `.spawn()` and saves the call_id; it never harvests in-process.
`_check_330_line_is_comment_or_literal` skips a line that *starts* with a quote
or contains ` `` `, but line 33 begins with `print(`, so a docstring
continuation line reads as code. The sibling occurrence at line 216 — the same
text inside an f-string — *is* correctly skipped because it starts with `f"`.
So the detector's own two hits on one file disagreed with each other.

Ladder placement: **WRONG-OBJECT family** (name/shape treated as identity when
content is the identity — here a raw text line treated as code). Sister:
SILENT-INSTRUMENT, since a line-level heuristic cannot distinguish prose from
executable code and reports both identically.

**`tools/modal_dispatch.py:216` — REAL, and exactly the bug class #330 exists
for.** `_live_call_state` is reached by `modal_dispatch status --live`. It
observed terminal provider state and returned a bare string:

| observation | returned | canonical ledger before this fix |
|---|---|---|
| `.get()` returns | `"completed"` | stuck at `dispatched` |
| `OutputExpired` | `"expired(>24h)"` | stuck at `dispatched` |
| `TimeoutError` | `"running"` | `dispatched` — correct, nonterminal |

Per the Catalog #330 text this is precisely *"a later concurrent harvester can
observe harvested / failed / stale / function_timeout and leave the call_id
stuck at dispatched."* Secondary defect at the same line: `.get()` retrieved the
full result payload and **bound it to nothing** — the measure-and-discard
signature, with a scalar status string kept in place of the structured signal.
Both defects have the same cure, which is why the fix is one edit.

### Fixes

**(a) `tools/modal_dispatch.py` — mirror terminal observations.** New
`_mirror_terminal_call_state()` delegating to
`tac.deploy.modal.harvest_outcomes.append_terminal_call_id_ledger_event` — the
identical thin-wrapper shape `tools/harvest_modal_calls.py:515` and
`tools/parallel_harvest_actuator.py:432` already use. Replicated, not
hand-rolled, so classification stays in one place.

Three precision properties, each with an executed test:

* **rc beats the fallback.** A returned payload with `rc=3` records `failed`,
  not `harvested`. The weaker "the provider returned a result" claim is applied
  **only** when rc-based classification yields `None`. No rc is ever fabricated.
* **Timeouts stay in-flight.** A bounded poll timeout is not terminalized.
* **Bounded projection.** `fire` wraps an arbitrary launcher command, so the
  result may carry anything. `_ledger_signal_fields()` projects onto the ten
  fields the helper actually reads, so a stranger's artifact blob never lands in
  the small shared append-only ledger JSONL. Artifact persistence remains
  `cmd_harvest`'s job. (Found in review pass 2, not pass 1.)

Ledger faults print a WARN and never break the read-only status command —
loud, not swallowed.

**(b) `experiments/modal_ot_offset_n600_gate.py` — stop teaching the bug class.**
Both the docstring and the operator-facing `print` advertised a raw
`FunctionCall.get` poll. An operator who pasted that command would perform
exactly the ledger-skipping harvest the gate refuses. Both now name the
canonical harvester (`tools/harvest_modal_calls.py --from-ledger --call-id
<id> --execute`; flags verified against `--help`, never invented). This removes
the false positive **and** an anti-pattern vector — it is not a backtick dodge.

**Result: check_330 offenders 2 → 0.**

### Class guard

check_330 already exists and is STRICT; the guard owed here is a **regression
test**, landed as `tools/tests/test_modal_dispatch_call_id_ledger_mirror.py`
(10 tests, all executed): rc0→harvested · rc3→failed · unclassifiable→harvested
· timeout→stays `dispatched` · expiry→stale · unknown exception→not terminalized
· ledger fault→status survives and warns · repeated polls→one terminal row ·
payload not leaked into the ledger · projection keeps only helper-read fields.

### CLASS-POPULATION line (meta-bug M1)

**Measured: 29 files** under `experiments/ tools/ scripts/ src/tac/` contain
`FunctionCall.from_id` (tests, `experiments/results/`, and `_intake_` excluded).
check_330 currently flags **0**. The population was swept, not assumed.

### Named debt (NOT edited — file boundary)

`src/tac/preflight.py` belongs to `ddm_sp2` this session, so the following are
recorded rather than applied:

1. **check_330 is not string-literal aware.** `_check_330_line_is_comment_or_literal`
   (`src/tac/preflight.py:62234`) is a line-level lexical guess. Cure: walk the
   AST and skip any hit inside a `Str`/docstring node. Population: docstrings
   that paste a Modal poll — 1 found and fixed at the offender surface today,
   but the detector will re-fire on the next one.
2. **The detection window is asymmetric: `-20/+180` lines**
   (`src/tac/preflight.py:62289`). A compliant file whose mirroring helper is
   defined *above* its `.get()` still reads as a violation. I hit this directly:
   the first version of the fix was correct and still flagged. Cure: widen the
   backward window, or resolve the enclosing function's call graph.

**Honest note on the workaround I used for (2):** rather than satisfy the gate
with prose, I moved `_mirror_terminal_call_state` *below* `_live_call_state` so
the detector's forward window sees the real `append_terminal_call_id_ledger_event(`
call. Semantically neutral (module-level defs resolve at call time; verified by
AST + the passing suite) and it reads better — probe first, plumbing after. But
it is a workaround for debt item 2, and should not be mistaken for the cure.

---

## §3 — Row 3: task #905, the missing hook branch

### Premise check: VERIFIED LIVE, unchanged since the 2026-08-16 diagnosis

Diagnosis: `.omx/research/ddm_rg2_red_gate_triage_20260816.md` §1a + §6 blocker 0,
independently re-derived by `ddm_rd1g` (`.omx/state/subagent_progress.jsonl`
lines 10560, 10564). Re-verified at source today: `_preflight_command()` was a
two-arm `if/else` on one env var. `grep -n '"--scope"' tools/preflight_hook.py`
returned exactly one hit, value `all`. Clean working tree; last commit touching
the file (`52a997981d`) was unrelated.

The re-diagnosis is right, and it explains the whole shape of #905:

| `PREFLIGHT_FULL` | command | gates | timeout |
|---|---|---|---|
| unset | `--no-codebase --acknowledge-empty-scope` | **0 of 27** | 30 s |
| `1` | `--scope all` | exhaustive release/custody sweep | 30 s |
| — | **`--scope dev`** | **NO BRANCH EXISTED** | — |

`tac.preflight`'s own `--scope` already *defaults* to `dev`, and layer 2 of the
hook's own module docstring advertised the bounded developer stack. The hook
simply never selected it. So "turn the hook on" meant running the release sweep
on every commit — gates that scan paths outside the repo and fire on untracked
files. **That is why #905 read as "the RED gates block us": the only reachable
non-empty mode was the wrong one.** Ladder: this is the M2 shape — a protection
suppressed by a mis-attributed constraint.

### Fix

`_preflight_scope() -> "none" | "dev" | "all"`, consumed by both
`_preflight_command()` and `_preflight_timeout_seconds()` (the mirrored half of
the gap — the dev scope had no timeout branch either).

**Non-weakening, by construction:**

* The **default is byte-identical** to before (0 gates, 30 s). No commit that
  passed before blocks now. The new mode is **opt-in** via `PREFLIGHT_SCOPE=dev`.
* `PREFLIGHT_FULL=1` keeps precedence, so every existing caller, runbook, and
  test is unchanged.
* The dev branch **deliberately omits `--acknowledge-empty-scope`**. That waiver
  is earned by the 0-gate mode alone. Letting it leak into dev mode would mean a
  dev scope examining 0 gates prints PASSED instead of refusing rc=3 —
  re-importing vacuity-indistinguishable-from-PASS through the new branch.
* An unrecognized `PREFLIGHT_SCOPE` falls back to the default; it never widens
  scope silently and never raises.

**Dev timeout = 60 s, derived not guessed.** MEASURED (ddm_rg2): 22.7 s warm /
24.3 s cold. The inherited 30 s bound leaves 19–24 % headroom, so one slow-gate
regression fails commits on the **clock** rather than on a finding — and to a
committer who sees only "hook failed", a timeout is indistinguishable from a
real refusal. Bound = 2× the cold measurement (48.6 s), rounded up to a whole
minute for legibility, matching `_HOOK_FIXED_OVERHEAD_SECONDS`'s convention.
`effective_hook_wall_clock_bound_seconds()` therefore widens by exactly 30 s in
dev mode, so the commit serializer's lock patience tracks it automatically —
that is why that function exists, and a test pins the relationship.

### Executed dry-runs (the charter's requirement)

1. **Default mode, the load-bearing commit path:** `tools/preflight_hook.py` →
   **rc=0**, and the coverage line now names an actionable env var
   (`PREFLIGHT_SCOPE=dev git commit ...`) instead of advising a mode the hook
   could not enter.
2. **The exact command the new branch emits:** `.venv/bin/python -m tac.preflight
   --scope dev` → **rc=1**, `PREFLIGHT FAILED: 7 of 25 declared developer gates
   are RED`. The invocation is valid and **non-vacuous** — it reports a real
   denominator.

### Class guard

8 new tests in `src/tac/tests/test_preflight_hook.py`: dev branch exists · dev
never acknowledges empty scope · dev timeout clears 2× cold · serializer bound
widens by 30 s · `PREFLIGHT_FULL` still wins · unknown scope falls back · scope
`all` matches `PREFLIGHT_FULL` · explicit timeout override still wins. Also
added `delenv("PREFLIGHT_SCOPE")` to the pre-existing default-mode test, which
otherwise silently depended on ambient shell state (STALENESS family, at the
test surface).

**Full suite: 91 passed** (`test_preflight_hook.py` +
`test_preflight_hook_heavy_import_scope.py`).

### CLASS-POPULATION line (meta-bug M1)

The class is "a mode the tool's own docstring advertises but no branch can
reach." **Measured within this hook: 1 of 3 modes was unreachable** — swept by
enumerating the full env matrix (10 combinations, all executed) and confirming
each documented mode now maps to an emitted command. A repo-wide sweep for
docstring-advertised-but-unreachable modes in other tools is **NOT** done; that
is a genuine open population, waived here as out of this batch's scope and
recorded below.

### What #905 still needs (honest scope boundary)

This fixes the **missing branch**, which was the diagnosed blocker. It does
**not** flip the hook on — 7 of 25 dev gates are RED and each must be
adjudicated on its own merits (preflight's own output says so: *"Curing a gate
by widening its exemptions is a weakening, not a fix"*). The precise list, for
whoever takes that next:

| red dev gate | count |
|---|---|
| `check_state_writers_strict_load_for_mutating_path` | 1 |
| `check_authoritative_tag_requires_custody_metadata` | 1 |
| `check_codebase_drift` | — |
| `check_dispatch_claim_helper_present` | — |
| `check_subagent_landing_has_solver_wire_in` | 124 memos |
| `check_lane_pre_registered_before_work_starts` | 3 |
| `check_substrate_score_aware_losses_use_canonical_scorer_contract` | 5 |

Flipping the hook on is now a one-env-var change once those are green, instead
of "run the exhaustive release sweep on every commit."

### Ledger note

Task **#905 is absent from the repo canonical ledger**
(`.omx/state/canonical_task_status.jsonl`); it lives only in the harness
TaskList. `.omx/research/ddm_oq1_drain_dispositions_20260817.json` already
records this with owner MAIN. I cannot close the row from here — MAIN owns the
harness-ledger reconciliation. This is the [[m89]] task-ledger split, still live.

---

## Pre-existing reds observed, NOT caused by this batch

Reported rather than absorbed, per the rv14f precedent — and **proved** rather
than asserted, because "it was already broken" is the easiest claim in the world
to make and the hardest to trust.

### The cross-suite interaction, isolated on a pristine HEAD worktree

Running my touched suites *together* produced 7 failures that each passed when
run alone. That is exactly the shape of a batch quietly breaking something, so I
did not wave it through. Bisected to a **pre-existing order dependency**:
`src/tac/tests/test_preflight_hook_heavy_import_scope.py` pollutes
`tools/tests/test_modal_endpoint_close.py` (the latter then refuses with
`REFUSED_DUAL_LEDGER` instead of `DRY_RUN_VALIDATED`). I touched **neither** file.

Proof, on a clean `git worktree` checked out at HEAD (`tac` import verified to
resolve *inside* the worktree, per the shared-venv hijack rule), running the
identical chain:

| tree | failures |
|---|---|
| pristine HEAD | **8 failed**, 157 passed |
| my working tree | **7 failed**, 185 passed |

The difference is exactly one test:
`test_check_330_live_repo_has_no_unmirrored_modal_harvesters` — **red on HEAD,
green in my tree.** Every one of the other 7 reproduces identically on HEAD.
This batch strictly *reduced* the failure count, by precisely the failure it
targeted, and introduced none. Worktree removed after the measurement.

**Owed to whoever picks it up:** the order dependency is a real STALENESS-family
instrument defect (a suite whose verdict depends on what ran before it). Not
mine to fix in a control-plane batch, but it will keep making honest batches
look guilty until someone isolates it.

### Other pre-existing reds

1. `test_check_245_modal_call_id_ledger_registration::test_check_245_live_repo_has_no_unregistered_modal_spawn_sites`
   — 1 violation at `src/tac/canonical_anti_patterns/pattern_matcher.py:678`.
   That file is **unmodified** in my working tree, so the violation is on HEAD.
   Sister genus to #330; not mine.
2. Two `SIM103` lints in `experiments/modal_ot_offset_n600_gate.py`
   (`_ignore_heavy`, lines 80 and 93) — confirmed present on the HEAD copy of
   the file, in code I did not touch. Not F-class; left alone rather than
   absorbing unrelated debt into a control-plane batch.

## Files changed

* `tools/launch_guard_hook.py` — two #1121 waiter detectors + block messages (row 1)
* `src/tac/tests/test_launch_guard_hook.py` — 9 new tests (row 1)
* `tools/modal_dispatch.py` — ledger mirroring + bounded projection (row 2)
* `experiments/modal_ot_offset_n600_gate.py` — canonical harvester in docs + print (row 2)
* `tools/tests/test_modal_dispatch_call_id_ledger_mirror.py` — NEW, 10 tests (row 2)
* `tools/preflight_hook.py` — third `--scope dev` branch + dev timeout (row 3)
* `src/tac/tests/test_preflight_hook.py` — 8 new tests + 1 env-hygiene fix (row 3)

## Test totals (all executed, this session)

| suite | result |
|---|---|
| `test_launch_guard_hook.py` | 44 passed (9 new) |
| `test_preflight_hook.py` + heavy-import-scope | 91 passed (8 new) |
| `test_modal_dispatch_call_id_ledger_mirror.py` | 10 passed (NEW) |
| `test_check_330…` + `test_modal_endpoint_close` + hook suite | 116 passed |
| `check_330` live repo | **2 → 0 offenders** |
| `check_389` live repo | 0 violations (unchanged) |

## Ladder summary

| row | bug | class gate | family | meta-bug |
|---|---|---|---|---|
| 1 | prompt-only waiter cure | `launch_guard_hook` detectors + 9 tests | ORPHANED-CURE | M3 (shipped-but-volitional), M1 |
| 2 | status probe left ledger at `dispatched` | check_330 (existing) + 10 regression tests | ORPHANED-CURE / WRONG-OBJECT (the false positive) | M1 |
| 3 | unreachable hook mode | 8 tests pinning all three modes | STALENESS (mis-attributed constraint) | M2 |

Common thread across all three: **a protection that exists but cannot be
reached** — a clause with no enforcement surface, a ledger helper the probe
never called, a scope the hook could not select. Each looked green from the
outside.
