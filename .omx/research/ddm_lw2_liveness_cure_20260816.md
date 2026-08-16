# ddm_lw2 — the liveness watcher false alarm: independent diagnosis, cure, and controls

**Arm:** `ddm_lw2_liveness_watcher_cure` · **Task:** #1064 · **Date:** 2026-08-16
**Files owned this turn:** `tools/launch_detached_process.py`, `tools/run_liveness_watcher.py`
**Authority:** local source inspection + executed controls. No score claim. Pointer UNMOVED.

---

## Answer first

MAIN's cure was aimed at the right derivation point but at a **conflation of two different
alerts from two different runs**, and the fix MAIN thought was missing was **already in the tree
since 2026-08-15 — landed but inert**, because its call site passed the wrong argv.

Three findings, all MEASURED:

1. **The "303 s" alert was `pidfile_missing_or_invalid`, not `child_dead`.** MAIN's `launcher/`
   path-drift mechanism is real and correctly identified — it just produced a different alert on
   a different run than the one MAIN attributed it to.
2. **The `child_dead` alert had a separate root cause**: a config with no `success_receipts`, so
   the watcher could not tell a clean rc=0 exit from a silent death. MAIN's proposed cure would
   not have fixed it.
3. **The existing `success_receipts` cure was inert**: `_augment_liveness_success_receipts` was
   called with `cmd` (the raw user command) instead of `effective_cmd` (the safe_run-wrapped
   argv). `--status-receipt` exists only in the latter, so the function returned unchanged on
   every launch since it landed.

Cured both, at the one derivation point, from the launch's own argv. Both control directions
executed: healthy child **rc=0** (no alert), genuinely dead child **rc=1 `child_dead`**.

---

## 1. Independent diagnosis (re-derived from artifacts, not from MAIN's account)

### 1.1 The watcher's own control flow forbids MAIN's attribution

`tools/run_liveness_watcher.py::evaluate` has two disjoint branches:

| Condition | Alert |
|---|---|
| `_read_pid(pid_file)` raises (missing/unparseable) — line 199-208 | `pidfile_missing_or_invalid` |
| pid read OK **and** `_pid_alive(pid)` false — line 209-233 | `child_dead` |

A `pid_file` pointing at a path that does not exist therefore **cannot** produce `child_dead`.
That is the first sign the account merged two events.

### 1.2 The three alerts actually on disk

| run | alert file | reason | wall time |
|---|---|---|---|
| `rt1_r4` | `.omx/tmp/rt1_r4_launch/liveness.alert.json` | `child_dead` (pid 78955) | 11:37:51 |
| `ra2c_a0` | `.omx/tmp/ra2c_a0/liveness.alert.json` | `child_dead` (pid 97187) | 15:36:33 |
| `ra2c_rank4` | `.omx/tmp/ra2c_rank4/liveness.alert.json` | **`pidfile_missing_or_invalid`** | 16:16:04 |

The rank4 alert body names the drifted path verbatim:

```
/Volumes/APDataStore/pact/ddm_ra2c_rank4_20260816/launcher/resource_safe_run_child.pid
```

### 1.3 MAIN's "303 s" is the rank4 alert

`ddm_ra2c_rank4` launch manifest `generated_utc = 2026-08-16T21:11:01Z`; the watcher's own event
receipt `watched_launch_116_liveness.done` records `elapsed_s = 303.340656`, `rc = 1`. The config
sets `warmup_s = 300`, so the watcher waited out its grace and then fired. The run finished
**rc=0** at 21:20:49 (`elapsed_s 587.62`, `status "ok"`, `peak_rss_mib 10480`, `kill_action null`).

MAIN's "6.8 GiB peak / status running" is consistent with reading the live status receipt at
alert time (t≈303 s of a run that finally peaked at 10.2 GiB) — a mid-run sample, not the final
peak. That part of the account holds.

**Why the path drifted — the two-hand-typed-paths mechanism, confirmed:**

| run | launcher `--child-pidfile` (argv) | config `pid_file` (hand-typed) | agree? |
|---|---|---|---|
| `a0` | `…/ddm_ra2c_alpha_ladder_a0_20260816/launcher/resource_safe_run_child.pid` | same | ✅ |
| `rank4` | `…/ddm_ra2c_rank4_20260816/resource_safe_run_child.pid` | `…/ddm_ra2c_rank4_20260816/**launcher/**resource_safe_run_child.pid` | ❌ |

The two runs used **different `--output-dir` layouts** — a0's output dir was a `launcher/`
subdirectory, rank4's was the run dir itself. The config was copied from a0 to rank4 with the
paths swapped, and the `launcher/` component survived the swap. MAIN's mechanism is correct.

### 1.4 The `child_dead` alerts have a *different* root cause

`ra2c_a0`'s `pid_file` was **right**. Its child 97187 really did exit — cleanly:

```
"exit": 0, "status": "ok", "elapsed_s": 594.207, "generated_utc": "2026-08-16T20:36:24Z"
```

The watcher published `child_dead` at 20:36:33 — **9 s after a successful completion**. Cause:
`ddm_ra2c_a0_liveness.json` declares no `success_receipts`, so in `evaluate` the `clean_exit`
branch (lines 210-222) is empty and in `run` the `success_settle_s` grace (lines 302-314) is
gated off. The rc=0 safe_run receipt sat unread in the same directory as the pidfile.

**Correction to MAIN, stated plainly:** one mechanism (`launcher/` drift) was generalized to
cover two alerts with two causes. Fixing only `pid_file` would have left every `child_dead`-on-
clean-exit alarm live.

### 1.5 The prior #1064 fix was landed but inert — the actual reason a0 still alarmed

`8c9536b19f` (2026-08-15 19:00:54) added `_augment_liveness_success_receipts`, whose docstring
names this exact false-positive class. Both 2026-08-16 runs ran at shas **containing** that
commit (`9a842d98de`, `17a18befb0` — both verified descendants). It never fired:

* no `liveness_config_effective.json` exists in either run's `watchers/` directory;
* both launch manifests record `watchers[].config_path` as the **original** hand-typed config.

Executed against the real a0 manifest:

```
raw cmd        has --status-receipt: False
effective_argv has --status-receipt: True
cmd (as called today)      -> augmented=False
effective_cmd (the fix)    -> augmented=True
```

`--status-receipt` is injected by `_derive_resource_budget`, which returns `effective_cmd`; the
call site at line 880 passed `cmd`. Present at that line in all three shas including HEAD.
Genus: **built-but-orphaned / the point-fix that looks like a fix** — correct function, wrong
argument, zero runtime effect, and a docstring that read as if the class were closed.

---

## 2. The cure, at the derivation point

`tools/launch_detached_process.py`:

* `_safe_run_wrapper_flag(effective_cmd, flag)` — reads a launcher-injected safe_run flag,
  scanning **only** the wrapper's own flag region (before the first bare `--`), so a same-named
  flag inside the wrapped user command can never be mistaken for one the launcher owns.
* `_derive_liveness_config(...)` replaces `_augment_liveness_success_receipts` and derives
  **both** run-specific values from the argv the child will actually receive:
  `pid_file` ← `--child-pidfile`, `success_receipts` ← `--status-receipt`.
* Call site fixed: `cmd` → `effective_cmd` (the inert-argument bug), with the reason in a
  comment at the line so it cannot be re-introduced by inspection.
* The derivation record lands in the launch manifest as `liveness_config_derivation`.

**Fail-closed policy — explicit, because silence here is the defect:**

| case | behavior |
|---|---|
| derived == declared | `argv_confirmed`; nothing rewritten |
| derived != declared | `argv_superseded`; **argv wins**, declared value preserved in the record, and a JSON supersession line printed to **stderr at launch** |
| no safe_run wrapper, or flag absent | `config_declared`; fall back to the hand-typed value. The launcher does **not** invent a path it does not own. `pid_file` stays mandatory in the watcher's `load_config`, so absence still fails closed there. |
| launcher argv and `resource_budget` disagree about the same path | **`LaunchRefusal` rc=10** — the launcher's two accounts of one path have diverged |

The caller's config file is never mutated; the effective config is written to
`<out>/watchers/liveness_config_effective.json`.

Deriving from argv rather than from `resource_budget` is deliberate: the argv is the thing
safe_run obeys. The budget record is cross-checked against it, not trusted over it.

**No alarm was weakened.** The wrong *input* was fixed; every alert branch is untouched.
`tools/run_liveness_watcher.py` required no change — it was reporting its inputs faithfully.

---

## 3. Controls — executed, both directions

Real watcher binary, real pids, hermetic root `/tmp/lw2ctl.EF89TZ`.

| control | setup | expected | **measured** |
|---|---|---|---|
| C1-rank4 | replay real rank4 manifest + drifted config | pid_file corrected | `argv_superseded` → `…/resource_safe_run_child.pid`, **`pid_file_exists: true`**; receipt `argv_derived` |
| C1-a0 | replay real a0 manifest + correct config | pid_file left alone | `argv_confirmed`; receipt `argv_derived` |
| **C2** | **healthy child, exits rc=0, success receipt `exit:0`** | **no alarm** | **rc=0**, `clean_exit`, **no alert file**, 5.1 s |
| **C3** | **genuinely dead child, no success receipt** | **alarm** | **rc=1**, `child_dead`, 0.0 s |
| C4 | dead child, receipt says `status:killed exit:137` | alarm | **rc=1**, `child_dead` after 3 s settle |

C2 vs C3 is the both-directions pair: the watcher is silent on health and still loud on death.
C4 additionally shows a success receipt cannot be abused to buy silence for a real failure.

C1 is the sharpest evidence for the diagnosis in §1: **the same cure returns different verdicts
on the two runs** (`argv_superseded` for rank4, `argv_confirmed` for a0) precisely because the
two runs had different faults — which is what MAIN's single-mechanism account predicted wrongly.

**Regression:** `ruff` clean; 62 passed across
`test_watched_launch_hardening` / `test_sigurg_kill_class_guard` /
`test_fire_watched_continuation` / `test_check_bl1_background_launcher_rc` /
`test_no_silent_failure_launch_hardening`.

---

## 4. Second half of #1064 — the monitor's receipt-namespace parsing is NOT a defect

`verdict_scope`: `tools/codex_arm_watch.py::format_events` / `_snapshot`, against the three real
receipts written by the 2026-08-16 launches. Executed, not reasoned:

```
globbed by _snapshot? {'watched_launch_116_liveness.done': True,
                       'watched_launch_114_liveness.done': True, 'ra2c_rank4.done': True}
EMIT: ARM ra2c_rank4 FINISHED rc=0 elapsed=587 launch_counter=116 … manifest=…rank4/launch_manifest.json
EMIT: ARM watched_launch_114_liveness ALERT rc=1 elapsed=603 launch_counter=114 … manifest=…a0/launcher/launch_manifest.json
EMIT: ARM watched_launch_116_liveness ALERT rc=1 elapsed=303 launch_counter=116 … manifest=…rank4/launch_manifest.json
```

Evidence it is sound:

1. No receipt is dropped — watcher receipts and arm receipts are both globbed.
2. `name.rsplit(".", 1)[0]` reproduces the payload's own `receipt_name` field **exactly** for all
   three, including the `watched_launch_<counter>_<label>` form. No mis-split. Dots are legal in
   `--done-receipt` names and `rsplit(…, 1)` handles them correctly.
3. rc classification is right in both directions (rc=0 → FINISHED, rc=1 → ALERT).
4. Delivery is confirmed: `.consumed.json` markers were written for both alerts in the same
   minute they were published.
5. The independently derived elapsed times (603 s for a0, 303 s for rank4) match §1 exactly.

**Residue, reported not landed:** a watcher receipt names the *watcher*
(`watched_launch_116_liveness`), not the *arm*, so a reader must open the printed manifest path
to learn which arm alarmed. This is an ergonomics gap, not silence — the manifest path is in the
line. I did not fix it because (a) the fix belongs in `tools/codex_arm_watch.py`, which a sister
arm owns, and (b) adding an arm-name field to the receipt from my side would be **inert** until
the monitor reads it — the #417 unconsumed-is-fake trap, which is the very genus this arm just
cured. Fixing it requires one owner holding both files.

---

## 5. Residue I did not close

* **The a0/rank4 configs themselves are unchanged.** The cure makes their `pid_file` and
  `success_receipts` no longer load-bearing at launch, but `.omx/research/ddm_ra2c_*_liveness.json`
  still carry the stale `launcher/` component. Harmless now (argv supersedes, loudly), and I left
  them so the next launch's stderr supersession line is observable evidence the cure fires.
* **No STRICT preflight gate landed** for the "derivation helper called with the unwrapped argv"
  class. The two-landing rule wants one; a static gate that proves an argv-scanning helper is fed
  the wrapped argv is not obviously expressible without over-fitting to this call site. Owed.
* **3 pre-existing failures in `tools/tests/test_modal_endpoint_close.py`** — A/B'd against
  pristine HEAD with my file reverted: **identical 3 failures**, so not mine. Cause is a stale
  active claim (`lane_ac1_test`, age 58.84 h) in `.omx/state/active_lane_dispatch_claims.md`, a
  Modal-ledger surface this arm is forbidden to touch. Flagged for its owner.
* **Configs remain hand-written** for the non-`--derive-resource-budgets` path (`mode:
  child_owned`), where the launcher owns no pidfile and cannot derive one. That path keeps the
  original drift exposure by construction; refusing there would have been a new false-refusal
  class, so it falls back explicitly and records `config_declared`.

---

## 6. Measurement

Watcher on a healthy rc=0 child: **rc=0**, `clean_exit`, no alert written.
Watcher on a genuinely dead child: **rc=1**, `child_dead`.
Replayed rank4 derivation: `pid_file` `argv_superseded` to a path that **exists** (`true`).
Pointer UNMOVED; `score_claim=false`.
