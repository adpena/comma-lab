# ddm_ad1 — three two-landing cures in the Modal fire / harvest / kill chain

**Date:** 2026-08-21 · **Arm:** ddm_ad1 · **Scope:** apparatus debt only. No dispatches, no
launches, no edits under `upstream/`, `submissions/exact_current/`, or the frozen gen6 packet.
**Pointer:** UNMOVED. This arm produced no score row and claims none — it is MEANS.

Each item is (a) a fix and (b) a class-protecting guard whose control was EXECUTED in both
directions. A guard that has never been observed to fire is the #1086 bug class, so every
control below is a command that ran, not a claim that it would.

| # | Item | Verdict | Fix | Guard |
|---|---|---|---|---|
| 1 | fire tool self-defeats `require_active` | **FIXED** | `fe5aa2941b` | `fe5aa2941b` (11 tests) |
| 2 | Modal harvest writes Python repr bytes | **PREMISE PARTLY STALE → gap closed** | `8c4e7abb2b` | `8c4e7abb2b` (5 new tests) |
| 3 | kill doesn't reach the tree | **FIXED** | `a11ff262e5` | `a11ff262e5` (16 tests) |

---

## ITEM 1 — the fire tool terminal-closed the claim its own dispatch was about to consume

### Defect, reproduced from the receipt

`tools/fire_modal_auth_eval.py` stage 4 auto-closes "an active Modal claim while the call-id
ledger has ZERO live rows" — failure F4, a phantom left by a call that finished. A claim
**pre-staged for the dispatch one stage away satisfies that trigger by construction**: it is
active, and its call does not exist yet.

Measured, rc2 T4 fire, 2026-08-20 —
`/Volumes/APDataStore/pact/ddm_rc2/t4_row_r1/FIRE_REFUSED.json`:

```
stage4_claims.closed  = [{lane_ddm_rc2_composed_cuda_20260820,
                          modal:ddm_rc2_composed_cuda_r1, rc 0}]
stage5_dispatch_argv  = [... --lane-id lane_ddm_rc2_composed_cuda_20260820
                             --instance-job-id modal:ddm_rc2_composed_cuda_r1
                             --claim-policy require_active]
entrypoint_refusal_lines[0] =
  "FATAL: ... --claim-policy require_active could not find an active lane claim:
   newest matching claim is terminal: ... status=stale_superseded_reconciled_no_live_call"
refusal_rc = 5   ("dispatch produced no spawn record — the fire DID NOT take")
```

Same lane, same job, same run. The tool disarmed its own guard, and the standing workaround
was `--claim-policy open` — i.e. turning the guard off.

### Fix

`reconcile_claims()` now takes the invocation's own `lane_id` + `instance_job_id` and exempts a
claim matching **both**, written within `SELF_CLAIM_MAX_AGE_HOURS = 6.0`. Identity alone is too
weak (a job id reused after an abandoned fire is a real phantom); freshness alone is too weak (a
sister lane's minutes-old claim is still a single-flight blocker).

The freshness leg was not expressible before: the reconciler's human table carries no
timestamp. `claim_lane_dispatch.py reconcile --format json` now emits `timestamp_utc` +
`age_hours` per active claim (additive; no other consumer of that JSON exists), and the fire
tool reads the structured report instead of scraping `claim: lane=…` lines. An unparseable
reconcile now closes **nothing** and records `reconcile_unparseable` — closing on a guess is
how the rc2 refusal happened.

### Controls, both executed

```
NO-SELF (pre-fix shape):  closed=[lane_test_a/modal:test_a rc 0]   exempt=[]
WITH-SELF (post-fix):     closed=[]   exempt=[lane_test_a/modal:test_a, age 0.0h]
                          CLAIM STILL ACTIVE: True   NO TERMINAL ROW: True
```

`tools/tests/test_fire_modal_auth_eval_claim_exemption.py` — **11 passed**. It drives the real
`claim_lane_dispatch.py` CLI against a tmp claims file and a tmp ledger; the canonical
`.omx/state/active_lane_dispatch_claims.md` is never touched (verified: 0 test rows in it).

- POSITIVE: fresh self-claim survives, and the on-disk newest row is still `active_paid_dispatch`.
- INVERSE ×3: another lane's claim closes; the same lane with a different job closes; the same
  lane AND job written 24 h ago closes. All verified terminal on disk, not just in the receipt.
- Mixed pass: two active claims, ours exempt and the phantom closed in the same sweep.
- Preconditions held: a live ledger row still suppresses every close; dry-run mutates nothing.
- WIRING: an AST assertion that `main()`'s call site passes both self-identity kwargs. A cure
  the entry point does not pass is an orphan cure.

### Class population — MEASURED

**1 site.** `grep -rl stale_superseded_reconciled_no_live_call --include=*.py tools src experiments`
returns `fire_modal_auth_eval.py` (the closer), `modal_harvest_poller.py` (a comment citing the
ck2 incident, not an emitter), and the new test. The fire tool's stage-4 closer is the only
automated claim terminal-closer in the repo, so the class is fully covered.

---

## ITEM 2 — PREMISE PARTLY STALE. The emitter was cured; the guard was instance-shaped.

### What was already done, with evidence

A prior ddm_ad1 session landed the emitter cure on **2026-08-20 07:35:35** in `36f4b29476`:
`src/tac/deploy/modal/result_json.py` (UTF-8 decode / base64 / record-the-transform, atomic
write, load-verified) and `tools/modal_harvest_poller.py` + `tools/harvest_click_polish_run.py`
routed through it. **I did not redo it and claim it.** The four repr-era receipts on the SSD
tier remain untouched (append-only): `ddm_jg5_custody/t4_receipts/harvested_artifacts/`,
`ddm_ps135_20260810/pass4_dispatch/artifacts_cuda/`,
`submittable_custody_mirror_20260811/leg_a_ps135_resync/…`, `ddm_t1r1_dispatch/artifacts_cuda/`.

### The gap that was NOT closed — and it is the one that matters

The anti-regression test read:

```python
    for relpath in ("tools/modal_harvest_poller.py", "tools/harvest_click_polish_run.py"):
```

A hand-typed denominator of 2. That is instance protection, not class protection — and a third
emitter already existed and was invisible to it: **`tools/harvest_modal_calls.py`**, the
canonical harvester CLAUDE.md names in "Modal `.spawn()` HARVEST OR LOSE", holds a raw
`fc.get()` result and carried **eleven** `default=str` dumps. `tools/parallel_harvest_actuator.py`
carried a twelfth.

Those twelve did not leak *today* — every value they encode is scalar-derived. But nothing
asserted that, and one added field (`"result": result`) reintroduces the defect silently. The
right frame: `default=str` is the **silencing mechanism**, not the bytes. With no `default`,
bytes raise `TypeError` — loud, unshippable. `default=str` converts that refusal into
`"b'{\n  \"final_score\"…'"`, which reads as data and fails `json.load` far downstream.

### Fix

`bytes_safe_json_default` (new, in `result_json.py`) keeps the `Path`/`datetime` coercion those
sites actually wanted and raises `BytesInSummaryError` naming `dump_modal_result_json` on
bytes. All 12 sites migrated. Both tools smoke-run clean read-only (rc=0).

### Guard: the class rule over a DISCOVERED denominator

> A production file that holds a raw Modal `FunctionCall` result may not hand `default=str` to a
> json dump.

The scan discovers its own population (`FunctionCall.from_id` / `poll_modal_call`), excludes
vendored `experiments/results/` snapshots and tests, and honours a same-line
`# MODAL_BYTES_REPR_OK:` waiver. A companion test asserts the denominator is ≥ 20 and still
contains both known harvesters, so the scan cannot pass by matching nothing.

### Controls, both executed

Applying the identical rule to the **pre-fix committed tree** (via `git show HEAD:<path>`):

```
PRE-FIX denominator (files holding a raw Modal result): 30
PRE-FIX offenders the class rule flags: 12
  tools/harvest_modal_calls.py:118,624,634,711,863,917,964,1011,1059,1110,1136
  tools/parallel_harvest_actuator.py:925
```

Post-fix: 0 offenders. Unit controls both ways: `default=str` demonstrably writes
`b'score 0.15'` into loadable-looking JSON; `bytes_safe_json_default` refuses that payload and
still coerces a `Path`. **13 passed** in `test_modal_result_json.py`; 72 passed across all
harvest-keyed suites.

### Class population — MEASURED

**33 production files** hold a raw Modal result in memory. **2** route through the canonical
projection (receipt writers — correct). **12 call sites across 2 files** carried the silencing
default and are now migrated. The remaining holders never json-dump a result-derived value, or
dump without a `default`, which fails loud. **Class coverage is complete under the stated rule.**

---

## ITEM 3 — a timeout that killed the shell and left the worker running

### Defect, from two independent measured instances

`subprocess.run(cmd, timeout=N)` kills the **direct child only**. When that child is a shell or
wrapper, the real worker is a GRANDCHILD and survives.

1. **The work keeps running after the harness gave up.** `ddm_cpu1`, 2026-08-20 —
   `src/tac/submission_chain.py::run_inflate` ran `["bash", "inflate.sh", …]` under
   `timeout=1800`. `TimeoutExpired` fired at **1799.99997045 s** and killed `bash`. The decoder
   ran on to **4,369.600210089 s** and wrote a complete 600-pair, 3,662,409,600 B report for a
   run the harness had already raised on — 2,570 s past a wall the caller believed it enforced.
   `ddm_rv17` round 1 records the same fact independently.
2. **The absence is then misread as "it never started."** `ddm_lc2`, 2026-08-10:
   `launch_detached_process.py` returned wrapper pid 21915, the dispatcher was grandchild 21916,
   `kill 21915` did not stop it, and "I killed it → it never spawned" went into a custody record
   as proof. `ddm_cd1` §7 records the same inference on `fire_modal_auth_eval.py` after an
   rc=144 reaper: the orphan kept walking toward a PAID dispatch while the empty ledger read as
   "it did nothing", and two firers briefly raced one lane.

### Fix — one canonical helper, not an eighth twin

`src/tac/process_group_kill.py`: `run_in_process_group` (spawn `start_new_session=True`, on
timeout escalate SIGTERM → grace → SIGKILL against the **group**), `kill_process_group`,
`group_alive`, and `ProcessGroupTimeout` — a `TimeoutExpired` subclass, so every existing
`except subprocess.TimeoutExpired` keeps working, carrying `group_survivors_after_kill`.

Seven private twins already existed (`memory_guard._kill_pgrp`, `safe_run._kill_group`,
`verdict_reclaim._kill_group`, `experiment_queue._signal_process_group`,
`dashboard_supervisor._killpg`, `launch_lane_with_retry`, `dashboard_ctl`) — none importable,
disagreeing on grace, escalation, and whether to verify the group died. This module is the
sibling of `tac.process_liveness`, which consolidated `_pid_alive` after the same drift across
11 copies. **Migrated sites: 3.**

| site | shape |
|---|---|
| `src/tac/submission_chain.py::run_inflate` | the MEASURED instance |
| `src/tac/submission_chain.py::run_upstream_evaluate` | `evaluate.py` spawns DataLoader workers |
| `modal_asymmetric_warp_deploy.py::_run_contest_compliant_auth_eval` | same bash-inflate shape, on METERED GPU |

Two leaf-binary calls (`git rev-parse`, `unzip`) carry `# GROUP_KILL_OK:` — they spawn nothing.
The Modal-side migration adds no remote dependency: the same function already imports
`tac.eval.auth_eval` 62 lines later.

### Two bugs the controls caught before landing

- **Zombie-only groups read ALIVE.** On macOS, `killpg(pgid, 0)` against a group whose only
  member is an unreaped zombie returns `EPERM`, so the first receipt said
  `survivors_after=True` for a tree that was fully dead — the same false inference this module
  exists to end. `kill_process_group` now takes a `reap` callback (`proc.poll`) invoked before
  every liveness read.
- **The cure would have orphaned on Ctrl-C.** `start_new_session` also detaches the child from
  our terminal's signal delivery. A `BaseException` leg now takes the tree down before
  re-raising, with its own test.

### Controls, both executed on the REAL migrated site

A fake `inflate.sh` whose "decoder" is a background grandchild, driven through the actual
`tac.submission_chain.run_inflate`:

```
POST-FIX (migrated run_inflate, timeout=2):
  decoder grandchild pid=41095   alive after timeout: False
  group_survivors_after_kill: False
  receipt: {pgid 41094, term_delivered True, kill_delivered False, survivors_after False, 0.104 s}

PRE-FIX (bare subprocess.run, same fixture):
  decoder grandchild pid=41219   alive after timeout: True
```

`src/tac/tests/test_process_group_kill.py` — **16 passed**. Every test spawns a real tree and
reads liveness from the OS; nothing mocks the kill. It leads with the positive control
(stock `subprocess.run` orphans the grandchild) so the cure's assertions are not measuring an
absent hazard, and covers SIGTERM-ignoring escalation, Ctrl-C, unchanged normal/rc/cwd/env
behaviour, and refusals (own group, non-positive pgid). Anti-twin tests assert the migrated
files route through the helper and hold no un-waived timed `subprocess.run`.

### Class population — MEASURED, and the backlog is named not hidden

**310** timed `subprocess.run` sites in `tools/` + `src/` (non-test). **29** launch a shell,
wrapper script, or nested python and are true class members. **3 migrated; 26 remain.**

The 26 are a real, owned backlog — not covered by this landing, and not claimed to be. Named
head of queue: `pb1_p5_byte_close_and_eval.py:428`, `pfs1_recompose_warp_base_and_eval.py:686`,
`run_decoder_q_candidate_inflate_controls.py`, `verify_distinguishing_feature_byte_mutation.py`,
`operator_authorize.py:1989/2159/2288`, `sweep_m5max_hnerv_cluster.py:371` (two nested timeout
layers, neither group-scoped), `dashboard_supervisor.py:154` (a timeout that can kill the killer
between its own SIGTERM and SIGKILL). Fire-condition: the next reviewed hygiene window on any of
those files, or the first time one of them is on a paid path.

Two existing gates do **not** cover this class: Catalog #389 matches only the
`nohup + bash -c + | tee + &` launch signature, and `check_retry_without_descendant_check`
matches only detach-spawning respawn helpers. Neither sees `subprocess.run(["bash", …], timeout=N)`,
which is where the whole population lives. A STRICT preflight gate over the 29-site rule is the
natural successor to the file-scoped anti-twin test landed here; it is owed, not done.

### The half this does NOT fix

`ddm_cd1` names the generalisable error as the **inference**, not the reaper: "absence of a
downstream artifact seconds after an rc=144 is indistinguishable from a process that never
started." A group kill fixes the mechanism. `group_survivors_after_kill` on the raised timeout
is this arm's contribution toward the inference half — a caller can now state that the tree is
down instead of assuming it — but no call site consumes it yet. Owed.

---

## Incidental fix: a leaked global `subprocess.run` patch

`src/tac/tests/test_preflight_hook_heavy_import_scope.py:325` did `hook.subprocess.run = _boom`.
`hook.subprocess` **is** the global `subprocess` module, so that bare write replaced
`subprocess.run` for the whole session; every later test running a real process died with
`FileNotFoundError: .venv/bin/python`. It was already breaking
`test_process_liveness::test_zombie_is_dead_not_alive` (verified at HEAD, in a file I never
touched) and would have made Item 3's grandchild control flake on every full-suite run.
Changed to `monkeypatch.setattr`; the assertion is unchanged. Combined run: **265 passed, 0 failed.**

## Test totals (honest counts)

| suite | result |
|---|---|
| `tools/tests/test_fire_modal_auth_eval_claim_exemption.py` (new) | 11 passed |
| `tools/tests/test_fire_modal_auth_eval_axis.py` + `test_candidate_seal.py` | 72 passed |
| `src/tac/tests/test_modal_result_json.py` (5 new) | 13 passed |
| all `-k "harvest_modal or modal_harvest"` | 72 passed |
| `src/tac/tests/test_process_group_kill.py` (new) | 16 passed |
| combined `-k submission_chain/process_liveness/process_group/warp_deploy/asymmetric/preflight_hook/candidate_seal/fire_modal/modal_result` | **265 passed, 0 failed** |

Ruff clean on every file touched. The 11 pre-existing errors in `modal_asymmetric_warp_deploy.py`
and 3 in `test_preflight_hook_heavy_import_scope.py` are unchanged from HEAD — I added none.
