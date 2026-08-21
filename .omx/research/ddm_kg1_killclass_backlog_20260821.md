# ddm_kg1 — the kill-doesn't-reach-the-tree backlog is drained, and the class now has a gate

**Date:** 2026-08-21 · **Arm:** ddm_kg1 · **Task:** #1177 · **Scope:** apparatus debt only.
No dispatches, no launches, no `upstream/` edits, no frozen-gen6 edits.
**Pointer:** UNMOVED. This arm produced no score row and claims none — it is MEANS.

---

## ANSWER FIRST

The ddm_ad1 memo named **26** remaining sites. Re-deriving the population from the AST
instead of inheriting the list found **48**. All 48 are resolved: **41 migrated** to
`tac.process_group_kill.run_in_process_group`, **7 waived** with real rationales, **live
count 0**. Catalog **#408** `check_no_timed_shell_wrapper_without_group_kill` landed
**STRICT from byte one** — the class that ate 2,570 s of un-killed decoder now fails a
commit instead of a run.

Three findings the inherited list would have hidden:

1. **The named backlog was partly wrong, in both directions.** `operator_authorize.py:1989/2159/2288`
   — named as head-of-queue — are `subprocess.call(...)` with **no `timeout` at all**. No
   timeout means no kill, so the class defect cannot fire; they are not members. Meanwhile
   20 genuine members were never named.
2. **Module-scope name resolution loses real sites.** A first-pass resolver that pooled
   assignments module-wide lost two genuine `["bash", …]` sites
   (`rehearse_ddm_tr1_runtime.py:461`, `run_compact_renderer_mlx_spine_runner.py:9126`)
   because a common variable name like `cmd` resolved to whichever assignment came first in
   walk order. The shipped gate resolves **function-scoped**.
3. **The highest-value site was the one the helper could not yet express.**
   `modal_train_lane.py:1717` runs `["bash", str(lane_path)]` on **PAID Modal compute** for
   up to **14 h**, streaming to a log fd. It is the exact ddm_cpu1 shape — the timeout kills
   `bash` while the trainer grandchild keeps billing — and it was unmigratable because
   `run_in_process_group` had no `stdout=`. Waiving it would have left the class's worst
   instance uncured on the only metered path. I extended the helper instead.

---

## Class population — MEASURED, with the denominator the gate itself computes

The gate **is** the census: one implementation, so the numerator and denominator can never
drift apart. This is the ddm_ad1 Item-2 lesson applied — that guard used a hand-typed
denominator of 2 and was blind to a third emitter.

| quantity | count |
|---|---:|
| Timed `subprocess.run/check_output/check_call/call` sites, production (tools+src+scripts+experiments, non-test, non-vendored) | **429** |
| — of which in `tools/` + `src/` (the ddm_ad1 scope) | 298 |
| **True class members** (argv[0] = shell / wrapper / `.sh`), un-migrated at re-derivation | **48** |
| — migrated to `run_in_process_group` | **41** |
| — waived with real rationale | **7** |
| — dead code | **0** (none claimed; see below) |
| Vendored/intake-excluded timed sites (11 shell-shaped) | 163 |
| Test-file-excluded timed sites (39 shell-shaped) | 341 |
| **Residual un-cured** | **0** |
| Previously migrated by ddm_ad1 | 3 |
| **Total class members ever identified** | **51** |

**Why 48 and not 26.** The ad1 memo's 29/26 came from a narrower pass. The re-derivation
counted every timed subprocess API (not just `.run`), resolved argv[0] through
function-scoped local assignments, and followed `Path(...) / 'inflate.sh'` expressions
transitively. Two independently-written detectors were cross-checked against each other and
agreed on **47 of 48** sites; the single apparent disagreement was the *same* site shifted
by my own edit, not a real difference.

**No dead-code claims.** Several files are old (`prove_pr95_*`, `train_substrate_z3_g1_*`).
I did not claim any of them dead: proving non-reachability across this repo's dispatch
surfaces costs more than the mechanical migration, and a wrong dead-code claim is a silent
false negative. Migration is correct whether or not the file ever runs again.

---

## The 7 waivers, and why each is honest

Two legitimate classes, both recorded in the gate docstring.

**(a) A leaf the resolver cannot prove leaf — 4 sites.**

| site | rationale |
|---|---|
| `src/tac/preflight.py` (bash syntax scan) | `bash -n` PARSES and exits; it never executes a command, so it forks nothing |
| `src/tac/phase1_packet_compiler.py:625` | same `bash -n` shape |
| `tools/build_track2_identity_packet.py:306` | same `bash -n` shape |
| `experiments/contest_auth_eval.py:1219` | argv[0] is a *function parameter*; every caller passes a leaf (`nvidia-smi` :1269/:1270, `ffmpeg` :1288/:1290, `git rev-parse` :1297/:1300). Also keeps `stderr=STDOUT`, which the capture path cannot express |

I migrated one `bash -n` site first and then reverted it, because treating the identical
shape two different ways in one landing is the inconsistency that makes a waiver class
meaningless. All three now read the same.

**(b) `tac` is not importable where the call executes — 3 sites.**
`experiments/modal_fx5_linux_dependency_closure.py:123/142/181` run inside a Modal container
whose image mounts only the fx1 runtime tree **as data**
(`add_local_dir(LOCAL_TREE, REMOTE_TREE)`) plus the entrypoint module, with
`include_source=False` and no tac wheel. `import tac` cannot resolve there. Mounting `src`
to satisfy the gate would also weaken the measurement, whose whole premise is a
provably-minimal container.

By contrast `modal_alpha_geo0_pose_regen.py` and `modal_train_lane.py` are *also* remote but
their images DO mount `src` with `PYTHONPATH` set — proven by their own module-level
`from tac...` imports executing in-container. Those were migrated, not waived. "Remote" is
not the discriminator; **mounted** is.

---

## The helper extension — required, not convenient

`run_in_process_group` gained `stdout=`/`stderr=` passthrough, `ValueError` on combining
either with `capture_output=True` (same contract as `subprocess.run`).

Without it, two sites were unmigratable, and the reason is not stylistic:
- `modal_train_lane.py:1717` — up to **14 h** under `stdout=logf`. `capture_output` would
  hold the entire run in parent RSS and destroy log liveness under a crash.
- `ddm_cp2_composition_receiver_and_harness.py:652` — appends to a live log fd across a full
  n600 `inflate.sh`, with `log.flush()` + `os.fsync` so a partial log survives.

Four new tests cover it, including `test_partial_log_survives_a_timeout_on_a_file_fd` and a
re-proof that the **group kill still reaches the grandchild on the fd path** — the cure had
only ever been demonstrated on the PIPE path.

---

## Catalog #408 — the gate the class lacked

**Why a new gate rather than extending one.** Catalog #389 matches only the
`nohup + bash -c + | tee + &` DETACHED-daemon launch signature.
`check_retry_without_descendant_check` matches only detach-spawning respawn helpers.
Neither sees `subprocess.run(["bash", …], timeout=N)` — which is where the entire population
lives. #408 is #389's sister at the **synchronous-timeout** surface.

**Detection is AST, not regex**: a timed `subprocess.run/check_output/check_call/call` whose
argv[0] resolves — through function-scoped local assignments, transitively through
`Path(...) / 'inflate.sh'` — to a shell, a wrapper binary, or a `.sh`/`.bash`/`.zsh` script.
A leaf binary (`git`, `unzip`, `ffprobe`) does not match. Tests and vendored/intake paths are
skipped. Waiver `# GROUP_KILL_OK:<rationale>` anywhere in the call's own source span;
placeholder rationales (`<rationale>`, `<reason>`, `TODO`, `TBD`) rejected per #287.

### Controls, both directions, EXECUTED

`src/tac/tests/test_check_408_timed_shell_wrapper_group_kill.py` — **26 passed**.

The suite **leads with the positive control**: the un-cured ddm_cpu1 shape must FIRE before
any "the cure passes" assertion is trusted. A guard never observed to fire is the #1086 bug
class.

- **FIRES**: the exact `bash inflate.sh` + `timeout=1800` shape · every shell head
  (`bash`/`sh`/`/bin/bash`/`/usr/bin/env`/`nohup`/`timeout`) · a bare `.sh` executed directly ·
  a variable `cmd` resolved through a local assignment (with a decoy `git` assignment in a
  sibling function, which must NOT fire) · a `Path` expression head · all four subprocess APIs.
- **PASSES**: a migrated site · leaf binaries · an **untimed** shell call (no timeout → no
  kill → class cannot fire) · a real waiver · tests and vendored paths.
- **REFUSES**: a placeholder waiver; strict mode raises `PreflightError`.
- **DENOMINATOR control**: `test_scanner_denominator_is_not_vacuous` asserts the scanner
  still fires on the canonical shape, so a future refactor that silently breaks detection
  fails there instead of reporting a clean repo. Vacuity==pass is the silent-instrument bug.
- **Live-count regression**: asserts 0 over the real tree.

**STRICT from byte one**, per the strict-flip atomicity rule — live count reached 0 inside
the landing batch, so warn-only purgatory was never entered.

**Cost**: 7.4 s over 4 directories. The first implementation took 10.3 s because it
re-walked every enclosing scope's full subtree once per candidate call; scope assignments
are now memoised per scope. It runs in the dev-scope `preflight_all`, not in the fast
commit hook, so 7.4 s is paid where the full gate set is already paid. I deliberately did
NOT add a cheap "does this file mention bash?" text pre-filter: the tokens that would have
to be in it (`env`, `timeout`, `make`) are common enough that the filter buys little, and a
wrong token list would convert the gate into a silent false negative — the exact vacuity
failure the denominator control exists to catch.

---

## What I verified, and what I did NOT

**Verified (executed, not reasoned):**
- Ruff, full repo config, **per-file delta against the HEAD version of the same file**, so
  only violations I *introduced* counted. Net: zero new lint; five pre-existing `I001`s fixed
  as a side effect.
- **Import-placement hazard scan**: caught 3 real hazards where the auto-inserted `tac`
  import landed *before* `ensure_repo_imports(REPO_ROOT)` — an ImportError under direct
  script execution. All three now import after the bootstrap with `# noqa: E402`.
- **Exec-module smoke** on every migrated file (catches ImportError, which `py_compile` does
  not). Zero import failures. Six files raise an `AttributeError` under my ad-hoc loader —
  **confirmed identical at HEAD**, so it is a loader artifact, not a regression.
- Semantic scan of all 41 migrated calls: no string argv, no unsupported kwarg, no extra
  positional, no `type(...) is TimeoutExpired` identity check that a subclass would break.
- Suites: `test_check_408_*` (26), `test_process_group_kill` (20, was 16),
  `test_submission_chain` + `test_archive_bound_*` + `test_build_phase1_packet_compiler` +
  `test_phase1_packet_compiler_*` (171 combined).

**NOT verified — stated plainly:**
- **No migrated site was executed end-to-end against a real `inflate.sh` run.** The grandchild
  kill is proven by `test_process_group_kill`'s real process trees (including on the fd path),
  and each call site is proven to still import and parse — but I did not run a full n600
  inflate, and I did not run anything on Modal.
- **The `modal_*` migrations are not proven in-container.** That `tac` is importable there is
  inferred from the image mount spec plus each module's own module-level `from tac...` import
  executing in-container — strong, but it is inference, not an executed remote run.
- **The inference half of the defect is still open.** `ddm_cd1` names the generalisable error
  as reading absence-after-a-kill as "it never started". `group_survivors_after_kill` now
  rides on every raised timeout at 41 more call sites, but **no call site consumes it yet**.
  Owed, and unchanged from ad1.
- The 39 shell-shaped sites inside test files are deliberately out of scope: tests drive both
  the cured and un-cured shape as controls.

---

## Owed

1. **Consume `group_survivors_after_kill`.** 41 call sites can now state whether the tree is
   provably down; none does. The highest-value consumer is `modal_train_lane`'s
   `except TimeoutExpired` → rc=124 path, which currently says "collecting partial artifacts"
   without saying whether the trainer actually stopped.
2. **The nested-python tier.** #408 covers the shell-wrapper shape my charter named. A
   sibling tier exists: `subprocess.run([sys.executable, ...], timeout=N)` where the nested
   python itself spawns — **~59 candidate sites in tools+src** at the re-derivation, not all
   true members (a `python -c "import torch"` probe spawns nothing). It needs its own
   spawns-or-not discriminator before a gate; naming it here so it is a tracked queue entry
   and not a forgotten default.
3. **`scripts/launch_lane_with_retry.py` and `tools/dashboard_supervisor.py`** carry their own
   private killpg twins (2 of the 7 ad1 named). They are now *callers* of the canonical helper
   at their timed sites but still hold private group-kill code — a consolidation, not a bug.
