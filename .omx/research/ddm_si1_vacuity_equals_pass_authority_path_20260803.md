# ddm_si1 — "failure and success emit the same symbol" (task #929)

**Axis:** `[repo-apparatus]` · no scorer, no GPU, $0 · `score_claim=false` · the exact pointer
did not move and nothing here is byte-closed.

---

## §0 The headline I was sent to act on is NARROWER than it reads — measured, not argued

The charter opened with *"`ob1`: AN INFLATE REPORTED EXIT 0 WHILE FAILING … on the one path
whose entire job is to be authoritative,"* and instructed me to reconcile that against `ob1`'s
other recorded fact — a decode bit-identical to its certificate. Both are true, and the
reconciliation decomposes the incident into three parts that were being carried as one:

| part | what it is | status |
|---|---|---|
| **the script** | the emitted `inflate.sh` | **NOT the liar.** With `set -euo pipefail`, run in the FOREGROUND with no interpreter, the pre-fix bare-`python` form returns **127** exactly as it should. Executed negative control. |
| **the trigger** | the bare `python` | **REAL, and bigger than a harness annoyance.** Executed control: on a **python3-only host** the old script dies `rc=127` while the fixed one succeeds. FIXED. |
| **the amplifier** | a backgrounding launcher | **REAL and UNFIXED.** A backgrounded job's exit status is the *launcher's*, not the job's. Reproduced live: launcher `rc=0`, job failed, zero output. |

`ob1`'s two facts are consistent because the failed run was **caught and discarded** — the arm
read the zero-byte log, re-ran with `PATH=…/.venv/bin` prefixed (visible in the `pu2` cleanup
certificate's own `rebuild_command`), and *that* run produced the bit-identical decode.

**What this refutes in my charter:** the defect was not in repo byte-close code. All three
emitters (`pb1_p5`, `rehearse_tr1`, `pfs1`) already inject `PATH` and check/record `rc` for
their own runs. The exit-0 lie came from an **ad-hoc hand-typed invocation**, which is itself the
finding: the operation had no fail-closed entry point, so an arm hand-rolled it and got bitten.

**What I am NOT claiming:** that the amplifier is closed. It is not. Any backgrounded invocation
of any script still has this property. It is pinned as a live executed control
(`test_amplifier_backgrounded_launcher_reports_zero_for_a_failed_job`) so the exposure stays
measured rather than implied. The cure is a completion marker carrying the job's real rc; no
canonical helper provides one, and I deliberately did not invent a new surface for it
(built-instead-of-paid).

---

## §1 Do any byte-closed rows need re-verification? **No — and here is the arithmetic**

Our own frontier arc is `v4d → pw1 → ms8 → pj2 → cx1 → pu2`. Checked at the authority end:

```
report.txt (600 samples, denominator STATED):
  Average SegNet  Distortion : 0.00431179
  Average PoseNet Distortion : 0.00154519
  Submission file size       : 353,805 bytes
recomputed  S = 100*seg + sqrt(10*pose) + 25*bytes/37_545_489 = 0.7910689
claimed     S = 0.7910689           delta = -1.5e-09
```

* `archive.zip` sha `c72ef357416b66e716b2863c4c49360306b80cc0fafd094e02394c8a4dd37209`
  matches the cleanup certificate **and** the live bytes on disk.
* `ob1` **independently re-inflated** that exact archive: bit-identical on **1194/1200** frames.
  The 6 differing frames are all `frame_0` of pairs `[16, 21, 67, 71, 74, 523]` = `pu2`'s
  re-solved tail pairs. SegNet reads `x[:, -1, ...]` and never sees `frame_0`, so `d_seg` is
  bit-exact across `cx1 → pu2` by construction.

A row whose components recompute to 1.5e-9, whose archive sha is stable, and which a second
party re-inflated bit-identically is verified by evidence that never passes through the
suspect wrapper. **No re-verification is owed.**

---

## §2 What I fixed, and the proof each fix is not decoration

Every guard below ships with **executed** controls, and every control set was
**mutation-checked**: I reverted the fix and confirmed the tests go RED. A green test that
cannot go red is the same bug wearing a lab coat.

| # | site | defect | controls | mutation |
|---|---|---|---|---|
| 1 | `tools/pb1_p5_byte_close_and_eval.py` · `tools/rehearse_ddm_tr1_runtime.py` · `tools/pfs1_recompose_warp_base_and_eval.py` | emitted `inflate.sh` ran bare `python` | 17 (5 per emitter + 2 shared) | 4 RED per emitter |
| 2 | `src/tac/subset_selection_gate.py` | `staged_py_files`/`in_scope_py_files` returned `[]` on git failure | 2 | 2 RED |
| 3 | `tools/preflight_hook.py` · `run_review_gate` | absent gate / absent interpreter returned `0` | 2 | 2 RED |

**Fix 1** resolves `PYTHON → python3 → python`, refuses `exit 127` with a diagnostic when none
exists, and `exec`s so the runner's rc cannot be swallowed. `pfs1` matters most: it produced the
canonical `v4d_cx1_pj2ix2` decode that `ob1` compared against, i.e. it is on the frontier arc's
authority path, not adjacent to it.

**Fix 2 is the sharpest thing I found.** `ss1` gave `added_lines()` a `None` "we could not tell"
channel with a precise comment explaining why `[]`-on-git-failure is indistinguishable from a
clean file. `staged_py_files()` sits **four lines above it** and returned `[]` on git failure —
docstring: *"Empty on any git failure."* That empty list flowed into `scan_staged` as a zero
denominator and printed `VACUOUS: 0 staged .py files`, which is not merely uninformative but
**false**: git had broken; the commit was not empty. And the test that **names the class**,
`test_git_failure_is_reported_as_unknown_not_as_no_added_lines`, asserted
`added_lines(...) is None` and `staged_py_files(tmp_path) == []` **on adjacent lines** — pinning
the uncured neighbour as expected behaviour. The cure does not generalise itself.

**Fix 3** is on the surface that fires every commit: `run_review_gate` returned `0` both when
`review_gate_hook.py` was absent and when the interpreter could not launch — the second being
`ob1`'s shape verbatim. That return becomes the hook's exit code to git.

---

## §3 Denominators (charter item 4 — no exhaustiveness claim)

Searched, tracked files only: **6,019 `.py`** (`tools/` 1,923 · `src/tac/` 3,202 ·
`experiments/` 867 · `scripts/` 27) and **350 `.sh`**. Derived:

* **462** subprocess sites near inflate/evaluate/archive/scoring; **70** with no rc check within
  28 lines; **41** with an rc test whose handler neither raises nor returns failure.
* **5,414** `inflate.sh` on disk; **214** carry a truly bare `python`; **1** git-tracked
  (`submissions/hnerv_lc_ac/inflate.sh:19`).
* **18** `tools/check_gate*.py` share one `main()` that prints `OK` when zero items were examined.
* **~50** narrowed empty-on-failure collectors; the dominant family is `src/tac/preflight.py`'s
  `_check_NNN_collect_violations_in_file`, where `OSError`/`SyntaxError` both return `[]` —
  "I could not read this file" encoded as "this file is clean".

**Not reached:** 52,674 untracked `.py` + ~25,000 untracked `.sh` under generated trees;
vendored `*_intake_*` clones; `.venv`; test trees; caller-side rc verification for helpers that
correctly return `CompletedProcess`. The `preflight.py` scan was regex+indent, not AST, so
multi-line `return (\n [] \n)` forms were missed. **I did not find every instance and do not
claim to have** — negative-existence claims are our #1 measured false-claim class.

---

## §4 Named remaining debt — owned, not parked

1. **`src/tac/pr103_lc_ac_runtime_adapter.py:727-732` ENFORCES THE BUG AS A CONTRACT.** It
   asserts `expected = 'python "$HERE/inflate.py" "$SRC" "$DST"'` and raises *"expected exactly
   one bare python inflate invocation"*. Applying the cure to a PR103 packet would be **refused
   by our own gate**. Not touched: PR103/PR101 are banned old lineage (harvest-only, never
   ships) and it is a frozen reproduction artifact. **Owner: deferred, fire-condition = any
   attempt to re-run or re-ship a PR103-lineage packet.**
2. **Three more emitters** carry bare `python`: `tools/witness_byte_close_and_eval.py:389`,
   `src/tac/v2_compose/archive_grammar.py:1044`,
   `src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py:2740`. The first is live
   witness-side and should take the same fix. **Owner: me, next unit.**
3. **The repo already had a partial cure** — `tools/build_pr101_frame_conditional_runtime_packet.py:1239`
   rewrites to `"${PYTHON:-python3}"`. It was never generalised, which is the same
   cure-does-not-spread pattern as §2 fix 2.
4. **The amplifier** (backgrounded launcher rc) — specification, not built. See §0.
5. **`tools/run_hi_nerv_backend_only_b2_exact_eval.py:771-812`** — on `rc != 0` it records a
   stderr tail and then *still* lifts `final_score` out of a possibly-stale `cae_json` and sets
   `pipeline_works = True`. Highest-severity class-A hit I did not fix. **Owner: deferred,
   fire-condition = any HiNeRV backend row entering a decision.**

**Reference shapes that already do it right** (use these, do not re-derive):
`tools/preflight_ddm_sf1_fit_context.py:40-52` prints the scanned count and treats a vacuous
scan as `return 1`; `src/tac/optimization/archive_bound_candidate_runtime_bridge.py:141-152`
distinguishes timed_out / nonzero / not_executed / output_missing / output_empty.

---

## §5 The class cure, stated so it can be applied without me

1. **Give "I examined nothing" its own type.** `None`, not `[]`/`{}`/`set()`/`0`. A falsy value
   coerces to "fine" at every call site; a distinct type forces the caller to say so out loud.
2. **Report the denominator, always.** Never a bare count and never a bare `OK`. `examined N of M`.
3. **Apply `sm1`'s law to every gauge:** *what does this read if the cure is applied and nothing
   else changes?* If the answer is "the same thing", the gauge is not measuring the cure.
4. **Mutation-check every guard.** Revert the fix; the controls must go RED. This is the only
   evidence a green is sensitive to the thing it claims to test — `ss1`'s lint printed
   "All checks passed!" over **zero** files, so a green check is not evidence a check ran.
   *(I ran this on my own `ruff` invocation this unit: `--show-files` confirmed 3, not 0.)*
5. **Prefer structural over procedural** (`sm1`): a type or a refusal beats a documented practice.

**Sixth, learned here and not in the prior four:** *the cure does not generalise itself.* All
three fixes were **neighbours of an existing cure** — `added_lines` four lines away, a
`${PYTHON:-python3}` rewrite already in the tree, a vacuity-refusing reference gate already
written. Fixing a site and not sweeping its siblings in the same file is how five instances
appear in twelve days. **When a cure lands, grep the file it landed in.**
