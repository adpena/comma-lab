# ddm_bl1_20260805 receipt

Task: BL1 / #937 backgrounding-launcher rc amplifier + pr103 bare-python contract.

Axis: `[repo-apparatus]`. Scorer-free. No paid dispatch. No exact/archive score claim.
Own-vehicle frontier unchanged: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.

## RECALL EVIDENCE

Governing reads:
- `.omx/tmp/codex_runs/bl1_prompt.md`
- `.omx/tmp/codex_runs/_common_contract.md`
- `PROGRAM.md`
- `CLAUDE.md` / `AGENTS.md` relevant bug/self-protection and serializer sections (`diff -q` showed the two files identical)
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`
- `.omx/research/ddm_si1_vacuity_equals_pass_authority_path_20260803.md`

Memory/query pass:
- `rg -n "bl1|BL1|common_contract|codex_runs|Block" /Users/adpena/.codex/memories/MEMORY.md` found only unrelated prior landing-review context.
- `rg -n "ddm_si1|si1|#937|pr103|backgrounded|launcher rc" /Users/adpena/.codex/memories/MEMORY.md` found no direct memory hit.

Corpus recall beyond charter seeds:
- `rg --files .omx/research | rg "ddm_si1|si1"` found the si1 memo named by the charter.
- Filtered canonical-equations query:
  `.venv/bin/python tools/list_canonical_equations.py --json | rg -n "launch_detached_process|backgrounded launcher|done-receipt|bare python|pr103_lc_ac|vacuity|returncode"`
  found only general vacuity lineage rows, no launcher/pr103-specific canonical equation.
- Filtered index/DAG/hot-state query over `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, `main_hot_state.md`, and `docs/operating_workflow_v2_velocity_rigor_autonomy_20260720.md` found no BL1-specific prior cure beyond the known `launch_detached_process` operational references.
- Live bare-python sweep:
  `rg -n "^\\s*python\\s+.*inflate\\.py" tools src/tac -g '*.py'`
  found the three si1 non-pr103 emitters still live:
  `tools/witness_byte_close_and_eval.py:389`,
  `src/tac/v2_compose/archive_grammar.py:1044`,
  `src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py:2740`.

What recall changed:
- Kept the cure on the existing `tools/launch_detached_process.py` surface; no new launcher.
- Added self-protection as a `tac.preflight` warn-only scanner rather than a hook-only check.
- Fixed the pr103 adapter instead of preserving the old bare-python contract.
- Left the three non-pr103 bare-python emitters queued with a named fire-order, rather than silently treating BL1 as a global bare-python cleanup.

## Changes

1. `tools/launch_detached_process.py`
   - The `--done-receipt` supervisor still writes the child process rc.
   - It now also writes fail-closed receipt rows for exec-start failures:
     `rc=127` for `FileNotFoundError`, `rc=126` for other `OSError`.

2. `tools/tests/test_sigurg_kill_class_guard.py`
   - Added executed BL1 positive control:
     parent launcher returns rc 0 after starting the detached supervisor, while the `.done` receipt records child `rc=7`.
   - The parent rc assertion carries `# LAUNCHER_RC_OK:` because it is launch-start health only.

3. `src/tac/preflight.py`
   - Added `check_background_launcher_rc_not_job_verdict(...)`, wired into `preflight_all()` with `strict=False`.
   - The scanner catches literal `launch_detached_process.py` `subprocess.run(..., check=True)`, `returncode == 0`, `0 == returncode`, and `not returncode` success verdicts.
   - Same-line waiver: `# LAUNCHER_RC_OK:<rationale>`; placeholder rationales are rejected.
   - `returncode != 0` launch-start failure checks remain allowed.

4. `src/tac/tests/test_check_bl1_background_launcher_rc.py`
   - Positive, negative, waiver, placeholder-waiver, shell `&&`, strict-mode, and live-repo zero-count tests.

5. `src/tac/pr103_lc_ac_runtime_adapter.py`
   - Replaced the adapter's bare-python shell contract with a portable fail-closed block:
     `PYTHON` env var, then `python3`, then `python`, else `exit 127`, and final `exec`.
   - The adapter refuses mixed/duplicate legacy vs portable shell forms.

6. `src/tac/tests/test_pr103_lc_ac_runtime_adapter.py`
   - Updated contract expectations: no bare python, `python3 python` discovery, `exit 127`, and `portable_python_fallback=true`.

## Positive-Control Transcript

Before/failure reproduction from si1:

```text
$ .venv/bin/python -m pytest -q src/tac/tests/test_emitted_inflate_sh_fails_closed.py::test_amplifier_backgrounded_launcher_reports_zero_for_a_failed_job
.
1 passed in 0.37s
```

After/fix control:

```text
$ .venv/bin/python -m pytest -q src/tac/tests/test_emitted_inflate_sh_fails_closed.py::test_amplifier_backgrounded_launcher_reports_zero_for_a_failed_job tools/tests/test_sigurg_kill_class_guard.py::test_launcher_done_receipt_records_child_nonzero_rc
..
2 passed in 0.42s
```

Meaning: the old hand-backgrounded launcher still reproduces the amplifier (`launcher rc=0`, job failed), and the canonical detached launcher path now has an executed control proving the child rc is read from the `.done` receipt (`rc=7`), not inferred from the parent launcher rc.

## Tests

```text
$ .venv/bin/python -m py_compile tools/launch_detached_process.py src/tac/pr103_lc_ac_runtime_adapter.py src/tac/preflight.py tools/tests/test_sigurg_kill_class_guard.py src/tac/tests/test_pr103_lc_ac_runtime_adapter.py src/tac/tests/test_check_bl1_background_launcher_rc.py
rc=0

$ .venv/bin/python -m pytest -q src/tac/tests/test_check_bl1_background_launcher_rc.py tools/tests/test_sigurg_kill_class_guard.py src/tac/tests/test_pr103_lc_ac_runtime_adapter.py
....................
20 passed in 21.17s

$ .venv/bin/python -m pytest -q src/tac/tests/test_pr103_lc_ac_runtime_adapter.py
.......
7 passed in 1.64s
```

Review tracker:

```text
Two `tools/review_tracker.py mark-file ... --status reviewed` passes completed for:
tools/launch_detached_process.py
tools/tests/test_sigurg_kill_class_guard.py
src/tac/pr103_lc_ac_runtime_adapter.py
src/tac/tests/test_pr103_lc_ac_runtime_adapter.py
src/tac/preflight.py
src/tac/tests/test_check_bl1_background_launcher_rc.py
```

## si1 Remaining Debt Disposition

1. `src/tac/pr103_lc_ac_runtime_adapter.py:727-732` enforced bare `python`.
   - Disposition: fixed here.

2. Three additional bare-python emitters:
   - `tools/witness_byte_close_and_eval.py:389`
   - `src/tac/v2_compose/archive_grammar.py:1044`
   - `src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py:2740`
   - Disposition: queued with fire-order `BL1-FOLLOW-BPY`: patch these three emitters to the same fail-closed shell block, extend emitted-inflate tests to cover them, run focused pytest, two review passes, serializer commit. Fire after this BL1 landing or at the next scorer-free apparatus cleanup boundary.

3. Partial PR101 cure already existed.
   - Disposition: already-fixed-elsewhere in `tools/build_pr101_frame_conditional_runtime_packet.py` (`git log -1`: `da76fbc08c Emit contracts from runtime proofs`). BL1 does not claim that partial cure was global.

4. Backgrounding launcher rc amplifier.
   - Disposition: fixed/self-protected here on the existing canonical detached launcher surface plus warn-only preflight scanner. No new launcher.

5. `tools/run_hi_nerv_backend_only_b2_exact_eval.py:771-812` stale-score-on-rc-failure path.
   - Disposition: queued with fire-order `BL1-FOLLOW-HINERV-RC`: before any HiNeRV backend row is used for a decision, patch the path so `rc != 0` fails closed before reading or reporting `final_score`; add a regression test where stale `cae_json` exists but subprocess rc is nonzero.

## Boundaries

- No scorer job, no GPU job, no paid dispatch, no exact eval.
- No live `w4` / `w4m` run-dir touch.
- No sealed trainer or `jd1` ticket touch.
- No `upstream/` mutation.
- No score movement claim.
