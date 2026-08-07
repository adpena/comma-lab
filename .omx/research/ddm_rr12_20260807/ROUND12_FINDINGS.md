# ddm_rr12 Round 12 Findings

status: NOT-CLEAN
round: 12
clean_pass_counter_after_round: 0/3
axis: apparatus / scorer-free
score_claim: false
frontier_moved: false

## Counts First

| item | count / status |
|---|---:|
| Findings filed | 3 |
| Fixed in this round | 3 |
| RR11-F1 ticket write paths found in worker modes | 0 |
| RR11-F1 regression tests added | 1 |
| v4 arm discriminator checks | 2/2 passed |
| v4 regen path leak checks | 3/3 artifacts passed |
| AB1 random trainer spot-check | 10 checked |
| AB1 placement defects in sample | 1 fixed |
| #254 strict scanner | 0 residuals / 150 scanned |
| Scorer / archive / remote launches | 0 |

## Findings

### RR12-F1 - HIGH - RR11-F1 fix lacked a durable regression test

The live code now writes `_ticket_path_for_args(args)` only in `mode == "probe"`, so I did
not find another ticket-write path in `mem-probe`, `mlx-train`, `mlx-parity`, or `torch-smoke`.
The missing regression from RR11 was real.

Fix landed: `experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py` runs `mode=mem-probe`
with a copied launch ticket and asserts the ticket bytes stay unchanged, including the
`argv_n32_arm_veh` `tq1c_seg_cache.pt` discriminator.

verdict_scope: APPARATUS / regression coverage.

### RR12-F2 - HIGH - safe_run needed child-pidfile kill procedure evidence

`tools/safe_run.py` already had a SIGTERM/SIGINT cascade handler that writes the status receipt
when `--status-receipt` is configured, but it did not emit a child pidfile or a child-only kill
command. That left operators tempted to use argv pattern matching such as `pkill -f "mode mem-probe"`,
which can match the safe_run wrapper argv itself.

Fix landed: `tools/safe_run.py` now supports `--child-pidfile` / `SAFE_RUN_CHILD_PIDFILE`,
derives a default pidfile next to `--status-receipt`, writes it atomically after child spawn,
fsyncs status receipt writes, and includes `child_pidfile`, `child_only_kill_command`, and
`operator_kill_rule` in every status receipt.

Test landed: `test_safe_run_external_sigterm_writes_killed_receipt_and_child_pidfile` starts a
real safe_run-wrapped sleep, waits for the running receipt and child pidfile, SIGTERMs the
wrapper, and verifies the killed receipt carries the external-signal kill action and child-only
kill command.

Boundary: if no status receipt or child pidfile is configured, safe_run still does not create
operator-facing evidence files by default. The class fix is the explicit receipt/pidfile surface,
not a retroactive receipt for old launches.

verdict_scope: APPARATUS / kill procedure and receipt custody.

### RR12-F3 - MEDIUM - One sampled AB1 trainer had guard placement after an MLX import check

The deterministic random AB1 sample was:

- `experiments/train_lane_12_v2_nerv_as_renderer.py`
- `experiments/train_substrate_pact_nerv_selector_v4_mlx_local.py`
- `experiments/train_substrate_rudin_floor_interpretable_ml.py`
- `experiments/train_substrate_wavelet.py`
- `experiments/train_nervdc_as_renderer.py`
- `experiments/train_substrate_pact_nerv_selector_v2_mlx_local.py`
- `experiments/train_substrate_tishby_ib_pure.py`
- `experiments/train_substrate_z6_predictive_coding_mlx.py`
- `experiments/train_substrate_pact_nerv_mamba.py`
- `experiments/train_substrate_z5_predictive_coding_world_model.py`

Nine had `assert_governed_admission(...)` immediately after argument parsing and before the
smoke/full/heavy call. `experiments/train_substrate_z6_predictive_coding_mlx.py` parsed args and
then guarded, but called `_require_mlx()` before parsing/guarding. I treated that as a placement
defect against the charter's stricter wording and moved `_require_mlx()` after
`assert_governed_admission(...)`.

The strict #254 scanner still reports 0 residuals / 150 scanned after the fix.

verdict_scope: INSTANCE / sampled AB1 placement.

## Clean Checks

### V4 ticket discriminator and path check

Checked `.omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json` directly:

- `argv_n32_arm_cap`: input cache ends with `gt_seg_cache.pt`, target cache ends with
  `gt_seg_cache.pt`.
- `argv_n32_arm_veh`: input cache ends with `tq1c_seg_cache.pt`, target cache ends with
  `gt_seg_cache.pt`.
- run dirs in the checked v4 ticket are under `.omx/research/ddm_mx1e_20260807/regen/...`.

Checked these regen/verdict artifacts for leaked old-path tokens `ddm_mx1d_20260807` and
`row1_v2_two_arm`; none were found:

- `.omx/research/ddm_mx1e_20260807/regen/probe_result.json`
- `.omx/research/ddm_mx1e_20260807/regen/launch_arm_cap/n32_metal/fire_guard_verdict.json`
- `.omx/research/ddm_mx1e_20260807/row1_v4_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json`

## Recall Evidence

Sources searched/read:

- Governing files: `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md` (byte-identical), `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Charter files: `.omx/tmp/codex_runs/rr12_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`.
- Memory quick pass: `/Users/adpena/.codex/memories/MEMORY.md` for `rr12`, `safe_run`, `#254`, `lane`, `frontier`.
- Corpus recall queries over `.omx/research`, `.omx/state`, `docs`, `src`, `experiments`, `tools`: `safe_run`, `mem-probe`, `_ticket_path_for_args`, `launch_ticket_v4`, `mode mem-probe`, `kill-on-sight`, `child-only`, `pidfile`, `SIGTERM`, `f06c8493f2`, `#254`.
- Canonical equations registry: `tools/list_canonical_equations.py --json`; relevant current anchors observed included `ddm_rr9_mem_probe_fire_protocol_v1` and `ddm_rr8_stage_rc_success_contract_v1`.
- AB1 receipt: `.omx/research/ddm_ab1_20260807/RECEIPT.md`.

Findings beyond the charter seeds:

- The v4 ticket discriminator state was clean in the checked artifact.
- The live #254 strict scanner was already clean; the sampled Z6 issue was a placement nuance,
  not a scanner residual.
- During the round, the mx1 source/test diff surface was co-mingled enough that I moved the
  RR11-F1 regression into a standalone file. Final rr12 status has no intended changes in
  `experiments/ddm_mx1_pr130_semantic_renderer.py` or `experiments/tests/test_ddm_mx1_memory_probe.py`.

What changed in plan:

- I moved the RR11-F1 regression into a new standalone test file so the review/commit surface
  is isolated from the mx1 source/test files.
- I kept the existing v4 ticket as an audited artifact and did not rewrite the pre-existing
  untracked `regen/` directory.

## Verification

Commands run:

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile tools/safe_run.py experiments/train_substrate_z6_predictive_coding_mlx.py experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py src/tac/tests/test_no_silent_failure_launch_hardening.py`
  - Result: passed.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py src/tac/tests/test_no_silent_failure_launch_hardening.py src/tac/tests/test_admission_coverage_gate.py`
  - Result: `33 passed in 10.21s`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY' ... check_heavy_witness_trainers_call_admission_guard(strict=True, verbose=True) ... PY`
  - Result: `OK (150 heavy trainer(s) scanned); strict_remaining 0`.
- v4 ticket parse/reverify script over `.omx/research/ddm_mx1e_20260807/launch_ticket_v4_fire_guarded.json` and three regen/verdict artifacts.
  - Result: `v4_ticket_reverify_ok`.
- `git diff --check -- tools/safe_run.py experiments/train_substrate_z6_predictive_coding_mlx.py experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py src/tac/tests/test_no_silent_failure_launch_hardening.py`
  - Result: passed.
- `tools/review_tracker.py mark-file ... --status reviewed` twice for the committed Python files.
- `tools/review_tracker.py policy-check ...` for committed Python files.
  - Result: 0 policy violations after re-marking newly scanned helpers/tests.

## Authority Boundary

This round is apparatus-only. It did not run `upstream/evaluate.py`, did not build or modify an
`archive.zip`, did not dispatch local/remote scorer work, and did not claim a contest or macOS
score row.

Own-vehicle frontier line remains:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.

## Commit Attempt

Serializer command attempted with post-edit SHA-256 declarations for:

- `tools/safe_run.py`
- `experiments/train_substrate_z6_predictive_coding_mlx.py`
- `experiments/tests/test_ddm_rr12_mx1_ticket_immutability.py`
- `src/tac/tests/test_no_silent_failure_launch_hardening.py`
- `.omx/research/ddm_rr12_20260807/ROUND12_FINDINGS.md`

Outcome: blocked before commit at `git add` with rc=128:

```text
[subagent-commit-serializer] git add failed (rc=128):
error: unable to create temporary file: Operation not permitted
error: experiments/train_substrate_z6_predictive_coding_mlx.py: failed to insert into database
error: unable to index file 'experiments/train_substrate_z6_predictive_coding_mlx.py'
fatal: updating files failed
```

Post-failure index check:

```text
git diff --cached --name-status
```

returned empty output. `HEAD` remained `f06c8493f2`.
