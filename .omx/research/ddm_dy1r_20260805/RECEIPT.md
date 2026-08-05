# ddm_dy1r Receipt: Scope-Law Rebase + Dy2 Tail-Law Closure

Captured: 2026-08-05T22:34:38Z

## RECALL EVIDENCE

Governing files read before the rebase work:

- `.omx/tmp/codex_runs/dy1r_prompt.md`
- `.omx/tmp/codex_runs/_common_contract.md`
- `PROGRAM.md`
- `CLAUDE.md`
- `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`

Recall and branch evidence:

- Memory quick pass found Pact operating/lane custody context in `MEMORY.md`.
- `git branch -a --list "*dy1*"` found `ddm/dy1_scope_law_resolver`.
- `git merge-base main ddm/dy1_scope_law_resolver` was
  `7ae1661e5836733db59b0fbfd394da8306e87910`.
- `git show --stat --oneline ddm/dy1_scope_law_resolver --` showed dy1 commit
  `a9eac92166 ddm_dy1: add scope-law resolver [no-triality] [p0-ledger-ok]`.
- `.omx/research/ddm_dy2_20260805/RECEIPT.md` confirmed dy2 tail-average
  semantics: explicit anchor, fail-closed misuse guards, and `ema_tail_average`
  gate-basis label.

Searches used for the merge:

- `rg -n "FORMALIZATION_PENDING" .omx/research/ddm_dy2_20260805 .omx/tmp/codex_runs/dy1r_prompt.md`
- `rg -n "_resolve_scope_law\\(\"jd3_|_resolve_scope_law\\(\"jd1_plateau" experiments/train_tr1_partition_renderer_mlx.py`
- `rg -n "scope_laws|ticket_payload_hash|refuse_declared_vs_resolved" ...`
- `.venv/bin/python tools/register_jd1_plateau_tail_average_ema_20260805.py --dry-run`

## Rebase Result

dy1 was manually ported onto current main because the trainer/spec/regenerator
seams had moved under dy2 and jd4 work. I did not launch scorer, MLX, training,
or archive work.

Conflict resolutions:

- Trainer collision: preserved dy2's explicit-anchor tail-average mode and
  added scope-law resolution around it. Tail updates now resolve
  `jd1_plateau_tail_average_ema`; JD3 stage EMA and realized-hold decisions
  resolve at consumption time.
- Spec collision: preserved dy2's `lever_jd1_plateau_tail_average_ema`, added
  optional `scope_laws`, and made ticket hashes bind those declarations.
- Launcher/regenerator collision: preserved current jd4 regeneration behavior,
  added scope-law validation/hash checks, and attached JD3 default declarations
  plus the dy2 tail declaration when the tail mode is present.
- Dy2 receipt collision: removed the queued tail-law waiver and pointed dy2 at
  `jd1_plateau_tail_average_ema_v1`.

Historical dy1 receipt files were used as recall evidence, but this dy1r receipt
is the current landing authority after the dy2/jd4 reconciliation.

## Merge Gates

Gate 1, inertness production wiring: PASS.

- JD3 stage EMA is consumed in
  `experiments/train_tr1_partition_renderer_mlx.py:4538`.
- JD3 pose retreat and max retreats are consumed at
  `experiments/train_tr1_partition_renderer_mlx.py:4213` and `:4222`.
- JD3 realized floor and margin are consumed at
  `experiments/train_tr1_partition_renderer_mlx.py:5180` and `:5183`.
- Dy2 tail-average live weight is consumed at
  `experiments/train_tr1_partition_renderer_mlx.py:4918`.
- Inertness positive control is tested in
  `src/tac/witness_dsl/tests/test_scope_laws.py`.

Gate 2, geometry-hash keying: PASS.

- `scope_law_geometry_hash` is defined in `src/tac/witness_dsl/scope_laws.py:193`.
- `jd3_stage_ema_decay` requires `run_geometry_hash` in
  `src/tac/witness_dsl/scope_laws.py:378`.
- The runtime stage-EMA resolver passes steps/epoch, horizon, and window at
  `experiments/train_tr1_partition_renderer_mlx.py:4538`.
- The hash split is tested by
  `test_scope_law_geometry_hash_keys_resolved_values_by_window_geometry`.

Gate 3, declared-vs-resolved EMA refuse: PASS.

- The fail-closed helper is
  `experiments/train_tr1_partition_renderer_mlx.py:3203`.
- The runtime reanchor uses it at
  `experiments/train_tr1_partition_renderer_mlx.py:4549`.
- `src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py:144` verifies no-op
  matching literals and refusal on mismatch.

## Tail-Law Registration

Registered via the canonical helper:

```bash
.venv/bin/python tools/register_jd1_plateau_tail_average_ema_20260805.py --dry-run
.venv/bin/python tools/register_jd1_plateau_tail_average_ema_20260805.py
```

Result: `jd1_plateau_tail_average_ema_v1` appended to
`.omx/state/canonical_equations_registry.jsonl` with `agent=codex` and
`subagent_id=ddm_dy1r`.

The scope-law ticket declaration now includes a LawRef to that equation id, and
`validate_ticket_scope_laws` refuses stale/missing LawRef declarations.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py experiments/ddm_jd1_ticket_regenerate.py src/tac/witness_dsl/scope_laws.py src/tac/witness_dsl/tests/test_scope_laws.py src/tac/witness_dsl/spec_tr1_renderer_20260728.py tools/launch_tr1_run.py src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py src/tac/canonical_equations/jd1_plateau_tail_average_ema_20260805.py tools/register_jd1_plateau_tail_average_ema_20260805.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py
```

Passed:

```bash
.venv/bin/python -m pytest src/tac/witness_dsl/tests/test_scope_laws.py src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py src/tac/tests/test_ddm_jd1_ticket_regenerate.py
```

Result: `31 passed in 0.66s`.

Passed:

```bash
git diff --check -- <dy1r files>
```

Review tracker:

- `tools/review_tracker.py scan`
- `tools/review_tracker.py mark-file <file> --status reviewed` twice for all
  nine touched Python files.

## Follow-Ons

FOLDED:

- dy2 tail-law formalization waiver: closed by `jd1_plateau_tail_average_ema_v1`
  and dy2 receipt update.

FIRED:

- canonical registry append for `jd1_plateau_tail_average_ema_v1`.

QUEUED-WITH-FIRE-ORDER:

1. At the jd4/tp1 boundary, MAIN selects a concrete tail anchor epoch from the
   tp1 Case-0 detector.
2. MAIN compiles the A/B through DSL with `lever_jd1_joint_pose_finish(...)` and
   `lever_jd1_plateau_tail_average_ema(anchor_epoch=<case0_epoch>)`.
3. On a Metal-capable host, run the true JD1 tail-state
   `save_checkpoint` -> `load_checkpoint` resume round-trip before launch.

## Post-Edit Hashes

| Path | SHA-256 |
|---|---|
| `experiments/train_tr1_partition_renderer_mlx.py` | `ca479b6ecbb59070da5714dde5ec2d0c096aa81a4e1034c34c84321d5aedcd61` |
| `experiments/ddm_jd1_ticket_regenerate.py` | `42d126db87595a0a62958312e517681bdd40c929142fedadf8f70b5ce9ce5ef7` |
| `src/tac/witness_dsl/scope_laws.py` | `a9311d28aab669dfc628f459459cba9e00483fbf7d307a00c4c40cc838c15a9f` |
| `src/tac/witness_dsl/tests/test_scope_laws.py` | `e9486f45c0db39f73587e73583ce62455896fae486dd9b8a53894b1ebe586db1` |
| `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` | `880749c354326ac27f0edb8784e80cc79a52ebd7f39a4661fbd4b44efcd726cd` |
| `tools/launch_tr1_run.py` | `3116388be841b288a4ab62a0b06d6b78927f3f96b633c1efae2dd0e075b201e4` |
| `src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py` | `4c3b28eb5cecc030cb04c424926e37cd0b94e047233222e4b1ea3671dc112703` |
| `src/tac/canonical_equations/jd1_plateau_tail_average_ema_20260805.py` | `fb43a3902b0f239504ea19f38ab00b5d5275a6d3f0d4441f43d9fb23426d0739` |
| `tools/register_jd1_plateau_tail_average_ema_20260805.py` | `b8b482d9a99e96e7783ffed13e7dea0f59b6d47b0c64fbfd642056920cc0c11e` |
| `.omx/research/ddm_dy2_20260805/RECEIPT.md` | `f82a1bbb9a035faf976d0b91a65db5fba4420abe8fef8c49f9039747fd3b8abf` |
| `.omx/research/ddm_dy2_20260805/NEXT_IF_RESUMED.md` | `c217b561d0ec9a0ca27134a294c4f9d057889e851e10333624d6e1eb47d2cfc6` |
| `.omx/state/canonical_equations_registry.jsonl` | `1f320908b6912ac45bf584d6fc21f4d94a59f525df8736c8488829b16a0bde12` |

## Boundaries

- MEASURED: resolver determinism, ticket-hash binding, LawRef declaration
  validation, tail-law equation construction, and dy2/JD1/ticket-regenerator
  focused tests.
- NOT MEASURED: scorer output, MLX training, archive bytes, contest CPU/CUDA
  score, or pointer movement.
- No scorer slot used.

Own-vehicle frontier line: S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
