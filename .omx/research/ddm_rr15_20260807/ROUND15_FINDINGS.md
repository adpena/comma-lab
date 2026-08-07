# ddm_rr15 Round 15 Findings

status: CLEAN
round: 15
clean_pass_counter_after_round: 1/3
axis: apparatus / scorer-free
score_claim: false
frontier_moved: false
tags: [no-triality] [p0-ledger-ok]

## Counts First

| item | count / status |
|---|---:|
| Findings filed | 0 |
| Fixed inline | 0 |
| Clean adjudications | 5 |
| Metal / scorer / archive launches | 0 |
| Python review tracker passes | 2 per reviewed .py |

## Verdict

Round 15 is CLEAN. I reviewed only the small post-round-14 diff named by the
charter:

- `29741ff843` (`ddm_rr14`: guard `reason` alias, resume-key PASS test,
  MX1H NPZ/history fail-closed tests, and ROUND14 findings)
- `7a168e546877995d2c9d4d7e3ef3819daa8c3f38` (SB3 lazy MLX runtime gate
  and named no-Metal xfail wiring)
- `ROUND14_FINDINGS.md` and `SB3_FINDINGS.md`

No code or artifact fix was required. This is the first clean pass after
rounds 13 and 14 reset the counter; the convergence counter advances to `1/3`.

## Clean Adjudications

### R1 - `reason` alias stays synchronized with `reason_code`

Clean. `tools/mx1_fire_guard.py` centralizes all guard verdict emission through
`_verdict(...)`, including pass, refusal, and `guard_internal_error` fallback
paths. `_verdict(...)` emits both:

- `reason_code: reason_code`
- `reason: reason_code`

I did not find a second fire-guard verdict construction site that could emit
one field without the other. The lower-level check rows also have a `reason`
field, but they are nested check diagnostics, not the top-level verdict schema
operator consumers read.

### R2 - Resume-key PASS test uses the true keyed resume receipt shape

Clean. The round-14 PASS test constructs `argv_n32_arm_cap_resume`, sets the
ticket's keyed `mem_probe_receipt_paths[resume_key]`, and writes the receipt at:

`launch_arm_cap/n32_metal/mem_probe_resume/mem_probe_receipt.json`

That matches the current ticket generator's resume path rule in
`experiments/ddm_mx1_pr130_semantic_renderer.py`: resume keys use the same arm
run directory, but their mem-probe receipts live under `mem_probe_resume/`.

The test is synthetic, but not a fake simplification of the reviewed binding:
it still runs the CLI guard, uses the current host fingerprint, requires the
receipt schema/status/clearance, validates samples and software memory-cap
summary, compares the effective GPU microbatch footprint, and asserts the
verdict binds to `argv_n32_arm_cap_resume` plus the resume-specific receipt
path. It intentionally does not run Metal or resume training.

### R3 - MX1H fail-closed tests exercise real loader refusal points

Clean. The added tests in `experiments/tests/test_ddm_mx1_memory_probe.py`
target the real strict paths:

- `_load_mlx_npz_checkpoint_for_torch(...)` refuses missing and unexpected
  `param::*` tensor keys by comparing the NPZ parameter set against the torch
  model state dict.
- `_history_row_at_step(...)` refuses when the checkpoint `step` has no exact
  history row carrying `d_seg_batch`.

These are test-evidence additions for code that was already strict in the
MX1H implementation. They do not fabricate a torch-verdict row or promote an
MLX proxy number.

### R4 - SB3 lazy MLX gate does not mask the original SIGBUS class

Clean. `_require_mlx_runtime_or_xfail()` imports MLX lazily and performs a
one-element `mx.array([0.0], dtype=mx.float32)` allocation. On this host it
xfails only when the RuntimeError string contains:

`[metal::load_device] No Metal device available`

That xfail reason is named `#856 known-red environment/MLX-gating`, so it is
distinguishable from the original #983 SIGBUS class. A different RuntimeError
is re-raised, and a true SIGBUS would not be converted into an xfail by this
predicate. The probe is tiny and test-local; it does not touch the live run
directory and does not allocate scorer state.

Boundary: on a real Metal-backed test host, this gate only proves a minimal MLX
allocation before entering the target test. It is not a proof that every later
adapter operation is SIGBUS-free. That is correct for this cure: no-Metal
environment failures are xfailed, while real Metal-backed failures remain
visible.

### R5 - Live fire-chain behavior was not broadened

Clean. The reviewed diffs are tests plus the guard verdict alias. I did not
find a behavior change to the live fire chain beyond:

- the guard verdict now also emits the top-level compatibility alias `reason`;
- tests now cover resume PASS and MX1H fail-closed negative paths;
- target MLX runtime tests now xfail only the no-Metal environment class.

No live run directory, scorer slot, archive builder, remote dispatch, or
`upstream/evaluate.py` path was touched by this round.

## Assumption Challenge

Shared assumption: this recursive review can certify the immediate fire-chain
apparatus without re-litigating whether the MX1 Row-1 PR130-derived renderer is
the right frontier vehicle.

Challenge result: holding that assumption is correct for round 15. Violating it
would broaden the charter away from the convergence objective and would not
produce an exact score row. The useful attack here is narrower: if the fire
chain is allowed to resume, it must fail closed on stale or mismatched memory
receipts and must not hide no-Metal test-environment failures as proof of a
real Metal path. Those properties survived this round.

## RECALL EVIDENCE

| scope | query / source | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | Read `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md` no-fake and goal sections, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the rr15 charter, and the common contract. | Hot state supersedes the common contract's older frontier line: own-vehicle frontier is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`, contest pointer borrowed/unmoved. It also records mx1g/rr13 as landed and the resume procedure as gated by a fresh mem-probe. | Kept the review scorer-free, no-Metal, and pointer-honest; used the live hot-state frontier line in this receipt. |
| Memory registry | Searched `/Users/adpena/.codex/memories/MEMORY.md` for `ddm_rr15`, `rr15`, `20260807`, DDM receiver/public-evaluator surfaces, and fire-chain terms. | No direct rr15 memory entry. Relevant prior-memory guidance says Pact DDM work must use typed denominators/source-verified queue ownership and re-read live state before action. | Treated the charter as the local authority but refreshed live board and source instead of relying on stale summary state. |
| Prior MX1 / RR / SB3 corpus | Searched `.omx/research`, `.omx/state`, docs, reports, tools, experiments, and src for `reason_code`, `reason alias`, `mem_probe_resume`, `resume-key`, `lazy Metal`, `usable Metal`, `SIGBUS`, `MX1H`, `history-step`, `REQUIRES_FRESH_MEM_PROBE`, and `fire_guard`. | Found mx1g's first-class resume keys and fresh-receipt projection, rr13/rr14 reset context, SB3's no-Metal diagnosis, AH1/CQ1 routing showing rr9/rr10 guard lineage, and stale mx1d/mx1e ticket shapes that should not be used as current authority. | Reviewed the round-14 resume PASS test against the current generator's `mem_probe_resume` path and reviewed SB3 as an environment-gate cure, not a production-adapter cure. |
| Canonical equations | Ran `.venv/bin/python tools/list_canonical_equations.py --json` filtered for `mx1`, `memory`, `resume`, `Metal`, `fire_guard`, `SIGBUS`, and `history`. | The relevant current law is `ddm_rr9_mem_probe_fire_protocol_v1`: safe-run admission is not a substitute for a passed Metal mem-probe receipt. No SB3-specific equation required update. | Preserved the fail-closed receipt-gate lens and made no equation-registry edit. |
| Current source and commits | Read `git show 29741ff843`, `git show 7a168e546877995d2c9d4d7e3ef3819daa8c3f38`, `ROUND14_FINDINGS.md`, `SB3_FINDINGS.md`, and the current implementations/tests for the reviewed surfaces. | The code matched the prior findings' claims. I found no top-level verdict alias desync, no simplified resume receipt path in the PASS test, and no xfail predicate that would catch non-no-Metal failures. | Filed CLEAN instead of adding a fix. |

Scoped negative: in the searched source/artifact scope, I did not find a
post-round-14 defect that changes behavior beyond tests or that weakens the
live fire-chain guard.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/tests/test_mx1_fire_guard.py experiments/tests/test_ddm_mx1_memory_probe.py -q
PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -X faulthandler -m pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_mlx_runtime_gate_recognizes_no_metal_error src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_live_birth_hysteresis_probe_restores_model_state src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_live_birth_survival_writes_four_arm_rows_when_birth_not_accepted src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_private_smoke_defaults_to_full_target_hydration_for_hard_pairs src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_private_smoke_forwards_explicit_pr95_curriculum_total_epochs src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_private_smoke_generates_startup_section_telemetry_for_qat_terms src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_execute_runs_training_archive_and_receiver_proof -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py
git diff --check -- tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py experiments/tests/test_ddm_mx1_memory_probe.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py
```

Results:

- Guard and MX1H tests: `30 passed`.
- SB3 focused MLX gate tests: `1 passed, 6 xfailed`.
- SB3 still emits the known no-Metal nanobind atexit RuntimeError after xfail;
  this is the same environment class SB3 documented, not a new failing test.
- Py compile: passed.
- Ruff: passed.
- Diff check: passed.
- Review tracker: marked `tools/mx1_fire_guard.py`, `tools/tests/test_mx1_fire_guard.py`,
  `experiments/tests/test_ddm_mx1_memory_probe.py`, and
  `src/tac/tests/test_compact_renderer_mlx_spine_runner.py` reviewed twice;
  policy checks reported 0 violations for all four files.

## Boundaries

- No Metal training, scorer slot, archive build, remote dispatch, or
  `upstream/evaluate.py` run was performed.
- No live run directory was touched.
- No `upstream/` file was touched.
- No Python file was edited in this round.
- Existing dirty work in the shared tree was left intact and unstaged.
- Follow-ons: none from this clean round. The next recursive review round is
  round 16 and starts from counter `1/3`.
- Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`;
  contest pointer remains `0.19108` borrowed/unmoved.
