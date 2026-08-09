# FX2 lift custody and adapter honesty cures

The exact pointer did not move. No scorer, evaluator, archive build, Metal run,
CUDA run, or score measurement ran in this unit. The live PR130 base remains
`S = 0.172141297491896447` at 191,052 B `[contest-CUDA, DALI GT, n600]`.

## 1. Per-file custody after cure

The reconstructive custody gate reads each source body from intake Git object
`e34f31bc4969042c0051ac81aa3c56884419a231`, recomputes its SHA-256, applies
only the enumerated adaptation, strips only the accounting header from the
lifted copy, and requires exact byte equality. It also requires the discovered
set of original-backed files to equal the enumerated set. This is not a
manifest-literal test.

All headers now distinguish:

- `source_repo_head`: the intake commit whose exact `source_path` bytes the
  `source_sha256` authenticates;
- `lifted_at_head`: the commit at which our custody copy was made;
- `adaptations`: the only permitted body patch, or `none`.

| lifted file | intake source | state after cure | permitted body adaptation | guard |
| --- | --- | --- | --- | --- |
| `lifted/evaluate_semantic_quantization.py` | `code/evaluate_semantic_quantization.py` | declared, exact body | none | reconstructive custody test |
| `lifted/semantic_renderer_oracle.py` | `code/semantic_renderer_oracle.py` | declared, exact body | none | reconstructive custody test |
| `lifted/train_semantic_full.py` | `code/train_semantic_full.py` | declared adaptation | `governed_admission_guard_after_argparse` | reconstructive custody test |
| `lifted/train_semantic_quantized.py` | `code/train_semantic_quantized.py` | declared adaptation | `governed_admission_guard_after_argparse` | reconstructive custody test |
| `pose/lifted/carrier_codec.py` | `code/carrier_codec.py` | declared, exact body | none | reconstructive custody test |
| `pose/lifted/learned_pose_carrier_oracle.py` | `code/learned_pose_carrier_oracle.py` | declared, exact body | none | reconstructive custody test |
| `pose/lifted/pack_semantic_pose.py` | `code/pack_semantic_pose.py` | declared, exact body | none | reconstructive custody test |
| `pose/lifted/pose_basis_oracle.py` | `code/pose_basis_oracle.py` | declared, exact body | none | reconstructive custody test |
| `pose/lifted/refine_pose_coeff_codes.py` | `code/refine_pose_coeff_codes.py` | declared, exact body | none | reconstructive custody test |
| `pose/lifted/repack_carrier.py` | `code/repack_carrier.py` | declared, exact body | none | reconstructive custody test |
| `pose/lifted/search_pose_coeff_cpu.py` | `code/search_pose_coeff_cpu.py` | declared, exact body | none | reconstructive custody test |
| `pose/lifted/train_pose_carrier_full.py` | `code/train_pose_carrier_full.py` | declared adaptation | `governed_admission_guard_after_argparse` | reconstructive custody test |
| `pose/lifted/__init__.py` | local scaffold, no intake source | declared local | local package initializer | denominator assertion |

Post-cure denominator: **13/13** Python files classified; **12/12**
original-backed source hashes authenticate the exact pinned source bytes;
**9/12** have no body adaptation and reconstruct exactly; **3/12** reconstruct
exactly after the one enumerated two-line admission patch; **0/12** remain
silently drifted. The pose vendor manifest also records the separated heads,
SHA scope, and adaptation for **8/8** entries and points to the reconstructive
test as the body authority.

## 2. Adapter selection cure

`reference-sparse` is now the default mode on CPU, CUDA, and MPS. It constructs
`nn.Embedding(..., sparse=True)` and the borrowed `RowLocalSparseAdam`. On MPS,
reference mode fails closed unless the public Torch version is exactly `2.10.0`,
the runtime covered by the pinned native receipt.

`dense-adapter` is a named CLI opt-in. Device type never selects it, and there
is no automatic fallback. Startup emits the selected optimizer event. A legacy
full-state checkpoint without a stored mode is interpreted from the mechanism
the old code actually used: old MPS state is dense-adapter; old CPU/CUDA state
is reference-sparse. Resuming old MPS state under the new sparse default is
refused until the caller explicitly selects `dense-adapter`.

The active wrapper has one production optimizer-selection call site. It now
passes the parsed mode explicitly: **1/1** active selection sites cured. The
previous `use_sparse = device.type != "mps"` selector is absent.

### Persisted provenance schema

Every full-state step checkpoint, full-state latest checkpoint, deploy latest
checkpoint, deploy best checkpoint, final checkpoint, and final JSON result
contains `execution_provenance` with:

| field | persisted meaning |
| --- | --- |
| `optimizer_class` | actual fully qualified runtime class |
| `row_local_mode` | `reference-sparse` or `dense-adapter` |
| `gradient_representation` | actual `sparse` or `dense` embedding gradient |
| `selection_event` | reference default or explicit adapter opt-in |
| `fallback_event` | `none`; there is no automatic fallback path |
| `fallback_policy` | `automatic_fallback_forbidden` |
| `torch_version` | public runtime version |
| `git_sha` | repository HEAD at execution |
| `argv` | exact Python `sys.argv` list |
| `native_probe_receipt` | path, expected and observed SHA-256, validation status, scope, and `score_claim=false` |
| `score_claim` | always `false` |

The native receipt identity is
`/Volumes/VertigoDataTier/pact/ddm_pq1_probe_20260809/probe_torch2100_pinned.json`,
SHA-256
`32ce0585d070fd578bea563f94b33fffe6e000b8cc608f827d4fcb5319893ec3`.
A present receipt with a different hash is refused. A missing receipt is
recorded as `missing_at_run`; its expected identity is still persisted.

All five model/full-state checkpoint destinations now use same-directory
temporary files plus atomic replace. This closes interrupted-write corruption;
it does not by itself prove uninterrupted-versus-resumed end-to-end trajectory
equality.

## 3. Trajectory comparison

MEASURED `[macOS-CPU advisory; optimizer-only]`: the CPU harness compares the
borrowed sparse reference and dense adapter for **64/64** scheduled steps, not
the former two-step control. Each step uses a deterministic repeated-ID batch,
gradient clipping, the actual row-local optimizers, and matched cosine
schedulers.

- weights: exact at **64/64** step boundaries;
- `row_step`, `exp_avg`, and `exp_avg_sq`: exact at **192/192** tensor checks;
- scheduler learning rate: exact at **64/64** step boundaries;
- per-step untouched rows: bit-identical for both paths at **128/128** checks;
- explicit-selection tests: sparse reference default and dense opt-in both
  construct the named mechanism;
- MPS runtime guard: a simulated Torch 2.9.0 reference selection is refused
  before MPS allocation.

This is a mechanism result, not a scorer result, training-quality result,
Metal-parity result, archive result, or score.

## 4. Sibling sweep

The primary denominator was both lifted trees: **13/13** Python files. The
broader search covered the live `src/tac/pr130_lift` tree at the sweep point:
**31 Python files and 1 JSON manifest**, including concurrently present FX3
work, for source-head/hash fields, optimizer selection, result/checkpoint
mechanism fields, device-derived sparse selection, and direct checkpoint saves.

Findings:

1. The **12/12** original-backed lifted headers and **8/8** pose-manifest rows
   are cured and reconstructively guarded.
2. The active adapter route has **1/1** selection sites and **5/5** checkpoint
   destinations cured. No sibling device-derived dense selection remains in
   the searched production scope.
3. `mlx_semantic_renderer.py` still uses the older single `source_head=2f94596`
   wording beside four original-source hashes. It is a port, not a claim that
   its current body equals intake, and its `ours` section declares the port;
   nevertheless the hash scope remains less explicit than the cured headers.
   I did not land that header change because a concurrent FX3 owner had
   uncommitted semantic-checkpoint changes in the same file. Staging the whole
   file would have captured unrelated work.
4. Direct `torch.save` calls remain in unadapted borrowed source bodies. Those
   are faithfully reconstructed intake mechanics, not hidden local
   substitutions. The active resumable wrapper now uses atomic saves and full
   provenance; the borrowed direct executables are not relabeled as resumable.

Verdict scope: **FORMULATION**, the current PR130 lifted/wrapper tree. This is
not a claim that every vendored or borrowed tree elsewhere in the repository
has the same custody guard.

## 5. Verification

- `PYTHONPATH=src .venv/bin/python -m pytest --timeout=180` over the four FX2
  custody/adapter/resume/manifest modules: **21 passed**.
- Ruff over the changed active wrapper, adapter, and FX2 tests: **all checks
  passed**.
- `git diff --check -- src/tac/pr130_lift`: passed.
- Full shared `src/tac/pr130_lift/tests`: **38 passed, 3 failed**. All three
  failures are in concurrently added, untracked
  `test_fx3_semantic_qat_resume.py` against concurrently added FX3 semantic
  resume code; none touches the FX2 files or assertions. They are not promoted
  to an FX2 failure, and this unit did not edit or capture those files.
- An earlier repeat hit the already-recorded shared-environment Torch/SymPy
  import defect: the first optimizer construction timed out while importing a
  SymPy bytecode file, poisoning later imports in that process. A fresh
  standalone SymPy/Torch import passed, and the terminal 21-test FX2 run then
  completed in 1.77 s. The failed process is retained as environment evidence,
  not hidden.

## 6. Recall evidence

Queries included `RowLocalSparseAdam`, `RowLocalDenseAdam`, `native sparse`,
`dense adapter`, `train_pose_carrier_full`, `pr130_lift`, source-head/hash
fields, and checkpoint/result writers across `.omx/research/`,
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `.omx/state/`, the task ledger,
the canonical equation registry, Git history, the intake Git objects, and the
current source/test tree.

Beyond the charter text, the most useful sources were:

- `ddm_pp2_20260809T121528Z/PP2_FINDINGS.md`: the 60-family device census and
  the row-clock mechanism risk;
- `ddm_pq1_20260809T125541Z/PQ1_FINDINGS.md` and
  `MAIN_METAL_RECEIPT.md`: the exact pinned runtime and real-MPS sparse PASS;
- `ddm_pr130_reproduce_20260809/RR4_CARRIER_LIFT_AUDIT.md`: the full 13-file
  denominator, checkpoint atomicity residual, and test omissions;
- `main_hot_state.md`: task `#995` remains training-owed and score-free here.

What changed in the plan: the already-passed native sparse receipt made sparse
the reference default rather than a speculative MPS option; PP2's row-clock
analysis made the longer harness compare optimizer state and schedulers, not
only final weights; RR4's checkpoint finding caused the provenance landing to
use atomic writes. The canonical equation search did not find an equation that
settles long-horizon adapter equivalence or full carrier training.

## 7. Ranked residuals and falsifiers

1. **End-to-end resume equality remains open.** The optimizer-only 64-step
   trajectories agree, and checkpoint writes are atomic, but the full wrapper's
   uninterrupted and split-resume runs have not been compared through model,
   optimizer, scheduler, RNG/order/cursor, history, best-state, and final result.
   Falsifier: one deterministic CPU real-row-local harness produces exact final
   equality across all those fields after a mid-run save/load.
2. **Full MPS carrier training remains open.** The native sparse path is proven
   only for the pinned two-step/four-row real-MPS receipt, while this unit's
   longer comparison is CPU-only. Falsifier: a governed terminal n600 run on
   pinned Torch 2.10.0 records `reference-sparse`, zero fallback, terminal
   status, every periodic/stage checkpoint, and its full provenance.
3. **The MLX semantic port header remains semantically older.** Its body is
   explicitly a port, but the four source hashes still sit under one
   `source_head` field. Falsifier: after the concurrent owner lands, split that
   header into `source_repo_head`, `lifted_at_head`, and explicit hash scope,
   then add it to a port-provenance test without altering the owner's code.

## 8. Could not check / why

- No Metal device was available to this arm; no new MPS execution was run.
- No scorer slot was owned; no scorer or n600 evaluation was run.
- No CUDA host was used, and MPS/CPU mechanism tests are never score authority.
- The full 4,000-step pose-carrier run remains absent in the searched receipts.
- The MLX semantic port header could not be landed safely because another owner
  had concurrent uncommitted changes in that same file.
- The three concurrent FX3 full-suite failures were not diagnosed or fixed;
  doing so would exceed this charter and capture another owner's work.

## 9. Landing status

The required serializer was invoked with the 21 intended files only, every
post-edit SHA-256, every tracked file's HEAD/base SHA-256, `base=new` for the
two new files, `[no-triality] [p0-ledger-ok]`, and no attribution trailer. It
failed during `git add` before staging:

```text
error: unable to create temporary file: Operation not permitted
error: src/tac/pr130_lift/__init__.py: failed to insert into database
fatal: updating files failed
```

The shared staged index remains empty. All FX2 changes and the report remain
uncommitted working-tree artifacts; no commit or landing claim is made.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN on a checkout with a writable Git
  object database; consumer store: the current 21-file FX2 working-tree change
  set and repository history; fire trigger: Git object writes are available,
  then re-run the exact focused verification, refresh every post-edit SHA, and
  commit only the enumerated FX2 files through
  `tools/subagent_commit_serializer.py`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: task `#995` pose-port owner; consumer
  store: `src/tac/pr130_lift/tests/` and
  `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/`; fire trigger: before
  authorizing the full 4,000-step run, add and pass exact uninterrupted-versus-
  resumed full-wrapper equality with the actual row-local optimizer state.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: task `#995` / governed local-training
  owner; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/` and the PR130 task
  ledger; fire trigger: resume equality, storage preflight, governed launcher,
  pinned Torch 2.10.0, and checkpoint custody all pass, then run the n600
  reference-sparse carrier training to terminal status.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: FX3 semantic-port custodian; consumer
  store: `src/tac/pr130_lift/mlx_semantic_renderer.py` and its provenance test;
  fire trigger: the concurrent FX3 edit is committed or relinquished, then
  split the port header's head/hash semantics without capturing unrelated code.

## LIVE-HYPOTHESES

- Native sparse should remain usable for the full MPS run because the exact
  embedding, coalesced COO, row-clock update, and zero-fallback path passed on
  the pinned runtime; long-horizon accumulation and full-graph behavior remain
  the untested parts.
- The dense adapter should remain a valid explicit portability option because
  64 scheduled repeated-ID CPU steps match reference weights, all row-local
  state, learning rates, and untouched-row identity exactly; device-level MPS
  numerical accumulation remains unmeasured.
- Full-wrapper resume equality is plausible because optimizer state shapes and
  legacy mode interpretation are now explicit and atomic, but equality must be
  tested through RNG/order/cursor and best-state selection before a long run.

## DEAD-ENDS

- Do not restore device-derived `use_sparse = device.type != "mps"`; it silently
  substitutes the adapter despite a passing pinned native-sparse receipt.
- Do not treat a log line as the cure. Selection must be explicit and the
  actual optimizer class, representation, runtime, fallback event, argv, Git
  SHA, and receipt identity must travel with results and checkpoints.
- Do not replace the reconstructive custody test with manifest literals. A
  literal can agree while the body drifts; expected bodies must be rebuilt from
  pinned intake bytes plus enumerated patches.
- Do not relabel the 64-step CPU mechanism test as Metal parity, training
  convergence, scorer evidence, archive evidence, or a score.
- Do not stage `mlx_semantic_renderer.py` while another owner has uncommitted
  changes in the same file; that would silently capture unrelated work.
