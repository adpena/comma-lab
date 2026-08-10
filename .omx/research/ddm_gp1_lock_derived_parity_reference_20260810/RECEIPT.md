# ddm_gp1 lock-derived parity reference receipt

## Verdict

**IMPLEMENTATION READY FOR GOVERNED LANDING; SCORE ROW NOT FIRED AND POINTER UNMOVED.**

The missing-`upstream/.venv/bin/python` branch now derives its parity reference from an exact
`uv export --frozen` of the declared upstream dependency group, evaluates requirement markers in
the interpreter that will run `upstream/evaluate.py`, and refuses every unresolved or ambiguous
case. The historical present-venv branch remains intact. Both Modal dispatchers declare their
existing axis-specific group.

No scorer or Modal job was launched by this arm. The observed candidate remains
`score_claim=false`; its recomputed value is recorded only as a refused
`[contest-CUDA env-mismatch advisory]` result and is not a score claim.

## RECALL EVIDENCE

Searched before implementation:

- Full-text queries across `.omx/research/`, the canonical research indexes, the sub-0.15 DAG,
  `.omx/state/main_hot_state.md`, and task/receipt surfaces for `gp1`, `environment parity`,
  `locked_env_probe`, `upstream-python`, `uv export`, `task #1004`, and `task #1005`.
- Canonical equation registry via
  `.venv/bin/python tools/list_canonical_equations.py --json`; no dedicated environment-parity
  equation was found in that bounded registry.
- Source ownership through `_resolve_evaluate_python`, `_record_provenance`, `main`, and
  `_run_upstream_evaluate` in `experiments/contest_auth_eval.py`.
- Sister implementation and tests in `src/tac/deploy/modal/locked_env_probe.py`.
- Refusal receipts in `.omx/research/ddm_uh1_20260805/`, `.omx/research/ddm_us2_20260805/`, and
  `.omx/research/ddm_main_paired_eval_20260810/`.

Findings beyond the charter seeds changed the implementation in two ways:

1. `auth_eval_python` is not wrapper-only metadata: the selected path becomes argv[0] for the
   actual `upstream/evaluate.py` subprocess. The parity branch therefore controls the real scorer
   interpreter.
2. The already-fired call `fc-01KZNZNYECSJP143ZHZ14452RX` had been harvested while this arm was
   working. Its provenance shows that the command declared
   `/opt/upstream-locked-venv/bin/python`, but `Path.resolve()` changed the actual
   `auth_eval_python` to `/usr/local/bin/python3.11`. That bypassed `pyvenv.cfg` and measured the
   image packages instead of the locked venv. The patch therefore preserves absolute venv launcher
   paths without dereferencing their symlinks, with a regression test.

An older UH1 receipt recommended passing `--upstream-python`; the 2026-08-10 hardening and this
source trace supersede that recommendation as proof. The flag selects an interpreter but never
proves parity.

## Implementation

- `experiments/contest_auth_eval.py`
  - Adds the explicit, default-free `--upstream-uv-group` declaration.
  - Runs the exact frozen export required by the charter.
  - Evaluates universal requirement markers inside the selected evaluation interpreter.
  - Requires exactly one surviving exact `==` row for `torch`, `torchvision`, `timm`, and
    `numpy`; zero, multiple, malformed, or non-exact rows refuse with typed reasons.
  - Records the group, exact export argv, uv executable/version, lock path and before/after SHA,
    marker environment, selected rows, and failure details.
  - Hashes the lock before and after export plus marker resolution and refuses any mutation.
  - Preserves venv launcher symlinks so the selected venv actually supplies the scorer packages.
- `experiments/modal_auth_eval.py` passes `--upstream-uv-group cu128` through its existing
  `UPSTREAM_UV_GROUP_CUDA` constant.
- `experiments/modal_auth_eval_cpu.py` passes `--upstream-uv-group cpu` through its existing
  `UPSTREAM_UV_GROUP_CPU` constant.
- Behavioral controls live in `src/tac/tests/test_contest_auth_eval.py` and
  `src/tac/tests/test_modal_auth_eval.py`.

No version literal is embedded in the implementation. Nothing under `upstream/` was edited.

## Executed controls

The real lock SHA was
`eca4542ad8d21354fd1f2bada74e8659329c0176b17f1ae808e04e023674231f` before and after every
control below.

- **Negative control, real match:** the real upstream CPU lock export compared with the real
  `upstream/.venv/bin/python` through a synthetic upstream root without `.venv`; no mismatch.
- **Positive control, real mismatch:** the repository `.venv/bin/python` compared with that same
  CPU reference; `env_mismatch` remained present, and the evidence contract returned
  `score_claim=false` and `promotion_eligible=false`.
- **Marker control:** the real cu128 export selected `torchvision==0.24.0+cu128` for simulated
  Linux x86_64 and `torchvision==0.24.0` for simulated Linux aarch64.
- **Refusal controls:** no group, uv absent, export nonzero, missing parity package, two surviving
  marker rows, and lock mutation each returned a distinct typed refusal.
- **Venv control:** an absolute venv launcher symlink remains the invoked executable and does not
  collapse to its base interpreter.

Executed results:

```text
.venv/bin/python -m pytest src/tac/tests/test_contest_auth_eval.py -q
69 passed in 1.39s; rc=0

.venv/bin/python -m pytest \
  src/tac/tests/test_contest_auth_eval.py::test_missing_venv_reference_is_derived_from_real_cpu_lock_and_matches \
  src/tac/tests/test_contest_auth_eval.py::test_missing_venv_reference_refuses_real_mismatched_interpreter \
  src/tac/tests/test_contest_auth_eval.py::test_cu128_torchvision_markers_resolve_in_both_linux_architectures -q
3 passed in 0.86s; rc=0

.venv/bin/python -m pytest \
  src/tac/tests/test_modal_auth_eval.py::test_source_uses_literal_cuda_canonical_contest_eval \
  src/tac/tests/test_modal_auth_eval.py::test_modal_auth_eval_images_include_hard_runtime_entropy_deps \
  src/tac/tests/test_modal_auth_eval.py::test_modal_uploaded_submission_dir_runtime_manifest_uses_remote_shape \
  src/tac/tests/test_locked_env_probe.py -q
15 passed in 0.30s; rc=0

.venv/bin/python -m py_compile \
  experiments/contest_auth_eval.py experiments/modal_auth_eval.py experiments/modal_auth_eval_cpu.py \
  src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_modal_auth_eval.py
rc=0

.venv/bin/ruff check --ignore C420 \
  experiments/contest_auth_eval.py experiments/modal_auth_eval.py experiments/modal_auth_eval_cpu.py \
  src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_modal_auth_eval.py
All checks passed; rc=0

git diff --check
rc=0
```

Both required review-tracker passes were recorded for each of the five touched Python files; no
review override was used.

The charter-requested combined suite was also executed. It reached `116 passed, 17 failed`. The
17 failures are bounded outside this patch: one sandbox Unix-socket bind `PermissionError`, live
single-flight claim state affecting Modal dispatcher tests, and the existing task #1005 CPU runtime
hash expectation drift. None reached the newly added group/lock behavior. This arm does not call
that combined suite green.

## Live refusal evidence and authority boundary

The harvested result at
`experiments/results/modal_auth_eval/archive_0f5a797fda84/harvested_result.json` is:

- call: `fc-01KZNZNYECSJP143ZHZ14452RX`
- archive: `0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84`,
  188,636 bytes
- axis: `[contest-CUDA env-mismatch advisory]`, n=600
- refusal: `passed=false`, rc=10, `score_claim=false`, `promotion_eligible=false`
- components observed before refusal: `d_seg=0.00029661`, `d_pose=0.00002332`; recomputed
  advisory value `0.17053685681621078`
- source commit: `0faa617bac54e26312e6d609284bb9e5022a8e0f`

That row cannot move any pointer. Its provenance SHA is
`b8e62717655ab76ae2f426eaff37b2711b47991c77c9b4e5759050172d303ee0`; the harvested result SHA
is `46e0cb2e870d73045994e943e87587a20352ee2035af86b07a1686ca3b256e82`.

The claim ledger still contains active rows for the harvested call. MAIN must append the terminal
disposition before another single-flight dispatch. In addition, the current Modal path retained
the archive and scalar/manifests but did not prove durable custody of the materialized decoded raw
payload. The P0 always-keep-the-payload rule therefore blocks a re-fire until task #1001 or an
equivalent landing makes that raw retention explicit and durable.

## Exact queued re-fire commands

Disposition: **QUEUED, NOT FIRED.** Owner: **MAIN**. Consumer store:
`.omx/state/active_lane_dispatch_claims.md` plus the per-axis result directories below. Fire trigger:
this change lands; the harvested call receives a terminal claim row; no other n600 scorer owns the
slot; and the Modal wrapper durably retains the materialized raw payload with bytes and SHA.

CUDA first:

```bash
.venv/bin/modal run --detach experiments/modal_auth_eval.py \
  --archive /Volumes/VertigoDataTier/pact/ddm_ai1_20260809/temporal_v2/retained/temporal_reversion/archive.zip \
  --expected-archive-sha256 0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84 \
  --inflate-sh inflate.sh \
  --output-dir experiments/results/modal_auth_eval/ai1_ans_temporal_188636_lockderived_gp1_20260810_cuda \
  --gpu T4 --detach --provider-detach-ack \
  --pair-group-id ai1_ans_temporal_188636_lockderived_gp1_20260810 \
  --lane-id lane_ddm_ai1_paired_exact_row_20260810_contest_cuda \
  --instance-job-id ai1_ans_temporal_188636_lockderived_gp1_20260810_cuda \
  --claim-agent MAIN \
  --claim-notes "ddm_gp1 lock-derived parity refire; axis=contest_cuda; archive_sha=0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84; bytes=188636" \
  --submission-dir src/tac/pr130_runtime/dv1_cpu_runtime \
  --expected-runtime-tree-sha256 30a6fb66cb2a32303cbb9b83fc3b882a946889d03453bfa512e58756ac9f006e
```

Then the CPU axis, using the separately resolved `cpu` lock group:

```bash
.venv/bin/modal run --detach experiments/modal_auth_eval_cpu.py \
  --archive /Volumes/VertigoDataTier/pact/ddm_ai1_20260809/temporal_v2/retained/temporal_reversion/archive.zip \
  --expected-archive-sha256 0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84 \
  --inflate-sh inflate.sh \
  --output-dir experiments/results/modal_auth_eval_cpu/ai1_ans_temporal_188636_lockderived_gp1_20260810_cpu \
  --detach --provider-detach-ack \
  --pair-group-id ai1_ans_temporal_188636_lockderived_gp1_20260810 \
  --lane-id lane_ddm_ai1_paired_exact_row_20260810_contest_cpu \
  --instance-job-id ai1_ans_temporal_188636_lockderived_gp1_20260810_cpu \
  --claim-agent MAIN \
  --claim-notes "ddm_gp1 lock-derived parity refire; axis=contest_cpu; archive_sha=0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84; bytes=188636" \
  --submission-dir src/tac/pr130_runtime/dv1_cpu_runtime \
  --expected-runtime-tree-sha256 fc665bb297e853d0230d2bec9ebdb3dcf1c9a8b75421b403447d855fc2cba30b
```

## NEXT_IF_RESUMED

- **QUEUED** — owner **task #1001 / its named retention successor**; consumer store **Modal
  result volume plus its machine-readable retention manifest**; fire trigger **land durable raw
  decoded-payload retention with bytes and SHA for auth-eval runs**.
- **QUEUED** — owner **MAIN**; consumer store **`.omx/state/active_lane_dispatch_claims.md`**; fire
  trigger **append a terminal disposition for `fc-01KZNZNYECSJP143ZHZ14452RX` and confirm no live
  duplicate lane claim**.
- **QUEUED** — owner **MAIN**; consumer store
  **`experiments/results/modal_auth_eval/ai1_ans_temporal_188636_lockderived_gp1_20260810_cuda`**;
  fire trigger **this commit is landed, raw retention is proven, the old claim is terminal, and the
  n600 scorer slot is free; execute the CUDA command above exactly**.
- **QUEUED** — owner **MAIN**; consumer store
  **`experiments/results/modal_auth_eval_cpu/ai1_ans_temporal_188636_lockderived_gp1_20260810_cpu`**;
  fire trigger **the paired CUDA result is harvested and the n600 scorer slot is free; execute the
  CPU command above exactly with the `cpu` group**.

Own-vehicle frontier unchanged: **S = 0.7539807296911207 @ 357,836 B
[macOS-CPU advisory] n600**.
