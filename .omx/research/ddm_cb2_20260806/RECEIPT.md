# ddm_cb2 Receipt — CI-blind debt #983

## Verdict

Status: **PATCHED, scorer-free; host Metal confirmation queued**.

Measured this turn:

- Named refusal-telemetry test is green in this checkout:
  `.venv/bin/python -m pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_execute_refuses_bad_archive_section_telemetry_before_training -q`
  -> 5 passed. The only extra output was the known no-Metal MLX atexit warning.
- Root cause of the mx2 hook blast radius was source-visible: staging
  `src/tac/pr130_lift/pose/__init__.py` emitted the bare token `pose`, and the
  CI-blind hook selected 29 unrelated MLX-gated targets that merely used ordinary
  pose terminology.
- Patched `tools/preflight_hook.py` so nested package `__init__.py` paths keep
  package suffix tokens such as `tac.pr130_lift.pose` and `pr130_lift.pose`, but
  drop the bare nested leaf token.
- Added regression tests in `src/tac/tests/test_preflight_hook.py`.
- Added MAIN-executable repro wrapper:
  `tools/repro_cb2_pr130_lift_pose_ci_blind_order.py`.

Not measured:

- No n600 scorer run.
- No `upstream/evaluate.py`.
- No Metal-host SIGBUS reproduction; this sandbox has no Metal device. The local
  pre-patch hook step failed earlier at MLX collection with no-Metal errors before
  reaching the host-reported SIGBUS path.

Score/frontier claim: **none**. Own-vehicle frontier remains
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains
borrowed/unmoved.

## Fix

The changed selector behavior:

```text
src/tac/pr130_lift/pose/__init__.py
before: tac.pr130_lift.pose, pr130_lift.pose, pose
after:  tac.pr130_lift.pose, pr130_lift.pose
```

Observed selection after the patch:

```text
.venv/bin/python tools/repro_cb2_pr130_lift_pose_ci_blind_order.py
current_count=0
legacy_count=29
```

Observed mx2-style staged-file selection after the patch:

```text
0
```

This is the expected hook result for a `pr130_lift/pose` package touch that has no
current MLX-gated test importer under `src/tac/tests`: run nothing rather than
running unrelated pose-domain modules.

## Telemetry Leg

The charter named
`test_hinerv_execute_refuses_bad_archive_section_telemetry_before_training`
with the `decoder_missing` parameter as a pre-existing failure. In this checkout,
that premise is stale:

- the parametrized test ran all 5 cases and passed;
- the `decoder_missing` fixture still expects
  `hi_nerv_archive_section_telemetry_decoder_state_missing`;
- the production validator still emits that blocker when no `decoder_state`
  section is present.

No xfail, skip, or expectation weakening was added.

## SIGBUS / Adapter Analysis

Static source read of
`src/tac/substrates/_shared/mlx_score_aware/adapter.py` at
`_score_aware_loss_part_metrics`:

- the helper computes `score_aware_loss(...)`, then for each part executes
  `mx.eval(value)` and immediately converts `value.item()` to a Python float;
- it stores only Python floats in `out`, not live MLX buffers;
- it does not retain references to `parts` outside the call;
- similar scalarization loops exist in sibling helpers, so changing only this
  line would be a point-fix without a source proof.

Source-side conclusion: I did not find a clear buffer-lifetime bug in this helper
from static inspection. The clearer defect was the hook selector creating a large
unrelated multi-module MLX process from the bare `pose` token. The host-reported
SIGBUS remains a Metal-host confirmation item, preserved by the repro script.

## Verification

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_hook.py::test_module_reference_tokens_package_init_resolves_to_the_package \
  src/tac/tests/test_preflight_hook.py::test_nested_package_init_drops_generic_leaf_token \
  src/tac/tests/test_preflight_hook.py::test_package_init_does_not_select_unrelated_mlx_modules \
  src/tac/tests/test_preflight_hook.py::test_nested_pose_package_init_does_not_select_pose_word_mlx_modules -q
-> 4 passed

.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_hook.py::test_nested_package_init_drops_generic_leaf_token \
  src/tac/tests/test_preflight_hook.py::test_nested_pose_package_init_does_not_select_pose_word_mlx_modules -q
-> 2 passed

.venv/bin/python -m pytest \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_execute_refuses_bad_archive_section_telemetry_before_training -q
-> 5 passed

.venv/bin/python -m py_compile \
  tools/preflight_hook.py \
  tools/repro_cb2_pr130_lift_pose_ci_blind_order.py \
  src/tac/tests/test_preflight_hook.py
-> pass

.venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa \
  tools/preflight_hook.py \
  tools/repro_cb2_pr130_lift_pose_ci_blind_order.py \
  src/tac/tests/test_preflight_hook.py
-> All checks passed

git diff --check -- \
  tools/preflight_hook.py \
  src/tac/tests/test_preflight_hook.py \
  tools/repro_cb2_pr130_lift_pose_ci_blind_order.py
-> pass
```

## Recall Evidence

| scope/query | finding | plan change |
|---|---|---|
| cb2 charter + common contract | Task #983 is scorer-free; requires refusal-telemetry adjudication, SIGBUS repro script/static analysis, serializer discipline, and receipts. | No scorer/eval dispatch; receipt and next file written. |
| `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Current own-vehicle pointer is tq1c; `ddm_mx2` active namespace is `src/tac/pr130_lift/pose/`; contest pointer is borrowed. | Kept score claims out and scoped to hook/test apparatus. |
| `.omx/research/ddm_mx2_20260806/RECEIPT.md` and `arm_final_messages/ddm_mx2_20260806T195845Z.md` | mx2 serializer was blocked by CI-blind MLX hook after staging `src/tac/pr130_lift/pose/` files; no skip flag used; index clean. | Reproduced selection source and patched the over-selection. |
| `.omx/research/ddm_hy1_20260805/HY1_RECEIPT.md` | HY1 pattern permits stale-expectation repair or named xfail only for real current failures; no silent skips. | Did not xfail the now-green telemetry test. |
| `tools/list_canonical_equations.py --json` filtered for `ci`, `preflight`, `mlx`, `sigbus`, `pr130_lift` | Many MLX/advisory equations, no cb2-specific canonical equation requiring update. | No equation registry edit. |
| bounded grep over the charter directory, `.omx/state/main_hot_state.md`, `ddm_mx2`, `ddm_hy1`, `tools/preflight_hook.py`, and `test_preflight_hook.py` for `#983`, `SIGBUS`, `pr130_lift/pose`, `CI-blind` | Found the charter, mx2 block, and the live selector surface; no additional task-specific receipt beyond mx2/hy1 in that scope. | Patch remained in hook selector, not adapter internals. |

## Follow-ons

- Telemetry refusal leg: **FOLDED** as stale-current-state after measured pass.
- SIGBUS host confirmation: **QUEUED-WITH-A-FIRE-ORDER** in `NEXT_IF_RESUMED.md`.
- Hook over-selection: **FIRED** by this patch and tests.
