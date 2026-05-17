# L5 v2 Euler Findings Disposition

Date: 2026-05-17
Verifier: Codex
Repo state verified: `main` at `d6c49f9fbf4b4654b89260b15a4eaeed52b76c5b`
Scope: close the four preserved Euler readiness/adversarial findings from
`.omx/research/l5_v2_subagent_findings_no_signal_loss_20260517_codex.md`
against the actual current code and tests. This is a no-signal-loss
disposition artifact, not a score claim.

## Summary

All four preserved Euler findings are verified addressed on current `main`.
No new score, promotion, dispatch, or rank/kill authority is claimed here.

The important current-state distinction is:

- materialized TT5L paired work units may surface an operator review command
  only when their archive/runtime/pair-axis custody validates and no active or
  invalid provider-blocker artifact is present;
- Lightning paired-axis dry-run structure is separate from current
  source-custody readiness;
- `exact_anchor_or_diagnostic_pair` consumes the pair-level anchor artifact
  when present, so stale per-axis status cannot keep the gate missing;
- paired work-unit and gate evidence both require CPU/CUDA runtime-content
  equality even though runtime tree SHAs may differ by axis.

## Finding 1: invalid Modal provider-blocker fallthrough

Disposition: fixed and tested.

Evidence:

- `src/tac/optimization/l5_staircase_v2.py` computes
  `suppress_execute_template` when a provider-blocker artifact exists but is
  invalid, not only when it is active.
- The TT5L next-action router now emits
  `refresh_or_retire_l5_v2_tt5l_modal_provider_blocker` with
  `ready_for_operator_dispatch=false`, `ready_for_provider_dispatch=false`,
  and no execute command when the blocker artifact is stale or invalid.
- Regression test:
  `test_l5_v2_tt5l_stale_modal_blocker_blocks_execute_command`.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_stale_modal_blocker_blocks_execute_command \
  -q
```

Result: passed as part of the targeted four-test Euler disposition run.

## Finding 2: Lightning dry-run readiness vs stale source custody

Disposition: fixed and tested.

Evidence:

- `src/tac/optimization/l5_staircase_v2.py` splits
  `all_cells_dry_run_structurally_valid` from `all_cells_dry_run_ready`.
  The latter is true only when source custody is current for execution.
- `source_relevant_diff_paths` is surfaced and
  `source_custody_current_for_execution=false` when relevant source paths
  changed after the dry-run plan source commit.
- The architecture-lock packet currently records structural dry-run validity
  separately from source-custody execution readiness.
- Regression tests:
  `test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan` and
  `test_l5_v2_tt5l_lightning_paired_axis_plan_status_blocks_relevant_source_drift`.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_blocks_relevant_source_drift \
  -q
```

Result: passed across the targeted readiness/disposition runs.

## Finding 3: stale per-axis state vs pair-level anchor

Disposition: fixed and tested.

Evidence:

- `l5_v2_canonical_anchor_pair_gate_evidence()` discovers
  `.omx/research/l5_v2_tt5l_paired_exact_anchor_pair_20260516_codex.json`
  and injects it as canonical `exact_anchor_or_diagnostic_pair` evidence when
  explicit gate evidence is absent.
- `_gate_semantic_blockers()` applies paired-anchor semantic checks to that
  pair-level artifact, including paired axis identity, archive identity,
  runtime-content identity, exact-eval custody, and diagnostic/exact anchor
  semantics.
- The tracked pair anchor has `paired_axis_status=paired_cpu_cuda_harvested`,
  both `contest_cpu` and `contest_cuda` rows, and non-promotional exact
  custody for both axes.
- Regression test:
  `test_l5_v2_dispatch_readiness_consumes_pair_anchor_artifact_over_stale_axis_rows`.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_consumes_pair_anchor_artifact_over_stale_axis_rows \
  -q
```

Result: passed as part of the targeted four-test Euler disposition run.

## Finding 4: materialized paired work-unit runtime-content mismatch

Disposition: fixed and tested.

Evidence:

- `_tt5l_materialized_paired_work_unit_status()` rejects divergent
  `expected_runtime_content_tree_sha256_by_axis` values with
  `l5_v2_tt5l_materialized_paired_work_unit_runtime_content_axis_mismatch`.
- Gate-level paired-axis evidence independently rejects CPU/CUDA
  `runtime_content_tree_sha256` divergence while still allowing axis-specific
  runtime tree SHAs.
- Regression tests:
  `test_l5_v2_tt5l_materialized_work_unit_rejects_runtime_content_axis_mismatch`,
  `test_l5_v2_dispatch_readiness_allows_axis_specific_runtime_trees`, and
  `test_l5_v2_dispatch_readiness_rejects_runtime_content_tree_mismatch`.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_runtime_content_axis_mismatch \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_allows_axis_specific_runtime_trees \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_rejects_runtime_content_tree_mismatch \
  -q
```

Result: passed across the targeted readiness/disposition runs.

## Targeted Verification Receipt

Commands run:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_stale_modal_blocker_blocks_execute_command \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_runtime_content_axis_mismatch \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_blocks_relevant_source_drift \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_consumes_pair_anchor_artifact_over_stale_axis_rows \
  -q

.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_allows_axis_specific_runtime_trees \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_rejects_runtime_content_tree_mismatch \
  -q
```

Observed results:

- `4 passed`
- `3 passed`

## Fresh Architecture-Lock Packet Receipt

After the disposition checks, the architecture-lock packet was regenerated on
current `main`:

```bash
.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py \
  --repo-root . \
  --output-json .omx/research/l5_v2_architecture_lock_packet_20260516_codex.json \
  --output-md .omx/research/l5_v2_architecture_lock_packet_20260516_codex.md
```

Observed result:

```text
[l5-v2-architecture-lock] architecture_lock_allowed=false blockers=['requires_all_l5_v2_gate_evidence_valid', 'requires_c1_z5_tt5l_probe_gate_evidence', 'requires_paired_cpu_cuda_sideinfo_effect_curve'] score_claim=false
```

Packet delta from the refresh:

- `current_head_commit` moved to
  `d6c49f9fbf4b4654b89260b15a4eaeed52b76c5b`.
- Lightning source-custody drift now includes
  `src/tac/optimization/l5_v2_measurement_schedule.py` in
  `source_relevant_diff_paths`, so the stale dry-run plan is explicitly not
  current for execution.
- The packet still refuses architecture lock and preserves
  `score_claim=false`.

## Next Concrete L5-v2 Action

Do not repeat these four Euler findings as open blockers. The next material
TT5L/L5-v2 action is to refresh the stale Lightning paired-axis dry-run plan
against the current source tree, or to keep the materialized TT5L work unit on
the provider-blocker path until Modal capacity or an alternate provider path is
actually executable. If an operator command is surfaced later, the required
next evidence is a claimed paired CPU/CUDA dispatch or a fail-closed
provider-capacity artifact, not another meta-review.
