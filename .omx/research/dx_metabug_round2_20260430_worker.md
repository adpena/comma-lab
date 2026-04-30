# DX Metabug Round 2 Worker Note - 2026-04-30

Scope: DX/preflight/metabug hardening only. No score claims, lane rankings, or
promotion evidence are recorded here.

## Context Inspected

- `AGENTS.md` sections for Lightning Batch Job artifact paths, MCP shutdown,
  and diagnostic component sensitivity promotion gates.
- `src/tac/preflight.py` Lightning supply-chain and MCP config checks.
- `src/tac/tests/test_lightning_batch_jobs.py` and
  `src/tac/deploy/lightning/batch_jobs.py` for the read-only
  `/teamspace/jobs/.../artifacts` failure class.
- `experiments/profile_component_sensitivity.py`,
  `experiments/build_component_sensitivity_manifest.py`, and focused
  component-sensitivity tests for diagnostic sensitivity promotion guards.
- `.omx/research/dx_self_protecting_harness_hardening_20260430_codex_progress.md`
  and `.omx/research/dx_metabug_greenup_20260430_worker.md`.

## Hardenings Landed

1. MCP config respawn guard tightened.
   - `src/tac/preflight.py` now treats repo `.vscode` MCP/settings files as
     repo-owned MCP config candidates.
   - JSON config files are no longer considered clean immediately after a
     successful parse. They still get line-scanned for known MCP helper command
     tokens such as `chrome-devtools-mcp` and `rbx-studio-mcp`.
   - This closes the gap where an editor/runtime JSON file could keep
     `mcpServers` empty but still contain a helper command that respawns MCP.

2. Diagnostic sensitivity manifest requests now fail before profiling.
   - `experiments/profile_component_sensitivity.py --manifest-output` now
     fails at argument-validation time because current outputs are diagnostic
     Fisher-proxy artifacts with `promotion_eligible=false`.
   - The script docstring/help text now says current outputs cannot assemble a
     promotable `component_sensitivity_v1` until official component-response
     artifacts exist.
   - The direct builder-side diagnostic artifact rejection remains in
     `experiments/build_component_sensitivity_manifest.py`.

## Changed Files

- `src/tac/preflight.py`
- `src/tac/tests/test_preflight_meta_bugs.py`
- `experiments/profile_component_sensitivity.py`
- `src/tac/tests/test_profile_component_sensitivity.py`
- `.omx/research/dx_metabug_round2_20260430_worker.md`

Note: `experiments/profile_component_sensitivity.py` and
`src/tac/tests/test_profile_component_sensitivity.py` were already untracked in
the dirty worktree before this pass; edits were made in place without reverting
or overwriting unrelated work.

## Verification

Passed:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_meta_bugs.py::TestNoActiveMcpServerConfig \
  src/tac/tests/test_profile_component_sensitivity.py::test_proxy_profile_outputs_cannot_assemble_promotable_manifest \
  src/tac/tests/test_profile_component_sensitivity.py::test_manifest_output_cli_rejected_before_profile \
  -q
# 8 passed

.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q
# 20 passed

.venv/bin/python -m pytest src/tac/tests/test_profile_component_sensitivity.py -q
# 13 passed

.venv/bin/python -m py_compile \
  src/tac/preflight.py \
  experiments/profile_component_sensitivity.py \
  src/tac/tests/test_preflight_meta_bugs.py \
  src/tac/tests/test_profile_component_sensitivity.py

.venv/bin/python - <<'PY'
from tac.preflight import check_no_active_mcp_server_config
print(check_no_active_mcp_server_config(strict=True, verbose=True))
PY
# [mcp-config-disabled] OK: 1 repo-owned MCP config file(s) scanned
# []

git diff --check -- \
  src/tac/preflight.py \
  src/tac/tests/test_preflight_meta_bugs.py \
  experiments/profile_component_sensitivity.py \
  src/tac/tests/test_profile_component_sensitivity.py
```

Additional whitespace check for pre-existing/new untracked files:

```bash
git diff --check --no-index -- /dev/null experiments/profile_component_sensitivity.py
git diff --check --no-index -- /dev/null src/tac/tests/test_profile_component_sensitivity.py
git diff --check --no-index -- /dev/null .omx/research/dx_metabug_round2_20260430_worker.md
```

These no-index checks produced no whitespace-error output. Exit code `1` is
expected for no-index comparisons against `/dev/null` because the files differ
from an empty path.

`bash -n` was not applicable; no shell scripts were touched.

## Follow-Up

- Keep default repo preflight scoped to repo-owned MCP configs. Home/editor
  config audits should continue to pass explicit `config_paths` rather than
  making preflight depend on user-specific home files.
- Update older component-sensitivity runbooks that still show
  `profile_component_sensitivity.py --manifest-output` before anyone reuses
  those commands for a CUDA run.
