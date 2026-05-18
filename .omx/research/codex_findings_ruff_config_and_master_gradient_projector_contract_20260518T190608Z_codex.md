# Codex Findings - Ruff Config And Master-Gradient Projector Contract

Date: 2026-05-18 19:06:08 UTC
Author: Codex

## Findings

1. Ruff did not need broader excludes. The blocking F821 command is green, and
   generated custody paths are already outside the lint target. The durable
   fix is config-level `force-exclude = true`, so explicit-path Ruff calls
   inherit the same generated-artifact discipline as CI and the git hook.

2. Full-project Ruff remains a large existing baseline, not a safe blocking
   preflight. The full rule set should stay non-blocking until the baseline is
   intentionally burned down; the critical gate remains F821.

3. ITEM_3's packed/length-prefixed master-gradient boundary is now explicit:
   archive grammar detection serializes a `projection_contract` with
   `anchor_emission_allowed=false`, `score_claim_allowed=false`, and the exact
   projector name required before a master-gradient anchor may be emitted.

## Patch

- `pyproject.toml`: set `tool.ruff.force-exclude = true`.
- `src/tac/tests/test_ci_ruff_scope.py`: regression for config-level
  force-exclude.
- `tools/extract_master_gradient.py`: added `ArchiveProjectionContract`,
  grammar-to-projector authority mapping, serialized `projection_contract`,
  and `--layout-contract-output`.
- `src/tac/tests/test_extract_master_gradient.py`: regression coverage for
  fail-closed projection contracts and pre-exit layout manifest writes.

## Real-Fixture Xray

- PR106 format0d: `grammar_name=pr106_format0d`,
  `required_projector=pr106_format0d_primary_payload_projector`,
  `anchor_emission_allowed=false`.
- Public PR106 packed: `grammar_name=pr106_ff_packed_hnerv`,
  `required_projector=pr106_packed_brotli_schema_projector`,
  `anchor_emission_allowed=false`.
- PR100 true HNeRV LC v2: `grammar_name=hnerv_lc_v2_length_prefixed`,
  `required_projector=hnerv_lc_v2_schema_projector`,
  `anchor_emission_allowed=false`.
- PR107 Apogee: `grammar_name=pr107_apogee_length_prefixed`,
  `required_projector=pr107_apogee_schema_projector`,
  `anchor_emission_allowed=false`.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_extract_master_gradient.py src/tac/tests/test_ci_ruff_scope.py]`
  - Result: `37 passed in 0.65s`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py src/tac/tests/test_ci_ruff_scope.py]`
  - Result: `All checks passed!`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --select F821 src/ experiments/ submissions/robust_current/ scripts/ tools/]`
  - Result: `All checks passed!`
- `[empirical:explicit generated-path Ruff probe under experiments/results/]`
  - Result: Ruff reported no Python files under the explicit ignored path,
    proving config-level `force-exclude = true` is active.

## Residual

This closes the false-authority half of ITEM_3 for unsupported packed and
length-prefixed archives. It does not implement the named projectors; those are
now explicit downstream work items rather than ambiguous extractor behavior.
