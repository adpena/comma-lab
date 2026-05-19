# Codex Findings - ITEM4 Inflate Runtime Reviewability Phase A

Date: 2026-05-19T01:23:00Z
Actor: codex
Task: `codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518::ITEM_4`

## Verdict

ITEM4 is advanced but not closed.

The original directive bundled OP-1, OP-2, and OP-5. Current authority status:

- OP-2 / OP-5: already landed before this pass as Catalog #328, with `tac.submission_inflate_loc_budget`, `tools/audit_submission_inflate_py_loc_budget.py`, `check_submission_inflate_py_under_loc_budget`, CLAUDE.md row, and warn-only preflight wiring.
- OP-1: missing as a reusable helper surface; this pass adds the first additive shared-runtime extension helper.
- Strict flip: not ready. Live scan reports 37 direct submission runtimes above the 100-line review target and 14 above the hard 200-line budget.
- Runtime migration: not done. Large `submissions/*/inflate.py` files have not yet been rewritten to consume the helper, and no byte-identical output parity proof exists for those migrations.

## Changes

Added `src/tac/substrates/_shared/inflate_runtime_extensions.py`.

The module provides:

- `iter_file_list_entries(file_list)`
- `inflate_loop_per_video(file_list, archive_dir, output_dir, render_fn)`
- `load_per_substrate_state_dict(archive_dir, state_relpath, expected_sha256=...)`
- `sha256_file(...)` and `require_sha256(...)`
- compatibility aliases `_inflate_loop_per_video` and `_load_per_substrate_state_dict` matching the directive names.

The helper is intentionally additive. It gives oversized runtimes a canonical migration target without changing any submission runtime bytes in this pass.

Added `tools/audit_inflate_py_loc_budget.py` as a compatibility wrapper around the committed Catalog #328 audit CLI, and added explicit `--summary` support to the canonical audit CLI.

Extended `tac.submission_inflate_loc_budget` with:

- 100-line default review target;
- 200-line hard budget;
- directive-aligned waiver tokens:
  - `INFLATE_LOC_DEFAULT_BUDGET_WAIVED:`
  - `INFLATE_LOC_WAIVER:`
  - legacy `INFLATE_PY_LOC_BUDGET_OK:` still accepted for Catalog #328 compatibility;
- JSON fields for `budget_tier`, `severity`, `size_driver_categories`, `technique_applicability`, `shared_runtime_helper_adopted`, `generated_at_utc`, and `git_head`.

## Adversarial Review

Bernoulli reviewed ITEM4 read-only and flagged the key false-authority risks:

- Do not close OP-1 merely because a helper exists.
- Do not strict-flip OP-5 while live violations remain.
- Do not claim LOC cleanup as score movement.
- Do not import `tac.substrates._shared.*` directly from contest submissions unless Catalog #295 empty-PYTHONPATH runtime closure is satisfied through vendoring or an explicit runtime package plan.
- The richer OP-2 audit needed classification metadata and the two waiver tiers; this pass adds those.

## Evidence

Commands:

```bash
.venv/bin/python -m pytest src/tac/tests/test_substrate_inflate_runtime.py src/tac/tests/test_check_328_submission_inflate_py_loc_budget.py -q
.venv/bin/ruff check src/tac/substrates/_shared/inflate_runtime_extensions.py src/tac/submission_inflate_loc_budget.py src/tac/tests/test_substrate_inflate_runtime.py src/tac/tests/test_check_328_submission_inflate_py_loc_budget.py tools/audit_inflate_py_loc_budget.py tools/audit_submission_inflate_py_loc_budget.py
tools/audit_inflate_py_loc_budget.py --json
tools/audit_inflate_py_loc_budget.py --json --max-lines 100
.venv/bin/python -c "from tac.preflight import check_submission_inflate_py_under_loc_budget; rows=check_submission_inflate_py_under_loc_budget(strict=False, verbose=False); print(len(rows))"
.venv/bin/python -m pytest src/tac/tests/test_check_295_submission_inflate_empty_pythonpath.py src/tac/tests/test_inflate_select_device_parity.py src/tac/tests/test_check_328_submission_inflate_py_loc_budget.py src/tac/tests/test_substrate_inflate_runtime.py -q
```

Results:

- `21 passed in 1.51s`
- touched-file ruff: `All checks passed!`
- audit `--json`: 37 findings, 14 hard-budget violations, 23 default-budget warnings
- audit `--json --max-lines 100`: 37 findings, 37 hard-budget violations, 0 default-budget warnings
- direct preflight helper: 37 warn-only findings
- expanded runtime/preflight tests: `147 passed in 1.27s`

Full `preflight_all(verbose=True)` was attempted and failed on an unrelated existing strict violation in `scripts/remote_lane_substrate_tishby_ib_pure.sh` (`check_dispatch_wrapper_stages_implemented`). That is not introduced by this ITEM4 patch and should be handled separately before claiming all-stack preflight green.

## Remaining ITEM4 Work

1. Migrate at least one named over-budget runtime to the shared helper with byte-identical output proof.
2. Repeat until the hard-budget live count is driven to zero or every remaining large runtime carries an explicit source-faithful waiver.
3. Only then consider OP-5 strict flip.

