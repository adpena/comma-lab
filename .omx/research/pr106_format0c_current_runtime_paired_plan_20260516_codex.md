# PR106 format0C current-runtime paired plan

Date: 2026-05-16
Agent: codex
Status: plan-only, not dispatched
Evidence grade: custody/plan artifact, no score claim

## Scope

This ledger records the current-runtime paired Modal auth-eval plan for the
PR106 PacketIR `format_0x0c_exact_radix` candidate after a dispatcher custody
bug was found and fixed.

Archive:

- Path: `experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip`
- Bytes: `186327`
- SHA-256: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`

Runtime:

- Submission dir: `submissions/pr106_latent_sidecar_r2_pr101_grammar`
- Inflater entrypoint: `inflate.sh`
- Modal uploaded runtime content SHA-256: `8790ec81e5153a8fe3cb250e82b522763ae82b052b48655556be94acb05d5d51`
- Contest-CUDA Modal runtime tree SHA-256: `e596b379c4978ad8ec892d2e7dae8f585941437d698a072cc6e1300fcab56db5`
- Contest-CPU Modal runtime tree SHA-256: `6b336e17784e6c793ce6638f6c5a0eb5beba8fb3d974d1aa286503702dd84b07`

Plan artifact:

- `experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/paired_dispatch_plan_current_runtime_20260516.json`
- Ignored rebuildable artifact; the custody-critical fields are mirrored above.

## Bug Found

`tools/dispatch_modal_paired_auth_eval.py` previously defaulted
`--inflate-sh` to `submissions/robust_current/inflate.sh` even when
`--submission-dir` was supplied. Modal auth-eval wrappers interpret
`--inflate-sh` relative to the uploaded runtime tree when `--submission-dir` is
present. Therefore a plan invoked with only:

```bash
--submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar
```

could generate commands that pointed inside the uploaded PR106 runtime at:

```text
submissions/pr106_latent_sidecar_r2_pr101_grammar/submissions/robust_current/inflate.sh
```

That file does not exist and would fail before exact-eval custody. The failure
class is a config/runtime-custody bug, not a PR106 method result.

## Fix

`tools/dispatch_modal_paired_auth_eval.py` now uses a context-aware CLI default:

- with `--submission-dir`, omitted `--inflate-sh` resolves to `inflate.sh`;
- without `--submission-dir`, omitted `--inflate-sh` resolves to
  `submissions/robust_current/inflate.sh`.

Regression coverage:

- `src/tac/tests/test_dispatch_modal_paired_auth_eval.py::test_cli_defaults_inflate_sh_to_uploaded_submission_runtime_root`

The same verification pass exposed a direct-call/local-entrypoint DX ambiguity:
both Modal auth-eval entrypoints accepted `expected_archive_sha256` as the
second positional parameter, while the tests and operator-facing direct-call
convention use the second positional parameter as `output_dir`. Named CLI flags
were unaffected, but the positional API was fragile. The CUDA and CPU
entrypoints now use `archive, output_dir, expected_archive_sha256` ordering,
while generated Modal commands still pass all custody-critical values by name.

Adversarial review found three additional custody/reporting issues and all are
now patched:

- Anchor reuse now refuses to skip an axis when that axis lacks a nonempty
  expected runtime tree SHA-256. Empty runtime expectation no longer degrades
  to archive-only anchor reuse.
- Paired plan generation now rejects absolute `--inflate-sh` paths outside
  `--submission-dir` before plan emission, matching the Modal wrappers'
  fail-closed runtime-tree contract.
- Modal CPU recovery summaries now report a valid `contest-CPU` canonical
  artifact as `score_claim=true` while preserving `promotion_eligible=false`.

Verification:

```bash
.venv/bin/python -m ruff check tools/dispatch_modal_paired_auth_eval.py src/tac/tests/test_dispatch_modal_paired_auth_eval.py
PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_dispatch_modal_paired_auth_eval.py -q
PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_paired_dispatch_anchor_lookup.py src/tac/tests/test_dispatch_modal_paired_auth_eval.py -q
PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_paired_dispatch_anchor_lookup.py src/tac/tests/test_dispatch_modal_paired_auth_eval.py src/tac/tests/test_modal_auth_eval_recovery.py -q
```

Result: `ruff` clean; focused dispatcher tests `9 passed`; Modal auth-eval
paired-dispatch suite `77 passed`; post-review Modal auth-eval/recovery suite
`89 passed`.

## Plan Regeneration

Command used:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip \
  --expected-archive-sha256 56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7 \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --label pr106_format0c_current_runtime_rerun \
  --lane-id-base pr106_packetir_format_0x0c_current_runtime \
  --pair-group-id pair_pr106_packetir_format_0x0c_current_runtime_56cdd10bdc43 \
  --expected-runtime-tree-sha256 auto \
  --json-out experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/paired_dispatch_plan_current_runtime_20260516.json
```

Plan result:

- `score_claim=false`
- `promotion_eligible=false`
- `axes_skipped_due_to_existing_anchor.contest_cuda=false`
- `axes_skipped_due_to_existing_anchor.contest_cpu=false`
- both generated commands carry `--inflate-sh inflate.sh`
- both generated commands carry the same archive SHA and pair group

## Dispatch Status

No PR106 paired dispatch was executed in this turn. The active Modal ledger still
contains pending training call `fc-01KRRBHZZR7RWBVH4PN65VX1M1` for
`lane_nscs06_carmack_hotz_strip_everything_20260515`; `tools/harvest_modal_calls.py
--from-ledger --get-timeout-seconds 1` reports one unharvested call, and
`tools/modal_function_status.py fc-01KRRBHZZR7RWBVH4PN65VX1M1 --get-timeout-s 5`
reports `result_state=pending`.

Next eligible action after active-lane coordination: execute the regenerated
paired PR106 plan or explicitly record a newer higher-EV dispatch decision.
