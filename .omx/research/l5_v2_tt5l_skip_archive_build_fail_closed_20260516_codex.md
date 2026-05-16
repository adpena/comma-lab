# L5 v2 TT5L `--skip-archive-build` Fail-Closed Hardening

Date: 2026-05-16
Author: Codex
Axis: TT5L / L5 v2 trainer hardening
Evidence grade: source-and-test hardening; no score claim

## Finding

`experiments/train_substrate_time_traveler_l5_autonomy.py` exposed
`--skip-archive-build`, but the full trainer ignored the flag and still built a
packet, emitted runtime bytes, and entered the auth-eval gate path.

That is a false-control surface for L5 v2 timing/proof work. A caller could
believe a run was training-only while the trainer still produced byte artifacts,
or future refactors could infer a score path from a run that intentionally had
no byte-closed packet.

## Change

The full TT5L trainer now treats `--skip-archive-build` as an explicit
non-promoting mode:

- stages `archive_build_skipped_by_operator_flag`;
- stages `auth_eval_skipped_archive_build_skipped`;
- leaves archive SHA/byte fields empty or zero;
- records `archive_build_skipped=true`;
- forces `score_claim=false`, `promotion_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`;
- appends dispatch blockers:
  - `archive_build_skipped_no_archive_zip`;
  - `auth_eval_skipped_archive_build_skipped`;
  - `no_byte_closed_packet_for_score_claim`.

## Guard

`src/tac/tests/test_train_time_traveler_full_cpu_mode.py` now directly tests
the provenance hardening helper so skipped-archive runs cannot become
promotion-eligible phantom packets.

Verification:

- `.venv/bin/python -m py_compile experiments/train_substrate_time_traveler_l5_autonomy.py`
- `.venv/bin/python -m pytest src/tac/tests/test_train_time_traveler_full_cpu_mode.py -q`
  - `27 passed`
- `ruff check experiments/train_substrate_time_traveler_l5_autonomy.py src/tac/tests/test_train_time_traveler_full_cpu_mode.py`
  - `All checks passed`

## No Score Claim

This is a control-plane correctness patch. It does not classify TT5L quality,
does not promote a packet, and does not replace the required byte-closed
side-info proof, timing smoke, or exact contest-axis evaluation.
