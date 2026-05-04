# PR85 QMA9 Native Grammar Worker - 2026-05-04

## Scope

Local-only QMA9-native grammar/table/run reduction screen for PR85. No scorer,
GPU eval, remote dispatch, runtime edit, or score claim was performed.

## Anchor

- Source archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- Source bytes: `236328`
- Source SHA-256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- PR85 QMA9 mask segment bytes: `159011`
- PR85 QMA9 mask segment SHA-256: `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
- Expected decoded token SHA-256: `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`

## Local Smoke

Command:

```bash
.venv/bin/python experiments/build_pr85_qma9_native_grammar_candidates.py
```

Artifact:

- `experiments/results/pr85_qma9_native_grammar_candidates_20260504/candidate_summary.json`

Result:

- `candidate_count`: `0`
- `best_byte_delta`: `null`
- `safe_for_remote_dispatch`: `false`
- `score_claim`: `false`
- `dispatch_performed`: `false`

## Fail-Closed Proof

The existing runtime-supported QMA9 grammar accepts only `QMA9` with a 20-byte
header and a declared arithmetic bitstream. Local planning magics such as
`QMB1`, `QMF1`, and `QMH1`, external context tables, or alternate adaptive
model initialization tables are not supported without runtime edits, so this
worker did not build archives for those formats.

The finite runtime-supported screens found:

- No bytes after the declared QMA9 bitstream: declared packed bytes equal the
  charged mask segment bytes (`159011`).
- No trailing zero bytes inside the declared QMA9 bitstream.
- Full local runtime decode parity against the token source passed:
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`.

Exact blockers:

- `source_qma9_segment_has_no_bytes_after_declared_bitstream`
- `source_qma9_declared_bitstream_has_no_trailing_zero_bytes`
- `no_byte_positive_runtime_supported_qma9_native_grammar_candidate`
- `alternate_qma9_grammar_magics_require_runtime_edits_and_are_planning_only`

## Tests

Commands:

```bash
.venv/bin/python -m py_compile experiments/build_pr85_qma9_native_grammar_candidates.py src/tac/tests/test_build_pr85_qma9_native_grammar_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr85_qma9_native_grammar_candidates.py -q
```

Focused pytest result: `3 passed`.

## Codex Continuation - Top Matrix Blocker Resolution

Command:

```bash
.venv/bin/python experiments/build_pr85_qma9_native_grammar_candidates.py \
  --out-dir experiments/results/pr85_qma9_native_grammar_candidates_20260504_codex \
  --max-prefix-trim-bytes 16
```

Artifact:

- `experiments/results/pr85_qma9_native_grammar_candidates_20260504_codex/candidate_summary.json`
- Artifact SHA-256:
  `7f1ec6ba1a32ef693d318df8656729080ffd515e30e23d807d7cc3ef412d4182`

Result:

- `family_id`: `qma9_native_run_grammar_or_table_reduction`
- `family_resolution.status`:
  `fail_closed_no_byte_positive_runtime_supported_or_screened_run_table_candidate`
- `candidate_count`: `0`
- `best_byte_delta`: `null`
- `score_claim`: `false`
- `dispatch_performed`: `false`
- `safe_for_remote_dispatch`: `false`

Additional local screen added:

- Decode-proven nonzero suffix trims for `1..16` removed QMA9 bitstream bytes.
- All 16 candidates decoded under `submissions/robust_current/range_mask_codec.cpp`
  to a token SHA different from the PR85 token source.
- The source QMA9 segment remains tight: `159011` charged bytes, no bytes after
  declared payload, and no trailing zero bytes inside the declared bitstream.

Related full-stream screens pulled into the summary:

- `qrg1_row_run_grammar`: `no_byte_positive_qrg1_row_run_candidate`
- `alternate_table_grammar`: `no_byte_positive_alt_table_candidate`
- `cpp_mode_sweep`: `no_exposed_cpp_mode_beats_adaptive9bin`
- `macro_prior_screen`: `no_macro_prior_payload_beats_source_qma9`

Minimal missing implementation before this blocker can reopen:

- A full-stream QMA9-compatible encoder that emits fewer than `159011` charged
  mask bytes while the submitted runtime reproduces token SHA
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`.
- Or a new charged runtime mask grammar with a distinct magic or mode header,
  robust-current decoder support, raw-token SHA parity, runtime output parity,
  deterministic single-member archive closure, and only then a future Level-2
  dispatch claim before exact CUDA auth eval.

Verification:

```bash
.venv/bin/python -m py_compile experiments/build_pr85_qma9_native_grammar_candidates.py src/tac/tests/test_build_pr85_qma9_native_grammar_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr85_qma9_native_grammar_candidates.py -q
.venv/bin/python -m pytest src/tac/tests/test_build_pr85_qma9_native_grammar_candidates.py src/tac/tests/test_build_pr85_qma9_run_grammar_candidates.py src/tac/tests/test_build_pr85_qma9_alt_grammar_candidates.py src/tac/tests/test_qma9_range_mask_contract.py -q
.venv/bin/python experiments/build_pr85_qma9_native_grammar_candidates.py --help
```

Results:

- Focused native pytest: `3 passed`.
- Neighboring QMA9 pytest bundle: `24 passed`.
- CLI help includes the new `--max-prefix-trim-bytes`,
  `--run-grammar-summary`, `--alt-grammar-summary`,
  `--mode-sweep-summary`, and `--macro-prior-dir` flags.
