# PR106 PacketIR matrix custody hardening

Date: 2026-05-16
Owner: Codex
Scope: PR106 PacketIR matrix / exact-eval custody / L5 v2 stack evidence

## Trigger

Read-only adversarial review found that the PR106 PacketIR matrix accepted
self-labelled exact-eval JSON from too few fields. The prior matrix checked
axis text, archive SHA, sample count, and score presence, but did not require
runtime-tree custody, hardware/device semantics, auth-eval command/log custody,
or score-formula closure.

## Fix

`src/tac/packet_compiler/pr106_candidate_matrix.py` now synthesizes each
exact-axis row into the shared `validate_exact_eval_evidence()` contract. The
matrix requires:

- archive SHA match;
- runtime tree SHA;
- runtime content tree SHA;
- contest CPU/CUDA hardware and device semantics;
- adjacent auth-eval command evidence;
- local stdout/stderr log evidence;
- artifact path evidence;
- official score-formula closure from stored component fields.

Runtime-consumption rows now require source archive SHA and runtime source-tree
custody, and still require all authority flags to remain false.

The markdown renderer also emits `status_blockers`, so rows with missing paired
axes no longer display as blocker-free.

## Result

The stricter matrix intentionally downgrades older weak rows:

- `paired_exact_measured`: `3`
- `single_axis_exact_measured_needs_pair`: `9`
- `runtime_consumed_needs_paired_exact_eval`: `4`

Regenerated artifacts:

- JSON SHA-256: `03889d2af21468a752fb031375b040cce00fa78a934e1224c217e1c6f64bdd23`
- Markdown SHA-256: `9fb06495194f1db9fd6aad5c2a059b00eb88f2ece0f8c0ead503654dc621ffd8`

L5 v2 stack evidence was rebound to the new matrix SHA. The extracted TT5L x
PR106 planning candidates are now:

- `format_0x0d_latent_score_table`
- `format_0x0c_exact_radix`
- `prefix_top_16_pr101grammar`

Rows `format_0x01_r2_release` and `format_0x02_pr101_grammar` are no longer
treated as paired exact under this stricter contract because their stored
component fields do not close the score formula at the shared exact-eval
tolerance.

## Verification

```bash
.venv/bin/ruff check src/tac/packet_compiler/pr106_candidate_matrix.py src/tac/tests/test_pr106_packetir_candidate_matrix.py src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py
.venv/bin/python -m pytest src/tac/tests/test_pr106_packetir_candidate_matrix.py src/tac/tests/test_l5_staircase_v2.py -q
.venv/bin/python tools/build_pr106_packetir_candidate_matrix.py
```

Observed:

- `All checks passed!`
- `52 passed`
- regenerated matrix status counts listed above.

## Follow-Up

Rebuild or re-adjudicate the downgraded rows with full component precision and
the shared exact-eval custody fields if they are still high-EV. Do not promote
or rank them as paired exact until the regenerated artifacts satisfy the same
validator.
