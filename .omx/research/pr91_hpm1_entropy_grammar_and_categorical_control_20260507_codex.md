# PR91 HPM1 Entropy Grammar Probe And Categorical Control - 2026-05-07

Scope: local CPU-only PR91/HPM1 semantic parity support. No GPU work, no
remote dispatch, no lane claim, no score claim.

## Artifacts

- HPM1 entropy grammar probe:
  `experiments/results/pr91_hpm1_entropy_failure_grammar_probe_20260507_codex/hpm1_entropy_failure_grammar_probe.json`
- Regenerated categorical candidate:
  `experiments/results/categorical_openpilot_payload_candidate_20260506_codex/`
- New label permutation control:
  `experiments/results/categorical_openpilot_payload_candidate_20260506_codex/label_permutation_control.json`

## Finding

The PR91 HPM1 entropy probe reproduces the submitted CPU runtime grammar and
fails at the same local blocker:

- Failure stage: `submitted_tokens_decode`
- Failure reason: `hpac_entropy_decode_contract_mismatch`
- Location: frame `0`, group `10`, symbol-in-group `191`
- Decoded symbols before failure: `5951`
- Token stream words: `29199` little-endian uint32 values

The exact missing grammar is now pinned as the semantic HPAC probability/range
contract that maps the charged HPM1 uint32 queue to all `600x384x512` class
tokens and re-emits the same token bytes. Archive custody, HPM1 section split,
PPMd HPAC model load, dependency versions, and first probability-row
construction are not the current missing pieces.

The categorical candidate now has a real label-permutation fail-closed control:
the builder reverses the charged `class_codebook.json` class order in a
temporary runtime root and verifies `runtime_consumer.py` rejects it with the
expected class-order mismatch. This removes the label permutation blocker from
the current generated readiness artifact without changing decode/reencode
readiness.

## Remaining Blockers

- `no_op_control_not_passed:decode_reencode_identity_control`
- `decode_reencode_parity_not_passed`
- `decode_reencode_full_decode_not_proven`
- `decode_reencode_byte_exact_reencode_not_proven`
- `decode_reencode_independent_proof_artifact_missing`
- `exact_eval_dispatch_requirements_missing`

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py src/tac/tests/test_build_categorical_candidate_payload.py -q
.venv/bin/python -m py_compile src/tac/pr91_hpm1_codec.py src/tac/categorical_payload_candidate.py tools/audit_pr91_hpm1_entropy_failure_grammar_probe.py tools/build_categorical_candidate_payload.py
```

Focused rerun during implementation:

```text
.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py::test_pr91_entropy_failure_grammar_probe_records_exact_failure_row src/tac/tests/test_pr91_hpm1_codec.py::test_pr91_entropy_failure_grammar_probe_cli_records_tool_manifest src/tac/tests/test_build_categorical_candidate_payload.py -q
```

Result: `4 passed` for the focused rerun. Full focused file rerun is recorded
in the turn report.
