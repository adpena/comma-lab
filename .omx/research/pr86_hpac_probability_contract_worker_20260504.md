# PR86 HPAC Probability Contract Worker - 2026-05-04

Scope: PR86 HPAC probability-model contract recovery only. Local CPU replay,
no GPU work, no remote dispatch, and no score claim.

## Inputs

- Archive:
  `experiments/results/public_pr86_intake_20260504_codex/archive.zip`
- Archive bytes: `207579`
- Archive SHA-256:
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`
- Submitted `tokens.bin` SHA-256:
  `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`
- Merged PR86 source head:
  `0eabe354f09b7490fd1cbb2b05a9102ab528d4d4`

## Implementation

Added explicit HPAC probability variants to the local replay shim:

- `source_float64_perfect_false`: merged-source contract, dispatchable only if
  full decode and byte-exact re-encode pass.
- `source_float32_perfect_false`: off-contract probability dtype probe.
- `source_float64_perfect_true`: off-contract constriction perfect-mode probe.
- `source_float32_perfect_true`: off-contract combined probe.

The replay CLI can now select a single variant with `--probability-variant` or
run the full matrix with `--all-probability-variants`. The worker artifact path
is:

```text
experiments/results/pr86_hpac_probability_contract_20260504_worker/pr86_hpac_probability_contract_variants.json
```

Artifact SHA-256:

```text
4358acd53222da2ff94fccefab0066ae22db6dc736ef3d7ae66eacedc948ea23
```

The fixed worker artifact is deterministic: volatile `recorded_at_utc` and
`elapsed_sec` fields are omitted from the artifact emitted by
`--worker-probability-contract-json-out`.

## Full Submitted-Token Variant Matrix

Command:

```text
.venv/bin/python experiments/replay_pr86_hpac_tokens.py --all-probability-variants --worker-probability-contract-json-out
```

Result: `failed_closed`.

| Variant | Status | First failure | Decoded symbols before failure | Byte-exact re-encode |
| --- | --- | --- | ---: | --- |
| `source_float64_perfect_false` | `failed_closed` | frame 0 / group 10 / symbol 191 | 5951 | not reached |
| `source_float32_perfect_false` | `failed_closed` | frame 0 / group 24 / symbol 561 | 30513 | not reached |
| `source_float64_perfect_true` | `failed_closed` | frame 0 / group 15 / symbol 1534 | 13822 | not reached |
| `source_float32_perfect_true` | `failed_closed` | frame 0 / group 15 / symbol 191 | 12479 | not reached |

None of the variants decoded the full submitted token stream. Therefore none
could attempt byte-exact re-encode against submitted `tokens.bin`.

## Dispatch State

`dispatch_unlocked=false`.

PR86 HPAC probability-contract transfer remains blocked. `float32` input and
`perfect=True` are longer-prefix diagnostics, not replay solutions, and remain
off-contract unless a separate future artifact proves full decode plus
byte-exact `tokens.bin` parity.

## Verification

```text
.venv/bin/python -m py_compile src/tac/pr86_hpac_codec.py experiments/replay_pr86_hpac_tokens.py
.venv/bin/python -m pytest src/tac/tests/test_pr86_hpac_codec.py -q
.venv/bin/python -m pytest src/tac/tests/test_diagnose_pr86_hpac_parity.py -q
```

Pytest results:

- PR86 HPAC codec tests: `12 passed, 1 warning`.
- PR86 parity diagnostic tests: `6 passed, 1 warning`.
