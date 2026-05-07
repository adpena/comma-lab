# PR91/HPM1 phase-major prefix re-encode blocker - 2026-05-07

Scope: local-only PR91/HPM1 grammar recovery blocker classification. No GPU
dispatch, no lane claim, and no score claim.

## Artifact

Command:

```text
.venv/bin/python tools/audit_pr91_hpm1_phase_major_prefix_reencode_blocker.py
  --archive experiments/results/public_pr91_intake_20260504_codex/archive.zip
  --reference-tokens experiments/results/pr85_qma9_mode_sweep_20260504_codex/adaptive6pr.decoded.raw
  --json-out experiments/results/pr91_hpm1_phase_major_prefix_reencode_blocker_20260507_codex/phase_major_prefix_reencode_blocker.json
```

Output:

- Path:
  `experiments/results/pr91_hpm1_phase_major_prefix_reencode_blocker_20260507_codex/phase_major_prefix_reencode_blocker.json`
- Bytes: `37737`
- SHA-256:
  `7b36f8f49e150a60b33074078e9ab2ed117ce4c540365dc4439533bcb3c7c871`

## Classification

- Status: `blocked_phase_major_reference_prefix_not_byte_exact_reencode`
- Blocker class:
  `phase_major_pr85_qma9_reference_symbols_do_not_reproduce_submitted_hpm1_stream`
- PR91 archive bytes: `222404`
- PR91 archive SHA-256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- Reference token bytes: `117964800`
- Reference token SHA-256:
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`

The phase-major target failure is reproduced at:

```text
frame=0, group=17, symbol_in_group=437,
decoded_symbol_count_before_failure=15989
```

The local reference re-emit diverges immediately:

- First local re-emit mismatch: `first_1_symbols`,
  `common_prefix_word_count=0`
- First submitted/reference prefix mismatch: checkpoint `first_8_symbols`,
  symbol index `7`, decoded `2`, reference `0`

## Consequence

This narrows the HPM1 blocker. The PR85/QMA9 phase-major semantic-symbol bridge
advances the failure row, but it is not a byte-exact explanation of PR91's
submitted HPM1 stream. Categorical/OpenPilot/HPM1 remains fail-closed until a
true PR91 encoder semantic-symbol/probability-row source or byte-exact range
re-emit proof exists.

The next implementation target is therefore the PR91-specific HPAC probability
context and range uint32 re-emission contract, not another archive repack.
