# PR85 Residual Sufficient-Program Curriculum Export - 2026-05-04

## Scope

Converted the local PR85 residual sufficient-program profile into a compact
training/export substrate for learned/native mask coders. This is planning
only: no archive was built, no scorer was loaded, no GPU work was dispatched,
and no score is claimed.

## Artifact

- JSON: `experiments/results/pr85_residual_sufficient_program_20260504_codex/curriculum_density/pr85_residual_sufficient_program_curriculum_density.json`
- JSON SHA-256: `e2cd1ca7b591265b9a77ba897b019259a0203f663ba2a1ba3a735b18393a5fd6`
- NPZ: `experiments/results/pr85_residual_sufficient_program_20260504_codex/curriculum_density/pr85_residual_sufficient_program_curriculum_density.npz`
- NPZ SHA-256: `25e8ab3258e75255e14931d923c4da12cb2524db01e3565eb41331b0c7ce7757`
- Predictor id: `left_zero_border`
- Top-frame records: 32
- Row-span record cap: 256
- Full row-span rows available before cap: 222328

## Custody

- Source profile: `experiments/results/pr85_residual_sufficient_program_20260504_codex/pr85_residual_sufficient_program_profile.json`
- Source profile SHA-256: `a11574e33d1072b7d527a70e9319931be526e7ba46c03e6a34889150432016db`
- Token source: `experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin`
- Token source SHA-256: `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- Render-order SHA-256: `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`

## Command

```bash
.venv/bin/python experiments/export_pr85_residual_sufficient_program_curriculum.py --top-frame-count 32 --max-row-spans 256
```

## Verification

```bash
.venv/bin/python -m py_compile experiments/export_pr85_residual_sufficient_program_curriculum.py src/tac/tests/test_export_pr85_residual_sufficient_program_curriculum.py
.venv/bin/python -m pytest src/tac/tests/test_export_pr85_residual_sufficient_program_curriculum.py -q
```

Focused synthetic pytest result: `3 passed in 0.14s`.
