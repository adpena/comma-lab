# Public frontier intake profiler rehydration - 2026-05-08

Evidence grade: `empirical` for byte-custody tooling. Score claim: `false`.

## What changed

`src/tac/public_frontier_intake.py` is no longer a recovery stub. It now
implements the public-frontier byte-intake contract used by
`experiments/profile_public_frontier_intake.py`:

- strict ZIP local/central name parity and safe member-name checks;
- archive/member byte counts, CRCs, SHA-256s, and compression methods;
- primary payload detection for `x`, `0.bin`, `p`, and renderer-payload names;
- charged side-info accounting;
- PR85-family segment slicing and baseline segment diffing when applicable;
- JSON and Markdown outputs with `score_claim=false`.

`src/tac/pr85_bundle.py` now defaults `pack_pr85_bundle(..., header_mode="v5")`,
matching the existing public-frontier tests and the public PR85-style archive
contract.

## Real archive validation

Generated artifacts:

- `experiments/results/public_frontier_intake_validation_20260508_codex/pr102_public_frontier_intake.json`
- `experiments/results/public_frontier_intake_validation_20260508_codex/pr102_public_frontier_intake.md`
- `experiments/results/public_frontier_intake_validation_20260508_codex/pr108_public_frontier_intake.json`
- `experiments/results/public_frontier_intake_validation_20260508_codex/pr108_public_frontier_intake.md`

PR102 byte custody:

- archive bytes: `178981`
- archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- strict ZIP: `valid=true`
- primary member: `0.bin`, `178873` bytes, stored, SHA-256
  `3234f0689164cfc95b7ee9f9cdf38ecf4d082cfb7048058e2b3ff0f54f864e43`
- charged side-info bytes: `0`

PR108 byte custody:

- archive bytes: `442979`
- archive SHA-256:
  `127b0b318ba2355cdac0d513f4027f0ca3297be4cba0f44e1ddb25cc70586804`
- strict ZIP: `valid=true`
- primary member: `0.mkv`, `442819` compressed bytes, deflated, SHA-256
  `3541f5031914a76d8632e094703ec1f96e59c7fb07942963379fc3d82bbe3035`
- charged side-info bytes: `0`

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_public_frontier_intake.py \
  src/tac/tests/test_archive_byte_profile.py -q

.venv/bin/python -m ruff check \
  src/tac/public_frontier_intake.py \
  src/tac/pr85_bundle.py \
  experiments/profile_public_frontier_intake.py

.venv/bin/python -m py_compile \
  src/tac/public_frontier_intake.py \
  src/tac/pr85_bundle.py \
  experiments/profile_public_frontier_intake.py
```

All three checks passed.

## Constraints

This remains byte-only evidence. It may unblock public replay custody,
archive-layout deconstruction, and exact-replay adapter work, but it cannot
promote, rank, retire, or kill any lane. PR102 still requires exact CUDA replay
of the exact archive and runtime adapter before its CPU/CUDA drift can change
frontier status.
