# L5 v2 Full-Frame Exact-Anchor Custody Hardening - 2026-05-16

## Scope

This ledger records the L5 v2 custody hardening prompted by adversarial review
of the Time-Traveler L5 v2 staircase and Cathedral autopilot rank/promotion
surface.

The failure class:

- exact anchors could pass L5 v2 semantic gates with generic artifact/log paths
  but without an `inflated_outputs_manifest` or raw inflated-output aggregate
  SHA;
- byte-closed temporal side-info proofs could prove parser consumption and a
  changed digest without proving full-frame inflate-output custody;
- CPU device checks used substring matching, so negated text such as `no-cpu`
  could satisfy CPU-axis custody;
- `auth_eval_command` only needed to be non-empty, so non-contest commands could
  satisfy exact-eval evidence rows.

## Code Changes

Touched files:

- `src/tac/exact_eval_custody.py`
- `src/tac/optimization/l5_staircase_v2.py`
- `src/tac/tests/test_exact_eval_custody.py`
- `src/tac/tests/test_l5_staircase_v2.py`

Hardening:

- CPU axis device checks now use the same non-negated token logic as CUDA.
- Exact-eval evidence requires a recognizable contest auth-eval/evaluate
  command shape when `require_auth_eval_command=True`.
- L5 v2 exact anchors require repo-local
  `inflated_outputs_manifest_path` custody and a valid
  `inflated_raw_output_aggregate_sha256`.
- L5 v2 temporal side-info byte-mutation proof requires:
  - repo-local inflated-output manifest;
  - valid raw inflated-output aggregate SHA;
  - canonical `inflate.sh archive_dir output_dir file_list` command signature.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_exact_eval_custody.py \
  src/tac/tests/test_l5_staircase_v2.py -q
```

Result: `42 passed`.

This is a custody/rank-authority hardening patch. It does not claim any score
movement and does not promote L5 v2. L5 v2 remains planning-only until its
byte-closed temporal side-info proof, C1/Z5/TT5L probe, paired CPU/CUDA plan,
and exact/diagnostic anchor evidence satisfy the stricter gates.

