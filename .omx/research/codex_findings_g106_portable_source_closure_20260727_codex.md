# G106 Codex finding — portable source closure before fresh V9 takeoff

Date: 2026-07-27  
Lane: `lane_g106_v9_g46_provenance_closure_20260727`  
Authority: implementation/custody guard only; no archive, score, or pointer claim  
Competitive target: upstream official `0.172`  
Pointer delta: none

## Premise falsified

The pre-G106 trainer did not stamp the G46 evaluator-source identity into a
fresh checkpoint. It called the full-tree `compute_upstream_snapshot_sha256`
over the host `upstream/` workspace. The current workspace contains
`upstream/.venv/bin/python3` as a symlink, so that canonical full-tree helper
correctly raises and the trainer silently fell back to `"unknown"`.

The G46 receipt's `upstream_closure.closure_sha256` is also not a portable
checkpoint identity: its digest includes each member's absolute `path`. A clean
main checkout at another path therefore has different closure bytes even when
all evaluator sources are identical. Requiring that path-bound digest while
also requiring a globally clean checkout made the fresh producer impossible to
reproduce honestly from the live shared worktree.

This is the same representation-boundary bug as charging proof hashes inside a
candidate packet: custody state was being confused with solution/executed-source
state.

## Landed correction

`tac.upstream_source_closure.v1` binds the exact frozen source members recursively
used by `upstream/evaluate.py`:

- `evaluate.py`
- `frame_utils.py`
- `modules.py`
- `public_test_video_names.txt`

The digest domain contains only the schema and ordered
`{relative_path, bytes, sha256}` rows. Absolute roots and unrelated workspace
entries are excluded. Required members and their path components must be real,
regular, non-symlinked files. Models, source video, and G46 targets remain
separately content-addressed encoder custody.

Current portable source closure:
`e93f6c744fe0025ecc30d1f1cef00617a3f1397b68cadb856817766cfec63279`.

The trainer now persists both of these scalars in deploy and resume NPZs:

- `__cfg_upstream_snapshot_schema = "tac.upstream_source_closure.v1"`
- `__cfg_upstream_snapshot_sha256 = <portable closure SHA-256>`

The old G46 path-bound digest `9c588c725d66...` remains valid only as part of
the immutable external G46 receipt. Candidate admission must reopen that
receipt, strip its four member rows to the portable three-field identity, and
require exact equality with G106. Neither receipt nor hashes belong in counted
candidate bytes.

## Triality and whole-object effect

- DSL: a typed portable source-closure schema and exact checkpoint fields.
- DAG: G46 external custody -> portable member equality -> fresh trainer
  checkpoint -> V9 adapter -> counted Y1 program.
- Equation: `C_src = H(domain || schema || ordered(relative_path, bytes,
  sha256(content)))`, independent of checkout root and unrelated files.

This removes an actual pre-launch impossibility without weakening custody. It
does not lower score by itself. The remaining takeoff gate is a clean-main
source capsule with fresh-init, periodic plus preserved per-stage checkpoints,
then exact V9 adapter/public runtime closure.

## Validation

- `4 passed` in `src/tac/tests/test_upstream_source_closure.py`
- Ruff clean on all new G106 source/tests
- `py_compile` clean on new source/tests and the patched trainer
- live four-member identity exactly equals the G46 member rows after removing
  absolute paths
- relocation and unrelated `.venv` symlink behavior tested

## Artifact identities before this memo

- `src/tac/upstream_source_closure.py`:
  `497e05d93c76d1fd05c02552f6a4699559c6c2ff36b20ac2d1ecb4dd5a518e9a`
- `src/tac/training_source_provenance.py`:
  `5ddee53802e7328461dc745e1d20e17edaa5488514c3ad83815f90336c0c4670`
- `src/tac/tests/test_upstream_source_closure.py`:
  `ea03f54ee3a37e237088b10628545999d1663f27dd1ee2426cd700cbb5fe1347`
- `experiments/train_levelset_witness_realized_through_R_mlx.py`:
  `07a100011c229e9378cdfd1efa0799c8daeefc3fca88ca1a0813d1152c8bccce`
